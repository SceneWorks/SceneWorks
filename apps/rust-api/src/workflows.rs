//! `POST /api/v1/workflows/inspect` — read a shared image's embedded workflow and say whether this
//! machine can run it (sc-15950, epic 15945).
//!
//! Read-only by construction. It creates **no asset, no job and no project mutation**: the upload
//! is streamed to the same swept `cache/uploads` staging area asset import uses, the PNG's text
//! chunk is read out of it, and the temp file is removed on every path. That matters because
//! sc-15951 prefills the studio from a dropped file the user may never import — asking them to
//! commit an asset in order to find out what a file contains would be exactly backwards.
//!
//! # Three outcomes, and only one of them is an error
//!
//! * The file carries an envelope → `200 { status: "workflow", workflow, resolution }`.
//! * The file is a PNG with no `sceneworks:workflow` chunk → `200 { status: "no_workflow",
//!   workflow: null, resolution: null, detail }`. **This is the common case** (any foreign PNG) and
//!   it must not look like a failure; sc-15951 branches on `status`.
//! * The file is not a PNG at all, or claims a workflow this build refuses to guess at → a typed
//!   4xx carrying a machine-readable `code` and a sentence fit to show the user.
//!
//! # Every failure carries a `code`
//!
//! Including the ones that never reach the handler body: a malformed multipart request (an empty
//! body, a JSON body, a `Content-Type` with no `boundary`) is rejected by the extractor before the
//! handler runs, and axum's own rejection is a PLAIN-TEXT body with no `code` at all. So the
//! handler takes `Result<Multipart, MultipartRejection>` and maps it itself
//! ([`INSPECT_CODE_BAD_MULTIPART`]) rather than letting sc-15951 parse a body that is not JSON.
//!
//! One failure is deliberately a 5xx: [`WorkflowChunkError::Io`] fires when OUR OWN staged temp
//! cannot be opened or read — the caller's bytes were written successfully by then, so an AV
//! sharing violation on `cache/uploads` or a disk fault is a server fault. Reporting it as "this
//! file is not a PNG" would be exactly the mislabeling this epic exists to stop.
//!
//! The resolution report itself is `sceneworks_core::workflow_resolution` — shared with the import
//! path (sc-15949) and the web (sc-15951 / sc-15952) rather than owned here. This module's whole
//! job is to bridge it to the catalogs, which live in this crate.

use super::*;

use sceneworks_core::project_store::IMPORTED_WORKFLOW_KEY;
use sceneworks_core::workflow_png::{read_workflow_chunk_file, WorkflowChunkError};
use sceneworks_core::workflow_resolution::{
    build_resolution_report, CatalogEntry, InstallAction, ResolutionReport, StaticCatalogs,
};
use sceneworks_core::workflow_share::{parse_workflow_share, WorkflowShare, WORKFLOW_KIND_IMAGE};

/// `status` when the file carried a readable envelope.
pub(crate) const INSPECT_STATUS_WORKFLOW: &str = "workflow";

/// `status` when the file is a readable image that simply has no workflow in it.
///
/// The distinct, unambiguous branch sc-15951 keys off. Deliberately a 200 with its own status
/// rather than a 404/422: a PNG from anywhere else in the world has no chunk, and calling that a
/// failure would make the normal case look broken.
pub(crate) const INSPECT_STATUS_NO_WORKFLOW: &str = "no_workflow";

/// `code` on the 400 for bytes that are not a PNG.
pub(crate) const INSPECT_CODE_NOT_PNG: &str = "workflow_inspect_not_png";

/// `code` on the 422 for a file that claims a workflow this build will not read (a truncated PNG,
/// two workflow chunks, a zip bomb, a newer schema version, a video envelope).
pub(crate) const INSPECT_CODE_UNREADABLE: &str = "workflow_inspect_unreadable";

/// `code` on the 500 for a SERVER-side read failure of our own staged temp.
///
/// Split out of [`INSPECT_CODE_NOT_PNG`] because the two are not the same fact and the user-facing
/// consequence of confusing them is a lie: `Io` is raised after the caller's bytes were written to
/// `cache/uploads` successfully, so it says nothing about the file. On Windows an antivirus
/// sharing violation on the staging directory is the likely cause, and telling the user their PNG
/// is malformed sends them to fix a file that is fine.
pub(crate) const INSPECT_CODE_READ_FAILED: &str = "workflow_inspect_read_failed";

/// `code` on the 4xx for a request that is not a usable multipart with exactly one `file` field —
/// an empty body, a JSON body, a missing `boundary`, a missing or duplicated `file`.
///
/// A `boundary` failure is rejected by the EXTRACTOR, before the handler body runs, and axum's own
/// rejection is plain text with no `code` at all — unparseable by a reader that branches on one.
pub(crate) const INSPECT_CODE_BAD_MULTIPART: &str = "workflow_inspect_bad_multipart";

/// `code` on the typed 413 for a body over [`MAX_WORKFLOW_INSPECT_BYTES`].
pub(crate) const INSPECT_CODE_TOO_LARGE: &str = "workflow_inspect_too_large";

/// `code` on the 5xx for a failure to WRITE the upload into `cache/uploads`.
pub(crate) const INSPECT_CODE_STAGE_FAILED: &str = "workflow_inspect_stage_failed";

/// `code` on the 5xx for a failure to read this install's catalogs, which the report is built
/// against. Distinct from the file-side codes: the caller's image was fine.
pub(crate) const INSPECT_CODE_CATALOG_FAILED: &str = "workflow_inspect_catalog_failed";

/// Stamp a machine-readable `code` on an error produced by a shared helper that has no per-route
/// vocabulary — the multipart machinery, `stream_multipart_field_to_file`, the catalog readers.
///
/// Every failure this route returns has to carry one, or sc-15951's `code` branch is handed an
/// envelope it cannot classify. A 413 is recognized by status wherever it comes from, since
/// "too large" is the same fact whichever layer noticed it. An error that already carries a code
/// keeps it.
fn coded(mut error: ApiError, fallback: &'static str) -> ApiError {
    if error.code.is_none() {
        error.code = Some(if error.status == StatusCode::PAYLOAD_TOO_LARGE {
            INSPECT_CODE_TOO_LARGE
        } else {
            fallback
        });
    }
    error
}

/// A `multer` failure surfaced by `next_field`/`Field::text`, given this route's vocabulary.
///
/// The status is the one `multer` chose (a field over an inner limit is a 413, malformed framing a
/// 400) rather than a flattened 400, so the caller is told which it was.
fn multipart_error(error: &axum::extract::multipart::MultipartError) -> ApiError {
    coded(
        ApiError {
            status: error.status(),
            detail: error.body_text(),
            context: None,
            code: None,
        },
        INSPECT_CODE_BAD_MULTIPART,
    )
}

/// The `{ workflow, resolution }` contract, plus the `status` discriminator.
///
/// `workflow` and `resolution` are always PRESENT — `null` in the no-workflow case — so a reader
/// never has to tell an absent key from a null one.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WorkflowInspectResponse {
    pub(crate) status: &'static str,
    pub(crate) workflow: Option<WorkflowShare>,
    pub(crate) resolution: Option<ResolutionReport>,
    /// A sentence for the no-workflow case. Absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
}

/// `POST /api/v1/workflows/inspect` — multipart `file` (required) plus an optional `projectId`.
///
/// `projectId` widens the LoRA and recipe-preset lookups to that project's scope (the same
/// builtin → global → project merge the generation path uses); without it only the install-wide
/// scopes are consulted. It is read-only either way — `get_project` reads `project.json` and the
/// project's `loras/manifest.jsonc`, and neither is created if absent.
pub(crate) async fn inspect_workflow(
    State(state): State<AppState>,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Result<Json<WorkflowInspectResponse>, ApiError> {
    // Taken as a `Result` so the EXTRACTOR's failure is mapped here too. Left to axum it renders a
    // plain-text body with no `code`, which is the one shape sc-15951 cannot parse.
    let mut multipart = multipart.map_err(|rejection| ApiError {
        status: rejection.status(),
        detail: rejection.body_text(),
        context: None,
        code: Some(INSPECT_CODE_BAD_MULTIPART),
    })?;
    let mut file: Option<PathBuf> = None;
    let mut project_id: Option<String> = None;
    // Same shape as `assets::import_asset`: the `file` field is staged before a later field can
    // error, so collection happens inside a fallible block whose error path removes the temp.
    let collect_result = async {
        while let Some(field) = multipart.next_field().await.map_err(|error| {
            // An EMPTY body reaches here rather than the extractor — `Multipart` is lazy, so it is
            // the first `next_field` that discovers there is no framing to read.
            multipart_error(&error)
        })? {
            match field.name() {
                Some("file") => {
                    if file.is_some() {
                        return Err(ApiError {
                            status: StatusCode::BAD_REQUEST,
                            detail: "Only one file field is allowed".to_owned(),
                            context: None,
                            code: Some(INSPECT_CODE_BAD_MULTIPART),
                        });
                    }
                    file = Some(stage_inspect_upload(&state, field).await?);
                }
                Some("projectId") => {
                    let value = field
                        .text()
                        .await
                        .map_err(|error| multipart_error(&error))?;
                    let value = value.trim();
                    if !value.is_empty() {
                        project_id = Some(value.to_owned());
                    }
                }
                _ => {}
            }
        }
        Ok::<(), ApiError>(())
    }
    .await;
    if let Err(error) = collect_result {
        if let Some(path) = file.as_ref() {
            let _ = tokio::fs::remove_file(path).await;
        }
        return Err(error);
    }

    let temp_path = file.ok_or_else(|| ApiError {
        status: StatusCode::BAD_REQUEST,
        detail: "Upload file field is required".to_owned(),
        context: None,
        code: Some(INSPECT_CODE_BAD_MULTIPART),
    })?;
    // The chunk read is bounded but synchronous (`workflow_png` walks chunk headers off the
    // executor's thread otherwise). The temp file is removed whether it succeeded, failed, or the
    // task itself panicked — nothing about the outcome may leave an upload behind.
    let read_path = temp_path.clone();
    let read = tokio::task::spawn_blocking(move || read_workflow_chunk_file(&read_path)).await;
    let _ = tokio::fs::remove_file(&temp_path).await;
    let read = read.map_err(|error| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        detail: format!("Workflow chunk read failed: {error}"),
        context: None,
        code: Some(INSPECT_CODE_READ_FAILED),
    })?;

    match read {
        Ok(Some(share)) => {
            let catalogs = inspect_catalogs(&state, project_id.as_deref())
                .await
                .map_err(|error| coded(error, INSPECT_CODE_CATALOG_FAILED))?;
            let resolution = build_resolution_report(&share, &catalogs);
            Ok(Json(WorkflowInspectResponse {
                status: INSPECT_STATUS_WORKFLOW,
                workflow: Some(share),
                resolution: Some(resolution),
                detail: None,
            }))
        }
        Ok(None) => Ok(Json(WorkflowInspectResponse {
            status: INSPECT_STATUS_NO_WORKFLOW,
            workflow: None,
            resolution: None,
            detail: Some(
                "This image carries no SceneWorks workflow, so there is no recipe to read from it."
                    .to_owned(),
            ),
        })),
        Err(error) => Err(inspect_error(&error)),
    }
}

/// `code` on the 5xx for an `extra.importedWorkflow` this build cannot read back.
///
/// Reachable only through a record OUR OWN import wrote, so it is a server-side fault and never a
/// statement about a file the caller sent. Its own code because the caller's next move is a bug
/// report, not a different image.
pub(crate) const ASSET_WORKFLOW_CODE_UNREADABLE: &str = "asset_workflow_unreadable";

/// `GET /api/v1/projects/:project_id/assets/:asset_id/workflow` — the recipe an IMPORTED asset is
/// carrying, plus what this install can do about it right now (sc-15952).
///
/// # Why a route at all, when the envelope is already on the asset
///
/// It is not the envelope that is missing. `list_assets` already hands the web
/// `extra.importedWorkflow` (sc-15949), so the recipe is in the browser before this route is
/// called. What is missing is the **report**: the envelope is a fact about the file and the report
/// is a fact about this machine at this moment, and the catalogs it is computed against
/// (`model_catalog`'s install-state sweep, the four-scope LoRA merge, the Style catalog, the preset
/// merge) live behind the API and are not in the client's hands at all.
///
/// # Why a GET keyed by asset, and not a POST of the envelope
///
/// Both were available. Posting `extra.importedWorkflow` back up and asking for a report would have
/// avoided a route, and it would have made the report a function of client-supplied JSON — an
/// envelope the server re-parses out of a request body is one a caller can edit, and the whole
/// point of sc-15949's "ours or absent" rule on that key is that the stored envelope is the one
/// THIS reader sanitized. Reading it server-side keeps it that way.
///
/// The second reason is the one the AC names: a report goes stale the moment a download finishes,
/// so re-resolution after an install has to be cheap and repeatable. A GET keyed by
/// `(project, asset)` is exactly a refetch — no upload, no multipart, no file the client has to
/// still be holding. (`POST /api/v1/workflows/inspect` could technically serve this by re-uploading
/// the asset's PNG, which is a multi-megabyte round trip to re-read bytes the server already has.)
///
/// Read-only: `get_asset` reads the project's index and nothing is written on any path.
pub(crate) async fn get_asset_workflow(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<WorkflowInspectResponse>, ApiError> {
    let lookup_project = project_id.clone();
    let asset = project_call(state.clone(), move |store| {
        store.get_asset(&lookup_project, &asset_id)
    })
    .await?;
    // Absent is the ordinary answer, not a failure: most assets are generated or were imported
    // from an image with no chunk in it. Same 200 + `no_workflow` shape the inspect route uses for
    // every foreign PNG, so the web has ONE branch.
    let Some(stored) = asset
        .get("extra")
        .and_then(|extra| extra.get(IMPORTED_WORKFLOW_KEY))
    else {
        return Ok(Json(WorkflowInspectResponse {
            status: INSPECT_STATUS_NO_WORKFLOW,
            workflow: None,
            resolution: None,
            detail: Some(
                "This image carries no SceneWorks workflow, so there is no recipe to read from it."
                    .to_owned(),
            ),
        }));
    };
    // The SAME parser the import wrote through — the contract's "one parse path" rule. A stored
    // envelope that fails it is our record being wrong rather than the user's file, so it is a 5xx
    // with its own code instead of being flattened into "no workflow", which would tell the user
    // their image lost a recipe it is visibly still carrying.
    //
    // Nothing in the app can produce this: `import_asset` writes only
    // `serde_json::to_value(&WorkflowShare)`, a value this same parser accepted a moment earlier.
    // It is reachable from OUTSIDE — the envelope lives in the asset's `.sceneworks.json` sidecar,
    // an ordinary file in the user's project that a sync tool, a hand edit or an interrupted
    // restore can leave malformed — which is why this is a real branch rather than a comment, and
    // why it is exercised:
    // `tests::workflows::the_asset_route_500s_with_its_own_code_when_the_stored_envelope_no_longer_parses`
    // plants a corrupt record on a sidecar and pins the coded 500.
    let share = parse_workflow_share(stored).map_err(|error| {
        tracing::error!(event = "asset_workflow_unreadable", %project_id, error = %error);
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: "SceneWorks recorded a workflow on this image but can no longer read it back. \
                     The image itself is unaffected."
                .to_owned(),
            context: None,
            code: Some(ASSET_WORKFLOW_CODE_UNREADABLE),
        }
    })?;
    // The SECOND gate, and the one this route owns: the parser accepts EVERY kind in
    // `WORKFLOW_KINDS`, so `parse_workflow_share` alone stopped being a container check the moment
    // sc-15956 widened the marker to `image` | `video`. Both other readers assert the kind their
    // own container carries — `workflow_png::read_workflow_chunk_file` demands `image`,
    // `workflow_mp4::read_workflow_metadata` demands `video` — and this route is the third reader,
    // so it owes the same assert against the lane it feeds.
    //
    // The lane is what makes this a refusal rather than a widening: the response goes to
    // `api.js::fetchAssetWorkflow` → `useWorkflowDrop.js` → `recipeFromWorkflowShare`, which builds
    // an **Image Studio** prefill. A video envelope reaching that produces a made-up image recipe
    // out of a clip's `durationSeconds` / `fps` / `quality` — the exact "presented as a kind it is
    // not" failure the kind gate exists to prevent, arriving through the one door with no container
    // of its own to check. `assetPanels.jsx`'s `asset.type === "image"` button guard is a proxy for
    // this and not a substitute: a video envelope stored on an image asset passes it unchanged.
    //
    // Same code as an unparseable record, because it is the same fact about the same record — ours,
    // and wrong — and the caller's next move is identical. Pinned by
    // `tests::workflows::the_asset_route_500s_when_the_stored_envelope_names_another_container`.
    if share.kind != WORKFLOW_KIND_IMAGE {
        tracing::error!(
            event = "asset_workflow_unreadable",
            %project_id,
            kind = %share.kind,
            "the stored envelope names a workflow kind this route has no reader for"
        );
        return Err(ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: "SceneWorks recorded a workflow on this image but can no longer read it back. \
                     The image itself is unaffected."
                .to_owned(),
            context: None,
            code: Some(ASSET_WORKFLOW_CODE_UNREADABLE),
        });
    }
    let catalogs = inspect_catalogs(&state, Some(&project_id))
        .await
        .map_err(|error| coded(error, INSPECT_CODE_CATALOG_FAILED))?;
    let resolution = build_resolution_report(&share, &catalogs);
    Ok(Json(WorkflowInspectResponse {
        status: INSPECT_STATUS_WORKFLOW,
        workflow: Some(share),
        resolution: Some(resolution),
        detail: None,
    }))
}

/// The per-field upload cap. [`MAX_WORKFLOW_INSPECT_BYTES`] in production; overridable in tests,
/// because even the reduced cap is 512 MiB and no test wants to send that (the
/// `max_lora_upload_bytes` pattern).
fn max_inspect_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_WORKFLOW_INSPECT_BYTES.with(std::cell::Cell::get);
        if limit > 0 {
            return limit;
        }
    }
    MAX_WORKFLOW_INSPECT_BYTES
}

/// The route's `DefaultBodyLimit` — the per-field cap PLUS multipart framing headroom.
///
/// Derived rather than fixed for two reasons. It cannot drift back into equality with the field
/// cap, which is the bug: with the two equal, a body AT the cap plus its boundaries and part
/// headers exceeds the router limit, so axum rejects it before `stage_inspect_upload` ever counts
/// a byte and the caller sees a plain 400 from `field.chunk()` instead of the typed 413. And it
/// tracks the test override, so the real boundary — a body one byte over the cap, framing included
/// — is reachable at a size a test can actually send.
pub(crate) fn max_inspect_multipart_body_bytes() -> usize {
    let cap = max_inspect_upload_bytes();
    if cap == MAX_WORKFLOW_INSPECT_BYTES {
        return MAX_WORKFLOW_INSPECT_MULTIPART_BODY_BYTES;
    }
    cap.saturating_add(MULTIPART_FRAMING_HEADROOM_BYTES)
}

/// Stream the uploaded image into the SAME `cache/uploads` staging area asset import uses, so the
/// existing `sweep_stale_asset_uploads` startup backstop already covers anything an aborted
/// inspect leaves behind. A near-twin of `assets::write_upload_field_to_temp_file`, split out only
/// so the cap can be lowered in tests.
async fn stage_inspect_upload(
    state: &AppState,
    field: axum::extract::multipart::Field<'_>,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state.settings.data_dir.join("cache").join("uploads");
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| {
            coded(
                ApiError::internal(error.to_string()),
                INSPECT_CODE_STAGE_FAILED,
            )
        })?;
    let temp_path = upload_dir.join(format!("upload-{}.tmp", Uuid::new_v4().simple()));
    stream_multipart_field_to_file(
        field,
        &temp_path,
        max_inspect_upload_bytes(),
        || "Uploaded image is too large".to_owned(),
        || async {
            let _ = tokio::fs::remove_file(&temp_path).await;
        },
    )
    .await
    // The shared streamer has no per-route vocabulary, so the `code` is stamped here. `coded`
    // recognizes the 413 by status; everything else it can return is a failure to WRITE the temp.
    .map_err(|error| coded(error, INSPECT_CODE_STAGE_FAILED))?;
    Ok(temp_path)
}

/// Map a reader failure onto a typed status, a machine-readable `code`, and a sentence fit to show
/// a person.
///
/// Two rules the naive `error.to_string()` broke:
///
/// * **Whose fault is it?** Every variant but [`WorkflowChunkError::Io`] is a fact about the bytes
///   the caller sent, so it is a 4xx. `Io` is raised when our OWN staged temp cannot be opened or
///   read — the upload already succeeded by then — so it is a 5xx, with its own code. Calling a
///   disk fault "this file is not a PNG" is the mislabeling epic 15945 exists to stop.
/// * **Is the sentence showable?** The variants that carry an upstream `detail` inline it into
///   their `Display`, and for a garbage chunk that upstream text is the `png` crate's `Debug`
///   output (`ChunkType { type: ageg, critical: false, … }`). That is diagnostic, not a sentence.
///   Those variants get a human one here and the raw text goes to the LOG, which is where a
///   developer reading a bug report will look for it.
fn inspect_error(error: &WorkflowChunkError) -> ApiError {
    // The full Display — upstream detail and all — always lands here, so nothing is lost by
    // showing the user a cleaner sentence. `Io` is the server's problem, hence the higher level.
    match error {
        WorkflowChunkError::Io { .. } => {
            tracing::error!(event = "workflow_inspect_read_failed", error = %error);
        }
        _ => tracing::debug!(event = "workflow_inspect_rejected", error = %error),
    }
    match error {
        // Not an image we can carry a workflow in. Distinct from "no workflow" on purpose: a JPEG
        // could never have one, where a PNG might.
        WorkflowChunkError::NotPng => ApiError {
            status: StatusCode::BAD_REQUEST,
            detail: error.to_string(),
            context: None,
            code: Some(INSPECT_CODE_NOT_PNG),
        },
        // OUR staged temp could not be opened or read. The caller's bytes were written
        // successfully before this point, so nothing is known to be wrong with their file and the
        // sentence must not imply otherwise.
        WorkflowChunkError::Io { .. } => ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: "SceneWorks could not read the uploaded image back off disk, so its workflow \
                     could not be checked. Nothing is known to be wrong with the file — try again."
                .to_owned(),
            context: None,
            code: Some(INSPECT_CODE_READ_FAILED),
        },
        // A PNG whose framing never got as far as our keyword. The upstream `detail` is `png`'s
        // Debug output; it goes to the log above, not to the user.
        WorkflowChunkError::Png { .. } => ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: "This PNG could not be read far enough to look for a workflow, so it may be \
                     truncated or damaged."
                .to_owned(),
            context: None,
            code: Some(INSPECT_CODE_UNREADABLE),
        },
        // Same treatment, same reason: a corrupt zlib stream's message is not a sentence.
        WorkflowChunkError::UnreadableChunk { limit, .. } => ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: format!(
                "This PNG's workflow text could not be read as text within {limit} bytes, so the \
                 recipe it claims cannot be shown."
            ),
            context: None,
            code: Some(INSPECT_CODE_UNREADABLE),
        },
        // `Encode` is unreachable on a read; given a sentence rather than left to a catch-all so a
        // new variant has to be classified here instead of silently inheriting one.
        WorkflowChunkError::Encode { .. } => ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: "This image's workflow could not be processed.".to_owned(),
            context: None,
            code: Some(INSPECT_CODE_UNREADABLE),
        },
        // The remaining variants' own sentences are already written FOR a person and name a
        // concrete number the user can act on (two chunks, N bytes over the limit, an envelope
        // this build will not read) — no upstream text is inlined into any of them.
        WorkflowChunkError::DuplicateChunk { .. }
        | WorkflowChunkError::MetadataTooLarge { .. }
        | WorkflowChunkError::TextTooLarge { .. }
        | WorkflowChunkError::Envelope(_) => ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: error.to_string(),
            context: None,
            code: Some(INSPECT_CODE_UNREADABLE),
        },
    }
}

/// Build the injected catalog lookup from this install's real catalogs.
///
/// Every read here is one the existing endpoints already do, and all four are read-only:
/// `model_catalog` is the shared install-state snapshot behind `GET /models`, `lora_catalog` is the
/// builtin → global → external → project merge, `styles_catalog` reads `builtin.styles.jsonc`, and
/// `recipe_preset_catalog_with` is the three-scope preset merge.
///
/// ONE [`JobCatalogSnapshot`] is threaded through both model readers (sc-8819). The preset merge
/// needs the model catalog too, so without the snapshot this request read it twice — two full deep
/// clones of the cached value, and two values that could DISAGREE if a model write landed between
/// them, leaving the report's `model` row classified against a different catalog than the presets.
async fn inspect_catalogs(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<StaticCatalogs, ApiError> {
    let snapshot = crate::generation::JobCatalogSnapshot::default();
    let models = snapshot.models(state).await?;
    let loras = snapshot.loras(state, project_id).await?;
    let styles = crate::styles::styles_catalog(state).await?;
    let presets =
        crate::recipe_presets::recipe_preset_catalog_with(state, project_id, Some(&snapshot))
            .await?;
    Ok(StaticCatalogs {
        // `filter_map`, not `map`: a row with no usable `id` is DROPPED rather than reduced to a
        // matchable `id: ""`. A manifest may omit `id` entirely (`load_manifest_entries` validates
        // nothing and `apply_model_catalog_entry` never inserts one), and a blank id used to match
        // an envelope whose `model` had been scrubbed to `""` — reporting a model the file never
        // named as installed. Same rule the styles/presets already followed through `named_entry`.
        models: models.iter().filter_map(model_catalog_entry).collect(),
        loras: loras.iter().filter_map(lora_catalog_entry).collect(),
        styles: style_catalog_entries(&styles),
        recipe_presets: presets.iter().filter_map(named_entry).collect(),
    })
}

fn entry_str(entry: &Value, key: &str) -> Option<String> {
    entry
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn is_installed(entry: &Value) -> bool {
    entry.get("installState").and_then(Value::as_str) == Some("installed")
}

/// A catalog row reduced to `{ id, name }` — for the catalogs where "on disk" is not a question
/// (styles, recipe presets). Rows with no usable id are dropped; they could never be matched.
fn named_entry(entry: &Value) -> Option<CatalogEntry> {
    Some(CatalogEntry {
        id: entry_str(entry, "id")?,
        name: entry_str(entry, "name"),
        repo: None,
        hash: None,
        installed: false,
        install: None,
    })
}

/// One model catalog row, or `None` for a row with no usable `id`.
///
/// `installed` is the `installState` the shared snapshot computed — the whole point of the
/// distinction the report draws, since a cache-only resolver never auto-downloads. The install
/// action is offered only when the row is `downloadable`, which is the same predicate
/// `create_model_download_job` needs to find a Hugging Face download to enqueue.
fn model_catalog_entry(entry: &Value) -> Option<CatalogEntry> {
    // An id-less row is unmatchable and unactionable — the download route is keyed by id — so it
    // is dropped rather than carried as `id: ""`. See `inspect_catalogs`.
    let id = entry_str(entry, "id")?;
    let installed = is_installed(entry);
    let downloadable = entry.get("downloadable").and_then(Value::as_bool) == Some(true);
    let install = (!installed && downloadable).then(|| InstallAction {
        method: "POST".to_owned(),
        path: format!("/api/v1/models/{id}/download"),
    });
    Some(CatalogEntry {
        id,
        name: entry_str(entry, "name"),
        repo: None,
        hash: None,
        installed,
        install,
    })
}

/// One LoRA catalog row.
///
/// `repo` is the Hugging Face source id, which is also the gate on the install action:
/// `create_lora_download_job` refuses a row with no `huggingface` provider, and it looks the row up
/// with `lora_catalog(.., None)` — so a PROJECT-scoped row can never be fetched through it and is
/// deliberately left without an action rather than pointed at a route that would 404.
fn lora_catalog_entry(entry: &Value) -> Option<CatalogEntry> {
    // Same drop as `model_catalog_entry`: an id-less row could only ever be matched by a blank
    // label, and nothing this report is asked about is blank on purpose.
    let id = entry_str(entry, "id")?;
    let source = entry.get("source").and_then(Value::as_object);
    let is_huggingface = source
        .and_then(|source| source.get("provider"))
        .or_else(|| entry.get("provider"))
        .and_then(Value::as_str)
        .map(str::trim)
        == Some("huggingface");
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| entry.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && is_huggingface)
        .map(str::to_owned);
    let installed = is_installed(entry);
    let fetchable = entry.get("scope").and_then(Value::as_str) != Some("project");
    let install = (!installed && repo.is_some() && fetchable).then(|| InstallAction {
        method: "POST".to_owned(),
        path: format!("/api/v1/loras/{id}/download"),
    });
    Some(CatalogEntry {
        id,
        name: entry_str(entry, "name"),
        repo,
        hash: entry_str(entry, "sha256"),
        installed,
        install,
    })
}

/// Flatten the Style catalog into its one id-space: every group id and every sub-style id, which is
/// exactly what `styles::style_text_for_id` resolves against.
fn style_catalog_entries(catalog: &Value) -> Vec<CatalogEntry> {
    let Some(groups) = catalog.get("groups").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for group in groups {
        if let Some(entry) = named_entry(group) {
            entries.push(entry);
        }
        let Some(styles) = group.get("styles").and_then(Value::as_array) else {
            continue;
        };
        entries.extend(styles.iter().filter_map(named_entry));
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_row_that_is_known_but_absent_gets_the_model_manager_download() {
        let entry = model_catalog_entry(&json!({
            "id": "krea_2_turbo",
            "name": "Krea 2 Turbo",
            "installState": "missing",
            "downloadable": true
        }))
        .expect("a row with an id survives");
        assert!(!entry.installed);
        assert_eq!(
            entry.install.as_ref().map(|action| action.path.as_str()),
            Some("/api/v1/models/krea_2_turbo/download")
        );
        assert_eq!(entry.name.as_deref(), Some("Krea 2 Turbo"));
    }

    #[test]
    fn an_installed_model_row_publishes_no_download() {
        let entry = model_catalog_entry(&json!({
            "id": "krea_2_turbo", "installState": "installed", "downloadable": true
        }))
        .expect("a row with an id survives");
        assert!(entry.installed);
        assert!(entry.install.is_none());
    }

    #[test]
    fn a_non_downloadable_model_row_gets_no_action() {
        // An external ComfyUI base or a `downloadable: false` row: the report must call it missing
        // rather than offer a button `create_model_download_job` would refuse.
        let entry = model_catalog_entry(&json!({
            "id": "external_base_x", "installState": "missing", "downloadable": false
        }))
        .expect("a row with an id survives");
        assert!(entry.install.is_none());
    }

    #[test]
    fn a_lora_row_carries_its_hugging_face_repo_and_download() {
        let entry = lora_catalog_entry(&json!({
            "id": "film_grain",
            "name": "Film Grain",
            "scope": "builtin",
            "installState": "missing",
            "source": { "provider": "huggingface", "repo": "acme/film-grain", "file": "x.safetensors" }
        }))
        .expect("a row with an id survives");
        assert_eq!(entry.repo.as_deref(), Some("acme/film-grain"));
        assert_eq!(
            entry.install.as_ref().map(|action| action.path.as_str()),
            Some("/api/v1/loras/film_grain/download")
        );
    }

    #[test]
    fn a_local_lora_row_has_no_repo_and_no_download() {
        let entry = lora_catalog_entry(&json!({
            "id": "aurora_v3",
            "name": "Aurora v3",
            "scope": "global",
            "installState": "installed",
            "source": { "provider": "local", "path": "loras/aurora_v3" }
        }))
        .expect("a row with an id survives");
        assert!(entry.repo.is_none());
        assert!(entry.install.is_none());
        assert!(entry.installed);
    }

    #[test]
    fn a_project_scoped_lora_row_is_never_pointed_at_the_install_route() {
        // `create_lora_download_job` looks the row up with `lora_catalog(.., None)`, so a
        // project-scoped id 404s there — an action would be a button that cannot work.
        let entry = lora_catalog_entry(&json!({
            "id": "proj_lora",
            "scope": "project",
            "installState": "missing",
            "source": { "provider": "huggingface", "repo": "acme/x" }
        }))
        .expect("a row with an id survives");
        assert_eq!(entry.repo.as_deref(), Some("acme/x"));
        assert!(entry.install.is_none());
    }

    #[test]
    fn the_style_catalog_flattens_groups_and_sub_styles_into_one_id_space() {
        let entries = style_catalog_entries(&json!({
            "groups": [{
                "id": "anime-style",
                "name": "Anime Style",
                "styles": [
                    { "id": "ghibli-style", "name": "Ghibli Style" },
                    { "id": "70s-anime", "name": "70s Anime" }
                ]
            }]
        }));
        let ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
        assert_eq!(ids, vec!["anime-style", "ghibli-style", "70s-anime"]);
        assert!(entries.iter().all(|entry| entry.install.is_none()));
    }

    #[test]
    fn an_empty_style_catalog_flattens_to_nothing() {
        assert!(style_catalog_entries(&json!({ "schemaVersion": 1, "groups": [] })).is_empty());
        assert!(style_catalog_entries(&json!({})).is_empty());
    }

    #[test]
    fn not_a_png_is_a_typed_400_and_a_bad_chunk_is_a_typed_422() {
        let not_png = inspect_error(&WorkflowChunkError::NotPng);
        assert_eq!(not_png.status, StatusCode::BAD_REQUEST);
        assert_eq!(not_png.code, Some(INSPECT_CODE_NOT_PNG));
        let duplicate = inspect_error(&WorkflowChunkError::DuplicateChunk { count: 2 });
        assert_eq!(duplicate.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(duplicate.code, Some(INSPECT_CODE_UNREADABLE));
        // Every variant that IS a fact about the caller's bytes is a typed 4xx. `Io` is not one of
        // them — see `a_server_side_read_failure_is_not_reported_as_a_bad_file`.
        for error in [
            WorkflowChunkError::Png {
                detail: "x".to_owned(),
            },
            WorkflowChunkError::MetadataTooLarge { limit: 1 },
            WorkflowChunkError::TextTooLarge { bytes: 2, limit: 1 },
            WorkflowChunkError::UnreadableChunk {
                limit: 1,
                detail: "x".to_owned(),
            },
            WorkflowChunkError::Envelope(
                sceneworks_core::workflow_share::WorkflowShareError::MissingMarker,
            ),
            WorkflowChunkError::Encode {
                detail: "x".to_owned(),
            },
        ] {
            let mapped = inspect_error(&error);
            assert!(
                mapped.status.is_client_error(),
                "{error:?} must be a typed 4xx, got {}",
                mapped.status
            );
            assert!(!mapped.detail.is_empty(), "{error:?} needs a sentence");
            assert!(mapped.code.is_some(), "{error:?} needs a machine code");
        }
    }

    #[test]
    fn a_server_side_read_failure_is_not_reported_as_a_bad_file() {
        // `WorkflowChunkError::Io` fires when OUR staged temp cannot be opened or read — the
        // caller's bytes were written successfully before that point. On Windows an antivirus
        // sharing violation on `cache/uploads` is the plausible cause. Calling that "not a PNG"
        // sends the user to fix a file that is fine.
        let mapped = inspect_error(&WorkflowChunkError::Io {
            detail: "The process cannot access the file because it is being used by another \
                     process. (os error 32)"
                .to_owned(),
        });
        assert_eq!(mapped.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(mapped.code, Some(INSPECT_CODE_READ_FAILED));
        assert_ne!(mapped.code, Some(INSPECT_CODE_NOT_PNG));
        assert!(
            !mapped.detail.contains("not a PNG"),
            "detail: {}",
            mapped.detail
        );
        // The OS message is diagnostic, not a sentence — it goes to the log, not the response.
        assert!(
            !mapped.detail.contains("os error"),
            "detail: {}",
            mapped.detail
        );
    }

    #[test]
    fn a_garbage_chunks_sentence_does_not_leak_the_png_crates_debug_output() {
        // The first time sc-15947's reader failure is shown to a person. `Png`/`UnreadableChunk`
        // inline the upstream text into their `Display`, and for a corrupt chunk that text is a
        // `Debug` dump of a `ChunkType` struct.
        let leaky = "ChunkType { type: ageg, critical: false, private: true, reserved: true, \
                     safecopy: true }";
        for error in [
            WorkflowChunkError::Png {
                detail: leaky.to_owned(),
            },
            WorkflowChunkError::UnreadableChunk {
                limit: 262_144,
                detail: leaky.to_owned(),
            },
        ] {
            let mapped = inspect_error(&error);
            assert_eq!(mapped.status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(mapped.code, Some(INSPECT_CODE_UNREADABLE));
            assert!(
                !mapped.detail.contains("ChunkType") && !mapped.detail.contains("safecopy"),
                "{error:?} leaked its upstream debug text: {}",
                mapped.detail
            );
            assert!(
                mapped.detail.ends_with('.'),
                "{error:?} must read as a sentence: {}",
                mapped.detail
            );
        }
    }

    #[test]
    fn a_catalog_row_with_no_usable_id_is_dropped_rather_than_carried_as_a_blank() {
        // A manifest may omit `id` entirely — `load_manifest_entries` validates nothing and
        // `apply_model_catalog_entry` defaults it. A blank id used to be matchable, and
        // `workflow_share` scrubs an unshareable `model` to exactly that.
        for row in [
            json!({ "name": "Some Other Model", "installState": "installed" }),
            json!({ "id": "", "name": "Some Other Model", "installState": "installed" }),
            json!({ "id": "   ", "name": "Some Other Model", "installState": "installed" }),
        ] {
            assert!(model_catalog_entry(&row).is_none(), "row: {row}");
            assert!(lora_catalog_entry(&row).is_none(), "row: {row}");
        }
        // A row that DOES have one still survives, so the guard drops only the unmatchable.
        assert!(model_catalog_entry(&json!({ "id": "krea_2_turbo" })).is_some());
        assert!(lora_catalog_entry(&json!({ "id": "film_grain" })).is_some());
    }

    #[test]
    fn the_route_body_limit_sits_above_the_per_field_cap() {
        // Issue 4: with the two EQUAL, a body at the cap plus multipart framing trips axum's own
        // limit first and surfaces as a plain 400 from `field.chunk()`, so the typed 413 is
        // unreachable at the only boundary that matters. Same headroom as the LoRA route.
        const {
            assert!(
                MAX_WORKFLOW_INSPECT_MULTIPART_BODY_BYTES > MAX_WORKFLOW_INSPECT_BYTES,
                "the router limit must leave room for boundaries and part headers"
            );
            assert!(
                MAX_WORKFLOW_INSPECT_MULTIPART_BODY_BYTES - MAX_WORKFLOW_INSPECT_BYTES
                    == MULTIPART_FRAMING_HEADROOM_BYTES
            );
            // Issue 8: inspect stages the body, throws it away and returns a small JSON report, so
            // it does not buy the asset route's 2 GiB. The cap must still clear any PNG a user
            // would plausibly drop.
            assert!(MAX_WORKFLOW_INSPECT_BYTES < MAX_UPLOAD_BYTES);
            assert!(
                MAX_WORKFLOW_INSPECT_BYTES >= 512 * 1024 * 1024,
                "a 16384² 8-bit RGB PNG (4096² through a 4x upscale) has to fit"
            );
        }
        // The production value the derived accessor yields, with no test override in force.
        assert_eq!(
            max_inspect_multipart_body_bytes(),
            MAX_WORKFLOW_INSPECT_MULTIPART_BODY_BYTES
        );
    }
}
