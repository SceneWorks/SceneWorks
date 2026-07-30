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
//!   4xx carrying a machine-readable `code` and the reader's own sentence. Never a 500.
//!
//! The resolution report itself is `sceneworks_core::workflow_resolution` — shared with the import
//! path (sc-15949) and the web (sc-15951 / sc-15952) rather than owned here. This module's whole
//! job is to bridge it to the catalogs, which live in this crate.

use super::*;

use sceneworks_core::workflow_png::{read_workflow_chunk_file, WorkflowChunkError};
use sceneworks_core::workflow_resolution::{
    build_resolution_report, CatalogEntry, InstallAction, ResolutionReport, StaticCatalogs,
};
use sceneworks_core::workflow_share::WorkflowShare;

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
    mut multipart: Multipart,
) -> Result<Json<WorkflowInspectResponse>, ApiError> {
    let mut file: Option<PathBuf> = None;
    let mut project_id: Option<String> = None;
    // Same shape as `assets::import_asset`: the `file` field is staged before a later field can
    // error, so collection happens inside a fallible block whose error path removes the temp.
    let collect_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            match field.name() {
                Some("file") => {
                    if file.is_some() {
                        return Err(ApiError::bad_request("Only one file field is allowed"));
                    }
                    file = Some(stage_inspect_upload(&state, field).await?);
                }
                Some("projectId") => {
                    let value = field
                        .text()
                        .await
                        .map_err(|error| ApiError::bad_request(error.to_string()))?;
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

    let temp_path = file.ok_or_else(|| ApiError::bad_request("Upload file field is required"))?;
    // The chunk read is bounded but synchronous (`workflow_png` walks chunk headers off the
    // executor's thread otherwise). The temp file is removed whether it succeeded, failed, or the
    // task itself panicked — nothing about the outcome may leave an upload behind.
    let read_path = temp_path.clone();
    let read = tokio::task::spawn_blocking(move || read_workflow_chunk_file(&read_path)).await;
    let _ = tokio::fs::remove_file(&temp_path).await;
    let read =
        read.map_err(|error| ApiError::internal(format!("Workflow chunk read failed: {error}")))?;

    match read {
        Ok(Some(share)) => {
            let catalogs = inspect_catalogs(&state, project_id.as_deref()).await?;
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

/// The per-field upload cap. `MAX_UPLOAD_BYTES` in production; overridable in tests, because the
/// real cap is 2 GiB and no test can send that (the `max_lora_upload_bytes` pattern).
fn max_inspect_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_WORKFLOW_INSPECT_BYTES.with(std::cell::Cell::get);
        if limit > 0 {
            return limit;
        }
    }
    MAX_UPLOAD_BYTES
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
        .map_err(|error| ApiError::internal(error.to_string()))?;
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
    .await?;
    Ok(temp_path)
}

/// Map a reader failure onto a typed 4xx. Never a 500: every variant here is a fact about the
/// bytes the caller sent, and each carries the reader's own user-facing sentence.
fn inspect_error(error: &WorkflowChunkError) -> ApiError {
    match error {
        // Not an image we can carry a workflow in. Distinct from "no workflow" on purpose: a JPEG
        // could never have one, where a PNG might.
        WorkflowChunkError::NotPng => ApiError {
            status: StatusCode::BAD_REQUEST,
            detail: error.to_string(),
            code: Some(INSPECT_CODE_NOT_PNG),
        },
        // Reading the FILE failed (unreadable temp). The caller's bytes did not describe an
        // openable file, so it is still their request that is at fault — and 500 would hide it.
        WorkflowChunkError::Io { .. } => ApiError {
            status: StatusCode::BAD_REQUEST,
            detail: error.to_string(),
            code: Some(INSPECT_CODE_NOT_PNG),
        },
        // A PNG that claims a workflow we will not guess at: truncated framing, two workflow
        // chunks, oversized metadata, a zip bomb, a newer schema version, a video envelope.
        WorkflowChunkError::Png { .. }
        | WorkflowChunkError::DuplicateChunk { .. }
        | WorkflowChunkError::MetadataTooLarge { .. }
        | WorkflowChunkError::TextTooLarge { .. }
        | WorkflowChunkError::UnreadableChunk { .. }
        | WorkflowChunkError::Envelope(_)
        // `Encode` is unreachable on a read; folded in rather than left to a catch-all so a new
        // variant has to be classified here instead of silently becoming a 422.
        | WorkflowChunkError::Encode { .. } => ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: error.to_string(),
            code: Some(INSPECT_CODE_UNREADABLE),
        },
    }
}

/// Build the injected catalog lookup from this install's real catalogs.
///
/// Every read here is one the existing endpoints already do, and all four are read-only:
/// `model_catalog` is the shared install-state snapshot behind `GET /models`, `lora_catalog` is the
/// builtin → global → external → project merge, `styles_catalog` reads `builtin.styles.jsonc`, and
/// `recipe_preset_catalog` is the three-scope preset merge.
async fn inspect_catalogs(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<StaticCatalogs, ApiError> {
    let models = crate::models::model_catalog(state).await?;
    let loras = crate::loras::lora_catalog(state, project_id).await?;
    let styles = crate::styles::styles_catalog(state).await?;
    let presets = crate::recipe_presets::recipe_preset_catalog(state, project_id).await?;
    Ok(StaticCatalogs {
        models: models.iter().map(model_catalog_entry).collect(),
        loras: loras.iter().map(lora_catalog_entry).collect(),
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
        installed: false,
        install: None,
    })
}

/// One model catalog row.
///
/// `installed` is the `installState` the shared snapshot computed — the whole point of the
/// distinction the report draws, since a cache-only resolver never auto-downloads. The install
/// action is offered only when the row is `downloadable`, which is the same predicate
/// `create_model_download_job` needs to find a Hugging Face download to enqueue.
fn model_catalog_entry(entry: &Value) -> CatalogEntry {
    let id = entry_str(entry, "id").unwrap_or_default();
    let installed = is_installed(entry);
    let downloadable = entry.get("downloadable").and_then(Value::as_bool) == Some(true);
    let install = (!installed && downloadable && !id.is_empty()).then(|| InstallAction {
        method: "POST".to_owned(),
        path: format!("/api/v1/models/{id}/download"),
    });
    CatalogEntry {
        id,
        name: entry_str(entry, "name"),
        repo: None,
        installed,
        install,
    }
}

/// One LoRA catalog row.
///
/// `repo` is the Hugging Face source id, which is also the gate on the install action:
/// `create_lora_download_job` refuses a row with no `huggingface` provider, and it looks the row up
/// with `lora_catalog(.., None)` — so a PROJECT-scoped row can never be fetched through it and is
/// deliberately left without an action rather than pointed at a route that would 404.
fn lora_catalog_entry(entry: &Value) -> CatalogEntry {
    let id = entry_str(entry, "id").unwrap_or_default();
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
    let install =
        (!installed && repo.is_some() && fetchable && !id.is_empty()).then(|| InstallAction {
            method: "POST".to_owned(),
            path: format!("/api/v1/loras/{id}/download"),
        });
    CatalogEntry {
        id,
        name: entry_str(entry, "name"),
        repo,
        installed,
        install,
    }
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
        }));
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
        }));
        assert!(entry.installed);
        assert!(entry.install.is_none());
    }

    #[test]
    fn a_non_downloadable_model_row_gets_no_action() {
        // An external ComfyUI base or a `downloadable: false` row: the report must call it missing
        // rather than offer a button `create_model_download_job` would refuse.
        let entry = model_catalog_entry(&json!({
            "id": "external_base_x", "installState": "missing", "downloadable": false
        }));
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
        }));
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
        }));
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
        }));
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
        // Never a 500, whatever the variant.
        for error in [
            WorkflowChunkError::Io {
                detail: "x".to_owned(),
            },
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
        }
    }
}
