use super::*;

/// Machine-readable code on the license-acknowledgment rejection (sc-17227). Mirrored in the web
/// client as `LICENSE_ACK_ERROR_CODE` (`apps/web/src/licenseAcknowledgment.js`) so both halves of
/// the gate name the same refusal rather than matching on prose.
pub(crate) const LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE: &str = "license_acknowledgment_required";

/// True when the catalog entry declares that the user must accept its license before the weights
/// may be downloaded (`requiresLicenseAcknowledgment`, sc-17227). Deliberately does NOT include
/// `gated`: see the call site for why the two are enforced differently.
pub(crate) fn model_requires_license_acknowledgment(model: &Value) -> bool {
    model
        .get("requiresLicenseAcknowledgment")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The payload key every job-creation door uses to carry the caller's acknowledgment through to
/// the queue (sc-17227). Stamped by `create_model_download_job` once its own gate has passed, so a
/// legacy explicit assertions survive retry; server-record authorization is rechecked
/// against current terms and withdrawal on each request.
pub(crate) const LICENSE_ACKNOWLEDGED_PAYLOAD_KEY: &str = "licenseAcknowledged";

/// Error code for a fetch of a repo a catalog entry declares as a BLOCKED conditional component
/// (sc-22998: YuE2's SheetSage2 / MERT-v2-FullSong cover closure). Distinct from
/// [`LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE`] because no acknowledgment clears it: the block is an
/// owner decision the user cannot make from a checkbox.
pub(crate) const COMPONENT_BLOCKED_CODE: &str = "component_blocked";

/// The derivation a `ModelDownload` job runs after fetching the original (sc-22999): the worker
/// derives this tier locally through the audio-lane preparer into
/// `local_derivation_snapshot_dir` and verifies it against the pinned weights.
pub(crate) const LOCAL_DERIVATION_PAYLOAD_KEY: &str = "localDerivation";

/// Canonical comparison key for a Hugging Face `owner/name`. Lowercased so a case-variant repo
/// string cannot walk past a gate keyed on the catalog's spelling — the hub resolves `owner/Name`
/// and `owner/name` to the same repository, so treating them as different would be a bypass.
///
/// A trailing `.git` is stripped for the same reason: `MiniMaxAI/MiniMax-H3.git` is the git-remote
/// spelling of the same repository, it passes the worker's `validate_hf_repo_id`, and it was the
/// one spelling that missed this index — leaving Hugging Face's own 401 as the only thing between
/// the request and the weights. Stripped AFTER the trailing-slash trim (and re-trimmed) so
/// `…/MiniMax-H3.git/` and `…/MiniMax-H3/.git` both normalize too.
pub(super) fn huggingface_repo_key(repo: &str) -> Option<String> {
    let repo = repo.trim().trim_end_matches('/').trim();
    let repo = match repo.rfind('.') {
        // `rfind` yields a char boundary, so the slice is safe; compared case-insensitively
        // because the lowercasing below happens only after this strip.
        Some(dot) if repo[dot..].eq_ignore_ascii_case(".git") => repo[..dot].trim_end_matches('/'),
        _ => repo,
    };
    let repo = repo.trim();
    if repo.is_empty() {
        return None;
    }
    Some(repo.to_ascii_lowercase())
}

/// The `owner/name` a huggingface.co URL addresses, or `None` for any other host. `/models/import`
/// accepts a `sourceUrl` as an alternative to `repo`, and
/// `https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/…` fetches exactly the same bytes as
/// `repo: "MiniMaxAI/MiniMax-H3"`, so a repo-keyed gate that read only `repo` would leave the
/// equivalent request open.
pub(super) fn huggingface_repo_from_url(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let (host, path) = rest.split_once('/')?;
    let host = host.split('@').next_back().unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host).to_ascii_lowercase();
    if host != "huggingface.co" && host != "www.huggingface.co" && host != "hf.co" {
        return None;
    }
    // `/models/<owner>/<name>` and `/<owner>/<name>` both address a model repo; `datasets/…` and
    // `spaces/…` are different namespaces and are left alone.
    let path = path.strip_prefix("models/").unwrap_or(path);
    let mut segments = path.split('/').filter(|segment| !segment.is_empty());
    let owner = segments.next()?;
    if matches!(owner, "datasets" | "spaces" | "api") {
        return None;
    }
    let name = segments.next()?;
    Some(format!("{owner}/{name}"))
}

/// Every Hugging Face repo declared by a catalog entry that requires a license acknowledgment,
/// mapped to the entry that declares it (sc-17227). Includes co-requisite rows: MiniMax-H3's text
/// encoder and both VAEs come straight from `MiniMaxAI/MiniMax-H3`, which is the repo the review's
/// bypass named, and a primary-only index would have missed it.
///
/// Also every `conditionalComponents[].repo` (sc-22998). Those repos are not `downloads[]` rows, so
/// no install fetches them, but every repo-keyed door (`POST /api/v1/jobs`, `/models/import`,
/// `/loras/import`, `/loras/:id/download`) takes a caller-supplied repo verbatim, so the gate must
/// know them. A component carrying `blocked` is indexed for EVERY entry, acknowledgment or not, and
/// is refused outright by [`ensure_license_acknowledged_for_source`]; a blocked entry takes
/// precedence over an acknowledgment entry for the same repo.
///
/// Read from the UNFILTERED manifest entries on purpose. The catalog snapshot narrows `downloads`
/// to the running OS (`retain_downloads_for_os`), and every MiniMax-H3 row is platform-scoped: the
/// MLX tiers and their co-requisites are `platforms: ["macos"]` and sc-19558's raw-snapshot set is
/// `platforms: ["windows", "linux"]`. An index built from the snapshot would therefore see only the
/// subset that survived the filter on the running host, and the gate would be keyed on a partial
/// view of the repos an entry can actually fetch — on exactly the hosts where the LAN-exposed jobs
/// API (epic 4484) is most likely to be reachable. A licence requirement is not a platform
/// capability.
pub(super) async fn license_acknowledgment_repo_index(
    state: &AppState,
) -> Result<std::collections::BTreeMap<String, LicenseAcknowledgmentSource>, ApiError> {
    let (models, _) = merged_model_manifest_entries(state).await?;
    Ok(license_acknowledgment_index_for(&models))
}

/// [`license_acknowledgment_repo_index`] over already-loaded manifest entries.
pub(super) fn license_acknowledgment_index_for(
    models: &[Value],
) -> std::collections::BTreeMap<String, LicenseAcknowledgmentSource> {
    license_acknowledgment_index_for_selected(models, None)
}

pub(super) fn license_acknowledgment_index_for_selected(
    models: &[Value],
    selected_model_id: Option<&str>,
) -> std::collections::BTreeMap<String, LicenseAcknowledgmentSource> {
    let mut index: std::collections::BTreeMap<String, LicenseAcknowledgmentSource> =
        std::collections::BTreeMap::new();
    // A catalog-validated selection owns its declared shared repositories. Raw
    // repository requests retain canonical manifest order; an arbitrary accepted
    // model must never grant access. Blocked components from ANY entry still win.
    let selected = |model: &&Value| {
        selected_model_id.is_some_and(|id| model.get("id").and_then(Value::as_str) == Some(id))
    };
    for model in models
        .iter()
        .filter(selected)
        .chain(models.iter().filter(|model| !selected(model)))
    {
        let Some(model_id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        let model_name = model
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(model_id)
            .to_owned();
        let source = |blocked: Option<BlockedComponent>| LicenseAcknowledgmentSource {
            model_id: model_id.to_owned(),
            model_name: model_name.clone(),
            terms: model.clone(),
            blocked,
        };
        let conditional = model
            .get("conditionalComponents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        // Blocked conditional components first, for EVERY entry: no acknowledgment unblocks them.
        for component in conditional.clone() {
            let Some(block) = component.get("blocked") else {
                continue;
            };
            let Some(key) = component
                .get("repo")
                .and_then(Value::as_str)
                .and_then(huggingface_repo_key)
            else {
                continue;
            };
            let text = |field: &str| {
                block
                    .get(field)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            index.insert(
                key,
                source(Some(BlockedComponent {
                    component_id: component
                        .get("componentId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    reason: text("reason"),
                    unblock: text("unblock"),
                })),
            );
        }
        if !model_requires_license_acknowledgment(model) {
            continue;
        }
        let downloads = model
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        for row in downloads.chain(conditional) {
            let Some(key) = row
                .get("repo")
                .and_then(Value::as_str)
                .and_then(huggingface_repo_key)
            else {
                continue;
            };
            index.entry(key).or_insert_with(|| source(None));
        }
    }
    index
}

/// SW-7100 (E23): MiniMax-H3-Image is gated at RUN as well as at install. Its weights are fetched
/// only through the acknowledgment gate, but a cache populated by another route (a shared HF cache,
/// a copied data directory) would otherwise render with no acceptance on record, and the
/// MiniMax-H3-Image agreement binds whoever uses the model (§I.9). Scoped to that family so no
/// other model's existing run behaviour changes; the acceptance is the same terms-bound server
/// record the download gate writes.
pub(crate) async fn ensure_run_license_acknowledged(
    state: &AppState,
    entry: &JsonObject,
) -> Result<(), ApiError> {
    // SW-7102: the prompt expander is under the same agreement and is gated the same way.
    if !matches!(
        entry.get("family").and_then(Value::as_str),
        Some(
            sceneworks_core::model_family::MINIMAX_H3_IMAGE
                | sceneworks_core::model_family::MINIMAX_H3_IMAGE_PE
        )
    ) {
        return Ok(());
    }
    let entry = Value::Object(entry.clone());
    if !model_requires_license_acknowledgment(&entry)
        || crate::yue2_jobs::current_license_acknowledgment(state, &entry)
            .await?
            .is_some()
    {
        return Ok(());
    }
    let model_id = entry.get("id").and_then(Value::as_str).unwrap_or_default();
    Err(ApiError {
        status: StatusCode::FORBIDDEN,
        detail: format!(
            "'{model_id}' requires accepting its license before it can be used. Accept the license \
             on the Models screen, then retry."
        ),
        code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
        context: None,
    })
}

/// A conditional component a catalog entry declares as `blocked` (sc-22998).
#[derive(Clone)]
pub(crate) struct BlockedComponent {
    pub(crate) component_id: String,
    pub(crate) reason: String,
    pub(crate) unblock: String,
}

/// The catalog entry whose licence acknowledgment covers a fetch of some repo. The NAME travels
/// with the id because the surfaces that have to explain the requirement — the LoRA rows on the
/// Models screen — have no licence copy of their own and must point the user at the model card
/// that does.
#[derive(Clone)]
pub(crate) struct LicenseAcknowledgmentSource {
    pub(crate) model_id: String,
    pub(crate) model_name: String,
    pub(super) terms: Value,
    /// Set when the repo is a BLOCKED conditional component (sc-22998): refused whatever the
    /// caller asserts.
    pub(crate) blocked: Option<BlockedComponent>,
}

/// Client-visible keys naming the model whose licence acknowledgment covers a catalog row that is
/// not itself a model (sc-17227). Written onto LoRA catalog rows by `list_loras`.
pub(crate) const LICENSE_ACKNOWLEDGMENT_MODEL_ID_KEY: &str = "licenseAcknowledgmentModelId";
pub(crate) const LICENSE_ACKNOWLEDGMENT_MODEL_NAME_KEY: &str = "licenseAcknowledgmentModelName";

/// Stamp each catalog row whose Hugging Face source repo is licence-gated with the model that
/// gates it (sc-17227), so a client can raise the SAME acknowledgment gate it raises on a model
/// and send the assertion the server now requires.
///
/// Without this, `create_lora_download_job`'s repo-keyed gate is unsatisfiable from the shipped UI:
/// the row carries nothing that says an acknowledgment is needed, `createLoraDownloadJob` sends no
/// `licenseAcknowledged`, and the click yields a bare 403 with no checkbox anywhere to clear it.
///
/// Derived here rather than authored in `builtin.loras.jsonc` on purpose. A manifest flag is a
/// second copy of a fact the model manifest already states, and the two drift; this reads the one
/// source. It is also the only form that is correct on every host — the index is built from the
/// UNFILTERED model manifest, so it does not evaporate on a platform where the gating model's
/// download rows are filtered out, which is exactly where a client-side re-derivation would fail.
///
/// Applied at the CATALOG-READ door (`list_loras`) and not inside `lora_catalog`, which the
/// per-job-create validation sweep also calls (sc-8819): the annotation is for rendering, and the
/// enforcement path resolves the repo itself.
pub(crate) async fn annotate_license_acknowledgment_sources(
    state: &AppState,
    rows: &mut [Value],
    repo_of: impl Fn(&Value) -> Option<String>,
) -> Result<(), ApiError> {
    let keyed: Vec<(usize, String)> = rows
        .iter()
        .enumerate()
        .filter_map(|(position, row)| {
            repo_of(row)
                .as_deref()
                .and_then(huggingface_repo_key)
                .map(|key| (position, key))
        })
        .collect();
    if keyed.is_empty() {
        return Ok(());
    }
    let index = license_acknowledgment_repo_index(state).await?;
    if index.is_empty() {
        return Ok(());
    }
    for (position, key) in keyed {
        let Some(source) = index
            .get(key.as_str())
            .filter(|source| source.blocked.is_none())
        else {
            continue;
        };
        let Some(object) = rows[position].as_object_mut() else {
            continue;
        };
        object.insert(
            LICENSE_ACKNOWLEDGMENT_MODEL_ID_KEY.to_owned(),
            Value::String(source.model_id.clone()),
        );
        object.insert(
            LICENSE_ACKNOWLEDGMENT_MODEL_NAME_KEY.to_owned(),
            Value::String(source.model_name.clone()),
        );
    }
    Ok(())
}

/// The license-acknowledgment refusal for a request that named its weights by REPO (sc-17227).
///
/// [`create_model_download_job`] gates the typed `POST /api/v1/models/:id/download` by catalog id.
/// That is not the only door: `POST /api/v1/jobs` enqueues a `model_download` payload VERBATIM
/// (`repo` + `files` + `revision`, no catalog lookup anywhere between the request and
/// `run_model_download_job`), and `POST /api/v1/models/import` fetches a caller-supplied repo or
/// URL with no licence logic of its own. Both reached `MiniMaxAI/MiniMax-H3` — a PUBLIC repo, so
/// nothing upstream refuses them — while the typed route answered 403. Keying on the repo rather
/// than on the model id is what lets ONE mechanism close both: the payloads have no `modelId` to
/// look up, but they must name the repo or they cannot fetch anything.
///
/// `repos` is a LIST because "the repo this request will fetch" is not always spelled `repo`: a
/// `model_convert` payload names its download target in `baseRepo` (the LTX converter's
/// `ensure_ltx_upscaler_cached` fetches it, and `upscalerFile` is a glob, so `"**"` pulls the whole
/// repo). Checking every repo-bearing key of a payload is what keeps this ONE predicate rather than
/// one per job type — a new key is an addition to the list, not a second gate.
///
/// `acknowledged` is a legacy explicit acceptance assertion. Without it, the existing
/// terms-bound server acknowledgment is required (see
/// `docs/minimax-h3-use-restriction-safeguards.md`).
pub(crate) async fn ensure_license_acknowledged_for_source(
    state: &AppState,
    repos: &[Option<&str>],
    source_url: Option<&str>,
    acknowledged: bool,
) -> Result<(), ApiError> {
    ensure_license_acknowledged_for_source_inner(state, repos, source_url, acknowledged, None).await
}

// Only catalog-resolved model/component download routes supply a selected ID.
// Other repository-bearing doors deliberately use the public canonical wrapper.
pub(super) async fn ensure_license_acknowledged_for_source_inner(
    state: &AppState,
    repos: &[Option<&str>],
    source_url: Option<&str>,
    acknowledged: bool,
    selected_model_id: Option<&str>,
) -> Result<(), ApiError> {
    // Each candidate keeps the caller's own spelling next to the lookup key, so the refusal echoes
    // what was actually requested rather than the lowercased index key.
    let candidates: Vec<(String, String)> = repos
        .iter()
        .copied()
        .flatten()
        .map(str::to_owned)
        .chain(source_url.and_then(huggingface_repo_from_url))
        .filter_map(|named| huggingface_repo_key(&named).map(|key| (named, key)))
        .collect();
    if candidates.is_empty() {
        return Ok(());
    }
    let index = if selected_model_id.is_some() {
        let (models, _) = merged_model_manifest_entries(state).await?;
        license_acknowledgment_index_for_selected(&models, selected_model_id)
    } else {
        license_acknowledgment_repo_index(state).await?
    };
    // A blocked conditional component is refused before anything else, and an acknowledgment does
    // not clear it (sc-22998).
    if let Some((requested, source, blocked)) = candidates.iter().find_map(|(named, key)| {
        index.get(key.as_str()).and_then(|source| {
            source
                .blocked
                .as_ref()
                .map(|blocked| (named, source, blocked))
        })
    }) {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            detail: format!(
                "'{requested}' is the '{component}' component of '{model_id}', which is blocked: \
                 {reason} Unblock condition: {unblock}",
                component = blocked.component_id,
                model_id = source.model_id,
                reason = blocked.reason,
                unblock = blocked.unblock,
            ),
            code: Some(COMPONENT_BLOCKED_CODE),
            context: None,
        });
    }
    // Every restricted source must have a current acknowledgment. Checking only the
    // first repo could authorize a second component governed by different terms.
    let mut missing = None;
    for (requested, key) in &candidates {
        let Some(source) = index.get(key.as_str()) else {
            continue;
        };
        if !acknowledged
            && crate::yue2_jobs::current_license_acknowledgment(state, &source.terms)
                .await?
                .is_none()
        {
            missing = Some((requested, source));
            break;
        }
    }
    let Some((requested, source)) = missing else {
        return Ok(());
    };
    let model_id = &source.model_id;
    Err(ApiError {
        status: StatusCode::FORBIDDEN,
        detail: format!(
            "'{requested}' supplies '{model_id}', which requires accepting its license before \
             download. Accept the license on the Models screen, or send \
             `licenseAcknowledged: true` to assert that the user has accepted it."
        ),
        code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
        context: None,
    })
}

/// Fine-tune license text never replaces obligations of the components it actually reuses.
/// The caller supplies a server-resolved entry; new imports use their immutable declaration and
/// older H3 entries derive obligations from the trusted builtin component catalog.
pub(crate) async fn ensure_inherited_component_licenses_acknowledged(
    state: &AppState,
    entry: &JsonObject,
    license_acknowledged: bool,
) -> Result<(), ApiError> {
    use sceneworks_core::license_acknowledgments as acknowledgments;
    // Architecture is not an import claim. Ordinary stock/custom HF or local catalog rows keep
    // their own license guards; only positive plan, declaration, or import-origin evidence owes
    // inherited-component authority. Never let a false author license/stock flag erase evidence.
    let import_path = |path: &str| {
        path.starts_with("models/imports/")
            || path.contains("/models/imports/")
            || path.starts_with("linked/")
    };
    let source = entry.get("source").and_then(Value::as_object);
    let import_origin = sceneworks_core::jobs_store::imported_entry_installed_path(entry)
        .is_some_and(import_path)
        || source.is_some_and(|source| {
            source.contains_key("rootId")
                || source
                    .get("path")
                    .and_then(Value::as_str)
                    .is_some_and(import_path)
        });
    let has_import_evidence = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
        .is_some()
        || [
            "importPlan",
            "baseModel",
            "minimaxH3Partition",
            "importSourceShape",
            "importComponentTier",
            "importQuantFormat",
            "modelDeclaration",
            "modelProvenance",
        ]
        .iter()
        .any(|key| entry.contains_key(*key))
        || entry
            .get("inheritedLicenseAcknowledgments")
            .is_some_and(|value| value.as_array().is_none_or(|rows| !rows.is_empty()))
        || import_origin;
    if !has_import_evidence {
        return Ok(());
    }
    let declaration =
        sceneworks_core::model_artifacts::provenance::import_declaration_context(entry)
            .map_err(ApiError::bad_request)?;
    let bound = sceneworks_core::model_artifacts::provenance::resolved_model_provenance(
        &state.settings.data_dir,
        entry,
    )
    .map_err(ApiError::bad_request)?;
    let inherited = bound
        .as_ref()
        .and_then(|value| value.get("baseLineage"))
        .and_then(|_| {
            bound
                .as_ref()
                .and_then(|value| value.get("inheritedLicenseAcknowledgments"))
        })
        .or_else(|| {
            declaration
                .get("modelDeclaration")
                .and_then(|value| value.get("inheritedLicenseAcknowledgments"))
        });
    let path = acknowledgments::acknowledgments_file(&state.settings.config_dir);
    for component in inherited.and_then(Value::as_array).into_iter().flatten() {
        if component
            .get("requiresLicenseAcknowledgment")
            .and_then(Value::as_bool)
            != Some(true)
        {
            continue;
        }
        let id = component
            .get("modelId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                ApiError::bad_request("Inherited component license has no model identity")
            })?;
        let terms = json!({"id":id,"license":component.get("license"),
            "licenseUrl":component.get("licenseUrl"),"licenseNotice":component.get("licenseNotice")});
        if component.get("termsSha256").and_then(Value::as_str)
            != Some(acknowledgments::terms_sha256(&terms).as_str())
        {
            return Err(ApiError::bad_request(
                "Inherited component license commitment is invalid",
            ));
        }
        if license_acknowledged {
            acknowledgments::record_acknowledgment(
                &path,
                &terms,
                "api-import-components",
                &sceneworks_core::time::utc_now(),
            )
            .map_err(|error| ApiError::internal(error.to_string()))?;
        } else if acknowledgments::current_acknowledgment(&path, &terms)
            .map_err(|error| ApiError::internal(error.to_string()))?
            .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: format!("Component '{id}' requires accepting its inherited license. Accept the component terms on the Models screen or send licenseAcknowledged: true."),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE), context: None,
            });
        }
    }
    Ok(())
}

/// Every payload key that can name a Hugging Face repo the WORKER will fetch (sc-17227). Keep this
/// aligned with the worker's own readers: `run_model_download_job` / `run_model_import_job` /
/// `run_lora_*_job` take `repo` (`crates/sceneworks-worker/src/model_jobs.rs`), and
/// `resolve_convert_plan` takes `baseRepo` — which the LTX arm hands to `ensure_ltx_upscaler_cached`
/// → `ensure_hf_files_cached`, a real download. `sourceRepo` is listed because it is the other repo
/// a convert payload names; it resolves against the local cache today (`huggingface_snapshot_dir`),
/// so gating it costs nothing and removes the question of which of the two a future arm fetches.
pub(super) const LICENSE_GATED_REPO_PAYLOAD_KEYS: &[&str] = &["repo", "baseRepo", "sourceRepo"];

/// [`ensure_license_acknowledged_for_source`] over a raw job payload — the shape
/// `POST /api/v1/jobs` (and the retry/duplicate re-validation) hands to the worker verbatim.
pub(crate) async fn ensure_job_payload_license_acknowledged(
    state: &AppState,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let repos: Vec<Option<&str>> = LICENSE_GATED_REPO_PAYLOAD_KEYS
        .iter()
        .map(|key| payload.get(*key).and_then(Value::as_str))
        .collect();
    ensure_license_acknowledged_for_source(
        state,
        &repos,
        payload.get("sourceUrl").and_then(Value::as_str),
        payload
            .get(LICENSE_ACKNOWLEDGED_PAYLOAD_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )
    .await
}
