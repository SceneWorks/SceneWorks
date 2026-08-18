//! The API's single model-source seam (sc-19708).
//!
//! Every job-creation path calls [`ensure_runtime_model_sources`] with its final payload. The
//! seam — never the route — attaches the generic model carriers a job needs (catalog manifest
//! entries, declarative data) and runs submission preflight through the one shared availability
//! resolver in `sceneworks_core::model_artifacts`. Route handlers contribute only data: the
//! payload fields their request contract already carries (`model`, `baseModel`, `embedder`,
//! `modelNameOrPath`, analyzer flags). Which payload fields reference models, and which job types
//! always load a fixed utility model, are declared in the two tables below — tables at the seam,
//! never logic in a route, and never a per-model branch.
//!
//! Preflight rejects a submission with the typed 503 `external_model_library_unavailable` only
//! when a model's durable install identity proves it lives on a currently unavailable external
//! library. Genuinely missing or incomplete installs keep their established download/on-demand
//! behavior. The worker's own pre-loader guard re-judges the same closure for the claiming
//! worker's platform at job start, so this preflight is user experience, not the enforcement
//! boundary.

use crate::error::ApiError;
use crate::AppState;
use sceneworks_core::contracts::{JobType, JsonObject};
use sceneworks_core::model_artifacts::artifact_selection::{
    requested_runtime_variant, selected_requirements_for_model, LocalCacheEligibility,
};
use sceneworks_core::model_artifacts::external_library::{
    resolve_model_availability, ExternalLibraryUnavailableContext, ModelAvailability,
    ModelResolution,
};
use sceneworks_core::model_artifacts::ResolvedModelArtifact;
use serde_json::{json, Value};
use std::path::Path;

/// How a payload field references a model. `CatalogId` values resolve through the catalog
/// manifest by id; `Repository` values (utility routes that historically named a provider repo)
/// resolve to the catalog entry whose supported download declares that repository.
#[derive(Clone, Copy)]
enum PayloadModelRef {
    CatalogId(&'static str),
    Repository(&'static str),
}

/// DATA TABLE: which payload fields of each job type reference a model. The default row covers
/// every generation route (their typed handlers already attach `modelManifestEntry`; the ids here
/// only fill gaps and are deduplicated against present carriers).
fn payload_reference_spec(job_type: &JobType) -> &'static [PayloadModelRef] {
    use PayloadModelRef::{CatalogId, Repository};
    match job_type {
        JobType::PromptRefine => &[Repository("model")],
        JobType::TrainingCaption => &[Repository("modelNameOrPath")],
        JobType::DatasetAnalysis => &[CatalogId("embedder")],
        JobType::LoraTrain | JobType::ControlTraining => &[CatalogId("baseModel")],
        _ => &[CatalogId("model"), CatalogId("modelId")],
    }
}

/// DATA TABLE: job types that always load a fixed utility model, keyed on the job type and the
/// request's own data (engine/flags) — never on a model name. Referenced ids are ordinary catalog
/// manifest entries; adding a model or utility changes manifest data, not this seam.
fn fixed_runtime_model_ids(job_type: &JobType, payload: &JsonObject) -> Vec<&'static str> {
    let payload_flag = |key: &str| payload.get(key).and_then(Value::as_bool).unwrap_or(false);
    match job_type {
        JobType::ImageVqa | JobType::ImageInterleave => vec!["sensenova_u1_8b"],
        JobType::PersonDetect => vec!["person_detector"],
        JobType::PersonTrack => vec!["person_detector", "sam3_person_segment"],
        JobType::PoseDetect => vec!["dwpose_pose_detector"],
        JobType::ImageSegment => vec!["sam3_person_segment"],
        JobType::ImageDetail
            if payload
                .get("model")
                .and_then(Value::as_str)
                .map_or(true, |model| model.trim().is_empty()) =>
        {
            vec!["realvisxl"]
        }
        JobType::VideoUpscale => vec!["seedvr2_upscaler"],
        JobType::ImageUpscale => {
            let seedvr2 = payload
                .get("engine")
                .and_then(Value::as_str)
                .is_some_and(|engine| engine.eq_ignore_ascii_case("seedvr2"));
            vec![if seedvr2 {
                "seedvr2_upscaler"
            } else {
                "real_esrgan"
            }]
        }
        JobType::DatasetUpscale => vec!["real_esrgan"],
        JobType::PromptRefine
            if payload
                .get("model")
                .and_then(Value::as_str)
                .map_or(true, |model| model.trim().is_empty()) =>
        {
            vec!["prompt_refine_anubis_8b"]
        }
        JobType::DatasetFaceAnalysis | JobType::FaceLikenessCompare | JobType::KpsExtract => {
            vec!["instantid_face_stack"]
        }
        JobType::CatalogAnalysis => {
            let mut ids = Vec::new();
            if payload_flag("structuredAnalysisEnabled") {
                ids.extend([
                    "person_detector",
                    "dwpose_pose_detector",
                    "instantid_face_stack",
                ]);
            }
            if payload_flag("visionAnalysisEnabled") {
                ids.push("vision_caption_qwen3vl_8b");
            }
            if payload_flag("semanticEmbeddingsEnabled") {
                ids.push("clip_vit_l14");
            }
            ids
        }
        _ => Vec::new(),
    }
}

fn payload_has_model_entry(payload: &JsonObject, model_id: &str) -> bool {
    payload_model_entries(payload)
        .into_iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
}

fn payload_model_entries(payload: &JsonObject) -> Vec<&Value> {
    ["modelManifestEntry", "baseModelManifestEntry"]
        .into_iter()
        .filter_map(|key| payload.get(key))
        .chain(
            payload
                .get("modelManifestEntries")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter(|entry| !entry.is_null())
        .collect()
}

fn push_model_entry(payload: &mut JsonObject, entry: Value) -> Result<(), ApiError> {
    payload
        .entry("modelManifestEntries".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| ApiError::bad_request("modelManifestEntries must be an array"))?
        .push(entry);
    Ok(())
}

/// Complete carrier attachment + submission preflight for the final job payload shape. See the
/// module docs for the contract. Explicit downloads carry a distinct typed operation because
/// their desired destination is intentionally missing and must not masquerade as a ready runtime
/// source.
pub(crate) async fn ensure_runtime_model_sources(
    state: &AppState,
    job_type: &JobType,
    payload: &mut JsonObject,
) -> Result<(), ApiError> {
    if matches!(job_type, JobType::ModelImport) {
        // Imports consume the request-owned staged upload, never a Hugging Face source library.
        return Ok(());
    }
    if matches!(job_type, JobType::ModelDownload | JobType::LoraDownload) {
        payload.insert(
            "modelArtifactOperation".to_owned(),
            json!({ "schemaVersion": 1, "kind": "explicit_download" }),
        );
        return Ok(());
    }
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining)
        && payload.get("dryRun").and_then(Value::as_bool) == Some(true)
    {
        // Dry-run planning performs no model I/O and stays available while a library is
        // disconnected. The worker guard applies the identical exemption.
        return Ok(());
    }
    if matches!(job_type, JobType::ModelConvert) {
        let repository = payload
            .get("sourceRepo")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ApiError::bad_request("model_convert requires sourceRepo"))?
            .to_owned();
        // The conversion's source repository, not its output model id, is the immutable runtime
        // input. A cataloged repo carries its full manifest entry; an uncataloged one (user
        // conversions of arbitrary repos are supported) carries a minimal declarative entry built
        // from the request's own data — its install evidence is then whatever receipts the repo
        // actually has, so a disconnect is still typed while a never-downloaded repo stays on the
        // established convert/download path.
        let entry = match resolve_model_manifest_entry_by_repo(state, &repository).await {
            Ok(entry) => entry,
            Err(_) => json!({
                "id": format!("model_convert:{repository}"),
                "downloads": [{ "provider": "huggingface", "repo": repository }],
            }),
        };
        payload.remove("modelManifestEntry");
        payload.insert("modelManifestEntries".to_owned(), json!([entry]));
        return preflight_payload_model_sources(state, payload).await;
    }

    let mut ids = fixed_runtime_model_ids(job_type, payload)
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut repositories = Vec::new();
    let has_primary_carrier = payload.contains_key("modelManifestEntry");
    for reference in payload_reference_spec(job_type) {
        let (key, is_repository) = match reference {
            PayloadModelRef::CatalogId(key) => (*key, false),
            PayloadModelRef::Repository(key) => (*key, true),
        };
        let Some(value) = payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        if is_repository {
            repositories.push(value.to_owned());
        } else if !has_primary_carrier {
            ids.push(value.to_owned());
        }
    }
    ids.sort();
    ids.dedup();
    for model_id in ids {
        if payload_has_model_entry(payload, &model_id) {
            continue;
        }
        let entry = crate::models::resolve_model_manifest_entry(state, &model_id).await?;
        if entry.as_object().map_or(true, JsonObject::is_empty) {
            return Err(ApiError::model_artifact_conflict(
                format!("Runtime model '{model_id}' is not registered in the model catalog."),
                "model_artifact_incomplete",
            ));
        }
        push_model_entry(payload, entry)?;
    }
    for repository in repositories {
        let entry = resolve_model_manifest_entry_by_repo(state, &repository).await?;
        let entry_id = entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if payload_has_model_entry(payload, &entry_id) {
            continue;
        }
        push_model_entry(payload, entry)?;
    }

    preflight_payload_model_sources(state, payload).await
}

/// Resolve the exact catalog model whose supported download names `repository`.
///
/// Utility jobs historically carried provider repository strings rather than catalog ids. They
/// now enter the same typed availability seam without trusting the caller to supply a parallel
/// identity. An unregistered repository fails closed instead of bypassing durable source binding.
pub(crate) async fn resolve_model_manifest_entry_by_repo(
    state: &AppState,
    repository: &str,
) -> Result<Value, ApiError> {
    let declares_repository = |entry: &Value| {
        entry
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|download| {
                sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download(
                    download,
                )
            })
            .any(|download| download.get("repo").and_then(Value::as_str) == Some(repository))
    };
    let mut resolved = crate::models::model_catalog(state)
        .await?
        .into_iter()
        .find(|entry| declares_repository(entry))
        .and_then(|entry| entry.get("id").and_then(Value::as_str).map(str::to_owned));
    if resolved.is_none() {
        resolved = crate::models::embedded_builtin_catalog_entry(declares_repository)?
            .and_then(|entry| entry.get("id").and_then(Value::as_str).map(str::to_owned));
    }
    let model_id = resolved.ok_or_else(|| {
        ApiError::model_artifact_conflict(
            format!(
                "Model source repository '{repository}' is not registered in the model catalog."
            ),
            "model_artifact_incomplete",
        )
    })?;
    let entry = crate::models::resolve_model_manifest_entry(state, &model_id).await?;
    if entry.as_object().map_or(true, JsonObject::is_empty) {
        return Err(ApiError::model_artifact_conflict(
            format!("Runtime model '{model_id}' is not registered in the model catalog."),
            "model_artifact_incomplete",
        ));
    }
    Ok(entry)
}

/// Tells the catalog cache what the resolved cache's published set looks like right now, so a
/// promotion published by the WORKER is reflected in the next catalog read (sc-19712 F-4).
///
/// Reads a handful of `stat`s and no locks — see
/// [`ResolvedCacheStore::published_generation`](sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::published_generation).
/// The catalog cache decides whether the key moved; this only observes it.
pub(crate) fn note_local_tier_freshness(state: &AppState) {
    if !state.settings.resolved_cache.enabled {
        return;
    }
    let key =
        sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::published_generation(
            &state.settings.data_dir,
        );
    state.model_catalog_cache.note_local_tier_key(key);
}

/// App-owned resolved-local artifacts for LocalReady detection. Empty when the resolved cache is
/// disabled or uninitialized; read-only (never creates a session or refreshes usage).
///
/// This is the SAME provider the worker's pre-loader guard uses (sc-19707), so catalog/preflight
/// and the actual load can never disagree about which entries are valid: an entry that is still
/// materializing, torn, unverifiable, or in a shape the runtime cannot serve is not a local
/// candidate on either side of the boundary.
///
/// The agreement is about the *judgement*, not the cost. The scan validates on paths and sizes;
/// the content re-hash lives at the lease boundary inside the store (sc-19712 F-3). Both sides
/// still read the same provider and reach the same verdict for every case a user can reach — a
/// bundle whose bytes were altered after publication is refused when the lease is taken, and the
/// caller falls back to the source tier, which is the same outcome the old cost bought.
pub(crate) fn local_resolved_artifacts(state: &AppState) -> Vec<ResolvedModelArtifact> {
    if !state.settings.resolved_cache.enabled {
        return Vec::new();
    }
    // Rejections are the worker guard's to report: it is the layer that emits the runtime's
    // model-source observability, and the catalog would otherwise re-announce the same entry on
    // every listing.
    sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::valid_local_artifacts(
        &state.settings.data_dir,
    )
    .artifacts
}

/// The one availability judgement for a manifest entry on this host: exact selected closure
/// (host platform, requested variant) through the shared resolver. Sync + blocking FS.
pub(crate) fn availability_for_entry(
    data_dir: &Path,
    entry: &Value,
    requested_variant: Option<&str>,
    local_artifacts: &[ResolvedModelArtifact],
) -> ModelResolution {
    let configured_library = sceneworks_core::hf_home::model_source_library(data_dir)
        .root()
        .to_path_buf();
    let selected =
        selected_requirements_for_model(entry, std::env::consts::OS, requested_variant, data_dir);
    resolve_model_availability(
        data_dir,
        &configured_library,
        &selected.requirements,
        selected.receipt_backed,
        local_artifacts,
    )
}

/// What a local copy of this entry could ever cover on this host, through the same shared
/// selection the availability judgement uses (sc-19712 F-5). `None` for an entry with no external
/// requirement closure at all — an imported or zero-download model is not served through the
/// resolved cache in the first place, so "how much can it cache" has no answer to give, and a
/// reassuring `full` would be an invention.
pub(crate) fn cache_eligibility_for_entry(
    data_dir: &Path,
    entry: &Value,
    requested_variant: Option<&str>,
) -> Option<LocalCacheEligibility> {
    let selected =
        selected_requirements_for_model(entry, std::env::consts::OS, requested_variant, data_dir);
    if selected.requirements.is_empty() {
        return None;
    }
    Some(
        sceneworks_core::model_artifacts::artifact_selection::local_cache_eligibility_for_model(
            entry,
            std::env::consts::OS,
            requested_variant,
            data_dir,
        ),
    )
}

/// Re-judge the externally sourced rows of a (possibly cached) catalog snapshot against the LIVE
/// library state. The snapshot is rebuilt only when its inputs change, but a drive can disconnect
/// or reconnect between rebuilds — the catalog must show the transition immediately.
///
/// This is a READ path, so it is deliberately cheap and write-free: the durable binding is loaded
/// once (shared lock) and its physical identity probed ONCE — never a per-row closure validation,
/// never an exclusive lock, and never a binding/ledger write, so an unreachable network volume
/// cannot stall catalog reads on per-file canonicalization. Only the binary disconnect/reconnect
/// transition flips here (the two states whose stamping already proved install evidence); full
/// closure validation stays where it has authority — snapshot builds, submission preflight, and
/// the worker's pre-loader guard.
pub(crate) async fn refresh_live_external_availability(
    data_dir: &Path,
    models: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let needs_refresh = models.iter().any(|model| {
        model
            .get("modelResolution")
            .is_some_and(|value| !value.is_null())
    });
    if !needs_refresh {
        return Ok(());
    }
    let data_dir = data_dir.to_path_buf();
    let input = std::mem::take(models);
    *models = tokio::task::spawn_blocking(move || {
        refresh_live_external_availability_blocking(&data_dir, input)
    })
    .await
    .map_err(|error| {
        ApiError::internal(format!("external availability probe failed: {error}"))
    })??;
    Ok(())
}

fn refresh_live_external_availability_blocking(
    data_dir: &Path,
    mut models: Vec<Value>,
) -> Result<Vec<Value>, ApiError> {
    use sceneworks_core::model_artifacts::external_library::{
        probe_binding, ExternalLibraryBindingStore, ExternalLibraryProbeStatus,
    };

    let Ok(store) = ExternalLibraryBindingStore::new(data_dir) else {
        return Ok(models);
    };
    let Ok(Some(binding)) = store.load() else {
        // No durable binding: nothing external was ever bound, so there is no live transition to
        // reflect and nothing to probe.
        return Ok(models);
    };
    // ONE physical-identity probe per refresh (read-only). Every stamped row that references the
    // bound library flips on this single answer.
    let library_available = probe_binding(&binding.configured_path, &binding).status
        == ExternalLibraryProbeStatus::Available;
    for model in &mut models {
        let Some(resolution_value) = model
            .get("modelResolution")
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        let Ok(resolution) = serde_json::from_value::<ModelResolution>(resolution_value.clone())
        else {
            continue;
        };
        // Only the two states whose durable evidence is a live question refresh here: an
        // external-ready row can disconnect and an unavailable row can reconnect between
        // snapshot rebuilds. Both proved install evidence when they were stamped.
        if !matches!(
            resolution.availability,
            ModelAvailability::ExternalReady | ModelAvailability::InstalledExternalUnavailable
        ) || resolution.requirements.is_empty()
            || resolution.configured_library_path != binding.configured_path
        {
            continue;
        }
        let live = if library_available {
            ModelResolution::external_ready(
                resolution.configured_library_path.clone(),
                binding.clone(),
                resolution.requirements.clone(),
            )
            .unwrap_or(resolution)
        } else {
            ModelResolution::unavailable(
                resolution.configured_library_path.clone(),
                Some(binding.clone()),
                resolution.requirements.clone(),
            )
            // Present-but-mismatched vs disconnected drives a different prompt (sc-19709), and the
            // catalog READ path has to make the same distinction the resolver does.
            .with_library_present(resolution.configured_library_path.is_dir())
        };
        if let Some(object) = model.as_object_mut() {
            object.insert(
                "modelAvailability".to_owned(),
                serde_json::to_value(&live.availability).map_err(|error| {
                    ApiError::internal(format!("serialize live model availability: {error}"))
                })?,
            );
            object.insert(
                "modelResolution".to_owned(),
                serde_json::to_value(&live).map_err(|error| {
                    ApiError::internal(format!("serialize live model resolution: {error}"))
                })?,
            );
        }
    }
    Ok(models)
}

/// Submission preflight over every carried model entry: reject with the typed 503 only when a
/// model's durable install identity proves it lives on an unavailable external library.
async fn preflight_payload_model_sources(
    state: &AppState,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let requested_variant = requested_runtime_variant(payload);
    let entries = payload_model_entries(payload)
        .into_iter()
        .filter(|entry| {
            entry
                .get("downloads")
                .and_then(Value::as_array)
                .is_some_and(|downloads| !downloads.is_empty())
        })
        .cloned()
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Ok(());
    }
    let data_dir = state.settings.data_dir.clone();
    let local_artifacts = local_resolved_artifacts(state);
    tokio::task::spawn_blocking(move || {
        for entry in &entries {
            let resolution = availability_for_entry(
                &data_dir,
                entry,
                requested_variant.as_deref(),
                &local_artifacts,
            );
            if resolution.availability == ModelAvailability::InstalledExternalUnavailable {
                let model_id = entry
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>");
                let model_name = entry.get("name").and_then(Value::as_str).map(str::to_owned);
                let context = ExternalLibraryUnavailableContext::from_resolution(
                    model_id,
                    model_name,
                    &resolution,
                );
                return Err(ApiError::external_model_library_unavailable(
                    format!(
                        "Model '{model_id}' is installed on an external model library that is \
                         currently unavailable. Reconnect the configured library and retry; the \
                         installation itself is preserved."
                    ),
                    &context,
                ));
            }
        }
        Ok(())
    })
    .await
    .map_err(|error| ApiError::internal(format!("model source preflight failed: {error}")))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::{
        ExternalArtifactRequirement, ExternalLibraryBindingStore,
    };
    use serde_json::json;
    use std::path::PathBuf;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    /// The catalog READ path must reflect disconnect/reconnect from ONE binding probe and must be
    /// write-free: no binding or validated-closure ledger byte may change on a catalog GET.
    #[test]
    fn live_refresh_flips_on_one_probe_and_never_writes_on_the_read_path() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let snapshot = library
            .join("models--owner--model")
            .join("snapshots")
            .join(REVISION);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
        let requirement = ExternalArtifactRequirement {
            repository: "owner/model".to_owned(),
            revision: Some(REVISION.to_owned()),
            variant: "default".to_owned(),
            files: vec![PathBuf::from("model.safetensors")],
            is_primary: true,
        };
        let store = ExternalLibraryBindingStore::new(&data).unwrap();
        let (binding, _) = store
            .bind_or_probe_validated(&library, std::slice::from_ref(&requirement))
            .unwrap();
        let stamped =
            ModelResolution::external_ready(library.clone(), binding, vec![requirement]).unwrap();
        let row = json!({ "id": "m", "modelResolution": stamped });

        let ledger_files = || {
            let models = data.join("models");
            [
                ".sceneworks-external-library-state.json",
                ".sceneworks-external-library-closures.json",
            ]
            .map(|name| std::fs::read(models.join(name)).unwrap())
        };
        let before = ledger_files();

        // Disconnect: the single probe flips the row to the typed unavailable state.
        std::fs::rename(&library, temp.path().join("detached")).unwrap();
        let offline =
            refresh_live_external_availability_blocking(&data, vec![row.clone()]).unwrap();
        assert_eq!(
            offline[0]["modelAvailability"],
            json!("installed_external_unavailable")
        );

        // Reconnect: the same probe flips it back to ready under the original binding.
        std::fs::rename(temp.path().join("detached"), &library).unwrap();
        let online = refresh_live_external_availability_blocking(&data, vec![row]).unwrap();
        assert_eq!(online[0]["modelAvailability"], json!("external_ready"));

        // Write-free: both durable ledgers are byte-identical after the two refreshes.
        assert_eq!(ledger_files(), before);
    }

    /// Rows with no external stamping (missing/incomplete/local/no resolution) pass through the
    /// live refresh untouched — a read must never invent availability.
    #[test]
    fn live_refresh_leaves_unstamped_and_non_external_rows_untouched() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let rows = vec![
            json!({ "id": "plain" }),
            json!({ "id": "null-resolution", "modelResolution": Value::Null }),
        ];
        let refreshed = refresh_live_external_availability_blocking(&data, rows.clone()).unwrap();
        assert_eq!(refreshed, rows);
    }
}
