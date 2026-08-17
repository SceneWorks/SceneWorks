//! Worker-side enforcement for API-stamped external model resolutions.
//!
//! The API contract is advisory-by-identity: this module re-probes the exact configured path,
//! canonical path and physical identity immediately before loader construction. On a later load
//! failure it probes again and changes the error class only when the source is now provably absent
//! or mismatched; an engine failure while the exact source remains available is preserved verbatim.

use crate::{JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::contracts::JobType;
use sceneworks_core::model_artifacts::external_library::{
    ExternalLibraryBindingStore, ExternalLibraryProbeStatus, ExternalSourceSession,
    ModelAvailability, ModelResolution, EXTERNAL_LIBRARY_UNAVAILABLE_CODE,
};

#[derive(Debug)]
pub(crate) struct RuntimeSourceGuard {
    resolutions: Vec<ModelResolution>,
    sessions: Vec<ExternalSourceSession>,
}

impl RuntimeSourceGuard {
    pub(crate) fn begin(
        job_type: &JobType,
        payload: &JsonObject,
        settings: &Settings,
    ) -> WorkerResult<Self> {
        let explicit_download = payload
            .get("modelArtifactOperation")
            .and_then(|value| value.get("kind"))
            .and_then(serde_json::Value::as_str)
            == Some("explicit_download");
        if explicit_download && !matches!(job_type, JobType::ModelDownload | JobType::LoraDownload)
        {
            return Err(WorkerError::InvalidPayload(
                "explicit-download artifact operation is invalid for this job type".to_owned(),
            ));
        }
        let model_free_dry_run = matches!(job_type, JobType::LoraTrain | JobType::ControlTraining)
            && payload.get("dryRun").and_then(serde_json::Value::as_bool) == Some(true);
        if explicit_download || model_free_dry_run {
            return Ok(Self {
                resolutions: Vec::new(),
                sessions: Vec::new(),
            });
        }

        let model_entries = payload_model_entries(payload);
        let mut resolutions = Vec::new();
        let mut uncovered_entries = Vec::new();
        for entry in &model_entries {
            match platform_resolution(entry) {
                Some(value) if !value.is_null() => {
                    let resolution = serde_json::from_value::<ModelResolution>(value.clone())
                        .map_err(|error| {
                            WorkerError::InvalidPayload(format!(
                                "invalid API-stamped model source resolution: {error}"
                            ))
                        })?;
                    if !resolutions.contains(&resolution) {
                        resolutions.push(resolution);
                    }
                }
                _ if entry_is_provably_non_hf_local(entry, settings)? => {}
                _ => uncovered_entries.push(
                    entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>")
                        .to_owned(),
                ),
            }
        }
        if job_requires_typed_model_source(job_type)
            && (model_entries.is_empty() || !uncovered_entries.is_empty())
        {
            return Err(WorkerError::InvalidPayload(format!(
                "{} job has no API-stamped typed model source resolution for: {}",
                job_type.as_str(),
                if model_entries.is_empty() {
                    "<none>".to_owned()
                } else {
                    uncovered_entries.join(", ")
                }
            )));
        }
        let mut sessions = Vec::new();
        for resolution in &resolutions {
            resolution.validate().map_err(|error| {
                WorkerError::InvalidPayload(format!("invalid model source resolution: {error}"))
            })?;
            match resolution.availability {
                ModelAvailability::LocalReady => {}
                ModelAvailability::InstalledExternalUnavailable => {
                    return Err(unavailable(
                        "The configured external model library is unavailable. Reconnect it and retry.",
                    ));
                }
                ModelAvailability::Incomplete | ModelAvailability::Missing => {
                    return Err(WorkerError::InvalidPayload(
                        "model source resolution is not ready for runtime use".to_owned(),
                    ));
                }
                ModelAvailability::ExternalReady => {
                    let store = ExternalLibraryBindingStore::new(&settings.data_dir)
                        .map_err(|error| unavailable(error.to_string()))?;
                    let probe = match store.probe_resolution(resolution) {
                        Ok(probe) => probe,
                        Err(error) => {
                            let binding =
                                resolution.expected_library.as_ref().ok_or_else(|| {
                                    WorkerError::InvalidPayload(
                                        "external-ready resolution has no source binding"
                                            .to_owned(),
                                    )
                                })?;
                            let identity =
                                store.probe_bound(&resolution.configured_library_path, binding);
                            if identity.status == ExternalLibraryProbeStatus::Available {
                                return Err(WorkerError::InvalidPayload(format!(
                                    "model source components are incomplete: {error}"
                                )));
                            }
                            return Err(unavailable(
                                "The configured external model library is disconnected or no longer has the expected physical identity. Reconnect it and retry."
                            ));
                        }
                    };
                    if probe.status != ExternalLibraryProbeStatus::Available {
                        return Err(unavailable(
                            "The configured external model library is disconnected or no longer has the expected physical identity. Reconnect it and retry."
                        ));
                    }
                    match ExternalSourceSession::begin(&settings.data_dir, resolution) {
                        Ok(session) => sessions.push(session),
                        Err(error) => {
                            let binding =
                                resolution.expected_library.as_ref().ok_or_else(|| {
                                    WorkerError::InvalidPayload(
                                        "external-ready resolution has no source binding"
                                            .to_owned(),
                                    )
                                })?;
                            let identity =
                                store.probe_bound(&resolution.configured_library_path, binding);
                            if identity.status == ExternalLibraryProbeStatus::Available {
                                return Err(WorkerError::InvalidPayload(format!(
                                    "model source session could not validate all components: {error}"
                                )));
                            }
                            return Err(unavailable(
                                "The external model library disconnected while the source session was starting. Reconnect it and retry."
                            ));
                        }
                    }
                }
            }
        }
        Ok(Self {
            resolutions,
            sessions,
        })
    }

    pub(crate) fn finish_success(mut self) -> WorkerResult<()> {
        for session in self.sessions.drain(..) {
            session
                .mark_success()
                .map_err(|error| WorkerError::Io(std::io::Error::other(error.to_string())))?;
        }
        Ok(())
    }

    pub(crate) fn classify_failure(
        &self,
        settings: &Settings,
        original: WorkerError,
    ) -> WorkerError {
        let store = match ExternalLibraryBindingStore::new(&settings.data_dir) {
            Ok(store) => store,
            Err(_) => return original,
        };
        let source_became_unavailable = self.resolutions.iter().any(|resolution| {
            resolution.availability == ModelAvailability::ExternalReady
                && store
                    .probe_resolution(resolution)
                    .map(|probe| probe_proves_source_unavailable(&probe.status))
                    .unwrap_or(false)
        });
        if source_became_unavailable {
            unavailable(format!(
                "The external model library disconnected or changed during model use; reconnect it and retry. Original load failure was suppressed ({EXTERNAL_LIBRARY_UNAVAILABLE_CODE})."
            ))
        } else {
            original
        }
    }
}

fn probe_proves_source_unavailable(status: &ExternalLibraryProbeStatus) -> bool {
    matches!(
        status,
        ExternalLibraryProbeStatus::Unavailable | ExternalLibraryProbeStatus::IdentityMismatch
    )
}

/// Exhaustive allowlist: only routes that cannot open model/HF bytes may omit a carrier. Keeping
/// this as a total match means a newly added job type fails compilation until it is classified.
fn job_requires_typed_model_source(job_type: &JobType) -> bool {
    match job_type {
        JobType::Placeholder
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::DatasetParquetImport
        | JobType::ModelImport
        | JobType::LoraImport => false,
        JobType::ImageGenerate
        | JobType::ImageEdit
        | JobType::ImageVqa
        | JobType::ImageInterleave
        | JobType::VideoGenerate
        | JobType::VideoExtend
        | JobType::VideoBridge
        | JobType::PersonDetect
        | JobType::PersonTrack
        | JobType::PersonReplace
        | JobType::AudioGenerate
        | JobType::PoseDetect
        | JobType::KpsExtract
        | JobType::ImageUpscale
        | JobType::ImageDetail
        | JobType::ImageSegment
        | JobType::VideoUpscale
        | JobType::ModelDownload
        | JobType::ModelConvert
        | JobType::LoraDownload
        | JobType::LoraTrain
        | JobType::ControlTraining
        | JobType::TrainingCaption
        | JobType::DatasetAnalysis
        | JobType::CatalogAnalysis
        | JobType::DatasetUpscale
        | JobType::DatasetFaceAnalysis
        | JobType::FaceLikenessCompare
        | JobType::PromptRefine
        | JobType::Unknown(_) => true,
        _ => true,
    }
}

fn payload_model_entries(payload: &JsonObject) -> Vec<&serde_json::Value> {
    ["modelManifestEntry", "baseModelManifestEntry"]
        .into_iter()
        .filter_map(|key| payload.get(key))
        .chain(
            payload
                .get("modelManifestEntries")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten(),
        )
        .collect()
}

/// Imported/app-owned and explicitly configured ComfyUI models are the only model entries that
/// may omit the HF resolution contract. They must have no download descriptors at all and every
/// concrete path they expose must pass the worker's existing app-managed confinement check. This
/// is deliberately per-entry: one valid primary carrier cannot hide an unstamped auxiliary.
fn entry_is_provably_non_hf_local(
    entry: &serde_json::Value,
    settings: &Settings,
) -> WorkerResult<bool> {
    if entry
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|downloads| !downloads.is_empty())
    {
        return Ok(false);
    }
    let mut paths = Vec::new();
    for key in ["modelPath", "installedPath"] {
        if let Some(path) = entry.get(key).and_then(serde_json::Value::as_str) {
            paths.push(path);
        }
    }
    for object_key in ["paths", "source"] {
        paths.extend(
            entry
                .get(object_key)
                .and_then(serde_json::Value::as_object)
                .into_iter()
                .flat_map(|object| object.values())
                .filter_map(serde_json::Value::as_str),
        );
    }
    paths.extend(
        entry
            .get("components")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|component| component.get("path"))
            .filter_map(serde_json::Value::as_str),
    );
    if paths.is_empty() {
        return Ok(false);
    }
    for path in paths {
        crate::paths::normalize_app_managed_model_path(settings, path, "non-HF model source")?;
    }
    Ok(true)
}

fn platform_resolution(entry: &serde_json::Value) -> Option<&serde_json::Value> {
    entry
        .get("modelResolutionsByPlatform")
        .and_then(|resolutions| resolutions.get(std::env::consts::OS))
        .or_else(|| entry.get("modelResolution"))
}

fn unavailable(detail: impl Into<String>) -> WorkerError {
    WorkerError::ExternalLibraryUnavailable(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::{
        ExternalArtifactRequirement, ExternalLibraryBindingStore,
    };
    use serde_json::json;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn settings(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir,
            config_dir: PathBuf::new(),
            worker_id: "worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: "https://huggingface.co".to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: 1,
            max_model_url_bytes: 1,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: false,
            backend_candle_enabled: false,
            external_model_roots: Vec::new(),
            gpu_memory_limit_bytes: 0,
        }
    }

    fn resolution(temp: &TempDir) -> (Settings, PathBuf, ModelResolution) {
        let data = temp.path().join("data");
        let library = temp.path().join("hf");
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
        let resolution =
            ModelResolution::external_ready(library.clone(), binding, vec![requirement]).unwrap();
        (settings(data), library, resolution)
    }

    fn payload(resolution: &ModelResolution) -> JsonObject {
        json!({ "modelManifestEntry": { "modelResolution": resolution } })
            .as_object()
            .unwrap()
            .clone()
    }

    #[test]
    fn worker_reprobes_before_load_and_disconnect_is_typed() {
        let temp = TempDir::new().unwrap();
        let (settings, library, resolution) = resolution(&temp);
        std::fs::rename(&library, temp.path().join("detached")).unwrap();
        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&resolution), &settings)
                .unwrap_err();
        assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
        assert!(error
            .to_string()
            .contains(EXTERNAL_LIBRARY_UNAVAILABLE_CODE));
    }

    #[test]
    fn worker_rejects_an_advisory_identity_that_differs_from_the_durable_binding() {
        let temp = TempDir::new().unwrap();
        let (settings, _, mut resolution) = resolution(&temp);
        resolution
            .expected_library
            .as_mut()
            .unwrap()
            .physical_identity
            .directory_id ^= 1;

        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&resolution), &settings)
                .unwrap_err();
        assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
    }

    #[test]
    fn worker_keeps_connected_component_loss_distinct_from_library_unavailability() {
        let temp = TempDir::new().unwrap();
        let (settings, library, resolution) = resolution(&temp);
        std::fs::remove_file(
            library
                .join("models--owner--model")
                .join("snapshots")
                .join(REVISION)
                .join("model.safetensors"),
        )
        .unwrap();
        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&resolution), &settings)
                .unwrap_err();
        assert!(matches!(&error, WorkerError::InvalidPayload(_)));
        assert!(!matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
    }

    #[test]
    fn terminal_remap_occurs_only_when_exact_source_probe_proves_disconnect() {
        let temp = TempDir::new().unwrap();
        let (settings, library, resolution) = resolution(&temp);
        let guard =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&resolution), &settings)
                .unwrap();
        let preserved = guard.classify_failure(
            &settings,
            WorkerError::Engine("real loader defect".to_owned()),
        );
        assert!(
            matches!(preserved, WorkerError::Engine(ref value) if value == "real loader defect")
        );

        std::fs::rename(&library, temp.path().join("detached")).unwrap();
        let remapped = guard.classify_failure(
            &settings,
            WorkerError::Engine("raw ENOENT from loader".to_owned()),
        );
        assert!(matches!(
            remapped,
            WorkerError::ExternalLibraryUnavailable(_)
        ));
        assert!(!remapped.to_string().contains("raw ENOENT"));
        drop(guard);
        let sessions = settings
            .data_dir
            .join("models/.sceneworks-external-source-sessions");
        assert_eq!(
            std::fs::read_dir(sessions).unwrap().count(),
            0,
            "failure cleanup removes only the operation-owned source session"
        );
    }

    #[test]
    fn scalar_and_multi_model_carriers_are_parsed_and_deduplicated() {
        let temp = TempDir::new().unwrap();
        let (settings, _, resolution) = resolution(&temp);
        let payload = json!({
            "modelManifestEntry": { "modelResolution": resolution },
            "modelManifestEntries": [
                { "modelResolution": resolution },
                { "modelResolution": resolution }
            ]
        })
        .as_object()
        .unwrap()
        .clone();
        let guard =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        assert_eq!(guard.resolutions, vec![resolution]);
    }

    #[test]
    fn model_routes_fail_closed_without_a_carrier_and_exact_no_model_routes_remain_allowed() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let empty = JsonObject::new();
        let error =
            RuntimeSourceGuard::begin(&JobType::DatasetUpscale, &empty, &settings).unwrap_err();
        assert!(matches!(error, WorkerError::InvalidPayload(_)));
        RuntimeSourceGuard::begin(&JobType::FrameExtract, &empty, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::DatasetParquetImport, &empty, &settings).unwrap();
    }

    #[test]
    fn explicit_download_is_not_mistaken_for_a_ready_runtime_source() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let missing = ModelResolution::not_ready(
            ModelAvailability::Missing,
            temp.path().join("hf"),
            Vec::new(),
        )
        .unwrap();
        let payload = json!({
            "modelArtifactOperation": { "schemaVersion": 1, "kind": "explicit_download" },
            "modelManifestEntry": { "modelResolution": missing }
        })
        .as_object()
        .unwrap()
        .clone();
        RuntimeSourceGuard::begin(&JobType::ModelDownload, &payload, &settings).unwrap();
        assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).is_err());
    }

    #[test]
    fn only_proven_disconnect_or_identity_mismatch_suppresses_the_loader_error() {
        assert!(!probe_proves_source_unavailable(
            &ExternalLibraryProbeStatus::Available
        ));
        assert!(!probe_proves_source_unavailable(
            &ExternalLibraryProbeStatus::Unknown
        ));
        assert!(probe_proves_source_unavailable(
            &ExternalLibraryProbeStatus::Unavailable
        ));
        assert!(probe_proves_source_unavailable(
            &ExternalLibraryProbeStatus::IdentityMismatch
        ));
    }

    #[test]
    fn worker_selects_its_own_platform_resolution_not_the_api_host_scalar() {
        let temp = TempDir::new().unwrap();
        let (settings, _, ready) = resolution(&temp);
        let mut host = ready.clone();
        host.availability = ModelAvailability::Missing;
        let payload = json!({
            "modelManifestEntry": {
                "modelResolution": host,
                "modelResolutionsByPlatform": { (std::env::consts::OS): ready }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let guard =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        assert_eq!(guard.resolutions, vec![ready]);
    }

    #[test]
    fn one_valid_primary_carrier_cannot_hide_an_unstamped_hf_auxiliary() {
        let temp = TempDir::new().unwrap();
        let (settings, _, ready) = resolution(&temp);
        let payload = json!({
            "modelManifestEntry": { "id": "primary", "modelResolution": ready },
            "modelManifestEntries": [{
                "id": "unstamped-auxiliary",
                "downloads": [{ "provider": "huggingface", "repo": "owner/auxiliary" }]
            }]
        })
        .as_object()
        .unwrap()
        .clone();
        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap_err();
        assert!(error.to_string().contains("unstamped-auxiliary"));
    }

    #[test]
    fn only_confined_download_free_local_entries_may_use_the_non_hf_exception() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let imported = settings.data_dir.join("models/imports/custom");
        std::fs::create_dir_all(&imported).unwrap();
        std::fs::write(imported.join("model.safetensors"), b"weights").unwrap();
        let local = json!({
            "modelManifestEntry": {
                "id": "imported",
                "downloads": [],
                "paths": { "model": imported }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        RuntimeSourceGuard::begin(&JobType::ImageGenerate, &local, &settings).unwrap();

        let outside = temp.path().join("outside/model.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"outside").unwrap();
        let escaped = json!({
            "modelManifestEntry": {
                "id": "forged-local",
                "downloads": [],
                "paths": { "model": outside }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &escaped, &settings).is_err());
    }

    #[test]
    fn unavailable_worker_platform_corequisites_block_even_when_host_scalar_is_ready() {
        let temp = TempDir::new().unwrap();
        let (settings, _, ready) = resolution(&temp);
        let unavailable = ModelResolution::unavailable(
            ready.configured_library_path.clone(),
            ready.expected_library.clone(),
            ready.requirements.clone(),
        );
        let payload = json!({
            "modelManifestEntry": {
                "modelResolution": ready,
                "modelResolutionsByPlatform": { (std::env::consts::OS): unavailable }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap_err();
        assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
    }

    #[test]
    fn worker_fails_closed_for_every_non_ready_typed_state() {
        let temp = TempDir::new().unwrap();
        let (settings, _, resolution) = resolution(&temp);

        let unavailable = ModelResolution::unavailable(
            resolution.configured_library_path.clone(),
            resolution.expected_library.clone(),
            resolution.requirements.clone(),
        );
        let error =
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&unavailable), &settings)
                .unwrap_err();
        assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));

        for availability in [ModelAvailability::Incomplete, ModelAvailability::Missing] {
            let mut not_ready = resolution.clone();
            not_ready.availability = availability;
            let error =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload(&not_ready), &settings)
                    .unwrap_err();
            assert!(matches!(error, WorkerError::InvalidPayload(_)));
        }
    }
}
