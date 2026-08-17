//! The worker's single pre-loader model-source guard.
//!
//! Every job dispatched by `run_utility_job` passes through [`RuntimeSourceGuard::begin`] before
//! its handler constructs a loader. The guard reads the job's generic model carriers
//! (`modelManifestEntry`, `baseModelManifestEntry`, `modelManifestEntries` — declarative manifest
//! data, never per-model wiring), reduces each to the exact requirement closure for THIS worker's
//! platform and the request's selected tier via
//! [`sceneworks_core::model_artifacts::artifact_selection`], and judges it through the one shared
//! resolver. There is deliberately no per-route or per-model availability code anywhere else in
//! this crate:
//!
//! - `local_ready` / `external_ready` proceed (an external source opens an operation-owned
//!   session whose physical identity was proven immediately before loader construction);
//! - `installed_external_unavailable` fails typed — installed state is preserved, receipts are
//!   never rewritten, and no silent re-download is attempted;
//! - `incomplete` / `missing` preserve the established on-demand download behavior of the
//!   handler (a model that was never installed keeps its existing install path).
//!
//! On a later load failure the guard re-probes the exact bound source and remaps the error to the
//! typed unavailable class only when the source is now provably absent or a different physical
//! volume — an engine defect while the source remains available is preserved verbatim, and a raw
//! loader ENOENT is never what a disconnect surfaces as.

use crate::{JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::contracts::JobType;
use sceneworks_core::model_artifacts::artifact_selection::{
    requested_runtime_variant, selected_requirements_for_model,
};
use sceneworks_core::model_artifacts::external_library::{
    resolve_model_availability, ExternalSourceSession, ModelAvailability, ModelResolution,
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
        // Fail closed on an EMPTY model carrier set for every model-backed route. Routes that
        // genuinely load no model bytes are declared in the exhaustive table below — a job type
        // absent from that allowlist cannot pass without carriers.
        if job_requires_typed_model_source(job_type) && model_entries.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "{} job carries no model manifest entries for the typed model-source guard",
                job_type.as_str()
            )));
        }

        let requested_variant = requested_runtime_variant(payload);
        let configured_library =
            sceneworks_core::hf_home::model_source_library(&settings.data_dir)
                .root()
                .to_path_buf();
        let mut resolutions = Vec::new();
        for entry in &model_entries {
            let has_hf_downloads = entry
                .get("downloads")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|downloads| {
                    downloads.iter().any(|download| {
                        sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download(
                            download,
                        )
                    })
                });
            if !has_hf_downloads {
                // Imported/app-owned and explicitly configured external-root models are the only
                // entries outside the HF-library contract; they must prove their confinement.
                if entry_is_provably_non_hf_local(entry, settings)? {
                    continue;
                }
                return Err(WorkerError::InvalidPayload(format!(
                    "model entry '{}' names no supported artifact source and no confined local path",
                    entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>")
                )));
            }
            // This worker's OWN platform and the request's selected tier define the closure —
            // never the API host's platform, never a sibling variant's receipts.
            let requirements = selected_requirements_for_model(
                entry,
                std::env::consts::OS,
                requested_variant.as_deref(),
                &settings.data_dir,
            );
            let resolution = resolve_model_availability(
                &settings.data_dir,
                &configured_library,
                &requirements,
                // Local-tier artifacts enter this seam in sc-19707; the resolver already accepts
                // them, so wiring local preference will not touch this call site's shape.
                &[],
            );
            if !resolutions.contains(&resolution) {
                resolutions.push(resolution);
            }
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
                        "The configured external model library holding this installed model is \
                         disconnected or no longer has the expected physical identity. Reconnect \
                         it and retry; the installation itself is preserved.",
                    ));
                }
                // A model that was never installed (or whose install is incomplete while the
                // library is provably present) keeps its established on-demand install path.
                ModelAvailability::Incomplete | ModelAvailability::Missing => {}
                ModelAvailability::ExternalReady => {
                    match ExternalSourceSession::begin(&settings.data_dir, resolution) {
                        Ok(session) => sessions.push(session),
                        Err(error) => {
                            // The session probe re-proves path + physical identity and re-walks
                            // the closure. Distinguish "components vanished while the library
                            // stayed provably present" from a disconnect mid-admission.
                            let recheck = resolve_model_availability(
                                &settings.data_dir,
                                &configured_library,
                                &resolution.requirements,
                                &[],
                            );
                            if recheck.availability
                                == ModelAvailability::InstalledExternalUnavailable
                            {
                                return Err(unavailable(
                                    "The external model library disconnected while the model \
                                     source was being admitted. Reconnect it and retry.",
                                ));
                            }
                            return Err(WorkerError::InvalidPayload(format!(
                                "model source components could not be validated: {error}"
                            )));
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

    /// Mid-load failure classification: re-probe every externally sourced model and suppress the
    /// raw loader error ONLY when the exact bound source is now provably unavailable. A load
    /// failure with the source still present is a real defect and is preserved verbatim.
    pub(crate) fn classify_failure(
        &self,
        settings: &Settings,
        original: WorkerError,
    ) -> WorkerError {
        let configured_library =
            sceneworks_core::hf_home::model_source_library(&settings.data_dir)
                .root()
                .to_path_buf();
        let source_became_unavailable = self.resolutions.iter().any(|resolution| {
            resolution.availability == ModelAvailability::ExternalReady
                && resolve_model_availability(
                    &settings.data_dir,
                    &configured_library,
                    &resolution.requirements,
                    &[],
                )
                .availability
                    == ModelAvailability::InstalledExternalUnavailable
        });
        if source_became_unavailable {
            unavailable(
                "The external model library disconnected or changed during model use; reconnect \
                 it and retry. The original load failure was suppressed because the model's \
                 source is provably unavailable.",
            )
        } else {
            original
        }
    }

    #[cfg(test)]
    pub(crate) fn resolutions(&self) -> &[ModelResolution] {
        &self.resolutions
    }
}

/// Exhaustive allowlist AT THE SEAM: only routes that cannot open model/HF bytes may run without
/// model carriers. A total match means a newly added job type fails compilation until it is
/// classified here — an unclassified HF-backed route can never pass silently.
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
        .filter(|entry| !entry.is_null())
        .collect()
}

/// Imported/app-owned and explicitly configured external-root models are the only model entries
/// that may omit the HF artifact contract. They must have no download descriptors at all and
/// every concrete path they expose must pass the worker's existing app-managed confinement
/// check. Deliberately per-entry: one valid primary carrier cannot hide an unconfined auxiliary.
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

fn unavailable(detail: impl Into<String>) -> WorkerError {
    WorkerError::ExternalLibraryUnavailable(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE;
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    const REV_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const REV_B: &str = "89abcdef0123456789abcdef0123456789abcdef";

    fn settings(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: crate::test_env::OFFLINE_URL.to_owned(),
            access_token: None,
            data_dir,
            config_dir: PathBuf::new(),
            worker_id: "worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: crate::test_env::OFFLINE_URL.to_owned(),
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

    /// Pin the configured source library to an explicit external dir for the whole body.
    /// `huggingface_hub_cache_dir` reads `HF_HUB_CACHE` before `data_dir`, and other tests in
    /// this crate mutate those vars — the shared `test_env` lock serializes against them, and the
    /// explicit dir models the external-drive scenario directly.
    fn with_library<T>(library: &Path, body: impl FnOnce() -> T) -> T {
        crate::test_env::temp_env_vars(
            &[("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path"))],
            body,
        )
    }

    fn seed_snapshot(library: &Path, repo: &str, revision: &str, file: &str) {
        let safe = sceneworks_core::hf_home::safe_repo_dir_name(repo).unwrap();
        let snapshot = library
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        if let Some(parent) = snapshot.join(file).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(snapshot.join(file), b"weights").unwrap();
    }

    fn write_receipts(data_dir: &Path, repo: &str, receipts: Value) {
        let managed = data_dir.join("models").join(
            sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(repo),
        );
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
        )
        .unwrap();
    }

    /// One installed single-variant model: manifest entry + receipt + library snapshot.
    fn installed_model(temp: &TempDir) -> (Settings, PathBuf, JsonObject) {
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "owner/model", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "owner/model",
            json!([{ "repo": "owner/model", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        let payload = json!({
            "model": "m",
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model",
                                "revision": REV_A, "files": ["model.safetensors"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        (settings(data), library, payload)
    }

    #[test]
    fn installed_model_admits_as_external_ready_with_a_source_session() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 1);
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            guard.finish_success().unwrap();
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(std::fs::read_dir(sessions).unwrap().count(), 0);
        });
    }

    #[test]
    fn disconnected_library_fails_typed_and_receipts_survive_untouched() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let receipt_path = settings.data_dir.join("models").join(
                sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(
                    "owner/model",
                ),
            );
            let receipt_before =
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap();
            // Bind first while connected, then disconnect the volume.
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let error = RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap_err();
            assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
            assert!(error.to_string().contains(EXTERNAL_LIBRARY_UNAVAILABLE_CODE));
            // Installed state is never lost: receipts are byte-identical after the disconnect.
            assert_eq!(
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap(),
                receipt_before
            );

            // Reconnect: the same payload admits again under the original binding.
            std::fs::rename(temp.path().join("detached"), &library).unwrap();
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        });
    }

    #[test]
    fn component_removal_while_connected_is_incomplete_never_the_disconnect_class() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            let safe = sceneworks_core::hf_home::safe_repo_dir_name("owner/model").unwrap();
            std::fs::remove_file(
                library
                    .join(format!("models--{safe}"))
                    .join("snapshots")
                    .join(REV_A)
                    .join("model.safetensors"),
            )
            .unwrap();
            // The library is provably present; the missing component makes the install stale, and
            // stale keeps its established on-demand install/repair semantics — it must NOT read
            // as a library disconnect.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn model_routes_fail_closed_without_carriers_and_no_model_routes_pass() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        with_library(&library, || {
            let empty = JsonObject::new();
            for model_backed in [
                JobType::ImageGenerate,
                JobType::DatasetUpscale,
                JobType::PromptRefine,
                JobType::CatalogAnalysis,
            ] {
                let error =
                    RuntimeSourceGuard::begin(&model_backed, &empty, &settings).unwrap_err();
                assert!(
                    matches!(error, WorkerError::InvalidPayload(_)),
                    "{model_backed:?} must fail closed on an empty carrier set"
                );
            }
            for model_free in [
                JobType::Placeholder,
                JobType::FrameExtract,
                JobType::TimelineExport,
                JobType::DatasetParquetImport,
            ] {
                RuntimeSourceGuard::begin(&model_free, &empty, &settings).unwrap();
            }
        });
    }

    #[test]
    fn completeness_is_judged_on_the_selected_tier_not_a_variant_union() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        // q4 fully installed; q8 receipt exists but its library snapshot was pruned.
        seed_snapshot(&library, "owner/matrix", REV_A, "q4/model.safetensors");
        write_receipts(
            &data,
            "owner/matrix",
            json!([
                { "repo": "owner/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "owner/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_B }
            ]),
        );
        let settings = settings(data);
        let entry = json!({
            "id": "m",
            "downloads": [
                { "provider": "huggingface", "repo": "owner/matrix", "variant": "q4",
                  "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "owner/matrix", "variant": "q8",
                  "files": ["q8/*"] }
            ]
        });
        let payload = |variant: &str| {
            json!({ "modelManifestEntry": entry, "variant": variant })
                .as_object()
                .unwrap()
                .clone()
        };
        with_library(&library, || {
            // The selected q4 tier is complete even though the q8 sibling receipt cannot
            // validate — a model-wide union of variant receipts would wrongly demote this to
            // incomplete.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert_eq!(guard.resolutions()[0].requirements.len(), 1);
            assert_eq!(guard.resolutions()[0].requirements[0].variant, "q4");
            drop(guard);
            // And the pruned q8 selection reads incomplete — the q4 install must not mask it.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn requirements_use_this_workers_platform_closure_not_another_hosts() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let this_os = std::env::consts::OS;
        let other_os = if this_os == "macos" { "windows" } else { "macos" };
        seed_snapshot(&library, "owner/native", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "owner/native",
            json!([{ "repo": "owner/native", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        // The other platform's primary and co-requisite are NOT installed anywhere. If the guard
        // used another host's platform closure (or no platform filter), admission would fail.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [
                    { "provider": "huggingface", "repo": "owner/native", "revision": REV_A,
                      "files": ["model.safetensors"], "platforms": [this_os] },
                    { "provider": "huggingface", "repo": "owner/foreign", "revision": REV_B,
                      "files": ["foreign.safetensors"], "platforms": [other_os] },
                    { "provider": "huggingface", "repo": "owner/foreign-corequisite",
                      "revision": REV_B, "coRequisite": true,
                      "files": ["encoder.safetensors"], "platforms": [other_os] }
                ]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let settings = settings(data);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            let repositories = guard.resolutions()[0]
                .requirements
                .iter()
                .map(|requirement| requirement.repository.as_str())
                .collect::<Vec<_>>();
            assert_eq!(repositories, ["owner/native"]);
        });
    }

    #[test]
    fn terminal_remap_occurs_only_when_the_exact_source_probe_proves_disconnect() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let preserved = guard.classify_failure(
                &settings,
                WorkerError::Engine("real loader defect".to_owned()),
            );
            assert!(
                matches!(preserved, WorkerError::Engine(ref value) if value == "real loader defect"),
                "an engine failure with the source present must be preserved verbatim"
            );

            std::fs::rename(&library, temp.path().join("detached")).unwrap();
            let remapped = guard.classify_failure(
                &settings,
                WorkerError::Engine("No such file or directory (os error 2)".to_owned()),
            );
            assert!(matches!(remapped, WorkerError::ExternalLibraryUnavailable(_)));
            assert!(
                !remapped.to_string().contains("os error 2"),
                "a proven disconnect must never surface as a raw ENOENT"
            );
            drop(guard);
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(
                std::fs::read_dir(sessions).unwrap().count(),
                0,
                "failure cleanup removes only the operation-owned source session"
            );
        });
    }

    #[test]
    fn never_installed_model_keeps_its_on_demand_install_path() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        std::fs::create_dir_all(&library).unwrap();
        let settings = settings(data);
        // Manifest present, but no receipt and glob files (no exact declaration): no durable
        // install identity, so availability is Missing and the job proceeds to its established
        // download path rather than failing typed.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "owner/new",
                                "revision": REV_A, "files": ["q4/*"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Missing
            );
        });
    }

    #[test]
    fn explicit_download_is_exempt_only_for_download_job_types() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let payload = json!({
            "modelArtifactOperation": { "schemaVersion": 1, "kind": "explicit_download" }
        })
        .as_object()
        .unwrap()
        .clone();
        RuntimeSourceGuard::begin(&JobType::ModelDownload, &payload, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::LoraDownload, &payload, &settings).unwrap();
        assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).is_err());
    }

    #[test]
    fn only_confined_download_free_local_entries_may_use_the_non_hf_exception() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
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
        let bare = json!({ "modelManifestEntry": { "id": "bare" } })
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &local, &settings).unwrap();
            assert!(
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &escaped, &settings).is_err()
            );
            // No downloads AND no paths proves nothing: fail closed.
            assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &bare, &settings).is_err());
        });
    }

    #[test]
    fn training_dry_run_is_model_free_but_a_real_run_is_not() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let dry = json!({ "dryRun": true }).as_object().unwrap().clone();
        RuntimeSourceGuard::begin(&JobType::LoraTrain, &dry, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::ControlTraining, &dry, &settings).unwrap();
        let real = JsonObject::new();
        assert!(RuntimeSourceGuard::begin(&JobType::LoraTrain, &real, &settings).is_err());
    }
}

