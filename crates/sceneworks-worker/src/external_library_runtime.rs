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

use crate::{emit_event_value, JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::contracts::JobType;
use sceneworks_core::model_artifacts::artifact_selection::{
    requested_runtime_variant, selected_requirements_for_model,
};
use sceneworks_core::model_artifacts::external_library::{
    resolve_model_availability, ExternalSourceSession, ModelAvailability, ModelResolution,
};
use sceneworks_core::model_artifacts::local_preference::ActiveLocalArtifacts;
use sceneworks_core::model_artifacts::resolved_cache::{ResolvedCacheLease, ResolvedCacheStore};
use sceneworks_core::model_artifacts::{ModelArtifactResolver, ResolvedModelArtifact};
use serde_json::json;
use tracing::Level;

#[derive(Debug)]
pub(crate) struct RuntimeSourceGuard {
    resolutions: Vec<ModelResolution>,
    sessions: Vec<ExternalSourceSession>,
    /// The process-wide local-tier preference scopes, one per locally served artifact. Declared
    /// BEFORE the leases so field drop order stops serving local paths first and only then
    /// releases the leases protecting those bytes — the reverse of acquisition.
    local_scopes: Vec<ActiveLocalArtifacts>,
    /// Cross-process cache leases (one per locally served artifact) held for the whole job. Each
    /// holds the entry's shared artifact lock, so an evictor taking that lock exclusively — even
    /// non-blockingly, and even when it re-verifies under the lock — cannot remove bytes a load is
    /// reading. Released on drop; `finish_success` crosses the promotion boundary first.
    cache_leases: Vec<ResolvedCacheLease>,
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
            return Ok(Self::none());
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
        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        // Every VALID app-owned bundle, read once for this job. Empty when the resolved cache is
        // disabled or uninitialized, and — by construction of the provider — never containing an
        // entry that is torn, unverifiable, or in a shape the runtime cannot serve, so a partial
        // local copy can only ever move the load back to the source tier.
        let local_artifacts = local_resolved_artifacts(settings);
        let mut resolutions = Vec::new();
        let mut cache_leases = Vec::new();
        let mut local_scopes = Vec::new();
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
            let selected = selected_requirements_for_model(
                entry,
                std::env::consts::OS,
                requested_variant.as_deref(),
                &settings.data_dir,
            );
            let mut resolution = resolve_model_availability(
                &settings.data_dir,
                &configured_library,
                &selected.requirements,
                selected.receipt_backed,
                &local_artifacts,
            );
            // Local preference is only real once the artifact is LEASED and actually reachable
            // through the shared snapshot resolvers. Anything that prevents either — a vanished or
            // concurrently evicted entry, a bundle another scope already serves — falls the whole
            // model back to the authoritative source tier by RE-RESOLVING without local
            // candidates, so the fallback keeps every typed disconnect/incomplete guarantee
            // instead of proceeding on a promise the runtime cannot keep.
            if resolution.availability == ModelAvailability::LocalReady {
                match admit_local_artifact(settings, resolution.local_artifact.as_ref()) {
                    Some((lease, scope)) => {
                        cache_leases.push(lease);
                        local_scopes.push(scope);
                    }
                    None => {
                        resolution = resolve_model_availability(
                            &settings.data_dir,
                            &configured_library,
                            &selected.requirements,
                            selected.receipt_backed,
                            &[],
                        );
                    }
                }
            }
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
                // Already leased and served from the app-owned tier above; the configured source
                // library is deliberately not consulted, opened, or probed for it.
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
                            // The requirements came from an external-ready resolution, so the
                            // install evidence is already proven durable.
                            let recheck = resolve_model_availability(
                                &settings.data_dir,
                                &configured_library,
                                &resolution.requirements,
                                true,
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
        emit_source_tiers(&resolutions);
        Ok(Self {
            resolutions,
            sessions,
            local_scopes,
            cache_leases,
        })
    }

    fn none() -> Self {
        Self {
            resolutions: Vec::new(),
            sessions: Vec::new(),
            local_scopes: Vec::new(),
            cache_leases: Vec::new(),
        }
    }

    pub(crate) fn finish_success(mut self) -> WorkerResult<()> {
        for session in self.sessions.drain(..) {
            session
                .mark_success()
                .map_err(|error| WorkerError::Io(std::io::Error::other(error.to_string())))?;
        }
        // Stop serving local paths before the leases protecting them are released.
        self.local_scopes.clear();
        for lease in self.cache_leases.drain(..) {
            // The promotion boundary for an artifact that just served a real load. Its own
            // entry is already published, so this only records the successful use.
            let _candidate = lease.mark_success();
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
        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        let source_became_unavailable = self.resolutions.iter().any(|resolution| {
            resolution.availability == ModelAvailability::ExternalReady
                && resolve_model_availability(
                    &settings.data_dir,
                    &configured_library,
                    &resolution.requirements,
                    true,
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

/// Every VALID app-owned resolved artifact on this host, or nothing when the resolved cache is
/// switched off. Read-only: enumeration never creates a cache session and never stamps usage, so
/// judging availability cannot look like using a model.
fn local_resolved_artifacts(settings: &Settings) -> Vec<ResolvedModelArtifact> {
    if !settings.resolved_cache_policy().enabled {
        return Vec::new();
    }
    ResolvedCacheStore::valid_local_artifacts(&settings.data_dir)
}

/// Take the runtime lease on one locally resolved artifact and start serving it, or `None` when it
/// cannot be served after all (concurrently evicted, no longer complete, or a scope for the same
/// snapshot is already active). Both halves are acquired together so a scope can never outlive the
/// lease that protects the bytes behind it.
fn admit_local_artifact(
    settings: &Settings,
    artifact: Option<&ResolvedModelArtifact>,
) -> Option<(ResolvedCacheLease, ActiveLocalArtifacts)> {
    let artifact = artifact?;
    let cache_key = artifact.cache_key().ok()?;
    let store = cache_store(settings)?;
    let resolver = ModelArtifactResolver::new(sceneworks_core::hf_home::model_source_library(
        &settings.data_dir,
    ));
    // The store takes the entry's SHARED artifact lock before it re-reads and re-validates the
    // published entry under it, then stamps usage. An evictor that reaches the entry first holds
    // the exclusive lock, so this waits; an evictor that arrives afterwards finds the lock held
    // and must leave the entry alone.
    let lease = store
        .acquire_complete(&cache_key, &resolver, &artifact.identity.repository)
        .ok()
        .flatten()?;
    match sceneworks_core::model_artifacts::local_preference::prefer_local_artifacts(
        std::slice::from_ref(lease.artifact()),
    ) {
        Ok(scope) => Some((lease, scope)),
        Err(error) => {
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "resolved_cache_local_tier_unsupported",
                    "cacheKey": cache_key,
                    "repository": artifact.identity.repository,
                    "revision": artifact.identity.revision,
                    "reason": error.to_string(),
                }),
            );
            None
        }
    }
}

/// One long-lived cache session per data directory. Opening a session takes a durable session lock
/// and writes session records, so a fresh session per job would churn the store; the worker claims
/// one job at a time and keeps this handle for the process lifetime instead.
fn cache_store(settings: &Settings) -> Option<ResolvedCacheStore> {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    static STORES: OnceLock<Mutex<BTreeMap<std::path::PathBuf, ResolvedCacheStore>>> =
        OnceLock::new();
    let stores = STORES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut stores = stores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(store) = stores.get(&settings.data_dir) {
        return Some(store.clone());
    }
    let store = ResolvedCacheStore::open(&settings.data_dir).ok()?;
    stores.insert(settings.data_dir.clone(), store.clone());
    Some(store)
}

/// Report which tier will serve each admitted model. This is the runtime's answer to "where did
/// this load's weights come from", emitted once per job before any loader runs.
fn emit_source_tiers(resolutions: &[ModelResolution]) {
    for resolution in resolutions {
        let tier = match resolution.availability {
            ModelAvailability::LocalReady => "resolved_local",
            ModelAvailability::ExternalReady => "source_library",
            _ => continue,
        };
        let (repository, revision) = match &resolution.local_artifact {
            Some(artifact) => (
                artifact.identity.repository.clone(),
                artifact.identity.revision.clone(),
            ),
            None => resolution
                .requirements
                .iter()
                .find(|requirement| requirement.is_primary)
                .map(|requirement| {
                    (
                        requirement.repository.clone(),
                        requirement.revision.clone().unwrap_or_default(),
                    )
                })
                .unwrap_or_default(),
        };
        emit_event_value(
            Level::INFO,
            json!({
                "event": "model_source_tier_selected",
                "tier": tier,
                "repository": repository,
                "revision": revision,
            }),
        );
    }
}

/// Allowlist AT THE SEAM: only routes that cannot open model/HF bytes may run without model
/// carriers. `JobType` is `#[non_exhaustive]`, so this match cannot force a compile error for a
/// new variant; instead the wildcard arm classifies anything unlisted as model-backed — an
/// unclassified route therefore defaults to FAIL CLOSED (requires carriers) and can only become
/// model-free by an explicit entry in the allowlist above.
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
    use sceneworks_core::model_artifacts::ArtifactSourceLibrary;
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
        let managed = data_dir
            .join("models")
            .join(sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(repo));
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

    /// Pin the configured source library AND switch the resolved cache on for the body. Under
    /// test `Settings::resolved_cache_policy` reads the same environment `Settings::from_env`
    /// would have read, so this is exactly the production "user enabled the cache" state.
    fn with_local_cache<T>(library: &Path, body: impl FnOnce() -> T) -> T {
        crate::test_env::temp_env_vars(
            &[
                ("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path")),
                (
                    sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                    "1",
                ),
            ],
            body,
        )
    }

    /// Publish a real bundle through the PRODUCTION materializer: the source snapshot is copied,
    /// verified and atomically published exactly as a promotion would do it, so these tests read
    /// back the same on-disk shape the runtime will meet.
    fn publish_bundle(
        data_dir: &Path,
        library: &Path,
        repository: &str,
        revision: &str,
        variant: &str,
        files: &[&str],
    ) -> sceneworks_core::model_artifacts::ResolvedModelArtifact {
        use sceneworks_core::model_artifacts::local_preference::hub_cache_member_destination;
        use sceneworks_core::model_artifacts::resolved_cache::{
            MaterializationCancellation, MaterializationOutcome, ResolvedCacheMaterializer,
        };
        use sceneworks_core::model_artifacts::{
            ArtifactAvailability, ArtifactFile, ArtifactIdentity, ArtifactMemberRole,
            ResolvedBundleClosure, ResolvedBundleMember,
        };

        let identity = ArtifactIdentity::pinned(repository, revision, variant).unwrap();
        let closure = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: identity.clone(),
            tier: Some(variant.to_owned()),
            source_subpath: PathBuf::new(),
            destination: hub_cache_member_destination(repository, revision, Path::new("")).unwrap(),
            files: files
                .iter()
                .map(|file| ArtifactFile::new(*file).unwrap())
                .collect(),
        }])
        .unwrap();
        let resolver = ModelArtifactResolver::new(ArtifactSourceLibrary::new(library).unwrap());
        let source = resolver
            .resolve_source(identity, closure, ArtifactAvailability::Available)
            .unwrap();
        let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
            cache_key: source.cache_key().unwrap(),
            artifact: source,
        };
        let store = ResolvedCacheStore::open(data_dir).unwrap();
        let outcome = ResolvedCacheMaterializer::new(store)
            .materialize(
                &candidate,
                library,
                repository,
                &MaterializationCancellation::default(),
            )
            .unwrap();
        match outcome {
            MaterializationOutcome::Published(metadata) => metadata.artifact,
            other => panic!("bundle was not published: {other:?}"),
        }
    }

    /// The walking skeleton: an installed model with a published bundle admits from the app-owned
    /// tier, the SHARED snapshot resolver every image/video/audio/utility loader uses returns the
    /// bundle path, an evictor cannot take the entry while the job holds it, and finishing the job
    /// restores the source tier.
    #[test]
    fn a_published_bundle_serves_the_load_and_is_lease_protected() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "owner/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let cache_key = artifact.cache_key().unwrap();
            let bundle_root = artifact.location.root().to_path_buf();

            // Before the guard runs, the shared resolver reads the configured source library.
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model")
                    .expect("source snapshot");
            assert!(source_snapshot.starts_with(&library));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );

            // The one shared snapshot resolver — every model-consuming runtime funnels through it
            // — now answers with the app-owned bundle.
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model")
                    .expect("leased local snapshot");
            assert_eq!(loaded, bundle_root.join(format!(
                "models--owner--model/snapshots/{REV_A}"
            )));
            assert!(loaded.join("model.safetensors").is_file());
            // And so does the pinned-component resolver used by the utility/co-requisite lanes.
            assert_eq!(
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    &settings.data_dir,
                    "owner/model",
                    REV_A
                ),
                Some(loaded.clone())
            );

            // An evictor takes the entry's artifact lock exclusively; the live lease denies it.
            let evictor = ResolvedCacheStore::open(&settings.data_dir).unwrap();
            let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
                cache_key: cache_key.clone(),
                artifact: ModelArtifactResolver::new(
                    ArtifactSourceLibrary::new(&library).unwrap(),
                )
                .resolve_source(
                    artifact.identity.clone(),
                    artifact.closure.clone(),
                    sceneworks_core::model_artifacts::ArtifactAvailability::Available,
                )
                .unwrap(),
            };
            assert!(matches!(
                evictor.reserve(&candidate, &library, "owner/model").unwrap(),
                sceneworks_core::model_artifacts::resolved_cache::ReservationOutcome::Contended
            ));

            guard.finish_success().unwrap();
            // Usage was recorded for a real load, and the entry survives untouched.
            let metadata = evictor.lookup_complete(&cache_key).unwrap().unwrap();
            assert!(metadata.last_used_at.is_some());
            // With the job finished the source tier serves again.
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model"),
                Some(source_snapshot)
            );
        });
    }

    /// The runtimes whose model directory is a concrete path resolved BEFORE the load — training
    /// base models, captioners, analyzers (the `ManagedModelPath` entrypoint) — and the
    /// receipt-provenance lane both read the leased bundle too, so no model-consuming surface is
    /// left on the source tier while a lease is active.
    #[test]
    fn payload_resolved_and_receipt_paths_are_served_from_the_leased_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "owner/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle_snapshot = artifact
                .location
                .root()
                .join(format!("models--owner--model/snapshots/{REV_A}"));
            let source_snapshot = ArtifactSourceLibrary::new(&library)
                .unwrap()
                .repository_root("owner/model")
                .unwrap()
                .join("snapshots")
                .join(REV_A);

            // Without a lease the authoritative source path survives untouched.
            let before = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert!(before.ends_with(format!("snapshots/{REV_A}")));
            assert!(!before.starts_with(artifact.location.root()));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let during = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert_eq!(during, bundle_snapshot);
            // The receipt lane proves its install against the source and then hands the loader the
            // leased local path.
            assert_eq!(
                crate::model_jobs::huggingface_receipt_weights_dir(
                    &settings.data_dir,
                    "owner/model",
                    Some("m"),
                    None
                ),
                Some(bundle_snapshot)
            );
            guard.finish_success().unwrap();

            // And the redirect is gone with the lease.
            assert_eq!(
                crate::paths::normalize_app_managed_model_path(
                    &settings,
                    source_snapshot.to_str().unwrap(),
                    "base model",
                )
                .unwrap(),
                before
            );
        });
    }

    /// The epic's headline outcome: with the configured library disconnected, a complete local
    /// bundle still admits and still resolves — without the runtime ever reaching a downloader or
    /// recreating the configured library root.
    #[test]
    fn a_disconnected_library_still_serves_a_complete_bundle_without_downloading() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "owner/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model")
                    .expect("leased local snapshot while disconnected");
            assert!(loaded.join("model.safetensors").is_file());
            let resolved_root =
                std::fs::canonicalize(settings.data_dir.join("models").join("resolved")).unwrap();
            assert!(loaded.starts_with(&resolved_root), "{}", loaded.display());
            // Nothing recreated the configured library or a parallel download destination: the
            // preference path never reaches the downloader.
            assert!(!library.exists());
            guard.finish_success().unwrap();
        });
    }

    /// Each invalid-entry case runs against a bundle that WOULD have served the load, with exactly
    /// one property broken, and asserts the control first — so a case can only pass because the
    /// broken property is what moved the load back to the authoritative source tier.
    fn invalid_local_entry_falls_back(break_it: impl FnOnce(&Path, &Path)) {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "owner/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle = artifact.location.root().to_path_buf();
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model")
                    .expect("source snapshot");

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the intact bundle must serve this load"
            );
            drop(guard);

            break_it(&settings.data_dir, &bundle);

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady,
                "an invalid local entry must fall back to the authoritative source"
            );
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model"),
                Some(source_snapshot),
                "and the load must read the source library, not the invalid bundle"
            );
        });
    }

    fn published_file(bundle: &Path) -> PathBuf {
        bundle.join(format!(
            "models--owner--model/snapshots/{REV_A}/model.safetensors"
        ))
    }

    #[test]
    fn a_torn_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::write(published_file(bundle), b"").unwrap();
        });
    }

    #[test]
    fn a_partial_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::remove_file(published_file(bundle)).unwrap();
        });
    }

    #[test]
    fn an_unverifiable_local_entry_never_wins() {
        invalid_local_entry_falls_back(|data_dir, _| {
            let entries = data_dir.join("models/resolved/entries");
            for entry in std::fs::read_dir(&entries).unwrap().flatten() {
                for slot in 0..=1 {
                    let path = entry.path().join(format!("metadata.{slot}.json"));
                    if path.exists() {
                        std::fs::write(&path, b"{\"not\":\"a journal\"}").unwrap();
                    }
                }
            }
        });
    }

    /// A bundle of a DIFFERENT revision or a DIFFERENT tier than the one this request selected is
    /// not this request's artifact: the selected closure keeps its source tier and the local entry
    /// is left alone.
    #[test]
    fn a_wrong_revision_or_wrong_tier_bundle_is_not_used() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "owner/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "owner/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "owner/matrix",
            json!([
                { "repo": "owner/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "owner/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
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
        with_local_cache(&library, || {
            // Only the q4 tier is published locally.
            publish_bundle(
                &settings.data_dir,
                &library,
                "owner/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            drop(guard);
            // The q8 selection must NOT be served by the q4 bundle even though both share one
            // repository and one immutable revision.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            drop(guard);

            // A bundle of another revision of the same model is likewise not this model.
            seed_snapshot(&library, "owner/other", REV_B, "model.safetensors");
            publish_bundle(
                &settings.data_dir,
                &library,
                "owner/other",
                REV_B,
                "default",
                &["model.safetensors"],
            );
            let other = json!({
                "modelManifestEntry": {
                    "id": "other",
                    "downloads": [{ "provider": "huggingface", "repo": "owner/other",
                                    "revision": REV_A, "files": ["model.safetensors"] }]
                }
            })
            .as_object()
            .unwrap()
            .clone();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &other, &settings).unwrap();
            assert_ne!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "a bundle of a different revision must never satisfy a pinned request"
            );
        });
    }

    /// The resolved cache is opt-in: with it switched off no entry is read, no session is opened,
    /// and every runtime keeps reading the configured source library byte for byte.
    #[test]
    fn a_disabled_cache_never_prefers_a_local_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "owner/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
        });
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "owner/model")
                    .expect("source snapshot")
                    .starts_with(&library)
            );
        });
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
            assert!(error
                .to_string()
                .contains(EXTERNAL_LIBRARY_UNAVAILABLE_CODE));
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
        let other_os = if this_os == "macos" {
            "windows"
        } else {
            "macos"
        };
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
            assert!(matches!(
                remapped,
                WorkerError::ExternalLibraryUnavailable(_)
            ));
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
