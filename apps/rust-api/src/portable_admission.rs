//! Freeze the server-authorized model closure before a job enters the queue.
//!
//! The persisted binding contains logical slots, immutable source identities and content hashes.
//! Paths and credentials are used only while capturing a local source or querying pinned metadata;
//! they never enter the portable job binding or its authored-options projection.

use crate::error::ApiError;
use crate::portable_routes::{SelectedRouteSource, SelectedSourceLocation};
use crate::AppState;
use sceneworks_core::contracts::{JobStatus, JobType, JsonObject};
use sceneworks_core::model_artifacts::artifact_selection::{
    is_co_requisite_download, is_supported_model_download, requested_runtime_variant,
    safe_download_dir, selected_portable_model_artifact_closure,
};
use sceneworks_core::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, enumerate_portable_array_slots, portable_closure_sha256,
    registered_digest_member_claim_id, resolve_pinned_hf_files, PortableArtifactFile,
    PortableArtifactMember, PortableCheckpointPlan, PortableJobBinding, PortableModelBinding,
    PortableRouteBinding, PortableSourceClaim, PortableSourceIdentity, PortableSourceProvenance,
};
use sceneworks_core::model_artifacts::{
    confine_artifact_path, ArtifactIdentity, ArtifactMemberRole, ModelArtifactResolver,
};
use sceneworks_core::verified_source_cache::{SourceVerification, VerifiedSourceCache};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Worker-private, path-free request to prove an app-managed utility source. The API retains the
/// source only briefly; a worker independently finds the same exact closure under its own roots.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PortableSourceChallenge {
    pub(crate) id: String,
    pub(crate) source_id: String,
    pub(crate) files: Vec<PortableArtifactFile>,
    #[serde(skip)]
    pub(crate) expires_at: Option<Instant>,
    #[serde(skip)]
    pub(crate) proved_worker_ids: BTreeSet<String>,
    #[serde(skip)]
    pub(crate) allowed_worker_ids: BTreeSet<String>,
}

const SOURCE_CHALLENGE_TIMEOUT: Duration = Duration::from_secs(3 * 60);

pub(crate) fn pending_portable_source_challenges(
    state: &AppState,
    worker_id: &str,
) -> Vec<PortableSourceChallenge> {
    let mut pending = state.portable_source_challenges.lock();
    let now = Instant::now();
    pending.retain(|_, challenge| challenge.expires_at.is_some_and(|until| until > now));
    pending
        .values()
        .filter(|challenge| challenge.allowed_worker_ids.contains(worker_id))
        .cloned()
        .collect()
}

fn route_has_local_utility_primary(job_type: &JobType, route: &PortableRouteBinding) -> bool {
    matches!(
        job_type,
        JobType::DatasetAnalysis | JobType::TrainingCaption
    ) && route.models.iter().any(|model| {
        model.slot == "modelManifestEntries/0"
            && model.members.iter().any(|member| {
                member.role == ArtifactMemberRole::Primary
                    && matches!(member.source, PortableSourceIdentity::Digest { .. })
            })
    })
}

async fn wait_for_utility_source_proof(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
    route: &PortableRouteBinding,
) -> Result<bool, ApiError> {
    let _capacity = state
        .portable_source_probe_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::service_unavailable(
                "source_proof_busy",
                "Local source proof capacity is busy; retry shortly",
            )
        })?;
    if !matches!(
        job_type,
        JobType::DatasetAnalysis | JobType::TrainingCaption
    ) {
        return Ok(false);
    }
    let Some(model) = route
        .models
        .iter()
        .find(|model| model.slot == "modelManifestEntries/0")
    else {
        return Ok(false);
    };
    let Some(member) = model.members.iter().find(|member| {
        member.role == ArtifactMemberRole::Primary
            && matches!(member.source, PortableSourceIdentity::Digest { .. })
    }) else {
        return Ok(false);
    };
    let source_id =
        registered_digest_member_claim_id(model, member, &Value::Object(payload.clone()))
            .map_err(|error| incomplete(error.to_string()))?;
    let challenge_id = uuid::Uuid::new_v4().to_string();
    let initial_sets = state
        .jobs_store
        .portable_source_claim_sets_for_job_route_with_ids(
            &route.backend,
            &route.platform,
            requested_gpu,
            state.settings.worker_timeout_seconds,
            job_type,
            payload,
        )
        .map_err(|error| incomplete(format!("worker source claims are unavailable: {error}")))?;
    let authenticated = state.authenticated_worker_claims.lock().await;
    let allowed_worker_ids = initial_sets
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| authenticated.contains_key(id))
        .collect::<BTreeSet<_>>();
    drop(authenticated);
    if allowed_worker_ids.is_empty() {
        return Ok(false);
    }
    state.portable_source_challenges.lock().insert(
        challenge_id.clone(),
        PortableSourceChallenge {
            id: challenge_id.clone(),
            source_id: source_id.clone(),
            files: member.files.clone(),
            expires_at: Some(Instant::now() + SOURCE_CHALLENGE_TIMEOUT),
            proved_worker_ids: BTreeSet::new(),
            allowed_worker_ids,
        },
    );
    let deadline = Instant::now() + SOURCE_CHALLENGE_TIMEOUT;
    loop {
        let proved_workers = state
            .portable_source_challenges
            .lock()
            .get(&challenge_id)
            .map(|challenge| challenge.proved_worker_ids.clone())
            .unwrap_or_default();
        let sets = state
            .jobs_store
            .portable_source_claim_sets_for_job_route_with_ids(
                &route.backend,
                &route.platform,
                requested_gpu,
                state.settings.worker_timeout_seconds,
                job_type,
                payload,
            )
            .map_err(|error| {
                incomplete(format!("worker source claims are unavailable: {error}"))
            })?;
        let authenticated = state.authenticated_worker_claims.lock().await;
        let proof_matches = sets.iter().any(|(worker_id, _)| {
            proved_workers.contains(worker_id)
                && authenticated.get(worker_id).is_some_and(|record| {
                    record.seen_at.elapsed().as_secs() <= state.settings.worker_timeout_seconds
                        && replay_route_claims_match(route, payload, &record.claims).is_ok()
                })
        });
        drop(authenticated);
        if proof_matches {
            state
                .portable_source_challenges
                .lock()
                .remove(&challenge_id);
            return Ok(true);
        }
        if Instant::now() >= deadline {
            state
                .portable_source_challenges
                .lock()
                .remove(&challenge_id);
            return Ok(false);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[cfg(test)]
use sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256;

const MAX_PINNED_METADATA_CACHE_ENTRIES: usize = 256;
const TRUSTED_PINNED_LEDGER_VERSION: u32 = 1;

/// The host's record of a pinned revision's file ledger as upstream reported it. Only a
/// successful upstream resolution writes one; it is the trust root admission compares an
/// installed copy's rehashed bytes against when upstream is not reachable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TrustedPinnedLedger {
    schema_version: u32,
    base_url: String,
    repo: String,
    revision: String,
    selectors: Vec<String>,
    files: Vec<PortableArtifactFile>,
}

fn trusted_pinned_ledger_path(data_dir: &Path, key: &str) -> PathBuf {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(key.as_bytes());
    data_dir
        .join("cache")
        .join("trusted-pinned-files")
        .join(format!("{digest:x}.json"))
}

/// The recorded files for exactly this request, or `None` when nothing usable is recorded (absent,
/// unreadable, another request's record, or files that are not a valid ledger).
fn read_trusted_pinned_ledger(
    path: &Path,
    request: &TrustedPinnedLedger,
) -> Option<Vec<PortableArtifactFile>> {
    let body = std::fs::read(path).ok()?;
    let recorded: TrustedPinnedLedger = serde_json::from_slice(&body).ok()?;
    let same_request = TrustedPinnedLedger {
        files: Vec::new(),
        ..recorded.clone()
    } == *request;
    let valid = !recorded.files.is_empty()
        && sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
            &recorded.files,
        )
        .is_ok();
    (same_request && valid).then_some(recorded.files)
}

fn write_trusted_pinned_ledger(path: &Path, ledger: &TrustedPinnedLedger) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().expect("ledger path has a parent"))?;
    let body = serde_json::to_vec_pretty(ledger).map_err(std::io::Error::other)?;
    sceneworks_core::fsutil::write_atomic(path, &body)
}
static PINNED_METADATA_CACHE: OnceLock<Mutex<BTreeMap<String, Vec<PortableArtifactFile>>>> =
    OnceLock::new();

/// File-content verification can read several GiB. Keep it off Tokio's request-polling threads.
/// Each installed file is read at most once per host while its filesystem stamp (size, mtime and,
/// on Unix, device, inode and ctime) is unchanged: the verdict lives in the host's
/// [`VerifiedSourceCache`], which the worker shares, and any stamp change re-hashes the file.
async fn blocking_source_capture<T: Send + 'static>(
    capture: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(capture)
        .await
        .map_err(|error| {
            ApiError::internal(format!("Model source verification task failed: {error}"))
        })?
}

/// This host's hash-once verdict store, the same one the worker opens.
fn verified_source_cache(state: &AppState) -> VerifiedSourceCache {
    VerifiedSourceCache::for_data_dir(&state.settings.data_dir)
}

#[cfg(test)]
thread_local! {
    /// Files this thread actually read to verify; a reused verdict reads none.
    static HASHED_SOURCE_FILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn note_source_verification(verification: SourceVerification) {
    #[cfg(test)]
    if verification == SourceVerification::Hashed {
        HASHED_SOURCE_FILES.with(|count| count.set(count.get() + 1));
    }
    #[cfg(not(test))]
    let _ = verification;
}

/// Admission never cancels a hash midway.
fn no_cancellation() -> std::io::Result<()> {
    Ok(())
}

/// The entry slots are part of the private worker payload, not the public catalog. Preserve the
/// exact slot, including an array index, so duplicate model ids cannot swap components on retry.
fn payload_model_entries(payload: &JsonObject) -> Result<Vec<(String, &Value)>, ApiError> {
    let mut entries = Vec::new();
    for key in ["modelManifestEntry", "baseModelManifestEntry"] {
        if let Some(entry) = payload.get(key).filter(|entry| !entry.is_null()) {
            if !entry.is_object() {
                return Err(ApiError::bad_request(format!("{key} must be an object")));
            }
            entries.push((key.to_owned(), entry));
        }
    }
    if let Some(many) = payload.get("modelManifestEntries") {
        let many = many
            .as_array()
            .ok_or_else(|| ApiError::bad_request("modelManifestEntries must be an array"))?;
        for (index, entry) in many.iter().enumerate() {
            if !entry.is_object() {
                return Err(ApiError::bad_request(format!(
                    "modelManifestEntries/{index} must be an object"
                )));
            }
            entries.push((format!("modelManifestEntries/{index}"), entry));
        }
    }
    Ok(entries)
}

/// A pre-registry job may be rebound only from its persisted, server-stamped source carrier.
/// Local paths are locators, not identity: a moved root is allowed, while a changed publisher,
/// import plan, producer, component declaration or model id is not. A path-only user entry has
/// no immutable byte authority and remains unbound.
pub(crate) async fn validate_historical_source_carrier(
    state: &AppState,
    original: &JsonObject,
    current: &JsonObject,
) -> Result<(), ApiError> {
    let old = payload_model_entries(original)?;
    let new = payload_model_entries(current)?;
    if old.is_empty() || old.len() != new.len() {
        return Err(ApiError::bad_request(
            "This historical job has no portable source binding. Upgrade its manifest before retrying.",
        ));
    }
    for ((old_slot, old_entry), (new_slot, new_entry)) in old.iter().zip(new.iter()) {
        if old_slot != new_slot
            || historical_source_identity(old_entry) != historical_source_identity(new_entry)
        {
            return Err(incomplete(format!(
                "historical model source {old_slot} differs from its original declaration"
            )));
        }
        let old_entry = old_entry
            .as_object()
            .ok_or_else(|| incomplete("historical model source is not an object"))?;
        let new_entry = new_entry
            .as_object()
            .ok_or_else(|| incomplete("current model source is not an object"))?;
        let plan = old_entry.get("importPlan").and_then(Value::as_object);
        let pinned_plan = plan.is_some_and(|plan| {
            ["checkpointId", "planId", "semanticDigest"]
                .into_iter()
                .all(|key| {
                    plan.get(key)
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                })
        });
        let produced = old_entry
            .get("provenance")
            .and_then(|value| value.get("trainingJobId"))
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty());
        let pinned = |download: &Value| {
            download
                .get("repo")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
                && download
                    .get("revision")
                    .and_then(Value::as_str)
                    .is_some_and(|s| {
                        s.len() == 40 && s.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
        };
        // A registered builtin's `paths` are the install locations of its own pinned downloads
        // (`${HF_CACHE}/<repo>`), not a second local authority: its bytes are bound by those
        // pinned revisions. Only a user/imported row's paths name bytes with no publisher.
        let registered_builtin = |entry: &JsonObject| {
            entry.get("catalogScope").and_then(Value::as_str) == Some("builtin")
                && entry
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| sceneworks_core::registry::builtin_model_spec(id).is_some())
        };
        let install_locators_are_pinned =
            registered_builtin(old_entry) && registered_builtin(new_entry);
        let pinned_download = (old_entry.get("download").is_some_and(pinned)
            || old_entry
                .get("downloads")
                .and_then(Value::as_array)
                .is_some_and(|downloads| !downloads.is_empty() && downloads.iter().all(pinned)))
            && !["modelPath", "installedPath"]
                .into_iter()
                .any(|key| old_entry.get(key).is_some_and(|value| !value.is_null()))
            && (install_locators_are_pinned || old_entry.get("paths").is_none_or(Value::is_null))
            && old_entry
                .get("components")
                .and_then(Value::as_array)
                .is_none_or(|components| {
                    components
                        .iter()
                        .all(|component| component.get("path").is_none_or(Value::is_null))
                });
        if !pinned_plan && !produced && !pinned_download {
            return Err(incomplete(format!(
                "historical model source {old_slot} has no immutable byte authority"
            )));
        }
        if pinned_plan {
            let plan = plan.expect("pinned plan has its source declaration");
            let checkpoint_id = plan["checkpointId"].as_str().unwrap();
            let data_dir = state.settings.data_dir.clone();
            let checkpoint_id = checkpoint_id.to_owned();
            let resolved = tokio::task::spawn_blocking(move || {
                sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data_dir)
                    .resolve(&checkpoint_id)
            })
            .await
            .map_err(|error| incomplete(format!("historical import verification failed: {error}")))?
            .map_err(|error| incomplete(format!("historical import bytes are invalid: {error}")))?;
            if resolved.plan.plan_id != plan["planId"].as_str().unwrap()
                || resolved.record.plan.semantic_digest != plan["semanticDigest"].as_str().unwrap()
            {
                return Err(incomplete(
                    "historical import plan differs from its source declaration",
                ));
            }
            let provenance =
                sceneworks_core::model_artifacts::provenance::resolved_model_provenance(
                    &state.settings.data_dir,
                    new_entry,
                )
                .map_err(|error| {
                    incomplete(format!("historical import plan is invalid: {error}"))
                })?;
            if provenance.is_none() {
                return Err(incomplete(
                    "historical import plan has no validated provenance",
                ));
            }
            crate::ensure_expected_imported_model_reference(
                state,
                original,
                &Value::Object(new_entry.clone()),
            )?;
        }
        if produced && !pinned_plan {
            // A produced (trained or converted) source has no publisher to re-fetch from: its
            // authority is the producer receipt over the exact local bytes. Verify those bytes
            // now, so a substituted or missing output fails the job instead of waiting forever
            // for a route that can never be proven.
            let source = ["modelPath", "installedPath"]
                .into_iter()
                .find_map(|key| new_entry.get(key).and_then(Value::as_str))
                .or_else(|| {
                    new_entry
                        .get("paths")
                        .and_then(|paths| paths.get("model"))
                        .and_then(Value::as_str)
                })
                .ok_or_else(|| {
                    incomplete(format!(
                        "historical model source {old_slot} has no local producer output"
                    ))
                })?;
            verify_produced_local_source_offloaded(
                state,
                &Value::Object(new_entry.clone()),
                Path::new(source),
            )
            .await
            .map_err(|error| {
                incomplete(format!(
                    "historical model source {old_slot} bytes are missing or changed: {}",
                    error.detail
                ))
            })?;
        }
    }
    Ok(())
}

/// The source identity of one model carrier: what names the bytes and the producer, never how
/// the catalog presents them. Display text, UI copy, defaults, limits, capabilities, tags,
/// visibility, license display text and memory evidence may change between releases (or through
/// a presentation-only user override) without changing which bytes a job loads, so they are not
/// compared. Local paths are locators: only their presence is identity, never their value.
pub(crate) fn historical_source_identity(entry: &Value) -> Value {
    const SCALARS: [&str; 10] = [
        "id",
        "type",
        "family",
        "adapter",
        "baseModel",
        "catalogScope",
        "importSourceShape",
        "importQuantFormat",
        "minimaxH3Partition",
        "quantization",
    ];
    // `path` here is a file inside the pinned repository (a swapped weight file is a different
    // source), not a host locator.
    const DOWNLOAD: [&str; 13] = [
        "provider",
        "repo",
        "revision",
        "files",
        "path",
        "variant",
        "subdir",
        "subfolder",
        "componentId",
        "expectedSha256",
        "expectedSizeBytes",
        "weightsSha256",
        "localDerivation",
    ];
    fn pick(value: &Value, keys: &[&str]) -> Value {
        let Some(object) = value.as_object() else {
            return value.clone();
        };
        Value::Object(
            keys.iter()
                .filter_map(|key| Some(((*key).to_owned(), object.get(*key)?.clone())))
                .collect(),
        )
    }
    fn locator() -> Value {
        Value::String("<locator>".to_owned())
    }
    let Some(entry) = entry.as_object() else {
        return Value::Null;
    };
    let mut identity = JsonObject::new();
    for key in SCALARS {
        if let Some(value) = entry.get(key) {
            identity.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(download) = entry.get("download") {
        identity.insert("download".to_owned(), pick(download, &DOWNLOAD));
    }
    for key in ["downloads", "conditionalComponents"] {
        if let Some(downloads) = entry.get(key) {
            let projected = match downloads.as_array() {
                Some(rows) => Value::Array(rows.iter().map(|row| pick(row, &DOWNLOAD)).collect()),
                None => downloads.clone(),
            };
            identity.insert(key.to_owned(), projected);
        }
    }
    if let Some(files) = entry.get("files") {
        identity.insert("files".to_owned(), files.clone());
    }
    if let Some(components) = entry.get("components") {
        let projected = match components.as_array() {
            Some(rows) => Value::Array(
                rows.iter()
                    .map(|row| match row.as_object() {
                        Some(component) => Value::Object(
                            component
                                .iter()
                                .filter(|(key, _)| {
                                    !matches!(
                                        key.as_str(),
                                        "name" | "label" | "description" | "displayName"
                                    )
                                })
                                .map(|(key, value)| {
                                    let value = if key == "path" {
                                        locator()
                                    } else {
                                        value.clone()
                                    };
                                    (key.clone(), value)
                                })
                                .collect(),
                        ),
                        None => row.clone(),
                    })
                    .collect(),
            ),
            None => components.clone(),
        };
        identity.insert("components".to_owned(), projected);
    }
    if let Some(source) = entry.get("source") {
        let mut projected = pick(source, &["provider", "repo", "revision", "variant"]);
        if source.get("path").is_some() {
            projected["path"] = locator();
        }
        identity.insert("source".to_owned(), projected);
    }
    if let Some(plan) = entry.get("importPlan") {
        identity.insert(
            "importPlan".to_owned(),
            pick(plan, &["checkpointId", "planId", "semanticDigest"]),
        );
    }
    if let Some(producer) = entry
        .get("provenance")
        .and_then(|provenance| provenance.get("trainingJobId"))
    {
        identity.insert(
            "provenance".to_owned(),
            serde_json::json!({"trainingJobId": producer}),
        );
    }
    if let Some(trained) = entry.get("trainedCheckpoint") {
        identity.insert(
            "trainedCheckpoint".to_owned(),
            pick(
                trained,
                &[
                    "modelId",
                    "engineModel",
                    "family",
                    "task",
                    "baseModel",
                    "layout",
                ],
            ),
        );
    }
    if let Some(encoder) = entry.get("resolvedTextEncoder") {
        // The authored encoder choice and its source kind are identity; the resolved path is a
        // locator re-derived from this host's catalog, and its bytes are captured separately.
        identity.insert(
            "resolvedTextEncoder".to_owned(),
            pick(encoder, &["selectionId", "sourceKind"]),
        );
    }
    if let Some(paths) = entry.get("paths") {
        let projected = match paths {
            Value::Object(paths) => Value::Object(
                paths
                    .iter()
                    .filter(|(_, value)| !value.is_null())
                    .map(|(key, _)| (key.clone(), locator()))
                    .collect(),
            ),
            _ => Value::Null,
        };
        identity.insert("paths".to_owned(), projected);
    }
    for key in ["modelPath", "installedPath"] {
        if entry.get(key).is_some_and(|value| !value.is_null()) {
            identity.insert(key.to_owned(), locator());
        }
    }
    Value::Object(identity)
}

/// Bind source-affecting authored choices. Prompt, seed and render controls remain in the
/// canonical job payload and may change on an explicitly duplicated job without changing the
/// frozen source. Model/variant/decoder/adapter choices cannot silently change that source.
pub(crate) fn portable_authored_options(payload: &JsonObject) -> Value {
    fn project(value: &Value, parent: Option<&str>) -> Value {
        match value {
            Value::Object(fields) => {
                let mut selected = JsonObject::new();
                for (key, value) in fields {
                    if is_private_or_derived_field(key) {
                        continue;
                    }
                    if key == "poses" {
                        selected.insert(
                            "posesSelected".to_owned(),
                            Value::Bool(value.as_array().is_some_and(|poses| !poses.is_empty())),
                        );
                    } else if key == "referenceAssetId" {
                        selected.insert(
                            "referenceSelected".to_owned(),
                            Value::Bool(value.as_str().is_some_and(|id| !id.trim().is_empty())),
                        );
                    } else if is_source_choice_container(key) {
                        selected.insert(key.clone(), project(value, Some(key)));
                    } else if is_source_choice_leaf(key, parent) {
                        selected.insert(key.clone(), value.clone());
                    }
                }
                Value::Object(selected)
            }
            Value::Array(values) => {
                Value::Array(values.iter().map(|value| project(value, parent)).collect())
            }
            _ => value.clone(),
        }
    }
    let mut options = project(&Value::Object(payload.clone()), None);
    if let Some(target) = payload.get("plan").and_then(|plan| plan.get("target")) {
        let selected = options
            .as_object_mut()
            .expect("projection returns an object");
        for (source_key, authored_key) in [
            ("targetId", "trainingTargetId"),
            ("kernel", "trainingKernel"),
            ("baseModel", "trainingBaseModel"),
        ] {
            if let Some(value) = target.get(source_key).and_then(Value::as_str) {
                selected.insert(authored_key.to_owned(), Value::String(value.to_owned()));
            }
        }
        if let Some(weight) = payload
            .get("plan")
            .and_then(|plan| plan.pointer("/config/advanced/iris/repa/weight"))
            .and_then(Value::as_f64)
        {
            selected.insert("irisRepaSelected".to_owned(), Value::Bool(weight > 0.0));
        }
    }
    options
}

fn is_source_choice_container(key: &str) -> bool {
    matches!(
        key,
        "advanced"
            | "workflow"
            | "rasterStage"
            | "vectorStage"
            | "phases"
            | "phase"
            | "loras"
            | "lokrs"
            | "controls"
            | "controlWeights"
            | "pidCheckpoint"
            | "veda"
            | "source"
            | "upscale"
            // The YuE2 block's `decoder` names the VAE snapshot the worker's
            // `selected_portable_model_artifact_closure` selects (the generic `decoder` leaf only
            // projects under keys that are already containers), so a duplicate that changes it
            // must not keep the frozen decoder binding (SW-7030).
            | "yue2"
    )
}

fn is_source_choice_leaf(key: &str, parent: Option<&str>) -> bool {
    matches!(
        key,
        "model"
            | "modelId"
            | "baseModel"
            | "irisModel"
            | "checkpointModelId"
            | "sourceRepo"
            | "converter"
            | "baseRepo"
            | "baseSubdir"
            | "sourceFile"
            | "upscalerFile"
            | "modelNameOrPath"
            | "embedder"
            | "engine"
            | "mode"
            | "editMode"
            | "variant"
            | "quantTier"
            | "mlxQuantize"
            | "convRot"
            | "decoder"
            | "controlMode"
            | "controlImage"
            | "overlayId"
            | "componentId"
            | "textEncoderModel"
            | "enhancerModel"
            | "useUncensoredEnhancer"
            | "enhancePrompt"
            | "tileControlNetRepo"
            | "pidGemma"
            | "usePid"
            | "structuredAnalysisEnabled"
            | "visionAnalysisEnabled"
            | "semanticEmbeddingsEnabled"
            | "enabled"
            | "repo"
            | "revision"
            | "filename"
    ) || matches!(key, "id" | "index") && matches!(parent, Some("loras" | "lokrs" | "controls"))
}

fn is_private_or_derived_field(key: &str) -> bool {
    if key == "baseSubdir" {
        // A tier directory name is authored source choice, unlike physical local locator paths.
        return false;
    }
    if matches!(
        key,
        "modelManifestEntry"
            | "baseModelManifestEntry"
            | "modelManifestEntries"
            | "modelProvenance"
            | "modelArtifactOperation"
            | "expectedImportedModelRef"
            | "repairExpectedBinding"
            | "importPlan"
            | "installedPath"
            | "sourcePath"
            | "sourceUrl"
            | "targetDir"
            | "paths"
            | "path"
            | "modelPath"
            | "outputDir"
            | "credentialHost"
            | "parentJobId"
            | "childJobId"
            | "intermediateAssetId"
            | "jobId"
            | "createdAt"
    ) {
        return true;
    }
    let lower = key.to_ascii_lowercase();
    lower.ends_with("path")
        || lower.ends_with("dir")
        || lower.contains("token")
        || lower.contains("credential")
        || lower.contains("password")
        || lower.contains("secret")
}

/// Called only after the route has canonicalized its payload and source policy. A later worker
/// claim selects one frozen backend/platform route; a retry compares this envelope with the
/// persisted one before any queue mutation.
pub(crate) async fn capture_portable_model_binding(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
) -> Result<Option<PortableJobBinding>, ApiError> {
    capture_portable_model_binding_inner(state, job_type, payload, requested_gpu, None).await
}

/// Replay keeps the admitted immutable source even when a mutable repository head moves.
/// The caller supplies the original canonical source carriers in `payload`; this checks their
/// present bytes, current server route policy, and every frozen route member before enqueue.
pub(crate) async fn validate_frozen_replay_binding(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
    frozen: &PortableJobBinding,
) -> Result<PortableJobBinding, ApiError> {
    frozen
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    // The private persisted ledger was authenticated at original admission. Replay checks the
    // current source policy and exact local/worker availability without contacting mutable HEAD
    // or requiring a second publisher metadata lookup during an API network outage.
    let candidate = replay_from_frozen_policy(
        &state.settings.data_dir,
        job_type,
        payload,
        requested_gpu,
        frozen,
    )?;
    for (key, route) in &candidate.routes {
        if frozen.routes.get(key) != Some(route) {
            return Err(incomplete(format!(
                "replay source route {key} differs from its frozen identity"
            )));
        }
    }
    filter_replay_routes_by_registered_claims(state, job_type, payload, requested_gpu, candidate)
        .await
}

fn replay_from_frozen_policy(
    data_dir: &Path,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
    frozen: &PortableJobBinding,
) -> Result<PortableJobBinding, ApiError> {
    let authored = portable_authored_options(payload);
    if authored != frozen.authored_options {
        return Err(incomplete(
            "replay source-affecting choices differ from frozen admission",
        ));
    }
    if payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("modelPath"))
        .and_then(Value::as_str)
        .is_some_and(|path| !path.trim().is_empty())
    {
        return Err(incomplete(
            "unregistered modelPath cannot be replayed from a remote worker root",
        ));
    }
    let entries = payload_model_entries(payload)?;
    let mut routes = BTreeMap::new();
    for (backend, platform) in crate::portable_routes::eligible_portable_routes(
        job_type,
        payload,
        &entries,
        requested_gpu,
        data_dir,
    )? {
        let key = format!("{backend}:{platform}");
        let Some(route) = frozen.routes.get(&key) else {
            // Initial admission may have frozen only routes with a worker able to prove the
            // complete local Digest closure. A newly eligible route cannot expand replay,
            // and an explicit GPU id selects a backend, not a platform: intersect the
            // enumerated platforms with the frozen routes for that backend and reject
            // only when no compatible route remains (the empty check below).
            continue;
        };
        if route.models.len() != entries.len() {
            return Err(incomplete("replay model carrier count changed"));
        }
        for (slot, entry) in &entries {
            let model = route
                .models
                .iter()
                .find(|model| model.slot == *slot)
                .ok_or_else(|| incomplete(format!("replay model slot {slot} was not frozen")))?;
            if entry.get("id").and_then(Value::as_str) != Some(&model.model_id) {
                return Err(incomplete(format!(
                    "replay model slot {slot} changed identity"
                )));
            }
        }
        let selected = crate::portable_routes::selected_route_source_rows(
            job_type, payload, &entries, &backend, &platform,
        )
        .map_err(incomplete)?;
        for source in selected {
            let model = route
                .models
                .iter()
                .find(|model| model.slot == source.model_slot)
                .ok_or_else(|| incomplete("replay selected source lost its model slot"))?;
            if frozen_source_member(Some(model), &source.role, &source.component_id)?.is_none() {
                return Err(incomplete("replay selected source was not frozen"));
            }
        }
        routes.insert(key, route.clone());
    }
    if routes.is_empty() {
        return Err(incomplete(
            "replay has no currently authorized source route",
        ));
    }
    PortableJobBinding::new(authored, routes).map_err(|error| incomplete(error.to_string()))
}

async fn filter_replay_routes_by_registered_claims(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
    candidate: PortableJobBinding,
) -> Result<PortableJobBinding, ApiError> {
    let mut available = BTreeMap::new();
    for (key, route) in &candidate.routes {
        let requires_worker_claim = route
            .models
            .iter()
            .flat_map(|model| &model.members)
            .any(|member| !matches!(member.source, PortableSourceIdentity::Revision { .. }));
        if !requires_worker_claim {
            available.insert(key.clone(), route.clone());
            continue;
        }
        let sets = state
            .jobs_store
            .portable_source_claim_sets_for_job_route(
                &route.backend,
                &route.platform,
                requested_gpu,
                state.settings.worker_timeout_seconds,
                job_type,
                payload,
            )
            .map_err(|error| {
                incomplete(format!(
                    "worker source availability is unavailable: {error}"
                ))
            })?;
        let matched = if route_has_local_utility_primary(job_type, route) {
            wait_for_utility_source_proof(state, job_type, payload, requested_gpu, route).await?
        } else {
            sets.iter()
                .any(|claims| replay_route_claims_match(route, payload, claims).is_ok())
        };
        if matched {
            available.insert(key.clone(), route.clone());
        }
    }
    if available.is_empty() {
        return Err(incomplete(
            "frozen local source is unavailable from every eligible worker",
        ));
    }
    PortableJobBinding::new(candidate.authored_options, available)
        .map_err(|error| incomplete(error.to_string()))
}

fn replay_route_claims_match(
    route: &PortableRouteBinding,
    payload: &JsonObject,
    claims: &[PortableSourceClaim],
) -> Result<(), ApiError> {
    let value = Value::Object(payload.clone());
    for model in &route.models {
        for member in &model.members {
            match &member.source {
                PortableSourceIdentity::Digest { sha256 } => {
                    let source_id = registered_digest_member_claim_id(model, member, &value)
                        .map_err(|error| incomplete(error.to_string()))?;
                    if !claims.iter().any(|claim| {
                        claim.source_id == source_id
                            && claim.provenance == PortableSourceProvenance::OperatorOverride
                            && claim.files == member.files
                            && portable_closure_sha256(&claim.files).ok().as_deref()
                                == Some(sha256.as_str())
                    }) {
                        return Err(incomplete(format!(
                            "registered worker source {source_id} differs from frozen bytes"
                        )));
                    }
                }
                PortableSourceIdentity::WorkerClaim { .. } => {
                    if !claims
                        .iter()
                        .any(|claim| claim.matches_worker_claim_member(member).is_ok())
                    {
                        return Err(incomplete("frozen worker source claim is unavailable"));
                    }
                }
                PortableSourceIdentity::Revision { .. } => {}
            }
        }
    }
    Ok(())
}

/// A vector workflow may fail before it creates its raster child. Its parent nevertheless froze
/// the raster carrier under `modelManifestEntries/0`; derive the child identity from those exact
/// bytes after checking the recorded raster model and current child route policy.
pub(crate) async fn derive_vector_raster_child_binding(
    state: &AppState,
    child_type: &JobType,
    child_payload: &JsonObject,
    requested_gpu: &str,
    parent_frozen: &PortableJobBinding,
) -> Result<PortableJobBinding, ApiError> {
    if *child_type != JobType::ImageGenerate {
        return Err(incomplete("vector raster child has an unexpected job type"));
    }
    parent_frozen
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    let recorded_model = parent_frozen
        .authored_options
        .pointer("/workflow/rasterStage/model")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("vector parent has no recorded raster model"))?;
    if child_payload.get("model").and_then(Value::as_str) != Some(recorded_model) {
        return Err(incomplete("vector raster child changed the recorded model"));
    }
    let entries = payload_model_entries(child_payload)?;
    if entries.len() != 1
        || entries[0].0 != "modelManifestEntry"
        || entries[0].1.get("id").and_then(Value::as_str) != Some(recorded_model)
    {
        return Err(incomplete(
            "vector raster child has no original model carrier",
        ));
    }
    let authored = portable_authored_options(child_payload);
    let authored_hash = canonical_authored_options_sha256(&authored)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut routes = BTreeMap::new();
    for (backend, platform) in crate::portable_routes::eligible_portable_routes(
        child_type,
        child_payload,
        &entries,
        requested_gpu,
        &state.settings.data_dir,
    )? {
        let key = format!("{backend}:{platform}");
        let Some(parent_route) =
            frozen_vector_child_route(&parent_frozen.routes, &key, requested_gpu)?
        else {
            continue;
        };
        let raster = parent_route
            .models
            .iter()
            .find(|model| model.slot == "modelManifestEntries/0")
            .ok_or_else(|| incomplete("vector parent did not freeze the raster source"))?;
        if raster.model_id != recorded_model {
            return Err(incomplete("vector parent raster identity changed"));
        }
        let members = raster
            .members
            .iter()
            .cloned()
            .map(|mut member| {
                if let Some(id) = member.component_id.as_mut() {
                    if let Some(suffix) = id.strip_prefix("modelManifestEntries/0/") {
                        *id = format!("modelManifestEntry/{suffix}");
                    }
                }
                member
            })
            .collect();
        let mut model = PortableModelBinding::new(
            "modelManifestEntry",
            &raster.model_id,
            &backend,
            &raster.variant,
            raster.tier.clone(),
            &authored_hash,
            members,
        )
        .map_err(|error| incomplete(error.to_string()))?;
        if let Some(plan) = &raster.checkpoint_plan {
            model = model
                .with_checkpoint_plan(plan.clone())
                .map_err(|error| incomplete(error.to_string()))?;
        }
        let route = PortableRouteBinding::new(&backend, &platform, vec![model])
            .map_err(|error| incomplete(error.to_string()))?;
        routes.insert(key, route);
    }
    if routes.is_empty() {
        return Err(incomplete(
            "vector parent has no compatible frozen child route",
        ));
    }
    let frozen_child =
        PortableJobBinding::new(authored, routes).map_err(|error| incomplete(error.to_string()))?;
    validate_frozen_replay_binding(
        state,
        child_type,
        child_payload,
        requested_gpu,
        &frozen_child,
    )
    .await
}

fn frozen_vector_child_route<'a>(
    routes: &'a BTreeMap<String, PortableRouteBinding>,
    key: &str,
    requested_gpu: &str,
) -> Result<Option<&'a PortableRouteBinding>, ApiError> {
    match routes.get(key) {
        Some(route) => Ok(Some(route)),
        // A local raster source may have been provable on only a subset of workers. A child
        // cannot add a route the parent did not freeze, but Auto can keep its valid routes.
        None if requested_gpu == "auto" => Ok(None),
        None => Err(incomplete(format!(
            "vector parent did not freeze child route {key}"
        ))),
    }
}

async fn capture_portable_model_binding_inner(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    requested_gpu: &str,
    frozen: Option<&PortableJobBinding>,
) -> Result<Option<PortableJobBinding>, ApiError> {
    if *job_type == JobType::ModelConvert {
        for key in ["sourceFile", "upscalerFile"] {
            if let Some(file) = payload.get(key).and_then(Value::as_str) {
                if key == "upscalerFile" && file.is_empty() {
                    continue;
                }
                if file.starts_with('/')
                    || file.contains(['\\', ':', '\0'])
                    || file
                        .split('/')
                        .any(|part| part.is_empty() || matches!(part, "." | ".."))
                {
                    return Err(incomplete(format!(
                        "conversion {key} is not a portable repository-relative file"
                    )));
                }
            }
        }
        if let Some(subdir) = payload.get("baseSubdir").and_then(Value::as_str) {
            if !subdir.is_empty()
                && (matches!(subdir, "." | "..")
                    || !subdir.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                    }))
            {
                return Err(incomplete(
                    "conversion baseSubdir is not a portable tier name",
                ));
            }
        }
    }
    if sceneworks_core::model_artifacts::portable_binding::model_free_job(job_type, payload) {
        return Ok(None);
    }
    let entries = payload_model_entries(payload)?;
    if entries.is_empty() {
        return Err(incomplete(
            "model-backed job has no canonical model source carrier",
        ));
    }
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        let base = payload
            .get("plan")
            .and_then(|plan| plan.pointer("/target/baseModel"))
            .and_then(Value::as_str)
            .ok_or_else(|| incomplete("training plan has no base model"))?;
        if !entries
            .iter()
            .any(|(_, entry)| entry.get("id").and_then(Value::as_str) == Some(base))
        {
            return Err(incomplete(
                "training base has no matching canonical model carrier",
            ));
        }
    }
    let authored_options = portable_authored_options(payload);
    if frozen.is_some_and(|frozen| frozen.authored_options != authored_options) {
        return Err(incomplete(
            "replay source-affecting choices differ from frozen admission",
        ));
    }
    let authored_hash = canonical_authored_options_sha256(&authored_options)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut routes = BTreeMap::new();
    let mut last_route_error = None;
    for (backend, platform) in crate::portable_routes::eligible_portable_routes(
        job_type,
        payload,
        &entries,
        requested_gpu,
        &state.settings.data_dir,
    )? {
        let frozen_route = frozen
            .map(|frozen| {
                frozen
                    .routes
                    .get(&format!("{backend}:{platform}"))
                    .ok_or_else(|| incomplete("replay route was not admitted originally"))
            })
            .transpose()?;
        let selected_sources = crate::portable_routes::selected_route_source_rows(
            job_type, payload, &entries, &backend, &platform,
        )
        .map_err(incomplete)?;
        for source in &selected_sources {
            if !entries.iter().any(|(slot, _)| *slot == source.model_slot) {
                return Err(incomplete(format!(
                    "selected source {} has no model carrier slot {}",
                    source.component_id, source.model_slot
                )));
            }
        }
        let uses_worker_claims = selected_sources.iter().any(|source| {
            matches!(
                &source.source,
                SelectedSourceLocation::OperatorCachedRepo(_)
                    | SelectedSourceLocation::ConversionBorrow { .. }
                    | SelectedSourceLocation::PreferredWorkerClaim { .. }
                    | SelectedSourceLocation::OperatorOverrideOrManifest { .. }
            )
        });
        let worker_only_source = selected_sources.iter().any(|source| {
            matches!(
                &source.source,
                SelectedSourceLocation::OperatorCachedRepo(_)
                    | SelectedSourceLocation::ConversionBorrow { .. }
                    | SelectedSourceLocation::PreferredWorkerClaim { .. }
            )
        });
        let mut claim_sets = if uses_worker_claims {
            state
                .jobs_store
                .portable_source_claim_sets_for_job_route(
                    &backend,
                    &platform,
                    requested_gpu,
                    state.settings.worker_timeout_seconds,
                    job_type,
                    payload,
                )
                .map_err(|error| {
                    incomplete(format!("worker source claims are unavailable: {error}"))
                })?
        } else {
            vec![Vec::new()]
        };
        if claim_sets.is_empty() && !worker_only_source {
            // A pinned manifest default remains admissible when no worker advertises an optional
            // operator override. Its exact Revision bytes are verified when a worker claims it.
            claim_sets.push(Vec::new());
        }
        let preferred_count = |claims: &[PortableSourceClaim]| {
            selected_sources
                .iter()
                .filter_map(|source| match &source.source {
                    SelectedSourceLocation::PreferredWorkerClaim { preferred_id, .. } => {
                        Some(*preferred_id)
                    }
                    SelectedSourceLocation::OperatorOverrideOrManifest { source_id, .. } => {
                        Some(*source_id)
                    }
                    _ => None,
                })
                .filter(|source_id| claims.iter().any(|claim| claim.source_id == *source_id))
                .count()
        };
        claim_sets.sort_by(|left, right| {
            preferred_count(right)
                .cmp(&preferred_count(left))
                .then_with(|| {
                    conversion_claim_tier_rank(&selected_sources, left)
                        .cmp(&conversion_claim_tier_rank(&selected_sources, right))
                })
        });
        let mut selected_route = None;
        for source_claims in &claim_sets {
            let mut models = Vec::with_capacity(entries.len());
            let mut failed = None;
            for (index, (slot, entry)) in entries.iter().enumerate() {
                let frozen_model = frozen_route
                    .map(|route| {
                        route
                            .models
                            .iter()
                            .find(|model| model.slot == *slot)
                            .ok_or_else(|| {
                                incomplete(format!("replay model slot {slot} was not frozen"))
                            })
                    })
                    .transpose()?;
                match capture_model_slot(
                    state,
                    job_type,
                    payload,
                    slot,
                    entry,
                    &backend,
                    &platform,
                    &authored_hash,
                    index == 0,
                    &selected_sources,
                    source_claims,
                    frozen_model,
                )
                .await
                {
                    Ok(model) => models.push(model),
                    Err(error) => {
                        failed = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = failed {
                last_route_error = Some(error);
                continue;
            }
            let route = PortableRouteBinding::new(&backend, &platform, models)
                .map_err(|error| incomplete(error.to_string()))?;
            let has_digest = route
                .models
                .iter()
                .flat_map(|model| &model.members)
                .any(|member| matches!(member.source, PortableSourceIdentity::Digest { .. }));
            if has_digest {
                let proof_sets = if uses_worker_claims {
                    vec![source_claims.clone()]
                } else {
                    state
                        .jobs_store
                        .portable_source_claim_sets_for_job_route(
                            &backend,
                            &platform,
                            requested_gpu,
                            state.settings.worker_timeout_seconds,
                            job_type,
                            payload,
                        )
                        .map_err(|error| {
                            incomplete(format!("worker source claims are unavailable: {error}"))
                        })?
                };
                let proved = if route_has_local_utility_primary(job_type, &route) {
                    wait_for_utility_source_proof(state, job_type, payload, requested_gpu, &route)
                        .await?
                } else {
                    proof_sets
                        .iter()
                        .any(|claims| replay_route_claims_match(&route, payload, claims).is_ok())
                };
                if !proved {
                    last_route_error = Some(incomplete(format!(
                        "no eligible worker advertises the complete local source closure for {backend}:{platform}"
                    )));
                    continue;
                }
            }
            selected_route = Some(route);
            break;
        }
        if let Some(route) = selected_route {
            routes.insert(route.key(), route);
        } else if worker_only_source && last_route_error.is_none() {
            last_route_error = Some(incomplete(format!(
                "no eligible worker advertises the complete selected source closure for {backend}:{platform}"
            )));
        }
    }
    if routes.is_empty() {
        return Err(last_route_error.unwrap_or_else(|| {
            incomplete("no authorized worker route has an exact source binding")
        }));
    }
    PortableJobBinding::new(authored_options, routes)
        .map(Some)
        .map_err(|error| incomplete(error.to_string()))
}

fn conversion_claim_tier_rank(
    selected_sources: &[SelectedRouteSource],
    claims: &[PortableSourceClaim],
) -> usize {
    selected_sources
        .iter()
        .filter_map(|source| match &source.source {
            SelectedSourceLocation::ConversionBorrow {
                repo,
                preferred_subdir,
            } => Some((repo.as_str(), preferred_subdir.as_deref())),
            _ => None,
        })
        .map(|(repo, preferred)| {
            let mut tiers = preferred
                .filter(|tier| !tier.is_empty())
                .map(|tier| vec![tier])
                .unwrap_or_else(|| vec!["root"]);
            if preferred.is_some() {
                for fallback in ["bf16", "q8", "q4"] {
                    if !tiers.contains(&fallback) {
                        tiers.push(fallback);
                    }
                }
            }
            claims
                .iter()
                .filter_map(|claim| {
                    let (claim_repo, _, tier) =
                        sceneworks_core::portable_sources::parse_flux2_convert_base_claim_id(
                            &claim.source_id,
                        )?;
                    (claim_repo == repo)
                        .then(|| tiers.iter().position(|candidate| *candidate == tier))
                        .flatten()
                })
                .min()
                .unwrap_or(usize::MAX / 2)
        })
        .sum::<usize>()
}

fn incomplete(message: impl Into<String>) -> ApiError {
    ApiError::model_artifact_conflict(message.into(), "model_artifact_incomplete")
}

struct TrainingBaseSelection {
    path: PathBuf,
    tier: String,
}

/// The native trainer loads plan.target.baseModelPath directly. Its effective tier is determined
/// by the training target, independently of a generation catalog's default tier.
fn training_base_selection(
    job_type: &JobType,
    payload: &JsonObject,
    model_id: &str,
) -> Result<Option<TrainingBaseSelection>, ApiError> {
    if !matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        return Ok(None);
    }
    let target = payload
        .get("plan")
        .and_then(|plan| plan.get("target"))
        .and_then(Value::as_object)
        .ok_or_else(|| incomplete("training plan has no canonical target"))?;
    let base = target
        .get("baseModel")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("training plan has no base model"))?;
    if base != model_id {
        return Ok(None);
    }
    let kernel = target
        .get("kernel")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("training plan has no native kernel"))?;
    let path = target
        .get("baseModelPath")
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| incomplete("training plan has no resolved base model path"))?;
    let tier = if base == "ltx_2_5" || kernel == "ltx_mlx_lora" {
        "q4"
    } else {
        "bf16"
    };
    Ok(Some(TrainingBaseSelection {
        path: PathBuf::from(path),
        tier: tier.to_owned(),
    }))
}

/// A builtin training path is allowed only inside the selected pinned snapshot. Rehash the exact
/// directory the native trainer will open and compare it to the publisher's immutable file tree.
async fn verify_training_primary_path(
    state: &AppState,
    selected_path: &Path,
    download: &Value,
    members: &[PortableArtifactMember],
) -> Result<(), ApiError> {
    let repo = required_download_string(download, "repo")?;
    let revision = required_download_string(download, "revision")?;
    let library = sceneworks_core::hf_home::model_source_library(&state.settings.data_dir);
    let allowed = vec![library.root().to_path_buf()];
    let resolver = ModelArtifactResolver::new(library);
    let (_, snapshot) = resolver
        .discover_source_snapshot(repo, Some(revision))
        .map_err(|error| {
            incomplete(format!(
                "training base pinned snapshot is unavailable: {error}"
            ))
        })?;
    let snapshot = confine_artifact_path(&snapshot, &allowed)
        .map_err(|error| incomplete(error.to_string()))?;
    let selected = confine_artifact_path(selected_path, &allowed)
        .map_err(|error| incomplete(error.to_string()))?;
    let prefix = selected
        .strip_prefix(&snapshot)
        .map_err(|_| incomplete("training base path is outside its pinned publisher snapshot"))?;
    let actual = capture_local_source_files_offloaded(state, &selected, &allowed).await?;
    let remote = members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::Primary)
        .ok_or_else(|| incomplete("training base has no pinned primary member"))?;
    let mut expected = remote
        .files
        .iter()
        .filter_map(|file| {
            file.relative_path
                .strip_prefix(prefix)
                .ok()
                .and_then(|relative| {
                    (!relative.as_os_str().is_empty()).then(|| PortableArtifactFile {
                        relative_path: relative.to_path_buf(),
                        size_bytes: file.size_bytes,
                        sha256: file.sha256.clone(),
                    })
                })
        })
        .collect::<Vec<_>>();
    expected.sort();
    if expected.is_empty() || actual != expected {
        return Err(incomplete(
            "training base bytes differ from selected pinned publisher files",
        ));
    }
    Ok(())
}

/// The variant a model slot's binding freezes: the training plan's tier, else the request's explicit
/// tier, else the tier the worker would run for an unselected image request.
fn slot_requested_variant(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    slot: &str,
    entry: &Value,
    training: Option<&TrainingBaseSelection>,
) -> Option<String> {
    training
        .map(|selection| selection.tier.clone())
        .or_else(|| requested_runtime_variant(payload))
        .or_else(|| implicit_standard_tier_variant(state, job_type, slot, entry))
}

/// The tier an image request that names none runs, so the binding freezes that tier rather than
/// the catalog's default DOWNLOAD (the smallest install, often q4). For a standard quant-tier model
/// the worker's implicit choice is the app-wide q8 default clamped up to `mlx.minQualityTier`,
/// falling back q8 → bf16 → q4 to a tier that is installed (`tier_resolver::preferred_tier` and
/// `standard_tier_subdir`). `None` when the request is not such an image, or no tier is installed
/// on this host, which keeps the catalog default download.
fn implicit_standard_tier_variant(
    state: &AppState,
    job_type: &JobType,
    slot: &str,
    entry: &Value,
) -> Option<String> {
    if slot != "modelManifestEntry"
        || !matches!(
            job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        )
    {
        return None;
    }
    let model_id = entry.get("id").and_then(Value::as_str)?;
    let builtin = sceneworks_core::registry::builtin_model_spec(model_id);
    let mlx = builtin
        .and_then(|spec| spec.product.fields.get("mlx"))
        .or_else(|| entry.get("mlx"));
    if mlx
        .and_then(|mlx| mlx.get("standardTierLayout"))
        .and_then(Value::as_bool)
        != Some(true)
    {
        return None;
    }
    let rank = |tier: &str| match tier {
        "bf16" => 3,
        "q8" => 2,
        "q4" => 1,
        _ => 0,
    };
    let preferred = entry
        .pointer("/mlx/minQualityTier")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|floor| rank(floor) > rank("q8"))
        .unwrap_or("q8");
    let installed = crate::models::installed_variant_keys(entry, &state.settings.data_dir);
    [preferred, "q8", "bf16", "q4"]
        .into_iter()
        .find(|tier| installed.iter().any(|variant| variant == tier))
        .map(str::to_owned)
}

// The slot capture receives one explicit value for each independent source policy dimension.
#[allow(clippy::too_many_arguments)]
async fn capture_model_slot(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
    slot: &str,
    entry: &Value,
    backend: &str,
    platform: &str,
    authored_hash: &str,
    owns_request_overlays: bool,
    selected_sources: &[SelectedRouteSource],
    source_claims: &[PortableSourceClaim],
    frozen_model: Option<&PortableModelBinding>,
) -> Result<PortableModelBinding, ApiError> {
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| incomplete(format!("{slot} has no canonical model id")))?;
    let training = training_base_selection(job_type, payload, model_id)?;
    let requested_variant =
        slot_requested_variant(state, job_type, payload, slot, entry, training.as_ref());
    let selected = selected_portable_model_artifact_closure(
        entry,
        backend,
        platform,
        requested_variant.as_deref(),
        payload,
    )
    .map_err(incomplete)?;
    let downloads = selected
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download))
        .filter(|download| {
            training.is_none()
                || !model_id.starts_with("ltx")
                || !is_co_requisite_download(download)
        })
        .collect::<Vec<_>>();
    let utility_local_selection = if matches!(
        job_type,
        JobType::DatasetAnalysis | JobType::TrainingCaption
    ) && slot == "modelManifestEntries/0"
    {
        payload
            .get("modelNameOrPath")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|selection| !selection.is_empty())
            .filter(|selection| {
                crate::model_sources::app_managed_utility_path_exists(
                    &state.settings.data_dir,
                    selection,
                ) || !downloads.iter().any(|download| {
                    !is_co_requisite_download(download)
                        && download.get("repo").and_then(Value::as_str) == Some(*selection)
                })
            })
            .map(|selection| selected_utility_model_path(state, selection))
            .transpose()?
    } else {
        None
    };
    let mut members = Vec::new();
    let variant;
    let tier;
    let local_plan_primary = entry.pointer("/importPlan/checkpointId").is_some()
        && downloads
            .iter()
            .all(|download| is_co_requisite_download(download));
    if downloads.is_empty() || local_plan_primary {
        if frozen_model.is_none() {
            let registered = crate::models::resolve_model_manifest_entry(state, model_id).await?;
            if registered.as_object().is_none_or(JsonObject::is_empty)
                || [
                    "downloads",
                    "conditionalComponents",
                    "paths",
                    "modelPath",
                    "installedPath",
                    "components",
                    "importPlan",
                ]
                .into_iter()
                .any(|key| entry.get(key) != registered.get(key))
            {
                return Err(incomplete(format!(
                    "{slot} local source differs from server model registration"
                )));
            }
        }
        let mut local = capture_local_model_entry_offloaded(state, entry, slot).await?;
        if let Some(training) = &training {
            let selected = capture_local_source_files_offloaded(
                state,
                &training.path,
                &allowed_local_roots(state),
            )
            .await?;
            if local
                .iter()
                .find(|member| member.role == ArtifactMemberRole::Primary)
                .is_none_or(|primary| primary.files != selected)
            {
                return Err(incomplete(
                    "training plan base path differs from its verified imported source",
                ));
            }
        }
        if let Some(training) = &training {
            if let Some(primary) = local
                .iter_mut()
                .find(|member| member.role == ArtifactMemberRole::Primary)
            {
                primary.tier = Some(training.tier.clone());
            }
        }
        members.extend(local);
        // A managed imported checkpoint supplies its own transformer. Its manifest downloads
        // name the publisher components that the plan reads alongside that local primary.
        for (index, download) in downloads.iter().enumerate() {
            members.push(
                capture_download_member(
                    state,
                    download,
                    ArtifactMemberRole::CoRequisite,
                    Some(format!("{slot}/downloads/{index}")),
                )
                .await?,
            );
        }
        variant = requested_variant.unwrap_or_else(|| "local".to_owned());
        tier = training.as_ref().map(|training| training.tier.clone());
    } else {
        let primary = downloads
            .iter()
            .find(|download| !is_co_requisite_download(download))
            .ok_or_else(|| incomplete(format!("{slot} has no selected primary download")))?;
        variant = primary
            .get("variant")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .to_owned();
        tier = Some(variant.clone());
        let produced_training = training.as_ref().is_some_and(|training| {
            let expected = state
                .settings
                .data_dir
                .join("models")
                .join("mlx")
                .join(model_id);
            matches!(
                (std::fs::canonicalize(&training.path), std::fs::canonicalize(expected)),
                (Ok(actual), Ok(expected)) if actual == expected
            )
        });
        if produced_training {
            verify_produced_local_source_offloaded(state, entry, &training.as_ref().unwrap().path)
                .await?;
        }
        for (index, download) in downloads.iter().enumerate() {
            let role = if is_co_requisite_download(download) {
                ArtifactMemberRole::CoRequisite
            } else {
                ArtifactMemberRole::Primary
            };
            if role == ArtifactMemberRole::Primary && utility_local_selection.is_some() {
                continue;
            }
            let component_id =
                (role != ArtifactMemberRole::Primary).then(|| format!("{slot}/downloads/{index}"));
            members.push(
                if role == ArtifactMemberRole::Primary && produced_training {
                    capture_local_member_offloaded(
                        state,
                        &training.as_ref().unwrap().path,
                        &allowed_local_roots(state),
                        role,
                        component_id,
                        tier.clone(),
                    )
                    .await?
                } else if role == ArtifactMemberRole::Primary
                    && *job_type == JobType::ModelConvert
                    && download.get("revision").is_none()
                {
                    capture_uncataloged_convert_source(state, payload, download, frozen_model)
                        .await?
                } else {
                    capture_download_member(state, download, role, component_id).await?
                },
            );
        }
        if !produced_training {
            if let Some(training) = &training {
                verify_training_primary_path(state, &training.path, primary, &members).await?;
            }
        }
    }
    if let Some(path) =
        selected_primary_override(payload, entry, owns_request_overlays, training.is_some())
    {
        let files =
            capture_local_source_files_offloaded(state, &path, &allowed_local_roots(state)).await?;
        let primary = members
            .iter_mut()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .ok_or_else(|| incomplete(format!("{slot} has no primary source to override")))?;
        if matches_pinned_closure(&files, &primary.files) {
            // This host can locate the exact publisher bytes under a different approved root.
            // The portable identity remains the trusted immutable revision.
        } else if (if path.is_file() {
            path.parent().unwrap_or(&path)
        } else {
            &path
        })
        .join(".sceneworks-produced-artifact.json")
        .is_file()
        {
            verify_produced_local_source_offloaded(state, entry, &path).await?;
            *primary = capture_local_member_offloaded(
                state,
                &path,
                &allowed_local_roots(state),
                ArtifactMemberRole::Primary,
                None,
                tier.clone(),
            )
            .await?;
        } else {
            return Err(incomplete(format!(
                "{slot} override is neither pinned publisher bytes nor a verified produced source"
            )));
        }
    }
    // The utility loader opens the top-level path. Capture that source before any superseded
    // catalog primary metadata lookup; publisher co-requisites above remain frozen separately.
    if let Some(selected) = utility_local_selection {
        let mut roots = allowed_local_roots(state);
        roots.push(
            sceneworks_core::hf_home::model_source_library(&state.settings.data_dir)
                .root()
                .to_path_buf(),
        );
        let local = capture_local_member_offloaded(
            state,
            &selected,
            &roots,
            ArtifactMemberRole::Primary,
            None,
            tier.clone(),
        )
        .await?;
        if let Some(primary) = members
            .iter_mut()
            .find(|member| member.role == ArtifactMemberRole::Primary)
        {
            *primary = local;
        } else {
            members.push(local);
        }
    }
    // The first model carrier owns selected request adapters and controls. Other carriers retain
    // their own component closures but never duplicate an adapter under a second logical slot.
    if owns_request_overlays {
        members.extend(capture_selected_request_overlays_offloaded(state, payload).await?);
    }
    for source in selected_sources
        .iter()
        .filter(|source| source.model_slot == slot)
    {
        let mut member = match &source.source {
            SelectedSourceLocation::ManifestDownload(download) => {
                capture_download_member(
                    state,
                    download,
                    source.role.clone(),
                    Some(source.component_id.clone()),
                )
                .await?
            }
            SelectedSourceLocation::LocalPath(path) => {
                capture_local_member_offloaded(
                    state,
                    Path::new(path),
                    &allowed_local_roots(state),
                    source.role.clone(),
                    Some(source.component_id.clone()),
                    source.selected_tier.clone(),
                )
                .await?
            }
            SelectedSourceLocation::SelectedTextEncoder { path, source_kind } => {
                capture_selected_text_encoder_member_offloaded(state, path, source_kind).await?
            }
            SelectedSourceLocation::OperatorCachedRepo(source_id) => {
                let claims = frozen_worker_claims(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    source_claims,
                )?;
                capture_worker_source_member(
                    state,
                    source_id,
                    source.role.clone(),
                    source.component_id.clone(),
                    &claims,
                )
                .await?
            }
            SelectedSourceLocation::PreferredWorkerClaim {
                preferred_id,
                fallback_id,
            } => {
                let selected_id = select_preferred_source_id(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    preferred_id,
                    fallback_id,
                    source_claims,
                )?;
                let claims = frozen_worker_claims(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    source_claims,
                )?;
                capture_worker_source_member(
                    state,
                    &selected_id,
                    source.role.clone(),
                    source.component_id.clone(),
                    &claims,
                )
                .await?
            }
            SelectedSourceLocation::ConversionBorrow {
                repo,
                preferred_subdir,
            } => {
                capture_conversion_borrow_member(
                    state,
                    repo,
                    preferred_subdir.as_deref(),
                    source.role.clone(),
                    source.component_id.clone(),
                    source_claims,
                    frozen_model,
                )
                .await?
            }
            SelectedSourceLocation::DynamicHfRepo(repo) => {
                let revision = match frozen_revision_for_source(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    repo,
                )? {
                    Some(revision) => revision,
                    None => resolve_dynamic_hf_revision(state, repo, "main").await?,
                };
                capture_dynamic_detail_file(
                    state,
                    repo,
                    &revision,
                    source.role.clone(),
                    source.component_id.clone(),
                )
                .await?
            }
            SelectedSourceLocation::DynamicHfRef {
                repo,
                reference,
                files,
            } => {
                let revision = match frozen_revision_for_source(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    repo,
                )? {
                    Some(revision) => revision,
                    None => resolve_dynamic_hf_revision(state, repo, reference).await?,
                };
                let download = serde_json::json!({
                    "provider":"huggingface",
                    "repo":repo,
                    "revision":revision,
                    "files":files
                });
                capture_download_member(
                    state,
                    &download,
                    source.role.clone(),
                    Some(source.component_id.clone()),
                )
                .await?
            }
            SelectedSourceLocation::DynamicHfFiles {
                repo,
                reference,
                files,
            } => {
                let revision = match frozen_revision_for_source(
                    frozen_model,
                    &source.role,
                    &source.component_id,
                    repo,
                )? {
                    Some(revision) => revision,
                    None => resolve_dynamic_hf_revision(state, repo, reference).await?,
                };
                let download = serde_json::json!({
                    "provider":"huggingface",
                    "repo":repo,
                    "revision":revision,
                    "files":files
                });
                capture_download_member(
                    state,
                    &download,
                    source.role.clone(),
                    Some(source.component_id.clone()),
                )
                .await?
            }
            SelectedSourceLocation::OperatorOverrideOrManifest { source_id, default } => {
                let frozen_source =
                    frozen_source_member(frozen_model, &source.role, &source.component_id)?;
                let use_worker_claim = match frozen_source.map(|member| &member.source) {
                    Some(PortableSourceIdentity::WorkerClaim { .. }) => true,
                    Some(PortableSourceIdentity::Revision { .. }) => false,
                    Some(_) => return Err(incomplete("replay auxiliary source kind changed")),
                    None => source_claims
                        .iter()
                        .any(|claim| claim.source_id == *source_id),
                };
                if use_worker_claim {
                    let claims = frozen_worker_claims(
                        frozen_model,
                        &source.role,
                        &source.component_id,
                        source_claims,
                    )?;
                    capture_worker_source_member(
                        state,
                        source_id,
                        source.role.clone(),
                        source.component_id.clone(),
                        &claims,
                    )
                    .await?
                } else {
                    capture_download_member(
                        state,
                        default,
                        source.role.clone(),
                        Some(source.component_id.clone()),
                    )
                    .await?
                }
            }
        };
        if let Some(tier) = &source.selected_tier {
            member.tier = Some(tier.clone());
        }
        member
            .validate()
            .map_err(|error| incomplete(error.to_string()))?;
        members.push(member);
    }
    let binding = PortableModelBinding::new(
        slot,
        model_id,
        backend,
        variant,
        tier,
        authored_hash,
        members,
    )
    .map_err(|error| incomplete(error.to_string()))?;
    let Some(checkpoint_id) = entry
        .pointer("/importPlan/checkpointId")
        .and_then(Value::as_str)
    else {
        return Ok(binding);
    };
    let resolved =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&state.settings.data_dir)
            .resolve(checkpoint_id)
            .map_err(|error| {
                incomplete(format!("imported checkpoint plan is unavailable: {error}"))
            })?;
    let prefix = format!("{slot}/importPlan/layers/");
    let missing = resolved
        .plan
        .layers
        .iter()
        .filter(|layer| {
            !binding.members.iter().any(|member| {
                member.component_id.as_deref()
                    == Some(format!("{prefix}{}", layer.layer_id).as_str())
            })
        })
        .map(|layer| layer.layer_id.clone())
        .collect::<Vec<_>>();
    if missing.len() > 1 {
        return Err(incomplete("checkpoint plan has missing frozen layers"));
    }
    let checkpoint_plan =
        PortableCheckpointPlan::from_import_plan(&resolved.plan, missing.into_iter().next())
            .map_err(|error| incomplete(error.to_string()))?;
    binding
        .with_checkpoint_plan(checkpoint_plan)
        .map_err(|error| incomplete(error.to_string()))
}

fn matches_pinned_closure(
    actual: &[PortableArtifactFile],
    pinned: &[PortableArtifactFile],
) -> bool {
    let Some(first) = pinned.first() else {
        return false;
    };
    let mut prefix = PathBuf::new();
    let mut candidates = vec![prefix.clone()];
    if let Some(parent) = first.relative_path.parent() {
        for component in parent.components() {
            prefix.push(component.as_os_str());
            candidates.push(prefix.clone());
        }
    }
    candidates.into_iter().any(|prefix| {
        let mut expected = Vec::with_capacity(pinned.len());
        for file in pinned {
            let Ok(relative) = file.relative_path.strip_prefix(&prefix) else {
                return false;
            };
            if relative.as_os_str().is_empty() {
                return false;
            }
            expected.push(PortableArtifactFile {
                relative_path: relative.to_path_buf(),
                size_bytes: file.size_bytes,
                sha256: file.sha256.clone(),
            });
        }
        expected.sort();
        expected == actual
    })
}

async fn capture_dynamic_detail_file(
    state: &AppState,
    repo: &str,
    revision: &str,
    role: ArtifactMemberRole,
    component_id: String,
) -> Result<PortableArtifactMember, ApiError> {
    let files =
        resolve_trusted_pinned_files(state, repo, revision, &["*.safetensors".to_owned()]).await?;
    let file = selected_detail_weight_file(&files)
        .ok_or_else(|| incomplete("custom Detail repository has no selected safetensors file"))?;
    let identity = ArtifactIdentity::pinned(repo, revision, "default")
        .map_err(|error| incomplete(error.to_string()))?;
    let member = PortableArtifactMember {
        role,
        component_id: Some(component_id),
        source: PortableSourceIdentity::Revision { identity },
        tier: Some("default".to_owned()),
        files: vec![file.clone()],
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

fn selected_detail_weight_file(files: &[PortableArtifactFile]) -> Option<&PortableArtifactFile> {
    files
        .iter()
        .filter(|file| {
            file.relative_path
                .extension()
                .is_some_and(|extension| extension == "safetensors")
                && file
                    .relative_path
                    .components()
                    .all(|component| !component.as_os_str().to_string_lossy().starts_with('.'))
        })
        .min_by(|a, b| a.relative_path.cmp(&b.relative_path))
}

fn frozen_source_member<'a>(
    model: Option<&'a PortableModelBinding>,
    role: &ArtifactMemberRole,
    component_id: &str,
) -> Result<Option<&'a PortableArtifactMember>, ApiError> {
    model
        .map(|model| {
            model
                .members
                .iter()
                .find(|member| {
                    &member.role == role && member.component_id.as_deref() == Some(component_id)
                })
                .ok_or_else(|| {
                    incomplete(format!(
                        "replay source component {component_id} was not frozen"
                    ))
                })
        })
        .transpose()
}

fn frozen_revision_for_source(
    model: Option<&PortableModelBinding>,
    role: &ArtifactMemberRole,
    component_id: &str,
    repo: &str,
) -> Result<Option<String>, ApiError> {
    let Some(member) = frozen_source_member(model, role, component_id)? else {
        return Ok(None);
    };
    match &member.source {
        PortableSourceIdentity::Revision { identity } if identity.repository == repo => {
            Ok(Some(identity.revision.clone()))
        }
        _ => Err(incomplete(format!(
            "replay source {component_id} changed repository or identity kind"
        ))),
    }
}

fn frozen_worker_claims(
    model: Option<&PortableModelBinding>,
    role: &ArtifactMemberRole,
    component_id: &str,
    claims: &[PortableSourceClaim],
) -> Result<Vec<PortableSourceClaim>, ApiError> {
    let Some(member) = frozen_source_member(model, role, component_id)? else {
        return Ok(claims.to_vec());
    };
    if !matches!(member.source, PortableSourceIdentity::WorkerClaim { .. }) {
        return Err(incomplete(format!(
            "replay source {component_id} is no longer a worker claim"
        )));
    }
    let matching = claims
        .iter()
        .filter(|claim| claim.matches_worker_claim_member(member).is_ok())
        .cloned()
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(incomplete(format!(
            "replay source {component_id} has no matching live worker claim"
        )));
    }
    Ok(matching)
}

fn select_preferred_source_id(
    frozen_model: Option<&PortableModelBinding>,
    role: &ArtifactMemberRole,
    component_id: &str,
    preferred_id: &str,
    fallback_id: &str,
    claims: &[PortableSourceClaim],
) -> Result<String, ApiError> {
    let frozen_source = frozen_source_member(frozen_model, role, component_id)?;
    match frozen_source.map(|member| &member.source) {
        Some(PortableSourceIdentity::WorkerClaim { source_id, .. })
            if source_id == preferred_id || source_id == fallback_id =>
        {
            Ok(source_id.clone())
        }
        Some(_) => Err(incomplete(format!(
            "replay source {component_id} changed preferred identity"
        ))),
        None if claims.iter().any(|claim| claim.source_id == preferred_id) => {
            Ok(preferred_id.to_owned())
        }
        None => Ok(fallback_id.to_owned()),
    }
}

/// Resolve a mutable, user-selected repository to a publisher commit once, at admission.
/// The worker sees only the immutable revision and exact file ledger in its frozen route.
async fn resolve_dynamic_hf_revision(
    state: &AppState,
    repo: &str,
    reference: &str,
) -> Result<String, ApiError> {
    #[cfg(test)]
    if let Some(fixture) = *state.test_hooks.dynamic_hf_revision_override.lock() {
        if let Some(revision) = fixture(repo, reference) {
            sceneworks_core::model_artifacts::validate_immutable_revision(revision)
                .map_err(|error| incomplete(error.to_string()))?;
            return Ok(revision.to_owned());
        }
    }
    let base_url = huggingface_base_url(state);
    #[cfg(test)]
    if !test_network::is_loopback(&base_url) {
        return Err(incomplete(format!(
            "{repo}@{reference}: unit tests never resolve a mutable revision over the network; \
             set test_hooks.dynamic_hf_revision_override"
        )));
    }
    let token = huggingface_token(state)?;
    resolve_dynamic_hf_revision_with_client(
        &state.http_client,
        &base_url,
        token.as_deref(),
        repo,
        reference,
    )
    .await
}

async fn resolve_dynamic_hf_revision_with_client(
    client: &reqwest::Client,
    base_url: &str,
    token: Option<&str>,
    repo: &str,
    reference: &str,
) -> Result<String, ApiError> {
    let repo = sceneworks_core::hf_repo_id::namespaced(repo)
        .map_err(|error| incomplete(format!("custom model repository is invalid: {error}")))?;
    let reference_parts = reference.split('/').collect::<Vec<_>>();
    if reference_parts.is_empty()
        || reference_parts.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(incomplete("custom model reference is invalid"));
    }
    let mut url = reqwest::Url::parse(base_url)
        .map_err(|error| incomplete(format!("Hugging Face endpoint is invalid: {error}")))?;
    {
        let mut parts = url
            .path_segments_mut()
            .map_err(|_| incomplete("Hugging Face endpoint cannot carry model paths"))?;
        parts.pop_if_empty();
        parts.push("api").push("models");
        for segment in repo.split('/') {
            parts.push(segment);
        }
        parts.push("revision").push(reference);
    }
    let expected_origin = url.origin();
    let mut request = client.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| incomplete(format!("custom model revision is unavailable: {error}")))?;
    if response.url().origin() != expected_origin {
        return Err(incomplete(
            "custom model revision metadata left the authorized endpoint",
        ));
    }
    if !response.status().is_success() {
        return Err(incomplete(format!(
            "custom model revision lookup failed with status {}",
            response.status()
        )));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| incomplete(format!("custom model revision response failed: {error}")))?
    {
        if bytes.len().saturating_add(chunk.len()) > 1_048_576 {
            return Err(incomplete("custom model revision response exceeds one MiB"));
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        incomplete(format!(
            "custom model revision response is invalid: {error}"
        ))
    })?;
    let revision = value
        .get("sha")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("custom model revision response has no commit SHA"))?;
    sceneworks_core::model_artifacts::validate_immutable_revision(revision)
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(revision.to_owned())
}

fn capture_local_member(
    cache: &VerifiedSourceCache,
    path: &Path,
    allowed_roots: &[PathBuf],
    role: ArtifactMemberRole,
    component_id: Option<String>,
    tier: Option<String>,
) -> Result<PortableArtifactMember, ApiError> {
    let files = capture_local_source_files(cache, path, allowed_roots)?;
    let digest = portable_closure_sha256(&files).map_err(|error| incomplete(error.to_string()))?;
    let member = PortableArtifactMember {
        role,
        component_id,
        source: PortableSourceIdentity::Digest { sha256: digest },
        tier,
        files,
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

async fn capture_local_member_offloaded(
    state: &AppState,
    path: &Path,
    allowed_roots: &[PathBuf],
    role: ArtifactMemberRole,
    component_id: Option<String>,
    tier: Option<String>,
) -> Result<PortableArtifactMember, ApiError> {
    let cache = verified_source_cache(state);
    let path = path.to_path_buf();
    let allowed_roots = allowed_roots.to_vec();
    blocking_source_capture(move || {
        capture_local_member(&cache, &path, &allowed_roots, role, component_id, tier)
    })
    .await
}

fn capture_selected_text_encoder_member(
    state: &AppState,
    path: &str,
    source_kind: &str,
) -> Result<PortableArtifactMember, ApiError> {
    let selected = Path::new(path);
    let roots = allowed_local_roots(state);
    let cache = verified_source_cache(state);
    let mut files = match source_kind {
        "directory" if selected.is_dir() => capture_local_source_files(&cache, selected, &roots)?,
        "file" if selected.is_file() => {
            // A File encoder reads its sibling behavior config. Its tokenizer comes from the
            // already-bound base; a complete snapshot instead carries its own tokenizer below.
            let config = selected
                .parent()
                .ok_or_else(|| incomplete("selected text encoder has no parent"))?
                .join("config.json");
            let mut files = capture_local_source_files(&cache, selected, &roots)?;
            files.extend(capture_local_source_files(&cache, &config, &roots)?);
            files
        }
        _ => return Err(incomplete("selected text encoder source kind changed")),
    };
    files.sort();
    files.dedup();
    let digest = portable_closure_sha256(&files).map_err(|error| incomplete(error.to_string()))?;
    let member = PortableArtifactMember {
        role: ArtifactMemberRole::OptionalComponent,
        component_id: Some("selected_text_encoder".to_owned()),
        source: PortableSourceIdentity::Digest { sha256: digest },
        tier: None,
        files,
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

async fn capture_selected_text_encoder_member_offloaded(
    state: &AppState,
    path: &str,
    source_kind: &str,
) -> Result<PortableArtifactMember, ApiError> {
    let state = state.clone();
    let path = path.to_owned();
    let source_kind = source_kind.to_owned();
    blocking_source_capture(move || {
        capture_selected_text_encoder_member(&state, &path, &source_kind)
    })
    .await
}

async fn capture_worker_source_member(
    state: &AppState,
    source_id: &str,
    role: ArtifactMemberRole,
    component_id: String,
    claims: &[PortableSourceClaim],
) -> Result<PortableArtifactMember, ApiError> {
    let mut selected = claims.iter().filter(|claim| claim.source_id == source_id);
    let claim = selected.next().ok_or_else(|| {
        incomplete(format!(
            "selected source {source_id} has no fresh worker claim"
        ))
    })?;
    claim
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    if selected.any(|other| other != claim) {
        return Err(incomplete(format!(
            "selected source {source_id} has ambiguous worker identities"
        )));
    }
    match &claim.provenance {
        PortableSourceProvenance::PinnedRepo {
            repository,
            revision,
        } => {
            let shipped = sceneworks_core::control_weights::shipped_auxiliary_source(source_id)
                .ok_or_else(|| {
                    incomplete(format!(
                        "selected source {source_id} has no server-owned pinned declaration"
                    ))
                })?;
            if repository != shipped.repo || revision != shipped.revision {
                return Err(incomplete(format!(
                    "selected source {source_id} differs from the pinned declaration"
                )));
            }
            verify_pinned_worker_claim(state, claim, shipped.files).await?;
        }
        PortableSourceProvenance::OperatorOverride => {
            if source_id == "sdxl_detail_tile_control"
                || (source_id != "ltx_amoral_gemma_3_12b"
                    && source_id != "ltx_training_gemma_override"
                    && source_id != "ltx_training_text_encoder"
                    && sceneworks_core::control_weights::shipped_auxiliary_source(source_id)
                        .is_none())
            {
                return Err(incomplete(format!(
                    "selected source {source_id} has no authorized operator override"
                )));
            }
        }
    }
    let member = PortableArtifactMember {
        role,
        component_id: Some(component_id),
        source: PortableSourceIdentity::WorkerClaim {
            source_id: claim.source_id.clone(),
            provenance: claim.provenance.clone(),
            closure_sha256: claim.closure_sha256.clone(),
        },
        tier: None,
        files: claim.files.clone(),
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

/// The worker claim uses paths relative to its selected file/directory. Rebase that exact ledger
/// under the immutable publisher subtree and require every claimed file and declared loader file
/// to match trusted metadata. Extra files cannot silently inherit the publisher identity.
async fn verify_pinned_worker_claim(
    state: &AppState,
    claim: &PortableSourceClaim,
    required_paths: &[&str],
) -> Result<(), ApiError> {
    let PortableSourceProvenance::PinnedRepo {
        repository,
        revision,
    } = &claim.provenance
    else {
        return Err(incomplete(
            "pinned worker claim has no publisher provenance",
        ));
    };
    let parent = common_parent(required_paths)?;
    let mut selectors = Vec::with_capacity(claim.files.len());
    for file in &claim.files {
        let candidate = parent.join(&file.relative_path);
        let path = candidate
            .to_str()
            .ok_or_else(|| incomplete("worker source file path is not portable UTF-8"))?;
        selectors.push(path.to_owned());
    }
    for required in required_paths {
        if !selectors.iter().any(|path| {
            sceneworks_core::model_artifacts::artifact_selection::pattern_matches(required, path)
        }) {
            return Err(incomplete(format!(
                "pinned worker source is missing required file {required}"
            )));
        }
    }
    let remote = resolve_trusted_pinned_files(state, repository, revision, &selectors).await?;
    let actual = selectors
        .iter()
        .zip(&claim.files)
        .map(|(path, file)| PortableArtifactFile {
            relative_path: PathBuf::from(path),
            size_bytes: file.size_bytes,
            sha256: file.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let mut actual = actual;
    actual.sort();
    if actual != remote {
        return Err(incomplete(
            "worker source bytes differ from the pinned publisher",
        ));
    }
    Ok(())
}

/// Select the exact installed FLUX.2 Klein base tier the converter will borrow. Claims are
/// authenticated worker inventory, but their hashes still need publisher proof for this
/// request-selected repository and immutable revision. A different worker may claim the job only
/// if it advertises the same frozen source ID and file ledger.
async fn capture_conversion_borrow_member(
    state: &AppState,
    repo: &str,
    preferred_subdir: Option<&str>,
    role: ArtifactMemberRole,
    component_id: String,
    claims: &[PortableSourceClaim],
    frozen_model: Option<&PortableModelBinding>,
) -> Result<PortableArtifactMember, ApiError> {
    let repo = sceneworks_core::hf_repo_id::namespaced(repo)
        .map_err(|error| incomplete(format!("conversion base repository is invalid: {error}")))?;
    let tiers = if let Some(preferred) = preferred_subdir.filter(|tier| !tier.is_empty()) {
        let mut tiers = vec![preferred];
        for fallback in ["bf16", "q8", "q4"] {
            if !tiers.contains(&fallback) {
                tiers.push(fallback);
            }
        }
        tiers
    } else {
        vec!["root"]
    };
    let allowed = |claim: &PortableSourceClaim| {
        let (claim_repo, revision, tier) =
            sceneworks_core::portable_sources::parse_flux2_convert_base_claim_id(&claim.source_id)?;
        let PortableSourceProvenance::PinnedRepo {
            repository,
            revision: provenance_revision,
        } = &claim.provenance
        else {
            return None;
        };
        (claim_repo == repo && repository == repo && revision == provenance_revision)
            .then(|| tiers.iter().position(|candidate| *candidate == tier))
            .flatten()
    };
    let chosen = if let Some(member) = frozen_source_member(frozen_model, &role, &component_id)? {
        claims
            .iter()
            .find(|claim| {
                allowed(claim).is_some() && claim.matches_worker_claim_member(member).is_ok()
            })
            .ok_or_else(|| incomplete("frozen conversion base has no matching live worker claim"))?
    } else {
        let mut candidates = claims
            .iter()
            .filter_map(|claim| allowed(claim).map(|priority| (claim, priority)))
            .collect::<Vec<_>>();
        candidates.sort_by(|(left, left_priority), (right, right_priority)| {
            left_priority
                .cmp(right_priority)
                .then_with(|| left.source_id.cmp(&right.source_id))
                .then_with(|| left.closure_sha256.cmp(&right.closure_sha256))
        });
        candidates
            .first()
            .map(|(claim, _)| *claim)
            .ok_or_else(|| incomplete("conversion base has no fresh installed tier claim"))?
    };
    chosen
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    let (_, revision, tier) =
        sceneworks_core::portable_sources::parse_flux2_convert_base_claim_id(&chosen.source_id)
            .ok_or_else(|| incomplete("conversion base worker claim ID is invalid"))?;
    verify_conversion_borrow_claim(state, chosen, repo, revision, tier).await?;
    let member = PortableArtifactMember {
        role,
        component_id: Some(component_id),
        source: PortableSourceIdentity::WorkerClaim {
            source_id: chosen.source_id.clone(),
            provenance: chosen.provenance.clone(),
            closure_sha256: chosen.closure_sha256.clone(),
        },
        tier: Some(tier.to_owned()),
        files: chosen.files.clone(),
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

async fn verify_conversion_borrow_claim(
    state: &AppState,
    claim: &PortableSourceClaim,
    repo: &str,
    revision: &str,
    tier: &str,
) -> Result<(), ApiError> {
    let prefix = if tier == "root" {
        String::new()
    } else {
        format!("{tier}/")
    };
    let selectors =
        ["tokenizer/*", "text_encoder/*", "vae/*"].map(|selector| format!("{prefix}{selector}"));
    let remote = resolve_trusted_pinned_files(state, repo, revision, &selectors).await?;
    let mut normalized = remote
        .into_iter()
        .map(|mut file| {
            file.relative_path = file
                .relative_path
                .strip_prefix(&prefix)
                .map_err(|_| incomplete("conversion base file escaped selected tier"))?
                .to_path_buf();
            Ok(file)
        })
        .collect::<Result<Vec<PortableArtifactFile>, ApiError>>()?;
    normalized.sort();
    if normalized != claim.files {
        return Err(incomplete(
            "conversion base worker bytes differ from selected pinned tier",
        ));
    }
    Ok(())
}

fn common_parent(paths: &[&str]) -> Result<PathBuf, ApiError> {
    let first = paths
        .first()
        .ok_or_else(|| incomplete("pinned source declares no required files"))?;
    let mut parent = Path::new(first)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    for path in &paths[1..] {
        while !Path::new(path).starts_with(&parent) {
            if !parent.pop() {
                break;
            }
        }
    }
    Ok(parent)
}

/// Resolve the utility loader's authored model selection under an app-managed root.
fn selected_utility_model_path(state: &AppState, selection: &str) -> Result<PathBuf, ApiError> {
    // Match the worker's repository-first resolution. A missing repository snapshot must
    // never be reinterpreted as a local relative directory.
    let parts = selection.split('/').collect::<Vec<_>>();
    let repository = parts.len() == 2
        && parts
            .iter()
            .all(|part| !part.is_empty() && *part != "." && *part != ".." && !part.contains('\\'));
    if repository
        && !crate::model_sources::app_managed_utility_path_exists(
            &state.settings.data_dir,
            selection,
        )
    {
        let library = sceneworks_core::hf_home::model_source_library(&state.settings.data_dir);
        let resolver = ModelArtifactResolver::new(library);
        return resolver
            .discover_source_snapshot(selection, None)
            .map(|(_, snapshot)| snapshot)
            .map_err(|error| {
                incomplete(format!("selected utility snapshot is unavailable: {error}"))
            });
    }
    let path = PathBuf::from(selection);
    let path = if path.is_absolute() {
        path
    } else {
        state.settings.data_dir.join(path)
    };
    if !path.is_dir() {
        return Err(incomplete(
            "selected utility model must be an installed snapshot directory",
        ));
    }
    Ok(path)
}

/// Loader precedence for a request-owned source override. The chosen path is captured as a
/// content digest before enqueue, rather than silently retaining the catalog's HF primary.
fn selected_primary_override(
    payload: &JsonObject,
    entry: &Value,
    owns_request_overlays: bool,
    is_training_base: bool,
) -> Option<PathBuf> {
    if is_training_base {
        return None;
    }
    owns_request_overlays
        .then(|| payload.get("advanced").and_then(Value::as_object))
        .flatten()
        .and_then(|advanced| advanced.get("modelPath"))
        .or_else(|| entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

/// An uncataloged conversion source is intentionally accepted by the existing API. Resolve its
/// mutable HEAD once and freeze the complete publisher tree that the conversion worker may read.
/// Replay uses the original revision, so an unrelated HEAD move cannot replace or reject it.
async fn capture_uncataloged_convert_source(
    state: &AppState,
    payload: &JsonObject,
    download: &Value,
    frozen_model: Option<&PortableModelBinding>,
) -> Result<PortableArtifactMember, ApiError> {
    let repo = required_download_string(download, "repo")?;
    let revision = if let Some(model) = frozen_model {
        let primary = model
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .ok_or_else(|| incomplete("frozen conversion has no primary source"))?;
        match &primary.source {
            PortableSourceIdentity::Revision { identity } if identity.repository == repo => {
                identity.revision.clone()
            }
            _ => return Err(incomplete("frozen conversion repository has changed")),
        }
    } else {
        resolve_dynamic_hf_revision(state, repo, "main").await?
    };
    let converter = payload
        .get("converter")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let source_file = payload.get("sourceFile").and_then(Value::as_str);
    let selectors =
        sceneworks_core::portable_convert_sources::portable_convert_primary_file_selectors(
            converter,
            source_file,
        )
        .map_err(incomplete)?;
    let pinned = serde_json::json!({
        "provider":"huggingface",
        "repo":repo,
        "revision":revision,
        "files":selectors
    });
    capture_download_member(state, &pinned, ArtifactMemberRole::Primary, None).await
}

async fn capture_download_member(
    state: &AppState,
    download: &Value,
    role: ArtifactMemberRole,
    component_id: Option<String>,
) -> Result<PortableArtifactMember, ApiError> {
    let repo = required_download_string(download, "repo")?;
    let revision = required_download_string(download, "revision")?;
    let variant = download
        .get("variant")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let identity = ArtifactIdentity::pinned(repo, revision, variant)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut selectors = download
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| incomplete(format!("{repo} has no file selectors")))?
        .iter()
        .map(|file| {
            file.as_str()
                .filter(|file| !file.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| incomplete(format!("{repo} has an invalid file selector")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if selectors.is_empty() {
        // The model catalog's legacy `files: []` means the selected pinned snapshot as a whole.
        // Resolve that wildcard now; an empty portable closure would leave loader-visible bytes
        // outside the frozen identity.
        selectors.push("*".to_owned());
    }
    let files = resolve_trusted_pinned_files(state, repo, revision, &selectors).await?;
    let local = {
        let state = state.clone();
        let repo = repo.to_owned();
        let revision = revision.to_owned();
        let variant = variant.to_owned();
        let selectors = selectors.clone();
        let pinned = files.clone();
        blocking_source_capture(move || {
            capture_receipted_local_files(&state, &repo, &revision, &variant, &selectors, &pinned)
        })
        .await?
    };
    if let Some(local) = local {
        if local != files {
            return Err(incomplete(format!(
                "installed {repo}@{revision} files differ from the pinned source"
            )));
        }
    }
    let member = PortableArtifactMember {
        role,
        component_id,
        source: PortableSourceIdentity::Revision { identity },
        tier: Some(variant.to_owned()),
        files,
    };
    member
        .validate()
        .map_err(|error| incomplete(error.to_string()))?;
    Ok(member)
}

/// A completed local receipt has no content hashes, so it never substitutes for the immutable
/// upstream file proof. Successful pinned metadata is cacheable: a commit and concrete file
/// selectors cannot change without an upstream integrity failure. It is cached for this process
/// and persisted as the host's trusted pinned-file ledger, so an installed source admitted once
/// online is admitted again after a restart without network; the installed bytes are still
/// rehashed against that ledger by the caller. A revision this host never recorded is refused
/// with a typed error when upstream is unreachable.
async fn resolve_trusted_pinned_files(
    state: &AppState,
    repo: &str,
    revision: &str,
    selectors: &[String],
) -> Result<Vec<PortableArtifactFile>, ApiError> {
    #[cfg(test)]
    {
        let gate = state.test_hooks.pinned_metadata_gate.lock().clone();
        if let Some(mut gate) = gate {
            let _ = gate.wait_for(|open| *open).await;
        }
    }
    #[cfg(test)]
    if let Some(fixture) = state.test_hooks.pinned_metadata_override.lock().clone() {
        return Ok(fixture(repo, revision, selectors));
    }
    let base_url = huggingface_base_url(state);
    let key = format!("{base_url}\0{repo}\0{revision}\0{}", selectors.join("\0"));
    let cache = PINNED_METADATA_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(files) = cache
        .lock()
        .map_err(|_| ApiError::internal("Pinned metadata cache is poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(files);
    }
    let ledger = TrustedPinnedLedger {
        schema_version: TRUSTED_PINNED_LEDGER_VERSION,
        base_url: base_url.clone(),
        repo: repo.to_owned(),
        revision: revision.to_owned(),
        selectors: selectors.to_vec(),
        files: Vec::new(),
    };
    let ledger_path = trusted_pinned_ledger_path(&state.settings.data_dir, &key);
    let recorded = {
        let (path, ledger) = (ledger_path.clone(), ledger.clone());
        blocking_source_capture(move || Ok(read_trusted_pinned_ledger(&path, &ledger))).await?
    };
    let files = if let Some(files) = recorded {
        files
    } else {
        let files = fetch_pinned_hf_files(state, &base_url, repo, revision, selectors)
            .await
            .map_err(|error| {
                incomplete(format!(
                    "{repo}@{revision} has no trusted file ledger on this host and its pinned \
                     metadata could not be fetched: {error}"
                ))
            })?;
        let ledger = TrustedPinnedLedger {
            files: files.clone(),
            ..ledger
        };
        blocking_source_capture(move || {
            if let Err(error) = write_trusted_pinned_ledger(&ledger_path, &ledger) {
                // The process cache still serves this run; only offline reuse after a restart
                // is lost, and that refuses with a typed error rather than trusting less.
                tracing::warn!(%error, "trusted pinned-file ledger could not be persisted");
            }
            Ok(())
        })
        .await?;
        files
    };
    let mut cache = cache
        .lock()
        .map_err(|_| ApiError::internal("Pinned metadata cache is poisoned"))?;
    if cache.len() >= MAX_PINNED_METADATA_CACHE_ENTRIES {
        cache.pop_first();
    }
    cache.insert(key, files.clone());
    Ok(files)
}

/// The one upstream request for a pinned revision's file ledger (tree listing plus remote hashes
/// of non-LFS files).
async fn fetch_pinned_hf_files(
    state: &AppState,
    base_url: &str,
    repo: &str,
    revision: &str,
    selectors: &[String],
) -> Result<Vec<PortableArtifactFile>, String> {
    #[cfg(test)]
    if let Some(remote) = state.test_hooks.pinned_metadata_remote.lock().clone() {
        return remote(repo, revision, selectors);
    }
    #[cfg(test)]
    if !test_network::is_loopback(base_url) {
        return test_network::offline_pinned_files(repo, revision, selectors);
    }
    let credential = huggingface_token(state).map_err(|error| error.detail)?;
    resolve_pinned_hf_files(
        &state.http_client,
        base_url,
        credential.as_deref(),
        repo,
        revision,
        selectors,
    )
    .await
    .map_err(|error| error.to_string())
}

/// Unit tests never reach Hugging Face. A test that needs exact upstream metadata installs a
/// `pinned_metadata_remote`/`pinned_metadata_override` hook or a loopback stub server
/// (`SCENEWORKS_HUGGINGFACE_BASE_URL`); every other admission gets a deterministic offline
/// ledger built from the request's own selectors, so a suite run is independent of the network,
/// its latency and its rate limits.
#[cfg(test)]
pub(crate) mod test_network {
    use sceneworks_core::model_artifacts::portable_binding::PortableArtifactFile;
    use sha2::{Digest, Sha256};

    pub(crate) fn is_loopback(base_url: &str) -> bool {
        reqwest::Url::parse(base_url).is_ok_and(|url| {
            matches!(
                url.host_str(),
                Some("localhost" | "127.0.0.1" | "[::1]" | "::1")
            )
        })
    }

    /// One concrete file per selector: the selector itself, or for a glob one name that matches
    /// it. Content identity is a digest of the repository, revision and path, so every run and
    /// every host admits the same frozen ledger.
    pub(crate) fn offline_pinned_files(
        repo: &str,
        revision: &str,
        selectors: &[String],
    ) -> Result<Vec<PortableArtifactFile>, String> {
        let mut files = std::collections::BTreeMap::new();
        for selector in selectors {
            let concrete = selector.replace('*', "fixture").replace('?', "x");
            let matches = glob::Pattern::new(selector)
                .map_err(|error| format!("offline fixture selector {selector}: {error}"))?
                .matches(&concrete);
            if !matches {
                return Err(format!(
                    "offline fixture cannot name a file for selector {selector}; install a \
                     pinned_metadata_override"
                ));
            }
            let digest = Sha256::digest(format!("{repo}@{revision}/{concrete}").as_bytes());
            files.insert(
                concrete.clone(),
                PortableArtifactFile {
                    relative_path: concrete.into(),
                    size_bytes: 1 + u64::from(digest[0]),
                    sha256: format!("sha256:{digest:x}"),
                },
            );
        }
        Ok(files.into_values().collect())
    }
}

/// The Hugging Face endpoint for pinned metadata and mutable-revision resolution.
fn huggingface_base_url(state: &AppState) -> String {
    #[cfg(test)]
    {
        state
            .test_hooks
            .huggingface_base_url
            .lock()
            .clone()
            .unwrap_or_else(|| "https://huggingface.co".to_owned())
    }
    #[cfg(not(test))]
    {
        let _ = state;
        std::env::var("SCENEWORKS_HUGGINGFACE_BASE_URL")
            .unwrap_or_else(|_| "https://huggingface.co".to_owned())
    }
}

fn required_download_string<'a>(download: &'a Value, key: &str) -> Result<&'a str, ApiError> {
    download
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| incomplete(format!("selected download has no {key}")))
}

fn huggingface_token(state: &AppState) -> Result<Option<String>, ApiError> {
    let credential =
        sceneworks_core::credentials::CredentialFileStore::new(&state.settings.credentials_dir)
            .load()
            .map_err(|error| {
                incomplete(format!(
                    "Hugging Face credential store is unavailable: {error}"
                ))
            })?
            .remove("huggingface.co")
            .map(|entry| entry.token);
    Ok(credential.or_else(|| std::env::var("HF_TOKEN").ok()))
}

/// Prefer a completed, exact install receipt on this host. The receipt enumerates the concrete
/// files for a pinned revision; each file is verified against its `pinned` ledger entry through
/// the host's hash-once record, so an unchanged install is not read again while a stale size or a
/// substituted file (any stamp change) is rehashed and never inherits the old identity. Without
/// such a receipt, admission consults pinned HF metadata.
fn capture_receipted_local_files(
    state: &AppState,
    repo: &str,
    revision: &str,
    variant: &str,
    selectors: &[String],
    pinned: &[PortableArtifactFile],
) -> Result<Option<Vec<PortableArtifactFile>>, ApiError> {
    let managed = state
        .settings
        .data_dir
        .join("models")
        .join(safe_download_dir(repo))
        .join(".sceneworks-download-complete.json");
    let body = match std::fs::read_to_string(managed) {
        Ok(body) => body,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(incomplete(format!(
                "model install receipt is unreadable: {error}"
            )))
        }
    };
    let receipt: Value = serde_json::from_str(&body)
        .map_err(|error| incomplete(format!("model install receipt is invalid: {error}")))?;
    let rows = receipt
        .get("receipts")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_else(|| std::slice::from_ref(&receipt));
    let matching = rows.iter().find(|row| {
        row.get("repo").and_then(Value::as_str) == Some(repo)
            && row.get("snapshotRevision").and_then(Value::as_str) == Some(revision)
            && row
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("default")
                == variant
    });
    let Some(row) = matching else { return Ok(None) };
    let receipt_files = row
        .get("resolvedFiles")
        .and_then(Value::as_array)
        .ok_or_else(|| incomplete("completed model receipt has no resolved files"))?
        .iter()
        .map(|file| {
            file.as_str()
                .map(str::to_owned)
                .ok_or_else(|| incomplete("completed model receipt contains a non-string file"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut selected = BTreeSet::new();
    for selector in selectors {
        let matched = receipt_files
            .iter()
            .filter(|path| {
                sceneworks_core::model_artifacts::artifact_selection::pattern_matches(
                    selector, path,
                )
            })
            .collect::<Vec<_>>();
        if matched.is_empty() {
            return Ok(None);
        }
        selected.extend(matched.into_iter().cloned());
    }
    if selected.is_empty() {
        return Ok(None);
    }
    let library = sceneworks_core::hf_home::model_source_library(&state.settings.data_dir);
    let allowed = vec![library.root().to_path_buf()];
    let resolver = ModelArtifactResolver::new(library);
    let (_, root) = match resolver.discover_source_snapshot(repo, Some(revision)) {
        Ok(snapshot) => snapshot,
        Err(_) => return Ok(None),
    };
    let cache = verified_source_cache(state);
    let mut files = Vec::with_capacity(selected.len());
    for relative in selected {
        let path = confine_artifact_path(&root.join(&relative), &allowed)
            .map_err(|error| incomplete(error.to_string()))?;
        let metadata = std::fs::metadata(&path)
            .map_err(|error| incomplete(format!("receipted model file is unavailable: {error}")))?;
        if !metadata.is_file() {
            return Err(incomplete("receipted model source is not a regular file"));
        }
        let cannot_hash =
            |error| incomplete(format!("receipted model file cannot be hashed: {error}"));
        let expected = pinned
            .iter()
            .find(|file| file.relative_path == Path::new(&relative));
        let sha256 = match expected {
            Some(expected) => match cache.verify(&path, &expected.sha256, &no_cancellation) {
                Ok(verification) => {
                    note_source_verification(verification);
                    expected.sha256.clone()
                }
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                    note_source_verification(SourceVerification::Hashed);
                    return Err(incomplete(format!(
                        "installed {repo}@{revision} files differ from the pinned source"
                    )));
                }
                Err(error) => return Err(cannot_hash(error)),
            },
            // A file the ledger does not name fails the caller's comparison; hash it as before.
            None => {
                let (sha256, verification) =
                    cache.sha256(&path, &no_cancellation).map_err(cannot_hash)?;
                note_source_verification(verification);
                sha256
            }
        };
        files.push(PortableArtifactFile {
            relative_path: PathBuf::from(relative),
            size_bytes: metadata.len(),
            sha256,
        });
    }
    Ok(Some(files))
}

/// A non-HF imported, external, or trained entry is bound from its actual loader locators. A
/// provenance-only `source.path` is deliberately omitted: the loader never opens that field.
pub(crate) fn capture_local_model_entry(
    state: &AppState,
    entry: &Value,
    slot: &str,
) -> Result<Vec<PortableArtifactMember>, ApiError> {
    let mut locators = Vec::<(String, PathBuf)>::new();
    let mut allowed_roots = allowed_local_roots(state);
    for key in ["modelPath", "installedPath"] {
        if let Some(path) = entry.get(key).and_then(Value::as_str) {
            locators.push((key.to_owned(), PathBuf::from(path)));
        }
    }
    if let Some(paths) = entry.get("paths").and_then(Value::as_object) {
        for (key, path) in paths {
            if let Some(path) = path.as_str() {
                locators.push((format!("paths/{key}"), PathBuf::from(path)));
            }
        }
    }
    if let Some(components) = entry.get("components").and_then(Value::as_array) {
        for (index, component) in components.iter().enumerate() {
            if let Some(path) = component.get("path").and_then(Value::as_str) {
                locators.push((format!("components/{index}"), PathBuf::from(path)));
            }
        }
    }
    if let Some(checkpoint_id) = entry
        .pointer("/importPlan/checkpointId")
        .and_then(Value::as_str)
    {
        // The plan store verifies its trusted record, root grants, every source fingerprint and
        // layer content before exposing these paths. A linked checkpoint may live under a root
        // that is not the app data dir or a general external-model root.
        let resolved = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
            &state.settings.data_dir,
        )
        .resolve(checkpoint_id)
        .map_err(|error| incomplete(format!("imported checkpoint plan is unavailable: {error}")))?;
        if resolved.layers.is_empty() {
            return Err(incomplete("imported checkpoint plan has no source layers"));
        }
        for layer in resolved.layers {
            if let Some(parent) = layer.path.parent() {
                allowed_roots.push(parent.to_path_buf());
            }
            locators.push((
                format!("importPlan/layers/{}", layer.layer.layer_id),
                layer.path,
            ));
        }
    }
    if locators.is_empty() {
        return Err(incomplete(format!(
            "{slot} has no pinned download or loadable local source"
        )));
    }
    // The loader prefers an explicit modelPath, then installedPath, then paths.model. Keep all
    // other declared component locators in the closure so an overlay cannot clear one of them.
    locators.sort_by_key(|(key, _)| {
        if key == "modelPath" {
            0
        } else if key == "installedPath" {
            1
        } else if key == "paths/model" {
            2
        } else {
            3
        }
    });
    if entry.pointer("/importPlan/checkpointId").is_none() {
        let model_id = entry
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| incomplete("local model entry has no registered id"))?;
        let builtin = crate::models::embedded_builtin_catalog_entry(|builtin| {
            builtin.get("id").and_then(Value::as_str) == Some(model_id)
        })?
        .is_some();
        let declared_training_provenance = entry
            .pointer("/provenance/trainingJobId")
            .and_then(Value::as_str)
            .is_some();
        if builtin || declared_training_provenance {
            // A publisher builtin or a claimed trained product must prove its producer. A
            // separately registered user model is authenticated by its server-owned catalog
            // record plus the exact Digest bytes frozen below and worker availability claims.
            verify_produced_local_source(state, entry, &locators[0].1)?;
        }
    }
    let cache = verified_source_cache(state);
    let mut members = Vec::new();
    for (index, (key, path)) in locators.into_iter().enumerate() {
        let mut files = capture_local_source_files(&cache, &path, &allowed_roots)?;
        if key.starts_with("importPlan/layers/") {
            if files.len() != 1 {
                return Err(incomplete("checkpoint plan layer is not one exact file"));
            }
            // The worker can own identical bytes under another filename. The private plan's
            // targetPath supplies the loader layout; this logical claim names only the bytes.
            files[0].relative_path = PathBuf::from("artifact");
        }
        let digest =
            portable_closure_sha256(&files).map_err(|error| incomplete(error.to_string()))?;
        let member = PortableArtifactMember {
            role: if index == 0 {
                ArtifactMemberRole::Primary
            } else {
                ArtifactMemberRole::RequiredComponent
            },
            component_id: (index != 0).then(|| format!("{slot}/{key}")),
            source: PortableSourceIdentity::Digest { sha256: digest },
            tier: None,
            files,
        };
        member
            .validate()
            .map_err(|error| incomplete(error.to_string()))?;
        members.push(member);
    }
    Ok(members)
}

async fn capture_local_model_entry_offloaded(
    state: &AppState,
    entry: &Value,
    slot: &str,
) -> Result<Vec<PortableArtifactMember>, ApiError> {
    let state = state.clone();
    let entry = entry.clone();
    let slot = slot.to_owned();
    blocking_source_capture(move || capture_local_model_entry(&state, &entry, &slot)).await
}

/// Conversion and trained full-base roots become portable only after the producer has written a
/// strong receipt and the API can tie its IDs and output path to a completed source job.
fn verify_produced_local_source(
    state: &AppState,
    entry: &Value,
    source: &Path,
) -> Result<(), ApiError> {
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("local model entry has no id"))?;
    let output_model_id = if entry.get("family").and_then(Value::as_str)
        == Some(sceneworks_core::training::iris::IRIS_FAMILY)
        && entry
            .pointer("/trainedCheckpoint/task")
            .and_then(Value::as_str)
            == Some("generation")
        && entry
            .pointer("/trainedCheckpoint/engineModel")
            .and_then(Value::as_str)
            == Some(model_id)
    {
        entry
            .pointer("/trainedCheckpoint/modelId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .unwrap_or(model_id)
    } else {
        model_id
    };
    let root = if source.is_file() {
        source
            .parent()
            .ok_or_else(|| incomplete("local model file has no root"))?
    } else {
        source
    };
    let root = confine_artifact_path(root, &allowed_local_roots(state))
        .map_err(|error| incomplete(error.to_string()))?;
    let marker = root.join(".sceneworks-produced-artifact.json");
    let receipt: sceneworks_worker::ProducedArtifactReceipt =
        serde_json::from_slice(&std::fs::read(marker).map_err(|error| {
            incomplete(format!(
                "local model producer receipt is unavailable: {error}"
            ))
        })?)
        .map_err(|error| incomplete(format!("local model producer receipt is invalid: {error}")))?;
    if receipt.output_model_id != output_model_id {
        return Err(incomplete(
            "local model producer output id differs from catalog",
        ));
    }
    let producer = state
        .jobs_store
        .get_job(&receipt.producer_job_id)
        .map_err(|error| incomplete(format!("local model producer job is unavailable: {error}")))?;
    if producer.status != JobStatus::Completed || producer.completed_at.is_none() {
        return Err(incomplete("local model producer job is not complete"));
    }
    match producer.job_type {
        JobType::ModelConvert => {
            let expected = state
                .settings
                .data_dir
                .join("models")
                .join("mlx")
                .join(output_model_id);
            let expected = confine_artifact_path(&expected, &allowed_local_roots(state))
                .map_err(|error| incomplete(error.to_string()))?;
            if root != expected
                || producer.payload.get("modelId").and_then(Value::as_str) != Some(output_model_id)
                || producer.payload.get("sourceRepo").and_then(Value::as_str)
                    != Some(receipt.source_artifact_id.as_str())
            {
                return Err(incomplete(
                    "converted model receipt differs from completed source job",
                ));
            }
        }
        JobType::LoraTrain => {
            let plan: sceneworks_core::training::TrainingPlan = serde_json::from_value(
                producer
                    .payload
                    .get("plan")
                    .cloned()
                    .ok_or_else(|| incomplete("trained model producer has no plan"))?,
            )
            .map_err(|error| incomplete(format!("trained model plan is invalid: {error}")))?;
            let (output, _) =
                crate::training::resolve_finetune_output_location(state, output_model_id);
            let output = if entry.get("family").and_then(Value::as_str)
                == Some(sceneworks_core::training::iris::IRIS_FAMILY)
            {
                crate::iris_training::iris_full_export_dir(&output, &plan.output.file_name)
                    .ok_or_else(|| incomplete("Iris producer has no valid export path"))?
            } else {
                output
            };
            let output = confine_artifact_path(&output, &allowed_local_roots(state))
                .map_err(|error| incomplete(error.to_string()))?;
            if root != output
                || producer
                    .payload
                    .get("manifestEntry")
                    .and_then(|manifest| manifest.get("id"))
                    .and_then(Value::as_str)
                    != Some(output_model_id)
                || plan.target.base_model != receipt.source_artifact_id
                || entry
                    .pointer("/provenance/trainingJobId")
                    .and_then(Value::as_str)
                    != Some(producer.id.as_str())
            {
                return Err(incomplete(
                    "trained model receipt differs from completed source job",
                ));
            }
        }
        _ => {
            return Err(incomplete(
                "local model receipt names an unsupported producer job",
            ))
        }
    }
    sceneworks_worker::verify_produced_artifact_receipt(
        &root,
        &producer.id,
        &receipt.source_artifact_id,
        output_model_id,
    )
    .map_err(|error| incomplete(format!("local model producer bytes changed: {error}")))?;
    Ok(())
}

async fn verify_produced_local_source_offloaded(
    state: &AppState,
    entry: &Value,
    source: &Path,
) -> Result<(), ApiError> {
    let state = state.clone();
    let entry = entry.clone();
    let source = source.to_path_buf();
    blocking_source_capture(move || verify_produced_local_source(&state, &entry, &source)).await
}

fn allowed_local_roots(state: &AppState) -> Vec<PathBuf> {
    allowed_local_roots_for(
        &state.settings.data_dir,
        &state.settings.external_model_roots,
    )
}

fn allowed_local_roots_for(data_dir: &Path, external_model_roots: &[PathBuf]) -> Vec<PathBuf> {
    // The configured Hugging Face cache is a managed root in its own right: it
    // commonly lives outside `data_dir` (the default `~/.cache/huggingface/hub`),
    // and installed HF-backed LoRAs and non-default text encoders resolve their
    // installed paths into it. The worker's `allowed_model_roots` already
    // authorizes it; capture must agree or those installed sources are refused
    // as "outside every managed root" (SW-7030, review 4533400131).
    std::iter::once(data_dir.to_path_buf())
        .chain(std::iter::once(
            sceneworks_core::hf_home::model_source_library(data_dir)
                .root()
                .to_path_buf(),
        ))
        .chain(external_model_roots.iter().cloned())
        .collect()
}

fn capture_local_source_files(
    cache: &VerifiedSourceCache,
    source: &Path,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PortableArtifactFile>, ApiError> {
    let source = confine_artifact_path(source, allowed_roots)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut selected = Vec::new();
    collect_source_files(cache, &source, &source, allowed_roots, &mut selected, 0)?;
    if selected.is_empty() {
        return Err(incomplete(
            "selected local model source has no regular files",
        ));
    }
    selected.sort();
    Ok(selected)
}

async fn capture_local_source_files_offloaded(
    state: &AppState,
    source: &Path,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PortableArtifactFile>, ApiError> {
    let cache = verified_source_cache(state);
    let source = source.to_path_buf();
    let allowed_roots = allowed_roots.to_vec();
    blocking_source_capture(move || capture_local_source_files(&cache, &source, &allowed_roots))
        .await
}

fn collect_source_files(
    cache: &VerifiedSourceCache,
    root: &Path,
    path: &Path,
    allowed_roots: &[PathBuf],
    selected: &mut Vec<PortableArtifactFile>,
    depth: usize,
) -> Result<(), ApiError> {
    if depth > 32 || selected.len() >= 10_000 {
        return Err(incomplete(
            "local model source exceeds portable closure bounds",
        ));
    }
    // Confinement, directory reads and hashing resolve through the CANONICAL target; the
    // loader-visible relative name follows the link itself. Replacing `path` with its canonical
    // symlink target (the previous behavior) rebased each linked child onto the target's root,
    // so `strip_prefix(root)` rejected a valid linked file — a converted install's borrowed
    // text-encoder/VAE absolute symlinks — as "escaping its selected source". The two concerns
    // stay separate: a link must still confine to a managed root (a target outside every allowed
    // root is refused below), but its in-source name is what the loader opens, including inside
    // linked directories.
    let canonical = confine_artifact_path(path, allowed_roots)
        .map_err(|error| incomplete(error.to_string()))?;
    let metadata = std::fs::metadata(&canonical)
        .map_err(|error| incomplete(format!("local model source is unavailable: {error}")))?;
    if metadata.is_dir() {
        let mut children = std::fs::read_dir(&canonical)
            .map_err(|error| incomplete(format!("local model directory is unreadable: {error}")))?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| incomplete(format!("local model directory is unreadable: {error}")))?;
        children.sort();
        for child in children {
            collect_source_files(
                cache,
                root,
                &path.join(child),
                allowed_roots,
                selected,
                depth + 1,
            )?;
        }
        return Ok(());
    }
    if !metadata.is_file() {
        return Err(incomplete(
            "selected local model source is not a regular file",
        ));
    }
    let relative_path = if root == path {
        PathBuf::from(
            path.file_name()
                .ok_or_else(|| incomplete("local model file has no name"))?,
        )
    } else {
        path.strip_prefix(root)
            .map_err(|_| incomplete("local model file escapes its selected source"))?
            .to_path_buf()
    };
    if matches!(
        relative_path.to_str(),
        Some(
            ".sceneworks-artifact-provenance.json"
                | ".sceneworks-artifact-provenance.lock"
                | ".sceneworks-produced-artifact.json"
                | ".sceneworks-produced-artifact.lock"
        )
    ) {
        // Producer receipts attest the loader-visible content and exclude their own mutable
        // markers. Keep the portable byte ledger identical to that content scope.
        return Ok(());
    }
    let size_bytes = metadata.len();
    // Hashed once per host while the file's stamp is unchanged (shared with the worker's check).
    let (sha256, verification) = cache
        .sha256(&canonical, &no_cancellation)
        .map_err(|error| incomplete(format!("local model file cannot be hashed: {error}")))?;
    note_source_verification(verification);
    selected.push(PortableArtifactFile {
        relative_path,
        size_bytes,
        sha256,
    });
    Ok(())
}

fn capture_selected_request_overlays(
    state: &AppState,
    payload: &JsonObject,
) -> Result<Vec<PortableArtifactMember>, ApiError> {
    let mut members = Vec::new();
    let value = Value::Object(payload.clone());
    let allowed_roots = allowed_local_roots(state);
    let cache = verified_source_cache(state);
    for (slot, record) in enumerate_portable_array_slots(&value, &["loras", "lokrs", "controls"]) {
        if slot.starts_with("lokrs/") || slot.contains("/lokrs/") {
            return Err(incomplete(format!(
                "selected LoKr source {slot} has no registered native loader contract"
            )));
        }
        let role = if slot.contains("/controls/") || slot.starts_with("controls/") {
            ArtifactMemberRole::Control
        } else {
            ArtifactMemberRole::Lora
        };
        let locator = if role == ArtifactMemberRole::Lora {
            sceneworks_core::lora_family::adapter_record_path(record)
        } else {
            record
                .get("modelPath")
                .or_else(|| record.get("installedPath"))
                .or_else(|| record.get("path"))
                .and_then(Value::as_str)
                .map(PathBuf::from)
        };
        let Some(locator) = locator else {
            if role == ArtifactMemberRole::Control {
                // A control record without model source is conditioning input, not a weight.
                continue;
            }
            if record.get("index").is_some()
                && record.get("id").is_none()
                && record.get("path").is_none()
                && record.get("installedPath").is_none()
                && record.get("sourcePath").is_none()
            {
                // A phase can refer to an already selected top-level adapter by index. It opens
                // no additional source and remains bound by the authored options digest.
                continue;
            }
            return Err(incomplete(format!(
                "selected adapter {slot} has no loadable source"
            )));
        };
        let files = capture_local_source_files(&cache, &locator, &allowed_roots)?;
        let digest =
            portable_closure_sha256(&files).map_err(|error| incomplete(error.to_string()))?;
        let member = PortableArtifactMember {
            role,
            component_id: Some(slot),
            source: PortableSourceIdentity::Digest { sha256: digest },
            tier: None,
            files,
        };
        member
            .validate()
            .map_err(|error| incomplete(error.to_string()))?;
        members.push(member);
    }
    Ok(members)
}

async fn capture_selected_request_overlays_offloaded(
    state: &AppState,
    payload: &JsonObject,
) -> Result<Vec<PortableArtifactMember>, ApiError> {
    let state = state.clone();
    let payload = payload.clone();
    blocking_source_capture(move || capture_selected_request_overlays(&state, &payload)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verdict store outside the captured trees, so its records never join a closure.
    fn scratch_cache() -> (tempfile::TempDir, VerifiedSourceCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerifiedSourceCache::new(dir.path().join("verified"));
        (dir, cache)
    }
    use sceneworks_core::model_artifacts::portable_binding::model_free_job;
    use serde_json::json;

    /// SW-7031 AC4 / Codex 4541859826: presentation is not source identity. A rename, UI copy,
    /// defaults, limits, capabilities, tags, visibility, license display text or memory evidence
    /// change must still bind an unchanged historical source; the authored text encoder stays
    /// compared by its selection, not its host path.
    #[tokio::test]
    async fn historical_identity_ignores_presentation_but_not_source() {
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let old = json!({"modelManifestEntry":{
            "id":"example", "family":"flux", "type":"image", "catalogScope":"builtin",
            "name":"Example", "ui":{"label":"Example","description":"old copy"},
            "defaults":{"steps":20,"resolution":"1024x1024"}, "limits":{"count":[1,2]},
            "capabilities":["text_to_image"], "tags":["old"], "hidden":false,
            "licenseNotice":"old notice",
            "mlx":{"minMemoryGb":24},
            "downloads":[{"provider":"huggingface","repo":"owner/model","revision":"a".repeat(40),
                "files":["model.safetensors"],"estimatedSizeBytes":10,"footprint":{"peakMemoryBytes":1}}],
            "resolvedTextEncoder":{"selectionId":"encoder-a","sourceKind":"directory","path":"/root-a/te"}
        }})
        .as_object()
        .unwrap()
        .clone();
        let mut presentation = old.clone();
        {
            let entry = presentation.get_mut("modelManifestEntry").unwrap();
            entry["name"] = json!("Renamed example");
            entry["ui"] = json!({"label":"Renamed","description":"new copy"});
            entry["defaults"] = json!({"steps":8,"resolution":"768x768"});
            entry["limits"] = json!({"count":[1]});
            entry["capabilities"] = json!(["text_to_image", "image_to_image"]);
            entry["tags"] = json!(["new"]);
            entry["hidden"] = json!(true);
            entry["licenseNotice"] = json!("new notice");
            entry["mlx"] = json!({"minMemoryGb":32});
            entry["downloads"][0]["estimatedSizeBytes"] = json!(11);
            entry["downloads"][0]["footprint"] = json!({"peakMemoryBytes":2});
            entry["resolvedTextEncoder"]["path"] = json!("/root-b/te");
        }
        assert_eq!(
            historical_source_identity(&old["modelManifestEntry"]),
            historical_source_identity(&presentation["modelManifestEntry"])
        );
        validate_historical_source_carrier(&state, &old, &presentation)
            .await
            .expect("presentation-only edits keep the historical source");

        for (pointer, replacement) in [
            ("/downloads/0/revision", json!("b".repeat(40))),
            ("/downloads/0/files", json!(["other.safetensors"])),
            ("/resolvedTextEncoder/selectionId", json!("encoder-b")),
            ("/family", json!("other_family")),
            ("/baseModel", json!("replacement-base")),
        ] {
            let mut changed = old.clone();
            let entry = changed.get_mut("modelManifestEntry").unwrap();
            if let Some(slot) = entry.pointer_mut(pointer) {
                *slot = replacement;
            } else {
                entry[pointer.trim_start_matches('/')] = replacement;
            }
            assert!(
                validate_historical_source_carrier(&state, &old, &changed)
                    .await
                    .is_err(),
                "{pointer} is source identity"
            );
        }
        let mut dropped_encoder = old.clone();
        dropped_encoder
            .get_mut("modelManifestEntry")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("resolvedTextEncoder");
        assert!(
            validate_historical_source_carrier(&state, &old, &dropped_encoder)
                .await
                .is_err(),
            "an authored encoder must not be silently dropped"
        );
    }

    /// A builtin carrier stamped by the pre-registry code (memory evidence inline on the entry,
    /// older presentation) must bind against today's registry projection of the same pinned
    /// source. Its `paths` are install locators of that pinned download, not a local authority.
    #[tokio::test]
    async fn stamped_builtin_carrier_binds_against_the_current_registry_projection() {
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let mut current = json!({"model":"z_image_turbo","mode":"text_to_image","prompt":"p"})
            .as_object()
            .unwrap()
            .clone();
        crate::jobs::canonicalize_image_model_payload(
            &state,
            &JobType::ImageGenerate,
            &mut current,
        )
        .await
        .expect("builtin canonicalizes");
        assert!(
            current["modelManifestEntry"].get("paths").is_some(),
            "the shipped builtin carries install locators: {}",
            current["modelManifestEntry"]
        );
        validate_historical_source_carrier(&state, &current, &current)
            .await
            .expect("today's stamped builtin binds");
        let mut pre_registry = current.clone();
        {
            let entry = pre_registry.get_mut("modelManifestEntry").unwrap();
            entry["name"] = json!("Z-Image Turbo (old name)");
            entry["defaults"]["steps"] = json!(4);
            entry["mlx"]["minMemoryGb"] = json!(40);
            entry["mlx"]["calibrations"] = json!([{"abi":3}]);
            entry["candle"]["supportsSequentialOffload"] = json!(true);
            entry["downloads"][0]["footprint"]["residentMemoryBytes"] = json!(5776463496_u64);
        }
        validate_historical_source_carrier(&state, &pre_registry, &current)
            .await
            .expect("a pre-registry stamp of the same pinned builtin binds");
        let mut replaced = pre_registry.clone();
        replaced.get_mut("modelManifestEntry").unwrap()["downloads"][0]["revision"] =
            json!("c".repeat(40));
        assert!(
            validate_historical_source_carrier(&state, &replaced, &current)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn historical_carrier_allows_root_move_but_not_catalog_replacement() {
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let old = json!({"modelManifestEntry":{
            "id":"example", "family":"flux", "type":"image",
            "download":{"repo":"owner/model", "revision":"a".repeat(40)},
            "files":["model.safetensors"]
        }})
        .as_object()
        .unwrap()
        .clone();
        validate_historical_source_carrier(&state, &old, &old)
            .await
            .unwrap();
        let mut replaced = old.clone();
        replaced.get_mut("modelManifestEntry").unwrap()["download"]["revision"] =
            json!("b".repeat(40));
        assert!(validate_historical_source_carrier(&state, &old, &replaced)
            .await
            .is_err());
        let mut swapped_file = old.clone();
        swapped_file.get_mut("modelManifestEntry").unwrap()["download"]["path"] =
            json!("other-weights.safetensors");
        assert!(
            validate_historical_source_carrier(&state, &old, &swapped_file)
                .await
                .is_err()
        );
        let mut path_only = old.clone();
        path_only.get_mut("modelManifestEntry").unwrap()["download"] = Value::Null;
        path_only.get_mut("modelManifestEntry").unwrap()["paths"] =
            json!({"model":"/root-a/model.safetensors"});
        let mut moved = path_only.clone();
        moved.get_mut("modelManifestEntry").unwrap()["paths"] =
            json!({"model":"/root-b/model.safetensors"});
        assert_eq!(
            historical_source_identity(&path_only["modelManifestEntry"]),
            historical_source_identity(&moved["modelManifestEntry"])
        );
        assert!(
            validate_historical_source_carrier(&state, &path_only, &moved)
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn installed_hash_does_not_block_unrelated_request_task() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("model.safetensors");
        std::fs::write(&file, b"weights").unwrap();
        let expected = sceneworks_core::hashing::sha256_file(&file).unwrap();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        // The watchdog releases the hash even if someone accidentally moves it back onto the
        // current-thread scheduler. That regression then fails without hanging the test suite.
        let release = std::thread::spawn(move || {
            let progressed = progress_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_ok();
            let _ = release_tx.send(());
            progressed
        });
        let hash = tokio::spawn(async move {
            blocking_source_capture(move || {
                let _ = entered_tx.send(());
                release_rx.recv().unwrap();
                sceneworks_core::hashing::sha256_file(&file)
                    .map_err(|error| ApiError::internal(error.to_string()))
            })
            .await
        });
        entered_rx.await.unwrap();
        tokio::spawn(async move { progress_tx.send(()).unwrap() })
            .await
            .unwrap();
        assert_eq!(hash.await.unwrap().unwrap(), expected);
        assert!(
            release.join().unwrap(),
            "a blocking content hash stalled the Tokio request task"
        );
    }

    /// A fresh API process has an empty process cache. Model the restart for one repository
    /// without disturbing other tests' cache entries.
    fn forget_process_pinned_metadata(repo: &str) {
        let marker = format!("\0{repo}\0");
        if let Some(cache) = PINNED_METADATA_CACHE.get() {
            cache
                .lock()
                .unwrap()
                .retain(|key, _| !key.contains(&marker));
        }
    }

    /// Codex review of #3375 (finding A): an installed builtin admitted once online is admitted
    /// again after an API restart with Hugging Face unreachable. The trusted file ledger recorded
    /// from upstream is persisted; the installed bytes are still rehashed against it, so a
    /// substituted install is refused; and a revision this host never recorded is refused with a
    /// typed error, not admitted on the strength of a local receipt.
    #[tokio::test]
    async fn installed_pinned_source_is_admitted_offline_after_restart_from_the_trusted_ledger() {
        let _hf_cache = crate::tests::support::isolate_hf_cache();
        let root = tempfile::tempdir().unwrap();
        let settings = crate::tests::support::test_settings(&root);
        let (_, state) = crate::create_app_with_state(settings.clone()).expect("test API state");
        let repo = "owner/offline-ledger-fixture";
        let revision = "b".repeat(40);
        let relative = "model.safetensors";
        let snapshot =
            sceneworks_core::hf_home::huggingface_repo_cache_path(&state.settings.data_dir, repo)
                .unwrap()
                .join("snapshots")
                .join(&revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join(relative);
        std::fs::write(&path, b"trusted").unwrap();
        let receipt_dir = state
            .settings
            .data_dir
            .join("models")
            .join(safe_download_dir(repo));
        std::fs::create_dir_all(&receipt_dir).unwrap();
        std::fs::write(
            receipt_dir.join(".sceneworks-download-complete.json"),
            json!({"repo":repo,"snapshotRevision":revision,"variant":"default",
                "resolvedFiles":[relative]})
            .to_string(),
        )
        .unwrap();
        let trusted = PortableArtifactFile {
            relative_path: relative.into(),
            size_bytes: 7,
            sha256: format!(
                "sha256:{}",
                sceneworks_core::hashing::sha256_bytes(b"trusted")
            ),
        };
        let download = json!({"repo":repo,"revision":revision,"files":[relative]});

        // Online: upstream supplies the trusted ledger.
        let upstream = trusted.clone();
        *state.test_hooks.pinned_metadata_remote.lock() =
            Some(std::sync::Arc::new(move |_, _, _| {
                Ok(vec![upstream.clone()])
            }));
        let online = capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect("online admission records the trusted ledger");
        assert_eq!(online.files, vec![trusted.clone()]);
        drop(state);

        // Restart with the network unreachable: a new process state over the same data dir.
        forget_process_pinned_metadata(repo);
        let (_, state) = crate::create_app_with_state(settings).expect("restarted API state");
        let remote_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = remote_calls.clone();
        *state.test_hooks.pinned_metadata_remote.lock() =
            Some(std::sync::Arc::new(move |_, _, _| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err("network unreachable".to_owned())
            }));
        let offline = capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect("an installed source admitted before is admitted offline after a restart");
        assert_eq!(offline, online);
        assert_eq!(
            remote_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the persisted trusted ledger answers without a network request"
        );

        // The installed bytes are still verified against the ledger.
        std::fs::write(&path, b"changed").unwrap();
        let error = capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect_err("a substituted install is refused offline");
        assert!(error.detail.contains("files differ from the pinned source"));
        std::fs::write(&path, b"trusted").unwrap();

        // A revision with no recorded ledger is refused, typed and promptly, even when a local
        // receipt names it: a receipt carries no content hashes and is never the trust root.
        let unrecorded = "c".repeat(40);
        std::fs::write(
            receipt_dir.join(".sceneworks-download-complete.json"),
            json!({"repo":repo,"snapshotRevision":unrecorded,"variant":"default",
                "resolvedFiles":[relative]})
            .to_string(),
        )
        .unwrap();
        let error = capture_download_member(
            &state,
            &json!({"repo":repo,"revision":unrecorded,"files":[relative]}),
            ArtifactMemberRole::Primary,
            None,
        )
        .await
        .expect_err("no trusted ledger and no network refuses the source");
        assert_eq!(error.code, Some("model_artifact_incomplete"));
        assert!(
            error.detail.contains("network unreachable"),
            "{}",
            error.detail
        );
        assert_eq!(remote_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn receipted_pinned_file_is_rehashed_after_substitution() {
        let _hf_cache = crate::tests::support::isolate_hf_cache();
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let repo = "owner/receipt-fixture";
        let revision = "a".repeat(40);
        let relative = "model.safetensors";
        let snapshot =
            sceneworks_core::hf_home::huggingface_repo_cache_path(&state.settings.data_dir, repo)
                .unwrap()
                .join("snapshots")
                .join(&revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join(relative);
        std::fs::write(&path, b"trusted").unwrap();
        let receipt_dir = state
            .settings
            .data_dir
            .join("models")
            .join(safe_download_dir(repo));
        std::fs::create_dir_all(&receipt_dir).unwrap();
        std::fs::write(
            receipt_dir.join(".sceneworks-download-complete.json"),
            json!({"repo":repo,"snapshotRevision":revision,"variant":"default",
                "resolvedFiles":[relative]})
            .to_string(),
        )
        .unwrap();
        let trusted = PortableArtifactFile {
            relative_path: relative.into(),
            size_bytes: 7,
            sha256: format!(
                "sha256:{}",
                sceneworks_core::hashing::sha256_bytes(b"trusted")
            ),
        };
        *state.test_hooks.pinned_metadata_override.lock() =
            Some(std::sync::Arc::new(move |_, _, _| vec![trusted.clone()]));
        let download = json!({"repo":repo,"revision":revision,"files":[relative]});
        capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect("original installed bytes match the pinned source");
        std::fs::write(&path, b"changed").unwrap();
        let error = capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect_err("same-size substituted installed bytes are refused");
        assert!(error.detail.contains("files differ from the pinned source"));
    }

    /// Older than the verified-source stamp's racy window, so a recorded verdict may stand in
    /// for a hash.
    fn settle_mtime(path: &Path, modified: std::time::SystemTime) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
    }

    fn hashed_source_files() -> usize {
        HASHED_SOURCE_FILES.with(std::cell::Cell::get)
    }

    /// One receipted install of `owner/hash-once` whose pinned ledger names `bytes`.
    fn receipted_install(
        state: &AppState,
        bytes: &[u8],
    ) -> (PathBuf, Vec<PortableArtifactFile>, Value) {
        let repo = "owner/hash-once";
        let revision = "a".repeat(40);
        let relative = "model.safetensors";
        let snapshot =
            sceneworks_core::hf_home::huggingface_repo_cache_path(&state.settings.data_dir, repo)
                .unwrap()
                .join("snapshots")
                .join(&revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join(relative);
        std::fs::write(&path, bytes).unwrap();
        let receipt_dir = state
            .settings
            .data_dir
            .join("models")
            .join(safe_download_dir(repo));
        std::fs::create_dir_all(&receipt_dir).unwrap();
        std::fs::write(
            receipt_dir.join(".sceneworks-download-complete.json"),
            json!({"repo":repo,"snapshotRevision":revision,"variant":"default",
                "resolvedFiles":[relative]})
            .to_string(),
        )
        .unwrap();
        let pinned = vec![PortableArtifactFile {
            relative_path: relative.into(),
            size_bytes: bytes.len() as u64,
            sha256: format!("sha256:{}", sceneworks_core::hashing::sha256_bytes(bytes)),
        }];
        let download = json!({"repo":repo,"revision":revision,"files":[relative]});
        (path, pinned, download)
    }

    fn capture_receipted(
        state: &AppState,
        pinned: &[PortableArtifactFile],
    ) -> Result<Option<Vec<PortableArtifactFile>>, ApiError> {
        capture_receipted_local_files(
            state,
            "owner/hash-once",
            &"a".repeat(40),
            "default",
            &["model.safetensors".to_owned()],
            pinned,
        )
    }

    #[tokio::test]
    async fn receipted_install_is_hashed_once_while_its_stamp_is_unchanged() {
        let _hf_cache = crate::tests::support::isolate_hf_cache();
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let (path, pinned, download) = receipted_install(&state, b"trusted");
        let settled = std::time::SystemTime::now() - std::time::Duration::from_secs(10);
        settle_mtime(&path, settled);
        let start = hashed_source_files();
        assert_eq!(
            capture_receipted(&state, &pinned).unwrap(),
            Some(pinned.clone())
        );
        assert_eq!(
            hashed_source_files(),
            start + 1,
            "the first check reads the file"
        );
        assert_eq!(
            capture_receipted(&state, &pinned).unwrap(),
            Some(pinned.clone())
        );
        assert_eq!(
            hashed_source_files(),
            start + 1,
            "a second job creation reads no bytes of the unchanged install"
        );
        // The admission path that job creation runs reuses the same verdict end to end.
        let fixture = pinned.clone();
        *state.test_hooks.pinned_metadata_override.lock() =
            Some(std::sync::Arc::new(move |_, _, _| fixture.clone()));
        let member = capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
            .await
            .expect("the unchanged install is admitted");
        assert_eq!(member.files, pinned);

        // Same size and the same mtime, different bytes: the ctime/inode move, so the file is
        // hashed again and refused with the existing message. (Windows stamps carry no ctime.)
        if cfg!(unix) {
            std::fs::write(&path, b"changed").unwrap();
            settle_mtime(&path, settled);
            let before = hashed_source_files();
            let error = capture_receipted(&state, &pinned).expect_err("substitution is refused");
            assert!(
                error.detail.contains("files differ from the pinned source"),
                "{}",
                error.detail
            );
            assert_eq!(hashed_source_files(), before + 1);
            let error =
                capture_download_member(&state, &download, ArtifactMemberRole::Primary, None)
                    .await
                    .expect_err("substitution is refused at admission");
            assert!(error.detail.contains("files differ from the pinned source"));
        }
    }

    #[tokio::test]
    async fn a_worker_verdict_is_reused_by_api_admission() {
        let _hf_cache = crate::tests::support::isolate_hf_cache();
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let (path, pinned, _) = receipted_install(&state, b"trusted");
        settle_mtime(
            &path,
            std::time::SystemTime::now() - std::time::Duration::from_secs(10),
        );
        // The worker opens the host store with `VerifiedSourceCache::for_data_dir(data_dir)`.
        assert_eq!(
            VerifiedSourceCache::for_data_dir(&state.settings.data_dir)
                .verify(
                    &std::fs::canonicalize(&path).unwrap(),
                    &pinned[0].sha256,
                    &no_cancellation
                )
                .unwrap(),
            SourceVerification::Hashed
        );
        let start = hashed_source_files();
        assert_eq!(capture_receipted(&state, &pinned).unwrap(), Some(pinned));
        assert_eq!(
            hashed_source_files(),
            start,
            "admission reused the worker's verdict"
        );
    }

    #[test]
    fn local_source_capture_is_hashed_once_while_its_stamp_is_unchanged() {
        let root = tempfile::tempdir().unwrap();
        let (_verified, cache) = scratch_cache();
        let model = root.path().join("checkpoint");
        std::fs::create_dir_all(&model).unwrap();
        let weights = model.join("model.safetensors");
        std::fs::write(&weights, b"weights").unwrap();
        let settled = std::time::SystemTime::now() - std::time::Duration::from_secs(10);
        settle_mtime(&weights, settled);
        let roots = [root.path().to_path_buf()];
        let start = hashed_source_files();
        let first = capture_local_source_files(&cache, &model, &roots).unwrap();
        assert_eq!(hashed_source_files(), start + 1);
        assert_eq!(
            capture_local_source_files(&cache, &model, &roots).unwrap(),
            first
        );
        assert_eq!(hashed_source_files(), start + 1, "no bytes are read again");
        if cfg!(unix) {
            std::fs::write(&weights, b"WEIGHTS").unwrap();
            settle_mtime(&weights, settled);
            let changed = capture_local_source_files(&cache, &model, &roots).unwrap();
            assert_eq!(hashed_source_files(), start + 2);
            assert_ne!(changed, first, "the substituted bytes get their own digest");
        }
    }

    #[test]
    fn vector_child_auto_keeps_only_parent_frozen_worker_routes() {
        let routes = BTreeMap::from([(
            "candle:linux".to_owned(),
            PortableRouteBinding {
                backend: "candle".to_owned(),
                platform: "linux".to_owned(),
                models: Vec::new(),
            },
        )]);
        let eligible = ["candle:linux", "candle:windows", "mlx:macos"];
        let selected = eligible
            .into_iter()
            .filter(|key| {
                frozen_vector_child_route(&routes, key, "auto")
                    .unwrap()
                    .is_some()
            })
            .collect::<Vec<_>>();
        assert_eq!(selected, ["candle:linux"]);
        assert!(frozen_vector_child_route(&routes, "candle:windows", "candle").is_err());
    }

    #[test]
    fn conversion_claims_prefer_authored_tier_across_workers() {
        let repo = "fixture/convert-base";
        let revision = "a".repeat(40);
        let row = SelectedRouteSource {
            model_slot: "modelManifestEntries/0".into(),
            component_id: "flux2_convert_base".into(),
            role: ArtifactMemberRole::RequiredComponent,
            source: SelectedSourceLocation::ConversionBorrow {
                repo: repo.into(),
                preferred_subdir: Some("bf16".into()),
            },
            selected_tier: None,
        };
        let claim = |tier: &str| {
            PortableSourceClaim::new(
                sceneworks_core::portable_sources::flux2_convert_base_claim_id(
                    repo, &revision, tier,
                )
                .expect("source ID"),
                PortableSourceProvenance::PinnedRepo {
                    repository: repo.into(),
                    revision: revision.clone(),
                },
                vec![PortableArtifactFile {
                    relative_path: "vae/weights.safetensors".into(),
                    size_bytes: 7,
                    sha256: format!(
                        "sha256:{}",
                        sceneworks_core::hashing::sha256_bytes(b"fixture")
                    ),
                }],
            )
            .expect("source claim")
        };
        let q4_worker = vec![claim("q4")];
        let bf16_worker = vec![claim("bf16")];
        assert!(
            conversion_claim_tier_rank(std::slice::from_ref(&row), &bf16_worker)
                < conversion_claim_tier_rank(std::slice::from_ref(&row), &q4_worker)
        );
    }

    #[test]
    fn registered_single_file_digest_claim_uses_content_hash_and_exact_ledger() {
        let file = PortableArtifactFile {
            relative_path: PathBuf::from("weights.safetensors"),
            size_bytes: 4,
            sha256: format!("sha256:{}", "a".repeat(64)),
        };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "user_model",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            vec![PortableArtifactMember {
                role: ArtifactMemberRole::Primary,
                component_id: None,
                source: PortableSourceIdentity::Digest {
                    sha256: file.sha256.clone(),
                },
                tier: None,
                files: vec![file.clone()],
            }],
        )
        .unwrap();
        let route = PortableRouteBinding::new("mlx", "macos", vec![model]).unwrap();
        let claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model",
            "user_model",
            None,
            &portable_file_ledger_sha256(std::slice::from_ref(&file)).unwrap(),
        )
        .unwrap();
        let claim = PortableSourceClaim::new(
            claim_id,
            PortableSourceProvenance::OperatorOverride,
            vec![file],
        )
        .unwrap();
        assert_ne!(claim.closure_sha256, claim.files[0].sha256);
        assert!(replay_route_claims_match(
            &route,
            &JsonObject::new(),
            std::slice::from_ref(&claim)
        )
        .is_ok());
        let mut changed = claim;
        changed.files[0].size_bytes += 1;
        assert!(replay_route_claims_match(&route, &JsonObject::new(), &[changed]).is_err());
    }

    #[test]
    fn registered_pose_overlay_replay_requires_the_same_worker_claim() {
        let file = |name: &str, byte: &str| PortableArtifactFile {
            relative_path: PathBuf::from(name),
            size_bytes: 4,
            sha256: format!("sha256:{}", byte.repeat(64)),
        };
        let primary = file("model.safetensors", "a");
        let overlay = file("pose.safetensors", "b");
        let member =
            |role, component_id: Option<&str>, file: PortableArtifactFile| PortableArtifactMember {
                role,
                component_id: component_id.map(str::to_owned),
                source: PortableSourceIdentity::Digest {
                    sha256: file.sha256.clone(),
                },
                tier: None,
                files: vec![file],
            };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "krea_2_turbo",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&json!({})).unwrap(),
            vec![
                member(ArtifactMemberRole::Primary, None, primary.clone()),
                member(
                    ArtifactMemberRole::Control,
                    Some("strict_control"),
                    overlay.clone(),
                ),
            ],
        )
        .unwrap();
        let route = PortableRouteBinding::new("mlx", "macos", vec![model]).unwrap();
        let claim = |kind, id, file: PortableArtifactFile| {
            let ledger = portable_file_ledger_sha256(std::slice::from_ref(&file)).unwrap();
            PortableSourceClaim::new(
                sceneworks_core::portable_sources::registered_digest_claim_id(
                    kind, id, None, &ledger,
                )
                .unwrap(),
                PortableSourceProvenance::OperatorOverride,
                vec![file],
            )
            .unwrap()
        };
        let model_claim = claim("model", "krea_2_turbo", primary);
        let overlay_claim = claim("control", "pose-approved", overlay);
        let payload = json!({"advanced":{"controlWeights":{
            "overlayId":"pose-approved", "path":"/api-host/pose.safetensors",
            "_catalogAuthorized":true
        }}})
        .as_object()
        .unwrap()
        .clone();
        assert!(replay_route_claims_match(
            &route,
            &payload,
            &[model_claim.clone(), overlay_claim.clone()]
        )
        .is_ok());
        assert!(
            replay_route_claims_match(&route, &payload, std::slice::from_ref(&model_claim))
                .is_err()
        );
        let mut changed = payload.clone();
        changed["advanced"]["controlWeights"]["overlayId"] = json!("pose-other");
        assert!(
            replay_route_claims_match(&route, &changed, &[model_claim, overlay_claim]).is_err()
        );
    }

    #[test]
    fn frozen_replay_uses_original_immutable_dynamic_revision() {
        let old = "a".repeat(40);
        let file = PortableArtifactFile {
            relative_path: PathBuf::from("weights.safetensors"),
            size_bytes: 4,
            sha256: format!("sha256:{}", "b".repeat(64)),
        };
        let member = |role, component_id, repo: &str| PortableArtifactMember {
            role,
            component_id,
            source: PortableSourceIdentity::Revision {
                identity: ArtifactIdentity::pinned(repo, &old, "default").unwrap(),
            },
            tier: Some("default".to_owned()),
            files: vec![file.clone()],
        };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "model",
            "mlx",
            "default",
            Some("default".to_owned()),
            canonical_authored_options_sha256(&json!({})).unwrap(),
            vec![
                member(ArtifactMemberRole::Primary, None, "owner/base"),
                member(
                    ArtifactMemberRole::Control,
                    Some("dynamic_control".to_owned()),
                    "owner/control",
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            frozen_revision_for_source(
                Some(&model),
                &ArtifactMemberRole::Control,
                "dynamic_control",
                "owner/control"
            )
            .unwrap(),
            Some(old)
        );
        assert!(frozen_revision_for_source(
            Some(&model),
            &ArtifactMemberRole::Control,
            "dynamic_control",
            "owner/substitute"
        )
        .is_err());
    }

    #[test]
    fn local_produced_closure_excludes_mutable_receipt_files() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("weights.safetensors"), b"weight").unwrap();
        for marker in [
            ".sceneworks-artifact-provenance.json",
            ".sceneworks-artifact-provenance.lock",
            ".sceneworks-produced-artifact.json",
            ".sceneworks-produced-artifact.lock",
        ] {
            std::fs::write(root.path().join(marker), b"mutable").unwrap();
        }
        let (_verified, cache) = scratch_cache();
        let files =
            capture_local_source_files(&cache, root.path(), &[root.path().to_path_buf()]).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, PathBuf::from("weights.safetensors"));
    }

    #[cfg(unix)]
    #[test]
    fn linked_local_sources_keep_loader_names_and_stay_confined() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let install = root.path().join("models/mlx/anima_base/bf16");
        std::fs::create_dir_all(install.join("vae")).unwrap();
        std::fs::write(install.join("dit.safetensors"), b"packed dit").unwrap();
        // A converted install borrows its dense VAE and text encoder as absolute symlinks into a
        // managed root OUTSIDE the selected source (the HF cache under the same data dir).
        let borrowed = root.path().join("models/hf-cache/borrowed");
        std::fs::create_dir_all(borrowed.join("text_encoder")).unwrap();
        std::fs::write(borrowed.join("vae.safetensors"), b"borrowed vae").unwrap();
        std::fs::write(
            borrowed.join("text_encoder/model.safetensors"),
            b"borrowed text encoder",
        )
        .unwrap();
        symlink(
            borrowed.join("vae.safetensors"),
            install.join("vae/qwen_image_vae.safetensors"),
        )
        .unwrap();
        symlink(borrowed.join("text_encoder"), install.join("text_encoder")).unwrap();

        let digest = |path: std::path::PathBuf| {
            format!(
                "sha256:{}",
                sceneworks_core::hashing::sha256_file(&path).unwrap()
            )
        };
        // Linked files — and the contents of a linked directory — keep their loader-visible
        // in-source names, hashed through the link target's bytes.
        let expected = [
            (
                PathBuf::from("dit.safetensors"),
                std::fs::metadata(install.join("dit.safetensors"))
                    .unwrap()
                    .len(),
                digest(install.join("dit.safetensors")),
            ),
            (
                PathBuf::from("text_encoder/model.safetensors"),
                std::fs::metadata(borrowed.join("text_encoder/model.safetensors"))
                    .unwrap()
                    .len(),
                digest(borrowed.join("text_encoder/model.safetensors")),
            ),
            (
                PathBuf::from("vae/qwen_image_vae.safetensors"),
                std::fs::metadata(borrowed.join("vae.safetensors"))
                    .unwrap()
                    .len(),
                digest(borrowed.join("vae.safetensors")),
            ),
        ];
        let (_verified, cache) = scratch_cache();
        let actual = capture_local_source_files(&cache, &install, &[root.path().to_path_buf()])
            .unwrap()
            .into_iter()
            .map(|file| (file.relative_path, file.size_bytes, file.sha256))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        // A link whose target escapes every managed root stays refused before any read.
        let escape = tempfile::tempdir().unwrap();
        std::fs::write(escape.path().join("escape.safetensors"), b"outside").unwrap();
        symlink(
            escape.path().join("escape.safetensors"),
            install.join("escaped.safetensors"),
        )
        .unwrap();
        let error = capture_local_source_files(&cache, &install, &[root.path().to_path_buf()])
            .unwrap_err()
            .detail;
        assert!(
            error.contains("outside every managed root"),
            "escaping link must stay refused: {error}"
        );
    }

    /// SW-7033 Tier 2 (MLX flux_dev, seed 33111): an image request that names no tier ran the
    /// worker's implicit q8 default on main before the registry (q8 installed), but the binding
    /// froze the catalog's default DOWNLOAD (q4), so the same request rendered a different tier.
    /// The binding must freeze the tier the worker would run: q8 clamped up to the quality floor,
    /// falling back q8 → bf16 → q4 to an installed tier; the default download only when none is.
    #[tokio::test]
    async fn an_unselected_standard_tier_image_freezes_the_workers_implicit_tier() {
        let _hf_cache = crate::tests::support::isolate_hf_cache();
        let root = tempfile::tempdir().unwrap();
        let (_, state) = crate::create_app_with_state(crate::tests::support::test_settings(&root))
            .expect("test API state");
        let entry = Value::Object(
            sceneworks_core::builtin_manifests::builtin_model_entry("flux_dev")
                .expect("flux_dev is registered")
                .clone(),
        );
        let primary = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["variant"] == "q4")
            .expect("flux_dev q4 download");
        let repo = primary["repo"].as_str().unwrap();
        let revision = primary["revision"].as_str().unwrap();
        let install = |tier: &str| {
            let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(
                &state.settings.data_dir,
                repo,
            )
            .unwrap()
            .join("snapshots")
            .join(revision)
            .join(tier);
            for file in ["transformer/model.safetensors", "vae/model.safetensors"] {
                let path = snapshot.join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"weights").unwrap();
            }
        };
        let unselected = json!({"model":"flux_dev","mode":"text_to_image","prompt":"p"});
        let unselected = unselected.as_object().unwrap();
        let variant = |job_type: &JobType, payload: &JsonObject, slot: &str| {
            slot_requested_variant(&state, job_type, payload, slot, &entry, None)
        };
        let image = JobType::ImageGenerate;

        // Nothing installed: keep the catalog default download.
        assert_eq!(variant(&image, unselected, "modelManifestEntry"), None);
        install("q4");
        assert_eq!(
            variant(&image, unselected, "modelManifestEntry").as_deref(),
            Some("q4"),
            "only q4 installed: the worker's fallback lands on q4"
        );
        install("q8");
        assert_eq!(
            variant(&image, unselected, "modelManifestEntry").as_deref(),
            Some("q8"),
            "q8 installed: the implicit default is q8, not the q4 default download"
        );
        install("bf16");
        assert_eq!(
            variant(&image, unselected, "modelManifestEntry").as_deref(),
            Some("q8")
        );
        // An explicit tier is always the request's own.
        let explicit = json!({"model":"flux_dev","advanced":{"mlxQuantize":4}});
        assert_eq!(
            variant(&image, explicit.as_object().unwrap(), "modelManifestEntry").as_deref(),
            Some("q4")
        );
        // Only the primary image slot of an image job takes the implicit image default.
        assert_eq!(variant(&image, unselected, "baseModelManifestEntry"), None);
        assert_eq!(
            variant(&JobType::VideoGenerate, unselected, "modelManifestEntry"),
            None
        );
        // The frozen closure then names that tier's files only.
        let selected = selected_portable_model_artifact_closure(
            &entry,
            "mlx",
            "macos",
            variant(&image, unselected, "modelManifestEntry").as_deref(),
            unselected,
        )
        .unwrap();
        let primary_files = selected["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| !is_co_requisite_download(row))
            .flat_map(|row| row["files"].as_array().unwrap().clone())
            .collect::<Vec<_>>();
        assert_eq!(primary_files, vec![json!("q8/*")]);
    }

    #[test]
    fn allowed_local_roots_include_data_hf_cache_and_external_roots() {
        // SW-7030 (review 4533400131): the configured HF cache is a managed
        // capture root even when it lives outside the data dir.
        let data_dir = PathBuf::from("/srv/sceneworks/data");
        let external = vec![PathBuf::from("/mnt/comfyui/models")];
        let roots = allowed_local_roots_for(&data_dir, &external);
        assert!(roots.contains(&data_dir));
        assert!(roots.contains(
            &sceneworks_core::hf_home::model_source_library(&data_dir)
                .root()
                .to_path_buf()
        ));
        assert!(roots.contains(&PathBuf::from("/mnt/comfyui/models")));
    }

    #[test]
    fn authored_projection_keeps_choices_without_local_paths_or_credentials() {
        let payload = json!({
            "model":"flux_schnell", "prompt":"red kite", "variant":"q4",
            "baseSubdir":"bf16", "modelNameOrPath":"/host/private/checkpoint",
            "modelManifestEntry":{"id":"flux_schnell","paths":{"model":"/host/a"}},
            "advanced":{"modelPath":"/host/a","quantTier":"q4"},
            "loras":[{"id":"style","weight":0.6,"installedPath":"/host/loras/style.safetensors"}],
            "phase":{"loras":[{"id":"detail","source":{"path":"/host/b"}}]},
            "accessToken":"private"
        });
        let actual = portable_authored_options(payload.as_object().unwrap());
        assert_eq!(actual["model"], "flux_schnell");
        assert!(actual.get("prompt").is_none());
        assert_eq!(actual["advanced"]["quantTier"], "q4");
        assert_eq!(actual["baseSubdir"], "bf16");
        assert_eq!(actual["loras"][0]["id"], "style");
        assert_eq!(actual["phase"]["loras"][0]["id"], "detail");
        let encoded = actual.to_string();
        assert!(!encoded.contains("/host"));
        assert!(!encoded.contains("private"));
        assert!(!encoded.contains("modelManifestEntry"));
    }

    #[test]
    fn duplicate_model_ids_keep_distinct_carrier_slots() {
        let payload = json!({
            "modelManifestEntry":{"id":"same"},
            "baseModelManifestEntry":{"id":"same"},
            "modelManifestEntries":[{"id":"same"},{"id":"other"}]
        });
        let slots = payload_model_entries(payload.as_object().unwrap())
            .unwrap()
            .into_iter()
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>();
        assert_eq!(
            slots,
            [
                "modelManifestEntry",
                "baseModelManifestEntry",
                "modelManifestEntries/0",
                "modelManifestEntries/1"
            ]
        );
    }

    #[test]
    fn local_content_identity_is_root_independent_and_detects_substitution() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        for root in [first.path(), second.path()] {
            let model = root.join("checkpoint");
            std::fs::create_dir_all(&model).unwrap();
            std::fs::write(model.join("model.safetensors"), b"same weights").unwrap();
            std::fs::write(model.join("config.json"), b"same config").unwrap();
        }
        let (_verified, cache) = scratch_cache();
        let first_files = capture_local_source_files(
            &cache,
            &first.path().join("checkpoint"),
            &[first.path().to_path_buf()],
        )
        .unwrap();
        let second_files = capture_local_source_files(
            &cache,
            &second.path().join("checkpoint"),
            &[second.path().to_path_buf()],
        )
        .unwrap();
        assert_eq!(first_files, second_files);
        assert_eq!(
            portable_closure_sha256(&first_files).unwrap(),
            portable_closure_sha256(&second_files).unwrap()
        );
        std::fs::write(
            second.path().join("checkpoint/model.safetensors"),
            b"other weights",
        )
        .unwrap();
        let changed = capture_local_source_files(
            &cache,
            &second.path().join("checkpoint"),
            &[second.path().to_path_buf()],
        )
        .unwrap();
        assert_ne!(first_files, changed);
    }

    #[test]
    fn alternate_local_root_must_match_the_complete_pinned_tier() {
        let file = |path: &str, digest: &str| PortableArtifactFile {
            relative_path: PathBuf::from(path),
            size_bytes: 1,
            sha256: format!("sha256:{}", digest.repeat(64)),
        };
        let pinned = vec![
            file("bf16/config.json", "a"),
            file("bf16/model.safetensors", "b"),
        ];
        let alternate = vec![file("config.json", "a"), file("model.safetensors", "b")];
        assert!(matches_pinned_closure(&alternate, &pinned));
        assert!(!matches_pinned_closure(&alternate[..1], &pinned));
        let substituted = vec![file("config.json", "a"), file("model.safetensors", "c")];
        assert!(!matches_pinned_closure(&substituted, &pinned));
    }

    #[test]
    fn selected_overlay_keeps_choice_but_scrubs_local_locator() {
        let payload = json!({
            "advanced": {"controlWeights": {
                "overlayId":"pose-approved", "path":"/host/private/overlay.safetensors",
                "repo":"approved/pose", "revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }}
        });
        let portable = portable_authored_options(payload.as_object().unwrap());
        assert_eq!(
            portable["advanced"]["controlWeights"]["overlayId"],
            "pose-approved"
        );
        assert!(portable["advanced"]["controlWeights"].get("path").is_none());
        assert!(!portable.to_string().contains("/host/private"));
    }

    #[test]
    fn duplicate_prompt_change_keeps_source_choice_but_adapter_change_does_not() {
        let original = json!({
            "model":"flux_schnell", "prompt":"red kite", "seed":1,
            "loras":[{"id":"style-a","weight":0.4,"path":"/root/a"}],
            "workflow":{"id":"generated-1","rasterStage":{"model":"flux_schnell"}}
        });
        let changed_prompt = json!({
            "model":"flux_schnell", "prompt":"blue kite", "seed":2,
            "loras":[{"id":"style-a","weight":0.9,"path":"/other-root/a"}],
            "workflow":{"id":"generated-2","rasterStage":{"model":"flux_schnell"}}
        });
        assert_eq!(
            portable_authored_options(original.as_object().unwrap()),
            portable_authored_options(changed_prompt.as_object().unwrap())
        );
        let changed_adapter = json!({
            "model":"flux_schnell", "loras":[{"id":"style-b","path":"/root/b"}]
        });
        assert_ne!(
            portable_authored_options(original.as_object().unwrap()),
            portable_authored_options(changed_adapter.as_object().unwrap())
        );
    }

    #[test]
    fn duplicate_krea_convrot_change_requires_fresh_source_choice() {
        let ordinary = json!({
            "model":"krea_2_turbo", "prompt":"red kite",
            "advanced":{"convRot":false}
        });
        let changed_prompt = json!({
            "model":"krea_2_turbo", "prompt":"blue kite",
            "advanced":{"convRot":false}
        });
        let convrot = json!({
            "model":"krea_2_turbo", "prompt":"red kite",
            "advanced":{"convRot":true}
        });
        let source = portable_authored_options(ordinary.as_object().unwrap());
        assert_eq!(
            source,
            portable_authored_options(changed_prompt.as_object().unwrap())
        );
        assert_ne!(
            source,
            portable_authored_options(convrot.as_object().unwrap())
        );
    }

    #[test]
    fn duplicate_yue2_decoder_change_requires_fresh_source_choice() {
        let ordinary = json!({
            "model":"yue2", "prompt":"red kite",
            "yue2":{"kind":"create","lyrics":"la","computePolicy":"bf16",
                     "decoder":"legacy","runId":"yue2run_one"}
        });
        // A duplicate rewrites the prompt and mints a fresh run id: neither is a source choice.
        let changed_prompt = json!({
            "model":"yue2", "prompt":"blue kite",
            "yue2":{"kind":"create","lyrics":"la","computePolicy":"bf16",
                     "decoder":"legacy","runId":"yue2run_two"}
        });
        let decoder = json!({
            "model":"yue2", "prompt":"red kite",
            "yue2":{"kind":"create","lyrics":"la","computePolicy":"bf16",
                     "decoder":"standard","runId":"yue2run_one"}
        });
        let source = portable_authored_options(ordinary.as_object().unwrap());
        // Only the decoder is source-affecting inside the block (the worker's
        // `selected_portable_model_artifact_closure` reads it as the VAE choice).
        assert_eq!(source["yue2"], json!({"decoder":"legacy"}));
        assert_eq!(
            source,
            portable_authored_options(changed_prompt.as_object().unwrap())
        );
        assert_ne!(
            source,
            portable_authored_options(decoder.as_object().unwrap())
        );
    }

    #[test]
    fn pid_source_override_is_authored_without_physical_locator() {
        let original = json!({"advanced":{
            "usePid":true,
            "pidCheckpoint":{"repo":"owner/first","filename":"first.safetensors","path":"/private/a"},
            "pidGemma":"owner/gemma"
        }});
        let changed = json!({"advanced":{
            "usePid":true,
            "pidCheckpoint":{"repo":"owner/second","filename":"first.safetensors","path":"/private/b"},
            "pidGemma":"owner/gemma"
        }});
        let first = portable_authored_options(original.as_object().unwrap());
        let second = portable_authored_options(changed.as_object().unwrap());
        assert_eq!(first["advanced"]["pidCheckpoint"]["repo"], "owner/first");
        assert!(!first.to_string().contains("/private"));
        assert_ne!(first, second);
    }

    #[test]
    fn request_model_override_precedes_manifest_path_only_for_primary_slot() {
        let payload = json!({"advanced":{"modelPath":"/new/root/model"}});
        let entry = json!({"modelPath":"/old/root/model"});
        assert_eq!(
            selected_primary_override(payload.as_object().unwrap(), &entry, true, false),
            Some(PathBuf::from("/new/root/model"))
        );
        assert_eq!(
            selected_primary_override(payload.as_object().unwrap(), &entry, false, false),
            Some(PathBuf::from("/old/root/model"))
        );
        assert_eq!(
            selected_primary_override(payload.as_object().unwrap(), &entry, true, true),
            None,
            "a training plan's verified base path cannot be replaced by a generation override"
        );
    }

    #[test]
    fn training_plan_freezes_native_base_tier_instead_of_generation_default() {
        for (base, kernel, expected) in [
            ("z_image_turbo", "z_image_lora", "bf16"),
            ("ltx_2_3", "ltx_mlx_lora", "q4"),
            ("ltx_2_5", "ltx25_mlx_lora", "q4"),
        ] {
            let payload = json!({"plan":{"target":{
                "baseModel":base,"kernel":kernel,"baseModelPath":"/approved/snapshot"
            }}});
            let selected =
                training_base_selection(&JobType::LoraTrain, payload.as_object().unwrap(), base)
                    .unwrap()
                    .unwrap();
            assert_eq!(selected.tier, expected, "{base}");
        }
    }

    #[test]
    fn training_authored_choice_binds_target_and_repa_without_host_path() {
        let payload = json!({"plan":{
            "target":{
                "targetId":"iris_repa", "kernel":"iris_mlx_lora",
                "baseModel":"iris_3b", "baseModelPath":"/host/private/base"
            },
            "config":{"advanced":{"iris":{"repa":{"weight":0.5}}}}
        }});
        let options = portable_authored_options(payload.as_object().unwrap());
        assert_eq!(options["trainingBaseModel"], "iris_3b");
        assert_eq!(options["trainingKernel"], "iris_mlx_lora");
        assert_eq!(options["irisRepaSelected"], true);
        assert!(!options.to_string().contains("/host/private"));
    }

    #[test]
    fn catalog_analysis_requires_explicit_model_free_flags() {
        assert!(!model_free_job(
            &JobType::CatalogAnalysis,
            &JsonObject::new()
        ));
        let no_models = json!({
            "structuredAnalysisEnabled":false,
            "visionAnalysisEnabled":false,
            "semanticEmbeddingsEnabled":false
        });
        assert!(model_free_job(
            &JobType::CatalogAnalysis,
            no_models.as_object().unwrap()
        ));
    }

    #[test]
    fn preview_person_jobs_are_model_free_while_real_runs_freeze_bindings() {
        // A procedural preview (`preview: true`) runs the CPU utility placeholder and
        // loads no model bytes, so admission freezes no binding for it — even though the
        // request still carries the detector/segmenter catalog entries. The real run
        // keeps its binding.
        let preview = json!({"sourceAssetId":"asset", "preview":true});
        assert!(model_free_job(
            &JobType::PersonDetect,
            preview.as_object().unwrap()
        ));
        assert!(model_free_job(
            &JobType::PersonTrack,
            preview.as_object().unwrap()
        ));
        let real = json!({"sourceAssetId":"asset", "preview":false});
        assert!(!model_free_job(
            &JobType::PersonDetect,
            real.as_object().unwrap()
        ));
        assert!(!model_free_job(
            &JobType::PersonTrack,
            real.as_object().unwrap()
        ));
    }

    #[tokio::test]
    async fn mutable_custom_repo_head_is_frozen_to_each_immutable_file_closure() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let first = "a".repeat(40);
        let second = "b".repeat(40);
        let first_for_server = first.clone();
        let second_for_server = second.clone();
        let server = tokio::spawn(async move {
            for request_number in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 4096];
                let count = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..count]);
                let revision = if request_number < 2 {
                    &first_for_server
                } else {
                    &second_for_server
                };
                let body = if request.contains("/revision/main") {
                    serde_json::json!({"sha":revision}).to_string()
                } else {
                    assert!(request.contains(&format!("/tree/{revision}")));
                    let oid = if request_number < 2 {
                        "1".repeat(64)
                    } else {
                        "2".repeat(64)
                    };
                    serde_json::json!([{
                        "path":"diffusion_pytorch_model.safetensors",
                        "size":4,
                        "lfs":{"oid":oid}
                    }])
                    .to_string()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let client = reqwest::Client::new();
        let first_revision =
            resolve_dynamic_hf_revision_with_client(&client, &base, None, "owner/control", "main")
                .await
                .unwrap();
        let first_files = resolve_pinned_hf_files(
            &client,
            &base,
            None,
            "owner/control",
            &first_revision,
            &["*".to_owned()],
        )
        .await
        .unwrap();
        let second_revision =
            resolve_dynamic_hf_revision_with_client(&client, &base, None, "owner/control", "main")
                .await
                .unwrap();
        let second_files = resolve_pinned_hf_files(
            &client,
            &base,
            None,
            "owner/control",
            &second_revision,
            &["*".to_owned()],
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(first_revision, first);
        assert_eq!(second_revision, second);
        assert_ne!(
            selected_detail_weight_file(&first_files).unwrap(),
            selected_detail_weight_file(&second_files).unwrap()
        );
    }

    #[test]
    fn detail_custom_source_freezes_one_deterministic_visible_weight_file() {
        let file = |path: &str| PortableArtifactFile {
            relative_path: PathBuf::from(path),
            size_bytes: 4,
            sha256: format!("sha256:{}", "a".repeat(64)),
        };
        let files = [
            file("z/model.safetensors"),
            file("._sidecar.safetensors"),
            file("a/model.safetensors"),
            file("config.json"),
        ];
        assert_eq!(
            selected_detail_weight_file(&files).unwrap().relative_path,
            PathBuf::from("a/model.safetensors")
        );
    }

    #[tokio::test]
    async fn server_owned_kolors_ref_resolves_to_immutable_commit() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let expected = "c".repeat(40);
        let expected_for_server = expected.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..count]);
            assert!(request
                .contains("/api/models/Kwai-Kolors/Kolors-IP-Adapter-Plus/revision/refs%2Fpr%2F4"));
            let body = serde_json::json!({"sha":expected_for_server}).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let revision = resolve_dynamic_hf_revision_with_client(
            &reqwest::Client::new(),
            &base,
            None,
            "Kwai-Kolors/Kolors-IP-Adapter-Plus",
            "refs/pr/4",
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(revision, expected);
    }
}
