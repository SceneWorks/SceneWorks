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
    safe_download_dir, selected_model_artifact_closure,
};
use sceneworks_core::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, enumerate_portable_array_slots, portable_closure_sha256,
    resolve_pinned_hf_files, PortableArtifactFile, PortableArtifactMember, PortableJobBinding,
    PortableModelBinding, PortableRouteBinding, PortableSourceClaim, PortableSourceIdentity,
    PortableSourceProvenance,
};
use sceneworks_core::model_artifacts::{
    confine_artifact_path, ArtifactIdentity, ArtifactMemberRole, ModelArtifactResolver,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MAX_PINNED_METADATA_CACHE_ENTRIES: usize = 256;
static PINNED_METADATA_CACHE: OnceLock<Mutex<BTreeMap<String, Vec<PortableArtifactFile>>>> =
    OnceLock::new();

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

/// Bind source-affecting authored choices. Prompt, seed and render controls remain in the
/// canonical job payload and may change on an explicitly duplicated job without changing the
/// frozen source. Model/variant/decoder/adapter choices cannot silently change that source.
fn portable_authored_options(payload: &JsonObject) -> Value {
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
    project(&Value::Object(payload.clone()), None)
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
            | "veda"
            | "source"
            | "upscale"
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
            | "modelNameOrPath"
            | "embedder"
            | "engine"
            | "mode"
            | "editMode"
            | "variant"
            | "quantTier"
            | "mlxQuantize"
            | "decoder"
            | "controlMode"
            | "overlayId"
            | "componentId"
            | "textEncoderModel"
            | "enhancerModel"
            | "useUncensoredEnhancer"
            | "enhancePrompt"
            | "tileControlNetRepo"
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
    if model_free_job(job_type, payload) {
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
    let authored_hash = canonical_authored_options_sha256(&authored_options)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut routes = BTreeMap::new();
    for (backend, platform) in crate::portable_routes::eligible_portable_routes(
        job_type,
        payload,
        &entries,
        requested_gpu,
    )? {
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
        let source_claims = if selected_sources.iter().any(|source| {
            matches!(
                &source.source,
                SelectedSourceLocation::OperatorCachedRepo(_)
                    | SelectedSourceLocation::OperatorOverrideOrManifest { .. }
            )
        }) {
            state
                .jobs_store
                .portable_source_claims_for_route(
                    &backend,
                    &platform,
                    requested_gpu,
                    state.settings.worker_timeout_seconds,
                )
                .map_err(|error| {
                    incomplete(format!("worker source claims are unavailable: {error}"))
                })?
        } else {
            Vec::new()
        };
        let mut models = Vec::with_capacity(entries.len());
        for (index, (slot, entry)) in entries.iter().enumerate() {
            models.push(
                capture_model_slot(
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
                    &source_claims,
                )
                .await?,
            );
        }
        let route = PortableRouteBinding::new(&backend, &platform, models)
            .map_err(|error| incomplete(error.to_string()))?;
        routes.insert(route.key(), route);
    }
    if routes.is_empty() {
        return Err(incomplete(
            "no authorized worker route has an exact source binding",
        ));
    }
    PortableJobBinding::new(authored_options, routes)
        .map(Some)
        .map_err(|error| incomplete(error.to_string()))
}

fn incomplete(message: impl Into<String>) -> ApiError {
    ApiError::model_artifact_conflict(message.into(), "model_artifact_incomplete")
}

fn model_free_job(job_type: &JobType, payload: &JsonObject) -> bool {
    matches!(
        job_type,
        JobType::Placeholder
            | JobType::FrameExtract
            | JobType::TimelineExport
            | JobType::DatasetParquetImport
            | JobType::ModelImport
            | JobType::LoraImport
            | JobType::ModelDownload
            | JobType::LoraDownload
    ) || matches!(job_type, JobType::LoraTrain | JobType::ControlTraining)
        && payload.get("dryRun").and_then(Value::as_bool) == Some(true)
        || matches!(job_type, JobType::TrainingCaption)
            && payload.get("mode").and_then(Value::as_str) == Some("triggerOnly")
        || matches!(job_type, JobType::CatalogAnalysis)
            && [
                "structuredAnalysisEnabled",
                "visionAnalysisEnabled",
                "semanticEmbeddingsEnabled",
            ]
            .into_iter()
            .all(|key| payload.get(key).and_then(Value::as_bool) == Some(false))
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
    let actual = capture_local_source_files(&selected, &allowed)?;
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
) -> Result<PortableModelBinding, ApiError> {
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| incomplete(format!("{slot} has no canonical model id")))?;
    let training = training_base_selection(job_type, payload, model_id)?;
    let requested_variant = training
        .as_ref()
        .map(|selection| selection.tier.clone())
        .or_else(|| requested_runtime_variant(payload));
    let selected = selected_model_artifact_closure(entry, platform, requested_variant.as_deref());
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
    let mut members = Vec::new();
    let variant;
    let tier;
    if downloads.is_empty() {
        let local = capture_local_model_entry(state, entry, slot)?;
        if let Some(training) = &training {
            if entry.pointer("/importPlan/checkpointId").is_none() {
                return Err(incomplete(
                    "local training base has no verified checkpoint plan provenance",
                ));
            }
            let selected = capture_local_source_files(&training.path, &allowed_local_roots(state))?;
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
        members.extend(local);
        variant = requested_variant.unwrap_or_else(|| "local".to_owned());
        tier = None;
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
            verify_produced_local_source(state, entry, &training.as_ref().unwrap().path)?;
        }
        for (index, download) in downloads.iter().enumerate() {
            let role = if is_co_requisite_download(download) {
                ArtifactMemberRole::CoRequisite
            } else {
                ArtifactMemberRole::Primary
            };
            let component_id =
                (role != ArtifactMemberRole::Primary).then(|| format!("{slot}/downloads/{index}"));
            members.push(
                if role == ArtifactMemberRole::Primary && produced_training {
                    capture_local_member(
                        &training.as_ref().unwrap().path,
                        &allowed_local_roots(state),
                        role,
                        component_id,
                        tier.clone(),
                    )?
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
        let files = capture_local_source_files(&path, &allowed_local_roots(state))?;
        let digest =
            portable_closure_sha256(&files).map_err(|error| incomplete(error.to_string()))?;
        let override_member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest { sha256: digest },
            tier: tier.clone(),
            files,
        };
        override_member
            .validate()
            .map_err(|error| incomplete(error.to_string()))?;
        let primary = members
            .iter_mut()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .ok_or_else(|| incomplete(format!("{slot} has no primary source to override")))?;
        *primary = override_member;
    }
    // The first model carrier owns selected request adapters and controls. Other carriers retain
    // their own component closures but never duplicate an adapter under a second logical slot.
    if owns_request_overlays {
        members.extend(capture_selected_request_overlays(state, payload)?);
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
            SelectedSourceLocation::LocalPath(path) => capture_local_member(
                Path::new(path),
                &allowed_local_roots(state),
                source.role.clone(),
                Some(source.component_id.clone()),
                source.selected_tier.clone(),
            )?,
            SelectedSourceLocation::OperatorCachedRepo(source_id) => {
                capture_worker_source_member(
                    state,
                    source_id,
                    source.role.clone(),
                    source.component_id.clone(),
                    source_claims,
                )
                .await?
            }
            SelectedSourceLocation::DynamicHfRepo(repo) => {
                let revision = resolve_dynamic_hf_revision(state, repo).await?;
                let download = serde_json::json!({
                    "provider":"huggingface",
                    "repo":repo,
                    "revision":revision,
                    "files":["*"]
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
                if source_claims
                    .iter()
                    .any(|claim| claim.source_id == *source_id)
                {
                    capture_worker_source_member(
                        state,
                        source_id,
                        source.role.clone(),
                        source.component_id.clone(),
                        source_claims,
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
    PortableModelBinding::new(
        slot,
        model_id,
        backend,
        variant,
        tier,
        authored_hash,
        members,
    )
    .map_err(|error| incomplete(error.to_string()))
}

/// Resolve a mutable, user-selected repository to a publisher commit once, at admission.
/// The worker sees only the immutable revision and exact file ledger in its frozen route.
async fn resolve_dynamic_hf_revision(state: &AppState, repo: &str) -> Result<String, ApiError> {
    let base_url = std::env::var("SCENEWORKS_HUGGINGFACE_BASE_URL")
        .unwrap_or_else(|_| "https://huggingface.co".to_owned());
    let token = huggingface_token(state)?;
    resolve_dynamic_hf_revision_with_client(&state.http_client, &base_url, token.as_deref(), repo)
        .await
}

async fn resolve_dynamic_hf_revision_with_client(
    client: &reqwest::Client,
    base_url: &str,
    token: Option<&str>,
    repo: &str,
) -> Result<String, ApiError> {
    let repo = sceneworks_core::hf_repo_id::namespaced(repo)
        .map_err(|error| incomplete(format!("custom model repository is invalid: {error}")))?;
    let mut url = reqwest::Url::parse(&base_url)
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
        parts.push("revision").push("main");
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
    path: &Path,
    allowed_roots: &[PathBuf],
    role: ArtifactMemberRole,
    component_id: Option<String>,
    tier: Option<String>,
) -> Result<PortableArtifactMember, ApiError> {
    let files = capture_local_source_files(path, allowed_roots)?;
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
    let selectors = download
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
    let files = resolve_trusted_pinned_files(state, repo, revision, &selectors).await?;
    if let Some(local) = capture_receipted_local_files(state, repo, revision, variant, &selectors)?
    {
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
/// upstream file proof. Successful pinned metadata is cacheable for this process: a commit and
/// concrete file selectors cannot change without an upstream integrity failure.
async fn resolve_trusted_pinned_files(
    state: &AppState,
    repo: &str,
    revision: &str,
    selectors: &[String],
) -> Result<Vec<PortableArtifactFile>, ApiError> {
    let base_url = std::env::var("SCENEWORKS_HUGGINGFACE_BASE_URL")
        .unwrap_or_else(|_| "https://huggingface.co".to_owned());
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
    let credential = huggingface_token(state)?;
    let files = resolve_pinned_hf_files(
        &state.http_client,
        &base_url,
        credential.as_deref(),
        repo,
        revision,
        selectors,
    )
    .await
    .map_err(|error| incomplete(error.to_string()))?;
    let mut cache = cache
        .lock()
        .map_err(|_| ApiError::internal("Pinned metadata cache is poisoned"))?;
    if cache.len() >= MAX_PINNED_METADATA_CACHE_ENTRIES {
        cache.pop_first();
    }
    cache.insert(key, files.clone());
    Ok(files)
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
/// files for a pinned revision; each file is rehashed now, so a stale size or a substituted file
/// never inherits the old identity. Without such a receipt, admission consults pinned HF metadata.
fn capture_receipted_local_files(
    state: &AppState,
    repo: &str,
    revision: &str,
    variant: &str,
    selectors: &[String],
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
    let mut files = Vec::with_capacity(selected.len());
    for relative in selected {
        let path = confine_artifact_path(&root.join(&relative), &allowed)
            .map_err(|error| incomplete(error.to_string()))?;
        let metadata = std::fs::metadata(&path)
            .map_err(|error| incomplete(format!("receipted model file is unavailable: {error}")))?;
        if !metadata.is_file() {
            return Err(incomplete("receipted model source is not a regular file"));
        }
        let digest = sceneworks_core::hashing::sha256_file(&path).map_err(|error| {
            incomplete(format!("receipted model file cannot be hashed: {error}"))
        })?;
        files.push(PortableArtifactFile {
            relative_path: PathBuf::from(relative),
            size_bytes: metadata.len(),
            sha256: format!("sha256:{digest}"),
        });
    }
    Ok(Some(files))
}

/// A non-HF imported, external, or trained entry is bound from its actual loader locators. A
/// provenance-only `source.path` is deliberately omitted: the loader never opens that field.
fn capture_local_model_entry(
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
        verify_produced_local_source(state, entry, &locators[0].1)?;
    }
    let mut members = Vec::new();
    for (index, (key, path)) in locators.into_iter().enumerate() {
        let files = capture_local_source_files(&path, &allowed_roots)?;
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
    if receipt.output_model_id != model_id {
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
                .join(model_id);
            let expected = confine_artifact_path(&expected, &allowed_local_roots(state))
                .map_err(|error| incomplete(error.to_string()))?;
            if root != expected
                || producer.payload.get("modelId").and_then(Value::as_str) != Some(model_id)
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
            let (output, _) = crate::training::resolve_finetune_output_location(state, model_id);
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
                    != Some(model_id)
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
        model_id,
    )
    .map_err(|error| incomplete(format!("local model producer bytes changed: {error}")))?;
    Ok(())
}

fn allowed_local_roots(state: &AppState) -> Vec<PathBuf> {
    std::iter::once(state.settings.data_dir.clone())
        .chain(state.settings.external_model_roots.iter().cloned())
        .collect()
}

fn capture_local_source_files(
    source: &Path,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PortableArtifactFile>, ApiError> {
    let source = confine_artifact_path(source, allowed_roots)
        .map_err(|error| incomplete(error.to_string()))?;
    let mut selected = Vec::new();
    collect_source_files(&source, &source, allowed_roots, &mut selected, 0)?;
    if selected.is_empty() {
        return Err(incomplete(
            "selected local model source has no regular files",
        ));
    }
    selected.sort();
    Ok(selected)
}

fn collect_source_files(
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
    let path = confine_artifact_path(path, allowed_roots)
        .map_err(|error| incomplete(error.to_string()))?;
    if path.is_dir() {
        let mut children = std::fs::read_dir(&path)
            .map_err(|error| incomplete(format!("local model directory is unreadable: {error}")))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| incomplete(format!("local model directory is unreadable: {error}")))?;
        children.sort();
        for child in children {
            collect_source_files(root, &child, allowed_roots, selected, depth + 1)?;
        }
        return Ok(());
    }
    if !path.is_file() {
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
    let size_bytes = std::fs::metadata(&path)
        .map_err(|error| incomplete(format!("local model file is unavailable: {error}")))?
        .len();
    let sha256 = sceneworks_core::hashing::sha256_file(&path)
        .map_err(|error| incomplete(format!("local model file cannot be hashed: {error}")))?;
    selected.push(PortableArtifactFile {
        relative_path,
        size_bytes,
        sha256: format!("sha256:{sha256}"),
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
    for (slot, record) in enumerate_portable_array_slots(&value, &["loras", "lokrs", "controls"]) {
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
        let files = capture_local_source_files(&locator, &allowed_roots)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn authored_projection_keeps_choices_without_local_paths_or_credentials() {
        let payload = json!({
            "model":"flux_schnell", "prompt":"red kite", "variant":"q4",
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
        let first_files = capture_local_source_files(
            &first.path().join("checkpoint"),
            &[first.path().to_path_buf()],
        )
        .unwrap();
        let second_files = capture_local_source_files(
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
            &second.path().join("checkpoint"),
            &[second.path().to_path_buf()],
        )
        .unwrap();
        assert_ne!(first_files, changed);
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
            resolve_dynamic_hf_revision_with_client(&client, &base, None, "owner/control")
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
            resolve_dynamic_hf_revision_with_client(&client, &base, None, "owner/control")
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
        assert_ne!(first_files, second_files);
    }
}
