//! Freeze the server-authorized model closure before a job enters the queue.
//!
//! The persisted binding contains logical slots, immutable source identities and content hashes.
//! Paths and credentials are used only while capturing a local source or querying pinned metadata;
//! they never enter the portable job binding or its authored-options projection.

use crate::error::ApiError;
use crate::portable_routes::{SelectedRouteSource, SelectedSourceLocation};
use crate::AppState;
use sceneworks_core::contracts::{JobType, JsonObject};
use sceneworks_core::model_artifacts::artifact_selection::{
    is_co_requisite_download, is_supported_model_download, requested_runtime_variant,
    safe_download_dir, selected_model_artifact_closure,
};
use sceneworks_core::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, enumerate_portable_array_slots, portable_closure_sha256,
    resolve_pinned_hf_files, PortableArtifactFile, PortableArtifactMember, PortableJobBinding,
    PortableModelBinding, PortableRouteBinding, PortableSourceIdentity,
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
    let entries = payload_model_entries(payload)?;
    if entries.is_empty() {
        return if model_free_job(job_type, payload) {
            Ok(None)
        } else {
            Err(incomplete(
                "model-backed job has no canonical model source carrier",
            ))
        };
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
        let mut models = Vec::with_capacity(entries.len());
        for (index, (slot, entry)) in entries.iter().enumerate() {
            models.push(
                capture_model_slot(
                    state,
                    payload,
                    slot,
                    entry,
                    &backend,
                    &platform,
                    &authored_hash,
                    index == 0,
                    &selected_sources,
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
}

async fn capture_model_slot(
    state: &AppState,
    payload: &JsonObject,
    slot: &str,
    entry: &Value,
    backend: &str,
    platform: &str,
    authored_hash: &str,
    owns_request_overlays: bool,
    selected_sources: &[SelectedRouteSource],
) -> Result<PortableModelBinding, ApiError> {
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| incomplete(format!("{slot} has no canonical model id")))?;
    let requested_variant = requested_runtime_variant(payload);
    let selected = selected_model_artifact_closure(entry, platform, requested_variant.as_deref());
    let downloads = selected
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download))
        .collect::<Vec<_>>();
    let mut members = Vec::new();
    let variant;
    let tier;
    if downloads.is_empty() {
        let local = capture_local_model_entry(state, entry, slot)?;
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
        for (index, download) in downloads.iter().enumerate() {
            let role = if is_co_requisite_download(download) {
                ArtifactMemberRole::CoRequisite
            } else {
                ArtifactMemberRole::Primary
            };
            let component_id =
                (role != ArtifactMemberRole::Primary).then(|| format!("{slot}/downloads/{index}"));
            members.push(capture_download_member(state, download, role, component_id).await?);
        }
    }
    if let Some(path) = selected_primary_override(payload, entry, owns_request_overlays) {
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
                return Err(incomplete(format!(
                    "operator source {source_id} has no registered exact worker claim"
                )));
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

/// Loader precedence for a request-owned source override. The chosen path is captured as a
/// content digest before enqueue, rather than silently retaining the catalog's HF primary.
fn selected_primary_override(
    payload: &JsonObject,
    entry: &Value,
    owns_request_overlays: bool,
) -> Option<PathBuf> {
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
            selected_primary_override(payload.as_object().unwrap(), &entry, true),
            Some(PathBuf::from("/new/root/model"))
        );
        assert_eq!(
            selected_primary_override(payload.as_object().unwrap(), &entry, false),
            Some(PathBuf::from("/old/root/model"))
        );
    }
}
