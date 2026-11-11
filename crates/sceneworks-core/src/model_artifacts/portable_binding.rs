//! Portable, byte-exact source identity frozen with a job at admission.
//!
//! Logical names and content digests cross hosts. Physical locations are supplied only by the
//! receiving worker after its own source resolution and are never serialized into a job binding.

use super::{
    confine_artifact_path, validate_concrete_relative_file, validate_sha256, ArtifactContractError,
    ArtifactIdentity, ArtifactMemberRole,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const PORTABLE_MODEL_BINDING_VERSION: u32 = 2;
const MAX_NON_LFS_BYTES: u64 = 256 * 1024 * 1024;
const MAX_HF_TREE_PAGES: usize = 10_000;

/// A scanned external base has a host-local catalog ID. Its encoder choice is instead
/// registered under the imported family, with the exact selected file ledger in the claim.
pub fn selected_text_encoder_claim_model_id(model_id: &str, entry: &Value) -> Option<String> {
    if !model_id.starts_with("external_base_") {
        return Some(model_id.to_owned());
    }
    if entry.get("id").and_then(Value::as_str) != Some(model_id)
        || entry.get("catalogScope").and_then(Value::as_str) != Some("external")
        || entry.get("importSourceShape").and_then(Value::as_str) != Some("comfy_ui_tree")
    {
        return None;
    }
    let family = entry.get("family").and_then(Value::as_str)?;
    if family.is_empty() {
        return None;
    }
    Some(format!("external-comfyui:{family}"))
}

/// Content identity for one selected member. A single file keeps its existing source SHA-256;
/// a multi-file member hashes a canonical path/size/content-digest ledger.
pub fn portable_closure_sha256(
    files: &[PortableArtifactFile],
) -> Result<String, ArtifactContractError> {
    if files.is_empty() {
        return Err(error("portable artifact closure is empty"));
    }
    if files.len() == 1 {
        validate_sha256(&files[0].sha256)?;
        return Ok(files[0].sha256.clone());
    }
    portable_file_ledger_sha256(files)
}

pub fn portable_file_ledger_sha256(
    files: &[PortableArtifactFile],
) -> Result<String, ArtifactContractError> {
    if files.is_empty() {
        return Err(error("portable artifact closure is empty"));
    }
    let mut digest = Sha256::new();
    let mut previous = None;
    for file in files {
        let path = super::portable_relative_path(&file.relative_path, "portable artifact file")?;
        validate_sha256(&file.sha256)?;
        if previous.as_ref().is_some_and(|previous| previous >= &path) {
            return Err(error(
                "portable artifact files are duplicated or not canonical",
            ));
        }
        digest.update(path.as_bytes());
        digest.update([0]);
        digest.update(file.size_bytes.to_string().as_bytes());
        digest.update([0]);
        digest.update(file.sha256.as_bytes());
        digest.update(b"\n");
        previous = Some(path);
    }
    Ok(format!(
        "sha256:{}",
        crate::hashing::hex(&digest.finalize())
    ))
}

/// Resolve a pinned Hugging Face closure before admission. LFS object IDs are authoritative
/// SHA-256 hashes. Small Git files have no LFS OID, so their exact pinned content is streamed and
/// hashed here. No model is loaded and no weight tensor is materialized.
pub async fn resolve_pinned_hf_files(
    client: &reqwest::Client,
    base_url: &str,
    auth_token: Option<&str>,
    repository: &str,
    revision: &str,
    patterns: &[String],
) -> Result<Vec<PortableArtifactFile>, ArtifactContractError> {
    super::validate_repository(repository)?;
    super::validate_immutable_revision(revision)?;
    if patterns.is_empty() {
        return Err(error("pinned artifact has no file selectors"));
    }
    let selectors = patterns
        .iter()
        .map(|pattern| {
            validate_concrete_selector(pattern)?;
            glob::Pattern::new(pattern)
                .map_err(|cause| error(format!("invalid artifact file selector: {cause}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let base = reqwest::Url::parse(base_url)
        .map_err(|cause| error(format!("invalid Hugging Face base URL: {cause}")))?;
    if !matches!(base.scheme(), "https" | "http") || base.host_str().is_none() {
        return Err(error("Hugging Face base URL is not HTTP(S)"));
    }
    let tree = pinned_url(&base, repository, revision, None, true)?;
    let tree_path = tree.path().to_owned();
    let mut next = Some(tree);
    let mut selected = BTreeMap::<PathBuf, PortableArtifactFile>::new();
    let mut matched = vec![false; selectors.len()];
    for _ in 0..MAX_HF_TREE_PAGES {
        let Some(url) = next.take() else { break };
        let mut request = client.get(url).timeout(Duration::from_secs(120));
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|cause| error(format!("pinned artifact tree lookup failed: {cause}")))?;
        next = next_tree_page(&base, &tree_path, response.headers())?;
        let payload = response
            .json::<Value>()
            .await
            .map_err(|cause| error(format!("pinned artifact tree response is invalid: {cause}")))?;
        let entries = payload
            .as_array()
            .or_else(|| payload.get("siblings").and_then(Value::as_array))
            .ok_or_else(|| error("pinned artifact tree has no file entries"))?;
        for entry in entries {
            if entry
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind != "file")
            {
                continue;
            }
            let Some(path) = entry
                .get("path")
                .or_else(|| entry.get("rfilename"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let relative_path = PathBuf::from(path);
            validate_concrete_relative_file(&relative_path)?;
            let mut any_match = false;
            for (index, selector) in selectors.iter().enumerate() {
                if selector.matches(path) {
                    matched[index] = true;
                    any_match = true;
                }
            }
            if !any_match {
                continue;
            }
            if selected.contains_key(&relative_path) {
                return Err(error("pinned artifact tree repeats a selected file"));
            }
            let size_bytes = entry
                .get("size")
                .and_then(Value::as_u64)
                .ok_or_else(|| error(format!("pinned artifact file {path} has no exact size")))?;
            let sha256 = if let Some(oid) = entry
                .get("lfs")
                .and_then(|lfs| lfs.get("oid"))
                .and_then(Value::as_str)
            {
                let oid = format!("sha256:{}", oid.trim_start_matches("sha256:"));
                validate_sha256(&oid)?;
                oid
            } else {
                if size_bytes > MAX_NON_LFS_BYTES {
                    return Err(error(format!(
                        "non-LFS pinned artifact file {path} exceeds the bounded content proof"
                    )));
                }
                hash_pinned_non_lfs(
                    client, &base, auth_token, repository, revision, path, size_bytes,
                )
                .await?
            };
            selected.insert(
                relative_path.clone(),
                PortableArtifactFile {
                    relative_path,
                    size_bytes,
                    sha256,
                },
            );
        }
    }
    if next.is_some() {
        return Err(error("pinned artifact tree pagination exceeded its bound"));
    }
    if matched.iter().any(|matched| !matched) {
        return Err(error("pinned artifact file selector matched no files"));
    }
    if selected.is_empty() {
        return Err(error("pinned artifact closure is empty"));
    }
    Ok(selected.into_values().collect())
}

fn validate_concrete_selector(selector: &str) -> Result<(), ArtifactContractError> {
    if selector.trim().is_empty()
        || selector.starts_with('/')
        || selector.contains("..")
        || selector.contains('\\')
    {
        return Err(error("artifact file selector is empty or not portable"));
    }
    Ok(())
}

fn pinned_url(
    base: &reqwest::Url,
    repository: &str,
    revision: &str,
    file: Option<&str>,
    tree: bool,
) -> Result<reqwest::Url, ArtifactContractError> {
    let mut url = base.clone();
    {
        let mut parts = url
            .path_segments_mut()
            .map_err(|_| error("Hugging Face base URL cannot carry path segments"))?;
        parts.pop_if_empty();
        if tree {
            parts.push("api").push("models");
        }
        for part in repository.split('/') {
            parts.push(part);
        }
        parts
            .push(if tree { "tree" } else { "resolve" })
            .push(revision);
        if let Some(file) = file {
            for part in file.split('/') {
                parts.push(part);
            }
        }
    }
    if tree {
        url.set_query(Some("recursive=1&expand=1"));
    } else {
        url.set_query(None);
    }
    Ok(url)
}

fn next_tree_page(
    base: &reqwest::Url,
    tree_path: &str,
    headers: &reqwest::header::HeaderMap,
) -> Result<Option<reqwest::Url>, ArtifactContractError> {
    let Some(link) = headers.get(reqwest::header::LINK) else {
        return Ok(None);
    };
    let link = link
        .to_str()
        .map_err(|_| error("pinned artifact tree pagination header is invalid"))?;
    for part in link.split(',') {
        let mut pieces = part.split(';');
        let url = pieces
            .next()
            .unwrap_or_default()
            .trim()
            .trim_start_matches('<')
            .trim_end_matches('>');
        if pieces.any(|piece| matches!(piece.trim(), "rel=\"next\"" | "rel=next")) {
            let next = base
                .join(url)
                .map_err(|_| error("pinned artifact tree pagination URL is invalid"))?;
            if next.scheme() != base.scheme()
                || next.host_str() != base.host_str()
                || next.port_or_known_default() != base.port_or_known_default()
                || next.path() != tree_path
                || !next.username().is_empty()
                || next.password().is_some()
            {
                return Err(error("pinned artifact tree pagination changed origin"));
            }
            return Ok(Some(next));
        }
    }
    Ok(None)
}

async fn hash_pinned_non_lfs(
    client: &reqwest::Client,
    base: &reqwest::Url,
    auth_token: Option<&str>,
    repository: &str,
    revision: &str,
    path: &str,
    expected_size: u64,
) -> Result<String, ArtifactContractError> {
    let url = pinned_url(base, repository, revision, Some(path), false)?;
    let mut request = client.get(url).timeout(Duration::from_secs(120));
    if let Some(token) = auth_token {
        request = request.bearer_auth(token);
    }
    let mut response = request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|cause| error(format!("pinned non-LFS artifact proof failed: {cause}")))?;
    let mut digest = Sha256::new();
    let mut received = 0_u64;
    while let Some(chunk) = response.chunk().await.map_err(|cause| {
        error(format!(
            "pinned non-LFS artifact proof interrupted: {cause}"
        ))
    })? {
        received = received.saturating_add(chunk.len() as u64);
        if received > expected_size || received > MAX_NON_LFS_BYTES {
            return Err(error("pinned non-LFS artifact exceeds its declared size"));
        }
        digest.update(&chunk);
    }
    if received != expected_size {
        return Err(error("pinned non-LFS artifact size changed during proof"));
    }
    Ok(format!(
        "sha256:{}",
        crate::hashing::hex(&digest.finalize())
    ))
}

/// Hash only the API-authorized options object, never a raw request carrying local paths or
/// credentials. Both admission and the receiving worker use the same canonical key order.
pub fn canonical_authored_options_sha256(options: &Value) -> Result<String, ArtifactContractError> {
    canonical_object_sha256(options)
}

/// Hash a complete object with stable recursive key order. The import claim fence uses this on
/// the stored payload before the API removes private filesystem locators from its public view.
pub fn canonical_object_sha256(options: &Value) -> Result<String, ArtifactContractError> {
    if !options.is_object() {
        return Err(error("authored options must be an object"));
    }
    let canonical = canonical_json(options);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|cause| error(format!("cannot encode authored options: {cause}")))?;
    Ok(format!("sha256:{}", crate::hashing::sha256_bytes(&bytes)))
}

/// Enumerate selected source arrays at every depth of a canonical job payload. The slot path,
/// not an author-visible model ID, distinguishes repeated adapters in phases or nested recipes.
pub fn enumerate_portable_array_slots<'a>(
    payload: &'a Value,
    field_names: &[&str],
) -> Vec<(String, &'a Value)> {
    fn walk<'a>(
        value: &'a Value,
        path: &str,
        field_names: &[&str],
        result: &mut Vec<(String, &'a Value)>,
    ) {
        match value {
            Value::Object(fields) => {
                for (name, child) in fields {
                    let slot = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}/{name}")
                    };
                    if field_names.contains(&name.as_str()) {
                        if let Some(entries) = child.as_array() {
                            for (index, entry) in entries.iter().enumerate() {
                                result.push((format!("{slot}/{index}"), entry));
                            }
                        }
                    }
                    walk(child, &slot, field_names, result);
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    walk(child, &format!("{path}/{index}"), field_names, result);
                }
            }
            _ => {}
        }
    }
    let mut result = Vec::new();
    walk(payload, "", field_names, &mut result);
    result.sort_by(|left, right| left.0.cmp(&right.0));
    result
}

/// The same logical, file-ledger keyed Digest claim must be checked at admission and claim.
/// Adapter IDs come from the canonical job payload; slot paths alone are not catalog identities.
pub fn registered_digest_member_claim_id(
    model: &PortableModelBinding,
    member: &PortableArtifactMember,
    payload: &Value,
) -> Result<String, ArtifactContractError> {
    if !matches!(member.source, PortableSourceIdentity::Digest { .. }) {
        return Err(error("registered digest claim requires a Digest member"));
    }
    if let Some(plan) = &model.checkpoint_plan {
        let is_layer =
            (member.role == ArtifactMemberRole::Primary && plan.primary_layer_id.is_some())
                || member.component_id.as_deref().is_some_and(|id| {
                    id.starts_with(&format!("{}/importPlan/layers/", model.slot))
                });
        if is_layer {
            let ledger = portable_file_ledger_sha256(&member.files)?;
            return crate::portable_sources::registered_digest_claim_id(
                "model-bytes",
                &model.model_id,
                None,
                &ledger,
            )
            .map_err(error);
        }
    }
    let primary_catalog_id = (member.role == ArtifactMemberRole::Primary)
        .then(|| registered_primary_digest_catalog_id(model, payload))
        .transpose()?;
    let (kind, logical_id, component_id) = match member.role {
        ArtifactMemberRole::Primary => (
            "model",
            primary_catalog_id
                .as_deref()
                .expect("primary catalog id resolved"),
            None,
        ),
        ArtifactMemberRole::OptionalComponent
            if member.component_id.as_deref() == Some("selected_text_encoder") =>
        {
            let selected = payload
                .pointer("/modelManifestEntry/resolvedTextEncoder")
                .ok_or_else(|| error("selected text encoder has no private resolution"))?;
            let selected_id = payload
                .pointer("/advanced/textEncoderModel")
                .and_then(Value::as_str)
                .filter(|id| *id != "default" && !id.is_empty())
                .ok_or_else(|| error("selected text encoder is no longer authored"))?;
            if selected.get("selectionId").and_then(Value::as_str) != Some(selected_id)
                || selected
                    .get("path")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                return Err(error(
                    "selected text encoder differs from private resolution",
                ));
            }
            let source_kind = selected
                .get("sourceKind")
                .and_then(Value::as_str)
                .filter(|kind| matches!(*kind, "file" | "directory"))
                .ok_or_else(|| error("selected text encoder has no source kind"))?;
            let entry = payload
                .get("modelManifestEntry")
                .ok_or_else(|| error("selected text encoder has no model carrier"))?;
            let claim_model_id = selected_text_encoder_claim_model_id(&model.model_id, entry)
                .ok_or_else(|| error("selected text encoder has no registered model scope"))?;
            return crate::portable_sources::registered_digest_claim_id(
                "text-encoder",
                &format!("{claim_model_id}:{source_kind}"),
                None,
                &portable_file_ledger_sha256(&member.files)?,
            )
            .map_err(error);
        }
        ArtifactMemberRole::Control if member.component_id.as_deref() == Some("strict_control") => {
            let selected = payload
                .pointer("/advanced/controlWeights")
                .ok_or_else(|| error("registered control overlay is no longer selected"))?;
            let id = selected
                .get("overlayId")
                .and_then(Value::as_str)
                .filter(|id| !id.trim().is_empty() && *id == id.trim())
                .ok_or_else(|| error("registered control overlay has no logical id"))?;
            if selected.get("_catalogAuthorized").and_then(Value::as_bool) != Some(true)
                || selected
                    .get("path")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                return Err(error(
                    "registered control overlay is not locally authorized",
                ));
            }
            ("control", id, None)
        }
        ArtifactMemberRole::Lora | ArtifactMemberRole::Control => {
            let slot = member
                .component_id
                .as_deref()
                .ok_or_else(|| error("adapter has no frozen slot"))?;
            let record = enumerate_portable_array_slots(payload, &["loras", "lokrs", "controls"])
                .into_iter()
                .find(|(record_slot, _)| record_slot == slot)
                .map(|(_, record)| record)
                .ok_or_else(|| error("adapter slot is no longer selected"))?;
            let id = record
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| error("adapter has no registered logical id"))?;
            let kind = if member.role == ArtifactMemberRole::Lora {
                "lora"
            } else {
                "control"
            };
            (kind, id, None)
        }
        ArtifactMemberRole::RequiredComponent => {
            let component = member
                .component_id
                .as_deref()
                .ok_or_else(|| error("component has no frozen id"))?;
            let stable = component
                .strip_prefix(&format!("{}/", model.slot))
                .unwrap_or(component);
            ("component", model.model_id.as_str(), Some(stable))
        }
        _ => (
            "component",
            model.model_id.as_str(),
            Some(
                member
                    .component_id
                    .as_deref()
                    .ok_or_else(|| error("component has no frozen id"))?,
            ),
        ),
    };
    let ledger = portable_file_ledger_sha256(&member.files)?;
    crate::portable_sources::registered_digest_claim_id(kind, logical_id, component_id, &ledger)
        .map_err(error)
}

/// A trained Iris checkpoint renders through its builtin engine carrier, while its registered
/// source bytes belong to the selected checkpoint's catalog ID. Require the server-authored alias
/// fields to agree before using that ID for a private worker availability claim.
pub fn registered_primary_digest_catalog_id(
    model: &PortableModelBinding,
    payload: &Value,
) -> Result<String, ArtifactContractError> {
    if model.slot != "modelManifestEntry" {
        return Ok(model.model_id.clone());
    }
    let Some(checkpoint_id) = payload.get("checkpointModelId").and_then(Value::as_str) else {
        return Ok(model.model_id.clone());
    };
    let entry = payload
        .get(&model.slot)
        .ok_or_else(|| error("trained checkpoint has no model carrier"))?;
    let selected = entry
        .pointer("/trainedCheckpoint/modelId")
        .and_then(Value::as_str);
    if model.model_id != "iris_3b"
        || entry.get("id").and_then(Value::as_str) != Some(model.model_id.as_str())
        || entry
            .pointer("/trainedCheckpoint/engineModel")
            .and_then(Value::as_str)
            != Some(model.model_id.as_str())
        || entry
            .pointer("/trainedCheckpoint/task")
            .and_then(Value::as_str)
            != Some("generation")
        || selected != Some(checkpoint_id)
        || checkpoint_id.is_empty()
    {
        return Err(error(
            "trained checkpoint alias differs from registered source",
        ));
    }
    Ok(checkpoint_id.to_owned())
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(fields) => {
            let sorted: BTreeMap<_, _> = fields
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        _ => value.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableModelBinding {
    pub schema_version: u32,
    /// Stable field slot in the private job carrier, such as `modelManifestEntry` or
    /// `modelManifestEntries/0`. Distinct slots may select the same logical model.
    pub slot: String,
    pub model_id: String,
    pub backend: String,
    pub variant: String,
    pub tier: Option<String>,
    /// Hash of the canonical, already-authorized private request options kept with the job.
    /// The options themselves stay in that request; this binds them without copying secrets.
    pub authored_options_sha256: String,
    pub members: Vec<PortableArtifactMember>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_plan: Option<PortableCheckpointPlan>,
}

/// Loader semantics of a trusted checkpoint plan, without its host-specific plan id,
/// root id, install id, source locator, or physical path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableCheckpointPlan {
    pub semantic_digest: String,
    pub family: String,
    pub primary_layer_id: Option<String>,
    pub layers: Vec<PortableCheckpointLayer>,
    pub context: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableCheckpointLayer {
    pub layer_id: String,
    pub role: String,
    pub target_path: String,
    pub container: crate::checkpoint_import::CheckpointContainerV1,
    pub sha256: String,
}

/// One worker route can execute a job containing several model entries. All selected sources,
/// including adapters and trained bases, are frozen before the route is offered to a worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableRouteBinding {
    pub backend: String,
    pub platform: String,
    pub models: Vec<PortableModelBinding>,
}

/// Private admission envelope. Every permitted backend/platform route is frozen once, and a
/// claim selects exactly one route. The authored options are the API-approved portable subset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableJobBinding {
    pub schema_version: u32,
    pub authored_options: Value,
    pub authored_options_sha256: String,
    pub routes_sha256: String,
    pub routes: BTreeMap<String, PortableRouteBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PortableSourceIdentity {
    Revision {
        identity: ArtifactIdentity,
    },
    Digest {
        sha256: String,
    },
    /// A server-authorized reference to a private worker advertisement. This is distinct from a
    /// portable imported digest: only a worker with the matching source claim may take the job.
    WorkerClaim {
        source_id: String,
        provenance: PortableSourceProvenance,
        closure_sha256: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableArtifactMember {
    pub role: ArtifactMemberRole,
    pub component_id: Option<String>,
    pub source: PortableSourceIdentity,
    pub tier: Option<String>,
    pub files: Vec<PortableArtifactFile>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableArtifactFile {
    /// Logical path inside this member's closure, independent of a host's installation root.
    pub relative_path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

/// A private worker advertisement for an operator-selected source that cannot be inferred from
/// public worker capability or the static model registry. The file names are logical and relative;
/// the worker retains all physical paths and leases locally.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PortableSourceProvenance {
    PinnedRepo {
        repository: String,
        revision: String,
    },
    OperatorOverride,
}

impl PortableSourceProvenance {
    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        match self {
            Self::PinnedRepo {
                repository,
                revision,
            } => {
                super::validate_repository(repository)?;
                super::validate_immutable_revision(revision)
            }
            Self::OperatorOverride => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PortableSourceClaim {
    pub schema_version: u32,
    pub source_id: String,
    pub provenance: PortableSourceProvenance,
    pub files: Vec<PortableArtifactFile>,
    pub closure_sha256: String,
}

impl PortableSourceClaim {
    pub fn new(
        source_id: impl Into<String>,
        provenance: PortableSourceProvenance,
        mut files: Vec<PortableArtifactFile>,
    ) -> Result<Self, ArtifactContractError> {
        files.sort();
        let closure_sha256 = portable_file_ledger_sha256(&files)?;
        let claim = Self {
            schema_version: PORTABLE_MODEL_BINDING_VERSION,
            source_id: source_id.into(),
            provenance,
            files,
            closure_sha256,
        };
        claim.validate()?;
        Ok(claim)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        if self.schema_version != PORTABLE_MODEL_BINDING_VERSION {
            return Err(error("unsupported portable source claim version"));
        }
        validate_logical_name("source id", &self.source_id)?;
        self.provenance.validate()?;
        let mut previous = None;
        for file in &self.files {
            super::portable_relative_path(&file.relative_path, "portable source claim file")?;
            validate_sha256(&file.sha256)?;
            if previous
                .as_ref()
                .is_some_and(|previous| previous >= &file.relative_path)
            {
                return Err(error(
                    "portable source claim files are duplicated or not canonical",
                ));
            }
            previous = Some(file.relative_path.clone());
        }
        if portable_file_ledger_sha256(&self.files)? != self.closure_sha256 {
            return Err(error(
                "portable source claim digest differs from its file closure",
            ));
        }
        Ok(())
    }

    /// Admission compares a worker advertisement with the authoritative source policy, not just
    /// a caller-chosen source-id string. Only a route that explicitly permits an operator
    /// override may pass that provenance as the expected value.
    pub fn matches_authorized_source(
        &self,
        source_id: &str,
        provenance: &PortableSourceProvenance,
        files: &[PortableArtifactFile],
    ) -> Result<(), ArtifactContractError> {
        self.validate()?;
        let expected = Self::new(source_id, provenance.clone(), files.to_vec())?;
        if self != &expected {
            return Err(error(
                "worker source claim differs from authorized identity",
            ));
        }
        Ok(())
    }

    pub fn matches_worker_claim_member(
        &self,
        member: &PortableArtifactMember,
    ) -> Result<(), ArtifactContractError> {
        self.validate()?;
        member.validate()?;
        let PortableSourceIdentity::WorkerClaim {
            source_id,
            provenance,
            closure_sha256,
        } = &member.source
        else {
            return Err(error(
                "artifact member is not bound to a worker source claim",
            ));
        };
        if self.source_id != *source_id
            || self.provenance != *provenance
            || self.closure_sha256 != *closure_sha256
            || self.files != member.files
        {
            return Err(error(
                "worker source claim differs from route-bound artifact",
            ));
        }
        Ok(())
    }
}

pub fn validate_portable_source_claims(
    claims: &[PortableSourceClaim],
) -> Result<(), ArtifactContractError> {
    let mut ids = std::collections::BTreeSet::new();
    for claim in claims {
        claim.validate()?;
        if !ids.insert(&claim.source_id) {
            return Err(error("duplicate portable source claim id"));
        }
    }
    Ok(())
}

/// Worker-only mapping from a logical file to the selected local file. A Hugging Face snapshot
/// link may resolve into its repository's blob store; a different host may use another layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalArtifactFile {
    pub relative_path: PathBuf,
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalArtifactMember {
    pub role: ArtifactMemberRole,
    pub component_id: Option<String>,
    pub source: PortableSourceIdentity,
    pub tier: Option<String>,
    pub files: Vec<LocalArtifactFile>,
}

impl PortableCheckpointPlan {
    pub fn from_import_plan(
        plan: &crate::checkpoint_import::ImportPlanV1,
        primary_layer_id: Option<String>,
    ) -> Result<Self, ArtifactContractError> {
        plan.validate().map_err(|cause| error(cause.to_string()))?;
        let layers = plan
            .layers
            .iter()
            .map(|layer| {
                Ok(PortableCheckpointLayer {
                    layer_id: layer.layer_id.clone(),
                    role: layer.role.clone(),
                    target_path: layer.target_path.clone(),
                    container: layer.container,
                    sha256: format!(
                        "sha256:{}",
                        layer
                            .source
                            .content_digest()
                            .map_err(|cause| error(cause.to_string()))?
                    ),
                })
            })
            .collect::<Result<Vec<_>, ArtifactContractError>>()?;
        let portable = Self {
            semantic_digest: plan
                .semantic_digest()
                .map_err(|cause| error(cause.to_string()))?,
            family: plan.family.clone(),
            primary_layer_id,
            layers,
            context: plan.context.clone(),
        };
        portable.validate()?;
        Ok(portable)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        validate_sha256(&self.semantic_digest)?;
        validate_logical_name("checkpoint family", &self.family)?;
        if self.layers.is_empty() {
            return Err(error("portable checkpoint plan has no layers"));
        }
        fn reject_host_locators(value: &Value) -> Result<(), ArtifactContractError> {
            match value {
                Value::Object(fields) => {
                    for (key, value) in fields {
                        if [
                            "token",
                            "password",
                            "secret",
                            "credential",
                            "absolutePath",
                            "rootPath",
                            "localPath",
                        ]
                        .iter()
                        .any(|forbidden| key.eq_ignore_ascii_case(forbidden))
                        {
                            return Err(error("portable checkpoint context contains a private locator or credential"));
                        }
                        reject_host_locators(value)?;
                    }
                }
                Value::Array(values) => {
                    for value in values {
                        reject_host_locators(value)?;
                    }
                }
                Value::String(value)
                    if value.starts_with('/')
                        || value.starts_with('\\')
                        || (value.len() > 1 && value.as_bytes()[1] == b':') =>
                {
                    return Err(error(
                        "portable checkpoint context contains an absolute path",
                    ));
                }
                _ => {}
            }
            Ok(())
        }
        reject_host_locators(&Value::Object(self.context.clone().into_iter().collect()))?;
        let mut import_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let raw = layer
                .sha256
                .strip_prefix("sha256:")
                .ok_or_else(|| error("checkpoint layer digest has no sha256 prefix"))?;
            validate_sha256(&layer.sha256)?;
            import_layers.push(crate::checkpoint_import::ImportLayerV1 {
                layer_id: layer.layer_id.clone(),
                role: layer.role.clone(),
                target_path: layer.target_path.clone(),
                container: layer.container,
                source: crate::checkpoint_import::SourceLocatorV1::linked(
                    "portable", "artifact", raw,
                )
                .map_err(|cause| error(cause.to_string()))?,
            });
        }
        let mut reconstructed =
            crate::checkpoint_import::ImportPlanV1::new("portable", &self.family, import_layers)
                .map_err(|cause| error(cause.to_string()))?;
        reconstructed.context = self.context.clone();
        let digest = reconstructed
            .semantic_digest()
            .map_err(|cause| error(cause.to_string()))?;
        if digest != self.semantic_digest {
            return Err(error(
                "portable checkpoint semantic digest differs from frozen plan",
            ));
        }
        if let Some(primary) = &self.primary_layer_id {
            if !self.layers.iter().any(|layer| &layer.layer_id == primary) {
                return Err(error("portable checkpoint primary layer is absent"));
            }
        }
        Ok(())
    }

    fn validate_against(
        &self,
        slot: &str,
        members: &[PortableArtifactMember],
    ) -> Result<(), ArtifactContractError> {
        self.validate()?;
        let prefix = format!("{slot}/importPlan/layers/");
        for layer in &self.layers {
            let member = if self.primary_layer_id.as_deref() == Some(&layer.layer_id) {
                members
                    .iter()
                    .find(|member| member.role == ArtifactMemberRole::Primary)
            } else {
                let id = format!("{prefix}{}", layer.layer_id);
                members.iter().find(|member| {
                    member.role == ArtifactMemberRole::RequiredComponent
                        && member.component_id.as_deref() == Some(&id)
                })
            }
            .ok_or_else(|| error("checkpoint layer has no frozen artifact member"))?;
            if member.files.len() != 1
                || member.files[0].relative_path != Path::new("artifact")
                || member.files[0].sha256 != layer.sha256
                || member.source
                    != (PortableSourceIdentity::Digest {
                        sha256: layer.sha256.clone(),
                    })
            {
                return Err(error("checkpoint layer differs from frozen artifact bytes"));
            }
        }
        for member in members {
            if let Some(layer_id) = member
                .component_id
                .as_deref()
                .and_then(|id| id.strip_prefix(&prefix))
            {
                if !self.layers.iter().any(|layer| layer.layer_id == layer_id) {
                    return Err(error("frozen checkpoint artifact has no plan layer"));
                }
            }
        }
        Ok(())
    }
}

impl PortableModelBinding {
    pub fn with_checkpoint_plan(
        mut self,
        checkpoint_plan: PortableCheckpointPlan,
    ) -> Result<Self, ArtifactContractError> {
        self.checkpoint_plan = Some(checkpoint_plan);
        self.validate()?;
        Ok(self)
    }

    pub fn new(
        slot: impl Into<String>,
        model_id: impl Into<String>,
        backend: impl Into<String>,
        variant: impl Into<String>,
        tier: Option<String>,
        authored_options_sha256: impl Into<String>,
        mut members: Vec<PortableArtifactMember>,
    ) -> Result<Self, ArtifactContractError> {
        for member in &mut members {
            member.files.sort();
        }
        members.sort_by(|left, right| {
            member_key(&left.role, &left.component_id)
                .cmp(&member_key(&right.role, &right.component_id))
        });
        let binding = Self {
            schema_version: PORTABLE_MODEL_BINDING_VERSION,
            slot: slot.into(),
            model_id: model_id.into(),
            backend: backend.into(),
            variant: variant.into(),
            tier,
            authored_options_sha256: authored_options_sha256.into(),
            members,
            checkpoint_plan: None,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        if self.schema_version != PORTABLE_MODEL_BINDING_VERSION {
            return Err(error("unsupported portable model binding version"));
        }
        for (label, value) in [
            ("model slot", self.slot.as_str()),
            ("model id", self.model_id.as_str()),
            ("backend", self.backend.as_str()),
            ("variant", self.variant.as_str()),
        ] {
            validate_logical_name(label, value)?;
        }
        if let Some(tier) = &self.tier {
            validate_logical_name("tier", tier)?;
        }
        validate_sha256(&self.authored_options_sha256)?;
        if self.members.is_empty() {
            return Err(error("portable model binding has no artifact members"));
        }
        if self
            .members
            .iter()
            .filter(|member| member.role == ArtifactMemberRole::Primary)
            .count()
            != 1
        {
            return Err(error(
                "portable model binding requires exactly one primary member",
            ));
        }
        let mut previous = None;
        for member in &self.members {
            member.validate()?;
            let key = member_key(&member.role, &member.component_id);
            if previous.as_ref().is_some_and(|previous| previous >= &key) {
                return Err(error(
                    "portable artifact members are duplicated or not canonical",
                ));
            }
            previous = Some(key);
        }
        let primary = self
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .expect("primary count checked");
        if primary.tier != self.tier {
            return Err(error("portable primary tier does not match selected tier"));
        }
        if let Some(plan) = &self.checkpoint_plan {
            plan.validate_against(&self.slot, &self.members)?;
        }
        Ok(())
    }

    /// Verify the complete selected closure under worker-owned roots before any fetch or load.
    /// The local map must contain exactly the frozen logical members and paths; no substitute or
    /// omitted protected component can pass. Physical paths exist only in this call's arguments.
    pub fn verify_local(
        &self,
        local: &[LocalArtifactMember],
        allowed_roots: &[PathBuf],
    ) -> Result<(), ArtifactContractError> {
        self.validate()?;
        if allowed_roots.is_empty() || local.len() != self.members.len() {
            return Err(error("portable artifact closure is incomplete"));
        }
        let mut candidates = BTreeMap::new();
        for member in local {
            let key = member_key(&member.role, &member.component_id);
            if candidates.insert(key, member).is_some() {
                return Err(error("duplicate local artifact member"));
            }
        }
        for expected in &self.members {
            let actual = candidates
                .get(&member_key(&expected.role, &expected.component_id))
                .ok_or_else(|| error("portable artifact member is missing locally"))?;
            if actual.source != expected.source || actual.tier != expected.tier {
                return Err(error(
                    "local artifact source or tier differs from admitted identity",
                ));
            }
            if actual.files.len() != expected.files.len() {
                return Err(error(
                    "local artifact file closure differs from admitted identity",
                ));
            }
            let mut files = BTreeMap::new();
            for file in &actual.files {
                validate_concrete_relative_file(&file.relative_path)?;
                if files.insert(&file.relative_path, &file.path).is_some() {
                    return Err(error("duplicate local artifact file"));
                }
            }
            for file in &expected.files {
                let path = files
                    .get(&file.relative_path)
                    .ok_or_else(|| error("admitted artifact file is missing locally"))?;
                verify_file(path, file, allowed_roots)?;
            }
        }
        Ok(())
    }
}

impl PortableRouteBinding {
    pub fn new(
        backend: impl Into<String>,
        platform: impl Into<String>,
        mut models: Vec<PortableModelBinding>,
    ) -> Result<Self, ArtifactContractError> {
        models.sort_by(|left, right| left.slot.cmp(&right.slot));
        let route = Self {
            backend: backend.into(),
            platform: platform.into(),
            models,
        };
        route.validate()?;
        Ok(route)
    }

    pub fn key(&self) -> String {
        format!("{}:{}", self.backend, self.platform)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        validate_route_token("backend", &self.backend)?;
        validate_route_token("platform", &self.platform)?;
        if self.models.is_empty() {
            return Err(error("portable route has no model bindings"));
        }
        let mut previous = None;
        for model in &self.models {
            model.validate()?;
            if previous
                .as_deref()
                .is_some_and(|slot| slot >= model.slot.as_str())
            {
                return Err(error(
                    "portable route model slots are duplicated or not canonical",
                ));
            }
            previous = Some(model.slot.clone());
        }
        Ok(())
    }
}

impl PortableJobBinding {
    pub fn new(
        authored_options: Value,
        routes: BTreeMap<String, PortableRouteBinding>,
    ) -> Result<Self, ArtifactContractError> {
        let authored_options_sha256 = canonical_authored_options_sha256(&authored_options)?;
        let routes_sha256 = route_map_sha256(&routes)?;
        let binding = Self {
            schema_version: PORTABLE_MODEL_BINDING_VERSION,
            authored_options,
            authored_options_sha256,
            routes_sha256,
            routes,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        if self.schema_version != PORTABLE_MODEL_BINDING_VERSION {
            return Err(error("unsupported portable job binding version"));
        }
        if canonical_authored_options_sha256(&self.authored_options)?
            != self.authored_options_sha256
        {
            return Err(error("portable authored options digest changed"));
        }
        if route_map_sha256(&self.routes)? != self.routes_sha256 {
            return Err(error("portable backend closure changed after admission"));
        }
        if self.routes.is_empty() {
            return Err(error("portable job binding has no eligible routes"));
        }
        let mut expected_models: Option<Vec<(&str, &str)>> = None;
        for (key, route) in &self.routes {
            route.validate()?;
            if key != &route.key() {
                return Err(error("portable route key differs from backend/platform"));
            }
            let models = route
                .models
                .iter()
                .map(|model| (model.slot.as_str(), model.model_id.as_str()))
                .collect::<Vec<_>>();
            if let Some(expected) = &expected_models {
                if *expected != models {
                    return Err(error("portable routes disagree on logical model slots"));
                }
            } else {
                expected_models = Some(models);
            }
            if route
                .models
                .iter()
                .any(|model| model.authored_options_sha256 != self.authored_options_sha256)
            {
                return Err(error("portable route changes authorized authored intent"));
            }
        }
        Ok(())
    }

    pub fn select_route(
        &self,
        backend: &str,
        platform: &str,
    ) -> Result<&PortableRouteBinding, ArtifactContractError> {
        self.validate()?;
        self.routes
            .get(&format!("{backend}:{platform}"))
            .ok_or_else(|| error("worker route is not frozen in job admission"))
    }
}

fn route_map_sha256(
    routes: &BTreeMap<String, PortableRouteBinding>,
) -> Result<String, ArtifactContractError> {
    let mut canonical = serde_json::to_value(routes)
        .map_err(|cause| error(format!("cannot encode portable routes: {cause}")))?;
    canonical.sort_all_objects();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|cause| error(format!("cannot encode portable routes: {cause}")))?;
    Ok(format!("sha256:{}", crate::hashing::sha256_bytes(&bytes)))
}

fn validate_route_token(label: &str, value: &str) -> Result<(), ArtifactContractError> {
    validate_logical_name(label, value)?;
    if value.contains([':', '/', '\\']) {
        return Err(error(format!("portable {label} is not a route token")));
    }
    Ok(())
}

impl PortableArtifactMember {
    pub fn capture(
        local: &LocalArtifactMember,
        allowed_roots: &[PathBuf],
    ) -> Result<Self, ArtifactContractError> {
        if allowed_roots.is_empty() {
            return Err(error("artifact capture has no managed roots"));
        }
        let mut files = Vec::with_capacity(local.files.len());
        for file in &local.files {
            validate_concrete_relative_file(&file.relative_path)?;
            let path = confine_artifact_path(&file.path, allowed_roots)?;
            let metadata = std::fs::metadata(&path)
                .map_err(|cause| error(format!("artifact file is unavailable: {cause}")))?;
            if !metadata.is_file() {
                return Err(error("artifact closure entry is not a regular file"));
            }
            let digest = sceneworks_core_hash(&path)?;
            files.push(PortableArtifactFile {
                relative_path: file.relative_path.clone(),
                size_bytes: metadata.len(),
                sha256: format!("sha256:{digest}"),
            });
        }
        files.sort();
        let member = Self {
            role: local.role.clone(),
            component_id: local.component_id.clone(),
            source: local.source.clone(),
            tier: local.tier.clone(),
            files,
        };
        member.validate()?;
        Ok(member)
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        match &self.source {
            PortableSourceIdentity::Revision { identity } => identity.validate()?,
            PortableSourceIdentity::Digest { sha256 } => {
                validate_sha256(sha256)?;
                if *sha256 != portable_closure_sha256(&self.files)? {
                    return Err(error(
                        "artifact source digest differs from its file closure",
                    ));
                }
            }
            PortableSourceIdentity::WorkerClaim {
                source_id,
                provenance,
                closure_sha256,
            } => {
                validate_logical_name("worker source id", source_id)?;
                provenance.validate()?;
                validate_sha256(closure_sha256)?;
                if *closure_sha256 != portable_file_ledger_sha256(&self.files)? {
                    return Err(error(
                        "worker source reference differs from its file closure",
                    ));
                }
            }
        }
        if self.role == ArtifactMemberRole::Primary {
            if self.component_id.is_some() {
                return Err(error("primary artifact has a component id"));
            }
        } else {
            validate_logical_name(
                "component id",
                self.component_id
                    .as_deref()
                    .ok_or_else(|| error("artifact component id is missing"))?,
            )?;
        }
        if let Some(tier) = &self.tier {
            validate_logical_name("member tier", tier)?;
        }
        if self.files.is_empty() {
            return Err(error("portable artifact member has no files"));
        }
        let mut previous = None;
        for file in &self.files {
            validate_concrete_relative_file(&file.relative_path)?;
            validate_sha256(&file.sha256)?;
            if previous
                .as_ref()
                .is_some_and(|previous| previous >= &file.relative_path)
            {
                return Err(error(
                    "portable artifact files are duplicated or not canonical",
                ));
            }
            previous = Some(file.relative_path.clone());
        }
        Ok(())
    }
}

fn verify_file(
    path: &Path,
    expected: &PortableArtifactFile,
    allowed_roots: &[PathBuf],
) -> Result<(), ArtifactContractError> {
    let path = confine_artifact_path(path, allowed_roots)?;
    let metadata = std::fs::metadata(&path)
        .map_err(|cause| error(format!("artifact file is unavailable: {cause}")))?;
    if !metadata.is_file() || metadata.len() != expected.size_bytes {
        return Err(error("artifact file size differs from admitted identity"));
    }
    if format!("sha256:{}", sceneworks_core_hash(&path)?) != expected.sha256 {
        return Err(error("artifact file bytes differ from admitted identity"));
    }
    Ok(())
}

fn sceneworks_core_hash(path: &Path) -> Result<String, ArtifactContractError> {
    let input = std::fs::File::open(path)
        .map_err(|cause| error(format!("artifact file is unavailable: {cause}")))?;
    crate::hashing::sha256_reader_retry(input)
        .map(|(digest, _)| digest)
        .map_err(|cause| error(format!("artifact file hash failed: {cause}")))
}

fn member_key(
    role: &ArtifactMemberRole,
    component_id: &Option<String>,
) -> (ArtifactMemberRole, Option<String>) {
    (role.clone(), component_id.clone())
}

fn validate_logical_name(label: &str, value: &str) -> Result<(), ArtifactContractError> {
    if value.trim().is_empty() || value != value.trim() || value.chars().any(char::is_control) {
        Err(error(format!("portable {label} is empty or malformed")))
    } else {
        Ok(())
    }
}

fn error(message: impl Into<String>) -> ArtifactContractError {
    ArtifactContractError(message.into())
}

/// Jobs with no model bytes to bind may keep the historical versionless claim protocol.
/// Keep this predicate shared by API admission and the store's claim gate.
pub fn model_free_job(
    job_type: &crate::contracts::JobType,
    payload: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    use crate::contracts::JobType;
    use serde_json::Value;
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
        || matches!(job_type, JobType::PersonDetect | JobType::PersonTrack)
            && payload.get("preview").and_then(Value::as_bool) == Some(true)
        || matches!(job_type, JobType::CatalogAnalysis)
            && [
                "structuredAnalysisEnabled",
                "visionAnalysisEnabled",
                "semanticEmbeddingsEnabled",
            ]
            .into_iter()
            .all(|key| payload.get(key).and_then(Value::as_bool) == Some(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn checkpoint_plan_freezes_semantics_without_host_locator() {
        use crate::checkpoint_import::{
            CheckpointContainerV1, ImportLayerV1, ImportPlanV1, SourceLocatorV1,
        };
        let sha = "a".repeat(64);
        let make_plan = |root_id: &str| {
            let mut plan = ImportPlanV1::new(
                format!("plan-{root_id}"),
                "h3",
                vec![ImportLayerV1 {
                    layer_id: "artifact:transformer/model.safetensors".into(),
                    role: "transformer".into(),
                    target_path: "transformer/model.safetensors".into(),
                    container: CheckpointContainerV1::Safetensors,
                    source: SourceLocatorV1::linked(root_id, "host/model.safetensors", &sha)
                        .unwrap(),
                }],
            )
            .unwrap();
            plan.context
                .insert("partition".into(), serde_json::json!({"kind":"video"}));
            plan
        };
        let first = PortableCheckpointPlan::from_import_plan(
            &make_plan("root-one"),
            Some("artifact:transformer/model.safetensors".into()),
        )
        .unwrap();
        let second = PortableCheckpointPlan::from_import_plan(
            &make_plan("root-two"),
            Some("artifact:transformer/model.safetensors".into()),
        )
        .unwrap();
        assert_eq!(first, second);
        let serialized = serde_json::to_string(&first).unwrap();
        assert!(!serialized.contains("root-one"));
        assert!(!serialized.contains("host/model"));
        let member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256: format!("sha256:{sha}"),
            },
            tier: None,
            files: vec![PortableArtifactFile {
                relative_path: "artifact".into(),
                size_bytes: 4,
                sha256: format!("sha256:{sha}"),
            }],
        };
        let binding = PortableModelBinding::new(
            "modelManifestEntry",
            "model",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&serde_json::json!({})).unwrap(),
            vec![member],
        )
        .unwrap()
        .with_checkpoint_plan(first.clone())
        .unwrap();
        assert!(binding.validate().is_ok());
        let mut changed = first;
        changed.layers[0].target_path = "other/model.safetensors".into();
        assert!(binding.with_checkpoint_plan(changed).is_err());
    }

    #[test]
    fn registered_component_claim_id_is_stable_across_payload_slots() {
        let files = vec![PortableArtifactFile {
            relative_path: "vae/weights.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }];
        let source = PortableSourceIdentity::Digest {
            sha256: portable_closure_sha256(&files).unwrap(),
        };
        let make = |slot: &str| {
            PortableModelBinding::new(
                slot,
                "registered_model",
                "mlx",
                "local",
                None,
                canonical_authored_options_sha256(&serde_json::json!({})).unwrap(),
                vec![
                    PortableArtifactMember {
                        role: ArtifactMemberRole::Primary,
                        component_id: None,
                        source: source.clone(),
                        tier: None,
                        files: files.clone(),
                    },
                    PortableArtifactMember {
                        role: ArtifactMemberRole::RequiredComponent,
                        component_id: Some(format!("{slot}/paths/vae")),
                        source: source.clone(),
                        tier: None,
                        files: files.clone(),
                    },
                ],
            )
            .unwrap()
        };
        let first = make("modelManifestEntry");
        let second = make("modelManifestEntries/0");
        let first_id =
            registered_digest_member_claim_id(&first, &first.members[1], &serde_json::json!({}))
                .unwrap();
        let second_id =
            registered_digest_member_claim_id(&second, &second.members[1], &serde_json::json!({}))
                .unwrap();
        assert_eq!(first_id, second_id);
    }

    #[test]
    fn selected_encoder_claim_uses_model_and_bytes_not_host_option_id() {
        let files = vec![PortableArtifactFile {
            relative_path: "model.safetensors".into(),
            size_bytes: 7,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }];
        let source = PortableSourceIdentity::Digest {
            sha256: portable_closure_sha256(&files).unwrap(),
        };
        let make = |model_id: &str| {
            PortableModelBinding::new(
                "modelManifestEntry",
                model_id,
                "mlx",
                "local",
                None,
                canonical_authored_options_sha256(&serde_json::json!({})).unwrap(),
                vec![
                    PortableArtifactMember {
                        role: ArtifactMemberRole::Primary,
                        component_id: None,
                        source: source.clone(),
                        tier: None,
                        files: files.clone(),
                    },
                    PortableArtifactMember {
                        role: ArtifactMemberRole::OptionalComponent,
                        component_id: Some("selected_text_encoder".to_owned()),
                        source: source.clone(),
                        tier: None,
                        files: files.clone(),
                    },
                ],
            )
            .unwrap()
        };
        let payload = |option: &str, path: &str| {
            serde_json::json!({
                "advanced":{"textEncoderModel":option},
                "modelManifestEntry":{"resolvedTextEncoder":{
                    "selectionId":option,"sourceKind":"file","path":path
                }}
            })
        };
        let model = make("qwen_image");
        let api = registered_digest_member_claim_id(
            &model,
            &model.members[1],
            &payload("text_encoder_api", "/api/encoder.safetensors"),
        )
        .unwrap();
        let worker = registered_digest_member_claim_id(
            &model,
            &model.members[1],
            &payload("text_encoder_worker", "/worker/encoder.safetensors"),
        )
        .unwrap();
        assert_eq!(api, worker);
        let other = make("qwen_image_edit");
        assert_ne!(
            api,
            registered_digest_member_claim_id(
                &other,
                &other.members[1],
                &payload("text_encoder_worker", "/worker/encoder.safetensors")
            )
            .unwrap()
        );
    }

    #[test]
    fn trained_iris_primary_claim_uses_selected_checkpoint_catalog_id() {
        let files = vec![PortableArtifactFile {
            relative_path: "model.safetensors".into(),
            size_bytes: 7,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }];
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "iris_3b",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&serde_json::json!({})).unwrap(),
            vec![PortableArtifactMember {
                role: ArtifactMemberRole::Primary,
                component_id: None,
                source: PortableSourceIdentity::Digest {
                    sha256: portable_closure_sha256(&files).unwrap(),
                },
                tier: None,
                files: files.clone(),
            }],
        )
        .unwrap();
        let payload = serde_json::json!({
            "checkpointModelId":"trained_iris",
            "modelManifestEntry":{
                "id":"iris_3b",
                "trainedCheckpoint":{
                    "engineModel":"iris_3b", "modelId":"trained_iris", "task":"generation"
                }
            }
        });
        let expected = crate::portable_sources::registered_digest_claim_id(
            "model",
            "trained_iris",
            None,
            &portable_file_ledger_sha256(&files).unwrap(),
        )
        .unwrap();
        assert_eq!(
            registered_digest_member_claim_id(&model, &model.members[0], &payload).unwrap(),
            expected
        );
        let mut substituted = payload;
        substituted["modelManifestEntry"]["trainedCheckpoint"]["modelId"] =
            serde_json::json!("other_iris");
        assert!(
            registered_digest_member_claim_id(&model, &model.members[0], &substituted).is_err()
        );
    }

    #[test]
    fn scanned_external_encoder_claim_uses_family_and_rejects_unscoped_rows() {
        let scoped = |id: &str| {
            serde_json::json!({
                "id": id,
                "catalogScope": "external",
                "importSourceShape": "comfy_ui_tree",
                "family": "qwen-image"
            })
        };
        assert_eq!(
            selected_text_encoder_claim_model_id(
                "external_base_qwen_api",
                &scoped("external_base_qwen_api")
            ),
            selected_text_encoder_claim_model_id(
                "external_base_qwen_worker",
                &scoped("external_base_qwen_worker")
            )
        );
        assert_eq!(
            selected_text_encoder_claim_model_id(
                "external_base_qwen_api",
                &scoped("external_base_qwen_api")
            ),
            Some("external-comfyui:qwen-image".to_owned())
        );
        let mut wrong_scope = scoped("external_base_qwen_api");
        wrong_scope["catalogScope"] = serde_json::json!("builtin");
        assert!(
            selected_text_encoder_claim_model_id("external_base_qwen_api", &wrong_scope).is_none()
        );
        assert!(selected_text_encoder_claim_model_id(
            "external_base_qwen_api",
            &scoped("external_base_other")
        )
        .is_none());
        assert_eq!(
            selected_text_encoder_claim_model_id("qwen_image", &scoped("external_base_qwen_api")),
            Some("qwen_image".to_owned())
        );
    }

    #[test]
    fn registered_pose_overlay_claim_uses_catalog_id_and_frozen_bytes() {
        let primary_files = vec![PortableArtifactFile {
            relative_path: "model.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }];
        let overlay_files = vec![PortableArtifactFile {
            relative_path: "pose.safetensors".into(),
            size_bytes: 2,
            sha256: format!("sha256:{}", "b".repeat(64)),
        }];
        let source = |files: &[PortableArtifactFile]| PortableSourceIdentity::Digest {
            sha256: portable_closure_sha256(files).unwrap(),
        };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "krea_2_turbo",
            "mlx",
            "local",
            None,
            canonical_authored_options_sha256(&serde_json::json!({})).unwrap(),
            vec![
                PortableArtifactMember {
                    role: ArtifactMemberRole::Primary,
                    component_id: None,
                    source: source(&primary_files),
                    tier: None,
                    files: primary_files,
                },
                PortableArtifactMember {
                    role: ArtifactMemberRole::Control,
                    component_id: Some("strict_control".to_owned()),
                    source: source(&overlay_files),
                    tier: None,
                    files: overlay_files,
                },
            ],
        )
        .unwrap();
        let overlay = &model.members[1];
        let payload = serde_json::json!({"advanced":{"controlWeights":{
            "overlayId":"pose-approved", "path":"/api-host/pose.safetensors",
            "_catalogAuthorized":true
        }}});
        let ledger = portable_file_ledger_sha256(&overlay.files).unwrap();
        let expected = crate::portable_sources::registered_digest_claim_id(
            "control",
            "pose-approved",
            None,
            &ledger,
        )
        .unwrap();
        assert_eq!(
            registered_digest_member_claim_id(&model, overlay, &payload).unwrap(),
            expected
        );
        let changed = serde_json::json!({"advanced":{"controlWeights":{
            "overlayId":"pose-substituted", "path":"/api-host/pose.safetensors",
            "_catalogAuthorized":true
        }}});
        assert_ne!(
            registered_digest_member_claim_id(&model, overlay, &changed).unwrap(),
            expected
        );
        for invalid in [
            serde_json::json!({"advanced":{"controlWeights":{"overlayId":"pose-approved","path":"/api-host/pose.safetensors"}}}),
            serde_json::json!({"advanced":{"controlWeights":{"overlayId":"pose-approved","_catalogAuthorized":true}}}),
            serde_json::json!({"advanced":{"controlWeights":{"path":"/api-host/pose.safetensors","_catalogAuthorized":true}}}),
        ] {
            assert!(registered_digest_member_claim_id(&model, overlay, &invalid).is_err());
        }
    }

    fn local(root: &std::path::Path, component: bool, bytes: &[u8]) -> LocalArtifactMember {
        let name = if component {
            "adapter.bin"
        } else {
            "model.bin"
        };
        fs::create_dir_all(root).unwrap();
        fs::write(root.join(name), bytes).unwrap();
        LocalArtifactMember {
            role: if component {
                ArtifactMemberRole::Lora
            } else {
                ArtifactMemberRole::Primary
            },
            component_id: component.then(|| "author/adapter".to_owned()),
            source: PortableSourceIdentity::Digest {
                sha256: format!("sha256:{}", crate::hashing::sha256_bytes(bytes)),
            },
            tier: None,
            files: vec![LocalArtifactFile {
                relative_path: PathBuf::from(name),
                path: root.join(name),
            }],
        }
    }

    #[test]
    fn identical_bytes_bind_on_second_root_and_substitution_refuses() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let primary_a = local(&a.path().join("primary"), false, b"model");
        let adapter_a = local(&a.path().join("adapter"), true, b"adapter");
        let members = [&primary_a, &adapter_a]
            .into_iter()
            .map(|member| {
                PortableArtifactMember::capture(member, &[a.path().to_path_buf()]).unwrap()
            })
            .collect();
        let binding = PortableModelBinding::new(
            "modelManifestEntry",
            "model",
            "mlx",
            "default",
            None,
            canonical_authored_options_sha256(&serde_json::json!({"steps": 3})).unwrap(),
            members,
        )
        .unwrap();
        let primary_b = local(&b.path().join("different"), false, b"model");
        let adapter_b = local(&b.path().join("other"), true, b"adapter");
        let roots = [b.path().to_path_buf()];
        assert!(binding
            .verify_local(&[adapter_b.clone(), primary_b.clone()], &roots)
            .is_ok());
        assert!(binding
            .verify_local(std::slice::from_ref(&primary_b), &roots)
            .is_err());
        fs::write(&adapter_b.files[0].path, b"changed").unwrap();
        assert!(binding
            .verify_local(&[adapter_b, primary_b], &roots)
            .is_err());
    }

    #[test]
    fn malformed_binding_and_escape_refuse() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let primary = local(outside.path(), false, b"model");
        assert!(PortableArtifactMember::capture(&primary, &[root.path().to_path_buf()]).is_err());
        let primary = local(root.path(), false, b"model");
        let mut member =
            PortableArtifactMember::capture(&primary, &[root.path().to_path_buf()]).unwrap();
        member.files[0].relative_path = PathBuf::from("../model.bin");
        assert!(member.validate().is_err());
    }

    #[test]
    fn authored_options_hash_has_stable_key_order() {
        let first = serde_json::json!({"z": 1, "a": {"d": 2, "b": 3}});
        let second = serde_json::json!({"a": {"b": 3, "d": 2}, "z": 1});
        assert_eq!(
            canonical_authored_options_sha256(&first).unwrap(),
            canonical_authored_options_sha256(&second).unwrap()
        );
        assert!(canonical_authored_options_sha256(&Value::Null).is_err());
    }

    #[test]
    fn pinned_hf_urls_and_pagination_keep_identity_and_origin() {
        let base = reqwest::Url::parse("https://hf.example.test/").unwrap();
        let revision = "a".repeat(40);
        let tree = pinned_url(&base, "owner/model", &revision, None, true).unwrap();
        assert_eq!(
            tree.path(),
            format!("/api/models/owner/model/tree/{revision}")
        );
        assert_eq!(tree.query(), Some("recursive=1&expand=1"));
        let file = pinned_url(
            &base,
            "owner/model",
            &revision,
            Some("q4/model.safetensors"),
            false,
        )
        .unwrap();
        assert_eq!(
            file.path(),
            format!("/owner/model/resolve/{revision}/q4/model.safetensors")
        );
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::LINK,
            "<https://evil.test/next>; rel=\"next\"".parse().unwrap(),
        );
        assert!(next_tree_page(&base, tree.path(), &headers).is_err());
        headers.insert(
            reqwest::header::LINK,
            "</api/models/other/model/tree/main?cursor=2>; rel=\"next\""
                .parse()
                .unwrap(),
        );
        assert!(next_tree_page(&base, tree.path(), &headers).is_err());
        headers.insert(
            reqwest::header::LINK,
            format!("</api/models/owner/model/tree/{revision}?cursor=2>; rel=\"next\"")
                .parse()
                .unwrap(),
        );
        assert!(next_tree_page(&base, tree.path(), &headers)
            .unwrap()
            .is_some());
    }

    #[test]
    fn auto_routes_freeze_distinct_backend_variants() {
        let root = tempfile::tempdir().unwrap();
        let source = local(root.path(), false, b"model");
        let mut primary =
            PortableArtifactMember::capture(&source, &[root.path().to_path_buf()]).unwrap();
        primary.tier = Some("q4".to_owned());
        let options = serde_json::json!({"selectedVariant": "auto"});
        let hash = canonical_authored_options_sha256(&options).unwrap();
        let mlx = PortableModelBinding::new(
            "modelManifestEntry",
            "model",
            "mlx",
            "q4",
            Some("q4".to_owned()),
            hash.clone(),
            vec![primary.clone()],
        )
        .unwrap();
        primary.tier = Some("bf16".to_owned());
        let candle = PortableModelBinding::new(
            "modelManifestEntry",
            "model",
            "candle",
            "bf16",
            Some("bf16".to_owned()),
            hash,
            vec![primary],
        )
        .unwrap();
        let mut routes = BTreeMap::new();
        routes.insert(
            "mlx:macos".to_owned(),
            PortableRouteBinding::new("mlx", "macos", vec![mlx]).unwrap(),
        );
        routes.insert(
            "candle:linux".to_owned(),
            PortableRouteBinding::new("candle", "linux", vec![candle]).unwrap(),
        );
        let mut envelope = PortableJobBinding::new(options, routes).unwrap();
        assert!(envelope.select_route("mlx", "macos").is_ok());
        assert!(envelope.select_route("candle", "linux").is_ok());
        assert!(envelope.select_route("candle", "windows").is_err());
        envelope.routes.get_mut("candle:linux").unwrap().models[0].variant = "q8".to_owned();
        assert!(envelope.validate().is_err());
    }

    #[test]
    fn route_hash_survives_recursively_sorted_store_json() {
        let root = tempfile::tempdir().unwrap();
        let source = local(root.path(), false, b"model");
        let primary =
            PortableArtifactMember::capture(&source, &[root.path().to_path_buf()]).unwrap();
        let options = serde_json::json!({"model":"example"});
        let hash = canonical_authored_options_sha256(&options).unwrap();
        let mut model = PortableModelBinding::new(
            "modelManifestEntry",
            "example",
            "mlx",
            "default",
            None,
            hash,
            vec![primary],
        )
        .unwrap();
        model.checkpoint_plan = Some(PortableCheckpointPlan {
            semantic_digest: "a".repeat(64),
            family: "example".into(),
            primary_layer_id: None,
            layers: vec![],
            context: BTreeMap::from([("nested".into(), serde_json::json!({"z":1,"a":2}))]),
        });
        let route = PortableRouteBinding {
            backend: "mlx".into(),
            platform: "macos".into(),
            models: vec![model],
        };
        let routes = BTreeMap::from([(route.key(), route)]);
        let before = route_map_sha256(&routes).unwrap();
        let mut stored = serde_json::to_value(&routes).unwrap();
        stored.sort_all_objects();
        let restored: BTreeMap<String, PortableRouteBinding> =
            serde_json::from_value(stored).unwrap();
        assert_eq!(before, route_map_sha256(&restored).unwrap());
    }

    #[test]
    fn nested_adapter_slots_preserve_repeated_ids() {
        let payload = serde_json::json!({
            "loras": [{"id": "same"}],
            "advanced": {"phases": [{"loras": [{"id": "same"}]}]}
        });
        let slots = enumerate_portable_array_slots(&payload, &["loras"]);
        assert_eq!(
            slots
                .iter()
                .map(|(slot, _)| slot.as_str())
                .collect::<Vec<_>>(),
            ["advanced/phases/0/loras/0", "loras/0"]
        );
    }

    #[test]
    fn private_source_claim_binds_exact_logical_closure() {
        let file = PortableArtifactFile {
            relative_path: PathBuf::from("encoder/model.safetensors"),
            size_bytes: 4,
            sha256: format!("sha256:{}", crate::hashing::sha256_bytes(b"data")),
        };
        let claim = PortableSourceClaim::new(
            "ltx/alternate-text-encoder",
            PortableSourceProvenance::OperatorOverride,
            vec![file],
        )
        .unwrap();
        assert!(validate_portable_source_claims(std::slice::from_ref(&claim)).is_ok());
        assert!(validate_portable_source_claims(&[claim.clone(), claim.clone()]).is_err());
        let mut changed = claim;
        changed.files[0].size_bytes += 1;
        assert!(changed.validate().is_err());
    }

    #[test]
    fn pinned_source_cannot_masquerade_as_operator_override_or_change_revision() {
        let file = PortableArtifactFile {
            relative_path: PathBuf::from("encoder.safetensors"),
            size_bytes: 4,
            sha256: format!("sha256:{}", crate::hashing::sha256_bytes(b"data")),
        };
        let revision = "a".repeat(40);
        let pinned = PortableSourceProvenance::PinnedRepo {
            repository: "owner/encoder".to_owned(),
            revision: revision.clone(),
        };
        let claim =
            PortableSourceClaim::new("ltx/encoder", pinned.clone(), vec![file.clone()]).unwrap();
        assert!(claim
            .matches_authorized_source("ltx/encoder", &pinned, std::slice::from_ref(&file))
            .is_ok());
        assert!(claim
            .matches_authorized_source(
                "ltx/encoder",
                &PortableSourceProvenance::OperatorOverride,
                std::slice::from_ref(&file)
            )
            .is_err());
        let changed_revision = PortableSourceProvenance::PinnedRepo {
            repository: "owner/encoder".to_owned(),
            revision: "b".repeat(40),
        };
        assert!(claim
            .matches_authorized_source(
                "ltx/encoder",
                &changed_revision,
                std::slice::from_ref(&file)
            )
            .is_err());
        assert!(PortableSourceClaim::new(
            "ltx/encoder",
            PortableSourceProvenance::PinnedRepo {
                repository: "owner/encoder".to_owned(),
                revision: "main".to_owned()
            },
            vec![file]
        )
        .is_err());
    }

    #[test]
    fn worker_claim_member_requires_exact_private_claim_reference() {
        let file = PortableArtifactFile {
            relative_path: PathBuf::from("encoder.safetensors"),
            size_bytes: 4,
            sha256: format!("sha256:{}", crate::hashing::sha256_bytes(b"data")),
        };
        let claim = PortableSourceClaim::new(
            "ltx/alternate-encoder",
            PortableSourceProvenance::OperatorOverride,
            vec![file.clone()],
        )
        .unwrap();
        let mut member = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::WorkerClaim {
                source_id: claim.source_id.clone(),
                provenance: claim.provenance.clone(),
                closure_sha256: claim.closure_sha256.clone(),
            },
            tier: None,
            files: vec![file],
        };
        assert!(claim.matches_worker_claim_member(&member).is_ok());
        if let PortableSourceIdentity::WorkerClaim { provenance, .. } = &mut member.source {
            *provenance = PortableSourceProvenance::PinnedRepo {
                repository: "owner/encoder".to_owned(),
                revision: "a".repeat(40),
            };
        }
        assert!(claim.matches_worker_claim_member(&member).is_err());
        if let PortableSourceIdentity::WorkerClaim { closure_sha256, .. } = &mut member.source {
            *closure_sha256 = format!("sha256:{}", "b".repeat(64));
        }
        assert!(member.validate().is_err());
    }

    #[tokio::test]
    async fn pinned_tree_expands_pages_and_hashes_non_lfs_bytes() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let path = request.split_whitespace().nth(1).unwrap();
                let (body, link) = if path.contains("/resolve/") {
                    ("cfg".to_owned(), None)
                } else if path.contains("cursor=next") {
                    (
                        serde_json::json!([{
                            "type": "file", "path": "q4/model.safetensors", "size": 5,
                            "lfs": {"oid": "a".repeat(64)}
                        }])
                        .to_string(),
                        None,
                    )
                } else {
                    (
                        serde_json::json!([{"type": "file", "path": "config.json", "size": 3}]).to_string(),
                        Some(format!("Link: <http://{address}/api/models/owner/model/tree/{}?cursor=next>; rel=\"next\"\r\n", "a".repeat(40))),
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
                    body.len(), link.unwrap_or_default(), body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let files = resolve_pinned_hf_files(
            &client,
            &format!("http://{address}"),
            None,
            "owner/model",
            &"a".repeat(40),
            &["config.json".to_owned(), "q4/*.safetensors".to_owned()],
        )
        .await
        .unwrap();
        server.join().unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].relative_path, PathBuf::from("config.json"));
        assert_eq!(
            files[0].sha256,
            format!("sha256:{}", crate::hashing::sha256_bytes(b"cfg"))
        );
        assert_eq!(
            files[1].relative_path,
            PathBuf::from("q4/model.safetensors")
        );
        assert_eq!(files[1].sha256, format!("sha256:{}", "a".repeat(64)));
    }
}
