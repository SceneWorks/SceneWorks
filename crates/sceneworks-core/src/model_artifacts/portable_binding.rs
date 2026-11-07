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

pub const PORTABLE_MODEL_BINDING_VERSION: u32 = 1;
const MAX_NON_LFS_BYTES: u64 = 256 * 1024 * 1024;
const MAX_HF_TREE_PAGES: usize = 10_000;

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
        next = next_tree_page(&base, response.headers())?;
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
    Revision { identity: ArtifactIdentity },
    Digest { sha256: String },
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

impl PortableModelBinding {
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
    let bytes = serde_json::to_vec(routes)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
        assert!(next_tree_page(&base, &headers).is_err());
        headers.insert(
            reqwest::header::LINK,
            "</api/models/owner/model/tree/next?cursor=2>; rel=\"next\""
                .parse()
                .unwrap(),
        );
        assert!(next_tree_page(&base, &headers).unwrap().is_some());
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
}
