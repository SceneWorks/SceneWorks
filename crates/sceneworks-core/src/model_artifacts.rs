//! Typed two-tier model-artifact contract.
//!
//! Model manifests describe an authoritative source library. Runtimes consume either that
//! source library directly or an app-owned resolved-local bundle. Keeping those locations
//! distinct prevents a cache path from silently becoming artifact identity, and gives the hot
//! cache a behavior-preserving migration seam.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub const MODEL_ARTIFACT_CONTRACT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactContractError(pub String);

impl std::fmt::Display for ArtifactContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactContractError {}

/// Immutable source identity. `revision` is always the resolved 40-character commit, never a
/// mutable ref such as `main`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub repository: String,
    pub revision: String,
    pub variant: String,
    pub fingerprint: String,
}

impl ArtifactIdentity {
    pub fn pinned(
        repository: impl Into<String>,
        revision: impl Into<String>,
        variant: impl Into<String>,
    ) -> Result<Self, ArtifactContractError> {
        let repository = repository.into();
        let revision = revision.into();
        let variant = variant.into();
        validate_repository(&repository)?;
        validate_immutable_revision(&revision)?;
        if variant.trim().is_empty()
            || variant.contains(['/', '\\'])
            || matches!(variant.as_str(), "." | "..")
        {
            return Err(ArtifactContractError(
                "artifact variant is empty".to_owned(),
            ));
        }
        let fingerprint = format!("{repository}@{revision}:{variant}");
        Ok(Self {
            repository,
            revision,
            variant,
            fingerprint,
        })
    }

    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        let expected = Self::pinned(&self.repository, &self.revision, &self.variant)?;
        if self.fingerprint != expected.fingerprint {
            return Err(ArtifactContractError(
                "artifact identity fingerprint does not match repository, revision, and variant"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactProvenance {
    pub identity: ArtifactIdentity,
    pub fixed_artifact_tier: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactMemberRole {
    Primary,
    RequiredComponent,
    OptionalComponent,
    CoRequisite,
    Lora,
    Control,
    DerivedOverlay,
    Utility,
    TrainingBase,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactFile {
    pub relative_path: PathBuf,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

impl ArtifactFile {
    pub fn new(relative_path: impl Into<PathBuf>) -> Result<Self, ArtifactContractError> {
        let relative_path = relative_path.into();
        validate_concrete_relative_file(&relative_path)?;
        Ok(Self {
            relative_path,
            size_bytes: None,
            sha256: None,
        })
    }
}

/// One concrete member of the immutable bundle closure. Optional components appear here only
/// after selection; later materialization never has to infer them from a manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolvedBundleMember {
    pub role: ArtifactMemberRole,
    pub component_id: Option<String>,
    pub source: ArtifactIdentity,
    pub tier: Option<String>,
    pub source_subpath: PathBuf,
    pub destination: PathBuf,
    pub files: Vec<ArtifactFile>,
}

impl ResolvedBundleMember {
    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        self.source.validate()?;
        validate_relative_path_allow_empty(&self.source_subpath, "artifact source subpath")?;
        validate_relative_path_allow_empty(&self.destination, "artifact destination")?;
        if self.role != ArtifactMemberRole::Primary
            && self
                .component_id
                .as_deref()
                .map_or(true, |value| value.trim().is_empty())
        {
            return Err(ArtifactContractError(format!(
                "{:?} bundle member has no component id",
                self.role
            )));
        }
        if self.files.is_empty() {
            return Err(ArtifactContractError(format!(
                "bundle member {:?} has no concrete file closure",
                self.component_id
            )));
        }
        for file in &self.files {
            validate_concrete_relative_file(&file.relative_path)?;
            if let Some(hash) = &file.sha256 {
                validate_sha256(hash)?;
            }
        }
        Ok(())
    }
}

/// Canonical, immutable closure of every file-bearing primary, required, selected optional,
/// co-requisite, adapter/control and derived-overlay member used by one runtime load.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolvedBundleClosure {
    pub members: Vec<ResolvedBundleMember>,
}

impl ResolvedBundleClosure {
    pub fn new(mut members: Vec<ResolvedBundleMember>) -> Result<Self, ArtifactContractError> {
        for member in &mut members {
            member.files.sort();
            member.files.dedup();
            member.validate()?;
        }
        members.sort_by(|left, right| member_sort_key(left).cmp(&member_sort_key(right)));
        if members
            .iter()
            .filter(|member| member.role == ArtifactMemberRole::Primary)
            .count()
            != 1
        {
            return Err(ArtifactContractError(
                "a resolved bundle must contain exactly one primary member".to_owned(),
            ));
        }
        for pair in members.windows(2) {
            if pair[0].destination == pair[1].destination {
                return Err(ArtifactContractError(format!(
                    "duplicate bundle destination {}",
                    pair[0].destination.display()
                )));
            }
        }
        Ok(Self { members })
    }

    pub fn validate_at_root(&self, root: &Path) -> Result<(), ArtifactContractError> {
        if Self::new(self.members.clone())? != *self {
            return Err(ArtifactContractError(
                "artifact closure is not in canonical sorted form".to_owned(),
            ));
        }
        let root = canonical_or_absolute(root)?;
        for member in &self.members {
            member.validate()?;
            let member_root = rebase_under(&root, &member.destination)?;
            for file in &member.files {
                let path = rebase_under(&member_root, &file.relative_path)?;
                let metadata = std::fs::metadata(&path).map_err(|error| {
                    ArtifactContractError(format!(
                        "artifact closure file {} is unavailable: {error}",
                        path.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(ArtifactContractError(format!(
                        "artifact closure entry {} is not a file",
                        path.display()
                    )));
                }
                if file
                    .size_bytes
                    .is_some_and(|expected| expected != metadata.len())
                {
                    return Err(ArtifactContractError(format!(
                        "artifact closure size changed for {}",
                        path.display()
                    )));
                }
                if let Some(expected) = &file.sha256 {
                    use std::io::Read;
                    let mut input = std::fs::File::open(&path).map_err(|error| {
                        ArtifactContractError(format!("cannot hash {}: {error}", path.display()))
                    })?;
                    let mut digest = Sha256::new();
                    let mut buffer = [0_u8; 64 * 1024];
                    loop {
                        let read = input.read(&mut buffer).map_err(|error| {
                            ArtifactContractError(format!(
                                "cannot hash {}: {error}",
                                path.display()
                            ))
                        })?;
                        if read == 0 {
                            break;
                        }
                        digest.update(&buffer[..read]);
                    }
                    if format!("sha256:{:x}", digest.finalize()) != *expected {
                        return Err(ArtifactContractError(format!(
                            "artifact closure hash changed for {}",
                            path.display()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn rebased_paths(&self, root: &Path) -> Result<Vec<PathBuf>, ArtifactContractError> {
        let root = canonical_or_absolute(root)?;
        self.members
            .iter()
            .flat_map(|member| {
                member.files.iter().map(|file| {
                    let member_root = rebase_under(&root, &member.destination)?;
                    rebase_under(&member_root, &file.relative_path)
                })
            })
            .collect()
    }
}

fn member_sort_key(member: &ResolvedBundleMember) -> (&Path, &ArtifactMemberRole, Option<&str>) {
    (
        member.destination.as_path(),
        &member.role,
        member.component_id.as_deref(),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactCompleteness {
    Complete,
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactAvailability {
    Available,
    UsableStale,
    Missing,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactLocation {
    /// Authoritative, immutable snapshot in the configured source library.
    SourceLibrary { root: PathBuf },
    /// App-owned bundle whose identity and closure were copied from the source artifact.
    ResolvedLocal { root: PathBuf },
}

impl ArtifactLocation {
    pub fn root(&self) -> &Path {
        match self {
            Self::SourceLibrary { root } | Self::ResolvedLocal { root } => root,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolvedModelArtifact {
    pub schema_version: u32,
    pub identity: ArtifactIdentity,
    pub location: ArtifactLocation,
    pub closure: ResolvedBundleClosure,
    pub provenance: ArtifactProvenance,
    pub completeness: ArtifactCompleteness,
    pub availability: ArtifactAvailability,
}

impl ResolvedModelArtifact {
    pub fn validate(&self) -> Result<(), ArtifactContractError> {
        if self.schema_version != MODEL_ARTIFACT_CONTRACT_VERSION {
            return Err(ArtifactContractError(format!(
                "unsupported model artifact contract version {}",
                self.schema_version
            )));
        }
        self.identity.validate()?;
        self.provenance.identity.validate()?;
        if self.identity != self.provenance.identity {
            return Err(ArtifactContractError(
                "artifact identity and provenance identity differ".to_owned(),
            ));
        }
        if self.completeness != ArtifactCompleteness::Complete
            || !matches!(
                self.availability,
                ArtifactAvailability::Available | ArtifactAvailability::UsableStale
            )
        {
            return Err(ArtifactContractError(
                "artifact is not complete and runtime-loadable".to_owned(),
            ));
        }
        self.closure.validate_at_root(self.location.root())
    }

    /// Stable key over the versioned immutable identity, full sorted closure and provenance.
    /// Physical location, availability and usage state are deliberately excluded.
    pub fn cache_key(&self) -> Result<String, ArtifactContractError> {
        self.identity.validate()?;
        let closure = ResolvedBundleClosure::new(self.closure.members.clone())?;
        let canonical = serde_json::to_vec(&(
            self.schema_version,
            &self.identity,
            &closure,
            &self.provenance,
        ))
        .map_err(|error| ArtifactContractError(format!("cannot encode artifact key: {error}")))?;
        Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
    }

    /// Runtime-only lease acquisition seam. Catalog/status callers receive immutable resolution
    /// values but never call this and therefore never stamp usage.
    pub fn acquire_runtime_lease(
        self: &Arc<Self>,
    ) -> Result<ActiveArtifactLease, ArtifactContractError> {
        self.validate()?;
        Ok(ActiveArtifactLease {
            artifact: Arc::clone(self),
            state: Some(Arc::new(LeaseState)),
        })
    }
}

#[derive(Debug)]
struct LeaseState;

/// RAII lifetime guard for one runtime use. Dropping an unmarked lease is merely a release; only
/// `mark_success` crosses the promotion boundary used by later cache stories.
#[derive(Debug)]
pub struct ActiveArtifactLease {
    artifact: Arc<ResolvedModelArtifact>,
    state: Option<Arc<LeaseState>>,
}

impl ActiveArtifactLease {
    pub fn artifact(&self) -> &ResolvedModelArtifact {
        &self.artifact
    }

    pub fn mark_success(mut self) -> PromotionCandidate {
        let _ = self.state.take();
        PromotionCandidate {
            cache_key: self
                .artifact
                .cache_key()
                .expect("a leased artifact has already passed contract validation"),
            identity: self.artifact.identity.clone(),
            closure: self.artifact.closure.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PromotionCandidate {
    pub cache_key: String,
    pub identity: ArtifactIdentity,
    pub closure: ResolvedBundleClosure,
}

/// Typed source-library root. Compatibility path helpers delegate to this type so even legacy
/// consumers enter the shared discovery seam instead of reconstructing configured-HF paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactSourceLibrary {
    root: PathBuf,
}

/// Behavior-preserving runtime resolver. Source discovery and runtime validation are separate:
/// callers first freeze a full concrete closure, then resolve it at the source tier. Later policy
/// can select `ResolvedLocal` without changing any consumer or rediscovering components.
#[derive(Clone, Debug)]
pub struct ModelArtifactResolver {
    source_library: ArtifactSourceLibrary,
}

impl ModelArtifactResolver {
    pub fn new(source_library: ArtifactSourceLibrary) -> Self {
        Self { source_library }
    }

    pub fn source_library(&self) -> &ArtifactSourceLibrary {
        &self.source_library
    }

    pub fn resolve_source(
        &self,
        identity: ArtifactIdentity,
        closure: ResolvedBundleClosure,
        availability: ArtifactAvailability,
    ) -> Result<ResolvedModelArtifact, ArtifactContractError> {
        identity.validate()?;
        let (_, root) = self
            .source_library
            .discover_snapshot(&identity.repository, Some(&identity.revision))?;
        let fixed_artifact_tier = primary_tier(&closure);
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            provenance: ArtifactProvenance {
                identity: identity.clone(),
                fixed_artifact_tier,
            },
            identity,
            location: ArtifactLocation::SourceLibrary { root },
            closure,
            completeness: ArtifactCompleteness::Complete,
            availability,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validate an already materialized app-owned result against the exact source identity and
    /// closure. This does not choose it over the source tier; that policy belongs to later work.
    pub fn resolve_local(
        &self,
        identity: ArtifactIdentity,
        closure: ResolvedBundleClosure,
        root: PathBuf,
        availability: ArtifactAvailability,
    ) -> Result<ResolvedModelArtifact, ArtifactContractError> {
        identity.validate()?;
        let fixed_artifact_tier = primary_tier(&closure);
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            provenance: ArtifactProvenance {
                identity: identity.clone(),
                fixed_artifact_tier,
            },
            identity,
            location: ArtifactLocation::ResolvedLocal { root },
            closure,
            completeness: ArtifactCompleteness::Complete,
            availability,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl ArtifactSourceLibrary {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, ArtifactContractError> {
        let root = root.into();
        if root.as_os_str().is_empty() {
            return Err(ArtifactContractError(
                "source library root is empty".to_owned(),
            ));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn repository_root(&self, repository: &str) -> Result<PathBuf, ArtifactContractError> {
        validate_repository(repository)?;
        let safe = repository.replace('/', "--");
        Ok(self.root.join(format!("models--{safe}")))
    }

    /// Compatibility entry for the historic cross-language repository slug contract. New typed
    /// resolution uses [`Self::repository_root`]; the old path-returning facade retains its exact
    /// slug behavior while still entering this source-library seam.
    pub fn repository_root_from_safe_name(
        &self,
        safe_repository: &str,
    ) -> Result<PathBuf, ArtifactContractError> {
        if safe_repository.is_empty()
            || safe_repository.contains('/')
            || safe_repository.contains('\\')
            || safe_repository == "."
            || safe_repository == ".."
        {
            return Err(ArtifactContractError(format!(
                "invalid safe artifact repository name {safe_repository:?}"
            )));
        }
        Ok(self.root.join(format!("models--{safe_repository}")))
    }

    /// Resolve an immutable source snapshot. `None` reads `refs/main` only to obtain its exact
    /// commit; a caller-provided mutable revision is rejected.
    pub fn discover_snapshot(
        &self,
        repository: &str,
        revision: Option<&str>,
    ) -> Result<(ArtifactIdentity, PathBuf), ArtifactContractError> {
        let repo_root = self.repository_root(repository)?;
        let revision = match revision {
            Some(revision) => {
                validate_immutable_revision(revision)?;
                revision.to_owned()
            }
            None => std::fs::read_to_string(repo_root.join("refs/main"))
                .map_err(|error| {
                    ArtifactContractError(format!("cannot resolve {repository} refs/main: {error}"))
                })?
                .trim()
                .to_owned(),
        };
        validate_immutable_revision(&revision)?;
        let snapshot = repo_root.join("snapshots").join(&revision);
        if !snapshot.is_dir() {
            return Err(ArtifactContractError(format!(
                "source snapshot {repository}@{revision} is not installed"
            )));
        }
        Ok((
            ArtifactIdentity::pinned(repository, revision, "default")?,
            snapshot,
        ))
    }
}

pub fn validate_immutable_revision(revision: &str) -> Result<(), ArtifactContractError> {
    if revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ArtifactContractError(format!(
            "artifact revision {revision:?} is mutable or malformed; expected a lowercase 40-hex commit"
        )))
    }
}

fn validate_repository(repository: &str) -> Result<(), ArtifactContractError> {
    let mut parts = repository.split('/');
    let valid_part = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
            })
    };
    match (parts.next(), parts.next(), parts.next()) {
        (Some(owner), Some(name), None) if valid_part(owner) && valid_part(name) => Ok(()),
        _ => Err(ArtifactContractError(format!(
            "invalid artifact repository {repository:?}"
        ))),
    }
}

fn validate_sha256(hash: &str) -> Result<(), ArtifactContractError> {
    if hash.strip_prefix("sha256:").is_some_and(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        Ok(())
    } else {
        Err(ArtifactContractError(format!(
            "invalid artifact sha256 {hash:?}"
        )))
    }
}

fn validate_relative_path(path: &Path, label: &str) -> Result<(), ArtifactContractError> {
    if path.as_os_str().is_empty() {
        return Err(ArtifactContractError(format!("{label} is empty")));
    }
    validate_relative_path_allow_empty(path, label)
}

fn validate_concrete_relative_file(path: &Path) -> Result<(), ArtifactContractError> {
    validate_relative_path(path, "artifact file")?;
    if path.to_string_lossy().contains(['*', '?', '[', ']']) {
        return Err(ArtifactContractError(format!(
            "artifact closure requires concrete files, not a pattern: {}",
            path.display()
        )));
    }
    Ok(())
}

fn primary_tier(closure: &ResolvedBundleClosure) -> Option<String> {
    closure
        .members
        .iter()
        .find(|member| member.role == ArtifactMemberRole::Primary)
        .and_then(|member| member.tier.clone())
}

fn validate_relative_path_allow_empty(
    path: &Path,
    label: &str,
) -> Result<(), ArtifactContractError> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ArtifactContractError(format!(
            "{label} must be a confined relative path: {}",
            path.display()
        )));
    }
    Ok(())
}

fn canonical_or_absolute(path: &Path) -> Result<PathBuf, ArtifactContractError> {
    if let Ok(path) = std::fs::canonicalize(path) {
        return Ok(path);
    }
    let mut absolute = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().map_err(|error| ArtifactContractError(error.to_string()))?
    };
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => absolute.push(prefix.as_os_str()),
            Component::RootDir => absolute.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !absolute.pop() {
                    return Err(ArtifactContractError(format!(
                        "path escapes filesystem root: {}",
                        path.display()
                    )));
                }
            }
            Component::Normal(value) => absolute.push(value),
        }
    }
    Ok(absolute)
}

fn rebase_under(root: &Path, relative: &Path) -> Result<PathBuf, ArtifactContractError> {
    validate_relative_path_allow_empty(relative, "artifact closure path")?;
    let candidate = root.join(relative);
    let resolved = canonical_or_absolute(&candidate)?;
    if !resolved.starts_with(root) {
        return Err(ArtifactContractError(format!(
            "artifact closure path escaped root: {}",
            candidate.display()
        )));
    }
    Ok(resolved)
}

/// Canonicalize and confine a runtime model path to app/operator-owned roots. This is deliberately
/// independent of source-library discovery: a configured HF root is one allowed source, not the
/// definition of every valid model location.
pub fn confine_artifact_path(
    path: &Path,
    roots: &[PathBuf],
) -> Result<PathBuf, ArtifactContractError> {
    let path = canonical_or_absolute(path)?;
    for root in roots {
        let root = canonical_or_absolute(root)?;
        if path.starts_with(&root) {
            return Ok(path);
        }
    }
    Err(ArtifactContractError(format!(
        "artifact path {} is outside every managed root",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const REV: &str = "0123456789abcdef0123456789abcdef01234567";

    fn primary(destination: &str, file: &str) -> ResolvedBundleMember {
        ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap(),
            tier: Some("q8".to_owned()),
            source_subpath: PathBuf::new(),
            destination: PathBuf::from(destination),
            files: vec![ArtifactFile::new(file).unwrap()],
        }
    }

    #[test]
    fn mutable_revision_and_traversal_fail_closed() {
        assert!(ArtifactIdentity::pinned("SceneWorks/model", "main", "q8").is_err());
        assert!(ArtifactIdentity::pinned("../escape", REV, "q8").is_err());
        assert!(ArtifactFile::new("../weights.safetensors").is_err());
        assert!(ArtifactFile::new("/tmp/weights.safetensors").is_err());
        assert!(ArtifactFile::new("*.safetensors").is_err());
    }

    #[test]
    fn closure_is_canonical_and_cache_key_preserves_every_member() {
        let mut optional = primary("optional", "adapter.safetensors");
        optional.role = ArtifactMemberRole::OptionalComponent;
        optional.component_id = Some("adapter".to_owned());
        let a =
            ResolvedBundleClosure::new(vec![optional.clone(), primary("", "model.safetensors")])
                .unwrap();
        let b =
            ResolvedBundleClosure::new(vec![primary("", "model.safetensors"), optional]).unwrap();
        assert_eq!(a, b);

        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("model.safetensors"), b"model").unwrap();
        std::fs::create_dir(root.path().join("optional")).unwrap();
        std::fs::write(root.path().join("optional/adapter.safetensors"), b"adapter").unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::SourceLibrary {
                root: root.path().to_owned(),
            },
            closure: a,
            provenance: ArtifactProvenance {
                identity,
                fixed_artifact_tier: Some("q8".to_owned()),
            },
            completeness: ArtifactCompleteness::Complete,
            availability: ArtifactAvailability::UsableStale,
        };
        artifact.validate().unwrap();
        let before = artifact.cache_key().unwrap();
        let mut changed = artifact.clone();
        changed.closure.members[1].component_id = Some("different".to_owned());
        assert_ne!(before, changed.cache_key().unwrap());
    }

    #[test]
    fn incomplete_artifact_cannot_acquire_a_runtime_lease() {
        let root = tempfile::tempdir().unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let artifact = Arc::new(ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::ResolvedLocal {
                root: root.path().to_owned(),
            },
            closure: ResolvedBundleClosure::new(vec![primary("", "missing.safetensors")]).unwrap(),
            provenance: ArtifactProvenance {
                identity,
                fixed_artifact_tier: Some("q8".to_owned()),
            },
            completeness: ArtifactCompleteness::Incomplete,
            availability: ArtifactAvailability::Available,
        });
        assert!(artifact.acquire_runtime_lease().is_err());
    }

    #[test]
    fn source_discovery_resolves_ref_to_an_immutable_identity() {
        let root = tempfile::tempdir().unwrap();
        let library = ArtifactSourceLibrary::new(root.path()).unwrap();
        let repo = library.repository_root("SceneWorks/model").unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join(REV)).unwrap();
        std::fs::write(repo.join("refs/main"), format!("{REV}\n")).unwrap();
        let (identity, snapshot) = library.discover_snapshot("SceneWorks/model", None).unwrap();
        assert_eq!(identity.revision, REV);
        assert_eq!(snapshot, repo.join("snapshots").join(REV));
        assert!(library
            .discover_snapshot("SceneWorks/model", Some("main"))
            .is_err());
    }

    #[test]
    fn source_and_local_resolution_preserve_identity_tier_closure_and_cache_key() {
        let source = tempfile::tempdir().unwrap();
        let library = ArtifactSourceLibrary::new(source.path()).unwrap();
        let repo = library.repository_root("SceneWorks/model").unwrap();
        let snapshot = repo.join("snapshots").join(REV);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"model").unwrap();
        let closure = ResolvedBundleClosure::new(vec![primary("", "model.safetensors")]).unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let resolver = ModelArtifactResolver::new(library);
        let from_source = resolver
            .resolve_source(
                identity.clone(),
                closure.clone(),
                ArtifactAvailability::UsableStale,
            )
            .unwrap();
        assert_eq!(
            from_source.provenance.fixed_artifact_tier.as_deref(),
            Some("q8")
        );
        let serialized = serde_json::to_vec(&from_source).unwrap();
        let deserialized: ResolvedModelArtifact = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(deserialized, from_source);

        let local = tempfile::tempdir().unwrap();
        std::fs::write(local.path().join("model.safetensors"), b"model").unwrap();
        let from_local = resolver
            .resolve_local(
                identity,
                closure,
                local.path().to_owned(),
                ArtifactAvailability::Available,
            )
            .unwrap();
        assert_eq!(from_source.identity, from_local.identity);
        assert_eq!(from_source.closure, from_local.closure);
        assert_eq!(
            from_source.cache_key().unwrap(),
            from_local.cache_key().unwrap()
        );
        let candidate = Arc::new(from_local)
            .acquire_runtime_lease()
            .unwrap()
            .mark_success();
        assert_eq!(candidate.cache_key, from_source.cache_key().unwrap());

        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        assert!(
            from_source.validate().is_err(),
            "usable-stale still requires its exact closure"
        );
    }
}
