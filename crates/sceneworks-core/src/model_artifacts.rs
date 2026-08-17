//! Typed two-tier model-artifact contract.
//!
//! Model manifests describe an authoritative source library. Runtimes consume either that
//! source library directly or an app-owned resolved-local bundle. Keeping those locations
//! distinct prevents a cache path from silently becoming artifact identity, and gives the hot
//! cache a behavior-preserving migration seam.

#[path = "resolved_cache.rs"]
pub mod resolved_cache;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

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

/// One exact source file and its final logical output path. Indices refer back to the canonical
/// closure so callers do not duplicate identity, role, component, tier, size, or hash fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedBundleFileLocation {
    pub member_index: usize,
    pub file_index: usize,
    pub source_path: PathBuf,
    pub output_path: PathBuf,
}

impl ResolvedBundleClosure {
    pub fn new(mut members: Vec<ResolvedBundleMember>) -> Result<Self, ArtifactContractError> {
        for member in &mut members {
            member.files.sort();
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
        let mut output_paths = Vec::new();
        for member in &members {
            for file in &member.files {
                let path = member.destination.join(&file.relative_path);
                output_paths.push(portable_relative_path(&path, "artifact output file")?);
            }
        }
        output_paths.sort();
        for (index, path) in output_paths.iter().enumerate() {
            if output_paths.get(index + 1) == Some(path) {
                return Err(ArtifactContractError(format!(
                    "duplicate artifact output file {path}"
                )));
            }
            for descendant in output_paths.iter().skip(index + 1) {
                if descendant
                    .strip_prefix(path)
                    .is_some_and(|suffix| suffix.starts_with('/'))
                {
                    return Err(ArtifactContractError(format!(
                        "artifact output file {path} is an ancestor of {descendant}"
                    )));
                }
            }
        }
        Ok(Self { members })
    }

    pub fn validate_at_root(&self, root: &Path) -> Result<(), ArtifactContractError> {
        self.validate_canonical()?;
        let root = canonical_or_absolute(root)?;
        for member in &self.members {
            member.validate()?;
            let member_root = rebase_under(&root, &member.destination)?;
            for file in &member.files {
                let path = rebase_under(&member_root, &file.relative_path)?;
                validate_artifact_file(&path, file)?;
            }
        }
        Ok(())
    }

    /// Resolve every member from its exact repository, immutable revision and source subpath.
    /// Hugging Face's normal final-file symlink into the same repository's `blobs` directory is
    /// accepted; intermediate links and links into another repository or arbitrary host path fail
    /// closed. Returned source paths are canonical targets selected during validation, so a later
    /// swap of a snapshot-visible link cannot redirect the materializer outside the repository.
    pub fn source_file_locations(
        &self,
        primary_identity: &ArtifactIdentity,
        primary_snapshot_root: &Path,
    ) -> Result<Vec<ResolvedBundleFileLocation>, ArtifactContractError> {
        self.validate_canonical()?;
        primary_identity.validate()?;
        let library = source_library_for_primary_snapshot(primary_identity, primary_snapshot_root)?;
        let primary = self
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .expect("canonical closure has exactly one primary");
        if &primary.source != primary_identity {
            return Err(ArtifactContractError(
                "primary bundle member identity differs from artifact identity".to_owned(),
            ));
        }

        let mut locations = Vec::new();
        for (member_index, member) in self.members.iter().enumerate() {
            member.validate()?;
            let (_, snapshot) = library
                .discover_snapshot(&member.source.repository, Some(&member.source.revision))?;
            validate_source_snapshot(&library, &member.source, &snapshot)?;
            let member_root = snapshot.join(&member.source_subpath);
            validate_source_ancestors(&snapshot, &member_root)?;
            for (file_index, file) in member.files.iter().enumerate() {
                let source_path = member_root.join(&file.relative_path);
                let source_path =
                    validate_source_file(&library, &member.source, &snapshot, &source_path, file)?;
                let output_path = member.destination.join(&file.relative_path);
                validate_relative_path(&output_path, "artifact output file")?;
                locations.push(ResolvedBundleFileLocation {
                    member_index,
                    file_index,
                    source_path,
                    output_path,
                });
            }
        }
        Ok(locations)
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

    fn validate_canonical(&self) -> Result<(), ArtifactContractError> {
        if Self::new(self.members.clone())? != *self {
            return Err(ArtifactContractError(
                "artifact closure is not in canonical sorted form".to_owned(),
            ));
        }
        Ok(())
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
        match &self.location {
            ArtifactLocation::SourceLibrary { root } => {
                self.closure.source_file_locations(&self.identity, root)?;
                Ok(())
            }
            ArtifactLocation::ResolvedLocal { root } => self.closure.validate_at_root(root),
        }
    }

    /// Stable key over the versioned immutable identity, full sorted closure and provenance.
    /// Physical location, availability, usage state, and post-copy verification enrichment are
    /// deliberately excluded.
    pub fn cache_key(&self) -> Result<String, ArtifactContractError> {
        self.identity.validate()?;
        self.provenance.identity.validate()?;
        if self.identity != self.provenance.identity {
            return Err(ArtifactContractError(
                "artifact identity and provenance identity differ".to_owned(),
            ));
        }
        let closure = ResolvedBundleClosure::new(self.closure.members.clone())?;
        let canonical = cache_key_payload(
            self.schema_version,
            &self.identity,
            &closure,
            &self.provenance,
        )?;
        Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheKeyIdentity<'a> {
    repository: &'a str,
    revision: &'a str,
    variant: &'a str,
    fingerprint: &'a str,
}

impl<'a> From<&'a ArtifactIdentity> for CacheKeyIdentity<'a> {
    fn from(identity: &'a ArtifactIdentity) -> Self {
        Self {
            repository: &identity.repository,
            revision: &identity.revision,
            variant: &identity.variant,
            fingerprint: &identity.fingerprint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheKeyMember<'a> {
    role: &'a ArtifactMemberRole,
    component_id: Option<&'a str>,
    source: CacheKeyIdentity<'a>,
    tier: Option<&'a str>,
    source_subpath: String,
    destination: String,
    files: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheKeyProvenance<'a> {
    identity: CacheKeyIdentity<'a>,
    fixed_artifact_tier: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheKeyPayload<'a> {
    schema_version: u32,
    identity: CacheKeyIdentity<'a>,
    members: Vec<CacheKeyMember<'a>>,
    provenance: CacheKeyProvenance<'a>,
}

fn cache_key_payload(
    schema_version: u32,
    identity: &ArtifactIdentity,
    closure: &ResolvedBundleClosure,
    provenance: &ArtifactProvenance,
) -> Result<Vec<u8>, ArtifactContractError> {
    let mut members = closure
        .members
        .iter()
        .map(|member| {
            let mut files = member
                .files
                .iter()
                .map(|file| portable_relative_path(&file.relative_path, "artifact file"))
                .collect::<Result<Vec<_>, _>>()?;
            files.sort();
            Ok(CacheKeyMember {
                role: &member.role,
                component_id: member.component_id.as_deref(),
                source: CacheKeyIdentity::from(&member.source),
                tier: member.tier.as_deref(),
                source_subpath: portable_relative_path(
                    &member.source_subpath,
                    "artifact source subpath",
                )?,
                destination: portable_relative_path(&member.destination, "artifact destination")?,
                files,
            })
        })
        .collect::<Result<Vec<_>, ArtifactContractError>>()?;
    members.sort();
    serde_json::to_vec(&CacheKeyPayload {
        schema_version,
        identity: CacheKeyIdentity::from(identity),
        members,
        provenance: CacheKeyProvenance {
            identity: CacheKeyIdentity::from(&provenance.identity),
            fixed_artifact_tier: provenance.fixed_artifact_tier.as_deref(),
        },
    })
    .map_err(|error| ArtifactContractError(format!("cannot encode artifact key: {error}")))
}

/// Shared runtime lease accounting for eviction/materialization policy. Resolvers that share this
/// value observe one active count; immutable catalog resolution never touches it.
#[derive(Clone, Debug, Default)]
pub struct ActiveArtifactLeaseRegistry {
    active: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl ActiveArtifactLeaseRegistry {
    pub fn active_lease_count(&self, cache_key: &str) -> usize {
        *lock_unpoisoned(&self.active).get(cache_key).unwrap_or(&0)
    }

    pub fn is_active(&self, cache_key: &str) -> bool {
        self.active_lease_count(cache_key) != 0
    }

    pub fn active_cache_keys(&self) -> Vec<String> {
        lock_unpoisoned(&self.active).keys().cloned().collect()
    }

    fn acquire(&self, cache_key: &str) {
        *lock_unpoisoned(&self.active)
            .entry(cache_key.to_owned())
            .or_default() += 1;
    }

    fn release(&self, cache_key: &str) {
        let mut active = lock_unpoisoned(&self.active);
        let remove = match active.get_mut(cache_key) {
            Some(count) if *count > 1 => {
                *count -= 1;
                false
            }
            Some(_) => true,
            None => false,
        };
        if remove {
            active.remove(cache_key);
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug)]
struct LeaseState {
    registry: ActiveArtifactLeaseRegistry,
    cache_key: String,
}

/// RAII lifetime guard for one runtime use. Dropping an unmarked lease is merely a release; only
/// `mark_success` crosses the promotion boundary used by later cache stories.
#[derive(Debug)]
pub struct ActiveArtifactLease {
    artifact: Arc<ResolvedModelArtifact>,
    state: Option<LeaseState>,
}

impl ActiveArtifactLease {
    pub fn artifact(&self) -> &ResolvedModelArtifact {
        &self.artifact
    }

    pub fn mark_success(mut self) -> PromotionCandidate {
        let candidate = PromotionCandidate {
            cache_key: self
                .artifact
                .cache_key()
                .expect("a leased artifact has already passed contract validation"),
            artifact: self.artifact.as_ref().clone(),
        };
        // Success is deliberately separate from release. Taking the state here releases the
        // active count only after the promotion candidate has been formed.
        if let Some(state) = self.state.take() {
            state.registry.release(&state.cache_key);
        }
        candidate
    }
}

impl Drop for ActiveArtifactLease {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.registry.release(&state.cache_key);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PromotionCandidate {
    pub cache_key: String,
    pub artifact: ResolvedModelArtifact,
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
    lease_registry: ActiveArtifactLeaseRegistry,
}

impl ModelArtifactResolver {
    pub fn new(source_library: ArtifactSourceLibrary) -> Self {
        Self {
            source_library,
            lease_registry: ActiveArtifactLeaseRegistry::default(),
        }
    }

    pub fn with_lease_registry(
        source_library: ArtifactSourceLibrary,
        lease_registry: ActiveArtifactLeaseRegistry,
    ) -> Self {
        Self {
            source_library,
            lease_registry,
        }
    }

    pub fn source_library(&self) -> &ArtifactSourceLibrary {
        &self.source_library
    }

    pub fn lease_registry(&self) -> &ActiveArtifactLeaseRegistry {
        &self.lease_registry
    }

    /// Resolve only the immutable source snapshot identity and location. This is the typed
    /// behavior-preserving migration seam for legacy runtime loaders that still assemble their
    /// concrete component closure at a higher layer.
    pub fn discover_source_snapshot(
        &self,
        repository: &str,
        revision: Option<&str>,
    ) -> Result<(ArtifactIdentity, PathBuf), ArtifactContractError> {
        self.source_library.discover_snapshot(repository, revision)
    }

    /// Resolve a configured ref name (for example `main`) to its immutable snapshot identity.
    /// The returned contract never retains the mutable name.
    pub fn discover_source_reference(
        &self,
        repository: &str,
        reference: &str,
    ) -> Result<(ArtifactIdentity, PathBuf), ArtifactContractError> {
        self.source_library
            .discover_snapshot_reference(repository, reference)
    }

    /// Runtime-only lease acquisition. Catalog/status callers resolve artifacts but do not call
    /// this boundary, so discovery cannot stamp usage.
    pub fn acquire_runtime_lease(
        &self,
        artifact: &Arc<ResolvedModelArtifact>,
    ) -> Result<ActiveArtifactLease, ArtifactContractError> {
        artifact.validate()?;
        let cache_key = artifact.cache_key()?;
        self.lease_registry.acquire(&cache_key);
        Ok(ActiveArtifactLease {
            artifact: Arc::clone(artifact),
            state: Some(LeaseState {
                registry: self.lease_registry.clone(),
                cache_key,
            }),
        })
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

    /// Typed compatibility seam for maintenance code that already holds a repository root rather
    /// than its logical repo id. The root must be one direct `models--*` child of this library.
    pub fn snapshots_root_from_repository_root(
        &self,
        repository_root: &Path,
    ) -> Result<PathBuf, ArtifactContractError> {
        let library_root = canonical_or_absolute(&self.root)?;
        let repository_root = canonical_or_absolute(repository_root)?;
        let is_repository = repository_root.parent() == Some(library_root.as_path())
            && repository_root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--") && name.len() > "models--".len());
        if !is_repository {
            return Err(ArtifactContractError(format!(
                "artifact repository root {} is not a direct source-library repository",
                repository_root.display()
            )));
        }
        Ok(repository_root.join("snapshots"))
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

    pub fn discover_snapshot_reference(
        &self,
        repository: &str,
        reference: &str,
    ) -> Result<(ArtifactIdentity, PathBuf), ArtifactContractError> {
        if validate_immutable_revision(reference).is_ok() {
            return self.discover_snapshot(repository, Some(reference));
        }
        validate_relative_path(Path::new(reference), "artifact source reference")?;
        if Path::new(reference).components().count() != 1 {
            return Err(ArtifactContractError(format!(
                "artifact source reference {reference:?} is not a confined ref name"
            )));
        }
        let repo_root = self.repository_root(repository)?;
        let revision = std::fs::read_to_string(repo_root.join("refs").join(reference))
            .map_err(|error| {
                ArtifactContractError(format!(
                    "cannot resolve {repository} ref {reference:?}: {error}"
                ))
            })?
            .trim()
            .to_owned();
        self.discover_snapshot(repository, Some(&revision))
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

fn validate_artifact_file(path: &Path, file: &ArtifactFile) -> Result<(), ArtifactContractError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
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
        let mut input = std::fs::File::open(path).map_err(|error| {
            ArtifactContractError(format!("cannot hash {}: {error}", path.display()))
        })?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = input.read(&mut buffer).map_err(|error| {
                ArtifactContractError(format!("cannot hash {}: {error}", path.display()))
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
    Ok(())
}

fn source_library_for_primary_snapshot(
    identity: &ArtifactIdentity,
    snapshot: &Path,
) -> Result<ArtifactSourceLibrary, ArtifactContractError> {
    let revision = snapshot.file_name().and_then(|value| value.to_str());
    let snapshots = snapshot.parent();
    let repository = snapshots.and_then(Path::parent);
    let library = repository.and_then(Path::parent);
    if revision != Some(identity.revision.as_str())
        || snapshots
            .and_then(Path::file_name)
            .and_then(|value| value.to_str())
            != Some("snapshots")
    {
        return Err(ArtifactContractError(format!(
            "source snapshot {} does not encode the artifact revision",
            snapshot.display()
        )));
    }
    let library = library.ok_or_else(|| {
        ArtifactContractError(format!(
            "source snapshot {} is outside a Hugging Face repository",
            snapshot.display()
        ))
    })?;
    let library = ArtifactSourceLibrary::new(library)?;
    let expected = library
        .repository_root(&identity.repository)?
        .join("snapshots")
        .join(&identity.revision);
    if canonical_or_absolute(&expected)? != canonical_or_absolute(snapshot)? {
        return Err(ArtifactContractError(format!(
            "source snapshot {} does not match {}@{}",
            snapshot.display(),
            identity.repository,
            identity.revision
        )));
    }
    validate_source_snapshot(&library, identity, snapshot)?;
    Ok(library)
}

fn validate_source_snapshot(
    library: &ArtifactSourceLibrary,
    identity: &ArtifactIdentity,
    snapshot: &Path,
) -> Result<(), ArtifactContractError> {
    let repository = library.repository_root(&identity.repository)?;
    let snapshots = repository.join("snapshots");
    let expected = snapshots.join(&identity.revision);
    if canonical_or_absolute(&expected)? != canonical_or_absolute(snapshot)? {
        return Err(ArtifactContractError(format!(
            "source snapshot {} differs from the exact immutable repository location",
            snapshot.display()
        )));
    }
    for (path, label) in [
        (&repository, "source repository"),
        (&snapshots, "source snapshots directory"),
        (&expected, "source snapshot"),
    ] {
        let metadata = std::fs::symlink_metadata(path).map_err(|error| {
            ArtifactContractError(format!(
                "{label} {} is unavailable: {error}",
                path.display()
            ))
        })?;
        if metadata.file_type().is_symlink()
            || metadata_is_reparse_point(&metadata)
            || !metadata.is_dir()
        {
            return Err(ArtifactContractError(format!(
                "{label} {} is a link, reparse point, or not a directory",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_source_ancestors(
    snapshot: &Path,
    descendant: &Path,
) -> Result<(), ArtifactContractError> {
    let relative = descendant.strip_prefix(snapshot).map_err(|_| {
        ArtifactContractError(format!(
            "artifact source path escaped snapshot: {}",
            descendant.display()
        ))
    })?;
    let mut cursor = snapshot.to_path_buf();
    for component in relative.components() {
        cursor.push(component.as_os_str());
        match std::fs::symlink_metadata(&cursor) {
            Ok(metadata)
                if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) =>
            {
                return Err(ArtifactContractError(format!(
                    "artifact source ancestor {} is a link or reparse point",
                    cursor.display()
                )));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(ArtifactContractError(format!(
                    "artifact source ancestor {} is not a directory",
                    cursor.display()
                )));
            }
            Ok(_) => {}
            Err(error) => {
                return Err(ArtifactContractError(format!(
                    "artifact source ancestor {} is unavailable: {error}",
                    cursor.display()
                )));
            }
        }
    }
    Ok(())
}

fn validate_source_file(
    library: &ArtifactSourceLibrary,
    identity: &ArtifactIdentity,
    snapshot: &Path,
    path: &Path,
    file: &ArtifactFile,
) -> Result<PathBuf, ArtifactContractError> {
    let parent = path.parent().ok_or_else(|| {
        ArtifactContractError(format!(
            "artifact source file {} has no parent",
            path.display()
        ))
    })?;
    validate_source_ancestors(snapshot, parent)?;
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        ArtifactContractError(format!(
            "artifact closure file {} is unavailable: {error}",
            path.display()
        ))
    })?;
    if metadata_is_reparse_point(&metadata) && !metadata.file_type().is_symlink() {
        return Err(ArtifactContractError(format!(
            "artifact source file {} is a reparse point",
            path.display()
        )));
    }
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        ArtifactContractError(format!(
            "cannot resolve artifact source file {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        let repository = library.repository_root(&identity.repository)?;
        let blobs = repository.join("blobs");
        let blobs_metadata = std::fs::symlink_metadata(&blobs).map_err(|error| {
            ArtifactContractError(format!(
                "source blobs directory {} is unavailable: {error}",
                blobs.display()
            ))
        })?;
        if blobs_metadata.file_type().is_symlink()
            || metadata_is_reparse_point(&blobs_metadata)
            || !blobs_metadata.is_dir()
        {
            return Err(ArtifactContractError(format!(
                "source blobs directory {} is linked, reparsed, or not a directory",
                blobs.display()
            )));
        }
        let canonical_repository = std::fs::canonicalize(&repository).map_err(|error| {
            ArtifactContractError(format!(
                "source repository {} is unavailable: {error}",
                repository.display()
            ))
        })?;
        let blobs = std::fs::canonicalize(&blobs).map_err(|error| {
            ArtifactContractError(format!(
                "source blobs directory {} is unavailable: {error}",
                blobs.display()
            ))
        })?;
        if blobs.parent() != Some(canonical_repository.as_path()) || !canonical.starts_with(&blobs)
        {
            return Err(ArtifactContractError(format!(
                "artifact source link {} escapes its repository blobs",
                path.display()
            )));
        }
    } else if !canonical.starts_with(canonical_or_absolute(snapshot)?) {
        return Err(ArtifactContractError(format!(
            "artifact source file {} escapes its immutable snapshot",
            path.display()
        )));
    }
    validate_artifact_file(&canonical, file)?;
    Ok(canonical)
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    false
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

fn portable_relative_path(path: &Path, label: &str) -> Result<String, ArtifactContractError> {
    validate_relative_path_allow_empty(path, label)?;
    path.components()
        .map(|component| match component {
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    ArtifactContractError(format!("{label} is not valid UTF-8: {}", path.display()))
                })?;
                if value.contains('\\') {
                    return Err(ArtifactContractError(format!(
                        "{label} uses a non-portable separator: {}",
                        path.display()
                    )));
                }
                Ok(value)
            }
            _ => unreachable!("relative-path validation permits only normal components"),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|components| components.join("/"))
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

    // Canonicalize the deepest existing ancestor, then append only its missing tail. Falling back
    // to the wholly lexical path would let `root/link/missing` pass a prefix check even when
    // `link` is an in-root symlink to a directory outside `root`.
    let mut ancestor = absolute.as_path();
    let mut missing = Vec::new();
    loop {
        match std::fs::canonicalize(ancestor) {
            Ok(mut canonical) => {
                for component in missing.iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let Some(name) = ancestor.file_name() else {
                    return Err(ArtifactContractError(format!(
                        "cannot resolve path {}: {error}",
                        path.display()
                    )));
                };
                missing.push(name.to_os_string());
                let Some(parent) = ancestor.parent() else {
                    return Err(ArtifactContractError(format!(
                        "cannot resolve path {}: {error}",
                        path.display()
                    )));
                };
                ancestor = parent;
            }
            Err(error) => {
                return Err(ArtifactContractError(format!(
                    "cannot resolve path {}: {error}",
                    path.display()
                )));
            }
        }
    }
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
    const OTHER_REV: &str = "1111111111111111111111111111111111111111";

    fn snapshot(root: &Path, repository: &str, revision: &str) -> PathBuf {
        let library = ArtifactSourceLibrary::new(root).unwrap();
        let snapshot = library
            .repository_root(repository)
            .unwrap()
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        snapshot
    }

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

    fn component(
        role: ArtifactMemberRole,
        component_id: &str,
        destination: &str,
        file: &str,
    ) -> ResolvedBundleMember {
        let mut member = primary(destination, file);
        member.role = role;
        member.component_id = Some(component_id.to_owned());
        member
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
        let optional = component(
            ArtifactMemberRole::OptionalComponent,
            "adapter",
            "optional",
            "adapter.safetensors",
        );
        let a =
            ResolvedBundleClosure::new(vec![optional.clone(), primary("", "model.safetensors")])
                .unwrap();
        let b =
            ResolvedBundleClosure::new(vec![primary("", "model.safetensors"), optional]).unwrap();
        assert_eq!(a, b);

        let root = tempfile::tempdir().unwrap();
        let snapshot = snapshot(root.path(), "SceneWorks/model", REV);
        std::fs::write(snapshot.join("model.safetensors"), b"model").unwrap();
        std::fs::write(snapshot.join("adapter.safetensors"), b"adapter").unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::SourceLibrary { root: snapshot },
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
    fn closure_accepts_disjoint_multi_member_outputs_and_rejects_file_collisions() {
        let shared_destination = ResolvedBundleClosure::new(vec![
            primary("bundle", "model.safetensors"),
            component(
                ArtifactMemberRole::RequiredComponent,
                "encoder",
                "bundle",
                "encoder.safetensors",
            ),
            component(
                ArtifactMemberRole::CoRequisite,
                "tokenizer",
                "bundle/tokenizer",
                "tokenizer.json",
            ),
        ])
        .expect("disjoint files may share and nest member destinations");
        assert_eq!(shared_destination.members.len(), 3);

        let duplicate = ResolvedBundleClosure::new(vec![
            primary("bundle", "model.safetensors"),
            component(
                ArtifactMemberRole::RequiredComponent,
                "duplicate",
                "bundle",
                "model.safetensors",
            ),
        ])
        .unwrap_err();
        assert_eq!(
            duplicate.0,
            "duplicate artifact output file bundle/model.safetensors"
        );

        let ancestor = ResolvedBundleClosure::new(vec![
            primary("", "weights"),
            component(
                ArtifactMemberRole::RequiredComponent,
                "descendant",
                "weights",
                "encoder.safetensors",
            ),
        ])
        .unwrap_err();
        assert_eq!(
            ancestor.0,
            "artifact output file weights is an ancestor of weights/encoder.safetensors"
        );
    }

    #[test]
    fn cache_key_uses_portable_paths_and_ignores_verification_enrichment() {
        let mut native_path = PathBuf::from("weights");
        native_path.push("nested");
        native_path.push("model.safetensors");
        assert_eq!(
            portable_relative_path(&native_path, "test path").unwrap(),
            "weights/nested/model.safetensors"
        );

        let mut primary = primary("bundle", "weights/model.safetensors");
        primary.source_subpath = PathBuf::from("q8/transformer");
        primary
            .files
            .push(ArtifactFile::new("config/config.json").unwrap());
        let closure = ResolvedBundleClosure::new(vec![
            component(
                ArtifactMemberRole::OptionalComponent,
                "adapter",
                "bundle",
                "adapters/style.safetensors",
            ),
            primary,
        ])
        .unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let provenance = ArtifactProvenance {
            identity: identity.clone(),
            fixed_artifact_tier: Some("q8".to_owned()),
        };
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::SourceLibrary {
                root: PathBuf::from("/source/is/not/cache/identity"),
            },
            closure,
            provenance: provenance.clone(),
            completeness: ArtifactCompleteness::Complete,
            availability: ArtifactAvailability::Available,
        };
        let payload = String::from_utf8(
            cache_key_payload(
                artifact.schema_version,
                &identity,
                &artifact.closure,
                &provenance,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(payload.contains("q8/transformer"));
        assert!(payload.contains("weights/model.safetensors"));
        assert!(payload.contains("adapters/style.safetensors"));
        assert!(!payload.contains('\\'));
        assert!(!payload.contains("sizeBytes"));
        assert!(!payload.contains("sha256"));

        let before = artifact.cache_key().unwrap();
        let mut enriched = artifact.clone();
        for (member_index, member) in enriched.closure.members.iter_mut().enumerate() {
            for (file_index, file) in member.files.iter_mut().enumerate() {
                file.size_bytes = Some((member_index * 10 + file_index + 1) as u64);
                file.sha256 = Some(format!("sha256:{:064x}", member_index * 10 + file_index));
            }
        }
        assert_eq!(before, enriched.cache_key().unwrap());

        let mut reordered = artifact.clone();
        reordered.closure.members.reverse();
        for member in &mut reordered.closure.members {
            member.files.reverse();
        }
        assert_eq!(before, reordered.cache_key().unwrap());

        let mut changed_schema = artifact.clone();
        changed_schema.schema_version += 1;
        assert_ne!(before, changed_schema.cache_key().unwrap());

        let mut changed_identity = artifact.clone();
        let changed = ArtifactIdentity::pinned(
            "SceneWorks/model",
            "1111111111111111111111111111111111111111",
            "q4",
        )
        .unwrap();
        changed_identity.identity = changed.clone();
        changed_identity.provenance.identity = changed;
        assert_ne!(before, changed_identity.cache_key().unwrap());

        let mut changed_role = artifact.clone();
        changed_role
            .closure
            .members
            .iter_mut()
            .find(|member| member.component_id.as_deref() == Some("adapter"))
            .unwrap()
            .role = ArtifactMemberRole::RequiredComponent;
        assert_ne!(before, changed_role.cache_key().unwrap());

        let mut changed_file = artifact.clone();
        changed_file.closure.members[0].files[0].relative_path =
            PathBuf::from("adapters/other.safetensors");
        assert_ne!(before, changed_file.cache_key().unwrap());

        let mut changed_provenance = artifact;
        changed_provenance.provenance.fixed_artifact_tier = Some("q4".to_owned());
        assert_ne!(before, changed_provenance.cache_key().unwrap());
    }

    #[test]
    fn incomplete_artifact_cannot_acquire_a_runtime_lease() {
        let root = tempfile::tempdir().unwrap();
        let resolver = ModelArtifactResolver::new(
            ArtifactSourceLibrary::new(root.path().join("source")).unwrap(),
        );
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
        assert!(resolver.acquire_runtime_lease(&artifact).is_err());
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
        let expected_source = from_source.clone();
        let candidate = resolver
            .acquire_runtime_lease(&Arc::new(from_source.clone()))
            .unwrap()
            .mark_success();
        assert_eq!(candidate.cache_key, from_source.cache_key().unwrap());
        assert_eq!(candidate.artifact, expected_source);
        assert!(matches!(
            candidate.artifact.location,
            ArtifactLocation::SourceLibrary { .. }
        ));
        assert_eq!(
            candidate.artifact.availability,
            ArtifactAvailability::UsableStale
        );
        let serialized = serde_json::to_value(&candidate).unwrap();
        assert_eq!(
            serialized["artifact"]["schemaVersion"],
            MODEL_ARTIFACT_CONTRACT_VERSION
        );
        let round_trip: PromotionCandidate = serde_json::from_value(serialized).unwrap();
        assert_eq!(round_trip, candidate);

        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        assert!(
            from_source.validate().is_err(),
            "usable-stale still requires its exact closure"
        );
    }

    #[test]
    fn source_enumeration_resolves_multi_repo_tier_and_derived_members_exactly() {
        let source = tempfile::tempdir().unwrap();
        let primary_snapshot = snapshot(source.path(), "SceneWorks/model", REV);
        let tokenizer_snapshot = snapshot(source.path(), "SceneWorks/tokenizer", OTHER_REV);
        std::fs::create_dir_all(primary_snapshot.join("q8")).unwrap();
        std::fs::create_dir_all(primary_snapshot.join("derived")).unwrap();
        std::fs::create_dir_all(tokenizer_snapshot.join("tokenizer")).unwrap();
        std::fs::write(primary_snapshot.join("q8/model.safetensors"), b"model").unwrap();
        std::fs::write(
            primary_snapshot.join("derived/tokenizer_config.json"),
            b"derived",
        )
        .unwrap();
        std::fs::write(
            tokenizer_snapshot.join("tokenizer/tokenizer.json"),
            b"tokenizer",
        )
        .unwrap();

        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let mut primary = primary("", "model.safetensors");
        primary.source_subpath = PathBuf::from("q8");
        let mut tokenizer = component(
            ArtifactMemberRole::CoRequisite,
            "tokenizer",
            "tokenizer",
            "tokenizer.json",
        );
        tokenizer.source =
            ArtifactIdentity::pinned("SceneWorks/tokenizer", OTHER_REV, "default").unwrap();
        tokenizer.source_subpath = PathBuf::from("tokenizer");
        let mut derived = component(
            ArtifactMemberRole::DerivedOverlay,
            "tokenizer-config",
            "tokenizer",
            "tokenizer_config.json",
        );
        derived.source_subpath = PathBuf::from("derived");
        let closure = ResolvedBundleClosure::new(vec![tokenizer, primary, derived]).unwrap();
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::SourceLibrary {
                root: primary_snapshot.clone(),
            },
            closure,
            provenance: ArtifactProvenance {
                identity,
                fixed_artifact_tier: Some("q8".to_owned()),
            },
            completeness: ArtifactCompleteness::Complete,
            availability: ArtifactAvailability::Available,
        };
        artifact.validate().unwrap();
        let locations = artifact
            .closure
            .source_file_locations(&artifact.identity, &primary_snapshot)
            .unwrap();
        assert_eq!(locations.len(), 3);
        assert!(locations.iter().any(|location| {
            location.source_path
                == std::fs::canonicalize(tokenizer_snapshot.join("tokenizer/tokenizer.json"))
                    .unwrap()
                && location.output_path == Path::new("tokenizer/tokenizer.json")
        }));
        assert!(locations.iter().any(|location| {
            location.source_path
                == std::fs::canonicalize(primary_snapshot.join("derived/tokenizer_config.json"))
                    .unwrap()
                && location.output_path == Path::new("tokenizer/tokenizer_config.json")
        }));

        std::fs::remove_file(tokenizer_snapshot.join("tokenizer/tokenizer.json")).unwrap();
        assert!(artifact.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn source_accepts_only_same_repository_blob_symlinks() {
        use std::os::unix::fs::symlink;

        let source = tempfile::tempdir().unwrap();
        let snapshot = snapshot(source.path(), "SceneWorks/model", REV);
        let repository = ArtifactSourceLibrary::new(source.path())
            .unwrap()
            .repository_root("SceneWorks/model")
            .unwrap();
        let blobs = repository.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("model-blob"), b"model").unwrap();
        symlink("../../blobs/model-blob", snapshot.join("model.safetensors")).unwrap();
        let closure = ResolvedBundleClosure::new(vec![primary("", "model.safetensors")]).unwrap();
        let identity = ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap();
        let artifact = ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: identity.clone(),
            location: ArtifactLocation::SourceLibrary {
                root: snapshot.clone(),
            },
            closure,
            provenance: ArtifactProvenance {
                identity,
                fixed_artifact_tier: Some("q8".to_owned()),
            },
            completeness: ArtifactCompleteness::Complete,
            availability: ArtifactAvailability::Available,
        };
        artifact.validate().unwrap();

        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("model-blob"), b"model").unwrap();
        symlink(
            outside.path().join("model-blob"),
            snapshot.join("model.safetensors"),
        )
        .unwrap();
        assert!(artifact.validate().is_err());

        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        std::fs::remove_dir_all(&blobs).unwrap();
        symlink(outside.path(), &blobs).unwrap();
        symlink("../../blobs/model-blob", snapshot.join("model.safetensors")).unwrap();
        assert!(artifact.validate().is_err());
        assert_eq!(
            std::fs::read(outside.path().join("model-blob")).unwrap(),
            b"model"
        );
    }

    #[test]
    fn shared_registry_observes_multiple_leases_drop_and_success_separately() {
        let source = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        std::fs::write(local.path().join("model.safetensors"), b"model").unwrap();
        let registry = ActiveArtifactLeaseRegistry::default();
        let resolver = ModelArtifactResolver::with_lease_registry(
            ArtifactSourceLibrary::new(source.path()).unwrap(),
            registry.clone(),
        );
        let artifact = Arc::new(
            resolver
                .resolve_local(
                    ArtifactIdentity::pinned("SceneWorks/model", REV, "q8").unwrap(),
                    ResolvedBundleClosure::new(vec![primary("", "model.safetensors")]).unwrap(),
                    local.path().to_owned(),
                    ArtifactAvailability::Available,
                )
                .unwrap(),
        );
        let cache_key = artifact.cache_key().unwrap();

        let first = resolver.acquire_runtime_lease(&artifact).unwrap();
        let second = resolver.clone().acquire_runtime_lease(&artifact).unwrap();
        assert_eq!(registry.active_lease_count(&cache_key), 2);
        assert_eq!(registry.active_cache_keys(), vec![cache_key.clone()]);

        drop(first);
        assert_eq!(registry.active_lease_count(&cache_key), 1);
        let candidate = second.mark_success();
        assert_eq!(candidate.cache_key, cache_key);
        assert!(!registry.is_active(&cache_key));
        assert!(registry.active_cache_keys().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn rebase_rejects_missing_tail_beneath_intermediate_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join("escape")).unwrap();
        let closure =
            ResolvedBundleClosure::new(vec![primary("escape/missing", "weights.safetensors")])
                .unwrap();

        assert!(closure.rebased_paths(root.path()).is_err());
        assert!(confine_artifact_path(
            &root.path().join("escape/missing/weights.safetensors"),
            &[root.path().to_owned()]
        )
        .is_err());
    }
}
