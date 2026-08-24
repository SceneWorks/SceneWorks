//! Checkpoint derivatives, held in the epic-19703 resolved-artifact cache (epic 20398, sc-20635).
//!
//! Compiling an import plan does not make a checkpoint cheap to load. A loader may still need a
//! derived index over the plan's layers, the weights re-laid-out into the component layout the
//! family adapter expects, or the tensors repacked into one backend's native representation. Those
//! outputs are expensive, reproducible, and worthless once their input changes — which is exactly
//! the contract [`crate::model_artifacts::resolved_cache`] already implements. So they live THERE,
//! in the same store, under the same receipts, leases, retention, pinning and eviction. There is no
//! second cache (E9).
//!
//! Two things had to be added to the store rather than worked around:
//!
//! * [`ResolvedCacheStore::reserve_derived`] — the copy path requires a source-library second copy
//!   of every byte before it will reserve. A derivative's bytes do not exist until a producer
//!   makes them, so a derived reservation stages an entry whose bundle is still to be written, and
//!   [`ResolvedCacheReservation::publish_produced`] enriches and publishes it through the same
//!   atomic rename, the same full re-validation, and the same completion receipt.
//! * [`crate::model_artifacts::resolved_cache::ResolvedCacheProduction`] — retention only evicts a
//!   copied bundle while the source library still proves a complete second copy. A derivative has
//!   no second copy anywhere, so retention judges it by its own rule; every other guard (pins,
//!   leases, freshness, shape) is unchanged.
//!
//! **The source weights are never written.** A producer is handed the resolved plan's layer paths
//! to READ and a staging directory to WRITE, and it can create files nowhere else: every output is
//! declared up front, created inside the staging root, and anything undeclared refuses at
//! publication. Nothing in this module opens a linked file for writing or removes one (E6).
//!
//! # Keying
//!
//! `ArtifactIdentity` carries the whole key material AC3 asks for:
//!
//! | key component | where it lands |
//! | --- | --- |
//! | content | `revision` — digests every plan layer's source fingerprint |
//! | semantic plan | `revision` — digests the record's `semanticDigest` and the plan id/family |
//! | adapter / codec version | `revision` — digests the adapter and codec ids and versions |
//! | backend representation | `revision` and `variant` |
//! | derivative kind | `variant` and the primary member's `componentId` |
//!
//! Anything that changes any of those changes the cache key, so a derivative produced for other
//! inputs can never be served for these ones.
//!
//! Nothing LOCATIONAL is in there. The same checkpoint reached through two approved libraries, or
//! through one library that was relinked to a new path, keys identically and shares one entry
//! (E1) — which is the whole point of a content-addressed cache. Which checkpoint's producer
//! actually created a bundle is recorded beside the key, in
//! [`ResolvedCacheMetadata::derived_from`], purely so a "forget this checkpoint" action can reach
//! it.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint_import::SourceLocatorV1;
use crate::checkpoint_plan_store::{
    parse_linked_checkpoint_id, CheckpointPlanError, CheckpointPlanStore, ResolvedCheckpointV1,
    RootRemovalV1,
};
use crate::model_artifacts::resolved_cache::{
    DerivedArtifactPlan, ManualRemovalOutcome, ReservationOutcome, ResolvedCacheError,
    ResolvedCacheLease, ResolvedCacheMetadata, ResolvedCacheStore,
};
use crate::model_artifacts::{
    ArtifactFile, ArtifactIdentity, ArtifactMemberRole, ModelArtifactResolver, ResolvedBundleMember,
};

/// Bumping this invalidates every persisted derivative: it is folded into the derived revision, so
/// old entries key differently and are simply never looked up again (and age out of the cache).
pub const CHECKPOINT_DERIVATIVE_FORMAT_VERSION: u32 = 1;

/// The synthetic repository every checkpoint derivative is filed under. It is deliberately a
/// CONSTANT: the repository is cache-key material, so putting anything locational in it (a library
/// root id, a path) would make one checkpoint's derivative key differently depending on where the
/// user keeps their library. Real Hugging Face repositories cannot collide with it — the owner is
/// not a namespace this app ever downloads from, and this string is only ever produced here.
const DERIVATIVE_REPOSITORY: &str = "sceneworks-checkpoint-derivative/linked";

/// The most files one derivative may declare.
const MAX_DERIVATIVE_OUTPUTS: usize = 4096;

/// What a checkpoint derivative is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDerivativeKindV1 {
    /// A derived index over the plan's layers (tensor → file/offset), so a load does not re-read
    /// every container header.
    DerivedIndex,
    /// The plan's weights re-expressed in the canonical component layout the family adapter loads.
    NormalizedLayout,
    /// The plan's weights repacked into one backend's native packed representation.
    BackendRepack,
}

impl CheckpointDerivativeKindV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DerivedIndex => "derived-index",
            Self::NormalizedLayout => "normalized-layout",
            Self::BackendRepack => "backend-repack",
        }
    }
}

/// Everything a derivative is keyed on beyond the checkpoint's own content and semantic plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointDerivativeRequestV1 {
    pub kind: CheckpointDerivativeKindV1,
    /// The family adapter that produces this derivative, and the version of its output format.
    pub adapter_id: String,
    pub adapter_version: String,
    /// The quant codec set the derivative is expressed in, and the version of its packing.
    pub codec_id: String,
    pub codec_version: String,
    /// The backend representation the output is in (`mlx`, `candle_cuda`, `portable`, …).
    pub backend: String,
    /// Every file the producer will write, as portable bundle-relative paths. Declared up front so
    /// the cache key covers the shape of the output and publication can refuse a partial bundle.
    pub outputs: Vec<String>,
}

impl CheckpointDerivativeRequestV1 {
    fn validate(&self) -> Result<(), CheckpointDerivativeError> {
        let invalid = |reason: String| CheckpointDerivativeError::InvalidRequest { reason };
        for (label, token) in [
            ("adapterId", &self.adapter_id),
            ("adapterVersion", &self.adapter_version),
            ("codecId", &self.codec_id),
            ("codecVersion", &self.codec_version),
            ("backend", &self.backend),
        ] {
            validate_key_token(label, token)?;
        }
        if self.outputs.is_empty() {
            return Err(invalid(
                "a derivative must declare at least one output".to_owned(),
            ));
        }
        if self.outputs.len() > MAX_DERIVATIVE_OUTPUTS {
            return Err(invalid(format!(
                "a derivative may declare at most {MAX_DERIVATIVE_OUTPUTS} outputs"
            )));
        }
        let mut seen = BTreeSet::new();
        for output in &self.outputs {
            validate_output_path(output)?;
            if !seen.insert(output.as_str()) {
                return Err(invalid(format!("output {output:?} is declared twice")));
            }
        }
        Ok(())
    }

    /// `variant` carries the derivative kind and its backend representation, so two backends'
    /// repacks of one checkpoint are two entries rather than one that silently serves both.
    fn variant(&self) -> String {
        format!("{}-{}", self.kind.as_str(), self.backend)
    }
}

/// One key-material token: nonempty, bounded, and free of anything that could make it ambiguous
/// once it is concatenated into the digest preimage or a path component.
fn validate_key_token(label: &str, token: &str) -> Result<(), CheckpointDerivativeError> {
    let valid = !token.is_empty()
        && token.len() <= 64
        && token.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        });
    if valid {
        Ok(())
    } else {
        Err(CheckpointDerivativeError::InvalidRequest {
            reason: format!("{label} {token:?} must be 1..=64 characters of [A-Za-z0-9._-]"),
        })
    }
}

/// A declared output path: portable, relative, confined, and never a link-bearing component.
fn validate_output_path(output: &str) -> Result<PathBuf, CheckpointDerivativeError> {
    let invalid = |reason: &str| CheckpointDerivativeError::InvalidRequest {
        reason: format!("derivative output {output:?} is invalid: {reason}"),
    };
    if output.trim().is_empty() {
        return Err(invalid("empty"));
    }
    if output.contains('\\') {
        return Err(invalid("must use '/' separators"));
    }
    let mut path = PathBuf::new();
    for component in Path::new(output).components() {
        match component {
            Component::Normal(part) => path.push(part),
            Component::CurDir => return Err(invalid("contains '.'")),
            Component::ParentDir => return Err(invalid("contains '..'")),
            Component::RootDir | Component::Prefix(_) => return Err(invalid("must be relative")),
        }
    }
    if path.as_os_str().is_empty() {
        return Err(invalid("empty"));
    }
    Ok(path)
}

/// What `ensure` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointDerivativeOutcomeV1 {
    /// The derivative was already published, verified and reusable. The producer never ran.
    AlreadyPresent(Box<ResolvedCacheMetadata>),
    /// The producer ran and its bundle was published.
    Produced(Box<ResolvedCacheMetadata>),
    /// Another process holds this entry (producing it, leasing it, or removing it). The caller
    /// should proceed without the derivative rather than wait.
    Contended,
}

/// Every way producing or reading a derivative refuses.
#[derive(Debug)]
pub enum CheckpointDerivativeError {
    /// The checkpoint itself did not resolve; no derivative work was started.
    Plan(CheckpointPlanError),
    /// The resolved-artifact cache refused.
    Cache(ResolvedCacheError),
    /// The request cannot be keyed.
    InvalidRequest { reason: String },
    /// The producer failed. Its staging directory was discarded, so nothing partial was published.
    Producer {
        checkpoint_id: String,
        message: String,
    },
}

impl fmt::Display for CheckpointDerivativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(f, "{error}"),
            Self::Cache(error) => write!(f, "[checkpoint-derivative:cache] {error}"),
            Self::InvalidRequest { reason } => {
                write!(f, "[checkpoint-derivative:invalid-request] {reason}")
            }
            Self::Producer {
                checkpoint_id,
                message,
            } => write!(
                f,
                "[checkpoint-derivative:producer-failed] producing a derivative of checkpoint \
                 {checkpoint_id:?} failed: {message}"
            ),
        }
    }
}

impl std::error::Error for CheckpointDerivativeError {}

impl From<CheckpointPlanError> for CheckpointDerivativeError {
    fn from(error: CheckpointPlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<ResolvedCacheError> for CheckpointDerivativeError {
    fn from(error: ResolvedCacheError) -> Self {
        Self::Cache(error)
    }
}

/// The only writing surface a producer gets.
///
/// It holds the resolved checkpoint (read-only: the source paths a producer reads from) and the
/// reservation's staging directory (the ONLY place it can write). Every create is confined to the
/// staging root and must name a path the request declared, so a producer cannot write beside its
/// declared outputs and cannot reach a linked source file at all.
pub struct CheckpointDerivativeOutputs<'a> {
    resolved: &'a ResolvedCheckpointV1,
    staging: &'a Path,
    declared: BTreeSet<PathBuf>,
}

impl<'a> CheckpointDerivativeOutputs<'a> {
    /// The verified checkpoint this derivative is being produced from. Its layer paths are the
    /// files a producer READS; they are never opened for writing here.
    pub fn resolved(&self) -> &'a ResolvedCheckpointV1 {
        self.resolved
    }

    /// Create one declared output file inside the staging directory.
    pub fn create(&self, output: &str) -> Result<fs::File, CheckpointDerivativeError> {
        let relative = validate_output_path(output)?;
        if !self.declared.contains(&relative) {
            return Err(CheckpointDerivativeError::InvalidRequest {
                reason: format!("output {output:?} was not declared by the derivative request"),
            });
        }
        let path = self.staging.join(&relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| CheckpointDerivativeError::Producer {
                checkpoint_id: self.resolved.checkpoint_id.clone(),
                message: format!("create {}: {error}", parent.display()),
            })?;
        }
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| CheckpointDerivativeError::Producer {
                checkpoint_id: self.resolved.checkpoint_id.clone(),
                message: format!("create {}: {error}", path.display()),
            })
    }
}

/// Checkpoint derivatives over the shared resolved-artifact cache.
#[derive(Clone, Debug)]
pub struct CheckpointDerivativeStore {
    plans: CheckpointPlanStore,
    cache: ResolvedCacheStore,
    data_dir: PathBuf,
}

impl CheckpointDerivativeStore {
    pub fn open(data_dir: &Path) -> Result<Self, CheckpointDerivativeError> {
        Ok(Self {
            plans: CheckpointPlanStore::open(data_dir),
            cache: ResolvedCacheStore::open(data_dir)?,
            data_dir: data_dir.to_path_buf(),
        })
    }

    pub fn plans(&self) -> &CheckpointPlanStore {
        &self.plans
    }

    pub fn cache(&self) -> &ResolvedCacheStore {
        &self.cache
    }

    /// The identity a derivative of `resolved` occupies.
    pub fn identity(
        &self,
        resolved: &ResolvedCheckpointV1,
        request: &CheckpointDerivativeRequestV1,
    ) -> Result<ArtifactIdentity, CheckpointDerivativeError> {
        request.validate()?;
        let revision = derivative_revision(resolved, request);
        ArtifactIdentity::pinned(DERIVATIVE_REPOSITORY, revision, request.variant()).map_err(
            |error| CheckpointDerivativeError::InvalidRequest {
                reason: error.to_string(),
            },
        )
    }

    /// The cache plan (identity + declared closure) for a derivative. Computable before any bytes
    /// exist, which is what makes the cache lookup meaningful.
    pub fn plan_for(
        &self,
        resolved: &ResolvedCheckpointV1,
        request: &CheckpointDerivativeRequestV1,
    ) -> Result<DerivedArtifactPlan, CheckpointDerivativeError> {
        let identity = self.identity(resolved, request)?;
        let mut files = Vec::with_capacity(request.outputs.len());
        for output in &request.outputs {
            let relative = validate_output_path(output)?;
            files.push(ArtifactFile::new(relative).map_err(|error| {
                CheckpointDerivativeError::InvalidRequest {
                    reason: error.to_string(),
                }
            })?);
        }
        let member = ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            // The derivative kind, again: `component_id` is key material, and naming the kind
            // here rather than the checkpoint keeps the key content-addressed.
            component_id: Some(request.kind.as_str().to_owned()),
            source: identity.clone(),
            tier: None,
            source_subpath: PathBuf::new(),
            destination: PathBuf::new(),
            files,
        };
        Ok(DerivedArtifactPlan::new(identity, vec![member])?)
    }

    pub fn cache_key(
        &self,
        resolved: &ResolvedCheckpointV1,
        request: &CheckpointDerivativeRequestV1,
    ) -> Result<String, CheckpointDerivativeError> {
        Ok(self.plan_for(resolved, request)?.cache_key()?)
    }

    /// The published derivative for these inputs, if the cache holds a complete one.
    pub fn lookup(
        &self,
        resolved: &ResolvedCheckpointV1,
        request: &CheckpointDerivativeRequestV1,
    ) -> Result<Option<ResolvedCacheMetadata>, CheckpointDerivativeError> {
        let cache_key = self.cache_key(resolved, request)?;
        Ok(self.cache.lookup_complete(&cache_key)?)
    }

    /// Lease a published derivative for the life of a load.
    ///
    /// The lease is the store's own: it re-hashes the bundle under the entry's locks before
    /// handing it over, and it holds the shared artifact lock, which is exactly what makes the
    /// entry unevictable while it is in use.
    pub fn acquire(
        &self,
        resolved: &ResolvedCheckpointV1,
        request: &CheckpointDerivativeRequestV1,
    ) -> Result<Option<ResolvedCacheLease>, CheckpointDerivativeError> {
        let cache_key = self.cache_key(resolved, request)?;
        let resolver =
            ModelArtifactResolver::new(crate::hf_home::model_source_library(&self.data_dir));
        Ok(self.cache.acquire_complete(
            &cache_key,
            &resolver,
            &derivative_owner(&resolved.checkpoint_id),
        )?)
    }

    /// Produce the derivative if the cache does not already hold it.
    ///
    /// The checkpoint is resolved FIRST — record ↔ plan agreement, approved root present, every
    /// layer's bytes still the bytes the plan was compiled from. A drifted or missing source
    /// refuses here, before a key is computed, so a derivative is never produced from, or served
    /// for, content the plan no longer describes.
    pub fn ensure(
        &self,
        checkpoint_id: &str,
        request: &CheckpointDerivativeRequestV1,
        produce: impl FnOnce(&CheckpointDerivativeOutputs<'_>) -> Result<(), String>,
    ) -> Result<CheckpointDerivativeOutcomeV1, CheckpointDerivativeError> {
        let resolved = self.plans.resolve(checkpoint_id)?;
        let plan = self.plan_for(&resolved, request)?;
        let cache_key = plan.cache_key()?;
        if let Some(metadata) = self.cache.lookup_complete(&cache_key)? {
            return Ok(CheckpointDerivativeOutcomeV1::AlreadyPresent(Box::new(
                metadata,
            )));
        }
        // The input this derivative is derived from. Recording it is what lets source-lifecycle
        // reconciliation reach the derivative when the library goes away.
        let (root_id, _) = parse_linked_checkpoint_id(checkpoint_id).ok_or_else(|| {
            CheckpointDerivativeError::InvalidRequest {
                reason: format!("checkpoint {checkpoint_id:?} is not a linked checkpoint"),
            }
        })?;
        let root_path = self.plans.resolve_root(root_id)?;
        let owner = derivative_owner(checkpoint_id);
        let reservation =
            match self
                .cache
                .reserve_derived(&plan, &root_path, checkpoint_id, &owner)?
            {
                ReservationOutcome::AlreadyComplete(metadata) => {
                    return Ok(CheckpointDerivativeOutcomeV1::AlreadyPresent(metadata));
                }
                ReservationOutcome::Contended => {
                    return Ok(CheckpointDerivativeOutcomeV1::Contended);
                }
                ReservationOutcome::Acquired(reservation) => *reservation,
            };
        reservation.prepare_for_materialization()?;
        let outputs = CheckpointDerivativeOutputs {
            resolved: &resolved,
            staging: reservation.staging_path(),
            declared: request
                .outputs
                .iter()
                .map(|output| validate_output_path(output))
                .collect::<Result<BTreeSet<_>, _>>()?,
        };
        if let Err(message) = produce(&outputs) {
            // Nothing partial survives: the staging tree is removed and the entry is recorded
            // interrupted, so the next attempt starts empty and no reader can ever see it.
            reservation.discard()?;
            return Err(CheckpointDerivativeError::Producer {
                checkpoint_id: checkpoint_id.to_owned(),
                message,
            });
        }
        drop(outputs);
        match reservation.publish_produced(&plan) {
            Ok(metadata) => Ok(CheckpointDerivativeOutcomeV1::Produced(Box::new(metadata))),
            Err(error) => Err(CheckpointDerivativeError::Cache(error)),
        }
    }

    /// Remove every derivative produced for one checkpoint.
    ///
    /// Matches on the entry's recorded `derivedFrom`, not on its key: the key is content-addressed
    /// on purpose. Refuses an entry that is leased, pinned or being produced — the shared store's
    /// own removal rule, unchanged — and never touches the linked source.
    pub fn remove_derivatives_for_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<Vec<ManualRemovalOutcome>, CheckpointDerivativeError> {
        self.remove_matching(|metadata| metadata.derived_from.as_deref() == Some(checkpoint_id))
    }

    /// Remove every derivative produced from one approved library root.
    pub fn remove_derivatives_for_root(
        &self,
        root_id: &str,
    ) -> Result<Vec<ManualRemovalOutcome>, CheckpointDerivativeError> {
        let prefix = format!("linked/{root_id}/");
        self.remove_matching(|metadata| {
            metadata
                .derived_from
                .as_deref()
                .is_some_and(|checkpoint_id| checkpoint_id.starts_with(&prefix))
        })
    }

    fn remove_matching(
        &self,
        matches: impl Fn(&ResolvedCacheMetadata) -> bool,
    ) -> Result<Vec<ManualRemovalOutcome>, CheckpointDerivativeError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default();
        let mut removed = Vec::new();
        for summary in self.cache.enumerate()? {
            let Some(metadata) = summary.metadata else {
                continue;
            };
            if !metadata.production.is_derived() || !matches(&metadata) {
                continue;
            }
            removed.push(self.cache.remove_entry(&metadata.cache_key, now)?);
        }
        Ok(removed)
    }

    /// Forget one linked checkpoint: its record, plan and bindings, plus every derivative of it.
    ///
    /// This is the whole of what removal owns (E6) — SceneWorks-owned plan documents and
    /// SceneWorks-owned derivatives. The linked source is not touched.
    pub fn invalidate_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<(bool, Vec<ManualRemovalOutcome>), CheckpointDerivativeError> {
        // Derivatives first: if removing one refuses because it is leased, the plan must still be
        // there for the lease holder, so nothing is half-removed.
        let derivatives = self.remove_derivatives_for_checkpoint(checkpoint_id)?;
        let removed = self.plans.invalidate(checkpoint_id)?;
        Ok((removed, derivatives))
    }

    /// Forget an approved root: its plans and every derivative produced from it.
    pub fn remove_root(
        &self,
        root_id: &str,
    ) -> Result<(RootRemovalV1, Vec<ManualRemovalOutcome>), CheckpointDerivativeError> {
        let derivatives = self.remove_derivatives_for_root(root_id)?;
        let removal = self.plans.remove_root(root_id)?;
        Ok((removal, derivatives))
    }
}

/// The logical model owner recorded on the cache's session records and pins for a derivative.
fn derivative_owner(checkpoint_id: &str) -> String {
    format!("checkpoint-derivative:{checkpoint_id}")
}

/// The 40-hex "revision" a derivative is pinned to: a digest over its content, its semantic plan,
/// its adapter and codec versions, and its backend representation.
///
/// It is domain-separated and length-prefixed so no two different inputs can produce the same
/// preimage by shifting a delimiter, and it is derived from the plan's per-layer FINGERPRINTS
/// rather than the file paths, so the same bytes reached through a relinked root key identically
/// (E1: locator independence) while different bytes never do.
fn derivative_revision(
    resolved: &ResolvedCheckpointV1,
    request: &CheckpointDerivativeRequestV1,
) -> String {
    let mut hasher = Sha256::new();
    let mut field = |value: &[u8]| {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    };
    field(b"sceneworks.checkpoint.derivative.v1");
    field(&CHECKPOINT_DERIVATIVE_FORMAT_VERSION.to_le_bytes());
    // Semantic plan. The plan's SEMANTIC digest, deliberately, not its `plan_id`: the plan id is
    // derived from the checkpoint's locator, so folding it in would key one checkpoint's
    // derivative differently in two libraries holding the same bytes.
    field(resolved.record.plan.semantic_digest.as_bytes());
    field(resolved.plan.family.as_bytes());
    // Content: every layer's role and the exact bytes the plan was compiled from.
    field(&(resolved.plan.layers.len() as u64).to_le_bytes());
    for layer in &resolved.plan.layers {
        field(layer.layer_id.as_bytes());
        field(layer.role.as_bytes());
        match &layer.source {
            SourceLocatorV1::Linked { fingerprint, .. } => field(fingerprint.as_bytes()),
            SourceLocatorV1::Managed { sha256, .. } => field(sha256.as_bytes()),
        }
    }
    // Adapter, codec and backend representation.
    field(request.kind.as_str().as_bytes());
    field(request.adapter_id.as_bytes());
    field(request.adapter_version.as_bytes());
    field(request.codec_id.as_bytes());
    field(request.codec_version.as_bytes());
    field(request.backend.as_bytes());
    // The declared output shape, in a canonical order.
    let mut outputs: Vec<&str> = request.outputs.iter().map(String::as_str).collect();
    outputs.sort_unstable();
    field(&(outputs.len() as u64).to_le_bytes());
    for output in outputs {
        field(output.as_bytes());
    }
    format!("{:x}", hasher.finalize())[..40].to_owned()
}
