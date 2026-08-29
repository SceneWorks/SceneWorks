//! Durable app-owned resolved-model cache policy and store.
//!
//! The store is deliberately separate from the authoritative source library. It owns only
//! `<data_dir>/models/resolved`, refuses to adopt an unmarked directory, and names entries from
//! the portable immutable cache key defined by [`crate::model_artifacts`].

#[path = "resolved_cache/materialization.rs"]
mod materialization;
pub use materialization::{
    IdlePromotionOutcome, MaterializationCancellation, MaterializationOutcome,
    PromotionScheduleOutcome, ResolvedCacheMaterializer, ResolvedCachePromotionScheduler,
};

#[path = "resolved_cache/retention.rs"]
mod retention;
pub use retention::{
    EvictedRecord, EvictionCause, ManualRemovalOutcome, ManualRemovalPins, ManualRemovalPreview,
    ReconciliationReport, ResolvedCacheRetention, RetainedRecord, RetentionCheckpointOutcome,
    RetentionHold, RetentionReport, SourceLifecycleSelector,
};

use crate::model_artifacts::{
    ActiveArtifactLease, ArtifactAvailability, ArtifactCompleteness, ArtifactIdentity,
    ArtifactLocation, ArtifactProvenance, ClosureFileStat, ModelArtifactResolver,
    PromotionCandidate, ResolvedBundleClosure, ResolvedBundleMember, ResolvedModelArtifact,
    MODEL_ARTIFACT_CONTRACT_VERSION,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub const RESOLVED_CACHE_POLICY_VERSION: u32 = 1;
pub const RESOLVED_CACHE_DEFAULT_MAX_BYTES: u64 = 68_719_476_736;
pub const RESOLVED_CACHE_DEFAULT_INACTIVITY_SECONDS: u64 = 1_209_600;
pub const RESOLVED_CACHE_ENABLED_ENV: &str = "SCENEWORKS_RESOLVED_CACHE_ENABLED";
pub const RESOLVED_CACHE_MAX_BYTES_ENV: &str = "SCENEWORKS_RESOLVED_CACHE_MAX_BYTES";
pub const RESOLVED_CACHE_INACTIVITY_SECONDS_ENV: &str =
    "SCENEWORKS_RESOLVED_CACHE_INACTIVITY_SECONDS";
pub const RESOLVED_CACHE_STORE_VERSION: u32 = 1;
/// The journal version a DERIVED entry is written at (sc-20635).
///
/// `production` and `derivedFrom` widen the metadata document, and `ResolvedCacheMetadata` is
/// `deny_unknown_fields`, so a binary that predates them cannot read a derived journal. Declaring
/// the higher version on exactly the documents that carry the new fields is what makes that a
/// version statement rather than an unexplained decode failure: a reader that knows about versions
/// but not about these fields refuses with [`UNSUPPORTED_JOURNAL_VERSION`] naming the version it
/// found.
///
/// A SOURCE-COPY entry keeps writing [`RESOLVED_CACHE_STORE_VERSION`] and, because both new fields
/// are `skip_serializing_if`-defaulted, its document is byte-for-byte what it was before this
/// story. That is what makes the bump safe to land warm: an existing cache full of v1 source
/// copies is still read, still leased and still evicted by this binary.
pub const RESOLVED_CACHE_DERIVED_STORE_VERSION: u32 = 2;
/// The typed refusal a journal from a NEWER writer gets, instead of "decode cache metadata: …".
const UNSUPPORTED_JOURNAL_VERSION: &str = "cache metadata was written by a newer SceneWorks";

const STORE_MARKER: &str = ".sceneworks-resolved-cache-v1";
const STORE_MARKER_BODY: &[u8] = b"sceneworks-resolved-cache\nschema=1\n";
const EVICTED_MARKER_FILE: &str = "evicted.marker.json";
const AUDIT_DIR: &str = "audit";
/// The one journal-read failure that means "this entry holds no recoverable state", as opposed to
/// a transient or environmental read failure. Manual removal is allowed to clear it; every other
/// read error must refuse, because it cannot rule out a pin or an in-flight materialization.
const BOTH_METADATA_SLOTS_CORRUPT: &str = "both cache metadata slots are corrupt";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ResolvedCachePolicy {
    pub enabled: bool,
    pub max_bytes: u64,
    pub inactivity_seconds: u64,
}

impl Default for ResolvedCachePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: RESOLVED_CACHE_DEFAULT_MAX_BYTES,
            inactivity_seconds: RESOLVED_CACHE_DEFAULT_INACTIVITY_SECONDS,
        }
    }
}

impl ResolvedCachePolicy {
    pub fn validate(&self) -> Result<(), ResolvedCacheError> {
        if self.max_bytes == 0 {
            return Err(ResolvedCacheError::new(
                "resolved cache maxBytes must be finite and greater than zero",
            ));
        }
        if self.inactivity_seconds == 0 {
            return Err(ResolvedCacheError::new(
                "resolved cache inactivitySeconds must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn env_pairs(&self) -> Result<[(&'static str, String); 3], ResolvedCacheError> {
        self.validate()?;
        Ok([
            (
                RESOLVED_CACHE_ENABLED_ENV,
                if self.enabled { "true" } else { "false" }.to_owned(),
            ),
            (RESOLVED_CACHE_MAX_BYTES_ENV, self.max_bytes.to_string()),
            (
                RESOLVED_CACHE_INACTIVITY_SECONDS_ENV,
                self.inactivity_seconds.to_string(),
            ),
        ])
    }

    pub fn from_env() -> Result<Self, ResolvedCacheError> {
        Self::from_env_values(|name| std::env::var(name).ok())
    }

    pub fn from_env_or_safe_default() -> Self {
        match Self::from_env() {
            Ok(policy) => policy,
            Err(error) => {
                tracing::warn!(error = %error, "invalid resolved-cache environment; disabling cache");
                Self::default()
            }
        }
    }

    pub fn from_env_values(
        mut value: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self, ResolvedCacheError> {
        let defaults = Self::default();
        let enabled = match value(RESOLVED_CACHE_ENABLED_ENV) {
            None => defaults.enabled,
            Some(raw) => parse_bool(&raw).ok_or_else(|| {
                ResolvedCacheError::new(format!(
                    "{RESOLVED_CACHE_ENABLED_ENV} must be true/false or 1/0"
                ))
            })?,
        };
        let max_bytes = parse_optional_u64(
            value(RESOLVED_CACHE_MAX_BYTES_ENV),
            RESOLVED_CACHE_MAX_BYTES_ENV,
            defaults.max_bytes,
        )?;
        let inactivity_seconds = parse_optional_u64(
            value(RESOLVED_CACHE_INACTIVITY_SECONDS_ENV),
            RESOLVED_CACHE_INACTIVITY_SECONDS_ENV,
            defaults.inactivity_seconds,
        )?;
        let policy = Self {
            enabled,
            max_bytes,
            inactivity_seconds,
        };
        policy.validate()?;
        Ok(policy)
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" => Some(true),
        "0" | "false" => Some(false),
        _ => None,
    }
}

fn parse_optional_u64(
    value: Option<String>,
    name: &str,
    default: u64,
) -> Result<u64, ResolvedCacheError> {
    match value {
        None => Ok(default),
        Some(value) => value
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| ResolvedCacheError::new(format!("{name} must be a positive integer"))),
    }
}

/// Whether a healthy store REFUSED a request, or a broken one could not answer it. Carried on the
/// error itself rather than recovered by matching on its message, so an HTTP surface can answer
/// 4xx and 5xx correctly without a second, drifting classification built out of string comparisons.
///
/// The line is drawn at "is anything actually wrong with the machine":
///
/// * [`Request`](Self::Request) — the store is intact and is declining THIS request: a malformed
///   cache key, an entry it does not hold, or an entry whose own sanctioned state (pinned, leased,
///   being materialized, being evicted) forbids the operation. Nothing is broken, and the reply
///   tells the caller what to change.
/// * [`Internal`](Self::Internal) — the store could not answer at all: IO faults, lock-file
///   failures, clock failures, damaged journals and unreadable tombstones. Reporting these as
///   client errors is what sends an operator to fix their request while the real fault sits on the
///   host, and it hides genuine damage from anything watching 5xx rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedCacheErrorKind {
    /// An intact store declining this particular request.
    Request,
    /// A store or host that could not answer. Not caused by, and not fixable from, the request.
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCacheError {
    message: String,
    kind: ResolvedCacheErrorKind,
}

impl ResolvedCacheError {
    /// The default construction. Unattributed failures are [`ResolvedCacheErrorKind::Internal`] on
    /// purpose: a new failure path that nobody classified must not silently start reporting host
    /// faults as the caller's mistake.
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ResolvedCacheErrorKind::Internal,
        }
    }

    /// An intact store declining this request. Used only where nothing is wrong with the store or
    /// the host — a malformed key, an entry that is not held, or an entry state that forbids the
    /// operation.
    fn request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ResolvedCacheErrorKind::Request,
        }
    }

    pub fn kind(&self) -> ResolvedCacheErrorKind {
        self.kind
    }

    /// True only for the journal read failure that proves the entry retains no recoverable
    /// state — never for a transient, environmental, or fail-closed refusal.
    pub fn is_unrecoverable_metadata(&self) -> bool {
        self.message == BOTH_METADATA_SLOTS_CORRUPT
    }
}

impl std::fmt::Display for ResolvedCacheError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ResolvedCacheError {}

impl From<std::io::Error> for ResolvedCacheError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedCacheEntryState {
    Pending,
    Materializing,
    Interrupted,
    Complete,
    Corrupt,
    /// Summary-only state for an entry with a durable eviction tombstone whose removal has not
    /// finished yet. Never persisted into the metadata journal; `validate_metadata_shape_with` rejects
    /// it there.
    Evicting,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    Clean,
    RecoveredFromOlderSlot,
    ReconstructedFromCompleteReceipt,
    InterruptedReservation,
    CorruptUnrecoverable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceVolumeRelation {
    Same,
    Different,
    Unavailable,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceVolumeObservation {
    pub relation: SourceVolumeRelation,
    pub source_identity: Option<u64>,
    pub resolved_identity: Option<u64>,
    pub observed_at: u64,
}

/// How an entry's bundle came to exist, and therefore what retention may assume about it.
///
/// The distinction is load-bearing exactly once, in [`retention`]'s eviction proof. A
/// [`Self::SourceCopy`] entry is only evictable while the source library still holds a verified
/// second copy of every byte — deleting it would otherwise destroy the user's only copy. A
/// [`Self::Derived`] entry has no second copy anywhere by construction: it was COMPUTED from an
/// input the cache does not hold, so demanding a second copy would make it permanently unevictable
/// and the cache would grow without bound. Its safety property is the other one — it is
/// reproducible from its input, and worthless without it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedCacheProduction {
    /// Copied from a complete second copy in the configured source library.
    #[default]
    SourceCopy,
    /// Computed by a producer (epic 20398 checkpoint derivatives: derived indexes, normalised
    /// layouts, backend repacks).
    Derived,
}

impl ResolvedCacheProduction {
    /// Journals written before sc-20635 carry no `production` field and are all source copies, so
    /// the default round-trips byte-for-byte and only derived entries widen the document.
    fn is_source_copy(&self) -> bool {
        *self == Self::SourceCopy
    }

    pub fn is_derived(&self) -> bool {
        *self == Self::Derived
    }

    /// The journal version a document with this production mode is written at, and the only
    /// version a reader accepts for it.
    fn store_version(&self) -> u32 {
        match self {
            Self::SourceCopy => RESOLVED_CACHE_STORE_VERSION,
            Self::Derived => RESOLVED_CACHE_DERIVED_STORE_VERSION,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolvedCacheMetadata {
    pub schema_version: u32,
    pub cache_key: String,
    #[serde(
        default,
        skip_serializing_if = "ResolvedCacheProduction::is_source_copy"
    )]
    pub production: ResolvedCacheProduction,
    /// For a derived entry, the logical input whose producer created this bundle — a checkpoint id
    /// for epic-20398 derivatives.
    ///
    /// Deliberately NOT part of the cache key: a derivative is content-addressed, so two
    /// checkpoints with byte-identical inputs share one entry and the second one is a cache hit
    /// rather than a second copy. This field is lifecycle bookkeeping only, so a "forget this
    /// checkpoint" action can reach the derivatives that were produced for it. Scoping a removal
    /// this way can cost a checkpoint that was sharing the entry a re-production; it can never
    /// lose data, because a derivative is reproducible from its input by definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
    pub artifact: ResolvedModelArtifact,
    pub source_configured_path: PathBuf,
    pub source_canonical_path: PathBuf,
    pub entry_relative_path: PathBuf,
    pub bundle_relative_path: PathBuf,
    pub state: ResolvedCacheEntryState,
    pub verified_bytes: u64,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_used_at: Option<u64>,
    pub artifact_pinned: bool,
    pub model_pin_owners: BTreeSet<String>,
    pub effective_pin: bool,
    pub reservation_id: Option<String>,
    pub reservation_owner: Option<String>,
    pub session_id: Option<String>,
    pub recovery_status: RecoveryStatus,
    pub source_volume: SourceVolumeObservation,
}

/// A recorded verdict from the load boundary: at `verified_at`, every file in the closure was read
/// and matched its recorded SHA-256, while the closure looked exactly like `files` on disk.
///
/// The verdict is only redeemable while that second half still holds. `files` is therefore not
/// bookkeeping — it is the entire precondition, and it is compared in full (membership included, so
/// an added or removed closure file invalidates the receipt just as a rewritten one does).
///
/// ## Why this is a sidecar and not a metadata field
///
/// [`ResolvedCacheMetadata`] is `deny_unknown_fields`, and the journal keeps only two slots. Adding
/// a field there would mean that after two acquisitions BOTH slots carry it — and an older build,
/// which is exactly the build a rollback runs, would then find zero readable slots and report
/// `BOTH_METADATA_SLOTS_CORRUPT` for every entry it had been serving happily. Corrupt is the one
/// classification manual removal is allowed to clear, so a downgrade would invite deleting (and
/// re-copying) an otherwise perfect cache: here, 139 GB of it.
///
/// A sidecar file keeps the journal byte-identical to what every existing build already writes, so
/// a downgrade sees a store it fully understands and simply ignores one extra file. It also fails
/// safe in every direction a sidecar can fail: unreadable, unparseable, forward-versioned, or
/// belonging to another entry all mean "no verdict", which is precisely the pre-receipt behaviour of
/// re-reading the bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ClosureContentReceipt {
    pub schema_version: u32,
    /// The entry this verdict was recorded for. The file already lives in a digest-keyed directory,
    /// so this is defence in depth against a receipt that was copied rather than written here.
    pub cache_key: String,
    pub verified_at: u64,
    pub files: Vec<ClosureFileStat>,
}

impl ResolvedCacheMetadata {
    fn refresh_effective_pin(&mut self) {
        self.effective_pin = self.artifact_pinned || !self.model_pin_owners.is_empty();
    }
}

/// One published entry the local tier cannot serve because of its SHAPE — the bundle is intact and
/// verified, but its layout is not one the shared snapshot resolvers can be handed. Kept distinct
/// from every other rejection class (torn, incomplete, unverifiable) because this is the only one
/// worth reporting: those are ordinary fail-closed fallbacks, this is a bundle that will never
/// serve until something changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalArtifactRejection {
    pub cache_key: String,
    pub repository: String,
    pub revision: String,
    pub reason: String,
}

/// The result of enumerating the local tier: what can serve, and what was rejected for an
/// unsupported shape. Rejections are returned rather than dropped so the caller that emits the
/// runtime's observability can name them — dropping them at the scan made the local-tier-unsupported
/// class unreachable for exactly the case it is named after.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalArtifactScan {
    pub artifacts: Vec<ResolvedModelArtifact>,
    pub rejections: Vec<LocalArtifactRejection>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCacheEntrySummary {
    pub cache_key: String,
    pub state: ResolvedCacheEntryState,
    pub metadata: Option<ResolvedCacheMetadata>,
}

#[derive(Debug)]
pub enum ReservationOutcome {
    Acquired(Box<ResolvedCacheReservation>),
    AlreadyComplete(Box<ResolvedCacheMetadata>),
    Contended,
}

/// A bundle the cache will hold that no source library holds a second copy of, described BEFORE
/// its bytes exist (epic 20398, sc-20635).
///
/// A [`PromotionCandidate`] carries a validated artifact because its bytes are already installed;
/// a derivative's are not, so the plan carries only what the cache key is computed from — identity,
/// closure and provenance — and the file digests are enriched from the staged bytes at publication,
/// exactly as the copy path enriches them after copying. The key is therefore known before any work
/// is done, which is what makes the cache lookup meaningful.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedArtifactPlan {
    identity: ArtifactIdentity,
    closure: ResolvedBundleClosure,
    provenance: ArtifactProvenance,
}

impl DerivedArtifactPlan {
    /// `members` is the full closure, with `size_bytes`/`sha256` left unset: the producer has not
    /// written the bytes yet, and those fields are deliberately excluded from the cache key.
    pub fn new(
        identity: ArtifactIdentity,
        members: Vec<ResolvedBundleMember>,
    ) -> Result<Self, ResolvedCacheError> {
        identity
            .validate()
            .map_err(|error| ResolvedCacheError::request(error.to_string()))?;
        let closure = ResolvedBundleClosure::new(members)
            .map_err(|error| ResolvedCacheError::request(error.to_string()))?;
        Ok(Self {
            provenance: ArtifactProvenance {
                identity: identity.clone(),
                fixed_artifact_tier: None,
            },
            identity,
            closure,
        })
    }

    pub fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    pub fn closure(&self) -> &ResolvedBundleClosure {
        &self.closure
    }

    /// The key this derivative will occupy. Computable with nothing on disk.
    pub fn cache_key(&self) -> Result<String, ResolvedCacheError> {
        self.artifact_at(Path::new("/"), ArtifactState::Pending)
            .cache_key()
            .map_err(|error| ResolvedCacheError::request(error.to_string()))
    }

    /// The artifact document for this plan rooted at `root`.
    ///
    /// While the producer is still running the bundle genuinely does not exist, so the pending form
    /// says so (`Incomplete`/`Missing`). Neither field is part of the cache key, so the pending and
    /// published forms occupy the same entry.
    fn artifact_at(&self, root: &Path, state: ArtifactState) -> ResolvedModelArtifact {
        let (completeness, availability) = match state {
            ArtifactState::Pending => (
                ArtifactCompleteness::Incomplete,
                ArtifactAvailability::Missing,
            ),
            ArtifactState::Published => (
                ArtifactCompleteness::Complete,
                ArtifactAvailability::Available,
            ),
        };
        ResolvedModelArtifact {
            schema_version: MODEL_ARTIFACT_CONTRACT_VERSION,
            identity: self.identity.clone(),
            location: ArtifactLocation::ResolvedLocal {
                root: root.to_path_buf(),
            },
            closure: self.closure.clone(),
            provenance: self.provenance.clone(),
            completeness,
            availability,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArtifactState {
    Pending,
    Published,
}

#[derive(Clone)]
pub struct ResolvedCacheStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    root: PathBuf,
    session_id: String,
    /// `None` only for the read-only inspection handle created by
    /// [`ResolvedCacheStore::enumerate_existing`]; every runtime session holds its lock.
    _session_lock: Option<File>,
}

impl std::fmt::Debug for ResolvedCacheStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedCacheStore")
            .field("root", &self.inner.root)
            .field("session_id", &self.inner.session_id)
            .finish()
    }
}

impl ResolvedCacheStore {
    pub fn open(data_dir: &Path) -> Result<Self, ResolvedCacheError> {
        let models = data_dir.join("models");
        std::fs::create_dir_all(&models).map_err(|error| {
            ResolvedCacheError::new(format!(
                "create model directory {}: {error}",
                models.display()
            ))
        })?;
        ensure_regular_directory(&models)?;
        let root = models.join("resolved");
        initialize_or_validate_root(&root)?;
        let root = std::fs::canonicalize(&root).map_err(|error| {
            ResolvedCacheError::new(format!("canonicalize resolved root: {error}"))
        })?;
        let session_id = random_id()?;
        let session_lock_path = root.join("sessions").join(format!("{session_id}.lock"));
        let session_lock = open_lock_file(&session_lock_path)?;
        FileExt::try_lock_exclusive(&session_lock).map_err(|error| {
            ResolvedCacheError::new(format!("lock cache session {session_id}: {error}"))
        })?;
        std::fs::create_dir(root.join("sessions").join(&session_id)).map_err(|error| {
            ResolvedCacheError::new(format!("create cache session records: {error}"))
        })?;
        Ok(Self {
            inner: Arc::new(StoreInner {
                root,
                session_id,
                _session_lock: Some(session_lock),
            }),
        })
    }

    /// Inspect an already-initialized cache without creating a runtime session or refreshing
    /// usage. Missing cache roots are an empty cache; unmanaged or invalid roots still fail
    /// closed. This is the catalog/preflight read path (sc-19708): discovery must never stamp
    /// usage or take a session slot.
    ///
    /// Read paths use [`EntryListing::Inspect`]: identity, shape, confinement, file presence and
    /// sizes are still proven, but bundle bytes are not re-hashed. Re-hashing every complete
    /// bundle here would put gigabytes of I/O behind a catalog GET (and behind any status poll),
    /// and it would prove nothing a read surface may act on — the runtime load path
    /// ([`Self::acquire_complete`]) still re-hashes before handing an artifact to a loader.
    pub fn enumerate_existing(
        data_dir: &Path,
    ) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        Self::open_for_inspection(data_dir)?
            .map_or_else(|| Ok(Vec::new()), |store| store.list(EntryListing::Inspect))
    }

    /// Where this cache lives under a data dir. One definition, so a reader that never opens the
    /// store cannot drift from one that does.
    fn store_root(data_dir: &Path) -> PathBuf {
        data_dir.join("models").join("resolved")
    }

    /// Read-only handle onto an already-initialized cache, or `None` when no cache root exists.
    /// Never creates the root, never takes a session slot, never writes.
    ///
    /// sc-19707 introduced the same handle independently as a private `open_read_only`; the two
    /// bodies were identical, so they are ONE function here. The public name is kept because the
    /// API's cache-status surface needs it from outside this crate, and every caller — the
    /// catalog/preflight read, the status read, and the local-tier scan below — wants exactly the
    /// same no-session, no-usage, no-write handle.
    pub fn open_for_inspection(data_dir: &Path) -> Result<Option<Self>, ResolvedCacheError> {
        let root = Self::store_root(data_dir);
        if !root.exists() {
            return Ok(None);
        }
        ensure_regular_directory(&root)?;
        let marker = std::fs::read(root.join(STORE_MARKER)).map_err(|_| {
            ResolvedCacheError::new("resolved cache root is missing its reserved marker")
        })?;
        if marker != STORE_MARKER_BODY {
            return Err(ResolvedCacheError::new(
                "resolved cache root has an invalid reserved marker",
            ));
        }
        let root = std::fs::canonicalize(root)?;
        Ok(Some(Self {
            inner: Arc::new(StoreInner {
                root,
                session_id: "read-only-inspection".to_owned(),
                _session_lock: None,
            }),
        }))
    }

    /// A cheap change key for the published local tier, safe to read on every request (sc-19712
    /// F-4).
    ///
    /// The **worker** publishes bundles and the **API** answers catalog reads from a cached
    /// snapshot, and nothing crossed that process boundary: a model promoted after running from
    /// the source library kept reading `installed_external_unavailable` until the snapshot
    /// happened to be rebuilt, so the Model Manager withheld — with a "reconnect the library"
    /// prompt — the very models a user had just made survive a disconnect.
    ///
    /// This is a `stat` per entry directory over the two files that mark the transitions the local
    /// tier can see: `complete.receipt.json`, written once by `record_complete` and by nothing
    /// else, and the eviction tombstone, which withdraws an entry before its directory is gone. It
    /// deliberately does NOT move when an entry's journal is rewritten to stamp usage, because a
    /// load changes no availability and re-deriving the catalog on every load would trade one
    /// stale answer for a permanent rebuild. Entries not present, unreadable, or missing a receipt
    /// simply contribute nothing.
    ///
    /// It is a change *detector*, not an ordering: compare it for equality only. A cache that does
    /// not exist yet answers 0, which is also what an unreadable root answers — the caller's
    /// fallback for both is the same, to keep whatever it had.
    pub fn published_generation(data_dir: &Path) -> u64 {
        let entries_root = Self::store_root(data_dir).join("entries");
        let Ok(entries) = std::fs::read_dir(&entries_root) else {
            return 0;
        };
        let mut observed = BTreeSet::new();
        for item in entries.flatten() {
            let Some(digest) = item
                .file_name()
                .to_str()
                .filter(|value| is_lower_hex_64(value))
                .map(str::to_owned)
            else {
                continue;
            };
            let entry = entries_root.join(&digest);
            let receipt = std::fs::metadata(entry.join("complete.receipt.json"))
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|since| (since.as_secs(), since.subsec_nanos()));
            let withdrawn = std::fs::symlink_metadata(entry.join(EVICTED_MARKER_FILE)).is_ok();
            observed.insert((digest, receipt, withdrawn));
        }
        if observed.is_empty() {
            return 0;
        }
        let mut hasher = Sha256::new();
        for (digest, receipt, withdrawn) in observed {
            hasher.update(digest.as_bytes());
            let (seconds, nanos) = receipt.unwrap_or((0, 0));
            hasher.update(seconds.to_le_bytes());
            hasher.update(nanos.to_le_bytes());
            hasher.update([u8::from(withdrawn)]);
        }
        let digest = hasher.finalize();
        let mut key = [0_u8; 8];
        key.copy_from_slice(&digest[..8]);
        // Never collide with the "nothing to report" answer, so a real cache always reads as a
        // change against an empty one.
        u64::from_le_bytes(key) | 1
    }

    /// Every published artifact this cache can actually serve a runtime load from (sc-19707).
    ///
    /// Validity is judged per entry and fails CLOSED to the source tier: an entry that is not
    /// `Complete`, whose journal or receipt does not verify, whose bundle no longer matches the
    /// recorded closure by path or size (a torn, truncated, or hand-edited bundle), or whose shape
    /// the local tier cannot serve is skipped rather than poisoning the whole answer. Read-only:
    /// never creates a session, never stamps usage, and never repairs.
    ///
    /// Verification here is [`ContentVerification::PathsAndSizesOnly`] — this is a *scan*, and its
    /// cost is paid per job submission and per catalog build. The content re-hash belongs to
    /// [`Self::acquire_complete`], which re-runs this same validation at full strength under the
    /// entry's locks before a lease is issued, so an entry this scan offers can still be refused
    /// at the load boundary. See sc-19712 F-3.
    ///
    /// Entries rejected for an UNSUPPORTED SHAPE are reported separately rather than dropped, so
    /// the guard can name them: they are the only rejection class a user can act on (the bundle is
    /// intact, but its layout is one the shared snapshot resolvers cannot be handed). Every other
    /// rejection is a torn/absent/unverifiable entry, which is a fallback rather than a report, and
    /// is logged at debug with the reason instead of being swallowed silently.
    pub fn valid_local_artifacts(data_dir: &Path) -> LocalArtifactScan {
        let store = match Self::open_for_inspection(data_dir) {
            Ok(Some(store)) => store,
            Ok(None) => return LocalArtifactScan::default(),
            Err(error) => {
                tracing::debug!(
                    data_dir = %data_dir.display(),
                    %error,
                    "resolved cache could not be opened for local-tier enumeration; every model \
                     stays on the source tier"
                );
                return LocalArtifactScan::default();
            }
        };
        let entries_root = store.inner.root.join("entries");
        let entries = match std::fs::read_dir(&entries_root) {
            Ok(entries) => entries,
            Err(error) => {
                // A cache that has never published anything has no `entries/` directory at all;
                // that is the ordinary empty case, not a fault worth reporting.
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::debug!(
                        entries_root = %entries_root.display(),
                        %error,
                        "resolved cache entries could not be enumerated for the local tier"
                    );
                }
                return LocalArtifactScan::default();
            }
        };
        let mut scan = LocalArtifactScan::default();
        for item in entries.flatten() {
            let Some(digest) = item
                .file_name()
                .to_str()
                .filter(|value| is_lower_hex_64(value))
                .map(str::to_owned)
            else {
                continue;
            };
            if !item.file_type().is_ok_and(|kind| kind.is_dir()) {
                continue;
            }
            let metadata_lock = match store.lock_metadata(&digest) {
                Ok(lock) => lock,
                Err(error) => {
                    tracing::debug!(%digest, %error, "resolved cache entry could not be locked for \
                         local-tier enumeration");
                    continue;
                }
            };
            let read = store.read_metadata_unlocked(&digest);
            drop(metadata_lock);
            let metadata = match read {
                Ok(JournalRead::Valid { metadata, .. }) => metadata,
                Ok(other) => {
                    tracing::debug!(
                        %digest,
                        journal = ?std::mem::discriminant(&other),
                        "resolved cache entry journal is not valid; the model stays on the source \
                         tier"
                    );
                    continue;
                }
                Err(error) => {
                    tracing::debug!(%digest, %error, "resolved cache entry journal could not be read");
                    continue;
                }
            };
            if metadata.state != ResolvedCacheEntryState::Complete {
                tracing::debug!(
                    %digest,
                    state = ?metadata.state,
                    "resolved cache entry is not complete; the model stays on the source tier"
                );
                continue;
            }
            // Paths and sizes only, NOT a content re-hash (sc-19712 F-3). This scan is a read:
            // it answers "which entries could serve a load", for the API's per-submission
            // preflight and catalog build and for the worker's pre-loader guard. Re-hashing here
            // made that answer cost the whole cache — measured at 929.6 s for one 5.57 GB bundle
            // on a single job submission, against a default budget of 64 GiB — so populating the
            // cache made every submission slower than the load the cache exists to save.
            //
            // What makes the cheap scan safe is that it is not the last word: `acquire_complete`
            // re-verifies at full strength under the entry's locks immediately before the lease
            // that hands bytes to a runtime, and every write validates at full strength. A scan
            // that offers an entry whose bytes were altered after publication therefore cannot
            // load it; the lease refuses and the caller falls back to the source tier.
            if let Err(error) = validate_complete_metadata_inner(
                &store,
                &metadata,
                ContentVerification::PathsAndSizesOnly,
            ) {
                tracing::debug!(
                    %digest,
                    %error,
                    "resolved cache entry did not re-verify; the model stays on the source tier"
                );
                continue;
            }
            // A published bundle that is not stored in the source-library layout cannot be handed
            // to the shared snapshot resolvers, so it is not a local-tier candidate at all. This
            // is REPORTED rather than dropped: the guard emits it as the local-tier-unsupported
            // class, which is otherwise unreachable for the very case it names.
            if let Err(error) =
                crate::model_artifacts::local_preference::overlay_entries_for_artifact(
                    &metadata.artifact,
                )
            {
                scan.rejections.push(LocalArtifactRejection {
                    cache_key: metadata.cache_key.clone(),
                    repository: metadata.artifact.identity.repository.clone(),
                    revision: metadata.artifact.identity.revision.clone(),
                    reason: error.to_string(),
                });
                continue;
            }
            scan.artifacts.push(metadata.artifact);
        }
        scan
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    pub fn entry_path(&self, cache_key: &str) -> Result<PathBuf, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        Ok(self.inner.root.join("entries").join(digest))
    }

    pub fn bundle_path(&self, cache_key: &str) -> Result<PathBuf, ResolvedCacheError> {
        Ok(self.entry_path(cache_key)?.join("bundle"))
    }

    pub fn compare_source_volume(
        &self,
        source_root: &Path,
    ) -> Result<SourceVolumeObservation, ResolvedCacheError> {
        let observed_at = now_seconds()?;
        if !source_root.exists() {
            return Ok(SourceVolumeObservation {
                relation: SourceVolumeRelation::Unavailable,
                source_identity: None,
                resolved_identity: volume_identity(&self.inner.root).ok(),
                observed_at,
            });
        }
        let source = match volume_identity(source_root) {
            Ok(identity) => identity,
            Err(_) => {
                return Ok(SourceVolumeObservation {
                    relation: SourceVolumeRelation::Unknown,
                    source_identity: None,
                    resolved_identity: volume_identity(&self.inner.root).ok(),
                    observed_at,
                });
            }
        };
        let resolved = match volume_identity(&self.inner.root) {
            Ok(identity) => identity,
            Err(_) => {
                return Ok(SourceVolumeObservation {
                    relation: SourceVolumeRelation::Unknown,
                    source_identity: Some(source),
                    resolved_identity: None,
                    observed_at,
                });
            }
        };
        Ok(SourceVolumeObservation {
            relation: relation_for_volume_identities(Some(source), Some(resolved), true),
            source_identity: Some(source),
            resolved_identity: Some(resolved),
            observed_at,
        })
    }

    pub fn reserve(
        &self,
        candidate: &PromotionCandidate,
        source_configured_path: &Path,
        logical_model_owner: &str,
    ) -> Result<ReservationOutcome, ResolvedCacheError> {
        validate_candidate(candidate)?;
        self.reserve_inner(
            &candidate.cache_key,
            source_configured_path,
            logical_model_owner,
            ResolvedCacheProduction::SourceCopy,
            None,
            |_staging| candidate.artifact.clone(),
        )
    }

    /// Reserve an entry for a bundle that will be PRODUCED rather than copied (sc-20635).
    ///
    /// Same entry, same locks, same journal, same staging directory, same atomic publication and
    /// the same receipt as [`Self::reserve`] — the only differences are that no source-library
    /// second copy is required to exist and the entry records
    /// [`ResolvedCacheProduction::Derived`] so retention judges it by the right rule.
    ///
    /// `source_configured_path` is the input the derivative was derived FROM (for a linked
    /// checkpoint, its approved library root). It must exist: a derivative of an absent input is
    /// not something to start producing, and recording it is what lets source-lifecycle
    /// reconciliation reach the derivative when that input goes away.
    pub fn reserve_derived(
        &self,
        plan: &DerivedArtifactPlan,
        source_configured_path: &Path,
        derived_from: &str,
        logical_model_owner: &str,
    ) -> Result<ReservationOutcome, ResolvedCacheError> {
        let cache_key = plan.cache_key()?;
        self.reserve_inner(
            &cache_key,
            source_configured_path,
            logical_model_owner,
            ResolvedCacheProduction::Derived,
            Some(derived_from.to_owned()),
            |staging| plan.artifact_at(staging, ArtifactState::Pending),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reserve_inner(
        &self,
        cache_key: &str,
        source_configured_path: &Path,
        logical_model_owner: &str,
        production: ResolvedCacheProduction,
        derived_from: Option<String>,
        artifact_for_staging: impl FnOnce(&Path) -> ResolvedModelArtifact,
    ) -> Result<ReservationOutcome, ResolvedCacheError> {
        let logical_model_owner = validate_model_owner(logical_model_owner)?.to_owned();
        let source_canonical_path =
            std::fs::canonicalize(source_configured_path).map_err(|error| {
                ResolvedCacheError::new(format!(
                    "source root {} is unavailable: {error}",
                    source_configured_path.display()
                ))
            })?;
        if !source_canonical_path.is_dir() {
            return Err(ResolvedCacheError::new("source root is not a directory"));
        }
        let digest = cache_key_digest(cache_key)?;
        let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => {}
            Err(error) if is_lock_contended(&error) => {
                return Ok(ReservationOutcome::Contended);
            }
            Err(error) => return Err(error.into()),
        }
        let entry = self.entry_path(cache_key)?;
        ensure_managed_entry_dir(&entry)?;
        let _metadata_lock = self.lock_metadata(&digest)?;
        let existing = match self.read_metadata_unlocked(&digest)? {
            JournalRead::Missing => None,
            JournalRead::Evicted { .. } => {
                // A crash interrupted a sanctioned removal. Finish it under the locks this
                // reservation already holds, then materialize from scratch.
                self.finish_pending_eviction(&digest)?;
                ensure_managed_entry_dir(&entry)?;
                None
            }
            JournalRead::Valid { metadata, .. } => Some(*metadata),
        };
        if let Some(mut metadata) = existing {
            match metadata.state {
                ResolvedCacheEntryState::Complete => {
                    // Paths and sizes only (sc-21534): "already complete" is a read verdict that
                    // skips materialization; the load boundary still re-hashes before use. A
                    // same-size alteration is therefore refused at load (falling back to the
                    // source tier) rather than re-materialized here — the same posture as the
                    // local-tier scan.
                    validate_complete_metadata_inner(
                        self,
                        &metadata,
                        ContentVerification::PathsAndSizesOnly,
                    )?;
                    return Ok(ReservationOutcome::AlreadyComplete(Box::new(metadata)));
                }
                ResolvedCacheEntryState::Materializing => {
                    let session = metadata.session_id.as_deref().ok_or_else(|| {
                        ResolvedCacheError::new("materializing cache entry has no owning session")
                    })?;
                    if !self.session_lock_is_acquirable(session)? {
                        return Ok(ReservationOutcome::Contended);
                    }
                    metadata.state = ResolvedCacheEntryState::Interrupted;
                    metadata.recovery_status = RecoveryStatus::InterruptedReservation;
                    metadata.reservation_id = None;
                    metadata.reservation_owner = None;
                    metadata.session_id = None;
                    metadata.updated_at = now_seconds()?;
                    self.write_metadata_unlocked(&digest, &metadata)?;
                }
                _ => {}
            }
        }
        let reservation_id = random_id()?;
        let staging = self
            .inner
            .root
            .join("staging")
            .join(format!("{digest}-{reservation_id}"));
        std::fs::create_dir(&staging).map_err(|error| {
            ResolvedCacheError::new(format!("create materialization staging: {error}"))
        })?;
        let now = now_seconds()?;
        let mut metadata = ResolvedCacheMetadata {
            // The document's own version, not the store's: only a derived entry carries the
            // widened fields, so only it declares the higher version.
            schema_version: production.store_version(),
            cache_key: cache_key.to_owned(),
            production,
            derived_from,
            artifact: artifact_for_staging(&staging),
            source_configured_path: source_configured_path.to_path_buf(),
            source_canonical_path,
            entry_relative_path: PathBuf::from("entries").join(&digest),
            bundle_relative_path: PathBuf::from("entries").join(&digest).join("bundle"),
            state: ResolvedCacheEntryState::Materializing,
            verified_bytes: 0,
            created_at: now,
            updated_at: now,
            last_used_at: None,
            artifact_pinned: false,
            model_pin_owners: BTreeSet::new(),
            effective_pin: false,
            reservation_id: Some(reservation_id.clone()),
            reservation_owner: Some(logical_model_owner.clone()),
            session_id: Some(self.inner.session_id.clone()),
            recovery_status: RecoveryStatus::Clean,
            source_volume: self.compare_source_volume(source_configured_path)?,
        };
        // A reservation is about to (re)write this entry's bundle, so any verdict recorded for the
        // bytes that were there is void. Removing it here rather than trusting the stat comparison
        // to notice keeps the receipt's lifetime tied to the bundle's, and costs one unlink.
        self.remove_content_receipt(&digest)?;
        if let Ok(existing) = self.read_metadata_locked(&digest) {
            metadata.created_at = existing.created_at;
            metadata.artifact_pinned = existing.artifact_pinned;
            metadata.model_pin_owners = existing.model_pin_owners;
            metadata.refresh_effective_pin();
        }
        self.write_metadata_unlocked(&digest, &metadata)?;
        let record = SessionRecord {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: cache_key.to_owned(),
            operation_id: reservation_id.clone(),
            model_owner: logical_model_owner.clone(),
            kind: SessionRecordKind::Reservation,
            acquired_at: now,
        };
        let record_path = self.write_session_record(&record)?;
        Ok(ReservationOutcome::Acquired(Box::new(
            ResolvedCacheReservation {
                store: self.clone(),
                cache_key: cache_key.to_owned(),
                digest,
                reservation_id,
                reservation_owner: logical_model_owner,
                session_id: self.inner.session_id.clone(),
                staging_path: staging,
                record_path,
                artifact_lock: Some(artifact_lock),
                finished: false,
            },
        )))
    }

    pub fn lookup_complete(
        &self,
        cache_key: &str,
    ) -> Result<Option<ResolvedCacheMetadata>, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        let metadata_lock = self.lock_metadata(&digest)?;
        let result = match self.read_metadata_unlocked(&digest)? {
            JournalRead::Valid { metadata, .. }
                if metadata.state == ResolvedCacheEntryState::Complete =>
            {
                // Paths and sizes only (sc-21534): this is a read — callers use it to decide
                // whether materialization can be skipped, and the load boundary
                // ([`Self::acquire_complete`]) still re-hashes before any bytes reach a runtime.
                validate_complete_metadata_inner(
                    self,
                    &metadata,
                    ContentVerification::PathsAndSizesOnly,
                )?;
                Some(*metadata)
            }
            _ => None,
        };
        drop(metadata_lock);
        Ok(result)
    }

    /// Full runtime listing: complete entries are validated on identity, shape, confinement, file
    /// presence and sizes (not content — see [`EntryListing`]). A malformed app-owned entry is
    /// reported as corrupt without hiding unrelated entries; recovery and retention can then
    /// handle that one entry without disabling the whole cache.
    pub fn enumerate(&self) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        self.list(EntryListing::Runtime)
    }

    /// Read-only listing for status/inspection surfaces. Identity, shape, confinement, file
    /// presence and sizes are still proven; bundle bytes are not re-hashed, and an entry that
    /// fails validation degrades to [`ResolvedCacheEntryState::Corrupt`] instead of erasing the
    /// whole listing — a status surface has to be able to *label* the damaged entry rather than
    /// report an empty cache.
    pub fn inspect(&self) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        self.list(EntryListing::Inspect)
    }

    fn list(
        &self,
        listing: EntryListing,
    ) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        let mut entries = Vec::new();
        for item in std::fs::read_dir(self.inner.root.join("entries"))? {
            let item = item?;
            let Some(digest) = item
                .file_name()
                .to_str()
                .filter(|value| is_lower_hex_64(value))
                .map(str::to_owned)
            else {
                tracing::debug!(entry = %item.path().display(), "ignoring foreign resolved-cache entry");
                continue;
            };
            if !item.file_type()?.is_dir() {
                tracing::debug!(entry = %item.path().display(), "ignoring foreign resolved-cache entry");
                continue;
            }
            // The exclusive metadata lock is scoped to the journal READ; the entry is then judged
            // unlocked (sc-19712). Listings validate only paths and sizes, but keeping even that
            // judgement outside the lock prevents `valid_local_artifacts`, the availability read
            // behind every job submission, from parking behind a maintenance sweep. Nothing is
            // read twice here — the judgement is made against the value the read returned, and
            // every caller that ACTS on a summary (`recover`, retention) re-proves it under fresh
            // locks first.
            let journal = {
                let _metadata_lock = self.lock_metadata(&digest)?;
                self.read_metadata_unlocked(&digest)
            };
            #[cfg(test)]
            run_listing_validation_observer();
            let corrupt = |digest: &str| ResolvedCacheEntrySummary {
                cache_key: format!("sha256:{digest}"),
                state: ResolvedCacheEntryState::Corrupt,
                metadata: None,
            };
            let summary = match journal {
                Ok(JournalRead::Valid { metadata, .. }) => {
                    let validated = if metadata.state == ResolvedCacheEntryState::Complete {
                        validate_complete_metadata_inner(self, &metadata, listing.verification())
                    } else {
                        Ok(())
                    };
                    match validated {
                        Ok(()) => ResolvedCacheEntrySummary {
                            cache_key: metadata.cache_key.clone(),
                            state: metadata.state.clone(),
                            metadata: Some(*metadata),
                        },
                        Err(_) => corrupt(&digest),
                    }
                }
                Ok(JournalRead::Evicted { .. }) => ResolvedCacheEntrySummary {
                    cache_key: format!("sha256:{digest}"),
                    state: ResolvedCacheEntryState::Evicting,
                    metadata: None,
                },
                Ok(JournalRead::Missing) | Err(_) => corrupt(&digest),
            };
            entries.push(summary);
        }
        entries.sort_by(|left, right| left.cache_key.cmp(&right.cache_key));
        Ok(entries)
    }

    pub fn checked_verified_bytes(&self) -> Result<u64, ResolvedCacheError> {
        self.enumerate()?
            .into_iter()
            .filter_map(|entry| entry.metadata)
            .try_fold(0_u64, |total, metadata| {
                total.checked_add(metadata.verified_bytes).ok_or_else(|| {
                    ResolvedCacheError::new("resolved-cache verified byte total overflow")
                })
            })
    }

    pub fn set_artifact_pin(
        &self,
        cache_key: &str,
        pinned: bool,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        self.update_metadata(cache_key, |metadata| {
            metadata.artifact_pinned = pinned;
            metadata.refresh_effective_pin();
            Ok(())
        })
    }

    pub fn set_model_pin(
        &self,
        cache_key: &str,
        owner: &str,
        pinned: bool,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let owner = validate_model_owner(owner)?.to_owned();
        self.update_metadata(cache_key, move |metadata| {
            if pinned {
                metadata.model_pin_owners.insert(owner);
            } else {
                metadata.model_pin_owners.remove(&owner);
            }
            metadata.refresh_effective_pin();
            Ok(())
        })
    }

    pub fn effective_pin(&self, cache_key: &str) -> Result<bool, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        let _lock = self.lock_metadata(&digest)?;
        Ok(self.read_metadata_locked(&digest)?.effective_pin)
    }

    pub fn acquire_complete(
        &self,
        cache_key: &str,
        resolver: &ModelArtifactResolver,
        logical_model_owner: &str,
    ) -> Result<Option<ResolvedCacheLease>, ResolvedCacheError> {
        let logical_model_owner = validate_model_owner(logical_model_owner)?.to_owned();
        let digest = cache_key_digest(cache_key)?;
        let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
        FileExt::lock_shared(&artifact_lock)?;
        let metadata_lock = self.lock_metadata(&digest)?;
        let mut metadata = match self.read_metadata_unlocked(&digest)? {
            JournalRead::Valid { metadata, .. }
                if metadata.state == ResolvedCacheEntryState::Complete =>
            {
                *metadata
            }
            _ => return Ok(None),
        };
        // FULL STRENGTH, deliberately — and exactly ONCE (sc-21534): this is the load boundary.
        // The scan ([`Self::valid_local_artifacts`]), the listings, and the retention sweep all
        // validate on paths and sizes only, and this re-hash under the entry's locks is precisely
        // what makes that safe — it is the ONE check that must refuse an altered bundle before any
        // bytes reach a runtime. The lease acquisition and the usage stamp below run under these
        // same locks in this same call, so re-hashing there added no window this check does not
        // already close; it only multiplied the cost (4 full hashes of a 33 GB bundle = the Krea
        // bf16 4.5-minute verify that outlasted the stale-worker sweep). The property is pinned by
        // `the_local_tier_scan_skips_content_hashing_while_the_lease_boundary_still_refuses_altered_bytes`
        // and the single-pass cost by `the_load_boundary_hashes_the_closure_exactly_once`.
        //
        // The pass is now receipt-gated (see `validate_complete_metadata`): it still refuses an
        // altered bundle, but it re-reads the bytes only when the closure's stat identity differs
        // from the one the last successful verification recorded. It reads and stamps that verdict
        // under the `metadata_lock` held above — the same lock the usage stamp below is written
        // under — so a verdict is never recorded without the lock that made it true.
        validate_complete_metadata(self, &metadata)?;
        let artifact = Arc::new(metadata.artifact.clone());
        let runtime_lease = resolver
            .acquire_runtime_lease_prevalidated(&artifact)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        let lease_id = random_id()?;
        let now = now_seconds()?;
        let record = SessionRecord {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: cache_key.to_owned(),
            operation_id: lease_id,
            model_owner: logical_model_owner,
            kind: SessionRecordKind::RuntimeLease,
            acquired_at: now,
        };
        let record_path = self.write_session_record(&record)?;
        metadata.last_used_at = Some(now);
        metadata.updated_at = now;
        self.write_metadata_unlocked(&digest, &metadata)?;
        drop(metadata_lock);
        Ok(Some(ResolvedCacheLease {
            runtime_lease: Some(runtime_lease),
            artifact_lock,
            record_path,
        }))
    }

    pub fn recover(&self) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        let summaries = self.enumerate()?;
        for summary in &summaries {
            let digest = cache_key_digest(&summary.cache_key)?;
            let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
            match FileExt::try_lock_exclusive(&artifact_lock) {
                Ok(()) => {}
                Err(error) if is_lock_contended(&error) => continue,
                Err(error) => return Err(error.into()),
            }
            let _metadata_lock = self.lock_metadata(&digest)?;
            match self.read_metadata_unlocked(&digest) {
                Ok(JournalRead::Valid {
                    mut metadata,
                    had_invalid_slot,
                }) => {
                    let mut changed = false;
                    // The opening listing found a structurally damaged Complete entry. Persist a
                    // per-entry observation so it remains visible for manual recovery, but do
                    // not let this one residue prevent recovery of the rest of the cache. This
                    // stays paths-and-sizes only: `acquire_complete` is still the sole strict
                    // digest boundary before any artifact reaches a runtime.
                    if summary.state == ResolvedCacheEntryState::Corrupt
                        && metadata.state == ResolvedCacheEntryState::Complete
                        && validate_complete_metadata_inner(
                            self,
                            &metadata,
                            ContentVerification::PathsAndSizesOnly,
                        )
                        .is_err()
                    {
                        self.write_corrupt_marker(&digest)?;
                        continue;
                    }
                    if had_invalid_slot {
                        // FULL STRENGTH before the pin (sc-21534, same defense as the receipt
                        // resurrection below): this branch re-pins a Complete entry recovered
                        // from its older journal slot, and the listings — including this
                        // method's opening `enumerate()` — no longer hash content. A
                        // content-altered bundle pinned here would be refused at every load yet
                        // never evicted — a permanent disk leak — so prove the bytes first. On
                        // failure, record the observation and leave the entry UNPINNED and
                        // unhealed: the load boundary refuses it regardless, and staying
                        // unpinned is what lets retention reclaim it (after proving the source
                        // holds a complete second copy) so the next use re-materializes clean.
                        if metadata.state == ResolvedCacheEntryState::Complete
                            && validate_complete_metadata(self, &metadata).is_err()
                        {
                            self.write_corrupt_marker(&digest)?;
                        } else {
                            metadata.recovery_status = RecoveryStatus::RecoveredFromOlderSlot;
                            metadata.artifact_pinned = true;
                            metadata.refresh_effective_pin();
                            changed = true;
                        }
                    }
                    let materializing_session_is_stale =
                        if metadata.state == ResolvedCacheEntryState::Materializing {
                            match metadata.session_id.as_deref() {
                                Some(session) => self.session_lock_is_acquirable(session)?,
                                None => false,
                            }
                        } else {
                            false
                        };
                    if materializing_session_is_stale {
                        metadata.state = ResolvedCacheEntryState::Interrupted;
                        metadata.recovery_status = RecoveryStatus::InterruptedReservation;
                        metadata.reservation_id = None;
                        metadata.reservation_owner = None;
                        metadata.session_id = None;
                        changed = true;
                    }
                    if changed {
                        metadata.updated_at = now_seconds()?;
                        self.write_metadata_unlocked(&digest, &metadata)?;
                    }
                }
                Ok(JournalRead::Evicted { .. }) => {
                    // A valid tombstone means removal was sanctioned before the interruption;
                    // finishing it is the safe convergence.
                    self.finish_pending_eviction(&digest)?;
                }
                Ok(JournalRead::Missing) => {
                    // Another session may have finished an eviction (removing the whole entry
                    // directory) between enumeration and this pass; only an entry that still
                    // exists without readable metadata is parked as corrupt.
                    if self
                        .inner
                        .root
                        .join("entries")
                        .join(&digest)
                        .symlink_metadata()
                        .is_ok()
                    {
                        self.write_corrupt_marker(&digest)?;
                    }
                }
                Err(_) => {
                    // FULL STRENGTH kept here (sc-21534): unlike the listings, this path
                    // resurrects an entry from its receipt and PINS it. A content-corrupt bundle
                    // resurrected pinned would be refused at every load yet never evicted, so the
                    // rare invalid-tombstone recovery proves the bytes before it re-pins them.
                    let receipt = self.read_complete_receipt(&digest).and_then(|metadata| {
                        validate_complete_metadata(self, &metadata)?;
                        Ok(metadata)
                    });
                    if let Ok(mut metadata) = receipt {
                        // An INVALID tombstone routed us here (a valid one is handled above). It
                        // does not prove a sanctioned eviction, so fail safe: drop the garbage
                        // tombstone and resurrect the receipt-validated entry pinned.
                        let marker_path = self.eviction_marker_path(&digest);
                        if std::fs::symlink_metadata(&marker_path).is_ok() {
                            std::fs::remove_file(&marker_path)?;
                        }
                        metadata.recovery_status = RecoveryStatus::ReconstructedFromCompleteReceipt;
                        metadata.artifact_pinned = true;
                        metadata.refresh_effective_pin();
                        metadata.updated_at = now_seconds()?;
                        self.write_metadata_unlocked(&digest, &metadata)?;
                    } else {
                        self.write_corrupt_marker(&digest)?;
                    }
                }
            }
        }
        self.cleanup_stale_staging()?;
        self.clean_stale_sessions()?;
        self.enumerate()
    }

    /// Remove only exact, lock-authorized stale materialization staging directories. Malformed
    /// names, reparse points and live reservations are retained fail-closed. Complete bundles live
    /// under `entries/` and are never traversed by this cleanup.
    pub fn cleanup_stale_staging(&self) -> Result<usize, ResolvedCacheError> {
        let staging_root = self.inner.root.join("staging");
        let mut removed = 0;
        for item in std::fs::read_dir(&staging_root)? {
            let item = item?;
            let name = item.file_name();
            let Some((digest, reservation_id)) = name.to_str().and_then(parse_staging_name) else {
                continue;
            };
            let metadata = item.file_type()?;
            if !metadata.is_dir() || metadata.is_symlink() {
                continue;
            }
            let artifact_lock = open_lock_file(&self.artifact_lock_path(digest))?;
            match FileExt::try_lock_exclusive(&artifact_lock) {
                Ok(()) => {}
                Err(error) if is_lock_contended(&error) => continue,
                Err(error) => return Err(error.into()),
            }
            let _metadata_lock = self.lock_metadata(digest)?;
            let live = match self.read_metadata_unlocked(digest) {
                Ok(JournalRead::Valid { metadata, .. })
                    if metadata.state == ResolvedCacheEntryState::Materializing
                        && metadata.reservation_id.as_deref() == Some(reservation_id) =>
                {
                    match metadata.session_id.as_deref() {
                        Some(session) => !self.session_lock_is_acquirable(session)?,
                        None => true,
                    }
                }
                _ => false,
            };
            if live {
                continue;
            }
            remove_managed_tree(&item.path(), &staging_root)?;
            removed += 1;
        }
        Ok(removed)
    }

    fn update_metadata(
        &self,
        cache_key: &str,
        update: impl FnOnce(&mut ResolvedCacheMetadata) -> Result<(), ResolvedCacheError>,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        let _lock = self.lock_metadata(&digest)?;
        let mut metadata = self.read_metadata_locked(&digest)?;
        update(&mut metadata)?;
        metadata.updated_at = now_seconds()?;
        self.write_metadata_unlocked(&digest, &metadata)?;
        Ok(metadata)
    }

    fn artifact_lock_path(&self, digest: &str) -> PathBuf {
        self.inner
            .root
            .join("locks")
            .join(format!("{digest}.artifact.lock"))
    }

    fn metadata_lock_path(&self, digest: &str) -> PathBuf {
        self.inner
            .root
            .join("locks")
            .join(format!("{digest}.metadata.lock"))
    }

    fn lock_metadata(&self, digest: &str) -> Result<File, ResolvedCacheError> {
        let file = open_lock_file(&self.metadata_lock_path(digest))?;
        FileExt::lock_exclusive(&file)?;
        Ok(file)
    }

    fn read_metadata_locked(
        &self,
        digest: &str,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        match self.read_metadata_unlocked(digest)? {
            JournalRead::Valid { metadata, .. } => Ok(*metadata),
            // Both are an intact store declining this request, not a store that broke: a removal
            // is genuinely in flight for the first, and the second names an entry it does not hold.
            JournalRead::Evicted { .. } => Err(ResolvedCacheError::request(
                "cache entry has a pending eviction tombstone",
            )),
            JournalRead::Missing => Err(ResolvedCacheError::request("cache metadata is missing")),
        }
    }

    fn eviction_marker_path(&self, digest: &str) -> PathBuf {
        self.inner
            .root
            .join("entries")
            .join(digest)
            .join(EVICTED_MARKER_FILE)
    }

    /// Reads the eviction tombstone if one exists. An unreadable or checksum-invalid tombstone is
    /// an error, never a sanction to delete: readers surface it and recovery either resurrects the
    /// entry from a valid complete receipt or parks it as corrupt.
    ///
    /// This probe runs on *every* metadata read, so the overwhelmingly common "no tombstone" case
    /// must stay cheap and must not consume a file descriptor: it is decided by a single
    /// `symlink_metadata`, and only a tombstone that is actually present is opened and read. An
    /// earlier revision read the path unconditionally, adding an open/read/close to every metadata
    /// read; under descriptor pressure that turned into an unreadable-metadata error inside
    /// `ResolvedCacheReservation::drop`, which left the entry `Materializing` and reddened the
    /// pre-existing stale-staging cleanup test on the hosted macOS lane. The tombstone is also
    /// confinement-checked here like every other managed path in this module.
    fn read_eviction_marker(
        &self,
        digest: &str,
    ) -> Result<Option<EvictionMarker>, ResolvedCacheError> {
        let path = self.eviction_marker_path(digest);
        match std::fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || metadata_is_reparse_point(&metadata)
                    || !metadata.is_file()
                {
                    return Err(ResolvedCacheError::new(format!(
                        "eviction tombstone {} is a link or not a regular file",
                        path.display()
                    )));
                }
            }
        }
        let body = match std::fs::read(&path) {
            Ok(body) => body,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let envelope: EvictionMarkerEnvelope = serde_json::from_slice(&body).map_err(|error| {
            ResolvedCacheError::new(format!("decode eviction tombstone: {error}"))
        })?;
        envelope.validate(digest)?;
        Ok(Some(envelope.marker))
    }

    /// Writes the durable eviction tombstone. Caller holds the exclusive artifact lock and the
    /// metadata lock and has already re-verified every eviction protection under those locks.
    fn write_eviction_marker(
        &self,
        digest: &str,
        marker: &EvictionMarker,
    ) -> Result<(), ResolvedCacheError> {
        if cache_key_digest(&marker.cache_key)? != digest {
            return Err(ResolvedCacheError::new(
                "eviction tombstone cache key does not match its entry",
            ));
        }
        let envelope = EvictionMarkerEnvelope::new(marker.clone())?;
        atomic_write_json(&self.eviction_marker_path(digest), &envelope)
    }

    /// Finishes a tombstoned removal: removes the whole entry directory (bundle, journal slots,
    /// receipt, tombstone) with the confined deleter, then records the audit trail outside the
    /// entry. Caller holds the exclusive artifact lock and the metadata lock. Idempotent under
    /// interruption: rerunning after a crash converges because the tombstone survives until the
    /// directory removal completes.
    ///
    /// The audit trail is two-state so a removal failure and an audit-write failure stay
    /// distinguishable. A `Started` record is written *before* the last unlink, so the intent
    /// survives outside the entry even though the tombstone dies with it; the record is then
    /// rewritten `Completed` once the bytes are actually gone. A `Started` record left behind by a
    /// crash or a sharing violation is therefore an honest "attempted, unconfirmed" — never the
    /// false "completed" an earlier revision recorded — and the deterministic per-entry record
    /// path means a retry overwrites its own record instead of accumulating duplicates.
    ///
    /// Audit-write failures after a successful removal do not fail the eviction: the bytes really
    /// are gone, so reporting failure would corrupt size accounting and re-drive eviction. Those
    /// are warned instead.
    fn finish_pending_eviction(&self, digest: &str) -> Result<EvictionMarker, ResolvedCacheError> {
        let marker = self.read_eviction_marker(digest)?.ok_or_else(|| {
            ResolvedCacheError::new("cache entry has no eviction tombstone to finish")
        })?;
        if let Err(error) =
            self.write_eviction_audit_record(digest, &marker, EvictionAuditStatus::Started)
        {
            tracing::warn!(
                cache_key = %marker.cache_key,
                error = %error,
                "could not record resolved-cache eviction intent before removal"
            );
        }
        let entries = self.inner.root.join("entries");
        remove_managed_tree(&entries.join(digest), &entries)?;
        if let Err(error) =
            self.write_eviction_audit_record(digest, &marker, EvictionAuditStatus::Completed)
        {
            tracing::warn!(
                cache_key = %marker.cache_key,
                error = %error,
                "resolved-cache entry was removed but its completed audit record could not be \
                 written; the started record remains"
            );
        }
        Ok(marker)
    }

    fn write_eviction_audit_record(
        &self,
        digest: &str,
        marker: &EvictionMarker,
        status: EvictionAuditStatus,
    ) -> Result<(), ResolvedCacheError> {
        let audit = self.inner.root.join(AUDIT_DIR);
        match std::fs::create_dir(&audit) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        ensure_regular_directory(&audit)?;
        let record = EvictionAuditRecord {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            marker: marker.clone(),
            status,
            recorded_at: now_seconds()?,
            recorded_by_session: self.inner.session_id.clone(),
        };
        let name = format!("{digest}-{}.json", marker.requested_at);
        atomic_write_json(&audit.join(name), &record)
    }

    fn read_metadata_unlocked(&self, digest: &str) -> Result<JournalRead, ResolvedCacheError> {
        if let Some(marker) = self.read_eviction_marker(digest)? {
            return Ok(JournalRead::Evicted {
                marker: Box::new(marker),
            });
        }
        let entry = self.inner.root.join("entries").join(digest);
        let mut valid = Vec::new();
        let mut had_file = false;
        let mut had_invalid_slot = false;
        let mut unsupported_version: Option<ResolvedCacheError> = None;
        for slot in 0..=1 {
            let path = entry.join(format!("metadata.{slot}.json"));
            if !path.exists() {
                continue;
            }
            had_file = true;
            // Slot SELECTION is paths-and-sizes only. A journal slot's own integrity is proven by
            // its checksummed envelope; re-hashing the whole bundle to decide which slot to read
            // would put the bundle's byte count behind every pin read, every status listing and
            // every catalog GET, and it would prove nothing a caller may act on — the load path
            // ([`validate_complete_metadata`]) still re-hashes before an artifact reaches a
            // runtime, and every write still validates at full strength.
            match read_journal(&path) {
                Ok(envelope)
                    if validate_metadata_shape_with(
                        &envelope.metadata,
                        digest,
                        ContentVerification::PathsAndSizesOnly,
                    )
                    .is_ok() =>
                {
                    valid.push(envelope)
                }
                Err(error) if error.to_string().contains(UNSUPPORTED_JOURNAL_VERSION) => {
                    had_invalid_slot = true;
                    unsupported_version = Some(error);
                }
                _ => had_invalid_slot = true,
            }
        }
        valid.sort_by_key(|envelope| envelope.generation);
        if let Some(envelope) = valid.pop() {
            return Ok(JournalRead::Valid {
                metadata: Box::new(envelope.metadata),
                had_invalid_slot,
            });
        }
        // A journal a NEWER build wrote is not a corrupt store, and saying so matters: the
        // corrupt classification is the one manual removal is allowed to clear, so reporting a
        // forward-version entry as corrupt would invite deleting an entry this binary simply
        // cannot read yet (sc-20635).
        if let Some(error) = unsupported_version {
            return Err(error);
        }
        if had_file {
            Err(ResolvedCacheError::new(BOTH_METADATA_SLOTS_CORRUPT))
        } else {
            Ok(JournalRead::Missing)
        }
    }

    fn write_metadata_unlocked(
        &self,
        digest: &str,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<(), ResolvedCacheError> {
        // Shape only (sc-21534): a journal write records a state or usage transition; it never
        // changes bundle bytes, and re-hashing the closure here charged a full-bundle SHA-256 to
        // every usage stamp and pin flip. Every boundary that STAMPS `Complete` content proves it
        // itself immediately before writing (`record_complete` / `publish_staged` / `recover`'s
        // receipt path run a full `artifact.validate()`), and the load boundary
        // (`acquire_complete`) re-hashes before any bytes reach a runtime.
        validate_metadata_shape_with(metadata, digest, ContentVerification::PathsAndSizesOnly)?;
        let generation = match self.read_metadata_unlocked(digest) {
            Ok(JournalRead::Valid { .. }) => {
                highest_generation(&self.inner.root.join("entries").join(digest))?
                    .checked_add(1)
                    .ok_or_else(|| ResolvedCacheError::new("cache metadata generation overflow"))?
            }
            _ => 1,
        };
        let envelope = JournalEnvelope::new(generation, metadata.clone())?;
        let slot = generation % 2;
        let path = self
            .inner
            .root
            .join("entries")
            .join(digest)
            .join(format!("metadata.{slot}.json"));
        atomic_write_json(&path, &envelope)
    }

    fn write_complete_receipt(
        &self,
        digest: &str,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<(), ResolvedCacheError> {
        let envelope = ReceiptEnvelope::new(metadata.clone())?;
        atomic_write_json(
            &self
                .inner
                .root
                .join("entries")
                .join(digest)
                .join("complete.receipt.json"),
            &envelope,
        )
    }

    fn read_complete_receipt(
        &self,
        digest: &str,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let path = self
            .inner
            .root
            .join("entries")
            .join(digest)
            .join("complete.receipt.json");
        let body = std::fs::read(&path)?;
        let envelope: ReceiptEnvelope = serde_json::from_slice(&body).map_err(|error| {
            ResolvedCacheError::new(format!("decode complete receipt: {error}"))
        })?;
        envelope.validate()?;
        Ok(envelope.metadata)
    }

    fn content_receipt_path(&self, digest: &str) -> PathBuf {
        self.inner
            .root
            .join("entries")
            .join(digest)
            .join("content.verification.json")
    }

    /// The recorded content verdict for this entry, or `None` for every reason there might not be
    /// one.
    ///
    /// Deliberately infallible-by-collapse: absent, unreadable, unparseable, written by a newer
    /// build, or carrying another entry's key all return `None`. Each of those means the caller
    /// re-reads the closure, which is both the safe answer and the behaviour that predates
    /// receipts — so there is no failure here worth propagating to a load.
    fn read_content_receipt(&self, digest: &str, cache_key: &str) -> Option<ClosureContentReceipt> {
        let body = std::fs::read(self.content_receipt_path(digest)).ok()?;
        let receipt: ClosureContentReceipt = serde_json::from_slice(&body).ok()?;
        (receipt.schema_version == RESOLVED_CACHE_STORE_VERSION && receipt.cache_key == cache_key)
            .then_some(receipt)
    }

    fn write_content_receipt(
        &self,
        digest: &str,
        receipt: &ClosureContentReceipt,
    ) -> Result<(), ResolvedCacheError> {
        atomic_write_json(&self.content_receipt_path(digest), receipt)
    }

    fn remove_content_receipt(&self, digest: &str) -> Result<(), ResolvedCacheError> {
        match std::fs::remove_file(self.content_receipt_path(digest)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn write_corrupt_marker(&self, digest: &str) -> Result<(), ResolvedCacheError> {
        let marker = CorruptMarker {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: format!("sha256:{digest}"),
            state: ResolvedCacheEntryState::Corrupt,
            recovery_status: RecoveryStatus::CorruptUnrecoverable,
            observed_at: now_seconds()?,
        };
        atomic_write_json(
            &self
                .inner
                .root
                .join("entries")
                .join(digest)
                .join("corrupt.marker.json"),
            &marker,
        )
    }

    fn write_session_record(&self, record: &SessionRecord) -> Result<PathBuf, ResolvedCacheError> {
        let path = self
            .inner
            .root
            .join("sessions")
            .join(&self.inner.session_id)
            .join(format!("{}.json", record.operation_id));
        atomic_write_json(&path, record)?;
        Ok(path)
    }

    fn session_lock_is_acquirable(&self, session: &str) -> Result<bool, ResolvedCacheError> {
        if !is_valid_session_id(session) || session == self.inner.session_id {
            return Ok(false);
        }
        let path = self
            .inner
            .root
            .join("sessions")
            .join(format!("{session}.lock"));
        let file = open_lock_file(&path)?;
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => Ok(true),
            Err(error) if is_lock_contended(&error) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn clean_stale_sessions(&self) -> Result<(), ResolvedCacheError> {
        let sessions = self.inner.root.join("sessions");
        for item in std::fs::read_dir(&sessions)? {
            let item = item?;
            let name = item.file_name().to_string_lossy().into_owned();
            let Some(session) = name.strip_suffix(".lock") else {
                continue;
            };
            if !is_valid_session_id(session) || session == self.inner.session_id {
                continue;
            }
            let file = open_lock_file(&item.path())?;
            match FileExt::try_lock_exclusive(&file) {
                Ok(()) => {
                    let records = sessions.join(session);
                    if records.is_dir() {
                        std::fs::remove_dir_all(records)?;
                    }
                    drop(file);
                    let _ = std::fs::remove_file(item.path());
                }
                Err(error) if is_lock_contended(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

pub struct ResolvedCacheReservation {
    store: ResolvedCacheStore,
    cache_key: String,
    digest: String,
    reservation_id: String,
    reservation_owner: String,
    session_id: String,
    staging_path: PathBuf,
    record_path: PathBuf,
    artifact_lock: Option<File>,
    finished: bool,
}

impl std::fmt::Debug for ResolvedCacheReservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedCacheReservation")
            .field("cache_key", &self.cache_key)
            .field("reservation_id", &self.reservation_id)
            .field("staging_path", &self.staging_path)
            .finish()
    }
}

impl ResolvedCacheReservation {
    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }

    pub fn reservation_id(&self) -> &str {
        &self.reservation_id
    }

    pub fn staging_path(&self) -> &Path {
        &self.staging_path
    }

    pub fn bundle_path(&self) -> Result<PathBuf, ResolvedCacheError> {
        self.store.bundle_path(&self.cache_key)
    }

    pub(crate) fn prepare_for_materialization(&self) -> Result<(), ResolvedCacheError> {
        let _metadata_lock = self.store.lock_metadata(&self.digest)?;
        let metadata = self.store.read_metadata_locked(&self.digest)?;
        self.verify_ownership(&metadata)?;
        reject_link_or_reparse(&self.staging_path, "resolved cache staging directory")?;
        if std::fs::read_dir(&self.staging_path)?.next().is_some() {
            return Err(ResolvedCacheError::new(
                "materialization staging directory is not empty",
            ));
        }
        let bundle = self.bundle_path()?;
        if std::fs::symlink_metadata(&bundle).is_ok() {
            let entry = self.store.entry_path(&self.cache_key)?;
            remove_managed_tree(&bundle, &entry)?;
        }
        Ok(())
    }

    pub(crate) fn publish_staged(
        self,
        mut artifact: ResolvedModelArtifact,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        if !matches!(&artifact.location, ArtifactLocation::ResolvedLocal { root } if root == &self.staging_path)
        {
            return Err(ResolvedCacheError::new(
                "staged artifact is not rooted at its reservation staging directory",
            ));
        }
        validate_staging_confinement(&self)?;
        artifact
            .validate()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        if artifact
            .cache_key()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            != self.cache_key
        {
            return Err(ResolvedCacheError::new(
                "staged artifact cache key differs from its reservation",
            ));
        }
        sync_tree(&self.staging_path)?;
        let entry = self.store.entry_path(&self.cache_key)?;
        let bundle = self.bundle_path()?;
        if std::fs::symlink_metadata(&bundle).is_ok() {
            return Err(ResolvedCacheError::new(
                "resolved cache bundle appeared before atomic publication",
            ));
        }
        {
            let _metadata_lock = self.store.lock_metadata(&self.digest)?;
            let metadata = self.store.read_metadata_locked(&self.digest)?;
            self.verify_ownership(&metadata)?;
        }
        publish_rename(&self.staging_path, &bundle)?;
        sync_dir(&entry).map_err(|error| {
            ResolvedCacheError::new(format!(
                "sync published entry directory {}: {error}",
                entry.display()
            ))
        })?;
        artifact.location = ArtifactLocation::ResolvedLocal { root: bundle };
        self.record_complete(artifact)
    }

    /// Publish a bundle the producer wrote into [`Self::staging_path`] (sc-20635).
    ///
    /// The producer declares its outputs up front in `plan`; this reads back exactly those staged
    /// files, enriches the closure with their measured size and SHA-256, and hands the result to
    /// the same [`Self::publish_staged`] the copy path uses — so the artifact is re-validated
    /// against the bytes on disk, its cache key is proven unchanged, the tree is fsynced, and the
    /// staging directory is atomically renamed into place.
    ///
    /// Every way a producer can leave a partial or wrong bundle refuses here rather than being
    /// published: a declared file it did not write, a declared path that is a symlink or a
    /// directory, or a file it wrote that the plan does not declare (which would leave bytes the
    /// entry's own accounting never counts and retention would over-reclaim against).
    pub(crate) fn publish_produced(
        self,
        plan: &DerivedArtifactPlan,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let mut artifact = plan.artifact_at(&self.staging_path, ArtifactState::Published);
        let mut declared = BTreeSet::new();
        for member in &mut artifact.closure.members {
            for file in &mut member.files {
                let relative = member.destination.join(&file.relative_path);
                let path = self.staging_path.join(&relative);
                let (size_bytes, sha256) = measure_produced_file(&path)?;
                file.size_bytes = Some(size_bytes);
                file.sha256 = Some(sha256);
                declared.insert(relative);
            }
        }
        artifact.closure = ResolvedBundleClosure::new(artifact.closure.members)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        reject_undeclared_produced_files(&self.staging_path, &declared)?;
        self.publish_staged(artifact)
    }

    /// Abandon a reservation whose producer failed: drop whatever it staged and record the entry
    /// as interrupted, so the next attempt starts from an empty staging directory rather than
    /// inheriting a half-written one.
    pub(crate) fn discard(mut self) -> Result<(), ResolvedCacheError> {
        if self.staging_path.exists() {
            let staging_root = self.store.root().join("staging");
            remove_managed_tree(&self.staging_path, &staging_root)?;
        }
        self.mark_interrupted()?;
        Ok(())
    }

    /// Marks only this still-owned reservation interrupted. The private reservation/session
    /// identities and held artifact lock prevent key-only or cross-owner invalidation.
    pub fn mark_interrupted(&mut self) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let _metadata_lock = self.store.lock_metadata(&self.digest)?;
        let mut metadata = self.store.read_metadata_locked(&self.digest)?;
        self.verify_ownership(&metadata)?;
        metadata.state = ResolvedCacheEntryState::Interrupted;
        metadata.reservation_id = None;
        metadata.reservation_owner = None;
        metadata.session_id = None;
        metadata.recovery_status = RecoveryStatus::InterruptedReservation;
        metadata.updated_at = now_seconds()?;
        self.store
            .write_metadata_unlocked(&self.digest, &metadata)?;
        self.finished = true;
        let _ = std::fs::remove_file(&self.record_path);
        self.artifact_lock.take();
        Ok(metadata)
    }

    /// Records publication after a later materializer has atomically installed and validated the
    /// bundle. This method does not copy, move, or delete any model bytes.
    pub fn record_complete(
        mut self,
        artifact: ResolvedModelArtifact,
    ) -> Result<ResolvedCacheMetadata, ResolvedCacheError> {
        let expected_root = self.bundle_path()?;
        if !matches!(&artifact.location, ArtifactLocation::ResolvedLocal { root } if root == &expected_root)
        {
            return Err(ResolvedCacheError::new(
                "complete artifact is not rooted at its resolved-cache bundle",
            ));
        }
        validate_completion_confinement(&self.store, &self.cache_key, &artifact)?;
        artifact
            .validate()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        if artifact
            .cache_key()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            != self.cache_key
        {
            return Err(ResolvedCacheError::new(
                "complete artifact cache key differs from its reservation",
            ));
        }
        let verified_bytes = checked_artifact_bytes(&artifact)?;
        let _metadata_lock = self.store.lock_metadata(&self.digest)?;
        let mut metadata = self.store.read_metadata_locked(&self.digest)?;
        self.verify_ownership(&metadata)?;
        metadata.artifact = artifact;
        metadata.state = ResolvedCacheEntryState::Complete;
        metadata.verified_bytes = verified_bytes;
        metadata.updated_at = now_seconds()?;
        metadata.reservation_id = None;
        metadata.reservation_owner = None;
        metadata.session_id = None;
        metadata.recovery_status = RecoveryStatus::Clean;
        self.store.write_complete_receipt(&self.digest, &metadata)?;
        self.store
            .write_metadata_unlocked(&self.digest, &metadata)?;
        self.finished = true;
        let _ = std::fs::remove_file(&self.record_path);
        self.artifact_lock.take();
        Ok(metadata)
    }

    fn verify_ownership(&self, metadata: &ResolvedCacheMetadata) -> Result<(), ResolvedCacheError> {
        if metadata.state != ResolvedCacheEntryState::Materializing
            || metadata.reservation_id.as_deref() != Some(&self.reservation_id)
            || metadata.reservation_owner.as_deref() != Some(&self.reservation_owner)
            || metadata.session_id.as_deref() != Some(&self.session_id)
        {
            return Err(ResolvedCacheError::new(
                "materialization reservation ownership changed",
            ));
        }
        Ok(())
    }
}

impl Drop for ResolvedCacheReservation {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // A reservation that cannot record its own interruption leaves a `Materializing` entry
        // owned by this still-live session, which every liveness check must keep treating as live.
        // Recovery cannot converge that until the process exits and its session lock frees, so the
        // condition is warned rather than silently swallowed.
        if let Err(error) = self.mark_interrupted_on_drop() {
            tracing::warn!(
                cache_key = %self.cache_key,
                reservation_id = %self.reservation_id,
                error = %error,
                "resolved-cache reservation could not record its interruption; the entry stays \
                 materializing until this session ends"
            );
        }
        let _ = std::fs::remove_file(&self.record_path);
        self.artifact_lock.take();
    }
}

impl ResolvedCacheReservation {
    fn mark_interrupted_on_drop(&self) -> Result<(), ResolvedCacheError> {
        let _metadata_lock = self.store.lock_metadata(&self.digest)?;
        let mut metadata = self.store.read_metadata_locked(&self.digest)?;
        if self.verify_ownership(&metadata).is_err() {
            // Another owner legitimately took the entry over; nothing of ours to interrupt.
            return Ok(());
        }
        metadata.state = ResolvedCacheEntryState::Interrupted;
        metadata.reservation_id = None;
        metadata.reservation_owner = None;
        metadata.session_id = None;
        metadata.recovery_status = RecoveryStatus::InterruptedReservation;
        metadata.updated_at = now_seconds().unwrap_or(metadata.updated_at);
        self.store.write_metadata_unlocked(&self.digest, &metadata)
    }
}

#[derive(Debug)]
pub struct ResolvedCacheLease {
    runtime_lease: Option<ActiveArtifactLease>,
    #[allow(dead_code)]
    artifact_lock: File,
    record_path: PathBuf,
}

impl ResolvedCacheLease {
    pub fn artifact(&self) -> &ResolvedModelArtifact {
        self.runtime_lease
            .as_ref()
            .expect("resolved cache lease is active")
            .artifact()
    }

    pub fn mark_success(mut self) -> PromotionCandidate {
        let candidate = self
            .runtime_lease
            .take()
            .expect("resolved cache lease is active")
            .mark_success();
        let _ = std::fs::remove_file(&self.record_path);
        candidate
    }
}

impl Drop for ResolvedCacheLease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.record_path);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct JournalEnvelope {
    schema_version: u32,
    generation: u64,
    checksum: String,
    metadata: ResolvedCacheMetadata,
}

impl JournalEnvelope {
    fn new(generation: u64, metadata: ResolvedCacheMetadata) -> Result<Self, ResolvedCacheError> {
        let checksum = journal_checksum(generation, &metadata)?;
        Ok(Self {
            // The envelope declares the version of the document it carries: an older reader stops
            // at the envelope, so a derived document's higher version has to be visible there.
            schema_version: metadata.schema_version,
            generation,
            checksum,
            metadata,
        })
    }

    fn validate(&self) -> Result<(), ResolvedCacheError> {
        if self.schema_version != self.metadata.schema_version
            || self.checksum != journal_checksum(self.generation, &self.metadata)?
        {
            return Err(ResolvedCacheError::new(
                "cache metadata checksum is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReceiptEnvelope {
    schema_version: u32,
    checksum: String,
    metadata: ResolvedCacheMetadata,
}

impl ReceiptEnvelope {
    fn new(metadata: ResolvedCacheMetadata) -> Result<Self, ResolvedCacheError> {
        let checksum = metadata_checksum(&metadata)?;
        Ok(Self {
            schema_version: metadata.schema_version,
            checksum,
            metadata,
        })
    }

    fn validate(&self) -> Result<(), ResolvedCacheError> {
        if self.schema_version != self.metadata.schema_version
            || self.metadata.state != ResolvedCacheEntryState::Complete
            || self.checksum != metadata_checksum(&self.metadata)?
        {
            return Err(ResolvedCacheError::new(
                "complete receipt checksum is invalid",
            ));
        }
        Ok(())
    }
}

enum JournalRead {
    Missing,
    /// A durable eviction tombstone governs this entry: removal was sanctioned and must finish
    /// before the entry can be read or reused. The journal slots are no longer authoritative.
    Evicted {
        marker: Box<EvictionMarker>,
    },
    Valid {
        metadata: Box<ResolvedCacheMetadata>,
        had_invalid_slot: bool,
    },
}

/// Durable, checksummed eviction tombstone. Written under the exclusive artifact and metadata
/// locks before any byte of the entry is deleted, so an interrupted removal always converges:
/// every reader treats a valid tombstone as "this entry is gone", and recovery finishes the
/// deletion instead of resurrecting a half-removed entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvictionMarker {
    pub schema_version: u32,
    pub cache_key: String,
    pub cause: retention::EvictionCause,
    pub reclaimable_bytes: u64,
    pub requested_at: u64,
    pub session_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvictionMarkerEnvelope {
    schema_version: u32,
    checksum: String,
    marker: EvictionMarker,
}

impl EvictionMarkerEnvelope {
    fn new(marker: EvictionMarker) -> Result<Self, ResolvedCacheError> {
        let checksum = eviction_marker_checksum(&marker)?;
        Ok(Self {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            checksum,
            marker,
        })
    }

    fn validate(&self, digest: &str) -> Result<(), ResolvedCacheError> {
        if self.schema_version != RESOLVED_CACHE_STORE_VERSION
            || self.marker.schema_version != RESOLVED_CACHE_STORE_VERSION
            || self.checksum != eviction_marker_checksum(&self.marker)?
            || cache_key_digest(&self.marker.cache_key)? != digest
        {
            return Err(ResolvedCacheError::new(
                "eviction tombstone checksum is invalid",
            ));
        }
        Ok(())
    }
}

fn eviction_marker_checksum(marker: &EvictionMarker) -> Result<String, ResolvedCacheError> {
    let bytes = serde_json::to_vec(marker).map_err(|error| {
        ResolvedCacheError::new(format!("encode eviction tombstone checksum: {error}"))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

/// Audit record persisted outside the removed entry so the eviction remains auditable after the
/// entry directory (and the tombstone inside it) is gone.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvictionAuditRecord {
    pub schema_version: u32,
    pub marker: EvictionMarker,
    pub status: EvictionAuditStatus,
    pub recorded_at: u64,
    pub recorded_by_session: String,
}

/// `Started` means the removal was authorized and attempted; `Completed` means the bytes are
/// confirmed gone. A record left at `Started` is an attempted-but-unconfirmed removal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionAuditStatus {
    Started,
    Completed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SessionRecordKind {
    Reservation,
    RuntimeLease,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SessionRecord {
    schema_version: u32,
    cache_key: String,
    operation_id: String,
    model_owner: String,
    kind: SessionRecordKind,
    acquired_at: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CorruptMarker {
    schema_version: u32,
    cache_key: String,
    state: ResolvedCacheEntryState,
    recovery_status: RecoveryStatus,
    observed_at: u64,
}

fn initialize_or_validate_root(root: &Path) -> Result<(), ResolvedCacheError> {
    match std::fs::symlink_metadata(root) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink()
                || metadata_is_reparse_point(&metadata)
                || !metadata.is_dir()
            {
                return Err(ResolvedCacheError::new(format!(
                    "resolved cache root {} is a symlink or not a directory",
                    root.display()
                )));
            }
            let marker = std::fs::read(root.join(STORE_MARKER)).map_err(|_| {
                ResolvedCacheError::new(format!(
                    "refusing to adopt unmanaged resolved cache root {}",
                    root.display()
                ))
            })?;
            if marker != STORE_MARKER_BODY {
                return Err(ResolvedCacheError::new(
                    "resolved cache root marker is invalid or unsupported",
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = root
                .parent()
                .ok_or_else(|| ResolvedCacheError::new("resolved cache root has no parent"))?;
            let staging = parent.join(format!(".resolved-init-{}", random_id()?));
            std::fs::create_dir(&staging)?;
            for name in ["entries", "locks", "sessions", "staging"] {
                std::fs::create_dir(staging.join(name))?;
            }
            write_synced_file(&staging.join(STORE_MARKER), STORE_MARKER_BODY)?;
            sync_dir(&staging)?;
            match std::fs::rename(&staging, root) {
                Ok(()) => sync_dir(parent)?,
                Err(_rename_error) if root.exists() => {
                    std::fs::remove_dir_all(&staging)?;
                    initialize_or_validate_root(root)?;
                }
                Err(rename_error) => {
                    let _ = std::fs::remove_dir_all(&staging);
                    return Err(rename_error.into());
                }
            }
        }
        Err(error) => return Err(error.into()),
    }
    for name in ["entries", "locks", "sessions", "staging"] {
        ensure_regular_directory(&root.join(name))?;
    }
    Ok(())
}

fn ensure_regular_directory(path: &Path) -> Result<(), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        ResolvedCacheError::new(format!(
            "required cache directory {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
        || !metadata.is_dir()
    {
        return Err(ResolvedCacheError::new(format!(
            "cache path {} is a symlink or not a directory",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_managed_entry_dir(path: &Path) -> Result<(), ResolvedCacheError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata)
            if metadata.file_type().is_symlink()
                || metadata_is_reparse_point(&metadata)
                || !metadata.is_dir() =>
        {
            Err(ResolvedCacheError::new(format!(
                "cache entry {} is unmanaged",
                path.display()
            )))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(path)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn open_lock_file(path: &Path) -> Result<File, ResolvedCacheError> {
    if let Ok(metadata) = std::fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink()
            || metadata_is_reparse_point(&metadata)
            || !metadata.is_file()
        {
            return Err(ResolvedCacheError::new(format!(
                "cache lock {} is not a regular file",
                path.display()
            )));
        }
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(Into::into)
}

fn is_lock_contended(error: &std::io::Error) -> bool {
    let fs2_contended = fs2::lock_contended_error().raw_os_error();
    error.kind() == std::io::ErrorKind::WouldBlock
        || fs2_contended.is_some() && error.raw_os_error() == fs2_contended
}

/// Everything [`ResolvedModelArtifact::validate`] proves except what needs the bundle to exist.
///
/// Used for exactly one shape: a derived entry still being produced. It must be `Incomplete` /
/// `Missing` — a pending derivative that claims to be complete is the shape a stale journal would
/// have, and admitting it would let an unfinished bundle be treated as loadable.
fn validate_pending_derivative_shape(
    artifact: &ResolvedModelArtifact,
) -> Result<(), ResolvedCacheError> {
    let refuse = |message: &str| ResolvedCacheError::new(message.to_owned());
    if artifact.schema_version != MODEL_ARTIFACT_CONTRACT_VERSION {
        return Err(refuse(
            "pending derivative has an unsupported contract version",
        ));
    }
    artifact
        .identity
        .validate()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    artifact
        .provenance
        .identity
        .validate()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    if artifact.identity != artifact.provenance.identity {
        return Err(refuse("pending derivative identity and provenance differ"));
    }
    if artifact.completeness != ArtifactCompleteness::Incomplete
        || artifact.availability != ArtifactAvailability::Missing
    {
        return Err(refuse(
            "pending derivative must be recorded as incomplete and missing until it publishes",
        ));
    }
    if !matches!(artifact.location, ArtifactLocation::ResolvedLocal { .. }) {
        return Err(refuse("pending derivative is not app-owned"));
    }
    let canonical = ResolvedBundleClosure::new(artifact.closure.members.clone())
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    if canonical != artifact.closure {
        return Err(refuse(
            "pending derivative closure is not in canonical form",
        ));
    }
    Ok(())
}

/// Measure one file a producer staged. Never follows a link and never accepts anything but a
/// regular file, so a producer cannot publish bytes that live outside the staging directory.
fn measure_produced_file(path: &Path) -> Result<(u64, String), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        ResolvedCacheError::new(format!(
            "produced derivative file {} is missing: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
        || !metadata.is_file()
    {
        return Err(ResolvedCacheError::new(format!(
            "produced derivative file {} is linked, reparsed, or not a file",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut size_bytes = 0_u64;
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        size_bytes = size_bytes
            .checked_add(read as u64)
            .ok_or_else(|| ResolvedCacheError::new("produced derivative byte count overflow"))?;
    }
    if size_bytes != metadata.len() {
        return Err(ResolvedCacheError::new(format!(
            "produced derivative file {} changed while it was being measured",
            path.display()
        )));
    }
    Ok((size_bytes, format!("sha256:{:x}", digest.finalize())))
}

/// Refuse a staged tree holding anything the plan did not declare.
fn reject_undeclared_produced_files(
    staging: &Path,
    declared: &BTreeSet<PathBuf>,
) -> Result<(), ResolvedCacheError> {
    let mut stack = vec![staging.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            let relative = path.strip_prefix(staging).map_err(|_| {
                ResolvedCacheError::new("produced derivative file escaped its staging directory")
            })?;
            if !declared.contains(relative) {
                return Err(ResolvedCacheError::new(format!(
                    "produced derivative staged an undeclared file {}",
                    relative.display()
                )));
            }
        }
    }
    Ok(())
}

fn validate_candidate(candidate: &PromotionCandidate) -> Result<(), ResolvedCacheError> {
    candidate
        .artifact
        .validate()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    if !matches!(
        candidate.artifact.location,
        ArtifactLocation::SourceLibrary { .. }
    ) {
        return Err(ResolvedCacheError::new(
            "only a source-library artifact can be reserved for materialization",
        ));
    }
    let key = candidate
        .artifact
        .cache_key()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    if key != candidate.cache_key {
        return Err(ResolvedCacheError::new(
            "promotion candidate cache key does not match its artifact",
        ));
    }
    Ok(())
}

fn validate_metadata_shape_with(
    metadata: &ResolvedCacheMetadata,
    digest: &str,
    verification: ContentVerification,
) -> Result<(), ResolvedCacheError> {
    let artifact_key = metadata
        .artifact
        .cache_key()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    // A derived entry that has not published yet has no bytes anywhere — that is what its producer
    // is running to create, and an interrupted one had its staging tree removed — so there is
    // nothing on disk for `validate` to read. Its identity, provenance agreement and canonical
    // closure are still checked (and so, below, is its cache key), and the pending shape check
    // requires it to say `Incomplete`/`Missing`, so an unfinished bundle can never be mistaken for
    // a loadable one. The exemption is exactly as narrow as the fact: only Derived, and only while
    // the entry is not Complete.
    let is_pending_derivative =
        metadata.production.is_derived() && metadata.state != ResolvedCacheEntryState::Complete;
    if is_pending_derivative {
        validate_pending_derivative_shape(&metadata.artifact)?;
    } else {
        match verification {
            ContentVerification::RehashEveryFile => metadata
                .artifact
                .validate()
                .map_err(|error| ResolvedCacheError::new(error.to_string()))?,
            ContentVerification::PathsAndSizesOnly => {
                artifact_without_content_hashes(&metadata.artifact)
                    .validate()
                    .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            }
        }
    }
    let reservation_shape_is_valid = match metadata.state {
        ResolvedCacheEntryState::Materializing => {
            metadata.reservation_id.is_some()
                && metadata.reservation_owner.is_some()
                && metadata.session_id.is_some()
        }
        _ => {
            metadata.reservation_id.is_none()
                && metadata.reservation_owner.is_none()
                && metadata.session_id.is_none()
        }
    };
    if metadata.schema_version != metadata.production.store_version()
        || metadata.state == ResolvedCacheEntryState::Evicting
        || cache_key_digest(&metadata.cache_key)? != digest
        || artifact_key != metadata.cache_key
        || metadata.entry_relative_path != PathBuf::from("entries").join(digest)
        || metadata.bundle_relative_path != PathBuf::from("entries").join(digest).join("bundle")
        || metadata.effective_pin
            != (metadata.artifact_pinned || !metadata.model_pin_owners.is_empty())
        || !reservation_shape_is_valid
        || metadata
            .reservation_id
            .as_deref()
            .is_some_and(|value| !is_valid_session_id(value))
        || metadata
            .session_id
            .as_deref()
            .is_some_and(|value| !is_valid_session_id(value))
        || metadata
            .reservation_owner
            .as_deref()
            .is_some_and(|owner| validate_model_owner(owner) != Ok(owner))
        || metadata
            .model_pin_owners
            .iter()
            .any(|owner| validate_model_owner(owner) != Ok(owner.as_str()))
    {
        return Err(ResolvedCacheError::new(
            "cache metadata invariant is invalid",
        ));
    }
    Ok(())
}

/// Full-strength validation of a complete entry, with the content re-read gated on a receipt.
///
/// sc-21534 established that this is the ONE boundary that must refuse a content-altered bundle,
/// and it earned that by re-hashing the whole closure. The cost of asking, though, was paid on
/// every acquisition rather than every change: a 19 GB bundle measured ~61 s of SHA-256 between
/// "Worker claimed job" and "Preparing N image(s)", on a bundle whose bytes had not moved since the
/// last job asked the same question and got the same answer.
///
/// So the answer is now recorded. A successful content pass stamps the closure's stat identity
/// (`ClosureFileStat` — bundle-relative path, size, mtime, per file) into the entry's
/// [`ClosureContentReceipt`] sidecar, and a later acquisition that observes the SAME identity is
/// entitled to the recorded verdict without re-reading the bytes. Anything else — a receipt that is
/// absent or unreadable, a file whose size or mtime moved, a closure whose membership changed —
/// falls straight back to the full re-hash and re-stamps.
///
/// What is NOT weakened: the paths-and-sizes pass still runs unconditionally on both branches, so
/// presence, sizes, confinement, closure shape, artifact identity and `verified_bytes` are proven
/// at every acquisition exactly as before. The receipt only ever elides re-reading bytes that no
/// filesystem operation has touched. A write to a bundle file cannot leave both size and mtime
/// unchanged by accident, which is the drift this boundary exists to catch (see `ClosureFileStat`
/// for what this does and does not claim).
///
/// Every caller holds the entry's metadata lock, which is what makes the read-decide-write below a
/// single transition rather than three racing ones. Entries that predate receipts simply have none:
/// their first acquisition after upgrade hashes once, as today, and stamps.
fn validate_complete_metadata(
    store: &ResolvedCacheStore,
    metadata: &ResolvedCacheMetadata,
) -> Result<(), ResolvedCacheError> {
    let digest = cache_key_digest(&metadata.cache_key)?;
    let bundle = store.bundle_path(&metadata.cache_key)?;
    // Observed BEFORE the decision, so the same observation is what the receipt is judged against
    // and what a fresh receipt would record. A stat failure here is not diagnosed: it means the
    // closure is not intact, and `validate_complete_metadata_inner` below says so precisely.
    let before = metadata
        .artifact
        .closure
        .stat_identity_at_root(&bundle)
        .ok();
    let recorded = store.read_content_receipt(&digest, &metadata.cache_key);
    if let (Some(recorded), Some(before)) = (&recorded, &before) {
        if &recorded.files == before {
            return validate_complete_metadata_inner(
                store,
                metadata,
                ContentVerification::PathsAndSizesOnly,
            );
        }
    }

    // Retract first. A verdict that no longer describes the bundle must not survive the pass that
    // is about to re-decide it — least of all if that pass fails and leaves this entry to be read
    // again by something that would have trusted the stale answer.
    store.remove_content_receipt(&digest)?;
    validate_complete_metadata_inner(store, metadata, ContentVerification::RehashEveryFile)?;

    // Re-observe AFTER the read and require both observations to agree before recording a verdict
    // — the same before/after guard `bind_or_probe_validated` uses around a library probe and
    // `trusted_imported_model_hash` uses around a checkpoint hash. Without it a file rewritten
    // mid-pass could have its POST-write stat identity stamped as "these bytes were verified",
    // which is precisely the claim the receipt must never make falsely.
    let Some(before) = before else {
        // The closure hashed clean but could not be stat-ed as a whole. Nothing to record; the next
        // acquisition re-hashes, which is exactly the behaviour before receipts existed.
        return Ok(());
    };
    let after = metadata
        .artifact
        .closure
        .stat_identity_at_root(&bundle)
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    if before != after {
        return Err(ResolvedCacheError::new(
            "cache bundle changed while its contents were being verified",
        ));
    }
    store.write_content_receipt(
        &digest,
        &ClosureContentReceipt {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: metadata.cache_key.clone(),
            verified_at: now_seconds()?,
            files: after,
        },
    )
}

/// How a whole-store listing treats complete entries. `Runtime` is the internal listing used by
/// recovery, staging cleanup and byte accounting; `Inspect` is the read-only status listing. Both
/// validate on paths and sizes only (sc-21534: recovery re-hashed every complete entry in the
/// cache at startup, the same whole-cache cost sc-19712 F-3 removed from job submission; byte
/// accounting needs sizes, and every caller that ACTS on a summary re-proves it under fresh
/// locks). An invalid managed entry degrades to `Corrupt` in either mode so it cannot hide valid
/// entries or stop their recovery/retention; load acquisition still re-hashes every file.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryListing {
    Runtime,
    Inspect,
}

impl EntryListing {
    fn verification(self) -> ContentVerification {
        match self {
            Self::Runtime | Self::Inspect => ContentVerification::PathsAndSizesOnly,
        }
    }
}

/// How thoroughly a complete entry's own bundle bytes are re-verified.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ContentVerification {
    /// Re-reads and re-hashes every bundle file. Required before handing an artifact to a runtime
    /// load, and the cost is proportional to the bundle size.
    RehashEveryFile,
    /// Validates identity, shape, confinement, file presence and sizes, but does not re-hash file
    /// contents. Used by every path that is *reading* rather than loading: journal slot selection,
    /// the status listing, the local-tier scan, and retention. Retention is deciding whether to
    /// *delete* the bundle rather than load it, so the link/escape confinement checks are what
    /// keep a removal inside the managed root; re-hashing gigabytes would hold locks that block
    /// model loads.
    ///
    /// This mode is safe on a read path only because it is never the last word: the load boundary
    /// ([`ResolvedCacheStore::acquire_complete`]) re-verifies at [`Self::RehashEveryFile`] under
    /// the entry's locks before any bytes reach a runtime, and every boundary that stamps
    /// `Complete` content (`record_complete`, `publish_staged`, `recover`'s receipt resurrection)
    /// proves the bytes itself immediately before writing. Do not weaken the load boundary — it
    /// is what this mode leans on.
    PathsAndSizesOnly,
}

fn validate_complete_metadata_inner(
    store: &ResolvedCacheStore,
    metadata: &ResolvedCacheMetadata,
    verification: ContentVerification,
) -> Result<(), ResolvedCacheError> {
    let digest = cache_key_digest(&metadata.cache_key)?;
    // Shape only here: the mode-specific content check is the `match verification` below, so
    // passing `verification` through would hash the whole closure TWICE per full-strength call
    // (sc-21534 — half of the 4x-per-load multiple behind the Krea bf16 4.5-minute verify).
    validate_metadata_shape_with(metadata, &digest, ContentVerification::PathsAndSizesOnly)?;
    if metadata.state != ResolvedCacheEntryState::Complete
        || metadata.reservation_id.is_some()
        || metadata.reservation_owner.is_some()
        || metadata.session_id.is_some()
    {
        return Err(ResolvedCacheError::new("cache entry is not complete"));
    }
    let bundle = store.bundle_path(&metadata.cache_key)?;
    if !matches!(&metadata.artifact.location, ArtifactLocation::ResolvedLocal { root } if root == &bundle)
    {
        return Err(ResolvedCacheError::new(
            "complete cache artifact has the wrong local root",
        ));
    }
    validate_completion_confinement(store, &metadata.cache_key, &metadata.artifact)?;
    // Stripping the recorded hashes makes `validate` check identity, confinement, file presence
    // and sizes while skipping the content re-read; every other invariant is unchanged, and the
    // cache key deliberately excludes post-copy verification enrichment, so it is unaffected.
    match verification {
        ContentVerification::RehashEveryFile => metadata
            .artifact
            .validate()
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?,
        ContentVerification::PathsAndSizesOnly => {
            artifact_without_content_hashes(&metadata.artifact)
                .validate()
                .map_err(|error| ResolvedCacheError::new(error.to_string()))?
        }
    }
    if metadata
        .artifact
        .cache_key()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?
        != metadata.cache_key
        || checked_artifact_bytes(&metadata.artifact)? != metadata.verified_bytes
    {
        return Err(ResolvedCacheError::new(
            "complete cache artifact identity or verified bytes changed",
        ));
    }
    Ok(())
}

// Runs once at the point a whole-store listing judges an entry — after its journal has been read
// and the metadata lock released, with the (possibly full-strength) validation still to come. A
// test observes from here which locks are held, because this is the window in which the
// availability read behind every job submission used to be parked (sc-19712).
#[cfg(test)]
thread_local! {
    static LISTING_VALIDATION_OBSERVER: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_listing_validation_observer(observer: impl FnOnce() + 'static) {
    LISTING_VALIDATION_OBSERVER.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(observer));
    });
}

#[cfg(test)]
fn run_listing_validation_observer() {
    LISTING_VALIDATION_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow_mut().take() {
            observer();
        }
    });
}

fn artifact_without_content_hashes(artifact: &ResolvedModelArtifact) -> ResolvedModelArtifact {
    let mut artifact = artifact.clone();
    for member in &mut artifact.closure.members {
        for file in &mut member.files {
            file.sha256 = None;
        }
    }
    artifact
}

fn validate_completion_confinement(
    store: &ResolvedCacheStore,
    cache_key: &str,
    artifact: &ResolvedModelArtifact,
) -> Result<(), ResolvedCacheError> {
    let canonical_root = std::fs::canonicalize(store.root())?;
    if canonical_root != store.root() {
        return Err(ResolvedCacheError::new(
            "resolved cache root changed after store open",
        ));
    }
    let entries = store.root().join("entries");
    reject_link_or_reparse(&entries, "resolved cache entries directory")?;
    let canonical_entries = std::fs::canonicalize(&entries)?;
    if canonical_entries.parent() != Some(canonical_root.as_path()) {
        return Err(ResolvedCacheError::new(
            "resolved cache entries directory escaped its managed root",
        ));
    }
    let entry = store.entry_path(cache_key)?;
    reject_link_or_reparse(&entry, "resolved cache entry")?;
    let canonical_entry = std::fs::canonicalize(&entry)?;
    if canonical_entry.parent() != Some(canonical_entries.as_path()) {
        return Err(ResolvedCacheError::new(
            "resolved cache entry escaped its managed entries directory",
        ));
    }
    let bundle = entry.join("bundle");
    reject_link_or_reparse(&bundle, "resolved cache bundle")?;
    let canonical_bundle = std::fs::canonicalize(&bundle)?;
    if canonical_bundle.parent() != Some(canonical_entry.as_path())
        || canonical_bundle != canonical_entry.join("bundle")
    {
        return Err(ResolvedCacheError::new(
            "resolved cache bundle escaped its managed entry",
        ));
    }
    for path in artifact
        .closure
        .rebased_paths(artifact.location.root())
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?
    {
        if !path.starts_with(&bundle) {
            return Err(ResolvedCacheError::new(
                "resolved artifact file escaped its lexical bundle",
            ));
        }
        let mut cursor = Some(path.as_path());
        while let Some(current) = cursor {
            if !current.starts_with(&entry) {
                return Err(ResolvedCacheError::new(
                    "resolved artifact ancestor escaped its managed entry",
                ));
            }
            if !std::fs::canonicalize(current)?.starts_with(&canonical_entry) {
                return Err(ResolvedCacheError::new(
                    "resolved artifact ancestor resolves outside its managed entry",
                ));
            }
            if current == entry {
                break;
            }
            cursor = current.parent();
        }
    }
    Ok(())
}

fn validate_staging_confinement(
    reservation: &ResolvedCacheReservation,
) -> Result<(), ResolvedCacheError> {
    let staging_root = reservation.store.root().join("staging");
    reject_link_or_reparse(&staging_root, "resolved cache staging root")?;
    reject_link_or_reparse(
        &reservation.staging_path,
        "resolved cache reservation staging directory",
    )?;
    let canonical_root = std::fs::canonicalize(&staging_root)?;
    let canonical_staging = std::fs::canonicalize(&reservation.staging_path)?;
    if canonical_staging.parent() != Some(canonical_root.as_path()) {
        return Err(ResolvedCacheError::new(
            "materialization staging directory escaped its managed root",
        ));
    }
    validate_regular_tree(&reservation.staging_path)
}

fn validate_regular_tree(path: &Path) -> Result<(), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
        return Err(ResolvedCacheError::new(format!(
            "managed cache path {} is a link or reparse point",
            path.display()
        )));
    }
    if metadata.is_dir() {
        for item in std::fs::read_dir(path)? {
            validate_regular_tree(&item?.path())?;
        }
    } else if !metadata.is_file() {
        return Err(ResolvedCacheError::new(format!(
            "managed cache path {} is not a regular file or directory",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn remove_managed_tree(path: &Path, managed_parent: &Path) -> Result<(), ResolvedCacheError> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;

    ensure_regular_directory(managed_parent)?;
    if path.parent() != Some(managed_parent) {
        return Err(ResolvedCacheError::new(format!(
            "refusing to remove non-child managed path {}",
            path.display()
        )));
    }
    let name = path
        .file_name()
        .ok_or_else(|| ResolvedCacheError::new("managed path has no final name"))?;
    if name.as_bytes().contains(&b'/') {
        return Err(ResolvedCacheError::new(
            "managed path name contains a separator",
        ));
    }
    let parent = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(managed_parent)?;
    remove_entry_at(&parent, name)
}

#[cfg(windows)]
fn remove_managed_tree(path: &Path, managed_parent: &Path) -> Result<(), ResolvedCacheError> {
    use remove_dir_all::RemoveDir;

    let (parent, directory, name) = windows_confined_directory(path, managed_parent)?;
    let mut handle = directory;
    handle
        .remove_dir_contents(Some(path))
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    drop(handle);
    fs_at::OpenOptions::default().rmdir_at(&parent, name)?;
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn remove_managed_tree(_path: &Path, _managed_parent: &Path) -> Result<(), ResolvedCacheError> {
    Err(ResolvedCacheError::new(
        "managed cache cleanup is unsupported on this platform",
    ))
}

#[cfg(windows)]
fn windows_confined_directory(
    path: &Path,
    managed_parent: &Path,
) -> Result<(File, File, std::ffi::OsString), ResolvedCacheError> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

    ensure_regular_directory(managed_parent)?;
    if path.parent() != Some(managed_parent) {
        return Err(ResolvedCacheError::new(format!(
            "managed directory {} is not a direct child of {}",
            path.display(),
            managed_parent.display()
        )));
    }
    let name = path
        .file_name()
        .ok_or_else(|| ResolvedCacheError::new("managed directory has no final name"))?
        .to_owned();
    reject_link_or_reparse(path, "managed cache directory")?;
    #[cfg(test)]
    run_windows_directory_after_validation_hook();
    // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE: this handle must never make a
    // concurrent owner's rename or delete of the managed parent fail while we hold it open.
    let parent = OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001 | 0x0000_0002 | 0x0000_0004)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(managed_parent)?;
    let mut options = fs_at::OpenOptions::default();
    options.read(true).follow(false);
    let directory = options.open_dir_at(&parent, &name)?;
    reject_link_or_reparse(path, "managed cache directory")?;
    let current = options.open_dir_at(&parent, &name)?;
    let directory_information = winapi_util::file::information(&directory)?;
    let current_information = winapi_util::file::information(&current)?;
    if directory_information.volume_serial_number() != current_information.volume_serial_number()
        || directory_information.file_index() != current_information.file_index()
    {
        return Err(ResolvedCacheError::new(format!(
            "managed cache directory {} changed while it was opened",
            path.display()
        )));
    }
    Ok((parent, directory, name))
}

#[cfg(all(test, windows))]
thread_local! {
    static WINDOWS_DIRECTORY_AFTER_VALIDATION_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(all(test, windows))]
fn set_windows_directory_after_validation_hook(hook: impl FnOnce() + 'static) {
    WINDOWS_DIRECTORY_AFTER_VALIDATION_HOOK.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

#[cfg(all(test, windows))]
fn run_windows_directory_after_validation_hook() {
    WINDOWS_DIRECTORY_AFTER_VALIDATION_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(unix)]
fn remove_entry_at(parent: &File, name: &std::ffi::OsStr) -> Result<(), ResolvedCacheError> {
    use rustix::fs::{openat, statat, unlinkat, AtFlags, Dir, FileType, Mode, OFlags};
    use std::os::unix::ffi::OsStrExt;

    let stat = statat(parent, name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    #[cfg(test)]
    run_remove_entry_after_stat_hook();
    let kind = FileType::from_raw_mode(stat.st_mode);
    if kind == FileType::Symlink {
        return Err(ResolvedCacheError::new(
            "refusing to remove linked managed path",
        ));
    }
    if kind == FileType::Directory {
        let descriptor = openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
        let directory = File::from(descriptor);
        let names = Dir::read_from(&directory)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?
            .map(|item| {
                item.map(|item| std::ffi::OsStr::from_bytes(item.file_name().to_bytes()).to_owned())
                    .map_err(|error| ResolvedCacheError::new(error.to_string()))
            })
            .filter(|item| {
                !matches!(item, Ok(name) if name == std::ffi::OsStr::new(".") || name == std::ffi::OsStr::new(".."))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for child in names {
            remove_entry_at(&directory, &child)?;
        }
        unlinkat(parent, name, AtFlags::REMOVEDIR)
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    } else if kind == FileType::RegularFile {
        unlinkat(parent, name, AtFlags::empty())
            .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    } else {
        return Err(ResolvedCacheError::new(
            "refusing to remove unmanaged filesystem entry",
        ));
    }
    Ok(())
}

#[cfg(all(test, unix))]
thread_local! {
    static REMOVE_ENTRY_AFTER_STAT_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(all(test, unix))]
fn set_remove_entry_after_stat_hook(hook: impl FnOnce() + 'static) {
    REMOVE_ENTRY_AFTER_STAT_HOOK.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

#[cfg(all(test, unix))]
fn run_remove_entry_after_stat_hook() {
    REMOVE_ENTRY_AFTER_STAT_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

fn parse_staging_name(name: &str) -> Option<(&str, &str)> {
    let (digest, reservation_id) = name.split_once('-')?;
    if is_lower_hex_64(digest) && is_valid_session_id(reservation_id) {
        Some((digest, reservation_id))
    } else {
        None
    }
}

fn reject_link_or_reparse(path: &Path, label: &str) -> Result<(), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
        return Err(ResolvedCacheError::new(format!(
            "{label} {} is a symlink or reparse point",
            path.display()
        )));
    }
    if !metadata.is_dir() {
        return Err(ResolvedCacheError::new(format!(
            "{label} {} is not a directory",
            path.display()
        )));
    }
    Ok(())
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

fn checked_artifact_bytes(artifact: &ResolvedModelArtifact) -> Result<u64, ResolvedCacheError> {
    artifact
        .closure
        .rebased_paths(artifact.location.root())
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?
        .into_iter()
        .try_fold(0_u64, |total, path| {
            let size = std::fs::metadata(&path)
                .map_err(|error| {
                    ResolvedCacheError::new(format!(
                        "measure resolved artifact file {}: {error}",
                        path.display()
                    ))
                })?
                .len();
            total.checked_add(size).ok_or_else(|| {
                ResolvedCacheError::new("resolved artifact verified byte total overflow")
            })
        })
}

fn validate_model_owner(owner: &str) -> Result<&str, ResolvedCacheError> {
    let owner = owner.trim();
    if owner.is_empty() || owner.len() > 256 || owner.chars().any(char::is_control) {
        Err(ResolvedCacheError::new(
            "model pin owner must be a nonempty stable logical id",
        ))
    } else {
        Ok(owner)
    }
}

fn cache_key_digest(cache_key: &str) -> Result<String, ResolvedCacheError> {
    let digest = cache_key
        .strip_prefix("sha256:")
        .filter(|value| is_lower_hex_64(value))
        .ok_or_else(|| {
            // The caller's own key is unusable — nothing about the store or the host is wrong.
            ResolvedCacheError::request("resolved-cache key must be sha256:<64 lower hex>")
        })?;
    Ok(digest.to_owned())
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_valid_session_id(value: &str) -> bool {
    value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn journal_checksum(
    generation: u64,
    metadata: &ResolvedCacheMetadata,
) -> Result<String, ResolvedCacheError> {
    let mut digest = Sha256::new();
    digest.update(generation.to_le_bytes());
    digest.update(serde_json::to_vec(metadata).map_err(|error| {
        ResolvedCacheError::new(format!("encode cache metadata checksum: {error}"))
    })?);
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn metadata_checksum(metadata: &ResolvedCacheMetadata) -> Result<String, ResolvedCacheError> {
    let bytes = serde_json::to_vec(metadata).map_err(|error| {
        ResolvedCacheError::new(format!("encode complete receipt checksum: {error}"))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

/// The envelope's declared version, read WITHOUT decoding the document it wraps.
///
/// `ResolvedCacheMetadata` is `deny_unknown_fields`, so a journal a future writer widened fails to
/// decode before anything gets to look at its version. Probing the version first turns that into a
/// version statement — the reader can say "this was written by a newer SceneWorks" instead of
/// reporting the store as corrupt, which is the difference between "upgrade" and "your cache is
/// broken". sc-20635's derived entries are the first documents to use it.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct JournalVersionProbe {
    schema_version: u32,
}

fn read_journal(path: &Path) -> Result<JournalEnvelope, ResolvedCacheError> {
    let body = std::fs::read(path)?;
    if let Ok(probe) = serde_json::from_slice::<JournalVersionProbe>(&body) {
        if probe.schema_version > RESOLVED_CACHE_DERIVED_STORE_VERSION {
            return Err(ResolvedCacheError::new(format!(
                "{UNSUPPORTED_JOURNAL_VERSION} (journal version {}, newest supported {})",
                probe.schema_version, RESOLVED_CACHE_DERIVED_STORE_VERSION
            )));
        }
    }
    let envelope: JournalEnvelope = serde_json::from_slice(&body)
        .map_err(|error| ResolvedCacheError::new(format!("decode cache metadata: {error}")))?;
    envelope.validate()?;
    Ok(envelope)
}

fn highest_generation(entry: &Path) -> Result<u64, ResolvedCacheError> {
    let mut highest = 0;
    for slot in 0..=1 {
        let path = entry.join(format!("metadata.{slot}.json"));
        if let Ok(envelope) = read_journal(&path) {
            highest = highest.max(envelope.generation);
        }
    }
    Ok(highest)
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<(), ResolvedCacheError> {
    let body = serde_json::to_vec_pretty(value)
        .map_err(|error| ResolvedCacheError::new(format!("encode cache record: {error}")))?;
    let parent = path
        .parent()
        .ok_or_else(|| ResolvedCacheError::new("cache record has no parent"))?;
    ensure_regular_directory(parent)?;
    let temporary = parent.join(format!(".tmp-{}", random_id()?));
    let result = (|| {
        write_synced_file(&temporary, &body)?;
        std::fs::rename(&temporary, path)?;
        sync_dir(parent)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(|error| {
        ResolvedCacheError::new(format!("write cache record {}: {error}", path.display()))
    })
}

fn write_synced_file(path: &Path, body: &[u8]) -> Result<(), ResolvedCacheError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(body)?;
    file.sync_all()?;
    Ok(())
}

/// Directory-entry durability walk over a fully staged tree, run immediately before
/// publication.
///
/// File *content* durability is deliberately NOT re-established here. `copy_and_verify`
/// already calls `File::sync_all` on the very write handle that produced the verified
/// bytes, which binds the flush to the verified identity by construction — no path
/// reopen, no TOCTOU window. Reopening each staged file by path here (as earlier
/// revisions did) reintroduced exactly that window on Windows: a parent-directory
/// junction swap or delete-and-recreate between validation and the reopen would flush
/// the wrong file while publication proceeded. It also required a write-capable reopen
/// for `FlushFileBuffers`, which failed closed (`ERROR_ACCESS_DENIED`) or handed
/// interference filters (AV/indexers) a fresh dirty-close to scan right before the
/// publish rename. So this walk only:
/// - re-validates that every staged entry is still a regular file or directory, and
/// - on Unix, syncs each directory so the entries naming the already-durable inodes are
///   durable before the rename. Windows has no POSIX directory fsync (`sync_dir` is a
///   no-op there); journaled NTFS metadata plus the write-through publish rename cover
///   that side.
fn sync_tree(path: &Path) -> Result<(), ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        ResolvedCacheError::new(format!("inspect staged path {}: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
        return Err(ResolvedCacheError::new(format!(
            "cannot sync linked managed path {}",
            path.display()
        )));
    }
    if metadata.is_dir() {
        let items = std::fs::read_dir(path).map_err(|error| {
            ResolvedCacheError::new(format!(
                "enumerate staged directory {}: {error}",
                path.display()
            ))
        })?;
        for item in items {
            let item = item.map_err(|error| {
                ResolvedCacheError::new(format!(
                    "enumerate staged directory {}: {error}",
                    path.display()
                ))
            })?;
            sync_tree(&item.path())?;
        }
        sync_dir(path).map_err(|error| {
            ResolvedCacheError::new(format!("sync staged directory {}: {error}", path.display()))
        })?;
    } else if !metadata.is_file() {
        return Err(ResolvedCacheError::new(format!(
            "cannot sync unmanaged filesystem entry {}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> Result<(), ResolvedCacheError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> Result<(), ResolvedCacheError> {
    Ok(())
}

/// Atomically renames the validated staging tree onto its bundle path.
#[cfg(not(windows))]
fn publish_rename(staging: &Path, bundle: &Path) -> Result<(), ResolvedCacheError> {
    std::fs::rename(staging, bundle).map_err(|error| publish_rename_error(staging, bundle, &error))
}

/// Atomically renames the validated staging tree onto its bundle path.
///
/// Windows refuses to rename a directory while ANY handle is open on it or a descendant —
/// including handles the process never opened: antivirus and search-indexer filters briefly open
/// freshly written files after their write handles close. Those transient
/// `ERROR_ACCESS_DENIED`/`ERROR_SHARING_VIOLATION` holds are retried with a bounded backoff
/// before failing closed with the operation and both paths named. Durability of the rename
/// record itself relies on journaled NTFS metadata: Windows has no POSIX directory fsync, std
/// `fs::rename` does not expose `MOVEFILE_WRITE_THROUGH`, and this workspace forbids the direct
/// `unsafe` FFI that reaching it would need.
#[cfg(windows)]
fn publish_rename(staging: &Path, bundle: &Path) -> Result<(), ResolvedCacheError> {
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ATTEMPTS: u32 = 8;

    let mut delay = std::time::Duration::from_millis(30);
    let mut attempt = 0;
    loop {
        attempt += 1;
        match std::fs::rename(staging, bundle) {
            Ok(()) => return Ok(()),
            Err(error)
                if attempt < ATTEMPTS
                    && matches!(
                        error.raw_os_error(),
                        Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION)
                    ) =>
            {
                std::thread::sleep(delay);
                delay = delay.saturating_mul(2);
            }
            Err(error) => return Err(publish_rename_error(staging, bundle, &error)),
        }
    }
}

fn publish_rename_error(
    staging: &Path,
    bundle: &Path,
    error: &std::io::Error,
) -> ResolvedCacheError {
    ResolvedCacheError::new(format!(
        "publish staged bundle {} -> {}: {error}",
        staging.display(),
        bundle.display()
    ))
}

fn random_id() -> Result<String, ResolvedCacheError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|error| ResolvedCacheError::new(format!("generate cache id: {error}")))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn now_seconds() -> Result<u64, ResolvedCacheError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| ResolvedCacheError::new(format!("system time before epoch: {error}")))
}

fn relation_for_volume_identities(
    source: Option<u64>,
    resolved: Option<u64>,
    source_exists: bool,
) -> SourceVolumeRelation {
    if !source_exists {
        SourceVolumeRelation::Unavailable
    } else {
        match (source, resolved) {
            (Some(source), Some(resolved)) if source == resolved => SourceVolumeRelation::Same,
            (Some(_), Some(_)) => SourceVolumeRelation::Different,
            _ => SourceVolumeRelation::Unknown,
        }
    }
}

#[cfg(unix)]
fn volume_identity(path: &Path) -> Result<u64, ResolvedCacheError> {
    use std::os::unix::fs::MetadataExt;
    Ok(std::fs::metadata(path)?.dev())
}

#[cfg(windows)]
fn volume_identity(path: &Path) -> Result<u64, ResolvedCacheError> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    Ok(winapi_util::file::information(&file)
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?
        .volume_serial_number())
}

#[cfg(not(any(unix, windows)))]
fn volume_identity(_path: &Path) -> Result<u64, ResolvedCacheError> {
    Err(ResolvedCacheError::new(
        "volume identity is unavailable on this platform",
    ))
}

#[cfg(test)]
#[path = "resolved_cache/tests.rs"]
mod tests;
