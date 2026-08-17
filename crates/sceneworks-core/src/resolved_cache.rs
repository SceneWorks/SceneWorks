//! Durable app-owned resolved-model cache policy and store.
//!
//! The store is deliberately separate from the authoritative source library. It owns only
//! `<data_dir>/models/resolved`, refuses to adopt an unmarked directory, and names entries from
//! the portable immutable cache key defined by [`crate::model_artifacts`].

use crate::model_artifacts::{
    ActiveArtifactLease, ArtifactLocation, ModelArtifactResolver, PromotionCandidate,
    ResolvedModelArtifact,
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

const STORE_MARKER: &str = ".sceneworks-resolved-cache-v1";
const STORE_MARKER_BODY: &[u8] = b"sceneworks-resolved-cache\nschema=1\n";

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCacheError(String);

impl ResolvedCacheError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ResolvedCacheError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ResolvedCacheError {}

impl From<std::io::Error> for ResolvedCacheError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolvedCacheMetadata {
    pub schema_version: u32,
    pub cache_key: String,
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

impl ResolvedCacheMetadata {
    fn refresh_effective_pin(&mut self) {
        self.effective_pin = self.artifact_pinned || !self.model_pin_owners.is_empty();
    }
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
    pub fn enumerate_existing(
        data_dir: &Path,
    ) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        let root = data_dir.join("models").join("resolved");
        if !root.exists() {
            return Ok(Vec::new());
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
        Self {
            inner: Arc::new(StoreInner {
                root,
                session_id: "read-only-inspection".to_owned(),
                _session_lock: None,
            }),
        }
        .enumerate()
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
        let digest = cache_key_digest(&candidate.cache_key)?;
        let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => {}
            Err(error) if is_lock_contended(&error) => {
                return Ok(ReservationOutcome::Contended);
            }
            Err(error) => return Err(error.into()),
        }
        let entry = self.entry_path(&candidate.cache_key)?;
        ensure_managed_entry_dir(&entry)?;
        let _metadata_lock = self.lock_metadata(&digest)?;
        let existing = match self.read_metadata_unlocked(&digest)? {
            JournalRead::Missing => None,
            JournalRead::Valid { metadata, .. } => Some(*metadata),
        };
        if let Some(mut metadata) = existing {
            match metadata.state {
                ResolvedCacheEntryState::Complete => {
                    validate_complete_metadata(self, &metadata)?;
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
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: candidate.cache_key.clone(),
            artifact: candidate.artifact.clone(),
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
        if let Ok(existing) = self.read_metadata_locked(&digest) {
            metadata.created_at = existing.created_at;
            metadata.artifact_pinned = existing.artifact_pinned;
            metadata.model_pin_owners = existing.model_pin_owners;
            metadata.refresh_effective_pin();
        }
        self.write_metadata_unlocked(&digest, &metadata)?;
        let record = SessionRecord {
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            cache_key: candidate.cache_key.clone(),
            operation_id: reservation_id.clone(),
            model_owner: logical_model_owner.clone(),
            kind: SessionRecordKind::Reservation,
            acquired_at: now,
        };
        let record_path = self.write_session_record(&record)?;
        Ok(ReservationOutcome::Acquired(Box::new(
            ResolvedCacheReservation {
                store: self.clone(),
                cache_key: candidate.cache_key.clone(),
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
                validate_complete_metadata(self, &metadata)?;
                Some(*metadata)
            }
            _ => None,
        };
        drop(metadata_lock);
        Ok(result)
    }

    pub fn enumerate(&self) -> Result<Vec<ResolvedCacheEntrySummary>, ResolvedCacheError> {
        let mut entries = Vec::new();
        for item in std::fs::read_dir(self.inner.root.join("entries"))? {
            let item = item?;
            if !item.file_type()?.is_dir() {
                return Err(ResolvedCacheError::new(format!(
                    "unmanaged resolved-cache entry {}",
                    item.path().display()
                )));
            }
            let digest = item
                .file_name()
                .to_str()
                .filter(|value| is_lower_hex_64(value))
                .ok_or_else(|| ResolvedCacheError::new("invalid resolved-cache entry name"))?
                .to_owned();
            let metadata_lock = self.lock_metadata(&digest)?;
            let summary = match self.read_metadata_unlocked(&digest) {
                Ok(JournalRead::Valid { metadata, .. }) => {
                    if metadata.state == ResolvedCacheEntryState::Complete {
                        validate_complete_metadata(self, &metadata)?;
                    }
                    ResolvedCacheEntrySummary {
                        cache_key: metadata.cache_key.clone(),
                        state: metadata.state.clone(),
                        metadata: Some(*metadata),
                    }
                }
                Ok(JournalRead::Missing) => ResolvedCacheEntrySummary {
                    cache_key: format!("sha256:{digest}"),
                    state: ResolvedCacheEntryState::Corrupt,
                    metadata: None,
                },
                Err(_) => ResolvedCacheEntrySummary {
                    cache_key: format!("sha256:{digest}"),
                    state: ResolvedCacheEntryState::Corrupt,
                    metadata: None,
                },
            };
            drop(metadata_lock);
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
        validate_complete_metadata(self, &metadata)?;
        let artifact = Arc::new(metadata.artifact.clone());
        let runtime_lease = resolver
            .acquire_runtime_lease(&artifact)
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
                    if had_invalid_slot {
                        metadata.recovery_status = RecoveryStatus::RecoveredFromOlderSlot;
                        metadata.artifact_pinned = true;
                        metadata.refresh_effective_pin();
                        changed = true;
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
                Ok(JournalRead::Missing) => self.write_corrupt_marker(&digest)?,
                Err(_) => {
                    let receipt = self.read_complete_receipt(&digest).and_then(|metadata| {
                        validate_complete_metadata(self, &metadata)?;
                        Ok(metadata)
                    });
                    if let Ok(mut metadata) = receipt {
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
        self.clean_stale_sessions()?;
        self.enumerate()
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
            JournalRead::Missing => Err(ResolvedCacheError::new("cache metadata is missing")),
        }
    }

    fn read_metadata_unlocked(&self, digest: &str) -> Result<JournalRead, ResolvedCacheError> {
        let entry = self.inner.root.join("entries").join(digest);
        let mut valid = Vec::new();
        let mut had_file = false;
        let mut had_invalid_slot = false;
        for slot in 0..=1 {
            let path = entry.join(format!("metadata.{slot}.json"));
            if !path.exists() {
                continue;
            }
            had_file = true;
            match read_journal(&path) {
                Ok(envelope) if validate_metadata_shape(&envelope.metadata, digest).is_ok() => {
                    valid.push(envelope)
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
        if had_file {
            Err(ResolvedCacheError::new(
                "both cache metadata slots are corrupt",
            ))
        } else {
            Ok(JournalRead::Missing)
        }
    }

    fn write_metadata_unlocked(
        &self,
        digest: &str,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<(), ResolvedCacheError> {
        validate_metadata_shape(metadata, digest)?;
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
        if let Ok(_metadata_lock) = self.store.lock_metadata(&self.digest) {
            if let Ok(mut metadata) = self.store.read_metadata_locked(&self.digest) {
                if self.verify_ownership(&metadata).is_ok() {
                    metadata.state = ResolvedCacheEntryState::Interrupted;
                    metadata.reservation_id = None;
                    metadata.reservation_owner = None;
                    metadata.session_id = None;
                    metadata.recovery_status = RecoveryStatus::InterruptedReservation;
                    metadata.updated_at = now_seconds().unwrap_or(metadata.updated_at);
                    let _ = self.store.write_metadata_unlocked(&self.digest, &metadata);
                }
            }
        }
        let _ = std::fs::remove_file(&self.record_path);
        self.artifact_lock.take();
    }
}

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
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            generation,
            checksum,
            metadata,
        })
    }

    fn validate(&self) -> Result<(), ResolvedCacheError> {
        if self.schema_version != RESOLVED_CACHE_STORE_VERSION
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
            schema_version: RESOLVED_CACHE_STORE_VERSION,
            checksum,
            metadata,
        })
    }

    fn validate(&self) -> Result<(), ResolvedCacheError> {
        if self.schema_version != RESOLVED_CACHE_STORE_VERSION
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
    Valid {
        metadata: Box<ResolvedCacheMetadata>,
        had_invalid_slot: bool,
    },
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

fn validate_metadata_shape(
    metadata: &ResolvedCacheMetadata,
    digest: &str,
) -> Result<(), ResolvedCacheError> {
    let artifact_key = metadata
        .artifact
        .cache_key()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
    metadata
        .artifact
        .validate()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
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
    if metadata.schema_version != RESOLVED_CACHE_STORE_VERSION
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

fn validate_complete_metadata(
    store: &ResolvedCacheStore,
    metadata: &ResolvedCacheMetadata,
) -> Result<(), ResolvedCacheError> {
    let digest = cache_key_digest(&metadata.cache_key)?;
    validate_metadata_shape(metadata, &digest)?;
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
    metadata
        .artifact
        .validate()
        .map_err(|error| ResolvedCacheError::new(error.to_string()))?;
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
            let size = std::fs::metadata(&path)?.len();
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
            ResolvedCacheError::new("resolved-cache key must be sha256:<64 lower hex>")
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

fn read_journal(path: &Path) -> Result<JournalEnvelope, ResolvedCacheError> {
    let body = std::fs::read(path)?;
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
    result
}

fn write_synced_file(path: &Path, body: &[u8]) -> Result<(), ResolvedCacheError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(body)?;
    file.sync_all()?;
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
