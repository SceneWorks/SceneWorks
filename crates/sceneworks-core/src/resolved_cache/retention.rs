//! Retention enforcement, LRU eviction, manual removal, and source-lifecycle reconciliation for
//! the resolved cache.
//!
//! Every removal in this module is two-phase: a checksummed eviction tombstone is written under
//! the exclusive artifact lock and the metadata lock, an audit record is persisted outside the
//! entry, and only then is the entry directory removed with the confined deleter. Interruption at
//! any point converges on the next pass because every reader treats a valid tombstone as "already
//! gone" and recovery finishes the deletion.
//!
//! Automatic eviction fails safe on any doubt: an entry is removed only when, under the locks, it
//! is `Complete`, unpinned, not leased, revalidated on disk, unchanged since it was scanned, and
//! its authoritative source has been re-verified file-by-file (sizes and, where enriched, sha256
//! hashes) as a complete second copy. Everything else is retained and reported with a reason.

use super::*;
use crate::model_artifacts::ArtifactSourceLibrary;

/// Why an entry was (or is being) removed. Persisted inside eviction tombstones and audit
/// records.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionCause {
    TtlExpired,
    SizePressure,
    ManualRemove,
    SourceRemoved,
}

/// Why automatic cleanup retained an entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetentionHold {
    /// A runtime lease or another session's reservation holds the artifact lock.
    ActiveUse,
    /// The artifact or one of its model owners pinned the entry.
    Pinned,
    /// A live session is materializing this entry right now.
    MaterializationInProgress,
    /// Pending, interrupted, stale-materializing, or unreadable entries belong to recovery, never
    /// to eviction.
    RecoveryCandidate,
    /// The authoritative source could not be re-verified as a complete, reachable second copy;
    /// deleting the resolved bundle could destroy the last copy.
    SourceUnverified,
    /// Inside the inactivity TTL and not needed for size pressure.
    Fresh,
}

impl std::fmt::Display for RetentionHold {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ActiveUse => "active lease or reservation",
            Self::Pinned => "pinned",
            Self::MaterializationInProgress => "materialization in progress",
            Self::RecoveryCandidate => "incomplete recovery candidate",
            Self::SourceUnverified => "authoritative source unverified",
            Self::Fresh => "within retention policy",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvictedRecord {
    pub cache_key: String,
    pub bytes: u64,
    pub cause: EvictionCause,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedRecord {
    pub cache_key: String,
    pub bytes: u64,
    pub hold: RetentionHold,
    pub detail: Option<String>,
}

/// Outcome of one retention pass. `retained` explains, per entry, why space could not be
/// reclaimed; `limit_satisfied` is false when protected entries kept the store above the
/// configured size limit.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RetentionReport {
    pub evicted: Vec<EvictedRecord>,
    pub retained: Vec<RetainedRecord>,
    /// Entries whose removal failed mid-flight (for example an open file handle on Windows).
    /// Their tombstones persist, so a later pass or recovery converges.
    pub failed: Vec<(String, String)>,
    pub complete_bytes_before: u64,
    pub complete_bytes_after: u64,
    pub limit_satisfied: bool,
}

/// Pre-removal report for one entry: exact reclaimable bytes, what currently blocks removal, and
/// whether removal would make the model unavailable until its external source returns.
/// Pin state as far as the preview could establish it. `Unknown` is deliberately not collapsed
/// into "not pinned": a caller that cannot distinguish them would show a pinned entry as freely
/// removable whenever a metadata read fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManualRemovalPins {
    Known {
        artifact_pinned: bool,
        owners: Vec<String>,
    },
    Unknown,
}

impl ManualRemovalPins {
    /// True only when the entry is provably unpinned.
    pub fn is_provably_unpinned(&self) -> bool {
        matches!(
            self,
            Self::Known {
                artifact_pinned: false,
                owners,
            } if owners.is_empty()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManualRemovalPreview {
    pub cache_key: String,
    pub state: ResolvedCacheEntryState,
    pub reclaimable_bytes: u64,
    pub pins: ManualRemovalPins,
    /// `Some` when the authoritative source is currently unreachable or incomplete, so removing
    /// this entry leaves the model unavailable until the source returns.
    pub source_unavailable_warning: Option<String>,
    /// `Some` when `remove_entry` would currently refuse, with the refusal reason.
    pub blocked: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManualRemovalOutcome {
    pub cache_key: String,
    pub reclaimed_bytes: u64,
    /// Carried over from the preview probe taken under the removal locks: the model is now
    /// unavailable until its external source returns.
    pub source_unavailable_warning: Option<String>,
}

/// Selects resolved-cache entries by the primary artifact identity of their provenance. `None`
/// fields match everything, so `{ repository, None, None }` reconciles a full model uninstall,
/// adding `revision` reconciles a revision replacement, and adding `tier` reconciles a
/// single-tier deletion. Entries that merely share a component with the selected model are not
/// selected: their self-contained bundles stay valid until their own primary is removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLifecycleSelector {
    pub repository: String,
    pub revision: Option<String>,
    pub tier: Option<String>,
}

/// Outcome of reconciling an install/uninstall lifecycle event with the resolved cache.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ReconciliationReport {
    pub removed: Vec<EvictedRecord>,
    /// Matching entries that could not be removed now (active lease or live materialization).
    /// Callers re-run reconciliation to converge; nothing is stranded silently.
    pub deferred: Vec<RetainedRecord>,
    /// Entries whose metadata is unreadable, so lifecycle matching was impossible. They are
    /// surfaced here (and in `enumerate`) instead of being silently skipped.
    /// Digest-derived keys of entries whose metadata could not be read, so lifecycle matching was
    /// impossible. Named rather than merely counted so a caller can log, surface and act on the
    /// specific stranded entries instead of just knowing that some exist.
    pub unmatched_unreadable: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetentionCheckpointOutcome {
    NotIdle,
    Disabled,
    Ran(RetentionReport),
}

/// Caller-driven retention checkpoints. The caller owns the thread or blocking-task lifetime,
/// exactly like [`ResolvedCachePromotionScheduler`]: nothing here detaches work, and every
/// artifact lock is taken non-blocking, so model loads are never blocked by a sweep.
#[derive(Clone, Debug)]
pub struct ResolvedCacheRetention {
    store: ResolvedCacheStore,
    policy: ResolvedCachePolicy,
}

impl ResolvedCacheRetention {
    pub fn new(
        store: ResolvedCacheStore,
        policy: ResolvedCachePolicy,
    ) -> Result<Self, ResolvedCacheError> {
        policy.validate()?;
        Ok(Self { store, policy })
    }

    pub fn store(&self) -> &ResolvedCacheStore {
        &self.store
    }

    pub fn policy(&self) -> &ResolvedCachePolicy {
        &self.policy
    }

    /// Startup checkpoint: recover the store (which also finishes interrupted evictions), then
    /// enforce retention when the cache is enabled.
    pub fn run_after_recovery(
        &self,
        now: u64,
    ) -> Result<RetentionCheckpointOutcome, ResolvedCacheError> {
        self.store.recover()?;
        if !self.policy.enabled {
            return Ok(RetentionCheckpointOutcome::Disabled);
        }
        Ok(RetentionCheckpointOutcome::Ran(
            self.store.enforce_retention(&self.policy, now)?,
        ))
    }

    /// Idle checkpoint: run one retention pass only when the caller proves its worker is idle.
    pub fn run_if_idle(
        &self,
        idle: bool,
        now: u64,
    ) -> Result<RetentionCheckpointOutcome, ResolvedCacheError> {
        if !self.policy.enabled {
            return Ok(RetentionCheckpointOutcome::Disabled);
        }
        if !idle {
            return Ok(RetentionCheckpointOutcome::NotIdle);
        }
        Ok(RetentionCheckpointOutcome::Ran(
            self.store.enforce_retention(&self.policy, now)?,
        ))
    }
}

/// One scanned eviction candidate. `activity` is the deterministic LRU key: the last runtime use,
/// or the entry's creation when it was never used.
#[derive(Clone, Debug)]
struct EvictionCandidate {
    cache_key: String,
    bytes: u64,
    activity: u64,
    created_at: u64,
}

enum EvictAttempt {
    Evicted(EvictedRecord),
    Retained(RetainedRecord),
    AlreadyGone,
    Failed(String),
}

impl ResolvedCacheStore {
    /// Enforces the inactivity TTL and size limit with deterministic LRU ordering under the
    /// caller-supplied clock. Never blocks on artifact locks and never removes an entry that any
    /// protection covers; the report explains every retained entry.
    ///
    /// Clock sensitivity is deliberate and bounded. `now` is caller-supplied wall-clock seconds, so
    /// a large *forward* jump (a corrected clock, a restored VM snapshot, a bad NTP step) makes
    /// every entry look inactive at once and can expire the whole eligible set in a single pass.
    /// That is bounded by the protections rather than by the clock: pinned, leased, in-flight and
    /// recovery entries are still retained, and each eviction still has to prove the authoritative
    /// source is a complete second copy, so a spurious sweep costs re-materialization time, never
    /// data. A *backward* jump fails safe on its own — `saturating_sub` yields zero elapsed time,
    /// so nothing expires.
    pub fn enforce_retention(
        &self,
        policy: &ResolvedCachePolicy,
        now: u64,
    ) -> Result<RetentionReport, ResolvedCacheError> {
        policy.validate()?;
        let mut report = RetentionReport::default();
        let mut candidates = Vec::new();
        for digest in self.entry_digests()? {
            self.scan_entry(&digest, &mut report, &mut candidates)?;
        }
        let mut complete_total = candidates
            .iter()
            .map(|candidate| candidate.bytes)
            .chain(report.retained.iter().map(|retained| retained.bytes))
            .try_fold(0_u64, |total, bytes| {
                total.checked_add(bytes).ok_or_else(|| {
                    ResolvedCacheError::new("resolved-cache retention byte total overflow")
                })
            })?;
        report.complete_bytes_before = complete_total;
        candidates.sort_by(|left, right| {
            (left.activity, left.created_at, left.cache_key.as_str()).cmp(&(
                right.activity,
                right.created_at,
                right.cache_key.as_str(),
            ))
        });
        let mut survivors = Vec::new();
        for candidate in candidates {
            let expired = now.saturating_sub(candidate.activity) >= policy.inactivity_seconds;
            if expired {
                self.apply_eviction_attempt(
                    self.evict_candidate(&candidate, EvictionCause::TtlExpired, now),
                    &candidate,
                    &mut report,
                    &mut complete_total,
                );
            } else {
                survivors.push(candidate);
            }
        }
        for candidate in survivors {
            if complete_total <= policy.max_bytes {
                report.retained.push(RetainedRecord {
                    cache_key: candidate.cache_key,
                    bytes: candidate.bytes,
                    hold: RetentionHold::Fresh,
                    detail: None,
                });
                continue;
            }
            self.apply_eviction_attempt(
                self.evict_candidate(&candidate, EvictionCause::SizePressure, now),
                &candidate,
                &mut report,
                &mut complete_total,
            );
        }
        report.complete_bytes_after = complete_total;
        report.limit_satisfied = complete_total <= policy.max_bytes;
        Ok(report)
    }

    /// Reports what removing one entry would reclaim and whether removal is currently possible or
    /// advisable. Never mutates anything.
    pub fn manual_removal_preview(
        &self,
        cache_key: &str,
    ) -> Result<ManualRemovalPreview, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        let _metadata_lock = self.lock_metadata(&digest)?;
        let entry = self.inner.root.join("entries").join(&digest);
        if std::fs::symlink_metadata(&entry).is_err() {
            return Err(ResolvedCacheError::new("no such resolved-cache entry"));
        }
        // A preview must stay renderable even for an entry that cannot be walked (for example one
        // holding a link, which the confined deleter would also refuse). Measurement failure
        // degrades to an unknown size and becomes a refusal reason rather than an error.
        let (reclaimable_bytes, unmeasurable) = match measure_entry_bytes(&entry) {
            Ok(bytes) => (bytes, None),
            Err(error) => (
                0,
                Some(format!("entry cannot be measured for removal: {error}")),
            ),
        };
        let active = self.artifact_lock_is_contended(&digest)?;
        let active_block =
            || active.then(|| "an active lease or reservation holds this entry".to_owned());
        let (state, pins, source_warning, blocked) = match self.read_metadata_unlocked(&digest) {
            Ok(JournalRead::Valid { metadata, .. }) => {
                let live_materializing = metadata.state == ResolvedCacheEntryState::Materializing
                    && self.materializing_session_is_live(&metadata)?;
                let warning = if metadata.state == ResolvedCacheEntryState::Complete {
                    self.probe_source_reachable(&metadata).err()
                } else {
                    None
                };
                let blocked = if active {
                    active_block()
                } else if live_materializing {
                    Some("a live session is materializing this entry".to_owned())
                } else if metadata.effective_pin {
                    Some("entry is pinned; unpin it before removal".to_owned())
                } else {
                    None
                };
                (
                    metadata.state.clone(),
                    ManualRemovalPins::Known {
                        artifact_pinned: metadata.artifact_pinned,
                        owners: metadata.model_pin_owners.iter().cloned().collect(),
                    },
                    warning,
                    blocked,
                )
            }
            // A tombstoned entry is already gone as far as pins are concerned: the removal was
            // authorized before the tombstone was written.
            Ok(JournalRead::Evicted { .. }) => (
                ResolvedCacheEntryState::Evicting,
                ManualRemovalPins::Known {
                    artifact_pinned: false,
                    owners: Vec::new(),
                },
                None,
                active_block(),
            ),
            // No journal, or both slots proven corrupt: no recoverable state can hold a pin.
            Ok(JournalRead::Missing) => (
                ResolvedCacheEntryState::Corrupt,
                ManualRemovalPins::Known {
                    artifact_pinned: false,
                    owners: Vec::new(),
                },
                None,
                active_block(),
            ),
            Err(error) if error.is_unrecoverable_metadata() => (
                ResolvedCacheEntryState::Corrupt,
                ManualRemovalPins::Known {
                    artifact_pinned: false,
                    owners: Vec::new(),
                },
                None,
                active_block(),
            ),
            // The state could not be read, so the pins are genuinely unknown. Reporting "not
            // pinned" here would present a pinned entry to the UI as freely removable, so the
            // preview reports unknown and blocks, matching what `remove_entry` will do.
            Err(error) => (
                ResolvedCacheEntryState::Corrupt,
                ManualRemovalPins::Unknown,
                None,
                Some(format!(
                    "cannot remove a resolved-cache entry whose state cannot be read: {error}"
                )),
            ),
        };
        Ok(ManualRemovalPreview {
            cache_key: cache_key.to_owned(),
            state,
            reclaimable_bytes,
            pins,
            source_unavailable_warning: source_warning,
            blocked: blocked.or(unmeasurable),
        })
    }

    /// Explicit manual removal. Refuses active leases, live materializations, and pinned entries
    /// (unpin first); removes everything else, including entries whose source is currently
    /// unavailable — the returned warning tells the caller the model stays unavailable until the
    /// external source returns.
    pub fn remove_entry(
        &self,
        cache_key: &str,
        now: u64,
    ) -> Result<ManualRemovalOutcome, ResolvedCacheError> {
        let digest = cache_key_digest(cache_key)?;
        let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => {}
            Err(error) if is_lock_contended(&error) => {
                return Err(ResolvedCacheError::new(
                    "cannot remove a resolved-cache entry with an active lease or reservation",
                ));
            }
            Err(error) => return Err(error.into()),
        }
        let _metadata_lock = self.lock_metadata(&digest)?;
        let entry = self.inner.root.join("entries").join(&digest);
        if std::fs::symlink_metadata(&entry).is_err() {
            return Err(ResolvedCacheError::new("no such resolved-cache entry"));
        }
        let mut warning = None;
        match self.read_metadata_unlocked(&digest) {
            Ok(JournalRead::Evicted { .. }) => {
                let marker = self.finish_pending_eviction(&digest)?;
                return Ok(ManualRemovalOutcome {
                    cache_key: cache_key.to_owned(),
                    reclaimed_bytes: marker.reclaimable_bytes,
                    source_unavailable_warning: None,
                });
            }
            Ok(JournalRead::Valid { metadata, .. }) => {
                if metadata.state == ResolvedCacheEntryState::Materializing
                    && self.materializing_session_is_live(&metadata)?
                {
                    return Err(ResolvedCacheError::new(
                        "cannot remove a resolved-cache entry that a live session is materializing",
                    ));
                }
                if metadata.effective_pin {
                    return Err(ResolvedCacheError::new(
                        "resolved-cache entry is pinned; unpin it before removal",
                    ));
                }
                if metadata.state == ResolvedCacheEntryState::Complete {
                    warning = self.probe_source_reachable(&metadata).err();
                }
            }
            // No journal at all, or both slots proven corrupt: there is no recoverable state left
            // that could carry a pin or a live materialization, and manual removal is exactly how a
            // user clears that residue after recovery has given up on it.
            Ok(JournalRead::Missing) => {}
            Err(error) if error.is_unrecoverable_metadata() => {}
            // Every other read failure — transient IO, descriptor pressure, or the fail-closed
            // refusal of a linked/non-regular tombstone — leaves the pin and materialization state
            // unknown. Removing here could delete a pinned entry, so it refuses instead.
            Err(error) => {
                return Err(ResolvedCacheError::new(format!(
                    "cannot remove a resolved-cache entry whose state cannot be read: {error}"
                )));
            }
        }
        let reclaimed_bytes = measure_entry_bytes(&entry)?;
        self.write_eviction_marker(
            &digest,
            &EvictionMarker {
                schema_version: RESOLVED_CACHE_STORE_VERSION,
                cache_key: cache_key.to_owned(),
                cause: EvictionCause::ManualRemove,
                reclaimable_bytes: reclaimed_bytes,
                requested_at: now,
                session_id: self.inner.session_id.clone(),
            },
        )?;
        self.finish_pending_eviction(&digest)?;
        Ok(ManualRemovalOutcome {
            cache_key: cache_key.to_owned(),
            reclaimed_bytes,
            source_unavailable_warning: warning,
        })
    }

    /// Reconciles a source-library lifecycle event (full uninstall, single-tier deletion,
    /// revision replacement) with the resolved cache: every entry whose primary provenance matches
    /// the selector is removed, while active leases and live materializations are deferred and
    /// reported for a later pass. Entries that only share components with the selected model are
    /// untouched and stay valid, because their bundles are self-contained.
    ///
    /// **Pinned entries are removed here.** This is the one deliberate carve-out from the
    /// otherwise absolute never-remove-pinned invariant, and it is not an oversight: a pin means
    /// "keep this local copy of this model", but reconciliation only runs because the model itself
    /// was explicitly uninstalled. Honoring the pin would keep bytes for a model the user removed
    /// and that nothing can load — stranded state, not a kept promise. Automatic retention
    /// (TTL/LRU/size) never does this; only an explicit lifecycle event does.
    pub fn reconcile_removed_source(
        &self,
        selector: &SourceLifecycleSelector,
        now: u64,
    ) -> Result<ReconciliationReport, ResolvedCacheError> {
        if selector.repository.trim().is_empty() {
            return Err(ResolvedCacheError::new(
                "lifecycle selector repository must be nonempty",
            ));
        }
        let mut report = ReconciliationReport::default();
        for digest in self.entry_digests()? {
            let metadata = {
                let _metadata_lock = self.lock_metadata(&digest)?;
                match self.read_metadata_unlocked(&digest) {
                    Ok(JournalRead::Valid { metadata, .. }) => *metadata,
                    // Pending evictions already have their own audited convergence.
                    Ok(JournalRead::Evicted { .. }) | Ok(JournalRead::Missing) => continue,
                    Err(_) => {
                        report.unmatched_unreadable.push(format!("sha256:{digest}"));
                        continue;
                    }
                }
            };
            if !selector_matches(selector, &metadata) {
                continue;
            }
            let entry = self.inner.root.join("entries").join(&digest);
            let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
            match FileExt::try_lock_exclusive(&artifact_lock) {
                Ok(()) => {}
                Err(error) if is_lock_contended(&error) => {
                    report.deferred.push(RetainedRecord {
                        cache_key: metadata.cache_key.clone(),
                        bytes: entry_bytes(&entry, &metadata),
                        hold: RetentionHold::ActiveUse,
                        detail: None,
                    });
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
            let _metadata_lock = self.lock_metadata(&digest)?;
            let metadata = match self.read_metadata_unlocked(&digest) {
                Ok(JournalRead::Valid { metadata, .. })
                    if selector_matches(selector, &metadata) =>
                {
                    *metadata
                }
                _ => continue,
            };
            if metadata.state == ResolvedCacheEntryState::Materializing
                && self.materializing_session_is_live(&metadata)?
            {
                report.deferred.push(RetainedRecord {
                    cache_key: metadata.cache_key.clone(),
                    bytes: entry_bytes(&entry, &metadata),
                    hold: RetentionHold::MaterializationInProgress,
                    detail: None,
                });
                continue;
            }
            let bytes = measure_entry_bytes(&entry)?;
            self.write_eviction_marker(
                &digest,
                &EvictionMarker {
                    schema_version: RESOLVED_CACHE_STORE_VERSION,
                    cache_key: metadata.cache_key.clone(),
                    cause: EvictionCause::SourceRemoved,
                    reclaimable_bytes: bytes,
                    requested_at: now,
                    session_id: self.inner.session_id.clone(),
                },
            )?;
            self.finish_pending_eviction(&digest)?;
            report.removed.push(EvictedRecord {
                cache_key: metadata.cache_key,
                bytes,
                cause: EvictionCause::SourceRemoved,
            });
        }
        Ok(report)
    }

    fn entry_digests(&self) -> Result<Vec<String>, ResolvedCacheError> {
        let mut digests = Vec::new();
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
            digests.push(digest);
        }
        digests.sort();
        Ok(digests)
    }

    /// Classifies one entry for the retention pass. Complete, unpinned, unleased entries become
    /// candidates; everything else is retained with a reason. Locks are dropped after
    /// classification — `evict_candidate` re-verifies everything under fresh locks.
    fn scan_entry(
        &self,
        digest: &str,
        report: &mut RetentionReport,
        candidates: &mut Vec<EvictionCandidate>,
    ) -> Result<(), ResolvedCacheError> {
        let entry = self.inner.root.join("entries").join(digest);
        let _metadata_lock = self.lock_metadata(digest)?;
        match self.read_metadata_unlocked(digest) {
            Ok(JournalRead::Valid { metadata, .. }) => match metadata.state {
                ResolvedCacheEntryState::Complete => {
                    if let Err(error) = validate_complete_metadata(self, &metadata) {
                        report.retained.push(RetainedRecord {
                            cache_key: metadata.cache_key.clone(),
                            bytes: entry_bytes(&entry, &metadata),
                            hold: RetentionHold::RecoveryCandidate,
                            detail: Some(error.to_string()),
                        });
                        return Ok(());
                    }
                    if metadata.effective_pin {
                        report.retained.push(RetainedRecord {
                            cache_key: metadata.cache_key.clone(),
                            bytes: entry_bytes(&entry, &metadata),
                            hold: RetentionHold::Pinned,
                            detail: None,
                        });
                        return Ok(());
                    }
                    if self.artifact_lock_is_contended(digest)? {
                        report.retained.push(RetainedRecord {
                            cache_key: metadata.cache_key.clone(),
                            bytes: entry_bytes(&entry, &metadata),
                            hold: RetentionHold::ActiveUse,
                            detail: None,
                        });
                        return Ok(());
                    }
                    candidates.push(EvictionCandidate {
                        cache_key: metadata.cache_key.clone(),
                        bytes: entry_bytes(&entry, &metadata),
                        activity: entry_activity(&metadata),
                        created_at: metadata.created_at,
                    });
                }
                ResolvedCacheEntryState::Materializing => {
                    let hold = if self.materializing_session_is_live(&metadata)? {
                        RetentionHold::MaterializationInProgress
                    } else {
                        RetentionHold::RecoveryCandidate
                    };
                    report.retained.push(RetainedRecord {
                        cache_key: metadata.cache_key.clone(),
                        bytes: entry_bytes(&entry, &metadata),
                        hold,
                        detail: None,
                    });
                }
                _ => {
                    report.retained.push(RetainedRecord {
                        cache_key: metadata.cache_key.clone(),
                        bytes: entry_bytes(&entry, &metadata),
                        hold: RetentionHold::RecoveryCandidate,
                        detail: None,
                    });
                }
            },
            Ok(JournalRead::Evicted { marker }) => {
                // Finish a previously interrupted eviction; contention just defers it.
                drop(_metadata_lock);
                match self.try_finish_pending_eviction(digest) {
                    Ok(Some(finished)) => report.evicted.push(EvictedRecord {
                        cache_key: finished.cache_key,
                        bytes: finished.reclaimable_bytes,
                        cause: finished.cause,
                    }),
                    Ok(None) => report.retained.push(RetainedRecord {
                        cache_key: marker.cache_key.clone(),
                        bytes: 0,
                        hold: RetentionHold::ActiveUse,
                        detail: Some("pending eviction awaiting artifact lock".to_owned()),
                    }),
                    Err(error) => report
                        .failed
                        .push((marker.cache_key.clone(), error.to_string())),
                }
            }
            Ok(JournalRead::Missing) => {}
            Err(error) => report.retained.push(RetainedRecord {
                cache_key: format!("sha256:{digest}"),
                bytes: 0,
                hold: RetentionHold::RecoveryCandidate,
                detail: Some(error.to_string()),
            }),
        }
        Ok(())
    }

    fn apply_eviction_attempt(
        &self,
        attempt: Result<EvictAttempt, ResolvedCacheError>,
        candidate: &EvictionCandidate,
        report: &mut RetentionReport,
        complete_total: &mut u64,
    ) {
        match attempt {
            Ok(EvictAttempt::Evicted(record)) => {
                *complete_total = complete_total.saturating_sub(record.bytes);
                report.evicted.push(record);
            }
            Ok(EvictAttempt::Retained(record)) => report.retained.push(record),
            Ok(EvictAttempt::AlreadyGone) => {
                *complete_total = complete_total.saturating_sub(candidate.bytes);
            }
            Ok(EvictAttempt::Failed(error)) | Err(ResolvedCacheError(error)) => {
                report.failed.push((candidate.cache_key.clone(), error));
            }
        }
    }

    /// Re-acquires the locks and re-verifies every protection before removing one candidate. Any
    /// doubt — pin, lease, state change, usage since the scan, local validation failure, or an
    /// unverifiable source — retains the entry.
    ///
    /// The sole-copy proof re-hashes every source file, which is minutes of work on a multi-GB
    /// bundle, so it deliberately runs with **no artifact lock held**: `acquire_complete` takes the
    /// artifact lock *blocking*, and hashing underneath the exclusive lock would stall any model
    /// load of that artifact for the whole hash. Phase one establishes the proof unlocked; phase
    /// two takes the exclusive lock and re-verifies only cheap facts — journal generation, state,
    /// pin, activity, path/size validation and a stat-only source recheck — before the tombstone is
    /// written. Anything that changed in between retains the entry rather than trusting phase one.
    fn evict_candidate(
        &self,
        candidate: &EvictionCandidate,
        cause: EvictionCause,
        now: u64,
    ) -> Result<EvictAttempt, ResolvedCacheError> {
        let digest = cache_key_digest(&candidate.cache_key)?;
        let retained = |hold: RetentionHold, detail: Option<String>| {
            Ok(EvictAttempt::Retained(RetainedRecord {
                cache_key: candidate.cache_key.clone(),
                bytes: candidate.bytes,
                hold,
                detail,
            }))
        };

        // Phase one: no artifact lock. Read the entry, cheaply reject anything already protected,
        // then pay for the sole-copy proof while concurrent loads can still acquire the artifact.
        // Tombstones, missing journals and unreadable entries carry no usable proof; phase two
        // resolves each of them under the exclusive lock, where they can be acted on safely.
        let scanned = {
            let _metadata_lock = self.lock_metadata(&digest)?;
            match self.read_metadata_unlocked(&digest) {
                Ok(JournalRead::Valid { metadata, .. }) => Some(*metadata),
                _ => None,
            }
        };
        let source_proof = match &scanned {
            Some(scanned)
                if scanned.state == ResolvedCacheEntryState::Complete
                    && !scanned.effective_pin
                    && entry_activity(scanned) == candidate.activity =>
            {
                Some(self.verify_source_complete(scanned))
            }
            _ => None,
        };

        // Phase two: exclusive artifact lock, then re-verify everything cheaply.
        let artifact_lock = open_lock_file(&self.artifact_lock_path(&digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => {}
            Err(error) if is_lock_contended(&error) => {
                return retained(RetentionHold::ActiveUse, None);
            }
            Err(error) => return Err(error.into()),
        }
        let _metadata_lock = self.lock_metadata(&digest)?;
        let entry = self.inner.root.join("entries").join(&digest);
        let metadata = match self.read_metadata_unlocked(&digest) {
            Ok(JournalRead::Valid { metadata, .. }) => *metadata,
            Ok(JournalRead::Evicted { .. }) => {
                let marker = self.finish_pending_eviction(&digest)?;
                return Ok(EvictAttempt::Evicted(EvictedRecord {
                    cache_key: marker.cache_key,
                    bytes: marker.reclaimable_bytes,
                    cause: marker.cause,
                }));
            }
            Ok(JournalRead::Missing) => {
                // Only an entry directory that is genuinely gone may be subtracted from the size
                // accounting; a present-but-journal-less entry still occupies its bytes.
                return if std::fs::symlink_metadata(&entry).is_err() {
                    Ok(EvictAttempt::AlreadyGone)
                } else {
                    retained(
                        RetentionHold::RecoveryCandidate,
                        Some("entry has no readable journal".to_owned()),
                    )
                };
            }
            Err(error) => {
                return retained(RetentionHold::RecoveryCandidate, Some(error.to_string()));
            }
        };
        if metadata.state != ResolvedCacheEntryState::Complete {
            return retained(RetentionHold::RecoveryCandidate, None);
        }
        if metadata.effective_pin {
            return retained(RetentionHold::Pinned, None);
        }
        if entry_activity(&metadata) != candidate.activity {
            // The entry was used between the scan and this attempt: its LRU position is stale, so
            // this round retains it.
            return retained(
                RetentionHold::Fresh,
                Some("entry was used after it was scanned".to_owned()),
            );
        }
        // The unlocked proof is only usable if the entry is byte-for-byte the one it was taken
        // against; any journal write in between invalidates it.
        let snapshot = match source_proof {
            Some(Ok(snapshot))
                if scanned
                    .as_ref()
                    .is_some_and(|scanned| scanned.updated_at == metadata.updated_at) =>
            {
                snapshot
            }
            Some(Err(detail)) => return retained(RetentionHold::SourceUnverified, Some(detail)),
            _ => {
                return retained(
                    RetentionHold::Fresh,
                    Some("entry changed while its source was being verified".to_owned()),
                );
            }
        };
        if let Err(error) = validate_complete_metadata_inner(
            self,
            &metadata,
            ContentVerification::PathsAndSizesOnly,
        ) {
            return retained(RetentionHold::RecoveryCandidate, Some(error.to_string()));
        }
        if let Err(detail) = revalidate_source_snapshot(&snapshot) {
            return retained(RetentionHold::SourceUnverified, Some(detail));
        }
        self.write_eviction_marker(
            &digest,
            &EvictionMarker {
                schema_version: RESOLVED_CACHE_STORE_VERSION,
                cache_key: metadata.cache_key.clone(),
                cause,
                // Actual on-disk bytes, deduplicated across hard links within the entry, rather
                // than the logical closure sum: materialization hard-links files repeated inside
                // one bundle, so the logical sum over-reports what removal reclaims and would
                // over-drive size enforcement.
                reclaimable_bytes: measure_entry_bytes(&entry).unwrap_or(metadata.verified_bytes),
                requested_at: now,
                session_id: self.inner.session_id.clone(),
            },
        )?;
        match self.finish_pending_eviction(&digest) {
            Ok(marker) => Ok(EvictAttempt::Evicted(EvictedRecord {
                cache_key: marker.cache_key,
                bytes: marker.reclaimable_bytes,
                cause: marker.cause,
            })),
            // The tombstone is durable, so a failed removal (for example a Windows sharing
            // violation) converges on the next pass or recovery instead of being lost.
            Err(error) => Ok(EvictAttempt::Failed(error.to_string())),
        }
    }

    /// Finishes a pending eviction if the artifact lock is free; `None` when contended.
    fn try_finish_pending_eviction(
        &self,
        digest: &str,
    ) -> Result<Option<EvictionMarker>, ResolvedCacheError> {
        let artifact_lock = open_lock_file(&self.artifact_lock_path(digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => {}
            Err(error) if is_lock_contended(&error) => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let _metadata_lock = self.lock_metadata(digest)?;
        match self.read_metadata_unlocked(digest)? {
            JournalRead::Evicted { .. } => Ok(Some(self.finish_pending_eviction(digest)?)),
            _ => Ok(None),
        }
    }

    fn artifact_lock_is_contended(&self, digest: &str) -> Result<bool, ResolvedCacheError> {
        let artifact_lock = open_lock_file(&self.artifact_lock_path(digest))?;
        match FileExt::try_lock_exclusive(&artifact_lock) {
            Ok(()) => Ok(false),
            Err(error) if is_lock_contended(&error) => Ok(true),
            Err(error) => Err(error.into()),
        }
    }

    fn materializing_session_is_live(
        &self,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<bool, ResolvedCacheError> {
        match metadata.session_id.as_deref() {
            Some(session) => Ok(!self.session_lock_is_acquirable(session)?),
            None => Ok(false),
        }
    }

    /// Authoritative pre-eviction source verification: rebuilds the source snapshot location from
    /// the recorded configured library and re-runs the full closure resolution, which checks
    /// reachability, confinement, sizes, and the enriched sha256 hashes of every member file. Any
    /// failure means the resolved bundle may be the sole complete copy.
    ///
    /// This is the expensive half of eviction and runs with no artifact lock held. It returns the
    /// exact identity of what it verified so the decision can be re-confirmed cheaply under the
    /// lock; see [`revalidate_source_snapshot`].
    fn verify_source_complete(
        &self,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<SourceSnapshot, String> {
        let library = self.source_library_for(metadata)?;
        let snapshot = library
            .repository_root(&metadata.artifact.identity.repository)
            .map_err(|error| error.to_string())?
            .join("snapshots")
            .join(&metadata.artifact.identity.revision);
        let locations = metadata
            .artifact
            .closure
            .source_file_locations(&metadata.artifact.identity, &snapshot)
            .map_err(|error| error.to_string())?;
        #[cfg(test)]
        run_source_hash_observer();
        let mut files = Vec::with_capacity(locations.len());
        for location in locations {
            let file_metadata =
                std::fs::symlink_metadata(&location.source_path).map_err(|error| {
                    format!(
                        "verified source file {} became unavailable: {error}",
                        location.source_path.display()
                    )
                })?;
            files.push(VerifiedSourceFile {
                len: file_metadata.len(),
                modified: file_metadata.modified().ok(),
                path: location.source_path,
            });
        }
        Ok(SourceSnapshot { files })
    }

    /// Advisory source probe for manual-removal warnings: reachability plus per-file existence
    /// and size checks, without the full hashing pass.
    fn probe_source_reachable(&self, metadata: &ResolvedCacheMetadata) -> Result<(), String> {
        let library = self.source_library_for(metadata)?;
        for member in &metadata.artifact.closure.members {
            let (_, snapshot) = library
                .discover_snapshot(&member.source.repository, Some(&member.source.revision))
                .map_err(|error| error.to_string())?;
            let member_root = snapshot.join(&member.source_subpath);
            for file in &member.files {
                let path = member_root.join(&file.relative_path);
                let file_metadata = std::fs::metadata(&path).map_err(|error| {
                    format!("source file {} is unavailable: {error}", path.display())
                })?;
                if !file_metadata.is_file() {
                    return Err(format!("source entry {} is not a file", path.display()));
                }
                if file
                    .size_bytes
                    .is_some_and(|expected| expected != file_metadata.len())
                {
                    return Err(format!("source file {} size changed", path.display()));
                }
            }
        }
        Ok(())
    }

    fn source_library_for(
        &self,
        metadata: &ResolvedCacheMetadata,
    ) -> Result<ArtifactSourceLibrary, String> {
        let configured = &metadata.source_configured_path;
        let canonical = std::fs::canonicalize(configured).map_err(|error| {
            format!(
                "source library {} is unavailable: {error}",
                configured.display()
            )
        })?;
        ArtifactSourceLibrary::new(canonical).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
thread_local! {
    static SOURCE_HASH_OBSERVER: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs once, from inside the unlocked source-verification work, so a test can observe which locks
/// are held at that moment.
#[cfg(test)]
fn set_source_hash_observer(observer: impl FnOnce() + 'static) {
    SOURCE_HASH_OBSERVER.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(observer));
    });
}

#[cfg(test)]
fn run_source_hash_observer() {
    SOURCE_HASH_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow_mut().take() {
            observer();
        }
    });
}

/// Actual on-disk bytes of one entry, falling back to the logical closure sum when the entry
/// cannot be walked. Materialization hard-links files repeated inside a bundle, so the logical sum
/// over-reports what a removal would reclaim; measuring keeps reporting and size enforcement honest.
fn entry_bytes(entry: &Path, metadata: &ResolvedCacheMetadata) -> u64 {
    measure_entry_bytes(entry).unwrap_or(metadata.verified_bytes)
}

/// Exactly which source bytes the unlocked sole-copy proof covered.
#[derive(Clone, Debug)]
struct SourceSnapshot {
    files: Vec<VerifiedSourceFile>,
}

#[derive(Clone, Debug)]
struct VerifiedSourceFile {
    path: PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
}

/// Cheap re-confirmation of an already-hashed source, run under the exclusive lock: every verified
/// file must still be a regular file of the same length and modification time. This is stat-only by
/// design — re-hashing here is exactly what would stall a concurrent model load.
fn revalidate_source_snapshot(snapshot: &SourceSnapshot) -> Result<(), String> {
    for file in &snapshot.files {
        let current = std::fs::symlink_metadata(&file.path).map_err(|error| {
            format!(
                "verified source file {} became unavailable: {error}",
                file.path.display()
            )
        })?;
        if current.file_type().is_symlink() || !current.is_file() {
            return Err(format!(
                "verified source file {} is no longer a regular file",
                file.path.display()
            ));
        }
        if current.len() != file.len || current.modified().ok() != file.modified {
            return Err(format!(
                "verified source file {} changed while the eviction was being authorized",
                file.path.display()
            ));
        }
    }
    Ok(())
}

fn entry_activity(metadata: &ResolvedCacheMetadata) -> u64 {
    metadata.last_used_at.unwrap_or(metadata.created_at)
}

fn selector_matches(selector: &SourceLifecycleSelector, metadata: &ResolvedCacheMetadata) -> bool {
    let identity = &metadata.artifact.identity;
    // MSRV 1.80: `Option::is_none_or` is 1.82, so use `map_or(true, …)`.
    identity.repository == selector.repository
        && selector
            .revision
            .as_deref()
            .map_or(true, |revision| identity.revision == revision)
        // Tier labels reach this seam from routes that lowercase them, so match case-insensitively
        // rather than letting a "Q8"/"q8" mismatch silently strand an orphaned entry.
        && selector.tier.as_deref().map_or(true, |tier| {
            metadata
                .artifact
                .provenance
                .fixed_artifact_tier
                .as_deref()
                .is_some_and(|current| current.eq_ignore_ascii_case(tier))
        })
}

/// Measures the actual on-disk bytes of one entry directory: regular files only, links and
/// reparse points fail closed (the confined deleter would refuse them too), and files hard-linked
/// several times inside the same entry are counted once on Unix.
fn measure_entry_bytes(path: &Path) -> Result<u64, ResolvedCacheError> {
    #[cfg(unix)]
    let mut seen = std::collections::BTreeSet::new();
    #[cfg(unix)]
    let mut count = |metadata: &std::fs::Metadata| -> u64 {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() > 1 && !seen.insert((metadata.dev(), metadata.ino())) {
            return 0;
        }
        metadata.len()
    };
    #[cfg(not(unix))]
    let mut count = |metadata: &std::fs::Metadata| -> u64 { metadata.len() };
    measure_entry_bytes_walk(path, &mut count)
}

fn measure_entry_bytes_walk(
    path: &Path,
    count: &mut impl FnMut(&std::fs::Metadata) -> u64,
) -> Result<u64, ResolvedCacheError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
        return Err(ResolvedCacheError::new(format!(
            "cannot measure linked managed path {}",
            path.display()
        )));
    }
    if metadata.is_file() {
        return Ok(count(&metadata));
    }
    if !metadata.is_dir() {
        return Err(ResolvedCacheError::new(format!(
            "cannot measure unmanaged filesystem entry {}",
            path.display()
        )));
    }
    let mut total = 0_u64;
    for item in std::fs::read_dir(path)? {
        let bytes = measure_entry_bytes_walk(&item?.path(), count)?;
        total = total.checked_add(bytes).ok_or_else(|| {
            ResolvedCacheError::new("resolved-cache entry byte measurement overflow")
        })?;
    }
    Ok(total)
}

#[cfg(test)]
#[path = "retention_tests.rs"]
mod tests;
