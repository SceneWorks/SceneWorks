//! The worker's single pre-loader model-source guard.
//!
//! Every job dispatched by `run_utility_job` passes through [`RuntimeSourceGuard::begin`] before
//! its handler constructs a loader. The guard reads the job's generic model carriers
//! (`modelManifestEntry`, `baseModelManifestEntry`, `modelManifestEntries` — declarative manifest
//! data, never per-model wiring), reduces each to the exact requirement closure for THIS worker's
//! platform and the request's selected tier via
//! [`sceneworks_core::model_artifacts::artifact_selection`], and judges it through the one shared
//! resolver. There is deliberately no per-route or per-model availability code anywhere else in
//! this crate:
//!
//! - `local_ready` / `external_ready` proceed (an external source opens an operation-owned
//!   session whose physical identity was proven immediately before loader construction);
//! - `installed_external_unavailable` fails typed — installed state is preserved, receipts are
//!   never rewritten, and no silent re-download is attempted;
//! - `incomplete` / `missing` preserve the established on-demand download behavior of the
//!   handler (a model that was never installed keeps its existing install path).
//!
//! On a later load failure the guard re-probes the exact bound source and remaps the error to the
//! typed unavailable class only when the source is now provably absent or a different physical
//! volume — an engine defect while the source remains available is preserved verbatim, and a raw
//! loader ENOENT is never what a disconnect surfaces as.

use crate::{emit_event_value, JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::contracts::JobType;
use sceneworks_core::model_artifacts::artifact_selection::{
    requested_runtime_variant, selected_requirements_for_model,
};
use sceneworks_core::model_artifacts::external_library::{
    resolve_model_availability, ExternalArtifactRequirement, ExternalSourceSession,
    ModelAvailability, ModelResolution,
};
use sceneworks_core::model_artifacts::local_preference::{
    ActiveLocalArtifacts, LocalArtifactOverlayEntry,
};
use sceneworks_core::model_artifacts::resolved_cache::{ResolvedCacheLease, ResolvedCacheStore};
use sceneworks_core::model_artifacts::ModelArtifactResolver;
use serde_json::json;
use std::path::{Path, PathBuf};
use tracing::Level;

#[derive(Debug)]
pub(crate) struct RuntimeSourceGuard {
    resolutions: Vec<ModelResolution>,
    sessions: Vec<ExternalSourceSession>,
    /// The ONE process-wide local-tier preference scope for this job, holding exactly the
    /// `(repository, revision)` pairs the guard selected. Declared BEFORE the leases so field drop
    /// order stops serving local paths first and only then releases the leases protecting those
    /// bytes — the reverse of acquisition.
    ///
    /// One scope rather than one per model is load-bearing, not tidiness: the coverage decision is
    /// made across the WHOLE job, so a per-model scope could install a bundle for one model that a
    /// sibling model must not be served from, and installing them one at a time leaves a narrower
    /// bundle serving the process while a later model falls back.
    local_scope: Option<ActiveLocalArtifacts>,
    /// Cross-process cache leases (one per locally served artifact) held for the whole job. Each
    /// holds the entry's shared artifact lock, so an evictor taking that lock exclusively — even
    /// non-blockingly, and even when it re-verifies under the lock — cannot remove bytes a load is
    /// reading. Released on drop; `finish_success` crosses the promotion boundary first.
    cache_leases: Vec<ResolvedCacheLease>,
    /// sc-19706: the closures this job served from the SOURCE tier, recorded here so
    /// [`Self::finish_success`] can hand them to the promotion intake without doing any work.
    ///
    /// This is the whole of the trigger's job-path cost: a `Vec` built from values the guard had
    /// already computed for admission. Building the promotion candidate needs the source library
    /// walked, every closure member stat-ed and canonicalized, and a reservation taken — all of it
    /// belongs on the idle drain (`crate::resolved_cache_promotion`), never in the job's critical
    /// path, so nothing here touches the filesystem.
    pending_promotions: Vec<crate::resolved_cache_promotion::PendingPromotion>,
}

impl RuntimeSourceGuard {
    pub(crate) fn begin(
        job_type: &JobType,
        payload: &JsonObject,
        settings: &Settings,
    ) -> WorkerResult<Self> {
        let explicit_download = payload
            .get("modelArtifactOperation")
            .and_then(|value| value.get("kind"))
            .and_then(serde_json::Value::as_str)
            == Some("explicit_download");
        if explicit_download && !matches!(job_type, JobType::ModelDownload | JobType::LoraDownload)
        {
            return Err(WorkerError::InvalidPayload(
                "explicit-download artifact operation is invalid for this job type".to_owned(),
            ));
        }
        let model_free_dry_run = matches!(job_type, JobType::LoraTrain | JobType::ControlTraining)
            && payload.get("dryRun").and_then(serde_json::Value::as_bool) == Some(true);
        if explicit_download || model_free_dry_run {
            return Ok(Self::none());
        }

        let model_entries = payload_model_entries(payload);
        // Fail closed on an EMPTY model carrier set for every model-backed route. Routes that
        // genuinely load no model bytes are declared in the exhaustive table below — a job type
        // absent from that allowlist cannot pass without carriers.
        if job_requires_typed_model_source(job_type) && model_entries.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "{} job carries no model manifest entries for the typed model-source guard",
                job_type.as_str()
            )));
        }

        let requested_variant = requested_runtime_variant(payload);
        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        // Every VALID app-owned bundle, read once for this job. Empty when the resolved cache is
        // disabled or uninitialized, and — by construction of the provider — never containing an
        // entry that is torn, unverifiable, or in a shape the runtime cannot serve, so a partial
        // local copy can only ever move the load back to the source tier.
        let local_scan = local_resolved_artifacts(settings);
        // A bundle whose SHAPE the local tier cannot serve is the one rejection class a user can
        // act on, so it is named here rather than dropped at the scan.
        for rejection in &local_scan.rejections {
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "resolved_cache_local_tier_unsupported",
                    "cacheKey": rejection.cache_key,
                    "repository": rejection.repository,
                    "revision": rejection.revision,
                    "reason": rejection.reason,
                }),
            );
        }
        let local_artifacts = local_scan.artifacts;
        // PASS 1 — resolve every model entry WITH local candidates, keeping the selected closure
        // beside each resolution. Nothing is leased and nothing is served yet: which pairs may be
        // served at all cannot be known until every entry in the job has been resolved.
        let mut plans: Vec<EntryPlan> = Vec::new();
        for entry in &model_entries {
            let has_hf_downloads = entry
                .get("downloads")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|downloads| {
                    downloads.iter().any(|download| {
                        sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download(
                            download,
                        )
                    })
                });
            if !has_hf_downloads {
                // Imported/app-owned and explicitly configured external-root models are the only
                // entries outside the HF-library contract; they must prove their confinement.
                if entry_is_provably_non_hf_local(entry, settings)? {
                    continue;
                }
                return Err(WorkerError::InvalidPayload(format!(
                    "model entry '{}' names no supported artifact source and no confined local path",
                    entry
                        .get("id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("<unknown>")
                )));
            }
            // This worker's OWN platform and the request's selected tier define the closure —
            // never the API host's platform, never a sibling variant's receipts.
            let selected = selected_requirements_for_model(
                entry,
                std::env::consts::OS,
                requested_variant.as_deref(),
                &settings.data_dir,
            );
            let resolution = resolve_model_availability(
                &settings.data_dir,
                &configured_library,
                &selected.requirements,
                selected.receipt_backed,
                &local_artifacts,
            );
            plans.push(EntryPlan {
                requirements: selected.requirements,
                receipt_backed: selected.receipt_backed,
                resolution,
            });
        }

        // PASSES 2 and 3 — decide which `(repository, revision)` pairs the whole job may serve
        // locally, take the leases behind exactly those, and re-resolve every model that did not
        // get its ENTIRE closure served. This is the only layer that can see every model a job
        // loads, and therefore the only layer that can answer the coverage question at all.
        let (local_scope, cache_leases) =
            select_local_tier(settings, &configured_library, &mut plans);

        // sc-19706 — the PRODUCER trigger, recorded here and fired only on success. Read AFTER
        // pass 3 so it reflects the tier that will really serve: a model that fell back from the
        // local tier is external-ready now, and its own closure is exactly what is missing from
        // the cache. Only `ExternalReady` qualifies — a local-ready model is already promoted,
        // and incomplete/missing/unavailable models have nothing installed to copy.
        let pending_promotions = crate::resolved_cache_promotion::pending_for_plans(
            settings,
            &configured_library,
            plans.iter().filter_map(|plan| {
                (plan.resolution.availability == ModelAvailability::ExternalReady)
                    .then_some(plan.requirements.as_slice())
            }),
        );

        let mut resolutions: Vec<ModelResolution> = Vec::new();
        for plan in plans {
            if !resolutions.contains(&plan.resolution) {
                resolutions.push(plan.resolution);
            }
        }

        let mut sessions = Vec::new();
        for resolution in &resolutions {
            resolution.validate().map_err(|error| {
                WorkerError::InvalidPayload(format!("invalid model source resolution: {error}"))
            })?;
            match resolution.availability {
                // Already leased and served from the app-owned tier above; the configured source
                // library is deliberately not consulted, opened, or probed for it.
                ModelAvailability::LocalReady => {}
                ModelAvailability::InstalledExternalUnavailable => {
                    return Err(unavailable(
                        "The configured external model library holding this installed model is \
                         disconnected or no longer has the expected physical identity. Reconnect \
                         it and retry; the installation itself is preserved.",
                    ));
                }
                // A model that was never installed (or whose install is incomplete while the
                // library is provably present) keeps its established on-demand install path.
                ModelAvailability::Incomplete | ModelAvailability::Missing => {}
                ModelAvailability::ExternalReady => {
                    match ExternalSourceSession::begin(&settings.data_dir, resolution) {
                        Ok(session) => sessions.push(session),
                        Err(error) => {
                            // The session probe re-proves path + physical identity and re-walks
                            // the closure. Distinguish "components vanished while the library
                            // stayed provably present" from a disconnect mid-admission.
                            // The requirements came from an external-ready resolution, so the
                            // install evidence is already proven durable.
                            let recheck = resolve_model_availability(
                                &settings.data_dir,
                                &configured_library,
                                &resolution.requirements,
                                true,
                                &[],
                            );
                            if recheck.availability
                                == ModelAvailability::InstalledExternalUnavailable
                            {
                                return Err(unavailable(
                                    "The external model library disconnected while the model \
                                     source was being admitted. Reconnect it and retry.",
                                ));
                            }
                            return Err(WorkerError::InvalidPayload(format!(
                                "model source components could not be validated: {error}"
                            )));
                        }
                    }
                }
            }
        }
        emit_source_tiers(&resolutions);
        Ok(Self {
            resolutions,
            sessions,
            local_scope,
            cache_leases,
            pending_promotions,
        })
    }

    fn none() -> Self {
        Self {
            resolutions: Vec::new(),
            sessions: Vec::new(),
            local_scope: None,
            cache_leases: Vec::new(),
            pending_promotions: Vec::new(),
        }
    }

    pub(crate) fn finish_success(mut self) -> WorkerResult<()> {
        for session in self.sessions.drain(..) {
            session
                .mark_success()
                .map_err(|error| WorkerError::Io(std::io::Error::other(error.to_string())))?;
        }
        // Stop serving local paths before the leases protecting them are released.
        self.local_scope = None;
        for lease in self.cache_leases.drain(..) {
            // The promotion boundary for an artifact that just served a real load. Its own
            // entry is already published, so this only records the successful use.
            let _candidate = lease.mark_success();
        }
        // sc-19706 — the only place a real install becomes a promotion candidate. A job that
        // reached here loaded and ran to completion against the configured source library, which
        // is the exact evidence promotion needs: the bytes exist, they are valid, and something
        // actually used them. Pushing is a lock-and-move into a bounded in-memory queue and is
        // deliberately the last thing the job does — the drain that turns these into bundles runs
        // on the blocking pool when the worker next claims nothing.
        crate::resolved_cache_promotion::record_pending(std::mem::take(
            &mut self.pending_promotions,
        ));
        Ok(())
    }

    /// Mid-load failure classification: re-probe every externally sourced model and suppress the
    /// raw loader error ONLY when the exact bound source is now provably unavailable. A load
    /// failure with the source still present is a real defect and is preserved verbatim.
    pub(crate) fn classify_failure(
        &self,
        settings: &Settings,
        original: WorkerError,
    ) -> WorkerError {
        let configured_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir)
            .root()
            .to_path_buf();
        let source_became_unavailable = self.resolutions.iter().any(|resolution| {
            resolution.availability == ModelAvailability::ExternalReady
                && resolve_model_availability(
                    &settings.data_dir,
                    &configured_library,
                    &resolution.requirements,
                    true,
                    &[],
                )
                .availability
                    == ModelAvailability::InstalledExternalUnavailable
        });
        if source_became_unavailable {
            unavailable(
                "The external model library disconnected or changed during model use; reconnect \
                 it and retry. The original load failure was suppressed because the model's \
                 source is provably unavailable.",
            )
        } else {
            original
        }
    }

    #[cfg(test)]
    pub(crate) fn resolutions(&self) -> &[ModelResolution] {
        &self.resolutions
    }
}

/// One model entry's pass-1 outcome: the resolution, plus the SELECTED closure that produced it.
///
/// The closure is kept here and never re-read off the resolution, because
/// `ModelResolution::local_ready` deliberately carries `requirements: Vec::new()` — a local-ready
/// resolution names an artifact, not a source-library closure. Coverage and the pass-3 fallback
/// both need the real requirement list, so losing it to the resolution would make every
/// locally-resolved model look like it required nothing at all.
struct EntryPlan {
    requirements: Vec<ExternalArtifactRequirement>,
    receipt_backed: bool,
    resolution: ModelResolution,
}

/// Every VALID app-owned resolved artifact on this host, plus the entries rejected for a shape the
/// local tier cannot serve. Empty when the resolved cache is switched off. Read-only: enumeration
/// never creates a cache session and never stamps usage, so judging availability cannot look like
/// using a model.
fn local_resolved_artifacts(
    settings: &Settings,
) -> sceneworks_core::model_artifacts::resolved_cache::LocalArtifactScan {
    if !settings.resolved_cache_policy().enabled {
        return Default::default();
    }
    ResolvedCacheStore::valid_local_artifacts(&settings.data_dir)
}

/// Decide the job's local tier, take the leases behind it, and install the one preference scope.
///
/// # The rule
///
/// > A `(repository, revision)` pair is served locally **iff** one leased bundle holds a SUPERSET
/// > of the union of file requirements every model entry in this job needs from that pair.
/// > Otherwise the pair is served to NOBODY and every model needing it re-resolves against the
/// > authoritative source tier.
///
/// # Why it has to be here
///
/// The preference seam is directory-level: it answers `(repository, revision)` with a snapshot
/// path, and once that path is out it cannot re-ask "does this bundle hold what THIS caller
/// needs". So the question must be settled before any path is handed out, by the only layer that
/// can see every model the job will load — this one. A per-model decision is precisely the defect:
/// a job carrying `owner/matrix` q4 (bundled) and q8 (not bundled) would admit q4, install the
/// overlay for `owner/matrix@REV`, and then resolve the q8 load into the q4 bundle root, which has
/// no `q8/` subtree at all.
///
/// # The deliberate consequence
///
/// In that same q4+q8 job the q4 model ALSO falls back to the source tier, because the union for
/// the shared pair includes q8's files and no bundle covers it. That is the correct conservative
/// answer, not a gap to engineer around: partial service of a pair is exactly what cannot be made
/// safe at a directory-level seam.
///
/// A requirement carrying `revision: None` (a legacy receipt with no recorded snapshot) makes the
/// WHOLE repository unserveable — there is no pair to compare coverage against, so no bundle can
/// be shown to hold what that requirement needs.
fn select_local_tier(
    settings: &Settings,
    configured_library: &std::path::Path,
    plans: &mut [EntryPlan],
) -> (Option<ActiveLocalArtifacts>, Vec<ResolvedCacheLease>) {
    use std::collections::{BTreeMap, BTreeSet};

    let re_resolve_to_source = |plans: &mut [EntryPlan]| {
        for plan in plans.iter_mut() {
            if plan.resolution.availability == ModelAvailability::LocalReady {
                plan.resolution = resolve_model_availability(
                    &settings.data_dir,
                    configured_library,
                    &plan.requirements,
                    plan.receipt_backed,
                    &[],
                );
            }
        }
    };

    if !plans
        .iter()
        .any(|plan| plan.resolution.availability == ModelAvailability::LocalReady)
    {
        return (None, Vec::new());
    }

    // PASS 2a — the union of file requirements per pair, across ALL resolutions, local AND
    // non-local. The non-local sibling is what poisons a pair: it is a model that will read this
    // very snapshot from the source tier, so a bundle that does not hold its files cannot be
    // allowed to answer for the pair at all.
    let mut required: BTreeMap<(String, String), BTreeSet<PathBuf>> = BTreeMap::new();
    let mut unserveable_repositories: BTreeSet<String> = BTreeSet::new();
    for plan in plans.iter() {
        for requirement in &plan.requirements {
            let Some(revision) = requirement.revision.as_ref() else {
                unserveable_repositories.insert(requirement.repository.clone());
                continue;
            };
            required
                .entry((requirement.repository.clone(), revision.clone()))
                .or_default()
                .extend(requirement.files.iter().cloned());
        }
    }

    // PASS 2b — candidate entries, read from the SCANNED artifacts. NOTHING is leased yet, and
    // that is deliberate: `acquire_complete` stamps `last_used_at` and takes the entry's shared
    // artifact lock through the BLOCKING `FileExt::lock_shared`. Leasing a candidate that coverage
    // then rejects would both mark an entry this job never reads as recently used — skewing
    // retention's LRU against the entries that really served — and stall the whole guard behind an
    // evictor holding a lock on a bundle no load here will ever touch.
    let mut candidates: Vec<(String, LocalArtifactOverlayEntry)> = Vec::new();
    // cache key -> repository, the only thing the lease needs from the artifact.
    let mut candidate_repositories: BTreeMap<String, String> = BTreeMap::new();
    for plan in plans.iter() {
        if plan.resolution.availability != ModelAvailability::LocalReady {
            continue;
        }
        let Some(artifact) = plan.resolution.local_artifact.as_ref() else {
            continue;
        };
        let Ok(cache_key) = artifact.cache_key() else {
            continue;
        };
        if candidate_repositories.contains_key(&cache_key) {
            continue;
        }
        match sceneworks_core::model_artifacts::local_preference::overlay_entries_for_artifact(
            artifact,
        ) {
            Ok(entries) => {
                candidate_repositories
                    .insert(cache_key.clone(), artifact.identity.repository.clone());
                candidates.extend(entries.into_iter().map(|entry| (cache_key.clone(), entry)));
            }
            Err(error) => {
                // The scan already reports this class, so reaching it here means the artifact the
                // resolver chose differs from what the scan offered. Named the same way either way.
                emit_event_value(
                    Level::WARN,
                    json!({
                        "event": "resolved_cache_local_tier_unsupported",
                        "cacheKey": cache_key,
                        "repository": artifact.identity.repository,
                        "revision": artifact.identity.revision,
                        "reason": error.to_string(),
                    }),
                );
            }
        }
    }

    // PASS 2c — tentatively select at most one serving entry per pair, and only on full coverage.
    let mut tentative: Vec<(String, LocalArtifactOverlayEntry)> = Vec::new();
    for ((repository, revision), files) in &required {
        if unserveable_repositories.contains(repository) {
            continue;
        }
        let files: Vec<PathBuf> = files.iter().cloned().collect();
        let Some((cache_key, entry)) = candidates.iter().find(|(_, entry)| {
            entry.repository == *repository
                && entry.revision == *revision
                && sceneworks_core::model_artifacts::local_preference::entry_covers(entry, &files)
        }) else {
            continue;
        };
        tentative.push((cache_key.clone(), entry.clone()));
    }

    // PASS 2d — lease EXACTLY the winners, and re-prove every selection against the LEASED copy.
    // The lease re-reads and re-validates the published entry under its shared artifact lock, so a
    // selection that no longer matches that copy was made against a stale scan; its pair is dropped
    // (the models needing it fall back in pass 3) rather than served on stale evidence.
    let mut leases: Vec<ResolvedCacheLease> = Vec::new();
    let mut selected_entries: Vec<LocalArtifactOverlayEntry> = Vec::new();
    let mut served_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    let winners: BTreeSet<String> = tentative.iter().map(|(key, _)| key.clone()).collect();
    for cache_key in &winners {
        let Some(repository) = candidate_repositories.get(cache_key) else {
            continue;
        };
        let Some(lease) = lease_local_artifact(settings, cache_key, repository) else {
            continue;
        };
        let Ok(leased) =
            sceneworks_core::model_artifacts::local_preference::overlay_entries_for_artifact(
                lease.artifact(),
            )
        else {
            continue;
        };
        let mut serves_a_pair = false;
        for (_, entry) in tentative.iter().filter(|(key, _)| key == cache_key) {
            // IDENTICAL, not merely present: the coverage decision was made about this exact
            // snapshot root and file set, and anything else is a different bundle.
            if !leased.contains(entry) {
                continue;
            }
            served_pairs.insert((entry.repository.clone(), entry.revision.clone()));
            selected_entries.push(entry.clone());
            serves_a_pair = true;
        }
        if serves_a_pair {
            leases.push(lease);
        }
    }

    // PASS 3 — a model stays LocalReady only if EVERY pair in its closure was selected. Any other
    // model re-resolves against the source tier, keeping every typed disconnect/incomplete
    // guarantee instead of proceeding on a promise the runtime cannot keep.
    for plan in plans.iter_mut() {
        if plan.resolution.availability != ModelAvailability::LocalReady {
            continue;
        }
        let fully_served = plan.requirements.iter().all(|requirement| {
            requirement.revision.as_ref().is_some_and(|revision| {
                served_pairs.contains(&(requirement.repository.clone(), revision.clone()))
            })
        });
        if fully_served {
            continue;
        }
        emit_event_value(
            Level::INFO,
            json!({
                "event": "resolved_cache_local_tier_not_selected",
                "repository": plan
                    .requirements
                    .iter()
                    .find(|requirement| requirement.is_primary)
                    .map(|requirement| requirement.repository.clone())
                    .unwrap_or_default(),
                "reason": "no leased bundle covers every file this job needs from one of the \
                           model's (repository, revision) pairs",
            }),
        );
        plan.resolution = resolve_model_availability(
            &settings.data_dir,
            configured_library,
            &plan.requirements,
            plan.receipt_backed,
            &[],
        );
    }

    if selected_entries.is_empty() {
        return (None, Vec::new());
    }
    // ONE scope holding EXACTLY the selected entries. A pair in it is serveable to every consumer
    // in the process by construction, including a model that fell back above for a DIFFERENT pair
    // — the union that admitted this pair already covered that model's needs from it.
    match sceneworks_core::model_artifacts::local_preference::prefer_local_snapshots(
        selected_entries,
    ) {
        Ok(scope) => (Some(scope), leases),
        Err(error) => {
            // FAIL CLOSED, and actually close: nothing is installed, every lease is dropped, and
            // every locally resolved model goes back to the source tier. Installing a partial
            // scope here is the exact defect this function exists to prevent.
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "resolved_cache_local_tier_not_selected",
                    "reason": error.to_string(),
                }),
            );
            re_resolve_to_source(plans);
            (None, Vec::new())
        }
    }
}

/// Take the runtime lease on one locally resolved artifact, or `None` when it cannot be leased
/// after all (concurrently evicted, no longer complete). The lease is what keeps the bytes behind a
/// served path from being evicted mid-load, so it is always acquired BEFORE the scope that serves
/// those bytes is installed.
fn lease_local_artifact(
    settings: &Settings,
    cache_key: &str,
    repository: &str,
) -> Option<ResolvedCacheLease> {
    let store = cache_store(settings)?;
    let resolver = ModelArtifactResolver::new(sceneworks_core::hf_home::model_source_library(
        &settings.data_dir,
    ));
    // The store takes the entry's SHARED artifact lock before it re-reads and re-validates the
    // published entry under it, then stamps usage. An evictor that reaches the entry first holds
    // the exclusive lock, so this waits; an evictor that arrives afterwards finds the lock held
    // and must leave the entry alone.
    store
        .acquire_complete(cache_key, &resolver, repository)
        .ok()
        .flatten()
}

/// One long-lived cache session per data directory. Opening a session takes a durable session lock
/// and writes session records, so a fresh session per job would churn the store; the worker claims
/// one job at a time and keeps this handle for the process lifetime instead.
fn cache_store(settings: &Settings) -> Option<ResolvedCacheStore> {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    static STORES: OnceLock<Mutex<BTreeMap<std::path::PathBuf, ResolvedCacheStore>>> =
        OnceLock::new();
    let stores = STORES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut stores = stores
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(store) = stores.get(&settings.data_dir) {
        return Some(store.clone());
    }
    let store = ResolvedCacheStore::open(&settings.data_dir).ok()?;
    stores.insert(settings.data_dir.clone(), store.clone());
    Some(store)
}

/// Report which tier will serve each admitted model. This is the runtime's answer to "where did
/// this load's weights come from", emitted once per job before any loader runs.
fn emit_source_tiers(resolutions: &[ModelResolution]) {
    for resolution in resolutions {
        let tier = match resolution.availability {
            ModelAvailability::LocalReady => "resolved_local",
            ModelAvailability::ExternalReady => "source_library",
            _ => continue,
        };
        let (repository, revision) = match &resolution.local_artifact {
            Some(artifact) => (
                artifact.identity.repository.clone(),
                artifact.identity.revision.clone(),
            ),
            None => resolution
                .requirements
                .iter()
                .find(|requirement| requirement.is_primary)
                .map(|requirement| {
                    (
                        requirement.repository.clone(),
                        requirement.revision.clone().unwrap_or_default(),
                    )
                })
                .unwrap_or_default(),
        };
        emit_event_value(
            Level::INFO,
            json!({
                "event": "model_source_tier_selected",
                "tier": tier,
                "repository": repository,
                "revision": revision,
            }),
        );
    }
}

/// Allowlist AT THE SEAM: only routes that cannot open model/HF bytes may run without model
/// carriers. `JobType` is `#[non_exhaustive]`, so this match cannot force a compile error for a
/// new variant; instead the wildcard arm classifies anything unlisted as model-backed — an
/// unclassified route therefore defaults to FAIL CLOSED (requires carriers) and can only become
/// model-free by an explicit entry in the allowlist above.
fn job_requires_typed_model_source(job_type: &JobType) -> bool {
    match job_type {
        JobType::Placeholder
        | JobType::FrameExtract
        | JobType::TimelineExport
        | JobType::DatasetParquetImport
        | JobType::ModelImport
        | JobType::LoraImport => false,
        JobType::ImageGenerate
        | JobType::ImageEdit
        | JobType::ImageVqa
        | JobType::ImageInterleave
        | JobType::VideoGenerate
        | JobType::VideoExtend
        | JobType::VideoBridge
        | JobType::PersonDetect
        | JobType::PersonTrack
        | JobType::PersonReplace
        | JobType::AudioGenerate
        | JobType::PoseDetect
        | JobType::KpsExtract
        | JobType::ImageUpscale
        | JobType::ImageDetail
        | JobType::ImageSegment
        | JobType::VideoUpscale
        | JobType::ModelDownload
        | JobType::ModelConvert
        | JobType::LoraDownload
        | JobType::LoraTrain
        | JobType::ControlTraining
        | JobType::TrainingCaption
        | JobType::DatasetAnalysis
        | JobType::CatalogAnalysis
        | JobType::DatasetUpscale
        | JobType::DatasetFaceAnalysis
        | JobType::FaceLikenessCompare
        | JobType::PromptRefine
        | JobType::Unknown(_) => true,
        _ => true,
    }
}

fn payload_model_entries(payload: &JsonObject) -> Vec<&serde_json::Value> {
    ["modelManifestEntry", "baseModelManifestEntry"]
        .into_iter()
        .filter_map(|key| payload.get(key))
        .chain(
            payload
                .get("modelManifestEntries")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter(|entry| !entry.is_null())
        .collect()
}

/// Imported/app-owned and explicitly configured external-root models are the only model entries
/// that may omit the HF artifact contract. They must have no download descriptors at all and
/// every concrete path they expose must pass the worker's existing app-managed confinement
/// check. Deliberately per-entry: one valid primary carrier cannot hide an unconfined auxiliary.
///
/// "Concrete path" means the fields that really name a location — `modelPath`, `installedPath`,
/// every value of the `paths` locator map, `source.path`, and each component's `path` — and
/// nothing else. Provenance fields that merely *describe* where the model came from are not
/// filesystem inputs and are not confined (sc-20524).
///
/// The two kinds of locator are confined DIFFERENTLY, and deliberately so. A LOADER locator
/// (`modelPath` / `installedPath` / `paths.*` / `components[].path`) is a string some consumer will
/// re-resolve verbatim, and every one of those consumers anchors a relative path on the process cwd
/// (`crate::paths::normalize_absolute_path`, `model_artifacts::canonical_or_absolute`), so this
/// guard must judge exactly the same string the loader will open — otherwise the two halves of the
/// two-boundary defense (`apps/rust-api/src/jobs.rs`) stop talking about the same file. Only
/// `source.path` — provenance, which nothing ever opens — is anchored on the data dir, because that
/// is the base the model-import job wrote it against.
fn entry_is_provably_non_hf_local(
    entry: &serde_json::Value,
    settings: &Settings,
) -> WorkerResult<bool> {
    if entry
        .get("downloads")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|downloads| !downloads.is_empty())
    {
        return Ok(false);
    }
    let mut loader_paths = Vec::new();
    for key in ["modelPath", "installedPath"] {
        if let Some(path) = entry.get(key).and_then(serde_json::Value::as_str) {
            loader_paths.push(path);
        }
    }
    // `paths` is a locator map — every value in it is a place on disk (`{ "model": … }`).
    loader_paths.extend(
        entry
            .get("paths")
            .and_then(serde_json::Value::as_object)
            .into_iter()
            .flat_map(|object| object.values())
            .filter_map(serde_json::Value::as_str),
    );
    loader_paths.extend(
        entry
            .get("components")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|component| component.get("path"))
            .filter_map(serde_json::Value::as_str),
    );
    // `source`, by contrast, is a PROVENANCE record, and only its `path` names a location:
    // `provider` is a literal ("local", "huggingface", "url"), `repo` is a repository id or
    // null, `url` is a URL. sc-20524 — confining every value in this object as a filesystem
    // path resolved `"local"` against the process cwd and rejected EVERY user-imported model
    // ("non-HF model source must be inside an app-managed directory") before any loader ran.
    let source_path = entry
        .get("source")
        .and_then(|source| source.get("path"))
        .and_then(serde_json::Value::as_str);
    if loader_paths.is_empty() && source_path.is_none() {
        return Ok(false);
    }
    // A loader locator is judged exactly as its consumer will re-resolve it: raw, cwd-anchored
    // when relative. Same string, same base, same verdict on both sides of the boundary.
    for path in loader_paths {
        crate::paths::normalize_app_managed_model_path(settings, path, LOCATOR_LABEL)?;
    }
    if let Some(path) = source_path {
        confine_provenance_path(settings, path)?;
    }
    Ok(true)
}

const LOCATOR_LABEL: &str = "non-HF model source";

/// Confine the ONE provenance locator, `source.path`.
///
/// Nothing opens this field — it records where an imported model came from — so it is the one
/// locator whose base this guard may choose, and the model-import job writes it relative to the
/// data dir (`models/imports/<name>`), never to the worker's cwd. Anchoring it there is what lets
/// an imported entry be judged against the base it was actually written against.
///
/// The invariant this preserves: no locator that resolves outside every allowed root is admitted,
/// under data-dir anchoring of `source.path`. Anchoring changes which path a relative
/// `source.path` names; it never skips the check. A `..` (or a symlink) that walks back out of the
/// data dir still lands outside every root and is still rejected, which is what keeps a forged
/// "local" entry out.
///
/// Windows has two spellings that are neither absolute nor anchorable, and `PathBuf::join` silently
/// does the wrong thing with both: `Path::is_absolute` is FALSE for the root-relative
/// `\Windows\System32` (`data_dir.join(..)` keeps only the drive prefix, so nothing of the data dir
/// survives), and false for the drive-relative `D:models` (`join` replaces the data dir outright,
/// leaving a path resolved against that drive's per-process cwd). Neither has any meaning relative
/// to the data dir, so both are refused here rather than confined as a path the anchoring never
/// produced.
fn confine_provenance_path(settings: &Settings, raw: &str) -> WorkerResult<()> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "{LOCATOR_LABEL} is required."
        )));
    }
    let candidate = Path::new(trimmed);
    let anchored = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else if candidate.components().any(|component| {
        matches!(
            component,
            std::path::Component::Prefix(_) | std::path::Component::RootDir
        )
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "{LOCATOR_LABEL} must be an absolute path or relative to the app data directory: {trimmed}"
        )));
    } else {
        settings.data_dir.join(candidate)
    };
    // Path-valued so the anchored form is never round-tripped through `to_string_lossy`: a
    // non-UTF-8 component would become U+FFFD and the check would judge a different path.
    crate::paths::normalize_app_managed_model_entry_path(settings, &anchored, LOCATOR_LABEL)?;
    Ok(())
}

fn unavailable(detail: impl Into<String>) -> WorkerError {
    WorkerError::ExternalLibraryUnavailable(detail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE;
    use sceneworks_core::model_artifacts::ArtifactSourceLibrary;
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    // FILE-UNIQUE fixture identities, deliberately. Some of these tests install the PROCESS-WIDE
    // local-tier preference overlay, and the overlay is keyed on `(repository, revision)` — so a
    // fixture repository or revision shared with an unrelated test elsewhere in this crate
    // (`model_jobs.rs`, `image_jobs/tests.rs`, `flux1_control_candle.rs` all used the same
    // `owner/...` names and the same `0123456789ab…` revision) lets a live overlay here answer for
    // a snapshot that test expects to read from its own temp library. The suite runs in parallel,
    // so that is a rare, load-dependent, entirely genuine false red in a file that never mentions
    // the resolved cache. The `guardfx/` prefix and the `e19706…` revisions appear nowhere else.
    const REV_A: &str = "e19706aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REV_B: &str = "e19706bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn settings(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: crate::test_env::OFFLINE_URL.to_owned(),
            access_token: None,
            data_dir,
            config_dir: PathBuf::new(),
            worker_id: "worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: crate::test_env::OFFLINE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: 1,
            max_model_url_bytes: 1,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: false,
            backend_candle_enabled: false,
            external_model_roots: Vec::new(),
            gpu_memory_limit_bytes: 0,
        }
    }

    /// Pin the configured source library to an explicit external dir for the whole body.
    /// `huggingface_hub_cache_dir` reads `HF_HUB_CACHE` before `data_dir`, and other tests in
    /// this crate mutate those vars — the shared `test_env` lock serializes against them, and the
    /// explicit dir models the external-drive scenario directly.
    fn with_library<T>(library: &Path, body: impl FnOnce() -> T) -> T {
        crate::test_env::temp_env_vars(
            &[("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path"))],
            body,
        )
    }

    fn seed_snapshot(library: &Path, repo: &str, revision: &str, file: &str) {
        let safe = sceneworks_core::hf_home::safe_repo_dir_name(repo).unwrap();
        let snapshot = library
            .join(format!("models--{safe}"))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        if let Some(parent) = snapshot.join(file).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(snapshot.join(file), b"weights").unwrap();
    }

    fn write_receipts(data_dir: &Path, repo: &str, receipts: Value) {
        let managed = data_dir
            .join("models")
            .join(sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
        )
        .unwrap();
    }

    /// One installed single-variant model: manifest entry + receipt + library snapshot.
    fn installed_model(temp: &TempDir) -> (Settings, PathBuf, JsonObject) {
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/model", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "guardfx/model",
            json!([{ "repo": "guardfx/model", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        let payload = json!({
            "model": "m",
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/model",
                                "revision": REV_A, "files": ["model.safetensors"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        (settings(data), library, payload)
    }

    /// Serialize every test that installs a local-tier preference scope. The scope is PROCESS-WIDE
    /// by design (model paths resolve on engine threads far below the job task), so two of these
    /// bodies running concurrently would read each other's overlay.
    ///
    /// This mutex is the ONLY thing serializing them. `.cargo/config.toml` sets no
    /// `RUST_TEST_THREADS`, so this suite really does run in parallel and the interleaving is live
    /// rather than hypothetical — an earlier version of this comment claimed the harness already
    /// serialized the suite, which was false and would have made dropping the mutex look safe.
    fn overlay_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Pin the configured source library AND switch the resolved cache on for the body. Under
    /// test `Settings::resolved_cache_policy` reads the same environment `Settings::from_env`
    /// would have read, so this is exactly the production "user enabled the cache" state.
    fn with_local_cache<T>(library: &Path, body: impl FnOnce() -> T) -> T {
        let _serialized = overlay_guard();
        crate::test_env::temp_env_vars(
            &[
                ("HF_HUB_CACHE", library.to_str().expect("utf-8 temp path")),
                (
                    sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                    "1",
                ),
            ],
            body,
        )
    }

    /// Publish a real bundle through the PRODUCTION materializer: the source snapshot is copied,
    /// verified and atomically published exactly as a promotion would do it, so these tests read
    /// back the same on-disk shape the runtime will meet.
    fn publish_bundle(
        data_dir: &Path,
        library: &Path,
        repository: &str,
        revision: &str,
        variant: &str,
        files: &[&str],
    ) -> sceneworks_core::model_artifacts::ResolvedModelArtifact {
        let destination =
            sceneworks_core::model_artifacts::local_preference::hub_cache_member_destination(
                repository,
                revision,
                Path::new(""),
            )
            .unwrap();
        publish_bundle_at(
            data_dir,
            library,
            repository,
            revision,
            variant,
            files,
            destination,
        )
    }

    /// `publish_bundle`, but with the bundle-relative member destination chosen by the caller — the
    /// one property that decides whether the published bundle is a shape the local tier can serve.
    fn publish_bundle_at(
        data_dir: &Path,
        library: &Path,
        repository: &str,
        revision: &str,
        variant: &str,
        files: &[&str],
        destination: PathBuf,
    ) -> sceneworks_core::model_artifacts::ResolvedModelArtifact {
        use sceneworks_core::model_artifacts::resolved_cache::{
            MaterializationCancellation, MaterializationOutcome, ResolvedCacheMaterializer,
        };
        use sceneworks_core::model_artifacts::{
            ArtifactAvailability, ArtifactFile, ArtifactIdentity, ArtifactMemberRole,
            ResolvedBundleClosure, ResolvedBundleMember,
        };

        let identity = ArtifactIdentity::pinned(repository, revision, variant).unwrap();
        let closure = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: identity.clone(),
            tier: Some(variant.to_owned()),
            source_subpath: PathBuf::new(),
            destination,
            files: files
                .iter()
                .map(|file| ArtifactFile::new(*file).unwrap())
                .collect(),
        }])
        .unwrap();
        let resolver = ModelArtifactResolver::new(ArtifactSourceLibrary::new(library).unwrap());
        let source = resolver
            .resolve_source(identity, closure, ArtifactAvailability::Available)
            .unwrap();
        let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
            cache_key: source.cache_key().unwrap(),
            artifact: source,
        };
        let store = ResolvedCacheStore::open(data_dir).unwrap();
        let outcome = ResolvedCacheMaterializer::new(store)
            .materialize(
                &candidate,
                library,
                repository,
                &MaterializationCancellation::default(),
            )
            .unwrap();
        match outcome {
            MaterializationOutcome::Published(metadata) => metadata.artifact,
            other => panic!("bundle was not published: {other:?}"),
        }
    }

    /// The walking skeleton: an installed model with a published bundle admits from the app-owned
    /// tier, the SHARED snapshot resolver every image/video/audio/utility loader uses returns the
    /// bundle path, an evictor cannot take the entry while the job holds it, and finishing the job
    /// restores the source tier.
    #[test]
    fn a_published_bundle_serves_the_load_and_is_lease_protected() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let cache_key = artifact.cache_key().unwrap();
            let bundle_root = artifact.location.root().to_path_buf();

            // Before the guard runs, the shared resolver reads the configured source library.
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(source_snapshot.starts_with(&library));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );

            // The one shared snapshot resolver — every model-consuming runtime funnels through it
            // — now answers with the app-owned bundle.
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot");
            assert_eq!(
                loaded,
                bundle_root.join(format!("models--guardfx--model/snapshots/{REV_A}"))
            );
            assert!(loaded.join("model.safetensors").is_file());
            // And so does the pinned-component resolver used by the utility/co-requisite lanes.
            assert_eq!(
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    &settings.data_dir,
                    "guardfx/model",
                    REV_A
                ),
                Some(loaded.clone())
            );

            // An evictor takes the entry's artifact lock exclusively; the live lease denies it.
            let evictor = ResolvedCacheStore::open(&settings.data_dir).unwrap();
            let candidate = sceneworks_core::model_artifacts::PromotionCandidate {
                cache_key: cache_key.clone(),
                artifact: ModelArtifactResolver::new(ArtifactSourceLibrary::new(&library).unwrap())
                    .resolve_source(
                        artifact.identity.clone(),
                        artifact.closure.clone(),
                        sceneworks_core::model_artifacts::ArtifactAvailability::Available,
                    )
                    .unwrap(),
            };
            assert!(matches!(
                evictor
                    .reserve(&candidate, &library, "guardfx/model")
                    .unwrap(),
                sceneworks_core::model_artifacts::resolved_cache::ReservationOutcome::Contended
            ));

            guard.finish_success().unwrap();
            // Usage was recorded for a real load, and the entry survives untouched.
            let metadata = evictor.lookup_complete(&cache_key).unwrap().unwrap();
            assert!(metadata.last_used_at.is_some());
            // With the job finished the source tier serves again.
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model"),
                Some(source_snapshot)
            );
        });
    }

    /// A job that carries several models (a primary plus utility/co-requisite carriers) leases and
    /// serves EVERY one of them from the local tier — admission is per artifact, so one model can
    /// never silently drag another back onto the source tier.
    #[test]
    fn every_model_a_job_carries_is_served_from_its_own_leased_bundle() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        for repo in ["guardfx/primary", "guardfx/utility"] {
            seed_snapshot(&library, repo, REV_A, "model.safetensors");
            write_receipts(
                &data,
                repo,
                json!([{ "repo": repo, "modelId": repo,
                         "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
            );
        }
        let settings = settings(data);
        let entry = |repo: &str| {
            json!({
                "id": repo,
                "downloads": [{ "provider": "huggingface", "repo": repo, "revision": REV_A,
                                "files": ["model.safetensors"] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("guardfx/primary"),
            "modelManifestEntries": [entry("guardfx/utility")]
        })
        .as_object()
        .unwrap()
        .clone();
        with_local_cache(&library, || {
            for repo in ["guardfx/primary", "guardfx/utility"] {
                publish_bundle(
                    &settings.data_dir,
                    &library,
                    repo,
                    REV_A,
                    "default",
                    &["model.safetensors"],
                );
            }
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 2);
            assert!(guard
                .resolutions()
                .iter()
                .all(|resolution| resolution.availability == ModelAvailability::LocalReady));
            for repo in ["guardfx/primary", "guardfx/utility"] {
                let loaded = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
                    .expect("leased local snapshot");
                assert!(loaded.join("model.safetensors").is_file());
                assert!(!loaded.starts_with(&library));
            }
            guard.finish_success().unwrap();
        });
    }

    /// The runtimes whose model directory is a concrete path resolved BEFORE the load — training
    /// base models, captioners, analyzers (the `ManagedModelPath` entrypoint) — and the
    /// receipt-provenance lane both read the leased bundle too, so no model-consuming surface is
    /// left on the source tier while a lease is active.
    #[test]
    fn payload_resolved_and_receipt_paths_are_served_from_the_leased_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle_snapshot = artifact
                .location
                .root()
                .join(format!("models--guardfx--model/snapshots/{REV_A}"));
            let source_snapshot = ArtifactSourceLibrary::new(&library)
                .unwrap()
                .repository_root("guardfx/model")
                .unwrap()
                .join("snapshots")
                .join(REV_A);

            // Without a lease the authoritative source path survives untouched.
            let before = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert!(before.ends_with(format!("snapshots/{REV_A}")));
            assert!(!before.starts_with(artifact.location.root()));

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let during = crate::paths::normalize_app_managed_model_path(
                &settings,
                source_snapshot.to_str().unwrap(),
                "base model",
            )
            .unwrap();
            assert_eq!(during, bundle_snapshot);
            // The receipt lane proves its install against the source and then hands the loader the
            // leased local path.
            assert_eq!(
                crate::model_jobs::huggingface_receipt_weights_dir(
                    &settings.data_dir,
                    "guardfx/model",
                    Some("m"),
                    None
                ),
                Some(bundle_snapshot)
            );
            guard.finish_success().unwrap();

            // And the redirect is gone with the lease.
            assert_eq!(
                crate::paths::normalize_app_managed_model_path(
                    &settings,
                    source_snapshot.to_str().unwrap(),
                    "base model",
                )
                .unwrap(),
                before
            );
        });
    }

    /// The epic's headline outcome: with the configured library disconnected, a complete local
    /// bundle still admits and still resolves — without the runtime ever reaching a downloader or
    /// recreating the configured library root.
    #[test]
    fn a_disconnected_library_still_serves_a_complete_bundle_without_downloading() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot while disconnected");
            assert!(loaded.join("model.safetensors").is_file());
            let resolved_root =
                std::fs::canonicalize(settings.data_dir.join("models").join("resolved")).unwrap();
            assert!(loaded.starts_with(&resolved_root), "{}", loaded.display());
            // Nothing recreated the configured library or a parallel download destination: the
            // preference path never reaches the downloader.
            assert!(!library.exists());
            guard.finish_success().unwrap();
        });
    }

    /// Each invalid-entry case runs against a bundle that WOULD have served the load, with exactly
    /// one property broken, and asserts the control first — so a case can only pass because the
    /// broken property is what moved the load back to the authoritative source tier.
    fn invalid_local_entry_falls_back(break_it: impl FnOnce(&Path, &Path)) {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            let artifact = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
            let bundle = artifact.location.root().to_path_buf();
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the intact bundle must serve this load"
            );
            drop(guard);

            break_it(&settings.data_dir, &bundle);

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady,
                "an invalid local entry must fall back to the authoritative source"
            );
            assert_eq!(
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model"),
                Some(source_snapshot),
                "and the load must read the source library, not the invalid bundle"
            );
        });
    }

    fn published_file(bundle: &Path) -> PathBuf {
        bundle.join(format!(
            "models--guardfx--model/snapshots/{REV_A}/model.safetensors"
        ))
    }

    #[test]
    fn a_torn_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::write(published_file(bundle), b"").unwrap();
        });
    }

    #[test]
    fn a_partial_local_entry_never_wins() {
        invalid_local_entry_falls_back(|_, bundle| {
            std::fs::remove_file(published_file(bundle)).unwrap();
        });
    }

    #[test]
    fn an_unverifiable_local_entry_never_wins() {
        invalid_local_entry_falls_back(|data_dir, _| {
            let entries = data_dir.join("models/resolved/entries");
            for entry in std::fs::read_dir(&entries).unwrap().flatten() {
                for slot in 0..=1 {
                    let path = entry.path().join(format!("metadata.{slot}.json"));
                    if path.exists() {
                        std::fs::write(&path, b"{\"not\":\"a journal\"}").unwrap();
                    }
                }
            }
        });
    }

    /// A bundle of a DIFFERENT revision or a DIFFERENT tier than the one this request selected is
    /// not this request's artifact: the selected closure keeps its source tier and the local entry
    /// is left alone.
    #[test]
    fn a_wrong_revision_or_wrong_tier_bundle_is_not_used() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = json!({
            "id": "m",
            "downloads": [
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q4",
                  "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q8",
                  "files": ["q8/*"] }
            ]
        });
        let payload = |variant: &str| {
            json!({ "modelManifestEntry": entry, "variant": variant })
                .as_object()
                .unwrap()
                .clone()
        };
        with_local_cache(&library, || {
            // Only the q4 tier is published locally.
            let q4_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let q4_root = q4_bundle.location.root().to_path_buf();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the q4 selection IS served by the q4 bundle"
            );
            drop(guard);

            // The q8 selection must NOT be served by the q4 bundle even though both share one
            // repository and one immutable revision.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            // AND — the assertion this test used to skip by calling `drop(guard)` first — the PATH
            // layer must agree. Availability said "source tier"; the bug lived one layer down,
            // where the process-wide overlay would still redirect `guardfx/matrix@REV_A` into the q4
            // bundle root, which holds no `q8/` subtree at all. With the guard STILL HELD:
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/matrix")
                    .expect("source snapshot");
            assert!(
                loaded.starts_with(&library),
                "the q8 load must read the source library, not the q4 bundle: {}",
                loaded.display()
            );
            assert!(!loaded.starts_with(&q4_root), "{}", loaded.display());
            assert!(
                loaded.join("q8/model.safetensors").is_file(),
                "and the path it was handed must actually hold the q8 weights"
            );
            drop(guard);

            // A bundle of another revision of the same model is likewise not this model.
            seed_snapshot(&library, "guardfx/other", REV_B, "model.safetensors");
            let other_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/other",
                REV_B,
                "default",
                &["model.safetensors"],
            );
            let other = json!({
                "modelManifestEntry": {
                    "id": "other",
                    "downloads": [{ "provider": "huggingface", "repo": "guardfx/other",
                                    "revision": REV_A, "files": ["model.safetensors"] }]
                }
            })
            .as_object()
            .unwrap()
            .clone();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &other, &settings).unwrap();
            assert_ne!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "a bundle of a different revision must never satisfy a pinned request"
            );
            // Path layer, guard still held: the REV_B bundle must not answer for a REV_A request.
            assert!(crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                "guardfx/other",
                REV_A
            )
            .is_none_or(|path| !path.starts_with(other_bundle.location.root())));
            drop(guard);
        });
    }

    /// The DELIBERATE consequence of deciding coverage across the whole job: when a job carries
    /// both the q4 and the q8 selection of one model and only q4 is bundled, the q4 model ALSO
    /// falls back to the source tier.
    ///
    /// That is the correct conservative answer, not a gap. The two selections share one
    /// `(repository, revision)` pair, and the preference seam is directory-level: it answers that
    /// pair with ONE snapshot path and can never re-ask "which of you is asking" afterwards. Since
    /// no bundle holds the union of both selections' files, the pair is served to NOBODY. Serving
    /// q4 alone is exactly the defect — it would put the q8 load inside the q4 bundle root.
    #[test]
    fn a_pair_one_sibling_selection_does_not_cover_is_served_to_neither() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "q4model", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "q8model", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = |id: &str, variant: &str, glob: &str| {
            json!({
                "id": id,
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/matrix",
                                "variant": variant, "default": true, "files": [glob] }]
            })
        };
        // ONE job carrying BOTH selections of the same repository+revision.
        let payload = json!({
            "modelManifestEntry": entry("q4model", "q4", "q4/*"),
            "modelManifestEntries": [entry("q8model", "q8", "q8/*")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&library, || {
            let q4_bundle = publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let q4_root = q4_bundle.location.root().to_path_buf();

            // Control: ALONE, the q4 selection is served from the bundle. This is what makes the
            // assertion below about the union rather than about the bundle being unusable.
            let alone = json!({ "modelManifestEntry": entry("q4model", "q4", "q4/*") })
                .as_object()
                .unwrap()
                .clone();
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &alone, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the q4 selection alone IS served locally"
            );
            drop(guard);

            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 2);
            assert!(
                guard
                    .resolutions()
                    .iter()
                    .all(|resolution| resolution.availability != ModelAvailability::LocalReady),
                "neither selection may be served when no bundle covers their shared pair: {:?}",
                guard
                    .resolutions()
                    .iter()
                    .map(|resolution| &resolution.availability)
                    .collect::<Vec<_>>()
            );
            // The load-bearing half, at the path layer with the guard held: NOTHING resolves into
            // the q4 bundle, so the q8 load cannot land in a root that has no `q8/` subtree.
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/matrix")
                    .expect("source snapshot");
            assert!(!loaded.starts_with(&q4_root), "{}", loaded.display());
            assert!(loaded.starts_with(&library), "{}", loaded.display());
            assert!(loaded.join("q4/model.safetensors").is_file());
            assert!(loaded.join("q8/model.safetensors").is_file());
        });
    }

    /// A requirement whose receipt carries NO `snapshotRevision` makes the WHOLE repository
    /// unserveable, not merely that one requirement.
    ///
    /// A legacy receipt yields `revision: None` (`artifact_selection.rs` maps a missing
    /// `snapshotRevision` straight through), and there is then no pair to compare coverage against
    /// — no bundle can be *shown* to hold what that requirement needs. Leaving the pinned sibling
    /// served anyway is not safe: the unpinned model re-resolves to the source tier, and its
    /// unpinned load walks `refs/main` into the overlay, which would hand it a bundle whose file
    /// set was never in the union.
    #[test]
    fn a_requirement_with_no_recorded_revision_makes_its_whole_repository_unserveable() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/shared", REV_A, "pinned.safetensors");
        seed_snapshot(&library, "guardfx/shared", REV_A, "legacy.safetensors");
        // `refs/main` is what an unpinned resolve reads — the walk that would reach the overlay.
        let repo_root = library.join("models--guardfx--shared");
        std::fs::create_dir_all(repo_root.join("refs")).unwrap();
        std::fs::write(repo_root.join("refs").join("main"), REV_A).unwrap();

        let pinned_receipt = json!({ "repo": "guardfx/shared", "modelId": "pinned",
                                     "resolvedFiles": ["pinned.safetensors"],
                                     "snapshotRevision": REV_A });
        // The legacy receipt: same repository, NO `snapshotRevision`.
        let legacy_receipt = json!({ "repo": "guardfx/shared", "modelId": "legacy",
                                     "resolvedFiles": ["legacy.safetensors"] });
        let pinned_entry = json!({
            "id": "pinned",
            "downloads": [{ "provider": "huggingface", "repo": "guardfx/shared",
                            "revision": REV_A, "files": ["pinned.safetensors"] }]
        });
        // The legacy carrier declares NO revision, so nothing overrides its receipt's `None` with a
        // declared-exact pin — this is what makes `revision: None` actually reach the guard.
        let legacy_entry = json!({
            "id": "legacy",
            "downloads": [{ "provider": "huggingface", "repo": "guardfx/shared",
                            "files": ["legacy.safetensors"] }]
        });

        // Control: the pinned selection ALONE is served locally. Without this the assertion below
        // could pass merely because the bundle was unusable.
        write_receipts(&data, "guardfx/shared", json!([pinned_receipt.clone()]));
        let settings = settings(data.clone());
        let alone = json!({ "modelManifestEntry": pinned_entry.clone() })
            .as_object()
            .unwrap()
            .clone();
        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/shared",
                REV_A,
                "default",
                &["pinned.safetensors"],
            );
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &alone, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "control: the pinned selection alone IS served locally"
            );
            drop(guard);
        });

        // Now add the legacy carrier on the SAME repository.
        write_receipts(
            &data,
            "guardfx/shared",
            json!([pinned_receipt, legacy_receipt]),
        );
        let payload = json!({
            "modelManifestEntry": pinned_entry,
            "modelManifestEntries": [legacy_entry]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&library, || {
            let bundle_root = {
                let scan = sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore::valid_local_artifacts(
                    &settings.data_dir,
                );
                scan.artifacts[0].location.root().to_path_buf()
            };
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert!(
                guard
                    .resolutions()
                    .iter()
                    .all(|resolution| resolution.availability != ModelAvailability::LocalReady),
                "an unrevisioned requirement poisons the whole repository: {:?}",
                guard
                    .resolutions()
                    .iter()
                    .map(|resolution| &resolution.availability)
                    .collect::<Vec<_>>()
            );
            // The load-bearing half, at the PATH layer with the guard held: neither the pinned nor
            // the unpinned resolve may reach the bundle.
            let pinned = crate::model_jobs::huggingface_pinned_snapshot_dir(
                &settings.data_dir,
                "guardfx/shared",
                REV_A,
            )
            .expect("source snapshot");
            assert!(!pinned.starts_with(&bundle_root), "{}", pinned.display());
            assert!(pinned.starts_with(&library), "{}", pinned.display());
            // The unpinned walk (`refs/main`) is the one the poison exists to stop.
            let unpinned =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/shared")
                    .expect("source snapshot");
            assert!(
                !unpinned.starts_with(&bundle_root),
                "an unpinned load must never walk refs/main into a bundle whose files were never \
                 in the union: {}",
                unpinned.display()
            );
            assert!(unpinned.join("legacy.safetensors").is_file());
        });
    }

    /// Run `body` with `tracing` captured and return everything it emitted.
    fn captured_events<T>(body: impl FnOnce() -> T) -> (T, String) {
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        // Concurrent subscriber-less tests also drive these callsites; without the floor their
        // first hit can cache `Interest::never` and silently empty this capture (see test_env).
        crate::test_env::install_tracing_interest_floor();

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .finish();
        let value = tracing::subscriber::with_default(subscriber, body);
        let logged = String::from_utf8_lossy(
            &capture
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
        .into_owned();
        (value, logged)
    }

    /// A published, verified bundle whose LAYOUT the shared snapshot resolvers cannot be handed is
    /// reported as the local-tier-unsupported class and the model stays on the source tier.
    ///
    /// This class was previously unreachable for exactly the case it names: the scan dropped
    /// layout-unsupported entries with a bare `continue` before the guard ever saw them, so nothing
    /// could emit it. The event is what tells a user their local copy will never serve.
    #[test]
    fn an_unsupported_bundle_shape_is_reported_and_the_model_stays_on_the_source_tier() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            // Published and complete, but stored FLAT rather than in the source-library layout.
            publish_bundle_at(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
                PathBuf::new(),
            );

            let (guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });

            assert!(
                logged.contains("resolved_cache_local_tier_unsupported"),
                "the unsupported shape must be reported: {logged}"
            );
            assert!(
                logged.contains("source-library layout"),
                "and it must name WHY: {logged}"
            );
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady,
                "an unsupported shape must never serve the load"
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(loaded.starts_with(&library), "{}", loaded.display());
        });
    }

    /// "Where did this load's weights come from" is answered once per job, before any loader runs,
    /// for BOTH tiers — and the coverage refusal is reported as its OWN class rather than borrowing
    /// the unsupported-shape label, which describes a different fault entirely.
    #[test]
    fn the_serving_tier_is_reported_for_every_admitted_model() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        for repo in ["guardfx/bundled", "guardfx/external"] {
            seed_snapshot(&library, repo, REV_A, "model.safetensors");
            write_receipts(
                &data,
                repo,
                json!([{ "repo": repo, "modelId": repo,
                         "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
            );
        }
        let settings = settings(data);
        let entry = |repo: &str| {
            json!({
                "id": repo,
                "downloads": [{ "provider": "huggingface", "repo": repo, "revision": REV_A,
                                "files": ["model.safetensors"] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("guardfx/bundled"),
            "modelManifestEntries": [entry("guardfx/external")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&library, || {
            // Only ONE of the two models is bundled locally, so one job reports both tiers. They
            // are different repositories, so neither poisons the other's pair.
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/bundled",
                REV_A,
                "default",
                &["model.safetensors"],
            );

            let (guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });

            assert!(
                logged.contains("model_source_tier_selected"),
                "every admitted model must report its tier: {logged}"
            );
            assert!(
                logged.contains("resolved_local"),
                "the bundled model must report the app-owned tier: {logged}"
            );
            assert!(
                logged.contains("source_library"),
                "the unbundled model must report the source tier: {logged}"
            );
            assert!(
                logged.contains("guardfx/bundled") && logged.contains("guardfx/external"),
                "each report must name its repository: {logged}"
            );
            // The unsupported-shape class describes a bundle that cannot be served at all; nothing
            // here is unsupported, so borrowing that label would be a lie.
            assert!(
                !logged.contains("resolved_cache_local_tier_unsupported"),
                "an ordinary source-tier model is not an unsupported shape: {logged}"
            );

            let tiers: Vec<_> = guard
                .resolutions()
                .iter()
                .map(|resolution| resolution.availability.clone())
                .collect();
            assert!(tiers.contains(&ModelAvailability::LocalReady), "{tiers:?}");
            assert!(
                tiers.contains(&ModelAvailability::ExternalReady),
                "{tiers:?}"
            );
            guard.finish_success().unwrap();
        });
    }

    /// A coverage refusal is reported as its own class. It is not an unsupported shape — the bundle
    /// is perfectly well-formed; it simply does not hold everything THIS job needs from the pair.
    #[test]
    fn a_coverage_refusal_is_reported_as_its_own_class_not_as_an_unsupported_shape() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q8/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "q4model", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "q8model", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_A }
            ]),
        );
        let settings = settings(data);
        let entry = |id: &str, variant: &str, glob: &str| {
            json!({
                "id": id,
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/matrix",
                                "variant": variant, "default": true, "files": [glob] }]
            })
        };
        let payload = json!({
            "modelManifestEntry": entry("q4model", "q4", "q4/*"),
            "modelManifestEntries": [entry("q8model", "q8", "q8/*")]
        })
        .as_object()
        .unwrap()
        .clone();

        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/matrix",
                REV_A,
                "q4",
                &["q4/model.safetensors"],
            );
            let (_guard, logged) = captured_events(|| {
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap()
            });
            assert!(
                logged.contains("resolved_cache_local_tier_not_selected"),
                "the coverage refusal must be reported: {logged}"
            );
            assert!(
                !logged.contains("resolved_cache_local_tier_unsupported"),
                "a well-formed bundle that merely does not cover the job is NOT an unsupported \
                 shape: {logged}"
            );
        });
    }

    /// The resolved cache is opt-in: with it switched off no entry is read, no session is opened,
    /// and every runtime keeps reading the configured source library byte for byte.
    #[test]
    fn a_disabled_cache_never_prefers_a_local_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            publish_bundle(
                &settings.data_dir,
                &library,
                "guardfx/model",
                REV_A,
                "default",
                &["model.safetensors"],
            );
        });
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert!(crate::model_jobs::huggingface_snapshot_dir(
                &settings.data_dir,
                "guardfx/model"
            )
            .expect("source snapshot")
            .starts_with(&library));
        });
    }

    /// sc-19706 — THE test the reopened story exists for: production populates the cache.
    ///
    /// Nothing here publishes a bundle by hand. A job resolves against the configured source
    /// library, runs, and finishes; the trigger records the closure; the idle drain builds the
    /// candidate and materializes it; and a SECOND guard pass over the SAME payload resolves the
    /// model from the app-owned tier, with the shared snapshot resolver answering from the bundle.
    /// That whole chain is the epic's user-visible value, and before this story none of it ran
    /// outside a test that published the bundle itself.
    ///
    /// It also pins the two-stage split: the bundle must NOT exist when the job finishes. If
    /// `finish_success` ever materialized inline, the first assertion below fails — which is the
    /// only way to catch source-library I/O silently moving onto the job path.
    #[test]
    fn a_successful_source_tier_job_promotes_itself_and_the_next_job_loads_from_the_bundle() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_local_cache(&library, || {
            // FIRST JOB — nothing is cached, so it is served by the configured source library.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            let source_snapshot =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("source snapshot");
            assert!(source_snapshot.starts_with(&library));

            guard.finish_success().unwrap();

            // The job path did the queueing and NOTHING else: no bundle, no store, no copy.
            assert!(
                crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "a successful source-tier load must record a promotion candidate"
            );
            assert!(
                !settings
                    .data_dir
                    .join("models")
                    .join("resolved")
                    .join("entries")
                    .is_dir(),
                "finish_success must not materialize inline — that is source-library I/O on the \
                 job path"
            );

            // IDLE DRAIN — what the worker's poll loop runs on the blocking pool when a claim
            // returns nothing.
            crate::resolved_cache_promotion::drain_intake(&settings);
            assert!(
                !crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "the drain consumes the intake"
            );

            let store = ResolvedCacheStore::open(&settings.data_dir).unwrap();
            let published = store
                .enumerate()
                .unwrap()
                .into_iter()
                .filter_map(|entry| entry.metadata)
                .find(|metadata| {
                    metadata.state
                        == sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheEntryState::Complete
                })
                .expect("the drain published a complete bundle");
            assert_eq!(published.artifact.identity.repository, "guardfx/model");
            assert_eq!(published.artifact.identity.revision, REV_A);

            // SECOND JOB — same payload, same worker, nothing else changed. It must now be served
            // from the app-owned tier.
            let second =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                second.resolutions()[0].availability,
                ModelAvailability::LocalReady,
                "the bundle production just wrote must serve the next load"
            );
            let loaded =
                crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, "guardfx/model")
                    .expect("leased local snapshot");
            // Compared against the published bundle root rather than a `starts_with` on
            // `data_dir`: the store records the CANONICAL root, and on macOS the temp dir's
            // `/var` -> `/private/var` symlink makes a lexical prefix check fail on a path that is
            // in fact inside the bundle.
            assert_eq!(
                loaded,
                published
                    .artifact
                    .location
                    .root()
                    .join(format!("models--guardfx--model/snapshots/{REV_A}")),
                "the shared snapshot resolver must answer from the bundle"
            );
            assert!(loaded.join("model.safetensors").is_file());
            second.finish_success().unwrap();

            // A LOCALLY served job records NOTHING. This is the qualification rule asserted in the
            // widening direction, which the `Complete == 1` check below cannot do: relaxing the
            // filter to record local-ready closures too would still leave exactly one complete
            // entry, because the redundant candidate is admitted as `AlreadyComplete`. Only the
            // intake being empty distinguishes "never recorded" from "recorded and then declined",
            // and the difference is a whole idle turn plus a full source-library closure walk
            // burned on a bundle that is already published.
            assert!(
                !crate::resolved_cache_promotion::work_pending(&settings.data_dir),
                "a job served from the local tier is already promoted and must record nothing"
            );

            // And the already-published entry is not promoted a second time: the drain admits it
            // as already complete and copies nothing.
            crate::resolved_cache_promotion::drain_intake(&settings);
            assert_eq!(
                store
                    .enumerate()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.state
                        == sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheEntryState::Complete)
                    .count(),
                1
            );
        });
    }

    #[test]
    fn installed_model_admits_as_external_ready_with_a_source_session() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(guard.resolutions().len(), 1);
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            guard.finish_success().unwrap();
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(std::fs::read_dir(sessions).unwrap().count(), 0);
        });
    }

    #[test]
    fn disconnected_library_fails_typed_and_receipts_survive_untouched() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let receipt_path = settings.data_dir.join("models").join(
                sceneworks_core::model_artifacts::artifact_selection::safe_download_dir(
                    "guardfx/model",
                ),
            );
            let receipt_before =
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap();
            // Bind first while connected, then disconnect the volume.
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            std::fs::rename(&library, temp.path().join("detached")).unwrap();

            let error = RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap_err();
            assert!(matches!(error, WorkerError::ExternalLibraryUnavailable(_)));
            assert!(error
                .to_string()
                .contains(EXTERNAL_LIBRARY_UNAVAILABLE_CODE));
            // Installed state is never lost: receipts are byte-identical after the disconnect.
            assert_eq!(
                std::fs::read(receipt_path.join(".sceneworks-download-complete.json")).unwrap(),
                receipt_before
            );

            // Reconnect: the same payload admits again under the original binding.
            std::fs::rename(temp.path().join("detached"), &library).unwrap();
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        });
    }

    #[test]
    fn component_removal_while_connected_is_incomplete_never_the_disconnect_class() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                .unwrap()
                .finish_success()
                .unwrap();
            let safe = sceneworks_core::hf_home::safe_repo_dir_name("guardfx/model").unwrap();
            std::fs::remove_file(
                library
                    .join(format!("models--{safe}"))
                    .join("snapshots")
                    .join(REV_A)
                    .join("model.safetensors"),
            )
            .unwrap();
            // The library is provably present; the missing component makes the install stale, and
            // stale keeps its established on-demand install/repair semantics — it must NOT read
            // as a library disconnect.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn model_routes_fail_closed_without_carriers_and_no_model_routes_pass() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        with_library(&library, || {
            let empty = JsonObject::new();
            for model_backed in [
                JobType::ImageGenerate,
                JobType::DatasetUpscale,
                JobType::PromptRefine,
                JobType::CatalogAnalysis,
            ] {
                let error =
                    RuntimeSourceGuard::begin(&model_backed, &empty, &settings).unwrap_err();
                assert!(
                    matches!(error, WorkerError::InvalidPayload(_)),
                    "{model_backed:?} must fail closed on an empty carrier set"
                );
            }
            for model_free in [
                JobType::Placeholder,
                JobType::FrameExtract,
                JobType::TimelineExport,
                JobType::DatasetParquetImport,
            ] {
                RuntimeSourceGuard::begin(&model_free, &empty, &settings).unwrap();
            }
        });
    }

    #[test]
    fn completeness_is_judged_on_the_selected_tier_not_a_variant_union() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        // q4 fully installed; q8 receipt exists but its library snapshot was pruned.
        seed_snapshot(&library, "guardfx/matrix", REV_A, "q4/model.safetensors");
        write_receipts(
            &data,
            "guardfx/matrix",
            json!([
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q4",
                  "resolvedFiles": ["q4/model.safetensors"], "snapshotRevision": REV_A },
                { "repo": "guardfx/matrix", "modelId": "m", "variant": "q8",
                  "resolvedFiles": ["q8/model.safetensors"], "snapshotRevision": REV_B }
            ]),
        );
        let settings = settings(data);
        let entry = json!({
            "id": "m",
            "downloads": [
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q4",
                  "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "guardfx/matrix", "variant": "q8",
                  "files": ["q8/*"] }
            ]
        });
        let payload = |variant: &str| {
            json!({ "modelManifestEntry": entry, "variant": variant })
                .as_object()
                .unwrap()
                .clone()
        };
        with_library(&library, || {
            // The selected q4 tier is complete even though the q8 sibling receipt cannot
            // validate — a model-wide union of variant receipts would wrongly demote this to
            // incomplete.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q4"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            assert_eq!(guard.resolutions()[0].requirements.len(), 1);
            assert_eq!(guard.resolutions()[0].requirements[0].variant, "q4");
            drop(guard);
            // And the pruned q8 selection reads incomplete — the q4 install must not mask it.
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload("q8"), &settings)
                    .unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Incomplete
            );
        });
    }

    #[test]
    fn requirements_use_this_workers_platform_closure_not_another_hosts() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        let this_os = std::env::consts::OS;
        let other_os = if this_os == "macos" {
            "windows"
        } else {
            "macos"
        };
        seed_snapshot(&library, "guardfx/native", REV_A, "model.safetensors");
        write_receipts(
            &data,
            "guardfx/native",
            json!([{ "repo": "guardfx/native", "modelId": "m",
                     "resolvedFiles": ["model.safetensors"], "snapshotRevision": REV_A }]),
        );
        // The other platform's primary and co-requisite are NOT installed anywhere. If the guard
        // used another host's platform closure (or no platform filter), admission would fail.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [
                    { "provider": "huggingface", "repo": "guardfx/native", "revision": REV_A,
                      "files": ["model.safetensors"], "platforms": [this_os] },
                    { "provider": "huggingface", "repo": "guardfx/foreign", "revision": REV_B,
                      "files": ["foreign.safetensors"], "platforms": [other_os] },
                    { "provider": "huggingface", "repo": "guardfx/foreign-corequisite",
                      "revision": REV_B, "coRequisite": true,
                      "files": ["encoder.safetensors"], "platforms": [other_os] }
                ]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let settings = settings(data);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
            let repositories = guard.resolutions()[0]
                .requirements
                .iter()
                .map(|requirement| requirement.repository.as_str())
                .collect::<Vec<_>>();
            assert_eq!(repositories, ["guardfx/native"]);
        });
    }

    #[test]
    fn terminal_remap_occurs_only_when_the_exact_source_probe_proves_disconnect() {
        let temp = TempDir::new().unwrap();
        let (settings, library, payload) = installed_model(&temp);
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            let preserved = guard.classify_failure(
                &settings,
                WorkerError::Engine("real loader defect".to_owned()),
            );
            assert!(
                matches!(preserved, WorkerError::Engine(ref value) if value == "real loader defect"),
                "an engine failure with the source present must be preserved verbatim"
            );

            std::fs::rename(&library, temp.path().join("detached")).unwrap();
            let remapped = guard.classify_failure(
                &settings,
                WorkerError::Engine("No such file or directory (os error 2)".to_owned()),
            );
            assert!(matches!(
                remapped,
                WorkerError::ExternalLibraryUnavailable(_)
            ));
            assert!(
                !remapped.to_string().contains("os error 2"),
                "a proven disconnect must never surface as a raw ENOENT"
            );
            drop(guard);
            let sessions = settings
                .data_dir
                .join("models/.sceneworks-external-source-sessions");
            assert_eq!(
                std::fs::read_dir(sessions).unwrap().count(),
                0,
                "failure cleanup removes only the operation-owned source session"
            );
        });
    }

    #[test]
    fn never_installed_model_keeps_its_on_demand_install_path() {
        let temp = TempDir::new().unwrap();
        let data = temp.path().join("data");
        let library = temp.path().join("external-hf");
        std::fs::create_dir_all(&library).unwrap();
        let settings = settings(data);
        // Manifest present, but no receipt and glob files (no exact declaration): no durable
        // install identity, so availability is Missing and the job proceeds to its established
        // download path rather than failing typed.
        let payload = json!({
            "modelManifestEntry": {
                "id": "m",
                "downloads": [{ "provider": "huggingface", "repo": "guardfx/new",
                                "revision": REV_A, "files": ["q4/*"] }]
            }
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::Missing
            );
        });
    }

    #[test]
    fn explicit_download_is_exempt_only_for_download_job_types() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let payload = json!({
            "modelArtifactOperation": { "schemaVersion": 1, "kind": "explicit_download" }
        })
        .as_object()
        .unwrap()
        .clone();
        RuntimeSourceGuard::begin(&JobType::ModelDownload, &payload, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::LoraDownload, &payload, &settings).unwrap();
        assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).is_err());
    }

    #[test]
    fn only_confined_download_free_local_entries_may_use_the_non_hf_exception() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let imported = settings.data_dir.join("models/imports/custom");
        std::fs::create_dir_all(&imported).unwrap();
        std::fs::write(imported.join("model.safetensors"), b"weights").unwrap();
        let local = json!({
            "modelManifestEntry": {
                "id": "imported",
                "downloads": [],
                "paths": { "model": imported }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let outside = temp.path().join("outside/model.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"outside").unwrap();
        let escaped = json!({
            "modelManifestEntry": {
                "id": "forged-local",
                "downloads": [],
                "paths": { "model": outside }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let bare = json!({ "modelManifestEntry": { "id": "bare" } })
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &local, &settings).unwrap();
            assert!(
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &escaped, &settings).is_err()
            );
            // No downloads AND no paths proves nothing: fail closed.
            assert!(RuntimeSourceGuard::begin(&JobType::ImageGenerate, &bare, &settings).is_err());
        });
    }

    /// The FULL shape the model-import job writes (`apps/rust-api/src/models.rs`): an absolute
    /// `paths.model` install dir PLUS a `source` PROVENANCE block carrying a data-dir-relative
    /// `path`, a `provider` literal, a `repo` that is null for a file import, and a `url`.
    fn imported_entry(data_dir: &Path, id: &str) -> Value {
        let relative = format!("models/imports/{id}");
        let install = data_dir.join(&relative);
        std::fs::create_dir_all(&install).unwrap();
        std::fs::write(install.join("model.safetensors"), b"weights").unwrap();
        json!({
            "id": id,
            "name": id,
            "type": "image",
            "family": "krea_2",
            "downloads": [],
            "source": {
                "provider": "local",
                "repo": Value::Null,
                "path": relative,
                "url": "https://example.invalid/kreamania.safetensors",
            },
            "files": ["model.safetensors"],
            "paths": { "model": install },
        })
    }

    /// sc-20524 — a `source` block is provenance, not a path set. `provider` ("local"), `repo`
    /// and `url` are identifiers; only `source.path` names a location, and it is written
    /// data-dir-relative. Confining every `source` value as a filesystem path resolved `"local"`
    /// against the process cwd, so EVERY user-imported model failed the pre-loader guard with
    /// "non-HF model source must be inside an app-managed directory".
    #[test]
    fn imported_entry_source_provenance_is_not_confined_as_a_path_set() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let payload = json!({
            "model": "kreamania_v1",
            "modelManifestEntry": imported_entry(&settings.data_dir, "kreamania_v1"),
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
        });
    }

    /// The whole reported population: every imported KreaMania checkpoint must clear the guard.
    #[test]
    fn every_imported_kreamania_checkpoint_clears_the_non_hf_guard() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        with_library(&library, || {
            for id in [
                "kreamania_v1",
                "kreamania_v2",
                "kreamania_v3",
                "kreamania_v4_int8",
                "kreamania_v5",
                "kreamania_v6",
            ] {
                let payload = json!({
                    "model": id,
                    "modelManifestEntry": imported_entry(&settings.data_dir, id),
                })
                .as_object()
                .unwrap()
                .clone();
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings)
                    .unwrap_or_else(|error| panic!("{id} must clear the guard: {error}"));
            }
        });
    }

    /// The real manifest written on Windows carries `paths.model` in VERBATIM (`\\?\C:\...`)
    /// form — the model-import job canonicalizes the install dir before recording it. That is
    /// the exact user-reported shape (kreamania_v5/v6), so pin it verbatim.
    #[cfg(windows)]
    #[test]
    fn imported_entry_with_verbatim_install_path_clears_the_guard() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let mut entry = imported_entry(&settings.data_dir, "kreamania_v5");
        let install = settings.data_dir.join("models/imports/kreamania_v5");
        let verbatim = std::fs::canonicalize(&install).unwrap();
        assert!(
            verbatim.to_string_lossy().starts_with(r"\\?\"),
            "canonicalize must produce the verbatim form this test exists to pin"
        );
        entry["paths"]["model"] = json!(verbatim);
        let payload = json!({ "model": "kreamania_v5", "modelManifestEntry": entry })
            .as_object()
            .unwrap()
            .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap_or_else(
                |error| panic!("verbatim install path must clear the guard: {error}"),
            );
        });
    }

    /// Reading only `source.path` must not become "read nothing": a `source.path` that escapes
    /// the app-managed roots is still a rejection, and a benign-looking `source` block never
    /// launders an entry whose real `paths` point outside.
    #[test]
    fn source_path_is_still_confined_and_cannot_launder_an_outside_entry() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("model.safetensors"), b"outside").unwrap();
        let carrier = |entry: Value| {
            json!({ "modelManifestEntry": entry })
                .as_object()
                .unwrap()
                .clone()
        };
        // `..` out of the data dir through the relative locator.
        let escaping_relative = carrier(json!({
            "id": "escaping-relative",
            "downloads": [],
            "source": { "provider": "local", "path": "../outside" },
        }));
        // An absolute `source.path` outside every root.
        let escaping_absolute = carrier(json!({
            "id": "escaping-absolute",
            "downloads": [],
            "source": { "provider": "local", "path": outside },
        }));
        // Provenance that says "local" while the actual weights live outside.
        let laundered = carrier(json!({
            "id": "laundered",
            "downloads": [],
            "source": { "provider": "local", "repo": Value::Null, "path": "models/imports/ok" },
            "paths": { "model": outside },
        }));
        with_library(&library, || {
            for (label, payload) in [
                ("escaping relative source.path", &escaping_relative),
                ("absolute outside source.path", &escaping_absolute),
                ("outside paths.model", &laundered),
            ] {
                // Assert the CONFINEMENT rejection specifically. `is_err()` alone would stay green
                // if a future change stopped reading these locators at all: the locator set would
                // go empty, the entry would fail the "no supported artifact source" branch, and the
                // test would pass while the mechanism it exists to pin had been removed.
                assert_confinement_rejection(
                    label,
                    RuntimeSourceGuard::begin(&JobType::ImageGenerate, payload, &settings),
                );
            }
        });
    }

    #[track_caller]
    fn assert_confinement_rejection(label: &str, outcome: WorkerResult<RuntimeSourceGuard>) {
        match outcome {
            Err(WorkerError::InvalidPayload(message))
                if message.contains("must be inside an app-managed directory") =>
            {
                assert!(
                    message.starts_with("non-HF model source"),
                    "{label} must be rejected by the non-HF locator guard, got: {message}"
                );
            }
            Err(error) => panic!("{label} must fail confinement, not something else: {error}"),
            Ok(_) => panic!("{label} must still be rejected"),
        }
    }

    /// Windows spells two path forms that `Path::is_absolute` calls relative while `PathBuf::join`
    /// refuses to anchor: root-relative (`\Windows\System32` — join keeps only the data dir's drive
    /// prefix) and drive-relative (`D:models` — join replaces the data dir outright). Neither means
    /// anything relative to the data dir, so `source.path` must refuse both rather than confine a
    /// path the anchoring never produced. On Unix both strings are ordinary single-component
    /// filenames and genuinely do anchor, so this is Windows-only by construction.
    #[cfg(windows)]
    #[test]
    fn windows_root_relative_and_drive_relative_source_paths_are_refused() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        let drive_relative = format!(
            "{}models",
            settings
                .data_dir
                .components()
                .next()
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .expect("a Windows temp dir carries a drive prefix")
        );
        with_library(&library, || {
            for path in [r"\Windows\System32", drive_relative.as_str()] {
                let payload = json!({
                    "modelManifestEntry": {
                        "id": "windows-relative",
                        "downloads": [],
                        "source": { "provider": "local", "path": path },
                    }
                })
                .as_object()
                .unwrap()
                .clone();
                match RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings) {
                    Err(WorkerError::InvalidPayload(message)) => assert!(
                        message.contains("relative to the app data directory"),
                        "{path} must be refused as unanchorable, got: {message}"
                    ),
                    Err(error) => panic!("{path} must be refused as unanchorable: {error}"),
                    Ok(_) => panic!("{path} must not be admitted"),
                }
            }
        });
    }

    /// A LOADER locator must be judged as the loader will re-resolve it. Every consumer of
    /// `modelPath` / `installedPath` / `paths.*` / `components[].path` re-reads the raw string and
    /// anchors a relative one on the process cwd (`paths::normalize_absolute_path`,
    /// `model_artifacts::canonical_or_absolute`), so anchoring these on the data dir here would
    /// make the guard prove a different path than the one that gets opened — admitting a locator
    /// that names nothing (`totally/made/up/never/created`) and "proving confined" an unexpanded
    /// template literal by planting it under the data dir. Only `source.path`, which nothing opens,
    /// is data-dir anchored.
    #[test]
    fn relative_loader_locators_are_judged_where_the_loader_will_resolve_them() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let library = temp.path().join("external-hf");
        // Anchored on the data dir both of these look like plausible children of an allowed root;
        // anchored where the loader anchors them (the worker's cwd) neither is confined.
        with_library(&library, || {
            for locator in ["totally/made/up/never/created", "${HF_CACHE}/models/model"] {
                let payload = json!({
                    "modelManifestEntry": {
                        "id": "relative-locator",
                        "downloads": [],
                        "paths": { "model": locator },
                    }
                })
                .as_object()
                .unwrap()
                .clone();
                assert_confinement_rejection(
                    locator,
                    RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings),
                );
            }
        });
    }

    /// Confinement posture for the two neighbouring classes is unchanged: an HF-backed entry is
    /// judged by its downloads (its `source` provenance is never confined at all), and an entry
    /// under a declared external root still passes on its real paths.
    #[test]
    fn hf_backed_and_external_root_entries_keep_their_posture() {
        let temp = TempDir::new().unwrap();
        let (settings, library, mut payload) = installed_model(&temp);
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        // Provenance on an HF-backed entry never reaches the non-HF confinement branch.
        payload["modelManifestEntry"]["source"] =
            json!({ "provider": "huggingface", "repo": "guardfx/model", "path": outside });
        with_library(&library, || {
            let guard =
                RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &settings).unwrap();
            assert_eq!(
                guard.resolutions()[0].availability,
                ModelAvailability::ExternalReady
            );
        });

        let comfy = temp.path().join("comfy/models/checkpoints");
        std::fs::create_dir_all(&comfy).unwrap();
        let mut external = settings;
        external.external_model_roots = vec![temp.path().join("comfy")];
        let payload = json!({
            "modelManifestEntry": {
                "id": "comfy",
                "downloads": [],
                "paths": { "model": comfy },
            }
        })
        .as_object()
        .unwrap()
        .clone();
        with_library(&library, || {
            RuntimeSourceGuard::begin(&JobType::ImageGenerate, &payload, &external).unwrap();
        });
    }

    #[test]
    fn training_dry_run_is_model_free_but_a_real_run_is_not() {
        let temp = TempDir::new().unwrap();
        let settings = settings(temp.path().join("data"));
        let dry = json!({ "dryRun": true }).as_object().unwrap().clone();
        RuntimeSourceGuard::begin(&JobType::LoraTrain, &dry, &settings).unwrap();
        RuntimeSourceGuard::begin(&JobType::ControlTraining, &dry, &settings).unwrap();
        let real = JsonObject::new();
        assert!(RuntimeSourceGuard::begin(&JobType::LoraTrain, &real, &settings).is_err());
    }
}
