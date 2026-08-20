//! sc-19706 — promotion activation: the idle half of the two-stage producer.
//!
//! sc-19704 froze the artifact contract, sc-19705 the store, sc-19706 the materializer and its
//! scheduler, sc-19708 the one selection module that reduces a manifest entry to an exact
//! requirement closure, and `model_artifacts::promotion` joined them into a candidate producer.
//! None of that ran in production: `<data_dir>/models/resolved/` was populated only by tests. This
//! module is what makes a real install promote itself.
//!
//! # Two stages, and why the split is load-bearing
//!
//! [`RuntimeSourceGuard::finish_success`](crate::external_library_runtime) records a
//! [`PendingPromotion`] — a `Vec` of values the guard already had — into a bounded in-memory
//! intake. That is the ENTIRE cost on the job path: one mutex, one move, no filesystem.
//!
//! [`drain_intake`] does the expensive half, on the blocking pool, only when the worker's claim
//! poll returned nothing. Building one candidate resolves the whole closure against the source
//! library (dozens of stats and canonicalizes, plus a runtime lease), and materializing one copies
//! and hashes gigabytes. Doing either where the guard runs would put source-library I/O directly in
//! front of the load the user is waiting on — the thing the local tier exists to make faster.
//!
//! # What this module deliberately does not decide
//!
//! Every policy question — is the cache enabled, is the entry already published, does the bundle
//! fit beside the pinned set, is the source still there — belongs to
//! [`ResolvedCachePromotionScheduler`], and is answered at DRAIN time against the world as it is
//! then, not at record time. The intake is a hint, never a decision. There is likewise no
//! per-model, per-route or per-job-type code here: a candidate is built from the generic closure
//! the selection module produced, so a new model is a manifest-data change and nothing else.
//!
//! Nothing here downloads. Promotion copies bytes that are already installed; a source that no
//! longer resolves declines, and no fetch of any kind is reachable from this module.

use crate::{emit_event_value, Settings};
use sceneworks_core::model_artifacts::external_library::ExternalArtifactRequirement;
use sceneworks_core::model_artifacts::promotion::promotion_candidate_for_requirements;
use sceneworks_core::model_artifacts::resolved_cache::{
    IdlePromotionOutcome, PromotionScheduleOutcome, ResolvedCacheMaterializer,
    ResolvedCachePromotionScheduler, ResolvedCacheStore,
};
use sceneworks_core::model_artifacts::{ArtifactSourceLibrary, ModelArtifactResolver};
use serde_json::json;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use tracing::Level;

/// How many closures may wait for a drain per data directory.
///
/// The intake is a hint that a load happened, so dropping the oldest under sustained pressure loses
/// nothing durable: the next job that loads the same model records it again. Bounded rather than
/// unbounded because a worker that never goes idle would otherwise accumulate one entry per job
/// forever.
const INTAKE_CAPACITY: usize = 16;

/// How many admitted candidates the scheduler may hold. Kept small on purpose: each is a whole
/// bundle copy, and the worker runs at most one at a time.
const SCHEDULER_CAPACITY: usize = 4;

/// One source-tier load worth promoting, captured with NO filesystem work.
///
/// It carries the requirement closure rather than a built candidate precisely because building the
/// candidate is the expensive part. `requirements` is the closure
/// `artifact_selection::selected_requirements_for_model` produced for this host and this request's
/// tier — the same value that admitted the model — so the promoted bundle holds exactly what the
/// job loaded, never a variant union.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingPromotion {
    pub(crate) data_dir: PathBuf,
    /// The configured source-library root the closure was admitted against. Recorded rather than
    /// re-derived at drain time so a library reconfigured between the load and the drain cannot
    /// silently promote from a different root than the one that served the job.
    pub(crate) source_configured_path: PathBuf,
    pub(crate) requirements: Vec<ExternalArtifactRequirement>,
    pub(crate) logical_model_owner: String,
}

/// Build the intake records for the closures a job served from the source tier.
///
/// Returns an empty vector when the cache is switched off, so an opt-out install never accumulates
/// promotion state. A closure with no primary requirement is skipped: the promotion producer
/// requires exactly one, and a malformed carrier is not something to discover at drain time.
pub(crate) fn pending_for_plans<'a>(
    settings: &Settings,
    source_configured_path: &Path,
    closures: impl Iterator<Item = &'a [ExternalArtifactRequirement]>,
) -> Vec<PendingPromotion> {
    if !settings.resolved_cache_policy().enabled {
        return Vec::new();
    }
    closures
        .filter_map(|requirements| {
            let owner = requirements
                .iter()
                .find(|requirement| requirement.is_primary)?;
            Some(PendingPromotion {
                data_dir: settings.data_dir.clone(),
                source_configured_path: source_configured_path.to_path_buf(),
                // The primary repository is the stable logical id the store already uses for
                // ownership everywhere else (`acquire_complete` is handed the same value), so a
                // promotion and a later lease of the same bundle agree on who owns it.
                logical_model_owner: owner.repository.clone(),
                requirements: requirements.to_vec(),
            })
        })
        .collect()
}

/// Record closures for a later drain. Lock-and-move only; never touches the filesystem.
///
/// Identical closures coalesce here, before any I/O could reveal that they share a cache key: a job
/// carrying the same model twice, or two jobs draining together, must not both walk the source.
pub(crate) fn record_pending(pending: Vec<PendingPromotion>) {
    if pending.is_empty() {
        return;
    }
    let mut intake = lock_intake();
    for item in pending {
        let queue = intake.entry(item.data_dir.clone()).or_default();
        if queue.contains(&item) {
            continue;
        }
        if queue.len() >= INTAKE_CAPACITY {
            queue.pop_front();
        }
        queue.push_back(item);
    }
}

/// Whether an idle turn has promotion work to do. Reads only in-memory state — it sits directly on
/// the claim path, so it must never open the store or stat anything.
///
/// The scheduler queue is included because one drain materializes at most ONE bundle: a job that
/// admitted three closures needs three idle turns, and reporting "nothing pending" after the first
/// would strand the rest until the next load re-recorded them.
pub(crate) fn work_pending(data_dir: &Path) -> bool {
    if lock_intake()
        .get(data_dir)
        .is_some_and(|queue| !queue.is_empty())
    {
        return true;
    }
    lock_schedulers()
        .get(data_dir)
        .is_some_and(|scheduler| scheduler.queued_len() > 0)
}

/// Drain the intake and run at most one promotion. Blocking; call from `spawn_blocking`.
///
/// Order matters: every recorded closure is turned into a candidate and offered to the scheduler
/// FIRST, and only then does one materialization run. Building candidates is cheap relative to a
/// bundle copy, and admitting them all up front is what lets the scheduler coalesce duplicates and
/// apply its bound before any bytes move.
pub(crate) fn drain_intake(settings: &Settings) {
    let policy = settings.resolved_cache_policy();
    if !policy.enabled {
        // An operator turning the cache off between a load and this drain must not have the
        // recorded hints promoted afterwards.
        lock_intake().remove(&settings.data_dir);
        return;
    }
    let pending: Vec<PendingPromotion> = lock_intake()
        .get_mut(&settings.data_dir)
        .map(|queue| queue.drain(..).collect())
        .unwrap_or_default();
    let Some(scheduler) = scheduler_for(settings) else {
        return;
    };
    for item in pending {
        // The AUTHORITATIVE source library, never the preferring one: promoting a bundle by
        // reading from a bundle would launder an unverified copy into a published entry. No
        // preference scope is installed on an idle worker anyway, so this states the requirement
        // rather than relying on the timing.
        let resolver = ModelArtifactResolver::new(
            match ArtifactSourceLibrary::new(&item.source_configured_path) {
                Ok(library) => library,
                Err(error) => {
                    decline(&item, &error.to_string());
                    continue;
                }
            },
        );
        // Resolves and fully validates the closure against that library. A disconnected or
        // incomplete source produces an error and therefore no candidate — never a fetch.
        let candidate = match promotion_candidate_for_requirements(&resolver, &item.requirements) {
            Ok(candidate) => candidate,
            Err(error) => {
                decline(&item, &error.to_string());
                continue;
            }
        };
        let cache_key = candidate.cache_key.clone();
        match scheduler.schedule(
            candidate,
            item.source_configured_path.clone(),
            item.logical_model_owner.clone(),
        ) {
            Ok(PromotionScheduleOutcome::Enqueued) => emit_event_value(
                Level::INFO,
                json!({
                    "event": "resolved_cache_promotion_scheduled",
                    "cacheKey": cache_key,
                    "repository": item.logical_model_owner,
                }),
            ),
            // Coalesced/Full/Disabled/Stopped are ordinary outcomes, not failures.
            Ok(_) => {}
            Err(error) => decline(&item, &error.to_string()),
        }
    }
    match scheduler.run_next_if_idle(true) {
        Ok(IdlePromotionOutcome::Materialized(outcome)) => emit_event_value(
            Level::INFO,
            json!({
                "event": "resolved_cache_promotion_materialized",
                "outcome": format!("{outcome:?}"),
            }),
        ),
        Ok(IdlePromotionOutcome::Declined(reason)) => emit_event_value(
            Level::INFO,
            json!({
                "event": "resolved_cache_promotion_declined",
                "reason": reason.to_string(),
            }),
        ),
        Ok(_) => {}
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache promotion drain failed")
        }
    }
}

fn decline(item: &PendingPromotion, reason: &str) {
    emit_event_value(
        Level::INFO,
        json!({
            "event": "resolved_cache_promotion_declined",
            "repository": item.logical_model_owner,
            "reason": reason,
        }),
    );
}

/// One long-lived scheduler per data directory.
///
/// It must outlive a single drain: the queue, the coalescing key set and the single active slot all
/// live in it, so a fresh scheduler per idle turn would lose the queue every time and re-materialize
/// whatever a previous turn had already admitted.
///
/// # The memo lock is never held across I/O
///
/// `ResolvedCacheStore::open` takes a durable session lock and writes session records. That lock is
/// therefore taken, dropped, and only re-taken to publish the result — because [`work_pending`]
/// locks the same mutex and runs **on the claim path**, where blocking behind a store open would
/// delay the next job. Today the single maintenance slot means no two callers can reach the I/O at
/// once, so the contention is unreachable; that is a positional accident of the poll loop's shape,
/// not a property of this function, and this ordering makes it structural instead.
///
/// The race that ordering admits — two callers both building a scheduler — resolves **first-wins**:
/// a loser drops its own scheduler and adopts the installed one. Last-wins would replace a
/// scheduler that may already hold queued work AND leave two live "single active slot" owners for
/// one store, which is exactly the concurrent-materialization case the slot exists to prevent.
fn scheduler_for(settings: &Settings) -> Option<ResolvedCachePromotionScheduler> {
    if let Some(scheduler) = lock_schedulers().get(&settings.data_dir) {
        return Some(scheduler.clone());
    }
    // Unlocked: real filesystem work.
    let store = match ResolvedCacheStore::open(&settings.data_dir) {
        Ok(store) => store,
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache store unavailable; promotion is skipped");
            return None;
        }
    };
    let scheduler = match ResolvedCachePromotionScheduler::new(
        settings.resolved_cache_policy(),
        ResolvedCacheMaterializer::new(store),
        SCHEDULER_CAPACITY,
    ) {
        Ok(scheduler) => scheduler,
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache promotion policy is invalid; promotion is skipped");
            return None;
        }
    };
    Some(
        lock_schedulers()
            .entry(settings.data_dir.clone())
            .or_insert(scheduler)
            .clone(),
    )
}

type Intake = BTreeMap<PathBuf, VecDeque<PendingPromotion>>;

fn lock_intake() -> MutexGuard<'static, Intake> {
    static INTAKE: OnceLock<Mutex<Intake>> = OnceLock::new();
    INTAKE
        .get_or_init(|| Mutex::new(Intake::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

type Schedulers = BTreeMap<PathBuf, ResolvedCachePromotionScheduler>;

fn lock_schedulers() -> MutexGuard<'static, Schedulers> {
    static SCHEDULERS: OnceLock<Mutex<Schedulers>> = OnceLock::new();
    SCHEDULERS
        .get_or_init(|| Mutex::new(Schedulers::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(repository: &str, is_primary: bool) -> ExternalArtifactRequirement {
        ExternalArtifactRequirement {
            repository: repository.to_owned(),
            revision: Some("c19706cccccccccccccccccccccccccccccccccc".to_owned()),
            variant: "default".to_owned(),
            files: vec![PathBuf::from("model.safetensors")],
            is_primary,
        }
    }

    fn pending(data_dir: &Path, repository: &str) -> PendingPromotion {
        PendingPromotion {
            data_dir: data_dir.to_path_buf(),
            source_configured_path: data_dir.join("library"),
            requirements: vec![requirement(repository, true)],
            logical_model_owner: repository.to_owned(),
        }
    }

    /// The intake is a HINT, so its bound must drop the oldest rather than refuse the newest: the
    /// most recent load is the one most likely to be loaded again. An unbounded queue on a worker
    /// that never idles would grow one entry per job forever.
    #[test]
    fn the_intake_is_bounded_and_drops_the_oldest_hint() {
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("bounded");
        for index in 0..(INTAKE_CAPACITY + 3) {
            record_pending(vec![pending(&data_dir, &format!("guardfx/model-{index}"))]);
        }
        let intake = lock_intake();
        let queue = intake
            .get(&data_dir)
            .expect("the intake holds this data dir");
        assert_eq!(queue.len(), INTAKE_CAPACITY);
        assert_eq!(
            queue.front().unwrap().logical_model_owner,
            "guardfx/model-3"
        );
        assert_eq!(
            queue.back().unwrap().logical_model_owner,
            format!("guardfx/model-{}", INTAKE_CAPACITY + 2)
        );
    }

    /// Two loads of the same closure must not both walk the source library at drain time. The
    /// scheduler coalesces on cache key, but learning the cache key costs the very I/O the intake
    /// exists to defer, so the identical-closure case has to collapse here.
    #[test]
    fn an_identical_closure_is_recorded_once() {
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("coalesce");
        let item = pending(&data_dir, "guardfx/repeated");
        record_pending(vec![item.clone(), item.clone()]);
        record_pending(vec![item]);
        assert_eq!(lock_intake().get(&data_dir).map(VecDeque::len), Some(1));
    }

    /// A closure with no primary requirement cannot produce a candidate, so it must never reach the
    /// intake — discovering it at drain time would burn a whole idle turn on a guaranteed failure.
    #[test]
    fn a_closure_without_a_primary_is_never_recorded() {
        crate::test_env::temp_env_vars(
            &[(
                sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                "1",
            )],
            || {
                let temp = tempfile::tempdir().expect("tempdir");
                let settings = Settings::for_test(temp.path().join("data"));
                let closure = vec![requirement("guardfx/corequisite-only", false)];
                assert!(pending_for_plans(
                    &settings,
                    &temp.path().join("library"),
                    std::iter::once(closure.as_slice()),
                )
                .is_empty());
            },
        );
    }

    /// An opt-out install must accumulate no promotion state at all: not a bundle, and not the
    /// in-memory record that would promote one the moment the cache was switched on.
    #[test]
    fn a_disabled_cache_records_nothing() {
        crate::test_env::temp_env_vars(
            &[(
                sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                "0",
            )],
            || {
                let temp = tempfile::tempdir().expect("tempdir");
                let settings = Settings::for_test(temp.path().join("data"));
                let closure = vec![requirement("guardfx/disabled", true)];
                assert!(pending_for_plans(
                    &settings,
                    &temp.path().join("library"),
                    std::iter::once(closure.as_slice()),
                )
                .is_empty());
            },
        );
    }

    /// Turning the cache off BETWEEN a load and the drain must discard what the load recorded, and
    /// must not create the managed cache root while doing it.
    #[test]
    fn a_disabled_drain_discards_the_intake_without_creating_the_store() {
        crate::test_env::temp_env_vars(
            &[(
                sceneworks_core::model_artifacts::resolved_cache::RESOLVED_CACHE_ENABLED_ENV,
                "0",
            )],
            || {
                let temp = tempfile::tempdir().expect("tempdir");
                let data_dir = temp.path().join("switched-off");
                std::fs::create_dir_all(&data_dir).expect("data dir creates");
                record_pending(vec![pending(&data_dir, "guardfx/switched-off")]);
                assert!(work_pending(&data_dir));
                let settings = Settings::for_test(data_dir.clone());
                drain_intake(&settings);
                assert!(!work_pending(&data_dir));
                assert!(!data_dir.join("models").join("resolved").exists());
            },
        );
    }
}
