//! Generic single-resident, dedicated-OS-thread model cache (sc-11191, F-019).
//!
//! [`crate::generator_cache`] (image/video `Generator`) and [`crate::refine_model_cache`] (text
//! `core_llm::TextLlm`) previously re-implemented the SAME scaffolding almost verbatim (~250 lines
//! each): a `Fingerprint` content-change proxy, a `panic_message` payload formatter, a dedicated
//! worker thread that owns one resident model keyed by a load spec, an idle-timeout eviction event
//! loop, evict-on-panic containment, and a `oneshot`-reply job seam. This module hoists all of that
//! into one generic [`CacheThread<K, M>`] parameterized over the cache key `K`, the resident model
//! `M`, and a caller-supplied loader closure; the two caches become thin wrappers that supply their
//! key/model types, their loader, their idle-eviction log, and their user-facing message strings.
//!
//! ## Preserved divergence: evict-before-load (do NOT unify away)
//!
//! The two caches deliberately differ in ONE behavior, kept here as the [`CacheThread::new`]
//! `evict_before_load` flag so each retains its CURRENT semantics exactly:
//!
//! - **generator cache — `evict_before_load = false`.** On a cold miss it clears the resident entry
//!   (`self.entry = None`) and then loads; it does NOT free the backend cache before the new load.
//!   The unified-memory fit-gate / residency policy (epic 10834) runs inside the loader closure to
//!   size the load, so an explicit pre-load backend trim is not part of its contract.
//! - **refine cache — `evict_before_load = true`.** The ~16 GB text model frees the prior resident
//!   model's backend allocation ([`release_backend_cache_after_evict`]) BEFORE the new load
//!   allocates, so peak memory is one model, not two.
//!
//! This is a real, intentional difference (not drift); unifying it would change the memory profile of
//! one lane. If a future change makes a single policy correct for both, make that an explicit decision
//! and update both wrappers — do not silently flip the flag here.

use std::path::Path;
use std::sync::{mpsc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use tokio::sync::oneshot;

use crate::{WorkerError, WorkerResult};

/// Lock a process-wide cache, discarding suspect model state after an unwinding job.
/// The reset policy retains each cache's cardinality/container contract (a single
/// `Option` or a factor/path map). Deliberately retain the existing sticky poison:
/// every later acquisition resets again; this helper does not certify the backend
/// healthy or clear its poison. The returned guard keeps the model operation locked.
pub(crate) fn lock_resetting<T>(cache: &Mutex<T>, reset: impl FnOnce(&mut T)) -> MutexGuard<'_, T> {
    cache.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        reset(&mut guard);
        guard
    })
}

/// Reuse one resident entry according to the caller's existing key policy, or load
/// its replacement. Evaluate the fallible loader before dropping the previous
/// entry, so a failed replacement retains it and the next request can retry.
/// No thread-safety bounds are imposed on the resident model.
pub(crate) fn get_or_load<T>(
    slot: &mut Option<T>,
    matches: impl FnOnce(&T) -> bool,
    load: impl FnOnce() -> WorkerResult<T>,
) -> WorkerResult<&mut T> {
    if !slot.as_ref().is_some_and(matches) {
        *slot = Some(load()?);
    }
    Ok(slot.as_mut().expect("cache entry loaded"))
}

/// Multi-resident sibling: retain every other factor/path entry on a miss or
/// failed load, matching the upscaler's existing `HashMap::entry` contract.
pub(crate) fn get_or_load_by_key<K: Eq + std::hash::Hash, T>(
    entries: &mut std::collections::HashMap<K, T>,
    key: K,
    load: impl FnOnce() -> WorkerResult<T>,
) -> WorkerResult<&mut T> {
    use std::collections::hash_map::Entry;
    Ok(match entries.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(load()?),
    })
}

/// Thread-local sibling for `!Send` models. Release the lookup borrow before
/// building, then hold a shared borrow through use, exactly like the prior callers.
#[cfg(any(native_lane, test))]
pub(crate) fn get_or_load_local<T>(
    cell: &std::cell::RefCell<Option<T>>,
    matches: impl FnOnce(&T) -> bool,
    load: impl FnOnce() -> WorkerResult<T>,
) -> WorkerResult<std::cell::Ref<'_, T>> {
    let cache_matches = cell.borrow().as_ref().is_some_and(matches);
    if !cache_matches {
        let value = load()?;
        *cell.borrow_mut() = Some(value);
    }
    Ok(std::cell::Ref::map(cell.borrow(), |slot| {
        slot.as_ref().expect("cache entry loaded")
    }))
}

/// All resident model caches share one user setting and the same five-minute default.
pub(crate) const CACHE_IDLE_SECONDS_ENV: &str = "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS";
pub(crate) const DEFAULT_CACHE_IDLE_SECONDS: u64 = 300;

pub(crate) fn idle_timeout_from_env() -> Option<Duration> {
    idle_timeout_from_secs(
        std::env::var(CACHE_IDLE_SECONDS_ENV).ok().as_deref(),
        DEFAULT_CACHE_IDLE_SECONDS,
    )
}

/// A unit of work handed to a cache worker thread: a boxed closure that runs against the resident
/// [`CacheThread`] on that thread and signals completion through its own captured reply channel. The
/// closure is `Send` (it crosses the channel) but the resident model `M` never leaves the thread, so
/// `M` itself may be `!Send` (e.g. a `Box<dyn TextLlm>` holding an `Rc`).
pub(crate) type CacheJob<K, M> = Box<dyn FnOnce(&mut CacheThread<K, M>) + Send + 'static>;

/// Content-change proxy for a weights/adapter file (or HF snapshot dir) referenced in a cache key
/// (sc-8841, F-039). A pre-fingerprint key identified weights/adapters by path only, so a file
/// replaced at the SAME path — a re-imported LoRA, a re-converted checkpoint — was served from the
/// resident model with the OLD tensors until the idle timeout. Folding `(size, mtime)` into the key
/// self-heals for ANY overwrite path (in-process re-import/re-convert, out-of-band replacement,
/// manual swap) without an explicit evict hook that only fires for the code paths that remember it.
///
/// We deliberately stat metadata rather than hashing contents: mtime+size is a cheap, good
/// content-change proxy, and hashing multi-GB weight files on every request would be a severe perf
/// regression.
///
/// For a `Dir` (an HF snapshot) callers fingerprint the DIRECTORY's own metadata. A re-convert lands
/// via a finalize/rename path that replaces directory entries and bumps the dir's mtime, so a
/// re-converted snapshot at the same path is a distinct key. `size` for a dir is its own metadata
/// length (not the recursive content size) — it only needs to move on change, not be meaningful;
/// mtime carries the signal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Fingerprint {
    /// `metadata()` succeeded: `(len, modified)`. `modified` is `None` on the rare platform/FS that
    /// does not report an mtime, in which case `len` alone carries the (weaker) signal.
    Present {
        size: u64,
        mtime: Option<SystemTime>,
    },
    /// `metadata()` errored (path missing, transient stat failure, permissions). Kept DISTINCT from
    /// any `Present` value so a stat error forces a cache MISS (rebuild) rather than serving a stale
    /// entry — the load that follows surfaces the real error. Two `Unavailable`s compare equal
    /// (`Eq` must stay reflexive), but that only arises when the file is genuinely gone on both the
    /// cached and the incoming key, in which case the reload fails loudly anyway.
    Unavailable,
}

impl Fingerprint {
    /// Snapshot `(size, mtime)` for `path` once, at key-construction time, so the fingerprint is
    /// stable across the lookup within a single request (no mtime drift mid-request).
    pub(crate) fn of(path: &Path) -> Self {
        match std::fs::metadata(path) {
            Ok(meta) => Self::Present {
                size: meta.len(),
                mtime: meta.modified().ok(),
            },
            Err(_) => Self::Unavailable,
        }
    }
}

/// A single-resident model cache. Holds at most one loaded model (`M`) keyed by `K`; a lookup for a
/// different key is a miss that drops the resident model and loads the new one.
pub(crate) struct CacheThread<K, M> {
    entry: Option<Entry<K, M>>,
    /// See the module docs: `true` frees the backend cache before loading a miss (refine cache),
    /// `false` does not (generator cache). Kept as a per-cache parameter to preserve each lane's
    /// current memory behavior exactly.
    evict_before_load: bool,
    /// Generator execution panics can leave Metal state unusable. Other cache lanes retain their
    /// historical per-request panic containment until they explicitly opt into process retirement.
    fatal_panics: bool,
    poisoned: bool,
}

struct Entry<K, M> {
    key: K,
    model: M,
}

/// Whether the current cache run loaded a new entry or reused the resident one.
///
/// This is deliberately reported to the run closure instead of being encoded in the key: request
/// policy must be re-evaluated on every warm run without fragmenting a single loaded model by
/// geometry or strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheAccess {
    Cold,
    Warm,
}

impl<K, M> CacheThread<K, M>
where
    K: Clone + PartialEq,
{
    pub(crate) fn new(evict_before_load: bool) -> Self {
        Self {
            entry: None,
            evict_before_load,
            fatal_panics: false,
            poisoned: false,
        }
    }

    fn mark_poisoned(&mut self) {
        self.poisoned = true;
    }

    fn mark_poisoned_with_error(&mut self, _error: &WorkerError) {
        self.mark_poisoned();
        #[cfg(all(target_os = "macos", not(test)))]
        if let WorkerError::ProcessPoisoned(detail) = _error {
            crate::mlx_worker_recovery::global().observe_process_fault(detail);
        }
    }

    /// Drop the resident model so the next job reloads from scratch. Returns the evicted key (for the
    /// idle-eviction log), or `None` when nothing was resident.
    pub(crate) fn evict(&mut self) -> Option<K> {
        self.entry.take().map(|entry| entry.key)
    }

    /// The resident model, if any (read-only; the YuE2 admission reads what it would free).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn resident_model(&self) -> Option<&M> {
        self.entry.as_ref().map(|entry| &entry.model)
    }

    /// Load (on a miss) or reuse (on a hit) the model for `key`, then run `run` against it.
    ///
    /// A miss clears the resident entry, optionally frees the backend cache first (see
    /// `evict_before_load`), then invokes the caller's `load` closure — which owns whatever
    /// per-lane policy the load needs (fit-gate/residency selection for the generator, error-context
    /// wrapping for either). `entry_missing_msg` is the (should-be-impossible) error surfaced if the
    /// entry vanished after a successful load.
    #[cfg(test)]
    pub(crate) fn with_model<R>(
        &mut self,
        key: K,
        load: impl FnOnce() -> WorkerResult<M>,
        run: impl FnOnce(&M) -> WorkerResult<R>,
        entry_missing_msg: &str,
    ) -> WorkerResult<R> {
        self.with_model_access(key, load, |model, _access| run(model), entry_missing_msg)
    }

    /// Load or reuse the resident model while also reporting the request's true cold/warm cache
    /// access state.
    #[cfg(test)]
    pub(crate) fn with_model_access<R>(
        &mut self,
        key: K,
        load: impl FnOnce() -> WorkerResult<M>,
        run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R>,
        entry_missing_msg: &str,
    ) -> WorkerResult<R> {
        self.with_model_access_after_cold_admission(key, |_| Ok(()), load, run, entry_missing_msg)
    }

    /// [`Self::with_model_access`] with a hook that runs only for a genuine miss, while the previous
    /// resident entry is still alive. A live-memory gate can therefore observe the raw budget before
    /// replacement, account for the entry the cache is about to drop, and avoid touching an exact-key
    /// warm hit. The hook receives the different resident key and model (or `None` on an empty cold
    /// start), so callers can credit only metadata bound to that exact entry rather than a stale
    /// process-global high-water, and can report why the key missed. The entry is cleared only after
    /// admission succeeds.
    pub(crate) fn with_model_access_after_cold_admission<R>(
        &mut self,
        key: K,
        cold_admission: impl FnOnce(Option<(&K, &M)>) -> WorkerResult<()>,
        load: impl FnOnce() -> WorkerResult<M>,
        run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R>,
        entry_missing_msg: &str,
    ) -> WorkerResult<R> {
        let access = if self.entry.as_ref().is_some_and(|entry| entry.key == key) {
            CacheAccess::Warm
        } else {
            cold_admission(self.entry.as_ref().map(|entry| (&entry.key, &entry.model)))?;
            self.entry = None;
            if self.evict_before_load {
                release_backend_cache_after_evict();
            }
            let model = load()?;
            self.entry = Some(Entry {
                key: key.clone(),
                model,
            });
            CacheAccess::Cold
        };

        let Some(entry) = self.entry.as_ref() else {
            return Err(WorkerError::Engine(entry_missing_msg.to_owned()));
        };
        run(&entry.model, access)
    }

    /// SCAIL-specific cold admission seam: reuse a matching resident without calling
    /// `cold_admission`; on a miss, drop the old resident, then admit and load while this cache
    /// thread still holds the only mutable cache reference. Existing cache users intentionally stay
    /// on [`Self::with_model_access`]; this extra seam prevents SCAIL from doing a racy external
    /// peek-then-act check while leaving every other loader byte-for-byte unchanged.
    #[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
    pub(crate) fn with_model_access_after_cold_evict<R>(
        &mut self,
        key: K,
        cancel_check: impl Fn() -> WorkerResult<()>,
        cold_admission: impl FnOnce() -> WorkerResult<()>,
        load: impl FnOnce() -> WorkerResult<M>,
        run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R>,
        entry_missing_msg: &str,
    ) -> WorkerResult<R> {
        // The async waiter may have been aborted while this closure was queued on the unbounded
        // std::mpsc channel. Check the request-owned flag on the cache thread before even comparing
        // keys so an abandoned job cannot touch a still-useful resident entry.
        cancel_check()?;
        let access = if self.entry.as_ref().is_some_and(|entry| entry.key == key) {
            CacheAccess::Warm
        } else {
            // A full Candle generator/device drop returns the resident cudarc allocation to the
            // driver. Admission therefore samples the real post-evict budget, without guessing at
            // reclaimable bytes. A rejection deliberately leaves the single slot cold.
            cancel_check()?;
            self.entry = None;
            if self.evict_before_load {
                release_backend_cache_after_evict();
            }
            cancel_check()?;
            cold_admission()?;
            cancel_check()?;
            let model = load()?;
            // Loading can take minutes. The request may have been canceled explicitly or its
            // async waiter may have been dropped while the backend was allocating. Do not turn
            // that orphaned result into a warm resident: drop it locally, release any backend
            // cache made reclaimable by the drop, and leave the single slot cold.
            if let Err(error) = cancel_check() {
                drop(model);
                release_backend_cache_after_evict();
                return Err(error);
            }
            self.entry = Some(Entry {
                key: key.clone(),
                model,
            });
            CacheAccess::Cold
        };

        // Check at the actual run boundary for both warm and cold paths. If cancellation races
        // with a cold model's installation, remove that newly loaded resident; a canceled warm
        // request must not evict the useful model it found.
        if let Err(error) = cancel_check() {
            if access == CacheAccess::Cold {
                if let Some(entry) = self.entry.take() {
                    drop(entry);
                    release_backend_cache_after_evict();
                }
            }
            return Err(error);
        }
        let Some(entry) = self.entry.as_ref() else {
            return Err(WorkerError::Engine(entry_missing_msg.to_owned()));
        };
        run(&entry.model, access)
    }

    /// Whether a model is currently resident. Test/introspection helper.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entry.is_none()
    }

    /// The resident entry's key, if any. Test/introspection helper.
    #[cfg(test)]
    pub(crate) fn resident_key(&self) -> Option<&K> {
        self.entry.as_ref().map(|entry| &entry.key)
    }

    /// Install a resident entry directly (test seam — production goes through [`Self::with_model`]).
    #[cfg(test)]
    pub(crate) fn install(&mut self, key: K, model: M) {
        self.entry = Some(Entry { key, model });
    }
}

/// The event a cache worker's receive step yields.
enum WorkerEvent<J> {
    Job(J),
    IdleTimeout,
    Disconnected,
}

/// Block for the next job, or report an idle timeout / disconnect. With `idle_timeout = None` the
/// worker blocks forever for a job (idle eviction disabled).
fn recv_job<J>(rx: &mpsc::Receiver<J>, idle_timeout: Option<Duration>) -> WorkerEvent<J> {
    match idle_timeout {
        Some(timeout) => match rx.recv_timeout(timeout) {
            Ok(job) => WorkerEvent::Job(job),
            Err(mpsc::RecvTimeoutError::Timeout) => WorkerEvent::IdleTimeout,
            Err(mpsc::RecvTimeoutError::Disconnected) => WorkerEvent::Disconnected,
        },
        None => match rx.recv() {
            Ok(job) => WorkerEvent::Job(job),
            Err(_) => WorkerEvent::Disconnected,
        },
    }
}

/// The dedicated cache thread's event loop: own a single-resident [`CacheThread`], serve jobs from
/// `rx`, and evict the resident model after `idle_timeout` of inactivity. `log_idle_evict(key,
/// idle_seconds)` emits the lane-specific idle-eviction telemetry.
///
/// A panic escaping a job is contained (`catch_unwind`). Ordinary cache lanes evict and keep
/// serving (sc-6067); a generator owner with fatal-panics policy retires without GPU cleanup.
pub(crate) fn run_cache_worker<K, M>(
    rx: mpsc::Receiver<CacheJob<K, M>>,
    idle_timeout: Option<Duration>,
    evict_before_load: bool,
    log_idle_evict: impl Fn(&K, u64),
) where
    K: Clone + PartialEq,
{
    run_cache_worker_with_panic_policy(rx, idle_timeout, evict_before_load, false, log_idle_evict)
}

/// The generator owner uses this policy because a panic after a backend submission cannot prove
/// that dropping the model or clearing the backend cache is safe. The process supervisor replaces
/// this owner after the active job's original failure has been persisted.
pub(crate) fn run_cache_worker_with_fatal_panics<K, M>(
    rx: mpsc::Receiver<CacheJob<K, M>>,
    idle_timeout: Option<Duration>,
    evict_before_load: bool,
    log_idle_evict: impl Fn(&K, u64),
) where
    K: Clone + PartialEq,
{
    run_cache_worker_with_panic_policy(rx, idle_timeout, evict_before_load, true, log_idle_evict)
}

pub(crate) fn run_cache_worker_with_panic_policy<K, M>(
    rx: mpsc::Receiver<CacheJob<K, M>>,
    idle_timeout: Option<Duration>,
    evict_before_load: bool,
    fatal_panics: bool,
    log_idle_evict: impl Fn(&K, u64),
) where
    K: Clone + PartialEq,
{
    let mut cache = CacheThread::<K, M>::new(evict_before_load);
    cache.fatal_panics = fatal_panics;
    loop {
        let job = match recv_job(&rx, idle_timeout) {
            WorkerEvent::Job(job) => job,
            WorkerEvent::IdleTimeout => {
                if let Some(key) = cache.evict() {
                    release_backend_cache_after_evict();
                    log_idle_evict(&key, idle_timeout.map_or(0, |timeout| timeout.as_secs()));
                }
                continue;
            }
            WorkerEvent::Disconnected => break,
        };
        // Backstop for a panic that escaped the job's own reply guard. Ordinary cache lanes evict;
        // the MLX generator owner retires because its backend state may be poisoned.
        if let Err(_panic) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&mut cache)))
        {
            if fatal_panics {
                cache.mark_poisoned();
                #[cfg(all(target_os = "macos", not(test)))]
                crate::mlx_worker_recovery::global().observe_process_fault(&format!(
                    "Generator owner queue panicked: {}",
                    panic_message(_panic.as_ref())
                ));
            } else if cache.evict().is_some() {
                release_backend_cache_after_evict();
            }
        }
        if cache.poisoned {
            // A backend destructor may itself submit to the poisoned device. Abandon this process's
            // resident object; OS process retirement releases it without another GPU operation.
            std::mem::forget(cache);
            break;
        }
    }
}

/// User-facing message strings for [`run_cached`], so each lane keeps its exact wording.
#[derive(Clone, Copy)]
pub(crate) struct SeamMessages {
    /// Surfaced if the resident entry vanished right after a successful load (should be impossible).
    pub entry_missing: &'static str,
    /// Prefix for a contained-panic error; the caught payload text is appended as `"{prefix}: {msg}"`.
    pub panic_reset: &'static str,
    /// Surfaced when the worker thread's channel is closed (worker gone).
    pub worker_stopped: &'static str,
    /// Surfaced when the worker dropped the job's reply channel without answering.
    pub worker_dropped: &'static str,
}

/// Run `run` against the cached (or freshly-loaded) model for `key` on the dedicated cache `worker`
/// thread, and await the result. The model lives on the worker thread — `load` builds it there and
/// `run` executes there (so it may hold a `!Send` reference) — and only the `R` result crosses back.
///
/// A panic from inside load/run fails this request with the original diagnostic. Ordinary caches
/// evict and continue; the MLX generator owner returns [`WorkerError::ProcessPoisoned`] and retires
/// without backend cleanup.
pub(crate) async fn run_cached<K, M, R>(
    worker: &mpsc::Sender<CacheJob<K, M>>,
    key: K,
    load: impl FnOnce() -> WorkerResult<M> + Send + 'static,
    run: impl FnOnce(&M) -> WorkerResult<R> + Send + 'static,
    msgs: SeamMessages,
) -> WorkerResult<R>
where
    K: Clone + PartialEq + Send + 'static,
    M: 'static,
    R: Send + 'static,
{
    run_cached_with_access(worker, key, load, |model, _access| run(model), msgs).await
}

/// [`run_cached`] plus the true cold/warm access state for request-scoped policy evaluation.
pub(crate) async fn run_cached_with_access<K, M, R>(
    worker: &mpsc::Sender<CacheJob<K, M>>,
    key: K,
    load: impl FnOnce() -> WorkerResult<M> + Send + 'static,
    run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R> + Send + 'static,
    msgs: SeamMessages,
) -> WorkerResult<R>
where
    K: Clone + PartialEq + Send + 'static,
    M: 'static,
    R: Send + 'static,
{
    run_cached_with_access_after_cold_admission(worker, key, |_| Ok(()), load, run, msgs).await
}

/// [`run_cached_with_access`] with an admission hook that executes atomically on the cache thread
/// only for a miss and before the old entry is dropped.
pub(crate) async fn run_cached_with_access_after_cold_admission<K, M, R>(
    worker: &mpsc::Sender<CacheJob<K, M>>,
    key: K,
    cold_admission: impl FnOnce(Option<(&K, &M)>) -> WorkerResult<()> + Send + 'static,
    load: impl FnOnce() -> WorkerResult<M> + Send + 'static,
    run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R> + Send + 'static,
    msgs: SeamMessages,
) -> WorkerResult<R>
where
    K: Clone + PartialEq + Send + 'static,
    M: 'static,
    R: Send + 'static,
{
    run_cached_with_access_after_cold_admission_prepared(
        worker,
        key,
        || Ok(()),
        cold_admission,
        load,
        move |model, access, ()| run(model, access),
        msgs,
    )
    .await
}

/// Compile request state on the cache owner immediately before its load/run transaction. This
/// keeps global-seed preparation, materialization, cancellation and teardown in one queue entry.
pub(crate) async fn run_cached_with_access_after_cold_admission_prepared<K, M, P, R>(
    worker: &mpsc::Sender<CacheJob<K, M>>,
    key: K,
    prepare: impl FnOnce() -> WorkerResult<P> + Send + 'static,
    cold_admission: impl FnOnce(Option<(&K, &M)>) -> WorkerResult<()> + Send + 'static,
    load: impl FnOnce() -> WorkerResult<M> + Send + 'static,
    run: impl FnOnce(&M, CacheAccess, P) -> WorkerResult<R> + Send + 'static,
    msgs: SeamMessages,
) -> WorkerResult<R>
where
    K: Clone + PartialEq + Send + 'static,
    M: 'static,
    P: 'static,
    R: Send + 'static,
{
    let (reply_tx, reply_rx) = oneshot::channel::<WorkerResult<R>>();
    let progress_context = crate::load_progress::Context::capture();
    let job: CacheJob<K, M> = Box::new(move |cache: &mut CacheThread<K, M>| {
        let _progress = progress_context.install();
        // Contain a panic at this job's reply seam. Ordinary caches evict and continue; a fatal
        // owner returns a typed process fault and retires without touching backend state.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let prepared = prepare()?;
            cache.with_model_access_after_cold_admission(
                key,
                cold_admission,
                load,
                move |model, access| run(model, access, prepared),
                msgs.entry_missing,
            )
        })) {
            Ok(result) => result,
            Err(panic) => {
                if cache.fatal_panics {
                    cache.mark_poisoned();
                } else if cache.evict().is_some() {
                    release_backend_cache_after_evict();
                }
                let detail = format!("{}: {}", msgs.panic_reset, panic_message(panic.as_ref()));
                if cache.fatal_panics {
                    Err(WorkerError::ProcessPoisoned(detail))
                } else {
                    Err(WorkerError::Engine(detail))
                }
            }
        };
        if let Err(error @ WorkerError::ProcessPoisoned(_)) = &result {
            cache.mark_poisoned_with_error(error);
        }
        let _ = reply_tx.send(result);
    });
    worker
        .send(job)
        .map_err(|_| owner_stopped_error(msgs.worker_stopped))?;
    reply_rx
        .await
        .map_err(|_| owner_stopped_error(msgs.worker_dropped))?
}

/// SCAIL-only sibling of [`run_cached_with_access`]. The admission closure runs on the dedicated
/// cache worker after a true miss has dropped its resident and before the incoming loader allocates;
/// a warm hit never invokes it. Taking an owned sender keeps the production worker and isolated race
/// tests on the identical serialized job seam without changing the established cache APIs.
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
pub(crate) async fn run_cached_with_access_after_cold_evict<K, M, R>(
    worker: mpsc::Sender<CacheJob<K, M>>,
    key: K,
    cancel_check: impl Fn() -> WorkerResult<()> + Send + 'static,
    cold_admission: impl FnOnce() -> WorkerResult<()> + Send + 'static,
    load: impl FnOnce() -> WorkerResult<M> + Send + 'static,
    run: impl FnOnce(&M, CacheAccess) -> WorkerResult<R> + Send + 'static,
    msgs: SeamMessages,
) -> WorkerResult<R>
where
    K: Clone + PartialEq + Send + 'static,
    M: 'static,
    R: Send + 'static,
{
    let (reply_tx, reply_rx) = oneshot::channel::<WorkerResult<R>>();
    let progress_context = crate::load_progress::Context::capture();
    let job: CacheJob<K, M> = Box::new(move |cache: &mut CacheThread<K, M>| {
        let _progress = progress_context.install();
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let request_active = || {
                // A Tokio task can be aborted/dropped without first tripping its engine cancel
                // flag. The std::mpsc job outlives that waiter, so the reply channel is the
                // authoritative abandonment signal for that path.
                if reply_tx.is_closed() {
                    Err(WorkerError::Canceled(
                        "Model cache request was abandoned.".to_owned(),
                    ))
                } else {
                    cancel_check()
                }
            };
            cache.with_model_access_after_cold_evict(
                key,
                request_active,
                cold_admission,
                load,
                run,
                msgs.entry_missing,
            )
        })) {
            Ok(result) => result,
            Err(panic) => {
                if cache.fatal_panics {
                    cache.mark_poisoned();
                } else if cache.evict().is_some() {
                    release_backend_cache_after_evict();
                }
                let detail = format!("{}: {}", msgs.panic_reset, panic_message(panic.as_ref()));
                if cache.fatal_panics {
                    Err(WorkerError::ProcessPoisoned(detail))
                } else {
                    Err(WorkerError::Engine(detail))
                }
            }
        };
        if let Err(error @ WorkerError::ProcessPoisoned(_)) = &result {
            cache.mark_poisoned_with_error(error);
        }
        let _ = reply_tx.send(result);
    });
    worker
        .send(job)
        .map_err(|_| owner_stopped_error(msgs.worker_stopped))?;
    reply_rx
        .await
        .map_err(|_| owner_stopped_error(msgs.worker_dropped))?
}

fn owner_stopped_error(fallback: &str) -> WorkerError {
    #[cfg(all(target_os = "macos", not(test)))]
    if let Some(detail) = crate::mlx_worker_recovery::global().process_fault_detail() {
        return WorkerError::ProcessPoisoned(detail);
    }
    WorkerError::Engine(fallback.to_owned())
}

/// Parse a cache idle-eviction window (seconds) from an env value, falling back to `default_secs`
/// when absent/blank/unparseable. `0` disables idle eviction (returns `None`); any positive value is
/// that many seconds.
pub(crate) fn idle_timeout_from_secs(raw: Option<&str>, default_secs: u64) -> Option<Duration> {
    let seconds = raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default_secs);
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

/// Best-effort human-readable text from a caught panic payload — the `&str`/`String` a `panic!`
/// produces. mlx-rs `.unwrap()`/`.expect()` panics carry their formatted message as a `String`
/// (e.g. the `[metal::malloc] Attempting to allocate …` Metal OOM), so this surfaces the real cause.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn release_backend_cache_after_evict() {
    mlx_rs::memory::clear_cache();
}

/// Off-Mac (candle/CUDA) this is intentionally a no-op (epic 10765, sc-10766). candle's CUDA backend
/// uses cudarc, which exposes no `empty_cache`/trim to call here — but none is needed: dropping the
/// evicted model is what frees its VRAM. GPU-measured (sc-13960, RTX PRO 6000): a full generator drop
/// returns most of that VRAM to the DRIVER — `nvidia-smi` free RISES (a resident ~45.7 GB QwenEdit's
/// drop gave ~45 GB back) — so the incoming load has room without any separate driver-level trim. (The
/// stronger "cudarc keeps the freed pages pooled in-process so `nvidia-smi` free stays flat" framing
/// holds at most for a WITHIN-DEVICE component free under sequential residency, where the device stays
/// alive; it does NOT hold for a full evict, which drops the device.) The VRAM fit-gate therefore
/// budgets on predicted peak, not resident deltas — and predicts the post-evict `free` via
/// `reclaimable_pool_gb` rather than trying to read it (the gate runs before the evict).
#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn release_backend_cache_after_evict() {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    const TEST_SEAM_MESSAGES: SeamMessages = SeamMessages {
        entry_missing: "test cache entry missing",
        panic_reset: "test cache panic",
        worker_stopped: "test cache worker stopped",
        worker_dropped: "test cache worker dropped reply",
    };

    fn start_isolated_cache_worker() -> (
        mpsc::Sender<CacheJob<u32, DropProbe>>,
        thread::JoinHandle<()>,
    ) {
        let (tx, rx) = mpsc::channel::<CacheJob<u32, DropProbe>>();
        let worker = thread::spawn(move || run_cache_worker(rx, None, false, |_key, _seconds| {}));
        (tx, worker)
    }

    #[tokio::test]
    async fn fatal_owner_panic_preserves_original_failure_without_gpu_cleanup_or_reuse() {
        let (worker, rx) = mpsc::channel::<CacheJob<u32, DropProbe>>();
        let owner = thread::spawn(move || {
            run_cache_worker_with_fatal_panics(rx, None, false, |_key, _seconds| {})
        });
        let drops = Arc::new(AtomicUsize::new(0));
        let loaded_drops = Arc::clone(&drops);
        run_cached(
            &worker,
            7,
            move || Ok(DropProbe(loaded_drops)),
            |_model| Ok(()),
            TEST_SEAM_MESSAGES,
        )
        .await
        .expect("initial owner access");

        let failure = run_cached(
            &worker,
            7,
            || panic!("warm load must not execute"),
            |_model| -> WorkerResult<()> { panic!("first Metal owner fault") },
            TEST_SEAM_MESSAGES,
        )
        .await
        .expect_err("panic must poison the owner");
        assert!(matches!(failure, WorkerError::ProcessPoisoned(_)));
        assert!(failure.to_string().contains("first Metal owner fault"));
        owner.join().expect("poisoned owner retires");
        assert_eq!(drops.load(Ordering::SeqCst), 0, "no backend destructor ran");
        assert!(worker
            .send(Box::new(|_| panic!("retired owner ran again")))
            .is_err());
    }

    #[tokio::test]
    async fn two_prepared_jobs_serialize_materialization_and_destruction_on_owner() {
        use std::sync::Mutex;

        struct OwnerDrop {
            id: u32,
            trace: Arc<Mutex<Vec<(u32, &'static str, thread::ThreadId)>>>,
        }
        impl Drop for OwnerDrop {
            fn drop(&mut self) {
                self.trace
                    .lock()
                    .unwrap()
                    .push((self.id, "drop", thread::current().id()));
            }
        }

        let (worker, rx) = mpsc::channel::<CacheJob<u32, OwnerDrop>>();
        let owner = thread::spawn(move || run_cache_worker(rx, None, false, |_key, _seconds| {}));
        let trace = Arc::new(Mutex::new(Vec::new()));
        let (ready_tx, ready_rx) = oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first_worker = worker.clone();
        let first_trace = Arc::clone(&trace);
        let first = tokio::spawn(async move {
            let prepared_trace = Arc::clone(&first_trace);
            let loaded_trace = Arc::clone(&first_trace);
            run_cached_with_access_after_cold_admission_prepared(
                &first_worker,
                104729,
                move || {
                    let id = thread::current().id();
                    prepared_trace.lock().unwrap().push((104729, "prepare", id));
                    ready_tx.send(id).expect("owner entered first prepare");
                    release_rx.recv().expect("release first prepare");
                    Ok(104729_u32)
                },
                |_| Ok(()),
                move || {
                    loaded_trace
                        .lock()
                        .unwrap()
                        .push((104729, "load", thread::current().id()));
                    Ok(OwnerDrop {
                        id: 104729,
                        trace: loaded_trace,
                    })
                },
                move |_, _, seed| {
                    assert_eq!(seed, 104729);
                    first_trace
                        .lock()
                        .unwrap()
                        .push((104729, "run", thread::current().id()));
                    Ok(())
                },
                TEST_SEAM_MESSAGES,
            )
            .await
        });
        let owner_id = ready_rx.await.expect("first prepare starts");
        let second_worker = worker.clone();
        let second_trace = Arc::clone(&trace);
        let second = tokio::spawn(async move {
            let prepared_trace = Arc::clone(&second_trace);
            let loaded_trace = Arc::clone(&second_trace);
            run_cached_with_access_after_cold_admission_prepared(
                &second_worker,
                104730,
                move || {
                    prepared_trace.lock().unwrap().push((
                        104730,
                        "prepare",
                        thread::current().id(),
                    ));
                    Ok(104730_u32)
                },
                |_| Ok(()),
                move || {
                    loaded_trace
                        .lock()
                        .unwrap()
                        .push((104730, "load", thread::current().id()));
                    Ok(OwnerDrop {
                        id: 104730,
                        trace: loaded_trace,
                    })
                },
                move |_, _, seed| {
                    assert_eq!(seed, 104730);
                    second_trace
                        .lock()
                        .unwrap()
                        .push((104730, "run", thread::current().id()));
                    Ok(())
                },
                TEST_SEAM_MESSAGES,
            )
            .await
        });
        release_tx.send(()).expect("resume first request");
        first.await.expect("first task").expect("first request");
        second.await.expect("second task").expect("second request");
        drop(worker);
        owner.join().expect("owner exits and drops second model");
        let observed = trace.lock().unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|(id, phase, _)| (*id, *phase))
                .collect::<Vec<_>>(),
            [
                (104729, "prepare"),
                (104729, "load"),
                (104729, "run"),
                (104730, "prepare"),
                (104729, "drop"),
                (104730, "load"),
                (104730, "run"),
                (104730, "drop"),
            ]
        );
        assert!(observed.iter().all(|(_, _, id)| *id == owner_id));
    }

    fn cache_state(worker: &mpsc::Sender<CacheJob<u32, DropProbe>>) -> (bool, Option<u32>) {
        let (state_tx, state_rx) = mpsc::channel();
        worker
            .send(Box::new(move |cache: &mut CacheThread<u32, DropProbe>| {
                state_tx
                    .send((cache.is_empty(), cache.resident_key().copied()))
                    .expect("send cache state");
            }))
            .expect("inspect isolated cache");
        state_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("receive cache state")
    }

    async fn assert_subsequent_live_request_loads_and_reuses_warm(
        worker: mpsc::Sender<CacheJob<u32, DropProbe>>,
        drops: Arc<AtomicUsize>,
        loads: Arc<AtomicUsize>,
        runs: Arc<AtomicUsize>,
    ) {
        let cold_drops = drops.clone();
        let cold_loads = loads.clone();
        let cold_runs = runs.clone();
        let cold_access = run_cached_with_access_after_cold_evict(
            worker.clone(),
            7,
            || Ok(()),
            || Ok(()),
            move || {
                cold_loads.fetch_add(1, Ordering::SeqCst);
                Ok(DropProbe(cold_drops))
            },
            move |_model, access| {
                cold_runs.fetch_add(1, Ordering::SeqCst);
                Ok(access)
            },
            TEST_SEAM_MESSAGES,
        )
        .await
        .expect("subsequent live request loads");
        assert_eq!(cold_access, CacheAccess::Cold);

        let warm_runs = runs.clone();
        let access = run_cached_with_access_after_cold_evict(
            worker.clone(),
            7,
            || Ok(()),
            || panic!("warm request must bypass cold admission"),
            || panic!("warm request must reuse the resident model"),
            move |_model, access| {
                warm_runs.fetch_add(1, Ordering::SeqCst);
                Ok(access)
            },
            TEST_SEAM_MESSAGES,
        )
        .await
        .expect("subsequent warm request reuses resident");
        assert_eq!(access, CacheAccess::Warm);
        assert_eq!(cache_state(&worker), (false, Some(7)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn canceled_during_load_drops_model_without_install_or_run_then_recovers() {
        let (worker, worker_thread) = start_isolated_cache_worker();
        let cancel = gen_core::CancelFlag::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let (load_started_tx, load_started_rx) = mpsc::channel();
        let (load_release_tx, load_release_rx) = mpsc::channel();

        let request_cancel = cancel.clone();
        let request_drops = drops.clone();
        let request_loads = loads.clone();
        let request_runs = runs.clone();
        let mut request = Box::pin(run_cached_with_access_after_cold_evict(
            worker.clone(),
            7,
            move || {
                if request_cancel.is_cancelled() {
                    Err(WorkerError::Canceled(
                        "test request canceled by user".to_owned(),
                    ))
                } else {
                    Ok(())
                }
            },
            || Ok(()),
            move || {
                request_loads.fetch_add(1, Ordering::SeqCst);
                load_started_tx.send(()).expect("signal loader started");
                load_release_rx
                    .recv_timeout(Duration::from_secs(2))
                    .expect("release blocked loader");
                Ok(DropProbe(request_drops))
            },
            move |_model, _access| {
                request_runs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            TEST_SEAM_MESSAGES,
        ));
        tokio::select! {
            biased;
            result = &mut request => panic!("blocked load resolved unexpectedly: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        load_started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("loader is blocked");

        cancel.cancel();
        load_release_tx.send(()).expect("release loader");
        assert!(matches!(request.await, Err(WorkerError::Canceled(_))));
        assert_eq!(cache_state(&worker), (true, None));
        assert_eq!(drops.load(Ordering::SeqCst), 1, "loaded model was dropped");
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(runs.load(Ordering::SeqCst), 0, "canceled request never ran");

        assert_subsequent_live_request_loads_and_reuses_warm(
            worker.clone(),
            drops.clone(),
            loads.clone(),
            runs.clone(),
        )
        .await;
        assert_eq!(loads.load(Ordering::SeqCst), 2, "warm reuse did not reload");
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "live cold and warm requests ran"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "live model remains resident"
        );

        drop(worker);
        worker_thread.join().expect("isolated cache worker exits");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abandoned_during_load_drops_model_without_install_or_run_then_recovers() {
        let (worker, worker_thread) = start_isolated_cache_worker();
        let cancel = gen_core::CancelFlag::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let (load_started_tx, load_started_rx) = mpsc::channel();
        let (load_release_tx, load_release_rx) = mpsc::channel();

        let request_cancel = cancel.clone();
        let request_drops = drops.clone();
        let request_loads = loads.clone();
        let request_runs = runs.clone();
        let mut request = Box::pin(run_cached_with_access_after_cold_evict(
            worker.clone(),
            7,
            move || {
                if request_cancel.is_cancelled() {
                    Err(WorkerError::Canceled(
                        "test request canceled by user".to_owned(),
                    ))
                } else {
                    Ok(())
                }
            },
            || Ok(()),
            move || {
                request_loads.fetch_add(1, Ordering::SeqCst);
                load_started_tx.send(()).expect("signal loader started");
                load_release_rx
                    .recv_timeout(Duration::from_secs(2))
                    .expect("release blocked loader");
                Ok(DropProbe(request_drops))
            },
            move |_model, _access| {
                request_runs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            TEST_SEAM_MESSAGES,
        ));
        tokio::select! {
            biased;
            result = &mut request => panic!("blocked load resolved unexpectedly: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        load_started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("loader is blocked");

        // Dropping the future closes the oneshot receiver without tripping the explicit flag.
        drop(request);
        assert!(!cancel.is_cancelled());
        load_release_tx.send(()).expect("release loader");
        assert_eq!(cache_state(&worker), (true, None));
        assert_eq!(drops.load(Ordering::SeqCst), 1, "loaded model was dropped");
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(
            runs.load(Ordering::SeqCst),
            0,
            "abandoned request never ran"
        );

        assert_subsequent_live_request_loads_and_reuses_warm(
            worker.clone(),
            drops.clone(),
            loads.clone(),
            runs.clone(),
        )
        .await;
        assert_eq!(loads.load(Ordering::SeqCst), 2, "warm reuse did not reload");
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "live cold and warm requests ran"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "live model remains resident"
        );

        drop(worker);
        worker_thread.join().expect("isolated cache worker exits");
    }

    #[test]
    fn final_run_check_preserves_warm_entry_and_discards_new_cold_entry() {
        use std::cell::Cell;

        let drops = Arc::new(AtomicUsize::new(0));
        let mut warm_cache = CacheThread::<u32, DropProbe>::new(false);
        warm_cache.install(7, DropProbe(drops.clone()));
        let warm_checks = Cell::new(0usize);
        let warm_runs = Cell::new(0usize);
        let warm = warm_cache.with_model_access_after_cold_evict(
            7,
            || {
                warm_checks.set(warm_checks.get() + 1);
                if warm_checks.get() == 2 {
                    Err(WorkerError::Canceled("canceled before warm run".to_owned()))
                } else {
                    Ok(())
                }
            },
            || panic!("warm request must bypass admission"),
            || panic!("warm request must bypass load"),
            |_model, _access| {
                warm_runs.set(warm_runs.get() + 1);
                Ok(())
            },
            "missing",
        );
        assert!(matches!(warm, Err(WorkerError::Canceled(_))));
        assert_eq!(warm_runs.get(), 0);
        assert_eq!(warm_cache.resident_key(), Some(&7));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "warm resident is preserved"
        );
        warm_cache.evict();

        let mut cold_cache = CacheThread::<u32, DropProbe>::new(false);
        let cold_checks = Cell::new(0usize);
        let cold_runs = Cell::new(0usize);
        let cold = cold_cache.with_model_access_after_cold_evict(
            7,
            || {
                cold_checks.set(cold_checks.get() + 1);
                if cold_checks.get() == 6 {
                    Err(WorkerError::Canceled("canceled before cold run".to_owned()))
                } else {
                    Ok(())
                }
            },
            || Ok(()),
            || Ok(DropProbe(drops.clone())),
            |_model, _access| {
                cold_runs.set(cold_runs.get() + 1);
                Ok(())
            },
            "missing",
        );
        assert!(matches!(cold, Err(WorkerError::Canceled(_))));
        assert_eq!(cold_checks.get(), 6);
        assert_eq!(cold_runs.get(), 0);
        assert!(cold_cache.is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 2, "both models were dropped");
    }

    // The generic evict-before-load flag must gate the pre-load backend trim. We can't observe the
    // (cfg'd-out under test) backend call, so exercise the ORDERING through the loader closure: a
    // loader that records whether the resident entry was already cleared when it ran. Both flag
    // values must clear the resident entry before loading a miss (single-resident); the flag only
    // toggles the backend trim, which is a no-op under `cfg(test)`.
    #[test]
    fn with_model_reloads_on_miss_and_reuses_on_hit() {
        use std::cell::Cell;
        let mut cache = CacheThread::<u32, u32>::new(false);
        let loads = Cell::new(0u32);
        let run = |cache: &mut CacheThread<u32, u32>, value: u32| {
            cache
                .with_model(
                    value,
                    || {
                        loads.set(loads.get() + 1);
                        Ok(value)
                    },
                    |m: &u32| Ok(*m),
                    "missing",
                )
                .unwrap()
        };
        assert_eq!(run(&mut cache, 7), 7);
        assert_eq!(loads.get(), 1, "cold key loads");
        // Same key → hit, no reload.
        assert_eq!(run(&mut cache, 7), 7);
        assert_eq!(loads.get(), 1, "same key reuses the resident model");
        // Different key → miss, reload.
        assert_eq!(run(&mut cache, 9), 9);
        assert_eq!(loads.get(), 2, "a new key reloads");
        assert_eq!(cache.resident_key(), Some(&9));
    }

    #[test]
    fn with_model_access_reports_cold_then_warm_without_rekeying() {
        use std::cell::Cell;
        let mut cache = CacheThread::<u32, u32>::new(false);
        let loads = Cell::new(0u32);
        let run = |cache: &mut CacheThread<u32, u32>| {
            cache
                .with_model_access(
                    7,
                    || {
                        loads.set(loads.get() + 1);
                        Ok(70)
                    },
                    |model, access| Ok((*model, access)),
                    "missing",
                )
                .unwrap()
        };

        assert_eq!(run(&mut cache), (70, CacheAccess::Cold));
        assert_eq!(run(&mut cache), (70, CacheAccess::Warm));
        assert_eq!(
            loads.get(),
            1,
            "request policy does not fragment the cache key"
        );
    }

    #[test]
    fn cold_admission_is_bypassed_for_an_exact_hit_and_rejection_leaves_no_entry() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct DropProbe(Rc<Cell<bool>>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let mut cache = CacheThread::<u32, DropProbe>::new(false);
        cache.install(7, DropProbe(dropped.clone()));

        let warm_gate_called = Cell::new(false);
        let warm = cache
            .with_model_access_after_cold_evict(
                7,
                || Ok(()),
                || {
                    warm_gate_called.set(true);
                    Err(WorkerError::InvalidPayload(
                        "warm gate must be bypassed".to_owned(),
                    ))
                },
                || panic!("exact warm key must not load"),
                |_model, access| Ok(access),
                "missing",
            )
            .expect("exact warm key succeeds even when a cold gate would reject");
        assert_eq!(warm, CacheAccess::Warm);
        assert!(!warm_gate_called.get());
        assert!(!dropped.get(), "warm access must keep the resident alive");

        let load_called = Cell::new(false);
        let rejected = cache.with_model_access_after_cold_evict(
            8,
            || Ok(()),
            || {
                assert!(
                    dropped.get(),
                    "a different resident must be dropped before admission samples free VRAM"
                );
                Err(WorkerError::InvalidPayload(
                    "cold admission rejected".to_owned(),
                ))
            },
            || {
                load_called.set(true);
                Ok(DropProbe(Rc::new(Cell::new(false))))
            },
            |_model, _access| Ok(()),
            "missing",
        );
        assert!(matches!(rejected, Err(WorkerError::InvalidPayload(_))));
        assert!(
            !load_called.get(),
            "a rejected cold miss must not start loading"
        );
        assert!(
            cache.is_empty(),
            "a rejected miss must leave no partial entry"
        );
    }

    // The evict-before-load flag is preserved per-cache: only when set does a miss free the backend
    // cache before the load. `release_backend_cache_after_evict` is a no-op under `cfg(test)`, so we
    // assert the flag is plumbed through `new` and that a miss still clears the prior entry in both
    // modes (the flag only adds the trim, never changes the single-resident replacement).
    #[test]
    fn evict_before_load_flag_is_per_cache_and_miss_replaces_entry() {
        for evict_before_load in [false, true] {
            let mut cache = CacheThread::<u32, u32>::new(evict_before_load);
            cache.install(1, 100);
            assert_eq!(cache.resident_key(), Some(&1));
            // Miss on a new key replaces the resident entry regardless of the flag.
            let got = cache
                .with_model(2, || Ok(200u32), |m: &u32| Ok(*m), "missing")
                .unwrap();
            assert_eq!(got, 200);
            assert_eq!(cache.resident_key(), Some(&2));
        }
    }

    #[test]
    fn evict_clears_resident_and_returns_key() {
        let mut cache = CacheThread::<u32, u32>::new(false);
        assert!(cache.is_empty());
        cache.install(5, 50);
        assert!(!cache.is_empty());
        assert_eq!(cache.evict(), Some(5));
        assert!(cache.is_empty());
        assert_eq!(cache.evict(), None);
    }

    #[test]
    fn idle_timeout_defaults_parses_and_disables() {
        assert_eq!(
            idle_timeout_from_secs(None, 300),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            idle_timeout_from_secs(Some(""), 300),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            idle_timeout_from_secs(Some("not-a-number"), 300),
            Some(Duration::from_secs(300))
        );
        assert_eq!(idle_timeout_from_secs(Some("0"), 300), None);
        assert_eq!(
            idle_timeout_from_secs(Some("42"), 300),
            Some(Duration::from_secs(42))
        );
    }

    #[test]
    fn fingerprint_of_reports_present_and_unavailable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("weights.bin");
        std::fs::write(&path, b"bytes").expect("write");
        assert!(matches!(
            Fingerprint::of(&path),
            Fingerprint::Present { .. }
        ));
        assert_eq!(
            Fingerprint::of(&dir.path().join("missing")),
            Fingerprint::Unavailable
        );
    }

    #[test]
    fn panic_message_extracts_str_and_string_payloads() {
        let s = std::panic::catch_unwind(|| panic!("boom-str")).unwrap_err();
        assert_eq!(panic_message(s.as_ref()), "boom-str");
        let s = std::panic::catch_unwind(|| panic!("{}", "boom-string".to_owned())).unwrap_err();
        assert_eq!(panic_message(s.as_ref()), "boom-string");
    }
}

#[cfg(test)]
#[path = "cache_thread/process_cache_tests.rs"]
mod process_cache_tests;

#[cfg(test)]
mod shared_idle_configuration_tests {
    use super::*;
    #[test]
    fn shared_idle_env_preserves_default_parse_and_disabled_behavior() {
        for (raw, seconds) in [
            ("", Some(300)),
            ("bad", Some(300)),
            (" 42 ", Some(42)),
            ("0", None),
        ] {
            crate::test_env::temp_env_var(CACHE_IDLE_SECONDS_ENV, raw, || {
                assert_eq!(idle_timeout_from_env(), seconds.map(Duration::from_secs));
            });
        }
        assert_eq!(DEFAULT_CACHE_IDLE_SECONDS, 300);
    }
}
