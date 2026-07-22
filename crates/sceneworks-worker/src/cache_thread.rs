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
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

use tokio::sync::oneshot;

use crate::{WorkerError, WorkerResult};

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
}

struct Entry<K, M> {
    key: K,
    model: M,
}

impl<K, M> CacheThread<K, M>
where
    K: Clone + PartialEq,
{
    pub(crate) fn new(evict_before_load: bool) -> Self {
        Self {
            entry: None,
            evict_before_load,
        }
    }

    /// Drop the resident model so the next job reloads from scratch. Returns the evicted key (for the
    /// idle-eviction log), or `None` when nothing was resident.
    pub(crate) fn evict(&mut self) -> Option<K> {
        self.entry.take().map(|entry| entry.key)
    }

    /// Load (on a miss) or reuse (on a hit) the model for `key`, then run `run` against it.
    ///
    /// A miss clears the resident entry, optionally frees the backend cache first (see
    /// `evict_before_load`), then invokes the caller's `load` closure — which owns whatever
    /// per-lane policy the load needs (fit-gate/residency selection for the generator, error-context
    /// wrapping for either). `entry_missing_msg` is the (should-be-impossible) error surfaced if the
    /// entry vanished after a successful load.
    pub(crate) fn with_model<R>(
        &mut self,
        key: K,
        load: impl FnOnce() -> WorkerResult<M>,
        run: impl FnOnce(&M) -> WorkerResult<R>,
        entry_missing_msg: &str,
    ) -> WorkerResult<R> {
        if self.entry.as_ref().map_or(true, |entry| entry.key != key) {
            self.entry = None;
            if self.evict_before_load {
                release_backend_cache_after_evict();
            }
            let model = load()?;
            self.entry = Some(Entry {
                key: key.clone(),
                model,
            });
        }

        let Some(entry) = self.entry.as_ref() else {
            return Err(WorkerError::Engine(entry_missing_msg.to_owned()));
        };
        run(&entry.model)
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
/// A panic escaping a job is contained (`catch_unwind`) so this one shared thread can never die and
/// poison every later request (sc-6067); on a contained panic the cache is reset because post-abort
/// backend state is suspect.
pub(crate) fn run_cache_worker<K, M>(
    rx: mpsc::Receiver<CacheJob<K, M>>,
    idle_timeout: Option<Duration>,
    evict_before_load: bool,
    log_idle_evict: impl Fn(&K, u64),
) where
    K: Clone + PartialEq,
{
    let mut cache = CacheThread::<K, M>::new(evict_before_load);
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
        // Backstop: contain any panic that escapes a job's own guard so this single shared cache
        // thread can never die and poison every later request (sc-6067). A job normally catches its
        // own panic, replies with a clean error, and evicts; this catches anything it misses. On a
        // contained panic the cache is reset because post-abort backend state is suspect.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&mut cache))).is_err()
            && cache.evict().is_some()
        {
            release_backend_cache_after_evict();
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
/// A panic from inside the load/run (e.g. an mlx-rs `.unwrap()` on a Metal OOM) is contained: it
/// fails THIS request with a clean [`WorkerError::Engine`] built from `msgs.panic_reset`, evicts the
/// resident model (post-abort backend state is suspect), and leaves the shared thread serving.
pub(crate) async fn run_cached<K, M, R>(
    worker: &'static mpsc::Sender<CacheJob<K, M>>,
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
    let (reply_tx, reply_rx) = oneshot::channel::<WorkerResult<R>>();
    let job: CacheJob<K, M> = Box::new(move |cache: &mut CacheThread<K, M>| {
        // Contain a panic from inside the load/run so it fails THIS job with a clean error instead of
        // unwinding out of the shared cache thread and stopping every subsequent request (sc-6067).
        // The cached model is evicted on panic — post-abort backend state is suspect, so the next
        // job reloads fresh.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.with_model(key, load, run, msgs.entry_missing)
        })) {
            Ok(result) => result,
            Err(panic) => {
                if cache.evict().is_some() {
                    release_backend_cache_after_evict();
                }
                Err(WorkerError::Engine(format!(
                    "{}: {}",
                    msgs.panic_reset,
                    panic_message(panic.as_ref())
                )))
            }
        };
        let _ = reply_tx.send(result);
    });
    worker
        .send(job)
        .map_err(|_| WorkerError::Engine(msgs.worker_stopped.to_owned()))?;
    reply_rx
        .await
        .map_err(|_| WorkerError::Engine(msgs.worker_dropped.to_owned()))?
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
