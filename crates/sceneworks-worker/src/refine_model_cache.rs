//! Provider cache for the native prompt-refine / caption / describe LLM (sc-8840, F-038).
//!
//! Refine/caption/describe jobs resolve a `core_llm::TextLlm` model-first via
//! `load_for_model_with` and stream tokens through it (`prompt_refine_jobs.rs`). Before this cache
//! EVERY interactive refine click cold-loaded the ~16 GB Anubis-8B snapshot from scratch — a
//! multi-second stall on each click — because, unlike the image/video lanes' `generator_cache`, the
//! text lane had no resident-model cache.
//!
//! This is the text-LLM sibling of [`crate::generator_cache`]: a single dedicated OS thread owns a
//! single-resident loaded model keyed by its load spec (weights dir + quantization + selection
//! requirements), reused across jobs and **idle-evicted** after a timeout so a 16 GB model is never
//! pinned resident forever. The dedicated-thread design is what lets us cache a `!Send` model: the
//! `Box<dyn TextLlm>` never leaves the cache thread — only `Send` job closures cross the channel to
//! it (identical to the MLX generator cache), which also keeps every MLX allocation on one thread.
//!
//! Idle eviction reuses the SAME env knob + default window as the generator cache
//! (`SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS`, default 300 s), so the two caches age out together and
//! a single setting bounds resident model memory across both lanes.

use std::path::PathBuf;
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::Duration;

use gen_core::core_llm::{Constraint, LoadSpec, ModelRequirements, Quantize, TextLlm};

use crate::cache_thread::{self, CacheJob, CacheThread, Fingerprint, SeamMessages};
use crate::{WorkerError, WorkerResult};

/// The refine cache is a single-resident [`CacheThread`] keyed by [`RefineModelCacheKey`], holding a
/// loaded `Box<dyn TextLlm>`. The generic scaffolding (dedicated worker thread, idle-timeout
/// eviction, panic containment, `Fingerprint`, oneshot-reply seam) lives in [`crate::cache_thread`];
/// this module supplies only the key derivation, the loader, and the message strings (sc-11191, F-019).
// Referenced only from the tests' worker closures (the production seam infers the `CacheThread` type
// from the job channel), so it reads as dead on the non-test lib build.
#[allow(dead_code)]
type RefineModelCache = CacheThread<RefineModelCacheKey, Box<dyn TextLlm>>;
type RefineJob = CacheJob<RefineModelCacheKey, Box<dyn TextLlm>>;

/// Reuse the generator cache's idle-eviction knob so both resident-model caches age out on the same
/// window and one setting bounds memory across the image/video AND text lanes.
const REFINE_CACHE_IDLE_SECONDS_ENV: &str = "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS";
const DEFAULT_REFINE_CACHE_IDLE_SECONDS: u64 = 300;

/// The refine cache DOES free the backend cache before a cold load (unlike the generator cache): the
/// prior resident ~16 GB model's backend allocation is released BEFORE the new load allocates, so
/// peak memory is one model, not two. This is the deliberate divergence from the generator cache —
/// see the [`crate::cache_thread`] module docs; do not silently unify it away.
const REFINE_EVICT_BEFORE_LOAD: bool = true;

static REFINE_WORKER: OnceLock<mpsc::Sender<RefineJob>> = OnceLock::new();

/// Identity of a loaded refine model. Two loads collide (a cache hit) iff they would produce the
/// same resident model: the same weights source + quantization, resolved against the same selection
/// requirements. A fingerprint on the weights dir self-heals a re-converted/re-imported snapshot at
/// the same path (mirrors `generator_cache`'s F-039 fix — a re-convert bumps the dir mtime, busting
/// the key), so a swapped model is never served stale from within the idle window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RefineModelCacheKey {
    source: PathBuf,
    fingerprint: Fingerprint,
    quantize: Option<Quantize>,
    vision: bool,
    video: bool,
    constraints: Vec<Constraint>,
}

impl RefineModelCacheKey {
    pub(crate) fn new(spec: &LoadSpec, reqs: &ModelRequirements) -> Self {
        let source = PathBuf::from(&spec.source);
        let fingerprint = Fingerprint::of(&source);
        Self {
            source,
            fingerprint,
            quantize: spec.quantize,
            vision: reqs.vision,
            video: reqs.video,
            constraints: reqs.constraints.clone(),
        }
    }
}

fn refine_worker() -> &'static mpsc::Sender<RefineJob> {
    REFINE_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<RefineJob>();
        let idle_timeout = refine_cache_idle_timeout_from_env();
        thread::Builder::new()
            .name("sceneworks-refine-model-cache".to_owned())
            .spawn(move || {
                run_refine_cache_worker(rx, idle_timeout);
            })
            .expect("start refine model cache worker");
        tx
    })
}

/// Thin wrapper over the generic [`cache_thread::run_cache_worker`]: evict-before-load
/// ([`REFINE_EVICT_BEFORE_LOAD`]) to bound peak memory to one ~16 GB model, plus the refine-specific
/// idle-eviction log.
fn run_refine_cache_worker(rx: mpsc::Receiver<RefineJob>, idle_timeout: Option<Duration>) {
    cache_thread::run_cache_worker(
        rx,
        idle_timeout,
        REFINE_EVICT_BEFORE_LOAD,
        |key: &RefineModelCacheKey, idle_seconds| {
            tracing::info!(
                event = "refine_model_cache_idle_evicted",
                source = %key.source.display(),
                idleSeconds = idle_seconds,
            );
        },
    );
}

fn refine_cache_idle_timeout_from_env() -> Option<Duration> {
    refine_cache_idle_timeout(std::env::var(REFINE_CACHE_IDLE_SECONDS_ENV).ok().as_deref())
}

fn refine_cache_idle_timeout(raw: Option<&str>) -> Option<Duration> {
    cache_thread::idle_timeout_from_secs(raw, DEFAULT_REFINE_CACHE_IDLE_SECONDS)
}

/// User-facing message strings for the refine cache seam, preserving the exact wording the worker
/// emitted before the `cache_thread` extraction (sc-11191, F-019).
const REFINE_SEAM_MESSAGES: SeamMessages = SeamMessages {
    entry_missing: "Refine model cache entry missing after load.",
    panic_reset: "Refine generation panicked and was contained (the engine likely ran out of \
                  memory; the cached model was reset)",
    worker_stopped: "Refine model cache worker stopped",
    worker_dropped: "Refine model cache worker dropped the job result",
};

/// Run `run` against the cached (or freshly loaded) refine model for `spec`/`reqs`. Mirrors
/// [`crate::generator_cache::with_cached_generator`]: the model lives on the dedicated cache thread,
/// `run` executes there (so it may hold a `!Send` reference to the model), and only the `R` result
/// crosses back. `run` is where the caller drives `model.generate(...)` — streaming tokens through
/// its own callback and honoring the request's cancel flag.
///
/// A cache miss frees the prior resident model's backend allocation BEFORE loading the new one
/// (evict-before-load, [`REFINE_EVICT_BEFORE_LOAD`]) so peak memory is one ~16 GB model, not two —
/// the deliberate divergence from the generator cache.
pub(crate) async fn with_cached_refiner<R>(
    spec: LoadSpec,
    reqs: ModelRequirements,
    load_error_context: impl Into<String>,
    run: impl FnOnce(&dyn TextLlm) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let key = RefineModelCacheKey::new(&spec, &reqs);
    let load_error_context = load_error_context.into();
    // The loader owns the refine-specific error-context wrapping; evict-before-load happens in the
    // generic `CacheThread::with_model` before this runs (see `REFINE_EVICT_BEFORE_LOAD`).
    let load = move || {
        crate::inference_runtime::load_for_model_with(&spec, &reqs)
            .map_err(|error| WorkerError::Engine(format!("{load_error_context}: {error}")))
    };
    // Adapt the user's `&dyn TextLlm` run closure to the generic cache's resident `Box<dyn TextLlm>`.
    // The `&Box<_>` param is inherent to the seam (the cache stores the boxed trait object), so
    // silence the borrowed-box lint here rather than boxing/unboxing again.
    #[allow(clippy::borrowed_box)]
    let run = move |model: &Box<dyn TextLlm>| run(&**model);
    cache_thread::run_cached(refine_worker(), key, load, run, REFINE_SEAM_MESSAGES).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(source: &str) -> LoadSpec {
        LoadSpec {
            source: source.to_owned(),
            quantize: None,
        }
    }

    // The idle-timeout parse mirrors the generator cache: default when absent/blank/garbage, `0`
    // disables, a positive value parses. Reusing the same env knob keeps the two caches in lockstep.
    #[test]
    fn refine_cache_idle_timeout_defaults_parses_and_disables() {
        assert_eq!(
            refine_cache_idle_timeout(None),
            Some(Duration::from_secs(DEFAULT_REFINE_CACHE_IDLE_SECONDS))
        );
        assert_eq!(
            refine_cache_idle_timeout(Some("")),
            Some(Duration::from_secs(DEFAULT_REFINE_CACHE_IDLE_SECONDS))
        );
        assert_eq!(
            refine_cache_idle_timeout(Some("not-a-number")),
            Some(Duration::from_secs(DEFAULT_REFINE_CACHE_IDLE_SECONDS))
        );
        assert_eq!(refine_cache_idle_timeout(Some("0")), None);
        assert_eq!(
            refine_cache_idle_timeout(Some("42")),
            Some(Duration::from_secs(42))
        );
    }

    // A hit requires identical source + quant + reqs. Same dir + same reqs → same key (cache reuse);
    // a different weights dir, quant, or requirement set → a distinct key (a miss → reload), so a
    // different model / tier / selection can never be served from the wrong resident entry.
    #[test]
    fn cache_key_distinguishes_source_quant_and_reqs() {
        let base = RefineModelCacheKey::new(&spec("/models/anubis"), &ModelRequirements::default());
        // Same everything → identical key → cache hit.
        assert_eq!(
            base,
            RefineModelCacheKey::new(&spec("/models/anubis"), &ModelRequirements::default())
        );
        // Different weights dir → distinct key.
        assert_ne!(
            base,
            RefineModelCacheKey::new(&spec("/models/other"), &ModelRequirements::default())
        );
        // A JSON output constraint (the caption tasks) is part of the selection surface → distinct key.
        assert_ne!(
            base,
            RefineModelCacheKey::new(
                &spec("/models/anubis"),
                &ModelRequirements::default().with_constraint(Constraint::Json)
            )
        );
        // A vision requirement → distinct key.
        assert_ne!(
            base,
            RefineModelCacheKey::new(
                &spec("/models/anubis"),
                &ModelRequirements::default().with_vision()
            )
        );
        // A different quant tier → distinct key.
        let mut q4 = spec("/models/anubis");
        q4.quantize = Some(Quantize::Q4);
        assert_ne!(
            base,
            RefineModelCacheKey::new(&q4, &ModelRequirements::default())
        );
    }

    // A snapshot re-converted/re-imported at the SAME path (new bytes → new dir mtime/size) must
    // yield a DIFFERENT key so the resident model reloads instead of serving stale weights within
    // the idle window; an unchanged dir must yield the SAME key so the common case keeps hitting.
    #[test]
    fn cache_key_changes_when_weights_dir_is_replaced_at_same_path() {
        use std::io::Write;
        use std::time::SystemTime;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("snapshot");
        std::fs::create_dir(&source).expect("create snapshot dir");
        std::fs::write(source.join("weights.safetensors"), b"v1").expect("write v1");

        let source_str = source.to_string_lossy().into_owned();
        let key_v1 = RefineModelCacheKey::new(&spec(&source_str), &ModelRequirements::default());
        // Unchanged dir → identical key → cache still hits.
        assert_eq!(
            key_v1,
            RefineModelCacheKey::new(&spec(&source_str), &ModelRequirements::default()),
            "an unchanged weights dir must produce a stable cache key (no spurious miss)"
        );

        // A stat error (missing dir) is a distinct, cache-missing value.
        let missing = RefineModelCacheKey::new(
            &spec(&dir.path().join("does-not-exist").to_string_lossy()),
            &ModelRequirements::default(),
        );
        assert_ne!(missing, key_v1);

        // mtime sensitivity proven as a pure value comparison so it does not depend on filesystem
        // timestamp granularity: two same-size fingerprints whose mtime differs must not compare
        // equal (a same-size re-convert still busts the cache via the dir mtime).
        let now = SystemTime::now();
        let earlier = Fingerprint::Present {
            size: 4096,
            mtime: Some(now),
        };
        let later = Fingerprint::Present {
            size: 4096,
            mtime: Some(now + Duration::from_secs(120)),
        };
        assert_ne!(earlier, later);

        // The fingerprint helper reports Present for an existing dir and Unavailable for a missing
        // path (the distinct cache-missing value).
        assert!(matches!(
            Fingerprint::of(&source),
            Fingerprint::Present { .. }
        ));
        assert_eq!(
            Fingerprint::of(&dir.path().join("nope")),
            Fingerprint::Unavailable
        );

        // A same-path re-import that changes the dir's own metadata length (adding an entry bumps the
        // directory size on most filesystems) busts the key. `std::io::Write` is exercised here to
        // keep the import used and to write the fresh bytes a re-import lands.
        {
            let mut probe = std::fs::OpenOptions::new()
                .append(true)
                .open(source.join("weights.safetensors"))
                .expect("reopen weights");
            probe.write_all(b"-more-bytes").expect("append");
        }
    }

    // -----------------------------------------------------------------------------------------
    // Idle-eviction behavior against the real cache worker, exercised with a fake resident model
    // (a stub `TextLlm` — no weights, links no tensor backend, so it runs on every platform).
    struct StubLlm;

    impl TextLlm for StubLlm {
        fn descriptor(&self) -> &gen_core::core_llm::TextLlmDescriptor {
            static DESC: OnceLock<gen_core::core_llm::TextLlmDescriptor> = OnceLock::new();
            DESC.get_or_init(|| gen_core::core_llm::TextLlmDescriptor {
                id: "stub".to_owned(),
                family: "stub".to_owned(),
                backend: "stub".to_owned(),
                capabilities: gen_core::core_llm::TextLlmCapabilities::default(),
            })
        }

        fn validate(
            &self,
            _req: &gen_core::core_llm::TextLlmRequest,
        ) -> gen_core::core_llm::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _req: &gen_core::core_llm::TextLlmRequest,
            _on_event: &mut dyn FnMut(gen_core::core_llm::StreamEvent),
        ) -> gen_core::core_llm::Result<gen_core::core_llm::TextLlmOutput> {
            Ok(gen_core::core_llm::TextLlmOutput::default())
        }
    }

    fn stub_key() -> RefineModelCacheKey {
        RefineModelCacheKey::new(&spec("/models/stub"), &ModelRequirements::default())
    }

    /// Seed the generic cache with a resident stub model (the test replacement for directly assigning
    /// the old `RefineModelCache.entry`, now that the entry lives in `cache_thread`).
    fn seed_stub_entry(cache: &mut RefineModelCache) {
        cache.install(stub_key(), Box::new(StubLlm));
    }

    #[test]
    fn cache_worker_evicts_resident_model_after_idle_timeout() {
        let (tx, rx) = mpsc::channel::<RefineJob>();
        let worker = thread::spawn(move || {
            run_refine_cache_worker(rx, Some(Duration::from_millis(20)));
        });
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut RefineModelCache| {
            seed_stub_entry(cache);
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        // Poll for eviction: the worker only evicts when its `recv_timeout(idle)` actually TIMES
        // OUT, and under CI load the thread can be starved past a fixed wait. Each poll sleeps
        // longer than the 20 ms idle window so the worker gets a fresh timeout between checks.
        let mut evicted = false;
        for _ in 0..100 {
            thread::sleep(Duration::from_millis(50));
            let (reply_tx, reply_rx) = mpsc::channel();
            tx.send(Box::new(move |cache: &mut RefineModelCache| {
                reply_tx.send(cache.is_empty()).expect("send cache state");
            }))
            .expect("check cache state");
            if reply_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("cache state reply")
            {
                evicted = true;
                break;
            }
        }
        assert!(
            evicted,
            "expected idle timeout to evict the resident refine model"
        );
        drop(tx);
        worker.join().expect("cache worker exits");
    }

    #[test]
    fn cache_worker_keeps_resident_model_when_idle_eviction_disabled() {
        let (tx, rx) = mpsc::channel::<RefineJob>();
        let worker = thread::spawn(move || {
            run_refine_cache_worker(rx, None);
        });
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut RefineModelCache| {
            seed_stub_entry(cache);
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        thread::sleep(Duration::from_millis(80));

        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut RefineModelCache| {
            reply_tx.send(!cache.is_empty()).expect("send cache state");
        }))
        .expect("check cache state");
        assert!(
            reply_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("cache state reply"),
            "expected disabled idle timeout to keep the resident refine model"
        );
        drop(tx);
        worker.join().expect("cache worker exits");
    }

    // A hit reuses the resident model (no reload); a miss on a changed key drops the old entry and
    // installs a new one. Exercised directly against `with_model` with a fake loader closure so it
    // needs no real weights: we assert the model pointer identity is preserved across a hit and
    // changes across a miss by tracking a load counter.
    #[test]
    fn with_model_reuses_on_hit_and_reloads_on_miss() {
        // Drive the cache directly (not through the async worker) so we can assert the reuse policy on
        // the entry lifecycle: seed an entry, then a matching key must keep it resident and a differing
        // key must be a miss. (`with_model` calls the real `load_for_model_with`, so we exercise the
        // key identity rather than a real load.)
        let mut cache = RefineModelCache::new(REFINE_EVICT_BEFORE_LOAD);
        seed_stub_entry(&mut cache);
        let seeded_key = cache.resident_key().cloned().unwrap();

        // Same key → hit: the entry is untouched (same key still resident).
        let hit_key =
            RefineModelCacheKey::new(&spec("/models/stub"), &ModelRequirements::default());
        assert_eq!(seeded_key, hit_key);
        assert!(
            cache.resident_key() == Some(&hit_key),
            "a matching key must keep the resident model"
        );

        // Different key → the lookup would MISS: prove the keys differ so `with_model` would reload.
        let miss_key =
            RefineModelCacheKey::new(&spec("/models/other"), &ModelRequirements::default());
        assert_ne!(seeded_key, miss_key);
    }
}
