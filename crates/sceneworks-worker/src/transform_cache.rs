//! Resident-model cache for non-prompt image→image [`gen_core::Transform`]s (sc-25683, epic 25678).
//!
//! The Iris-3B restoration task is a `Transform` (`iris_3b_restore`), not a `Generator`, so it cannot
//! live in [`crate::generator_cache`]'s `Box<dyn Generator>` slot. Rather than fork the cache
//! semantics, this is a third thin wrapper over the same [`crate::cache_thread::CacheThread`]
//! scaffolding the generator and refine caches already share: one dedicated OS thread owns at most
//! one loaded transform keyed by its load identity, reused across jobs, panic-contained, and
//! idle-evicted on the SAME knob + window as the other two caches
//! (`SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS`, default 300 s).
//!
//! ## One resident engine across the generator and transform caches
//!
//! SeedVR2 — the only other model-backed upscaler — runs through the generator cache, so loading it
//! evicts any cached image-generation engine and peak memory stays one model. The transform cache
//! keeps that invariant across the two caches instead of letting a ~6 GB restorer sit beside a
//! resident generator:
//!
//! * a transform COLD load first evicts the resident generator (the injected `before_cold_load` hook,
//!   run on the transform thread only for a genuine miss — a warm hit never touches the generator);
//! * a generator load first evicts a resident transform ([`evict_resident_transform`], called from
//!   the generator cache's async entry points; a no-op when no transform was ever loaded).
//!
//! Like the refine cache, a miss frees the prior resident transform's backend allocation BEFORE the
//! new load ([`TRANSFORM_EVICT_BEFORE_LOAD`]), so two transforms are never resident at once either.
//!
//! Memory ADMISSION is not a cache concern here: a transform's declared floor is checked by its job
//! before the cache is entered (see `iris_restore::admission_error`), exactly as the bespoke
//! SeedVR2/Mochi gates run ahead of the generator cache.

use std::path::{Path, PathBuf};
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::Duration;

use gen_core::{LoadSpec, Precision, Quant, Transform, WeightsSource};

use crate::cache_thread::{self, CacheJob, Fingerprint, SeamMessages};
use crate::{WorkerError, WorkerResult};

type TransformJob = CacheJob<TransformIdentity, Box<dyn Transform>>;

/// Same knob and default as the generator and refine caches, so one setting bounds resident model
/// memory across all three lanes.
const TRANSFORM_CACHE_IDLE_SECONDS_ENV: &str = "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS";
const DEFAULT_TRANSFORM_CACHE_IDLE_SECONDS: u64 = 300;

/// Free the prior resident transform's backend allocation before a cold load (the refine cache's
/// policy): a transform has no fit-gate/residency ladder that sizes the load inside the loader, so
/// the only way to keep peak memory at one model is to drop the old one first.
const TRANSFORM_EVICT_BEFORE_LOAD: bool = true;

static TRANSFORM_WORKER: OnceLock<mpsc::Sender<TransformJob>> = OnceLock::new();

/// Identity of a loaded transform. Two loads collide (a warm hit) iff they would produce the same
/// resident model: the same registered id, the same weights path with the same on-disk content proxy
/// (the directory's own fingerprint plus every direct entry's `(size, mtime)` — so a re-installed
/// `upscaler/` export at the same path reloads instead of serving stale tensors, the F-039 rule the
/// other caches follow), and the same numeric load options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TransformIdentity {
    engine_id: String,
    weights: PathBuf,
    weights_fingerprint: Fingerprint,
    entries: Vec<(String, Fingerprint)>,
    quantize: Option<Quant>,
    precision: Precision,
}

impl TransformIdentity {
    pub(crate) fn new(engine_id: &str, spec: &LoadSpec) -> Self {
        let weights = match &spec.weights {
            WeightsSource::Dir(path) | WeightsSource::File(path) => path.clone(),
        };
        Self {
            engine_id: engine_id.to_owned(),
            weights_fingerprint: Fingerprint::of(&weights),
            entries: directory_entry_fingerprints(&weights),
            weights,
            quantize: spec.quantize,
            precision: spec.precision,
        }
    }
}

/// `(name, fingerprint)` of every direct entry of `dir`, sorted (empty for a file or an unreadable
/// dir — the load that follows surfaces the real error). `Fingerprint::of` follows symlinks, so an
/// HF snapshot entry fingerprints its blob.
fn directory_entry_fingerprints(dir: &Path) -> Vec<(String, Fingerprint)> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, Fingerprint)> = read
        .filter_map(Result::ok)
        .map(|entry| {
            (
                entry.file_name().to_string_lossy().into_owned(),
                Fingerprint::of(&entry.path()),
            )
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn transform_worker() -> &'static mpsc::Sender<TransformJob> {
    TRANSFORM_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<TransformJob>();
        let idle_timeout = transform_cache_idle_timeout(
            std::env::var(TRANSFORM_CACHE_IDLE_SECONDS_ENV)
                .ok()
                .as_deref(),
        );
        thread::Builder::new()
            .name("sceneworks-transform-cache".to_owned())
            .spawn(move || run_transform_cache_worker(rx, idle_timeout))
            .expect("start transform cache worker");
        tx
    })
}

fn run_transform_cache_worker(rx: mpsc::Receiver<TransformJob>, idle_timeout: Option<Duration>) {
    cache_thread::run_cache_worker(
        rx,
        idle_timeout,
        TRANSFORM_EVICT_BEFORE_LOAD,
        |key: &TransformIdentity, idle_seconds| {
            tracing::info!(
                event = "transform_cache_idle_evicted",
                engine = %key.engine_id,
                weights = %key.weights.display(),
                idleSeconds = idle_seconds,
            );
        },
    );
}

fn transform_cache_idle_timeout(raw: Option<&str>) -> Option<Duration> {
    cache_thread::idle_timeout_from_secs(raw, DEFAULT_TRANSFORM_CACHE_IDLE_SECONDS)
}

const TRANSFORM_SEAM_MESSAGES: SeamMessages = SeamMessages {
    entry_missing: "Transform cache entry missing after load.",
    panic_reset: "Transform run panicked and was contained (the engine likely ran out of memory; \
                  the cached transform was reset)",
    worker_stopped: "Transform cache worker stopped",
    worker_dropped: "Transform cache worker dropped the job result",
};

/// The production pre-cold-load hook: evict the resident image/video generator so the incoming
/// transform is the only resident engine. Runs on the transform cache thread (a plain OS thread),
/// so it waits for the generator thread's reply synchronously; the generator thread never waits on
/// the transform thread (its own eviction of a resident transform runs from async code), so this
/// cannot deadlock.
fn evict_resident_generator_before_transform_load() -> WorkerResult<()> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let evicted = crate::generator_cache::evict_cached_generator_blocking()?;
        if evicted {
            tracing::info!(event = "transform_cache_evicted_resident_generator");
        }
    }
    Ok(())
}

/// Run `run` against the cached (or freshly loaded) transform `engine_id` for `spec`, loading it
/// through `load_transform` on a miss — the production loader is
/// [`crate::inference_runtime::load_transform`]; a test injects a stub transform here, the same
/// `_using` seam the generator cache exposes.
pub(crate) async fn with_cached_transform_using<R>(
    engine_id: &str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    load_transform: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Transform>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Transform) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    with_cached_transform_on(
        transform_worker(),
        engine_id,
        spec,
        load_error_context,
        evict_resident_generator_before_transform_load,
        load_transform,
        run,
    )
    .await
}

/// [`with_cached_transform_using`] against a caller-supplied cache worker and pre-cold-load hook —
/// the seam the unit tests drive their own isolated cache thread through.
async fn with_cached_transform_on<R>(
    worker: &mpsc::Sender<TransformJob>,
    engine_id: &str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    before_cold_load: impl FnOnce() -> WorkerResult<()> + Send + 'static,
    load_transform: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Transform>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Transform) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let key = TransformIdentity::new(engine_id, &spec);
    let engine_id = engine_id.to_owned();
    let load_error_context = load_error_context.into();
    let load = move || {
        load_transform(&engine_id, &spec).map_err(|error| match error {
            // A typed refusal (wrong task, missing file, unsupported option) stays user-facing.
            gen_core::Error::Unsupported(_) => {
                WorkerError::InvalidPayload(format!("{load_error_context}: {error}"))
            }
            other => WorkerError::Engine(format!("{load_error_context}: {other}")),
        })
    };
    #[allow(clippy::borrowed_box)]
    let run = move |model: &Box<dyn Transform>, _access| run(&**model);
    cache_thread::run_cached_with_access_after_cold_admission(
        worker,
        key,
        move |_previous| before_cold_load(),
        load,
        run,
        TRANSFORM_SEAM_MESSAGES,
    )
    .await
}

/// Drop the resident transform (if any) so a generator load does not stack on top of it. A no-op —
/// and no thread is started — when no transform has ever been loaded in this process. Returns
/// whether something was evicted.
pub(crate) async fn evict_resident_transform() -> WorkerResult<bool> {
    match TRANSFORM_WORKER.get() {
        Some(worker) => evict_resident_transform_on(worker).await,
        None => Ok(false),
    }
}

async fn evict_resident_transform_on(worker: &mpsc::Sender<TransformJob>) -> WorkerResult<bool> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<bool>();
    let job: TransformJob = Box::new(move |cache| {
        let evicted = cache.evict().is_some();
        if evicted {
            cache_thread::release_backend_cache_after_evict();
        }
        let _ = reply_tx.send(evicted);
    });
    worker
        .send(job)
        .map_err(|_| WorkerError::Engine(TRANSFORM_SEAM_MESSAGES.worker_stopped.to_owned()))?;
    reply_rx
        .await
        .map_err(|_| WorkerError::Engine(TRANSFORM_SEAM_MESSAGES.worker_dropped.to_owned()))
}

/// Test seam: load a stub transform from `dir` into the PROCESS-GLOBAL cache through the production
/// entry point (including its pre-cold-load generator eviction). Callers hold
/// [`crate::test_env::env_lock`] so tests touching the global cache serialize.
#[cfg(test)]
pub(crate) async fn load_stub_into_global_cache(dir: &Path) -> WorkerResult<()> {
    with_cached_transform_using(
        "stub_restore",
        LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())),
        "stub load",
        |_, _| Ok(tests::stub("stub_restore")),
        |_| Ok(()),
    )
    .await
}

/// Test seam: whether the process-global cache holds a resident transform (without evicting it).
#[cfg(test)]
pub(crate) async fn global_transform_resident() -> bool {
    match TRANSFORM_WORKER.get() {
        Some(worker) => transform_resident_on(worker).await,
        None => false,
    }
}

/// Test seam: a dedicated (non-global) transform cache holding a stub, so a generator-cache test can
/// check the injected transform eviction without racing the process-global cache.
#[cfg(test)]
pub(crate) struct IsolatedTransformCache(mpsc::Sender<TransformJob>);

#[cfg(test)]
impl IsolatedTransformCache {
    pub(crate) async fn with_stub(dir: &Path) -> Self {
        let worker = tests::spawn_isolated_cache();
        with_cached_transform_on(
            &worker,
            "stub_restore",
            LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())),
            "stub load",
            || Ok(()),
            |_, _| Ok(tests::stub("stub_restore")),
            |_| Ok(()),
        )
        .await
        .expect("stub transform loads");
        Self(worker)
    }

    pub(crate) fn evict(
        &self,
    ) -> impl std::future::Future<Output = WorkerResult<bool>> + Send + '_ {
        evict_resident_transform_on(&self.0)
    }

    pub(crate) async fn resident(&self) -> bool {
        transform_resident_on(&self.0).await
    }
}

#[cfg(test)]
async fn transform_resident_on(worker: &mpsc::Sender<TransformJob>) -> bool {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<bool>();
    let job: TransformJob = Box::new(move |cache| {
        let _ = reply_tx.send(cache.resident_model().is_some());
    });
    worker.send(job).expect("transform worker alive");
    reply_rx.await.expect("transform worker replies")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gen_core::{Image, TransformCapabilities, TransformDescriptor, TransformRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct StubTransform {
        descriptor: TransformDescriptor,
    }

    impl Transform for StubTransform {
        fn descriptor(&self) -> &TransformDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &TransformRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn apply(
            &self,
            req: &TransformRequest,
            _on_progress: &mut dyn FnMut(gen_core::Progress),
        ) -> gen_core::Result<Image> {
            Ok(req.image.clone())
        }
    }

    pub(crate) fn stub(id: &'static str) -> Box<dyn Transform> {
        Box::new(StubTransform {
            descriptor: TransformDescriptor {
                id,
                family: "stub",
                backend: "stub",
                capabilities: TransformCapabilities::default(),
            },
        })
    }

    pub(super) fn spawn_isolated_cache() -> mpsc::Sender<TransformJob> {
        let (tx, rx) = mpsc::channel::<TransformJob>();
        thread::spawn(move || run_transform_cache_worker(rx, None));
        tx
    }

    fn staged_export(root: &Path) -> PathBuf {
        let dir = root.join("upscaler");
        std::fs::create_dir_all(&dir).unwrap();
        for file in [
            "config.yaml",
            "model.safetensors",
            "empty_prompt.safetensors",
        ] {
            std::fs::write(dir.join(file), file.as_bytes()).unwrap();
        }
        dir
    }

    async fn run_once(
        worker: &mpsc::Sender<TransformJob>,
        dir: &Path,
        loads: &Arc<AtomicUsize>,
        cold_hooks: &Arc<AtomicUsize>,
    ) -> WorkerResult<&'static str> {
        let (loads, cold_hooks) = (loads.clone(), cold_hooks.clone());
        with_cached_transform_on(
            worker,
            "stub_restore",
            LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())),
            "stub load",
            move || {
                cold_hooks.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            move |id, _spec| {
                loads.fetch_add(1, Ordering::SeqCst);
                assert_eq!(id, "stub_restore");
                Ok(stub("stub_restore"))
            },
            |transform| Ok(transform.descriptor().id),
        )
        .await
    }

    /// A warm hit reuses the resident transform and never runs the pre-cold-load hook (which in
    /// production evicts the generator); a miss runs the hook exactly once, before the load.
    #[tokio::test]
    async fn warm_hits_reuse_the_resident_transform_and_skip_the_cold_hook() {
        let worker = spawn_isolated_cache();
        let temp = tempfile::tempdir().unwrap();
        let dir = staged_export(temp.path());
        let (loads, hooks) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        assert_eq!(
            run_once(&worker, &dir, &loads, &hooks).await.unwrap(),
            "stub_restore"
        );
        run_once(&worker, &dir, &loads, &hooks).await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1, "the second run is warm");
        assert_eq!(hooks.load(Ordering::SeqCst), 1, "only the cold load evicts");
    }

    /// Replacing a file of the export at the same path is a different identity (F-039): the stale
    /// resident is dropped and the new bytes load.
    #[tokio::test]
    async fn a_replaced_export_file_reloads() {
        let worker = spawn_isolated_cache();
        let temp = tempfile::tempdir().unwrap();
        let dir = staged_export(temp.path());
        let (loads, hooks) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        run_once(&worker, &dir, &loads, &hooks).await.unwrap();
        std::fs::write(
            dir.join("model.safetensors"),
            b"a re-installed, larger export",
        )
        .unwrap();
        run_once(&worker, &dir, &loads, &hooks).await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert_eq!(hooks.load(Ordering::SeqCst), 2);
    }

    /// Eviction frees the slot (so the next run is cold again), and reports nothing on an empty
    /// cache.
    #[tokio::test]
    async fn eviction_frees_the_slot() {
        let worker = spawn_isolated_cache();
        let temp = tempfile::tempdir().unwrap();
        let dir = staged_export(temp.path());
        let (loads, hooks) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        assert!(!evict_resident_transform_on(&worker).await.unwrap());
        run_once(&worker, &dir, &loads, &hooks).await.unwrap();
        assert!(evict_resident_transform_on(&worker).await.unwrap());
        assert!(!evict_resident_transform_on(&worker).await.unwrap());
        run_once(&worker, &dir, &loads, &hooks).await.unwrap();
        assert_eq!(
            loads.load(Ordering::SeqCst),
            2,
            "an evicted transform reloads"
        );
    }

    /// A failing pre-cold-load hook refuses the load (nothing loads, nothing becomes resident), and
    /// a typed engine refusal from the loader stays a user-facing `InvalidPayload`.
    #[tokio::test]
    async fn a_refused_cold_load_leaves_nothing_resident_and_keeps_typed_errors() {
        let worker = spawn_isolated_cache();
        let temp = tempfile::tempdir().unwrap();
        let dir = staged_export(temp.path());
        let error = with_cached_transform_on(
            &worker,
            "stub_restore",
            LoadSpec::new(WeightsSource::Dir(dir.clone())),
            "stub load",
            || Err(WorkerError::Engine("generator would not evict".to_owned())),
            |_, _| -> gen_core::Result<Box<dyn Transform>> {
                panic!("must not load after a refused hook")
            },
            |_| Ok(()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("generator would not evict"));
        assert!(!evict_resident_transform_on(&worker).await.unwrap());

        let error = with_cached_transform_on(
            &worker,
            "stub_restore",
            LoadSpec::new(WeightsSource::Dir(dir)),
            "Iris restoration load",
            || Ok(()),
            |_, _| -> gen_core::Result<Box<dyn Transform>> {
                Err(gen_core::Error::Unsupported(
                    "this export is the depth task, not restoration".to_owned(),
                ))
            },
            |_| Ok(()),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&error, WorkerError::InvalidPayload(message)
                if message.contains("Iris restoration load") && message.contains("depth task")),
            "{error:?}"
        );
    }

    #[test]
    fn idle_timeout_shares_the_generator_cache_knob() {
        assert_eq!(
            TRANSFORM_CACHE_IDLE_SECONDS_ENV,
            "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS"
        );
        assert_eq!(
            transform_cache_idle_timeout(None),
            Some(Duration::from_secs(DEFAULT_TRANSFORM_CACHE_IDLE_SECONDS))
        );
        assert_eq!(transform_cache_idle_timeout(Some("0")), None);
    }
}
