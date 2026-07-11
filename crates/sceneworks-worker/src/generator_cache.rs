use std::path::{Path, PathBuf};
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime};

use gen_core::{
    AdapterKind, AdapterSpec, Generator, LoadSpec, MoeExpert, Precision, Quant, WeightsSource,
};
use tokio::sync::oneshot;

use crate::{WorkerError, WorkerResult};

type GeneratorJob = Box<dyn FnOnce(&mut GeneratorCache) + Send + 'static>;

const GENERATOR_CACHE_IDLE_SECONDS_ENV: &str = "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS";
const DEFAULT_GENERATOR_CACHE_IDLE_SECONDS: u64 = 300;

static GENERATOR_WORKER: OnceLock<mpsc::Sender<GeneratorJob>> = OnceLock::new();

struct GeneratorCache {
    entry: Option<GeneratorCacheEntry>,
}

struct GeneratorCacheEntry {
    key: GeneratorCacheKey,
    generator: Box<dyn Generator>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GeneratorCacheKey {
    engine_id: String,
    weights: CacheWeightsSource,
    quantize: Option<Quant>,
    precision: Precision,
    control: Option<CacheWeightsSource>,
    extra_controls: Vec<CacheWeightsSource>,
    ip_adapter: Option<CacheWeightsSource>,
    adapters: Vec<CacheAdapterSpec>,
    // Per-generation PiD decoder aux-weights (epic 7840, sc-7849): `(checkpoint, gemma)` when the
    // generator was loaded with `LoadSpec::with_pid`, else `None`. Keyed so a PiD-equipped load is a
    // distinct cache entry from the plain VAE load — toggling `usePid` reloads rather than reusing a
    // generator with the wrong decoder.
    pid: Option<(CacheWeightsSource, CacheWeightsSource)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CacheWeightsSource {
    Dir(PathBuf, Fingerprint),
    File(PathBuf, Fingerprint),
}

/// Content-change proxy for a weights/adapter file (or HF snapshot dir) referenced in the cache key
/// (sc-8841, F-039). The pre-fingerprint key identified weights/adapters by path + scale only, so a
/// file replaced at the SAME path — a re-imported LoRA, a re-converted checkpoint — was served from
/// the resident generator with the OLD tensors until the 300 s idle timeout ("my new LoRA does
/// nothing"). Folding `(size, mtime)` into the key self-heals for ANY overwrite path (in-process
/// re-import/re-convert, out-of-band replacement, manual swap) without an explicit evict hook that
/// only fires for the code paths that remember to call it.
///
/// We deliberately stat metadata rather than hashing contents: mtime+size is a cheap, good
/// content-change proxy, and hashing multi-GB weight files on every generation would be a severe
/// perf regression.
///
/// For a `Dir` (an HF snapshot) we fingerprint the DIRECTORY's own metadata. A re-convert lands via
/// the finalize/rename path (`finalize_converted_dir`), which replaces directory entries and bumps
/// the dir's mtime, so a re-converted snapshot at the same path is a distinct key. `size` for a dir
/// is its own metadata length (not the recursive content size) — it only needs to move on change,
/// not be meaningful; mtime carries the signal.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Fingerprint {
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
    /// stable across the lookup within a single generation (no mtime drift mid-request).
    fn of(path: &Path) -> Self {
        match std::fs::metadata(path) {
            Ok(meta) => Self::Present {
                size: meta.len(),
                mtime: meta.modified().ok(),
            },
            Err(_) => Self::Unavailable,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheAdapterSpec {
    path: PathBuf,
    fingerprint: Fingerprint,
    scale_bits: u32,
    kind: AdapterKind,
    pass_scale_bits: Option<Vec<u32>>,
    moe_expert: Option<MoeExpert>,
}

impl GeneratorCacheKey {
    pub(crate) fn from_load_spec(engine_id: &str, spec: &LoadSpec) -> Self {
        Self {
            engine_id: engine_id.to_owned(),
            weights: CacheWeightsSource::from(&spec.weights),
            quantize: spec.quantize,
            precision: spec.precision,
            control: spec.control.as_ref().map(CacheWeightsSource::from),
            extra_controls: spec
                .extra_controls
                .iter()
                .map(CacheWeightsSource::from)
                .collect(),
            ip_adapter: spec.ip_adapter.as_ref().map(CacheWeightsSource::from),
            adapters: spec.adapters.iter().map(CacheAdapterSpec::from).collect(),
            pid: spec.pid.as_ref().map(|pid| {
                (
                    CacheWeightsSource::from(&pid.checkpoint),
                    CacheWeightsSource::from(&pid.gemma),
                )
            }),
        }
    }
}

impl From<&WeightsSource> for CacheWeightsSource {
    fn from(source: &WeightsSource) -> Self {
        match source {
            WeightsSource::Dir(path) => Self::Dir(path.clone(), Fingerprint::of(path)),
            WeightsSource::File(path) => Self::File(path.clone(), Fingerprint::of(path)),
        }
    }
}

impl From<&AdapterSpec> for CacheAdapterSpec {
    fn from(spec: &AdapterSpec) -> Self {
        Self {
            path: spec.path.clone(),
            fingerprint: Fingerprint::of(&spec.path),
            scale_bits: spec.scale.to_bits(),
            kind: spec.kind,
            pass_scale_bits: spec
                .pass_scales
                .as_ref()
                .map(|scales| scales.iter().map(|scale| scale.to_bits()).collect()),
            moe_expert: spec.moe_expert,
        }
    }
}

impl GeneratorCache {
    fn new() -> Self {
        Self { entry: None }
    }

    /// Drop the resident generator so the next job reloads from scratch.
    fn evict(&mut self) -> Option<GeneratorCacheKey> {
        self.entry.take().map(|entry| entry.key)
    }

    fn with_generator<R>(
        &mut self,
        key: GeneratorCacheKey,
        spec: LoadSpec,
        load_error_context: String,
        run: impl FnOnce(&dyn Generator) -> WorkerResult<R>,
    ) -> WorkerResult<R> {
        if self.entry.as_ref().map_or(true, |entry| entry.key != key) {
            self.entry = None;
            // Pre-load unified-memory fit-gate (epic 10834 Phase 0, sc-10835): reject a model that
            // can't fit this machine's unified memory BEFORE gen_core::load allocates — a wired
            // overcommit SIGKILLs the worker mid-load rather than returning a catchable error. This
            // is the cold-load path only (a warm cache hit takes the branch above and skips it), so
            // an already-resident model is never re-gated.
            let weight_bytes = match &spec.weights {
                WeightsSource::Dir(dir) => crate::mlx_fit_gate::sum_safetensors_bytes(dir),
                WeightsSource::File(file) => std::fs::metadata(file).map_or(0, |meta| meta.len()),
            };
            crate::mlx_fit_gate::ensure_fits(weight_bytes, &key.engine_id)?;
            let generator = gen_core::load(&key.engine_id, &spec)
                .map_err(|error| crate::classify_engine_error(&load_error_context, error))?;
            self.entry = Some(GeneratorCacheEntry {
                key: key.clone(),
                generator,
            });
        }

        let Some(entry) = self.entry.as_ref() else {
            return Err(WorkerError::Engine(
                "Generator cache entry missing after load.".to_owned(),
            ));
        };
        run(entry.generator.as_ref())
    }
}

fn generator_worker() -> &'static mpsc::Sender<GeneratorJob> {
    GENERATOR_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let idle_timeout = generator_cache_idle_timeout_from_env();
        thread::Builder::new()
            .name("sceneworks-mlx-generator-cache".to_owned())
            .spawn(move || {
                run_generator_cache_worker(rx, idle_timeout);
            })
            .expect("start MLX generator cache worker");
        tx
    })
}

fn run_generator_cache_worker(rx: mpsc::Receiver<GeneratorJob>, idle_timeout: Option<Duration>) {
    let mut cache = GeneratorCache::new();
    loop {
        let job = match recv_generator_job(&rx, idle_timeout) {
            GeneratorWorkerEvent::Job(job) => job,
            GeneratorWorkerEvent::IdleTimeout => {
                if let Some(key) = cache.evict() {
                    release_backend_cache_after_evict();
                    // Documented event (docs/observability.md): expected idle-timeout
                    // eviction, so info level with the engine + idle window.
                    tracing::info!(
                        event = "generator_cache_idle_evicted",
                        engine = %key.engine_id,
                        idleSeconds = idle_timeout.map_or(0, |timeout| timeout.as_secs()),
                    );
                }
                continue;
            }
            GeneratorWorkerEvent::Disconnected => break,
        };
        // Backstop: contain any panic that escapes a job's own guard so this single
        // shared cache thread can never die and poison every later generation (sc-6067).
        // A job normally catches its own panic, replies with a clean error, and evicts;
        // this catches anything it misses. On a contained panic the cache is reset
        // because post-abort MLX/Metal state is suspect.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&mut cache))).is_err()
            && cache.evict().is_some()
        {
            release_backend_cache_after_evict();
        }
    }
}

enum GeneratorWorkerEvent {
    Job(GeneratorJob),
    IdleTimeout,
    Disconnected,
}

fn recv_generator_job(
    rx: &mpsc::Receiver<GeneratorJob>,
    idle_timeout: Option<Duration>,
) -> GeneratorWorkerEvent {
    match idle_timeout {
        Some(timeout) => match rx.recv_timeout(timeout) {
            Ok(job) => GeneratorWorkerEvent::Job(job),
            Err(mpsc::RecvTimeoutError::Timeout) => GeneratorWorkerEvent::IdleTimeout,
            Err(mpsc::RecvTimeoutError::Disconnected) => GeneratorWorkerEvent::Disconnected,
        },
        None => match rx.recv() {
            Ok(job) => GeneratorWorkerEvent::Job(job),
            Err(_) => GeneratorWorkerEvent::Disconnected,
        },
    }
}

fn generator_cache_idle_timeout_from_env() -> Option<Duration> {
    generator_cache_idle_timeout(
        std::env::var(GENERATOR_CACHE_IDLE_SECONDS_ENV)
            .ok()
            .as_deref(),
    )
}

fn generator_cache_idle_timeout(raw: Option<&str>) -> Option<Duration> {
    let seconds = raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_GENERATOR_CACHE_IDLE_SECONDS);
    (seconds > 0).then(|| Duration::from_secs(seconds))
}

#[cfg(all(target_os = "macos", not(test)))]
fn release_backend_cache_after_evict() {
    mlx_rs::memory::clear_cache();
}

/// Off-Mac (candle/CUDA) this is intentionally a no-op (epic 10765, sc-10766). candle's CUDA backend
/// uses cudarc's stream-ordered caching allocator, which exposes no `empty_cache` and does not reclaim
/// on `Device::synchronize()`: dropping the evicted generator already returns its pages to candle's
/// in-process pool (where the next load reuses them), but there is no supported way to hand those pages
/// back to the driver — so `nvidia-smi` resident VRAM stays flat across an evict regardless. A real
/// driver-level trim would need a candle/cudarc fork (tracked as a separate optional spike under epic
/// 10765), not this seam. The VRAM fit-gate therefore budgets on predicted peak, not resident deltas.
#[cfg(any(not(target_os = "macos"), test))]
fn release_backend_cache_after_evict() {}

/// Apply the user-configured GPU memory ceiling to the MLX runtime (epic 7819, sc-7820).
///
/// `bytes == 0` is a no-op — MLX keeps its own default budget (1.5× the device recommended working
/// set), so an unset limit is byte-identical to prior behavior. When non-zero we set two MLX knobs:
/// - `set_memory_limit` — soft backpressure: when active memory exceeds the limit MLX blocks and
///   waits for in-flight GPU work to drain rather than hard-failing. It is a target, not a hard
///   sandbox; a single oversized allocation can still exceed it (and on a too-low cap a model whose
///   working set genuinely needs more will thrash/swap or hit a Metal OOM — already contained by the
///   `catch_unwind` guard above).
/// - `set_wired_limit` — caps pinned (non-pageable) residency so the OS can reclaim the rest of
///   unified memory for other apps. macOS 15+.
///
/// We deliberately leave `set_cache_limit` at its default: forcing it low causes reallocation storms
/// between steps (the fork's own doc warns about this).
///
/// The MLX limit is **process-global**, so calling this once at worker startup (before any model
/// load) covers generations, upscales, AND LoRA training — even though training takes a separate
/// path from the generator cache.
/// The GPU memory ceiling (bytes) currently applied to this process's MLX runtime, so the live
/// sync (sc-7824) only re-applies on an actual change. `u64::MAX` is the "nothing applied yet"
/// sentinel — distinct from `0` ("no limit"), so the first real value (including a deliberate `0`
/// that clears a prior cap) always takes effect.
#[cfg(all(target_os = "macos", not(test)))]
static APPLIED_GPU_MEMORY_LIMIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

#[cfg(all(target_os = "macos", not(test)))]
fn set_gpu_memory_limit_inner(bytes: u64) {
    use std::sync::atomic::Ordering;
    let limit = bytes as usize;
    let previous_limit = mlx_rs::memory::set_memory_limit(limit);
    let previous_wired = mlx_rs::memory::set_wired_limit(limit);
    APPLIED_GPU_MEMORY_LIMIT.store(bytes, Ordering::SeqCst);
    tracing::info!(
        event = "gpu_memory_limit_applied",
        limitBytes = limit,
        previousLimitBytes = previous_limit,
        previousWiredLimitBytes = previous_wired,
        "applied user-configured GPU memory ceiling to the MLX runtime"
    );
}

#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn apply_gpu_memory_limit(bytes: u64) {
    if bytes == 0 {
        // Unset at startup: leave MLX on its own default budget — byte-identical to prior behavior.
        // (The live sync below still applies a deliberate `0` to *clear* a previously-set cap.)
        return;
    }
    set_gpu_memory_limit_inner(bytes);
}

/// Re-read the live GPU-memory-limit handoff file and apply it if it changed since the last applied
/// value (epic 7819, sc-7824). Called from the worker poll loop *between jobs*, so moving the
/// Settings slider takes effect on the next job without a worker restart. An absent file is a
/// no-op (the spawn-time `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES` value stays in force); a written `0`
/// actively clears a previously-applied cap (MLX treats `0` as "no limit").
#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn sync_gpu_memory_limit(config_dir: &Path) {
    use std::sync::atomic::Ordering;
    let path = sceneworks_core::app_paths::gpu_memory_limit_file(config_dir);
    let Some(bytes) = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
    else {
        return;
    };
    if APPLIED_GPU_MEMORY_LIMIT.load(Ordering::SeqCst) != bytes {
        set_gpu_memory_limit_inner(bytes);
    }
}

/// Publish a snapshot of the MLX runtime's process-global memory counters to the telemetry file for
/// the desktop Settings readout (epic 7819, sc-7825). `limit_bytes` reports the cap this worker has
/// actually applied (`0` = none), not MLX's internal default budget, so the UI can show "peak vs
/// limit" honestly. Best-effort: a write failure is ignored (the readout just goes stale).
#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn write_gpu_telemetry(config_dir: &Path) {
    use std::sync::atomic::Ordering;
    let applied = APPLIED_GPU_MEMORY_LIMIT.load(Ordering::SeqCst);
    let telemetry = sceneworks_core::app_paths::GpuMemoryTelemetry {
        active_bytes: mlx_rs::memory::get_active_memory() as u64,
        peak_bytes: mlx_rs::memory::get_peak_memory() as u64,
        cache_bytes: mlx_rs::memory::get_cache_memory() as u64,
        limit_bytes: if applied == u64::MAX { 0 } else { applied },
    };
    if let Ok(json) = serde_json::to_string(&telemetry) {
        let path = sceneworks_core::app_paths::gpu_telemetry_file(config_dir);
        let _ = std::fs::write(&path, json);
    }
}

/// Spawn a background task that republishes MLX memory telemetry on a short interval (epic 7819,
/// sc-7825). Runs independently of the job poll loop so the readout reflects usage *during* a
/// generation, not only between jobs. The first tick fires immediately. The task lives for the
/// worker's lifetime (aborted when the process exits).
#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn spawn_gpu_telemetry(config_dir: PathBuf) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(2));
        loop {
            ticker.tick().await;
            write_gpu_telemetry(&config_dir);
        }
    });
}

#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn apply_gpu_memory_limit(_bytes: u64) {}

#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn sync_gpu_memory_limit(_config_dir: &Path) {}

#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn spawn_gpu_telemetry(_config_dir: PathBuf) {}

/// Best-effort human-readable text from a caught panic payload — the `&str`/`String` a `panic!`
/// produces. mlx-rs `.unwrap()`/`.expect()` panics carry their formatted message as a `String`
/// (e.g. the `[metal::malloc] Attempting to allocate …` Metal OOM), so this surfaces the real cause.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

pub(crate) async fn with_cached_generator<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let key = GeneratorCacheKey::from_load_spec(engine_id, &spec);
    let load_error_context = load_error_context.into();
    let (reply_tx, reply_rx) = oneshot::channel::<WorkerResult<R>>();
    let job = Box::new(move |cache: &mut GeneratorCache| {
        // Contain a panic from inside the engine (e.g. mlx-rs `.unwrap()`-ing a Metal allocation
        // failure) so it fails THIS job with a clean error instead of unwinding out of the shared
        // cache thread and stopping every subsequent generation (sc-6067). The cached generator is
        // evicted on panic — post-abort MLX/Metal state is suspect, so the next job reloads fresh.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.with_generator(key, spec, load_error_context, run)
        })) {
            Ok(result) => result,
            Err(panic) => {
                if cache.evict().is_some() {
                    release_backend_cache_after_evict();
                }
                Err(WorkerError::Engine(format!(
                    "MLX generation panicked and was contained (the engine likely ran out of \
                     memory; the cached generator was reset): {}",
                    panic_message(panic.as_ref())
                )))
            }
        };
        let _ = reply_tx.send(result);
    });
    generator_worker()
        .send(job)
        .map_err(|_| WorkerError::Engine("MLX generator cache worker stopped".to_owned()))?;
    reply_rx.await.map_err(|_| {
        WorkerError::Engine("MLX generator cache worker dropped the job result".to_owned())
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_includes_adapter_fingerprint() {
        let base = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        let mut with_adapter = base.clone();
        with_adapter.adapters = vec![AdapterSpec::new(
            PathBuf::from("/loras/style.safetensors"),
            0.8,
            AdapterKind::Lora,
        )];
        let mut different_scale = with_adapter.clone();
        different_scale.adapters[0].scale = 0.9;

        assert_ne!(
            GeneratorCacheKey::from_load_spec("z_image_turbo", &base),
            GeneratorCacheKey::from_load_spec("z_image_turbo", &with_adapter)
        );
        assert_ne!(
            GeneratorCacheKey::from_load_spec("z_image_turbo", &with_adapter),
            GeneratorCacheKey::from_load_spec("z_image_turbo", &different_scale)
        );
    }

    // sc-8841 (F-039): the fingerprint helper is the core of the fix — it must report a DIFFERENT
    // value when a file at the same path changes (size or mtime), and `Unavailable` (a distinct,
    // cache-missing value) when the path can't be stat'd.
    #[test]
    fn fingerprint_tracks_content_change_and_missing_files() {
        use std::io::Write;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("weights.safetensors");
        std::fs::write(&path, b"original").expect("write original");
        let original = Fingerprint::of(&path);
        assert!(
            matches!(original, Fingerprint::Present { .. }),
            "an existing file must fingerprint as Present, got {original:?}"
        );
        // Re-stat with no change: same fingerprint → the common case still hits the cache.
        assert_eq!(
            original,
            Fingerprint::of(&path),
            "an unchanged file must produce a stable fingerprint (no spurious cache miss)"
        );

        // Grow the file (size changes) — must differ even if the clock granularity hides the mtime.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("open for append");
            f.write_all(b"-more-bytes").expect("append");
        }
        assert_ne!(
            original,
            Fingerprint::of(&path),
            "a size change at the same path must change the fingerprint"
        );

        // mtime sensitivity, proven as a pure value comparison so it does not depend on filesystem
        // timestamp granularity or a coarse system clock: two same-size fingerprints whose mtime
        // differs must NOT compare equal (a same-size overwrite — e.g. a re-convert that lands an
        // identically-sized file — still busts the cache via the mtime).
        let now = SystemTime::now();
        let earlier = Fingerprint::Present {
            size: 4096,
            mtime: Some(now),
        };
        let later = Fingerprint::Present {
            size: 4096,
            mtime: Some(now + Duration::from_secs(120)),
        };
        assert_ne!(
            earlier, later,
            "a bumped mtime at the same size must change the fingerprint"
        );

        // Missing path → Unavailable, distinct from any Present value so a stat error rebuilds
        // rather than serving a stale entry.
        let missing = Fingerprint::of(&dir.path().join("does-not-exist"));
        assert_eq!(missing, Fingerprint::Unavailable);
        assert_ne!(missing, original);
        assert_ne!(missing, earlier);
    }

    // sc-8841 (F-039): the whole-key oracle. A LoRA re-imported at the SAME path (new bytes, same
    // name) must yield a DIFFERENT cache key so the resident generator reloads instead of silently
    // reusing the stale adapter within the 300 s idle window. An unchanged file must yield the SAME
    // key so the common case keeps hitting the cache (no perf regression from spurious misses).
    #[test]
    fn cache_key_changes_when_adapter_file_is_replaced_at_same_path() {
        use std::io::Write;

        let base_dir = tempfile::tempdir().expect("base tempdir");
        let lora_dir = tempfile::tempdir().expect("lora tempdir");
        let lora_path = lora_dir.path().join("style.safetensors");
        std::fs::write(&lora_path, b"v1-tensors").expect("write lora v1");

        let make_spec = || {
            let mut spec = LoadSpec::new(WeightsSource::Dir(base_dir.path().to_path_buf()));
            spec.adapters = vec![AdapterSpec::new(lora_path.clone(), 0.8, AdapterKind::Lora)];
            spec
        };

        let key_v1 = GeneratorCacheKey::from_load_spec("z_image_turbo", &make_spec());
        // Same file, no change → identical key → cache still hits.
        assert_eq!(
            key_v1,
            GeneratorCacheKey::from_load_spec("z_image_turbo", &make_spec()),
            "an unchanged adapter file must produce an identical cache key (cache hit preserved)"
        );

        // Re-import the LoRA at the same path with new, DIFFERENTLY-SIZED bytes (a re-import writes
        // a fresh file). The size delta alone busts the key regardless of clock granularity; the
        // mtime path is covered as a pure value comparison in `fingerprint_tracks_content_change_*`.
        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&lora_path)
                .expect("reopen lora");
            f.write_all(b"v2-completely-different-tensors-and-longer")
                .expect("write lora v2");
        }

        let key_v2 = GeneratorCacheKey::from_load_spec("z_image_turbo", &make_spec());
        assert_ne!(
            key_v1, key_v2,
            "re-importing a LoRA at the same path must change the cache key so the stale adapter \
             is not served from cache"
        );
    }

    /// sc-9092 (epic 9083): the candle-lane A/B quant toggle must MISS the generator cache so the new
    /// tier is loaded rather than the resident one reused. On the candle lane (now routed through the
    /// shared `standard_tier_subdir`, sc-9092) toggling `advanced.mlxQuantize` changes BOTH the resolved
    /// tier subdir (`q4/` ↔ `q8/` ↔ `bf16/`) AND the load `quantize` — either alone flips the key, so a
    /// toggle can never collide with the cached generator (reload-always on toggle, epic 8506). This is
    /// the candle sibling of the MLX A/B behaviour: `GeneratorCacheKey` already keys on both fields.
    #[test]
    fn cache_key_includes_quant_tier_toggle() {
        // q4 tier: `<root>/q4` weights + Q4 load quant.
        let mut q4 = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/lens/q4")));
        q4.quantize = Some(Quant::Q4);
        // q8 tier: `<root>/q8` weights + Q8 load quant (the A/B toggle target).
        let mut q8 = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/lens/q8")));
        q8.quantize = Some(Quant::Q8);
        // bf16 tier: `<root>/bf16` weights, dense (no quant).
        let bf16 = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/lens/bf16")));

        // Every pairwise toggle is a distinct cache entry → a miss → a reload, never a wrong-tier reuse.
        assert_ne!(
            GeneratorCacheKey::from_load_spec("lens", &q4),
            GeneratorCacheKey::from_load_spec("lens", &q8)
        );
        assert_ne!(
            GeneratorCacheKey::from_load_spec("lens", &q8),
            GeneratorCacheKey::from_load_spec("lens", &bf16)
        );
        assert_ne!(
            GeneratorCacheKey::from_load_spec("lens", &q4),
            GeneratorCacheKey::from_load_spec("lens", &bf16)
        );
        // The `quantize` field alone flips the key even if the tier dir were identical — the candle lane
        // has always keyed on it (generator_cache.rs), so the A/B toggle is safe regardless of layout.
        let mut same_dir_q8 = q4.clone();
        same_dir_q8.quantize = Some(Quant::Q8);
        assert_ne!(
            GeneratorCacheKey::from_load_spec("lens", &q4),
            GeneratorCacheKey::from_load_spec("lens", &same_dir_q8)
        );
    }

    #[test]
    fn cache_key_includes_control_and_ip_components() {
        let mut control = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        control.control = Some(WeightsSource::File(PathBuf::from(
            "/controls/pose.safetensors",
        )));
        let mut ip = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/base")));
        ip.ip_adapter = Some(WeightsSource::Dir(PathBuf::from("/ip-adapter")));

        assert_ne!(
            GeneratorCacheKey::from_load_spec("sdxl", &control),
            GeneratorCacheKey::from_load_spec("sdxl", &ip)
        );
    }

    // -------------------------------------------------------------------------
    // Backend-neutral acceptance seam (epic 3720, sc-3724). A pure-`gen_core`
    // `Generator` registered into the same `inventory` registry the real provider crates use
    // (with a UNIQUE id so it never collides with a real engine or the engines.rs derivation
    // stubs). It links NO tensor backend, so these tests run on Linux/Windows AND macOS, proving
    // the load→progress→cancel→output contract that `with_cached_generator` is the production seam
    // for. Mirrors the inventory pattern at engines.rs.
    struct StubGenerator {
        descriptor: gen_core::ModelDescriptor,
    }

    impl Generator for StubGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn validate(&self, _req: &gen_core::GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            req: &gen_core::GenerationRequest,
            on_progress: &mut dyn FnMut(gen_core::Progress),
        ) -> gen_core::Result<gen_core::GenerationOutput> {
            on_progress(gen_core::Progress::Step {
                current: 1,
                total: 2,
            });
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            on_progress(gen_core::Progress::Step {
                current: 2,
                total: 2,
            });
            Ok(gen_core::GenerationOutput::Images(vec![gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0u8; 12],
            }]))
        }
    }

    fn stub_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "sc3724_stub",
            family: "test",
            backend: "stub",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }

    fn stub_load(_spec: &gen_core::LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
        Ok(Box::new(StubGenerator {
            descriptor: stub_descriptor(),
        }))
    }

    inventory::submit! {
        gen_core::registry::ModelRegistration { descriptor: stub_descriptor, load: stub_load }
    }

    fn stub_cache_key() -> GeneratorCacheKey {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/stub")));
        GeneratorCacheKey::from_load_spec("sc3724_stub", &spec)
    }

    fn stub_cache_entry() -> GeneratorCacheEntry {
        GeneratorCacheEntry {
            key: stub_cache_key(),
            generator: Box::new(StubGenerator {
                descriptor: stub_descriptor(),
            }),
        }
    }

    #[test]
    fn generator_cache_idle_timeout_defaults_parses_and_disables() {
        assert_eq!(
            generator_cache_idle_timeout(None),
            Some(Duration::from_secs(DEFAULT_GENERATOR_CACHE_IDLE_SECONDS))
        );
        assert_eq!(
            generator_cache_idle_timeout(Some("")),
            Some(Duration::from_secs(DEFAULT_GENERATOR_CACHE_IDLE_SECONDS))
        );
        assert_eq!(
            generator_cache_idle_timeout(Some("not-a-number")),
            Some(Duration::from_secs(DEFAULT_GENERATOR_CACHE_IDLE_SECONDS))
        );
        assert_eq!(generator_cache_idle_timeout(Some("0")), None);
        assert_eq!(
            generator_cache_idle_timeout(Some("42")),
            Some(Duration::from_secs(42))
        );
    }

    #[test]
    fn cache_worker_evicts_resident_generator_after_idle_timeout() {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let worker = thread::spawn(move || {
            run_generator_cache_worker(rx, Some(Duration::from_millis(20)));
        });
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            cache.entry = Some(stub_cache_entry());
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        // Poll for eviction instead of asserting after a single fixed sleep. The worker only evicts
        // when its `recv_timeout(idle)` actually TIMES OUT; under CI load the worker thread can be
        // starved past a fixed wait, then wake to find the check job already queued and return it as
        // `Ok` — resetting the idle window without ever evicting (the old flake). Each poll sleeps
        // longer than the 20ms idle window so the worker gets a fresh timeout between checks, and the
        // generous iteration budget tolerates a slow runner. Still verifies the same thing: idle
        // timeout evicts the resident generator.
        let mut evicted = false;
        for _ in 0..100 {
            thread::sleep(Duration::from_millis(50));
            let (reply_tx, reply_rx) = mpsc::channel();
            tx.send(Box::new(move |cache: &mut GeneratorCache| {
                reply_tx
                    .send(cache.entry.is_none())
                    .expect("send cache state");
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
            "expected idle timeout to evict the resident generator"
        );
        drop(tx);
        worker.join().expect("cache worker exits");
    }

    #[test]
    fn cache_worker_keeps_resident_generator_when_idle_eviction_disabled() {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let worker = thread::spawn(move || {
            run_generator_cache_worker(rx, None);
        });
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            cache.entry = Some(stub_cache_entry());
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        thread::sleep(Duration::from_millis(80));

        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            reply_tx
                .send(cache.entry.is_some())
                .expect("send cache state");
        }))
        .expect("check cache state");

        assert!(
            reply_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("cache state reply"),
            "expected disabled idle timeout to keep the resident generator"
        );
        drop(tx);
        worker.join().expect("cache worker exits");
    }

    // load → progress → asset: drive the production cache seam end to end with a backend-neutral
    // generator. Collect progress, take the produced image, write a PNG, and build a minimal
    // asset-fact JSON — the same shape (load → generate → persist) the macOS image path follows.
    #[tokio::test]
    async fn cached_generator_loads_progresses_and_writes_asset() {
        let weights = tempfile::tempdir().expect("weights tempdir");
        let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let assets = tempfile::tempdir().expect("asset tempdir");
        let png_path = assets.path().join("stub.png");
        let png_path_for_run = png_path.clone();

        let fact = with_cached_generator("sc3724_stub", spec, "stub load", move |generator| {
            let req = gen_core::GenerationRequest {
                width: 2,
                height: 2,
                ..Default::default()
            };
            let mut steps: Vec<gen_core::Progress> = Vec::new();
            let output = generator
                .generate(&req, &mut |progress| steps.push(progress))
                .map_err(|error| WorkerError::Engine(error.to_string()))?;
            let image = match output {
                gen_core::GenerationOutput::Images(mut images) => images.remove(0),
                other => {
                    return Err(WorkerError::Engine(format!(
                        "expected images, got {other:?}"
                    )))
                }
            };
            let buffer = image::RgbImage::from_raw(image.width, image.height, image.pixels)
                .ok_or_else(|| WorkerError::Engine("stub image buffer size mismatch".to_owned()))?;
            buffer
                .save(&png_path_for_run)
                .map_err(|error| WorkerError::Engine(error.to_string()))?;
            let step_count = steps
                .iter()
                .filter(|p| matches!(p, gen_core::Progress::Step { .. }))
                .count();
            Ok(serde_json::json!({
                "assetId": uuid::Uuid::new_v4().to_string(),
                "path": png_path_for_run.display().to_string(),
                "steps": step_count,
            }))
        })
        .await
        .expect("stub generate succeeds");

        assert!(
            fact.get("steps")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                >= 1,
            "expected at least one Progress::Step"
        );
        assert!(png_path.exists(), "expected the PNG asset to be written");
        assert!(
            fact.get("assetId")
                .and_then(serde_json::Value::as_str)
                .is_some(),
            "expected the asset fact to carry an asset id"
        );
    }

    // cancel honored: a pre-tripped CancelFlag makes the generator return `Error::Canceled`, which
    // the seam maps to `WorkerError::Canceled` (the typed cancellation the worker distinguishes
    // from generic failure).
    #[tokio::test]
    async fn cached_generator_honors_cancel() {
        let weights = tempfile::tempdir().expect("weights tempdir");
        let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));

        let result = with_cached_generator("sc3724_stub", spec, "stub load", move |generator| {
            let cancel = gen_core::runtime::CancelFlag::new();
            cancel.cancel();
            let req = gen_core::GenerationRequest {
                width: 2,
                height: 2,
                cancel,
                ..Default::default()
            };
            generator
                .generate(&req, &mut |_progress| {})
                .map(|_| ())
                .map_err(|error| match error {
                    gen_core::Error::Canceled => WorkerError::Canceled(error.to_string()),
                    other => WorkerError::Engine(other.to_string()),
                })
        })
        .await;

        assert!(
            matches!(result, Err(WorkerError::Canceled(_))),
            "expected the cancel flag to map to WorkerError::Canceled, got {result:?}"
        );
    }

    // sc-6067: a panic inside a job closure (e.g. mlx-rs `.unwrap()`-ing a Metal OOM) must be
    // CONTAINED — it fails only that job with a clean error AND the single shared cache thread keeps
    // serving. Without the `catch_unwind` guard the worker thread unwinds and dies, and every later
    // generation fails with "MLX generator cache worker stopped" until a process restart. (The panic
    // backtrace this test triggers is printed by the default panic hook — that is expected.)
    #[tokio::test]
    async fn panicking_job_is_contained_and_worker_keeps_serving() {
        let weights = tempfile::tempdir().expect("weights tempdir");
        let spec = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));

        // A run closure that panics mid-generation → comes back as a clean Engine error, not a hang.
        let panicked = with_cached_generator(
            "sc3724_stub",
            spec.clone(),
            "stub load",
            move |_generator| -> WorkerResult<()> {
                panic!("simulated mlx-rs Metal allocation panic");
            },
        )
        .await;
        let Err(WorkerError::Engine(msg)) = &panicked else {
            panic!("a job-closure panic must map to a clean Engine error, got {panicked:?}");
        };
        assert!(
            msg.contains("was contained"),
            "contained-panic message: {msg}"
        );
        assert!(
            msg.contains("simulated mlx-rs Metal allocation panic"),
            "the original panic text must surface for diagnostics: {msg}"
        );

        // The shared cache thread must still be alive and serving: a subsequent job succeeds.
        let after = with_cached_generator("sc3724_stub", spec, "stub load", move |generator| {
            let req = gen_core::GenerationRequest {
                width: 2,
                height: 2,
                ..Default::default()
            };
            generator
                .generate(&req, &mut |_progress| {})
                .map(|_| ())
                .map_err(|error| WorkerError::Engine(error.to_string()))
        })
        .await;
        assert!(
            after.is_ok(),
            "worker must keep serving after a contained panic, got {after:?}"
        );
    }
}
