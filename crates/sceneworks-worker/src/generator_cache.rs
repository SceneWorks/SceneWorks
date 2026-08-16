use std::path::{Path, PathBuf};
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::Duration;

use gen_core::{
    AdapterKind, AdapterSpec, Generator, LoadShape, LoadSpec, MemoryCacheState, MoeExpert,
    OffloadPolicy, Precision, Quant, WeightsSource,
};

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
use crate::cache_thread::CacheThread;
use crate::cache_thread::{self, CacheAccess, CacheJob, Fingerprint, SeamMessages};
use crate::WorkerResult;

/// The generator cache is a single-resident [`crate::cache_thread::CacheThread`] keyed by
/// [`GeneratorCacheKey`], holding a loaded `Box<dyn Generator>`. The generic scaffolding (dedicated
/// worker thread, idle-timeout eviction, panic containment, `Fingerprint`, oneshot-reply seam) lives
/// in [`crate::cache_thread`]; this module supplies only the key derivation, the loader, and the
/// message strings (sc-11191, F-019).
struct CachedGenerator {
    generator: Box<dyn Generator>,
    load_policy: OffloadPolicy,
    /// Process-global MLX active bytes that predated this cached generator. Request admission must
    /// never mistake these unrelated allocations for already-resident generator weights.
    external_committed_bytes: u64,
}

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
type GeneratorCache = CacheThread<GeneratorCacheKey, CachedGenerator>;
type GeneratorJob = CacheJob<GeneratorCacheKey, CachedGenerator>;

const GENERATOR_CACHE_IDLE_SECONDS_ENV: &str = "SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS";
const DEFAULT_GENERATOR_CACHE_IDLE_SECONDS: u64 = 300;

/// The generator cache does NOT free the backend cache before a cold load (unlike the refine cache,
/// which sets this `true` to bound peak memory to one ~16 GB model). A cold miss here clears the
/// resident generator and loads, sizing the load via the fit-gate/residency policy in the loader
/// closure rather than a pre-load backend trim. This divergence is deliberate and documented — see
/// the [`crate::cache_thread`] module docs; do not silently unify it away.
const GENERATOR_EVICT_BEFORE_LOAD: bool = false;

#[cfg(target_os = "macos")]
fn capture_external_committed_bytes() -> u64 {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::get_active_memory() as u64
}

#[cfg(not(target_os = "macos"))]
fn capture_external_committed_bytes() -> u64 {
    0
}

static GENERATOR_WORKER: OnceLock<mpsc::Sender<GeneratorJob>> = OnceLock::new();

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
    // Phase release and block materialization are independent load-time identities (SC-15998).
    // Reusing a generator across either boundary would execute a different physical load shape than
    // the caller requested.
    offload_policy: OffloadPolicy,
    load_shape: LoadShape,
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
            offload_policy: spec.offload_policy,
            load_shape: spec.load_shape,
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

/// Thin wrapper over the generic [`cache_thread::run_cache_worker`]: no evict-before-load
/// ([`GENERATOR_EVICT_BEFORE_LOAD`]) and the generator-specific idle-eviction log.
fn run_generator_cache_worker(rx: mpsc::Receiver<GeneratorJob>, idle_timeout: Option<Duration>) {
    cache_thread::run_cache_worker(
        rx,
        idle_timeout,
        GENERATOR_EVICT_BEFORE_LOAD,
        |key: &GeneratorCacheKey, idle_seconds| {
            // Documented event (docs/observability.md): expected idle-timeout eviction, so info
            // level with the engine + idle window.
            tracing::info!(
                event = "generator_cache_idle_evicted",
                engine = %key.engine_id,
                idleSeconds = idle_seconds,
            );
        },
    );
}

fn generator_cache_idle_timeout_from_env() -> Option<Duration> {
    generator_cache_idle_timeout(
        std::env::var(GENERATOR_CACHE_IDLE_SECONDS_ENV)
            .ok()
            .as_deref(),
    )
}

fn generator_cache_idle_timeout(raw: Option<&str>) -> Option<Duration> {
    cache_thread::idle_timeout_from_secs(raw, DEFAULT_GENERATOR_CACHE_IDLE_SECONDS)
}

/// Apply the GPU memory ceiling to the MLX runtime (epic 7819, sc-7820).
///
/// `bytes == 0` means the user configured no ceiling; a **derived default** is applied instead — see
/// [`resolve_gpu_memory_limit`] (GitHub #1932). When non-zero we set two MLX knobs:
/// - `set_memory_limit` — soft backpressure: when active memory exceeds the limit MLX blocks and
///   waits for in-flight GPU work to drain rather than hard-failing. It is a target, not a hard
///   sandbox; a single oversized allocation can still exceed it (and on a too-low cap a model whose
///   working set genuinely needs more will thrash/swap or hit a Metal OOM — already contained by the
///   `catch_unwind` guard above).
/// - `set_wired_limit` — caps pinned (non-pageable) residency so the OS can reclaim the rest of
///   unified memory for other apps. macOS 15+. **Clamped to the device wired ceiling** — MLX throws
///   if asked for more than the device `recommendedMaxWorkingSetSize`, and its default error handler
///   answers that throw with `exit(-1)`, killing the worker at startup (sc-12178, GitHub #1544: an
///   8 GB Mac's ceiling is ~5.3 GB, so a 6–7 GB user cap crashed the worker). See
///   [`clamp_wired_limit`].
///
/// We deliberately leave `set_cache_limit` at its default: forcing it low causes reallocation storms
/// between steps (the fork's own doc warns about this).
///
/// The MLX limit is **process-global**, so calling this once at worker startup (before any model
/// load) covers generations, upscales, AND LoRA training — even though training takes a separate
/// path from the generator cache.
/// The GPU memory ceiling (bytes) currently *requested* of this process's MLX runtime, so the live
/// sync (sc-7824) only re-applies on an actual change. `u64::MAX` is the "nothing applied yet"
/// sentinel — distinct from `0` ("no user ceiling"), so the first real value (including a deliberate
/// `0` that drops back to the derived default) always takes effect.
///
/// This holds the REQUESTED value, not the resolved one, precisely so the dedupe stays stable: a
/// user with no ceiling writes `0` to the handoff file forever, and comparing that against a
/// *resolved* default would re-apply on every poll. The resolved figure lives in
/// [`EFFECTIVE_GPU_MEMORY_LIMIT`].
#[cfg(all(target_os = "macos", not(test)))]
static APPLIED_GPU_MEMORY_LIMIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// The ceiling (bytes) actually handed to `set_memory_limit`, for the Settings telemetry readout.
/// `0` = nothing applied (no user ceiling AND no memory probe signal, so MLX keeps its own default).
#[cfg(all(target_os = "macos", not(test)))]
static EFFECTIVE_GPU_MEMORY_LIMIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Clamp a requested wired-residency cap (bytes) to the device's wired ceiling (sc-12178).
///
/// MLX's `set_wired_limit` THROWS when asked for more than the device `recommendedMaxWorkingSetSize`,
/// and MLX's *default* error handler answers that throw with `exit(-1)` — an uncatchable libc exit
/// (not a Rust panic the worker's `catch_unwind` guard could contain) that hard-kills the worker at
/// startup, before it ever claims a job. That is the GitHub #1544 crash: on an 8 GB Mac the ceiling
/// is ~5.3 GB, so a 6–7 GB user cap (the natural "leave RAM for the OS" choice) killed the worker.
///
/// A cap at or below the ceiling never throws — and a cap ABOVE it is meaningless anyway, since the
/// device already bounds wired residency there. `device_ceiling == 0` (ceiling unknown) yields `0`,
/// which MLX reads as "no wired cap" (its default): the safe fall-back.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn clamp_wired_limit(requested: usize, device_ceiling: usize) -> usize {
    if device_ceiling == 0 {
        return 0;
    }
    requested.min(device_ceiling)
}

/// Resolve the memory limit (bytes) to hand MLX, given the user's configured ceiling (`0` = none)
/// and the machine's total unified memory. `0` out means "apply nothing, leave MLX on its default".
///
/// ## Why an unset ceiling can NOT stay on the MLX default (GitHub #1932)
/// MLX's default budget is **1.5× the device recommended working set**, and that working set is
/// itself ~2/3 of unified memory on Apple Silicon — see [`device_wired_ceiling_bytes`], which
/// recovers the ceiling by dividing the default limit by 1.5. Composing the two: MLX's default
/// budget is ≈ **100% of physical RAM**. On a large Mac the headroom above a model's real footprint
/// hides that; on an 8 GB Mac it means MLX will keep climbing until the machine has nothing left for
/// macOS itself, and a unified-memory exhaustion with GPU allocations outstanding takes the whole
/// system down (the #1932 report: an 8 GB iMac M3 hard-restarting within ~10 s of pressing Generate,
/// far too fast to be the thermal event it looks like). MLX's limit is soft backpressure — on
/// reaching it MLX frees its cache and waits for in-flight GPU work to drain instead of climbing —
/// so a limit set *below* physical RAM is exactly the mechanism that keeps the pressure recoverable.
///
/// The derived default is `total − legacy_unified_reserve(total)`, reusing the fit gate's typed
/// fallback policy rather than a second, independent number. That agreement
/// is load-bearing in both directions: the fit gate's weights-fit floor (sc-12179, GitHub #1544)
/// admits a model whose weights fit the same legacy ceiling, so a runtime ceiling derived the same
/// way can never refuse to hold a tier the gate just admitted. On an 8 GB Mac that is a 6 GiB
/// ceiling, which still clears the 5.49 GiB z-image-turbo q4 baseline the floor is anchored on.
///
/// A configured ceiling always wins unchanged — this only fills in the unset case. No probe signal
/// ⇒ `0` ⇒ MLX keeps its default, the same fail-open the fit gate takes when it cannot size the
/// machine. `total <= reserve` likewise yields `0`: a machine smaller than the OS reserve has no
/// sensible ceiling to derive, and a nonsense-small limit would thrash rather than protect.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn resolve_gpu_memory_limit(requested: u64, total_unified_bytes: Option<u64>) -> u64 {
    if requested > 0 {
        return requested;
    }
    let Some(total) = total_unified_bytes else {
        return 0;
    };
    let total_gb = total as f64 / crate::fit_gate::BYTES_PER_GIB;
    let reserve = (crate::fit_gate::legacy_unified_reserve(total_gb).gb
        * crate::fit_gate::BYTES_PER_GIB) as u64;
    total.saturating_sub(reserve)
}

/// The device's wired-residency ceiling in bytes (`recommendedMaxWorkingSetSize`), derived once.
///
/// MLX documents its default memory limit as 1.5× the device recommended working set
/// (`get_memory_limit`), so reading that default and dividing by 1.5 recovers the true hardware
/// ceiling with no new binding or Metal query. The read MUST happen before the first
/// `set_memory_limit` (after which `get_memory_limit` returns our value, not the default); the
/// `OnceLock` both caches the constant hardware property and pins the read to the first application,
/// while the limit is still MLX's untouched default. `/ 3 * 2` (rather than `* 2 / 3`) makes any
/// integer rounding go DOWNward, staying at or below the ceiling — which never throws.
#[cfg(all(target_os = "macos", not(test)))]
fn device_wired_ceiling_bytes() -> usize {
    static CEILING: OnceLock<usize> = OnceLock::new();
    *CEILING.get_or_init(|| mlx_rs::memory::get_memory_limit() / 3 * 2)
}

#[cfg(all(target_os = "macos", not(test)))]
fn set_gpu_memory_limit_inner(requested: u64) {
    use std::sync::atomic::Ordering;
    // Capture the device wired ceiling BEFORE mutating the memory limit — the derivation reads MLX's
    // default `get_memory_limit`, which is only the hardware default until the first `set_memory_limit`.
    let wired_ceiling = device_wired_ceiling_bytes();
    let effective = resolve_gpu_memory_limit(
        requested,
        crate::mlx_fit_gate::probe_total_unified_memory_bytes(),
    );
    APPLIED_GPU_MEMORY_LIMIT.store(requested, Ordering::SeqCst);
    if effective == 0 {
        // No user ceiling and no probe signal: leave MLX on its own default budget, byte-identical
        // to the pre-#1932 behavior. Nothing to report to the telemetry readout either.
        EFFECTIVE_GPU_MEMORY_LIMIT.store(0, Ordering::SeqCst);
        return;
    }
    let limit = effective as usize;
    let previous_limit = mlx_rs::memory::set_memory_limit(limit);
    // The wired cap is only touched for a ceiling the USER configured. `set_wired_limit` raises the
    // amount MLX PINS (macOS cannot reclaim pinned pages), so applying one off the derived default
    // would push the small Mac in #1932 the wrong way — the point there is to let the OS reclaim.
    // Clamped so `set_wired_limit` can never throw and `exit(-1)` the worker (sc-12178, GitHub #1544).
    // `0` in both log fields means "left untouched", which is exactly what MLX reads `0` as too.
    let (wired_limit, previous_wired) = if requested > 0 {
        let wired_limit = clamp_wired_limit(limit, wired_ceiling);
        (wired_limit, mlx_rs::memory::set_wired_limit(wired_limit))
    } else {
        (0, 0)
    };
    EFFECTIVE_GPU_MEMORY_LIMIT.store(effective, Ordering::SeqCst);
    tracing::info!(
        event = "gpu_memory_limit_applied",
        requestedBytes = requested,
        limitBytes = limit,
        wiredLimitBytes = wired_limit,
        deviceWiredCeilingBytes = wired_ceiling,
        previousLimitBytes = previous_limit,
        previousWiredLimitBytes = previous_wired,
        derivedDefault = requested == 0,
        "applied GPU memory ceiling to the MLX runtime"
    );
}

#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn apply_gpu_memory_limit(bytes: u64) {
    set_gpu_memory_limit_inner(bytes);
}

/// Restores the process-global MLX soft limit after one exact evidence-covered request.
///
/// Jobs are serialized on the generator thread, so this guard cannot race another generation. It
/// intentionally never calls `set_wired_limit`: #1947 established that a derived path must not raise
/// pinned residency.
#[cfg(all(target_os = "macos", not(test)))]
pub(crate) struct RequestGpuMemoryLimitGuard {
    previous: usize,
}

#[cfg(all(target_os = "macos", not(test)))]
impl Drop for RequestGpuMemoryLimitGuard {
    fn drop(&mut self) {
        mlx_rs::memory::set_memory_limit(self.previous);
    }
}

#[cfg(all(target_os = "macos", test))]
pub(crate) struct RequestGpuMemoryLimitGuard;

#[cfg(all(target_os = "macos", not(test)))]
pub(crate) fn apply_request_gpu_memory_limit(
    evidence_ceiling_bytes: u64,
) -> Option<RequestGpuMemoryLimitGuard> {
    if evidence_ceiling_bytes == 0 {
        return None;
    }
    let previous = mlx_rs::memory::get_memory_limit();
    let evidence_ceiling = usize::try_from(evidence_ceiling_bytes).unwrap_or(usize::MAX);
    let applied = previous.min(evidence_ceiling);
    mlx_rs::memory::set_memory_limit(applied);
    tracing::info!(
        event = "mlx_request_memory_limit_applied",
        evidenceCeilingBytes = evidence_ceiling,
        appliedBytes = applied,
        previousBytes = previous,
        wiredLimitChanged = false,
        "applied request-scoped MLX soft limit from verified memory evidence"
    );
    Some(RequestGpuMemoryLimitGuard { previous })
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn apply_request_gpu_memory_limit(
    evidence_ceiling_bytes: u64,
) -> Option<RequestGpuMemoryLimitGuard> {
    (evidence_ceiling_bytes > 0).then_some(RequestGpuMemoryLimitGuard)
}

/// Re-read the live GPU-memory-limit handoff file and apply it if it changed since the last applied
/// value (epic 7819, sc-7824). Called from the worker poll loop *between jobs*, so moving the
/// Settings slider takes effect on the next job without a worker restart. An absent file is a
/// no-op (the spawn-time `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES` value stays in force); a written `0`
/// actively drops a previously-applied cap back to the derived default ([`resolve_gpu_memory_limit`]).
/// The dedupe compares REQUESTED values, so a user with no ceiling (a permanent `0` in the file)
/// re-applies once and then stays quiet.
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
    // The EFFECTIVE ceiling, so "peak vs limit" reads honestly on a machine running the derived
    // default (GitHub #1932) — that is a cap this worker really applied, not MLX's internal budget.
    // Still `0` when nothing was applied at all.
    let telemetry = sceneworks_core::app_paths::GpuMemoryTelemetry {
        active_bytes: mlx_rs::memory::get_active_memory() as u64,
        peak_bytes: mlx_rs::memory::get_peak_memory() as u64,
        cache_bytes: mlx_rs::memory::get_cache_memory() as u64,
        limit_bytes: EFFECTIVE_GPU_MEMORY_LIMIT.load(Ordering::SeqCst),
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

/// User-facing message strings for the generator cache seam, preserving the exact wording the worker
/// emitted before the `cache_thread` extraction (sc-11191, F-019).
const GENERATOR_SEAM_MESSAGES: SeamMessages = SeamMessages {
    entry_missing: "Generator cache entry missing after load.",
    panic_reset: "MLX generation panicked and was contained (the engine likely ran out of memory; \
                  the cached generator was reset)",
    worker_stopped: "MLX generator cache worker stopped",
    worker_dropped: "MLX generator cache worker dropped the job result",
};

/// One fallible admission check for a true generator-cache cold miss. It is consumed on the cache
/// worker only after a different resident has been dropped, and is never touched for an exact warm
/// [`GeneratorCacheKey`] hit. That key already includes weight fingerprints, quant/precision,
/// adapters, controls, offload policy, and load shape, so every field that changes resident tensors
/// forces this cold path instead of borrowing another layout's admission.
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
pub(crate) struct GeneratorColdLoadAdmission {
    gate: Box<dyn FnOnce() -> WorkerResult<()> + Send + 'static>,
}

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
impl GeneratorColdLoadAdmission {
    pub(crate) fn new(gate: impl FnOnce() -> WorkerResult<()> + Send + 'static) -> Self {
        Self {
            gate: Box::new(gate),
        }
    }

    fn admit(self) -> WorkerResult<()> {
        (self.gate)()
    }
}

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
struct GeneratorColdLoadTransaction {
    request_cancel: gen_core::CancelFlag,
    admission: GeneratorColdLoadAdmission,
}

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
impl GeneratorColdLoadTransaction {
    fn new(request_cancel: gen_core::CancelFlag, admission: GeneratorColdLoadAdmission) -> Self {
        Self {
            request_cancel,
            admission,
        }
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
    with_cached_generator_for_request_using(
        engine_id,
        spec,
        load_error_context,
        crate::inference_runtime::load,
        move |generator, _cache_state, _load_policy, _external_committed_bytes| run(generator),
    )
    .await
}

/// Run one request against a cached generator while exposing the independent request-policy inputs
/// that do not belong in [`GeneratorCacheKey`].
pub(crate) async fn with_cached_generator_for_request<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    run: impl FnOnce(&dyn Generator, MemoryCacheState, OffloadPolicy, u64) -> WorkerResult<R>
        + Send
        + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    with_cached_generator_for_request_using(
        engine_id,
        spec,
        load_error_context,
        crate::inference_runtime::load,
        run,
    )
    .await
}

/// [`with_cached_generator`] with the loader supplied by the caller — the seam a test injects a
/// backend-neutral stub `Generator` through (sc-3724), so the load→progress→cancel→output contract can
/// be driven with no tensor backend linked.
///
/// `pub(crate)` for sc-12318: `video_jobs`' `generate_video_using` threads its own loader down to here,
/// which is what makes the async per-family generation arms (`generate_mochi`,
/// `generate_candle_video`) reachable from a unit test. Their pre-load decisions — the frame lattice
/// and the Mochi fit gate — are otherwise unpinned, since a test can assert the free functions an arm
/// calls but never that it calls them.
pub(crate) async fn with_cached_generator_using<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    with_cached_generator_for_request_using(
        engine_id,
        spec,
        load_error_context,
        load_generator,
        move |generator, _cache_state, _load_policy, _external_committed_bytes| run(generator),
    )
    .await
}

/// Candle SCAIL sibling of [`with_cached_generator_using`]: an exact warm key reuses the resident
/// without admission, while a miss evicts first and runs `cold_admission` immediately before the
/// loader. The lookup/evict/gate/load sequence is one cache-worker job, never an external
/// peek-then-act pair.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn with_cached_generator_using_cold_admission<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    request_cancel: gen_core::CancelFlag,
    cold_admission: GeneratorColdLoadAdmission,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    with_cached_generator_using_cold_admission_on(
        generator_worker().clone(),
        engine_id,
        spec,
        load_error_context,
        GeneratorColdLoadTransaction::new(request_cancel, cold_admission),
        load_generator,
        run,
    )
    .await
}

#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
async fn with_cached_generator_using_cold_admission_on<R>(
    worker: mpsc::Sender<GeneratorJob>,
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    cold_load: GeneratorColdLoadTransaction,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let GeneratorColdLoadTransaction {
        request_cancel,
        admission: cold_admission,
    } = cold_load;
    let key = GeneratorCacheKey::from_load_spec(engine_id, &spec);
    let load_error_context = load_error_context.into();
    let load = move || {
        let spec = crate::mlx_fit_gate::apply_residency_policy(spec, engine_id)?;
        let load_policy = spec.offload_policy;
        let external_committed_bytes = capture_external_committed_bytes();
        let generator = load_generator(engine_id, &spec)
            .map_err(|error| crate::classify_engine_error(&load_error_context, error))?;
        Ok(CachedGenerator {
            generator,
            load_policy,
            external_committed_bytes,
        })
    };
    cache_thread::run_cached_with_access_after_cold_evict(
        worker,
        key,
        move || {
            if request_cancel.is_cancelled() {
                Err(crate::WorkerError::Canceled(
                    "Video generation canceled by user.".to_owned(),
                ))
            } else {
                Ok(())
            }
        },
        move || cold_admission.admit(),
        load,
        move |cached, _access| run(cached.generator.as_ref()),
        GENERATOR_SEAM_MESSAGES,
    )
    .await
}

pub(crate) async fn with_cached_generator_for_request_using<R>(
    engine_id: &'static str,
    spec: LoadSpec,
    load_error_context: impl Into<String>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    run: impl FnOnce(&dyn Generator, MemoryCacheState, OffloadPolicy, u64) -> WorkerResult<R>
        + Send
        + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let key = GeneratorCacheKey::from_load_spec(engine_id, &spec);
    let load_error_context = load_error_context.into();
    // The loader owns the generator-specific cold-load policy. Pre-load unified-memory fit-gate +
    // residency selection (epic 10834; sc-10835 Phase 0, sc-10839 Phase 1): BEFORE crate::inference_runtime::load
    // allocates, either reject a model that can't fit this machine's unified memory (a wired
    // overcommit SIGKILLs the worker mid-load rather than returning a catchable error) OR, for a
    // provider that supports sequential component residency, select `OffloadPolicy::Sequential` when
    // the resident sum won't fit but the staged max-single-component will. This runs only on a cold
    // miss (a warm cache hit never invokes the loader), so an already-resident model is never re-gated.
    let load = move || {
        let spec = crate::mlx_fit_gate::apply_residency_policy(spec, engine_id)?;
        let load_policy = spec.offload_policy;
        let external_committed_bytes = capture_external_committed_bytes();
        let generator = load_generator(engine_id, &spec)
            .map_err(|error| crate::classify_engine_error(&load_error_context, error))?;
        Ok(CachedGenerator {
            generator,
            load_policy,
            external_committed_bytes,
        })
    };
    let run = move |cached: &CachedGenerator, access| {
        let cache_state = match access {
            CacheAccess::Cold => MemoryCacheState::Cold,
            CacheAccess::Warm => MemoryCacheState::Warm,
        };
        run(
            cached.generator.as_ref(),
            cache_state,
            cached.load_policy,
            cached.external_committed_bytes,
        )
    };
    cache_thread::run_cached_with_access(
        generator_worker(),
        key,
        load,
        run,
        GENERATOR_SEAM_MESSAGES,
    )
    .await
}

/// Run `run` against a freshly-loaded, **uncached** generator on the shared cache thread (epic 10451
/// Phase 2c, sc-10671). Unlike [`with_cached_generator`], the generator is built by the caller-supplied
/// `load` closure (not `crate::inference_runtime::load` from a `LoadSpec`) — the path an in-place ComfyUI base takes,
/// whose weights are per-file and don't fit a registry `(engine_id, spec)` key. Any resident cached
/// generator is **evicted first** (freeing its VRAM back to the backend pool) so a large fresh load —
/// e.g. a ~28 GB in-place Wan MoE (two 14B experts) — has room; the fresh generator is dropped when
/// `run` returns (never cached). Runs on the cache thread, so it keeps that thread's serialization and
/// panic containment (an engine OOM fails only this job, and evicts).
///
/// Candle-only: the sole caller is the in-place ComfyUI Wan base lane (`video_jobs`, candle-gated), so
/// this is dead code on the MLX / non-candle builds — gated to match the caller.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn with_uncached_generator<R>(
    load: impl FnOnce() -> WorkerResult<Box<dyn Generator>> + Send + 'static,
    run: impl FnOnce(&dyn Generator) -> WorkerResult<R> + Send + 'static,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    use crate::WorkerError;
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<WorkerResult<R>>();
    let job: GeneratorJob = Box::new(move |cache: &mut GeneratorCache| {
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Free the resident cached generator (if any) before loading the fresh one, so the process
            // has room for the large in-place weights. On CUDA `release_backend_cache_after_evict` is a
            // no-op (cudarc has no empty_cache); the drop itself frees the VRAM (GPU-measured sc-13960:
            // a full generator drop returns most of it to the driver — `nvidia-smi` free rises).
            if cache.evict().is_some() {
                cache_thread::release_backend_cache_after_evict();
            }
            let generator = load()?;
            run(generator.as_ref())
        })) {
            Ok(result) => result,
            Err(panic) => {
                // Post-panic backend state is suspect; the resident (already-evicted) cache stays empty.
                cache_thread::release_backend_cache_after_evict();
                Err(WorkerError::Engine(format!(
                    "generation panicked and was contained (the engine likely ran out of memory): {}",
                    cache_thread::panic_message(panic.as_ref())
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

/// Evict the resident cached generator (if any) from the single-slot cache, freeing its cudarc pool
/// pages, and wait for the eviction to complete. Returns `true` if a generator was evicted, `false`
/// when the slot was already empty (both are success). The evict-then-reclaim primitive for the
/// bespoke candle edit/control lanes (sc-13960).
///
/// Those lanes load a bespoke `runtime_cuda` provider (`QwenEdit` / `KreaStrictControl`) off the cache
/// thread through `start_gen_stream`, so — unlike the txt2img gate (base.rs, which loads THROUGH
/// [`with_cached_generator`], evicting on a cold miss) or the video comfyui lane ([`with_uncached_generator`])
/// — nothing frees the resident txt2img generator's pages before their load. That is why they budget
/// against RAW free (sc-13588): a live co-resident generator's cudarc pool pages are NOT reclaimable,
/// so crediting them would over-admit an OOM. This gives them the missing lever: evict FIRST — freeing
/// the resident generator's VRAM back for the incoming load — then the caller may safely fold
/// [`crate::vram_gate::with_reclaimable`]. (Whether the freed VRAM returns to the driver or stays in
/// cudarc's in-process pool depends on device ownership; measured on the RTX PRO 6000, a full generator
/// drop returns most of it to the driver — either way it is available to the next load, the same
/// property the shipped video [`with_uncached_generator`] lane relies on.) The eviction is the exact
/// `cache.evict()` + [`cache_thread::release_backend_cache_after_evict`] pair [`with_uncached_generator`]
/// performs before its in-place load; on CUDA the release is a no-op (cudarc has no `empty_cache`).
///
/// **Concurrency.** The evict runs as a job on the single cache thread, so it serializes with every
/// cache load/evict; there is no window where a load observes a half-evicted slot. The worker processes
/// one generation at a time, so no concurrent job re-populates the cache between this evict and the
/// bespoke lane's subsequent `start_gen_stream` load — the same single-in-flight assumption base.rs's
/// reclaim already relies on. (Idle-timeout eviction only ever *evicts*, never loads, so it cannot
/// re-occupy the pool either.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn evict_cached_generator() -> WorkerResult<bool> {
    evict_cached_generator_on(generator_worker()).await
}

/// [`evict_cached_generator`] against a caller-supplied cache-worker sender — the seam a unit test drives
/// its own seeded [`GeneratorCache`] worker through (the production entry point uses the process-global
/// [`generator_worker`]).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn evict_cached_generator_on(worker: &mpsc::Sender<GeneratorJob>) -> WorkerResult<bool> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<bool>();
    let job: GeneratorJob = Box::new(move |cache: &mut GeneratorCache| {
        let evicted = cache.evict().is_some();
        if evicted {
            // On CUDA a no-op (cudarc has no empty_cache); the evicted generator's drop already
            // returned its allocation to the process pool for the incoming bespoke load to reuse.
            cache_thread::release_backend_cache_after_evict();
        }
        let _ = reply_tx.send(evicted);
    });
    worker
        .send(job)
        .map_err(|_| crate::WorkerError::Engine("MLX generator cache worker stopped".to_owned()))?;
    reply_rx.await.map_err(|_| {
        crate::WorkerError::Engine("MLX generator cache worker dropped the job result".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkerError;

    // sc-12178 (GitHub #1544): the requested GPU cap is clamped to the device wired ceiling so
    // `set_wired_limit` can never throw and `exit(-1)` the worker. An 8 GB Mac's ceiling is ~5.3 GB,
    // so a 7 GB cap must come back as the ceiling, not 7 GB.
    #[test]
    fn clamp_wired_limit_never_exceeds_the_device_ceiling() {
        let gib = 1024 * 1024 * 1024_usize;
        let ceiling = 5 * gib + gib / 3; // ~5.3 GiB, a realistic 8 GB-Mac working set.

        // A cap ABOVE the ceiling (the #1544 crash trigger) is pulled down to the ceiling.
        assert_eq!(clamp_wired_limit(7 * gib, ceiling), ceiling);
        // A cap BELOW the ceiling is honored unchanged.
        assert_eq!(clamp_wired_limit(4 * gib, ceiling), 4 * gib);
        // Exactly at the ceiling is allowed (set_wired_limit throws only on STRICTLY greater).
        assert_eq!(clamp_wired_limit(ceiling, ceiling), ceiling);
        // Clearing the cap (0) stays 0 regardless of ceiling.
        assert_eq!(clamp_wired_limit(0, ceiling), 0);
        // Unknown ceiling (0) ⇒ 0 ⇒ MLX default "no wired cap" (never a spurious clamp-to-something).
        assert_eq!(clamp_wired_limit(7 * gib, 0), 0);
    }

    // sc-12178 on-device probe: the clamp derives the device wired ceiling as
    // `get_memory_limit() / 1.5` (MLX documents its default limit as 1.5× the recommended working
    // set). Pure unit tests can't validate that assumption against real hardware, so this ignored
    // test does: it confirms the derived ceiling is a plausible fraction of unified memory AND that
    // `set_wired_limit(ceiling)` does NOT throw (a throw would `exit(-1)` this test process — the
    // exact #1544 crash — so a clean return IS the assertion). Run explicitly:
    //   cargo test -p sceneworks-worker --lib -- --ignored --nocapture device_wired_ceiling
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs a Metal device; run explicitly on a real Mac"]
    fn device_wired_ceiling_is_a_plausible_fraction_and_never_throws() {
        let default_limit = mlx_rs::memory::get_memory_limit();
        assert!(
            default_limit > 0,
            "MLX default memory limit should be positive"
        );
        let ceiling = default_limit / 3 * 2;

        let total: u64 = String::from_utf8_lossy(
            &std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .expect("sysctl hw.memsize")
                .stdout,
        )
        .trim()
        .parse()
        .expect("hw.memsize parses");

        eprintln!(
            "get_memory_limit()={default_limit} derived_ceiling={ceiling} hw.memsize={total} \
             (ceiling = {:.0}% of RAM)",
            ceiling as f64 / total as f64 * 100.0
        );
        // recommendedMaxWorkingSetSize is ~50–80% of unified memory on Apple Silicon; the derived
        // ceiling must land in that band (guards against the 1.5× assumption silently breaking).
        assert!(
            (ceiling as f64) > 0.4 * total as f64 && (ceiling as f64) < 0.95 * total as f64,
            "derived ceiling {ceiling} is not a plausible fraction of {total}"
        );
        // The clamp target must not throw (would exit(-1) this process). Restore the prior value after.
        let prev = mlx_rs::memory::set_wired_limit(clamp_wired_limit(usize::MAX, ceiling));
        mlx_rs::memory::set_wired_limit(prev);
    }

    // GitHub #1932: with no user-configured ceiling, MLX's own default budget is ~1.5x the device
    // recommended working set — itself ~2/3 of unified memory — i.e. roughly ALL of physical RAM. On
    // an 8 GB Mac that lets MLX starve macOS and take the machine down. The derived default keeps a
    // reserve free so MLX hits soft backpressure first.
    #[test]
    fn unset_gpu_ceiling_derives_a_default_that_reserves_memory_for_the_os() {
        let gib = 1024 * 1024 * 1024_u64;
        let reserve = 2 * gib; // fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB

        // An 8 GB Mac (the #1932 machine): 6 GiB, which still clears the 5.49 GiB z-image-turbo q4
        // baseline the fit gate's weights-fit floor admits — the ceiling never refuses a tier the
        // gate just let in.
        assert_eq!(
            resolve_gpu_memory_limit(0, Some(8 * gib)),
            8 * gib - reserve
        );
        assert!(resolve_gpu_memory_limit(0, Some(8 * gib)) < 8 * gib);
        // Same shape on bigger machines — a flat reserve, never a fraction that would lock a large
        // Mac out of its own memory.
        assert_eq!(
            resolve_gpu_memory_limit(0, Some(128 * gib)),
            128 * gib - reserve
        );

        // Load-bearing #1947 invariant, through both real decision functions: every synthetic tier
        // admitted only by the Decision 2 legacy override has a resident/staged weight set no larger
        // than the process ceiling derived from the same typed reserve.
        for (host_gib, total_weights_gib, text_encoder_gib) in
            [(8_u64, 5_u64, 2_u64), (32, 28, 12), (128, 124, 60)]
        {
            let admitted = crate::mlx_fit_gate::decide_residency(
                total_weights_gib * gib,
                text_encoder_gib * gib,
                Some(crate::mlx_fit_gate::MlxMemoryBudget {
                    total_gb: host_gib as f64,
                }),
                true,
            );
            assert!(
                !matches!(
                    admitted,
                    crate::mlx_fit_gate::ResidencyOutcome::Reject { .. }
                ),
                "fixture must exercise an admitted legacy tier"
            );
            let staged_weights = text_encoder_gib.max(total_weights_gib - text_encoder_gib) * gib;
            assert!(
                staged_weights <= resolve_gpu_memory_limit(0, Some(host_gib * gib)),
                "an admitted staged working set cannot exceed the derived process ceiling"
            );
        }

        // A configured ceiling always wins unchanged; the derived default only fills the unset case.
        assert_eq!(resolve_gpu_memory_limit(4 * gib, Some(8 * gib)), 4 * gib);
        // ...including one ABOVE the derived default: the user's explicit choice is not second-guessed.
        assert_eq!(resolve_gpu_memory_limit(7 * gib, Some(8 * gib)), 7 * gib);

        // No probe signal ⇒ 0 ⇒ leave MLX on its default, the same fail-open the fit gate takes when
        // it cannot size the machine (and the pre-#1932 behavior).
        assert_eq!(resolve_gpu_memory_limit(0, None), 0);
        // A machine at or below the reserve has no sensible ceiling to derive — 0, not a nonsense
        // limit that would thrash instead of protect.
        assert_eq!(resolve_gpu_memory_limit(0, Some(reserve)), 0);
        assert_eq!(resolve_gpu_memory_limit(0, Some(gib)), 0);
        // A configured ceiling is still honored on an unprobeable machine.
        assert_eq!(resolve_gpu_memory_limit(4 * gib, None), 4 * gib);
    }

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

    #[test]
    fn cache_key_separates_phase_residency_from_materialization_shape() {
        let base = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/z-image/q4")));
        let staged = base.clone().with_offload_policy(OffloadPolicy::Sequential);
        let deferred = base
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        let staged_deferred = staged
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);

        let keys = [&base, &staged, &deferred, &staged_deferred]
            .map(|spec| GeneratorCacheKey::from_load_spec("z_image_turbo", spec));
        for left in 0..keys.len() {
            for right in (left + 1)..keys.len() {
                assert_ne!(
                    keys[left], keys[right],
                    "all four residency/materialization combinations need distinct cache entries"
                );
            }
        }
    }

    // sc-8841 (F-039): the fingerprint helper is the core of the fix — it must report a DIFFERENT
    // value when a file at the same path changes (size or mtime), and `Unavailable` (a distinct,
    // cache-missing value) when the path can't be stat'd.
    #[test]
    fn fingerprint_tracks_content_change_and_missing_files() {
        use std::io::Write;
        use std::time::SystemTime;

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
    // `Generator` injected through the cache's explicit loader seam. It links NO tensor backend,
    // so these tests run on Linux/Windows AND macOS, proving
    // the load→progress→cancel→output contract that `with_cached_generator` is the production seam
    // for without mutating process-global discovery state.
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
            required_components: &[],
            control_kinds: None,
        }
    }

    fn stub_load(_spec: &gen_core::LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
        Ok(Box::new(StubGenerator {
            descriptor: stub_descriptor(),
        }))
    }

    fn stub_cache_key() -> GeneratorCacheKey {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/models/stub")));
        GeneratorCacheKey::from_load_spec("sc3724_stub", &spec)
    }

    /// Seed the generic cache with a resident stub generator (the test replacement for directly
    /// assigning the old `GeneratorCache.entry`, now that the entry lives in `cache_thread`).
    fn seed_stub_entry(cache: &mut GeneratorCache) {
        cache.install(
            stub_cache_key(),
            CachedGenerator {
                generator: Box::new(StubGenerator {
                    descriptor: stub_descriptor(),
                }),
                load_policy: OffloadPolicy::Resident,
                external_committed_bytes: 0,
            },
        );
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
    fn warm_hit_keeps_cold_load_policy_but_gets_fresh_access_state() {
        use std::cell::Cell;
        let mut cache = GeneratorCache::new(false);
        let loads = Cell::new(0);
        let key = stub_cache_key();
        let run = |cache: &mut GeneratorCache| {
            cache
                .with_model_access(
                    key.clone(),
                    || {
                        loads.set(loads.get() + 1);
                        Ok(CachedGenerator {
                            generator: Box::new(StubGenerator {
                                descriptor: stub_descriptor(),
                            }),
                            load_policy: OffloadPolicy::Sequential,
                            external_committed_bytes: 0,
                        })
                    },
                    |cached, access| Ok((access, cached.load_policy)),
                    "missing",
                )
                .unwrap()
        };

        assert_eq!(
            run(&mut cache),
            (CacheAccess::Cold, OffloadPolicy::Sequential)
        );
        assert_eq!(
            run(&mut cache),
            (CacheAccess::Warm, OffloadPolicy::Sequential)
        );
        assert_eq!(loads.get(), 1, "geometry-independent key loads only once");
    }

    fn start_isolated_generator_worker() -> (mpsc::Sender<GeneratorJob>, thread::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let worker = thread::spawn(move || run_generator_cache_worker(rx, None));
        (tx, worker)
    }

    fn stub_box() -> Box<dyn Generator> {
        Box::new(StubGenerator {
            descriptor: stub_descriptor(),
        })
    }

    fn cold_load_transaction(
        gate: impl FnOnce() -> WorkerResult<()> + Send + 'static,
    ) -> GeneratorColdLoadTransaction {
        GeneratorColdLoadTransaction::new(
            gen_core::CancelFlag::new(),
            GeneratorColdLoadAdmission::new(gate),
        )
    }

    #[test]
    fn cold_admission_exact_key_covers_scail_precision_adapters_and_load_layout() {
        let weights = tempfile::tempdir().expect("weights");
        let adapter = weights.path().join("style.safetensors");
        std::fs::write(&adapter, b"adapter").expect("adapter fixture");
        let base = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let key = GeneratorCacheKey::from_load_spec("scail2", &base);
        assert_eq!(
            key,
            GeneratorCacheKey::from_load_spec("scail2", &base),
            "an identical resident identity is the only warm-hit shape"
        );

        let mut precision = base.clone();
        precision.precision = Precision::Fp32;
        let mut adapted = base.clone();
        adapted.adapters = vec![AdapterSpec::new(adapter, 0.75, AdapterKind::Lora)];
        let layout = base
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        let offload = base.clone().with_offload_policy(OffloadPolicy::Sequential);
        for (field, changed) in [
            ("precision", precision),
            ("adapters", adapted),
            ("load shape", layout),
            ("offload policy", offload),
        ] {
            assert_ne!(
                key,
                GeneratorCacheKey::from_load_spec("scail2", &changed),
                "changing {field} must force cold admission before a different resident loads"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_admission_sequence_loads_then_reuses_warm_and_rejects_a_different_key() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let weights = tempfile::tempdir().expect("weights");
        let base = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let mut different = base.clone();
        different.precision = Precision::Fp32;
        let (tx, worker) = start_isolated_generator_worker();
        let gates = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));

        let first_gates = gates.clone();
        let first_loads = loads.clone();
        with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            base.clone(),
            "stub load",
            cold_load_transaction(move || {
                first_gates.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
            move |_id, _spec| {
                first_loads.fetch_add(1, Ordering::SeqCst);
                Ok(stub_box())
            },
            |_generator| Ok(()),
        )
        .await
        .expect("first cold request is admitted and loaded");

        with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            base,
            "stub load",
            cold_load_transaction(|| {
                Err(WorkerError::InvalidPayload(
                    "synthetic low-free budget".to_owned(),
                ))
            }),
            |_id, _spec| panic!("same exact key must not reload"),
            |_generator| Ok(()),
        )
        .await
        .expect("same-key warm request bypasses cold-load admission");

        let rejected_loads = loads.clone();
        let rejected = with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            different,
            "stub load",
            cold_load_transaction(|| {
                Err(WorkerError::InvalidPayload(
                    "synthetic low-free budget".to_owned(),
                ))
            }),
            move |_id, _spec| {
                rejected_loads.fetch_add(1, Ordering::SeqCst);
                Ok(stub_box())
            },
            |_generator| Ok(()),
        )
        .await;
        assert!(matches!(rejected, Err(WorkerError::InvalidPayload(_))));
        assert_eq!(
            gates.load(Ordering::SeqCst),
            1,
            "warm hit bypassed its gate"
        );
        assert_eq!(
            loads.load(Ordering::SeqCst),
            1,
            "rejected miss never loaded"
        );

        let (state_tx, state_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            state_tx.send(cache.is_empty()).expect("cache state");
        }))
        .expect("inspect cache");
        assert!(state_rx.recv().expect("cache state reply"));
        drop(tx);
        worker.join().expect("isolated cache worker exits");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn canceled_queued_cold_miss_preserves_resident_without_gate_load_or_run() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let weights = tempfile::tempdir().expect("weights");
        let base = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let resident_key = GeneratorCacheKey::from_load_spec("scail2", &base);
        let mut different = base.clone();
        different.precision = Precision::Fp32;
        let (tx, worker) = start_isolated_generator_worker();

        with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            base,
            "stub load",
            cold_load_transaction(|| Ok(())),
            |_id, _spec| Ok(stub_box()),
            |_generator| Ok(()),
        )
        .await
        .expect("seed resident generator");

        // Hold the cache thread so the canceled request is definitely an abandoned queued job,
        // matching a Tokio waiter aborted by the video cancel/join guard before std::mpsc receives
        // its closure.
        let (block_started_tx, block_started_rx) = mpsc::channel();
        let (block_release_tx, block_release_rx) = mpsc::channel();
        tx.send(Box::new(move |_cache: &mut GeneratorCache| {
            block_started_tx.send(()).expect("blocker started");
            block_release_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("release cache blocker");
        }))
        .expect("queue cache blocker");
        block_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache worker is blocked");

        let gates = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let gate_count = gates.clone();
        let load_count = loads.clone();
        let run_count = runs.clone();
        let request_cancel = gen_core::CancelFlag::new();
        let mut queued = Box::pin(with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            different,
            "stub load",
            GeneratorColdLoadTransaction::new(
                request_cancel.clone(),
                GeneratorColdLoadAdmission::new(move || {
                    gate_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ),
            move |_id, _spec| {
                load_count.fetch_add(1, Ordering::SeqCst);
                Ok(stub_box())
            },
            move |_generator| {
                run_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        ));
        // The first poll runs through the unbounded send and parks on the oneshot reply. Trip the
        // request flag while its std::mpsc job remains queued, then keep the waiter alive so this
        // case proves the explicit typed-cancel path independently of receiver abandonment.
        tokio::select! {
            biased;
            result = &mut queued => panic!("blocked cache job resolved unexpectedly: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        request_cancel.cancel();
        block_release_tx.send(()).expect("release cache worker");
        assert!(matches!(queued.await, Err(WorkerError::Canceled(_))));

        let (state_tx, state_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            state_tx
                .send(cache.resident_key().cloned())
                .expect("cache state");
        }))
        .expect("inspect cache after canceled queued job");
        assert_eq!(
            state_rx.recv().expect("cache state reply"),
            Some(resident_key),
            "a canceled queued miss must not evict or replace the useful resident"
        );
        assert_eq!(gates.load(Ordering::SeqCst), 0, "admission must not run");
        assert_eq!(loads.load(Ordering::SeqCst), 0, "loader must not run");
        assert_eq!(runs.load(Ordering::SeqCst), 0, "generation must not run");

        drop(tx);
        worker.join().expect("isolated cache worker exits");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abandoned_queued_cold_miss_preserves_resident_without_gate_load_or_run() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let weights = tempfile::tempdir().expect("weights");
        let base = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let resident_key = GeneratorCacheKey::from_load_spec("scail2", &base);
        let mut different = base.clone();
        different.precision = Precision::Fp32;
        let (tx, worker) = start_isolated_generator_worker();
        with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            base,
            "stub load",
            cold_load_transaction(|| Ok(())),
            |_id, _spec| Ok(stub_box()),
            |_generator| Ok(()),
        )
        .await
        .expect("seed resident generator");

        let (block_started_tx, block_started_rx) = mpsc::channel();
        let (block_release_tx, block_release_rx) = mpsc::channel();
        tx.send(Box::new(move |_cache: &mut GeneratorCache| {
            block_started_tx.send(()).expect("blocker started");
            block_release_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("release cache blocker");
        }))
        .expect("queue cache blocker");
        block_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache worker is blocked");

        let gates = Arc::new(AtomicUsize::new(0));
        let loads = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let gate_count = gates.clone();
        let load_count = loads.clone();
        let run_count = runs.clone();
        let mut queued = Box::pin(with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            different,
            "stub load",
            cold_load_transaction(move || {
                gate_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
            move |_id, _spec| {
                load_count.fetch_add(1, Ordering::SeqCst);
                Ok(stub_box())
            },
            move |_generator| {
                run_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        ));
        tokio::select! {
            biased;
            result = &mut queued => panic!("blocked cache job resolved unexpectedly: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        // Simulate task abort/panic/drop without a cooperative CancelFlag trip. Dropping the future
        // closes the oneshot receiver while leaving the already-enqueued std::mpsc closure alive.
        drop(queued);
        block_release_tx.send(()).expect("release cache worker");

        let (state_tx, state_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            state_tx
                .send(cache.resident_key().cloned())
                .expect("cache state");
        }))
        .expect("inspect cache after abandoned queued job");
        assert_eq!(
            state_rx.recv().expect("cache state reply"),
            Some(resident_key),
            "an abandoned queued miss must not evict or replace the useful resident"
        );
        assert_eq!(gates.load(Ordering::SeqCst), 0, "admission must not run");
        assert_eq!(loads.load(Ordering::SeqCst), 0, "loader must not run");
        assert_eq!(runs.load(Ordering::SeqCst), 0, "generation must not run");

        drop(tx);
        worker.join().expect("isolated cache worker exits");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_admission_is_serialized_behind_an_in_flight_warm_request() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let weights = tempfile::tempdir().expect("weights");
        let base = LoadSpec::new(WeightsSource::Dir(weights.path().to_path_buf()));
        let mut different = base.clone();
        different.precision = Precision::Fp32;
        let (tx, worker) = start_isolated_generator_worker();
        with_cached_generator_using_cold_admission_on(
            tx.clone(),
            "scail2",
            base.clone(),
            "stub load",
            cold_load_transaction(|| Ok(())),
            |_id, _spec| Ok(stub_box()),
            |_generator| Ok(()),
        )
        .await
        .expect("seed cold load");

        let (warm_started_tx, warm_started_rx) = mpsc::channel();
        let (warm_release_tx, warm_release_rx) = mpsc::channel();
        let warm_tx = tx.clone();
        let warm = tokio::spawn(async move {
            with_cached_generator_using_cold_admission_on(
                warm_tx,
                "scail2",
                base,
                "stub load",
                cold_load_transaction(|| panic!("the queued exact warm hit must bypass admission")),
                |_id, _spec| panic!("the queued exact warm hit must not reload"),
                move |_generator| {
                    warm_started_tx.send(()).expect("warm started");
                    warm_release_rx
                        .recv_timeout(Duration::from_secs(2))
                        .expect("release warm run");
                    Ok(())
                },
            )
            .await
        });
        warm_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("warm request owns the cache transaction");

        let gate_called = Arc::new(AtomicBool::new(false));
        let cold_gate_called = gate_called.clone();
        let cold_tx = tx.clone();
        let cold = tokio::spawn(async move {
            with_cached_generator_using_cold_admission_on(
                cold_tx,
                "scail2",
                different,
                "stub load",
                cold_load_transaction(move || {
                    cold_gate_called.store(true, Ordering::SeqCst);
                    Err(WorkerError::InvalidPayload(
                        "synthetic low-free budget".to_owned(),
                    ))
                }),
                |_id, _spec| panic!("rejected queued miss must not load"),
                |_generator| Ok(()),
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !gate_called.load(Ordering::SeqCst),
            "a queued miss cannot peek/admit while a warm run owns the cache"
        );
        warm_release_tx.send(()).expect("release warm request");
        warm.await.expect("warm task joins").expect("warm succeeds");
        let rejected = cold.await.expect("cold task joins");
        assert!(matches!(rejected, Err(WorkerError::InvalidPayload(_))));
        assert!(gate_called.load(Ordering::SeqCst));

        let (state_tx, state_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            state_tx.send(cache.is_empty()).expect("cache state");
        }))
        .expect("inspect cache");
        assert!(state_rx.recv().expect("cache state reply"));
        drop(tx);
        worker.join().expect("isolated cache worker exits");
    }

    #[test]
    fn cache_worker_evicts_resident_generator_after_idle_timeout() {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let worker = thread::spawn(move || {
            run_generator_cache_worker(rx, Some(Duration::from_millis(20)));
        });
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            seed_stub_entry(cache);
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
            seed_stub_entry(cache);
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        thread::sleep(Duration::from_millis(80));

        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            reply_tx.send(!cache.is_empty()).expect("send cache state");
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

    // sc-13960: the evict-then-reclaim primitive frees the single resident slot on demand (the
    // bespoke edit/control lanes call it before folding `with_reclaimable`). It must (1) evict a
    // resident generator and report `true`, (2) leave the slot empty, and (3) no-op with `false` when
    // the slot is already empty — so a lane that evicts on an already-cold worker neither errors nor
    // lies about having freed pages. Drives a locally-seeded worker through the `_on` seam rather than
    // the process-global cache. Candle-gated to match the primitive.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[tokio::test]
    async fn evict_cached_generator_frees_the_resident_slot_and_no_ops_when_empty() {
        let (tx, rx) = mpsc::channel::<GeneratorJob>();
        let worker = thread::spawn(move || {
            run_generator_cache_worker(rx, None);
        });

        // Seed a resident stub generator (the test replacement for a warm txt2img generator).
        let (seed_tx, seed_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            seed_stub_entry(cache);
            seed_tx.send(()).expect("ack cache seed");
        }))
        .expect("seed cache entry");
        seed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache seed ack");

        // First evict frees the resident slot and reports it evicted something.
        assert!(
            evict_cached_generator_on(&tx)
                .await
                .expect("evict succeeds"),
            "evicting a resident generator must report true"
        );

        // The slot is now empty.
        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(Box::new(move |cache: &mut GeneratorCache| {
            reply_tx.send(cache.is_empty()).expect("send cache state");
        }))
        .expect("check cache state");
        assert!(
            reply_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("cache state reply"),
            "the resident generator must be gone after an evict"
        );

        // A second evict on the now-empty slot is a no-op that reports false (never an error).
        assert!(
            !evict_cached_generator_on(&tx)
                .await
                .expect("evict on empty succeeds"),
            "evicting an empty slot must report false, not error"
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

        let fact = with_cached_generator_using(
            "sc3724_stub",
            spec,
            "stub load",
            |_id, spec| stub_load(spec),
            move |generator| {
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
                    .ok_or_else(|| {
                        WorkerError::Engine("stub image buffer size mismatch".to_owned())
                    })?;
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
            },
        )
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

        let result = with_cached_generator_using(
            "sc3724_stub",
            spec,
            "stub load",
            |_id, spec| stub_load(spec),
            move |generator| {
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
            },
        )
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
        let panicked = with_cached_generator_using(
            "sc3724_stub",
            spec.clone(),
            "stub load",
            |_id, spec| stub_load(spec),
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
        let after = with_cached_generator_using(
            "sc3724_stub",
            spec,
            "stub load",
            |_id, spec| stub_load(spec),
            move |generator| {
                let req = gen_core::GenerationRequest {
                    width: 2,
                    height: 2,
                    ..Default::default()
                };
                generator
                    .generate(&req, &mut |_progress| {})
                    .map(|_| ())
                    .map_err(|error| WorkerError::Engine(error.to_string()))
            },
        )
        .await;
        assert!(
            after.is_ok(),
            "worker must keep serving after a contained panic, got {after:?}"
        );
    }
}
