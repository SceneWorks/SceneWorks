//! Per-job generation-metrics probe (epic 10402, sc-10404).
//!
//! Wraps a job's execution in `run_utility_job` to capture the
//! externally-observable hardware metrics — peak GPU memory, peak GPU load, and
//! total wall-clock — and POST them once to `/api/v1/jobs/:id/metrics` on
//! completion. Every generation runs one-at-a-time on the single
//! generator-cache thread (`generator_cache::with_cached_generator`), so a
//! process-global `reset_peak_memory` / `get_peak_memory` window is safe per
//! job. Resolved settings (S4) and per-phase timings (S3) are posted separately
//! by the handlers and coalesce-merge into the same row on the server.
//!
//! This restores — and broadens to every job type — the per-job peak GPU
//! sampling that the Python→Rust cutover dropped (sc-2086).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sceneworks_core::contracts::GenerationMetrics;

use crate::api_client::ApiClient;
use crate::gpu::gpu_utilization;

/// Reset the MLX peak-memory high-water mark at job start (macOS/MLX only). The
/// counter is process-global, so this scopes the subsequent `get_peak_memory`
/// read to this job. Matches the `generator_cache::write_gpu_telemetry` gating —
/// tests never touch real MLX.
#[cfg(all(target_os = "macos", not(test)))]
fn reset_peak_memory() {
    mlx_rs::memory::reset_peak_memory();
}
#[cfg(any(not(target_os = "macos"), test))]
fn reset_peak_memory() {}

/// Read the MLX peak-memory high-water mark in bytes (macOS/MLX only). `None` on
/// other backends, where peak memory is derived from the sampled `memory.used`
/// high-water mark instead.
#[cfg(all(target_os = "macos", not(test)))]
fn mlx_peak_memory_bytes() -> Option<u64> {
    Some(mlx_rs::memory::get_peak_memory() as u64)
}
#[cfg(any(not(target_os = "macos"), test))]
fn mlx_peak_memory_bytes() -> Option<u64> {
    None
}

/// Total device memory in bytes for the pct denominator. macOS reads the cached
/// `sysctl hw.memsize` (unified memory); other backends fall back to the sampled
/// `memory.total` from the utilization snapshot.
async fn total_memory_bytes(sampled_total_mb: u64) -> Option<u64> {
    #[cfg(all(target_os = "macos", not(test)))]
    {
        if let Some(gb) = crate::gpu::total_unified_memory_gb().await {
            return Some((gb * 1024.0 * 1024.0 * 1024.0) as u64);
        }
    }
    (sampled_total_mb > 0).then_some(sampled_total_mb * 1024 * 1024)
}

/// Normalize the worker's `gpu_id` to the contract's backend vocabulary. The
/// Rust worker is either the Apple-Silicon MLX worker (`gpu_id == "mlx"`), a
/// candle/NVIDIA worker (a device index), or CPU-only.
fn normalize_backend(gpu_id: &str) -> &'static str {
    match gpu_id {
        "mlx" => "mlx",
        "" | "cpu" => "cpu",
        _ => "cuda",
    }
}

/// A running probe scoped to a single job. Created via [`JobMetricsProbe::start`]
/// before the handler runs and consumed by [`JobMetricsProbe::finish`] after, so
/// the peak-memory window and GPU-load samples cover the whole job (model load +
/// generation + any inline upscale).
pub(crate) struct JobMetricsProbe {
    started: Instant,
    backend: &'static str,
    sampler: Option<tokio::task::JoinHandle<()>>,
    load_permille_max: Arc<AtomicU32>,
    mem_used_mb_max: Arc<AtomicU64>,
    mem_total_mb: Arc<AtomicU64>,
}

/// How often the per-job sampler reads GPU load/memory. Deliberately short (and
/// decoupled from the 5–15s heartbeat cadence) so even a few-second turbo
/// generation lands samples during the active GPU window — the first tick fires
/// at t=0 while the model is still loading (idle), so a slower cadence would miss
/// the denoise/decode burst on a fast job entirely. The underlying probe is a
/// lightweight `ioreg` / `nvidia-smi` read; `MissedTickBehavior::Delay`
/// self-throttles if one ever runs long, so this never stacks subprocesses.
const GPU_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

impl JobMetricsProbe {
    /// Reset the MLX peak counter and spawn the background GPU-load/memory
    /// sampler (see [`GPU_SAMPLE_INTERVAL`]). A CPU-only worker has no GPU to
    /// query, so no sampler is spawned.
    pub(crate) fn start(gpu_id: &str) -> Self {
        reset_peak_memory();
        let load_permille_max = Arc::new(AtomicU32::new(0));
        let mem_used_mb_max = Arc::new(AtomicU64::new(0));
        let mem_total_mb = Arc::new(AtomicU64::new(0));
        let sampler = if gpu_id.is_empty() || gpu_id == "cpu" {
            None
        } else {
            let gpu_id = gpu_id.to_owned();
            let load = load_permille_max.clone();
            let used = mem_used_mb_max.clone();
            let total = mem_total_mb.clone();
            Some(tokio::spawn(async move {
                let mut ticker = tokio::time::interval(GPU_SAMPLE_INTERVAL);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    ticker.tick().await;
                    if let Some(snapshot) = gpu_utilization(&gpu_id).await {
                        if let Some(load_pct) = snapshot.gpu_load_percent {
                            let permille = (load_pct * 10.0).round().clamp(0.0, 1000.0) as u32;
                            load.fetch_max(permille, Ordering::Relaxed);
                        }
                        if let Some(used_mb) = snapshot.memory_used_mb {
                            used.fetch_max(used_mb, Ordering::Relaxed);
                        }
                        if let Some(total_mb) = snapshot.memory_total_mb {
                            total.store(total_mb, Ordering::Relaxed);
                        }
                    }
                }
            }))
        };
        Self {
            started: Instant::now(),
            backend: normalize_backend(gpu_id),
            sampler,
            load_permille_max,
            mem_used_mb_max,
            mem_total_mb,
        }
    }

    /// Stop sampling and build the metrics block: peak memory (MLX exact on
    /// macOS, sampled max elsewhere), peak GPU load, total wall-clock, backend.
    pub(crate) async fn finish(mut self) -> GenerationMetrics {
        if let Some(handle) = self.sampler.take() {
            handle.abort();
        }
        let total_ms = self.started.elapsed().as_millis() as u64;
        let load_permille = self.load_permille_max.load(Ordering::Relaxed);
        let peak_gpu_load_pct = (load_permille > 0)
            .then(|| serde_json::Number::from_f64(load_permille as f64 / 10.0))
            .flatten();
        let sampled_used_mb = self.mem_used_mb_max.load(Ordering::Relaxed);
        let peak_memory_bytes = mlx_peak_memory_bytes()
            .or_else(|| (sampled_used_mb > 0).then_some(sampled_used_mb * 1024 * 1024));
        let total_bytes = total_memory_bytes(self.mem_total_mb.load(Ordering::Relaxed)).await;
        let peak_memory_pct = match (peak_memory_bytes, total_bytes) {
            (Some(peak), Some(total)) if total > 0 => {
                serde_json::Number::from_f64(peak as f64 / total as f64 * 100.0)
            }
            _ => None,
        };
        GenerationMetrics {
            backend: Some(self.backend.to_owned()),
            total_ms: Some(total_ms),
            peak_memory_bytes,
            peak_memory_pct,
            peak_gpu_load_pct,
            ..Default::default()
        }
    }
}

/// Accumulates load / sample / decode wall-clock from a generation's
/// phase-boundary events (epic 10402, sc-10405). Timestamps are passed in rather
/// than read from a clock, so the accumulation is unit-testable and the logic is
/// shared verbatim by the image and video stream consumers:
/// load = start→first step, sample = step→decoding, decode = decoding→item-done,
/// summed across a batch's items (the model loads once, so load is one-time).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct PhaseTimer {
    start: Instant,
    load_ms: Option<u64>,
    sample_ms: u64,
    decode_ms: u64,
    sample_started: Option<Instant>,
    decode_started: Option<Instant>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl PhaseTimer {
    pub(crate) fn new(start: Instant) -> Self {
        Self {
            start,
            load_ms: None,
            sample_ms: 0,
            decode_ms: 0,
            sample_started: None,
            decode_started: None,
        }
    }

    /// A denoise/sample step arrived. The first step across the whole job ends
    /// the load phase; the first step of each item opens its sample span.
    pub(crate) fn mark_sample_step(&mut self, now: Instant) {
        if self.load_ms.is_none() {
            self.load_ms = Some(now.saturating_duration_since(self.start).as_millis() as u64);
        }
        if self.sample_started.is_none() {
            self.sample_started = Some(now);
        }
    }

    /// Decoding began — close the open sample span and open the decode span.
    pub(crate) fn mark_decoding(&mut self, now: Instant) {
        if let Some(started_at) = self.sample_started.take() {
            self.sample_ms += now.saturating_duration_since(started_at).as_millis() as u64;
        }
        if self.decode_started.is_none() {
            self.decode_started = Some(now);
        }
    }

    /// An item finished (image ready / engine returned) — close the decode span,
    /// or fold the remaining time into sample if the engine emitted no decoding
    /// event.
    pub(crate) fn mark_item_done(&mut self, now: Instant) {
        if let Some(started_at) = self.decode_started.take() {
            self.decode_ms += now.saturating_duration_since(started_at).as_millis() as u64;
        } else if let Some(started_at) = self.sample_started.take() {
            self.sample_ms += now.saturating_duration_since(started_at).as_millis() as u64;
        }
    }

    /// Consume into a phase-only metrics block, closing any span still open at
    /// `now` (video's decode ends when the engine returns, with no event).
    /// Returns None when nothing was measured, so the caller skips an empty POST.
    pub(crate) fn into_metrics(mut self, now: Instant) -> Option<GenerationMetrics> {
        self.mark_item_done(now);
        if self.load_ms.is_none() && self.sample_ms == 0 && self.decode_ms == 0 {
            return None;
        }
        Some(GenerationMetrics {
            load_ms: self.load_ms,
            sample_ms: (self.sample_ms > 0).then_some(self.sample_ms),
            decode_ms: (self.decode_ms > 0).then_some(self.decode_ms),
            ..Default::default()
        })
    }
}

/// Best-effort POST of a job's metrics block. A failure is logged and swallowed
/// so telemetry never fails the job. Coalesce-merges server-side, so this
/// composes with the settings/timing blocks the handlers post separately.
pub(crate) async fn post_generation_metrics(
    api: &ApiClient,
    job_id: &str,
    metrics: &GenerationMetrics,
) {
    if let Err(error) = api
        .post_json::<GenerationMetrics, GenerationMetrics>(
            &format!("/api/v1/jobs/{job_id}/metrics"),
            metrics,
        )
        .await
    {
        tracing::warn!(
            event = "job_metrics_post_failed",
            jobId = %job_id,
            error = %error,
            "failed to post generation metrics"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_backend_maps_worker_gpu_ids() {
        assert_eq!(normalize_backend("mlx"), "mlx");
        assert_eq!(normalize_backend("cpu"), "cpu");
        assert_eq!(normalize_backend(""), "cpu");
        assert_eq!(normalize_backend("0"), "cuda");
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn phase_timer_splits_load_sample_decode_and_sums_items() {
        let t0 = Instant::now();
        let mut timer = PhaseTimer::new(t0);
        // Item 0: cold load (200ms) → sample (300ms) → decode (100ms).
        timer.mark_sample_step(t0 + Duration::from_millis(200));
        timer.mark_sample_step(t0 + Duration::from_millis(350));
        timer.mark_decoding(t0 + Duration::from_millis(500));
        timer.mark_item_done(t0 + Duration::from_millis(600));
        // Item 1: model cached (no new load) → sample (300ms) → decode (50ms).
        timer.mark_sample_step(t0 + Duration::from_millis(700));
        timer.mark_decoding(t0 + Duration::from_millis(1000));
        timer.mark_item_done(t0 + Duration::from_millis(1050));
        let metrics = timer
            .into_metrics(t0 + Duration::from_millis(1050))
            .expect("measured");
        assert_eq!(
            metrics.load_ms,
            Some(200),
            "load is one-time (first step only)"
        );
        assert_eq!(metrics.sample_ms, Some(600), "sample sums across items");
        assert_eq!(metrics.decode_ms, Some(150), "decode sums across items");
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn phase_timer_folds_into_sample_without_decoding_event() {
        let t0 = Instant::now();
        let mut timer = PhaseTimer::new(t0);
        timer.mark_sample_step(t0 + Duration::from_millis(100));
        // Engine returns without a decoding event → the open span becomes sample.
        let metrics = timer
            .into_metrics(t0 + Duration::from_millis(500))
            .expect("measured");
        assert_eq!(metrics.load_ms, Some(100));
        assert_eq!(metrics.sample_ms, Some(400));
        assert_eq!(metrics.decode_ms, None);
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn phase_timer_reports_nothing_without_events() {
        let t0 = Instant::now();
        let timer = PhaseTimer::new(t0);
        assert!(timer.into_metrics(t0 + Duration::from_millis(50)).is_none());
    }

    #[tokio::test]
    async fn cpu_probe_reports_wall_clock_without_gpu_samples() {
        // A CPU worker spawns no sampler; finish() still reports total_ms + a
        // normalized backend, and no GPU peaks (nothing to sample).
        let probe = JobMetricsProbe::start("cpu");
        let metrics = probe.finish().await;
        assert_eq!(metrics.backend.as_deref(), Some("cpu"));
        assert!(metrics.total_ms.is_some());
        assert!(metrics.peak_gpu_load_pct.is_none());
        assert!(metrics.peak_memory_pct.is_none());
    }
}
