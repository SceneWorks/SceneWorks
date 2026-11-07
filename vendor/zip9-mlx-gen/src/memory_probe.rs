//! Peak MLX **allocator footprint** sampling (`active + cache`).
//!
//! # Why `get_peak_memory` is not the whole residency picture
//!
//! MLX publishes two live counters: [`get_active_memory`](mlx_rs::memory::get_active_memory), the
//! bytes currently held by live arrays, and [`get_cache_memory`](mlx_rs::memory::get_cache_memory),
//! the bytes MLX has freed internally but retained for reuse rather than returning to the OS.
//! `get_peak_memory` is the high-water mark of the **first** only.
//!
//! An operating system that kills processes for using too much memory does not make that
//! distinction. Darwin's `phys_footprint` — the quantity iOS jetsam reads — counts both. A Z-Image
//! 1024² render on an iPhone held `active + cache` at a conserved **6068 MiB** across every sample,
//! the cache absorbing exactly what active released, and was killed with 4 MiB of headroom while
//! `get_peak_memory` reported 2901 MiB against a 6136 MiB cap. The peak was not wrong; it was
//! answering a different question than the one jetsam asks.
//!
//! # What this measures, and what it does not predict
//!
//! **This is not a portable process-footprint estimate or an admission verdict.** MLX sizes its cache
//! limit from the host's recommended working set, so on a 64 GB Mac the cache can grow almost without
//! bound. This probe then reports unconstrained allocator retention, not what the same workload will
//! retain in a differently capped process. The host values are also not monotone in obvious workload
//! parameters: a Z-Image 1024² decode measured 16002 MiB at a 512 px tile, 6488 MiB at 256 px, and
//! 43157 MiB at 640 px (reproducible to ±0.2%; different sizes hit different allocator size classes).
//!
//! A useful model for a cache-bounded target is:
//!
//! ```text
//! allocator footprint ≈ peak active + min(cache wanted, configured cache limit)
//! ```
//!
//! Therefore `get_peak_memory` remains the evidence/admission currency: it measures the irreducible
//! live working set. This probe answers a different question: whether cache bounding is required and
//! whether the configured reclaimer actually keeps allocator retention bounded. Calibration
//! harnesses may record it as auxiliary telemetry, but must not write it into
//! `MemoryEvidence::{predicted,observed}_peak_bytes` or compare an unconstrained host sample directly
//! with a target-device cap.
//!
//! The two MLX counters are separate API reads rather than one atomic snapshot. A transfer from
//! active storage into the reuse cache can occur between them, so one sample can transiently over- or
//! under-count. The running maximum is therefore an approximate retention diagnostic, not an exact
//! process measurement; it also excludes every non-MLX allocation in the process.
//!
//! # Why sampling, and not a read at the end
//!
//! The cache term moves fast and in the opposite direction to `active` — freeing a large array
//! lowers active and raises cache by the same amount within one allocation. Reading after the work
//! finishes therefore observes a quiet moment, not the peak, and reading only at phase boundaries
//! observes whichever moment the phase happened to end on. A background thread at a fixed interval
//! is the only way to catch an excursion that occurs *between* the points a harness thinks to look —
//! and on a memory-capped device the excursion that matters can be the one that ended the process.
//!
//! The measurements and kill attribution that motivated this shared probe were produced by the iOS
//! runtime work tracked in Shortcut epic 16774; sc-16784 is the backend-contract half of that work.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// A running maximum of MLX `active + cache`, sampled on a background thread.
///
/// Start it before the work being measured and call [`FootprintProbe::finish`] after. See the module
/// docs for the strict distinction between this allocator-local telemetry, process footprint, and
/// the live-allocation value used by memory-ladder evidence.
///
/// ```no_run
/// # use mlx_gen::memory_probe::FootprintProbe;
/// let probe = FootprintProbe::start_default();
/// // ... run a generation ...
/// let peak_bytes = probe.finish();
/// ```
pub struct FootprintProbe {
    /// Dropping this disconnects the channel the sampler waits on, waking it immediately.
    ///
    /// An `AtomicBool` checked around a `thread::sleep` is the obvious shape and is wrong here: the
    /// thread can only observe the flag *between* sleeps, so stopping a probe blocks for up to one
    /// full interval. That is invisible at 50 ms and a hang at any interval chosen to be lazy —
    /// [`FootprintProbe::finish`] joins, so the caller inherits the wait. A disconnect-driven
    /// `recv_timeout` makes the interval an upper bound on *sampling* rather than on *stopping*.
    stop: Option<Sender<()>>,
    peak_bytes: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// The default sampling interval: fast enough to catch a single VAE tile's transient (tens of ms on
/// a host), slow enough that the two atomic reads cost nothing beside the work being measured.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(50);

/// One near-simultaneous read of MLX's live active and cache counters.
///
/// MLX exposes the counters through separate calls, so the pair is approximate rather than atomic;
/// keeping both values from the same sampler tick is nevertheless strictly more honest than adding
/// independent high-water marks that may never have coexisted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocatorSample {
    pub active_bytes: u64,
    pub cache_bytes: u64,
}

impl AllocatorSample {
    pub fn footprint_bytes(self) -> u64 {
        self.active_bytes.saturating_add(self.cache_bytes)
    }
}

/// A raw sampler gap at least this many intervals long is kept as a [`SamplerGap`] record.
pub const GAP_RECORD_INTERVALS: u64 = 4;
/// At most this many [`SamplerGap`] records are kept; later ones are only counted.
pub const MAX_GAP_RECORDS: usize = 256;

/// One sampler gap of at least [`GAP_RECORD_INTERVALS`] intervals, with what was in flight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplerGap {
    /// Gap start (the previous tick), microseconds after the probe's first sample.
    pub start_micros: u64,
    /// Raw time between the two sampler ticks.
    pub duration_micros: u64,
    /// The longest stretch of the gap that no [`clear_cache`] window covers.
    pub uncovered_micros: u64,
    /// Time inside the gap covered by [`clear_cache`] windows.
    pub release_micros: u64,
    /// Number of [`clear_cache`] windows that overlap the gap.
    pub release_windows: u64,
    /// The [`set_phase`] label in force at the gap's start and at its end.
    pub phase_at_start: Option<String>,
    pub phase_at_end: Option<String>,
}

/// Coverage and peak receipt from one fixed-cadence allocator sampling interval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocatorProbeReport {
    pub interval_micros: u64,
    pub sample_count: u64,
    pub periodic_sample_count: u64,
    pub sampling_span_micros: u64,
    /// Longest raw time between two sampler ticks.
    pub max_gap_micros: u64,
    /// Longest time no sample point bounded `active + cache`: ticks, plus the entry and exit samples
    /// of every [`clear_cache`] window, whose interior is excluded because `active + cache` cannot
    /// rise while the evaluating thread is inside `clear_cache` (see [`clear_cache`]).
    pub max_uncovered_gap_micros: u64,
    /// Every raw gap of at least [`GAP_RECORD_INTERVALS`] intervals (first [`MAX_GAP_RECORDS`]).
    pub gaps: Vec<SamplerGap>,
    /// Recordable gaps beyond [`MAX_GAP_RECORDS`].
    pub gaps_not_recorded: u64,
    /// [`clear_cache`] windows observed while sampling.
    pub release_window_count: u64,
    /// Whether the sampler thread runs under a Mach time-constraint (real-time) policy.
    pub realtime_sampler: bool,
    pub sampled_active_peak_bytes: u64,
    pub sampled_cache_peak_bytes: u64,
    /// Maximum `active + cache` from one sampler tick, never a sum of independent maxima.
    pub sampled_footprint_peak_bytes: u64,
    /// Active bytes from the exact sampler tick that established `sampled_footprint_peak_bytes`.
    pub footprint_peak_active_bytes: u64,
    /// Cache bytes from the exact sampler tick that established `sampled_footprint_peak_bytes`.
    pub footprint_peak_cache_bytes: u64,
    pub boundary_active_bytes: u64,
    pub boundary_cache_bytes: u64,
}

impl AllocatorProbeReport {
    fn fold(&mut self, sample: AllocatorSample) {
        self.sampled_active_peak_bytes = self.sampled_active_peak_bytes.max(sample.active_bytes);
        self.sampled_cache_peak_bytes = self.sampled_cache_peak_bytes.max(sample.cache_bytes);
        let footprint = sample.footprint_bytes();
        if footprint > self.sampled_footprint_peak_bytes {
            self.sampled_footprint_peak_bytes = footprint;
            self.footprint_peak_active_bytes = sample.active_bytes;
            self.footprint_peak_cache_bytes = sample.cache_bytes;
        }
    }
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------------------------
// Process-wide observation hooks
// ---------------------------------------------------------------------------------------------

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now_micros() -> u64 {
    duration_micros(epoch().elapsed())
}

static PHASE: Mutex<Option<String>> = Mutex::new(None);
static LISTENERS: Mutex<Vec<(u64, Sender<ProbeMessage>)>> = Mutex::new(Vec::new());
static NEXT_LISTENER: AtomicU64 = AtomicU64::new(0);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Label the phase now in flight; an [`AllocatorProbe`] records it on every [`SamplerGap`].
pub fn set_phase(label: &str) {
    *lock(&PHASE) = Some(label.to_owned());
}

fn current_phase() -> Option<String> {
    lock(&PHASE).clone()
}

/// A [`clear_cache`] call bracketed by samples taken on the calling thread.
#[derive(Clone, Copy, Debug)]
struct ReleaseWindow {
    start_micros: u64,
    end_micros: u64,
    before: AllocatorSample,
    after: AllocatorSample,
}

enum ProbeMessage {
    Release(ReleaseWindow),
    Stop,
}

fn mlx_sample() -> AllocatorSample {
    AllocatorSample {
        active_bytes: mlx_rs::memory::get_active_memory() as u64,
        cache_bytes: mlx_rs::memory::get_cache_memory() as u64,
    }
}

/// `mlx_rs::memory::clear_cache`, reported to every running [`AllocatorProbe`] as a release window.
///
/// The window is bracketed by an `active + cache` sample on the calling thread immediately before
/// and after the clear. Inside it `active + cache` cannot rise when the caller is the thread that
/// evaluates MLX work: `clear_cache` holds the allocator mutex that every `malloc` and `free` also
/// takes, and it only returns cached buffers to the OS. So a sampler tick missed inside the window
/// hides no peak, and the window interior does not count towards
/// [`AllocatorProbeReport::max_uncovered_gap_micros`]. Without a running probe this is exactly
/// `mlx_rs::memory::clear_cache`.
pub fn clear_cache() {
    observe_release(mlx_sample, mlx_rs::memory::clear_cache);
}

fn observe_release(sample: impl Fn() -> AllocatorSample, release: impl FnOnce()) {
    if lock(&LISTENERS).is_empty() {
        release();
        return;
    }
    let before = sample();
    let start_micros = now_micros();
    release();
    let end_micros = now_micros();
    let after = sample();
    let window = ReleaseWindow {
        start_micros,
        end_micros,
        before,
        after,
    };
    for (_, listener) in lock(&LISTENERS).iter() {
        let _ = listener.send(ProbeMessage::Release(window));
    }
}

/// Run the calling sampler thread under a Mach time-constraint policy.
///
/// The SC-20684 campaign children inherit the self-hosted runner's utility QoS clamp (scheduler
/// priority 20), under which macOS coalesces a 50 ms `recv_timeout` to ~150 ms mean and 200+ ms
/// gaps; neither `std::thread::sleep`, `mach_wait_until` nor USER_INTERACTIVE QoS escapes the clamp.
/// A time-constraint thread does (50.0 ms mean and max under `taskpolicy -c utility`), and the
/// kernel demotes it if it ever overruns its tiny computation budget.
#[cfg(target_os = "macos")]
fn request_realtime_sampling(interval: Duration) -> bool {
    // <mach/thread_policy.h>, <mach/mach_time.h>; declared here because libc deprecates its copies.
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    #[repr(C)]
    struct TimeConstraintPolicy {
        period: u32,
        computation: u32,
        constraint: u32,
        preemptible: u32,
    }
    const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;
    const THREAD_TIME_CONSTRAINT_POLICY_COUNT: u32 = 4;
    extern "C" {
        fn mach_timebase_info(info: *mut Timebase) -> i32;
        fn pthread_mach_thread_np(thread: libc::pthread_t) -> u32;
        fn thread_policy_set(thread: u32, flavor: u32, info: *mut i32, count: u32) -> i32;
    }
    let mut timebase = Timebase { numer: 0, denom: 0 };
    // SAFETY: writes the timebase through a valid pointer.
    if unsafe { mach_timebase_info(&mut timebase) } != 0 || timebase.numer == 0 {
        return false;
    }
    let ticks = |d: Duration| -> u32 {
        let nanos = u128::from(duration_micros(d)) * 1_000;
        u32::try_from(nanos * u128::from(timebase.denom) / u128::from(timebase.numer))
            .unwrap_or(u32::MAX)
    };
    let constraint = interval.min(Duration::from_millis(2));
    let mut policy = TimeConstraintPolicy {
        period: ticks(interval),
        computation: ticks(constraint / 2),
        constraint: ticks(constraint),
        preemptible: 1,
    };
    // SAFETY: sets this thread's own policy (pthread_mach_thread_np adds no port reference) from a
    // valid policy struct of the declared count.
    let status = unsafe {
        thread_policy_set(
            pthread_mach_thread_np(libc::pthread_self()),
            THREAD_TIME_CONSTRAINT_POLICY,
            std::ptr::addr_of_mut!(policy).cast(),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        )
    };
    status == 0
}

#[cfg(not(target_os = "macos"))]
fn request_realtime_sampling(_interval: Duration) -> bool {
    false
}

/// The probe thread's view of one interval between ticks.
struct GapTracker {
    first_micros: u64,
    interval_micros: u64,
    last_tick_micros: u64,
    phase_at_last_tick: Option<String>,
    windows: Vec<(u64, u64)>,
}

impl GapTracker {
    /// Account the interval ending at a tick at `now_micros`.
    fn tick(&mut self, report: &mut AllocatorProbeReport, now_micros: u64, phase: Option<String>) {
        let start = self.last_tick_micros;
        let raw = now_micros.saturating_sub(start);
        let mut windows: Vec<(u64, u64)> = self
            .windows
            .drain(..)
            .map(|(s, e)| (s.max(start), e.min(now_micros)))
            .filter(|(s, e)| s < e)
            .collect();
        windows.sort_unstable();
        let (mut cursor, mut uncovered, mut covered) = (start, 0u64, 0u64);
        for (s, e) in &windows {
            if *s > cursor {
                uncovered = uncovered.max(s - cursor);
            }
            if *e > cursor {
                covered += e - cursor.max(*s);
                cursor = *e;
            }
        }
        uncovered = uncovered.max(now_micros.saturating_sub(cursor));
        report.max_gap_micros = report.max_gap_micros.max(raw);
        report.max_uncovered_gap_micros = report.max_uncovered_gap_micros.max(uncovered);
        if raw >= GAP_RECORD_INTERVALS.saturating_mul(self.interval_micros) {
            if report.gaps.len() < MAX_GAP_RECORDS {
                report.gaps.push(SamplerGap {
                    start_micros: start.saturating_sub(self.first_micros),
                    duration_micros: raw,
                    uncovered_micros: uncovered,
                    release_micros: covered,
                    release_windows: windows.len() as u64,
                    phase_at_start: self.phase_at_last_tick.take(),
                    phase_at_end: phase.clone(),
                });
            } else {
                report.gaps_not_recorded += 1;
            }
        }
        report.sampling_span_micros = now_micros.saturating_sub(self.first_micros);
        self.last_tick_micros = now_micros;
        self.phase_at_last_tick = phase;
    }
}

/// Background paired active/cache sampler used by phase-aware performance evidence.
pub struct AllocatorProbe {
    stop: Option<Sender<ProbeMessage>>,
    listener: Option<u64>,
    handle: Option<std::thread::JoinHandle<AllocatorProbeReport>>,
}

impl AllocatorProbe {
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    pub fn start(interval: Duration) -> Self {
        Self::start_with_sampler(interval, mlx_sample)
    }

    fn start_with_sampler<F>(interval: Duration, sample: F) -> Self
    where
        F: Fn() -> AllocatorSample + Send + 'static,
    {
        let (stop, rx) = channel::<ProbeMessage>();
        let (ready_tx, ready_rx) = sync_channel::<()>(0);
        let handle = std::thread::spawn(move || {
            let realtime_sampler = request_realtime_sampling(interval);
            let first_micros = now_micros();
            let first = sample();
            let mut report = AllocatorProbeReport {
                interval_micros: duration_micros(interval),
                sample_count: 1,
                periodic_sample_count: 0,
                sampling_span_micros: 0,
                max_gap_micros: 0,
                max_uncovered_gap_micros: 0,
                gaps: Vec::new(),
                gaps_not_recorded: 0,
                release_window_count: 0,
                realtime_sampler,
                sampled_active_peak_bytes: first.active_bytes,
                sampled_cache_peak_bytes: first.cache_bytes,
                sampled_footprint_peak_bytes: first.footprint_bytes(),
                footprint_peak_active_bytes: first.active_bytes,
                footprint_peak_cache_bytes: first.cache_bytes,
                boundary_active_bytes: first.active_bytes,
                boundary_cache_bytes: first.cache_bytes,
            };
            let mut tracker = GapTracker {
                first_micros,
                interval_micros: duration_micros(interval),
                last_tick_micros: first_micros,
                phase_at_last_tick: current_phase(),
                windows: Vec::new(),
            };
            ready_tx
                .send(())
                .expect("allocator probe starter dropped before the first sample");
            let mut deadline = Instant::now() + interval;
            loop {
                let message = rx.recv_timeout(deadline.saturating_duration_since(Instant::now()));
                let periodic = match message {
                    Ok(ProbeMessage::Release(window)) => {
                        report.fold(window.before);
                        report.fold(window.after);
                        report.release_window_count += 1;
                        tracker
                            .windows
                            .push((window.start_micros, window.end_micros));
                        continue;
                    }
                    Err(RecvTimeoutError::Timeout) => true,
                    Ok(ProbeMessage::Stop) | Err(RecvTimeoutError::Disconnected) => false,
                };
                let now = now_micros();
                let sampled = sample();
                report.sample_count += 1;
                if periodic {
                    report.periodic_sample_count += 1;
                }
                report.fold(sampled);
                report.boundary_active_bytes = sampled.active_bytes;
                report.boundary_cache_bytes = sampled.cache_bytes;
                tracker.tick(&mut report, now, current_phase());
                if !periodic {
                    return report;
                }
                deadline = Instant::now() + interval;
            }
        });
        ready_rx
            .recv()
            .expect("allocator probe sampler panicked before the first sample");
        let id = NEXT_LISTENER.fetch_add(1, Ordering::Relaxed);
        lock(&LISTENERS).push((id, stop.clone()));
        Self {
            stop: Some(stop),
            listener: Some(id),
            handle: Some(handle),
        }
    }

    pub fn finish(mut self) -> AllocatorProbeReport {
        self.stop_and_join()
            .expect("allocator probe sampler panicked")
    }

    fn stop_and_join(&mut self) -> std::thread::Result<AllocatorProbeReport> {
        if let Some(id) = self.listener.take() {
            lock(&LISTENERS).retain(|(listener, _)| *listener != id);
        }
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(ProbeMessage::Stop);
        }
        self.handle
            .take()
            .expect("an active allocator probe owns one sampler")
            .join()
    }
}

impl Drop for AllocatorProbe {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.stop_and_join();
        }
    }
}

impl FootprintProbe {
    /// Start sampling at [`DEFAULT_INTERVAL`].
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    /// Start sampling at `interval`.
    ///
    /// The first sample is taken immediately, so a probe started and finished around a synchronous
    /// block still observes at least one value.
    pub fn start(interval: Duration) -> Self {
        Self::start_with_sampler(interval, || {
            (mlx_rs::memory::get_active_memory() as u64)
                .saturating_add(mlx_rs::memory::get_cache_memory() as u64)
        })
    }

    fn start_with_sampler<F>(interval: Duration, sample: F) -> Self
    where
        F: Fn() -> u64 + Send + 'static,
    {
        let (stop, rx) = channel::<()>();
        let (ready_tx, ready_rx) = sync_channel::<()>(0);
        let peak_bytes = Arc::new(AtomicU64::new(0));
        let handle = {
            let peak = Arc::clone(&peak_bytes);
            std::thread::spawn(move || {
                request_realtime_sampling(interval);
                peak.fetch_max(sample(), Ordering::Relaxed);
                ready_tx
                    .send(())
                    .expect("footprint probe starter dropped before the first sample");
                loop {
                    match rx.recv_timeout(interval) {
                        Err(RecvTimeoutError::Timeout) => {}
                        // Disconnected (or signalled): take one FINAL sample after the stop rather than
                        // returning on it. A probe stopped immediately after the allocation it exists to
                        // catch would otherwise miss the very thing it was started for.
                        _ => {
                            peak.fetch_max(sample(), Ordering::Relaxed);
                            return;
                        }
                    }
                    peak.fetch_max(sample(), Ordering::Relaxed);
                }
            })
        };
        ready_rx
            .recv()
            .expect("footprint probe sampler panicked before the first sample");
        Self {
            stop: Some(stop),
            peak_bytes,
            handle: Some(handle),
        }
    }

    /// The peak `active + cache` observed so far, in bytes. Safe to call while sampling.
    pub fn peak_footprint_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Stop sampling and return the peak `active + cache` in bytes.
    pub fn finish(mut self) -> u64 {
        self.stop_and_join()
            .expect("footprint probe sampler panicked");
        self.peak_bytes.load(Ordering::Relaxed)
    }

    fn stop_and_join(&mut self) -> std::thread::Result<()> {
        drop(self.stop.take());
        if let Some(handle) = self.handle.take() {
            handle.join()?;
        }
        Ok(())
    }
}

impl Drop for FootprintProbe {
    /// Joins the sampler even when `finish` was never reached, so an error path cannot leave a
    /// thread polling the allocator for the rest of the process.
    fn drop(&mut self) {
        // `finish` propagates sampler panics. Drop must still join on early-return paths, but cannot
        // safely introduce a second panic while unwinding an unrelated error.
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Release windows are broadcast process-wide; probe tests that assert exact receipts must not
    /// see another test's windows.
    static PROBE_TESTS: Mutex<()> = Mutex::new(());

    fn tracker(interval_micros: u64) -> (GapTracker, AllocatorProbeReport) {
        let report = AllocatorProbeReport {
            interval_micros,
            sample_count: 1,
            periodic_sample_count: 0,
            sampling_span_micros: 0,
            max_gap_micros: 0,
            max_uncovered_gap_micros: 0,
            gaps: Vec::new(),
            gaps_not_recorded: 0,
            release_window_count: 0,
            realtime_sampler: false,
            sampled_active_peak_bytes: 0,
            sampled_cache_peak_bytes: 0,
            sampled_footprint_peak_bytes: 0,
            footprint_peak_active_bytes: 0,
            footprint_peak_cache_bytes: 0,
            boundary_active_bytes: 0,
            boundary_cache_bytes: 0,
        };
        let tracker = GapTracker {
            first_micros: 1_000,
            interval_micros,
            last_tick_micros: 1_000,
            phase_at_last_tick: Some("packed-decode".into()),
            windows: Vec::new(),
        };
        (tracker, report)
    }

    #[test]
    fn release_windows_cover_a_gap_only_where_they_lie() {
        // A 1 s gap whose middle 800 ms is one clear_cache window: 100 ms uncovered each side.
        let (mut gaps, mut report) = tracker(50);
        gaps.windows.push((1_100, 1_900));
        gaps.tick(&mut report, 2_000, Some("dense-parity-entry".into()));
        assert_eq!(
            (report.max_gap_micros, report.max_uncovered_gap_micros),
            (1_000, 100)
        );
        assert_eq!(
            report.gaps,
            vec![SamplerGap {
                start_micros: 0,
                duration_micros: 1_000,
                uncovered_micros: 100,
                release_micros: 800,
                release_windows: 1,
                phase_at_start: Some("packed-decode".into()),
                phase_at_end: Some("dense-parity-entry".into()),
            }]
        );

        // Overlapping windows merge; a window reaching past either tick is clipped to the gap.
        let (mut gaps, mut report) = tracker(50);
        gaps.windows
            .extend([(1_500, 2_200), (900, 1_300), (1_200, 1_400)]);
        gaps.tick(&mut report, 2_000, None);
        assert_eq!(report.max_uncovered_gap_micros, 100);
        assert_eq!(report.gaps[0].release_micros, 900);
        assert_eq!(report.gaps[0].release_windows, 3);

        // The same stall with no release in flight is uncovered end to end.
        let (mut gaps, mut report) = tracker(50);
        gaps.tick(&mut report, 2_000, None);
        assert_eq!(
            (report.max_gap_micros, report.max_uncovered_gap_micros),
            (1_000, 1_000)
        );
        assert_eq!(report.gaps[0].release_windows, 0);

        // Short gaps are measured but not recorded.
        let (mut gaps, mut report) = tracker(50);
        gaps.tick(&mut report, 1_199, None);
        assert_eq!(report.max_gap_micros, 199);
        assert!(report.gaps.is_empty());
    }

    #[test]
    fn gap_records_are_capped_and_counted() {
        let (mut gaps, mut report) = tracker(1);
        for tick in 1..=(MAX_GAP_RECORDS as u64 + 3) {
            gaps.tick(&mut report, 1_000 + tick * 10, None);
        }
        assert_eq!(report.gaps.len(), MAX_GAP_RECORDS);
        assert_eq!(report.gaps_not_recorded, 3);
    }

    /// End to end through the probe thread: a sampler stalled for the whole of a release window
    /// leaves a long raw gap that the window covers, while the same stall outside a release is an
    /// uncovered gap. The window's entry sample is folded into the peak.
    #[test]
    fn a_stall_inside_clear_cache_is_covered_and_one_outside_is_not() {
        let _serial = lock(&PROBE_TESTS);
        for in_release in [true, false] {
            let gate = Arc::new(Mutex::new(()));
            let probe_gate = Arc::clone(&gate);
            let probe = AllocatorProbe::start_with_sampler(Duration::from_millis(5), move || {
                drop(lock(&probe_gate)); // stalls while the gate is held
                AllocatorSample {
                    active_bytes: 10,
                    cache_bytes: 0,
                }
            });
            set_phase("synthetic-decode");
            std::thread::sleep(Duration::from_millis(20));
            let stall = || {
                let _held = lock(&gate);
                std::thread::sleep(Duration::from_millis(150));
            };
            if in_release {
                // Entry sample: 90 cached bytes about to be released; exit sample: none left.
                let calls = AtomicU64::new(0);
                observe_release(
                    || AllocatorSample {
                        active_bytes: 10,
                        cache_bytes: if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                            90
                        } else {
                            0
                        },
                    },
                    stall,
                );
            } else {
                stall();
            }
            std::thread::sleep(Duration::from_millis(20));
            let report = probe.finish();
            assert!(report.max_gap_micros >= 140_000, "{report:?}");
            let gap = report
                .gaps
                .iter()
                .max_by_key(|gap| gap.duration_micros)
                .expect("the stall is recorded");
            assert_eq!(gap.phase_at_end.as_deref(), Some("synthetic-decode"));
            if in_release {
                assert_eq!(report.release_window_count, 1);
                assert_eq!(gap.release_windows, 1);
                assert!(gap.release_micros >= 140_000, "{gap:?}");
                assert!(report.max_uncovered_gap_micros < 50_000, "{report:?}");
                assert_eq!(report.sampled_footprint_peak_bytes, 100);
            } else {
                assert_eq!(report.release_window_count, 0);
                assert_eq!(gap.release_windows, 0);
                assert!(report.max_uncovered_gap_micros >= 140_000, "{report:?}");
                assert_eq!(report.sampled_footprint_peak_bytes, 10);
            }
        }
    }

    #[test]
    fn clear_cache_without_a_running_probe_only_releases() {
        let _serial = lock(&PROBE_TESTS);
        let sampled = AtomicU64::new(0);
        let released = AtomicU64::new(0);
        observe_release(
            || {
                sampled.fetch_add(1, Ordering::Relaxed);
                AllocatorSample {
                    active_bytes: 0,
                    cache_bytes: 0,
                }
            },
            || {
                released.fetch_add(1, Ordering::Relaxed);
            },
        );
        assert_eq!(
            (
                sampled.load(Ordering::Relaxed),
                released.load(Ordering::Relaxed)
            ),
            (0, 1)
        );
    }

    /// The probe must observe a transient that exists only between caller-visible boundaries.
    #[test]
    fn observes_a_controlled_transient_that_is_gone_before_finish() {
        let current = Arc::new(AtomicU64::new(7));
        let sample = Arc::clone(&current);
        let (observed_tx, observed_rx) = sync_channel::<()>(1);
        let probe = FootprintProbe::start_with_sampler(Duration::from_millis(2), move || {
            let value = sample.load(Ordering::Relaxed);
            if value > 7 {
                let _ = observed_tx.try_send(());
            }
            value
        });
        current.store(64 * 1024 * 1024 + 7, Ordering::Relaxed);
        observed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("sampler did not acknowledge the controlled transient");
        current.store(7, Ordering::Relaxed);
        let peak = probe.finish();
        assert_eq!(
            peak,
            64 * 1024 * 1024 + 7,
            "probe missed or distorted the controlled transient"
        );
    }

    /// `finish` must take a final sample at stop time, and must not wait out the interval to do it.
    ///
    /// The hour-long interval is the point. It pins both halves at once: a probe that only sampled
    /// on its own schedule would report whatever it saw at construction and miss the allocation
    /// entirely, and a probe that checked a stop flag *between* sleeps would take an hour to join.
    /// The first version of this module did the latter, and this test hung rather than failing.
    #[test]
    fn final_sample_is_taken_at_stop_without_waiting_out_the_interval() {
        let started = std::time::Instant::now();
        let current = Arc::new(AtomicU64::new(11));
        let sample = Arc::clone(&current);
        let probe = FootprintProbe::start_with_sampler(Duration::from_secs(3600), move || {
            sample.load(Ordering::Relaxed)
        });
        current.store(64 * 1024 * 1024 + 11, Ordering::Relaxed);
        let peak = probe.finish();
        assert_eq!(peak, 64 * 1024 * 1024 + 11, "final sample was not taken");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "finish() waited on the sampling interval instead of interrupting it"
        );
    }

    #[test]
    fn finish_propagates_sampler_panics() {
        let calls = Arc::new(AtomicU64::new(0));
        let sampler_calls = Arc::clone(&calls);
        let probe = FootprintProbe::start_with_sampler(Duration::from_secs(3600), move || {
            if sampler_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                1
            } else {
                panic!("controlled sampler failure")
            }
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe.finish()));
        assert!(result.is_err(), "finish hid a sampler thread panic");
    }

    #[test]
    fn paired_probe_never_sums_noncoincident_component_peaks() {
        let _serial = lock(&PROBE_TESTS);
        let calls = Arc::new(AtomicU64::new(0));
        let sampler_calls = Arc::clone(&calls);
        let probe = AllocatorProbe::start_with_sampler(Duration::from_millis(2), move || {
            if sampler_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                AllocatorSample {
                    active_bytes: 100,
                    cache_bytes: 0,
                }
            } else {
                AllocatorSample {
                    active_bytes: 0,
                    cache_bytes: 90,
                }
            }
        });
        while calls.load(Ordering::Relaxed) < 2 {
            std::thread::yield_now();
        }
        let report = probe.finish();

        assert_eq!(report.sampled_active_peak_bytes, 100);
        assert_eq!(report.sampled_cache_peak_bytes, 90);
        assert_eq!(report.sampled_footprint_peak_bytes, 100);
        assert_eq!(report.footprint_peak_active_bytes, 100);
        assert_eq!(report.footprint_peak_cache_bytes, 0);
        assert_ne!(report.sampled_footprint_peak_bytes, 190);
        assert_eq!(
            report.sample_count,
            report.periodic_sample_count + 2,
            "receipt must account for immediate, periodic, and final samples"
        );
    }

    /// Real-GPU lock evidence (W2 run 36821253163): a sampler reading MLX's active/cache counters
    /// never waits on `clear_cache`, even while it holds the allocator mutex releasing gigabytes;
    /// a control reader that takes the mutex (`reset_peak_memory`) waits out the whole clear.
    /// Fills the cache with `MEMORY_PROBE_EVIDENCE_GIB` (default 8) GiB of 32 MiB buffers.
    #[test]
    #[ignore = "allocates GiB of Metal buffers; run by hand on an idle Mac"]
    fn counter_reads_never_wait_on_clear_cache() {
        use std::sync::atomic::AtomicBool;
        let gib: usize = std::env::var("MEMORY_PROBE_EVIDENCE_GIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        for locking in [false, true] {
            mlx_rs::memory::clear_cache();
            let arrays: Vec<mlx_rs::Array> = (0..gib * 32)
                .map(|_| {
                    let a = mlx_rs::ops::ones::<f32>(&[8 * 1024 * 1024]).unwrap();
                    a.eval().unwrap();
                    a
                })
                .collect();
            drop(arrays);
            let cached = mlx_rs::memory::get_cache_memory();
            let stop = Arc::new(AtomicBool::new(false));
            let reader_stop = Arc::clone(&stop);
            let (ready_tx, ready_rx) = sync_channel::<()>(0);
            let reader = std::thread::spawn(move || {
                let mut max_read = Duration::ZERO;
                ready_tx.send(()).unwrap();
                while !reader_stop.load(Ordering::Relaxed) {
                    let before = Instant::now();
                    if locking {
                        mlx_rs::memory::reset_peak_memory();
                    } else {
                        std::hint::black_box(mlx_sample());
                    }
                    max_read = max_read.max(before.elapsed());
                }
                max_read
            });
            ready_rx.recv().unwrap();
            std::thread::sleep(Duration::from_millis(20));
            let started = Instant::now();
            mlx_rs::memory::clear_cache();
            let clear = started.elapsed();
            std::thread::sleep(Duration::from_millis(20));
            stop.store(true, Ordering::Relaxed);
            let max_read = reader.join().unwrap();
            eprintln!(
                "clear_cache of {:.2} GiB took {:.3} ms; {} reader's longest read {:.3} ms",
                cached as f64 / (1u64 << 30) as f64,
                clear.as_secs_f64() * 1e3,
                if locking { "mutex-taking" } else { "counter" },
                max_read.as_secs_f64() * 1e3,
            );
            if locking {
                assert!(
                    max_read * 2 >= clear,
                    "control reader never met the allocator mutex"
                );
            } else {
                assert!(max_read * 10 < clear, "counter reads waited on clear_cache");
            }
        }
    }

    /// Real-GPU cadence evidence: run as `taskpolicy -c utility <test binary> --ignored
    /// sampler_cadence` to reproduce the self-hosted runner's QoS clamp. Plain waits coalesce to
    /// ~150 ms there; the probe's time-constraint sampler holds 50 ms.
    #[test]
    #[ignore = "drives the GPU for a few seconds; run by hand under taskpolicy -c utility"]
    fn sampler_cadence_under_gpu_load() {
        for realtime in [false, true] {
            let (stop_tx, stop_rx) = channel::<()>();
            let sampler = std::thread::spawn(move || {
                let granted = realtime && request_realtime_sampling(DEFAULT_INTERVAL);
                let mut ticks = vec![Instant::now()];
                while let Err(RecvTimeoutError::Timeout) = stop_rx.recv_timeout(DEFAULT_INTERVAL) {
                    std::hint::black_box(mlx_sample());
                    ticks.push(Instant::now());
                }
                (granted, ticks)
            });
            let until = Instant::now() + Duration::from_secs(3);
            let a = mlx_rs::ops::ones::<f32>(&[4096, 4096]).unwrap();
            while Instant::now() < until {
                let mut x = a.clone();
                for _ in 0..8 {
                    x = mlx_rs::ops::matmul(&x, &a).unwrap();
                }
                x.eval().unwrap();
            }
            drop(stop_tx);
            let (granted, ticks) = sampler.join().unwrap();
            let gaps: Vec<f64> = ticks
                .windows(2)
                .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1e3)
                .collect();
            eprintln!(
                "realtime requested={realtime} granted={granted}: {} ticks, mean {:.1} ms, max {:.1} ms",
                gaps.len(),
                gaps.iter().sum::<f64>() / gaps.len().max(1) as f64,
                gaps.iter().cloned().fold(0.0, f64::max),
            );
        }
    }

    #[test]
    fn paired_probe_records_interruptible_final_boundary_sample() {
        let _serial = lock(&PROBE_TESTS);
        let current = Arc::new(AtomicU64::new(11));
        let sample = Arc::clone(&current);
        let probe = AllocatorProbe::start_with_sampler(Duration::from_secs(3600), move || {
            AllocatorSample {
                active_bytes: sample.load(Ordering::Relaxed),
                cache_bytes: 7,
            }
        });
        current.store(99, Ordering::Relaxed);
        let report = probe.finish();

        assert_eq!(report.sample_count, 2);
        assert_eq!(report.periodic_sample_count, 0);
        assert_eq!(report.boundary_active_bytes, 99);
        assert_eq!(report.boundary_cache_bytes, 7);
        assert_eq!(report.sampled_footprint_peak_bytes, 106);
        assert_eq!(report.footprint_peak_active_bytes, 99);
        assert_eq!(report.footprint_peak_cache_bytes, 7);
    }
}
