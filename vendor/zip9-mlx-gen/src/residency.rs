//! MLX adapter for gen-core's backend-neutral request-scoped residency state machine.
//!
//! Ownership, warm-cache transitions, phase order, cancellation, progress, and `StagedHeavy` live in
//! `gen_core::residency`. MLX contributes only its allocator-cache release hook and tensor-specific
//! materialization closures at call sites.

use std::time::Duration;

use crate::Error;

/// MLX release behavior for the shared residency driver.
pub struct MlxResidencyRuntime;

impl gen_core::ResidencyRuntime for MlxResidencyRuntime {
    type Error = Error;

    fn after_component_drop() {
        drain_allocator_cache();
    }
}

/// The one shared residency implementation, specialized only by MLX cleanup behavior.
pub type Residency<Text, Heavy> = gen_core::Residency<Text, Heavy, MlxResidencyRuntime>;

pub use gen_core::StagedHeavy;

/// Warn when staged execution over a dense quantized snapshot would re-quantize every request.
pub fn warn_sequential_requantize(model_id: &str, bits: i32) {
    eprintln!(
        "{model_id}: staged residency with Q{bits} over a dense snapshot re-quantizes the whole \
         model on EVERY generate (repeated compute; the dense transient makes the per-phase peak the \
         dense component size, not the packed one — shrinking the memory win). Point at a \
         pre-quantized Q{bits} snapshot to avoid both."
    );
}

/// Drain passes [`drain_allocator_cache`] makes before giving up. 8 × [`DRAIN_BACKOFF`] bounds the
/// whole drain at ~16 ms, against phases measured in minutes.
const DRAIN_ATTEMPTS: usize = 8;

/// Time given back to the Metal backend between drain passes, so a command buffer that has not
/// retired yet gets a chance to before the next sweep.
const DRAIN_BACKOFF: Duration = Duration::from_millis(2);

/// The repeat-while-active-falls loop, with both runtime probes injected.
///
/// Injected so the loop's *control flow* — how many sweeps it makes, and what stops it — is
/// testable without Metal; `drain_allocator_cache` is the only production caller and binds them to
/// MLX. Returns the number of `clear` passes made, which is what the CPU-side tests assert on.
fn drain_while_active_falls(mut clear: impl FnMut(), mut active: impl FnMut() -> usize) -> usize {
    let mut previous = usize::MAX;
    for pass in 1..=DRAIN_ATTEMPTS {
        clear();
        let now = active();
        if now >= previous {
            return pass;
        }
        previous = now;
        std::thread::sleep(DRAIN_BACKOFF);
    }
    clear();
    DRAIN_ATTEMPTS + 1
}

/// Return a shed component's buffers to the system allocator, retrying while MLX keeps handing
/// them back. **The drain called by [`MlxResidencyRuntime`]'s `after_component_drop` and by both
/// MiniMax-H3 render-path drain sites** — its `model::release` helper (which its eight phase
/// boundaries all shed through) and the AdaLN eviction in `dit::adaln` (sc-17151).
///
/// # Scope, because the wider reading is false and was written here once
///
/// This is *not* "the one drain every MLX component-release site calls". Counted on this commit,
/// **72 production `clear_cache()` call sites remain across 18 `mlx-gen` crates** — `mlx-gen-wan`
/// (12), `mlx-gen-lens` (7), `mlx-gen-z-image` and `mlx-gen-ltx` (6 each), `mlx-gen-sdxl` and
/// `mlx-gen-mage` (5 each), and so on down — plus `mlx-gen`'s own
/// [`crate::request_scope`] `MlxScopeCleanup::Device`, which is a *shared* seam that still sheds
/// through one sweep behind an eval barrier. Every one of them is a single sweep with the
/// straggler exposure described below.
///
/// Widening the drain to them changes every other family's release behavior and belongs in its own
/// reviewed change, not in this story. MiniMax-H3's `convert.rs` is deliberately excluded too: its
/// two sweeps bound an offline shard-at-a-time conversion, not a render component handoff.
///
/// # One [`mlx_rs::memory::clear_cache`] is not reliably enough
///
/// A buffer reaches MLX's allocator cache when its last reference drops, but the Metal command
/// buffer that used it retains its resources until it is retired, and that retirement is not
/// guaranteed to have happened by the time [`mlx_rs::transforms::eval`] returns. A single sweep can
/// therefore run *before* some of the weights have been handed back, and those land in the cache
/// with nothing left to sweep them.
///
/// Measured (sc-17145) on a synthetic 8-block stack under concurrent test load: a single sweep left
/// one block's 18 MiB projection still counted as **active** — its last real reference being an
/// unretired command buffer — in roughly one run in five, while the same measurement in isolation
/// released all 8 every time.
///
/// # Why the loop watches *active* and not the cache
///
/// Stopping as soon as the cache reads empty returns on the first pass — the already-freed buffers
/// sweep cleanly and the cache genuinely *is* empty — leaving the straggler to reach the cache a
/// moment later with nothing left to sweep it. That version was shipped first and did not fix the
/// flake. Stopping when active stops falling is the condition that actually describes "the runtime
/// has finished handing buffers back".
///
/// # It cannot mask a real leak
///
/// A buffer that is still referenced never reaches the cache at all, so no amount of draining
/// releases it and the loop exits after two passes having freed nothing.
/// `mlx-gen-minimax-h3/tests/adaln_evict_memory.rs`'s control arm is the proof — with the forced
/// evaluation removed the drain frees 0.0 of 144.3 MiB, however many times it runs.
///
/// Returns the number of sweeps performed.
pub fn drain_allocator_cache() -> usize {
    #[cfg(test)]
    stub::begin_release();
    drain_while_active_falls(sweep_once, active_bytes)
}

#[cfg(test)]
fn sweep_once() {
    stub::clear();
}

#[cfg(not(test))]
fn sweep_once() {
    mlx_rs::memory::clear_cache();
}

#[cfg(test)]
fn active_bytes() -> usize {
    stub::active()
}

#[cfg(not(test))]
fn active_bytes() -> usize {
    mlx_rs::memory::get_active_memory()
}

/// A scripted stand-in for MLX's allocator, so `cargo test -p mlx-gen` exercises the real
/// [`drain_allocator_cache`] wiring — release points *and* sweep count — with no Metal device.
///
/// The script is the shape a straggler produces: active falls on the first [`stub::FALLS`] sweeps
/// of a release and then plateaus, so the loop must make `FALLS + 1` sweeps to observe the plateau.
/// A single-sweep drain makes exactly one, which is what the mutation check turns on.
///
/// # It is process-global, so its tests must not run concurrently
///
/// `ACTIVE` and `SWEEPS_THIS_RELEASE` are not counters the tests merely read — they are what the
/// retry loop's exit condition reads, so a second test entering [`begin_release`] mid-loop resets
/// `ACTIVE` upward and changes how many sweeps the first one observes. That is a red rather than a
/// false green, but a guard on the shared ladder seam that goes red at random gets muted, so every
/// stub-backed test takes [`LOCK`] for its whole body. `cargo test` runs a binary's tests on
/// several threads by default and nothing in this crate passes `--test-threads=1`.
#[cfg(test)]
mod stub {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Serializes every test that drives the stub. Poisoning is recovered from rather than
    /// propagated: one panicking test must not turn the rest into confusing secondary failures.
    pub static LOCK: Mutex<()> = Mutex::new(());

    /// Sweeps for which the scripted allocator keeps releasing before it plateaus.
    pub const FALLS: usize = 3;
    const STEP: usize = 16 << 20;

    /// Component-release points — how many times [`super::drain_allocator_cache`] was entered.
    pub static RELEASE_POINTS: AtomicUsize = AtomicUsize::new(0);
    /// Individual sweeps the drain performed across all release points.
    pub static SWEEPS: AtomicUsize = AtomicUsize::new(0);

    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static SWEEPS_THIS_RELEASE: AtomicUsize = AtomicUsize::new(0);

    pub fn begin_release() {
        RELEASE_POINTS.fetch_add(1, Ordering::SeqCst);
        SWEEPS_THIS_RELEASE.store(0, Ordering::SeqCst);
        ACTIVE.store(STEP * FALLS, Ordering::SeqCst);
    }

    pub fn clear() {
        SWEEPS.fetch_add(1, Ordering::SeqCst);
        if SWEEPS_THIS_RELEASE.fetch_add(1, Ordering::SeqCst) < FALLS {
            ACTIVE.fetch_sub(STEP, Ordering::SeqCst);
        }
    }

    pub fn active() -> usize {
        ACTIVE.load(Ordering::SeqCst)
    }

    pub fn reset() {
        RELEASE_POINTS.store(0, Ordering::SeqCst);
        SWEEPS.store(0, Ordering::SeqCst);
    }

    pub fn release_points() -> usize {
        RELEASE_POINTS.load(Ordering::SeqCst)
    }

    pub fn sweeps() -> usize {
        SWEEPS.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancelFlag, Result};

    /// Take the stub lock **and** zero the counters, in that order.
    ///
    /// The guard is returned rather than dropped here, so the caller holds it for the rest of the
    /// test body. It must be bound to a NAME (`let _stub = …`); `let _ = …` drops it immediately and
    /// restores the race this exists to close, which is why the `#[must_use]` below names that.
    #[must_use = "the returned guard is what serializes the test; dropping it here reopens the race"]
    fn reset_counters() -> std::sync::MutexGuard<'static, ()> {
        let guard = stub::LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stub::reset();
        guard
    }

    /// Sweeps the scripted allocator makes the retry loop perform for one release point: it falls
    /// on the first [`stub::FALLS`] and then plateaus, and the plateau costs one more sweep to
    /// observe. A single-sweep drain would make exactly 1.
    const SWEEPS_PER_RELEASE: usize = stub::FALLS + 1;

    /// **The drain retries — asserted on the sweep count, not on the doc comment.**
    ///
    /// The scripted allocator keeps releasing for three sweeps. A drain that called
    /// `clear_cache()` once — the shape `after_component_drop` and `model.rs`'s `release` both had
    /// before sc-17151 — makes one sweep and leaves the last two steps unswept. This is the
    /// assertion that goes red on that shape.
    #[test]
    fn a_component_release_sweeps_until_active_stops_falling() {
        let _stub = reset_counters();
        let sweeps = drain_allocator_cache();
        assert_eq!(
            sweeps,
            SWEEPS_PER_RELEASE,
            "the drain made {sweeps} sweeps against a scripted allocator that keeps releasing for \
             {} — a single-pass drain makes 1 and returns with buffers still to come back",
            stub::FALLS
        );
        assert!(
            sweeps > 1,
            "the drain did not retry at all; this is the single-pass shape sc-17145 measured as \
             leaving a straggler active in roughly 1 run in 5"
        );
        assert_eq!(stub::sweeps(), SWEEPS_PER_RELEASE);
        assert_eq!(stub::release_points(), 1);
    }

    /// **A plateau exits after two sweeps, so the drain cannot mask a leak.**
    ///
    /// A still-referenced buffer never reaches the cache, so active never moves; the loop must
    /// notice that on the second sweep rather than spinning to its bound.
    #[test]
    fn a_flat_allocator_exits_after_two_sweeps() {
        let mut sweeps = 0usize;
        let passes = drain_while_active_falls(
            || sweeps += 1,
            || 144 << 20, // never moves: the leak case
        );
        assert_eq!(
            passes, 2,
            "a flat active reading must stop the loop at once"
        );
        assert_eq!(sweeps, 2);
    }

    /// **The drain is bounded** — an allocator that never stops releasing cannot spin it forever.
    #[test]
    fn a_never_settling_allocator_is_bounded_at_the_attempt_limit() {
        let mut sweeps = 0usize;
        let mut active = usize::MAX - 1;
        let passes = drain_while_active_falls(
            || sweeps += 1,
            || {
                active -= 1;
                active
            },
        );
        assert_eq!(passes, DRAIN_ATTEMPTS + 1);
        assert_eq!(
            sweeps,
            DRAIN_ATTEMPTS + 1,
            "the bounded loop makes one sweep per attempt plus the final unconditional one"
        );
    }

    #[test]
    fn staged_run_flushes_after_both_component_release_points() {
        let _stub = reset_counters();
        let residency = Residency::request_scoped(|_| Ok(2u8), |_, _| Ok(3u8));
        let output = residency
            .run_request_scoped(
                true,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text + 1),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
            .unwrap();
        assert_eq!(output, 6);
        assert_eq!(stub::release_points(), 2);
        // …and each of those release points ran the RETRY drain, not one sweep. This is the
        // shared-seam half of the evidence: `MlxResidencyRuntime::after_component_drop` is what
        // every family on the ladder sheds through.
        assert_eq!(stub::sweeps(), 2 * SWEEPS_PER_RELEASE);
    }

    #[test]
    fn warm_to_staged_flushes_eviction_text_and_heavy() {
        let _stub = reset_counters();
        let residency = Residency::request_scoped(|_| Ok(2u8), |_, _| Ok(3u8));
        let run = |stage| -> Result<u8> {
            residency.run_request_scoped(
                stage,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
        };
        assert_eq!(run(false).unwrap(), 5);
        assert_eq!(run(true).unwrap(), 5);
        assert_eq!(stub::release_points(), 3);
        assert_eq!(stub::sweeps(), 3 * SWEEPS_PER_RELEASE);
    }

    #[test]
    fn staged_error_still_flushes_the_loaded_phase() {
        let _stub = reset_counters();
        let residency = Residency::<u8, u8>::request_scoped(|_| Ok(2), |_, _| Ok(3));
        let output: Result<()> = residency.run_request_scoped(
            true,
            false,
            &CancelFlag::new(),
            false,
            &mut |_| {},
            |_| -> Result<u8> { Err(Error::Msg("encode failed".into())) },
            |_| Ok(()),
            |_, _, _| Ok(()),
        );
        assert!(matches!(output, Err(Error::Msg(ref message)) if message == "encode failed"));
        assert_eq!(stub::release_points(), 1);
        assert_eq!(stub::sweeps(), SWEEPS_PER_RELEASE);
    }
}
