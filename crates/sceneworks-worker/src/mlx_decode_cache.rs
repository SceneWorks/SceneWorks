//! Bounds MLX's free-buffer cache across one long LLM decode (sc-24029).
//!
//! Every resident-model text decode in this crate — `prompt_refine_jobs`, `catalog_semantic_jobs`,
//! `vector_jobs` (native StarVector) and `caption_jobs` (JoyCaption) — streams its progress through
//! a callback that runs ON the generating thread, interleaved with the decode loop. That callback is
//! not chosen because the allocator is thread-local — it is NOT: MLX's freed-buffer cache is
//! PROCESS-GLOBAL (see inference `crates/llm/mlx-llm/src/starvector_8b.rs`, whose `unload` declines
//! to clear it for exactly that reason). The callback is chosen because it is the only hook this
//! crate holds that runs *during* a `generate` call; every other seam we own runs before it starts or
//! after it returns, by which point the cache has already grown to its peak.
//!
//! Side effect of a process-global cache: a refine / StarVector / caption clear also discards buffers
//! that a CONCURRENT image render had cached for reuse, up to once per [`CLEAR_INTERVAL_EVENTS`]
//! streamed token events plus once at decode end. That render then re-allocates from the system
//! allocator instead of the cache — slower, but bounded, and the alternative measured below is an
//! unbounded cache that panics the host.
//!
//! Testability: [`clear_mlx_buffer_cache`] is compiled to an empty body under `cfg(test)`, so no
//! in-crate smoke ever exercises the REAL clear — the tests below only pin the counting policy
//! through an injected closure. Validation that the real `mlx_rs::memory::clear_cache()` bounds
//! `cacheBytes` comes from a production run on real weights, not from this crate's unit tests.

/// Streamed token events between MLX free-buffer-cache clears inside one decode.
///
/// Counts STREAM EVENTS, not generated tokens: the worker-side callbacks are not 1:1 with the
/// model's tokens (JoyCaption emits two `Progress::Step` events per item in total; other providers
/// may coalesce or emit non-token events), so this is an upper bound on work between clears and
/// never a token count.
///
/// Measured (sc-24029): the pinned `mlx-llm` KV cache CONCATENATES K/V per layer per token, so every
/// token allocates a strictly larger buffer and frees the previous, smaller one — MLX's freed-buffer
/// cache can never satisfy the next (larger) request from them, so it only grows and nothing on the
/// decode path ever cleared it. One `prompt_refine` decode held activeBytes flat at 16.8 GB while
/// cacheBytes reached 68 GB after 801 tokens (~92 MB/token at ~5.6k context), 90 GB resident, and
/// panicked the host.
const CLEAR_INTERVAL_EVENTS: u32 = 16;

/// Releases MLX's process-global freed-buffer cache.
///
/// `not(test)` mirrors the [`crate::generator_cache`] precedent: unit tests must never initialise
/// Metal, and every test of this module injects its own clear through [`DecodeCacheBound::new`].
#[cfg(all(target_os = "macos", not(test)))]
fn clear_mlx_buffer_cache() {
    mlx_rs::memory::clear_cache();
}

/// Off-Mac (and under test) there is no MLX allocator to bound, so the whole policy is inert and the
/// candle lane compiles unchanged.
#[cfg(not(all(target_os = "macos", not(test))))]
fn clear_mlx_buffer_cache() {}

/// Event-counting bound a decode drives from its own stream callback.
///
/// Hold it across one `generate` call: feed [`Self::note_event`] from the callback, and the terminal
/// clear fires on drop — so it covers a clean finish, an error `?` and an early cancel return alike.
pub(crate) struct DecodeCacheBound<F: FnMut()> {
    clear: F,
    /// Streamed events since the last clear. Zero *and* [`Self::saw_event`] means the previous
    /// `note_event` landed exactly on an interval boundary and already cleared.
    since_clear: u32,
    saw_event: bool,
}

impl DecodeCacheBound<fn()> {
    /// The production bound, clearing the real MLX cache on whichever thread drives the decode.
    pub(crate) fn mlx() -> Self {
        Self::new(clear_mlx_buffer_cache)
    }
}

impl<F: FnMut()> DecodeCacheBound<F> {
    /// Builds a bound over an injected clear — the seam the unit tests count through.
    pub(crate) fn new(clear: F) -> Self {
        Self {
            clear,
            since_clear: 0,
            saw_event: false,
        }
    }

    /// Records one streamed token event, clearing the cache every [`CLEAR_INTERVAL_EVENTS`].
    pub(crate) fn note_event(&mut self) {
        self.saw_event = true;
        self.since_clear += 1;
        if self.since_clear >= CLEAR_INTERVAL_EVENTS {
            self.since_clear = 0;
            (self.clear)();
        }
    }
}

impl<F: FnMut()> Drop for DecodeCacheBound<F> {
    /// The terminal clear: the last partial interval's buffers are orphaned the moment the decode
    /// ends, so release them before the thread picks up the next job.
    ///
    /// Skipped on an exact interval boundary — `note_event` just cleared and nothing has been freed
    /// since, so a second clear would only re-pay the cost (and re-punish a concurrent render). A
    /// decode that streamed NO event still clears: prefill allocated, and a cancel or error during
    /// vision-encode/prefill returns before token 1.
    fn drop(&mut self) {
        if self.since_clear > 0 || !self.saw_event {
            (self.clear)();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DecodeCacheBound, CLEAR_INTERVAL_EVENTS};
    use std::cell::Cell;

    #[test]
    fn bound_clears_once_per_interval_of_streamed_events() {
        let clears = Cell::new(0_u32);
        let events = CLEAR_INTERVAL_EVENTS * 3;
        let observed = {
            let mut bound = DecodeCacheBound::new(|| clears.set(clears.get() + 1));
            let mut at_boundaries = Vec::new();
            for event in 1..=events {
                bound.note_event();
                at_boundaries.push((event, clears.get()));
            }
            at_boundaries
        };

        // Three interval clears, and each landed on exactly the interval boundary — not one event
        // early or late.
        for (event, clears_after) in observed {
            assert_eq!(
                clears_after,
                event / CLEAR_INTERVAL_EVENTS,
                "unexpected clear count after {event} streamed token events"
            );
        }
        // The decode ended ON an interval boundary, so the drop adds NO fourth clear.
        assert_eq!(
            clears.get(),
            3,
            "the drop must not double-clear on an interval boundary"
        );
    }

    #[test]
    fn bound_clears_once_when_a_short_decode_finishes() {
        let clears = Cell::new(0_u32);
        {
            let mut bound = DecodeCacheBound::new(|| clears.set(clears.get() + 1));
            for _ in 0..CLEAR_INTERVAL_EVENTS - 1 {
                bound.note_event();
            }
            assert_eq!(
                clears.get(),
                0,
                "a sub-interval decode must not clear mid-run"
            );
        }
        assert_eq!(clears.get(), 1, "the terminal clear must fire on drop");
    }

    #[test]
    fn bound_clears_on_drop_when_the_decode_never_streamed_an_event() {
        // The JoyCaption prefill/cancel shape: `generate` allocated, but the worker saw no event.
        let clears = Cell::new(0_u32);
        drop(DecodeCacheBound::new(|| clears.set(clears.get() + 1)));
        assert_eq!(
            clears.get(),
            1,
            "a decode that streamed no event still allocated during prefill"
        );
    }

    #[test]
    fn bound_clears_on_drop_when_the_decode_closure_returns_err() {
        // The `?` path: the bound is dropped by the stack unwind of an early `return Err(..)`, with a
        // partial interval outstanding. The terminal clear must still fire — that is the only thing
        // standing between an aborted decode and a cache the next job inherits.
        fn decode<F: FnMut()>(bound: &mut DecodeCacheBound<F>) -> Result<(), &'static str> {
            for _ in 0..CLEAR_INTERVAL_EVENTS + 3 {
                bound.note_event();
            }
            Err("engine failure mid-decode")
        }

        let clears = Cell::new(0_u32);
        let outcome = {
            let mut bound = DecodeCacheBound::new(|| clears.set(clears.get() + 1));
            let outcome = decode(&mut bound);
            // One interval clear so far; the 3 events past the boundary are still outstanding.
            assert_eq!(clears.get(), 1, "only the interval clear has fired yet");
            outcome
        };

        assert!(outcome.is_err(), "the decode closure must have failed");
        assert_eq!(
            clears.get(),
            2,
            "the terminal clear must fire on the Err path too"
        );
    }
}
