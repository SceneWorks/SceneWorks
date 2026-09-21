//! Bounds MLX's free-buffer cache across one long LLM decode (sc-24029).
//!
//! Every resident-model text decode in this crate — `prompt_refine_jobs`, `catalog_semantic_jobs`,
//! `vector_jobs` (native StarVector) and `caption_jobs` (JoyCaption) — streams tokens through a
//! per-token callback that runs ON the generating thread. MLX's allocator state is per-thread here,
//! so that callback is the only correct place to release the cache; this module owns the single
//! policy those four call sites share, rather than each growing its own copy.

/// Generated tokens between MLX free-buffer-cache clears inside one decode.
///
/// Measured (sc-24029): the pinned `mlx-llm` KV cache CONCATENATES K/V per layer per token, so every
/// token allocates a strictly larger buffer and frees the previous, smaller one — MLX's freed-buffer
/// cache can never satisfy the next (larger) request from them, so it only grows and nothing on the
/// decode path ever cleared it. One `prompt_refine` decode held activeBytes flat at 16.8 GB while
/// cacheBytes reached 68 GB after 801 tokens (~92 MB/token at ~5.6k context), 90 GB resident, and
/// panicked the host.
const CLEAR_INTERVAL_TOKENS: u32 = 16;

/// Releases MLX's freed-buffer cache on the CALLING thread.
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

/// Token-counting bound a decode drives from its own stream callback.
///
/// Hold it across one `generate` call: feed [`Self::note_token`] from the callback, and the terminal
/// clear fires on drop — so it covers a clean finish, an error `?` and an early cancel return alike.
pub(crate) struct DecodeCacheBound<F: FnMut()> {
    clear: F,
    since_clear: u32,
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
        }
    }

    /// Records one generated token, clearing the cache every [`CLEAR_INTERVAL_TOKENS`].
    pub(crate) fn note_token(&mut self) {
        self.since_clear += 1;
        if self.since_clear >= CLEAR_INTERVAL_TOKENS {
            self.since_clear = 0;
            (self.clear)();
        }
    }
}

impl<F: FnMut()> Drop for DecodeCacheBound<F> {
    /// The terminal clear: the last partial interval's buffers are orphaned the moment the decode
    /// ends, so release them before the thread picks up the next job.
    fn drop(&mut self) {
        (self.clear)();
    }
}

#[cfg(test)]
mod tests {
    use super::{DecodeCacheBound, CLEAR_INTERVAL_TOKENS};
    use std::cell::Cell;

    #[test]
    fn bound_clears_once_per_interval_of_generated_tokens() {
        let clears = Cell::new(0_u32);
        let tokens = CLEAR_INTERVAL_TOKENS * 3;
        let observed = {
            let mut bound = DecodeCacheBound::new(|| clears.set(clears.get() + 1));
            let mut at_boundaries = Vec::new();
            for token in 1..=tokens {
                bound.note_token();
                at_boundaries.push((token, clears.get()));
            }
            at_boundaries
        };

        // Three interval clears, and each landed on exactly the interval boundary — not one token
        // early or late.
        for (token, clears_after) in observed {
            assert_eq!(
                clears_after,
                token / CLEAR_INTERVAL_TOKENS,
                "unexpected clear count after {token} generated tokens"
            );
        }
        // Plus the terminal clear the drop above fired.
        assert_eq!(clears.get(), 4);
    }

    #[test]
    fn bound_clears_once_when_a_short_decode_finishes() {
        let clears = Cell::new(0_u32);
        {
            let mut bound = DecodeCacheBound::new(|| clears.set(clears.get() + 1));
            for _ in 0..CLEAR_INTERVAL_TOKENS - 1 {
                bound.note_token();
            }
            assert_eq!(
                clears.get(),
                0,
                "a sub-interval decode must not clear mid-run"
            );
        }
        assert_eq!(clears.get(), 1, "the terminal clear must fire on drop");
    }
}
