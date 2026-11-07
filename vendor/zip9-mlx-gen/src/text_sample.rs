//! The media pipelines' text/LLM token draw — the knobs ([`SampleParams`], with the LTX enhancer
//! and Lens reasoner presets) over **mlx-llm's shared sampler**. There is no sampler math here: the
//! draw is [`mlx_llm::primitives::sample`] and the PRNG is [`mlx_llm::primitives::SplitMix64`]
//! (re-exported), one implementation for every MLX decode (epic sc-24432 E8).
//!
//! **The one named policy** ([`sample_token`]): a media pipeline draws on mlx-llm's *heap-order
//! host reference* ([`sample`](mlx_llm::primitives::sample) — candidates in index order, a
//! heap-selected nucleus, the categorical inverse-CDF walked in that order, a degenerate row taking
//! the argmax without consuming a draw), **not** the decode loops' index-order device-matched draw
//! ([`draw_token`](mlx_llm::primitives::sampler::draw_token)). The LTX-2.3 enhancer and the Lens
//! PromptReasoner reproduce their upstream `make_sampler` / `make_logits_processors` decode with a
//! seeded stream of their own, and mlx-llm keeps that reference draw for exactly these pipelines;
//! routing them through the device sampler would change every seeded rewrite with no upstream
//! reason.
//!
//! sc-24446 consolidation (behaviour, vs the crate-local copy this module used to carry): the
//! repetition penalty now applies **once per distinct id** in the window — the upstream
//! `make_repetition_penalty` gathers and scatters, so a repeated id was never meant to compound
//! (the local copy divided once per occurrence); and a custom top-k / top-p now selects with the
//! shared tie order (descending weight, ties to the lower index) and an f64 nucleus mass. Greedy
//! and pure-temperature draws (the Lens preset, the LTX uncensored preset) are unchanged.

use mlx_rs::Array;

pub use mlx_llm::primitives::SplitMix64;

use crate::{Error, Result};

/// Sampling parameters. `top_k <= 0` and `top_p >= 1.0` disable those filters. `repetition_penalty`
/// (`None` ⇒ off) divides the logit of each distinct token seen in the last `repetition_context`
/// positions once (multiplies when the logit is negative), matching the reference
/// `make_logits_processors`.
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: Option<f32>,
    pub repetition_context: usize,
}

impl SampleParams {
    /// Pure temperature sampling: no top-k / top-p narrowing, no repetition penalty. The neutral
    /// preset for an LLM decode that only wants `temperature` (e.g. the lens PromptReasoner's vendor
    /// default `temperature = 0.7`). `temperature <= 0` ⇒ greedy (argmax) in [`sample_token`].
    pub fn temperature(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: None,
            repetition_context: 0,
        }
    }

    /// The LTX censored `enhance_t2v` sampler: `make_sampler(temp, 1.0, top_k=-1)` +
    /// `make_logits_processors(None, repetition_penalty=1.3, repetition_context_size=20)`.
    pub fn censored(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: -1,
            top_p: 1.0,
            repetition_penalty: Some(1.3),
            repetition_context: 20,
        }
    }

    /// The LTX uncensored `enhance_with_model` sampler: `make_sampler(temp, 1.0, 0.0, 1, top_k=0)` —
    /// pure temperature sampling, no repetition penalty (identical to [`temperature`](Self::temperature)).
    pub fn uncensored(temperature: f32) -> Self {
        Self::temperature(temperature)
    }
}

impl SampleParams {
    /// These knobs in the shared sampler's vocabulary: `top_k <= 0` disables top-k, and a missing
    /// or non-positive repetition penalty disables the penalty (`1.0`).
    pub fn to_sampling(&self) -> mlx_llm::primitives::SamplingParams {
        mlx_llm::primitives::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: usize::try_from(self.top_k).unwrap_or(0),
            presence_penalty: 0.0,
            repetition_penalty: self
                .repetition_penalty
                .filter(|&penalty| penalty > 0.0)
                .unwrap_or(1.0),
            repetition_context: self.repetition_context,
        }
    }
}

/// Draw a token id from `logits` (`[vocab]` or `[1, vocab]`) on mlx-llm's heap-order host reference
/// ([`mlx_llm::primitives::sample`], the module's one policy): the repetition penalty over the tail
/// of `history`, then temperature + optional top-k / top-p, from `rng`. Greedy (the argmax, ties to
/// the lowest index) when `temperature <= 0` or the shaped mass is not a positive finite number.
pub fn sample_token(
    logits: &Array,
    history: &[i32],
    p: &SampleParams,
    rng: &mut SplitMix64,
) -> Result<i32> {
    mlx_llm::primitives::sample(logits, history, &p.to_sampling(), rng, None)
        .map_err(|e| Error::Msg(format!("text sample: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_llm::primitives::TokenRng as _;

    fn row(v: &[f32]) -> Array {
        Array::from_slice(v, &[v.len() as i32])
    }

    fn draw(logits: &[f32], history: &[i32], p: &SampleParams, rng: &mut SplitMix64) -> i32 {
        sample_token(&row(logits), history, p, rng).unwrap()
    }

    #[test]
    fn splitmix64_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn presets_match_reference() {
        let t = SampleParams::temperature(0.7);
        assert_eq!(t.top_k, 0);
        assert_eq!(t.top_p, 1.0);
        assert_eq!(t.repetition_penalty, None);

        let c = SampleParams::censored(0.7);
        assert_eq!(c.repetition_penalty, Some(1.3));
        assert_eq!(c.repetition_context, 20);

        let u = SampleParams::uncensored(0.7);
        assert_eq!(u.repetition_penalty, None);
        assert_eq!(u.top_k, 0);
    }

    #[test]
    fn temperature_zero_is_greedy_argmax() {
        // Deterministic argmax regardless of the rng draw.
        let logits = [0.1, 2.5, -1.0, 2.4, 0.0];
        let mut rng = SplitMix64::new(1);
        let params = SampleParams::temperature(0.0);
        for _ in 0..8 {
            assert_eq!(draw(&logits, &[], &params, &mut rng), 1);
        }
    }

    #[test]
    fn sampling_is_seed_reproducible() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SampleParams::temperature(1.0);
        let seq = |seed: u64| {
            let mut rng = SplitMix64::new(seed);
            (0..64)
                .map(|_| draw(&logits, &[], &params, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(seq(7), seq(7), "same seed must reproduce the same draws");
        // A uniform categorical over 6 tokens with two different seeds should (overwhelmingly) differ.
        assert_ne!(seq(7), seq(8));
    }

    #[test]
    fn top_k_one_forces_the_argmax_token() {
        // top_k = 1 collapses the candidate set to the single largest logit → deterministic.
        let logits = [0.0, 0.5, 3.0, 0.5, 1.0];
        let mut params = SampleParams::temperature(1.0);
        params.top_k = 1;
        let mut rng = SplitMix64::new(3);
        for _ in 0..8 {
            assert_eq!(draw(&logits, &[], &params, &mut rng), 2);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_recent_tokens() {
        // Token 0 has the top logit; penalizing it (in history) below token 1 flips the argmax at temp 0.
        let logits = [2.0, 1.5, 0.0];
        let mut params = SampleParams::temperature(0.0); // greedy so the penalty effect is exact
        params.repetition_penalty = Some(2.0);
        params.repetition_context = 4;
        let mut rng = SplitMix64::new(0);
        // 2.0 / 2.0 = 1.0 < 1.5 → token 1 wins once token 0 is in the recent history.
        assert_eq!(draw(&logits, &[0], &params, &mut rng), 1);
        // Without the history the penalty does not apply → token 0 wins.
        assert_eq!(draw(&logits, &[], &params, &mut rng), 0);
    }

    /// The upstream penalty gathers and scatters: an id repeated in the window is penalized once,
    /// never compounded by its count (sc-24446; the local copy this module used to carry divided once
    /// per occurrence).
    #[test]
    fn a_repeated_id_is_penalized_once() {
        // 3.0 / 2 = 1.5 > 1.4 keeps token 0 on top; 3.0 / 4 (compounded) = 0.75 would not.
        let logits = [3.0, 1.4, 0.0];
        let mut params = SampleParams::temperature(0.0);
        params.repetition_penalty = Some(2.0);
        params.repetition_context = 8;
        let mut rng = SplitMix64::new(0);
        assert_eq!(draw(&logits, &[0, 0, 0], &params, &mut rng), 0);
    }

    /// One implementation (E8): every preset's draw is mlx-llm's heap-order host reference from the
    /// same stream, token for token — this module adds the knobs, not a sampler.
    #[test]
    fn the_draw_is_the_shared_host_reference() {
        let logits: Vec<f32> = (0..97)
            .map(|i| ((i * 37) % 23) as f32 * 0.21 - 1.7)
            .collect();
        let history: Vec<i32> = (0..40).map(|i| (i * 13) % 97).collect();
        let mut custom = SampleParams::censored(0.9);
        custom.top_k = 30;
        custom.top_p = 0.8;
        for params in [
            SampleParams::temperature(0.7),
            SampleParams::censored(0.7),
            SampleParams::uncensored(1.0),
            custom,
        ] {
            let (mut ours, mut shared) = (SplitMix64::new(5), SplitMix64::new(5));
            for step in 0..32 {
                let history = &history[..8 + step];
                assert_eq!(
                    draw(&logits, history, &params, &mut ours),
                    mlx_llm::primitives::sample(
                        &row(&logits),
                        history,
                        &params.to_sampling(),
                        &mut shared,
                        None
                    )
                    .unwrap(),
                    "{params:?} step {step}"
                );
            }
        }
    }
}
