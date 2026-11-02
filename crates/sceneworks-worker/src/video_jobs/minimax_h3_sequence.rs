//! MiniMax-H3 forward-progress watchdog budgets, sized on the **packed DiT sequence** (sc-26691).
//!
//! # Why the target geometry alone is the wrong size
//!
//! H3 packs one joint sequence per denoise evaluation: the text rows (prompt framing **plus every
//! Qwen3-VL vision token** of every visual reference), the reference image/clip latent rows, the
//! reference and generated audio rows, and only then the generated video rows. A `ref2va` request
//! therefore carries far more rows than its canvas implies. The measured sc-26691 ordered-mixed case
//! (1088x544x124 target, two 576x320 image plates encoded at the 2048 short edge, two 576x320 clips
//! put on the 768-short-edge canvas, two voice references) packs ~141k rows against the ~21k a
//! target-only estimate sees, and the per-step DiT cost is not linear in rows: every block runs
//! dense self-attention over the whole sequence. A pixel-frame budget cancelled healthy renders.
//!
//! # Where the sequence size comes from
//!
//! Not re-derived here. [`gen_core_h3::veda::workspace::h3_row_bound`] is the inference contract's
//! own tensor-free H3 layout envelope — the same packing rules (target lattice `17n + 5 → 5n + 2`
//! latents at `pixels / 32` patches, reference image short edge or resize mode, the clip canvas
//! envelope at the 768 x 1344 area, 2 fps vision sampling merged in pairs with the Qwen3-VL
//! `smart_resize` token count, 40 Hz audio rows per channel, prompt framing) Veda admission and the
//! native H3 memory accounting price with, and it moves with the pinned engine (the sc-26691
//! `match` reference-image resize mode lands there with the engine change). It is an **upper**
//! envelope: a reference clip is priced at the generated latent-frame count and a padded grid, so
//! a short clip is over-counted. That is the safe direction for a stall budget — a larger window
//! only detects a true wedge later; a smaller one cancels a progressing render.
//!
//! # The per-step cost model
//!
//! `cost(S) = S · (P + S)`, the linear projections plus the quadratic attention term of one DiT
//! block per packed row, with `P` = [`MINIMAX_H3_ATTENTION_PARITY_ROWS`]: the sequence length at
//! which a row's attention MACs (`2 · S · inner`, QKᵀ and AV) equal its projection MACs
//! (`hidden · (4 · inner + 3 · ffn)`, QKV/O plus the SwiGLU FFN). The step budget scales the
//! measured 600 s baseline step budget by `cost(S) / cost(S₀)`, where `S₀` is the same envelope for
//! the shortest full-canvas text-only request (768 x 1344 x 124) with the same prompt. A request at
//! or below `S₀` keeps 600 s; above it the budget grows linearly-plus-quadratically and stops at
//! [`MINIMAX_H3_STEP_STALL_CEILING`] so a budget is never unbounded.
//!
//! # The window before the first step
//!
//! The candle H3 engine emits **no** progress event until its first denoise step *completes*
//! (`Progress::Step { current: 1 }` is reported after the step), so the first silent window spans
//! the cold model load, the reference phases (vision tower over every visual reference, the text
//! encoder over that grounded prompt, the video/audio VAE encodes of every reference) **and** one
//! full step. The MLX H3 engine additionally reports `Loading(TextEncoder)` / `Loading(Renderer)`,
//! and the worker's watchdog resets on every progress event it receives, so each such phase gets the
//! whole pre-step window again. The pre-step window is therefore
//! `VIDEO_PRE_STEP_STALL_TIMEOUT` (the measured cold load, with headroom) `+` a reference-encode
//! allowance (one baseline step budget per `S₀` rows of reference/vision/audio conditioning,
//! linear: the encoders do not attend across references) `+` the step budget, capped at
//! [`MINIMAX_H3_PRE_STEP_STALL_CEILING`].

use super::prelude::{Conditioning, Duration};
use super::wan::{VideoStallTimeoutPolicy, VIDEO_PRE_STEP_STALL_TIMEOUT, VIDEO_STALL_TIMEOUT};

/// The packed length at which one row's attention MACs equal its projection MACs in a shipped H3
/// DiT block: `hidden · (4 · inner + 3 · ffn) / (2 · inner)` = `5376 · (4·7168 + 3·14336) /
/// (2·7168)` = 26,880 (hidden 5376, 56 heads x 128 = 7168 inner, SwiGLU ffn 14336; the
/// `MiniMaxH3DitConfig` defaults at the pinned engine).
pub(super) const MINIMAX_H3_ATTENTION_PARITY_ROWS: u64 = 26_880;

/// Hard ceiling on one H3 step's silent window, whatever the estimate: a wedge is still caught.
pub(super) const MINIMAX_H3_STEP_STALL_CEILING: Duration = Duration::from_secs(4 * 60 * 60);

/// Hard ceiling on the silent window before the first step event (load + references + one step).
pub(super) const MINIMAX_H3_PRE_STEP_STALL_CEILING: Duration = Duration::from_secs(8 * 60 * 60);

/// The shortest full-canvas request the 600 s baseline was measured at: 768 x 1344 x 124.
const BASELINE_SHORT_EDGE: u32 = 768;
const BASELINE_LONG_EDGE: u32 = 1344;

/// What the H3 estimator reads from the resolved engine input beyond the target geometry.
#[derive(Clone, Copy, Debug)]
pub(super) struct MinimaxH3StallRequest<'a> {
    pub(super) prompt: &'a str,
    pub(super) conditioning: &'a [Conditioning],
    /// The request's `advanced.referenceImageShortEdge`; `None` ⇒ the engine's own default.
    pub(super) reference_image_short_edge: Option<u32>,
    /// The request's `advanced.referenceImageResizeMode` (sc-26691); `None` ⇒ the engine's `match`.
    pub(super) reference_image_resize_mode: Option<gen_core::ReferenceImageResizeMode>,
}

#[cfg(test)]
impl MinimaxH3StallRequest<'_> {
    /// A text-only request with an empty prompt — the shape of the baseline.
    pub(super) const EMPTY: MinimaxH3StallRequest<'static> = MinimaxH3StallRequest {
        prompt: "",
        conditioning: &[],
        reference_image_short_edge: None,
        reference_image_resize_mode: None,
    };
}

/// Clamp `frames` onto H3's legal lattice bounds, so out-of-contract input cannot reach the
/// envelope (which refuses `< 5`) or inflate it past the engine's longest clip.
fn legal_frames(frames: u32) -> u32 {
    let legal = sceneworks_core::video_request::MINIMAX_H3_LEGAL_FRAME_COUNTS;
    frames.clamp(legal[0], legal[legal.len() - 1])
}

/// The packed DiT rows the inference contract's H3 layout envelope prices for this request, or
/// `None` when it refuses the request (zero extent, an unsupported carrier, overflow).
pub(super) fn minimax_h3_packed_rows(
    width: u32,
    height: u32,
    frames: u32,
    request: MinimaxH3StallRequest<'_>,
) -> Option<u64> {
    gen_core_h3::veda::workspace::h3_row_bound(gen_core_h3::veda::workspace::Request {
        veda: None,
        width,
        height,
        frames: legal_frames(frames),
        count: 1,
        prompt: request.prompt,
        conditioning: request.conditioning,
        reference_image_short_edge: request.reference_image_short_edge,
        reference_image_resize_mode: request.reference_image_resize_mode,
    })
    .ok()
    .map(|(packed, _)| packed)
}

/// `S · (P + S)` — see the module docs.
fn step_cost(rows: u64) -> u128 {
    u128::from(rows) * (u128::from(MINIMAX_H3_ATTENTION_PARITY_ROWS) + u128::from(rows))
}

/// `ceil(base · numerator / denominator)` in seconds, saturating, as a `Duration`.
fn scaled(base: Duration, numerator: u128, denominator: u128) -> Duration {
    let secs = (u128::from(base.as_secs()) * numerator).div_ceil(denominator.max(1));
    Duration::from_secs(u64::try_from(secs).unwrap_or(u64::MAX))
}

/// The H3 forward-progress budget for one resolved request — see the module docs.
pub(super) fn minimax_h3_stall_timeout(
    width: u32,
    height: u32,
    frames: u32,
    request: MinimaxH3StallRequest<'_>,
) -> VideoStallTimeoutPolicy {
    let text_only = MinimaxH3StallRequest {
        conditioning: &[],
        ..request
    };
    let baseline = minimax_h3_packed_rows(BASELINE_SHORT_EDGE, BASELINE_LONG_EDGE, 0, text_only);
    let packed = minimax_h3_packed_rows(width, height, frames, request);
    let target_only = minimax_h3_packed_rows(width, height, frames, text_only);
    let (Some(baseline), Some(packed), Some(target_only)) = (baseline, packed, target_only) else {
        // The envelope refused the request, so its size is unknown: never guess small.
        return VideoStallTimeoutPolicy {
            timeout: MINIMAX_H3_STEP_STALL_CEILING,
            pre_step: MINIMAX_H3_PRE_STEP_STALL_CEILING,
            basis: "minimax_h3_sequence_unknown",
        };
    };

    let step_unbounded = scaled(VIDEO_STALL_TIMEOUT, step_cost(packed), step_cost(baseline))
        .max(VIDEO_STALL_TIMEOUT);
    let timeout = step_unbounded.min(MINIMAX_H3_STEP_STALL_CEILING);
    let basis = if step_unbounded > MINIMAX_H3_STEP_STALL_CEILING {
        "minimax_h3_step_ceiling"
    } else if packed <= baseline {
        "minimax_h3_baseline"
    } else {
        "minimax_h3_packed_sequence"
    };

    // Reference/vision/audio conditioning rows beyond the target-only request: encoded once,
    // before the first step, at one baseline step budget per baseline-sequence of rows.
    let reference_rows = packed.saturating_sub(target_only);
    let encode = scaled(
        VIDEO_STALL_TIMEOUT,
        u128::from(reference_rows),
        u128::from(baseline),
    );
    let pre_step = VIDEO_PRE_STEP_STALL_TIMEOUT
        .saturating_add(encode)
        .saturating_add(timeout)
        .min(MINIMAX_H3_PRE_STEP_STALL_CEILING);

    VideoStallTimeoutPolicy {
        timeout,
        pre_step,
        basis,
    }
}

#[cfg(test)]
mod tests {
    use super::super::prelude::Image;
    use super::*;
    use gen_core::AudioTrack;

    /// Geometry only: the envelope never reads pixels, so the tests allocate none.
    fn image(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: Vec::new(),
        }
    }

    fn audio(seconds: u32, channels: u16) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; (seconds * 32_000 * u32::from(channels)) as usize],
            sample_rate: 32_000,
            channels,
            ..AudioTrack::default()
        }
    }

    fn clip(width: u32, height: u32, frames: usize, sound: bool) -> Conditioning {
        Conditioning::ReferenceVideo {
            frames: vec![image(width, height); frames],
            fps: 24.0,
            audio: sound.then(|| audio(2, 1)),
        }
    }

    /// Packed rows under the `diffusers` resize mode — the forced-2048 short edge every measured
    /// row below was recorded with (the only behaviour the engine had before sc-26691).
    fn rows(width: u32, height: u32, frames: u32, conditioning: &[Conditioning]) -> u64 {
        minimax_h3_packed_rows(
            width,
            height,
            frames,
            MinimaxH3StallRequest {
                conditioning,
                reference_image_resize_mode: Some(gen_core::ReferenceImageResizeMode::Diffusers),
                ..MinimaxH3StallRequest::EMPTY
            },
        )
        .unwrap()
    }

    /// sc-26691: under the engine's default `match` mode a plate already below the canvas area is
    /// priced at its own size — the short-edge knob is inert there, so it cannot inflate the budget.
    #[test]
    fn match_mode_prices_a_small_plate_at_its_own_size() {
        let plate = [Conditioning::Reference {
            image: image(576, 320),
            strength: None,
        }];
        let matched = |edge| {
            minimax_h3_packed_rows(
                1088,
                544,
                124,
                MinimaxH3StallRequest {
                    conditioning: &plate,
                    reference_image_short_edge: edge,
                    ..MinimaxH3StallRequest::EMPTY
                },
            )
            .unwrap()
        };
        let bare = rows(1088, 544, 124, &[]);
        // 576x320 on the stride-32 lattice: 18 x 10 = 180 latent rows, plus its vision tokens.
        assert!(matched(None) - bare < rows(1088, 544, 124, &plate) - bare);
        assert_eq!(matched(None), matched(Some(1024)));
        assert_eq!(matched(None), matched(Some(2048)));
    }

    /// The envelope's arithmetic, restated so a silent change to the contract is visible here.
    #[test]
    fn packed_rows_follow_the_inference_layout_envelope() {
        // Target: (124 - 5).div_ceil(17) * 5 + 2 = 37 latents x (544/32) x (1088/32) = 21,386
        // rows; globals: 512 framing + ceil(124 * 40 / 24) = 207 audio latents x 2 channels.
        assert_eq!(rows(1088, 544, 124, &[]), 37 * 17 * 34 + 512 + 207 * 2);
        // The baseline canvas: 37 x 24 x 42 = 37,296 video rows.
        assert_eq!(rows(1344, 768, 124, &[]), 37 * 24 * 42 + 512 + 414);
        // A 576x320 plate at the pinned 2048 short edge is 3680x2048: 115 x 64 = 7,360 latent
        // rows plus the same 7,360 vision tokens and 64 framing tokens.
        let plate = [Conditioning::Reference {
            image: image(576, 320),
            strength: None,
        }];
        assert_eq!(
            rows(1088, 544, 124, &plate) - rows(1088, 544, 124, &[]),
            7_360 + 7_360 + 64
        );
        // An explicit 1024 short edge prices the 1856x1024 plate (58 x 32 = 1,856) twice.
        let short = minimax_h3_packed_rows(
            1088,
            544,
            124,
            MinimaxH3StallRequest {
                conditioning: &plate,
                reference_image_short_edge: Some(1024),
                reference_image_resize_mode: Some(gen_core::ReferenceImageResizeMode::Diffusers),
                ..MinimaxH3StallRequest::EMPTY
            },
        )
        .unwrap();
        assert_eq!(short - rows(1088, 544, 124, &[]), 1_856 * 2 + 64);
    }

    #[test]
    fn the_measured_ordered_mixed_case_is_priced_near_its_141k_rows() {
        // sc-26691 case 1 at the pinned engine: two plates, two 6 s 576x320 clips with their own
        // mono soundtrack, two ~6 s mono voices. The engine packed ~141k rows; the envelope is an
        // upper bound on that, nowhere near the ~21k target-only size.
        let mixed = [
            Conditioning::Reference {
                image: image(576, 320),
                strength: None,
            },
            Conditioning::Reference {
                image: image(576, 320),
                strength: None,
            },
            Conditioning::ReferenceVideo {
                frames: vec![image(576, 320); 144],
                fps: 24.0,
                audio: Some(audio(6, 1)),
            },
            Conditioning::ReferenceVideo {
                frames: vec![image(576, 320); 144],
                fps: 24.0,
                audio: Some(audio(6, 1)),
            },
            Conditioning::ReferenceAudio {
                audio: audio(6, 1),
                strength: None,
            },
            Conditioning::ReferenceAudio {
                audio: audio(6, 1),
                strength: None,
            },
        ];
        let packed = rows(1088, 544, 124, &mixed);
        assert!((141_000..160_000).contains(&packed), "{packed}");
        assert!(packed > 6 * rows(1088, 544, 124, &[]));
    }

    #[test]
    fn baseline_and_smaller_requests_keep_the_600_second_step() {
        for (w, h) in [(1344, 768), (768, 1344), (1088, 544), (576, 320)] {
            let policy = minimax_h3_stall_timeout(w, h, 124, MinimaxH3StallRequest::EMPTY);
            assert_eq!(policy.timeout, VIDEO_STALL_TIMEOUT, "{w}x{h}");
            assert_eq!(policy.basis, "minimax_h3_baseline");
            // Load + no references + one step.
            assert_eq!(
                policy.pre_step,
                VIDEO_PRE_STEP_STALL_TIMEOUT + VIDEO_STALL_TIMEOUT
            );
        }
        // The prompt rides both sides of the ratio, so it never inflates the baseline itself.
        let prompted = MinimaxH3StallRequest {
            prompt: "a long prompt that adds framing rows to the target and the baseline alike",
            ..MinimaxH3StallRequest::EMPTY
        };
        assert_eq!(
            minimax_h3_stall_timeout(1344, 768, 124, prompted).timeout,
            VIDEO_STALL_TIMEOUT
        );
    }

    #[test]
    fn step_budget_is_linear_plus_quadratic_in_packed_rows() {
        let s0 = rows(1344, 768, 124, &[]);
        let s = rows(1344, 768, 243, &[]);
        let p = u128::from(MINIMAX_H3_ATTENTION_PARITY_ROWS);
        let expected = (600 * u128::from(s) * (p + u128::from(s)))
            .div_ceil(u128::from(s0) * (p + u128::from(s0)));
        let policy = minimax_h3_stall_timeout(1344, 768, 243, MinimaxH3StallRequest::EMPTY);
        assert_eq!(policy.timeout, Duration::from_secs(expected as u64));
        assert_eq!(policy.basis, "minimax_h3_packed_sequence");
        // Super-linear: twice-ish the rows costs more than twice the time.
        let ratio = s as f64 / s0 as f64;
        assert!(policy.timeout.as_secs_f64() > 600.0 * ratio);
        // Orientation cannot change packed work.
        assert_eq!(
            policy,
            minimax_h3_stall_timeout(768, 1344, 243, MinimaxH3StallRequest::EMPTY)
        );
    }

    #[test]
    fn references_raise_both_the_step_and_the_pre_step_window() {
        let clips = [
            clip(576, 320, 48, true),
            Conditioning::Reference {
                image: image(576, 320),
                strength: None,
            },
        ];
        let bare = minimax_h3_stall_timeout(1088, 544, 124, MinimaxH3StallRequest::EMPTY);
        let referenced = minimax_h3_stall_timeout(
            1088,
            544,
            124,
            MinimaxH3StallRequest {
                conditioning: &clips,
                reference_image_resize_mode: Some(gen_core::ReferenceImageResizeMode::Diffusers),
                ..MinimaxH3StallRequest::EMPTY
            },
        );
        assert!(referenced.timeout >= bare.timeout);
        assert!(referenced.pre_step > bare.pre_step);
        // pre-step = load + encode allowance + one step, exactly.
        let s0 = rows(1344, 768, 124, &[]);
        let extra = rows(1088, 544, 124, &clips) - rows(1088, 544, 124, &[]);
        let encode = Duration::from_secs((600 * extra).div_ceil(s0));
        assert_eq!(
            referenced.pre_step,
            VIDEO_PRE_STEP_STALL_TIMEOUT + encode + referenced.timeout
        );
    }

    #[test]
    fn budgets_are_monotonic_on_the_legal_lattice_and_always_finite() {
        let mut previous = Duration::ZERO;
        for frames in sceneworks_core::video_request::MINIMAX_H3_LEGAL_FRAME_COUNTS {
            let policy = minimax_h3_stall_timeout(1344, 768, frames, MinimaxH3StallRequest::EMPTY);
            assert!(policy.timeout >= previous, "regressed at {frames}");
            assert!(policy.timeout <= MINIMAX_H3_STEP_STALL_CEILING);
            assert!(policy.pre_step <= MINIMAX_H3_PRE_STEP_STALL_CEILING);
            previous = policy.timeout;
        }
        // Out-of-lattice frame counts clamp onto it.
        assert_eq!(
            minimax_h3_stall_timeout(1344, 768, 0, MinimaxH3StallRequest::EMPTY),
            minimax_h3_stall_timeout(1344, 768, 124, MinimaxH3StallRequest::EMPTY)
        );
        assert_eq!(
            minimax_h3_stall_timeout(1344, 768, u32::MAX, MinimaxH3StallRequest::EMPTY),
            minimax_h3_stall_timeout(1344, 768, 345, MinimaxH3StallRequest::EMPTY)
        );
        // A huge reference load saturates at the ceilings instead of growing without bound.
        let heavy: Vec<Conditioning> = (0..9)
            .map(|_| Conditioning::Reference {
                image: image(2048, 512),
                strength: None,
            })
            .chain((0..3).map(|_| clip(1344, 768, 345, true)))
            .collect();
        let policy = minimax_h3_stall_timeout(
            1344,
            768,
            345,
            MinimaxH3StallRequest {
                conditioning: &heavy,
                ..MinimaxH3StallRequest::EMPTY
            },
        );
        assert_eq!(policy.timeout, MINIMAX_H3_STEP_STALL_CEILING);
        assert_eq!(policy.basis, "minimax_h3_step_ceiling");
        assert!(policy.pre_step > VIDEO_PRE_STEP_STALL_TIMEOUT + MINIMAX_H3_STEP_STALL_CEILING);
        assert!(policy.pre_step <= MINIMAX_H3_PRE_STEP_STALL_CEILING);
        // A request the envelope refuses (zero extent) is never budgeted small.
        let unknown = minimax_h3_stall_timeout(0, 0, 124, MinimaxH3StallRequest::EMPTY);
        assert_eq!(unknown.timeout, MINIMAX_H3_STEP_STALL_CEILING);
        assert_eq!(unknown.pre_step, MINIMAX_H3_PRE_STEP_STALL_CEILING);
        assert_eq!(unknown.basis, "minimax_h3_sequence_unknown");
    }

    #[test]
    fn parity_rows_match_the_shipped_dit_dimensions() {
        let (hidden, inner, ffn) = (5_376u64, 56 * 128u64, 14_336u64);
        assert_eq!(
            hidden * (4 * inner + 3 * ffn) / (2 * inner),
            MINIMAX_H3_ATTENTION_PARITY_ROWS
        );
    }
}
