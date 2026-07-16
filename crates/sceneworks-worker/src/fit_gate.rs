//! Backend-neutral fit-gate outcome and sequential-residency transition (sc-12127), plus the
//! backend-neutral Mochi decode arithmetic (sc-12306).
//!
//! MLX and candle derive their predicted resident and sequential peaks differently, but once a
//! resident fit decision exists they share this state transition exactly. Keeping it here prevents
//! the two admission gates from drifting while leaving their backend-specific budgeting untouched.
//!
//! The Mochi block below is the second thing the two lanes genuinely share: an ARCHITECTURAL formula
//! (tensor shapes × dtype), not a backend calibration. It lives here so the candle video gate and the
//! MLX one cannot drift — see [`mochi_decode_peak_gb`].

/// The shared outcome of comparing a predicted resident peak with an available memory budget.
/// `Unknown` means the backend has no reliable signal and therefore must not reject the job.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FitDecision {
    Unknown,
    Fits,
    /// The resident peak does not fit, but the provider supports sequential component residency.
    /// The caller must still run its backend-specific sequential-peak check before loading.
    Offload {
        needed_gb: f64,
        available_gb: f64,
    },
    TooBig {
        needed_gb: f64,
        available_gb: f64,
    },
}

/// Select sequential residency when a resident load is too large and the provider advertises that
/// capability. Every other decision is preserved unchanged.
pub(crate) fn resolve_offload(decision: FitDecision, sequential_capable: bool) -> FitDecision {
    match decision {
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } if sequential_capable => FitDecision::Offload {
            needed_gb,
            available_gb,
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode arithmetic, shared by BOTH lanes (epic 1788, sc-12306)
// ---------------------------------------------------------------------------------------------
//
// Originally derived MLX-side (sc-11992) and lifted here when the candle video lane needed the same
// gate (sc-12306). It sits in the backend-neutral module because every term is a fact about the MODEL
// — tensor shapes and the dtype the decoder pins — not about MLX or candle. Verified against BOTH
// providers rather than assumed:
//
//                              mlx-gen-mochi              candle-gen-mochi
//   decode dtype               f32 (vae.rs from_weights)  f32 (vae.rs:260-264, `DType::F32`)
//   tiled?                     no (`vae.decode(latents)`) no (pipeline.rs:151, one whole-clip call)
//   decoder_block_out_channels [128, 256, 512, 768]       [128, 256, 512, 768] (config.rs:86)
//   temporal_expansions        [1, 2, 3] ⇒ ratio 6        [1, 2, 3] ⇒ ratio 6  (config.rs:88)
//   supports_sequential_offload false                     false (lib.rs:115)
//
// candle's own config module calls itself "the candle twin of `mlx-gen-mochi`'s `config` (byte-for-byte
// the same geometry)", so the shapes are shared by construction. Duplicating the arithmetic per lane
// would let the two gates disagree about one model's memory profile, which is exactly the drift this
// module exists to prevent.
//
// ## This is a FLOOR, and on candle a slightly loose one
// candle's VAE inserts `.contiguous()` after every permute — notably the rank-8 depth-to-space in
// `UpBlock::forward` (vae.rs:170-174), which materializes a full extra copy of the expanded tensor at
// each up-stage; MLX's `transpose_axes` is lazy and fuses. So candle's REAL peak runs at or above what
// this predicts. That direction is safe for an admission gate — it refuses the clearly-impossible and
// never wall-rejects something that would have fit — but it means a MARGINAL candle job can still OOM.
// On CUDA that surfaces as a catchable allocator error via `classify_engine_error`, not the SIGKILL the
// MLX lane faces. Inflating these constants with an invented candle fudge factor would be worse than
// stating the bound: nothing has been measured on-device yet (B5/sc-11995 backfills the real
// `footprint.peakMemoryBytes`; sc-12291 tiles the decode and should then cut the frame term sharply).

/// Bytes in a GiB. The one definition both gates divide by, so "GB" in a user-facing fit message means
/// the same thing on both lanes (it is GiB throughout — see [`mochi_decode_peak_gb`]'s note on units).
pub(crate) const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// The AsymmVAE decoder's final-stage channel count — `decoder_block_out_channels[0]` (config default
/// `[128, 256, 512, 768]`). The last `block_out` mid-block runs THIS many channels at FULL output
/// resolution, which is the decode's peak stage: every earlier stage is either fewer pixels
/// (pre-upsample) or, at 256/512/768 channels, at 1/4, 1/16, 1/64 of the spatial extent, so their
/// products are strictly smaller.
const MOCHI_DECODE_CHANNELS: f64 = 128.0;

/// Bytes per element in the AsymmVAE decode. **f32 (4), NOT bf16 (2)** — BOTH providers pin f32:
/// mlx-gen-mochi's `MochiVaeDecoder::from_weights` passes `Dtype::Float32`, and candle-gen-mochi's
/// `load` passes `DType::F32` twice (vae.rs:260-264), its module doc explaining why ("bf16
/// intermediates reach O(100), outside bf16's range"). Each `decode_denormalized` then casts the
/// incoming latent to that dtype, so the whole decode flows f32 on both lanes.
///
/// ⚠️ This DELIBERATELY disagrees with the sc-12291 / B1-manifest derivation, which assumed **bf16**
/// (`128 × 156 × 480 × 848 × 2 B`) and therefore under-predicts the real peak by 2×. Surfaced as a
/// follow-up on sc-12291 — that story's table (19→7, 61→19, 151→45 GiB) and B1's `mlx.minMemoryGb: 96`
/// are both bf16-derived and want re-deriving against the shipped f32 decode. This gate uses the dtype
/// the code actually runs.
const MOCHI_DECODE_BYTES_PER_ELEM: f64 = 4.0;

/// Full-resolution tensors live simultaneously at the decode peak. `MochiResnetBlock3D` is
/// `GroupNorm → silu → CausalConv3d → GroupNorm → silu → CausalConv3d → + residual`, so the residual
/// and the in-flight intermediate are both live across the block — the "residual + intermediate live at
/// once" claim B1's manifest derivation makes. B1 then hedged "2–3×"; 2 is the concrete, defensible
/// liveness (3 assumes an extra un-freed temporary, which neither allocator need hold). Combined with
/// the f32 correction above this lands at ~79 GiB for the shipped 5 s / 151-frame default — consistent
/// with B1's declared `mlx.minMemoryGb: 96` floor.
const MOCHI_DECODE_LIVE_TENSORS: f64 = 2.0;

/// Extra leading frames the causal decode materializes before `drop_last_temporal_frames` trims them:
/// `temporal_ratio − 1` = 5 (ratio = ∏`temporal_expansions` = 1×2×3 = 6 on both lanes). The peak is
/// paid on the UNTRIMMED length, so a 151-frame clip decodes 156 frames wide.
const MOCHI_DECODE_EXTRA_FRAMES: u32 = 5;

/// Predicted AsymmVAE decode peak (GiB) for a `frames` × `width` × `height` clip.
///
/// `live × C × (frames + 5) × H × W × 4 B`. Linear in BOTH clip length and pixel count, because the
/// peak tensor is `[1, C, T, H, W]` and the decode is untiled on both lanes. This is the term a
/// per-tier constant (MLX's `HEADROOM_GB`, candle's `candle.vramGbByTier`) structurally cannot
/// represent: those are calibrated per LOAD, and the load cannot see the request geometry.
///
/// DERIVED, not measured. Sanity points at the native 848×480 bucket — all **GiB**, the unit this
/// function returns (it divides by [`BYTES_PER_GIB`]): 7 frames ⇒ ~4.66, 19 ⇒ ~9.32, 61 ⇒ ~25.62,
/// 151 ⇒ ~60.56, 163 ⇒ ~65.21.
///
/// (These are the constants the whole derivation leans on, so they are stated in ONE unit on purpose:
/// an earlier revision mixed decimal GB into this list — 5.0/10.1/27.5 are the same three points in
/// GB — which made the table read as if the peak grew faster than it does.)
pub(crate) fn mochi_decode_peak_gb(frames: u32, width: u32, height: u32) -> f64 {
    let decoded_frames = f64::from(frames.saturating_add(MOCHI_DECODE_EXTRA_FRAMES));
    MOCHI_DECODE_LIVE_TENSORS
        * MOCHI_DECODE_CHANNELS
        * decoded_frames
        * f64::from(width)
        * f64::from(height)
        * MOCHI_DECODE_BYTES_PER_ELEM
        / BYTES_PER_GIB
}

/// Predicted whole-generation peak (GiB) for a Mochi request: the ALL-RESIDENT weights
/// (`supports_sequential_offload: false` on BOTH providers ⇒ T5-XXL + AsymmDiT + VAE are held for the
/// whole run, so nothing drops) + the frame-dependent decode peak + `reserve_gb`.
///
/// **`reserve_gb` is passed in, not baked, because the two lanes reserve for DIFFERENT reasons and the
/// terms only coincidentally agree at 2.0 today.** MLX passes `OS_RESERVE_GB`: on unified memory the OS
/// draws from the same pool the model does, so the reserve keeps the machine alive. candle passes
/// `vram_gate::HEADROOM_GB`: the OS does NOT draw from discrete VRAM, so that term instead covers
/// allocator slack and CUDA context overhead. Baking either constant in would silently move the other
/// lane's gate the day someone tunes it.
///
/// `None` when the weights are unmeasurable (`weight_bytes == 0`) ⇒ no signal ⇒ the gate never blocks,
/// matching [`FitDecision::Unknown`].
pub(crate) fn mochi_needed_gb(
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    reserve_gb: f64,
) -> Option<f64> {
    (weight_bytes > 0).then(|| {
        weight_bytes as f64 / BYTES_PER_GIB
            + mochi_decode_peak_gb(frames, width, height)
            + reserve_gb
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mochi's q4 resident weights (GiB): DiT 9.007 + T5-XXL bf16 8.871 + VAE 0.856 = 18.73 (the exact
    /// hosted bytes B1's manifest derivation pins).
    const MOCHI_Q4_RESIDENT_BYTES: u64 = 9_670_883_602 + 9_524_669_250 + 919_551_200;

    /// The decode peak must scale with CLIP LENGTH — the whole reason Mochi cannot ride EITHER lane's
    /// resolution-blind per-load gate. Pins linearity in frames and in pixels, and the architectural
    /// anchor. This runs on every lane (the module is ungated), so the windows-candle lane pins the
    /// same arithmetic its own gate depends on.
    #[test]
    fn mochi_decode_peak_scales_with_frames_and_pixels() {
        // Strictly monotonic in frames.
        let peaks: Vec<f64> = [7, 19, 61, 151, 163]
            .iter()
            .map(|f| mochi_decode_peak_gb(*f, 848, 480))
            .collect();
        for pair in peaks.windows(2) {
            assert!(
                pair[1] > pair[0],
                "decode peak must grow with clip length: {peaks:?}"
            );
        }

        // The architectural anchor: 2 live × 128 ch × (151+5) frames × 848×480 × 4 B (f32) = 60.56 GiB.
        // This is the number a bf16 derivation would halve — pin it so the dtype can't silently drift.
        // BOTH providers pin f32 (mlx-gen-mochi `from_weights`, candle-gen-mochi vae.rs:260-264).
        let at_151 = mochi_decode_peak_gb(151, 848, 480);
        assert!(
            (at_151 - 60.56).abs() < 0.1,
            "151-frame 848x480 decode peak should be ~60.56 GiB (f32), got {at_151:.2}"
        );

        // The 5 s default costs ~50 GiB MORE than the engine's 19-frame default on the same machine —
        // the spread a flat per-load constant cannot express.
        let at_19 = mochi_decode_peak_gb(19, 848, 480);
        assert!(
            at_151 - at_19 > 45.0,
            "151 vs 19 frames must differ by tens of GiB (got {at_19:.1} → {at_151:.2})"
        );

        // Linear in pixel count: halving the height halves the peak.
        let half = mochi_decode_peak_gb(151, 848, 240);
        assert!(
            (at_151 / 2.0 - half).abs() < 0.01,
            "decode peak must be linear in pixels: {at_151:.2} vs {half:.2}"
        );

        // The causal decode pays for `temporal_ratio − 1` extra leading frames before they're trimmed,
        // so the 1-frame floor is not free.
        assert!(mochi_decode_peak_gb(1, 848, 480) > 0.0);
    }

    /// The reserve is a PARAMETER, not a baked constant — the two lanes reserve for different reasons
    /// (MLX: the OS shares the unified pool; candle: allocator slack, since the OS does not draw from
    /// VRAM). They coincide at 2.0 today, so nothing else would catch a re-baking: this test fails if
    /// someone folds either lane's constant back into the shared formula.
    #[test]
    fn mochi_needed_gb_adds_the_callers_reserve_not_a_baked_one() {
        let at_2 =
            mochi_needed_gb(MOCHI_Q4_RESIDENT_BYTES, 19, 848, 480, 2.0).expect("weights > 0");
        let at_5 =
            mochi_needed_gb(MOCHI_Q4_RESIDENT_BYTES, 19, 848, 480, 5.0).expect("weights > 0");
        assert!(
            (at_5 - at_2 - 3.0).abs() < 1e-6,
            "the reserve must pass straight through: {at_2:.3} vs {at_5:.3}"
        );

        // And the total is weights + decode + reserve, not one of them alone.
        let expected = MOCHI_Q4_RESIDENT_BYTES as f64 / BYTES_PER_GIB
            + mochi_decode_peak_gb(19, 848, 480)
            + 2.0;
        assert!(
            (at_2 - expected).abs() < 1e-6,
            "got {at_2:.3}, want {expected:.3}"
        );
    }

    /// Unmeasurable weights ⇒ no signal ⇒ `None` ⇒ every caller admits. The gate never blocks without
    /// evidence, on either lane.
    #[test]
    fn mochi_needed_gb_has_no_signal_without_weights() {
        assert!(mochi_needed_gb(0, 151, 848, 480, 2.0).is_none());
        assert!(mochi_needed_gb(1, 151, 848, 480, 2.0).is_some());
    }
}
