//! Krea pose-ControlNet VRAM fit ladder (sc-11754, epic 8459 → epic 10765).
//!
//! The Krea control lane is diverted around the base.rs `generate_candle_stream` fit-gate ([`crate::vram_gate`]):
//! it loads through the bespoke `Krea2Control` provider, not the shared txt2img path, so the epic-10765
//! admission check never sees it. This module is its dedicated fit-gate — the SAME live/capped budget
//! ([`crate::gpu::nvidia_vram_budget_gb`] + `SCENEWORKS_CUDA_VRAM_CAP_GB`) vs a control-lane-specific
//! predicted peak, walked down a **cheapest-cost-first rung ladder** so the memory optimizations engage
//! only when VRAM is actually constrained: on a 96 GB card nothing engages (bf16 branch, zero penalty);
//! on a 24/16 GB card the minimum set engages to fit.
//!
//! ## The ladder
//! The control lane's peak is the base tier + the ~6.6 GB bf16 control branch + activations + the
//! end-of-render VAE-decode spike. When that peak won't fit, rungs engage in increasing (Δcost) order,
//! stopping as soon as the predicted peak ≤ budget:
//!  1. **packed base** — default, ~free (candle-gen #480, shipped). Always on; the peak below already
//!     reflects it.
//!  2. **VAE-decode tiling** (sc-11744, candle-gen #492) — force the seam-free tiled VAE tail (a *speed*
//!     cost, no quality cost) to cap the end-of-render decode spike. Engaged first, when the measured
//!     `decodeTileSaveGb` is enough to fit; also stays on underneath the quant rungs (it is cheaper than
//!     quant and every bit helps).
//!  3. **activation chunking / resolution cap** (sc-11745) — engage when the denoise steady state is the
//!     overflow. Still pending its candle-gen mechanism (`WIRED_ACTIVATION_CHUNKING = false`); slots in
//!     here between tiling and branch-quant when it lands.
//!  4. **control-branch quant** (sc-11743, candle-gen #483) — bf16 → q8 → q4. A *quality cost* (the
//!     residual is precision-sensitive, RMS-clamped at τ), so it is the **last-resort rung**. Composes
//!     with tiling (both reduce the decode-dominated peak).
//!
//! Rung 3 needs its own candle-gen mechanism (sc-11745); until then it is declared but **not yet
//! engageable** — the ladder walks packed → tiling → branch-quant and, if even (tiling + q4) won't fit,
//! rejects-before-OOM noting the pending activation-chunking rung. `decodeTileSaveGb` absent for a tier
//! (unmeasured) ⇒ the tiling rung is skipped for that tier exactly like an unmeasured branch-quant save.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and the
//! wiring is in `generate_candle_krea_control_stream` (image_jobs/krea_control_candle.rs).

use super::*;
use gen_core::Quant;
use serde_json::Value;

/// Fixed transient/runtime headroom (GB) added on top of the MEASURED control-lane peak
/// (`candle.control.peakGbByTier`), mirroring [`crate::vram_gate`]'s `HEADROOM_GB` — covers allocator
/// slack + activation spikes not captured by the steady peak.
const HEADROOM_GB: f64 = 2.0;

/// Whether the activation-chunking / resolution-cap rung (sc-11745) is wired yet. `false` until that
/// story ships its candle-gen mechanism + the `Krea2ControlRequest` toggle it engages; the ladder then
/// inserts it between the VAE-tiling rung (sc-11744, already wired below) and the branch-quant rung.
/// Flipping this (and threading the toggle) is sc-11745's one-line hook into this gate. (The VAE-tiling
/// rung is now live — gated instead by the presence of the measured `decodeTileSaveGb`.)
const WIRED_ACTIVATION_CHUNKING: bool = false;

/// The outcome of walking the Krea control fit ladder. `Unknown` = no signal (no `candle.control` block,
/// or a non-NVIDIA host) ⇒ never block, run the bf16 branch — exactly like [`crate::vram_gate::FitDecision::Unknown`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KreaControlFit {
    /// No measured control peak or no live budget ⇒ don't gate; load the default bf16 branch.
    Unknown,
    /// The predicted peak fits after engaging the minimal sufficient set of rungs. The big-card fast
    /// path is `{ tile_vae_decode: false, branch_quant: None }` (monolithic full-speed decode, bf16
    /// branch, zero penalty). `tile_vae_decode = true` is the cheapest rung (sc-11744 — a *speed* cost
    /// only, seam-free); `branch_quant = Some(Q8)`/`Some(Q4)` is the last-resort *quality* rung, engaged
    /// only after tiling was insufficient.
    Fits {
        /// Force the seam-free tiled VAE decode (sc-11744) to cap the end-of-render decode spike. The
        /// worker threads this into `Krea2ControlRequest::tile_vae_decode`.
        tile_vae_decode: bool,
        branch_quant: Option<Quant>,
    },
    /// Won't fit even at the last rung (q4 branch). Reject-before-OOM with an actionable message rather
    /// than a reactive CUDA OOM mid-render. `needed_gb` is the best-case (q4-branch) predicted peak.
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// Predicted control-lane peak VRAM (GB) for `tier_key` (the BASE tier — `bf16`/`q8`/`q4`), with a bf16
/// control branch and no rungs engaged: `candle.control.peakGbByTier[tier_key]` + [`HEADROOM_GB`], else
/// `None` (unmeasured ⇒ the gate no-ops). The control peak exceeds the txt2img `candle.vramGbByTier`
/// because the ~6.6 GB control branch is co-resident.
pub(crate) fn predicted_control_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("peakGbByTier")
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
        .map(|gb| gb + HEADROOM_GB)
}

/// Measured peak reduction (GB) from quantizing the control branch bf16 → `quant` (`"q8"`/`"q4"`):
/// `candle.control.branchQuantSaveGb[quant]`. `None` when unmeasured (the branch-quant rung is then
/// unavailable and the gate can only reject). Published measured (sc-11743, candle-gen #483): q8 −8.4,
/// q4 −10.2 GiB.
pub(crate) fn branch_quant_save_gb(manifest_entry: &JsonObject, quant: &str) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("branchQuantSaveGb")
        .and_then(|saves| saves.get(quant))
        .and_then(json_f64)
}

/// Measured peak reduction (GB) from forcing the seam-free tiled VAE decode for BASE tier `tier_key`
/// (sc-11744, candle-gen #492): `candle.control.decodeTileSaveGb[tier_key]` — the monolithic whole-render
/// peak minus the tiled one (the decode spike capped toward the denoise steady state). `None` when
/// unmeasured for that tier ⇒ the VAE-tiling rung is unavailable there and the ladder walks straight to
/// branch-quant. Tier-keyed like [`predicted_control_peak_gb`] because whether the decode spike *is* the
/// global peak (and thus how much tiling recovers) depends on the tier's denoise-steady floor.
pub(crate) fn decode_tile_save_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("decodeTileSaveGb")
        .and_then(|saves| saves.get(tier_key))
        .and_then(json_f64)
}

/// Walk the fit ladder: given the bf16-branch predicted peak, the (capped) live budget, the measured
/// VAE-decode tiling saving, and the measured branch-quant savings, engage the minimal sufficient set of
/// rungs in increasing (Δcost) order and return the [`KreaControlFit`].
///
/// Walk: if the monolithic peak fits, run untiled at the bf16 branch (big-card fast path, no penalty);
/// else force the seam-free tiled decode (sc-11744 — a *speed* cost only); else — keeping tiling on —
/// quantize the control branch, preferring q8 (near-lossless) then q4 (pose-locked, non-pose drift);
/// else reject-before-OOM. The activation-chunking rung (sc-11745) slots between tiling and branch-quant
/// once [`WIRED_ACTIVATION_CHUNKING`] flips. An unmeasured saving skips that rung. Missing peak or budget
/// ⇒ [`KreaControlFit::Unknown`] (never block).
pub(crate) fn fit_ladder(
    peak_bf16_branch_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    q8_save_gb: Option<f64>,
    q4_save_gb: Option<f64>,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_bf16_branch_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let free = budget.free_gb;
    let fits = |needed: f64| free + f64::EPSILON >= needed;

    // (The activation-chunking / res-cap rung, sc-11745, engages here — between tiling and branch-quant —
    //  once it wires its candle-gen toggle; see WIRED_ACTIVATION_CHUNKING.)
    let _ = WIRED_ACTIVATION_CHUNKING;

    // Rung 1 (packed base, always on) — big-card fast path: the monolithic bf16-branch peak already fits,
    // so nothing engages: full-speed decode, zero quality penalty.
    if fits(peak) {
        return KreaControlFit::Fits {
            tile_vae_decode: false,
            branch_quant: None,
        };
    }
    // Rung 2 (VAE-decode tiling, sc-11744) — cheapest: a *speed* cost, no quality loss (seam-free). An
    // unmeasured tier saving (None) leaves this rung unavailable, degenerating to the old quant-only walk.
    if let Some(tile_save) = decode_tile_save_gb {
        if fits(peak - tile_save) {
            return KreaControlFit::Fits {
                tile_vae_decode: true,
                branch_quant: None,
            };
        }
    }
    // Past here tiling alone was insufficient, so keep it on beneath the quant rungs (it is cheaper than
    // quant and every GB helps); if it was unmeasured, `tile_on` stays false and the peak is unreduced.
    let tile_on = decode_tile_save_gb.is_some();
    let peak_after_tile = peak - decode_tile_save_gb.unwrap_or(0.0);

    // Rung 4 (control-branch quant, sc-11743) — last resort, a *quality* cost. Prefer q8 (effectively
    // free quality) before q4 (visible non-pose drift). Each save is measured; an unmeasured save skips.
    if let Some(save) = q8_save_gb {
        if fits(peak_after_tile - save) {
            return KreaControlFit::Fits {
                tile_vae_decode: tile_on,
                branch_quant: Some(Quant::Q8),
            };
        }
    }
    if let Some(save) = q4_save_gb {
        if fits(peak_after_tile - save) {
            return KreaControlFit::Fits {
                tile_vae_decode: tile_on,
                branch_quant: Some(Quant::Q4),
            };
        }
    }
    // Won't fit even at (tiling + q4). Report the best-case peak (tiling on + the deepest branch quant) so
    // the reject message is honest about what still overflows (the sc-11745 rung would add more headroom).
    let best_case = peak_after_tile - q4_save_gb.or(q8_save_gb).unwrap_or(0.0);
    KreaControlFit::TooBig {
        needed_gb: best_case,
        available_gb: free,
    }
}

/// Parse a JSON float from either a number or a numeric string (mirrors [`crate::vram_gate`]'s helper).
fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vram_gate::VramBudget;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    fn krea_manifest() -> JsonObject {
        obj(json!({
            "candle": {
                "minMemoryGb": 32,
                "vramGbByTier": { "q4": 26.4, "q8": 31.0, "bf16": 44.0 },
                "control": {
                    // q4 measured (sc-11744), bf16 measured (sc-11743), q8 estimated.
                    "peakGbByTier": { "q4": 30.9, "q8": 35.5, "bf16": 46.2 },
                    // VAE-decode tiling saving (sc-11744); measured only for the constrained q4 tier here
                    // (the decode spike is the peak there). Illustrative test value, not the shipped one.
                    "decodeTileSaveGb": { "q4": 6.9 },
                    // measured sc-11743 (dense base isolates the branch).
                    "branchQuantSaveGb": { "q8": 8.4, "q4": 10.2 }
                }
            }
        }))
    }

    /// Read the q4 rung savings the ladder tests exercise (tiling + both branch quants).
    fn q4_saves(m: &JsonObject) -> (Option<f64>, Option<f64>, Option<f64>) {
        (
            decode_tile_save_gb(m, "q4"),
            branch_quant_save_gb(m, "q8"),
            branch_quant_save_gb(m, "q4"),
        )
    }

    #[test]
    fn predicted_control_peak_reads_the_tier_plus_headroom() {
        let m = krea_manifest();
        assert_eq!(
            predicted_control_peak_gb(&m, "q4"),
            Some(30.9 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_control_peak_gb(&m, "bf16"),
            Some(46.2 + HEADROOM_GB)
        );
        // Tier absent from peakGbByTier ⇒ None (gate no-ops for that tier).
        assert_eq!(predicted_control_peak_gb(&m, "int8-convrot"), None);
        // No control block ⇒ None (unmeasured control lane).
        let no_control = obj(json!({ "candle": { "vramGbByTier": { "q4": 26.4 } } }));
        assert_eq!(predicted_control_peak_gb(&no_control, "q4"), None);
        // No candle block ⇒ None.
        assert_eq!(predicted_control_peak_gb(&obj(json!({})), "q4"), None);
    }

    #[test]
    fn branch_quant_save_reads_the_measured_deltas() {
        let m = krea_manifest();
        assert_eq!(branch_quant_save_gb(&m, "q8"), Some(8.4));
        assert_eq!(branch_quant_save_gb(&m, "q4"), Some(10.2));
        assert_eq!(branch_quant_save_gb(&m, "q2"), None);
        assert_eq!(branch_quant_save_gb(&obj(json!({})), "q8"), None);
    }

    #[test]
    fn decode_tile_save_reads_the_measured_delta() {
        let m = krea_manifest();
        assert_eq!(decode_tile_save_gb(&m, "q4"), Some(6.9));
        // Unmeasured for a tier ⇒ None (the tiling rung is unavailable there).
        assert_eq!(decode_tile_save_gb(&m, "bf16"), None);
        // No control block / no candle block ⇒ None.
        assert_eq!(decode_tile_save_gb(&obj(json!({})), "q4"), None);
    }

    fn budget(free: f64) -> VramBudget {
        VramBudget {
            free_gb: free,
            total_gb: free,
        }
    }

    /// The big-card fast path: untiled monolithic decode, bf16 branch, no rung engaged.
    fn fits_untiled_bf16() -> KreaControlFit {
        KreaControlFit::Fits {
            tile_vae_decode: false,
            branch_quant: None,
        }
    }

    #[test]
    fn big_card_keeps_the_bf16_branch_with_no_penalty() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, q8, q4) = q4_saves(&m);
        // 96 GB card: the monolithic peak fits outright — nothing engages (no tiling, no quant).
        assert_eq!(
            fit_ladder(peak, Some(budget(90.0)), tile, q8, q4),
            fits_untiled_bf16()
        );
    }

    #[test]
    fn vae_tiling_is_the_cheapest_rung_and_keeps_the_bf16_branch() {
        let m = krea_manifest();
        // q4 base: monolithic peak 30.9 + 2 = 32.9; tiling saves 6.9 ⇒ tiled peak 26.0.
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, q8, q4) = q4_saves(&m);
        // 26 GB card: monolithic (32.9) won't fit, but the tiled decode (26.0) does ⇒ tile on, bf16 branch
        // kept (no quality penalty). Without sc-11744 this card would have dropped to a q8 branch.
        assert_eq!(
            fit_ladder(peak, Some(budget(26.0)), tile, q8, q4),
            KreaControlFit::Fits {
                tile_vae_decode: true,
                branch_quant: None,
            }
        );
    }

    #[test]
    fn tiling_composes_with_branch_quant_when_still_constrained() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4"); // 32.9; tiled 26.0
        let (tile, q8, q4) = q4_saves(&m);
        // 24 GB card: tiled peak 26.0 > 24; tiling stays on and q8 engages (26.0 − 8.4 = 17.6 ≤ 24) ⇒
        // tiling + q8 (near-lossless) instead of the old tiling-less q4-branch drop.
        assert_eq!(
            fit_ladder(peak, Some(budget(24.0)), tile, q8, q4),
            KreaControlFit::Fits {
                tile_vae_decode: true,
                branch_quant: Some(Quant::Q8),
            }
        );
        // 16 GB card: tiling + q8 (17.6) > 16; tiling + q4 (26.0 − 10.2 = 15.8 ≤ 16) fits ⇒ the deepest
        // rung. Without the tiling rung this card OOM-rejected (q4-alone peak 22.7 > 16).
        assert_eq!(
            fit_ladder(peak, Some(budget(16.0)), tile, q8, q4),
            KreaControlFit::Fits {
                tile_vae_decode: true,
                branch_quant: Some(Quant::Q4),
            }
        );
        // 15 GB card: even tiling + q4 (15.8) won't fit ⇒ reject-before-OOM at the best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(15.0)), tile, q8, q4),
            15.8,
            15.0,
        );
    }

    /// Assert a [`KreaControlFit::TooBig`] whose `needed_gb` is ~`expected` (float-tolerant).
    fn assert_too_big(fit: KreaControlFit, expected_needed: f64, expected_free: f64) {
        match fit {
            KreaControlFit::TooBig {
                needed_gb,
                available_gb,
            } => {
                assert!(
                    (needed_gb - expected_needed).abs() < 1e-6,
                    "needed_gb {needed_gb} ≉ {expected_needed}"
                );
                assert!((available_gb - expected_free).abs() < 1e-6);
            }
            other => panic!("expected TooBig, got {other:?}"),
        }
    }

    #[test]
    fn pure_quant_walk_when_the_tier_has_no_tiling_measurement() {
        let m = krea_manifest();
        // The bf16 BASE tier carries no `decodeTileSaveGb` (the decode spike is not its peak), so the
        // tiling rung is unavailable and the walk degenerates to the old quant-only ladder. Peak 48.2.
        let peak = predicted_control_peak_gb(&m, "bf16");
        let tile = decode_tile_save_gb(&m, "bf16"); // None
        let q8 = branch_quant_save_gb(&m, "q8"); // 8.4
        let q4 = branch_quant_save_gb(&m, "q4"); // 10.2

        // 48.2 fits ⇒ untiled bf16 branch.
        assert_eq!(
            fit_ladder(peak, Some(budget(49.0)), tile, q8, q4),
            fits_untiled_bf16()
        );
        // 48.2 > 41, 48.2 − 8.4 = 39.8 ≤ 41 ⇒ q8 (preferred, near-lossless), tiling unavailable.
        assert_eq!(
            fit_ladder(peak, Some(budget(41.0)), tile, q8, q4),
            KreaControlFit::Fits {
                tile_vae_decode: false,
                branch_quant: Some(Quant::Q8),
            }
        );
        // 39.8 > 39, 48.2 − 10.2 = 38.0 ≤ 39 ⇒ q4 (last resort).
        assert_eq!(
            fit_ladder(peak, Some(budget(39.0)), tile, q8, q4),
            KreaControlFit::Fits {
                tile_vae_decode: false,
                branch_quant: Some(Quant::Q4),
            }
        );
        // Even q4 (~38.0) won't fit a 30 GB budget ⇒ reject, reporting the best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(30.0)), tile, q8, q4),
            38.0,
            30.0,
        );
    }

    #[test]
    fn missing_signal_never_blocks() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, q8, q4) = q4_saves(&m);
        // No live budget ⇒ Unknown (never block).
        assert_eq!(
            fit_ladder(peak, None, tile, q8, q4),
            KreaControlFit::Unknown
        );
        // No measured peak ⇒ Unknown.
        assert_eq!(
            fit_ladder(None, Some(budget(8.0)), tile, q8, q4),
            KreaControlFit::Unknown
        );
    }

    #[test]
    fn unmeasured_rungs_can_only_reject_at_the_raw_peak() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4"); // 32.9
                                                        // No tiling save and no branch-quant savings measured ⇒ no rung is available, so a too-small card
                                                        // rejects reporting the raw peak itself (nothing to subtract).
        assert_too_big(
            fit_ladder(peak, Some(budget(20.0)), None, None, None),
            32.9,
            20.0,
        );
    }
}
