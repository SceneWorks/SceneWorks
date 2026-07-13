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
//!  2. **VAE-decode tiling** (sc-11744) — engage when the decode-phase spike is the overflow.
//!  3. **activation chunking / resolution cap** (sc-11745) — engage when the denoise steady state is the
//!     overflow.
//!  4. **control-branch quant** (sc-11743, candle-gen #483) — bf16 → q8 → q4. A *quality cost* (the
//!     residual is precision-sensitive, RMS-clamped at τ), so it is the **last-resort rung**.
//!
//! Rungs 2–3 need their own candle-gen mechanism, which ships with sc-11744 / sc-11745; until then they
//! are declared but **not yet engageable** (`WIRED_CHEAPER_RUNGS = false`) — the ladder skips straight to
//! the branch-quant rung and, if even q4 won't fit, rejects-before-OOM noting the pending cheaper rungs.
//! When those stories land they slot in AHEAD of branch-quant here (a localized change), so a constrained
//! card prefers the cheaper rungs and keeps a bf16 branch more often.
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

/// Whether the cheaper-than-branch-quant rungs (VAE-decode tiling sc-11744, activation chunking /
/// resolution cap sc-11745) are wired yet. `false` until those stories ship their candle-gen mechanism +
/// the `Krea2ControlRequest` toggles they engage; the ladder then inserts them AHEAD of the branch-quant
/// rung. Flipping this (and threading the toggles) is those stories' one-line hook into this gate.
const WIRED_CHEAPER_RUNGS: bool = false;

/// The outcome of walking the Krea control fit ladder. `Unknown` = no signal (no `candle.control` block,
/// or a non-NVIDIA host) ⇒ never block, run the bf16 branch — exactly like [`crate::vram_gate::FitDecision::Unknown`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KreaControlFit {
    /// No measured control peak or no live budget ⇒ don't gate; load the default bf16 branch.
    Unknown,
    /// The predicted peak fits (possibly after engaging branch quant). `branch_quant = None` is the
    /// big-card fast path (bf16 branch, zero quality penalty); `Some(Q8)`/`Some(Q4)` is the last-resort
    /// rung the gate engaged to fit a constrained card.
    Fits { branch_quant: Option<Quant> },
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

/// Walk the fit ladder: given the bf16-branch predicted peak, the (capped) live budget, and the measured
/// branch-quant savings, engage the minimal sufficient set of rungs and return the [`KreaControlFit`].
///
/// Today the only wired rung is branch-quant (bf16 → q8 → q4, the LAST resort), so the walk is: if the
/// bf16 peak fits, load bf16 (big-card fast path, no penalty); else prefer q8 (near-lossless), then q4
/// (pose-locked, non-pose drift); else reject. The cheaper rungs (sc-11744/45) engage AHEAD of q8 once
/// [`WIRED_CHEAPER_RUNGS`] flips. Missing peak or budget ⇒ [`KreaControlFit::Unknown`] (never block).
pub(crate) fn fit_ladder(
    peak_bf16_branch_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    q8_save_gb: Option<f64>,
    q4_save_gb: Option<f64>,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_bf16_branch_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let free = budget.free_gb;
    let fits = |needed: f64| free + f64::EPSILON >= needed;

    // (Cheaper rungs — VAE-decode tiling, activation chunking / res-cap — engage here, ahead of
    //  branch-quant, once sc-11744 / sc-11745 wire their candle-gen toggles; see WIRED_CHEAPER_RUNGS.)
    let _ = WIRED_CHEAPER_RUNGS;

    // Big-card fast path: the bf16 branch already fits — no rung engages, zero quality penalty.
    if fits(peak) {
        return KreaControlFit::Fits { branch_quant: None };
    }
    // Last-resort rung: quantize the control branch. Prefer q8 (effectively free quality) before q4
    // (visible non-pose drift). Each save is measured (sc-11743); an unmeasured save skips that option.
    if let Some(save) = q8_save_gb {
        if fits(peak - save) {
            return KreaControlFit::Fits {
                branch_quant: Some(Quant::Q8),
            };
        }
    }
    if let Some(save) = q4_save_gb {
        if fits(peak - save) {
            return KreaControlFit::Fits {
                branch_quant: Some(Quant::Q4),
            };
        }
    }
    // Won't fit even at q4 (the cheaper sc-11744/45 rungs would add more headroom once shipped). Report
    // the best-case (q4-branch) peak so the reject message is honest about what still overflows.
    let best_case = q4_save_gb.or(q8_save_gb).map_or(peak, |save| peak - save);
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
                    // measured sc-11743 (dense base isolates the branch).
                    "branchQuantSaveGb": { "q8": 8.4, "q4": 10.2 }
                }
            }
        }))
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

    fn budget(free: f64) -> VramBudget {
        VramBudget {
            free_gb: free,
            total_gb: free,
        }
    }

    #[test]
    fn big_card_keeps_the_bf16_branch_with_no_penalty() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let q8 = branch_quant_save_gb(&m, "q8");
        let q4 = branch_quant_save_gb(&m, "q4");
        // 96 GB card: the bf16 branch fits outright — nothing engages.
        assert_eq!(
            fit_ladder(peak, Some(budget(90.0)), q8, q4),
            KreaControlFit::Fits { branch_quant: None }
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
    fn constrained_card_prefers_q8_then_q4() {
        let m = krea_manifest();
        // Use the bf16 BASE tier so the peak (46.2 + 2 = 48.2) exercises both branch-quant rungs.
        let peak = predicted_control_peak_gb(&m, "bf16");
        let q8 = branch_quant_save_gb(&m, "q8"); // 8.4
        let q4 = branch_quant_save_gb(&m, "q4"); // 10.2

        // 48.2 fits ⇒ bf16 branch.
        assert_eq!(
            fit_ladder(peak, Some(budget(49.0)), q8, q4),
            KreaControlFit::Fits { branch_quant: None }
        );
        // 48.2 > 41, 48.2 − 8.4 = 39.8 ≤ 41 ⇒ q8 (preferred, near-lossless).
        assert_eq!(
            fit_ladder(peak, Some(budget(41.0)), q8, q4),
            KreaControlFit::Fits {
                branch_quant: Some(Quant::Q8)
            }
        );
        // 39.8 > 39, 48.2 − 10.2 = 38.0 ≤ 39 ⇒ q4 (last resort).
        assert_eq!(
            fit_ladder(peak, Some(budget(39.0)), q8, q4),
            KreaControlFit::Fits {
                branch_quant: Some(Quant::Q4)
            }
        );
        // Even q4 (~38.0) won't fit a 30 GB budget ⇒ reject, reporting the best-case peak.
        assert_too_big(fit_ladder(peak, Some(budget(30.0)), q8, q4), 38.0, 30.0);
    }

    #[test]
    fn q4_base_tier_fits_a_24gb_card_at_q4_branch() {
        let m = krea_manifest();
        // The common small-card install: q4 base. Peak 30.9 + 2 = 32.9.
        let peak = predicted_control_peak_gb(&m, "q4");
        let q8 = branch_quant_save_gb(&m, "q8");
        let q4 = branch_quant_save_gb(&m, "q4");
        // 32.9 > 24; 32.9 − 8.4 = 24.5 > 24; 32.9 − 10.2 = 22.7 ≤ 24 ⇒ q4 branch fits the 24 GB card.
        assert_eq!(
            fit_ladder(peak, Some(budget(24.0)), q8, q4),
            KreaControlFit::Fits {
                branch_quant: Some(Quant::Q4)
            }
        );
        // A 16 GB card can't fit even q4 (~22.7) ⇒ reject (the sc-11744/45 rungs would add headroom).
        assert_too_big(fit_ladder(peak, Some(budget(16.0)), q8, q4), 22.7, 16.0);
    }

    #[test]
    fn missing_signal_never_blocks() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let q8 = branch_quant_save_gb(&m, "q8");
        let q4 = branch_quant_save_gb(&m, "q4");
        // No live budget ⇒ Unknown (never block).
        assert_eq!(fit_ladder(peak, None, q8, q4), KreaControlFit::Unknown);
        // No measured peak ⇒ Unknown.
        assert_eq!(
            fit_ladder(None, Some(budget(8.0)), q8, q4),
            KreaControlFit::Unknown
        );
    }

    #[test]
    fn unmeasured_branch_quant_can_only_reject() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4"); // 32.9
                                                        // No branch-quant savings measured ⇒ the only rung is unavailable, so a too-small card rejects
                                                        // reporting the bf16 peak itself (no rung to subtract).
        assert_too_big(fit_ladder(peak, Some(budget(20.0)), None, None), 32.9, 20.0);
    }
}
