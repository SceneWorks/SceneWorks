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
//!  2. **sequential residency** (sc-12176) — encode/drop Qwen3-VL before loading the heavy phase. The
//!     cheapest adaptation and no quality cost; its measured per-tier peak enables a clean second-stage
//!     reject when even the staged working set will not fit.
//!  3. **VAE-decode tiling** (sc-11744, candle-gen #492) — force the seam-free tiled VAE tail (a *speed*
//!     cost, no quality cost) to cap the end-of-render decode spike. Engaged after residency, when the measured
//!     `decodeTileSaveGb` is enough to fit; also stays on underneath the quant rungs (it is cheaper than
//!     quant and every bit helps).
//!  4. **activation chunking** (sc-11745, candle-gen #496) — engage sc-6217-style query-row attention
//!     chunking on the composable base stack + control branch when the denoise steady state is the
//!     overflow. A *speed* cost (~+6%) with byte-identical output; slots between tiling and branch-quant.
//!     Gated by the presence of the measured scalar `chunkAttnSaveGb`.
//!  5. **control-branch quant** (sc-11743, candle-gen #483) — bf16 → q8 → q4. A *quality cost* (the
//!     residual is precision-sensitive, RMS-clamped at τ), so it is the **last-resort rung**. Composes
//!     with the cheaper rungs (all reduce the co-resident peak).
//!
//! `decodeTileSaveGb` / `chunkAttnSaveGb` absent (unmeasured) ⇒ that rung is skipped exactly like an
//! unmeasured branch-quant save — the ladder walks the rungs it *can* measure and, if even
//! (sequential residency + tiling + chunking + q4) won't fit, rejects-before-OOM at the honest
//! best-case peak.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and the
//! wiring is in `generate_candle_krea_control_stream` (image_jobs/krea_control_candle.rs).

use super::*;
use gen_core::{OffloadPolicy, Quant};
use serde_json::Value;

/// Fixed transient/runtime headroom (GB) added on top of the MEASURED control-lane peak
/// (`candle.control.peakGbByTier`), mirroring [`crate::vram_gate`]'s `HEADROOM_GB` — covers allocator
/// slack + activation spikes not captured by the steady peak.
const HEADROOM_GB: f64 = 2.0;

/// The outcome of walking the Krea control fit ladder. `Unknown` = no signal (no `candle.control` block,
/// or a non-NVIDIA host) ⇒ never block, run the bf16 branch — exactly like [`crate::vram_gate::FitDecision::Unknown`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KreaControlFit {
    /// No measured control peak or no live budget ⇒ don't gate; load the default bf16 branch.
    Unknown,
    /// The predicted peak fits after engaging the minimal sufficient set of rungs. The big-card fast
    /// path uses resident components with `{ tile_vae_decode: false, chunk_attention: false,
    /// branch_quant: None }` (monolithic full-speed decode, unchunked attention, bf16 branch, zero
    /// penalty). Sequential residency is the first adaptation; `tile_vae_decode = true` is the next rung
    /// (sc-11744 — a *speed* cost only, seam-free); `chunk_attention = true` follows (sc-11745 — a
    /// *speed* cost only, byte-identical output); `branch_quant = Some(Q8)`/`Some(Q4)` is the last-resort
    /// *quality* rung, engaged only after the cheaper rungs were insufficient.
    Fits {
        /// Component residency selected before any deeper rung. Sequential is the cheapest adaptation:
        /// it drops Qwen3-VL before loading the DiT + control branch + VAE.
        offload_policy: OffloadPolicy,
        /// Force the seam-free tiled VAE decode (sc-11744) to cap the end-of-render decode spike. The
        /// worker threads this into `Krea2ControlRequest::tile_vae_decode`.
        tile_vae_decode: bool,
        /// Engage sc-6217-style query-row attention chunking (sc-11745) to bound the denoise activation
        /// peak. The worker threads this into `Krea2ControlPaths::chunk_attention` (a load-time toggle).
        chunk_attention: bool,
        branch_quant: Option<Quant>,
    },
    /// Won't fit even at the last rung (q4 branch). Reject-before-OOM with an actionable message rather
    /// than a reactive CUDA OOM mid-render. `needed_gb` is the best-case sequential + tiled + chunked +
    /// q4-branch predicted peak.
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

/// Predicted Sequential control-lane peak for `tier_key`:
/// `candle.control.sequentialPeakGbByTier[tier_key]` + [`HEADROOM_GB`]. This is the measured largest
/// single working set after Qwen3-VL has been encoded and dropped. `None` preserves best-effort
/// offload behavior: the lane still stages, but cannot perform an honest second-stage reject.
pub(crate) fn predicted_control_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("sequentialPeakGbByTier")
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

/// Measured peak reduction (GB) from engaging sc-6217-style query-row attention chunking on the composable
/// base stack + control branch (sc-11745, candle-gen #496): `candle.control.chunkAttnSaveGb`. Bounds the
/// DENOISE-phase activation peak — a *speed* cost (~+6%) with byte-identical output (the chunked forward is
/// numerically identical to the unchunked one, the sc-6217 query-row-independence invariant). Unlike the
/// tier-keyed [`decode_tile_save_gb`]/[`predicted_control_peak_gb`], this is a SCALAR: the denoise activation
/// peak is bf16 regardless of the base weight tier, so the saving is tier-independent. `None` when unmeasured
/// ⇒ the chunking rung is unavailable and the ladder walks tiling → branch-quant. Published measured (RTX PRO
/// 6000, dense bf16 base, 1024²/8-step): −2.43 GiB.
pub(crate) fn chunk_attn_save_gb(manifest_entry: &JsonObject) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("chunkAttnSaveGb")
        .and_then(json_f64)
}

/// Walk the fit ladder: given the bf16-branch predicted peak, the (capped) live budget, the measured
/// VAE-decode tiling saving, the measured activation-chunking saving, and the measured branch-quant
/// savings, engage the minimal sufficient set of rungs in increasing (Δcost) order and return the
/// [`KreaControlFit`].
///
/// Walk: if the monolithic peak fits, run resident/untiled/unchunked at the bf16 branch (big-card fast
/// path, no penalty); else stage text and heavy components sequentially. If the measured sequential peak
/// still exceeds the budget, force the seam-free tiled decode (sc-11744 — a *speed* cost only); then,
/// keeping tiling on, engage query-row attention chunking (sc-11745 — a *speed* cost only,
/// byte-identical); then, keeping both speed rungs on, quantize the control branch, preferring q8
/// (near-lossless) then q4 (pose-locked, non-pose drift); else reject-before-OOM. Any unmeasured saving
/// (`None`) skips its rung. Missing resident peak or budget ⇒ [`KreaControlFit::Unknown`] (never block).
pub(crate) fn fit_ladder(
    peak_bf16_branch_gb: Option<f64>,
    sequential_peak_bf16_branch_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    chunk_attn_save_gb: Option<f64>,
    q8_save_gb: Option<f64>,
    q4_save_gb: Option<f64>,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_bf16_branch_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let free = budget.free_gb;
    let fits = |needed: f64| free + f64::EPSILON >= needed;

    // Rung 1 (packed base, always on) — big-card fast path: the monolithic bf16-branch peak already fits,
    // so nothing engages: full-speed decode, unchunked attention, zero quality penalty.
    if fits(peak) {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        };
    }

    // Rung 2 (sequential residency, sc-12176) — the cheapest adaptation. If its measured peak is
    // absent, preserve the shared gate's best-effort contract: stage anyway and rely on the reactive
    // CUDA-OOM backstop rather than engaging costlier/quality-affecting rungs from an unknown baseline.
    let Some(peak) = sequential_peak_bf16_branch_gb else {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        };
    };
    if fits(peak) {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        };
    }
    // Rung 3 (VAE-decode tiling, sc-11744) — a *speed* cost, no quality loss (seam-free). An
    // unmeasured tier saving (None) leaves this rung unavailable, degenerating toward the quant-only walk.
    if let Some(tile_save) = decode_tile_save_gb {
        if fits(peak - tile_save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                branch_quant: None,
            };
        }
    }
    // Past here tiling alone was insufficient, so keep it on beneath the deeper rungs (it is cheaper than
    // chunking or quant and every GB helps); if it was unmeasured, `tile_on` stays false / the peak unreduced.
    let tile_on = decode_tile_save_gb.is_some();
    let peak_after_tile = peak - decode_tile_save_gb.unwrap_or(0.0);

    // Rung 4 (activation chunking, sc-11745, candle-gen #496) — a *speed* cost (~+6%), byte-identical output
    // (the chunked forward is numerically identical). Cheaper than the quality-costing branch quant, so it is
    // engaged BEFORE it. An unmeasured saving (None) leaves this rung unavailable.
    if let Some(chunk_save) = chunk_attn_save_gb {
        if fits(peak_after_tile - chunk_save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: tile_on,
                chunk_attention: true,
                branch_quant: None,
            };
        }
    }
    // Chunking alone was insufficient too, so keep it on beneath the quant rungs (byte-identical, cheaper
    // than quant); if it was unmeasured, `chunk_on` stays false and the peak is unreduced by it.
    let chunk_on = chunk_attn_save_gb.is_some();
    let peak_after_chunk = peak_after_tile - chunk_attn_save_gb.unwrap_or(0.0);

    // Rung 5 (control-branch quant, sc-11743) — last resort, a *quality* cost. Prefer q8 (effectively
    // free quality) before q4 (visible non-pose drift). Each save is measured; an unmeasured save skips.
    if let Some(save) = q8_save_gb {
        if fits(peak_after_chunk - save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: tile_on,
                chunk_attention: chunk_on,
                branch_quant: Some(Quant::Q8),
            };
        }
    }
    if let Some(save) = q4_save_gb {
        if fits(peak_after_chunk - save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: tile_on,
                chunk_attention: chunk_on,
                branch_quant: Some(Quant::Q4),
            };
        }
    }
    // Won't fit even at (tiling + chunking + q4). Report the best-case peak (every cheaper rung on + the
    // deepest branch quant) so the reject message is honest about what still overflows.
    let best_case = peak_after_chunk - q4_save_gb.or(q8_save_gb).unwrap_or(0.0);
    KreaControlFit::TooBig {
        needed_gb: best_case,
        available_gb: free,
    }
}

/// The VRAM peak (GB) a Krea control render actually incurs under `fit`, for the reclaimable high-water
/// ([`crate::vram_gate::note_loaded_peak`], sc-13960 — the control lane, unlike the txt2img/edit lanes,
/// recorded NO peak before this, so a repeated control render could not reclaim the first render's
/// dropped-but-pooled pages). Mirrors [`fit_ladder`]'s own arithmetic: the resident or staged base
/// peak, less each engaged speed/quality rung's measured saving.
///
/// Returns `None` for a non-admit (`Unknown` / `TooBig`) or an admitted-but-unmeasured peak — there is
/// nothing honest to record, exactly as base.rs records nothing for an unmeasured tier. Never
/// OVER-counts the pool the load leaves behind: an over-count would let a LATER gate credit more than is
/// actually free and over-admit an OOM, so only the ENGAGED rungs' MEASURED savings are subtracted (a
/// rung the ladder could not have engaged has an unmeasured, `None` saving and contributes no phantom
/// reduction). The base peak carries [`HEADROOM_GB`] like the txt2img `predicted_peak_gb` this mirrors,
/// so the small headroom over-count is the same one base.rs's reclaim already tolerates (bounded by the
/// next load's own headroom).
pub(crate) fn incurred_peak_gb(
    fit: &KreaControlFit,
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let KreaControlFit::Fits {
        offload_policy,
        tile_vae_decode,
        chunk_attention,
        branch_quant,
    } = fit
    else {
        return None; // Unknown / TooBig — no admitted load to record.
    };
    // The base peak the ladder admitted at: the resident whole-model peak, or the staged working set.
    let mut peak = if *offload_policy == OffloadPolicy::Resident {
        predicted_control_peak_gb(manifest_entry, tier_key)?
    } else {
        predicted_control_sequential_peak_gb(manifest_entry, tier_key)?
    };
    if *tile_vae_decode {
        peak -= decode_tile_save_gb(manifest_entry, tier_key).unwrap_or(0.0);
    }
    if *chunk_attention {
        peak -= chunk_attn_save_gb(manifest_entry).unwrap_or(0.0);
    }
    match branch_quant {
        Some(Quant::Q8) => peak -= branch_quant_save_gb(manifest_entry, "q8").unwrap_or(0.0),
        Some(Quant::Q4) => peak -= branch_quant_save_gb(manifest_entry, "q4").unwrap_or(0.0),
        _ => {}
    }
    Some(peak.max(0.0))
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
                    "sequentialPeakGbByTier": { "q4": 25.0, "q8": 30.0, "bf16": 40.0 },
                    // VAE-decode tiling saving (sc-11744); measured only for the constrained q4 tier here
                    // (the decode spike is the peak there). Illustrative test value, not the shipped one.
                    "decodeTileSaveGb": { "q4": 6.9 },
                    // measured sc-11743 (dense base isolates the branch).
                    "branchQuantSaveGb": { "q8": 8.4, "q4": 10.2 }
                }
            }
        }))
    }

    /// [`krea_manifest`] plus a measured scalar `chunkAttnSaveGb` (sc-11745, candle-gen #496) — exercises
    /// the activation-chunking rung between VAE-decode tiling and branch-quant. `2.43` mirrors the shipped
    /// measured Δ (RTX PRO 6000, dense bf16 base, 1024²/8-step).
    fn krea_manifest_with_chunking() -> JsonObject {
        let mut m = krea_manifest();
        m.get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("control"))
            .and_then(Value::as_object_mut)
            .expect("control block")
            .insert("chunkAttnSaveGb".to_owned(), json!(2.43));
        m
    }

    /// Read the q4-tier rung savings the ladder tests exercise: (tiling, chunking, q8-branch, q4-branch).
    /// [`krea_manifest`] measures no chunking (the scalar `chunkAttnSaveGb` is absent), so `chunk` is None
    /// here — the existing tests stay a tiling→quant regression walk; the activation-chunking rung has its
    /// own chunk-enabled manifest + tests below ([`chunking_is_the_rung_before_branch_quant`]).
    fn q4_saves(m: &JsonObject) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        (
            decode_tile_save_gb(m, "q4"),
            chunk_attn_save_gb(m),
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
    fn predicted_control_sequential_peak_reads_the_tier_plus_headroom() {
        let m = krea_manifest();
        assert_eq!(predicted_control_sequential_peak_gb(&m, "q4"), Some(27.0));
        assert_eq!(predicted_control_sequential_peak_gb(&m, "bf16"), Some(42.0));
        assert_eq!(predicted_control_sequential_peak_gb(&m, "nvfp4"), None);
        assert_eq!(
            predicted_control_sequential_peak_gb(&obj(json!({})), "q4"),
            None
        );
    }

    #[test]
    fn sequential_is_the_first_rung_and_has_a_clean_second_stage_reject() {
        let resident = Some(40.0);
        let sequential = Some(30.0);

        assert_eq!(
            super::fit_ladder(
                resident,
                sequential,
                Some(budget(35.0)),
                None,
                None,
                None,
                None,
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                branch_quant: None,
            }
        );
        assert_too_big(
            super::fit_ladder(
                resident,
                sequential,
                Some(budget(29.0)),
                None,
                None,
                None,
                None,
            ),
            30.0,
            29.0,
        );
    }

    #[test]
    fn missing_sequential_measurement_keeps_best_effort_staging() {
        assert_eq!(
            super::fit_ladder(
                Some(40.0),
                None,
                Some(budget(20.0)),
                Some(5.0),
                Some(2.0),
                Some(8.0),
                Some(10.0),
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                branch_quant: None,
            }
        );
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

    /// Keep the pre-sc-12176 rung tests focused on tiling/chunking/branch quant by modeling a staged
    /// peak equal to the old resident peak. Dedicated tests below cover the residency reduction itself.
    fn fit_ladder(
        peak: Option<f64>,
        budget: Option<VramBudget>,
        tile: Option<f64>,
        chunk: Option<f64>,
        q8: Option<f64>,
        q4: Option<f64>,
    ) -> KreaControlFit {
        super::fit_ladder(peak, peak, budget, tile, chunk, q8, q4)
    }

    /// The big-card fast path: untiled monolithic decode, bf16 branch, no rung engaged.
    fn fits_untiled_bf16() -> KreaControlFit {
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        }
    }

    #[test]
    fn big_card_keeps_the_bf16_branch_with_no_penalty() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, chunk, q8, q4) = q4_saves(&m);
        // 96 GB card: the monolithic peak fits outright — nothing engages (no tiling, no quant).
        assert_eq!(
            fit_ladder(peak, Some(budget(90.0)), tile, chunk, q8, q4),
            fits_untiled_bf16()
        );
    }

    #[test]
    fn vae_tiling_is_the_cheapest_rung_and_keeps_the_bf16_branch() {
        let m = krea_manifest();
        // q4 base: monolithic peak 30.9 + 2 = 32.9; tiling saves 6.9 ⇒ tiled peak 26.0.
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, chunk, q8, q4) = q4_saves(&m);
        // 26 GB card: monolithic (32.9) won't fit, but the tiled decode (26.0) does ⇒ tile on, bf16 branch
        // kept (no quality penalty). Without sc-11744 this card would have dropped to a q8 branch.
        assert_eq!(
            fit_ladder(peak, Some(budget(26.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                branch_quant: None,
            }
        );
    }

    #[test]
    fn tiling_composes_with_branch_quant_when_still_constrained() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4"); // 32.9; tiled 26.0
        let (tile, chunk, q8, q4) = q4_saves(&m);
        // 24 GB card: tiled peak 26.0 > 24; tiling stays on and q8 engages (26.0 − 8.4 = 17.6 ≤ 24) ⇒
        // tiling + q8 (near-lossless) instead of the old tiling-less q4-branch drop.
        assert_eq!(
            fit_ladder(peak, Some(budget(24.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                branch_quant: Some(Quant::Q8),
            }
        );
        // 16 GB card: tiling + q8 (17.6) > 16; tiling + q4 (26.0 − 10.2 = 15.8 ≤ 16) fits ⇒ the deepest
        // rung. Without the tiling rung this card OOM-rejected (q4-alone peak 22.7 > 16).
        assert_eq!(
            fit_ladder(peak, Some(budget(16.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                branch_quant: Some(Quant::Q4),
            }
        );
        // 15 GB card: even tiling + q4 (15.8) won't fit ⇒ reject-before-OOM at the best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(15.0)), tile, chunk, q8, q4),
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
        let chunk = chunk_attn_save_gb(&m); // None (base manifest measures no chunking)
        let q8 = branch_quant_save_gb(&m, "q8"); // 8.4
        let q4 = branch_quant_save_gb(&m, "q4"); // 10.2

        // 48.2 fits ⇒ untiled bf16 branch.
        assert_eq!(
            fit_ladder(peak, Some(budget(49.0)), tile, chunk, q8, q4),
            fits_untiled_bf16()
        );
        // 48.2 > 41, 48.2 − 8.4 = 39.8 ≤ 41 ⇒ q8 (preferred, near-lossless), tiling unavailable.
        assert_eq!(
            fit_ladder(peak, Some(budget(41.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                branch_quant: Some(Quant::Q8),
            }
        );
        // 39.8 > 39, 48.2 − 10.2 = 38.0 ≤ 39 ⇒ q4 (last resort).
        assert_eq!(
            fit_ladder(peak, Some(budget(39.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                branch_quant: Some(Quant::Q4),
            }
        );
        // Even q4 (~38.0) won't fit a 30 GB budget ⇒ reject, reporting the best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(30.0)), tile, chunk, q8, q4),
            38.0,
            30.0,
        );
    }

    #[test]
    fn missing_signal_never_blocks() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, chunk, q8, q4) = q4_saves(&m);
        // No live budget ⇒ Unknown (never block).
        assert_eq!(
            fit_ladder(peak, None, tile, chunk, q8, q4),
            KreaControlFit::Unknown
        );
        // No measured peak ⇒ Unknown.
        assert_eq!(
            fit_ladder(None, Some(budget(8.0)), tile, chunk, q8, q4),
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
            fit_ladder(peak, Some(budget(20.0)), None, None, None, None),
            32.9,
            20.0,
        );
    }

    #[test]
    fn chunk_attn_save_reads_the_measured_scalar() {
        // Present (chunk-enabled manifest) ⇒ the measured scalar; tier-independent (no tier arg).
        assert_eq!(
            chunk_attn_save_gb(&krea_manifest_with_chunking()),
            Some(2.43)
        );
        // Absent in the base manifest ⇒ None (the chunking rung is unavailable).
        assert_eq!(chunk_attn_save_gb(&krea_manifest()), None);
        // No control block / no candle block ⇒ None.
        assert_eq!(chunk_attn_save_gb(&obj(json!({}))), None);
    }

    #[test]
    fn chunking_is_the_rung_before_branch_quant() {
        let m = krea_manifest_with_chunking();
        // q4 base: monolithic peak 32.9; tiled 26.0; tiled + chunked 26.0 − 2.43 = 23.57.
        let peak = predicted_control_peak_gb(&m, "q4");
        let (tile, chunk, q8, q4) = q4_saves(&m); // chunk now Some(2.43)

        // 24 GB card: tiling alone (26.0) > 24, but tiling + chunking (23.57 ≤ 24) fits ⇒ the bf16 branch is
        // KEPT (both cheaper rungs are speed-only, byte-identical). Without sc-11745 this dropped to q8.
        assert_eq!(
            fit_ladder(peak, Some(budget(24.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
                branch_quant: None,
            }
        );
        // 16 GB card: tiling + chunking (23.57) > 16; both stay on beneath q8 (23.57 − 8.4 = 15.17 ≤ 16) ⇒
        // tiling + chunking + q8 (near-lossless) — a shallower branch quant than the chunk-less walk needed
        // (which reached q4 at 16 GB, see tiling_composes_with_branch_quant_when_still_constrained).
        assert_eq!(
            fit_ladder(peak, Some(budget(16.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
                branch_quant: Some(Quant::Q8),
            }
        );
        // 14 GB card: q8 (15.17) > 14; q4 (23.57 − 10.2 = 13.37 ≤ 14) fits ⇒ every cheaper rung on + q4.
        assert_eq!(
            fit_ladder(peak, Some(budget(14.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
                branch_quant: Some(Quant::Q4),
            }
        );
        // 13 GB card: even (tiling + chunking + q4) = 13.37 won't fit ⇒ reject at the best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(13.0)), tile, chunk, q8, q4),
            13.37,
            13.0,
        );
    }

    #[test]
    fn chunking_engages_without_tiling_when_the_tier_has_no_tiling_measurement() {
        let m = krea_manifest_with_chunking();
        // The bf16 base tier carries no `decodeTileSaveGb` (the decode spike isn't its peak), but the SCALAR
        // chunk saving applies to every tier. Peak 48.2; chunked (tiling unavailable) 48.2 − 2.43 = 45.77.
        let peak = predicted_control_peak_gb(&m, "bf16");
        let tile = decode_tile_save_gb(&m, "bf16"); // None
        let chunk = chunk_attn_save_gb(&m); // Some(2.43)
        let q8 = branch_quant_save_gb(&m, "q8");
        let q4 = branch_quant_save_gb(&m, "q4");
        // 46 GB card: monolithic (48.2) > 46; tiling unavailable; chunking alone (45.77 ≤ 46) fits ⇒ chunk
        // on, tiling off, bf16 branch kept — the denoise-peak rung standing in for the absent decode rung.
        assert_eq!(
            fit_ladder(peak, Some(budget(46.0)), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: true,
                branch_quant: None,
            }
        );
    }

    /// sc-13960: [`incurred_peak_gb`] records the peak a control render actually leaves in the cudarc
    /// pool — the resident/staged base peak less every ENGAGED rung's measured saving — so a later gate's
    /// reclaim credit is honest. It must mirror the ladder and never OVER-count (which would let a later
    /// gate over-admit an OOM).
    #[test]
    fn incurred_peak_mirrors_the_ladder_and_never_over_counts() {
        let m = krea_manifest_with_chunking();
        let tier = "q4";

        // Fast-path resident admit → exactly the resident control peak (with headroom).
        let resident = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        };
        assert_eq!(
            incurred_peak_gb(&resident, &m, tier),
            predicted_control_peak_gb(&m, tier) // 32.9
        );

        // Staged + tiled + chunked + q4: the sequential peak (27.0) less every engaged measured saving.
        let laddered = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            branch_quant: Some(Quant::Q4),
        };
        let expected = predicted_control_sequential_peak_gb(&m, tier).unwrap()
            - decode_tile_save_gb(&m, tier).unwrap()
            - chunk_attn_save_gb(&m).unwrap()
            - branch_quant_save_gb(&m, "q4").unwrap(); // 27 − 6.9 − 2.43 − 10.2 = 7.47
        assert!((incurred_peak_gb(&laddered, &m, tier).unwrap() - expected).abs() < 1e-6);

        // Non-admits record nothing (no pooled pages to reclaim).
        assert_eq!(incurred_peak_gb(&KreaControlFit::Unknown, &m, tier), None);
        assert_eq!(
            incurred_peak_gb(
                &KreaControlFit::TooBig {
                    needed_gb: 50.0,
                    available_gb: 10.0
                },
                &m,
                tier
            ),
            None
        );

        // An admitted-but-unmeasured staged peak → None, not a guess (mirrors base.rs's None ⇒ no-op).
        let no_seq = obj(json!({ "candle": { "control": { "peakGbByTier": { "q4": 30.9 } } } }));
        let staged = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            branch_quant: None,
        };
        assert_eq!(incurred_peak_gb(&staged, &no_seq, "q4"), None);

        // Safety: whatever the ladder ADMITS, the peak recorded for it never exceeds the budget it was
        // admitted against — so the pool the load leaves behind is never over-reported.
        let b = budget(24.0);
        let fit = super::fit_ladder(
            predicted_control_peak_gb(&m, tier),
            predicted_control_sequential_peak_gb(&m, tier),
            Some(b),
            decode_tile_save_gb(&m, tier),
            chunk_attn_save_gb(&m),
            branch_quant_save_gb(&m, "q8"),
            branch_quant_save_gb(&m, "q4"),
        );
        if let Some(p) = incurred_peak_gb(&fit, &m, tier) {
            assert!(
                p <= b.free_gb + 1e-6,
                "incurred peak {p} must not exceed the admitting budget {}",
                b.free_gb
            );
        }
    }

    /// sc-13960: on a warm worker, crediting the cudarc pool the previous control render left behind
    /// FLIPS the ladder off its needless rungs — the repeated-control-render scenario the story names.
    /// Pins the arithmetic the two-pass evict-reclaim gate performs (`fit_ladder(raw)` vs
    /// `fit_ladder(with_reclaimable(raw, pool))`).
    #[test]
    fn reclaim_flips_a_warm_control_gate_back_to_resident() {
        let m = krea_manifest();
        let tier = "bf16"; // control peak 48.2, sequential 42.0, no tiling rung for this tier
        let peak = predicted_control_peak_gb(&m, tier);
        let seq = predicted_control_sequential_peak_gb(&m, tier);
        let tile = decode_tile_save_gb(&m, tier); // None
        let chunk = chunk_attn_save_gb(&m); // None
        let q8 = branch_quant_save_gb(&m, "q8");
        let q4 = branch_quant_save_gb(&m, "q4");

        // Second bf16 control render on a 96 GB card: the first (48.2 peak) dropped but its pages are
        // pooled, so RAW free is ~47.8 GB — the resident peak no longer fits, so the ladder needlessly
        // stages sequentially.
        let raw = VramBudget {
            free_gb: 47.8,
            total_gb: 96.0,
        };
        assert_eq!(
            super::fit_ladder(peak, seq, Some(raw), tile, chunk, q8, q4),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                branch_quant: None,
            }
        );
        // Crediting the 48.2 GB the first render left in-pool readmits it at the big-card fast path.
        let reclaimed = crate::vram_gate::with_reclaimable(raw, 48.2);
        assert_eq!(
            super::fit_ladder(peak, seq, Some(reclaimed), tile, chunk, q8, q4),
            fits_untiled_bf16()
        );

        // A cold pool (reclaimable 0) is a no-op: the raw plan stands (a genuine first-load on a
        // constrained card is gated exactly as before).
        let reclaimed_cold = crate::vram_gate::with_reclaimable(raw, 0.0);
        assert_eq!(
            super::fit_ladder(peak, seq, Some(reclaimed_cold), tile, chunk, q8, q4),
            super::fit_ladder(peak, seq, Some(raw), tile, chunk, q8, q4)
        );
    }
}
