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
//! The control lane's peak is the base tier + the control branch (at the tier
//! [`gen_core::tier_integrity::control_branch_tier`] assigns it) + activations + the end-of-render
//! VAE-decode spike. When that peak won't fit, rungs engage in increasing (Δcost) order, stopping as
//! soon as the predicted peak ≤ budget:
//!  1. **packed base** — default, ~free (candle-gen #480, shipped). Always on; the peak below already
//!     reflects it.
//!  2. **sequential residency** (sc-12176) — encode/drop Qwen3-VL before loading the heavy phase. The
//!     cheapest adaptation and no quality cost; its measured per-tier peak enables a clean second-stage
//!     reject when even the staged working set will not fit.
//!  3. **VAE-decode tiling** (sc-11744, candle-gen #492) — force the seam-free tiled VAE tail (a *speed*
//!     cost, no quality cost) to cap the end-of-render decode spike. Engaged after residency, when the
//!     measured `decodeTileSaveGb` is enough to fit; also stays on underneath the chunking rung (it is
//!     cheaper and every bit helps).
//!  4. **activation chunking** (sc-11745, candle-gen #496) — engage sc-6217-style query-row attention
//!     chunking on the composable base stack + control branch when the denoise steady state is the
//!     overflow. A *speed* cost (~+6%) with byte-identical output; the deepest rung this lane has.
//!     Gated by the presence of the measured scalar `chunkAttnSaveGb`.
//!
//! `decodeTileSaveGb` / `chunkAttnSaveGb` absent (unmeasured) ⇒ that rung is skipped — the ladder walks
//! the rungs it *can* measure and, if even (sequential residency + tiling + chunking) won't fit,
//! rejects-before-OOM at the honest best-case peak.
//!
//! ## The branch tier is NOT a rung (sc-15799)
//!
//! There used to be a fifth, last-resort rung: quantize the control branch bf16 → q8 → q4. It is gone.
//! **No component is resident above the user's selected tier unless a declared, measured exception says
//! otherwise** (`docs/memory-strategy-contract.md`, "Tier integrity"), so the branch's tier follows the
//! base tier — with the one declared exception that a q4 base floors its branch at q8, because a q4
//! control residual measures "pose-locked; non-pose details drift". As a rung it sat LAST, which meant a
//! constrained card staged the text phase, tiled its decode and chunked its attention to claw back
//! single-digit GB while 8.4 GB of precision a q8 render never asked for sat in the branch the whole
//! time. [`control_branch_tier_for_key`] is now the whole decision, and it reads no budget.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and the
//! wiring is in `generate_candle_krea_control_stream` (image_jobs/krea_control_candle.rs).

use super::*;
use gen_core::{OffloadPolicy, Quant};
#[cfg(test)]
use serde_json::Value;

/// Fixed transient/runtime headroom (GB) added on top of the control-lane peak
/// ([`predicted_control_peak_gb`]), mirroring [`crate::vram_gate`]'s `HEADROOM_GB` — covers allocator
/// slack + activation spikes not captured by the steady peak.
const HEADROOM_GB: f64 = 2.0;

/// The outcome of walking the Krea control fit ladder. `Unknown` = no signal (no `candle.control` block,
/// or a non-NVIDIA host) ⇒ never block — exactly like [`crate::vram_gate::FitDecision::Unknown`].
///
/// No variant carries the branch tier: it is not a rung outcome (sc-15799). The caller derives it from
/// the resolved base tier with [`control_branch_tier_for_key`] on *every* path, `Unknown` included, so a
/// missing manifest signal can never leave an above-tier branch resident.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KreaControlFit {
    /// No measured control peak or no live budget ⇒ don't gate.
    Unknown,
    /// The predicted peak fits after engaging the minimal sufficient set of rungs. The big-card fast
    /// path uses resident components with `{ tile_vae_decode: false, chunk_attention: false }`
    /// (monolithic full-speed decode, unchunked attention, zero penalty). Sequential residency is the
    /// first adaptation; `tile_vae_decode = true` is the next rung (sc-11744 — a *speed* cost only,
    /// seam-free); `chunk_attention = true` is the deepest (sc-11745 — a *speed* cost only,
    /// byte-identical output).
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
    },
    /// Won't fit even at the deepest rung. Reject-before-OOM with an actionable message rather than a
    /// reactive CUDA OOM mid-render. `needed_gb` is the best-case sequential + tiled + chunked predicted
    /// peak.
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// The [`Quant`] a control-lane tier KEY names — the `candle.control.*ByTier` / `vramGbByTier` key space
/// (`bf16` / `q8` / `q4` / `nvfp4` / `int8-convrot`), not a bit width.
///
/// `int8-convrot` (sc-9300) maps to [`Quant::Q8`]: it is an int8 weight regime with online rotation, so
/// for the purpose of "what precision may a component be resident at" it sits exactly where q8 does. An
/// unrecognized key is treated as dense, the conservative direction — a tier we cannot classify must not
/// be reported as *below* something, or the branch would be packed further than the base.
fn quant_for_tier_key(tier_key: &str) -> Option<Quant> {
    match tier_key {
        "q8" | "int8-convrot" => Some(Quant::Q8),
        "q4" => Some(Quant::Q4),
        "nvfp4" => Some(Quant::Nvfp4),
        _ => None,
    }
}

/// The tier the pose-control branch is packed to for a resolved base tier KEY (sc-15799) — the shared
/// [`gen_core::tier_integrity::control_branch_tier`] rule, keyed off the manifest's tier names.
///
/// This is the whole branch-tier decision on the candle lane. It reads no budget and returns the same
/// answer on a 16 GB card and a 96 GB card, because the user's tier choice — not free VRAM — is what
/// decides how much precision the control residual is allowed to hold. `None` = dense branch, which
/// happens only for a dense base.
pub(crate) fn control_branch_tier_for_key(tier_key: &str) -> Option<Quant> {
    gen_core::tier_integrity::control_branch_tier(quant_for_tier_key(tier_key))
}

/// The manifest key under `candle.control.branchPackSaveGb` for a branch [`Quant`], or `None` for a
/// dense branch (nothing was packed, so nothing was saved).
fn branch_pack_key(branch_tier: Option<Quant>) -> Option<&'static str> {
    match branch_tier {
        Some(Quant::Q8) => Some("q8"),
        Some(Quant::Q4) => Some("q4"),
        // NVFP4 has no published branch artifact and `control_branch_tier` never returns it.
        Some(Quant::Nvfp4) | None => None,
    }
}

/// Read one `candle.control.<key>[tier_key]` number.
fn control_tier_gb(manifest_entry: &JsonObject, key: &str, tier_key: &str) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get(key)
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
}

/// Measured weight-side reduction (GB) from packing the bf16 control branch to `q8`/`q4`:
/// `candle.control.branchPackSaveGb[quant]`. Published measured (sc-11743, candle-gen #483, dense base
/// so the rest is held fixed and only the branch moves): q8 −8.4, q4 −10.2 GB.
///
/// Formerly `branchQuantSaveGb`, read as the *saving of a rung*. It is now the packing delta that
/// converts the superseded bf16-branch rows into the configuration that actually ships (sc-15799) — see
/// [`predicted_control_peak_gb`]. Renamed rather than reused so a stale reader cannot keep treating it
/// as a lever's budget.
fn branch_pack_save_gb(manifest_entry: &JsonObject, quant: &str) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("branchPackSaveGb")
        .and_then(|saves| saves.get(quant))
        .and_then(json_f64)
}

/// The branch-packing correction (GB) to subtract from a `bf16Branch*` row for `tier_key`: the measured
/// [`branch_pack_save_gb`] of the tier [`control_branch_tier_for_key`] assigns, or `0.0` when the branch
/// is dense (nothing packed) or the delta is unmeasured.
///
/// `0.0` on an unmeasured delta is the CONSERVATIVE direction: the uncorrected bf16-branch row is an
/// upper bound on the packed configuration, so the gate over-predicts and adapts/rejects slightly early
/// rather than admitting a load that can OOM.
fn branch_pack_correction_gb(manifest_entry: &JsonObject, tier_key: &str) -> f64 {
    branch_pack_key(control_branch_tier_for_key(tier_key))
        .and_then(|quant| branch_pack_save_gb(manifest_entry, quant))
        .unwrap_or(0.0)
}

/// Predicted control-lane peak VRAM (GB) for `tier_key` (the BASE tier — `bf16`/`q8`/`q4`/…) with no
/// rungs engaged: `candle.control.bf16BranchPeakGbByTier[tier_key]` less the measured branch-packing
/// delta for the tier the branch now loads at, plus [`HEADROOM_GB`]; `None` (absent ⇒ the gate no-ops).
/// The control peak exceeds the txt2img `candle.vramGbByTier` because the control branch is co-resident.
///
/// **Why the row is named for a branch tier that no longer ships.** Every control-lane peak in the
/// catalog was captured with a **bf16** branch (sc-12176's two-process probe). sc-15799 packs the branch
/// to the base tier, so those rows no longer describe any shipping configuration and are renamed to say
/// so; `candle.control.measured` is `false` until a control-lane re-measure lands. What this function
/// returns is therefore a **derived** estimate, not a measurement — but it is derived by exactly the
/// arithmetic the shipped ladder already performed when its branch-quant rung engaged (row − the measured
/// packing delta), so it composes two measured quantities and invents nothing. It is deliberately still
/// wired: a stale row that is a sound upper bound is better protection for a constrained card than no
/// signal at all, and the naming makes it impossible to mistake for current calibrated evidence.
pub(crate) fn predicted_control_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let bf16_branch_row = control_tier_gb(manifest_entry, "bf16BranchPeakGbByTier", tier_key)?;
    Some(
        (bf16_branch_row - branch_pack_correction_gb(manifest_entry, tier_key)).max(0.0)
            + HEADROOM_GB,
    )
}

/// Predicted Sequential control-lane peak for `tier_key`:
/// `candle.control.bf16BranchSequentialPeakGbByTier[tier_key]`, corrected and headroomed exactly like
/// [`predicted_control_peak_gb`]. This is the largest single working set after Qwen3-VL has been encoded
/// and dropped. `None` preserves best-effort offload behavior: the lane still stages, but cannot perform
/// an honest second-stage reject.
pub(crate) fn predicted_control_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let bf16_branch_row =
        control_tier_gb(manifest_entry, "bf16BranchSequentialPeakGbByTier", tier_key)?;
    Some(
        (bf16_branch_row - branch_pack_correction_gb(manifest_entry, tier_key)).max(0.0)
            + HEADROOM_GB,
    )
}

/// Measured peak reduction (GB) from forcing the seam-free tiled VAE decode for BASE tier `tier_key`
/// (sc-11744, candle-gen #492): `candle.control.decodeTileSaveGb[tier_key]` — the monolithic whole-render
/// peak minus the tiled one (the decode spike capped toward the denoise steady state). `None` when
/// unmeasured for that tier ⇒ the VAE-tiling rung is unavailable there and the ladder walks straight to
/// attention chunking. Tier-keyed like [`predicted_control_peak_gb`] because whether the decode spike
/// *is* the global peak (and thus how much tiling recovers) depends on the tier's denoise-steady floor.
///
/// Carried across sc-15799 unchanged. It is a DELTA between a monolithic and a tiled decode of the same
/// load, so the resident branch is present in both arms and cancels; a lighter branch lowers the denoise
/// floor the spike is measured against, which if anything makes the decode spike *more* dominant, so the
/// delta does not shrink. (`candle.control.measured` is nonetheless `false` until the control lane is
/// re-measured — this is a reasoned carry-over, not fresh evidence.)
pub(crate) fn decode_tile_save_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    control_tier_gb(manifest_entry, "decodeTileSaveGb", tier_key)
}

/// Measured peak reduction (GB) from engaging sc-6217-style query-row attention chunking on the composable
/// base stack + control branch (sc-11745, candle-gen #496): `candle.control.chunkAttnSaveGb`. Bounds the
/// DENOISE-phase activation peak — a *speed* cost (~+6%) with byte-identical output (the chunked forward is
/// numerically identical to the unchunked one, the sc-6217 query-row-independence invariant). Unlike the
/// tier-keyed [`decode_tile_save_gb`]/[`predicted_control_peak_gb`], this is a SCALAR: the denoise activation
/// peak is bf16 regardless of the base weight tier, so the saving is tier-independent. `None` when unmeasured
/// ⇒ the chunking rung is unavailable and the ladder can only reject. Published measured (RTX PRO
/// 6000, dense bf16 base, 1024²/8-step): −2.43 GiB.
pub(crate) fn chunk_attn_save_gb(manifest_entry: &JsonObject) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("chunkAttnSaveGb")
        .and_then(json_f64)
}

/// Walk the fit ladder: given the predicted resident and staged peaks (already corrected for the branch
/// tier by [`predicted_control_peak_gb`]), the (capped) live budget, and the measured VAE-decode tiling
/// and activation-chunking savings, engage the minimal sufficient set of rungs in increasing (Δcost)
/// order and return the [`KreaControlFit`].
///
/// Walk: if the monolithic peak fits, run resident/untiled/unchunked (big-card fast path, no penalty);
/// else stage text and heavy components sequentially. If the staged peak still exceeds the budget, force
/// the seam-free tiled decode (sc-11744 — a *speed* cost only); then, keeping tiling on, engage query-row
/// attention chunking (sc-11745 — a *speed* cost only, byte-identical); else reject-before-OOM. Any
/// unmeasured saving (`None`) skips its rung. Missing resident peak or budget ⇒
/// [`KreaControlFit::Unknown`] (never block).
///
/// There is no branch-quant rung (sc-15799): the branch's tier is fixed by the base tier before this
/// runs, so both peaks arrive already priced at it and there is nothing left for the ladder to trade.
pub(crate) fn fit_ladder(
    peak_gb: Option<f64>,
    sequential_peak_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    chunk_attn_save_gb: Option<f64>,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let free = budget.free_gb;
    let fits = |needed: f64| free + f64::EPSILON >= needed;

    // Rung 1 (packed base, always on) — big-card fast path: the monolithic peak already fits, so nothing
    // engages: full-speed decode, unchunked attention, zero penalty.
    if fits(peak) {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
        };
    }

    // Rung 2 (sequential residency, sc-12176) — the cheapest adaptation. If its measured peak is
    // absent, preserve the shared gate's best-effort contract: stage anyway and rely on the reactive
    // CUDA-OOM backstop rather than engaging costlier rungs from an unknown baseline.
    let Some(peak) = sequential_peak_gb else {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
        };
    };
    if fits(peak) {
        return KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
        };
    }
    // Rung 3 (VAE-decode tiling, sc-11744) — a *speed* cost, no quality loss (seam-free). An
    // unmeasured tier saving (None) leaves this rung unavailable.
    if let Some(tile_save) = decode_tile_save_gb {
        if fits(peak - tile_save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
            };
        }
    }
    // Past here tiling alone was insufficient, so keep it on beneath the deeper rung (it is cheaper than
    // chunking and every GB helps); if it was unmeasured, `tile_on` stays false / the peak unreduced.
    let tile_on = decode_tile_save_gb.is_some();
    let peak_after_tile = peak - decode_tile_save_gb.unwrap_or(0.0);

    // Rung 4 (activation chunking, sc-11745, candle-gen #496) — a *speed* cost (~+6%), byte-identical
    // output (the chunked forward is numerically identical). The DEEPEST rung this lane has, now that the
    // branch tier is not a rung (sc-15799). An unmeasured saving (None) leaves it unavailable.
    if let Some(chunk_save) = chunk_attn_save_gb {
        if fits(peak_after_tile - chunk_save) {
            return KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: tile_on,
                chunk_attention: true,
            };
        }
    }
    // Won't fit even at (staged + tiling + chunking). Report the best-case peak (every rung on) so the
    // reject message is honest about what still overflows.
    let best_case = peak_after_tile - chunk_attn_save_gb.unwrap_or(0.0);
    KreaControlFit::TooBig {
        needed_gb: best_case,
        available_gb: free,
    }
}

/// The VRAM peak (GB) a Krea control render actually incurs under `fit`, for the reclaimable high-water
/// ([`crate::vram_gate::note_loaded_peak`], sc-13960 — the control lane, unlike the txt2img/edit lanes,
/// recorded NO peak before this, so a repeated control render could not reclaim the first render's
/// dropped-but-pooled pages). Mirrors [`fit_ladder`]'s own arithmetic: the resident or staged base
/// peak (already branch-tier-corrected), less each engaged speed rung's measured saving.
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
    // No branch-tier term: `predicted_control_*_peak_gb` already applied the branch-packing correction,
    // so subtracting it again here would double-count and under-report the pool this load leaves behind
    // (which would let a later gate credit more than is free and over-admit).
    Some(peak.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vram_gate::VramBudget;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    /// A Krea-shaped control block in the post-sc-15799 schema. Illustrative values, NOT the shipped
    /// ones — the shipped numbers are pinned by `tests/gpu_and_manifest.rs`.
    fn krea_manifest() -> JsonObject {
        obj(json!({
            "candle": {
                "minMemoryGb": 32,
                "vramGbByTier": { "q4": 26.4, "q8": 31.0, "bf16": 44.0 },
                "control": {
                    // Captured with a bf16 branch (sc-12176) — superseded by sc-15799, retained as the
                    // bound the branch-packing correction is applied to.
                    "bf16BranchPeakGbByTier": { "q4": 30.9, "q8": 35.5, "bf16": 46.2 },
                    "bf16BranchSequentialPeakGbByTier": { "q4": 25.0, "q8": 30.0, "bf16": 40.0 },
                    // VAE-decode tiling saving (sc-11744); measured only for the constrained q4 tier here
                    // (the decode spike is the peak there).
                    "decodeTileSaveGb": { "q4": 6.9 },
                    // Measured bf16 → q8/q4 branch packing delta (sc-11743, dense base isolates it).
                    "branchPackSaveGb": { "q8": 8.4, "q4": 10.2 }
                }
            }
        }))
    }

    /// [`krea_manifest`] plus a measured scalar `chunkAttnSaveGb` (sc-11745, candle-gen #496) — the
    /// deepest rung. `2.43` mirrors the shipped measured Δ (RTX PRO 6000, dense bf16 base, 1024²/8-step).
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

    fn budget(free: f64) -> VramBudget {
        VramBudget {
            free_gb: free,
            total_gb: free,
        }
    }

    /// Keep the rung tests focused on tiling/chunking by modeling a staged peak equal to the resident
    /// peak. Dedicated tests below cover the residency reduction itself.
    fn fit_ladder(
        peak: Option<f64>,
        budget: Option<VramBudget>,
        tile: Option<f64>,
        chunk: Option<f64>,
    ) -> KreaControlFit {
        super::fit_ladder(peak, peak, budget, tile, chunk)
    }

    /// The big-card fast path: monolithic decode, unchunked attention, nothing engaged.
    fn fits_nothing_engaged() -> KreaControlFit {
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
        }
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

    // ── sc-15799: the branch tier is derived from the base tier, and reads no budget. ───────────────

    /// The invariant: the branch follows the base tier, with the one declared q4 → q8 floor. `int8-convrot`
    /// is an int8 regime so its branch is q8; a dense base keeps a dense branch. A mutation restoring the
    /// old "bf16 unless constrained" behavior cannot pass this — there is no budget argument to vary.
    #[test]
    fn branch_tier_follows_the_base_tier_with_the_declared_q4_floor() {
        assert_eq!(control_branch_tier_for_key("bf16"), None);
        assert_eq!(control_branch_tier_for_key("q8"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("q4"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("int8-convrot"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("nvfp4"), Some(Quant::Q8));
        // An unclassifiable tier is treated as dense — never packed further than the base.
        assert_eq!(control_branch_tier_for_key("who-knows"), None);
    }

    /// A packed base never carries a dense branch. This is the defect sc-15799 removes, stated as a test.
    #[test]
    fn a_packed_base_never_carries_a_dense_branch() {
        for tier in ["q4", "q8", "int8-convrot", "nvfp4"] {
            assert!(
                control_branch_tier_for_key(tier).is_some(),
                "{tier} must not carry a dense control branch"
            );
        }
    }

    // ── The predicted peaks, corrected for the tier the branch now loads at. ────────────────────────

    /// The packed tiers' rows are corrected DOWN by the measured branch-packing delta (they were captured
    /// with a bf16 branch); the dense tier's row is untouched, because its branch really is dense.
    #[test]
    fn predicted_control_peak_corrects_the_bf16_branch_row_for_packed_tiers() {
        let m = krea_manifest();
        // q4 base → q8 branch: 30.9 − 8.4 + headroom.
        assert_eq!(
            predicted_control_peak_gb(&m, "q4"),
            Some(30.9 - 8.4 + HEADROOM_GB)
        );
        // q8 base → q8 branch: the same correction applies.
        assert_eq!(
            predicted_control_peak_gb(&m, "q8"),
            Some(35.5 - 8.4 + HEADROOM_GB)
        );
        // bf16 base → dense branch: nothing was packed, so nothing is subtracted.
        assert_eq!(
            predicted_control_peak_gb(&m, "bf16"),
            Some(46.2 + HEADROOM_GB)
        );
        // Tier absent from the row ⇒ None (gate no-ops for that tier).
        assert_eq!(predicted_control_peak_gb(&m, "int8-convrot"), None);
        // No control block ⇒ None (unmeasured control lane).
        let no_control = obj(json!({ "candle": { "vramGbByTier": { "q4": 26.4 } } }));
        assert_eq!(predicted_control_peak_gb(&no_control, "q4"), None);
        // No candle block ⇒ None.
        assert_eq!(predicted_control_peak_gb(&obj(json!({})), "q4"), None);
    }

    /// An UNMEASURED packing delta must leave the bf16-branch row uncorrected — the conservative
    /// direction. Correcting by a guess would under-predict, and an under-prediction admits an OOM.
    #[test]
    fn an_unmeasured_packing_delta_leaves_the_bound_uncorrected() {
        let mut m = krea_manifest();
        m.get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("control"))
            .and_then(Value::as_object_mut)
            .expect("control block")
            .remove("branchPackSaveGb");
        assert_eq!(
            predicted_control_peak_gb(&m, "q4"),
            Some(30.9 + HEADROOM_GB),
            "with no measured delta the uncorrected bf16-branch row must stand as an upper bound"
        );
    }

    #[test]
    fn predicted_control_sequential_peak_is_corrected_the_same_way() {
        let m = krea_manifest();
        // 25.0 − 8.4 + 2.0
        assert_eq!(predicted_control_sequential_peak_gb(&m, "q4"), Some(18.6));
        // 40.0 + 2.0, dense branch.
        assert_eq!(predicted_control_sequential_peak_gb(&m, "bf16"), Some(42.0));
        assert_eq!(predicted_control_sequential_peak_gb(&m, "nvfp4"), None);
        assert_eq!(
            predicted_control_sequential_peak_gb(&obj(json!({})), "q4"),
            None
        );
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

    // ── The ladder: staged residency → decode tiling → attention chunking, then reject. ─────────────

    #[test]
    fn sequential_is_the_first_rung_and_has_a_clean_second_stage_reject() {
        let resident = Some(40.0);
        let sequential = Some(30.0);

        assert_eq!(
            super::fit_ladder(resident, sequential, Some(budget(35.0)), None, None),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
            }
        );
        assert_too_big(
            super::fit_ladder(resident, sequential, Some(budget(29.0)), None, None),
            30.0,
            29.0,
        );
    }

    #[test]
    fn missing_sequential_measurement_keeps_best_effort_staging() {
        assert_eq!(
            super::fit_ladder(Some(40.0), None, Some(budget(20.0)), Some(5.0), Some(2.0)),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
            }
        );
    }

    #[test]
    fn big_card_engages_nothing() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // 96 GB card: the monolithic peak fits outright — nothing engages.
        assert_eq!(
            fit_ladder(peak, Some(budget(90.0)), tile, chunk),
            fits_nothing_engaged()
        );
    }

    /// The repack's payoff: a card that USED to need the tiling rung now clears the monolithic peak,
    /// because the branch no longer carries 8.4 GB of unrequested precision.
    #[test]
    fn packing_the_branch_admits_a_card_the_bf16_branch_would_have_laddered() {
        let m = krea_manifest();
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // Corrected q4 peak = 30.9 − 8.4 + 2.0 = 24.5. A 26 GB card fits it outright…
        assert_eq!(
            fit_ladder(
                predicted_control_peak_gb(&m, "q4"),
                Some(budget(26.0)),
                tile,
                chunk
            ),
            fits_nothing_engaged()
        );
        // …whereas the UNCORRECTED bf16-branch peak (32.9) would have forced staging + tiling on it.
        assert_eq!(
            fit_ladder(Some(30.9 + HEADROOM_GB), Some(budget(26.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
            }
        );
    }

    #[test]
    fn vae_tiling_is_the_cheapest_deeper_rung() {
        let m = krea_manifest();
        // q4 base: corrected monolithic peak 24.5; tiling saves 6.9 ⇒ tiled peak 17.6.
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // 18 GB card: monolithic (24.5) won't fit, the tiled decode (17.6) does ⇒ tiling on, no quality cost.
        assert_eq!(
            fit_ladder(peak, Some(budget(18.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
            }
        );
    }

    #[test]
    fn chunking_is_the_deepest_rung_and_then_the_ladder_rejects() {
        let m = krea_manifest_with_chunking();
        // q4 base: corrected peak 24.5; tiled 17.6; tiled + chunked 17.6 − 2.43 = 15.17.
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);

        // 16 GB card: tiling alone (17.6) > 16, tiling + chunking (15.17 ≤ 16) fits. Both rungs are
        // speed-only and byte-identical, so this card pays NO quality cost — where the old ladder
        // reached for a q4 branch at this budget.
        assert_eq!(
            fit_ladder(peak, Some(budget(16.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
            }
        );
        // 15 GB card: nothing deeper exists ⇒ reject-before-OOM at the honest best-case peak.
        assert_too_big(
            fit_ladder(peak, Some(budget(15.0)), tile, chunk),
            15.17,
            15.0,
        );
    }

    #[test]
    fn chunking_engages_without_tiling_when_the_tier_has_no_tiling_measurement() {
        let m = krea_manifest_with_chunking();
        // The bf16 base tier carries no `decodeTileSaveGb` (the decode spike isn't its peak), but the
        // SCALAR chunk saving applies to every tier. Peak 48.2; chunked 48.2 − 2.43 = 45.77.
        let peak = predicted_control_peak_gb(&m, "bf16");
        let tile = decode_tile_save_gb(&m, "bf16"); // None
        let chunk = chunk_attn_save_gb(&m); // Some(2.43)
        assert_eq!(
            fit_ladder(peak, Some(budget(46.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: true,
            }
        );
    }

    #[test]
    fn missing_signal_never_blocks() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // No live budget ⇒ Unknown (never block).
        assert_eq!(fit_ladder(peak, None, tile, chunk), KreaControlFit::Unknown);
        // No measured peak ⇒ Unknown.
        assert_eq!(
            fit_ladder(None, Some(budget(8.0)), tile, chunk),
            KreaControlFit::Unknown
        );
    }

    #[test]
    fn unmeasured_rungs_can_only_reject_at_the_raw_peak() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4"); // 24.5
                                                        // No tiling and no chunking measured ⇒ no rung is available, so a too-small card rejects
                                                        // reporting the raw peak itself (nothing to subtract).
        assert_too_big(fit_ladder(peak, Some(budget(20.0)), None, None), 24.5, 20.0);
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
        };
        assert_eq!(
            incurred_peak_gb(&resident, &m, tier),
            predicted_control_peak_gb(&m, tier) // 24.5
        );

        // Staged + tiled + chunked: the sequential peak (18.6) less every engaged measured saving. The
        // branch-packing correction is already inside that peak and must NOT be applied twice.
        let laddered = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
        };
        let expected = predicted_control_sequential_peak_gb(&m, tier).unwrap()
            - decode_tile_save_gb(&m, tier).unwrap()
            - chunk_attn_save_gb(&m).unwrap(); // 18.6 − 6.9 − 2.43 = 9.27
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
        let no_seq =
            obj(json!({ "candle": { "control": { "bf16BranchPeakGbByTier": { "q4": 30.9 } } } }));
        let staged = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
        };
        assert_eq!(incurred_peak_gb(&staged, &no_seq, "q4"), None);

        // Safety: whatever the ladder ADMITS, the peak recorded for it never exceeds the budget it was
        // admitted against — so the pool the load leaves behind is never over-reported.
        let b = budget(16.0);
        let fit = super::fit_ladder(
            predicted_control_peak_gb(&m, tier),
            predicted_control_sequential_peak_gb(&m, tier),
            Some(b),
            decode_tile_save_gb(&m, tier),
            chunk_attn_save_gb(&m),
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

        // Second bf16 control render on a 96 GB card: the first (48.2 peak) dropped but its pages are
        // pooled, so RAW free is ~47.8 GB — the resident peak no longer fits, so the ladder needlessly
        // stages sequentially.
        let raw = VramBudget {
            free_gb: 47.8,
            total_gb: 96.0,
        };
        assert_eq!(
            super::fit_ladder(peak, seq, Some(raw), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
            }
        );
        // Crediting the 48.2 GB the first render left in-pool readmits it at the big-card fast path.
        let reclaimed = crate::vram_gate::with_reclaimable(raw, 48.2);
        assert_eq!(
            super::fit_ladder(peak, seq, Some(reclaimed), tile, chunk),
            fits_nothing_engaged()
        );

        // A cold pool (reclaimable 0) is a no-op: the raw plan stands (a genuine first-load on a
        // constrained card is gated exactly as before).
        let reclaimed_cold = crate::vram_gate::with_reclaimable(raw, 0.0);
        assert_eq!(
            super::fit_ladder(peak, seq, Some(reclaimed_cold), tile, chunk),
            super::fit_ladder(peak, seq, Some(raw), tile, chunk)
        );
    }
}
