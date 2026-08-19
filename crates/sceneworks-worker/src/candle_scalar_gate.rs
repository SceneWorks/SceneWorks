//! The candle legacy SCALAR admission gate — the pure arithmetic half, lifted out of
//! [`crate::vram_gate`] so it compiles on every lane (sc-19049, epic 19048 slice 1).
//!
//! ## Why this module exists
//!
//! `vram_gate` is `#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]`: it owns the
//! CUDA probe, the `nvidia-smi` reading, the Krea runtime ladder and the video weight sizing, none
//! of which mean anything off a CUDA lane. But the part of it that decides *whether a candle image
//! request is admitted* — per-tier manifest scalar + headroom, compared to a budget — is a pure
//! function of `serde_json` data and two floats. Nothing in it touches candle, CUDA, MLX or the
//! filesystem.
//!
//! Epic 19048's first slice has to record a DECISION BASELINE that later slices go red against
//! (requirement R6). A baseline is only worth committing if it is produced by the gate the request
//! actually reaches; a re-implementation in the generator is a second law that drifts silently — it
//! reds when it disagrees with itself, never when it disagrees with production. Leaving this
//! arithmetic behind the candle cfg meant the only lane that could drive it was the self-hosted
//! `windows-candle` one, which PR-triggers on `main` only. Moving the pure half here lets
//! `candle_admission_decisions.rs` drive **these exact functions** on the ordinary `cargo test`
//! lane, on every platform.
//!
//! ## This is a MOVE, not a fork
//!
//! Every item below came out of `vram_gate.rs` verbatim, documentation included, and `vram_gate`
//! re-exports all of them (`pub(crate) use crate::candle_scalar_gate::*`), so every call site,
//! every doc link and every existing `vram_gate` unit test resolves unchanged. There is exactly one
//! definition of the candle scalar gate in the tree, and this is it.

use crate::fit_gate::{dedicated_vram_reserve, resolve_offload, FitDecision, BYTES_PER_GIB};
use crate::image_jobs::NVFP4_TIER;
use crate::payload::json_f64;
use sceneworks_core::contracts::JsonObject;

/// Fixed transient/runtime headroom (GB) added on top of a per-tier MEASURED peak (`candle.vramGbByTier`)
/// to cover allocator slack + activation spikes not captured by the steady peak. Not added on top of
/// `candle.minMemoryGb`, which the manifest already pads over the measured peak (sc-9094).
/// Candle/CUDA's dedicated-VRAM allocator/context slack. This aliases the backend-neutral typed
/// policy so the generic gate, control gate, Mochi, and the web mirror cannot assign independent
/// meanings or values to the same dedicated pool.
pub(crate) const HEADROOM_GB: f64 = dedicated_vram_reserve().gb;

/// A live (or capped) VRAM budget for the selected GPU, in GB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VramBudget {
    pub free_gb: f64,
    pub total_gb: f64,
}

/// Predicted peak VRAM (GB) for `tier_key` from the manifest `candle` block:
/// `candle.vramGbByTier[tier_key]` (measured peak) + [`HEADROOM_GB`], else `candle.minMemoryGb`
/// (already padded), else `None` — an unmeasured model (no `candle` block, e.g. the dense sc-3675
/// families) skips the gate entirely.
///
/// **An `nvfp4` tier with no measured row degrades to the `q8` row, NOT to `minMemoryGb` (sc-11042).**
/// sc-11043 owns the convert-at-install loop and **must add an `nvfp4` row to `vramGbByTier` when it
/// converts a tier** — this is the documented behavior until it does. `minMemoryGb` is the WRONG floor
/// to land on here: per the manifest schema it is "the measured overall-peak of the DEFAULT (lightest,
/// typically q4) hosted tier", which heavier tiers are explicitly allowed to exceed — so falling
/// through to it would size an FP4 render against the lightest tier's number and fail PERMISSIVELY
/// (an under-prediction admits a load that can OOM). The `q8` row instead OVER-predicts (q8's weights
/// are ~2× NVFP4's ~4.5 effective bits), which is both safe and exactly the number this gate already
/// used for an NVFP4 request before the tier had its own key — the bits-derived
/// [`crate::vram_gate::requested_tier_key`] returned `"q8"` for it. So the sizing is no less
/// conservative than the status quo, and a missing row can only ever cost a spurious
/// `TooBig`/`Offload`, never an OOM.
pub(crate) fn predicted_peak_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    let candle = manifest_entry.get("candle")?;
    let measured = |key: &str| {
        candle
            .get("vramGbByTier")
            .and_then(|tiers| tiers.get(key))
            .and_then(json_f64)
    };
    if let Some(gb) = measured(tier_key) {
        return Some(gb + HEADROOM_GB);
    }
    // NVFP4 with no measured row → the q8 row (a deliberate over-prediction; see the note above).
    if tier_key == NVFP4_TIER {
        if let Some(gb) = measured("q8") {
            return Some(gb + HEADROOM_GB);
        }
    }
    candle.get("minMemoryGb").and_then(json_f64)
}

/// Resident prediction with a load-exact independently resident adapter stack. Callers pass zero
/// for a dense/folded load; packed providers pass the measured source bytes retained as residuals.
pub(crate) fn predicted_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Predicted SEQUENTIAL peak VRAM (GB) for `tier_key`: `candle.sequentialPeakGb[tier_key]` (the measured
/// largest single working set of the sequential-residency path, sc-10856) + [`HEADROOM_GB`], mirroring
/// [`predicted_peak_gb`]'s headroom. `None` when unmeasured (no `sequentialPeakGb`, or no entry for this
/// tier) — then the [`FitDecision::Offload`] path keeps today's best-effort behavior: run sequentially
/// and lean on the reactive Metal/CUDA-OOM containment backstop rather than reject.
///
/// An `nvfp4` tier with no measured row falls back to the `q8` row, mirroring [`predicted_peak_gb`]'s
/// NVFP4 note (q8's sequential working set is an over-prediction of NVFP4's, so the second-stage gate
/// degrades conservatively rather than best-effort). sc-11043 must add the real `nvfp4` rows.
pub(crate) fn predicted_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let sequential = manifest_entry.get("candle")?.get("sequentialPeakGb")?;
    let measured = |key: &str| sequential.get(key).and_then(json_f64);
    measured(tier_key)
        .or_else(|| (tier_key == NVFP4_TIER).then(|| measured("q8")).flatten())
        .map(|gb| gb + HEADROOM_GB)
}

/// Sequential prediction with the same adapter residency charged in every lifecycle policy.
pub(crate) fn predicted_sequential_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_sequential_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Decide whether the predicted peak fits the (possibly capped) live budget. Missing either input ⇒
/// `Unknown` (never block). Compares against `free_gb` — what is actually allocatable now — mirroring
/// the flux2 edit guard's use of `available_gb`.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<VramBudget>) -> FitDecision {
    let (Some(needed_gb), Some(budget)) = (needed_gb, budget) else {
        return FitDecision::Unknown;
    };
    if budget.free_gb + f64::EPSILON < needed_gb {
        FitDecision::TooBig {
            needed_gb,
            available_gb: budget.free_gb,
        }
    } else {
        FitDecision::Fits
    }
}

/// Second-stage gate for an [`FitDecision::Offload`] (epic 10765, sc-10856): sequential residency was
/// selected because the RESIDENT peak won't fit, on the promise that the sequential working set will.
/// If the tier's MEASURED sequential peak is known (`sequential_needed_gb` = [`predicted_sequential_peak_gb`])
/// and STILL exceeds the budget, this returns `Some(needed_gb)` so the caller can reject before load with
/// an actionable message instead of running into a reactive Metal/CUDA OOM. `None` — the sequential peak
/// fits, is unmeasured for this tier, or there is no live budget — keeps today's best-effort run.
pub(crate) fn sequential_overflow_gb(
    sequential_needed_gb: Option<f64>,
    budget: Option<VramBudget>,
) -> Option<f64> {
    let (needed_gb, budget) = (sequential_needed_gb?, budget?);
    (budget.free_gb + f64::EPSILON < needed_gb).then_some(needed_gb)
}

/// The resolved residency PLAN for a sequential-capable candle image lane: the pure composition of
/// [`fit_decision`] → [`resolve_offload`] → [`sequential_overflow_gb`] that base.rs's txt2img gate and
/// the bespoke Qwen-Edit gate share. Carries only the ACTION — never a budget-derived number — so two
/// plans computed against different budgets compare equal **iff they take the same action**. That is
/// the invariant the sc-13960 evict-then-reclaim two-pass leans on: gate once against raw free and once
/// against `free + reclaimable_pool`, and evict only when the plan actually improves (never for two
/// rejects that differ only in their reported free-VRAM number).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadPlan {
    /// Load at full residency — the whole-model resident peak fits (or the tier is unmeasured, so the
    /// gate never blocks).
    Resident,
    /// Load with sequential component residency — the resident peak overflows but the staged working
    /// set fits (or its peak is unmeasured, so best-effort staging).
    Sequential,
    /// Reject before OOM — the resident peak overflows AND either the measured sequential peak also
    /// overflows or the engine cannot stage.
    Reject,
}

impl LoadPlan {
    /// The stable wire label used by the sc-19049 decision baseline. Named here rather than in the
    /// baseline emitter so a renamed variant cannot silently keep its serialized name.
    ///
    /// Test-only: production branches on the variant and never serializes it. Without the `cfg` the
    /// candle lane's `-D dead-code` rejects it, which is the exact class `rust:check:candle` exists
    /// to surface — the module-level `allow(dead_code)` above is inert on precisely that build.
    #[cfg(test)]
    pub(crate) fn label(self) -> &'static str {
        match self {
            LoadPlan::Resident => "resident",
            LoadPlan::Sequential => "sequential",
            LoadPlan::Reject => "reject",
        }
    }
}

/// Resolve the [`LoadPlan`] for `needed` (resident peak) and `sequential_needed` (measured staged peak,
/// [`predicted_sequential_peak_gb`]) against `budget`. `None` peaks are unmeasured ⇒ never block (admit
/// resident, or stage best-effort). `sequential_capable` is the provider's staging capability: a lane
/// that cannot stage turns a resident overflow straight into [`LoadPlan::Reject`].
///
/// Monotonic in the budget: a larger `free_gb` can only move the plan `Reject → Sequential → Resident`,
/// never the other way — which is what makes "plan changed after reclaim ⇒ plan improved" sound.
pub(crate) fn load_plan(
    needed: Option<f64>,
    sequential_needed: Option<f64>,
    budget: Option<VramBudget>,
    sequential_capable: bool,
) -> LoadPlan {
    match resolve_offload(fit_decision(needed, budget), sequential_capable) {
        FitDecision::Offload { .. } => {
            if sequential_overflow_gb(sequential_needed, budget).is_some() {
                LoadPlan::Reject
            } else {
                LoadPlan::Sequential
            }
        }
        // Reached only when the engine cannot stage (resolve_offload leaves TooBig intact); a
        // sequential-capable lane rewrites TooBig → Offload above.
        FitDecision::TooBig { .. } => LoadPlan::Reject,
        // Fits (resident) or Unknown (unmeasured / no budget) — admit resident, never block.
        _ => LoadPlan::Resident,
    }
}

/// The stable wire label for a [`FitDecision`], used by the sc-19049 decision baseline. Defined
/// beside the gate for the same reason `LoadPlan::label` is: a renamed or re-shaped variant must
/// break the baseline rather than serialize under its old name. Test-only, same as that one.
#[cfg(test)]
pub(crate) fn fit_decision_label(decision: &FitDecision) -> &'static str {
    match decision {
        FitDecision::Unknown => "unknown",
        FitDecision::Fits => "fits",
        FitDecision::Offload { .. } => "offload",
        FitDecision::TooBig { .. } => "too_big",
    }
}
