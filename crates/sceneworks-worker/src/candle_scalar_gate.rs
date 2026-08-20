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

/// What the manifest `candle` block's scalar may claim for THIS request geometry (sc-19054, epic
/// 19048 R3). The classification law lives in the generic mechanism
/// ([`crate::estimate_synthesis::declared_scalar_class`]); this reads the lane's own keys —
/// epic 18472's bare-boolean `measured` flag and its `vramMeasuredPixels` capture geometry — and
/// additionally demotes any row the tier does not OWN:
///
/// * an `nvfp4` request served by the degraded `q8` row is sized by a measurement of a DIFFERENT
///   tier (a deliberate over-prediction, see [`predicted_peak_gb`]) — evidence about q8, a floor
///   for nvfp4;
/// * the `minMemoryGb` fallback is by definition the default tier's padded floor.
///
/// `rows_key` selects which scalar table is being classified (`"vramGbByTier"` for the resident
/// peak, `"sequentialPeakGb"` for the staged working set); both share the block-level `measured`
/// flag and capture geometry, which is what "a bare boolean sibling of vramGbByTier" means.
pub(crate) fn scalar_peak_class(
    manifest_entry: &JsonObject,
    rows_key: &str,
    tier_key: &str,
    geometry: gen_core::MemoryGeometry,
) -> crate::estimate_synthesis::DeclaredScalarClass {
    let candle = manifest_entry.get("candle");
    let own_row = candle
        .and_then(|candle| candle.get(rows_key))
        .and_then(|rows| rows.get(tier_key))
        .and_then(json_f64)
        .is_some();
    if !own_row {
        return crate::estimate_synthesis::DeclaredScalarClass::DeclaredFloor;
    }
    let measured = candle
        .and_then(|candle| candle.get("measured"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let declared_pixels = candle
        .and_then(|candle| candle.get("vramMeasuredPixels"))
        .and_then(serde_json::Value::as_u64);
    crate::estimate_synthesis::declared_scalar_class(measured, declared_pixels, geometry)
}

/// Grade a scalar-derived prediction for admission (sc-19054): a measured peak that covers the
/// request geometry is compared as-is (byte-for-byte the pre-story behavior), while a
/// [`DeclaredScalarClass::DeclaredFloor`] is widened by the candle ESTIMATE margin — exactly the
/// grade the shared selector gives an `EstimateFloor` candidate, applied here only because the
/// legacy scalar gate has no selector downstream of it. The widening arithmetic is the selector's
/// own (`memory_strategy::widened_peak_bytes` over integer bytes), never a second law.
///
/// [`DeclaredScalarClass::DeclaredFloor`]: crate::estimate_synthesis::DeclaredScalarClass
fn graded_scalar_gb(peak_gb: f64, class: crate::estimate_synthesis::DeclaredScalarClass) -> f64 {
    match class {
        crate::estimate_synthesis::DeclaredScalarClass::MeasuredPeak => peak_gb,
        crate::estimate_synthesis::DeclaredScalarClass::DeclaredFloor => {
            let bytes = (peak_gb * BYTES_PER_GIB).ceil().clamp(0.0, u64::MAX as f64) as u64;
            crate::memory_strategy::peak_bytes_to_gb(crate::memory_strategy::widened_peak_bytes(
                bytes,
                crate::memory_strategy::estimate_margin(gen_core::MemoryBackend::Candle),
            ))
        }
    }
}

/// **The geometry-aware resident admission prediction** (sc-19054, epic 19048 R3): the
/// [`predicted_peak_gb`] scalar plus the adapter stack, GRADED for this request's geometry —
/// presented as the peak only where its measurement covers the request
/// ([`scalar_peak_class`]), widened by the candle estimate margin where it is only the declared
/// floor. This is what makes a 1024² and a 2048² request predict DIFFERENT admission ceilings on
/// a route whose scalar was measured at 1024².
///
/// Image-lane pre-gates (`fit_decision` / `load_plan` comparisons) call this; the shared-selector
/// entry points keep passing the RAW [`predicted_peak_gb_with_adapter_bytes`] number, because the
/// selector applies the identical grade itself through the resident candidate's evidence class
/// (`candle_memory_strategy`) and a pre-widened input would double-charge the margin. The flat
/// video gates stay on the raw floor by design until sc-19055 migrates them.
pub(crate) fn predicted_peak_gb_for_request(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    geometry: gen_core::MemoryGeometry,
) -> Option<f64> {
    predicted_peak_gb_with_adapter_bytes(manifest_entry, tier_key, adapter_bytes).map(|peak| {
        graded_scalar_gb(
            peak,
            scalar_peak_class(manifest_entry, "vramGbByTier", tier_key, geometry),
        )
    })
}

/// [`predicted_peak_gb_for_request`]'s sequential sibling: the measured staged working set
/// (`sequentialPeakGb`) under the same scalar grade. The second-stage offload gate compares this,
/// so a staged load at a geometry the capture never covered is graded as the floor it is.
pub(crate) fn predicted_sequential_peak_gb_for_request(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    geometry: gen_core::MemoryGeometry,
) -> Option<f64> {
    predicted_sequential_peak_gb_with_adapter_bytes(manifest_entry, tier_key, adapter_bytes).map(
        |peak| {
            graded_scalar_gb(
                peak,
                scalar_peak_class(manifest_entry, "sequentialPeakGb", tier_key, geometry),
            )
        },
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate_synthesis::DeclaredScalarClass;
    use serde_json::json;

    fn entry(value: serde_json::Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    fn image(width: u32, height: u32) -> gen_core::MemoryGeometry {
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        }
    }

    fn measured_entry(measured: bool) -> JsonObject {
        entry(json!({
            "candle": {
                "measured": measured,
                "vramMeasuredPixels": 1_048_576,
                "minMemoryGb": 40,
                "vramGbByTier": { "q4": 30.0 },
                "sequentialPeakGb": { "q4": 10.0 }
            }
        }))
    }

    /// **sc-19054 headline: the scalar gate is geometry-aware.** On a `measured: true` route
    /// captured at 1024², a covered request compares the raw measured peak (byte-for-byte the
    /// pre-story number) while a request BEYOND the capture geometry is graded as the declared
    /// floor it is — widened by exactly the candle estimate margin, the same grade the shared
    /// selector gives an `EstimateFloor` candidate.
    ///
    /// Mutation coverage: dropping the geometry conjunct (grading every cell `MeasuredPeak`)
    /// flips the 2048² inequality red; grading every cell `DeclaredFloor` flips the 1024²
    /// equality red; zeroing the margin flips the exact-ratio assertion red; widening with the
    /// STALE margin instead of the estimate margin also flips it (0.02 != 0.04).
    #[test]
    fn a_covered_request_takes_the_measured_peak_and_a_larger_one_takes_the_widened_floor() {
        let manifest = measured_entry(true);
        let raw = 30.0 + HEADROOM_GB;
        let at = |width: u32, height: u32| {
            predicted_peak_gb_for_request(&manifest, "q4", 0, image(width, height))
                .expect("q4 row predicts")
        };
        assert_eq!(at(1024, 1024), raw, "covered geometry: the measured peak, ungraded");
        assert_eq!(
            at(512, 512),
            raw,
            "the measured peak bounds every smaller request from above"
        );
        let beyond = at(2048, 2048);
        assert!(
            beyond > raw,
            "a floor captured at 1024² must not be presented as the peak for 4 MP"
        );
        let expected =
            (raw * (1.0 + crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN) * BYTES_PER_GIB)
                .ceil()
                / BYTES_PER_GIB;
        assert!(
            (beyond - expected).abs() < 1e-9,
            "the floor grade is the candle ESTIMATE margin over integer bytes (got {beyond}, \
             expected {expected})"
        );

        // The sequential sibling takes the identical grade.
        let sequential_raw = 10.0 + HEADROOM_GB;
        assert_eq!(
            predicted_sequential_peak_gb_for_request(&manifest, "q4", 0, image(1024, 1024)),
            Some(sequential_raw)
        );
        let sequential_beyond =
            predicted_sequential_peak_gb_for_request(&manifest, "q4", 0, image(2048, 2048))
                .expect("sequential row predicts");
        assert!(sequential_beyond > sequential_raw);
    }

    /// `measured: false` (epic 18472) demotes the scalar at EVERY geometry — the numbers were
    /// never captured, so even the declared 1024² cell is only a floor.
    #[test]
    fn an_unmeasured_scalar_is_a_floor_at_every_geometry() {
        let manifest = measured_entry(false);
        let raw = 30.0 + HEADROOM_GB;
        let graded = predicted_peak_gb_for_request(&manifest, "q4", 0, image(1024, 1024))
            .expect("q4 row predicts");
        assert!(
            graded > raw,
            "an unmeasured scalar must be graded as a floor even inside its declared geometry"
        );
        assert_eq!(
            scalar_peak_class(&manifest, "vramGbByTier", "q4", image(512, 512)),
            DeclaredScalarClass::DeclaredFloor
        );
    }

    /// Rows the tier does not OWN are floors regardless of the flag: the nvfp4→q8 degrade is a
    /// measurement of a DIFFERENT tier, and the `minMemoryGb` fallback is the default tier's
    /// padded floor. Adapter bytes are charged before the grade, mirroring the selector's
    /// overlay accounting order.
    #[test]
    fn borrowed_rows_and_min_memory_fallbacks_are_floors_and_adapters_precede_the_grade() {
        let manifest = measured_entry(true);
        assert_eq!(
            scalar_peak_class(&manifest, "vramGbByTier", crate::image_jobs::NVFP4_TIER, image(1024, 1024)),
            DeclaredScalarClass::DeclaredFloor,
            "an nvfp4 request served by the q8 row is sized by another tier's measurement"
        );
        assert_eq!(
            scalar_peak_class(&manifest, "vramGbByTier", "bf16", image(1024, 1024)),
            DeclaredScalarClass::DeclaredFloor,
            "a tier with no row of its own falls to minMemoryGb, the padded default-tier floor"
        );
        let min_memory = predicted_peak_gb_for_request(&manifest, "bf16", 0, image(1024, 1024))
            .expect("minMemoryGb fallback predicts");
        assert!(
            min_memory > 40.0,
            "the minMemoryGb fallback is floor-graded (got {min_memory})"
        );

        // Adapter bytes join the peak BEFORE the grade: widen(peak + adapter), never
        // widen(peak) + adapter.
        let adapter = 1u64 << 30;
        let graded = predicted_peak_gb_for_request(&manifest, "q4", adapter, image(2048, 2048))
            .expect("q4 row predicts");
        let expected = ((30.0 + HEADROOM_GB + 1.0)
            * (1.0 + crate::ladder_margin_policy::CANDLE_ESTIMATE_MARGIN)
            * BYTES_PER_GIB)
            .ceil()
            / BYTES_PER_GIB;
        assert!(
            (graded - expected).abs() < 1e-9,
            "adapter bytes precede the grade (got {graded}, expected {expected})"
        );
    }
}
