use super::{gate_with_evict_reclaim, Path, Settings, WorkerResult};

use crate::conditioning_fit::{ConditioningFit, ConditioningFootprint};

// The ONE admission seam every candle conditioning route (ControlNet / IP-Adapter / identity encoder)
// calls before it allocates (sc-16069, epic 15448). The pure decision lives in
// `crate::conditioning_fit`; this is its live wiring — budget resolution + the sc-13960 two-pass
// evict-then-reclaim — kept in ONE place so eleven lanes cannot drift into eleven slightly different
// gates.
//
// **Why the lanes needed a shared seam rather than a copy each.** `resolve_candle_image_route` claims
// every conditioning route BEFORE the generic `CandleTxt2Img` arm (deliberately — they share candle
// txt2img model ids, so without the divert a pose job would render as plain txt2img with the poses
// dropped), which routes them around both the `generate_candle_stream` `vram_gate` and the
// `generator_cache` `apply_residency_policy`. The gate keys on the ROUTE, but the property that makes
// one necessary — a second network held resident — is a property of the PROVIDER, and nothing connected
// the two. So adding a route silently opted out of admission. The strict-control lanes now gate through
// the REQUIRED `CandleStrictControl::conditioning_admission` (one call site in
// `run_candle_strict_control`, so a new strict-control lane cannot compile without declaring how it is
// admitted); the ip-adapter / identity lanes call `admit_conditioning_paths` directly; and the Krea
// pose-control lane calls it in its own preamble, because its check must precede its own
// `note_loaded_peak` (`ConditioningAdmission::GatedInPreamble`).
// `image_jobs/tests.rs::every_candle_conditioning_route_is_admitted_through_a_gate` is the single-source
// guard that keeps the ladder and the gated set in step.

/// Admit (or refuse) a candle conditioning render against the live VRAM budget.
///
/// Resolves the same budget the other bespoke candle lanes use — the live `nvidia-smi` free/total for
/// `settings.gpu_id`, capped by `SCENEWORKS_CUDA_VRAM_CAP_GB` so a small card can be emulated on the
/// dev box — then walks [`crate::conditioning_fit::decide`] against it and logs the outcome on EVERY
/// path (including both admit-without-evidence paths, which is how "no evidence, no rejection" stays
/// visible instead of silent).
///
/// ## The two-pass budget (sc-13960)
///
/// Every conditioning lane loads through the UNcached `start_gen_stream`, so — unlike the txt2img lane,
/// whose single-slot generator cache evicts its occupant before the incoming load — a resident txt2img
/// generator stays live and CO-RESIDENT with this load. Its cudarc pool pages are therefore NOT free,
/// and crediting them against a raw gate would over-admit an OOM; so the first pass budgets RAW live
/// free. But refusing a render that a warm-cache evict would have admitted is exactly the false reject
/// this gate is least allowed to have, so [`gate_with_evict_reclaim`] takes a second pass against
/// `free + reclaimable_pool` and, only when that turns the refusal into an admit, EVICTS the resident
/// generator so the credit is honest. Same treatment as `qwen_edit_candle` and `krea_control_candle`,
/// the other two `start_gen_stream` lanes.
///
/// ## Why no `note_loaded_peak`
///
/// The sibling lanes record their admitted peak as the reclaimable high-water. This one deliberately
/// does not: [`crate::vram_gate::note_loaded_peak`] must never OVER-count the pool a load leaves behind
/// (a later gate credits it, so an over-count over-admits an OOM), and this gate's number is a weights
/// FLOOR — it under-states the real peak by every unpriced transient, which is the safe direction for
/// admission and would be the unsafe direction here. The same reasoning
/// `krea_control_fit::incurred_peak_gb` applies when control evidence is not current. The cost is bounded and is a
/// *speed* cost only: a following gate under-credits the pool and may stage needlessly. Once an overlay
/// cell is measured there is a peak honest enough to record.
pub(super) async fn admit_conditioning_overlay(
    settings: &Settings,
    footprint: &ConditioningFootprint,
) -> WorkerResult<()> {
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let (fit, _budget) = gate_with_evict_reclaim(
        &settings.gpu_id,
        raw_budget,
        |budget| crate::conditioning_fit::decide(footprint, budget),
        // Evict the warm txt2img cache ONLY when reclaiming its pool turns a refusal into an admit.
        // Two refusals that differ only in their reported free number are the same non-outcome, and an
        // already-admitting gate has nothing to gain — so neither triggers a pointless evict.
        |raw, reclaimed| {
            matches!(raw, ConditioningFit::TooBig { .. })
                && !matches!(reclaimed, ConditioningFit::TooBig { .. })
        },
    )
    .await?;
    crate::conditioning_fit::admit(&fit, footprint, &settings.gpu_id)
}

/// Build a footprint from a lane's already-resolved base dir + overlay path(s) and admit it. The shape
/// every conditioning lane actually needs, so the lanes stay one line each.
pub(super) async fn admit_conditioning_paths(
    settings: &Settings,
    model_label: &str,
    overlay_label: &'static str,
    base_dir: &Path,
    overlays: &[&Path],
) -> WorkerResult<()> {
    let footprint =
        ConditioningFootprint::from_paths(model_label, overlay_label, base_dir, overlays);
    admit_conditioning_overlay(settings, &footprint).await
}
