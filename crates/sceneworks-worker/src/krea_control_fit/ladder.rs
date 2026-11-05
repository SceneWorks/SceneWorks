//! Krea control ladder coordination.
use super::*;

#[path = "ladder/candidates.rs"]
mod candidates;
#[path = "ladder/evidence.rs"]
mod evidence;
#[path = "ladder/verdict.rs"]
mod verdict;

#[derive(Clone, Copy)]
struct ControlLadderContext<'a> {
    contract: &'a MemoryProviderContract,
    numeric_tier: MemoryNumericTier,
    request_geometry: MemoryGeometry,
    measured_geometry: MemoryGeometry,
    overlay: &'a String,
    evidence_is_current: bool,
    runtime_verified: bool,
    adapter_gb: f64,
    measured_closure_digest: &'a str,
}

struct ControlCandidateRows {
    measured: Vec<(MemoryStrategy, f64)>,
    selections: Vec<MemorySelection>,
    evidence: Vec<MemoryEvidence>,
    estimates: Vec<(MemorySelection, MemoryEvidence)>,
}

fn bytes(gb: f64) -> u64 {
    (gb.max(0.0) * crate::fit_gate::BYTES_PER_GIB)
        .round()
        .clamp(0.0, u64::MAX as f64) as u64
}

/// Submit the Krea control lane's directly measured resident, staged, tiled-decode, and
/// chunked-attention candidates to the one shared selector. This function reads evidence and maps the
/// selected shared strategy to the provider's legacy booleans; it does not own strategy order.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_ladder_for_tier(
    contract: Option<&MemoryProviderContract>,
    tier_key: &str,
    request_geometry: MemoryGeometry,
    peak_gb: Option<f64>,
    sequential_peak_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    chunk_attn_save_gb: Option<f64>,
    evidence_is_current: bool,
    runtime_verified: bool,
    adapter_gb: f64,
    // The `candle.control.inferenceClosureDigest` these manifest rows were measured under
    // (sc-17774). Read from the manifest, never hardcoded — and since sc-22738 carried into the
    // evidence receipt as provenance only, never compared.
    measured_closure_digest: &str,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let Some(contract) = contract else {
        return KreaControlFit::Unknown;
    };
    let numeric_tier = control_numeric_tier(tier_key);
    let measured_geometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
    };
    let overlay = if adapter_gb > 0.0 {
        format!(
            "control_branch+adapters_bytes={}",
            (adapter_gb * crate::fit_gate::BYTES_PER_GIB).round() as u64
        )
    } else {
        "control_branch".to_owned()
    };
    let request = RequestScope {
        resolved_route: KREA_CONTROL_ROUTE,
        backend: "candle",
        tier: numeric_tier,
        mode: KREA_CONTROL_MODE,
        overlay: Some(&overlay),
        geometry: request_geometry,
    };
    let context = ControlLadderContext {
        contract,
        numeric_tier,
        request_geometry,
        measured_geometry,
        overlay: &overlay,
        evidence_is_current,
        runtime_verified,
        adapter_gb,
        measured_closure_digest,
    };
    let rows = context.candidate_rows(
        peak,
        sequential_peak_gb,
        decode_tile_save_gb,
        chunk_attn_save_gb,
    );
    let candidates = rows.candidates();
    let selected = crate::memory_strategy::select_strategy(
        request,
        contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            // A zero-free synthetic test budget still represents a real device whose total is not
            // zero. Preserve the old starved-card behavior while satisfying the shared budget type's
            // physical-total invariant.
            total_gb: budget.total_gb.max(f64::EPSILON),
            reserved_headroom_gb: HEADROOM_GB,
        }),
        &candidates,
    );
    context.verdict(selected, budget, &rows.measured)
}
