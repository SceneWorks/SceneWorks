//! Ladder margin policy derived from repeat-capture variance (sc-18094, epic 18093).
//!
//! Epic 18093 retires "measurement currency as a gate". Instead of invalidating evidence whose
//! closure digest went stale, the selector keeps it eligible behind a widened margin (sc-18095,
//! applied in `crate::memory_strategy::select_strategy`), and will admit estimate-backed rungs
//! nobody has measured behind a wider one still (sc-18096/18097). This module owns ONLY the margin
//! constants those consumers read; it changes no selector behavior by itself.
//!
//! Every value here is pinned to a committed derivation —
//! `scripts/derive-ladder-margins.mjs`, run against
//! `docs/generated/memory-calibration-evidence.json` (89 records as derived) — and
//! `scripts/derive-ladder-margins.test.mjs` fails if a constant and the derivation output ever
//! disagree, so evidence growth that pushes observed variance past a floor reds CI instead of
//! silently under-margining. Derivation summary as of the 89-record corpus:
//!
//! * mlx: 22 repeat groups (85 pairs) sharing the full evidence key
//!   (route/backend/tier/rung/parameters/geometry). Max binding-relevant capture-to-capture
//!   spread on a phase peak: 12.6018% (the historical versus SC-19753 Z-Image q4 bounded-
//!   transformer-residency conditioning peak). Doubled as a safety term: 25.2037%, so variance
//!   now overrides the 5% floor. The binding/non-binding split is a SAME-CELL argument only;
//!   the max CAN-BIND per-phase spread (any phase, accounting flips excluded) is 17.1369%, whose
//!   fully widened 68.55% estimate term still exceeds the shipped 50.4073% estimate margin — see
//!   [`ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`].
//! * candle: 15 records, all unique keys, ZERO repeat pairs. Per the sc-18094 rule the hard
//!   floor is the whole margin; no variance number is invented.
//!
//! A margin is a fraction: a candidate fits a budget when
//! `peak_bytes * (1.0 + margin) <= budget_bytes`.

/// Hard floor for MLX margins, applied when repeat-capture variance is thin. The 89-record corpus's
/// 25.2037% variance term currently exceeds it; it remains the minimum for a future thinner corpus.
/// Why 5%:
///
/// 1. An MLX allocator overshoot aborts the whole worker process via the default MLX error
///    handler (no recoverable `Err` exists on that lane), so the margin must cover variance not
///    yet sampled, not just the 12.60% max observed across 85 pairs in the current corpus.
/// 2. The evidence corpus itself demonstrates ~5% envelope headroom: across all 74 MLX records
///    the shipped predictor's envelope gap `(predicted - observed) / predicted` spans
///    4.76%..5.58% — computed and printed by `scripts/derive-ladder-margins.mjs`
///    (`predictorEnvelopeGapRange`) and pinned in `scripts/derive-ladder-margins.test.mjs`, not
///    asserted from prose. A 5% floor keeps stale/estimate admission no more aggressive than
///    the gap the calibrated pipeline already carries on every measured cell. (The epic
///    15448-era ~5-6% cold-allocator settle observations are not restated anywhere in `docs/`,
///    so the floor is anchored to the evidence file rather than to unverifiable history.)
pub const LADDER_MARGIN_HARD_FLOOR_MLX: f64 = 0.05;

/// Hard floor for candle margins — and, with zero candle repeat pairs in the corpus, the whole
/// candle margin. Looser than MLX because the failure mode is cheaper and the accounting is
/// tighter: a CUDA/candle allocation failure is a recoverable `Err` that fails the job while the
/// worker survives, and candle evidence is deterministic live-allocation counting (10 of 15
/// records report observed == predicted to the byte, none show reclaimable slack).
pub const LADDER_MARGIN_HARD_FLOOR_CANDLE: f64 = 0.02;

/// Margin applied to MEASURED evidence whose closure digest is stale (sc-18095), MLX lane.
/// Derived: `max(LADDER_MARGIN_HARD_FLOOR_MLX, 2 x 12.6018% observed max binding spread)`;
/// variance binds at 25.2037%. Stale admission is SAME-CELL admission (the cell being admitted is
/// the cell that was measured), which is exactly the scope where the derivation's non-binding
/// exclusion is sound: a phase far below the measured envelope cannot become the envelope within
/// the admitted spread bounds.
pub const MLX_STALE_MEASURED_MARGIN: f64 = 0.2520367016951188;

/// Margin applied to ESTIMATE-BACKED unmeasured candidates (sc-18096/18097), MLX lane. Double
/// the stale-measured margin: an estimate carries model extrapolation error on top of the
/// capture variance a re-measured cell would show.
///
/// SCOPE: this margin absorbs re-capture variance only for candidates whose predicted BINDING
/// PHASE matches the measured cell's. The corpus demonstrates a 17.1369% per-phase re-capture
/// spread (denoise) whose fully widened 68.55% derivation term exceeds this margin; admission that
/// extrapolates the binding phase is forbidden by
/// [`ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`] instead.
pub const MLX_ESTIMATE_MARGIN: f64 = 0.5040734033902377;

/// Margin applied to MEASURED evidence whose closure digest is stale (sc-18095), candle lane.
/// Equal to `LADDER_MARGIN_HARD_FLOOR_CANDLE`: the corpus has no candle repeat pairs, so the
/// documented hard floor is the whole margin.
pub const CANDLE_STALE_MEASURED_MARGIN: f64 = 0.02;

/// Margin applied to ESTIMATE-BACKED unmeasured candidates (sc-18096/18097), candle lane.
/// Double the candle stale-measured margin, same widening rationale as
/// [`MLX_ESTIMATE_MARGIN`].
pub const CANDLE_ESTIMATE_MARGIN: f64 = 0.04;

/// Constraint inherited by the estimate-admission follow-ups (sc-18096/18097), pinned here
/// because the margins above cannot carry it: estimate-backed admission MUST NOT admit a
/// candidate whose predicted binding phase differs from the measured cell's binding phase
/// without per-phase variance re-derivation for that phase.
///
/// Why a constraint and not a wider margin: the corpus demonstrates a 17.1369%
/// cross-fingerprint same-key re-capture spread on a phase peak (denoise/activeBytes,
/// imc-5ea462dfe3101260a9b1 vs imc-da3533c476605929f10d). That phase was non-binding in its
/// measured cell (a 16 GB text-encoder conditioning peak dominated at 1024 squared), so it cannot
/// flip a same-cell admission; but an estimate
/// extrapolating to a different rung (bounded conditioning) or larger geometry (MLX activation
/// transients scale linearly in area) can make denoise carry the request peak — the fatal-OOM
/// direction on MLX. Folding that spread into the estimate margin per the derivation rule
/// (x2 safety, floor, x2 widening) yields 68.55% — unusable — so the risk is carried by this
/// rule instead. `scripts/derive-ladder-margins.test.mjs` pins this constant against the
/// script's mirror export and asserts its fully widened 68.55% term still exceeds
/// [`MLX_ESTIMATE_MARGIN`], i.e. the constraint stays load-bearing.
///
/// SCOPE: this constraint governs estimate candidates extrapolated from a measured cell
/// (fitted per-phase curves). Candidates with no measured cell in their extrapolation basis —
/// the weights + headroom floor path of epic 18093 R1 — have no measured binding phase to
/// match and are NOT gated by this constraint; their risk is carried by the headroom floor and
/// the estimate margin, not this rule.
pub const ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE: bool = true;

/// Structural invariants of the policy, enforced at COMPILE TIME (a violating edit fails
/// `cargo build`, not just a test lane), independent of the current corpus: margins never dip
/// under their backend's floor, estimates are strictly wider than stale-measured, the
/// fatal-OOM backend (MLX) is never less conservative than the recoverable one (candle), and
/// the estimate-admission binding-phase constraint stays declared (it may only be retired
/// together with a per-phase variance re-derivation — see its doc).
const _: () = {
    assert!(MLX_STALE_MEASURED_MARGIN >= LADDER_MARGIN_HARD_FLOOR_MLX);
    assert!(CANDLE_STALE_MEASURED_MARGIN >= LADDER_MARGIN_HARD_FLOOR_CANDLE);
    assert!(MLX_ESTIMATE_MARGIN > MLX_STALE_MEASURED_MARGIN);
    assert!(CANDLE_ESTIMATE_MARGIN > CANDLE_STALE_MEASURED_MARGIN);
    assert!(MLX_STALE_MEASURED_MARGIN >= CANDLE_STALE_MEASURED_MARGIN);
    assert!(MLX_ESTIMATE_MARGIN >= CANDLE_ESTIMATE_MARGIN);
    assert!(ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact pins mirroring `scripts/derive-ladder-margins.mjs` output. The node-side test
    /// (`scripts/derive-ladder-margins.test.mjs`, wired into `npm run check`) is the live
    /// coupling to the derivation; this pin makes a drive-by constant edit red in `rust:check`
    /// too, without waiting for the node lane.
    #[test]
    fn constants_match_the_sc_18094_derivation() {
        assert_eq!(LADDER_MARGIN_HARD_FLOOR_MLX, 0.05);
        assert_eq!(LADDER_MARGIN_HARD_FLOOR_CANDLE, 0.02);
        assert_eq!(MLX_STALE_MEASURED_MARGIN, 0.2520367016951188);
        assert_eq!(MLX_ESTIMATE_MARGIN, 0.5040734033902377);
        assert_eq!(CANDLE_STALE_MEASURED_MARGIN, 0.02);
        assert_eq!(CANDLE_ESTIMATE_MARGIN, 0.04);
        // ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE is pinned in the compile-time
        // invariants block above (clippy forbids constant assertions in a runtime test) and
        // against the script's mirror export by scripts/derive-ladder-margins.test.mjs.
    }
}
