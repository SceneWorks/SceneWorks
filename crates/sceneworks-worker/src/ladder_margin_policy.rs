//! Ladder margin policy derived from repeat-capture variance (sc-18094, epic 18093).
//!
//! Epic 18093 retires "measurement currency as a gate". Instead of invalidating evidence whose
//! closure digest went stale, the selector will keep it eligible behind a widened margin
//! (sc-18095), and will admit estimate-backed rungs nobody has measured behind a wider one still
//! (sc-18096/18097). This module owns ONLY the margin constants those follow-ups consume; it
//! changes no selector behavior by itself.
//!
//! Every value here is pinned to a committed derivation —
//! `scripts/derive-ladder-margins.mjs`, run against
//! `docs/generated/memory-calibration-evidence.json` (65 records as derived) — and
//! `scripts/derive-ladder-margins.test.mjs` fails if a constant and the derivation output ever
//! disagree, so evidence growth that pushes observed variance past a floor reds CI instead of
//! silently under-margining. Derivation summary as of the 65-record corpus:
//!
//! * mlx: 17 repeat groups (39 pairs) sharing the full evidence key
//!   (route/backend/tier/rung/parameters/geometry). Max binding-relevant capture-to-capture
//!   spread on a phase peak: 1.2384% (qwen_image q4 bounded_attention denoise). Doubled as a
//!   safety term: 2.4767% — under the floor, so the floor binds.
//! * candle: 15 records, all unique keys, ZERO repeat pairs. Per the sc-18094 rule the hard
//!   floor is the whole margin; no variance number is invented.
//!
//! A margin is a fraction: a candidate fits a budget when
//! `peak_bytes * (1.0 + margin) <= budget_bytes`.

/// Hard floor for MLX margins, applied when repeat-capture variance is thin — and, as of the
/// 65-record corpus, the binding value (observed variance term 2.4767% < 5%). Why 5%:
///
/// 1. An MLX allocator overshoot aborts the whole worker process via the default MLX error
///    handler (no recoverable `Err` exists on that lane), so the margin must cover variance not
///    yet sampled, not just the 1.24% max observed across 39 pairs on one machine and two model
///    families.
/// 2. The evidence corpus itself demonstrates ~5% envelope headroom: across all 50 MLX records
///    the shipped predictor's envelope gap `(predicted - observed) / predicted` spans
///    4.76%..5.21%. A 5% floor keeps stale/estimate admission no more aggressive than the gap
///    the calibrated pipeline already carries on every measured cell. (The epic 15448-era
///    ~5-6% cold-allocator settle observations are not restated anywhere in `docs/`, so the
///    floor is anchored to the evidence file rather than to unverifiable history.)
pub const LADDER_MARGIN_HARD_FLOOR_MLX: f64 = 0.05;

/// Hard floor for candle margins — and, with zero candle repeat pairs in the corpus, the whole
/// candle margin. Looser than MLX because the failure mode is cheaper and the accounting is
/// tighter: a CUDA/candle allocation failure is a recoverable `Err` that fails the job while the
/// worker survives, and candle evidence is deterministic live-allocation counting (10 of 15
/// records report observed == predicted to the byte, none show reclaimable slack).
pub const LADDER_MARGIN_HARD_FLOOR_CANDLE: f64 = 0.02;

/// Margin applied to MEASURED evidence whose closure digest is stale (sc-18095), MLX lane.
/// Derived: `max(LADDER_MARGIN_HARD_FLOOR_MLX, 2 x 1.2384% observed max binding spread)` — the
/// floor binds.
pub const MLX_STALE_MEASURED_MARGIN: f64 = 0.05;

/// Margin applied to ESTIMATE-BACKED unmeasured candidates (sc-18096/18097), MLX lane. Double
/// the stale-measured margin: an estimate carries model extrapolation error on top of the
/// capture variance a re-measured cell would show.
pub const MLX_ESTIMATE_MARGIN: f64 = 0.10;

/// Margin applied to MEASURED evidence whose closure digest is stale (sc-18095), candle lane.
/// Equal to `LADDER_MARGIN_HARD_FLOOR_CANDLE`: the corpus has no candle repeat pairs, so the
/// documented hard floor is the whole margin.
pub const CANDLE_STALE_MEASURED_MARGIN: f64 = 0.02;

/// Margin applied to ESTIMATE-BACKED unmeasured candidates (sc-18096/18097), candle lane.
/// Double the candle stale-measured margin, same widening rationale as
/// [`MLX_ESTIMATE_MARGIN`].
pub const CANDLE_ESTIMATE_MARGIN: f64 = 0.04;

/// Structural invariants of the policy, enforced at COMPILE TIME (a violating edit fails
/// `cargo build`, not just a test lane), independent of the current corpus: margins never dip
/// under their backend's floor, estimates are strictly wider than stale-measured, and the
/// fatal-OOM backend (MLX) is never less conservative than the recoverable one (candle).
const _: () = {
    assert!(MLX_STALE_MEASURED_MARGIN >= LADDER_MARGIN_HARD_FLOOR_MLX);
    assert!(CANDLE_STALE_MEASURED_MARGIN >= LADDER_MARGIN_HARD_FLOOR_CANDLE);
    assert!(MLX_ESTIMATE_MARGIN > MLX_STALE_MEASURED_MARGIN);
    assert!(CANDLE_ESTIMATE_MARGIN > CANDLE_STALE_MEASURED_MARGIN);
    assert!(MLX_STALE_MEASURED_MARGIN >= CANDLE_STALE_MEASURED_MARGIN);
    assert!(MLX_ESTIMATE_MARGIN >= CANDLE_ESTIMATE_MARGIN);
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
        assert_eq!(MLX_STALE_MEASURED_MARGIN, 0.05);
        assert_eq!(MLX_ESTIMATE_MARGIN, 0.10);
        assert_eq!(CANDLE_STALE_MEASURED_MARGIN, 0.02);
        assert_eq!(CANDLE_ESTIMATE_MARGIN, 0.04);
    }
}
