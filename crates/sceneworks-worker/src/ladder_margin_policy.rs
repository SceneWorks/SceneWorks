//! Per-term admission allowances (sc-22508, epic 22505).
//!
//! Epic 18093 shipped ONE multiplicative margin per backend per currency — `peak * 1.5041` on the
//! MLX estimate path, `peak * 1.2520` on the MLX stale path — applied to the whole predicted peak
//! regardless of which part of that peak was actually uncertain. On a 60 GB derived peak the MLX
//! estimate margin alone added 30 GB of pad, which is what kept a request that truly fits a 128 GB
//! host out of rungs it fits on. Epic 22505 E3 retires that shape: **an allowance is priced against
//! the specific term whose value is uncertain, never against the whole peak just because the whole
//! peak is what the selector happens to hold.**
//!
//! There are exactly three terms, and each names the uncertainty it covers:
//!
//! * [`AdmissionTerm::FullyPriced`] — nothing is left for the selector to add. Two bases reach it.
//!   A MEASURED cell under the live closure is the measurement. An
//!   [`CandidateBasis::EstimateAnchorDerived`] peak already carries both of its uncertainties
//!   inside the derivation (`sceneworks_core::memory_anchor`): coefficient uncertainty is priced
//!   INSIDE each coefficient (every slope sits at or above the highest measured within-cell slope)
//!   and the MLX allocator envelope above a phase's active peak is priced by
//!   `ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`. Re-widening it here would double-charge terms the
//!   derivation already paid for.
//! * [`AdmissionTerm::SameCellRecaptureSpread`] — capture-to-capture spread of the SAME cell's
//!   binding phase. This one IS proportionate to the whole peak, because the quantity that moves
//!   between two captures of one cell is the peak itself. Derived, not invented:
//!   `scripts/derive-ladder-margins.mjs` reports the max binding-phase spread across the corpus's
//!   repeat pairs, and the epic-18093 "x2 safety" and "x2 estimate widening" multipliers that sat
//!   on top of it — neither of which named a term — are gone.
//! * [`AdmissionTerm::AllocatorEnvelopeOverActivation`] — the allocator envelope that sits above a
//!   floor's modelled ACTIVATION bytes. It is proportionate ONLY to that activation (headroom)
//!   term. The weights half of a floor is counted bytes off the manifest that the allocator holds
//!   exactly once, so charging it a percentage was the single largest piece of the retired blanket
//!   margin. The fraction is per-backend ([`FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`] on MLX,
//!   [`CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`] on candle) because the envelope is a property of
//!   the allocator, not of the term's name.
//!
//! The margin constants are pinned against the derivation by
//! `scripts/derive-ladder-margins.test.mjs`, so corpus growth that widens the measured spread reds
//! CI instead of silently under-charging.

use gen_core::MemoryBackend;

use crate::memory_strategy::CandidateBasis;

/// Same-cell capture-to-capture spread on the MLX lane: the max binding-phase spread over the
/// corpus's repeat pairs (`scripts/derive-ladder-margins.mjs`, `recaptureSpread`).
///
/// UNCERTAINTY COVERED: re-running one measured cell lands on a different peak. Nothing else. The
/// epic-18093 constant doubled this number as "protection against the next capture landing outside
/// the sampled range" and then doubled it AGAIN for estimates; neither doubling named a term, and a
/// margin that cannot name its term is what E3 retires. The failure posture for what the sampled
/// range does not cover is runtime catching (E6), not a standing 4x pad on every admission.
pub const MLX_RECAPTURE_SPREAD: f64 = 0.1260183508475594;

/// Same-cell spread on the candle lane. The corpus has ZERO candle repeat pairs, so no spread is
/// measurable and none is invented; this is the documented accounting-residual floor instead.
///
/// UNCERTAINTY COVERED: candle evidence is deterministic live-allocation counting rather than an
/// allocator-pool envelope (10 of 16 records report observed == predicted to the byte, none show
/// reclaimable slack), so the residual is small and bounded by the accounting, not by variance.
pub const CANDLE_RECAPTURE_SPREAD: f64 = 0.02;

/// Allowance on the ACTIVATION (headroom) term of an MLX weights+headroom floor, as a fraction of
/// THAT TERM — not of the floor.
///
/// UNCERTAINTY COVERED: the MLX allocator envelope that sits above the modelled active bytes
/// (cache retention across phase transitions). RE-DERIVED ON THE BASE IT CHARGES (epic 22505
/// feature-end fix round, E3): `scripts/derive-ladder-margins.mjs#deriveFloorEnvelopeAllowance`
/// takes, over every MLX record that reports a steady-state residency, the maximum of
/// `envelope_bytes / activation_bytes` where `envelope_bytes` is the allocator envelope above the
/// measured active peak and `activation_bytes` is the active peak above the post-cleanup resident
/// weight set — i.e. the exact uncertainty over the exact term this fraction multiplies. The
/// binding record is the flux2_dev q4 768x768 eager resident capture: a 26.40 GB retained
/// envelope over an 8.50 GB activation transient, a ratio of ~3.10.
///
/// The previous 0.17 measured the SAME envelope as a fraction of the whole binding active phase
/// (weights included) and then charged it against the activation term alone — under-charging by
/// exactly the weights/activation ratio, which on the retained image renders is most of the
/// envelope. It also happened to equal
/// `sceneworks_core::memory_anchor::ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`; the two are now different
/// numbers BECAUSE their bases differ, and both derivations are stated: the anchor margin
/// multiplies a WHOLE derived phase estimate (weights included), for which the envelope-over-
/// binding-phase measurement (15.84%, bounded at 17%) remains the honest fraction, while this
/// allowance multiplies the activation term alone, for which envelope-over-activation is. One
/// measured phenomenon, two bases, two correctly-based fractions — the retired equality pin was a
/// pin on a coincidence of spelling, not of meaning.
///
/// A fraction ABOVE 1.0 is not a blanket widening here: it says the retained cache the MLX
/// allocator holds above the active peak is ~3x the activation transient on the widest retained
/// render, which is a measurement, not a safety factor. The floor is the LAST-RESORT basis — an
/// anchor-derived candidate outranks it wherever an anchor is current — so the honest charge
/// costs admission nothing on any anchored lane.
///
/// NOT COVERED, deliberately: whether one flat headroom number is the right ACTIVE model for this
/// geometry at all. That residual is unmeasured, and epic 22505 E6 makes runtime catching its
/// failure posture rather than a standing multiple of an already-modelled allowance.
pub const FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE: f64 = 3.104_173_817_050_811;

/// The candle lane's counterpart to [`FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE`], charged against the
/// SAME term (a floor's activation bytes) but named separately because the uncertainty is a
/// different physical thing and must not silently inherit an MLX allocator measurement.
///
/// UNCERTAINTY COVERED: candle has no allocator-pool envelope to measure — its evidence is
/// deterministic live-allocation counting, and `scripts/derive-ladder-margins.mjs` records that 10
/// of the corpus's candle records report observed == predicted to the byte with no record showing
/// reclaimable slack. What remains above a candle floor's modelled activation is that accounting
/// residual, which is exactly the quantity [`CANDLE_RECAPTURE_SPREAD`] documents; this constant is
/// defined as that value rather than restating a second number the corpus does not separately
/// measure.
///
/// It is deliberately NOT 17%: the 17% figure is a measurement of the MLX allocator's retention
/// across phase transitions, a mechanism candle's counted allocations do not have. Borrowing it
/// would be picking a margin by magnitude, which is what E3 retires.
pub const CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE: f64 = CANDLE_RECAPTURE_SPREAD;

/// Constraint inherited by the estimate-admission follow-ups (sc-18096/18097), pinned here
/// because the allowances above cannot carry it: estimate-backed admission MUST NOT admit a
/// candidate whose predicted binding phase differs from the measured cell's binding phase
/// without per-phase variance re-derivation for that phase.
///
/// Why a constraint and not a wider margin: the corpus demonstrates a 17.1369%
/// cross-fingerprint same-key re-capture spread on a phase peak (denoise/activeBytes,
/// imc-5ea462dfe3101260a9b1 vs imc-da3533c476605929f10d). That phase was non-binding in its
/// measured cell (a 16 GB text-encoder conditioning peak dominated at 1024 squared), so it cannot
/// flip a same-cell admission; but an estimate extrapolating to a different rung (bounded
/// conditioning) or larger geometry (MLX activation transients scale linearly in area) can make
/// denoise carry the request peak — the fatal-OOM direction on MLX. That spread belongs to a phase
/// the candidate's own term does not cover, so pricing it as a fraction of the peak would be
/// exactly the untethered widening E3 retires; the risk is carried by this rule instead.
///
/// SCOPE: this constraint governs estimate candidates extrapolated from a measured cell
/// (fitted per-phase curves). Candidates with no measured cell in their extrapolation basis —
/// the weights + headroom floor path of epic 18093 R1 — have no measured binding phase to
/// match and are NOT gated by this constraint; their risk is carried by the headroom floor and
/// its own allowance, not this rule.
pub const ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE: bool = true;

/// Ratified exemption for a prediction that evaluates **every phase independently**, adds that
/// phase's observed maximum fit/held-out absolute residual, and only then takes the maximum over
/// phases at the request geometry. Such an envelope does not extrapolate from one measured
/// binding-phase label: whichever phase binds is already represented by its own conservative law.
///
/// This exemption is deliberately structural, not provider-wide. It applies only to the fitted
/// video-curve candidate assembled by `video_admission::fitted_or_floor_phase_peaks`; scalar floors,
/// single-phase fits, curves without residual bounds, and any prediction that reuses a binding
/// phase remain governed by [`ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]. The ordinary
/// backend estimate margin remains applied after the max-over-phases envelope.
pub const RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN: bool = true;

/// The named uncertainty one admission allowance covers, and — decisively — the term it is
/// proportionate to. See the module header for what each one covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionTerm {
    /// Nothing unpriced; the allowance is zero bytes.
    FullyPriced,
    /// Proportionate to the whole peak, because a re-capture moves the peak itself.
    SameCellRecaptureSpread,
    /// Proportionate to the floor's modelled activation (headroom) term ONLY, never to its
    /// counted weights.
    AllocatorEnvelopeOverActivation,
}

impl AdmissionTerm {
    /// Stable label for tracing/telemetry, so an admission event names the uncertainty it paid for.
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::FullyPriced => "fully_priced",
            Self::SameCellRecaptureSpread => "same_cell_recapture_spread",
            Self::AllocatorEnvelopeOverActivation => "allocator_envelope_over_activation",
        }
    }
}

/// One priced allowance: a named term and a fraction OF THAT TERM's bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmissionAllowance {
    pub term: AdmissionTerm,
    /// Fraction of the term's own bytes. It equals a fraction of the peak only for
    /// [`AdmissionTerm::SameCellRecaptureSpread`], whose term IS the peak.
    pub fraction: f64,
}

impl AdmissionAllowance {
    /// Nothing left to charge.
    pub const NONE: Self = Self {
        term: AdmissionTerm::FullyPriced,
        fraction: 0.0,
    };

    /// The bytes this allowance adds, given the peak it grades and the candidate's declared
    /// unmodeled-activation headroom.
    ///
    /// Rounded UP in integer bytes, so an admitted ceiling is never under the exact product and
    /// the GiB conversion stays a single downstream step.
    pub fn bytes(self, peak_bytes: u64, term_bytes: u64) -> u64 {
        let base = match self.term {
            AdmissionTerm::FullyPriced => return 0,
            AdmissionTerm::SameCellRecaptureSpread => peak_bytes,
            AdmissionTerm::AllocatorEnvelopeOverActivation => term_bytes,
        };
        (base as f64 * self.fraction)
            .ceil()
            .clamp(0.0, u64::MAX as f64) as u64
    }
}

/// Everything the policy needs to name a candidate's remaining uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionSubject {
    pub backend: MemoryBackend,
    pub basis: CandidateBasis,
    /// The candidate's evidence was measured under a closure digest that has since moved.
    pub closure_is_stale: bool,
    /// The portion of the peak that is a flat, phase-blind activation ALLOWANCE rather than counted
    /// weights, declared by the site that built the floor. `None` where the basis does not
    /// decompose its peak.
    pub unmodeled_activation_bytes: Option<u64>,
}

/// Same-cell recapture spread for one backend. Matched exhaustively so a new backend cannot compile
/// without choosing a value.
const fn recapture_spread(backend: MemoryBackend) -> f64 {
    match backend {
        MemoryBackend::Candle => CANDLE_RECAPTURE_SPREAD,
        MemoryBackend::Mlx => MLX_RECAPTURE_SPREAD,
    }
}

/// Allocator envelope over a declared floor's ACTIVATION term, for one backend. Matched
/// exhaustively for the same reason as [`recapture_spread`]: a new backend must name its own
/// envelope rather than inherit one measured on somebody else's allocator.
const fn floor_envelope_allowance(backend: MemoryBackend) -> f64 {
    match backend {
        MemoryBackend::Candle => CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE,
        MemoryBackend::Mlx => FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE,
    }
}

/// The one allowance the selector adds on top of a candidate's peak.
///
/// The undeclared-floor arm is the only approximation left, and it is stated rather than hidden: a
/// floor that does not publish its weights/headroom split cannot have the headroom term charged, so
/// it falls back to the backend's whole-peak accounting residual. Runtime catching owns what that
/// does not cover (E6); the fix for a lane that wants better is to declare the split, not to widen
/// the number.
pub fn admission_allowance(subject: AdmissionSubject) -> AdmissionAllowance {
    let spread = AdmissionAllowance {
        term: AdmissionTerm::SameCellRecaptureSpread,
        fraction: recapture_spread(subject.backend),
    };
    match subject.basis {
        // A measurement under the live closure is the measurement. A moved closure leaves exactly
        // the same-cell recapture question open — the cell being admitted IS the cell measured.
        CandidateBasis::Measured => {
            if subject.closure_is_stale {
                spread
            } else {
                AdmissionAllowance::NONE
            }
        }
        // The fitted per-phase laws already carry each phase's max fit/held-out residual, so what
        // remains is the recapture spread of the cell the curve was fitted through.
        CandidateBasis::EstimateFittedCurve => spread,
        // Coefficient uncertainty is inside the coefficients and the allocator envelope is inside
        // `ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`. Nothing is left for the selector to add.
        CandidateBasis::EstimateAnchorDerived => AdmissionAllowance::NONE,
        CandidateBasis::EstimateFloor => {
            if subject.unmodeled_activation_bytes.is_some() {
                AdmissionAllowance {
                    term: AdmissionTerm::AllocatorEnvelopeOverActivation,
                    fraction: floor_envelope_allowance(subject.backend),
                }
            } else {
                spread
            }
        }
    }
}

/// Structural invariants of the policy, enforced at COMPILE TIME (a violating edit fails
/// `cargo build`, not just a test lane), independent of the current corpus.
const _: () = {
    // Every spread is a real, bounded fraction of its own term. An allowance at or above 1.0 on the
    // recapture term would be a blanket doubling wearing a term's name.
    assert!(MLX_RECAPTURE_SPREAD > 0.0 && MLX_RECAPTURE_SPREAD < 1.0);
    assert!(CANDLE_RECAPTURE_SPREAD > 0.0 && CANDLE_RECAPTURE_SPREAD < 1.0);
    // The fatal-OOM lane is never charged less than the recoverable one for the SAME term.
    assert!(MLX_RECAPTURE_SPREAD >= CANDLE_RECAPTURE_SPREAD);
    // The MLX floor envelope allowance is a fraction of the ACTIVATION term, and the measured
    // envelope above active runs to ~3.1x that term on the retained image renders — above 1.0 is
    // the measurement, not a blanket doubling (see the constant's doc). The upper sanity bound
    // says only that a runaway derivation (an order of magnitude past anything measured) cannot
    // land silently.
    assert!(FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE > 0.0 && FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE < 10.0);
    assert!(
        CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE > 0.0
            && CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE < 1.0
    );
    // Same ordering rule as the recapture term: the fatal-OOM lane is never charged less than the
    // recoverable one for the SAME term.
    assert!(FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE >= CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE);
    assert!(ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE);
    assert!(RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact pins mirroring `scripts/derive-ladder-margins.mjs` output. The node-side test
    /// (`scripts/derive-ladder-margins.test.mjs`, wired into `npm run check`) is the live
    /// coupling to the derivation; this pin makes a drive-by constant edit red in `rust:check`
    /// too, without waiting for the node lane.
    #[test]
    fn constants_match_the_sc_22508_derivation() {
        assert_eq!(MLX_RECAPTURE_SPREAD, 0.1260183508475594);
        assert_eq!(CANDLE_RECAPTURE_SPREAD, 0.02);
        // Re-derived on the base it charges (epic 22505 feature-end fix round, E3): the corpus
        // max of envelope-above-active over activation-above-weights, from
        // `scripts/derive-ladder-margins.mjs#deriveFloorEnvelopeAllowance` (binding record
        // imc-d778d59acb0aae38dcbe).
        assert_eq!(FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE, 3.104_173_817_050_811);
        assert_eq!(
            CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE,
            CANDLE_RECAPTURE_SPREAD
        );
        // The retired equality pin against `ANCHOR_ALLOCATOR_ENVELOPE_MARGIN` is REFRAMED, not
        // moved: the two constants price one measured phenomenon (the MLX allocator envelope)
        // against DIFFERENT bases, so they are now different numbers by derivation. What must
        // hold is the ordering that difference entails — the fraction charged against the
        // narrower base (activation alone) is necessarily larger than the fraction charged
        // against the whole phase (weights included), because the envelope bytes are the same
        // and the base shrank.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(
                FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE
                    > sceneworks_core::memory_anchor::ANCHOR_ALLOCATOR_ENVELOPE_MARGIN,
                "the activation-based fraction must exceed the whole-phase-based fraction"
            );
        }
    }

    /// The floor envelope is a property of the ALLOCATOR, so the two lanes must not share one
    /// fraction. A candle floor that inherited the MLX allocator's 17% would be charged 8.5x its
    /// own measured accounting residual on a term candle counts exactly.
    #[test]
    fn each_backend_prices_its_own_floor_envelope() {
        let headroom = 18_000_000_000_u64;
        let mlx = admission_allowance(subject(CandidateBasis::EstimateFloor, Some(headroom)));
        let candle = admission_allowance(AdmissionSubject {
            backend: MemoryBackend::Candle,
            ..subject(CandidateBasis::EstimateFloor, Some(headroom))
        });
        assert_eq!(mlx.term, AdmissionTerm::AllocatorEnvelopeOverActivation);
        assert_eq!(candle.term, AdmissionTerm::AllocatorEnvelopeOverActivation);
        assert_eq!(mlx.fraction, FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE);
        assert_eq!(candle.fraction, CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE);
        assert!(
            candle.bytes(60_000_000_000, headroom) < mlx.bytes(60_000_000_000, headroom),
            "the recoverable lane must not be charged the fatal lane's allocator envelope"
        );
    }

    fn subject(basis: CandidateBasis, headroom: Option<u64>) -> AdmissionSubject {
        AdmissionSubject {
            backend: MemoryBackend::Mlx,
            basis,
            closure_is_stale: false,
            unmodeled_activation_bytes: headroom,
        }
    }

    /// E3, stated as arithmetic: the floor's allowance tracks its HEADROOM term and is blind to its
    /// weights. Two floors with the same headroom and wildly different weights must be charged the
    /// same bytes — which no fraction-of-the-peak margin can do.
    #[test]
    fn the_floor_allowance_is_proportionate_to_headroom_not_to_the_peak() {
        let headroom = 18_000_000_000_u64;
        let allowance = admission_allowance(subject(CandidateBasis::EstimateFloor, Some(headroom)));
        assert_eq!(
            allowance.term,
            AdmissionTerm::AllocatorEnvelopeOverActivation
        );

        let expected = (headroom as f64 * FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE).ceil() as u64;
        let small_weights = allowance.bytes(20_000_000_000 + headroom, headroom);
        let huge_weights = allowance.bytes(200_000_000_000 + headroom, headroom);
        assert_eq!(small_weights, expected);
        assert_eq!(huge_weights, expected);

        // And it does track the term: double the headroom, double the allowance.
        assert_eq!(
            allowance.bytes(20_000_000_000 + 2 * headroom, 2 * headroom),
            ((2 * headroom) as f64 * FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE).ceil() as u64
        );
    }

    /// The recapture term is the one allowance that IS a fraction of the peak, and its doc says so
    /// because the quantity that moves between captures is the peak. Pin that it is the measured
    /// spread alone — not the retired x2/x4 widenings.
    #[test]
    fn the_recapture_allowance_is_the_measured_spread_with_no_blanket_widening() {
        let stale = AdmissionSubject {
            closure_is_stale: true,
            ..subject(CandidateBasis::Measured, None)
        };
        let allowance = admission_allowance(stale);
        assert_eq!(allowance.term, AdmissionTerm::SameCellRecaptureSpread);
        assert_eq!(allowance.fraction, MLX_RECAPTURE_SPREAD);
        assert_eq!(
            allowance.bytes(100_000_000_000, 0),
            (100_000_000_000.0 * MLX_RECAPTURE_SPREAD).ceil() as u64
        );
    }

    /// The anchor derivation prices its own two terms, so the selector adds nothing. This is the
    /// bullet that lets a 60 GB derived peak reach the rungs it fits.
    #[test]
    fn an_anchor_derived_peak_carries_no_selector_allowance() {
        let allowance = admission_allowance(subject(CandidateBasis::EstimateAnchorDerived, None));
        assert_eq!(allowance.term, AdmissionTerm::FullyPriced);
        assert_eq!(allowance.bytes(60 * 1024 * 1024 * 1024, 0), 0);
    }

    /// A current measurement is the measurement; an undeclared floor states its approximation by
    /// falling back to the whole-peak accounting residual rather than to a wider invented number.
    #[test]
    fn current_measurements_and_undeclared_floors_take_their_documented_arms() {
        assert_eq!(
            admission_allowance(subject(CandidateBasis::Measured, None)).term,
            AdmissionTerm::FullyPriced
        );
        let undeclared = admission_allowance(subject(CandidateBasis::EstimateFloor, None));
        assert_eq!(undeclared.term, AdmissionTerm::SameCellRecaptureSpread);
        assert_eq!(undeclared.fraction, MLX_RECAPTURE_SPREAD);
        assert_eq!(
            admission_allowance(AdmissionSubject {
                backend: MemoryBackend::Candle,
                ..subject(CandidateBasis::EstimateFloor, None)
            })
            .fraction,
            CANDLE_RECAPTURE_SPREAD
        );
    }
}
