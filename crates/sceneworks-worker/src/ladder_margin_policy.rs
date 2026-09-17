//! Admission uncertainty above a modeled active working set.
//!
//! Measurements are used directly. Derived estimates retain the observed same-cell recapture
//! spread unless their law already includes it. Reclaimable MLX cache is not physical residency:
//! it must not be multiplied into either image or video requirements. Candle retains its distinct
//! allocation-accounting residual. Evidence-derived constants are checked by
//! `scripts/derive-ladder-margins.test.mjs`.

use gen_core::MemoryBackend;

use crate::memory_strategy::{AnchorDerivationLane, CandidateBasis};

/// Same-cell capture-to-capture spread on the MLX lane: the max binding-phase spread over the
/// corpus's repeat pairs (`scripts/derive-ladder-margins.mjs`, `recaptureSpread`).
///
/// UNCERTAINTY COVERED: re-running one measured cell lands on a different peak. Nothing else. The
/// epic-18093 constant doubled this number as "protection against the next capture landing outside
/// the sampled range" and then doubled it AGAIN for estimates; neither doubling named a term, and a
/// margin that cannot name its term is what E3 retires. The failure posture for what the sampled
/// range does not cover is runtime catching (E6), not a standing 4x pad on every admission.
pub const MLX_RECAPTURE_SPREAD: f64 = 0.1260183508475594;
const _: () =
    assert!(MLX_RECAPTURE_SPREAD == sceneworks_core::memory_anchor::MLX_ACTIVE_RECAPTURE_SPREAD);

/// Same-cell spread on the candle lane. The corpus has ZERO candle repeat pairs, so no spread is
/// measurable and none is invented; this is the documented accounting-residual floor instead.
///
/// UNCERTAINTY COVERED: candle evidence is deterministic live-allocation counting rather than an
/// allocator-pool envelope (10 of 16 records report observed == predicted to the byte, none show
/// reclaimable slack), so the residual is small and bounded by the accounting, not by variance.
pub const CANDLE_RECAPTURE_SPREAD: f64 = 0.02;

/// Historical maximum reclaimable-cache/activation ratio, retained for the evidence audit.
/// This is NOT an admission allowance: `Phase::non_reclaimable_bytes` explicitly excludes
/// reclaimable MLX cache from physical-memory requirements. The population is predominantly
/// FLUX.2 on a large host and cannot price another model's working set under pressure.
#[cfg(test)]
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

/// The uncertainty allowance on a candidate's active peak. MLX uses measured recapture
/// spread for every estimated peak; cache is reclaimable. Candle preserves its existing
/// activation-only accounting allowance where the producer declares that split.
pub fn admission_allowance(subject: AdmissionSubject) -> AdmissionAllowance {
    let spread = AdmissionAllowance {
        term: AdmissionTerm::SameCellRecaptureSpread,
        fraction: recapture_spread(subject.backend),
    };
    match subject.basis {
        // A measurement is the measurement. There is no stale arm (sc-22738): the recapture
        // spread used to be charged to a measured cell whose provider closure had moved, which
        // made a shared-engine fix silently widen — and on a tight host refuse — a request the
        // measurement admitted the day before.
        CandidateBasis::Measured => AdmissionAllowance::NONE,
        // The fitted per-phase laws already carry each phase's max fit/held-out residual, so what
        // remains is the recapture spread of the cell the curve was fitted through.
        CandidateBasis::EstimateFittedCurve => spread,
        // Video derivation includes its lane's uncertainty once, so do not add it again.
        CandidateBasis::EstimateAnchorDerived {
            lane: AnchorDerivationLane::Video,
        } => AdmissionAllowance::NONE,
        // The image law widens nothing (sc-22663), so what remains over its derived peak is the
        // re-capture spread of the cell it was derived from — the same term a fitted curve pays.
        CandidateBasis::EstimateAnchorDerived {
            lane: AnchorDerivationLane::Image,
        } => spread,
        CandidateBasis::EstimateFloor => {
            // MLX cache is elastic and reclaimed under pressure. Charging a historical
            // cache/activation ratio as mandatory residency contradicted the v5 evidence
            // contract and made optimized image AND video floors larger than resident ones.
            if subject.backend == MemoryBackend::Mlx {
                return spread;
            }
            if subject.unmodeled_activation_bytes.is_some() {
                AdmissionAllowance {
                    term: AdmissionTerm::AllocatorEnvelopeOverActivation,
                    fraction: CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE,
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
    assert!(
        CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE > 0.0
            && CANDLE_FLOOR_ALLOCATOR_ENVELOPE_ALLOWANCE < 1.0
    );
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
        assert_eq!(mlx.term, AdmissionTerm::SameCellRecaptureSpread);
        assert_eq!(candle.term, AdmissionTerm::AllocatorEnvelopeOverActivation);
        assert_eq!(mlx.fraction, MLX_RECAPTURE_SPREAD);
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
            unmodeled_activation_bytes: headroom,
        }
    }

    #[test]
    fn mlx_floor_prices_active_peak_without_a_cross_family_cache_surcharge() {
        let peak = 5 * 1024 * 1024 * 1024;
        for activation in [None, Some(0), Some(peak / 2), Some(peak)] {
            let allowance = admission_allowance(subject(CandidateBasis::EstimateFloor, activation));
            assert_eq!(allowance.term, AdmissionTerm::SameCellRecaptureSpread);
            assert_eq!(
                allowance.bytes(peak, activation.unwrap_or(0)),
                (peak as f64 * MLX_RECAPTURE_SPREAD).ceil() as u64
            );
            assert!(peak + allowance.bytes(peak, activation.unwrap_or(0)) < 6 * 1024 * 1024 * 1024);
        }
    }

    /// The recapture term is the one allowance that IS a fraction of the peak, and its doc says so
    /// because the quantity that moves between captures is the peak. Pin that it is the measured
    /// spread alone — not the retired x2/x4 widenings.
    #[test]
    fn the_recapture_allowance_is_the_measured_spread_with_no_blanket_widening() {
        let allowance = admission_allowance(subject(CandidateBasis::EstimateFittedCurve, None));
        assert_eq!(allowance.term, AdmissionTerm::SameCellRecaptureSpread);
        assert_eq!(allowance.fraction, MLX_RECAPTURE_SPREAD);
        assert_eq!(
            allowance.bytes(100_000_000_000, 0),
            (100_000_000_000.0 * MLX_RECAPTURE_SPREAD).ceil() as u64
        );
    }

    /// The VIDEO anchor derivation prices its own terms (bounded coefficients, every phase widened
    /// by the allocator-envelope margin), so the selector adds nothing. This is the bullet that
    /// lets a 60 GB derived peak reach the rungs it fits.
    #[test]
    fn a_video_anchor_derived_peak_is_not_widened_twice() {
        let allowance = admission_allowance(subject(
            CandidateBasis::EstimateAnchorDerived {
                lane: AnchorDerivationLane::Video,
            },
            None,
        ));
        assert_eq!(allowance.term, AdmissionTerm::FullyPriced);
        assert_eq!(allowance.bytes(60 * 1024 * 1024 * 1024, 0), 0);
    }

    /// The IMAGE law widens nothing (sc-22663), so an image-lane anchor derivation is charged the
    /// backend's same-cell recapture spread on the whole peak — the term a fitted curve pays —
    /// and never the video lane's zero.
    #[test]
    fn an_image_anchor_derived_peak_is_charged_the_recapture_spread() {
        for (backend, spread) in [
            (MemoryBackend::Mlx, MLX_RECAPTURE_SPREAD),
            (MemoryBackend::Candle, CANDLE_RECAPTURE_SPREAD),
        ] {
            let allowance = admission_allowance(AdmissionSubject {
                backend,
                ..subject(
                    CandidateBasis::EstimateAnchorDerived {
                        lane: AnchorDerivationLane::Image,
                    },
                    None,
                )
            });
            assert_eq!(allowance.term, AdmissionTerm::SameCellRecaptureSpread);
            assert_eq!(allowance.fraction, spread);
            let peak = 60 * 1024 * 1024 * 1024;
            assert_eq!(
                allowance.bytes(peak, 0),
                (peak as f64 * spread).ceil() as u64,
                "{backend:?}: the image lane pays the recapture spread on the whole peak"
            );
            assert!(allowance.bytes(peak, 0) > 0);
        }
    }

    /// A measurement is the measurement; an undeclared floor states its approximation by falling
    /// back to the whole-peak accounting residual rather than to a wider invented number.
    ///
    /// sc-22738: the policy cannot even be told a measurement is stale — `AdmissionSubject` has no
    /// such field — so a measured cell is fully priced under every closure. MUTATION: re-adding a
    /// `closure_is_stale` arm that charges the spread turns the first assertion red only if the
    /// field comes back; the structural guard (`scripts/runtime-admission-currency.test.mjs`)
    /// keeps the field out.
    #[test]
    fn measurements_and_undeclared_floors_take_their_documented_arms() {
        assert_eq!(
            admission_allowance(subject(CandidateBasis::Measured, None)),
            AdmissionAllowance::NONE
        );
        assert_eq!(
            admission_allowance(AdmissionSubject {
                backend: MemoryBackend::Candle,
                ..subject(CandidateBasis::Measured, None)
            }),
            AdmissionAllowance::NONE
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
