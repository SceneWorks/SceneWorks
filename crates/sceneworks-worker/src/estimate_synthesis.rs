//! **The two-basis estimate synthesis mechanism, backend-generic** (sc-19050, epic 19048 R1).
//!
//! Epic 19048 R1: *"One prediction mechanism, two bases. Measured-evidence extrapolation (exact
//! calibration identity, geometry-scaled) + declared floor. Geometry axes (area, frames, batch,
//! reference_count) defined once at the mechanism level; a new axis lands in one place and both
//! lanes get it. No prediction law lives in a backend-local module; backends supply budget probe,
//! margin, and failure posture as parameters."*
//!
//! This module is that home. Everything here was extracted verbatim from
//! `crates/sceneworks-worker/src/mlx_fit_gate.rs` (sc-18096's `collect_estimate_bases` filter,
//! `synthesize_estimate_ladder`, `estimate_floor_weights_bytes`, `estimate_floor_parameters`,
//! `estimate_evidence`, `binding_phase`) — pure code motion, zero admission changes, proven by
//! [`crate::mlx_admission_decisions`] whose committed artifact was emitted by the PRE-extraction
//! gate and is re-verified against this module's output.
//!
//! # What is a parameter, and why
//!
//! | backend-supplied | how it arrives | why it cannot be a constant here |
//! |---|---|---|
//! | backend identity | [`EstimateLane::backend`] | the evidence key is lane-scoped |
//! | headroom law | [`EstimateRequest::headroom_bytes`] | MLX charges a fixed OS/app reserve plus an area-scaled activation transient (`MlxRequestPlan::generic_headroom_bytes`); candle charges `vram_gate::HEADROOM_GB` |
//! | margin | [`EstimateLane::estimate_margin`] | derived per lane in [`crate::ladder_margin_policy`] from that lane's own repeat-capture corpus |
//! | failure posture | [`EstimateLane::failure_posture`] | an MLX allocator overshoot aborts the process; a CUDA OOM is a recoverable `Err`. This is *why* the margins differ, and recording it beside them stops a future edit from equalizing them by eye |
//!
//! The **budget probe** is the fourth backend-supplied input R1 names, and it is deliberately absent
//! from this module's signatures: synthesis produces candidates, it never grades them against a
//! budget. Grading is [`crate::memory_strategy::select_strategy`], which already takes the probed
//! `MemoryBudget` as a parameter — MLX from `mlx_fit_gate::live_request_budget`, candle from
//! `vram_gate`'s `VramBudget`. Adding an unread probe field here would be a speculative parameter,
//! not a satisfied requirement.
//!
//! # The two bases, in preference order
//!
//! 1. **Fitted basis** — a verified measured cell of the same provider/tier/mode/overlay at a
//!    different geometry ([`MeasuredRungBasis`]), extrapolated over the scaling regressor
//!    ([`voxels`]). Gated by the binding-phase constraint
//!    ([`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]).
//! 2. **Weights + headroom floor** — [`floor_weights_bytes`] plus the lane's own headroom law.
//!
//! The estimate margin is NOT applied here — the selector owns margin widening
//! (`memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.

use gen_core::{
    MemoryConformanceState, MemoryDecodePolicyQuery, MemoryDecodeQualityDisposition,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryEvidenceVerdict,
    MemoryGeometry, MemoryNumericTier, MemoryParityContract, MemoryParityResult,
    MemoryProviderContract, MemorySelection, MemoryStrategy,
};
use sceneworks_core::memory_calibration::{Geometry as CalibrationGeometry, StrategyRung};

use crate::memory_strategy::{memory_mode_from_mode_key, CandidateBasis};

/// The request-scope evidence revision stamped on every synthesized candidate. Lives here rather
/// than in `mlx_fit_gate` because the video lane already stamps the same value through the same
/// constructor; it identifies the REQUEST-SCOPE evidence shape, not a backend.
pub(crate) const REQUEST_EVIDENCE_REVISION: &str = "sc-15507-request-scope-v1";

/// The inference contract revision stamped alongside it. Same reasoning.
pub(crate) const INFERENCE_CONTRACT_REVISION: &str = "1c4354b4b22d7f2cf5c4ea5fe17a83ab6c655e82";

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Geometry axes — defined ONCE (epic 19048 R1)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The geometry axes the mechanism knows about, in the order R1 names them.
///
/// A PIN, not a declaration: the axes are really defined by [`voxels`] (the regressor),
/// [`basis_geometry_is_scalable`] (which axes an extrapolation may span) and the evidence key. This
/// list exists so the set is asserted rather than narrated, and so a fifth axis is a visible edit
/// here as well as in those two places.
#[cfg(test)]
const GEOMETRY_AXES: [&str; 4] = ["area", "frames", "batch", "reference_count"];

/// **The scaling regressor: voxels = area x frames.**
///
/// sc-18812 established the discipline (extrapolate within a measured hull, over the regressor the
/// workload actually scales in); sc-18829 landed `frames` as a first-class axis on the MLX basis
/// filter. Defining the regressor as voxels rather than area is what makes the frames axis land in
/// ONE place for both lanes.
///
/// This is behaviour-preserving for every basis this mechanism will accept today:
/// [`basis_geometry_is_scalable`] requires the basis and the request to agree on `frames`, so
/// `request_voxels / basis_voxels` reduces exactly to the area ratio the sc-18096 law used. The
/// axis becomes live the moment a lane relaxes that conjunct with a fitted temporal term behind it
/// — which is a decision change, and therefore a later slice's, not this extraction's.
pub(crate) const fn voxels(width: u32, height: u32, frames: u32) -> u128 {
    (width as u128) * (height as u128) * (frames as u128)
}

/// [`voxels`] of a request geometry.
pub(crate) const fn request_voxels(geometry: MemoryGeometry) -> u128 {
    voxels(geometry.width, geometry.height, geometry.frames)
}

/// [`voxels`] of a measured calibration cell.
pub(crate) const fn basis_voxels(geometry: CalibrationGeometry) -> u128 {
    voxels(geometry.width, geometry.height, geometry.frames)
}

/// Whether a measured cell may seed an extrapolation for this request.
///
/// The cell must differ from the request (an identical cell is exact evidence, not a basis) and
/// must differ ONLY in axes the regressor spans. `batch` and `frames` are held equal: a different
/// batch or frame count is a different workload SHAPE, not a scalable geometry, and no per-axis
/// term has been fitted for either. Extracted as-is from `mlx_fit_gate::collect_estimate_bases`
/// (the `frames` conjunct is sc-18829's, taken over unchanged).
///
/// `reference_count` is deliberately NOT compared: it does not exist on
/// [`CalibrationGeometry`] — the calibration cell identity is `(width, height, batch, frames)` —
/// so a basis cannot disagree about it. The axis is still named in [`GEOMETRY_AXES`] because it
/// exists on the REQUEST side ([`MemoryGeometry::reference_count`]) and rides the evidence key;
/// the day it gains a calibration axis, this predicate is where it lands.
pub(crate) fn basis_geometry_is_scalable(
    basis: CalibrationGeometry,
    request: CalibrationGeometry,
) -> bool {
    basis != request && basis.batch == request.batch && basis.frames == request.frames
}

/// The extrapolation scale from a measured cell to the request, floored at 1.0.
///
/// The floor is the conservative reading of the corpus, not a leftover: below 1024² the measured
/// transient stops falling off proportionally (illustrious 0.305x and qwen 0.512x of their anchors
/// at 0.25x area, both ABOVE the 0.25x a proportional term predicts), so a smaller-than-measured
/// request never predicts below the measurement.
pub(crate) fn extrapolation_scale(basis: CalibrationGeometry, request: MemoryGeometry) -> f64 {
    let measured = basis_voxels(basis) as f64;
    if measured <= 0.0 {
        return 1.0;
    }
    (request_voxels(request) as f64 / measured).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The binding phase — ONE argmax, three lanes
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The canonical measurement phases, in ladder order.
///
/// A property of the GEOMETRY, not of the model: the binding phase is measured to flip inside a
/// single model's envelope (sc-18812). Every lane labels these phases in its own vocabulary
/// (`vram_gate::BindingPhase::Text`, `video_admission::VideoBindingPhase::Conditioning`) but there
/// is exactly one rule for picking one, and it is [`binding_phase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindingPhase {
    Conditioning,
    Denoise,
    Decode,
}

impl BindingPhase {
    /// The phase index (0 conditioning, 1 denoise, 2 decode) the sc-18097 comparison seam is
    /// written in. Lives on the shared type so the Krea ladder does not have to keep a lane-local
    /// copy of the ordering.
    ///
    /// Its only production consumer is `vram_gate::krea_binding_phase`, which compiles on the
    /// candle lane alone — hence the dead-code allowance on every other build, the same shape
    /// `lib.rs` uses for the candle-only modules themselves.
    #[cfg_attr(
        not(all(not(target_os = "macos"), feature = "backend-candle")),
        allow(dead_code)
    )]
    pub(crate) const fn index(self) -> u8 {
        match self {
            Self::Conditioning => 0,
            Self::Denoise => 1,
            Self::Decode => 2,
        }
    }
}

/// The phase carrying the peak of a `(conditioning, denoise, decode)` triple.
///
/// Ties resolve to the LATER phase deterministically. Generic in the magnitude type because the
/// lanes carry different ones — MLX and the video lane predict `u64` bytes, the Krea turbo ladder
/// predicts `f64` GB — while the rule is identical. `>=` on a NaN is false, so a NaN phase can
/// never claim the binding label, which is the fail-closed direction.
///
/// Before sc-19050 this rule existed three times, each copy documented as "mirroring" the others.
/// That is precisely the drift hazard epic 19048 R1 forbids: three mirrors have three chances to
/// stop mirroring.
pub(crate) fn binding_phase<T: PartialOrd>(conditioning: T, denoise: T, decode: T) -> BindingPhase {
    let mut phase = BindingPhase::Conditioning;
    let mut peak = conditioning;
    if denoise >= peak {
        phase = BindingPhase::Denoise;
        peak = denoise;
    }
    if decode >= peak {
        phase = BindingPhase::Decode;
    }
    phase
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Rung <-> calibration-rung mapping
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The calibration rung a gen-core strategy records evidence under.
pub(crate) const fn strategy_rung(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

/// The inverse of [`strategy_rung`].
pub(crate) const fn evidence_strategy(rung: StrategyRung) -> MemoryStrategy {
    match rung {
        StrategyRung::Resident => MemoryStrategy::Resident,
        StrategyRung::StagedResidency => MemoryStrategy::StagedResidency,
        StrategyRung::BoundedDecode => MemoryStrategy::BoundedDecode,
        StrategyRung::BoundedAttention => MemoryStrategy::BoundedAttention,
        StrategyRung::BoundedTransformerResidency => MemoryStrategy::BoundedTransformerResidency,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Lane parameters
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// What an admission overshoot costs on a lane. This is the *reason* the two lanes' margins differ,
/// carried beside them so a future edit cannot equalize the numbers without confronting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailurePosture {
    /// MLX: the default error handler calls `exit(-1)`; there is no recoverable `Err` on that lane,
    /// so an under-prediction takes the whole worker process down mid-render.
    FatalProcessAbort,
    /// candle/CUDA: an allocation failure is a recoverable `Err` that fails the job while the
    /// worker survives.
    RecoverableError,
}

/// Everything the mechanism needs to know about the backend it is synthesizing for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EstimateLane {
    pub(crate) backend: gen_core::MemoryBackend,
    /// Stable tracing label for this lane's synthesis events.
    pub(crate) label: &'static str,
    pub(crate) failure_posture: FailurePosture,
}

impl EstimateLane {
    /// The margin the SELECTOR will widen this lane's candidates by.
    ///
    /// **Read from the one per-backend lookup, never restated.** Copying the constant into this
    /// struct would create a second declaration of a derived number — the drift hazard this module
    /// exists to remove — so the lane carries the POSTURE (which is nowhere else expressed as data)
    /// and defers the margin to `memory_strategy::estimate_margin`.
    pub(crate) const fn estimate_margin(self) -> f64 {
        crate::memory_strategy::estimate_margin(self.backend)
    }
}

/// The MLX lane: an allocator overshoot aborts the process, which is why its margin is the wider
/// one.
pub(crate) const MLX_LANE: EstimateLane = EstimateLane {
    backend: gen_core::MemoryBackend::Mlx,
    label: "mlx",
    failure_posture: FailurePosture::FatalProcessAbort,
};

/// The candle lane: an allocation failure is a recoverable `Err`, which is why its margin is the
/// narrower one. Consumed by the video gate, which runs on both lanes.
pub(crate) const CANDLE_LANE: EstimateLane = EstimateLane {
    backend: gen_core::MemoryBackend::Candle,
    label: "candle",
    failure_posture: FailurePosture::RecoverableError,
};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The measured basis
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// One verified measured cell usable as the extrapolation basis for a fitted estimate (sc-18096):
/// same provider, tier, mode, and overlay as the request, artifact-current AND closure-current
/// binding, but a geometry that differs only in the scalable axes
/// ([`basis_geometry_is_scalable`]) — the cell the request itself could not be admitted on.
///
/// Closure-current is a deliberate restriction, not an oversight: the estimate margin was derived
/// to cover extrapolation error on top of same-closure re-capture variance
/// ([`crate::ladder_margin_policy`]). A stale-closure record already carries its own corpus-derived
/// drift allowance on the MEASURED path; stacking that drift under an extrapolation would spend the
/// estimate margin twice, and no derivation covers the sum — so a stale record may keep serving its
/// own cell behind the stale margin (sc-18095) but may not seed an extrapolated estimate.
///
/// Everything the extrapolation, the binding-phase constraint, and the loaded-provider identity
/// gate need is captured here, so synthesis never re-reads the evidence bundle.
#[derive(Clone, Debug)]
pub(crate) struct MeasuredRungBasis {
    pub(crate) rung: StrategyRung,
    pub(crate) parameters: gen_core::MemoryStrategyParameters,
    pub(crate) engaged_composition: Vec<MemoryStrategy>,
    pub(crate) load_shape: gen_core::LoadShape,
    /// The calibration identity the basis binding was measured under. [`synthesize_estimate_ladder`]
    /// requires it to equal the LOADED contract's identity: a provider whose estimator drifted from
    /// the packaged records must not receive fitted candidates built from them (sc-18096 review).
    /// This gate cannot be left to the `carries_verified_claim` demotion, which only fires when the
    /// route carries a verified claim — bases ride on legacy routes where that may not hold.
    pub(crate) calibration_abi: u32,
    pub(crate) calibration_fingerprint: String,
    pub(crate) geometry: CalibrationGeometry,
    /// Per-phase predicted peaks from the measured record, in canonical phase order
    /// (conditioning, denoise, decode). The binding phase is their argmax.
    pub(crate) conditioning_peak_bytes: u64,
    pub(crate) denoise_peak_bytes: u64,
    pub(crate) decode_peak_bytes: u64,
    /// The measured admission envelope peak the extrapolated estimate scales.
    pub(crate) envelope_peak_bytes: u64,
    pub(crate) record_id: String,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Synthesized candidates
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// One estimate-backed candidate synthesized for an implemented-but-unmeasured rung (sc-18096).
#[derive(Clone, Debug)]
pub(crate) struct SynthesizedEstimate {
    pub(crate) selection: MemorySelection,
    pub(crate) evidence: MemoryEvidence,
    pub(crate) basis: CandidateBasis,
    pub(crate) decode_quality: DecodeQualityRequestDecision,
}

/// What the provider's declared decode-quality scope said about this request's tile domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DecodeQualityRequestDecision {
    NotDeclared,
    NotEngaged {
        strategy: MemoryStrategy,
    },
    Admitted {
        strategy: MemoryStrategy,
        tile_edge: u32,
        overlap: u32,
        evidence_sha256: String,
    },
    Refused {
        strategy: MemoryStrategy,
        tile_edge: Option<u32>,
        overlap: Option<u32>,
        evidence_sha256: Option<String>,
        reason: String,
    },
    Unmeasured {
        strategy: MemoryStrategy,
        reason: String,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct EstimateParameterCandidate {
    pub(crate) parameters: gen_core::MemoryStrategyParameters,
    pub(crate) decode_quality: DecodeQualityRequestDecision,
}

#[derive(Default)]
pub(crate) struct SynthesizedEstimateLadder {
    pub(crate) estimates: Vec<SynthesizedEstimate>,
    pub(crate) decode_quality_decisions: Vec<DecodeQualityRequestDecision>,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Fabricated evidence for a synthesized candidate
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Fabricated evidence for a synthesized estimate candidate, following the
/// `generic_mlx_shared_observation` / `resident_evidence` pattern: `ImplementedUnverified`
/// conformance, no observed peak, parity not run — the record claims exactly what an estimate can
/// claim and nothing more. The selector's estimate-scoped eligibility wrap (sc-18096,
/// `memory_strategy::candidate_exclusion`) is what admits it.
///
/// `backend` is a parameter rather than a constant because sc-18814 routes the VIDEO lane through
/// this same shape on both backends.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_evidence(
    contract: &MemoryProviderContract,
    backend: gen_core::MemoryBackend,
    tier: MemoryNumericTier,
    mode: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    selection: MemorySelection,
    predicted_peak_bytes: u64,
    calibration_fingerprint: Option<&str>,
) -> MemoryEvidence {
    MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: contract.provider_id.clone(),
            backend,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
            overlay: overlay.map(str::to_owned),
            geometry,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition_for_selection(&selection),
            parameters: selection.parameters,
        },
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
        sceneworks_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        inference_revision: INFERENCE_CONTRACT_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Basis 2 — the weights + headroom floor
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The floor's per-rung WEIGHTS term, derived only from the provider contract's own declarations
/// (sc-18096). Nothing here is a tuned coefficient:
///
/// * `StagedResidency` engaged ⇒ the co-residency drop the rung exists for: the resident working
///   set is the larger of the conditioning stack and everything else, exactly the
///   `staged_weights_gb` split the load-time gate has always used.
/// * `BoundedTransformerResidency` engaged ⇒ the transformer's declared bytes leave the resident
///   floor: the rung windows them, and the window slice plus scratch is carried by the headroom
///   term and the estimate margin, not by a guessed window fraction.
/// * Rungs 2 and 3 bound TRANSIENTS, not weights, so they take no weights reduction here — and
///   deliberately no transient reduction either, because no measured basis for one exists on an
///   unmeasured cell. Their floor equals rung 1's, which keeps them selectable without ever
///   promising an unmeasured saving.
/// * Auxiliary components (control branches, adapter stacks, …) stay resident unless the contract
///   itself declares them `bounded_by` a rung the composition engages.
pub(crate) fn floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> u64 {
    let facts = contract.asset_facts;
    let conditioning = facts.conditioning_bytes;
    let mut heavy = facts.base_bytes.saturating_sub(conditioning);
    if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
        heavy = heavy.saturating_sub(facts.transformer_bytes);
    }
    let base = if engaged.contains(&MemoryStrategy::StagedResidency) {
        conditioning.max(heavy)
    } else {
        conditioning.saturating_add(heavy)
    };
    let auxiliary = contract
        .resident_components()
        .iter()
        .filter(|component| component.kind.is_auxiliary())
        .filter(|component| match component.bounded_by {
            Some(bounding) => !engaged.contains(&bounding),
            None => true,
        })
        .fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        });
    base.saturating_add(auxiliary)
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as far
/// below the floor's unreduced headroom charge as the provider allows. `None` when a required knob
/// has no declared range: such a selection cannot be validated, so no candidate is synthesized for
/// the rung.
///
/// This is the whole parameter law for the video lane and for the candle floor arm. The image
/// lane's [`floor_parameter_candidates`] wraps it with the decode-quality decisions those two have
/// no use for.
pub(crate) fn floor_smallest_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
    Some(gen_core::MemoryStrategyParameters {
        decode_tile_edge: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedDecode,
            |r| &r.decode_tile_edges,
        )?,
        decode_overlap: smallest_declared(contract, engaged, MemoryStrategy::BoundedDecode, |r| {
            &r.decode_overlaps
        })?,
        attention_chunk_size: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedAttention,
            |r| &r.attention_chunk_sizes,
        )?,
        transformer_window_size: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedTransformerResidency,
            |r| &r.transformer_window_sizes,
        )?,
        transformer_window_component: None,
    })
}

/// `Some(None)` when the rung is not engaged (the knob is not required), `Some(Some(v))` for the
/// smallest declared value, `None` when the rung IS engaged but publishes no range — the
/// unvalidatable case every caller turns into "synthesize nothing for this rung".
fn smallest_declared(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    strategy: MemoryStrategy,
    pick: fn(&gen_core::MemoryParameterRanges) -> &Vec<u32>,
) -> Option<Option<u32>> {
    if !engaged.contains(&strategy) {
        return Some(None);
    }
    pick(&contract.capability(strategy)?.parameters)
        .iter()
        .copied()
        .min()
        .map(Some)
}

/// The image lane's per-candidate parameter builder: [`floor_smallest_parameters`]'s knobs plus the
/// declared decode-quality scope's admitted tile domain, one candidate per admitted `(edge,
/// overlap)` row, with every refusal/unmeasured decision reported alongside.
#[allow(clippy::too_many_arguments)]
pub(crate) fn floor_parameter_candidates(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    strategy: MemoryStrategy,
    tier: MemoryNumericTier,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    use_pid: bool,
) -> (
    Vec<EstimateParameterCandidate>,
    Vec<DecodeQualityRequestDecision>,
) {
    let Some(attention_chunk_size) =
        smallest_declared(contract, engaged, MemoryStrategy::BoundedAttention, |r| {
            &r.attention_chunk_sizes
        })
    else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "required attention chunk range is empty".to_owned(),
            }],
        );
    };
    let Some(transformer_window_size) = smallest_declared(
        contract,
        engaged,
        MemoryStrategy::BoundedTransformerResidency,
        |r| &r.transformer_window_sizes,
    ) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "required transformer window range is empty".to_owned(),
            }],
        );
    };
    let base_parameters = gen_core::MemoryStrategyParameters {
        decode_tile_edge: None,
        decode_overlap: None,
        attention_chunk_size,
        transformer_window_size,
        transformer_window_component: None,
    };
    if !engaged.contains(&MemoryStrategy::BoundedDecode) {
        return (
            vec![EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            }],
            Vec::new(),
        );
    }
    let Some(decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode capability is absent".to_owned(),
            }],
        );
    };
    if decode.parameters.decode_geometry_policies.is_empty() {
        if contract.decode_geometry_policy_authoritative {
            let refusal = DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode has a declared quality scope but no applicable row for the loaded tier/load shape"
                    .to_owned(),
            };
            // The decode rung itself has no exact authority. A higher independent rung retains its
            // own estimate with decode omitted, so request-selection-aware engagement cannot
            // accidentally resurrect the route-blind tile domain.
            let candidates = (strategy != MemoryStrategy::BoundedDecode)
                .then_some(EstimateParameterCandidate {
                    parameters: base_parameters,
                    decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
                })
                .into_iter()
                .collect();
            return (candidates, vec![refusal]);
        }
        let pair = if let Some(routes) = &contract.pid_decode_routes {
            let route = if use_pid { &routes.pid } else { &routes.native };
            route
                .tile_edges
                .iter()
                .copied()
                .min()
                .zip(Some(route.tile_overlap))
        } else {
            decode
                .parameters
                .decode_tile_edges
                .iter()
                .copied()
                .min()
                .zip(decode.parameters.decode_overlaps.iter().copied().min())
        };
        let Some((tile_edge, overlap)) = pair else {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: "legacy bounded decode has no complete edge/overlap pair".to_owned(),
                }],
            );
        };
        let mut parameters = base_parameters;
        parameters.decode_tile_edge = Some(tile_edge);
        parameters.decode_overlap = Some(overlap);
        return (
            vec![EstimateParameterCandidate {
                parameters,
                decode_quality: DecodeQualityRequestDecision::NotDeclared,
            }],
            Vec::new(),
        );
    }

    let query = MemoryDecodePolicyQuery {
        tier,
        load_shape: contract.load_shape,
        mode_key,
        overlay,
        geometry,
        use_pid,
    };
    let rows = match contract.decode_geometry_policies_for_request(query) {
        Ok(rows) => rows,
        Err(error) => {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: error.to_string(),
                }],
            )
        }
    };
    let mut rows = rows;
    rows.sort_by_key(|policy| (policy.tile_edge, policy.overlap));
    let mut candidates = Vec::new();
    let mut decisions = Vec::new();
    for policy in rows {
        match &policy.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                let mut parameters = base_parameters;
                parameters.decode_tile_edge = Some(policy.tile_edge);
                parameters.decode_overlap = Some(policy.overlap);
                candidates.push(EstimateParameterCandidate {
                    parameters,
                    decode_quality: DecodeQualityRequestDecision::Admitted {
                        strategy,
                        tile_edge: policy.tile_edge,
                        overlap: policy.overlap,
                        evidence_sha256: policy.production_evidence_sha256.clone(),
                    },
                });
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                decisions.push(DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: Some(policy.tile_edge),
                    overlap: Some(policy.overlap),
                    evidence_sha256: Some(policy.production_evidence_sha256.clone()),
                    reason: reason.clone(),
                });
            }
        }
    }
    if candidates.is_empty() {
        if decisions.is_empty() {
            decisions.push(DecodeQualityRequestDecision::Unmeasured {
                strategy,
                reason: format!(
                    "no exact production-latent row for {}x{} load_shape={:?} mode={} overlay={overlay:?} use_pid={use_pid}",
                    geometry.width, geometry.height, contract.load_shape, mode_key,
                ),
            });
        }
        // Bounded decode itself cannot exist without an admitted pair. A higher independent rung
        // keeps its own parameters and omits decode so selection-aware engagement excludes rung 2.
        if strategy != MemoryStrategy::BoundedDecode {
            candidates.push(EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            });
        }
    }
    (candidates, decisions)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The synthesis entry point
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Everything one lane's request contributes to synthesis. Grouped into a struct rather than passed
/// as eleven positional arguments so a caller cannot transpose two `&str`s silently.
pub(crate) struct EstimateRequest<'a> {
    pub(crate) lane: EstimateLane,
    pub(crate) contract: &'a MemoryProviderContract,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) mode_key: &'a str,
    pub(crate) overlay: Option<&'a str>,
    pub(crate) geometry: MemoryGeometry,
    pub(crate) use_pid: bool,
    pub(crate) calibration_fingerprint: Option<&'a str>,
    /// **The backend-supplied headroom law**, already evaluated at this request's geometry: the
    /// fixed reserve plus the geometry-scaled activation transient this lane charges. MLX passes
    /// `MlxRequestPlan::generic_headroom_bytes(geometry)` — the exact same headroom convention its
    /// resident baseline charges, so only the weights term differs per rung.
    pub(crate) headroom_bytes: u64,
}

/// Synthesize estimate-backed candidates for every optimized rung the provider contract marks
/// `Implemented` (sc-18096, epic 18093 R1a). Called only on legacy admission routes — a covered
/// cell is authorized by its exact measured ladder and gets no synthetic sibling.
///
/// Peak source per rung, in preference order:
///
/// 1. **Fitted basis** — a verified measured cell at a different geometry ([`MeasuredRungBasis`]),
///    extrapolated over [`voxels`]: the conditioning peak is regressor-flat (text encoding does not
///    grow with the render target) while denoise, decode, and the admission envelope scale by
///    [`extrapolation_scale`]. Gated by
///    [`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]: if the
///    extrapolated triple's binding phase differs from the measured cell's, the fitted candidate is
///    NOT emitted (no per-phase variance re-derivation exists) and the rung falls back to the
///    floor, whose no-measured-basis path the constraint's scope sentence explicitly exempts.
/// 2. **Weights + headroom floor** — [`floor_weights_bytes`] plus
///    [`EstimateRequest::headroom_bytes`].
pub(crate) fn synthesize_estimate_ladder(
    request: &EstimateRequest<'_>,
    bases: &[MeasuredRungBasis],
) -> SynthesizedEstimateLadder {
    use crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE;

    let contract = request.contract;
    let geometry = request.geometry;
    let lane = request.lane.label;
    let mut ladder = SynthesizedEstimateLadder::default();
    for strategy in MemoryStrategy::ALL {
        if strategy == MemoryStrategy::Resident {
            // The resident baseline candidate already exists on every legacy route.
            continue;
        }
        if !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(gen_core::MemoryStrategySupport::Implemented)
        ) {
            continue;
        }
        let declared_engaged = contract.engaged_composition(strategy);
        let (parameter_candidates, decisions) = floor_parameter_candidates(
            contract,
            &declared_engaged,
            strategy,
            request.tier,
            request.mode_key,
            request.overlay,
            geometry,
            request.use_pid,
        );
        ladder.decode_quality_decisions.extend(decisions);

        for parameter_candidate in parameter_candidates {
            let floor_selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: request.tier,
            };
            let engaged = contract.engaged_composition_for_selection(&floor_selection);

            // 1. Fitted basis: the closest measured geometry below the request, else the smallest
            //    above it (whose clamp-at-1.0 scaling degenerates to the measurement itself). The
            //    basis must have been measured under the LOADED provider's exact calibration
            //    identity: a drifted estimator invalidates the measured numbers as an extrapolation
            //    seed, and this is the only gate on legacy routes (the `carries_verified_claim`
            //    demotion never fires without a verified claim on the route). A contract with no
            //    calibration identity gets no fitted candidates at all — fail closed.
            //
            //    The load-shape conjunct compares CONTRACT shape only — deliberately, and unlike
            //    the Evidence-path filter's measured-candidate leg, which also compares
            //    `identity.load_shape` (sc-18251). An estimate-basis candidate is graded downstream
            //    by the estimate wrap of `optimized_eligibility`, which short-circuits at the
            //    conformance gate's `Unverified` BEFORE the identity load-shape comparison ever
            //    runs, so the identity's shape is never consulted for an estimate. Adding the
            //    conjunct here would be stricter than the gate this filter anticipates.
            let fitted = bases
                .iter()
                .filter(|basis| {
                    basis.rung == strategy_rung(strategy)
                        && basis.load_shape == contract.load_shape
                        && basis.engaged_composition == engaged
                        && contract.calibration.as_ref().is_some_and(|identity| {
                            identity.abi == basis.calibration_abi
                                && identity.fingerprint == basis.calibration_fingerprint
                        })
                })
                .max_by_key(|basis| {
                    let measured = basis_voxels(basis.geometry);
                    let below = measured <= request_voxels(geometry);
                    // Rank every below-request basis above every above-request one; among "below"
                    // take the largest, among "above" the smallest.
                    (
                        below,
                        if below {
                            measured as i128
                        } else {
                            -(measured as i128)
                        },
                    )
                })
                .and_then(|basis| {
                    let mut parameters = basis.parameters;
                    parameters.decode_tile_edge = parameter_candidate.parameters.decode_tile_edge;
                    parameters.decode_overlap = parameter_candidate.parameters.decode_overlap;
                    let selection = MemorySelection {
                        strategy,
                        parameters,
                        tier: request.tier,
                    };
                    if contract.validate_selection(&selection).is_err() {
                        return None;
                    }
                    let scale = extrapolation_scale(basis.geometry, geometry);
                    let scaled = |bytes: u64| {
                        (bytes as f64 * scale).ceil().clamp(0.0, u64::MAX as f64) as u64
                    };
                    let measured_binding = binding_phase(
                        basis.conditioning_peak_bytes,
                        basis.denoise_peak_bytes,
                        basis.decode_peak_bytes,
                    );
                    let extrapolated_binding = binding_phase(
                        basis.conditioning_peak_bytes,
                        scaled(basis.denoise_peak_bytes),
                        scaled(basis.decode_peak_bytes),
                    );
                    if ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
                        && extrapolated_binding != measured_binding
                    {
                        // The pinned sc-18094 constraint: the corpus shows a 17.14% per-phase
                        // re-capture spread that no margin in the policy absorbs, so an
                        // extrapolation that moves the request peak onto a different phase than the
                        // one measured is refused rather than margined. The rung falls back to the
                        // floor path below.
                        tracing::info!(
                            route = contract.provider_id,
                            backend = lane,
                            ?strategy,
                            failure_posture = ?request.lane.failure_posture,
                            basis_record = basis.record_id,
                            measured_binding_phase = ?measured_binding,
                            extrapolated_binding_phase = ?extrapolated_binding,
                            "fitted-curve estimate rejected: extrapolation flips the binding phase \
                             (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
                        );
                        return None;
                    }
                    let predicted_peak_bytes = scaled(basis.envelope_peak_bytes);
                    tracing::info!(
                        route = contract.provider_id,
                        backend = lane,
                        ?strategy,
                        basis_record = basis.record_id,
                        basis_geometry =
                            format!("{}x{}", basis.geometry.width, basis.geometry.height),
                        raw_peak_bytes = predicted_peak_bytes,
                        estimate_margin = request.lane.estimate_margin(),
                        voxel_scale = scale,
                        "synthesized fitted-curve estimate candidate from a measured cell"
                    );
                    Some(SynthesizedEstimate {
                        selection,
                        evidence: estimate_evidence(
                            contract,
                            request.lane.backend,
                            request.tier,
                            request.mode_key,
                            request.overlay,
                            geometry,
                            selection,
                            predicted_peak_bytes,
                            request.calibration_fingerprint,
                        ),
                        basis: CandidateBasis::EstimateFittedCurve,
                        decode_quality: parameter_candidate.decode_quality.clone(),
                    })
                });
            if let Some(candidate) = fitted {
                ladder
                    .decode_quality_decisions
                    .push(candidate.decode_quality.clone());
                ladder.estimates.push(candidate);
                continue;
            }

            // 2. Weights + headroom floor — no measured basis, so the binding-phase constraint does
            //    not gate it (scope sentence on the constraint's doc).
            let selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: request.tier,
            };
            if contract.validate_selection(&selection).is_err() {
                continue;
            }
            let predicted_peak_bytes =
                floor_weights_bytes(contract, &engaged).saturating_add(request.headroom_bytes);
            tracing::info!(
                route = contract.provider_id,
                backend = lane,
                ?strategy,
                failure_posture = ?request.lane.failure_posture,
                estimate_margin = request.lane.estimate_margin(),
                raw_peak_bytes = predicted_peak_bytes,
                "synthesized weights+headroom floor estimate candidate"
            );
            let candidate = SynthesizedEstimate {
                selection,
                evidence: estimate_evidence(
                    contract,
                    request.lane.backend,
                    request.tier,
                    request.mode_key,
                    request.overlay,
                    geometry,
                    selection,
                    predicted_peak_bytes,
                    request.calibration_fingerprint,
                ),
                basis: CandidateBasis::EstimateFloor,
                decode_quality: parameter_candidate.decode_quality,
            };
            ladder
                .decode_quality_decisions
                .push(candidate.decode_quality.clone());
            ladder.estimates.push(candidate);
        }
    }
    ladder
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: u32, height: u32, frames: u32) -> CalibrationGeometry {
        CalibrationGeometry {
            width,
            height,
            batch: 1,
            frames,
        }
    }

    fn request(width: u32, height: u32, frames: u32) -> MemoryGeometry {
        MemoryGeometry {
            width,
            height,
            batch: 1,
            frames,
            reference_count: 0,
        }
    }

    /// R1's axis list is asserted, not narrated. A fifth axis must be added here and to
    /// [`basis_geometry_is_scalable`] in the same edit.
    #[test]
    fn the_mechanism_names_exactly_the_four_geometry_axes() {
        assert_eq!(
            GEOMETRY_AXES,
            ["area", "frames", "batch", "reference_count"]
        );
    }

    /// The regressor is voxels, and it REDUCES to the pre-extraction area law on every basis the
    /// scalable-axis predicate admits. This is the assertion that makes "voxels" a safe extraction
    /// rather than a decision change: `basis_geometry_is_scalable` holds frames equal, so the
    /// voxel ratio and the area ratio are the same number.
    #[test]
    fn the_voxel_regressor_equals_the_area_law_wherever_a_basis_is_admissible() {
        for (bw, bh, rw, rh, frames) in [
            (1024_u32, 1024_u32, 2048_u32, 2048_u32, 1_u32),
            (1024, 1024, 1024, 1536, 1),
            (768, 512, 1280, 704, 121),
            (2048, 2048, 1024, 1024, 9),
        ] {
            let basis = geometry(bw, bh, frames);
            let req = request(rw, rh, frames);
            assert!(
                basis_geometry_is_scalable(
                    basis,
                    CalibrationGeometry {
                        width: rw,
                        height: rh,
                        batch: 1,
                        frames
                    }
                ) || (bw == rw && bh == rh),
                "fixture must be an admissible basis"
            );
            let area_law =
                (f64::from(rw) * f64::from(rh) / (f64::from(bw) * f64::from(bh))).max(1.0);
            assert_eq!(extrapolation_scale(basis, req), area_law);
        }
    }

    /// The frames axis is LIVE in the regressor even though today's basis predicate holds it
    /// equal: this is what "a new axis lands in one place and both lanes get it" has to mean.
    #[test]
    fn the_regressor_spans_frames_not_only_area() {
        let basis = geometry(1024, 1024, 1);
        assert_eq!(extrapolation_scale(basis, request(1024, 1024, 4)), 4.0);
        assert_eq!(voxels(1024, 1024, 4), voxels(2048, 1024, 2));
    }

    /// A cell that differs in batch or frames is NOT a scalable basis, and an identical cell is
    /// exact evidence rather than a basis. Extracted as-is from `collect_estimate_bases`.
    #[test]
    fn only_the_scalable_axes_may_differ_between_a_basis_and_its_request() {
        let req = geometry(1024, 1024, 1);
        assert!(!basis_geometry_is_scalable(req, req), "identical cell");
        assert!(basis_geometry_is_scalable(geometry(768, 768, 1), req));
        assert!(
            !basis_geometry_is_scalable(geometry(768, 768, 2), req),
            "a different frame count is a different workload shape"
        );
        assert!(
            !basis_geometry_is_scalable(
                CalibrationGeometry {
                    width: 768,
                    height: 768,
                    batch: 2,
                    frames: 1
                },
                req
            ),
            "a different batch is a different workload shape"
        );
    }

    /// The scale never predicts below the measurement.
    #[test]
    fn the_extrapolation_scale_is_floored_at_one() {
        assert_eq!(
            extrapolation_scale(geometry(2048, 2048, 1), request(512, 512, 1)),
            1.0
        );
        // A degenerate zero-voxel basis cannot divide; it degenerates to the measurement.
        assert_eq!(
            extrapolation_scale(geometry(0, 1024, 1), request(1024, 1024, 1)),
            1.0
        );
    }

    /// One argmax, ties to the LATER phase, on both magnitude types the lanes carry.
    #[test]
    fn the_binding_phase_is_the_argmax_with_ties_to_the_later_phase() {
        assert_eq!(binding_phase(9_u64, 1, 1), BindingPhase::Conditioning);
        assert_eq!(binding_phase(1_u64, 9, 1), BindingPhase::Denoise);
        assert_eq!(binding_phase(1_u64, 1, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9_u64, 9, 1), BindingPhase::Denoise);
        assert_eq!(binding_phase(9_u64, 1, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9_u64, 9, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9.0_f64, 9.0, 1.0), BindingPhase::Denoise);
        assert_eq!(binding_phase(1.0_f64, 1.0, 1.0), BindingPhase::Decode);
        // NaN can never claim the binding label — the fail-closed direction.
        assert_eq!(
            binding_phase(5.0_f64, f64::NAN, f64::NAN),
            BindingPhase::Conditioning
        );
    }

    /// The rung mapping is a bijection. A new rung on either side must be added to both arms, and
    /// a transposed arm reds here rather than silently mislabelling every record.
    #[test]
    fn the_rung_mapping_round_trips_in_both_directions() {
        for strategy in MemoryStrategy::ALL {
            assert_eq!(evidence_strategy(strategy_rung(strategy)), strategy);
        }
        for rung in [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedDecode,
            StrategyRung::BoundedAttention,
            StrategyRung::BoundedTransformerResidency,
        ] {
            assert_eq!(strategy_rung(evidence_strategy(rung)), rung);
        }
    }

    /// A contract that DECLARES a decode-quality scope but publishes no admitted `(edge, overlap)`
    /// row for this request: `BoundedDecode` itself cannot exist, while a higher independent rung
    /// keeps its own estimate with decode omitted.
    ///
    /// Written for sc-19050 because this guard survived every mutation the shipped corpus could
    /// reach — the weights-free contract surfaces the decision baseline drives carry no adopted
    /// decode-geometry policies at all (production adopts them from the manifest at load time), so
    /// the `candidates.is_empty()` arm was unreachable there. Dropping the
    /// `strategy != BoundedDecode` conjunct would let rung 2 be selected with a tile domain nobody
    /// measured, which is exactly the route-blind admission the declared scope exists to stop.
    #[test]
    fn bounded_decode_cannot_survive_without_an_admitted_tile_domain() {
        let mut contract = MemoryProviderContract::compatibility_default(
            "sc19050-decode-scope",
            gen_core::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        for capability in &mut contract.strategies {
            match capability.strategy {
                MemoryStrategy::BoundedDecode => {
                    capability.support = gen_core::MemoryStrategySupport::Implemented;
                    capability.parameters.decode_tile_edges = vec![512];
                    capability.parameters.decode_overlaps = vec![64];
                    // Declared, but keyed to a DIFFERENT mode, so no row matches this request and
                    // `decode_geometry_policies_for_request` returns an empty set.
                    capability.parameters.decode_geometry_policies =
                        vec![unmatched_decode_policy()];
                }
                MemoryStrategy::BoundedAttention => {
                    capability.support = gen_core::MemoryStrategySupport::Implemented;
                    capability.parameters.attention_chunk_sizes = vec![256];
                }
                _ => {}
            }
        }

        let tier = gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        let geometry = request(1024, 1024, 1);

        let (decode_candidates, decode_decisions) = floor_parameter_candidates(
            &contract,
            &[MemoryStrategy::Resident, MemoryStrategy::BoundedDecode],
            MemoryStrategy::BoundedDecode,
            tier,
            "text_to_image",
            None,
            geometry,
            false,
        );
        assert!(
            decode_candidates.is_empty(),
            "bounded decode was offered without an admitted tile domain: {decode_candidates:?}"
        );
        assert!(
            decode_decisions.iter().any(|decision| matches!(
                decision,
                DecodeQualityRequestDecision::Unmeasured { .. }
            )),
            "the absence of an exact production-latent row must be REPORTED, not silent: \
             {decode_decisions:?}"
        );

        // A higher independent rung keeps its estimate, with decode omitted so selection-aware
        // engagement cannot resurrect the route-blind tile domain.
        let (attention_candidates, _) = floor_parameter_candidates(
            &contract,
            &[MemoryStrategy::Resident, MemoryStrategy::BoundedAttention],
            MemoryStrategy::BoundedAttention,
            tier,
            "text_to_image",
            None,
            geometry,
            false,
        );
        assert_eq!(attention_candidates.len(), 1);
        assert_eq!(attention_candidates[0].parameters.decode_tile_edge, None);
        assert_eq!(attention_candidates[0].parameters.decode_overlap, None);
        assert_eq!(
            attention_candidates[0].parameters.attention_chunk_size,
            Some(256)
        );
    }

    /// A syntactically complete decode-quality row keyed to a mode this test never requests, so the
    /// scope is DECLARED (non-empty) while no row matches — the state that reaches the
    /// `candidates.is_empty()` arm.
    fn unmatched_decode_policy() -> gen_core::MemoryDecodeGeometryPolicy {
        gen_core::MemoryDecodeGeometryPolicy {
            quality_abi: gen_core::MEMORY_DECODE_QUALITY_ABI,
            family: "sc19050".to_owned(),
            resolved_route: "sc19050-decode-scope".to_owned(),
            backend: gen_core::MemoryBackend::Mlx,
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            load_shape: gen_core::LoadShape::EagerMaterialization,
            artifact: gen_core::MemoryDecodeArtifactIdentity {
                repository: "SceneWorks/sc19050".to_owned(),
                revision: "0".repeat(40),
                variant: "bf16".to_owned(),
                fingerprint: "sc19050-fingerprint".to_owned(),
            },
            implementation_fingerprint: "sc19050-implementation".to_owned(),
            // The axis that makes this row unmatched: the request under test is text-to-image.
            mode: gen_core::MemoryMode::Edit,
            overlay: None,
            geometry: request(1024, 1024, 1),
            use_pid: false,
            tile_edge: 512,
            overlap: 64,
            metric: "sc19050".to_owned(),
            maximum_error: 0,
            fixtures: Vec::new(),
            production_evidence_sha256: "sc19050-evidence".to_owned(),
            disposition: gen_core::MemoryDecodeQualityDisposition::Admitted,
        }
    }

    /// The two lanes' postures and margins travel together: the fatal-abort lane is never the
    /// narrower one. `ladder_margin_policy`'s compile-time block pins the margin ordering; this
    /// pins that the POSTURE explaining it is attached to the right lane.
    #[test]
    fn the_fatal_lane_carries_the_wider_margin() {
        assert_eq!(MLX_LANE.failure_posture, FailurePosture::FatalProcessAbort);
        assert_eq!(
            CANDLE_LANE.failure_posture,
            FailurePosture::RecoverableError
        );
        assert!(MLX_LANE.estimate_margin() > CANDLE_LANE.estimate_margin());
        assert_eq!(MLX_LANE.backend, gen_core::MemoryBackend::Mlx);
        assert_eq!(CANDLE_LANE.backend, gen_core::MemoryBackend::Candle);
    }
}
