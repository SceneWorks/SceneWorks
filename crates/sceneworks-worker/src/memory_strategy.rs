//! Worker-owned, backend-neutral memory-strategy selection (sc-15449).
//!
//! Providers own capabilities and lifecycle hooks. Evidence owns measured request cells. The worker
//! owns live-budget arithmetic and the normative strategy order. Optimized candidates are admitted
//! only through gen-core's canonical evidence validator.
//!
//! Closure-digest currency is a signal, not a gate (sc-18095, epic 18093): a fully-verified
//! measured cell whose provider closure moved stays eligible with its peak widened by the
//! backend's stale-measured margin from [`crate::ladder_margin_policy`], and current evidence is
//! strictly preferred over stale for the same key. Structural verdicts (`Invalid`,
//! `OutOfEnvelope`, `CompositionMismatch`, unverified conformance) still exclude.
//!
//! Estimate-backed candidates admit the full ladder (sc-18096, epic 18093 R1): a caller may
//! submit synthesized candidates ([`CandidateBasis::EstimateFittedCurve`] /
//! [`CandidateBasis::EstimateFloor`]) for implemented-but-unmeasured rungs. They are graded at
//! their peak widened by the backend's ESTIMATE margin, every structural exclusion still applies,
//! and any eligible measured candidate at the same rung supersedes them — the normative
//! precedence is measured-current > stale-measured > estimate.

use std::collections::BTreeSet;

use gen_core::{
    MemoryBackend, MemoryCleanupSemantics, MemoryEvidence, MemoryEvidenceDimension,
    MemoryEvidenceVerdict, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryProviderContract,
    MemorySelection, MemoryStrategy, MemoryStrategySupport,
};
use sceneworks_core::memory_calibration::StrategyRung;

use crate::ladder_margin_policy::{
    CANDLE_ESTIMATE_MARGIN, CANDLE_STALE_MEASURED_MARGIN, MLX_ESTIMATE_MARGIN,
    MLX_STALE_MEASURED_MARGIN,
};

/// Bridge a calibration receipt's persisted materialization shape to the gen-core contract type.
/// The two enums are the same axis; `sceneworks-core` keeps its own spelling because it
/// deliberately has no gen-core dependency.
pub(crate) fn load_shape_from_receipt(
    key: sceneworks_core::memory_calibration::LoadShapeKey,
) -> gen_core::LoadShape {
    match key {
        sceneworks_core::memory_calibration::LoadShapeKey::EagerMaterialization => {
            gen_core::LoadShape::EagerMaterialization
        }
        sceneworks_core::memory_calibration::LoadShapeKey::DeferredMaterialization => {
            gen_core::LoadShape::DeferredMaterialization
        }
    }
}

/// Bridge a calibration receipt's rung spelling to the gen-core strategy it names.
pub(crate) const fn strategy_from_rung(rung: StrategyRung) -> MemoryStrategy {
    match rung {
        StrategyRung::Resident => MemoryStrategy::Resident,
        StrategyRung::StagedResidency => MemoryStrategy::StagedResidency,
        StrategyRung::BoundedDecode => MemoryStrategy::BoundedDecode,
        StrategyRung::BoundedAttention => MemoryStrategy::BoundedAttention,
        StrategyRung::BoundedTransformerResidency => MemoryStrategy::BoundedTransformerResidency,
    }
}

/// Canonical label a calibration binding and an evidence receipt both spell a rung with.
pub(crate) const fn rung_key(rung: StrategyRung) -> &'static str {
    match rung {
        StrategyRung::Resident => "resident",
        StrategyRung::StagedResidency => "staged_residency",
        StrategyRung::BoundedDecode => "bounded_decode",
        StrategyRung::BoundedAttention => "bounded_attention",
        StrategyRung::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

/// Parse one canonical rung label, the inverse of [`rung_key`].
pub(crate) fn rung_from_key(value: &str) -> Option<StrategyRung> {
    StrategyRung::ALL
        .into_iter()
        .find(|rung| rung_key(*rung) == value)
}

/// The composition a selection engages when a binding publishes no cheaper verified one.
///
/// Read through [`MemoryStrategy::engages`] — the contract's own defeasible cost-order policy seam —
/// rather than restated here as an ordinal `<=`. Restating it would let the two drift silently, and
/// the drift decides which strategy parameters a binding is obliged to name.
pub(crate) fn default_engaged_composition(rung: StrategyRung) -> Vec<StrategyRung> {
    let selection = strategy_from_rung(rung);
    StrategyRung::ALL
        .into_iter()
        .filter(|candidate| selection.engages(strategy_from_rung(*candidate)))
        .collect()
}

/// Validate a binding's declared `engagedRungs`, returning the composition to validate against.
///
/// Absent, the shared cost-order default applies, which is byte-identical to the pre-sc-17728
/// cumulative behaviour. Declared, it is authoritative — that is how a provider which implements
/// rung 4 while declaring rungs 2 and 3 `Missing` publishes evidence for the composition it actually
/// ran. The declaration is still closed: it must be canonical and unique, it must contain `resident`
/// and the selected rung, and it can never engage a rung costlier than the one it selected.
pub(crate) fn engaged_composition(
    rung: StrategyRung,
    declared: Option<&[StrategyRung]>,
) -> Result<Vec<StrategyRung>, String> {
    let Some(declared) = declared else {
        return Ok(default_engaged_composition(rung));
    };
    let unique = declared.iter().copied().collect::<BTreeSet<_>>();
    let canonical = StrategyRung::ALL
        .into_iter()
        .filter(|candidate| unique.contains(candidate))
        .collect::<Vec<_>>();
    if unique.len() != declared.len() || canonical != declared {
        return Err("engagedRungs must be a unique set in canonical ladder order".to_owned());
    }
    if !unique.contains(&StrategyRung::Resident) || !unique.contains(&rung) {
        return Err(format!(
            "engagedRungs must contain resident and the selected rung {}",
            rung_key(rung)
        ));
    }
    if let Some(costlier) = declared.iter().find(|candidate| **candidate > rung) {
        return Err(format!(
            "engagedRungs cannot engage {}, which is costlier than the selected rung {}",
            rung_key(*costlier),
            rung_key(rung)
        ));
    }
    Ok(canonical)
}

/// Numeric strategy parameters an engaged composition must name, with their minimums.
///
/// A rung the composition does not engage owns no parameter, so naming one is an error rather than
/// harmless noise — the same rule gen-core's `validate_selected_parameters` applies to a runtime
/// selection, keyed on engagement rather than on the selected rung's ordinal.
pub(crate) fn required_numeric_parameters(engaged: &[StrategyRung]) -> Vec<(&'static str, u32)> {
    let mut required = Vec::new();
    if engaged.contains(&StrategyRung::BoundedDecode) {
        required.push(("decodeTileEdge", 1));
        required.push(("decodeOverlap", 0));
    }
    if engaged.contains(&StrategyRung::BoundedAttention) {
        required.push(("attentionChunkSize", 1));
    }
    if engaged.contains(&StrategyRung::BoundedTransformerResidency) {
        required.push(("transformerWindowSize", 1));
    }
    required
}

/// Typed [`MemoryMode`] for a canonical worker mode-key label, `as_key(from(key)) == key` for every
/// input, so evidence written before sc-16583 typed the field compares identically. Shared by the
/// MLX fit gate and the candle memory-strategy lane, which key evidence cells by the same labels.
pub(crate) fn memory_mode_from_mode_key(key: &str) -> MemoryMode {
    match key {
        "text_to_image" => MemoryMode::TextToImage,
        "image_to_image" => MemoryMode::ImageToImage,
        "edit" => MemoryMode::Edit,
        other => MemoryMode::Other(other.to_owned()),
    }
}

/// Execute one provider request through the adopted safety/lifecycle seam. A created scope receives
/// exactly one explicit terminal outcome; its Drop remains only a panic/unwind backstop.
pub fn generate_with_scope(
    generator: &dyn gen_core::Generator,
    request: &mut gen_core::GenerationRequest,
    context: Option<&gen_core::MemoryRunContext>,
    on_progress: &mut dyn FnMut(gen_core::Progress),
) -> gen_core::Result<gen_core::GenerationOutput> {
    let Some(context) = context else {
        return generator.generate(request, on_progress);
    };
    if let gen_core::MemorySafetyDecision::Reject { reason } =
        generator.memory_strategy_safety_check(context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut scope = generator.begin_memory_strategy_request(context)?;
    if context.selection.strategy.is_optimized() && scope.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} accepted an optimized memory-strategy selection without opening a request scope",
            generator.descriptor().id
        )));
    }
    if let Some(scope) = scope.as_mut() {
        if let Err(error) = scope.configure_request(request) {
            let message = error.to_string();
            let _ = scope.finish(gen_core::MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(error);
        }
    }
    let result = generator.generate(request, on_progress);
    if let Some(scope) = scope.as_mut() {
        let outcome = match &result {
            Ok(_) => gen_core::MemoryRunOutcome::Complete,
            Err(gen_core::Error::Canceled) => gen_core::MemoryRunOutcome::Canceled,
            Err(error) => gen_core::MemoryRunOutcome::Error {
                message: error.to_string(),
            },
        };
        if let Err(finish_error) = scope.finish(outcome) {
            if result.is_ok() {
                return Err(finish_error);
            }
        }
    }
    result
}

/// Exact request identity that one evidence cell must cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestScope<'a> {
    pub resolved_route: &'a str,
    pub backend: &'a str,
    pub tier: MemoryNumericTier,
    pub mode: &'a str,
    pub overlay: Option<&'a str>,
    pub geometry: MemoryGeometry,
    /// The live compile-closure digest of the provider being admitted (sc-17774).
    ///
    /// Read from `sceneworks_core::memory_calibration::packaged_closure_digest`. It replaces
    /// `expected_inference_revision`, which every lane satisfied with its OWN frozen constant — four
    /// separate per-model mechanisms doing the same job differently, none of which could tell a
    /// change to its own model from a change to somebody else's.
    pub expected_closure_digest: &'a str,
}

/// What produced a candidate's peak number (sc-18096, epic 18093 R1).
///
/// A measured candidate is backed by a calibration record (its currency — current vs stale
/// closure — is decided separately, from the digest pair). An estimate candidate was synthesized
/// by the caller for an implemented-but-unmeasured rung; the two estimate variants exist because
/// the binding-phase constraint
/// ([`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]) governs
/// only candidates extrapolated from a measured cell, and because admission telemetry must name
/// which basis carried a selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateBasis {
    /// Backed by a measured calibration record.
    Measured,
    /// Synthesized estimate extrapolated from a measured cell's per-phase peaks (sc-18096). The
    /// caller must have already honored the binding-phase constraint before submitting it.
    EstimateFittedCurve,
    /// Synthesized estimate from the weights + headroom floor — no measured cell in its
    /// extrapolation basis, so the binding-phase constraint does not apply (see the scope
    /// sentence on the constraint's doc).
    EstimateFloor,
}

impl CandidateBasis {
    pub const fn is_estimate(self) -> bool {
        matches!(self, Self::EstimateFittedCurve | Self::EstimateFloor)
    }

    /// Stable label for tracing/telemetry.
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::EstimateFittedCurve => "fitted_curve",
            Self::EstimateFloor => "floor",
        }
    }
}

/// A provider estimate submitted to the selector. Cost is intentionally absent: strategy order is
/// worker-owned and follows [`MemoryStrategy::ALL`].
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub selection: MemorySelection,
    pub evidence: &'a MemoryEvidence,
    /// The provider closure digest this candidate's evidence was MEASURED under (sc-17774).
    ///
    /// It sits on `Candidate` rather than on `evidence` only because `MemoryEvidence` is a gen-core
    /// type owned by the inference repository, which SceneWorks pins by SHA; the field cannot move
    /// there without an inference change and a pin bump. The comparison itself is unaffected and is
    /// applied to every lane identically.
    pub closure_digest: &'a str,
    /// Whether the peak is a measurement or a synthesized estimate (sc-18096). Like
    /// `closure_digest`, this lives on `Candidate` because `MemoryEvidence` is pinned gen-core.
    pub basis: CandidateBasis,
}

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Evidence stores an integer byte ceiling. Admission converts that canonical value (after any
/// stale-margin widening, which happens in integer bytes) to GiB exactly once; callers cannot
/// submit a second floating-point estimate with a lower coefficient.
fn peak_bytes_to_gb(peak_bytes: u64) -> f64 {
    peak_bytes as f64 / BYTES_PER_GIB
}

/// How a non-excluded candidate passed [`candidate_exclusion`] (sc-18095/sc-18096): with evidence
/// measured under the request's live compile closure, with stale-closure measured evidence that
/// stays eligible behind the widened stale-measured margin, or as a synthesized estimate behind
/// the wider estimate margin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateCurrency {
    Current,
    StaleClosure,
    Estimate,
}

impl CandidateCurrency {
    /// Normative precedence (sc-18096): measured-current > stale-measured > estimate.
    const fn precedence(self) -> u8 {
        match self {
            Self::Current => 2,
            Self::StaleClosure => 1,
            Self::Estimate => 0,
        }
    }
}

/// The stale-measured margin for one backend (sc-18094 derivation, consumed here by sc-18095).
///
/// Matched on the canonical [`MemoryBackend`] family so a new backend cannot compile without
/// choosing a margin.
const fn stale_measured_margin(backend: MemoryBackend) -> f64 {
    match backend {
        MemoryBackend::Candle => CANDLE_STALE_MEASURED_MARGIN,
        MemoryBackend::Mlx => MLX_STALE_MEASURED_MARGIN,
    }
}

/// The estimate margin for one backend (sc-18094 derivation, consumed here by sc-18096). Same
/// exhaustive-match rationale as [`stale_measured_margin`].
const fn estimate_margin(backend: MemoryBackend) -> f64 {
    match backend {
        MemoryBackend::Candle => CANDLE_ESTIMATE_MARGIN,
        MemoryBackend::Mlx => MLX_ESTIMATE_MARGIN,
    }
}

/// Widen a stale-closure or estimate candidate's canonical byte ceiling by its policy margin
/// (sc-18095/sc-18096). The widening happens in integer bytes with a ceil, so the admitted ceiling
/// is never under the exact `peak * (1 + margin)` product and the GiB conversion stays a single
/// downstream step.
fn widened_peak_bytes(peak_bytes: u64, margin: f64) -> u64 {
    (peak_bytes as f64 * (1.0 + margin))
        .ceil()
        .clamp(0.0, u64::MAX as f64) as u64
}

/// The selector's own stale-measured widening (sc-18095), exported so the MLX gate's
/// evidence-path budget pre-check grades a stale candidate at the SAME admitted ceiling
/// `select_strategy` will (sc-18096). The pre-check runs against each candidate's captured
/// foreign reserve, which the selector's uniform budget cannot carry, so the two must share this
/// one policy function rather than each deriving a margin.
pub(crate) fn stale_widened_peak_bytes(backend: MemoryBackend, peak_bytes: u64) -> u64 {
    widened_peak_bytes(peak_bytes, stale_measured_margin(backend))
}

/// Worker-owned live budget. Headroom is removed once from the live/reclaimable pool and once from
/// the physical ceiling; exact equality fits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub available_gb: f64,
    pub reclaimable_gb: f64,
    pub total_gb: f64,
    pub reserved_headroom_gb: f64,
}

impl Budget {
    pub fn effective_gb(self) -> Option<f64> {
        if !self.available_gb.is_finite()
            || !self.reclaimable_gb.is_finite()
            || !self.total_gb.is_finite()
            || !self.reserved_headroom_gb.is_finite()
            || self.available_gb < 0.0
            || self.reclaimable_gb < 0.0
            || self.total_gb <= 0.0
            || self.reserved_headroom_gb < 0.0
        {
            return None;
        }
        Some(
            (self.available_gb + self.reclaimable_gb - self.reserved_headroom_gb)
                .max(0.0)
                .min((self.total_gb - self.reserved_headroom_gb).max(0.0)),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Selection {
    Selected {
        selection: MemorySelection,
        needed_gb: f64,
        available_gb: f64,
    },
    Reject {
        needed_gb: f64,
        available_gb: f64,
    },
    Unverified {
        reason: MemoryEvidenceVerdict,
    },
}

fn verdict_priority(verdict: MemoryEvidenceVerdict) -> u8 {
    // Generic absence is least actionable. Concrete integrity, calibration, revision, and envelope
    // failures win deterministically regardless of candidate declaration order.
    match verdict {
        MemoryEvidenceVerdict::Satisfied => 0,
        MemoryEvidenceVerdict::Unverified => 1,
        MemoryEvidenceVerdict::Missing => 2,
        MemoryEvidenceVerdict::OutOfEnvelope => 3,
        MemoryEvidenceVerdict::Stale => 4,
        MemoryEvidenceVerdict::FingerprintMismatch => 5,
        MemoryEvidenceVerdict::CompositionMismatch => 6,
        MemoryEvidenceVerdict::Invalid => 7,
    }
}

fn accumulate_reason(
    current: &mut Option<MemoryEvidenceVerdict>,
    candidate: MemoryEvidenceVerdict,
) {
    let replace = match *current {
        Some(reason) => verdict_priority(candidate) > verdict_priority(reason),
        None => true,
    };
    if replace {
        *current = Some(candidate);
    }
}

fn specific_unverified_reason(evidence: &MemoryEvidence) -> MemoryEvidenceVerdict {
    let mut reason = None;
    for dimension in MemoryEvidenceDimension::ALL {
        let verdict = evidence.dimensions.verdict(dimension);
        if verdict != MemoryEvidenceVerdict::Satisfied {
            accumulate_reason(&mut reason, verdict);
        }
    }
    reason.unwrap_or(MemoryEvidenceVerdict::Unverified)
}

fn candidate_exclusion(
    request: RequestScope<'_>,
    contract: &MemoryProviderContract,
    candidate: &Candidate<'_>,
) -> Result<CandidateCurrency, MemoryEvidenceVerdict> {
    if request.resolved_route != contract.provider_id
        || request.backend != contract.backend.backend_id()
        || candidate.evidence.key.resolved_route != request.resolved_route
        || candidate.evidence.key.resolved_route != contract.provider_id
        || candidate.evidence.key.backend.as_key() != contract.backend.backend_id()
        || candidate.selection.tier != request.tier
        || candidate.evidence.key.tier != request.tier
        || candidate.evidence.key.strategy != candidate.selection.strategy
        || candidate.evidence.key.parameters != candidate.selection.parameters
    {
        return Err(MemoryEvidenceVerdict::Invalid);
    }
    let key = &candidate.evidence.key;
    if key.backend.as_key() != request.backend
        || key.mode.as_key() != request.mode
        || key.overlay.as_deref() != request.overlay
        || key.geometry != request.geometry
    {
        return Err(MemoryEvidenceVerdict::OutOfEnvelope);
    }
    if contract.validate_selection(&candidate.selection).is_err() {
        return Err(MemoryEvidenceVerdict::Invalid);
    }
    if candidate.basis.is_estimate() {
        // sc-18096: worker-side eligibility wrap for estimate-scoped candidates. gen-core's
        // `optimized_eligibility` (pinned inference revision) hard-requires `Verified` conformance
        // and a `Passed` parity result for every optimized rung, which no synthesized estimate can
        // carry — and bumping the pin for this was explicitly ruled out. The predicate is still
        // consulted rather than re-implemented: its STRUCTURAL prefix (canonical engaged
        // composition ⇒ `Invalid`, contract composition agreement ⇒ `CompositionMismatch`) runs
        // before the conformance gate, so those verdicts exclude an estimate exactly as they
        // exclude a measured cell. The only accepted failure is the `Unverified` verdict the
        // conformance gate necessarily yields for a synthesized record; the calibration-identity
        // checks it short-circuits do not apply to a candidate with no calibration record behind
        // it.
        return match candidate.evidence.optimized_eligibility(contract) {
            Ok(()) | Err(MemoryEvidenceVerdict::Unverified) => Ok(CandidateCurrency::Estimate),
            Err(reason) => Err(reason),
        };
    }
    if let Some(reason) = candidate
        .evidence
        .optimized_eligibility(contract)
        .err()
        .map(|reason| {
            if reason == MemoryEvidenceVerdict::Unverified
                && candidate.evidence.conformance != gen_core::MemoryConformanceState::Verified
            {
                specific_unverified_reason(candidate.evidence)
            } else {
                reason
            }
        })
    {
        return Err(reason);
    }
    // sc-17774: the provider's own compiled closure, not an inference revision. `evidence
    // .inference_revision` stays as capture provenance and is deliberately not compared — comparing
    // it is what let a commit to one model stale every other model's measurements.
    //
    // sc-18095 (epic 18093): a mismatch no longer EXCLUDES the candidate. Currency is a signal, not
    // a gate: measured evidence whose closure moved stays eligible with its peak widened by the
    // backend's stale-measured margin (`crate::ladder_margin_policy`), applied by the selector.
    // Every structural exclusion above still runs first, so `Invalid`, `OutOfEnvelope`, and
    // `CompositionMismatch` exclude a stale candidate exactly as they exclude a current one — only
    // fully-verified measured cells reach the stale-admission path.
    if candidate.closure_digest != request.expected_closure_digest {
        return Ok(CandidateCurrency::StaleClosure);
    }
    Ok(CandidateCurrency::Current)
}

/// Select the first fitting candidate in the normative resident → staged → bounded-decode →
/// bounded-attention → bounded-transformer order.
pub fn select_strategy(
    request: RequestScope<'_>,
    contract: &MemoryProviderContract,
    budget: Option<Budget>,
    candidates: &[Candidate<'_>],
) -> Selection {
    if !contract.conformance_errors().is_empty()
        || contract.runtime.cancellation
            != MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
        || contract.runtime.error
            != MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
    {
        return Selection::Unverified {
            reason: MemoryEvidenceVerdict::Invalid,
        };
    }
    let Some(available_gb) = budget.and_then(Budget::effective_gb) else {
        return Selection::Unverified {
            reason: MemoryEvidenceVerdict::Missing,
        };
    };
    // Evidence is an integer-byte ceiling. Canonicalize the effective budget to the same unit before
    // comparing, so a decimal GiB equality (for example 28.6 - 2.0 == 26.6) cannot miss by one byte
    // after the evidence ceiling conversion and spuriously select a deeper rung.
    let available_bytes = (available_gb * BYTES_PER_GIB)
        .ceil()
        .clamp(0.0, u64::MAX as f64) as u64;
    let stale_margin = stale_measured_margin(contract.backend.backend_kind());
    let estimate_margin = estimate_margin(contract.backend.backend_kind());
    let mut deepest = None;
    let mut first_unknown = None;
    for strategy in MemoryStrategy::ALL {
        let support = contract
            .capability(strategy)
            .map(|capability| &capability.support);
        if matches!(
            support,
            Some(MemoryStrategySupport::StructurallyNotApplicable { .. })
        ) {
            continue;
        }
        if !matches!(support, Some(MemoryStrategySupport::Implemented)) {
            continue;
        }
        let rung_candidates = candidates
            .iter()
            .filter(|candidate| candidate.selection.strategy == strategy)
            .collect::<Vec<_>>();
        if rung_candidates.is_empty() {
            accumulate_reason(&mut first_unknown, MemoryEvidenceVerdict::Missing);
            continue;
        }
        let mut eligible: Vec<(&Candidate<'_>, CandidateCurrency, u64)> = Vec::new();
        let mut first_exclusion = None;
        for candidate in rung_candidates {
            match candidate_exclusion(request, contract, candidate) {
                Err(reason) => {
                    accumulate_reason(&mut first_exclusion, reason);
                    tracing::warn!(
                        route = request.resolved_route,
                        backend = request.backend,
                        ?strategy,
                        ?reason,
                        "memory-strategy candidate excluded"
                    );
                }
                Ok(currency) => {
                    let admitted_peak_bytes = match currency {
                        CandidateCurrency::Current => candidate.evidence.predicted_peak_bytes,
                        CandidateCurrency::StaleClosure => widened_peak_bytes(
                            candidate.evidence.predicted_peak_bytes,
                            stale_margin,
                        ),
                        CandidateCurrency::Estimate => widened_peak_bytes(
                            candidate.evidence.predicted_peak_bytes,
                            estimate_margin,
                        ),
                    };
                    if currency == CandidateCurrency::StaleClosure {
                        // sc-18095: record the admission and the widened peak so a later OOM under
                        // this selection is attributable to stale-closure evidence.
                        tracing::info!(
                            route = request.resolved_route,
                            backend = request.backend,
                            ?strategy,
                            raw_peak_bytes = candidate.evidence.predicted_peak_bytes,
                            widened_peak_bytes = admitted_peak_bytes,
                            stale_margin,
                            candidate_closure_digest = candidate.closure_digest,
                            expected_closure_digest = request.expected_closure_digest,
                            "stale-closure memory-strategy candidate admitted with widened margin"
                        );
                    }
                    if currency == CandidateCurrency::Estimate {
                        // sc-18096: record the estimate admission, which basis produced it, and
                        // both the raw and widened peaks, so a later OOM under this selection is
                        // attributable to a synthesized estimate rather than a measurement.
                        tracing::info!(
                            route = request.resolved_route,
                            backend = request.backend,
                            ?strategy,
                            basis = candidate.basis.as_key(),
                            raw_peak_bytes = candidate.evidence.predicted_peak_bytes,
                            widened_peak_bytes = admitted_peak_bytes,
                            estimate_margin,
                            "estimate-backed memory-strategy candidate admitted with widened margin"
                        );
                    }
                    eligible.push((candidate, currency, admitted_peak_bytes));
                }
            }
        }
        // sc-18095: current evidence is strictly preferred over stale for the same key — a stale
        // cell whose exact key was re-measured under the live closure never competes with the
        // re-measurement.
        //
        // sc-18096: any eligible MEASURED candidate at this rung — current or stale — supersedes
        // every estimate at the rung. An estimate exists to cover a rung nobody measured; where a
        // measurement exists it is authoritative in both directions, including "this rung's
        // measured peak does not fit", which a synthesized guess must not overrule on a
        // fatal-OOM lane.
        let eligible = {
            let superseded_by_current = |candidate: &Candidate<'_>| {
                eligible.iter().any(|(current, currency, _)| {
                    *currency == CandidateCurrency::Current
                        && current.evidence.key == candidate.evidence.key
                })
            };
            let rung_has_measured = eligible
                .iter()
                .any(|(_, currency, _)| *currency != CandidateCurrency::Estimate);
            eligible
                .iter()
                .filter(|(candidate, currency, _)| match currency {
                    CandidateCurrency::Current => true,
                    CandidateCurrency::StaleClosure => !superseded_by_current(candidate),
                    CandidateCurrency::Estimate => !rung_has_measured,
                })
                .copied()
                .collect::<Vec<_>>()
        };
        if eligible.is_empty() {
            accumulate_reason(
                &mut first_unknown,
                first_exclusion.unwrap_or(MemoryEvidenceVerdict::Missing),
            );
            continue;
        }
        let parameter_preference = |candidate: &Candidate<'_>| {
            let parameters = candidate.selection.parameters;
            (
                parameters.decode_tile_edge.unwrap_or(0),
                parameters.attention_chunk_size.unwrap_or(0),
                parameters.transformer_window_size.unwrap_or(0),
                parameters.decode_overlap.unwrap_or(0),
            )
        };
        if let Some((candidate, currency, admitted_peak_bytes)) = eligible
            .iter()
            .filter(|(_, _, admitted_peak_bytes)| *admitted_peak_bytes <= available_bytes)
            .max_by(
                |(left, left_currency, left_peak), (right, right_currency, right_peak)| {
                    left_peak
                        .cmp(right_peak)
                        // The normative precedence breaks peak ties between distinct keys too
                        // (current > stale > estimate); on an all-current set this arm is always
                        // `Equal`, so pre-sc-18095 selection is byte-for-byte unchanged.
                        .then_with(|| left_currency.precedence().cmp(&right_currency.precedence()))
                        .then_with(|| parameter_preference(left).cmp(&parameter_preference(right)))
                },
            )
        {
            let needed_gb = peak_bytes_to_gb(*admitted_peak_bytes);
            if *currency == CandidateCurrency::StaleClosure {
                tracing::warn!(
                    route = request.resolved_route,
                    backend = request.backend,
                    ?strategy,
                    raw_peak_bytes = candidate.evidence.predicted_peak_bytes,
                    widened_peak_bytes = *admitted_peak_bytes,
                    stale_margin,
                    "memory-strategy selection uses stale-closure evidence at the widened peak"
                );
            }
            if *currency == CandidateCurrency::Estimate {
                tracing::warn!(
                    route = request.resolved_route,
                    backend = request.backend,
                    ?strategy,
                    basis = candidate.basis.as_key(),
                    raw_peak_bytes = candidate.evidence.predicted_peak_bytes,
                    widened_peak_bytes = *admitted_peak_bytes,
                    estimate_margin,
                    "memory-strategy selection uses an estimate-backed candidate at the widened peak"
                );
            }
            return Selection::Selected {
                selection: candidate.selection,
                needed_gb,
                available_gb,
            };
        }
        let minimum = eligible
            .iter()
            .map(|(_, _, admitted_peak_bytes)| peak_bytes_to_gb(*admitted_peak_bytes))
            .min_by(f64::total_cmp)
            .expect("eligible rung is non-empty");
        deepest = Some(deepest.map_or(minimum, |current: f64| current.min(minimum)));
    }
    if let Some(reason) = first_unknown {
        Selection::Unverified { reason }
    } else {
        deepest.map_or(
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Missing,
            },
            |needed_gb| Selection::Reject {
                needed_gb,
                available_gb,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        LoadShape, MemoryBackendRealization, MemoryBudget, MemoryCacheState,
        MemoryCalibrationIdentity, MemoryConformanceState, MemoryEvidenceDimensions,
        MemoryEvidenceKey, MemoryFormulaKind, MemoryLifecycleCapabilities, MemoryMode,
        MemoryParameterRanges, MemoryParityContract, MemoryParityResult, MemoryPhase,
        MemoryPrerequisiteScope, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
        MemoryStrategyCapability, MemoryStrategyParameters, MemoryStrategyPrerequisite,
        MemoryWindowMaterialization, Precision, Quant,
    };
    use std::sync::{Arc, Mutex};

    const FP: &str = "provider-formula-v1";
    const SW: &str = "sc-15449-contract-v1";
    const INF: &str = "0c85bc9ff9fe161227efebf396a83db5e967d9ad";
    /// A closure digest that is deliberately NOT the request's, for candidates a test needs stale.
    const STALE_CLOSURE: &str = "stale-closure-digest";

    fn tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        }
    }

    fn contract() -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default(
            "test",
            MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
                // SC-16090 made this field required, deliberately: a default would be an opt-out that
                // reads as a declaration. This is a synthetic fixture (`provider_id: "test"`) with no
                // loader behind it, so the value is a choice about coverage rather than a claim — the
                // conforming one keeps the rest of this module reading as the steady state, and
                // `a_declared_host_conversion_remains_selectable` covers the compatibility arm.
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        );
        contract.strategies = MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: MemoryStrategySupport::Implemented,
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![1024],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency => MemoryParameterRanges {
                        transformer_window_sizes: vec![1],
                        ..Default::default()
                    },
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle = MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        // Rung 4's shared prerequisite is the independent load-time materialization shape, not
        // staged residency. This all-rungs fixture must therefore model a generator loaded through
        // the deferred path; provider-specific rung dependencies belong in
        // `additional_prerequisites`.
        contract.load_shape = LoadShape::DeferredMaterialization;
        contract.formula = MemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            FP,
            LoadShape::DeferredMaterialization,
        ));
        contract
    }

    fn params(strategy: MemoryStrategy) -> MemoryStrategyParameters {
        match strategy {
            MemoryStrategy::BoundedDecode => MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => MemoryStrategyParameters {
                attention_chunk_size: Some(1024),
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategyParameters {
                transformer_window_size: Some(1),
                ..Default::default()
            },
            _ => Default::default(),
        }
    }

    fn evidence(strategy: MemoryStrategy) -> MemoryEvidence {
        MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "test".into(),
                backend: gen_core::MemoryBackend::Candle,
                tier: tier(),
                load_shape: LoadShape::DeferredMaterialization,
                mode: MemoryMode::TextToImage,
                overlay: None,
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                strategy,
                engaged_composition: contract().engaged_composition(strategy),
                parameters: params(strategy),
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: FP.into(),
            sceneworks_revision: SW.into(),
            inference_revision: INF.into(),
            harness_version: "test-harness-v1".into(),
            predicted_peak_bytes: 1,
            observed_peak_bytes: Some(1),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        }
    }

    fn rekey_composition(evidence: &mut MemoryEvidence, contract: &MemoryProviderContract) {
        evidence.key.engaged_composition = contract.engaged_composition(evidence.key.strategy);
    }

    fn request() -> RequestScope<'static> {
        RequestScope {
            resolved_route: "test",
            backend: "candle",
            tier: tier(),
            mode: "text_to_image",
            overlay: None,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            expected_closure_digest: INF,
        }
    }

    /// `sceneworks-core`'s receipt layer deliberately has no gen-core dependency, so its
    /// calibration ABI is a separate constant that must move in lockstep with gen-core's on every
    /// inference pin bump. A skew silently downgrades every calibrated admission to the legacy
    /// path (receipts verified against one ABI, provider identities minted under the other).
    #[test]
    fn receipt_layer_calibration_abi_tracks_gen_core() {
        assert_eq!(
            sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
            gen_core::MEMORY_CALIBRATION_ABI,
            "bump sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI (and migrate the \
             receipt schema) together with the inference pin"
        );
    }

    #[test]
    fn sceneworks_revision_is_provenance_and_does_not_exclude_a_candidate() {
        let mut source_only_change = evidence(MemoryStrategy::Resident);
        source_only_change.sceneworks_revision = "provenance-only-source-revision".into();
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &source_only_change,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };

        assert!(matches!(
            select_strategy(
                request(),
                &contract(),
                Some(Budget {
                    available_gb: 1.0,
                    reclaimable_gb: 0.0,
                    total_gb: 1.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[candidate],
            ),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn decimal_gib_exact_equality_fits_after_byte_ceiling_canonicalization() {
        let mut exact = evidence(MemoryStrategy::Resident);
        exact.predicted_peak_bytes = (26.6 * BYTES_PER_GIB).ceil() as u64;
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &exact,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };

        assert!(matches!(
            select_strategy(
                request(),
                &contract(),
                Some(Budget {
                    available_gb: 28.6,
                    reclaimable_gb: 0.0,
                    total_gb: 96.0,
                    reserved_headroom_gb: 2.0,
                }),
                &[candidate],
            ),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn normative_order_ignores_caller_array_order_and_exact_boundary_fits() {
        let mut resident = evidence(MemoryStrategy::Resident);
        resident.predicted_peak_bytes = (24.0 * BYTES_PER_GIB) as u64;
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.predicted_peak_bytes = (16.0 * BYTES_PER_GIB) as u64;
        let candidates = [
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: params(MemoryStrategy::StagedResidency),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    parameters: params(MemoryStrategy::Resident),
                    tier: tier(),
                },
                evidence: &resident,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
        ];
        let mut provider = contract();
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            provider.capability(strategy);
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = MemoryStrategySupport::Missing;
        }
        assert!(matches!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 16.0,
                    reclaimable_gb: 0.0,
                    total_gb: 24.0,
                    reserved_headroom_gb: 0.0,
                }),
                &candidates,
            ),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                available_gb: 16.0,
                ..
            }
        ));
    }

    #[test]
    fn canonical_evidence_reason_is_preserved() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.predicted_peak_bytes = (8.0 * BYTES_PER_GIB) as u64;
        staged.observed_peak_bytes = None;
        let mut provider = contract();
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::Resident)
            .unwrap()
            .support = MemoryStrategySupport::Missing;
        rekey_composition(&mut staged, &provider);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 10.0,
                    reclaimable_gb: 0.0,
                    total_gb: 10.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[candidate],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Invalid,
            }
        );
    }

    #[test]
    fn generic_unverified_conformance_surfaces_the_specific_dimension_reason() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.conformance = MemoryConformanceState::ImplementedUnverified;
        staged.dimensions.static_implementation = MemoryEvidenceVerdict::Unverified;
        staged.dimensions.current_environment_verification = MemoryEvidenceVerdict::Stale;
        staged.parity_result = MemoryParityResult::NotRun;
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 10.0,
                    reclaimable_gb: 0.0,
                    total_gb: 10.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[candidate],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Stale,
            }
        );
    }

    #[test]
    fn later_fingerprint_failure_outranks_earlier_generic_and_missing_dimensions() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.conformance = MemoryConformanceState::ImplementedUnverified;
        staged.dimensions.static_implementation = MemoryEvidenceVerdict::Unverified;
        staged.dimensions.declared_calibration = MemoryEvidenceVerdict::Missing;
        staged.dimensions.current_environment_verification =
            MemoryEvidenceVerdict::FingerprintMismatch;
        staged.parity_result = MemoryParityResult::NotRun;
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 10.0,
                    reclaimable_gb: 0.0,
                    total_gb: 10.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[candidate],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::FingerprintMismatch,
            }
        );
    }

    #[test]
    fn non_default_engagement_edge_makes_formerly_green_evidence_ineligible() {
        let captured = evidence(MemoryStrategy::BoundedDecode);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedDecode,
                parameters: params(MemoryStrategy::BoundedDecode),
                tier: tier(),
            },
            evidence: &captured,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let mut changed_contract = contract();
        changed_contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));

        assert_eq!(
            select_strategy(
                request(),
                &changed_contract,
                Some(Budget {
                    available_gb: 10.0,
                    reclaimable_gb: 0.0,
                    total_gb: 10.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[candidate],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::CompositionMismatch,
            }
        );
    }

    #[test]
    fn exclusion_reason_aggregation_is_order_independent_and_prefers_specific_causes() {
        let aggregate = |reasons: &[MemoryEvidenceVerdict]| {
            let mut result = None;
            for reason in reasons {
                accumulate_reason(&mut result, *reason);
            }
            result
        };
        let forward = [
            MemoryEvidenceVerdict::Unverified,
            MemoryEvidenceVerdict::Missing,
            MemoryEvidenceVerdict::Stale,
        ];
        let reverse = [
            MemoryEvidenceVerdict::Stale,
            MemoryEvidenceVerdict::Missing,
            MemoryEvidenceVerdict::Unverified,
        ];
        assert_eq!(aggregate(&forward), Some(MemoryEvidenceVerdict::Stale));
        assert_eq!(aggregate(&reverse), aggregate(&forward));
    }

    #[test]
    fn excluded_cheaper_rung_does_not_block_a_verified_deeper_fit() {
        let mut excluded_staged = evidence(MemoryStrategy::StagedResidency);
        // sc-18095: a stale CLOSURE no longer excludes (it admits with a widened margin), so this
        // test's excluded cheaper rung is now structurally `Invalid` — its evidence key names the
        // wrong backend family. Leaving it written against staleness would have quietly stopped
        // exercising the excluded-rung path at all, exactly as the sc-17774 rewrite before it.
        excluded_staged.key.backend = gen_core::MemoryBackend::Mlx;
        let mut bounded_decode = evidence(MemoryStrategy::BoundedDecode);
        bounded_decode.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        let mut provider = contract();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = MemoryStrategySupport::Missing;
        }
        rekey_composition(&mut excluded_staged, &provider);
        rekey_composition(&mut bounded_decode, &provider);
        let candidates = [
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &excluded_staged,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    parameters: params(MemoryStrategy::BoundedDecode),
                    tier: tier(),
                },
                evidence: &bounded_decode,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
        ];
        assert!(matches!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 8.0,
                    reclaimable_gb: 0.0,
                    total_gb: 8.0,
                    reserved_headroom_gb: 0.0,
                }),
                &candidates,
            ),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    ..
                },
                needed_gb: 8.0,
                ..
            }
        ));
    }

    #[test]
    fn resident_baseline_remains_eligible_before_verified_optimized_rungs() {
        let mut resident = evidence(MemoryStrategy::Resident);
        resident.conformance = MemoryConformanceState::ImplementedUnverified;
        resident.predicted_peak_bytes = 20 * 1024 * 1024 * 1024;
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        let candidates = [
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &resident,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            },
        ];
        let mut provider = contract();
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = MemoryStrategySupport::Missing;
        }
        let budget = |available_gb| {
            Some(Budget {
                available_gb,
                reclaimable_gb: 0.0,
                total_gb: available_gb,
                reserved_headroom_gb: 0.0,
            })
        };
        assert!(matches!(
            select_strategy(request(), &provider, budget(20.0), &candidates),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            select_strategy(request(), &provider, budget(8.0), &candidates),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn canonical_character_identity_evidence_is_selectable_but_aliases_are_not() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.key.mode = MemoryMode::Other("character_image".to_owned());
        staged.key.overlay = Some("identity".to_owned());
        staged.key.geometry.reference_count = 1;
        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let mut scope = request();
        scope.mode = "character_image";
        scope.overlay = Some("identity");
        scope.geometry.reference_count = 1;
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        });

        assert!(matches!(
            select_strategy(scope, &provider, budget, &[candidate]),
            Selection::Selected {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    ..
                },
                ..
            }
        ));

        scope.overlay = Some("ip_adapter");
        assert_eq!(
            select_strategy(scope, &provider, budget, &[candidate]),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::OutOfEnvelope,
            }
        );
    }

    fn staged_only_provider() -> MemoryProviderContract {
        let mut provider = contract();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = MemoryStrategySupport::Missing;
        }
        provider
    }

    #[test]
    fn canonical_predicted_peak_bytes_are_the_only_admission_number() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024 + 1;
        let selection = MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: tier(),
        };
        let candidate = Candidate {
            selection,
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        });
        assert!(matches!(
            select_strategy(request(), &provider, budget, &[candidate]),
            Selection::Reject { .. }
        ));

        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        assert!(matches!(
            select_strategy(
                request(),
                &provider,
                budget,
                &[Candidate {
                    selection,
                    evidence: &staged,
                    closure_digest: INF,
                    basis: CandidateBasis::Measured,
                }],
            ),
            Selection::Selected {
                needed_gb: 8.0,
                available_gb: 8.0,
                ..
            }
        ));
    }

    #[test]
    fn route_and_backend_identity_are_bound_across_request_contract_and_evidence() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        });
        let mut wrong_route = request();
        wrong_route.resolved_route = "other";
        assert_eq!(
            select_strategy(wrong_route, &provider, budget, &[candidate],),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Invalid,
            }
        );

        let mut wrong_backend = staged.clone();
        wrong_backend.key.backend = gen_core::MemoryBackend::Mlx;
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                budget,
                &[Candidate {
                    selection: candidate.selection,
                    evidence: &wrong_backend,
                    closure_digest: INF,
                    basis: CandidateBasis::Measured,
                }],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Invalid,
            }
        );
    }

    #[test]
    fn within_rung_choice_is_fit_aware_and_declaration_order_invariant() {
        let mut high = evidence(MemoryStrategy::BoundedDecode);
        high.predicted_peak_bytes = 9 * 1024 * 1024 * 1024;
        high.key.parameters.decode_tile_edge = Some(768);
        let mut low = evidence(MemoryStrategy::BoundedDecode);
        low.predicted_peak_bytes = 7 * 1024 * 1024 * 1024;
        let mut provider = contract();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = MemoryStrategySupport::Missing;
        }
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters
            .decode_tile_edges = vec![512, 768];
        rekey_composition(&mut high, &provider);
        rekey_composition(&mut low, &provider);
        let high_candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedDecode,
                parameters: params(MemoryStrategy::BoundedDecode),
                tier: tier(),
            },
            evidence: &high,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let low_candidate = Candidate {
            evidence: &low,
            ..high_candidate
        };
        let mut high_selection = high_candidate.selection;
        high_selection.parameters.decode_tile_edge = Some(768);
        let high_candidate = Candidate {
            selection: high_selection,
            evidence: &high,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        });
        let forward = select_strategy(
            request(),
            &provider,
            budget,
            &[high_candidate, low_candidate],
        );
        let reverse = select_strategy(
            request(),
            &provider,
            budget,
            &[low_candidate, high_candidate],
        );
        assert_eq!(forward, reverse);
        assert!(matches!(
            forward,
            Selection::Selected { needed_gb: 7.0, .. }
        ));

        let mut tied_high = high.clone();
        tied_high.predicted_peak_bytes = 7 * 1024 * 1024 * 1024;
        let tied = select_strategy(
            request(),
            &provider,
            budget,
            &[
                low_candidate,
                Candidate {
                    selection: high_selection,
                    evidence: &tied_high,
                    closure_digest: INF,
                    basis: CandidateBasis::Measured,
                },
            ],
        );
        assert!(matches!(
            tied,
            Selection::Selected {
                selection: MemorySelection {
                    parameters: MemoryStrategyParameters {
                        decode_tile_edge: Some(768),
                        ..
                    },
                    ..
                },
                ..
            }
        ));

        let mut invalid = high.clone();
        invalid.key.backend = gen_core::MemoryBackend::Mlx;
        assert!(matches!(
            select_strategy(
                request(),
                &provider,
                budget,
                &[
                    Candidate {
                        evidence: &invalid,
                        ..high_candidate
                    },
                    low_candidate,
                ],
            ),
            Selection::Selected { needed_gb: 7.0, .. }
        ));
    }

    #[derive(Clone, Copy)]
    enum MockGenerate {
        Complete,
        Canceled,
        Error,
    }

    #[derive(Default)]
    struct LifecycleRecord {
        configure_calls: usize,
        finish_calls: Vec<MemoryRunOutcome>,
    }

    struct MockScope {
        record: Arc<Mutex<LifecycleRecord>>,
        configure_fails: bool,
    }

    impl MemoryRequestScope for MockScope {
        fn configure_request(
            &mut self,
            _request: &mut gen_core::GenerationRequest,
        ) -> gen_core::Result<()> {
            self.record.lock().unwrap().configure_calls += 1;
            if self.configure_fails {
                Err(gen_core::Error::Unsupported("configure failed".into()))
            } else {
                Ok(())
            }
        }

        fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn configure_decode(
            &mut self,
            _tile_edge: u32,
            _overlap: u32,
            _geometry: MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }

        fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
            Ok(())
        }

        fn materialize_transformer_window(
            &mut self,
            _first_block: u32,
            _block_count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }

        fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
            self.record.lock().unwrap().finish_calls.push(outcome);
            Ok(())
        }
    }

    struct MockGenerator {
        descriptor: gen_core::ModelDescriptor,
        contract: MemoryProviderContract,
        record: Arc<Mutex<LifecycleRecord>>,
        generate: MockGenerate,
        configure_fails: bool,
    }

    impl MockGenerator {
        fn new(generate: MockGenerate, configure_fails: bool) -> Self {
            Self {
                descriptor: gen_core::ModelDescriptor {
                    denoiser_output_latent_space: None,
                    id: "lifecycle_mock",
                    family: "test",
                    backend: "candle",
                    modality: gen_core::Modality::Image,
                    capabilities: Default::default(),
                    required_components: &[],
                    control_kinds: None,
                },
                contract: contract(),
                record: Default::default(),
                generate,
                configure_fails,
            }
        }
    }

    impl gen_core::Generator for MockGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
            Some(&self.contract)
        }

        fn begin_memory_strategy_request(
            &self,
            _context: &MemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + '_>>> {
            Ok(Some(Box::new(MockScope {
                record: Arc::clone(&self.record),
                configure_fails: self.configure_fails,
            })))
        }

        fn validate(&self, _request: &gen_core::GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _request: &gen_core::GenerationRequest,
            _on_progress: &mut dyn FnMut(gen_core::Progress),
        ) -> gen_core::Result<gen_core::GenerationOutput> {
            match self.generate {
                MockGenerate::Complete => {
                    Ok(gen_core::GenerationOutput::Images(vec![Default::default()]))
                }
                MockGenerate::Canceled => Err(gen_core::Error::Canceled),
                MockGenerate::Error => Err(gen_core::Error::Unsupported("render failed".into())),
            }
        }
    }

    fn run_context() -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            // sc-16590 hardened the handshake: with a calibration identity on the contract, the
            // context must match its abi, fingerprint, and load shape exactly or the provider
            // safety check rejects before generate.
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: FP.to_owned(),
            load_shape: LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 16,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 8,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".into(),
        }
    }

    fn assert_single_terminal(
        generate: MockGenerate,
        configure_fails: bool,
        expected: MemoryRunOutcome,
    ) {
        let generator = MockGenerator::new(generate, configure_fails);
        let mut request = gen_core::GenerationRequest::default();
        let mut progress = |_| {};
        let result = generate_with_scope(
            &generator,
            &mut request,
            Some(&run_context()),
            &mut progress,
        );
        if matches!(expected, MemoryRunOutcome::Complete) {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
        let record = generator.record.lock().unwrap();
        assert_eq!(record.configure_calls, 1);
        assert_eq!(record.finish_calls, vec![expected]);
    }

    #[test]
    fn lifecycle_finishes_exactly_once_on_success_cancel_error_and_configure_failure() {
        assert_single_terminal(MockGenerate::Complete, false, MemoryRunOutcome::Complete);
        assert_single_terminal(MockGenerate::Canceled, false, MemoryRunOutcome::Canceled);
        assert_single_terminal(
            MockGenerate::Error,
            false,
            MemoryRunOutcome::Error {
                message: "unsupported: render failed".into(),
            },
        );
        assert_single_terminal(
            MockGenerate::Complete,
            true,
            MemoryRunOutcome::Error {
                message: "unsupported: configure failed".into(),
            },
        );
    }

    /// Every parameter an engaged rung owns, for a selection at `strategy` on [`contract`].
    fn cumulative_params(strategy: MemoryStrategy) -> MemoryStrategyParameters {
        MemoryStrategyParameters {
            decode_tile_edge: strategy
                .engages(MemoryStrategy::BoundedDecode)
                .then_some(512),
            decode_overlap: strategy
                .engages(MemoryStrategy::BoundedDecode)
                .then_some(128),
            attention_chunk_size: strategy
                .engages(MemoryStrategy::BoundedAttention)
                .then_some(1024),
            transformer_window_size: strategy
                .engages(MemoryStrategy::BoundedTransformerResidency)
                .then_some(1),
            transformer_window_component: None,
        }
    }

    /// **SC-16090 — why rung 4's per-backend cost magnitude cannot change a LADDER selection.**
    ///
    /// SC-15791 measured rung 4 at ~26× more per denoise step on Candle than on MLX for the same rung,
    /// same model, same snapshot, and asked whether the ladder's cost order must therefore fork per
    /// backend. The answer is no, and this is the executable half of the reason: the selector consults
    /// the *order*, never a magnitude, and it stops at the cheapest **selectable** rung. So rung 4
    /// never wins over a cheaper rung that fits, and a multiplier on it cannot flip that comparison in
    /// either direction.
    ///
    /// Four arms over one candidate set whose peaks descend with the ladder:
    ///
    /// 1. A cheaper rung that fits **wins**, even though rung 4's peak is 5× smaller and would leave
    ///    far more headroom. A selector that preferred the smallest peak, or that weighed a declared
    ///    cost, would pick differently here.
    /// 2. Rung 4 is selected **only** once nothing cheaper fits.
    /// 3. One notch tighter and the outcome is `Reject` — no render at all.
    /// 4. And the honest caveat, because arm 3 is not the *only* way rung 4 gets reached: a cheaper
    ///    rung whose peak WOULD have fit but whose evidence is excluded does not stop the walk, so
    ///    rung 4 can be selected over a cheaper rung that physically fits (the pre-existing
    ///    `excluded_cheaper_rung_does_not_block_a_verified_deeper_fit` pins that path directly). It
    ///    does not rescue the argument's premise and it does not damage it either: the alternative is
    ///    then `Unverified`, which is still no render. What never happens is rung 4 beating a cheaper
    ///    rung that is *selectable*.
    ///
    /// The fix therefore belongs in the Candle realization
    /// (`gen_core::memory_strategy::MemoryWindowMaterialization`), where Krea now declares
    /// `DeviceFormatTransfer`, not in a forked cost order. **Scope warning:** this is a statement about
    /// the ladder only. One layer up, the Krea capability-downtier collapses "fits resident" and "fits
    /// only via rung 4" into the same `TierFit::Fits`, and there the magnitude genuinely does decide —
    /// sc-16104.
    #[test]
    fn the_cheapest_selectable_rung_wins_and_rung_four_never_beats_one_that_fits() {
        // Peaks in GiB, descending with the ladder: 40 / 30 / 20 / 10 / 2.
        let peaks = [40.0, 30.0, 20.0, 10.0, 2.0];
        let evidences = MemoryStrategy::ALL
            .into_iter()
            .zip(peaks)
            .map(|(strategy, gb)| {
                let mut record = evidence(strategy);
                record.key.parameters = cumulative_params(strategy);
                record.predicted_peak_bytes = (gb * BYTES_PER_GIB) as u64;
                record.observed_peak_bytes = Some(record.predicted_peak_bytes);
                record
            })
            .collect::<Vec<_>>();
        let candidates = MemoryStrategy::ALL
            .into_iter()
            .zip(&evidences)
            .map(|(strategy, record)| Candidate {
                selection: MemorySelection {
                    strategy,
                    parameters: cumulative_params(strategy),
                    tier: tier(),
                },
                evidence: record,
                closure_digest: INF,
                basis: CandidateBasis::Measured,
            })
            .collect::<Vec<_>>();
        let provider = contract();
        let select = |available_gb: f64| {
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb,
                    reclaimable_gb: 0.0,
                    total_gb: 48.0,
                    reserved_headroom_gb: 0.0,
                }),
                &candidates,
            )
        };

        // 1. Rungs 3 and 4 both fit at 12 GiB. The ORDER decides, not the peak: rung 3 wins even
        //    though rung 4 would leave 10 GiB more headroom.
        assert!(
            matches!(
                select(12.0),
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedAttention,
                        ..
                    },
                    ..
                }
            ),
            "the cheapest fitting rung must win over a deeper one with a smaller peak: {:?}",
            select(12.0)
        );

        // 2. Nothing cheaper fits at 3 GiB — now, and only now, rung 4 is selected.
        assert!(
            matches!(
                select(3.0),
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedTransformerResidency,
                        ..
                    },
                    ..
                }
            ),
            "rung 4 must be reachable once nothing cheaper fits: {:?}",
            select(3.0)
        );

        // 3. One notch tighter and there is no render. So rung 4's per-step cost is weighed against
        //    "cannot render" — never against a cheaper rung that was available.
        assert!(
            matches!(select(1.0), Selection::Reject { .. }),
            "below rung 4's peak the alternative is rejection, not a cheaper rung: {:?}",
            select(1.0)
        );

        // 4. The caveat, asserted rather than asserted-away: excluding a cheaper rung's evidence does
        //    NOT stop the walk, so rung 4 is reachable while rung 3's peak still fits. The alternative
        //    is `Unverified` — still no render — which is why the conclusion survives. The trigger is
        //    structural (`Invalid` here, via an evidence key naming the wrong backend family): as of
        //    sc-18095 a stale closure is no longer an exclusion at all — it admits the candidate at a
        //    widened peak — so only the structural verdicts still open this path.
        let mut invalid_attention = evidences[3].clone();
        invalid_attention.key.backend = gen_core::MemoryBackend::Mlx;
        let mut with_invalid = candidates.clone();
        with_invalid[3].evidence = &invalid_attention;
        let selection = select_strategy(
            request(),
            &provider,
            Some(Budget {
                available_gb: 12.0,
                reclaimable_gb: 0.0,
                total_gb: 48.0,
                reserved_headroom_gb: 0.0,
            }),
            &with_invalid,
        );
        assert!(
            matches!(
                selection,
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedTransformerResidency,
                        ..
                    },
                    ..
                }
            ),
            "an excluded cheaper rung does not stop the walk, so rung 4 IS reachable at 12 GiB where \
             rung 3's 10 GiB peak fits — the alternative there is Unverified, not a cheaper render: \
             {selection:?}"
        );
    }

    /// **SC-16090 — declaring a non-conforming window realization must not veto it.**
    ///
    /// The contract still lets a provider ship a rung-4 realization that converts formats per window,
    /// so long as it says what it converts and names the story removing it. Krea no longer needs that
    /// compatibility arm: its content-addressed sidecars make the steady-state realization a
    /// `DeviceFormatTransfer`. The generic arm remains a deliberate contract choice because within
    /// this ladder rung 4 is reached only when nothing cheaper fits, so refusing it would trade a slow
    /// render for no render.
    ///
    /// gen-core pins that such a contract is conformance-CLEAN. Nothing pinned that it is still
    /// SELECTED, and that half is this repo's: `select_strategy` bails to
    /// `Unverified { Invalid }` on any non-empty `conformance_errors`, so a later "tightening" of that
    /// rule into a veto would not fail a gen-core test — it would silently drop Krea's entire ladder to
    /// resident-only gating, on every candle job. This is the assertion that turns that red.
    ///
    /// The undeclared arm is asserted alongside it, so the test cannot pass by the rule being absent.
    #[test]
    fn a_declared_host_conversion_remains_selectable() {
        let converting = |owner_story: &str| MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::HostFormatConversion {
                converts: "MLX affine packed triple -> GGML Q4_1 per projection, per window"
                    .to_owned(),
                owner_story: owner_story.to_owned(),
            },
        };
        let mut record = evidence(MemoryStrategy::BoundedTransformerResidency);
        record.key.parameters = cumulative_params(MemoryStrategy::BoundedTransformerResidency);
        record.predicted_peak_bytes = (2.0 * BYTES_PER_GIB) as u64;
        record.observed_peak_bytes = Some(record.predicted_peak_bytes);
        let candidates = [Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: cumulative_params(MemoryStrategy::BoundedTransformerResidency),
                tier: tier(),
            },
            evidence: &record,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        }];
        let select = |provider: &MemoryProviderContract| {
            select_strategy(
                request(),
                provider,
                Some(Budget {
                    available_gb: 8.0,
                    reclaimable_gb: 0.0,
                    total_gb: 8.0,
                    reserved_headroom_gb: 0.0,
                }),
                &candidates,
            )
        };

        let mut declared = contract();
        declared.backend = converting("sc-fixture-remediation");
        assert!(
            matches!(
                select(&declared),
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedTransformerResidency,
                        ..
                    },
                    ..
                }
            ),
            "a declared per-window conversion must stay SELECTABLE — the rule refuses an undeclared \
             realization, never a slow one: {:?}",
            select(&declared)
        );

        // The other arm: undeclared is a contract-level error, and the severity is wider than rung 4
        // by design — the whole contract becomes uninterpretable, so the selector refuses everything.
        let mut undeclared = contract();
        undeclared.backend = converting("");
        assert!(
            matches!(
                select(&undeclared),
                Selection::Unverified {
                    reason: MemoryEvidenceVerdict::Invalid
                }
            ),
            "an UNDECLARED conversion must fail closed rather than select: {:?}",
            select(&undeclared)
        );
    }

    /// SC-15805 — the selector/validator contradiction, pinned closed on the contract shape that
    /// exposes it: rung 1 declared `Missing`.
    ///
    /// `select_strategy` runs every candidate through `contract.validate_selection`
    /// (see `candidate_exclusion`), so the two layers cannot return *different* selections — the
    /// disagreement surfaces instead as OVER-REFUSAL, the selector silently excluding rungs the
    /// provider implements and the request falling back to a more expensive rung or being rejected.
    /// This asserts the agreement AND the verdict each rung should get, so a graph that drops rung
    /// 4's edge or invents one on rung 2 turns it red.
    #[test]
    fn selector_and_validator_agree_rung_by_rung_when_rung_one_is_missing() {
        let mut provider = contract();
        // Candle Krea retains this realization-specific dependency even though rung 1 is no longer
        // the shared rung-4 prerequisite. Keep the selector/validator regression pinned to that
        // additive graph edge instead of recreating the obsolete global rule.
        provider.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("rung 1 capability")
            .support = MemoryStrategySupport::Missing;
        assert!(
            provider.conformance_errors().is_empty(),
            "a Missing rung 1 is a legal declaration: {:?}",
            provider.conformance_errors()
        );

        for probe in MemoryStrategy::ALL {
            // Only the probed rung fits the 8 GiB budget, so the selector's answer is exactly
            // "is this rung selectable on this contract".
            let evidences = MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| {
                    let mut record = evidence(strategy);
                    record.key.parameters = cumulative_params(strategy);
                    record.predicted_peak_bytes = if strategy == probe {
                        1024 * 1024 * 1024
                    } else {
                        100 * 1024 * 1024 * 1024
                    };
                    record
                })
                .collect::<Vec<_>>();
            let candidates = MemoryStrategy::ALL
                .into_iter()
                .zip(&evidences)
                .map(|(strategy, record)| Candidate {
                    selection: MemorySelection {
                        strategy,
                        parameters: cumulative_params(strategy),
                        tier: tier(),
                    },
                    evidence: record,
                    closure_digest: INF,
                    basis: CandidateBasis::Measured,
                })
                .collect::<Vec<_>>();

            let probed = MemorySelection {
                strategy: probe,
                parameters: cumulative_params(probe),
                tier: tier(),
            };
            let validator_accepts = provider.validate_selection(&probed).is_ok();
            let selector_accepts = matches!(
                select_strategy(
                    request(),
                    &provider,
                    Some(Budget {
                        available_gb: 8.0,
                        reclaimable_gb: 0.0,
                        total_gb: 8.0,
                        reserved_headroom_gb: 0.0,
                    }),
                    &candidates,
                ),
                Selection::Selected { selection, .. } if selection.strategy == probe
            );

            let expected = match probe {
                // Rungs 2 and 3 bound scratch and depend on nothing — the case that was impossible
                // under the numeric-order walk.
                MemoryStrategy::Resident
                | MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention => true,
                // Rung 1 is Missing; this provider declares an additive rung-4 engagement
                // prerequisite on it.
                MemoryStrategy::StagedResidency | MemoryStrategy::BoundedTransformerResidency => {
                    false
                }
            };
            assert_eq!(
                validator_accepts, expected,
                "validate_selection verdict for {probe:?} under a Missing rung 1"
            );
            assert_eq!(
                selector_accepts, expected,
                "select_strategy verdict for {probe:?} under a Missing rung 1"
            );
        }
    }

    // ── sc-18095: currency as a signal — stale-closure evidence admits behind a widened margin ──

    /// Regression pin: an all-current candidate set selects byte-for-byte as before sc-18095. The
    /// exact-boundary fit is the sharp edge — if the selector widened CURRENT evidence by any
    /// margin at all, an 8 GiB peak would stop fitting an 8 GiB budget, and `needed_gb` would stop
    /// being the raw evidence ceiling.
    #[test]
    fn exact_current_selection_is_byte_for_byte_unchanged_and_never_widened() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        let selection = MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: tier(),
        };
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 8.0,
                    reclaimable_gb: 0.0,
                    total_gb: 8.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[Candidate {
                    selection,
                    evidence: &staged,
                    closure_digest: INF,
                    basis: CandidateBasis::Measured,
                }],
            ),
            Selection::Selected {
                selection,
                needed_gb: 8.0,
                available_gb: 8.0,
            }
        );
    }

    /// sc-18095: a stale-only cell stays eligible, graded at exactly the policy-widened peak. The
    /// expected value is recomputed here from the POLICY constant rather than the selector's own
    /// helper, so a selector that stops widening (or widens by the wrong number) turns this red.
    #[test]
    fn stale_closure_evidence_is_admitted_at_exactly_the_policy_widened_peak() {
        let raw_peak_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = select_strategy(
            request(),
            &provider,
            Some(Budget {
                available_gb: 11.0,
                reclaimable_gb: 0.0,
                total_gb: 11.0,
                reserved_headroom_gb: 0.0,
            }),
            &[Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: STALE_CLOSURE,
                basis: CandidateBasis::Measured,
            }],
        );
        let Selection::Selected { needed_gb, .. } = result else {
            panic!("a stale-only cell must stay eligible under sc-18095: {result:?}");
        };
        assert!(
            needed_gb > peak_bytes_to_gb(raw_peak_bytes),
            "the admitted peak must be WIDENED past the raw measurement, got {needed_gb}"
        );
        let expected_widened_bytes =
            (raw_peak_bytes as f64 * (1.0 + CANDLE_STALE_MEASURED_MARGIN)).ceil();
        assert_eq!(needed_gb, expected_widened_bytes / BYTES_PER_GIB);
    }

    /// sc-18095 acceptance: a fully stale ladder still reaches a deep rung instead of collapsing to
    /// the legacy rung-1/2 fallback — the demotion this story retires. Peaks descend with the
    /// ladder and only rung 4's widened peak fits the budget.
    #[test]
    fn a_fully_stale_ladder_can_still_select_a_deep_rung() {
        let peaks = [40.0, 30.0, 20.0, 10.0, 2.0];
        let evidences = MemoryStrategy::ALL
            .into_iter()
            .zip(peaks)
            .map(|(strategy, gb)| {
                let mut record = evidence(strategy);
                record.key.parameters = cumulative_params(strategy);
                record.predicted_peak_bytes = (gb * BYTES_PER_GIB) as u64;
                record.observed_peak_bytes = Some(record.predicted_peak_bytes);
                record
            })
            .collect::<Vec<_>>();
        let candidates = MemoryStrategy::ALL
            .into_iter()
            .zip(&evidences)
            .map(|(strategy, record)| Candidate {
                selection: MemorySelection {
                    strategy,
                    parameters: cumulative_params(strategy),
                    tier: tier(),
                },
                evidence: record,
                closure_digest: STALE_CLOSURE,
                basis: CandidateBasis::Measured,
            })
            .collect::<Vec<_>>();
        let selection = select_strategy(
            request(),
            &contract(),
            Some(Budget {
                available_gb: 3.0,
                reclaimable_gb: 0.0,
                total_gb: 48.0,
                reserved_headroom_gb: 0.0,
            }),
            &candidates,
        );
        assert!(
            matches!(
                selection,
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedTransformerResidency,
                        ..
                    },
                    ..
                }
            ),
            "a stale ladder must stay walkable down to rung 4: {selection:?}"
        );
    }

    /// Mutation check demanded by the story: prove the widening is APPLIED, not just plumbed. This
    /// stale cell's raw peak fits the budget exactly — a selector whose stale margin is zeroed (or
    /// that forgets to widen) selects it, flipping this test red. The real selector grades it at
    /// the widened peak, which overflows the budget into `Reject`.
    #[test]
    fn zeroing_the_stale_margin_would_flip_this_rejection_into_a_selection() {
        let raw_peak_bytes: u64 = 8 * 1024 * 1024 * 1024;
        let budget = Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        };
        // The zero-margin outcome would FIT: the raw peak is exactly the effective budget.
        assert!(raw_peak_bytes <= (8.0 * BYTES_PER_GIB) as u64);
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = select_strategy(
            request(),
            &provider,
            Some(budget),
            &[Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: STALE_CLOSURE,
                basis: CandidateBasis::Measured,
            }],
        );
        let Selection::Reject { needed_gb, .. } = result else {
            panic!(
                "a stale cell must be graded at its WIDENED peak; selecting at the raw peak means \
                 the stale margin was not applied: {result:?}"
            );
        };
        assert!(
            needed_gb > 8.0,
            "the reported requirement must carry the widening: {needed_gb}"
        );
    }

    /// sc-18095: current evidence is strictly preferred over stale FOR THE SAME KEY, even when the
    /// stale cell's widened peak would otherwise win the within-rung largest-fitting-peak choice —
    /// and the preference is declaration-order invariant.
    #[test]
    fn current_evidence_beats_stale_for_the_same_key_in_either_declaration_order() {
        let provider = staged_only_provider();
        let mut current = evidence(MemoryStrategy::StagedResidency);
        rekey_composition(&mut current, &provider);
        current.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        // Same evidence KEY, different measurement: the stale capture saw a larger peak, and its
        // widened value is larger still — the naive max-by-peak winner.
        let mut stale = current.clone();
        stale.predicted_peak_bytes = 9 * 1024 * 1024 * 1024;
        let selection = MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: tier(),
        };
        let current_candidate = Candidate {
            selection,
            evidence: &current,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let stale_candidate = Candidate {
            selection,
            evidence: &stale,
            closure_digest: STALE_CLOSURE,
            basis: CandidateBasis::Measured,
        };
        let budget = Some(Budget {
            available_gb: 10.0,
            reclaimable_gb: 0.0,
            total_gb: 10.0,
            reserved_headroom_gb: 0.0,
        });
        let forward = select_strategy(
            request(),
            &provider,
            budget,
            &[current_candidate, stale_candidate],
        );
        let reverse = select_strategy(
            request(),
            &provider,
            budget,
            &[stale_candidate, current_candidate],
        );
        assert_eq!(forward, reverse);
        assert!(
            matches!(forward, Selection::Selected { needed_gb: 8.0, .. }),
            "the current re-measurement must win over the stale cell with the same key: {forward:?}"
        );
    }

    /// sc-18095: staleness is not a bypass — a stale candidate that also fails a structural check
    /// is excluded with the structural verdict, exactly like a current one. `CompositionMismatch`
    /// and `Invalid` both still exclude.
    #[test]
    fn structural_exclusions_still_exclude_a_stale_candidate() {
        // CompositionMismatch: the contract grew an engagement edge the captured composition lacks.
        let captured = evidence(MemoryStrategy::BoundedDecode);
        let mut changed_contract = contract();
        changed_contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        let budget = Some(Budget {
            available_gb: 10.0,
            reclaimable_gb: 0.0,
            total_gb: 10.0,
            reserved_headroom_gb: 0.0,
        });
        assert_eq!(
            select_strategy(
                request(),
                &changed_contract,
                budget,
                &[Candidate {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedDecode,
                        parameters: params(MemoryStrategy::BoundedDecode),
                        tier: tier(),
                    },
                    evidence: &captured,
                    closure_digest: STALE_CLOSURE,
                    basis: CandidateBasis::Measured,
                }],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::CompositionMismatch,
            }
        );

        // Invalid: the evidence key names the wrong backend family.
        let provider = staged_only_provider();
        let mut wrong_backend = evidence(MemoryStrategy::StagedResidency);
        rekey_composition(&mut wrong_backend, &provider);
        wrong_backend.key.backend = gen_core::MemoryBackend::Mlx;
        assert_eq!(
            select_strategy(
                request(),
                &provider,
                budget,
                &[Candidate {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::StagedResidency,
                        parameters: Default::default(),
                        tier: tier(),
                    },
                    evidence: &wrong_backend,
                    closure_digest: STALE_CLOSURE,
                    basis: CandidateBasis::Measured,
                }],
            ),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Invalid,
            }
        );
    }

    /// sc-18095: the stale margin is per-backend. The budget sits between the candle-widened and
    /// mlx-widened peaks of the same 10 GiB cell, so grading an MLX lane with the (narrower) candle
    /// margin — or with no margin — flips the first arm.
    #[test]
    fn mlx_stale_admission_uses_the_mlx_margin() {
        let mut provider = staged_only_provider();
        provider.backend = MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        };
        let raw_peak_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.key.backend = gen_core::MemoryBackend::Mlx;
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let mut scope = request();
        scope.backend = "mlx";
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
            closure_digest: STALE_CLOSURE,
            basis: CandidateBasis::Measured,
        };
        let budget = |available_gb| {
            Some(Budget {
                available_gb,
                reclaimable_gb: 0.0,
                total_gb: 48.0,
                reserved_headroom_gb: 0.0,
            })
        };
        // candle widening: 10.2 GiB; mlx widening: 10.5 GiB. 10.3 GiB must NOT fit on mlx.
        assert!(
            matches!(
                select_strategy(scope, &provider, budget(10.3), &[candidate]),
                Selection::Reject { .. }
            ),
            "10.3 GiB fits a candle-widened 10 GiB cell but must not fit the MLX-widened one"
        );
        let Selection::Selected { needed_gb, .. } =
            select_strategy(scope, &provider, budget(11.0), &[candidate])
        else {
            panic!("the MLX-widened peak fits an 11 GiB budget");
        };
        let expected_widened_bytes =
            (raw_peak_bytes as f64 * (1.0 + MLX_STALE_MEASURED_MARGIN)).ceil();
        assert_eq!(needed_gb, expected_widened_bytes / BYTES_PER_GIB);
    }

    /// sc-18095: the selector's tracing records the stale admission and the widened peak it used,
    /// so a later OOM under this selection is attributable to stale-closure evidence.
    #[test]
    fn tracing_records_stale_admission_and_the_widened_peak() {
        use std::io::Write;

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .finish();

        let raw_peak_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        let provider = staged_only_provider();
        rekey_composition(&mut staged, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = tracing::subscriber::with_default(subscriber, || {
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 11.0,
                    reclaimable_gb: 0.0,
                    total_gb: 11.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[Candidate {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::StagedResidency,
                        parameters: Default::default(),
                        tier: tier(),
                    },
                    evidence: &staged,
                    closure_digest: STALE_CLOSURE,
                    basis: CandidateBasis::Measured,
                }],
            )
        });
        assert!(matches!(result, Selection::Selected { .. }), "{result:?}");
        let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
        let widened_peak_bytes =
            (raw_peak_bytes as f64 * (1.0 + CANDLE_STALE_MEASURED_MARGIN)).ceil() as u64;
        assert!(
            output.contains("stale-closure memory-strategy candidate admitted with widened margin"),
            "admission event missing from trace output: {output}"
        );
        assert!(
            output.contains(&format!("widened_peak_bytes={widened_peak_bytes}")),
            "widened peak missing from trace output: {output}"
        );
        assert!(
            output.contains("memory-strategy selection uses stale-closure evidence"),
            "stale-selection event missing from trace output: {output}"
        );
    }

    // ── sc-18096: estimate-backed candidates admit the full ladder behind the estimate margin ──

    /// An estimate-basis candidate for one strategy: synthesized evidence, `ImplementedUnverified`
    /// conformance, no observed peak, parity not run — the shape the MLX gate fabricates.
    fn estimate_evidence(
        strategy: MemoryStrategy,
        contract: &MemoryProviderContract,
    ) -> MemoryEvidence {
        let mut record = evidence(strategy);
        record.conformance = MemoryConformanceState::ImplementedUnverified;
        record.observed_peak_bytes = None;
        record.parity_result = MemoryParityResult::NotRun;
        rekey_composition(&mut record, contract);
        record
    }

    /// sc-18096: an estimate-only cell is admitted at exactly the policy-widened peak. The
    /// expected value is recomputed from the POLICY constant, so a selector that stops widening
    /// estimates (or grades them with the narrower stale margin) turns this red.
    #[test]
    fn estimate_candidates_are_admitted_at_exactly_the_policy_widened_peak() {
        let raw_peak_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let provider = staged_only_provider();
        let mut staged = estimate_evidence(MemoryStrategy::StagedResidency, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = select_strategy(
            request(),
            &provider,
            Some(Budget {
                available_gb: 11.0,
                reclaimable_gb: 0.0,
                total_gb: 11.0,
                reserved_headroom_gb: 0.0,
            }),
            &[Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: INF,
                basis: CandidateBasis::EstimateFloor,
            }],
        );
        let Selection::Selected { needed_gb, .. } = result else {
            panic!("an implemented rung's estimate candidate must be selectable: {result:?}");
        };
        let expected_widened_bytes =
            (raw_peak_bytes as f64 * (1.0 + CANDLE_ESTIMATE_MARGIN)).ceil();
        assert_eq!(needed_gb, expected_widened_bytes / BYTES_PER_GIB);
        // The candle estimate margin is strictly wider than the candle stale margin — pinned at
        // COMPILE TIME by `ladder_margin_policy`'s invariant block — so grading an estimate with
        // the stale margin would have produced a smaller needed_gb: this equality pins the
        // ESTIMATE margin specifically.
    }

    /// Mutation check demanded by the story: prove the estimate widening is APPLIED. This
    /// estimate's raw peak fits the budget exactly — a selector whose estimate margin is zeroed
    /// selects it, flipping this red. The real selector grades it at the widened peak, which
    /// overflows into `Reject`.
    #[test]
    fn zeroing_the_estimate_margin_would_flip_this_rejection_into_a_selection() {
        let raw_peak_bytes: u64 = 8 * 1024 * 1024 * 1024;
        let provider = staged_only_provider();
        let mut staged = estimate_evidence(MemoryStrategy::StagedResidency, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = select_strategy(
            request(),
            &provider,
            Some(Budget {
                available_gb: 8.0,
                reclaimable_gb: 0.0,
                total_gb: 8.0,
                reserved_headroom_gb: 0.0,
            }),
            &[Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
                closure_digest: INF,
                basis: CandidateBasis::EstimateFloor,
            }],
        );
        let Selection::Reject { needed_gb, .. } = result else {
            panic!(
                "an estimate must be graded at its WIDENED peak; selecting at the raw peak means \
                 the estimate margin was not applied: {result:?}"
            );
        };
        assert!(
            needed_gb > 8.0,
            "the reported requirement must carry the widening: {needed_gb}"
        );
    }

    /// sc-18096 precedence: ANY eligible measured candidate at a rung — current or stale —
    /// supersedes every estimate at that rung, in either declaration order, and a current
    /// measured selection stays byte-for-byte unwidened while the estimate coexists.
    #[test]
    fn measured_candidates_supersede_estimates_at_the_same_rung() {
        let provider = staged_only_provider();
        let selection = MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: tier(),
        };
        let mut measured = evidence(MemoryStrategy::StagedResidency);
        rekey_composition(&mut measured, &provider);
        measured.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        // The estimate claims a SMALLER peak than the measurement — the dangerous direction on a
        // fatal-OOM lane. It must not win merely by fitting more comfortably.
        let mut estimate = estimate_evidence(MemoryStrategy::StagedResidency, &provider);
        estimate.predicted_peak_bytes = 4 * 1024 * 1024 * 1024;
        let measured_candidate = Candidate {
            selection,
            evidence: &measured,
            closure_digest: INF,
            basis: CandidateBasis::Measured,
        };
        let estimate_candidate = Candidate {
            selection,
            evidence: &estimate,
            closure_digest: INF,
            basis: CandidateBasis::EstimateFloor,
        };
        let budget = Some(Budget {
            available_gb: 10.0,
            reclaimable_gb: 0.0,
            total_gb: 10.0,
            reserved_headroom_gb: 0.0,
        });
        let forward = select_strategy(
            request(),
            &provider,
            budget,
            &[measured_candidate, estimate_candidate],
        );
        let reverse = select_strategy(
            request(),
            &provider,
            budget,
            &[estimate_candidate, measured_candidate],
        );
        assert_eq!(forward, reverse);
        assert!(
            matches!(forward, Selection::Selected { needed_gb: 8.0, .. }),
            "the CURRENT measurement must win at its raw, unwidened peak: {forward:?}"
        );

        // A STALE measurement also supersedes the estimate: the widened measured 8.4 GiB peak is
        // the admitted requirement, not the estimate's 4.4.
        let stale_candidate = Candidate {
            selection,
            evidence: &measured,
            closure_digest: STALE_CLOSURE,
            basis: CandidateBasis::Measured,
        };
        let result = select_strategy(
            request(),
            &provider,
            budget,
            &[estimate_candidate, stale_candidate],
        );
        let Selection::Selected { needed_gb, .. } = result else {
            panic!("the stale measurement must stay selectable: {result:?}");
        };
        let expected_widened_bytes =
            (8.0 * BYTES_PER_GIB * (1.0 + CANDLE_STALE_MEASURED_MARGIN)).ceil();
        assert_eq!(
            needed_gb,
            expected_widened_bytes / BYTES_PER_GIB,
            "the stale MEASUREMENT outranks the smaller estimate at the same rung"
        );

        // And where the measurement — even widened — does not fit, the estimate must NOT rescue
        // the rung: a guess never overrules a measurement's refusal.
        let tight = Some(Budget {
            available_gb: 5.0,
            reclaimable_gb: 0.0,
            total_gb: 5.0,
            reserved_headroom_gb: 0.0,
        });
        assert!(
            matches!(
                select_strategy(
                    request(),
                    &provider,
                    tight,
                    &[estimate_candidate, measured_candidate],
                ),
                Selection::Reject { .. }
            ),
            "an estimate must not overrule the measured refusal at its own rung"
        );
    }

    /// sc-18096: a fully estimate-backed ladder walks down to rung 4 under pressure, exactly like
    /// the measured and stale ladders — the epic's capability claim at the selector seam.
    #[test]
    fn a_fully_estimate_backed_ladder_can_select_a_deep_rung() {
        let provider = contract();
        let peaks = [40.0, 30.0, 20.0, 10.0, 2.0];
        let evidences = MemoryStrategy::ALL
            .into_iter()
            .zip(peaks)
            .map(|(strategy, gb)| {
                let mut record = estimate_evidence(strategy, &provider);
                record.key.parameters = cumulative_params(strategy);
                record.predicted_peak_bytes = (gb * BYTES_PER_GIB) as u64;
                record
            })
            .collect::<Vec<_>>();
        let candidates = MemoryStrategy::ALL
            .into_iter()
            .zip(&evidences)
            .map(|(strategy, record)| Candidate {
                selection: MemorySelection {
                    strategy,
                    parameters: cumulative_params(strategy),
                    tier: tier(),
                },
                evidence: record,
                closure_digest: INF,
                basis: CandidateBasis::EstimateFloor,
            })
            .collect::<Vec<_>>();
        let selection = select_strategy(
            request(),
            &provider,
            Some(Budget {
                available_gb: 3.0,
                reclaimable_gb: 0.0,
                total_gb: 48.0,
                reserved_headroom_gb: 0.0,
            }),
            &candidates,
        );
        assert!(
            matches!(
                selection,
                Selection::Selected {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::BoundedTransformerResidency,
                        ..
                    },
                    ..
                }
            ),
            "an estimate ladder must stay walkable down to rung 4: {selection:?}"
        );
    }

    /// sc-18096: the selector's tracing records the estimate admission, which BASIS produced it,
    /// and both the raw and widened peaks, so a later OOM under this selection is attributable to
    /// a synthesized estimate rather than a measurement.
    #[test]
    fn tracing_records_estimate_admission_with_its_basis_and_the_widened_peak() {
        use std::io::Write;

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .finish();

        let raw_peak_bytes: u64 = 10 * 1024 * 1024 * 1024;
        let provider = staged_only_provider();
        let mut staged = estimate_evidence(MemoryStrategy::StagedResidency, &provider);
        staged.predicted_peak_bytes = raw_peak_bytes;
        let result = tracing::subscriber::with_default(subscriber, || {
            select_strategy(
                request(),
                &provider,
                Some(Budget {
                    available_gb: 11.0,
                    reclaimable_gb: 0.0,
                    total_gb: 11.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[Candidate {
                    selection: MemorySelection {
                        strategy: MemoryStrategy::StagedResidency,
                        parameters: Default::default(),
                        tier: tier(),
                    },
                    evidence: &staged,
                    closure_digest: INF,
                    basis: CandidateBasis::EstimateFittedCurve,
                }],
            )
        });
        assert!(matches!(result, Selection::Selected { .. }), "{result:?}");
        let output = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
        let widened_peak_bytes =
            (raw_peak_bytes as f64 * (1.0 + CANDLE_ESTIMATE_MARGIN)).ceil() as u64;
        assert!(
            output
                .contains("estimate-backed memory-strategy candidate admitted with widened margin"),
            "estimate admission event missing from trace output: {output}"
        );
        assert!(
            output.contains("basis=\"fitted_curve\"") || output.contains("basis=fitted_curve"),
            "the admission event must name which basis produced the estimate: {output}"
        );
        assert!(
            output.contains(&format!("raw_peak_bytes={raw_peak_bytes}")),
            "raw peak missing from trace output: {output}"
        );
        assert!(
            output.contains(&format!("widened_peak_bytes={widened_peak_bytes}")),
            "widened peak missing from trace output: {output}"
        );
        assert!(
            output.contains("memory-strategy selection uses an estimate-backed candidate"),
            "estimate-selection event missing from trace output: {output}"
        );
    }
}
