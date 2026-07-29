//! Worker-owned, backend-neutral memory-strategy selection (sc-15449).
//!
//! Providers own capabilities and lifecycle hooks. Evidence owns measured request cells. The worker
//! owns live-budget arithmetic and the normative strategy order. Optimized candidates are admitted
//! only through gen-core's canonical evidence validator.

use gen_core::{
    MemoryCleanupSemantics, MemoryEvidence, MemoryEvidenceDimension, MemoryEvidenceVerdict,
    MemoryGeometry, MemoryNumericTier, MemoryProviderContract, MemorySelection, MemoryStrategy,
    MemoryStrategySupport,
};

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
    pub expected_sceneworks_revision: &'a str,
    pub expected_inference_revision: &'a str,
}

/// A provider estimate submitted to the selector. Cost is intentionally absent: strategy order is
/// worker-owned and follows [`MemoryStrategy::ALL`].
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub selection: MemorySelection,
    pub evidence: &'a MemoryEvidence,
}

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Evidence stores an integer byte ceiling. Admission converts that canonical value to GiB exactly
/// once; callers cannot submit a second floating-point estimate with a lower coefficient.
fn evidence_peak_gb(evidence: &MemoryEvidence) -> f64 {
    evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB
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
        MemoryEvidenceVerdict::Invalid => 6,
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
) -> Option<MemoryEvidenceVerdict> {
    if request.resolved_route != contract.provider_id
        || request.backend != contract.backend.backend_id()
        || candidate.evidence.key.resolved_route != request.resolved_route
        || candidate.evidence.key.resolved_route != contract.provider_id
        || candidate.evidence.key.backend != contract.backend.backend_id()
        || candidate.selection.tier != request.tier
        || candidate.evidence.key.tier != request.tier
        || candidate.evidence.key.strategy != candidate.selection.strategy
        || candidate.evidence.key.parameters != candidate.selection.parameters
    {
        return Some(MemoryEvidenceVerdict::Invalid);
    }
    let key = &candidate.evidence.key;
    if key.backend != request.backend
        || key.mode != request.mode
        || key.overlay.as_deref() != request.overlay
        || key.geometry != request.geometry
    {
        return Some(MemoryEvidenceVerdict::OutOfEnvelope);
    }
    if candidate.evidence.sceneworks_revision != request.expected_sceneworks_revision
        || candidate.evidence.inference_revision != request.expected_inference_revision
    {
        return Some(MemoryEvidenceVerdict::Stale);
    }
    if contract.validate_selection(&candidate.selection).is_err() {
        return Some(MemoryEvidenceVerdict::Invalid);
    }
    candidate
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
        let mut eligible = Vec::new();
        let mut first_exclusion = None;
        for candidate in rung_candidates {
            if let Some(reason) = candidate_exclusion(request, contract, candidate) {
                accumulate_reason(&mut first_exclusion, reason);
                tracing::warn!(
                    route = request.resolved_route,
                    backend = request.backend,
                    ?strategy,
                    ?reason,
                    "memory-strategy candidate excluded"
                );
            } else {
                eligible.push(candidate);
            }
        }
        if eligible.is_empty() {
            accumulate_reason(
                &mut first_unknown,
                first_exclusion.unwrap_or(MemoryEvidenceVerdict::Missing),
            );
            continue;
        }
        let parameter_preference = |candidate: &&Candidate<'_>| {
            let parameters = candidate.selection.parameters;
            (
                parameters.decode_tile_edge.unwrap_or(0),
                parameters.attention_chunk_size.unwrap_or(0),
                parameters.transformer_window_size.unwrap_or(0),
                parameters.decode_overlap.unwrap_or(0),
            )
        };
        if let Some(candidate) = eligible
            .iter()
            .filter(|candidate| evidence_peak_gb(candidate.evidence) <= available_gb)
            .max_by(|left, right| {
                left.evidence
                    .predicted_peak_bytes
                    .cmp(&right.evidence.predicted_peak_bytes)
                    .then_with(|| parameter_preference(left).cmp(&parameter_preference(right)))
            })
        {
            let needed_gb = evidence_peak_gb(candidate.evidence);
            return Selection::Selected {
                selection: candidate.selection,
                needed_gb,
                available_gb,
            };
        }
        let minimum = eligible
            .iter()
            .map(|candidate| evidence_peak_gb(candidate.evidence))
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
        MemoryBackendRealization, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity,
        MemoryConformanceState, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryFormulaKind,
        MemoryLifecycleCapabilities, MemoryMode, MemoryParameterRanges, MemoryParityContract,
        MemoryParityResult, MemoryPhase, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
        MemoryStrategyCapability, MemoryStrategyParameters, Precision, Quant,
    };
    use std::sync::{Arc, Mutex};

    const FP: &str = "provider-formula-v1";
    const SW: &str = "sc-15449-contract-v1";
    const INF: &str = "0c85bc9ff9fe161227efebf396a83db5e967d9ad";

    fn tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
        }
    }

    fn contract() -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default(
            "test",
            MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
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
        contract.formula = MemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.calibration = Some(MemoryCalibrationIdentity::new(FP));
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
                backend: "candle".into(),
                tier: tier(),
                mode: "text_to_image".into(),
                overlay: None,
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                },
                strategy,
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
            },
            expected_sceneworks_revision: SW,
            expected_inference_revision: INF,
        }
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
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    parameters: params(MemoryStrategy::Resident),
                    tier: tier(),
                },
                evidence: &resident,
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
                .support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "test".into(),
            };
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
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
        };
        let mut provider = contract();
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::Resident)
            .unwrap()
            .support = MemoryStrategySupport::StructurallyNotApplicable {
            reason: "test".into(),
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
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
        };
        assert_eq!(
            select_strategy(
                request(),
                &staged_only_provider(),
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
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
        };
        assert_eq!(
            select_strategy(
                request(),
                &staged_only_provider(),
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
        let mut stale_staged = evidence(MemoryStrategy::StagedResidency);
        stale_staged.inference_revision = "1111111111111111111111111111111111111111".into();
        let mut bounded_decode = evidence(MemoryStrategy::BoundedDecode);
        bounded_decode.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        let candidates = [
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &stale_staged,
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::BoundedDecode,
                    parameters: params(MemoryStrategy::BoundedDecode),
                    tier: tier(),
                },
                evidence: &bounded_decode,
            },
        ];
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
                .support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "selector unit-test envelope".into(),
            };
        }
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
            },
            Candidate {
                selection: MemorySelection {
                    strategy: MemoryStrategy::StagedResidency,
                    parameters: Default::default(),
                    tier: tier(),
                },
                evidence: &staged,
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
                .support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "selector unit-test envelope".into(),
            };
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
                .support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "selector unit-test envelope".into(),
            };
        }
        provider
    }

    #[test]
    fn canonical_predicted_peak_bytes_are_the_only_admission_number() {
        let mut staged = evidence(MemoryStrategy::StagedResidency);
        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024 + 1;
        let selection = MemorySelection {
            strategy: MemoryStrategy::StagedResidency,
            parameters: Default::default(),
            tier: tier(),
        };
        let candidate = Candidate {
            selection,
            evidence: &staged,
        };
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 0.0,
            total_gb: 8.0,
            reserved_headroom_gb: 0.0,
        });
        assert!(matches!(
            select_strategy(request(), &staged_only_provider(), budget, &[candidate]),
            Selection::Reject { .. }
        ));

        staged.predicted_peak_bytes = 8 * 1024 * 1024 * 1024;
        assert!(matches!(
            select_strategy(
                request(),
                &staged_only_provider(),
                budget,
                &[Candidate {
                    selection,
                    evidence: &staged,
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
        let staged = evidence(MemoryStrategy::StagedResidency);
        let candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            evidence: &staged,
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
            select_strategy(wrong_route, &staged_only_provider(), budget, &[candidate],),
            Selection::Unverified {
                reason: MemoryEvidenceVerdict::Invalid,
            }
        );

        let mut wrong_backend = staged.clone();
        wrong_backend.key.backend = "mlx".into();
        assert_eq!(
            select_strategy(
                request(),
                &staged_only_provider(),
                budget,
                &[Candidate {
                    selection: candidate.selection,
                    evidence: &wrong_backend,
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
        let high_candidate = Candidate {
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedDecode,
                parameters: params(MemoryStrategy::BoundedDecode),
                tier: tier(),
            },
            evidence: &high,
        };
        let low_candidate = Candidate {
            evidence: &low,
            ..high_candidate
        };
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
                .support = MemoryStrategySupport::StructurallyNotApplicable {
                reason: "selector unit-test envelope".into(),
            };
        }
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters
            .decode_tile_edges = vec![512, 768];
        let mut high_selection = high_candidate.selection;
        high_selection.parameters.decode_tile_edge = Some(768);
        let high_candidate = Candidate {
            selection: high_selection,
            evidence: &high,
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
        invalid.key.backend = "mlx".into();
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
                    id: "lifecycle_mock",
                    family: "test",
                    backend: "candle",
                    modality: gen_core::Modality::Image,
                    capabilities: Default::default(),
                    required_components: &[],
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
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
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
                // Rung 1 is Missing; rung 4 declares an engagement prerequisite on it.
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
}
