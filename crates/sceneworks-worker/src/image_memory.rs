//! Worker-owned, backend-neutral image-memory selection (sc-15449).
//!
//! Providers own capabilities and lifecycle hooks. Evidence owns measured request cells. The worker
//! owns live-budget arithmetic and the normative strategy order. Optimized candidates are admitted
//! only through gen-core's canonical evidence validator.

use gen_core::{
    ImageMemoryCleanupSemantics, ImageMemoryEvidence, ImageMemoryEvidenceVerdict,
    ImageMemoryGeometry, ImageMemoryNumericTier, ImageMemoryProviderContract, ImageMemorySelection,
    ImageMemoryStrategy, ImageMemoryStrategySupport,
};

/// Execute one provider request through the adopted safety/lifecycle seam. A created scope receives
/// exactly one explicit terminal outcome; its Drop remains only a panic/unwind backstop.
pub fn generate_with_scope(
    generator: &dyn gen_core::Generator,
    request: &mut gen_core::GenerationRequest,
    context: Option<&gen_core::ImageMemoryRunContext>,
    on_progress: &mut dyn FnMut(gen_core::Progress),
) -> gen_core::Result<gen_core::GenerationOutput> {
    let Some(context) = context else {
        return generator.generate(request, on_progress);
    };
    if let gen_core::ImageMemorySafetyDecision::Reject { reason } =
        generator.image_memory_safety_check(context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let mut scope = generator.begin_image_memory_request(context)?;
    if context.selection.strategy.is_optimized() && scope.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} accepted an optimized image-memory selection without opening a request scope",
            generator.descriptor().id
        )));
    }
    if let Some(scope) = scope.as_mut() {
        if let Err(error) = scope.configure_request(request) {
            let message = error.to_string();
            let _ = scope.finish(gen_core::ImageMemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(error);
        }
    }
    let result = generator.generate(request, on_progress);
    if let Some(scope) = scope.as_mut() {
        let outcome = match &result {
            Ok(_) => gen_core::ImageMemoryRunOutcome::Complete,
            Err(gen_core::Error::Canceled) => gen_core::ImageMemoryRunOutcome::Canceled,
            Err(error) => gen_core::ImageMemoryRunOutcome::Error {
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
    pub backend: &'a str,
    pub tier: ImageMemoryNumericTier,
    pub mode: &'a str,
    pub overlay: Option<&'a str>,
    pub geometry: ImageMemoryGeometry,
    pub expected_sceneworks_revision: &'a str,
    pub expected_inference_revision: &'a str,
}

/// A provider estimate submitted to the selector. Cost is intentionally absent: strategy order is
/// worker-owned and follows [`ImageMemoryStrategy::ALL`].
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub selection: ImageMemorySelection,
    pub needed_gb: f64,
    pub evidence: &'a ImageMemoryEvidence,
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
        selection: ImageMemorySelection,
        needed_gb: f64,
        available_gb: f64,
    },
    Reject {
        needed_gb: f64,
        available_gb: f64,
    },
    Unverified {
        reason: ImageMemoryEvidenceVerdict,
    },
}

fn candidate_exclusion(
    request: RequestScope<'_>,
    contract: &ImageMemoryProviderContract,
    candidate: &Candidate<'_>,
) -> Option<ImageMemoryEvidenceVerdict> {
    if candidate.selection.tier != request.tier
        || candidate.evidence.key.tier != request.tier
        || candidate.evidence.key.strategy != candidate.selection.strategy
        || candidate.evidence.key.parameters != candidate.selection.parameters
    {
        return Some(ImageMemoryEvidenceVerdict::Invalid);
    }
    let key = &candidate.evidence.key;
    if key.backend != request.backend
        || key.mode != request.mode
        || key.overlay.as_deref() != request.overlay
        || key.geometry != request.geometry
    {
        return Some(ImageMemoryEvidenceVerdict::OutOfEnvelope);
    }
    if candidate.evidence.sceneworks_revision != request.expected_sceneworks_revision
        || candidate.evidence.inference_revision != request.expected_inference_revision
    {
        return Some(ImageMemoryEvidenceVerdict::Stale);
    }
    if contract.validate_selection(&candidate.selection).is_err() {
        return Some(ImageMemoryEvidenceVerdict::Invalid);
    }
    candidate.evidence.optimized_eligibility(contract).err()
}

/// Select the first fitting candidate in the normative resident → staged → bounded-decode →
/// bounded-attention → bounded-transformer order.
pub fn select_strategy(
    request: RequestScope<'_>,
    contract: &ImageMemoryProviderContract,
    budget: Option<Budget>,
    candidates: &[Candidate<'_>],
) -> Selection {
    if !contract.conformance_errors().is_empty()
        || contract.runtime.cancellation
            != ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
        || contract.runtime.error
            != ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows
    {
        return Selection::Unverified {
            reason: ImageMemoryEvidenceVerdict::Invalid,
        };
    }
    let Some(available_gb) = budget.and_then(Budget::effective_gb) else {
        return Selection::Unverified {
            reason: ImageMemoryEvidenceVerdict::Missing,
        };
    };

    let mut deepest = None;
    for strategy in ImageMemoryStrategy::ALL {
        let support = contract
            .capability(strategy)
            .map(|capability| &capability.support);
        if matches!(
            support,
            Some(ImageMemoryStrategySupport::StructurallyNotApplicable { .. })
        ) {
            continue;
        }
        if !matches!(support, Some(ImageMemoryStrategySupport::Implemented)) {
            continue;
        }
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.selection.strategy == strategy)
        else {
            return Selection::Unverified {
                reason: ImageMemoryEvidenceVerdict::Missing,
            };
        };
        if !candidate.needed_gb.is_finite() || candidate.needed_gb < 0.0 {
            return Selection::Unverified {
                reason: ImageMemoryEvidenceVerdict::Invalid,
            };
        }
        if let Some(reason) = candidate_exclusion(request, contract, candidate) {
            return Selection::Unverified { reason };
        }
        deepest = Some(candidate.needed_gb);
        if candidate.needed_gb <= available_gb {
            return Selection::Selected {
                selection: candidate.selection,
                needed_gb: candidate.needed_gb,
                available_gb,
            };
        }
    }
    deepest.map_or(
        Selection::Unverified {
            reason: ImageMemoryEvidenceVerdict::Missing,
        },
        |needed_gb| Selection::Reject {
            needed_gb,
            available_gb,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        ImageMemoryBackendRealization, ImageMemoryBudget, ImageMemoryCacheState,
        ImageMemoryCalibrationIdentity, ImageMemoryConformanceState, ImageMemoryEvidenceDimensions,
        ImageMemoryEvidenceKey, ImageMemoryFormulaKind, ImageMemoryLifecycleCapabilities,
        ImageMemoryMode, ImageMemoryParameterRanges, ImageMemoryParityContract,
        ImageMemoryParityResult, ImageMemoryPhase, ImageMemoryRequestScope, ImageMemoryRunContext,
        ImageMemoryRunOutcome, ImageMemoryStrategyCapability, ImageMemoryStrategyParameters,
        Precision, Quant,
    };
    use std::sync::{Arc, Mutex};

    const FP: &str = "provider-formula-v1";
    const SW: &str = "sc-15449-contract-v1";
    const INF: &str = "0c85bc9ff9fe161227efebf396a83db5e967d9ad";

    fn tier() -> ImageMemoryNumericTier {
        ImageMemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
        }
    }

    fn contract() -> ImageMemoryProviderContract {
        let mut contract = ImageMemoryProviderContract::compatibility_default(
            "test",
            ImageMemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: true,
                host_to_device_block_materialization: true,
            },
        );
        contract.strategies = ImageMemoryStrategy::ALL
            .into_iter()
            .map(|strategy| ImageMemoryStrategyCapability {
                strategy,
                support: ImageMemoryStrategySupport::Implemented,
                parameters: match strategy {
                    ImageMemoryStrategy::BoundedDecode => ImageMemoryParameterRanges {
                        decode_tile_edges: vec![512],
                        decode_overlaps: vec![128],
                        ..Default::default()
                    },
                    ImageMemoryStrategy::BoundedAttention => ImageMemoryParameterRanges {
                        attention_chunk_sizes: vec![1024],
                        ..Default::default()
                    },
                    ImageMemoryStrategy::BoundedTransformerResidency => {
                        ImageMemoryParameterRanges {
                            transformer_window_sizes: vec![1],
                            ..Default::default()
                        }
                    }
                    _ => Default::default(),
                },
            })
            .collect();
        contract.lifecycle = ImageMemoryLifecycleCapabilities {
            phases: vec![
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: true,
        };
        contract.formula = ImageMemoryFormulaKind::AssetBytesPlusHeadroom;
        contract.calibration = Some(ImageMemoryCalibrationIdentity::new(FP));
        contract
    }

    fn params(strategy: ImageMemoryStrategy) -> ImageMemoryStrategyParameters {
        match strategy {
            ImageMemoryStrategy::BoundedDecode => ImageMemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(128),
                ..Default::default()
            },
            ImageMemoryStrategy::BoundedAttention => ImageMemoryStrategyParameters {
                attention_chunk_size: Some(1024),
                ..Default::default()
            },
            ImageMemoryStrategy::BoundedTransformerResidency => ImageMemoryStrategyParameters {
                transformer_window_size: Some(1),
                ..Default::default()
            },
            _ => Default::default(),
        }
    }

    fn evidence(strategy: ImageMemoryStrategy) -> ImageMemoryEvidence {
        ImageMemoryEvidence {
            key: ImageMemoryEvidenceKey {
                resolved_route: "test".into(),
                backend: "candle".into(),
                tier: tier(),
                mode: "text_to_image".into(),
                overlay: None,
                geometry: ImageMemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                },
                strategy,
                parameters: params(strategy),
            },
            conformance: ImageMemoryConformanceState::Verified,
            dimensions: ImageMemoryEvidenceDimensions::VERIFIED,
            calibration_abi: gen_core::IMAGE_MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: FP.into(),
            sceneworks_revision: SW.into(),
            inference_revision: INF.into(),
            harness_version: "test-harness-v1".into(),
            predicted_peak_bytes: 1,
            observed_peak_bytes: Some(1),
            parity: ImageMemoryParityContract::Exact,
            parity_result: ImageMemoryParityResult::Passed,
        }
    }

    fn request() -> RequestScope<'static> {
        RequestScope {
            backend: "candle",
            tier: tier(),
            mode: "text_to_image",
            overlay: None,
            geometry: ImageMemoryGeometry {
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
        let resident = evidence(ImageMemoryStrategy::Resident);
        let staged = evidence(ImageMemoryStrategy::StagedResidency);
        let candidates = [
            Candidate {
                selection: ImageMemorySelection {
                    strategy: ImageMemoryStrategy::StagedResidency,
                    parameters: params(ImageMemoryStrategy::StagedResidency),
                    tier: tier(),
                },
                needed_gb: 16.0,
                evidence: &staged,
            },
            Candidate {
                selection: ImageMemorySelection {
                    strategy: ImageMemoryStrategy::Resident,
                    parameters: params(ImageMemoryStrategy::Resident),
                    tier: tier(),
                },
                needed_gb: 24.0,
                evidence: &resident,
            },
        ];
        let mut provider = contract();
        for strategy in [
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
            ImageMemoryStrategy::BoundedTransformerResidency,
        ] {
            provider.capability(strategy);
            provider
                .strategies
                .iter_mut()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support = ImageMemoryStrategySupport::StructurallyNotApplicable {
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
                selection: ImageMemorySelection {
                    strategy: ImageMemoryStrategy::StagedResidency,
                    ..
                },
                available_gb: 16.0,
                ..
            }
        ));
    }

    #[test]
    fn canonical_evidence_reason_is_preserved() {
        let mut staged = evidence(ImageMemoryStrategy::StagedResidency);
        staged.observed_peak_bytes = None;
        let candidate = Candidate {
            selection: ImageMemorySelection {
                strategy: ImageMemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            needed_gb: 8.0,
            evidence: &staged,
        };
        let mut provider = contract();
        provider
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == ImageMemoryStrategy::Resident)
            .unwrap()
            .support = ImageMemoryStrategySupport::StructurallyNotApplicable {
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
                reason: ImageMemoryEvidenceVerdict::Invalid,
            }
        );
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
        finish_calls: Vec<ImageMemoryRunOutcome>,
    }

    struct MockScope {
        record: Arc<Mutex<LifecycleRecord>>,
        configure_fails: bool,
    }

    impl ImageMemoryRequestScope for MockScope {
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

        fn enter_phase(&mut self, _phase: ImageMemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn leave_phase(&mut self, _phase: ImageMemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn configure_decode(
            &mut self,
            _tile_edge: u32,
            _overlap: u32,
            _geometry: ImageMemoryGeometry,
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

        fn finish(&mut self, outcome: ImageMemoryRunOutcome) -> gen_core::Result<()> {
            self.record.lock().unwrap().finish_calls.push(outcome);
            Ok(())
        }
    }

    struct MockGenerator {
        descriptor: gen_core::ModelDescriptor,
        contract: ImageMemoryProviderContract,
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

        fn image_memory_contract(&self) -> Option<&ImageMemoryProviderContract> {
            Some(&self.contract)
        }

        fn begin_image_memory_request(
            &self,
            _context: &ImageMemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn ImageMemoryRequestScope + '_>>> {
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

    fn run_context() -> ImageMemoryRunContext {
        ImageMemoryRunContext {
            selection: ImageMemorySelection {
                strategy: ImageMemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: tier(),
            },
            mode: ImageMemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: ImageMemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: ImageMemoryBudget {
                total_bytes: 16,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 8,
            cache_state: ImageMemoryCacheState::Cold,
            evidence_revision: "test".into(),
        }
    }

    fn assert_single_terminal(
        generate: MockGenerate,
        configure_fails: bool,
        expected: ImageMemoryRunOutcome,
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
        if matches!(expected, ImageMemoryRunOutcome::Complete) {
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
        assert_single_terminal(
            MockGenerate::Complete,
            false,
            ImageMemoryRunOutcome::Complete,
        );
        assert_single_terminal(
            MockGenerate::Canceled,
            false,
            ImageMemoryRunOutcome::Canceled,
        );
        assert_single_terminal(
            MockGenerate::Error,
            false,
            ImageMemoryRunOutcome::Error {
                message: "unsupported: render failed".into(),
            },
        );
        assert_single_terminal(
            MockGenerate::Complete,
            true,
            ImageMemoryRunOutcome::Error {
                message: "unsupported: configure failed".into(),
            },
        );
    }
}
