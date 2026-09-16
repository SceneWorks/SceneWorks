//! Request admission before tier loading must use the same ladder as the loaded request gate.
//!
//! All byte facts below are synthetic selector fixtures, never measured anchors or a claim that
//! these models run on an 8 GiB Mac. The linked-provider fixtures retain the production contract's
//! strategies, composition, parameters, precision floors, and materialization shape.

use super::*;

struct ContractOnlyGenerator {
    descriptor: gen_core::ModelDescriptor,
    contract: Option<MemoryProviderContract>,
}

impl gen_core::Generator for ContractOnlyGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        _: &gen_core::GenerationRequest,
        _: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        panic!("admission must not allocate tensors or generate images")
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        self.contract.as_ref()
    }
}

fn generator(
    route: &'static str,
    contract: Option<MemoryProviderContract>,
) -> ContractOnlyGenerator {
    ContractOnlyGenerator {
        descriptor: gen_core::ModelDescriptor {
            id: route,
            family: "admission-test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities {
                component_precision_floors: declared_component_floors(route),
                ..Default::default()
            },
            encoder_contract: None,
            denoiser_output_latent_space: None,
            required_components: &[],
            control_kinds: None,
        },
        contract,
    }
}

fn synthetic_plan(route: &'static str, assets_gib: f64, headroom_gib: f64) -> MlxRequestPlan {
    MlxRequestPlan {
        engine_id: route,
        model_id: route.to_owned(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
        asset_bytes: gib_to_bytes(assets_gib),
        folded_control_bytes: 0,
        folded_adapter_bytes: 0,
        activation_headroom_bytes: gib_to_bytes(headroom_gib),
        fixed_reserve_bytes: gib_to_bytes(2.0),
        calibration: MlxCalibrationConfig::Absent,
        load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
    }
}

fn inputs(width: u32, height: u32, count: u32) -> MlxRequestInputs {
    MlxRequestInputs {
        width,
        height,
        count,
        mode: "image_generation".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
        conditioning_windows: None,
    }
}

fn budget(total_gib: f64) -> MemoryBudget {
    MemoryBudget {
        total_bytes: gib_to_bytes(total_gib),
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// Exercise the production pre-load selector and loaded wrapper for the same cold request.
/// A resident constructor has loaded its declared weights before the loaded wrapper runs.
/// The wrapper's Generator cannot generate or load anything; only its declared contract is read.
fn select_with_loaded_parity(
    generator: &ContractOnlyGenerator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    load_policy: OffloadPolicy,
    budget: MemoryBudget,
) -> WorkerResult<MlxRequestAdmission> {
    let total_peak = plan.generic_total_peak_bytes(request_geometry(inputs));
    let pre_load = select_request_with_budget_using_bundle(
        MlxAdmissionPhase::TierProbe,
        generator.descriptor.capabilities.component_precision_floors,
        generator.contract.as_ref(),
        |base| {
            generator.contract.as_ref().map_or(
                gen_core::MemoryPeakBreakdown::from_unattributed(base),
                |contract| contract.predicted_peak_from_base(base),
            )
        },
        plan,
        inputs,
        MemoryCacheState::Cold,
        load_policy,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        budget,
        total_peak,
        0,
        &[],
        None,
    );
    let loaded_budget = if load_policy == OffloadPolicy::Resident {
        MemoryBudget {
            committed_bytes: generator
                .contract
                .as_ref()
                .map_or(plan.asset_bytes, |contract| contract.total_resident_bytes()),
            ..budget
        }
    } else {
        budget
    };
    let loaded = evaluate_request_with_budget(
        generator,
        plan,
        inputs,
        MemoryCacheState::Cold,
        load_policy,
        loaded_budget,
        total_peak,
        0,
        &[],
    );
    match (&pre_load, loaded) {
        (Ok(MlxRequestAdmission::Admitted(before)), Ok(after)) => {
            assert_eq!(before.context.selection, after.context.selection);
            assert_eq!(before.context.geometry, after.context.geometry);
            assert_eq!(before.context.load_shape, after.context.load_shape);
            assert_eq!(before.context.mode, after.context.mode);
            assert_eq!(
                before.context.optimization_authority,
                after.context.optimization_authority
            );
            assert_eq!(before.process_limit_bytes, after.process_limit_bytes);
            assert_eq!(
                format!("{:?}", before.memory),
                format!("{:?}", after.memory)
            );
        }
        (Ok(MlxRequestAdmission::Rejected(before)), Err(after)) | (Err(before), Err(after)) => {
            assert_eq!(before.to_string(), after.to_string());
        }
        (Ok(MlxRequestAdmission::Admitted(_)), Err(error)) => {
            panic!("pre-load admitted but loaded selector refused: {error}")
        }
        (_, Ok(_)) => panic!("loaded selector admitted a pre-load refusal"),
    }
    pre_load
}

fn admitted(result: WorkerResult<MlxRequestAdmission>) -> Box<MlxRequestEvaluation> {
    match result.expect("a valid contract must produce a budget decision") {
        MlxRequestAdmission::Admitted(evaluation) => evaluation,
        MlxRequestAdmission::Rejected(error) => panic!("expected admission: {error}"),
    }
}

fn rejected(result: WorkerResult<MlxRequestAdmission>) -> String {
    match result.expect("a no-fit request is a tier refusal, not an invalid contract") {
        MlxRequestAdmission::Rejected(error) => error.to_string(),
        MlxRequestAdmission::Admitted(_) => panic!("expected a request-scoped budget refusal"),
    }
}

#[test]
fn absent_provider_contract_preserves_resident_only_admission_and_refusal() {
    let generator = generator("admission_fixture", None);
    let plan = synthetic_plan("admission_fixture", 12.0, 18.0);
    let inputs = inputs(1152, 2048, 4);
    let fit = admitted(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs,
        OffloadPolicy::Resident,
        budget(256.0),
    ));
    assert_eq!(fit.context.selection.strategy, MemoryStrategy::Resident);
    assert_eq!(fit.context.geometry.batch, 1);
    let error = rejected(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs,
        OffloadPolicy::Resident,
        budget(8.0),
    ));
    assert!(error.contains("1152x2048 count 4"), "{error}");
    assert!(error.contains("complete pipeline"), "{error}");
}

#[test]
fn invalid_provider_contract_is_not_a_retryable_tier_budget_refusal() {
    let mut contract = MemoryProviderContract::compatibility_default(
        "admission_fixture",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: false,
            cache_eviction: true,
        },
    );
    // A declared base total that disagrees with its components is structurally invalid.
    contract.asset_facts.base_bytes = gib_to_bytes(12.0);
    assert!(!contract.conformance_errors().is_empty());
    let result = select_with_loaded_parity(
        &generator("admission_fixture", Some(contract)),
        &synthetic_plan("admission_fixture", 12.0, 18.0),
        &inputs(1024, 1024, 1),
        OffloadPolicy::Resident,
        budget(256.0),
    );
    let Err(error) = result else {
        panic!("invalid contracts must not be classified as a lower-tier budget retry")
    };
    assert!(
        error.to_string().contains("structurally admissible"),
        "{error}"
    );
}

#[test]
fn exact_measured_candidate_excluded_by_foreign_reserve_is_a_retryable_tier_refusal() {
    use gen_core::{
        MemoryCalibrationIdentity, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
        MemoryStrategyCapability, MemoryStrategySupport,
    };

    // Reuse the checked-in synthetic capture without rewriting its measurements or identities.
    // Its one exact q4/1024-square bounded-decode envelope is 5 GiB plus 2.25 GiB foreign reserve.
    let bundle = sceneworks_core::memory_calibration::load_bundle(include_str!(
        "../tests/fixtures/mlx-memory-calibration.json"
    ))
    .expect("valid synthetic calibration fixture");
    let parameters = JsonObject::from_iter([
        ("decodeTileEdge".to_owned(), serde_json::json!(512)),
        ("decodeOverlap".to_owned(), serde_json::json!(128)),
    ]);
    let fingerprint = format!("SceneWorks/fixture@{}:packed-q4", "c".repeat(40));
    let binding = MlxCalibrationBinding {
        query: CalibrationBinding {
            abi: sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI,
            load_shape: LoadShapeKey::EagerMaterialization,
            fingerprint: "fixture-formula-v2".to_owned(),
            scene_works_revision: "a".repeat(40),
            matrix_source_revision: "source-tree:1111111".to_owned(),
            inference_revision: "b".repeat(40),
            inference_closure_digest: "d".repeat(64),
            artifact_repository: "SceneWorks/fixture".to_owned(),
            artifact_resolved_revision: "c".repeat(40),
            artifact_variant: "packed-q4".to_owned(),
            resolved_path_fingerprint: fingerprint.clone(),
        },
        provider: "fixture_provider".to_owned(),
        tier: "q4".to_owned(),
        mode: "text_to_image".to_owned(),
        overlay: "none".to_owned(),
        geometry: CalibrationGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
        },
        rung: StrategyRung::BoundedDecode,
        selection_parameters: parse_evidence_parameters(
            StrategyRung::BoundedDecode,
            &crate::memory_strategy::default_engaged_composition(StrategyRung::BoundedDecode),
            &parameters,
        )
        .expect("valid captured decode parameters"),
        parameters,
    };
    let mut plan = synthetic_plan("fixture_provider", 3.0, 6.0);
    plan.model_id = "fixture_model".to_owned();
    plan.calibration = MlxCalibrationConfig::Valid(MlxCalibrationSet {
        bindings: vec![binding],
        resolved: ResolvedArtifactProvenance {
            identity: crate::model_jobs::ResolvedArtifactIdentity {
                repository: "SceneWorks/fixture".to_owned(),
                revision: "c".repeat(40),
                variant: "packed-q4".to_owned(),
                fingerprint,
            },
            fixed_artifact_tier: Some("q4".to_owned()),
        },
    });
    let mut contract = MemoryProviderContract::compatibility_default(
        "fixture_provider",
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        "fixture-formula-v2",
        gen_core::LoadShape::EagerMaterialization,
    ));
    contract.asset_facts.base_bytes = gib_to_bytes(3.0);
    contract.asset_facts.transformer_bytes = gib_to_bytes(3.0);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if matches!(
                strategy,
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
            ) {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            },
            parameters: if strategy == MemoryStrategy::BoundedDecode {
                MemoryParameterRanges {
                    decode_tile_edges: vec![512],
                    decode_overlaps: vec![128],
                    ..Default::default()
                }
            } else {
                MemoryParameterRanges::default()
            },
        })
        .collect();
    assert!(contract.conformance_errors().is_empty());
    let request = inputs(1024, 1024, 4);
    let constrained = budget(6.0);
    let route = evidence_admission_route(&bundle, &plan, &request, "text_to_image", constrained)
        .expect("the exact binding must remain an evidence route even when it does not fit");
    assert_eq!(route.path, AdmissionPath::Evidence);
    assert_eq!(route.evidence.len(), 1);
    let candidate = &route.evidence[0];
    assert_eq!(candidate.evidence.predicted_peak_bytes, gib_to_bytes(5.0));
    assert_eq!(candidate.foreign_reserve_bytes, gib_to_bytes(2.25));
    assert!(candidate.evidence.predicted_peak_bytes <= constrained.total_bytes);
    assert!(
        candidate.evidence.predicted_peak_bytes + candidate.foreign_reserve_bytes
            > constrained.total_bytes
    );

    let select = |live_budget| {
        select_request_with_budget_using_bundle(
            MlxAdmissionPhase::TierProbe,
            &[],
            Some(&contract),
            |base| contract.predicted_peak_from_base(base),
            &plan,
            &request,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            live_budget,
            gib_to_bytes(4.0),
            0,
            &[],
            Some(&bundle),
        )
    };
    let error = rejected(select(constrained));
    assert!(error.contains("1024x1024 count 4"), "{error}");
    assert!(error.contains("smallest verified"), "{error}");
    assert!(error.contains("no exact candidate fits"), "{error}");

    let fitting = admitted(select(budget(8.5)));
    assert_eq!(
        fitting.context.selection.strategy,
        MemoryStrategy::BoundedDecode
    );
    assert_eq!(fitting.context.geometry.batch, 1);
    assert_eq!(
        fitting.context.optimization_authority,
        MemoryOptimizationAuthority::Calibrated
    );
    assert!(fitting.process_limit_bytes.is_some());
}

#[cfg(target_os = "macos")]
fn shipped_synthetic_contract(route: &'static str, assets_gib: f64) -> MemoryProviderContract {
    let mut contract = crate::inference_runtime::media()
        .memory_contract_surfaces()
        .expect("weights-free linked-provider inventory")
        .into_iter()
        .find(|surface| {
            surface.contract.provider_id == route
                && surface.selector.tier == gen_core::MemoryContractSurfaceTier::Q4
                && surface.selector.offload_policy == OffloadPolicy::Sequential
                && surface.selector.load_shape == gen_core::LoadShape::DeferredMaterialization
        })
        .unwrap_or_else(|| panic!("missing shipped q4/deferred/sequential surface for {route}"))
        .contract;
    contract.asset_facts.base_bytes = gib_to_bytes(assets_gib);
    contract.asset_facts.conditioning_bytes = gib_to_bytes(assets_gib / 8.0);
    contract.asset_facts.transformer_bytes = gib_to_bytes(assets_gib * 7.0 / 8.0);
    contract.asset_facts.decoder_bytes = 0;
    // This helper uses synthetic component sizes, so its exact streaming blocks are synthetic.
    if let Some(phase) = contract.phase_facts.as_mut() {
        phase.transformer_stream = Some(gen_core::StreamedWeightFacts {
            resident_bytes: 0,
            stacks: vec![vec![contract.asset_facts.transformer_bytes / 70; 70]],
        });
    }

    assert!(
        contract.conformance_errors().is_empty(),
        "{route}: {:?}",
        contract.conformance_errors()
    );
    contract
}

#[cfg(target_os = "macos")]
#[test]
fn krea_legacy_weight_rejection_reaches_the_actual_request_ladder() {
    assert!(matches!(
        decide_residency_with_headroom(
            gib_to_bytes(12.0),
            gib_to_bytes(1.5),
            Some(MlxMemoryBudget { total_gb: 8.0 }),
            true,
            18.0,
        ),
        ResidencyOutcome::Reject { .. }
    ));
    let contract = shipped_synthetic_contract("krea_2_turbo", 12.0);
    assert!(matches!(
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .map(|c| &c.support),
        Some(gen_core::MemoryStrategySupport::Implemented)
    ));
    let error = rejected(select_with_loaded_parity(
        &generator("krea_2_turbo", Some(contract)),
        &synthetic_plan("krea_2_turbo", 12.0, 18.0),
        &inputs(1152, 2048, 4),
        OffloadPolicy::Sequential,
        budget(8.0),
    ));
    assert!(error.contains("1152x2048 count 4"), "{error}");
    assert!(error.contains("safely available"), "{error}");
    assert!(error.contains("complete pipeline"), "{error}");
}

#[cfg(target_os = "macos")]
#[test]
fn krea_preload_bounded_rung_wins_and_larger_geometry_refuses_without_changing_count() {
    let generator = generator(
        "krea_2_turbo",
        Some(shipped_synthetic_contract("krea_2_turbo", 40.0)),
    );
    let plan = synthetic_plan("krea_2_turbo", 40.0, 6.0);
    // The old gate cannot load the 35 GiB staged weights on a 32 GiB host. The provider's
    // deferred windowed strategy has a smaller working set and must get an admission decision.
    assert!(matches!(
        decide_residency_with_headroom(
            gib_to_bytes(40.0),
            gib_to_bytes(5.0),
            Some(MlxMemoryBudget { total_gb: 32.0 }),
            true,
            6.0,
        ),
        ResidencyOutcome::Reject { .. }
    ));
    let fit = admitted(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs(1024, 1024, 4),
        OffloadPolicy::Sequential,
        budget(32.0),
    ));
    assert_eq!(
        fit.context.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    assert_eq!(
        fit.context.optimization_authority,
        MemoryOptimizationAuthority::Estimated
    );
    assert_eq!(fit.context.geometry.batch, 1);
    assert!(fit.memory.stream_transformer_blocks);
    assert!(fit.memory.stage_residency);
    assert_eq!(
        fit.context.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
    let single = admitted(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs(1024, 1024, 1),
        OffloadPolicy::Sequential,
        budget(32.0),
    ));
    assert_eq!(fit.context.selection, single.context.selection);
    let error = rejected(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs(4096, 4096, 4),
        OffloadPolicy::Sequential,
        budget(32.0),
    ));
    assert!(error.contains("4096x4096 count 4"), "{error}");
}

#[cfg(target_os = "macos")]
#[test]
fn refused_load_declaration_does_not_rediscover_krea_optimized_rungs() {
    let generator = generator(
        "krea_2_turbo",
        Some(shipped_synthetic_contract("krea_2_turbo", 40.0)),
    );
    let mut plan = synthetic_plan("krea_2_turbo", 40.0, 6.0);
    plan.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
    rejected(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs(1024, 1024, 1),
        OffloadPolicy::Sequential,
        budget(32.0),
    ));
    let fit = admitted(select_with_loaded_parity(
        &generator,
        &plan,
        &inputs(1024, 1024, 1),
        OffloadPolicy::Sequential,
        budget(256.0),
    ));
    assert_eq!(fit.context.selection.strategy, MemoryStrategy::Resident);
    assert!(!fit.memory.stream_transformer_blocks);
}

#[cfg(target_os = "macos")]
#[test]
fn krea_zimage_anima_declared_contracts_keep_preload_and_loaded_admission_in_sync() {
    for route in ["krea_2_turbo", "z_image_turbo", "anima_base"] {
        let generator = generator(route, Some(shipped_synthetic_contract(route, 40.0)));
        let plan = synthetic_plan(route, 40.0, 6.0);
        let request = inputs(1024, 1024, 3);
        let fit = admitted(select_with_loaded_parity(
            &generator,
            &plan,
            &request,
            OffloadPolicy::Sequential,
            budget(256.0),
        ));
        assert_eq!(fit.context.selection.tier.quant, Some(Quant::Q4), "{route}");
        assert_eq!(fit.context.geometry.batch, 1, "{route}");
        assert_eq!(
            fit.context.load_shape,
            gen_core::LoadShape::DeferredMaterialization,
            "{route}"
        );
        let error = rejected(select_with_loaded_parity(
            &generator,
            &plan,
            &request,
            OffloadPolicy::Sequential,
            budget(1.0),
        ));
        assert!(error.contains(route), "{error}");
        assert!(error.contains("1024x1024 count 3"), "{error}");
    }
}

#[test]
fn resident_tier_probe_preserves_actual_loaded_allowance_without_discounting_pipeline_bytes() {
    let generator = generator("resident_fixture", None);
    let plan = synthetic_plan("resident_fixture", 100.0, 18.0);
    let request = inputs(1024, 1024, 1);
    let total_peak = plan.generic_total_peak_bytes(request_geometry(&request));
    assert_eq!(total_peak, gib_to_bytes(118.0));
    let mut cold = budget(128.0);
    cold.reserved_headroom_bytes = gib_to_bytes(2.0);
    let select = |phase| {
        select_request_with_budget_using_bundle(
            phase,
            &[],
            None,
            gen_core::MemoryPeakBreakdown::from_unattributed,
            &plan,
            &request,
            MemoryCacheState::Cold,
            OffloadPolicy::Resident,
            crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
            cold,
            total_peak,
            0,
            &[],
            None,
        )
    };
    // The naive pre-load extraction applied the loaded calculation without its real resident
    // allocation credit. This is the reproduced false rejection at the same host boundary.
    rejected(select(MlxAdmissionPhase::Generation));
    let before = admitted(select(MlxAdmissionPhase::TierProbe));
    assert_eq!(before.context.budget.committed_bytes, 0);
    assert_eq!(before.context.predicted_peak_bytes, total_peak);
    assert_eq!(before.context.selection.strategy, MemoryStrategy::Resident);
    let loaded_budget = MemoryBudget {
        committed_bytes: gib_to_bytes(100.0),
        ..cold
    };
    let after = evaluate_request_with_budget(
        &generator,
        &plan,
        &request,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        loaded_budget,
        total_peak,
        0,
        &[],
    )
    .expect("the real loaded resident path fits at this boundary");
    assert_eq!(before.context.selection, after.context.selection);
    // Known weights only leave the uncertainty allowance. A physically undersized machine still
    // refuses, because the complete pipeline remains charged before any weights are loaded.
    let smaller = MemoryBudget {
        total_bytes: gib_to_bytes(110.0),
        ..cold
    };
    rejected(select_request_with_budget_using_bundle(
        MlxAdmissionPhase::TierProbe,
        &[],
        None,
        gen_core::MemoryPeakBreakdown::from_unattributed,
        &plan,
        &request,
        MemoryCacheState::Cold,
        OffloadPolicy::Resident,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        smaller,
        total_peak,
        0,
        &[],
        None,
    ));
}
