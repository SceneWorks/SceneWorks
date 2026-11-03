use super::*;
use mlx_gen::gen_core::MemoryStrategySupport;

fn minimal_request(provider: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": "sdxl",
                "tier": "q4",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "loadShape": "deferred_materialization",
            "fixture": "sdxl-base-mlx-q4-768-seed18379-step2",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} }
        }
    })
}

fn fixture_spec(root: &std::path::Path) -> LoadSpec {
    for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
        let directory = root.join(component);
        std::fs::create_dir_all(&directory).unwrap();
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
    }
    std::fs::write(
        root.join("unet").join("config.json"),
        r#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .unwrap();
    LoadSpec::new(WeightsSource::Dir(root.to_owned()))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization)
}

#[test]
fn sdxl_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
    for provider in ["realvisxl", "sdxl_control", "qwen_image"] {
        let error = run_sdxl(&minimal_request(provider))
            .expect_err("a foreign provider must not reach the SDXL arm");
        assert_eq!(
            error,
            format!("MLX SDXL base calibration does not implement provider {provider:?}")
        );
    }
}

#[test]
fn sdxl_admission_context_is_reference_free_text_to_image() {
    let calibration = MemoryCalibrationIdentity::new(
        SDXL_CALIBRATION_FINGERPRINT,
        LoadShape::DeferredMaterialization,
    );
    let context = sdxl_context(
        MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        },
        &calibration,
        &calibration.fingerprint,
        768,
        768,
        1,
        1,
    );
    assert_eq!(context.mode, MemoryMode::TextToImage);
    assert!(!context.has_reference);
    assert_eq!(context.geometry.reference_count, 0);
    assert!(context.overlay.is_none());
    assert!(!context.use_pid);
}

#[test]
fn sdxl_arm_rejects_every_non_base_target_axis_before_weight_work() {
    let base = minimal_request(SDXL_PROVIDER);
    assert!(validate_sdxl_target(&base).is_ok());
    for (pointer, value) in [
        // sc-22729: `realvisxl` used to be the mutation here, and it is now a REAL family
        // member with its own artifact family. A model id no member serves is still refused.
        ("/planned/target/modelId", json!("sdxl_turbo")),
        ("/planned/target/mode", json!("image_to_image")),
        ("/planned/target/geometry/batch", json!(2)),
        ("/planned/target/geometry/frames", json!(2)),
    ] {
        let mut request = base.clone();
        *request.pointer_mut(pointer).unwrap_or_else(|| {
            panic!("test mutation path must exist before assignment: {pointer}")
        }) = value;
        assert!(
            validate_sdxl_target(&request).is_err(),
            "SDXL target mutation {pointer} must fail closed"
        );
    }

    for (field, value) in [("referenceCount", json!(1)), ("hasReference", json!(true))] {
        let mut request = base.clone();
        request["planned"]["target"][field] = value;
        assert!(
            validate_sdxl_target(&request).is_err(),
            "SDXL target mutation {field} must fail closed"
        );
    }
}

#[test]
fn sdxl_selection_rejects_a_non_dit_or_detached_window_component() {
    let mut request = minimal_request(SDXL_PROVIDER);
    request["planned"]["strategy"] = json!({
        "rung": "bounded_transformer_residency",
        "engagedRungs": ["resident", "staged_residency", "bounded_decode", "bounded_transformer_residency"],
        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64, "transformerWindowSize": 1, "transformerWindowComponent": "dit" }
    });
    assert_eq!(
        planned_selection(&request)
            .unwrap()
            .parameters
            .transformer_window_component,
        Some(TransformerComponent::Dit)
    );
    validate_sdxl_selection_parameters(&request, &planned_selection(&request).unwrap()).unwrap();
    for missing in ["decodeTileEdge", "decodeOverlap"] {
        let mut incomplete = request.clone();
        incomplete["planned"]["strategy"]["parameters"]
            .as_object_mut()
            .unwrap()
            .remove(missing);
        let selection = planned_selection(&incomplete).unwrap();
        assert!(validate_sdxl_selection_parameters(&incomplete, &selection).is_err());
    }

    for component in ["text_encoder", "unknown"] {
        request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] =
            json!(component);
        assert!(
            planned_selection(&request).is_err(),
            "component {component:?} must not execute as Dit"
        );
    }
    // sc-18663 taught the SHARED parser `"both"`, because `minimax_h3` declares exactly that.
    // The SDXL guarantee is unchanged and is asserted here rather than assumed: the parse now
    // succeeds, it does NOT produce `Dit`, and the SDXL arm still refuses it.
    request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] = json!("both");
    let both = planned_selection(&request).expect("the shared parser accepts \"both\"");
    assert_eq!(
        both.parameters.transformer_window_component,
        Some(TransformerComponent::Both),
        "\"both\" must not execute as Dit"
    );
    assert!(
        validate_sdxl_selection_parameters(&request, &both).is_err(),
        "the SDXL arm calibrates an explicit Dit window only"
    );
    request["planned"]["strategy"]["parameters"]["transformerWindowComponent"] = json!("dit");
    request["planned"]["strategy"]["parameters"] = json!({ "transformerWindowComponent": "dit" });
    assert!(planned_selection(&request).is_err());

    request["planned"]["strategy"] = json!({
        "rung": "resident",
        "engagedRungs": ["resident"],
        "parameters": { "unknownParameter": 1 }
    });
    let selection = planned_selection(&request).unwrap();
    assert!(validate_sdxl_selection_parameters(&request, &selection).is_err());

    request["planned"]["strategy"] = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
    });
    let selection = planned_selection(&request).unwrap();
    assert!(validate_sdxl_selection_parameters(&request, &selection).is_ok());
    for parameters in [
        json!({ "decodeTileEdge": 512 }),
        json!({ "decodeOverlap": 64 }),
        json!({ "decodeTileEdge": 127, "decodeOverlap": 64 }),
        json!({ "decodeTileEdge": 512, "decodeOverlap": 128 }),
        json!({ "decodeTileEdge": 512, "decodeOverlap": 64, "unknownParameter": 1 }),
    ] {
        request["planned"]["strategy"]["parameters"] = parameters;
        let selection = planned_selection(&request).unwrap();
        assert!(validate_sdxl_selection_parameters(&request, &selection).is_err());
    }
    request["planned"]["strategy"] = json!({
        "rung": "bounded_attention",
        "engagedRungs": ["resident", "bounded_attention"],
        "parameters": {}
    });
    let selection = planned_selection(&request).unwrap();
    assert!(validate_sdxl_selection_parameters(&request, &selection).is_err());
}

#[test]
fn sdxl_runtime_complete_sweep_attests_the_exact_tuple_without_a_string_axis() {
    let mut request = minimal_request(SDXL_PROVIDER);
    request["planned"]["strategy"] = json!({
        "rung": "bounded_transformer_residency",
        "engagedRungs": ["resident", "staged_residency", "bounded_decode", "bounded_transformer_residency"],
        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64, "transformerWindowSize": 5, "transformerWindowComponent": "dit" }
    });
    assert_eq!(
        sdxl_runtime_complete_sweep(&request).unwrap(),
        json!({
            "axes": [
                { "parameter": "decodeTileEdge", "testedValues": [512] },
                { "parameter": "decodeOverlap", "testedValues": [64] },
                { "parameter": "transformerWindowSize", "testedValues": [5] }
            ],
            "cases": [{
                "parameters": {
                    "decodeTileEdge": 512,
                    "decodeOverlap": 64,
                    "transformerWindowSize": 5,
                    "transformerWindowComponent": "dit"
                },
                "result": "passed"
            }],
            "rangeVerified": true
        })
    );

    request["planned"]["strategy"] =
        json!({ "rung": "resident", "engagedRungs": ["resident"], "parameters": {} });
    assert_eq!(
        sdxl_runtime_complete_sweep(&request).unwrap()["axes"],
        json!([])
    );
}

#[test]
fn pinned_sdxl_contract_exposes_the_four_implemented_rungs() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root = fixtures.path("pinned-sdxl-contract-exposes-the-four-implemented-rungs");
    std::fs::create_dir_all(&root).unwrap();
    let spec = fixture_spec(&root);
    let contract = mlx_gen_sdxl::provider_registry()
        .unwrap()
        .memory_strategy_contract(SDXL_PROVIDER, &spec)
        .unwrap()
        .expect("the pinned SDXL provider contract");
    assert_eq!(contract.provider_id, SDXL_PROVIDER);
    assert_eq!(
        contract.calibration.as_ref().unwrap().fingerprint,
        SDXL_CALIBRATION_FINGERPRINT
    );
    for strategy in [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            MemoryStrategySupport::Implemented,
            "{strategy:?}"
        );
    }
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap()
            .support,
        MemoryStrategySupport::Missing,
        "bounded attention must stay withheld"
    );
    let transformer = contract
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .unwrap();
    assert_eq!(
        transformer.parameters.transformer_window_sizes,
        vec![1, 2, 5, 10]
    );
    assert_eq!(
        transformer.parameters.transformer_window_components,
        vec![TransformerComponent::Dit]
    );
    assert_eq!(
        contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
        vec![
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedTransformerResidency,
        ]
    );
    let mut request = minimal_request(SDXL_PROVIDER);
    request["planned"]["strategy"] = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
    });
    let selection = planned_selection(&request).unwrap();
    contract.validate_selection(&selection).unwrap();
    let controls = contract.generation_memory(&selection).unwrap();
    let selected_spec = sdxl_spec_at(SDXL_BASE_ARM, root.clone(), "q4", &selection).unwrap();
    assert_eq!(
        selected_spec.offload_policy == OffloadPolicy::Sequential,
        controls.stage_residency
    );
    assert!(controls.tile_vae_decode);
    assert_eq!(controls.decode_tile_edge, Some(512));
    assert_eq!(controls.decode_overlap, Some(64));
    attested_strategy(
        &request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )
    .unwrap();
    let calibration = contract.calibration.as_ref().unwrap();
    let context = sdxl_context(
        selection,
        calibration,
        &calibration.fingerprint,
        768,
        768,
        1,
        1,
    );
    assert_eq!(context.selection.parameters.decode_tile_edge, Some(512));
    assert_eq!(context.selection.parameters.decode_overlap, Some(64));
    assert!(!context.use_pid);
    assert_eq!(
        sdxl_runtime_complete_sweep(&request).unwrap()["axes"],
        json!([
            { "parameter": "decodeTileEdge", "testedValues": [512] },
            { "parameter": "decodeOverlap", "testedValues": [64] }
        ])
    );
    assert!(matches!(
        &contract.formula,
        mlx_gen::gen_core::MemoryFormulaKind::PhaseEnvelope { variables, .. }
            if variables.contains(&mlx_gen::gen_core::MemoryFormulaVariable::DecodeTileArea)
    ));
    request["planned"]["strategy"] = json!({
        "rung": "bounded_transformer_residency",
        "engagedRungs": ["resident", "staged_residency", "bounded_decode", "bounded_transformer_residency"],
        "parameters": {
            "decodeTileEdge": 512, "decodeOverlap": 64,
            "transformerWindowSize": 5, "transformerWindowComponent": "dit"
        }
    });
    let selection = planned_selection(&request).unwrap();
    validate_sdxl_selection_parameters(&request, &selection).unwrap();
    contract.validate_selection(&selection).unwrap();
    attested_strategy(
        &request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )
    .unwrap();
    let controls = contract.generation_memory(&selection).unwrap();
    let selected_spec = sdxl_spec_at(SDXL_BASE_ARM, root.clone(), "q4", &selection).unwrap();
    assert_eq!(
        selected_spec.offload_policy == OffloadPolicy::Sequential,
        controls.stage_residency
    );
    assert!(
        controls.stage_residency && controls.tile_vae_decode && controls.stream_transformer_blocks
    );
    assert_eq!(controls.decode_tile_edge, Some(512));
    assert_eq!(controls.decode_overlap, Some(64));
    assert_eq!(controls.transformer_window_size, Some(5));
    std::fs::copy(
        root.join("vae/model.safetensors"),
        root.join("vae/diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
    let fixed_spec = spec.with_component(
        mlx_gen_sdxl::loader::VAE_FP16_FIX_COMPONENT,
        WeightsSource::Dir(root.join("vae")),
    );
    let fixed = mlx_gen_sdxl::provider_registry()
        .unwrap()
        .memory_strategy_contract(SDXL_PROVIDER, &fixed_spec)
        .unwrap()
        .unwrap();
    assert_eq!(
        fixed.calibration.as_ref().unwrap().fingerprint,
        mlx_gen_sdxl::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT_VAE_FP16
    );
    assert_ne!(fixed.calibration, contract.calibration);
}
