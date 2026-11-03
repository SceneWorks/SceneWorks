use super::apply_measured_mlx_load_shape;
use super::apply_measured_mlx_load_shape_for_request;
use super::super::LoadSpec;
use super::super::WeightsSource;
use crate::Value;
use gen_core::{
    AdapterKind, AdapterSpec, MemoryContractSurfaceTier, MemoryProviderContract, MemoryStrategy,
    MemoryStrategySupport, OffloadPolicy, Quant, TransformerComponent,
};

fn fixture_spec(root: &std::path::Path, quant_bits: Option<u8>) -> LoadSpec {
    for component in ["text_encoder", "transformer", "vae"] {
        let dir = root.join(component);
        std::fs::create_dir_all(&dir).unwrap();
        // A minimal VALID safetensors file carrying one f32 scalar: the asset-facts
        // unification (inference PR #395) parses headers — and requires at least one tensor —
        // when the provider contract is built, so a zero-filled stub now fails the lookup.
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0f32.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
    }
    std::fs::write(
        root.join("text_encoder/config.json"),
        quant_bits.map_or_else(
            || r#"{"dtype":"bfloat16"}"#.to_owned(),
            |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("transformer/config.json"),
        quant_bits.map_or_else(
            || r#"{"dtype":"bfloat16"}"#.to_owned(),
            |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        ),
    )
    .unwrap();
    LoadSpec::new(WeightsSource::Dir(root.to_owned()))
}

/// Resolve the provider's shipped declaration fixture without traversing model assets.
///
/// Production contract construction deliberately seals every selected encoder file. These
/// tests grade route declaration and capability shape, not file identity, so hashing sparse
/// production-sized encoder fixtures both duplicates the ArtifactSeal tests and turns a pure
/// contract check into minutes of I/O. The registry-owned surface resolver is the exact
/// weights-free seam for an already-resolved artifact tier. Route-local fields are copied from
/// the production-shaped spec so adapters and resolved catalog identity remain under test.
fn weights_free_contract(
    provider_id: &str,
    tier: MemoryContractSurfaceTier,
    spec: &LoadSpec,
) -> MemoryProviderContract {
    let registry = crate::inference_runtime::media();
    let fixture = registry
        .memory_contract_fixture_registrations()
        .find(|fixture| fixture.provider_id == provider_id)
        .unwrap_or_else(|| panic!("{provider_id} weights-free contract fixture"));
    let mut surface = (fixture.surface_specs)()
        .into_iter()
        .find(|surface| {
            surface.selector.tier == tier
                && surface.selector.offload_policy == spec.offload_policy
                && surface.selector.load_shape == spec.load_shape
        })
        .unwrap_or_else(|| {
            panic!(
                "{provider_id} weights-free {:?}/{:?}/{:?} surface",
                tier, spec.offload_policy, spec.load_shape
            )
        });
    surface.spec.resolved_route = spec.resolved_route.clone();
    surface.spec.adapters = spec.adapters.clone();
    surface.spec.load_shape_declaration_result = spec.load_shape_declaration_result;

    match registry
        .memory_contract_surface_resolver_registrations()
        .find(|resolver| resolver.provider_id == provider_id)
    {
        Some(resolver) => (resolver.contract)(&surface),
        None => (fixture.contract)(&surface.spec),
    }
    .unwrap_or_else(|error| panic!("{provider_id} weights-free contract: {error}"))
}

fn rung_four_support(engine_id: &str, spec: &LoadSpec) -> MemoryStrategySupport {
    crate::inference_runtime::media()
        .memory_strategy_contract(engine_id, spec)
        .unwrap()
        .expect("measured provider registers a memory-strategy contract")
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .expect("compatibility contract contains rung 4")
        .support
        .clone()
}

#[test]
fn worker_lens_specs_reach_only_their_exact_measured_contracts() {
    let bf16_dir = tempfile::tempdir().unwrap();
    let bf16 = fixture_spec(bf16_dir.path(), None);
    let q4_dir = tempfile::tempdir().unwrap();
    let q4 = fixture_spec(q4_dir.path(), Some(4)).with_quant(Quant::Q4);

    let turbo = apply_measured_mlx_load_shape("lens_turbo", bf16.clone());
    assert_eq!(
        turbo.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
    // sc-18605 (inference) declared the Lens and Lens-Turbo MLX rung-4 ladders and made them
    // reachable, so the dense Lens-Turbo rung no longer needs `Sequential` to appear. It is
    // Implemented at the deferred load shape alone; the Sequential case below still holds and
    // is kept because it pins the rung's window parameters, not merely its presence.
    assert_eq!(
        rung_four_support("lens_turbo", &turbo),
        MemoryStrategySupport::Implemented,
        "the Lens-Turbo rung-4 ladder is reachable at the deferred load shape"
    );
    let turbo = turbo.with_offload_policy(OffloadPolicy::Sequential);
    let turbo_contract = crate::inference_runtime::media()
        .memory_strategy_contract("lens_turbo", &turbo)
        .unwrap()
        .expect("Lens-Turbo contract");
    let turbo_rung = turbo_contract
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .expect("Lens-Turbo rung 4");
    assert_eq!(turbo_rung.support, MemoryStrategySupport::Implemented);
    assert_eq!(turbo_rung.parameters.transformer_window_sizes, vec![1]);
    assert_eq!(
        turbo_rung.parameters.transformer_window_components,
        vec![TransformerComponent::TextEncoder]
    );

    let lens = apply_measured_mlx_load_shape("lens", q4.clone());
    assert_eq!(
        lens.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    );
    let lens_contract = crate::inference_runtime::media()
        .memory_strategy_contract("lens", &lens)
        .unwrap()
        .expect("Lens contract");
    let lens_rung = lens_contract
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .expect("Lens rung 4");
    assert_eq!(lens_rung.support, MemoryStrategySupport::Implemented);
    assert_eq!(lens_rung.parameters.transformer_window_sizes, vec![1]);
    assert_eq!(
        lens_rung.parameters.transformer_window_components,
        vec![TransformerComponent::Both]
    );

    let adapter = AdapterSpec::new(
        q4_dir.path().join("adapter.safetensors"),
        1.0,
        AdapterKind::Lora,
    );
    let unsupported = [
        ("lens", bf16.clone()),
        ("lens_turbo", q4),
        ("lens", bf16.clone().with_quant(Quant::Q8)),
        ("lens", bf16.clone().with_adapters(vec![adapter])),
        (
            "lens",
            bf16.with_pid(
                WeightsSource::File(q4_dir.path().join("pid.safetensors")),
                WeightsSource::Dir(q4_dir.path().join("gemma")),
            ),
        ),
    ];
    for (engine_id, spec) in unsupported {
        let shaped = apply_measured_mlx_load_shape(engine_id, spec)
            .with_offload_policy(OffloadPolicy::Sequential);
        assert_eq!(
            shaped.load_shape,
            gen_core::LoadShape::EagerMaterialization,
            "unmeasured Lens entry/tier/overlay combinations must remain eager"
        );
        assert_eq!(
            rung_four_support(engine_id, &shaped),
            MemoryStrategySupport::Missing
        );
    }
}

#[test]
fn worker_qwen_specs_reach_the_shared_base_edit_and_lightning_contract() {
    let bf16_root = std::path::PathBuf::from("/__sceneworks_qwen_bf16_contract_fixture__");
    let bf16 = LoadSpec::new(WeightsSource::Dir(bf16_root.clone()));
    let q4_root = std::path::PathBuf::from("/__sceneworks_qwen_q4_contract_fixture__");
    let q4 = LoadSpec::new(WeightsSource::Dir(q4_root.clone())).with_quant(Quant::Q4);
    let lightning_adapter = AdapterSpec::new(
        q4_root.join("lightning.safetensors"),
        1.0,
        AdapterKind::Lora,
    );

    for (label, engine_id, tier, spec) in [
        (
            "qwen_image",
            "qwen_image",
            MemoryContractSurfaceTier::Bf16,
            bf16.clone(),
        ),
        (
            "qwen_image_edit_2511",
            "qwen_image_edit",
            MemoryContractSurfaceTier::Q4,
            q4.clone(),
        ),
        (
            "qwen_image_edit_2511_lightning",
            "qwen_image_edit",
            MemoryContractSurfaceTier::Q4,
            q4.with_adapters(vec![lightning_adapter]),
        ),
    ] {
        let shaped = apply_measured_mlx_load_shape(engine_id, spec);
        assert_eq!(
            shaped.load_shape,
            gen_core::LoadShape::DeferredMaterialization,
            "{label} must resolve to the shared Qwen provider load shape"
        );
        let sequential = shaped.with_offload_policy(OffloadPolicy::Sequential);
        let contract = weights_free_contract(engine_id, tier, &sequential);
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4");
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(rung.parameters.transformer_window_sizes, vec![1]);
        assert_eq!(
            rung.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }

    let pid = bf16.clone().with_pid(
        WeightsSource::File(bf16_root.join("pid.safetensors")),
        WeightsSource::Dir(bf16_root.join("gemma")),
    );
    assert_eq!(
        apply_measured_mlx_load_shape("qwen_image", pid).load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "the native-Qwen-VAE ladder must not opt PiD into its load shape"
    );
    assert_eq!(
        apply_measured_mlx_load_shape("qwen_image_control", bf16).load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "the unbounded five-block control side branch is not advertised as rung 4"
    );
}

#[test]
fn worker_krea_declarations_reach_the_full_ladder_without_legacy_shaping() {
    let source = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin model manifest")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(source))
            .expect("builtin model manifest parses");
    let turbo_manifest = manifest["models"]
        .as_array()
        .expect("model rows")
        .iter()
        .find(|model| model["id"] == "krea_2_turbo")
        .and_then(Value::as_object)
        .expect("Krea Turbo row");
    let context = crate::memory_route_registry::MemoryRouteRequestContext {
        mode: crate::memory_route_registry::MemoryRouteMode::TextToImage,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    };

    for (tier, quant_bits, surface_tier) in [
        ("bf16", None, MemoryContractSurfaceTier::Bf16),
        ("q4", Some(4), MemoryContractSurfaceTier::Q4),
        ("q8", Some(8), MemoryContractSurfaceTier::Q8),
    ] {
        // Declaration evaluation reads the tiny transformer marker; provider capability
        // construction below remains weights-free and never seals the encoder payload.
        let root = tempfile::tempdir().unwrap();
        let spec = fixture_spec(root.path(), quant_bits).with_resolved_route("krea_2_turbo");
        assert_eq!(spec.quantize, None, "{tier} is a prepacked artifact tier");
        let shaped =
            crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request_with(
                "krea_2_turbo",
                Some(tier),
                Some(context.mode),
                turbo_manifest,
                spec,
                context,
                |candidate| {
                    weights_free_contract("krea_2_turbo", surface_tier, candidate)
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .is_some_and(|capability| {
                            capability.support == MemoryStrategySupport::Implemented
                        })
                },
            );
        assert_eq!(
            shaped.load_shape,
            gen_core::LoadShape::DeferredMaterialization,
            "plain Krea {tier} must use the production deferred load shape"
        );
        let resident = shaped.clone().with_offload_policy(OffloadPolicy::Resident);
        let resident_contract = weights_free_contract("krea_2_turbo", surface_tier, &resident);
        assert_eq!(
            resident_contract
                .capability(MemoryStrategy::Resident)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "plain Krea {tier} must remain reachable on a roomy resident host"
        );

        let sequential = shaped.with_offload_policy(OffloadPolicy::Sequential);
        let sequential_contract = weights_free_contract("krea_2_turbo", surface_tier, &sequential);
        let rung = sequential_contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("Krea compatibility contract contains rung 4");
        assert_eq!(rung.support, MemoryStrategySupport::Implemented, "{tier}");
        assert_eq!(rung.parameters.transformer_window_sizes, vec![1]);
        assert_eq!(
            rung.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }

    let root = tempfile::tempdir().unwrap();
    let base = fixture_spec(root.path(), Some(4)).with_resolved_route("krea_2_turbo");
    assert_eq!(
        crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request_with(
            "krea_2_turbo",
            Some("q4"),
            Some(crate::memory_route_registry::MemoryRouteMode::EditImage),
            turbo_manifest,
            base.clone(),
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode: crate::memory_route_registry::MemoryRouteMode::EditImage,
                reference_count: 1,
                use_pid: false,
                has_phases: false,
            },
            |_| true,
        )
        .load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "native Turbo cannot consume the route-local edit declaration"
    );
    let control = base.with_control(WeightsSource::File(root.path().join("control.safetensors")));
    let refused = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request_with(
        "krea_2_turbo",
        Some("q4"),
        Some(context.mode),
        turbo_manifest,
        control,
        context,
        |_| true,
    );
    assert_eq!(
        refused.load_shape,
        gen_core::LoadShape::EagerMaterialization
    );
    assert_eq!(
        refused.load_shape_declaration_result,
        gen_core::LoadShapeDeclarationResult::Refused,
        "control cannot fall through to the former Turbo legacy shaper"
    );
}

#[test]
fn worker_plain_sdxl_specs_reach_the_pinned_four_rung_contract() {
    fn sdxl_spec(root: &std::path::Path, quant_bits: Option<u8>, quant: Option<Quant>) -> LoadSpec {
        for component in ["text_encoder", "text_encoder_2", "unet", "vae"] {
            let directory = root.join(component);
            std::fs::create_dir_all(&directory).unwrap();
            let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&0_f32.to_le_bytes());
            std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
        }
        if let Some(bits) = quant_bits {
            std::fs::write(
                root.join("unet").join("config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        spec
    }

    for (tier, quant_bits, quant) in [
        ("bf16", None, None),
        ("q4", Some(4), Some(Quant::Q4)),
        ("q8", Some(8), Some(Quant::Q8)),
    ] {
        let root = tempfile::tempdir().unwrap();
        let shaped = apply_measured_mlx_load_shape_for_request(
            "sdxl",
            sdxl_spec(root.path(), quant_bits, quant),
            true,
        );
        assert_eq!(
            shaped.load_shape,
            gen_core::LoadShape::DeferredMaterialization,
            "plain SDXL {tier} must use the production deferred load shape"
        );
        let contract = crate::inference_runtime::media()
            .memory_strategy_contract(
                "sdxl",
                &shaped.with_offload_policy(OffloadPolicy::Sequential),
            )
            .unwrap()
            .expect("plain SDXL must resolve its pinned provider contract before GPU work");
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "{tier}"
        );
        for implemented in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                contract.capability(implemented).unwrap().support,
                MemoryStrategySupport::Implemented,
                "SDXL {tier} must expose its pinned v5 rung {implemented:?}"
            );
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "SDXL {tier} must not invent bounded attention"
        );
        let native_decode = &contract.pid_decode_routes.as_ref().unwrap().native;
        assert_eq!(
            native_decode.tile_edges,
            [896, 768, 640, 512, 448, 384, 320, 256]
        );
        assert_eq!(native_decode.tile_overlap, 64);
        let mut decode = gen_core::MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: gen_core::MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant,
                component_precision_floors: &[],
            },
        };
        contract
            .validate_selection(&decode)
            .expect("native decode selection");
        let memory = contract
            .generation_memory(&decode)
            .expect("implemented bounded decode configures request memory");
        assert!(memory.tile_vae_decode);
        assert!(!memory.stage_residency);
        assert!(!memory.stream_transformer_blocks);
        decode.parameters.decode_tile_edge = Some(128);
        assert!(
            contract.validate_selection(&decode).is_err(),
            "unpublished decode geometry"
        );
    }

    let root = tempfile::tempdir().unwrap();
    let base = sdxl_spec(root.path(), Some(4), Some(Quant::Q4));
    // sc-21609: edit-routed requests now share the deferred production shape — the Mlx sdxl
    // registry row covers edit_image, matching the advanced lane the worker actually routes.
    assert_eq!(
        apply_measured_mlx_load_shape_for_request("sdxl", base.clone(), false).load_shape,
        gen_core::LoadShape::DeferredMaterialization,
        "SDXL edit requests share the production deferred load shape"
    );
    let adapter = AdapterSpec::new(
        root.path().join("adapter.safetensors"),
        1.0,
        AdapterKind::Lora,
    );
    // sc-21609: a LoRA load defers — the widened row covers the lora profile the worker routes
    // (and the measured rung-4 evidence was taken with LoRA armed) — while a bare control
    // component at text_to_image still finds no row: the tile-ControlNet profile is scoped to
    // the image_detail lane, so this coordinate stays eager rather than borrowing a shape no
    // production request carries.
    assert_eq!(
        apply_measured_mlx_load_shape_for_request(
            "sdxl",
            base.clone().with_adapters(vec![adapter]),
            true
        )
        .load_shape,
        gen_core::LoadShape::DeferredMaterialization,
        "SDXL LoRA loads share the production deferred load shape"
    );
    assert_eq!(
        apply_measured_mlx_load_shape_for_request(
            "sdxl",
            base.with_control(WeightsSource::File(root.path().join("control.safetensors"))),
            true
        )
        .load_shape,
        gen_core::LoadShape::EagerMaterialization,
        "a control load at text_to_image matches no routed lane and must stay eager"
    );
}
