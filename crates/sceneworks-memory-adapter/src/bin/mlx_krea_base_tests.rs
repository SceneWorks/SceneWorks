use super::*;
use mlx_gen::gen_core::MemoryStrategySupport;

fn minimal_request(provider: &str, rung: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                // `modelId` tracks the provider so a member-resolution test is not silently
                // answering the `requires modelId` clause instead (sc-22735).
                "modelId": provider,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 }
            },
            "strategy": { "rung": rung, "engagedRungs": ["resident"], "parameters": {} }
        }
    })
}

/// A fully-formed plan row for one member at one tier, matching the checked-in plan's shape.
fn planned_row(arm: KreaBaseArm, tier: &str) -> Value {
    let mut request = minimal_request(arm.provider, "resident");
    request["planned"]["target"]["tier"] = json!(tier);
    request["planned"]["loadShape"] = json!(load_shape_key(arm.load_shape));
    request["planned"]["fixture"] = json!(format!(
        "{}-{tier}-768-seed{}-step2",
        arm.fixture_stem, arm.seed
    ));
    request["planned"]["calibrationFingerprint"] = json!(krea_base_pinned_fingerprint(arm, tier)
        .expect("both members resolve a pinned fingerprint at every numeric tier"));
    request
}

fn fixture_spec(root: &std::path::Path) -> LoadSpec {
    let encoder_contract = mlx_gen_krea::provider_registry()
        .unwrap()
        .provider_encoder_contract(KREA_BASE_PROVIDER)
        .expect("the pinned Krea base encoder contract");
    gen_core_testkit::write_encoder_contract_fixture_with_quant(
        &root.join("text_encoder"),
        encoder_contract,
        Some(4),
    )
    .expect("registry-owned Krea text encoder fixture");
    for component in ["transformer", "vae"] {
        let directory = root.join(component);
        std::fs::create_dir_all(&directory).unwrap();
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        std::fs::write(directory.join("model.safetensors"), bytes).unwrap();
    }
    std::fs::write(
        root.join("transformer").join("config.json"),
        r#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .unwrap();
    LoadSpec::new(WeightsSource::Dir(root.to_owned()))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization)
}

#[test]
fn the_base_arm_refuses_a_foreign_provider_before_environment_or_weight_work() {
    // sc-22735: the refusal moved from the (then only) member's own label onto the FAMILY
    // resolver, because with two members "which member" is a question that must be answered
    // before any member-specific validation runs. The guarantee is unchanged and strengthened
    // below: the provider is named, and nothing about the environment or a contract is
    // mentioned, so no env read or path work happened first.
    for provider in [
        "krea_2_turbo_edit",
        "krea_2_turbo_control",
        "qwen_image",
        "krea_2_edit",
    ] {
        let error = run_krea_base(&minimal_request(provider, "resident"))
            .expect_err("a foreign provider must not reach the Krea base arm");
        assert_eq!(
            error,
            format!("the MLX Krea base arm does not implement provider {provider:?}")
        );
        for leaked in ["SCENEWORKS_", "contract", "canonicalize", "fingerprint"] {
            assert!(
                !error.contains(leaked),
                "the refusal must precede every environment and contract read, saw \
                     {leaked:?} in {error}"
            );
        }
    }
}

/// Both shipped members resolve to their own row, and every field that differentiates them is
/// the one the plan and the shipped worker require.
#[test]
fn each_shipped_provider_resolves_its_own_family_member() {
    let turbo = krea_base_arm(&minimal_request(KREA_BASE_PROVIDER, "resident")).unwrap();
    assert_eq!(turbo, KREA_TURBO_ARM);
    let raw = krea_base_arm(&minimal_request(KREA_RAW_PROVIDER, "resident")).unwrap();
    assert_eq!(raw, KREA_RAW_ARM);

    assert_eq!(turbo.provider, "krea_2_turbo");
    assert_eq!(turbo.load_shape, LoadShape::DeferredMaterialization);
    assert!(turbo.load_spec_carries_quant);
    assert_eq!(turbo.expected_repository, protocol::KREA_REPOSITORY);
    assert_eq!(turbo.seed, KREA_BASE_SEED);

    assert_eq!(raw.provider, "krea_2_raw");
    // The Raw plan rows declare EAGER; Turbo's declare deferred. Neither is a constant.
    assert_eq!(raw.load_shape, LoadShape::EagerMaterialization);
    // THE POINT OF THE STORY: the shipped worker passes `quantize = None` for the Krea image
    // engines at every tier (`image_jobs/base.rs::mlx_load_quant_for_resolved_artifact`).
    assert!(!raw.load_spec_carries_quant);
    assert_eq!(raw.expected_repository, protocol::KREA_RAW_REPOSITORY);
    assert_eq!(raw.seed, 22735);
    assert_ne!(turbo.slug, raw.slug);
    assert_ne!(turbo.execution_path, raw.execution_path);
    assert_ne!(turbo.repository_env, raw.repository_env);
}

/// The three literal environment-variable names `measure-memory-catalog.mjs` exports for the
/// Raw artifact family, read back from the arm table rather than restated.
#[test]
fn the_raw_member_reads_the_catalogs_own_environment_family() {
    assert_eq!(
        KREA_RAW_ARM.repository_env,
        "SCENEWORKS_KREA_RAW_REPOSITORY"
    );
    assert_eq!(KREA_RAW_ARM.revision_env, "SCENEWORKS_KREA_RAW_REVISION");
    assert_eq!(KREA_RAW_ARM.root_env, "SCENEWORKS_KREA_RAW_ROOT");
}

/// A plan row that names the OTHER member's load shape is refused by name, at both members —
/// the accepted shape is a property of the arm, not a hardcoded `deferred_materialization`.
#[test]
fn each_member_refuses_the_other_members_load_shape() {
    for (arm, wrong) in [
        (KREA_TURBO_ARM, protocol::LOAD_SHAPE_EAGER),
        (KREA_RAW_ARM, protocol::LOAD_SHAPE_DEFERRED),
    ] {
        let mut request = planned_row(arm, "q4");
        request["planned"]["loadShape"] = json!(wrong);
        let error = run_krea_base(&request).expect_err("the wrong load shape must be refused");
        assert!(
            error.contains(&format!(
                "must use the production {}",
                load_shape_key(arm.load_shape)
            )) && error.contains(wrong),
            "{}: {error}",
            arm.provider
        );
    }
}

/// The tier-exact snapshot root every member binds against, built the way a real export lays
/// one out.
fn snapshot_root(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
    _tag: &str,
) -> PathBuf {
    fixtures.snapshot(repository, revision, Some(tier))
}

/// The Raw load spec carries NO quant at ANY tier, and still binds the tier-exact root — driven
/// env-free through [`krea_base_spec_at`], the way `z_image_load_spec_at` is unit-tested.
#[test]
fn the_raw_load_spec_carries_no_quant_and_binds_the_planned_tier() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let repository = "SceneWorks/krea-2-raw-mlx";
    let revision = "a".repeat(40);
    for tier in ["bf16", "q4", "q8"] {
        let root = snapshot_root(fixtures, repository, &revision, tier, "raw");
        let selection = planned_selection(&planned_row(KREA_RAW_ARM, tier)).unwrap();
        let spec = krea_base_spec_at(
            KREA_RAW_ARM,
            tier,
            repository,
            &revision,
            root.clone(),
            &selection,
        )
        .expect("the tier-exact Raw root resolves");
        assert!(
            spec.quantize.is_none(),
            "{tier}: the worker passes no LoadSpec quant for krea_2_raw at any tier"
        );
        assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);

        // A stale export of ANOTHER tier can never satisfy this plan row — and the refusal has
        // to SAY which tier it wanted. sc-22735 fix pass: a bare `expect_err` here passed on any
        // error at all, so a refusal that stopped naming the tier (or that started refusing for
        // an unrelated reason) stayed green and left the operator holding "it failed". The
        // message `validate_snapshot_suffix` builds is the whole expected suffix, so assert the
        // planned tier appears in it AND that it appears as the `snapshots/<revision>/<tier>`
        // tail rather than incidentally anywhere in the string.
        for stale in ["bf16", "q4", "q8"] {
            if stale == tier {
                continue;
            }
            let stale_root = root.parent().unwrap().join(stale);
            std::fs::create_dir_all(&stale_root).unwrap();
            let error = krea_base_spec_at(
                KREA_RAW_ARM,
                tier,
                repository,
                &revision,
                stale_root,
                &selection,
            )
            .expect_err(&format!(
                "a .../{stale} root must not satisfy a {tier} plan"
            ));
            assert!(
                error.contains(&format!("snapshots/{revision}/{tier}")),
                "the {tier} refusal against a .../{stale} root must name the tier-exact \
                     snapshot suffix it required, got {error:?}"
            );
        }
    }
}

/// Turbo's spec is untouched by the generalization: its already-priced cells were measured with
/// the planned tier's quant on the spec, and this story does not re-measure them.
#[test]
fn the_turbo_load_spec_still_carries_its_tier_quant() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let repository = "SceneWorks/krea-2-turbo-mlx";
    let revision = "b".repeat(40);
    for (tier, expected) in [
        ("bf16", None),
        ("q4", Some(Quant::Q4)),
        ("q8", Some(Quant::Q8)),
    ] {
        let root = snapshot_root(fixtures, repository, &revision, tier, "turbo");
        let selection = planned_selection(&planned_row(KREA_TURBO_ARM, tier)).unwrap();
        let spec = krea_base_spec_at(
            KREA_TURBO_ARM,
            tier,
            repository,
            &revision,
            root.clone(),
            &selection,
        )
        .unwrap();
        assert_eq!(spec.quantize, expected, "{tier}");
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
    }
}

#[test]
fn the_base_arm_fails_closed_on_a_non_plain_target_before_weight_work() {
    for (pointer, value, expected) in [
        (
            "/planned/target/modelId",
            json!("krea_2_turbo_edit"),
            "requires modelId",
        ),
        (
            "/planned/target/mode",
            json!("edit_image"),
            "requires reference-free text_to_image mode",
        ),
        (
            "/planned/target/geometry/batch",
            json!(2),
            "requires geometry.batch == 1",
        ),
        (
            "/planned/target/geometry/frames",
            json!(2),
            "requires geometry.frames == 1",
        ),
    ] {
        let mut request = minimal_request(KREA_BASE_PROVIDER, "resident");
        *request.pointer_mut(pointer).unwrap() = value;
        let error = run_krea_base(&request)
            .expect_err("a non-plain Krea target must fail before environment or weights");
        assert!(error.contains(expected), "{pointer}: {error}");
    }

    for (field, value) in [("referenceCount", json!(1)), ("hasReference", json!(true))] {
        let mut request = minimal_request(KREA_BASE_PROVIDER, "resident");
        request["planned"]["target"][field] = value;
        let error = run_krea_base(&request)
            .expect_err("a referenced Krea target must fail before environment or weights");
        assert!(error.contains(field), "{field}: {error}");
    }
}

#[test]
fn complete_sweep_attests_only_the_exact_executed_krea_parameters() {
    let request = json!({
        "planned": {
            "strategy": {
                "parameters": {
                    "decodeTileEdge": 512,
                    "decodeOverlap": 64,
                    "attentionChunkSize": 67_108_864,
                    "transformerWindowSize": 1
                }
            }
        }
    });
    let sweep = krea_base_complete_sweep(&request).unwrap();
    assert_eq!(sweep["rangeVerified"], true);
    assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
    assert_eq!(
        sweep["cases"][0]["parameters"],
        request["planned"]["strategy"]["parameters"]
    );
    for (parameter, expected) in [
        ("decodeTileEdge", json!([512])),
        ("decodeOverlap", json!([64])),
        ("attentionChunkSize", json!([67_108_864])),
        ("transformerWindowSize", json!([1])),
    ] {
        assert!(sweep["axes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|axis| { axis["parameter"] == parameter && axis["testedValues"] == expected }));
    }
}

#[test]
fn base_admission_context_is_reference_free_text_to_image() {
    let calibration = MemoryCalibrationIdentity::new(
        mlx_production_fingerprint(KREA_BASE_PROVIDER, "q4")
            .expect("krea_2_turbo publishes a production identity at q4"),
        LoadShape::DeferredMaterialization,
    );
    let context = krea_base_context(
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
fn pinned_base_contract_exposes_the_exact_declared_full_ladder() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root = fixtures.path("pinned-base-contract-exposes-the-exact-declared-full-ladder");
    std::fs::create_dir_all(&root).unwrap();
    let spec = fixture_spec(&root);
    let contract = mlx_gen_krea::provider_registry()
        .unwrap()
        .memory_strategy_contract(KREA_BASE_PROVIDER, &spec)
        .unwrap()
        .expect("the pinned Krea base provider contract");
    assert_eq!(contract.provider_id, KREA_BASE_PROVIDER);
    // The fixture packs the transformer at q4, so the pinned provider names the turbo q4 cell —
    // never the retired string every tier used to share.
    assert_eq!(
        contract.calibration.as_ref().unwrap().fingerprint,
        mlx_production_fingerprint(KREA_BASE_PROVIDER, "q4").unwrap()
    );
    assert_ne!(
        contract.calibration.as_ref().unwrap().fingerprint,
        RETIRED_KREA_BASE_CALIBRATION_FINGERPRINT
    );
    for strategy in [
        MemoryStrategy::Resident,
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            MemoryStrategySupport::Implemented,
            "{strategy:?}"
        );
    }
    let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
    assert!(decode.parameters.decode_tile_edges.contains(&512));
    assert!(decode.parameters.decode_overlaps.contains(&64));
    let routes = contract
        .pid_decode_routes
        .as_ref()
        .expect("Krea distinguishes native and PiD decode domains");
    assert_eq!(routes.native.tile_edges, vec![512]);
    assert_eq!(routes.native.tile_overlap, 64);
    let attention = contract
        .capability(MemoryStrategy::BoundedAttention)
        .unwrap();
    assert_eq!(attention.parameters.attention_chunk_sizes, vec![67_108_864]);
    let transformer = contract
        .capability(MemoryStrategy::BoundedTransformerResidency)
        .unwrap();
    assert_eq!(transformer.parameters.transformer_window_sizes, vec![1]);
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
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
    );
}
