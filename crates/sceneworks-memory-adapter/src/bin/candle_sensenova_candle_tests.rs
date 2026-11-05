use super::*;

fn sensenova_planned(provider: &str, model_id: &str, tier: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "candle",
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

/// The member is resolved from the PAIR, every per-member field is unique across the family,
/// and an unserved id or a crossed quality/fast pair is refused by name.
#[test]
fn sensenova_candle_arm_is_resolved_from_the_plans_provider_and_model_id() {
    for arm in SENSENOVA_CANDLE_FAMILY {
        assert_eq!(
            sensenova_candle_arm(&sensenova_planned(arm.provider, arm.model_id, "q4")).unwrap(),
            arm
        );
    }
    for field in [
        SENSENOVA_CANDLE_FAMILY.map(|arm| arm.model_id).to_vec(),
        SENSENOVA_CANDLE_FAMILY.map(|arm| arm.slug).to_vec(),
        SENSENOVA_CANDLE_FAMILY
            .map(|arm| arm.repository_env)
            .to_vec(),
        SENSENOVA_CANDLE_FAMILY.map(|arm| arm.revision_env).to_vec(),
        SENSENOVA_CANDLE_FAMILY.map(|arm| arm.root_env).to_vec(),
        SENSENOVA_CANDLE_FAMILY
            .map(|arm| arm.expected_repository)
            .to_vec(),
        SENSENOVA_CANDLE_FAMILY
            .map(|arm| arm.execution_path)
            .to_vec(),
        SENSENOVA_CANDLE_FAMILY
            .map(|arm| arm.still_calibration)
            .to_vec(),
    ] {
        let mut unique = field.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), field.len(), "collision in {field:?}");
    }
    for absent in ["sensenova_u1_8b_infographic_v4", "z_image", "qwen_image"] {
        let error = sensenova_candle_arm(&sensenova_planned(SENSENOVA_ID, absent, "q4"))
            .expect_err("an unserved model id must be refused");
        assert!(error.contains(&format!("modelId {absent:?}")), "{error}");
    }
    let crossed = sensenova_candle_arm(&sensenova_planned(
        SENSENOVA_FAST_ID,
        "sensenova_u1_8b",
        "q4",
    ))
    .expect_err("a crossed (provider, modelId) pair must be refused");
    assert!(crossed.contains(SENSENOVA_FAST_ID), "{crossed}");
}

/// The spec is the shape the WORKER loads on this lane — resident, eager, the resolved route
/// bound, and the tier's own quant, which `candle-gen-sensenova` REQUIRES (its
/// `validate_numeric_tier` errors when `spec.quantize` disagrees with the checkpoint packing).
#[test]
fn sensenova_candle_spec_binds_the_worker_shape_and_the_planned_tier() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    for arm in SENSENOVA_CANDLE_FAMILY {
        for (tier, expected) in [
            ("bf16", None),
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
        ] {
            let root = PathBuf::from(format!(
                "/cache/models--{}/snapshots/{revision}/{tier}",
                arm.expected_repository.replace('/', "--")
            ));
            let spec = sensenova_candle_spec_at(
                arm,
                arm.expected_repository,
                revision,
                root.clone(),
                tier,
            )
            .unwrap();
            assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(spec.quantize, expected, "{}:{tier}", arm.model_id);
            assert_eq!(spec.resolved_route.as_deref(), Some(arm.model_id));
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == &root));
            assert!(spec.adapters.is_empty(), "SenseNova attaches no adapters");
            assert!(spec.components.is_empty(), "the distill LoRA is pre-merged");
        }
        let root = PathBuf::from(format!(
            "/cache/models--{}/snapshots/{revision}/q4",
            arm.expected_repository.replace('/', "--")
        ));
        assert!(
            sensenova_candle_spec_at(arm, arm.expected_repository, revision, root.clone(), "q8")
                .is_err(),
            "a q4 root must not satisfy a q8 plan"
        );
        assert!(
            sensenova_candle_spec_at(arm, "sensenova/SenseNova-U1-8B-MoT", revision, root, "q4")
                .is_err(),
            "the upstream snapshot must not satisfy the tiered rehost plan"
        );
    }
}

/// Every SenseNova Candle cell the committed plan declares resolves to an implemented member,
/// names the production identity its loaded generator publishes, and holds a distinct identity.
/// The expected set is DERIVED from the family and the shipped tiers, never a count literal.
#[test]
fn every_planned_sensenova_candle_cell_resolves_and_names_its_production_identity() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    let mut identities = std::collections::BTreeMap::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":candle") || !matches!(provider, SENSENOVA_ID | SENSENOVA_FAST_ID) {
            continue;
        }
        seen.insert(key.clone());
        let model_id = key.split(':').next().unwrap();
        let tier = key.split(':').nth(1).unwrap();
        let request = json!({ "planned": {
            "target": {
                "provider": provider,
                "modelId": model_id,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        let arm = sensenova_candle_arm(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        validate_sensenova_candle_fixture(&request, arm, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(
            entry["loadShape"].as_str(),
            Some("eager_materialization"),
            "{key}: the worker loads this family eagerly on Candle too"
        );
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !planned.contains("weights-free"),
            "{key}: {planned} is a weights-free conformance identity no production load returns"
        );
        assert_eq!(
            planned,
            sensenova_candle_calibration_fingerprint(arm, tier),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    let expected: std::collections::BTreeSet<String> = SENSENOVA_CANDLE_FAMILY
        .iter()
        .flat_map(|arm| {
            ["bf16", "q4", "q8"]
                .iter()
                .map(move |tier| format!("{}:{tier}:candle", arm.model_id))
        })
        .collect();
    assert_eq!(
        seen, expected,
        "every shipped SenseNova Candle cell, exactly once"
    );
    assert_eq!(
        identities.len(),
        SENSENOVA_CANDLE_FAMILY.len() * 3,
        "one distinct identity per (route, shipped tier)"
    );
}

/// The two lanes agree on the family: same six model ids, same two engine ids, same slugs and
/// same env families. The tables are separate because the binaries are, so nothing but a test
/// stops them drifting into two different families with one name.
#[test]
fn the_candle_family_table_matches_the_mlx_arms_env_and_repository_bindings() {
    let mlx = include_str!("mlx.rs");
    for arm in SENSENOVA_CANDLE_FAMILY {
        for env in [arm.repository_env, arm.revision_env, arm.root_env] {
            assert!(
                mlx.contains(&format!("\"{env}\"")),
                "{}: the MLX arm never reads {env}",
                arm.model_id
            );
        }
        assert!(
            mlx.contains(&format!("model_id: \"{}\"", arm.model_id)),
            "{}: the MLX family does not carry this member",
            arm.model_id
        );
        assert!(
            mlx.contains(&format!("slug: \"{}\"", arm.slug)),
            "{}: the two lanes disagree on the route slug, so their identities would differ",
            arm.model_id
        );
    }
}

/// The fixture is bound to the member, the tier and the geometry edge.
#[test]
fn sensenova_candle_fixture_is_bound_to_the_member_tier_and_edge() {
    let request = sensenova_planned(SENSENOVA_ID, "sensenova_u1_8b", "q4");
    let arm = sensenova_candle_arm(&request).unwrap();
    let bind = |fixture: &str, tier| {
        validate_sensenova_candle_fixture(
            &json!({ "planned": {
                "fixture": fixture,
                "target": request["planned"]["target"].clone(),
            }}),
            arm,
            tier,
        )
    };
    assert!(bind("quality-candle-q4-1024-seed22734-step2", "q4").is_ok());
    for (fixture, expected) in [
        (
            "infographic-v2-candle-q4-1024-seed22734-step2",
            "must start with",
        ),
        ("quality-candle-q8-1024-seed22734-step2", "must start with"),
        ("quality-mlx-q4-1024-seed22734-step2", "must start with"),
        ("quality-candle-q4-1024-seed1-step2", "does not match"),
        ("quality-candle-q4-1024-seed22734-step4", "two-step"),
    ] {
        let error = bind(fixture, "q4").expect_err("the fixture must be bound to its cell");
        assert!(error.contains(expected), "{fixture}: {error}");
    }
}
