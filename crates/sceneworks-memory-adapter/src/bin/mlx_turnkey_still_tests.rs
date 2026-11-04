use super::*;

const MEMBERS: [(&str, &str); 7] = [
    (KOLORS_PROVIDER, "kolors"),
    (IDEOGRAM_PROVIDER, "ideogram-4"),
    (IDEOGRAM_TURBO_PROVIDER, "ideogram-4-turbo"),
    (LENS_PROVIDER, "lens"),
    (LENS_TURBO_PROVIDER, "lens-turbo"),
    (QWEN_IMAGE_2_1_PROVIDER, "qwen-image-2-1"),
    (QWEN_IMAGE_2_1_TURBO_PROVIDER, "qwen-image-2-1-turbo"),
];

fn turnkey_planned(provider: &str, mode: &str, tier: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": mode,
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

fn turnkey_snapshot_root(repository: &str, revision: &str, tier: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir()
        .join(format!("sc-22732-{}-{nonce}", std::process::id()))
        .join(format!("models--{}", repository.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join(tier);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// The root a family's `tier` loads from: `<snapshot>/<tier>` for a tiered family, and the
/// snapshot ROOT for Qwen-Image 2.1's untiered bf16 leg (sc-24112).
fn turnkey_family_root(family: TurnkeyFamily, revision: &str, tier: &str) -> PathBuf {
    let root = turnkey_snapshot_root(family.expected_repository, revision, tier);
    if family.tiered {
        root
    } else {
        root.parent().unwrap().to_path_buf()
    }
}

/// The member is read off the plan's `(provider, mode)`, and a pair no member serves is refused
/// BY NAME rather than measured as its nearest neighbour. Kolors advertises `edit_image` and
/// `character_image`, and both Ideogram members advertise `edit_image` and `image_inpaint` —
/// real routes with their own admission contexts, none of which is this arm's plain
/// text-to-image measurement.
#[test]
fn the_turnkey_arm_resolves_the_member_and_refuses_every_pair_no_member_serves() {
    for (provider, slug) in MEMBERS {
        let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", "q4")).unwrap();
        assert_eq!(arm.provider, provider);
        assert_eq!(arm.slug, slug);
        for mode in [
            "edit_image",
            "character_image",
            "image_inpaint",
            "text_to_video",
        ] {
            let error = turnkey_arm(&turnkey_planned(provider, mode, "q4"))
                .expect_err("a mode no member serves must be refused");
            assert_eq!(
                error,
                format!(
                    "the MLX turnkey still arm does not implement provider {provider:?} in \
                         mode {mode:?}"
                )
            );
        }
    }
    let error = turnkey_arm(&turnkey_planned("sana_1600m", "text_to_image", "q4"))
        .expect_err("a provider no member serves must be refused");
    assert!(error.contains("\"sana_1600m\""), "{error}");
}

/// Every member binds ITS OWN artifact family, and Ideogram binds a SECOND one for bf16. A
/// wrong binding still produces a well-formed record; the only place it would ever show is the
/// repository and revision inside the loadability fingerprint, so it is asserted here.
#[test]
fn each_member_binds_its_own_artifact_family_and_ideogram_splits_bf16() {
    let expect = |arm: TurnkeyArm, tier: &str, repository: &str, env: &str| {
        let family = arm.family_for(tier);
        assert_eq!(family.expected_repository, repository, "{tier}");
        assert_eq!(
            family.repository_env,
            format!("SCENEWORKS_{env}_REPOSITORY")
        );
        assert_eq!(family.revision_env, format!("SCENEWORKS_{env}_REVISION"));
        assert_eq!(family.root_env, format!("SCENEWORKS_{env}_ROOT"));
    };
    for tier in ["q4", "q8", "bf16"] {
        expect(KOLORS_ARM, tier, protocol::KOLORS_REPOSITORY, "KOLORS");
        expect(LENS_ARM, tier, protocol::LENS_REPOSITORY, "LENS");
        expect(
            LENS_TURBO_ARM,
            tier,
            protocol::LENS_TURBO_REPOSITORY,
            "LENS_TURBO",
        );
    }
    for arm in [IDEOGRAM_ARM, IDEOGRAM_TURBO_ARM] {
        for tier in ["q4", "q8"] {
            expect(arm, tier, protocol::IDEOGRAM_REPOSITORY, "IDEOGRAM");
        }
        expect(
            arm,
            "bf16",
            protocol::IDEOGRAM_BF16_REPOSITORY,
            "IDEOGRAM_BF16",
        );
    }
    // sc-24112: Qwen-Image 2.1 is the split with the halves swapped — the packed tiers are the
    // SceneWorks re-host, and bf16 is the upstream snapshot at its ROOT.
    for tier in ["q4", "q8"] {
        expect(
            QWEN_IMAGE_2_1_ARM,
            tier,
            protocol::QWEN_IMAGE_2_1_REPOSITORY,
            "QWEN_IMAGE_2_1",
        );
        assert!(QWEN_IMAGE_2_1_ARM.family_for(tier).tiered);
    }
    expect(
        QWEN_IMAGE_2_1_ARM,
        "bf16",
        protocol::QWEN_IMAGE_2_1_BF16_REPOSITORY,
        "QWEN_IMAGE_2_1_BF16",
    );
    assert!(!QWEN_IMAGE_2_1_ARM.family_for("bf16").tiered);
    // sc-25907: Turbo has the same split over its OWN two repositories and env families.
    for tier in ["q4", "q8"] {
        expect(
            QWEN_IMAGE_2_1_TURBO_ARM,
            tier,
            protocol::QWEN_IMAGE_2_1_TURBO_REPOSITORY,
            "QWEN_IMAGE_2_1_TURBO",
        );
        assert!(QWEN_IMAGE_2_1_TURBO_ARM.family_for(tier).tiered);
    }
    expect(
        QWEN_IMAGE_2_1_TURBO_ARM,
        "bf16",
        protocol::QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY,
        "QWEN_IMAGE_2_1_TURBO_BF16",
    );
    assert!(!QWEN_IMAGE_2_1_TURBO_ARM.family_for("bf16").tiered);
    // Base and Turbo must never share an artifact at any tier: a Turbo plan satisfied by base
    // weights would re-label base's peaks as the distilled checkpoint's (E10).
    for tier in ["q4", "q8", "bf16"] {
        assert_ne!(
            QWEN_IMAGE_2_1_ARM.family_for(tier).expected_repository,
            QWEN_IMAGE_2_1_TURBO_ARM
                .family_for(tier)
                .expected_repository,
            "{tier}"
        );
    }
    // Base Lens and Lens-Turbo must never share a family: a turbo plan satisfied by base
    // weights would re-label the base model's peaks as the distilled model's.
    assert_ne!(
        LENS_ARM.family.expected_repository,
        LENS_TURBO_ARM.family.expected_repository
    );
}

/// The root must end in the PLANNED tier's directory, and the spec must open exactly that tier
/// under the policy the worker loads: `Resident`, plus the request-derived quant the worker
/// forwards for all five (`mlx_load_quant_for_resolved_artifact` nulls it only for the krea and
/// flux2-klein ids).
#[test]
fn the_turnkey_root_must_carry_the_planned_tier_and_the_spec_binds_it() {
    let revision = "a".repeat(40);
    for (provider, _) in MEMBERS {
        let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", "q4")).unwrap();
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let family = arm.family_for(tier);
            let root = turnkey_family_root(family, &revision, tier);
            let artifact = turnkey_load_spec_at(
                &turnkey_planned(provider, "text_to_image", tier),
                LoadShape::EagerMaterialization,
                family.expected_repository.to_owned(),
                revision.clone(),
                root.clone(),
            )
            .unwrap_or_else(|error| panic!("{provider} {tier}: {error}"));
            assert_eq!(artifact.tier, tier);
            assert_eq!(artifact.spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(artifact.spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(artifact.spec.quantize, quant, "{provider} {tier}");
            assert_eq!(
                artifact.loadability_fingerprint(),
                format!("{}@{revision}:{tier}", family.expected_repository)
            );

            // A root exported for ANOTHER tier can never satisfy this plan.
            let wrong = if tier == "q4" { "q8" } else { "q4" };
            turnkey_load_spec_at(
                &turnkey_planned(provider, "text_to_image", wrong),
                LoadShape::EagerMaterialization,
                family.expected_repository.to_owned(),
                revision.clone(),
                root.clone(),
            )
            .expect_err("a mismatched tier root must be refused");
        }
    }
}

/// Ideogram's bf16 plan must be refused against the PACKED repository, and its packed plans
/// against the bf16 repository. Both are real repositories at real revisions, so nothing but
/// this check stands between a bf16 record and the packed artifact's identity.
#[test]
fn an_ideogram_tier_cannot_be_satisfied_by_the_other_repositorys_snapshot() {
    let revision = "b".repeat(40);
    for arm in [IDEOGRAM_ARM, IDEOGRAM_TURBO_ARM] {
        for (tier, foreign) in [
            ("bf16", protocol::IDEOGRAM_REPOSITORY),
            ("q4", protocol::IDEOGRAM_BF16_REPOSITORY),
        ] {
            let root = turnkey_snapshot_root(foreign, &revision, tier);
            let error = turnkey_load_spec_at(
                &turnkey_planned(arm.provider, "text_to_image", tier),
                LoadShape::EagerMaterialization,
                foreign.to_owned(),
                revision.clone(),
                root,
            )
            .expect_err("the other Ideogram repository must not satisfy this tier");
            assert!(
                error.contains(arm.family_for(tier).expected_repository),
                "{error}"
            );
        }
    }
}

/// Qwen-Image 2.1's bf16 leg is the upstream snapshot ROOT, so a `.../bf16` sub-directory — a
/// path no production load opens — is refused, and neither repository satisfies the other's
/// tiers (sc-24112).
///
/// sc-25907: the same holds for the Turbo member over its own two repositories, and on top of
/// it neither checkpoint's repositories satisfy the OTHER checkpoint's plan at any tier.
#[test]
fn the_qwen_image_2_1_bf16_leg_binds_the_upstream_snapshot_root() {
    let revision = "c".repeat(40);
    for (arm, sibling) in [
        (QWEN_IMAGE_2_1_ARM, QWEN_IMAGE_2_1_TURBO_ARM),
        (QWEN_IMAGE_2_1_TURBO_ARM, QWEN_IMAGE_2_1_ARM),
    ] {
        let dense = arm.family_for("bf16").expected_repository;
        let packed = arm.family_for("q8").expected_repository;
        let tiered_bf16 = turnkey_snapshot_root(dense, &revision, "bf16");
        let error = turnkey_load_spec_at(
            &turnkey_planned(arm.provider, "text_to_image", "bf16"),
            LoadShape::EagerMaterialization,
            dense.to_owned(),
            revision.clone(),
            tiered_bf16,
        )
        .expect_err("a bf16/ sub-directory is not the upstream snapshot root");
        assert!(
            error.contains(dense.replace('/', "--").as_str()),
            "{}: {error}",
            arm.provider
        );
        for (tier, foreign) in [
            ("bf16", packed),
            ("q8", dense),
            ("bf16", sibling.family_for("bf16").expected_repository),
            ("q8", sibling.family_for("q8").expected_repository),
            ("q4", sibling.family_for("q4").expected_repository),
        ] {
            let root = if tier == "bf16" && foreign != packed {
                turnkey_snapshot_root(foreign, &revision, tier)
                    .parent()
                    .unwrap()
                    .to_path_buf()
            } else {
                turnkey_snapshot_root(foreign, &revision, tier)
            };
            let error = turnkey_load_spec_at(
                &turnkey_planned(arm.provider, "text_to_image", tier),
                LoadShape::EagerMaterialization,
                foreign.to_owned(),
                revision.clone(),
                root,
            )
            .expect_err("a foreign Qwen-Image 2.1 repository must not satisfy this tier");
            assert!(
                error.contains(arm.family_for(tier).expected_repository),
                "{} {tier} vs {foreign}: {error}",
                arm.provider
            );
        }
    }
}

/// Every member refuses a non-still geometry before it resolves an environment variable or
/// touches a weight snapshot (sc-18808), under its OWN calibration label.
#[test]
fn every_turnkey_member_refuses_a_multi_frame_geometry() {
    for (provider, _) in MEMBERS {
        let mut request = turnkey_planned(provider, "text_to_image", "q4");
        request["planned"]["target"]["geometry"]["frames"] = json!(2);
        let arm = turnkey_arm(&request).unwrap();
        let error = protocol::validate_still_geometry(&request, arm.still_calibration)
            .expect_err("a multi-frame geometry must be refused");
        assert!(error.contains(arm.still_calibration), "{error}");
    }
}

/// The fixture binds member, tier, edge, seed and step count, so a bf16 record can never be
/// emitted against a q4 capture that merely reused the fixture string.
#[test]
fn the_turnkey_fixture_binds_member_tier_edge_seed_and_steps() {
    for (provider, slug) in MEMBERS {
        let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", "q4")).unwrap();
        let case = |fixture: &str, tier: &str| {
            let mut request = turnkey_planned(provider, "text_to_image", tier);
            request["planned"]["fixture"] = json!(fixture);
            validate_turnkey_fixture(&request, arm, tier)
        };
        case(&format!("{slug}-mlx-q4-1024-seed22732-step2"), "q4").unwrap();
        // A fixture naming another tier, another edge, another seed, or another step count is
        // refused on its own; so is one naming the five-rung seed the CANDLE lane renders at.
        assert!(case(&format!("{slug}-mlx-q8-1024-seed22732-step2"), "q4").is_err());
        assert!(case(&format!("{slug}-mlx-q4-768-seed22732-step2"), "q4").is_err());
        assert!(case(&format!("{slug}-mlx-q4-1024-seed16402-step2"), "q4").is_err());
        assert!(case(&format!("{slug}-mlx-q4-1024-seed22732-step4"), "q4").is_err());
    }
    // And one member's fixture never satisfies another member's plan.
    let arm = turnkey_arm(&turnkey_planned(LENS_PROVIDER, "text_to_image", "q4")).unwrap();
    let mut request = turnkey_planned(LENS_PROVIDER, "text_to_image", "q4");
    request["planned"]["fixture"] = json!("lens-turbo-mlx-q4-1024-seed22732-step2");
    validate_turnkey_fixture(&request, arm, "q4")
        .expect_err("Lens-Turbo's fixture must not satisfy a base Lens plan");
    // sc-25907: nor does either Qwen-Image 2.1 checkpoint's fixture satisfy the other's plan.
    for (provider, foreign) in [
        (
            QWEN_IMAGE_2_1_PROVIDER,
            "qwen-image-2-1-turbo-mlx-q4-1024-seed22732-step2",
        ),
        (
            QWEN_IMAGE_2_1_TURBO_PROVIDER,
            "qwen-image-2-1-mlx-q4-1024-seed22732-step2",
        ),
    ] {
        let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", "q4")).unwrap();
        let mut request = turnkey_planned(provider, "text_to_image", "q4");
        request["planned"]["fixture"] = json!(foreign);
        validate_turnkey_fixture(&request, arm, "q4")
            .expect_err("the sibling checkpoint's fixture must not satisfy this plan");
    }
}

/// The committed plan's MLX turnkey cells, as an exact key set: five members x three tiers,
/// each served by an arm whose fixture, tier, geometry and load shape all bind.
#[test]
fn every_planned_turnkey_mlx_cell_is_served_by_an_arm() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        if !key.ends_with(":mlx") {
            continue;
        }
        let provider = entry["provider"].as_str().unwrap();
        if !MEMBERS.iter().any(|(member, _)| *member == provider) {
            continue;
        }
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "backend": "mlx",
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        let arm = turnkey_arm(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(planned_qwen_tier(&request).unwrap(), tier, "{key}");
        validate_turnkey_fixture(&request, arm, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        protocol::validate_still_geometry(&request, arm.still_calibration)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        // The anchor rung on this lane is `resident` (`memory-calibration-harness.mjs`
        // ANCHOR_STRATEGY.mlx), and the worker materializes all five eagerly under it.
        assert_eq!(
            planned_load_shape(&request).unwrap(),
            LoadShape::EagerMaterialization,
            "{key}"
        );
    }
    let expected: std::collections::BTreeSet<String> = [
        "kolors",
        "ideogram_4",
        "ideogram_4_turbo",
        "lens",
        "lens_turbo",
        "qwen_image_2_1",
        "qwen_image_2_1_turbo",
    ]
    .iter()
    .flat_map(|model| {
        ["bf16", "q4", "q8"]
            .iter()
            .map(move |tier| format!("{model}:{tier}:mlx"))
    })
    .collect();
    assert_eq!(seen, expected);
}

/// sc-22732 review: the identity table must REFUSE an unknown coordinate, not synthesize one.
/// The per-member arms bind `tier` as a free variable, so before the guard
/// `turnkey_calibration_fingerprint(kolors_arm, "q2")` returned
/// `"kolors-q2-mlx-shared-ladder-v1"` — a well-formed identity no engine publishes — and the
/// plan check compared a plan row against it instead of refusing the coordinate.
#[test]
fn the_turnkey_mlx_identity_table_refuses_an_unshipped_tier() {
    for (provider, _) in MEMBERS {
        // Every shipped tier stays nameable, so the guard cannot pass by refusing everything.
        for tier in ["bf16", "q4", "q8"] {
            let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", tier)).unwrap();
            assert!(
                turnkey_calibration_fingerprint(arm, tier).is_some(),
                "{provider} {tier} is a shipped turnkey cell"
            );
        }
        // Tiers this family does not ship: the axis the free variable let through.
        let arm = turnkey_arm(&turnkey_planned(provider, "text_to_image", "q4")).unwrap();
        for tier in ["q2", "q6", "nvfp4", "fp8", "bf16 ", "", "Q4"] {
            assert_eq!(
                turnkey_calibration_fingerprint(arm, tier),
                None,
                "{provider} {tier:?} is not a shipped turnkey tier"
            );
        }
    }
}

/// Every turnkey MLX plan row names the production calibration identity its loaded generator
/// publishes (the sc-22732 inference head's per-(route, tier) tables): the two measured keys
/// are the engine constants byte-for-byte, no row names a weights-free conformance string or
/// the legacy dense Lens-Turbo key, and the fifteen identities are distinct. The
/// member-resolution test above never looks at `calibrationFingerprint`, which is how rows
/// naming strings no `Resident` load can return shipped in the first draft of this plan.
#[test]
fn every_planned_turnkey_mlx_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    let mut qwen_image_2_1 = Vec::new();
    let mut qwen_image_2_1_turbo = Vec::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":mlx") || !MEMBERS.iter().any(|(member, _)| *member == provider) {
            continue;
        }
        // sc-24112: `mlx-gen-qwen-image-2-1` publishes ONE identity for every tier, so its
        // three cells share it by construction and are held to the engine constant instead.
        // sc-25907: the same per checkpoint — Turbo's three cells share Turbo's OWN constant.
        if provider == QWEN_IMAGE_2_1_PROVIDER || provider == QWEN_IMAGE_2_1_TURBO_PROVIDER {
            let bucket = if provider == QWEN_IMAGE_2_1_PROVIDER {
                &mut qwen_image_2_1
            } else {
                &mut qwen_image_2_1_turbo
            };
            bucket.push((
                key.clone(),
                entry["calibrationFingerprint"].as_str().unwrap().to_owned(),
            ));
            continue;
        }
        let tier = key.split(':').nth(1).unwrap();
        let mode = entry["mode"].as_str().unwrap();
        let arm = turnkey_arm(&turnkey_planned(provider, mode, tier)).unwrap();
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !is_turnkey_weights_free_fingerprint(planned),
            "{key}: {planned} is a weights-free conformance identity no production load returns"
        );
        assert_eq!(
            planned,
            turnkey_calibration_fingerprint(arm, tier)
                .unwrap_or_else(|| panic!("{key}: no turnkey identity"))
                .as_str(),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    assert_eq!(
        identities.len(),
        15,
        "fifteen distinct turnkey MLX identities"
    );
    assert_eq!(
        qwen_image_2_1.len(),
        3,
        "one Qwen-Image 2.1 MLX cell per tier"
    );
    for (key, planned) in &qwen_image_2_1 {
        assert_eq!(
                planned,
                runtime_macos::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
                "{key}: the plan row must name the loaded generator's production identity"
            );
        assert!(!identities.contains_key(planned.as_str()), "{key}");
    }
    assert_eq!(
        runtime_macos::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "qwen-image-2-1-mlx-derived-2026-09-22-v1"
    );
    // sc-25907: Turbo's cells name Turbo's OWN engine constant — never base's.
    assert_eq!(
        qwen_image_2_1_turbo.len(),
        3,
        "one Qwen-Image 2.1 Turbo MLX cell per tier"
    );
    for (key, planned) in &qwen_image_2_1_turbo {
        assert_eq!(
                planned,
                runtime_macos::providers::qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT,
                "{key}: the plan row must name the loaded Turbo generator's production identity"
            );
        assert_eq!(
            Some(planned.clone()),
            turnkey_calibration_fingerprint(
                QWEN_IMAGE_2_1_TURBO_ARM,
                key.split(':').nth(1).unwrap()
            ),
            "{key}"
        );
        assert!(!identities.contains_key(planned.as_str()), "{key}");
    }
    assert_eq!(
            runtime_macos::providers::qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT,
            "qwen-image-2-1-turbo-mlx-derived-2026-10-13-v1"
        );
    assert_ne!(
            runtime_macos::providers::qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT,
            runtime_macos::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
        );
    // The preserved measured keys, byte-for-byte, so the table cannot drift off the engine.
    assert_eq!(
        identities["kolors-mlx-chatglm3-sdxl-unet-ladder-v1"],
        "kolors:q4:mlx"
    );
    assert_eq!(
        identities["lens-mlx-shared-ladder-2026-08-03-v1"],
        "lens:q4:mlx"
    );
    assert_eq!(
        runtime_macos::providers::kolors::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "kolors-mlx-chatglm3-sdxl-unet-ladder-v1"
    );
    assert_eq!(
        runtime_macos::providers::lens::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "lens-mlx-shared-ladder-2026-08-03-v1"
    );
    // The legacy dense Lens-Turbo key is a real engine constant and it is NOT a cell here.
    assert_eq!(
        runtime_macos::providers::lens::memory_strategy::LEGACY_TEXT_ENCODER_FINGERPRINT,
        "lens-text-encoder-window-2026-07-31-v1"
    );
    assert!(!identities.contains_key("lens-text-encoder-window-2026-07-31-v1"));
    // The thirteen new cells, spelled out: the strings the sc-22732 inference head mints.
    for (key, identity) in [
        ("kolors:q8:mlx", "kolors-q8-mlx-shared-ladder-v1"),
        ("kolors:bf16:mlx", "kolors-bf16-mlx-shared-ladder-v1"),
        ("ideogram_4:q4:mlx", "ideogram-4-q4-mlx-shared-ladder-v1"),
        ("ideogram_4:q8:mlx", "ideogram-4-q8-mlx-shared-ladder-v1"),
        (
            "ideogram_4:bf16:mlx",
            "ideogram-4-bf16-mlx-shared-ladder-v1",
        ),
        (
            "ideogram_4_turbo:q4:mlx",
            "ideogram-4-turbo-q4-mlx-shared-ladder-v1",
        ),
        (
            "ideogram_4_turbo:q8:mlx",
            "ideogram-4-turbo-q8-mlx-shared-ladder-v1",
        ),
        (
            "ideogram_4_turbo:bf16:mlx",
            "ideogram-4-turbo-bf16-mlx-shared-ladder-v1",
        ),
        ("lens:q8:mlx", "lens-q8-mlx-shared-ladder-v1"),
        ("lens:bf16:mlx", "lens-bf16-mlx-shared-ladder-v1"),
        ("lens_turbo:q4:mlx", "lens-turbo-q4-mlx-shared-ladder-v1"),
        ("lens_turbo:q8:mlx", "lens-turbo-q8-mlx-shared-ladder-v1"),
        (
            "lens_turbo:bf16:mlx",
            "lens-turbo-bf16-mlx-shared-ladder-v1",
        ),
    ] {
        assert_eq!(identities[identity], key);
    }
}

/// The arm binds the plan row to the table BEFORE any environment or weight work, naming
/// both strings, so a stale row cannot cost a load.
#[test]
fn a_plan_row_naming_a_foreign_identity_is_refused_before_the_load() {
    for (provider, slug, foreign) in [
            (
                LENS_TURBO_PROVIDER,
                "lens-turbo",
                "lens-text-encoder-window-2026-07-31-v1",
            ),
            (
                KOLORS_PROVIDER,
                "kolors",
                "kolors-mlx-registry-behavior-v1-bf16-dense-resident",
            ),
            (
                IDEOGRAM_PROVIDER,
                "ideogram-4",
                "ideogram4-mlx-registry-behavior-v1-ideogram-4-bf16-dense-resident",
            ),
            // sc-25907 (E10): a Turbo row naming BASE's identity is refused, so base's measured
            // record can never be presented as Turbo's.
            (
                QWEN_IMAGE_2_1_TURBO_PROVIDER,
                "qwen-image-2-1-turbo",
                runtime_macos::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
            ),
        ] {
            let mut request = turnkey_planned(provider, "text_to_image", "bf16");
            request["planned"]["fixture"] = json!(format!(
                "{slug}-mlx-bf16-1024-seed{TURNKEY_SEED}-step{TURNKEY_STEPS}"
            ));
            request["planned"]["calibrationFingerprint"] = json!(foreign);
            let error = run(&request).expect_err("a conformance identity is not capturable");
            let arm = turnkey_arm(&request).unwrap();
            assert!(
                error.contains(&format!("plan={foreign}"))
                    && error.contains(
                        &turnkey_calibration_fingerprint(arm, "bf16")
                            .expect("bf16 is a shipped turnkey tier")
                    ),
                "{provider}: {error}"
            );
            // Not an env error: the refusal fired before `SCENEWORKS_<MEMBER>_*` was read.
            assert!(!error.contains("required environment variable"), "{error}");
        }
}
