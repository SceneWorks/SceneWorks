use super::*;

/// The three shipped tiers of this lane. One spelling, so the expectation below and the tables
/// it is compared against cannot disagree about which cells exist.
const SANA_CHROMA_MLX_TIERS: [&str; 3] = ["q4", "q8", "bf16"];

/// The plan keys this arm must serve, DERIVED from `SANA_CHROMA_ARMS` rather than counted.
/// A member added to the table without a plan row (or a plan row without a member) names itself
/// in the diff; a frozen `15` only ever said "there are sixteen, not fifteen".
fn expected_sana_chroma_mlx_keys() -> std::collections::BTreeSet<String> {
    SANA_CHROMA_ARMS
        .iter()
        .flat_map(|arm| {
            SANA_CHROMA_MLX_TIERS
                .iter()
                .map(move |tier| format!("{}:{tier}:mlx", arm.provider))
        })
        .collect()
}

/// The shape this member's plan rows carry, so no fixture spells a materialization the arm
/// would refuse. Foreign providers (the misroute cases) keep the deferred default.
fn planned_load_shape_key(provider: &str) -> &'static str {
    load_shape_key(
        SANA_CHROMA_ARMS
            .iter()
            .find(|arm| arm.provider == provider)
            .map(|arm| arm.load_shape)
            .unwrap_or(LoadShape::DeferredMaterialization),
    )
}

fn sana_chroma_planned(provider: &str, tier: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": planned_load_shape_key(provider),
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": "unused"
        }
    })
}

fn sana_chroma_snapshot_root(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
) -> PathBuf {
    fixtures.snapshot(repository, revision, Some(tier))
}

const REVISION: &str = "ac421696dd6eb0b41f446f4c45a53ccc057d82a1";

/// Every SANA/Chroma1 MLX plan row names the production calibration identity its loaded
/// generator publishes (inference PR 951's per-(provider, tier) tables): the two measured keys
/// are the engine constants byte-for-byte, no row names a weights-free conformance string, and
/// the fifteen identities are distinct.
#[test]
fn every_planned_sana_chroma_mlx_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    let mut visited = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":mlx") || !SANA_CHROMA_ARMS.iter().any(|arm| arm.provider == provider) {
            continue;
        }
        visited.insert(key.clone());
        let tier = key.split(':').nth(1).unwrap();
        let arm = sana_chroma_arm(&sana_chroma_planned(provider, tier)).unwrap();
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !is_sana_chroma_mlx_weights_free_fingerprint(planned),
            "{key}: {planned} is a weights-free conformance identity no production load returns"
        );
        assert_eq!(
            planned,
            sana_chroma_calibration_fingerprint(arm, tier),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    // A SET, not a count (sc-22731 review): the expectation is derived from `SANA_CHROMA_ARMS`
    // x the shipped tiers, so a missing or extra plan row names ITSELF rather than reporting
    // "there are fourteen, not fifteen".
    assert_eq!(
        visited,
        expected_sana_chroma_mlx_keys(),
        "the MLX plan rows must be exactly the arm table's members x the shipped tiers"
    );
    // ...and every one of them carries a DISTINCT identity, so the map covers the same set.
    assert_eq!(
        identities
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        expected_sana_chroma_mlx_keys()
    );
    // The preserved measured keys, byte-for-byte, so the table cannot drift off the engines.
    assert_eq!(
        identities["sana-mlx-full-ladder-2026-08-06-v1-resident"],
        "sana_1600m:q4:mlx"
    );
    assert_eq!(
        identities["chroma1-base-q4-mlx-shared-ladder-2026-08-05-v1"],
        "chroma1_base:q4:mlx"
    );
    assert_eq!(
        runtime_macos::providers::sana::memory_strategy::RESIDENT_MEMORY_CALIBRATION_FINGERPRINT,
        "sana-mlx-full-ladder-2026-08-06-v1-resident"
    );
    assert_eq!(
        runtime_macos::providers::chroma::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "chroma1-base-q4-mlx-shared-ladder-2026-08-05-v1"
    );
    // The thirteen new cells, spelled out: the strings inference PR 951 mints.
    for (key, identity) in [
        (
            "sana_1600m:q8:mlx",
            "sana-base-q8-mlx-full-ladder-v1-resident",
        ),
        (
            "sana_1600m:bf16:mlx",
            "sana-base-bf16-mlx-full-ladder-v1-resident",
        ),
        (
            "sana_sprint_1600m:q4:mlx",
            "sana-sprint-q4-mlx-full-ladder-v1-resident",
        ),
        (
            "sana_sprint_1600m:q8:mlx",
            "sana-sprint-q8-mlx-full-ladder-v1-resident",
        ),
        (
            "sana_sprint_1600m:bf16:mlx",
            "sana-sprint-bf16-mlx-full-ladder-v1-resident",
        ),
        ("chroma1_hd:q4:mlx", "chroma1-hd-q4-mlx-shared-ladder-v1"),
        ("chroma1_hd:q8:mlx", "chroma1-hd-q8-mlx-shared-ladder-v1"),
        (
            "chroma1_hd:bf16:mlx",
            "chroma1-hd-bf16-mlx-shared-ladder-v1",
        ),
        (
            "chroma1_base:q8:mlx",
            "chroma1-base-q8-mlx-shared-ladder-v1",
        ),
        (
            "chroma1_base:bf16:mlx",
            "chroma1-base-bf16-mlx-shared-ladder-v1",
        ),
        (
            "chroma1_flash:q4:mlx",
            "chroma1-flash-q4-mlx-shared-ladder-v1",
        ),
        (
            "chroma1_flash:q8:mlx",
            "chroma1-flash-q8-mlx-shared-ladder-v1",
        ),
        (
            "chroma1_flash:bf16:mlx",
            "chroma1-flash-bf16-mlx-shared-ladder-v1",
        ),
    ] {
        assert_eq!(identities[identity], key);
    }
}

/// The arm binds the plan row to the table BEFORE any environment or weight work, naming both
/// strings, so a stale row cannot cost a load.
#[test]
fn a_plan_row_naming_a_foreign_identity_is_refused_before_the_load() {
    let mut request = sana_chroma_planned(CHROMA1_FLASH_PROVIDER, "q8");
    request["planned"]["fixture"] = json!(format!(
        "chroma1-flash-mlx-q8-1024-seed{SANA_CHROMA_SEED}-step2"
    ));
    request["planned"]["calibrationFingerprint"] =
        json!("chroma1-mlx-registry-behavior-v1-chroma1-flash");
    let error = run(&request).expect_err("a conformance identity is not capturable");
    assert!(
        error.contains("plan=chroma1-mlx-registry-behavior-v1-chroma1-flash")
            && error.contains("chroma1-flash-q8-mlx-shared-ladder-v1"),
        "{error}"
    );
    // Not an env error: the refusal fired before `SCENEWORKS_CHROMA1_FLASH_*` was read.
    assert!(!error.contains("required environment variable"), "{error}");
}

/// A provider no member serves is refused by name, and the refusal does not read as a malformed
/// plan entry (the sc-18104 misroute lesson).
#[test]
fn a_provider_no_member_serves_is_refused_by_name() {
    for provider in ["sana", "chroma1", "chroma1_hd_edit"] {
        let error = sana_chroma_arm(&sana_chroma_planned(provider, "q4"))
            .expect_err("a foreign provider must not resolve to a member");
        assert_eq!(
            error,
            format!("the MLX SANA/Chroma1 arm does not implement provider {provider:?}")
        );
    }
}

/// The LoadSpec is the shape the WORKER loads: Resident, the planned tier on `quantize`, the
/// exact resolved route, and the planned materialization shape.
#[test]
fn the_load_spec_is_resident_and_carries_the_planned_tier_and_route() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for (arm, tier, quant) in [
        (SANA_ARM, "q4", Some(Quant::Q4)),
        (SANA_ARM, "q8", Some(Quant::Q8)),
        (SANA_ARM, "bf16", None),
        (CHROMA1_HD_ARM, "q4", Some(Quant::Q4)),
        (CHROMA1_FLASH_ARM, "bf16", None),
    ] {
        let root = sana_chroma_snapshot_root(fixtures, arm.expected_repository, REVISION, tier);
        let artifact = sana_chroma_load_spec_at(
            &sana_chroma_planned(arm.provider, tier),
            arm.load_shape,
            arm.expected_repository.to_owned(),
            REVISION.to_owned(),
            root,
        )
        .unwrap_or_else(|error| panic!("{}:{tier}: {error}", arm.provider));
        assert_eq!(artifact.tier, tier);
        assert_eq!(artifact.spec.offload_policy, OffloadPolicy::Resident);
        assert_eq!(artifact.spec.quantize, quant);
        assert_eq!(
            artifact.spec.resolved_route.as_deref(),
            Some(arm.provider),
            "candle-gen-chroma's twin refuses a spec without it; the worker sets it on both lanes"
        );
        // The member's OWN shape reaches the spec — SANA eager, Chroma deferred — so the two
        // families can never be loaded under one another's materialization.
        assert_eq!(artifact.spec.load_shape, arm.load_shape);
    }
}

/// A root exported for ANOTHER tier is refused naming the tier, so a stale `…/q4` export can
/// never satisfy a q8 or bf16 plan and quietly re-label another tier's peaks.
#[test]
fn a_root_of_another_tier_is_refused_naming_the_tier() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for arm in SANA_CHROMA_ARMS {
        let root = sana_chroma_snapshot_root(fixtures, arm.expected_repository, REVISION, "q4");
        let error = sana_chroma_load_spec_at(
            &sana_chroma_planned(arm.provider, "q8"),
            LoadShape::DeferredMaterialization,
            arm.expected_repository.to_owned(),
            REVISION.to_owned(),
            root,
        )
        .expect_err("a q4 root must not satisfy a q8 plan");
        assert!(
            error.contains("q8"),
            "{}: the refusal must name the planned tier: {error}",
            arm.provider
        );
    }
}

/// ...and the companion direction: a root exported for a DIFFERENT repository is refused too,
/// so a Flash plan can never be satisfied by HD weights.
#[test]
fn a_root_of_another_route_is_refused() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root = sana_chroma_snapshot_root(fixtures, protocol::CHROMA1_HD_REPOSITORY, REVISION, "q4");
    let error = sana_chroma_load_spec_at(
        &sana_chroma_planned(CHROMA1_FLASH_PROVIDER, "q4"),
        LoadShape::DeferredMaterialization,
        protocol::CHROMA1_HD_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        root,
    )
    .expect_err("HD weights must not satisfy a Flash plan");
    assert!(
        error.contains(protocol::CHROMA1_FLASH_REPOSITORY),
        "{error}"
    );
}

/// The fixture binds the member, the tier, the edge, the seed and the step count.
#[test]
fn the_fixture_must_name_the_member_tier_edge_and_seed() {
    let request = sana_chroma_planned(SANA_SPRINT_PROVIDER, "q8");
    for (fixture, needle) in [
        // another member's fixture
        ("sana-1600m-mlx-q8-1024-seed22731-step2", "must start with"),
        // another tier
        ("sana-sprint-mlx-q4-1024-seed22731-step2", "must start with"),
        // another edge
        ("sana-sprint-mlx-q8-768-seed22731-step2", "must start with"),
        // another seed
        ("sana-sprint-mlx-q8-1024-seed16402-step2", "does not match"),
        // another step count
        ("sana-sprint-mlx-q8-1024-seed22731-step4", "two-step"),
    ] {
        let mut request = request.clone();
        request["planned"]["fixture"] = json!(fixture);
        let error = validate_sana_chroma_fixture(&request, SANA_SPRINT_ARM, "q8")
            .expect_err("a mislabelled fixture must be refused");
        assert!(error.contains(needle), "{fixture}: {error}");
    }
    let mut request = request;
    request["planned"]["fixture"] = json!("sana-sprint-mlx-q8-1024-seed22731-step2");
    validate_sana_chroma_fixture(&request, SANA_SPRINT_ARM, "q8")
        .expect("the planned fixture must be accepted");
}

/// Every plan row's fixture is the one this arm accepts, so the plan and the arm cannot drift.
#[test]
fn every_planned_mlx_fixture_is_accepted_by_the_arm() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut checked = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        let Some(arm) = SANA_CHROMA_ARMS
            .iter()
            .copied()
            .find(|arm| arm.provider == provider)
        else {
            continue;
        };
        if !key.ends_with(":mlx") {
            continue;
        }
        let tier = key.split(':').nth(1).unwrap();
        let mut request = sana_chroma_planned(provider, tier);
        request["planned"]["fixture"] = entry["fixture"].clone();
        request["planned"]["target"]["geometry"] = entry["geometry"].clone();
        // The plan row's OWN materialization shape, copied in rather than assumed (sc-22731
        // review): `sana_chroma_planned` derives its shape from the arm table, so the plan's
        // own `loadShape` would otherwise never be read by any test at all.
        // `planned_load_shape` is the same parse `run_sana_chroma` performs.
        request["planned"]["loadShape"] = entry["loadShape"].clone();
        assert_eq!(
            planned_load_shape(&request).unwrap_or_else(|error| panic!("{key}: {error}")),
            arm.load_shape,
            "{key}: the plan row must name the shape this member is captured under — \
                 mlx-gen-sana publishes its resident ladder EAGER (the windowed rung is declared \
                 only under Sequential, which no anchor plans), mlx-gen-chroma's is DEFERRED"
        );
        validate_sana_chroma_fixture(&request, arm, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        checked.insert(key.clone());
    }
    // Set equality, not a count: see `expected_sana_chroma_mlx_keys`.
    assert_eq!(checked, expected_sana_chroma_mlx_keys());
}
