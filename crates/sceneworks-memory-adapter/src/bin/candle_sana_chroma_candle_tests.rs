use super::*;

const SANA_CHROMA_PROVIDERS: [&str; 5] = [
    SANA_ID,
    SANA_SPRINT_ID,
    CHROMA1_HD_ID,
    CHROMA1_BASE_ID,
    CHROMA1_FLASH_ID,
];

/// The three shipped tiers of the packed turnkeys, one spelling for every expectation below.
const SANA_CHROMA_CANDLE_TIERS: [&str; 3] = ["q4", "q8", "bf16"];

/// The Candle plan keys these two families must serve, DERIVED: the five routes crossed with
/// the shipped tiers, MINUS every (route, tier) `validate_five_rung_lane_tier` refuses — which
/// is SANA's q4/q8 alone, because there is no packed SANA artifact off-Mac. A frozen `11` said
/// nothing about WHICH eleven; this fails naming the cell that appeared or vanished, and it
/// re-derives itself the moment the lane's tier rule changes.
fn expected_sana_chroma_candle_keys() -> std::collections::BTreeSet<String> {
    SANA_CHROMA_PROVIDERS
        .iter()
        .flat_map(|provider| {
            SANA_CHROMA_CANDLE_TIERS
                .iter()
                .map(move |tier| (*provider, *tier))
        })
        .filter(|(provider, tier)| validate_five_rung_lane_tier(provider, tier).is_ok())
        .map(|(provider, tier)| format!("{provider}:{tier}:candle"))
        .collect()
}

fn planned(provider: &str, tier: &str, fixture: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "candle",
            "loadShape": "deferred_materialization",
            "strategy": { "rung": "staged_residency", "engagedRungs": ["resident", "staged_residency"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": fixture
        }
    })
}

/// Every SANA/Chroma1 CANDLE plan row names the identity its loaded generator publishes, and
/// the identities are distinct. `candle-gen-sana` mints one per route (bf16 is its only tier);
/// `candle-gen-chroma` mints one per (route, tier) from the load receipt (inference PR 951 —
/// before it, `build_contract` hard-coded `calibration: None` and no Chroma anchor could be
/// recorded on this lane at all).
#[test]
fn every_planned_sana_chroma_candle_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    let mut visited = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":candle") || !SANA_CHROMA_PROVIDERS.contains(&provider) {
            continue;
        }
        visited.insert(key.clone());
        let tier = key.split(':').nth(1).unwrap();
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert_eq!(
            Some(planned.to_owned()),
            five_rung_calibration_fingerprint(provider, tier),
            "{key}: the plan row must name the loaded generator's production identity"
        );
        assert!(
            identities.insert(planned.to_owned(), key.clone()).is_none(),
            "{key}: {planned} is already claimed by {}",
            identities[planned]
        );
    }
    // A SET, not a count (sc-22731 review): see `expected_sana_chroma_candle_keys`.
    assert_eq!(
        visited,
        expected_sana_chroma_candle_keys(),
        "the Candle plan rows must be exactly the routes x tiers this lane can open"
    );
    // ...and every one carries a DISTINCT identity, so the map covers the same set.
    assert_eq!(
        identities
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        expected_sana_chroma_candle_keys()
    );
    for (key, identity) in [
        (
            "sana_1600m:bf16:candle",
            "sana-candle-dense-base-full-ladder-v1",
        ),
        (
            "sana_sprint_1600m:bf16:candle",
            "sana-candle-dense-sprint-full-ladder-v1",
        ),
        (
            "chroma1_hd:q4:candle",
            "chroma1-hd-q4-cuda-request-scoped-staged-residency-v1",
        ),
        (
            "chroma1_base:q8:candle",
            "chroma1-base-q8-cuda-request-scoped-staged-residency-v1",
        ),
        (
            "chroma1_flash:bf16:candle",
            "chroma1-flash-bf16-cuda-request-scoped-staged-residency-v1",
        ),
    ] {
        assert_eq!(identities[identity], key);
    }
}

/// The Candle SANA lane has ONE tier. A packed plan row is refused by name, before any
/// environment is read, rather than reaching `candle-gen-sana` and coming back as a
/// quantization complaint that reads like a spec bug.
#[test]
fn the_candle_sana_lane_refuses_a_packed_tier_by_name() {
    for provider in [SANA_ID, SANA_SPRINT_ID] {
        for tier in ["q4", "q8"] {
            let error = validate_five_rung_lane_tier(provider, tier)
                .expect_err("there is no packed SANA artifact off-Mac");
            assert!(error.contains(provider) && error.contains(tier), "{error}");
        }
        validate_five_rung_lane_tier(provider, "bf16").expect("bf16 is the dense lane tier");
    }
    // ...and the constraint is SANA's alone: every Chroma1 tier is a real Candle cell.
    for provider in [CHROMA1_HD_ID, CHROMA1_BASE_ID, CHROMA1_FLASH_ID] {
        for tier in ["q4", "q8", "bf16"] {
            validate_five_rung_lane_tier(provider, tier).unwrap();
        }
    }
}

/// The SANA routes load the upstream dense SNAPSHOT ROOT — no tier component — because that is
/// what the worker resolves for them (`huggingface_pinned_snapshot_dir`) and what
/// `candle-gen-sana`'s `validate_immutable_root` requires. Every other route is tiered.
#[test]
fn only_the_sana_routes_load_an_untiered_snapshot_root() {
    assert!(!five_rung_root_is_tiered(SANA_ID, "bf16"));
    assert!(!five_rung_root_is_tiered(SANA_SPRINT_ID, "bf16"));
    for provider in [
        CHROMA1_HD_ID,
        CHROMA1_BASE_ID,
        CHROMA1_FLASH_ID,
        FLUX1_DEV_ID,
        QWEN_ID,
        KREA_ID,
    ] {
        for tier in ["bf16", "q4", "q8"] {
            assert!(
                five_rung_root_is_tiered(provider, tier),
                "{provider} {tier}"
            );
        }
    }
    // sc-24112: Qwen-Image 2.1 is untiered at bf16 ALONE (the upstream snapshot root); its
    // packed tiers are the re-host's `<tier>/` subdirs.
    assert!(!five_rung_root_is_tiered(QWEN_IMAGE_2_1_ID, "bf16"));
    assert!(five_rung_root_is_tiered(QWEN_IMAGE_2_1_ID, "q4"));
    assert!(five_rung_root_is_tiered(QWEN_IMAGE_2_1_ID, "q8"));
    // sc-25907: Turbo has base's shape.
    assert!(!five_rung_root_is_tiered(QWEN_IMAGE_2_1_TURBO_ID, "bf16"));
    assert!(five_rung_root_is_tiered(QWEN_IMAGE_2_1_TURBO_ID, "q4"));
    assert!(five_rung_root_is_tiered(QWEN_IMAGE_2_1_TURBO_ID, "q8"));
}

/// Every one of the five routes is named by BOTH shared dispatch tables, so neither the plain
/// execution path nor the still-geometry refusal can claim this adapter does not implement it.
#[test]
fn every_route_is_named_by_both_shared_dispatch_tables() {
    for provider in SANA_CHROMA_PROVIDERS {
        let request = planned(provider, "bf16", "unused");
        let path =
            plain_execution_path(&request).unwrap_or_else(|error| panic!("{provider}: {error}"));
        assert!(path.starts_with("the Candle "), "{provider}: {path}");
        let label =
            still_calibration_label(&request).unwrap_or_else(|error| panic!("{provider}: {error}"));
        assert!(label.starts_with("Candle "), "{provider}: {label}");
        assert!(five_rung_family_slug(provider).is_some(), "{provider}");
    }
}

/// The fixture binds the route, the tier, the geometry edge, the seed and the step count.
#[test]
fn the_candle_fixture_must_name_the_route_tier_edge_and_seed() {
    for (fixture, needle) in [
        // another route
        (
            "fresh-five-rung-chroma1-base-q8-1024-seed16402-step2",
            "must start with",
        ),
        // another tier
        (
            "fresh-five-rung-chroma1-hd-q4-1024-seed16402-step2",
            "must start with",
        ),
        // another edge
        (
            "fresh-five-rung-chroma1-hd-q8-768-seed16402-step2",
            "must start with",
        ),
        // another seed
        (
            "fresh-five-rung-chroma1-hd-q8-1024-seed22726-step2",
            "does not match",
        ),
        // another step count
        (
            "fresh-five-rung-chroma1-hd-q8-1024-seed16402-step4",
            "two-step",
        ),
    ] {
        let request = planned(CHROMA1_HD_ID, "q8", fixture);
        let error = validate_fixture_binds_tier_and_geometry(&request)
            .expect_err("a mislabelled fixture must be refused");
        assert!(error.contains(needle), "{fixture}: {error}");
    }
    validate_fixture_binds_tier_and_geometry(&planned(
        CHROMA1_HD_ID,
        "q8",
        "fresh-five-rung-chroma1-hd-q8-1024-seed16402-step2",
    ))
    .expect("the planned fixture must be accepted");
}

/// Every plan row's fixture is one this arm accepts AND one that routes to the five-rung
/// reference path, so a plan row can never land in the Krea arm's catch-all.
#[test]
fn every_planned_candle_fixture_is_accepted_and_routes_to_the_five_rung_path() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut checked = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":candle") || !SANA_CHROMA_PROVIDERS.contains(&provider) {
            continue;
        }
        let tier = key.split(':').nth(1).unwrap();
        let request = planned(provider, tier, entry["fixture"].as_str().unwrap());
        assert!(
            routes_to_five_rung_reference(&request).unwrap(),
            "{key}: the fixture must route to the five-rung reference arm"
        );
        validate_fixture_binds_tier_and_geometry(&request)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        validate_five_rung_lane_tier(provider, tier)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        checked.insert(key.clone());
    }
    // Set equality, not a count: see `expected_sana_chroma_candle_keys`.
    assert_eq!(checked, expected_sana_chroma_candle_keys());
}
