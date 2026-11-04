use super::*;

/// The tiers the plan is required to cover for both models. DERIVED from what the checked-in
/// plan carries and asserted to equal this set, rather than a frozen row count.
const NUMERIC_TIERS: [&str; 3] = ["bf16", "q4", "q8"];

fn plan() -> Value {
    serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses")
}

/// (a) Every planned `mlx` row of `krea_2_raw` and `krea_realtime_14b` names exactly the string
/// [`mlx_production_fingerprint`] resolves for its own `(provider, tier)`, and between them the
/// two models cover exactly `{bf16, q4, q8}`. This needs no weights and no environment, and it
/// is what makes a plan/adapter drift impossible in either direction: a plan row this table
/// cannot produce reds here, and a table entry the plan does not use is caught by the coverage
/// assertion below.
#[test]
fn every_planned_mlx_row_of_the_widened_models_is_producible() {
    let plan = plan();
    let anchors = plan["anchors"].as_object().expect("anchors object");
    let mut covered: std::collections::BTreeMap<&str, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for (key, entry) in anchors {
        if !key.ends_with(":mlx") {
            continue;
        }
        let provider = entry["provider"].as_str().expect("provider string");
        if !matches!(
            provider,
            KREA_BASE_PROVIDER | KREA_RAW_PROVIDER | KREA_REALTIME_PROVIDER
        ) {
            continue;
        }
        // `<provider>:<tier>:<backend>` — read the tier from the key rather than assuming it.
        let tier = key
            .split(':')
            .nth(1)
            .expect("an anchor key carries provider:tier:backend");
        let expected = mlx_production_fingerprint(provider, tier).unwrap_or_else(|| {
            panic!("{key}: no production MLX identity is modelled for ({provider}, {tier})")
        });
        assert_eq!(
            entry["calibrationFingerprint"].as_str(),
            Some(expected.as_str()),
            "{key}"
        );
        covered.entry(provider).or_default().insert(tier.to_owned());
    }
    let wanted: std::collections::BTreeSet<String> = NUMERIC_TIERS
        .iter()
        .map(|tier| (*tier).to_owned())
        .collect();
    for provider in [
        KREA_BASE_PROVIDER,
        KREA_RAW_PROVIDER,
        KREA_REALTIME_PROVIDER,
    ] {
        assert_eq!(
            covered.get(provider),
            Some(&wanted),
            "{provider} must carry exactly one planned MLX row per numeric tier"
        );
    }
}

/// (b) All NINE modelled cells — Turbo's three included, sc-22735 fix pass — are pairwise
/// distinct, each keys on its own tier, and none of them is the retired string every route used
/// to share. Turbo is no longer carved out: nothing was ever measured against that string on
/// this lane, so keeping it collapsed three tiers onto one key for no evidentiary gain.
#[test]
fn every_modelled_cell_is_pairwise_distinct_and_keys_on_its_own_tier() {
    let mut seen = std::collections::BTreeSet::new();
    for provider in [
        KREA_BASE_PROVIDER,
        KREA_RAW_PROVIDER,
        KREA_REALTIME_PROVIDER,
    ] {
        for tier in NUMERIC_TIERS {
            let identity = mlx_production_fingerprint(provider, tier)
                .unwrap_or_else(|| panic!("({provider}, {tier}) must be modelled"));
            assert_ne!(
                identity, RETIRED_KREA_BASE_CALIBRATION_FINGERPRINT,
                "({provider}, {tier}) republishes the retired shared MLX key"
            );
            assert!(
                identity.contains(tier),
                "({provider}, {tier}) must key on its own tier, got {identity}"
            );
            assert!(
                seen.insert(identity.clone()),
                "duplicate identity {identity} across the modelled cells"
            );
        }
    }
    assert_eq!(seen.len(), 9);
    // An unmodelled provider or tier resolves to nothing rather than to a fabricated string.
    assert_eq!(mlx_production_fingerprint("qwen_image", "q4"), None);
    assert_eq!(mlx_production_fingerprint(KREA_RAW_PROVIDER, "q6"), None);
    assert_eq!(mlx_production_fingerprint(KREA_BASE_PROVIDER, "q6"), None);
}

/// The pre-load check refuses a plan row naming an unproducible identity, and is SILENT for a
/// provider the table does not model — so it can never become a blanket gate.
#[test]
fn the_preload_check_refuses_only_a_modelled_providers_wrong_identity() {
    let request = json!({
        "planned": { "calibrationFingerprint": "krea-2-raw-q4-mlx-shared-ladder-v1" }
    });
    validate_planned_fingerprint_is_producible(KREA_RAW_PROVIDER, "q4", &request)
        .expect("the plan's own identity is producible");
    let error = validate_planned_fingerprint_is_producible(KREA_RAW_PROVIDER, "q8", &request)
        .expect_err("a q4 identity must not satisfy a q8 plan row");
    assert!(
        error.contains("krea-2-raw-q8-mlx-shared-ladder-v1"),
        "{error}"
    );
    assert!(error.contains(KREA_RAW_PROVIDER), "{error}");

    // sc-22735 fix pass: Turbo is no longer the silence example — the table models it now, so
    // this row would be REFUSED (it names Raw's identity) and the case would be asserting the
    // opposite of what it claims. Use a provider the table genuinely does not model.
    assert_eq!(mlx_production_fingerprint("qwen_image", "q4"), None);
    validate_planned_fingerprint_is_producible("qwen_image", "q4", &request)
        .expect("the table must stay silent for a provider it does not model");
    // And the silence really is per-provider, not a dead check: the same wrong identity IS
    // refused for Turbo, which the table now models.
    let turbo_error =
        validate_planned_fingerprint_is_producible(KREA_BASE_PROVIDER, "q4", &request)
            .expect_err("a modelled provider must refuse another route's identity");
    assert!(
        turbo_error.contains("krea-2-turbo-q4-mlx-shared-ladder-v1"),
        "{turbo_error}"
    );
}
