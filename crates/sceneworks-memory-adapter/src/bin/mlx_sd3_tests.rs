use super::*;

/// sc-22738. `sd3_5_large` at bf16, q4 and q8 rendered, then had their records refused at
/// `imc-*.quality.rootMeanSquareError must be a nonnegative finite number` — this arm published
/// max/mean only, and the runtime-complete quality contract reads three metrics. The assertion
/// is the shared filing guard rather than a hand-listed field set, so this test states the same
/// contract the harness does instead of a second copy of it.
#[test]
fn the_sd3_quality_block_carries_every_runtime_complete_metric() {
    let quality = sd3_quality(0.004, 0.0005, 0.001);
    protocol::validate_quality_metrics(&json!({
        "status": "runtime_complete",
        "quality": quality.clone(),
    }))
    .expect("the SD3.5 quality block must be fileable as a runtime-complete record");
    assert_eq!(quality["rootMeanSquareError"], json!(0.001));
    assert_eq!(
        quality["rootMeanSquareErrorThreshold"],
        json!(SD3_RMS_THRESHOLD)
    );
}

/// The SD3.5 thresholds are the shared image envelope, and the RMS bound is the maximum one —
/// the same choice `SDXL_RMS_THRESHOLD` makes, and the reason neither arm invents a tolerance
/// ahead of a physical campaign.
#[test]
fn the_sd3_rms_threshold_is_the_shared_maximum_envelope() {
    assert_eq!(SD3_RMS_THRESHOLD, MAX_THRESHOLD);
    assert_eq!(SD3_RMS_THRESHOLD, SDXL_RMS_THRESHOLD);
}

/// The metric is measured by the SHARED helper, not by a second implementation: an identical
/// pair is zero on all three, and a uniform per-channel bias moves max, mean and RMS together.
#[test]
fn the_sd3_metric_helper_is_the_shared_max_mean_rms_comparison() {
    let image = Image {
        width: 2,
        height: 2,
        pixels: vec![0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176],
    };
    let (maximum, mean, rms) = image_max_mean_rms_abs(&image, &image).unwrap();
    assert_eq!((maximum, mean, rms), (0.0, 0.0, 0.0));

    let mutated = qwen_negative_mutation(&image);
    let (maximum, mean, rms) = image_max_mean_rms_abs(&mutated, &image).unwrap();
    for metric in [maximum, mean, rms] {
        assert!(metric.is_finite(), "{metric}");
        assert!(metric > SD3_RMS_THRESHOLD, "{metric}");
    }
    // A uniform bias makes all three the same number; anything else means the RMS accumulator
    // is not walking the same pairs.
    assert!((rms - mean).abs() < 1e-12);
    assert!((maximum - mean).abs() < 1e-12);
}

/// An unusable comparison is a REFUSAL, not a silent metric: the shared helper refuses a shape
/// mismatch and an empty image outright, so no NaN or zero-length division can reach the record.
#[test]
fn the_sd3_metric_helper_refuses_an_uncomparable_pair() {
    let left = Image {
        width: 2,
        height: 1,
        pixels: vec![0, 1, 2, 3, 4, 5],
    };
    let right = Image {
        width: 1,
        height: 1,
        pixels: vec![0, 1, 2],
    };
    assert!(image_max_mean_rms_abs(&left, &right)
        .unwrap_err()
        .contains("image shape mismatch"));

    let empty = Image {
        width: 0,
        height: 0,
        pixels: Vec::new(),
    };
    assert!(image_max_mean_rms_abs(&empty, &empty)
        .unwrap_err()
        .contains("image shape mismatch"));
}

fn sd3_snapshot_root(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
) -> PathBuf {
    fixtures.snapshot(repository, revision, Some(tier))
}

fn sd3_planned(provider: &str, mode: &str, tier: &str, fixture: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": tier,
                "mode": mode,
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 }
            },
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": fixture
        }
    })
}

/// The arm resolves on `(provider, mode)` and refuses every other pair BY NAME. SD3.5 declares
/// an `image_to_image` route on both lanes which no anchor plans and this arm does not
/// implement; measuring it as its nearest text-to-image neighbour would be a mislabelled
/// record, so it is refused rather than silently served.
#[test]
fn the_sd3_arm_resolves_on_provider_and_mode_and_refuses_everything_else() {
    for arm in SD3_ARMS {
        let request = sd3_planned(arm.provider, "text_to_image", "q4", "unused");
        assert_eq!(sd3_arm(&request).unwrap(), arm, "{}", arm.provider);
    }
    for (provider, mode) in [
        (SD3_LARGE_PROVIDER, "image_to_image"),
        (SD3_MEDIUM_PROVIDER, "edit_image"),
        ("sd3_5_enormous", "text_to_image"),
    ] {
        let request = sd3_planned(provider, mode, "q4", "unused");
        let error = sd3_arm(&request).expect_err("an unimplemented pair must be refused");
        assert!(error.contains(provider), "{error} must name {provider}");
        assert!(error.contains(mode), "{error} must name {mode}");
    }
}

/// Each member binds its OWN artifact family, and the root must end in the PLANNED tier's
/// directory: a staged `…/q4` export can never satisfy a q8 or bf16 plan, and one member's
/// repository can never satisfy another's.
#[test]
fn the_sd3_load_spec_binds_the_member_and_the_planned_tier() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for arm in SD3_ARMS {
        for tier in ["q4", "q8", "bf16"] {
            let revision = "0".repeat(40);
            let root = sd3_snapshot_root(fixtures, arm.expected_repository, &revision, tier);
            let fixture = format!("{}-mlx-{tier}-1024-seed{SD3_SEED}-step2", arm.slug);
            let request = sd3_planned(arm.provider, "text_to_image", tier, &fixture);
            let artifact = sd3_load_spec_at(
                &request,
                LoadShape::EagerMaterialization,
                arm.expected_repository.to_owned(),
                revision.clone(),
                root.clone(),
            )
            .unwrap_or_else(|error| panic!("{} {tier}: {error}", arm.provider));
            assert_eq!(artifact.arm, arm, "{} {tier}", arm.provider);
            assert_eq!(artifact.tier, tier, "{}", arm.provider);
            // The worker's still-image shape: Resident, eager, and the load quant settled onto
            // the resolved tier — which is exactly what `mlx-gen-sd3` requires before it will
            // publish a production identity at all (`artifact_tier != spec.quantize` → none).
            assert_eq!(artifact.spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(artifact.spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(artifact.spec.precision, Precision::Bf16);
            assert_eq!(artifact.spec.quantize, tier_precision_quant(tier).1);

            // Another tier's plan against this root is refused.
            for other in ["q4", "q8", "bf16"] {
                if other == tier {
                    continue;
                }
                let other_fixture = format!("{}-mlx-{other}-1024-seed{SD3_SEED}-step2", arm.slug);
                let crossed = sd3_planned(arm.provider, "text_to_image", other, &other_fixture);
                let error = sd3_load_spec_at(
                    &crossed,
                    LoadShape::EagerMaterialization,
                    arm.expected_repository.to_owned(),
                    revision.clone(),
                    root.clone(),
                )
                .expect_err(&format!(
                    "{} must refuse a {other} plan against a {tier} root",
                    arm.provider
                ));
                // The refusal must be ABOUT the crossed tier: `.is_err()` alone would be
                // satisfied by an unrelated failure (a missing subdir, a bad fixture) and
                // would keep passing if the tier check itself were deleted.
                assert!(
                    error.contains(other),
                    "{} refused a {other} plan against a {tier} root for the wrong reason: \
                         {error}",
                    arm.provider
                );
            }

            // A SIBLING member's repository is refused before any weight work.
            for other in SD3_ARMS {
                if other.provider == arm.provider {
                    continue;
                }
                let error = sd3_load_spec_at(
                    &request,
                    LoadShape::EagerMaterialization,
                    other.expected_repository.to_owned(),
                    revision.clone(),
                    root.clone(),
                )
                .expect_err(&format!(
                    "{} must refuse {}'s repository",
                    arm.provider, other.provider
                ));
                // And it must name the repository it was handed, not fail for some other
                // reason that would survive deleting `validate_artifact_identity`.
                assert!(
                    error.contains(other.expected_repository),
                    "{} refused {}'s repository for the wrong reason: {error}",
                    arm.provider,
                    other.provider
                );
            }
        }
    }
}

/// The fixture names the member, tier, geometry edge, seed and step count it measured.
#[test]
fn the_mlx_sd3_fixture_binds_member_tier_edge_seed_and_steps() {
    for arm in SD3_ARMS {
        for tier in ["q4", "q8", "bf16"] {
            let good = format!("{}-mlx-{tier}-1024-seed{SD3_SEED}-step2", arm.slug);
            let request = sd3_planned(arm.provider, "text_to_image", tier, &good);
            assert_eq!(
                planned_sd3_seed(arm, &request, tier, 1024).unwrap(),
                SD3_SEED,
                "{} {tier}",
                arm.provider
            );
            for bad in [
                format!("{}-mlx-{tier}-768-seed{SD3_SEED}-step2", arm.slug),
                format!("{}-mlx-{tier}-1024-seed{SD3_SEED}-step3", arm.slug),
                format!("{}-candle-{tier}-1024-seed{SD3_SEED}-step2", arm.slug),
            ] {
                let request = sd3_planned(arm.provider, "text_to_image", tier, &bad);
                assert!(
                    planned_sd3_seed(arm, &request, tier, 1024).is_err(),
                    "{} {tier} must refuse {bad}",
                    arm.provider
                );
            }
            // A sibling member's fixture is refused: the slug is part of the prefix, and
            // `sd3-5-large` is a PREFIX of `sd3-5-large-turbo`, so the tier token that follows
            // it is what actually separates the two.
            for other in SD3_ARMS {
                if other.provider == arm.provider {
                    continue;
                }
                let crossed = format!("{}-mlx-{tier}-1024-seed{SD3_SEED}-step2", other.slug);
                let request = sd3_planned(arm.provider, "text_to_image", tier, &crossed);
                assert!(
                    planned_sd3_seed(arm, &request, tier, 1024).is_err(),
                    "{} must refuse {}'s fixture {crossed}",
                    arm.provider,
                    other.provider
                );
            }
        }
    }
}

/// Every SD3.5 MLX cell the committed plan declares resolves to an implemented member, and its
/// `calibrationFingerprint` is BYTE-FOR-BYTE the production identity this arm demands of the
/// loaded provider.
///
/// This is the JS↔Rust↔plan binding for the one claim a weights-free host cannot re-derive.
/// At the COMPILED PIN `mlx-gen-sd3` publishes no production identity at all — that is what
/// inference PR 950 adds — so there is no engine constant for CI at the old pin to read, and
/// the arm table is the authority the plan is checked against here. The remaining half of the
/// binding, plan-string == LOADED-contract-string, is enforced inside `run_sd3`, which refuses
/// a provider whose published identity differs from `sd3_calibration_fingerprint(arm, artifact.tier)`.
#[test]
fn every_planned_sd3_mlx_cell_is_priced_at_the_arms_production_identity() {
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
        let Some(arm) = SD3_ARMS.into_iter().find(|arm| arm.provider == provider) else {
            continue;
        };
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "backend": "mlx",
            "target": {
                "provider": provider,
                "modelId": provider,
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
        assert_eq!(sd3_arm(&request).unwrap(), arm, "{key}");
        assert_eq!(planned_qwen_tier(&request).unwrap(), tier, "{key}");
        assert_eq!(
            entry["calibrationFingerprint"].as_str().unwrap(),
            sd3_calibration_fingerprint(arm, tier),
            "{key}"
        );
        // The MLX anchor composition is the resident/eager shape the worker loads.
        assert_eq!(
            entry["loadShape"].as_str().unwrap(),
            "eager_materialization",
            "{key}"
        );
        let (width, _) = protocol::target_geometry(&request).unwrap();
        assert_eq!(
            planned_sd3_seed(arm, &request, tier, width).unwrap(),
            SD3_SEED,
            "{key}"
        );
    }
    let expected: std::collections::BTreeSet<String> =
        ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
            .iter()
            .flat_map(|model| {
                ["bf16", "q4", "q8"]
                    .iter()
                    .map(move |tier| format!("{model}:{tier}:mlx"))
            })
            .collect();
    assert_eq!(seen, expected);
}
