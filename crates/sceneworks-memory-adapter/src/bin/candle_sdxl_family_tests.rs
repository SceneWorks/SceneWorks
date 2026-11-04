use super::*;

// CI_IMPACT: pure strings and injected monotonic clock; no process, model or device execution.
#[test]
fn sdxl_endpoint_device_binding_requires_one_physical_index_and_canonical_uuid() {
    let uuid = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
    for index in ["0", "1"] {
        assert!(validate_sdxl_endpoint_binding(index, index, uuid).is_ok());
    }
    for (visible, physical, expected) in [
        ("0", "1", uuid),
        ("1", "0", uuid),
        ("0,1", "0", uuid),
        ("2", "2", uuid),
        ("1", "1", "GPU-other"),
        ("1", "1", "1"),
        ("1", "1", "GPU-e4b799317be6-f216-460a-f5405cfafffe"),
    ] {
        assert!(validate_sdxl_endpoint_binding(visible, physical, expected).is_err());
    }
}

fn fake_sdxl_idle(rows: &[&str]) -> (Result<Vec<u64>, String>, usize) {
    let clock = std::cell::Cell::new(Duration::ZERO);
    let calls = std::cell::Cell::new(0usize);
    let result = wait_for_sdxl_idle(
        |remaining| {
            assert!(remaining > Duration::ZERO && remaining <= Duration::from_secs(5));
            let index = calls.get();
            calls.set(index + 1);
            Ok(rows[index.min(rows.len() - 1)].to_owned())
        },
        || clock.get(),
        |duration| clock.set(clock.get() + duration),
    );
    (result, calls.get())
}

#[test]
fn sdxl_endpoint_idle_wait_accepts_counter_decay_and_resets_memory_stability() {
    let (idle, calls) = fake_sdxl_idle(&["2048,96000,0,0"]);
    assert_eq!(idle.unwrap(), vec![2048; 5]);
    assert_eq!(calls, 5);
    let (decay, calls) = fake_sdxl_idle(&["2048,96000,1,1", "2048,96000,0,1", "2048,96000,0,0"]);
    assert_eq!(decay.unwrap(), vec![2048; 5]);
    assert_eq!(calls, 7);
    let (drift, calls) = fake_sdxl_idle(&["2048,96000,0,0", "2176,96000,0,0"]);
    assert_eq!(drift.unwrap(), vec![2176; 5]);
    assert_eq!(calls, 6);
}

#[test]
fn sdxl_endpoint_idle_wait_rejects_invalid_telemetry_and_times_out_with_samples() {
    for invalid in [
        "2048,96000,0",
        "2048,96000,NaN,0",
        "2048,96000,101,0",
        "2048,96000,0,101",
        "-1,96000,0,0",
    ] {
        let (result, calls) = fake_sdxl_idle(&[invalid]);
        let error = result.unwrap_err();
        assert!(error.contains("invalid activity telemetry") && error.contains(invalid));
        assert_eq!(calls, 1);
    }
    let unstable: Vec<_> = (0..150)
        .map(|index| {
            if index % 2 == 0 {
                "2048,96000,0,0"
            } else {
                "2176,96000,0,0"
            }
        })
        .collect();
    for rows in [vec!["2048,96000,1,1"], vec!["2048,15000,0,0"], unstable] {
        let (result, calls) = fake_sdxl_idle(&rows);
        let error = result.unwrap_err();
        assert!(error.contains("within 30s") && error.contains(rows[0]));
        assert_eq!(calls, 150);
    }
    let clock = std::cell::Cell::new(Duration::ZERO);
    let calls = std::cell::Cell::new(0usize);
    let late = wait_for_sdxl_idle(
        |_| {
            calls.set(calls.get() + 1);
            if calls.get() == 5 {
                clock.set(Duration::from_secs(31));
            }
            Ok("2048,96000,0,0".to_owned())
        },
        || clock.get(),
        |duration| clock.set(clock.get() + duration),
    );
    assert!(late.unwrap_err().contains("within 30s"));
}

#[test]
fn kolors_capture_revision_passes_real_provider_admission() {
    use runtime_cuda::providers::kolors::memory_strategy as kolors;
    for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/__unused_kolors_fixture__".into()));
        spec.quantize = quant;
        let contract = kolors::weights_free_contract(&spec).unwrap();
        let fixtures =
            kolors::registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap();
        assert!(!fixtures.is_empty());
        for fixture in fixtures {
            let mut context = fixture.context;
            context.evidence_revision = five_rung_evidence_revision(KOLORS_ID);
            assert!(matches!(
                kolors::registered_safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Accept
            ));
            context.evidence_revision = format!("sc-22732@{}", protocol::INFERENCE_PIN);
            assert!(matches!(
                kolors::registered_safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }
}

#[test]
fn sd3_capture_revision_passes_real_provider_admission() {
    use runtime_cuda::providers::sd3::memory_strategy as sd3;

    for provider in [SD3_5_LARGE_ID, SD3_5_LARGE_TURBO_ID, SD3_5_MEDIUM_ID] {
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/__unused_sd3_fixture__".into()));
            spec.quantize = quant;
            let contract = sd3::weights_free_contract(provider, &spec).unwrap();
            let fixtures =
                sd3::registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                    .unwrap();
            assert!(!fixtures.is_empty());
            for fixture in fixtures {
                let exact = fixture.load_spec.as_ref().unwrap();
                let mut context = fixture.context;
                context.evidence_revision = five_rung_evidence_revision(provider);
                assert!(matches!(
                    sd3::registered_safety_check(exact, &contract, &context),
                    MemorySafetyDecision::Accept
                ));
                context.evidence_revision = format!("sc-22730@{}", protocol::INFERENCE_PIN);
                assert!(matches!(
                    sd3::registered_safety_check(exact, &contract, &context),
                    MemorySafetyDecision::Reject { .. }
                ));
            }
        }
    }
}

#[test]
fn sana_requests_carry_the_providers_evidence_token() {
    for provider in [SANA_ID, SANA_SPRINT_ID] {
        assert_eq!(
            five_rung_evidence_revision(provider),
            runtime_cuda::providers::sana::memory_strategy::REQUEST_EVIDENCE_REVISION
        );
    }
    assert_eq!(
        five_rung_evidence_revision(QWEN_ID),
        format!(
            "{}@{}",
            five_rung_evidence_story(QWEN_ID),
            protocol::INFERENCE_PIN
        )
    );
}

#[test]
fn bernini_still_plan_builds_an_explicit_advertised_single_frame_request() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .unwrap();
    for tier in ["bf16", "q4", "q8"] {
        let row = &plan["anchors"][format!("bernini_image:{tier}:candle")];
        let width = row["geometry"]["width"].as_u64().unwrap() as u32;
        let height = row["geometry"]["height"].as_u64().unwrap() as u32;
        assert_eq!((width, height), (512, 512));
        let generation =
            five_rung_provider_generation_request(BERNINI_CANDLE_ID, width, height, false);
        assert_eq!(generation.video_mode.as_deref(), Some("t2i"));
        assert_eq!(generation.frames, Some(1));
        assert_eq!(generation.image_reference_count(), 0);
        assert!(row["fixture"]
            .as_str()
            .unwrap()
            .contains("-512-seed16402-step2"));
    }
}

#[test]
fn instantid_measured_fragment_preserves_measurements_without_claiming_promotion() {
    let strategy = json!({ "rung": "resident", "engagedRungs": ["resident"], "parameters": {} });
    let request = json!({ "planned": { "strategy": strategy } });
    let fragment = instantid_measured_fragment(
        &request,
        strategy,
        "eager_materialization",
        artifact("fixture/repo", "1234567", "bf16"),
        "fixture+identity:abc".to_owned(),
        [10, 20, 30, 40, 50, 60],
    )
    .unwrap();
    assert_eq!(fragment["status"], "gated");
    assert_eq!(fragment["quality"]["result"], "not_run");
    assert_eq!(fragment["sweep"]["cases"][0]["result"], "passed");
    assert_eq!(fragment["sweep"]["rangeVerified"], false);
    assert_eq!(fragment["observedMemory"]["overall"]["activeBytes"], 60);
    for scenario in fragment["scenarios"].as_array().unwrap() {
        if ["warm_repeat", "cancel", "error"].contains(&scenario["name"].as_str().unwrap()) {
            assert_eq!(scenario["result"], "not_run");
        }
    }
    let schema: Value = serde_json::from_str(include_str!(
        "../../../../packages/schemas/memory-calibration.schema.json"
    ))
    .unwrap();
    let properties = &schema["$defs"]["record"]["properties"];
    assert!(properties["status"]["enum"]
        .as_array()
        .unwrap()
        .contains(&fragment["status"]));
    for key in ["artifact", "quality", "sweep"] {
        let value = fragment[key].as_object().expect(key);
        for field in value.keys() {
            assert!(
                properties[key]["properties"].get(field).is_some(),
                "{key}.{field}"
            );
        }
        if let Some(required) = properties[key]["required"].as_array() {
            for field in required {
                assert!(value.contains_key(field.as_str().unwrap()), "{key}.{field}");
            }
        }
    }
}

#[test]
fn pulid_capture_uses_the_known_face_fixture() {
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../fixtures/pulid-reference.png"))
        ),
        "3995f2e856346748588e76a5557516d0218f44f5663701c5a50139d50c86a7be"
    );
    let reference = pulid_reference_image().unwrap();
    assert_eq!((reference.width, reference.height), (1024, 1024));
    assert_eq!(reference.pixels.len(), 1024 * 1024 * 3);
    assert_ne!(
        reference.pixels,
        protocol::synthetic_reference_rgb(1024, 1024)
    );
    let request = json!({"planned": {"target": {"geometry": {"width": 1024}},
            "fixture": "pulid-flux-candle-q4-1024-seed22726-step2"}});
    assert!(validate_flux_one_fixture(&request, PULID_FLUX_ID, "q4").is_err());
}

fn planned(provider: &str, model_id: &str) -> Value {
    json!({ "planned": { "target": { "provider": provider, "modelId": model_id } } })
}

/// sc-22729: the member is read off `modelId`, and a model id no member serves is refused BY
/// NAME. All five load through the same `sdxl` provider, so the engine id alone cannot select
/// an artifact family.
#[test]
fn sdxl_arm_is_resolved_from_the_plans_model_id() {
    for (model_id, repository) in [
        ("sdxl", protocol::SDXL_REPOSITORY),
        ("realvisxl", protocol::REALVISXL_REPOSITORY),
        (
            "realvisxl_lightning",
            protocol::REALVISXL_LIGHTNING_REPOSITORY,
        ),
        ("illustrious_xl_v1", protocol::ILLUSTRIOUS_XL_V1_REPOSITORY),
        ("illustrious_xl_v2", protocol::ILLUSTRIOUS_XL_V2_REPOSITORY),
    ] {
        let arm = sdxl_candle_arm(&planned(SDXL_ID, model_id)).unwrap();
        assert_eq!(arm.model_id, model_id);
        assert_eq!(arm.expected_repository, repository);
        // sc-22738: DERIVED from the linked engine crate, never transcribed here — so this
        // asserts the lookup resolves, not that a literal in this file is 40 characters long.
        let route_revision = sdxl_candle_route_revision(&arm).expect(model_id);
        assert_eq!(route_revision.len(), 40, "{model_id} route revision");
    }
    // Every member's env family and execution path is its own.
    for field in [
        SDXL_CANDLE_FAMILY.map(|arm| arm.root_env).to_vec(),
        SDXL_CANDLE_FAMILY.map(|arm| arm.repository_env).to_vec(),
        SDXL_CANDLE_FAMILY.map(|arm| arm.revision_env).to_vec(),
        SDXL_CANDLE_FAMILY
            .map(|arm| arm.expected_repository)
            .to_vec(),
        SDXL_CANDLE_FAMILY.map(|arm| arm.execution_path).to_vec(),
        SDXL_CANDLE_FAMILY.map(|arm| arm.still_calibration).to_vec(),
        SDXL_CANDLE_FAMILY
            .map(|arm| sdxl_candle_route_revision(&arm).expect(arm.model_id))
            .to_vec(),
    ] {
        let mut unique = field.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), field.len(), "collision in {field:?}");
    }
    for absent in ["instantid_realvisxl", "sdxl_turbo", "z_image"] {
        let error = sdxl_candle_arm(&planned(SDXL_ID, absent)).unwrap_err();
        assert!(error.contains(&format!("modelId {absent:?}")), "{error}");
    }
}

/// sc-22729: the fingerprints the plan may cite are exactly the strings `candle-gen-sdxl`
/// mints, including the `v1 -> rev1` rewrite that keeps a single version token in an id that
/// already ends in one. `realvisxl` and `realvisxl_lightning` are the pair a PREFIX comparison
/// would confuse, which is why the whole string is the identity.
#[test]
fn candle_route_fingerprints_are_the_engines_own_strings() {
    for (model_id, expected) in [
        ("sdxl", "sdxl-candle-sdxl-staged-decode-attention-v3"),
        (
            "realvisxl",
            "sdxl-candle-realvisxl-staged-decode-attention-v3",
        ),
        (
            "realvisxl_lightning",
            "sdxl-candle-realvisxl-lightning-staged-decode-attention-v3",
        ),
        (
            "illustrious_xl_v1",
            "sdxl-candle-illustrious-xl-rev1-staged-decode-attention-v3",
        ),
        (
            "illustrious_xl_v2",
            "sdxl-candle-illustrious-xl-rev2-staged-decode-attention-v3",
        ),
    ] {
        assert_eq!(sdxl_candle_route_fingerprint(model_id), expected);
        // Exactly one `vN` token, which is what `validate_calibration_fingerprint` requires.
        let versions = expected
            .split('-')
            .filter(|token| {
                token.strip_prefix('v').is_some_and(|rest| {
                    !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
                })
            })
            .count();
        assert_eq!(versions, 1, "{expected}");
    }
    assert_ne!(
        sdxl_candle_route_fingerprint("realvisxl"),
        sdxl_candle_route_fingerprint("realvisxl_lightning"),
        "the lightning route must never resolve to the edit-capable realvisxl route"
    );
}

/// sc-22729: every planned SDXL-family candle row cites a fingerprint the engine can actually
/// emit, and every planned MLX row cites the one route-independent string `mlx-gen-sdxl` mints.
/// A plan row carrying a hand-written label would never match the loaded contract.
#[test]
fn every_planned_sdxl_row_cites_an_emittable_fingerprint() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    let anchors = plan["anchors"].as_object().expect("anchors is an object");
    let mut checked: Vec<(String, String)> = Vec::new();
    for (key, anchor) in anchors {
        if anchor["provider"] != json!(SDXL_ID) {
            continue;
        }
        let (model_id, rest) = key.split_once(':').expect("anchor key shape");
        let backend = rest.rsplit(':').next().expect("anchor key backend");
        let fingerprint = anchor["calibrationFingerprint"]
            .as_str()
            .expect("calibrationFingerprint is a string");
        let expected = match backend {
            "mlx" => "sdxl-mlx-unet-shared-ladder-v5".to_owned(),
            _ => sdxl_candle_route_fingerprint(model_id),
        };
        assert_eq!(fingerprint, expected, "{key}");
        assert!(
            SDXL_CANDLE_FAMILY
                .iter()
                .any(|arm| arm.model_id == model_id),
            "{key} names a model no family member serves"
        );
        checked.push((model_id.to_owned(), backend.to_owned()));
    }
    // Not a floor: the EXACT set of `(modelId, lane)` pairs the family plans. A floor stayed
    // green with a whole member's lane deleted from the plan, which is the one thing this case
    // is here to notice — every family member is planned on BOTH lanes (the candle lane routes
    // all five, `routing/candle.rs` `is_sdxl_family_candle_model` / `SDXL_CONTROL_MODELS`), and
    // an engine defect that blocks a CAPTURE never removes the DECLARATION.
    checked.sort();
    checked.dedup();
    let expected = SDXL_CANDLE_FAMILY
        .iter()
        .flat_map(|arm| {
            ["candle", "mlx"]
                .into_iter()
                .map(|backend| (arm.model_id.to_owned(), backend.to_owned()))
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(checked, expected);
}

/// sc-22729: the candle InstantID lane is dense-only, so a packed tier is refused before any
/// environment work — it could only ever measure bf16 weights.
#[test]
fn candle_instantid_refuses_a_packed_tier() {
    for tier in ["q4", "q8"] {
        let error = match instantid_candle_binding(tier) {
            Err(error) => error,
            Ok(_) => panic!("{tier} must be refused before any environment work"),
        };
        assert!(error.contains("dense-only"), "{tier}: {error}");
        assert!(error.contains(INSTANTID_CANDLE_TIER), "{tier}: {error}");
    }
}

use protocol::staging::StagedInstantId;

fn digest_of(paths: &[PathBuf]) -> String {
    instantid_artifact_fingerprint(&paths.iter().map(PathBuf::as_path).collect::<Vec<_>>())
}

/// sc-22729 review: the identity arm driven PAST the tier guard on a synthetic tree.
///
/// Until this existed, the tier guard's refusal was the only thing any test reached, so
/// deleting `protocol::instantid_controlnet_dir()?` — or a file out of the identity bundle, or
/// a path out of the digest — reddened nothing at all.
#[test]
fn instantid_candle_identity_names_the_missing_identitynet_weight() {
    let staged = StagedInstantId::install(
        "candle-instantid",
        protocol::REALVISXL_REPOSITORY,
        INSTANTID_CANDLE_TIER,
    );
    let error = match instantid_candle_identity(INSTANTID_CANDLE_TIER) {
        Err(error) => error,
        Ok(_) => panic!("an IdentityNet directory carrying no weight file must be refused"),
    };
    assert!(
        error.contains(protocol::INSTANTID_CONTROLNET_WEIGHT_FILE),
        "{error}"
    );
    assert!(
        error.contains(&staged.identitynet.display().to_string())
            || error.contains(
                &std::fs::canonicalize(&staged.identitynet)
                    .unwrap()
                    .display()
                    .to_string()
            ),
        "the refusal must name the directory it looked in: {error}"
    );

    staged.stage_identitynet_weight();
    let identity = match instantid_candle_identity(INSTANTID_CANDLE_TIER) {
        Ok(identity) => identity,
        Err(error) => panic!("a complete staged tree must bind: {error}"),
    };
    assert_eq!(identity.repository, protocol::REALVISXL_REPOSITORY);
    assert_eq!(identity.revision, StagedInstantId::REVISION);

    // The digest covers EXACTLY these five, in exactly this order: a permutation and every
    // four-path subset both differ.
    let ordered = staged.ordered_paths();
    assert_eq!(identity.artifact_fingerprint, digest_of(&ordered));
    let mut swapped = ordered.clone();
    swapped.swap(3, 4);
    assert_ne!(
        identity.artifact_fingerprint,
        digest_of(&swapped),
        "the digest must be order-sensitive across the face stack"
    );
    for dropped in 0..ordered.len() {
        let mut short = ordered.clone();
        short.remove(dropped);
        assert_ne!(
            identity.artifact_fingerprint,
            digest_of(&short),
            "path {dropped} is not covered by the digest"
        );
    }
}

/// sc-22729: the artifact digest is over the ORDERED paths with a record separator, the axis the
/// provider's `overlay_key()` carries.
#[test]
fn instantid_artifact_fingerprint_is_ordered_and_separated() {
    let a = std::path::Path::new("/w/a");
    let b = std::path::Path::new("/w/b");
    assert_ne!(
        instantid_artifact_fingerprint(&[a, b]),
        instantid_artifact_fingerprint(&[b, a])
    );
    assert_ne!(
        instantid_artifact_fingerprint(&[std::path::Path::new("/ab"), std::path::Path::new("/c")]),
        instantid_artifact_fingerprint(&[std::path::Path::new("/a"), std::path::Path::new("/bc")])
    );
    let digest = instantid_artifact_fingerprint(&[a, b]);
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(!digest.contains('='));
}
