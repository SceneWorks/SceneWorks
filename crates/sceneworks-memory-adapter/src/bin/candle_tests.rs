use super::*;

#[test]
fn bernini_reference_probe_passes_the_provider_scope_and_binds_its_image() {
    use runtime_cuda::gen_core::{MemoryBehaviorRoute, MemoryNumericTier};
    use runtime_cuda::providers::bernini::memory_strategy as bernini;
    let arm = BERNINI_CANDLE_VIDEO_ARM;
    for (tier, quant) in [
        ("bf16", None),
        ("q4", Some(Quant::Q4)),
        ("q8", Some(Quant::Q8)),
    ] {
        let spec = bernini_candle_load_spec(
            LoadSpec::new(WeightsSource::Dir(PathBuf::from("unused"))),
            tier,
            arm.model_id,
        )
        .unwrap();
        let contract =
            bernini::weights_free_memory_strategy_contract(arm.engine_id, &spec).unwrap();
        let target = Sc22737VideoTarget {
            tier: tier.to_owned(),
            width: 848,
            height: 480,
            frames: 45,
            fps: 16,
            seed: SC22737_CANDLE_SEED,
        };
        let mut generation = sc22737_generation_request(arm, &target);
        let overlay = sc22737_conditioning_overlay(arm, &generation, &contract).unwrap();
        let numeric = MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        };
        let mut context = runtime_cuda::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            numeric,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other(arm.mode.to_owned()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay,
            },
        )
        .unwrap();
        context.geometry.width = target.width;
        context.geometry.height = target.height;
        context.geometry.frames = target.frames;
        assert!(matches!(
            bernini::registered_safety_check(&spec, &contract, &context),
            runtime_cuda::gen_core::MemorySafetyDecision::Accept
        ));
        let mut scope =
            bernini::registered_begin_request(arm.engine_id, &spec, &contract, &context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut generation).unwrap();
        if let Conditioning::MultiReference { images } = &mut generation.conditioning[0] {
            images[0].pixels[0] ^= 1;
        }
        assert!(scope.configure_request(&mut generation).is_err());
        context.geometry.frames = 49;
        assert!(matches!(
            bernini::registered_safety_check(&spec, &contract, &context),
            runtime_cuda::gen_core::MemorySafetyDecision::Reject { .. }
        ));
    }
}

#[test]
fn ltx_probe_loads_the_declared_packed_tier_and_fits_one_reference() {
    for (tier, quant) in [("q4", Quant::Q4), ("q8", Quant::Q8)] {
        let spec =
            ltx23_candle_load_spec(PathBuf::from(tier), PathBuf::from("gemma"), tier).unwrap();
        assert_eq!(spec.quantize, Some(quant));
        let target = Sc22737VideoTarget {
            tier: tier.to_owned(),
            width: 768,
            height: 512,
            frames: 97,
            fps: 24,
            seed: SC22737_CANDLE_SEED,
        };
        let generation = sc22737_generation_request(LTX23_CANDLE_ARM, &target);
        let [Conditioning::Reference { image, strength }] = generation.conditioning.as_slice()
        else {
            panic!("LTX must carry one fitted reference")
        };
        assert_eq!(
            (image.width, image.height, image.pixels.len()),
            (768, 512, 768 * 512 * 3)
        );
        assert_eq!(strength.unwrap_or(1.0), 1.0);
    }
    assert!(ltx23_candle_load_spec(PathBuf::new(), PathBuf::new(), "bf16").is_err());
}

/// sc-22738: every plan row an sc-22737 video arm serves declares the load shape that arm
/// stages. The engines behind these arms execute exactly one shape (see
/// `Sc22737VideoArm::load_shape`), and the only production check — the plan/provider
/// re-assert in `run_sc22737_video_capture` — fires after a real load on a CUDA host. sc-22737
/// shipped all fourteen rows as `deferred_materialization` against eager-only engines, so
/// none was capturable and nothing weights-free said so.
#[test]
fn sc22737_plan_rows_declare_the_load_shape_their_arm_stages() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("memory calibration plan parses");
    let anchors = plan["anchors"].as_object().expect("plan anchors object");
    for arm in SC22737_VIDEO_ARMS {
        let expected = match arm.load_shape {
            LoadShape::EagerMaterialization => protocol::LOAD_SHAPE_EAGER,
            LoadShape::DeferredMaterialization => protocol::LOAD_SHAPE_DEFERRED,
        };
        let mut rows = 0_usize;
        for (key, row) in anchors {
            let mut coordinates = key.split(':');
            let (Some(model_id), Some(_tier), Some("candle"), None) = (
                coordinates.next(),
                coordinates.next(),
                coordinates.next(),
                coordinates.next(),
            ) else {
                continue;
            };
            if model_id != arm.model_id {
                continue;
            }
            assert_eq!(
                row["provider"].as_str(),
                Some(arm.engine_id),
                "{key}: filed under {} but names another engine",
                arm.model_id
            );
            assert_eq!(
                row["loadShape"].as_str(),
                Some(expected),
                "{key}: the {} arm stages {expected}, which is the only shape its engine \
                     executes; a row declaring another shape cannot be captured",
                arm.engine_id
            );
            if matches!(arm.engine_id, BERNINI_CANDLE_ID | LTX23_CANDLE_ID) {
                let tier = key.split(':').nth(1).unwrap();
                let request = json!({"planned": {"target": {
                        "provider": arm.engine_id, "modelId": arm.model_id, "tier": tier,
                        "mode": row["mode"], "referenceCount": row["referenceCount"],
                        "geometry": row["geometry"] }, "fixture": row["fixture"] }});
                assert!(sc22737_video_target(&request, arm).is_ok(), "{key}");
                let mut legacy = request.clone();
                legacy["planned"]["target"]["mode"] = json!("text_to_video");
                assert!(sc22737_video_target(&legacy, arm).is_err());
                let mut missing_count = request;
                missing_count["planned"]["target"]
                    .as_object_mut()
                    .unwrap()
                    .remove("referenceCount");
                assert!(sc22737_video_target(&missing_count, arm).is_err());
            }
            rows += 1;
        }
        assert!(
            rows > 0,
            "the plan carries no candle row for {} — the arm has nothing to serve",
            arm.model_id
        );
    }
}

#[test]
fn candle_krea_wddm_idle_proof_keeps_the_measured_strict_bounds() {
    let config = certifying_wddm_idle_config();
    assert_eq!(config.max_baseline_gb, 2.0);
    assert_eq!(config.sample_count, 5);
    assert_eq!(config.max_drift_mib, 64);
    assert_eq!(config.sample_interval_ms, 200);
}

#[test]
fn candle_krea_quality_uses_normalized_channel_error_and_the_approved_mean_bound() {
    let golden = runtime_cuda::gen_core::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 128, 255],
    };
    let within = runtime_cuda::gen_core::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 128, 250],
    };
    let (maximum, mean) = pixel_error(&golden, &within).unwrap();
    assert_eq!(maximum, 5.0 / 255.0);
    assert_eq!(mean, 5.0 / (3.0 * 255.0));
    ensure_krea_quality(maximum, mean, "golden policy").unwrap();

    let above_mean = runtime_cuda::gen_core::Image {
        width: 1,
        height: 1,
        pixels: vec![6, 134, 249],
    };
    let (maximum, mean) = pixel_error(&golden, &above_mean).unwrap();
    assert!(maximum < KREA_CANDLE_MAX_THRESHOLD);
    assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
    assert!(ensure_krea_quality(maximum, mean, "regression").is_err());
    assert!(ensure_krea_quality(1.0, 0.01681, "exact boundary").is_ok());
    assert!(ensure_krea_quality(1.0, 0.01681 + 1e-12, "mean above boundary").is_err());
    assert!(ensure_krea_quality(1.0 + 1e-12, 0.01681, "maximum above boundary").is_err());
}

#[test]
fn candle_krea_negative_mutation_breaches_the_certifying_mean_envelope() {
    let image = runtime_cuda::gen_core::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 128, 255],
    };
    let mutated = negative_mutation(&image);
    assert_eq!(mutated.pixels, vec![64, 192, 63]);
    let (maximum, mean) = pixel_error(&image, &mutated).unwrap();
    assert!(maximum > KREA_CANDLE_MEAN_THRESHOLD);
    assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
}

#[test]
fn candle_krea_complete_sweep_certifies_only_the_v1_production_tuple() {
    let request = json!({
        "planned": {
            "calibrationFingerprint": "krea-turbo-cuda-phase-curves-v1"
        }
    });
    let parameters = Map::from_iter([
        ("decodeTileEdge".to_owned(), json!(512)),
        ("decodeOverlap".to_owned(), json!(128)),
        ("attentionChunkSize".to_owned(), json!(134_217_728)),
        ("transformerWindowSize".to_owned(), json!(1)),
    ]);
    let sweep = complete_sweep(&request, &parameters).unwrap();
    assert_eq!(sweep["rangeVerified"], json!(true));
    assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
    assert_eq!(sweep["cases"][0]["result"], json!("passed"));
    assert_eq!(sweep["cases"][0]["parameters"], json!(parameters));
}

fn qwen_request() -> Value {
    json!({
        "planned": {
            "target": { "provider": "qwen_image", "overlay": "none" },
            "strategy": { "rung": "resident", "parameters": {} },
            "loadShape": "deferred_materialization"
        }
    })
}

#[test]
fn qwen_plan_routes_to_the_qwen_base_execution_path() {
    let request = qwen_request();
    assert_eq!(planned_provider(&request).unwrap(), "qwen_image");
    assert_eq!(
        plain_execution_path(&request).unwrap(),
        QWEN_PLAIN_EXECUTION_PATH
    );
    assert_eq!(
        planned_memory_strategy(&request).unwrap(),
        MemoryStrategy::Resident
    );
}

/// The edit provider must never be measured through the five-rung txt2img path. sc-22728 gave it
/// its own arm (`run_qwen_edit`, dispatched ahead of this one), and the five-rung route still
/// refuses it by name — which is what keeps a mis-ordered dispatch a refusal rather than a
/// txt2img record wearing an edit label.
#[test]
fn the_edit_provider_is_never_served_by_the_five_rung_txt2img_path() {
    let mut request = qwen_request();
    request["planned"]["target"]["provider"] = json!("qwen_image_edit");
    let error = plain_execution_path(&request).unwrap_err();
    assert!(error.contains("qwen_image_edit"));
    assert!(error.contains("does not implement"));
    assert!(still_calibration_label(&request).is_err());
    assert!(load_five_rung_generator(&request).is_err());
}

fn qwen_edit_planned(model_id: &str, tier: &str, overlay: &str, fixture: &str) -> Value {
    json!({
        "planned": {
            "target": {
                "provider": "qwen_image_edit",
                "modelId": model_id,
                "tier": tier,
                "mode": "edit_image",
                "overlay": overlay,
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

/// sc-22728: the two shipped edit catalog ids are ONE engine provider, so only the model id
/// separates them; the arm is resolved from `(provider, modelId)` and any other pair is refused
/// by name rather than measured as its neighbour.
#[test]
fn the_candle_qwen_edit_arm_is_resolved_from_the_plans_provider_and_model_id() {
    let base = qwen_edit_arm(&qwen_edit_planned(
        "qwen_image_edit_2511",
        "q4",
        "none",
        "qwen-edit-candle-q4-seed15817-step2",
    ))
    .unwrap();
    assert_eq!(base, QWEN_EDIT_ARM);
    assert!(!base.lightning);
    assert_eq!(base.overlay, "none");
    let lightning = qwen_edit_arm(&qwen_edit_planned(
        "qwen_image_edit_2511_lightning",
        "q4",
        "lora",
        "qwen-edit-lightning-candle-q4-seed15817-step4",
    ))
    .unwrap();
    assert_eq!(lightning, QWEN_EDIT_LIGHTNING_ARM);
    assert!(lightning.lightning);
    assert_eq!(lightning.overlay, "lora");
    assert_eq!(lightning.steps, 4, "the official lightx2v 4-step recipe");
    assert_ne!(base.slug, lightning.slug, "one diagnostics source each");
    for (provider, model_id) in [
        ("qwen_image_edit", "qwen_image_edit_2509"),
        ("qwen_image", "qwen_image_edit_2511"),
    ] {
        let mut request = qwen_edit_planned(model_id, "q4", "none", "unused");
        request["planned"]["target"]["provider"] = json!(provider);
        let error = qwen_edit_arm(&request).unwrap_err();
        assert!(
            error.contains(&format!("provider {provider:?} for model {model_id:?}")),
            "{provider}/{model_id}: {error}"
        );
    }
}

/// sc-22728: the fixture binds the member, the tier and the recipe's step count, so a 4-step
/// distilled capture can never be recorded under the 2-step production fixture or another tier's.
#[test]
fn the_candle_qwen_edit_fixture_binds_the_member_the_tier_and_the_step_count() {
    for (arm, model_id, prefix, steps) in [
        (
            QWEN_EDIT_ARM,
            "qwen_image_edit_2511",
            "qwen-edit-candle",
            2_usize,
        ),
        (
            QWEN_EDIT_LIGHTNING_ARM,
            "qwen_image_edit_2511_lightning",
            "qwen-edit-lightning-candle",
            4,
        ),
    ] {
        for tier in ["q4", "q8", "bf16"] {
            let fixture = format!("{prefix}-{tier}-seed15817-step{steps}");
            let request = qwen_edit_planned(model_id, tier, arm.overlay, &fixture);
            assert_eq!(planned_qwen_edit_seed(&request, arm, tier).unwrap(), 15817);
            let other = if tier == "q4" { "q8" } else { "q4" };
            let error = planned_qwen_edit_seed(&request, arm, other).unwrap_err();
            assert!(error.contains(&format!("{prefix}-{other}-seed")), "{error}");
        }
        let wrong_steps = if steps == 2 { 4 } else { 2 };
        let fixture = format!("{prefix}-q4-seed15817-step{wrong_steps}");
        let error = planned_qwen_edit_seed(
            &qwen_edit_planned(model_id, "q4", arm.overlay, &fixture),
            arm,
            "q4",
        )
        .unwrap_err();
        assert!(error.contains(&format!("{steps}-step")), "{error}");
    }
}

const QWEN_EDIT_TEST_REVISION: &str = "bb2bc9893b3c49ae96c813350775f791a2e8bc80";

#[test]
fn qwen_edit_generation_uses_the_lightning_recipe_only_for_the_distilled_arm() {
    let lightning = qwen_edit_generation_recipe(QWEN_EDIT_LIGHTNING_ARM);
    assert!(lightning.lightning);
    assert_eq!(lightning.steps, 4);
    assert_eq!(lightning.guidance, 1.0);
    assert!(lightning.negative.is_empty());

    let base = qwen_edit_generation_recipe(QWEN_EDIT_ARM);
    assert!(!base.lightning);
    assert_eq!(base.steps, 2);
    assert_eq!(base.guidance, 4.0);
    assert_eq!(base.prompt, lightning.prompt);
}

fn qwen_edit_source(
    fixtures: &crate::adapter_test_support::Fixtures,
    arm: QwenEditArm,
    tier: &str,
) -> QwenEditArtifactSource {
    fixtures.qwen_edit_source(arm.lightning, tier, QWEN_EDIT_TEST_REVISION)
}

/// sc-22728: the ONE thing that makes the Lightning member Lightning at load time. Every
/// downstream claim — the record's `builtInAdapters` count, the gated fragment's overlay
/// selection, and the run context's `overlay` (which `validate_edit_route` pins against the
/// load's own adapter stack) — is read off `spec.adapters`, so if this attachment were lost the
/// arm would publish an authoritative record asserting a distill that never participated. The
/// stack itself is therefore asserted here, on BOTH members and at every shipped tier.
#[test]
fn the_candle_qwen_edit_load_spec_carries_the_distill_on_exactly_the_lightning_member() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for (arm, model_id) in QWEN_EDIT_MEMBERS {
        for (tier, quant) in [
            ("q4", Some(Quant::Q4)),
            ("q8", Some(Quant::Q8)),
            ("bf16", None),
        ] {
            let source = qwen_edit_source(fixtures, arm, tier);
            let artifact =
                qwen_edit_load_spec(arm, tier, &source, LoadShape::DeferredMaterialization)
                    .unwrap_or_else(|error| panic!("{model_id}/{tier}: {error}"));
            assert_eq!(
                artifact.spec.quantize, quant,
                "{model_id}/{tier} load quant"
            );
            assert_eq!(
                artifact.spec.resolved_route.as_deref(),
                Some(arm.model_id),
                "{model_id}/{tier} resolved route"
            );
            // The worker's `provider_load_spec` for this lane never sets `offload_policy`, so
            // the capture must load under the gen-core default it leaves in place.
            assert_eq!(
                artifact.spec.offload_policy,
                OffloadPolicy::Resident,
                "{model_id}/{tier} offload policy"
            );
            assert_eq!(
                artifact.spec.adapters.len(),
                usize::from(arm.lightning),
                "{model_id}/{tier}: the distill stack must be exactly the member's"
            );
            if arm.lightning {
                let expected = source
                    .lightning
                    .as_ref()
                    .unwrap()
                    .root
                    .canonicalize()
                    .unwrap()
                    .join(protocol::QWEN_EDIT_LIGHTNING_FILE);
                let adapter = &artifact.spec.adapters[0];
                assert_eq!(adapter.path, expected, "{model_id}/{tier} distill path");
                assert!(
                    adapter.path.ends_with(protocol::QWEN_EDIT_LIGHTNING_FILE),
                    "{model_id}/{tier}: {}",
                    adapter.path.display()
                );
                assert_eq!(adapter.scale, 1.0, "{model_id}/{tier} distill scale");
                assert!(
                    matches!(adapter.kind, AdapterKind::Lora),
                    "{model_id}/{tier}: {:?}",
                    adapter.kind
                );
            }
        }
    }
}

/// sc-22728: the root the capture opens must end in the PLANNED tier's directory, per member —
/// a stale `…/q4` export cannot satisfy a q8 plan and quietly re-label another tier's peaks.
#[test]
fn a_candle_qwen_edit_root_of_another_tier_is_refused_naming_the_planned_tier() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    for (arm, model_id) in QWEN_EDIT_MEMBERS {
        // The plan is q8; the root on disk is the q4 export.
        let source = qwen_edit_source(fixtures, arm, "q4");
        let error = qwen_edit_load_spec(arm, "q8", &source, LoadShape::DeferredMaterialization)
            .expect_err("a q8 plan must not be satisfied by a q4 root");
        assert!(
            error.ends_with(&format!("/snapshots/{QWEN_EDIT_TEST_REVISION}/q8")),
            "{model_id}: {error}"
        );
        // The wrong artifact family is refused before the root is even looked at.
        let mut foreign = qwen_edit_source(fixtures, arm, "q8");
        foreign.repository = protocol::QWEN_EDIT_LIGHTNING_REPOSITORY.to_owned();
        let error = qwen_edit_load_spec(arm, "q8", &foreign, LoadShape::DeferredMaterialization)
            .expect_err("the distill repository is not the base artifact");
        assert!(
            error.contains(protocol::QWEN_EDIT_REPOSITORY),
            "{model_id}: {error}"
        );
    }
}

/// sc-22728: the reference is one interleaved RGB frame at the request geometry — the shape the
/// engine hard-requires and the worker always supplies.
#[test]
fn the_candle_qwen_edit_reference_is_one_frame_at_the_request_geometry() {
    let reference = qwen_edit_reference(1024, 768);
    assert_eq!((reference.width, reference.height), (1024, 768));
    assert_eq!(reference.pixels.len(), 1024 * 768 * 3);
}

#[test]
fn deferred_materialization_establishes_retention_baseline_after_resident_rung() {
    let mut baseline = None;
    update_warmed_retention_baseline(&mut baseline, 12 * GIB).unwrap();
    assert_eq!(baseline, Some(12 * GIB));
    update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB).unwrap();
}

#[test]
fn warmed_retention_baseline_rejects_later_growth() {
    let mut baseline = Some(12 * GIB);
    let error =
        update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB + 1).unwrap_err();
    assert!(error.contains("above the warmed resident baseline"));
}

/// A single planned case at `frames`, minimal enough that the geometry guard is the FIRST thing
/// that can reject it — no weight root, no environment.
///
/// `fixture` is LOAD-BEARING here, not decoration. [`run`] picks its dispatch branch from it: a
/// `fresh-five-rung-` prefix routes into [`run_five_rung_reference`], and ANY OTHER fixture on
/// `krea_2_turbo` falls through to the inline Krea arm instead. A table that only ever passes
/// one prefix therefore exercises only one of the two branches, which is precisely how this
/// test shipped blind to [`run`]'s own guard once (sc-18808 re-review).
fn still_planned_case_with_fixture(
    provider: &str,
    rung: &str,
    frames: u64,
    fixture: &str,
) -> Value {
    json!({
        "backend": "candle",
        "target": {
            "provider": provider,
            "modelId": provider,
            "tier": "q4",
            "mode": "text_to_image",
            "overlay": "none",
            "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": frames }
        },
        "loadShape": "deferred_materialization",
        "strategy": { "rung": rung, "parameters": {} },
        "calibrationFingerprint": "unused",
        "fixture": fixture
    })
}

/// The five-rung shape — the only one [`run_five_rung_batch`] accepts.
fn still_planned_case(provider: &str, rung: &str, frames: u64) -> Value {
    still_planned_case_with_fixture(provider, rung, frames, "fresh-five-rung-unused")
}

/// The same shape with the CATALOG model id spelled independently of the provider — the axis
/// the two klein models differ on (sc-22727).
fn still_planned_case_for(provider: &str, model_id: &str, rung: &str, frames: u64) -> Value {
    let mut planned = still_planned_case(provider, rung, frames);
    planned["target"]["modelId"] = json!(model_id);
    planned
}

/// The same shape in a declared plan mode. `edit_image` on the Turbo provider is the
/// `z_image_edit` route (sc-22724), which is its own execution path with its own refusal label.
fn still_planned_case_in_mode(provider: &str, rung: &str, frames: u64, mode: &str) -> Value {
    let mut planned = still_planned_case(provider, rung, frames);
    planned["target"]["mode"] = json!(mode);
    planned
}

/// The canonical five-rung batch shape `run_five_rung_batch` requires, at `frames` and `mode`.
fn still_batch_request_in_mode(provider: &str, frames: u64, mode: &str) -> Value {
    let planned: Vec<Value> = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ]
    .into_iter()
    .map(|rung| still_planned_case_in_mode(provider, rung, frames, mode))
    .collect();
    json!({ "action": "run_batch", "planned": planned })
}

/// sc-18808 — the Candle twin of the MLX adapter's
/// `every_image_arm_still_refuses_a_multi_frame_geometry`.
///
/// BOTH Candle arms hardcoded `frames: 1` into `MemoryGeometry` while reading only
/// `width`/`height` from the plan, so a plan row declaring any other frame count would have
/// rendered ONE frame and emitted a record claiming a geometry it was never asked for.
///
/// Two entry points are reachable from `main`, and both must refuse with the exact pinned
/// wording before any environment or weight work:
///
/// * `run` — the dispatcher. Its guard stands in front of BOTH of its branches, and the branch
///   is chosen by `planned.fixture`, which is why the table below carries the fixture instead of
///   hardcoding one. A `fresh-five-rung-` prefix (or the `qwen_image` provider) routes into
///   `run_five_rung_reference`; ANY OTHER fixture on `krea_2_turbo` falls through to the INLINE
///   Krea arm — the arm that resolves `SCENEWORKS_KREA_REPOSITORY` and then writes its own
///   `MemoryGeometry { frames: 1 }`. Five SHIPPED Candle plan rows carry the second shape
///   (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings), so it is the live one; the third
///   row below is one of them verbatim. Until it was added, every case in this table began
///   `fresh-five-rung-`, so all of them short-circuited into `run_five_rung_reference` and
///   `run`'s own guard was shadowed by the redundant copy at the head of that function —
///   deleting `run`'s guard left this test green.
/// * `run_five_rung_batch` — reached straight from `main`, so `run`'s guard never sees it. Its
///   per-item pre-load loop is the guard under test; the fixture is irrelevant to it, so the
///   canonical five-rung shape is the only one it is exercised with.
///
/// The other two copies of the refusal are redundant defense-in-depth and are NOT what this
/// test pins: the one at the head of `run_five_rung_reference` (whose only caller is `run`,
/// which already refused) and the one in `run_five_rung_reference_loaded` (reachable only after
/// a real generator load, so no unit test can enter it). Removing either of those alone leaves
/// this suite green — which is exactly why a future reader must not "clean up" `run`'s or
/// `run_five_rung_batch`'s on the strength of them still being there.
#[test]
fn every_candle_arm_still_refuses_a_multi_frame_geometry() {
    for (provider, model_id, mode, label, fixture) in [
        (
            QWEN_ID,
            QWEN_ID,
            "text_to_image",
            QWEN_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        (
            KREA_ID,
            KREA_ID,
            "text_to_image",
            KREA_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        (
            Z_IMAGE_TURBO_ID,
            Z_IMAGE_TURBO_ID,
            "text_to_image",
            Z_IMAGE_TURBO_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        // The `z_image_edit` route is its own execution path and refuses under its own label
        // (sc-22724): the two Turbo rows must not report the same sentence.
        (
            Z_IMAGE_TURBO_ID,
            Z_IMAGE_TURBO_ID,
            "edit_image",
            Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        (
            Z_IMAGE_ID,
            Z_IMAGE_ID,
            "text_to_image",
            Z_IMAGE_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        // sc-22727: three FLUX.2 catalog models over two registry ids. The two klein rows share
        // `flux2_klein_9b` and must NOT report the same sentence.
        (
            FLUX2_DEV_ID,
            "flux2_dev",
            "text_to_image",
            FLUX2_DEV_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        (
            FLUX2_KLEIN_ID,
            "flux2_klein_9b",
            "text_to_image",
            FLUX2_KLEIN_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        (
            FLUX2_KLEIN_ID,
            "flux2_klein_9b_kv",
            "text_to_image",
            FLUX2_KLEIN_KV_STILL_CALIBRATION,
            "fresh-five-rung-unused",
        ),
        // The inline Krea arm — a real shipped plan fixture, which the rows above cannot
        // reach.
        (
            KREA_ID,
            KREA_ID,
            "text_to_image",
            KREA_STILL_CALIBRATION,
            "krea-q4-1024-seed42",
        ),
    ] {
        for frames in [0_u64, 2, 97] {
            let expected = format!("{label} requires geometry.frames == 1, got {frames}");
            let mut planned =
                still_planned_case_with_fixture(provider, "resident", frames, fixture);
            planned["target"]["modelId"] = json!(model_id);
            planned["target"]["mode"] = json!(mode);
            let request = json!({ "action": "run", "planned": planned });
            assert_eq!(
                run(&request).expect_err("the Candle dispatcher must refuse a video geometry"),
                expected,
                "run: {provider}/{model_id}/{mode} at frames={frames} via fixture {fixture:?}"
            );
        }
    }
    for (provider, mode, label) in [
        (QWEN_ID, "text_to_image", QWEN_STILL_CALIBRATION),
        (KREA_ID, "text_to_image", KREA_STILL_CALIBRATION),
        (
            Z_IMAGE_TURBO_ID,
            "text_to_image",
            Z_IMAGE_TURBO_STILL_CALIBRATION,
        ),
        (
            Z_IMAGE_TURBO_ID,
            "edit_image",
            Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION,
        ),
        (Z_IMAGE_ID, "text_to_image", Z_IMAGE_STILL_CALIBRATION),
    ] {
        for frames in [0_u64, 2, 97] {
            let expected = format!("{label} requires geometry.frames == 1, got {frames}");
            assert_eq!(
                run_five_rung_batch(&still_batch_request_in_mode(provider, frames, mode))
                    .expect_err("the Candle batch arm must refuse a video geometry"),
                expected,
                "run_batch: {provider}/{mode} at frames={frames}"
            );
        }
    }
}

/// The third row above is not decoration: the fixtures the SHIPPED Candle plan gives the inline
/// Krea arm really do take that branch, so `run`'s own guard — not the redundant copy inside
/// [`run_five_rung_reference`] — is the one standing in front of them.
///
/// Asserted against the dispatch predicate itself rather than by observing an error, because
/// both branches resolve the same `SCENEWORKS_KREA_REPOSITORY` and would report the same
/// sentence: the error is not a routing witness, the predicate is. Widening the prefix (or
/// renaming these fixtures into it) would re-shadow `run`'s guard, and this reds when it does.
#[test]
fn the_shipped_krea_fixtures_take_the_inline_arm_not_the_five_rung_branch() {
    for fixture in [
        "krea-q4-1024-seed42",
        "krea-q8-1024-seed42",
        "krea-bf16-1024-seed42",
        "krea-q4-768-seed42",
        "krea-q4-1024-seed42-v2-candidate",
    ] {
        let request = json!({
            "planned": still_planned_case_with_fixture(KREA_ID, "resident", 1, fixture)
        });
        assert!(
            !routes_to_five_rung_reference(&request).unwrap(),
            "{fixture} must reach the inline Krea arm"
        );
    }
    for (provider, fixture) in [
        (KREA_ID, "fresh-five-rung-krea-q4-1024-seed16402-step2"),
        (QWEN_ID, "qwen-image-candle-q4-seed15817-step2"),
        (
            Z_IMAGE_TURBO_ID,
            "fresh-five-rung-z-image-turbo-q4-1024-seed16402-step2",
        ),
        // No inline arm exists for Z-Image-Turbo, so an off-prefix fixture still routes here.
        (Z_IMAGE_TURBO_ID, "z-image-turbo-any-other-fixture"),
        // sc-22724: nor for the base, whose shipped fixtures carry the sc-16170 spelling.
        (
            Z_IMAGE_ID,
            "sc-16170-z-image-q4-1024-text_to_image-none-seed16170",
        ),
    ] {
        let request = json!({
            "planned": still_planned_case_with_fixture(provider, "resident", 1, fixture)
        });
        assert!(
            routes_to_five_rung_reference(&request).unwrap(),
            "{fixture} must reach the five-rung reference path"
        );
    }
}

/// sc-22727: the FLUX.2 family is three catalog models over two registry ids, and each member
/// carries its OWN execution path, refusal label, artifact family and diagnostics slug. Two
/// members share `flux2_klein_9b`, so `modelId` — never `provider` — is the discriminator, and
/// a pair no member serves is refused by name rather than measured as its nearest neighbour.
#[test]
fn the_candle_flux2_family_is_resolved_from_the_plans_provider_and_model_id() {
    for (provider, model_id, expected) in [
        (FLUX2_DEV_ID, "flux2_dev", FLUX2_DEV_ARM),
        (FLUX2_KLEIN_ID, "flux2_klein_9b", FLUX2_KLEIN_ARM),
        (FLUX2_KLEIN_ID, "flux2_klein_9b_kv", FLUX2_KLEIN_KV_ARM),
    ] {
        let request =
            json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
        assert_eq!(flux2_arm(&request).unwrap(), Some(expected));
        assert_eq!(
            plain_execution_path(&request).unwrap(),
            expected.execution_path
        );
        assert_eq!(
            still_calibration_label(&request).unwrap(),
            expected.still_calibration
        );
        // No inline arm exists for FLUX.2, so every fixture routes to the five-rung path.
        assert!(routes_to_five_rung_reference(&request).unwrap());
    }
    for (provider, model_id) in [
        (FLUX2_DEV_ID, "flux2_klein_9b"),
        (FLUX2_KLEIN_ID, "flux2_dev"),
        // A real catalog model on the klein provider that this adapter does NOT serve: its
        // snapshot is an assembled convert dir, not a tiered rehost.
        (FLUX2_KLEIN_ID, "flux2_klein_9b_true_v2"),
    ] {
        let request =
            json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
        let error = flux2_arm(&request).expect_err("an unserved pair must be refused by name");
        assert!(
            error.contains(&format!("provider {provider:?} for model {model_id:?}")),
            "{provider}/{model_id}: {error}"
        );
        // And the refusal reaches the callers rather than being swallowed into a default path.
        assert_eq!(plain_execution_path(&request).unwrap_err(), error);
        assert_eq!(still_calibration_label(&request).unwrap_err(), error);
    }
    // A non-FLUX.2 plan resolves no member at all, and says so without erroring.
    assert_eq!(
        flux2_arm(&json!({ "planned": still_planned_case(Z_IMAGE_ID, "resident", 1) })).unwrap(),
        None
    );
    // Every member is distinguishable from every other on every identity axis: a collision
    // would let one artifact satisfy another's plan, or make two records indistinguishable.
    for field in [
        FLUX2_ARMS.map(|arm| arm.model_id),
        FLUX2_ARMS.map(|arm| arm.execution_path),
        FLUX2_ARMS.map(|arm| arm.still_calibration),
        FLUX2_ARMS.map(|arm| arm.repository_env),
        FLUX2_ARMS.map(|arm| arm.revision_env),
        FLUX2_ARMS.map(|arm| arm.root_env),
        FLUX2_ARMS.map(|arm| arm.expected_repository),
        FLUX2_ARMS.map(|arm| arm.slug),
    ] {
        let mut unique = field.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), FLUX2_ARMS.len(), "collision in {field:?}");
    }
    assert_eq!(FLUX2_KLEIN_ARM.provider, FLUX2_KLEIN_KV_ARM.provider);
    assert_eq!(
        FLUX2_KLEIN_ARM.expected_repository,
        protocol::FLUX2_KLEIN_REPOSITORY
    );
    assert_eq!(
        FLUX2_KLEIN_KV_ARM.expected_repository,
        protocol::FLUX2_KLEIN_KV_REPOSITORY
    );
    // Which member hands the planned tier to the loader, stated as data (sc-22727 review):
    // only dev folds it; both klein turnkeys are dense-TE tiers the worker loads with
    // `Quant::None`, and candle-gen-flux2 would otherwise re-quantize their packed DiT.
    assert_eq!(
        FLUX2_ARMS.map(|arm| arm.tier_quant_reaches_the_loader),
        [true, false, false],
        "only the dev route folds the planned tier into LoadSpec::quantize"
    );
    // ...and bound to the manifest the worker reads that decision from: `is_dense_te_tier` is
    // exactly `mlx.denseTextEncoderTier == true`, so the flag must be its negation per member.
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    ))
    .expect("the shipped models manifest parses");
    for arm in FLUX2_ARMS {
        let entry = manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|entry| entry["id"] == arm.model_id)
            .unwrap_or_else(|| panic!("{} is not a shipped model", arm.model_id));
        let dense_te = entry["mlx"]["denseTextEncoderTier"] == json!(true);
        assert_eq!(
            arm.tier_quant_reaches_the_loader, !dense_te,
            "{}: the worker loads a dense-TE tier with Quant::None (is_dense_te_tier)",
            arm.model_id
        );
    }
}

/// sc-22724: the `z_image_edit` route is the Turbo provider in `edit_image` mode — the same
/// loader, a distinct execution path, and one reference on the request — and only the Turbo
/// arm has that second mode.
#[test]
fn the_z_image_edit_route_is_the_turbo_arm_in_edit_mode() {
    let mut edit = still_planned_case(Z_IMAGE_TURBO_ID, "resident", 1);
    edit["target"]["mode"] = json!("edit_image");
    edit["target"]["modelId"] = json!("z_image_edit");
    let request = json!({ "planned": edit });
    assert!(is_z_image_edit(&request).unwrap());
    assert_eq!(
        plain_execution_path(&request).unwrap(),
        Z_IMAGE_TURBO_EDIT_EXECUTION_PATH
    );
    assert!(routes_to_five_rung_reference(&request).unwrap());
    let plain = json!({ "planned": still_planned_case(Z_IMAGE_TURBO_ID, "resident", 1) });
    assert!(!is_z_image_edit(&plain).unwrap());
    assert_eq!(
        plain_execution_path(&plain).unwrap(),
        Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH
    );
    // The base has no edit route; its mode does not change its path.
    let mut base = still_planned_case(Z_IMAGE_ID, "resident", 1);
    base["target"]["mode"] = json!("edit_image");
    let base = json!({ "planned": base });
    assert!(!is_z_image_edit(&base).unwrap());
    assert_eq!(
        plain_execution_path(&base).unwrap(),
        Z_IMAGE_PLAIN_EXECUTION_PATH
    );
}

/// sc-22724: the edit request is the worker's — one reference at the target geometry plus the
/// production strength — and the text-to-image request carries none.
#[test]
fn the_edit_generation_request_carries_one_reference_at_the_target_geometry() {
    let edit = five_rung_generation_request(1024, 768, true);
    assert_eq!(edit.conditioning.len(), 1);
    match &edit.conditioning[0] {
        Conditioning::Reference { image, strength } => {
            assert_eq!((image.width, image.height), (1024, 768));
            assert_eq!(image.pixels.len(), 1024 * 768 * 3);
            assert_eq!(*strength, Some(Z_IMAGE_EDIT_STRENGTH));
        }
        other => panic!("expected one Reference, got {other:?}"),
    }
    // The worker sets ONLY the per-reference strength (`build_lane_conditioning`,
    // image_jobs/base.rs:7136); the request-level lever stays unset, and so does this arm's.
    assert_eq!(edit.strength, None);
    // floor(4 * 0.6) = 2, so two executed denoise steps remain behind the conditioning
    // boundary — the engine's `init_time_step` law, not this arm's.
    assert_eq!(edit.steps, Some(Z_IMAGE_EDIT_STEPS));
    let plain = five_rung_generation_request(1024, 1024, false);
    assert!(plain.conditioning.is_empty());
    assert_eq!(plain.strength, None);
    assert_eq!(plain.steps, Some(2));
}

/// And the guard is the frames axis rather than a blanket rejection: the same still geometry
/// passes it on both Candle labels, so the refusals above cannot be an unconditional error.
#[test]
fn the_candle_still_geometry_guard_is_not_a_blanket_refusal() {
    for (provider, model_id) in [
        (QWEN_ID, QWEN_ID),
        (KREA_ID, KREA_ID),
        (Z_IMAGE_TURBO_ID, Z_IMAGE_TURBO_ID),
        (Z_IMAGE_ID, Z_IMAGE_ID),
        (FLUX2_DEV_ID, "flux2_dev"),
        (FLUX2_KLEIN_ID, "flux2_klein_9b"),
        (FLUX2_KLEIN_ID, "flux2_klein_9b_kv"),
        (FLUX1_DEV_ID, FLUX1_DEV_ID),
        (FLUX1_SCHNELL_ID, FLUX1_SCHNELL_ID),
        (PULID_FLUX_ID, PULID_FLUX_ID),
    ] {
        let request =
            json!({ "planned": still_planned_case_for(provider, model_id, "resident", 1) });
        let label = still_calibration_label(&request).unwrap();
        protocol::validate_still_geometry(&request, label)
            .unwrap_or_else(|error| panic!("{provider}: {error}"));
    }
}

// -----------------------------------------------------------------------------------------
// sc-22730 — the SD3.5 family on the Candle lane.
// -----------------------------------------------------------------------------------------

/// All three SD3.5 members reach the shared five-rung reference path under their OWN execution
/// paths and refusal labels. None has an inline arm, so an off-prefix fixture must still route
/// here — proving the provider-id branch does it, not the fixture-prefix branch.
#[test]
fn the_sd3_5_members_route_to_the_five_rung_reference() {
    for (provider, path, label) in [
        (
            SD3_5_LARGE_ID,
            SD3_5_LARGE_PLAIN_EXECUTION_PATH,
            SD3_5_LARGE_STILL_CALIBRATION,
        ),
        (
            SD3_5_LARGE_TURBO_ID,
            SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH,
            SD3_5_LARGE_TURBO_STILL_CALIBRATION,
        ),
        (
            SD3_5_MEDIUM_ID,
            SD3_5_MEDIUM_PLAIN_EXECUTION_PATH,
            SD3_5_MEDIUM_STILL_CALIBRATION,
        ),
    ] {
        let request = json!({
            "planned": still_planned_case_with_fixture(provider, "staged_residency", 1, "sc-22730-off-prefix")
        });
        assert!(
            routes_to_five_rung_reference(&request).unwrap(),
            "{provider}"
        );
        assert_eq!(plain_execution_path(&request).unwrap(), path, "{provider}");
        assert_eq!(
            still_calibration_label(&request).unwrap(),
            label,
            "{provider}"
        );
        assert_eq!(five_rung_evidence_story(provider), "sc-22730", "{provider}");
    }
}

/// Every member refuses a non-still geometry under its OWN label, before any env or weight
/// work — the sc-18808 guard, on this family.
#[test]
fn every_candle_sd3_5_member_refuses_a_multi_frame_geometry() {
    for (provider, label) in [
        (SD3_5_LARGE_ID, SD3_5_LARGE_STILL_CALIBRATION),
        (SD3_5_LARGE_TURBO_ID, SD3_5_LARGE_TURBO_STILL_CALIBRATION),
        (SD3_5_MEDIUM_ID, SD3_5_MEDIUM_STILL_CALIBRATION),
    ] {
        for frames in [0_u64, 2, 97] {
            let request =
                json!({ "planned": still_planned_case(provider, "staged_residency", frames) });
            let error = run(&request).expect_err("a non-still geometry must be refused");
            assert_eq!(
                error,
                format!("{label} requires geometry.frames == 1, got {frames}"),
                "{provider}"
            );
        }
    }
}

/// The fixture binds member, tier, geometry edge, seed and step count, so a Medium q8 record
/// can never be attributed to a Large bf16 capture that merely reused the string.
#[test]
fn the_candle_sd3_5_fixture_binds_member_tier_edge_seed_and_steps() {
    let case = |provider: &str, tier: &str, fixture: &str| {
        let mut planned = still_planned_case_with_fixture(provider, "staged_residency", 1, fixture);
        planned["target"]["tier"] = json!(tier);
        json!({ "planned": planned })
    };
    for (provider, slug) in SD3_5_FIXTURE_SLUGS {
        for tier in ["q4", "q8", "bf16"] {
            let good = format!("fresh-five-rung-{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2");
            validate_fixture_binds_tier_and_geometry(&case(provider, tier, &good))
                .unwrap_or_else(|error| panic!("{provider} {tier}: {error}"));

            // Another tier's token, another edge, the MLX arm's seed, and a step count this
            // arm never renders are each refused by name.
            for (bad, expected) in [
                (
                    format!("fresh-five-rung-{slug}-{tier}-768-seed{FIVE_RUNG_SEED}-step2"),
                    "must start with",
                ),
                (
                    format!("fresh-five-rung-{slug}-{tier}-1024-seed22730-step2"),
                    "does not match the seed",
                ),
                (
                    format!("fresh-five-rung-{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step3"),
                    "two-step",
                ),
                (
                    format!("{slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2"),
                    "must start with",
                ),
            ] {
                let error = validate_fixture_binds_tier_and_geometry(&case(provider, tier, &bad))
                    .expect_err(&format!("{provider} {tier} {bad} must be refused"));
                assert!(
                    error.contains(expected),
                    "{provider} {tier} {bad}: {error} lacks {expected:?}"
                );
            }

            // A SIBLING member's fixture is refused: the slug is part of the prefix.
            for (other, other_slug) in SD3_5_FIXTURE_SLUGS {
                if other == provider {
                    continue;
                }
                let crossed =
                    format!("fresh-five-rung-{other_slug}-{tier}-1024-seed{FIVE_RUNG_SEED}-step2");
                // `sd3-5-large` is a PREFIX of `sd3-5-large-turbo`, so the Large binding would
                // accept the Turbo fixture on a naive `starts_with`; the tier token that
                // follows the slug is what actually separates them.
                assert!(
                    validate_fixture_binds_tier_and_geometry(&case(provider, tier, &crossed))
                        .is_err(),
                    "{provider} {tier} must refuse {other}'s fixture {crossed}"
                );
            }
        }
    }
    // An unknown provider is refused by name rather than silently skipped.
    let error = validate_sd35_fixture(
        &case("sd3_5_enormous", "q4", "fresh-five-rung-x"),
        "sd3_5_enormous",
        "q4",
    )
    .expect_err("an unknown SD3.5 member must be refused");
    assert!(error.contains("sd3_5_enormous"), "{error}");
}

/// Every SD3.5 candle cell the committed plan declares is served by an arm: an execution path,
/// a refusal label, a tier, a fixture that satisfies the binding, and the five-rung route.
/// The exact cell set is asserted, not a count.
#[test]
fn every_planned_sd3_5_candle_cell_is_served_by_an_arm() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        if !key.ends_with(":candle") {
            continue;
        }
        let provider = entry["provider"].as_str().unwrap();
        if !SD3_5_FIXTURE_SLUGS.iter().any(|(id, _)| *id == provider) {
            continue;
        }
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "backend": "candle",
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "strategy": { "rung": "staged_residency", "parameters": {} },
            "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        plain_execution_path(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        still_calibration_label(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(planned_tier(&request).unwrap(), tier, "{key}");
        validate_fixture_binds_tier_and_geometry(&request)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert!(routes_to_five_rung_reference(&request).unwrap(), "{key}");
        // The candle anchor is the SHALLOW STAGED composition the derivation law prices
        // (`extract-memory-anchors.mjs` `isDerivable`), which the manifest declares under
        // `requiredOffloadPolicy: sequential` — the shape `load_five_rung_generator` builds.
        assert_eq!(
            entry["loadShape"].as_str().unwrap(),
            "deferred_materialization",
            "{key}"
        );
    }
    let expected: std::collections::BTreeSet<String> =
        ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
            .iter()
            .flat_map(|model| {
                ["bf16", "q4", "q8"]
                    .iter()
                    .map(move |tier| format!("{model}:{tier}:candle"))
            })
            .collect();
    assert_eq!(seen, expected);
}

// -----------------------------------------------------------------------------------------
// sc-22726 — the FLUX.1 family on the Candle lane.
// -----------------------------------------------------------------------------------------

fn flux_one_temp_dir(fixtures: &crate::adapter_test_support::Fixtures, label: &str) -> PathBuf {
    fixtures.path(label)
}

fn flux_one_snapshot_root(
    fixtures: &crate::adapter_test_support::Fixtures,
    repository: &str,
    revision: &str,
    tier: &str,
) -> PathBuf {
    fixtures.snapshot(repository, revision, Some(tier))
}

fn staged_pulid_bundle(
    fixtures: &crate::adapter_test_support::Fixtures,
) -> protocol::PulidIdentityBundle {
    fixtures.pulid_bundle()
}

/// The two FLUX.1 BASE providers reach the shared five-rung reference path under their own
/// execution paths and refusal labels; PuLID deliberately does NOT, because it is a bespoke
/// route that never touches the provider registry.
#[test]
fn the_flux_one_base_providers_route_to_the_five_rung_reference_and_pulid_does_not() {
    for (provider, path, label) in [
        (
            FLUX1_DEV_ID,
            FLUX1_DEV_PLAIN_EXECUTION_PATH,
            FLUX1_DEV_STILL_CALIBRATION,
        ),
        (
            FLUX1_SCHNELL_ID,
            FLUX1_SCHNELL_PLAIN_EXECUTION_PATH,
            FLUX1_SCHNELL_STILL_CALIBRATION,
        ),
    ] {
        // An off-prefix fixture must still route here: these providers have no inline arm.
        let request = json!({
            "planned": still_planned_case_with_fixture(provider, "staged_residency", 1, "sc-22726-off-prefix")
        });
        assert!(
            routes_to_five_rung_reference(&request).unwrap(),
            "{provider}"
        );
        assert_eq!(plain_execution_path(&request).unwrap(), path);
        assert_eq!(still_calibration_label(&request).unwrap(), label);
    }
    let pulid = json!({
        "planned": still_planned_case_with_fixture(PULID_FLUX_ID, "staged_residency", 1, "sc-22726-pulid")
    });
    assert!(
        !routes_to_five_rung_reference(&pulid).unwrap(),
        "PuLID must not be served by the registry five-rung path"
    );
    assert_eq!(
        plain_execution_path(&pulid).unwrap(),
        PULID_FLUX_EXECUTION_PATH
    );
    assert_eq!(
        still_calibration_label(&pulid).unwrap(),
        PULID_FLUX_STILL_CALIBRATION
    );
}

/// Every FLUX.1 member still refuses a video geometry under its OWN label, before it resolves
/// an environment variable or opens a snapshot (sc-18808). PuLID goes through its own
/// dispatch, so it is checked through `run` too.
#[test]
fn every_candle_flux_one_member_refuses_a_multi_frame_geometry() {
    for (provider, mode, label) in [
        (FLUX1_DEV_ID, "text_to_image", FLUX1_DEV_STILL_CALIBRATION),
        (
            FLUX1_SCHNELL_ID,
            "text_to_image",
            FLUX1_SCHNELL_STILL_CALIBRATION,
        ),
        (
            PULID_FLUX_ID,
            "character_image",
            PULID_FLUX_STILL_CALIBRATION,
        ),
    ] {
        for frames in [0_u64, 2, 97] {
            let mut planned = still_planned_case_in_mode(provider, "resident", frames, mode);
            if provider == PULID_FLUX_ID {
                planned["target"]["overlay"] = json!("identity");
            }
            let error = run(&json!({ "action": "run", "planned": planned }))
                .expect_err("a video geometry must be refused");
            assert_eq!(
                error,
                format!("{label} requires geometry.frames == 1, got {frames}"),
                "{provider}/{frames}"
            );
        }
    }
}

/// The PuLID arm binds the FLUX.1-dev backbone at the PLANNED tier and the staged identity
/// stack into the exact `PulidFluxPaths` the worker builds — a q8 plan against a q4 export is
/// refused naming the tier, and all three tiers round-trip.
#[test]
fn the_candle_pulid_binding_carries_the_planned_tier_and_the_identity_stack() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    let pulid_case = |tier: &str| {
        let mut planned =
            still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
        planned["target"]["tier"] = json!(tier);
        planned["target"]["overlay"] = json!("identity");
        planned["target"]["modelId"] = json!("pulid_flux_dev");
        planned["fixture"] = json!(format!(
            "pulid-flux-face-sc16956-candle-{tier}-1024-seed{FLUX1_SEED}-step2"
        ));
        json!({ "planned": planned })
    };
    let q4_root = flux_one_snapshot_root(fixtures, protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
    let error = pulid_flux_binding_at(
        &pulid_case("q8"),
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q4_root.clone(),
        staged_pulid_bundle(fixtures),
    )
    .expect_err("a q8 plan must not be satisfied by a q4 root");
    assert!(
        error.ends_with(&format!("/snapshots/{REVISION}/q8")),
        "{error}"
    );

    for tier in ["q4", "q8", "bf16"] {
        let bundle = staged_pulid_bundle(fixtures);
        let root = flux_one_snapshot_root(fixtures, protocol::FLUX1_DEV_REPOSITORY, REVISION, tier);
        let binding = pulid_flux_binding_at(
            &pulid_case(tier),
            protocol::FLUX1_DEV_REPOSITORY.to_owned(),
            REVISION.to_owned(),
            root,
            bundle.clone(),
        )
        .unwrap_or_else(|error| panic!("{tier}: {error}"));
        assert_eq!(binding.tier, tier);
        assert!(binding.paths.flux_base.ends_with(tier));
        assert_eq!(binding.paths.pulid_weights, bundle.adapter);
        assert_eq!(binding.paths.eva_weights, bundle.eva);
        // The engine reads scrfd / arcface / bisenet out of `face_dir` BY NAME, so the bundle
        // root IS the face dir.
        assert_eq!(binding.paths.face_dir, bundle.root);
        // A ladder-admitted PuLID render carries no LoRA (`pulid_memory_ladder_eligible`).
        assert!(binding.paths.adapters.is_empty());
        assert!(binding.loadability_fingerprint().starts_with(&format!(
            "{}@{REVISION}:{tier}",
            protocol::FLUX1_DEV_REPOSITORY
        )));
        // By CONTENT, never by the host path the bundle was staged at.
        assert!(binding
            .loadability_fingerprint()
            .ends_with(&format!("+identity:{}", bundle.composite_sha256)));
        assert!(!binding
            .loadability_fingerprint()
            .contains(&bundle.root.display().to_string()));
        let artifact = binding.artifact_json();
        serde_json::from_value::<sceneworks_core::memory_calibration::Artifact>(artifact.clone())
            .expect("PuLID capture artifact must pass the production reader");
        assert_eq!(artifact["variant"].as_str(), Some(tier));
        assert_eq!(
            artifact["inventorySha256"].as_str(),
            Some(bundle.composite_sha256.as_str())
        );
    }
    // A fixture naming the five-rung seed is refused: PuLID renders at FLUX1_SEED.
    let mut wrong_seed = pulid_case("q4");
    wrong_seed["planned"]["fixture"] = json!(format!(
        "pulid-flux-face-sc16956-candle-q4-1024-seed{FIVE_RUNG_SEED}-step2"
    ));
    let error = pulid_flux_binding_at(
        &wrong_seed,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q4_root.clone(),
        staged_pulid_bundle(fixtures),
    )
    .expect_err("the fixture seed must be the seed the capture renders at");
    assert!(error.contains("does not match the seed"), "{error}");

    // The wrong artifact family is refused before the root is looked at.
    let error = pulid_flux_binding_at(
        &pulid_case("q4"),
        protocol::FLUX1_SCHNELL_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q4_root.clone(),
        staged_pulid_bundle(fixtures),
    )
    .expect_err("the schnell artifact must be refused");
    assert!(error.contains(protocol::FLUX1_DEV_REPOSITORY), "{error}");

    // PuLID is character_image only, and its overlay is exactly `identity`.
    let mut wrong_mode = pulid_case("q4");
    wrong_mode["planned"]["target"]["mode"] = json!("text_to_image");
    let error = pulid_flux_binding_at(
        &wrong_mode,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q4_root.clone(),
        staged_pulid_bundle(fixtures),
    )
    .expect_err("PuLID has no text-to-image route");
    assert!(error.contains("character_image only"), "{error}");
    let mut wrong_overlay = pulid_case("q4");
    wrong_overlay["planned"]["target"]["overlay"] = json!("none");
    let error = pulid_flux_binding_at(
        &wrong_overlay,
        protocol::FLUX1_DEV_REPOSITORY.to_owned(),
        REVISION.to_owned(),
        q4_root,
        staged_pulid_bundle(fixtures),
    )
    .expect_err("the identity route must require its own overlay");
    assert!(error.contains("executes exactly \"identity\""), "{error}");
}

/// An incomplete staged bundle is refused before the load, naming the missing files.
#[test]
fn an_incomplete_candle_pulid_bundle_is_refused_naming_the_missing_files() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    let root = flux_one_temp_dir(fixtures, "pulid-partial");
    std::fs::create_dir_all(&root).unwrap();
    for file in protocol::PULID_IDENTITY_BUNDLE_FILES.iter().skip(1) {
        std::fs::write(root.join(file), b"weights").unwrap();
    }
    let error =
        protocol::pulid_identity_bundle_at(root).expect_err("an incomplete bundle must be refused");
    assert!(error.contains(protocol::PULID_ADAPTER_FILE), "{error}");
    assert!(
        error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV),
        "{error}"
    );
}

/// `pulid_flux_binding` reads the identity bundle from `SCENEWORKS_PULID_WEIGHTS` and the
/// backbone from the FLUX1_DEV family — never the schnell one. Serialized on
/// [`PULID_ENV_LOCK`]: the process environment is global, and every other PuLID test in this
/// binary fails before it reads an env var.
#[test]
fn the_env_bound_pulid_binding_reads_the_bundle_env_and_the_dev_family() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    let _guard = PULID_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let restore = PulidEnv::capture();
    let bundle = staged_pulid_bundle(fixtures);
    let root = flux_one_snapshot_root(fixtures, protocol::FLUX1_DEV_REPOSITORY, REVISION, "q4");
    let mut planned =
        still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
    planned["target"]["overlay"] = json!("identity");
    planned["fixture"] = json!(format!(
        "pulid-flux-face-sc16956-candle-q4-1024-seed{FLUX1_SEED}-step2"
    ));
    let request = json!({ "planned": planned });
    std::env::set_var(
        "SCENEWORKS_FLUX1_DEV_REPOSITORY",
        protocol::FLUX1_DEV_REPOSITORY,
    );
    std::env::set_var("SCENEWORKS_FLUX1_DEV_REVISION", REVISION);
    std::env::set_var("SCENEWORKS_FLUX1_DEV_ROOT", &root);
    std::env::remove_var(protocol::PULID_IDENTITY_BUNDLE_ENV);
    let error = pulid_flux_binding(&request).expect_err("the bundle env is required");
    assert!(
        error.contains(protocol::PULID_IDENTITY_BUNDLE_ENV),
        "{error}"
    );
    std::env::set_var(protocol::PULID_IDENTITY_BUNDLE_ENV, &bundle.root);
    let binding = pulid_flux_binding(&request).unwrap();
    assert_eq!(binding.bundle.composite_sha256, bundle.composite_sha256);
    assert_eq!(
        binding.bundle.root,
        std::fs::canonicalize(&bundle.root).unwrap(),
        "the env value is canonicalized, not used as spelled"
    );
    // The schnell family is not consulted on the dev backbone.
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_ROOT", "/nonexistent/sc-22726");
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REPOSITORY", "not/a-repo");
    std::env::set_var("SCENEWORKS_FLUX1_SCHNELL_REVISION", "junk");
    pulid_flux_binding(&request).unwrap();
    drop(restore);
}

/// Serializes the tests that mutate the FLUX.1 env families and the PuLID bundle env.
static PULID_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every env var the PuLID binding reads, restored on drop — including on a panic.
struct PulidEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl PulidEnv {
    const NAMES: [&'static str; 7] = [
        "SCENEWORKS_FLUX1_DEV_REPOSITORY",
        "SCENEWORKS_FLUX1_DEV_REVISION",
        "SCENEWORKS_FLUX1_DEV_ROOT",
        "SCENEWORKS_FLUX1_SCHNELL_REPOSITORY",
        "SCENEWORKS_FLUX1_SCHNELL_REVISION",
        "SCENEWORKS_FLUX1_SCHNELL_ROOT",
        protocol::PULID_IDENTITY_BUNDLE_ENV,
    ];

    fn capture() -> Self {
        Self(
            Self::NAMES
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        )
    }
}

impl Drop for PulidEnv {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

/// The PuLID request and admission context are the worker's: the photoreal preset, one
/// reference at the target geometry, `overlay: identity`, and the provider mode
/// `character_image` the pinned route gate checks.
#[test]
fn the_candle_pulid_request_and_context_match_the_worker_route() {
    let generation = pulid_flux_generation_request(1024, 768);
    assert_eq!((generation.width, generation.height), (1024, 768));
    assert_eq!(generation.steps, 2);
    assert_eq!(generation.guidance, PULID_FLUX_GUIDANCE);
    assert_eq!(generation.id_weight, PULID_FLUX_ID_WEIGHT);
    assert_eq!(generation.seed, FLUX1_SEED);
    assert!(!generation.use_pid);
    assert!(generation.sampler.is_none() && generation.scheduler.is_none());

    let calibration = runtime_cuda::gen_core::MemoryCalibrationIdentity::new(
        "pulid-flux-cuda-identity-stack-staged-decode-attention-block-window-v1",
        LoadShape::DeferredMaterialization,
    );
    let mut planned =
        still_planned_case_in_mode(PULID_FLUX_ID, "staged_residency", 1, "character_image");
    planned["target"]["overlay"] = json!("identity");
    let selection = planned_selection(&json!({ "planned": planned })).unwrap();
    let context = pulid_flux_context(
        selection,
        &calibration,
        &calibration.fingerprint,
        1024,
        1024,
        1,
        1,
    );
    assert_eq!(
        context.mode,
        MemoryMode::Other("character_image".to_owned())
    );
    assert!(context.has_reference);
    assert_eq!(context.geometry.reference_count, 1);
    assert_eq!(context.overlay.as_deref(), Some("identity"));
    assert!(!context.use_pid);
    assert!(!context.has_phases);
}

/// The Wan 2.2 / SCAIL-2 pre-gate in `run` spells its four ids as consts so
/// `scripts/stale-lane-report.mjs` can read them off the source (sc-22736); this pins the
/// spelled set to the set the module actually implements, in both directions.
#[test]
fn the_pre_gate_names_exactly_the_providers_the_module_implements() {
    let spelled = [WAN_TI2V_5B_ID, WAN_T2V_A14B_ID, WAN_I2V_A14B_ID, SCAIL2_ID];
    for provider in spelled {
        assert!(candle_wan_scail2::implements(provider), "{provider}");
    }
    let mut implemented = candle_wan_scail2::providers().to_vec();
    implemented.sort_unstable();
    let mut spelled = spelled.to_vec();
    spelled.sort_unstable();
    assert_eq!(implemented, spelled);
    assert!(!candle_wan_scail2::implements(LTX25_ID));
    assert!(!candle_wan_scail2::implements(KREA_ID));
}

/// Every FLUX.1 cell the committed plan declares for this lane must name a provider this
/// adapter implements and a mode its arm serves — the plan/arm agreement E3 asks for, checked
/// against the checked-in plan rather than a hand-written sample.
#[test]
fn every_planned_flux_one_candle_cell_is_served_by_an_arm() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        if !key.ends_with(":candle") {
            continue;
        }
        let provider = entry["provider"].as_str().unwrap();
        if !matches!(provider, FLUX1_DEV_ID | FLUX1_SCHNELL_ID | PULID_FLUX_ID) {
            continue;
        }
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "backend": "candle",
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "strategy": { "rung": "staged_residency", "parameters": {} },
            "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        plain_execution_path(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        still_calibration_label(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(planned_tier(&request).unwrap(), tier, "{key}");
        // The fixture must satisfy the same binding a capture applies: member, tier, edge,
        // the seed that member renders at, and the step count.
        validate_fixture_binds_tier_and_geometry(&request)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        // The base providers ride the registry five-rung path; PuLID is bespoke.
        assert_eq!(
            routes_to_five_rung_reference(&request).unwrap(),
            provider != PULID_FLUX_ID,
            "{key}"
        );
    }
    // The exact cell set, not a count: three members x three tiers, each named once.
    let expected: std::collections::BTreeSet<String> =
        ["flux_dev", "flux_schnell", "pulid_flux_dev"]
            .iter()
            .flat_map(|model| {
                ["bf16", "q4", "q8"]
                    .iter()
                    .map(move |tier| format!("{model}:{tier}:candle"))
            })
            .collect();
    assert_eq!(seen, expected);
}

/// The FLUX.1 fixture binding on this lane: the base members name the five-rung seed they
/// render at, PuLID names its own, and the member, tier, edge and step count are all bound.
#[test]
fn the_candle_flux_one_fixture_binds_member_tier_edge_seed_and_steps() {
    let case = |provider: &str, tier: &str, fixture: &str| {
        let mut planned = still_planned_case_with_fixture(provider, "resident", 1, fixture);
        planned["target"]["tier"] = json!(tier);
        json!({ "planned": planned })
    };
    for (provider, prefix, seed) in [
        (FLUX1_DEV_ID, "fresh-five-rung-flux1-dev", FIVE_RUNG_SEED),
        (
            FLUX1_SCHNELL_ID,
            "fresh-five-rung-flux1-schnell",
            FIVE_RUNG_SEED,
        ),
        (PULID_FLUX_ID, "pulid-flux-face-sc16956-candle", FLUX1_SEED),
    ] {
        for tier in ["q4", "q8", "bf16"] {
            let good = format!("{prefix}-{tier}-1024-seed{seed}-step2");
            validate_flux_one_fixture(&case(provider, tier, &good), provider, tier)
                .unwrap_or_else(|error| panic!("{provider}/{tier}: {error}"));
        }
        let other_seed = if seed == FLUX1_SEED {
            FIVE_RUNG_SEED
        } else {
            FLUX1_SEED
        };
        for (fixture, expected) in [
            (
                format!("{prefix}-q8-1024-seed{seed}-step2"),
                "must start with",
            ),
            (
                format!("{prefix}-q4-768-seed{seed}-step2"),
                "must start with",
            ),
            (
                format!("{prefix}-q4-1024-seed{other_seed}-step2"),
                "does not match the seed",
            ),
            (format!("{prefix}-q4-1024-seed{seed}-step3"), "two-step"),
            ("fresh-five-rung-unused".to_owned(), "must start with"),
        ] {
            let error = validate_flux_one_fixture(&case(provider, "q4", &fixture), provider, "q4")
                .expect_err("the fixture must be bound to its cell");
            assert!(error.contains(expected), "{provider}: {fixture}: {error}");
        }
    }
    // The member prefixes are not interchangeable: a schnell fixture on a dev plan is refused.
    let crossed = format!("fresh-five-rung-flux1-schnell-q4-1024-seed{FIVE_RUNG_SEED}-step2");
    let error = validate_flux_one_fixture(&case(FLUX1_DEV_ID, "q4", &crossed), FLUX1_DEV_ID, "q4")
        .unwrap_err();
    assert!(error.contains("must start with"), "{error}");
    // ...and the shared entry point routes the FLUX.1 members here while leaving Qwen alone.
    validate_fixture_binds_tier_and_geometry(&case(FLUX1_DEV_ID, "q4", "fresh-five-rung-unused"))
        .expect_err("the shared validator must apply the FLUX.1 binding");
    validate_fixture_binds_tier_and_geometry(&case(QWEN_ID, "q4", "fresh-five-rung-unused"))
        .expect("Qwen's fixtures predate the convention and stay unbound");
}

/// The base arm reads the tier off the snapshot's `transformer/config.json` through
/// `candle-gen-flux`'s own resolver, so a directory merely NAMED `q8` cannot satisfy a q8
/// plan with q4 (or dense) weights inside it.
#[test]
fn the_candle_flux_one_base_arm_reads_the_tier_off_the_transformer_config() {
    let fixture_owner = crate::adapter_test_support::Fixtures::default();
    let fixtures = &fixture_owner;
    const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
    for (tier, config) in [
        (
            "q4",
            Some(json!({ "quantization": { "bits": 4, "group_size": 64 } })),
        ),
        (
            "q8",
            Some(json!({ "quantization": { "bits": 8, "group_size": 64 } })),
        ),
        ("bf16", None),
    ] {
        let root = flux_one_snapshot_root(fixtures, protocol::FLUX1_DEV_REPOSITORY, REVISION, tier);
        if let Some(config) = &config {
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(
                root.join("transformer/config.json"),
                serde_json::to_vec(config).unwrap(),
            )
            .unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        validate_flux_one_snapshot_tier(&spec, FLUX1_DEV_ID, tier)
            .unwrap_or_else(|error| panic!("{tier}: {error}"));
        for other in ["q4", "q8", "bf16"] {
            if other == tier {
                continue;
            }
            let error = validate_flux_one_snapshot_tier(&spec, FLUX1_DEV_ID, other)
                .expect_err("the declared tier must match the planned one");
            assert!(
                error.contains(&format!("planned tier {other} does not match")),
                "{tier}/{other}: {error}"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------
// sc-22732 — the turnkey still family on this lane: `kolors`, `ideogram_4`,
// `ideogram_4_turbo`, `lens` and `lens_turbo`. All five ride the shared five-rung reference
// path, so what is proven here is the plan binding: execution path, calibration label, fixture,
// artifact family (including Ideogram's split bf16 repository) and quant rule.
// ------------------------------------------------------------------------------------------

const TURNKEY_MEMBERS: [(&str, &str); 7] = [
    (KOLORS_ID, "kolors"),
    (IDEOGRAM_ID, "ideogram-4"),
    (IDEOGRAM_TURBO_ID, "ideogram-4-turbo"),
    (LENS_ID, "lens"),
    (LENS_TURBO_ID, "lens-turbo"),
    (QWEN_IMAGE_2_1_ID, "qwen-image-2-1"),
    (QWEN_IMAGE_2_1_TURBO_ID, "qwen-image-2-1-turbo"),
];

fn turnkey_request(provider: &str, tier: &str, fixture: &str) -> Value {
    json!({ "planned": {
        "backend": "candle",
        "target": {
            "provider": provider,
            "tier": tier,
            "mode": "text_to_image",
            "overlay": "none",
            "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 },
        },
        "loadShape": "deferred_materialization",
        "strategy": { "rung": "staged_residency", "parameters": {} },
        "calibrationFingerprint": "unused",
        "fixture": fixture,
    }})
}

/// Every member is served by the five-rung reference path under its OWN execution path and
/// calibration label, and a provider no member serves is refused by the shared sentence.
#[test]
fn every_turnkey_member_rides_the_five_rung_path_under_its_own_labels() {
    let mut paths = std::collections::BTreeSet::new();
    let mut labels = std::collections::BTreeSet::new();
    for (provider, slug) in TURNKEY_MEMBERS {
        let request = turnkey_request(
            provider,
            "q4",
            &format!("fresh-five-rung-{slug}-q4-1024-seed16402-step2"),
        );
        let path = plain_execution_path(&request).unwrap();
        let label = still_calibration_label(&request).unwrap();
        assert!(
            routes_to_five_rung_reference(&request).unwrap(),
            "{provider}"
        );
        assert_eq!(turnkey_fixture_slug(provider), Some(slug));
        // Each member's own sentence — a shared one would let a record name the wrong route.
        assert!(paths.insert(path), "{provider} reuses {path:?}");
        assert!(labels.insert(label), "{provider} reuses {label:?}");
    }
    // Kolors' two bespoke candle routes are NOT served here: they are different providers with
    // different overlays, and this arm measures only the plain base route.
    for provider in ["candle_kolors_ipadapter", "candle_kolors_control"] {
        let request = turnkey_request(provider, "q4", "unused");
        let error = plain_execution_path(&request).expect_err("a bespoke route is not served");
        assert_eq!(
            error,
            format!("Candle five-rung calibration does not implement provider {provider:?}")
        );
        assert_eq!(turnkey_fixture_slug(provider), None);
    }
}

/// Ideogram is the only member whose tiers span two repositories: q4/q8 bind the packed family,
/// bf16 the separate one. Both members share both.
#[test]
fn the_ideogram_five_rung_family_splits_bf16_onto_its_own_repository() {
    for (provider, path) in [
        (IDEOGRAM_ID, IDEOGRAM_PLAIN_EXECUTION_PATH),
        (IDEOGRAM_TURBO_ID, IDEOGRAM_TURBO_PLAIN_EXECUTION_PATH),
    ] {
        for tier in ["q4", "q8"] {
            let bound = ideogram_five_rung_family(provider, path, tier);
            assert_eq!(bound.2, "SCENEWORKS_IDEOGRAM_REPOSITORY", "{tier}");
            assert_eq!(bound.3, "SCENEWORKS_IDEOGRAM_REVISION", "{tier}");
            assert_eq!(bound.4, "SCENEWORKS_IDEOGRAM_ROOT", "{tier}");
            assert_eq!(bound.5, protocol::IDEOGRAM_REPOSITORY, "{tier}");
        }
        let dense = ideogram_five_rung_family(provider, path, "bf16");
        assert_eq!(dense.2, "SCENEWORKS_IDEOGRAM_BF16_REPOSITORY");
        assert_eq!(dense.3, "SCENEWORKS_IDEOGRAM_BF16_REVISION");
        assert_eq!(dense.4, "SCENEWORKS_IDEOGRAM_BF16_ROOT");
        assert_eq!(dense.5, protocol::IDEOGRAM_BF16_REPOSITORY);
        // The two repositories are genuinely different, which is the whole reason for the split.
        assert_ne!(
            protocol::IDEOGRAM_REPOSITORY,
            protocol::IDEOGRAM_BF16_REPOSITORY
        );
        // The provider and its execution path ride through unchanged on both branches.
        assert_eq!((dense.0, dense.1), (provider, path));
    }
}

/// Qwen-Image 2.1 (sc-24112) is Ideogram's split with the halves swapped: q4/q8 bind the
/// SceneWorks re-host's tier subdirs, bf16 the upstream snapshot at its ROOT.
#[test]
fn the_qwen_image_2_1_five_rung_family_splits_bf16_onto_the_upstream_snapshot_root() {
    for tier in ["q4", "q8"] {
        let bound = qwen_image_2_1_five_rung_family(tier);
        assert_eq!(bound.2, "SCENEWORKS_QWEN_IMAGE_2_1_REPOSITORY", "{tier}");
        assert_eq!(bound.3, "SCENEWORKS_QWEN_IMAGE_2_1_REVISION", "{tier}");
        assert_eq!(bound.4, "SCENEWORKS_QWEN_IMAGE_2_1_ROOT", "{tier}");
        assert_eq!(bound.5, protocol::QWEN_IMAGE_2_1_REPOSITORY, "{tier}");
        assert!(five_rung_root_is_tiered(QWEN_IMAGE_2_1_ID, tier), "{tier}");
    }
    let dense = qwen_image_2_1_five_rung_family("bf16");
    assert_eq!(dense.2, "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REPOSITORY");
    assert_eq!(dense.3, "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REVISION");
    assert_eq!(dense.4, "SCENEWORKS_QWEN_IMAGE_2_1_BF16_ROOT");
    assert_eq!(dense.5, protocol::QWEN_IMAGE_2_1_BF16_REPOSITORY);
    assert!(!five_rung_root_is_tiered(QWEN_IMAGE_2_1_ID, "bf16"));
    assert_eq!(
        (dense.0, dense.1),
        (QWEN_IMAGE_2_1_ID, QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH)
    );
    // sc-25907: Turbo has the same split over its OWN two repositories and env families, and
    // shares no binding with base at any tier.
    for tier in ["q4", "q8"] {
        let bound = qwen_image_2_1_turbo_five_rung_family(tier);
        assert_eq!(
            bound.2, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REPOSITORY",
            "{tier}"
        );
        assert_eq!(
            bound.3, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REVISION",
            "{tier}"
        );
        assert_eq!(bound.4, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_ROOT", "{tier}");
        assert_eq!(bound.5, protocol::QWEN_IMAGE_2_1_TURBO_REPOSITORY, "{tier}");
        assert!(
            five_rung_root_is_tiered(QWEN_IMAGE_2_1_TURBO_ID, tier),
            "{tier}"
        );
    }
    let dense = qwen_image_2_1_turbo_five_rung_family("bf16");
    assert_eq!(dense.2, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY");
    assert_eq!(dense.3, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REVISION");
    assert_eq!(dense.4, "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_ROOT");
    assert_eq!(dense.5, protocol::QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY);
    assert!(!five_rung_root_is_tiered(QWEN_IMAGE_2_1_TURBO_ID, "bf16"));
    assert_eq!(
        (dense.0, dense.1),
        (
            QWEN_IMAGE_2_1_TURBO_ID,
            QWEN_IMAGE_2_1_TURBO_PLAIN_EXECUTION_PATH
        )
    );
    for tier in ["bf16", "q4", "q8"] {
        let base = qwen_image_2_1_five_rung_family(tier);
        let turbo = qwen_image_2_1_turbo_five_rung_family(tier);
        assert_ne!(base.5, turbo.5, "{tier}");
        assert_ne!(base.2, turbo.2, "{tier}");
        assert_ne!(base.4, turbo.4, "{tier}");
    }
}

/// The literal identity this lane binds Qwen-Image 2.1's plan rows to IS the pinned engine's
/// constant — the binding the macOS host that writes the arm cannot compile (sc-24112).
#[test]
fn the_qwen_image_2_1_candle_identity_is_the_engine_constant() {
    assert_eq!(
        QWEN_IMAGE_2_1_CANDLE_FINGERPRINT,
        runtime_cuda::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
    );
    for tier in ["bf16", "q4", "q8"] {
        assert_eq!(
            turnkey_calibration_fingerprint(QWEN_IMAGE_2_1_ID, tier).as_deref(),
            Some(QWEN_IMAGE_2_1_CANDLE_FINGERPRINT),
            "{tier}"
        );
    }
    // sc-25907: and Turbo's literal IS the engine's Turbo constant, distinct from base's.
    assert_eq!(
            QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT,
            runtime_cuda::providers::qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT
        );
    assert_ne!(
        QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT,
        QWEN_IMAGE_2_1_CANDLE_FINGERPRINT
    );
    for tier in ["bf16", "q4", "q8"] {
        assert_eq!(
            turnkey_calibration_fingerprint(QWEN_IMAGE_2_1_TURBO_ID, tier).as_deref(),
            Some(QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT),
            "{tier}"
        );
    }
}

/// The turnkey fixture binds member, tier, edge, seed and step count on this lane too — at the
/// FIVE-RUNG seed, because that is what the shared reference render uses. A fixture naming the
/// MLX turnkey seed, or another member's slug, is refused.
#[test]
fn the_candle_turnkey_fixture_binds_member_tier_edge_seed_and_steps() {
    for (provider, slug) in TURNKEY_MEMBERS {
        let case = |fixture: &str, tier: &str| {
            validate_fixture_binds_tier_and_geometry(&turnkey_request(provider, tier, fixture))
        };
        case(
            &format!("fresh-five-rung-{slug}-q4-1024-seed16402-step2"),
            "q4",
        )
        .unwrap();
        for bad in [
            format!("fresh-five-rung-{slug}-q8-1024-seed16402-step2"),
            format!("fresh-five-rung-{slug}-q4-768-seed16402-step2"),
            // 22726 is the seed the PuLID/MLX-turnkey fixtures name, never this one.
            format!("fresh-five-rung-{slug}-q4-1024-seed22726-step2"),
            format!("fresh-five-rung-{slug}-q4-1024-seed16402-step4"),
            format!("{slug}-q4-1024-seed16402-step2"),
        ] {
            assert!(
                case(&bad, "q4").is_err(),
                "{provider}: {bad} must be refused"
            );
        }
    }
    // One member's fixture never satisfies another member's plan.
    assert!(validate_fixture_binds_tier_and_geometry(&turnkey_request(
        LENS_ID,
        "q4",
        "fresh-five-rung-lens-turbo-q4-1024-seed16402-step2"
    ))
    .is_err());
    // sc-25907: nor does either Qwen-Image 2.1 checkpoint's fixture satisfy the other's plan.
    assert!(validate_fixture_binds_tier_and_geometry(&turnkey_request(
        QWEN_IMAGE_2_1_ID,
        "q4",
        "fresh-five-rung-qwen-image-2-1-turbo-q4-1024-seed16402-step2"
    ))
    .is_err());
    assert!(validate_fixture_binds_tier_and_geometry(&turnkey_request(
        QWEN_IMAGE_2_1_TURBO_ID,
        "q4",
        "fresh-five-rung-qwen-image-2-1-q4-1024-seed16402-step2"
    ))
    .is_err());
}

/// The committed plan's Candle turnkey cells, as an exact key set: five members x three tiers,
/// each served by an arm whose execution path, label, tier, fixture and route all bind.
#[test]
fn every_planned_turnkey_candle_cell_is_served_by_an_arm() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut seen = std::collections::BTreeSet::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        if !key.ends_with(":candle") {
            continue;
        }
        let provider = entry["provider"].as_str().unwrap();
        if !TURNKEY_MEMBERS
            .iter()
            .any(|(member, _)| *member == provider)
        {
            continue;
        }
        seen.insert(key.clone());
        let (_, rest) = key.split_once(':').unwrap();
        let tier = rest.split_once(':').unwrap().0;
        let request = json!({ "planned": {
            "backend": "candle",
            "target": {
                "provider": provider,
                "tier": tier,
                "mode": entry["mode"].clone(),
                "overlay": entry["overlay"].clone(),
                "geometry": entry["geometry"].clone(),
            },
            "loadShape": entry["loadShape"].clone(),
            "strategy": { "rung": "staged_residency", "parameters": {} },
            "calibrationFingerprint": entry["calibrationFingerprint"].clone(),
            "fixture": entry["fixture"].clone(),
        }});
        plain_execution_path(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        still_calibration_label(&request).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(planned_tier(&request).unwrap(), tier, "{key}");
        validate_fixture_binds_tier_and_geometry(&request)
            .unwrap_or_else(|error| panic!("{key}: {error}"));
        assert!(routes_to_five_rung_reference(&request).unwrap(), "{key}");
        // The worker resolves a real `Quant` from the tier for all five
        // (`candle_quant_for_resolved_tier` carves out sana, dense-TE tiers, chroma, sd3.5 and
        // the two FLUX.1 routes — none of these), so the anchor must load the same shape.
        assert_eq!(
            numeric_tier(tier).unwrap().quant.is_some(),
            tier != "bf16",
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
            .map(move |tier| format!("{model}:{tier}:candle"))
    })
    .collect();
    assert_eq!(seen, expected);
}

/// sc-22732 review: the identity table must REFUSE an unknown coordinate, not synthesize one.
/// The per-member arms bind `tier` as a free variable, so before the guard
/// `turnkey_calibration_fingerprint("kolors", "q2")` happily returned
/// `"kolors-candle-kolors-q2-staged-chatglm-unet-f32-vae-v1"` — a well-formed identity no
/// engine publishes — and `validate_turnkey_identity` would have accepted a plan row naming it,
/// then carried a capture all the way to a real load under a fabricated key.
#[test]
fn the_turnkey_candle_identity_table_refuses_an_unshipped_coordinate() {
    // Every shipped coordinate is nameable, so the guard cannot pass by refusing everything.
    for provider in [
        KOLORS_ID,
        IDEOGRAM_ID,
        IDEOGRAM_TURBO_ID,
        LENS_ID,
        LENS_TURBO_ID,
        QWEN_IMAGE_2_1_ID,
        QWEN_IMAGE_2_1_TURBO_ID,
    ] {
        for tier in ["bf16", "q4", "q8"] {
            assert!(
                turnkey_calibration_fingerprint(provider, tier).is_some(),
                "{provider} {tier} is a shipped turnkey cell"
            );
        }
    }
    // Tiers this family does not ship, on a member it DOES ship: the axis the free variable let
    // through.
    for tier in ["q2", "q6", "nvfp4", "fp8", "bf16 ", "", "Q4"] {
        for provider in [
            KOLORS_ID,
            IDEOGRAM_ID,
            IDEOGRAM_TURBO_ID,
            LENS_ID,
            LENS_TURBO_ID,
            QWEN_IMAGE_2_1_ID,
            QWEN_IMAGE_2_1_TURBO_ID,
        ] {
            assert_eq!(
                turnkey_calibration_fingerprint(provider, tier),
                None,
                "{provider} {tier:?} is not a shipped turnkey tier"
            );
        }
    }
    // A provider outside the family stays unnameable at every shipped tier, including the
    // composed Kolors routes, which are separate evidence identities.
    for provider in ["candle_kolors_ipadapter", "candle_kolors_control", "sdxl"] {
        for tier in ["bf16", "q4", "q8"] {
            assert_eq!(
                turnkey_calibration_fingerprint(provider, tier),
                None,
                "{provider} is not a turnkey member"
            );
        }
    }
    // The refusal reaches the plan check as an error rather than a silent pass.
    let request = turnkey_request(
        KOLORS_ID,
        "q2",
        "fresh-five-rung-kolors-q2-1024-seed16402-step2",
    );
    let error = validate_turnkey_identity(&request, KOLORS_ID, "q2")
        .expect_err("an unshipped tier has no identity to match");
    assert!(
        error.contains("no turnkey member kolors at tier q2"),
        "{error}"
    );
}

/// Every turnkey Candle plan row names the production calibration identity its loaded
/// generator publishes (the sc-22732 inference head's per-(route, tier) tables): the one
/// measured key is preserved byte-for-byte at `lens_turbo` q4, no row names a weights-free
/// conformance string or the old every-tier Kolors constant, and the fifteen identities are
/// distinct. `every_planned_turnkey_candle_cell_is_served_by_an_arm` never looks at
/// `calibrationFingerprint`, which is how the first draft of this plan shipped nine rows
/// naming the `…-static-v1-<route>` literals no engine has ever published.
#[test]
fn every_planned_turnkey_candle_cell_names_the_identity_its_loaded_generator_publishes() {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan parses");
    let mut identities = std::collections::BTreeMap::new();
    let mut qwen_image_2_1 = Vec::new();
    let mut qwen_image_2_1_turbo = Vec::new();
    for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
        let provider = entry["provider"].as_str().unwrap();
        if !key.ends_with(":candle") || turnkey_candle_member(provider).is_none() {
            continue;
        }
        // sc-24112: `candle-gen-qwen-image-2-1` publishes ONE identity for every tier, so its
        // three cells share it by construction and are held to the engine constant instead.
        // sc-25907: the same per checkpoint — Turbo's three cells share Turbo's OWN literal.
        if provider == QWEN_IMAGE_2_1_ID || provider == QWEN_IMAGE_2_1_TURBO_ID {
            let bucket = if provider == QWEN_IMAGE_2_1_ID {
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
        let planned = entry["calibrationFingerprint"]
            .as_str()
            .unwrap_or_else(|| panic!("{key}: calibrationFingerprint must be a string"));
        assert!(
            !is_turnkey_weights_free_fingerprint(planned),
            "{key}: {planned} is a weights-free conformance identity no production load returns"
        );
        assert_eq!(
            Some(planned.to_owned()),
            turnkey_calibration_fingerprint(provider, tier),
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
        "fifteen distinct turnkey Candle identities"
    );
    assert_eq!(
        qwen_image_2_1.len(),
        3,
        "one Qwen-Image 2.1 Candle cell per tier"
    );
    for (key, planned) in &qwen_image_2_1 {
        assert_eq!(planned, QWEN_IMAGE_2_1_CANDLE_FINGERPRINT, "{key}");
        assert!(!identities.contains_key(planned.as_str()), "{key}");
    }
    assert_eq!(
        qwen_image_2_1_turbo.len(),
        3,
        "one Qwen-Image 2.1 Turbo Candle cell per tier"
    );
    for (key, planned) in &qwen_image_2_1_turbo {
        assert_eq!(planned, QWEN_IMAGE_2_1_TURBO_CANDLE_FINGERPRINT, "{key}");
        assert!(!identities.contains_key(planned.as_str()), "{key}");
    }
    // The preserved measured key, at exactly the cell `candle-gen-lens` documents as measured.
    assert_eq!(
        identities["lens-candle-cuda-shared-ladder-device-format-blocks-v1"],
        "lens_turbo:q4:candle"
    );
    // The fourteen new cells, spelled out: the strings the sc-22732 inference head mints.
    for (key, identity) in [
        (
            "kolors:q4:candle",
            "kolors-candle-kolors-q4-staged-chatglm-unet-f32-vae-v1",
        ),
        (
            "kolors:q8:candle",
            "kolors-candle-kolors-q8-staged-chatglm-unet-f32-vae-v1",
        ),
        (
            "kolors:bf16:candle",
            "kolors-candle-kolors-bf16-staged-chatglm-unet-f32-vae-v1",
        ),
        (
            "ideogram_4:q4:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-base-q4",
        ),
        (
            "ideogram_4:q8:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-base-q8",
        ),
        (
            "ideogram_4:bf16:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-base-bf16",
        ),
        (
            "ideogram_4_turbo:q4:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-turbo-q4",
        ),
        (
            "ideogram_4_turbo:q8:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-turbo-q8",
        ),
        (
            "ideogram_4_turbo:bf16:candle",
            "ideogram4-candle-request-scoped-staged-residency-v1-turbo-bf16",
        ),
        (
            "lens:q4:candle",
            "lens-base-q4-candle-cuda-shared-ladder-v1",
        ),
        (
            "lens:q8:candle",
            "lens-base-q8-candle-cuda-shared-ladder-v1",
        ),
        (
            "lens:bf16:candle",
            "lens-base-bf16-candle-cuda-shared-ladder-v1",
        ),
        (
            "lens_turbo:q8:candle",
            "lens-turbo-q8-candle-cuda-shared-ladder-v1",
        ),
        (
            "lens_turbo:bf16:candle",
            "lens-turbo-bf16-candle-cuda-shared-ladder-v1",
        ),
    ] {
        assert_eq!(identities[identity], key);
    }
}

/// The arm binds the plan row to the table BEFORE any environment or weight work, naming
/// both strings, so a stale row cannot cost a load.
#[test]
fn a_turnkey_plan_row_naming_a_foreign_identity_is_refused_before_the_load() {
    for (provider, slug, foreign) in [
        (
            IDEOGRAM_ID,
            "ideogram-4",
            "ideogram4-candle-request-scoped-staged-residency-static-v1-base",
        ),
        (
            KOLORS_ID,
            "kolors",
            "kolors-candle-staged-chatglm-unet-f32-vae-v1",
        ),
        (
            LENS_ID,
            "lens",
            "lens-candle-cuda-shared-ladder-device-format-blocks-v1",
        ),
    ] {
        let mut request = turnkey_request(
            provider,
            "q8",
            &format!("fresh-five-rung-{slug}-q8-1024-seed16402-step2"),
        );
        request["planned"]["calibrationFingerprint"] = json!(foreign);
        // `Err` is matched by hand: the `Ok` side carries a loaded generator, which has no
        // `Debug` for `expect_err` to print.
        let error = match load_five_rung_generator(&request) {
            Err(error) => error,
            Ok(_) => panic!("{provider}: a conformance identity is not capturable"),
        };
        assert!(
            error.contains(&format!("plan={foreign}"))
                && error.contains(&turnkey_calibration_fingerprint(provider, "q8").unwrap()),
            "{provider}: {error}"
        );
        // Not an env error: the refusal fired before `SCENEWORKS_<MEMBER>_*` was read.
        assert!(!error.contains("required environment variable"), "{error}");
    }
}

/// The per-member quant decision, stated as data: the two Ideogram routes are the only members
/// whose packed tier must NOT reach the loader, because `candle-gen-ideogram`'s exact route
/// refuses `quantize: Some(_)`; the other three mirror the worker's
/// `candle_quant_for_resolved_tier`, which forwards `Some(Q4)`/`Some(Q8)` for them.
#[test]
fn the_turnkey_candle_members_fold_the_tier_quant_per_member() {
    assert_eq!(
        TURNKEY_CANDLE_MEMBERS
            .iter()
            .map(|member| (member.provider_id, member.tier_quant_reaches_the_loader))
            .collect::<Vec<_>>(),
        vec![
            (KOLORS_ID, true),
            (IDEOGRAM_ID, false),
            (IDEOGRAM_TURBO_ID, false),
            (LENS_ID, true),
            (LENS_TURBO_ID, true),
            (QWEN_IMAGE_2_1_ID, true),
            (QWEN_IMAGE_2_1_TURBO_ID, true),
        ]
    );
    assert_eq!(turnkey_candle_member("candle_kolors_ipadapter"), None);
}

/// The table above is only a declaration. THIS is the binding: the `LoadSpec` the capture hands
/// `catalog.media().load` ([`five_rung_load_spec`]) is built for every planned member x tier and
/// its `quantize` read back.
///
/// sc-22732 review: the fold was reachable only through `load_five_rung_generator`, which loads
/// real weights on a CUDA host, so nothing tied `tier_quant_reaches_the_loader` to the produced
/// spec — an unconditional `spec.with_quant(quant)` would have left both the table test above
/// and the JS mirror green while every Ideogram capture died inside
/// `candle-gen-ideogram::validate_load_shape`. Mirrors the MLX arm's
/// `the_turnkey_root_must_carry_the_planned_tier_and_the_spec_binds_it`.
///
/// The tiers are the ones the plan actually declares for this lane, filtered by the arm's own
/// `validate_five_rung_lane_tier`, so a lane that gains or loses a tier is covered without a
/// second list to keep in step.
#[test]
fn the_turnkey_candle_spec_binds_the_tier_quant_per_member() {
    let mut seen = Vec::new();
    for (provider, slug) in TURNKEY_MEMBERS {
        for tier in ["bf16", "q4", "q8"] {
            if validate_five_rung_lane_tier(provider, tier).is_err() {
                continue;
            }
            let request = turnkey_request(
                provider,
                tier,
                &format!("fresh-five-rung-{slug}-{tier}-1024-seed16402-step2"),
            );
            let spec = five_rung_load_spec(
                &request,
                provider,
                tier,
                PathBuf::from("/nonexistent/turnkey").join(tier),
            )
            .unwrap_or_else(|error| panic!("{provider} {tier}: {error}"));
            // The engine reads the route off the spec: `candle-gen-ideogram`'s
            // `validate_load_shape` refuses a spec whose `resolved_route` is not its own id.
            assert_eq!(
                spec.resolved_route.as_deref(),
                Some(provider),
                "{provider} {tier}: the resolved route must reach the loader"
            );
            // bf16 is dense on every member; a packed tier reaches the loader only when the
            // member says so. Ideogram is the one that must not: its exact directory route
            // refuses `quantize: Some(_)` outright and proves the tier off the shards.
            let expected = match (tier, provider.starts_with("ideogram_4")) {
                ("bf16", _) | (_, true) => None,
                ("q4", false) => Some(Quant::Q4),
                ("q8", false) => Some(Quant::Q8),
                _ => unreachable!("the three shipped tiers"),
            };
            assert_eq!(
                spec.quantize, expected,
                "{provider} {tier}: LoadSpec::quantize"
            );
            // sc-24112: Qwen-Image 2.1 loads under the gen-core default eager shape the worker
            // leaves in place; the rest keep the five-rung deferred shape.
            assert_eq!(
                spec.load_shape,
                if provider == QWEN_IMAGE_2_1_ID || provider == QWEN_IMAGE_2_1_TURBO_ID {
                    LoadShape::EagerMaterialization
                } else {
                    LoadShape::DeferredMaterialization
                },
                "{provider} {tier}: LoadSpec::load_shape"
            );
            seen.push((provider, tier, spec.quantize));
        }
    }
    // Stated as data too, so the loop cannot pass by every cell answering the same way — and so
    // a member or tier silently dropped from the walk fails here rather than vacuously passing.
    assert_eq!(
        seen,
        vec![
            (KOLORS_ID, "bf16", None),
            (KOLORS_ID, "q4", Some(Quant::Q4)),
            (KOLORS_ID, "q8", Some(Quant::Q8)),
            (IDEOGRAM_ID, "bf16", None),
            (IDEOGRAM_ID, "q4", None),
            (IDEOGRAM_ID, "q8", None),
            (IDEOGRAM_TURBO_ID, "bf16", None),
            (IDEOGRAM_TURBO_ID, "q4", None),
            (IDEOGRAM_TURBO_ID, "q8", None),
            (LENS_ID, "bf16", None),
            (LENS_ID, "q4", Some(Quant::Q4)),
            (LENS_ID, "q8", Some(Quant::Q8)),
            (LENS_TURBO_ID, "bf16", None),
            (LENS_TURBO_ID, "q4", Some(Quant::Q4)),
            (LENS_TURBO_ID, "q8", Some(Quant::Q8)),
            (QWEN_IMAGE_2_1_ID, "bf16", None),
            (QWEN_IMAGE_2_1_ID, "q4", Some(Quant::Q4)),
            (QWEN_IMAGE_2_1_ID, "q8", Some(Quant::Q8)),
            (QWEN_IMAGE_2_1_TURBO_ID, "bf16", None),
            (QWEN_IMAGE_2_1_TURBO_ID, "q4", Some(Quant::Q4)),
            (QWEN_IMAGE_2_1_TURBO_ID, "q8", Some(Quant::Q8)),
        ]
    );
}
