//! Actual exported model facts through the exact portable/native owner. Metadata only.
use sceneworks_core::{options, options_context, registry::builtin_model_spec, request_validation};
use serde_json::{json, Value};

fn facts(id: &str, mode: &str) -> Value {
    options_context::for_model(
        builtin_model_spec(id).expect("shipped model"),
        "mlx",
        mode,
        None,
    )
    .model
}
fn compare(
    kind: &str,
    payload: Value,
    entry: Value,
) -> Result<(), options::request_validation::ValidationRefusal> {
    let payload = payload.as_object().unwrap();
    let entry = entry.as_object().unwrap();
    let portable = options::request_validation::validate(kind, payload, entry);
    assert_eq!(portable, request_validation::validate(kind, payload, entry));
    portable
}

#[test]
fn strict_raw_image_integers_and_numeric_string_steps_do_not_disappear() {
    let entry = facts("flux_dev", "text_to_image");
    let valid =
        json!({"projectId":"p","model":"flux_dev","prompt":"a tree","advanced":{"steps":"20"}});
    compare("image", valid.clone(), entry.clone()).unwrap();
    for (key, value, field) in [
        ("width", json!("1024"), "width"),
        ("count", json!(1.5), "count"),
        ("advanced", json!({"steps":"oops"}), "advanced.steps"),
        ("referenceAssetIds", json!(["a", 17]), "referenceAssetIds"),
    ] {
        let mut request = valid.clone();
        request[key] = value;
        let refusal = compare("image", request, entry.clone()).unwrap_err();
        assert_eq!(refusal.code, "invalid_generation_option");
        assert_eq!(refusal.field, field);
        assert_eq!(refusal.context["model"], "flux_dev");
    }
}

#[test]
fn defaults_and_capabilities_come_from_real_model_facts() {
    let entry = facts("flux_dev", "text_to_image");
    let payload = json!({"projectId":"p","model":"flux_dev","prompt":"a tree"});
    let resolved = options::request_validation::resolve(
        "image",
        payload.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap();
    let mut native = payload.as_object().unwrap().clone();
    native.insert("modelManifestEntry".into(), entry);
    let request = sceneworks_core::image_request::ImageRequest::from_payload(&native);
    assert_eq!(resolved["width"], request.width);
    assert_eq!(resolved["height"], request.height);
    assert_eq!(resolved["count"], request.count);
    assert!(payload.get("count").is_none());
    let mut supported_low_res = payload.clone();
    supported_low_res["advanced"] = json!({"lowResStart":{"version":1,"k":2,"scale":0.5}});
    compare(
        "image",
        supported_low_res,
        facts("flux_dev", "text_to_image"),
    )
    .expect("flux_dev supports lowResStart");

    let mut invalid_low_res_k = payload.clone();
    invalid_low_res_k["advanced"] =
        json!({"steps":4,"lowResStart":{"version":1,"k":4,"scale":0.5}});
    assert_eq!(
        compare(
            "image",
            invalid_low_res_k,
            facts("flux_dev", "text_to_image")
        )
        .unwrap_err()
        .field,
        "advanced.lowResStart.k"
    );

    let unsupported = json!({
        "projectId": "p",
        "model": "qwen_image_2_1",
        "prompt": "a tree",
        "advanced": {"lowResStart":{"version":1,"k":2,"scale":0.5}}
    });
    assert_eq!(
        compare(
            "image",
            unsupported,
            facts("qwen_image_2_1", "text_to_image")
        )
        .unwrap_err()
        .field,
        "advanced.lowResStart"
    );
}

#[test]
fn video_limits_apply_before_tolerant_worker_parsing() {
    let entry = facts("ltx_2_5", "text_to_video");
    let valid = json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":"a tree"});
    compare("video", valid.clone(), entry.clone()).unwrap();
    for (key, value, field) in [
        ("duration", json!(30), "duration"),
        ("fps", json!(60), "fps"),
        (
            "referenceAudioAssetIds",
            json!(["sound"]),
            "referenceAssetIds",
        ),
        (
            "sourceClipAssetIds",
            json!(["clip", 7]),
            "sourceClipAssetIds",
        ),
    ] {
        let mut request = valid.clone();
        request[key] = value;
        assert_eq!(
            compare("video", request, entry.clone()).unwrap_err().field,
            field
        );
    }
}

#[test]
fn independent_pose_budget_and_moss_positive_embeddings_remain_supported() {
    let mut request = json!({"projectId":"p","model":"flux_dev","prompt":"poses","advanced":{"poses":vec![json!({"keypoints":vec![json!([0.123456789,0.987654321,1.0]);128]});64]}});
    options::request_validation::validate_shape("image", request.as_object().unwrap()).unwrap();
    request["advanced"]["poses"]
        .as_array_mut()
        .unwrap()
        .push(json!({}));
    assert!(
        options::request_validation::validate_shape("image", request.as_object().unwrap()).is_err()
    );
    let tensor = json!({"projectId":"p","model":"moss_image_base","mode":"edit_image","prompt":"","advanced":{"mossImageAdvanced":{"positive":{"values":{"assetId":"tensor","sha256":"a"}}}}});
    options::request_validation::validate_shape("image", tensor.as_object().unwrap()).unwrap();
}

#[test]
fn absent_image_reference_cap_is_not_the_video_default() {
    assert_eq!(
        options::image_limits::image_reference_limit_error("custom", 99, &Default::default()),
        None
    );
    let caps = options::video_request::reference_caps(&Default::default());
    assert_eq!((caps.images, caps.clips, caps.audio), (8, 8, 0));
}

#[test]
fn portable_corpus_model_facts_are_actual_exports() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../sceneworks-options/tests/fixtures/validation_corpus.json"
    ))
    .unwrap();
    for (id, entry) in corpus["modelFacts"].as_object().unwrap() {
        if id == "z_image_turbo_lora" || id == "krea_2_control" {
            let target = sceneworks_core::training::effective_training_targets_for_candle(
                id == "krea_2_control",
            )
            .targets
            .into_iter()
            .find(|target| target.id == *id)
            .unwrap();
            assert_eq!(serde_json::to_value(target).unwrap(), *entry);
            continue;
        }
        if id == "vectorPrompt" {
            assert_eq!(entry["raster"], corpus["modelFacts"]["flux_dev"]);
            assert_eq!(entry["vector"], corpus["modelFacts"]["starvector_1b"]);
            continue;
        }
        let mode = match entry["type"].as_str().unwrap() {
            "image" => "text_to_image",
            "video" => "text_to_video",
            "audio" => "text_to_speech",
            "vector" => "image_to_svg",
            _ => unreachable!(),
        };
        assert_eq!(
            &facts(id, mode),
            entry,
            "model facts must be regenerated from the actual ModelSpec: {id}"
        );
    }
}

#[test]
fn route_facts_are_compiled_and_ignore_live_availability() {
    let mut forged = json!({"id":"custom-video","optionRouteFacts":{"videoMlxRouted":true}});
    sceneworks_core::registry::project_option_route_facts(&mut forged);
    assert!(forged.get("optionRouteFacts").is_none());
    let mut entry = facts("ltx_2_5", "text_to_video");
    let trusted = entry["optionRouteFacts"].clone();
    entry["optionRouteFacts"] = json!({"videoMlxRouted":false});
    sceneworks_core::registry::project_option_route_facts(&mut entry);
    assert_eq!(entry["optionRouteFacts"], trusted);
    let payload = json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":"tree"});
    let before = options::request_validation::resolve(
        "video",
        payload.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap();
    entry["installed"] = json!(false);
    entry["usable"] = json!(false);
    entry["installState"] = json!("missing");
    entry["availability"] = json!({"device":"unavailable"});
    assert_eq!(
        before,
        options::request_validation::resolve(
            "video",
            payload.as_object().unwrap(),
            entry.as_object().unwrap()
        )
        .unwrap()
    );
}

#[test]
fn portable_video_refusal_agrees_with_native_lane_owner() {
    use sceneworks_core::{contracts::JobType, jobs_store::video_request_is_claimable_by_any_lane};
    for (model, mode, expected) in [
        ("ltx_2_5", "text_to_video", true),
        ("wan_2_2_i2v_14b", "first_last_frame", false),
    ] {
        let entry = facts(model, mode);
        let mut payload = json!({"projectId":"p","model":model,"mode":mode,"prompt":"tree"});
        if mode == "first_last_frame" {
            payload["sourceAssetId"] = json!("first");
            payload["lastFrameAssetId"] = json!("last");
        }
        payload["modelManifestEntry"] = entry.clone();
        assert_eq!(
            video_request_is_claimable_by_any_lane(
                &JobType::VideoGenerate,
                payload.as_object().unwrap()
            ),
            expected
        );
        let decision = compare("video", payload, entry);
        assert_eq!(decision.is_ok(), expected);
        if let Err(error) = decision {
            assert_eq!(error.field, "mode");
        }
    }
}

#[test]
fn effective_defaults_do_not_fabricate_unsupported_family_controls() {
    for model in ["moss_image_base", "moss_image_sft", "supra2_img"] {
        let entry = facts(model, "text_to_image");
        let payload = json!({"projectId":"p","model":model,"prompt":"tree"});
        let resolved = options::request_validation::resolve(
            "image",
            payload.as_object().unwrap(),
            entry.as_object().unwrap(),
        )
        .unwrap();
        assert!(resolved["advanced"].get("flashAttn").is_none());
        sceneworks_core::moss_image::native_request(&resolved, entry.as_object().unwrap()).unwrap();
        sceneworks_core::supra2::validate_request(&resolved, entry.as_object().unwrap()).unwrap();
    }
}

#[test]
fn shared_promptless_video_requires_authoritative_svd_facts() {
    use sceneworks_core::{options_context, registry::builtin_model_spec, request_validation};
    let model = options_context::for_model(
        builtin_model_spec("svd").unwrap(),
        "candle",
        "image_to_video",
        None,
    )
    .model;
    assert_eq!(model["promptless"], true);
    let payload = serde_json::json!({"projectId":"p","model":"svd","mode":"image_to_video","prompt":"","sourceAssetId":"image"});
    let resolved = request_validation::resolve(
        "video",
        payload.as_object().unwrap(),
        model.as_object().unwrap(),
    )
    .unwrap();
    assert_eq!(resolved["prompt"], "");
    for mut facts in [model.clone(), serde_json::json!({})] {
        if facts.get("id").is_some() {
            facts["promptless"] = serde_json::json!(false);
        }
        let error = request_validation::resolve(
            "video",
            payload.as_object().unwrap(),
            facts.as_object().unwrap(),
        )
        .unwrap_err();
        assert_eq!(error.field, "prompt");
    }
    let mut forged = payload.clone();
    forged["model"] = serde_json::json!("ltx_2_5");
    forged["promptless"] = serde_json::json!(true);
    assert_eq!(
        request_validation::resolve(
            "video",
            forged.as_object().unwrap(),
            model.as_object().unwrap()
        )
        .unwrap_err()
        .field,
        "prompt"
    );
    let mut oversized = payload.clone();
    oversized["prompt"] = serde_json::json!("x".repeat(4001));
    assert_eq!(
        request_validation::validate_shape("video", oversized.as_object().unwrap())
            .unwrap_err()
            .field,
        "prompt"
    );
}
