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
        "model": "flux3_klein",
        "prompt": "a tree",
        "advanced": {"lowResStart":{"version":1,"k":2,"scale":0.5}}
    });
    assert_eq!(
        compare("image", unsupported, facts("flux3_klein", "text_to_image"))
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
fn flash_attention_is_accepted_but_never_injected_into_effective_options() {
    // SW-7276: `advanced.flashAttn` is accepted for non-MOSS image families, but an absent value
    // stays absent so an unchanged job's effective payload (replay digests, recipe identity) does
    // not move; the worker treats absent as on. An authored value is kept.
    for model in ["flux_dev", "qwen_image_2_1", "flux3_klein"] {
        let entry = facts(model, "text_to_image");
        let entry = entry.as_object().unwrap();
        let bare = json!({"projectId":"p","model":model,"prompt":"tree"});
        let resolved =
            options::request_validation::resolve("image", bare.as_object().unwrap(), entry)
                .unwrap();
        assert!(resolved["advanced"].get("flashAttn").is_none(), "{model}");
        let authored =
            json!({"projectId":"p","model":model,"prompt":"tree","advanced":{"flashAttn":false}});
        let resolved =
            options::request_validation::resolve("image", authored.as_object().unwrap(), entry)
                .unwrap();
        assert_eq!(resolved["advanced"]["flashAttn"], json!(false), "{model}");
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

/// A model that declares a long encoder window takes prompts as long as that window, instead of the
/// flat 4,000-character cap that refused MiniMax-H3-Image prompts its 4,096-token encoder reads in
/// full. A model with no declared window keeps the 4,000 cap.
#[test]
fn image_prompt_ceiling_follows_the_declared_token_window() {
    let h3 = facts("minimax_h3_image", "text_to_image");
    assert_eq!(
        sceneworks_core::image_prompt_char_limit(h3.as_object().unwrap()),
        sceneworks_core::MAX_IMAGE_PROMPT_CHARS
    );
    let long = "x".repeat(10_000);
    compare(
        "image",
        json!({"projectId":"p","model":"minimax_h3_image","prompt":long}),
        h3.clone(),
    )
    .expect("a 10,000-character prompt fits H3-Image's 4,096-token window");
    let over = "x".repeat(sceneworks_core::MAX_IMAGE_PROMPT_CHARS + 1);
    assert_eq!(
        compare(
            "image",
            json!({"projectId":"p","model":"minimax_h3_image","prompt":over}),
            h3,
        )
        .unwrap_err()
        .field,
        "prompt"
    );

    let flux = facts("flux_dev", "text_to_image");
    assert_eq!(
        sceneworks_core::image_prompt_char_limit(flux.as_object().unwrap()),
        sceneworks_core::MAX_PROMPT_CHARS
    );
    let refusal = compare(
        "image",
        json!({"projectId":"p","model":"flux_dev","prompt":"x".repeat(4001)}),
        flux,
    )
    .unwrap_err();
    assert_eq!(refusal.field, "prompt");
    assert_eq!(
        refusal.detail,
        "prompt must be between 1 and 4000 characters"
    );
}

/// SW-6132: `advanced.lossyAttention` is validated against the catalog's `lossyAttention`
/// capability, which `sceneworks-core` projects from the checked-in copy of inference's
/// `SHARED_ATTENTION_ROUTING` (the candle worker's test pins the copy to the linked table).
#[test]
fn lossy_attention_is_admitted_only_where_the_shared_routing_serves_it() {
    let int8 = json!({ "int8": {} });
    let sol = json!({ "sol": { "topkRatio": 0.5 } });
    let resolve = |model: &str, lossy: &Value| {
        let entry = facts(model, "text_to_image");
        let payload = json!({"projectId":"p","model":model,"prompt":"tree",
                             "advanced":{"lossyAttention": lossy}});
        options::request_validation::resolve(
            "image",
            payload.as_object().unwrap(),
            entry.as_object().unwrap(),
        )
    };
    for model in ["krea_2_turbo", "flux_dev"] {
        let capability = &facts(model, "text_to_image")["lossyAttention"];
        assert_eq!(capability["int8"]["served"], json!(true), "{model}");
        assert_eq!(capability["sol"]["served"], json!(true), "{model}");
        for lossy in [&int8, &sol] {
            let resolved = resolve(model, lossy).unwrap_or_else(|e| panic!("{model}: {e:?}"));
            // The effective options carry the same configuration, with inference's defaults
            // materialized for the requested kernel (and nothing for the one not requested).
            let parse = options::lossy_attention::parse_wire_value;
            let effective = parse(&resolved["advanced"]["lossyAttention"]).unwrap();
            let authored = parse(lossy).unwrap();
            assert_eq!(effective.int8.is_some(), authored.int8.is_some(), "{model}");
            assert_eq!(effective.sol.is_some(), authored.sol.is_some(), "{model}");
            if let (Some(effective), Some(authored)) = (effective.int8, authored.int8) {
                assert_eq!(effective.pv_precision(), authored.pv_precision(), "{model}");
                assert_eq!(effective.k_centering(), authored.k_centering(), "{model}");
            }
            if let (Some(effective), Some(authored)) = (effective.sol, authored.sol) {
                assert_eq!(
                    effective.selection().unwrap(),
                    authored.selection().unwrap()
                );
                assert_eq!(effective.sink_blocks(), authored.sink_blocks());
                assert_eq!(effective.sink_q(), authored.sink_q());
                assert_eq!(effective.tail(), authored.tail());
                assert_eq!(
                    effective.query_chunk_blocks(),
                    authored.query_chunk_blocks()
                );
            }
        }
    }
    // INT8 only: block-causal attention gives Sol-Attn no self-attention to route.
    assert!(resolve("qwen_image_2_1", &int8).is_ok());
    let refused = resolve("qwen_image_2_1", &sol).unwrap_err();
    assert_eq!(refused.field, "advanced.lossyAttention.sol");
    assert!(
        refused.detail.contains("block-causally"),
        "{}",
        refused.detail
    );
    // Neither: 256-wide heads.
    let refused = resolve("ideogram_4", &int8).unwrap_err();
    assert!(refused.detail.contains("256-wide"), "{}", refused.detail);
    // Bespoke attention: no capability is projected, so the option is refused.
    for model in ["sana_1600m", "moss_image_base", "supra2_img"] {
        assert!(
            facts(model, "text_to_image")
                .get("lossyAttention")
                .is_none(),
            "{model}"
        );
        assert!(resolve(model, &int8).is_err(), "{model}");
    }
    // Out of range is a shape refusal whatever the model.
    let refused = resolve("krea_2_turbo", &json!({ "sol": { "topkRatio": 0 } })).unwrap_err();
    assert_eq!(refused.field, "advanced.lossyAttention.sol.topkRatio");
    // Absent stays absent (no injected default).
    let entry = facts("krea_2_turbo", "text_to_image");
    let bare = json!({"projectId":"p","model":"krea_2_turbo","prompt":"tree"});
    let resolved = options::request_validation::resolve(
        "image",
        bare.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap();
    assert!(resolved["advanced"].get("lossyAttention").is_none());
}

/// SW-6132: the typed image request carries the option to the worker unchanged, and the
/// provider wire (the recipe's raw settings) records it.
#[test]
fn lossy_attention_round_trips_through_the_typed_image_request() {
    let entry = facts("krea_2_turbo", "text_to_image");
    let lossy = json!({ "int8": { "pvPrecision": "int8", "kCentering": "always" } });
    let payload = json!({"projectId":"p","model":"krea_2_turbo","prompt":"tree",
                         "advanced":{"lossyAttention": lossy}});
    let resolved = options::request_validation::resolve(
        "image",
        payload.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap();
    let request = sceneworks_core::image_request::ImageRequest::from_resolved(
        payload.as_object().unwrap(),
        &resolved,
    )
    .unwrap();
    let wire = request.lossy_attention().expect("typed lossy attention");
    assert_eq!(serde_json::to_value(wire).unwrap(), lossy);
    assert_eq!(request.effective_advanced_wire()["lossyAttention"], lossy);
}

/// SW-6132: video models take the option per engine; the capability is every mode's engines.
#[test]
fn lossy_attention_is_validated_for_video_models() {
    let entry = facts("wan_2_2", "text_to_video");
    assert_eq!(entry["lossyAttention"]["sol"]["served"], json!(true));
    let payload = json!({"projectId":"p","model":"wan_2_2","prompt":"a fox","mode":"text_to_video",
                         "advanced":{"lossyAttention":{"sol":{"topkRatio":0.25}}}});
    options::request_validation::validate(
        "video",
        payload.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap();
    let request = sceneworks_core::video_request::VideoRequest::from_resolved(
        payload.as_object().unwrap(),
        payload.as_object().unwrap(),
    )
    .unwrap();
    let bound = request.options().lossy_attention.as_ref().expect("bound");
    assert_eq!(bound.value.sol.unwrap().topk_ratio, Some(0.25));
}
