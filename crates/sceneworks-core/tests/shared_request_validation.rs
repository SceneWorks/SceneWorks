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
    let mut unsupported = payload.clone();
    unsupported["advanced"] = json!({"lowResStart":{"version":1,"k":2,"scale":0.5}});
    assert_eq!(
        compare("image", unsupported, facts("flux_dev", "text_to_image"))
            .unwrap_err()
            .field,
        "advanced.lowResStart"
    );
}

#[test]
fn video_limits_apply_before_tolerant_worker_parsing() {
    let entry = facts("mochi_1", "text_to_video");
    let valid = json!({"projectId":"p","model":"mochi_1","mode":"text_to_video","prompt":"a tree"});
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
