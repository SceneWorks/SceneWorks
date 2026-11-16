//! Fixed policy expectations, deliberately independent of export_validation_corpus and its
//! generated `expected` values. Sources frozen before extraction:
//! - SceneWorks e94ecbace6: core moss_image.rs/supra2.rs and worker sensenova_jobs.rs
//!   (`payload_int`/`json_to_i64`, interleave parameter defaults).
//! - inference 4a36a7f8: gen-core MOSS resolve/edit defaults and checkpoint Supra2 resolve.
//!
//! Literals here must not be replaced by policy constants or generated resolver output.
use sceneworks_options::request_validation::resolve;
use serde_json::{json, Value};

fn image(model: &str) -> Value {
    json!({"projectId":"frozen-policy","model":model,"mode":"text_to_image","prompt":"日本語 <tag> {\"items\":[]}"})
}
fn resolved(kind: &str, payload: &Value, facts: &Value) -> Value {
    let authored = payload.clone();
    let entry = &facts[payload["model"].as_str().unwrap()];
    let effective = resolve(
        kind,
        payload.as_object().unwrap(),
        entry.as_object().unwrap(),
    )
    .unwrap_or_else(|error| panic!("{payload}: {error:?}"));
    assert_eq!(
        payload, &authored,
        "resolution must preserve authored input"
    );
    Value::Object(effective)
}
fn refused(payload: &Value, facts: &Value, reason: &str, field: Option<&str>) {
    let error = resolve(
        "image",
        payload.as_object().unwrap(),
        facts[payload["model"].as_str().unwrap()]
            .as_object()
            .unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code, "invalid_generation_option");
    assert!(error.detail.contains(reason), "{payload}: {error:?}");
    if let Some(field) = field {
        assert_eq!(error.field, field, "{payload}: {error:?}");
    }
    assert_eq!(
        *error.context,
        json!({
            "kind":"image", "model":payload["model"], "mode":payload["mode"], "field":error.field,
        })
    );
}

/// Run the same independent assertions on native Rust and the actual WASM binary.
pub fn run(facts: &Value) {
    for model in ["moss_image_base", "moss_image_sft"] {
        let payload = image(model);
        let effective = resolved("image", &payload, facts);
        assert_eq!(
            (
                effective["width"].as_u64(),
                effective["height"].as_u64(),
                effective["count"].as_u64()
            ),
            (Some(1024), Some(1024), Some(1))
        );
        let advanced = &effective["advanced"];
        assert_eq!(advanced["steps"], 50);
        assert_eq!(advanced["guidanceScale"], 3.0);
        assert_eq!(advanced["schedulerShift"], 5.0);
        assert_eq!(advanced["mossImage"]["acceleration"], "off");
        assert_eq!(advanced["mossImage"]["max_sequence_length"], 4096);
        assert_eq!(advanced["mossImage"]["training_shift"], 1.0);
        assert_eq!(advanced["mossImage"]["uncond_skip_layers"], json!([22]));
        assert!(
            advanced.get("flashAttn").is_none(),
            "generic defaults must not fabricate unsupported presence"
        );
        assert_eq!(effective["prompt"], payload["prompt"]);

        let mut edit = payload.clone();
        edit["mode"] = json!("edit_image");
        edit["referenceAssetIds"] = json!(["source"]);
        let effective = resolved("image", &edit, facts);
        assert_eq!(
            (effective["width"].as_u64(), effective["height"].as_u64()),
            (Some(512), Some(512))
        );
        let advanced = &effective["advanced"];
        assert_eq!(advanced["steps"], 30);
        assert_eq!(advanced["guidanceScale"], 4.0);
        assert_eq!(advanced["schedulerShift"], 5.0);
        assert_eq!(advanced["mossImage"]["uncond_skip_layers"], json!([]));
        assert_eq!(
            advanced["mossImageEdit"],
            json!({"reference_max_edge":512,"use_kv_cache":true})
        );
        edit["advanced"] = json!({"mossImage":{"acceleration":"unmerged_bf16"}});
        refused(&edit, facts, "edit requires ordinary Base/SFT", None);
    }
    for precision in ["unmerged_bf16", "full_fp32"] {
        let mut payload = image("moss_image_sft");
        payload["advanced"] = json!({"mossImage":{"acceleration":precision}});
        let effective = resolved("image", &payload, facts);
        let advanced = &effective["advanced"];
        assert_eq!(advanced["steps"], 8);
        assert_eq!(advanced["guidanceScale"], 0.0);
        assert_eq!(advanced["mossImage"]["acceleration"], precision);
        for key in ["schedulerShift", "sampler", "scheduler"] {
            assert!(advanced.get(key).is_none(), "{key}: {effective}");
        }
        for key in ["training_shift", "uncond_skip_layers"] {
            assert!(advanced["mossImage"][key].is_null(), "{key}: {effective}");
        }
        let mut base = payload.clone();
        base["model"] = json!("moss_image_base");
        refused(&base, facts, "DMD requires the exact SFT parent", None);
        for (key, value, reason) in [
            (
                "steps",
                json!(9),
                "DMD steps must select a trained prefix in 1..=8",
            ),
            (
                "guidanceScale",
                json!(3.0),
                "DMD does not support CFG or negative prompts",
            ),
            ("sampler", json!("euler"), "DMD uses fixed step codes"),
        ] {
            let mut invalid = payload.clone();
            invalid["advanced"][key] = value;
            refused(&invalid, facts, reason, None);
        }
    }
    let supra = image("supra2_img");
    let effective = resolved("image", &supra, facts);
    assert_eq!(
        (
            effective["width"].as_u64(),
            effective["height"].as_u64(),
            effective["count"].as_u64()
        ),
        (Some(256), Some(256), Some(1))
    );
    assert_eq!(effective["advanced"]["steps"], 50);
    assert_eq!(effective["advanced"]["guidanceScale"], 3.0);
    assert_eq!(effective["advanced"]["sampler"], "euler");
    assert_eq!(effective["advanced"]["scheduler"], "supra2_uniform_flow");
    for (key, value, reason, field) in [
        (
            "negativePrompt",
            json!("bad"),
            "a negative prompt is not supported",
            None,
        ),
        (
            "sourceAssetId",
            json!("source"),
            "`sourceAssetId` is not supported",
            Some("sourceAssetId"),
        ),
        (
            "prompts",
            json!([supra["prompt"]]),
            "`prompts` is not supported",
            Some("prompts"),
        ),
        (
            "width",
            json!(512),
            "native resolution is 256x256 only",
            None,
        ),
        (
            "width",
            json!("256"),
            "width has an invalid value for Integer",
            Some("width"),
        ),
    ] {
        let mut invalid = supra.clone();
        invalid[key] = value;
        refused(&invalid, facts, reason, field);
    }
    let mut invalid = supra.clone();
    invalid["advanced"] = json!({"schedulerShift":5.0});
    refused(
        &invalid,
        facts,
        "advanced.schedulerShift is not supported",
        Some("advanced.schedulerShift"),
    );
    invalid["advanced"] = json!({"sampler":"heun"});
    refused(&invalid, facts, "is not offered", Some("advanced.sampler"));
    invalid["advanced"] = json!({"steps":"bad"});
    refused(
        &invalid,
        facts,
        "positive unsigned 32-bit integer",
        Some("advanced.steps"),
    );

    let base = json!({"projectId":"frozen-policy","model":"sensenova_u1_8b","prompt":"tree"});
    for value in [
        json!("2.5"),
        json!("1e100"),
        json!("bad"),
        Value::Null,
        json!({}),
    ] {
        let mut payload = base.clone();
        payload["width"] = value.clone();
        payload["height"] = value.clone();
        payload["maxImages"] = value.clone();
        payload["advanced"] = json!({"numInferenceSteps":value,"maxNewTokens":value});
        let effective = resolved("interleave", &payload, facts);
        assert_eq!(
            (effective["width"].as_u64(), effective["height"].as_u64()),
            (Some(2048), Some(1152))
        );
        assert_eq!(effective["maxImages"], 6);
        assert_eq!(effective["advanced"]["maxImages"], 6);
        assert_eq!(effective["advanced"]["numInferenceSteps"], 50);
        assert_eq!(effective["advanced"]["maxNewTokens"], 2048);
        let vqa = json!({"projectId":"frozen-policy","model":"sensenova_u1_8b","question":"What?","sourceAssetId":"source","maxNewTokens":value,"maxImagePixels":value});
        let effective = resolved("vqa", &vqa, facts);
        assert_eq!(effective["maxNewTokens"], 256);
        assert_eq!(effective["maxImagePixels"], 589824);
    }
    for (top, advanced, expected) in [
        (json!(9), json!(3), 3),
        (json!(2), Value::Null, 6),
        (json!(2), json!("bad"), 6),
        (json!(2), json!(99), 10),
    ] {
        let mut payload = base.clone();
        payload["maxImages"] = top;
        payload["advanced"] = json!({"maxImages":advanced});
        let effective = resolved("interleave", &payload, facts);
        assert_eq!(effective["maxImages"], expected);
        assert_eq!(effective["advanced"]["maxImages"], expected);
    }
    let mut payload = base.clone();
    payload["maxImages"] = json!(2.9);
    assert_eq!(resolved("interleave", &payload, facts)["maxImages"], 2);
    payload["maxImages"] = json!("99");
    assert_eq!(resolved("interleave", &payload, facts)["maxImages"], 10);
    payload["advanced"] = json!({"timestepShift":1.5,"schedulerShift":9});
    assert_eq!(
        resolved("interleave", &payload, facts)["advanced"]["timestepShift"],
        1.5
    );
}

pub fn facts() -> Value {
    let corpus: Value =
        serde_json::from_str(include_str!("../fixtures/validation_corpus.json")).unwrap();
    corpus["modelFacts"].clone()
}
