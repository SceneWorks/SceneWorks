use sceneworks_options_wasm::{validate_request_json, validate_shape_json};
use serde_json::{json, Value};

#[test]
fn native_transport_matches_the_shared_browser_corpus() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../sceneworks-options/tests/fixtures/validation_corpus.json"
    ))
    .expect("shared validation corpus");
    assert_eq!(corpus["schemaVersion"], 1);
    for kind in [
        "image",
        "video",
        "audio",
        "vector",
        "vectorPrompt",
        "videoUpscale",
        "vqa",
        "interleave",
        "characterTest",
        "promptRefine",
        "yue2",
        "yue2Execution",
        "training",
        "lora_train",
        "control_training",
    ] {
        for expected_ok in [true, false] {
            assert!(
                corpus["cases"]
                    .as_array()
                    .expect("cases")
                    .iter()
                    .any(|case| case["kind"] == kind && case["expected"]["ok"] == expected_ok),
                "missing {kind} corpus case with ok={expected_ok}"
            );
        }
    }
    for case in corpus["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let entry = case["model"]
            .as_str()
            .map(|model| &corpus["modelFacts"][model])
            .unwrap_or(&Value::Null);
        let empty = json!({});
        let entry = if entry.is_null() { &empty } else { entry };
        let result: Value = serde_json::from_str(&validate_request_json(
            case["kind"].as_str().expect("kind"),
            &case["payload"].to_string(),
            &entry.to_string(),
        ))
        .expect("transport JSON");
        assert_eq!(result["validatorSchemaVersion"], 1, "{name}");
        assert_eq!(result["result"], case["expected"], "{name}");
    }
}

#[test]
fn future_request_version_refuses_on_both_full_and_shape_only_paths() {
    let payload = json!({"optionsSchemaVersion": 2, "model": "flux_dev", "prompt": "x"});
    for raw in [
        validate_request_json("image", &payload.to_string(), "{}"),
        validate_shape_json("image", &payload.to_string()),
    ] {
        let answer: Value = serde_json::from_str(&raw).expect("transport JSON");
        assert_eq!(answer["result"]["ok"], false);
        assert_eq!(
            answer["result"]["refusal"]["code"],
            "UnsupportedSchemaVersion"
        );
        assert_eq!(answer["result"]["refusal"]["field"], "optionsSchemaVersion");
    }
}

#[test]
fn conflicting_legacy_aliases_refuse_before_either_validator_path() {
    let payload = json!({
        "model": "flux_dev", "prompt": "x",
        "ipAdapterScale": 0.4, "ip_adapter_scale": 0.8,
    });
    for raw in [
        validate_request_json("image", &payload.to_string(), "{}"),
        validate_shape_json("image", &payload.to_string()),
    ] {
        let answer: Value = serde_json::from_str(&raw).expect("transport JSON");
        assert_eq!(answer["result"]["ok"], false);
        assert_eq!(answer["result"]["refusal"]["code"], "ConflictingAliases");
        assert_eq!(answer["result"]["refusal"]["field"], "ipAdapterScale");
    }
}
