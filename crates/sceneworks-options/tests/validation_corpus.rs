use sceneworks_options::request_validation::{resolve, VALIDATOR_SCHEMA_VERSION};
use serde_json::{json, Value};
#[test]
fn native_portable_validation_corpus() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/validation_corpus.json")).unwrap();
    assert_eq!(corpus["schemaVersion"], VALIDATOR_SCHEMA_VERSION);
    for case in corpus["cases"].as_array().unwrap() {
        let empty = serde_json::Map::new();
        let entry = corpus["modelFacts"][case["model"].as_str().unwrap()]
            .as_object()
            .unwrap_or(&empty);
        let actual = match resolve(
            case["kind"].as_str().unwrap(),
            case["payload"].as_object().unwrap(),
            entry,
        ) {
            Ok(resolved) => json!({"ok":true,"resolved":resolved}),
            Err(error) => json!({"ok":false,"refusal":error}),
        };
        assert_eq!(actual, case["expected"], "{}", case["name"]);
        if let Some(resolved) = actual.get("resolved").and_then(Value::as_object) {
            let second =
                resolve(case["kind"].as_str().unwrap(), resolved, entry).unwrap_or_else(|error| {
                    panic!(
                        "effective copy must remain valid for {}: {error:?}",
                        case["name"]
                    )
                });
            assert_eq!(
                &second, resolved,
                "resolution must be idempotent for {}",
                case["name"]
            );
        }
    }
}

#[test]
fn ltx_extend_fixture_asserts_the_public_mode_refusal() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/validation_corpus.json")).unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "video LTX extend mode refusal")
        .expect("the named mode-refusal fixture must remain in the shared corpus");
    let facts = corpus["modelFacts"]["ltx_2_5"].as_object().unwrap();
    let refusal = resolve("video", case["payload"].as_object().unwrap(), facts).unwrap_err();
    assert_eq!(refusal.field, "mode");
    assert!(
        refusal.detail.starts_with(
            "ltx_2_5 cannot render the \"extend_clip\" mode — no backend implements it"
        ),
        "{refusal:?}"
    );
    assert_eq!(refusal.context["mode"], "extend_clip");
}

#[test]
fn sensenova_full_resolver_preserves_legacy_values_and_authored_omission() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/validation_corpus.json")).unwrap();
    let facts = corpus["modelFacts"]["sensenova_u1_8b"].as_object().unwrap();
    let authored = json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"9999","height":12.9,"advanced":{"timestepShift":1.5,"schedulerShift":9}});
    let resolved = resolve("interleave", authored.as_object().unwrap(), facts).unwrap();
    assert_eq!(resolved["width"], 4096);
    assert_eq!(resolved["height"], 256);
    assert_eq!(resolved["advanced"]["timestepShift"], 1.5);
    assert_eq!(resolved["advanced"]["schedulerShift"], 9.0);
    assert_eq!(authored["width"], "9999");
    let omitted = json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree"});
    let resolved = resolve("interleave", omitted.as_object().unwrap(), facts).unwrap();
    assert_eq!(resolved["width"], 2048);
    assert_eq!(resolved["height"], 1152);
    for (top, advanced, expected) in [
        (None, None, 6),
        (Some(json!(2)), None, 2),
        (Some(json!(99)), None, 10),
        (Some(json!(0)), None, 1),
        (Some(json!(2)), Some(json!(4)), 4),
    ] {
        let mut payload = omitted.clone();
        if let Some(value) = top {
            payload["maxImages"] = value;
        }
        if let Some(value) = advanced {
            payload["advanced"] = json!({"maxImages":value});
        }
        let resolved = resolve("interleave", payload.as_object().unwrap(), facts).unwrap();
        assert_eq!(resolved["maxImages"], expected);
        assert_eq!(resolved["advanced"]["maxImages"], expected);
    }
    for invalid in [
        json!("NaN"),
        json!("inf"),
        json!(null),
        json!({}),
        json!("bad"),
    ] {
        let mut payload = omitted.clone();
        payload["width"] = invalid;
        assert_eq!(
            resolve("interleave", payload.as_object().unwrap(), facts)
                .unwrap_err()
                .field,
            "width"
        );
    }
}
