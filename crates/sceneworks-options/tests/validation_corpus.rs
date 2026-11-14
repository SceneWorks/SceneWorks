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
