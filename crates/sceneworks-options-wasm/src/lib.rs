//! Browser transport for the portable request validator. The Rust options crate owns every
//! coercion, default, exclusion and cross-field rule; this crate only decodes and encodes JSON.

use serde_json::{json, Map, Value};
use wasm_bindgen::prelude::*;

fn decode_object(field: &str, raw: &str) -> Result<Map<String, Value>, Value> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(value)) => Ok(value),
        Ok(_) => Err(json!({
            "code": "invalid_generation_option",
            "field": field,
            "detail": "Expected a JSON object.",
            "context": {},
        })),
        Err(error) => Err(json!({
            "code": "invalid_generation_option",
            "field": field,
            "detail": error.to_string(),
            "context": {},
        })),
    }
}

fn upgraded_request(kind: &str, payload_json: &str) -> Result<Value, Value> {
    let payload = decode_object("payload", payload_json)?;
    sceneworks_options::upgrade::upgrade(
        &Value::Object(payload),
        sceneworks_options::upgrade::request_carrier(kind),
    )
    .map_err(|error| {
        json!({
            "code": error.code,
            "field": error.field,
            "detail": error.message,
            "context": { "kind": kind },
        })
    })
}

fn response(result: Result<Value, Value>) -> String {
    json!({
        "validatorSchemaVersion": sceneworks_options::request_validation::VALIDATOR_SCHEMA_VERSION,
        "result": match result {
            Ok(value) => value,
            Err(refusal) => json!({ "ok": false, "refusal": refusal }),
        },
    })
    .to_string()
}

/// The wire result is identical on native and WASM. A tagged refusal stays data instead of
/// escaping as a JavaScript exception, so forms can render the same code/field/context.
pub fn validate_request_json(kind: &str, payload_json: &str, entry_json: &str) -> String {
    let result = (|| {
        let payload = upgraded_request(kind, payload_json)?;
        let entry = decode_object("entry", entry_json)?;
        let payload = payload
            .as_object()
            .expect("request upgrade preserves object");
        match sceneworks_options::request_validation::resolve(kind, payload, &entry) {
            Ok(resolved) => Ok(json!({ "ok": true, "resolved": resolved })),
            Err(refusal) => Err(serde_json::to_value(refusal).expect("serializable refusal")),
        }
    })();
    response(result)
}

#[wasm_bindgen]
pub fn validate_request(kind: &str, payload_json: &str, entry_json: &str) -> String {
    validate_request_json(kind, payload_json, entry_json)
}

/// When catalog facts are unavailable, only validate authored shape and version. A successful
/// shape answer never claims that server-owned model, device or installation checks passed.
pub fn validate_shape_json(kind: &str, payload_json: &str) -> String {
    response((|| {
        let payload = upgraded_request(kind, payload_json)?;
        let payload = payload
            .as_object()
            .expect("request upgrade preserves object");
        match sceneworks_options::request_validation::validate_shape(kind, payload) {
            Ok(()) => Ok(json!({ "ok": true, "shapeOnly": true })),
            Err(refusal) => Err(serde_json::to_value(refusal).expect("serializable refusal")),
        }
    })())
}

#[wasm_bindgen]
pub fn validate_shape(kind: &str, payload_json: &str) -> String {
    validate_shape_json(kind, payload_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_rejects_non_object_without_entering_rules() {
        let response: Value = serde_json::from_str(&validate_request_json("image", "[]", "{}"))
            .expect("JSON response");
        assert_eq!(response["validatorSchemaVersion"], 1);
        assert_eq!(response["result"]["ok"], false);
        assert_eq!(response["result"]["refusal"]["field"], "payload");
    }
}
