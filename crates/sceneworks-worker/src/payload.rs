//! Typed accessors for job-payload JSON fields.
use super::*;

pub(crate) use sceneworks_core::payload_util::json_size_to_u64;

/// Parse a JSON integer from either a number or a trimmed numeric string.
/// Booleans are deliberately rejected even though some dynamic languages treat
/// them as integers.
#[cfg(native_lane)]
pub(crate) use sceneworks_core::payload_util::json_integer as quant_int;

/// Common bits-to-tier ladder. The caller owns the omitted/invalid-value default.
#[cfg(any(test, native_lane))]
pub(crate) fn tier_for_bits(bits: Option<i64>, default: &str) -> &str {
    match bits {
        Some(bits) if bits <= 0 => "bf16",
        Some(bits) if bits <= 4 => "q4",
        Some(_) => "q8",
        None => default,
    }
}

/// Parse a JSON float from either a number or a trimmed numeric string.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(crate) fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// Required normalized text with a caller-owned diagnostic (payload and dataset-item contracts differ).
#[cfg(any(test, native_lane))]
pub(crate) fn required_trimmed_str<'a>(
    object: &'a JsonObject,
    field: &str,
    message: &str,
) -> WorkerResult<&'a str> {
    trimmed_string(object.get(field)).ok_or_else(|| WorkerError::InvalidPayload(message.to_owned()))
}

pub(crate) fn required_payload_string<'a>(
    payload: &'a JsonObject,
    field: &str,
) -> WorkerResult<&'a str> {
    optional_payload_string(payload, field)
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing payload.{field}")))
}

/// Required user text or an exact external key (for example a Parquet column name).
pub(crate) fn required_payload_text<'a>(
    payload: &'a JsonObject,
    field: &str,
) -> WorkerResult<&'a str> {
    optional_payload_text(payload, field)
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing payload.{field}")))
}

/// Optional identifier/path/selector text, normalized once at the boundary.
pub(crate) fn optional_payload_string<'a>(payload: &'a JsonObject, field: &str) -> Option<&'a str> {
    trimmed_string(payload.get(field))
}

/// Shared with JSON Value callers (including nested objects).
pub(crate) fn trimmed_string(value: Option<&Value>) -> Option<&str> {
    value.and_then(trimmed_value_str)
}

pub(crate) fn trimmed_value_str(value: &Value) -> Option<&str> {
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// User-authored text whose surrounding whitespace is meaningful (names, prompts, output).
pub(crate) fn optional_payload_text<'a>(payload: &'a JsonObject, field: &str) -> Option<&'a str> {
    payload.get(field).and_then(nonblank_value_text)
}

/// Nonblank text with its original whitespace, shared by stored JSON and payload callers.
fn nonblank_value_text(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.trim().is_empty())
}

pub(crate) fn payload_bool(payload: &JsonObject, field: &str) -> bool {
    payload.get(field).and_then(Value::as_bool).unwrap_or(false)
}

pub(crate) fn payload_string_array(payload: &JsonObject, field: &str) -> Vec<String> {
    payload
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn required_value_str<'a>(value: &'a Value, key: &str) -> WorkerResult<&'a str> {
    value
        .get(key)
        .and_then(nonblank_value_text)
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing {key}")))
}

pub(crate) fn payload_u32(payload: &JsonObject, field: &str, default: u32) -> u32 {
    payload
        .get(field)
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(default)
}

pub(crate) fn payload_f64(payload: &JsonObject, field: &str, default: f64) -> f64 {
    payload
        .get(field)
        .map_or(default, |value| value_f64(value, default))
}

pub(crate) fn item_f64(item: &Value, field: &str, default: f64) -> f64 {
    item.get(field)
        .map_or(default, |value| value_f64(value, default))
}

pub(crate) fn value_f64(value: &Value, default: f64) -> f64 {
    // Preserve the established generic payload contract: unlike the manifest/fit
    // parser above, whitespace-padded strings are invalid and fall back.
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .filter(|value: &f64| value.is_finite())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalized_strings_and_raw_text_are_explicit() {
        let payload =
            json!({"id":"  asset-1  ","prompt":"  hello\nworld  ","blank":" ","number":4})
                .as_object()
                .unwrap()
                .clone();
        assert_eq!(optional_payload_string(&payload, "id"), Some("asset-1"));
        assert_eq!(trimmed_string(Some(&json!("  asset-1  "))), Some("asset-1"));
        assert_eq!(required_payload_string(&payload, "id").unwrap(), "asset-1");
        assert_eq!(
            required_payload_text(&payload, "prompt").unwrap(),
            "  hello\nworld  "
        );
        assert_eq!(
            optional_payload_text(&payload, "prompt"),
            Some("  hello\nworld  ")
        );
        for key in ["blank", "number", "missing"] {
            assert_eq!(optional_payload_string(&payload, key), None);
            assert!(required_payload_string(&payload, key).is_err());
        }
    }

    #[test]
    fn tier_mapping_keeps_caller_defaults_and_integer_boundaries() {
        for default in ["q4", "q8", "bf16"] {
            assert_eq!(tier_for_bits(None, default), default);
        }
        for (bits, tier) in [
            (i64::MIN, "bf16"),
            (0, "bf16"),
            (1, "q4"),
            (4, "q4"),
            (5, "q8"),
            (8, "q8"),
            (16, "q8"),
            (i64::MAX, "q8"),
        ] {
            assert_eq!(tier_for_bits(Some(bits), "q4"), tier);
        }
    }

    #[test]
    fn required_trimmed_text_rejects_missing_empty_and_non_string_values() {
        for value in [json!(null), json!(false), json!("  ")] {
            let object = JsonObject::from_iter([("key".to_owned(), value)]);
            assert!(required_trimmed_str(&object, "key", "missing key").is_err());
        }
        let object = JsonObject::from_iter([("key".to_owned(), json!(" value "))]);
        assert_eq!(
            required_trimmed_str(&object, "key", "missing key").expect("text"),
            "value"
        );
        assert!(required_trimmed_str(&object, "absent", "missing key").is_err());
    }

    #[cfg(native_lane)]
    #[test]
    fn shared_integer_parser_accepts_numbers_and_trimmed_strings() {
        assert_eq!(quant_int(&json!(8)), Some(8));
        assert_eq!(quant_int(&json!(" 4 ")), Some(4));
        assert_eq!(quant_int(&json!(true)), None);
        assert_eq!(quant_int(&json!(4.5)), None);
    }

    #[test]
    fn shared_float_parser_accepts_numbers_and_trimmed_strings() {
        assert_eq!(json_f64(&json!(8.5)), Some(8.5));
        assert_eq!(json_f64(&json!(" 4.25 ")), Some(4.25));
        assert_eq!(json_f64(&json!(false)), None);
        assert_eq!(json_f64(&json!("nope")), None);
    }

    #[test]
    fn generic_payload_float_keeps_whitespace_strings_invalid() {
        assert_eq!(value_f64(&json!(" 4.25 "), 9.0), 9.0);
        assert_eq!(value_f64(&json!("4.25"), 9.0), 4.25);
    }
}
