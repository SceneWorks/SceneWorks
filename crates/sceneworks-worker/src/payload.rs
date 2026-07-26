//! Typed accessors for job-payload JSON fields.
use super::*;

pub(crate) fn json_size_to_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

/// Parse a JSON integer from either a number or a trimmed numeric string.
/// Booleans are deliberately rejected even though some dynamic languages treat
/// them as integers.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn quant_int(value: &Value) -> Option<i64> {
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// Parse a JSON float from either a number or a trimmed numeric string.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(crate) fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

pub(crate) fn required_payload_string<'a>(
    payload: &'a JsonObject,
    field: &str,
) -> WorkerResult<&'a str> {
    optional_payload_string(payload, field)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing payload.{field}")))
}

pub(crate) fn optional_payload_string<'a>(payload: &'a JsonObject, field: &str) -> Option<&'a str> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
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
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
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

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
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
