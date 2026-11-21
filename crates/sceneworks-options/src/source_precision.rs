//! Shared authored-request policy for a catalog-projected fixed checkpoint source.

use crate::contracts::JsonObject;
use crate::request_validation::ValidationRefusal;
use serde_json::Value;

pub fn applies_to(kind: &str) -> bool {
    matches!(kind, "image" | "imageDetail" | "video" | "interleave")
}

pub(crate) fn fixed_precision(entry: &JsonObject) -> Option<(&str, &str, &str)> {
    let source = entry.get("sourcePrecision")?;
    Some((
        source.get("format")?.as_str()?,
        source.get("codecId")?.as_str()?,
        source.get("label")?.as_str()?,
    ))
}

/// Validate every carrier independently, so one matching selector cannot conceal a
/// conflicting selector which another consumer might otherwise honor.
pub fn validate(payload: &JsonObject, entry: &JsonObject) -> Result<(), ValidationRefusal> {
    let Some((format, codec_id, label)) = fixed_precision(entry) else {
        return Ok(());
    };
    let matches_format = |value: &Value| {
        value.is_null()
            || value.as_str().is_some_and(|name| {
                let name = name.trim();
                name.is_empty()
                    || name.eq_ignore_ascii_case(format)
                    || name.eq_ignore_ascii_case(codec_id)
                    || (codec_id == "convrot-int8-native"
                        && name.eq_ignore_ascii_case("convrot_int8"))
            })
    };
    let refuse = |field: &str| {
        let detail = format!(
            "This checkpoint uses fixed source precision ({label}). Remove the conflicting {field} override."
        );
        ValidationRefusal::new(field, detail)
    };
    if payload
        .get("variant")
        .is_some_and(|value| !matches_format(value))
    {
        return Err(refuse("variant"));
    }
    if let Some(advanced) = payload.get("advanced").and_then(Value::as_object) {
        for key in ["quantTier", "quant", "precision", "quantization"] {
            if advanced.get(key).is_some_and(|value| {
                let auto = key == "quantization"
                    && value
                        .as_str()
                        .is_some_and(|text| text.trim().eq_ignore_ascii_case("auto"));
                !(matches_format(value) || auto)
            }) {
                return Err(refuse(&format!("advanced.{key}")));
            }
        }
        if advanced
            .get("mlxQuantize")
            .is_some_and(|value| !value.is_null())
        {
            return Err(refuse("advanced.mlxQuantize"));
        }
        if advanced.get("convRot").is_some_and(|value| {
            !value.is_null()
                && value
                    .as_bool()
                    .is_none_or(|enabled| enabled && codec_id != "convrot-int8-native")
        }) {
            return Err(refuse("advanced.convRot"));
        }
    }
    Ok(())
}

/// A verified source replaces historical conversion decisions, including redundant matching
/// selectors. The loader detects the original codec from the bound source bytes.
pub fn clear_precision_selectors(payload: &mut JsonObject) {
    payload.remove("variant");
    if let Some(advanced) = payload.get_mut("advanced").and_then(Value::as_object_mut) {
        clear_advanced_precision_selectors(advanced);
    }
}

pub fn clear_advanced_precision_selectors(advanced: &mut JsonObject) {
    for key in [
        "quantTier",
        "quant",
        "mlxQuantize",
        "mlxQuantizeExplicit",
        "convRot",
        "precision",
        "quantization",
    ] {
        advanced.remove(key);
    }
}
