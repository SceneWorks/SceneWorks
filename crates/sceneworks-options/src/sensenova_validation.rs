//! Pure SenseNova understanding and interleave rollout controls.
//!
//! These values mirror the worker's historical `safe_int` and `_advanced_float` reads. The
//! caller passes authored input separately from its effective clone so DTO defaults cannot erase
//! omission intent (notably interleave's 2048×1152 worker default).

use crate::contracts::JsonObject;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct VqaControls {
    pub max_new_tokens: i64,
    pub max_image_pixels: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterleaveControls {
    pub width: i64,
    pub height: i64,
    pub steps: i64,
    /// Preserve the JSON f64 until the native worker performs its historical f32 cast.
    pub guidance_scale: f64,
    pub image_guidance_scale: f64,
    pub timestep_shift: f64,
    pub max_new_tokens: i64,
    pub max_images: i64,
    pub think_mode: bool,
    /// `None` leaves the inference provider's system protocol message authoritative.
    pub system_message: Option<String>,
}

// Source-backed integer readers historically use their default for any value that cannot
// become an i64 (including decimal/exponent strings). Keep that choice explicit in the effective
// copy while preserving the complete authored carrier for history and provenance.
const VQA_INTEGERS: &[(&str, i64, i64, i64)] = &[
    ("maxNewTokens", 256, 16, 2048),
    ("maxImagePixels", 768 * 768, 256 * 256, 2048 * 2048),
];
const INTERLEAVE_INTEGERS: &[(&str, i64, i64, i64)] = &[
    ("width", 2048, 256, 4096),
    ("height", 1152, 256, 4096),
    ("maxImages", 6, 1, 10),
];
const INTERLEAVE_ADVANCED_INTEGERS: &[(&str, i64, i64, i64)] = &[
    ("numInferenceSteps", 50, 1, 100),
    ("maxNewTokens", 2048, 64, 8192),
    ("maxImages", 6, 1, 10),
];
fn legacy_integer(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .map(|number| number as i64)
        })
        .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
}
fn integer(object: &JsonObject, key: &str, specs: &[(&str, i64, i64, i64)]) -> i64 {
    let (_, default, low, high) = specs
        .iter()
        .find(|(name, ..)| *name == key)
        .expect("declared SenseNova integer");
    object
        .get(key)
        .and_then(legacy_integer)
        .unwrap_or(*default)
        .clamp(*low, *high)
}
/// Project historical integer fallback before generic shape checks without mutating authored data.
pub fn canonical_numbers(
    kind: &str,
    authored: &JsonObject,
) -> Result<JsonObject, crate::request_validation::ValidationRefusal> {
    let mut result = authored.clone();
    let specs = match kind {
        "vqa" => VQA_INTEGERS,
        "interleave" => INTERLEAVE_INTEGERS,
        _ => return Ok(result),
    };
    for (key, ..) in specs {
        if authored.contains_key(*key) {
            result.insert((*key).into(), json!(integer(authored, key, specs)));
        }
    }
    if kind == "interleave" {
        if let Some(advanced) = result.get_mut("advanced").and_then(Value::as_object_mut) {
            for (key, ..) in INTERLEAVE_ADVANCED_INTEGERS {
                if advanced.contains_key(*key) {
                    advanced.insert(
                        (*key).into(),
                        json!(integer(advanced, key, INTERLEAVE_ADVANCED_INTEGERS)),
                    );
                }
            }
            for key in ["guidanceScale", "imageGuidanceScale", "timestepShift"] {
                if let Some(value) = advanced.get(key) {
                    let number =
                        crate::effective_values::legacy_number(value).ok_or_else(|| {
                            crate::request_validation::ValidationRefusal::new(
                                format!("advanced.{key}"),
                                "must be a finite number or numeric string",
                            )
                        })?;
                    advanced.insert(key.into(), json!(number));
                }
            }
        }
    }
    Ok(result)
}

fn advanced_float(advanced: &JsonObject, key: &str, default: f64) -> f64 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(default)
}

pub fn resolve_vqa(authored: &JsonObject) -> VqaControls {
    VqaControls {
        max_new_tokens: integer(authored, "maxNewTokens", VQA_INTEGERS),
        max_image_pixels: integer(authored, "maxImagePixels", VQA_INTEGERS),
    }
}

pub fn resolve_interleave(authored: &JsonObject) -> InterleaveControls {
    let empty = JsonObject::new();
    let advanced = authored
        .get("advanced")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    InterleaveControls {
        width: integer(authored, "width", INTERLEAVE_INTEGERS),
        height: integer(authored, "height", INTERLEAVE_INTEGERS),
        steps: integer(advanced, "numInferenceSteps", INTERLEAVE_ADVANCED_INTEGERS),
        guidance_scale: advanced_float(advanced, "guidanceScale", 4.0),
        image_guidance_scale: advanced_float(advanced, "imageGuidanceScale", 1.0),
        // SenseNova reads this spelling directly. `schedulerShift` is a different generic image
        // option and must not override it when both are present.
        timestep_shift: advanced_float(advanced, "timestepShift", 3.0),
        max_new_tokens: integer(advanced, "maxNewTokens", INTERLEAVE_ADVANCED_INTEGERS),
        max_images: if advanced.contains_key("maxImages") {
            integer(advanced, "maxImages", INTERLEAVE_ADVANCED_INTEGERS)
        } else {
            integer(authored, "maxImages", INTERLEAVE_INTEGERS)
        },
        think_mode: advanced
            .get("thinkMode")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        system_message: advanced
            .get("systemMessage")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .map(str::to_owned),
    }
}

/// Write only the resolved SenseNova controls onto the already-validated execution clone.
/// The authored map remains available to persistence and provenance callers.
pub fn materialize_vqa(authored: &JsonObject, effective: &mut JsonObject) {
    let controls = resolve_vqa(authored);
    effective.insert("maxNewTokens".into(), json!(controls.max_new_tokens));
    effective.insert("maxImagePixels".into(), json!(controls.max_image_pixels));
}

pub fn materialize_interleave(authored: &JsonObject, effective: &mut JsonObject) {
    let controls = resolve_interleave(authored);
    effective.insert("width".into(), json!(controls.width));
    effective.insert("height".into(), json!(controls.height));
    effective.insert("maxImages".into(), json!(controls.max_images));
    let advanced = effective
        .entry("advanced")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("validated advanced object");
    for (key, value) in [
        ("numInferenceSteps", json!(controls.steps)),
        ("guidanceScale", json!(controls.guidance_scale)),
        ("imageGuidanceScale", json!(controls.image_guidance_scale)),
        ("timestepShift", json!(controls.timestep_shift)),
        ("maxNewTokens", json!(controls.max_new_tokens)),
        ("maxImages", json!(controls.max_images)),
        ("thinkMode", json!(controls.think_mode)),
    ] {
        advanced.insert(key.into(), value);
    }
    if let Some(message) = controls.system_message {
        advanced.insert("systemMessage".into(), Value::String(message));
    } else {
        advanced.remove("systemMessage");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn interleave_defaults_keep_worker_geometry_and_provider_message_absent() {
        let authored = object(json!({"projectId":"p","prompt":"document"}));
        let mut effective =
            object(json!({"width":1024,"height":1024,"advanced":{"schedulerShift":9}}));
        materialize_interleave(&authored, &mut effective);
        assert_eq!(effective["width"], 2048);
        assert_eq!(effective["height"], 1152);
        assert_eq!(effective["advanced"]["numInferenceSteps"], 50);
        assert_eq!(effective["advanced"]["guidanceScale"], 4.0);
        assert_eq!(effective["advanced"]["imageGuidanceScale"], 1.0);
        assert_eq!(effective["advanced"]["timestepShift"], 3.0);
        assert_eq!(effective["advanced"]["maxNewTokens"], 2048);
        assert_eq!(effective["advanced"]["maxImages"], 6);
        assert_eq!(effective["advanced"]["thinkMode"], false);
        assert!(effective["advanced"].get("systemMessage").is_none());
        assert!(authored.get("advanced").is_none());
    }

    #[test]
    fn interleave_uses_authored_numeric_compatibility_and_clamps() {
        let authored = object(json!({
            "width":"9999", "height":12.9,
            "advanced":{
                "numInferenceSteps":"0", "guidanceScale":"6.5",
                "imageGuidanceScale":2, "schedulerShift":99, "timestepShift":"1.5",
                "maxNewTokens":9000, "maxImages":0, "thinkMode":true,
                "systemMessage":"  custom protocol  "
            }
        }));
        let controls = resolve_interleave(&authored);
        assert_eq!(
            (controls.width, controls.height, controls.steps),
            (4096, 256, 1)
        );
        assert_eq!(
            (
                controls.guidance_scale,
                controls.image_guidance_scale,
                controls.timestep_shift
            ),
            (6.5, 2.0, 1.5)
        );
        assert_eq!((controls.max_new_tokens, controls.max_images), (8192, 1));
        assert!(controls.think_mode);
        assert_eq!(controls.system_message.as_deref(), Some("custom protocol"));
        let mut effective = authored.clone();
        materialize_interleave(&authored, &mut effective);
        assert_eq!(effective["advanced"]["timestepShift"], 1.5);
        assert_eq!(authored["advanced"]["timestepShift"], "1.5");
    }

    #[test]
    fn vqa_defaults_invalid_input_and_clamps_preserve_authored() {
        let authored = object(json!({"maxNewTokens":"4096","maxImagePixels":2.5}));
        let mut effective = authored.clone();
        materialize_vqa(&authored, &mut effective);
        assert_eq!(effective["maxNewTokens"], 2048);
        assert_eq!(effective["maxImagePixels"], 256 * 256);
        assert_eq!(authored["maxNewTokens"], "4096");
        let defaults = resolve_vqa(&JsonObject::new());
        assert_eq!(
            (defaults.max_new_tokens, defaults.max_image_pixels),
            (256, 768 * 768)
        );
        let invalid = object(json!({"maxNewTokens":"bad","maxImagePixels":"99999999"}));
        let controls = resolve_vqa(&invalid);
        assert_eq!(
            (controls.max_new_tokens, controls.max_image_pixels),
            (256, 2048 * 2048)
        );
    }

    #[test]
    fn blank_system_message_uses_provider_default() {
        let authored = object(json!({"advanced":{"systemMessage":"  ","thinkMode":"yes"}}));
        let mut effective = authored.clone();
        materialize_interleave(&authored, &mut effective);
        assert!(effective["advanced"].get("systemMessage").is_none());
        assert_eq!(effective["advanced"]["thinkMode"], false);
    }

    #[test]
    fn finite_json_float_reaches_native_f32_cast_without_premature_null() {
        let authored = object(json!({"advanced":{"guidanceScale":1e100}}));
        let mut effective = authored.clone();
        materialize_interleave(&authored, &mut effective);
        assert_eq!(effective["advanced"]["guidanceScale"], json!(1e100));
        assert!(resolve_interleave(&authored).guidance_scale.is_finite());
    }
}
