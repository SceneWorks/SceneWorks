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

/// Match the worker's numeric input order: integer, float cast to integer, then trimmed integer
/// string. Invalid input uses the caller's default before clamping.
fn safe_int(object: &JsonObject, key: &str, default: i64, low: i64, high: i64) -> i64 {
    object
        .get(key)
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_f64().map(|float| float as i64))
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(default)
        .clamp(low, high)
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
        max_new_tokens: safe_int(authored, "maxNewTokens", 256, 16, 2048),
        max_image_pixels: safe_int(
            authored,
            "maxImagePixels",
            768 * 768,
            256 * 256,
            2048 * 2048,
        ),
    }
}

pub fn resolve_interleave(authored: &JsonObject) -> InterleaveControls {
    let empty = JsonObject::new();
    let advanced = authored
        .get("advanced")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    InterleaveControls {
        width: safe_int(authored, "width", 2048, 256, 4096),
        height: safe_int(authored, "height", 1152, 256, 4096),
        steps: safe_int(advanced, "numInferenceSteps", 50, 1, 100),
        guidance_scale: advanced_float(advanced, "guidanceScale", 4.0),
        image_guidance_scale: advanced_float(advanced, "imageGuidanceScale", 1.0),
        // SenseNova reads this spelling directly. `schedulerShift` is a different generic image
        // option and must not override it when both are present.
        timestep_shift: advanced_float(advanced, "timestepShift", 3.0),
        max_new_tokens: safe_int(advanced, "maxNewTokens", 2048, 64, 8192),
        max_images: safe_int(advanced, "maxImages", 6, 1, 10),
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
