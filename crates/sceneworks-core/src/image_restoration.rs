//! Iris-3B restoration request contract (sc-25683, epic 25678).
//!
//! Iris restoration rides the existing image-upscale surfaces as the `engine: "iris"` upscaler —
//! the standalone `image_upscale` job (Image Editor + Library single/batch) and the Image Studio
//! inline `upscale` post-processing block — but it is sized differently from the other engines: by
//! an arbitrary positive `scale` (1× = restore only, default 4×), an input-sizing mode
//! (`budgeted`, the default, fits the input into the model's declared budget first; `original`
//! processes it at its own size) and an optional colour correction (default on).
//!
//! This module is the one parse of those three fields, shared by the API (submit-time validation
//! and the pre-submission plan preview) and the worker (the request it hands the engine), so the
//! two can never disagree about what a payload means. It carries no engine math: the planner
//! (`gen_core::iris::restoration::plan`) lives behind the worker crate.

use serde_json::{Map, Value};

/// The `engine` id the upscale surfaces use for Iris restoration.
pub const IRIS_RESTORE_ENGINE: &str = "iris";
/// The catalog entry (and registered transform id) restoration installs and runs from.
pub const IRIS_RESTORE_MODEL_ID: &str = "iris_3b_restore";
/// The payload keys only the Iris engine honours. Any other engine refuses them by name rather than
/// ignoring them (E3).
pub const IRIS_ONLY_KEYS: [&str; 4] = ["scale", "inputSizing", "colorFix", IRIS_MODEL_KEY];
/// The payload key naming a TRAINED restoration model (a Training Studio `iris_3b_restore` run's
/// registered export, sc-25688) to restore with instead of the installed release export. Absent (or
/// the release id itself) = the release export.
pub const IRIS_MODEL_KEY: &str = "irisModel";

/// Whether `id` is a well-formed catalog model id (the worker joins it as one directory component).
pub fn is_plain_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The trained restoration model a payload names (`irisModel`), validated; `None` for the release
/// export (absent, null, or the release id itself).
pub fn parse_iris_restore_model(object: &Map<String, Value>) -> Result<Option<String>, String> {
    match object.get(IRIS_MODEL_KEY) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if id == IRIS_RESTORE_MODEL_ID => Ok(None),
        Some(Value::String(id)) if is_plain_model_id(id) => Ok(Some(id.clone())),
        Some(_) => Err(format!(
            "{IRIS_MODEL_KEY} must be the id of a trained Iris restoration model"
        )),
    }
}

/// Is this (raw, un-normalized) engine string the Iris restoration engine?
pub fn is_iris_engine(engine: &str) -> bool {
    engine.trim().eq_ignore_ascii_case(IRIS_RESTORE_ENGINE)
}

/// How the input is sized before restoration runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RestoreInputSizing {
    /// Downscale (never upscale) the input into the model's declared budget first (the release
    /// default: short side ≤ 512, long side ≤ 1024).
    #[default]
    Budgeted,
    /// Process the input at its own size (compute and memory grow with the output area).
    Original,
}

impl RestoreInputSizing {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Budgeted => "budgeted",
            Self::Original => "original",
        }
    }
}

/// The parsed Iris restoration controls of one request. `scale: None` and `color_fix: None` mean
/// "the model's declared default" (4× and on) — resolved by the engine, never guessed here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IrisRestoreParams {
    pub scale: Option<f64>,
    pub input_sizing: RestoreInputSizing,
    pub color_fix: Option<bool>,
}

/// Parse the Iris controls from a payload object (a standalone `image_upscale` payload or an Image
/// Studio `upscale` block). Every malformed value is a typed refusal naming the field — a bad scale
/// is never coerced to a different one.
pub fn parse_iris_restore_params(object: &Map<String, Value>) -> Result<IrisRestoreParams, String> {
    let scale = match object.get("scale") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let scale = value
                .as_f64()
                .ok_or_else(|| "scale must be a number".to_owned())?;
            if !(scale.is_finite() && scale > 0.0) {
                return Err(format!(
                    "scale must be a positive finite number (got {scale})"
                ));
            }
            Some(scale)
        }
    };
    let input_sizing = match object.get("inputSizing") {
        None | Some(Value::Null) => RestoreInputSizing::Budgeted,
        Some(Value::String(raw)) => match raw.trim().to_ascii_lowercase().as_str() {
            "budgeted" => RestoreInputSizing::Budgeted,
            "original" => RestoreInputSizing::Original,
            other => {
                return Err(format!(
                    "inputSizing must be \"budgeted\" or \"original\" (got {other:?})"
                ))
            }
        },
        Some(_) => return Err("inputSizing must be \"budgeted\" or \"original\"".to_owned()),
    };
    parse_iris_restore_model(object)?;
    let color_fix = match object.get("colorFix") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(value)) => Some(*value),
        Some(_) => return Err("colorFix must be a boolean".to_owned()),
    };
    Ok(IrisRestoreParams {
        scale,
        input_sizing,
        color_fix,
    })
}

/// Refuse the Iris-only keys on a payload for a different engine: they are not controls of
/// Real-ESRGAN or SeedVR2, and silently ignoring them would hand back a result the caller did not
/// ask for (E3).
pub fn reject_iris_only_keys(object: &Map<String, Value>, engine: &str) -> Result<(), String> {
    if is_iris_engine(engine) {
        return Ok(());
    }
    match IRIS_ONLY_KEYS
        .iter()
        .find(|key| object.get(**key).is_some_and(|value| !value.is_null()))
    {
        Some(key) => Err(format!(
            "{key} is a control of the Iris restoration engine only; the {engine} upscaler does \
             not honour it"
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn defaults_are_the_models_own() {
        let params = parse_iris_restore_params(&obj(json!({}))).unwrap();
        assert_eq!(params, IrisRestoreParams::default());
        assert_eq!(
            params.scale, None,
            "an omitted scale is the model default, not a guess"
        );
        assert_eq!(params.input_sizing, RestoreInputSizing::Budgeted);
        assert_eq!(params.color_fix, None);
    }

    #[test]
    fn parses_every_control() {
        let params = parse_iris_restore_params(&obj(
            json!({ "scale": 1.3, "inputSizing": "original", "colorFix": false }),
        ))
        .unwrap();
        assert_eq!(params.scale, Some(1.3));
        assert_eq!(params.input_sizing, RestoreInputSizing::Original);
        assert_eq!(params.color_fix, Some(false));
        // 1× is a legal restore-only scale.
        let one = parse_iris_restore_params(&obj(json!({ "scale": 1 }))).unwrap();
        assert_eq!(one.scale, Some(1.0));
    }

    #[test]
    fn refuses_malformed_values_by_name() {
        for (payload, field) in [
            (json!({ "scale": 0 }), "scale"),
            (json!({ "scale": -2 }), "scale"),
            (json!({ "scale": "4" }), "scale"),
            (json!({ "inputSizing": "tiny" }), "inputSizing"),
            (json!({ "inputSizing": 1 }), "inputSizing"),
            (json!({ "colorFix": "yes" }), "colorFix"),
        ] {
            let error = parse_iris_restore_params(&obj(payload.clone())).unwrap_err();
            assert!(error.contains(field), "{payload}: {error}");
        }
    }

    #[test]
    fn other_engines_refuse_iris_only_keys() {
        let payload = obj(json!({ "scale": 2.0 }));
        assert!(reject_iris_only_keys(&payload, "iris").is_ok());
        let error = reject_iris_only_keys(&payload, "seedvr2").unwrap_err();
        assert!(
            error.contains("scale") && error.contains("seedvr2"),
            "{error}"
        );
        assert!(reject_iris_only_keys(&obj(json!({ "colorFix": true })), "real-esrgan").is_err());
        assert!(reject_iris_only_keys(&obj(json!({ "factor": 2 })), "real-esrgan").is_ok());
    }

    #[test]
    fn engine_aliases() {
        for engine in ["iris", "IRIS", " iris "] {
            assert!(is_iris_engine(engine), "{engine}");
        }
        assert!(!is_iris_engine("seedvr2"));
        // The generation model id is not an upscale engine.
        assert!(!is_iris_engine("iris_3b"));
        assert!(!is_iris_engine(""));
    }
}
