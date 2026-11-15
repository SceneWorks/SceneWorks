//! Static prompt-enhancement policy. Backend availability and edit-route asset checks stay native.
use crate::{contracts::JsonObject, request_validation::ValidationRefusal};
use serde_json::Value;

pub const FACT_KEY: &str = "promptEnhancement";
pub const MAX_TEMPERATURE: f64 = 2.0;

/// The pinned SDK's model ceilings, guarded against its exported policy by worker tests.
/// Keeping this tiny data projection here avoids importing native engines into WASM.
pub fn max_tokens(model: &str) -> u64 {
    match model {
        "flux3_klein" | "flux3_klein_base" => 3_584,
        _ => 2_048,
    }
}

pub fn fields(
    model: &str,
    advanced: &JsonObject,
) -> Result<(bool, Option<f32>, Option<u32>), ValidationRefusal> {
    let fail =
        |key: &str, detail: String| ValidationRefusal::new(format!("advanced.{key}"), detail);
    if advanced.contains_key(FACT_KEY) {
        return Err(fail(
            FACT_KEY,
            format!("advanced.{FACT_KEY} is worker-owned and cannot be supplied by a client"),
        ));
    }
    let enabled = match advanced.get("enhancePrompt") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => {
            return Err(fail(
                "enhancePrompt",
                "advanced.enhancePrompt must be a boolean".into(),
            ))
        }
    };
    let temperature = advanced
        .get("enhanceTemperature")
        .map(|value| {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    fail(
                        "enhanceTemperature",
                        "advanced.enhanceTemperature must be a finite number".into(),
                    )
                })?;
            if !(0.0..=MAX_TEMPERATURE).contains(&value) {
                return Err(fail(
                    "enhanceTemperature",
                    format!("advanced.enhanceTemperature must be between 0 and {MAX_TEMPERATURE}"),
                ));
            }
            if !enabled {
                return Err(fail(
                    "enhanceTemperature",
                    "advanced.enhanceTemperature requires advanced.enhancePrompt=true".into(),
                ));
            }
            Ok(value as f32)
        })
        .transpose()?;
    let limit = max_tokens(model);
    let tokens = advanced
        .get("enhanceMaxTokens")
        .map(|value| {
            let value = value.as_u64().ok_or_else(|| {
                fail(
                    "enhanceMaxTokens",
                    "advanced.enhanceMaxTokens must be an integer".into(),
                )
            })?;
            if !(1..=limit).contains(&value) {
                return Err(fail(
                    "enhanceMaxTokens",
                    format!("advanced.enhanceMaxTokens must be between 1 and {limit}"),
                ));
            }
            if !enabled {
                return Err(fail(
                    "enhanceMaxTokens",
                    "advanced.enhanceMaxTokens requires advanced.enhancePrompt=true".into(),
                ));
            }
            Ok(value as u32)
        })
        .transpose()?;
    Ok((enabled, temperature, tokens))
}

pub fn validate(model: &str, advanced: &JsonObject) -> Result<bool, ValidationRefusal> {
    let (enabled, _, _) = fields(model, advanced)?;
    if !enabled {
        return Ok(false);
    }
    if !matches!(model, "flux2_dev" | "flux3_klein" | "flux3_klein_base") {
        return Err(ValidationRefusal::new("advanced.enhancePrompt", "advanced.enhancePrompt is supported only by FLUX.2-dev and FLUX 3 Image Klein; FLUX.2-Klein and other models reject it"));
    }
    let strict_control = advanced
        .get("poses")
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
        || ["controlWeights", "controlImage", "controlMode"]
            .iter()
            .any(|key| advanced.contains_key(*key));
    if strict_control {
        return Err(ValidationRefusal::new(
            "advanced.enhancePrompt",
            "advanced.enhancePrompt cannot be combined with strict control",
        ));
    }
    Ok(true)
}
