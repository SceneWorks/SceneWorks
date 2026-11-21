//! Compatibility export of the portable policy owner.
pub use sceneworks_options::image_sampling_controls::*;

use crate::contracts::JsonObject;
use serde_json::Value;

pub const H3_IMAGE_SAMPLER_KEY: &str = "h3ImageSampler";
pub const H3_IMAGE_DEFAULT_S_NOISE: f32 = 1.1;
pub const H3_IMAGE_DEFAULT_S_NOISE_END: f32 = 0.9;
pub const H3_IMAGE_DEFAULT_NOISE_CLIP_STD: f32 = 0.0;

pub fn supports_h3_image_sampler(entry: &JsonObject) -> bool {
    entry
        .get("image")
        .and_then(Value::as_object)
        .and_then(|image| image.get("supportsH3ImageSampler"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// `advanced.h3ImageSampler`, typed (SW-7106). Absent or `null` ⇒ `Ok(None)` (the engine runs the
/// sampler `advanced.sampler` names; the worker then applies the LCM default). Otherwise an object
/// `{"sampler": "euler"}`, `{"sampler": "euler_ancestral"}` (SW-7224) or
/// `{"sampler": "lcm", "sNoise"?, "sNoiseEnd"?, "noiseClipStd"?}`: an
/// omitted LCM parameter takes the engine's H3-Image default (`sNoise` 1.1, `sNoiseEnd` 0.9,
/// `noiseClipStd` 0 = off); a `sNoise` given without `sNoiseEnd` is held constant, and the values must satisfy the engine's own rules (finite,
/// `sNoise > 0`, `sNoiseEnd >= 0`, `noiseClipStd >= 0`). An unknown field, an LCM parameter on
/// Euler, or an `advanced.sampler` name that disagrees is a typed error naming the field.
pub fn requested_h3_image_sampler(
    advanced: &JsonObject,
) -> Result<Option<gen_core::h3_image::H3ImageSampler>, String> {
    use gen_core::h3_image::{
        H3ImageSampler, LCM_NOISE_CLIP_STD_DEFAULT, LCM_S_NOISE_DEFAULT, LCM_S_NOISE_END_DEFAULT,
    };
    const PARAMS: [&str; 3] = ["sNoise", "sNoiseEnd", "noiseClipStd"];
    let key = H3_IMAGE_SAMPLER_KEY;
    let Some(value) = sceneworks_options::image_sampling_controls::h3_sampler_block(advanced)
    else {
        return Ok(None);
    };
    let block = value
        .as_object()
        .ok_or_else(|| format!("advanced.{key} must be an object, got {value}"))?;
    if let Some(unknown) = block
        .keys()
        .find(|field| *field != "sampler" && !PARAMS.contains(&field.as_str()))
    {
        return Err(format!(
            "advanced.{key}.{unknown} is not a sampler field (use sampler, {})",
            PARAMS.join(", ")
        ));
    }
    let param = |field: &str| -> Result<Option<f32>, String> {
        match block.get(field).filter(|value| !value.is_null()) {
            None => Ok(None),
            Some(value) => value
                .as_f64()
                .map(|number| Some(number as f32))
                .ok_or_else(|| format!("advanced.{key}.{field} must be a number, got {value}")),
        }
    };
    let name = block
        .get("sampler")
        .and_then(Value::as_str)
        .map(str::trim)
        .ok_or_else(|| {
            format!("advanced.{key}.sampler must be \"euler\", \"euler_ancestral\" or \"lcm\"")
        })?;
    let sampler = match name {
        "euler" | "euler_ancestral" => {
            let label = if name == "euler" {
                "Euler"
            } else {
                "Euler ancestral"
            };
            if let Some(field) = PARAMS.iter().find(|field| block.contains_key(**field)) {
                return Err(format!(
                    "advanced.{key}.{field} applies only to the LCM sampler, not {label}"
                ));
            }
            if name == "euler" {
                H3ImageSampler::Euler
            } else {
                H3ImageSampler::EulerAncestral
            }
        }
        "lcm" => {
            // With neither scale given, run the engine's 1.1 -> 0.9 ramp. A start scale given
            // alone keeps ComfyUI's constant `s_noise` (the end follows the start).
            let given_start = param("sNoise")?;
            let s_noise = given_start.unwrap_or(LCM_S_NOISE_DEFAULT);
            H3ImageSampler::Lcm {
                s_noise,
                s_noise_end: param("sNoiseEnd")?.unwrap_or(if given_start.is_some() {
                    s_noise
                } else {
                    LCM_S_NOISE_END_DEFAULT
                }),
                noise_clip_std: param("noiseClipStd")?.unwrap_or(LCM_NOISE_CLIP_STD_DEFAULT),
            }
        }
        other => {
            return Err(format!(
                "advanced.{key}.sampler \"{other}\" is not a MiniMax-H3-Image sampler (use euler, \
                 euler_ancestral or lcm)"
            ))
        }
    };
    // The engine's own rules, reported under the wire field names.
    sampler.validate().map_err(|refusal| {
        use gen_core::h3_image::H3ImageRefusal as Refusal;
        let (field, value, rule) = match refusal {
            Refusal::SamplerNonFinite { field, value } => (
                match field {
                    "s_noise" => "sNoise",
                    "s_noise_end" => "sNoiseEnd",
                    _ => "noiseClipStd",
                },
                value,
                "must be finite",
            ),
            Refusal::SamplerNoiseScale { value } => ("sNoise", value, "must be greater than 0"),
            Refusal::SamplerNoiseScaleEnd { value } => ("sNoiseEnd", value, "must be 0 or greater"),
            Refusal::SamplerNoiseClip { value } => {
                ("noiseClipStd", value, "must be 0 (off) or greater")
            }
            other => return format!("advanced.{key}: {other}"),
        };
        format!("advanced.{key}.{field} {value} {rule}")
    })?;
    if let Some(named) =
        sceneworks_options::image_sampling_controls::authored_sampler_name(advanced)
            .map(str::trim)
            .filter(|named| !named.is_empty() && *named != "default" && *named != sampler.name())
    {
        return Err(format!(
            "advanced.sampler \"{named}\" disagrees with advanced.{key}.sampler \"{}\"",
            sampler.name()
        ));
    }
    Ok(Some(sampler))
}

/// Keep the H3 sampler on the core boundary while the other image controls use the
/// portable owner. The sampler's engine type is part of the core runtime contract.
pub fn image_sampling_control_error(
    model: &str,
    advanced: &JsonObject,
    negative_prompt: &str,
    entry: &JsonObject,
) -> Option<String> {
    if let Some(refusal) = sceneworks_options::image_sampling_controls::image_sampling_control_error(
        model,
        advanced,
        negative_prompt,
        entry,
    ) {
        return Some(refusal);
    }
    match requested_h3_image_sampler(advanced) {
        Err(message) => Some(format!("{model}: {message}")),
        Ok(Some(_)) if !supports_h3_image_sampler(entry) => Some(format!(
            "{model} does not offer the MiniMax-H3-Image sampler (advanced.{H3_IMAGE_SAMPLER_KEY}); drop it"
        )),
        _ => None,
    }
}

#[cfg(test)]
mod h3_tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn h3_sampler_remains_declared_and_validated_after_portable_merge() {
        let entry = object(json!({"image":{"supportsH3ImageSampler":true}}));
        let lcm = object(json!({"h3ImageSampler":{"sampler":"lcm","sNoise":1.2}}));
        assert!(supports_h3_image_sampler(&entry));
        assert!(image_sampling_control_error("minimax_h3_image", &lcm, "", &entry).is_none());
        assert!(requested_h3_image_sampler(&lcm).unwrap().is_some());

        // SW-7224: euler_ancestral is a block sampler with no LCM parameters.
        let ancestral = object(json!({"h3ImageSampler":{"sampler":"euler_ancestral"}}));
        assert_eq!(
            requested_h3_image_sampler(&ancestral).unwrap(),
            Some(gen_core::h3_image::H3ImageSampler::EulerAncestral)
        );
        let ancestral_noise =
            object(json!({"h3ImageSampler":{"sampler":"euler_ancestral","sNoise":1.2}}));
        assert!(requested_h3_image_sampler(&ancestral_noise)
            .unwrap_err()
            .contains("not Euler ancestral"));
        let disagree = object(json!({
            "sampler": "euler",
            "h3ImageSampler": {"sampler": "euler_ancestral"}
        }));
        assert!(requested_h3_image_sampler(&disagree).is_err());

        let invalid = object(json!({"h3ImageSampler":{"sampler":"euler","sNoise":1.2}}));
        assert!(
            image_sampling_control_error("minimax_h3_image", &invalid, "", &entry)
                .unwrap()
                .contains("applies only to the LCM sampler")
        );
        assert!(
            image_sampling_control_error("plain", &lcm, "", &object(json!({})))
                .unwrap()
                .contains("does not offer")
        );
    }
}
