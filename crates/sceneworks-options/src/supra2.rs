//! Portable Supra2 product policy; native artifact checks remain in core.
use crate::contracts::JsonObject;
use gen_core_supra2_policy::{self as s2, Supra2Resolved};
use serde_json::Value;

/// Is `model` the Supra2-IMG catalog id?
pub fn is_supra2(model: &str) -> bool {
    model == s2::PROVIDER_ID
}

/// `advanced` knobs other lanes honor and the Supra2 engine has no input for. A meaningful value is
/// refused by name rather than dropped, so a recipe or headless call never renders something other
/// than what it asked for.
const UNSUPPORTED_ADVANCED: &[&str] = &[
    "schedulerShift",
    // The legacy alias the generic lane still reads into the schedule shift (`base.rs`).
    "timestepShift",
    "cfgInterval",
    "captionOverflow",
    "enhancePrompt",
    "mlxQuantize",
    "quantTier",
    "convRot",
    "usePid",
    "pidTarget",
    "strength",
    // Diagnostic context belongs to the img2img input, which Supra2 does not support.
    "imageReference",
    "ipAdapterScale",
    // The InstantID identity-structure slider's WIRE key (`imageJobAdvanced.js` /
    // `CharacterAdvancedOptions.jsx`); `identityStructure` / `controlnetScale` /
    // `variationStrength` are studio UI-state names that never reach a request.
    "controlnetConditioningScale",
    "trueCfgScale",
    "textStyleGain",
    "viewAngle",
    "viewAngles",
    "poses",
    "controlMode",
    "controlImage",
    "controlScale",
    "controlWeights",
    "phases",
    "transparentBackground",
    "textEncoderModel",
    // The Pose Library's face-restoration pass (sent with `poses`).
    "faceRestore",
    // The tier-pick marker that rides with `mlxQuantize`.
    "mlxQuantizeExplicit",
    // MOSS-Image's own wire objects (`imageJobAdvanced.js` / the MCP image tool).
    "mossImage",
    "mossImageEdit",
    "mossImageAdvanced",
    "h3ImageSampler",
    "promptBatch",
    "mossOrdinarySettings",
    // FLUX.3's authored layout controls (`imageJobAdvanced.js`): the shared studio builder emits
    // them only for the FLUX 3 models, but the same static key extractor that demands this
    // classification reads the builder's source, so Supra2 refuses them by wire name like every
    // other engine-specific object it has no input for.
    "flux3_layout",
    "flux3_reference_ids",
    // The character lane's Angle Set / Key Point Library extras (`characterPanels.jsx`).
    "angleSet",
    "keypointCollectionId",
    // A caller-chosen weights directory (sc-24217 pre-merge review): the API injects the converted
    // bundle as `modelManifestEntry.modelPath`, and the load, the failed-load diagnosis and the
    // asset's provenance all read that one bundle. No client sends this for Supra2; honoring it
    // would load a different bundle from the one the install's diagnosis and records describe.
    "modelPath",
];

/// Top-level carriers of a source, reference, mask, control, adapter or prompt batch.
const UNSUPPORTED_CARRIERS: &[&str] = &[
    "sourceAssetId",
    "referenceAssetId",
    "referenceAssetIds",
    "maskAssetId",
    "controls",
    "controlnets",
    "loras",
    "prompts",
    // A character identity (the studio sends these on `character_image`; the MCP image tool
    // forwards `characterId` with any mode). `fitMode` is NOT here: the API stamps its default on
    // every image job, and it only shapes an edit source, which text-to-image has none of.
    "characterId",
    "characterLookId",
];

/// A value that asks for something: not `null`, `""`, `[]`, `{}` or `false`.
fn meaningful(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(object) => !object.is_empty(),
        Value::Number(_) => true,
    }
}

/// A JSON number or numeric string (the worker's own `advanced` reading); `Err` for any other shape.
fn number(object: &JsonObject, key: &str) -> Result<Option<f64>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| format!("advanced.{key} must be a finite number, got {value}")),
    }
}

/// An optional menu choice that is only acceptable as the engine's own (or the `default` alias).
fn menu_choice(advanced: &JsonObject, key: &str, accepted: &[&str]) -> Result<(), String> {
    match advanced.get(key) {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(value))
            if value.trim().is_empty()
                || value == "default"
                || accepted.contains(&value.as_str()) =>
        {
            Ok(())
        }
        Some(value) => Err(format!(
            "advanced.{key} {value} is not offered (Supra2-IMG runs {} only)",
            accepted.join(" / ")
        )),
    }
}

/// The largest `advanced.steps` Supra2 accepts: the entry's declared `limits.hardMaxSteps` (the
/// shipped row declares the provider's own sanity cap, so the published interface is not narrowed),
/// else the generic lane's clamp. A larger request is refused, never clamped.
pub fn max_steps(entry: &JsonObject) -> u32 {
    crate::limits::hard_max_steps(entry)
        .unwrap_or_else(|| s2::DEFAULT_STEPS.max(crate::image_request::BASE_MAX_IMAGE_STEPS))
}

/// Validate the FINAL (post-preset, post-merge) image payload. `Ok(None)` for any other model;
/// `Ok(Some(resolved))` with the published defaults applied; `Err(message)` naming the first
/// unsupported input. Width, height and count fall back to `entry`'s declared defaults when the
/// payload does not name them yet (the API fills them later from the same keys).
pub fn validate_request(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<Supra2Resolved>, String> {
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
    if !is_supra2(model) {
        return Ok(None);
    }
    let named = |message: String| format!("{model}: {message}");
    match payload.get("mode") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) if mode.is_empty() || mode == "text_to_image" => {}
        Some(mode) => {
            return Err(named(format!(
                "mode {mode} is not offered; Supra2-IMG is text-to-image only"
            )))
        }
    }
    if let Some(key) = UNSUPPORTED_CARRIERS
        .iter()
        .find(|key| payload.get(**key).is_some_and(meaningful))
    {
        return Err(named(format!(
            "`{key}` is not supported; Supra2-IMG renders from one text prompt only"
        )));
    }
    let negative = payload
        .get("negativePrompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !negative.is_empty() {
        return Err(named(
            "a negative prompt is not supported (guidance steers against the checkpoint's own \
             unconditional embedding); clear it"
                .to_owned(),
        ));
    }
    let hires = payload
        .get("hiresFix")
        .and_then(Value::as_object)
        .and_then(|hires| hires.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if hires {
        return Err(named(
            "hires fix is not supported; Supra2-IMG renders at its native 256x256 only".to_owned(),
        ));
    }
    let empty = JsonObject::new();
    let advanced = match payload.get("advanced") {
        None | Some(Value::Null) => &empty,
        Some(Value::Object(advanced)) => advanced,
        Some(_) => return Err(named("advanced must be an object".to_owned())),
    };
    if let Some(key) = UNSUPPORTED_ADVANCED
        .iter()
        .find(|key| advanced.get(**key).is_some_and(meaningful))
    {
        return Err(named(format!(
            "advanced.{key} is not supported (Supra2-IMG takes prompt, seed, guidance, steps and \
             image count only)"
        )));
    }
    menu_choice(advanced, "sampler", &[s2::SAMPLER])
        .and_then(|()| menu_choice(advanced, "scheduler", &[s2::SCHEDULER]))
        .and_then(|()| menu_choice(advanced, "guidanceMethod", &["cfg"]))
        .and_then(|()| menu_choice(advanced, "decoder", &["native"]))
        .map_err(named)?;
    let steps = number(advanced, "steps").map_err(named)?;
    if let Some(steps) = steps {
        if steps.fract() != 0.0 || steps < 1.0 || steps > f64::from(max_steps(entry)) {
            return Err(named(format!(
                "advanced.steps {steps} is outside 1..={}",
                max_steps(entry)
            )));
        }
    }
    let guidance = number(advanced, "guidanceScale").map_err(named)?;
    let (default_width, default_height) =
        crate::image_request::default_resolution(entry).unwrap_or((s2::IMAGE_SIZE, s2::IMAGE_SIZE));
    let dimension = |key: &str, default: u32| match payload.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| named(format!("{key} must be a positive integer, got {value}"))),
    };
    let width = dimension("width", default_width)?;
    let height = dimension("height", default_height)?;
    let count = match payload.get("count") {
        None | Some(Value::Null) => crate::image_request::default_count(entry).unwrap_or(1),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|count| *count >= 1)
            .ok_or_else(|| named(format!("count must be a positive integer, got {value}")))?,
    };
    // The engine renders up to `MAX_IMAGES` (64) per call, but a SceneWorks job is capped at
    // `image_request::MAX_COUNT` (8): `ImageRequest::from_payload` CLAMPS above it, so a larger
    // count (e.g. through retry/duplicate, which skips the typed route's 1..=8 check) would
    // silently render fewer images than asked. Refuse it here instead.
    if count > crate::image_request::MAX_COUNT {
        return Err(named(format!(
            "count {count} exceeds the {} images a job renders (the engine allows up to {} per \
             call, SceneWorks caps a job at {})",
            crate::image_request::MAX_COUNT,
            s2::MAX_IMAGES,
            crate::image_request::MAX_COUNT
        )));
    }
    let request = s2::Supra2PolicyRequest {
        width,
        height,
        count,
        steps: steps.map(|steps| steps as u32),
        guidance: guidance.map(|guidance| guidance as f32),
        ..Default::default()
    };
    Supra2Resolved::resolve(&request)
        .map(Some)
        .map_err(|error| named(error.to_string()))
}

pub fn materialize(
    authored: &JsonObject,
    entry: &JsonObject,
    effective: &mut JsonObject,
) -> Result<(), String> {
    let Some(resolved) = validate_request(authored, entry)? else {
        return Ok(());
    };
    effective.insert("width".into(), Value::from(s2::IMAGE_SIZE));
    effective.insert("height".into(), Value::from(s2::IMAGE_SIZE));
    effective.insert("count".into(), Value::from(resolved.count));
    let advanced = effective
        .entry("advanced")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .expect("validated advanced");
    advanced.insert("steps".into(), Value::from(resolved.steps));
    advanced.insert("guidanceScale".into(), Value::from(resolved.guidance));
    advanced.insert("sampler".into(), Value::from(s2::SAMPLER));
    advanced.insert("scheduler".into(), Value::from(s2::SCHEDULER));
    Ok(())
}
