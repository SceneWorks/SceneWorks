//! Portable MOSS product policy. Native artifacts, device and source selection stay in core.
use crate::contracts::JsonObject;
use gen_core_moss_image_policy::{
    MossDmdPrecision, MossImageAcceleration, MossImageOptions, MossImageResolved, MossImageVariant,
    MossPolicyRequest,
};
use serde_json::{json, Value};
#[derive(Clone, Debug)]
pub struct MossRequest {
    pub prompt: String,
    pub prompt_batch: Vec<String>,
    pub policy: MossPolicyRequest,
}
impl std::ops::Deref for MossRequest {
    type Target = MossPolicyRequest;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
impl std::ops::DerefMut for MossRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.policy
    }
}
fn variant_for_model(model: &str) -> Option<MossImageVariant> {
    match model {
        "moss_image_base" => Some(MossImageVariant::Base),
        "moss_image_sft" => Some(MossImageVariant::Sft),
        _ => None,
    }
}
pub const OPTIONS_KEY: &str = "mossImage";
pub const EDIT_OPTIONS_KEY: &str = "mossImageEdit";
pub const EDIT_REPORT_KEY: &str = "mossImageEditExecutionReport";
pub const EDIT_PURPOSE: &str = "moss_image_edit";
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EditOptions {
    pub reference_max_edge: u32,
    pub use_kv_cache: bool,
}
impl Default for EditOptions {
    fn default() -> Self {
        Self {
            reference_max_edge: 512,
            use_kv_cache: true,
        }
    }
}
/// Shared discovery contract; runtime validation remains `edit_options`.
pub fn edit_options_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false,
        "properties": {
            "reference_max_edge": {"type": "integer", "minimum": 16, "maximum": 4096, "default": 512},
            "use_kv_cache": {"type": "boolean", "default": true}
        }
    })
}
pub fn edit_options(advanced: &JsonObject) -> Result<EditOptions, String> {
    let options: EditOptions = advanced.get(EDIT_OPTIONS_KEY).map_or_else(
        || Ok(EditOptions::default()),
        |v| serde_json::from_value(v.clone()).map_err(|e| format!("advanced.mossImageEdit: {e}")),
    )?;
    if !(16..=4096).contains(&options.reference_max_edge) {
        return Err("MOSS edit reference_max_edge must be in 16..=4096".into());
    }
    Ok(options)
}
/// The ordered plural carrier is authoritative. Aliases may only repeat its first element.
/// Never trim, deduplicate, or truncate semantic reference slots.
pub fn edit_reference_ids(payload: &JsonObject) -> Result<Vec<String>, String> {
    let ids = edit_reference_slots(payload)?;
    if !(1..=4).contains(&ids.len()) {
        return Err("MOSS edit requires one to four ordered reference slots".into());
    }
    Ok(ids)
}
/// Resolve carriers without applying an execution-cardinality policy. Sharing describes slots;
/// enqueue still applies the runtime's current bound through `edit_reference_ids`.
pub fn edit_reference_slots(payload: &JsonObject) -> Result<Vec<String>, String> {
    let mut ids = match payload.get("referenceAssetIds") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|v| {
                v.as_str()
                    .filter(|s| !s.is_empty() && s.trim() == *s)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        "referenceAssetIds must contain nonempty exact asset ids".to_owned()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("referenceAssetIds must be an array".into()),
    };
    for key in ["referenceAssetId", "sourceAssetId"] {
        if let Some(value) = payload.get(key).filter(|v| !v.is_null()) {
            let id = value
                .as_str()
                .filter(|s| !s.is_empty() && s.trim() == *s)
                .ok_or_else(|| format!("{key} must be a nonempty exact asset id"))?;
            if ids.is_empty() {
                ids.push(id.to_owned());
            } else if ids[0] != id {
                return Err(format!("{key} conflicts with the ordered MOSS references"));
            }
        }
    }
    Ok(ids)
}
pub fn is_edit(payload: &JsonObject) -> bool {
    payload.get("mode").and_then(Value::as_str) == Some("edit_image")
}
pub fn request_purpose(payload: &JsonObject, options: &MossImageOptions) -> Option<&'static str> {
    if is_edit(payload) {
        Some(EDIT_PURPOSE)
    } else {
        purpose(options)
    }
}
pub const REPORT_KEY: &str = "mossImageExecutionReport";
pub fn is_moss(model: &str) -> bool {
    variant_for_model(model).is_some()
}
/// Product editing availability for the actual enqueue lane, independent of ordinary/DMD support.
pub fn edit_backend_eligible(model: &str, backend: &str) -> bool {
    is_moss(model) && matches!(backend, "mlx" | "candle")
}

/// Discovery metadata only. Request validation always uses the native strict serde + resolver.
pub fn options_schema() -> Value {
    let modes = [
        MossImageAcceleration::Off,
        MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
    ]
    .into_iter()
    .map(|acceleration| {
        serde_json::to_value(MossImageOptions {
            acceleration,
            ..Default::default()
        })
        .expect("options serialize")["acceleration"]
            .clone()
    })
    .collect::<Vec<_>>();
    let layers = json!({"type":"array","items":{"type":"integer","minimum":0,"maximum":41}});
    json!({"type":"object","additionalProperties":false,"properties":{
        "acceleration":{"type":"string","enum":modes,"default":"off"},
        "max_sequence_length":{"type":"integer","minimum":1,"maximum":4096,"default":4096},
        "uncond_skip_layers":{"anyOf":[{"type":"null"},layers.clone()]},
        "training_shift":{"anyOf":[{"type":"null"},{"type":"number","exclusiveMinimum":0}]},
        "slg":{"anyOf":[{"type":"null"},{"type":"object","additionalProperties":false,"required":["layers","scale","start","stop"],"properties":{"layers":layers.clone(),"scale":{"type":"number","minimum":0},"start":{"type":"number","minimum":0,"maximum":1},"stop":{"type":"number","minimum":0,"maximum":1}}}]},
        "pag":{"anyOf":[{"type":"null"},{"type":"object","additionalProperties":false,"required":["scale"],"properties":{"scale":{"type":"number","minimum":0,"maximum":15},"layers":{"anyOf":[{"type":"null"},layers]},"start":{"anyOf":[{"type":"null"},{"type":"number","minimum":0,"maximum":1}]},"stop":{"anyOf":[{"type":"null"},{"type":"number","minimum":0,"maximum":1}]},"adaptive_scale":{"anyOf":[{"type":"null"},{"type":"number","minimum":0}]}}}]}
    }})
}
pub fn options(advanced: &JsonObject) -> Result<MossImageOptions, String> {
    match advanced.get(OPTIONS_KEY) {
        None => Ok(MossImageOptions::default()),
        Some(value) => {
            serde_json::from_value(value.clone()).map_err(|e| format!("advanced.mossImage: {e}"))
        }
    }
}
pub fn purpose(options: &MossImageOptions) -> Option<&'static str> {
    match options.acceleration {
        MossImageAcceleration::Off => None,
        MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => {
            Some("moss_dmd_unmerged_bf16")
        }
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => Some("moss_dmd_full_fp32"),
    }
}
fn number(advanced: &JsonObject, key: &str) -> Result<Option<f32>, String> {
    advanced
        .get(key)
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && (*n as f32).is_finite())
                .map(|n| n as f32)
                .ok_or_else(|| format!("advanced.{key} must be a finite number"))
        })
        .transpose()
}
fn integer(object: &JsonObject, key: &str, fallback: u32) -> Result<u32, String> {
    object.get(key).map_or(Ok(fallback), |v| {
        v.as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("{key} must be an unsigned integer"))
    })
}
fn text(advanced: &JsonObject, key: &str) -> Result<Option<String>, String> {
    advanced
        .get(key)
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("advanced.{key} must be a string"))
        })
        .transpose()
}
fn meaningful(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Bool(v) => *v,
        _ => true,
    }
}
/// Validate the FINAL post-preset / post-merge payload. Omitted acceleration is always OFF.
/// This also returns the exact native request fields used by worker planning; no copied defaults.
pub fn native_request(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<MossRequest>, String> {
    let advanced_inputs = crate::moss_advanced::options(payload)?;
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
    let advanced = payload
        .get("advanced")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let Some(variant) = variant_for_model(model) else {
        return if advanced.contains_key(OPTIONS_KEY) || advanced.contains_key(EDIT_OPTIONS_KEY) {
            Err("advanced.mossImage is only supported by MOSS-Image Base/SFT".into())
        } else {
            Ok(None)
        };
    };
    if payload.get("advanced").is_some_and(|v| !v.is_object()) {
        return Err("advanced must be an object".into());
    }
    let edit = is_edit(payload);
    if payload
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|s| s != "text_to_image" && s != "edit_image")
    {
        return Err("MOSS-Image supports text_to_image or explicit edit_image".into());
    }
    if edit {
        edit_reference_ids(payload)?;
        edit_options(&advanced)?;
    } else if advanced.contains_key(EDIT_OPTIONS_KEY) {
        return Err("advanced.mossImageEdit requires edit_image".into());
    }
    for key in [
        "sourceAssetId",
        "referenceAssetId",
        "referenceAssetIds",
        "maskAssetId",
        "characterId",
        "characterLookId",
    ] {
        if edit && ["sourceAssetId", "referenceAssetId", "referenceAssetIds"].contains(&key) {
            continue;
        }
        if payload.get(key).is_some_and(meaningful) {
            return Err(format!(
                "MOSS-Image does not support {key}; remove the conditioning/adapter"
            ));
        }
    }
    if advanced.contains_key("flashAttn") {
        return Err("MOSS-Image does not support advanced.flashAttn".into());
    }
    for key in [
        "poses",
        "controlImage",
        "controlMode",
        "controlWeights",
        "controlScale",
        "controlnetScale",
        "ipAdapterScale",
        "viewAngle",
        "strength",
        "trueCfgScale",
        "cfgInterval",
        "phases",
        "captionOverflow",
        "textEncoderModel",
        "transparentBackground",
        "usePid",
        "decoder",
        "mlxQuantize",
        "guidanceEta",
        "guidanceMomentum",
        "guidanceNormThreshold",
    ] {
        if advanced.get(key).is_some_and(meaningful) {
            return Err(format!("MOSS-Image does not support advanced.{key}"));
        }
    }
    if payload
        .get("hiresFix")
        .and_then(|v| v.get("enabled"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err("MOSS-Image does not support hiresFix".into());
    }
    let defaults = entry.get("defaults").and_then(Value::as_object);
    let (dw, dh) = if edit {
        (512, 512)
    } else {
        crate::image_request::default_resolution(entry).unwrap_or((1024, 1024))
    };
    let opts = options(&advanced)?;
    let pag = if let Some(raw) = advanced.get(crate::pag::PAG_KEY).filter(|v| !v.is_null()) {
        let cap = crate::pag::capability_from_entry(entry)
            .map_err(|e| e.detail)?
            .unwrap_or_else(gen_core_moss_image_policy::moss_image_pag_capability);
        let (cfg, _) =
            crate::pag::validate_pag_value_with_capability(raw, &cap).map_err(|e| e.detail)?;
        Some(cfg)
    } else {
        None
    };
    let mut request = MossRequest {
        prompt: payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        prompt_batch: Vec::new(),
        policy: MossPolicyRequest {
            width: integer(payload, "width", dw)?,
            height: integer(payload, "height", dh)?,
            count: integer(
                payload,
                "count",
                defaults
                    .and_then(|v| v.get("count"))
                    .and_then(Value::as_u64)
                    .unwrap_or(1) as u32,
            )?,
            moss_image: Some(opts),
            pag,
            negative_prompt: payload
                .get("negativePrompt")
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or("negativePrompt must be a string")
                })
                .transpose()?,
            guidance: number(&advanced, "guidanceScale")?,
            sampler: text(&advanced, "sampler")?,
            scheduler: text(&advanced, "scheduler")?,
            scheduler_shift: number(&advanced, "schedulerShift")?,
            guidance_method: text(&advanced, "guidanceMethod")?,
            ..Default::default()
        },
    };
    if advanced.contains_key("steps") {
        request.steps = Some(integer(&advanced, "steps", 50)?);
    }
    if let Some(prompts) = payload.get("prompts") {
        request.prompt_batch = serde_json::from_value::<Vec<String>>(prompts.clone())
            .map_err(|_| "prompts must be a string array")?;
        if !request.prompt_batch.is_empty() {
            if request.prompt != request.prompt_batch[0] {
                return Err("prompt must equal prompts[0]".into());
            }
            request.prompt.clear();
        }
    }
    if advanced_inputs
        .as_ref()
        .is_some_and(|a| a.negative_prompts.is_some())
    {
        request.negative_prompt = None;
    }
    let expanded = request
        .count
        .checked_mul(
            request
                .prompt_batch
                .len()
                .max(crate::moss_advanced::batch_hint(payload))
                .max(1) as u32,
        )
        .ok_or("image count overflow")?;
    if expanded == 0 || expanded > 8 {
        return Err("MOSS-Image prompts × count must be in 1..=8".into());
    }
    let envelope = crate::image_limits::image_dimension_envelope(entry);
    for (axis, value) in [("width", request.width), ("height", request.height)] {
        if value > 4096
            || envelope.min.is_some_and(|min| value < min)
            || envelope.max.is_some_and(|max| value > max)
        {
            return Err(format!(
                "MOSS-Image {axis} is outside the declared product size envelope"
            ));
        }
    }
    if edit {
        request.policy = gen_core_moss_image_policy::apply_edit_defaults(&request.policy)
            .map_err(|e| e.to_string())?;
    }
    let resolved =
        MossImageResolved::resolve(variant, &request.policy).map_err(|e| e.to_string())?;
    if edit {
        request
            .moss_image
            .as_mut()
            .expect("MOSS options")
            .training_shift = Some(resolved.training_shift);
    }
    Ok(Some(request))
}

/// Materialize only controls accepted by this exact MOSS route; DMD forbids ordinary-only fields.
pub fn materialize(
    authored: &JsonObject,
    entry: &JsonObject,
    effective: &mut JsonObject,
) -> Result<(), String> {
    crate::moss_advanced::normalize_carrier(effective)?;
    let Some(request) = native_request(authored, entry)? else {
        return Ok(());
    };
    let variant = variant_for_model(
        authored
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
    .expect("MOSS request");
    let resolved =
        MossImageResolved::resolve(variant, &request.policy).map_err(|error| error.to_string())?;
    for (key, value) in [
        ("width", request.width),
        ("height", request.height),
        ("count", request.count),
    ] {
        effective.insert(key.into(), json!(value));
    }
    let advanced = effective
        .entry("advanced")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("validated advanced");
    advanced.insert("steps".into(), json!(resolved.steps));
    advanced.insert("guidanceScale".into(), json!(resolved.guidance));
    let mut options = request.moss_image.clone().expect("MOSS options");
    if resolved.acceleration == MossImageAcceleration::Off {
        options.training_shift = Some(resolved.training_shift);
        options.uncond_skip_layers = Some(resolved.uncond_skip_layers);
        advanced.insert("schedulerShift".into(), json!(resolved.shift));
    }
    advanced.insert(
        OPTIONS_KEY.into(),
        serde_json::to_value(options).expect("MOSS options"),
    );
    if is_edit(authored) {
        advanced.insert(
            EDIT_OPTIONS_KEY.into(),
            serde_json::to_value(edit_options(advanced)?).expect("MOSS edit options"),
        );
    }
    Ok(())
}
