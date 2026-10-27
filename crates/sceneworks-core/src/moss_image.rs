//! One MOSS wire boundary for typed REST, generic jobs, replay and native worker shaping.
//! Native gen-core owns the options vocabulary and numerical compatibility rules.
use crate::contracts::JsonObject;
use gen_core::{
    moss_image::{
        MossDmdPrecision, MossImageAcceleration, MossImageOptions, MossImageResolved,
        MossImageVariant,
    },
    GenerationRequest,
};
use serde_json::{json, Value};

pub const OPTIONS_KEY: &str = "mossImage";
pub const EDIT_OPTIONS_KEY: &str = "mossImageEdit";
pub const EDIT_REPORT_KEY: &str = "mossImageEditExecutionReport";
pub const EDIT_PURPOSE: &str = "moss_image_edit";
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
    MossImageVariant::from_provider_id(model).is_some()
}
/// Product editing availability for the actual enqueue lane, independent of ordinary/DMD support.
pub fn edit_backend_eligible(model: &str, backend: &str) -> bool {
    is_moss(model) && matches!(backend, "mlx" | "candle")
}

/// Effective sampling controls persisted by every edit enqueue path, so history can replay
/// an omitted native default after the catalog or application defaults change.
pub fn edit_effective_controls(
    model: &str,
    request: &GenerationRequest,
) -> Result<JsonObject, String> {
    let variant = MossImageVariant::from_provider_id(model).ok_or("unknown MOSS model")?;
    let resolved =
        MossImageResolved::resolve(variant, request).map_err(|error| error.to_string())?;
    Ok(json!({"steps":resolved.steps,"guidanceScale":resolved.guidance,"schedulerShift":resolved.shift})
        .as_object().expect("controls object").clone())
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
        "slg":{"anyOf":[{"type":"null"},{"type":"object","additionalProperties":false,"required":["layers","scale","start","stop"],"properties":{"layers":layers,"scale":{"type":"number","minimum":0},"start":{"type":"number","minimum":0,"maximum":1},"stop":{"type":"number","minimum":0,"maximum":1}}}]}
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
) -> Result<Option<GenerationRequest>, String> {
    let advanced_inputs = crate::moss_image_advanced::options(payload)?;
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
    let advanced = payload
        .get("advanced")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let Some(variant) = MossImageVariant::from_provider_id(model) else {
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
        "loras",
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
    let mut request = GenerationRequest {
        prompt: payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
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
                .max(crate::moss_image_advanced::batch_hint(payload))
                .max(1) as u32,
        )
        .ok_or("image count overflow")?;
    if expanded == 0 || expanded > 8 {
        return Err("MOSS-Image prompts × count must be in 1..=8".into());
    }
    let envelope = crate::video_request::image_dimension_envelope(entry);
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
        request = gen_core::moss_image_edit::edit_request(&request).map_err(|e| e.to_string())?;
    }
    let resolved = MossImageResolved::resolve(variant, &request).map_err(|e| e.to_string())?;
    if edit {
        request
            .moss_image
            .as_mut()
            .expect("MOSS options")
            .training_shift = Some(resolved.training_shift);
    }
    Ok(Some(request))
}
/// Stage ONLY the selected optional package. Ordinary SFT remains usable without any DMD files.
pub fn selected_components(
    entry: &JsonObject,
    options: &MossImageOptions,
) -> Result<JsonObject, String> {
    selected_components_for_purpose(entry, purpose(options))
}
pub fn selected_components_for_purpose(
    entry: &JsonObject,
    purpose: Option<&str>,
) -> Result<JsonObject, String> {
    let mut result = entry.clone();
    let downloads = result
        .entry("downloads".to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or("downloads must be an array")?;
    downloads.retain(|v| {
        !v.get("componentId")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("moss-dmd-") || id == "moss-image-edit")
    });
    if let Some(purpose) = purpose {
        let mut components = crate::model_usage_policy::conditional_component_downloads(
            &Value::Object(entry.clone()),
            purpose,
        )
        .map_err(|e| e.to_string())?;
        for row in &mut components {
            if let Some(source) = entry
                .get("conditionalComponents")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|source| source.get("componentId") == row.get("componentId"))
            {
                for key in ["license", "licenseBasis"] {
                    if let Some(value) = source.get(key) {
                        row.as_object_mut()
                            .expect("component row")
                            .insert(key.to_owned(), value.clone());
                    }
                }
            }
        }
        downloads.extend(components);
    }
    Ok(result)
}
/// Reports are exclusively worker-produced. Replay keeps requested options but cannot reuse outcomes.
pub fn scrub_reports(payload: &mut JsonObject) {
    payload.remove("mossImageSampleIndex");
    payload.remove(EDIT_REPORT_KEY);
    payload.remove("moss_image_edit_execution_report");
    payload.remove(REPORT_KEY);
    payload.remove("moss_image_execution_report");
    if let Some(advanced) = payload.get_mut("advanced").and_then(Value::as_object_mut) {
        advanced.remove("mossImageSampleIndex");
        advanced.remove(EDIT_REPORT_KEY);
        advanced.remove("moss_image_edit_execution_report");
        advanced.remove(REPORT_KEY);
        advanced.remove("moss_image_execution_report");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry() -> JsonObject {
        json!({"defaults":{"resolution":"1024x1024","count":1},"limits":{"minDimension":16,"maxDimension":4096},"conditionalComponents":[{"provider":"huggingface","repo":"dmd","revision":"pin","componentId":"moss-dmd-lora","requiredFor":["moss_dmd_unmerged_bf16"],"files":["lora.safetensors"]},{"provider":"huggingface","repo":"dmd","revision":"pin","componentId":"moss-dmd-transformer","requiredFor":["moss_dmd_full_fp32"],"files":["transformer/*"]}]}).as_object().unwrap().clone()
    }
    fn body(advanced: Value) -> JsonObject {
        json!({"model":"moss_image_sft","mode":"text_to_image","prompt":"日本語 <tag> {\"items\":[]}","advanced":advanced}).as_object().unwrap().clone()
    }
    #[test]
    fn explicit_edit_preserves_single_reference_and_author_defaults() {
        for model in ["moss_image_base", "moss_image_sft"] {
            let p = json!({"model":model,"mode":"edit_image","prompt":"change the cup to blue","sourceAssetId":"ref-a","referenceAssetIds":["ref-a"],"advanced":{"mossImageEdit":{"reference_max_edge":768,"use_kv_cache":false}}}).as_object().unwrap().clone();
            assert_eq!(edit_reference_ids(&p).unwrap(), ["ref-a"]);
            let request = native_request(&p, &entry()).unwrap().unwrap();
            assert!(request.conditioning.is_empty());
            assert_eq!((request.width, request.height), (512, 512));
            let resolved = MossImageResolved::resolve(
                MossImageVariant::from_provider_id(model).unwrap(),
                &request,
            )
            .unwrap();
            assert_eq!(resolved.steps, 30);
            assert_eq!(resolved.guidance, 4.0);
            assert_eq!(resolved.shift, 5.0);
            assert!(resolved.uncond_skip_layers.is_empty());
            assert_eq!(
                request_purpose(&p, request.moss_image.as_ref().unwrap()),
                Some(EDIT_PURPOSE)
            );
            let options = edit_options(p["advanced"].as_object().unwrap()).unwrap();
            assert_eq!(options.reference_max_edge, 768);
            assert!(!options.use_kv_cache);
        }
    }
    #[test]
    fn edit_reference_slots_match_native_bounds_without_deduplicating_or_reordering() {
        use gen_core::{moss_image_edit::MossImageEditInputs, Image};
        for model in ["moss_image_base", "moss_image_sft"] {
            for ids in [
                vec!["a"],
                vec!["a", "b"],
                vec!["a", "b", "a"],
                vec!["c", "a", "b", "a"],
            ] {
                for ordered in [ids.clone(), ids.iter().rev().copied().collect()] {
                    let payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup",
                        "referenceAssetIds":ordered,"sourceAssetId":ordered[0],"referenceAssetId":ordered[0]})
                        .as_object().unwrap().clone();
                    assert_eq!(edit_reference_ids(&payload).unwrap(), ordered);
                    assert!(native_request(&payload, &entry()).unwrap().is_some());
                    let references = ordered
                        .iter()
                        .map(|id| {
                            let (width, height, pixel) = match *id {
                                "a" => (48, 16, 73),
                                "b" => (16, 48, 127),
                                _ => (32, 32, 211),
                            };
                            Image {
                                width,
                                height,
                                pixels: vec![pixel; width as usize * height as usize * 3],
                            }
                        })
                        .collect::<Vec<_>>();
                    let expected = references
                        .iter()
                        .map(|image| [image.width, image.height])
                        .collect::<Vec<_>>();
                    assert_eq!(
                        MossImageEditInputs {
                            references: &references,
                            ..Default::default()
                        }
                        .validate()
                        .unwrap(),
                        expected
                    );
                    for mode in ["text_to_image", "edit_image"] {
                        let mut unsupported = payload.clone();
                        unsupported.insert("mode".into(), json!(mode));
                        if mode == "edit_image" {
                            unsupported.insert(
                                "advanced".into(),
                                json!({"mossImage":{"acceleration":"unmerged_bf16"}}),
                            );
                        }
                        assert!(
                            native_request(&unsupported, &entry()).is_err(),
                            "ordinary/DMD cannot absorb edit references"
                        );
                    }
                }
            }
            for slots in [Vec::new(), vec!["a"; 5]] {
                let payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup","referenceAssetIds":slots})
                    .as_object().unwrap().clone();
                assert!(edit_reference_ids(&payload).is_err());
                assert!(native_request(&payload, &entry()).is_err());
                let references = slots
                    .iter()
                    .map(|_| Image {
                        width: 16,
                        height: 16,
                        pixels: vec![0; 16 * 16 * 3],
                    })
                    .collect::<Vec<_>>();
                assert!(MossImageEditInputs {
                    references: &references,
                    ..Default::default()
                }
                .validate()
                .is_err());
            }
            for changes in [
                json!({"referenceAssetIds":["a",null]}),
                json!({"referenceAssetIds":["a",2]}),
                json!({"referenceAssetIds":["a"," b"]}),
                json!({"referenceAssetIds":["a",""]}),
                json!({"referenceAssetIds":"a"}),
                json!({"sourceAssetId":"b"}),
                json!({"referenceAssetId":"b"}),
            ] {
                let mut payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup","referenceAssetIds":["a","b"]})
                    .as_object().unwrap().clone();
                payload.extend(changes.as_object().unwrap().clone());
                assert!(edit_reference_ids(&payload).is_err(), "{changes}");
                assert!(native_request(&payload, &entry()).is_err(), "{changes}");
            }
        }
    }
    #[test]
    fn edit_rejects_ambiguous_references_and_unimplemented_combinations() {
        let p = json!({"model":"moss_image_sft","mode":"edit_image","prompt":"blue cup","referenceAssetIds":["a"],"advanced":{}}).as_object().unwrap().clone();
        for (key, value) in [
            ("referenceAssetIds", json!([])),
            ("referenceAssetIds", json!(["a", "b", "a", "b", "a"])),
            ("referenceAssetIds", json!([1])),
            ("referenceAssetIds", json!([" a"])),
            ("sourceAssetId", json!("b")),
            ("maskAssetId", json!("mask")),
        ] {
            let mut malformed = p.clone();
            malformed.insert(key.into(), value);
            assert!(native_request(&malformed, &entry()).is_err(), "{key}");
        }
        for advanced in [
            json!({"strength":0.5}),
            json!({"mossImage":{"acceleration":"unmerged_bf16"}}),
            json!({"mossImage":{"uncond_skip_layers":[22]}}),
            json!({"mossImageEdit":{"reference_max_edge":0}}),
            json!({"mossImageEdit":{"use_kv_cache":"yes"}}),
            json!({"mossImageEdit":{"crop":true}}),
        ] {
            let mut malformed = p.clone();
            malformed.insert("advanced".into(), advanced);
            assert!(native_request(&malformed, &entry()).is_err());
        }
    }
    #[test]
    fn omitted_off_and_explicit_off_keep_author_defaults() {
        for advanced in [
            json!({}),
            json!({"mossImage":{"acceleration":"off"}}),
            json!({"steps":8}),
        ] {
            let req = native_request(&body(advanced), &entry()).unwrap().unwrap();
            let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &req).unwrap();
            assert_eq!(resolved.acceleration, MossImageAcceleration::Off);
            assert_eq!(resolved.guidance, 3.0);
            assert_eq!(resolved.uncond_skip_layers, vec![22]);
            assert_eq!(req.prompt, "日本語 <tag> {\"items\":[]}");
        }
    }
    #[test]
    fn both_dmd_precisions_and_explicit_empty_layers_are_lossless() {
        for mode in ["unmerged_bf16", "full_fp32"] {
            let req = native_request(&body(json!({"mossImage":{"acceleration":mode}})), &entry())
                .unwrap()
                .unwrap();
            assert_eq!(
                MossImageResolved::resolve(MossImageVariant::Sft, &req)
                    .unwrap()
                    .steps,
                8
            );
            let selected = selected_components(&entry(), req.moss_image.as_ref().unwrap()).unwrap();
            assert_eq!(selected["downloads"].as_array().unwrap().len(), 1);
        }
        let req = native_request(
            &body(json!({"mossImage":{"uncond_skip_layers":[]}})),
            &entry(),
        )
        .unwrap()
        .unwrap();
        assert!(MossImageResolved::resolve(MossImageVariant::Sft, &req)
            .unwrap()
            .uncond_skip_layers
            .is_empty());
        assert!(
            selected_components(&entry(), &MossImageOptions::default()).unwrap()["downloads"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn malformed_options_and_dmd_conflicts_fail_at_the_shared_boundary() {
        for advanced in [
            json!({"mossImage":null}),
            json!({"mossImage":{"unknown":1}}),
            json!({"mossImage":{"acceleration":"on"}}),
            json!({"mossImage":{"max_sequence_length":4097}}),
            json!({"mossImage":{"uncond_skip_layers":[42]}}),
            json!({"mossImage":{"slg":{"layers":[22],"scale":1,"start":0.8,"stop":0.2}}}),
        ] {
            assert!(native_request(&body(advanced), &entry()).is_err());
        }
        for (key, value) in [
            ("guidanceScale", json!(3)),
            ("steps", json!(9)),
            ("schedulerShift", json!(5)),
            ("sampler", json!("euler")),
            ("scheduler", json!("moss_training_grid")),
            ("guidanceMethod", json!("cfg")),
            ("cfgInterval", json!([0, 1])),
        ] {
            let mut advanced = json!({"mossImage":{"acceleration":"unmerged_bf16"}})
                .as_object()
                .unwrap()
                .clone();
            advanced.insert(key.into(), value);
            assert!(
                native_request(&body(Value::Object(advanced)), &entry()).is_err(),
                "{key}"
            );
        }
        for options in [
            json!({"acceleration":"unmerged_bf16","uncond_skip_layers":[]}),
            json!({"acceleration":"full_fp32","training_shift":1}),
        ] {
            assert!(native_request(&body(json!({"mossImage":options})), &entry()).is_err());
        }
    }
    #[test]
    fn variant_conditioning_stride_and_expanded_batch_fail_closed() {
        let mut p = body(json!({"mossImage":{"acceleration":"full_fp32"}}));
        p.insert("model".into(), json!("moss_image_base"));
        assert!(native_request(&p, &entry()).is_err());
        p.insert("model".into(), json!("iris_3b"));
        assert!(native_request(&p, &entry()).is_err());
        for (key, value) in [
            ("width", json!(17)),
            ("width", json!(0)),
            ("count", json!(9)),
            ("referenceAssetId", json!("ref")),
            ("loras", json!([{"id":"style"}])),
            ("mode", json!("edit_image")),
        ] {
            let mut p = body(json!({}));
            p.insert(key.into(), value);
            assert!(native_request(&p, &entry()).is_err(), "{key}");
        }
        let mut p = body(json!({}));
        p.extend(json!({"width":16,"height":16,"prompts":["日本語 <tag> {\"items\":[]}","other"],"count":4}).as_object().unwrap().clone());
        assert!(native_request(&p, &entry()).is_ok());
        p.insert("count".into(), json!(5));
        assert!(native_request(&p, &entry()).is_err());
    }
    #[test]
    fn reports_are_scrubbed_without_changing_requested_options() {
        let mut p = body(
            json!({"mossImage":{"acceleration":"off"},"mossImageExecutionReport":{"forged":true}}),
        );
        p.insert(REPORT_KEY.into(), json!({}));
        scrub_reports(&mut p);
        assert!(!p.contains_key(REPORT_KEY));
        assert!(p["advanced"].get(REPORT_KEY).is_none());
        assert_eq!(p["advanced"][OPTIONS_KEY]["acceleration"], "off");
    }
    #[test]
    fn shipped_catalog_closures_are_frozen_and_optional_packages_are_mode_specific() {
        use gen_core::moss_image::*;
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1;
        let manifest: Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw)).unwrap();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let entry = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == variant.provider_id())
                .unwrap();
            assert_eq!(entry["license"], "Apache-2.0");
            assert_eq!(entry["defaults"]["steps"], 50);
            assert_eq!(entry["capabilities"], json!(["text_to_image"]));
            for mode in ["off", "unmerged_bf16", "full_fp32"] {
                if variant == MossImageVariant::Base && mode != "off" {
                    continue;
                }
                let options: MossImageOptions =
                    serde_json::from_value(json!({"acceleration":mode})).unwrap();
                let selected = selected_components(entry.as_object().unwrap(), &options).unwrap();
                let mut components = Vec::new();
                for row in selected["downloads"].as_array().unwrap() {
                    let repo = row["repo"].as_str().unwrap();
                    let rev = row["revision"].as_str().unwrap();
                    let mut size = 0u64;
                    for path in row["files"].as_array().unwrap() {
                        let asset = moss_image_asset(repo, rev, path.as_str().unwrap())
                            .expect("every published download has frozen size+SHA identity");
                        size += asset.bytes;
                    }
                    assert_eq!(row["estimatedSizeBytes"].as_u64(), Some(size));
                    if let Some(id) = row["componentId"]
                        .as_str()
                        .filter(|id| id.starts_with("moss-dmd-"))
                    {
                        components.push(id.to_owned());
                        assert!(row.get("licenseBasis").is_some());
                    }
                }
                components.sort();
                let expected = match mode {
                    "off" => vec![],
                    "unmerged_bf16" => vec!["moss-dmd-dataspace", "moss-dmd-lora", "moss-dmd-time"],
                    _ => vec!["moss-dmd-dataspace", "moss-dmd-transformer"],
                };
                assert_eq!(components, expected);
            }
        }
    }
}
