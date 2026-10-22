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
pub const REPORT_KEY: &str = "mossImageExecutionReport";
pub fn is_moss(model: &str) -> bool {
    MossImageVariant::from_provider_id(model).is_some()
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
    let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
    let advanced = payload
        .get("advanced")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let Some(variant) = MossImageVariant::from_provider_id(model) else {
        return if advanced.contains_key(OPTIONS_KEY) {
            Err("advanced.mossImage is only supported by MOSS-Image Base/SFT".into())
        } else {
            Ok(None)
        };
    };
    if payload.get("advanced").is_some_and(|v| !v.is_object()) {
        return Err("advanced must be an object".into());
    }
    if payload
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|s| s != "text_to_image")
    {
        return Err("MOSS-Image supports text_to_image only".into());
    }
    for key in [
        "sourceAssetId",
        "referenceAssetId",
        "referenceAssetIds",
        "maskAssetId",
        "characterId",
        "characterLookId",
    ] {
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
    let (dw, dh) = crate::image_request::default_resolution(entry).unwrap_or((1024, 1024));
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
    let expanded = request
        .count
        .checked_mul(request.prompt_batch.len().max(1) as u32)
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
    MossImageResolved::resolve(variant, &request).map_err(|e| e.to_string())?;
    Ok(Some(request))
}
/// Stage ONLY the selected optional package. Ordinary SFT remains usable without any DMD files.
pub fn selected_components(
    entry: &JsonObject,
    options: &MossImageOptions,
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
            .is_some_and(|id| id.starts_with("moss-dmd-"))
    });
    if let Some(purpose) = purpose(options) {
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
    payload.remove(REPORT_KEY);
    payload.remove("moss_image_execution_report");
    if let Some(advanced) = payload.get_mut("advanced").and_then(Value::as_object_mut) {
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
