//! Static policies for utility request carriers. Files, authorization and native providers stay local.
use crate::{
    contracts::{JsonObject, VideoUpscaleRequest},
    generation_contracts::*,
    request_validation::{shape, ValidationRefusal},
};
use serde_json::{json, Value};
pub const MAX_MOOD_BOARD_IMAGES: usize = 6;
pub const MAX_QWEN_REWRITE_IMAGES: usize = 10;

pub fn validate_character_test(request: &CharacterTestRequest) -> Result<(), ValidationRefusal> {
    if request.prompt.is_empty() || request.prompt.chars().count() > crate::MAX_PROMPT_CHARS {
        return Err(ValidationRefusal::new(
            "prompt",
            format!(
                "prompt must be between 1 and {} characters",
                crate::MAX_PROMPT_CHARS
            ),
        ));
    }
    crate::image_request::image_count_error(request.count).map_or(Ok(()), |detail| {
        Err(ValidationRefusal::new("count", detail))
    })?;
    shape::validate_dimension(request.width, "width", 4096)?;
    shape::validate_dimension(request.height, "height", 4096)
}

pub fn vector_prompt_stages(
    request: &VectorPromptWorkflowRequest,
) -> (ImageJobRequest, VectorRequest) {
    // Composition is explicit: the prompt belongs to the raster stage; vector consumes its image.
    let raster = serde_json::from_value(json!({
        "projectId":request.project_id, "projectName":request.project_name, "mode":"text_to_image",
        "prompt":request.prompt,"negativePrompt":request.negative_prompt,"model":request.raster_model,
        "count":1,"seed":request.seed,"width":request.width,"height":request.height,"requestedGpu":request.requested_gpu
    })).expect("typed raster workflow fields");
    let vector = VectorRequest {
        project_id: request.project_id.clone(),
        project_name: request.project_name.clone(),
        mode: VectorMode::ImageToSvg,
        model: request.vector_model.clone(),
        source_asset_id: Some("workflow_pending_source".into()),
        prompt: String::new(),
        sampling: request.sampling.clone(),
        detail_budget: request.detail_budget.clone(),
        requested_gpu: request.requested_gpu.clone(),
    };
    (raster, vector)
}

pub fn validate_vector_prompt(
    request: &VectorPromptWorkflowRequest,
) -> Result<(), ValidationRefusal> {
    let (raster, vector) = vector_prompt_stages(request);
    shape::validate_image_job(&raster)?;
    shape::validate_vector_request(&vector)
}

pub fn video_upscale_factor(factor: u8) -> Result<u32, ValidationRefusal> {
    match factor {
        2 | 4 => Ok(u32::from(factor)),
        other => Err(ValidationRefusal::new(
            "factor",
            format!("Video upscale supports only factor 2 or 4 (got {other})."),
        )),
    }
}
pub fn validate_video_upscale(request: &VideoUpscaleRequest) -> Result<(), ValidationRefusal> {
    if request.source_asset_id.trim().is_empty() {
        return Err(ValidationRefusal::new(
            "sourceAssetId",
            "Video upscale jobs require a source video asset.",
        ));
    }
    let engine = request.engine.trim().to_ascii_lowercase();
    if !matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b") {
        return Err(ValidationRefusal::new(
            "engine",
            format!("This video upscaler supports only engine=seedvr2 (got {engine})."),
        ));
    }
    video_upscale_factor(request.factor)?;
    if !request.softness.is_finite() {
        return Err(ValidationRefusal::new(
            "softness",
            "softness must be finite",
        ));
    }
    Ok(())
}
pub fn video_upscale_dimensions(
    request: &VideoUpscaleRequest,
    source_width: u32,
    source_height: u32,
) -> (u32, u32) {
    let factor = u32::from(request.factor);
    let (width, height) = match (request.target_width, request.target_height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => (
            source_width.saturating_mul(factor),
            source_height.saturating_mul(factor),
        ),
    };
    let snap = |value: u32| (value.saturating_add(8) / 16 * 16).clamp(16, 4096);
    (snap(width), snap(height))
}

/// Canonical discriminator: native prompt-refinement historically treats unknown tasks as rewrite.
pub fn prompt_refine_task(task: Option<&str>) -> &'static str {
    match task
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "magic_prompt" => "magic_prompt",
        "image_caption" => "image_caption",
        "image_describe" => "image_describe",
        "film_plan" => "film_plan",
        "qwen_image_rewrite" => "qwen_image_rewrite",
        _ => "refine",
    }
}
pub fn prompt_refine_workflow(workflow: Option<&str>) -> &'static str {
    match workflow.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("video") => "video",
        Some(value) if value.eq_ignore_ascii_case("audio") => "audio",
        _ => "image",
    }
}
pub fn prompt_refine_caption_style(style: Option<&str>) -> &'static str {
    if style.is_some_and(|value| value.trim().eq_ignore_ascii_case("tags")) {
        "tags"
    } else {
        "prose"
    }
}
/// Apply only the selected task's static defaults to an execution copy.
pub fn resolve_prompt_refine(payload: &JsonObject) -> JsonObject {
    let mut resolved = payload.clone();
    let text = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let task = prompt_refine_task(text("task"));
    resolved.insert("task".into(), json!(task));
    if task == "refine" {
        resolved.insert(
            "workflow".into(),
            json!(prompt_refine_workflow(text("workflow"))),
        );
    }
    if task == "magic_prompt" {
        resolved.insert(
            "aspectRatio".into(),
            json!(text("aspectRatio").unwrap_or("1:1")),
        );
    }
    if task == "image_describe" {
        resolved.insert(
            "captionStyle".into(),
            json!(prompt_refine_caption_style(text("captionStyle"))),
        );
    }
    if matches!(
        task,
        "image_caption" | "image_describe" | "qwen_image_rewrite"
    ) {
        resolved
            .entry("sourceAssetIds")
            .or_insert_with(|| json!([]));
    }
    resolved
}

/// Accept authored asset ids or the API-resolved path carrier; path confinement remains native.
pub fn validate_prompt_refine(payload: &JsonObject) -> Result<(), ValidationRefusal> {
    let text = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
    };
    let task = prompt_refine_task(Some(text("task")));
    let vision = matches!(task, "image_caption" | "image_describe");
    let rewrite = task == "qwen_image_rewrite";
    if text("prompt").is_empty() && !vision {
        return Err(ValidationRefusal::new("prompt", "Prompt cannot be empty"));
    }
    if vision || rewrite {
        let count = payload
            .get("sourceAssetIds")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .or_else(|| payload.get("imagePaths").and_then(Value::as_array))
            .map(Vec::len)
            .unwrap_or_else(|| {
                usize::from(
                    ["sourceAssetId", "imagePath", "referencePath"]
                        .iter()
                        .any(|key| !text(key).is_empty()),
                )
            });
        if count == 0 && vision {
            return Err(ValidationRefusal::new(
                "sourceAssetIds",
                "sourceAssetId (or sourceAssetIds) is required for a reference-image task",
            ));
        }
        let cap = if rewrite {
            MAX_QWEN_REWRITE_IMAGES
        } else {
            MAX_MOOD_BOARD_IMAGES
        };
        if count > cap {
            return Err(ValidationRefusal::new(
                "sourceAssetIds",
                if rewrite {
                    format!("Qwen Image 2.1 prompt rewriting accepts at most {cap} reference images — the same ordered list the render takes")
                } else {
                    format!("A mood board accepts at most {cap} reference images")
                },
            ));
        }
    }
    for (key, allowed) in [
        ("kvCompression", &["off", "qualified"][..]),
        ("thinkingMode", &["disabled", "enabled", "auto"][..]),
    ] {
        if let Some(value) = payload.get(key).filter(|v| !v.is_null()) {
            if !value
                .as_str()
                .is_some_and(|s| s.trim().is_empty() || allowed.contains(&s.trim()))
            {
                return Err(ValidationRefusal::new(
                    key,
                    format!("{key} must be {}", allowed.join(" or ")),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonControls {
    pub confidence: f64,
    pub segment: bool,
}
/// Shared detector/tracker controls. A present advanced confidence owns precedence even when
/// null; only an absent key consults the historical top-level spelling.
pub fn person_controls(payload: &JsonObject) -> Result<PersonControls, ValidationRefusal> {
    let advanced = match payload.get("advanced") {
        None | Some(Value::Null) => None,
        Some(Value::Object(value)) => Some(value),
        Some(_) => {
            return Err(ValidationRefusal::new(
                "advanced",
                "advanced must be an object",
            ))
        }
    };
    let confidence = advanced
        .and_then(|value| value.get("confidence"))
        .or_else(|| payload.get("confidence"));
    let confidence = match confidence {
        None | Some(Value::Null) => 0.25,
        Some(value) => value
            .as_f64()
            .or_else(|| value.as_str()?.parse().ok())
            .filter(|value: &f64| value.is_finite())
            .ok_or_else(|| {
                ValidationRefusal::new(
                    "advanced.confidence",
                    "confidence must be a finite number or numeric string",
                )
            })?,
    }
    .clamp(0.01, 1.0);
    let segment = match advanced.and_then(|value| value.get("segment")) {
        None | Some(Value::Null) => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(ValidationRefusal::new(
                "advanced.segment",
                "segment must be a boolean",
            ))
        }
    };
    Ok(PersonControls {
        confidence,
        segment,
    })
}

#[cfg(test)]
#[test]
fn person_controls_preserve_precedence_and_refuse_malformed_authored_values() {
    let resolve = |value: Value| person_controls(value.as_object().unwrap());
    assert_eq!(
        resolve(json!({})).unwrap(),
        PersonControls {
            confidence: 0.25,
            segment: true
        }
    );
    assert_eq!(resolve(json!({"confidence":0.7})).unwrap().confidence, 0.7);
    assert_eq!(
        resolve(json!({"confidence":0.7,"advanced":{"confidence":null}}))
            .unwrap()
            .confidence,
        0.25
    );
    assert_eq!(
        resolve(json!({"confidence":0.7,"advanced":{"confidence":"3","segment":false}})).unwrap(),
        PersonControls {
            confidence: 1.0,
            segment: false
        }
    );
    for value in [json!("bad"), json!(" 0.3 "), json!({})] {
        assert_eq!(
            resolve(json!({"confidence":value})).unwrap_err().field,
            "advanced.confidence"
        );
    }
}

pub fn validate_person_detection_job(
    payload: &PersonDetectionJobRequest,
) -> Result<(), ValidationRefusal> {
    if payload.source_asset_id.is_empty() {
        return Err(ValidationRefusal::new(
            "sourceAssetId",
            "Source clip is required",
        ));
    }
    if payload
        .source_timestamp
        .is_some_and(|timestamp| !timestamp.is_finite() || timestamp < 0.0)
    {
        return Err(ValidationRefusal::new(
            "sourceTimestamp",
            "sourceTimestamp must be greater than or equal to 0",
        ));
    }
    Ok(())
}

pub fn validate_person_track_job(payload: &PersonTrackJobRequest) -> Result<(), ValidationRefusal> {
    if payload.source_asset_id.is_empty() {
        return Err(ValidationRefusal::new(
            "sourceAssetId",
            "Source clip is required",
        ));
    }
    if payload.representative_frame_asset_id.is_empty() {
        return Err(ValidationRefusal::new(
            "representativeFrameAssetId",
            "Representative frame asset is required",
        ));
    }
    if payload.track_name.is_empty() || payload.track_name.chars().count() > 120 {
        return Err(ValidationRefusal::new(
            "trackName",
            "trackName must be between 1 and 120 characters",
        ));
    }
    if !payload.detection.contains_key("id") {
        return Err(ValidationRefusal::new(
            "detection.id",
            "Selected detection metadata is required",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::request_validation::{resolve, validate_shape};
    #[test]
    fn person_requests_share_required_identity_and_semantics() {
        for kind in ["personDetect", "personTrack"] {
            let valid = if kind == "personTrack" {
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{"id":"person"}})
            } else {
                json!({"sourceAssetId":"clip"})
            };
            let resolved = resolve(kind, valid.as_object().unwrap(), &JsonObject::new()).unwrap();
            if kind == "personTrack" {
                assert_eq!(resolved["trackName"], "Selected person");
            }
            for key in if kind == "personTrack" {
                vec!["sourceAssetId", "representativeFrameAssetId", "detection"]
            } else {
                vec!["sourceAssetId"]
            } {
                for remove in [true, false] {
                    let mut invalid = valid.as_object().unwrap().clone();
                    if remove {
                        invalid.remove(key);
                    } else {
                        invalid.insert(key.into(), Value::Null);
                    }
                    assert_eq!(validate_shape(kind, &invalid).unwrap_err().field, key);
                }
            }
            for (key, value, expected) in [
                ("advanced", Value::Null, "advanced"),
                ("sourceAssetId", json!(""), "sourceAssetId"),
            ] {
                let mut invalid = valid.clone();
                invalid[key] = value;
                assert_eq!(
                    validate_shape(kind, invalid.as_object().unwrap())
                        .unwrap_err()
                        .field,
                    expected
                );
            }
        }
        for (kind, payload, field) in [
            (
                "personDetect",
                json!({"sourceAssetId":"clip","sourceTimestamp":-1}),
                "sourceTimestamp",
            ),
            (
                "personTrack",
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{}}),
                "detection.id",
            ),
            (
                "personTrack",
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{"id":"p"},"trackName":""}),
                "trackName",
            ),
            (
                "personTrack",
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{"id":"p"},"trackName":"x".repeat(121)}),
                "trackName",
            ),
        ] {
            assert_eq!(
                validate_shape(kind, payload.as_object().unwrap())
                    .unwrap_err()
                    .field,
                field
            );
        }
    }
    #[test]
    fn prompt_defaults_are_applicable_and_authored_nulls_do_not_disappear() {
        let authored = json!({"prompt":"tree"});
        let resolved = resolve(
            "promptRefine",
            authored.as_object().unwrap(),
            &JsonObject::new(),
        )
        .unwrap();
        assert_eq!(
            resolved,
            json!({"prompt":"tree","task":"refine","workflow":"image"})
                .as_object()
                .unwrap()
                .clone()
        );
        assert_eq!(authored, json!({"prompt":"tree"}));
        assert_eq!(prompt_refine_workflow(Some("unknown")), "image");
        assert_eq!(prompt_refine_workflow(Some(" VIDEO ")), "video");
        for (payload, field) in [
            (
                json!({"prompt":"tree","sourceAssetIds":null}),
                "sourceAssetIds",
            ),
            (
                json!({"task":"image_caption","sourceAssetId":"source","prompt":null}),
                "prompt",
            ),
        ] {
            assert_eq!(
                validate_shape("promptRefine", payload.as_object().unwrap())
                    .unwrap_err()
                    .field,
                field
            );
        }
        let magic = resolve_prompt_refine(json!({"task":"magic_prompt"}).as_object().unwrap());
        assert_eq!(magic["aspectRatio"], "1:1");
        assert!(!magic.contains_key("captionStyle"));
        assert!(!magic.contains_key("workflow"));
        let describe = resolve_prompt_refine(
            json!({"task":" IMAGE_DESCRIBE ","captionStyle":"unknown"})
                .as_object()
                .unwrap(),
        );
        assert_eq!(describe["captionStyle"], "prose");
        assert_eq!(describe["sourceAssetIds"], json!([]));
        assert!(!describe.contains_key("aspectRatio"));
    }
}
