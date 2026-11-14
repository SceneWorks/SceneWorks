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

/// Accept authored asset ids or the API-resolved path carrier; path confinement remains native.
pub fn validate_prompt_refine(payload: &JsonObject) -> Result<(), ValidationRefusal> {
    let text = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
    };
    let task = text("task");
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
