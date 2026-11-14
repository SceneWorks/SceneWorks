//! Executable static validation at every generation boundary. Inputs are canonical authored
//! JSON and authoritative catalog facts. Asset ownership, installation, device availability,
//! memory admission and native provider validation remain with their existing owners.
pub mod shape;

/// Version of the portable validator input/output contract (independent of workflow carriers).
pub const VALIDATOR_SCHEMA_VERSION: u32 = 1;

use crate::contracts::JsonObject;
use crate::generation_contracts::*;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ValidationRefusal {
    pub code: &'static str,
    pub field: String,
    pub detail: String,
    pub context: Box<Value>,
}

impl ValidationRefusal {
    pub fn new(field: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: "invalid_generation_option",
            field: field.into(),
            detail: detail.into(),
            context: Box::new(json!({})),
        }
    }
    // Existing source-owned compound rules retain their human-readable wording. The stable
    // machine field identifies a mentioned declaration, or the whole request for compound rules.
    pub fn bad_request(detail: impl Into<String>) -> Self {
        let detail = detail.into();
        let field = crate::registry()
            .filter_map(|option| {
                let path = if option.location.starts_with("advanced") {
                    option.id.to_owned()
                } else if let Some((_, nested)) = option.location.split_once('.') {
                    format!("{nested}.{}", option.key)
                } else {
                    option.key.to_owned()
                };
                if detail.contains(&path) {
                    Some((path.len() + 1000, path))
                } else if detail.contains(option.key) {
                    Some((option.key.len(), option.key.to_owned()))
                } else {
                    None
                }
            })
            .max_by_key(|(rank, _)| *rank)
            .map(|(_, field)| field)
            .unwrap_or_else(|| "request".into());
        Self::new(field, detail)
    }
    pub fn with_request(mut self, kind: &str, payload: &JsonObject) -> Self {
        let context = self
            .context
            .as_object_mut()
            .expect("refusal context object");
        context.insert("kind".into(), json!(kind));
        context.insert("model".into(), json!(payload.get("model")));
        context.insert("mode".into(), json!(payload.get("mode")));
        context.insert("field".into(), json!(self.field));
        self
    }
}
impl std::fmt::Display for ValidationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}
impl std::error::Error for ValidationRefusal {}

fn decode<T: DeserializeOwned>(payload: &JsonObject) -> Result<T, ValidationRefusal> {
    serde_path_to_error::deserialize(Value::Object(payload.clone())).map_err(|error| {
        ValidationRefusal::new(error.path().to_string(), error.inner().to_string())
    })
}
fn check(field: &str, error: Option<String>) -> Result<(), ValidationRefusal> {
    error.map_or(Ok(()), |detail| Err(ValidationRefusal::new(field, detail)))
}

/// Shape policy runs on raw JSON, before any tolerant historical request parser. This is also
/// the HTTP extractor's gate before DTO deserialization can erase malformed advanced carriers.
pub fn validate_shape(kind: &str, payload: &JsonObject) -> Result<(), ValidationRefusal> {
    (|| {
        crate::validate::shape(kind, &Value::Object(payload.clone()))
            .map_err(|error| ValidationRefusal::new(error.field, error.detail))?;
        match kind {
            "image" => shape::validate_image_job(&decode::<ImageJobRequest>(payload)?),
            "video" => shape::validate_video_job(&decode::<VideoJobRequest>(payload)?),
            "audio" => shape::validate_audio_job(&decode(payload)?),
            "vqa" => shape::validate_vqa_job(&decode(payload)?),
            "interleave" => shape::validate_interleave_job(&decode(payload)?),
            "vector" => shape::validate_vector_request(&decode_vector(payload)?),
            "vectorPrompt" => crate::utility_validation::validate_vector_prompt(&decode(payload)?),
            "characterTest" => {
                crate::utility_validation::validate_character_test(&decode(payload)?)
            }
            "videoUpscale" => crate::utility_validation::validate_video_upscale(&decode(payload)?),
            "promptRefine" => crate::utility_validation::validate_prompt_refine(payload),
            _ => Ok(()),
        }
    })()
    .map_err(|error: ValidationRefusal| error.with_request(kind, payload))
}

/// Resolve omitted scalar values using the same model defaults the native readers use. The
/// returned object is a preview/execution view; the authored payload and omission intent survive.
pub fn resolve(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<JsonObject, ValidationRefusal> {
    validate(kind, payload, entry)?;
    Ok(resolved_values(kind, payload, entry))
}
fn resolved_values(kind: &str, payload: &JsonObject, entry: &JsonObject) -> JsonObject {
    let mut result = payload.clone();
    let resolution = match kind {
        "image" => crate::image_request::default_resolution(entry).unwrap_or((1024, 1024)),
        "video" => crate::video_request::default_resolution(entry).unwrap_or((768, 512)),
        _ => return result,
    };
    let derived = kind == "image"
        && payload
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(crate::image_request::flux3::is_model)
        && crate::image_request::flux3::parse(payload).is_ok_and(|v| !v.asset_ids.is_empty());
    if !derived {
        for (key, value) in [("width", resolution.0), ("height", resolution.1)] {
            if result.get(key).is_none_or(Value::is_null) {
                result.insert(key.to_owned(), json!(value));
            }
        }
    }
    if kind == "image" && result.get("count").is_none_or(Value::is_null) {
        result.insert(
            "count".to_owned(),
            json!(crate::image_request::default_count(entry).unwrap_or(4)),
        );
    }
    if kind == "video" {
        if result.get("duration").is_none_or(Value::is_null) {
            result.insert(
                "duration".to_owned(),
                json!(crate::video_request::resolve_duration(None, entry)),
            );
        }
        if result.get("fps").is_none_or(Value::is_null) {
            result.insert(
                "fps".to_owned(),
                json!(crate::video_request::resolve_fps(None, entry)),
            );
        }
    }
    result
}
pub fn validate(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<(), ValidationRefusal> {
    validate_shape(kind, payload)?;
    (|| {
        let empty = JsonObject::new();
        let advanced = payload
            .get("advanced")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(value) = advanced.get("lowResStart").filter(|v| !v.is_null()) {
            if value.get("enabled") != Some(&Value::Bool(false))
                && entry.get("features").and_then(|f| f.get("low_res_start"))
                    != Some(&Value::Bool(true))
            {
                return Err(ValidationRefusal::new(
                    "advanced.lowResStart",
                    "The selected model route does not support Low Res Start.",
                ));
            }
        }
        if matches!(kind, "image" | "video") {
            if let Some(steps) = crate::limits::requested_steps(advanced) {
                check(
                    "advanced.steps",
                    crate::limits::steps_limit_error(model, steps, entry),
                )?;
            }
        }
        match kind {
            "image" => {
                crate::prompt_enhancement::validate(model, advanced)?;
                check(
                    "advanced.quantTier",
                    crate::image_request::image_quant_tier_refusal(model, advanced),
                )?;
                crate::image_request::validate_image_request_payload(payload, entry)
                    .map_err(ValidationRefusal::bad_request)?;
                let resolved = resolved_values(kind, payload, entry);
                let request: ImageJobRequest = decode(&resolved)?;
                let hi = crate::image_request::HiresFixInput {
                    hires_fix: &request.hires_fix,
                    upscale: &request.upscale,
                    mode: &request.mode,
                    model: &request.model,
                    model_manifest_entry: entry,
                    source_asset_id: &request.source_asset_id,
                    reference_asset_id: &request.reference_asset_id,
                    reference_asset_ids: &request.reference_asset_ids,
                    mask_asset_id: &request.mask_asset_id,
                    character_id: &request.character_id,
                    character_look_id: &request.character_look_id,
                    advanced: &request.advanced,
                    width: request.width.unwrap_or(1024),
                    height: request.height.unwrap_or(1024),
                };
                check(
                    "hiresFix",
                    crate::image_request::hires_fix_request_error(
                        &hi,
                        entry
                            .get("family")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                )?;
                check(
                    "advanced.sampler",
                    crate::image_limits::image_sampling_knob_error(model, advanced, entry),
                )?;
                check(
                    "advanced",
                    crate::image_sampling_controls::image_sampling_control_error(
                        model,
                        advanced,
                        &request.negative_prompt,
                        entry,
                    ),
                )?;
                check(
                    "prompts",
                    crate::image_sampling_controls::prompt_batch_error(
                        model,
                        &request.prompt,
                        &request.prompts,
                        request.count.unwrap_or(4),
                        entry,
                    ),
                )?;
                check(
                    "referenceAssetIds",
                    crate::image_request::ordered_image_reference_error(model, payload, entry),
                )?;
                let references =
                    crate::image_request::ordered_image_reference_ids_for_model(model, payload)
                        .map_or(0, |v| v.len());
                check(
                    "referenceAssetIds",
                    crate::image_limits::image_reference_limit_error(model, references, entry),
                )?;
                if let (Some(w), Some(h)) = (request.width, request.height) {
                    check(
                        "width",
                        crate::admission_geometry::refuse_over_envelope(
                            model,
                            entry,
                            w,
                            h,
                            u32::try_from(references).unwrap_or(u32::MAX),
                            request.count.unwrap_or(4),
                        ),
                    )?;
                }
            }
            "video" => {
                shape::validate_video_reference_asset_ids_payload(
                    payload,
                    &Value::Object(entry.clone()),
                )?;
                let request: VideoJobRequest = decode(payload)?;
                let auto = crate::video_request::normalize_ltx25_generation_controls(
                    model,
                    &mut payload.clone(),
                    &Value::Object(entry.clone()),
                )?;
                if !auto {
                    check(
                        "duration",
                        crate::video_request::duration_limit_error(
                            model,
                            crate::video_request::resolve_duration(payload.get("duration"), entry),
                            entry,
                        ),
                    )?;
                }
                crate::video_request::requested_reference_image_short_edge(advanced)
                    .map_err(ValidationRefusal::bad_request)?;
                crate::video_request::requested_reference_image_resize_mode(advanced)
                    .map_err(ValidationRefusal::bad_request)?;
                if advanced.contains_key("veda") {
                    let recipe =
                        crate::minimax_h3_turbo::resolve_turbo_recipe(model, &request.loras)
                            .map_err(ValidationRefusal::bad_request)?;
                    let steps = crate::limits::requested_steps(advanced)
                        .or_else(|| recipe.map(|r| r.steps))
                        .unwrap_or(50);
                    crate::h3_veda::requested_h3_veda(model, advanced, steps)
                        .map_err(ValidationRefusal::bad_request)?;
                }
                check(
                    "fps",
                    crate::video_request::fps_limit_error(
                        model,
                        crate::video_request::resolve_fps(payload.get("fps"), entry),
                        entry,
                    ),
                )?;
                check(
                    "referenceAssetIds",
                    crate::video_request::reference_limit_error(
                        model,
                        request.reference_asset_ids.len(),
                        request.source_clip_asset_ids.len(),
                        request.reference_audio_asset_ids.len(),
                        entry,
                    ),
                )?;
            }
            "vectorPrompt" => {
                let request: VectorPromptWorkflowRequest = decode(payload)?;
                let (raster, vector) = crate::utility_validation::vector_prompt_stages(&request);
                let raster_payload = serde_json::to_value(raster).expect("raster request");
                let vector_payload = serde_json::to_value(vector).expect("vector request");
                validate(
                    "image",
                    raster_payload.as_object().unwrap(),
                    entry
                        .get("raster")
                        .and_then(Value::as_object)
                        .unwrap_or(&empty),
                )?;
                validate(
                    "vector",
                    vector_payload.as_object().unwrap(),
                    entry
                        .get("vector")
                        .and_then(Value::as_object)
                        .unwrap_or(&empty),
                )?;
            }
            "vector" => {
                let request = decode_vector(payload)?;
                shape::validate_vector_model_manifest(
                    &request.model,
                    request.mode,
                    &Value::Object(entry.clone()),
                )?;
                shape::validate_vector_provider_request(&request, &Value::Object(entry.clone()))?;
            }
            "audio" => shape::validate_audio_job_for_model(
                &decode(payload)?,
                &Value::Object(entry.clone()),
            )?,
            _ => {}
        }
        Ok(())
    })()
    .map_err(|error: ValidationRefusal| error.with_request(kind, payload))
}

fn decode_vector(payload: &JsonObject) -> Result<VectorRequest, ValidationRefusal> {
    let mut authored = payload.clone();
    for key in [
        "modelManifestEntry",
        "modelProvenance",
        "workflow",
        "expectedImportedModelRef",
    ] {
        authored.remove(key);
    }
    decode(&authored)
}
