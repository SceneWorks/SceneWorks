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
        context.entry("kind").or_insert_with(|| json!(kind));
        context.insert(
            "model".into(),
            payload
                .get("model")
                .cloned()
                .unwrap_or_else(|| json!(default_model_for(kind))),
        );
        context.insert(
            "mode".into(),
            payload
                .get("mode")
                .cloned()
                .unwrap_or_else(|| json!(default_mode_for(kind))),
        );
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
    let mut fields = payload.clone();
    // Carrier version is checked by the canonical upgrader, not a generation DTO option.
    fields.remove("optionsSchemaVersion");
    serde_path_to_error::deserialize(Value::Object(fields)).map_err(|error| {
        let detail = error.inner().to_string();
        let field = detail
            .strip_prefix("missing field `")
            .and_then(|rest| rest.split('`').next())
            .map(str::to_owned)
            .unwrap_or_else(|| error.path().to_string());
        ValidationRefusal::new(field, detail)
    })
}
fn check(field: &str, error: Option<String>) -> Result<(), ValidationRefusal> {
    error.map_or(Ok(()), |detail| Err(ValidationRefusal::new(field, detail)))
}

/// Shape policy runs on raw JSON, before any tolerant historical request parser. This is also
/// the HTTP extractor's gate before DTO deserialization can erase malformed advanced carriers.
pub fn validate_shape(kind: &str, payload: &JsonObject) -> Result<(), ValidationRefusal> {
    if crate::training_validation::handles(kind) {
        return crate::training_validation::validate_shape(kind, payload);
    }
    if kind == "audio" && payload.contains_key("yue2") {
        let block = payload
            .get("yue2")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ValidationRefusal::new("yue2", "yue2 must be an object").with_request(kind, payload)
            })?;
        return validate_yue2("yue2Execution", block)
            .map_err(|error| error.with_request(kind, payload));
    }
    if matches!(kind, "yue2" | "yue2Execution") {
        return validate_yue2(kind, payload).map_err(|error| error.with_request(kind, payload));
    }
    (|| {
        let numeric_payload = crate::sensenova_validation::canonical_numbers(kind, payload)?;
        crate::validate::shape(kind, &Value::Object(numeric_payload))
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
            "promptRefine" => {
                decode::<PromptRefineRequest>(payload)?;
                crate::utility_validation::validate_prompt_refine(payload)
            }
            "personDetect" => {
                crate::utility_validation::validate_person_detection_job(&decode(payload)?)?;
                crate::utility_validation::person_controls(payload).map(|_| ())
            }
            "personTrack" => {
                crate::utility_validation::validate_person_track_job(&decode(payload)?)?;
                crate::utility_validation::person_controls(payload).map(|_| ())
            }
            _ => Ok(()),
        }
    })()
    .map_err(|error: ValidationRefusal| error.with_request(kind, payload))
}

/// Owner-issued effective options. Its private payload prevents execution bindings from claiming
/// an arbitrary authored map already passed static validation and lowering.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ResolvedRequest(JsonObject);
impl ResolvedRequest {
    pub fn as_payload(&self) -> &JsonObject {
        &self.0
    }
    pub fn into_payload(self) -> JsonObject {
        self.0
    }
}
pub fn resolve_request(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<ResolvedRequest, ValidationRefusal> {
    resolve(kind, payload, entry).map(ResolvedRequest)
}

/// Resolve omitted scalar values using the same model defaults the native readers use. The
/// returned object is a preview/execution view; the authored payload and omission intent survive.
pub fn resolve(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<JsonObject, ValidationRefusal> {
    if crate::training_validation::handles(kind) {
        return crate::training_validation::resolve_payload(kind, payload, entry);
    }
    validate(kind, payload, entry)?;
    let mut resolved = resolved_values(kind, payload, entry);
    let defaults = match kind {
        "vector" => Some(serde_json::to_value(decode_vector(payload)?).expect("vector defaults")),
        "vectorPrompt" => Some(
            serde_json::to_value(decode::<VectorPromptWorkflowRequest>(payload)?)
                .expect("workflow defaults"),
        ),
        "characterTest" => Some(
            serde_json::to_value(decode::<CharacterTestRequest>(payload)?)
                .expect("character defaults"),
        ),
        "videoUpscale" => Some(
            serde_json::to_value(decode::<crate::contracts::VideoUpscaleRequest>(payload)?)
                .expect("upscale defaults"),
        ),
        "vqa" => {
            Some(serde_json::to_value(decode::<VqaJobRequest>(payload)?).expect("VQA defaults"))
        }
        "interleave" => Some(
            serde_json::to_value(decode::<InterleaveJobRequest>(payload)?)
                .expect("interleave defaults"),
        ),
        "audio" if !payload.contains_key("yue2") => {
            Some(serde_json::to_value(decode::<AudioJobRequest>(payload)?).expect("audio defaults"))
        }
        _ => None,
    };
    if let Some(Value::Object(defaults)) = defaults {
        resolved.extend(defaults);
    }
    if matches!(kind, "image" | "video" | "vqa" | "interleave") {
        crate::effective_values::resolve_advanced(&mut resolved, entry);
    }
    match kind {
        "vqa" => crate::sensenova_validation::materialize_vqa(payload, &mut resolved),
        "interleave" => crate::sensenova_validation::materialize_interleave(payload, &mut resolved),
        _ => {}
    }
    if kind == "promptRefine" {
        resolved = crate::utility_validation::resolve_prompt_refine(&resolved);
    }
    if kind == "personTrack" && !resolved.contains_key("trackName") {
        resolved.insert("trackName".into(), json!(default_track_name()));
    }
    if matches!(kind, "personDetect" | "personTrack") {
        let controls = crate::utility_validation::person_controls(payload)?;
        let advanced = resolved.entry("advanced").or_insert_with(|| json!({}));
        if advanced.is_null() {
            *advanced = json!({});
        }
        advanced["confidence"] = json!(controls.confidence);
        advanced["segment"] = json!(controls.segment);
    }
    Ok(resolved)
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
    if crate::training_validation::handles(kind) {
        return crate::training_validation::validate(kind, payload, entry);
    }
    validate_shape(kind, payload)?;
    if kind == "audio" && payload.contains_key("yue2") {
        return Ok(());
    }
    (|| {
        let empty = JsonObject::new();
        let advanced = payload
            .get("advanced")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let default_model = default_model_for(kind);
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .or(default_model.as_deref())
            .unwrap_or_default();
        if advanced.get("lowResStart").is_some_and(|v| !v.is_null())
            && entry.get("features").and_then(|f| f.get("low_res_start"))
                != Some(&Value::Bool(true))
        {
            return Err(ValidationRefusal::new(
                "advanced.lowResStart",
                "The selected model route does not support Low Res Start.",
            ));
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
                let promptless = model == "svd"
                    && request.mode == "image_to_video"
                    && entry.get("id").and_then(Value::as_str) == Some(model)
                    && entry.get("promptless") == Some(&Value::Bool(true));
                if request.prompt.trim().is_empty() && !promptless {
                    return Err(ValidationRefusal::new("prompt", "prompt must be between 1 and 4000 characters"));
                }
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
                let mut canonical = payload.clone();
                canonical.insert("model".into(), json!(request.model));
                canonical.insert("mode".into(), json!(request.mode));
                if !crate::video_eligibility::any_eligible(
                    crate::video_eligibility::VideoJobKind::for_mode(&request.mode).as_str(),
                    &canonical,
                    entry,
                ) {
                    return Err(ValidationRefusal::new("mode", format!(
                        "{model} cannot render the \"{}\" mode — no backend implements it, so this job would wait for a worker that will never claim it. Choose a mode this model lists in its capabilities, or a model that supports this one.", request.mode
                    )));
                }
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

fn validate_yue2(kind: &str, payload: &JsonObject) -> Result<(), ValidationRefusal> {
    let spec: crate::yue2_validation::Yue2JobSpec = decode(payload)?;
    let result = if kind == "yue2Execution" {
        crate::yue2_validation::validate_for_execution(&spec)
    } else {
        crate::yue2_validation::validate_new_submission(&spec)
    };
    result.map_err(|error| ValidationRefusal {
        code: error.code,
        field: error.field,
        detail: error.message,
        context: Box::new(json!({})),
    })
}
