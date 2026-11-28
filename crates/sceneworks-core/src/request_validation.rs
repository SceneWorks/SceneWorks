//! Native job-kind adapter for the portable static option policy.
use crate::contracts::JsonObject;
use std::borrow::Cow;

pub mod shape;
pub use sceneworks_options::request_validation::{
    model_id_for_request, validate_shape, ResolvedRequest, ValidationRefusal,
    VALIDATOR_SCHEMA_VERSION,
};

/// The same refusal is used for current requests and stored carriers from older releases.
pub fn reject_retired_image_identity(field: &str, identity: &str) -> Result<(), ValidationRefusal> {
    let Some(detail) = crate::model_family::retired_image_detail(identity) else {
        return Ok(());
    };
    let mut refusal = ValidationRefusal::new(field, detail);
    refusal.code = "retired_model";
    refusal.context = Box::new(serde_json::json!({ "identity": identity, "field": field }));
    Err(refusal)
}

pub fn reject_retired_image_entry(entry: &JsonObject) -> Result<(), ValidationRefusal> {
    if let Some(identity) = crate::model_family::retired_image_entry(entry) {
        reject_retired_image_identity("model", identity)?;
    }
    Ok(())
}

pub fn reject_retired_image_job(
    job_type: &crate::contracts::JobType,
    payload: &JsonObject,
) -> Result<(), ValidationRefusal> {
    use crate::contracts::JobType::*;
    use serde_json::Value;
    let fields: &[&str] = match job_type {
        ImageGenerate | ImageEdit | ImageDetail | ImageUpscale | ImageInterleave | PromptRefine => {
            &["model", "checkpointModelId"]
        }
        ModelDownload | ModelConvert | ModelImport | LoraImport => {
            &["modelId", "family", "baseModel"]
        }
        LoraTrain | ControlTraining => &["baseModel", "targetId", "kernel"],
        _ => return Ok(()),
    };
    for field in fields {
        if let Some(identity) = payload.get(*field).and_then(Value::as_str) {
            reject_retired_image_identity(field, identity)?;
        }
        if let Some(plan) = payload.get("plan") {
            let identity = plan.get(*field).or_else(|| plan.get("target")?.get(*field));
            if let Some(identity) = identity.and_then(Value::as_str) {
                reject_retired_image_identity(field, identity)?;
            }
        }
    }
    for key in ["modelManifestEntry", "baseModelManifestEntry"] {
        if let Some(entry) = payload.get(key).and_then(Value::as_object) {
            reject_retired_image_entry(entry)?;
        }
    }
    if let Some(loras) = payload.get("loras").and_then(Value::as_array) {
        for lora in loras {
            if let Some(identity) = crate::lora_family::retired_lora_identity(lora) {
                reject_retired_image_identity("loras", &identity)?;
            }
        }
    }
    Ok(())
}

fn precision_facts(entry: &JsonObject) -> Cow<'_, JsonObject> {
    if crate::source_precision::SourcePrecision::for_entry(entry).is_none()
        && !entry.contains_key("sourcePrecision")
    {
        return Cow::Borrowed(entry);
    }
    let mut projected = entry.clone();
    crate::source_precision::SourcePrecision::project(&mut projected);
    Cow::Owned(projected)
}

pub fn validate(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<(), ValidationRefusal> {
    sceneworks_options::request_validation::validate(kind, payload, &precision_facts(entry))
}

pub fn resolve(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<JsonObject, ValidationRefusal> {
    sceneworks_options::request_validation::resolve(kind, payload, &precision_facts(entry))
}

pub fn resolve_request(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<ResolvedRequest, ValidationRefusal> {
    sceneworks_options::request_validation::resolve_request(kind, payload, &precision_facts(entry))
}
pub fn kind_for_job(job_type: &crate::contracts::JobType) -> Option<&'static str> {
    use crate::contracts::JobType::*;
    match job_type {
        ImageGenerate | ImageEdit => Some("image"),
        ImageDetail => Some("imageDetail"),
        ModelConvert | DatasetAnalysis | TrainingCaption => Some("utility"),
        VideoGenerate | VideoExtend | VideoBridge | PersonReplace => Some("video"),
        AudioGenerate => Some("audio"),
        LoraTrain => Some("lora_train"),
        ControlTraining => Some("control_training"),
        VectorGenerate => Some("vector"),
        VideoUpscale => Some("videoUpscale"),
        PromptRefine => Some("promptRefine"),
        PersonDetect => Some("personDetect"),
        PersonTrack => Some("personTrack"),
        ImageVqa => Some("vqa"),
        ImageInterleave => Some("interleave"),
        _ => None,
    }
}

pub use sceneworks_options::prompt_enhancement;

pub use sceneworks_options::utility_validation;

/// A server-created training plan carries checkpoint paths derived from previously validated
/// job ids. Revalidate its client controls without presenting those server-owned paths as new
/// client input. Call only after the plan has crossed the typed API admission boundary; keep the
/// original plan for engine execution.
pub fn trusted_training_plan_validation_view(
    payload: &crate::contracts::JsonObject,
) -> crate::contracts::JsonObject {
    let mut view = payload.clone();
    if let Some(advanced) = view
        .get_mut("plan")
        .and_then(|plan| plan.get_mut("config"))
        .and_then(|config| config.get_mut("advanced"))
        .and_then(serde_json::Value::as_object_mut)
    {
        for key in ["iris", "iris_restore"] {
            if let Some(options) = advanced
                .get_mut(key)
                .and_then(serde_json::Value::as_object_mut)
            {
                options.remove("resume_from");
                options.remove("load_from");
            }
        }
    }
    view
}

/// Authoritative catalog facts for training jobs are native metadata; callers never trust a
/// submitted plan's limit declarations. Generation keeps its already-resolved ModelSpec entry.
pub fn facts_for_job(
    job_type: &crate::contracts::JobType,
    payload: &crate::contracts::JsonObject,
) -> crate::contracts::JsonObject {
    if matches!(
        job_type,
        crate::contracts::JobType::LoraTrain | crate::contracts::JobType::ControlTraining
    ) {
        serde_json::to_value(crate::training::effective_training_targets_for_candle(
            !cfg!(target_os = "macos"),
        ))
        .expect("training catalog")
        .as_object()
        .unwrap()
        .clone()
    } else {
        let mut entry = payload
            .get("modelManifestEntry")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        crate::registry::project_option_route_facts(&mut entry);
        entry.as_object().cloned().unwrap_or_default()
    }
}
