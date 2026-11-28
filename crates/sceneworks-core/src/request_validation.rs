//! Native job-kind adapter for the portable static option policy.
use crate::contracts::JsonObject;
use std::borrow::Cow;

pub mod shape;
pub use sceneworks_options::request_validation::{
    model_id_for_request, validate_shape, ResolvedRequest, ValidationRefusal,
    VALIDATOR_SCHEMA_VERSION,
};

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
    sceneworks_options::request_validation::validate(kind, payload, &precision_facts(entry))?;
    admit_standalone_lossy_attention(kind, payload)
}

/// SW-6132: the shared validation picks the lossy-attention admission source by job type — the
/// model capability (inside the portable validator) for generation, the engine for a standalone
/// job (video upscale names SeedVR2, not a generation model). Image upscale has no validation
/// kind; the API's enqueue guard runs the same engine rule for it.
fn admit_standalone_lossy_attention(
    kind: &str,
    payload: &JsonObject,
) -> Result<(), ValidationRefusal> {
    if sceneworks_options::lossy_attention::is_standalone_validation_kind(kind) {
        crate::lossy_attention::standalone_lossy_attention(
            &crate::contracts::JobType::VideoUpscale,
            payload,
        )?;
    }
    Ok(())
}

pub fn resolve(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<JsonObject, ValidationRefusal> {
    let resolved =
        sceneworks_options::request_validation::resolve(kind, payload, &precision_facts(entry))?;
    admit_standalone_lossy_attention(kind, payload)?;
    Ok(resolved)
}

pub fn resolve_request(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<ResolvedRequest, ValidationRefusal> {
    let resolved = sceneworks_options::request_validation::resolve_request(
        kind,
        payload,
        &precision_facts(entry),
    )?;
    admit_standalone_lossy_attention(kind, payload)?;
    Ok(resolved)
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
