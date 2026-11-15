//! Native job-kind adapter for the portable static option policy.
pub mod shape;
pub use sceneworks_options::request_validation::{
    model_id_for_request, resolve, resolve_request, validate, validate_shape, ResolvedRequest,
    ValidationRefusal, VALIDATOR_SCHEMA_VERSION,
};
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
