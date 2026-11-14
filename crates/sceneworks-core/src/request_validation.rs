//! Native job-kind adapter for the portable static option policy.
pub mod shape;
pub use sceneworks_options::request_validation::{
    resolve, validate, validate_shape, ValidationRefusal,
};
pub fn kind_for_job(job_type: &crate::contracts::JobType) -> Option<&'static str> {
    use crate::contracts::JobType::*;
    match job_type {
        ImageGenerate | ImageEdit => Some("image"),
        VideoGenerate | VideoExtend | VideoBridge | PersonReplace => Some("video"),
        AudioGenerate => Some("audio"),
        LoraTrain => Some("lora_train"),
        ControlTraining => Some("control_training"),
        VectorGenerate => Some("vector"),
        VideoUpscale => Some("videoUpscale"),
        PromptRefine => Some("promptRefine"),
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
        payload
            .get("modelManifestEntry")
            .and_then(serde_json::Value::as_object)
            .cloned()
            .unwrap_or_default()
    }
}
