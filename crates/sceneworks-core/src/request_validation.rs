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
