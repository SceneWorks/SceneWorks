//! Native typed adapters retain appearance and tensor authorization carriers.
pub type ImageJobRequest = sceneworks_options::generation_contracts::ImageJobRequest<
    crate::appearance::AppearanceSelection,
    crate::moss_image_advanced::EditAdvanced,
>;
pub type VideoJobRequest = sceneworks_options::generation_contracts::VideoJobRequest<
    crate::appearance::AppearanceSelection,
>;
pub use sceneworks_options::generation_contracts::{
    AudioJobRequest, InterleaveJobRequest, SpeechSegmentDto, VqaJobRequest,
};

pub use sceneworks_options::generation_contracts::{
    VectorDetailBudget, VectorMode, VectorRequest, VectorSampling, MAX_VECTOR_WALL_TIME_MS,
};
