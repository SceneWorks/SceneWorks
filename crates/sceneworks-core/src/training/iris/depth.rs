//! Source-compatible native facade for portable Iris policy.
pub use sceneworks_options::training_validation::iris::depth::*;
pub fn validate_depth_dataset(
    dataset: &super::super::TrainingDataset,
) -> Result<(), super::super::TrainingPlanError> {
    sceneworks_options::training_validation::iris::depth::validate_depth_dataset(
        &super::super::training_dataset_facts(dataset),
    )
}
