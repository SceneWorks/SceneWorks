//! Source-compatible native facade for portable Iris policy.
pub use sceneworks_options::training_validation::iris::restore::*;
pub fn validate_restore_dataset(
    config: &super::super::TrainingConfig,
    dataset: &super::super::TrainingDataset,
) -> Result<(), super::super::TrainingPlanError> {
    sceneworks_options::training_validation::iris::restore::validate_restore_dataset(
        config,
        &super::super::training_dataset_facts(dataset),
    )
}
