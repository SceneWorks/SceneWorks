//! Native dataset adapters around the portable Iris option policy.
pub use sceneworks_options::training_validation::iris::*;
pub mod curricula;
pub mod depth;
pub mod floors;
pub mod restore;
use super::{TrainingConfig, TrainingDataset, TrainingPlanError, TrainingTarget};
pub fn validate_iris_dataset(
    target: &TrainingTarget,
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    sceneworks_options::training_validation::iris::validate_iris_dataset(
        target,
        config,
        &super::training_dataset_facts(dataset),
    )
}
pub fn validate_iris_dataset_captions(
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    sceneworks_options::training_validation::iris::validate_iris_dataset_captions(
        config,
        &super::training_dataset_facts(dataset),
    )
}
