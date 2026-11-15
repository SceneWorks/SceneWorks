//! Source-compatible native facade for portable Iris policy.
pub use sceneworks_options::training_validation::iris::curricula::*;
pub fn validate_iris_holdout(
    config: &super::super::TrainingConfig,
    dataset: &super::super::TrainingDataset,
) -> Result<(), super::super::TrainingPlanError> {
    sceneworks_options::training_validation::iris::curricula::validate_iris_holdout(
        config,
        &super::super::training_dataset_facts(dataset),
    )
}
pub fn is_holdout_item(item: &super::super::TrainingDatasetItem) -> bool {
    sceneworks_options::training_validation::iris::curricula::is_holdout_item(
        &super::super::training_item_facts(item),
    )
}
