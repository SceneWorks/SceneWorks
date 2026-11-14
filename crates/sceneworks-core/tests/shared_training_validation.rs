//! Weights-free native catalog evidence for the portable training policy.
use sceneworks_core::training::*;
use serde_json::json;

#[test]
fn every_real_training_target_default_passes_portable_policy_on_both_catalog_projections() {
    for candle in [false, true] {
        let catalog = effective_training_targets_for_candle(candle);
        for target in &catalog.targets {
            let facts = serde_json::to_value(target).unwrap();
            let payload = json!({"targetId": target.id, "datasetId": "dataset", "outputName": "adapter", "dryRun": false, "config": target.defaults});
            validate(
                "training",
                payload.as_object().unwrap(),
                facts.as_object().unwrap(),
            )
            .unwrap_or_else(|error| panic!("candle={candle}/{}: {error:?}", target.id));
        }
    }
}

#[test]
fn native_config_and_portable_boundaries_share_limits_and_preserve_catalog_metadata() {
    let target = builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .unwrap();
    let mut config = target.defaults.clone();
    config.rank = 129;
    let expected =
        policy_refusal(validate_training_config_for_target(&target, &config).unwrap_err());
    let payload = json!({"plan": {"target": {"targetId": target.id, "kernel": target.kernel, "baseModel": target.base_model}, "config": config}});
    let facts = serde_json::to_value(&target).unwrap();
    assert_eq!(
        validate(
            "lora_train",
            payload.as_object().unwrap(),
            facts.as_object().unwrap()
        )
        .unwrap_err(),
        expected
    );
    assert_eq!(expected.code, "training_target_limit");
    assert_eq!(expected.context["maximum"], 128);
}
