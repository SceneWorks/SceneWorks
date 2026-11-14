use sceneworks_options::training_validation::{self as training, iris, TrainingTarget};
use serde_json::{json, Value};

fn facts(target: &TrainingTarget) -> serde_json::Map<String, Value> {
    serde_json::to_value(target)
        .unwrap()
        .as_object()
        .unwrap()
        .clone()
}
fn submission(target: &TrainingTarget) -> serde_json::Map<String, Value> {
    json!({"targetId": target.id, "datasetId": "dataset", "outputName": "adapter", "config": target.defaults})
        .as_object().unwrap().clone()
}
fn job(target: &TrainingTarget, config: Value) -> serde_json::Map<String, Value> {
    json!({"plan": {"target": {"targetId": target.id, "kernel": target.kernel, "baseModel": target.base_model}, "config": config}})
        .as_object().unwrap().clone()
}

#[test]
fn submit_and_plan_seams_preserve_typed_target_bounds() {
    let target = iris::iris_generation_target();
    let mut submit = submission(&target);
    submit["config"]["rank"] = json!(100_000);
    let plan = job(&target, submit["config"].clone());
    let expected = training::validate("training", &submit, &facts(&target)).unwrap_err();
    for kind in ["lora_train", "control_training"] {
        assert_eq!(
            training::validate(kind, &plan, &facts(&target)).unwrap_err(),
            expected
        );
    }
    assert_eq!(expected.code, "training_target_limit");
    assert_eq!(expected.field, "rank");
    assert_eq!(expected.context["maximum"], 128);
}

#[test]
fn malformed_common_and_nested_authored_values_refuse_before_parsing() {
    let target = iris::iris_generation_target();
    for (path, invalid) in [
        (vec!["samplePrompts"], json!(["one", 2])),
        (vec!["requestedGpu"], json!({"device": "auto"})),
        (vec!["sampleCount"], json!(4294967296_u64)),
        (vec!["sampleEvery"], json!("later")),
        (vec!["cacheLatents"], json!("yes")),
        (vec!["iris", "dist", "world", "world_size"], json!(1.5)),
        (vec!["iris", "repa", "weight"], json!("bad")),
        (vec!["iris", "upstream_validation_prompts"], json!([1])),
    ] {
        let mut submit = submission(&target);
        let mut current = &mut submit["config"]["advanced"];
        for key in &path[..path.len() - 1] {
            if !current[*key].is_object() {
                current[*key] = json!({});
            }
            current = &mut current[*key];
        }
        current[path[path.len() - 1]] = invalid;
        let plan = job(&target, submit["config"].clone());
        let expected = training::validate_shape("training", &submit).unwrap_err();
        for kind in ["lora_train", "control_training"] {
            assert_eq!(
                training::validate_shape(kind, &plan).unwrap_err(),
                expected,
                "{path:?}/{kind}"
            );
        }
        assert_eq!(expected.code, "training_field_error");
        assert!(
            expected.field.contains(path.last().unwrap()),
            "{expected:?}"
        );
    }
}

#[test]
fn legacy_coercions_and_null_omission_survive_without_inserting_defaults() {
    let target = training::flux3::target(false);
    let mut submit = submission(&target);
    submit["config"]["advanced"]["cacheLatents"] = json!(" TrUe ");
    submit["config"]["advanced"]["sampleEvery"] = json!(" 0 ");
    submit["config"]["advanced"]["memoryLimitBytes"] = json!(" 4294967296 ");
    submit["config"]["advanced"]["memoryLimitFraction"] = json!(" 0.75 ");
    submit["config"]["advanced"]["weightNoiseSigma"] = Value::Null;
    let original = submit.clone();
    training::validate_shape("training", &submit).unwrap();
    assert_eq!(submit, original);
    let advanced = submit["config"]["advanced"].as_object().unwrap();
    assert!(training::preflight_bool(advanced, "cacheLatents", false).unwrap());
    assert_eq!(
        training::preflight_u32(advanced, "sampleEvery", 99).unwrap(),
        0
    );
    assert_eq!(
        training::preflight_u32(advanced, "omitted", 99).unwrap(),
        99
    );
    assert_eq!(
        training::training_output_scope(json!({}).as_object().unwrap()).unwrap(),
        "project"
    );
    assert!(!submit.contains_key("dryRun"));
}

#[test]
fn active_memory_caps_reject_invalid_values_instead_of_lowering_to_absence() {
    let target = training::flux3::target(false);
    for (key, value) in [
        ("memoryLimitBytes", json!(0)),
        ("memoryLimitBytes", json!(-1)),
        ("memoryLimitBytes", json!("18446744073709551616")),
        ("memoryLimitFraction", json!(1.00000000001_f64)),
        ("memoryLimitFraction", json!(1e-100_f64)),
        ("memoryLimitFraction", json!("NaN")),
    ] {
        let mut submit = submission(&target);
        submit["config"]["advanced"][key] = value;
        let error = training::validate_shape("training", &submit).unwrap_err();
        assert!(error.field.ends_with(key), "{error:?}");
    }
}

#[test]
fn authoritative_target_facts_are_required_and_plan_identity_is_resolved() {
    let target = training::flux3::target(false);
    let plan = job(&target, serde_json::to_value(&target.defaults).unwrap());
    let catalog = json!({"trainingTargets": [target]})
        .as_object()
        .unwrap()
        .clone();
    training::validate("lora_train", &plan, &catalog).unwrap();
    assert!(training::validate("lora_train", &plan, &Default::default()).is_err());
}

#[test]
fn portable_resolution_matches_native_legacy_defaults_and_preserves_authored_snapshot() {
    let mut target = training::flux3::target(false);
    // The target-independent generic mapper receives this legacy shape; source-family validation
    // still owns target-specific omissions and does not seed the snapshot from current defaults.
    target.defaults.advanced.clear();
    let legacy = training::resolve_advanced(&target.defaults.advanced).unwrap();
    assert_eq!(legacy.train_dtype, "bf16");
    assert_eq!(legacy.lr_scheduler, "constant");
    assert_eq!(legacy.decompose_factor, -1);
    assert_eq!(legacy.timestep_type, "sigmoid");
    assert!(!legacy.gradient_checkpointing);
    assert!(legacy.cache_latents && legacy.cache_text_embeddings);
    assert_eq!(legacy.sample_every, 0);
    assert_eq!(legacy.sample_steps, 20);
    assert_eq!(legacy.sample_count, 4);
    assert_eq!(legacy.output_scope, "project");
    assert_eq!(legacy.requested_gpu, "auto");
    let target = iris::iris_generation_target();
    let authored = submission(&target);
    let original = authored.clone();
    let effective = training::resolve_payload("training", &authored, &facts(&target)).unwrap();
    assert_eq!(authored, original);
    assert!(!authored.contains_key("dryRun"));
    assert_eq!(effective["dryRun"], true);
    assert_eq!(effective["config"]["advanced"]["cacheLatents"], true);
    assert_eq!(
        effective["config"]["advanced"]["memoryLimitBytes"],
        Value::Null
    );
    assert_eq!(
        training::moss_timestep_type(&Default::default(), "sigmoid".into()),
        "moss_stage3_v1"
    );
    assert_eq!(
        training::moss_model_options(&Default::default())["vae_posterior"],
        "posterior_mean_v1"
    );
}

#[test]
fn rectangle_carriers_use_one_geometry_parser_and_refuse_ambiguous_authoring() {
    let mut advanced = json!({"aspectRatioBuckets": [{"width":768,"height":1024,"repeats":3}]})
        .as_object()
        .unwrap()
        .clone();
    assert!(matches!(
        training::parse_training_bucket_config(&advanced).unwrap(),
        Some(training::ResolutionBucketConfig::AspectRatio(_))
    ));
    advanced.insert(
        "resolutionBuckets".into(),
        json!([{"resolution":512,"repeats":1}]),
    );
    assert!(training::parse_training_bucket_config(&advanced)
        .unwrap_err()
        .contains("both"));
    advanced.remove("resolutionBuckets");
    advanced["aspectRatioBuckets"][0]["width"] = json!(769);
    assert!(training::parse_training_bucket_config(&advanced).is_err());
}

#[test]
fn iris_fp16_survives_effective_submission_and_worker_plan() {
    let target = iris::iris_generation_target();
    let mut submit = submission(&target);
    submit["config"]["advanced"]["mixedPrecision"] = json!("fp16");
    let plan = job(&target, submit["config"].clone());
    let effective = training::resolve_payload("training", &submit, &facts(&target)).unwrap();
    let worker = training::resolve_payload("lora_train", &plan, &facts(&target)).unwrap();
    assert_eq!(effective["config"]["advanced"]["mixedPrecision"], "fp16");
    assert_eq!(
        worker["plan"]["config"]["advanced"]["mixedPrecision"],
        "fp16"
    );
    assert_eq!(training::train_dtype_for_kernel("other", "fp16"), "f32");
    assert_eq!(
        training::train_dtype_for_kernel(iris::IRIS_TRAINING_KERNEL, " FP16 "),
        "fp16"
    );
}
