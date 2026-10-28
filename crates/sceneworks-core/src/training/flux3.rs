//! FLUX.3 base adaptation targets. No distilled/quantized training or substituted objectives.
use super::*;

pub const BASE_ID: &str = "flux3_klein_base";
pub const BASE_REPO: &str = "black-forest-labs/flux-3-image-klein-base";
pub const BASE_REVISION: &str = "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4";
pub const KERNELS: [&str; 2] = ["flux3_klein_lora", "flux3_klein_edit_lora"];

pub fn target(edit: bool) -> TrainingTarget {
    let id = KERNELS[usize::from(edit)];
    let name = if edit {
        "FLUX.3 Klein Base Edit LoRA"
    } else {
        "FLUX.3 Klein Base LoRA"
    };
    let mut limits = object(json!({
        "rank": [1, 128], "alpha": [1, 128], "steps": [1, 100000],
        "batchSize": [1, 16], "resolutions": [256, 512, 768, 1024],
        "optimizers": ["adamw"], "networkTypes": ["lora"],
        "gradientCheckpointing": false,
        "timestepTypes": ["uniform", "linear", "sigmoid", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
        "timestepBiases": ["balanced"], "lossTypes": ["mse"], "mixedPrecisions": ["bf16", "f32"],
        "lrSchedulers": ["constant", "linear", "cosine"],
        "outputScopes": ["project", "global"]
    }));
    if edit {
        limits.insert(MAX_REFERENCE_IMAGES_LIMIT.to_owned(), json!(1));
    }
    TrainingTarget {
        id: id.to_owned(),
        name: name.to_owned(),
        modality: TrainingModality::Image,
        output_kind: TrainingOutputKind::Lora,
        family: "flux3".to_owned(),
        base_model: BASE_ID.to_owned(),
        base_model_repo: Some(BASE_REPO.to_owned()),
        kernel: id.to_owned(),
        defaults: TrainingConfig {
            rank: 4,
            alpha: 4.into(),
            learning_rate: ContractNumber::from_f64(0.0001).expect("finite"),
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: 512,
            save_every: 100,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16", "networkType": "lora",
                "gradientCheckpointing": false, "sampleEvery": 0,
                "timestepType": "uniform", "timestepBias": "balanced", "lossType": "mse",
                "weightDecay": 0.0001, "lrScheduler": "constant", "resume": false,
                "outputScope": "project", "requestedGpu": "auto",
                "flux3": {"mode": if edit { "edit" } else { "captioned" }}
            })),
            extra: ExtraFields::new(),
        },
        limits,
        ui: object(json!({
            "label": name,
            "description": "Native Candle/MLX additive LoRA on the undistilled dense FLUX.3 Klein base. Edit pairs require exactly one reference per target. No quantized training, activation checkpointing or in-training previews. Resume unchanged runs from the queue; duplicate starts a separate run. Source JSON is forwarded under flux3 only; unknown keys are refused. Source controls include caption_dropout, max_sequence_length, center_crop, random_flip, weighting_scheme, logit_mean, logit_std, mode_scale, max_grad_norm, lr_scheduler, lr_num_cycles and lr_power.",
            "datasetModality": "image",
            "datasetKind": if edit { "editPairs" } else { "captioned" },
            "recommendedFor": if edit { vec!["edit"] } else { vec!["character", "style"] }
        })),
        extra: ExtraFields::new(),
    }
}

/// Validate host fields before their legacy permissive worker mapping can normalize bad intent.
/// The engine additionally validates the namespaced source options and full checkpoint identity.
pub fn validate(target: &TrainingTarget, config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let fail = |message: &str| TrainingPlanError::InvalidConfig(format!("FLUX.3: {message}"));
    if target.base_model != BASE_ID || target.base_model_repo.as_deref() != Some(BASE_REPO) {
        return Err(fail(
            "adaptation requires the undistilled publisher base identity",
        ));
    }
    for (key, allowed, default) in [
        ("mixedPrecision", &["bf16", "f32"][..], "bf16"),
        ("networkType", &["lora"][..], "lora"),
        ("lossType", &["mse"][..], "mse"),
        (
            "timestepType",
            &[
                "uniform",
                "linear",
                "sigmoid",
                "logit_normal",
                "mode",
                "cosmap",
                "sigma_sqrt",
            ][..],
            "uniform",
        ),
        ("timestepBias", &["balanced"][..], "balanced"),
    ] {
        let value = config
            .advanced
            .get(key)
            .map_or(Some(default), Value::as_str);
        if !value.is_some_and(|value| allowed.contains(&value)) {
            return Err(fail(&format!("unsupported {key}")));
        }
    }
    for key in ["gradientCheckpointing", "resume"] {
        if let Some(value) = config.advanced.get(key) {
            if !value.is_boolean() {
                return Err(fail(&format!("{key} must be boolean")));
            }
        }
    }
    if config
        .advanced
        .get("gradientCheckpointing")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(fail("activation recomputation is not implemented"));
    }
    if config
        .advanced
        .get("sampleEvery")
        .is_some_and(|v| v.as_u64() != Some(0))
    {
        return Err(fail(
            "in-training previews are not implemented; sampleEvery must be 0",
        ));
    }
    if config.seed < 0 {
        return Err(fail("seed must be nonnegative"));
    }
    if let Some(value) = config.advanced.get("loraTargetModules") {
        if !value.as_array().is_some_and(|a| {
            a.iter()
                .all(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()))
        }) {
            return Err(fail("loraTargetModules must contain nonempty strings"));
        }
    }
    if let Some(value) = config.advanced.get("flux3") {
        let bag = value
            .as_object()
            .ok_or_else(|| fail("flux3 must be an object"))?;
        let mode = if target.kernel == KERNELS[1] {
            &["edit", "image_to_image"][..]
        } else {
            &["captioned", "text_to_image"][..]
        };
        for (key, value) in bag {
            let valid = match key.as_str() {
                "mode" => value.as_str().is_some_and(|v| mode.contains(&v)),
                "caption_dropout" => value.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v)),
                "max_sequence_length" => value
                    .as_u64()
                    .is_some_and(|v| v > 0 && v <= i32::MAX as u64),
                "weighting_scheme" => value.as_str().is_some_and(|v| {
                    ["none", "sigma_sqrt", "logit_normal", "mode", "cosmap"].contains(&v)
                }),
                "center_crop" | "random_flip" => value.is_boolean(),
                "logit_mean" | "mode_scale" => value
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && (v as f32).is_finite()),
                "logit_std" | "max_grad_norm" => value
                    .as_f64()
                    .is_some_and(|v| v >= 0.0 && (v as f32).is_finite()),
                "lr_scheduler" => value.as_str().is_some_and(|v| {
                    [
                        "constant",
                        "constant_with_warmup",
                        "linear",
                        "cosine",
                        "cosine_with_restarts",
                        "polynomial",
                    ]
                    .contains(&v)
                }),
                "lr_num_cycles" => value
                    .as_u64()
                    .is_some_and(|v| v > 0 && v <= u32::MAX as u64),
                "lr_power" => value.as_f64().is_some_and(|v| v.is_finite() && v > 0.0),
                "with_prior_preservation" | "use_rslora" | "use_dora" => {
                    value.as_bool() == Some(false)
                }
                "lora_dropout" => value.as_f64() == Some(0.0),
                "rank_pattern" | "alpha_pattern" => value.as_object().is_some_and(|v| v.is_empty()),
                _ => false,
            };
            if !valid {
                return Err(fail(&format!("unsupported flux3.{key}")));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flux3_targets_are_distinct_source_profiles() {
        for edit in [false, true] {
            let target = target(edit);
            validate_training_config_for_target(&target, &target.defaults).unwrap();
            assert_eq!(target.base_model, BASE_ID);
            assert_eq!(training_target_trains_edit_pairs(&target), edit);
            assert_eq!(
                training_target_max_reference_images(&target),
                u32::from(edit)
            );
            assert_eq!(target.defaults.optimizer, "adamw");
        }
    }
    #[test]
    fn flux3_advertised_choices_are_accepted_and_unsupported_choices_refused() {
        for edit in [false, true] {
            let target = target(edit);
            for (limit, key, rejected) in [
                ("timestepTypes", "timestepType", "weighted"),
                ("timestepBiases", "timestepBias", "high_noise"),
                ("lossTypes", "lossType", "mae"),
                ("mixedPrecisions", "mixedPrecision", "fp16"),
            ] {
                for choice in target.limits[limit].as_array().unwrap() {
                    let mut config = target.defaults.clone();
                    config.advanced.insert(key.into(), choice.clone());
                    validate_training_config_for_target(&target, &config).unwrap();
                }
                let mut config = target.defaults.clone();
                config.advanced.insert(key.into(), json!(rejected));
                assert!(
                    validate_training_config_for_target(&target, &config).is_err(),
                    "{key}={rejected}"
                );
            }
        }
    }

    #[test]
    fn flux3_namespace_is_strict_but_other_host_options_are_not_family_options() {
        let target = target(false);
        let mut config = target.defaults.clone();
        config
            .advanced
            .insert("unrelatedProductKey".into(), json!(true));
        validate(&target, &config).unwrap();
        for (key, value) in [
            ("typo", json!(1)),
            ("mode", json!("edit")),
            ("random_flip", json!("true")),
            ("use_dora", json!(true)),
        ] {
            config.advanced.insert("flux3".into(), json!({key: value}));
            assert!(validate(&target, &config).is_err(), "{key}");
        }
    }
}
