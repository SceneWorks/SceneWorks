//! Pure metadata contracts; separate from dispatch-only test discovery.
use super::moss_training_acceptance::{
    generation_advanced, preflight_verdict, selected_weight_files,
    validate_unpublished_training_outputs,
};

#[test]
fn dmd_regression_payload_omits_cfg_while_ordinary_preview_retains_sealed_guidance() {
    let ordinary = generation_advanced("off").unwrap();
    assert_eq!(ordinary["guidanceScale"], serde_json::json!(1.0));
    for mode in ["unmerged_bf16", "full_fp32"] {
        let dmd = generation_advanced(mode).unwrap();
        assert!(dmd.get("guidanceScale").is_none());
        assert!(dmd.get("negativePrompt").is_none());
        assert_eq!(dmd["steps"], serde_json::json!(8));
        assert_eq!(dmd["mossImage"]["acceleration"], mode);
    }
    assert!(generation_advanced("unknown").is_err());
}
use sceneworks_worker::test_support::gen_core;
use std::collections::BTreeSet;

#[test]
fn manual_cache_startup_uses_production_shared_default_before_isolated_settings() {
    let source = include_str!("moss_training_acceptance.rs");
    let run = source.split_once("async fn run(").unwrap().1;
    assert!(
        run.find("ensure_default_huggingface_home()").unwrap()
            < run.find("test_settings(&temp)").unwrap()
    );
    let shared = std::path::PathBuf::from("SCHEMA_TEST_ONLY/shared-huggingface");
    assert_eq!(
        sceneworks_core::hf_home::default_huggingface_home(None, None, None, Some(shared.clone())),
        Some(shared.clone())
    );
    for overrides in [
        (Some("selected-hub"), None, None),
        (None, Some("selected-legacy-hub"), None),
        (None, None, Some("selected-home")),
    ] {
        assert_eq!(
            sceneworks_core::hf_home::default_huggingface_home(
                overrides.0,
                overrides.1,
                overrides.2,
                Some(shared.clone())
            ),
            None,
            "an explicit approved cache root must not be replaced"
        );
    }
}

#[test]
fn weight_preflight_uses_complete_production_component_map_without_model_execution() {
    for parent in ["base", "sft"] {
        let files = selected_weight_files(parent, false).unwrap();
        let repositories: BTreeSet<_> = files.iter().map(|(repo, _, _)| repo.as_str()).collect();
        println!("{parent}/lora and {parent}/lokr production repository union: {repositories:?}");
        assert!(files
            .iter()
            .all(
                |(repo, rev, file)| gen_core::moss_image::moss_image_asset(repo, rev, file)
                    .is_some()
            ));
        assert!(files
            .iter()
            .any(|(_, _, file)| file.starts_with("text_encoder/")));
        assert!(files.iter().any(|(_, _, file)| file.starts_with("vae/")));
        assert!(files
            .iter()
            .any(|(_, _, file)| file.starts_with("transformer/")));
        if parent == "base" {
            assert!(files
                .iter()
                .all(|(repo, _, _)| repo != gen_core::moss_image::DMD_REPO));
        } else {
            assert!(files
                .iter()
                .any(|(repo, _, _)| repo == gen_core::moss_image::DMD_REPO));
        }
    }
}

#[test]
fn cross_load_selects_only_ordinary_generation_components() {
    for parent in ["base", "sft"] {
        let ordinary = selected_weight_files(parent, true).unwrap();
        let capture = selected_weight_files(parent, false).unwrap();
        assert!(ordinary.is_subset(&capture));
        assert!(ordinary
            .iter()
            .all(|(repo, _, _)| repo != gen_core::moss_image::DMD_REPO));
        for component in ["text_encoder/", "vae/", "transformer/"] {
            assert!(ordinary
                .iter()
                .any(|(_, _, file)| file.starts_with(component)));
        }
        if parent == "sft" {
            assert!(ordinary.len() < capture.len());
        }
    }
}

#[test]
fn cross_load_admits_ordinary_demand_without_training_or_dmd_requirements() {
    let mut metadata = serde_json::json!({"backend":"mlx", "resolvedCapBytes":600});
    let weights = serde_json::json!([{"present":true,"trustedHashMatches":true}]);
    let forecasts =
        serde_json::json!({"off":{"requiredBytes":500,"provenance":"production ordinary headers"}});
    assert_eq!(
        preflight_verdict(&metadata, &weights, "mlx", "sft", &forecasts, true).unwrap(),
        500
    );
    assert!(preflight_verdict(&metadata, &weights, "mlx", "sft", &forecasts, false).is_err());
    metadata["resolvedCapBytes"] = serde_json::json!(499);
    assert!(preflight_verdict(&metadata, &weights, "mlx", "sft", &forecasts, true).is_err());
    metadata["resolvedCapBytes"] = serde_json::json!(600);
    assert!(preflight_verdict(
        &metadata,
        &serde_json::json!([{"present":false}]),
        "mlx",
        "sft",
        &forecasts,
        true
    )
    .is_err());
}

#[test]
fn cross_load_ignores_unused_dmd_and_training_demand_but_capture_still_refuses() {
    let metadata = serde_json::json!({"backend":"mlx", "resolvedCapBytes":600,
        "observationSchema":{"gradientObservations":true,"fixedEvaluationPair":true,"numericFactorDelta":true,"preparedTensorSnapshots":true},
        "nativePlan":{"tensor_peak_bytes":900,"host_staging_peak_bytes":0,"encoder_only_peak_bytes":100,
            "load_bytes":0,"cast_bytes":0,"cache_bytes":0,"backward_bytes":0,"preview_bytes":0,"checkpoint_bytes":0}});
    let weights = serde_json::json!([{"present":true,"trustedHashMatches":true}]);
    let forecasts = serde_json::json!({
        "off":{"requiredBytes":500,"provenance":"production ordinary headers"},
        "unmerged_bf16":{"requiredBytes":700,"provenance":"production BF16 DMD headers"},
        "full_fp32":{"requiredBytes":800,"provenance":"production FP32 DMD headers"}});
    assert_eq!(
        preflight_verdict(&metadata, &weights, "mlx", "sft", &forecasts, true).unwrap(),
        500
    );
    assert!(
        preflight_verdict(&metadata, &weights, "mlx", "sft", &forecasts, false)
            .unwrap_err()
            .contains("demand exceeds")
    );
}

#[tokio::test]
async fn refused_training_allows_only_the_actual_submission_config_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("output");
    let plan = serde_json::json!({"schemaVersion":1,"jobId":"metadata-contract-only","config":{"steps":60},"output":{"fileName":"never-produced.safetensors"}});
    assert!(validate_unpublished_training_outputs(&[], &plan).is_ok());
    crate::training::write_training_config_snapshot(&output, &plan).await;
    let snapshot = output.join("training-config.json");
    assert!(validate_unpublished_training_outputs(std::slice::from_ref(&snapshot), &plan).is_ok());
    assert!(
        validate_unpublished_training_outputs(&[snapshot.clone(), snapshot.clone()], &plan)
            .is_err()
    );
    std::fs::write(&snapshot, b"{}").unwrap();
    assert!(validate_unpublished_training_outputs(std::slice::from_ref(&snapshot), &plan).is_err());
    std::fs::write(&snapshot, b"not json").unwrap();
    assert!(validate_unpublished_training_outputs(std::slice::from_ref(&snapshot), &plan).is_err());
    crate::training::write_training_config_snapshot(&output, &plan).await;
    for name in [
        "adapter.safetensors",
        "checkpoint",
        "preview.png",
        "unknown.json",
    ] {
        let extra = output.join(name);
        std::fs::write(&extra, b"forbidden output").unwrap();
        assert!(
            validate_unpublished_training_outputs(&[snapshot.clone(), extra.clone()], &plan)
                .is_err()
        );
        std::fs::remove_file(extra).unwrap();
    }
    std::fs::remove_file(&snapshot).unwrap();
    std::fs::create_dir(&snapshot).unwrap();
    assert!(validate_unpublished_training_outputs(std::slice::from_ref(&snapshot), &plan).is_err());
    std::fs::remove_dir(&snapshot).unwrap();
    #[cfg(unix)]
    {
        let target = root.path().join("authentic-plan.json");
        std::fs::write(&target, serde_json::to_vec(&plan).unwrap()).unwrap();
        std::os::unix::fs::symlink(&target, &snapshot).unwrap();
        assert!(
            validate_unpublished_training_outputs(std::slice::from_ref(&snapshot), &plan).is_err()
        );
    }
}
