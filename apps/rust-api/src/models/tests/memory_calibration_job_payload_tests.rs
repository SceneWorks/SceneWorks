use super::*;

fn model(calibrated: bool) -> Value {
    let calibrations = if calibrated {
        json!([{ "binding": "fixture" }])
    } else {
        json!([])
    };
    json!({
        "id": "fixture-model",
        "name": "Fixture Model",
        "family": "fixture",
        "mlx": {
            "requiresConversion": true,
            "convertSourceRepo": "owner/source",
            "calibrations": calibrations
        }
    })
}

fn download() -> Value {
    json!({
        "provider": "huggingface",
        "repo": "owner/artifact",
        "files": ["weights.safetensors"]
    })
}

#[test]
fn download_payload_derives_provenance_cost_only_from_the_catalog_primary_artifact() {
    let data = tempfile::tempdir().expect("data dir");
    let calibrated = model(true);
    let primary = build_model_download_job_payload(
        &calibrated,
        "fixture-model",
        &download(),
        None,
        true,
        false,
        data.path(),
    )
    .expect("primary payload");
    assert_eq!(
        primary.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
        Some(&Value::Bool(true))
    );

    let co_requisite = build_model_download_job_payload(
        &calibrated,
        "fixture-model",
        &download(),
        None,
        false,
        false,
        data.path(),
    )
    .expect("co-requisite payload");
    assert_eq!(
        co_requisite.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
        Some(&Value::Bool(false)),
        "co-requisites are not the artifact named by the calibration binding"
    );

    let client: ModelDownloadRequest = serde_json::from_value(json!({
        "memoryCalibrationProvenanceRequired": true
    }))
    .expect("typed request ignores unknown spoofed fields");
    let uncalibrated = build_model_download_job_payload(
        &model(false),
        "fixture-model",
        &download(),
        client.variant.as_deref(),
        true,
        false,
        data.path(),
    )
    .expect("uncalibrated payload");
    assert_eq!(
        uncalibrated.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
        Some(&Value::Bool(false)),
        "a client-supplied lookalike cannot enable the cold provenance cost"
    );
}

#[test]
fn convert_payload_derives_provenance_cost_only_from_the_catalog() {
    let data = tempfile::tempdir().expect("data dir");
    let request: ModelConvertRequest = serde_json::from_value(json!({
        "quantizeBits": 4,
        "memoryCalibrationProvenanceRequired": false
    }))
    .expect("typed request ignores unknown spoofed fields");
    let calibrated = model(true);
    let calibrated_payload = build_model_convert_job_payload(ModelConvertJobPayload {
        model: &calibrated,
        model_id: "fixture-model",
        mlx: calibrated["mlx"].as_object().expect("mlx"),
        source_repo: "owner/source",
        output_dir: data.path(),
        quantize_only: false,
        request: &request,
    });
    assert_eq!(
        calibrated_payload.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
        Some(&Value::Bool(true)),
        "a client-supplied lookalike cannot disable catalog-authorized provenance"
    );

    let spoofed: ModelConvertRequest = serde_json::from_value(json!({
        "memoryCalibrationProvenanceRequired": true
    }))
    .expect("typed request ignores unknown spoofed fields");
    let uncalibrated = model(false);
    let uncalibrated_payload = build_model_convert_job_payload(ModelConvertJobPayload {
        model: &uncalibrated,
        model_id: "fixture-model",
        mlx: uncalibrated["mlx"].as_object().expect("mlx"),
        source_repo: "owner/source",
        output_dir: data.path(),
        quantize_only: false,
        request: &spoofed,
    });
    assert_eq!(
        uncalibrated_payload.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
        Some(&Value::Bool(false)),
        "a client-supplied lookalike cannot enable the cold provenance cost"
    );
}
