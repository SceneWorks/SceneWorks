//! Model-less process smoke. Run only under the separately admitted Rust CPU verification lane.
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn run_witness(root: &Path, plan: &Path) -> Output {
    // File-backed collection avoids a pipe-capacity deadlock for the bounded factor witness.
    let mut stdout = tempfile::tempfile().unwrap();
    let mut stderr = tempfile::tempfile().unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_sceneworks-rust-api"));
    command.env_clear();
    for key in ["PATH", "SystemRoot", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let mut child = command
        .arg("--flux3-training-config-witness")
        .arg(plan)
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("SCENEWORKS_FLUX3_WITNESS_ROOT", root)
        .env("SCENEWORKS_BACKEND_CANDLE_ENABLED", "false")
        .env("SCENEWORKS_BACKEND_MLX_ENABLED", "false")
        .env("SCENEWORKS_RUN_UTILITY_INPROCESS", "0")
        .env("SCENEWORKS_API_HOST", "127.0.0.1")
        .env("SCENEWORKS_API_PORT", "0")
        .stdout(Stdio::from(stdout.try_clone().unwrap()))
        .stderr(Stdio::from(stderr.try_clone().unwrap()))
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill(); // ONLY this owned child; never kill by name.
            let _ = child.wait();
            panic!("CPU witness unexpectedly entered normal app startup");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut output = child.wait_with_output().unwrap();
    assert!(stdout.metadata().unwrap().len() < 1024 * 1024);
    assert!(stderr.metadata().unwrap().len() < 1024 * 1024);
    stdout.seek(SeekFrom::Start(0)).unwrap();
    stderr.seek(SeekFrom::Start(0)).unwrap();
    stdout.read_to_end(&mut output.stdout).unwrap();
    stderr.read_to_end(&mut output.stderr).unwrap();
    assert!(output.stdout.len() < 1024 * 1024);
    assert!(!String::from_utf8_lossy(&output.stdout).contains("api_listening"));
    output
}

#[test]
fn witness_cli_refuses_bad_input_without_starting_normal_app() {
    let temp = tempfile::tempdir().unwrap();
    let plan = temp.path().join("owned-plan.json");
    std::fs::write(&plan, b"{}").unwrap();
    let output = run_witness(temp.path(), &plan);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.len() < 1024);
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["status"], "error");
    assert_eq!(value["code"], "refused");
}

#[test]
fn witness_cli_binds_exact_bytes_without_weights_dataset_or_normal_app() {
    use sceneworks_core::training::flux3::{BASE_ID, BASE_REPO, BASE_REVISION};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().join(BASE_REVISION);
    std::fs::create_dir(&base).unwrap();
    let cfg = json!({
        "format_version":1,"model":"flux-3-image-klein-base","quantization":null,
        "torch_dtype":"bfloat16",
        "dit":{"hidden":3072,"heads":24,"depth":5,"depth_single":28,
            "mlp_ratio":3.0,"axes_dim":[32,32,32,32],"theta":10000,
            "in_channels":{"image":128,"image_cond":128,"text":20480},
            "out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5,"flow_map":false},
        "text_encoder":{"path":"text_encoder","tokenizer":"text_encoder",
            "layers":[4,8,12,16,20,24,28,32],"pad_multiple":80,
            "max_tokens":8192,"dtype":"bfloat16"},
        "vae":{"path":"ae.safetensors","dtype":"float32"},
        "sampler":{"steps":50,"guidance":3.5,"alpha":6.93,"size":[1024,1024]}
    });
    std::fs::write(base.join("config.json"), cfg.to_string()).unwrap();
    let mut config = sceneworks_core::training::flux3::target(false).defaults;
    config.rank = 4;
    config.alpha = 4;
    config.steps = 4;
    config.save_every = 1;
    config.resolution = 512;
    let plan = json!({
        "schemaVersion":1,"planVersion":1,"jobId":"cpu-witness-fixture",
        "target":{"targetId":"flux3_klein_lora","kernel":"flux3_klein_lora",
            "family":"flux3","modality":"image","outputKind":"lora",
            "baseModel":BASE_ID,"baseModelRepo":BASE_REPO,"baseModelPath":base},
        "dataset":{"datasetId":"absent-dataset","datasetVersion":1,
            "rootPath":temp.path().join("absent-dataset"),
            "items":[{"imagePath":temp.path().join("absent-image.png"),
                "caption":"PRIVATE_CAPTION_MUST_NOT_APPEAR_IN_STDOUT"}]},
        "config":config,
        "output":{"loraId":"cpu-fixture","outputDir":temp.path().join("absent-output"),
            "fileName":"lora.safetensors","format":"safetensors","triggerWords":[]},
        "provenance":{"datasetId":"absent-dataset","datasetVersion":1,
            "targetId":"flux3_klein_lora","baseModel":BASE_ID,"configSnapshot":{},
            "outputLoraId":"cpu-fixture","sourceJobId":"cpu-witness-fixture",
            "createdAt":"2026-10-20T00:00:00Z"}
    });
    // Deliberately pretty bytes: CLI must hash FILE BYTES, not reserialized JSON.
    let bytes = serde_json::to_vec_pretty(&plan).unwrap();
    let file = temp.path().join("owned-plan.json");
    std::fs::write(&file, &bytes).unwrap();
    let output = run_witness(temp.path(), &file);
    assert!(!String::from_utf8_lossy(&output.stdout).contains("PRIVATE_CAPTION"));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    if cfg!(any(target_os = "macos", feature = "backend-candle")) {
        assert!(output.status.success());
        assert_eq!(value["status"], "ok");
        let witness = &value["witness"];
        assert_eq!(
            witness["inputSha256"],
            format!("{:x}", Sha256::digest(&bytes))
        );
        assert_eq!(witness["formatVersion"], 1);
        assert!(witness["activeFactorKeys"]
            .as_array()
            .is_some_and(|v| !v.is_empty()));
        assert_eq!(witness["configSha256"].as_str().unwrap().len(), 64);
    } else {
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(value["code"], "unsupported");
    }
    assert_eq!(std::fs::read_dir(&base).unwrap().count(), 1);
    assert!(!temp.path().join("absent-output").exists());
    assert!(!temp.path().join("absent-dataset").exists());
}
