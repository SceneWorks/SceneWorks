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

#[test]
fn recovery_safetensors_compare_all_metadata_descriptors_and_exact_bits() {
    use super::moss_training_acceptance::exact_safetensors_equal;
    use std::path::Path;
    fn write(path: &Path, header: &str, payload: &[u8]) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::write(path, bytes).unwrap();
    }
    let root = tempfile::tempdir().unwrap();
    // Literal host bytes only: neither backend tensor nor model is constructed.
    for role in ["optimizer", "export"] {
        let left = root.path().join(format!("{role}-left.safetensors"));
        let right = root.path().join(format!("{role}-right.safetensors"));
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"__metadata__":{"optimizer":"adamw","optim.step":"60"}}"#;
        let reordered = r#"{"__metadata__":{"optim.step":"60","optimizer":"adamw"},"b":{"shape":[1],"data_offsets":[0,4],"dtype":"F32"},"a":{"shape":[1],"data_offsets":[4,8],"dtype":"F32"}}"#;
        let a = [0x01, 0, 0xc0, 0x7f]; // NaN payload bits, not float equality.
        let b = [0, 0, 0, 0x80]; // -0 must remain distinct from +0.
        write(&left, header, &[a, b].concat());
        write(&right, reordered, &[b, a].concat());
        assert_ne!(
            std::fs::read(&left).unwrap(),
            std::fs::read(&right).unwrap()
        );
        exact_safetensors_equal(&left, &right).unwrap();
        for changed in [
            header.replace("adamw", "adafactor"),
            header.replace(",\"optim.step\":\"60\"", ""),
            header.replace("\"a\":", "\"extra\":"),
            header.replace("\"shape\":[1]", "\"shape\":[1,1]"),
            header.replace("F32", "I32"),
            header.replace(
                ",\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}",
                "",
            ),
            header.replace(
                "\"__metadata__\":{\"optimizer\":\"adamw\",\"optim.step\":\"60\"}",
                "\"__metadata__\":{}",
            ),
        ] {
            write(&right, &changed, &[a, b].concat());
            assert!(
                exact_safetensors_equal(&left, &right).is_err(),
                "accepted {changed}"
            );
        }
        let missing = header.replace(
            ",\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}",
            "",
        );
        write(&right, &missing, &a);
        assert!(exact_safetensors_equal(&left, &right).is_err());
        let extra = header.replace(
            "\"__metadata__\":",
            "\"extra\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]},\"__metadata__\":",
        );
        write(&right, &extra, &[a, b, b].concat());
        assert!(exact_safetensors_equal(&left, &right).is_err());
        let mut changed = [a, b].concat();
        changed[7] ^= 0x80;
        write(&right, header, &changed);
        assert!(exact_safetensors_equal(&left, &right).is_err());
    }
}

#[test]
fn recovery_safetensors_streaming_checks_the_final_payload_chunk() {
    use super::moss_training_acceptance::exact_safetensors_equal;
    let root = tempfile::tempdir().unwrap();
    let left = root.path().join("a");
    let right = root.path().join("b");
    let header = br#"{"factor":{"dtype":"F32","shape":[16385],"data_offsets":[0,65540]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.resize(bytes.len() + 65540, 0);
    std::fs::write(&left, &bytes).unwrap();
    std::fs::write(&right, &bytes).unwrap();
    exact_safetensors_equal(&left, &right).unwrap();
    *bytes.last_mut().unwrap() = 1;
    std::fs::write(&right, &bytes).unwrap();
    assert!(exact_safetensors_equal(&left, &right).is_err());
}

// CI_IMPACT: loopback API, production CPU utility registration/dispatcher and one literal
// safetensors float. No model construction, tensor execution, GPU query or external download.
#[test]
fn adapter_import_uses_cpu_utility_identity_and_preserves_exact_library_bytes() {
    // Match the shipped worker entry and the real-utility-worker test pattern. Construct the
    // handler future on this thread too: its unoptimized platform handlers exceed libtest's stack.
    std::thread::Builder::new()
        .name(sceneworks_worker::WORKER_ENTRY_THREAD_NAME.to_owned())
        .stack_size(sceneworks_worker::WORKER_ENTRY_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("CPU import runtime builds")
                .block_on(adapter_import_cpu_contract());
        })
        .expect("CPU import test thread spawns")
        .join()
        .expect("CPU import test thread joins");
}

async fn adapter_import_cpu_contract() {
    use super::moss_training_acceptance::worker_settings;
    use super::support::*;
    use futures_util::FutureExt;
    use sceneworks_worker::test_support as native;
    use sha2::{Digest, Sha256};

    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let app = create_app(settings.clone()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker = worker_settings(
        format!("http://{}", listener.local_addr().unwrap()),
        &settings,
        "1".into(),
    );
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let served = app.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, served)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let outcome = std::panic::AssertUnwindSafe(async {
        // This synthetic GPU advertisement tests refusal only; it is never native execution proof.
        let (status, gpu) = request(app.clone(), "POST", "/api/v1/workers/register", json!({
            "workerId":worker.worker_id,"gpuId":worker.gpu_id,
            "gpuName":"schema-only GPU claimant, no device execution",
            "capabilities":["gpu","lora_train","lora_train_execute"],"loadedModels":[]
        })).await;
        assert_eq!(status, StatusCode::OK, "{gpu}");
        let (status, project) = request(app.clone(), "POST", "/api/v1/projects", json!({"name":"CPU import contract"})).await;
        assert_eq!(status, StatusCode::CREATED, "{project}");
        let incoming = settings.data_dir.join("loras/incoming");
        std::fs::create_dir_all(&incoming).unwrap();
        let source = incoming.join("literal.safetensors");
        let header = br#"{"__metadata__":{"family":"moss-image","baseModel":"moss_image_base","sceneworks_adapter_format":"moss_transformer_adapter_v1","networkType":"lora","rank":"4","alpha":"6"},"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&source, &bytes).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let (status, job) = request(app.clone(), "POST", "/api/v1/loras/import", json!({
            "sourcePath":source,"expectedSha256":hash,"baseModel":"moss_image_base",
            "scope":"project","projectId":project["id"],"name":"CPU imported adapter"
        })).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["type"], "lora_import");
        let (status, refused) = request(app.clone(), "POST", "/api/v1/jobs/claim", json!({"workerId":worker.worker_id})).await;
        assert_eq!(status, StatusCode::OK, "{refused}");
        assert!(refused["job"].is_null(), "GPU must not claim CPU import: {refused}");
        let (utility, registered) = native::register_acceptance_cpu_utility_worker(&worker).await.unwrap();
        assert_eq!(utility.gpu_id, "cpu");
        assert_ne!(utility.worker_id, worker.worker_id);
        assert_eq!(registered.gpu_id, "cpu");
        assert!(registered.capabilities.contains(&WorkerCapability::Cpu));
        assert!(registered.capabilities.contains(&WorkerCapability::LoraImport));
        assert!(!registered.capabilities.contains(&WorkerCapability::Gpu));
        let (status, claimed) = request(app.clone(), "POST", "/api/v1/jobs/claim", json!({"workerId":utility.worker_id})).await;
        assert_eq!(status, StatusCode::OK, "{claimed}");
        assert_eq!(claimed["job"]["id"], job["id"]);
        assert_eq!(claimed["job"]["workerId"], utility.worker_id);
        assert_eq!(claimed["job"]["assignedGpu"], "cpu");
        native::run_acceptance_utility_job(&utility, serde_json::from_value(claimed["job"].clone()).unwrap()).await;
        let (status, completed) = request(app.clone(), "GET", &format!("/api/v1/jobs/{}",job["id"].as_str().unwrap()), Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{completed}");
        assert_eq!(completed["status"], "completed", "{completed}");
        assert_eq!(completed["workerId"], utility.worker_id);
        assert_eq!(completed["assignedGpu"], "cpu");
        let (status, library) = request(app.clone(), "GET", &format!("/api/v1/loras?projectId={}",project["id"].as_str().unwrap()), Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{library}");
        let items = library.as_array().or_else(|| library["loras"].as_array()).unwrap();
        let entry = items.iter().find(|entry| entry["id"] == job["payload"]["loraId"]).unwrap();
        assert_eq!(entry["family"], "moss-image");
        assert_eq!(entry["baseModel"], "moss_image_base");
        let installed = std::path::PathBuf::from(entry["installedPath"].as_str().unwrap());
        let installed = if installed.is_dir() { installed.join(source.file_name().unwrap()) } else { installed };
        assert_eq!(format!("{:x}", Sha256::digest(std::fs::read(installed).unwrap())), hash);
        assert_eq!(std::fs::read(source).unwrap(), bytes);
        assert_eq!(worker.gpu_id, "1", "utility selection must not replace GPU settings");
    }).catch_unwind();
    let result = tokio::time::timeout(Duration::from_secs(30), outcome).await;
    let _ = stop.send(());
    assert!(server.await.unwrap().is_ok());
    match result.expect("bounded local import contract") {
        Ok(()) => (),
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[test]
fn training_library_resolves_declared_final_file_by_filesystem_identity() {
    use super::moss_training_acceptance::resolve_training_library_adapter;
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("library");
    std::fs::create_dir(&directory).unwrap();
    let output = directory.join("trained.safetensors");
    let checkpoint = directory.join("trained-step020.safetensors");
    let header = br#"{"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1f32.to_le_bytes());
    std::fs::write(&output, &bytes).unwrap();
    std::fs::write(&checkpoint, &bytes).unwrap();
    let expected = std::fs::canonicalize(&output).unwrap();
    let mut entry = serde_json::json!({"installedPath":directory, "files":["trained.safetensors"]});
    assert_eq!(
        resolve_training_library_adapter(&entry, &expected).unwrap(),
        expected
    );
    entry["installedPath"] = serde_json::json!(output);
    assert_eq!(
        resolve_training_library_adapter(&entry, &expected).unwrap(),
        expected
    );
    entry["installedPath"] = serde_json::json!(directory);
    entry["files"] = serde_json::json!(["trained-step020.safetensors"]);
    assert!(
        resolve_training_library_adapter(&entry, &expected).is_err(),
        "equal bytes from another file are not the final adapter"
    );
    entry["installedPath"] = serde_json::json!(root.path().join("missing"));
    assert!(resolve_training_library_adapter(&entry, &expected).is_err());
    #[cfg(unix)]
    {
        let alias = root.path().join("library-alias");
        std::os::unix::fs::symlink(&directory, &alias).unwrap();
        entry = serde_json::json!({"installedPath":alias,"files":["trained.safetensors"]});
        assert_eq!(
            resolve_training_library_adapter(&entry, &expected).unwrap(),
            expected
        );
    }
    #[cfg(windows)]
    {
        let canonical_dir = std::fs::canonicalize(&directory).unwrap();
        let plain = canonical_dir
            .to_str()
            .unwrap()
            .strip_prefix(r"\\?\")
            .unwrap()
            .replace('\\', "/");
        entry = serde_json::json!({"installedPath":plain,"files":["trained.safetensors"]});
        assert_ne!(std::path::Path::new(&plain), expected.parent().unwrap());
        assert_eq!(
            resolve_training_library_adapter(&entry, &expected).unwrap(),
            expected
        );
    }
}

// CI_IMPACT: one bounded CPU-only filesystem/catalog/API dispatcher guard before model admission.
// No loader/tensor/GPU execution, downloads, new workflow or extra campaign command.
#[test]
fn pre_model_library_and_dispatch_contracts() {
    // As in the existing CPU import contract, construct and poll the handler future
    // on the shipped worker stack without depending on ambient RUST_MIN_STACK.
    std::thread::Builder::new()
        .name(sceneworks_worker::WORKER_ENTRY_THREAD_NAME.to_owned())
        .stack_size(sceneworks_worker::WORKER_ENTRY_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("CPU dispatch runtime builds")
                .block_on(pre_model_library_and_dispatch_cpu_contract());
        })
        .expect("CPU dispatch test thread spawns")
        .join()
        .expect("CPU dispatch test thread joins");
}

async fn pre_model_library_and_dispatch_cpu_contract() {
    use super::moss_training_acceptance::{seed_acceptance_manifests, worker_settings};
    use super::support::*;
    use sceneworks_worker::test_support as native;
    training_library_resolves_declared_final_file_by_filesystem_identity();
    let (dispatcher_bytes, wrapper_bytes) = native::acceptance_utility_future_sizes();
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    seed_acceptance_manifests(&settings).unwrap();
    // Literal installed local adapters exercise the shipped catalog and API gate only.
    let mut adapters = Vec::new();
    for parent in ["base", "sft"] {
        for method in ["lora", "lokr"] {
            let id = format!("cpu_{parent}_{method}");
            let relative = format!("loras/{id}.safetensors");
            let path = settings.data_dir.join(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let header = serde_json::to_vec(&json!({"__metadata__":{
                "family":"moss-image","baseModel":format!("moss_image_{parent}"),
                "sceneworks_adapter_format":"moss_transformer_adapter_v1",
                "networkType":method,"rank":"4","alpha":"6"},
                "factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}))
            .unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
            std::fs::write(&path, bytes).unwrap();
            adapters.push(json!({"id":id,"name":id,"family":"moss-image",
                "baseModel":format!("moss_image_{parent}"),"networkType":method,
                "source":{"provider":"local","path":relative}}));
        }
    }
    std::fs::create_dir_all(settings.config_dir.join("manifests")).unwrap();
    std::fs::write(
        settings.config_dir.join("manifests/user.loras.jsonc"),
        serde_json::to_vec(&json!({"schemaVersion":1,"loras":adapters})).unwrap(),
    )
    .unwrap();
    let app = create_app(settings.clone()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let served = app.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, served)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let mut worker = worker_settings(url, &settings, "cpu".into());
    worker.backend_mlx_enabled = false;
    worker.backend_candle_enabled = false;
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"CPU dispatcher refusal"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{project}");
    for adapter in &adapters {
        for matches_parent in [true, false] {
            let parent = adapter["baseModel"].as_str().unwrap();
            let model = if matches_parent {
                parent
            } else if parent == "moss_image_base" {
                "moss_image_sft"
            } else {
                "moss_image_base"
            };
            let (status, submitted) = request(app.clone(), "POST", "/api/v1/image/jobs",
                json!({"projectId":project["id"],"model":model,"prompt":"CPU catalog admission only",
                    "width":256,"height":256,"loras":[{"id":adapter["id"],"weight":1.0}]})).await;
            if matches_parent {
                assert_eq!(status, StatusCode::CREATED, "{submitted}");
                assert_eq!(submitted["status"], "queued", "{submitted}");
                assert_eq!(submitted["payload"]["modelManifestEntry"]["id"], model);
                let id = submitted["id"].as_str().unwrap();
                let (status, canceled) = request(
                    app.clone(),
                    "POST",
                    &format!("/api/v1/jobs/{id}/cancel"),
                    Value::Null,
                )
                .await;
                assert_eq!(status, StatusCode::OK, "{canceled}");
                assert_eq!(canceled["status"], "canceled", "{canceled}");
            } else {
                assert_eq!(status, StatusCode::BAD_REQUEST, "{submitted}");
                assert!(
                    submitted["detail"].as_str().unwrap().contains("MOSS"),
                    "{submitted}"
                );
            }
        }
    }
    // Header validation must still refuse a literal local adapter with truncated data.
    let corrupt = &adapters[0];
    let path = settings
        .data_dir
        .join(corrupt["source"]["path"].as_str().unwrap());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.pop();
    std::fs::write(path, bytes).unwrap();
    let (status, refused) = request(app.clone(), "POST", "/api/v1/image/jobs",
        json!({"projectId":project["id"],"model":corrupt["baseModel"],"prompt":"CPU corrupt header refusal",
            "width":256,"height":256,"loras":[{"id":corrupt["id"],"weight":1.0}]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert!(
        refused["detail"]
            .as_str()
            .unwrap()
            .contains("incomplete or corrupt"),
        "{refused}"
    );
    let source_only = settings.data_dir.join("models/imports/source-only");
    std::fs::create_dir_all(&source_only).unwrap();
    for invalid_image in [false, true] {
        let (status, submitted) = request(app.clone(), "POST", "/api/v1/image/jobs",
            json!({"projectId":project["id"],"model":"moss_image_base","prompt":"metadata-only refusal","width":256,"height":256})).await;
        assert_eq!(status, StatusCode::CREATED, "{submitted}");
        let id = submitted["id"].as_str().unwrap();
        claim_job_as_worker(&app, id, &worker.worker_id, &["gpu", "image_generate"]).await;
        let (status, claimed) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let mut job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(claimed).unwrap();
        // Fault injection into the claimed local snapshot only. No model data
        // exists: the first case stops at the typed source guard; the second
        // admits an empty confined local directory, then hits the image handler's
        // missing-project check before every loader/tensor path.
        job.payload.clear();
        if invalid_image {
            job.payload
                .insert("model".into(), json!("cpu-only-invalid-image"));
            job.payload.insert("modelManifestEntry".into(), json!({"id":"cpu-only-invalid-image",
                "modelPath":source_only,"source":{"provider":"local","path":"models/imports/source-only"}}));
        }
        native::run_acceptance_utility_job(&worker, job).await;
        let (status, failed) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(failed["status"], "failed", "{failed}");
        let error = failed["error"].as_str().unwrap();
        assert!(
            error.contains(if invalid_image {
                "Missing payload.projectId"
            } else {
                "no model manifest entries"
            }),
            "{failed}"
        );
        assert!(
            failed["result"].get("assets").is_none(),
            "refusal published model output"
        );
    }
    let _ = stop.send(());
    assert!(
        server.await.unwrap().is_ok(),
        "owned loopback server did not join"
    );
    assert!(
        wrapper_bytes < dispatcher_bytes,
        "acceptance wrapper must heap-pin the production dispatcher"
    );
}
