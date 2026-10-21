//! Iris-3B DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) training through the API: the task
//! targets' submit-time contract (task installs + training-only resources, the paper-recipe defaults,
//! the per-objective refusals), dataset sidecars checked by the engine readers, registration of the
//! trained export as a depth / restoration model with its task identity, its selection in the depth
//! and restoration job paths, resume, and Model Manager delete.
use super::support::*;

const DEPTH_TARGET: &str = "iris_3b_depth";
const RESTORE_TARGET: &str = "iris_3b_restore";
const WEIGHTS_REPO: &str = "speridlabs/iris-3b";
const WEIGHTS_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
const DEPTH_CONFIG: &str = include_str!(
    "../../../../crates/sceneworks-worker/src/iris_restore/fixtures/depth_config.yaml"
);
const UPSCALER_CONFIG: &str = include_str!(
    "../../../../crates/sceneworks-worker/src/iris_restore/fixtures/upscaler_config.yaml"
);

struct Env {
    _hf: HfCacheEnvGuard,
    previous_cap: Option<String>,
}

const MLX_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

fn env(cap_gb: &str) -> Env {
    let hf = isolate_hf_cache();
    let previous_cap = std::env::var(MLX_CAP_ENV).ok();
    std::env::set_var(MLX_CAP_ENV, cap_gb);
    Env {
        _hf: hf,
        previous_cap,
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        match &self.previous_cap {
            Some(previous) => std::env::set_var(MLX_CAP_ENV, previous),
            None => std::env::remove_var(MLX_CAP_ENV),
        }
    }
}

fn snapshot(data_dir: &std::path::Path, repo: &str, revision: &str) -> std::path::PathBuf {
    sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo)
        .expect("repo cache path")
        .join("snapshots")
        .join(revision)
}

/// Stage a task export (`depth/` or `upscaler/`) at the pinned revision: the real release config,
/// placeholder weights.
fn stage_task_export(data_dir: &std::path::Path, subdir: &str, config: &str) {
    let dir = snapshot(data_dir, WEIGHTS_REPO, WEIGHTS_REVISION).join(subdir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.yaml"), config).unwrap();
    let header = br#"{"__metadata__":{"format":"pt"}}"#;
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend_from_slice(header);
    for file in ["model.safetensors", "empty_prompt.safetensors"] {
        std::fs::write(dir.join(file), &weights).unwrap();
    }
}

/// The engine's pinned training resources (`gen_core::iris::restoration_train::TRAINING_RESOURCES`):
/// `(component, repo, revision, file)`.
const TRAINING_RESOURCES: [(&str, &str, &str, &str); 3] = [
    (
        "lpips_vgg16",
        "timm/vgg16.tv_in1k",
        "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
        "model.safetensors",
    ),
    (
        "lpips_lin",
        "vivym/lpips",
        "270571f1fb2a2c4c5f920cec96cd87838c345982",
        "vgg_lpips_linear.safetensors",
    ),
    (
        "critic_trunk",
        "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
        "9f3e8ee3f383c672388d9178afe70af9e63ac9df",
        "open_clip_pytorch_model.bin",
    ),
];

fn stage_training_resources(data_dir: &std::path::Path, except: Option<&str>) {
    for (component, repo, revision, file) in TRAINING_RESOURCES {
        if Some(component) == except {
            continue;
        }
        let dir = snapshot(data_dir, repo, revision);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), b"placeholder").unwrap();
    }
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_fn(width, height, |x, y| image::Rgb([x as u8, y as u8, 7]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}

fn npy(height: u32, width: u32) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({height}, {width}), }}")
            .into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    for y in 0..height {
        for _ in 0..width {
            bytes.extend_from_slice(&(1.0f32 + y as f32).to_le_bytes());
        }
    }
    bytes
}

async fn project(app: &axum::Router) -> String {
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Iris Task Project" }),
    )
    .await;
    project["id"].as_str().expect("project id").to_owned()
}

/// A dataset of `count` real 32×16 PNG images (the sidecar checks decode them).
async fn dataset(app: &axum::Router, project_id: &str, count: usize) -> Value {
    let mut items = Vec::new();
    for index in 0..count {
        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            &format!("item{index}.png"),
            "image/png",
            &png(32, 16),
        )
        .await;
        items.push(json!({ "assetId": asset["id"], "caption": { "text": "" } }));
    }
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Iris task set", "items": items }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    dataset
}

#[allow(clippy::too_many_arguments)]
async fn upload_sidecar(
    app: &axum::Router,
    project_id: &str,
    dataset_id: &str,
    item_id: &str,
    kind: &str,
    filename: &str,
    bytes: &[u8],
    fields: &[(&str, &str)],
) -> (StatusCode, Value) {
    let boundary = "IRIS_SIDECAR_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/items/{item_id}/iris-sidecar/{kind}"
        ),
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn target(app: &axum::Router, id: &str) -> Value {
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    registry["targets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["id"] == id)
        .unwrap_or_else(|| panic!("{id} is served"))
        .clone()
}

async fn submit(
    app: &axum::Router,
    project_id: &str,
    dataset_id: &str,
    target_id: &str,
    config: Value,
) -> (StatusCode, Value) {
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Task Model",
            "dryRun": false
        }),
    )
    .await
}

async fn complete(app: &axum::Router, job_id: &str, status: &str, result: Value) -> Value {
    let (code, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": status, "stage": status, "progress": 1, "message": "done",
            "workerId": TEST_TRAINING_WORKER_ID, "result": result
        }),
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    body
}

/// The small depth config the tests submit (a 64×64 size: the MLX floor is the 16 384-pixel row,
/// inside the emulated 128 GB budget).
fn small_depth_config(target: &Value) -> Value {
    let mut config = target["defaults"].clone();
    config["gradientAccumulation"] = json!(32);
    config["steps"] = json!(64);
    config["saveEvery"] = json!(32);
    config["advanced"]["sampleEvery"] = json!(32);
    config["resolution"] = json!(64);
    config["advanced"]["iris"]["depth_size"] = json!([64, 64]);
    config
}

/// A stored job's payload (after carrier attachment).
async fn stored_payload(app: &axum::Router, job_id: &str) -> Value {
    let (status, job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{job}");
    job["payload"].clone()
}

fn carrier_ids(payload: &Value) -> Vec<String> {
    payload["modelManifestEntries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["id"].as_str().map(str::to_owned))
        .collect()
}

/// Depth: sidecars checked by the engine readers, the install requirement, the per-objective
/// refusals, registration as a depth MODEL with its task identity, selection by id in the depth job
/// (a model of another task refused), resume of the run, and delete removing the run.
#[tokio::test]
async fn depth_training_registers_a_selectable_depth_model() {
    let _env = env("128");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let dataset = dataset(&app, &project_id, 1).await;
    let dataset_id = dataset["id"].as_str().unwrap().to_owned();
    let item_id = dataset["items"][0]["id"].as_str().unwrap().to_owned();

    // ---- sidecars ----------------------------------------------------------------------------
    let (status, error) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "mask",
        "m.png",
        &png(32, 16),
        &[],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a mask needs its depth first: {error}"
    );
    let (status, error) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "depth",
        "d.npy",
        &npy(8, 8),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error.to_string().contains("aligned pixel for pixel"),
        "{error}"
    );
    let (status, updated) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "depth",
        "d.npy",
        &npy(16, 32),
        &[("scale", "0.001"), ("maxDepth", "80")],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{updated}");
    let depth = &updated["items"][0]["irisDepth"];
    assert_eq!(depth["path"], format!("sidecars/{item_id}.depth.npy"));
    assert_eq!(depth["scale"], json!(0.001));
    assert_eq!(depth["maxDepth"], json!(80.0));
    let (status, updated) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "mask",
        "m.png",
        &image_gray_png(32, 16),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{updated}");
    assert_eq!(
        updated["items"][0]["irisDepth"]["maskPath"],
        format!("sidecars/{item_id}.mask.png")
    );

    // ---- install requirement + refusals ---------------------------------------------------------
    let depth_target = target(&app, DEPTH_TARGET).await;
    assert_eq!(depth_target["outputKind"], "base_checkpoint");
    assert_eq!(depth_target["defaults"]["advanced"]["networkType"], "full");
    let config = small_depth_config(&depth_target);
    let (status, error) =
        submit(&app, &project_id, &dataset_id, DEPTH_TARGET, config.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error.to_string().contains("Iris 3B Depth is not installed"),
        "{error}"
    );
    // S8: typed like every other Iris training refusal.
    assert_eq!(error["code"], "iris_training_refused", "{error}");
    assert_eq!(error["context"]["kind"], "missing_export", "{error}");
    stage_task_export(&settings.data_dir, "depth", DEPTH_CONFIG);
    let mut prompts = config.clone();
    prompts["advanced"]["samplePrompts"] = json!(["a cat"]);
    let (status, error) = submit(&app, &project_id, &dataset_id, DEPTH_TARGET, prompts).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    let mut unknown = config.clone();
    unknown["advanced"]["iris"]["text_dropout"] = json!(0.1);
    let (status, _) = submit(&app, &project_id, &dataset_id, DEPTH_TARGET, unknown).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // ---- a run, registered as a depth model -----------------------------------------------------
    let (status, job) = submit(&app, &project_id, &dataset_id, DEPTH_TARGET, config.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(plan["target"]["outputKind"], "base_checkpoint");
    assert_eq!(
        plan["dataset"]["items"][0]["irisDepth"]["maskPath"],
        format!("sidecars/{item_id}.mask.png"),
        "the depth target rides on the plan item"
    );
    let entry = &job["payload"]["manifestEntry"];
    assert_eq!(entry["irisTask"], "depth");
    assert_eq!(entry["type"], "utility");
    assert_eq!(entry["trainedCheckpoint"]["engineModel"], "iris_3b_depth");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    assert!(output_dir.starts_with(settings.data_dir.join("models").join("finetunes")));
    let stem = plan["output"]["fileName"]
        .as_str()
        .unwrap()
        .trim_end_matches(".safetensors")
        .to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    let export = output_dir.join(&stem);
    std::fs::create_dir_all(&export).unwrap();
    std::fs::write(export.join("config.yaml"), DEPTH_CONFIG).unwrap();
    write_test_safetensors(&export.join("model.safetensors"));
    std::fs::write(export.join("empty_prompt.safetensors"), b"prompt").unwrap();
    let checkpoints = output_dir.join(format!("{stem}.checkpoints"));
    std::fs::create_dir_all(checkpoints.join("step_00000002")).unwrap();
    std::fs::write(checkpoints.join("step_00000002/state.json"), b"{}").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(
        completed["result"]["baseCheckpointRegistered"], true,
        "{completed}"
    );
    let model_id = completed["result"]["baseCheckpointModelId"]
        .as_str()
        .unwrap()
        .to_owned();
    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    let model = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the depth export is a catalog model")
        .clone();
    assert_eq!(model["type"], "utility");
    assert_eq!(model["irisTask"], "depth");
    assert_eq!(model["trainedCheckpoint"]["task"], "depth");
    assert_eq!(model["installState"], "installed", "{model:#}");
    assert_eq!(
        model["depth"]["admission"]["measured"], false,
        "the depth engine's admission declaration is overlaid: {model:#}"
    );

    // ---- selected by id in the depth job; a non-depth model is refused ----------------------------
    let (_, source) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "photo.png",
        "image/png",
        &png(64, 48),
    )
    .await;
    let source_id = source["id"].as_str().unwrap().to_owned();
    let (status, depth_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/depth/jobs",
        json!({ "projectId": project_id, "sourceAssetId": source_id, "irisModel": model_id }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{depth_job}");
    assert_eq!(depth_job["payload"]["irisModel"], json!(model_id));
    // S6: the trained run carries ITS OWN entry (with the engine's depth admission overlaid) and
    // never the release `iris_3b_depth` entry it does not load — so a disconnected library holding
    // the release copy cannot refuse it.
    let stored = stored_payload(&app, depth_job["id"].as_str().unwrap()).await;
    assert_eq!(carrier_ids(&stored), vec![model_id.clone()], "{stored:#}");
    assert_eq!(
        stored["modelManifestEntries"][0]["depth"]["admission"]["measured"],
        false
    );
    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/image/depth/jobs",
        json!({ "projectId": project_id, "sourceAssetId": source_id, "irisModel": "sdxl" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert!(
        refused
            .to_string()
            .contains("not a trained Iris depth model"),
        "{refused}"
    );

    // ---- resume: SceneWorks hands the trainer the run's checkpoint root ---------------------------
    let mut resume = config.clone();
    resume["advanced"]["resumeFromJobId"] = json!(job_id);
    resume["advanced"]["iris"]["resume_data_policy"] = json!("new_phase");
    let (status, resumed) = submit(&app, &project_id, &dataset_id, DEPTH_TARGET, resume).await;
    assert_eq!(status, StatusCode::CREATED, "{resumed}");
    let advanced = &resumed["payload"]["plan"]["config"]["advanced"];
    assert_eq!(advanced["resume"], true);
    assert_eq!(
        advanced["iris"]["resume_from"],
        json!(checkpoints.display().to_string())
    );

    // ---- delete removes the export AND its checkpoints ---------------------------------------------
    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/{model_id}?permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    assert!(!output_dir.exists(), "{}", output_dir.display());
}

fn image_gray_png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::GrayImage::from_pixel(width, height, image::Luma([255]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}

/// Restoration: the training-only resources are required (named by their Model Manager ids), the
/// recipe defaults are the paper's full fine-tune, aligned pairs attach as sidecars, a LoRA run
/// registers as a restoration MODEL, and the trained model is selectable in the restore job path
/// (a model of another task refused).
#[tokio::test]
async fn restoration_training_requires_its_resources_and_registers_a_restoration_model() {
    let _env = env("128");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let dataset = dataset(&app, &project_id, 1).await;
    let dataset_id = dataset["id"].as_str().unwrap().to_owned();
    let item_id = dataset["items"][0]["id"].as_str().unwrap().to_owned();

    // A ×2 aligned pair; a non-integer one is refused.
    let (status, error) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "degraded",
        "lq.png",
        &png(10, 16),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    let (status, updated) = upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "degraded",
        "lq.png",
        &png(16, 8),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{updated}");
    assert_eq!(
        updated["items"][0]["irisDegradedImage"],
        format!("sidecars/{item_id}.degraded.png")
    );

    let restore_target = target(&app, RESTORE_TARGET).await;
    let defaults = &restore_target["defaults"];
    assert_eq!(
        defaults["advanced"]["networkType"], "full",
        "the recipe is the default"
    );
    assert_eq!(defaults["learningRate"], json!(1e-5));
    assert_eq!(defaults["resolution"], 1024);
    assert_eq!(
        restore_target["trainingResources"][2]["modelId"],
        "iris_restore_train_critic_trunk"
    );
    let mut config = defaults.clone();
    config["advanced"]["networkType"] = json!("lora");
    config["resolution"] = json!(256);
    config["steps"] = json!(4);
    config["saveEvery"] = json!(2);
    config["advanced"]["sampleEvery"] = json!(2);

    stage_task_export(&settings.data_dir, "upscaler", UPSCALER_CONFIG);
    stage_training_resources(&settings.data_dir, Some("critic_trunk"));
    let (status, error) = submit(
        &app,
        &project_id,
        &dataset_id,
        RESTORE_TARGET,
        config.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error
            .to_string()
            .contains("iris_restore_train_critic_trunk"),
        "{error}"
    );
    // S8: typed.
    assert_eq!(error["code"], "iris_training_refused", "{error}");
    assert_eq!(
        error["context"]["kind"], "missing_training_resources",
        "{error}"
    );
    stage_training_resources(&settings.data_dir, None);
    // Aligned pairs are not synthetically degraded.
    let mut synthetic = config.clone();
    synthetic["advanced"]["iris_restore"]["degradation"] = json!({ "noise_range": [1, 30] });
    let (status, error) = submit(&app, &project_id, &dataset_id, RESTORE_TARGET, synthetic).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    // The step unit.
    let mut odd = config.clone();
    odd["saveEvery"] = json!(3);
    let (status, _) = submit(&app, &project_id, &dataset_id, RESTORE_TARGET, odd).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, job) = submit(&app, &project_id, &dataset_id, RESTORE_TARGET, config).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(
        plan["target"]["outputKind"], "base_checkpoint",
        "a LoRA restoration run still exports a restoration model"
    );
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let stem = plan["output"]["fileName"]
        .as_str()
        .unwrap()
        .trim_end_matches(".safetensors")
        .to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    let export = output_dir.join(&stem);
    std::fs::create_dir_all(&export).unwrap();
    std::fs::write(export.join("config.yaml"), UPSCALER_CONFIG).unwrap();
    write_test_safetensors(&export.join("model.safetensors"));
    std::fs::write(export.join("empty_prompt.safetensors"), b"prompt").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(
        completed["result"]["baseCheckpointRegistered"], true,
        "{completed}"
    );
    let model_id = completed["result"]["baseCheckpointModelId"]
        .as_str()
        .unwrap()
        .to_owned();
    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    let model = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the restoration export is a catalog model")
        .clone();
    assert_eq!(model["irisTask"], "restoration");
    assert_eq!(model["trainedCheckpoint"]["engineModel"], "iris_3b_restore");
    assert_eq!(model["installState"], "installed", "{model:#}");

    let (_, source) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "photo.png",
        "image/png",
        &png(64, 48),
    )
    .await;
    let source_id = source["id"].as_str().unwrap().to_owned();
    let upscale = |model: &str| {
        json!({
            "type": "image_upscale",
            "projectId": project_id,
            "requestedGpu": "auto",
            "payload": {
                "projectId": project_id, "sourceAssetId": source_id, "engine": "iris",
                "scale": 1.0, "irisModel": model
            }
        })
    };
    let (status, job) = request(app.clone(), "POST", "/api/v1/jobs", upscale(&model_id)).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    // S6: a trained restoration run never carries the release `iris_3b_restore` entry.
    let stored = stored_payload(&app, job["id"].as_str().unwrap()).await;
    assert_eq!(carrier_ids(&stored), vec![model_id.clone()], "{stored:#}");
    let (status, refused) = request(app.clone(), "POST", "/api/v1/jobs", upscale("sdxl")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    // Another engine never takes an Iris control.
    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_upscale", "projectId": project_id, "requestedGpu": "auto",
            "payload": { "projectId": project_id, "sourceAssetId": source_id,
                         "engine": "real-esrgan", "factor": 2, "irisModel": model_id }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
}

/// The pre-launch dataset check refuses non-task targets and reports an uninstalled task plainly.
#[tokio::test]
async fn the_dataset_check_reports_per_item_under_the_runs_policy() {
    let _env = env("128");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let dataset = dataset(&app, &project_id, 1).await;
    let dataset_id = dataset["id"].as_str().unwrap().to_owned();
    let item_id = dataset["items"][0]["id"].as_str().unwrap().to_owned();
    let uri = format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/iris-check");
    let depth_target = target(&app, DEPTH_TARGET).await;
    let config = small_depth_config(&depth_target);
    let (status, _) = request(
        app.clone(),
        "POST",
        &uri,
        json!({ "targetId": "iris_3b_generation", "config": config }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // No depth target on the item yet: an error row, never trainable.
    stage_task_export(&settings.data_dir, "depth", DEPTH_CONFIG);
    let (status, report) = request(
        app.clone(),
        "POST",
        &uri,
        json!({ "targetId": DEPTH_TARGET, "config": config }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["items"][0]["status"], "error", "{report:#}");
    assert_eq!(report["trainable"], false);

    upload_sidecar(
        &app,
        &project_id,
        &dataset_id,
        &item_id,
        "depth",
        "d.npy",
        &npy(16, 32),
        &[],
    )
    .await;
    let (_, report) = request(
        app.clone(),
        "POST",
        &uri,
        json!({ "targetId": DEPTH_TARGET, "config": config }),
    )
    .await;
    assert_eq!(report["policy"], "reject", "{report:#}");
    assert_eq!(report["items"][0]["status"], "ok", "{report:#}");
    assert_eq!(report["trainable"], true);
}

/// Every dataset file and every installed (HF snapshot) file under `root`, with its bytes — the
/// sources a training run reads and must never modify (sc-25690).
pub(super) fn source_tree(
    root: &std::path::Path,
) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut files = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .components()
                .any(|part| matches!(part.as_os_str().to_str(), Some("datasets" | "snapshots")))
            {
                files.insert(path.clone(), std::fs::read(&path).unwrap());
            }
        }
    }
    files
}

/// The ids of every catalog model.
pub(super) async fn model_ids(app: &axum::Router) -> Vec<String> {
    let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{models}");
    let mut ids: Vec<String> = models
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["id"].as_str().map(str::to_owned))
        .collect();
    ids.sort();
    ids
}

/// Epic AT6 (sc-25690) — a FAILED export of a depth and of a restoration run. The worst case is
/// staged on purpose: the complete export directory IS on disk at the run's output path (as if the
/// failure came after the bytes were written), and the job is then reported `failed`. Nothing is
/// registered — no result flag, no new catalog model — and the run's sources (its dataset files and
/// sidecars, the installed task export and training resources) are byte-identical. The same export
/// reported `completed` registers, so it is the failure that withholds registration, not the files.
#[tokio::test]
async fn a_failed_task_export_registers_no_model_and_leaves_its_sources_untouched() {
    let _env = env("128");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    stage_task_export(&settings.data_dir, "depth", DEPTH_CONFIG);
    stage_task_export(&settings.data_dir, "upscaler", UPSCALER_CONFIG);
    stage_training_resources(&settings.data_dir, None);

    for (target_id, export_config) in [
        (DEPTH_TARGET, DEPTH_CONFIG),
        (RESTORE_TARGET, UPSCALER_CONFIG),
    ] {
        let dataset = dataset(&app, &project_id, 1).await;
        let dataset_id = dataset["id"].as_str().unwrap().to_owned();
        let item_id = dataset["items"][0]["id"].as_str().unwrap().to_owned();
        let task_target = target(&app, target_id).await;
        let config = if target_id == DEPTH_TARGET {
            let (status, body) = upload_sidecar(
                &app,
                &project_id,
                &dataset_id,
                &item_id,
                "depth",
                "d.npy",
                &npy(16, 32),
                &[],
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{body}");
            small_depth_config(&task_target)
        } else {
            let (status, body) = upload_sidecar(
                &app,
                &project_id,
                &dataset_id,
                &item_id,
                "degraded",
                "lq.png",
                &png(16, 8),
                &[],
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{body}");
            let mut config = task_target["defaults"].clone();
            config["advanced"]["networkType"] = json!("lora");
            config["resolution"] = json!(256);
            config["steps"] = json!(4);
            config["saveEvery"] = json!(2);
            config["advanced"]["sampleEvery"] = json!(2);
            config
        };

        let mut outcomes = Vec::new();
        for status in ["failed", "completed"] {
            let models_before = model_ids(&app).await;
            let (code, job) =
                submit(&app, &project_id, &dataset_id, target_id, config.clone()).await;
            assert_eq!(code, StatusCode::CREATED, "{target_id}: {job}");
            let plan = &job["payload"]["plan"];
            let job_id = job["id"].as_str().unwrap().to_owned();
            let output_dir =
                std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
            let stem = plan["output"]["fileName"]
                .as_str()
                .unwrap()
                .trim_end_matches(".safetensors")
                .to_owned();
            claim_training_job_as_mlx_worker(&app, &job_id).await;
            let export = output_dir.join(&stem);
            std::fs::create_dir_all(&export).unwrap();
            std::fs::write(export.join("config.yaml"), export_config).unwrap();
            write_test_safetensors(&export.join("model.safetensors"));
            std::fs::write(export.join("empty_prompt.safetensors"), b"prompt").unwrap();
            let sources_before = source_tree(temp_dir.path());
            assert!(
                sources_before.keys().any(|path| path.ends_with(format!(
                    "sidecars/{item_id}.{}",
                    if target_id == DEPTH_TARGET {
                        "depth.npy"
                    } else {
                        "degraded.png"
                    }
                ))),
                "{target_id}: the dataset's sidecar is among the sources"
            );
            let reported = complete(&app, &job_id, status, json!({})).await;
            assert_eq!(reported["status"], status, "{target_id}: {reported}");
            assert_eq!(
                source_tree(temp_dir.path()),
                sources_before,
                "{target_id} ({status}): the run's sources are untouched"
            );
            let registered = reported["result"]["baseCheckpointRegistered"] == json!(true);
            let new_models = model_ids(&app)
                .await
                .into_iter()
                .filter(|id| !models_before.contains(id))
                .count();
            outcomes.push((status, registered, new_models, reported));
        }
        let (_, registered, new_models, reported) = &outcomes[0];
        assert!(
            !registered && *new_models == 0,
            "{target_id}: a failed export registers nothing: {reported}"
        );
        assert!(
            reported["result"].get("baseCheckpointModelId").is_none()
                && reported["result"].get("loraRegistered").is_none(),
            "{target_id}: {reported}"
        );
        let (_, registered, new_models, reported) = &outcomes[1];
        assert!(
            *registered && *new_models == 1,
            "{target_id}: the same export, completed, registers: {reported}"
        );
    }
}
