//! Real-weight acceptance for Iris-3B generation TRAINING through the production job path (epic
//! sc-25678; S7 sc-25685 on MLX, S8 sc-25686 on Candle/CUDA). Backend-neutral: the same test runs
//! the MLX trainer on Apple Silicon and the Candle trainer on NVIDIA.
//!
//! 1. One bounded LoRA `lora_train` run through `training_jobs::run_training_execution` — the plan
//!    built by the REAL `sceneworks_core` Iris target, the Iris memory floor + data walk, cache-only
//!    resolution of the backbone and its `text_encoder` co-requisite, the trainer, a durable
//!    checkpoint, a preview, and the export. The adapter must carry the Iris identity stamps
//!    (`family = iris`, `baseModel = iris_3b`, `irisTask = generation`, raw weights).
//! 2. Two `image_generate` jobs through `run_image_generate_job` at the same seed — without and with
//!    the trained adapter (as the registered LoRA entry hands it over) — and the adapter must change
//!    the render: the trained artifact is loadable and applied by the generation engine.
//!
//! sc-25690 adds, on the same path: a LoKr run and the official `stage1_256` FULL fine-tune each
//! canceled after a durable checkpoint and RESUMED (the resumed job's own progress must show it
//! continued from the stop, [`resume_continuation`]), and the restoration paper's FULL fine-tune —
//! so the admission gate's full-fine-tune floor rows are read by real jobs, not only by the
//! trainers' own tests.
//!
//! Nothing downloads: `HF_HUB_CACHE` must already hold both pinned snapshots.
//!
//! ```text
//! HF_HUB_CACHE=~/.cache/huggingface/hub \
//!   cargo test -p sceneworks-worker --release iris_training_gpu_smoke -- --ignored --nocapture --test-threads=1
//! # Windows/CUDA: add --features backend-candle
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::*;

const TRAIN_JOB: &str = "iris-train-smoke";
const BACKBONE_REPO: &str = "speridlabs/iris-3b";
const TEXT_ENCODER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const RESOLUTION: u32 = 256;
const TRAIN_STEPS: u32 = 8;
const RENDER_SIZE: u32 = 256;
const RENDER_STEPS: u32 = 20;

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

fn stub_job_snapshot(job_id: &str, job_type: &str, project_id: &str) -> Value {
    json!({
        "id": job_id,
        "type": job_type,
        "status": "running",
        "projectId": project_id,
        "projectName": "Iris training smoke",
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": "0",
        "workerId": "iris-train-smoke-worker",
        "progress": 0.1,
        "stage": "running",
        "message": "running",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-10-11T00:00:00Z",
        "updatedAt": "2026-10-11T00:00:00Z",
        "startedAt": "2026-10-11T00:00:00Z",
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": "2026-10-11T00:00:00Z"
    })
}

async fn stub_job(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
    Json(stub_job_snapshot(&job_id, "lora_train", "smoke"))
}

async fn stub_progress(
    State(posts): State<ProgressPosts>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    posts
        .lock()
        .expect("progress posts lock")
        .push((job_id.clone(), payload));
    Json(stub_job_snapshot(&job_id, "lora_train", "smoke"))
}

async fn stub_heartbeat(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
    Json(json!({
        "id": worker_id, "gpuId": "0", "gpuName": null, "status": "busy",
        "currentJobId": TRAIN_JOB, "capabilities": [], "loadedModels": [],
        "registeredAt": "2026-10-11T00:00:00Z", "lastSeenAt": "2026-10-11T00:00:00Z"
    }))
}

async fn stub_metrics(Json(payload): Json<Value>) -> Json<Value> {
    Json(payload)
}

async fn spawn_worker_api_stub() -> (String, ProgressPosts) {
    let posts: ProgressPosts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(stub_job))
        .route("/api/v1/jobs/:job_id/progress", post(stub_progress))
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("worker API stub listener binds");
    let address = listener.local_addr().expect("worker API stub has address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("worker API stub serves");
    });
    (format!("http://{address}"), posts)
}

fn shipped_iris_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(text)).expect("parse builtin.models.jsonc");
    manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "iris_3b"))
        .cloned()
        .expect("iris_3b is in the builtin catalog")
}

/// A small, learnable dataset: four flat-colour squares with their colour as the caption.
fn write_dataset(root: &Path) -> sceneworks_core::training::TrainingDataset {
    std::fs::create_dir_all(root.join("images")).expect("dataset dir");
    let colours = [
        ("red", [220u8, 30, 30]),
        ("green", [30, 200, 60]),
        ("blue", [40, 60, 220]),
        ("yellow", [230, 210, 40]),
    ];
    let items: Vec<Value> = colours
        .iter()
        .enumerate()
        .map(|(index, (name, rgb))| {
            let path = format!("images/{name}.png");
            let image = image::RgbImage::from_fn(RESOLUTION, RESOLUTION, |x, y| {
                // A soft vignette so the target is not a single constant.
                let edge = ((x.min(RESOLUTION - x)).min(y.min(RESOLUTION - y)) as f32
                    / (RESOLUTION as f32 / 2.0))
                    .min(1.0);
                image::Rgb(rgb.map(|channel| (channel as f32 * (0.6 + 0.4 * edge)) as u8))
            });
            image.save(root.join(&path)).expect("write dataset image");
            json!({
                "id": format!("item{index}"),
                "path": path,
                "displayName": format!("{name}.png"),
                "caption": { "text": format!("a flat {name} square, smoke style"), "source": "manual", "triggerWords": [] },
                "width": RESOLUTION,
                "height": RESOLUTION,
                "addedAt": "2026-10-11T00:00:00Z"
            })
        })
        .collect();
    serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_smoke", "version": 1, "projectId": "smoke",
        "name": "Iris smoke", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z",
        "items": items
    }))
    .expect("smoke dataset")
}

fn safetensors_metadata(path: &Path) -> serde_json::Map<String, Value> {
    let bytes = std::fs::read(path).expect("read adapter");
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + len]).expect("adapter header");
    header["__metadata__"]
        .as_object()
        .cloned()
        .unwrap_or_default()
}

async fn render(
    settings: &Settings,
    posts: &ProgressPosts,
    project: &sceneworks_core::contracts::JsonObject,
    job_id: &str,
    loras: Value,
) -> gen_core::Image {
    render_model(settings, posts, project, job_id, loras, None).await
}

/// [`render`], optionally with a TRAINED full checkpoint `(model id, export dir)` exactly as the
/// API hands one to the worker (`apps/rust-api` `alias_trained_checkpoint`): the payload names the
/// `iris_3b` engine, the engine's manifest entry carries `modelPath` = the export and the
/// `trainedCheckpoint` identity, and `checkpointModelId` records what was picked.
async fn render_model(
    settings: &Settings,
    posts: &ProgressPosts,
    project: &sceneworks_core::contracts::JsonObject,
    job_id: &str,
    loras: Value,
    checkpoint: Option<(&str, &Path)>,
) -> gen_core::Image {
    let mut entry = shipped_iris_entry();
    if let Some((model_id, export)) = checkpoint {
        entry["modelPath"] = json!(export.display().to_string());
        entry["trainedCheckpoint"] = json!({
            "engineModel": "iris_3b", "family": "iris", "task": "generation",
            "baseModel": "iris_3b", "layout": "iris_backbone_directory", "modelId": model_id,
        });
    }
    let mut snapshot = stub_job_snapshot(job_id, "image_generate", "smoke");
    snapshot["projectId"] = project["id"].clone();
    snapshot["payload"] = json!({
        "projectId": project["id"],
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": "a flat red square, smoke style",
        "width": RENDER_SIZE,
        "height": RENDER_SIZE,
        "count": 1,
        "seed": 11,
        "loras": loras,
        "advanced": { "steps": RENDER_STEPS, "guidanceScale": 3.0 },
        "modelManifestEntry": entry,
    });
    if let Some((model_id, _)) = checkpoint {
        snapshot["payload"]["checkpointModelId"] = json!(model_id);
    }
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("render job deserializes");
    let api = ApiClient::new(settings);
    crate::image_jobs::run_image_generate_job(&api, settings, &job)
        .await
        .unwrap_or_else(|error| panic!("the Iris render {job_id} completes: {error}"));
    let result = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == job_id && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the render posted a completed result");
    let media = result["assetWrites"][0]["mediaPath"]
        .as_str()
        .expect("the render wrote one asset")
        .to_owned();
    let decoded = image::open(Path::new(project["path"].as_str().unwrap()).join(media))
        .expect("open render")
        .to_rgb8();
    gen_core::Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris training smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke() {
    run_iris_training_smoke(1).await;
}

/// sc-25689: the same bounded LoRA run as TWO ddp ranks sharing the worker's one device
/// (`dist.world.share_device`) — two real rank processes of this test binary re-entered through the
/// production [`crate::iris_dist::iris_rank_entry`], rendezvousing over loopback TCP with the per-job
/// token, rank 0's events driving the job — then the adapter is applied by a generation job exactly
/// as the single-rank smoke's is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight two-rank Iris training smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke_two_rank_ddp() {
    run_iris_training_smoke(2).await;
}

async fn run_iris_training_smoke(ranks: u64) {
    let hub = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hub.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hub.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let data_dir = data_dir.canonicalize().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(data_dir.clone());
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-train-smoke-worker".to_owned();
    settings.heartbeat_seconds = 30;
    if cfg!(target_os = "macos") {
        settings.gpu_id = "mlx".to_owned();
        settings.backend_mlx_enabled = true;
    } else {
        // The worker's gpu id is the nvidia-smi index the Iris memory floor is read from: the
        // (physical) card CUDA_VISIBLE_DEVICES selects, as the production candle worker is launched.
        settings.gpu_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|devices| devices.split(',').next().map(|id| id.trim().to_owned()))
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        settings.backend_candle_enabled = true;
    }
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    assert!(
        crate::model_jobs::huggingface_snapshot_dir(&data_dir, TEXT_ENCODER_REPO).is_some(),
        "HF_HUB_CACHE must already hold the Qwen3-VL text encoder"
    );

    let project_store = ProjectStore::new(data_dir.clone(), "iris-train-smoke");
    let project = project_store
        .create_project("Iris training smoke")
        .expect("smoke project");
    let project_value = serde_json::to_value(&project).unwrap();
    let project_object = project_value.as_object().unwrap().clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    let output_dir = PathBuf::from(&project.path).join("loras/lora_iris_smoke");

    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let mut config = target.defaults.clone();
    config.rank = 4;
    config.alpha = 4;
    config.steps = TRAIN_STEPS;
    config.save_every = TRAIN_STEPS / 2;
    config.resolution = RESOLUTION;
    config.learning_rate =
        sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
    config.trigger_word = Some("smoke style".to_owned());
    config
        .advanced
        .insert("sampleEvery".into(), json!(TRAIN_STEPS));
    config.advanced.insert("sampleSteps".into(), json!(10));
    config.advanced.insert("sampleCount".into(), json!(1));
    config.advanced.insert(
        "samplePrompts".into(),
        json!(["a flat red square, smoke style"]),
    );
    if ranks > 1 {
        config.advanced.insert(
            "iris".into(),
            json!({
                "dist": {"strategy": "ddp", "world": {"world_size": ranks, "share_device": true}},
                "expected_global_batch": ranks,
            }),
        );
    }
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id: TRAIN_JOB,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "lora_iris_smoke",
            base_model_path: backbone.display().to_string(),
            dataset_root: &dataset_root,
            output_dir: &output_dir,
            file_name: "iris_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the smoke plan builds from the real target");

    let mut snapshot = stub_job_snapshot(TRAIN_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("train job deserializes");
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    if ranks > 1 {
        crate::training_jobs::run_training_execution_drive(
            &api,
            &settings,
            &job,
            &plan,
            |_: &str, _: &gen_core::LoadSpec| -> gen_core::Result<Box<dyn gen_core::Trainer>> {
                panic!("a multi-rank run never loads a trainer in the worker process")
            },
            Arc::new(crate::iris_dist::test_support::TestBinaryRankLauncher),
        )
        .await
        .expect("the two-rank Iris LoRA training job completes");
    } else {
        crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
            .await
            .expect("the Iris LoRA training job completes");
    }
    let train_seconds = started.elapsed().as_secs_f64();

    let result = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == TRAIN_JOB && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the training job posted a completed result");
    assert_eq!(result["stepsCompleted"], TRAIN_STEPS, "{result:#}");
    if ranks > 1 {
        let dist = &result["irisDistributed"];
        assert_eq!(dist["worldSize"], json!(ranks), "{result:#}");
        assert_eq!(dist["shareDevice"], json!(true), "{result:#}");
        assert_eq!(
            result["irisPlan"]["globalBatch"],
            json!(ranks),
            "{result:#}"
        );
    }
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");
    assert!(
        !result["checkpoints"].as_array().unwrap().is_empty(),
        "a durable checkpoint was published: {result:#}"
    );
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a preview rendered: {result:#}"
    );
    let adapter = output_dir.join("iris_smoke.safetensors");
    let metadata = safetensors_metadata(&adapter);
    for (key, expected) in [
        ("family", "iris"),
        ("baseModel", "iris_3b"),
        ("irisTask", "generation"),
        ("irisArtifact", "adapter"),
        ("networkType", "lora"),
        ("irisWeights", "raw"),
    ] {
        assert_eq!(
            metadata.get(key).and_then(Value::as_str),
            Some(expected),
            "adapter metadata {key}: {metadata:#?}"
        );
    }
    assert!(
        output_dir
            .join("iris_smoke.checkpoints")
            .join("latest")
            .is_file(),
        "the checkpoint root carries its latest pointer"
    );

    // The registered adapter, as the LoRA registry hands it to a generation job.
    let base = render(
        &settings,
        &posts,
        &project_object,
        "iris-render-base",
        json!([]),
    )
    .await;
    let tuned = render(
        &settings,
        &posts,
        &project_object,
        "iris-render-lora",
        json!([{
            "id": "lora_iris_smoke",
            "name": "Iris smoke",
            "family": "iris",
            "networkType": "lora",
            "installedPath": adapter.display().to_string(),
            "weight": 1.0
        }]),
    )
    .await;
    assert_eq!((tuned.width, tuned.height), (RENDER_SIZE, RENDER_SIZE));
    let std = crate::smoke_support::image_std(&tuned);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the adapter render looks degenerate (RGB std {std:.2})"
    );
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the trained adapter must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b ranks={ranks} backend={} steps={TRAIN_STEPS} resolution={RESOLUTION} \
         final_loss={final_loss:.4} train_s={train_seconds:.1} adapter={} render_std={std:.2} \
         adapter_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") { "mlx" } else { "candle" },
        adapter.display()
    );
}

// =============================================================================================
// Iris-3B DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) training — the same production path:
// a bounded run through `training_jobs::run_training_execution` from the REAL target, its export
// written where the API registers it (`<data>/models/finetunes/<id>/<stem>/`), then the trained
// model SELECTED by id (`irisModel`) in the depth / restoration job — no file surgery.
// =============================================================================================

fn smoke_settings(temp: &Path, posts_url: String, worker: &str) -> Settings {
    let data_dir = temp.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut settings = Settings::for_test(data_dir.canonicalize().unwrap());
    settings.api_url = posts_url;
    settings.config_dir = temp.join("config");
    settings.worker_id = worker.to_owned();
    settings.heartbeat_seconds = 30;
    if cfg!(target_os = "macos") {
        settings.gpu_id = "mlx".to_owned();
        settings.backend_mlx_enabled = true;
    } else {
        settings.gpu_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|devices| devices.split(',').next().map(|id| id.trim().to_owned()))
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        settings.backend_candle_enabled = true;
    }
    settings
}

fn write_npy_f32(path: &Path, h: u32, w: u32, f: impl Fn(u32, u32) -> f32) {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}").into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&f(y, x).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn task_target(base_model: &str) -> sceneworks_core::training::TrainingTarget {
    sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.base_model == base_model)
        .expect("the Iris task target is registered")
}

fn completed_result(posts: &ProgressPosts, job_id: &str) -> Value {
    posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == job_id && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .unwrap_or_else(|| panic!("{job_id} posted a completed result"))
}

fn import_source(
    store: &ProjectStore,
    project_id: &str,
    temp: &Path,
    image: image::RgbImage,
) -> String {
    let source_file = temp.join("source.png");
    image.save(&source_file).unwrap();
    let asset = store
        .import_asset(
            project_id,
            sceneworks_core::project_store::UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source_file,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("source imports");
    asset["id"].as_str().unwrap().to_owned()
}

/// Two aligned depth examples under `root`: an RGB gradient + a depth ramp (near at the bottom),
/// one with a validity mask.
fn write_depth_dataset(
    root: &Path,
    project_id: &str,
    size: u32,
) -> sceneworks_core::training::TrainingDataset {
    std::fs::create_dir_all(root.join("images")).unwrap();
    std::fs::create_dir_all(root.join("sidecars")).unwrap();
    let items: Vec<Value> = (0..2u32)
        .map(|index| {
            image::RgbImage::from_fn(size, size, |x, y| {
                image::Rgb([(x * 2) as u8, (255 - y * 2) as u8, (60 * index + 40) as u8])
            })
            .save(root.join(format!("images/i{index}.png")))
            .unwrap();
            write_npy_f32(
                &root.join(format!("sidecars/i{index}.depth.npy")),
                size,
                size,
                |y, _| 1.0 + 9.0 * (1.0 - y as f32 / (size - 1) as f32),
            );
            let mut depth =
                json!({ "path": format!("sidecars/i{index}.depth.npy"), "maxDepth": 80.0 });
            if index == 1 {
                image::GrayImage::from_fn(size, size, |x, _| image::Luma([u8::from(x > 4) * 255]))
                    .save(root.join("sidecars/i1.mask.png"))
                    .unwrap();
                depth["maskPath"] = json!("sidecars/i1.mask.png");
            }
            json!({
                "id": format!("i{index}"), "path": format!("images/i{index}.png"),
                "displayName": format!("i{index}.png"),
                "caption": { "text": "", "source": "manual", "triggerWords": [] },
                "addedAt": "2026-10-11T00:00:00Z", "irisDepth": depth
            })
        })
        .collect();
    serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_depth", "version": 1, "projectId": project_id,
        "name": "depth", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z", "items": items
    }))
    .unwrap()
}

/// The bounded depth config of a smoke: the real target's defaults (the paper recipe) at
/// `size`², `steps` optimizer steps, a durable checkpoint every `save_every`.
pub(crate) fn depth_smoke_config(
    size: u32,
    steps: u32,
    save_every: u32,
) -> sceneworks_core::training::TrainingConfig {
    let mut config = task_target("iris_3b_depth").defaults;
    config.steps = steps;
    config.save_every = save_every;
    config.gradient_accumulation = 1;
    config.resolution = size;
    config.learning_rate = sceneworks_core::contracts::ContractNumber::from_f64(1e-4).unwrap();
    config.advanced.insert("sampleEvery".into(), json!(steps));
    config.advanced["iris"]["depth_size"] = json!([size, size]);
    config.advanced["iris"]["preview_items"] = json!(1);
    config.advanced["iris"]["keep_last_checkpoints"] = json!(1);
    config
}

/// A depth job that selects the trained model `model_id` by id (`irisModel`), exactly as the API
/// hands a registered trained depth model over; the raw map must come back whole and finite.
#[allow(clippy::too_many_arguments)]
async fn depth_job_on_trained_model(
    settings: &Settings,
    posts: &ProgressPosts,
    store: &ProjectStore,
    project_id: &str,
    project_path: &str,
    temp: &Path,
    model_id: &str,
    depth_job_id: &str,
) {
    let source = import_source(
        store,
        project_id,
        temp,
        image::RgbImage::from_fn(256, 192, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
        }),
    );
    let mut snapshot = stub_job_snapshot(depth_job_id, "image_depth", project_id);
    let mut trained_entry =
        crate::training_jobs::builtin_model_manifest_entry("iris_3b_depth").unwrap();
    trained_entry["id"] = json!(model_id);
    snapshot["payload"] = json!({
        "projectId": project_id,
        "sourceAssetId": source,
        "resolution": { "mode": "capped", "maxSide": 256 },
        "irisModel": model_id,
        // The API attaches the trained model's own entry (its engine's depth declaration
        // overlaid) in place of the release entry.
        "modelManifestEntries": [trained_entry],
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(settings);
    crate::depth_jobs::run_image_depth_job(&api, settings, &job)
        .await
        .expect("the depth job runs on the trained model");
    let fact = completed_result(posts, depth_job_id)["assetWrites"][0].clone();
    assert_eq!(fact["model"], model_id, "{fact:#}");
    assert_eq!(fact["depth"]["model"]["trainedModelId"], model_id);
    let npy = std::fs::read(PathBuf::from(project_path).join(fact["mediaPath"].as_str().unwrap()))
        .unwrap();
    let map = sceneworks_core::depth_map::decode_npy_f32(&npy).unwrap();
    assert_eq!((map.width, map.height), (256, 192));
    assert!(map.values.iter().all(|v| v.is_finite()));
}

/// sc-25690: a depth full fine-tune UNDER gradient (activation) checkpointing, canceled through the
/// job API after its first durable checkpoint and RESUMED to completion — on real weights. Resume
/// materializes the lazily-loaded trainables inside the first resumed step's checkpointed region,
/// which the engine proves on a miniature only; this is that path on the 3B model. The resumed
/// job's own progress must continue one past the stop ([`resume_continuation`]), the resumed job's
/// plan — as recorded on the job — must still carry the toggle, the lane's trainer must accept
/// it, and the resumed export must drive a depth job.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris DEPTH checkpointed cancel/resume smoke; needs the pinned speridlabs/iris-3b depth/ export under HF_HUB_CACHE and a GPU (~65 GB)"]
async fn iris_training_gpu_smoke_depth_checkpointed_cancel_resume() {
    use sceneworks_core::training::iris::{
        iris_gradient_checkpointing, iris_gradient_checkpointing_lanes, IrisObjective,
    };
    const FIRST_JOB: &str = "iris-depth-ac";
    const RESUME_JOB: &str = "iris-depth-ac-resume";
    const SIZE: u32 = 128;
    const STEPS: u32 = 8;
    const SAVE_EVERY: u32 = 2;
    const FILE: &str = "depth_ac.safetensors";
    let lane = crate::iris_training::training_lane();
    assert!(
        iris_gradient_checkpointing_lanes(IrisObjective::Depth).contains(&lane),
        "the {lane} depth trainer is not listed as implementing gradient checkpointing"
    );
    let cancel: CancelSwitch = Arc::default();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_cancellable_api_stub(cancel.clone()).await;
    let settings = smoke_settings(temp.path(), api_url, "iris-depth-ac-smoke");
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), "iris-depth-ac-smoke");
    let project = store
        .create_project("Iris depth checkpointed cancel/resume smoke")
        .unwrap();
    let root = PathBuf::from(&project.path).join("training/datasets/ds_depth");
    let dataset = write_depth_dataset(&root, &project.id, SIZE);
    let target = task_target("iris_3b_depth");
    let mut config = depth_smoke_config(SIZE, STEPS, SAVE_EVERY);
    config
        .advanced
        .insert("gradientCheckpointing".into(), json!(true));
    let gate = admitted_floor(&settings, "iris_3b_depth", &config).await;
    let model_id = "finetune_depth_ac_smoke";
    let output_dir = data_dir.join("models/finetunes").join(model_id);
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot");
    let build = |job_id: &str, config| {
        sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id,
                target: &target,
                dataset: &dataset,
                config,
                preset: None,
                lora_id: model_id,
                base_model_path: backbone.display().to_string(),
                dataset_root: &root,
                output_dir: &output_dir,
                file_name: FILE.to_owned(),
                created_at: "2026-10-13T00:00:00Z".to_owned(),
            },
        )
        .expect("the checkpointed depth plan builds from the real target")
    };
    let api = ApiClient::new(&settings);

    // The lane's own trainer accepts the toggle (the answer the lane table is held to).
    let answer =
        engine_checkpointing_answer(&settings, &build("iris-depth-ac-probe", config.clone())).await;
    checkpointing_table_verdict("depth", lane, true, &answer)
        .unwrap_or_else(|error| panic!("{error}"));

    // 1. Run under checkpointing; cancel once the first durable checkpoint exists.
    let checkpoint_root = crate::iris_checkpoint_root(&output_dir, FILE);
    let watcher = {
        let latest = checkpoint_root.join("latest");
        let cancel = cancel.clone();
        tokio::spawn(async move {
            while !latest.is_file() {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let plan = build(FIRST_JOB, config.clone());
    let mut snapshot = stub_job_snapshot(FIRST_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution(&api, &settings, &job, &plan).await;
    let first_seconds = started.elapsed().as_secs_f64();
    watcher.abort();
    assert!(
        matches!(outcome, Err(WorkerError::Canceled(_))),
        "the canceled checkpointed depth run ends Canceled: {outcome:?}"
    );
    let canceled = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == FIRST_JOB && body["status"] == "canceled")
        .map(|(_, body)| body.clone())
        .expect("the run posted a terminal canceled update");
    let stopped_at = canceled["result"]["stepsCompleted"]
        .as_u64()
        .unwrap_or_else(|| panic!("a durable cancel reports its step: {canceled:#}"));
    assert!(
        (u64::from(SAVE_EVERY)..u64::from(STEPS)).contains(&stopped_at),
        "canceled after the first checkpoint and before the end: {canceled:#}"
    );
    assert!(checkpoint_root.join("latest").is_file());

    // 2. Resume from that checkpoint — still under checkpointing — to the end.
    cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut plan = build(RESUME_JOB, config);
    apply_resume_translation(&mut plan.config, &checkpoint_root);
    let mut snapshot = stub_job_snapshot(RESUME_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    // The resumed job AS RECORDED still asks for checkpointing: the plan the worker executes.
    let recorded: sceneworks_core::training::TrainingPlan =
        serde_json::from_value(job.payload["plan"].clone()).expect("the recorded plan");
    assert!(
        iris_gradient_checkpointing(&recorded.config),
        "the resumed job's recorded plan lost the gradient checkpointing toggle"
    );
    assert!(recorded.config.advanced["iris"]["resume_from"].is_string());
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .unwrap_or_else(|error| panic!("the resumed checkpointed depth run completes: {error}"));
    let resume_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, RESUME_JOB);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    let resumed_steps = posted_training_steps(&posts.lock().unwrap(), RESUME_JOB);
    let resumed_executed = resume_continuation(&resumed_steps, stopped_at, u64::from(STEPS))
        .unwrap_or_else(|error| panic!("{error}"));
    for checkpoint in result["checkpoints"].as_array().unwrap() {
        assert!(
            checkpoint["step"].as_u64() > Some(stopped_at),
            "a resumed checkpoint is past the stop at {stopped_at}: {result:#}"
        );
    }
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");

    // 3. The resumed export is a depth model a depth job runs on.
    let export = output_dir.join("depth_ac");
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        assert!(export.join(file).is_file(), "the depth export has {file}");
    }
    assert_eq!(
        safetensors_metadata(&export.join("model.safetensors"))
            .get("irisTask")
            .and_then(Value::as_str),
        Some("depth")
    );
    depth_job_on_trained_model(
        &settings,
        &posts,
        &store,
        &project.id,
        &project.path,
        temp.path(),
        model_id,
        "iris-depth-ac-trained-job",
    )
    .await;
    eprintln!(
        "RESULT status=pass trainer=iris_3b_depth checkpointing=on backend={lane} size={SIZE} {gate} \
         engine_accepts=true recorded_plan_checkpointing=true canceled_at_step={stopped_at} \
         first_checkpoint_step={SAVE_EVERY} resumed_first_step={} \
         resumed_steps_executed={resumed_executed} resumed_to={STEPS} first_s={first_seconds:.1} \
         resume_s={resume_seconds:.1} final_loss={final_loss:.4} depth_job_model={model_id}",
        resumed_steps[0],
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris DEPTH training smoke; needs the pinned speridlabs/iris-3b depth/ export under HF_HUB_CACHE and a GPU (~65 GB)"]
async fn iris_training_gpu_smoke_depth() {
    const JOB: &str = "iris-depth-train-smoke";
    const SIZE: u32 = 128;
    const STEPS: u32 = 4;
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let settings = smoke_settings(temp.path(), api_url, "iris-depth-train-smoke");
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), "iris-depth-train-smoke");
    let project = store.create_project("Iris depth training smoke").unwrap();

    let root = PathBuf::from(&project.path).join("training/datasets/ds_depth");
    let dataset = write_depth_dataset(&root, &project.id, SIZE);
    let target = task_target("iris_3b_depth");
    let config = depth_smoke_config(SIZE, STEPS, STEPS / 2);
    let base_config = config.clone();
    let model_id = "finetune_depth_smoke";
    let output_dir = data_dir.join("models/finetunes").join(model_id);
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id: JOB,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: model_id,
            base_model_path: crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
                .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot")
                .display()
                .to_string(),
            dataset_root: &root,
            output_dir: &output_dir,
            file_name: "depth_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the depth plan builds from the real target");
    let mut snapshot = stub_job_snapshot(JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris depth training job completes");
    let train_seconds = started.elapsed().as_secs_f64();
    let checkpointing = checkpointing_capability(
        &settings,
        &posts,
        &project.id,
        &target,
        &dataset,
        &root,
        base_config,
        "depth",
    )
    .await;
    let result = completed_result(&posts, JOB);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a depth preview rendered"
    );
    let export = output_dir.join("depth_smoke");
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        assert!(export.join(file).is_file(), "the depth export has {file}");
    }
    let metadata = safetensors_metadata(&export.join("model.safetensors"));
    assert_eq!(
        metadata.get("irisTask").and_then(Value::as_str),
        Some("depth")
    );

    depth_job_on_trained_model(
        &settings,
        &posts,
        &store,
        &project.id,
        &project.path,
        temp.path(),
        model_id,
        "iris-depth-trained-job",
    )
    .await;
    eprintln!(
        "RESULT status=pass trainer=iris_3b_depth backend={} steps={STEPS} size={SIZE} \
         {checkpointing} final_loss={} train_s={train_seconds:.1} export={} \
         depth_job_model={model_id}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        result["finalLoss"],
        export.display()
    );
}

/// The bounded restoration config of a smoke: the real target's defaults (the paper's recipe) at a
/// 256² center crop, two optimizer steps (= generator, critic, generator, critic); `full` keeps the
/// paper's FULL fine-tune, otherwise the LoRA arm.
pub(crate) fn restore_smoke_config(full: bool) -> sceneworks_core::training::TrainingConfig {
    let mut config = task_target("iris_3b_restore").defaults;
    config.steps = RESTORE_SMOKE_STEPS;
    config.save_every = RESTORE_SMOKE_STEPS;
    config.resolution = RESTORE_SMOKE_CROP;
    if !full {
        config.rank = 8;
        config.alpha = 8;
        config.advanced.insert("networkType".into(), json!("lora"));
    }
    config
        .advanced
        .insert("sampleEvery".into(), json!(RESTORE_SMOKE_STEPS));
    config.advanced["iris_restore"]["crop"] = json!("center");
    config.advanced["iris_restore"]["keep_last_checkpoints"] = json!(1);
    config
}

const RESTORE_SMOKE_CROP: u32 = 256;
// Two steps = generator, critic, generator, critic.
const RESTORE_SMOKE_STEPS: u32 = 4;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris RESTORATION training smoke (LoRA arm); needs the pinned upscaler/ export and the three training resources under HF_HUB_CACHE and a GPU (~30 GB)"]
async fn iris_training_gpu_smoke_restore() {
    run_restore_training_smoke(false).await;
}

/// sc-25690: the paper's FULL fine-tune through the job path at the smallest admitted shape (a
/// 256² crop, batch 1) — the run the restoration `full` floor rows gate — then its export selected
/// by id in a restore job.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris RESTORATION FULL fine-tune smoke; needs the pinned upscaler/ export and the three training resources under HF_HUB_CACHE and ~95 GiB of unified memory or VRAM"]
async fn iris_training_gpu_smoke_full_ft_restore() {
    run_restore_training_smoke(true).await;
}

async fn run_restore_training_smoke(full: bool) {
    const SIZE: u32 = RESTORE_SMOKE_CROP;
    const STEPS: u32 = RESTORE_SMOKE_STEPS;
    let arm = if full { "full" } else { "lora" };
    let job_name = format!("iris-restore-{arm}-train-smoke");
    let job_id = job_name.as_str();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let settings = smoke_settings(temp.path(), api_url, job_id);
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), job_id);
    let project = store
        .create_project("Iris restoration training smoke")
        .unwrap();

    // Clean images: synthetic (seeded) second-order degradation.
    let root = PathBuf::from(&project.path).join("training/datasets/ds_restore");
    std::fs::create_dir_all(root.join("images")).unwrap();
    let items: Vec<Value> = (0..2u32)
        .map(|index| {
            image::RgbImage::from_fn(SIZE, SIZE, |x, y| {
                image::Rgb([
                    ((x / 8 + y / 8 + index) % 2 * 200) as u8,
                    (x % 256) as u8,
                    (y % 256) as u8,
                ])
            })
            .save(root.join(format!("images/c{index}.png")))
            .unwrap();
            json!({
                "id": format!("c{index}"), "path": format!("images/c{index}.png"),
                "displayName": format!("c{index}.png"),
                "caption": { "text": "", "source": "manual", "triggerWords": [] },
                "addedAt": "2026-10-11T00:00:00Z"
            })
        })
        .collect();
    let dataset: sceneworks_core::training::TrainingDataset = serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_restore", "version": 1, "projectId": project.id,
        "name": "restore", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z", "items": items
    }))
    .unwrap();

    let target = task_target("iris_3b_restore");
    let config = restore_smoke_config(full);
    let gate = admitted_floor(&settings, "iris_3b_restore", &config).await;
    let base_config = config.clone();
    let model_name = format!("finetune_restore_{arm}_smoke");
    let model_id = model_name.as_str();
    let output_dir = data_dir.join("models/finetunes").join(model_id);
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: model_id,
            base_model_path: crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
                .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot")
                .display()
                .to_string(),
            dataset_root: &root,
            output_dir: &output_dir,
            file_name: "restore_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the restoration plan builds from the real target");
    let mut snapshot = stub_job_snapshot(job_id, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris restoration training job completes (training resources verified)");
    let train_seconds = started.elapsed().as_secs_f64();
    // The paper's FULL arm carries the checkpointing question (its `full_ac` rows); the LoRA arm
    // reads the adapter rows either way.
    let checkpointing = if full {
        checkpointing_capability(
            &settings,
            &posts,
            &project.id,
            &target,
            &dataset,
            &root,
            base_config,
            "restore",
        )
        .await
    } else {
        String::new()
    };
    let result = completed_result(&posts, job_id);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a restoration preview"
    );
    let export = output_dir.join("restore_smoke");
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        assert!(
            export.join(file).is_file(),
            "the upscaler export has {file}"
        );
    }
    let metadata = safetensors_metadata(&export.join("model.safetensors"));
    assert_eq!(
        metadata.get("irisTask").and_then(Value::as_str),
        Some("restoration")
    );

    // The trained model, selected by id in an Iris restore job (a 1× restore).
    let source = import_source(
        &store,
        &project.id,
        temp.path(),
        image::RgbImage::from_fn(128, 96, |x, y| {
            image::Rgb([
                (x * 2) as u8,
                (y * 2) as u8,
                ((x / 8 + y / 8) % 2 * 200) as u8,
            ])
        }),
    );
    let restore_job_id = "iris-restore-trained-job";
    let mut snapshot = stub_job_snapshot(restore_job_id, "image_upscale", &project.id);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "sourceAssetId": source,
        "engine": "iris",
        "scale": 1.0,
        "irisModel": model_id,
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    crate::upscale_jobs::run_image_upscale_job(&api, &settings, &job)
        .await
        .expect("the restore job runs on the trained model");
    let asset = completed_result(&posts, restore_job_id)["assetWrites"][0].clone();
    let restored =
        image::open(PathBuf::from(&project.path).join(asset["mediaPath"].as_str().unwrap()))
            .unwrap()
            .to_rgb8();
    assert_eq!(restored.dimensions(), (128, 96));
    eprintln!(
        "RESULT status=pass trainer=iris_3b_restore arm={arm} backend={} steps={STEPS} crop={SIZE} \
         {gate} {checkpointing} final_loss={} train_s={train_seconds:.1} export={} \
         restore_job_model={model_id}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        result["finalLoss"],
        export.display()
    );
}

// =============================================================================================
// sc-25690 (epic 25678 terminal evidence): a LoKr generation run CANCELED through the job API after
// its first durable checkpoint, then RESUMED from that checkpoint to completion — the production
// path end to end (`run_training_execution`, the in-loop cancel poller, the trainer's durable
// cancel, the API's `resumeFromJobId` → `iris.resume_from` translation reproduced exactly as
// `apps/rust-api` `resolve_iris_run_sources` writes it), then the LoKr adapter applied by a
// generation job.
// =============================================================================================

type CancelSwitch = Arc<std::sync::atomic::AtomicBool>;
type CancellableState = (ProgressPosts, CancelSwitch);

fn cancellable_snapshot(job_id: &str, cancel: &CancelSwitch) -> Value {
    let mut job = stub_job_snapshot(job_id, "lora_train", "smoke");
    job["cancelRequested"] = json!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    job
}

async fn cancellable_job(
    State((_, cancel)): State<CancellableState>,
    AxumPath(job_id): AxumPath<String>,
) -> Json<Value> {
    Json(cancellable_snapshot(&job_id, &cancel))
}

async fn cancellable_progress(
    State((posts, cancel)): State<CancellableState>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    posts.lock().unwrap().push((job_id.clone(), payload));
    Json(cancellable_snapshot(&job_id, &cancel))
}

/// The worker API stub with a user cancel: every job snapshot (the cancel poller's GET and each
/// progress-post reply) reports `cancelRequested` from `cancel`.
async fn spawn_cancellable_api_stub(cancel: CancelSwitch) -> (String, ProgressPosts) {
    let posts: ProgressPosts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(cancellable_job))
        .route("/api/v1/jobs/:job_id/progress", post(cancellable_progress))
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state((posts.clone(), cancel));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), posts)
}

/// What THIS lane's own trainer says to gradient (activation) checkpointing for `plan`'s task: the
/// production load (`preflight_training_run` → `TrainerLoadInputs` → `load_trainer`) and the
/// trainer's own `validate`, with the engine flag forced on. `Err` carries the trainer's refusal.
async fn engine_checkpointing_answer(
    settings: &Settings,
    plan: &sceneworks_core::training::TrainingPlan,
) -> Result<(), String> {
    let (settings, plan) = (settings.clone(), plan.clone());
    tokio::task::spawn_blocking(move || {
        let crate::training_jobs::PreparedTrainingRun {
            engine_id,
            mut request,
            prepared_inputs,
        } = crate::training_jobs::preflight_training_run(&settings, &plan, false)
            .expect("the probe plan preflights");
        request.config.gradient_checkpointing = true;
        // A task trainer's load inputs, exactly as the job path resolves them (its task export and
        // staged components; `TrainerLoadInputs::resolve` is the generation path's).
        let objective =
            crate::iris_task_training::task_objective(&plan.target.kernel, &plan.target.base_model)
                .expect("a depth / restoration plan");
        let inputs =
            crate::iris_task_training::task_load_inputs(&settings, objective, &plan.config)
                .expect("the task's weights and resources resolve");
        let spec = crate::training_jobs::TrainerLoadInputs::from_task(
            engine_id,
            inputs.weights,
            inputs.components,
        )
        .load_spec(engine_id, &request.config.train_dtype);
        let answer = crate::inference_runtime::load_trainer(engine_id, &spec)
            .and_then(|trainer| trainer.validate(&request))
            .map_err(|error| error.to_string());
        drop(request);
        crate::training_jobs::finish_prepared_input_operation(Ok(()), prepared_inputs)
            .expect("the probe's prepared inputs close");
        answer
    })
    .await
    .expect("the engine probe thread joins")
}

/// The verdict on the per-lane table `iris_gradient_checkpointing_lanes` against what the lane's
/// trainer actually answered. The engine publishes no declaration of this capability, so the table
/// is explicit — and this is what makes a drift loud: a lane whose trainer STARTS accepting it
/// while the table still says no fails here, naming what to do.
fn checkpointing_table_verdict(
    task: &str,
    lane: &str,
    listed: bool,
    answer: &Result<(), String>,
) -> Result<(), String> {
    match (answer, listed) {
        (Ok(()), true) => Ok(()),
        (Err(refusal), false) if refusal.contains("checkpointing") => Ok(()),
        (Ok(()), false) => Err(format!(
            "the {lane} Iris {task} trainer now ACCEPTS gradient (activation) checkpointing, but \
             `iris_gradient_checkpointing_lanes` does not list {lane}: add the lane there, measure \
             its `full_ac` floor rows and add them to the {task} floor table"
        )),
        (Err(refusal), true) => Err(format!(
            "`iris_gradient_checkpointing_lanes` lists {lane} for {task}, but its trainer refuses \
             gradient checkpointing: {refusal}"
        )),
        (Err(other), false) => Err(format!(
            "the {lane} Iris {task} trainer failed for a reason other than checkpointing, so its \
             answer is unknown: {other}"
        )),
    }
}

/// sc-25690: the depth / restoration gradient-checkpointing capability on this lane, on real
/// weights. (1) Ask the lane's own trainer and hold `iris_gradient_checkpointing_lanes` to its
/// answer ([`checkpointing_table_verdict`]). (2) On a lane that implements it, run a short job
/// through the production path WITH the toggle on — admitted at the lane's `full_ac` row — and
/// require it to complete; on a lane that does not, require the job path to refuse the toggle
/// before load. Returns the evidence for the smoke's RESULT line.
#[allow(clippy::too_many_arguments)]
async fn checkpointing_capability(
    settings: &Settings,
    posts: &ProgressPosts,
    project_id: &str,
    target: &sceneworks_core::training::TrainingTarget,
    dataset: &sceneworks_core::training::TrainingDataset,
    dataset_root: &Path,
    mut config: sceneworks_core::training::TrainingConfig,
    task: &str,
) -> String {
    use sceneworks_core::training::iris::{iris_gradient_checkpointing_lanes, IrisObjective};
    const STEPS: u32 = 2;
    let objective =
        IrisObjective::from_base_model(&target.base_model).expect("an Iris task target");
    let lane = crate::iris_training::training_lane();
    let listed = iris_gradient_checkpointing_lanes(objective).contains(&lane);
    config.steps = STEPS;
    config.save_every = STEPS;
    config.advanced.insert("sampleEvery".into(), json!(STEPS));
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot");
    let build = |suffix: &str, config| {
        let id = format!("finetune_{task}_{suffix}");
        let job = format!("iris-{task}-{suffix}");
        let output_dir = settings.data_dir.join("models/finetunes").join(&id);
        let plan = sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id: &job,
                target,
                dataset,
                config,
                preset: None,
                lora_id: &id,
                base_model_path: backbone.display().to_string(),
                dataset_root,
                output_dir: &output_dir,
                file_name: format!("{task}_{suffix}.safetensors"),
                created_at: "2026-10-13T00:00:00Z".to_owned(),
            },
        )
        .expect("the checkpointing plan builds from the real target");
        (job, plan)
    };

    // 1. The lane's own trainer, asked directly.
    let (_, probe) = build("ac_probe", config.clone());
    let answer = engine_checkpointing_answer(settings, &probe).await;
    checkpointing_table_verdict(task, lane, listed, &answer)
        .unwrap_or_else(|error| panic!("{error}"));

    // 2. The product path with the toggle on.
    let mut on = config;
    on.advanced
        .insert("gradientCheckpointing".into(), json!(true));
    let (job_id, plan) = build("ac_run", on.clone());
    let mut snapshot = stub_job_snapshot(&job_id, "lora_train", project_id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(settings);
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution(&api, settings, &job, &plan).await;
    if listed {
        let gate = admitted_floor(settings, &target.base_model, &on)
            .await
            .replace("gate_", "checkpointed_gate_");
        outcome.unwrap_or_else(|error| panic!("the checkpointed {task} job completes: {error}"));
        let result = completed_result(posts, &job_id);
        assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
        format!(
            "checkpointing_lane={lane} engine_accepts=true checkpointed_run=pass \
             checkpointed_steps={STEPS} checkpointed_s={:.1} {gate}",
            started.elapsed().as_secs_f64()
        )
    } else {
        match outcome {
            Err(WorkerError::InvalidPayload(message))
                if message.contains("checkpointing is not implemented") =>
            {
                format!(
                    "checkpointing_lane={lane} engine_accepts=false checkpointed_run=refused_before_load"
                )
            }
            other => panic!(
                "on {lane} the {task} job path must refuse the gradient checkpointing toggle \
                 before load, got {other:?}"
            ),
        }
    }
}

/// The floor row the admission gate reads for `config` on this lane, and the capacity it is judged
/// against — printed in a smoke's RESULT line so the evidence names the row a real job ran under.
/// Panics when the gate would refuse: the job below would be refused before load.
async fn admitted_floor(
    settings: &Settings,
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
) -> String {
    use sceneworks_core::training::iris::{curricula, iris_objective_memory_floor, IrisObjective};
    let lane = crate::iris_training::training_lane();
    #[cfg(target_os = "macos")]
    let capacity = {
        let _ = settings;
        crate::iris_training::mlx_training_capacity_gb()
    };
    #[cfg(not(target_os = "macos"))]
    let capacity = crate::iris_training::candle_training_capacity_gb(&settings.gpu_id)
        .await
        .ok();
    let objective = IrisObjective::from_base_model(base_model).expect("an Iris base model");
    let floor = match objective {
        IrisObjective::Generation => curricula::iris_run_memory_floor(lane, config, 1),
        _ => iris_objective_memory_floor(objective, lane, config),
    }
    .expect("a floor row covers the smoke's shape");
    let refusal =
        crate::iris_training::iris_plan_memory_refusal(lane, base_model, config, 1, capacity);
    assert_eq!(refusal, None, "the gate admits the smoke on this host");
    format!(
        "gate_floor_gib={} gate_floor_provenance={} gate_capacity_gib={}",
        floor.gib,
        floor.provenance.as_str(),
        capacity.map_or_else(|| "unread".to_owned(), |gib| format!("{gib:.2}"))
    )
}

/// The micro-steps a job's Training progress posts reported, in post order. The worker posts one
/// update per trainer `Training` event (`map_training_progress`: "Training step N of T (loss …)."),
/// and the Iris trainers emit that event only after an optimizer step completes.
fn posted_training_steps(posts: &[(String, Value)], job_id: &str) -> Vec<u64> {
    posts
        .iter()
        .filter(|(id, _)| id == job_id)
        .filter_map(|(_, body)| {
            body["message"]
                .as_str()?
                .strip_prefix("Training step ")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .collect()
}

/// sc-25690: what a RESUMED run's own progress must show to have CONTINUED from the canceled run's
/// durable stop at `stopped_at` rather than restarted. `stepsCompleted == total` alone cannot tell
/// the two apart — a run that ignored `iris.resume_from` and trained 1..=total also ends at
/// `total`. A continuation trains exactly `stopped_at + 1 ..= total`: its first step is
/// `stopped_at + 1`, it never reports a step at or under the stop, and it executes
/// `total − stopped_at` steps. Returns the executed step count.
fn resume_continuation(resumed_steps: &[u64], stopped_at: u64, total: u64) -> Result<u64, String> {
    let Some(first) = resumed_steps.first() else {
        return Err("the resumed run reported no training step".to_owned());
    };
    if let Some(repeated) = resumed_steps.iter().find(|step| **step <= stopped_at) {
        return Err(format!(
            "the resumed run trained step {repeated}, at or under the canceled run's stop at \
             {stopped_at}: it restarted instead of continuing (steps {resumed_steps:?})"
        ));
    }
    let expected: Vec<u64> = (stopped_at + 1..=total).collect();
    if resumed_steps != expected {
        return Err(format!(
            "the resumed run must train exactly steps {}..={total} (first {first}); it reported \
             {resumed_steps:?}",
            stopped_at + 1
        ));
    }
    Ok(expected.len() as u64)
}

/// The two generation runs the cancel → resume smoke drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelResumeArm {
    /// A LoKr adapter (256², AdamW).
    LoKr,
    /// The official `stage1_256` FULL fine-tune (REPA against the DINOv2 teacher, Muon), bounded.
    FullStage1,
}

const CANCEL_RESUME_STEPS: u32 = 8;

impl CancelResumeArm {
    fn label(self) -> &'static str {
        match self {
            Self::LoKr => "lokr",
            Self::FullStage1 => "full",
        }
    }

    /// The first durable checkpoint lands here; the cancel is requested once it has.
    fn save_every(self) -> u32 {
        match self {
            Self::LoKr => 2,
            // A full-model checkpoint is tens of GB: one before the cancel, one at the end.
            Self::FullStage1 => 4,
        }
    }

    /// The run's config, from the REAL target: the LoKr arm from its defaults; the full arm from
    /// the engine's own `stage1_256` preset as the Studio offers it
    /// (`iris_curricula::iris_stage_presets`), edited only for the bound — one device, batch 1 (so
    /// no multi-node mesh and no 1024 global batch), no EMA copy, eight steps with no warmup, one
    /// retained checkpoint, the smoke's preview prompt. The stage identity, Muon, the flow shift,
    /// the RoPE law and REPA are the preset's.
    pub(crate) fn config(self) -> sceneworks_core::training::TrainingConfig {
        let mut config = match self {
            Self::LoKr => {
                let mut config = sceneworks_core::training::iris::iris_generation_target().defaults;
                config.rank = 4;
                config.alpha = 4;
                config.advanced.insert("networkType".into(), json!("lokr"));
                config.advanced.insert("decomposeFactor".into(), json!(4));
                config
            }
            Self::FullStage1 => {
                let preset = crate::iris_curricula::iris_stage_presets()
                    .expect("the engine's stage presets translate")
                    .into_iter()
                    .find(|preset| preset["id"] == "stage1_256")
                    .expect("stage1_256 is an official stage");
                let mut config: sceneworks_core::training::TrainingConfig =
                    serde_json::from_value(preset["config"].clone())
                        .expect("a stage preset is a training config");
                config.batch_size = 1;
                config.gradient_accumulation = 1;
                config.advanced.insert("lrWarmupSteps".into(), json!(0));
                let options = config.advanced["iris"]
                    .as_object_mut()
                    .expect("a stage preset carries the iris options");
                for key in ["dist", "expected_global_batch"] {
                    options.remove(key);
                }
                options.insert("ema_enabled".into(), json!(false));
                options.insert("keep_last_checkpoints".into(), json!(1));
                options.insert("upstream_validation_prompts".into(), json!(false));
                config
            }
        };
        config.steps = CANCEL_RESUME_STEPS;
        config.save_every = self.save_every();
        config.resolution = RESOLUTION;
        config.learning_rate =
            sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
        config.trigger_word = Some("smoke style".to_owned());
        config
            .advanced
            .insert("sampleEvery".into(), json!(CANCEL_RESUME_STEPS));
        config.advanced.insert("sampleSteps".into(), json!(10));
        config.advanced.insert("sampleCount".into(), json!(1));
        config.advanced.insert(
            "samplePrompts".into(),
            json!(["a flat red square, smoke style"]),
        );
        config
    }
}

/// The resumed run's plan config, exactly as `apps/rust-api` `resolve_iris_run_sources` writes the
/// `resumeFromJobId` translation: `advanced.iris.resume_from` = the previous run's checkpoint root
/// and `advanced.resume` = true.
pub(crate) fn apply_resume_translation(
    config: &mut sceneworks_core::training::TrainingConfig,
    checkpoint_root: &Path,
) {
    let mut options = match config.advanced.remove("iris") {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    options.insert(
        "resume_from".to_owned(),
        json!(checkpoint_root.display().to_string()),
    );
    config
        .advanced
        .insert("iris".to_owned(), Value::Object(options));
    config.advanced.insert("resume".to_owned(), json!(true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris LoKr cancel/resume smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke_lokr_cancel_resume() {
    run_cancel_resume_smoke(CancelResumeArm::LoKr).await;
}

/// sc-25690: the generation FULL fine-tune through the job path — the official `stage1_256`
/// (REPA, Muon), admitted by the gate at its full-fine-tune floor row, canceled after its first
/// durable checkpoint, resumed to the end, exported, and the exported checkpoint rendered by a
/// generation job exactly as a registered trained model is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris generation FULL fine-tune cancel/resume smoke (stage1_256 + REPA); needs the pinned speridlabs/iris-3b, Qwen3-VL-4B-Instruct and facebook/dinov2-base snapshots under HF_HUB_CACHE and ~90 GiB of unified memory or VRAM"]
async fn iris_training_gpu_smoke_full_ft_generation_cancel_resume() {
    run_cancel_resume_smoke(CancelResumeArm::FullStage1).await;
}

async fn run_cancel_resume_smoke(arm: CancelResumeArm) {
    const STEPS: u32 = CANCEL_RESUME_STEPS;
    let label = arm.label();
    let full = arm == CancelResumeArm::FullStage1;
    let first_job = format!("iris-train-{label}");
    let resume_job = format!("iris-train-{label}-resume");
    let file = format!("iris_{label}.safetensors");
    let output_id = if full {
        "finetune_iris_full_smoke".to_owned()
    } else {
        "lora_iris_lokr".to_owned()
    };
    let cancel: CancelSwitch = Arc::default();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_cancellable_api_stub(cancel.clone()).await;
    let settings = smoke_settings(temp.path(), api_url, &format!("iris-{label}-smoke-worker"));
    let data_dir = settings.data_dir.clone();
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    let store = ProjectStore::new(data_dir.clone(), format!("iris-{label}-smoke"));
    let project = store.create_project("Iris cancel/resume smoke").unwrap();
    let project_object = serde_json::to_value(&project)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    // An adapter is written beside its project; a full run where the API registers a trained
    // model from (`<data>/models/finetunes/<id>`).
    let output_dir = if full {
        data_dir.join("models/finetunes").join(&output_id)
    } else {
        PathBuf::from(&project.path).join("loras").join(&output_id)
    };

    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let config = arm.config();
    let gate = admitted_floor(&settings, "iris_3b", &config).await;
    let build = |job_id: &str, config| {
        sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id,
                target: &target,
                dataset: &dataset,
                config,
                preset: None,
                lora_id: &output_id,
                base_model_path: backbone.display().to_string(),
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                file_name: file.clone(),
                created_at: "2026-10-12T00:00:00Z".to_owned(),
            },
        )
        .expect("the plan builds from the real target")
    };
    let api = ApiClient::new(&settings);

    // 1. Run, and request a cancel the moment the first durable checkpoint lands.
    let checkpoint_root = crate::iris_checkpoint_root(&output_dir, &file);
    let watcher = {
        let latest = checkpoint_root.join("latest");
        let cancel = cancel.clone();
        tokio::spawn(async move {
            while !latest.is_file() {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let plan = build(&first_job, config.clone());
    let mut snapshot = stub_job_snapshot(&first_job, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution(&api, &settings, &job, &plan).await;
    let first_seconds = started.elapsed().as_secs_f64();
    watcher.abort();
    assert!(
        matches!(outcome, Err(WorkerError::Canceled(_))),
        "the canceled {label} run ends Canceled: {outcome:?}"
    );
    let canceled = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| *id == first_job && body["status"] == "canceled")
        .map(|(_, body)| body.clone())
        .expect("the run posted a terminal canceled update");
    let stopped_at = canceled["result"]["stepsCompleted"]
        .as_u64()
        .unwrap_or_else(|| panic!("a durable cancel reports its step: {canceled:#}"));
    assert!(
        (u64::from(arm.save_every())..u64::from(STEPS)).contains(&stopped_at),
        "canceled after the first checkpoint and before the end: {canceled:#}"
    );
    assert!(
        checkpoint_root.join("latest").is_file(),
        "the canceled run kept its durable checkpoint"
    );

    // 2. Resume from that run's checkpoint (the API's `resumeFromJobId` translation) to the end.
    cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut plan = build(&resume_job, config);
    apply_resume_translation(&mut plan.config, &checkpoint_root);
    let mut snapshot = stub_job_snapshot(&resume_job, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .unwrap_or_else(|error| panic!("the resumed {label} run completes: {error}"));
    let resume_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, &resume_job);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    // The resumed job CONTINUED: its own progress starts one past the stop and never repeats a
    // step the canceled run already trained. (`stepsCompleted` alone is also what a restart from
    // step 0 reports.)
    let resumed_steps = posted_training_steps(&posts.lock().unwrap(), &resume_job);
    let resumed_executed = resume_continuation(&resumed_steps, stopped_at, u64::from(STEPS))
        .unwrap_or_else(|error| panic!("{error}"));
    let resumed_first = resumed_steps[0];
    for checkpoint in result["checkpoints"].as_array().unwrap() {
        assert!(
            checkpoint["step"].as_u64() > Some(stopped_at),
            "a resumed checkpoint is past the stop at {stopped_at}: {result:#}"
        );
    }
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");

    // 3. The resumed artifact is applied by a generation job.
    let base = render(
        &settings,
        &posts,
        &project_object,
        &format!("iris-{label}-base"),
        json!([]),
    )
    .await;
    let tuned = if full {
        // The full export, where the API registers it from, selected as a trained checkpoint.
        assert_eq!(
            result["irisPlan"]["stage"],
            json!("stage1_256"),
            "{result:#}"
        );
        assert!(
            result["irisPlan"]["repa"].is_object(),
            "the stage's REPA alignment ran: {result:#}"
        );
        let export = output_dir.join(format!("iris_{label}"));
        for name in ["config.yaml", "model.safetensors"] {
            assert!(
                export.join(name).is_file(),
                "the full export has {name}: {}",
                export.display()
            );
        }
        let metadata = safetensors_metadata(&export.join("model.safetensors"));
        assert_eq!(
            metadata.get("irisTask").and_then(Value::as_str),
            Some("generation"),
            "export metadata: {metadata:#?}"
        );
        render_model(
            &settings,
            &posts,
            &project_object,
            "iris-full-tuned",
            json!([]),
            Some((&output_id, &export)),
        )
        .await
    } else {
        let adapter = output_dir.join(&file);
        let metadata = safetensors_metadata(&adapter);
        for (key, expected) in [
            ("family", "iris"),
            ("baseModel", "iris_3b"),
            ("irisTask", "generation"),
            ("networkType", "lokr"),
        ] {
            assert_eq!(
                metadata.get(key).and_then(Value::as_str),
                Some(expected),
                "adapter metadata {key}: {metadata:#?}"
            );
        }
        render(
            &settings,
            &posts,
            &project_object,
            "iris-lokr-tuned",
            json!([{
                "id": output_id,
                "name": "Iris LoKr smoke",
                "family": "iris",
                "networkType": "lokr",
                "installedPath": adapter.display().to_string(),
                "weight": 1.0
            }]),
        )
        .await
    };
    let std = crate::smoke_support::image_std(&tuned);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the {label} render looks degenerate (RGB std {std:.2})"
    );
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the resumed {label} artifact must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b network={label} backend={} {gate} \
         canceled_at_step={stopped_at} first_checkpoint_step={} resumed_first_step={resumed_first} \
         resumed_steps_executed={resumed_executed} resumed_to={STEPS} first_s={first_seconds:.1} \
         resume_s={resume_seconds:.1} final_loss={final_loss:.4} render_std={std:.2} \
         artifact_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        arm.save_every(),
    );
}

// =============================================================================================
// sc-25690 (epic acceptance 5): REAL two-GPU distributed training through the job path — two rank
// processes on two DIFFERENT physical cards (`dist.world.share_device` off, a worker that owns
// both cards: `SCENEWORKS_GPU_ID=0,1` + `CUDA_VISIBLE_DEVICES=0,1`). Which card a rank ran on is
// read from the run itself — each rank process records its own pid, and `nvidia-smi` says which
// card's context that pid held — never from the plan the rank was handed.
//
// Deliberately NOT named `iris_training_gpu_smoke_*`: the default CUDA training filter runs with
// one visible card, where this smoke cannot run. Dispatch it by name with both cards exposed.
// =============================================================================================

/// The `(pid, gpu uuid)` pairs of
/// `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader`.
fn parse_compute_apps(csv: &str) -> Vec<(u64, String)> {
    csv.lines()
        .filter_map(|line| {
            let (pid, uuid) = line.split_once(',')?;
            let uuid = uuid.trim();
            if uuid.is_empty() {
                return None;
            }
            Some((pid.trim().parse().ok()?, uuid.to_owned()))
        })
        .collect()
}

/// The `(index, uuid, used MiB)` rows of
/// `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits`.
fn parse_gpu_memory(csv: &str) -> Vec<(u32, String, u64)> {
    csv.lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let index = fields.next()?.parse().ok()?;
            let uuid = fields.next().filter(|uuid| !uuid.is_empty())?.to_owned();
            let used = fields.next()?.parse().ok()?;
            Some((index, uuid, used))
        })
        .collect()
}

/// What `nvidia-smi` showed while a fleet ran: every `(pid, card)` context seen, and each card's
/// used memory before the run and at its highest.
#[derive(Debug, Default, Clone)]
struct CardObservations {
    contexts: std::collections::BTreeSet<(u64, String)>,
    baseline_mib: std::collections::BTreeMap<String, u64>,
    peak_mib: std::collections::BTreeMap<String, u64>,
    index: std::collections::BTreeMap<String, u32>,
}

impl CardObservations {
    fn record_memory(&mut self, rows: &[(u32, String, u64)]) {
        for (index, uuid, used) in rows {
            self.index.insert(uuid.clone(), *index);
            self.baseline_mib.entry(uuid.clone()).or_insert(*used);
            let peak = self.peak_mib.entry(uuid.clone()).or_insert(*used);
            *peak = (*peak).max(*used);
        }
    }
}

/// The card each rank ran on, from the run's own records: `rank_pids` is what the rank processes
/// recorded about themselves, `contexts` what the driver reported. Every one of the `world` ranks
/// must have held a context on exactly ONE card, and no two ranks on the same card.
fn ranks_on_distinct_cards(
    rank_pids: &[(u64, u64)],
    contexts: &std::collections::BTreeSet<(u64, String)>,
    world: u64,
) -> Result<Vec<(u64, String)>, String> {
    let mut placed = Vec::new();
    for rank in 0..world {
        let pids: Vec<u64> = rank_pids
            .iter()
            .filter(|(recorded, _)| *recorded == rank)
            .map(|(_, pid)| *pid)
            .collect();
        let [pid] = pids[..] else {
            return Err(format!(
                "rank {rank} of {world} recorded {} processes (records {rank_pids:?})",
                pids.len()
            ));
        };
        let cards: Vec<&String> = contexts
            .iter()
            .filter(|(context_pid, _)| *context_pid == pid)
            .map(|(_, uuid)| uuid)
            .collect();
        match cards[..] {
            [card] => placed.push((rank, card.clone())),
            [] => {
                return Err(format!(
                    "rank {rank} (pid {pid}) never held a GPU context (contexts {contexts:?})"
                ))
            }
            _ => {
                return Err(format!(
                    "rank {rank} (pid {pid}) held contexts on several cards: {cards:?}"
                ))
            }
        }
    }
    for (index, (rank, card)) in placed.iter().enumerate() {
        if let Some((other, _)) = placed[..index].iter().find(|(_, seen)| seen == card) {
            return Err(format!(
                "ranks {other} and {rank} ran on the SAME card {card}: the run was not distributed \
                 over two devices"
            ));
        }
    }
    Ok(placed)
}

/// Every card in `cards` gained at least `min_mib` of used memory over its pre-run baseline — the
/// second, pid-independent sign that both cards actually held a rank.
fn cards_gained_memory(
    observed: &CardObservations,
    cards: &[String],
    min_mib: u64,
) -> Result<Vec<(String, u64)>, String> {
    cards
        .iter()
        .map(|card| {
            let gained = observed
                .peak_mib
                .get(card)
                .zip(observed.baseline_mib.get(card))
                .map(|(peak, baseline)| peak.saturating_sub(*baseline))
                .ok_or_else(|| format!("card {card} was never sampled"))?;
            if gained < min_mib {
                return Err(format!(
                    "card {card} gained only {gained} MiB (< {min_mib} MiB): no rank trained on it"
                ));
            }
            Ok((card.clone(), gained))
        })
        .collect()
}

fn nvidia_smi(args: &[&str]) -> String {
    std::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

fn sample_cards(observed: &Mutex<CardObservations>) {
    let contexts = parse_compute_apps(&nvidia_smi(&[
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader",
    ]));
    let memory = parse_gpu_memory(&nvidia_smi(&[
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]));
    let mut observed = observed.lock().unwrap();
    observed.contexts.extend(contexts);
    observed.record_memory(&memory);
}

/// Sample `nvidia-smi` once a second on a thread of its own until the returned flag is set.
fn watch_cards() -> (
    Arc<Mutex<CardObservations>>,
    Arc<std::sync::atomic::AtomicBool>,
    std::thread::JoinHandle<()>,
) {
    let observed: Arc<Mutex<CardObservations>> = Arc::default();
    sample_cards(&observed); // the baseline, before any rank exists
    let stop: Arc<std::sync::atomic::AtomicBool> = Arc::default();
    let handle = {
        let (observed, stop) = (observed.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                sample_cards(&observed);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        })
    };
    (observed, stop, handle)
}

fn kill_process(pid: u64) {
    #[cfg(unix)]
    {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
    }
}

/// The two-card smoke's config: the LoRA smoke's bounded run as two ranks, one per card —
/// `share_device` OFF, so a rank whose card is missing is refused rather than doubled up — with
/// the global batch stated (`batch 1 × 2 ranks × accumulation 1`), which the engine checks.
pub(crate) fn two_card_config(strategy: &str) -> sceneworks_core::training::TrainingConfig {
    let mut config = sceneworks_core::training::iris::iris_generation_target().defaults;
    config.rank = 4;
    config.alpha = 4;
    config.steps = TWO_CARD_STEPS;
    config.save_every = 2;
    config.resolution = RESOLUTION;
    config.learning_rate =
        sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
    config.trigger_word = Some("smoke style".to_owned());
    config
        .advanced
        .insert("sampleEvery".into(), json!(TWO_CARD_STEPS));
    config.advanced.insert("sampleSteps".into(), json!(10));
    config.advanced.insert("sampleCount".into(), json!(1));
    config.advanced.insert(
        "samplePrompts".into(),
        json!(["a flat red square, smoke style"]),
    );
    let options = config.advanced["iris"]
        .as_object_mut()
        .expect("the Iris defaults carry the options bag");
    options.insert(
        "dist".into(),
        json!({"strategy": strategy, "world": {"world_size": TWO_CARD_WORLD, "share_device": false}}),
    );
    options.insert("expected_global_batch".into(), json!(TWO_CARD_WORLD));
    config
}

const TWO_CARD_WORLD: u64 = 2;
const TWO_CARD_STEPS: u32 = 8;
/// A LoRA rank at 256² holds ~20 GiB of VRAM; a card that gained less than this held no rank.
const TWO_CARD_MIN_GAIN_MIB: u64 = 4096;

struct TwoCardRun {
    outcome: crate::error::WorkerResult<()>,
    rank_pids: Vec<(u64, u64)>,
    observed: CardObservations,
    seconds: f64,
}

/// What a two-card fleet does mid-run.
#[derive(Clone, Copy, PartialEq)]
enum MidRun {
    Nothing,
    /// Request a user cancel once the first durable checkpoint exists.
    CancelAfterCheckpoint,
    /// Kill rank 1's process once training has started.
    KillRankOne,
}

#[allow(clippy::too_many_arguments)]
async fn drive_two_cards(
    settings: &Settings,
    posts: &ProgressPosts,
    cancel: &CancelSwitch,
    project_id: &str,
    job_id: &str,
    plan: &sceneworks_core::training::TrainingPlan,
    checkpoint_root: &Path,
    mid_run: MidRun,
) -> TwoCardRun {
    let records = tempfile::tempdir().unwrap();
    let (observed, stop, sampler) = watch_cards();
    let actor = {
        let (posts, cancel) = (posts.clone(), cancel.clone());
        let (latest, records, job_id) = (
            checkpoint_root.join("latest"),
            records.path().to_path_buf(),
            job_id.to_owned(),
        );
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                match mid_run {
                    MidRun::Nothing => return,
                    MidRun::CancelAfterCheckpoint if latest.is_file() => {
                        cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                        return;
                    }
                    MidRun::KillRankOne => {
                        let training =
                            !posted_training_steps(&posts.lock().unwrap(), &job_id).is_empty();
                        let rank_one = crate::iris_dist::test_support::records(&records)
                            .into_iter()
                            .find(|record| record["rank"] == 1)
                            .and_then(|record| record["pid"].as_u64());
                        if let (true, Some(pid)) = (training, rank_one) {
                            kill_process(pid);
                            return;
                        }
                    }
                    MidRun::CancelAfterCheckpoint => {}
                }
            }
        })
    };
    let mut snapshot = stub_job_snapshot(job_id, "lora_train", project_id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(settings);
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution_drive(
        &api,
        settings,
        &job,
        plan,
        |_: &str, _: &gen_core::LoadSpec| -> gen_core::Result<Box<dyn gen_core::Trainer>> {
            panic!("a multi-rank run never loads a trainer in the worker process")
        },
        Arc::new(crate::iris_dist::test_support::RecordingRankLauncher {
            record_dir: records.path().to_path_buf(),
        }),
    )
    .await;
    let seconds = started.elapsed().as_secs_f64();
    actor.abort();
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    sampler.join().unwrap();
    let rank_pids = crate::iris_dist::test_support::records(records.path())
        .iter()
        .filter_map(|record| Some((record["rank"].as_u64()?, record["pid"].as_u64()?)))
        .collect();
    let observed = observed.lock().unwrap().clone();
    TwoCardRun {
        outcome,
        rank_pids,
        observed,
        seconds,
    }
}

/// The cards the ranks of `run` trained on (rank order), proven two ways: by pid → context, and by
/// each of those cards' memory rising while the fleet ran.
fn two_card_placement(run: &TwoCardRun, leg: &str) -> Vec<(u64, String)> {
    let placed = ranks_on_distinct_cards(&run.rank_pids, &run.observed.contexts, TWO_CARD_WORLD)
        .unwrap_or_else(|error| panic!("{leg}: {error}"));
    let cards: Vec<String> = placed.iter().map(|(_, card)| card.clone()).collect();
    let gained = cards_gained_memory(&run.observed, &cards, TWO_CARD_MIN_GAIN_MIB)
        .unwrap_or_else(|error| panic!("{leg}: {error}"));
    eprintln!(
        "TWO_CARD leg={leg} {}",
        placed
            .iter()
            .zip(&gained)
            .map(|((rank, card), (_, mib))| format!(
                "rank{rank}=pid:{},card_index:{},uuid:{card},gained_mib:{mib}",
                run.rank_pids
                    .iter()
                    .find(|(recorded, _)| recorded == rank)
                    .map_or(0, |(_, pid)| *pid),
                run.observed.index.get(card).copied().unwrap_or(u32::MAX),
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );
    placed
}

fn assert_two_card_receipt(result: &Value, strategy: &str) {
    let dist = &result["irisDistributed"];
    assert_eq!(dist["worldSize"], json!(TWO_CARD_WORLD), "{result:#}");
    assert_eq!(dist["shareDevice"], json!(false), "{result:#}");
    assert_eq!(dist["rankProcesses"], json!(TWO_CARD_WORLD), "{result:#}");
    assert_eq!(dist["strategy"], json!(strategy), "{result:#}");
    assert_eq!(dist["globalBatch"], json!(TWO_CARD_WORLD), "{result:#}");
    // Global-batch accounting: batch 1 × 2 ranks × accumulation 1, as the engine resolved it.
    assert_eq!(
        result["irisPlan"]["globalBatch"],
        json!(TWO_CARD_WORLD),
        "{result:#}"
    );
}

/// Epic acceptance 5 on real hardware, in four legs through `run_training_execution_drive`:
///
/// 1. DDP (replication) on two cards, canceled after its first durable checkpoint;
/// 2. the same run RESUMED from that checkpoint to the end — the resumed job's own progress must
///    continue one past the stop ([`resume_continuation`]) — and its adapter applied by a
///    generation job;
/// 3. a rank FAILURE: rank 1's process is killed mid-run; the job fails typed, naming the rank, and
///    no rank process is left holding a card;
/// 4. FSDP (sharding) on two cards to completion — LAST, with its own RESULT line, so the first
///    three legs' evidence is printed whatever it does.
///
/// In every leg that trains, each rank ran on a different physical card
/// ([`ranks_on_distinct_cards`], [`cards_gained_memory`]) and the global batch is 2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris TWO-CARD distributed smoke; needs two free NVIDIA cards (CUDA_VISIBLE_DEVICES=0,1, CUDA_DEVICE_ORDER=PCI_BUS_ID) and the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE"]
async fn iris_two_card_distributed_smoke() {
    const STEPS: u32 = TWO_CARD_STEPS;
    let visible: Vec<String> = std::env::var("CUDA_VISIBLE_DEVICES")
        .unwrap_or_default()
        .split(',')
        .map(|id| id.trim().to_owned())
        .filter(|id| !id.is_empty())
        .collect();
    assert!(
        !cfg!(target_os = "macos") && visible.len() >= TWO_CARD_WORLD as usize,
        "the two-card smoke needs two NVIDIA cards exposed through CUDA_VISIBLE_DEVICES (got \
         {visible:?}); it never shares a device"
    );
    let cancel: CancelSwitch = Arc::default();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_cancellable_api_stub(cancel.clone()).await;
    let mut settings = smoke_settings(temp.path(), api_url, "iris-two-card-smoke-worker");
    // The worker owns BOTH cards (the production multi-GPU worker: SCENEWORKS_GPU_ID=0,1).
    settings.gpu_id = visible[..TWO_CARD_WORLD as usize].join(",");
    let data_dir = settings.data_dir.clone();
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    let store = ProjectStore::new(data_dir.clone(), "iris-two-card-smoke");
    let project = store.create_project("Iris two-card smoke").unwrap();
    let project_object = serde_json::to_value(&project)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let build = |job_id: &str, output_id: &str, config| {
        let output_dir = PathBuf::from(&project.path).join("loras").join(output_id);
        let plan = sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id,
                target: &target,
                dataset: &dataset,
                config,
                preset: None,
                lora_id: output_id,
                base_model_path: backbone.display().to_string(),
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                file_name: "iris_two_card.safetensors".to_owned(),
                created_at: "2026-10-12T00:00:00Z".to_owned(),
            },
        )
        .expect("the two-card plan builds from the real target");
        let root = crate::iris_checkpoint_root(&output_dir, "iris_two_card.safetensors");
        (plan, output_dir, root)
    };
    let ddp = two_card_config("ddp");
    let gate = admitted_two_cards(&settings, &ddp).await;

    // 1. DDP on two cards, canceled after the first durable checkpoint.
    let (plan, output_dir, checkpoint_root) = build("iris-two-card", "lora_two_card", ddp.clone());
    let first = drive_two_cards(
        &settings,
        &posts,
        &cancel,
        &project.id,
        "iris-two-card",
        &plan,
        &checkpoint_root,
        MidRun::CancelAfterCheckpoint,
    )
    .await;
    assert!(
        matches!(first.outcome, Err(WorkerError::Canceled(_))),
        "the canceled two-card run ends Canceled: {:?}",
        first.outcome
    );
    let first_cards = two_card_placement(&first, "ddp_cancel");
    let canceled = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == "iris-two-card" && body["status"] == "canceled")
        .map(|(_, body)| body.clone())
        .expect("the run posted a terminal canceled update");
    let stopped_at = canceled["result"]["stepsCompleted"]
        .as_u64()
        .unwrap_or_else(|| panic!("a durable cancel reports its step: {canceled:#}"));
    assert!(
        (2..u64::from(STEPS)).contains(&stopped_at),
        "canceled after the first checkpoint and before the end: {canceled:#}"
    );
    assert!(checkpoint_root.join("latest").is_file());

    // 2. Resumed from that checkpoint, on two cards again, to the end.
    cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    let (mut plan, _, _) = build("iris-two-card-resume", "lora_two_card", ddp.clone());
    apply_resume_translation(&mut plan.config, &checkpoint_root);
    let resumed = drive_two_cards(
        &settings,
        &posts,
        &cancel,
        &project.id,
        "iris-two-card-resume",
        &plan,
        &checkpoint_root,
        MidRun::Nothing,
    )
    .await;
    resumed
        .outcome
        .as_ref()
        .unwrap_or_else(|error| panic!("the resumed two-card run completes: {error}"));
    let resumed_cards = two_card_placement(&resumed, "ddp_resume");
    let result = completed_result(&posts, "iris-two-card-resume");
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    assert_two_card_receipt(&result, "ddp");
    let resumed_steps = posted_training_steps(&posts.lock().unwrap(), "iris-two-card-resume");
    let resumed_executed = resume_continuation(&resumed_steps, stopped_at, u64::from(STEPS))
        .unwrap_or_else(|error| panic!("{error}"));
    let adapter = output_dir.join("iris_two_card.safetensors");
    assert!(adapter.is_file(), "rank 0 exported the adapter");

    // 3. A rank failure: rank 1 is killed mid-run.
    let (plan, _, failed_root) = build("iris-two-card-lost", "lora_two_card_lost", ddp);
    let lost = drive_two_cards(
        &settings,
        &posts,
        &cancel,
        &project.id,
        "iris-two-card-lost",
        &plan,
        &failed_root,
        MidRun::KillRankOne,
    )
    .await;
    let failure = match &lost.outcome {
        Err(error) if !matches!(error, WorkerError::Canceled(_)) => error.to_string(),
        other => panic!("a run that lost a rank fails typed, got {other:?}"),
    };
    assert!(
        failure.contains("Iris distributed training failed [")
            && failure.contains("on rank 1 of 2")
            && failure.contains("Every rank process was stopped"),
        "the failure names the lost rank: {failure}"
    );
    two_card_placement(&lost, "rank_lost");
    // No rank process outlives the job holding a card.
    let lost_pids: Vec<u64> = lost.rank_pids.iter().map(|(_, pid)| *pid).collect();
    let mut holding = lost_pids.clone();
    for _ in 0..30 {
        let contexts = parse_compute_apps(&nvidia_smi(&[
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader",
        ]));
        holding = lost_pids
            .iter()
            .copied()
            .filter(|pid| contexts.iter().any(|(context_pid, _)| context_pid == pid))
            .collect();
        if holding.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    assert!(
        holding.is_empty(),
        "rank processes {holding:?} still hold a GPU context after the failed job returned"
    );

    // The resumed adapter is applied by a generation job (on the first card alone).
    let mut render_settings = settings.clone();
    render_settings.gpu_id = visible[0].clone();
    let base = render(
        &render_settings,
        &posts,
        &project_object,
        "iris-two-card-base",
        json!([]),
    )
    .await;
    let tuned = render(
        &render_settings,
        &posts,
        &project_object,
        "iris-two-card-tuned",
        json!([{
            "id": "lora_two_card", "name": "Iris two-card smoke", "family": "iris",
            "networkType": "lora", "installedPath": adapter.display().to_string(), "weight": 1.0
        }]),
    )
    .await;
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the two-card adapter must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b smoke=two_card backend=candle cards={} {gate} \
         ddp_cards={first_cards:?} resume_cards={resumed_cards:?} global_batch={TWO_CARD_WORLD} \
         canceled_at_step={stopped_at} resumed_first_step={} \
         resumed_steps_executed={resumed_executed} resumed_to={STEPS} rank_lost=\"{failure}\" \
         cancel_s={:.1} resume_s={:.1} lost_s={:.1} \
         adapter_effect_mean_abs_diff={mean_abs_diff:.3}",
        settings.gpu_id, resumed_steps[0], first.seconds, resumed.seconds, lost.seconds,
    );

    // 4. FSDP (sharded) on two cards, to completion.
    let (plan, _, fsdp_root) = build(
        "iris-two-card-fsdp",
        "lora_two_card_fsdp",
        two_card_config("fsdp2"),
    );
    let sharded = drive_two_cards(
        &settings,
        &posts,
        &cancel,
        &project.id,
        "iris-two-card-fsdp",
        &plan,
        &fsdp_root,
        MidRun::Nothing,
    )
    .await;
    sharded
        .outcome
        .as_ref()
        .unwrap_or_else(|error| panic!("the sharded two-card run completes: {error}"));
    let sharded_cards = two_card_placement(&sharded, "fsdp");
    let sharded_result = completed_result(&posts, "iris-two-card-fsdp");
    assert_eq!(
        sharded_result["stepsCompleted"], STEPS,
        "{sharded_result:#}"
    );
    assert_two_card_receipt(&sharded_result, "fsdp2");
    eprintln!(
        "RESULT status=pass trainer=iris_3b smoke=two_card_fsdp backend=candle cards={} \
         fsdp_cards={sharded_cards:?} global_batch={TWO_CARD_WORLD} fsdp_steps={} fsdp_s={:.1}",
        settings.gpu_id, sharded_result["stepsCompleted"], sharded.seconds,
    );
}

/// The floor the gate reads for a two-rank run over the worker's TWO cards (one rank per card),
/// judged against the smaller card. Panics when the gate would refuse.
async fn admitted_two_cards(
    settings: &Settings,
    config: &sceneworks_core::training::TrainingConfig,
) -> String {
    use sceneworks_core::training::iris::curricula;
    let devices = crate::iris_dist::visible_training_devices(settings);
    assert_eq!(devices, TWO_CARD_WORLD, "the worker owns both cards");
    assert_eq!(
        curricula::iris_topology_refusal(config, devices, "GPU"),
        None
    );
    #[cfg(target_os = "macos")]
    let capacity = crate::iris_training::mlx_training_capacity_gb();
    #[cfg(not(target_os = "macos"))]
    let capacity = crate::iris_training::candle_training_capacity_gb(&settings.gpu_id)
        .await
        .ok();
    let lane = crate::iris_training::training_lane();
    let floor = curricula::iris_run_memory_floor(lane, config, devices)
        .expect("a floor row covers the two-card shape");
    assert_eq!(
        crate::iris_training::iris_plan_memory_refusal(lane, "iris_3b", config, devices, capacity),
        None,
        "the gate admits one rank per card"
    );
    format!(
        "gate_floor_gib_per_card={} gate_floor_provenance={} gate_smallest_card_gib={}",
        floor.gib,
        floor.provenance.as_str(),
        capacity.map_or_else(|| "unread".to_owned(), |gib| format!("{gib:.2}"))
    )
}

/// What needs no GPU: the continuation rule the resume smokes assert, and that the full-fine-tune
/// smokes' own configs are valid plans the admission gate admits on the hosts they will run on —
/// checked here so a real-weight dispatch does not spend its hour discovering a refused config.
#[cfg(test)]
mod unit {
    use super::*;

    fn training(job: &str, step: u64) -> (String, Value) {
        (
            job.to_owned(),
            json!({ "status": "running", "stage": "training",
                    "message": format!("Training step {step} of 8 (loss 0.1234).") }),
        )
    }

    #[test]
    fn a_resumed_run_must_continue_one_past_the_stop() {
        // Canceled at 4 of 8: the continuation trains 5, 6, 7, 8.
        assert_eq!(resume_continuation(&[5, 6, 7, 8], 4, 8), Ok(4));
        assert_eq!(resume_continuation(&[8], 7, 8), Ok(1));
        // A run that ignored `iris.resume_from` restarts at 1 — and still ends at step 8.
        let restarted = resume_continuation(&[1, 2, 3, 4, 5, 6, 7, 8], 4, 8).unwrap_err();
        assert!(
            restarted.contains("trained step 1") && restarted.contains("restarted"),
            "{restarted}"
        );
        // Re-training the stop step itself is a repeat too.
        assert!(resume_continuation(&[4, 5, 6, 7, 8], 4, 8)
            .unwrap_err()
            .contains("trained step 4"));
        // Skipping ahead, stopping short, and reporting nothing are not continuations either.
        assert!(resume_continuation(&[6, 7, 8], 4, 8).is_err());
        assert!(resume_continuation(&[5, 6, 7], 4, 8).is_err());
        assert!(resume_continuation(&[5, 7, 8], 4, 8).is_err());
        assert!(resume_continuation(&[], 4, 8)
            .unwrap_err()
            .contains("no training step"));
    }

    #[test]
    fn training_steps_are_read_from_the_named_jobs_progress_posts_only() {
        let posts = vec![
            training("first", 1),
            training("first", 2),
            (
                "resumed".to_owned(),
                json!({ "status": "running", "message": "Loading base model." }),
            ),
            training("resumed", 3),
            (
                "resumed".to_owned(),
                json!({ "status": "running", "message": "Saved checkpoint at step 4." }),
            ),
            training("resumed", 4),
            (
                "resumed".to_owned(),
                json!({ "status": "completed", "result": { "stepsCompleted": 4 } }),
            ),
        ];
        assert_eq!(posted_training_steps(&posts, "resumed"), vec![3, 4]);
        assert_eq!(posted_training_steps(&posts, "first"), vec![1, 2]);
        assert!(posted_training_steps(&posts, "absent").is_empty());
        // The message is the worker's own: this is what `map_training_progress` posts.
        let (_, _, _, message) = crate::training_jobs::map_training_progress(
            gen_core::TrainingProgress::Training {
                step: 5,
                total: 8,
                loss: 0.25,
            },
            8,
        );
        assert_eq!(
            posted_training_steps(&[("j".to_owned(), json!({ "message": message }))], "j"),
            vec![5]
        );
    }

    /// The full-fine-tune smokes read the FULL floor rows, and the hosts they are dispatched to —
    /// nax-macos-2 (128 GiB unified) and the campaign RTX PRO 6000 (95.59 GiB) — are admitted.
    #[test]
    fn the_full_fine_tune_smokes_are_admitted_on_the_hosts_they_run_on() {
        use sceneworks_core::training::iris::floors::CAMPAIGN_CUDA_CARD;
        use sceneworks_core::training::iris::{
            curricula, iris_generation_memory_floor, restore, validate_iris_training_config,
        };
        let hosts = [("mlx", 128.0), ("candle", CAMPAIGN_CUDA_CARD.gib())];

        let generation = CancelResumeArm::FullStage1.config();
        validate_iris_training_config(
            &sceneworks_core::training::iris::iris_generation_target(),
            &generation,
        )
        .expect("the bounded stage1_256 config is a valid Iris request");
        assert_eq!(generation.advanced["networkType"], json!("full"));
        assert_eq!(generation.advanced["iris"]["stage"], json!("stage1_256"));
        assert_eq!(generation.optimizer, "muon");
        let options = sceneworks_core::training::iris::iris_options(&generation).unwrap();
        assert!(
            curricula::iris_repa_enabled(&options),
            "stage 1 aligns to REPA"
        );
        assert_eq!(curricula::iris_world_size(&generation), 1);
        for (lane, capacity) in hosts {
            // One rank's full 256² row plus the REPA teacher.
            let row = iris_generation_memory_floor(lane, "full", RESOLUTION, false).unwrap();
            assert_eq!(
                curricula::iris_run_memory_floor(lane, &generation, 1).map(|floor| floor.gib),
                Some(row.gib + curricula::REPA_EXTRA_FLOOR_GIB)
            );
            assert_eq!(
                crate::iris_training::iris_plan_memory_refusal(
                    lane,
                    "iris_3b",
                    &generation,
                    1,
                    Some(capacity)
                ),
                None,
                "{lane}: stage1_256 full fine-tune"
            );
        }
        // The resumed run is the same request plus the API-owned resume keys.
        let mut resumed = generation.clone();
        apply_resume_translation(&mut resumed, Path::new("/data/run.checkpoints"));
        assert_eq!(
            resumed.advanced["iris"]["resume_from"],
            json!("/data/run.checkpoints")
        );
        assert_eq!(resumed.advanced["resume"], json!(true));
        assert_eq!(resumed.advanced["iris"]["stage"], json!("stage1_256"));

        let restoration = restore_smoke_config(true);
        restore::validate_restore_config(&restoration)
            .expect("the bounded paper recipe is a valid restoration request");
        assert_eq!(restoration.advanced["networkType"], json!("full"));
        let lora = restore_smoke_config(false);
        for (lane, capacity) in hosts {
            let full = restore::restore_memory_floor(lane, &restoration).unwrap();
            assert!(
                full.gib > restore::restore_memory_floor(lane, &lora).unwrap().gib,
                "{lane}: the full smoke reads the full row"
            );
            assert_eq!(
                crate::iris_training::iris_plan_memory_refusal(
                    lane,
                    "iris_3b_restore",
                    &restoration,
                    1,
                    Some(capacity)
                ),
                None,
                "{lane}: restoration full fine-tune at 256²"
            );
        }
    }

    /// The table of lanes that implement task gradient checkpointing is held to the trainer's own
    /// answer: agreement passes; a trainer that starts accepting it on an unlisted lane, a listed
    /// lane whose trainer refuses, and an unrelated failure each fail, saying which.
    #[test]
    fn the_checkpointing_lane_table_is_held_to_the_trainers_answer() {
        let refuses = Err(
            "iris_3b_depth trainer: gradient (activation) checkpointing is not implemented for \
             the depth task on the MLX backend"
                .to_owned(),
        );
        assert_eq!(
            checkpointing_table_verdict("depth", "candle", true, &Ok(())),
            Ok(())
        );
        assert_eq!(
            checkpointing_table_verdict("depth", "mlx", false, &refuses),
            Ok(())
        );
        // The MLX twin lands: the trainer accepts, the table still says no -> loud, with the fix.
        let drift = checkpointing_table_verdict("depth", "mlx", false, &Ok(())).unwrap_err();
        assert!(
            drift.contains("now ACCEPTS")
                && drift.contains("add the lane")
                && drift.contains("full_ac"),
            "{drift}"
        );
        assert!(
            checkpointing_table_verdict("restoration", "candle", true, &refuses)
                .unwrap_err()
                .contains("lists candle")
        );
        assert!(checkpointing_table_verdict(
            "depth",
            "mlx",
            false,
            &Err("weights missing".to_owned())
        )
        .unwrap_err()
        .contains("other than checkpointing"));
    }

    const APPS: &str = "4321, GPU-aaaa\n8765, GPU-bbbb\n999, GPU-aaaa\n";

    fn contexts(csv: &str) -> std::collections::BTreeSet<(u64, String)> {
        parse_compute_apps(csv).into_iter().collect()
    }

    /// The driver's per-process listing and per-card memory, as `nvidia-smi` prints them.
    #[test]
    fn nvidia_smi_listings_are_parsed() {
        assert_eq!(
            parse_compute_apps(APPS),
            vec![
                (4321, "GPU-aaaa".to_owned()),
                (8765, "GPU-bbbb".to_owned()),
                (999, "GPU-aaaa".to_owned())
            ]
        );
        // Blank output (no compute process), a header-ish or torn line: nothing, never a panic.
        assert!(parse_compute_apps("").is_empty());
        assert!(parse_compute_apps("pid, gpu_uuid\n12,\nnot-a-pid, GPU-x\n").is_empty());
        assert_eq!(
            parse_gpu_memory("0, GPU-aaaa, 95262\n1, GPU-bbbb, 412\n"),
            vec![
                (0, "GPU-aaaa".to_owned(), 95_262),
                (1, "GPU-bbbb".to_owned(), 412)
            ]
        );
        assert!(parse_gpu_memory("0, GPU-aaaa, [N/A]\n").is_empty());
    }

    /// Each rank is placed by ITS OWN pid's context; two ranks on one card, a rank with no context,
    /// a rank on two cards, and a missing or duplicated rank record are all refused.
    #[test]
    fn each_rank_must_have_run_on_its_own_card() {
        let seen = contexts(APPS);
        assert_eq!(
            ranks_on_distinct_cards(&[(0, 4321), (1, 8765)], &seen, 2),
            Ok(vec![(0, "GPU-aaaa".to_owned()), (1, "GPU-bbbb".to_owned())])
        );
        // Both ranks on card aaaa: not a two-device run, whatever the plan said.
        let shared = ranks_on_distinct_cards(&[(0, 4321), (1, 999)], &seen, 2).unwrap_err();
        assert!(
            shared.contains("SAME card GPU-aaaa") && shared.contains("ranks 0 and 1"),
            "{shared}"
        );
        assert!(ranks_on_distinct_cards(&[(0, 4321), (1, 5555)], &seen, 2)
            .unwrap_err()
            .contains("rank 1 (pid 5555) never held a GPU context"));
        let both = contexts("4321, GPU-aaaa\n4321, GPU-bbbb\n8765, GPU-bbbb\n");
        assert!(ranks_on_distinct_cards(&[(0, 4321), (1, 8765)], &both, 2)
            .unwrap_err()
            .contains("several cards"));
        assert!(ranks_on_distinct_cards(&[(0, 4321)], &seen, 2)
            .unwrap_err()
            .contains("rank 1 of 2 recorded 0 processes"));
        assert!(
            ranks_on_distinct_cards(&[(0, 4321), (0, 8765), (1, 999)], &seen, 2)
                .unwrap_err()
                .contains("rank 0 of 2 recorded 2 processes")
        );
    }

    /// A card that held a rank gained memory over its pre-run baseline; a tenant's standing
    /// allocation is baseline, not gain.
    #[test]
    fn both_cards_must_have_gained_memory() {
        let mut observed = CardObservations::default();
        observed.record_memory(&parse_gpu_memory("0, GPU-aaaa, 500\n1, GPU-bbbb, 90000\n"));
        observed.record_memory(&parse_gpu_memory(
            "0, GPU-aaaa, 21000\n1, GPU-bbbb, 90100\n",
        ));
        observed.record_memory(&parse_gpu_memory("0, GPU-aaaa, 600\n1, GPU-bbbb, 90000\n"));
        let cards = ["GPU-aaaa".to_owned(), "GPU-bbbb".to_owned()];
        assert_eq!(
            cards_gained_memory(&observed, &cards[..1], 4096),
            Ok(vec![("GPU-aaaa".to_owned(), 20_500)])
        );
        // Card bbbb is full of someone else's 90 GB and gained 100 MiB: no rank trained there.
        let idle = cards_gained_memory(&observed, &cards, 4096).unwrap_err();
        assert!(idle.contains("GPU-bbbb gained only 100 MiB"), "{idle}");
        assert!(cards_gained_memory(&observed, &["GPU-cccc".to_owned()], 1)
            .unwrap_err()
            .contains("never sampled"));
        assert_eq!(observed.index["GPU-bbbb"], 1);
    }

    /// The two-card configs are valid requests that put ONE rank on each of two devices, state
    /// their global batch, and are refused on a one-device worker rather than doubled up.
    #[test]
    fn the_two_card_configs_need_two_devices_and_never_share_one() {
        use sceneworks_core::training::iris::floors::CAMPAIGN_CUDA_CARD;
        use sceneworks_core::training::iris::{curricula, validate_iris_training_config};
        let target = sceneworks_core::training::iris::iris_generation_target();
        for strategy in ["ddp", "fsdp2"] {
            let config = two_card_config(strategy);
            validate_iris_training_config(&target, &config)
                .unwrap_or_else(|error| panic!("{strategy}: {error}"));
            assert_eq!(curricula::iris_world_size(&config), 2);
            assert_eq!(curricula::iris_global_batch(&config, 2), 2);
            assert_eq!(config.advanced["iris"]["expected_global_batch"], json!(2));
            let receipt = curricula::iris_curriculum_receipt(&config);
            assert_eq!(receipt["dist"]["strategy"], json!(strategy));
            assert_eq!(receipt["dist"]["shareDevice"], json!(false));
            assert_eq!(curricula::iris_topology_refusal(&config, 2, "GPU"), None);
            assert!(
                curricula::iris_topology_refusal(&config, 1, "GPU").is_some(),
                "{strategy}: one device cannot host two unshared ranks"
            );
            // One rank per card: each card needs ONE rank's floor, and the campaign card has it.
            assert_eq!(
                curricula::iris_run_memory_floor("candle", &config, 2),
                curricula::iris_rank_memory_floor("candle", &config)
            );
            assert_eq!(
                crate::iris_training::iris_plan_memory_refusal(
                    "candle",
                    "iris_3b",
                    &config,
                    2,
                    Some(CAMPAIGN_CUDA_CARD.gib())
                ),
                None
            );
        }
    }
}
