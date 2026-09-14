//! Real-weight worker acceptance for a linked Krea 2 Turbo NVFP4 checkpoint (sc-21716).
//!
//! Unlike the engine-only NVFP4 harnesses, this drives both worker jobs that make an imported
//! checkpoint usable: `model_import` approves and compiles the linked file into the user manifest,
//! then `image_generate` resolves that stamped plan, admits and dispatches the native NVFP4 load,
//! renders one image, and reports the resulting asset through the worker progress API. The final
//! assertions deserialize the asset's `checkpointWeightFacts`, so a pre-load/default stamp cannot
//! masquerade as the engine's runtime materialization receipt.
//!
//! The smoke is intentionally fail-loud. Both environment variables and all pinned files must be
//! present; an absent runner cache is a failure, never a skip-and-pass.
//!
//! ```text
//! $env:SCENEWORKS_IMPORTED_NVFP4_CHECKPOINT = "E:\huggingface\hub\models--Comfy-Org--Krea-2\snapshots\952f49d49653cb42e7d6cf7cbfad74738073ec7d\diffusion_models\krea2_turbo_nvfp4.safetensors"
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! cargo test -p sceneworks-worker --features backend-candle --release imported_nvfp4_worker_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use sceneworks_core::checkpoint_weight_facts::{CheckpointWeightFactsV1, NVFP4_CODEC_ID};
use serde_json::{json, Value};

use super::*;

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

#[derive(Clone)]
struct WorkerApiStubState {
    progress_posts: ProgressPosts,
}

fn required_env_path(key: &str) -> PathBuf {
    let value = std::env::var(key).unwrap_or_else(|_| panic!("set ${key}"));
    let value = value.trim();
    assert!(!value.is_empty(), "${key} must not be empty");
    PathBuf::from(value)
}

fn stub_job_snapshot(job_id: &str) -> Value {
    let job_type = if job_id == "nvfp4-linked-import" {
        "model_import"
    } else {
        "image_generate"
    };
    json!({
        "id": job_id,
        "type": job_type,
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": "0",
        "workerId": "nvfp4-smoke-worker",
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
        "createdAt": "2026-08-28T00:00:00Z",
        "updatedAt": "2026-08-28T00:00:00Z",
        "startedAt": "2026-08-28T00:00:00Z",
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": "2026-08-28T00:00:00Z"
    })
}

fn stub_worker_snapshot(worker_id: &str) -> Value {
    json!({
        "id": worker_id,
        "gpuId": "0",
        "gpuName": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "status": "busy",
        "currentJobId": null,
        "capabilities": [],
        "loadedModels": [],
        "registeredAt": "2026-08-28T00:00:00Z",
        "lastSeenAt": "2026-08-28T00:00:00Z"
    })
}

async fn stub_job(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
    Json(stub_job_snapshot(&job_id))
}

async fn stub_progress(
    State(state): State<WorkerApiStubState>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    state
        .progress_posts
        .lock()
        .expect("progress posts lock")
        .push((job_id.clone(), payload));
    Json(stub_job_snapshot(&job_id))
}

async fn stub_heartbeat(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
    Json(stub_worker_snapshot(&worker_id))
}

async fn stub_metrics(Json(payload): Json<Value>) -> Json<Value> {
    Json(payload)
}

async fn spawn_worker_api_stub() -> (String, ProgressPosts) {
    let progress_posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(stub_job))
        .route("/api/v1/jobs/:job_id/progress", post(stub_progress))
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state(WorkerApiStubState {
            progress_posts: progress_posts.clone(),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("worker API stub listener binds");
    let address = listener.local_addr().expect("worker API stub has address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("worker API stub serves");
    });
    (format!("http://{address}"), progress_posts)
}

fn smoke_job(job_id: &str, job_type: &str, payload: Value) -> JobSnapshot {
    let mut snapshot = stub_job_snapshot(job_id);
    snapshot["type"] = Value::String(job_type.to_owned());
    snapshot["payload"] = payload;
    serde_json::from_value(snapshot).expect("smoke job snapshot deserializes")
}

fn imported_manifest_entry(manifest_path: &Path, model_id: &str) -> JsonObject {
    let raw = std::fs::read_to_string(manifest_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display()));
    let parsed: Value = serde_json::from_str(&strip_jsonc_comments(&raw))
        .unwrap_or_else(|error| panic!("parse {}: {error}", manifest_path.display()));
    parsed
        .get("models")
        .and_then(Value::as_array)
        .and_then(|models| {
            models
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        })
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "{model_id:?} was not written to {}",
                manifest_path.display()
            )
        })
}

fn completed_result(posts: &ProgressPosts, job_id: &str) -> Value {
    posts
        .lock()
        .expect("progress posts lock")
        .iter()
        .rev()
        .find(|(posted_job_id, body)| {
            posted_job_id == job_id
                && body.get("status").and_then(Value::as_str) == Some("completed")
        })
        .map(|(_, body)| body.get("result").cloned().unwrap_or(Value::Null))
        .unwrap_or_else(|| panic!("{job_id:?} never posted a completed result"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned Krea 2 Turbo NVFP4 checkpoint, the Krea bf16 companion snapshot, and an sm_120 CUDA device"]
async fn imported_nvfp4_worker_gpu_smoke() {
    let checkpoint = required_env_path("SCENEWORKS_IMPORTED_NVFP4_CHECKPOINT");
    assert!(
        checkpoint.is_file(),
        "SCENEWORKS_IMPORTED_NVFP4_CHECKPOINT is not a file: {}",
        checkpoint.display()
    );
    let hf_hub_cache = required_env_path("HF_HUB_CACHE");
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, progress_posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "nvfp4-smoke-worker".to_owned();
    settings.gpu_id = "cuda".to_owned();
    settings.backend_candle_enabled = true;
    settings.heartbeat_seconds = 30;

    let base = crate::model_jobs::huggingface_snapshot_dir(
        &settings.data_dir,
        "SceneWorks/krea-2-turbo-mlx",
    )
    .expect("HF_HUB_CACHE must contain SceneWorks/krea-2-turbo-mlx")
    .join("bf16");
    assert!(
        base.join("transformer/config.json").is_file()
            && base.join("tokenizer/tokenizer.json").is_file(),
        "the cached Krea 2 Turbo bf16 companion snapshot is incomplete: {}",
        base.display()
    );

    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest directory creates");
    let manifest_path = manifests.join("user.models.jsonc");
    std::fs::write(&manifest_path, r#"{ "schemaVersion": 1, "models": [] }"#)
        .expect("empty user model manifest writes");

    let checkpoint_parent = checkpoint
        .parent()
        .expect("checkpoint has a parent directory");
    let checkpoint_name = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .expect("checkpoint filename is UTF-8");
    let plan_store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let approved = plan_store
        .approve_root(checkpoint_parent)
        .expect("checkpoint parent approves as a linked library root");
    let model_id = "linked_krea2_turbo_nvfp4";
    let import_job = smoke_job(
        "nvfp4-linked-import",
        "model_import",
        json!({
            "modelId": model_id,
            "linkedRootId": approved.root_id,
            "linkedRelativePath": checkpoint_name,
            "manifestPath": manifest_path,
            "ownershipMode": "linked",
            "manifestEntry": {
                "id": model_id,
                "name": "Krea 2 Turbo NVFP4 worker smoke",
                "type": "image",
                "catalogScope": "user",
                "source": {
                    "provider": "linked-library",
                    "rootId": approved.root_id,
                    "relativePath": checkpoint_name
                }
            }
        }),
    );
    let api = ApiClient::new(&settings);
    crate::model_jobs::run_model_import_job(&api, &settings, &reqwest::Client::new(), &import_job)
        .await
        .expect("linked NVFP4 worker import completes");

    let imported = imported_manifest_entry(&manifest_path, model_id);
    assert_eq!(
        imported.get("family").and_then(Value::as_str),
        Some("krea_2"),
        "the checkpoint compiler determines the Krea family"
    );
    assert_eq!(
        imported.get("importSourceShape").and_then(Value::as_str),
        Some("transformer_file")
    );
    assert_eq!(
        imported.get("importQuantFormat").and_then(Value::as_str),
        Some("nvfp4"),
        "the linked import classifies the actual safetensors header"
    );
    let checkpoint_id = imported
        .get("importPlan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("checkpointId"))
        .and_then(Value::as_str)
        .expect("the linked import stamps its checkpoint plan id")
        .to_owned();
    assert!(
        plan_store.resolve(&checkpoint_id).is_ok(),
        "the stamped checkpoint plan resolves through the approved root"
    );
    assert!(
        imported
            .get("paths")
            .and_then(Value::as_object)
            .and_then(|paths| paths.get("model"))
            .is_none(),
        "a linked import must remain in-place rather than inventing a managed install"
    );
    let import_result = completed_result(&progress_posts, "nvfp4-linked-import");
    assert_eq!(
        import_result.get("checkpointId").and_then(Value::as_str),
        Some(checkpoint_id.as_str())
    );

    let project = ProjectStore::new(settings.data_dir.clone(), "nvfp4-smoke")
        .create_project("Imported NVFP4 worker smoke")
        .expect("smoke project creates");
    let image_job = smoke_job(
        "nvfp4-image-generate",
        "image_generate",
        json!({
            "projectId": project.id,
            "model": model_id,
            "mode": "text_to_image",
            "prompt": "a red fox in a sunlit autumn forest, sharp focus",
            "negativePrompt": "",
            "width": 256,
            "height": 256,
            "count": 1,
            "seed": 42,
            "advanced": { "steps": 1 },
            "modelManifestEntry": imported
        }),
    );
    crate::image_jobs::run_image_generate_job(&api, &settings, &image_job)
        .await
        .expect("linked NVFP4 worker generation completes");

    let image_result = completed_result(&progress_posts, "nvfp4-image-generate");
    let asset = image_result
        .get("assetWrites")
        .and_then(Value::as_array)
        .and_then(|assets| assets.first())
        .expect("the completed worker result carries one image asset");
    let raw = asset
        .get("rawAdapterSettings")
        .and_then(Value::as_object)
        .expect("the image asset carries worker-resolved settings");
    assert_eq!(
        raw.get("engine").and_then(Value::as_str),
        Some("candle_checkpoint_plan"),
        "the render must run through the worker checkpoint-plan route"
    );
    assert_eq!(
        raw.get("importPlanProvider").and_then(Value::as_str),
        Some("krea_2_turbo"),
        "the plan must dispatch to the imported Krea provider"
    );
    assert_eq!(
        raw.get("checkpointId").and_then(Value::as_str),
        Some(checkpoint_id.as_str())
    );

    let facts: CheckpointWeightFactsV1 = serde_json::from_value(
        raw.get(crate::checkpoint_weight_facts_host::FACTS_RAW_SETTINGS_KEY)
            .cloned()
            .expect("the asset carries checkpointWeightFacts"),
    )
    .expect("checkpointWeightFacts is a valid correlated fact set");
    assert!(
        facts.declares(NVFP4_CODEC_ID),
        "the asset must report source codec {NVFP4_CODEC_ID}"
    );
    assert_eq!(
        facts.executes_natively(NVFP4_CODEC_ID),
        Some(true),
        "the sm_120 load must report at least one native-packed NVFP4 tensor"
    );
    let representation = facts
        .representation_label(NVFP4_CODEC_ID)
        .expect("the runtime receipt reports the NVFP4 execution representation");
    assert!(
        representation == "native-packed" || representation.starts_with("mixed:"),
        "expected a native or mixed/native NVFP4 execution receipt, got {representation:?}"
    );

    let media_path = asset
        .get("mediaPath")
        .and_then(Value::as_str)
        .expect("the asset carries its media path");
    let rendered_path = PathBuf::from(&project.path).join(media_path);
    let decoded = image::open(&rendered_path)
        .unwrap_or_else(|error| panic!("open rendered asset {}: {error}", rendered_path.display()))
        .to_rgb8();
    let rendered = gen_core::Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    };
    let std = crate::smoke_support::image_std(&rendered);
    assert_eq!((rendered.width, rendered.height), (256, 256));
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the worker render looks degenerate (RGB std {std:.2})"
    );
    eprintln!(
        "RESULT status=pass route=candle_checkpoint_plan provider=krea_2_turbo source_codec={} \
         execution_representation={} geometry={}x{} rgb_std={std:.3} checkpoint={} output={}",
        NVFP4_CODEC_ID,
        representation,
        rendered.width,
        rendered.height,
        checkpoint.display(),
        rendered_path.display(),
    );
}
