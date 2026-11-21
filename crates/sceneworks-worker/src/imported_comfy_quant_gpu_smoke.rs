//! Real-weight worker acceptance for a linked Krea 2 checkpoint stored in one of the ComfyUI
//! descriptor-gated quant conventions (sc-27092).
//!
//! Like the engine-only harnesses, this drives both worker jobs that make an imported checkpoint
//! usable: `model_import` approves and compiles the linked file into the user manifest, then
//! `image_generate` resolves that stamped plan, admits and dispatches the native load, renders one
//! image, and reports the resulting asset through the worker progress API. The final assertions
//! deserialize the asset's `checkpointWeightFacts`, so a pre-load/default stamp cannot masquerade
//! as the engine's runtime materialization receipt.
//!
//! Two encodings share one body, selected by the environment:
//!
//! - `SCENEWORKS_IMPORTED_FP8_CHECKPOINT` — ComfyUI **scalar-scaled fp8** (E4M3FN): `F8_E4M3`
//!   weights with an `F32` per-tensor `.weight_scale` under a `.comfy_quant` descriptor.
//! - `SCENEWORKS_IMPORTED_MXFP8_CHECKPOINT` — ComfyUI Kitchen **MXFP8**: `F8_E4M3` weights with a
//!   `U8` block-scale companion, declared file-level.
//!
//! Native-format acceptance requires a measured packed execution row from the generated request.
//! Muse additionally runs twice with the same additive adapter to exercise resident reuse.
//!
//! The smoke is intentionally fail-loud. The environment variable and all pinned files must be
//! present; an absent runner cache is a failure, never a skip-and-pass.
//!
//! ```text
//! $env:SCENEWORKS_IMPORTED_FP8_CHECKPOINT = "C:\Users\Michael\Downloads\krea2SATDirtyrealism_krea2SATDirtyuncut_fp8.safetensors"
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! cargo test -p sceneworks-worker --features backend-candle --release imported_comfy_quant_worker_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::post;
use axum::Json;
use sceneworks_core::checkpoint_weight_facts::CheckpointWeightFactsV1;
use serde_json::{json, Value};

use super::*;

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

#[derive(Clone)]
struct WorkerApiStubState {
    progress_posts: ProgressPosts,
    metrics: ProgressPosts,
}

fn required_env_path(key: &str) -> PathBuf {
    let value = std::env::var(key).unwrap_or_else(|_| panic!("set ${key}"));
    let value = value.trim();
    assert!(!value.is_empty(), "${key} must not be empty");
    PathBuf::from(value)
}

fn linked_import_location(checkpoint: &Path, hub: &Path) -> (PathBuf, String) {
    // Hub snapshot files link into their repository's sibling blobs directory. Approving only the
    // snapshot parent would correctly fail the product's symlink confinement check. Approve this
    // one model repository, retaining normal confinement and the original .safetensors filename.
    let root = checkpoint
        .strip_prefix(hub)
        .ok()
        .and_then(|relative| relative.components().next())
        .filter(|component| {
            component
                .as_os_str()
                .to_str()
                .is_some_and(|name| name.starts_with("models--"))
        })
        .map(|component| hub.join(component.as_os_str()))
        .unwrap_or_else(|| {
            checkpoint
                .parent()
                .expect("checkpoint parent")
                .to_path_buf()
        });
    let relative = checkpoint
        .strip_prefix(&root)
        .unwrap()
        .components()
        .map(|component| {
            component
                .as_os_str()
                .to_str()
                .expect("UTF-8 linked relative path")
        })
        .collect::<Vec<_>>()
        .join("/");
    (root, relative)
}

#[test]
fn linked_import_approves_one_hub_repository_and_preserves_portable_relative_path() {
    let hub = std::env::temp_dir().join("checkpoint-fixture-cache");
    let checkpoint = hub.join("models--publisher--model/snapshots/rev/diffusion/model.safetensors");
    let (root, relative) = linked_import_location(&checkpoint, &hub);
    assert_eq!(root, hub.join("models--publisher--model"));
    assert_eq!(relative, "snapshots/rev/diffusion/model.safetensors");
    let local = hub
        .with_file_name("local-source")
        .join("community.safetensors");
    let (root, relative) = linked_import_location(&local, &hub);
    assert_eq!(root, local.parent().unwrap());
    assert_eq!(relative, "community.safetensors");
}

fn stub_job_snapshot(job_id: &str) -> Value {
    let job_type = if job_id.contains("import") {
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
        "assignedGpu": "1",
        "workerId": "comfy-quant-smoke-worker",
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
        "gpuId": "1",
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

async fn stub_metrics(
    State(state): State<WorkerApiStubState>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    state
        .metrics
        .lock()
        .expect("metrics lock")
        .push((job_id, payload.clone()));
    Json(payload)
}

async fn spawn_worker_api_stub() -> (String, ProgressPosts, ProgressPosts) {
    let progress_posts = Arc::new(Mutex::new(Vec::new()));
    let metrics = Arc::new(Mutex::new(Vec::new()));
    let app = crate::test_api_stub::job_progress_routes(stub_job, stub_progress)
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state(WorkerApiStubState {
            progress_posts: progress_posts.clone(),
            metrics: metrics.clone(),
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
    (format!("http://{address}"), progress_posts, metrics)
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

/// The one body both encodings run: import the linked checkpoint, render from it, and assert the
/// manifest's classification and the asset's measured facts.
struct SmokeCase {
    checkpoint_env: &'static str,
    model_id: &'static str,
    expected_quant_format: &'static str,
    expected_codec_id: &'static str,
    family: &'static str,
    provider: &'static str,
    companion_repo: Option<&'static str>,
    with_adapter_and_reuse: bool,
}

#[derive(Clone, Copy)]
enum AcceptanceStage {
    ImportOnly,
    Generate,
}

async fn run_imported_comfy_quant_smoke(case: SmokeCase, stage: AcceptanceStage) {
    let SmokeCase {
        checkpoint_env,
        model_id,
        expected_quant_format,
        expected_codec_id,
        family,
        provider,
        companion_repo,
        with_adapter_and_reuse,
    } = case;
    let _ = tracing_subscriber::fmt()
        .json()
        .with_env_filter("info")
        .try_init();
    let checkpoint = required_env_path(checkpoint_env);
    assert!(
        checkpoint.is_file(),
        "{checkpoint_env} is not a file: {}",
        checkpoint.display()
    );
    let hf_hub_cache = required_env_path("HF_HUB_CACHE");
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, progress_posts, metrics) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "comfy-quant-smoke-worker".to_owned();
    settings.gpu_id = "1".to_owned();
    settings.backend_candle_enabled = true;
    settings.heartbeat_seconds = 30;

    if let Some(companion_repo) = companion_repo {
        let base = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, companion_repo)
            .expect("HF_HUB_CACHE must contain the model's companion snapshot")
            .join("bf16");
        assert!(
            base.join("transformer/config.json").is_file()
                && base.join("tokenizer/tokenizer.json").is_file(),
            "the cached bf16 companion snapshot is incomplete: {}",
            base.display()
        );
    }

    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest directory creates");
    let manifest_path = manifests.join("user.models.jsonc");
    std::fs::write(&manifest_path, r#"{ "schemaVersion": 1, "models": [] }"#)
        .expect("empty user model manifest writes");

    let (checkpoint_root, checkpoint_relative) = linked_import_location(&checkpoint, &hf_hub_cache);
    let plan_store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let approved = plan_store
        .approve_root(&checkpoint_root)
        .expect("checkpoint parent approves as a linked library root");
    let import_job = smoke_job(
        "comfy-quant-linked-import",
        "model_import",
        json!({
            "modelId": model_id,
            "linkedRootId": approved.root_id,
            "linkedRelativePath": checkpoint_relative,
            "manifestPath": manifest_path,
            "ownershipMode": "linked",
            "manifestEntry": {
                "id": model_id,
                "name": "Shared-format worker acceptance",
                "type": "image",
                "catalogScope": "user",
                "source": {
                    "provider": "linked-library",
                    "rootId": approved.root_id,
                    "relativePath": checkpoint_relative
                }
            }
        }),
    );
    let api = ApiClient::new(&settings);
    crate::model_jobs::run_model_import_job(&api, &settings, &reqwest::Client::new(), &import_job)
        .await
        .expect("the linked ComfyUI-quantized worker import completes");

    let imported = imported_manifest_entry(&manifest_path, model_id);
    assert_eq!(
        imported.get("family").and_then(Value::as_str),
        Some(family),
        "the checkpoint compiler determines the architecture from the source"
    );
    assert_eq!(
        imported.get("importSourceShape").and_then(Value::as_str),
        Some(if family == "sdxl" {
            "fused_checkpoint"
        } else {
            "transformer_file"
        })
    );
    assert_eq!(
        imported.get("importQuantFormat").and_then(Value::as_str),
        Some(expected_quant_format),
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
    let import_result = completed_result(&progress_posts, "comfy-quant-linked-import");
    assert_eq!(
        import_result.get("checkpointId").and_then(Value::as_str),
        Some(checkpoint_id.as_str())
    );

    if matches!(stage, AcceptanceStage::ImportOnly) {
        eprintln!("IMPORT_FORMAT_PREFLIGHT model={model_id} source_codec={expected_codec_id} checkpoint={checkpoint_id}");
        return;
    }

    let project = ProjectStore::new(settings.data_dir.clone(), "comfy-quant-smoke")
        .create_project("Imported ComfyUI-quantized worker smoke")
        .expect("smoke project creates");
    let mut adapters = Vec::new();
    if with_adapter_and_reuse {
        let source = required_env_path("SCENEWORKS_IMPORTED_FP8_ADAPTER");
        assert!(source.is_file(), "required additive adapter must exist");
        let directory = settings.data_dir.join("loras").join("acceptance");
        std::fs::create_dir_all(&directory).expect("owned adapter directory creates");
        let selected = directory.join("adapter.safetensors");
        std::fs::copy(&source, &selected).expect("stage the actual additive adapter in owned data");
        adapters.push(json!({"path":selected,"weight":1.0}));
    }
    // Match the reported Muse load's geometry, step count, PAG and adapter scale. Content and seed
    // are fixed here for repeatability; the historical timing is a load baseline, not a pixel oracle.
    let (width, height, steps) = if with_adapter_and_reuse {
        (1408, 2048, 8)
    } else {
        (256, 256, 1)
    };
    let mut advanced = json!({"steps":steps});
    if with_adapter_and_reuse {
        advanced["pag"] = json!({"scale":1.2});
    }
    for iteration in 0..if with_adapter_and_reuse { 2 } else { 1 } {
        let image_job_id = format!("comfy-quant-image-generate-{iteration}");
        let image_job = smoke_job(
            &image_job_id,
            "image_generate",
            json!({
                "projectId": project.id,
                "model": model_id,
                "mode": "text_to_image",
                "prompt": "a red fox in a sunlit autumn forest, sharp focus",
                "negativePrompt": "",
                "width": width,
                "height": height,
                "count": 1,
                "seed": 42,
                "advanced": advanced,
                "loras": adapters,
                "modelManifestEntry": imported
            }),
        );
        let physical_gpu = std::env::var("SCENEWORKS_GPU_ID")
            .expect("manual proof must name its reserved physical GPU for hardware metrics");
        let hardware = crate::job_metrics::JobMetricsProbe::start(&physical_gpu);
        crate::image_jobs::run_image_generate_job(&api, &settings, &image_job, &Default::default())
            .await
            .expect("the linked ComfyUI-quantized worker generation completes");
        let hardware = hardware.finish().await;

        let image_result = completed_result(&progress_posts, &image_job_id);
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
            Some(provider),
            "the plan must dispatch to the source architecture's provider"
        );

        let facts: CheckpointWeightFactsV1 = serde_json::from_value(
            raw.get(crate::checkpoint_weight_facts_host::FACTS_RAW_SETTINGS_KEY)
                .cloned()
                .expect("the asset carries checkpointWeightFacts"),
        )
        .expect("checkpointWeightFacts is a valid correlated fact set");
        assert!(
            facts.declares(expected_codec_id),
            "the asset must report source codec {expected_codec_id}"
        );
        let representation = facts
            .representation_label(expected_codec_id)
            .expect("the runtime receipt reports the execution representation");
        assert_eq!(
        facts.executes_natively(expected_codec_id), Some(true),
        "expected measured native materialization for {expected_codec_id}, got {representation:?}"
    );

        let media_path = asset
            .get("mediaPath")
            .and_then(Value::as_str)
            .expect("the asset carries its media path");
        let rendered_path = PathBuf::from(&project.path).join(media_path);
        let decoded = image::open(&rendered_path)
            .unwrap_or_else(|error| {
                panic!("open rendered asset {}: {error}", rendered_path.display())
            })
            .to_rgb8();
        let rendered = gen_core::Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        };
        let std = crate::smoke_support::image_std(&rendered);
        assert_eq!((rendered.width, rendered.height), (width, height));
        assert!(
            std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
            "the worker render looks degenerate (RGB std {std:.2})"
        );
        eprintln!(
            "RESULT status=pass route=candle_checkpoint_plan provider={provider} source_codec={} \
         execution_representation={} geometry={}x{} rgb_std={std:.3} checkpoint={} output={}",
            expected_codec_id,
            representation,
            rendered.width,
            rendered.height,
            checkpoint.display(),
            rendered_path.display(),
        );
        if let Some(output) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
            let output = PathBuf::from(output);
            assert!(output.is_absolute() && output.is_dir());
            let image_name = format!("{model_id}-{iteration}.png");
            std::fs::copy(&rendered_path, output.join(&image_name))
                .expect("retain generated image");
            let packet = json!({
                "schemaVersion":1, "modelId":model_id, "provider":provider,
                "checkpointId":checkpoint_id,
                "importPlan":imported.get("importPlan"),
                "iteration":iteration, "sourceCodec":expected_codec_id,
                "runtimeFacts":facts, "adapterCount":adapters.len(),
                "physicalGpu":physical_gpu,"hardwareMetrics":hardware,
                "seed":42,"steps":steps,"width":width,"height":height,"output":image_name,
                "metrics":metrics.lock().expect("metrics lock").iter()
                    .filter(|(id, _)| id == &image_job_id).map(|(_,value)|value).collect::<Vec<_>>()
            });
            std::fs::write(
                output.join(format!("{model_id}-{iteration}.json")),
                serde_json::to_vec_pretty(&packet).unwrap(),
            )
            .expect("retain native execution receipt");
        }
    }
    let fqn = std::thread::current()
        .name()
        .expect("actual libtest FQN")
        .to_owned();
    eprintln!("MODEL_INFERENCE_BODY_COMPLETED {fqn}");
    // A successful request's measured NativePacked row comes from the source-aware CUDA loader;
    // its public constructor refuses CPU. The separately dispatched controller binds physical GPU1.
    eprintln!("MODEL_INFERENCE_DEVICE {fqn} candle-cuda");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs a ComfyUI scalar-fp8 Krea 2 checkpoint, the Krea bf16 companion snapshot, and a CUDA device"]
async fn imported_scalar_fp8_worker_gpu_smoke() {
    run_imported_comfy_quant_smoke(scalar_fp8_case(), AcceptanceStage::Generate).await;
}

fn scalar_fp8_case() -> SmokeCase {
    SmokeCase {
        checkpoint_env: "SCENEWORKS_IMPORTED_FP8_CHECKPOINT",
        model_id: "linked_krea2_scalar_fp8",
        expected_quant_format: "fp8_e4m3_scalar",
        expected_codec_id: sceneworks_core::checkpoint_weight_facts::FP8_E4M3_SCALAR_CODEC_ID,
        family: "krea_2",
        provider: "krea_2_turbo",
        companion_repo: Some("SceneWorks/krea-2-turbo-mlx"),
        with_adapter_and_reuse: true,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs a ComfyUI Kitchen MXFP8 Krea 2 checkpoint, the Krea bf16 companion snapshot, and a CUDA device"]
async fn imported_mxfp8_worker_gpu_smoke() {
    run_imported_comfy_quant_smoke(mxfp8_case(), AcceptanceStage::Generate).await;
}

fn mxfp8_case() -> SmokeCase {
    SmokeCase {
        checkpoint_env: "SCENEWORKS_IMPORTED_MXFP8_CHECKPOINT",
        model_id: "linked_krea2_mxfp8",
        expected_quant_format: "mxfp8",
        expected_codec_id: sceneworks_core::checkpoint_weight_facts::MXFP8_CODEC_ID,
        family: "krea_2",
        provider: "krea_2_turbo",
        companion_repo: Some("SceneWorks/krea-2-turbo-mlx"),
        with_adapter_and_reuse: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "manual native ConvRot product validation; real Muse checkpoint and companion snapshot"]
async fn imported_convrot_worker_gpu_smoke() {
    run_imported_comfy_quant_smoke(convrot_case(), AcceptanceStage::Generate).await;
}

fn convrot_case() -> SmokeCase {
    SmokeCase {
        checkpoint_env: "SCENEWORKS_IMPORTED_CONVROT_CHECKPOINT",
        model_id: "linked_krea2_convrot",
        expected_quant_format: "convrot_int8",
        expected_codec_id: "convrot-int8-native",
        family: "krea_2",
        provider: "krea_2_turbo",
        companion_repo: Some("SceneWorks/krea-2-turbo-mlx"),
        with_adapter_and_reuse: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "manual second-architecture native FP8 product validation; real fused SDXL E5M2 source"]
async fn imported_sdxl_fp8_worker_gpu_smoke() {
    run_imported_comfy_quant_smoke(sdxl_fp8_case(), AcceptanceStage::Generate).await;
}

fn sdxl_fp8_case() -> SmokeCase {
    SmokeCase {
        checkpoint_env: "SCENEWORKS_IMPORTED_SDXL_FP8_CHECKPOINT",
        model_id: "linked_sdxl_fp8",
        expected_quant_format: "fp8_e5m2_scalar",
        expected_codec_id: sceneworks_core::checkpoint_weight_facts::FP8_E5M2_SCALAR_CODEC_ID,
        family: "sdxl",
        provider: "sdxl",
        companion_repo: None,
        with_adapter_and_reuse: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "manual native NVFP4 shared-loader regression; real Krea source and companions"]
async fn imported_native_nvfp4_worker_gpu_smoke() {
    run_imported_comfy_quant_smoke(native_nvfp4_case(), AcceptanceStage::Generate).await;
}

fn native_nvfp4_case() -> SmokeCase {
    SmokeCase {
        checkpoint_env: "SCENEWORKS_NVFP4_CHECKPOINT",
        model_id: "linked_krea2_native_nvfp4",
        expected_quant_format: "nvfp4",
        expected_codec_id: sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID,
        family: "krea_2",
        provider: "krea_2_turbo",
        companion_repo: Some("SceneWorks/krea-2-turbo-mlx"),
        with_adapter_and_reuse: false,
    }
}

/// Only normal import/classification and source pinning. This case never loads a model or emits
/// the successful learned-execution protocol used by the separate native generation cases.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "metadata-only real-source preflight; requires the four pinned local artifacts"]
async fn imported_shared_formats_product_import_preflight() {
    for case in [
        scalar_fp8_case(),
        sdxl_fp8_case(),
        convrot_case(),
        native_nvfp4_case(),
    ] {
        run_imported_comfy_quant_smoke(case, AcceptanceStage::ImportOnly).await;
    }
}
