//! Real-weight worker acceptance for Iris-3B on Candle/CUDA (sc-25680, epic 25678).
//!
//! Drives ONE production `image_generate` job through `run_image_generate_job` — the shipped
//! `iris_3b` catalog entry, cache-only resolution of the backbone AND its `text_encoder`
//! co-requisite through the generic component seam, the cached `candle-gen-iris` load, the per-item
//! drive and `consume_gen_events` — and asserts the job completes with one non-degenerate image
//! asset at the requested geometry. sc-25681 runs it with NON-default controls (Euler / order-1
//! solver, a schedule shift, a CFG interval, the `warn` caption-overflow policy on a prompt longer
//! than the caption budget) and a tiny random-init generation-task LoRA stamped `family=iris`,
//! `irisTask=generation`, `baseModel=iris_3b` (written here, PEFT layout on three real backbone
//! projections), and asserts the live preview, the `caption_truncated` warning and the adapter's
//! applied report reach the job and the asset. Nothing is downloaded: `HF_HUB_CACHE` must already hold both
//! pinned snapshots (the inference `candle-iris` real-weight profile materializes them into the same
//! `E:\huggingface\hub` on the real-weights host). An absent snapshot is a failure, never a skip.
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! cargo test -p sceneworks-worker --features backend-candle --release iris_worker_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

// The single-image + LoRA smoke is CUDA-only; on macOS only the native-batch smoke compiles.
#![cfg_attr(target_os = "macos", allow(dead_code))]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::*;

const JOB_ID: &str = "iris-cuda-image-generate";
const BACKBONE_REPO: &str = "speridlabs/iris-3b";
const TEXT_ENCODER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const SIZE: u32 = 512;
const STEPS: u32 = 30;
const LORA_RANK: usize = 4;

/// Write a tiny random-init Iris generation-task LoRA (PEFT `lora_A`/`lora_B`, bare keys) on three
/// real backbone projections, with the identity stamps the engine requires. Small magnitudes: it
/// probes the adapter path on real weights, it is not a style.
#[cfg(not(target_os = "macos"))]
fn write_stamped_smoke_lora(path: &std::path::Path) {
    let targets: [(&str, usize, usize); 3] = [
        ("blocks.23.attn_proj", 2560, 2560),
        ("blocks.23.q_proj", 2560, 2560),
        ("blocks.22.mlp.w2", 2560, 6826),
    ];
    let mut state: u64 = 25_681;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02
    };
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (stem, out, inp) in targets {
        for (suffix, shape) in [("lora_A", [LORA_RANK, inp]), ("lora_B", [out, LORA_RANK])] {
            let data: Vec<u8> = (0..shape[0] * shape[1])
                .flat_map(|_| next().to_le_bytes())
                .collect();
            tensors.push((format!("{stem}.{suffix}.weight"), shape.to_vec(), data));
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        json!({
            "family": "iris", "irisTask": "generation", "baseModel": "iris_3b",
            "networkType": "lora", "rank": LORA_RANK.to_string(), "alpha": LORA_RANK.to_string(),
        }),
    );
    let mut offset = 0usize;
    for (name, shape, data) in &tensors {
        header.insert(
            name.clone(),
            json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + data.len()] }),
        );
        offset += data.len();
    }
    let mut raw = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    while !raw.len().is_multiple_of(8) {
        raw.push(b' ');
    }
    let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&raw);
    for (_, _, data) in &tensors {
        bytes.extend_from_slice(data);
    }
    std::fs::create_dir_all(path.parent().expect("lora dir")).expect("lora dir creates");
    std::fs::write(path, bytes).expect("smoke LoRA writes");
}

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

fn stub_job_snapshot(job_id: &str) -> Value {
    json!({
        "id": job_id,
        "type": "image_generate",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": "0",
        "workerId": "iris-smoke-worker",
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
        "createdAt": "2026-10-10T00:00:00Z",
        "updatedAt": "2026-10-10T00:00:00Z",
        "startedAt": "2026-10-10T00:00:00Z",
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": "2026-10-10T00:00:00Z"
    })
}

async fn stub_job(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
    Json(stub_job_snapshot(&job_id))
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
    Json(stub_job_snapshot(&job_id))
}

async fn stub_heartbeat(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
    Json(json!({
        "id": worker_id, "gpuId": "0", "gpuName": null, "status": "busy",
        "currentJobId": JOB_ID, "capabilities": [], "loadedModels": [],
        "registeredAt": "2026-10-10T00:00:00Z", "lastSeenAt": "2026-10-10T00:00:00Z"
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

/// The shipped `iris_3b` catalog entry (the real `coRequisite` declaration, not a test copy).
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

#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and a CUDA device"]
async fn iris_worker_gpu_smoke() {
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    settings.gpu_id = "cuda".to_owned();
    settings.backend_candle_enabled = true;
    settings.heartbeat_seconds = 30;

    for repo in [BACKBONE_REPO, TEXT_ENCODER_REPO] {
        assert!(
            crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo).is_some(),
            "HF_HUB_CACHE must already hold {repo} (no download at generation time)"
        );
    }

    let project = ProjectStore::new(settings.data_dir.clone(), "iris-smoke")
        .create_project("Iris-3B CUDA worker smoke")
        .expect("smoke project creates");
    let lora_path = settings
        .data_dir
        .join("loras")
        .join("iris-smoke-style")
        .join("adapter.safetensors");
    write_stamped_smoke_lora(&lora_path);
    // A prompt well past the 300-token caption budget, so the `warn` policy truncates and reports.
    let prompt = std::iter::repeat_n(
        "a lighthouse on a rocky coast at dusk, warm window light, crashing waves, gulls, weathered stone, glowing lantern, dramatic clouds, detailed",
        24,
    )
    .collect::<Vec<_>>()
    .join(", ");
    let mut snapshot = stub_job_snapshot(JOB_ID);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": prompt,
        "negativePrompt": "blurry",
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "seed": 7,
        "loras": [{ "id": "iris-smoke-style", "path": lora_path, "weight": 0.8 }],
        "advanced": {
            "steps": STEPS,
            "guidanceScale": 4.0,
            "sampler": "euler",
            "schedulerShift": 3.0,
            "cfgInterval": [0.05, 0.95],
            "captionOverflow": "warn",
        },
        "modelManifestEntry": shipped_iris_entry(),
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");

    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::image_jobs::run_image_generate_job(&api, &settings, &job)
        .await
        .expect("the Iris-3B CUDA worker job completes");
    let elapsed = started.elapsed();

    let posts = posts.lock().expect("progress posts lock").clone();
    let step_posts = posts
        .iter()
        .filter(|(_, body)| {
            body.get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains(&format!("/{STEPS}")))
        })
        .count();
    assert!(
        step_posts > 0,
        "the engine's step progress must reach the job API: {posts:#?}"
    );
    assert!(
        posts
            .iter()
            .any(|(_, body)| body["result"]["previewFrame"]["dataUrl"]
                .as_str()
                .is_some_and(|url| url.starts_with("data:image/"))),
        "the engine's per-step x0 preview must reach the job API"
    );
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == JOB_ID && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the job posted a completed result");
    let asset = result["assetWrites"]
        .as_array()
        .and_then(|assets| assets.first())
        .expect("the completed result carries one image asset");
    let settings_block = &asset["rawAdapterSettings"];
    assert!(
        settings_block["generationWarnings"]
            .as_array()
            .is_some_and(|warnings| warnings
                .iter()
                .any(|warning| warning["code"] == "caption_truncated")),
        "the over-long prompt's caption_truncated warning is on the asset: {settings_block}"
    );
    let applied = &settings_block["adapterApply"][0];
    assert!(
        applied["applied"].as_u64().is_some_and(|count| count > 0)
            && applied["sha256"].as_str().map(str::len) == Some(64),
        "the stamped LoRA's apply report + identity are on the asset: {settings_block}"
    );
    for (key, value) in [
        ("sampler", json!("euler")),
        ("cfgInterval", json!([0.05, 0.95])),
        ("captionOverflow", json!("warn")),
    ] {
        assert_eq!(settings_block[key], value, "{key} recorded with the image");
    }
    let media_path = asset["mediaPath"]
        .as_str()
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
    assert_eq!((rendered.width, rendered.height), (SIZE, SIZE));
    let std = crate::smoke_support::image_std(&rendered);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the worker render looks degenerate (RGB std {std:.2})"
    );
    eprintln!(
        "RESULT status=pass model=iris_3b backend=candle geometry={}x{} steps={STEPS} \
         sampler=euler shift=3.0 cfg_interval=0.05..0.95 lora_applied={} \
         elapsed_s={:.1} rgb_std={std:.3} output={}",
        rendered.width,
        rendered.height,
        applied["applied"],
        elapsed.as_secs_f64(),
        rendered_path.display(),
    );
}

/// One `nvidia-smi` reading of `memory.used` (MiB) on physical GPU `index`, or `None` when the
/// query fails.
#[cfg(not(target_os = "macos"))]
fn nvidia_memory_used_mib(index: &str) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            index,
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

/// sc-25684 (S6): the STAGED (sequential-residency) working set of an Iris-3B generation on real
/// weights. A 1024² job (count 1, 30 steps; the peak does not depend on the step count) runs through
/// the production Candle arm under a card cap just below the declared resident floor, so the fit gate
/// routes it to `OffloadPolicy::Sequential` — `candle-gen-iris` encodes with the Qwen3-VL text
/// encoder, drops it, then denoises with the backbone. A sampler thread polls `nvidia-smi
/// memory.used` on the job's card every 100 ms; the reported peak is the overall card high-water (the
/// same quantity as the recorded 21.3–21.4 GB RESIDENT peak), and the catalog's
/// `candle.sequentialPeakGb` row is set from it. Dispatch with
/// `-f iris_worker_smoke_filter=iris_staged_vram_measurement` to run this alone in a fresh process
/// (nothing else resident on the card). Deliberately NOT named `iris_worker_gpu_smoke*`: the default
/// smoke filter is a substring match and would run it last in the shared process, contaminating its
/// baseline and peak.
#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU measurement; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and a CUDA device"]
async fn iris_staged_vram_measurement() {
    const STAGED_JOB: &str = "iris-cuda-staged";
    const STAGED_SIZE: u32 = 1024;
    const STAGED_STEPS: u32 = 30;
    // The recorded RESIDENT overall peak at 1024² (inference CUDA runs 38068347300 / 38075865079).
    const RESIDENT_PEAK_GB: f64 = 21.3;
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(hf_hub_cache.is_dir(), "HF_HUB_CACHE is not a directory");
    // The worker's gpu_id is the nvidia-smi (physical, PCI-bus-ordered) index the admission budget is
    // read from; under `CUDA_VISIBLE_DEVICES` that is the visible device's index.
    let gpu_index = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|devices| devices.split(',').next().map(|d| d.trim().to_owned()))
        .filter(|device| !device.is_empty())
        .unwrap_or_else(|| "0".to_owned());
    let entry = shipped_iris_entry();
    let entry_object = entry.as_object().expect("catalog entry object");
    let tier = crate::vram_gate::requested_tier_key(&serde_json::Map::new(), entry_object, false);
    let resident_gb =
        crate::vram_gate::predicted_peak_gb(entry_object, tier).expect("declared resident floor");
    // Just under the resident floor: the resident load does not fit, the staged one does.
    let cap_gb = resident_gb - 0.5;
    let sequential_gb = crate::vram_gate::predicted_sequential_peak_gb(entry_object, tier)
        .expect("declared staged row");
    assert!(
        sequential_gb <= cap_gb,
        "the staged row ({sequential_gb} GB) must fit the {cap_gb} GB cap or the gate refuses"
    );
    assert_eq!(
        crate::vram_gate::load_plan(
            Some(resident_gb),
            Some(sequential_gb),
            Some(crate::vram_gate::VramBudget {
                free_gb: cap_gb,
                total_gb: cap_gb
            }),
            true
        ),
        crate::vram_gate::LoadPlan::Sequential,
        "the cap must route the job to staged residency"
    );
    let cap = format!("{cap_gb}");
    let _env = crate::test_env::EnvVars::set(&[(crate::vram_gate::CUDA_VRAM_CAP_ENV, &cap)]);

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    settings.gpu_id = gpu_index.clone();
    settings.backend_candle_enabled = true;
    settings.heartbeat_seconds = 30;
    let project = ProjectStore::new(settings.data_dir.clone(), "iris-smoke")
        .create_project("Iris-3B CUDA staged smoke")
        .expect("smoke project creates");
    let mut snapshot = stub_job_snapshot(STAGED_JOB);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": "a lighthouse on a rocky coast at dusk, warm window light",
        "width": STAGED_SIZE,
        "height": STAGED_SIZE,
        "count": 1,
        "seed": 3,
        "advanced": { "steps": STAGED_STEPS },
        "modelManifestEntry": entry,
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");

    let baseline_mib = nvidia_memory_used_mib(&gpu_index).expect("nvidia-smi memory.used reads");
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        let index = gpu_index.clone();
        std::thread::spawn(move || {
            let mut peak = 0_u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(used) = nvidia_memory_used_mib(&index) {
                    peak = peak.max(used);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            peak
        })
    };
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    let outcome = crate::image_jobs::run_image_generate_job(&api, &settings, &job).await;
    let elapsed = started.elapsed();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let peak_mib = sampler.join().expect("sampler thread joins");
    let gib = |mib: u64| mib as f64 / 1024.0;
    eprintln!(
        "RESULT model=iris_3b backend=candle residency=staged geometry={STAGED_SIZE}x{STAGED_SIZE} \
         steps={STAGED_STEPS} cap_gb={cap_gb} gpu={gpu_index} baseline_gb={:.2} peak_gb={:.2} \
         peak_delta_gb={:.2} elapsed_s={:.1} outcome={}",
        gib(baseline_mib),
        gib(peak_mib),
        gib(peak_mib.saturating_sub(baseline_mib)),
        elapsed.as_secs_f64(),
        match &outcome {
            Ok(()) => "completed".to_owned(),
            Err(error) => format!("failed: {error}"),
        },
    );
    outcome.expect("the staged Iris-3B CUDA job completes");
    let posts = posts.lock().expect("progress posts lock").clone();
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == STAGED_JOB && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the job posted a completed result");
    let asset = &result["assetWrites"][0];
    let path = PathBuf::from(&project.path).join(asset["mediaPath"].as_str().expect("media"));
    let decoded = image::open(&path).expect("open rendered asset").to_rgb8();
    assert_eq!(
        (decoded.width(), decoded.height()),
        (STAGED_SIZE, STAGED_SIZE)
    );
    let std = crate::smoke_support::image_std(&gen_core::Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    });
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the staged render looks degenerate (RGB std {std:.2})"
    );
    assert!(
        gib(peak_mib) < RESIDENT_PEAK_GB,
        "a staged run must peak below the {RESIDENT_PEAK_GB} GB resident peak (got {:.2} GB) — \
         the text encoder was not dropped before the denoise",
        gib(peak_mib)
    );
}

/// sc-25681: the NATIVE prompt batch on real weights — a 2-prompt × count-1 job is one
/// `candle-gen-iris` call (`prompt_batch`), published as two non-degenerate assets in prompt order
/// on seeds `seed` and `seed + 1`, each recording its own prompt and the batch. Runs in the same
/// `iris_worker_gpu_smoke` filter as the single-image smoke above on CUDA, and as the bounded MLX
/// real-weight check on a Mac (768², 20 steps).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots in the HF cache and a CUDA device or Apple GPU"]
async fn iris_worker_gpu_smoke_native_prompt_batch() {
    const BATCH_JOB: &str = "iris-cuda-native-batch";
    const BATCH_STEPS: u32 = 20;
    const BATCH_SIZE: u32 = 768;
    let prompts = [
        "a red fox in fresh snow, morning light",
        "a blue whale breaching at sunset",
    ];
    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    #[cfg(not(target_os = "macos"))]
    {
        settings.gpu_id = "cuda".to_owned();
        settings.backend_candle_enabled = true;
    }
    settings.heartbeat_seconds = 30;
    let project = ProjectStore::new(settings.data_dir.clone(), "iris-smoke")
        .create_project("Iris-3B native batch smoke")
        .expect("smoke project creates");
    let mut snapshot = stub_job_snapshot(BATCH_JOB);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": prompts[0],
        "prompts": prompts,
        "width": BATCH_SIZE,
        "height": BATCH_SIZE,
        "count": 1,
        "seed": 11,
        "advanced": { "steps": BATCH_STEPS },
        "modelManifestEntry": shipped_iris_entry(),
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::image_jobs::run_image_generate_job(&api, &settings, &job)
        .await
        .expect("the Iris-3B native batch job completes");
    let elapsed = started.elapsed();
    let posts = posts.lock().expect("progress posts lock").clone();
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == BATCH_JOB && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the job posted a completed result");
    let assets = result["assetWrites"].as_array().expect("asset writes");
    assert_eq!(assets.len(), 2, "one asset per prompt: {result}");
    let mut stds = Vec::new();
    for (index, asset) in assets.iter().enumerate() {
        assert_eq!(asset["prompt"], prompts[index]);
        assert_eq!(asset["seed"], 11 + index as i64);
        assert_eq!(asset["rawAdapterSettings"]["promptBatch"], json!(prompts));
        let path = PathBuf::from(&project.path).join(asset["mediaPath"].as_str().expect("media"));
        let decoded = image::open(&path).expect("open rendered asset").to_rgb8();
        assert_eq!(
            (decoded.width(), decoded.height()),
            (BATCH_SIZE, BATCH_SIZE)
        );
        let std = crate::smoke_support::image_std(&gen_core::Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        });
        assert!(
            std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
            "batch image {index} looks degenerate (RGB std {std:.2})"
        );
        stds.push(std);
    }
    eprintln!(
        "RESULT status=pass model=iris_3b native_batch=2x1 geometry={BATCH_SIZE}x{BATCH_SIZE} \
         steps={BATCH_STEPS} elapsed_s={:.1} rgb_std={stds:?}",
        elapsed.as_secs_f64(),
    );
}

/// Real-weight worker acceptance for Iris-3B RESTORATION on Candle/CUDA (sc-25683): the production
/// `image_upscale` dispatch with `engine: "iris"` — cache-only resolution of the task's own
/// `upscaler/` export (no text encoder), admission, the cached `candle-gen-iris` restore transform,
/// per-tile progress — for a multi-tile 4× (512x384 → 2048x1536, 6 tiles) and a 1× restore. The
/// `iris_worker_gpu_smoke` filter of the dispatch-only `iris-worker-smoke` job selects this too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned speridlabs/iris-3b upscaler/ export under HF_HUB_CACHE and a CUDA device or Apple GPU"]
async fn iris_worker_gpu_smoke_restore() {
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(hf_hub_cache.is_dir(), "HF_HUB_CACHE is not a directory");
    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    // sc-25690: the same restore jobs are the MLX real-weight check on a Mac (the worker's MLX
    // defaults), so only the off-Mac build pins the CUDA lane.
    #[cfg(not(target_os = "macos"))]
    {
        settings.gpu_id = "cuda".to_owned();
        settings.backend_candle_enabled = true;
    }
    settings.heartbeat_seconds = 30;
    let (export_dir, _) = crate::iris_restore::resolve_restore_export_dir(&settings)
        .expect("HF_HUB_CACHE must already hold the iris_3b_restore upscaler/ export");

    let store = ProjectStore::new(settings.data_dir.clone(), "iris-smoke");
    let project = store
        .create_project("Iris-3B CUDA restore smoke")
        .expect("smoke project creates");
    let source_file = temp.path().join("source.png");
    image::RgbImage::from_fn(512, 384, |x, y| {
        image::Rgb([
            (x % 256) as u8,
            (y % 256) as u8,
            ((x / 8 + y / 8) % 2 * 200) as u8,
        ])
    })
    .save(&source_file)
    .expect("source writes");
    let source = store
        .import_asset(
            &project.id,
            sceneworks_core::project_store::UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source_file,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("source imports");
    let source_id = source["id"].as_str().expect("asset id").to_owned();

    for (scale, expected, tiles) in [(4.0, (2048u32, 1536u32), 6usize), (1.0, (512, 384), 2)] {
        posts.lock().expect("progress posts lock").clear();
        let mut snapshot = stub_job_snapshot(JOB_ID);
        snapshot["type"] = json!("image_upscale");
        snapshot["payload"] = json!({
            "projectId": project.id,
            "sourceAssetId": source_id,
            "engine": "iris",
            "scale": scale,
        });
        let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");
        let api = ApiClient::new(&settings);
        let started = std::time::Instant::now();
        crate::upscale_jobs::run_image_upscale_job(&api, &settings, &job)
            .await
            .unwrap_or_else(|error| panic!("the Iris CUDA restore at {scale}x completes: {error}"));
        let elapsed = started.elapsed();
        let posts = posts.lock().expect("progress posts lock").clone();
        let tile_posts = posts
            .iter()
            .filter(|(_, body)| {
                body.get("message")
                    .and_then(Value::as_str)
                    .is_some_and(|message| message.starts_with("Restoring tile"))
            })
            .count();
        assert_eq!(tile_posts, tiles, "one progress post per planned tile");
        let result = posts
            .iter()
            .rev()
            .find(|(_, body)| body["status"] == "completed")
            .map(|(_, body)| body["result"].clone())
            .expect("the job posted a completed result");
        let asset = &result["assetWrites"][0];
        let path = PathBuf::from(&project.path).join(asset["mediaPath"].as_str().expect("path"));
        let decoded = image::open(&path).expect("restored asset opens").to_rgb8();
        assert_eq!(decoded.dimensions(), expected);
        let restored = gen_core::Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        };
        let std = crate::smoke_support::image_std(&restored);
        assert!(
            std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
            "the restored image looks degenerate (RGB std {std:.2})"
        );
        eprintln!(
            "RESULT status=pass model=iris_3b_restore backend={} scale={scale} geometry={}x{} \
             tiles={tiles} elapsed_s={:.1} rgb_std={std:.3} export={} output={}",
            if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "candle"
            },
            restored.width,
            restored.height,
            elapsed.as_secs_f64(),
            export_dir.display(),
            path.display(),
        );
    }
}
