//! Shared test support: imports, fixtures, and request helpers used across the
//! per-domain rust-api test modules. Split out of the former monolithic `tests.rs`
//! (sc-11217, F-030). Behavior-preserving move — no test logic changed.
#![allow(dead_code)]

pub(crate) use crate::auth::{loopback_trusted, requires_token};

pub(crate) use crate::events::{EventHub, EventMessage};

pub(crate) use crate::training::{
    insufficient_disk_space, resolve_base_model_path, training_base_model_installed,
    training_base_model_status, training_base_unavailable_message, TrainingBaseStatus,
};

pub(crate) use crate::workers::person_readiness_from_workers;

pub(crate) use crate::{
    create_app, create_app_with_state, huggingface_repo_cache_path, inject_converted_model_path,
    inprocess_utility_worker_id, inprocess_worker_gpu_id, lora_artifact_paths,
    merge_model_manifest_entry, mlx_catalog_status, open_bind_override_enabled,
    parse_inprocess_utility_worker_count, safe_download_dir, seed_mode_for_config_dir,
    serialize_job_lora, should_warn_open_bind, strip_jsonc_comments,
    sweep_stale_asset_uploads_before, sweep_stale_lora_uploads_before, sweep_stale_uploads,
    validate_model_id, Settings, WorkerCapability, WorkerSnapshot, WorkerStatus,
    API_MANAGED_MANIFEST_HEADER, DEFAULT_API_HOST, EVENT_BUFFER_SIZE, HEARTBEAT_SSE_DATA,
    HEARTBEAT_SSE_WIRE, TEST_MAX_LORA_UPLOAD_BYTES,
};

pub(crate) use axum::body::{to_bytes, Body};

pub(crate) use axum::http::{Request, StatusCode};

pub(crate) use serde_json::{json, Value};

pub(crate) use std::time::{Duration, SystemTime};

pub(crate) use tokio_stream::StreamExt;

pub(crate) use tower::ServiceExt;

pub(crate) const PNG_32X32: &[u8] = include_bytes!("../../../desktop/icons/32x32.png");

/// The single process-global lock every rust-api test that mutates HF-cache env serializes on.
/// rust-api's `#[test]`s run as threads in ONE binary, so a `set_var`/`remove_var` in one is visible
/// to all the others; mutual exclusion must therefore be crate-wide — a per-module lock serializes
/// nothing against the rest (mirrors `sceneworks_worker::test_env::ENV_LOCK`, sc-12380). Do NOT add a
/// second one.
static HF_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The HF-cache env vars neutralized (removed) for as long as this guard lives, restored on drop,
/// under [`HF_ENV_LOCK`]. `huggingface_repo_cache_path` reads `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`
/// / `HF_HOME` BEFORE `data_dir`, so a test that seeds a snapshot under a tempdir `data_dir` writes
/// into the developer's REAL cache unless these are cleared first — clobbering a real repo's
/// `refs/main` (the krea-2-raw-mlx pollution, sc-13834). The runner strips `HF_HOME` for `cargo test`,
/// but that does not cover ad-hoc / IDE runs; this makes the tempdir hermetic regardless. Restoring on
/// `Drop` (not around a closure) keeps a panicking assertion from leaking a cleared cache var into
/// every later test in the process.
#[must_use = "bind to a named `_env` local; dropping it immediately restores the env and pins nothing"]
pub(crate) struct HfCacheEnvGuard {
    restore: Vec<(&'static str, Option<String>)>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

/// Take [`HF_ENV_LOCK`] and clear the HF-cache env vars until the returned guard drops, so
/// `huggingface_repo_cache_path` / `resolve_base_model_path` resolve under the test's own `data_dir`
/// instead of the developer's real cache. See [`HfCacheEnvGuard`].
pub(crate) fn isolate_hf_cache() -> HfCacheEnvGuard {
    let guard = HF_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let restore = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"]
        .into_iter()
        .map(|key| {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
            (key, previous)
        })
        .collect();
    HfCacheEnvGuard {
        restore,
        _guard: guard,
    }
}

impl Drop for HfCacheEnvGuard {
    fn drop(&mut self) {
        for (key, previous) in &self.restore {
            match previous {
                Some(prior) => std::env::set_var(key, prior),
                None => std::env::remove_var(key),
            }
        }
    }
}

pub(crate) fn readiness_worker(
    id: &str,
    status: WorkerStatus,
    capabilities: Vec<WorkerCapability>,
) -> WorkerSnapshot {
    WorkerSnapshot {
        id: id.to_owned(),
        gpu_id: "0".to_owned(),
        gpu_name: None,
        status,
        current_job_id: None,
        capabilities,
        loaded_models: Vec::new(),
        utilization: None,
        registered_at: "2026-05-21T00:00:00Z".to_owned(),
        last_seen_at: "2026-05-21T00:00:00Z".to_owned(),
        extra: Default::default(),
    }
}

pub(crate) fn test_settings(temp_dir: &tempfile::TempDir) -> Settings {
    Settings {
        app_version: "test".to_owned(),
        host: "127.0.0.1".to_owned(),
        port: 0,
        data_dir: temp_dir.path().join("data"),
        config_dir: temp_dir.path().join("config"),
        access_token: String::new(),
        cors_origins: vec![
            "http://localhost:5173".to_owned(),
            "http://127.0.0.1:5173".to_owned(),
        ],
        worker_timeout_seconds: 90,
        jobs_db_path: temp_dir.path().join("jobs.db"),
        run_utility_inprocess: false,
        mlx_required: false,
        mlx_enforce_unsupported: false,
        candle_required: false,
        candle_enforce_unsupported: false,
        trust_loopback: false,
        // Placeholder for oneshot tests (the MCP self-client never dials it);
        // the live-listener MCP tests overwrite it with the bound address.
        mcp_api_url: "http://127.0.0.1:0".to_owned(),
        mcp_job_poll_interval: sceneworks_mcp::JobWaitConfig::default().poll_interval,
        mcp_job_timeout: sceneworks_mcp::JobWaitConfig::default().timeout,
        external_model_roots: Vec::new(),
        mcp_allowed_hosts_extra: Vec::new(),
    }
}

pub(crate) fn write_test_safetensors(path: &std::path::Path) {
    std::fs::write(path, test_safetensors_bytes()).expect("test safetensors writes");
}

pub(crate) fn write_test_safetensors_with_keys(path: &std::path::Path, tensor_keys: &[String]) {
    std::fs::write(path, test_safetensors_bytes_with_keys(tensor_keys))
        .expect("test safetensors writes");
}

pub(crate) fn test_safetensors_bytes() -> Vec<u8> {
    test_safetensors_bytes_with_keys(&[])
}

pub(crate) fn test_safetensors_bytes_with_keys(tensor_keys: &[String]) -> Vec<u8> {
    const TENSOR_DATA_END: u64 = 32768;
    let mut object = serde_json::Map::new();
    object.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    let mut data_end = 0_u64;
    for key in tensor_keys {
        object.insert(
            key.clone(),
            json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, TENSOR_DATA_END]}),
        );
        data_end = data_end.max(TENSOR_DATA_END);
    }
    let header = serde_json::to_vec(&Value::Object(object)).expect("header serializes");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    // The data section must hold every tensor the header declares — otherwise the
    // file is rejected as incomplete (sc-6072). Pad to the largest declared
    // `data_offsets` end; empty-key fixtures keep a few bytes so the file is a
    // non-empty but still complete safetensors.
    let data_len = usize::try_from(data_end.max(12)).expect("data length fits usize");
    bytes.resize(bytes.len() + data_len, 0);
    bytes
}

pub(crate) fn z_image_tensor_keys() -> Vec<String> {
    mm_dit_tensor_keys(24)
}

pub(crate) fn qwen_image_tensor_keys() -> Vec<String> {
    mm_dit_tensor_keys(60)
}

pub(crate) fn mm_dit_tensor_keys(block_count: usize) -> Vec<String> {
    let mut keys = Vec::new();
    for block in 0..block_count {
        for module in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "img_mlp.net.0.proj",
            "txt_mlp.net.0.proj",
        ] {
            keys.push(format!(
                "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
            ));
        }
    }
    keys
}

pub(crate) fn wan_video_tensor_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for block in 0..30 {
        for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
            keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
            keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
        }
    }
    keys
}

pub(crate) async fn request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
) -> (StatusCode, Value) {
    request_with_headers(app, method, uri, body, &[]).await
}

pub(crate) async fn request_with_headers(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let request = builder
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("response returns");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body parses")
    };
    (status, value)
}

/// Drive a request through the router with a simulated peer `SocketAddr` in the
/// request extensions, so the `Option<ConnectInfo<SocketAddr>>` extractor in the
/// auth middleware resolves to that peer (the plain `oneshot` path has no connect
/// info and so is never loopback-trusted). Used to exercise the epic-4484
/// loopback-trust bypass end-to-end (sc-8869).
pub(crate) async fn request_with_peer(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
    peer: &str,
) -> (StatusCode, Value) {
    use axum::extract::ConnectInfo;
    use std::net::SocketAddr;
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let addr: SocketAddr = peer.parse().expect("peer addr parses");
    request.extensions_mut().insert(ConnectInfo(addr));
    let response = app.oneshot(request).await.expect("response returns");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body parses")
    };
    (status, value)
}

/// Like `request_with_peer` but also attaches request headers (e.g. an
/// `authorization` token candidate), so the per-IP auth throttle (sc-8870) can be
/// exercised against the token oracle with a simulated remote peer.
pub(crate) async fn request_with_peer_headers(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
    peer: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, Value) {
    use axum::extract::ConnectInfo;
    use std::net::SocketAddr;
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let mut request = builder
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let addr: SocketAddr = peer.parse().expect("peer addr parses");
    request.extensions_mut().insert(ConnectInfo(addr));
    let response = app.oneshot(request).await.expect("response returns");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers");
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body parses")
    };
    (status, value)
}

pub(crate) async fn request_raw(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: impl Into<Body>,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let response = app
        .oneshot(builder.body(body.into()).expect("request builds"))
        .await
        .expect("response returns");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body buffers")
        .to_vec();
    (status, headers, bytes)
}

pub(crate) async fn request_multipart_upload(
    app: axum::Router,
    uri: &str,
    filename: &str,
    content_type: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_BOUNDARY";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        uri,
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

pub(crate) async fn request_multipart_lora_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    filename: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_LORA_BOUNDARY";
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
        app,
        "POST",
        "/api/v1/loras/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

// Collect the event names currently buffered for a subscriber, stopping after a brief
// quiet period (handlers publish synchronously before the request future resolves, so
// everything is already buffered by the time we drain).
pub(crate) async fn drain_event_names(
    events: &mut tokio_stream::wrappers::ReceiverStream<EventMessage>,
) -> Vec<String> {
    let mut names = Vec::new();
    while let Ok(Some(message)) =
        tokio::time::timeout(Duration::from_millis(150), events.next()).await
    {
        names.push(message.event);
    }
    names
}

/// Seeds the Z-Image-Turbo base model as installed (a managed-download
/// marker) so a real training run clears the missing-model guardrail.
pub(crate) fn seed_installed_base_model(data_dir: &std::path::Path) {
    // Seed the SceneWorks-managed install marker under the target's `base_model` id
    // (`models/z_image_turbo`), which `training_base_model_installed` checks as its final fallback
    // regardless of `base_model_repo`. Keying on the stable base_model (not the repo slug) keeps this
    // helper from breaking whenever a target's `base_model_repo` is re-homed — as it just was when
    // z_image_turbo moved off the flat `Tongyi-MAI/Z-Image-Turbo` upstream to its turnkey (sc-13860).
    let model_dir = data_dir
        .join("models")
        .join(safe_download_dir("z_image_turbo"));
    std::fs::create_dir_all(&model_dir).expect("model dir creates");
    std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("model marker writes");
}

/// Drives a project-scoped training job from submission to a completed result
/// and asserts the produced adapter is registered as a normal SceneWorks LoRA.
/// Seeds the base model so the real-run guardrails pass.
pub(crate) async fn submit_real_training_job(
    app: axum::Router,
    project_id: &str,
    data_dir: &std::path::Path,
) -> (String, std::path::PathBuf, std::path::PathBuf) {
    seed_installed_base_model(data_dir);
    let (_, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Portrait.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({
            "name": "Aurora set",
            "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
        }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    let target_id = registry["targets"][0]["id"]
        .as_str()
        .expect("target id")
        .to_owned();
    let mut config = registry["targets"][0]["defaults"].clone();
    // A trigger word flows from the config into the plan and the LoRA entry.
    config["triggerWord"] = json!("auroraStyle");

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": target_id,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Aurora Style",
            "dryRun": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = job["id"].as_str().expect("job id").to_owned();
    let output_dir = std::path::PathBuf::from(
        job["payload"]["plan"]["output"]["outputDir"]
            .as_str()
            .unwrap(),
    );
    let file_name = job["payload"]["plan"]["output"]["fileName"]
        .as_str()
        .unwrap()
        .to_owned();
    let adapter_path = output_dir.join(file_name);
    (job_id, output_dir, adapter_path)
}

/// Writes the final trained adapter into a training job's resolved output dir so
/// that — absent a gate — a `completed` report *would* register a LoRA. Returns
/// the adapter file name the registration would use.
pub(crate) fn stage_trained_adapter(output_dir: &std::path::Path, adapter_path: &std::path::Path) {
    std::fs::create_dir_all(output_dir).expect("output dir creates");
    write_test_safetensors(adapter_path);
}

/// Poll the job list until an Ideogram magic-prompt expansion (`prompt_refine`) job other than
/// `exclude` appears, and return its id. The image-job POST returns immediately (fully async,
/// sc-9120) and a background watcher enqueues this job, so it materializes promptly. `exclude` lets a
/// test wait for a *re-sampled* second job after completing the first.
pub(crate) async fn wait_for_prompt_refine_job_excluding(
    app: &axum::Router,
    exclude: Option<&str>,
) -> String {
    for _ in 0..250 {
        let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        if let Some(id) = jobs.as_array().and_then(|items| {
            items
                .iter()
                .filter(|job| job["type"] == "prompt_refine")
                .find_map(|job| job["id"].as_str().filter(|id| Some(*id) != exclude))
        }) {
            return id.to_owned();
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("the Ideogram magic-prompt prompt_refine job was never created");
}

/// Wait for the first Ideogram magic-prompt expansion job.
pub(crate) async fn wait_for_prompt_refine_job(app: &axum::Router) -> String {
    wait_for_prompt_refine_job_excluding(app, None).await
}

/// Complete a queued (unclaimed) magic-prompt job. The job has no owner, so the progress report omits
/// a workerId (matching the store's `(None, None)` ownership rule); `result` carries the model reply.
pub(crate) async fn complete_prompt_refine_job(app: &axum::Router, job_id: &str, result: Value) {
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Caption ready.",
            "result": result
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Either the image job left `pending_caption` (the watcher promoted it) or a new `prompt_refine`
/// re-sample appeared first. Lets a test drive the bounded re-sample loop without racing the two
/// async events (sc-9120).
pub(crate) enum PendingOrRefine {
    Promoted(Value),
    Refine(String),
}

/// Poll until EITHER the image job leaves `pending_caption` OR a `prompt_refine` job other than
/// `exclude_refine` appears — whichever happens first. Used by the bounded-resample degrade test to
/// feed each attempt a prose reply and then observe the eventual degrade to `queued` (sc-9120).
pub(crate) async fn wait_for_job_out_of_pending_caption_or_refine(
    app: &axum::Router,
    job_id: &str,
    exclude_refine: Option<&str>,
) -> PendingOrRefine {
    for _ in 0..250 {
        let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        let items = jobs.as_array().expect("jobs is an array");
        // Prefer promotion when the image job has already left pending_caption.
        if let Some(job) = items.iter().find(|job| job["id"] == job_id) {
            if job["status"] != "pending_caption" {
                return PendingOrRefine::Promoted(job.clone());
            }
        }
        if let Some(id) = items
            .iter()
            .filter(|job| job["type"] == "prompt_refine")
            .find_map(|job| job["id"].as_str().filter(|id| Some(*id) != exclude_refine))
        {
            return PendingOrRefine::Refine(id.to_owned());
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("neither a promotion nor a new refine job appeared for {job_id}");
}

/// Poll a job by id until it leaves `pending_caption` (the background caption watcher promoted it),
/// and return its final snapshot. Used by the fully-async Ideogram tests (sc-9120): the POST returns
/// immediately in `pending_caption`, and the prompt rewrite lands on a later async promotion.
pub(crate) async fn wait_for_job_out_of_pending_caption(app: &axum::Router, job_id: &str) -> Value {
    for _ in 0..250 {
        let (status, job) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        if job["status"] != "pending_caption" {
            return job;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("the pending_caption job {job_id} was never promoted out of pending_caption");
}

pub(crate) async fn request_multipart_lora_pair_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    primary: (&str, &[u8]),
    secondary: (&str, &[u8]),
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_LORA_PAIR_BOUNDARY";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    for (part_name, (filename, bytes)) in [("file", primary), ("secondaryFile", secondary)] {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{part_name}\"; filename=\"{filename}\"\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let (status, _, bytes) = request_raw(
        app,
        "POST",
        "/api/v1/loras/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

pub(crate) async fn request_multipart_model_upload(
    app: axum::Router,
    fields: &[(&str, &str)],
    filename: &str,
    bytes: &[u8],
) -> (StatusCode, Value) {
    let boundary = "SCENEWORKS_MODEL_BOUNDARY";
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
        app,
        "POST",
        "/api/v1/models/import",
        body,
        &[(
            "content-type",
            &format!("multipart/form-data; boundary={boundary}"),
        )],
    )
    .await;
    let value = serde_json::from_slice(&bytes).expect("json body parses");
    (status, value)
}

// Helper: write a minimal but COMPLETE set of weight-bearing components
// (text_encoder / transformer / vae, each with config.json + a weight file) into
// a diffusers snapshot dir, so a test can focus on a single weightless component.
pub(crate) fn write_complete_weight_bearing_components(cache_dir: &std::path::Path) {
    for dir in ["text_encoder", "transformer", "vae"] {
        let component_dir = cache_dir.join(dir);
        std::fs::create_dir_all(&component_dir).expect("component dir creates");
        std::fs::write(component_dir.join("config.json"), "{}").expect("config writes");
        std::fs::write(
            component_dir.join("diffusion_pytorch_model.safetensors"),
            "weights",
        )
        .expect("weights write");
    }
}

pub(crate) fn single_model_manifest(config_dir: &std::path::Path, id: &str, repo: &str) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        format!(
            r#"{{ "schemaVersion": 1, "models": [{{
                "id": "{id}", "name": "{id}", "type": "image", "family": "test",
                "downloads": [{{ "provider": "huggingface", "repo": "{repo}" }}]
            }}] }}"#
        ),
    )
    .expect("builtin models writes");
    for file in [
        "user.models.jsonc",
        "builtin.loras.jsonc",
        "user.loras.jsonc",
        "builtin.recipe-presets.jsonc",
        "user.recipe-presets.jsonc",
    ] {
        let key = if file.contains("preset") {
            "presets"
        } else if file.contains("lora") {
            "loras"
        } else {
            "models"
        };
        std::fs::write(
            config_dir.join(file),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty manifest writes");
    }
}

#[cfg(windows)]
pub(crate) fn create_test_symlink_file(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(unix)]
pub(crate) fn create_test_symlink_file(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Write the empty sibling manifests a `model_catalog` build requires, so a co-requisite test
/// (sc-9696) only has to author its own `builtin.models.jsonc`.
pub(crate) fn write_empty_sibling_manifests(config_dir: &std::path::Path) {
    for (name, key) in [
        ("user.models.jsonc", "models"),
        ("builtin.loras.jsonc", "loras"),
        ("user.loras.jsonc", "loras"),
        ("builtin.recipe-presets.jsonc", "presets"),
        ("user.recipe-presets.jsonc", "presets"),
    ] {
        std::fs::write(
            config_dir.join(name),
            format!("{{ \"schemaVersion\": 1, \"{key}\": [] }}"),
        )
        .expect("sibling manifest writes");
    }
}

/// Drive a request but read only the status line, dropping the body. Needed for the
/// SSE stream endpoint whose successful response body never ends (buffering it would
/// hang the test).
pub(crate) async fn request_status_only(app: axum::Router, method: &str, uri: &str) -> StatusCode {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .expect("request builds");
    app.oneshot(request)
        .await
        .expect("response returns")
        .status()
}

/// Status-only variant of `request_with_peer_headers`: rmcp's non-auth error
/// bodies (e.g. the 406 below) are plain text, so the JSON-parsing helpers
/// don't apply.
pub(crate) async fn mcp_status_with_peer(
    app: axum::Router,
    peer: &str,
    headers: &[(&str, &str)],
) -> StatusCode {
    use axum::extract::ConnectInfo;
    use std::net::SocketAddr;
    let mut builder = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        // Real HTTP/1.1 clients always send Host; the raw oneshot path doesn't,
        // and rmcp 400s a host-less request before the Accept check.
        .header("host", "127.0.0.1");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let mut request = builder
        .body(Body::from(json!({}).to_string()))
        .expect("request builds");
    let addr: SocketAddr = peer.parse().expect("peer addr parses");
    request.extensions_mut().insert(ConnectInfo(addr));
    let response = app.oneshot(request).await.expect("response returns");
    response.status()
}

pub(crate) fn mcp_tool_content_json(result: &rmcp::model::CallToolResult) -> Value {
    let text = result
        .content
        .first()
        .and_then(|block| block.as_text())
        .map(|text| text.text.as_str())
        .expect("tool result has one text content block");
    serde_json::from_str(text).expect("tool content is JSON")
}

/// A ComfyUI-style Wan adapter (keys taken from a real
/// `wan2.2_t2v_lightx2v_*.safetensors`) written at `path`, parent dirs included.
pub(crate) fn write_comfy_wan_adapter(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("adapter parent dir");
    }
    let keys: Vec<String> = [
        "diffusion_model.blocks.0.cross_attn.k.lora_down.weight",
        "diffusion_model.blocks.0.cross_attn.k.lora_up.weight",
        "diffusion_model.blocks.0.cross_attn.v.lora_down.weight",
        "diffusion_model.blocks.0.cross_attn.v.lora_up.weight",
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
        "diffusion_model.blocks.0.ffn.0.lora_down.weight",
        "diffusion_model.blocks.0.ffn.0.lora_up.weight",
    ]
    .iter()
    .map(|key| (*key).to_owned())
    .collect();
    write_test_safetensors_with_keys(path, &keys);
}
