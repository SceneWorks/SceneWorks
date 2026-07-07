//! generate_image round-trip tests (sc-10234): a REAL rmcp streamable-HTTP
//! client calls the blocking tool against a stub `/api/v1` job pipeline —
//! submit (`POST /image/jobs`) → scripted `GET /jobs/:id` polls → media bytes
//! from `GET /projects/:id/files/*`. Covers the acceptance criteria end to end:
//! inline base64 image results (all of them for `count > 1`), mid-call progress
//! notifications on a client-supplied progressToken, and clear errors (never a
//! hang) for failed / canceled / stuck jobs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use rmcp::handler::client::ClientHandler;
use rmcp::model::{
    CallToolRequestParams, ClientInfo, Meta, NumberOrString, ProgressNotificationParam,
    ProgressToken,
};
use rmcp::service::{NotificationContext, RoleClient};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use sceneworks_mcp::{ApiClientConfig, JobWaitConfig};
use serde_json::{json, Value};

const PNG_BYTES: &[u8] = b"fake-png-payload-0001";
const JPG_BYTES: &[u8] = b"fake-jpeg-payload-0002";
const PNG_PATH: &str = "assets/images/genset_1/img_0001.png";
const JPG_PATH: &str = "assets/images/genset_1/img_0002.jpg";

/// Scripted `/api/v1` job pipeline: the submit returns a queued JobSnapshot,
/// then each `GET /jobs/:id` poll steps through `snapshots` (the last repeats,
/// so a "stuck" script of `[running]` polls forever).
#[derive(Clone)]
struct StubState {
    submitted: Arc<Mutex<Vec<Value>>>,
    polls: Arc<Mutex<usize>>,
    snapshots: Arc<Vec<Value>>,
}

fn snapshot(status: &str, progress: f64, stage: &str, extra: Value) -> Value {
    let mut job = json!({
        "id": "job-1",
        "type": "image_generate",
        "status": status,
        "projectId": "p1",
        "progress": progress,
        "stage": stage,
        "message": "",
        "error": null,
        "result": {}
    });
    if let (Some(job_obj), Some(extra_obj)) = (job.as_object_mut(), extra.as_object()) {
        for (key, value) in extra_obj {
            job_obj.insert(key.clone(), value.clone());
        }
    }
    job
}

fn image_asset(id: &str, path: &str, mime: &str) -> Value {
    // The persisted sidecar shape `persist_reported_assets` embeds in
    // `result.assets` — media path + mime live under `file`.
    json!({
        "id": id,
        "type": "image",
        "file": { "path": path, "mimeType": mime }
    })
}

fn stub_api_router(state: StubState) -> Router {
    Router::new()
        .route(
            "/api/v1/image/jobs",
            post(
                |State(state): State<StubState>, Json(body): Json<Value>| async move {
                    state.submitted.lock().unwrap().push(body);
                    (
                        StatusCode::CREATED,
                        Json(snapshot("queued", 0.0, "queued", json!({}))),
                    )
                },
            ),
        )
        .route(
            "/api/v1/jobs/:job_id",
            get(
                |State(state): State<StubState>, Path(_job_id): Path<String>| async move {
                    let index = {
                        let mut polls = state.polls.lock().unwrap();
                        let index = *polls;
                        *polls += 1;
                        index
                    };
                    let clamped = index.min(state.snapshots.len() - 1);
                    Json(state.snapshots[clamped].clone())
                },
            ),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(
                |Path((_project_id, relative_path)): Path<(String, String)>| async move {
                    let (bytes, mime) = match relative_path.as_str() {
                        PNG_PATH => (PNG_BYTES, "image/png"),
                        JPG_PATH => (JPG_BYTES, "image/jpeg"),
                        _ => return Err(StatusCode::NOT_FOUND),
                    };
                    let mut headers = HeaderMap::new();
                    headers.insert(header::CONTENT_TYPE, mime.parse().unwrap());
                    Ok((headers, bytes.to_vec()))
                },
            ),
        )
        .with_state(state)
}

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub listener");
    let addr = listener.local_addr().expect("stub addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

/// A minimal MCP client handler that records every progress notification the
/// server pushes mid-call.
#[derive(Clone, Default)]
struct RecordingClient {
    progress: Arc<Mutex<Vec<ProgressNotificationParam>>>,
}

impl ClientHandler for RecordingClient {
    fn get_info(&self) -> ClientInfo {
        ClientInfo::default()
    }

    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        self.progress.lock().unwrap().push(params);
        std::future::ready(())
    }
}

struct Harness {
    client: rmcp::service::RunningService<RoleClient, RecordingClient>,
    submitted: Arc<Mutex<Vec<Value>>>,
    polls: Arc<Mutex<usize>>,
    progress: Arc<Mutex<Vec<ProgressNotificationParam>>>,
}

/// Stub API + mounted MCP service (fast 10ms polls) + connected recording client.
async fn harness(snapshots: Vec<Value>) -> Harness {
    let state = StubState {
        submitted: Arc::new(Mutex::new(Vec::new())),
        polls: Arc::new(Mutex::new(0)),
        snapshots: Arc::new(snapshots),
    };
    let submitted = state.submitted.clone();
    let polls = state.polls.clone();
    let api_base = spawn(stub_api_router(state)).await;
    let mcp_service = sceneworks_mcp::streamable_http_service_with(
        ApiClientConfig {
            base_url: api_base,
            access_token: None,
        },
        JobWaitConfig {
            poll_interval: Duration::from_millis(10),
            timeout: Duration::from_secs(10),
        },
    );
    let mcp_base = spawn(Router::new().nest_service("/mcp", mcp_service)).await;

    let handler = RecordingClient::default();
    let progress = handler.progress.clone();
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("{mcp_base}/mcp")),
    );
    let client = handler
        .serve(transport)
        .await
        .expect("MCP client initializes against the mounted /mcp service");
    Harness {
        client,
        submitted,
        polls,
        progress,
    }
}

fn generate_args(extra: Value) -> serde_json::Map<String, Value> {
    let mut args = json!({ "projectId": "p1", "prompt": "a city at night" });
    if let (Some(args_obj), Some(extra_obj)) = (args.as_object_mut(), extra.as_object()) {
        for (key, value) in extra_obj {
            args_obj.insert(key.clone(), value.clone());
        }
    }
    args.as_object().expect("args are an object").clone()
}

fn call_with_progress_token(args: serde_json::Map<String, Value>) -> CallToolRequestParams {
    let mut params = CallToolRequestParams::new("generate_image").with_arguments(args);
    params.meta = Some(Meta::with_progress_token(ProgressToken(
        NumberOrString::String("progress-tok-1".into()),
    )));
    params
}

fn error_text(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| block.as_text())
        .map(|text| text.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn happy_path_returns_inline_image_with_progress_notifications() {
    let harness = harness(vec![
        snapshot("queued", 0.0, "queued", json!({})),
        snapshot(
            "running",
            0.5,
            "generating",
            json!({ "message": "step 4/8" }),
        ),
        snapshot(
            "completed",
            1.0,
            "completed",
            json!({ "result": { "assets": [image_asset("asset_1", PNG_PATH, "image/png")],
                                "assetIds": ["asset_1"] } }),
        ),
    ])
    .await;

    let result = harness
        .client
        .call_tool(call_with_progress_token(generate_args(json!({
            "negativePrompt": "blurry",
            "model": "z_image_turbo",
            "seed": 7,
            "width": 1280,
            "height": 768
        }))))
        .await
        .expect("generate_image succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    // Exactly one inline image + the trailing JSON summary block.
    let images: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_image())
        .collect();
    assert_eq!(images.len(), 1, "one generated image: {result:?}");
    assert_eq!(images[0].data, BASE64.encode(PNG_BYTES));
    assert_eq!(images[0].mime_type, "image/png");
    let summary: Value = serde_json::from_str(
        &result
            .content
            .iter()
            .rev()
            .find_map(|block| block.as_text())
            .expect("summary text block")
            .text,
    )
    .expect("summary is JSON");
    assert_eq!(summary["jobId"], "job-1");
    assert_eq!(summary["assets"][0]["id"], "asset_1");
    assert_eq!(summary["assets"][0]["path"], PNG_PATH);

    // The submit body carried the mapped ImageJobRequest fields. (Clone out of
    // the lock: guards must not be held across the cancel().await below.)
    let submitted = harness.submitted.lock().unwrap().clone();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0]["mode"], "text_to_image");
    assert_eq!(submitted[0]["prompt"], "a city at night");
    assert_eq!(submitted[0]["negativePrompt"], "blurry");
    assert_eq!(submitted[0]["model"], "z_image_turbo");
    assert_eq!(submitted[0]["seed"], 7);
    assert_eq!(submitted[0]["width"], 1280);
    assert_eq!(submitted[0]["height"], 768);
    assert_eq!(submitted[0]["count"], 1);

    // The tool actually polled to terminal (queued → running → completed).
    assert!(*harness.polls.lock().unwrap() >= 3, "polled to terminal");

    // Progress was observable mid-call on the supplied token, ending at 100%.
    let progress = harness.progress.lock().unwrap().clone();
    assert!(
        progress.len() >= 2,
        "expected mid-call progress notifications: {progress:?}"
    );
    // NOTE: rmcp's client layer (send_request_with_option) overwrites any
    // caller-set progressToken with its own generated one, so we assert the
    // notifications all ride ONE request token rather than a literal value.
    let token = &progress[0].progress_token;
    assert!(progress
        .iter()
        .all(|notification| notification.progress_token == *token));
    assert!(progress
        .iter()
        .any(|notification| notification.message.as_deref() == Some("generating: step 4/8")));
    assert_eq!(progress.last().unwrap().progress, 100.0);
    assert_eq!(progress.last().unwrap().total, Some(100.0));

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn count_greater_than_one_returns_every_image() {
    let harness = harness(vec![
        snapshot("running", 0.5, "generating", json!({})),
        snapshot(
            "completed",
            1.0,
            "completed",
            json!({ "result": { "assets": [
                image_asset("asset_1", PNG_PATH, "image/png"),
                image_asset("asset_2", JPG_PATH, "image/jpeg"),
            ] } }),
        ),
    ])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image")
                .with_arguments(generate_args(json!({ "count": 2 }))),
        )
        .await
        .expect("generate_image succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    let images: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_image())
        .collect();
    assert_eq!(images.len(), 2, "count=2 returns both images: {result:?}");
    assert_eq!(images[0].data, BASE64.encode(PNG_BYTES));
    assert_eq!(images[0].mime_type, "image/png");
    assert_eq!(images[1].data, BASE64.encode(JPG_BYTES));
    assert_eq!(images[1].mime_type, "image/jpeg");

    assert_eq!(harness.submitted.lock().unwrap()[0]["count"], 2);

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn failed_job_surfaces_the_worker_error_message() {
    let harness = harness(vec![
        snapshot("running", 0.2, "loading_model", json!({})),
        snapshot(
            "failed",
            0.2,
            "failed",
            json!({ "error": "CUDA out of memory on gpu0" }),
        ),
    ])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("tool call transports (the failure is a tool-level error)");
    assert_eq!(
        result.is_error,
        Some(true),
        "failed job must error: {result:?}"
    );
    let text = error_text(&result);
    assert!(
        text.contains("CUDA out of memory on gpu0"),
        "error must carry the job's error message: {text}"
    );
    assert!(text.contains("job-1"), "error names the job: {text}");

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn canceled_job_surfaces_clearly_not_as_a_hang() {
    let harness = harness(vec![snapshot("canceled", 0.0, "canceled", json!({}))]).await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("tool call transports");
    assert_eq!(
        result.is_error,
        Some(true),
        "canceled job must error: {result:?}"
    );
    assert!(
        error_text(&result).contains("canceled"),
        "error must say the job was canceled: {result:?}"
    );

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn stuck_job_times_out_with_a_clear_error_instead_of_hanging() {
    // The script never leaves `running`; the (test-shortened) overall deadline
    // must turn that into a clear tool error, not an endless poll.
    let state = StubState {
        submitted: Arc::new(Mutex::new(Vec::new())),
        polls: Arc::new(Mutex::new(0)),
        snapshots: Arc::new(vec![snapshot("running", 0.5, "generating", json!({}))]),
    };
    let api_base = spawn(stub_api_router(state)).await;
    let mcp_service = sceneworks_mcp::streamable_http_service_with(
        ApiClientConfig {
            base_url: api_base,
            access_token: None,
        },
        JobWaitConfig {
            poll_interval: Duration::from_millis(10),
            timeout: Duration::from_millis(100),
        },
    );
    let mcp_base = spawn(Router::new().nest_service("/mcp", mcp_service)).await;
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("{mcp_base}/mcp")),
    );
    let client = RecordingClient::default()
        .serve(transport)
        .await
        .expect("MCP client initializes");

    let result = client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("tool call transports");
    assert_eq!(
        result.is_error,
        Some(true),
        "stuck job must time out: {result:?}"
    );
    let text = error_text(&result);
    assert!(
        text.contains("did not reach a terminal state"),
        "timeout must be explicit: {text}"
    );

    let _ = client.cancel().await;
}

#[tokio::test]
async fn edit_image_mode_maps_and_threads_the_source_asset() {
    let harness = harness(vec![snapshot(
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [image_asset("asset_1", PNG_PATH, "image/png")] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({
                "mode": "edit_image",
                "sourceAssetId": "asset_src",
                "maskAssetId": "asset_mask"
            }))),
        )
        .await
        .expect("generate_image succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    let submitted = harness.submitted.lock().unwrap().clone();
    assert_eq!(submitted[0]["mode"], "edit_image");
    assert_eq!(submitted[0]["sourceAssetId"], "asset_src");
    assert_eq!(submitted[0]["maskAssetId"], "asset_mask");

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn invalid_mode_is_rejected_before_submitting_a_job() {
    let harness = harness(vec![snapshot("queued", 0.0, "queued", json!({}))]).await;

    let outcome = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image")
                .with_arguments(generate_args(json!({ "mode": "style_variations" }))),
        )
        .await;
    match outcome {
        Err(_) => {}
        Ok(result) => assert_eq!(
            result.is_error,
            Some(true),
            "an unsupported mode must not look like success: {result:?}"
        ),
    }
    assert!(
        harness.submitted.lock().unwrap().is_empty(),
        "no job may be submitted for an invalid mode"
    );

    let _ = harness.client.cancel().await;
}
