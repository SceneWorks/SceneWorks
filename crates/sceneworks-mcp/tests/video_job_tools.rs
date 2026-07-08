//! Video job tool round-trip tests (sc-10235): a REAL rmcp streamable-HTTP
//! client drives the non-blocking submit/poll trio against a stub `/api/v1`
//! pipeline — `submit_video_job` (`POST /video/jobs`), `get_job_status`
//! (scripted `GET /jobs/:id`), and `get_job_result` (ticket minting via
//! `POST /files/ticket` → ticketed resource links, never inline bytes).
//! Covers the acceptance criteria: submit returns the job id + initial
//! snapshot, status polls report progress/stage/eta, terminal failure surfaces
//! the job error, a completed job hands back a ticketed URL another machine
//! can fetch, and the tools stay generic across image jobs.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::handler::client::ClientHandler;
use rmcp::model::{CallToolRequestParams, ClientInfo};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use sceneworks_mcp::{ApiClientConfig, JobWaitConfig};
use serde_json::{json, Value};

const TICKET: &str = "tkt0123456789abcdef";
const VIDEO_PATH: &str = "assets/videos/genset_1/clip_0001.mp4";
const IMAGE_PATH: &str = "assets/images/genset_1/img_0001.png";

/// Scripted `/api/v1` job pipeline for the non-blocking tools: the submit
/// records its body and returns a queued snapshot; each `GET /jobs/:id` steps
/// through `snapshots` (the last repeats); `POST /files/ticket` counts mints.
#[derive(Clone)]
struct StubState {
    submitted: Arc<Mutex<Vec<Value>>>,
    polls: Arc<Mutex<usize>>,
    snapshots: Arc<Vec<Value>>,
    tickets_minted: Arc<Mutex<usize>>,
}

fn snapshot(job_type: &str, status: &str, progress: f64, stage: &str, extra: Value) -> Value {
    let mut job = json!({
        "id": "job_v1",
        "type": job_type,
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

fn media_asset(id: &str, media_type: &str, path: &str, mime: &str) -> Value {
    // The persisted sidecar shape `persist_reported_assets` embeds in
    // `result.assets` — media path + mime live under `file`.
    json!({
        "id": id,
        "type": media_type,
        "file": { "path": path, "mimeType": mime }
    })
}

fn stub_api_router(state: StubState) -> Router {
    Router::new()
        .route(
            "/api/v1/video/jobs",
            post(
                |State(state): State<StubState>, Json(body): Json<Value>| async move {
                    state.submitted.lock().unwrap().push(body);
                    (
                        StatusCode::CREATED,
                        Json(snapshot(
                            "video_generate",
                            "queued",
                            0.0,
                            "queued",
                            json!({ "etaSeconds": 300 }),
                        )),
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
            "/api/v1/files/ticket",
            post(|State(state): State<StubState>| async move {
                *state.tickets_minted.lock().unwrap() += 1;
                Json(json!({ "ticket": TICKET, "expiresInSeconds": 600 }))
            }),
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

#[derive(Clone, Default)]
struct TestClient;

impl ClientHandler for TestClient {
    fn get_info(&self) -> ClientInfo {
        ClientInfo::default()
    }
}

struct Harness {
    client: rmcp::service::RunningService<rmcp::service::RoleClient, TestClient>,
    submitted: Arc<Mutex<Vec<Value>>>,
    tickets_minted: Arc<Mutex<usize>>,
    /// The base the client uses to reach `/mcp`. get_job_result derives the
    /// ticket URL host from the incoming request (sc-10290), so in this split
    /// harness (stub API on a different port than the mounted /mcp) the returned
    /// url is based on THIS, not the stub API base.
    mcp_base: String,
}

/// Stub API + mounted MCP service + connected client.
async fn harness(snapshots: Vec<Value>) -> Harness {
    let state = StubState {
        submitted: Arc::new(Mutex::new(Vec::new())),
        polls: Arc::new(Mutex::new(0)),
        snapshots: Arc::new(snapshots),
        tickets_minted: Arc::new(Mutex::new(0)),
    };
    let submitted = state.submitted.clone();
    let tickets_minted = state.tickets_minted.clone();
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
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("{mcp_base}/mcp")),
    );
    let client = TestClient
        .serve(transport)
        .await
        .expect("MCP client initializes against the mounted /mcp service");
    Harness {
        client,
        submitted,
        tickets_minted,
        mcp_base,
    }
}

/// The JSON payload of a tool result (its text content blocks parsed as JSON;
/// the last one wins — get_job_result appends the summary block last).
fn result_json(result: &rmcp::model::CallToolResult) -> Value {
    result
        .content
        .iter()
        .rev()
        .find_map(|block| block.as_text())
        .map(|text| serde_json::from_str(&text.text).expect("tool result text is JSON"))
        .expect("tool result carries a text block")
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
async fn submit_video_job_returns_job_id_and_initial_snapshot() {
    let harness = harness(vec![]).await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("submit_video_job").with_arguments(
                json!({
                    "projectId": "p1",
                    "prompt": "a lighthouse in a storm",
                    "sourceAssetId": "img_1",
                    "negativePrompt": "static",
                    "model": "ltx_2_3",
                    "duration": 8,
                    "fps": 24,
                    "width": 1280,
                    "height": 720,
                    "seed": 7
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
        )
        .await
        .expect("submit_video_job succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    // The tool returned the job id + initial snapshot and how to continue.
    let payload = result_json(&result);
    assert_eq!(payload["jobId"], "job_v1");
    assert_eq!(payload["status"], "queued");
    assert_eq!(payload["type"], "video_generate");
    assert_eq!(payload["etaSeconds"], 300);
    assert_eq!(payload["progressPercent"], 0);
    assert!(
        payload["next"]
            .as_str()
            .is_some_and(|next| next.contains("get_job_status")),
        "submit points at the polling tool: {payload}"
    );

    // The submit body carried the mapped VideoJobRequest fields — a start
    // image makes "generate" an image_to_video job.
    let submitted = harness.submitted.lock().unwrap().clone();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0]["mode"], "image_to_video");
    assert_eq!(submitted[0]["prompt"], "a lighthouse in a storm");
    assert_eq!(submitted[0]["sourceAssetId"], "img_1");
    assert_eq!(submitted[0]["negativePrompt"], "static");
    assert_eq!(submitted[0]["model"], "ltx_2_3");
    assert_eq!(submitted[0]["duration"], 8.0);
    assert_eq!(submitted[0]["fps"], 24);
    assert_eq!(submitted[0]["width"], 1280);
    assert_eq!(submitted[0]["height"], 720);
    assert_eq!(submitted[0]["seed"], 7);

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn submit_video_job_rejects_incomplete_mode_inputs_before_submitting() {
    let harness = harness(vec![]).await;

    // bridge without its right clip must fail fast, client-side.
    let outcome = harness
        .client
        .call_tool(
            CallToolRequestParams::new("submit_video_job").with_arguments(
                json!({
                    "projectId": "p1",
                    "prompt": "bridge these",
                    "mode": "bridge",
                    "sourceClipAssetId": "clip_left"
                })
                .as_object()
                .unwrap()
                .clone(),
            ),
        )
        .await;
    match outcome {
        Err(_) => {}
        Ok(result) => assert_eq!(
            result.is_error,
            Some(true),
            "an incomplete bridge call must not look like success: {result:?}"
        ),
    }
    assert!(
        harness.submitted.lock().unwrap().is_empty(),
        "no job may be submitted for invalid mode inputs"
    );

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_status_reports_progress_stage_and_eta() {
    let harness = harness(vec![snapshot(
        "video_generate",
        "running",
        0.4,
        "generating",
        json!({ "message": "step 16/40", "etaSeconds": 95, "elapsedSeconds": 63 }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_status")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
        )
        .await
        .expect("get_job_status succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    let payload = result_json(&result);
    assert_eq!(payload["jobId"], "job_v1");
    assert_eq!(payload["status"], "running");
    assert_eq!(payload["stage"], "generating");
    assert_eq!(payload["message"], "step 16/40");
    assert_eq!(payload["progressPercent"], 40);
    assert_eq!(payload["etaSeconds"], 95);
    assert_eq!(payload["elapsedSeconds"], 63);

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_status_surfaces_the_failed_job_error() {
    let harness = harness(vec![snapshot(
        "video_generate",
        "failed",
        0.4,
        "failed",
        json!({ "error": "CUDA out of memory on gpu0" }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_status")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
        )
        .await
        .expect("get_job_status succeeds");
    assert_ne!(
        result.is_error,
        Some(true),
        "status is a report: {result:?}"
    );

    let payload = result_json(&result);
    assert_eq!(payload["status"], "failed");
    assert_eq!(payload["error"], "CUDA out of memory on gpu0");

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_result_returns_ticketed_video_link_without_inline_bytes() {
    let harness = harness(vec![snapshot(
        "video_generate",
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [
            media_asset("vid_1", "video", VIDEO_PATH, "video/mp4"),
        ] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_result")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
        )
        .await
        .expect("get_job_result succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    // A resource link (never inline bytes) whose URL is the absolute ticketed
    // media URL. The host is derived from the incoming MCP request (sc-10290), so
    // it is the base the client used to reach /mcp — in production /mcp and
    // /api/v1 are the same app, but this split test harness proves the derivation
    // by using `mcp_base` (a different port than the stub `api_base`).
    let expected_url = format!(
        "{}/api/v1/projects/p1/files/{VIDEO_PATH}?ticket={TICKET}",
        harness.mcp_base
    );
    let links: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_resource_link())
        .collect();
    assert_eq!(links.len(), 1, "one video link: {result:?}");
    assert_eq!(links[0].uri, expected_url);
    assert_eq!(links[0].mime_type.as_deref(), Some("video/mp4"));
    assert!(
        !result
            .content
            .iter()
            .any(|block| block.as_image().is_some()),
        "get_job_result must never inline media bytes: {result:?}"
    );

    let payload = result_json(&result);
    assert_eq!(payload["jobId"], "job_v1");
    assert_eq!(payload["projectId"], "p1");
    assert_eq!(payload["assets"][0]["id"], "vid_1");
    assert_eq!(payload["assets"][0]["type"], "video");
    assert_eq!(payload["assets"][0]["url"], Value::String(expected_url));
    assert_eq!(
        payload["assets"][0]["relativeUrl"],
        format!("/api/v1/projects/p1/files/{VIDEO_PATH}?ticket={TICKET}")
    );
    assert_eq!(payload["ticketExpiresInSeconds"], 600);

    // Exactly one ticket mint covered the link set.
    assert_eq!(*harness.tickets_minted.lock().unwrap(), 1);

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_result_is_generic_over_image_jobs() {
    let harness = harness(vec![snapshot(
        "image_generate",
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [
            media_asset("asset_1", "image", IMAGE_PATH, "image/png"),
        ] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_result")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
        )
        .await
        .expect("get_job_result succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    let links: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_resource_link())
        .collect();
    assert_eq!(links.len(), 1, "one image link: {result:?}");
    assert!(links[0].uri.contains(IMAGE_PATH), "{}", links[0].uri);
    assert!(links[0].uri.contains(TICKET), "{}", links[0].uri);
    assert_eq!(links[0].mime_type.as_deref(), Some("image/png"));

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_result_on_a_running_job_reports_not_ready_without_error() {
    let harness = harness(vec![snapshot(
        "video_generate",
        "running",
        0.6,
        "generating",
        json!({ "etaSeconds": 40 }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_result")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
        )
        .await
        .expect("get_job_result succeeds");
    assert_ne!(
        result.is_error,
        Some(true),
        "a not-ready job is a report, not an error: {result:?}"
    );

    let payload = result_json(&result);
    assert_eq!(payload["ready"], false);
    assert_eq!(payload["status"], "running");
    assert_eq!(payload["progressPercent"], 60);
    assert!(
        payload["note"]
            .as_str()
            .is_some_and(|note| note.contains("get_job_status")),
        "not-ready points back at polling: {payload}"
    );
    assert_eq!(
        *harness.tickets_minted.lock().unwrap(),
        0,
        "no ticket may be minted for an unfinished job"
    );

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn get_job_result_on_a_failed_job_surfaces_the_error() {
    let harness = harness(vec![snapshot(
        "video_generate",
        "failed",
        0.2,
        "failed",
        json!({ "error": "model weights missing: ltx_2_3" }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("get_job_result")
                .with_arguments(json!({ "jobId": "job_v1" }).as_object().unwrap().clone()),
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
        text.contains("model weights missing: ltx_2_3"),
        "error must carry the job's error message: {text}"
    );
    assert!(text.contains("job_v1"), "error names the job: {text}");

    let _ = harness.client.cancel().await;
}
