//! generate_image round-trip tests (sc-10234): a REAL rmcp streamable-HTTP
//! client calls the blocking tool against a stub `/api/v1` job pipeline —
//! submit (`POST /image/jobs`) → scripted `GET /jobs/:id` polls → media bytes
//! from `GET /projects/:id/files/*`. Covers the acceptance criteria end to end:
//! inline base64 image results (all of them for `count > 1`), mid-call progress
//! notifications on a client-supplied progressToken, and clear errors (never a
//! hang) for failed / canceled / stuck jobs.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{convert::Infallible, iter};

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::{stream, StreamExt};
use rmcp::handler::client::ClientHandler;
use rmcp::model::{
    CallToolRequestParams, ClientInfo, ClientRequest, Meta, NumberOrString,
    ProgressNotificationParam, ProgressToken, Request,
};
use rmcp::service::{NotificationContext, PeerRequestOptions, RoleClient};
use sceneworks_mcp::JobWaitConfig;
use serde_json::{json, Value};

use common::{connect_mcp, error_text, fast_job_wait, spawn};

const PNG_BYTES: &[u8] = b"fake-png-payload-0001";
const JPG_BYTES: &[u8] = b"fake-jpeg-payload-0002";
const PNG_PATH: &str = "assets/images/genset_1/img_0001.png";
const WORKFLOW_PNG_PATH: &str = "assets/images/genset_1/img_workflow.png";
const JPG_PATH: &str = "assets/images/genset_1/img_0002.jpg";
// F-041 (sc-11236): an asset that exceeds the per-image inline cap (4 MiB), so
// generate_image must fall back to the ticketed-link response shape instead of
// base64-inlining it.
const LARGE_PATH: &str = "assets/images/genset_1/img_large.png";
const CHUNKED_LARGE_PATH: &str = "assets/images/genset_1/img_chunked_large.png";
const LARGE_IMAGE_LEN: usize = 5 * 1024 * 1024;
const TICKET: &str = "tkt-abc123";

/// Scripted `/api/v1` job pipeline: the submit returns a queued JobSnapshot,
/// then each `GET /jobs/:id` poll steps through `snapshots` (the last repeats,
/// so a "stuck" script of `[running]` polls forever).
#[derive(Clone)]
struct StubState {
    submitted: Arc<Mutex<Vec<Value>>>,
    polls: Arc<Mutex<usize>>,
    snapshots: Arc<Vec<Value>>,
    /// Job ids the tool asked to cancel via `POST /jobs/:id/cancel` (sc-10276).
    cancels: Arc<Mutex<Vec<String>>>,
    workflow_png: Arc<Vec<u8>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectFileQuery {
    strip_workflow: Option<bool>,
}

fn workflow_png_bytes() -> Vec<u8> {
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("workflow_share")
        .join("image-workflow-share.json");
    let share: sceneworks_core::workflow_share::WorkflowShare =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("reads workflow fixture"))
            .expect("parses workflow fixture");
    let temp_dir = tempfile::tempdir().expect("creates temp dir");
    let path = temp_dir.path().join("workflow.png");
    sceneworks_core::workflow_png::write_workflow_chunk(
        &image::RgbImage::new(2, 2),
        &path,
        Some(&share),
    )
    .expect("writes embedded workflow PNG");
    std::fs::read(path).expect("reads embedded workflow PNG")
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
            "/api/v1/jobs/:job_id/cancel",
            post(
                |State(state): State<StubState>, Path(job_id): Path<String>| async move {
                    state.cancels.lock().unwrap().push(job_id);
                    Json(json!({ "status": "canceled" }))
                },
            ),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(
                |State(state): State<StubState>,
                 Path((_project_id, relative_path)): Path<(String, String)>,
                 Query(query): Query<ProjectFileQuery>| async move {
                    if relative_path == CHUNKED_LARGE_PATH {
                        // No Content-Length and no end-of-stream: the client can return only by
                        // enforcing the inline cap while consuming chunks and dropping the body.
                        let chunks = stream::iter(
                            iter::repeat_with(|| {
                                Ok::<_, Infallible>(Bytes::from(vec![0x43; 1024 * 1024]))
                            })
                            .take(5),
                        )
                        .chain(stream::pending());
                        return (
                            [(header::CONTENT_TYPE, "image/png")],
                            Body::from_stream(chunks),
                        )
                            .into_response();
                    }
                    let (bytes, mime): (Vec<u8>, &str) = match relative_path.as_str() {
                        PNG_PATH => (PNG_BYTES.to_vec(), "image/png"),
                        WORKFLOW_PNG_PATH => {
                            let source = state.workflow_png.as_ref();
                            let bytes = if query.strip_workflow.unwrap_or(false) {
                                sceneworks_core::workflow_png::strip_workflow_chunk(source)
                                    .expect("stub uses the hardened strip helper")
                                    .unwrap_or_else(|| source.clone())
                            } else {
                                source.clone()
                            };
                            (bytes, "image/png")
                        }
                        JPG_PATH => (JPG_BYTES.to_vec(), "image/jpeg"),
                        LARGE_PATH => (vec![0x42u8; LARGE_IMAGE_LEN], "image/png"),
                        _ => return StatusCode::NOT_FOUND.into_response(),
                    };
                    let mut headers = HeaderMap::new();
                    headers.insert(header::CONTENT_TYPE, mime.parse().unwrap());
                    (headers, bytes).into_response()
                },
            ),
        )
        // F-041: the oversize-payload fallback mints a media ticket for its links.
        .route(
            "/api/v1/files/ticket",
            post(|| async { Json(json!({ "ticket": TICKET, "expiresInSeconds": 600 })) }),
        )
        .with_state(state)
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
    cancels: Arc<Mutex<Vec<String>>>,
}

/// Stub API + mounted MCP service (fast 10ms polls) + connected recording client.
async fn harness(snapshots: Vec<Value>) -> Harness {
    let state = StubState {
        submitted: Arc::new(Mutex::new(Vec::new())),
        polls: Arc::new(Mutex::new(0)),
        snapshots: Arc::new(snapshots),
        cancels: Arc::new(Mutex::new(Vec::new())),
        workflow_png: Arc::new(workflow_png_bytes()),
    };
    let submitted = state.submitted.clone();
    let polls = state.polls.clone();
    let cancels = state.cancels.clone();
    let api_base = spawn(stub_api_router(state)).await;

    let handler = RecordingClient::default();
    let progress = handler.progress.clone();
    let (_, client) = connect_mcp(api_base, None, fast_job_wait(), handler).await;
    Harness {
        client,
        submitted,
        polls,
        progress,
        cancels,
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

/// Poll `condition` until it holds, failing (never hanging) if it never does.
async fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition was not met within 5s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
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
    assert_eq!(summary["assets"][0]["workflowPolicy"], "strip-requested");
    assert!(summary.get("workflowIncluded").is_none());
    assert!(summary.get("workflowHandling").is_none());

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
async fn inline_png_is_stripped_of_its_workflow_by_default() {
    let harness = harness(vec![snapshot(
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [image_asset(
            "asset_workflow",
            WORKFLOW_PNG_PATH,
            "image/png"
        )] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("generate_image succeeds");
    let image = result
        .content
        .iter()
        .find_map(|block| block.as_image())
        .expect("inline image");
    let served = BASE64.decode(&image.data).expect("valid base64 image");

    assert_eq!(
        sceneworks_core::workflow_png::read_workflow_chunk(&served),
        Ok(None),
        "MCP's default inline bytes must not carry the workflow envelope"
    );

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn inline_png_can_include_its_workflow_by_explicit_request() {
    let harness = harness(vec![snapshot(
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [image_asset(
            "asset_workflow",
            WORKFLOW_PNG_PATH,
            "image/png"
        )] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({
                "includeWorkflow": true
            }))),
        )
        .await
        .expect("generate_image succeeds");
    let image = result
        .content
        .iter()
        .find_map(|block| block.as_image())
        .expect("inline image");
    let served = BASE64.decode(&image.data).expect("valid base64 image");

    assert!(
        matches!(
            sceneworks_core::workflow_png::read_workflow_chunk(&served),
            Ok(Some(_))
        ),
        "includeWorkflow=true must preserve the workflow envelope"
    );
    let summary: Value = serde_json::from_str(
        &result
            .content
            .iter()
            .rev()
            .find_map(|block| block.as_text())
            .expect("summary block")
            .text,
    )
    .expect("summary JSON");
    assert_eq!(
        summary["assets"][0]["workflowPolicy"],
        "preserve-if-present"
    );

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
async fn client_cancellation_propagates_to_the_job_cancel_route() {
    // The script never leaves "running", so the job would run forever on its own;
    // the ONLY way the tool returns is the client canceling the in-flight request
    // (sc-10276). That cancellation must reach POST /api/v1/jobs/:id/cancel.
    let harness = harness(vec![snapshot("running", 0.4, "generating", json!({}))]).await;

    let handle = harness
        .client
        .send_cancellable_request(
            ClientRequest::CallToolRequest(Request::new(
                CallToolRequestParams::new("generate_image")
                    .with_arguments(generate_args(json!({}))),
            )),
            PeerRequestOptions::no_options(),
        )
        .await
        .expect("cancellable generate_image request is sent");

    // Let the tool get past submit and into its poll loop, so the cancellation
    // lands mid-wait (the case the story is about), not before the job exists.
    wait_until(|| {
        !harness.submitted.lock().unwrap().is_empty() && *harness.polls.lock().unwrap() >= 1
    })
    .await;
    assert!(
        harness.cancels.lock().unwrap().is_empty(),
        "no cancel before the client asks for one"
    );

    // Client cancels the in-flight request (MCP notifications/cancelled).
    handle
        .cancel(Some("user canceled".to_owned()))
        .await
        .expect("cancel notification is sent");

    // The tool forwards it to the job cancel route, freeing the worker/GPU.
    wait_until(|| {
        harness
            .cancels
            .lock()
            .unwrap()
            .iter()
            .any(|id| id == "job-1")
    })
    .await;

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn transient_poll_failures_are_tolerated_until_the_job_completes() {
    // The first two `GET /jobs/:id` polls fail (500) — a transient blip — before
    // the job reports completed. The tool must ride through, not abort the render
    // (sc-10279).
    #[derive(Clone)]
    struct FlakyState {
        polls: Arc<Mutex<usize>>,
    }
    let state = FlakyState {
        polls: Arc::new(Mutex::new(0)),
    };
    let polls = state.polls.clone();
    let router = Router::new()
        .route(
            "/api/v1/image/jobs",
            post(|| async {
                (
                    StatusCode::CREATED,
                    Json(json!({
                        "id": "job-1", "status": "queued", "projectId": "p1",
                        "progress": 0.0, "stage": "queued"
                    })),
                )
            }),
        )
        .route(
            "/api/v1/jobs/:job_id",
            get(
                |State(state): State<FlakyState>, Path(_job_id): Path<String>| async move {
                    let n = {
                        let mut polls = state.polls.lock().unwrap();
                        let n = *polls;
                        *polls += 1;
                        n
                    };
                    if n < 2 {
                        // Two consecutive transient failures (< the tolerance).
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "detail": "temporary glitch" })),
                        )
                            .into_response();
                    }
                    Json(json!({
                        "id": "job-1", "status": "completed", "projectId": "p1",
                        "progress": 1.0, "stage": "completed",
                        "result": { "assets": [image_asset("asset_1", PNG_PATH, "image/png")] }
                    }))
                    .into_response()
                },
            ),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(
                |Path((_project_id, relative_path)): Path<(String, String)>| async move {
                    if relative_path == PNG_PATH {
                        let mut headers = HeaderMap::new();
                        headers.insert(header::CONTENT_TYPE, "image/png".parse().unwrap());
                        (headers, PNG_BYTES.to_vec()).into_response()
                    } else {
                        StatusCode::NOT_FOUND.into_response()
                    }
                },
            ),
        )
        .with_state(state);
    let api_base = spawn(router).await;
    let (_, client) =
        connect_mcp(api_base, None, fast_job_wait(), RecordingClient::default()).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("generate_image succeeds despite the transient poll failures");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");
    let images: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_image())
        .collect();
    assert_eq!(images.len(), 1, "the render survived the blips: {result:?}");
    // It really did fail twice and recover, not skip the failures.
    assert!(
        *polls.lock().unwrap() >= 3,
        "expected 2 failed polls + a successful one"
    );

    let _ = client.cancel().await;
}

#[tokio::test]
async fn stuck_job_times_out_with_a_clear_error_instead_of_hanging() {
    // The script never leaves `running`; the (test-shortened) overall deadline
    // must turn that into a clear tool error, not an endless poll.
    let state = StubState {
        submitted: Arc::new(Mutex::new(Vec::new())),
        polls: Arc::new(Mutex::new(0)),
        snapshots: Arc::new(vec![snapshot("running", 0.5, "generating", json!({}))]),
        cancels: Arc::new(Mutex::new(Vec::new())),
        workflow_png: Arc::new(workflow_png_bytes()),
    };
    let cancels = state.cancels.clone();
    let api_base = spawn(stub_api_router(state)).await;
    let (_, client) = connect_mcp(
        api_base,
        None,
        JobWaitConfig {
            poll_interval: Duration::from_millis(10),
            timeout: Duration::from_millis(100),
        },
        RecordingClient::default(),
    )
    .await;

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
    // A timeout is NOT a cancellation — the job is left running (sc-10276).
    assert!(
        cancels.lock().unwrap().is_empty(),
        "the timeout path must not cancel the job"
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

/// F-041 (sc-11236): an over-cap result (one 5 MiB image, above the 4 MiB
/// per-image inline cap) must NOT be base64-inlined — the tool falls back to the
/// `get_job_result` ticketed-link shape (resource links + a JSON summary carrying
/// ticketed URLs, zero inline image bytes).
#[tokio::test]
async fn oversize_result_falls_back_to_ticketed_links() {
    let harness = harness(vec![
        snapshot("running", 0.5, "generating", json!({})),
        snapshot(
            "completed",
            1.0,
            "completed",
            json!({ "result": { "assets": [image_asset("asset_big", LARGE_PATH, "image/png")] } }),
        ),
    ])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        )
        .await
        .expect("generate_image succeeds");
    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");

    // Zero inline image blocks: the oversize payload spilled to links.
    assert!(
        !result
            .content
            .iter()
            .any(|block| block.as_image().is_some()),
        "an over-cap result must not inline base64 image bytes: {result:?}"
    );

    // Exactly one ticketed resource link for the asset.
    let links: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_resource_link())
        .collect();
    assert_eq!(links.len(), 1, "one ticketed link: {result:?}");
    assert!(
        links[0].uri.contains("/api/v1/projects/p1/files/")
            && links[0]
                .uri
                .contains(&format!("?stripWorkflow=true&ticket={TICKET}")),
        "link is the ticketed media URL: {}",
        links[0].uri
    );

    // The JSON summary is the get_job_result shape (completed status + ticketed url).
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
    assert_eq!(summary["status"], "completed");
    assert_eq!(summary["assets"][0]["id"], "asset_big");
    assert_eq!(summary["assets"][0]["workflowPolicy"], "strip-requested");
    assert!(summary["assets"][0]["url"]
        .as_str()
        .is_some_and(|url| url.contains(&format!("?stripWorkflow=true&ticket={TICKET}"))));

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn chunked_oversize_result_aborts_without_waiting_for_end_of_stream() {
    let harness = harness(vec![snapshot(
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [
            image_asset("asset_chunked", CHUNKED_LARGE_PATH, "image/png")
        ] } }),
    )])
    .await;

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        harness.client.call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({}))),
        ),
    )
    .await
    .expect("bounded reader aborts the never-ending body once it crosses the cap")
    .expect("generate_image succeeds");

    assert_ne!(result.is_error, Some(true), "unexpected error: {result:?}");
    assert!(
        !result
            .content
            .iter()
            .any(|block| block.as_image().is_some()),
        "an over-cap chunked result must not inline bytes: {result:?}"
    );
    let links: Vec<_> = result
        .content
        .iter()
        .filter_map(|block| block.as_resource_link())
        .collect();
    assert_eq!(links.len(), 1, "one ticketed link: {result:?}");
    assert!(links[0]
        .uri
        .contains(&format!("?stripWorkflow=true&ticket={TICKET}")));

    let _ = harness.client.cancel().await;
}

#[tokio::test]
async fn oversize_result_can_include_workflow_in_links_by_explicit_request() {
    let harness = harness(vec![snapshot(
        "completed",
        1.0,
        "completed",
        json!({ "result": { "assets": [image_asset(
            "asset_big",
            LARGE_PATH,
            "image/png"
        )] } }),
    )])
    .await;

    let result = harness
        .client
        .call_tool(
            CallToolRequestParams::new("generate_image").with_arguments(generate_args(json!({
                "includeWorkflow": true
            }))),
        )
        .await
        .expect("generate_image succeeds");
    let link = result
        .content
        .iter()
        .find_map(|block| block.as_resource_link())
        .expect("resource link fallback");
    assert!(link.uri.contains(&format!("?ticket={TICKET}")));
    assert!(!link.uri.contains("stripWorkflow"));

    let summary: Value = serde_json::from_str(
        &result
            .content
            .iter()
            .rev()
            .find_map(|block| block.as_text())
            .expect("summary block")
            .text,
    )
    .expect("summary JSON");
    assert_eq!(
        summary["assets"][0]["workflowPolicy"],
        "preserve-if-present"
    );

    let _ = harness.client.cancel().await;
}
