
#[tokio::test]
async fn huggingface_snapshot_resolve_accepts_tree_and_sibling_shapes_with_auth() {
    let array_url = spawn_hf_stub(
        json!([
            { "type": "file", "path": "nested/model.safetensors", "size": 7 },
            { "type": "file", "path": "nested/model.ckpt", "size": 9 },
            { "type": "directory", "path": "nested" }
        ]),
        Some("hf_test"),
    )
    .await;
    let client = reqwest::Client::new();
    let array_settings = test_settings(array_url, Some("hf_test"));

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &array_settings,
        "owner/model",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("tree snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "nested/model.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(7));

    let siblings_url = spawn_hf_stub(
        json!({
            "siblings": [
                { "rfilename": "adapter.safetensors", "size": "5" }
            ]
        }),
        None,
    )
    .await;
    let siblings_settings = test_settings(siblings_url, None);

    let snapshot = HuggingFaceSnapshot::resolve(
        &client,
        &siblings_settings,
        "owner/lora",
        "main",
        &["*.safetensors".to_owned()],
    )
    .await
    .expect("siblings snapshot resolves");

    assert_eq!(snapshot.files.len(), 1);
    assert_eq!(snapshot.files[0].path, "adapter.safetensors");
    assert_eq!(snapshot.total_bytes(), Some(5));
}

#[derive(Clone)]
struct HfStubState {
    payload: serde_json::Value,
    token: Option<String>,
}

async fn spawn_hf_stub(payload: serde_json::Value, token: Option<&str>) -> String {
    let state = HfStubState {
        payload,
        token: token.map(str::to_owned),
    };
    let app = Router::new()
        .route("/api/models/:owner/:repo/tree/:revision", get(hf_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn hf_stub(State(state): State<HfStubState>, headers: HeaderMap) -> Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let authorized = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str());
        if !authorized {
            return (
                AxumStatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing token" })),
            )
                .into_response();
        }
    }
    Json(state.payload).into_response()
}

#[derive(Clone)]
struct BinaryStubState {
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
}

async fn spawn_binary_stub(bytes: Vec<u8>) -> String {
    spawn_binary_stub_with_options(bytes, AxumStatusCode::OK, false).await
}

async fn spawn_binary_stub_with_options(
    bytes: Vec<u8>,
    status: AxumStatusCode,
    cancel_requested: bool,
) -> String {
    let state = BinaryStubState {
        bytes,
        status,
        cancel_requested,
    };
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_stub))
        .route("/api/v1/jobs/:job_id/progress", post(progress_stub))
        .route("/*path", get(binary_stub).head(binary_head_stub))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

async fn binary_stub(State(state): State<BinaryStubState>, headers: HeaderMap) -> Response {
    let length = state.bytes.len();
    if headers
        .get(axum::http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("bytes={length}-"))
    {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = AxumStatusCode::RANGE_NOT_SATISFIABLE;
        response.headers_mut().insert(
            axum::http::header::CONTENT_RANGE,
            axum::http::HeaderValue::from_str(&format!("bytes */{length}"))
                .expect("content range header"),
        );
        return response;
    }
    if let Some(start) = headers
        .get(axum::http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.strip_suffix('-'))
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|start| *start < length)
    {
        let body = state.bytes[start..].to_vec();
        let mut response = body.into_response();
        *response.status_mut() = AxumStatusCode::PARTIAL_CONTENT;
        response.headers_mut().insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&(length - start).to_string())
                .expect("content length header"),
        );
        response.headers_mut().insert(
            axum::http::header::CONTENT_RANGE,
            axum::http::HeaderValue::from_str(&format!("bytes {start}-{}/{length}", length - 1))
                .expect("content range header"),
        );
        return response;
    }
    let mut response = state.bytes.into_response();
    *response.status_mut() = state.status;
    response.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
    );
    response
}

async fn binary_head_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = state.status;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&state.bytes.len().to_string())
            .expect("content length header"),
    );
    // Mirror Hugging Face's resolve metadata so download_snapshot can name blobs by
    // etag and record the commit (sc-1904).
    let last_segment = path.rsplit('/').next().unwrap_or("blob");
    headers.insert(
        axum::http::header::ETAG,
        axum::http::HeaderValue::from_str(&format!("\"etag-{last_segment}\""))
            .expect("etag header"),
    );
    headers.insert(
        "x-repo-commit",
        axum::http::HeaderValue::from_static("stubcommit"),
    );
    response
}

async fn job_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

async fn progress_stub(
    State(state): State<BinaryStubState>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
}

fn job_snapshot_json(job_id: &str, cancel_requested: bool) -> Value {
    json!({
        "id": job_id,
        "type": "lora_import",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.1,
        "stage": "importing",
        "message": "running",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": cancel_requested,
        "createdAt": "2026-05-18T00:00:00Z",
        "updatedAt": "2026-05-18T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    })
}

fn worker_snapshot_json(worker_id: &str) -> Value {
    json!({
        "id": worker_id,
        "gpuId": "cpu",
        "gpuName": null,
        "status": "busy",
        "currentJobId": "job-1",
        "capabilities": [],
        "loadedModels": [],
        "registeredAt": "2026-07-01T00:00:00Z",
        "lastSeenAt": "2026-07-01T00:00:00Z"
    })
}

/// sc-8806 — stub for the tick-driven download-cancel path. Counts GETs of the
/// job snapshot (the chunk loop must never poll it), serves the progress POST
/// with a configurable `cancelRequested` (the snapshot the tick reuses for its
/// cancel decision), answers worker heartbeats, and serves the binary either as
/// a short multi-chunk body or as a stream that stalls after the first chunk —
/// so only the interval tick can observe a cancel.
#[derive(Clone)]
struct CancelTickStubState {
    job_gets: Arc<AtomicUsize>,
    progress_cancel_requested: bool,
    stall_after_first_chunk: bool,
}

async fn spawn_cancel_tick_stub(state: CancelTickStubState) -> String {
    use futures_util::StreamExt;

    async fn job_route(
        State(state): State<CancelTickStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        state.job_gets.fetch_add(1, Ordering::SeqCst);
        // Deliberately NOT canceled: only the progress POST snapshot says
        // canceled, so a cancel can only come from reusing that snapshot.
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(
        State(state): State<CancelTickStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        Json(job_snapshot_json(&job_id, state.progress_cancel_requested)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    async fn binary_route(State(state): State<CancelTickStubState>) -> Response {
        let chunks = futures_util::stream::iter(vec![
            Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(b"url-")),
            Ok(axum::body::Bytes::from_static(b"lora")),
        ]);
        if state.stall_after_first_chunk {
            Body::from_stream(chunks.chain(futures_util::stream::pending())).into_response()
        } else {
            Body::from_stream(chunks).into_response()
        }
    }

    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .route("/*path", get(binary_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

fn test_settings(huggingface_base_url: String, huggingface_token: Option<&str>) -> Settings {
    Settings {
        api_url: "http://127.0.0.1:8000".to_owned(),
        access_token: None,
        data_dir: PathBuf::from("data"),
        config_dir: PathBuf::from("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "cpu".to_owned(),
        is_child_worker: true,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url,
        huggingface_token: huggingface_token.map(str::to_owned),
        credentials: Vec::new(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        allow_private_lora_urls: true,
        utility_workers: 1,
        backend_mlx_enabled: true,
        backend_candle_enabled: false,
        gpu_memory_limit_bytes: 0,
        external_model_roots: Vec::new(),
    }
}

#[test]
fn idle_heartbeat_is_due_immediately_then_waits_for_interval() {
    let mut heartbeat = IdleHeartbeat::new(Duration::from_secs(60));

    assert!(heartbeat.should_send());
    heartbeat.mark_sent();
    assert!(!heartbeat.should_send());
}

#[test]
fn idle_heartbeat_allows_immediate_resend_when_interval_is_zero() {
    let mut heartbeat = IdleHeartbeat::new(Duration::ZERO);

    assert!(heartbeat.should_send());
    heartbeat.mark_sent();
    assert!(heartbeat.should_send());
}

fn spawn_exit_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "exit /B 0"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "exit 0"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}

fn spawn_sleep_child() -> tokio::process::Child {
    let mut command = if cfg!(windows) {
        let mut command = tokio::process::Command::new("cmd");
        command.args(["/C", "ping -n 30 127.0.0.1 >NUL"]);
        command
    } else {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "sleep 30"]);
        command
    };
    command
        .stdout(StdStdio::null())
        .stderr(StdStdio::null())
        .spawn()
        .expect("test child starts")
}

/// sc-4174 — the in-band cancel poll for long generations must only cancel on
/// a confirmed user cancel. A transient API failure (on the GET, or on the
/// Canceled-status POST inside check_cancel) is tolerated and retried on the
/// next poll instead of aborting a multi-minute run.
#[derive(Clone)]
struct CancelPollStubState {
    get_status: AxumStatusCode,
    cancel_requested: bool,
    post_status: AxumStatusCode,
}

async fn spawn_cancel_poll_stub(state: CancelPollStubState) -> String {
    async fn job_route(
        State(state): State<CancelPollStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        if state.get_status != AxumStatusCode::OK {
            return (state.get_status, "stub GET failure").into_response();
        }
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }
    async fn progress_route(
        State(state): State<CancelPollStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        if state.post_status != AxumStatusCode::OK {
            return (state.post_status, "stub POST failure").into_response();
        }
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

/// sc-5515 — the in-loop image cancel poller uses a CHECK-ONLY peek that reads
/// `cancel_requested` without posting any terminal status. The terminal Canceled
/// is posted by `consume_gen_events` only after the blocking generation actually
/// stops, so the worker row isn't freed (and the next queued job isn't misled)
/// while the in-flight image is still rendering. `post_status` is wired to fail
/// here to prove the peek never touches the progress route.
async fn cancel_peek_with(get_status: AxumStatusCode, cancel_requested: bool) -> bool {
    let base_url = spawn_cancel_poll_stub(CancelPollStubState {
        get_status,
        cancel_requested,
        post_status: AxumStatusCode::INTERNAL_SERVER_ERROR,
    })
    .await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    cancel_requested_peek(&api, "job-1").await
}

#[tokio::test]
async fn cancel_peek_reports_confirmed_cancel_without_posting() {
    assert!(
        cancel_peek_with(AxumStatusCode::OK, true).await,
        "a confirmed cancel request must be reported by the check-only peek"
    );
}

#[tokio::test]
async fn cancel_peek_false_when_not_requested() {
    assert!(!cancel_peek_with(AxumStatusCode::OK, false).await);
}

#[tokio::test]
async fn cancel_peek_tolerates_transient_get_errors() {
    assert!(
        !cancel_peek_with(AxumStatusCode::INTERNAL_SERVER_ERROR, true).await,
        "a transient GET failure must not read as a user cancel"
    );
}

// sc-5516 — the in-loop video/training/detail cancel pollers DEFER the terminal `Canceled`
// to actual-stop: at acknowledgement they only trip the engine flag and post a NON-terminal
// "Cancelling…" update (so the worker row isn't freed while the in-flight step is still
// running). This stub captures every progress POST body so a test can assert the
// acknowledgement status is `running`, not the terminal `canceled`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn spawn_progress_capture_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

/// sc-8845 (F-043) — capture stub whose job GET reports NO user cancel, so the only cancel that can
/// fire in `run_placeholder_job` is the process-shutdown flag. Records every progress POST body and
/// answers heartbeats.
async fn spawn_no_user_cancel_capture_stub(
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        // No user cancel — a shutdown-driven cancel must be the ONLY thing that can trip.
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

fn placeholder_job_snapshot() -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job-1",
        "type": "placeholder",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.0,
        "stage": "queued",
        "message": "queued",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-07-03T00:00:00Z",
        "updatedAt": "2026-07-03T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("placeholder job snapshot deserializes")
}
