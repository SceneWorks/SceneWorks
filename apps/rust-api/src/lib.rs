use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, DuplicateJobRequest, JobCreateRequest, JobSnapshot,
    ProgressRequest, QueueSummary, WorkerHeartbeatRequest, WorkerRegisterRequest, WorkerSnapshot,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    WorkerHeartbeat, JOB_STATUSES,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::{AllowOrigin, CorsLayer};

const PUBLIC_PATHS: &[&str] = &[
    "/api/v1/health",
    "/api/v1/access",
    "/api/v1/auth/verify",
    "/api/v1/jobs/events",
];
const DEFAULT_CORS_ORIGINS: &str = concat!(
    "http://localhost:5173,http://127.0.0.1:5173,",
    "http://localhost:5174,http://127.0.0.1:5174,",
    "http://localhost:5175,http://127.0.0.1:5175,",
    "http://localhost:5176,http://127.0.0.1:5176"
);
static EVENT_TICKET_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct Settings {
    pub app_version: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub access_token: String,
    pub cors_origins: Vec<String>,
    pub worker_timeout_seconds: u64,
    pub jobs_db_path: PathBuf,
}

impl Settings {
    pub fn from_env() -> Self {
        let data_dir = env_path("SCENEWORKS_DATA_DIR", "data");
        let jobs_db_path = std::env::var("SCENEWORKS_JOBS_DB_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("cache").join("jobs.db"));
        Self {
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.1.0"),
            host: env_string("SCENEWORKS_API_HOST", "0.0.0.0"),
            port: std::env::var("SCENEWORKS_API_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8000),
            data_dir,
            config_dir: env_path("SCENEWORKS_CONFIG_DIR", "config"),
            access_token: std::env::var("SCENEWORKS_ACCESS_TOKEN")
                .unwrap_or_default()
                .trim()
                .to_owned(),
            cors_origins: env_string("SCENEWORKS_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
            worker_timeout_seconds: std::env::var("SCENEWORKS_WORKER_TIMEOUT_SECONDS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(90),
            jobs_db_path,
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }
}

#[derive(Clone)]
pub struct AppState {
    settings: Settings,
    jobs_store: Arc<JobsStore>,
    events: broadcast::Sender<EventMessage>,
    event_tickets: Arc<EventTicketStore>,
    interrupted_jobs_on_startup: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EventMessage {
    event: String,
    data: Value,
}

#[derive(Debug)]
struct EventTicketStore {
    ttl: Duration,
    tickets: Mutex<HashMap<String, Instant>>,
}

impl EventTicketStore {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            tickets: Mutex::new(HashMap::new()),
        }
    }

    fn issue(&self) -> Result<EventTicket, ApiError> {
        let now = Instant::now();
        let mut tickets = self
            .tickets
            .lock()
            .map_err(|_| ApiError::internal("Event ticket store lock poisoned"))?;
        prune_tickets(&mut tickets, now);
        let ticket = format!(
            "rust-event-ticket-{}-{}",
            unix_millis(),
            EVENT_TICKET_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        tickets.insert(ticket.clone(), now + self.ttl);
        Ok(EventTicket {
            ticket,
            expires_in_seconds: self.ttl.as_secs(),
        })
    }

    fn consume(&self, ticket: &str) -> Result<(), ApiError> {
        let now = Instant::now();
        let mut tickets = self
            .tickets
            .lock()
            .map_err(|_| ApiError::internal("Event ticket store lock poisoned"))?;
        prune_tickets(&mut tickets, now);
        match tickets.remove(ticket) {
            Some(expires_at) if expires_at >= now => Ok(()),
            _ => Err(ApiError::unauthorized(
                "Invalid or expired event stream ticket",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EventTicket {
    ticket: String,
    expires_in_seconds: u64,
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::from_env();
    let address: SocketAddr = format!("{}:{}", settings.host, settings.port).parse()?;
    let app = create_app(settings)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("SceneWorks Rust API listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn create_app(settings: Settings) -> Result<Router, JobsStoreError> {
    let jobs_store = Arc::new(JobsStore::new(&settings.jobs_db_path));
    jobs_store.initialize()?;
    let interrupted_jobs_on_startup = jobs_store.mark_interrupted_on_startup()?.len();
    let state = AppState {
        settings,
        jobs_store,
        events: broadcast::channel(100).0,
        event_tickets: Arc::new(EventTicketStore::new(30)),
        interrupted_jobs_on_startup,
    };
    let cors = cors_layer(&state.settings);

    Ok(Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route("/api/v1/jobs", get(list_jobs).post(create_job))
        .route("/api/v1/jobs/claim", post(claim_job))
        .route("/api/v1/jobs/events", get(job_events))
        .route("/api/v1/jobs/events/ticket", post(create_event_ticket))
        .route("/api/v1/jobs/:job_id", get(get_job))
        .route("/api/v1/jobs/:job_id/cancel", post(cancel_job))
        .route("/api/v1/jobs/:job_id/retry", post(retry_job))
        .route("/api/v1/jobs/:job_id/duplicate", post(duplicate_job))
        .route("/api/v1/jobs/:job_id/progress", post(update_job_progress))
        .route("/api/v1/queue", get(queue_summary))
        .route("/api/v1/workers", get(list_workers))
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, access_control))
        .layer(cors))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobsQuery {
    project_id: Option<String>,
    status: Option<String>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    ticket: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: String,
    auth_required: bool,
    directories: DirectoriesResponse,
    interrupted_jobs_on_startup: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DirectoriesResponse {
    data: String,
    config: String,
    projects: String,
    jobs_db: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessResponse {
    auth_required: bool,
    token_header: &'static str,
}

#[derive(Debug, Serialize)]
struct VerifyResponse {
    ok: bool,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
        version: state.settings.app_version.clone(),
        auth_required: !state.settings.access_token.is_empty(),
        directories: DirectoriesResponse {
            data: state.settings.data_dir.display().to_string(),
            config: state.settings.config_dir.display().to_string(),
            projects: state.settings.projects_dir().display().to_string(),
            jobs_db: state.settings.jobs_db_path.display().to_string(),
        },
        interrupted_jobs_on_startup: state.interrupted_jobs_on_startup,
    })
}

async fn access(State(state): State<AppState>) -> Json<AccessResponse> {
    Json(AccessResponse {
        auth_required: !state.settings.access_token.is_empty(),
        token_header: "X-SceneWorks-Token",
    })
}

async fn verify_access(State(state): State<AppState>, headers: HeaderMap) -> Json<VerifyResponse> {
    Json(VerifyResponse {
        ok: is_authorized(&headers, &state.settings),
    })
}

async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> Result<Json<Vec<JobSnapshot>>, ApiError> {
    if let Some(status) = &query.status {
        if !JOB_STATUSES.contains(&status.as_str()) {
            return Err(ApiError::bad_request("Unsupported job status"));
        }
    }
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_jobs(
                query.project_id.as_deref(),
                query.status.as_deref(),
                query.limit.unwrap_or(100),
            )
        })
        .await?,
    ))
}

async fn create_job(
    State(state): State<AppState>,
    Json(payload): Json<JobCreateRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: payload.job_type,
            project_id: payload.project_id,
            project_name: payload.project_name,
            payload: payload.payload,
            requested_gpu: payload.requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn claim_job(
    State(state): State<AppState>,
    Json(payload): Json<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let response = store_call(state.clone(), move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.claim_next_job(&payload.worker_id)
    })
    .await?;
    if let Some(job) = &response {
        publish(&state, "job.updated", job);
        publish_queue(&state).await?;
    }
    Ok(Json(ClaimResponse {
        job: response,
        extra: Default::default(),
    }))
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| store.get_job(&job_id)).await?,
    ))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.cancel_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.retry_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn duplicate_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(payload): Json<DuplicateJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.duplicate_job(
            &job_id,
            DuplicateJob {
                payload_changes: payload.payload_changes,
                requested_gpu: payload.requested_gpu,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn update_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(payload): Json<ProgressRequest>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let progress = number_to_f64(&payload.progress, "progress")?;
    let eta_seconds = optional_number_to_f64(payload.eta_seconds.as_ref(), "etaSeconds")?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.update_job_progress(
            &job_id,
            ProgressUpdate {
                status: payload.status,
                stage: payload.stage,
                progress,
                message: payload.message,
                error: payload.error,
                result: payload.result,
                eta_seconds,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

async fn queue_summary(State(state): State<AppState>) -> Result<Json<QueueSummary>, ApiError> {
    Ok(Json(queue_summary_snapshot(state).await?))
}

async fn list_workers(
    State(state): State<AppState>,
) -> Result<Json<Vec<WorkerSnapshot>>, ApiError> {
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_workers()
        })
        .await?,
    ))
}

async fn register_worker(
    State(state): State<AppState>,
    Json(payload): Json<WorkerRegisterRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.register_worker(RegisterWorker {
            worker_id: payload.worker_id,
            gpu_id: payload.gpu_id,
            gpu_name: payload.gpu_name,
            capabilities: payload.capabilities,
            loaded_models: payload.loaded_models,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    publish_queue(&state).await?;
    Ok(Json(worker))
}

async fn heartbeat_worker(
    State(state): State<AppState>,
    Path(worker_id): Path<String>,
    Json(payload): Json<WorkerHeartbeatRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.heartbeat_worker(WorkerHeartbeat {
            worker_id,
            status: payload.status,
            current_job_id: payload.current_job_id,
            loaded_models: payload.loaded_models,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    Ok(Json(worker))
}

async fn create_event_ticket(State(state): State<AppState>) -> Result<Json<EventTicket>, ApiError> {
    Ok(Json(state.event_tickets.issue()?))
}

async fn job_events(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if !state.settings.access_token.is_empty() {
        state
            .event_tickets
            .consume(query.ticket.as_deref().unwrap_or_default())?;
    }
    let ready = tokio_stream::iter([Ok(Event::default()
        .event("ready")
        .data(json!({ "status": "connected" }).to_string()))]);
    let messages =
        BroadcastStream::new(state.events.subscribe()).filter_map(|message| match message {
            Ok(message) => Some(Ok(Event::default()
                .event(message.event)
                .data(message.data.to_string()))),
            Err(_) => None,
        });
    Ok(Sse::new(ready.chain(messages)).keep_alive(KeepAlive::default()))
}

async fn access_control(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if request.method().as_str() == "OPTIONS"
        || PUBLIC_PATHS.contains(&request.uri().path())
        || is_authorized(request.headers(), &state.settings)
    {
        return next.run(request).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "detail": "SceneWorks access token required",
            "authRequired": true
        })),
    )
        .into_response()
}

fn cors_layer(settings: &Settings) -> CorsLayer {
    let origins = settings
        .cors_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            HeaderName::from_static("x-sceneworks-token"),
        ])
}

fn is_authorized(headers: &HeaderMap, settings: &Settings) -> bool {
    if settings.access_token.is_empty() {
        return true;
    }
    constant_time_eq(
        token_from_headers(headers).as_bytes(),
        settings.access_token.as_bytes(),
    )
}

fn token_from_headers(headers: &HeaderMap) -> String {
    if let Some(token) = headers
        .get("x-sceneworks-token")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return token.to_owned();
    }
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0, |difference, (left, right)| difference | (left ^ right))
        == 0
}

async fn store_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<JobsStore>, u64) -> Result<T, JobsStoreError> + Send + 'static,
{
    let timeout = state.settings.worker_timeout_seconds;
    let store = state.jobs_store.clone();
    tokio::task::spawn_blocking(move || operation(store, timeout))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(Into::into)
}

async fn queue_summary_snapshot(state: AppState) -> Result<QueueSummary, ApiError> {
    store_call(state, |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.queue_summary()
    })
    .await
}

async fn publish_queue(state: &AppState) -> Result<(), ApiError> {
    let queue = queue_summary_snapshot(state.clone()).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

fn publish<T: Serialize>(state: &AppState, event: &str, data: &T) {
    if let Ok(data) = serde_json::to_value(data) {
        let _ = state.events.send(EventMessage {
            event: event.to_owned(),
            data,
        });
    }
}

fn number_to_f64(number: &serde_json::Number, field: &'static str) -> Result<f64, ApiError> {
    number
        .as_f64()
        .ok_or_else(|| ApiError::bad_request(format!("Invalid numeric value for {field}")))
}

fn optional_number_to_f64(
    number: Option<&serde_json::Number>,
    field: &'static str,
) -> Result<Option<f64>, ApiError> {
    number.map(|value| number_to_f64(value, field)).transpose()
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(env_string(name, default))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    detail: String,
}

impl ApiError {
    fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
        }
    }

    fn unauthorized(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            detail: detail.into(),
        }
    }

    fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
        }
    }
}

impl From<JobsStoreError> for ApiError {
    fn from(error: JobsStoreError) -> Self {
        match error {
            JobsStoreError::NotFound(_) => Self {
                status: StatusCode::NOT_FOUND,
                detail: "Record not found".to_owned(),
            },
            JobsStoreError::InvalidStatus(status) => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Unsupported job status: {status}"),
            },
            JobsStoreError::InvalidNumber(field) => {
                Self::bad_request(format!("Invalid numeric value for {field}"))
            }
            JobsStoreError::RetryLimit { max_attempts } => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Job retry limit reached after {max_attempts} attempts."),
            },
            other => Self::internal(other.to_string()),
        }
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn prune_tickets(tickets: &mut HashMap<String, Instant>, now: Instant) {
    tickets.retain(|_, expires_at| *expires_at >= now);
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::{create_app, Settings};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn test_settings(temp_dir: &tempfile::TempDir) -> Settings {
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
        }
    }

    async fn request(
        app: axum::Router,
        method: &str,
        uri: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        request_with_headers(app, method, uri, body, &[]).await
    }

    async fn request_with_headers(
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

    #[tokio::test]
    async fn worker_can_register_claim_and_complete_job_through_http() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/workers/register",
            json!({
                "workerId": "worker-1",
                "gpuId": "gpu-0",
                "gpuName": "GPU 0",
                "capabilities": ["image_generate"],
                "loadedModels": []
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_generate",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": "mist over hills" },
                "requestedGpu": "auto"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let (status, claimed) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs/claim",
            json!({ "workerId": "worker-1" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(claimed["job"]["id"], created["id"]);
        assert_eq!(claimed["job"]["status"], "preparing");

        let job_id = created["id"].as_str().expect("job id is string");
        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Done",
                "result": { "assetIds": ["asset-1"] }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["result"], json!({ "assetIds": ["asset-1"] }));

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["completed"], 1);
        assert_eq!(queue["workers"][0]["status"], "idle");
    }

    #[tokio::test]
    async fn worker_heartbeat_interrupts_previous_active_job_through_http() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        request(
            app.clone(),
            "POST",
            "/api/v1/workers/register",
            json!({
                "workerId": "worker-1",
                "gpuId": "gpu-0",
                "gpuName": null,
                "capabilities": ["image_generate"],
                "loadedModels": []
            }),
        )
        .await;
        let (_, created) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({ "type": "image_generate", "payload": {}, "requestedGpu": "auto" }),
        )
        .await;
        request(
            app.clone(),
            "POST",
            "/api/v1/jobs/claim",
            json!({ "workerId": "worker-1" }),
        )
        .await;

        let (status, worker) = request(
            app.clone(),
            "POST",
            "/api/v1/workers/worker-1/heartbeat",
            json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(worker["currentJobId"], Value::Null);

        let job_id = created["id"].as_str().expect("job id is string");
        let (status, job) =
            request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(job["status"], "interrupted");
        assert_eq!(job["workerId"], Value::Null);
    }

    #[tokio::test]
    async fn access_token_is_enforced_on_protected_routes() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, access) = request(app.clone(), "GET", "/api/v1/access", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(access["authRequired"], true);

        let (status, error) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "SceneWorks access token required");

        let (status, jobs) = request_with_headers(
            app,
            "GET",
            "/api/v1/jobs",
            Value::Null,
            &[("x-sceneworks-token", "secret-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(jobs, json!([]));
    }

    #[tokio::test]
    async fn bearer_token_is_accepted_for_access_verification() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, verified) = request_with_headers(
            app,
            "POST",
            "/api/v1/auth/verify",
            Value::Null,
            &[("authorization", "Bearer secret-token")],
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(verified["ok"], true);
    }

    #[tokio::test]
    async fn event_tickets_are_protected_and_match_python_shape() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs/events/ticket",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "SceneWorks access token required");

        let (status, ticket) = request_with_headers(
            app.clone(),
            "POST",
            "/api/v1/jobs/events/ticket",
            Value::Null,
            &[("x-sceneworks-token", "secret-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(ticket["ticket"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert_eq!(ticket["expiresInSeconds"], 30);

        let (status, error) = request(
            app,
            "GET",
            "/api/v1/jobs/events?ticket=missing",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "Invalid or expired event stream ticket");
    }

    #[tokio::test]
    async fn cors_preflight_allows_frontend_origin_and_token_header() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let request = Request::builder()
            .method("OPTIONS")
            .uri("/api/v1/jobs")
            .header("origin", "http://localhost:5173")
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "X-SceneWorks-Token")
            .body(Body::empty())
            .expect("request builds");

        let response = app.oneshot(request).await.expect("response returns");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );
        assert!(response
            .headers()
            .get("access-control-allow-headers")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("x-sceneworks-token")));
    }
}
