use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
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
use serde_json::json;

#[derive(Debug, Clone)]
pub struct Settings {
    pub app_version: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub access_token: String,
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
    interrupted_jobs_on_startup: usize,
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
        interrupted_jobs_on_startup,
    };

    Ok(Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/jobs", get(list_jobs).post(create_job))
        .route("/api/v1/jobs/claim", post(claim_job))
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
        .with_state(state))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobsQuery {
    project_id: Option<String>,
    status: Option<String>,
    limit: Option<u32>,
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

async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> Result<Json<Vec<JobSnapshot>>, ApiError> {
    sweep_stale_workers(&state)?;
    if let Some(status) = &query.status {
        if !JOB_STATUSES.contains(&status.as_str()) {
            return Err(ApiError::bad_request("Unsupported job status"));
        }
    }
    Ok(Json(state.jobs_store.list_jobs(
        query.project_id.as_deref(),
        query.status.as_deref(),
        query.limit.unwrap_or(100),
    )?))
}

async fn create_job(
    State(state): State<AppState>,
    Json(payload): Json<JobCreateRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = state.jobs_store.create_job(CreateJob {
        job_type: payload.job_type,
        project_id: payload.project_id,
        project_name: payload.project_name,
        payload: payload.payload,
        requested_gpu: payload.requested_gpu,
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    })?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn claim_job(
    State(state): State<AppState>,
    Json(payload): Json<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    sweep_stale_workers(&state)?;
    Ok(Json(ClaimResponse {
        job: state.jobs_store.claim_next_job(&payload.worker_id)?,
        extra: Default::default(),
    }))
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(state.jobs_store.get_job(&job_id)?))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(state.jobs_store.cancel_job(&job_id)?))
}

async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    Ok((
        StatusCode::CREATED,
        Json(state.jobs_store.retry_job(&job_id)?),
    ))
}

async fn duplicate_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(payload): Json<DuplicateJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = state.jobs_store.duplicate_job(
        &job_id,
        DuplicateJob {
            payload_changes: payload.payload_changes,
            requested_gpu: payload.requested_gpu,
        },
    )?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn update_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(payload): Json<ProgressRequest>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = state.jobs_store.update_job_progress(
        &job_id,
        ProgressUpdate {
            status: payload.status,
            stage: payload.stage,
            progress: number_to_f64(&payload.progress, "progress")?,
            message: payload.message,
            error: payload.error,
            result: payload.result,
            eta_seconds: optional_number_to_f64(payload.eta_seconds.as_ref(), "etaSeconds")?,
        },
    )?;
    Ok(Json(job))
}

async fn queue_summary(State(state): State<AppState>) -> Result<Json<QueueSummary>, ApiError> {
    sweep_stale_workers(&state)?;
    Ok(Json(state.jobs_store.queue_summary()?))
}

async fn list_workers(
    State(state): State<AppState>,
) -> Result<Json<Vec<WorkerSnapshot>>, ApiError> {
    sweep_stale_workers(&state)?;
    Ok(Json(state.jobs_store.list_workers()?))
}

async fn register_worker(
    State(state): State<AppState>,
    Json(payload): Json<WorkerRegisterRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    Ok(Json(state.jobs_store.register_worker(RegisterWorker {
        worker_id: payload.worker_id,
        gpu_id: payload.gpu_id,
        gpu_name: payload.gpu_name,
        capabilities: payload.capabilities,
        loaded_models: payload.loaded_models,
    })?))
}

async fn heartbeat_worker(
    State(state): State<AppState>,
    Path(worker_id): Path<String>,
    Json(payload): Json<WorkerHeartbeatRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    Ok(Json(state.jobs_store.heartbeat_worker(
        WorkerHeartbeat {
            worker_id,
            status: payload.status,
            current_job_id: payload.current_job_id,
            loaded_models: payload.loaded_models,
        },
    )?))
}

fn sweep_stale_workers(state: &AppState) -> Result<(), JobsStoreError> {
    state
        .jobs_store
        .mark_stale_workers_interrupted(state.settings.worker_timeout_seconds)
        .map(|_| ())
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
            other => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                detail: other.to_string(),
            },
        }
    }
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
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
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
}
