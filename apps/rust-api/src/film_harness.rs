//! Local filmmaking harness (epic 22708, sc-22710): render a hand-authored production plan into a
//! SceneWorks sequence through the EXISTING API seams — projects, asset import, `/api/v1/video/jobs`,
//! job polling, timelines and timeline export — and leave a versioned run record behind.
//!
//! The driver is written against [`ApiTransport`], a two-method abstraction over the HTTP API, so
//! the same code runs in-process against `create_app` in tests (with a fake worker claiming the
//! jobs) and over loopback against a live API + GPU worker from the `film-harness` binary. It never
//! renders anything itself and never talks to an engine: every take is produced by whatever worker
//! claims the job, exactly as a Video Studio submission would be.
//!
//! Order of operations, and the guarantee each step gives:
//!
//! 1. read + validate the plan and reference pack (structure, cross-references, files on disk);
//! 2. resolve the model's catalog entry and validate every shot against its declared modes,
//!    menus, caps and memory minimum ([`sceneworks_core::film_plan`]);
//! 3. confirm a registered worker advertises `video_generate`, read the host memory it reports,
//!    and check the plan's memory budget against it —
//!    **no job is created while any finding is outstanding** (the run record is still written,
//!    with outcome `rejected`);
//! 4. create/reuse the project and import every approved reference as a project asset tagged
//!    with its role;
//! 5. dispatch the selected shots one at a time under the plan's wall-clock, attempt and memory
//!    limits — a limit that trips cancels the in-flight job (cooperatively, through the API) and
//!    stops new dispatch;
//! 6. assemble the rendered takes on a timeline and export it through the `timeline_export` job;
//! 7. write `run.json` (shot -> attempt -> job -> asset, timeline, export, observed
//!    model/backend/hardware) beside copies of the two source documents.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use sceneworks_core::film_plan::{
    self, AttemptRecord, ConditioningAssets, ExportRecord, HardwareRecord, IntendedState,
    ModelLane, ModelRecord, PlanDiagnostic, ProductionPlan, ReferenceAssetRecord, ReferencePack,
    RunOutcome, RunRecord, ShotOutcome, ShotRunRecord, SourceDocument, TakeRecord,
    TimelineItemRecord, TimelineRecord, RUN_RECORD_SCHEMA_VERSION,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Map as JsonObject, Value};
use sha2::{Digest, Sha256};
use tokio::time::Instant;

/// Statuses the job store treats as terminal.
const TERMINAL_STATUSES: &[&str] = &["completed", "failed", "canceled", "interrupted"];

/// How long to wait for a canceled job to reach a terminal state before the harness records it as
/// timed out and moves on. The worker cancels cooperatively between stages, so this bounds the wait
/// rather than the worker.
const CANCEL_GRACE: Duration = Duration::from_secs(30);

/// Export resolutions the timeline export route admits (`validate_timeline_export`).
const EXPORT_RESOLUTIONS: &[u32] = &[640, 720, 1024, 1280];

/// Tag every harness-imported reference carries beside its role tag.
const REFERENCE_TAG: &str = "film-harness-reference";

/// One request to the SceneWorks API.
#[derive(Debug, Clone)]
pub struct ApiRequest {
    pub method: &'static str,
    pub path: String,
    pub body: RequestBody,
}

#[derive(Debug, Clone)]
pub enum RequestBody {
    None,
    Json(Value),
    /// A ready-encoded `multipart/form-data` body with its boundary.
    Multipart {
        boundary: String,
        bytes: Vec<u8>,
    },
}

/// The API's answer: HTTP status plus the JSON body (`Value::Null` when the body is empty or not
/// JSON).
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub status: u16,
    pub body: Value,
}

pub type TransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ApiResponse, HarnessError>> + Send + 'a>>;

/// The harness's only dependency on the outside world. Implemented over `reqwest` for the binary
/// and over an in-process `axum::Router` in tests.
pub trait ApiTransport: Send + Sync {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_>;
}

#[derive(Debug)]
pub enum HarnessError {
    /// The plan was refused before any job was created. The run record carries the same findings.
    Validation(Vec<PlanDiagnostic>),
    /// The API answered a non-success status the harness cannot proceed past.
    Api {
        method: &'static str,
        path: String,
        status: u16,
        detail: String,
    },
    /// The transport itself failed (connection refused, malformed response, ...).
    Transport(String),
    Io(String),
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(findings) => {
                writeln!(
                    f,
                    "plan refused before dispatch ({} finding(s)):",
                    findings.len()
                )?;
                for finding in findings {
                    writeln!(f, "  - {finding}")?;
                }
                Ok(())
            }
            Self::Api {
                method,
                path,
                status,
                detail,
            } => write!(f, "{method} {path} -> {status}: {detail}"),
            Self::Transport(message) => write!(f, "transport error: {message}"),
            Self::Io(message) => write!(f, "io error: {message}"),
        }
    }
}

impl std::error::Error for HarnessError {}

impl From<std::io::Error> for HarnessError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Everything a run needs beyond the transport.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub plan_path: PathBuf,
    pub reference_pack_path: PathBuf,
    /// Reuse this project instead of creating one named after the plan.
    pub project_id: Option<String>,
    /// Render only these shot ids (plan order is kept). `None` renders every shot.
    pub shot_ids: Option<Vec<String>>,
    /// Where `run.json` and the copied source documents are written.
    pub out_dir: PathBuf,
    /// Job polling cadence.
    pub poll_interval: Duration,
    /// Assemble and export the timeline after the shots render.
    pub export: bool,
    /// Refuse a model whose catalog entry is not `installState: "installed"`. Tests drive a fake
    /// worker that owns no weights and turn this off; a real run keeps it on so an uninstalled
    /// tier is refused here rather than by a failed job.
    pub require_installed: bool,
}

/// Encode a single-file `multipart/form-data` body the asset import route accepts: the `file`
/// field plus a JSON `provenance` field.
pub fn encode_asset_upload(
    filename: &str,
    content_type: &str,
    bytes: &[u8],
    provenance: &Value,
) -> (String, Vec<u8>) {
    let boundary = format!("SceneWorksFilmHarness{}", uuid::Uuid::new_v4().simple());
    let mut body = Vec::with_capacity(bytes.len() + 512);
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
            filename.replace('"', "")
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"provenance\"\r\n\r\n");
    body.extend_from_slice(provenance.to_string().as_bytes());
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (boundary, body)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn api_detail(body: &Value) -> String {
    body.get("detail")
        .map(|detail| match detail {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| body.to_string())
}

/// The harness's view of one job snapshot, read off `GET /api/v1/jobs/:id`.
#[derive(Debug, Clone)]
struct JobView {
    status: String,
    error: Option<String>,
    message: String,
    peak_gpu_memory_pct: Option<f64>,
    backend: Option<String>,
    result: Value,
}

impl JobView {
    fn from_snapshot(snapshot: &Value) -> Option<Self> {
        Some(Self {
            status: snapshot.get("status")?.as_str()?.to_owned(),
            error: snapshot
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_owned),
            message: snapshot
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            peak_gpu_memory_pct: snapshot.get("peakGpuMemoryPct").and_then(Value::as_f64),
            backend: snapshot
                .get("backend")
                .and_then(Value::as_str)
                .map(str::to_owned),
            result: snapshot.get("result").cloned().unwrap_or(Value::Null),
        })
    }

    fn is_terminal(&self) -> bool {
        TERMINAL_STATUSES.contains(&self.status.as_str())
    }

    /// Whether a terminal snapshot's result is final. The API commits a worker's terminal status
    /// BEFORE it persists the reported `assetWrites` into project assets and rewrites the result
    /// as `assets` / `assetIds` (the durable two-phase handoff in `update_job_progress`), so a
    /// poller can observe `completed` with the raw facts still in place. Reading that snapshot as
    /// done would record "completed without an asset".
    fn is_settled(&self) -> bool {
        self.status != "completed"
            || self
                .result
                .get("assetWrites")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
    }

    fn failure_text(&self) -> String {
        self.error
            .clone()
            .filter(|error| !error.trim().is_empty())
            .unwrap_or_else(|| self.message.clone())
    }
}

/// Why a poll loop returned before the job finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PollStop {
    Terminal,
    /// The attempt's own budget ran out.
    ShotBudget,
    /// The run's budget ran out.
    RunBudget,
}

struct Client<'a> {
    transport: &'a dyn ApiTransport,
}

impl Client<'_> {
    async fn json(
        &self,
        method: &'static str,
        path: &str,
        body: Option<Value>,
    ) -> Result<ApiResponse, HarnessError> {
        self.transport
            .call(ApiRequest {
                method,
                path: path.to_owned(),
                body: body.map_or(RequestBody::None, RequestBody::Json),
            })
            .await
    }

    /// A request that must succeed (2xx) for the run to continue.
    async fn expect_ok(
        &self,
        method: &'static str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, HarnessError> {
        let response = self.json(method, path, body).await?;
        if (200..300).contains(&response.status) {
            Ok(response.body)
        } else {
            Err(HarnessError::Api {
                method,
                path: path.to_owned(),
                status: response.status,
                detail: api_detail(&response.body),
            })
        }
    }

    async fn get_job(&self, job_id: &str) -> Result<JobView, HarnessError> {
        let snapshot = self
            .expect_ok("GET", &format!("/api/v1/jobs/{job_id}"), None)
            .await?;
        JobView::from_snapshot(&snapshot).ok_or_else(|| {
            HarnessError::Transport(format!("job {job_id} snapshot has no status: {snapshot}"))
        })
    }

    /// Poll `job_id` until it is terminal or a deadline passes. On a deadline the job is canceled
    /// through the API and given [`CANCEL_GRACE`] to settle; the returned view is the last one
    /// observed either way.
    async fn wait_for_job(
        &self,
        job_id: &str,
        shot_deadline: Instant,
        run_deadline: Instant,
        poll_interval: Duration,
    ) -> Result<(JobView, PollStop), HarnessError> {
        loop {
            let view = self.get_job(job_id).await?;
            if view.is_terminal() && view.is_settled() {
                return Ok((view, PollStop::Terminal));
            }
            let now = Instant::now();
            let stop = if now >= run_deadline {
                Some(PollStop::RunBudget)
            } else if now >= shot_deadline {
                Some(PollStop::ShotBudget)
            } else {
                None
            };
            if let Some(stop) = stop {
                // Cooperative cancel: the worker observes `cancelRequested` between stages. A queued
                // job cancels immediately; a running one settles when the worker next checks.
                let _ = self
                    .json("POST", &format!("/api/v1/jobs/{job_id}/cancel"), None)
                    .await?;
                let grace_deadline = Instant::now() + CANCEL_GRACE;
                let mut last = view;
                while Instant::now() < grace_deadline {
                    tokio::time::sleep(poll_interval).await;
                    last = self.get_job(job_id).await?;
                    if last.is_terminal() && last.is_settled() {
                        break;
                    }
                }
                return Ok((last, stop));
            }
            tokio::time::sleep(poll_interval).await;
        }
    }
}

/// What the run learned about the host and the worker that will render.
#[derive(Debug, Clone, Default)]
struct HostFacts {
    host_memory_gb: Option<f64>,
    video_worker_id: Option<String>,
    video_gpu_name: Option<String>,
    export_worker: bool,
}

async fn discover_host(client: &Client<'_>) -> Result<HostFacts, HarnessError> {
    let workers = client.expect_ok("GET", "/api/v1/workers", None).await?;
    let mut facts = HostFacts::default();
    for worker in workers.as_array().into_iter().flatten() {
        let capabilities: Vec<&str> = worker
            .get("capabilities")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if capabilities.contains(&"video_generate") && facts.video_worker_id.is_none() {
            facts.video_worker_id = worker.get("id").and_then(Value::as_str).map(str::to_owned);
            facts.video_gpu_name = worker
                .get("gpuName")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if capabilities.contains(&"timeline_export") {
            facts.export_worker = true;
        }
    }
    let host = client
        .expect_ok("GET", "/api/v1/host-capabilities", None)
        .await?;
    facts.host_memory_gb = host
        .get("memoryGb")
        .and_then(Value::as_f64)
        .or_else(|| host.get("unifiedMemoryGb").and_then(Value::as_f64))
        .or_else(|| host.get("gpuMemoryGb").and_then(Value::as_f64));
    Ok(facts)
}

/// The catalog entry for `model_id` as `/api/v1/models` serves it (manifest fields plus the
/// derived install state), or `None` when the catalog has no such model.
async fn resolve_model_entry(
    client: &Client<'_>,
    model_id: &str,
) -> Result<Option<JsonObject<String, Value>>, HarnessError> {
    let catalog = client.expect_ok("GET", "/api/v1/models", None).await?;
    Ok(catalog
        .as_array()
        .into_iter()
        .flatten()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .and_then(Value::as_object)
        .cloned())
}

/// Whether the catalog reports the requested tier (or, with no tier named, the model) installed.
fn model_tier_installed(entry: &JsonObject<String, Value>, tier: Option<&str>) -> bool {
    if let Some(tier) = tier {
        if let Some(variants) = entry.get("variants").and_then(Value::as_array) {
            if let Some(variant) = variants
                .iter()
                .find(|variant| variant.get("variant").and_then(Value::as_str) == Some(tier))
            {
                return variant
                    .get("installed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            }
        }
    }
    entry.get("installState").and_then(Value::as_str) == Some("installed")
}

/// The manifest download row that carries the requested tier's primary weights, for the record.
fn primary_weights(entry: &JsonObject<String, Value>, tier: Option<&str>) -> Option<Value> {
    let downloads = entry.get("downloads").and_then(Value::as_array)?;
    let is_primary = |row: &&Value| {
        !row.get("coRequisite")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    let row = match tier {
        Some(tier) => downloads
            .iter()
            .filter(is_primary)
            .find(|row| row.get("variant").and_then(Value::as_str) == Some(tier)),
        None => downloads
            .iter()
            .filter(is_primary)
            .find(|row| row.get("default").and_then(Value::as_bool).unwrap_or(false))
            .or_else(|| downloads.iter().find(is_primary)),
    }?;
    Some(json!({
        "provider": row.get("provider"),
        "repo": row.get("repo"),
        "revision": row.get("revision"),
        "variant": row.get("variant"),
        "files": row.get("files"),
    }))
}

fn mlx_quantize_for_tier(tier: &str) -> Value {
    match tier {
        "bf16" => json!(0),
        "q8" => json!(8),
        _ => json!(4),
    }
}

/// The `POST /api/v1/video/jobs` body for one attempt of `shot`, with every reference role already
/// resolved to the asset id the harness imported.
/// One attempt of one shot, resolved to everything the video route needs.
struct ShotDispatch<'a> {
    plan: &'a ProductionPlan,
    shot: &'a film_plan::Shot,
    project_id: &'a str,
    run_id: &'a str,
    attempt: u32,
    fps: u32,
    width: u32,
    height: u32,
    assets: &'a ConditioningAssets,
}

fn video_job_body(dispatch: &ShotDispatch<'_>) -> Value {
    let ShotDispatch {
        plan,
        shot,
        project_id,
        run_id,
        attempt,
        fps,
        width,
        height,
        assets,
    } = *dispatch;
    let mut advanced = JsonObject::new();
    if let Some(tier) = plan.model.tier.as_deref() {
        advanced.insert("mlxQuantize".to_owned(), mlx_quantize_for_tier(tier));
    }
    advanced.insert(
        "filmHarness".to_owned(),
        json!({
            "runId": run_id,
            "planId": plan.id,
            "planVersion": plan.version,
            "shotId": shot.id,
            "attempt": attempt,
        }),
    );
    let mut body = json!({
        "projectId": project_id,
        "mode": shot.conditioning.mode,
        "model": plan.model.id,
        "prompt": shot.prompt,
        "duration": shot.target_duration_seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "fitMode": "crop",
        "requestedGpu": "auto",
        "advanced": advanced,
    });
    if let Some(negative) = shot.negative_prompt.as_deref() {
        body["negativePrompt"] = json!(negative);
    }
    if let Some(seed) = shot.seed {
        body["seed"] = json!(seed);
    }
    if let Some(first) = &assets.first_frame_asset_id {
        body["sourceAssetId"] = json!(first);
    }
    if let Some(last) = &assets.last_frame_asset_id {
        body["lastFrameAssetId"] = json!(last);
    }
    if !assets.reference_asset_ids.is_empty() {
        body["referenceAssetIds"] = json!(assets.reference_asset_ids);
    }
    body
}

fn take_from_result(result: &Value, model: &str, backend: Option<&str>) -> Option<TakeRecord> {
    let asset = result.get("assets")?.as_array()?.first()?;
    let file = asset.get("file").unwrap_or(&Value::Null);
    let recipe = asset.get("recipe").unwrap_or(&Value::Null);
    Some(TakeRecord {
        asset_id: asset.get("id")?.as_str()?.to_owned(),
        media_path: file
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        encoded_duration_seconds: file.get("duration").and_then(Value::as_f64),
        encoded_fps: file.get("fps").and_then(Value::as_f64),
        encoded_frame_count: file.get("frameCount").and_then(Value::as_u64),
        has_audio: file.get("hasAudio").and_then(Value::as_bool),
        seed: recipe.get("seed").and_then(Value::as_i64),
        adapter: recipe
            .get("adapter")
            .and_then(Value::as_str)
            .map(str::to_owned),
        backend: backend.map(str::to_owned),
        model: recipe
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model)
            .to_owned(),
        raw_adapter_settings: recipe
            .get("rawAdapterSettings")
            .cloned()
            .unwrap_or(Value::Null),
    })
}

fn aspect_ratio_for(width: u32, height: u32) -> &'static str {
    match width.cmp(&height) {
        std::cmp::Ordering::Greater => "16:9",
        std::cmp::Ordering::Less => "9:16",
        std::cmp::Ordering::Equal => "1:1",
    }
}

fn export_resolution_for(height: u32) -> u32 {
    EXPORT_RESOLUTIONS
        .iter()
        .copied()
        .find(|candidate| *candidate >= height)
        .unwrap_or(1280)
}

fn seconds_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

/// Write `record` as `run.json` under `out_dir` and, when the project directory is reachable on
/// this filesystem, as `<project>/film-harness/<run_id>/run.json` too. Source documents are copied
/// beside the out-dir record so the run is self-describing.
fn persist_record(
    record: &RunRecord,
    out_dir: &Path,
    plan_path: &Path,
    pack_path: &Path,
) -> Result<PathBuf, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let json = serde_json::to_string_pretty(&record.to_json())
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    let record_path = out_dir.join("run.json");
    std::fs::write(&record_path, &json)?;
    if let Ok(plan_text) = std::fs::read(plan_path) {
        std::fs::write(out_dir.join("plan.json"), plan_text)?;
    }
    if let Ok(pack_text) = std::fs::read(pack_path) {
        std::fs::write(out_dir.join("references.json"), pack_text)?;
    }
    if let Some(project_path) = record.project_path.as_deref().map(Path::new) {
        if project_path.is_dir() {
            let project_record_dir = project_path.join("film-harness").join(&record.run_id);
            std::fs::create_dir_all(&project_record_dir)?;
            std::fs::write(project_record_dir.join("run.json"), &json)?;
        }
    }
    Ok(record_path)
}

/// Read, validate and (optionally) check the documents against the live catalog without creating
/// anything. What `run` does before its first write, exposed for the `validate` subcommand.
pub async fn validate(
    transport: Option<&dyn ApiTransport>,
    options: &RunOptions,
) -> Result<(ProductionPlan, ReferencePack), HarnessError> {
    let plan = film_plan::read_plan_file(&options.plan_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack = film_plan::read_reference_pack_file(&options.reference_pack_path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    let pack_dir = options
        .reference_pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut findings = film_plan::validate_all(&plan, &pack, Some(&pack_dir), None);
    findings.extend(selection_findings(&plan, options.shot_ids.as_deref()));
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    if let Some(transport) = transport {
        let client = Client { transport };
        let entry = resolve_model_entry(&client, &plan.model.id).await?;
        let mut findings = model_findings(&plan, entry.as_ref(), options.require_installed);
        if findings.is_empty() {
            let facts = discover_host(&client).await?;
            findings.extend(host_findings(&plan, &facts, options.export));
        }
        if !findings.is_empty() {
            return Err(HarnessError::Validation(findings));
        }
    }
    Ok((plan, pack))
}

fn selection_findings(plan: &ProductionPlan, selection: Option<&[String]>) -> Vec<PlanDiagnostic> {
    let Some(selection) = selection else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    if selection.is_empty() {
        findings.push(PlanDiagnostic::plan(
            "selection",
            "at least one shot id must be selected",
        ));
    }
    for id in selection {
        if !plan.shots.iter().any(|shot| &shot.id == id) {
            findings.push(PlanDiagnostic::plan(
                "selection",
                format!(
                    "selected shot {id:?} is not in plan {:?} (shots: {})",
                    plan.id,
                    plan.shots
                        .iter()
                        .map(|shot| shot.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
    }
    findings
}

fn model_findings(
    plan: &ProductionPlan,
    entry: Option<&JsonObject<String, Value>>,
    require_installed: bool,
) -> Vec<PlanDiagnostic> {
    let Some(entry) = entry else {
        return vec![PlanDiagnostic::plan(
            "model.id",
            format!("{:?} is not in this API's model catalog", plan.model.id),
        )];
    };
    let mut findings = Vec::new();
    if entry.get("type").and_then(Value::as_str) != Some("video") {
        findings.push(PlanDiagnostic::plan(
            "model.id",
            format!("{:?} is not a video model", plan.model.id),
        ));
    }
    if require_installed && !model_tier_installed(entry, plan.model.tier.as_deref()) {
        findings.push(PlanDiagnostic::plan(
            "model.tier",
            format!(
                "{}{} is not installed on this host (catalog installState is not \"installed\"); \
                 download it in the Model Manager first",
                plan.model.id,
                plan.model
                    .tier
                    .as_deref()
                    .map(|tier| format!(" tier {tier}"))
                    .unwrap_or_default()
            ),
        ));
    }
    findings.extend(film_plan::validate_plan_against_model(
        plan,
        entry,
        ModelLane::for_current_platform(),
    ));
    findings
}

fn host_findings(plan: &ProductionPlan, facts: &HostFacts, export: bool) -> Vec<PlanDiagnostic> {
    let mut findings = Vec::new();
    if facts.video_worker_id.is_none() {
        findings.push(PlanDiagnostic::plan(
            "model.id",
            "no registered worker advertises video_generate; start the GPU worker \
             (SCENEWORKS_WORKER_ONLY=1) and wait for it to register",
        ));
    }
    if export && !facts.export_worker {
        findings.push(PlanDiagnostic::plan(
            "export",
            "no registered worker advertises timeline_export; run the API with \
             SCENEWORKS_RUN_UTILITY_INPROCESS=1 or start a utility worker",
        ));
    }
    match facts.host_memory_gb {
        Some(host) if plan.limits.max_memory_gb > host => findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            format!(
                "budget {} GB exceeds the {host:.1} GB the registered worker reports for this host",
                plan.limits.max_memory_gb
            ),
        )),
        Some(_) => {}
        None => findings.push(PlanDiagnostic::plan(
            "limits.maxMemoryGb",
            "no registered worker reports host memory, so the memory budget cannot be checked \
             before dispatch",
        )),
    }
    findings
}

/// Execute `options` end to end. Returns the run record (also written to `options.out_dir`) on
/// every path that got past validation, including runs that stopped on a limit; a refused plan
/// returns [`HarnessError::Validation`] after writing a `rejected` record.
pub async fn run(
    transport: &dyn ApiTransport,
    options: &RunOptions,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let run_id = format!("run_{}", uuid::Uuid::new_v4().simple());
    let plan_bytes = std::fs::read(&options.plan_path)?;
    let pack_bytes = std::fs::read(&options.reference_pack_path)?;
    let client = Client { transport };

    // Steps 1-3: refuse before the first write.
    let (plan, pack) = match validate(None, options).await {
        Ok(documents) => documents,
        Err(HarnessError::Validation(findings)) => {
            let record = rejected_record(
                &run_id,
                options,
                &plan_bytes,
                &pack_bytes,
                findings.clone(),
                started,
            );
            persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            )?;
            return Err(HarnessError::Validation(findings));
        }
        Err(other) => return Err(other),
    };
    let entry = resolve_model_entry(&client, &plan.model.id).await?;
    let mut findings = model_findings(&plan, entry.as_ref(), options.require_installed);
    let facts = if findings.is_empty() {
        let facts = discover_host(&client).await?;
        findings.extend(host_findings(&plan, &facts, options.export));
        facts
    } else {
        HostFacts::default()
    };
    if !findings.is_empty() {
        let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
        record.outcome = RunOutcome::Rejected;
        record.diagnostics = findings.clone();
        record.finished_at = Some(utc_now());
        record.elapsed_seconds = seconds_since(started);
        persist_record(
            &record,
            &options.out_dir,
            &options.plan_path,
            &options.reference_pack_path,
        )?;
        return Err(HarnessError::Validation(findings));
    }
    let entry = entry.expect("model findings are empty only with an entry");
    let fps = film_plan::plan_fps(&plan, &entry).expect("validated against the model");
    let lane = ModelLane::for_current_platform();

    let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
    record.model = Some(ModelRecord {
        id: plan.model.id.clone(),
        tier_requested: plan.model.tier.clone(),
        fps,
        lane: lane.manifest_key().to_owned(),
        backend_observed: None,
        weights: primary_weights(&entry, plan.model.tier.as_deref()),
        hardware: HardwareRecord {
            platform: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            host_memory_gb: facts.host_memory_gb,
            gpu_name: facts.video_gpu_name.clone(),
            worker_id: facts.video_worker_id.clone(),
        },
    });

    // Step 4: project + references.
    let project = match &options.project_id {
        Some(id) => {
            client
                .expect_ok("GET", &format!("/api/v1/projects/{id}"), None)
                .await?
        }
        None => {
            client
                .expect_ok(
                    "POST",
                    "/api/v1/projects",
                    Some(json!({ "name": plan.title })),
                )
                .await?
        }
    };
    let project_id = project
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| HarnessError::Transport(format!("project response has no id: {project}")))?
        .to_owned();
    record.project_id = Some(project_id.clone());
    record.project_path = project
        .get("path")
        .and_then(Value::as_str)
        .map(str::to_owned);

    let pack_dir = options
        .reference_pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut role_assets: BTreeMap<String, String> = BTreeMap::new();
    for reference in &pack.references {
        let path = pack_dir.join(&reference.file);
        let bytes = std::fs::read(&path)?;
        let sha256 = sha256_hex(&bytes);
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("reference.png")
            .to_owned();
        let content_type = match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("jpg" | "jpeg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };
        let provenance = json!({
            "filmHarness": {
                "kind": "reference",
                "role": reference.role,
                "referenceKind": reference.kind,
                "approved": reference.approved,
                "referencePackId": pack.id,
                "referencePackVersion": pack.version,
                "planId": plan.id,
                "planVersion": plan.version,
                "runId": run_id,
                "sourceFile": reference.file,
                "sha256": sha256,
            }
        });
        let (boundary, body) = encode_asset_upload(&filename, content_type, &bytes, &provenance);
        let path = format!("/api/v1/projects/{project_id}/assets");
        let response = transport
            .call(ApiRequest {
                method: "POST",
                path: path.clone(),
                body: RequestBody::Multipart {
                    boundary,
                    bytes: body,
                },
            })
            .await?;
        if !(200..300).contains(&response.status) {
            return Err(HarnessError::Api {
                method: "POST",
                path,
                status: response.status,
                detail: api_detail(&response.body),
            });
        }
        let asset_id = response
            .body
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "asset import response has no id: {}",
                    response.body
                ))
            })?
            .to_owned();
        client
            .expect_ok(
                "PATCH",
                &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                Some(json!({
                    "tags": [REFERENCE_TAG, format!("role:{}", reference.role), format!("pack:{}", pack.id)]
                })),
            )
            .await?;
        role_assets.insert(reference.role.clone(), asset_id.clone());
        record.references.push(ReferenceAssetRecord {
            role: reference.role.clone(),
            kind: reference.kind.clone(),
            file: reference.file.clone(),
            sha256,
            asset_id,
        });
    }

    // Step 5: shots, one at a time, under the declared limits.
    let run_deadline = started + Duration::from_secs(plan.limits.max_run_seconds);
    let shot_budget = Duration::from_secs(plan.limits.max_shot_seconds);
    let selected: Vec<&film_plan::Shot> = plan
        .shots
        .iter()
        .filter(|shot| record.selected_shot_ids.contains(&shot.id))
        .collect();
    let mut stop: Option<RunOutcome> = None;
    let mut rendered: Vec<(String, TakeRecord, (u32, u32))> = Vec::new();
    for shot in &plan.shots {
        let (width, height) =
            film_plan::shot_resolution(&plan, shot, &entry).expect("validated against the model");
        let conditioning = &shot.conditioning;
        let assets = ConditioningAssets {
            first_frame_asset_id: conditioning
                .first_frame_role
                .as_ref()
                .and_then(|role| role_assets.get(role).cloned()),
            last_frame_asset_id: conditioning
                .last_frame_role
                .as_ref()
                .and_then(|role| role_assets.get(role).cloned()),
            reference_asset_ids: conditioning
                .reference_roles
                .iter()
                .filter_map(|role| role_assets.get(role).cloned())
                .collect(),
        };
        let mut shot_record = ShotRunRecord {
            shot_id: shot.id.clone(),
            outcome: ShotOutcome::NotSelected,
            intended: IntendedState {
                mode: conditioning.mode.clone(),
                start_state: shot.start_state.clone(),
                end_state: shot.end_state.clone(),
                target_duration_seconds: shot.target_duration_seconds,
                width,
                height,
                fps,
                dialogue: shot.dialogue.clone(),
                sound: shot.sound.clone(),
            },
            conditioning_assets: assets.clone(),
            attempts: Vec::new(),
        };
        if !selected.iter().any(|candidate| candidate.id == shot.id) {
            record.shots.push(shot_record);
            continue;
        }
        if stop.is_some() {
            shot_record.outcome = ShotOutcome::NotDispatched;
            record.shots.push(shot_record);
            continue;
        }
        let mut outcome = ShotOutcome::Failed;
        for attempt in 1..=plan.limits.max_attempts_per_shot {
            if Instant::now() >= run_deadline {
                stop = Some(RunOutcome::StoppedRunBudget);
                break;
            }
            let attempt_started = Instant::now();
            let mut attempt_record = AttemptRecord {
                attempt,
                job_id: None,
                status: "dispatching".to_owned(),
                started_at: utc_now(),
                finished_at: None,
                elapsed_seconds: 0.0,
                peak_gpu_memory_pct: None,
                error: None,
                take: None,
            };
            let body = video_job_body(&ShotDispatch {
                plan: &plan,
                shot,
                project_id: &project_id,
                run_id: &run_id,
                attempt,
                fps,
                width,
                height,
                assets: &assets,
            });
            let response = client
                .json("POST", "/api/v1/video/jobs", Some(body))
                .await?;
            if !(200..300).contains(&response.status) {
                // A refused enqueue is deterministic: the same body would be refused again, so it
                // consumes the shot rather than every remaining attempt.
                attempt_record.status = "rejected".to_owned();
                attempt_record.error = Some(format!(
                    "POST /api/v1/video/jobs -> {}: {}",
                    response.status,
                    api_detail(&response.body)
                ));
                attempt_record.finished_at = Some(utc_now());
                attempt_record.elapsed_seconds = seconds_since(attempt_started);
                shot_record.attempts.push(attempt_record);
                outcome = ShotOutcome::Failed;
                break;
            }
            let job_id = response
                .body
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HarnessError::Transport(format!(
                        "video job response has no id: {}",
                        response.body
                    ))
                })?
                .to_owned();
            attempt_record.job_id = Some(job_id.clone());
            let shot_deadline = attempt_started + shot_budget;
            let (view, poll_stop) = client
                .wait_for_job(&job_id, shot_deadline, run_deadline, options.poll_interval)
                .await?;
            attempt_record.finished_at = Some(utc_now());
            attempt_record.elapsed_seconds = seconds_since(attempt_started);
            attempt_record.peak_gpu_memory_pct = view.peak_gpu_memory_pct;
            attempt_record.status = match poll_stop {
                PollStop::Terminal => view.status.clone(),
                PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
            };
            let memory_exceeded = match (view.peak_gpu_memory_pct, facts.host_memory_gb) {
                (Some(pct), Some(host)) => host * pct / 100.0 > plan.limits.max_memory_gb,
                _ => false,
            };
            match poll_stop {
                PollStop::Terminal if view.status == "completed" => {
                    match take_from_result(&view.result, &plan.model.id, view.backend.as_deref()) {
                        Some(take) => {
                            if let Some(model) = record.model.as_mut() {
                                if model.backend_observed.is_none() {
                                    model.backend_observed = take.backend.clone();
                                }
                            }
                            rendered.push((shot.id.clone(), take.clone(), (width, height)));
                            attempt_record.take = Some(take);
                            shot_record.attempts.push(attempt_record);
                            outcome = ShotOutcome::Rendered;
                        }
                        None => {
                            attempt_record.error = Some(format!(
                                "job {job_id} completed without an asset in its result: {}",
                                view.result
                            ));
                            shot_record.attempts.push(attempt_record);
                            outcome = ShotOutcome::Failed;
                        }
                    }
                    if memory_exceeded {
                        stop = Some(RunOutcome::StoppedMemoryLimit);
                    }
                    break;
                }
                PollStop::Terminal => {
                    attempt_record.error = Some(view.failure_text());
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::Failed;
                    if memory_exceeded {
                        stop = Some(RunOutcome::StoppedMemoryLimit);
                        break;
                    }
                }
                PollStop::ShotBudget => {
                    attempt_record.error = Some(format!(
                        "attempt exceeded the per-shot budget of {}s (last status {})",
                        plan.limits.max_shot_seconds, view.status
                    ));
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::TimedOut;
                    if memory_exceeded {
                        stop = Some(RunOutcome::StoppedMemoryLimit);
                        break;
                    }
                }
                PollStop::RunBudget => {
                    attempt_record.error = Some(format!(
                        "run exceeded its budget of {}s while this attempt was in flight (last \
                         status {})",
                        plan.limits.max_run_seconds, view.status
                    ));
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::TimedOut;
                    stop = Some(RunOutcome::StoppedRunBudget);
                    break;
                }
            }
        }
        shot_record.outcome = if shot_record.attempts.is_empty() && stop.is_some() {
            // The run budget was already spent before this shot's first attempt could start.
            ShotOutcome::NotDispatched
        } else {
            outcome
        };
        record.shots.push(shot_record);
    }

    // Step 6: timeline + export from whatever rendered.
    let mut export_ok = !options.export;
    if options.export && !rendered.is_empty() && stop != Some(RunOutcome::StoppedRunBudget) {
        let (_, _, (first_width, first_height)) = &rendered[0];
        let aspect_ratio = aspect_ratio_for(*first_width, *first_height);
        let timeline_name = format!("{} ({})", plan.title, run_id);
        let mut timeline = client
            .expect_ok(
                "POST",
                &format!("/api/v1/projects/{project_id}/timelines"),
                Some(json!({ "name": timeline_name, "aspectRatio": aspect_ratio, "fps": fps })),
            )
            .await?;
        let timeline_id = timeline
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("timeline response has no id: {timeline}"))
            })?
            .to_owned();
        let mut items = Vec::new();
        let mut item_records = Vec::new();
        let mut cursor = 0.0_f64;
        let mut tallest = 0_u32;
        for (shot_id, take, (_, height)) in &rendered {
            let shot = plan
                .shots
                .iter()
                .find(|shot| &shot.id == shot_id)
                .expect("rendered shots come from the plan");
            let length = take
                .encoded_duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(shot.target_duration_seconds);
            let item_id = format!("item_{}_{}", shot.id.to_ascii_lowercase(), &run_id[4..12]);
            let end = cursor + length;
            items.push(json!({
                "id": item_id,
                "trackId": "track_main",
                "assetId": take.asset_id,
                "type": "video",
                "displayName": format!("{} — {}", shot.id, shot.beat).chars().take(160).collect::<String>(),
                "sourceIn": 0.0,
                "sourceOut": length,
                "timelineStart": cursor,
                "timelineEnd": end,
                "speed": 1.0,
                "fit": "fit",
                "volume": 1.0,
                "versionHistory": [{
                    "assetId": take.asset_id,
                    "source": "original",
                    "jobId": record.shots.iter().find(|s| &s.shot_id == shot_id)
                        .and_then(|s| s.attempts.last()).and_then(|a| a.job_id.clone()),
                    "note": format!("film-harness {run_id} shot {}", shot.id),
                }],
            }));
            item_records.push(TimelineItemRecord {
                shot_id: shot.id.clone(),
                item_id,
                asset_id: take.asset_id.clone(),
                timeline_start: cursor,
                timeline_end: end,
            });
            cursor = end;
            tallest = tallest.max(*height);
        }
        if let Some(track) = timeline
            .get_mut("tracks")
            .and_then(Value::as_array_mut)
            .and_then(|tracks| {
                tracks
                    .iter_mut()
                    .find(|track| track.get("id").and_then(Value::as_str) == Some("track_main"))
            })
        {
            track["items"] = Value::Array(items);
        }
        client
            .expect_ok(
                "PUT",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
                Some(json!({ "timeline": timeline })),
            )
            .await?;
        record.timeline = Some(TimelineRecord {
            timeline_id: timeline_id.clone(),
            name: timeline_name,
            aspect_ratio: aspect_ratio.to_owned(),
            fps,
            items: item_records,
        });

        let export_job = client
            .expect_ok(
                "POST",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
                Some(json!({
                    "resolution": export_resolution_for(tallest),
                    "fps": fps,
                    "requestedGpu": "auto",
                })),
            )
            .await?;
        let export_job_id = export_job
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("export response has no id: {export_job}"))
            })?
            .to_owned();
        let export_started = Instant::now();
        let (view, poll_stop) = client
            .wait_for_job(
                &export_job_id,
                export_started + shot_budget,
                run_deadline,
                options.poll_interval,
            )
            .await?;
        let asset_id = view
            .result
            .get("assetIds")
            .and_then(Value::as_array)
            .and_then(|ids| ids.first())
            .and_then(Value::as_str)
            .map(str::to_owned);
        let render_path = view
            .result
            .get("renderPath")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let status = match poll_stop {
            PollStop::Terminal => view.status.clone(),
            PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        };
        export_ok = status == "completed" && asset_id.is_some();
        record.export = Some(ExportRecord {
            job_id: export_job_id,
            status,
            asset_id,
            render_path,
            error: (!export_ok).then(|| match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::ShotBudget => format!(
                    "export exceeded the per-job budget of {}s",
                    plan.limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "run exceeded its budget of {}s during the export",
                    plan.limits.max_run_seconds
                ),
            }),
        });
        if poll_stop == PollStop::RunBudget {
            stop = Some(RunOutcome::StoppedRunBudget);
        }
    }

    let all_rendered = record
        .shots
        .iter()
        .filter(|shot| record.selected_shot_ids.contains(&shot.shot_id))
        .all(|shot| shot.outcome == ShotOutcome::Rendered);
    record.outcome = match stop {
        Some(outcome) => outcome,
        None if all_rendered && export_ok => RunOutcome::Completed,
        None => RunOutcome::Failed,
    };
    record.finished_at = Some(utc_now());
    record.elapsed_seconds = seconds_since(started);
    persist_record(
        &record,
        &options.out_dir,
        &options.plan_path,
        &options.reference_pack_path,
    )?;
    Ok(record)
}

fn base_record(
    run_id: &str,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    options: &RunOptions,
    plan_bytes: &[u8],
    pack_bytes: &[u8],
) -> RunRecord {
    let selected_shot_ids = match &options.shot_ids {
        Some(ids) => plan
            .shots
            .iter()
            .filter(|shot| ids.contains(&shot.id))
            .map(|shot| shot.id.clone())
            .collect(),
        None => plan.shots.iter().map(|shot| shot.id.clone()).collect(),
    };
    RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        created_at: utc_now(),
        finished_at: None,
        outcome: RunOutcome::Failed,
        plan: SourceDocument {
            id: plan.id.clone(),
            version: plan.version,
            path: options.plan_path.display().to_string(),
            sha256: sha256_hex(plan_bytes),
        },
        reference_pack: SourceDocument {
            id: pack.id.clone(),
            version: pack.version,
            path: options.reference_pack_path.display().to_string(),
            sha256: sha256_hex(pack_bytes),
        },
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits: plan.limits.clone(),
        selected_shot_ids,
        references: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        diagnostics: Vec::new(),
        elapsed_seconds: 0.0,
    }
}

/// A `rejected` record for a plan that failed document-level validation — the documents may not
/// even have parsed, so ids and limits are read leniently from the raw JSON.
fn rejected_record(
    run_id: &str,
    options: &RunOptions,
    plan_bytes: &[u8],
    pack_bytes: &[u8],
    findings: Vec<PlanDiagnostic>,
    started: Instant,
) -> RunRecord {
    let lenient = |bytes: &[u8]| -> Value {
        std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| {
                serde_json::from_str::<Value>(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                    .ok()
            })
            .unwrap_or(Value::Null)
    };
    let plan_value = lenient(plan_bytes);
    let pack_value = lenient(pack_bytes);
    let string_at = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let version_at = |value: &Value| {
        value
            .get("version")
            .and_then(Value::as_u64)
            .and_then(|version| u32::try_from(version).ok())
            .unwrap_or(0)
    };
    let limits = plan_value
        .get("limits")
        .cloned()
        .and_then(|limits| serde_json::from_value(limits).ok())
        .unwrap_or(film_plan::PlanLimits {
            max_run_seconds: 0,
            max_shot_seconds: 0,
            max_attempts_per_shot: 0,
            max_memory_gb: 0.0,
        });
    RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        created_at: utc_now(),
        finished_at: Some(utc_now()),
        outcome: RunOutcome::Rejected,
        plan: SourceDocument {
            id: string_at(&plan_value, "id"),
            version: version_at(&plan_value),
            path: options.plan_path.display().to_string(),
            sha256: sha256_hex(plan_bytes),
        },
        reference_pack: SourceDocument {
            id: string_at(&pack_value, "id"),
            version: version_at(&pack_value),
            path: options.reference_pack_path.display().to_string(),
            sha256: sha256_hex(pack_bytes),
        },
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits,
        selected_shot_ids: options.shot_ids.clone().unwrap_or_default(),
        references: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        diagnostics: findings,
        elapsed_seconds: seconds_since(started),
    }
}

// ---------------------------------------------------------------------------------------------
// Fixture images
// ---------------------------------------------------------------------------------------------

/// The courier/workshop/red-parcel reference roles and the flat colour each placeholder plate is
/// painted with. Deterministic, so the checked-in PNGs under
/// `config/film-harness/courier-workshop/references/` are reproducible byte for byte and a test can
/// prove it.
pub const FIXTURE_REFERENCES: &[(&str, [u8; 3])] = &[
    ("courier", [64, 80, 120]),
    ("recipient", [120, 96, 64]),
    ("red_parcel", [200, 32, 32]),
    ("workshop_location", [96, 88, 72]),
    ("workbench_table", [140, 110, 70]),
    ("house_style", [40, 40, 48]),
    ("workshop_plate", [84, 76, 64]),
];

/// Width and height of every fixture plate: MiniMax-H3's cheapest declared canvas, so a keyframe
/// plate needs no resampling.
pub const FIXTURE_PLATE_SIZE: (u32, u32) = (576, 320);

/// Paint one deterministic placeholder plate: a flat field with a lighter horizontal band (a
/// "table line") and a darker margin, so the plates are distinguishable at a glance and give an
/// image-conditioned model something other than a single colour.
pub fn fixture_plate_png(role: &str, rgb: [u8; 3]) -> Result<Vec<u8>, HarnessError> {
    let (width, height) = FIXTURE_PLATE_SIZE;
    let mut image = image::RgbImage::new(width, height);
    let seed = role.bytes().fold(7_u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u32::from(byte))
    });
    let band_top = height * 3 / 5;
    let band_bottom = band_top + height / 12;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let mut color = rgb;
        if x < 16 || x >= width - 16 || y < 16 || y >= height - 16 {
            color = color.map(|channel| channel.saturating_sub(24));
        } else if (band_top..band_bottom).contains(&y) {
            color = color.map(|channel| channel.saturating_add(40));
        } else if (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13)) ^ seed) % 97 == 0 {
            color = color.map(|channel| channel.saturating_add(12));
        }
        *pixel = image::Rgb(color);
    }
    let mut bytes = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut bytes,
        image::codecs::png::CompressionType::Default,
        image::codecs::png::FilterType::NoFilter,
    );
    image::ImageEncoder::write_image(
        encoder,
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )
    .map_err(|error| HarnessError::Io(error.to_string()))?;
    Ok(bytes)
}

/// Write every fixture plate as `<out_dir>/<role>.png`; returns the written paths.
pub fn write_fixture_images(out_dir: &Path) -> Result<Vec<PathBuf>, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for (role, rgb) in FIXTURE_REFERENCES {
        let path = out_dir.join(format!("{role}.png"));
        std::fs::write(&path, fixture_plate_png(role, *rgb)?)?;
        written.push(path);
    }
    Ok(written)
}

// ---------------------------------------------------------------------------------------------
// HTTP transport (the binary's)
// ---------------------------------------------------------------------------------------------

/// [`ApiTransport`] over `reqwest` against a running SceneWorks API.
pub struct HttpTransport {
    base_url: String,
    token: Option<String>,
    client: reqwest::Client,
}

impl HttpTransport {
    pub fn new(base_url: &str, token: Option<String>) -> Result<Self, HarnessError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| HarnessError::Transport(error.to_string()))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            token: token.filter(|token| !token.trim().is_empty()),
            client,
        })
    }
}

impl ApiTransport for HttpTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        Box::pin(async move {
            let method = reqwest::Method::from_bytes(request.method.as_bytes())
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let url = format!("{}{}", self.base_url, request.path);
            let mut builder = self.client.request(method, &url);
            if let Some(token) = &self.token {
                builder = builder.header("x-sceneworks-token", token);
            }
            builder = match request.body {
                RequestBody::None => builder,
                RequestBody::Json(value) => builder.json(&value),
                RequestBody::Multipart { boundary, bytes } => builder
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(bytes),
            };
            let response = builder
                .send()
                .await
                .map_err(|error| HarnessError::Transport(format!("{url}: {error}")))?;
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .map_err(|error| HarnessError::Transport(format!("{url}: {error}")))?;
            let body = if text.trim().is_empty() {
                Value::Null
            } else {
                serde_json::from_str(&text).unwrap_or(Value::String(text))
            };
            Ok(ApiResponse { status, body })
        })
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn multipart_body_carries_file_and_provenance_fields() {
        let (boundary, body) =
            encode_asset_upload("plate.png", "image/png", b"\x89PNG", &json!({ "a": 1 }));
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.contains("name=\"file\"; filename=\"plate.png\""));
        assert!(text.contains("Content-Type: image/png"));
        assert!(text.contains("name=\"provenance\"\r\n\r\n{\"a\":1}"));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
    }

    #[test]
    fn tier_maps_to_the_shared_mlx_quantize_convention() {
        assert_eq!(mlx_quantize_for_tier("q4"), json!(4));
        assert_eq!(mlx_quantize_for_tier("q8"), json!(8));
        assert_eq!(mlx_quantize_for_tier("bf16"), json!(0));
    }

    #[test]
    fn export_resolution_is_the_smallest_admitted_value_covering_the_takes() {
        assert_eq!(export_resolution_for(320), 640);
        assert_eq!(export_resolution_for(720), 720);
        assert_eq!(export_resolution_for(768), 1024);
        assert_eq!(export_resolution_for(2000), 1280);
        assert_eq!(aspect_ratio_for(576, 320), "16:9");
        assert_eq!(aspect_ratio_for(320, 576), "9:16");
        assert_eq!(aspect_ratio_for(768, 768), "1:1");
    }

    #[test]
    fn fixture_plates_are_deterministic_and_decode_at_the_declared_size() {
        let first = fixture_plate_png("red_parcel", [200, 32, 32]).unwrap();
        let second = fixture_plate_png("red_parcel", [200, 32, 32]).unwrap();
        assert_eq!(first, second);
        let decoded = image::load_from_memory(&first).unwrap();
        assert_eq!((decoded.width(), decoded.height()), FIXTURE_PLATE_SIZE);
        assert_ne!(first, fixture_plate_png("courier", [64, 80, 120]).unwrap());
    }

    #[test]
    fn installed_state_reads_the_requested_tier_variant_first() {
        let entry: JsonObject<String, Value> = json!({
            "installState": "installed",
            "variants": [
                { "variant": "q4", "installed": true },
                { "variant": "q8", "installed": false }
            ]
        })
        .as_object()
        .cloned()
        .unwrap();
        assert!(model_tier_installed(&entry, Some("q4")));
        assert!(!model_tier_installed(&entry, Some("q8")));
        assert!(model_tier_installed(&entry, None));
        assert!(
            model_tier_installed(&entry, Some("bf16")),
            "unlisted tier falls back to the model state"
        );
    }
}
