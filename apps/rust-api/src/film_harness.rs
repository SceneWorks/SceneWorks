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
//! 2. read the API HOST's platform and memory from `GET /api/v1/host-capabilities` — `--api` may
//!    point at another machine, so the host's platform, not `cfg!(target_os)`, decides the lane
//!    and is what the record claims as the render hardware;
//! 3. resolve the model's catalog entry and validate every shot against its declared modes,
//!    menus, caps and lane memory minimum ([`sceneworks_core::film_plan`]), plus the enqueue
//!    route's own platform-reachability and reference-payload gates, and confirm a registered
//!    worker advertises `video_generate` —
//!    **no job is created while any finding is outstanding** (the run record is still written,
//!    with outcome `rejected`);
//! 4. create/reuse the project and import every reference as a project asset tagged with its role
//!    (an unapproved one is tagged apart and never used as conditioning);
//! 5. dispatch the selected shots one at a time under the plan's wall-clock, attempt and memory
//!    limits — a limit that trips cancels the in-flight job (cooperatively, through the API) and
//!    stops new dispatch; the observed peak comes from the job's metrics block
//!    (`GET /api/v1/jobs/:id/metrics`), which is where a real worker reports it;
//! 6. assemble the rendered takes on a timeline and export it through the `timeline_export` job;
//! 7. write `run.json` (shot -> attempt -> job -> asset, timeline, export, observed
//!    model/backend/hardware) beside copies of the two source documents — on every path past
//!    step 1, refusals and mid-run failures included.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sceneworks_core::film_compile::{
    compile_plan, CompileInputs, CompiledPlan, DispatchContext, ResolvedConditioning,
};
use sceneworks_core::film_plan::{
    self, AttemptRecord, ConditioningAssets, ExportRecord, GeneratedAudio, HardwareRecord,
    IntendedState, ModelLane, ModelRecord, PlanDiagnostic, ProductionPlan, ReferenceAssetRecord,
    ReferencePack, RunOutcome, RunRecord, ShotOutcome, ShotRunRecord, SoundBed, SoundBus,
    SourceDocument, TakeRecord, TimelineEditRecord, TimelineItemRecord, TimelineRecord,
    TimelineTrackRecord, RUN_RECORD_SCHEMA_VERSION,
};
use sceneworks_core::time::utc_now;
use serde_json::{json, Map as JsonObject, Value};
use sha2::{Digest, Sha256};
use tokio::time::Instant;

/// Statuses the job store treats as terminal.
const TERMINAL_STATUSES: &[&str] = &["completed", "failed", "canceled", "interrupted"];

/// How long to wait for a canceled job to reach a terminal state before the harness gives up on it.
/// The worker cancels cooperatively between stages, so this bounds the wait rather than the worker.
/// Capped at the plan's own per-shot budget, so a plan that declares a short shot also gets a short
/// grace.
const CANCEL_GRACE: Duration = Duration::from_secs(30);

/// How long a TERMINAL job may keep raw `assetWrites` in its result — the window between the
/// worker's terminal status and the API's asset persistence (`persist_reported_assets`) — before
/// the harness stops waiting and records the attempt as terminal-but-unsettled. Also capped at the
/// per-shot budget.
const ASSET_SETTLE_GRACE: Duration = Duration::from_secs(30);

/// How long to keep re-reading `GET /api/v1/jobs/:id/metrics` after a terminal attempt. The worker
/// POSTs its metrics block AFTER the terminal progress update, so the first read can legitimately
/// come back `null`.
const METRICS_GRACE: Duration = Duration::from_secs(5);

/// Cadence of the metrics re-read inside [`METRICS_GRACE`] (independent of the job poll cadence,
/// which a real run sets to seconds).
const METRICS_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// How close two `ln(w/h)` distances must be to count as a tie when picking a timeline aspect
/// ratio. 4:3 is EXACTLY equidistant from 1:1 and 16:9 on that scale (4/3 is their geometric
/// mean), so the tie is real arithmetic, not float noise, and must be broken deliberately.
const ASPECT_TIE_EPSILON: f64 = 1e-9;

/// Bytes per GB in every memory comparison the harness makes: GiB. `GET /api/v1/host-capabilities`
/// reports `memoryGb` as the worker's `memoryTotalMb / 1024`, and the manifests' `minMemoryGb` are
/// written in the same base, so `limits.maxMemoryGb` is a GiB budget and an observed peak in bytes
/// has to be divided by 1024^3 to be compared with it.
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Tag every harness-imported APPROVED reference carries beside its role tag.
const REFERENCE_TAG: &str = "film-harness-reference";

/// Tag an imported reference the pack has NOT approved carries instead, so a query for the
/// conditioning-eligible references cannot pick it up.
const UNAPPROVED_REFERENCE_TAG: &str = "film-harness-reference-unapproved";

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
    /// Compiled requests to dispatch. `None` looks for `compiled.json` beside the plan and, failing
    /// that, compiles the plan's own prompts in memory — which is exactly what a hand-authored plan
    /// has always done.
    pub compiled_path: Option<PathBuf>,
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

/// Cooperative cancellation for a run. The `film-harness` binary flips it from its `ctrl_c`
/// handler; the harness then cancels the in-flight job through the API, stops dispatching, and
/// still writes the run record — rather than leaving a 45-minute render on the GPU with no record
/// of it, which is the opposite of what the plan's limits are for.
#[derive(Debug, Clone, Default)]
pub struct RunControl {
    canceled: Arc<AtomicBool>,
}

impl RunControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the in-flight run to cancel its job and stop dispatching. Idempotent.
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::SeqCst)
    }
}

/// The multipart filename for `path`'s basename: every character outside `[A-Za-z0-9._-]` replaced,
/// so a name can never inject `Content-Disposition` headers. `validate_reference_pack` already
/// refuses such a basename; this is the second half of the same guard, at the encoder, so no future
/// caller can reach the header with an unchecked name.
pub fn sanitize_multipart_filename(filename: &str) -> String {
    let sanitized: String = filename
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.trim_matches('.').is_empty() {
        "reference.png".to_owned()
    } else {
        sanitized
    }
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
            sanitize_multipart_filename(filename)
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

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One request that must succeed, against a bare transport. The planner (sc-22713) drives the same
/// API through this, so both halves of the harness treat a non-2xx answer identically. Planning is
/// not cancellable mid-decode the way a render is, so it carries a fresh, never-flipped
/// [`RunControl`].
pub(crate) async fn expect_ok_on(
    transport: &dyn ApiTransport,
    method: &'static str,
    path: &str,
    body: Option<Value>,
) -> Result<Value, HarnessError> {
    let control = RunControl::new();
    Client {
        transport,
        control: &control,
    }
    .expect_ok(method, path, body)
    .await
}

/// The catalog entry for `model_id`, against a bare transport.
pub(crate) async fn model_entry_for(
    transport: &dyn ApiTransport,
    model_id: &str,
) -> Result<Option<JsonObject<String, Value>>, HarnessError> {
    let control = RunControl::new();
    resolve_model_entry(
        &Client {
            transport,
            control: &control,
        },
        model_id,
    )
    .await
}

/// The API host's facts, against a bare transport: the platform whose lane and reachability gate a
/// plan is judged on, and the memory the host reports. The planner needs them for the same reasons
/// the run does, and reads them through this rather than keeping its own copy.
pub(crate) async fn host_facts_for(
    transport: &dyn ApiTransport,
) -> Result<HostFacts, HarnessError> {
    let control = RunControl::new();
    discover_host(&Client {
        transport,
        control: &control,
    })
    .await
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
    ///
    /// The persisted side is the positive signal: once `assets` / `assetIds` are present the
    /// handoff is done, whatever else the result still carries. Waiting on the ABSENCE of
    /// `assetWrites` alone has no escape hatch — `persist_reported_assets` returns early without
    /// removing the key when the job row carries no `project_id`, and the recovery loop can retry a
    /// persistently failing handoff indefinitely — which would poll a finished job until the shot
    /// budget expired and then report it as a timeout with the take lost.
    fn is_settled(&self) -> bool {
        self.status != "completed"
            || self
                .result
                .get("assets")
                .and_then(Value::as_array)
                .is_some()
            || self
                .result
                .get("assetIds")
                .and_then(Value::as_array)
                .is_some()
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
    /// The operator asked the run to stop (SIGINT on the binary).
    Operator,
    /// The job reached a terminal status but the API never finished persisting its reported
    /// assets, so the result stayed non-final for [`ASSET_SETTLE_GRACE`].
    AssetsUnsettled,
}

/// Everything one poll loop is bounded by.
#[derive(Debug, Clone, Copy)]
struct PollBounds {
    shot_deadline: Instant,
    run_deadline: Instant,
    poll_interval: Duration,
    /// How long a canceled job gets to reach a terminal state: [`CANCEL_GRACE`] capped at the
    /// plan's per-shot budget.
    cancel_grace: Duration,
    /// How long a terminal job gets to finish persisting its reported assets:
    /// [`ASSET_SETTLE_GRACE`] capped at the plan's per-shot budget.
    settle_grace: Duration,
}

struct Client<'a> {
    transport: &'a dyn ApiTransport,
    control: &'a RunControl,
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

    /// Poll `job_id` until it is terminal or a deadline passes. On a deadline (or an operator
    /// cancel) the job is canceled through the API and given `bounds.grace` to settle; the returned
    /// view is the last one observed either way.
    ///
    /// A job that is already terminal but whose result the API has not finished persisting is
    /// waited out separately — never cancelled, since cancelling a finished job is meaningless —
    /// and bounded by the same grace, after which it comes back as
    /// [`PollStop::AssetsUnsettled`] rather than as a budget timeout.
    async fn wait_for_job(
        &self,
        job_id: &str,
        bounds: PollBounds,
    ) -> Result<(JobView, PollStop), HarnessError> {
        let mut unsettled_deadline: Option<Instant> = None;
        loop {
            let view = self.get_job(job_id).await?;
            if view.is_terminal() {
                if view.is_settled() {
                    return Ok((view, PollStop::Terminal));
                }
                let deadline =
                    *unsettled_deadline.get_or_insert(Instant::now() + bounds.settle_grace);
                if Instant::now() >= deadline {
                    return Ok((view, PollStop::AssetsUnsettled));
                }
                tokio::time::sleep(bounds.poll_interval.min(bounds.settle_grace)).await;
                continue;
            }
            let now = Instant::now();
            let stop = if self.control.is_canceled() {
                Some(PollStop::Operator)
            } else if now >= bounds.run_deadline {
                Some(PollStop::RunBudget)
            } else if now >= bounds.shot_deadline {
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
                let grace_deadline = Instant::now() + bounds.cancel_grace;
                let mut last = view;
                while Instant::now() < grace_deadline {
                    tokio::time::sleep(bounds.poll_interval.min(bounds.cancel_grace)).await;
                    last = self.get_job(job_id).await?;
                    if last.is_terminal() && last.is_settled() {
                        break;
                    }
                }
                return Ok((last, stop));
            }
            tokio::time::sleep(bounds.poll_interval).await;
        }
    }

    /// The job's `generation_metrics` block (`GET /api/v1/jobs/:id/metrics`), retried for
    /// [`METRICS_GRACE`] because the worker POSTs it AFTER the terminal progress update — the same
    /// two-phase shape the asset handoff has. `None` once the window closes with nothing recorded
    /// (a worker whose probe measured nothing posts no block at all), and never an error: telemetry
    /// must not fail a run that rendered.
    async fn job_metrics(&self, job_id: &str, poll_interval: Duration) -> Option<Value> {
        let path = format!("/api/v1/jobs/{job_id}/metrics");
        let deadline = Instant::now() + METRICS_GRACE;
        let cadence = poll_interval.min(METRICS_RETRY_INTERVAL);
        loop {
            if let Ok(response) = self.json("GET", &path, None).await {
                if (200..300).contains(&response.status) && !response.body.is_null() {
                    return Some(response.body);
                }
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(cadence).await;
        }
    }
}

/// One attempt's observed memory peak, and which signal it came from.
#[derive(Debug, Clone, Default)]
struct MemoryObservation {
    /// The peak in GiB — what `limits.maxMemoryGb` is compared against.
    gb: Option<f64>,
    /// The peak as a percentage of host memory, when the source expressed one.
    pct: Option<f64>,
    source: Option<String>,
}

/// Read an attempt's peak memory, preferring the real production signal.
///
/// `GenerationMetrics.peakMemoryBytes` (MLX `get_peak_memory` on macOS, the nvidia-smi high-water
/// mark on candle) is the only measured peak a shipped worker reports: EVERY `ProgressRequest` in
/// `sceneworks-worker` sets `peakGpuMemoryPct: None`, so the job snapshot's field is always null on
/// a real render and is kept here only as a last resort for a future worker that fills it.
fn memory_observation(
    metrics: Option<&Value>,
    view: &JobView,
    host_memory_gb: Option<f64>,
) -> MemoryObservation {
    // A zero peak is "nothing was measured", not "this run used no memory" — the export job's own
    // metrics row on a real run is exactly that — so it falls through to the next source.
    let number = |value: Option<&Value>| {
        value
            .and_then(Value::as_f64)
            .filter(|number| number.is_finite() && *number > 0.0)
    };
    let metrics_pct = number(metrics.and_then(|metrics| metrics.get("peakMemoryPct")));
    if let Some(bytes) = number(metrics.and_then(|metrics| metrics.get("peakMemoryBytes"))) {
        return MemoryObservation {
            gb: Some(bytes / BYTES_PER_GB),
            pct: metrics_pct,
            source: Some("metrics.peakMemoryBytes".to_owned()),
        };
    }
    if let (Some(pct), Some(host)) = (metrics_pct, host_memory_gb) {
        return MemoryObservation {
            gb: Some(host * pct / 100.0),
            pct: Some(pct),
            source: Some("metrics.peakMemoryPct".to_owned()),
        };
    }
    match (view.peak_gpu_memory_pct, host_memory_gb) {
        (Some(pct), Some(host)) => MemoryObservation {
            gb: Some(host * pct / 100.0),
            pct: Some(pct),
            source: Some("job.peakGpuMemoryPct".to_owned()),
        },
        (Some(pct), None) => MemoryObservation {
            gb: None,
            pct: Some(pct),
            source: Some("job.peakGpuMemoryPct".to_owned()),
        },
        _ => MemoryObservation::default(),
    }
}

/// What the run learned about the host and the worker that will render. Shared with the planner
/// (sc-22713), which judges a plan against the same host rather than against this process.
#[derive(Debug, Clone, Default)]
pub(crate) struct HostFacts {
    /// The API HOST's platform (`std::env::consts::OS` spelling), from
    /// `GET /api/v1/host-capabilities`. `--api` may point at another machine, so this — not
    /// `cfg!(target_os)` — decides the lane whose `minMemoryGb` the plan is checked against, the
    /// platform the route's own reachability gate is judged on, and the hardware the record claims.
    platform: Option<String>,
    host_memory_gb: Option<f64>,
    video_worker_id: Option<String>,
    video_gpu_name: Option<String>,
    export_worker: bool,
}

impl HostFacts {
    /// The lane the render host reads its memory minimum from, falling back to this process's own
    /// platform only when the API reports none.
    pub(crate) fn lane(&self) -> ModelLane {
        match self.platform.as_deref() {
            Some(platform) => ModelLane::for_platform(platform),
            None => ModelLane::for_current_platform(),
        }
    }

    pub(crate) fn platform_or_local(&self) -> &str {
        self.platform.as_deref().unwrap_or(std::env::consts::OS)
    }
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
    facts.platform = host
        .get("platform")
        .and_then(Value::as_str)
        .filter(|platform| !platform.trim().is_empty())
        .map(str::to_owned);
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

/// The compiled requests this run dispatches: the document beside the plan when there is one, else
/// the plan compiled in memory with its authored prompts (the hand-authored path). Either way the
/// job bodies come from [`CompiledRequest::to_job_body`], so what a reviewer reads in
/// `compiled.json` is what the API receives.
fn compiled_for_run(
    plan: &ProductionPlan,
    pack: &ReferencePack,
    entry: &JsonObject<String, Value>,
    lane: ModelLane,
    plan_sha256: &str,
    supplied: Option<CompiledPlan>,
) -> Result<CompiledPlan, HarnessError> {
    match supplied {
        Some(compiled) => {
            // Two questions, in order: were these requests compiled from THIS plan, and do they
            // still say what compiling it would say? The second is what keeps a hand-edited
            // `compiled.json` — the document every dispatched field but the prompt is read from —
            // from reaching the route unjudged, since `validate_all` only ever reads the plan.
            let mut findings = compiled.staleness_findings(plan, plan_sha256);
            if findings.is_empty() {
                findings = compiled.conformance_findings(plan, entry, lane);
            }
            if findings.is_empty() {
                Ok(compiled)
            } else {
                Err(HarnessError::Validation(findings))
            }
        }
        None => compile_plan(
            plan,
            pack,
            &CompileInputs {
                model_entry: entry,
                lane: lane.manifest_key(),
                plan_sha256,
                compiled_at: &utc_now(),
                refined_prompts: &BTreeMap::new(),
            },
        )
        .map_err(HarnessError::Validation),
    }
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

/// The timeline aspect ratio a take of this geometry is CREATED at: the admitted ratio closest to
/// the take's own, derived from the real geometry rather than from its orientation alone.
///
/// The timeline route admits only what `sceneworks_core::project_store::TIMELINE_ASPECT_RATIOS`
/// lists (`16:9` / `9:16` / `1:1`), and the fixture's 576x320 takes are 9:5 — so SOMETHING has to
/// be coerced, and the export pads the take into the frame (a 576x320 take in a 640-tall 16:9
/// export lands as 1138x640 with bars). Distance is measured on `ln(w/h)` — the scale-free
/// comparison, on which 2:1 is as far from 16:9 as 16:9 is from 1.0 — and a tie goes to the
/// candidate whose ORIENTATION matches the take's, because 4:3 sits exactly halfway between 1:1
/// and 16:9 on that scale and a landscape take belongs in a landscape frame.
/// [`reduced_aspect_ratio`] records what the takes actually are, so the record never states the
/// coerced ratio as a fact about the footage.
fn aspect_ratio_for(width: u32, height: u32) -> &'static str {
    let take = (f64::from(width.max(1)) / f64::from(height.max(1))).ln();
    let orientation = width.cmp(&height);
    let mut best: Option<(&'static str, f64, bool)> = None;
    for (name, candidate_width, candidate_height) in
        sceneworks_core::project_store::TIMELINE_ASPECT_RATIOS
    {
        let distance =
            (take - (f64::from(*candidate_width) / f64::from(*candidate_height)).ln()).abs();
        let matches_orientation = candidate_width.cmp(candidate_height) == orientation;
        let better = match best {
            None => true,
            Some((_, best_distance, best_matches)) => {
                if (distance - best_distance).abs() <= ASPECT_TIE_EPSILON {
                    matches_orientation && !best_matches
                } else {
                    distance < best_distance
                }
            }
        };
        if better {
            best = Some((name, distance, matches_orientation));
        }
    }
    best.map_or("16:9", |(name, _, _)| name)
}

/// The takes' own ratio in lowest terms, e.g. `9:5` for 576x320.
fn reduced_aspect_ratio(width: u32, height: u32) -> String {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a.max(1)
        } else {
            gcd(b, a % b)
        }
    }
    let divisor = gcd(width, height);
    format!("{}:{}", width / divisor, height / divisor)
}

fn export_resolution_for(height: u32) -> u32 {
    crate::TIMELINE_EXPORT_RESOLUTIONS
        .iter()
        .copied()
        .find(|candidate| *candidate >= height)
        .unwrap_or_else(|| {
            crate::TIMELINE_EXPORT_RESOLUTIONS
                .iter()
                .copied()
                .max()
                .unwrap_or(1280)
        })
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
    // The compiled requests travel with the record too: without them the run record says which
    // prompts were dispatched only by reference.
    if let Some(compiled) = record.compiled.as_ref() {
        if let Ok(text) = std::fs::read(&compiled.path) {
            std::fs::write(out_dir.join("compiled.json"), text)?;
        }
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
    // Compiled requests are checked against the plan they claim: a plan edited after the compile
    // would otherwise dispatch the prompts it no longer holds.
    let compiled = read_compiled_for(options)?;
    if let Some((compiled, path)) = compiled.as_ref() {
        let plan_bytes = std::fs::read(&options.plan_path)?;
        let mut stale = compiled.staleness_findings(&plan, &sha256_hex(&plan_bytes));
        if !stale.is_empty() {
            stale.insert(0, compiled_document_header(path));
            findings.append(&mut stale);
        }
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    if let Some(transport) = transport {
        let control = RunControl::default();
        let client = Client {
            transport,
            control: &control,
        };
        // The host comes FIRST: its platform decides which lane's `minMemoryGb` the plan is
        // checked against and which platform the route's reachability gate is judged on, so the
        // model checks cannot run before it is known.
        let facts = discover_host(&client).await?;
        let entry = resolve_model_entry(&client, &plan.model.id).await?;
        let mut findings = model_findings(&plan, entry.as_ref(), options.require_installed, &facts);
        if findings.is_empty() {
            findings.extend(host_findings(&plan, &facts, options.export));
        }
        // The document that is DISPATCHED, judged against the installed capabilities — not just
        // the plan it was compiled from. `--compiled FILE` takes a document from any path and
        // `execute_run` reads every field but the prompt straight out of it, so this is the only
        // place a hand-edited request meets the model's declared menus (sc-22713 review).
        if findings.is_empty() {
            if let (Some((compiled, path)), Some(entry)) = (compiled.as_ref(), entry.as_ref()) {
                let mut conformance = compiled.conformance_findings(&plan, entry, facts.lane());
                if !conformance.is_empty() {
                    findings.push(compiled_document_header(path));
                    findings.append(&mut conformance);
                }
            }
        }
        if !findings.is_empty() {
            return Err(HarnessError::Validation(findings));
        }
    }
    Ok((plan, pack))
}

/// The finding that says the findings after it are about the compiled document, not the plan.
fn compiled_document_header(path: &Path) -> PlanDiagnostic {
    PlanDiagnostic::plan(
        "compiled",
        format!(
            "the compiled requests at {} do not match this plan; every finding below is about \
             that document",
            path.display()
        ),
    )
}

/// The compiled requests this run should use, with the path they came from: the explicit
/// `--compiled` document, else a `compiled.json` sitting beside the plan, else nothing.
fn read_compiled_for(
    options: &RunOptions,
) -> Result<Option<(CompiledPlan, PathBuf)>, HarnessError> {
    let path = match &options.compiled_path {
        Some(path) => path.clone(),
        None => {
            let sibling = options
                .plan_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("compiled.json");
            if !sibling.is_file() {
                return Ok(None);
            }
            sibling
        }
    };
    let compiled = crate::film_planner::read_compiled_file(&path)
        .map_err(|finding| HarnessError::Validation(vec![finding]))?;
    Ok(Some((compiled, path)))
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

/// The route's own platform-reachability gate, run before dispatch instead of being discovered as
/// a 400 at enqueue: `POST /api/v1/video/jobs` calls exactly this function
/// ([`crate::generation::ensure_video_model_available_on_platform`]) on the resolved manifest
/// entry, so a plan this validator passes cannot be refused for the reason it checks.
fn platform_reachability_finding(
    model_id: &str,
    entry: &JsonObject<String, Value>,
    platform: &str,
) -> Option<PlanDiagnostic> {
    let value = Value::Object(entry.clone());
    crate::generation::ensure_video_model_available_on_platform(model_id, &value, platform)
        .err()
        .map(|error| PlanDiagnostic::plan("model.id", error.detail))
}

/// The route's reference-payload gate, run before dispatch on the payload shape each shot will
/// produce. Placeholder ids stand in for the asset ids the import has not created yet: they are
/// non-blank and untrimmed-free by construction, so the only questions this can answer are the ones
/// that depend on the PLAN — how many references a shot carries, and the model/mode spelling.
fn reference_payload_findings(
    plan: &ProductionPlan,
    entry: &JsonObject<String, Value>,
) -> Vec<PlanDiagnostic> {
    let value = Value::Object(entry.clone());
    let mut findings = Vec::new();
    for shot in &plan.shots {
        let mut payload = JsonObject::new();
        payload.insert("model".to_owned(), json!(plan.model.id));
        payload.insert("mode".to_owned(), json!(shot.conditioning.mode));
        payload.insert(
            "referenceAssetIds".to_owned(),
            Value::Array(
                (0..shot.conditioning.reference_roles.len())
                    .map(|index| json!(format!("placeholder_reference_{index}")))
                    .collect(),
            ),
        );
        if let Err(error) = crate::validate_video_reference_asset_ids_payload(&payload, &value) {
            findings.push(PlanDiagnostic::shot(
                &shot.id,
                "conditioning.referenceRoles",
                error.detail,
            ));
        }
    }
    findings
}

/// Findings about the catalog ENTRY: present, a video model, installed, and reachable on the render
/// host's platform. Shared with the planner (sc-22713), which must refuse an absent, uninstalled or
/// unreachable model before it spends a decode on a plan that could never be dispatched — rather
/// than keeping a second copy of these rules.
pub(crate) fn model_entry_findings(
    model_id: &str,
    entry: Option<&JsonObject<String, Value>>,
    tier: Option<&str>,
    require_installed: bool,
    facts: &HostFacts,
) -> Vec<PlanDiagnostic> {
    let Some(entry) = entry else {
        return vec![PlanDiagnostic::plan(
            "model.id",
            format!("{model_id:?} is not in this API's model catalog"),
        )];
    };
    let mut findings = Vec::new();
    if entry.get("type").and_then(Value::as_str) != Some("video") {
        findings.push(PlanDiagnostic::plan(
            "model.id",
            format!("{model_id:?} is not a video model"),
        ));
    }
    if require_installed && !model_tier_installed(entry, tier) {
        findings.push(PlanDiagnostic::plan(
            "model.tier",
            format!(
                "{model_id}{} is not installed on this host (catalog installState is not \
                 \"installed\"); download it in the Model Manager first",
                tier.map(|tier| format!(" tier {tier}")).unwrap_or_default()
            ),
        ));
    }
    findings.extend(platform_reachability_finding(
        model_id,
        entry,
        facts.platform_or_local(),
    ));
    findings
}

fn model_findings(
    plan: &ProductionPlan,
    entry: Option<&JsonObject<String, Value>>,
    require_installed: bool,
    facts: &HostFacts,
) -> Vec<PlanDiagnostic> {
    let mut findings = model_entry_findings(
        &plan.model.id,
        entry,
        plan.model.tier.as_deref(),
        require_installed,
        facts,
    );
    let Some(entry) = entry else {
        return findings;
    };
    findings.extend(film_plan::validate_plan_against_model(
        plan,
        entry,
        facts.lane(),
    ));
    findings.extend(reference_payload_findings(plan, entry));
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
    run_with_control(transport, options, &RunControl::default()).await
}

/// [`run`] with an operator cancellation handle. The `film-harness` binary passes one so SIGINT
/// cancels the in-flight job through the API and still leaves a record, instead of orphaning a
/// render on the GPU.
///
/// Every path past document validation writes `run.json` — including a transport/API failure
/// mid-run, which is exactly when the record matters most: by then the run may have created a
/// project, imported assets and dispatched jobs, and nothing else remembers that it did.
pub async fn run_with_control(
    transport: &dyn ApiTransport,
    options: &RunOptions,
    control: &RunControl,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let run_id = format!("run_{}", uuid::Uuid::new_v4().simple());
    let plan_bytes = std::fs::read(&options.plan_path)?;
    let pack_bytes = std::fs::read(&options.reference_pack_path)?;
    let client = Client { transport, control };

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

    let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
    let outcome = execute_run(
        &client,
        transport,
        options,
        &plan,
        &pack,
        &run_id,
        started,
        &mut record,
    )
    .await;
    record.finished_at = Some(utc_now());
    record.elapsed_seconds = seconds_since(started);
    match outcome {
        Ok(()) => {
            persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            )?;
            Ok(record)
        }
        Err(HarnessError::Validation(findings)) => {
            record.outcome = RunOutcome::Rejected;
            record.diagnostics = findings.clone();
            persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            )?;
            Err(HarnessError::Validation(findings))
        }
        Err(error) => {
            record.outcome = RunOutcome::Failed;
            record.diagnostics.push(PlanDiagnostic::plan(
                "run",
                format!("the run stopped on an error: {error}"),
            ));
            // Best effort: an io failure while writing the record must not replace the failure
            // that caused it with a less informative one.
            let _ = persist_record(
                &record,
                &options.out_dir,
                &options.plan_path,
                &options.reference_pack_path,
            );
            Err(error)
        }
    }
}

/// Steps 3-6 against a record the caller owns, so every early return still leaves a persistable
/// record behind. `Err(Validation)` means the catalog/host refused the plan before the first write;
/// any other `Err` is a transport/API failure partway through.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn execute_run(
    client: &Client<'_>,
    transport: &dyn ApiTransport,
    options: &RunOptions,
    plan: &ProductionPlan,
    pack: &ReferencePack,
    run_id: &str,
    started: Instant,
    record: &mut RunRecord,
) -> Result<(), HarnessError> {
    let facts = discover_host(client).await?;
    let entry = resolve_model_entry(client, &plan.model.id).await?;
    let mut findings = model_findings(plan, entry.as_ref(), options.require_installed, &facts);
    if findings.is_empty() {
        findings.extend(host_findings(plan, &facts, options.export));
    }
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let entry = entry.expect("model findings are empty only with an entry");
    let fps = film_plan::plan_fps(plan, &entry).expect("validated against the model");
    let lane = facts.lane();
    // The requests this run dispatches: the compiled document beside the plan when there is one,
    // else the plan's own prompts compiled in memory (sc-22713). `record.plan.sha256` is the hash
    // of the plan file as read, which is what a compiled document pins.
    let supplied_compiled = read_compiled_for(options)?;
    record.compiled = match supplied_compiled.as_ref() {
        // The hash is what makes the run reproducible, so an unreadable file is an error rather
        // than an empty string on the record. It was read successfully microseconds ago in
        // `read_compiled_for`, so `?` here can only surface a real io failure.
        Some((compiled, path)) => Some(SourceDocument {
            id: compiled.plan_id.clone(),
            version: compiled.plan_version,
            path: path.display().to_string(),
            sha256: sha256_hex(&std::fs::read(path)?),
        }),
        None => None,
    };
    let compiled = compiled_for_run(
        plan,
        pack,
        &entry,
        lane,
        &record.plan.sha256,
        supplied_compiled.map(|(compiled, _)| compiled),
    )?;

    record.model = Some(ModelRecord {
        id: plan.model.id.clone(),
        tier_requested: plan.model.tier.clone(),
        fps,
        lane: lane.manifest_key().to_owned(),
        backend_observed: None,
        weights: primary_weights(&entry, plan.model.tier.as_deref()),
        hardware: HardwareRecord {
            platform: facts.platform_or_local().to_owned(),
            // The host-capabilities route reports no arch, so this process's arch is the truth
            // only when the API is on this platform; otherwise claiming one would be a fabrication.
            arch: if facts.platform_or_local() == std::env::consts::OS {
                std::env::consts::ARCH.to_owned()
            } else {
                "unknown".to_owned()
            },
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
        // An unapproved reference is imported (so the record can point at it and a human can
        // review it) but tagged distinctly, so a query for the conditioning-eligible references
        // cannot pick it up — AC1 is about APPROVED references staying addressable.
        let kind_tag = if reference.approved {
            REFERENCE_TAG
        } else {
            UNAPPROVED_REFERENCE_TAG
        };
        client
            .expect_ok(
                "PATCH",
                &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                Some(json!({
                    "tags": [kind_tag, format!("role:{}", reference.role), format!("pack:{}", pack.id)]
                })),
            )
            .await?;
        if reference.approved {
            // Only approved roles resolve into a shot's conditioning slots. Validation already
            // refuses a plan that conditions on an unapproved role, so this is the second half of
            // the same guarantee: even a validator gap cannot put an unapproved plate on the wire.
            role_assets.insert(reference.role.clone(), asset_id.clone());
        }
        record.references.push(ReferenceAssetRecord {
            role: reference.role.clone(),
            kind: reference.kind.clone(),
            file: reference.file.clone(),
            sha256,
            asset_id,
            approved: reference.approved,
        });
    }

    // Sound imports the same way pictures do (sc-22712). Import normalises every clip to PCM-16
    // WAV and measures its duration off the stored file, so `durationSeconds` here is the length
    // the export will actually read rather than anything the plan claimed.
    //
    // Only what this run will PLACE is imported — the two beds plus the dialogue of the SELECTED
    // shots. Unlike a reference image, importing an audio clip costs an ffmpeg transcode, and a
    // pack legitimately carries sound for shots a `--shots` run left out. `record.sound` is
    // therefore the list of clips that were actually available to the mix.
    let placed_sound_roles: BTreeSet<String> = plan
        .sound
        .ambience
        .iter()
        .chain(plan.sound.music.iter())
        .map(|bed| bed.role.clone())
        .chain(
            plan.shots
                .iter()
                .filter(|shot| record.selected_shot_ids.contains(&shot.id))
                .filter_map(|shot| shot.dialogue_clip.as_ref().map(|clip| clip.role.clone())),
        )
        .collect();
    let mut sound_assets: BTreeMap<String, SoundAsset> = BTreeMap::new();
    for entry in pack
        .sound
        .iter()
        .filter(|entry| placed_sound_roles.contains(&entry.role))
    {
        let path = pack_dir.join(&entry.file);
        let bytes = std::fs::read(&path)?;
        let sha256 = sha256_hex(&bytes);
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("sound.wav")
            .to_owned();
        let provenance = json!({
            "filmHarness": {
                "kind": "sound",
                "role": entry.role,
                "soundKind": entry.kind,
                "referencePackId": pack.id,
                "referencePackVersion": pack.version,
                "planId": plan.id,
                "planVersion": plan.version,
                "runId": run_id,
                "sourceFile": entry.file,
                "sha256": sha256,
            }
        });
        let (boundary, body) =
            encode_asset_upload(&filename, audio_content_type(&path), &bytes, &provenance);
        let route = format!("/api/v1/projects/{project_id}/assets");
        let response = transport
            .call(ApiRequest {
                method: "POST",
                path: route.clone(),
                body: RequestBody::Multipart {
                    boundary,
                    bytes: body,
                },
            })
            .await?;
        if !(200..300).contains(&response.status) {
            return Err(HarnessError::Api {
                method: "POST",
                path: route,
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
                    "sound import response has no id: {}",
                    response.body
                ))
            })?
            .to_owned();
        let duration_seconds = response
            .body
            .get("file")
            .and_then(|file| file.get("duration"))
            .and_then(Value::as_f64)
            .filter(|seconds| *seconds > 0.0);
        client
            .expect_ok(
                "PATCH",
                &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                Some(json!({
                    "tags": [SOUND_TAG, format!("role:{}", entry.role), format!("pack:{}", pack.id)]
                })),
            )
            .await?;
        sound_assets.insert(
            entry.role.clone(),
            SoundAsset {
                asset_id: asset_id.clone(),
                duration_seconds,
            },
        );
        record.sound.push(ReferenceAssetRecord {
            role: entry.role.clone(),
            kind: entry.kind.clone(),
            file: entry.file.clone(),
            sha256,
            asset_id,
            // A sound entry has no approval flag of its own: approval gates CONDITIONING, and
            // sound is never conditioning. Everything in the pack's `sound` array is placeable.
            approved: true,
        });
    }

    // Step 5: shots, one at a time, under the declared limits.
    let run_deadline = started + Duration::from_secs(plan.limits.max_run_seconds);
    let shot_budget = Duration::from_secs(plan.limits.max_shot_seconds);
    // A cancel must not be given more room to settle than the plan gave the whole attempt.
    let cancel_grace = CANCEL_GRACE.min(shot_budget);
    let settle_grace = ASSET_SETTLE_GRACE.min(shot_budget);
    let selected: Vec<&film_plan::Shot> = plan
        .shots
        .iter()
        .filter(|shot| record.selected_shot_ids.contains(&shot.id))
        .collect();
    let mut stop: Option<RunOutcome> = None;
    let mut rendered: Vec<(String, TakeRecord, (u32, u32))> = Vec::new();
    for shot in &plan.shots {
        // The compiled request is the single source of the geometry, the timing and the prompt —
        // the same document `compiled.json` shows a reviewer (sc-22713).
        let request = compiled
            .request(&shot.id)
            .expect("every shot compiled a request");
        let (width, height) = (request.width, request.height);
        let resolved = request
            .resolve_conditioning(&role_assets)
            .map_err(HarnessError::Validation)?;
        let assets = ConditioningAssets {
            first_frame_asset_id: resolved.first_frame_asset_id.clone(),
            last_frame_asset_id: resolved.last_frame_asset_id.clone(),
            reference_asset_ids: resolved.reference_asset_ids.clone(),
        };
        let mut shot_record = ShotRunRecord {
            shot_id: shot.id.clone(),
            outcome: ShotOutcome::NotSelected,
            intended: IntendedState {
                mode: request.mode.clone(),
                start_state: shot.start_state.clone(),
                end_state: shot.end_state.clone(),
                target_duration_seconds: request.duration_seconds,
                width,
                height,
                fps: request.fps,
                dialogue: shot.dialogue.clone(),
                sound: shot.sound.clone(),
                generated_audio: resolved_generated_audio(plan, shot),
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
            if client.control.is_canceled() {
                stop = Some(RunOutcome::Failed);
                record.diagnostics.push(PlanDiagnostic::plan(
                    "run",
                    "canceled by operator before this attempt was dispatched",
                ));
                break;
            }
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
                peak_memory_gb: None,
                peak_memory_source: None,
                error: None,
                take: None,
            };
            let body = request.to_job_body_with(
                &DispatchContext {
                    project_id: &project_id,
                    run_id,
                    plan_id: &plan.id,
                    plan_version: plan.version,
                    attempt,
                    tier: plan.model.tier.as_deref(),
                    role_assets: &role_assets,
                },
                &ResolvedConditioning {
                    first_frame_asset_id: resolved.first_frame_asset_id.clone(),
                    last_frame_asset_id: resolved.last_frame_asset_id.clone(),
                    reference_asset_ids: resolved.reference_asset_ids.clone(),
                },
            );
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
            let (view, poll_stop) = client
                .wait_for_job(
                    &job_id,
                    PollBounds {
                        shot_deadline: attempt_started + shot_budget,
                        run_deadline,
                        poll_interval: options.poll_interval,
                        cancel_grace,
                        settle_grace,
                    },
                )
                .await?;
            attempt_record.finished_at = Some(utc_now());
            attempt_record.elapsed_seconds = seconds_since(attempt_started);
            attempt_record.status = match poll_stop {
                PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
                PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
                PollStop::Operator => "canceled_by_operator".to_owned(),
            };
            // The observed peak comes off the job's metrics block, which the worker POSTs after
            // its terminal progress — so read it once the attempt is terminal, whatever ended it.
            let metrics = client.job_metrics(&job_id, options.poll_interval).await;
            let memory = memory_observation(metrics.as_ref(), &view, facts.host_memory_gb);
            attempt_record.peak_gpu_memory_pct = memory.pct;
            attempt_record.peak_memory_gb = memory.gb;
            attempt_record.peak_memory_source = memory.source.clone();
            let memory_exceeded = memory
                .gb
                .is_some_and(|observed| observed > plan.limits.max_memory_gb);
            // Whether the cancel the harness posted was actually honoured. A job still running
            // after the grace is a render in flight that nothing here can stop.
            let cancel_honoured = view.is_terminal();
            match poll_stop {
                PollStop::AssetsUnsettled => {
                    // Terminal, but the API never finished the asset handoff. Retrying would
                    // dispatch a fresh render against a server-side condition a retry cannot fix.
                    attempt_record.error = Some(format!(
                        "job {job_id} reached {} but its assets never settled within {:.0}s (the \
                         result still carries raw assetWrites): {}",
                        view.status,
                        settle_grace.as_secs_f64(),
                        view.result
                    ));
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::Failed;
                    break;
                }
                PollStop::Operator => {
                    attempt_record.error = Some(if cancel_honoured {
                        format!("canceled by operator (job {job_id} is {})", view.status)
                    } else {
                        format!(
                            "canceled by operator; job {job_id} was still {} {:.0}s after the \
                             cancel was posted",
                            view.status,
                            cancel_grace.as_secs_f64()
                        )
                    });
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::Failed;
                    stop = Some(RunOutcome::Failed);
                    record.diagnostics.push(PlanDiagnostic::plan(
                        "run",
                        format!("canceled by operator during shot {}", shot.id),
                    ));
                    break;
                }
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
                    attempt_record.error = Some(if cancel_honoured {
                        format!(
                            "attempt exceeded the per-shot budget of {}s (last status {})",
                            plan.limits.max_shot_seconds, view.status
                        )
                    } else {
                        format!(
                            "attempt exceeded the per-shot budget of {}s and job {job_id} was \
                             still {} {:.0}s after the cancel was posted; stopping dispatch \
                             rather than running a second render against the same memory budget",
                            plan.limits.max_shot_seconds,
                            view.status,
                            cancel_grace.as_secs_f64()
                        )
                    });
                    shot_record.attempts.push(attempt_record);
                    outcome = ShotOutcome::TimedOut;
                    if !cancel_honoured {
                        // The render is still on the GPU. Neither the next attempt nor the next
                        // shot may go out beside it: the plan declared ONE memory budget, and two
                        // MiniMax-H3 renders in flight is exactly what it is there to prevent.
                        stop = Some(RunOutcome::Failed);
                        break;
                    }
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
        // The main video track, by id or — if the project store's default track ids ever change —
        // by kind. Writing the items nowhere and saving an EMPTY timeline while the record still
        // listed every shot is the one outcome that must not happen: the record would claim a
        // sequence the project does not hold.
        let tracks = timeline
            .get("tracks")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let track_index = tracks
            .iter()
            .position(|track| track.get("id").and_then(Value::as_str) == Some("track_main"))
            .or_else(|| {
                tracks
                    .iter()
                    .position(|track| track.get("kind").and_then(Value::as_str) == Some("video"))
            })
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "timeline {timeline_id} has no track_main and no video track to hold the \
                     rendered takes: {timeline}"
                ))
            })?;
        let track_id = tracks[track_index]
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("track_main")
            .to_owned();
        let mut items = Vec::new();
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
            let job_id = record
                .shots
                .iter()
                .find(|entry| &entry.shot_id == shot_id)
                .and_then(|entry| entry.attempts.last())
                .and_then(|attempt| attempt.job_id.clone());
            items.push(json!({
                "id": item_id,
                "trackId": track_id,
                "assetId": take.asset_id,
                "type": "video",
                "displayName": format!("{} — {}", shot.id, shot.beat).chars().take(160).collect::<String>(),
                "sourceIn": 0.0,
                "sourceOut": length,
                // Placed by `relayout_timeline` below; the cut order is the plan's.
                "timelineStart": 0.0,
                "timelineEnd": length,
                "speed": 1.0,
                "fit": "fit",
                "volume": 1.0,
                // The resolved policy, written into the timeline itself so the export obeys the
                // saved document rather than re-deriving anything from the plan (sc-22712).
                "generatedAudio": resolved_generated_audio(plan, shot).as_timeline_str(),
                "versionHistory": [{
                    "assetId": take.asset_id,
                    "source": "original",
                    "jobId": job_id,
                    "note": format!("film-harness {run_id} shot {}", shot.id),
                }],
                HARNESS_KEY: harness_block(ROLE_PICTURE, run_id, Some(&shot.id), 0.0),
            }));
            tallest = tallest.max(*height);
        }

        // Sound: one dialogue clip per shot that has one, and the two beds placed ONCE across the
        // whole sequence. `relayout_timeline` gives them their positions.
        let mut dialogue_items = Vec::new();
        for (shot_id, _, _) in &rendered {
            let shot = plan
                .shots
                .iter()
                .find(|shot| &shot.id == shot_id)
                .expect("rendered shots come from the plan");
            let Some(clip) = &shot.dialogue_clip else {
                continue;
            };
            let Some(asset) = sound_assets.get(&clip.role) else {
                continue;
            };
            let length = clip
                .duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .or_else(|| {
                    asset
                        .duration_seconds
                        .map(|total| (total - clip.source_in_seconds).max(MIN_ITEM_SECONDS))
                })
                .unwrap_or(MIN_ITEM_SECONDS)
                .max(MIN_ITEM_SECONDS);
            dialogue_items.push(json!({
                "id": format!("item_line_{}_{}", shot.id.to_ascii_lowercase(), &run_id[4..12]),
                "trackId": DIALOGUE_TRACK_ID,
                "assetId": asset.asset_id,
                "type": "audio",
                "displayName": format!("{} — dialogue ({})", shot.id, clip.role).chars().take(160).collect::<String>(),
                "sourceIn": clip.source_in_seconds,
                "sourceOut": clip.source_in_seconds + length,
                "timelineStart": 0.0,
                "timelineEnd": length,
                "speed": 1.0,
                "fit": "fit",
                "volume": clip.gain.clamp(0.0, 2.0),
                "fadeInSeconds": clip.fade_in_seconds,
                "fadeOutSeconds": clip.fade_out_seconds,
                HARNESS_KEY: harness_block(
                    ROLE_DIALOGUE,
                    run_id,
                    Some(&shot.id),
                    clip.offset_seconds,
                ),
            }));
        }

        // The ids of the picture items this run assembled, kept before `items` is moved onto the
        // track, so the save can be checked against them below.
        let intended_item_ids: Vec<String> = items
            .iter()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect();
        let mut tracks = vec![
            picture_track(&track_id, items),
            audio_track(
                DIALOGUE_TRACK_ID,
                "Dialogue",
                ROLE_DIALOGUE,
                &plan.sound.dialogue,
                dialogue_items,
            ),
        ];
        for (track_id, name, role, bed) in [
            (
                AMBIENCE_TRACK_ID,
                "Ambience",
                ROLE_AMBIENCE,
                plan.sound.ambience.as_ref(),
            ),
            (
                MUSIC_TRACK_ID,
                "Music",
                ROLE_MUSIC,
                plan.sound.music.as_ref(),
            ),
        ] {
            let Some(bed) = bed else { continue };
            let Some(asset) = sound_assets.get(&bed.role) else {
                continue;
            };
            tracks.push(bed_track(track_id, name, role, bed, asset, run_id));
        }
        // Keep every track the API created that the harness does not own (the overlay lane and the
        // editor's default audio lane) so a harness timeline opens in the editor unchanged.
        if let Some(existing) = timeline.get("tracks").and_then(Value::as_array) {
            for track in existing {
                let id = track.get("id").and_then(Value::as_str).unwrap_or_default();
                if !tracks
                    .iter()
                    .any(|kept| kept.get("id").and_then(Value::as_str) == Some(id))
                {
                    tracks.push(track.clone());
                }
            }
        }
        timeline["tracks"] = Value::Array(tracks);

        let plan_order: Vec<String> = rendered
            .iter()
            .map(|(shot_id, _, _)| shot_id.clone())
            .collect();
        let planned_duration = relayout_timeline(&mut timeline, Some(&plan_order))?;
        let saved = client
            .expect_ok(
                "PUT",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
                Some(json!({ "timeline": timeline })),
            )
            .await?;
        // Check the SAVED document before describing it, rather than trusting the harness's own
        // intent: the record then cannot claim a sequence the project does not hold. The store may
        // legitimately add or reshape keys, so what is checked is that every picture item the run
        // assembled came back — not that the document is byte-identical to what was sent.
        persisted_picture_items(&saved, &track_id, &intended_item_ids).ok_or_else(|| {
            HarnessError::Transport(format!(
                "saved timeline {timeline_id} does not hold the {} picture items the run \
                 assembled: {saved}",
                intended_item_ids.len()
            ))
        })?;
        // The store recomputes `duration` across every track on save, so read it back rather than
        // republishing the harness's own arithmetic.
        let duration = saved
            .get("duration")
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .unwrap_or(planned_duration);
        record.timeline = Some(timeline_record(
            &timeline_id,
            &timeline_name,
            aspect_ratio,
            Some(reduced_aspect_ratio(*first_width, *first_height)),
            Some(*first_width),
            Some(*first_height),
            fps,
            duration,
            &saved,
            &track_id,
            plan.sound.generated_audio,
            Vec::new(),
        ));

        let (export_record, poll_stop) = export_timeline(
            client,
            &project_id,
            &timeline_id,
            export_resolution_for(tallest),
            fps,
            PollBounds {
                shot_deadline: Instant::now() + shot_budget,
                run_deadline,
                poll_interval: options.poll_interval,
                cancel_grace,
                settle_grace,
            },
            &plan.limits,
        )
        .await?;
        export_ok = export_record.status == "completed" && export_record.asset_id.is_some();
        record.export = Some(export_record);
        match poll_stop {
            PollStop::RunBudget => stop = Some(RunOutcome::StoppedRunBudget),
            PollStop::Operator => {
                stop = Some(RunOutcome::Failed);
                record.diagnostics.push(PlanDiagnostic::plan(
                    "run",
                    "canceled by operator during the export",
                ));
            }
            _ => {}
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
    Ok(())
}

/// The timeline items as the SAVE persisted them, in the order the run assembled them, or `None`
/// when the saved document does not hold exactly the items that were sent (which would mean the
/// record and the project disagree about what the sequence is).
/// Every picture item the run assembled, as the SAVED document holds it — or `None` if the save
/// does not hold exactly them.
///
/// Scoped to the picture track since sc-22712: the sequence now also carries dialogue and bed
/// tracks, so "the saved item count equals the intended count" is only a true statement about the
/// track the shots live on. The guarantee is unchanged — the run record cannot describe a sequence
/// the project does not hold.
fn persisted_picture_items<'a>(
    saved: &'a Value,
    track_id: &str,
    intended_item_ids: &[String],
) -> Option<Vec<&'a Value>> {
    let items = saved
        .get("tracks")
        .and_then(Value::as_array)?
        .iter()
        .find(|track| track.get("id").and_then(Value::as_str) == Some(track_id))
        .and_then(|track| track.get("items").and_then(Value::as_array))?;
    let mut persisted = Vec::with_capacity(intended_item_ids.len());
    for item_id in intended_item_ids {
        persisted.push(
            items
                .iter()
                .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id.as_str()))?,
        );
    }
    (persisted.len() == items.len()).then_some(persisted)
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
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits: plan.limits.clone(),
        selected_shot_ids,
        references: Vec::new(),
        sound: Vec::new(),
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
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits,
        selected_shot_ids: options.shot_ids.clone().unwrap_or_default(),
        references: Vec::new(),
        sound: Vec::new(),
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

/// Placeholder sound for the fixture pack (sc-22712): `(role, seconds, hz, amplitude)`.
///
/// The two beds are long enough to play under the WHOLE six-shot sequence (6 x 5.1667s ~= 31s)
/// without running out, because a bed that stops partway would make the one thing this fixture is
/// meant to demonstrate — continuous sound across intentional cuts — unobservable.
pub const FIXTURE_SOUNDS: &[(&str, f64, u32, i16)] = &[
    ("courier_line", 2.0, 400, 9000),
    ("recipient_line", 2.0, 500, 9000),
    ("workshop_room_tone", 32.0, 100, 2600),
    ("main_theme", 32.0, 250, 3600),
];

/// Sample rate of every fixture clip. Low on purpose: import transcodes each one to 48 kHz PCM-16
/// anyway, and a placeholder tone gains nothing from being stored at the higher rate.
pub const FIXTURE_SOUND_RATE: u32 = 8_000;

/// Write one deterministic placeholder clip: a mono PCM-16 triangle wave.
///
/// **Every sample is integer arithmetic, with no floating point anywhere.** A sine would be the
/// obvious choice and would be wrong: `f64::sin` may differ by an ULP between platforms, which is
/// enough to change a rounded `i16` and break the byte-for-byte check on the checked-in fixture.
/// A triangle is exactly reproducible on every host, and is far gentler to listen to than the
/// square wave that would be the other integer option.
pub fn fixture_sound_wav(seconds: f64, hz: u32, amplitude: i16) -> Vec<u8> {
    let rate = FIXTURE_SOUND_RATE;
    let frames = ((seconds.max(0.0) * f64::from(rate)) as u32).max(1);
    let period = (rate / hz.max(1)).max(2) as i32;
    let half = period / 2;
    let amplitude = i32::from(amplitude);
    let mut samples = Vec::with_capacity(frames as usize * 2);
    for index in 0..frames as i32 {
        let phase = index % period;
        let ramp = if phase < half { phase } else { period - phase };
        let value = (amplitude * (2 * ramp - half)) / half.max(1);
        samples.extend_from_slice(&(value as i16).to_le_bytes());
    }
    let data_len = samples.len() as u32;
    let mut wav = Vec::with_capacity(samples.len() + 44);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1_u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2_u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&samples);
    wav
}

/// Write every fixture clip into `out_dir` as `<role>.wav`.
pub fn write_fixture_sound(out_dir: &Path) -> Result<Vec<PathBuf>, HarnessError> {
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for (role, seconds, hz, amplitude) in FIXTURE_SOUNDS {
        let path = out_dir.join(format!("{role}.wav"));
        std::fs::write(&path, fixture_sound_wav(*seconds, *hz, *amplitude))?;
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
    fn a_filename_cannot_inject_multipart_headers() {
        let (_, body) = encode_asset_upload(
            "plate\r\nX-Injected: 1\r\n\r\nevil.png",
            "image/png",
            b"\x89PNG",
            &json!({}),
        );
        let text = String::from_utf8_lossy(&body);
        assert!(!text.contains("X-Injected: 1\r\n"), "{text}");
        assert!(
            text.contains("filename=\"plate__X-Injected__1____evil.png\""),
            "{text}"
        );
        assert_eq!(sanitize_multipart_filename("a\"b.png"), "a_b.png");
        assert_eq!(
            sanitize_multipart_filename("../../etc/passwd"),
            ".._.._etc_passwd"
        );
        assert_eq!(sanitize_multipart_filename("..."), "reference.png");
    }

    #[test]
    fn the_platform_gate_is_the_routes_own_and_follows_the_api_host() {
        let mac_only: JsonObject<String, Value> =
            json!({ "id": "some_model", "type": "video", "macOnly": true })
                .as_object()
                .cloned()
                .unwrap();
        let portable: JsonObject<String, Value> = json!({ "id": "some_model", "type": "video" })
            .as_object()
            .cloned()
            .unwrap();
        let finding = platform_reachability_finding("some_model", &mac_only, "linux")
            .expect("a mac-only model is unreachable on linux");
        assert_eq!(finding.field, "model.id");
        assert!(finding.message.contains("only on macOS"), "{finding}");
        assert!(platform_reachability_finding("some_model", &mac_only, "macos").is_none());
        assert!(platform_reachability_finding("some_model", &portable, "linux").is_none());
    }

    #[test]
    fn the_reference_payload_gate_is_the_routes_own() {
        // Ten reference roles: past the route's blanket ceiling of nine, which no per-model
        // `limits.maxReferenceAssets` can raise. Found here rather than as a 400 at enqueue.
        let roles: Vec<String> = (0..10).map(|index| format!("role_{index}")).collect();
        let plan: ProductionPlan = serde_json::from_value(json!({
            "schemaVersion": 1,
            "id": "p", "version": 1, "title": "t",
            "model": { "id": "some_model" },
            "limits": { "maxRunSeconds": 10, "maxShotSeconds": 5, "maxAttemptsPerShot": 1, "maxMemoryGb": 8 },
            "shots": [{
                "id": "SH010", "beat": "b", "framing": "f", "prompt": "p",
                "targetDurationSeconds": 5.0, "startState": "s", "endState": "e",
                "conditioning": { "mode": "reference_to_video", "referenceRoles": roles }
            }]
        }))
        .expect("plan parses");
        let entry: JsonObject<String, Value> = json!({ "id": "some_model", "type": "video" })
            .as_object()
            .cloned()
            .unwrap();
        let findings = reference_payload_findings(&plan, &entry);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].shot_id.as_deref(), Some("SH010"));
        assert!(findings[0].message.contains("at most 9"), "{findings:?}");

        // The shipped shape — no references at all — passes.
        let plan: ProductionPlan = serde_json::from_value(json!({
            "schemaVersion": 1,
            "id": "p", "version": 1, "title": "t",
            "model": { "id": "some_model" },
            "limits": { "maxRunSeconds": 10, "maxShotSeconds": 5, "maxAttemptsPerShot": 1, "maxMemoryGb": 8 },
            "shots": [{
                "id": "SH010", "beat": "b", "framing": "f", "prompt": "p",
                "targetDurationSeconds": 5.0, "startState": "s", "endState": "e",
                "conditioning": { "mode": "text_to_video" }
            }]
        }))
        .expect("plan parses");
        assert!(reference_payload_findings(&plan, &entry).is_empty());
    }

    #[test]
    fn the_lane_and_platform_follow_the_api_host_not_this_process() {
        let remote = HostFacts {
            platform: Some("linux".to_owned()),
            ..HostFacts::default()
        };
        assert_eq!(remote.lane(), ModelLane::Candle);
        assert_eq!(remote.platform_or_local(), "linux");
        let mac = HostFacts {
            platform: Some("macos".to_owned()),
            ..HostFacts::default()
        };
        assert_eq!(mac.lane(), ModelLane::Mlx);
        // Only a host that reports no platform at all falls back to this process's own.
        let unknown = HostFacts::default();
        assert_eq!(unknown.lane(), ModelLane::for_current_platform());
        assert_eq!(unknown.platform_or_local(), std::env::consts::OS);
    }

    #[test]
    fn the_memory_peak_prefers_the_metrics_route_over_the_job_snapshot() {
        let view = |pct: Option<f64>| JobView {
            status: "completed".to_owned(),
            error: None,
            message: String::new(),
            peak_gpu_memory_pct: pct,
            backend: None,
            result: Value::Null,
        };
        let bytes = json!({ "peakMemoryBytes": 115.2 * BYTES_PER_GB, "peakMemoryPct": 90.0 });
        let observed = memory_observation(Some(&bytes), &view(Some(10.0)), Some(128.0));
        assert_eq!(
            observed.source.as_deref(),
            Some("metrics.peakMemoryBytes"),
            "the snapshot's 10% must not win over a measured byte count"
        );
        assert!(observed.gb.is_some_and(|gb| (gb - 115.2).abs() < 1e-6));

        let pct_only = json!({ "peakMemoryPct": 50.0 });
        let observed = memory_observation(Some(&pct_only), &view(None), Some(128.0));
        assert_eq!(observed.source.as_deref(), Some("metrics.peakMemoryPct"));
        assert_eq!(observed.gb, Some(64.0));

        // Last resort: the job snapshot's field, which every shipped worker leaves null.
        let observed = memory_observation(None, &view(Some(25.0)), Some(128.0));
        assert_eq!(observed.source.as_deref(), Some("job.peakGpuMemoryPct"));
        assert_eq!(observed.gb, Some(32.0));
        assert!(memory_observation(None, &view(None), Some(128.0))
            .gb
            .is_none());
        // A zero row (what the cpu export job records on a real run) is "nothing measured".
        let zeroed = json!({ "peakMemoryBytes": 0, "peakMemoryPct": 0.0 });
        assert!(memory_observation(Some(&zeroed), &view(None), Some(128.0))
            .gb
            .is_none());
    }

    #[test]
    fn a_completed_job_is_settled_once_its_assets_are_persisted() {
        let with_result = |result: Value| JobView {
            status: "completed".to_owned(),
            error: None,
            message: String::new(),
            peak_gpu_memory_pct: None,
            backend: None,
            result,
        };
        assert!(!with_result(json!({ "assetWrites": [{ "type": "video" }] })).is_settled());
        assert!(
            with_result(json!({ "assetWrites": [{ "type": "video" }], "assets": [] })).is_settled(),
            "the persisted side is the positive signal, whatever else the result still carries"
        );
        assert!(with_result(json!({ "assetIds": ["asset_1"] })).is_settled());
        assert!(with_result(json!({})).is_settled());
    }

    #[test]
    fn timeline_items_are_read_back_off_the_saved_document() {
        let intended = vec!["item_sh010_abcd1234".to_owned()];
        // Sound tracks sit beside the picture track since sc-22712, so the check is scoped to the
        // track the shots live on — "as many items as intended" is only true of that one.
        let saved = json!({ "tracks": [
            { "id": "track_main", "items": [{
                "id": "item_sh010_abcd1234", "assetId": "asset_1",
                "timelineStart": 0.0, "timelineEnd": 5.1667
            }] },
            { "id": "track_dialogue", "items": [{
                "id": "item_line_sh010_abcd1234", "assetId": "asset_2",
                "timelineStart": 1.0, "timelineEnd": 3.0
            }] },
        ] });
        let persisted =
            persisted_picture_items(&saved, "track_main", &intended).expect("items read back");
        assert_eq!(persisted.len(), 1, "only the picture track is counted");
        assert!(
            (persisted[0]["timelineEnd"].as_f64().unwrap() - 5.1667).abs() < 1e-9,
            "the SAVED value is what is read, not the intended one"
        );
        // A timeline the save dropped the items from cannot be recorded as if it held them.
        let empty = json!({ "tracks": [{ "id": "track_main", "items": [] }] });
        assert!(persisted_picture_items(&empty, "track_main", &intended).is_none());
        // Nor can one whose picture track grew an item the run never assembled.
        let extra = json!({ "tracks": [{ "id": "track_main", "items": [
            { "id": "item_sh010_abcd1234", "assetId": "asset_1",
              "timelineStart": 0.0, "timelineEnd": 5.1667 },
            { "id": "item_stowaway", "assetId": "asset_9",
              "timelineStart": 5.1667, "timelineEnd": 9.0 },
        ] }] });
        assert!(persisted_picture_items(&extra, "track_main", &intended).is_none());
    }

    // `tier_maps_to_the_shared_mlx_quantize_convention` moved to `sceneworks_core::film_compile`
    // with `mlx_quantize_for_tier` itself, which now builds every video job body (sc-22713).

    #[test]
    fn export_resolution_is_the_smallest_admitted_value_covering_the_takes() {
        assert_eq!(export_resolution_for(320), 640);
        assert_eq!(export_resolution_for(720), 720);
        assert_eq!(export_resolution_for(768), 1024);
        assert_eq!(export_resolution_for(2000), 1280);
        // The timeline ratio is the admitted one CLOSEST to the take's real geometry, and it is a
        // coercion whenever the take is not already 16:9 / 9:16 / 1:1: the fixture's 576x320 takes
        // are 9:5, so the 16:9 timeline pads them (the real smoke's export landed at 1138x640).
        assert_eq!(aspect_ratio_for(576, 320), "16:9");
        assert_eq!(reduced_aspect_ratio(576, 320), "9:5");
        assert_eq!(aspect_ratio_for(320, 576), "9:16");
        assert_eq!(reduced_aspect_ratio(320, 576), "5:9");
        assert_eq!(aspect_ratio_for(768, 768), "1:1");
        assert_eq!(reduced_aspect_ratio(768, 768), "1:1");
        // 4:3 is exactly equidistant from 1:1 and 16:9 on the log scale (4/3 is their geometric
        // mean), so the tie-break decides — and a landscape take belongs in a landscape frame.
        assert_eq!(aspect_ratio_for(1024, 768), "16:9");
        assert_eq!(aspect_ratio_for(768, 1024), "9:16");
        assert_eq!(reduced_aspect_ratio(1024, 768), "4:3");
        // Only an exact match is not a coercion.
        assert_eq!(aspect_ratio_for(1280, 720), "16:9");
        assert_eq!(reduced_aspect_ratio(1280, 720), "16:9");
        assert_eq!(reduced_aspect_ratio(1344, 768), "7:4");
    }

    #[test]
    fn the_export_resolution_menu_is_the_routes_own() {
        assert_eq!(
            crate::TIMELINE_EXPORT_RESOLUTIONS,
            &[640, 720, 1024, 1280],
            "the harness picks from the list validate_timeline_export enforces"
        );
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

// -------------------------------------------------------------------------------------------
// Editable picture and continuous sound (sc-22712)
// -------------------------------------------------------------------------------------------
//
// The sequence the harness assembles is a SceneWorks timeline and nothing else: a picture track of
// selected takes in cut order, a dialogue track whose clips sit against their shots, and two beds
// placed once across the whole thing. Everything the harness needs in order to re-lay that sequence
// later — which shot an item belongs to, what a dialogue line's offset means, where a bed starts —
// travels inside the timeline document under `filmHarness`, NOT in the run record.
//
// That is deliberate. The timeline is the editable artifact and the thing the export reads; a
// layout that could only be recomputed from a run record would be a layout the editor could break
// silently. The store's validators add and validate known keys and never strip unknown ones, so the
// block survives every round trip through `PUT /timelines/:id`.

/// Track ids the harness owns. The picture id is the store's own default track, so a harness
/// timeline opens in the editor with its picture where the editor expects it.
pub const PICTURE_TRACK_ID: &str = "track_main";

/// The index of the sequence's picture track: `track_main` by id, or — if the store's default
/// track ids ever change — the first `kind: "video"` track. Same rule the assembly uses to decide
/// where the takes go (sc-22710), so a re-layout can never disagree with the assembly about which
/// track holds the shots.
fn picture_track_index(timeline: &Value) -> Option<usize> {
    let tracks = timeline.get("tracks")?.as_array()?;
    tracks
        .iter()
        .position(|track| track.get("id").and_then(Value::as_str) == Some(PICTURE_TRACK_ID))
        .or_else(|| {
            tracks
                .iter()
                .position(|track| track.get("kind").and_then(Value::as_str) == Some("video"))
        })
}
pub const DIALOGUE_TRACK_ID: &str = "track_dialogue";
pub const AMBIENCE_TRACK_ID: &str = "track_ambience";
pub const MUSIC_TRACK_ID: &str = "track_music";

/// Where the harness's own annotation lives on a timeline item.
const HARNESS_KEY: &str = "filmHarness";

const ROLE_PICTURE: &str = "picture";
const ROLE_DIALOGUE: &str = "dialogue";
const ROLE_AMBIENCE: &str = "ambience";
const ROLE_MUSIC: &str = "music";

/// Tag every harness-imported sound clip carries beside its role tag.
const SOUND_TAG: &str = "film-harness-sound";

/// The store refuses `timelineEnd <= timelineStart`, so every placed item needs a floor. One frame
/// at 25 fps is a length no edit can round away.
const MIN_ITEM_SECONDS: f64 = 0.04;

/// A sound file the run imported, as the API reported it back.
#[derive(Debug, Clone)]
struct SoundAsset {
    asset_id: String,
    /// Measured off the STORED wav by the import route, not claimed by the plan.
    duration_seconds: Option<f64>,
}

fn audio_content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("m4a" | "aac") => "audio/mp4",
        Some("flac") => "audio/flac",
        Some("ogg" | "opus") => "audio/ogg",
        _ => "audio/wav",
    }
}

/// Which generated-audio policy a shot actually runs under: its own if it declared one, the run's
/// otherwise. Both default to `mute`, which is the only default that cannot double a dialogue clip.
fn resolved_generated_audio(plan: &ProductionPlan, shot: &film_plan::Shot) -> GeneratedAudio {
    shot.generated_audio.unwrap_or(plan.sound.generated_audio)
}

fn harness_block(role: &str, run_id: &str, shot_id: Option<&str>, offset: f64) -> Value {
    json!({
        "runId": run_id,
        "role": role,
        "shotId": shot_id,
        "offsetSeconds": offset,
    })
}

fn harness_str<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(HARNESS_KEY)?.get(key)?.as_str()
}

fn harness_f64(item: &Value, key: &str) -> f64 {
    item.get(HARNESS_KEY)
        .and_then(|block| block.get(key))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

fn number(item: &Value, key: &str, fallback: f64) -> f64 {
    item.get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

/// How long an item occupies the timeline, from its own source range and speed. This — not the
/// current `timelineStart`/`timelineEnd` — is what survives a trim or a reorder, which is why every
/// layout is recomputed from it rather than nudged.
fn item_span(item: &Value) -> f64 {
    let source_in = number(item, "sourceIn", 0.0).max(0.0);
    let source_out = number(item, "sourceOut", source_in + MIN_ITEM_SECONDS);
    let speed = number(item, "speed", 1.0).clamp(0.1, 8.0);
    ((source_out - source_in) / speed).max(MIN_ITEM_SECONDS)
}

/// Positions are written UNROUNDED, and one accumulated cursor supplies both an item's end and the
/// next item's start.
///
/// Rounding here would be the bug, not the tidiness: `plan_segments` inserts a black gap wherever
/// an item does not abut the previous one, so a half-millisecond of rounding drift between two
/// shots that are supposed to cut straight together becomes a black frame in the export. Writing
/// the same `f64` to both ends makes the two exactly equal, by construction, at any precision.
fn ms(value: f64) -> f64 {
    value
}

fn picture_track(track_id: &str, items: Vec<Value>) -> Value {
    json!({
        "id": track_id,
        "name": "Main",
        "kind": "video",
        "role": ROLE_PICTURE,
        "locked": false,
        "muted": false,
        "gain": 1.0,
        "items": items,
    })
}

fn audio_track(id: &str, name: &str, role: &str, bus: &SoundBus, items: Vec<Value>) -> Value {
    json!({
        "id": id,
        "name": name,
        "kind": "audio",
        "role": role,
        "locked": false,
        "muted": bus.muted,
        "gain": bus.gain.clamp(0.0, 4.0),
        "items": items,
    })
}

/// A bed track carrying exactly one item, placed once for the whole sequence.
fn bed_track(
    track_id: &str,
    name: &str,
    role: &str,
    bed: &SoundBed,
    asset: &SoundAsset,
    run_id: &str,
) -> Value {
    let mut block = harness_block(role, run_id, None, 0.0);
    block["startSeconds"] = json!(bed.start_seconds);
    let item = json!({
        "id": format!("item_{role}_{}", &run_id[4..12]),
        "trackId": track_id,
        "assetId": asset.asset_id,
        "type": "audio",
        "displayName": format!("{name} — {}", bed.role).chars().take(160).collect::<String>(),
        "sourceIn": bed.source_in_seconds,
        // Given its real span by `relayout_timeline`, which is the only place that knows how long
        // the assembled picture turned out to be.
        "sourceOut": bed.source_in_seconds + MIN_ITEM_SECONDS,
        "timelineStart": 0.0,
        "timelineEnd": MIN_ITEM_SECONDS,
        "speed": 1.0,
        "fit": "fit",
        "volume": 1.0,
        "fadeInSeconds": bed.fade_in_seconds,
        "fadeOutSeconds": bed.fade_out_seconds,
        HARNESS_KEY: block,
    });
    json!({
        "id": track_id,
        "name": name,
        "kind": "audio",
        "role": role,
        "locked": false,
        "muted": bed.muted,
        "gain": bed.gain.clamp(0.0, 4.0),
        "items": [item],
    })
}

/// Re-lay the whole sequence and return its new duration.
///
/// This is the single place that decides where anything sits, and every editing command goes
/// through it rather than adjusting positions itself — which is what makes "trim", "reorder" and
/// "replace the take" three ways of changing ONE input to the same function instead of three
/// chances to get the ripple wrong.
///
/// 1. Picture items are laid end to end from zero in `order` (or in their current order), each
///    taking exactly the span its own source range and speed imply. There are no gaps: the cuts are
///    the cuts.
/// 2. Each dialogue clip is re-placed at its shot's new start plus the offset it has always had, so
///    a line stays against its beat no matter what happened to the shots before it.
/// 3. Each bed is re-spanned from its declared start to the new end of the picture. A bed is placed
///    ONCE, so it plays straight through the cuts rather than restarting at each one.
///
/// Sound is clamped to the picture: a clip that would overrun the last frame is shortened and one
/// that would start past it is dropped. That is what keeps the saved timeline's duration and the
/// exported file's duration the same number.
fn relayout_timeline(timeline: &mut Value, order: Option<&[String]>) -> Result<f64, HarnessError> {
    let picture_index = picture_track_index(timeline).ok_or_else(|| {
        HarnessError::Transport(
            "timeline has no track_main and no video track to re-lay".to_owned(),
        )
    })?;
    let tracks = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| HarnessError::Transport("timeline has no tracks array".to_owned()))?;

    let mut shot_starts: BTreeMap<String, f64> = BTreeMap::new();
    let mut duration = 0.0_f64;
    if let Some(track) = tracks.get_mut(picture_index) {
        let items = track
            .get_mut("items")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| HarnessError::Transport("picture track has no items".to_owned()))?;
        if let Some(order) = order {
            let present: BTreeSet<String> = items
                .iter()
                .filter_map(|item| harness_str(item, "shotId").map(str::to_owned))
                .collect();
            let requested: BTreeSet<String> = order.iter().cloned().collect();
            if present != requested {
                return Err(HarnessError::Transport(format!(
                    "the requested order names {:?} but the sequence holds {:?}; a reorder must \
                     list every shot on the picture track exactly once",
                    requested, present
                )));
            }
            items.sort_by_key(|item| {
                harness_str(item, "shotId")
                    .and_then(|shot| order.iter().position(|id| id == shot))
                    .unwrap_or(usize::MAX)
            });
        } else {
            items.sort_by(|left, right| {
                number(left, "timelineStart", 0.0).total_cmp(&number(right, "timelineStart", 0.0))
            });
        }
        let mut cursor = 0.0_f64;
        for item in items.iter_mut() {
            let shot_id = harness_str(item, "shotId").map(str::to_owned);
            let span = item_span(item);
            let start = ms(cursor);
            cursor += span;
            let end = ms(cursor);
            item["timelineStart"] = json!(start);
            item["timelineEnd"] = json!(end);
            if let Some(shot_id) = shot_id {
                shot_starts.insert(shot_id, start);
            }
        }
        duration = ms(cursor);
    }

    for track in tracks.iter_mut() {
        if track.get("kind").and_then(Value::as_str) != Some("audio") {
            continue;
        }
        let Some(items) = track.get_mut("items").and_then(Value::as_array_mut) else {
            continue;
        };
        items.retain_mut(|item| {
            let role = harness_str(item, "role").unwrap_or_default().to_owned();
            match role.as_str() {
                ROLE_DIALOGUE | ROLE_AMBIENCE | ROLE_MUSIC => {}
                // A clip the harness did not place — the editor's own — is NOT the harness's to
                // delete. Dropping anything that merely lands near the new end (which the shared
                // rule below does, for items the harness owns and can re-place from the plan)
                // would destroy a user's own work with no diagnostic and no undo, so an unowned
                // clip is only clamped into the sequence, and dropped only when the re-layout
                // leaves it no room at all.
                _ => {
                    let start = number(item, "timelineStart", 0.0).max(0.0);
                    let end = number(item, "timelineEnd", start + item_span(item)).min(duration);
                    if start >= duration || end <= start {
                        let item_id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        tracing::warn!(
                            item_id,
                            start,
                            duration,
                            "film-harness: an editor-placed audio clip starts past the end of the \
                             re-laid sequence and has been dropped"
                        );
                        return false;
                    }
                    item["timelineStart"] = json!(ms(start));
                    item["timelineEnd"] = json!(ms(end));
                    return true;
                }
            }
            let (start, end) = match role.as_str() {
                ROLE_DIALOGUE => {
                    let Some(shot_start) =
                        harness_str(item, "shotId").and_then(|shot| shot_starts.get(shot).copied())
                    else {
                        // The shot this line belongs to is no longer in the sequence.
                        return false;
                    };
                    let start = shot_start + harness_f64(item, "offsetSeconds");
                    (start, start + item_span(item))
                }
                // ROLE_AMBIENCE | ROLE_MUSIC, the only other arm the match above admits.
                _ => {
                    let start = harness_f64(item, "startSeconds");
                    (start, duration)
                }
            };
            if start >= duration - MIN_ITEM_SECONDS {
                return false;
            }
            let end = end.min(duration).max(start + MIN_ITEM_SECONDS);
            item["timelineStart"] = json!(ms(start));
            item["timelineEnd"] = json!(ms(end));
            if matches!(role.as_str(), ROLE_AMBIENCE | ROLE_MUSIC) {
                // A bed's source range follows its span, so the whole stretch of the file that
                // plays under the sequence is asked for rather than a fixed four seconds.
                let source_in = number(item, "sourceIn", 0.0).max(0.0);
                item["sourceOut"] = json!(ms(source_in + (end - start)));
            }
            true
        });
    }

    timeline["duration"] = json!(duration);
    Ok(duration)
}

fn item_record(item: &Value, track_gain: f64) -> TimelineItemRecord {
    let source_in = number(item, "sourceIn", 0.0);
    TimelineItemRecord {
        shot_id: harness_str(item, "shotId").map(str::to_owned),
        item_id: item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        asset_id: item
            .get("assetId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        timeline_start: number(item, "timelineStart", 0.0),
        timeline_end: number(item, "timelineEnd", 0.0),
        source_in,
        source_out: number(item, "sourceOut", source_in),
        gain: track_gain * number(item, "volume", 1.0),
        fade_in_seconds: number(item, "fadeInSeconds", 0.0),
        fade_out_seconds: number(item, "fadeOutSeconds", 0.0),
        generated_audio: match item.get("generatedAudio").and_then(Value::as_str) {
            Some("include") => Some(GeneratedAudio::Include),
            Some("mute") => Some(GeneratedAudio::Mute),
            _ => None,
        },
    }
}

/// Describe the assembled sequence for the run record, read back off the timeline document rather
/// than rebuilt from the plan — so the record says what was SAVED, including anything an edit
/// changed.
#[allow(clippy::too_many_arguments)]
fn timeline_record(
    timeline_id: &str,
    name: &str,
    aspect_ratio: &str,
    source_aspect_ratio: Option<String>,
    source_width: Option<u32>,
    source_height: Option<u32>,
    fps: u32,
    duration: f64,
    timeline: &Value,
    picture_track_id: &str,
    generated_audio_default: GeneratedAudio,
    edits: Vec<TimelineEditRecord>,
) -> TimelineRecord {
    let mut tracks = Vec::new();
    let mut picture_items = Vec::new();
    for track in timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let track_id = track
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let kind = track
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("video")
            .to_owned();
        let role = track
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("sound")
            .to_owned();
        let gain = number(track, "gain", 1.0);
        let muted = track.get("muted").and_then(Value::as_bool).unwrap_or(false);
        let items: Vec<TimelineItemRecord> = track
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|item| item_record(item, gain))
            .collect();
        if items.is_empty() && track_id != picture_track_id {
            continue;
        }
        if track_id == picture_track_id {
            picture_items = items.clone();
        }
        tracks.push(TimelineTrackRecord {
            track_id,
            kind,
            role,
            gain,
            muted,
            items,
        });
    }
    TimelineRecord {
        timeline_id: timeline_id.to_owned(),
        name: name.to_owned(),
        aspect_ratio: aspect_ratio.to_owned(),
        source_aspect_ratio,
        source_width,
        source_height,
        fps,
        duration_seconds: duration,
        items: picture_items,
        tracks,
        generated_audio_default,
        edits,
    }
}

/// Dispatch the `timeline_export` job and wait for it under the run's budgets.
#[allow(clippy::too_many_arguments)]
async fn export_timeline(
    client: &Client<'_>,
    project_id: &str,
    timeline_id: &str,
    resolution: u32,
    fps: u32,
    bounds: PollBounds,
    limits: &film_plan::PlanLimits,
) -> Result<(ExportRecord, PollStop), HarnessError> {
    let export_job = client
        .expect_ok(
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
            Some(json!({
                "resolution": resolution,
                "fps": fps,
                "requestedGpu": "auto",
            })),
        )
        .await?;
    let export_job_id = export_job
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| HarnessError::Transport(format!("export response has no id: {export_job}")))?
        .to_owned();
    let settle_grace = bounds.settle_grace;
    let (view, poll_stop) = client.wait_for_job(&export_job_id, bounds).await?;
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
        PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
        PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        PollStop::Operator => "canceled_by_operator".to_owned(),
    };
    let ok = status == "completed" && asset_id.is_some();
    Ok((
        ExportRecord {
            job_id: export_job_id,
            status,
            asset_id,
            render_path,
            error: (!ok).then(|| match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the export job reached {} but its assets never settled within {:.0}s",
                    view.status,
                    settle_grace.as_secs_f64()
                ),
                PollStop::Operator => "canceled by operator during the export".to_owned(),
                PollStop::ShotBudget => format!(
                    "export exceeded the per-job budget of {}s",
                    limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "run exceeded its budget of {}s during the export",
                    limits.max_run_seconds
                ),
            }),
        },
        poll_stop,
    ))
}

/// One change to an assembled sequence.
///
/// Each of these changes exactly one input to [`relayout_timeline`] and then lets it recompute
/// every position, which is why a trim ripples, a reorder keeps every line against its beat, and a
/// replaced take re-times the sequence around its new length — without any of the three knowing
/// about the other two.
#[derive(Debug, Clone)]
pub enum TimelineEdit {
    /// Change a shot's source range. `None` leaves that end where it is.
    Trim {
        shot_id: String,
        source_in: Option<f64>,
        source_out: Option<f64>,
    },
    /// Put the picture track in this order. Must name every shot on it, exactly once.
    Reorder { shot_ids: Vec<String> },
    /// Point a shot at a different take. The asset must already exist in the project — choosing it
    /// is sc-22711's job, placing it is this one's.
    ReplaceTake { shot_id: String, asset_id: String },
}

impl TimelineEdit {
    fn kind(&self) -> &'static str {
        match self {
            Self::Trim { .. } => "trim",
            Self::Reorder { .. } => "reorder",
            Self::ReplaceTake { .. } => "replace_take",
        }
    }
}

/// Everything an edit needs beyond the transport.
#[derive(Debug, Clone)]
pub struct EditOptions {
    /// The `run.json` written by [`run`]. It names the project and the timeline, and it is
    /// rewritten in place with the edited sequence.
    pub run_record_path: PathBuf,
    /// Re-export the MP4 after the edit.
    pub export: bool,
    pub poll_interval: Duration,
}

/// Apply one edit to a run's assembled sequence: change the timeline, save it, rewrite the run
/// record, and optionally re-export.
///
/// Shot -> asset links are never rebuilt from the plan here. The picture item already carries its
/// shot id and its version history, so a trim or a reorder moves the item that is already bound to
/// the take a human chose, and a replacement appends to that history instead of overwriting it.
pub async fn edit_timeline(
    transport: &dyn ApiTransport,
    options: &EditOptions,
    edit: TimelineEdit,
) -> Result<RunRecord, HarnessError> {
    // An edit re-exports at most one job and is interruptible through the same control a run uses;
    // nothing here dispatches a render, so the default (never canceled) is the whole contract.
    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    let text = std::fs::read_to_string(&options.run_record_path)?;
    let mut record: RunRecord = serde_json::from_str(&text).map_err(|error| {
        HarnessError::Io(format!(
            "{} is not a film-harness run record: {error}",
            options.run_record_path.display()
        ))
    })?;
    let project_id = record
        .project_id
        .clone()
        .ok_or_else(|| HarnessError::Io("run record has no project id".to_owned()))?;
    let existing = record.timeline.clone().ok_or_else(|| {
        HarnessError::Io("run record has no assembled timeline to edit".to_owned())
    })?;
    let timeline_id = existing.timeline_id.clone();

    let mut timeline = client
        .expect_ok(
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            None,
        )
        .await?;

    let (order, detail) = match &edit {
        TimelineEdit::Reorder { shot_ids } => (Some(shot_ids.clone()), shot_ids.join(" -> ")),
        TimelineEdit::Trim {
            shot_id,
            source_in,
            source_out,
        } => {
            // Measure the take FIRST, the way `ReplaceTake` does. An out point past the end of the
            // media is silent otherwise: `relayout_timeline` writes a `timelineEnd` longer than the
            // file, `render_item_segment` reports the DECLARED duration while `-t` yields a short
            // segment, and the picture then comes out shorter than the sequence it was saved from —
            // the same drift a crossfade used to cause, plus a wrong duration in the sidecar.
            let take_id = picture_item_mut(&mut timeline, shot_id)?
                .get("assetId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let take_seconds = if take_id.is_empty() {
                None
            } else {
                client
                    .expect_ok(
                        "GET",
                        &format!("/api/v1/projects/{project_id}/assets/{take_id}"),
                        None,
                    )
                    .await?
                    .get("file")
                    .and_then(|file| file.get("duration"))
                    .and_then(Value::as_f64)
                    .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            };
            let item = picture_item_mut(&mut timeline, shot_id)?;
            let current_in = number(item, "sourceIn", 0.0);
            let current_out = number(item, "sourceOut", current_in + MIN_ITEM_SECONDS);
            let new_in = source_in.unwrap_or(current_in).max(0.0);
            let requested_out = source_out.unwrap_or(current_out);
            // Clamp rather than refuse when there is still a usable range left: "keep everything
            // from 1s on" is a reasonable thing to ask of a take whose length the caller does not
            // know. Only a range that lands entirely past the end of the media is an error, and it
            // says how long the take actually is.
            let new_out = match take_seconds {
                Some(seconds) if requested_out > seconds => {
                    if new_in + MIN_ITEM_SECONDS >= seconds {
                        return Err(HarnessError::Io(format!(
                            "trim of {shot_id} asks for {new_in:.3}..{requested_out:.3} but its \
                             take is only {seconds:.3}s long; the in point must be at least \
                             {MIN_ITEM_SECONDS}s before the end of the take"
                        )));
                    }
                    seconds
                }
                _ => requested_out,
            };
            if !new_in.is_finite() || !new_out.is_finite() || new_out <= new_in + MIN_ITEM_SECONDS {
                return Err(HarnessError::Io(format!(
                    "trim of {shot_id} would leave a source range of {new_in}..{new_out}; the out \
                     point must be at least {MIN_ITEM_SECONDS}s after the in point"
                )));
            }
            item["sourceIn"] = json!(ms(new_in));
            item["sourceOut"] = json!(ms(new_out));
            (
                None,
                format!("{shot_id} source range {new_in:.3}..{new_out:.3}"),
            )
        }
        TimelineEdit::ReplaceTake { shot_id, asset_id } => {
            // Measure the replacement before touching the timeline, so a bad asset id fails before
            // the sequence is half-edited.
            let asset = client
                .expect_ok(
                    "GET",
                    &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
                    None,
                )
                .await?;
            let duration = asset
                .get("file")
                .and_then(|file| file.get("duration"))
                .and_then(Value::as_f64)
                .filter(|seconds| *seconds > 0.0);
            let item = picture_item_mut(&mut timeline, shot_id)?;
            let previous = item
                .get("assetId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let span = duration.unwrap_or_else(|| item_span(item));
            item["assetId"] = json!(asset_id);
            item["currentVersionAssetId"] = json!(asset_id);
            item["sourceIn"] = json!(0.0);
            item["sourceOut"] = json!(ms(span.max(MIN_ITEM_SECONDS)));
            if let Some(history) = item.get_mut("versionHistory").and_then(Value::as_array_mut) {
                history.push(json!({
                    "assetId": asset_id,
                    "source": "replacement",
                    "createdAt": utc_now(),
                    "note": format!("film-harness take replacement for {shot_id}"),
                }));
            }
            if let Some(versions) = item
                .get_mut("versionAssetIds")
                .and_then(Value::as_array_mut)
            {
                if !versions.iter().any(|value| value == &json!(asset_id)) {
                    versions.push(json!(asset_id));
                }
            }
            (
                None,
                format!("{shot_id} take {previous} -> {asset_id} ({span:.3}s)"),
            )
        }
    };

    let duration = relayout_timeline(&mut timeline, order.as_deref())?;
    let saved = client
        .expect_ok(
            "PUT",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            Some(json!({ "timeline": timeline })),
        )
        .await?;

    let mut edits = existing.edits.clone();
    edits.push(TimelineEditRecord {
        kind: edit.kind().to_owned(),
        applied_at: utc_now(),
        detail,
        duration_seconds: duration,
    });
    let picture_track_id = picture_track_index(&saved)
        .and_then(|index| saved["tracks"][index]["id"].as_str())
        .unwrap_or(PICTURE_TRACK_ID)
        .to_owned();
    // The take geometry the sequence was sized from does not change when the picture is re-cut, so
    // it is carried over from the record rather than re-derived (sc-22710).
    record.timeline = Some(timeline_record(
        &timeline_id,
        &existing.name,
        &existing.aspect_ratio,
        existing.source_aspect_ratio.clone(),
        existing.source_width,
        existing.source_height,
        existing.fps,
        saved
            .get("duration")
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .unwrap_or(duration),
        &saved,
        &picture_track_id,
        existing.generated_audio_default,
        edits,
    ));

    if options.export {
        let tallest = saved
            .get("height")
            .and_then(Value::as_u64)
            .unwrap_or(u64::from(DEFAULT_EXPORT_HEIGHT)) as u32;
        let shot_budget = Duration::from_secs(record.limits.max_shot_seconds);
        let (export_record, _) = export_timeline(
            &client,
            &project_id,
            &timeline_id,
            export_resolution_for(tallest),
            existing.fps,
            PollBounds {
                shot_deadline: Instant::now() + shot_budget,
                run_deadline: Instant::now() + Duration::from_secs(record.limits.max_run_seconds),
                poll_interval: options.poll_interval,
                cancel_grace: CANCEL_GRACE.min(shot_budget),
                settle_grace: ASSET_SETTLE_GRACE.min(shot_budget),
            },
            &record.limits,
        )
        .await?;
        record.outcome = if export_record.status == "completed" {
            RunOutcome::Completed
        } else {
            RunOutcome::Failed
        };
        record.export = Some(export_record);
    }

    let json = serde_json::to_string_pretty(&record)
        .map_err(|error| HarnessError::Io(error.to_string()))?;
    std::fs::write(&options.run_record_path, format!("{json}\n"))?;
    Ok(record)
}

/// Fallback export height when a timeline document does not carry one.
const DEFAULT_EXPORT_HEIGHT: u32 = 720;

fn picture_item_mut<'a>(
    timeline: &'a mut Value,
    shot_id: &str,
) -> Result<&'a mut Value, HarnessError> {
    let index = picture_track_index(timeline);
    index
        .and_then(|index| {
            timeline
                .get_mut("tracks")
                .and_then(Value::as_array_mut)
                .and_then(|tracks| tracks.get_mut(index))
        })
        .and_then(|track| track.get_mut("items").and_then(Value::as_array_mut))
        .and_then(|items| {
            items
                .iter_mut()
                .find(|item| harness_str(item, "shotId") == Some(shot_id))
        })
        .ok_or_else(|| {
            HarnessError::Io(format!(
                "shot {shot_id:?} is not on the sequence's picture track"
            ))
        })
}
