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

use sceneworks_core::film_plan::{
    self, AttemptRecord, ConditioningAssets, ExportPending, ExportRecord, HardwareRecord,
    IntendedState, ModelLane, ModelRecord, PlanDiagnostic, ProductionDecision, ProductionPlan,
    ReferenceAssetRecord, ReferencePack, ReviewFlag, RunOutcome, RunRecord, RunState, RunStop,
    ShotOutcome, ShotRunRecord, SourceDocument, TakeRecord, TakeRejection, TimelineItemRecord,
    TimelineRecord, RUN_RECORD_SCHEMA_VERSION,
};
use sceneworks_core::time::{parse_utc_seconds, utc_now};
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

/// File name of the run record inside the run directory.
pub const RUN_RECORD_FILE: &str = "run.json";

/// File `film-harness cancel` drops in the run directory. A controller in another process polls for
/// it, which is what makes cancellation work without a shared handle to the running controller
/// (sc-22711).
pub const CANCEL_SENTINEL_FILE: &str = "cancel.requested";

/// How many jobs a reconciliation lists when looking for a job this run created.
const JOB_LOOKUP_LIMIT: u32 = 500;

/// Attempt statuses that need no further reconciliation on a resume.
const TERMINAL_ATTEMPT_STATUSES: &[&str] = &[
    "completed",
    "failed",
    "canceled",
    "canceled_by_operator",
    "interrupted",
    "timed_out",
    "rejected",
];

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
    /// The requested action does not apply to the run record on disk — it is finished and not
    /// resumable, its documents no longer hash to what the run was started from, or it names no
    /// such shot. Nothing was dispatched (sc-22711).
    Refused(String),
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
            Self::Refused(message) => write!(f, "refused: {message}"),
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

/// Cooperative cancellation for a run. The `film-harness` binary flips it from its `ctrl_c`
/// handler; the harness then cancels the in-flight job through the API, stops dispatching, and
/// still writes the run record — rather than leaving a 45-minute render on the GPU with no record
/// of it, which is the opposite of what the plan's limits are for.
/// sc-22711 adds the second source: a controller can also be asked to stop from ANOTHER process, by
/// `film-harness cancel --out DIR` dropping [`CANCEL_SENTINEL_FILE`] in the run directory. A
/// controller checks both before every dispatch and on every poll, so a cancel lands within one
/// poll interval either way.
#[derive(Debug, Clone, Default)]
pub struct RunControl {
    canceled: Arc<AtomicBool>,
    sentinel: Option<PathBuf>,
}

impl RunControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// A control that is also tripped by the presence of `<run_dir>/cancel.requested`.
    pub fn watching(run_dir: &Path) -> Self {
        Self {
            canceled: Arc::new(AtomicBool::new(false)),
            sentinel: Some(run_dir.join(CANCEL_SENTINEL_FILE)),
        }
    }

    /// Ask the in-flight run to cancel its job and stop dispatching. Idempotent.
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::SeqCst);
    }

    pub fn is_canceled(&self) -> bool {
        if self.canceled.load(Ordering::SeqCst) {
            return true;
        }
        self.sentinel
            .as_deref()
            .is_some_and(|path| path.try_exists().unwrap_or(false))
    }
}

/// Ask the run in `run_dir` to stop, from outside the process running it. Returns the sentinel it
/// wrote. A run that is not currently held picks this up on its next start, so
/// [`clear_cancel_request`] runs before a resume.
pub fn request_cancel(run_dir: &Path) -> Result<PathBuf, HarnessError> {
    std::fs::create_dir_all(run_dir)?;
    let path = run_dir.join(CANCEL_SENTINEL_FILE);
    std::fs::write(&path, format!("{}\n", utc_now()))?;
    Ok(path)
}

/// Remove a stale cancel sentinel so a resume is not cancelled by the request that stopped the
/// previous controller.
pub fn clear_cancel_request(run_dir: &Path) -> Result<(), HarnessError> {
    let path = run_dir.join(CANCEL_SENTINEL_FILE);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// What `resume` and `replace-take` need. Everything else — which plan, which pack, which project,
/// which shots, which limits — comes from the run record in `out_dir`, because those are properties
/// of the run being continued, not of the invocation continuing it (sc-22711).
#[derive(Debug, Clone)]
pub struct ResumeOptions {
    /// The run directory holding `run.json`.
    pub out_dir: PathBuf,
    pub poll_interval: Duration,
    /// Assemble and export the timeline. For `replace-take`, whether to re-export after the new
    /// take is selected (without it the existing export is simply marked stale).
    pub export: bool,
    pub require_installed: bool,
    pub control: RunControl,
}

impl ResumeOptions {
    /// Defaults for a run directory: watch that directory's cancel sentinel, poll every 5s, export.
    pub fn new(out_dir: PathBuf) -> Self {
        let control = RunControl::watching(&out_dir);
        Self {
            out_dir,
            poll_interval: Duration::from_secs(5),
            export: true,
            require_installed: true,
            control,
        }
    }
}

/// Read the run record in `run_dir` without touching the API. What `film-harness status` prints and
/// what `resume` / `replace-take` start from.
pub fn read_run_record(run_dir: &Path) -> Result<RunRecord, HarnessError> {
    let path = run_dir.join(RUN_RECORD_FILE);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        HarnessError::Refused(format!("cannot read {}: {error}", path.display()))
    })?;
    serde_json::from_str(&text).map_err(|error| {
        HarnessError::Refused(format!("{} is not a run record: {error}", path.display()))
    })
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

    /// Every job the API holds for `project_id`, newest first.
    async fn project_jobs(&self, project_id: &str) -> Result<Vec<Value>, HarnessError> {
        let jobs = self
            .expect_ok(
                "GET",
                &format!("/api/v1/jobs?projectId={project_id}&limit={JOB_LOOKUP_LIMIT}"),
                None,
            )
            .await?;
        Ok(jobs.as_array().cloned().unwrap_or_default())
    }

    /// The job this run already created for `key`, if any.
    ///
    /// This is the whole answer to the "created the job, died before recording its id" window: the
    /// key is in the payload the API persisted, so the controller that comes back finds its own job
    /// instead of enqueuing a second one for the same attempt (sc-22711).
    async fn find_job_by_idempotency_key(
        &self,
        project_id: &str,
        key: &str,
    ) -> Result<Option<String>, HarnessError> {
        Ok(self
            .project_jobs(project_id)
            .await?
            .iter()
            .find(|job| {
                job.pointer("/payload/advanced/filmHarness/idempotencyKey")
                    .and_then(Value::as_str)
                    == Some(key)
            })
            .and_then(|job| job.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned))
    }

    /// The newest `timeline_export` job this project holds for `timeline_id`, ignoring `exclude`
    /// (the export a re-export supersedes). The export route takes no payload field of our own, so
    /// the timeline id — which the harness creates, names after the run and never shares — is the
    /// key.
    async fn find_export_job(
        &self,
        project_id: &str,
        timeline_id: &str,
        exclude: Option<&str>,
    ) -> Result<Option<String>, HarnessError> {
        let jobs = self.project_jobs(project_id).await?;
        let mut candidates: Vec<(&str, &str)> = jobs
            .iter()
            .filter(|job| job.get("type").and_then(Value::as_str) == Some("timeline_export"))
            .filter(|job| {
                job.pointer("/payload/timelineId").and_then(Value::as_str) == Some(timeline_id)
            })
            .filter_map(|job| {
                Some((
                    job.get("id")?.as_str()?,
                    job.get("createdAt").and_then(Value::as_str).unwrap_or(""),
                ))
            })
            .filter(|(id, _)| Some(*id) != exclude)
            .collect();
        candidates.sort_by(|left, right| left.1.cmp(right.1));
        Ok(candidates.last().map(|(id, _)| (*id).to_owned()))
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

/// What the run learned about the host and the worker that will render.
#[derive(Debug, Clone, Default)]
struct HostFacts {
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
    fn lane(&self) -> ModelLane {
        match self.platform.as_deref() {
            Some(platform) => ModelLane::for_platform(platform),
            None => ModelLane::for_current_platform(),
        }
    }

    fn platform_or_local(&self) -> &str {
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
    /// [`idempotency_key`] for this attempt, recorded before the job is created and stamped into
    /// the payload so a replay can recognise its own job (sc-22711).
    idempotency_key: &'a str,
    fps: u32,
    width: u32,
    height: u32,
    assets: &'a ConditioningAssets,
}

/// The key one attempt of one shot dispatches under. Stable across restarts because every part of
/// it is: the run id is in the record, the shot id is in the plan, and attempt numbers never repeat
/// within a shot (sc-22711).
pub fn idempotency_key(run_id: &str, shot_id: &str, attempt: u32) -> String {
    format!("{run_id}:{shot_id}:a{attempt}")
}

fn video_job_body(dispatch: &ShotDispatch<'_>) -> Value {
    let ShotDispatch {
        plan,
        shot,
        project_id,
        run_id,
        attempt,
        idempotency_key,
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
            "idempotencyKey": idempotency_key,
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
    let record_path = out_dir.join(RUN_RECORD_FILE);
    write_atomically(&record_path, json.as_bytes())?;
    // Written once and then left alone, so an edit to the source plan mid-run cannot rewrite the
    // copy the run was actually started from — which is what a resume re-hashes (sc-22711).
    for (source, copy) in [
        (plan_path, out_dir.join("plan.json")),
        (pack_path, out_dir.join("references.json")),
    ] {
        if !copy.exists() {
            if let Ok(text) = std::fs::read(source) {
                write_atomically(&copy, &text)?;
            }
        }
    }
    if let Some(project_path) = record.project_path.as_deref().map(Path::new) {
        if project_path.is_dir() {
            let project_record_dir = project_path.join("film-harness").join(&record.run_id);
            std::fs::create_dir_all(&project_record_dir)?;
            write_atomically(&project_record_dir.join(RUN_RECORD_FILE), json.as_bytes())?;
        }
    }
    Ok(record_path)
}

/// Write `bytes` to `path` through a sibling temp file and a rename.
///
/// sc-22711 rewrites the record at every state transition rather than once at the end, so a
/// controller killed during a write must leave the PREVIOUS record intact rather than a truncated
/// file no resume can parse.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), HarnessError> {
    let temp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    std::fs::write(&temp, bytes)?;
    std::fs::rename(&temp, path)?;
    Ok(())
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

fn model_findings(
    plan: &ProductionPlan,
    entry: Option<&JsonObject<String, Value>>,
    require_installed: bool,
    facts: &HostFacts,
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
    findings.extend(platform_reachability_finding(
        &plan.model.id,
        entry,
        facts.platform_or_local(),
    ));
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

/// The model entry, host facts and fps a run needs once the documents themselves are valid.
struct Prepared {
    entry: JsonObject<String, Value>,
    facts: HostFacts,
    fps: u32,
}

/// Resolve the catalog entry and the host facts and judge the plan against both. `Ok(Err(findings))`
/// is a refusal: the caller writes a `rejected` record and creates nothing.
async fn prepare(
    client: &Client<'_>,
    plan: &ProductionPlan,
    export: bool,
    require_installed: bool,
) -> Result<Result<Prepared, Vec<PlanDiagnostic>>, HarnessError> {
    let facts = discover_host(client).await?;
    let entry = resolve_model_entry(client, &plan.model.id).await?;
    let mut findings = model_findings(plan, entry.as_ref(), require_installed, &facts);
    if findings.is_empty() {
        findings.extend(host_findings(plan, &facts, export));
    }
    if !findings.is_empty() {
        return Ok(Err(findings));
    }
    let entry = entry.expect("model findings are empty only with an entry");
    let fps = film_plan::plan_fps(plan, &entry).expect("validated against the model");
    Ok(Ok(Prepared { entry, facts, fps }))
}

/// One shot's selected take, resolved to everything the timeline needs.
struct SelectedTake {
    shot_id: String,
    take: TakeRecord,
    width: u32,
    height: u32,
    job_id: Option<String>,
}

/// An asset `run_id` already imported for `role`, matched on the provenance the import stamped.
/// This is what keeps a replay from importing the same reference twice when the controller died
/// between the upload and the record write (sc-22711).
fn find_imported_reference(
    assets: &[Value],
    run_id: &str,
    role: &str,
    sha256: &str,
) -> Option<String> {
    assets
        .iter()
        .find(|asset| {
            let field = |name: &str| {
                asset
                    .pointer("/extra/filmHarness")
                    .and_then(|block| block.get(name))
                    .and_then(Value::as_str)
            };
            field("runId") == Some(run_id)
                && field("role") == Some(role)
                && field("sha256") == Some(sha256)
        })
        .and_then(|asset| asset.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// How long an attempt has already been in flight.
///
/// A controller that died mid-poll never wrote `elapsedSeconds`, so the recorded value alone would
/// hand a hung job a fresh per-shot budget on every restart. `startedAt` IS persisted at dispatch,
/// and the job kept running while nothing was watching it, so wall-clock since then is the honest
/// number; the larger of the two wins.
fn attempt_spent_seconds(attempt: &AttemptRecord) -> f64 {
    let recorded = attempt.elapsed_seconds.max(0.0);
    let by_clock = parse_utc_seconds(&attempt.started_at)
        .map(|started| (sceneworks_core::time::now_unix_seconds() - started).max(0) as f64)
        .unwrap_or(0.0);
    recorded.max(by_clock)
}

/// One controller holding one run.
///
/// Everything that reaches the API goes through here, and every state change is followed by a
/// [`Session::persist`], so the record on disk is never more than one transition behind what the
/// API has. That is what makes the three entry points ([`run`], [`resume`], [`replace_take`]) the
/// same code: they differ only in the record they start from and the deadline they run under.
struct Session<'a> {
    client: Client<'a>,
    transport: &'a dyn ApiTransport,
    plan_path: PathBuf,
    pack_path: PathBuf,
    out_dir: PathBuf,
    poll_interval: Duration,
    export: bool,
    plan: ProductionPlan,
    pack: ReferencePack,
    entry: JsonObject<String, Value>,
    facts: HostFacts,
    fps: u32,
    record: RunRecord,
    started: Instant,
    /// Wall-clock earlier controllers already spent on this run. The plan's `maxRunSeconds` bounds
    /// the run, not one attempt at it, so a resume inherits the spend.
    prior_elapsed: f64,
    run_deadline: Instant,
    role_assets: BTreeMap<String, String>,
    /// Set the moment dispatch stops, with the outcome that stop implies.
    stop: Option<(RunOutcome, RunStop)>,
}

impl Session<'_> {
    /// Total wall-clock this run has consumed, across every controller that has held it.
    fn elapsed(&self) -> f64 {
        self.prior_elapsed + self.started.elapsed().as_secs_f64()
    }

    /// Write the record. Called after every state transition — this is the durability contract.
    fn persist(&mut self) -> Result<(), HarnessError> {
        self.record.elapsed_seconds = self.elapsed();
        persist_record(
            &self.record,
            &self.out_dir,
            &self.plan_path,
            &self.pack_path,
        )?;
        Ok(())
    }

    fn note_decision(&mut self, action: &str, shot_id: Option<&str>, detail: String) {
        self.record.decisions.push(ProductionDecision {
            at: utc_now(),
            action: action.to_owned(),
            shot_id: shot_id.map(str::to_owned),
            detail,
        });
    }

    fn halt(&mut self, outcome: RunOutcome, reason: &str, detail: String, resumable: bool) {
        if self.stop.is_none() {
            self.stop = Some((
                outcome,
                RunStop {
                    reason: reason.to_owned(),
                    detail,
                    resumable,
                },
            ));
        }
    }

    fn canceled(&self) -> bool {
        self.client.control.is_canceled()
    }

    /// The per-shot budget, and the graces derived from it: a cancel is never given more room to
    /// settle than the plan gave the whole attempt.
    fn shot_budget(&self) -> Duration {
        Duration::from_secs(self.plan.limits.max_shot_seconds)
    }

    fn bounds(&self, shot_deadline: Instant) -> PollBounds {
        let shot_budget = self.shot_budget();
        PollBounds {
            shot_deadline,
            run_deadline: self.run_deadline,
            poll_interval: self.poll_interval,
            cancel_grace: CANCEL_GRACE.min(shot_budget),
            settle_grace: ASSET_SETTLE_GRACE.min(shot_budget),
        }
    }

    fn project_id(&self) -> Result<String, HarnessError> {
        self.record
            .project_id
            .clone()
            .ok_or_else(|| HarnessError::Transport("the run has no project yet".to_owned()))
    }

    // -----------------------------------------------------------------------------------------
    // Project and references — idempotent, so a replay adopts instead of re-creating
    // -----------------------------------------------------------------------------------------

    /// The project this run writes into. A run that created its own names it after the plan AND the
    /// run, which is what lets a controller that died right after `POST /projects` find that exact
    /// project again instead of creating a second one (sc-22711).
    fn project_name(&self) -> String {
        format!("{} ({})", self.plan.title, self.record.run_id)
    }

    async fn ensure_project(&mut self) -> Result<(), HarnessError> {
        if let Some(id) = self.record.project_id.clone() {
            let project = self
                .client
                .expect_ok("GET", &format!("/api/v1/projects/{id}"), None)
                .await?;
            self.record.project_path = project
                .get("path")
                .and_then(Value::as_str)
                .map(str::to_owned);
            self.persist()?;
            return Ok(());
        }
        let name = self.project_name();
        let existing = self
            .client
            .expect_ok("GET", "/api/v1/projects", None)
            .await?
            .as_array()
            .into_iter()
            .flatten()
            .find(|project| project.get("name").and_then(Value::as_str) == Some(name.as_str()))
            .cloned();
        let project = match existing {
            Some(project) => project,
            None => {
                self.client
                    .expect_ok("POST", "/api/v1/projects", Some(json!({ "name": name })))
                    .await?
            }
        };
        self.record.project_id = Some(
            project
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    HarnessError::Transport(format!("project response has no id: {project}"))
                })?
                .to_owned(),
        );
        self.record.project_path = project
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_owned);
        self.persist()?;
        Ok(())
    }

    /// Import every pack reference the record does not already name. An import the record missed
    /// (imported, then the controller died) is found by its `filmHarness` provenance rather than
    /// imported twice, so replay never doubles a project's reference assets.
    async fn ensure_references(&mut self) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        for existing in &self.record.references {
            // Only APPROVED roles resolve into a shot's conditioning slots, on a resume exactly as
            // on a first run.
            if existing.approved {
                self.role_assets
                    .insert(existing.role.clone(), existing.asset_id.clone());
            }
        }
        let imported: BTreeSet<String> = self
            .record
            .references
            .iter()
            .map(|reference| reference.role.clone())
            .collect();
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let references = self.pack.references.clone();
        // One listing for the whole pass, not one per reference: anything imported later in this
        // loop is this controller's own and is already recorded.
        let already_imported = if references
            .iter()
            .any(|reference| !imported.contains(&reference.role))
        {
            self.client
                .expect_ok(
                    "GET",
                    &format!("/api/v1/projects/{project_id}/assets"),
                    None,
                )
                .await?
                .as_array()
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        for reference in &references {
            if imported.contains(&reference.role) {
                continue;
            }
            let path = pack_dir.join(&reference.file);
            let bytes = std::fs::read(&path)?;
            let sha256 = sha256_hex(&bytes);
            let asset_id = match find_imported_reference(
                &already_imported,
                &self.record.run_id,
                &reference.role,
                &sha256,
            ) {
                Some(asset_id) => asset_id,
                None => {
                    self.import_reference(&project_id, reference, &path, &sha256)
                        .await?
                }
            };
            if reference.approved {
                self.role_assets
                    .insert(reference.role.clone(), asset_id.clone());
            }
            self.record.references.push(ReferenceAssetRecord {
                role: reference.role.clone(),
                kind: reference.kind.clone(),
                file: reference.file.clone(),
                sha256,
                asset_id,
                approved: reference.approved,
            });
            self.persist()?;
        }
        Ok(())
    }

    async fn import_reference(
        &self,
        project_id: &str,
        reference: &film_plan::ReferenceEntry,
        path: &Path,
        sha256: &str,
    ) -> Result<String, HarnessError> {
        let bytes = std::fs::read(path)?;
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
                "referencePackId": self.pack.id,
                "referencePackVersion": self.pack.version,
                "planId": self.plan.id,
                "planVersion": self.plan.version,
                "runId": self.record.run_id,
                "sourceFile": reference.file,
                "sha256": sha256,
            }
        });
        let (boundary, body) = encode_asset_upload(&filename, content_type, &bytes, &provenance);
        let route = format!("/api/v1/projects/{project_id}/assets");
        let response = self
            .transport
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
                    "asset import response has no id: {}",
                    response.body
                ))
            })?
            .to_owned();
        // An unapproved reference is imported (so the record can point at it and a human can review
        // it) but tagged distinctly, so a query for the conditioning-eligible references cannot
        // pick it up.
        let kind_tag = if reference.approved {
            REFERENCE_TAG
        } else {
            UNAPPROVED_REFERENCE_TAG
        };
        self.client
            .expect_ok(
                "PATCH",
                &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                Some(json!({
                    "tags": [
                        kind_tag,
                        format!("role:{}", reference.role),
                        format!("pack:{}", self.pack.id)
                    ]
                })),
            )
            .await?;
        Ok(asset_id)
    }

    // -----------------------------------------------------------------------------------------
    // Shots
    // -----------------------------------------------------------------------------------------

    /// Make sure the record holds one entry per plan shot, in plan order, without disturbing the
    /// entries a previous controller wrote.
    fn ensure_shot_records(&mut self) {
        let mut ordered: Vec<ShotRunRecord> = Vec::with_capacity(self.plan.shots.len());
        for shot in &self.plan.shots {
            if let Some(index) = self
                .record
                .shots
                .iter()
                .position(|existing| existing.shot_id == shot.id)
            {
                ordered.push(self.record.shots.remove(index));
                continue;
            }
            let (width, height) = film_plan::shot_resolution(&self.plan, shot, &self.entry)
                .expect("validated against the model");
            let assets = ConditioningAssets {
                first_frame_asset_id: shot
                    .conditioning
                    .first_frame_role
                    .as_ref()
                    .and_then(|role| self.role_assets.get(role).cloned()),
                last_frame_asset_id: shot
                    .conditioning
                    .last_frame_role
                    .as_ref()
                    .and_then(|role| self.role_assets.get(role).cloned()),
                reference_asset_ids: shot
                    .conditioning
                    .reference_roles
                    .iter()
                    .filter_map(|role| self.role_assets.get(role).cloned())
                    .collect(),
            };
            ordered.push(ShotRunRecord {
                shot_id: shot.id.clone(),
                outcome: ShotOutcome::NotSelected,
                intended: IntendedState {
                    mode: shot.conditioning.mode.clone(),
                    start_state: shot.start_state.clone(),
                    end_state: shot.end_state.clone(),
                    target_duration_seconds: shot.target_duration_seconds,
                    width,
                    height,
                    fps: self.fps,
                    dialogue: shot.dialogue.clone(),
                    sound: shot.sound.clone(),
                },
                conditioning_assets: assets,
                attempts: Vec::new(),
                selected_attempt: None,
                needs_review: Vec::new(),
            });
        }
        self.record.shots = ordered;
    }

    fn is_selected(&self, shot_id: &str) -> bool {
        self.record.selected_shot_ids.iter().any(|id| id == shot_id)
    }

    /// Reconcile and, where the limits still allow it, dispatch every selected shot.
    async fn work_shots(&mut self) -> Result<(), HarnessError> {
        self.ensure_shot_records();
        self.persist()?;
        for index in 0..self.plan.shots.len() {
            let shot = self.plan.shots[index].clone();
            if !self.is_selected(&shot.id) {
                if let Some(record) = self.record.shot_mut(&shot.id) {
                    record.outcome = ShotOutcome::NotSelected;
                }
                continue;
            }
            self.work_shot(&shot).await?;
            self.persist()?;
        }
        Ok(())
    }

    /// One shot: adopt whatever the API already holds for it, then dispatch what is still owed and
    /// still affordable. Never creates a job for an attempt that already has one.
    async fn work_shot(&mut self, shot: &film_plan::Shot) -> Result<(), HarnessError> {
        self.reconcile_shot(shot).await?;
        loop {
            let Some(index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
                return Ok(());
            };
            if self.record.shots[index].selected_attempt.is_some() {
                self.record.shots[index].outcome = ShotOutcome::Rendered;
                return Ok(());
            }
            if self.stop.is_some() {
                break;
            }
            // An attempt a previous controller recorded but never finished is continued, not
            // replaced: it already holds (or is about to hold) a job.
            let pending = self.record.shots[index]
                .attempts
                .iter()
                .position(|attempt| !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str()));
            let attempt_index = match pending {
                Some(pending) => pending,
                None => {
                    if self.record.shots[index].automatic_attempts()
                        >= self.plan.limits.max_attempts_per_shot
                    {
                        break;
                    }
                    if self.canceled() {
                        self.halt(
                            RunOutcome::Canceled,
                            "canceled",
                            format!("canceled before shot {} was dispatched", shot.id),
                            true,
                        );
                        break;
                    }
                    if Instant::now() >= self.run_deadline {
                        self.halt(
                            RunOutcome::StoppedRunBudget,
                            "run_budget",
                            format!(
                                "the run's {}s wall-clock budget was spent before shot {} could be \
                                 dispatched; raise limits.maxRunSeconds and start a new run",
                                self.plan.limits.max_run_seconds, shot.id
                            ),
                            false,
                        );
                        break;
                    }
                    let number = self.record.shots[index].next_attempt_number();
                    let key = idempotency_key(&self.record.run_id, &shot.id, number);
                    self.record.shots[index].attempts.push(AttemptRecord {
                        attempt: number,
                        idempotency_key: key,
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
                        rejection: None,
                        human_requested: false,
                    });
                    // Persisted BEFORE the job exists: this is the record a replay reads to know a
                    // job may already be out there under this key.
                    self.persist()?;
                    self.record.shots[index].attempts.len() - 1
                }
            };
            let progressed = self
                .work_attempt(shot, index, attempt_index, pending.is_none())
                .await?;
            if !progressed {
                break;
            }
        }
        let Some(index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
            return Ok(());
        };
        let record = &mut self.record.shots[index];
        record.outcome = match record
            .attempts
            .last()
            .map(|attempt| attempt.status.as_str())
        {
            _ if record.selected_attempt.is_some() => ShotOutcome::Rendered,
            None => ShotOutcome::NotDispatched,
            Some("timed_out") => ShotOutcome::TimedOut,
            Some("canceled" | "canceled_by_operator") => ShotOutcome::Canceled,
            Some(_) => ShotOutcome::Failed,
        };
        // A shot that spent its attempts does NOT stop the run: the remaining shots are independent
        // work and the human decides what to do about this one. Only a limit, a cancel or a refused
        // enqueue stops dispatch; `finish` turns a shot with no take into the run's outcome.
        Ok(())
    }

    /// Bring one recorded attempt to a terminal state. Returns whether the shot may keep going
    /// (another attempt is worth trying); `false` means a limit, a cancel or a refusal ended it.
    async fn work_attempt(
        &mut self,
        shot: &film_plan::Shot,
        shot_index: usize,
        attempt_index: usize,
        fresh: bool,
    ) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let key = self.record.shots[shot_index].attempts[attempt_index]
            .idempotency_key
            .clone();
        let attempt_number = self.record.shots[shot_index].attempts[attempt_index].attempt;

        // 1. The job. Adopt one already created under this key before creating anything.
        let mut job_id = self.record.shots[shot_index].attempts[attempt_index]
            .job_id
            .clone();
        if job_id.is_none() {
            job_id = self
                .client
                .find_job_by_idempotency_key(&project_id, &key)
                .await?;
        }
        if job_id.is_none() {
            let assets = self.record.shots[shot_index].conditioning_assets.clone();
            let intended = self.record.shots[shot_index].intended.clone();
            let body = video_job_body(&ShotDispatch {
                plan: &self.plan,
                shot,
                project_id: &project_id,
                run_id: &self.record.run_id,
                attempt: attempt_number,
                idempotency_key: &key,
                fps: self.fps,
                width: intended.width,
                height: intended.height,
                assets: &assets,
            });
            let response = self
                .client
                .json("POST", "/api/v1/video/jobs", Some(body))
                .await?;
            if !(200..300).contains(&response.status) {
                // A refused enqueue is deterministic: the same body would be refused again, so it
                // consumes the shot rather than every remaining attempt.
                let detail = format!(
                    "POST /api/v1/video/jobs -> {}: {}",
                    response.status,
                    api_detail(&response.body)
                );
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.status = "rejected".to_owned();
                attempt.error = Some(detail.clone());
                attempt.finished_at = Some(utc_now());
                self.persist()?;
                self.halt(
                    RunOutcome::Failed,
                    "enqueue_refused",
                    format!("shot {} could not be enqueued: {detail}", shot.id),
                    false,
                );
                return Ok(false);
            }
            job_id = Some(
                response
                    .body
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!(
                            "video job response has no id: {}",
                            response.body
                        ))
                    })?
                    .to_owned(),
            );
        }
        let job_id = job_id.expect("set on every branch above");
        {
            let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
            attempt.job_id = Some(job_id.clone());
            attempt.status = "running".to_owned();
        }
        self.persist()?;

        // 2. Wait for it, under whatever is left of this attempt's and the run's budgets.
        let attempt_started = Instant::now();
        // An attempt this controller just created has spent nothing; one it adopted from a
        // controller that died has been in flight since its recorded `startedAt`.
        let spent = if fresh {
            0.0
        } else {
            attempt_spent_seconds(&self.record.shots[shot_index].attempts[attempt_index])
        };
        let shot_deadline = attempt_started
            + Duration::from_secs_f64((self.plan.limits.max_shot_seconds as f64 - spent).max(0.0));
        let (view, poll_stop) = self
            .client
            .wait_for_job(&job_id, self.bounds(shot_deadline))
            .await?;

        // 3. Settle it into the record. The observed peak comes off the job's metrics block, which
        // the worker POSTs after its terminal progress — so read it once the attempt is terminal,
        // whatever ended it.
        let metrics = self.client.job_metrics(&job_id, self.poll_interval).await;
        let memory = memory_observation(metrics.as_ref(), &view, self.facts.host_memory_gb);
        let memory_exceeded = memory
            .gb
            .is_some_and(|observed| observed > self.plan.limits.max_memory_gb);
        // Whether the cancel the harness posted was actually honoured. A job still running after
        // the grace is a render in flight that nothing here can stop.
        let cancel_honoured = view.is_terminal();
        let settle_grace = ASSET_SETTLE_GRACE.min(self.shot_budget());
        let cancel_grace = CANCEL_GRACE.min(self.shot_budget());
        let take = (poll_stop == PollStop::Terminal && view.status == "completed")
            .then(|| take_from_result(&view.result, &self.plan.model.id, view.backend.as_deref()))
            .flatten();
        {
            let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
            attempt.finished_at = Some(utc_now());
            attempt.elapsed_seconds = spent + attempt_started.elapsed().as_secs_f64();
            attempt.peak_gpu_memory_pct = memory.pct;
            attempt.peak_memory_gb = memory.gb;
            attempt.peak_memory_source = memory.source.clone();
            attempt.status = match poll_stop {
                // Terminal either way — for `AssetsUnsettled` the JOB finished and it is the
                // API's asset handoff that stalled, so the attempt records the job's own status
                // and says what stalled in `error`. A resume re-reads a completed attempt that
                // carries no take, which gives the handoff another chance to have landed.
                PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
                PollStop::Operator => "canceled_by_operator".to_owned(),
                PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
            };
            attempt.error = match poll_stop {
                PollStop::Terminal if view.status == "completed" && take.is_none() => {
                    Some(format!(
                        "job {job_id} completed without an asset in its result: {}",
                        view.result
                    ))
                }
                PollStop::Terminal if view.status == "completed" => None,
                PollStop::Terminal => Some(view.failure_text()),
                PollStop::AssetsUnsettled => Some(format!(
                    "job {job_id} reached {} but its assets never settled within {:.0}s (the \
                     result still carries raw assetWrites): {}",
                    view.status,
                    settle_grace.as_secs_f64(),
                    view.result
                )),
                PollStop::Operator => Some(if cancel_honoured {
                    format!("canceled by operator (job {job_id} is {})", view.status)
                } else {
                    format!(
                        "canceled by operator; job {job_id} was still {} {:.0}s after the cancel \
                         was posted",
                        view.status,
                        cancel_grace.as_secs_f64()
                    )
                }),
                PollStop::ShotBudget if !cancel_honoured => Some(format!(
                    "attempt exceeded the per-shot budget of {}s and job {job_id} is still \
                     running {:.0}s after the cancel was posted — stopping dispatch rather than \
                     running a second render beside it",
                    self.plan.limits.max_shot_seconds,
                    cancel_grace.as_secs_f64()
                )),
                PollStop::ShotBudget => Some(format!(
                    "attempt exceeded the per-shot budget of {}s (last status {})",
                    self.plan.limits.max_shot_seconds, view.status
                )),
                PollStop::RunBudget if !cancel_honoured => Some(format!(
                    "run exceeded its budget of {}s and job {job_id} is still running {:.0}s \
                     after the cancel was posted — stopping dispatch rather than running a second \
                     render beside it",
                    self.plan.limits.max_run_seconds,
                    cancel_grace.as_secs_f64()
                )),
                PollStop::RunBudget => Some(format!(
                    "run exceeded its budget of {}s while this attempt was in flight (last status \
                     {})",
                    self.plan.limits.max_run_seconds, view.status
                )),
            };
            attempt.take = take.clone();
        }
        if let Some(take) = &take {
            // The first usable take of a shot becomes its selected take. Replay never reaches this
            // line for a shot that already has one, and nothing but `replace-take` moves it.
            if self.record.shots[shot_index].selected_attempt.is_none() {
                self.record.shots[shot_index].selected_attempt = Some(attempt_number);
            }
            if let Some(model) = self.record.model.as_mut() {
                if model.backend_observed.is_none() {
                    model.backend_observed = take.backend.clone();
                }
            }
        }
        self.persist()?;

        // 4. Decide whether the run may keep dispatching.
        match poll_stop {
            PollStop::AssetsUnsettled => {
                // Retrying would dispatch a fresh render against a server-side condition a retry
                // cannot fix.
                self.halt(
                    RunOutcome::Failed,
                    "assets_unsettled",
                    format!(
                        "shot {}'s job finished but the API never persisted its assets; the take \
                         cannot be adopted and a retry would not fix it",
                        shot.id
                    ),
                    false,
                );
                return Ok(false);
            }
            PollStop::Operator => {
                self.halt(
                    RunOutcome::Canceled,
                    "canceled",
                    if cancel_honoured {
                        format!("canceled while shot {} was in flight", shot.id)
                    } else {
                        format!(
                            "canceled while shot {} was in flight, and the job was still {} when \
                             the grace ran out — check the GPU before resuming",
                            shot.id, view.status
                        )
                    },
                    true,
                );
                return Ok(false);
            }
            PollStop::RunBudget => {
                self.halt(
                    RunOutcome::StoppedRunBudget,
                    "run_budget",
                    format!(
                        "the run's {}s wall-clock budget ran out while shot {} was in flight; \
                         raise limits.maxRunSeconds and start a new run",
                        self.plan.limits.max_run_seconds, shot.id
                    ),
                    false,
                );
                return Ok(false);
            }
            PollStop::ShotBudget if !cancel_honoured => {
                // The render is still on the GPU. Neither the next attempt nor the next shot may go
                // out beside it: the plan declared ONE memory budget, and two renders in flight is
                // exactly what it is there to prevent.
                self.halt(
                    RunOutcome::Failed,
                    "cancel_not_honoured",
                    format!(
                        "shot {}'s job was still {} after the cancel grace, so a render is in \
                         flight that nothing here can stop; no further dispatch",
                        shot.id, view.status
                    ),
                    false,
                );
                return Ok(false);
            }
            _ => {}
        }
        if memory_exceeded {
            self.halt(
                RunOutcome::StoppedMemoryLimit,
                "memory_limit",
                format!(
                    "shot {} peaked at {:.1} GB ({}), over the plan's {} GB budget; raise \
                     limits.maxMemoryGb or pick a cheaper tier and start a new run",
                    shot.id,
                    memory.gb.unwrap_or_default(),
                    memory.source.as_deref().unwrap_or("no source"),
                    self.plan.limits.max_memory_gb
                ),
                false,
            );
            return Ok(false);
        }
        Ok(true)
    }

    /// Adopt whatever the API holds for attempts a previous controller left unfinished.
    ///
    /// Every unfinished attempt is looked up (by its recorded job id, or by its idempotency key
    /// when the controller died before recording one) and read to its current state: a completed
    /// job is imported as a take, a failed one counts as a spent attempt, and a job still running
    /// is left for the dispatch loop to resume polling. Nothing here creates a job.
    async fn reconcile_shot(&mut self, shot: &film_plan::Shot) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        let Some(shot_index) = self.record.shots.iter().position(|s| s.shot_id == shot.id) else {
            return Ok(());
        };
        let unfinished: Vec<usize> = self.record.shots[shot_index]
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, attempt)| {
                !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str())
                    || (attempt.status == "completed" && attempt.take.is_none())
            })
            .map(|(index, _)| index)
            .collect();
        for attempt_index in unfinished {
            let attempt = &self.record.shots[shot_index].attempts[attempt_index];
            let key = attempt.idempotency_key.clone();
            let job_id = match attempt.job_id.clone() {
                Some(job_id) => Some(job_id),
                None => {
                    self.client
                        .find_job_by_idempotency_key(&project_id, &key)
                        .await?
                }
            };
            let Some(job_id) = job_id else {
                // The controller died before the job was created. The attempt stays pending and the
                // dispatch loop creates it under the same key.
                continue;
            };
            let view = self.client.get_job(&job_id).await?;
            let terminal = view.is_terminal() && view.is_settled();
            let take = (terminal && view.status == "completed")
                .then(|| {
                    take_from_result(&view.result, &self.plan.model.id, view.backend.as_deref())
                })
                .flatten();
            if terminal {
                let metrics = self.client.job_metrics(&job_id, self.poll_interval).await;
                let memory = memory_observation(metrics.as_ref(), &view, self.facts.host_memory_gb);
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.peak_gpu_memory_pct = memory.pct;
                attempt.peak_memory_gb = memory.gb;
                attempt.peak_memory_source = memory.source;
                attempt.job_id = Some(job_id.clone());
                attempt.status = view.status.clone();
                attempt.finished_at.get_or_insert_with(utc_now);
                attempt.error = match (&view.status[..], take.is_some()) {
                    ("completed", true) => None,
                    ("completed", false) => Some(format!(
                        "job {job_id} completed without an asset in its result: {}",
                        view.result
                    )),
                    _ => Some(view.failure_text()),
                };
                attempt.take = take.clone();
            } else {
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.job_id = Some(job_id.clone());
                attempt.status = "running".to_owned();
            }
            if take.is_some() && self.record.shots[shot_index].selected_attempt.is_none() {
                let number = self.record.shots[shot_index].attempts[attempt_index].attempt;
                self.record.shots[shot_index].selected_attempt = Some(number);
            }
            if let Some(take) = &take {
                if let Some(model) = self.record.model.as_mut() {
                    if model.backend_observed.is_none() {
                        model.backend_observed = take.backend.clone();
                    }
                }
            }
            self.persist()?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // Timeline and export
    // -----------------------------------------------------------------------------------------

    /// The selected take of every selected shot, in plan order — what the timeline is built from.
    fn selected_takes(&self) -> Vec<SelectedTake> {
        self.record
            .shots
            .iter()
            .filter(|shot| self.is_selected(&shot.shot_id))
            .filter_map(|shot| {
                let attempt = shot.selected()?;
                Some(SelectedTake {
                    shot_id: shot.shot_id.clone(),
                    take: attempt.take.clone()?,
                    width: shot.intended.width,
                    height: shot.intended.height,
                    job_id: attempt.job_id.clone(),
                })
            })
            .collect()
    }

    /// The tallest selected take, which fixes the export ladder rung the render uses.
    fn tallest_selected(&self) -> u32 {
        self.selected_takes()
            .iter()
            .map(|selected| selected.height)
            .max()
            .unwrap_or(0)
    }

    /// Create (or update) the run's timeline from the selected takes. Pure API writes, no job:
    /// re-running it after a take changed rewrites one item and leaves the rest alone.
    async fn assemble_timeline(&mut self) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let takes = self.selected_takes();
        if takes.is_empty() {
            return Ok(false);
        }
        let (first_width, first_height) = (takes[0].width, takes[0].height);
        let aspect_ratio = aspect_ratio_for(first_width, first_height);
        let existing_id = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.timeline_id.clone());
        let timeline_name = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.name.clone())
            .unwrap_or_else(|| format!("{} ({})", self.plan.title, self.record.run_id));
        let mut timeline = match &existing_id {
            Some(id) => {
                self.client
                    .expect_ok(
                        "GET",
                        &format!("/api/v1/projects/{project_id}/timelines/{id}"),
                        None,
                    )
                    .await?
            }
            None => {
                self.client
                    .expect_ok(
                        "POST",
                        &format!("/api/v1/projects/{project_id}/timelines"),
                        Some(json!({
                            "name": timeline_name,
                            "aspectRatio": aspect_ratio,
                            "fps": self.fps
                        })),
                    )
                    .await?
            }
        };
        let timeline_id = timeline
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessError::Transport(format!("timeline response has no id: {timeline}"))
            })?
            .to_owned();
        // The main video track, by id or — if the project store's default track ids ever change —
        // by kind. Writing the items nowhere and saving an EMPTY timeline while the record still
        // listed every shot is the one outcome that must not happen.
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
        let mut item_records = Vec::new();
        let mut cursor = 0.0_f64;
        for SelectedTake {
            shot_id,
            take,
            job_id,
            ..
        } in &takes
        {
            let shot = self
                .plan
                .shots
                .iter()
                .find(|shot| &shot.id == shot_id)
                .expect("selected takes come from the plan");
            let length = take
                .encoded_duration_seconds
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(shot.target_duration_seconds);
            let item_id = format!(
                "item_{}_{}",
                shot.id.to_ascii_lowercase(),
                &self.record.run_id[4..12]
            );
            let end = cursor + length;
            items.push(json!({
                "id": item_id,
                "trackId": track_id,
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
                    "jobId": job_id,
                    "note": format!("film-harness {} shot {}", self.record.run_id, shot.id),
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
        }
        timeline["tracks"][track_index]["items"] = Value::Array(items);
        let saved = self
            .client
            .expect_ok(
                "PUT",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
                Some(json!({ "timeline": timeline })),
            )
            .await?;
        // Read the items back off the SAVED document rather than trusting the harness's own intent:
        // the record then cannot describe a timeline that was never persisted.
        let persisted = persisted_timeline_items(&saved, &item_records).ok_or_else(|| {
            HarnessError::Transport(format!(
                "saved timeline {timeline_id} does not hold the {} items the run assembled: {saved}",
                item_records.len()
            ))
        })?;
        self.record.timeline = Some(TimelineRecord {
            timeline_id,
            name: timeline_name,
            aspect_ratio: aspect_ratio.to_owned(),
            source_aspect_ratio: Some(reduced_aspect_ratio(first_width, first_height)),
            source_width: Some(first_width),
            source_height: Some(first_height),
            fps: self.fps,
            items: persisted,
        });
        self.persist()?;
        Ok(true)
    }

    /// Render the timeline through the `timeline_export` job, adopting an export this run already
    /// dispatched rather than starting a second render.
    async fn run_export(&mut self) -> Result<bool, HarnessError> {
        let project_id = self.project_id()?;
        let Some(timeline) = self.record.timeline.clone() else {
            return Ok(false);
        };
        if let Some(export) = &self.record.export {
            if export.status == "completed" && export.asset_id.is_some() && !export.stale {
                return Ok(true);
            }
        }
        if self.canceled() {
            self.halt(
                RunOutcome::Canceled,
                "canceled",
                "canceled before the timeline export was dispatched".to_owned(),
                true,
            );
            return Ok(false);
        }
        // An export that is stale or finished badly is SUPERSEDED: the next one is a different job,
        // and saying so is what lets a resume tell the new export apart from the old one on the same
        // timeline. An export merely recorded as `running` is not superseded — it is the very job
        // this pass is about to adopt again.
        let superseded = self
            .record
            .export
            .as_ref()
            .filter(|export| {
                export.stale
                    || (export.status != "running"
                        && !(export.status == "completed" && export.asset_id.is_some()))
            })
            .map(|export| export.job_id.clone());
        if self.record.export_pending.is_none() {
            self.record.export_pending = Some(ExportPending {
                requested_at: utc_now(),
                supersedes: superseded,
            });
            self.record.export = None;
            // Persisted BEFORE the export job exists, for the same reason a shot attempt is.
            self.persist()?;
        }
        let supersedes = self
            .record
            .export_pending
            .as_ref()
            .and_then(|pending| pending.supersedes.clone());
        let existing = self
            .client
            .find_export_job(&project_id, &timeline.timeline_id, supersedes.as_deref())
            .await?;
        let export_job_id = match existing {
            Some(job_id) => job_id,
            None => {
                let export_job = self
                    .client
                    .expect_ok(
                        "POST",
                        &format!(
                            "/api/v1/projects/{project_id}/timelines/{}/exports",
                            timeline.timeline_id
                        ),
                        Some(json!({
                            "resolution": export_resolution_for(self.tallest_selected()),
                            "fps": self.fps,
                            "requestedGpu": "auto",
                        })),
                    )
                    .await?;
                export_job
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!("export response has no id: {export_job}"))
                    })?
                    .to_owned()
            }
        };
        self.record.export = Some(ExportRecord {
            job_id: export_job_id.clone(),
            status: "running".to_owned(),
            stale: false,
            asset_id: None,
            render_path: None,
            error: None,
        });
        self.record.export_pending = None;
        self.persist()?;

        let export_started = Instant::now();
        let settle_grace = ASSET_SETTLE_GRACE.min(self.shot_budget());
        let (view, poll_stop) = self
            .client
            .wait_for_job(
                &export_job_id,
                self.bounds(export_started + self.shot_budget()),
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
            PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
            PollStop::Operator => "canceled_by_operator".to_owned(),
            PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        };
        let export_ok = status == "completed" && asset_id.is_some();
        self.record.export = Some(ExportRecord {
            job_id: export_job_id,
            status,
            stale: false,
            asset_id,
            render_path,
            error: (!export_ok).then(|| match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the export job reached {} but its assets never settled within {:.0}s",
                    view.status,
                    settle_grace.as_secs_f64()
                ),
                PollStop::Operator => "canceled by operator during the export".to_owned(),
                PollStop::ShotBudget => format!(
                    "export exceeded the per-job budget of {}s",
                    self.plan.limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "run exceeded its budget of {}s during the export",
                    self.plan.limits.max_run_seconds
                ),
            }),
        });
        self.persist()?;
        match poll_stop {
            PollStop::Operator => self.halt(
                RunOutcome::Canceled,
                "canceled",
                "canceled while the timeline export was in flight".to_owned(),
                true,
            ),
            PollStop::RunBudget => self.halt(
                RunOutcome::StoppedRunBudget,
                "run_budget",
                format!(
                    "the run's {}s wall-clock budget ran out during the export; raise \
                     limits.maxRunSeconds and start a new run",
                    self.plan.limits.max_run_seconds
                ),
                false,
            ),
            _ => {}
        }
        Ok(export_ok)
    }

    /// Assemble and export, unless a stop already landed. A canceled run stops here with its takes
    /// intact: the assets are already the project's, and a resume finishes the assembly.
    async fn assemble_and_export(&mut self) -> Result<bool, HarnessError> {
        if !self.export {
            return Ok(true);
        }
        if matches!(
            self.stop.as_ref().map(|(outcome, _)| *outcome),
            Some(RunOutcome::Canceled | RunOutcome::StoppedRunBudget)
        ) {
            return Ok(false);
        }
        if !self.assemble_timeline().await? {
            return Ok(false);
        }
        self.run_export().await
    }

    /// Close the run: classify the outcome, stamp the record `finished` and write it one last time.
    fn finish(&mut self, export_ok: bool) -> Result<(), HarnessError> {
        let all_rendered = self
            .record
            .shots
            .iter()
            .filter(|shot| self.is_selected(&shot.shot_id))
            .all(|shot| shot.selected_attempt.is_some());
        let (outcome, stop) = match self.stop.take() {
            Some((outcome, stop)) => (outcome, Some(stop)),
            None if all_rendered && export_ok => (RunOutcome::Completed, None),
            None if all_rendered => (
                RunOutcome::Failed,
                Some(RunStop {
                    reason: "export_failed".to_owned(),
                    detail: "every selected shot rendered but the timeline export did not \
                             complete; `film-harness resume` retries the export"
                        .to_owned(),
                    resumable: true,
                }),
            ),
            None => (
                RunOutcome::Failed,
                Some(RunStop {
                    reason: "attempts_exhausted".to_owned(),
                    detail: "a selected shot has no usable take and no attempts left; \
                             `film-harness replace-take --shot <id>` authorises one more"
                        .to_owned(),
                    resumable: false,
                }),
            ),
        };
        self.record.outcome = outcome;
        self.record.stop = stop;
        self.record.state = RunState::Finished;
        self.record.finished_at = Some(utc_now());
        self.persist()
    }

    /// Project -> references -> shots -> timeline -> export -> close.
    async fn drive(mut self) -> Result<RunRecord, HarnessError> {
        let outcome = self.drive_inner().await;
        match outcome {
            Ok(export_ok) => {
                self.finish(export_ok)?;
                Ok(self.record)
            }
            // A transport/API failure partway through is exactly when the record matters most: by
            // then the run may have created a project, imported assets and dispatched jobs. It
            // stays `running`, because that is what it is — a resume reconciles it.
            Err(error) => {
                self.record.diagnostics.push(PlanDiagnostic::plan(
                    "run",
                    format!("the run stopped on an error: {error}"),
                ));
                // Best effort: an io failure while writing the record must not replace the failure
                // that caused it with a less informative one.
                let _ = self.persist();
                Err(error)
            }
        }
    }

    async fn drive_inner(&mut self) -> Result<bool, HarnessError> {
        self.ensure_project().await?;
        self.ensure_references().await?;
        self.work_shots().await?;
        self.assemble_and_export().await
    }

    /// Flag every shot that declared a dependency on `shot_id` AND already has a take of its own.
    /// A shot that has not rendered yet needs no flag: it will be rendered against the current
    /// state. Direct dependents only — a flagged shot's own take did not change, so the signal does
    /// not cascade on its own.
    fn flag_dependents(&mut self, shot_id: &str, reason: &str) {
        let dependents: Vec<(String, String, String)> =
            film_plan::direct_dependents(&self.plan, shot_id)
                .into_iter()
                .map(|(shot, edge)| (shot.id.clone(), edge.kind.clone(), edge.note.clone()))
                .collect();
        let raised_at = utc_now();
        for (dependent_id, kind, note) in dependents {
            let Some(record) = self.record.shot_mut(&dependent_id) else {
                continue;
            };
            if record.selected_attempt.is_none() {
                continue;
            }
            let detail = if note.trim().is_empty() {
                String::new()
            } else {
                format!(" ({note})")
            };
            record.needs_review.push(ReviewFlag {
                raised_at: raised_at.clone(),
                source_shot_id: shot_id.to_owned(),
                dependency: kind.clone(),
                reason: format!(
                    "shot {shot_id}'s selected take was replaced ({reason}); this shot's {kind} \
                     depends on it{detail} — review it and replace it too if it no longer matches"
                ),
            });
        }
    }
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
/// Every path past document validation writes `run.json` — sc-22711 writes it at every state
/// transition, not only at the end, so a controller killed anywhere leaves a record `resume` can
/// reconcile against the API.
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
    let prepared = match prepare(&client, &plan, options.export, options.require_installed).await? {
        Ok(prepared) => prepared,
        Err(findings) => {
            let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
            record.state = RunState::Finished;
            record.outcome = RunOutcome::Rejected;
            record.stop = Some(RunStop {
                reason: "rejected".to_owned(),
                detail: "the plan was refused against this API's catalog and host".to_owned(),
                resumable: false,
            });
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
    };
    let lane = prepared.facts.lane();
    let mut record = base_record(&run_id, &plan, &pack, options, &plan_bytes, &pack_bytes);
    record.model = Some(ModelRecord {
        id: plan.model.id.clone(),
        tier_requested: plan.model.tier.clone(),
        fps: prepared.fps,
        lane: lane.manifest_key().to_owned(),
        backend_observed: None,
        weights: primary_weights(&prepared.entry, plan.model.tier.as_deref()),
        hardware: HardwareRecord {
            platform: prepared.facts.platform_or_local().to_owned(),
            // The host-capabilities route reports no arch, so this process's arch is the truth only
            // when the API is on this platform; otherwise claiming one would be a fabrication.
            arch: if prepared.facts.platform_or_local() == std::env::consts::OS {
                std::env::consts::ARCH.to_owned()
            } else {
                "unknown".to_owned()
            },
            host_memory_gb: prepared.facts.host_memory_gb,
            gpu_name: prepared.facts.video_gpu_name.clone(),
            worker_id: prepared.facts.video_worker_id.clone(),
        },
    });
    let run_deadline = started + Duration::from_secs(plan.limits.max_run_seconds);
    let mut session = Session {
        client,
        transport,
        plan_path: options.plan_path.clone(),
        pack_path: options.reference_pack_path.clone(),
        out_dir: options.out_dir.clone(),
        poll_interval: options.poll_interval,
        export: options.export,
        plan,
        pack,
        entry: prepared.entry,
        facts: prepared.facts,
        fps: prepared.fps,
        record,
        started,
        prior_elapsed: 0.0,
        run_deadline,
        role_assets: BTreeMap::new(),
        stop: None,
    };
    // The record exists before the first API write, so even a controller killed during project
    // creation leaves something `resume` can reconcile.
    session.persist()?;
    session.drive().await
}

/// What `resume` and `replace_take` both need: the record, its documents re-read and re-hashed, and
/// the catalog/host judged again (either can have changed while the run was not held).
struct Continued {
    record: RunRecord,
    plan: ProductionPlan,
    pack: ReferencePack,
    plan_path: PathBuf,
    pack_path: PathBuf,
    prepared: Prepared,
}

/// Read `<out_dir>/run.json`, re-read the documents it was started from, and refuse unless they
/// still hash to what the run recorded.
///
/// The hash check is the load-bearing part: resuming against an edited plan would silently render
/// shots the record's takes were never meant to sit beside. An edited plan is a new run, and the
/// refusal says so.
async fn continue_run(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<Continued, HarnessError> {
    let record = read_run_record(&options.out_dir)?;
    if record.schema_version != RUN_RECORD_SCHEMA_VERSION {
        return Err(HarnessError::Refused(format!(
            "run record schema version {} (this build reads {RUN_RECORD_SCHEMA_VERSION})",
            record.schema_version
        )));
    }
    let (plan_path, plan_bytes) = read_source(
        Path::new(&record.plan.path),
        &options.out_dir.join("plan.json"),
        "plan",
    )?;
    let (pack_path, pack_bytes) = read_source(
        Path::new(&record.reference_pack.path),
        &options.out_dir.join("references.json"),
        "reference pack",
    )?;
    for (label, bytes, recorded) in [
        ("plan", &plan_bytes, &record.plan.sha256),
        ("reference pack", &pack_bytes, &record.reference_pack.sha256),
    ] {
        let actual = sha256_hex(bytes);
        if &actual != recorded {
            return Err(HarnessError::Refused(format!(
                "the {label} changed since run {} started (recorded {recorded}, found {actual}); a \
                 changed {label} is a new run, not a resume",
                record.run_id
            )));
        }
    }
    let plan = film_plan::parse_plan(std::str::from_utf8(&plan_bytes).unwrap_or_default())
        .map_err(|error| HarnessError::Refused(format!("plan: {error}")))?;
    let pack =
        film_plan::parse_reference_pack(std::str::from_utf8(&pack_bytes).unwrap_or_default())
            .map_err(|error| HarnessError::Refused(format!("reference pack: {error}")))?;
    let pack_dir = pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let findings = film_plan::validate_all(&plan, &pack, Some(&pack_dir), None);
    if !findings.is_empty() {
        return Err(HarnessError::Validation(findings));
    }
    let client = Client {
        transport,
        control: &options.control,
    };
    let prepared = prepare(&client, &plan, options.export, options.require_installed)
        .await?
        .map_err(HarnessError::Validation)?;
    Ok(Continued {
        record,
        plan,
        pack,
        plan_path,
        pack_path,
        prepared,
    })
}

/// The source document at `original`, falling back to the copy the run kept beside its record.
fn read_source(
    original: &Path,
    fallback: &Path,
    label: &str,
) -> Result<(PathBuf, Vec<u8>), HarnessError> {
    if let Ok(bytes) = std::fs::read(original) {
        return Ok((original.to_path_buf(), bytes));
    }
    let bytes = std::fs::read(fallback).map_err(|error| {
        HarnessError::Refused(format!(
            "cannot read the {label} from {} or {}: {error}",
            original.display(),
            fallback.display()
        ))
    })?;
    Ok((fallback.to_path_buf(), bytes))
}

fn session_from<'a>(
    transport: &'a dyn ApiTransport,
    options: &'a ResumeOptions,
    continued: Continued,
    started: Instant,
    run_deadline: Instant,
) -> Session<'a> {
    Session {
        client: Client {
            transport,
            control: &options.control,
        },
        transport,
        plan_path: continued.plan_path,
        pack_path: continued.pack_path,
        out_dir: options.out_dir.clone(),
        poll_interval: options.poll_interval,
        export: options.export,
        plan: continued.plan,
        pack: continued.pack,
        entry: continued.prepared.entry,
        facts: continued.prepared.facts,
        fps: continued.prepared.fps,
        prior_elapsed: continued.record.elapsed_seconds,
        record: continued.record,
        started,
        run_deadline,
        role_assets: BTreeMap::new(),
        stop: None,
    }
}

/// Pick a run back up where a crash, a cancel or a failed export left it.
///
/// Every completed take is reused, every job the record names is read back from the API and adopted
/// at whatever state it actually reached, and the run continues under whatever is LEFT of the
/// plan's wall-clock budget and per-shot attempt cap — so a restart loop cannot turn a bounded run
/// into an unbounded one. The selected take of every shot is left exactly as it was.
pub async fn resume(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let continued = continue_run(transport, options).await?;
    if !continued.record.is_resumable() {
        let detail = continued
            .record
            .stop
            .as_ref()
            .map(|stop| format!("{}: {}", stop.reason, stop.detail))
            .unwrap_or_else(|| format!("outcome {:?}", continued.record.outcome));
        return Err(HarnessError::Refused(format!(
            "run {} is not resumable ({detail})",
            continued.record.run_id
        )));
    }
    // A cancel that stopped the previous controller must not cancel this one on its first poll.
    clear_cancel_request(&options.out_dir)?;
    let remaining =
        (continued.plan.limits.max_run_seconds as f64 - continued.record.elapsed_seconds).max(0.0);
    let run_deadline = started + Duration::from_secs_f64(remaining);
    let mut session = session_from(transport, options, continued, started, run_deadline);
    session.record.state = RunState::Running;
    session.record.stop = None;
    session.record.finished_at = None;
    session.note_decision(
        "resume",
        None,
        format!(
            "resumed with {remaining:.0}s of the plan's {}s budget left",
            session.plan.limits.max_run_seconds
        ),
    );
    session.persist()?;
    if remaining <= 0.0 {
        session.halt(
            RunOutcome::StoppedRunBudget,
            "run_budget",
            format!(
                "the run already spent its {}s wall-clock budget; raise limits.maxRunSeconds and \
                 start a new run",
                session.plan.limits.max_run_seconds
            ),
            false,
        );
        session.finish(false)?;
        return Ok(session.record);
    }
    session.drive().await
}

/// Reject the take a shot is currently carrying and render exactly one more, for that shot alone.
///
/// This is a human decision, not a retry: it authorises ONE bounded attempt (it does not loop, and
/// it does not spend or respect the plan's automatic attempt cap), it touches no other shot's
/// takes, jobs or assets, and it leaves the rejected take in the record beside the new one. When
/// the replacement lands, every shot that DECLARED a dependency on this one is flagged
/// `needs_review` and the export is marked stale — flagged, never regenerated.
pub async fn replace_take(
    transport: &dyn ApiTransport,
    options: &ResumeOptions,
    shot_id: &str,
    reason: &str,
) -> Result<RunRecord, HarnessError> {
    let started = Instant::now();
    let continued = continue_run(transport, options).await?;
    if continued.record.project_id.is_none() {
        return Err(HarnessError::Refused(format!(
            "run {} never created a project, so it has no take to replace",
            continued.record.run_id
        )));
    }
    let Some(shot) = continued
        .plan
        .shots
        .iter()
        .find(|shot| shot.id == shot_id)
        .cloned()
    else {
        return Err(HarnessError::Refused(format!(
            "{shot_id:?} is not a shot in plan {:?}",
            continued.plan.id
        )));
    };
    if !continued
        .record
        .selected_shot_ids
        .iter()
        .any(|id| id == shot_id)
    {
        return Err(HarnessError::Refused(format!(
            "shot {shot_id} is not in run {}'s selection",
            continued.record.run_id
        )));
    }
    clear_cancel_request(&options.out_dir)?;
    // One replacement is bounded by the per-shot budget, not by whatever is left of a run budget a
    // previous controller may already have spent: the human just authorised this one attempt.
    let run_deadline = started + Duration::from_secs(continued.plan.limits.max_shot_seconds);
    let mut session = session_from(transport, options, continued, started, run_deadline);
    session.record.state = RunState::Running;
    session.record.stop = None;
    session.record.finished_at = None;

    let Some(index) = session
        .record
        .shots
        .iter()
        .position(|record| record.shot_id == shot_id)
    else {
        return Err(HarnessError::Refused(format!(
            "run {} has no record for shot {shot_id}",
            session.record.run_id
        )));
    };
    // An attempt that has not settled is one this shot is still owed — from a controller that died,
    // or from a `run` that is still going. Starting a second one here would leave the first
    // orphaned, so say so and point at the command that settles it.
    if let Some(pending) = session.record.shots[index]
        .attempts
        .iter()
        .find(|attempt| !TERMINAL_ATTEMPT_STATUSES.contains(&attempt.status.as_str()))
    {
        return Err(HarnessError::Refused(format!(
            "shot {shot_id} still has attempt {} in flight (job {}); run `film-harness resume \
             --out {}` to settle it before replacing the take",
            pending.attempt,
            pending.job_id.as_deref().unwrap_or("not yet created"),
            options.out_dir.display()
        )));
    }
    // Reject whatever this shot is carrying: the selected take, or the last take it produced.
    let rejected = session.record.shots[index].selected_attempt.or_else(|| {
        session.record.shots[index]
            .attempts
            .iter()
            .rev()
            .find(|attempt| attempt.has_live_take())
            .map(|attempt| attempt.attempt)
    });
    if let Some(number) = rejected {
        if let Some(attempt) = session.record.shots[index]
            .attempts
            .iter_mut()
            .find(|attempt| attempt.attempt == number)
        {
            attempt.rejection = Some(TakeRejection {
                at: utc_now(),
                reason: reason.to_owned(),
            });
        }
    }
    session.record.shots[index].selected_attempt = None;
    let previous_asset = rejected
        .and_then(|number| {
            session.record.shots[index]
                .attempts
                .iter()
                .find(|attempt| attempt.attempt == number)
        })
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone());
    session.note_decision(
        "replace_take",
        Some(shot_id),
        match (&rejected, &previous_asset) {
            (Some(number), Some(asset)) => {
                format!("rejected attempt {number} (asset {asset}): {reason}")
            }
            (Some(number), None) => format!("rejected attempt {number}: {reason}"),
            _ => format!("no take to reject; rendering one: {reason}"),
        },
    );
    session.persist()?;

    // Exactly one attempt, marked as the human decision it is so it never spends the plan's cap.
    let number = session.record.shots[index].next_attempt_number();
    let key = idempotency_key(&session.record.run_id, shot_id, number);
    session.record.shots[index].attempts.push(AttemptRecord {
        attempt: number,
        idempotency_key: key,
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
        rejection: None,
        human_requested: true,
    });
    session.persist()?;
    let attempt_index = session.record.shots[index].attempts.len() - 1;
    session
        .work_attempt(&shot, index, attempt_index, true)
        .await?;

    let replaced = session.record.shots[index].selected_attempt.is_some();
    session.record.shots[index].outcome = if replaced {
        ShotOutcome::Rendered
    } else {
        ShotOutcome::Failed
    };
    if replaced {
        session.flag_dependents(shot_id, reason);
        if let Some(export) = session.record.export.as_mut() {
            export.stale = true;
        }
        session.persist()?;
        // Rewriting the timeline is a PUT, not a job: every other shot's item keeps its asset.
        session.assemble_timeline().await?;
    } else if session.stop.is_none() {
        session.halt(
            RunOutcome::Failed,
            "replacement_failed",
            format!(
                "the replacement attempt for shot {shot_id} produced no take; the rejected take is \
                 still recorded and `film-harness replace-take --shot {shot_id}` authorises another"
            ),
            false,
        );
    }
    // Not re-exporting is a deliberate choice, not a failure: the replacement succeeded, the
    // existing MP4 is marked stale, and the run is as finished as this invocation was asked to make
    // it. Only a failed replacement leaves the run failed (with the stop set above).
    let export_ok = if replaced && session.export {
        session.run_export().await?
    } else {
        replaced
    };
    session.finish(export_ok)?;
    Ok(session.record)
}

/// The timeline items as the SAVE persisted them, in the order the run assembled them, or `None`
/// when the saved document does not hold exactly the items that were sent (which would mean the
/// record and the project disagree about what the sequence is).
fn persisted_timeline_items(
    saved: &Value,
    intended: &[TimelineItemRecord],
) -> Option<Vec<TimelineItemRecord>> {
    let items: Vec<&Value> = saved
        .get("tracks")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .collect();
    let mut persisted = Vec::with_capacity(intended.len());
    for record in intended {
        let item = items
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(record.item_id.as_str()))?;
        persisted.push(TimelineItemRecord {
            shot_id: record.shot_id.clone(),
            item_id: record.item_id.clone(),
            asset_id: item.get("assetId").and_then(Value::as_str)?.to_owned(),
            timeline_start: item.get("timelineStart").and_then(Value::as_f64)?,
            timeline_end: item.get("timelineEnd").and_then(Value::as_f64)?,
        });
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
        // A record is born `running`: it exists before the first API write, and only `finish`
        // decides what it became (sc-22711).
        state: RunState::Running,
        outcome: RunOutcome::Failed,
        stop: None,
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
        export_pending: None,
        diagnostics: Vec::new(),
        decisions: Vec::new(),
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
        state: RunState::Finished,
        outcome: RunOutcome::Rejected,
        stop: Some(RunStop {
            reason: "rejected".to_owned(),
            detail: "the plan was refused before anything was created; fix the findings and start \
                     a new run"
                .to_owned(),
            resumable: false,
        }),
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
        export_pending: None,
        diagnostics: findings,
        decisions: Vec::new(),
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
        let intended = vec![TimelineItemRecord {
            shot_id: "SH010".to_owned(),
            item_id: "item_sh010_abcd1234".to_owned(),
            asset_id: "asset_1".to_owned(),
            timeline_start: 0.0,
            timeline_end: 5.0,
        }];
        let saved = json!({ "tracks": [{ "id": "track_main", "items": [{
            "id": "item_sh010_abcd1234", "assetId": "asset_1",
            "timelineStart": 0.0, "timelineEnd": 5.1667
        }] }, { "id": "track_audio", "items": [] }] });
        let persisted = persisted_timeline_items(&saved, &intended).expect("items read back");
        assert_eq!(persisted[0].shot_id, "SH010");
        assert!(
            (persisted[0].timeline_end - 5.1667).abs() < 1e-9,
            "the SAVED value wins over the intended one"
        );
        // A timeline the save dropped the items from cannot be recorded as if it held them.
        let empty = json!({ "tracks": [{ "id": "track_main", "items": [] }] });
        assert!(persisted_timeline_items(&empty, &intended).is_none());
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
