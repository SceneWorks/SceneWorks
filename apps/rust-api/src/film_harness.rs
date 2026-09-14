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
//!
//! **One controller per run directory, and nothing locks it.** `run`, [`resume`] and
//! [`replace_take`] each rewrite `run.json` as they go; two of them held against the same directory
//! at the same time interleave their writes and the last one wins. The idempotency keys make a
//! SEQUENTIAL replay safe — they are not a lock between two live controllers. `run` refuses
//! outright when the directory already holds a run record, and `replace_take` refuses while the
//! shot still has an unsettled attempt; neither is a substitute for not starting two at once.

pub mod review;

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
    self, AttemptRecord, ConditioningAssets, ExportPending, ExportRecord, GeneratedAudio,
    HardwareRecord, IntendedState, ModelLane, ModelRecord, PlanDiagnostic, ProductionDecision,
    ProductionPlan, ReferenceAssetRecord, ReferencePack, ReviewFlag, RunOutcome, RunRecord,
    RunState, RunStop, ShotOutcome, ShotRunRecord, SoundBed, SoundBus, SourceDocument, TakeRecord,
    TakeRejection, TimelineEditRecord, TimelineItemRecord, TimelineRecord, TimelineTrackRecord,
    RUN_RECORD_SCHEMA_VERSION,
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
///
/// A directory with no run record in it is REFUSED rather than created: `cancel --out /typo/path`
/// otherwise printed "cancel requested" and exited 0 while the real 45-minute render kept going,
/// which is the one thing a cancel must never do.
pub fn request_cancel(run_dir: &Path) -> Result<PathBuf, HarnessError> {
    let record = run_dir.join(RUN_RECORD_FILE);
    if !record.try_exists().unwrap_or(false) {
        return Err(HarnessError::Refused(format!(
            "no run record in {} — nothing there to cancel (a run writes {RUN_RECORD_FILE} before \
             its first API call; check the --out path)",
            run_dir.display()
        )));
    }
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

/// Merge several job listings into one, newest first, keeping one entry per job id. The pages
/// overlap by construction (an unfiltered page plus one page per status), so the dedupe is the
/// point rather than a precaution.
fn merge_job_pages(pages: Vec<Vec<Value>>) -> Vec<Value> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut merged: Vec<Value> = Vec::new();
    for job in pages.into_iter().flatten() {
        let Some(id) = job.get("id").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        if seen.insert(id) {
            merged.push(job);
        }
    }
    merged.sort_by(|left, right| {
        let created = |job: &Value| {
            job.get("createdAt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        created(right).cmp(&created(left))
    });
    merged
}

/// The newest `timeline_export` job for `timeline_id` that is neither in `exclude` nor older than
/// `not_before`. Pure, so the adoption rule the export's correctness rests on is provable without a
/// server: see [`Client::find_export_job`] for what each guard is for.
///
/// A job whose `createdAt` cannot be read is NOT adopted: the harness would rather dispatch a
/// second export than record an unrelated job's asset as this run's delivered MP4.
fn newest_export_job(
    jobs: &[Value],
    timeline_id: &str,
    exclude: &[String],
    not_before: &str,
) -> Option<String> {
    let floor = parse_utc_seconds(not_before);
    let mut candidates: Vec<(&str, i64)> = jobs
        .iter()
        .filter(|job| job.get("type").and_then(Value::as_str) == Some("timeline_export"))
        .filter(|job| {
            job.pointer("/payload/timelineId").and_then(Value::as_str) == Some(timeline_id)
        })
        .filter_map(|job| {
            let id = job.get("id")?.as_str()?;
            let created = parse_utc_seconds(job.get("createdAt")?.as_str()?)?;
            Some((id, created))
        })
        .filter(|(id, _)| !exclude.iter().any(|excluded| excluded == id))
        .filter(|(_, created)| floor.is_none_or(|floor| *created >= floor))
        .collect();
    candidates.sort_by_key(|(_, created)| *created);
    candidates.last().map(|(id, _)| (*id).to_owned())
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

/// The `peakMemoryBytes` a job's metrics block reports, against a bare transport, retried for the
/// same [`METRICS_GRACE`] a render's metrics get. `None` when no block (or no peak) was posted.
pub(crate) async fn job_peak_memory_bytes(
    transport: &dyn ApiTransport,
    job_id: &str,
) -> Option<u64> {
    let control = RunControl::new();
    let client = Client {
        transport,
        control: &control,
    };
    client
        .job_metrics(job_id, METRICS_RETRY_INTERVAL)
        .await
        .and_then(|metrics| metrics.get("peakMemoryBytes").and_then(Value::as_u64))
        .filter(|bytes| *bytes > 0)
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
    /// The run's cumulative wall-clock deadline, or `None` for work that runs outside the plan's
    /// automatic run budget — a human-requested replacement and the export it re-runs, or an
    /// edit's `--export` (sc-22715). Those are bounded by `shot_deadline` alone.
    run_deadline: Option<Instant>,
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

    /// One page of `GET /api/v1/jobs`, optionally narrowed to one status.
    async fn job_page(
        &self,
        project_id: &str,
        status: Option<&str>,
    ) -> Result<Vec<Value>, HarnessError> {
        let mut path = format!("/api/v1/jobs?projectId={project_id}&limit={JOB_LOOKUP_LIMIT}");
        if let Some(status) = status {
            path.push_str("&status=");
            path.push_str(status);
        }
        let jobs = self.expect_ok("GET", &path, None).await?;
        Ok(jobs.as_array().cloned().unwrap_or_default())
    }

    /// Every job the API holds for `project_id`, newest first.
    ///
    /// `GET /api/v1/jobs` clamps `limit` at [`JOB_LOOKUP_LIMIT`] and takes no offset, so a FULL
    /// page may be a truncated one — with `--project-id` reusing a busy project, the run's own jobs
    /// can sit past the cut and a lookup that assumed one page would silently answer "no such job"
    /// and enqueue a duplicate render. A full page is therefore widened by asking for each status
    /// separately (the route's only other axis) and merging on job id; the common case still costs
    /// exactly one request.
    ///
    /// What no listing can reach: a job the operator CLEARED from the queue. `list_jobs` filters
    /// `cleared_at is null` on every path, so a cleared job is invisible here and a replay will
    /// re-enqueue its attempt. Documented in `docs/film-harness.md`.
    async fn project_jobs(&self, project_id: &str) -> Result<Vec<Value>, HarnessError> {
        let first = self.job_page(project_id, None).await?;
        if first.len() < JOB_LOOKUP_LIMIT as usize {
            return Ok(first);
        }
        let mut pages = vec![first];
        for status in sceneworks_core::jobs_store::JOB_STATUSES {
            pages.push(self.job_page(project_id, Some(status)).await?);
        }
        Ok(merge_job_pages(pages))
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

    /// The `timeline_export` job THIS pass created for `timeline_id`, if the record lost it: the
    /// newest one that is not in `exclude` and was created no earlier than `not_before`.
    ///
    /// The export route takes no payload field of our own, so the timeline id — which the harness
    /// creates, names after the run and never shares — is the key, and every export the run ever
    /// dispatches carries the same one. Two guards keep that from adopting an OLD export as the new
    /// one: `exclude` is every export job the record has ever held and superseded (not merely the
    /// last), and `not_before` is the `requested_at` this pass wrote before it POSTed, so a job
    /// created before this pass can never be adopted.
    async fn find_export_job(
        &self,
        project_id: &str,
        timeline_id: &str,
        exclude: &[String],
        not_before: &str,
    ) -> Result<Option<String>, HarnessError> {
        let jobs = self.project_jobs(project_id).await?;
        Ok(newest_export_job(&jobs, timeline_id, exclude, not_before))
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
            } else if bounds.run_deadline.is_some_and(|deadline| now >= deadline) {
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
    /// Worker rows that advertise `video_generate` / `timeline_export` but are NOT live, so a
    /// refusal can name the stale row that would have fooled a capability-only check.
    stale_video_workers: Vec<String>,
    stale_export_workers: Vec<String>,
}

/// Worker statuses under which a registered row can actually claim a job. A row that still
/// advertises a capability with `status: "offline"` is a worker that left; counting it would queue
/// work nothing will ever claim — the sc-22714 smoke did exactly that against a data dir seeded
/// from an earlier run, and the review preflight learned the lesson first. Every preflight in the
/// harness (run, planner, review) now shares [`live_worker_advertising`] (sc-22715).
pub const LIVE_STATUSES: &[&str] = &["idle", "busy"];

/// The registered workers advertising `capability`, split into the first LIVE one and the rows
/// that advertise it but are not live (`"<id> (<status>)"`, for a refusal to name).
pub(crate) struct WorkerAdvert<'a> {
    pub(crate) live: Option<&'a Value>,
    pub(crate) stale: Vec<String>,
}

/// One rule for "is there a worker that will claim this job": a row counts only while its status
/// is one of [`LIVE_STATUSES`]. Shared by the run preflight (`video_generate`, `timeline_export`),
/// the planner preflight (`prompt_refine`) and the review preflight (`image_vqa`), so no preflight
/// can drift back to counting by capability alone.
pub(crate) fn live_worker_advertising<'a>(
    workers: &'a Value,
    capability: &str,
) -> WorkerAdvert<'a> {
    let advertises = |worker: &Value| {
        worker
            .get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap == capability))
    };
    let live = |worker: &Value| {
        worker
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| LIVE_STATUSES.contains(&status))
    };
    let rows: Vec<&Value> = workers.as_array().into_iter().flatten().collect();
    WorkerAdvert {
        live: rows
            .iter()
            .copied()
            .find(|worker| advertises(worker) && live(worker)),
        stale: rows
            .iter()
            .filter(|worker| advertises(worker) && !live(worker))
            .map(|worker| {
                format!(
                    "{} ({})",
                    worker.get("id").and_then(Value::as_str).unwrap_or("?"),
                    worker.get("status").and_then(Value::as_str).unwrap_or("?")
                )
            })
            .collect(),
    }
}

/// The phrase a refusal appends when stale rows advertise the capability nobody live does.
pub(crate) fn stale_workers_detail(stale: &[String]) -> String {
    if stale.is_empty() {
        String::new()
    } else {
        format!(
            " ({} advertise(s) it but is not live: {})",
            stale.len(),
            stale.join(", ")
        )
    }
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

    /// The memory the API host reports, in GiB, when any registered worker reports one.
    pub(crate) fn host_memory_gb(&self) -> Option<f64> {
        self.host_memory_gb
    }
}

async fn discover_host(client: &Client<'_>) -> Result<HostFacts, HarnessError> {
    let workers = client.expect_ok("GET", "/api/v1/workers", None).await?;
    let mut facts = HostFacts::default();
    // Live rows only (sc-22715): a stale `offline` row still advertising `video_generate` is not
    // a GPU worker, and a run that counted it would dispatch a render nothing claims.
    let video = live_worker_advertising(&workers, "video_generate");
    if let Some(worker) = video.live {
        facts.video_worker_id = worker.get("id").and_then(Value::as_str).map(str::to_owned);
        facts.video_gpu_name = worker
            .get("gpuName")
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    facts.stale_video_workers = video.stale;
    let export = live_worker_advertising(&workers, "timeline_export");
    facts.export_worker = export.live.is_some();
    facts.stale_export_workers = export.stale;
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

/// The key one attempt of one shot dispatches under. Stable across restarts because every part of
/// it is: the run id is in the record, the shot id is in the plan, and attempt numbers never repeat
/// within a shot (sc-22711).
///
/// It is stamped into the dispatched body's `advanced.filmHarness` block (sc-22713's
/// [`CompiledRequest::to_job_body_with`] carries it through [`DispatchContext`]), which is what lets
/// a controller that died between the POST and the record write find its OWN job instead of
/// enqueuing a second render for the same attempt.
pub fn idempotency_key(run_id: &str, shot_id: &str, attempt: u32) -> String {
    format!("{run_id}:{shot_id}:a{attempt}")
}

/// The key one dialogue synthesis dispatches under (sc-23404).
///
/// Keyed on the CONTENT as well as the role — model, voice and the trimmed line — rather than on
/// the role alone. A role-only key would be stable across restarts too, but it would also make
/// re-casting a line (a new voice, a rewritten line) adopt the job that spoke the OLD one, and the
/// film would quietly keep saying the wrong thing.
///
/// `attempt` is the retry axis, the render key's `a{n}` under a different name: a synthesis that
/// FAILED must be re-dispatched by the next resume, and a key without it would find the failed job
/// and re-read the same failure forever. Every part is stable across restarts — the run id is in
/// the record, the content is in the pack, and the attempt is in the record.
pub fn dialogue_idempotency_key(
    run_id: &str,
    role: &str,
    model: &str,
    voice: Option<&str>,
    text: &str,
    attempt: u32,
) -> String {
    let digest =
        sha256_hex(format!("{model}\n{}\n{}", voice.unwrap_or(""), text.trim()).as_bytes());
    format!("{run_id}:sound:{role}:{}:a{attempt}", &digest[..12])
}

/// Where a synthesized line's WAV lands in the pack directory when the entry pins no `file`.
///
/// Deterministic in the role and the line, so the same pack run twice writes the same name and a
/// resume finds the clip it wrote — and so a changed line is a different file rather than a silent
/// overwrite of the one the last export used.
pub fn synthesized_sound_file(role: &str, text_sha256: &str) -> String {
    let digest: String = text_sha256.chars().take(12).collect();
    format!("sound/{role}.tts-{digest}.wav")
}

/// The `type: audio` asset one completed `audio_generate` job produced: its id and its
/// project-relative media path.
///
/// Read off the job result the API rewrote from the worker's `assetWrites` (the same `assets` block
/// [`take_from_result`] reads a render out of), not off the worker's own fact — the rewrite is what
/// says the asset is actually persisted.
fn audio_asset_from_result(result: &Value) -> Option<(String, String)> {
    let asset = result.get("assets")?.as_array()?.first()?;
    let id = asset.get("id")?.as_str()?.to_owned();
    let path = asset.pointer("/file/path")?.as_str()?.to_owned();
    (!path.is_empty()).then_some((id, path))
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

/// The audio layers a `timeline_export` result says it mixed WITHOUT (`droppedAudioLayers`, one
/// object per placed clip whose asset, file or audio stream was missing — sc-22715), copied into
/// the run record's export entry verbatim. Empty for a result that reports none.
fn dropped_audio_layers(result: &Value) -> Vec<Value> {
    result
        .get("droppedAudioLayers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
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
///
/// The temp file is `sync_all`'d before the rename, so the rename cannot be ordered ahead of the
/// data it publishes: without it a power loss can leave the record's NAME pointing at a file whose
/// contents never reached the disk. The directory entry itself is not fsynced, so a power loss can
/// still lose the rename and leave the previous record in place — which is the safe direction, and
/// what a resume reconciles against the API anyway.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), HarnessError> {
    use std::io::Write;
    let temp = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
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
            format!(
                "no live registered worker advertises video_generate{}; start the GPU worker \
                 (SCENEWORKS_WORKER_ONLY=1) and wait for it to register, or clear a stale worker \
                 row that is shadowing it",
                stale_workers_detail(&facts.stale_video_workers)
            ),
        ));
    }
    if export && !facts.export_worker {
        findings.push(PlanDiagnostic::plan(
            "export",
            format!(
                "no live registered worker advertises timeline_export{}; run the API with \
                 SCENEWORKS_RUN_UTILITY_INPROCESS=1 or start a utility worker",
                stale_workers_detail(&facts.stale_export_workers)
            ),
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
    /// The attempt number the take belongs to — stamped into the picture item's harness block
    /// (`filmHarness.attempt`) so a later merge can tell "the selection changed" apart from "a
    /// person pointed this item somewhere else" (sc-22715).
    attempt: u32,
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
///
/// An attempt with NO job has spent nothing, whatever its `startedAt` says. The attempt record is
/// written BEFORE the job is created (that is what makes the idempotency key work), so a controller
/// that died in that window leaves `status: "dispatching", jobId: null` — nothing was rendered, and
/// charging it the wall clock since then would make a resume the next morning compute
/// `spent > maxShotSeconds`, dispatch a real job and cancel it on the first poll, spending the
/// shot's attempt on zero work.
fn attempt_spent_seconds(attempt: &AttemptRecord) -> f64 {
    if attempt.job_id.is_none() {
        return 0.0;
    }
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
    /// The requests this session dispatches (sc-22713): the compiled document beside the plan, or
    /// the plan compiled in memory. Every job body comes from here, so what a reviewer reads in
    /// `compiled.json` is what the API receives — on a resume and a replacement exactly as on the
    /// first run.
    compiled: CompiledPlan,
    facts: HostFacts,
    fps: u32,
    record: RunRecord,
    started: Instant,
    /// Wall-clock earlier controllers already spent on this run. The plan's `maxRunSeconds` bounds
    /// the run, not one attempt at it, so a resume inherits the spend.
    prior_elapsed: f64,
    /// Wall-clock earlier HUMAN-REQUESTED work already spent on this run
    /// (`humanRequestedElapsedSeconds`), inherited the same way.
    prior_human_elapsed: f64,
    /// Whether this controller's own wall-clock is charged to the run's automatic budget
    /// (`elapsedSeconds`) or booked as human-requested work (`humanRequestedElapsedSeconds`).
    /// `run` and `resume` charge the run; `replace-take` / `request-repair` do not (sc-22715): a
    /// replacement the person asked for must not spend the budget the run's own `resume` needs.
    charges_run_budget: bool,
    /// `None` for a session outside the run budget (a replacement), so nothing it polls — the
    /// attempt or the export — can be classified `run_budget`.
    run_deadline: Option<Instant>,
    role_assets: BTreeMap<String, String>,
    /// The pack's sound clips this run has imported, by role (sc-22712). Rebuilt from the record on
    /// a resume exactly as [`Session::role_assets`] is, so a re-layout after a replacement places
    /// the same clips the first controller did without importing (and transcoding) them twice.
    sound_assets: BTreeMap<String, SoundAsset>,
    /// Set the moment dispatch stops, with the outcome that stop implies.
    stop: Option<(RunOutcome, RunStop)>,
}

impl Session<'_> {
    /// Total AUTOMATIC wall-clock this run has consumed, across every controller that has held it.
    /// A controller that does not charge the run budget contributes nothing here.
    fn elapsed(&self) -> f64 {
        if self.charges_run_budget {
            self.prior_elapsed + self.started.elapsed().as_secs_f64()
        } else {
            self.prior_elapsed
        }
    }

    /// Total human-requested wall-clock, the mirror of [`Session::elapsed`].
    fn human_requested_elapsed(&self) -> f64 {
        if self.charges_run_budget {
            self.prior_human_elapsed
        } else {
            self.prior_human_elapsed + self.started.elapsed().as_secs_f64()
        }
    }

    /// Write the record. Called after every state transition — this is the durability contract.
    fn persist(&mut self) -> Result<(), HarnessError> {
        self.record.elapsed_seconds = self.elapsed();
        self.record.human_requested_elapsed_seconds = self.human_requested_elapsed();
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

    /// Whether `memory` is over the plan's budget — and, when it is, halt the run with the memory
    /// stop.
    ///
    /// Both paths that settle an attempt go through here: [`Session::work_attempt`], which watched
    /// the render land, and [`Session::reconcile_shot`], which adopts one a dead controller was
    /// watching. A crash around an over-budget render must stop new dispatch exactly as observing
    /// it does — otherwise a resume adopts the take and keeps dispatching against a budget the
    /// evidence already says was blown.
    fn memory_over_budget(&mut self, shot_id: &str, memory: &MemoryObservation) -> bool {
        if !memory
            .gb
            .is_some_and(|observed| observed > self.plan.limits.max_memory_gb)
        {
            return false;
        }
        self.halt(
            RunOutcome::StoppedMemoryLimit,
            "memory_limit",
            format!(
                "shot {shot_id} peaked at {:.1} GB ({}), over the plan's {} GB budget; raise \
                 limits.maxMemoryGb or pick a cheaper tier and start a new run",
                memory.gb.unwrap_or_default(),
                memory.source.as_deref().unwrap_or("no source"),
                self.plan.limits.max_memory_gb
            ),
            false,
        );
        true
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
        // `includeRejected` / `includeTrashed` default to FALSE on the route, and a human reviewing
        // the project between two controllers can reject or trash an imported plate. Adopting one
        // the listing hid is the point of this lookup, so ask for them.
        let already_imported = if references
            .iter()
            .any(|reference| !imported.contains(&reference.role))
        {
            self.client
                .expect_ok(
                    "GET",
                    &format!(
                        "/api/v1/projects/{project_id}/assets\
                         ?includeRejected=true&includeTrashed=true"
                    ),
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
            // On BOTH branches: the upload and the tag PATCH are two writes, so a controller that
            // died between them leaves an asset the adoption finds but nothing has tagged. The
            // PATCH replaces the tag set, so re-applying it to an already-tagged asset is a no-op.
            self.tag_reference(&project_id, &asset_id, reference)
                .await?;
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

    /// Import every pack sound clip this run will PLACE, and re-adopt the ones an earlier
    /// controller already imported (sc-22712, under sc-22711's resumable session).
    ///
    /// Only what the run PLACES is imported — the two beds plus the dialogue of the SELECTED shots.
    /// Unlike a reference image, importing an audio clip costs an ffmpeg transcode, and a pack
    /// legitimately carries sound for shots a `--shots` run left out. `record.sound` is therefore
    /// the list of clips that were actually available to the mix.
    ///
    /// Resume-safe exactly the way [`Session::ensure_references`] is: a clip the record already
    /// names is adopted rather than re-imported, and one imported by a controller that died before
    /// recording it is found by its `filmHarness` provenance instead of being imported twice. The
    /// durations the beds and lines are laid out from are measured by the import route off the
    /// STORED wav, so they are read back off the project's assets rather than re-derived here —
    /// which is also what lets a resume place the same spans the first controller did.
    async fn ensure_sound(&mut self) -> Result<(), HarnessError> {
        let project_id = self.project_id()?;
        let placed: BTreeSet<String> = self
            .plan
            .sound
            .ambience
            .iter()
            .chain(self.plan.sound.music.iter())
            .map(|bed| bed.role.clone())
            .chain(
                self.plan
                    .shots
                    .iter()
                    .filter(|shot| self.record.selected_shot_ids.contains(&shot.id))
                    .filter_map(|shot| shot.dialogue_clip.as_ref().map(|clip| clip.role.clone())),
            )
            .collect();
        let mut entries: Vec<film_plan::SoundEntry> = self
            .pack
            .sound
            .iter()
            .filter(|entry| placed.contains(&entry.role))
            .cloned()
            .collect();
        if entries.is_empty() {
            return Ok(());
        }
        // Speak every placed line that has no clip yet, BEFORE the listing below: synthesis creates
        // assets and writes files, and the import pass that follows must see a pack directory that
        // already holds them (sc-23404). Each entry comes back with the `file` synthesis wrote, so
        // from here down a spoken line and a recorded one are the same thing.
        //
        // The worker preflight comes first and is scoped to the lines still OWED: a run whose clips
        // are all already spoken needs no TTS worker at all (that is the `replace-take` case), and
        // a run that does need one must be told so here rather than enqueue a job nobody claims and
        // spend the whole per-job budget waiting for it. Same rule as the `video_generate` and
        // `image_vqa` preflights — LIVE rows only.
        let owed: Vec<&film_plan::SoundEntry> = entries
            .iter()
            .filter(|entry| entry.is_synthesized() && !self.line_already_spoken(entry))
            .collect();
        if !owed.is_empty() {
            let roles: Vec<&str> = owed.iter().map(|entry| entry.role.as_str()).collect();
            let workers = self
                .client
                .expect_ok("GET", "/api/v1/workers", None)
                .await?;
            let audio = live_worker_advertising(&workers, "audio_generate");
            if audio.live.is_none() {
                self.halt(
                    RunOutcome::Failed,
                    "no_audio_worker",
                    format!(
                        "the pack asks this run to speak {} ({}) but no live registered worker \
                         advertises audio_generate{}; start a worker with the audio lane and \
                         `film-harness resume` speaks them",
                        roles.len(),
                        roles.join(", "),
                        stale_workers_detail(&audio.stale)
                    ),
                    true,
                );
                return Ok(());
            }
        }
        for entry in &mut entries {
            if !entry.is_synthesized() {
                continue;
            }
            let Some(file) = self.synthesize_dialogue(&project_id, entry).await? else {
                // Halted: the stop is recorded, the clips already spoken and imported stay where
                // they are, and `drive_inner` stops before dispatching a render.
                return Ok(());
            };
            entry.file = Some(file);
        }
        // ONE listing for the whole pass. It does double duty: it carries the stored duration of
        // every clip an earlier controller already imported, and the provenance that finds one it
        // imported but died before recording.
        let assets = self
            .client
            .expect_ok(
                "GET",
                &format!(
                    "/api/v1/projects/{project_id}/assets\
                     ?includeRejected=true&includeTrashed=true"
                ),
                None,
            )
            .await?
            .as_array()
            .cloned()
            .unwrap_or_default();
        let stored_duration = |asset_id: &str| -> Option<f64> {
            assets
                .iter()
                .find(|asset| asset.get("id").and_then(Value::as_str) == Some(asset_id))
                .and_then(|asset| asset.pointer("/file/duration"))
                .and_then(Value::as_f64)
                .filter(|seconds| *seconds > 0.0)
        };
        let recorded: BTreeMap<String, String> = self
            .record
            .sound
            .iter()
            .map(|clip| (clip.role.clone(), clip.asset_id.clone()))
            .collect();
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        for entry in &entries {
            if let Some(asset_id) = recorded.get(&entry.role) {
                self.sound_assets.insert(
                    entry.role.clone(),
                    SoundAsset {
                        asset_id: asset_id.clone(),
                        duration_seconds: stored_duration(asset_id),
                    },
                );
                continue;
            }
            // Set on every entry by now: a recorded clip declares it, and a synthesized one was
            // given it by `synthesize_dialogue` above. An entry with neither is a validation
            // finding the run never gets past.
            let file = entry.file.clone().ok_or_else(|| {
                HarnessError::Transport(format!(
                    "sound {:?} has no file to import; the pack declares neither `file` nor `text`",
                    entry.role
                ))
            })?;
            let path = pack_dir.join(&file);
            let bytes = std::fs::read(&path)?;
            let sha256 = sha256_hex(&bytes);
            // A clip imported by THIS pass is not in the listing above (it was fetched before the
            // upload), so its duration comes back with the import response. Only an ADOPTED clip —
            // one an earlier controller uploaded — is measured off the listing.
            let (asset_id, duration_seconds) =
                match find_imported_reference(&assets, &self.record.run_id, &entry.role, &sha256) {
                    Some(asset_id) => {
                        let duration = stored_duration(&asset_id);
                        (asset_id, duration)
                    }
                    None => {
                        self.import_sound(&project_id, entry, &file, &path, &sha256)
                            .await?
                    }
                };
            // As with a reference: the upload and the tag PATCH are two writes, so re-applying the
            // tags to an adopted asset is the no-op that closes the window between them.
            self.client
                .expect_ok(
                    "PATCH",
                    &format!("/api/v1/projects/{project_id}/assets/{asset_id}/tags"),
                    Some(json!({
                        "tags": [
                            SOUND_TAG,
                            format!("role:{}", entry.role),
                            format!("pack:{}", self.pack.id)
                        ]
                    })),
                )
                .await?;
            self.sound_assets.insert(
                entry.role.clone(),
                SoundAsset {
                    asset_id: asset_id.clone(),
                    duration_seconds,
                },
            );
            self.record.sound.push(ReferenceAssetRecord {
                role: entry.role.clone(),
                kind: entry.kind.clone(),
                file: file.clone(),
                sha256,
                asset_id,
                // A sound entry has no approval flag of its own: approval gates CONDITIONING, and
                // sound is never conditioning. Everything in the pack's `sound` array is placeable.
                approved: true,
            });
            self.persist()?;
        }
        Ok(())
    }

    /// Speak one `dialogue` entry's line through `POST /api/v1/audio/jobs` and leave the WAV in the
    /// pack directory, so the import pass that follows treats it exactly as a pre-recorded clip
    /// (sc-23404).
    ///
    /// Returns the pack-relative path the clip is at, or `None` when the run has HALTED — the job
    /// failed, or it ran past a declared limit. A halt leaves everything already spoken and imported
    /// in the record and stops `drive_inner` before the first render, so a resume picks the sequence
    /// up where it is rather than re-speaking what is already there.
    ///
    /// Resume discipline is the renders': the idempotency key is stamped into the dispatched body's
    /// `advanced.filmHarness` block and a controller that died between the POST and the record write
    /// finds its OWN job by that key. What the key covers is model + voice + text + attempt, not
    /// just the role — so re-casting a line is a different key rather than an adoption of the clip
    /// that says the old thing in the old voice, and a retry after a failure is a new job rather
    /// than a re-read of the same failure.
    ///
    /// The clip's filename is `<role>.<sha256(text)[..12]>.wav` unless the entry pins one with
    /// `file`, in which case synthesis writes THERE — that is how a pack keeps a stable name for a
    /// line it means to check in.
    async fn synthesize_dialogue(
        &mut self,
        project_id: &str,
        entry: &film_plan::SoundEntry,
    ) -> Result<Option<String>, HarnessError> {
        let role = entry.role.clone();
        let text = entry
            .synthesis_text()
            .ok_or_else(|| {
                HarnessError::Transport(format!("sound {role:?} declares an empty `text`"))
            })?
            .to_owned();
        let model = entry.synthesis_model().to_owned();
        let voice = entry.synthesis_voice().map(str::to_owned);
        let text_sha256 = sha256_hex(text.as_bytes());
        let file = entry
            .file
            .clone()
            .unwrap_or_else(|| synthesized_sound_file(&role, &text_sha256));
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let destination = pack_dir.join(&file);

        // 1. Adopt, continue, or start a new attempt. Exactly one record per role — the record says
        //    what the film HAS, and two entries for one role would leave a reader guessing.
        //
        //    - The line this run already spoke, whose WAV is still where it was written: adopt it.
        //      No second job, however many times the run is resumed.
        //    - A record for the same line whose job has NOT ended: continue it under its own key,
        //      which is the "died between the POST and the record write" window.
        //    - A record whose job ENDED without a clip, or one made for a line the pack has since
        //      re-cast: a NEW attempt under a NEW key. Re-polling the finished job under the old
        //      key would make every resume re-read the same failure and never speak the line.
        let existing = self
            .record
            .synthesized_sound
            .iter()
            .position(|line| line.role == role);
        let continues = existing.is_some_and(|index| {
            let line = &self.record.synthesized_sound[index];
            line.matches(&model, voice.as_deref(), &text)
        });
        if continues {
            let index = existing.expect("continues implies a record");
            // Already IMPORTED: the clip is a project asset the dialogue bus is playing, so whether
            // the WAV is still in the pack directory no longer matters — a cleaned pack, or a
            // replacement running months later, must adopt rather than speak the line again.
            let imported = self
                .record
                .sound
                .iter()
                .any(|clip| clip.role == role)
                .then(|| self.record.synthesized_sound[index].file.clone())
                .flatten();
            if self.record.synthesized_sound[index].is_usable() {
                if let Some(file) = imported {
                    return Ok(Some(file));
                }
                if destination.is_file() {
                    return Ok(Some(file));
                }
            }
        }
        let index = match existing {
            Some(index) if continues && !self.record.synthesized_sound[index].is_terminal() => {
                index
            }
            other => {
                let attempt = other
                    .map(|index| self.record.synthesized_sound[index].attempt + 1)
                    .unwrap_or(1);
                let fresh = film_plan::SynthesizedSoundRecord {
                    role: role.clone(),
                    text: text.clone(),
                    text_sha256: text_sha256.clone(),
                    model: model.clone(),
                    voice: voice.clone(),
                    attempt,
                    idempotency_key: dialogue_idempotency_key(
                        &self.record.run_id,
                        &role,
                        &model,
                        voice.as_deref(),
                        &text,
                        attempt,
                    ),
                    job_id: None,
                    status: "dispatching".to_owned(),
                    asset_id: None,
                    file: None,
                    error: None,
                    started_at: utc_now(),
                    finished_at: None,
                };
                let index = match other {
                    Some(index) => {
                        self.record.synthesized_sound[index] = fresh;
                        index
                    }
                    None => {
                        self.record.synthesized_sound.push(fresh);
                        self.record.synthesized_sound.len() - 1
                    }
                };
                // Persisted BEFORE the job exists, exactly as an attempt is: that is what makes the
                // key findable by the controller that comes back.
                self.persist()?;
                index
            }
        };
        let key = self.record.synthesized_sound[index].idempotency_key.clone();
        if self.canceled() {
            self.halt(
                RunOutcome::Canceled,
                "canceled",
                format!("canceled before the dialogue line for {role:?} was synthesized"),
                true,
            );
            return Ok(None);
        }

        // 2. The job. Adopt one already created under this key before creating anything.
        let mut job_id = self.record.synthesized_sound[index].job_id.clone();
        if job_id.is_none() {
            job_id = self
                .client
                .find_job_by_idempotency_key(project_id, &key)
                .await?;
        }
        let created_here = job_id.is_none();
        if job_id.is_none() {
            let mut body = json!({
                "projectId": project_id,
                "prompt": text,
                "model": model,
                "requestedGpu": "auto",
                "advanced": {
                    "filmHarness": {
                        "idempotencyKey": key,
                        "kind": "dialogue",
                        "role": role,
                        "runId": self.record.run_id,
                        "planId": self.plan.id,
                        "planVersion": self.plan.version,
                        "referencePackId": self.pack.id,
                        "referencePackVersion": self.pack.version,
                        "textSha256": text_sha256,
                    }
                }
            });
            if let Some(voice) = &voice {
                body["voice"] = json!(voice);
            }
            let response = self
                .client
                .json("POST", "/api/v1/audio/jobs", Some(body))
                .await?;
            if !(200..300).contains(&response.status) {
                // A refused enqueue is deterministic — an unknown voice, a model with no weights —
                // so retrying it would refuse identically. Say which line, and stop.
                let detail = format!(
                    "POST /api/v1/audio/jobs -> {}: {}",
                    response.status,
                    api_detail(&response.body)
                );
                self.fail_synthesis(index, "rejected", detail.clone());
                self.persist()?;
                self.halt(
                    RunOutcome::Failed,
                    "dialogue_synthesis_refused",
                    format!(
                        "the dialogue line for sound role {role:?} could not be enqueued: \
                         {detail}; fix the pack entry and `film-harness resume` speaks it"
                    ),
                    true,
                );
                return Ok(None);
            }
            job_id = Some(
                response
                    .body
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        HarnessError::Transport(format!(
                            "audio job response has no id: {}",
                            response.body
                        ))
                    })?
                    .to_owned(),
            );
        }
        let job_id = job_id.expect("set on every branch above");
        {
            let line = &mut self.record.synthesized_sound[index];
            if created_here {
                // Nothing has ever run under this key, so the clock starts here — the same reason a
                // render's does (a record written by a controller that died before its POST landed
                // must not be charged every hour since).
                line.started_at = utc_now();
            }
            line.job_id = Some(job_id.clone());
            line.status = "running".to_owned();
        }
        self.persist()?;

        // 3. Wait for it, under the plan's own declared limits: one synthesis is bounded by
        //    `limits.maxShotSeconds` (the same per-job budget the export runs under) and the run by
        //    `limits.maxRunSeconds`. Neither is a new knob — a speech job is a job.
        let started = Instant::now();
        let (view, poll_stop) = self
            .client
            .wait_for_job(&job_id, self.bounds(started + self.shot_budget()))
            .await?;
        let status = match poll_stop {
            PollStop::Terminal | PollStop::AssetsUnsettled => view.status.clone(),
            PollStop::Operator => "canceled_by_operator".to_owned(),
            PollStop::ShotBudget | PollStop::RunBudget => "timed_out".to_owned(),
        };
        let asset = (status == "completed")
            .then(|| audio_asset_from_result(&view.result))
            .flatten();
        let Some((asset_id, media_path)) = asset else {
            let detail = match poll_stop {
                PollStop::Terminal => view.failure_text(),
                PollStop::AssetsUnsettled => format!(
                    "the synthesis job reached {} but its asset never settled",
                    view.status
                ),
                PollStop::Operator => "canceled by operator during synthesis".to_owned(),
                PollStop::ShotBudget => format!(
                    "synthesis exceeded the per-job budget of {}s",
                    self.plan.limits.max_shot_seconds
                ),
                PollStop::RunBudget => format!(
                    "the run's {}s budget ran out during synthesis",
                    self.plan.limits.max_run_seconds
                ),
            };
            self.fail_synthesis(index, &status, detail.clone());
            self.persist()?;
            let (outcome, reason) = match poll_stop {
                PollStop::Operator => (RunOutcome::Canceled, "canceled"),
                PollStop::RunBudget => (RunOutcome::StoppedRunBudget, "run_budget"),
                _ => (RunOutcome::Failed, "dialogue_synthesis_failed"),
            };
            // Resumable on every one of these: the clip is missing, nothing downstream has been
            // written against it, and a resume re-dispatches this one line and continues. The rest
            // of the sequence — the clips already spoken, the imports already made — is untouched.
            self.halt(
                outcome,
                reason,
                format!(
                    "the dialogue line for sound role {role:?} was not synthesized ({detail}); \
                     `film-harness resume` speaks it and continues the sequence"
                ),
                true,
            );
            return Ok(None);
        };

        // 4. The WAV, into the pack directory. The worker writes canonical PCM-16 RIFF/WAVE, which
        //    is the one encoding `media_convert::is_canonical_pcm16_wav` copies through — so the
        //    import below needs no ffmpeg for a clip this run spoke, which is what lets the hosted
        //    macOS lane (no ffmpeg) exercise the whole path.
        let project_path = self
            .record
            .project_path
            .as_deref()
            .map(PathBuf::from)
            .ok_or_else(|| {
                HarnessError::Transport(
                    "the run has no project path, so the synthesized clip cannot be written into \
                     the pack"
                        .to_owned(),
                )
            })?;
        let source = project_path.join(&media_path);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = std::fs::read(&source).map_err(|error| {
            HarnessError::Io(format!(
                "the synthesis job for {role:?} reported {} but it could not be read: {error}",
                source.display()
            ))
        })?;
        std::fs::write(&destination, &bytes)?;
        {
            let line = &mut self.record.synthesized_sound[index];
            line.status = "completed".to_owned();
            line.asset_id = Some(asset_id);
            line.file = Some(file.clone());
            line.error = None;
            line.finished_at = Some(utc_now());
        }
        self.persist()?;
        Ok(Some(file))
    }

    /// Whether this run has already spoken exactly this entry's line and still has the clip — the
    /// same adoption test [`Session::synthesize_dialogue`] applies, read-only, so the TTS worker
    /// preflight can be scoped to the lines that are actually still owed.
    fn line_already_spoken(&self, entry: &film_plan::SoundEntry) -> bool {
        let Some(text) = entry.synthesis_text() else {
            return false;
        };
        let Some(line) = self
            .record
            .synthesized_sound
            .iter()
            .find(|line| line.role == entry.role)
        else {
            return false;
        };
        if !line.matches(entry.synthesis_model(), entry.synthesis_voice(), text)
            || !line.is_usable()
        {
            return false;
        }
        if self.record.sound.iter().any(|clip| clip.role == entry.role) {
            return true;
        }
        let pack_dir = self
            .pack_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        line.file
            .as_deref()
            .is_some_and(|file| pack_dir.join(file).is_file())
    }

    /// Settle one synthesis record as unfinished, keeping whatever it already knew.
    fn fail_synthesis(&mut self, index: usize, status: &str, detail: String) {
        let line = &mut self.record.synthesized_sound[index];
        line.status = status.to_owned();
        line.error = Some(detail);
        line.finished_at = Some(utc_now());
    }

    /// Upload one pack sound clip, with the same `filmHarness` provenance a reference carries so a
    /// controller that dies before recording it can find it again.
    ///
    /// Returns the asset id and the duration the import route MEASURED off the stored PCM-16 wav —
    /// not anything the plan claimed — which is what the beds and lines are then laid out from.
    async fn import_sound(
        &self,
        project_id: &str,
        entry: &film_plan::SoundEntry,
        file: &str,
        path: &Path,
        sha256: &str,
    ) -> Result<(String, Option<f64>), HarnessError> {
        let bytes = std::fs::read(path)?;
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
                "referencePackId": self.pack.id,
                "referencePackVersion": self.pack.version,
                "planId": self.plan.id,
                "planVersion": self.plan.version,
                "runId": self.record.run_id,
                "sourceFile": file,
                // Whether the clip was SPOKEN by this run rather than put on disk by a human
                // (sc-23404), so a reader of the project's assets can tell them apart without the
                // run record in hand.
                "synthesized": entry.is_synthesized(),
                "sha256": sha256,
            }
        });
        let (boundary, body) =
            encode_asset_upload(&filename, audio_content_type(path), &bytes, &provenance);
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
            .map(str::to_owned)
            .ok_or_else(|| {
                HarnessError::Transport(format!(
                    "sound import response has no id: {}",
                    response.body
                ))
            })?;
        let duration_seconds = response
            .body
            .pointer("/file/duration")
            .and_then(Value::as_f64)
            .filter(|seconds| *seconds > 0.0);
        Ok((asset_id, duration_seconds))
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
        Ok(asset_id)
    }

    /// Tag an imported reference with its kind, role and pack.
    ///
    /// An unapproved reference is imported (so the record can point at it and a human can review
    /// it) but tagged distinctly, so a query for the conditioning-eligible references cannot pick
    /// it up. Applied by [`Session::ensure_references`] to a freshly imported asset AND to an
    /// adopted one, because the tags are a second write the crash window can swallow.
    async fn tag_reference(
        &self,
        project_id: &str,
        asset_id: &str,
        reference: &film_plan::ReferenceEntry,
    ) -> Result<(), HarnessError> {
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
        Ok(())
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
            // Geometry, timing and mode come from the COMPILED request, not from the plan read a
            // second time: the compiled document is what the job body is built from (sc-22713), so
            // the record's `intended` has to be the same document or it describes something the
            // route was never asked for.
            let request = self
                .compiled
                .request(&shot.id)
                .expect("every plan shot compiled a request");
            let (width, height) = (request.width, request.height);
            let assets = ConditioningAssets {
                first_frame_asset_id: request
                    .first_frame_role
                    .as_ref()
                    .and_then(|role| self.role_assets.get(role).cloned()),
                last_frame_asset_id: request
                    .last_frame_role
                    .as_ref()
                    .and_then(|role| self.role_assets.get(role).cloned()),
                reference_asset_ids: request
                    .reference_roles
                    .iter()
                    .filter_map(|role| self.role_assets.get(role).cloned())
                    .collect(),
            };
            ordered.push(ShotRunRecord {
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
                    // The policy the export will obey for this shot — its own override, or the
                    // run-level default it inherits (sc-22712). Resolved here, at the same moment
                    // the rest of the intended state is, so the record says what was intended
                    // rather than what a later reader would re-derive from the plan.
                    generated_audio: resolved_generated_audio(&self.plan, shot),
                },
                conditioning_assets: assets,
                attempts: Vec::new(),
                selected_attempt: None,
                needs_review: Vec::new(),
                reviews: Vec::new(),
                human_decision: None,
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
                    if self
                        .run_deadline
                        .is_some_and(|deadline| Instant::now() >= deadline)
                    {
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
        // Neither the record nor the API has a job for this attempt, so this pass creates it: the
        // attempt has been in flight for zero seconds however old its record is.
        let created_here = job_id.is_none();
        if job_id.is_none() {
            let assets = self.record.shots[shot_index].conditioning_assets.clone();
            // The body is the COMPILED request's, always (sc-22713): the document a reviewer reads
            // in `compiled.json` is what the route receives, on a resume and a replacement exactly
            // as on the first run. The record's own resolved conditioning is passed in rather than
            // re-resolved, so an attempt dispatched after a crash conditions on the same assets the
            // record already names.
            let request = self
                .compiled
                .request(&shot.id)
                .expect("every plan shot compiled a request");
            let body = request.to_job_body_with(
                &DispatchContext {
                    project_id: &project_id,
                    run_id: &self.record.run_id,
                    plan_id: &self.plan.id,
                    plan_version: self.plan.version,
                    attempt: attempt_number,
                    tier: self.plan.model.tier.as_deref(),
                    idempotency_key: Some(&key),
                    role_assets: &self.role_assets,
                },
                &ResolvedConditioning {
                    first_frame_asset_id: assets.first_frame_asset_id.clone(),
                    last_frame_asset_id: assets.last_frame_asset_id.clone(),
                    reference_asset_ids: assets.reference_asset_ids.clone(),
                },
            );
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
            if created_here {
                // Nothing has ever run under this key: neither the record nor the API held a job
                // for it, and the one that exists now was created a moment ago. The clock starts
                // HERE. An attempt recorded by a controller that died before its POST reached the
                // API would otherwise be charged every hour since, and a resume the next morning
                // would dispatch a real render only to cancel it on the first poll.
                attempt.started_at = utc_now();
            }
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
        if self.memory_over_budget(&shot.id, &memory) {
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
            let mut adopted_memory = None;
            if terminal {
                let metrics = self.client.job_metrics(&job_id, self.poll_interval).await;
                let memory = memory_observation(metrics.as_ref(), &view, self.facts.host_memory_gb);
                let attempt = &mut self.record.shots[shot_index].attempts[attempt_index];
                attempt.peak_gpu_memory_pct = memory.pct;
                attempt.peak_memory_gb = memory.gb;
                attempt.peak_memory_source = memory.source.clone();
                adopted_memory = Some(memory);
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
            // The plan's memory budget is judged on the EVIDENCE, not on who was watching when it
            // landed: an adopted terminal attempt whose peak is over budget stops new dispatch here
            // exactly as it would in `work_attempt`. The take itself is kept — it exists — but the
            // run goes no further (E5 / AC3).
            if let Some(memory) = adopted_memory {
                self.memory_over_budget(&shot.id, &memory);
            }
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
                    attempt: attempt.attempt,
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

    /// The id of this project's timeline named `name`, if it already has one.
    async fn find_timeline_by_name(
        &self,
        project_id: &str,
        name: &str,
    ) -> Result<Option<String>, HarnessError> {
        Ok(self
            .client
            .expect_ok(
                "GET",
                &format!("/api/v1/projects/{project_id}/timelines"),
                None,
            )
            .await?
            .as_array()
            .into_iter()
            .flatten()
            .find(|timeline| timeline.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|timeline| timeline.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned))
    }

    /// Create (or MERGE INTO) the run's timeline from the selected takes. Pure API writes, no job:
    /// re-running it after a take changed rewrites that one item and leaves the rest — order,
    /// source ranges, version histories, editor-placed items, bus faders — exactly as the saved
    /// sequence holds them (sc-22715).
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
            // Adopt by identity, exactly as the project is. `POST /timelines` ALWAYS creates a new
            // row, and the record's `timeline` is only written after the PUT lands — so a
            // controller killed in that window leaves a timeline nothing names, and a resume that
            // just POSTed again would leave the project holding two identical sequences with the
            // run record pointing at one of them. The name carries the run id, which is what makes
            // the lookup exact (sc-22711).
            None => match self
                .find_timeline_by_name(&project_id, &timeline_name)
                .await?
            {
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
            },
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
        // MERGE, never rebuild (sc-22715). The first assembly of a run creates every picture item;
        // every later pass — a resume, a replacement — starts from the SAVED sequence and changes
        // only what the selection changed. Rebuilding from `selected_takes()` (the sc-22710 shape)
        // silently undid every `trim` / `reorder` / `swap-take` a person had applied through
        // `edit_timeline`, while `timeline.edits` and the decision log went on recording them: a
        // record that said "trimmed, reordered" over a sequence that was neither.
        //
        // What is kept: every existing picture item — its order, its source range, its
        // `versionHistory`, its `generatedAudio`, and an item the harness did not place at all.
        // What changes: an item whose shot now selects a DIFFERENT take is pointed at it (source
        // range reset to the new take's length, one `replacement` entry appended to its history),
        // and a shot with no item yet gets one. Positions are then handed to `relayout_timeline`,
        // the single place that knows how the sequence is timed, in the sequence's EXISTING order
        // with any new shots after it.
        let existing_tracks: Vec<Value> = tracks.to_vec();
        let mut items: Vec<Value> = existing_tracks[track_index]
            .get("items")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        items.sort_by(|left, right| {
            number(left, "timelineStart", 0.0).total_cmp(&number(right, "timelineStart", 0.0))
        });
        for SelectedTake {
            shot_id,
            attempt,
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
            let existing = items
                .iter_mut()
                .find(|item| harness_str(item, "shotId") == Some(shot_id.as_str()));
            // "Did the SELECTION change since this item was last aligned with it?" — not "does
            // the item show the selected asset?". The two differ exactly when a person used
            // `swap-take` to point the item at a foreign asset (an imported clip): the selection
            // is unchanged, the item deliberately is not, and rewriting it would revert their
            // edit. The item's harness block records the attempt it was aligned with; an item
            // from before that stamp existed falls back to the asset comparison.
            let selection_changed = |item: &Value| match item
                .get(HARNESS_KEY)
                .and_then(|block| block.get("attempt"))
                .and_then(Value::as_u64)
            {
                Some(aligned) => aligned != u64::from(*attempt),
                None => item.get("assetId").and_then(Value::as_str) != Some(&take.asset_id),
            };
            match existing {
                Some(item) if !selection_changed(item) => {
                    // The sequence already reflects this selection: nothing about the item moves.
                }
                Some(item) => {
                    // A replacement landed for this shot. Only THIS item changes, and its history
                    // grows rather than restarting — the take that was there stays addressable.
                    item["assetId"] = json!(take.asset_id);
                    item["currentVersionAssetId"] = json!(take.asset_id);
                    item["sourceIn"] = json!(0.0);
                    item["sourceOut"] = json!(length);
                    item[HARNESS_KEY]["attempt"] = json!(attempt);
                    let entry = json!({
                        "assetId": take.asset_id,
                        "source": "replacement",
                        "jobId": job_id,
                        "createdAt": utc_now(),
                        "note": format!("film-harness {} replaced shot {}", self.record.run_id, shot.id),
                    });
                    match item.get_mut("versionHistory").and_then(Value::as_array_mut) {
                        Some(history) => history.push(entry),
                        None => item["versionHistory"] = json!([entry]),
                    }
                    if let Some(versions) = item
                        .get_mut("versionAssetIds")
                        .and_then(Value::as_array_mut)
                    {
                        if !versions.iter().any(|value| value == &json!(take.asset_id)) {
                            versions.push(json!(take.asset_id));
                        }
                    }
                }
                None => {
                    let item_id = format!(
                        "item_{}_{}",
                        shot.id.to_ascii_lowercase(),
                        &self.record.run_id[4..12]
                    );
                    items.push(json!({
                        "id": item_id,
                        "trackId": track_id,
                        "assetId": take.asset_id,
                        "type": "video",
                        "displayName": format!("{} — {}", shot.id, shot.beat).chars().take(160).collect::<String>(),
                        "sourceIn": 0.0,
                        "sourceOut": length,
                        "timelineStart": 0.0,
                        "timelineEnd": length,
                        "speed": 1.0,
                        "fit": "fit",
                        "volume": 1.0,
                        // The RESOLVED policy, written into the timeline itself so the export
                        // obeys the saved document rather than re-deriving anything from the plan
                        // (sc-22712).
                        "generatedAudio": resolved_generated_audio(&self.plan, shot).as_timeline_str(),
                        "versionHistory": [{
                            "assetId": take.asset_id,
                            "source": "original",
                            "jobId": job_id,
                            "note": format!("film-harness {} shot {}", self.record.run_id, shot.id),
                        }],
                        HARNESS_KEY: picture_block(&self.record.run_id, &shot.id, *attempt),
                    }));
                }
            }
        }
        // The cut order the re-layout keeps: the saved sequence's own order, new shots after it.
        let order: Vec<String> = items
            .iter()
            .filter_map(|item| harness_str(item, "shotId").map(str::to_owned))
            .collect();

        // Sound: one dialogue clip per shot that has one, each keeping its offset from the start of
        // its OWN shot, and the two beds placed ONCE across the whole sequence rather than
        // restarting at every cut. `relayout_timeline` gives all of them their positions. These are
        // the HARNESS-placed items (each carries a `filmHarness.role`), re-derived from the plan on
        // every pass exactly as a re-layout after an edit re-derives them; an item on one of these
        // tracks that the harness did NOT place — the editor's own — is carried over untouched,
        // as is the track's own fader (`gain` / `muted`), which a person may have moved.
        //
        // Derived from `order` — the shots the merged PICTURE holds — and not from
        // `selected_takes()` (sc-22715). The two are not the same set: the picture is merged onto
        // the saved sequence, so a shot whose take was rejected (`reject-take`, or a replacement
        // that produced nothing) keeps its item and stays in the cut, while its selection is gone.
        // Deriving the lines from the selection dropped exactly those shots' dialogue on the next
        // re-assembly — the 2026-09-14 evaluation's own film lost SH050's line to a `replace-take`
        // on SH040, and the line is not the harness's to delete while the shot it belongs to is
        // still on screen. A line whose shot HAS left the cut is still dropped, by
        // `relayout_timeline`, which is where "is this shot in the sequence" is actually known.
        let mut dialogue_items = Vec::new();
        for shot_id in &order {
            let Some(shot) = self.plan.shots.iter().find(|shot| &shot.id == shot_id) else {
                // An item the harness placed for a shot the plan no longer names. Its picture item
                // is carried over untouched above; there is no plan clip to derive a line from.
                continue;
            };
            let Some(clip) = &shot.dialogue_clip else {
                continue;
            };
            let Some(asset) = self.sound_assets.get(&clip.role) else {
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
                "id": format!(
                    "item_line_{}_{}",
                    shot.id.to_ascii_lowercase(),
                    &self.record.run_id[4..12]
                ),
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
                    &self.record.run_id,
                    Some(&shot.id),
                    clip.offset_seconds,
                ),
            }));
        }

        // The ids of every picture item the merged sequence holds, kept before `items` is moved
        // onto the track, so the save can be checked against them below.
        let intended_item_ids: Vec<String> = items
            .iter()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect();
        let mut picture = existing_tracks[track_index].clone();
        picture["items"] = Value::Array(items);
        let mut tracks = vec![
            picture,
            merge_harness_audio_track(
                &existing_tracks,
                audio_track(
                    DIALOGUE_TRACK_ID,
                    "Dialogue",
                    ROLE_DIALOGUE,
                    &self.plan.sound.dialogue,
                    dialogue_items,
                ),
            ),
        ];
        for (bed_track_id, name, role, bed) in [
            (
                AMBIENCE_TRACK_ID,
                "Ambience",
                ROLE_AMBIENCE,
                self.plan.sound.ambience.as_ref(),
            ),
            (
                MUSIC_TRACK_ID,
                "Music",
                ROLE_MUSIC,
                self.plan.sound.music.as_ref(),
            ),
        ] {
            let Some(bed) = bed else { continue };
            let Some(asset) = self.sound_assets.get(&bed.role) else {
                continue;
            };
            tracks.push(merge_harness_audio_track(
                &existing_tracks,
                bed_track(bed_track_id, name, role, bed, asset, &self.record.run_id),
            ));
        }
        // Keep every track the API created that the harness does not own (the overlay lane and the
        // editor's default audio lane) so a harness timeline opens in the editor unchanged.
        for track in &existing_tracks {
            let id = track.get("id").and_then(Value::as_str).unwrap_or_default();
            if !tracks
                .iter()
                .any(|kept| kept.get("id").and_then(Value::as_str) == Some(id))
            {
                tracks.push(track.clone());
            }
        }
        timeline["tracks"] = Value::Array(tracks);

        let planned_duration = relayout_timeline(&mut timeline, Some(&order))?;
        let saved = self
            .client
            .expect_ok(
                "PUT",
                &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
                Some(json!({ "timeline": timeline })),
            )
            .await?;
        // Check the SAVED document before describing it, rather than trusting the harness's own
        // intent: the record then cannot claim a sequence the project does not hold. The store may
        // legitimately add or reshape keys, so what is checked is that every picture item this pass
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
        // Edits a human applied to this sequence are HISTORY, not something a re-assembly erases: a
        // resume or a replacement re-lays the picture, but the record must still say the trim or
        // reorder happened (sc-22712 under sc-22711's record).
        let prior_edits = self
            .record
            .timeline
            .as_ref()
            .map(|timeline| timeline.edits.clone())
            .unwrap_or_default();
        self.record.timeline = Some(timeline_record(
            &timeline_id,
            &timeline_name,
            aspect_ratio,
            Some(reduced_aspect_ratio(first_width, first_height)),
            Some(first_width),
            Some(first_height),
            self.fps,
            duration,
            &saved,
            &track_id,
            self.plan.sound.generated_audio,
            prior_edits,
        ));
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
        // An export recorded as `running` is the very job this pass is about to poll again, so it
        // is adopted BY ID — never looked up in the listing, which is what makes the `requested_at`
        // floor below safe to apply to everything the listing can return.
        let in_flight = self
            .record
            .export
            .as_ref()
            .filter(|export| Some(&export.job_id) != superseded.as_ref())
            .map(|export| export.job_id.clone());
        if let Some(job_id) = superseded.clone() {
            // The exclusion set is CUMULATIVE: a second re-export must exclude the first export as
            // well as the second, or it adopts the first one's finished job and records its
            // pre-replacement asset as the current MP4.
            if !self.record.superseded_export_job_ids.contains(&job_id) {
                self.record.superseded_export_job_ids.push(job_id);
            }
        }
        if self.record.export_pending.is_none() {
            self.record.export_pending = Some(ExportPending {
                requested_at: utc_now(),
                supersedes: superseded,
            });
            self.record.export = None;
            // Persisted BEFORE the export job exists, for the same reason a shot attempt is.
            self.persist()?;
        }
        let requested_at = self
            .record
            .export_pending
            .as_ref()
            .map(|pending| pending.requested_at.clone())
            .unwrap_or_else(utc_now);
        let superseded_ids = self.record.superseded_export_job_ids.clone();
        let existing = match in_flight {
            Some(job_id) => Some(job_id),
            None => {
                self.client
                    .find_export_job(
                        &project_id,
                        &timeline.timeline_id,
                        &superseded_ids,
                        &requested_at,
                    )
                    .await?
            }
        };
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
            dropped_audio_layers: Vec::new(),
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
            dropped_audio_layers: dropped_audio_layers(&view.result),
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

    /// Close a `replace-take` invocation, which is scoped to ONE shot.
    ///
    /// [`Session::finish`] classifies the RUN: its `all_rendered` is computed over every selected
    /// shot, so closing a run that still has unrendered shots through it overwrites whatever stop
    /// the run actually had with `attempts_exhausted` / `resumable: false` — permanently blocking
    /// the `resume` that was going to render them. A replacement decides one shot's outcome and
    /// nothing else: when the run carried a resumable stop and other selected shots are still
    /// outstanding, that stop is left exactly where it was.
    ///
    /// The same rule holds for a run that had NO stop, which is to say one that `completed`
    /// (sc-22715). `all_rendered` is false the moment any selected shot has no selection — a
    /// `reject-take` leaves exactly that — so classifying through [`Session::finish`] turned a
    /// completed run into `failed` / `attempts_exhausted` because a DIFFERENT shot was replaced.
    /// That is the `edit_timeline` rule stated for the generation side: a replacement is not a
    /// run, and a verdict the run reached is not re-opened by one. The evaluation film's own
    /// record read `failed` for exactly this reason while its snapshots read `completed`.
    fn finish_replacement(
        &mut self,
        export_ok: bool,
        replaced_shot_id: &str,
        prior: (RunOutcome, Option<RunStop>),
    ) -> Result<(), HarnessError> {
        let outstanding: Vec<String> = self
            .record
            .shots
            .iter()
            .filter(|shot| shot.shot_id != replaced_shot_id)
            .filter(|shot| self.is_selected(&shot.shot_id) && shot.selected_attempt.is_none())
            .map(|shot| shot.shot_id.clone())
            .collect();
        // Did THIS replacement land a take on the shot it was asked about? That, and not the state
        // of every other shot, is what this invocation decided.
        let replacement_succeeded = self
            .record
            .shots
            .iter()
            .find(|shot| shot.shot_id == replaced_shot_id)
            .is_some_and(|shot| shot.selected_attempt.is_some());
        // An `export_failed` stop is about an export this replacement did not redo (no
        // `--export`): the run still has no MP4, so the stop — and the `resume` that retries the
        // export — must survive too (sc-22715).
        let export_still_owed = !self.export
            && prior
                .1
                .as_ref()
                .is_some_and(|stop| stop.resumable && stop.reason == "export_failed");
        match prior {
            (outcome, Some(stop))
                if stop.resumable && (!outstanding.is_empty() || export_still_owed) =>
            {
                self.note_decision(
                    "replace_take",
                    Some(replaced_shot_id),
                    if outstanding.is_empty() {
                        format!(
                            "the run's own stop ({}) is left in place: the export was not redone, \
                             so the run is still resumable",
                            stop.reason
                        )
                    } else {
                        format!(
                            "the run's own stop ({}) is left in place: {} still {} no take, so the \
                             run is still resumable",
                            stop.reason,
                            outstanding.join(", "),
                            if outstanding.len() == 1 { "has" } else { "have" }
                        )
                    },
                );
                self.stop = None;
                self.record.outcome = outcome;
                self.record.stop = Some(stop);
                self.record.state = RunState::Finished;
                self.record.finished_at = Some(utc_now());
                self.persist()
            }
            // The run reached its verdict with no stop of its own — it COMPLETED — and this
            // replacement landed. `finish` would re-derive that verdict from every selected shot,
            // so a shot whose take a human rejected flips the whole run to `attempts_exhausted`
            // because another shot was replaced. Only the export moves: it no longer matches the
            // sequence, which `replace_take` already recorded as `stale`.
            (RunOutcome::Completed, None) if replacement_succeeded && self.stop.is_none() => {
                if export_ok {
                    self.note_decision(
                        "replace_take",
                        Some(replaced_shot_id),
                        format!(
                            "the run's own outcome (completed) is left in place: this replacement \
                             decided shot {replaced_shot_id} and no other shot's state was \
                             re-judged"
                        ),
                    );
                    self.record.outcome = RunOutcome::Completed;
                    self.record.stop = None;
                } else {
                    // The take landed and the re-export did not: that, and only that, is what
                    // this invocation failed at — `attempts_exhausted` would name a different
                    // shot's state as the reason and refuse the `resume` that retries the export.
                    self.record.outcome = RunOutcome::Failed;
                    self.record.stop = Some(RunStop {
                        reason: "export_failed".to_owned(),
                        detail: "the replacement landed but its re-export did not complete; \
                                 `film-harness resume` retries the export"
                            .to_owned(),
                        resumable: true,
                    });
                }
                self.record.state = RunState::Finished;
                self.record.finished_at = Some(utc_now());
                self.persist()
            }
            _ => self.finish(export_ok),
        }
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
        // Before the shots, not after: a resume and a `replace-take` both re-lay the timeline at
        // the end of this same method, and the re-layout needs the clips to place. Importing here
        // means the sound of a run is settled by the time the first take exists, whichever entry
        // point is driving (sc-22712).
        self.ensure_sound().await?;
        // A synthesis that failed, was cancelled or ran past a declared limit has already recorded
        // its stop (sc-23404). Dispatching renders against a sequence that is missing a line would
        // spend GPU hours on a film the operator would have to re-export anyway, so the run stops
        // here with everything it has and `resume` speaks the missing line.
        if self.stop.is_some() {
            return Ok(false);
        }
        self.work_shots().await?;
        self.assemble_and_export().await
    }

    /// Flag every shot that declared a dependency on `shot_id`. See [`flag_dependents`], which the
    /// human reject path in [`review::decide_take`] shares, so a take that stops being the selected
    /// one raises exactly the same signal however it stopped.
    fn flag_dependents(&mut self, shot_id: &str, reason: &str) {
        flag_dependents(
            &mut self.record,
            &self.plan,
            shot_id,
            &format!("its selected take was replaced ({reason})"),
        );
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
    // A `run` over a directory that already holds a record would mint a NEW run id and persist over
    // the previous run's state, while `persist_record` keeps the plan/pack copies it finds (they
    // are written once and then left alone) — so the surviving documents would belong to the old
    // run and the record's hashes to the new one, and a later resume would refuse or mis-hash.
    // `scripts/film-harness-smoke.sh` pins `--out "$SMOKE_DIR/run"`, so a second invocation is
    // exactly this. The takes, decisions and provenance of a run are not something a typo may
    // overwrite (E2).
    if let Ok(existing) = read_run_record(&options.out_dir) {
        return Err(HarnessError::Refused(format!(
            "{} already holds run {} ({:?}); `film-harness status --out {}` prints it, \
             `film-harness resume --out {}` continues it, and a NEW run needs a different --out",
            options.out_dir.join(RUN_RECORD_FILE).display(),
            existing.run_id,
            existing.outcome,
            options.out_dir.display(),
            options.out_dir.display()
        )));
    }
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
    // The requests this run dispatches: the compiled document beside the plan when there is one,
    // else the plan's own prompts compiled in memory (sc-22713). `record.plan.sha256` is the hash
    // of the plan file as read, which is what a compiled document pins. A resume re-reads the same
    // document and re-checks the same hash, so the requests a replay dispatches are the ones the
    // first controller did.
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
        &plan,
        &pack,
        &prepared.entry,
        lane,
        &record.plan.sha256,
        supplied_compiled.map(|(compiled, _)| compiled),
    )?;
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
        compiled,
        facts: prepared.facts,
        fps: prepared.fps,
        record,
        started,
        prior_elapsed: 0.0,
        prior_human_elapsed: 0.0,
        charges_run_budget: true,
        run_deadline: Some(run_deadline),
        role_assets: BTreeMap::new(),
        sound_assets: BTreeMap::new(),
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
    /// The requests the run was dispatching, re-read from the document the record names (and
    /// re-checked against its recorded hash) or recompiled from the plan when it named none — so a
    /// resume dispatches what the first controller did, not a fresh compilation of an edited file.
    compiled: CompiledPlan,
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
    // The requests the run was dispatching. A record that named a compiled document is held to it,
    // hash and all — re-reading the file the run started from is what keeps a resume from
    // dispatching an edited `compiled.json` under the takes the first controller already made, the
    // same rule the plan and the pack are held to. A record that named none compiles the plan in
    // memory exactly as the first controller did (sc-22713).
    let supplied = match &record.compiled {
        Some(document) => {
            let (_, bytes) = read_source(
                Path::new(&document.path),
                &options.out_dir.join("compiled.json"),
                "compiled requests",
            )?;
            let actual = sha256_hex(&bytes);
            if actual != document.sha256 {
                return Err(HarnessError::Refused(format!(
                    "the compiled requests changed since run {} started (recorded {}, found \
                     {actual}); recompile and start a new run",
                    record.run_id, document.sha256
                )));
            }
            Some(
                serde_json::from_str::<CompiledPlan>(
                    &sceneworks_core::jsonc::strip_jsonc_comments(
                        std::str::from_utf8(&bytes).unwrap_or_default(),
                    ),
                )
                .map_err(|error| HarnessError::Refused(format!("compiled requests: {error}")))?,
            )
        }
        None => None,
    };
    let compiled = compiled_for_run(
        &plan,
        &pack,
        &prepared.entry,
        prepared.facts.lane(),
        &record.plan.sha256,
        supplied,
    )?;
    Ok(Continued {
        record,
        plan,
        pack,
        compiled,
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

/// `run_deadline: None` is a session that runs OUTSIDE the plan's automatic run budget (a
/// replacement): its wall-clock is booked as human-requested and nothing it polls can stop on
/// `run_budget`.
fn session_from<'a>(
    transport: &'a dyn ApiTransport,
    options: &'a ResumeOptions,
    continued: Continued,
    started: Instant,
    run_deadline: Option<Instant>,
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
        compiled: continued.compiled,
        facts: continued.prepared.facts,
        fps: continued.prepared.fps,
        prior_elapsed: continued.record.elapsed_seconds,
        prior_human_elapsed: continued.record.human_requested_elapsed_seconds,
        charges_run_budget: run_deadline.is_some(),
        record: continued.record,
        started,
        run_deadline,
        role_assets: BTreeMap::new(),
        sound_assets: BTreeMap::new(),
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
    let mut session = session_from(transport, options, continued, started, Some(run_deadline));
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
    // One replacement runs OUTSIDE the run budget (sc-22715): the attempt is bounded by the
    // per-shot budget and the re-export by its own, and neither is charged to `elapsedSeconds` —
    // the human just authorised this one attempt, and a replacement that spent the run's remaining
    // wall-clock would make the run's own `resume` refuse with "raise limits.maxRunSeconds". Its
    // wall-clock is booked in `humanRequestedElapsedSeconds` instead, so the total cost is still
    // on the record. With no run deadline, an export that overruns is `export_failed` (resumable),
    // never `run_budget` (terminal).
    let mut session = session_from(transport, options, continued, started, None);
    // What the RUN said before this replacement: a replacement is scoped to one shot and must not
    // re-classify the run, so a resumable stop (a cancel, a crash, a failed export) is restored by
    // `finish_replacement` when other selected shots are still outstanding — and a run that
    // finished with no stop at all (`completed`) keeps that verdict, which is why the OUTCOME is
    // carried here even when there is no stop beside it (sc-22715).
    let prior_verdict = (session.record.outcome, session.record.stop.clone());
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
        // The re-assembly re-derives the harness's own dialogue and bed items from the plan and
        // needs the sound assets to place them (sc-22715 evaluation). Without this the merge
        // re-derived an EMPTY dialogue track over the saved one and every line the run had placed
        // was dropped from the sequence — the beds only survived because a bed track with no
        // asset is skipped and then kept as "not the harness's". `ensure_sound` adopts the clips
        // the record already names — including every line this run SPOKE (sc-23404) — so it
        // uploads nothing and speaks nothing on this path.
        session.ensure_sound().await?;
        // A clip that could not be re-hydrated (its file gone from the pack, a re-synthesis that
        // failed, no live TTS worker) leaves the session's map short, and re-assembling from a
        // short map is exactly the deletion the call above exists to prevent. Leave the saved
        // timeline alone and let the stop say so: the sequence keeps the sound it has, and the
        // replacement's take is still recorded.
        if session.stop.is_none() {
            // Rewriting the timeline is a PUT, not a job: every other shot's item keeps its asset.
            session.assemble_timeline().await?;
        }
    } else {
        // The shot now has NO selected take, and the timeline was deliberately not rewritten (that
        // would drop the shot out of the sequence entirely). So the timeline — and the MP4 rendered
        // from it — still carry the take the human just rejected, and the record has to say so
        // rather than leave `stale: false` claiming the export is current.
        let rejected_note = format!(
            "the replacement attempt for shot {shot_id} produced no take, so the timeline and the \
             exported MP4 still carry the REJECTED take"
        );
        if let Some(export) = session.record.export.as_mut() {
            export.stale = true;
        }
        session.note_decision("replace_take", Some(shot_id), rejected_note.clone());
        session.persist()?;
        if session.stop.is_none() {
            session.halt(
                RunOutcome::Failed,
                "replacement_failed",
                format!(
                    "{rejected_note}; the rejected take is still recorded and `film-harness \
                     replace-take --shot {shot_id}` authorises another"
                ),
                false,
            );
        }
    }
    // Not re-exporting is a deliberate choice, not a failure: the replacement succeeded, the
    // existing MP4 is marked stale, and the run is as finished as this invocation was asked to make
    // it. Only a failed replacement leaves the run failed (with the stop set above).
    let export_ok = if replaced && session.export {
        session.run_export().await?
    } else {
        replaced
    };
    session.finish_replacement(export_ok, shot_id, prior_verdict)?;
    Ok(session.record)
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

/// Flag every shot that declared a dependency on `shot_id` AND already has a take of its own.
/// A shot that has not rendered yet needs no flag: it will be rendered against the current
/// state. Direct dependents only — a flagged shot's own take did not change, so the signal does
/// not cascade on its own. Returns how many shots were flagged.
///
/// Shared by [`replace_take`] and by the human reject path in [`review::decide_take`], so a take
/// that stops being the selected one raises exactly the same signal however it stopped.
fn flag_dependents(
    record: &mut RunRecord,
    plan: &ProductionPlan,
    shot_id: &str,
    what_happened: &str,
) -> usize {
    let dependents: Vec<(String, String, String)> = film_plan::direct_dependents(plan, shot_id)
        .into_iter()
        .map(|(shot, edge)| (shot.id.clone(), edge.kind.clone(), edge.note.clone()))
        .collect();
    let raised_at = utc_now();
    let mut flagged = 0;
    for (dependent_id, kind, note) in dependents {
        let Some(shot_record) = record.shot_mut(&dependent_id) else {
            continue;
        };
        if shot_record.selected_attempt.is_none() {
            continue;
        }
        // One STANDING flag per (source shot, dependency kind): the same unread signal raised
        // twice is still one thing to look at. It is keyed on what is standing rather than on
        // history, so a flag a person RESOLVED (`film-harness accept-take`, sc-22714) is raised
        // again by the next change upstream — which is the whole point of resolving it.
        if shot_record
            .needs_review
            .iter()
            .any(|flag| flag.source_shot_id == shot_id && flag.dependency == kind)
        {
            continue;
        }
        let detail = if note.trim().is_empty() {
            String::new()
        } else {
            format!(" ({note})")
        };
        shot_record.needs_review.push(ReviewFlag {
            raised_at: raised_at.clone(),
            source_shot_id: shot_id.to_owned(),
            dependency: kind.clone(),
            reason: format!(
                "shot {shot_id}: {what_happened}; this shot's {kind} depends on it{detail} — \
                 review it and replace it too if it no longer matches"
            ),
        });
        flagged += 1;
    }
    flagged
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
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits: plan.limits.clone(),
        selected_shot_ids,
        references: Vec::new(),
        sound: Vec::new(),
        synthesized_sound: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        export_pending: None,
        superseded_export_job_ids: Vec::new(),
        diagnostics: Vec::new(),
        decisions: Vec::new(),
        elapsed_seconds: 0.0,
        human_requested_elapsed_seconds: 0.0,
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
            planner_max_memory_gb: None,
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
        compiled: None,
        project_id: options.project_id.clone(),
        project_path: None,
        model: None,
        limits,
        selected_shot_ids: options.shot_ids.clone().unwrap_or_default(),
        references: Vec::new(),
        sound: Vec::new(),
        synthesized_sound: Vec::new(),
        shots: Vec::new(),
        timeline: None,
        export: None,
        export_pending: None,
        superseded_export_job_ids: Vec::new(),
        diagnostics: findings,
        decisions: Vec::new(),
        elapsed_seconds: seconds_since(started),
        human_requested_elapsed_seconds: 0.0,
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

/// Placeholder BED sound for the fixture pack (sc-22712): `(role, seconds, hz, amplitude)`.
///
/// The two beds are long enough to play under the WHOLE six-shot sequence (6 x 5.1667s ~= 31s)
/// without running out, because a bed that stops partway would make the one thing this fixture is
/// meant to demonstrate — continuous sound across intentional cuts — unobservable.
///
/// The fixture's three DIALOGUE roles are NOT here (sc-23404): they carry `text` and are spoken by
/// the run through the audio route, so there is no placeholder tone left to write for them. The
/// tones were placeholders for exactly this, and a "with-dialogue" export that carried a 400 Hz
/// beep instead of a line was the thing they were standing in for.
pub const FIXTURE_SOUNDS: &[(&str, f64, u32, i16)] = &[
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

    #[test]
    fn a_re_derived_harness_audio_item_keeps_the_editors_volume_and_fades() {
        let harness_item =
            |id: &str, shot: Option<&str>, volume: f64, fade_in: f64, fade_out: f64| {
                json!({
                    "id": id, "trackId": "track_dialogue", "assetId": "asset_line", "type": "audio",
                    "sourceIn": 0.0, "sourceOut": 2.0, "timelineStart": 0.0, "timelineEnd": 2.0,
                    "volume": volume, "fadeInSeconds": fade_in, "fadeOutSeconds": fade_out,
                    HARNESS_KEY: harness_block(ROLE_DIALOGUE, "run_0123456789ab", shot, 1.2),
                })
            };
        // The saved sequence: the editor turned SH020's line down and gave it fades, left
        // SH050's line alone, and placed one clip of their own on the same track.
        let saved = json!({ "tracks": [{
            "id": "track_dialogue", "name": "Dialogue", "kind": "audio", "role": "dialogue",
            "gain": 0.7, "muted": false,
            "items": [
                harness_item("item_line_sh020", Some("SH020"), 0.4, 0.25, 0.5),
                harness_item("item_line_sh050", Some("SH050"), 1.0, 0.0, 0.0),
                { "id": "item_editor", "trackId": "track_dialogue", "assetId": "asset_foreign",
                  "type": "audio", "sourceIn": 0.0, "sourceOut": 1.0,
                  "timelineStart": 3.0, "timelineEnd": 4.0, "volume": 0.9 },
            ],
        }] });
        let tracks = saved["tracks"].as_array().cloned().unwrap();
        // A later pass re-derives every harness item from the plan — at the plan's gain, with the
        // plan's fades — and this time SH060 has a take, so its line appears for the first time.
        let fresh = json!({
            "id": "track_dialogue", "name": "Dialogue", "kind": "audio", "role": "dialogue",
            "gain": 1.0, "muted": false,
            "items": [
                harness_item("item_line_sh020", Some("SH020"), 1.0, 0.0, 0.0),
                harness_item("item_line_sh050", Some("SH050"), 1.0, 0.0, 0.0),
                harness_item("item_line_sh060", Some("SH060"), 0.8, 0.0, 0.4),
            ],
        });
        let merged = merge_harness_audio_track(&tracks, fresh);
        let items = merged["items"].as_array().unwrap();
        let by_id = |id: &str| {
            items
                .iter()
                .find(|item| item["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("{id} is on the merged track"))
        };
        let sh020 = by_id("item_line_sh020");
        assert_eq!(
            sh020["volume"],
            json!(0.4),
            "the editor's volume survives the pass"
        );
        assert_eq!(sh020["fadeInSeconds"], json!(0.25));
        assert_eq!(sh020["fadeOutSeconds"], json!(0.5));
        assert_eq!(by_id("item_line_sh050")["volume"], json!(1.0));
        let sh060 = by_id("item_line_sh060");
        assert_eq!(
            sh060["volume"],
            json!(0.8),
            "an item with no saved counterpart is the fresh one"
        );
        assert_eq!(sh060["fadeOutSeconds"], json!(0.4));
        assert_eq!(
            by_id("item_editor")["volume"],
            json!(0.9),
            "the editor's own clip is kept"
        );
        assert_eq!(items.len(), 4);
        assert_eq!(merged["gain"], json!(0.7), "the saved fader wins");
        // A bed carries no shot id: it matches its saved counterpart on the role alone.
        let bed = |volume: f64| {
            json!({ "id": "track_music", "kind": "audio", "role": "music", "gain": 0.2,
                "items": [{ "id": "item_music", "type": "audio", "volume": volume,
                    "fadeInSeconds": 2.0, "fadeOutSeconds": 3.0,
                    HARNESS_KEY: harness_block(ROLE_MUSIC, "run_0123456789ab", None, 0.0) }] })
        };
        let merged_bed = merge_harness_audio_track(&[bed(0.5)], bed(1.0));
        assert_eq!(merged_bed["items"][0]["volume"], json!(0.5));
        // With no saved track at all, the fresh one is used as is.
        assert_eq!(
            merge_harness_audio_track(&[], bed(1.0))["items"][0]["volume"],
            json!(1.0)
        );
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
    fn an_export_lookup_excludes_every_export_the_record_ever_held() {
        let export = |id: &str, created: &str, timeline: &str| {
            json!({
                "id": id,
                "type": "timeline_export",
                "createdAt": created,
                "payload": { "timelineId": timeline }
            })
        };
        let jobs = vec![
            export("export_1", "2026-09-13T10:00:00Z", "tl_run"),
            export("export_2", "2026-09-13T11:00:00Z", "tl_run"),
            export("export_3", "2026-09-13T12:00:00Z", "tl_other"),
            json!({
                "id": "video_1",
                "type": "video_generate",
                "createdAt": "2026-09-13T11:30:00Z",
                "payload": { "timelineId": "tl_run" }
            }),
        ];
        // The FIRST re-export: export_1 is superseded, and nothing newer than the request exists.
        assert_eq!(
            newest_export_job(
                &jobs,
                "tl_run",
                &["export_1".to_owned()],
                "2026-09-13T11:30:00Z"
            ),
            None,
            "export_2 predates this request, so it is not the job this pass created"
        );
        // The SECOND re-export: both earlier exports are excluded, so nothing is adoptable — the
        // bug was adopting export_1 here and recording its pre-replacement asset as current.
        assert_eq!(
            newest_export_job(
                &jobs,
                "tl_run",
                &["export_1".to_owned(), "export_2".to_owned()],
                "2026-09-13T09:00:00Z"
            ),
            None
        );
        // Excluding only the most recent one is exactly the defect.
        assert_eq!(
            newest_export_job(
                &jobs,
                "tl_run",
                &["export_2".to_owned()],
                "2026-09-13T09:00:00Z"
            ),
            Some("export_1".to_owned())
        );
        // The job this pass really did create is adopted: newer than the request, not excluded.
        let mut with_new = jobs.clone();
        with_new.push(export("export_4", "2026-09-13T12:30:00Z", "tl_run"));
        assert_eq!(
            newest_export_job(
                &with_new,
                "tl_run",
                &["export_1".to_owned(), "export_2".to_owned()],
                "2026-09-13T12:00:00Z"
            ),
            Some("export_4".to_owned())
        );
        // A job with no readable `createdAt` is never adopted: the harness would rather run a
        // second export than record an unrelated asset as this run's delivered MP4.
        let undated = vec![json!({
            "id": "export_5",
            "type": "timeline_export",
            "payload": { "timelineId": "tl_run" }
        })];
        assert_eq!(
            newest_export_job(&undated, "tl_run", &[], "2026-09-13T09:00:00Z"),
            None
        );
    }

    #[test]
    fn job_pages_merge_on_id_newest_first() {
        let job = |id: &str, created: &str| json!({ "id": id, "createdAt": created });
        let merged = merge_job_pages(vec![
            vec![
                job("a", "2026-09-13T10:00:00Z"),
                job("b", "2026-09-13T12:00:00Z"),
            ],
            vec![
                job("b", "2026-09-13T12:00:00Z"),
                job("c", "2026-09-13T11:00:00Z"),
            ],
            vec![json!({ "type": "video_generate" })],
        ]);
        let ids: Vec<&str> = merged
            .iter()
            .filter_map(|job| job.get("id").and_then(Value::as_str))
            .collect();
        assert_eq!(ids, vec!["b", "c", "a"], "one entry per id, newest first");
    }

    #[test]
    fn an_attempt_with_no_job_has_spent_nothing() {
        let attempt = |job_id: Option<&str>, started_at: &str| AttemptRecord {
            attempt: 1,
            idempotency_key: "run:SH010:a1".to_owned(),
            job_id: job_id.map(str::to_owned),
            status: "dispatching".to_owned(),
            started_at: started_at.to_owned(),
            finished_at: None,
            elapsed_seconds: 0.0,
            peak_gpu_memory_pct: None,
            peak_memory_gb: None,
            peak_memory_source: None,
            error: None,
            take: None,
            rejection: None,
            human_requested: false,
        };
        let yesterday = sceneworks_core::time::format_unix_seconds(
            sceneworks_core::time::now_unix_seconds() - 86_400,
        );
        assert_eq!(
            attempt_spent_seconds(&attempt(None, &yesterday)),
            0.0,
            "an attempt recorded before its job POST landed rendered nothing, whatever its \
             startedAt says"
        );
        assert!(
            attempt_spent_seconds(&attempt(Some("job_1"), &yesterday)) > 86_000.0,
            "an attempt that HAS a job has been in flight since it started"
        );
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

/// A picture item's harness block: the shot, plus the attempt whose take the item was ALIGNED
/// with when the harness last wrote it (sc-22715). A merge compares this against the shot's
/// current `selectedAttempt` to decide whether the selection moved.
fn picture_block(run_id: &str, shot_id: &str, attempt: u32) -> Value {
    let mut block = harness_block(ROLE_PICTURE, run_id, Some(shot_id), 0.0);
    block["attempt"] = json!(attempt);
    block
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

/// Merge a freshly derived harness audio track (`fresh`, with only harness-placed items) onto the
/// saved one of the same id, if the sequence already has it (sc-22715).
///
/// The saved track wins on everything a person may have changed — its fader (`gain`, `muted`),
/// its name, and every item the harness did NOT place (no `filmHarness.role`); the harness's own
/// items are the fresh ones, because they are re-derived from the plan on every pass exactly as
/// `relayout_timeline` re-places them. A fresh item that has a saved counterpart — the same
/// `filmHarness.role` and `shotId` — keeps that item's `volume`, `fadeInSeconds` and
/// `fadeOutSeconds`, which are the editor's per-item controls and were being reset to the plan's
/// values on every resume and replacement (sc-22715). With no saved track the fresh one is used
/// as is.
///
/// Dropping the saved harness items is only safe because the fresh set is derived from the shots
/// the merged PICTURE holds, not from the selection: a saved harness item whose shot is still in
/// the cut has a fresh counterpart here, and one whose shot has left the cut is meant to go. When
/// the fresh set came from `selected_takes()` instead, a shot that kept its picture item but lost
/// its selection (a rejected take) had neither — and its line vanished from the sequence.
fn merge_harness_audio_track(existing_tracks: &[Value], fresh: Value) -> Value {
    let id = fresh.get("id").and_then(Value::as_str).unwrap_or_default();
    let Some(saved) = existing_tracks
        .iter()
        .find(|track| track.get("id").and_then(Value::as_str) == Some(id))
    else {
        return fresh;
    };
    let saved_items: Vec<Value> = saved
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut merged = saved.clone();
    let mut items: Vec<Value> = fresh
        .get("items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut item| {
            let counterpart = saved_items.iter().find(|candidate| {
                harness_str(candidate, "role").is_some()
                    && harness_str(candidate, "role") == harness_str(&item, "role")
                    && harness_str(candidate, "shotId") == harness_str(&item, "shotId")
            });
            if let Some(counterpart) = counterpart {
                for key in ["volume", "fadeInSeconds", "fadeOutSeconds"] {
                    if let Some(kept) = counterpart.get(key) {
                        item[key] = kept.clone();
                    }
                }
            }
            item
        })
        .collect();
    items.extend(
        saved_items
            .iter()
            .filter(|item| harness_str(item, "role").is_none())
            .cloned(),
    );
    merged["items"] = Value::Array(items);
    merged
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
            // This export just ran against the timeline as it stands, so it matches the selected
            // takes by construction. Only a LATER take change makes an export stale (sc-22711).
            stale: false,
            asset_id,
            render_path,
            dropped_audio_layers: dropped_audio_layers(&view.result),
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
    SwapTake { shot_id: String, asset_id: String },
}

impl TimelineEdit {
    fn kind(&self) -> &'static str {
        match self {
            Self::Trim { .. } => "trim",
            Self::Reorder { .. } => "reorder",
            Self::SwapTake { .. } => "swap_take",
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

    let (order, detail, swapped_attempt) = match &edit {
        TimelineEdit::Reorder { shot_ids } => (Some(shot_ids.clone()), shot_ids.join(" -> "), None),
        TimelineEdit::Trim {
            shot_id,
            source_in,
            source_out,
        } => {
            // Measure the take FIRST, the way `SwapTake` does. An out point past the end of the
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
                None,
            )
        }
        TimelineEdit::SwapTake { shot_id, asset_id } => {
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
            // The attempt this asset belongs to, when it is one of the shot's OWN takes. The
            // selection follows it below, and so does the item's aligned-attempt stamp — a later
            // merge then reads "selection N, item aligned with N" and leaves the swap alone. A
            // swap onto a FOREIGN asset moves no selection and leaves the stamp as it is, for the
            // same reason: the selection did not change, so the merge must not touch the item
            // (sc-22715).
            let own_attempt = record.shot(shot_id).and_then(|shot| {
                shot.attempts
                    .iter()
                    .find(|attempt| {
                        attempt
                            .take
                            .as_ref()
                            .is_some_and(|take| &take.asset_id == asset_id)
                    })
                    .map(|attempt| attempt.attempt)
            });
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
            if let Some(attempt) = own_attempt {
                item[HARNESS_KEY]["attempt"] = json!(attempt);
            }
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
                own_attempt,
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
        detail: detail.clone(),
        duration_seconds: duration,
    });
    // An edit is a HUMAN decision about this run, so it belongs in the run's decision log next to
    // the generation-side ones (sc-22711 schema 2). `timeline.edits` records what the sequence now
    // IS; the decision log records that somebody changed it, and when — the two are not the same
    // question, and only the decision log is read in run order beside the takes.
    record.decisions.push(ProductionDecision {
        at: utc_now(),
        action: edit.kind().to_owned(),
        shot_id: match &edit {
            TimelineEdit::Trim { shot_id, .. } | TimelineEdit::SwapTake { shot_id, .. } => {
                Some(shot_id.clone())
            }
            TimelineEdit::Reorder { .. } => None,
        },
        detail,
    });
    // A swap to an asset this shot ALREADY rendered is a change of selection, so the record's
    // selected attempt has to follow it. Leaving it behind would point `status`, a later
    // `replace-take` and the dependency flags at an attempt the sequence no longer shows. A swap to
    // a foreign asset (an imported clip, not a take of this run) selects no attempt: there is none
    // to select, and the timeline item is then the only thing that says what is in the cut.
    if let (TimelineEdit::SwapTake { shot_id, .. }, Some(attempt)) = (&edit, swapped_attempt) {
        if let Some(shot) = record.shot_mut(shot_id) {
            shot.selected_attempt = Some(attempt);
        }
    }
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
        // Human-requested work, like a replacement (sc-22715): bounded by the export's own
        // per-job budget, charged to `humanRequestedElapsedSeconds`, and never classified
        // `run_budget` — an edit's re-export must not spend the run's automatic wall-clock.
        let export_started = Instant::now();
        let (export_record, _) = export_timeline(
            &client,
            &project_id,
            &timeline_id,
            export_resolution_for(tallest),
            existing.fps,
            PollBounds {
                shot_deadline: export_started + shot_budget,
                run_deadline: None,
                poll_interval: options.poll_interval,
                cancel_grace: CANCEL_GRACE.min(shot_budget),
                settle_grace: ASSET_SETTLE_GRACE.min(shot_budget),
            },
            &record.limits,
        )
        .await?;
        record.human_requested_elapsed_seconds += export_started.elapsed().as_secs_f64();
        let export_ok = export_record.status == "completed" && export_record.asset_id.is_some();
        record.export = Some(export_record);
        // The run's `outcome` / `stop` describe the RUN, and an edit is not a run (sc-22715): a
        // successful re-export leaves them exactly as they were — a `canceled` or an
        // `attempts_exhausted` stop is still true after a trim — except an `export_failed` stop,
        // which is about precisely the export this one just replaced. A failed re-export is
        // recorded as one, resumable, unless the run already carries a terminal stop of its own.
        if export_ok {
            if record
                .stop
                .as_ref()
                .is_some_and(|stop| stop.reason == "export_failed")
            {
                record.outcome = RunOutcome::Completed;
                record.stop = None;
            }
        } else if record.stop.as_ref().is_none_or(|stop| stop.resumable) {
            record.outcome = RunOutcome::Failed;
            record.stop = Some(RunStop {
                reason: "export_failed".to_owned(),
                detail: "the edit's re-export did not complete; `film-harness resume` retries the \
                         export"
                    .to_owned(),
                resumable: true,
            });
        }
    } else if let Some(export) = &mut record.export {
        // The sequence just changed under the MP4 that was exported from it, so that MP4 no longer
        // matches the run. 22711's rule holds here exactly as it does for a re-rendered take: the
        // export is FLAGGED, never silently re-run — re-exporting stays an explicit `--export`.
        export.stale = true;
    }

    // The same write every controller makes (sc-22715): atomic, and mirrored to
    // `<project>/film-harness/<run_id>/run.json` — an edit rewrote the record with a plain
    // `fs::write` and left the project's copy describing a sequence that no longer existed.
    let out_dir = options
        .run_record_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    persist_record(
        &record,
        &out_dir,
        Path::new(&record.plan.path),
        Path::new(&record.reference_pack.path),
    )?;
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
