//! API transport, request encoding, polling and response admission.

use super::*;

/// How many jobs a reconciliation lists when looking for a job this run created.
pub(super) const JOB_LOOKUP_LIMIT: u32 = 500;

/// Attempt statuses that need no further reconciliation on a resume.
pub(super) const TERMINAL_ATTEMPT_STATUSES: &[&str] = &[
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

/// The API's answer to a file request: HTTP status plus the raw body. A non-2xx body is kept as-is
/// so a refusal's JSON detail can still be read out of it.
#[derive(Debug, Clone)]
pub struct BytesResponse {
    pub status: u16,
    pub bytes: Vec<u8>,
}

pub type TransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ApiResponse, HarnessError>> + Send + 'a>>;

pub type BytesTransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<BytesResponse, HarnessError>> + Send + 'a>>;

/// The harness's only dependency on the outside world. Implemented over `reqwest` for the binary
/// and over an in-process `axum::Router` in tests.
pub trait ApiTransport: Send + Sync {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_>;

    /// GET a file the API serves as BYTES rather than as JSON — a project's stored media
    /// (`/api/v1/projects/:id/files/*path`). Two callers share it: `make-references` (sc-23403)
    /// downloads a rendered plate into the pack directory, and a synthesized dialogue clip
    /// (sc-23404) reaches the pack directory the same way.
    ///
    /// A separate method rather than a flag on [`ApiRequest`] because the two answers have
    /// different shapes: [`ApiResponse`] parses its body as JSON, and neither a PNG nor a WAV is
    /// JSON. It is the
    /// download seam, so it must go through the transport like every other request: `--api` may
    /// legitimately point at a SceneWorks API on another machine on the private network, whose
    /// project directory this process cannot read.
    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_>;
}

#[derive(Debug)]
pub enum HarnessError {
    /// The plan was refused before any job was created. The run record carries the same findings.
    Validation(Vec<PlanDiagnostic>),
    /// The planner exhausted its bounded repair loop after protocol-valid replies. Keep every
    /// sanitized execution receipt beside the findings so a failed plan has the same provider,
    /// settings, duration, and usage provenance as a successful one.
    PlannerValidation {
        findings: Vec<PlanDiagnostic>,
        executions: Vec<PlannerExecutionRecord>,
    },
    /// The API answered a non-success status the harness cannot proceed past.
    Api {
        method: &'static str,
        path: String,
        status: u16,
        detail: String,
    },
    /// The transport itself failed (connection refused, malformed response, ...).
    Transport(String),
    /// A planner returned a protocol-valid response that cannot be accepted as plan text. The
    /// sanitized execution survives with the actionable failure instead of being discarded.
    PlannerResponse {
        detail: String,
        execution: Box<PlannerExecutionRecord>,
    },
    /// A later planner request failed after earlier protocol-valid replies. Preserve the earlier
    /// sanitized receipts while retaining the original error's status and display semantics.
    PlannerExecutionFailure {
        source: Box<HarnessError>,
        executions: Vec<PlannerExecutionRecord>,
    },
    /// The requested action does not apply to the run record on disk — it is finished and not
    /// resumable, its documents no longer hash to what the run was started from, or it names no
    /// such shot. Nothing was dispatched (sc-22711).
    Refused(String),
    /// An accepted action was canceled before it could create work.
    Canceled(String),
    Io(String),
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(findings) | Self::PlannerValidation { findings, .. } => {
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
            Self::PlannerResponse { detail, .. } => write!(f, "transport error: {detail}"),
            Self::PlannerExecutionFailure { source, .. } => write!(f, "{source}"),
            Self::Refused(message) => write!(f, "refused: {message}"),
            Self::Canceled(message) => write!(f, "canceled: {message}"),
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
pub(super) fn merge_job_pages(pages: Vec<Vec<Value>>) -> Vec<Value> {
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
pub(super) fn newest_export_job(
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

pub(super) fn api_detail(body: &Value) -> String {
    body.get("detail")
        .map(|detail| match detail {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| body.to_string())
}

/// The harness's view of one job snapshot, read off `GET /api/v1/jobs/:id`.
#[derive(Debug, Clone)]
pub(super) struct JobView {
    pub(super) status: String,
    pub(super) stage: String,
    pub(super) error: Option<String>,
    pub(super) message: String,
    pub(super) peak_gpu_memory_pct: Option<f64>,
    pub(super) backend: Option<String>,
    pub(super) result: Value,
    /// The resize mode the PERSISTED job names (sc-26691): stamped at creation or, for a job queued
    /// before the key existed, at the claim — so it is what the job rendered (or will render) with.
    /// `None` on a job that has not been claimed since the upgrade.
    pub(super) reference_image_resize_mode:
        Option<sceneworks_core::video_request::ReferenceImageResizeMode>,
}

impl JobView {
    pub(super) fn from_snapshot(snapshot: &Value) -> Option<Self> {
        Some(Self {
            status: snapshot.get("status")?.as_str()?.to_owned(),
            stage: snapshot
                .get("stage")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
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
            reference_image_resize_mode: snapshot
                .get("payload")
                .and_then(|payload| payload.get("advanced"))
                .and_then(Value::as_object)
                .and_then(|advanced| {
                    sceneworks_core::video_request::requested_reference_image_resize_mode(advanced)
                        .ok()
                        .flatten()
                }),
        })
    }

    pub(super) fn is_terminal(&self) -> bool {
        TERMINAL_STATUSES.contains(&self.status.as_str())
    }

    /// Whether a terminal snapshot's result is final. The API commits a worker's terminal status
    /// BEFORE it persists the reported `assetWrites` into project assets and rewrites the result
    /// as `assets` / `assetIds` (the durable two-phase handoff in `update_job_progress`), so a
    /// poller can observe `completed` with the raw facts still in place. Reading that snapshot as
    /// done would record "completed without an asset".
    ///
    /// Every reported write must be covered by a persisted asset identity: an earlier running
    /// update may have published only the first of several outputs. Raw facts may remain after a
    /// successful handoff, so their absence is not required. A no-project or persistently failing
    /// handoff is still bounded by the existing asset-settle grace and reported as unsettled,
    /// rather than being retried or consuming the whole shot budget.
    pub(super) fn is_settled(&self) -> bool {
        if self.status != "completed" {
            return true;
        }
        let Some(writes) = self.result.get("assetWrites").and_then(Value::as_array) else {
            return true;
        };
        writes.iter().all(|write| {
            let Some(id) = write
                .get("assetId")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            else {
                return false;
            };
            self.result
                .get("assets")
                .and_then(Value::as_array)
                .is_some_and(|assets| {
                    assets
                        .iter()
                        .any(|asset| asset.get("id").and_then(Value::as_str) == Some(id))
                })
                || self
                    .result
                    .get("assetIds")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| ids.iter().any(|asset_id| asset_id.as_str() == Some(id)))
        })
    }

    pub(super) fn failure_text(&self) -> String {
        self.error
            .clone()
            .filter(|error| !error.trim().is_empty())
            .unwrap_or_else(|| self.message.clone())
    }
}

/// Why a poll loop returned before the job finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PollStop {
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
pub(super) struct PollBounds {
    pub(super) shot_deadline: Instant,
    /// The run's cumulative wall-clock deadline, or `None` for work that runs outside the plan's
    /// automatic run budget — a human-requested replacement and the export it re-runs, or an
    /// edit's `--export` (sc-22715). Those are bounded by `shot_deadline` alone.
    pub(super) run_deadline: Option<Instant>,
    pub(super) poll_interval: Duration,
    /// How long a canceled job gets to reach a terminal state: [`CANCEL_GRACE`] capped at the
    /// plan's per-shot budget.
    pub(super) cancel_grace: Duration,
    /// How long a terminal job gets to finish persisting its reported assets:
    /// [`ASSET_SETTLE_GRACE`] capped at the plan's per-shot budget.
    pub(super) settle_grace: Duration,
}

pub(super) struct Client<'a> {
    pub(super) transport: &'a dyn ApiTransport,
    pub(super) control: &'a RunControl,
}

impl Client<'_> {
    pub(super) async fn json(
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
    pub(super) async fn expect_ok(
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

    pub(super) async fn get_job(&self, job_id: &str) -> Result<JobView, HarnessError> {
        let snapshot = self
            .expect_ok("GET", &format!("/api/v1/jobs/{job_id}"), None)
            .await?;
        JobView::from_snapshot(&snapshot).ok_or_else(|| {
            HarnessError::Transport(format!("job {job_id} snapshot has no status: {snapshot}"))
        })
    }

    /// One page of `GET /api/v1/jobs`, optionally narrowed to one status.
    pub(super) async fn job_page(
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
    /// `GET /api/v1/jobs` takes no offset and answers at most the `limit` it is asked for
    /// ([`JOB_LOOKUP_LIMIT`] here), so a FULL page may be a truncated one — with `--project-id`
    /// reusing a busy project, the run's own jobs
    /// can sit past the cut and a lookup that assumed one page would silently answer "no such job"
    /// and enqueue a duplicate render. A full page is therefore widened by asking for each status
    /// separately (the route's only other axis) and merging on job id; the common case still costs
    /// exactly one request.
    ///
    /// What no listing can reach: a job the operator CLEARED from the queue. `list_jobs` filters
    /// `cleared_at is null` on every path, so a cleared job is invisible here and a replay will
    /// re-enqueue its attempt. Documented in `docs/film-harness.md`.
    pub(super) async fn project_jobs(&self, project_id: &str) -> Result<Vec<Value>, HarnessError> {
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
    pub(super) async fn find_job_by_idempotency_key(
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
    pub(super) async fn find_export_job(
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
    pub(super) async fn wait_for_job(
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
    pub(super) async fn job_metrics(&self, job_id: &str, poll_interval: Duration) -> Option<Value> {
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

/// Download one of a project's stored media files over the API.
///
/// Through the transport, not off the disk: `--api` may point at a SceneWorks API elsewhere on the
/// private network, whose project directory this process cannot read.
pub(super) async fn download_media(
    transport: &dyn ApiTransport,
    project_id: &str,
    media_path: &str,
) -> Result<Vec<u8>, HarnessError> {
    if !is_safe_media_path(media_path) {
        return Err(HarnessError::Transport(format!(
            "the API reported the file at {media_path:?}, which is not a plain relative media \
             path inside the project"
        )));
    }
    let path = format!("/api/v1/projects/{project_id}/files/{media_path}");
    let response = transport.get_bytes(path.clone()).await?;
    if !(200..300).contains(&response.status) {
        return Err(HarnessError::Api {
            method: "GET",
            path,
            status: response.status,
            detail: api_detail(&serde_json::from_slice(&response.bytes).unwrap_or(Value::Null)),
        });
    }
    if response.bytes.is_empty() {
        return Err(HarnessError::Transport(format!(
            "GET {path} returned an empty file"
        )));
    }
    Ok(response.bytes)
}

/// A project-relative media path that is safe to interpolate into the file route: no absolute
/// path, no `..`, no escaping or encoding needed.
pub(super) use sceneworks_core::safe_path::is_safe_media_path;

// ---------------------------------------------------------------------------------------------
// HTTP transport (the binary's)
// ---------------------------------------------------------------------------------------------

/// [`ApiTransport`] over `reqwest` against a running SceneWorks API.
pub struct HttpTransport {
    pub(super) base_url: String,
    pub(super) token: Option<String>,
    pub(super) client: reqwest::Client,
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

    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        Box::pin(async move {
            let url = format!("{}{}", self.base_url, path);
            let mut builder = self.client.get(&url);
            if let Some(token) = &self.token {
                builder = builder.header("x-sceneworks-token", token);
            }
            let response = builder
                .send()
                .await
                .map_err(|error| HarnessError::Transport(format!("{url}: {error}")))?;
            let status = response.status().as_u16();
            let bytes = response
                .bytes()
                .await
                .map_err(|error| HarnessError::Transport(format!("{url}: {error}")))?
                .to_vec();
            Ok(BytesResponse { status, bytes })
        })
    }
}
