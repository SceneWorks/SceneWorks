use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use parking_lot::Mutex;
use rusqlite::{
    params, params_from_iter, Connection, OptionalExtension, Row, ToSql, TransactionBehavior,
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Number, Value};

use crate::contracts::{
    ContractNumber, GenerationMetrics, GenerationMetricsRow, JobSnapshot, JobStatus, JobType,
    ProgressStage, QueueSummary, WorkerCapability, WorkerSnapshot, WorkerStatus,
    WorkerUtilizationSnapshot,
};
use crate::store_util::{ensure_column, parse_string_enum, random_hex};
use crate::time::{format_unix_seconds, now_unix_seconds, parse_utc_seconds, utc_now};

mod routing;

// Re-export the moved routing/gating surface so the store's remaining SQL-coupled dispatch,
// the `super::*` test modules below, and external consumers keep resolving these names through
// `jobs_store::` unchanged (sc-8816 — a pure code move, no API change). The dispatch code uses
// the gaps/mlx/candle predicates directly; the catalog lists are exercised only by the
// `#[cfg(test)]` routing suites, so that glob is test-gated to stay warning-clean.
pub(crate) use routing::candle::*;
#[cfg(test)]
pub(crate) use routing::catalog::*;
pub(crate) use routing::gaps::*;
pub(crate) use routing::mlx::*;

// External re-export surface: `apps/rust-api/src/lib.rs` and the integration test
// (`tests/jobs_store.rs`) import these already-public items from `jobs_store::` directly.
pub use routing::catalog::{
    mac_capabilities, model_mac_support, MacCapabilities, MAC_NOT_AVAILABLE_LABEL,
};
pub use routing::gaps::{candle_supported, mac_rust_supported, UnsupportedReason};

pub const ACTIVE_STATUSES: &[&str] = &[
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
];
pub const TERMINAL_STATUSES: &[&str] = &["completed", "failed", "canceled", "interrupted"];
pub const JOB_STATUSES: &[&str] = &[
    "queued",
    // Accepted-but-not-yet-claimable: awaiting the API-side async payload rewrite (Ideogram 4
    // auto-caption, sc-9120) before it becomes `queued`. Deliberately absent from both
    // ACTIVE_STATUSES and TERMINAL_STATUSES (like `queued`) so the claim SELECT ignores it and
    // the queue summary counts it as an in-flight, non-terminal job. See JobStatus::PendingCaption.
    "pending_caption",
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
    "completed",
    "failed",
    "canceled",
    "interrupted",
];
pub const NON_GPU_JOB_TYPES: &[&str] = &[
    "model_download",
    "model_import",
    "model_convert",
    "lora_import",
    "lora_download",
];
pub const MAX_JOB_ATTEMPTS: u32 = 5;

/// The non-GPU job types as a quoted SQL list for `type in (...)` / `type not in
/// (...)` dispatch filters, derived once from [`NON_GPU_JOB_TYPES`]. This keeps
/// the SQL from drifting away from the declared contract — the drift this fixes
/// was `model_convert` living in the const but missing from the hard-coded SQL
/// lists (sc-1629). Values are crate constants, never user input, so direct
/// interpolation is safe.
fn non_gpu_job_types_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        NON_GPU_JOB_TYPES
            .iter()
            .map(|job_type| format!("'{job_type}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The active (non-terminal, non-queued) statuses as a quoted SQL list for
/// `status in (...)` stale-sweep / claim-guard filters, derived once from
/// [`ACTIVE_STATUSES`] — same anti-drift rationale as [`non_gpu_job_types_sql`]
/// (sc-4207 / F-CORE-3): the list was copy-pasted into five SQL statements, so
/// adding/renaming an active status risked missing one. Values are crate
/// constants, never user input, so direct interpolation is safe.
fn active_statuses_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        ACTIVE_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

/// The terminal statuses as a quoted SQL list for `status not in (...)` filters,
/// derived once from [`TERMINAL_STATUSES`] — same anti-drift rationale as
/// [`active_statuses_sql`]. Used to select the non-terminal (still in-flight,
/// including `queued`) jobs for the queue summary. Values are crate constants,
/// never user input, so direct interpolation is safe.
fn terminal_statuses_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        TERMINAL_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}
const DISPATCH_MEMORY_NOT_WORSE_TOLERANCE_MB: f64 = 512.0;
const DISPATCH_MEMORY_RELIEF_THRESHOLD_MB: f64 = 1024.0;
const DISPATCH_LOW_MEMORY_THRESHOLD_MB: f64 = 2048.0;
const DISPATCH_HEALTHY_MEMORY_THRESHOLD_MB: f64 = 4096.0;
const DISPATCH_LOAD_NOT_WORSE_TOLERANCE_PERCENT: f64 = 10.0;
const DISPATCH_LOAD_RELIEF_THRESHOLD_PERCENT: f64 = 15.0;
const DISPATCH_HIGH_LOAD_THRESHOLD_PERCENT: f64 = 85.0;
const DISPATCH_RECOVERED_LOAD_THRESHOLD_PERCENT: f64 = 75.0;
const DISPATCH_MEMORY_USAGE_NOT_WORSE_TOLERANCE_PERCENT: f64 = 10.0;
const DISPATCH_MEMORY_USAGE_RELIEF_THRESHOLD_PERCENT: f64 = 10.0;
const DISPATCH_HIGH_MEMORY_USAGE_THRESHOLD_PERCENT: f64 = 90.0;
const DISPATCH_RECOVERED_MEMORY_USAGE_THRESHOLD_PERCENT: f64 = 80.0;

pub type JobsStoreResult<T> = Result<T, JobsStoreError>;

#[derive(Debug)]
pub enum JobsStoreError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    NotFound(String),
    InvalidStatus(String),
    InvalidNumber(String),
    InvalidRequestedGpu(String),
    RetryLimit {
        max_attempts: u32,
    },
    /// A progress report tried to change a job that already reached a terminal
    /// status. Terminal jobs are immutable; only an idempotent re-report of the
    /// same terminal status succeeds (sc-4172).
    TerminalJobImmutable {
        job_id: String,
        status: String,
    },
    /// A progress report came from a worker that no longer owns the job — the
    /// job was swept/canceled (worker_id cleared) or reclaimed. The worker
    /// should abandon the job (sc-4172).
    NotJobOwner {
        job_id: String,
    },
    /// `create_job` was asked to create a job in a status other than the two
    /// legal pre-worker statuses (`queued` / `pending_caption`), e.g. a
    /// mid-lifecycle or terminal status. A programmer error, not user input.
    InvalidInitialStatus(String),
}

impl std::fmt::Display for JobsStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::NotFound(id) => write!(formatter, "Record not found: {id}"),
            Self::InvalidStatus(status) => write!(formatter, "Unsupported job status: {status}"),
            Self::InvalidNumber(field) => write!(formatter, "Invalid numeric value for {field}"),
            Self::InvalidRequestedGpu(detail) => write!(formatter, "{detail}"),
            Self::RetryLimit { max_attempts } => {
                write!(
                    formatter,
                    "Job retry limit reached after {max_attempts} attempts."
                )
            }
            Self::TerminalJobImmutable { job_id, status } => {
                write!(
                    formatter,
                    "Job {job_id} is already {status}; terminal jobs cannot be updated."
                )
            }
            Self::NotJobOwner { job_id } => {
                write!(
                    formatter,
                    "Progress rejected: the reporting worker no longer owns job {job_id}."
                )
            }
            Self::InvalidInitialStatus(status) => write!(
                formatter,
                "A job can only be created in 'queued' or 'pending_caption' status, not '{status}'."
            ),
        }
    }
}

impl std::error::Error for JobsStoreError {}

impl From<std::io::Error> for JobsStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for JobsStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for JobsStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug)]
pub struct JobsStore {
    db_path: PathBuf,
    lock: Mutex<()>,
}

#[derive(Debug, Clone)]
pub struct CreateJob {
    pub job_type: JobType,
    pub project_id: Option<String>,
    pub project_name: Option<String>,
    pub payload: Map<String, Value>,
    pub requested_gpu: String,
    pub source_job_id: Option<String>,
    pub duplicate_of_job_id: Option<String>,
    pub attempts: u32,
    /// Status the job is created in. `None` means the default `queued` (immediately
    /// claimable). `Some(JobStatus::PendingCaption)` creates the job NON-claimable so an
    /// API-side async pre-step (the Ideogram 4 auto-caption, sc-9120) can rewrite its
    /// payload and promote it to `queued` before any worker sees it. Only `queued` and
    /// `pending_caption` are valid initial statuses; any other value is rejected so a job
    /// can't be born mid-lifecycle (e.g. `running`) or terminal.
    pub initial_status: Option<JobStatus>,
}

impl CreateJob {
    /// The initial status string for the insert, defaulting to `queued`. Enforces the
    /// invariant that a job is only ever born `queued` or `pending_caption` — the two
    /// pre-worker statuses — so a caller can't inject a mid-lifecycle or terminal status.
    fn initial_status_str(&self) -> JobsStoreResult<&'static str> {
        match &self.initial_status {
            None | Some(JobStatus::Queued) => Ok("queued"),
            Some(JobStatus::PendingCaption) => Ok("pending_caption"),
            Some(other) => Err(JobsStoreError::InvalidInitialStatus(
                other.as_str().to_owned(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DuplicateJob {
    pub payload_changes: Map<String, Value>,
    pub requested_gpu: Option<String>,
}

/// Outcome of [`JobsStore::promote_pending_caption_job`] (sc-9120). `promoted` is `true` when the
/// job was still `pending_caption` and this call transitioned it to `queued`; `false` when the
/// guarded UPDATE matched nothing because the job had already left `pending_caption` (canceled by
/// the user, or recovered to `queued` on an API restart) — in which case the caller must NOT treat
/// the caption as having been applied. `job` is the row's current snapshot either way.
#[derive(Debug, Clone)]
pub struct PendingCaptionPromotion {
    pub promoted: bool,
    pub job: JobSnapshot,
}

#[derive(Debug, Clone)]
pub struct RetryJob {
    pub payload_changes: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct RegisterWorker {
    pub worker_id: String,
    pub gpu_id: String,
    pub gpu_name: Option<String>,
    pub capabilities: Vec<WorkerCapability>,
    pub loaded_models: Vec<String>,
    pub utilization: Option<WorkerUtilizationSnapshot>,
}

#[derive(Debug, Clone)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub status: WorkerStatus,
    pub current_job_id: Option<String>,
    pub loaded_models: Vec<String>,
    pub utilization: Option<WorkerUtilizationSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub status: JobStatus,
    pub stage: ProgressStage,
    pub progress: f64,
    pub message: String,
    pub error: Option<String>,
    pub result: Option<Map<String, Value>>,
    pub eta_seconds: Option<f64>,
    /// Sampled GPU memory percentage observed by the worker at this progress
    /// point (0..100). The store keeps a running max across a job's progress
    /// updates (sc-2086) so completed-row meters render the peak.
    pub peak_gpu_memory_pct: Option<f64>,
    /// Sampled GPU load percentage observed at this progress point (0..100).
    /// Same running-max semantics as peak_gpu_memory_pct.
    pub peak_gpu_load_pct: Option<f64>,
    /// Runtime backend label the worker reports for this job
    /// ("mlx" / "mps" / "cuda" / "cpu"). First non-null value sticks — once a
    /// worker tells us which backend ran the job, subsequent status-only
    /// progress updates can't accidentally clear it. Drives the
    /// WorkerProgressCard arch pill.
    pub backend: Option<String>,
    /// Id of the worker reporting this progress. When set, the store rejects
    /// the update unless the job's `worker_id` still matches — a zombie worker
    /// whose job was swept to `interrupted` (worker_id cleared) or reclaimed by
    /// another worker can no longer resurrect or corrupt it (sc-4172). `None`
    /// keeps legacy trusted-caller behavior.
    pub worker_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StaleSweep {
    pub workers: Vec<WorkerSnapshot>,
    pub jobs: Vec<JobSnapshot>,
}

impl JobsStore {
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn initialize(&self) -> JobsStoreResult<()> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(
            "
            create table if not exists jobs (
              id text primary key,
              type text not null,
              status text not null,
              project_id text,
              project_name text,
              payload_json text not null,
              result_json text not null default '{}',
              requested_gpu text not null default 'auto',
              assigned_gpu text,
              worker_id text,
              progress real not null default 0,
              stage text not null default 'queued',
              message text not null default '',
              error text,
              eta_seconds real,
              attempts integer not null default 1,
              source_job_id text,
              duplicate_of_job_id text,
              cancel_requested integer not null default 0,
              created_at text not null,
              updated_at text not null,
              started_at text,
              completed_at text,
              canceled_at text,
              last_heartbeat_at text
            );

            create index if not exists idx_jobs_status_created
              on jobs(status, created_at);
            create index if not exists idx_jobs_project_created
              on jobs(project_id, created_at);
            create index if not exists idx_jobs_assigned_gpu_status
              on jobs(assigned_gpu, status);

            create table if not exists workers (
              id text primary key,
              gpu_id text not null,
              gpu_name text,
              status text not null,
              current_job_id text,
              capabilities_json text not null,
              loaded_models_json text not null,
              utilization_json text,
              registered_at text not null,
              last_seen_at text not null
            );
            ",
        )?;
        ensure_column(&transaction, "workers", "utilization_json", "text")?;
        // sc-2086: per-job peak GPU memory % and load %, written by the worker
        // along with progress so a completed row shows the peak the run reached.
        ensure_column(&transaction, "jobs", "peak_gpu_memory_pct", "real")?;
        ensure_column(&transaction, "jobs", "peak_gpu_load_pct", "real")?;
        // Runtime backend label written by the worker ("mlx" / "mps" / "cuda"
        // / "cpu"). First-non-null wins so the WorkerProgressCard's arch pill
        // stays stable across the run.
        ensure_column(&transaction, "jobs", "backend", "text")?;
        // Structured per-run generation metrics (epic 10402). A companion table
        // keyed 1:1 by job id — kept out of the hot `jobs` row so the queue
        // read path stays lean. Written by the worker on completion and read
        // back by the Generation Stats views. Every settings/timing/hardware
        // column is nullable so any job type populates only what applies.
        transaction.execute_batch(
            "
            create table if not exists generation_metrics (
              job_id text primary key,
              model text,
              quant_label text,
              quant_bits integer,
              sampler text,
              scheduler text,
              scheduler_shift real,
              steps integer,
              image_count integer,
              guidance_scale real,
              true_cfg_scale real,
              guidance_method text,
              use_pid integer,
              pid_target text,
              width integer,
              height integer,
              seed integer,
              loras_json text,
              load_ms integer,
              sample_ms integer,
              decode_ms integer,
              total_ms integer,
              peak_memory_bytes integer,
              peak_memory_pct real,
              peak_gpu_load_pct real,
              backend text,
              updated_at text not null
            );

            create index if not exists idx_genmetrics_model
              on generation_metrics(model);
            create index if not exists idx_genmetrics_quant
              on generation_metrics(quant_label);
            ",
        )?;
        // Batch size per job (epic 10402, sc-10426) — added after the table shipped,
        // so back-fill the column on existing generation_metrics tables.
        ensure_column(&transaction, "generation_metrics", "image_count", "integer")?;
        transaction.commit()?;
        Ok(())
    }

    pub fn mark_interrupted_on_startup(&self) -> JobsStoreResult<Vec<JobSnapshot>> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let interrupted = self.list_jobs_by_status_on_connection(&transaction, ACTIVE_STATUSES)?;
        // A `pending_caption` job (sc-9120) is owned by an API-side background task, not a worker,
        // so an API restart LOSES its caption watcher — the row would otherwise sit un-claimable
        // forever (it is not `queued`, so no worker claims it, and it is not an ACTIVE status, so
        // the interrupt sweep above skips it). RECOVER it instead of failing it: promote it to
        // `queued` with its ORIGINAL prompt (the payload was never rewritten), so the job still
        // dispatches and the worker's format-guard + reseed net produces a render. Degrading is
        // strictly better than interrupting: the user's job survives the restart.
        let stranded_pending: Vec<JobSnapshot> =
            self.list_jobs_by_status_on_connection(&transaction, &["pending_caption"])?;
        let stranded_pending_ids = stranded_pending
            .iter()
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        let interrupted_ids = interrupted
            .iter()
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        let now = utc_now();
        transaction.execute(
            &format!(
                "
            update jobs
               set status = 'interrupted',
                   stage = 'interrupted',
                   message = 'Job was interrupted by a backend restart.',
                   error = 'The backend restarted before this job finished.',
                   completed_at = ?1,
                   updated_at = ?1,
                   worker_id = null
             where status in ({active})
            ",
                active = active_statuses_sql()
            ),
            params![now],
        )?;
        transaction.execute(
            "
            update jobs
               set status = 'queued',
                   stage = 'queued',
                   message = 'Waiting for an available worker.',
                   updated_at = ?1
             where status = 'pending_caption'
            ",
            params![now],
        )?;
        transaction.execute(
            "update workers set status = 'offline', current_job_id = null where status != 'offline'",
            [],
        )?;
        let updated_jobs = interrupted_ids
            .iter()
            .chain(stranded_pending_ids.iter())
            .map(|job_id| self.get_job_on_connection(&transaction, job_id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(updated_jobs)
    }

    pub fn create_job(&self, request: CreateJob) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.create_job_on_connection(&transaction, request, None)?;
        transaction.commit()?;
        Ok(job)
    }

    /// Create a job under a caller-supplied id. Used when the payload must
    /// reference its own job id before insertion — e.g. a `lora_train` job whose
    /// resolved [`crate::training::TrainingPlan`] embeds `jobId`/`sourceJobId`.
    /// The id must be unique; a collision surfaces as a SQLite error.
    pub fn create_job_with_id(
        &self,
        id: String,
        request: CreateJob,
    ) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.create_job_on_connection(&transaction, request, Some(id))?;
        transaction.commit()?;
        Ok(job)
    }

    /// Promote a `pending_caption` job to `queued`, optionally rewriting its payload first
    /// (sc-9120). This is the ONE method that patches a created job's payload: the Ideogram 4
    /// auto-caption background task calls it with `Some(new_payload)` once the magic-prompt
    /// expansion lands (rewriting `payload.prompt` to the rich caption), or with `None` to
    /// degrade the job to `queued` with its original prompt when the expansion is
    /// unavailable/times out — either way the job becomes claimable and the worker's
    /// format-guard + reseed net remains the fallback.
    ///
    /// Race-free by construction: it runs under `BEGIN IMMEDIATE` and the UPDATE is guarded by
    /// `status = 'pending_caption'`, so if the job was canceled (→ `canceled`) or already
    /// recovered on a restart (→ `queued`) in the meantime, the UPDATE matches zero rows and the
    /// method reports `promoted = false` WITHOUT clobbering the newer status. The returned
    /// snapshot always reflects the row's current state.
    ///
    /// `new_payload` fully REPLACES the stored payload (the caller reads the current payload,
    /// rewrites `prompt`, and passes the whole object back), matching how `retry`/`duplicate`
    /// carry a full payload — there is no partial-merge ambiguity.
    pub fn promote_pending_caption_job(
        &self,
        job_id: &str,
        new_payload: Option<Map<String, Value>>,
    ) -> JobsStoreResult<PendingCaptionPromotion> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = utc_now();
        let affected = match new_payload {
            Some(payload) => transaction.execute(
                "
                update jobs
                   set payload_json = ?1,
                       status = 'queued',
                       stage = 'queued',
                       message = 'Waiting for an available worker.',
                       updated_at = ?2
                 where id = ?3 and status = 'pending_caption'
                ",
                params![dumps(&payload)?, now, job_id],
            )?,
            None => transaction.execute(
                "
                update jobs
                   set status = 'queued',
                       stage = 'queued',
                       message = 'Waiting for an available worker.',
                       updated_at = ?1
                 where id = ?2 and status = 'pending_caption'
                ",
                params![now, job_id],
            )?,
        };
        let job = self.get_job_on_connection(&transaction, job_id)?;
        transaction.commit()?;
        Ok(PendingCaptionPromotion {
            promoted: affected > 0,
            job,
        })
    }

    /// Find an in-flight (non-terminal) `prompt_refine` job whose payload matches the given
    /// `prompt` + `aspect_ratio`, so a repeated Ideogram auto-caption (an impatient client
    /// re-POSTing the same image job) can REUSE an already-running magic-prompt expansion instead
    /// of stacking a fresh refine job every time (sc-9120 acceptance: retries can't pile up
    /// unbounded refine jobs). Returns the newest such job, or `None` when none is in flight.
    ///
    /// Read-only single-SELECT: no write mutex, relies on WAL reader isolation like `list_jobs`
    /// (sc-8950 / F-148). Matching is by the expander's two inputs — the raw `prompt` and the
    /// reduced `aspectRatio` label — which are exactly what `enqueue_magic_prompt_job` writes, so
    /// two requests that would produce the same expansion collapse onto one refine job.
    pub fn find_reusable_prompt_refine_job(
        &self,
        prompt: &str,
        aspect_ratio: &str,
    ) -> JobsStoreResult<Option<JobSnapshot>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(&format!(
            "
            select * from jobs
             where type = 'prompt_refine'
               and status not in ({terminal})
             order by created_at desc
            ",
            terminal = terminal_statuses_sql()
        ))?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        Ok(candidates.into_iter().find(|job| {
            let payload = &job.payload;
            payload.get("task").and_then(Value::as_str) == Some("magic_prompt")
                && payload.get("prompt").and_then(Value::as_str) == Some(prompt)
                && payload.get("aspectRatio").and_then(Value::as_str) == Some(aspect_ratio)
        }))
    }

    pub fn list_jobs(
        &self,
        project_id: Option<&str>,
        status: Option<&str>,
        limit: u32,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        // Read-only, single-SELECT method: it deliberately does NOT take the
        // process-wide write mutex (sc-8950 / F-148). connect() runs in WAL mode
        // (see `connect`), where a reader takes a consistent snapshot and runs
        // concurrently with an in-flight writer instead of blocking on it. The
        // mutex exists only to serialize WRITES across our own connections; a
        // pure read never mutates and never needs it, so keeping it here would
        // pointlessly stall list/get/summary traffic behind every claim or
        // progress update. All mutating methods still hold the mutex.
        let connection = self.connect()?;
        let limit = limit.clamp(1, 500);
        let mut conditions: Vec<&str> = Vec::new();
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(project_id) = project_id {
            conditions.push("project_id = ?");
            bindings.push(Box::new(project_id.to_owned()));
        }
        if let Some(status) = status {
            conditions.push("status = ?");
            bindings.push(Box::new(status.to_owned()));
        }
        let mut sql = String::from("select * from jobs");
        if !conditions.is_empty() {
            sql.push_str(" where ");
            sql.push_str(&conditions.join(" and "));
        }
        sql.push_str(" order by created_at desc limit ?");
        bindings.push(Box::new(limit));
        let mut statement = connection.prepare(&sql)?;
        let jobs =
            collect_jobs(statement.query_map(params_from_iter(bindings.iter()), row_to_job)?)?;
        Ok(jobs)
    }

    pub fn get_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.connect()?;
        self.get_job_on_connection(&connection, job_id)
    }

    /// Upsert the structured generation metrics for a job (epic 10402). Called
    /// by the worker on completion via `POST /api/v1/jobs/:id/metrics`. Merges
    /// with any existing row via `coalesce(excluded, existing)` so a partial
    /// second report never wipes a field a prior report set. Holds the write
    /// mutex like every other mutating method.
    pub fn upsert_generation_metrics(
        &self,
        job_id: &str,
        metrics: &GenerationMetrics,
    ) -> JobsStoreResult<()> {
        let _guard = self.lock.lock();
        let connection = self.connect()?;
        let now = utc_now();
        let loras_json = optional_dumps(metrics.loras.as_ref())?;
        connection.execute(
            "
            insert into generation_metrics (
                job_id, model, quant_label, quant_bits, sampler, scheduler,
                scheduler_shift, steps, guidance_scale, true_cfg_scale,
                guidance_method, use_pid, pid_target, width, height, seed,
                loras_json, load_ms, sample_ms, decode_ms, total_ms,
                peak_memory_bytes, peak_memory_pct, peak_gpu_load_pct, backend,
                image_count, updated_at
            ) values (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
            )
            on conflict(job_id) do update set
                model = coalesce(excluded.model, generation_metrics.model),
                quant_label = coalesce(excluded.quant_label, generation_metrics.quant_label),
                quant_bits = coalesce(excluded.quant_bits, generation_metrics.quant_bits),
                sampler = coalesce(excluded.sampler, generation_metrics.sampler),
                scheduler = coalesce(excluded.scheduler, generation_metrics.scheduler),
                scheduler_shift = coalesce(excluded.scheduler_shift, generation_metrics.scheduler_shift),
                steps = coalesce(excluded.steps, generation_metrics.steps),
                guidance_scale = coalesce(excluded.guidance_scale, generation_metrics.guidance_scale),
                true_cfg_scale = coalesce(excluded.true_cfg_scale, generation_metrics.true_cfg_scale),
                guidance_method = coalesce(excluded.guidance_method, generation_metrics.guidance_method),
                use_pid = coalesce(excluded.use_pid, generation_metrics.use_pid),
                pid_target = coalesce(excluded.pid_target, generation_metrics.pid_target),
                width = coalesce(excluded.width, generation_metrics.width),
                height = coalesce(excluded.height, generation_metrics.height),
                seed = coalesce(excluded.seed, generation_metrics.seed),
                loras_json = coalesce(excluded.loras_json, generation_metrics.loras_json),
                load_ms = coalesce(excluded.load_ms, generation_metrics.load_ms),
                sample_ms = coalesce(excluded.sample_ms, generation_metrics.sample_ms),
                decode_ms = coalesce(excluded.decode_ms, generation_metrics.decode_ms),
                total_ms = coalesce(excluded.total_ms, generation_metrics.total_ms),
                peak_memory_bytes = coalesce(excluded.peak_memory_bytes, generation_metrics.peak_memory_bytes),
                peak_memory_pct = coalesce(excluded.peak_memory_pct, generation_metrics.peak_memory_pct),
                peak_gpu_load_pct = coalesce(excluded.peak_gpu_load_pct, generation_metrics.peak_gpu_load_pct),
                backend = coalesce(excluded.backend, generation_metrics.backend),
                image_count = coalesce(excluded.image_count, generation_metrics.image_count),
                updated_at = excluded.updated_at
            ",
            params![
                job_id,
                metrics.model,
                metrics.quant_label,
                metrics.quant_bits,
                metrics.sampler,
                metrics.scheduler,
                metrics.scheduler_shift.as_ref().and_then(Number::as_f64),
                metrics.steps,
                metrics.guidance_scale.as_ref().and_then(Number::as_f64),
                metrics.true_cfg_scale.as_ref().and_then(Number::as_f64),
                metrics.guidance_method,
                metrics.use_pid,
                metrics.pid_target,
                metrics.width,
                metrics.height,
                metrics.seed,
                loras_json,
                metrics.load_ms,
                metrics.sample_ms,
                metrics.decode_ms,
                metrics.total_ms,
                metrics.peak_memory_bytes,
                metrics.peak_memory_pct.as_ref().and_then(Number::as_f64),
                metrics.peak_gpu_load_pct.as_ref().and_then(Number::as_f64),
                metrics.backend,
                metrics.image_count,
                now,
            ],
        )?;
        Ok(())
    }

    /// Read the structured metrics for a single job (epic 10402). Returns None
    /// when the job predates metrics capture or never recorded any (e.g. an old
    /// row). Read-only — no write mutex (WAL reader isolation, see `list_jobs`).
    pub fn get_generation_metrics(
        &self,
        job_id: &str,
    ) -> JobsStoreResult<Option<GenerationMetrics>> {
        let connection = self.connect()?;
        let metrics = connection
            .query_row(
                "select * from generation_metrics where job_id = ?1",
                params![job_id],
                row_to_generation_metrics,
            )
            .optional()?;
        Ok(metrics)
    }

    /// Aggregate metrics feed for the comparison charts (epic 10402): every
    /// metrics row joined to its job's identity, newest first, optionally
    /// filtered by job type / model / quant. Read-only — no write mutex.
    pub fn list_generation_metrics(
        &self,
        job_type: Option<&str>,
        model: Option<&str>,
        quant_label: Option<&str>,
        limit: u32,
    ) -> JobsStoreResult<Vec<GenerationMetricsRow>> {
        let connection = self.connect()?;
        let limit = limit.clamp(1, 5000);
        let mut conditions: Vec<&str> = Vec::new();
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(job_type) = job_type {
            conditions.push("j.type = ?");
            bindings.push(Box::new(job_type.to_owned()));
        }
        if let Some(model) = model {
            conditions.push("m.model = ?");
            bindings.push(Box::new(model.to_owned()));
        }
        if let Some(quant_label) = quant_label {
            conditions.push("m.quant_label = ?");
            bindings.push(Box::new(quant_label.to_owned()));
        }
        let mut sql = String::from(
            "select m.*, j.type as j_type, j.status as j_status, \
             j.project_id as j_project_id, j.created_at as j_created_at \
             from generation_metrics m join jobs j on j.id = m.job_id",
        );
        if !conditions.is_empty() {
            sql.push_str(" where ");
            sql.push_str(&conditions.join(" and "));
        }
        sql.push_str(" order by j.created_at desc limit ?");
        bindings.push(Box::new(limit));
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(
            params_from_iter(bindings.iter()),
            row_to_generation_metrics_row,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn cancel_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(job.status.as_str()) {
            return Ok(job);
        }

        let now = utc_now();
        // A `queued` OR `pending_caption` job has no worker to acknowledge the cancel, so it
        // goes straight to terminal `canceled` here. `pending_caption` (sc-9120) shares this
        // fast path: no worker owns it, and its background caption watcher promotes only a row
        // that is STILL `pending_caption` (a race-free guarded UPDATE), so it can't resurrect a
        // just-canceled job. Any active (worker-owned) status falls to the cooperative branch
        // below that requests acknowledgement.
        if job.status == JobStatus::Queued || job.status == JobStatus::PendingCaption {
            transaction.execute(
                "
                update jobs
                   set status = 'canceled',
                       stage = 'canceled',
                       progress = 1,
                       cancel_requested = 1,
                       message = 'Canceled before a worker started.',
                       canceled_at = ?1,
                       completed_at = ?1,
                       updated_at = ?1
                 where id = ?2
                ",
                params![now, job_id],
            )?;
        } else {
            transaction.execute(
                "
                update jobs
                   set cancel_requested = 1,
                       message = 'Cancellation requested. Waiting for worker acknowledgement.',
                       updated_at = ?1
                 where id = ?2
                ",
                params![now, job_id],
            )?;
        }
        let job = self.get_job_on_connection(&transaction, job_id)?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn retry_job(&self, job_id: &str, request: RetryJob) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if job.attempts >= MAX_JOB_ATTEMPTS {
            return Err(JobsStoreError::RetryLimit {
                max_attempts: MAX_JOB_ATTEMPTS,
            });
        }
        let mut payload = job.payload;
        payload.extend(request.payload_changes);
        let job = self.create_job_on_connection(
            &transaction,
            CreateJob {
                job_type: job.job_type,
                project_id: job.project_id,
                project_name: job.project_name,
                payload,
                requested_gpu: job.requested_gpu,
                source_job_id: Some(job.id),
                duplicate_of_job_id: None,
                attempts: job.attempts + 1,
                // A retry re-enters the queue claimable: its payload is whatever the original
                // ran with (already caption-rewritten if it was an Ideogram auto-caption job),
                // so it never re-enters `pending_caption` (sc-9120).
                initial_status: None,
            },
            None,
        )?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn duplicate_job(
        &self,
        job_id: &str,
        request: DuplicateJob,
    ) -> JobsStoreResult<JobSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        let mut payload = job.payload;
        payload.extend(request.payload_changes);
        let job = self.create_job_on_connection(
            &transaction,
            CreateJob {
                job_type: job.job_type,
                project_id: job.project_id,
                project_name: job.project_name,
                payload,
                requested_gpu: request.requested_gpu.unwrap_or(job.requested_gpu),
                source_job_id: None,
                duplicate_of_job_id: Some(job.id),
                attempts: 1,
                // A duplicate copies the (already-rewritten) payload and re-enters the queue
                // claimable — never `pending_caption` (sc-9120).
                initial_status: None,
            },
            None,
        )?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn register_worker(&self, request: RegisterWorker) -> JobsStoreResult<WorkerSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = utc_now();
        transaction.execute(
            "
            insert into workers (
              id, gpu_id, gpu_name, status, current_job_id, capabilities_json,
              loaded_models_json, utilization_json, registered_at, last_seen_at
            ) values (?1, ?2, ?3, 'idle', null, ?4, ?5, ?6, ?7, ?7)
            on conflict(id) do update set
              gpu_id = excluded.gpu_id,
              gpu_name = excluded.gpu_name,
              status = case when workers.current_job_id is null then 'idle' else workers.status end,
              capabilities_json = excluded.capabilities_json,
              loaded_models_json = excluded.loaded_models_json,
              utilization_json = excluded.utilization_json,
              last_seen_at = excluded.last_seen_at
            ",
            params![
                request.worker_id,
                request.gpu_id,
                request.gpu_name,
                dumps(&request.capabilities)?,
                dumps(&request.loaded_models)?,
                optional_dumps(request.utilization.as_ref())?,
                now,
            ],
        )?;
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        transaction.commit()?;
        Ok(worker)
    }

    pub fn heartbeat_worker(&self, request: WorkerHeartbeat) -> JobsStoreResult<WorkerSnapshot> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        let now = utc_now();
        if request.current_job_id.is_none() {
            if let Some(previous_job_id) = worker.current_job_id {
                let previous_job = self.get_job_on_connection(&transaction, &previous_job_id)?;
                // Only interrupt a worker's previous active job on an idle heartbeat
                // if that job has already heartbeated at least once. A job that was
                // *just* claimed (no heartbeat yet) may be one another incarnation of
                // the same worker_id claimed microseconds ago — an idle heartbeat
                // racing the claim must not kill it. The time-based stale sweep still
                // reclaims a job abandoned before its first heartbeat.
                if is_active_status(previous_job.status.as_str())
                    && previous_job.last_heartbeat_at.is_some()
                {
                    transaction.execute(
                        "
                        update jobs
                           set status = 'interrupted',
                               stage = 'interrupted',
                               message = 'Job was interrupted after its worker restarted.',
                               error = 'Worker heartbeat no longer referenced the active job.',
                               completed_at = ?1,
                               updated_at = ?1,
                               worker_id = null
                         where id = ?2
                        ",
                        params![now, previous_job_id],
                    )?;
                }
            }
        }
        transaction.execute(
            "
            update workers
               set status = ?1,
                   current_job_id = ?2,
                   loaded_models_json = ?3,
                   utilization_json = ?4,
                   last_seen_at = ?5
             where id = ?6
            ",
            params![
                request.status.as_str(),
                request.current_job_id,
                dumps(&request.loaded_models)?,
                optional_dumps(request.utilization.as_ref())?,
                now,
                request.worker_id,
            ],
        )?;
        if let Some(job_id) = request.current_job_id {
            // Verify ownership before letting a heartbeat refresh the job's
            // liveness timestamps (sc-8873 / F-071). The progress path was
            // hardened this way in sc-4172, but the heartbeat wasn't: a stale
            // worker still heartbeating an old `current_job_id` it no longer
            // owns (the job was swept to `interrupted` — worker_id cleared — or
            // reclaimed by another worker) would keep bumping last_heartbeat_at,
            // masking the job as alive and blocking the time-based stale sweep
            // from ever reclaiming it. Scoping the UPDATE to the reporting
            // worker's own rows means a non-owning heartbeat is a silent no-op.
            transaction.execute(
                "update jobs set last_heartbeat_at = ?1, updated_at = ?1 \
                 where id = ?2 and worker_id = ?3",
                params![now, job_id, request.worker_id],
            )?;
        }
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        transaction.commit()?;
        Ok(worker)
    }

    pub fn mark_stale_workers_interrupted(
        &self,
        timeout_seconds: u64,
    ) -> JobsStoreResult<StaleSweep> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let timeout = i64::try_from(timeout_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(timeout));
        let now_text = format_unix_seconds(now);
        let mut statement = transaction.prepare(
            "
            select * from workers
             where status != 'offline'
               and last_seen_at < ?1
            ",
        )?;
        let stale_workers = collect_workers(statement.query_map(params![cutoff], row_to_worker)?)?;
        if stale_workers.is_empty() {
            return Ok(StaleSweep {
                workers: Vec::new(),
                jobs: Vec::new(),
            });
        }

        let worker_ids = stale_workers
            .iter()
            .map(|worker| worker.id.clone())
            .collect::<Vec<_>>();
        drop(statement);
        let active_jobs = self.active_jobs_for_workers(&transaction, &worker_ids)?;
        let placeholders = placeholders_from(2, worker_ids.len());
        let mut job_params = vec![now_text.as_str()];
        job_params.extend(worker_ids.iter().map(String::as_str));
        transaction.execute(
            &format!(
                "
                update jobs
                   set status = 'interrupted',
                       stage = 'interrupted',
                       message = 'Lost contact with the worker.',
                       error = 'No heartbeat from the worker for {timeout_seconds}s. The worker may have crashed, hung, or lost its connection to the app. If it reconnects you can retry the job; if this keeps happening, check System → Logs.',
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where worker_id in ({placeholders})
                   and status in ({active})
                ",
                active = active_statuses_sql()
            ),
            params_from_iter(job_params),
        )?;

        let mut worker_params = vec![now_text.as_str()];
        worker_params.extend(worker_ids.iter().map(String::as_str));
        transaction.execute(
            &format!(
                "
                update workers
                   set status = 'offline',
                       current_job_id = null,
                       last_seen_at = ?1
                 where id in ({placeholders})
                "
            ),
            params_from_iter(worker_params),
        )?;

        let updated_workers = self.workers_by_ids(&transaction, &worker_ids)?;
        let updated_jobs = active_jobs
            .iter()
            .map(|job| self.get_job_on_connection(&transaction, &job.id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(StaleSweep {
            workers: updated_workers,
            jobs: updated_jobs,
        })
    }

    /// Surface a worker's abnormal death — killed by an uncatchable signal
    /// (SIGKILL/OOM, SIGABRT, SIGSEGV, …) or exited on its own with a non-zero
    /// status (e.g. a Rust panic, exit code 101) — as a terminal job FAILURE,
    /// instead of letting the heartbeat sweep later mark it the generic
    /// `interrupted` (which reads to the user like a frozen progress bar). The
    /// supervisor that reaped the child observes the termination — the only layer
    /// that can, since the death is uncatchable in-process — and calls this with
    /// the signal (when killed) or exit code (when it self-exited non-zero); a
    /// clean exit-0 is graceful and is never reported here. We fail the worker's
    /// still-active job with an actionable, attributed error and release the worker
    /// so the UI doesn't show it pinned to a dead job. Returns the failed job if
    /// the worker had an active one (else `None` — it died idle between jobs).
    /// (sc-4881 signals; sc-6320 non-signal exits)
    pub fn fail_worker_job_terminated(
        &self,
        worker_id: &str,
        signal: Option<i32>,
        exit_code: Option<i32>,
    ) -> JobsStoreResult<Option<JobSnapshot>> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = utc_now();
        let worker_ids = [worker_id.to_owned()];
        let active_jobs = self.active_jobs_for_workers(&transaction, &worker_ids)?;
        let mut failed = None;
        if let Some(job) = active_jobs.into_iter().next() {
            // Tailor the OOM/signal hint to the dead job's kind so the guidance is
            // actionable (sc-5567): an image-batch SIGKILL points at count/resolution,
            // not the training-only gradient-checkpointing remediation.
            let error = termination_failure_error(signal, exit_code, Some(&job.job_type));
            transaction.execute(
                &format!(
                    "
                    update jobs
                       set status = 'failed',
                           stage = 'failed',
                           message = 'Worker process terminated unexpectedly.',
                           error = ?2,
                           completed_at = ?1,
                           updated_at = ?1,
                           worker_id = null
                     where id = ?3
                       and status in ({active})
                    ",
                    active = active_statuses_sql()
                ),
                params![now, error, job.id],
            )?;
            failed = Some(self.get_job_on_connection(&transaction, &job.id)?);
        }
        // Release the worker so it isn't shown pinned to a now-failed job; the
        // supervisor restarts the child, which re-registers itself fresh.
        transaction.execute(
            "
            update workers
               set status = 'offline',
                   current_job_id = null,
                   last_seen_at = ?1
             where id = ?2
            ",
            params![now, worker_id],
        )?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-required" grace sweep (epic 3482 / sc-3483). When `mlx_required`, the
    /// non-mlx (MPS) worker never claims an MLX-eligible job — it defers unconditionally
    /// to the in-process `mlx` worker (see `should_defer_*`). If no **live** `mlx` worker
    /// claims such a job within the grace window — because the worker is down, never
    /// started, or has been crashed longer than the supervisor's auto-restart can
    /// self-heal — the job would otherwise sit queued forever. This fails those jobs
    /// terminal (`status = failed`) with an actionable `mlx_unavailable` error naming the
    /// model + job type, so the failure is loud and points at the real gap instead of
    /// silently falling back to MPS.
    ///
    /// "Live `mlx` worker" = a `gpu_id = 'mlx'` worker that is not offline and has
    /// heartbeat within the grace window. While one exists (even if it is merely busy),
    /// this is a no-op and the job waits to be claimed; a transient `mlx` crash that the
    /// supervisor restarts inside the window therefore never fails a job. `grace_seconds`
    /// reuses the stale-worker timeout for exactly that reason.
    ///
    /// Off (`mlx_required == false`) it returns immediately, so Windows/Linux/Docker and
    /// the Mac build before the final cutover (sc-3492) are completely unaffected. Returns
    /// the jobs it failed so the caller can surface the structured event in System → Logs
    /// and publish their updates.
    pub fn fail_stranded_mlx_jobs(
        &self,
        mlx_required: bool,
        grace_seconds: u64,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if !mlx_required {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let grace = i64::try_from(grace_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(grace));

        // A live `mlx` worker (not offline, heartbeat within the window) means MLX-eligible
        // jobs should wait for it — it may simply be busy. Only when none has checked in
        // within the window do we treat MLX as unavailable and fail the stranded jobs.
        let live_mlx_worker = transaction
            .query_row(
                "
                select 1 from workers
                 where gpu_id = 'mlx'
                   and status != 'offline'
                   and last_seen_at >= ?1
                 limit 1
                ",
                params![cutoff],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();
        if live_mlx_worker {
            return Ok(Vec::new());
        }

        // Candidates: still queued and old enough to have outlived the grace window. A job
        // newer than the cutoff keeps waiting (bounded), so a job created mid-outage isn't
        // failed instantly — it gets the full window for an `mlx` worker to appear.
        let mut statement = transaction.prepare(
            "
            select * from jobs
             where status = 'queued'
               and created_at < ?1
             order by created_at asc
            ",
        )?;
        let candidates = collect_jobs(statement.query_map(params![cutoff], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now);
        let mut failed_ids = Vec::new();
        for job in candidates {
            if !job_is_any_mlx_eligible(&job) {
                continue;
            }
            let error = mlx_unavailable_error(&job, grace_seconds);
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'MLX worker unavailable.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, error, job.id],
            )?;
            failed_ids.push(job.id.clone());
        }
        let failed = failed_ids
            .iter()
            .map(|id| self.get_job_on_connection(&transaction, id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-unsupported" enforce sweep (epic 3482 / sc-3484). When `mlx_required` AND
    /// `enforce`, fails every queued job the Rust/MLX flow can't run (`mac_rust_supported`
    /// returns `Err`) terminal with a feature-precise `mlx_unsupported` error — the forcing
    /// function that turns "still on torch" into a loud, named failure instead of a silent
    /// fallback. Unlike the stranded sweep there is no grace window: an unsupported job is
    /// permanently unsupported until its surface is ported or dropped, so it fails immediately.
    ///
    /// Default mode is **warn** (`enforce == false`) → this is a no-op and the gap is logged
    /// at claim time instead (the job still runs on torch), so flipping `mlx_required` on for
    /// observation surfaces the gap list without breaking anything. Off (`!mlx_required`) →
    /// immediate no-op, so Windows/Linux/Docker are unaffected. MLX-*eligible* jobs are
    /// `Ok` here and handled by `fail_stranded_mlx_jobs`/routing — the two sweeps partition
    /// the queue and never touch the same job. Returns `(job, reason)` pairs so the caller can
    /// emit the structured event.
    pub fn fail_unsupported_mlx_jobs(
        &self,
        mlx_required: bool,
        enforce: bool,
    ) -> JobsStoreResult<Vec<(JobSnapshot, UnsupportedReason)>> {
        if !mlx_required || !enforce {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut statement = transaction
            .prepare("select * from jobs where status = 'queued' order by created_at asc")?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now_unix_seconds());
        let mut failed = Vec::new();
        for job in candidates {
            let Err(reason) = mac_rust_supported(&job) else {
                continue;
            };
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'Not supported by the MLX flow on macOS.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, reason.error_message(), job.id],
            )?;
            let updated = self.get_job_on_connection(&transaction, &job.id)?;
            failed.push((updated, reason));
        }
        transaction.commit()?;
        Ok(failed)
    }

    /// Off-Mac candle grace sweep (sc-5502, epic 5483) — the Windows/Linux twin of
    /// [`Self::fail_stranded_mlx_jobs`]. When `candle_required`, fails any candle-eligible job left
    /// queued past the grace window when no live candle worker exists, terminal with
    /// `candle_unavailable` — so a retired-torch deployment fails loudly instead of queuing forever.
    ///
    /// "Live candle worker" = a worker advertising the `candle` marker capability that is not
    /// offline and has a heartbeat within `grace_seconds` (the marker is a fixed JSON string in
    /// `capabilities_json`, matched as a substring — the candle worker runs on a real CUDA gpu
    /// index, not the `mlx` sentinel, so it can't be matched by `gpu_id`; see [`worker_is_candle`]).
    /// While one exists (even merely busy) this is a no-op and candle-eligible jobs wait, so a
    /// transient candle crash the supervisor restarts inside the window never fails a job. Off
    /// (`!candle_required`) it returns immediately, so a deployment still keeping the Python torch
    /// worker is completely unaffected. Returns the jobs it failed so the caller can surface the
    /// structured event and publish their updates.
    pub fn fail_stranded_candle_jobs(
        &self,
        candle_required: bool,
        grace_seconds: u64,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if !candle_required {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let grace = i64::try_from(grace_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(grace));

        // A live candle worker means candle-eligible jobs should wait for it — it may simply be
        // busy. Only when none has checked in within the window do we treat candle as unavailable
        // and fail the stranded jobs.
        let live_candle_worker = transaction
            .query_row(
                "
                select 1 from workers
                 where status != 'offline'
                   and last_seen_at >= ?1
                   and capabilities_json like '%\"candle\"%'
                 limit 1
                ",
                params![cutoff],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();
        if live_candle_worker {
            return Ok(Vec::new());
        }

        // Candidates: still queued and old enough to have outlived the grace window. A job newer
        // than the cutoff keeps waiting (bounded), so a job created mid-outage isn't failed
        // instantly — it gets the full window for a candle worker to appear.
        let mut statement = transaction.prepare(
            "
            select * from jobs
             where status = 'queued'
               and created_at < ?1
             order by created_at asc
            ",
        )?;
        let candidates = collect_jobs(statement.query_map(params![cutoff], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now);
        let mut failed_ids = Vec::new();
        for job in candidates {
            if !job_is_any_candle_eligible(&job) {
                continue;
            }
            let error = candle_unavailable_error(&job, grace_seconds);
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'Candle worker unavailable.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, error, job.id],
            )?;
            failed_ids.push(job.id.clone());
        }
        let failed = failed_ids
            .iter()
            .map(|id| self.get_job_on_connection(&transaction, id))
            .collect::<JobsStoreResult<Vec<_>>>()?;
        transaction.commit()?;
        Ok(failed)
    }

    /// Off-Mac "candle-unsupported" enforce sweep (sc-5502, epic 5483) — the Windows/Linux twin of
    /// [`Self::fail_unsupported_mlx_jobs`]. When `candle_required` AND `enforce`, fails every queued
    /// job the candle/CUDA flow can't run ([`candle_supported`] returns `Err`) terminal with a
    /// feature-precise `candle_unsupported` error — the forcing function that turns "still on torch"
    /// into a loud, named failure instead of a silent fallback. Unlike the stranded sweep there is
    /// no grace window: an unsupported job is permanently unsupported until its surface is ported or
    /// dropped, so it fails immediately.
    ///
    /// Default mode is **warn** (`enforce == false`) → no-op, and the gap is logged at claim time
    /// instead (the job still runs on torch), so flipping `candle_required` on for observation
    /// surfaces the gap list without breaking anything. Off (`!candle_required`) → immediate no-op,
    /// so a deployment still keeping the torch worker is unaffected. Candle-*eligible* jobs are `Ok`
    /// here and handled by routing / [`Self::fail_stranded_candle_jobs`] — the two sweeps partition
    /// the queue and never touch the same job. Returns `(job, reason)` pairs so the caller can emit
    /// the structured event.
    pub fn fail_unsupported_candle_jobs(
        &self,
        candle_required: bool,
        enforce: bool,
    ) -> JobsStoreResult<Vec<(JobSnapshot, UnsupportedReason)>> {
        if !candle_required || !enforce {
            return Ok(Vec::new());
        }
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut statement = transaction
            .prepare("select * from jobs where status = 'queued' order by created_at asc")?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now_unix_seconds());
        let mut failed = Vec::new();
        for job in candidates {
            let Err(reason) = candle_supported(&job) else {
                continue;
            };
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'Not supported by the candle/CUDA flow off-Mac.',
                       error = ?2,
                       completed_at = ?1,
                       updated_at = ?1,
                       worker_id = null
                 where id = ?3 and status = 'queued'
                ",
                params![now_text, reason.candle_error_message(), job.id],
            )?;
            let updated = self.get_job_on_connection(&transaction, &job.id)?;
            failed.push((updated, reason));
        }
        transaction.commit()?;
        Ok(failed)
    }

    pub fn claim_next_job(&self, worker_id: &str) -> JobsStoreResult<Option<JobSnapshot>> {
        Ok(self.claim_next_job_routed(worker_id, false)?.0)
    }

    /// Like [`Self::claim_next_job`], but also reports the MLX↔torch routing decision
    /// so the caller (the API claim handler) can log *why* a job landed where it did —
    /// the single most useful line for diagnosing "MLX-eligible job ran on torch"
    /// (sc-3449). A `None` decision means the claim was routing-neutral: no job was
    /// available, an unrelated balancing deferral fired, or the job is one no `mlx`
    /// worker would ever want.
    pub fn claim_next_job_routed(
        &self,
        worker_id: &str,
        mlx_required: bool,
    ) -> JobsStoreResult<(Option<JobSnapshot>, Option<RouteDecision>)> {
        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        // BEGIN IMMEDIATE: take the write lock up front. The claim reads the worker, the
        // active-gpu-job guard and the full queued set before deciding, then writes. A
        // DEFERRED transaction holds only a read lock through those reads and tries to
        // upgrade at the first UPDATE — and SQLite returns SQLITE_BUSY *immediately* on a
        // lock upgrade (busy_timeout does not retry upgrades, to avoid deadlock), so two
        // overlapping claims would race and one would fail. IMMEDIATE serializes claimers.
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let worker = self.get_worker_on_connection(&transaction, worker_id)?;
        let worker_gpu_id = worker.gpu_id.clone();
        let has_active_gpu_job = active_gpu_job_exists(&transaction, &worker.gpu_id)?;

        let mut statement = transaction.prepare(&format!(
            "
            select * from jobs
             where status = 'queued'
               and (type in ({list}) or requested_gpu = 'auto' or requested_gpu = ?1)
               and (?2 = 0 or type in ({list}))
             order by created_at asc
            ",
            list = non_gpu_job_types_sql()
        ))?;
        let queued_rows = collect_jobs(statement.query_map(
            params![worker.gpu_id, i64::from(has_active_gpu_job)],
            row_to_job,
        )?)?;
        // No row cap (sc-1630): choose_claimable_job must see every gpu/type-gated queued row,
        // or a capability-incompatible prefix (e.g. 50+ jobs the worker can't run) would hide a
        // later compatible job and the worker would sit idle. It also needs the whole compatible
        // set for its priority pass (an explicit-GPU / loaded-model job jumps ahead of an earlier
        // auto-GPU one), so a bounded scan can't preserve that anyway. The WHERE above already
        // narrows rows to this worker's gpu/type lane; pushing the capability filter into SQL is
        // the scale lever if queues ever grow large enough for the full scan to matter.
        let queued = choose_claimable_job(queued_rows, &worker);
        let Some(queued) = queued else {
            return Ok((None, None));
        };
        drop(statement);
        if should_defer_auto_gpu_claim(&transaction, &queued, &worker)? {
            return Ok((None, None));
        }
        if should_defer_image_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_video_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_training_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_caption_to_mlx_worker(&transaction, &queued, &worker, mlx_required)?
            || should_defer_understanding_to_mlx_worker(
                &transaction,
                &queued,
                &worker,
                mlx_required,
            )?
        {
            // A non-mlx worker is yielding this MLX-eligible job to an idle mlx worker.
            let decision = RouteDecision::new(
                &queued,
                &worker_gpu_id,
                worker_id,
                "deferred_to_mlx",
                "idle_mlx_available",
            );
            return Ok((None, Some(decision)));
        }

        let assigned_gpu = if is_non_gpu_job_type(queued.job_type.as_str()) {
            "cpu".to_owned()
        } else {
            worker_gpu_id.clone()
        };
        let now = utc_now();
        transaction.execute(
            "
            update jobs
               set status = 'preparing',
                   assigned_gpu = ?1,
                   worker_id = ?2,
                   stage = 'preparing',
                   message = 'Worker claimed job.',
                   started_at = coalesce(started_at, ?3),
                   updated_at = ?3
             where id = ?4 and status = 'queued'
            ",
            params![assigned_gpu, worker_id, now, queued.id],
        )?;
        transaction.execute(
            "update workers set status = 'busy', current_job_id = ?1, last_seen_at = ?2 where id = ?3",
            params![queued.id, now, worker_id],
        )?;
        let job = self.get_job_on_connection(&transaction, &queued.id)?;
        transaction.commit()?;
        let decision = route_decision_for_claim(&queued, &worker);
        Ok((Some(job), decision))
    }

    pub fn update_job_progress(
        &self,
        job_id: &str,
        update: ProgressUpdate,
    ) -> JobsStoreResult<JobSnapshot> {
        if !JOB_STATUSES.contains(&update.status.as_str()) {
            return Err(JobsStoreError::InvalidStatus(
                update.status.as_str().to_owned(),
            ));
        }

        if !update.progress.is_finite() {
            return Err(JobsStoreError::InvalidNumber("progress".to_owned()));
        }
        if update.eta_seconds.is_some_and(|value| !value.is_finite()) {
            return Err(JobsStoreError::InvalidNumber("etaSeconds".to_owned()));
        }
        if update
            .peak_gpu_memory_pct
            .is_some_and(|value| !value.is_finite())
        {
            return Err(JobsStoreError::InvalidNumber("peakGpuMemoryPct".to_owned()));
        }
        if update
            .peak_gpu_load_pct
            .is_some_and(|value| !value.is_finite())
        {
            return Err(JobsStoreError::InvalidNumber("peakGpuLoadPct".to_owned()));
        }

        let _guard = self.lock.lock();
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Guard against zombie-worker writes (sc-4172): a worker that went
        // silent long enough for the stale sweep to mark its job `interrupted`
        // (or whose job the user canceled) must not resurrect it with a late
        // progress report — that's exactly the failure mode the heartbeat
        // machinery exists to handle.
        let current = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(current.status.as_str()) {
            // Idempotent re-report of the same terminal status (e.g. a retried
            // "canceled" POST) succeeds without touching the row.
            if current.status == update.status {
                return Ok(current);
            }
            return Err(JobsStoreError::TerminalJobImmutable {
                job_id: job_id.to_owned(),
                status: current.status.as_str().to_owned(),
            });
        }
        match (update.worker_id.as_deref(), current.worker_id.as_deref()) {
            (Some(reporter), Some(owner)) if reporter == owner => {}
            (None, None) => {}
            _ => {
                return Err(JobsStoreError::NotJobOwner {
                    job_id: job_id.to_owned(),
                });
            }
        }
        let now = utc_now();
        let completed_at = is_terminal_status(update.status.as_str()).then_some(now.clone());
        let canceled_at = (update.status == JobStatus::Canceled).then_some(now.clone());
        let progress = update.progress.clamp(0.0, 1.0);
        // Peaks are clamped to 0..100 and persisted as a running max so a stale
        // progress report (lower sample) can't ratchet the peak down (sc-2086).
        let peak_memory = update
            .peak_gpu_memory_pct
            .map(|value| value.clamp(0.0, 100.0));
        let peak_load = update
            .peak_gpu_load_pct
            .map(|value| value.clamp(0.0, 100.0));
        let mut result = update.result;
        if let Some(result) = result.as_mut() {
            // Reuse the result we already read above (same transaction/row) rather
            // than re-selecting result_json each update (sc-4274 / F-CORE-14).
            merge_training_sample_history(Some(&current.result), result);
        }
        transaction.execute(
            "
            update jobs
               set status = ?1,
                   stage = ?2,
                   progress = ?3,
                   message = ?4,
                   error = ?5,
                   result_json = coalesce(?6, result_json),
                   eta_seconds = ?7,
                   completed_at = coalesce(?8, completed_at),
                   canceled_at = coalesce(?9, canceled_at),
                   updated_at = ?10,
                   peak_gpu_memory_pct = case
                       when ?11 is null then peak_gpu_memory_pct
                       else max(coalesce(peak_gpu_memory_pct, 0), ?11)
                   end,
                   peak_gpu_load_pct = case
                       when ?12 is null then peak_gpu_load_pct
                       else max(coalesce(peak_gpu_load_pct, 0), ?12)
                   end,
                   backend = coalesce(backend, ?13)
             where id = ?14
            ",
            params![
                update.status.as_str(),
                update.stage.as_str(),
                progress,
                update.message,
                update.error,
                optional_dumps(result.as_ref())?,
                update.eta_seconds,
                completed_at,
                canceled_at,
                now,
                peak_memory,
                peak_load,
                update.backend,
                job_id,
            ],
        )?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if is_terminal_status(update.status.as_str()) {
            if let Some(worker_id) = &job.worker_id {
                transaction.execute(
                    "update workers set status = 'idle', current_job_id = null, last_seen_at = ?1 where id = ?2",
                    params![now, worker_id],
                )?;
            }
        }
        transaction.commit()?;
        Ok(job)
    }

    pub fn list_workers(&self) -> JobsStoreResult<Vec<WorkerSnapshot>> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.connect()?;
        let mut statement = connection.prepare("select * from workers order by gpu_id, id")?;
        let workers = collect_workers(statement.query_map([], row_to_worker)?)?;
        Ok(workers)
    }

    pub fn get_worker(&self, worker_id: &str) -> JobsStoreResult<WorkerSnapshot> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.connect()?;
        self.get_worker_on_connection(&connection, worker_id)
    }

    pub fn queue_summary(&self) -> JobsStoreResult<QueueSummary> {
        // Read-only aggregate: several SELECTs (per-status counts + active jobs +
        // workers), no writes, so it takes NO write mutex and relies on WAL
        // reader isolation like the other reads (sc-8950 / F-148). The counts and
        // active-jobs queries run on one connection and list_workers opens its
        // own; a writer committing between them can only make the snapshot a hair
        // fresher, never inconsistent for the operator's queue view. (Before
        // sc-8950 this method took the mutex and had to hoist list_workers out
        // first to dodge a self-deadlock on the non-reentrant mutex; dropping the
        // mutex removes that hazard entirely.)
        let workers = self.list_workers()?;
        let connection = self.connect()?;

        // Per-status counts over the WHOLE table — never a capped/newest-N sample.
        // Filtering an already-capped list silently undercounts once a project
        // exceeds the cap (sc-4208 / F-CORE-4). Seed every known status at 0 so
        // the map shape is stable for callers regardless of what rows exist.
        let mut counts = JOB_STATUSES
            .iter()
            .map(|status| (parse_string_enum::<JobStatus>(status), 0u32))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut statement =
            connection.prepare("select status, count(*) from jobs group by status")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (status, count) = row?;
            // Writes are constrained to JOB_STATUSES so the seeded entry exists;
            // or_insert keeps an unexpected value counted rather than dropped.
            *counts
                .entry(parse_string_enum::<JobStatus>(&status))
                .or_insert(0) += u32::try_from(count).unwrap_or(u32::MAX);
        }

        // Active (non-terminal, includes `queued`) jobs come from a dedicated
        // uncapped query so an old still-queued/running job can't fall out of the
        // newest-N window and become invisible to the operator.
        let mut statement = connection.prepare(&format!(
            "select * from jobs where status not in ({terminal}) order by created_at desc",
            terminal = terminal_statuses_sql()
        ))?;
        let active_jobs = collect_jobs(statement.query_map([], row_to_job)?)?;

        Ok(QueueSummary {
            counts,
            active_jobs,
            workers,
            max_job_attempts: MAX_JOB_ATTEMPTS,
            extra: Default::default(),
        })
    }

    fn connect(&self) -> JobsStoreResult<Connection> {
        if let Some(parent) = self.db_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&self.db_path)?;
        // Wait (instead of failing instantly) when another connection/process holds the
        // database lock. rusqlite's default busy timeout is 0ms, so any cross-process
        // overlap — e.g. a sidecar restart where the old process hasn't fully released the
        // db, or a concurrent claim/heartbeat — surfaces as `database is locked` and the
        // job loses its claim (MLX-eligible jobs then fall through to the torch worker).
        // A 5s wait lets the holder finish; paired with BEGIN IMMEDIATE on write
        // transactions (below), writers queue cleanly rather than deadlocking on lock upgrade.
        connection.busy_timeout(Duration::from_millis(5000))?;
        match connection.pragma_update(None, "journal_mode", "wal") {
            Ok(()) => {}
            Err(error) => {
                // WAL almost always succeeds. When it can't be set, do NOT delete
                // the `-wal`/`-shm` sidecars: they may belong to a live connection
                // in another process, and removing them can corrupt that
                // connection's view. Nor do we silently force `delete` mode — the
                // 5s busy_timeout reasoning above assumes WAL lets writers queue,
                // so a silent drop to rollback-journal would change concurrency
                // semantics for the rest of the process with no signal. Leave the
                // connection in whatever mode it opened with and warn loudly
                // instead (sc-4275 / F-CORE-16).
                tracing::warn!(
                    event = "sqlite_wal_enable_failed",
                    dbPath = %self.db_path.display(),
                    error = %error,
                    "could not enable SQLite WAL mode; continuing in the default rollback-journal \
                     mode — cross-process write concurrency will be more serialized than usual"
                );
            }
        }
        connection.pragma_update(None, "foreign_keys", "on")?;
        Ok(connection)
    }

    fn create_job_on_connection(
        &self,
        connection: &Connection,
        request: CreateJob,
        job_id: Option<String>,
    ) -> JobsStoreResult<JobSnapshot> {
        let requested_gpu = normalize_requested_gpu(&request.requested_gpu);
        if job_requires_gpu(&request.job_type) && requested_gpu == "cpu" {
            return Err(JobsStoreError::InvalidRequestedGpu(format!(
                "{} jobs cannot target CPU workers. Choose auto or a GPU id.",
                request.job_type.as_str()
            )));
        }
        let now = utc_now();
        let job_id = match job_id {
            Some(job_id) => job_id,
            None => {
                // sc-4209 / sc-8888 (F-086): pull the id from the OS CSPRNG via the
                // shared `random_hex` helper instead of a per-call SQLite
                // `hex(randomblob(16))`, which turned id generation into a SQLite
                // failure surface. `random_hex` fails only if the OS CSPRNG does;
                // fold that into `Io` so the caller's error type is unchanged.
                let job_hex = random_hex(16).map_err(|error| {
                    JobsStoreError::Io(std::io::Error::other(error.to_string()))
                })?;
                format!("job_{job_hex}")
            }
        };
        // A job is born either `queued` (immediately claimable) or, for an API-side async
        // pre-step, `pending_caption` (sc-9120) — status and stage move in lockstep, and the
        // waiting message reflects which gate the job is behind so the queue view reads
        // correctly before a worker (or the background rewrite) ever touches it.
        let initial_status = request.initial_status_str()?;
        let initial_message = match initial_status {
            "pending_caption" => "Preparing the prompt before dispatch.",
            _ => "Waiting for an available worker.",
        };
        connection.execute(
            "
            insert into jobs (
              id, type, status, project_id, project_name, payload_json, result_json,
              requested_gpu, progress, stage, message, attempts, source_job_id,
              duplicate_of_job_id, created_at, updated_at
            ) values (?1, ?2, ?12, ?3, ?4, ?5, '{}', ?6, 0, ?12, ?7, ?8, ?9, ?10, ?11, ?11)
            ",
            params![
                job_id,
                request.job_type.as_str(),
                request.project_id,
                request.project_name,
                dumps(&request.payload)?,
                requested_gpu,
                initial_message,
                request.attempts,
                request.source_job_id,
                request.duplicate_of_job_id,
                now,
                initial_status,
            ],
        )?;
        self.get_job_on_connection(connection, &job_id)
    }

    fn list_jobs_by_status_on_connection(
        &self,
        connection: &Connection,
        statuses: &[&str],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        // One prepared statement + one table scan instead of preparing and
        // executing `where status = ?` once per status (sc-8896 / F-094). The
        // status list is quoted from the caller-provided `&[&str]` — always
        // crate constants (e.g. ACTIVE_STATUSES), never user input — so direct
        // interpolation is safe, matching active_statuses_sql()'s rationale.
        // The old per-status loop returned rows grouped by status in the input
        // order with no intra-group ordering; the single caller
        // (mark_interrupted_on_startup) uses only the ids, so ordering is not
        // load-bearing. We add an explicit `order by created_at desc` anyway to
        // make the result deterministic and consistent with list_jobs/queue
        // reads rather than leaving it to SQLite's unspecified row order.
        if statuses.is_empty() {
            return Ok(Vec::new());
        }
        let status_list = statuses
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut statement = connection.prepare(&format!(
            "select * from jobs where status in ({status_list}) order by created_at desc"
        ))?;
        let jobs = collect_jobs(statement.query_map([], row_to_job)?)?;
        Ok(jobs)
    }

    fn active_jobs_for_workers(
        &self,
        connection: &Connection,
        worker_ids: &[String],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if worker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = placeholders_from(1, worker_ids.len());
        let mut statement = connection.prepare(&format!(
            "
            select * from jobs
             where worker_id in ({placeholders})
               and status in ({active})
            ",
            active = active_statuses_sql()
        ))?;
        let jobs = collect_jobs(statement.query_map(
            params_from_iter(worker_ids.iter().map(String::as_str)),
            row_to_job,
        )?)?;
        Ok(jobs)
    }

    fn workers_by_ids(
        &self,
        connection: &Connection,
        worker_ids: &[String],
    ) -> JobsStoreResult<Vec<WorkerSnapshot>> {
        if worker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = placeholders_from(1, worker_ids.len());
        let mut statement = connection.prepare(&format!(
            "select * from workers where id in ({placeholders}) order by gpu_id, id"
        ))?;
        let workers = collect_workers(statement.query_map(
            params_from_iter(worker_ids.iter().map(String::as_str)),
            row_to_worker,
        )?)?;
        Ok(workers)
    }

    fn get_job_on_connection(
        &self,
        connection: &Connection,
        job_id: &str,
    ) -> JobsStoreResult<JobSnapshot> {
        connection
            .query_row(
                "select * from jobs where id = ?1",
                params![job_id],
                row_to_job,
            )
            .optional()?
            .ok_or_else(|| JobsStoreError::NotFound(job_id.to_owned()))
    }

    fn get_worker_on_connection(
        &self,
        connection: &Connection,
        worker_id: &str,
    ) -> JobsStoreResult<WorkerSnapshot> {
        connection
            .query_row(
                "select * from workers where id = ?1",
                params![worker_id],
                row_to_worker,
            )
            .optional()?
            .ok_or_else(|| JobsStoreError::NotFound(worker_id.to_owned()))
    }
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<JobSnapshot> {
    let progress: f64 = row.get("progress")?;
    let eta_seconds: Option<f64> = row.get("eta_seconds")?;
    let peak_memory: Option<f64> = row.get("peak_gpu_memory_pct").ok().flatten();
    let peak_load: Option<f64> = row.get("peak_gpu_load_pct").ok().flatten();
    let backend: Option<String> = row.get("backend").ok().flatten();
    let created_at: String = row.get("created_at")?;
    let started_at: Option<String> = row.get("started_at")?;
    let completed_at: Option<String> = row.get("completed_at")?;
    let elapsed_seconds = started_at
        .as_deref()
        .and_then(|started| elapsed_seconds(started, completed_at.as_deref()));
    let job_type: JobType = parse_string_enum(&row.get::<_, String>("type")?);
    let payload = loads_object(row.get::<_, Option<String>>("payload_json")?.as_deref());
    let title = derive_job_title(&job_type, &payload);
    Ok(JobSnapshot {
        id: row.get("id")?,
        job_type,
        status: parse_string_enum(&row.get::<_, String>("status")?),
        project_id: row.get("project_id")?,
        project_name: row.get("project_name")?,
        payload,
        result: loads_object(row.get::<_, Option<String>>("result_json")?.as_deref()),
        requested_gpu: row.get("requested_gpu")?,
        assigned_gpu: row.get("assigned_gpu")?,
        worker_id: row.get("worker_id")?,
        progress: number_from_f64(progress),
        stage: parse_string_enum(&row.get::<_, String>("stage")?),
        message: row.get("message")?,
        error: row.get("error")?,
        eta_seconds: eta_seconds.map(number_from_f64),
        elapsed_seconds,
        attempts: row.get::<_, u32>("attempts")?,
        source_job_id: row.get("source_job_id")?,
        duplicate_of_job_id: row.get("duplicate_of_job_id")?,
        cancel_requested: row.get::<_, i64>("cancel_requested")? != 0,
        created_at,
        updated_at: row.get("updated_at")?,
        started_at,
        completed_at,
        canceled_at: row.get("canceled_at")?,
        last_heartbeat_at: row.get("last_heartbeat_at")?,
        peak_gpu_memory_pct: peak_memory.map(number_from_f64),
        peak_gpu_load_pct: peak_load.map(number_from_f64),
        backend,
        title,
        extra: Default::default(),
    })
}

/// Map a `generation_metrics` row to the contract struct (epic 10402). Reads
/// every metrics column by name, so it works both for a bare `select *` and for
/// the joined aggregate query (whose extra job-identity columns are aliased
/// `j_*` and ignored here).
fn row_to_generation_metrics(row: &Row<'_>) -> rusqlite::Result<GenerationMetrics> {
    let scheduler_shift: Option<f64> = row.get("scheduler_shift")?;
    let guidance_scale: Option<f64> = row.get("guidance_scale")?;
    let true_cfg_scale: Option<f64> = row.get("true_cfg_scale")?;
    let peak_memory_pct: Option<f64> = row.get("peak_memory_pct")?;
    let peak_gpu_load_pct: Option<f64> = row.get("peak_gpu_load_pct")?;
    let loras: Option<String> = row.get("loras_json")?;
    Ok(GenerationMetrics {
        model: row.get("model")?,
        quant_label: row.get("quant_label")?,
        quant_bits: row.get("quant_bits")?,
        sampler: row.get("sampler")?,
        scheduler: row.get("scheduler")?,
        scheduler_shift: scheduler_shift.map(number_from_f64),
        steps: row.get("steps")?,
        image_count: row.get("image_count")?,
        guidance_scale: guidance_scale.map(number_from_f64),
        true_cfg_scale: true_cfg_scale.map(number_from_f64),
        guidance_method: row.get("guidance_method")?,
        use_pid: row.get("use_pid")?,
        pid_target: row.get("pid_target")?,
        width: row.get("width")?,
        height: row.get("height")?,
        seed: row.get("seed")?,
        loras: loras.and_then(|value| serde_json::from_str(&value).ok()),
        load_ms: row.get("load_ms")?,
        sample_ms: row.get("sample_ms")?,
        decode_ms: row.get("decode_ms")?,
        total_ms: row.get("total_ms")?,
        peak_memory_bytes: row.get("peak_memory_bytes")?,
        peak_memory_pct: peak_memory_pct.map(number_from_f64),
        peak_gpu_load_pct: peak_gpu_load_pct.map(number_from_f64),
        backend: row.get("backend")?,
        extra: Default::default(),
    })
}

/// Map a joined aggregate row (metrics + `j_*`-aliased job identity) to a
/// `GenerationMetricsRow` for the `GET /api/v1/metrics` feed (epic 10402).
fn row_to_generation_metrics_row(row: &Row<'_>) -> rusqlite::Result<GenerationMetricsRow> {
    Ok(GenerationMetricsRow {
        job_id: row.get("job_id")?,
        job_type: parse_string_enum(&row.get::<_, String>("j_type")?),
        status: parse_string_enum(&row.get::<_, String>("j_status")?),
        project_id: row.get("j_project_id")?,
        created_at: row.get("j_created_at")?,
        metrics: row_to_generation_metrics(row)?,
    })
}

/// Server-side derivation of the human-readable job title surfaced in the
/// queue and WorkerProgressCard (sc-2087). Mirrors the Job Title table in
/// docs/design/worker-progress-card.md. Returns None for types where the
/// payload doesn't carry a meaningful subject — the frontend then falls back
/// to its own derivation, keeping the queue from ever showing only a raw job
/// id as the row identifier.
fn derive_job_title(job_type: &JobType, payload: &Map<String, Value>) -> Option<String> {
    /// Find the first string value at any of the candidate keys.
    fn first_str<'a>(payload: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
        keys.iter()
            .find_map(|key| payload.get(*key).and_then(Value::as_str))
            .filter(|value| !value.trim().is_empty())
    }
    /// Truncate a prompt to ~max chars on a word boundary, append an ellipsis
    /// when truncated. Mirrors the JS helper in WorkerProgressCard.jsx.
    fn truncate_prompt(prompt: &str, max: usize) -> String {
        if prompt.len() <= max {
            return prompt.to_owned();
        }
        let mut cut = prompt[..max].to_owned();
        if let Some(space) = cut.rfind(' ') {
            if space > (max * 6) / 10 {
                cut.truncate(space);
            }
        }
        format!("{}…", cut.trim_end())
    }

    match job_type {
        JobType::LoraTrain => {
            let subject = first_str(payload, &["loraName", "outputName", "targetName", "loraId"])
                .map(str::to_owned)
                .or_else(|| {
                    payload
                        .get("plan")
                        .and_then(|plan| plan.get("output"))
                        .and_then(|output| output.get("loraId"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "(unnamed LoRA)".to_owned());
            Some(format!("Training Run — {subject}"))
        }
        JobType::TrainingCaption => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Captioning — {subject}"))
        }
        JobType::DatasetAnalysis => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Analysis — {subject}"))
        }
        JobType::DatasetUpscale => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Upscaling Dataset Images — {subject}"))
        }
        JobType::DatasetFaceAnalysis => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Face Analysis — {subject}"))
        }
        JobType::FaceLikenessCompare => {
            // sc-4415: compare a candidate asset to a source identity reference. The candidate is the
            // user-facing subject of the row; fall back to a plain label when the payload omits it.
            let subject =
                first_str(payload, &["candidateName", "candidateAssetId"]).unwrap_or("(image)");
            Some(format!("Compare Likeness — {subject}"))
        }
        JobType::ImageGenerate
        | JobType::ImageEdit
        | JobType::ImageVqa
        | JobType::ImageInterleave => {
            // Character Turnaround override: a character generation has
            // characterId + characterName on the payload.
            if payload.get("characterId").and_then(Value::as_str).is_some() {
                if let Some(name) = first_str(payload, &["characterName"]) {
                    return Some(format!("Character Turnaround — {name}"));
                }
            }
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Generate Image — {}", truncate_prompt(prompt, 80)))
        }
        JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Generate Video — {}", truncate_prompt(prompt, 80)))
        }
        JobType::PersonReplace => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(no prompt)");
            Some(format!("Person Replace — {}", truncate_prompt(prompt, 80)))
        }
        JobType::ModelDownload | JobType::ModelImport | JobType::ModelConvert => {
            let subject =
                first_str(payload, &["modelName", "filename", "modelId", "repo"]).unwrap_or("");
            if subject.is_empty() {
                Some("Model Import".to_owned())
            } else {
                Some(format!("Model Import — {subject}"))
            }
        }
        JobType::LoraImport => {
            let subject = first_str(payload, &["loraName", "filename", "loraId"]).unwrap_or("");
            if subject.is_empty() {
                Some("LoRA Import".to_owned())
            } else {
                Some(format!("LoRA Import — {subject}"))
            }
        }
        JobType::LoraDownload => {
            let subject = first_str(payload, &["loraName", "loraId", "repo"]).unwrap_or("");
            if subject.is_empty() {
                Some("LoRA Download".to_owned())
            } else {
                Some(format!("LoRA Download — {subject}"))
            }
        }
        JobType::PromptRefine => {
            let prompt = first_str(payload, &["prompt"]).unwrap_or("(empty prompt)");
            Some(format!("Prompt Refine — {}", truncate_prompt(prompt, 60)))
        }
        // Person detect/track/segment + anything else — let the frontend
        // fall back to its own derivation.
        _ => None,
    }
}

fn row_to_worker(row: &Row<'_>) -> rusqlite::Result<WorkerSnapshot> {
    Ok(WorkerSnapshot {
        id: row.get("id")?,
        gpu_id: row.get("gpu_id")?,
        gpu_name: row.get("gpu_name")?,
        status: parse_string_enum(&row.get::<_, String>("status")?),
        current_job_id: row.get("current_job_id")?,
        capabilities: loads_vec(
            row.get::<_, Option<String>>("capabilities_json")?
                .as_deref(),
        ),
        loaded_models: loads_vec(
            row.get::<_, Option<String>>("loaded_models_json")?
                .as_deref(),
        ),
        utilization: loads_optional(row.get::<_, Option<String>>("utilization_json")?.as_deref()),
        registered_at: row.get("registered_at")?,
        last_seen_at: row.get("last_seen_at")?,
        extra: Default::default(),
    })
}

fn collect_jobs<F>(rows: rusqlite::MappedRows<'_, F>) -> JobsStoreResult<Vec<JobSnapshot>>
where
    F: FnMut(&Row<'_>) -> rusqlite::Result<JobSnapshot>,
{
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn collect_workers<F>(rows: rusqlite::MappedRows<'_, F>) -> JobsStoreResult<Vec<WorkerSnapshot>>
where
    F: FnMut(&Row<'_>) -> rusqlite::Result<WorkerSnapshot>,
{
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn dumps<T: serde::Serialize>(value: &T) -> JobsStoreResult<String> {
    let mut value = serde_json::to_value(value)?;
    sort_json_value(&mut value);
    serde_json::to_string(&value).map_err(Into::into)
}

fn optional_dumps<T: serde::Serialize>(value: Option<&T>) -> JobsStoreResult<Option<String>> {
    value.map(dumps).transpose()
}

fn loads_object(value: Option<&str>) -> Map<String, Value> {
    value
        .and_then(|text| serde_json::from_str::<Map<String, Value>>(text).ok())
        .unwrap_or_default()
}

/// Merge accumulated `trainingSamples` history into an incoming progress
/// result. `existing_result` is the job's current result, which
/// `update_job_progress` has already read in the same transaction — so this no
/// longer re-`select`s `result_json` per update (sc-4274 / F-CORE-14).
fn merge_training_sample_history(
    existing_result: Option<&Map<String, Value>>,
    incoming: &mut Map<String, Value>,
) {
    let has_training_samples = incoming
        .get("trainingSamples")
        .and_then(Value::as_array)
        .is_some();
    let has_latest_training_samples = incoming
        .get("latestTrainingSamples")
        .and_then(Value::as_array)
        .is_some();
    if !has_training_samples && !has_latest_training_samples {
        return;
    }

    let mut samples = Vec::new();
    let mut seen = std::collections::HashSet::new();
    append_training_samples(
        &mut samples,
        &mut seen,
        existing_result.and_then(|result| result.get("trainingSamples")),
    );
    append_training_samples(&mut samples, &mut seen, incoming.get("trainingSamples"));
    append_training_samples(
        &mut samples,
        &mut seen,
        incoming.get("latestTrainingSamples"),
    );

    if !samples.is_empty() {
        incoming.insert("trainingSamples".to_owned(), Value::Array(samples));
    }
}

fn append_training_samples(
    samples: &mut Vec<Value>,
    seen: &mut std::collections::HashSet<String>,
    value: Option<&Value>,
) {
    let Some(array) = value.and_then(Value::as_array) else {
        return;
    };
    for sample in array {
        let key = training_sample_key(sample, samples.len());
        if seen.insert(key) {
            samples.push(sample.clone());
        }
    }
}

fn training_sample_key(sample: &Value, fallback_index: usize) -> String {
    let Some(object) = sample.as_object() else {
        return format!("sample:{fallback_index}");
    };
    for key in ["relativePath", "path", "url"] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            if !value.is_empty() {
                return format!("{key}:{value}");
            }
        }
    }
    let step = object
        .get("step")
        .map(Value::to_string)
        .unwrap_or_else(|| "unknown".to_owned());
    let prompt = object
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!("step:{step}:prompt:{prompt}:index:{fallback_index}")
}

fn loads_vec<T>(value: Option<&str>) -> Vec<T>
where
    T: DeserializeOwned,
{
    value
        .and_then(|text| serde_json::from_str::<Vec<T>>(text).ok())
        .unwrap_or_default()
}

fn loads_optional<T>(value: Option<&str>) -> Option<T>
where
    T: DeserializeOwned,
{
    // Best-effort worker telemetry should disappear rather than poison the queue.
    value.and_then(|text| serde_json::from_str::<T>(text).ok())
}

fn number_from_f64(value: f64) -> ContractNumber {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}

fn elapsed_seconds(started_at: &str, completed_at: Option<&str>) -> Option<ContractNumber> {
    let started = parse_utc_seconds(started_at)?;
    let ended = completed_at.map_or_else(|| Some(now_unix_seconds()), parse_utc_seconds)?;
    let seconds = ended.saturating_sub(started).max(0);
    Some(Number::from(seconds))
}

fn is_active_status(status: &str) -> bool {
    ACTIVE_STATUSES.contains(&status)
}

fn is_terminal_status(status: &str) -> bool {
    TERMINAL_STATUSES.contains(&status)
}

fn is_non_gpu_job_type(job_type: &str) -> bool {
    NON_GPU_JOB_TYPES.contains(&job_type)
}

/// The GPU routing decision for a single claim, emitted as a structured log event
/// (`gpu_route_decision`) by the API so operators can see *which backend ran a job, and
/// why* (sc-3449). Every label is named after the backend that actually claimed the job,
/// never as a deficiency: on Windows/Linux a candle (CUDA) claim is the normal happy path,
/// so the line must never read like an MLX worker is missing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteDecision {
    pub job_id: String,
    pub job_type: String,
    pub model: Option<String>,
    pub requested_gpu: String,
    pub worker_id: String,
    pub gpu_id: String,
    /// `deferred_to_mlx` | `claimed_by_mlx` | `claimed_by_candle` | `claimed_by_gpu` |
    /// `explicit_gpu`.
    pub decision: &'static str,
    /// Machine-readable cause: `idle_mlx_available`, `mlx_worker`, `candle_worker`,
    /// `gpu_worker`, or `explicit_gpu`.
    pub reason: &'static str,
}

impl RouteDecision {
    fn new(
        job: &JobSnapshot,
        gpu_id: &str,
        worker_id: &str,
        decision: &'static str,
        reason: &'static str,
    ) -> Self {
        Self {
            job_id: job.id.clone(),
            job_type: job.job_type.as_str().to_owned(),
            model: job
                .payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            requested_gpu: job.requested_gpu.clone(),
            worker_id: worker_id.to_owned(),
            gpu_id: gpu_id.to_owned(),
            decision,
            reason,
        }
    }
}

/// Classify a *successful* claim for routing observability, named after the backend that
/// actually took the job. `None` means the claim was routing-neutral (nothing an `mlx`
/// worker would ever want, so there is nothing to explain). Every label describes what
/// happened, never a deficiency: an `mlx` worker claim is `claimed_by_mlx`, a candle
/// (Windows/Linux CUDA) claim is `claimed_by_candle`, and a user-pinned GPU is
/// `explicit_gpu`. Candle is identified by the `candle` capability marker
/// (`worker_is_candle`) — it runs on a real GPU index, so `gpu_id` alone can't distinguish
/// it. Any other GPU worker falls to the generic `claimed_by_gpu` catch-all: with the
/// Python torch worker retired from every surface, nothing else should claim these jobs, so
/// the label names no specific backend. The deferral path (a non-mlx worker yielding to an
/// idle mlx worker on Mac) is reported separately inside `claim_next_job_routed` as
/// `deferred_to_mlx`.
fn route_decision_for_claim(job: &JobSnapshot, worker: &WorkerSnapshot) -> Option<RouteDecision> {
    if !job_is_any_mlx_eligible(job) {
        return None;
    }
    let gpu_id = worker.gpu_id.as_str();
    let worker_id = worker.id.as_str();
    if gpu_id.eq_ignore_ascii_case("mlx") {
        return Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "claimed_by_mlx",
            "mlx_worker",
        ));
    }
    // An explicit (non-`auto`) GPU pin is always honoured as the user asked.
    if job.requested_gpu != "auto" {
        return Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "explicit_gpu",
            "explicit_gpu",
        ));
    }
    // An `auto` claim by a non-mlx GPU worker. On Windows/Linux the candle (CUDA) lane is
    // the expected home, not a fallback. The `else` is a defensive catch-all for any other
    // GPU worker — with the Python torch worker retired from every surface it should not
    // fire in practice, so it is named generically rather than after a backend that no
    // longer exists.
    if worker_is_candle(worker) {
        Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "claimed_by_candle",
            "candle_worker",
        ))
    } else {
        Some(RouteDecision::new(
            job,
            gpu_id,
            worker_id,
            "claimed_by_gpu",
            "gpu_worker",
        ))
    }
}

fn should_defer_auto_gpu_claim(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
) -> JobsStoreResult<bool> {
    if job.requested_gpu != "auto"
        || is_non_gpu_job_type(job.job_type.as_str())
        || worker.gpu_id == "cpu"
    {
        return Ok(false);
    }
    // The in-process `mlx` worker is the designated home for the jobs it claims
    // (a non-mlx worker defers MLX-eligible jobs to it via
    // `should_defer_image_to_mlx_worker` & siblings). It must never hand one of
    // those jobs to a "healthier" non-mlx GPU through this health-based dispatch:
    // on Apple Silicon the `mlx` and `mps` workers share the same physical GPU,
    // and that worker would only defer the job straight back, deadlocking it in
    // the queue. Keeping the mlx worker out of the auto-GPU health comparison
    // breaks that cycle regardless of whether it reports utilization.
    if worker.gpu_id.eq_ignore_ascii_case("mlx") {
        return Ok(false);
    }
    let current_score = dispatch_score(job, worker);
    if !current_score.has_utilization {
        return Ok(false);
    }

    let mut statement = connection.prepare(
        "
        select * from workers
         where id != ?1
           and gpu_id != 'cpu'
           and status = 'idle'
         order by gpu_id, id
        ",
    )?;
    let candidates = collect_workers(statement.query_map(params![worker.id], row_to_worker)?)?;
    // Cache the active-GPU-job fact per gpu_id so two idle workers sharing a GPU
    // don't each re-run the same `active_gpu_job_exists` query (sc-4273).
    let mut active_by_gpu: std::collections::HashMap<String, bool> =
        std::collections::HashMap::new();
    for candidate in candidates {
        if !worker_supports_job(&candidate, job) {
            continue;
        }
        let gpu_busy = match active_by_gpu.get(&candidate.gpu_id) {
            Some(&busy) => busy,
            None => {
                let busy = active_gpu_job_exists(connection, &candidate.gpu_id)?;
                active_by_gpu.insert(candidate.gpu_id.clone(), busy);
                busy
            }
        };
        if gpu_busy {
            continue;
        }
        let candidate_score = dispatch_score(job, &candidate);
        if dispatch_score_is_better(candidate_score, current_score) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Epic 3018 routing — prefer the in-process MLX worker for MLX-eligible image
/// jobs. A non-mlx GPU worker defers an `auto` `image_generate` job the mlx
/// worker can run when an idle `mlx` worker exists, so the fast NAX path claims
/// it. When no mlx worker is registered (Windows/Linux, or the mlx worker is
/// down), nothing defers and the torch worker is the fallback — a job is never
/// stuck. An explicit (non-`auto`) GPU choice is always honoured, never deferred.
fn should_defer_image_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS "MLX-required" (epic 3482 / sc-3483): the non-mlx (MPS) worker NEVER claims
    // an MLX-eligible job — it yields unconditionally, even when no idle `mlx` worker is
    // ready *right now*. The job waits for the `mlx` worker and, if none takes it within
    // the grace window, `fail_stranded_mlx_jobs` fails it terminal with `mlx_unavailable`
    // rather than letting MPS silently run it. This covers explicit-GPU pins too: "never
    // MPS" is absolute on Mac.
    if mlx_required {
        return Ok(true);
    }
    // Off (Windows/Linux/Docker, and Mac pre-cutover): unchanged — defer only an `auto`
    // job to an actually-idle `mlx` worker; otherwise the torch worker is the fallback and
    // an explicit (non-`auto`) GPU choice is always honoured.
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Video sibling of [`should_defer_image_to_mlx_worker`] (sc-3036): a non-mlx GPU
/// worker defers an `auto` MLX-eligible `video_generate` job when an idle `mlx`
/// worker can run it. Same fallback guarantees — no mlx worker / explicit GPU →
/// never deferred.
fn should_defer_video_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !video_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Training sibling of [`should_defer_image_to_mlx_worker`] (epic 3039): a non-mlx
/// GPU worker defers an `auto` MLX-eligible `lora_train` job when an idle `mlx`
/// worker can run it, so the native Rust trainer (`mlx_gen::load_trainer`) claims
/// it. Same fallback guarantees — no mlx worker registered (Windows/Linux, or the
/// mlx worker is down) → nothing defers and the Python torch trainer runs it; an
/// explicit (non-`auto`) GPU choice is always honoured. The torch trainers stay
/// the cross-platform path + the Mac fallback (sc-3049), so a job is never stuck.
fn should_defer_training_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !training_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Captioning sibling of [`should_defer_image_to_mlx_worker`] (sc-3556): a non-mlx
/// GPU worker defers JoyCaption dataset-caption jobs to an idle mlx worker, so the
/// native Rust captioner (`mlx_gen::load_captioner`) can run them. Windows/Linux and
/// explicit non-auto GPU requests keep the existing Python torch captioner fallback.
fn should_defer_caption_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !caption_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Understanding sibling of [`should_defer_image_to_mlx_worker`] (sc-3905): a non-mlx GPU worker
/// defers an `auto` MLX-eligible SenseNova-U1 `image_vqa` / `image_interleave` job to an idle mlx
/// worker, so the in-process `T2iModel` (`vqa` / `interleave_gen`) claims it. Windows/Linux and
/// explicit non-auto GPU requests keep the Python torch SenseNova path.
fn should_defer_understanding_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !understanding_job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS MLX-required (sc-3483): yield unconditionally, same as the image sibling.
    if mlx_required {
        return Ok(true);
    }
    if job.requested_gpu != "auto" {
        return Ok(false);
    }
    idle_mlx_worker_can_claim(connection, job, worker)
}

/// Whether an idle `mlx` worker (other than `worker`) exists that supports `job`
/// and has no active GPU job — the shared tail of the image/video MLX deferral.
fn idle_mlx_worker_can_claim(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
) -> JobsStoreResult<bool> {
    let mut statement = connection.prepare(
        "
        select * from workers
         where id != ?1
           and gpu_id = 'mlx'
           and status = 'idle'
         order by id
        ",
    )?;
    let candidates = collect_workers(statement.query_map(params![worker.id], row_to_worker)?)?;
    // Every candidate here has `gpu_id = 'mlx'`, so the active-GPU-job fact is
    // identical for all of them — resolve a supporting candidate first, then run
    // `active_gpu_job_exists` once instead of once per candidate (sc-4273).
    let Some(candidate) = candidates.iter().find(|c| worker_supports_job(c, job)) else {
        return Ok(false);
    };
    Ok(!active_gpu_job_exists(connection, &candidate.gpu_id)?)
}

fn active_gpu_job_exists(connection: &Connection, gpu_id: &str) -> JobsStoreResult<bool> {
    if is_apple_unified_gpu_id(gpu_id) {
        return Ok(connection
            .query_row(
                &format!(
                    "
            select id from jobs
             where lower(assigned_gpu) in ('mlx', 'mps')
               and status in ({active})
               and type not in ({})
             limit 1
            ",
                    non_gpu_job_types_sql(),
                    active = active_statuses_sql()
                ),
                [],
                |_row| Ok(()),
            )
            .optional()?
            .is_some());
    }
    Ok(connection
        .query_row(
            &format!(
                "
            select id from jobs
             where assigned_gpu = ?1
               and status in ({active})
               and type not in ({})
             limit 1
            ",
                non_gpu_job_types_sql(),
                active = active_statuses_sql()
            ),
            params![gpu_id],
            |_row| Ok(()),
        )
        .optional()?
        .is_some())
}

fn is_apple_unified_gpu_id(gpu_id: &str) -> bool {
    gpu_id.eq_ignore_ascii_case("mlx") || gpu_id.eq_ignore_ascii_case("mps")
}

fn worker_supports_job(worker: &WorkerSnapshot, job: &JobSnapshot) -> bool {
    if job_requires_gpu(&job.job_type) && worker.gpu_id.eq_ignore_ascii_case("cpu") {
        return false;
    }
    // Epic 3039 (sc-3049): a training kernel with no torch fallback (the retired Python
    // MLX LTX trainer) runs only on a Rust worker — a non-mlx worker must refuse it
    // (leaving it queued for the mlx worker) instead of claiming it and failing. The
    // exception (sc-8614): `krea_lora` is no-torch-fallback AND has a candle trainer, so a
    // candle worker it is candle-eligible for must NOT be refused here (the candle training
    // gate below admits it); only torch (and any non-candle non-mlx worker) still defers.
    if !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && training_kernel_is_mlx_only(job)
        && !(worker_is_candle(worker) && training_job_is_candle_eligible(job))
    {
        return false;
    }
    // Epic 3018/3041 + sc-3036: the in-process MLX worker (gpu_id "mlx") serves a fixed
    // set of model families. It must not claim a job that needs the torch path — a family
    // not yet ported, an unsupported shape, or a third-party LyCORIS LoRA — those stay on
    // the Python worker. Non-mlx workers are unaffected here; the *preference* to route
    // eligible jobs to an idle mlx worker is a soft deferral in the claim path.
    if worker.gpu_id.eq_ignore_ascii_case("mlx") {
        // Image: sc-3026 txt2img/LoRA + sc-3060 reference/edit/inpaint/outpaint +
        // image_detail + sc-3513 the `image_edit` job type (plain Image Edit). A
        // torch-only edit model (kolors/lens/pulid) is not MLX-eligible, so the mlx
        // worker refuses it and it stays on torch. (z_image_edit was ported to MLX,
        // epic 3529 / sc-3923; instantid + sensenova are MLX-routed too.)
        if matches!(
            job.job_type,
            JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
        ) && !job_is_mlx_eligible(job)
        {
            return false;
        }
        // Video (sc-3036 + the epic-3040 cutover): the mlx worker claims MLX-eligible
        // `video_generate` jobs (Wan/LTX text_to_video / image_to_video + SVD
        // image_to_video) plus the advanced job types now ported to the Rust engine —
        // `first_last_frame` (LTX + Wan TI2V-5B, sc-3520), `extend_clip` / `video_bridge`
        // (LTX IC-LoRA, sc-3522), and `person_replace` → native Wan-VACE (sc-3521). The
        // per-(model, mode) gate in `video_job_is_mlx_eligible` keeps each mode to its
        // capable engines; everything it rejects — a non-MLX model, Wan extend/bridge
        // (no IC-LoRA keyframe-append path), LoKr-on-Wan — stays on the Python worker.
        if matches!(
            job.job_type,
            JobType::VideoGenerate
                | JobType::VideoExtend
                | JobType::VideoBridge
                | JobType::PersonReplace
        ) && !video_job_is_mlx_eligible(job)
        {
            return false;
        }
        // Training (epic 3039): the mlx worker trains only the MLX-native families
        // (z_image / sdxl / kolors / wan / ltx) via `mlx_gen::load_trainer`. `lens_lora`
        // (sidecar, no mlx-gen crate) and LoKr-on-Wan stay on the Python torch worker.
        // Applies to both dry-run and real runs.
        if matches!(job.job_type, JobType::LoraTrain) && !training_job_is_mlx_eligible(job) {
            return false;
        }
        // Dataset captioning (sc-3556): the mlx worker claims only JoyCaption jobs
        // backed by the mlx-gen provider. Any future non-JoyCaption captioner stays
        // on the worker that advertises that capability.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
            return false;
        }
        // Image upscale (sc-3489): the mlx worker runs Real-ESRGAN (the default engine) via
        // `ort`/CoreML and SeedVR2 via in-process `mlx-gen-seedvr2` (sc-4815). `aura-sr` has no
        // Rust path, so the mlx worker refuses it and it stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageUpscale) && !upscale_job_is_mlx_eligible(job) {
            return false;
        }
        // Video upscale (epic 4811 / sc-4816): the mlx worker runs the native SeedVR2 engine
        // (`mlx-gen-seedvr2`). Any non-SeedVR2 engine is refused; since there is no torch
        // video-upscale backend, this is mac-only by construction.
        if matches!(job.job_type, JobType::VideoUpscale) && !video_upscale_job_is_mlx_eligible(job)
        {
            return false;
        }
        // SenseNova-U1 understanding (sc-3905): the mlx worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (the sole in-process understanding
        // path). A non-SenseNova understanding job is not MLX-eligible, so the mlx worker
        // refuses it and it stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
    }
    // No-silent-T2I / no-torch-fallback (sc-5968, epic 5483): the co-resident Python torch worker (a
    // non-candle, non-mlx GPU worker) must DECLINE the unsupported-pose shapes the candle worker
    // owns-to-reject (a `advanced.poses` job on a candle model with no pose lane, e.g. sdxl) — so torch
    // can't claim + silently render an unconditioned T2I image, and the candle worker reliably wins
    // them (then rejects with a typed error). Mac is unaffected: those shapes are MLX-served there
    // (model_mac_support pose), so the `mlx` worker still claims them and only torch/`mps` declines.
    if !worker_is_candle(worker)
        && !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && image_job_candle_pose_reject(job)
    {
        return false;
    }
    // Candle (Windows/CUDA) lane (epic 3672 image sc-3678; epic 5095 image families sc-5096 + video
    // sc-5097): the candle worker advertises `image_generate` (+ `video_generate` once video engines
    // are wired) and serves gated, narrow **txt2img / txt2video-only** lanes. It must refuse every
    // other shape — a non-candle family, or a conditioned (img2img/edit/reference/inpaint/pose/
    // i2v/extend/bridge/replace) / LoRA request — so those transparently fall back to the Python torch
    // worker that co-resides on the box. Identified by the `candle` marker capability (not `gpu_id`,
    // which is a real CUDA index here). When candle is disabled the marker is absent and this is inert,
    // so production routing is unchanged until the lane is turned on.
    if worker_is_candle(worker) {
        // ImageGenerate + ImageEdit: claim the candle-served shapes (incl. the sc-5487
        // SdxlEdit/Flux2Edit/QwenEdit `image_edit` lanes) AND the unsupported-pose shapes the candle
        // worker must OWN to reject (a `advanced.poses` job on a candle model with no pose lane, e.g.
        // sdxl) — so those fail loudly on candle instead of falling back to torch + silently rendering
        // an unconditioned T2I image (sc-5968, the no-torch-fallback / no-silent-T2I directive). Every
        // other shape candle declines, staying on the co-resident torch worker. `image_edit` is gated
        // here too (mirroring the mlx `JobType::ImageGenerate | JobType::ImageEdit` claim arm): without
        // it a torch-only edit model would be claimed by candle and fail instead of falling back.
        if matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit)
            && !(image_job_is_candle_eligible(job) || image_job_candle_pose_reject(job))
        {
            return false;
        }
        // The candle worker advertises only the base `video_generate` (txt2video); refuse the
        // advanced video job types and every non-eligible `video_generate` shape.
        if matches!(
            job.job_type,
            JobType::VideoGenerate
                | JobType::VideoExtend
                | JobType::VideoBridge
                | JobType::PersonReplace
        ) && !video_job_is_candle_eligible(job)
        {
            return false;
        }
        // Training (sc-7817, epic 5164): the candle worker trains only the candle-native families
        // (sdxl / z_image / lens / the Wan A14B T2V MoE) via `gen_core::load_trainer`. Everything
        // else — Kolors, LTX, the dense Wan 5B, the Wan I2V A14B — has no candle trainer and stays on
        // the co-resident Python torch worker. WITHOUT this gate the candle worker would claim a real
        // training job it can't execute (the `lora_train_execute` advertisement is coarse — it lights
        // up whenever ANY candle trainer is registered) and fail it terminally instead of leaving it
        // for torch. Applies to both dry-run and real runs; mirrors the mlx training gate above.
        if matches!(job.job_type, JobType::LoraTrain) && !training_job_is_candle_eligible(job) {
            return false;
        }
        // Dataset captioning (sc-5098): the candle worker serves only JoyCaption (the candle
        // captioner provider). A non-`joy_caption` caption job stays on the Python torch worker.
        // Eligibility is backend-neutral (captioner == joy_caption), so reuse the mlx gate.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
            return false;
        }
        // SenseNova-U1 understanding (sc-5501): the candle worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (via the concrete candle `T2iModel::{vqa,
        // interleave_gen}` — the off-Mac sibling of the MLX understanding path). Eligibility is
        // backend-neutral (the model is SenseNova-U1), so reuse the understanding gate; a
        // non-SenseNova understanding job stays on the Python torch worker.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
        // Image upscale (sc-5928 SeedVR2 + sc-5499 Real-ESRGAN, epic 4811 / epic 5482): the candle
        // worker serves Real-ESRGAN (`ort`/CUDA, sc-5499) AND SeedVR2 (`candle-gen-seedvr2`, the
        // Windows/CUDA sibling of mlx-gen-seedvr2). Only `aura-sr` has no candle path — it is refused
        // and runs on the Python torch worker until Phase 7.
        if matches!(job.job_type, JobType::ImageUpscale) && !upscale_job_is_candle_eligible(job) {
            return false;
        }
        // Video upscale (sc-5928): the candle worker serves the net-new SeedVR2 video upscaler. A
        // non-SeedVR2 engine is refused (no other video-upscale backend exists off-Mac).
        if matches!(job.job_type, JobType::VideoUpscale)
            && !video_upscale_job_is_candle_eligible(job)
        {
            return false;
        }
    }
    // SeedVR2 upscaling has NO torch backend — it runs on the native MLX worker (Mac) or the candle
    // worker (Windows/Linux). A plain torch GPU/CPU worker (neither `mlx` nor candle) must refuse a
    // SeedVR2 `image_upscale` job so it stays queued for the mlx/candle worker instead of being
    // claimed and failing with "no generator registered". This is the inverse of the AuraSR gate
    // (torch-only → mlx/candle refuse it). `video_upscale` is candle/mlx-only by capability (a torch
    // worker never advertises it), so it needs no torch guard here.
    if !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && !worker_is_candle(worker)
        && upscale_job_requests_seedvr2(job)
    {
        return false;
    }
    let advertises = |capability: &str| {
        worker
            .capabilities
            .iter()
            .any(|owned| owned.as_str() == capability)
    };
    if !advertises(required_capability(job)) {
        return false;
    }
    // A real (non-dry-run) LoRA training job additionally needs the execute
    // capability, which a worker advertises only when its inference backend is
    // available. Dry-run plan validation needs just the base `lora_train`
    // capability. This keeps a real run queued for a capable worker instead of
    // failing terminally after a torch-less worker claims it.
    if is_real_training_job(job) {
        return advertises(WorkerCapability::LoraTrainExecute.as_str());
    }
    true
}

/// True when a job is a real (non-dry-run) LoRA training run. The training
/// payload defaults to dry-run; only an explicit `dryRun: false` is a real run.
fn is_real_training_job(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::LoraTrain)
        && job.payload.get("dryRun").and_then(Value::as_bool) == Some(false)
}

/// The worker capability a job requires. Person detection/tracking default to
/// the real, model-backed capability served by the Python GPU worker; an
/// explicit `preview: true` payload requests the Rust utility worker's
/// procedural preview capability instead — so a real job never routes to the
/// placeholder. Mirrors the dry-run training capability split.
fn required_capability(job: &JobSnapshot) -> &str {
    match job.job_type {
        JobType::PersonDetect if person_job_is_preview(job) => {
            WorkerCapability::PersonDetectPreview.as_str()
        }
        JobType::PersonTrack if person_job_is_preview(job) => {
            WorkerCapability::PersonTrackPreview.as_str()
        }
        _ => job.job_type.as_str(),
    }
}

/// True when a person detection/tracking job explicitly opts into the procedural
/// preview path (`preview: true`); real model-backed runs are the default.
fn person_job_is_preview(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::PersonDetect | JobType::PersonTrack)
        && job.payload.get("preview").and_then(Value::as_bool) == Some(true)
}

#[derive(Debug, Clone, Copy)]
struct DispatchScore {
    has_utilization: bool,
    free_memory_mb: f64,
    memory_usage_percent: f64,
    gpu_load_percent: f64,
    warm_model: bool,
}

fn dispatch_score(job: &JobSnapshot, worker: &WorkerSnapshot) -> DispatchScore {
    let utilization = worker.utilization.as_ref();
    let total = utilization.and_then(|item| item.memory_total_mb);
    let used = utilization.and_then(|item| item.memory_used_mb);
    let gpu_load = utilization.and_then(|item| item.gpu_load_percent);
    // Derive free memory only from data the worker actually reported: an explicit
    // free reading, or total-minus-used when both are present. A worker that
    // reports no utilization at all must stay `has_utilization = false` so the
    // auto-GPU dispatcher leaves it alone — the earlier `total.checked_sub(used)`
    // with total/used defaulted to 0 yielded `Some(0)`, which scored a
    // no-utilization worker as a real GPU with 0 MB free. That made the
    // Apple-Silicon `mlx` worker (whose nvidia-smi probe finds nothing, so it
    // never reports utilization) always look "worse" than the idle Python `mps`
    // worker, so it deferred every MLX-eligible job to `mps` — which deferred the
    // same job right back to `mlx` (`should_defer_image_to_mlx_worker`), leaving
    // it queued on "Waiting for an available worker" forever (sc-3289 regression).
    let free = utilization
        .and_then(|item| item.memory_free_mb)
        .or_else(|| match (total, used) {
            (Some(total), Some(used)) => total.checked_sub(used),
            _ => None,
        });
    let memory_usage_percent = match (total, used) {
        (Some(total), Some(used)) if total > 0 => used as f64 / total as f64 * 100.0,
        _ => 0.0,
    };
    DispatchScore {
        has_utilization: free.is_some() || gpu_load.is_some() || total.is_some(),
        free_memory_mb: free.unwrap_or(0) as f64,
        memory_usage_percent,
        gpu_load_percent: gpu_load.unwrap_or(0.0),
        warm_model: job_matches_loaded_model(job, worker),
    }
}

fn dispatch_score_is_better(candidate: DispatchScore, current: DispatchScore) -> bool {
    if !candidate.has_utilization || !current.has_utilization {
        return false;
    }

    let free_delta = candidate.free_memory_mb - current.free_memory_mb;
    let load_delta = current.gpu_load_percent - candidate.gpu_load_percent;
    let usage_delta = current.memory_usage_percent - candidate.memory_usage_percent;
    // Prefer a meaningfully freer/lower-load GPU, with tolerance bands so two
    // similarly healthy GPUs do not trade claims back and forth on tiny deltas.
    let candidate_is_not_worse = candidate.free_memory_mb + DISPATCH_MEMORY_NOT_WORSE_TOLERANCE_MB
        >= current.free_memory_mb
        && candidate.gpu_load_percent
            <= current.gpu_load_percent + DISPATCH_LOAD_NOT_WORSE_TOLERANCE_PERCENT
        && candidate.memory_usage_percent
            <= current.memory_usage_percent + DISPATCH_MEMORY_USAGE_NOT_WORSE_TOLERANCE_PERCENT;
    let candidate_relief = free_delta >= DISPATCH_MEMORY_RELIEF_THRESHOLD_MB
        || load_delta >= DISPATCH_LOAD_RELIEF_THRESHOLD_PERCENT
        || usage_delta >= DISPATCH_MEMORY_USAGE_RELIEF_THRESHOLD_PERCENT;

    if candidate_is_not_worse && candidate_relief {
        return true;
    }
    if candidate_is_not_worse && candidate.warm_model && !current.warm_model {
        return true;
    }
    (current.free_memory_mb < DISPATCH_LOW_MEMORY_THRESHOLD_MB
        && candidate.free_memory_mb >= DISPATCH_HEALTHY_MEMORY_THRESHOLD_MB)
        || (current.gpu_load_percent >= DISPATCH_HIGH_LOAD_THRESHOLD_PERCENT
            && candidate.gpu_load_percent <= DISPATCH_RECOVERED_LOAD_THRESHOLD_PERCENT)
        || (current.memory_usage_percent >= DISPATCH_HIGH_MEMORY_USAGE_THRESHOLD_PERCENT
            && candidate.memory_usage_percent <= DISPATCH_RECOVERED_MEMORY_USAGE_THRESHOLD_PERCENT)
}

fn choose_claimable_job(rows: Vec<JobSnapshot>, worker: &WorkerSnapshot) -> Option<JobSnapshot> {
    let compatible = rows
        .into_iter()
        .filter(|job| worker_supports_job(worker, job))
        .collect::<Vec<_>>();
    let first = compatible.first()?;
    if is_non_gpu_job_type(first.job_type.as_str()) || first.requested_gpu != "auto" {
        return compatible.into_iter().next();
    }
    if let Some(explicit_gpu_job) = compatible
        .iter()
        .find(|job| !is_non_gpu_job_type(job.job_type.as_str()) && job.requested_gpu != "auto")
        .cloned()
    {
        return Some(explicit_gpu_job);
    }
    compatible
        .iter()
        .find(|job| job_matches_loaded_model(job, worker))
        .cloned()
        .or_else(|| compatible.into_iter().next())
}

fn job_matches_loaded_model(job: &JobSnapshot, worker: &WorkerSnapshot) -> bool {
    if job.requested_gpu != "auto"
        || is_non_gpu_job_type(job.job_type.as_str())
        || worker.loaded_models.is_empty()
    {
        return false;
    }
    let keys = desired_model_keys(&job.payload);
    worker
        .loaded_models
        .iter()
        .any(|loaded_model| keys.iter().any(|key| key == loaded_model))
}

fn desired_model_keys(payload: &Map<String, Value>) -> Vec<String> {
    let mut keys = Vec::new();
    push_string_value(&mut keys, payload.get("model"));
    push_string_value(&mut keys, payload.get("repo"));
    if let Some(advanced) = payload.get("advanced").and_then(Value::as_object) {
        push_string_value(&mut keys, advanced.get("modelRepo"));
        push_string_value(&mut keys, advanced.get("repo"));
    }
    keys.sort();
    keys.dedup();
    keys
}

fn push_string_value(output: &mut Vec<String>, value: Option<&Value>) {
    if let Some(value) = value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        output.push(value.to_owned());
    }
}

fn normalize_requested_gpu(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "auto".to_owned()
    } else if trimmed.eq_ignore_ascii_case("auto") || trimmed.eq_ignore_ascii_case("cpu") {
        trimmed.to_ascii_lowercase()
    } else {
        trimmed.to_owned()
    }
}

// Keep GPU-required job types in sync with the native worker dispatch
// (crates/sceneworks-worker/src/lib.rs::run_utility_job) and
// apps/web/src/screens/QueueScreen.jsx::gpuRequiredJobTypes.
// `lora_train` is GPU-required like generation, but its worker capability is
// advertised separately (the dry-run plan validation needs no inference backend;
// real execution is gated per platform in story 1417).
fn job_requires_gpu(job_type: &JobType) -> bool {
    matches!(
        job_type,
        JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageVqa
            | JobType::ImageInterleave
            | JobType::ImageUpscale
            | JobType::ImageDetail
            | JobType::ImageSegment
            | JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::VideoUpscale
            | JobType::PersonReplace
            | JobType::LoraTrain
            | JobType::TrainingCaption
            | JobType::DatasetAnalysis
            | JobType::DatasetUpscale
            | JobType::DatasetFaceAnalysis
            | JobType::FaceLikenessCompare
    )
}

fn placeholders_from(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn sort_json_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            let mut entries = map
                .iter_mut()
                .map(|(key, value)| {
                    sort_json_value(value);
                    (key.clone(), value.clone())
                })
                .collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            map.clear();
            map.extend(entries);
        }
        Value::Array(items) => {
            for item in items {
                sort_json_value(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

#[cfg(test)]
mod active_statuses_sql_tests {
    use super::{active_statuses_sql, ACTIVE_STATUSES};

    /// Anti-drift guard for sc-4207 / F-CORE-3: the five `status in (...)` SQL
    /// statements now interpolate [`active_statuses_sql`] instead of a
    /// copy-pasted literal, so the generated list must stay exactly in sync with
    /// [`ACTIVE_STATUSES`] — every status quoted, comma-separated, none dropped.
    #[test]
    fn sql_list_matches_active_statuses_const() {
        let expected = ACTIVE_STATUSES
            .iter()
            .map(|status| format!("'{status}'"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(active_statuses_sql(), expected);

        // Each status appears as a quoted token, guarding against a future const
        // edit that silently fails to reach the SQL filters.
        for status in ACTIVE_STATUSES {
            assert!(
                active_statuses_sql().contains(&format!("'{status}'")),
                "active status {status:?} missing from SQL list"
            );
        }
    }
}

#[cfg(test)]
mod termination_failure_error_tests {
    //! sc-4881 signal attribution + sc-5567 job-kind-aware OOM remediation: a signal-9
    //! (SIGKILL/OOM) death must give guidance that fits the dead job — count/resolution
    //! for an image batch, frames for video, gradient checkpointing only for training —
    //! and non-OOM uncatchable deaths must keep naming their real cause. sc-6320: a
    //! non-signal non-zero exit (a self-terminated process / panic) names the exit code.
    use super::{termination_failure_error, JobType};

    #[test]
    fn signal_9_image_batch_points_at_count_not_gradient_checkpointing() {
        let msg = termination_failure_error(Some(9), None, Some(&JobType::ImageGenerate));
        assert!(msg.contains("signal 9 (SIGKILL)"), "{msg}");
        assert!(msg.contains("out-of-memory"), "{msg}");
        assert!(msg.contains("image count or resolution"), "{msg}");
        // The old training-only hint must NOT leak onto an image batch (the sc-5567 bug).
        assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
        assert!(!msg.contains("training step"), "{msg}");
    }

    #[test]
    fn signal_9_training_keeps_gradient_checkpointing_hint() {
        let msg = termination_failure_error(Some(9), None, Some(&JobType::LoraTrain));
        assert!(msg.contains("Gradient Checkpointing"), "{msg}");
        assert!(msg.contains("training step"), "{msg}");
    }

    #[test]
    fn signal_9_video_points_at_frame_count() {
        let msg = termination_failure_error(Some(9), None, Some(&JobType::VideoGenerate));
        assert!(msg.contains("out-of-memory"), "{msg}");
        assert!(msg.contains("frame count"), "{msg}");
        assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
    }

    #[test]
    fn signal_9_unknown_and_idle_fall_back_to_generic_oom() {
        // No active job (worker died idle) and an unmapped job kind both get the generic
        // OOM hint rather than a misleading training/image/video-specific one.
        for job_type in [None, Some(&JobType::Unknown("future".to_owned()))] {
            let msg = termination_failure_error(Some(9), None, job_type);
            assert!(msg.contains("out-of-memory"), "{msg}");
            assert!(!msg.contains("Gradient Checkpointing"), "{msg}");
            assert!(!msg.contains("image count"), "{msg}");
            assert!(!msg.contains("frame count"), "{msg}");
        }
    }

    #[test]
    fn non_oom_signals_keep_their_own_cause_regardless_of_job_kind() {
        // SIGABRT / SIGSEGV are not OOM, so the job kind must not turn them into one.
        let abort = termination_failure_error(Some(6), None, Some(&JobType::ImageGenerate));
        assert!(abort.contains("signal 6 (SIGABRT)"), "{abort}");
        assert!(abort.contains("GPU/Metal command-buffer abort"), "{abort}");
        assert!(!abort.contains("out-of-memory"), "{abort}");

        let segv = termination_failure_error(Some(11), None, Some(&JobType::LoraTrain));
        assert!(segv.contains("signal 11 (SIGSEGV)"), "{segv}");
        assert!(segv.contains("segmentation fault"), "{segv}");
        assert!(!segv.contains("Gradient Checkpointing"), "{segv}");
    }

    #[test]
    fn panic_exit_code_101_self_names_without_claiming_a_signal() {
        // sc-6320: a Rust panic unwinds to exit 101 (no signal). The attribution must
        // name the panic + code and must NOT fabricate a signal or an OOM hint.
        let msg = termination_failure_error(None, Some(101), Some(&JobType::ImageGenerate));
        assert!(msg.contains("panicked"), "{msg}");
        assert!(msg.contains("101"), "{msg}");
        assert!(!msg.contains("signal"), "{msg}");
        assert!(!msg.contains("out-of-memory"), "{msg}");
    }

    #[test]
    fn other_non_zero_exit_reports_the_raw_code() {
        // A non-zero, non-101 self-exit reports the raw code so the cause is greppable.
        let msg = termination_failure_error(None, Some(2), None);
        assert!(msg.contains("exited unexpectedly (code 2)"), "{msg}");
        assert!(!msg.contains("signal"), "{msg}");
    }

    #[test]
    fn signal_takes_precedence_when_both_are_present() {
        // Defensive: if both somehow arrive, the signal (the harder cause) wins.
        let msg = termination_failure_error(Some(11), Some(101), None);
        assert!(msg.contains("signal 11 (SIGSEGV)"), "{msg}");
        assert!(!msg.contains("101"), "{msg}");
    }
}

#[cfg(test)]
mod candle_routing_tests {
    //! Candle (Windows/CUDA) SDXL lane routing (epic 3672, sc-3678): the candle worker serves a
    //! gated, narrow SDXL/RealVisXL **txt2img-only** lane and must defer every other shape to the
    //! Python torch worker. These tests pin the lane boundary (`image_request_candle_eligible`) and
    //! the full claim gate (`worker_supports_job` via the `candle` marker capability).
    use super::*;
    use serde_json::{json, Value};

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().expect("test value is an object").clone()
    }

    /// A queued `image_generate` job carrying `payload`, built via serde so the test never has to
    /// spell out the full `JobSnapshot` field set.
    fn image_generate_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_1",
            "type": "image_generate",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-12T00:00:00Z",
            "updatedAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// A queued `image_edit` job carrying `payload` — the distinct job type the API stamps for the
    /// Image Studio/Editor "plain Image Edit" (`mode == "edit_image"`, `apps/rust-api` generation.rs).
    /// The candle edit lanes (sc-5487) are reached via this type, so the routing/claim tests must probe
    /// it directly rather than only via `image_generate_job`.
    fn image_edit_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_1",
            "type": "image_edit",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-12T00:00:00Z",
            "updatedAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// A worker on a real CUDA gpu index advertising `capabilities` (string ids). The candle worker
    /// carries the `candle` marker; the torch worker on the same box does not.
    fn gpu_worker(capabilities: &[&str]) -> WorkerSnapshot {
        serde_json::from_value(json!({
            "id": "worker_1",
            "gpuId": "0",
            "status": "idle",
            "capabilities": capabilities,
            "loadedModels": [],
            "registeredAt": "2026-06-12T00:00:00Z",
            "lastSeenAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot")
    }

    // Mirrors the real candle advertised set (`with_candle_capabilities`): `image_generate` (derived)
    // plus the `image_edit` carve-out (sc-5487 edit lanes) and the `candle` lane marker.
    const CANDLE_CAPS: &[&str] = &["gpu", "image_generate", "image_edit", "candle"];
    // The Python torch worker advertises the broad image surface but no `candle` marker.
    const TORCH_CAPS: &[&str] = &["gpu", "image_generate", "image_edit", "image_detail"];

    #[test]
    fn candle_routed_models_plain_txt2img_are_eligible() {
        // SDXL/RealVisXL (sc-3678) + the image families wired in sc-5096 — every base txt2img id, now
        // INCLUDING base `z_image` (sc-8679): the registered candle `z_image` base generator makes a plain
        // txt2img `z_image` job candle-eligible, the base sibling of `z_image_turbo`. (Its strict-pose
        // control lane is branched out earlier in `image_job_is_candle_eligible`; its edit shapes reject
        // below — see `new_candle_families_conditioning_shapes_fall_back_to_torch`.)
        for model in CANDLE_ROUTED_MODELS {
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
                "{model} plain txt2img should be candle-eligible"
            );
        }
    }

    #[test]
    fn base_z_image_txt2img_is_candle_eligible_but_edit_shapes_are_not() {
        // sc-8679: base `z_image` plain txt2img rides the candle lane (the base sibling of z_image_turbo);
        // its edit / reference / mask conditioning shapes still defer to the Python torch worker (no candle
        // base-z-image edit provider — that is a separate story).
        assert!(
            image_request_candle_eligible("z_image", &object(json!({ "prompt": "a red fox" }))),
            "base z_image plain txt2img must be candle-eligible (sc-8679)"
        );
        for payload in [
            json!({ "prompt": "p", "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "prompt": "p", "referenceAssetId": "a" }),
            json!({ "prompt": "p", "maskAssetId": "a" }),
        ] {
            assert!(
                !image_request_candle_eligible("z_image", &object(payload.clone())),
                "base z_image conditioning shape must fall back to torch: {payload}"
            );
        }
    }

    #[test]
    fn non_candle_families_and_variants_are_never_candle_eligible() {
        // A family with no candle provider at all (`bernini_image`) AND the still-unwired weight/shape
        // variants of wired families (edit ids) all stay on the Python torch worker.
        // (chroma / kolors / sensenova ARE candle-routed now — sc-5484 / sc-5576 — for txt2img; the
        // FLUX.2-klein `_kv` / `_true_v2` weight variants are too — sc-7459 — see the dedicated test below.)
        for model in ["bernini_image", "z_image_edit", "qwen_image_edit"] {
            assert!(
                !image_request_candle_eligible(model, &object(json!({ "prompt": "p" }))),
                "{model} must fall back to the Python worker"
            );
        }
    }

    #[test]
    fn flux2_klein_weight_variants_route_txt2img_to_candle() {
        // sc-7459 (epic 6564 story 3): both klein weight variants serve plain txt2img on the candle lane
        // via the shared `flux2_klein_9b` loader — a weights swap, not a new arch.
        for model in ["flux2_klein_9b_kv", "flux2_klein_9b_true_v2"] {
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
                "{model} plain txt2img should be candle-eligible"
            );
        }
        // ...but their reference/edit shapes are NOT in scope (txt2img weight parity only). The `_kv`
        // checkpoint's whole point is the reference-edit KV-cache accel — that stays on the Python torch
        // worker (the candle lane has no klein edit path), same as every other candle conditioning shape.
        for payload in [
            json!({ "referenceAssetId": "a" }),
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        ] {
            assert!(
                !image_request_candle_eligible("flux2_klein_9b_kv", &object(payload.clone())),
                "flux2_klein_9b_kv conditioning shape must fall back to torch: {payload}"
            );
        }
    }

    #[test]
    fn new_candle_families_conditioning_shapes_fall_back_to_torch() {
        // Every candle image family is txt2img-only on candle: any conditioning shape defers to torch
        // (the worker advertises none of these, so this is the no-silently-dropped-control boundary).
        let cases = [
            (
                "z_image_turbo",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("flux_dev", json!({ "referenceAssetId": "a" })),
            ("flux_schnell", json!({ "loras": [{ "name": "x" }] })),
            (
                "qwen_image",
                json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
            ),
            // NB: `flux2_klein_9b` + `edit_image` is NOT here — sc-5487 wired it to the candle `Flux2Edit`
            // lane (asserted via `image_job_is_candle_eligible` in `candle_worker_claims_*`), like SDXL
            // edit. The txt2img gate still rejects it (it rejects all `edit_image`), but it no longer
            // "falls back to torch" at the router level.
            // sc-5484 / sc-5576: Chroma / Kolors / SenseNova-U1 are pure T2I on candle. Their MLX-only
            // conditioning shapes (Kolors edit / IP-reference / pose-control; SenseNova edit) defer.
            (
                "chroma1_hd",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            (
                "kolors",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("kolors", json!({ "referenceAssetId": "a" })),
            (
                "kolors",
                json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
            ),
            (
                "sensenova_u1_8b",
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            ),
            ("sensenova_u1_8b_fast", json!({ "referenceAssetId": "a" })),
        ];
        for (model, payload) in cases {
            assert!(
                !image_request_candle_eligible(model, &object(payload.clone())),
                "{model} conditioning shape must fall back to torch: {payload}"
            );
        }
    }

    #[test]
    fn ideogram_candle_txt2img_and_edit_route_to_candle() {
        // sc-6597 (epic 6561): `ideogram_4` + `ideogram_4_turbo` route to the candle lane for plain
        // text-to-image via the generic `image_request_candle_eligible` gate. sc-6598: img2img / Remix +
        // mask inpaint / outpaint now route to candle too — via the bespoke `ideogram_edit_candle_eligible`
        // branch in `image_job_is_candle_eligible` (the generic gate stays txt2img-only, like every other
        // candle edit family). A pure `referenceAssetId` (IP-Adapter — no candle Ideogram path) still
        // defers to torch.
        for model in ["ideogram_4", "ideogram_4_turbo"] {
            // Plain txt2img → the generic gate.
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "an aurora" }))),
                "{model} plain txt2img must be candle-eligible"
            );
            // sc-9607/sc-9983: a Q8/Q4 tier-select stays on candle (ideogram is in CANDLE_QUANT_MODELS —
            // the packed q4/q8 turnkeys load off-Mac; the `mlxQuantize` value picks the subdir).
            for bits in [8, 4] {
                assert!(
                    image_request_candle_eligible(
                        model,
                        &object(
                            json!({ "prompt": "an aurora", "advanced": { "mlxQuantize": bits } })
                        )
                    ),
                    "{model} Q{bits} tier-select should stay on candle"
                );
            }
            // Edit shapes → the bespoke dispatcher branch (img2img, inpaint, outpaint all need a source).
            for payload in [
                json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a" }),
                json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a", "maskAssetId": "m" }),
                json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a", "fit_mode": "outpaint" }),
            ] {
                assert!(
                    ideogram_edit_candle_eligible(&object(payload.clone())),
                    "{model} edit shape must be candle-eligible: {payload}"
                );
                assert!(
                    image_job_is_candle_eligible(&image_edit_job(payload.clone())),
                    "{model} edit job must route to candle: {payload}"
                );
                // The generic txt2img gate still rejects the edit_image family (the bespoke lane handles it).
                assert!(!image_request_candle_eligible(model, &object(payload)));
            }
            // `edit_image` WITHOUT a source → nothing to edit → not this lane.
            assert!(!ideogram_edit_candle_eligible(&object(json!({
                "model": model, "mode": "edit_image"
            }))));
            // Pure IP-Adapter reference (Ideogram has no candle IP path) still defers to torch.
            assert!(!image_request_candle_eligible(
                model,
                &object(json!({ "referenceAssetId": "a" }))
            ));
        }
    }

    #[test]
    fn boogu_text_to_image_and_edit_route_to_candle() {
        // sc-7524 (epic 6831): the candle parity of `boogu_text_to_image_and_edit_route_to_mlx`. The
        // three Boogu ids are in `CANDLE_ROUTED_MODELS`; Base + Turbo are pure txt2img (the generic gate),
        // and `boogu_image_edit`'s `edit_image` shape routes via the bespoke `boogu_edit_candle_eligible`
        // branch (the source `Reference` is resolved in-lane by `generate_candle_stream`, like Ideogram).
        for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
            // Plain txt2img → the generic gate (the edit checkpoint can also T2I, mirroring MLX).
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "a red panda" }))),
                "{model} plain txt2img must be candle-eligible"
            );
        }
        // Edit (source instruction) is the Edit checkpoint's capability ONLY.
        let edit_payload = |model: &str| json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" });
        // `boogu_image_edit` + edit_image + source → the bespoke branch claims it for candle.
        assert!(boogu_edit_candle_eligible(&object(edit_payload(
            "boogu_image_edit"
        ))));
        assert!(image_job_is_candle_eligible(&image_edit_job(edit_payload(
            "boogu_image_edit"
        ))));
        // The generic txt2img gate still rejects the edit_image family (the bespoke lane handles it).
        assert!(!image_request_candle_eligible(
            "boogu_image_edit",
            &object(edit_payload("boogu_image_edit"))
        ));
        // Base / Turbo do NOT edit — an edit_image job on them is not candle-eligible (no edit lane; the
        // generic gate rejects edit_image and the boogu edit branch is gated to `boogu_image_edit`).
        assert!(!image_job_is_candle_eligible(&image_edit_job(
            edit_payload("boogu_image")
        )));
        assert!(!image_job_is_candle_eligible(&image_edit_job(
            edit_payload("boogu_image_turbo")
        )));
        // sc-7645: the multi-image picker sends plural `referenceAssetIds` (no `sourceAssetId`) — the
        // bespoke branch still claims it for candle (the Boogu DiT packs up to 5 references).
        assert!(boogu_edit_candle_eligible(&object(json!({
            "model": "boogu_image_edit", "mode": "edit_image",
            "referenceAssetIds": ["a", "b"]
        }))));
        // `edit_image` WITHOUT a source → nothing to edit → not this lane.
        assert!(!boogu_edit_candle_eligible(&object(json!({
            "model": "boogu_image_edit", "mode": "edit_image"
        }))));
        // An empty plural list with no `sourceAssetId` is also nothing to edit.
        assert!(!boogu_edit_candle_eligible(&object(json!({
            "model": "boogu_image_edit", "mode": "edit_image", "referenceAssetIds": []
        }))));
        // sc-9607/sc-9983: a Q8/Q4 tier-select now STAYS on candle (boogu is in CANDLE_QUANT_MODELS — the
        // packed q4/q8 turnkeys load off-Mac, the `mlxQuantize` value picks the subdir). A LoRA still
        // defers (boogu advertises no inference LoRA on candle).
        for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 8 } }))
                ),
                "{model} Q8 tier-select should stay on candle"
            );
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 4 } }))
                ),
                "{model} Q4 tier-select should stay on candle"
            );
            assert!(
                !image_request_candle_eligible(
                    model,
                    &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
                ),
                "{model} with a LoRA must defer to torch (no candle inference LoRA)"
            );
        }
    }

    #[test]
    fn explicit_quantization_falls_back_to_torch_image_and_video() {
        // sc-5099: a candle provider that advertises NO quant (supported_quants: &[]) must route an
        // explicit `advanced.mlxQuantize > 0` to Python rather than silently running dense. chroma1_hd
        // is such a dense-only candle family (contrast the SDXL family, sc-10767, which now advertises
        // Q4/Q8 packed tiers and stays on candle — covered by `sdxl_family_quant_and_lora_stay_on_candle`).
        assert!(!image_request_candle_eligible(
            "chroma1_hd",
            &object(json!({ "advanced": { "mlxQuantize": 8 } }))
        ));
        assert!(!image_request_candle_eligible(
            "qwen_image",
            &object(json!({ "advanced": { "mlxQuantize": 4 } }))
        ));
        assert!(!video_request_candle_eligible(
            "wan_2_2",
            &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } }))
        ));
        // Dense (<= 0) or absent quant leaves a dense candle family on its native path → still eligible.
        assert!(image_request_candle_eligible(
            "chroma1_hd",
            &object(json!({ "advanced": { "mlxQuantize": 0 } }))
        ));
        assert!(image_request_candle_eligible(
            "chroma1_hd",
            &object(json!({ "advanced": { "steps": 30 } }))
        ));
    }

    #[test]
    fn sdxl_family_quant_and_lora_stay_on_candle() {
        // sc-10767 (epic 9083): the SDXL family advertises Q4/Q8 packed tiers (candle-gen sc-9416/9527)
        // AND inference LoRA/LoKr on a packed tier (sc-9528), so a quant tier-select AND a LoRA both
        // stay on the candle lane rather than deferring to the retired torch fallback. Mirrors the
        // boogu/lens quant-stays coverage; the inverse of the old dense-only behavior.
        for model in [
            "sdxl",
            "realvisxl",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ] {
            for bits in [8, 4] {
                assert!(
                    image_request_candle_eligible(
                        model,
                        &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": bits } }))
                    ),
                    "{model} Q{bits} tier-select should stay on candle (sc-10767)"
                );
            }
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
                ),
                "{model} with a LoRA should stay on candle (sc-10767)"
            );
        }
    }

    #[test]
    fn lens_quant_and_lora_stay_on_the_candle_lane() {
        // sc-5126: Lens / Lens-Turbo advertise Q4/Q8 + LoRA/LoKr, so — UNLIKE the sc-3675/sc-5096
        // families — a quant request or a LoRA does NOT defer to torch; the candle lane maps both into
        // the LoadSpec.
        for model in ["lens", "lens_turbo"] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": 8 } }))
                ),
                "{model} Q8 request should stay on candle"
            );
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": 4 } }))
                ),
                "{model} Q4 request should stay on candle"
            );
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
                ),
                "{model} with a LoRA should stay on candle"
            );
        }
    }

    #[test]
    fn lens_conditioning_shapes_fall_back_to_torch() {
        // Lens is pure T2I (the port has no img2img/edit/reference/ControlNet), so every conditioning
        // shape still defers to the Python worker — quant/LoRA being allowed does not widen this.
        let cases = [
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "referenceAssetId": "a" }),
            json!({ "maskAssetId": "m" }),
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ];
        for model in ["lens", "lens_turbo"] {
            for case in &cases {
                assert!(
                    !image_request_candle_eligible(model, &object(case.clone())),
                    "{model} conditioning shape must fall back to torch: {case}"
                );
            }
        }
    }

    #[test]
    fn sd3_5_quant_stays_on_candle_but_lora_and_conditioning_defer() {
        // sc-7880 (epic 7982): the candle SD3.5 descriptor advertises supported_quants: [Q4, Q8] but
        // supports_lora: false, so — unlike Lens — an explicit quant request stays on the candle lane
        // while a LoRA (and every conditioning shape) still defers to the Python torch worker.
        for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
            // Plain txt2img is eligible.
            assert!(
                image_request_candle_eligible(model, &object(json!({ "prompt": "a misty fjord" }))),
                "{model} plain txt2img should be candle-eligible"
            );
            // Q8 / Q4 requests stay on candle (descriptor-gated quant, resolved worker-side).
            for bits in [8, 4] {
                assert!(
                    image_request_candle_eligible(
                        model,
                        &object(json!({ "advanced": { "mlxQuantize": bits } }))
                    ),
                    "{model} Q{bits} request should stay on candle"
                );
            }
            // A LoRA defers (SD3.5 has no inference-LoRA candle path yet).
            assert!(
                !image_request_candle_eligible(
                    model,
                    &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
                ),
                "{model} with a LoRA must fall back to torch"
            );
            // Every conditioning shape defers (txt2img only).
            for case in [
                json!({ "mode": "edit_image", "sourceAssetId": "a" }),
                json!({ "referenceAssetId": "a" }),
                json!({ "maskAssetId": "m" }),
                json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
            ] {
                assert!(
                    !image_request_candle_eligible(model, &object(case.clone())),
                    "{model} conditioning shape must fall back to torch: {case}"
                );
            }
        }
    }

    #[test]
    fn krea_lora_and_quant_stay_on_candle_but_conditioning_defers() {
        // sc-7836 (epic 7565 P4) + sc-9607/sc-9983 (epic 9083): the candle `candle-gen-krea` descriptor
        // advertises supports_lora/supports_lokr: true (it merges a `krea_2_raw`-trained adapter at Turbo
        // inference) AND, since sc-9607, `supported_quants: [Q4, Q8]` (a no-op on the already-packed q4/q8
        // turnkey subdir), so BOTH a LoRA and a Q8/Q4 tier-select stay on the candle lane (Krea is in
        // CANDLE_QUANT_LORA_MODELS). Only the conditioning shapes (edit/reference/mask/pose) defer to the
        // Python torch worker. Regression guard for the two missed router un-gates: before sc-7836 a Krea
        // LoRA, and before sc-9983 a Krea Q8/Q4, each hit the no-torch-fallback candle gap off-Mac.
        let model = "krea_2_turbo";
        // Plain txt2img is eligible.
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "an emerald forest" }))),
            "{model} plain txt2img should be candle-eligible"
        );
        // A LoRA stays on candle (descriptor-gated adapter merge, resolved worker-side).
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} with a LoRA should stay on candle"
        );
        // sc-9607/sc-9983: a Q8 / Q4 tier-select now STAYS on candle (the packed turnkey loads off-Mac).
        for bits in [8, 4] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} tier-select should stay on candle"
            );
        }
        // Every conditioning shape defers (txt2img + LoRA only).
        for case in [
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "referenceAssetId": "a" }),
            json!({ "maskAssetId": "m" }),
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ] {
            assert!(
                !image_request_candle_eligible(model, &object(case.clone())),
                "{model} conditioning shape must fall back to torch: {case}"
            );
        }
    }

    #[test]
    fn sdxl_advanced_shapes_fall_back_to_torch() {
        // Every conditioning shape the txt2img candle lane can't honor must be ineligible. A LoRA is NOT
        // in this set anymore (sc-10767): the SDXL family advertises inference LoRA on candle, so a plain
        // LoRA txt2img stays on the candle lane (see `sdxl_family_quant_and_lora_stay_on_candle`). Only
        // the genuine conditioning shapes (img2img / reference / mask / strict-pose) fall back.
        let cases = [
            json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }), // img2img / inpaint / outpaint
            json!({ "referenceAssetId": "asset_1" }),                    // IP-Adapter reference
            json!({ "mode": "edit_image", "sourceAssetId": "a", "maskAssetId": "m" }), // inpaint
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),    // strict-pose ControlNet
        ];
        for case in cases {
            assert!(
                !image_request_candle_eligible("sdxl", &object(case.clone())),
                "sdxl shape must fall back to torch: {case}"
            );
        }
    }

    #[test]
    fn blank_conditioning_ids_are_treated_as_absent() {
        // Whitespace/empty ids are not real conditioning → still plain txt2img → eligible.
        assert!(image_request_candle_eligible(
            "sdxl",
            &object(
                json!({ "referenceAssetId": "  ", "sourceAssetId": "", "advanced": { "poses": [] } })
            )
        ));
    }

    #[test]
    fn candle_worker_claims_txt2img_but_refuses_unsupported_shapes() {
        let candle = gpu_worker(CANDLE_CAPS);
        // Claims the lane — SDXL plus every wired candle image family, all plain txt2img.
        for model in [
            "sdxl",
            "realvisxl",
            // sc-7176: RealVisXL Lightning routes to candle for plain txt2img (forced lightning sampler).
            "realvisxl_lightning",
            "z_image_turbo",
            "flux_dev",
            // sc-7458: FLUX.2-dev (the 32B flagship) routes to candle for plain txt2img off-Mac (loads
            // the dense snapshot + Q4-quantizes at load). Edit (sc-7736) + strict pose (sc-7736) are
            // candle lanes too now — covered by the dedicated assertions below.
            "flux2_dev",
            "qwen_image",
            "chroma1_hd",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
        ] {
            assert!(
                worker_supports_job(
                    &candle,
                    &image_generate_job(json!({ "model": model, "prompt": "a red fox" }))
                ),
                "candle worker should claim {model} plain txt2img"
            );
        }
        // Refuses a family with no candle provider, and a conditioning shape on a wired family —
        // both defer to torch.
        assert!(!worker_supports_job(
            &candle,
            &image_generate_job(json!({ "model": "bernini_image", "prompt": "p" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "kolors",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // sc-5489: `qwen_image` + `advanced.poses` IS now a candle lane (the bespoke strict-pose
        // ControlNet route), so the candle worker claims it (was deferred to torch before this slice).
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "qwen_image",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim qwen_image strict-pose (sc-5489)"
        );
        // sc-5489: `kolors` + `advanced.poses` is also a candle lane now (the Kolors strict-pose
        // ControlNet route), so the candle worker claims it too.
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "kolors",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim kolors strict-pose (sc-5489)"
        );
        // sc-5489: `z_image_turbo` + `advanced.poses` is the LAST strict-pose family wired (the VACE
        // Fun-ControlNet route) — all three (qwen / kolors / z_image) are candle lanes now.
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": "z_image_turbo",
                    "advanced": { "poses": [{ "id": "pose_1" }] }
                }))
            ),
            "candle worker should claim z_image_turbo strict-pose (sc-5489)"
        );
        // sc-5968: plain `sdxl` + poses has NO candle pose lane (SDXL pose ships via InstantID), and
        // the torch `sdxl` adapter has no pose path either — so the candle worker CLAIMS it (to reject
        // with a typed error in the handler) rather than declining → torch silently rendering an
        // unconditioned T2I image. `worker_supports_job` is therefore TRUE here (candle owns it to fail
        // it loudly); the handler's `candle_unsupported_pose_reject` guard does the rejecting.
        assert!(worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "sdxl",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
        // sc-5487: a plain SDXL edit (img2img: `edit_image` + a source) is now a candle lane (the
        // bespoke `SdxlEdit` route), so the candle worker CLAIMS it — it no longer declines → torch.
        assert!(worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "sdxl",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // sc-5487: a FLUX.2-klein edit (`edit_image` + a source) is now the candle `Flux2Edit` lane.
        // klein has no torch path, so the candle worker CLAIMS it (the only off-Mac lane for it).
        assert!(worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "flux2_klein_9b",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // The -kv distill edit has no candle provider yet (needs the reference-K/V cache port) → NOT
        // claimed by candle; it stays on the MLX/torch path.
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "flux2_klein_9b_kv",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))));
        // sc-7736 (epic 6564): FLUX.2-dev edit (`edit_image` + a source) is NOW the candle `Flux2Edit`
        // dev lane (`load_dev`, Q4) — the worker CLAIMS it (was deferred to torch under sc-7458's
        // txt2img-only slice). Multi-reference (the plural `referenceAssetIds`) rides the same lane.
        assert!(image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "flux2_dev",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))));
        assert!(image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "flux2_dev",
            "mode": "edit_image",
            "sourceAssetId": "asset_1",
            "referenceAssetIds": ["asset_1", "asset_2"]
        }))));
        // sc-7736: a pure-reference flux2_dev job (a `referenceAssetId`, NO `edit_image` source, NO poses)
        // is neither the edit lane (needs `edit_image` + a source) nor the control lane (needs poses), so
        // the txt2img gate rejects the reference shape → it still defers to torch.
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "flux2_dev",
            "referenceAssetId": "asset_1"
        }))));
        // sc-7736: FLUX.2-dev strict pose (`advanced.poses`, not edit) is the candle `Flux2Control`
        // Fun-Controlnet-Union lane — the worker CLAIMS it (the 4th wired strict-pose family). A pose job
        // with no poses array is plain txt2img (claimed by the generic candle lane, not the control one).
        assert!(image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "flux2_dev",
            "advanced": { "poses": [{ "keypoints": [] }] }
        }))));
        assert!(flux2_dev_control_candle_eligible(&object(json!({
            "advanced": { "poses": [{ "keypoints": [] }] }
        }))));
        // An `edit_image` flux2_dev job is the edit lane, not the control lane (disjoint gates).
        assert!(!flux2_dev_control_candle_eligible(&object(json!({
            "mode": "edit_image",
            "advanced": { "poses": [{ "keypoints": [] }] }
        }))));
        // sc-5487: a Qwen-Image-Edit edit (`edit_image` + a source) is now the candle `QwenEdit` lane
        // (dual-latent reference editing). Off-Mac this was a torch fallback; the candle worker CLAIMS
        // it. The `-2511_lightning` distill (sc-6220) is the same `-2511` base with the lightx2v 4-step
        // LoRA folded into the MMDiT at load, so it is candle-claimed too.
        for model in [
            "qwen_image_edit",
            "qwen_image_edit_2509",
            "qwen_image_edit_2511",
            "qwen_image_edit_2511_lightning",
        ] {
            assert!(
                worker_supports_job(
                    &candle,
                    &image_generate_job(json!({
                        "model": model,
                        "mode": "edit_image",
                        "sourceAssetId": "asset_1"
                    }))
                ),
                "candle worker should claim {model} edit (sc-5487 / sc-6220)"
            );
        }
        // A Qwen-Image-Edit job with no source image is not the edit lane → not claimed (would defer).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "qwen_image_edit",
            "mode": "edit_image"
        }))));
    }

    #[test]
    fn torch_worker_claims_everything_the_candle_worker_defers() {
        // The co-resident Python torch worker (no `candle` marker) is ungated here: it claims the
        // shapes the candle worker refused, so nothing is stranded — EXCEPT the unsupported-pose shapes
        // the candle worker now owns-to-reject (sc-5968, asserted at the end of this test): torch
        // declines those so it can't silently render an unconditioned T2I image.
        let torch = gpu_worker(TORCH_CAPS);
        // A family with no candle provider, and a conditioning shape on a wired family.
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({ "model": "bernini_image", "prompt": "p" }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "kolors",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "qwen_image",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "sdxl",
                "mode": "edit_image",
                "sourceAssetId": "asset_1"
            }))
        ));
        // sc-5968: but torch DECLINES the unsupported-pose shape the candle worker owns-to-reject
        // (sdxl + poses) — so it can't silently render an unconditioned T2I; only candle takes it (and
        // rejects). On Mac the same shape is MLX-served, so the `mlx` worker still claims it (asserted
        // in `unsupported_pose_is_owned_by_candle_declined_by_torch_served_by_mlx`).
        assert!(!worker_supports_job(
            &torch,
            &image_generate_job(json!({
                "model": "sdxl",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ));
    }

    /// sc-5968: the unsupported-pose routing across the three GPU workers — candle OWNS it (to reject),
    /// torch DECLINES it (no silent T2I), and the Mac `mlx` worker still SERVES it (no Mac regression,
    /// `sdxl_mlx_eligible` is unconditional). Plus: the wired candle pose families are unaffected, and
    /// `image_job_is_candle_eligible` still reports sdxl+poses as NOT candle-*served* (it's owned only
    /// to reject — the distinction the worker's dispatch guard keys on).
    #[test]
    fn unsupported_pose_is_owned_by_candle_declined_by_torch_served_by_mlx() {
        let candle = gpu_worker(CANDLE_CAPS);
        let torch = gpu_worker(TORCH_CAPS);
        let mlx: WorkerSnapshot = serde_json::from_value(json!({
            "id": "worker_mlx",
            "gpuId": "mlx",
            "status": "idle",
            "capabilities": ["gpu", "image_generate"],
            "loadedModels": [],
            "registeredAt": "2026-06-16T00:00:00Z",
            "lastSeenAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot");
        let sdxl_pose = image_generate_job(
            json!({ "model": "sdxl", "advanced": { "poses": [{ "id": "p" }] } }),
        );

        assert!(image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        assert!(worker_supports_job(&candle, &sdxl_pose), "candle owns it");
        assert!(
            !worker_supports_job(&torch, &sdxl_pose),
            "torch declines it"
        );
        assert!(worker_supports_job(&mlx, &sdxl_pose), "mlx still serves it");
        // It is NOT candle-*served* (only owned-to-reject); the worker's dispatch guard rejects it.
        assert!(!image_job_is_candle_eligible(&sdxl_pose));

        // A wired candle pose family is NOT a reject shape, and edit_image is never a reject shape.
        assert!(!image_request_candle_pose_reject(
            "qwen_image",
            &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        // sc-7736: flux2_dev now HAS a candle pose lane (Flux2Control), so its pose job is served, not
        // rejected.
        assert!(!image_request_candle_pose_reject(
            "flux2_dev",
            &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        assert!(!image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "mode": "edit_image", "advanced": { "poses": [{ "id": "p" }] } }))
        ));
        // No poses → not a reject shape (plain txt2img stays candle-eligible).
        assert!(!image_request_candle_pose_reject(
            "sdxl",
            &object(json!({ "prompt": "a fox" }))
        ));
    }

    // ---- Candle video lane (sc-5097) ----

    /// A queued `video_generate` job carrying `payload`.
    fn video_generate_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_v",
            "type": "video_generate",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    // The candle worker on the video lane advertises `video_generate` + the `candle` marker.
    const CANDLE_VIDEO_CAPS: &[&str] = &["gpu", "video_generate", "candle"];
    const TORCH_VIDEO_CAPS: &[&str] = &["gpu", "video_generate"];

    #[test]
    fn candle_routed_video_models_are_eligible_in_their_native_shape() {
        // txt2video lane: the 5B, ltx, and the 14B T2V (text-only) are eligible for text_to_video.
        for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b"] {
            assert!(
                video_request_candle_eligible(
                    model,
                    &object(json!({ "mode": "text_to_video", "prompt": "a river at dawn" }))
                ),
                "{model} text_to_video should be candle-eligible"
            );
        }
        // image→video lane: the 14B I2V + SVD are eligible only with the i2v mode + a source image
        // (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            assert!(
                video_request_candle_eligible(
                    model,
                    &object(
                        json!({ "mode": "image_to_video", "sourceAssetId": "asset_1", "prompt": "p" })
                    )
                ),
                "{model} image_to_video with a source should be candle-eligible"
            );
        }
    }

    #[test]
    fn non_candle_video_models_and_conditioned_shapes_fall_back() {
        // `ltx_2_3_eros` now routes to candle for plain text_to_video (sc-5495 — it's a full dense
        // LTX-2.3 fine-tune on the `ltx_2_3_distilled` engine), but any conditioned eros shape stays on
        // the Python torch worker (the candle LTX lane is txt2video-only).
        assert!(
            video_request_candle_eligible(
                "ltx_2_3_eros",
                &object(json!({ "mode": "text_to_video" }))
            ),
            "ltx_2_3_eros text_to_video must route to the candle lane"
        );
        assert!(
            !video_request_candle_eligible(
                "ltx_2_3_eros",
                &object(json!({ "mode": "first_last_frame" }))
            ),
            "a conditioned ltx_2_3_eros shape must fall back to the Python worker"
        );
        // A genuinely non-candle video model stays on torch.
        assert!(
            !video_request_candle_eligible(
                "some_unported_model",
                &object(json!({ "mode": "text_to_video" }))
            ),
            "an unported model must fall back to the Python worker"
        );
        // A txt2video model in any conditioned shape (default/i2v mode, a source, or a LoRA) → torch.
        let cases = [
            json!({ "prompt": "p" }), // no mode → defaults to i2v
            json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
            json!({ "mode": "first_last_frame" }),
            json!({ "mode": "text_to_video", "sourceAssetId": "a" }), // txt mode but conditioned
            json!({ "mode": "text_to_video", "loras": [{ "name": "x" }] }),
        ];
        for case in cases {
            assert!(
                !video_request_candle_eligible("wan_2_2", &object(case.clone())),
                "wan_2_2 shape must fall back to torch: {case}"
            );
        }
        // The 14B T2V is text-only: any image_to_video / sourced shape falls back to torch (sc-5175).
        for case in [
            json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
            json!({ "mode": "text_to_video", "sourceAssetId": "a" }),
        ] {
            assert!(
                !video_request_candle_eligible("wan_2_2_t2v_14b", &object(case.clone())),
                "wan_2_2_t2v_14b conditioned shape must fall back to torch: {case}"
            );
        }
        // The 14B I2V + SVD are image→video only: a txt2video shape or an i2v with no source → torch
        // (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            for case in [
                json!({ "mode": "text_to_video", "prompt": "p" }),
                json!({ "mode": "image_to_video" }), // i2v but no source image
            ] {
                assert!(
                    !video_request_candle_eligible(model, &object(case.clone())),
                    "{model} non-i2v shape must fall back to torch: {case}"
                );
            }
        }
        // SVD has no candle LoRA slot, so a LoRA even on its valid i2v shape still falls back; the
        // Wan-14B I2V now ACCEPTS a user LoRA on candle (sc-10539) — see `candle_wan_14b_video_accepts_user_loras`.
        assert!(
            !video_request_candle_eligible(
                "svd",
                &object(
                    json!({ "mode": "image_to_video", "sourceAssetId": "a", "loras": [{ "name": "x" }] })
                )
            ),
            "svd (no candle LoRA slot) must fall back to torch on an i2v+LoRA shape"
        );
    }

    #[test]
    fn candle_wan_14b_video_accepts_user_loras() {
        // sc-10539: the Wan-14B MoE engines advertise `supports_lora` and their candle worker path
        // (`candle_resolve_wan_adapters`) applies each user LoRA — including an external ComfyUI file
        // read in place — so a LoRA-carrying job stays on candle instead of the old blanket exclusion
        // (there is no torch fallback now; epic 8283). GPU-validated: an external `Wan/detailz-wan`
        // adapter rendered a candle Wan-14B clip that differs from the no-LoRA baseline at the same seed.
        assert!(
            video_request_candle_eligible(
                "wan_2_2_t2v_14b",
                &object(json!({ "mode": "text_to_video", "loras": [{ "id": "external_x" }] }))
            ),
            "wan_2_2_t2v_14b text_to_video + user LoRA must stay on candle"
        );
        assert!(
            video_request_candle_eligible(
                "wan_2_2_i2v_14b",
                &object(json!({
                    "mode": "image_to_video",
                    "sourceAssetId": "a",
                    "loras": [{ "id": "external_x" }],
                }))
            ),
            "wan_2_2_i2v_14b i2v + source + user LoRA must stay on candle"
        );
        // Families whose candle provider advertises no LoRA slot still refuse a LoRA (wan-5B TI2V / LTX / SVD).
        for (model, payload) in [
            (
                "wan_2_2",
                json!({ "mode": "text_to_video", "loras": [{ "id": "x" }] }),
            ),
            (
                "ltx_2_3",
                json!({ "mode": "text_to_video", "loras": [{ "id": "x" }] }),
            ),
            (
                "svd",
                json!({ "mode": "image_to_video", "sourceAssetId": "a", "loras": [{ "id": "x" }] }),
            ),
        ] {
            assert!(
                !video_request_candle_eligible(model, &object(payload.clone())),
                "{model} has no candle LoRA slot — a LoRA job must not route to candle: {payload}"
            );
        }
    }

    #[test]
    fn candle_vace_modes_eligible_with_required_assets() {
        // replace_person (PersonReplace): needs the source clip + person track + character.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({
                "sourceClipAssetId": "clip_1",
                "personTrackId": "track_1",
                "characterId": "char_1"
            })),
            &JobType::PersonReplace
        ));
        // extend_clip (VideoExtend): needs a source clip.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2_t2v_14b",
            &object(json!({ "sourceClipAssetId": "clip_1" })),
            &JobType::VideoExtend
        ));
        // video_bridge (VideoBridge): needs both clips.
        assert!(video_request_candle_vace_eligible(
            "wan_2_2_i2v_14b",
            &object(json!({ "sourceClipAssetId": "l", "bridgeRightClipAssetId": "r" })),
            &JobType::VideoBridge
        ));
    }

    #[test]
    fn candle_vace_modes_fall_back_without_assets_or_for_unsupported_models() {
        // Missing required assets → torch.
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "clip_1" })), // no personTrackId / characterId
            &JobType::PersonReplace
        ));
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "l" })), // bridge needs the right clip too
            &JobType::VideoBridge
        ));
        // SCAIL-2 is a DISTINCT candle engine, not a VACE model → the VACE gate rejects it (the SCAIL-2
        // candle replace path is `scail2_replace_candle_eligible`, sc-6837).
        assert!(!video_request_candle_vace_eligible(
            "scail2_14b",
            &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
            &JobType::PersonReplace
        ));
        // A LoRA shape → torch (the candle VACE provider advertises no adapters).
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({
                "sourceClipAssetId": "c",
                "personTrackId": "t",
                "characterId": "ch",
                "loras": [{ "name": "x" }]
            })),
            &JobType::PersonReplace
        ));
        // A non-VACE job type is never VACE-eligible (the base txt2video gate handles VideoGenerate).
        assert!(!video_request_candle_vace_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
            &JobType::VideoGenerate
        ));
    }

    // ---- Candle SCAIL-2 character animation + replace_person (sc-6837, epic 6563) ----

    /// A queued `person_replace` job carrying `payload` (the PersonReplace job type the API stamps for
    /// the integrated replace_person pipeline).
    fn person_replace_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_pr",
            "type": "person_replace",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-20T00:00:00Z",
            "updatedAt": "2026-06-20T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    #[test]
    fn scail2_candle_serves_animation_and_replace_in_native_shape() {
        // Standalone character animation: scail2_14b + animate_character + a reference + a driving clip.
        // The reference can be referenceAssetIds, a bare referenceAssetId, or the i2v sourceAssetId.
        for reference in [
            json!({ "referenceAssetIds": ["ref_1"] }),
            json!({ "referenceAssetId": "ref_1" }),
            json!({ "sourceAssetId": "img_1" }),
        ] {
            let mut payload = object(reference);
            payload.insert("mode".into(), json!("animate_character"));
            payload.insert("sourceClipAssetId".into(), json!("clip_1"));
            assert!(
                scail2_animate_candle_eligible("scail2_14b", &payload),
                "scail2 animate_character must be candle-eligible: {payload:?}"
            );
        }
        // An animate job carrying an inference LoRA (DPO / lightning / user adapter) stays on candle —
        // the provider merges it into the dense DiT (sc-6838); only on-the-fly quant defers to torch.
        assert!(
            scail2_animate_candle_eligible(
                "scail2_14b",
                &object(json!({
                    "mode": "animate_character",
                    "referenceAssetIds": ["ref_1"],
                    "sourceClipAssetId": "clip_1",
                    "loras": [{ "name": "scail2-dpo" }]
                }))
            ),
            "scail2 animate with a LoRA must stay candle-eligible (sc-6838)"
        );
        // Cross-identity replacement: scail2_14b PersonReplace with the clip + track + character.
        assert!(scail2_replace_candle_eligible(
            "scail2_14b",
            &object(json!({
                "sourceClipAssetId": "clip_1",
                "personTrackId": "track_1",
                "characterId": "char_1"
            }))
        ));
        // Through the full video claim gate: animate_character (VideoGenerate) + replace (PersonReplace).
        assert!(video_job_is_candle_eligible(&video_generate_job(json!({
            "model": "scail2_14b",
            "mode": "animate_character",
            "referenceAssetIds": ["ref_1"],
            "sourceClipAssetId": "clip_1"
        }))));
        assert!(video_job_is_candle_eligible(&person_replace_job(json!({
            "model": "scail2_14b",
            "sourceClipAssetId": "clip_1",
            "personTrackId": "track_1",
            "characterId": "char_1"
        }))));
    }

    #[test]
    fn scail2_candle_rejects_incomplete_or_wrong_shape() {
        // animate_character needs BOTH a reference image and a driving clip.
        assert!(!scail2_animate_candle_eligible(
            "scail2_14b",
            &object(json!({ "mode": "animate_character", "referenceAssetIds": ["ref_1"] }))
        ));
        assert!(!scail2_animate_candle_eligible(
            "scail2_14b",
            &object(json!({ "mode": "animate_character", "sourceClipAssetId": "clip_1" }))
        ));
        // Wrong mode / wrong model never claim the SCAIL-2 candle lane.
        assert!(!scail2_animate_candle_eligible(
            "scail2_14b",
            &object(json!({
                "mode": "text_to_video",
                "sourceAssetId": "i",
                "sourceClipAssetId": "c"
            }))
        ));
        assert!(!scail2_animate_candle_eligible(
            "wan_2_2",
            &object(json!({
                "mode": "animate_character",
                "sourceAssetId": "i",
                "sourceClipAssetId": "c"
            }))
        ));
        // On-the-fly quant still defers to torch (the candle SCAIL-2 provider is dense).
        {
            let mut payload = object(json!({
                "mode": "animate_character",
                "sourceAssetId": "i",
                "sourceClipAssetId": "c"
            }));
            payload.insert("advanced".into(), json!({ "mlxQuantize": 8 }));
            assert!(
                !scail2_animate_candle_eligible("scail2_14b", &payload),
                "scail2 animate with on-the-fly quant must defer to torch: {payload:?}"
            );
        }
        // replace_person needs the clip + track + character; missing any → torch.
        for case in [
            json!({ "sourceClipAssetId": "c", "personTrackId": "t" }),
            json!({ "sourceClipAssetId": "c", "characterId": "ch" }),
            json!({ "personTrackId": "t", "characterId": "ch" }),
        ] {
            assert!(
                !scail2_replace_candle_eligible("scail2_14b", &object(case.clone())),
                "incomplete scail2 replace must defer to torch: {case}"
            );
        }
        // A non-SCAIL-2 model never claims the SCAIL-2 replace lane (it routes via Wan-VACE instead).
        assert!(!scail2_replace_candle_eligible(
            "wan_2_2",
            &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" }))
        ));
    }

    #[test]
    fn candle_worker_claims_txt2video_but_refuses_other_video_shapes() {
        let candle = gpu_worker(CANDLE_VIDEO_CAPS);
        // Claims wan + ltx + the 14B T2V plain txt2video.
        for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b"] {
            assert!(
                worker_supports_job(
                    &candle,
                    &video_generate_job(json!({ "model": model, "mode": "text_to_video" }))
                ),
                "candle worker should claim {model} txt2video"
            );
        }
        // Claims the 14B I2V + SVD in their image→video shape (with a source image) (sc-5175 / sc-5493).
        for model in ["wan_2_2_i2v_14b", "svd"] {
            assert!(
                worker_supports_job(
                    &candle,
                    &video_generate_job(json!({
                        "model": model,
                        "mode": "image_to_video",
                        "sourceAssetId": "a"
                    }))
                ),
                "candle worker should claim {model} image_to_video"
            );
        }
        // Claims `ltx_2_3_eros` text_to_video (sc-5495 — the candle LTX engine serves the eros fine-tune
        // too). Refuses an unported model, a conditioned (i2v) shape on a txt2video model, an image→video
        // model (svd) in a txt2video shape, and the 14B I2V in a txt2video shape (both image→video only).
        assert!(worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "ltx_2_3_eros", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "some_unported_model", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "svd", "mode": "text_to_video" }))
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(
                json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
            )
        ));
        assert!(!worker_supports_job(
            &candle,
            &video_generate_job(json!({ "model": "wan_2_2_i2v_14b", "mode": "text_to_video" }))
        ));
        // The co-resident torch worker claims everything the candle worker defers.
        let torch = gpu_worker(TORCH_VIDEO_CAPS);
        assert!(worker_supports_job(
            &torch,
            &video_generate_job(
                json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
            )
        ));
    }

    // ---- SeedVR2 video upscale (epic 4811 / sc-4816) ----

    /// A queued `video_upscale` job carrying `payload`.
    fn video_upscale_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_vu",
            "type": "video_upscale",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// An idle MLX (`gpu_id = "mlx"`) worker advertising `capabilities`.
    fn mlx_worker(capabilities: &[&str]) -> WorkerSnapshot {
        serde_json::from_value(json!({
            "id": "worker_mlx",
            "gpuId": "mlx",
            "status": "idle",
            "capabilities": capabilities,
            "loadedModels": [],
            "registeredAt": "2026-06-12T00:00:00Z",
            "lastSeenAt": "2026-06-12T00:00:00Z",
        }))
        .expect("valid WorkerSnapshot")
    }

    #[test]
    fn video_upscale_seedvr2_is_mlx_eligible_other_engines_are_not() {
        // seedvr2 (alias + 3b id) and the absent-engine default are eligible.
        for engine in [json!("seedvr2"), json!("seedvr2_3b"), Value::Null] {
            let payload = if engine.is_null() {
                json!({ "sourceAssetId": "a" })
            } else {
                json!({ "sourceAssetId": "a", "engine": engine })
            };
            assert!(
                video_upscale_job_is_mlx_eligible(&video_upscale_job(payload.clone())),
                "video_upscale should be MLX-eligible for {payload}"
            );
        }
        // An unknown engine is not eligible (no torch video upscaler exists).
        assert!(!video_upscale_job_is_mlx_eligible(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "aura-sr" })
        )));
        // The predicate is gated to the job type.
        assert!(!video_upscale_job_is_mlx_eligible(&video_generate_job(
            json!({ "model": "wan_2_2" })
        )));
    }

    #[test]
    fn mlx_worker_claims_seedvr2_video_upscale_and_refuses_other_engines() {
        let mlx = mlx_worker(&["gpu", "video_upscale"]);
        assert!(worker_supports_job(
            &mlx,
            &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
        ));
        // A non-SeedVR2 engine is refused by the mlx worker (mac-only; nowhere else to run).
        assert!(!worker_supports_job(
            &mlx,
            &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
        ));
    }

    #[test]
    fn video_upscale_requires_gpu() {
        assert!(job_requires_gpu(&JobType::VideoUpscale));
    }

    #[test]
    fn mac_capabilities_advertises_video_upscale() {
        let caps = mac_capabilities("darwin", true);
        let feature = caps
            .features
            .get("videoUpscale")
            .expect("videoUpscale feature present");
        assert!(feature.supported);
        assert!(feature.reason.is_none());
    }

    #[test]
    fn mac_rust_supports_seedvr2_video_upscale_only() {
        assert!(mac_rust_supported(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "seedvr2" })
        ))
        .is_ok());
        assert!(mac_rust_supported(&video_upscale_job(
            json!({ "sourceAssetId": "a", "engine": "aura-sr" })
        ))
        .is_err());
    }

    // ---- Candle SeedVR2 upscale lane (sc-5928) ----

    /// A queued `image_upscale` job carrying `payload`.
    fn image_upscale_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_iu",
            "type": "image_upscale",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-16T00:00:00Z",
            "updatedAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// sc-5499 + sc-5928: the candle worker claims both off-Mac image upscalers — Real-ESRGAN
    /// (`ort`/CUDA, sc-5499, incl. the default engine) and SeedVR2 (`candle-gen-seedvr2`, sc-5928).
    /// Only `aura-sr` (an offered engine dropped on every platform, sc-3668 / sc-5499) has no candle
    /// path → refused (it runs only on the Python torch worker until Phase 7).
    #[test]
    fn candle_worker_claims_real_esrgan_and_seedvr2_image_upscale_refuses_aura_sr() {
        let candle = gpu_worker(&["gpu", "image_upscale", "candle"]);
        assert!(worker_supports_job(
            &candle,
            &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
        ));
        // Real-ESRGAN (incl. the default engine) now has a candle path (the off-Mac ort/CUDA upscaler).
        assert!(worker_supports_job(
            &candle,
            &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "real-esrgan" }))
        ));
        assert!(worker_supports_job(
            &candle,
            &image_upscale_job(json!({ "sourceAssetId": "a" })) // default = real-esrgan
        ));
        // AuraSR is dropped as an offered engine → no candle path → refused.
        assert!(!worker_supports_job(
            &candle,
            &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
        ));
    }

    /// sc-5928: the candle worker claims the net-new SeedVR2 `video_upscale` (default/seedvr2 ids) and
    /// refuses other engines, exactly like the mlx worker (the engine set is shared).
    #[test]
    fn candle_worker_claims_seedvr2_video_upscale_and_refuses_other_engines() {
        let candle = gpu_worker(&["gpu", "video_upscale", "candle"]);
        for engine in [json!("seedvr2"), json!("seedvr2_3b"), Value::Null] {
            let payload = if engine.is_null() {
                json!({ "sourceAssetId": "a" })
            } else {
                json!({ "sourceAssetId": "a", "engine": engine })
            };
            assert!(
                worker_supports_job(&candle, &video_upscale_job(payload.clone())),
                "candle should claim video_upscale for {payload}"
            );
        }
        assert!(!worker_supports_job(
            &candle,
            &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
        ));
    }

    /// sc-5928: SeedVR2 has no torch backend, so a plain torch GPU worker (neither `mlx` nor candle)
    /// REFUSES a `seedvr2` image upscale — it stays queued for the mlx/candle worker instead of being
    /// claimed and failing. Real-ESRGAN (the torch engine) is still claimed. The inverse of AuraSR.
    #[test]
    fn torch_worker_refuses_seedvr2_image_upscale_but_claims_real_esrgan() {
        let torch = gpu_worker(&["gpu", "image_upscale"]); // no candle marker, gpu_id != "mlx"
        assert!(!worker_supports_job(
            &torch,
            &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
        ));
        assert!(worker_supports_job(
            &torch,
            &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "real-esrgan" }))
        ));
    }

    // ---- Candle kps_extract lane (sc-5497, epic 5482) ----

    /// A queued `kps_extract` job carrying `payload`.
    fn kps_extract_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_kps",
            "type": "kps_extract",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-16T00:00:00Z",
            "updatedAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// sc-5497: the candle worker advertises `kps_extract` (the candle SCRFD/ArcFace face stack) and
    /// claims a kps_extract job — the off-Mac sibling of the native-MLX path. UNLIKE SeedVR2, the Python
    /// InsightFace path CAN serve kps_extract, so there is NO torch-refusal gate: a co-resident torch
    /// worker that advertises the capability still claims it (the candle worker just runs it Python-free
    /// when it polls first; the Python path is retired wholesale in Phase 7, epic 5483). A worker that
    /// never advertises the capability (e.g. a candle-disabled box) refuses it.
    #[test]
    fn candle_worker_claims_kps_extract_no_torch_refusal() {
        let payload = json!({ "sourceAssetId": "a", "projectId": "p" });
        let candle = gpu_worker(&["gpu", "kps_extract", "candle"]);
        assert!(
            worker_supports_job(&candle, &kps_extract_job(payload.clone())),
            "candle worker should claim kps_extract"
        );
        let torch = gpu_worker(&["gpu", "kps_extract"]);
        assert!(
            worker_supports_job(&torch, &kps_extract_job(payload.clone())),
            "torch worker still claims kps_extract (no refusal — it has the InsightFace path)"
        );
        let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
        assert!(
            !worker_supports_job(&no_cap, &kps_extract_job(payload)),
            "a worker not advertising kps_extract refuses it"
        );
    }

    // ---- Candle pose_detect (DWPose) lane (sc-5496, epic 5482) ----

    /// A queued `pose_detect` job carrying `payload`.
    fn pose_detect_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_pose",
            "type": "pose_detect",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-16T00:00:00Z",
            "updatedAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// sc-5496: the candle worker advertises `pose_detect` (the DWPose RTMW detector via the `ort` CUDA
    /// EP) and claims a pose_detect job — the off-Mac sibling of the macOS `ort`/CoreML path. Like
    /// kps_extract (and unlike SeedVR2), the Python rtmlib path CAN serve pose_detect, so there is NO
    /// torch-refusal gate: a co-resident torch worker that advertises the capability still claims it (the
    /// candle worker just runs it Python-free when it polls first; the Python path is retired wholesale in
    /// Phase 7, epic 5483). A worker that never advertises the capability (e.g. a candle-disabled box)
    /// refuses it.
    #[test]
    fn candle_worker_claims_pose_detect_no_torch_refusal() {
        let payload = json!({ "sources": [{ "assetId": "a" }], "projectId": "p" });
        let candle = gpu_worker(&["gpu", "pose_detect", "candle"]);
        assert!(
            worker_supports_job(&candle, &pose_detect_job(payload.clone())),
            "candle worker should claim pose_detect"
        );
        let torch = gpu_worker(&["gpu", "pose_detect"]);
        assert!(
            worker_supports_job(&torch, &pose_detect_job(payload.clone())),
            "torch worker still claims pose_detect (no refusal — it has the rtmlib path)"
        );
        let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
        assert!(
            !worker_supports_job(&no_cap, &pose_detect_job(payload)),
            "a worker not advertising pose_detect refuses it"
        );
    }

    // ---- Candle person detect/track lane (sc-5498) ----

    /// A queued real (non-preview) `person_detect` job carrying `payload`.
    fn person_detect_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_person_detect",
            "type": "person_detect",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-16T00:00:00Z",
            "updatedAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// A queued real (non-preview) `person_track` job carrying `payload`.
    fn person_track_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_person_track",
            "type": "person_track",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-16T00:00:00Z",
            "updatedAt": "2026-06-16T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    /// sc-5498: the candle worker advertises `person_detect` + `person_track` (YOLO11 via the `ort`
    /// CUDA EP + the pure-Rust ByteTrack) and claims both — the off-Mac sibling of the macOS
    /// native-MLX path (sc-3633/sc-3634). Like kps_extract / pose_detect (and unlike SeedVR2), the
    /// Python Ultralytics path CAN serve them, so there is NO torch-refusal gate: a co-resident
    /// torch worker that advertises the capability still claims it (the candle worker just runs it
    /// Python-free when it polls first; the Python path is retired wholesale in Phase 7, epic 5483).
    /// A worker that never advertises the capability refuses the job. (These are the real,
    /// non-preview jobs; the procedural `preview: true` path keys off the separate
    /// `person_detect_preview` / `person_track_preview` capabilities.)
    #[test]
    fn candle_worker_claims_person_detect_and_track_no_torch_refusal() {
        let payload = json!({ "projectId": "p", "sourceAssetId": "a" });
        let candle = gpu_worker(&["gpu", "person_detect", "person_track", "candle"]);
        assert!(
            worker_supports_job(&candle, &person_detect_job(payload.clone())),
            "candle worker should claim person_detect"
        );
        assert!(
            worker_supports_job(&candle, &person_track_job(payload.clone())),
            "candle worker should claim person_track"
        );
        let torch = gpu_worker(&["gpu", "person_detect", "person_track"]);
        assert!(
            worker_supports_job(&torch, &person_detect_job(payload.clone())),
            "torch worker still claims person_detect (no refusal — it has the Ultralytics path)"
        );
        assert!(
            worker_supports_job(&torch, &person_track_job(payload.clone())),
            "torch worker still claims person_track (no refusal — it has the Ultralytics path)"
        );
        let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
        assert!(
            !worker_supports_job(&no_cap, &person_detect_job(payload.clone())),
            "a worker not advertising person_detect refuses it"
        );
        assert!(
            !worker_supports_job(&no_cap, &person_track_job(payload)),
            "a worker not advertising person_track refuses it"
        );
    }

    // ---- Candle caption lane (sc-5098) ----

    /// A queued `training_caption` job carrying `payload`.
    fn caption_job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_c",
            "type": "training_caption",
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-13T00:00:00Z",
            "updatedAt": "2026-06-13T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    }

    #[test]
    fn candle_worker_claims_joycaption_but_refuses_other_captioners() {
        let candle = gpu_worker(&["gpu", "training_caption", "candle"]);
        // Claims a JoyCaption job.
        assert!(worker_supports_job(
            &candle,
            &caption_job(json!({ "captioner": "joy_caption", "datasetId": "ds_1" }))
        ));
        // Refuses a non-JoyCaption captioner → falls back to the Python torch worker.
        assert!(!worker_supports_job(
            &candle,
            &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
        ));
        let torch = gpu_worker(&["gpu", "training_caption"]);
        assert!(worker_supports_job(
            &torch,
            &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
        ));
    }

    /// sc-5501: the candle worker claims SenseNova-U1 `image_vqa` / `image_interleave` (served off-Mac
    /// by the concrete candle `T2iModel::{vqa, interleave_gen}`) but refuses other models, which stay
    /// on the Python torch worker.
    #[test]
    fn candle_worker_claims_sensenova_understanding_but_refuses_other_models() {
        let candle = gpu_worker(&["gpu", "image_vqa", "image_interleave", "candle"]);
        let understanding_job = |job_type: &str, payload: Value| -> JobSnapshot {
            serde_json::from_value(json!({
                "id": "job_u",
                "type": job_type,
                "status": "queued",
                "payload": payload,
                "result": {},
                "requestedGpu": "auto",
                "progress": 0,
                "stage": "queued",
                "message": "",
                "attempts": 1,
                "cancelRequested": false,
                "createdAt": "2026-06-14T00:00:00Z",
                "updatedAt": "2026-06-14T00:00:00Z",
            }))
            .expect("valid JobSnapshot")
        };
        // Claims SenseNova-U1 VQA + interleave (base + `_fast` ids).
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_vqa",
                json!({ "model": "sensenova_u1_8b", "question": "what is this?", "sourceAssetId": "a1" })
            )
        ));
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_interleave",
                json!({ "model": "sensenova_u1_8b_fast", "prompt": "a short illustrated story" })
            )
        ));
        // Infographic-V2 base advertises the SAME understanding surface (epic 9959): the eligibility
        // list must include its id, else V2 VQA / Document-Studio jobs never route to the in-process
        // worker (regression guard for the sc-9963 fix).
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_vqa",
                json!({ "model": "sensenova_u1_8b_infographic_v2", "question": "what is this?", "sourceAssetId": "a1" })
            )
        ));
        assert!(worker_supports_job(
            &candle,
            &understanding_job(
                "image_interleave",
                json!({ "model": "sensenova_u1_8b_infographic_v2", "prompt": "an illustrated explainer" })
            )
        ));
        // Refuses a non-SenseNova understanding job → falls back to the Python torch worker.
        assert!(!worker_supports_job(
            &candle,
            &understanding_job(
                "image_vqa",
                json!({ "model": "some_other_vlm", "question": "?", "sourceAssetId": "a1" })
            )
        ));
    }

    #[test]
    fn instantid_character_jobs_route_to_candle_off_mac() {
        // The candle InstantID provider (sc-5491) serves the SAME surface as the MLX path off-Mac, so
        // every character_image + referenceAssetId shape is candle-eligible — via the bespoke
        // `image_job_is_candle_eligible` branch, NOT the txt2img-only `image_request_candle_eligible`
        // gate (which rejects `referenceAssetId`, which InstantID requires).
        for advanced in [
            json!({}),
            json!({ "angleSet": true }),
            json!({ "poses": [{ "id": "a" }] }),
            json!({ "faceRestore": true }),
            json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
        ] {
            let payload = json!({
                "model": "instantid_realvisxl",
                "mode": "character_image",
                "referenceAssetId": "asset_1",
                "advanced": advanced,
            });
            assert!(instantid_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }

        // No reference face → not candle-eligible (mirrors the MLX gate).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "instantid_realvisxl",
            "mode": "character_image"
        }))));
        // Non-character mode → not candle-eligible (InstantID is a character flow).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "instantid_realvisxl",
            "mode": "text_to_image",
            "referenceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn sdxl_ipadapter_reference_jobs_route_to_candle() {
        // A pure SDXL/RealVisXL reference (IP-Adapter) job routes to the candle lane (sc-5488) via the
        // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
        // `referenceAssetId`).
        for model in [
            "sdxl",
            "realvisxl",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ] {
            let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
            assert!(sdxl_ipadapter_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }
        // No reference → not an IP-Adapter job (plain txt2img routes via the txt2img gate instead).
        assert!(!sdxl_ipadapter_candle_eligible(&object(
            json!({ "model": "sdxl" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
            "model": "sdxl", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn sdxl_edit_jobs_route_to_candle() {
        // SDXL/RealVisXL img2img / inpaint / outpaint edit jobs (sc-5487) route to the bespoke candle
        // `SdxlEdit` lane via the new branch, NOT the txt2img `image_request_candle_eligible` gate
        // (which rejects the whole `edit_image` family).
        for model in [
            "sdxl",
            "realvisxl",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ] {
            // img2img (source, no mask).
            let img2img = json!({ "model": model, "mode": "edit_image", "sourceAssetId": "src_1" });
            assert!(sdxl_edit_candle_eligible(&object(img2img.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(img2img)));
            // inpaint (source + mask).
            let inpaint = json!({
                "model": model, "mode": "edit_image", "sourceAssetId": "src_1", "maskAssetId": "m_1"
            });
            assert!(sdxl_edit_candle_eligible(&object(inpaint.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(inpaint)));
            // outpaint (source + fitMode outpaint).
            let outpaint = json!({
                "model": model, "mode": "edit_image", "sourceAssetId": "src_1", "fitMode": "outpaint"
            });
            assert!(sdxl_edit_candle_eligible(&object(outpaint.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(outpaint)));
        }
        // `edit_image` WITHOUT a source → not this lane (nothing to edit).
        assert!(!sdxl_edit_candle_eligible(&object(json!({
            "model": "sdxl", "mode": "edit_image"
        }))));
        // A reference (IP-Adapter) job is NOT the edit lane (no source, not `edit_image`) — it's sc-5488.
        assert!(!sdxl_edit_candle_eligible(&object(json!({
            "model": "sdxl", "referenceAssetId": "a"
        }))));
        // A plain txt2img sdxl job → not the edit lane.
        assert!(!sdxl_edit_candle_eligible(&object(
            json!({ "model": "sdxl" })
        )));
    }

    #[test]
    fn zimage_edit_jobs_route_to_candle() {
        // Z-Image img2img / edit jobs (sc-6595) route to the bespoke candle `ZImageEdit` lane via the new
        // branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects `edit_image`). Both
        // the txt2img id in edit mode (`z_image_turbo`) and the dedicated `z_image_edit` id are served.
        for model in ["z_image_turbo", "z_image_edit"] {
            let edit = json!({ "model": model, "mode": "edit_image", "sourceAssetId": "src_1" });
            assert!(zimage_edit_candle_eligible(&object(edit.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(
                edit.clone()
            )));
            // Reached through the real `image_edit` job type too (the type the Image Editor submits).
            assert!(image_job_is_candle_eligible(&image_edit_job(edit)));
        }
        // `edit_image` WITHOUT a source → not this lane (nothing to edit).
        assert!(!zimage_edit_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "edit_image"
        }))));
        // A plain txt2img z_image_turbo job → not the edit lane (it routes via the txt2img gate instead).
        assert!(!zimage_edit_candle_eligible(&object(
            json!({ "model": "z_image_turbo" })
        )));
        // A z_image_turbo strict-pose job (advanced.poses, not edit_image) is the control lane, not edit.
        assert!(!zimage_edit_candle_eligible(&object(json!({
            "model": "z_image_turbo", "advanced": { "poses": [{}] }
        }))));
    }

    #[test]
    fn zimage_identity_with_character_jobs_route_to_candle() {
        // Z-Image identity-init "With Character" jobs (sc-8409): a `z_image_turbo` `character_image` job
        // with a `referenceAssetId` + `advanced.referenceStrength > 0` routes to the bespoke candle
        // `ZImageEdit` identity lane via the new branch, NOT the txt2img `image_request_candle_eligible`
        // gate (which rejects any `referenceAssetId`). Without this the off-Mac job fell through to plain
        // txt2img, dropping the reference (no identity, no score).
        let with_character = json!({
            "model": "z_image_turbo",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
            "advanced": { "referenceStrength": 0.6 }
        });
        assert!(zimage_identity_candle_eligible(&object(
            with_character.clone()
        )));
        assert!(image_job_is_candle_eligible(&image_generate_job(
            with_character
        )));
        // A numeric-string referenceStrength engages too (the web sends strings).
        assert!(zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image",
            "referenceAssetId": "asset_1", "advanced": { "referenceStrength": "0.45" }
        }))));

        // No referenceStrength (or <= 0) → stays plain txt2img on both backends (parity), NOT this lane.
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1"
        }))));
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image",
            "referenceAssetId": "asset_1", "advanced": { "referenceStrength": 0.0 }
        }))));
        // No reference face → no identity source → not this lane.
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image",
            "advanced": { "referenceStrength": 0.6 }
        }))));
        // Non-character mode → not this lane (an `edit_image` job is the edit lane, sc-6595).
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "edit_image",
            "referenceAssetId": "asset_1", "advanced": { "referenceStrength": 0.6 }
        }))));
        // Angle set + pose set are `character_image` too but route to their own lanes — excluded here so
        // this plain With-Character gate never steals them.
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1",
            "advanced": { "referenceStrength": 0.6, "angleSet": true }
        }))));
        assert!(!zimage_identity_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1",
            "advanced": { "referenceStrength": 0.6, "poses": [{ "id": "a" }] }
        }))));
    }

    #[test]
    fn image_edit_job_type_routes_through_candle_edit_lane() {
        // Regression for the sc-5487 edit lanes being unreachable through the actual `image_edit` job
        // type the Image Editor submits (the prior tests only exercised `image_generate` jobs with
        // `mode == "edit_image"`, so the `JobType::ImageEdit`-only gap was invisible). A plain SDXL edit
        // submitted as `image_edit` must: be candle-eligible, survive the `candle_required` enforce sweep
        // (`candle_supported` → Ok), and be claimed by the candle worker — NOT enforce-failed
        // `candle_unsupported`.
        let sdxl_edit = json!({
            "model": "sdxl",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        });
        assert!(
            image_job_is_candle_eligible(&image_edit_job(sdxl_edit.clone())),
            "an `image_edit`-typed SDXL edit must reach the candle SdxlEdit lane"
        );
        assert!(
            candle_supported(&image_edit_job(sdxl_edit.clone())).is_ok(),
            "an eligible `image_edit` SDXL job must not be enforce-failed candle_unsupported"
        );
        assert!(
            worker_supports_job(&gpu_worker(CANDLE_CAPS), &image_edit_job(sdxl_edit.clone())),
            "the candle worker (advertising `image_edit`) must claim the SDXL edit job"
        );
        // The FLUX.2-klein, Qwen-Image-Edit, and Z-Image edit lanes are reached through the same job type.
        for model in [
            "flux2_klein_9b",
            "qwen_image_edit",
            "qwen_image_edit_2511_lightning",
            "z_image_turbo",
            "z_image_edit",
        ] {
            let job = image_edit_job(json!({
                "model": model, "mode": "edit_image", "sourceAssetId": "asset_1"
            }));
            assert!(
                image_job_is_candle_eligible(&job) && candle_supported(&job).is_ok(),
                "{model} edit via the `image_edit` job type must reach its candle lane"
            );
        }
        // A torch-only edit family (`kolors` has a candle txt2img lane but no candle EDIT lane) submitted
        // as `image_edit` is NOT candle-eligible: the candle worker must refuse it so it falls back to the
        // co-resident torch worker, which claims it. (Mirrors the `image_generate` + edit_image case.)
        let kolors_edit = json!({
            "model": "kolors",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        });
        assert!(!image_job_is_candle_eligible(&image_edit_job(
            kolors_edit.clone()
        )));
        assert!(!worker_supports_job(
            &gpu_worker(CANDLE_CAPS),
            &image_edit_job(kolors_edit.clone())
        ));
        assert!(
            worker_supports_job(&gpu_worker(TORCH_CAPS), &image_edit_job(kolors_edit)),
            "a torch-only edit model must still be claimable by the co-resident torch worker"
        );
        // An `image_edit` job with no source image is not the edit lane → not candle-eligible.
        assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
            "model": "sdxl", "mode": "edit_image"
        }))));
    }

    #[test]
    fn kolors_ipadapter_reference_jobs_route_to_candle() {
        // A pure Kolors reference (IP-Adapter) job routes to the candle lane (sc-5488) via the bespoke
        // branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects `referenceAssetId`).
        let payload = json!({ "model": "kolors", "referenceAssetId": "asset_1" });
        assert!(kolors_ipadapter_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No reference → plain txt2img routes via the txt2img gate instead.
        assert!(!kolors_ipadapter_candle_eligible(&object(
            json!({ "model": "kolors" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!kolors_ipadapter_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn flux_ipadapter_reference_jobs_route_to_candle() {
        // A pure FLUX reference (XLabs IP-Adapter) job routes to the candle lane (sc-5872) via the
        // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
        // `referenceAssetId`). Both variants.
        for model in ["flux_dev", "flux_schnell"] {
            let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
            assert!(flux_ipadapter_candle_eligible(&object(payload.clone())));
            assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        }
        // No reference → plain txt2img routes via the txt2img gate instead.
        assert!(!flux_ipadapter_candle_eligible(&object(
            json!({ "model": "flux_dev" })
        )));
        // img2img / inpaint / edit shapes are NOT this lane (those are sc-5487, still torch).
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_dev", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_dev", "referenceAssetId": "a", "sourceAssetId": "s"
        }))));
        assert!(!flux_ipadapter_candle_eligible(&object(json!({
            "model": "flux_schnell", "referenceAssetId": "a", "maskAssetId": "m"
        }))));
    }

    #[test]
    fn pulid_flux_character_jobs_route_to_candle_off_mac() {
        // The candle PuLID-FLUX provider (sc-5492) serves the SAME surface as the MLX path off-Mac, so
        // a `pulid_flux_dev` character_image + referenceAssetId job is candle-eligible via the bespoke
        // `image_job_is_candle_eligible` branch, NOT the txt2img-only `image_request_candle_eligible`
        // gate (which rejects `referenceAssetId`, which PuLID requires). The distinct `pulid_flux_dev`
        // model id cleanly disambiguates it from the FLUX XLabs IP-Adapter lane (`flux_dev`).
        let payload = json!({
            "model": "pulid_flux_dev",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
        });
        assert!(pulid_flux_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));

        // No reference face → not candle-eligible (mirrors the MLX gate).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "pulid_flux_dev",
            "mode": "character_image"
        }))));
        // Non-character mode → not candle-eligible (PuLID is a character flow).
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "pulid_flux_dev",
            "mode": "text_to_image",
            "referenceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn qwen_control_pose_jobs_route_to_candle() {
        // qwen_image + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
        // branch, NOT the txt2img gate (which DEFERS any advanced.poses job to torch).
        let payload =
            json!({ "model": "qwen_image", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(qwen_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!qwen_control_candle_eligible(&object(
            json!({ "model": "qwen_image" })
        )));
        assert!(!qwen_control_candle_eligible(&object(json!({
            "model": "qwen_image", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!qwen_control_candle_eligible(&object(json!({
            "model": "qwen_image", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
        // Plain `sdxl` + poses is NOT candle-*served* (no plain-SDXL pose lane — SDXL pose ships via
        // InstantID): the qwen branch is specific and the txt2img gate's has_poses check rejects it, so
        // `image_job_is_candle_eligible` is false. (It is, however, candle-*owned-to-reject* at the
        // worker layer per sc-5968 — see `unsupported_pose_is_owned_by_candle_*`; that claim lives in
        // `worker_supports_job`, not here. z_image_turbo + poses IS a candle lane — `zimage_control_*`.)
        assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
            "model": "sdxl", "advanced": { "poses": [{}] }
        }))));
    }

    #[test]
    fn kolors_control_pose_jobs_route_to_candle() {
        // kolors + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
        // branch, NOT the txt2img gate (which DEFERS any advanced.poses job to torch).
        let payload = json!({ "model": "kolors", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(kolors_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!kolors_control_candle_eligible(&object(
            json!({ "model": "kolors" })
        )));
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
        // A kolors reference job (no poses) still routes via the IP-Adapter branch, not this one.
        assert!(!kolors_control_candle_eligible(&object(json!({
            "model": "kolors", "referenceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn zimage_control_pose_jobs_route_to_candle() {
        // z_image_turbo + advanced.poses routes to the candle VACE strict-pose lane (sc-5489, the last
        // family) via the bespoke branch, NOT the txt2img gate (which DEFERS any advanced.poses to torch).
        let payload =
            json!({ "model": "z_image_turbo", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(zimage_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
        assert!(!zimage_control_candle_eligible(&object(
            json!({ "model": "z_image_turbo" })
        )));
        assert!(!zimage_control_candle_eligible(&object(json!({
            "model": "z_image_turbo", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!zimage_control_candle_eligible(&object(json!({
            "model": "z_image_turbo", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
    }

    #[test]
    fn zimage_base_control_pose_jobs_route_to_candle() {
        // sc-8379: the BASE z_image model + advanced.poses routes to the same candle strict-control lane
        // as Turbo (the base Fun-Controlnet-Union branch) via the bespoke branch, NOT the txt2img gate.
        let payload = json!({ "model": "z_image", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(zimage_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // A base z_image with no poses is plain txt2img — now a candle lane too (sc-8679: the registered
        // candle `z_image` base generator), so it routes to the generic candle txt2img gate rather than
        // deferring to torch. It is NOT this strict-control lane, though.
        let plain = json!({ "model": "z_image", "prompt": "a misty fjord" });
        assert!(!zimage_control_candle_eligible(&object(plain.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(plain)));
        // Both Turbo and base have a candle pose lane (so neither is pose-rejected).
        assert!(model_has_candle_pose_lane("z_image"));
        assert!(model_has_candle_pose_lane("z_image_turbo"));
    }

    #[test]
    fn flux1_dev_control_pose_jobs_route_to_candle() {
        // sc-8412: flux_dev + advanced.poses routes to the candle Shakker Union-Pro-2.0 strict-control
        // lane via the bespoke branch, NOT the txt2img gate (which DEFERS any advanced.poses to torch).
        let payload =
            json!({ "model": "flux_dev", "advanced": { "poses": [{ "keypoints": [] }] } });
        assert!(flux1_control_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
        // No poses → plain txt2img routes via the txt2img gate instead.
        assert!(!flux1_control_candle_eligible(&object(
            json!({ "model": "flux_dev" })
        )));
        assert!(!flux1_control_candle_eligible(&object(json!({
            "model": "flux_dev", "advanced": { "poses": [] }
        }))));
        // edit_image with poses is NOT this lane.
        assert!(!flux1_control_candle_eligible(&object(json!({
            "model": "flux_dev", "mode": "edit_image", "advanced": { "poses": [{}] }
        }))));
        // A flux_dev reference job (no poses) routes via the FLUX XLabs IP-Adapter branch, not this one.
        assert!(!flux1_control_candle_eligible(&object(json!({
            "model": "flux_dev", "referenceAssetId": "asset_1"
        }))));
        // flux_dev now HAS a candle pose lane (so it is not pose-rejected); schnell does not.
        assert!(model_has_candle_pose_lane("flux_dev"));
        assert!(!model_has_candle_pose_lane("flux_schnell"));
    }
}

#[cfg(test)]
mod mlx_routing_tests {
    use super::{
        flux2_mlx_eligible, flux_mlx_eligible, image_request_mlx_eligible, instantid_mlx_eligible,
        model_mac_support, qwen_edit_mlx_eligible, qwen_mlx_eligible, sdxl_mlx_eligible,
        video_mode_is_mlx_eligible, z_image_mlx_eligible, VIDEO_MLX_ROUTED_MODELS,
    };
    use serde_json::{json, Map, Value};

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().expect("test value is an object").clone()
    }

    #[test]
    fn z_image_plain_txt2img_is_eligible() {
        assert!(z_image_mlx_eligible(&object(
            json!({ "prompt": "a misty fjord" })
        )));
        assert!(z_image_mlx_eligible(&Map::new()));
    }

    #[test]
    fn z_image_edit_mode_with_source_is_eligible() {
        // epic 3529: img2img-edit (sourceAssetId) now routes to MLX via the engine's
        // `Conditioning::Reference` img2img path.
        assert!(z_image_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))));
    }

    #[test]
    fn z_image_edit_mode_without_source_is_not_eligible() {
        // An edit with nothing to edit (no/blank sourceAssetId) stays off MLX.
        assert!(!z_image_mlx_eligible(&object(
            json!({ "mode": "edit_image" })
        )));
        assert!(!z_image_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "sourceAssetId": "   "
        }))));
    }

    #[test]
    fn z_image_reference_without_poses_is_eligible() {
        // sc-3619: reference-identity img2img-init (no pose) now routes to MLX — the
        // base engine already supports the plain img2img path, and torch dropped the
        // reference entirely (it was a no-op on the fallback).
        assert!(z_image_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // Empty/whitespace reference id is treated as absent → plain txt2img, eligible.
        assert!(z_image_mlx_eligible(&object(
            json!({ "referenceAssetId": "   " })
        )));
        // A reference with empty poses is still reference-only → eligible (not the
        // pose tier, which needs a non-empty pose set).
        assert!(z_image_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [] }
        }))));
    }

    #[test]
    fn z_image_reference_with_poses_stays_on_mlx() {
        // The strict pose ControlNet tier lives only on MLX, so a reference+pose
        // job must route to the mlx worker, not torch (which would drop the poses).
        assert!(z_image_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }))));
    }

    #[test]
    fn z_image_peft_lokr_and_thirdparty_lycoris_both_route_mlx() {
        // SceneWorks peft LoKr applies natively on the MLX Z-Image path → eligible.
        assert!(z_image_mlx_eligible(&object(json!({
            "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
        }))));
        // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX too.
        assert!(z_image_mlx_eligible(&object(json!({
            "loras": [{ "path": "b.safetensors", "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn flux_plain_txt2img_is_eligible() {
        assert!(flux_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        assert!(flux_mlx_eligible(&Map::new()));
        // A LoRA is fine on the MLX flux path (engine applies LoRA + peft LoKr).
        assert!(flux_mlx_eligible(&object(json!({
            "loras": [{ "path": "a.safetensors", "networkType": "lora" }]
        }))));
    }

    #[test]
    fn flux_reference_is_eligible() {
        // Reference (XLabs IP-Adapter, epic 3621) now routes to MLX on both variants —
        // the Rust engine has no diffusers schnell limitation.
        assert!(flux_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // A reference + LoRA is still fine.
        assert!(flux_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "loras": [{ "networkType": "lora" }]
        }))));
    }

    #[test]
    fn flux_only_edit_falls_back_lycoris_routes_mlx() {
        // edit_image (no FLUX.1 edit on any platform — future Kontext) is the only fall-back.
        assert!(!flux_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX.
        assert!(flux_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        // Reference + a LyCORIS LoRA also routes MLX now.
        assert!(flux_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn qwen_plain_txt2img_is_eligible() {
        assert!(qwen_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        // A negative prompt + LoRA are fine on the MLX qwen path (true CFG + LoRA wired).
        assert!(qwen_mlx_eligible(&object(json!({
            "negativePrompt": "blurry",
            "loras": [{ "networkType": "lokr" }]
        }))));
    }

    #[test]
    fn qwen_edit_reference_falls_back_but_pose_and_lycoris_route_mlx() {
        assert!(!qwen_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(!qwen_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        // Strict pose ControlNet (sc-2291 / sc-3575) routes to MLX, even if a reference is
        // present; the strict-pose tier is pose-from-prompt and ignores the reference.
        assert!(qwen_mlx_eligible(&object(json!({
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        assert!(qwen_mlx_eligible(&object(json!({
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        // Third-party LyCORIS on a plain txt2img qwen job now routes MLX (epic 3641).
        assert!(qwen_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn qwen_edit_routes_edit_and_reference_flows_to_mlx() {
        // sc-3397: the qwen_image_edit ids run the engine's `qwen_image_edit` model.
        // edit_image with a source → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1"
        }))));
        // character_image with a reference (subject variation) → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1"
        }))));
        // character_image + reference + best-effort poses → still eligible. Unlike the base
        // Qwen strict-pose ControlNet (torch until epic 3401), the edit best-effort pose tier
        // is native multi-image ([reference, skeleton]) → MLX.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1",
            "advanced": { "poses": [{ "id": "p1" }] }
        }))));
        // character_image + reference + angle set → eligible.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "character_image", "referenceAssetId": "ref_1",
            "advanced": { "angleSet": true }
        }))));
        // A peft LoKr is fine on the MLX edit path.
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1",
            "loras": [{ "networkType": "lokr" }]
        }))));
    }

    #[test]
    fn qwen_edit_without_reference_falls_back_to_torch() {
        // edit_image with nothing to edit (no source, no reference) → torch.
        assert!(!qwen_edit_mlx_eligible(&object(
            json!({ "mode": "edit_image" })
        )));
        // character_image without a reference → torch (the edit model needs a reference).
        assert!(!qwen_edit_mlx_eligible(&object(
            json!({ "mode": "character_image" })
        )));
        // A plain txt2img mode is not an edit job (that's the base qwen_image MLX path).
        assert!(!qwen_edit_mlx_eligible(&object(json!({
            "mode": "text_to_image", "sourceAssetId": "src_1"
        }))));
        // Whitespace-only ids are treated as absent.
        assert!(!qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "   "
        }))));
        // A third-party LyCORIS LoRA on an otherwise-eligible edit job now routes MLX (epic 3641).
        assert!(qwen_edit_mlx_eligible(&object(json!({
            "mode": "edit_image", "sourceAssetId": "src_1",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn flux2_txt2img_edit_and_lycoris_all_route_mlx() {
        // FLUX.2 is MLX-only: txt2img (sc-3025), edit/reference (sc-3029), and — since epic 3641 —
        // third-party LyCORIS all route MLX.
        assert!(flux2_mlx_eligible(&object(
            json!({ "prompt": "a red fox" })
        )));
        assert!(flux2_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(flux2_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        assert!(flux2_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn sdxl_eligible_for_txt2img_edit_reference_lokr_and_lycoris() {
        assert!(sdxl_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
        // peft LoKr stays on MLX (the Rust SDXL path supports LoKr, unlike the old vendored path).
        assert!(sdxl_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lokr" }]
        }))));
        // sc-3060: the Rust engine now handles the advanced shapes, so edit_image
        // (img2img / inpaint / outpaint) and reference/IP-Adapter route to MLX too.
        assert!(sdxl_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
        assert!(sdxl_mlx_eligible(&object(
            json!({ "referenceAssetId": "asset_1" })
        )));
        assert!(sdxl_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "maskAssetId": "mask_1"
        }))));
        // Third-party LyCORIS now applies on the SDXL merge path (epic 3641, sc-3671) → MLX,
        // including on an edit job.
        assert!(sdxl_mlx_eligible(&object(json!({
            "loras": [{ "networkType": "lycoris" }]
        }))));
        assert!(sdxl_mlx_eligible(&object(json!({
            "mode": "edit_image",
            "loras": [{ "networkType": "lycoris" }]
        }))));
    }

    #[test]
    fn instantid_routes_all_character_modes_to_mlx() {
        // The full InstantID surface is native (sc-3345 identity + angle; sc-3381 pose + restore):
        // every character_image + referenceAssetId shape routes to MLX.
        for advanced in [
            json!({}),
            json!({ "angleSet": true }),
            json!({ "poses": [{ "id": "a" }] }),
            json!({ "faceRestore": true }),
            json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
        ] {
            let payload = object(json!({
                "model": "instantid_realvisxl",
                "mode": "character_image",
                "referenceAssetId": "asset_1",
                "advanced": advanced,
            }));
            assert!(instantid_mlx_eligible(&payload));
            assert!(image_request_mlx_eligible("instantid_realvisxl", &payload));
        }

        // No reference face → not eligible.
        assert!(!instantid_mlx_eligible(&object(json!({
            "model": "instantid_realvisxl",
            "mode": "character_image"
        }))));

        // Non-character mode → not eligible (InstantID is a character flow).
        assert!(!instantid_mlx_eligible(&object(json!({
            "model": "instantid_realvisxl",
            "mode": "text_to_image"
        }))));
    }

    #[test]
    fn ideogram_4_text_to_image_and_edit_route_to_mlx() {
        // sc-6302 + sc-6303: `ideogram_4` is in MLX_ROUTED_MODELS, and the native engine now serves
        // both text-to-image and img2img/mask-inpaint edit — both route to the in-process MLX worker.
        assert!(image_request_mlx_eligible(
            "ideogram_4",
            &object(json!({ "prompt": "a neon city skyline" }))
        ));
        assert!(image_request_mlx_eligible("ideogram_4", &Map::new()));
        // Edit (img2img / inpaint) now routes to MLX (sc-6303 — `resolve_ideogram_edit`).
        assert!(image_request_mlx_eligible(
            "ideogram_4",
            &object(json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }))
        ));

        // The UI gating oracle: Ideogram 4 is macSupport.supported (reaches the Text → Image picker)
        // and `features.edit` is now true (drives the Image Studio Edit tab alongside the catalog
        // `edit_image` capability). `reference`/`pose` remain true — inert, since the catalog
        // capabilities (not these flags) drive the UI affordances.
        let support = model_mac_support("ideogram_4", "image");
        assert!(support.supported, "ideogram_4 must be Mac-supported");
        assert!(
            support.features.edit,
            "ideogram_4 now supports edit (sc-6303)"
        );

        // Turbo is the same base model + the bundled TurboTime LoRA, so it routes + edits identically
        // (sc-6303). It was never registered in core before this (sc-6302 added only the base id), so
        // this also restores its Text → Image picker visibility.
        assert!(image_request_mlx_eligible("ideogram_4_turbo", &Map::new()));
        assert!(image_request_mlx_eligible(
            "ideogram_4_turbo",
            &object(json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }))
        ));
        let turbo = model_mac_support("ideogram_4_turbo", "image");
        assert!(turbo.supported, "ideogram_4_turbo must be Mac-supported");
        assert!(turbo.features.edit, "ideogram_4_turbo supports edit");
    }

    #[test]
    fn boogu_text_to_image_and_edit_route_to_mlx() {
        // sc-6399 (epic 6387): the three Boogu ids are in MLX_ROUTED_MODELS and route to the native
        // `mlx-gen-boogu` engine. Base + Turbo are text-to-image; Edit is the instruction image-edit.
        for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
            assert!(
                image_request_mlx_eligible(
                    model,
                    &object(json!({ "model": model, "prompt": "p" }))
                ),
                "{model} text-to-image must route to MLX"
            );
            assert!(
                image_request_mlx_eligible(model, &Map::new()),
                "{model} bare payload"
            );
        }

        // Edit routes to MLX for the Edit checkpoint only — Base/Turbo are text-to-image (their
        // semantic-edit path is incoherent without the Edit fine-tune, E7b-3).
        let edit_payload = |model: &str| {
            object(json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" }))
        };
        assert!(image_request_mlx_eligible(
            "boogu_image_edit",
            &edit_payload("boogu_image_edit")
        ));
        assert!(!image_request_mlx_eligible(
            "boogu_image",
            &edit_payload("boogu_image")
        ));
        assert!(!image_request_mlx_eligible(
            "boogu_image_turbo",
            &edit_payload("boogu_image_turbo")
        ));

        // UI gating oracle: all three are Mac-supported (reach the Text → Image picker); only Edit
        // advertises `features.edit` (Base/Turbo are T2I — the catalog `edit_image` capability +
        // this flag both gate the Edit tab to `boogu_image_edit`).
        for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
            assert!(
                model_mac_support(model, "image").supported,
                "{model} must be Mac-supported"
            );
        }
        assert!(
            model_mac_support("boogu_image_edit", "image").features.edit,
            "boogu_image_edit supports edit"
        );
        assert!(
            !model_mac_support("boogu_image", "image").features.edit,
            "boogu_image (Base) is text-to-image only"
        );
        assert!(
            !model_mac_support("boogu_image_turbo", "image")
                .features
                .edit,
            "boogu_image_turbo is text-to-image only"
        );
    }

    #[test]
    fn krea_2_turbo_text_to_image_routes_to_mlx() {
        // sc-7572: Krea 2 Turbo has a native `mlx-gen-krea` text-to-image engine and should not be
        // hidden by the Mac model-card gating. It is T2I-only, so edit remains ineligible.
        assert!(image_request_mlx_eligible(
            "krea_2_turbo",
            &object(json!({ "model": "krea_2_turbo", "prompt": "cinematic editorial portrait" }))
        ));
        assert!(image_request_mlx_eligible("krea_2_turbo", &Map::new()));
        assert!(!image_request_mlx_eligible(
            "krea_2_turbo",
            &object(
                json!({ "model": "krea_2_turbo", "mode": "edit_image", "sourceAssetId": "asset_1" })
            )
        ));

        let support = model_mac_support("krea_2_turbo", "image");
        assert!(support.supported, "krea_2_turbo must be Mac-supported");
        assert!(!support.features.edit, "krea_2_turbo is text-to-image only");
    }

    #[test]
    fn sd3_5_text_to_image_routes_to_mlx() {
        // sc-7873 (epic 7841): the three SD3.5 variants have native `mlx-gen-sd3` text-to-image engines
        // (S2 MODEL_TABLE), so they must reach the Text → Image picker (macSupport.supported) rather than
        // being hidden as torch-only. All three are text-to-image only — `edit_image` is rejected.
        for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
            assert!(
                image_request_mlx_eligible(
                    model,
                    &object(json!({ "model": model, "prompt": "a misty alpine lake at dawn" }))
                ),
                "{model} text-to-image must route to MLX"
            );
            assert!(
                image_request_mlx_eligible(model, &Map::new()),
                "{model} bare payload"
            );
            assert!(
                !image_request_mlx_eligible(
                    model,
                    &object(
                        json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" })
                    )
                ),
                "{model} edit is not supported (text-to-image only)"
            );

            // UI gating oracle: Mac-supported (reaches the picker), text-to-image only (no edit tab).
            let support = model_mac_support(model, "image");
            assert!(support.supported, "{model} must be Mac-supported");
            assert!(!support.features.edit, "{model} is text-to-image only");
        }
    }

    #[test]
    fn video_mode_eligibility_admits_flf_only_on_flf_capable_engines() {
        // image_to_video is MLX on every routed model EXCEPT Bernini (text_to_video only — its
        // renderer is Wan2.2-T2V, no still-image-to-video) and SCAIL-2 (animate_character only);
        // text_to_video on every routed model EXCEPT SVD (image-conditioned only, sc-3523) and
        // SCAIL-2 (animate_character only — sc-5448).
        for model in VIDEO_MLX_ROUTED_MODELS {
            assert_eq!(
                video_mode_is_mlx_eligible(model, "image_to_video"),
                *model != "bernini" && *model != "scail2_14b",
                "image_to_video eligibility for {model}"
            );
            assert_eq!(
                video_mode_is_mlx_eligible(model, "text_to_video"),
                *model != "svd" && *model != "scail2_14b",
                "text_to_video eligibility for {model}"
            );
        }
        // SVD serves image_to_video ONLY — no text_to_video, FLF, or anything else.
        assert!(video_mode_is_mlx_eligible("svd", "image_to_video"));
        for mode in [
            "text_to_video",
            "first_last_frame",
            "replace_person",
            "nonsense",
        ] {
            assert!(!video_mode_is_mlx_eligible("svd", mode));
        }
        // Bernini serves text_to_video + the planner editing/reference video modes (sc-4703:
        // video_to_video / reference_to_video / reference_video_to_video) + the multi-source
        // modes (sc-5425: multi_video_to_video / ads2v). It has no classic still-image-to-video
        // / FLF / replace_person (its renderer is Wan2.2-T2V).
        for mode in [
            "text_to_video",
            "video_to_video",
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ] {
            assert!(
                video_mode_is_mlx_eligible("bernini", mode),
                "bernini should serve {mode}"
            );
        }
        for mode in [
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
            "nonsense",
        ] {
            assert!(
                !video_mode_is_mlx_eligible("bernini", mode),
                "bernini should not serve {mode}"
            );
        }
        // The editing/reference + multi-source modes are Bernini-only — every other routed
        // model rejects them.
        for model in VIDEO_MLX_ROUTED_MODELS {
            if *model == "bernini" {
                continue;
            }
            for mode in [
                "video_to_video",
                "reference_to_video",
                "reference_video_to_video",
                "multi_video_to_video",
                "ads2v",
            ] {
                assert!(
                    !video_mode_is_mlx_eligible(model, mode),
                    "{mode} should be Bernini-only, not eligible on {model}"
                );
            }
        }
        // SCAIL-2 serves the standalone character-animation mode (sc-5448, the worker paints its
        // masks from native SAM3) AND cross-identity replace_person (sc-5452, the integrated backend
        // behind the person-track pipeline). No text/image-to-video.
        for mode in ["animate_character", "replace_person"] {
            assert!(
                video_mode_is_mlx_eligible("scail2_14b", mode),
                "scail2 should serve {mode}"
            );
        }
        for mode in [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "video_to_video",
            "nonsense",
        ] {
            assert!(
                !video_mode_is_mlx_eligible("scail2_14b", mode),
                "scail2 should not serve {mode}"
            );
        }
        // animate_character is SCAIL-2-only — every other routed model rejects it.
        for model in VIDEO_MLX_ROUTED_MODELS {
            if *model == "scail2_14b" {
                continue;
            }
            assert!(
                !video_mode_is_mlx_eligible(model, "animate_character"),
                "animate_character should be SCAIL-2-only, not eligible on {model}"
            );
        }
        // first_last_frame: MLX on LTX (base + eros) + Wan TI2V-5B (sc-3055 cutover).
        assert!(video_mode_is_mlx_eligible("ltx_2_3", "first_last_frame"));
        assert!(video_mode_is_mlx_eligible(
            "ltx_2_3_eros",
            "first_last_frame"
        ));
        assert!(video_mode_is_mlx_eligible("wan_2_2", "first_last_frame"));
        // FLF stays torch on the 14B Wan MoE engines (no engine Keyframe path).
        assert!(!video_mode_is_mlx_eligible(
            "wan_2_2_t2v_14b",
            "first_last_frame"
        ));
        assert!(!video_mode_is_mlx_eligible(
            "wan_2_2_i2v_14b",
            "first_last_frame"
        ));
        // extend_clip / video_bridge: MLX on the LTX IC-LoRA path (sc-3522) and Wan TI2V-5B
        // (`wan_2_2`, single-frame boundary keyframe conditioning — sc-3357).
        for mode in ["extend_clip", "video_bridge"] {
            assert!(video_mode_is_mlx_eligible("ltx_2_3", mode));
            assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", mode));
            assert!(video_mode_is_mlx_eligible("wan_2_2", mode));
            // The 14B Wan MoE engines have no `Keyframe` path → torch.
            assert!(!video_mode_is_mlx_eligible("wan_2_2_t2v_14b", mode));
            assert!(!video_mode_is_mlx_eligible("wan_2_2_i2v_14b", mode));
        }
        // replace_person → native Wan-VACE is MLX on the replace-capable models (sc-3521).
        assert!(video_mode_is_mlx_eligible("ltx_2_3", "replace_person"));
        assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", "replace_person"));
        assert!(video_mode_is_mlx_eligible("wan_2_2", "replace_person"));
        // Unknown modes are never eligible.
        assert!(!video_mode_is_mlx_eligible("ltx_2_3", "nonsense"));
    }
}
