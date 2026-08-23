use std::collections::HashMap;
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
    ContractNumber, ExtraFields, GenerationMetrics, GenerationMetricsRow, JobSnapshot, JobStatus,
    JobType, ProgressStage, QueueSummary, WorkerCapability, WorkerSnapshot, WorkerStatus,
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
#[cfg(not(test))]
pub(crate) use routing::catalog::image_family_is_mlx_routed;
#[cfg(test)]
pub(crate) use routing::catalog::*;
// The video memory gate (`crate::video_request`, sc-18814) reads its per-family backend surface
// from the catalog in every build, not only under `cfg(test)`, so these two escape the test-gated
// glob above by name.
#[cfg(not(test))]
pub(crate) use routing::catalog::{
    video_model_has_candle_video_route, video_model_is_mlx_video_routed,
};
pub(crate) use routing::gaps::*;
pub(crate) use routing::mlx::*;

// External re-export surface: `apps/rust-api/src/lib.rs` and the integration test
// (`tests/jobs_store.rs`) import these already-public items from `jobs_store::` directly.
pub use routing::catalog::{
    candle_routed_image_models, checkpoint_plan_checkpoint_id, imported_control_intent_is_material,
    imported_entry_installed_path, imported_image_model_lora_advertisement,
    imported_image_request_provider_eligible, imported_pose_control_mode_is_supported,
    imported_provider_routes, is_builtin_image_model, mac_capabilities, model_candle_support,
    model_mac_support, video_job_type_for_mode, ImportedProviderSurface, MacCapabilities,
    ModelCandleSupport, MAC_NOT_AVAILABLE_LABEL, MLX_ROUTED_TRAINING_KERNELS,
};
pub use routing::gaps::{
    candle_supported, convert_artifact_required_here, mac_rust_supported,
    video_request_is_claimable_by_any_lane, video_request_is_claimable_on_platform,
    UnsupportedReason, CANDLE_NATIVE_CONVERTERS, NATIVE_CONVERTERS,
};
pub use routing::matrix::{backend_capability_matrix, BackendCapabilityMatrix};
pub use routing::{
    canonical_video_route_probe, video_backend_mode_supported,
    video_mode_conditioning_requirements, video_ui_modes,
};

pub const ACTIVE_STATUSES: &[&str] = &[
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
];

const PROGRESS_SIDE_EFFECT_RETRY_BASE_SECONDS: i64 = 5;
const PROGRESS_SIDE_EFFECT_RETRY_MAX_SECONDS: i64 = 5 * 60;
pub const TERMINAL_STATUSES: &[&str] = &["completed", "failed", "canceled", "interrupted"];
/// Pending (accepted-but-not-yet-worker-owned) statuses: a job in one of these is
/// waiting in the queue with no worker to acknowledge a cancel, so it can be
/// terminated immediately. Exactly the two statuses the `cancel_job` fast path — and
/// the bulk [`JobsStore::cancel_pending_jobs`] — flip straight to terminal `canceled`.
/// Kept in lockstep with that fast-path branch and the web `pendingStatuses` set.
pub const PENDING_STATUSES: &[&str] = &["queued", "pending_caption"];
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
    "dataset_parquet_import",
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

/// The pending statuses as a quoted SQL list for `status in (...)` filters, derived
/// once from [`PENDING_STATUSES`] — same anti-drift rationale as
/// [`terminal_statuses_sql`]. Used to select the not-yet-started jobs the bulk
/// [`JobsStore::cancel_pending_jobs`] terminates. Values are crate constants, never
/// user input, so direct interpolation is safe.
fn pending_statuses_sql() -> &'static str {
    static SQL: OnceLock<String> = OnceLock::new();
    SQL.get_or_init(|| {
        PENDING_STATUSES
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

impl JobsStoreError {
    /// Machine-readable SQLite lock classification for the API/worker retry seam. Do not make
    /// callers parse rusqlite's rendered wording: SQLite exposes both BUSY and LOCKED as stable
    /// result codes, while their human messages are not an API contract.
    pub fn is_database_locked(&self) -> bool {
        matches!(
            self,
            Self::Sqlite(error)
                if matches!(
                    error.sqlite_error_code(),
                    Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
                )
        )
    }
}

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
    /// Serializes writers AND owns the process's single long-lived write
    /// connection (sc-11202 / F-025). Every mutating method takes this mutex and
    /// reuses the connection inside it, so the hot claim/heartbeat/progress path
    /// no longer pays a fresh `Connection::open` + `create_dir_all` + WAL/
    /// busy_timeout/foreign_keys pragma round-trip per call. The connection is
    /// created lazily on first write (so `new` stays infallible) and its
    /// WAL/busy_timeout/foreign_keys pragmas are established exactly once when it
    /// opens. The mutex still serializes writers exactly as before, and because
    /// the connection lives entirely behind it, it is never touched by two
    /// threads at once. Read-only methods still open their own short-lived
    /// connection off the mutex and rely on WAL reader isolation (see
    /// `list_jobs`); the separate worker PROCESS keeps its own connection, so
    /// cross-process access and WAL semantics are unchanged.
    lock: Mutex<Option<Connection>>,
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
    /// Host-side remedy for a [`WorkerStatus::Unhealthy`] worker (sc-16260); `None` otherwise.
    pub status_reason: Option<String>,
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
    /// Id of the worker reporting this progress. The store rejects the update
    /// unless this value and the job's `worker_id` are both present and match — a zombie worker
    /// whose job was swept to `interrupted` (worker_id cleared) or reclaimed by
    /// another worker can no longer resurrect or corrupt it (sc-4172). `None`
    /// is retained on the internal type so the wire contract can return the
    /// ownership 409 instead of failing JSON deserialization.
    pub worker_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProgressUpdateOutcome {
    pub job: JobSnapshot,
    pub previous_status: JobStatus,
    /// True only when this call won the ownership/terminal-state race and
    /// committed the supplied progress. Same-status terminal retries return
    /// the existing job with `applied == false`.
    pub applied: bool,
    /// A terminal progress report was accepted but its API-owned catalog /
    /// project side effects have not yet been durably folded into `result_json`.
    /// The owning worker may resume this idempotent work with a same-terminal
    /// retry; competing terminal reporters may not.
    pub side_effects_pending: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StaleSweep {
    pub workers: Vec<WorkerSnapshot>,
    pub jobs: Vec<JobSnapshot>,
}

/// Outcome of [`JobsStore::heartbeat_worker`] (sc-18182).
///
/// The heartbeat can terminate a job as a side effect: an idle heartbeat from a
/// restarted worker orphans the job its previous incarnation was running. Callers
/// must be able to see that job, because an API caller has to publish `job.updated`
/// for it — the web client is SSE-driven and does not poll, so an unannounced
/// terminal transition leaves its progress bar frozen forever.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerHeartbeatOutcome {
    /// The refreshed worker snapshot (this call's primary result).
    pub worker: WorkerSnapshot,
    /// The previously-active job this heartbeat marked `interrupted`, if any.
    pub interrupted_job: Option<JobSnapshot>,
}

impl JobsStore {
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
            lock: Mutex::new(None),
        }
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn initialize(&self) -> JobsStoreResult<()> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(
            "
            create table if not exists jobs (
              id text primary key,
              type text not null,
              status text not null,
              queue_rank integer not null default 0,
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
            create index if not exists idx_jobs_created
              on jobs(created_at);
            create index if not exists idx_jobs_assigned_gpu_status
              on jobs(assigned_gpu, status);
            create index if not exists idx_jobs_worker_status
              on jobs(worker_id, status);

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
        // Durable worker-queue priority. Zero is the normal FIFO lane; positive ranks are
        // assigned monotonically whenever a job jumps to the front. Prompt-refinement jobs get a
        // rank automatically at creation, while the Queue screen can rank any still-pending jobs
        // explicitly. The claim query reads this column before its existing GPU-affinity
        // optimizations, so priority survives restarts and never preempts work already in flight.
        ensure_column(
            &transaction,
            "jobs",
            "queue_rank",
            "integer not null default 0",
        )?;
        transaction.execute_batch(
            "
            create index if not exists idx_jobs_status_queue_rank_created
              on jobs(status, queue_rank desc, created_at);
            ",
        )?;
        // sc-16260: why an `unhealthy` worker withdrew its capabilities — the host-side remedy,
        // so the Queue screen can explain a stalled queue instead of leaving an operator to read
        // container logs. Nullable and absent on every healthy worker.
        ensure_column(&transaction, "workers", "status_reason", "text")?;
        // sc-2086: per-job peak GPU memory % and load %, written by the worker
        // along with progress so a completed row shows the peak the run reached.
        ensure_column(&transaction, "jobs", "peak_gpu_memory_pct", "real")?;
        ensure_column(&transaction, "jobs", "peak_gpu_load_pct", "real")?;
        // Runtime backend label written by the worker ("mlx" / "mps" / "cuda" / "cpu"); `mps`
        // remains readable for historical rows. First-non-null wins so the WorkerProgressCard's
        // arch pill stays stable across the run.
        ensure_column(&transaction, "jobs", "backend", "text")?;
        // Durable per-row ordering for SSE reconciliation. updated_at is
        // intentionally second-granularity, so it cannot distinguish two
        // commits in the same second. The trigger advances revision for every
        // UPDATE without requiring each mutation site to remember the field.
        ensure_column(
            &transaction,
            "jobs",
            "revision",
            "integer not null default 0",
        )?;
        transaction.execute_batch(
            "
            create trigger if not exists jobs_revision_after_update
            after update on jobs
            when new.revision = old.revision
            begin
              update jobs
                 set revision = old.revision + 1
               where id = new.id;
            end;
            ",
        )?;
        // Durable handoff between terminal progress acceptance and the API's
        // idempotent catalog/project writes. The bit is set in the same
        // transaction as terminal acceptance and cleared only by the guarded
        // result CAS after those writes succeed.
        ensure_column(
            &transaction,
            "jobs",
            "progress_side_effects_pending",
            "integer not null default 0",
        )?;
        // Persist retry scheduling with the handoff itself. Failed side effects
        // leave the due set until their backoff expires, preventing a fixed
        // oldest-first batch from hot-looping poison rows and starving newer
        // handoffs after an API restart.
        ensure_column(
            &transaction,
            "jobs",
            "progress_side_effects_retry_count",
            "integer not null default 0",
        )?;
        ensure_column(
            &transaction,
            "jobs",
            "progress_side_effects_retry_at",
            "integer not null default 0",
        )?;
        transaction.execute_batch(
            "
            create index if not exists idx_jobs_pending_side_effect_retry
              on jobs(progress_side_effects_retry_at, updated_at, id)
             where progress_side_effects_pending = 1
               and status in ('completed', 'failed', 'canceled', 'interrupted');
            ",
        )?;
        // Soft-hide marker for the "Clear completed" queue action (sc-12231, issue #1556).
        // A cleared job is dropped from the operator's queue list + counts (see
        // `list_jobs` / `queue_summary`) but the row is deliberately kept: the
        // Generation Stats feed (epic 10402) inner-joins `generation_metrics` to
        // `jobs`, so deleting the row would silently wipe that run's stats and
        // orphan its metrics. Non-null == "cleared from the queue", timestamped.
        ensure_column(&transaction, "jobs", "cleared_at", "text")?;
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
        // Retention may remove the owning queue row, but Generation Stats is
        // historical product data rather than queue history. Materialize the
        // joined row before purging so aggregate charts remain complete.
        transaction.execute_batch(
            "
            create table if not exists generation_metrics_history as
              select m.*, j.type as j_type, j.status as j_status,
                     j.project_id as j_project_id, j.created_at as j_created_at
                from generation_metrics m join jobs j on j.id = m.job_id
               where 0;
            create unique index if not exists idx_genmetrics_history_job
              on generation_metrics_history(job_id);
            create index if not exists idx_genmetrics_history_created
              on generation_metrics_history(j_created_at);
            ",
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Remove terminal queue history older than `retention_days`.
    ///
    /// Zero disables retention. The cutoff is resolved to a fixed timestamp
    /// here, in Rust, rather than being left as a `datetime('now', ...)`
    /// modifier for SQLite to evaluate: `now` is constant only within a single
    /// `sqlite3_step`, so a modifier re-evaluated by each of the four statements
    /// below would let the cutoff ADVANCE mid-transaction. A job whose
    /// `completed_at` fell between two of those evaluations was then deleted
    /// from `jobs` and `generation_metrics` without ever being materialized
    /// into the history table — a silent, permanent loss of the run from
    /// Generation Stats, re-rolled on every API start (sc-17597).
    pub fn purge_terminal_jobs_older_than(&self, retention_days: u32) -> JobsStoreResult<usize> {
        if retention_days == 0 {
            return Ok(0);
        }
        let cutoff = format_unix_seconds(now_unix_seconds() - i64::from(retention_days) * 86_400);
        self.purge_terminal_jobs_completed_before(&cutoff)
    }

    /// Remove terminal queue history completed strictly before `cutoff` (a
    /// `YYYY-MM-DDTHH:MM:SSZ` UTC timestamp, the shape [`utc_now`] emits).
    ///
    /// Job-owned metrics are materialized into the independent Generation Stats
    /// history and then deleted in the same immediate transaction as their
    /// owning jobs. Legacy orphan metrics are also removed. Every statement
    /// shares the one caller-supplied cutoff, so the window cannot move
    /// underneath the sweep; a job completed exactly ON the cutoff is retained.
    pub fn purge_terminal_jobs_completed_before(&self, cutoff: &str) -> JobsStoreResult<usize> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let terminal = terminal_statuses_sql();
        let predicate = format!(
            "status in ({terminal}) and completed_at is not null \
             and datetime(completed_at) < datetime(?1)"
        );
        transaction.execute(
            &format!(
                "insert or replace into generation_metrics_history
                 select m.*, j.type, j.status, j.project_id, j.created_at
                   from generation_metrics m join jobs j on j.id = m.job_id
                  where j.{predicate}"
            ),
            params![cutoff],
        )?;
        transaction.execute(
            &format!(
                "delete from generation_metrics where job_id in \
                 (select id from jobs where {predicate})"
            ),
            params![cutoff],
        )?;
        transaction.execute(
            "delete from generation_metrics
              where not exists (select 1 from jobs where jobs.id = generation_metrics.job_id)",
            [],
        )?;
        let deleted = transaction.execute(
            &format!("delete from jobs where {predicate}"),
            params![cutoff],
        )?;
        transaction.commit()?;
        Ok(deleted)
    }

    pub fn mark_interrupted_on_startup(&self) -> JobsStoreResult<Vec<JobSnapshot>> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
            // sc-16260: `status_reason` is cleared alongside the status everywhere `offline` is
            // written. It describes why an ALIVE worker withdrew its capabilities; on a worker we
            // have just declared gone it is stale by construction, and `GET /api/v1/workers` would
            // otherwise keep handing clients a host remedy for a worker that is no longer there.
            "update workers set status = 'offline', current_job_id = null, status_reason = null              where status != 'offline'",
            [],
        )?;
        let updated_ids = interrupted_ids
            .iter()
            .chain(stranded_pending_ids.iter())
            .cloned()
            .collect::<Vec<_>>();
        let updated_jobs = self.jobs_by_ids(&transaction, &updated_ids)?;
        transaction.commit()?;
        Ok(updated_jobs)
    }

    pub fn create_job(&self, request: CreateJob) -> JobsStoreResult<JobSnapshot> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let connection = self.open_connection()?;
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

    /// Find an in-flight (non-terminal) `model_download` job for `model_id`, so the convert request
    /// boundary can refuse a convert whose source weights are still streaming instead of queueing a
    /// job that fails the moment a worker claims it. Returns the newest such job, or `None`.
    ///
    /// Keyed on the payload `modelId`, not the repo: a shared source repo backs several catalog cards
    /// (the three Anima variants live in `circlestone-labs/Anima`), and converting variant A while
    /// variant B downloads is legitimate — A's own weights are already on disk. The file-presence
    /// half of the gate (`convert_source_state`) covers the rest.
    ///
    /// Read-only single-SELECT: no write mutex, relies on WAL reader isolation like `list_jobs`
    /// (sc-8950 / F-148).
    pub fn find_active_model_download_job(
        &self,
        model_id: &str,
    ) -> JobsStoreResult<Option<JobSnapshot>> {
        let connection = self.open_connection()?;
        let mut statement = connection.prepare(&format!(
            "
            select * from jobs
             where type = 'model_download'
               and status not in ({terminal})
             order by created_at desc
            ",
            terminal = terminal_statuses_sql()
        ))?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        Ok(candidates
            .into_iter()
            .find(|job| job.payload.get("modelId").and_then(Value::as_str) == Some(model_id)))
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
        let connection = self.open_connection()?;
        let limit = limit.clamp(1, 500);
        // A cleared job (sc-12231, issue #1556) is soft-hidden from every queue
        // list surface — the operator asked for it gone, so it never comes back
        // via a status filter either. The row still exists for Generation Stats.
        let mut conditions: Vec<&str> = vec!["cleared_at is null"];
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

    /// Bounded reconnect history ordered by the last durable mutation rather
    /// than creation time. The SSE endpoint supplements this generic history
    /// with every pre-disconnect active id named by the client, so correctness
    /// does not depend on a terminal transition remaining inside this cap.
    pub fn list_jobs_recently_updated(&self, limit: u32) -> JobsStoreResult<Vec<JobSnapshot>> {
        let connection = self.open_connection()?;
        let limit = limit.clamp(1, 500);
        let mut statement = connection.prepare(
            "select * from jobs
              where cleared_at is null
              order by updated_at desc, id desc
              limit ?1",
        )?;
        let jobs = collect_jobs(statement.query_map(params![limit], row_to_job)?)?;
        Ok(jobs)
    }

    /// Load every retained job named by a reconnecting client, preserving the
    /// requested order and ignoring ids whose rows were already purged. This
    /// supplements the bounded recent-history snapshot with the client's exact
    /// pre-disconnect active set, so an older terminal transition cannot fall
    /// outside that history window.
    pub fn list_existing_jobs_by_ids(
        &self,
        job_ids: &[String],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if job_ids.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.open_connection()?;
        let ids_json = dumps(job_ids)?;
        let mut statement = connection.prepare(
            "select jobs.*
               from json_each(?1) as requested
               join jobs on jobs.id = requested.value
              where jobs.cleared_at is null
              order by cast(requested.key as integer)",
        )?;
        let jobs = collect_jobs(statement.query_map(params![ids_json], row_to_job)?)?;
        Ok(jobs)
    }

    pub fn get_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.open_connection()?;
        self.get_job_on_connection(&connection, job_id)
    }

    /// Return clear tombstones only for the reconnecting client's bounded known
    /// rows. Tombstones live with retained jobs and disappear under normal
    /// retention, but reconnect cost must never scale with the full retained
    /// history (including deployments where retention is disabled).
    pub fn cleared_job_ids_by_ids(&self, job_ids: &[String]) -> JobsStoreResult<Vec<String>> {
        if job_ids.is_empty() {
            return Ok(Vec::new());
        }
        let connection = self.open_connection()?;
        let ids_json = dumps(job_ids)?;
        let mut statement = connection.prepare(
            "select jobs.id
               from json_each(?1) as requested
               join jobs on jobs.id = requested.value
              where jobs.cleared_at is not null
              order by cast(requested.key as integer)",
        )?;
        let ids = statement
            .query_map(params![ids_json], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(JobsStoreError::from)?;
        Ok(ids)
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let connection = self.open_connection()?;
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
        let connection = self.open_connection()?;
        let limit = limit.clamp(1, 5000);
        let mut conditions: Vec<&str> = Vec::new();
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(job_type) = job_type {
            conditions.push("stats.j_type = ?");
            bindings.push(Box::new(job_type.to_owned()));
        }
        if let Some(model) = model {
            conditions.push("stats.model = ?");
            bindings.push(Box::new(model.to_owned()));
        }
        if let Some(quant_label) = quant_label {
            conditions.push("stats.quant_label = ?");
            bindings.push(Box::new(quant_label.to_owned()));
        }
        let mut sql = String::from(
            "select stats.* from (
               select m.*, j.type as j_type, j.status as j_status,
                      j.project_id as j_project_id, j.created_at as j_created_at
                 from generation_metrics m join jobs j on j.id = m.job_id
               union all
               select * from generation_metrics_history
             ) stats",
        );
        if !conditions.is_empty() {
            sql.push_str(" where ");
            sql.push_str(&conditions.join(" and "));
        }
        sql.push_str(" order by stats.j_created_at desc limit ?");
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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

    /// Bulk-cancel every PENDING (not-yet-worker-owned) job — the fleet analog of the
    /// `queued`/`pending_caption` fast path in [`cancel_job`] (issue: "cancel ALL
    /// pending jobs"). Optionally scoped to one project (matching the queue's project
    /// filter); omitted / `None` cancels every project's pending jobs. Each matching
    /// row goes straight to terminal `canceled` in one UPDATE — no worker owns it to
    /// acknowledge — and the updated snapshots are returned newest first so the caller
    /// can broadcast `job.updated` per job and hand the acting client the flipped
    /// cards immediately.
    ///
    /// Deliberately scoped to `queued` + `pending_caption` ([`PENDING_STATUSES`])
    /// ONLY. An active (worker-owned) job needs cooperative acknowledgement and is
    /// left untouched — cancel those one at a time via [`cancel_job`] so the owning
    /// worker self-terminates; already-terminal jobs never match either. The
    /// `pending_caption` guarded promotion (sc-9120) only ever advances a row that is
    /// STILL `pending_caption`, so flipping it to `canceled` here can't be resurrected
    /// — same race-free reasoning as the single-job fast path.
    pub fn cancel_pending_jobs(
        &self,
        project_id: Option<&str>,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Collect the ids first (scoped so the borrowed statement drops before the
        // UPDATE runs on the same transaction), mirroring `clear_terminal_jobs`.
        let mut select_sql = format!(
            "select id from jobs where status in ({pending})",
            pending = pending_statuses_sql()
        );
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(project_id) = project_id {
            select_sql.push_str(" and project_id = ?");
            bindings.push(Box::new(project_id.to_owned()));
        }
        select_sql.push_str(" order by created_at desc");
        let pending_ids: Vec<String> = {
            let mut statement = transaction.prepare(&select_sql)?;
            let rows = statement.query_map(params_from_iter(bindings.iter()), |row| {
                row.get::<_, String>(0)
            })?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row?);
            }
            ids
        };

        if !pending_ids.is_empty() {
            let now = utc_now();
            // Same terminal transition as the `cancel_job` fast path (lines above),
            // applied set-wide. The `status in (...)` guard is re-checked inside the
            // UPDATE so a row that just left the pending stage under a concurrent
            // writer is skipped rather than clobbered.
            let mut update_sql = format!(
                "
                update jobs
                   set status = 'canceled',
                       stage = 'canceled',
                       progress = 1,
                       cancel_requested = 1,
                       message = 'Canceled before a worker started.',
                       canceled_at = ?,
                       completed_at = ?,
                       updated_at = ?
                 where status in ({pending})
                ",
                pending = pending_statuses_sql()
            );
            let mut update_bindings: Vec<Box<dyn ToSql>> =
                vec![Box::new(now.clone()), Box::new(now.clone()), Box::new(now)];
            if let Some(project_id) = project_id {
                update_sql.push_str(" and project_id = ?");
                update_bindings.push(Box::new(project_id.to_owned()));
            }
            transaction.execute(&update_sql, params_from_iter(update_bindings.iter()))?;
        }

        // Re-read the updated snapshots (newest first) so callers broadcast the real
        // post-cancel state, not the stale pre-update rows.
        let canceled = self.jobs_by_ids(&transaction, &pending_ids)?;
        transaction.commit()?;
        Ok(canceled)
    }

    /// Move selected not-yet-started jobs to the front of the worker queue.
    ///
    /// Ranks are monotonic rather than a boolean priority flag: every invocation really does
    /// "jump to top" relative to earlier automatic or manual promotions. The selected jobs keep
    /// their current scheduling order as a group. Active and terminal jobs are ignored under the
    /// same transaction, so a stale Queue-screen selection can never interrupt worker-owned work.
    pub fn prioritize_jobs(&self, job_ids: &[String]) -> JobsStoreResult<Vec<JobSnapshot>> {
        if job_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let ids_json = dumps(job_ids)?;
        let pending = pending_statuses_sql();
        let mut statement = transaction.prepare(&format!(
            "
            select * from jobs
             where id in (select distinct value from json_each(?1))
               and status in ({pending})
             order by queue_rank desc, created_at asc, id asc
            "
        ))?;
        let selected = collect_jobs(statement.query_map(params![ids_json], row_to_job)?)?;
        drop(statement);
        if selected.is_empty() {
            transaction.commit()?;
            return Ok(Vec::new());
        }

        let current_max =
            transaction.query_row("select coalesce(max(queue_rank), 0) from jobs", [], |row| {
                row.get::<_, i64>(0)
            })?;
        let now = utc_now();
        let selected_count = i64::try_from(selected.len()).unwrap_or(i64::MAX);
        for (index, job) in selected.iter().enumerate() {
            let offset = selected_count.saturating_sub(i64::try_from(index).unwrap_or(i64::MAX));
            let queue_rank = current_max.saturating_add(offset);
            transaction.execute(
                &format!(
                    "update jobs
                        set queue_rank = ?1,
                            updated_at = ?2
                      where id = ?3 and status in ({pending})"
                ),
                params![queue_rank, now, job.id],
            )?;
        }

        let prioritized_ids = selected
            .iter()
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        let prioritized = self.jobs_by_ids(&transaction, &prioritized_ids)?;
        transaction.commit()?;
        Ok(prioritized)
    }

    pub fn retry_job(&self, job_id: &str, request: RetryJob) -> JobsStoreResult<JobSnapshot> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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

    /// Soft-hide every terminal (completed / failed / canceled / interrupted) job
    /// from the operator's queue, optionally scoped to one project (sc-12231,
    /// issue #1556 — "clear completed items from the queue"). Stamps `cleared_at`
    /// on the matching rows so `list_jobs` and `queue_summary` drop them, and
    /// returns the cleared ids (newest first) so the caller can prune them from
    /// live client state.
    ///
    /// The rows are deliberately KEPT, not deleted: the Generation Stats feed
    /// (`list_generation_metrics`, epic 10402) inner-joins `generation_metrics`
    /// to `jobs`, so a hard delete would silently wipe those runs from the stats
    /// charts and orphan the metrics rows. Generated assets live in the project
    /// store independently, so clearing a job never removes its outputs.
    /// Already-cleared rows and still-active jobs are left untouched.
    pub fn clear_terminal_jobs(&self, project_id: Option<&str>) -> JobsStoreResult<Vec<String>> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Collect the ids first so the caller can prune exactly what was cleared
        // (the set can exceed the client's visible cap). Scoped so the borrowed
        // statement is dropped before the UPDATE runs on the same transaction.
        let mut select_sql = format!(
            "select id from jobs where status in ({terminal}) and cleared_at is null",
            terminal = terminal_statuses_sql()
        );
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
        if let Some(project_id) = project_id {
            select_sql.push_str(" and project_id = ?");
            bindings.push(Box::new(project_id.to_owned()));
        }
        select_sql.push_str(" order by created_at desc");
        let cleared_ids: Vec<String> = {
            let mut statement = transaction.prepare(&select_sql)?;
            let rows = statement.query_map(params_from_iter(bindings.iter()), |row| {
                row.get::<_, String>(0)
            })?;
            let mut ids = Vec::new();
            for row in rows {
                ids.push(row?);
            }
            ids
        };

        if !cleared_ids.is_empty() {
            let now = utc_now();
            let mut update_sql = format!(
                "update jobs set cleared_at = ? where status in ({terminal}) and cleared_at is null",
                terminal = terminal_statuses_sql()
            );
            let mut update_bindings: Vec<Box<dyn ToSql>> = vec![Box::new(now)];
            if let Some(project_id) = project_id {
                update_sql.push_str(" and project_id = ?");
                update_bindings.push(Box::new(project_id.to_owned()));
            }
            transaction.execute(&update_sql, params_from_iter(update_bindings.iter()))?;
        }
        transaction.commit()?;
        Ok(cleared_ids)
    }

    /// Soft-hide a single terminal job from the queue (sc-12231, issue #1556) —
    /// the per-card "×" dismiss, the individual-item twin of
    /// [`clear_terminal_jobs`]. Stamps `cleared_at` so the row drops out of
    /// `list_jobs` / `queue_summary` while staying in the table for Generation
    /// Stats, and returns the updated snapshot.
    ///
    /// Only a TERMINAL job can be cleared: an active job would keep emitting
    /// progress and be re-added to the client's queue on the next SSE tick, and
    /// "clear" means tidying finished work — cancel an in-flight job instead. A
    /// non-terminal job is rejected with [`JobsStoreError::InvalidStatus`] (400);
    /// an already-cleared job is idempotent (the guarded UPDATE no-ops).
    pub fn clear_job(&self, job_id: &str) -> JobsStoreResult<JobSnapshot> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        if !is_terminal_status(job.status.as_str()) {
            return Err(JobsStoreError::InvalidStatus(format!(
                "job {job_id} is {} — only completed, failed, canceled, or interrupted jobs can be cleared",
                job.status.as_str()
            )));
        }
        transaction.execute(
            "update jobs set cleared_at = ?1 where id = ?2 and cleared_at is null",
            params![utc_now(), job_id],
        )?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        transaction.commit()?;
        Ok(job)
    }

    pub fn register_worker(&self, request: RegisterWorker) -> JobsStoreResult<WorkerSnapshot> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
              -- A registration is the worker re-advertising what it can serve, so it also
              -- retires any previous unhealthy reason (sc-16260): the recovery path re-registers
              -- with the full capability set, and leaving the old remedy behind would tell an
              -- operator to fix a host that is already fixed. An unhealthy worker re-asserts its
              -- reason on the very next heartbeat, and its capabilities are withheld at
              -- registration either way, so nothing routes to it during the gap.
              status_reason = null,
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

    pub fn heartbeat_worker(
        &self,
        request: WorkerHeartbeat,
    ) -> JobsStoreResult<WorkerHeartbeatOutcome> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let worker = self.get_worker_on_connection(&transaction, &request.worker_id)?;
        let now = utc_now();
        // Reported to the caller so it can publish `job.updated` for the job this
        // heartbeat terminated (sc-18182).
        let mut interrupted_job = None;
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
                    interrupted_job =
                        Some(self.get_job_on_connection(&transaction, &previous_job_id)?);
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
                   status_reason = ?5,
                   last_seen_at = ?6
             where id = ?7
            ",
            params![
                request.status.as_str(),
                request.current_job_id,
                dumps(&request.loaded_models)?,
                optional_dumps(request.utilization.as_ref())?,
                // Written unconditionally, so a worker that recovers clears its own reason on the
                // very next heartbeat rather than carrying a stale remedy for a fixed host
                // (sc-16260 AC 4). `None` on every non-unhealthy heartbeat.
                request.status_reason,
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
        Ok(WorkerHeartbeatOutcome {
            worker,
            interrupted_job,
        })
    }

    pub fn mark_stale_workers_interrupted(
        &self,
        timeout_seconds: u64,
    ) -> JobsStoreResult<StaleSweep> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
                       status_reason = null,
                       last_seen_at = ?1
                 where id in ({placeholders})
                "
            ),
            params_from_iter(worker_params),
        )?;

        let updated_workers = self.workers_by_ids(&transaction, &worker_ids)?;
        let active_job_ids = active_jobs
            .iter()
            .map(|job| job.id.clone())
            .collect::<Vec<_>>();
        let updated_jobs = self.jobs_by_ids(&transaction, &active_job_ids)?;
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
                   status_reason = null,
                   last_seen_at = ?1
             where id = ?2
            ",
            params![now, worker_id],
        )?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-required" grace sweep (epic 3482 / sc-3483). When `mlx_required`, the
    /// synthetic non-mlx descriptor never claims an MLX-eligible job — it defers unconditionally
    /// to the in-process `mlx` worker (see `should_defer_*`). If no **live** `mlx` worker
    /// claims such a job within the grace window — because the worker is down, never
    /// started, or has been crashed longer than the supervisor's auto-restart can
    /// self-heal — the job would otherwise sit queued forever. This fails those jobs
    /// terminal (`status = failed`) with an actionable `mlx_unavailable` error naming the
    /// model + job type, so the failure is loud and points at the real gap instead of
    /// entering the legacy MPS compatibility branch.
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
        let failed = self.jobs_by_ids(&transaction, &failed_ids)?;
        transaction.commit()?;
        Ok(failed)
    }

    /// macOS "MLX-unsupported" enforce sweep (epic 3482 / sc-3484). When `mlx_required` AND
    /// `enforce`, fails every queued job the Rust/MLX flow can't run (`mac_rust_supported`
    /// returns `Err`) terminal with a feature-precise `mlx_unsupported` error — the forcing
    /// function that turns an unsupported native gap into a loud, named failure instead of leaving
    /// it queued. Unlike the stranded sweep there is no grace window: an unsupported job is
    /// permanently unsupported until its surface is ported or dropped, so it fails immediately.
    ///
    /// Default mode is **warn** (`enforce == false`) → this sweep is a no-op; normal capability
    /// routing either finds a capable native worker or leaves the job queued. Flipping
    /// `mlx_required` on for observation surfaces the gap list. Off (`!mlx_required`) →
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
            failed.push((job.id, reason));
        }
        let failed_ids = failed
            .iter()
            .map(|(job_id, _reason)| job_id.clone())
            .collect::<Vec<_>>();
        let updated_jobs = self.jobs_by_ids(&transaction, &failed_ids)?;
        transaction.commit()?;
        Ok(updated_jobs
            .into_iter()
            .zip(failed.into_iter().map(|(_job_id, reason)| reason))
            .collect())
    }

    /// Off-Mac candle grace sweep (sc-5502, epic 5483) — the Windows/Linux twin of
    /// [`Self::fail_stranded_mlx_jobs`]. When `candle_required`, fails any candle-eligible job left
    /// queued past the grace window when no live candle worker exists, terminal with
    /// `candle_unavailable` — so a deployment can fail loudly instead of queuing forever.
    ///
    /// "Live candle worker" = a worker advertising the exact `candle` marker capability that is not
    /// offline and has a heartbeat within `grace_seconds`. Capabilities are decoded through the same
    /// typed parser as [`row_to_worker`], rather than substring-matching JSON.
    /// While one exists (even merely busy) this is a no-op and candle-eligible jobs wait, so a
    /// transient candle crash the supervisor restarts inside the window never fails a job. Off
    /// (`!candle_required`) it returns immediately, leaving normal capability routing unaffected.
    /// Returns the jobs it failed so the caller can surface the
    /// structured event and publish their updates.
    pub fn fail_stranded_candle_jobs(
        &self,
        candle_required: bool,
        grace_seconds: u64,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if !candle_required {
            return Ok(Vec::new());
        }
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();
        let grace = i64::try_from(grace_seconds.max(1)).unwrap_or(i64::MAX);
        let cutoff = format_unix_seconds(now.saturating_sub(grace));

        // A live candle worker means candle-eligible jobs should wait for it — it may simply be
        // busy. Only when none has checked in within the window do we treat candle as unavailable
        // and fail the stranded jobs.
        let live_candle_worker = {
            let mut statement = transaction.prepare(
                "
                select capabilities_json from workers
                 where status != 'offline'
                   and last_seen_at >= ?1
                ",
            )?;
            let capabilities = statement
                .query_map(params![cutoff], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            capabilities
                .iter()
                .any(|encoded| encoded_worker_has_capability(encoded, "candle"))
        };
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
        let failed = self.jobs_by_ids(&transaction, &failed_ids)?;
        transaction.commit()?;
        Ok(failed)
    }

    /// **The platform-reachability sweep (sc-19570).** Fails, terminal, every queued video job
    /// whose (model, mode) pair no lane that can exist on `host_os` will ever claim.
    ///
    /// This is where the platform-conditional refusal lives, and the reason it lives HERE rather
    /// than at `POST /api/v1/video/jobs`: an HTTP contract is not platform-dependent. The route
    /// answers `201` for the same body on every host; what varies is the job's *execution outcome*,
    /// which is inherently a property of the machine. sc-19570 shipped the refusal as a `400` first
    /// and that was ruled out — the published surface must not differ by OS.
    ///
    /// It closes the real defect, which is not "the request was accepted" but "the job never
    /// terminates": an MLX-only pair submitted off-Mac sat `queued` / "Waiting for an available
    /// worker." with no error and no terminal state, forever. None of the four existing sweeps
    /// rescues it. [`Self::fail_stranded_candle_jobs`] returns early the moment ANY live candle
    /// worker exists — and one normally does; the job is unclaimable, not unserved. Its `mlx` twin
    /// is `mlx_required`-gated and inert off-Mac. Both `fail_unsupported_*` sweeps default to
    /// **warn**. So the job fell through all four.
    ///
    /// **No flag and no grace window,** unlike every sweep above it. Unreachability is structural
    /// rather than transient: no worker capable of claiming the job can register on this OS at all,
    /// so there is no window to wait out, and gating it behind a rollout switch would leave the
    /// hang in place for every default deployment — which is exactly the state sc-19570 found. On
    /// macOS it is inert by construction ([`video_request_is_claimable_on_platform`] returns `true`
    /// there unconditionally), so no Mac-served pair is touched.
    ///
    /// Scoped to the four video job types via [`video_job_is_platform_unreachable`] — the same
    /// range `create_video_job` enqueues — so it can never reach an image, training or upscale job.
    /// Returns the jobs it failed so the caller can emit the structured event and publish updates.
    pub fn fail_platform_unreachable_jobs(
        &self,
        host_os: &str,
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        // Cheap exit on the platform where this can never fire, before taking the write lock.
        if matches!(host_os, "macos" | "darwin") {
            return Ok(Vec::new());
        }
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_unix_seconds();

        let mut statement = transaction.prepare(
            "
            select * from jobs
             where status = 'queued'
             order by created_at asc
            ",
        )?;
        let candidates = collect_jobs(statement.query_map([], row_to_job)?)?;
        drop(statement);

        let now_text = format_unix_seconds(now);
        let mut failed_ids = Vec::new();
        for job in candidates {
            if !video_job_is_platform_unreachable(&job, host_os) {
                continue;
            }
            let error = platform_unreachable_error(&job, host_os);
            transaction.execute(
                "
                update jobs
                   set status = 'failed',
                       stage = 'failed',
                       message = 'This mode has no backend on this platform.',
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
        let failed = self.jobs_by_ids(&transaction, &failed_ids)?;
        transaction.commit()?;
        Ok(failed)
    }

    /// Off-Mac "candle-unsupported" enforce sweep (sc-5502, epic 5483) — the Windows/Linux twin of
    /// [`Self::fail_unsupported_mlx_jobs`]. When `candle_required` AND `enforce`, fails every queued
    /// job the candle/CUDA flow can't run ([`candle_supported`] returns `Err`) terminal with a
    /// feature-precise `candle_unsupported` error — the forcing function that turns an unsupported
    /// native gap into a loud, named failure instead of leaving it queued. Unlike the stranded sweep there is
    /// no grace window: an unsupported job is permanently unsupported until its surface is ported or
    /// dropped, so it fails immediately.
    ///
    /// Default mode is **warn** (`enforce == false`) → this sweep is a no-op; normal capability
    /// routing either finds a capable native worker or leaves the job queued. Flipping
    /// `candle_required` on for observation surfaces the gap list. Off (`!candle_required`) →
    /// immediate no-op. Candle-*eligible* jobs are `Ok`
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
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
            failed.push((job.id, reason));
        }
        let failed_ids = failed
            .iter()
            .map(|(job_id, _reason)| job_id.clone())
            .collect::<Vec<_>>();
        let updated_jobs = self.jobs_by_ids(&transaction, &failed_ids)?;
        transaction.commit()?;
        Ok(updated_jobs
            .into_iter()
            .zip(failed.into_iter().map(|(_job_id, reason)| reason))
            .collect())
    }

    pub fn claim_next_job(&self, worker_id: &str) -> JobsStoreResult<Option<JobSnapshot>> {
        Ok(self.claim_next_job_routed(worker_id, false)?.0)
    }

    /// Like [`Self::claim_next_job`], but also reports the native-GPU routing decision
    /// so the caller (the API claim handler) can log *why* a job landed where it did —
    /// the single most useful line for diagnosing an MLX-eligible job claimed elsewhere
    /// (sc-3449). A `None` decision means the claim was routing-neutral: no job was
    /// available, an unrelated balancing deferral fired, or the job is one no `mlx`
    /// worker would ever want.
    pub fn claim_next_job_routed(
        &self,
        worker_id: &str,
        mlx_required: bool,
    ) -> JobsStoreResult<(Option<JobSnapshot>, Option<RouteDecision>)> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
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
             order by queue_rank desc, created_at asc
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
        self.update_job_progress_internal(job_id, update, false)
            .map(|outcome| outcome.job)
    }

    /// Accept progress for the HTTP API, durably marking terminal reports until
    /// their API-owned catalog/project side effects and result augmentation
    /// complete. Direct store callers use [`Self::update_job_progress`] and do
    /// not participate in that API handoff.
    pub fn update_job_progress_with_outcome(
        &self,
        job_id: &str,
        update: ProgressUpdate,
    ) -> JobsStoreResult<ProgressUpdateOutcome> {
        self.update_job_progress_internal(job_id, update, true)
    }

    fn update_job_progress_internal(
        &self,
        job_id: &str,
        update: ProgressUpdate,
        track_terminal_side_effects: bool,
    ) -> JobsStoreResult<ProgressUpdateOutcome> {
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

        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Guard against zombie-worker writes (sc-4172): a worker that went
        // silent long enough for the stale sweep to mark its job `interrupted`
        // (or whose job the user canceled) must not resurrect it with a late
        // progress report — that's exactly the failure mode the heartbeat
        // machinery exists to handle.
        let current = self.get_job_on_connection(&transaction, job_id)?;
        let previous_status = current.status.clone();
        if is_terminal_status(current.status.as_str()) {
            let side_effects_pending =
                self.progress_side_effects_pending_on_connection(&transaction, job_id)?;
            // Idempotent re-report of the same terminal status (e.g. a retried
            // "canceled" POST) succeeds without touching the row, but it is
            // still an authenticated worker operation. Clearing a terminal row
            // only hides it from queue surfaces; it must not turn the no-op path
            // into an ownership bypass.
            if current.status == update.status {
                match (update.worker_id.as_deref(), current.worker_id.as_deref()) {
                    (Some(reporter), Some(owner)) if reporter == owner => {}
                    _ => {
                        return Err(JobsStoreError::NotJobOwner {
                            job_id: job_id.to_owned(),
                        });
                    }
                }
                return Ok(ProgressUpdateOutcome {
                    job: current,
                    previous_status,
                    applied: false,
                    side_effects_pending,
                });
            }
            return Err(JobsStoreError::TerminalJobImmutable {
                job_id: job_id.to_owned(),
                status: current.status.as_str().to_owned(),
            });
        }
        match (update.worker_id.as_deref(), current.worker_id.as_deref()) {
            (Some(reporter), Some(owner)) if reporter == owner => {}
            _ => {
                return Err(JobsStoreError::NotJobOwner {
                    job_id: job_id.to_owned(),
                });
            }
        }
        let now = utc_now();
        let completed_at = is_terminal_status(update.status.as_str()).then_some(now.clone());
        let canceled_at = (update.status == JobStatus::Canceled).then_some(now.clone());
        let side_effects_pending =
            track_terminal_side_effects && is_terminal_status(update.status.as_str());
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
                   backend = coalesce(backend, ?13),
                   progress_side_effects_pending = ?14,
                   progress_side_effects_retry_count = 0,
                   progress_side_effects_retry_at = 0
             where id = ?15
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
                side_effects_pending,
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
        Ok(ProgressUpdateOutcome {
            job,
            previous_status,
            applied: true,
            side_effects_pending,
        })
    }

    /// Re-check a durable terminal side-effect handoff immediately before the
    /// API performs external writes. The API serializes this check + work within
    /// one process; the database predicate preserves ownership across retries
    /// and process restarts.
    pub fn pending_terminal_progress_side_effects(
        &self,
        job_id: &str,
        worker_id: Option<&str>,
        status: JobStatus,
    ) -> JobsStoreResult<Option<JobSnapshot>> {
        let connection = self.open_connection()?;
        let current = self.get_job_on_connection(&connection, job_id)?;
        if current.status != status
            || !self.progress_side_effects_pending_on_connection(&connection, job_id)?
        {
            return Ok(None);
        }
        match (worker_id, current.worker_id.as_deref()) {
            (Some(reporter), Some(owner)) if reporter == owner => Ok(Some(current)),
            _ => Err(JobsStoreError::NotJobOwner {
                job_id: job_id.to_owned(),
            }),
        }
    }

    /// The batch that is due *now* — the shape production recovery wants.
    /// Resolving the instant here, rather than inside the query, is what lets
    /// [`Self::pending_terminal_progress_side_effect_job_ids_as_of`] be driven
    /// from a fixed instant; see it for the behavior and for why.
    pub fn pending_terminal_progress_side_effect_job_ids(
        &self,
        limit: usize,
    ) -> JobsStoreResult<Vec<String>> {
        self.pending_terminal_progress_side_effect_job_ids_as_of(now_unix_seconds(), limit)
    }

    /// Return a bounded batch of terminal jobs whose API-owned side effects
    /// still need recovery, judging dueness as of `as_of` (Unix seconds)
    /// instead of the wall clock. The API drains this durable queue at startup
    /// and on a background cadence, so recovery does not depend on a worker
    /// repeating a terminal progress report after it already observed the
    /// committed terminal state. A row is due once its deadline has *arrived*,
    /// so a row whose `progress_side_effects_retry_at` equals `as_of` is
    /// included.
    ///
    /// Production always passes `now`. Tests pass a *frozen* instant so that an
    /// assertion about which rows are due describes the durable retry schedule
    /// rather than how long the test itself took to run: with the wall clock,
    /// a slow pass could let the 5-second first-failure backoff expire between
    /// deferring a row and scanning for it, and rows that were correctly
    /// deferred would legitimately come back due (sc-17640).
    pub fn pending_terminal_progress_side_effect_job_ids_as_of(
        &self,
        as_of: i64,
        limit: usize,
    ) -> JobsStoreResult<Vec<String>> {
        let connection = self.open_connection()?;
        let mut statement = connection.prepare(
            "select id
               from jobs
              where progress_side_effects_pending = 1
                and status in ('completed', 'failed', 'canceled', 'interrupted')
                and progress_side_effects_retry_at <= ?1
              order by progress_side_effects_retry_at asc, updated_at asc, id asc
              limit ?2",
        )?;
        let ids = statement
            .query_map(
                params![as_of, i64::try_from(limit).unwrap_or(i64::MAX)],
                |row| row.get(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Persist a bounded retry after one terminal side-effect attempt fails.
    /// Moving the row's due time forward rotates it out of the current batch,
    /// while the durable count makes the exponential backoff survive restart.
    pub fn defer_pending_terminal_progress_side_effects(
        &self,
        job_id: &str,
    ) -> JobsStoreResult<bool> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let retry_count = transaction
            .query_row(
                "select progress_side_effects_retry_count
                   from jobs
                  where id = ?1 and progress_side_effects_pending = 1",
                params![job_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        let Some(retry_count) = retry_count else {
            transaction.commit()?;
            return Ok(false);
        };
        let next_retry_count = retry_count.max(0).saturating_add(1);
        let exponent = u32::try_from(next_retry_count.saturating_sub(1))
            .unwrap_or(u32::MAX)
            .min(16);
        let delay = PROGRESS_SIDE_EFFECT_RETRY_BASE_SECONDS
            .saturating_mul(1_i64.checked_shl(exponent).unwrap_or(i64::MAX))
            .min(PROGRESS_SIDE_EFFECT_RETRY_MAX_SECONDS);
        let retry_at = now_unix_seconds().saturating_add(delay);
        let changed = transaction.execute(
            "update jobs
                set progress_side_effects_retry_count = ?1,
                    progress_side_effects_retry_at = ?2
              where id = ?3 and progress_side_effects_pending = 1",
            params![next_retry_count, retry_at, job_id],
        )?;
        transaction.commit()?;
        Ok(changed == 1)
    }

    /// Replace an accepted progress report's result with its post-acceptance
    /// augmentation (catalog registration / persisted asset sidecars). The
    /// compare-and-swap guards prevent a slower request from overwriting a
    /// competing status or result update that committed in the meantime.
    pub fn replace_job_result_after_progress(
        &self,
        job_id: &str,
        worker_id: Option<&str>,
        status: JobStatus,
        expected_result: &Map<String, Value>,
        result: &Map<String, Value>,
        clear_terminal_side_effects: bool,
    ) -> JobsStoreResult<JobSnapshot> {
        let mut guard = self.lock.lock();
        let connection = self.write_connection(&mut guard)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = self.get_job_on_connection(&transaction, job_id)?;
        if current.worker_id.as_deref() != worker_id
            || current.status != status
            || &current.result != expected_result
        {
            return Ok(current);
        }
        transaction.execute(
            "update jobs
                set result_json = ?1,
                    updated_at = ?2,
                    progress_side_effects_pending =
                      case when ?3 then 0 else progress_side_effects_pending end,
                    progress_side_effects_retry_count =
                      case when ?3 then 0 else progress_side_effects_retry_count end,
                    progress_side_effects_retry_at =
                      case when ?3 then 0 else progress_side_effects_retry_at end
              where id = ?4",
            params![
                dumps(result)?,
                utc_now(),
                clear_terminal_side_effects,
                job_id
            ],
        )?;
        let job = self.get_job_on_connection(&transaction, job_id)?;
        transaction.commit()?;
        Ok(job)
    }

    fn progress_side_effects_pending_on_connection(
        &self,
        connection: &Connection,
        job_id: &str,
    ) -> JobsStoreResult<bool> {
        connection
            .query_row(
                "select progress_side_effects_pending from jobs where id = ?1",
                params![job_id],
                |row| row.get::<_, bool>(0),
            )
            .optional()?
            .ok_or_else(|| JobsStoreError::NotFound(job_id.to_owned()))
    }

    pub fn list_workers(&self) -> JobsStoreResult<Vec<WorkerSnapshot>> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.open_connection()?;
        let mut statement = connection.prepare("select * from workers order by gpu_id, id")?;
        let workers = collect_workers(statement.query_map([], row_to_worker)?)?;
        Ok(workers)
    }

    pub fn get_worker(&self, worker_id: &str) -> JobsStoreResult<WorkerSnapshot> {
        // Read-only single-SELECT: no write mutex, relies on WAL reader isolation
        // (sc-8950 / F-148 — see list_jobs for the full rationale).
        let connection = self.open_connection()?;
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
        let connection = self.open_connection()?;

        // Per-status counts over the WHOLE table — never a capped/newest-N sample.
        // Filtering an already-capped list silently undercounts once a project
        // exceeds the cap (sc-4208 / F-CORE-4). Seed every known status at 0 so
        // the map shape is stable for callers regardless of what rows exist.
        let mut counts = JOB_STATUSES
            .iter()
            .map(|status| (parse_string_enum::<JobStatus>(status), 0u32))
            .collect::<std::collections::BTreeMap<_, _>>();
        // Cleared jobs (sc-12231, issue #1556) are excluded from the operator's
        // status counts too, matching `list_jobs` — a cleared "completed" run must
        // not keep inflating the completed badge after the user tidied the queue.
        let mut statement = connection.prepare(
            "select status, count(*) from jobs where cleared_at is null group by status",
        )?;
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

    /// Borrow the store's single long-lived write connection, lazily opening it
    /// on first use (sc-11202 / F-025). The caller must already hold the write
    /// mutex (`guard` is its contents), so the returned `&mut Connection` is only
    /// ever reachable by one thread at a time. The WAL/busy_timeout/foreign_keys
    /// pragmas are established once, when the connection first opens; subsequent
    /// writes skip the per-op `Connection::open` + pragma round-trip entirely.
    fn write_connection<'a>(
        &self,
        guard: &'a mut Option<Connection>,
    ) -> JobsStoreResult<&'a mut Connection> {
        if guard.is_none() {
            *guard = Some(self.open_connection()?);
        }
        Ok(guard
            .as_mut()
            .expect("write connection was just initialized"))
    }

    fn open_connection(&self) -> JobsStoreResult<Connection> {
        if let Some(parent) = self.db_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&self.db_path)?;
        // Wait (instead of failing instantly) when another connection/process holds the
        // database lock. rusqlite's default busy timeout is 0ms, so any cross-process
        // overlap — e.g. a sidecar restart where the old process hasn't fully released the
        // db, or a concurrent claim/heartbeat — surfaces as `database is locked` and the
        // job loses its claim (the job then remains available for another capable worker).
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
        let automatically_prioritized = job_type_automatically_jumps_queue(&request.job_type);
        connection.execute(
            "
            insert into jobs (
              id, type, status, queue_rank, project_id, project_name, payload_json, result_json,
              requested_gpu, progress, stage, message, attempts, source_job_id,
              duplicate_of_job_id, created_at, updated_at
            ) values (
              ?1, ?2, ?12,
              case when ?13 != 0
                   then (select coalesce(max(queue_rank), 0) + 1 from jobs)
                   else 0 end,
              ?3, ?4, ?5, '{}', ?6, 0, ?12, ?7, ?8, ?9, ?10, ?11, ?11
            )
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
                i64::from(automatically_prioritized),
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

    /// Load a batch of jobs with one JSON-table join, then restore the caller's
    /// id ordering. One JSON bind avoids SQLite's variable limit for large bulk
    /// transitions while retaining the single-query behavior.
    fn jobs_by_ids(
        &self,
        connection: &Connection,
        job_ids: &[String],
    ) -> JobsStoreResult<Vec<JobSnapshot>> {
        if job_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut statement = connection.prepare(
            "select jobs.* from jobs join json_each(?1) as requested on jobs.id = requested.value",
        )?;
        let ids_json = dumps(job_ids)?;
        let jobs = collect_jobs(statement.query_map(params![ids_json], row_to_job)?)?;
        let mut by_id = jobs
            .into_iter()
            .map(|job| (job.id.clone(), job))
            .collect::<HashMap<_, _>>();
        job_ids
            .iter()
            .map(|job_id| {
                by_id
                    .remove(job_id)
                    .ok_or_else(|| JobsStoreError::NotFound(job_id.clone()))
            })
            .collect()
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
    let revision = row.get::<_, i64>("revision").unwrap_or_default().max(0);
    let queue_rank = row.get::<_, i64>("queue_rank").unwrap_or_default().max(0);
    let mut extra = ExtraFields::default();
    extra.insert("revision".to_owned(), Value::from(revision));
    extra.insert("queueRank".to_owned(), Value::from(queue_rank));
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
        extra,
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
        // `max` is a byte budget. Prompts are arbitrary user text (CJK, emoji,
        // accents), so slicing at a raw byte index can land mid-way through a
        // multi-byte UTF-8 char and panic. Cut at the largest char boundary at
        // or below `max`, keeping the byte-budget intent while never panicking.
        let boundary = (0..=max)
            .rev()
            .find(|&i| prompt.is_char_boundary(i))
            .unwrap_or(0);
        let mut cut = prompt[..boundary].to_owned();
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
        JobType::ControlTraining => {
            let subject = first_str(payload, &["loraName", "outputName", "targetName"])
                .map(str::to_owned)
                .or_else(|| {
                    payload
                        .get("plan")
                        .and_then(|plan| plan.get("output"))
                        .and_then(|output| output.get("loraId"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "(unnamed control branch)".to_owned());
            Some(format!("ControlNet Training — {subject}"))
        }
        JobType::TrainingCaption => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Captioning — {subject}"))
        }
        JobType::DatasetParquetImport => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Parquet Dataset Import - {subject}"))
        }
        JobType::DatasetAnalysis => {
            let subject = first_str(payload, &["datasetName", "datasetId"])
                .unwrap_or("(unnamed dataset)")
                .to_owned();
            Some(format!("Dataset Analysis — {subject}"))
        }
        JobType::CatalogAnalysis => {
            let subject = first_str(payload, &["catalogName", "catalogId"])
                .unwrap_or("(unnamed catalog)")
                .to_owned();
            Some(format!("Catalog Analysis - {subject}"))
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
        status_reason: row.get("status_reason")?,
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

fn dumps<T: serde::Serialize + ?Sized>(value: &T) -> JobsStoreResult<String> {
    // Workspace feature unification can enable serde_json/preserve_order even
    // though this crate does not request it. Sort recursively so the stored
    // bytes stay compatible with Python's sort_keys=True in either build mode.
    let mut value = serde_json::to_value(value)?;
    value.sort_all_objects();
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

fn encoded_worker_has_capability(encoded: &str, expected: &str) -> bool {
    loads_vec::<WorkerCapability>(Some(encoded))
        .iter()
        .any(|capability| capability.as_str() == expected)
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
    // those jobs to a "healthier" non-mlx GPU through this health-based dispatch. The synthetic
    // MPS compatibility descriptor used by legacy tests represents the same physical Apple GPU
    // and would defer the job straight back, deadlocking it in the queue. Production runs the
    // supported job on the native mlx lane. Keeping mlx out of the auto-GPU health comparison
    // breaks the compatibility-path cycle regardless of utilization reporting.
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
/// down), nothing defers; normal capability routing either finds another capable worker or
/// leaves the job queued. An explicit (non-`auto`) GPU choice is never soft-deferred.
fn should_defer_image_to_mlx_worker(
    connection: &Connection,
    job: &JobSnapshot,
    worker: &WorkerSnapshot,
    mlx_required: bool,
) -> JobsStoreResult<bool> {
    if worker.gpu_id.eq_ignore_ascii_case("mlx") || !job_is_mlx_eligible(job) {
        return Ok(false);
    }
    // macOS "MLX-required" (epic 3482 / sc-3483): a synthetic non-mlx descriptor NEVER claims
    // an MLX-eligible job — it yields unconditionally, even when no idle `mlx` worker is
    // ready *right now*. The job waits for the `mlx` worker and, if none takes it within
    // the grace window, `fail_stranded_mlx_jobs` fails it terminal with `mlx_unavailable`
    // rather than entering the legacy MPS compatibility branch. This covers explicit-GPU pins too:
    // production has no MPS fallback.
    if mlx_required {
        return Ok(true);
    }
    // Off (Windows/Linux/Docker, and Mac pre-cutover): defer only an `auto` job to an
    // actually-idle `mlx` worker; otherwise normal capability routing decides. An explicit
    // (non-`auto`) GPU choice is never soft-deferred.
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
/// it. With no mlx worker registered (Windows/Linux, or the mlx worker is down), nothing
/// defers; only a worker advertising a compatible native trainer may claim it. An explicit
/// (non-`auto`) GPU choice is never soft-deferred.
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
/// native Rust captioner (`mlx_gen::load_captioner`) can run them. Without a compatible
/// native captioner, the job remains queued; explicit non-auto requests are not soft-deferred.
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
/// worker, so the in-process `T2iModel` (`vqa` / `interleave_gen`) claims it. Without a compatible
/// native understanding worker, the job remains queued; explicit non-auto requests are not soft-deferred.
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
    // sc-16260: a worker that has declared itself unhealthy — its accelerator is unusable, so
    // every job it claims is one it is certain to fail — is handed nothing at all. This is the
    // BACKSTOP, not the primary gate: the worker also withholds the capabilities it can no
    // longer serve (`with_candle_capabilities`), which is what keeps `image_generate` and
    // friends queued for a host that gets fixed. Both exist because they fail differently — the
    // capability half is what the web's queue explanation and tier pickers already read, while
    // this half holds for any future unhealthy reason whose capability set we don't yet know to
    // trim. Refuses EVERY job type, not just GPU ones: "I cannot run work" is the whole claim
    // the status makes, and the CPU utility lane is a separate worker.
    if worker.status == WorkerStatus::Unhealthy {
        return false;
    }
    if job_requires_gpu(&job.job_type) && worker.gpu_id.eq_ignore_ascii_case("cpu") {
        return false;
    }
    // Epic 3039 (sc-3049): a native-only training kernel (the retired Python MLX LTX trainer)
    // runs only on a Rust worker — a non-mlx worker must refuse it
    // (leaving it queued for the mlx worker) instead of claiming it and failing. The
    // exception (sc-8614): `krea_lora` has a candle trainer, so a
    // candle worker it is candle-eligible for must NOT be refused here (the candle training
    // gate below admits it); any non-candle, non-mlx worker still defers.
    if !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && training_kernel_is_mlx_only(job)
        && !(worker_is_candle(worker) && training_job_is_candle_eligible(job))
    {
        return false;
    }
    // Epic 3018/3041 + sc-3036: the in-process MLX worker (gpu_id "mlx") serves a fixed
    // set of model families. It must not claim an unsupported family or request shape; those
    // remain queued unless another capable native worker registers. Non-mlx workers are
    // unaffected here; the *preference* to route
    // eligible jobs to an idle mlx worker is a soft deferral in the claim path.
    if worker.gpu_id.eq_ignore_ascii_case("mlx") {
        // Image: sc-3026 txt2img/LoRA + sc-3060 reference/edit/inpaint/outpaint +
        // image_detail + sc-3513 the `image_edit` job type (plain Image Edit). A
        // non-MLX edit model (kolors/lens/pulid) is not MLX-eligible, so the mlx
        // worker refuses it. (z_image_edit was ported to MLX,
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
        // (no IC-LoRA keyframe-append path), LoKr-on-Wan — is refused by this worker.
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
        // (sidecar, no mlx-gen crate) and LoKr-on-Wan are refused by this worker.
        // Applies to both dry-run and real runs.
        if matches!(job.job_type, JobType::LoraTrain | JobType::ControlTraining)
            && !training_job_is_mlx_eligible(job)
        {
            // ControlNet studio jobs (epic 10159) are candle-only today (no MLX control trainer — that
            // is B5/sc-10177), so `training_job_is_mlx_eligible` returns false for them and the mlx
            // worker refuses to claim, leaving the job for a candle worker.
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
        // Rust path, so the mlx worker refuses it and it remains queued.
        if matches!(job.job_type, JobType::ImageUpscale) && !upscale_job_is_mlx_eligible(job) {
            return false;
        }
        // Video upscale (epic 4811 / sc-4816): the MLX worker runs the native SeedVR2 engine
        // (`mlx-gen-seedvr2`). Any non-SeedVR2 engine is refused; Candle owns the same SeedVR2-only
        // contract off-Mac.
        if matches!(job.job_type, JobType::VideoUpscale) && !video_upscale_job_is_mlx_eligible(job)
        {
            return false;
        }
        // SenseNova-U1 understanding (sc-3905): the mlx worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (the sole in-process understanding
        // path). A non-SenseNova understanding job is not MLX-eligible, so the mlx worker
        // refuses it and it remains queued.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
    }
    // No-silent-T2I / no-fallback (sc-5968, epic 5483): any non-candle, non-mlx GPU descriptor must
    // DECLINE the unsupported-pose shapes the candle worker owns-to-reject (an `advanced.poses` job
    // on a candle model with no pose lane, e.g. sdxl), so no generic claimant can silently render an
    // unconditioned T2I image and the candle worker reliably wins
    // them (then rejects with a typed error). Mac is unaffected: those shapes are MLX-served there
    // (model_mac_support pose), so the `mlx` worker still claims them and other descriptors decline.
    if !worker_is_candle(worker)
        && !worker.gpu_id.eq_ignore_ascii_case("mlx")
        && image_job_candle_pose_reject(job)
    {
        return false;
    }
    // Candle (Windows/CUDA) lane (epic 3672 image sc-3678; epic 5095 image families sc-5096 + video
    // sc-5097): the candle worker advertises broad image/video job capabilities, then the route
    // predicates narrow them to concrete per-family base, conditioned, adapter, and control lanes.
    // It must refuse every model/request-shape/tier/adapter combination those predicates do not own,
    // so unsupported work remains queued unless another capable native worker registers. Identified
    // by the `candle` marker capability (not `gpu_id`, which is a real CUDA index here). When candle
    // is disabled the marker is absent and this is inert.
    if worker_is_candle(worker) {
        // ImageGenerate + ImageEdit: claim the candle-served shapes (incl. the sc-5487
        // SdxlEdit/Flux2Edit/QwenEdit `image_edit` lanes) AND the unsupported-pose shapes the candle
        // worker must OWN to reject (a `advanced.poses` job on a candle model with no pose lane, e.g.
        // sdxl) — so those fail loudly on candle instead of silently rendering an unconditioned T2I
        // image (sc-5968, the no-fallback / no-silent-T2I directive). Every other unsupported shape is
        // declined and remains queued. `image_edit` is gated
        // here too (mirroring the mlx `JobType::ImageGenerate | JobType::ImageEdit` claim arm): without
        // it an unsupported edit model would be claimed by candle and fail instead of remaining queued.
        if matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit)
            && !(image_job_is_candle_eligible(job) || image_job_candle_pose_reject(job))
        {
            return false;
        }
        if matches!(job.job_type, JobType::ImageDetail) && !image_detail_native_eligible(job) {
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
        // Training (sc-7817 / sc-13870): the candle worker trains only candle-native families
        // (sdxl / z_image / lens / Krea / LTX / the Wan A14B T2V MoE) via
        // `gen_core::load_trainer`. Everything else — Kolors, the dense Wan 5B, Wan I2V A14B — remains
        // queued. WITHOUT this gate the candle worker would claim a real
        // training job it can't execute (the `lora_train_execute` advertisement is coarse — it lights
        // up whenever ANY candle trainer is registered) and fail it terminally instead of leaving it
        // queued. Applies to both dry-run and real runs; mirrors the mlx training gate above.
        if matches!(job.job_type, JobType::LoraTrain | JobType::ControlTraining)
            && !training_job_is_candle_eligible(job)
        {
            // Same gate for the ControlNet studio job (epic 10159): the candle worker claims it only
            // when its resolved plan's kernel (`krea_control`) has a candle trainer registered;
            // otherwise it stays queued (never mis-claimed and failed terminally).
            return false;
        }
        // Dataset captioning (sc-5098): the candle worker serves only JoyCaption (the candle
        // captioner provider). A non-`joy_caption` caption job is refused and remains queued.
        // Eligibility is backend-neutral (captioner == joy_caption), so reuse the mlx gate.
        if matches!(job.job_type, JobType::TrainingCaption) && !caption_job_is_mlx_eligible(job) {
            return false;
        }
        // SenseNova-U1 understanding (sc-5501): the candle worker serves `image_vqa` /
        // `image_interleave` only for the SenseNova-U1 ids (via the concrete candle `T2iModel::{vqa,
        // interleave_gen}` — the off-Mac sibling of the MLX understanding path). Eligibility is
        // backend-neutral (the model is SenseNova-U1), so reuse the understanding gate; a
        // non-SenseNova understanding job is refused and remains queued.
        if matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave)
            && !understanding_job_is_mlx_eligible(job)
        {
            return false;
        }
        // Image upscale (sc-5928 SeedVR2 + sc-5499 Real-ESRGAN, epic 4811 / epic 5482): the candle
        // worker serves Real-ESRGAN (`ort`/CUDA, sc-5499) AND SeedVR2 (`candle-gen-seedvr2`, the
        // Windows/CUDA sibling of mlx-gen-seedvr2). Only `aura-sr` has no candle path, so it is
        // refused and remains queued.
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
    // SeedVR2 upscaling runs on the native MLX worker (Mac) or the candle worker (Windows/Linux).
    // Any generic GPU/CPU descriptor (neither `mlx` nor candle) must refuse a
    // SeedVR2 `image_upscale` job so it stays queued for the mlx/candle worker instead of being
    // claimed and failing with "no generator registered". This is the inverse of the AuraSR gate
    // (unsupported by native lanes → mlx/candle refuse it). `video_upscale` is candle/mlx-only by
    // capability, so it needs no extra generic-worker guard here.
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
    // failing terminally after a worker without a matching native engine claims it.
    if is_real_training_job(job) {
        return advertises(WorkerCapability::LoraTrainExecute.as_str());
    }
    true
}

/// True when a job is a real (non-dry-run) training run — it needs the execute capability
/// ([`WorkerCapability::LoraTrainExecute`]), not just the base plan-validation capability. A
/// `lora_train` payload defaults to dry-run so only an explicit `dryRun: false` counts; a
/// `control_training` studio job (epic 10159) has no dry-run mode — it always renders + trains — so it
/// is always a real run.
fn is_real_training_job(job: &JobSnapshot) -> bool {
    match job.job_type {
        JobType::ControlTraining => true,
        JobType::LoraTrain => job.payload.get("dryRun").and_then(Value::as_bool) == Some(false),
        _ => false,
    }
}

/// The worker capability a job requires. Person detection/tracking default to
/// the real, model-backed capability served by a native GPU worker; an
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
        // The ControlNet studio job (epic 10159) trains through the same executor as `lora_train` and
        // is served by any worker that advertises the training capability — it has no dedicated
        // capability of its own (the real-run gate below additionally requires `lora_train_execute`).
        JobType::ControlTraining => WorkerCapability::LoraTrain.as_str(),
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
    // The SQL input is ordered by durable queue rank first. Restrict the existing explicit-GPU /
    // warm-model optimization to that highest compatible tier so affinity can optimize peers but
    // can never leapfrog an automatically or manually prioritized job.
    let highest_rank = job_queue_rank(first);
    let compatible = compatible
        .into_iter()
        .take_while(|job| job_queue_rank(job) == highest_rank)
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

fn job_queue_rank(job: &JobSnapshot) -> i64 {
    job.extra
        .get("queueRank")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        .max(0)
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

/// Queue policy for short, latency-sensitive work that should run before the normal FIFO lane.
/// Keeping it as a typed predicate makes adding another automatic priority job an explicit review
/// of the job inventory instead of scattering string comparisons through SQL and API handlers.
fn job_type_automatically_jumps_queue(job_type: &JobType) -> bool {
    matches!(job_type, JobType::PromptRefine)
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
            | JobType::ControlTraining
            | JobType::TrainingCaption
            | JobType::DatasetAnalysis
            | JobType::CatalogAnalysis
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

#[cfg(test)]
mod tests;
