//! Process-lifecycle surface of the rust-api crate: the `Settings` config
//! struct, the shared `AppState` handed to every axum handler, and the two
//! binary entrypoints (`run` for the HTTP API, `run_worker` for the standalone
//! GPU-worker mode).
//!
//! Extracted from the crate-root grab-bag (sc-9736, the deferred remainder of
//! sc-8890 / F-088). Behavior-preserving: `lib.rs` re-exports these four items
//! (`pub use server::{run, run_worker, AppState, Settings}`) so every external
//! reference — `main.rs`, the handler modules'
//! `use super::*`, and `tests.rs`'s `use super::{Settings, ...}` — keeps
//! resolving unchanged. The startup helpers (`create_app`,
//! `spawn_inprocess_utility_worker`, `shutdown_signal`, `should_warn_open_bind`,
//! `open_bind_override_enabled`, `parent_death`, `parent_pid_to_watch`,
//! `env_string`, `env_path_or`) and the config constants stay in `lib.rs`; this
//! module reaches them through `crate::`.

use std::collections::HashMap;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tower::ServiceExt;

use sceneworks_core::jobs_store::JobsStore;
use sceneworks_core::project_store::ProjectStore;

use crate::auth::{cors_layer, AuthThrottle};
use crate::catalog_scan_supervisor::CatalogScanSupervisor;
use crate::events::EventHub;
use crate::external_base_models::ExternalBaseModelCache;
use crate::external_loras::ExternalLoraCache;
use crate::manifest::ManifestCache;
use crate::models::{ModelCatalogCache, ModelSizeCache};
use crate::startup::{
    StartupCriticality, StartupMaintenance, StartupPhaseTimer, BACKGROUND_MAINTENANCE_TIMEOUT,
};
use crate::tickets::TicketStore;
use crate::{
    create_app_with_pending_startup_maintenance, env_path_or, env_string,
    open_bind_override_enabled, parent_death, parent_pid_to_watch, seed_mode_for_config_dir,
    should_warn_open_bind, shutdown_signal, spawn_inprocess_utility_worker, DEFAULT_API_HOST,
    DEFAULT_CORS_ORIGINS,
};

#[derive(Debug, Clone)]
pub struct Settings {
    pub app_version: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    /// Finite, disabled-by-default app-owned resolved-model cache policy shared with workers.
    pub resolved_cache: sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy,
    pub config_dir: PathBuf,
    /// Directory holding the `credentials.json` store (sc-16540). Resolved by
    /// [`sceneworks_core::credentials::credentials_dir`] and deliberately independent
    /// of `config_dir`, which is pointed at the repo checkout in dev — see that
    /// function for the precedence and why a plaintext token must not live there.
    pub credentials_dir: PathBuf,
    pub access_token: String,
    pub cors_origins: Vec<String>,
    pub worker_timeout_seconds: u64,
    pub jobs_db_path: PathBuf,
    /// Terminal job history retention. Zero preserves all history.
    pub jobs_retention_days: u32,
    pub run_utility_inprocess: bool,
    /// Epic 3482 — macOS "MLX-required" mode. When set (the desktop sets it on macOS,
    /// where it spawns the in-process `mlx` worker), any non-MLX GPU descriptor never claims an
    /// MLX-eligible job: it defers unconditionally to the `mlx` worker, and a job no live
    /// `mlx` worker takes within the grace window fails terminal with `mlx_unavailable`
    /// instead of remaining queued indefinitely (sc-3483). Absent on Windows/Linux/Docker
    /// (no `mlx` worker). Ships default OFF (observe); the
    /// final cutover (sc-3492) flips it on for the packaged Mac build.
    pub mlx_required: bool,
    /// Epic 3482 / sc-3484 — when MLX-required, what to do with a job the Rust/MLX flow can't
    /// run (`mac_rust_supported` returns `Err`). **false = warn-only** (default): log a
    /// structured `mlx_unsupported` gap event at claim time but still run the job on the
    /// normal capability routing; without a capable native worker the job remains queued. Flipping
    /// `mlx_required` on for observation materializes the gap list. **true = enforce**: fail the job terminal with
    /// `mlx_unsupported`. Read from `SCENEWORKS_MLX_UNSUPPORTED_MODE` (`enforce` vs anything
    /// else). Irrelevant unless `mlx_required`.
    pub mlx_enforce_unsupported: bool,
    /// Epic 5483 (sc-5502) — the off-Mac (Windows/Linux/Docker) twin of `mlx_required`. When set,
    /// the candle (CUDA) worker is the only GPU backend: a candle-eligible job no live candle
    /// worker takes within the grace window fails terminal with `candle_unavailable` instead of
    /// waiting forever, and (under `candle_enforce_unsupported`) a job the candle/CUDA flow can't
    /// serve (`candle_supported` returns `Err`) fails with `candle_unsupported` instead of remaining
    /// queued without a capable native worker. Ships default OFF; flip it on per-deployment when
    /// terminal gap reporting is desired. Read from
    /// `SCENEWORKS_CANDLE_REQUIRED`. Absent on macOS (the `mlx_required` path governs there).
    pub candle_required: bool,
    /// Epic 5483 (sc-5502) — the candle twin of `mlx_enforce_unsupported`. **false = warn-only**
    /// (default): log a structured `candle_unsupported` gap event if an unsupported job is claimed;
    /// otherwise normal capability routing leaves it queued. Flipping `candle_required` on for
    /// observation materializes the off-Mac gap list. **true = enforce**: fail the
    /// job terminal with `candle_unsupported`. Read from `SCENEWORKS_CANDLE_UNSUPPORTED_MODE`
    /// (`enforce` vs anything else). Irrelevant unless `candle_required`.
    pub candle_enforce_unsupported: bool,
    /// Epic 4484 — trust loopback peers to bypass the access token. When LAN remote
    /// access binds `0.0.0.0` with the password as `access_token`, the embedded desktop
    /// UI and the local GPU worker(s) still reach the API over loopback with no password;
    /// trusting `127.0.0.1`/`::1` peers keeps local use password-free while LAN callers
    /// (other source IPs) stay gated. The desktop sets `SCENEWORKS_TRUST_LOOPBACK`;
    /// Docker/server never does, so a reverse-proxied deployment stays fail-closed.
    pub trust_loopback: bool,
    /// sc-19570 — the API host's OS (`std::env::consts::OS` in production), threaded through
    /// `Settings` rather than read at each use site so the OFF-MAC branch of the per-mode
    /// reachability sweep can be exercised on a Mac. macOS structurally cannot detect that class of
    /// defect by running on itself: the sweep fails only what no Windows/Linux lane will claim, and
    /// on a Mac that branch never executes, so an untestable `std::env::consts::OS` call inside
    /// `JobsStore::fail_platform_unreachable_jobs`'s callers would ship a guard whose only proof is
    /// its own doc comment.
    ///
    /// It decides a job's EXECUTION outcome, never an HTTP status code: `POST /api/v1/video/jobs`
    /// answers `201` for the same body on every platform, and this field only determines whether
    /// the job it creates is `queued` or terminal `failed`. A read of this field that changes a
    /// response code is a bug — that shape was ruled out.
    ///
    /// Values are the `std::env::consts::OS` vocabulary (`"macos"` / `"windows"` / `"linux"`).
    pub host_os: String,
    /// Epic 10231 (sc-10233) — base URL the embedded MCP server's thin API client
    /// uses to call back into this API's `/api/v1/*` routes (the MCP tools are a
    /// thin HTTP client over the existing surface, mirroring the Rust worker — no
    /// direct engine/DB path). Read from `SCENEWORKS_API_URL` (the same variable
    /// the worker uses); defaults to loopback on this API's own port. The client
    /// sends `access_token` as `X-SceneWorks-Token`, so the self-calls pass the
    /// access-control gate whether or not loopback trust is enabled.
    pub mcp_api_url: String,
    /// Epic 10231 (sc-10277) — poll cadence for the MCP blocking job tools
    /// (`generate_image`): how often they re-poll `GET /api/v1/jobs/:id` while
    /// waiting for a terminal status. Read from `SCENEWORKS_MCP_JOB_POLL_INTERVAL`
    /// (whole seconds); defaults to the `JobWaitConfig` default. Clamped non-zero
    /// at the mount point.
    pub mcp_job_poll_interval: Duration,
    /// Epic 10231 (sc-10277) — overall deadline for the MCP blocking job tools:
    /// after this long without a terminal status the tool returns a clear timeout
    /// error (the job itself is NOT canceled). Read from
    /// `SCENEWORKS_MCP_JOB_TIMEOUT` (whole seconds); defaults to the
    /// `JobWaitConfig` default (30 min). Clamped to ≥ the poll interval.
    pub mcp_job_timeout: Duration,
    /// Epic 10451 (sc-10452) — operator-configured directories outside the app data dir
    /// that hold model weights the user already has, typically a ComfyUI `models/` tree.
    /// The API scans each root's `loras/` subdirectory to surface those adapters in the
    /// LoRA catalog (read in place, never copied); the worker holds the same list and
    /// admits it in its adapter-path confinement. Read from
    /// `SCENEWORKS_EXTERNAL_MODEL_ROOTS`; empty (feature off) by default and always empty
    /// on macOS — see `sceneworks_core::external_roots`.
    pub external_model_roots: Vec<PathBuf>,
    /// F-040 (sc-11236) — operator-declared extra Host-header authorities the
    /// `/mcp` DNS-rebinding defense accepts, on top of the always-allowed loopback
    /// set. Read from `SCENEWORKS_MCP_ALLOWED_HOSTS` (comma-separated `host` or
    /// `host:port` entries). Only needed for the wildcard LAN bind
    /// (`SCENEWORKS_API_HOST=0.0.0.0`), whose reachable interface addresses can't
    /// be enumerated: set it to the machine's LAN hostname/IP(s) to keep the
    /// Host check ON for remote clients. Empty by default; irrelevant for a
    /// loopback bind (loopback is always allowed) — see
    /// [`sceneworks_mcp::mcp_allowed_hosts`].
    pub mcp_allowed_hosts_extra: Vec<String>,
}

impl Settings {
    pub fn from_env() -> Self {
        let defaults = sceneworks_core::app_paths::AppPaths::platform_default();
        let data_dir = env_path_or("SCENEWORKS_DATA_DIR", &defaults.data_dir);
        let jobs_db_path = std::env::var("SCENEWORKS_JOBS_DB_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("cache").join("jobs.db"));
        let port: u16 = std::env::var("SCENEWORKS_API_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8000);
        let host = env_string("SCENEWORKS_API_HOST", DEFAULT_API_HOST);
        let config_dir = env_path_or("SCENEWORKS_CONFIG_DIR", &defaults.config_dir);
        let credentials_dir = sceneworks_core::credentials::credentials_dir(&config_dir);
        Self {
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.2.0"),
            host: host.clone(),
            port,
            data_dir,
            resolved_cache:
                sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy::from_env_or_safe_default(),
            config_dir,
            credentials_dir,
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
            jobs_retention_days: std::env::var("SCENEWORKS_JOBS_RETENTION_DAYS")
                .ok()
                .and_then(|value| value.trim().parse().ok())
                .unwrap_or(90),
            run_utility_inprocess: std::env::var("SCENEWORKS_RUN_UTILITY_INPROCESS")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
            mlx_required: std::env::var("SCENEWORKS_MLX_REQUIRED")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
            mlx_enforce_unsupported: std::env::var("SCENEWORKS_MLX_UNSUPPORTED_MODE")
                .map(|value| value.trim().eq_ignore_ascii_case("enforce"))
                .unwrap_or(false),
            candle_required: std::env::var("SCENEWORKS_CANDLE_REQUIRED")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
            candle_enforce_unsupported: std::env::var("SCENEWORKS_CANDLE_UNSUPPORTED_MODE")
                .map(|value| value.trim().eq_ignore_ascii_case("enforce"))
                .unwrap_or(false),
            trust_loopback: std::env::var("SCENEWORKS_TRUST_LOOPBACK")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
            // Always the real host OS in production — never an env override. The field exists to
            // be substitutable in a test, not configurable in a deployment: a deployment that
            // could claim to be Windows would re-open the exact hang sc-19570 closes.
            host_os: std::env::consts::OS.to_owned(),
            // The MCP self-calls originate on this machine; derive a base URL that
            // resolves to an interface this API actually binds (sc-10260).
            // `SCENEWORKS_API_URL` still overrides for reverse-proxy/container setups.
            mcp_api_url: env_string("SCENEWORKS_API_URL", &default_mcp_api_url(&host, port)),
            // MCP blocking-job wait knobs (sc-10277); whole seconds, default from
            // JobWaitConfig. Invariants are enforced by JobWaitConfig::clamped at
            // the mount point, so an out-of-range value here can't misbehave.
            mcp_job_poll_interval: env_secs(
                "SCENEWORKS_MCP_JOB_POLL_INTERVAL",
                sceneworks_mcp::JobWaitConfig::default().poll_interval,
            ),
            mcp_job_timeout: env_secs(
                "SCENEWORKS_MCP_JOB_TIMEOUT",
                sceneworks_mcp::JobWaitConfig::default().timeout,
            ),
            external_model_roots: sceneworks_core::external_roots::external_model_roots_from_env(),
            // F-040: extra Host authorities for the /mcp DNS-rebinding allow-list.
            mcp_allowed_hosts_extra: env_string("SCENEWORKS_MCP_ALLOWED_HOSTS", "")
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }
}

/// The self-call base URL the embedded MCP client uses when `SCENEWORKS_API_URL`
/// is unset (sc-10260). The MCP tools call back into THIS process's `/api/v1/*`,
/// so the URL must resolve to an interface this API is actually listening on:
///   - a wildcard bind (`0.0.0.0`/`::`) or an explicit loopback host → dial
///     loopback (`127.0.0.1`), which a wildcard socket also accepts;
///   - a specific interface IP (e.g. `192.168.4.97`) → dial THAT host, because a
///     socket bound only to it never accepts a `127.0.0.1` connection (the old
///     hardcoded loopback default failed every catalog tool call in that setup).
fn default_mcp_api_url(host: &str, port: u16) -> String {
    let host = host.trim();
    let is_wildcard_or_loopback = host.is_empty()
        || host == "0.0.0.0"
        || host == "::"
        || host == "[::]"
        || host == "127.0.0.1"
        || host == "::1"
        || host == "[::1]"
        || host.eq_ignore_ascii_case("localhost");
    let dial_host = if is_wildcard_or_loopback {
        "127.0.0.1".to_owned()
    } else if host.contains(':') && !host.starts_with('[') {
        // Bare IPv6 literal (e.g. `fe80::1`) needs bracketing in a URL authority.
        format!("[{host}]")
    } else {
        host.to_owned()
    };
    format!("http://{dial_host}:{port}")
}

/// Parse a whole-seconds env var into a [`Duration`], keeping `default` when the
/// variable is unset or not a non-negative integer (sc-10277).
fn env_secs(name: &str, default: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(default)
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) settings: Settings,
    pub(crate) jobs_store: Arc<JobsStore>,
    pub(crate) project_store: Arc<ProjectStore>,
    pub(crate) events: Arc<EventHub>,
    /// Serializes authoritative SSE snapshots with queue publications. A
    /// reconnect subscribes while holding this lock, so a queue mutation is
    /// represented either in the initial snapshot or by a later live event,
    /// never by a buffered event that can be replayed behind a newer snapshot.
    pub(crate) queue_snapshot_lock: Arc<AsyncMutex<()>>,
    pub(crate) event_tickets: Arc<TicketStore>,
    pub(crate) media_tickets: Arc<TicketStore>,
    /// Bounds CPU and memory consumed by on-demand Library thumbnail backfills.
    pub(crate) thumbnail_generation_slots: Arc<Semaphore>,
    /// Bounds memory consumed by `?stripWorkflow=true` downloads (sc-15953).
    ///
    /// The sibling of `thumbnail_generation_slots`, deliberately: they are the two derived
    /// representations of the same route, and both read a whole file into memory. A strip holds
    /// the source buffer for as long as the response body is being written, so without a gate the
    /// ceiling is "however many clients ask at once" times the file size.
    pub(crate) workflow_strip_slots: Arc<Semaphore>,
    // sc-8870 (F-068): per-peer-IP failed-token throttle for the auth oracle.
    pub(crate) auth_throttle: Arc<AuthThrottle>,
    /// The ONE resolved-model cache session this API process holds (sc-19711).
    ///
    /// `ResolvedCacheStore::open` takes a session slot — a `sessions/<id>/` directory plus its own
    /// exclusive lock file — and the store's design is that a slot belongs to a PROCESS for that
    /// process's lifetime, with dead processes' slots reclaimed by a later `recover()`. Opening a
    /// store per HTTP request would therefore leave a fresh, permanently-held slot behind on every
    /// pin, preview and removal the UI performs, and nothing would clear them until a worker next
    /// recovered. The handle is created lazily on the first mutating request so that a deployment
    /// which only ever reads status never materializes the cache root as a side effect.
    pub(crate) resolved_cache_session: Arc<
        AsyncMutex<Option<sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore>>,
    >,
    pub(crate) manifest_cache: Arc<Mutex<ManifestCache>>,
    pub(crate) manifest_write_locks: Arc<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>>,
    /// One generation-keyed install-state snapshot shared by `/models`, recipe
    /// preset reads, and request-scoped job validation. The cache serializes a
    /// cold build so concurrent callers join the same filesystem sweep; model
    /// lifecycle completion and model-manifest writes advance its generation.
    pub(crate) model_catalog_cache: Arc<ModelCatalogCache>,
    pub(crate) model_size_cache: Arc<Mutex<ModelSizeCache>>,
    #[cfg(test)]
    pub(crate) model_size_estimate_test_hook:
        Arc<Mutex<Option<crate::models::ModelSizeEstimateTestHook>>>,
    #[cfg(test)]
    pub(crate) model_size_estimate_disabled_override: Arc<Mutex<Option<bool>>>,
    #[cfg(test)]
    pub(crate) video_platform_override: Arc<Mutex<Option<&'static str>>>,
    /// sc-10452 — memoized family detection for adapters scanned out of the operator's
    /// external model roots, keyed by each file's size + mtime. `lora_catalog` runs on
    /// every job-create; without this, every generation would re-parse every ComfyUI
    /// adapter's safetensors header.
    pub(crate) external_lora_cache: Arc<Mutex<ExternalLoraCache>>,
    /// sc-10667 — memoized base-weight detection for the transformer/encoder/VAE
    /// files scanned out of the external roots' base subtrees, keyed by size +
    /// mtime. `model_catalog` runs on every job-create; without this, every
    /// generation would re-parse every ComfyUI base file's safetensors header.
    pub(crate) external_base_model_cache: Arc<Mutex<ExternalBaseModelCache>>,
    pub(crate) http_client: reqwest::Client,
    pub(crate) interrupted_jobs_on_startup: usize,
    /// Safe, path-free status for bounded stale-upload cleanup that starts
    /// after the listener is bound and readiness-critical initialization ends.
    pub(crate) startup_maintenance: StartupMaintenance,
    /// Serializes the durable terminal-progress handoff from the jobs DB to
    /// idempotent catalog/project writes. A retry re-checks the DB after taking
    /// this lock, so two overlapping terminal reports cannot duplicate work.
    pub(crate) progress_side_effects_lock: Arc<AsyncMutex<()>>,
    /// Owns, deduplicates, cancels, and drains background catalog scans.
    pub(crate) catalog_scan_supervisor: Arc<CatalogScanSupervisor>,
    /// Catalog ids for which this AppState has already published an invalid
    /// persisted recovery plan. Valid scheduled recoveries remain retryable if
    /// their generation later exits incomplete.
    pub(crate) catalog_scan_invalid_recovery_reported:
        Arc<AsyncMutex<std::collections::HashSet<String>>>,
    /// Caps filesystem/Parquet metadata validation before it reaches Tokio's
    /// shared blocking pool, preserving capacity for catalog DB operations.
    pub(crate) catalog_scan_preflight_slots: Arc<Semaphore>,
    /// Separately caps long-running scanner startup and row passes. Tasks wait
    /// asynchronously before entering Tokio's shared blocking pool, reserving
    /// capacity for status/query database work even when many catalogs run.
    pub(crate) catalog_scan_work_slots: Arc<Semaphore>,
    /// Deterministic catalog scheduler seams. Production has no hook fields or
    /// branch overhead.
    #[cfg(test)]
    pub(crate) catalog_scan_before_driver_start_once: Arc<Mutex<Option<Arc<tokio::sync::Barrier>>>>,
    #[cfg(test)]
    pub(crate) catalog_scan_stop_after_pass_once: Arc<AtomicBool>,
    #[cfg(test)]
    pub(crate) catalog_scan_before_terminal_exit_once: Arc<AtomicBool>,
    #[cfg(test)]
    pub(crate) catalog_scan_terminal_exit_reached: Arc<tokio::sync::Notify>,
    #[cfg(test)]
    pub(crate) catalog_scan_terminal_exit_release: Arc<tokio::sync::Notify>,
    #[cfg(test)]
    pub(crate) catalog_scan_preflight_delay_ms_once: Arc<AtomicU64>,
    #[cfg(test)]
    pub(crate) catalog_scan_preflight_started: Arc<tokio::sync::Notify>,
    #[cfg(test)]
    pub(crate) catalog_scan_preflight_admission_timeout_ms: Arc<AtomicU64>,
    #[cfg(test)]
    pub(crate) catalog_scan_preflight_execution_timeout_ms: Arc<AtomicU64>,
    #[cfg(test)]
    pub(crate) catalog_scan_preflight_test_ticks: Arc<AtomicU64>,
    #[cfg(test)]
    pub(crate) catalog_scan_injected_sqlite_busy_failures: Arc<AtomicU64>,
    #[cfg(test)]
    pub(crate) catalog_scan_contention_backoff_started: Arc<tokio::sync::Notify>,
    /// Deterministic race hook for progress acceptance tests. Production builds
    /// contain no hook or synchronization overhead.
    #[cfg(test)]
    pub(crate) progress_before_accept_once: Arc<Mutex<Option<Arc<tokio::sync::Barrier>>>>,
    /// Deterministic rendezvous after an SSE snapshot is read but before its
    /// live subscription is installed. Production builds contain no hook.
    #[cfg(test)]
    pub(crate) sse_snapshot_before_subscribe_once: Arc<Mutex<Option<Arc<tokio::sync::Barrier>>>>,
    /// One-shot failure after terminal acceptance, used to prove the durable
    /// pending handoff survives an error and resumes on the owner's retry.
    #[cfg(test)]
    pub(crate) progress_side_effects_fail_once: Arc<Mutex<bool>>,
    /// Persistent per-job failures and attempt counts for recovery fairness
    /// tests. Production builds contain neither field.
    #[cfg(test)]
    pub(crate) progress_side_effects_fail_job_ids: Arc<Mutex<std::collections::HashSet<String>>>,
    #[cfg(test)]
    pub(crate) progress_side_effects_attempts: Arc<Mutex<HashMap<String, usize>>>,
}

#[derive(Clone)]
struct BootstrapState {
    settings: Settings,
    ready_app: Arc<OnceLock<Router>>,
    startup_maintenance: StartupMaintenance,
}

/// Self-refreshing holding page served to browser navigations that arrive before
/// readiness-critical startup finishes (sc-15591). The SPA lives at `/` on the
/// ready router, so until that router is installed a navigation lands on the
/// fallback below — and rendering the raw `api_initializing` JSON leaves the tab
/// permanently parked on machine-readable text that never re-navigates itself.
/// The `<meta http-equiv="refresh">` costs nothing while starting and turns into
/// the app on the first poll after `api_ready`.
const INITIALIZING_PAGE: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="refresh" content="2" />
    <title>Starting SceneWorks…</title>
    <style>
      :root { color-scheme: dark; font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }
      html, body { margin: 0; height: 100%; }
      body { display: grid; place-items: center; background: #0f1117; color: #e7e9ee; }
      main { width: min(32rem, 88vw); padding: 2rem; text-align: center; }
      h1 { margin: 0 0 0.5rem; font-size: 1.5rem; letter-spacing: -0.01em; font-weight: 600; }
      p { margin: 0; color: #9aa1ad; font-size: 0.95rem; line-height: 1.5; }
      .spinner {
        width: 1.4rem; height: 1.4rem; margin: 0 auto 1.25rem;
        border: 2px solid #2a2f3a; border-top-color: #6c8cff; border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
    </style>
  </head>
  <body>
    <main>
      <div class="spinner"></div>
      <h1>Starting SceneWorks…</h1>
      <p>Preparing your library. This page loads the app automatically when startup finishes.</p>
    </main>
  </body>
</html>
"#;

/// Whether the caller is a browser navigating to a page rather than an API client
/// reading JSON. Only an explicit `text/html` in `Accept` counts: a `*/*`-only
/// client (curl, the worker, a health probe) keeps the machine-readable 503.
fn accepts_html(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|accept| {
            accept.split(',').any(|entry| {
                entry
                    .split(';')
                    .next()
                    .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/html"))
            })
        })
}

/// `pub(crate)` so the startup tests can drive the pre-readiness surface directly
/// (an empty `ready_app` is exactly the window this router exists to cover).
pub(crate) fn bootstrap_router(
    settings: Settings,
    ready_app: Arc<OnceLock<Router>>,
    startup_maintenance: StartupMaintenance,
) -> Router {
    let cors = cors_layer(&settings);
    Router::new()
        .fallback(bootstrap_dispatch)
        .with_state(BootstrapState {
            settings,
            ready_app,
            startup_maintenance,
        })
        // Keep the browser-visible readiness surface usable while the real
        // application router is still performing readiness-critical startup.
        // The same configured policy is retained after the ready router is
        // installed; duplicate allow-origin insertion is idempotent.
        .layer(cors)
}

async fn bootstrap_dispatch(
    State(state): State<BootstrapState>,
    request: Request<Body>,
) -> Response {
    if let Some(app) = state.ready_app.get() {
        return match app.clone().oneshot(request).await {
            Ok(response) => response,
            Err(error) => match error {},
        };
    }

    match request.uri().path() {
        "/api/v1/health" => {
            let token_configured = !state.settings.access_token.is_empty();
            Json(json!({
                "status": "starting",
                "service": "sceneworks-api",
                "runtime": "rust",
                "version": state.settings.app_version,
                "authRequired": token_configured,
                "directories": if token_configured {
                    None
                } else {
                    Some(json!({
                        "data": state.settings.data_dir.display().to_string(),
                        "config": state.settings.config_dir.display().to_string(),
                        "projects": state.settings.projects_dir().display().to_string(),
                        "jobsDb": state.settings.jobs_db_path.display().to_string(),
                    }))
                },
                "interruptedJobsOnStartup": 0,
                "readiness": {
                    "status": "initializing",
                    "criticality": "readiness-critical",
                },
                "startupMaintenance": state.startup_maintenance.snapshot(),
            }))
            .into_response()
        }
        "/api/v1/access" => Json(json!({
            "authRequired": !state.settings.access_token.is_empty(),
            "tokenHeader": "X-SceneWorks-Token",
        }))
        .into_response(),
        // Everything else — including `/`, where the SPA lives once the ready
        // router is installed. A browser gets the self-refreshing holding page so
        // the tab becomes the app on its own; API clients get the unchanged
        // machine-readable 503 with `retry-after`.
        _ if accepts_html(request.headers()) => (
            StatusCode::SERVICE_UNAVAILABLE,
            // `no-store`: the holding page must never be what a later navigation
            // replays from cache once the app is up. `Html` sets the content type.
            [("retry-after", "1"), ("cache-control", "no-store")],
            Html(INITIALIZING_PAGE),
        )
            .into_response(),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "1")],
            Json(json!({
                "detail": "SceneWorks API readiness-critical startup is still running",
                "code": "api_initializing",
            })),
        )
            .into_response(),
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Install the tracing backbone first so every line below (and every request)
    // flows through the format-adaptive subscriber. The buffer variant also feeds
    // this process's own ring buffer, served by `GET /api/v1/logs` (sc-3453).
    sceneworks_core::observability::init_logging_with_buffer(crate::logs::api_session_log());
    // Host mode (no HF cache env set): default HF_HOME to the shared ~/.cache/
    // huggingface so the catalog and downloads agree on the OS cache rather than
    // the private data dir (sc-1904 follow-up). Desktop/Compose already inject it.
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        tracing::info!(
            event = "hf_home_defaulted",
            home = %home.display(),
            "SceneWorks API defaulting HF_HOME"
        );
    }
    let settings = Settings::from_env();
    // A populated builtin catalog is mandatory — model->file resolution depends on
    // it. The desktop wrapper and the Compose bind mount normally provide it; seed
    // any missing manifests here so launching the API binary directly works too,
    // and fail loudly rather than serving an empty catalog if seeding can't finish.
    //
    // Seed mode by config-dir origin (sc-10212, sc-15504): an EXPLICIT `SCENEWORKS_CONFIG_DIR`
    // marks an operator-owned dir — a repo checkout, a Compose bind mount, or a RunPod
    // persistent volume — so seed `SyncFromEmbedded`: refresh each builtin manifest that
    // is missing or has DRIFTED from the binary's embedded copy, while leaving a
    // byte-identical file untouched so a matching checkout is never dirtied. That stops a
    // directly-launched API binary from serving a STALE seeded catalog after an upgrade —
    // the failure that hid the sc-10193 img2img flag and the Krea Turbo memory-ladder curves
    // (a persisted `builtin.models.jsonc` without `turboFit` made a 24 GB card wrongly reject
    // a q4/1024² render). A deployment that intentionally ships its OWN `builtin.*.jsonc` and
    // wants it used verbatim opts out with a truthy `SCENEWORKS_OWN_MANIFESTS` (→ `IfMissing`:
    // fill gaps only, never self-heal). When `SCENEWORKS_CONFIG_DIR` is unset, `config_dir`
    // is the platform default app-owned dir (the same one the desktop seeds `Overwrite`), so
    // refresh unconditionally on launch. Builtin manifests are app-managed; operator
    // customizations live in the separate `user.*.jsonc` files, which seeding never touches.
    let seed_mode = seed_mode_for_config_dir(
        std::env::var("SCENEWORKS_CONFIG_DIR").ok().as_deref(),
        std::env::var("SCENEWORKS_OWN_MANIFESTS").ok().as_deref(),
    );
    {
        let _phase =
            StartupPhaseTimer::start("builtin_manifest_seed", StartupCriticality::BindCritical);
        if let Err(error) = sceneworks_core::builtin_manifests::seed_builtin_manifests(
            &settings.config_dir,
            seed_mode,
        ) {
            return Err(format!(
                "failed to seed builtin manifests into {}: {error}",
                settings.config_dir.join("manifests").display()
            )
            .into());
        }
    }
    let address: SocketAddr = format!("{}:{}", settings.host, settings.port).parse()?;
    // sc-4201 (F-API-1) / sc-5720 (API-001): a non-loopback bind with no access token
    // serves every endpoint — file reads, credential writes, job creation, large
    // uploads — to the whole network without authentication. The default is loopback;
    // refuse to start on an open bind without a token unless the operator explicitly
    // opts in with SCENEWORKS_ALLOW_OPEN_BIND=1 (then warn loudly instead).
    if should_warn_open_bind(&settings.access_token, address.ip()) {
        let override_raw = std::env::var("SCENEWORKS_ALLOW_OPEN_BIND").unwrap_or_default();
        if open_bind_override_enabled(&override_raw) {
            tracing::warn!(
                event = "open_bind_without_token",
                address = %address,
                "SceneWorks API is binding with no SCENEWORKS_ACCESS_TOKEN set — every endpoint is \
                 reachable without authentication from the network. Proceeding because \
                 SCENEWORKS_ALLOW_OPEN_BIND is set; ensure this host is on a trusted network."
            );
        } else {
            return Err(format!(
                "Refusing to bind to {address} with no SCENEWORKS_ACCESS_TOKEN set: every endpoint \
                 would be reachable without authentication from the network. Set \
                 SCENEWORKS_ACCESS_TOKEN, bind to 127.0.0.1, or set SCENEWORKS_ALLOW_OPEN_BIND=1 to \
                 override (only on a trusted network)."
            )
            .into());
        }
    }
    // The credential store is created 0600 and the API never returns tokens over
    // HTTP, but a sysadmin or restore could leave the on-disk file group/world
    // readable. Warn (don't fail) at startup so the secret's only at-rest
    // protection — its file mode — is visibly broken instead of silently so.
    // Move a pre-sc-16540 store out of the config dir before anything reads it. The
    // API is the file's sole writer, so this is the one place it can happen without a
    // race. A failure here is logged, not fatal: the worker still falls back to the
    // legacy path, so an un-migrated install keeps working.
    match sceneworks_core::credentials::migrate_legacy_store(
        &settings.config_dir,
        &settings.credentials_dir,
    ) {
        Ok(sceneworks_core::credentials::LegacyMigration::Migrated(from)) => {
            tracing::info!(
                event = "credentials_store_migrated",
                from = %from.display(),
                to = %settings.credentials_dir.display(),
                "moved the credential store out of the config dir (sc-16540)",
            );
        }
        Ok(sceneworks_core::credentials::LegacyMigration::BothPresent(legacy)) => {
            tracing::warn!(
                event = "credentials_store_duplicate",
                legacy = %legacy.display(),
                active = %settings.credentials_dir.display(),
                "a credential store exists at BOTH the old and new locations; the new one is \
                 authoritative. Delete the old file — it still holds tokens in plaintext.",
            );
        }
        Ok(sceneworks_core::credentials::LegacyMigration::NotNeeded) => {}
        Err(error) => {
            tracing::warn!(
                event = "credentials_store_migration_failed",
                error = %error,
                "could not migrate the credential store out of the config dir; leaving it in \
                 place (readers fall back to the old path)",
            );
        }
    }
    #[cfg(unix)]
    {
        let creds_path = settings
            .credentials_dir
            .join(sceneworks_core::credentials::CREDENTIALS_FILENAME);
        if let Some(mode) = sceneworks_core::credentials::loose_credentials_mode(&creds_path) {
            tracing::warn!(
                event = "credentials_file_loose_mode",
                path = %creds_path.display(),
                mode = format!("{mode:o}"),
                "credentials file is group/world accessible — it holds download tokens that should \
                 be owner-only. Run `chmod 600` on it to restrict access."
            );
        }
    }
    let run_utility_inprocess = settings.run_utility_inprocess;
    let utility_data_dir = settings.data_dir.clone();
    // Bind before any readiness-critical filesystem/SQLite initialization and
    // before background-safe network-volume sweeps. The socket can accept as
    // soon as the router is ready; stale upload cleanup can never hold the port
    // closed indefinitely.
    let listener = {
        let _phase = StartupPhaseTimer::start("listener_bind", StartupCriticality::BindCritical);
        tokio::net::TcpListener::bind(address).await?
    };
    // Use the actual bound address so port 0 (OS-assigned) is reported and the
    // in-process worker connects to the real port. `api_listening` now means the
    // bootstrap health/access surface is accepting; `api_ready` below means all
    // readiness-critical invariants are complete.
    let bound = listener.local_addr()?;
    let port = bound.port();
    let ready_app = Arc::new(OnceLock::new());
    let bootstrap_maintenance = StartupMaintenance::pending();
    let bootstrap = bootstrap_router(
        settings.clone(),
        ready_app.clone(),
        bootstrap_maintenance.clone(),
    );
    tracing::info!(
        event = "api_listening",
        address = %bound,
        "SceneWorks API bootstrap health/access surface listening"
    );

    // `into_make_service_with_connect_info` exposes the peer `SocketAddr` to
    // both the bootstrap dispatcher and, once installed, the real router's auth
    // middleware. Poll this server while readiness-critical filesystem/SQLite
    // work builds the real router on the blocking pool.
    let serve = axum::serve(
        listener,
        bootstrap.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .into_future();
    tokio::pin!(serve);
    let mut build =
        tokio::task::spawn_blocking(move || create_app_with_pending_startup_maintenance(settings));
    let build_result = tokio::select! {
        result = &mut build => result?,
        result = &mut serve => {
            bootstrap_maintenance.cancel();
            build.abort();
            result?;
            return Ok(());
        }
    };
    let (app, state) = build_result?;
    state.startup_maintenance.start(
        state.settings.data_dir.clone(),
        BACKGROUND_MAINTENANCE_TIMEOUT,
    );
    ready_app
        .set(app)
        .map_err(|_| "SceneWorks API router was initialized more than once")?;
    tracing::info!(
        event = "api_ready",
        address = %bound,
        "SceneWorks API readiness-critical startup completed"
    );
    let progress_side_effect_recovery =
        crate::jobs::spawn_terminal_progress_side_effect_recovery(state.clone());

    // Conversion recovery is readiness-critical for the utility worker, not for
    // HTTP. Keep polling the already-ready server while the worker initializes.
    let utility_worker = if run_utility_inprocess {
        tokio::select! {
            worker = spawn_inprocess_utility_worker(port, utility_data_dir) => Some(worker?),
            result = &mut serve => {
                state.startup_maintenance.cancel();
                progress_side_effect_recovery.abort();
                shutdown_catalog_scans(&state).await;
                result?;
                return Ok(());
            }
        }
    } else {
        None
    };

    let serve_result = serve.await;
    state.startup_maintenance.cancel();
    progress_side_effect_recovery.abort();
    shutdown_catalog_scans(&state).await;
    serve_result?;

    if let Some(worker) = utility_worker {
        worker.shutdown().await;
    }
    Ok(())
}

async fn shutdown_catalog_scans(state: &AppState) {
    let result = state.catalog_scan_supervisor.shutdown().await;
    if result.requested > 0 {
        tracing::info!(
            event = "catalog_scan_shutdown_complete",
            requested = result.requested,
            joined = result.joined,
            "catalog scans drained during graceful shutdown"
        );
    }
}

/// Dispatched from `main` when `SCENEWORKS_WORKER_ONLY=1`; the desktop app uses
/// it to launch the Apple-Silicon MLX GPU worker (`SCENEWORKS_GPU_ID=mlx`,
/// sc-3289) as a crash-isolated sibling of the API process — reusing this binary
/// because it already links the mlx-gen engine.
///
/// Delegates to [`sceneworks_worker::run`] (which reads `SCENEWORKS_GPU_ID` +
/// `SCENEWORKS_API_URL` and, for a non-`auto`/non-`cpu` id, runs a single worker
/// loop), raced against the same parent-death watchdog the API uses: the desktop
/// sets `SCENEWORKS_PARENT_PID` to its own PID, and a force-quit/crash skips the
/// shell's graceful teardown — so without this a worker would orphan to launchd
/// with its multi-GB MLX model resident.
pub async fn run_worker() -> Result<(), Box<dyn std::error::Error>> {
    // GPU-worker path of the shared binary (SCENEWORKS_WORKER_ONLY=1): no in-process
    // log buffer — its stdout is captured by the desktop wrapper / Docker.
    sceneworks_core::observability::init_logging();
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        tracing::info!(
            event = "hf_home_defaulted",
            home = %home.display(),
            "SceneWorks Rust worker defaulting HF_HOME"
        );
    }
    // Race the worker against the parent-death watchdog on every platform. The
    // desktop sets SCENEWORKS_PARENT_PID to its own PID; a force-quit/crash skips
    // the shell's graceful teardown, and a graceful quit on Windows TerminateProcess-
    // kills only the `auto` supervisor — never its per-GPU/CPU children. Without this
    // watchdog those children orphan, holding multi-GB CUDA contexts and a jobs.db
    // handle until the next launch reaps them (and the reap only knows the supervisor
    // PID, so the children accumulated unbounded). Unset (server/Docker) -> the
    // watchdog future stays pending and never fires.
    tokio::select! {
        result = sceneworks_worker::run() => result?,
        _ = parent_death(parent_pid_to_watch()) => {
            tracing::info!(
                event = "worker_parent_gone",
                "SceneWorks Rust worker: watched parent process gone, exiting"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod server_tests {
    use super::{bootstrap_router, default_mcp_api_url, env_secs};
    use crate::startup::StartupMaintenance;
    use crate::tests::support::test_settings;
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    #[test]
    fn env_secs_parses_whole_seconds_and_falls_back() {
        // A unique name so this never collides with a real SCENEWORKS_* var.
        let name = "SCENEWORKS_TEST_ENV_SECS_SC10277";
        let default = Duration::from_secs(1800);

        std::env::remove_var(name);
        assert_eq!(env_secs(name, default), default, "unset → default");

        std::env::set_var(name, "42");
        assert_eq!(env_secs(name, default), Duration::from_secs(42));

        std::env::set_var(name, "  7  ");
        assert_eq!(env_secs(name, default), Duration::from_secs(7), "trimmed");

        std::env::set_var(name, "notanumber");
        assert_eq!(env_secs(name, default), default, "unparseable → default");

        std::env::remove_var(name);
    }

    #[tokio::test]
    async fn bootstrap_listener_serves_health_and_access_before_readiness() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let ready_app = Arc::new(OnceLock::new());
        let bootstrap = bootstrap_router(
            settings.clone(),
            ready_app.clone(),
            StartupMaintenance::pending(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let address = listener.local_addr().expect("bound address reads");
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                bootstrap.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
        });
        let client = reqwest::Client::new();

        let bootstrap_started = std::time::Instant::now();
        let health_response = client
            .get(format!("http://{address}/api/v1/health"))
            .header("origin", "http://localhost:5173")
            .send()
            .await
            .expect("bootstrap health responds");
        assert_eq!(
            health_response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );
        let health = health_response
            .json::<serde_json::Value>()
            .await
            .expect("bootstrap health parses");
        assert_eq!(health["status"], "starting");
        assert_eq!(health["readiness"]["status"], "initializing");
        let access = client
            .get(format!("http://{address}/api/v1/access"))
            .header("origin", "http://localhost:5173")
            .send()
            .await
            .expect("bootstrap access responds");
        assert_eq!(access.status(), reqwest::StatusCode::OK);
        assert_eq!(
            access
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );
        let preflight = client
            .request(
                reqwest::Method::OPTIONS,
                format!("http://{address}/api/v1/access"),
            )
            .header("origin", "http://localhost:5173")
            .header("access-control-request-method", "GET")
            .send()
            .await
            .expect("bootstrap preflight responds");
        assert_eq!(preflight.status(), reqwest::StatusCode::OK);
        assert_eq!(
            preflight
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );
        eprintln!(
            "SC-14788 socket readiness timing: bootstrap /health + /access={:?}",
            bootstrap_started.elapsed()
        );
        let unavailable = client
            .get(format!("http://{address}/api/v1/projects"))
            .send()
            .await
            .expect("bootstrap non-readiness route responds");
        assert_eq!(
            unavailable.status(),
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );

        let (app, _) = crate::create_app_with_state(settings).expect("ready app creates");
        ready_app.set(app).expect("ready app installs once");
        let ready = client
            .get(format!("http://{address}/api/v1/health"))
            .send()
            .await
            .expect("ready health responds")
            .json::<serde_json::Value>()
            .await
            .expect("ready health parses");
        assert_eq!(ready["status"], "ok");
        assert_eq!(ready["readiness"]["status"], "ready");
        server.abort();
    }

    #[test]
    fn default_mcp_api_url_dials_loopback_for_wildcard_and_loopback_hosts() {
        for host in [
            "0.0.0.0",
            "::",
            "[::]",
            "127.0.0.1",
            "::1",
            "[::1]",
            "localhost",
            "LocalHost",
            "",
        ] {
            assert_eq!(
                default_mcp_api_url(host, 8000),
                "http://127.0.0.1:8000",
                "host {host:?} should self-dial loopback"
            );
        }
    }

    #[test]
    fn default_mcp_api_url_dials_a_specific_interface_ip() {
        // A socket bound only to this IP never accepts a 127.0.0.1 connection, so
        // the self-call must target the bound interface (sc-10260).
        assert_eq!(
            default_mcp_api_url("192.168.4.97", 8000),
            "http://192.168.4.97:8000"
        );
        assert_eq!(
            default_mcp_api_url("  10.0.0.5  ", 9001),
            "http://10.0.0.5:9001",
            "surrounding whitespace is trimmed"
        );
    }

    #[test]
    fn default_mcp_api_url_brackets_a_bare_ipv6_literal() {
        assert_eq!(
            default_mcp_api_url("fe80::1", 8000),
            "http://[fe80::1]:8000"
        );
        // An already-bracketed literal is left as-is.
        assert_eq!(
            default_mcp_api_url("[fe80::1]", 8000),
            "http://[fe80::1]:8000"
        );
    }
}
