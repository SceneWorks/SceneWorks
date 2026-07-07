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
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use sceneworks_core::jobs_store::JobsStore;
use sceneworks_core::project_store::ProjectStore;

use crate::auth::AuthThrottle;
use crate::events::EventHub;
use crate::manifest::ManifestCache;
use crate::models::ModelSizeCache;
use crate::tickets::TicketStore;
use crate::{
    create_app, env_path_or, env_string, open_bind_override_enabled, parent_death,
    parent_pid_to_watch, seed_mode_for_config_dir, should_warn_open_bind, shutdown_signal,
    spawn_inprocess_utility_worker, DEFAULT_API_HOST, DEFAULT_CORS_ORIGINS,
};

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
    pub run_utility_inprocess: bool,
    /// Epic 3482 — macOS "MLX-required" mode. When set (the desktop sets it on macOS,
    /// where it spawns the in-process `mlx` worker), the MPS torch worker never claims an
    /// MLX-eligible job: it defers unconditionally to the `mlx` worker, and a job no live
    /// `mlx` worker takes within the grace window fails terminal with `mlx_unavailable`
    /// instead of silently falling back to MPS (sc-3483). Absent on Windows/Linux/Docker
    /// (no `mlx` worker) → today's behaviour unchanged. Ships default OFF (observe); the
    /// final cutover (sc-3492) flips it on for the packaged Mac build.
    pub mlx_required: bool,
    /// Epic 3482 / sc-3484 — when MLX-required, what to do with a job the Rust/MLX flow can't
    /// run (`mac_rust_supported` returns `Err`). **false = warn-only** (default): log a
    /// structured `mlx_unsupported` gap event at claim time but still run the job on the
    /// existing torch path, so flipping `mlx_required` on for observation materializes the gap
    /// list without breaking anything. **true = enforce**: fail the job terminal with
    /// `mlx_unsupported`. Read from `SCENEWORKS_MLX_UNSUPPORTED_MODE` (`enforce` vs anything
    /// else). Irrelevant unless `mlx_required`.
    pub mlx_enforce_unsupported: bool,
    /// Epic 5483 (sc-5502) — the off-Mac (Windows/Linux/Docker) twin of `mlx_required`. When set,
    /// the candle (CUDA) worker is the only GPU backend: a candle-eligible job no live candle
    /// worker takes within the grace window fails terminal with `candle_unavailable` instead of
    /// waiting forever, and (under `candle_enforce_unsupported`) a job the candle/CUDA flow can't
    /// serve (`candle_supported` returns `Err`) fails with `candle_unsupported` instead of silently
    /// falling back to torch. Ships default OFF (the Python torch worker is still the fallback until
    /// the Phase-7 cutover); flip it on per-deployment as candle reaches parity. Read from
    /// `SCENEWORKS_CANDLE_REQUIRED`. Absent on macOS (the `mlx_required` path governs there).
    pub candle_required: bool,
    /// Epic 5483 (sc-5502) — the candle twin of `mlx_enforce_unsupported`. **false = warn-only**
    /// (default): log a structured `candle_unsupported` gap event at claim time but still let the
    /// job run on the existing torch path, so flipping `candle_required` on for observation
    /// materializes the off-Mac gap list without breaking anything. **true = enforce**: fail the
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
        Self {
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.2.0"),
            host: env_string("SCENEWORKS_API_HOST", DEFAULT_API_HOST),
            port: std::env::var("SCENEWORKS_API_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8000),
            data_dir,
            config_dir: env_path_or("SCENEWORKS_CONFIG_DIR", &defaults.config_dir),
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
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }
}

#[derive(Clone)]
pub struct AppState {
    pub(crate) settings: Settings,
    pub(crate) jobs_store: Arc<JobsStore>,
    pub(crate) project_store: Arc<ProjectStore>,
    pub(crate) events: Arc<EventHub>,
    pub(crate) event_tickets: Arc<TicketStore>,
    pub(crate) media_tickets: Arc<TicketStore>,
    // sc-8870 (F-068): per-peer-IP failed-token throttle for the auth oracle.
    pub(crate) auth_throttle: Arc<AuthThrottle>,
    pub(crate) manifest_cache: Arc<Mutex<ManifestCache>>,
    pub(crate) manifest_write_locks: Arc<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>>,
    pub(crate) model_size_cache: Arc<Mutex<ModelSizeCache>>,
    pub(crate) http_client: reqwest::Client,
    pub(crate) interrupted_jobs_on_startup: usize,
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
            "SceneWorks Rust API defaulting HF_HOME"
        );
    }
    let settings = Settings::from_env();
    // A populated builtin catalog is mandatory — model->file resolution depends on
    // it. The desktop wrapper and the Compose bind mount normally provide it; seed
    // any missing manifests here so launching the API binary directly works too,
    // and fail loudly rather than serving an empty catalog if seeding can't finish.
    //
    // Seed mode by config-dir origin (sc-10212): an EXPLICIT `SCENEWORKS_CONFIG_DIR`
    // marks an operator-owned dir — a repo checkout or a Compose bind mount — that must
    // stay authoritative, so keep `IfMissing` there (fill gaps, never clobber an edited
    // copy or dirty a checked-out `config/`). When unset, `config_dir` is the platform
    // default app-owned dir (the same one the desktop seeds `Overwrite`), so refresh it
    // on launch — otherwise a directly-launched API binary keeps serving a STALE seeded
    // catalog after an upgrade (the sc-10193 img2img flag stayed invisible because the
    // months-old seed was never rewritten). Builtin manifests are app-managed; operator
    // customizations live in the separate `user.*.jsonc` files, which seeding never touches.
    let seed_mode =
        seed_mode_for_config_dir(std::env::var("SCENEWORKS_CONFIG_DIR").ok().as_deref());
    if let Err(error) =
        sceneworks_core::builtin_manifests::seed_builtin_manifests(&settings.config_dir, seed_mode)
    {
        return Err(format!(
            "failed to seed builtin manifests into {}: {error}",
            settings.config_dir.join("manifests").display()
        )
        .into());
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
    #[cfg(unix)]
    {
        let creds_path = settings
            .config_dir
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
    let app = create_app(settings)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    // Use the actual bound address so port 0 (OS-assigned) is reported and the
    // in-process worker connects to the real port.
    let bound = listener.local_addr()?;
    let port = bound.port();
    tracing::info!(
        event = "api_listening",
        address = %bound,
        "SceneWorks Rust API listening"
    );

    let utility_worker = run_utility_inprocess.then(|| spawn_inprocess_utility_worker(port));

    // `into_make_service_with_connect_info` exposes the peer `SocketAddr` to the auth
    // middleware so loopback callers can be trusted (epic 4484: keep the local desktop UI
    // and worker password-free while LAN clients stay gated).
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    if let Some(worker) = utility_worker {
        worker.shutdown().await;
    }
    Ok(())
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
