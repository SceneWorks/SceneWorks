//! First-run Python venv bootstrap + startup orchestration (sc-1348).
//!
//! The frontend setup screen calls the `start_setup` command once it is ready to
//! receive events; this provisions the uv-managed venv (streaming progress),
//! then spawns the API sidecar, health-gates it, and navigates the window to the
//! local API. `start_setup` is also the retry entry point.

#[cfg(any(all(unix, not(target_os = "macos")), test))]
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sceneworks_core::session_log::{LogEntry, LogQuery, SessionLog};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::{Command, CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Process-global in-app session log (sc-3451). Every captured sidecar line
/// (api/worker/mlx-worker) is mirrored here as it's appended to disk, so the
/// `get_session_logs` command can serve the current session's activity — the
/// MLX routing decisions, claim contention and worker phases — without parsing
/// the append-only files in `~/Library/Logs/SceneWorks/`. "Current session" =
/// this desktop process's lifetime (the buffer is created on first capture).
static SESSION_LOG: OnceLock<SessionLog> = OnceLock::new();

pub fn session_log() -> &'static SessionLog {
    SESSION_LOG.get_or_init(SessionLog::default)
}

/// Read back the current session's log entries for the in-app Logs screen
/// (sc-3452). `after_seq` tails only new lines; the rest are filters.
#[tauri::command]
pub fn get_session_logs(
    after_seq: Option<u64>,
    limit: Option<usize>,
    source: Option<String>,
    level: Option<String>,
    search: Option<String>,
) -> Vec<LogEntry> {
    session_log().query(&LogQuery {
        after_seq,
        limit,
        source,
        level,
        search,
    })
}

const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Process handles + run guards shared across the app.
#[derive(Default)]
pub struct Managed {
    pub api: Mutex<Option<CommandChild>>,
    /// The Apple-Silicon MLX GPU worker (sc-3289): the same `sceneworks-api`
    /// binary re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`,
    /// `SCENEWORKS_GPU_ID=mlx`). Only populated on macOS.
    pub mlx_worker: Mutex<Option<CommandChild>>,
    /// The Windows/Linux candle (CUDA) GPU worker supervisor (sc-5561/sc-10375):
    /// the same
    /// `sceneworks-api` binary re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`,
    /// `SCENEWORKS_GPU_ID=auto`, `SCENEWORKS_BACKEND_CANDLE_ENABLED=true`). `auto`
    /// makes it the multi-GPU supervisor — it spawns one candle child per NVIDIA GPU
    /// (those children are owned by the supervisor, not tracked here). Only populated
    /// on the Windows or Linux candle build.
    pub candle_worker: Mutex<Option<CommandChild>>,
    /// On-demand keychain credential socket served to the MLX worker (sc-5891).
    /// Started once before the worker spawns; the worker pulls a host's secret from
    /// it the first time a download needs auth, so the keychain is read lazily
    /// instead of eagerly at launch. macOS-only.
    #[cfg(target_os = "macos")]
    pub cred_ipc: Mutex<Option<crate::cred_ipc::CredIpc>>,
    /// OS-assigned API port, discovered from the sidecar's startup line.
    api_port: Mutex<Option<u16>>,
    /// PIDs of the spawned sidecars, persisted to disk so an unclean exit
    /// (crash/force-quit) doesn't leave them orphaned — the next launch reaps
    /// any survivors before spawning fresh ones.
    pids: Mutex<SidecarPids>,
    running: AtomicBool,
    pub shutting_down: AtomicBool,
    /// Single-live-supervisor guard for the GPU-worker respawn loop (sc-13605).
    /// An API crash clears the `api` slot so a Retry re-runs `spawn_api` +
    /// `gate_window`; without this guard `gate_window` would stack a second
    /// supervisor thread on the one worker slot every cycle. See `SupervisorSlot`.
    worker_supervisor: SupervisorSlot,
}

/// One-live-supervisor guard for the GPU-worker respawn loop (sc-13605).
///
/// Before sc-13605 every `gate_window` call unconditionally spawned a worker
/// supervisor thread. `handle_api_exit` clears the API slot on a crash, so a
/// Retry re-runs `spawn_api` + `gate_window` — and each API-crash → Retry cycle
/// stacked another supervisor on the *single* `mlx_worker` / `candle_worker`
/// slot. The stale supervisors never exited (`shutting_down` stays false) and
/// kept respawning a worker pointed at the dead port captured at their spawn
/// (perpetual backoff churn + a zombie worker), while `restart_gpu_worker`
/// (sc-13584) could only kill whichever child happened to occupy the slot.
///
/// The guard enforces "at most one live supervisor": the first supervisor to
/// [`try_acquire`](SupervisorSlot::try_acquire) owns the slot; a `gate_window`
/// that runs again while one is live does NOT start a second. The single
/// survivor re-reads the current API port every iteration (see
/// [`current_api_url`]), so after a Retry it points the worker at the NEW port
/// rather than a URL captured at spawn. The flag is cleared on every supervisor
/// exit path by [`SupervisorLease`]'s `Drop`, so a supervisor that legitimately
/// exits (e.g. a sidecar-locate failure) never wedges the flag against a later
/// restart. Kept deliberately small and self-contained; the F-053 dedup
/// (sc-13615) will fold the two near-identical supervisor loops together.
#[derive(Default)]
struct SupervisorSlot {
    /// True while a supervisor loop owns the slot.
    live: AtomicBool,
}

impl SupervisorSlot {
    /// Try to become the sole live supervisor. Returns `true` if the slot was
    /// free (the caller starts its loop), or `false` if a supervisor is already
    /// live (the caller must NOT start a second one — the live one re-reads the
    /// current API port and keeps the single worker slot correct).
    fn try_acquire(&self) -> bool {
        // swap→was-live: previously-false means we won the slot.
        !self.live.swap(true, Ordering::SeqCst)
    }

    /// Release the slot so a later `gate_window` (e.g. after this supervisor
    /// exited on a sidecar-locate failure) can start a fresh one.
    fn release(&self) {
        self.live.store(false, Ordering::SeqCst);
    }
}

/// RAII lease that clears `Managed.worker_supervisor` liveness on *every* exit
/// path of a supervisor thread — early `return`, loop break, or panic — so a
/// stale supervisor can never leave the flag stuck and block a legitimate later
/// restart (sc-13605). Held for the lifetime of the supervisor thread closure.
struct SupervisorLease {
    app: AppHandle,
}

impl SupervisorLease {
    /// Acquire the sole-supervisor slot, or `None` if one is already live (in
    /// which case the caller must return without starting a supervisor loop).
    fn acquire(app: &AppHandle) -> Option<Self> {
        if app.state::<Managed>().worker_supervisor.try_acquire() {
            Some(SupervisorLease { app: app.clone() })
        } else {
            None
        }
    }
}

impl Drop for SupervisorLease {
    fn drop(&mut self) {
        self.app.state::<Managed>().worker_supervisor.release();
    }
}

/// The worker-facing URL for the API's currently-discovered port, re-read from
/// shared `Managed` state (never a value captured at supervisor spawn), or
/// `None` if no port has been discovered yet (sc-13605). The supervisor loops
/// derive their per-attempt target from this every iteration so that after an
/// API-crash → Retry the respawned worker targets the NEW port, not a stale one.
fn current_api_url(managed: &Managed) -> Option<String> {
    let port = *managed.api_port.lock().expect("api port lock");
    port.map(|port| format!("http://127.0.0.1:{port}"))
}

/// What a supervisor iteration does next, resolved *purely* from the live
/// shutdown flag + currently-discovered API url — both re-read every iteration,
/// never captured at spawn (sc-13605). Factored out (with
/// [`plan_supervisor_action`] / [`resolve_supervisor_action`]) so the
/// per-attempt port re-read is unit-testable and identical across the two
/// `supervise_*` loops.
#[derive(Debug, PartialEq, Eq)]
enum SupervisorAction {
    /// A shutdown has latched — exit the loop (the lease `Drop` clears liveness).
    Exit,
    /// No API port is currently published (post-crash, pre-Retry) — park briefly
    /// and re-decide, so the worker is (re)spawned only against a live port.
    WaitForPort,
    /// (Re)spawn the worker against this url.
    Spawn(String),
}

/// Pure core of the per-iteration decision: given the freshly-read shutdown flag
/// and current API url, what should the supervisor do? (sc-13605)
fn plan_supervisor_action(
    shutting_down: bool,
    current_api_url: Option<String>,
) -> SupervisorAction {
    match (shutting_down, current_api_url) {
        (true, _) => SupervisorAction::Exit,
        (false, None) => SupervisorAction::WaitForPort,
        (false, Some(url)) => SupervisorAction::Spawn(url),
    }
}

/// Re-read the live supervisor inputs from `Managed` and resolve the next action.
/// The single seam both `supervise_*` loops go through every iteration (top of
/// loop AND the post-spawn recheck), so the port is never captured — a
/// regression that memoized it here would flip the `supervisor_tests`
/// re-read assertions (sc-13605).
#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn resolve_supervisor_action(managed: &Managed) -> SupervisorAction {
    plan_supervisor_action(
        managed.shutting_down.load(Ordering::SeqCst),
        current_api_url(managed),
    )
}

/// Verdict for the child a supervisor just spawned + stored, from re-reading the
/// current action AFTER the store (sc-13605 correlated-death guard). Closes the
/// window where an API crash clears the port and `handle_api_exit`'s
/// `restart_gpu_worker` kill fires *before* the child was stored (so it misses
/// it): if the current target no longer matches what we launched against, the
/// child is stale and must be killed — then the loop exits (shutdown) or retries
/// (re-reads the fresh port), so the sole supervisor never blocks on a zombie
/// pointed at the dead port.
#[derive(Debug, PartialEq, Eq)]
enum SpawnVerdict {
    /// Target still current — keep the child and block on its event stream.
    Keep,
    /// Shutdown latched meanwhile — kill the child and exit the loop.
    KillAndExit,
    /// Port changed/cleared meanwhile — kill the child and retry the loop.
    KillAndRetry,
}

/// Decide a just-stored child's fate from the target it was launched against and
/// the action re-read after storing it (sc-13605).
fn verify_spawned_target(spawned_url: &str, recheck: &SupervisorAction) -> SpawnVerdict {
    match recheck {
        SupervisorAction::Spawn(url) if url.as_str() == spawned_url => SpawnVerdict::Keep,
        SupervisorAction::Exit => SpawnVerdict::KillAndExit,
        // WaitForPort (port cleared) or Spawn(other) (port changed) — stale.
        _ => SpawnVerdict::KillAndRetry,
    }
}

/// PIDs of the API + GPU worker (MLX on macOS, candle on Windows/Linux) sidecars
/// owned by this launch.
#[derive(Default, Clone, Serialize, Deserialize)]
struct SidecarPids {
    api: Option<u32>,
    /// The MLX GPU worker (sc-3289). `#[serde(default)]` so a pidfile written by
    /// an older build (no such field) still deserializes for reaping.
    #[serde(default)]
    mlx_worker: Option<u32>,
    /// The Windows/Linux candle GPU worker (sc-5561/sc-10375). `#[serde(default)]`
    /// so an older
    /// pidfile (no such field) still deserializes for reaping.
    #[serde(default)]
    candle_worker: Option<u32>,
}

#[derive(Clone, Serialize)]
struct SetupStatus {
    phase: String,
    message: String,
    error: bool,
}

pub(crate) fn emit(app: &AppHandle, phase: &str, message: impl Into<String>, error: bool) {
    let _ = app.emit(
        "setup-status",
        SetupStatus {
            phase: phase.to_owned(),
            message: message.into(),
            error,
        },
    );
}

#[cfg(any(all(unix, not(target_os = "macos")), test))]
const APP_DIR_NAME: &str = "SceneWorks";

/// Linux desktop paths resolved from the XDG base-directory environment.
///
/// Kept as a pure helper (rather than reading the process environment directly)
/// so all XDG override and fallback behavior is testable on every CI host. A
/// missing absolute XDG base and missing absolute `HOME` is an error: falling
/// back to `temp_dir()` would trust `TMPDIR`, permit relative/CWD writes, and use
/// a predictable cross-user directory.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(all(unix, not(target_os = "macos")), test))]
struct LinuxDesktopPaths {
    data_dir: PathBuf,
    config_dir: PathBuf,
    cache_dir: PathBuf,
    state_dir: PathBuf,
}

#[cfg(any(all(unix, not(target_os = "macos")), test))]
impl LinuxDesktopPaths {
    fn absolute_linux_path(value: OsString) -> Option<PathBuf> {
        let path = PathBuf::from(value);
        path.as_os_str()
            .to_string_lossy()
            .starts_with('/')
            .then_some(path)
    }

    fn resolve(get_env: impl Fn(&str) -> Option<OsString>) -> Result<LinuxDesktopPaths, String> {
        let home = get_env("HOME").and_then(LinuxDesktopPaths::absolute_linux_path);
        let xdg_dir = |name: &str, fallback: &[&str], temp_leaf: &str| {
            get_env(name)
                .and_then(LinuxDesktopPaths::absolute_linux_path)
                .or_else(|| {
                    home.as_ref().map(|home| {
                        fallback
                            .iter()
                            .fold(home.clone(), |path, component| path.join(component))
                    })
                })
                .map(|base| base.join(APP_DIR_NAME))
                .ok_or_else(|| {
                    format!(
                        "cannot resolve Linux {temp_leaf} directory: set {name} or HOME to an absolute path"
                    )
                })
        };

        Ok(LinuxDesktopPaths {
            data_dir: xdg_dir("XDG_DATA_HOME", &[".local", "share"], "data")?,
            config_dir: xdg_dir("XDG_CONFIG_HOME", &[".config"], "config")?,
            cache_dir: xdg_dir("XDG_CACHE_HOME", &[".cache"], "cache")?,
            state_dir: xdg_dir("XDG_STATE_HOME", &[".local", "state"], "state")?,
        })
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    fn from_process_env() -> Result<LinuxDesktopPaths, String> {
        Self::resolve(|name| std::env::var_os(name))
    }

    fn settings_file(&self) -> PathBuf {
        self.config_dir.join("settings.json")
    }

    fn data_dir(&self) -> PathBuf {
        self.data_dir.clone()
    }

    fn config_dir(&self) -> PathBuf {
        self.config_dir.clone()
    }

    fn logs_dir(&self) -> PathBuf {
        self.state_dir.join("logs")
    }

    fn huggingface_home(&self) -> PathBuf {
        self.cache_dir.join("huggingface")
    }

    fn gpu_runtime_dir(&self) -> PathBuf {
        self.data_dir.join("gpu-runtime")
    }

    fn sidecar_pidfile(&self) -> PathBuf {
        self.state_dir.join("desktop-sidecars.json")
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn linux_desktop_paths() -> LinuxDesktopPaths {
    LinuxDesktopPaths::from_process_env()
        .unwrap_or_else(|error| panic!("SceneWorks Linux path configuration is invalid: {error}"))
}

/// Per-OS application support root: `~/Library/Application Support/SceneWorks`
/// (macOS), `%APPDATA%\SceneWorks` (Windows), `$XDG_DATA_HOME/SceneWorks` or
/// `~/.local/share/SceneWorks` (Linux).
pub fn app_support_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("SceneWorks");
    }
    #[cfg(target_os = "windows")]
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("SceneWorks");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().data_dir()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        std::env::temp_dir().join("SceneWorks")
    }
}

/// Platform-appropriate logs directory (also used for the API/worker logs).
pub fn logs_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Logs")
            .join("SceneWorks");
    }
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("SceneWorks").join("logs");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().logs_dir()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        std::env::temp_dir().join("SceneWorks").join("logs")
    }
}

/// Platform default workspace data directory, used when the user hasn't picked a
/// custom location in the first-run splash / Settings.
pub fn default_data_dir() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().data_dir()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        app_support_dir().join("data")
    }
}

pub(crate) fn config_dir() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().config_dir()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        app_support_dir().join("config")
    }
}

pub(crate) fn settings_file() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().settings_file()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        app_support_dir().join("settings.json")
    }
}

/// Root for the provisioned native GPU runtime. Linux provisioning (sc-10376)
/// consumes this path so its runtime stays under the XDG data base; the Windows
/// value remains `%APPDATA%\SceneWorks\gpu-runtime`.
pub fn gpu_runtime_dir() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().gpu_runtime_dir()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        app_support_dir().join("gpu-runtime")
    }
}

/// Shared per-user Hugging Face cache — the default `HF_HOME` when the user
/// hasn't chosen a custom location. Linux uses
/// `$XDG_CACHE_HOME/SceneWorks/huggingface` (or
/// `~/.cache/SceneWorks/huggingface`); macOS and Windows retain the existing
/// `~/.cache/huggingface` convention.
pub fn shared_huggingface_home() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().huggingface_home()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        if let Some(home) = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            return PathBuf::from(home).join(".cache").join("huggingface");
        }
        app_support_dir().join("cache").join("huggingface")
    }
}

/// Hugging Face cache home injected into both sidecars so the rust-api model
/// catalog and the native inference worker resolve weights from the same root.
/// Without this the API falls back to `<data_dir>/cache/huggingface` while the
/// worker uses the HF hub default `~/.cache/huggingface`, so they disagree and
/// every catalog entry shows "missing" (sc-1473 Step 1 gap).
/// Resolution order: an explicit `HF_HOME` in the environment (useful under
/// `tauri dev`), then the user's persisted choice from the first-run splash, then
/// the shared per-user cache. Because the splash persists this *before* the
/// sidecars spawn, the chosen location takes effect with no app restart.
fn select_huggingface_home(
    ambient: Option<&str>,
    persisted: Option<&str>,
    shared: PathBuf,
    linux_absolute: bool,
) -> PathBuf {
    ambient
        .and_then(|value| crate::settings::storage_override_path(value, linux_absolute))
        .or_else(|| {
            persisted
                .and_then(|value| crate::settings::storage_override_path(value, linux_absolute))
        })
        .unwrap_or(shared)
}

fn huggingface_cache_env(hf_home: &str, linux_absolute: bool) -> Vec<(&'static str, String)> {
    let home = crate::settings::storage_override_path(hf_home, linux_absolute)
        .expect("resolved HF_HOME must satisfy the platform path invariant");
    let home = home.to_string_lossy().into_owned();
    let mut env = vec![("HF_HOME", home.clone())];
    if linux_absolute {
        let hub = if home == "/" {
            "/hub".to_owned()
        } else {
            format!("{}/hub", home.trim_end_matches('/'))
        };
        // The core resolver reads these before HF_HOME. Pin both like Docker so
        // inherited relative values cannot redirect Linux sidecars into CWD.
        env.push(("HF_HUB_CACHE", hub.clone()));
        env.push(("HUGGINGFACE_HUB_CACHE", hub));
    }
    env
}

fn inject_huggingface_cache_env(command: Command, hf_home: &str) -> Command {
    huggingface_cache_env(hf_home, cfg!(all(unix, not(target_os = "macos"))))
        .into_iter()
        .fold(command, |command, (name, value)| command.env(name, value))
}

fn huggingface_home() -> PathBuf {
    let ambient = std::env::var("HF_HOME").ok();
    let persisted = crate::settings::load_settings().hf_home;
    select_huggingface_home(
        ambient.as_deref(),
        persisted.as_deref(),
        shared_huggingface_home(),
        cfg!(all(unix, not(target_os = "macos"))),
    )
}

/// Seed the builtin model/LoRA/recipe-preset catalogs into the desktop's
/// `config_dir/manifests`, overwriting on every launch so they track the app
/// version. The server stack ships these in the repo's `config/`, but the desktop
/// must provide them itself or Model Manager is empty and the native LTX/Wan
/// adapters can't map model resources to files. User customizations live in the
/// separate `user.*.jsonc` files, which seeding never touches. Delegates to the
/// shared `sceneworks_core` seeder (same embedded copies the rust-api uses);
/// returns an error if any required manifest can't be installed so the caller
/// aborts setup rather than starting with missing model mappings.
fn seed_builtin_manifests() -> Result<(), String> {
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &config_dir(),
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .map_err(|error| error.to_string())
}

/// Data directory: the settings override if set, otherwise the platform default.
fn resolved_data_dir() -> PathBuf {
    crate::settings::load_settings()
        .data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir)
}

fn append_log(path: &Path, line: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
    // Mirror into the in-app session buffer (sc-3451), tagged by the log's file stem
    // ("worker.log" -> "worker", "mlx-worker.log" -> "mlx-worker").
    let source = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("app");
    session_log().push_line(source, line);
}

/// Extract the port from the API's bound-address startup line. In loopback mode the
/// API logs `event="api_listening" … address=127.0.0.1:PORT`; in LAN mode (epic 4484)
/// it binds 0.0.0.0 and logs `address=0.0.0.0:PORT`. (LAN mode also pre-sets the known
/// fixed port, so this is the fallback for the dynamic loopback case.)
///
/// Anchored to the API's own startup marker (F-127, sc-8929): the port is read from the
/// `address=` field of the `api_listening` event ONLY. An earlier diagnostic line that
/// merely mentions a loopback `host:port` (e.g. a credential-socket or health-probe log)
/// no longer seeds the wrong port, which previously left window-gating polling a dead
/// port until the 30 s timeout fired.
fn parse_listening_port(line: &str) -> Option<u16> {
    // Only the API's bound-address startup line carries a port we should trust.
    if !line.contains("api_listening") {
        return None;
    }
    // Read the port from the `address` field specifically (not any bare host:port
    // elsewhere on the line — a peer addr, a health probe). The packaged sidecar logs
    // JSON to its piped stdout (`"address":"127.0.0.1:PORT"`), while a TTY/dev stdout
    // logs the pretty/logfmt form (`address=127.0.0.1:PORT`). Anchor on the field name
    // and read the first bind marker that follows, so BOTH formats resolve the port —
    // keying only on `address=` (F-127, sc-8929) silently broke the JSON path, which is
    // the default packaged loopback launch, stranding window-gating until the 30 s
    // timeout ("The local API did not start in time.").
    let rest = line.split("address").nth(1)?;
    for marker in ["127.0.0.1:", "0.0.0.0:"] {
        if let Some(index) = rest.find(marker) {
            let digits: String = rest[index + marker.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(port) = digits.parse() {
                return Some(port);
            }
        }
    }
    None
}

/// Health check that also confirms the responder is genuinely the SceneWorks API
/// (HTTP 200 with the expected service/runtime in the JSON body) before we
/// navigate the privileged Tauri window to it — a foreign service that grabbed
/// the port must not be trusted.
fn health_is_sceneworks(port: u16) -> bool {
    use std::io::Read;
    use std::net::TcpStream;
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /api/v1/health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    let ok_status = response
        .lines()
        .next()
        .is_some_and(|status_line| status_line.contains(" 200"));
    ok_status
        && response.contains("\"service\":\"sceneworks-api\"")
        && response.contains("\"runtime\":\"rust\"")
}

/// Resolve the ffmpeg binary the Rust worker shells out to (frame sampling,
/// frame extract, timeline export, video-gen audio mux — all via
/// `media_jobs::run_ffmpeg`, which honors `SCENEWORKS_FFMPEG`). The desktop ships
/// no system ffmpeg, so without this those jobs fail. Prefers the static ffmpeg
/// bundled next to the app (staged by build-sidecar.mjs into the `ffmpeg` resource
/// dir, so a packaged Python-free Mac still works — epic 3482, sc-3767). Returns
/// None when it isn't bundled (pre-bundle / dev — caller then leaves
/// `SCENEWORKS_FFMPEG` unset → PATH ffmpeg).
fn resolve_bundled_ffmpeg(app: &AppHandle) -> Option<String> {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("ffmpeg").join(if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        });
        if bundled.exists() {
            return Some(bundled.to_string_lossy().to_string());
        }
    }
    None
}

/// Resolve the onnxruntime dynamic library the Rust worker's DWPose pose detector
/// (`ort`, sc-3487) dlopens at runtime via `ORT_DYLIB_PATH` (the `load-dynamic`
/// feature). Prefers the dylib bundled next to the app (staged by build-sidecar.mjs
/// into the `onnxruntime` resource dir, so a packaged Python-free Mac still detects
/// poses), the same CoreML-enabled build. Returns None when it isn't bundled
/// (pre-bundle / dev). macOS-only — pose detection on the Rust worker is macOS-only,
/// so this returns None elsewhere.
#[cfg(target_os = "macos")]
fn resolve_bundled_onnxruntime(app: &AppHandle) -> Option<String> {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("onnxruntime").join("libonnxruntime.dylib");
        if bundled.exists() {
            return Some(bundled.to_string_lossy().to_string());
        }
    }
    None
}

/// Resolve MLX's compiled Metal shader library (`mlx.metallib`), which the
/// in-process MLX worker loads at runtime. It is NOT embedded in the binary: the
/// pmetal-mlx-rs fork's resolver (sc-7898) searches `PMETAL_METALLIB_PATH`, then a
/// path into the *build machine's* target dir baked into the binary, then
/// `~/.cache/pmetal` — none of which exist on a clean end-user Mac (the cache is
/// only populated as a side effect of a local `cargo build`), so a packaged app must
/// ship the file and point the worker at it or MLX fails on first use with "Failed
/// to load the default metallib. library not found" (sc-10349). Prefers the copy
/// bundled next to the app (staged by build-sidecar.mjs into the `mlx` resource
/// dir); returns None in dev/pre-bundle, where the fork's own build-tree / cache
/// resolution applies (caller then leaves `PMETAL_METALLIB_PATH` unset). macOS-only —
/// MLX is the macOS inference backend.
#[cfg(target_os = "macos")]
fn resolve_bundled_metallib(app: &AppHandle) -> Option<String> {
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("mlx").join("mlx.metallib");
        if bundled.exists() {
            return Some(bundled.to_string_lossy().to_string());
        }
    }
    None
}

/// Resolve the CUDA-enabled onnxruntime DLL the candle worker's `ort` paths (DWPose
/// pose_detect sc-5496, + YOLO/Real-ESRGAN, epic 5482) dlopen at runtime via
/// `ORT_DYLIB_PATH` (the `load-dynamic` feature). The Windows/CUDA analogue of the
/// macOS CoreML resolver above. The onnxruntime-gpu DLLs are no longer bundled (the
/// ~2.7 GB GPU runtime blew past NSIS's datablock limit); they're downloaded on first
/// run into `%APPDATA%\SceneWorks\gpu-runtime\onnxruntime` (cuda_provision.rs) and
/// resolved from there. Returns None until that first-run provisioning completes — the
/// non-candle / dev path never reaches the candle worker that consumes it. Windows-only
/// (the candle GPU worker is Windows-gated here).
#[cfg(target_os = "windows")]
fn resolve_bundled_onnxruntime(_app: &AppHandle) -> Option<String> {
    crate::cuda_provision::onnxruntime_dll_if_present().map(|dll| dll.to_string_lossy().to_string())
}

/// Resolve the CUDA runtime redistributable DLL directory (sc-5560). The candle
/// (Windows/CUDA) worker links cudarc with dynamic-linking, which `LoadLibrary`s
/// cudart/cublas/cublasLt/curand/nvrtc by name at runtime. These DLLs are no longer
/// bundled (the ~2.7 GB GPU runtime exceeded NSIS's ~2 GB datablock limit); they're
/// downloaded on first run into `%APPDATA%\SceneWorks\gpu-runtime\cuda`
/// (cuda_provision.rs) and resolved from there. `spawn_api` /
/// `supervise_candle_worker` prepend this dir to the sidecar's PATH so the loader
/// finds them. Returns None until first-run provisioning has written the DLLs (probes
/// `cudart64_12.dll`); this also gates the candle worker spawn + cuda_preflight, so a
/// pre-provision / failed-provision state leaves the candle lane dormant rather than
/// spawning a worker whose CUDA load would fail. Windows-only (candle is Windows-
/// gated); the driver-side CUDA (nvcuda.dll) is NOT downloaded — it comes with the
/// user's NVIDIA display driver (>= 576.02).
#[cfg(target_os = "windows")]
fn resolve_bundled_cuda_dir(_app: &AppHandle) -> Option<std::path::PathBuf> {
    crate::cuda_provision::cuda_dir_if_present()
}

/// The provisioned Linux candle runtime resolved from the XDG-managed
/// `gpu-runtime` root. Provisioning is intentionally owned by sc-10376; this
/// story only consumes a completed runtime and keeps the lane dormant while the
/// required sentinels are absent.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct LinuxCandleRuntime {
    ort_dylib: PathBuf,
    cuda_dir: PathBuf,
    cudnn_dir: PathBuf,
    loader_dirs: Vec<PathBuf>,
}

/// Find an ELF shared object by its unversioned name or a versioned sibling
/// (`libfoo.so.1`, `libfoo.so.1.2`, ...).
#[cfg(any(target_os = "linux", test))]
fn find_linux_shared_object(dir: &Path, basename: &str) -> Option<PathBuf> {
    let exact = dir.join(basename);
    if exact.is_file() {
        return Some(exact);
    }
    let versioned_prefix = format!("{basename}.");
    let mut candidates = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            (name.starts_with(&versioned_prefix) && entry.path().is_file()).then(|| entry.path())
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

/// Known library locations supported by the Linux provisioner contract. The
/// flat `<root>/cuda` layout mirrors Windows; the component-specific paths
/// mirror the PyPI-wheel and Docker layouts sc-10376 will stage.
#[cfg(any(target_os = "linux", test))]
fn linux_runtime_library_dirs(root: &Path) -> Vec<PathBuf> {
    [
        "onnxruntime/capi",
        "onnxruntime",
        "cudnn/lib",
        "nvidia/cudnn/lib",
        "cufft/lib",
        "nvidia/cufft/lib",
        "nvjitlink/lib",
        "nvidia/nvjitlink/lib",
        "cuda_nvrtc/lib",
        "nvidia/cuda_nvrtc/lib",
        "cublas/lib",
        "nvidia/cublas/lib",
        "curand/lib",
        "nvidia/curand/lib",
        "cuda/lib64",
        "cuda",
        "nvidia/cuda_runtime/lib",
    ]
    .into_iter()
    .map(|path| root.join(path))
    .filter(|path| {
        std::fs::read_dir(path)
            .ok()
            .is_some_and(|entries| entries.flatten().any(|entry| entry.path().is_file()))
    })
    .collect()
}

/// Detect a complete-enough Linux runtime for starting the candle supervisor.
/// Requiring onnxruntime + CUDA runtime + cuDNN sentinels is the pre-provision
/// crash-loop gate. Full component completeness remains sc-10376's concern.
#[cfg(any(target_os = "linux", test))]
fn resolve_linux_candle_runtime(root: &Path) -> Option<LinuxCandleRuntime> {
    let loader_dirs = linux_runtime_library_dirs(root);
    let ort_dylib = loader_dirs
        .iter()
        .find_map(|dir| find_linux_shared_object(dir, "libonnxruntime.so"))?;
    // Candle itself dynamically loads the CUDA generation set; onnxruntime's
    // CUDA EP adds cuDNN/cuFFT/nvJitLink. A partial subset would pass a one-file
    // probe only to crash after the supervisor starts, so require one sentinel
    // from every provisioned component before enabling the lane.
    for required in [
        "libcudart.so",
        "libcublas.so",
        "libcublasLt.so",
        "libcurand.so",
        "libnvrtc.so",
        "libcudnn.so",
        "libcufft.so",
        "libnvJitLink.so",
    ] {
        loader_dirs
            .iter()
            .find_map(|dir| find_linux_shared_object(dir, required))?;
    }
    let cuda_dir = loader_dirs
        .iter()
        .find(|dir| find_linux_shared_object(dir, "libcudart.so").is_some())?
        .clone();
    let cudnn_dir = loader_dirs
        .iter()
        .find(|dir| find_linux_shared_object(dir, "libcudnn.so").is_some())?
        .clone();
    Some(LinuxCandleRuntime {
        ort_dylib,
        cuda_dir,
        cudnn_dir,
        loader_dirs,
    })
}

/// Prepend runtime loader directories to inherited ones, preserving order and
/// removing duplicates. Kept path-based so tests can validate Linux loader-env
/// composition on non-Linux hosts.
#[cfg(any(target_os = "linux", test))]
fn prepend_loader_paths(
    runtime_dirs: &[PathBuf],
    inherited_dirs: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let mut combined = Vec::new();
    for path in runtime_dirs.iter().cloned().chain(inherited_dirs) {
        if !combined.contains(&path) {
            combined.push(path);
        }
    }
    combined
}

#[cfg(target_os = "linux")]
fn linux_candle_runtime() -> Option<LinuxCandleRuntime> {
    resolve_linux_candle_runtime(&gpu_runtime_dir())
}

/// Apply the Linux analogue of the Windows candle PATH/ORT block. Both the API
/// sidecar and GPU-worker supervisor use this seam.
#[cfg(target_os = "linux")]
fn inject_linux_candle_runtime_env(mut command: Command, runtime: &LinuxCandleRuntime) -> Command {
    let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let paths = prepend_loader_paths(&runtime.loader_dirs, std::env::split_paths(&inherited));
    if let Ok(joined) = std::env::join_paths(paths) {
        command = command.env("LD_LIBRARY_PATH", joined);
    }
    command
        .env(
            "ORT_DYLIB_PATH",
            runtime.ort_dylib.to_string_lossy().to_string(),
        )
        .env(
            "SCENEWORKS_ORT_CUDA_DIR",
            runtime.cuda_dir.to_string_lossy().to_string(),
        )
        .env(
            "SCENEWORKS_ORT_CUDNN_DIR",
            runtime.cudnn_dir.to_string_lossy().to_string(),
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DesktopPlatform {
    Macos,
    Windows,
    Linux,
    Other,
}

const fn current_desktop_platform() -> DesktopPlatform {
    if cfg!(target_os = "macos") {
        DesktopPlatform::Macos
    } else if cfg!(target_os = "windows") {
        DesktopPlatform::Windows
    } else if cfg!(target_os = "linux") {
        DesktopPlatform::Linux
    } else {
        DesktopPlatform::Other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerSupervisor {
    Mlx,
    Candle,
    Dormant,
}

/// Pure platform/runtime selection used by `gate_window`. Linux selects candle
/// exactly like Windows once its runtime is present; pre-provision Linux stays
/// dormant instead of starting a crash-looping supervisor.
fn select_worker_supervisor(
    platform: DesktopPlatform,
    candle_runtime_present: bool,
) -> WorkerSupervisor {
    match platform {
        DesktopPlatform::Macos => WorkerSupervisor::Mlx,
        DesktopPlatform::Windows | DesktopPlatform::Linux if candle_runtime_present => {
            WorkerSupervisor::Candle
        }
        _ => WorkerSupervisor::Dormant,
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn candle_runtime_present(app: &AppHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        resolve_bundled_cuda_dir(app).is_some()
    }
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        linux_candle_runtime().is_some()
    }
}

/// The resolved bind/auth environment for the API sidecar for one launch (epic 4484
/// stories 2/3). Pure output of [`decide_api_bind_env`] so the loopback-vs-LAN choice
/// is unit-tested without spawning anything.
#[derive(Debug, PartialEq)]
struct ApiBindEnv {
    /// `SCENEWORKS_API_HOST`: `127.0.0.1` (loopback) or `0.0.0.0` (LAN, all interfaces).
    host: &'static str,
    /// `SCENEWORKS_API_PORT`: `"0"` (OS-assigned dynamic) in loopback mode, or the
    /// configured fixed port (as a string) in LAN mode.
    port: String,
    /// `SCENEWORKS_ACCESS_TOKEN`: the user's password in LAN mode, `None` otherwise.
    /// When set it is also injected into the GPU worker(s) so they can still reach the
    /// now-authenticated API.
    access_token: Option<String>,
    /// The fixed bound port when known up-front (LAN mode), so window-gating doesn't
    /// depend on parsing the API's stdout marker. `None` ⇒ discover from the log.
    fixed_port: Option<u16>,
    /// A non-fatal warning to log when remote access is requested but can't be honored
    /// safely (enabled without a password) — the bind falls back to loopback rather
    /// than exposing the host unauthenticated (fail-closed, story 3).
    warning: Option<String>,
}

/// Choose the API sidecar's bind/auth env from the persisted remote-access settings.
///
/// Fail-closed security invariant (epic 4484 story 3): a non-loopback bind happens
/// ONLY when remote access is explicitly enabled AND a non-empty password is present.
/// Disabled → today's loopback/dynamic/no-token behavior, byte-for-byte. Enabled but
/// no password → loopback with a warning (NEVER an open unauthenticated bind). The
/// desktop never sets `SCENEWORKS_ALLOW_OPEN_BIND`, so even a hand-edited settings.json
/// can't get past the server's own open-bind refusal.
fn decide_api_bind_env(enabled: bool, port: Option<u16>, password: Option<String>) -> ApiBindEnv {
    let loopback = || ApiBindEnv {
        host: "127.0.0.1",
        // OS-assigned free port (no reserve/release TOCTOU); discovered from the log.
        port: "0".to_owned(),
        access_token: None,
        fixed_port: None,
        warning: None,
    };
    if !enabled {
        return loopback();
    }
    let password = password
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let Some(password) = password else {
        return ApiBindEnv {
            warning: Some(
                "remote access is enabled but no password is set — binding loopback-only. \
                 Set a password in Settings → Remote Access to allow LAN connections."
                    .to_owned(),
            ),
            ..loopback()
        };
    };
    let port = port.unwrap_or(crate::settings::DEFAULT_REMOTE_PORT);
    ApiBindEnv {
        host: "0.0.0.0",
        port: port.to_string(),
        access_token: Some(password),
        fixed_port: Some(port),
        warning: None,
    }
}

/// The API access token for this launch when LAN remote access is on (the user's
/// password), or `None` for the default loopback mode. Used to authenticate the GPU
/// worker(s) to the now-protected API; mirrors the fail-closed rule in
/// [`decide_api_bind_env`] (enabled AND a non-empty password). Returns `None` without
/// touching the keychain when remote access is disabled (the password read self-gates
/// on the `remote_password_set` metadata).
fn lan_access_token() -> Option<String> {
    if !crate::settings::load_settings().remote_access_enabled {
        return None;
    }
    crate::settings::read_remote_password()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Spawn the API sidecar, pipe its output to api.log, and return the chosen port.
fn spawn_api(app: &AppHandle) -> Result<(), String> {
    // epic 4484 stories 2/3: pick loopback/dynamic (default) vs 0.0.0.0/fixed-port +
    // password-as-access-token (LAN) from the persisted settings, fail-closed.
    let settings = crate::settings::load_settings();
    let bind = decide_api_bind_env(
        settings.remote_access_enabled,
        settings.remote_port,
        crate::settings::read_remote_password(),
    );
    if let Some(warning) = &bind.warning {
        append_log(
            &logs_dir().join("api.log"),
            &format!("[desktop] {warning}\n"),
        );
    }
    let hf_home = huggingface_home().to_string_lossy().into_owned();
    let mut command = app
        .shell()
        .sidecar("sceneworks-api")
        .map_err(|error| format!("locate api: {error}"))?
        // Loopback/dynamic by default; 0.0.0.0/fixed-port in LAN mode (epic 4484).
        .env("SCENEWORKS_API_HOST", bind.host)
        .env("SCENEWORKS_API_PORT", &bind.port)
        // Epic 4484: in LAN mode the password becomes the API access token, but the
        // embedded desktop UI and local GPU worker(s) reach the API over loopback with no
        // password. Trust loopback peers so local use stays password-free while LAN
        // callers stay gated. A no-op in loopback-only mode (no token is set). Never set
        // by Docker/server, so a reverse-proxied deployment stays fail-closed.
        .env("SCENEWORKS_TRUST_LOOPBACK", "true")
        .env("SCENEWORKS_RUN_UTILITY_INPROCESS", "true")
        // Parent-death watchdog: a force-quit/crash skips `begin_shutdown`, so
        // without this the API orphans to launchd (PPID=1), holding its
        // OS-assigned port + a jobs.db handle. The API self-terminates when this
        // PID disappears; unset (server/Docker) -> the watchdog never fires.
        .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
        .env(
            "SCENEWORKS_DATA_DIR",
            resolved_data_dir().to_string_lossy().to_string(),
        )
        // Pin the config dir so the API and worker share one root on all
        // platforms (Linux otherwise splits XDG data vs config).
        .env(
            "SCENEWORKS_CONFIG_DIR",
            config_dir().to_string_lossy().to_string(),
        );
    // The catalog's install-state detection resolves the HF cache from these;
    // they must match the worker's download root or every model reads "missing".
    command = inject_huggingface_cache_env(command, &hf_home);
    // Epic 3482 (Python Eradication) final cutover (sc-3492) — macOS runs MLX-only.
    // `Settings.mlx_required` ← `SCENEWORKS_MLX_REQUIRED` (sc-3483): the MPS/torch worker
    // never claims an MLX-eligible job, and an MLX-eligible job that no live `mlx` worker
    // takes fails `mlx_unavailable` instead of falling back to MPS. Every Mac Python
    // *inference* surface is now ported to the in-process Rust/MLX worker or UI-gated
    // (sc-3486), and the Python torch worker is no longer spawned on macOS (see
    // `gate_window`), so the flag is enforced here.
    #[cfg(target_os = "macos")]
    {
        command = command.env("SCENEWORKS_MLX_REQUIRED", "1");
    }
    // Off-Mac (epic 5483 Phase 7, sc-5563): candle is the ONLY backend on the desktop —
    // the Python torch worker is no longer spawned (see `gate_window`) and no venv is
    // bundled or bootstrapped. Mirror the Mac MLX-required flip: require candle so a
    // candle-eligible job stranded with no live candle worker fails `candle_unavailable`,
    // and enforce so a shape candle can't serve fails `candle_unsupported` — never a silent
    // torch fallback (there is no torch worker left to fall back to). The candle sweeps are
    // biased to Ok, so only the true generation gaps fail; the CV-aux / segment / training
    // surfaces stay served by the candle worker.
    #[cfg(not(target_os = "macos"))]
    {
        command = command
            .env("SCENEWORKS_CANDLE_REQUIRED", "1")
            .env("SCENEWORKS_CANDLE_UNSUPPORTED_MODE", "enforce");
    }
    // The in-process utility worker shells out to ffmpeg; point it at the bundled
    // static binary (sc-3767) since the desktop has no system ffmpeg on PATH.
    if let Some(ffmpeg) = resolve_bundled_ffmpeg(app) {
        command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
    }
    // DWPose pose detection (sc-3487) loads onnxruntime dynamically; point `ort` at
    // the bundled CoreML-enabled dylib so a packaged Python-free Mac can detect poses.
    #[cfg(target_os = "macos")]
    if let Some(ort_dylib) = resolve_bundled_onnxruntime(app) {
        command = command.env("ORT_DYLIB_PATH", ort_dylib);
    }
    // MLX loads its Metal shader library (mlx.metallib) at runtime; point the pmetal
    // resolver at the bundled copy so a packaged Mac (no build tree, no
    // ~/.cache/pmetal) finds it instead of failing "Failed to load the default
    // metallib" (sc-10349). The API sidecar's in-process utility worker touches MLX
    // too (e.g. the native YOLO11 person detector), so it needs this — not just the
    // separately-spawned MLX GPU worker below.
    #[cfg(target_os = "macos")]
    if let Some(metallib) = resolve_bundled_metallib(app) {
        command = command.env("PMETAL_METALLIB_PATH", metallib);
    }
    // The candle (Windows/CUDA) worker's cudarc dynamic-linking `LoadLibrary`s the
    // CUDA runtime DLLs by name; prepend the bundled redist dir to the sidecar's
    // PATH so they resolve without a CUDA Toolkit on the machine (sc-5560). No-op on
    // a plain build — the resolver returns None when only the placeholder is staged.
    #[cfg(target_os = "windows")]
    if let Some(cuda_dir) = resolve_bundled_cuda_dir(app) {
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![cuda_dir];
        paths.extend(std::env::split_paths(&existing));
        if let Ok(joined) = std::env::join_paths(paths) {
            command = command.env("PATH", joined);
        }
    }
    #[cfg(target_os = "linux")]
    if let Some(runtime) = linux_candle_runtime() {
        command = inject_linux_candle_runtime_env(command, &runtime);
    }
    // LAN mode (epic 4484): hand the API the user's password as the access token so it
    // requires auth on the now-network-reachable bind. The server ALSO refuses any
    // non-loopback bind without a token (API-001 gate), so this is what unlocks the
    // 0.0.0.0 bind — and we deliberately never set SCENEWORKS_ALLOW_OPEN_BIND, leaving
    // that server gate as the backstop. The in-process utility worker inherits this
    // env; the separately-spawned GPU worker(s) get it via `lan_access_token()` below.
    if let Some(token) = &bind.access_token {
        command = command.env("SCENEWORKS_ACCESS_TOKEN", token);
    }
    // FLUX.2-klein true_v2 single-file conversion is now in-process Rust/MLX
    // (mlx_gen_flux2::convert_and_assemble, sc-3136) — no sidecar venv / converter
    // script, so no SCENEWORKS_MLX_FLUX_* env wiring.
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("spawn api: {error}"))?;
    record_api_pid(app, child.pid());
    // sc-11946: confine the API sidecar (and every process it later spawns) to a kill-on-close
    // Job Object, so the whole subtree dies with the desktop and no orphan can pin the API port
    // on the next launch.
    #[cfg(windows)]
    sidecar_job::confine(child.pid());
    app.state::<Managed>()
        .api
        .lock()
        .expect("api lock")
        .replace(child);
    // In LAN mode the bound port is known up-front (fixed), and the API logs it as
    // `0.0.0.0:<port>` rather than the `127.0.0.1:` marker the stdout parser keys on —
    // so seed it directly. Window-gating + the loopback health check then proceed
    // without depending on the marker (0.0.0.0 includes 127.0.0.1).
    if let Some(fixed_port) = bind.fixed_port {
        *app.state::<Managed>()
            .api_port
            .lock()
            .expect("api port lock") = Some(fixed_port);
    }

    let log_path = logs_dir().join("api.log");
    let _ = std::fs::create_dir_all(logs_dir());
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .ok();
        while let Some(event) = events.recv().await {
            let entry = match event {
                CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    // Discover the OS-assigned port from the API's startup line.
                    {
                        let state = app_handle.state::<Managed>();
                        let mut port = state.api_port.lock().expect("api port lock");
                        if port.is_none() {
                            if let Some(found) = parse_listening_port(&text) {
                                *port = Some(found);
                            }
                        }
                    }
                    // Remote worker-restart sentinel (epic 4484 story 12): the API
                    // prints this when an authenticated remote admin POSTs
                    // /api/v1/worker/restart. Do the same kill-and-respawn as the local
                    // "Restart worker" command.
                    if text.contains(sceneworks_core::WORKER_RESTART_SENTINEL) {
                        restart_gpu_worker(&app_handle);
                    }
                    text
                }
                CommandEvent::Terminated(payload) => {
                    let line = format!(
                        "[desktop] api sidecar terminated: code={:?} signal={:?}\n",
                        payload.code, payload.signal
                    );
                    handle_api_exit(&app_handle);
                    line
                }
                CommandEvent::Error(error) => {
                    let line = format!("[desktop] api sidecar error: {error}\n");
                    handle_api_exit(&app_handle);
                    line
                }
                _ => continue,
            };
            if let Some(file) = file.as_mut() {
                let _ = file.write_all(entry.as_bytes());
                let _ = file.flush();
            }
            // Mirror the API sidecar's output into the in-app session buffer (sc-3451);
            // this loop writes its own file handle so it doesn't go through append_log.
            session_log().push_line("api", &entry);
        }
    });
    Ok(())
}

/// Handle an unexpected exit of the API sidecar (F-128, sc-8930). The API is spawned
/// once and `run_startup`'s `Managed.api.is_some()` guard blocks a re-spawn; the GPU
/// workers self-supervise with backoff but the API did not, so if it crashed mid-session
/// the webview pointed at a dead origin with the "Retry" button inert (its `start_setup`
/// re-entry short-circuited on the still-populated `api` slot).
///
/// Clear the `Managed.api` slot + the recorded PID so the guard now sees `None` and a
/// Retry re-spawns the API, and surface a setup `error` event so the UI shows the
/// recoverable error screen instead of a silently-dead window. No-op during a graceful
/// shutdown (the child was killed on purpose) — see `begin_shutdown`, which `take()`s the
/// child before this reader observes the resulting `Terminated`.
///
/// Also recycle the GPU worker (sc-13605): the MLX/candle worker does NOT exit when its
/// API becomes unreachable — its poll loop just logs the connection error and retries the
/// dead port forever (`run_worker_loop`'s generic-error arm). Left alone, the single
/// supervisor stays blocked on that zombie's event stream and never respawns it, so after
/// a Retry the worker would still point at the dead original port. Killing the child here
/// (after clearing the port) breaks the supervisor's inner loop; its next
/// `resolve_supervisor_action` reads `WaitForPort` and parks until the Retry's fresh
/// `spawn_api` publishes the NEW port, then respawns the worker against it. Ordering
/// matters: clear `api_port` FIRST so the supervisor can never re-read the stale port
/// between the kill and the respawn. (The supervisor's own post-store recheck closes the
/// residual window where this kill races ahead of the child being stored.)
fn handle_api_exit(app: &AppHandle) {
    if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
        return;
    }
    // Drop the child handle + its recorded PID so `run_startup`'s spawn-once guard
    // re-arms and a Retry actually re-spawns the API.
    let _ = app.state::<Managed>().api.lock().expect("api lock").take();
    record_api_pid_cleared(app);
    // Forget the discovered port so window-gating re-derives it from the fresh spawn.
    *app.state::<Managed>()
        .api_port
        .lock()
        .expect("api port lock") = None;
    // Recycle the now-orphaned GPU worker so the single supervisor (sc-13605) re-reads the
    // Retry's fresh port instead of churning against the dead one. `restart_gpu_worker`
    // (sc-13584) kills whichever child occupies the slot — the same one the sole supervisor
    // manages — and is a no-op off macOS/Windows and when the slot is already empty.
    restart_gpu_worker(app);
    emit(
        app,
        "error",
        "The local API stopped unexpectedly. Click Retry to restart it.",
        true,
    );
}

/// Health-gate the window on a background thread: wait for the API's
/// OS-assigned port, confirm the responder is genuinely SceneWorks, then
/// navigate and start the platform inference worker(s) — the MLX GPU worker on
/// macOS (MLX-only, sc-3492), the Python torch worker elsewhere; show an error
/// after the timeout.
fn gate_window(app: AppHandle) {
    std::thread::spawn(move || {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        loop {
            let port = *app
                .state::<Managed>()
                .api_port
                .lock()
                .expect("api port lock");
            if let Some(port) = port {
                if health_is_sceneworks(port) {
                    if let Ok(url) = format!("http://127.0.0.1:{port}").parse() {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.navigate(url);
                        }
                    }
                    #[cfg(target_os = "macos")]
                    {
                        debug_assert_eq!(
                            select_worker_supervisor(current_desktop_platform(), false),
                            WorkerSupervisor::Mlx
                        );
                        // Epic 3482 final cutover (sc-3492): macOS is MLX-only — the
                        // Python torch/MPS worker is no longer spawned. Only the
                        // Apple-Silicon MLX GPU worker (sc-3289) runs, executing
                        // MLX-eligible image/video jobs on the in-process Rust mlx-gen
                        // engine. Any MLX-ineligible job fails `mlx_unsupported` /
                        // `mlx_unavailable` (never MPS) per `Settings.mlx_required`.
                        //
                        // Start the on-demand credential socket first (sc-5891) so the
                        // worker can pull a recorded keychain secret lazily at download
                        // time instead of us reading it eagerly here at launch.
                        ensure_cred_ipc(&app);
                        // sc-13605: self-guarded — spawns a supervisor only if
                        // one isn't already live, so an API-crash → Retry (which
                        // re-runs gate_window) never stacks a second supervisor
                        // on the single worker slot. The live supervisor re-reads
                        // the current port, so it repoints at the fresh API.
                        supervise_mlx_worker(app);
                    }
                    #[cfg(any(target_os = "windows", target_os = "linux"))]
                    {
                        // Epic 5483 Phase 7 (sc-5563): off-Mac is candle-only — the Python
                        // torch worker is no longer spawned (its venv + bundle were dropped),
                        // exactly as macOS went MLX-only in sc-3492. The Windows/Linux candle
                        // (CUDA)
                        // GPU worker runs the candle-eligible surface; anything candle can't
                        // serve fails loudly (candle_unsupported / candle_unavailable) per
                        // Settings.candle_required (set in spawn_api), never a silent torch
                        // fallback. Spawned only when the native runtime is present; before
                        // provisioning completes there is no GPU supervisor to crash-loop.
                        let runtime_present = candle_runtime_present(&app);
                        if matches!(
                            select_worker_supervisor(current_desktop_platform(), runtime_present),
                            WorkerSupervisor::Candle
                        ) {
                            // sc-13605: self-guarded, same as the macOS branch —
                            // one supervisor per worker slot, re-reads the port.
                            supervise_candle_worker(app);
                        } else {
                            append_log(
                                &logs_dir().join("candle-worker.log"),
                                "[desktop] candle worker dormant: native CUDA/cuDNN/onnxruntime \
                                 runtime is not provisioned\n",
                            );
                        }
                    }
                    return;
                }
            }
            if Instant::now() >= deadline {
                emit(&app, "error", "The local API did not start in time.", true);
                return;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    });
}

/// Start the on-demand credential socket (sc-5891) once and stash it in `Managed`.
/// The MLX worker is handed its socket path + token at spawn and pulls a recorded
/// keychain secret from it the first time a download needs auth — so the keychain is
/// read lazily, not eagerly at launch. Idempotent; a start failure is logged and the
/// worker simply gets no credentials (a gated download then fails with an auth error
/// rather than the app prompting at launch).
#[cfg(target_os = "macos")]
fn ensure_cred_ipc(app: &AppHandle) {
    let managed = app.state::<Managed>();
    let mut slot = managed.cred_ipc.lock().expect("cred_ipc lock");
    if slot.is_some() {
        return;
    }
    let socket = app_support_dir().join("cred-ipc.sock");
    match crate::cred_ipc::start(socket) {
        Some(handle) => *slot = Some(handle),
        None => append_log(
            &logs_dir().join("mlx-worker.log"),
            "[desktop] credential socket failed to start; gated downloads will need a re-entered token\n",
        ),
    }
}

/// Drop a host's cached secret from the credential socket (sc-5891) so a later pull
/// re-reads the keychain. Called when the user updates or removes a credential, so a
/// revoked/changed token stops being served without an app restart. No-op off macOS
/// (no socket there).
pub fn invalidate_credential_cache(app: &AppHandle, host: &str) {
    #[cfg(target_os = "macos")]
    {
        if let Some(ipc) = app
            .state::<Managed>()
            .cred_ipc
            .lock()
            .expect("cred_ipc lock")
            .as_ref()
        {
            ipc.invalidate(host);
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, host);
    }
}

/// Kill the current GPU worker child so its supervisor respawns it — the shared core of
/// the local "Restart worker" Tauri command and the remote REST restart (epic 4484
/// story 12, triggered when the API prints `WORKER_RESTART_SENTINEL` to stdout). macOS
/// runs the MLX worker; Windows/Linux run the candle worker.
pub fn restart_gpu_worker(app: &AppHandle) {
    #[cfg(target_os = "macos")]
    let child = app
        .state::<Managed>()
        .mlx_worker
        .lock()
        .expect("mlx worker lock")
        .take();
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    let child = app
        .state::<Managed>()
        .candle_worker
        .lock()
        .expect("candle worker lock")
        .take();
    #[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
    if let Some(child) = child {
        // macOS runs a single MLX worker process — CommandChild::kill() (TerminateProcess
        // on Windows / a plain kill here) reaps it fully. Off-Mac runs the candle `auto`
        // supervisor, which spawns one child per GPU plus a CPU child, each inheriting
        // SCENEWORKS_PARENT_PID = this desktop PID. Unlike a quit, a restart leaves the
        // desktop alive, so plain-killing only the supervisor orphans those children:
        // their parent-death watchdog still sees the live desktop and never fires, and the
        // respawned supervisor spawns a duplicate set contending for the same GPUs and
        // worker IDs. Use the platform PID teardown: taskkill /T /F on Windows; SIGTERM
        // on Linux, where the supervisor handles it by stopping and reaping its children.
        #[cfg(target_os = "windows")]
        kill_pid(child.pid());
        #[cfg(target_os = "linux")]
        terminate_linux_candle_tree(child.pid());
        #[cfg(target_os = "macos")]
        let _ = child.kill();
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    let _ = app;
}

/// Per-spawn context handed to a worker's `env_builder` (sc-13615). Bundles the
/// values the skeleton computes each iteration that the env block needs: the app
/// handle (for the bundled-resource / credential lookups), the CURRENT API url
/// (re-read every attempt, never captured — sc-13605), the per-launch worker id,
/// and the shared HF cache root. Passed by value (all borrows) so the closure
/// bound stays a simple `for<'a> Fn(Command, WorkerSpawnCtx<'a>)`.
#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
struct WorkerSpawnCtx<'a> {
    app: &'a AppHandle,
    api_url: &'a str,
    worker_id: &'a str,
    hf_home: &'a str,
}

/// Shared spawn-and-supervise skeleton for the native GPU worker (F-053,
/// sc-13615). `supervise_mlx_worker` (macOS/MLX) and `supervise_candle_worker`
/// (Windows/candle) were ~230-line near-verbatim twins; the only real
/// differences are the child's env block, which `Managed` slot + pidfile field
/// holds the child, the log/id naming, and how a stale child is killed. This owns
/// everything else so a backoff/restart/logging change (or the sc-13605
/// lease/seam/recheck fix) lands once, in one place.
///
/// Behavior is identical to the old pair: the sole-supervisor lease, the
/// per-iteration `resolve_supervisor_action` port re-read, the post-store
/// correlated-death recheck + KillAndRetry throttle, the pipe pump, and the
/// exponential backoff are unchanged — only the differing axes are parameterized:
/// * `label` — names the log file (`<label>-worker.log`) and every log line.
/// * `id_prefix` — the jobs.db worker-id prefix (`<id_prefix>-<pid>-<millis>`).
/// * `slot` — the `Managed` mutex that holds this worker's child.
/// * `record_pid` — persists the child PID to the pidfile.
/// * `kill_stale` — how the post-store recheck kills a stale child: MLX kills the
///   single child directly (`CommandChild::kill`); candle uses platform PID
///   teardown so the `auto` supervisor also stops its per-GPU children (Windows
///   tree-kill; Linux supervisor-handled SIGTERM).
/// * `env_builder` — builds the child's env block from the located sidecar.
#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
fn supervise_worker(
    app: AppHandle,
    label: &'static str,
    id_prefix: &'static str,
    slot: fn(&Managed) -> &Mutex<Option<CommandChild>>,
    record_pid: fn(&AppHandle, Option<u32>),
    kill_stale: fn(CommandChild),
    env_builder: impl Fn(Command, WorkerSpawnCtx) -> Result<Command, String> + Send + 'static,
) {
    std::thread::spawn(move || {
        // sc-13605: exactly one supervisor owns the single worker slot. If one is
        // already live (e.g. an API-crash → Retry re-ran gate_window), don't stack
        // a second — return, letting the live supervisor keep the slot and re-read
        // the current port. The lease clears the liveness flag on every exit path
        // below (return / break-out / panic).
        let Some(_lease) = SupervisorLease::acquire(&app) else {
            return;
        };
        let log_path = logs_dir().join(format!("{label}-worker.log"));
        // Match the API sidecar's HF cache root so the engine reads the same
        // downloaded weights the catalog tracks.
        let hf_home = huggingface_home().to_string_lossy().to_string();
        // Unique per launch (distinct prefix from the Python `worker-local-*` and
        // the in-process `rust-utility-worker`) so the workers never collide in the
        // shared jobs.db.
        let worker_id = format!(
            "{id_prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default()
        );
        let mut backoff = 1u64;
        loop {
            // sc-13605: re-read shutdown + the CURRENT API url every iteration
            // (never a URL captured at spawn) through the shared
            // `resolve_supervisor_action` seam, so after an API-crash → Retry the
            // worker points at the NEW port. Park (not exit) while no port is
            // published; exit only on shutdown.
            let api_url = match resolve_supervisor_action(&app.state::<Managed>()) {
                SupervisorAction::Exit => return,
                SupervisorAction::WaitForPort => {
                    std::thread::sleep(Duration::from_millis(300));
                    continue;
                }
                SupervisorAction::Spawn(url) => url,
            };
            let sidecar = match app.shell().sidecar("sceneworks-api") {
                Ok(command) => command,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] {label} worker: locate sidecar failed: {error}\n"),
                    );
                    return;
                }
            };
            // The ONLY substantive per-worker difference: build the child's env
            // block. Everything around it (spawn, PID record, recheck, backoff) is
            // identical across workers.
            let command = match env_builder(
                sidecar,
                WorkerSpawnCtx {
                    app: &app,
                    api_url: &api_url,
                    worker_id: &worker_id,
                    hf_home: &hf_home,
                },
            ) {
                Ok(command) => command,
                Err(error) => {
                    // Linux Secret Service reads are deliberately fallible. Never
                    // launch without recorded credentials: preserve the actionable
                    // error, then retry so unlocking the service recovers without an
                    // orphaned credential-less worker.
                    append_log(
                        &log_path,
                        &format!("[desktop] {label} worker environment setup failed: {error}\n"),
                    );
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            let spawned = command.spawn();
            let (mut events, child) = match spawned {
                Ok(pair) => pair,
                Err(error) => {
                    append_log(
                        &log_path,
                        &format!("[desktop] {label} worker spawn failed: {error}\n"),
                    );
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            record_pid(&app, Some(child.pid()));
            slot(&app.state::<Managed>())
                .lock()
                .expect("worker lock")
                .replace(child);
            // sc-13605 correlated-death guard: re-read the target AFTER storing the
            // child. If the API port changed/cleared in the window between the read
            // above and this store, `handle_api_exit`'s `restart_gpu_worker` kill may
            // have run while the slot was still empty and missed this child — so the
            // worker would sit pointed at a dead port and the supervisor would block
            // on its (never-terminating) event stream. Kill+retry (or exit) instead.
            let verdict = verify_spawned_target(
                &api_url,
                &resolve_supervisor_action(&app.state::<Managed>()),
            );
            if !matches!(verdict, SpawnVerdict::Keep) {
                // Take the stale child out of the slot and DROP the guard before the
                // kill (mirrors `restart_gpu_worker`) so the worker mutex is never
                // held across the kill and a concurrent handle_api_exit isn't blocked.
                // `kill_stale` preserves each worker's kill semantics (MLX direct,
                // candle tree-kill).
                let stale = slot(&app.state::<Managed>())
                    .lock()
                    .expect("worker lock")
                    .take();
                if let Some(child) = stale {
                    kill_stale(child);
                }
                record_pid(&app, None);
                if matches!(verdict, SpawnVerdict::KillAndExit) {
                    return;
                }
                append_log(
                    &log_path,
                    &format!(
                        "[desktop] {label} worker target port changed before start; respawning\n"
                    ),
                );
                // sc-13605: brief defensive throttle so a pathological rapid
                // crash-loop can't hot-spin on spawns via this recheck path, which
                // `continue`s past the bottom-of-loop exponential backoff. Short
                // fixed delay (the minimum backoff), NOT the exponential.
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
            let started = Instant::now();
            loop {
                match tauri::async_runtime::block_on(events.recv()) {
                    Some(CommandEvent::Stdout(bytes)) | Some(CommandEvent::Stderr(bytes)) => {
                        append_log(&log_path, &String::from_utf8_lossy(&bytes));
                    }
                    Some(CommandEvent::Terminated(payload)) => {
                        append_log(
                            &log_path,
                            &format!(
                                "[desktop] {label} worker terminated: code={:?} signal={:?}\n",
                                payload.code, payload.signal
                            ),
                        );
                        break;
                    }
                    Some(CommandEvent::Error(error)) => {
                        append_log(
                            &log_path,
                            &format!("[desktop] {label} worker error: {error}\n"),
                        );
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            let _ = slot(&app.state::<Managed>())
                .lock()
                .expect("worker lock")
                .take();
            record_pid(&app, None);
            if app.state::<Managed>().shutting_down.load(Ordering::SeqCst) {
                return;
            }
            if started.elapsed() > Duration::from_secs(20) {
                backoff = 1;
            }
            append_log(
                &log_path,
                &format!("[desktop] restarting {label} worker in {backoff}s\n"),
            );
            std::thread::sleep(Duration::from_secs(backoff));
            backoff = (backoff * 2).min(30);
        }
    });
}

/// The `Managed` slot holding the MLX worker child (sc-13615). A named fn so it
/// coerces to the `fn(&Managed) -> &Mutex<..>` `supervise_worker` takes with the
/// right higher-ranked lifetime.
#[cfg(target_os = "macos")]
fn mlx_worker_slot(managed: &Managed) -> &Mutex<Option<CommandChild>> {
    &managed.mlx_worker
}

/// Spawn and supervise the Apple-Silicon MLX GPU worker (sc-3289): the same
/// `sceneworks-api` sidecar binary re-launched in worker mode
/// (`SCENEWORKS_WORKER_ONLY=1`) with `SCENEWORKS_GPU_ID=mlx`, so MLX-eligible
/// image/video jobs run on the in-process Rust mlx-gen engine instead of the
/// Python torch/MPS path. A crash-isolated sibling of the API process; restarted
/// with exponential backoff while the app is open. Output goes to mlx-worker.log.
///
/// Without this worker registered, `jobs_store::should_defer_image_to_mlx_worker`
/// has nowhere to defer and the Python `mps` worker is the fallback — which is
/// why image/video jobs reported MPS before this landed.
#[cfg(target_os = "macos")]
fn supervise_mlx_worker(app: AppHandle) {
    supervise_worker(
        app,
        "mlx",
        "mlx-worker-local",
        mlx_worker_slot,
        record_mlx_worker_pid,
        // macOS runs a single MLX worker process — CommandChild::kill() (a plain
        // kill here) reaps it fully. There are no auto-spawned children to orphan,
        // so a direct kill is correct (contrast candle's tree-kill).
        |child| {
            let _ = child.kill();
        },
        |sidecar, ctx| {
            let mut command = sidecar
                // Dispatches `main` to `run_worker()` (HTTP API never starts).
                .env("SCENEWORKS_WORKER_ONLY", "1")
                .env("SCENEWORKS_GPU_ID", "mlx")
                .env("SCENEWORKS_WORKER_ID", ctx.worker_id)
                .env("SCENEWORKS_API_URL", ctx.api_url)
                // Parent-death watchdog (run_worker() honours this): a force-quit
                // self-terminates the worker so its multi-GB MLX model isn't
                // orphaned to launchd.
                .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
                .env(
                    "SCENEWORKS_DATA_DIR",
                    resolved_data_dir().to_string_lossy().to_string(),
                )
                .env(
                    "SCENEWORKS_CONFIG_DIR",
                    config_dir().to_string_lossy().to_string(),
                );
            command = inject_huggingface_cache_env(command, ctx.hf_home);
            // sc-7821 (epic 7819): the user's GPU memory ceiling, as fraction × total unified
            // memory. run_worker_loop applies it to the MLX runtime process-globally (covers
            // generations, upscales, AND LoRA training). Absent ⇒ no env ⇒ MLX default budget.
            // Read here at spawn, so a slider change takes effect on the next worker restart.
            if let Some(bytes) = crate::settings::gpu_memory_limit_bytes() {
                command = command.env("SCENEWORKS_GPU_MEMORY_LIMIT_BYTES", bytes.to_string());
            }
            // The worker muxes generated video with ffmpeg; the desktop ships no
            // system ffmpeg, so point it at the bundled binary (as spawn_api does).
            if let Some(ffmpeg) = resolve_bundled_ffmpeg(ctx.app) {
                command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
            }
            // This is the worker that advertises `pose_detect` (epic 3482, sc-3487);
            // point `ort` at the bundled CoreML onnxruntime dylib it dlopens.
            if let Some(ort_dylib) = resolve_bundled_onnxruntime(ctx.app) {
                command = command.env("ORT_DYLIB_PATH", ort_dylib);
            }
            // This is the process that runs MLX generation; point the pmetal resolver
            // at the bundled Metal shader library so a packaged Mac (no build tree, no
            // ~/.cache/pmetal) finds it instead of failing "Failed to load the default
            // metallib" on first MLX use (sc-10349, as spawn_api does).
            if let Some(metallib) = resolve_bundled_metallib(ctx.app) {
                command = command.env("PMETAL_METALLIB_PATH", metallib);
            }
            // Lazy credentials (sc-5891): instead of reading the keychain here and
            // injecting HF_TOKEN/SCENEWORKS_CREDENTIALS (which prompted at launch),
            // hand the worker the credential socket + token + the NON-secret list of
            // recorded hosts. The worker pulls a secret only when a download for a
            // recorded host needs it, so nothing-recorded ⇒ no socket call ⇒ no
            // keychain touch. Credential changes still take effect on worker restart.
            {
                let managed = ctx.app.state::<Managed>();
                let guard = managed.cred_ipc.lock().expect("cred_ipc lock");
                if let Some(ipc) = guard.as_ref() {
                    command = command
                        .env(
                            "SCENEWORKS_CRED_IPC_SOCKET",
                            ipc.socket.to_string_lossy().to_string(),
                        )
                        .env("SCENEWORKS_CRED_IPC_TOKEN", &ipc.token);
                    let hosts = crate::settings::recorded_credential_hosts().join(",");
                    if !hosts.is_empty() {
                        command = command.env("SCENEWORKS_CREDENTIAL_HOSTS", hosts);
                    }
                }
            }
            // LAN mode (epic 4484): the API now requires the password as an access
            // token, so the MLX worker must send it on every API call (register/claim/
            // heartbeat) or it'd be 401'd. `None` in the default loopback mode, so this
            // is a no-op unless the user opted into LAN remote access.
            if let Some(token) = lan_access_token() {
                command = command.env("SCENEWORKS_ACCESS_TOKEN", token);
            }
            Ok(command)
        },
    );
}

/// Spawn and supervise the Windows/Linux candle (CUDA) GPU worker(s)
/// (sc-5561/sc-10375): the same
/// `sceneworks-api` sidecar re-launched in worker mode (`SCENEWORKS_WORKER_ONLY=1`)
/// with `SCENEWORKS_GPU_ID=auto` and the candle backend enabled
/// (`SCENEWORKS_BACKEND_CANDLE_ENABLED=true`), so candle-eligible image/video/caption
/// jobs run on the in-process candle gen-core engines. `auto` makes this process the
/// multi-GPU supervisor: it discovers every NVIDIA GPU and spawns one crash-isolated
/// candle child per GPU (restarted with exponential backoff), so a multi-GPU box uses
/// ALL its GPUs rather than just index 0. A crash-isolated sibling of the API process;
/// output goes to candle-worker.log.
///
/// The off-Mac analogue of `supervise_mlx_worker`, and the desktop twin of the
/// server/Docker candle worker (which also runs `SCENEWORKS_GPU_ID=auto`). Off-Mac is
/// candle-only post-Phase-7 (sc-5563): anything candle can't serve fails loudly
/// (`candle_unsupported`/`candle_unavailable`) rather than a silent torch fallback.
/// Only spawned when the platform-native candle runtime is present.
#[cfg(any(target_os = "windows", target_os = "linux", test))]
fn candle_worker_mode_env() -> [(&'static str, &'static str); 3] {
    [
        ("SCENEWORKS_WORKER_ONLY", "1"),
        ("SCENEWORKS_GPU_ID", "auto"),
        ("SCENEWORKS_BACKEND_CANDLE_ENABLED", "true"),
    ]
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn supervise_candle_worker(app: AppHandle) {
    supervise_worker(
        app,
        "candle",
        "candle-worker-local",
        candle_worker_slot,
        record_candle_worker_pid,
        // Platform PID teardown (as `restart_gpu_worker` does): this is the candle
        // `auto` multi-GPU supervisor, whose per-GPU children must stop with it.
        // Windows tree-kills; Linux SIGTERMs the supervisor, whose signal handler
        // stops and reaps every child. Preserve this vs the MLX direct kill.
        |child| {
            #[cfg(target_os = "windows")]
            kill_pid(child.pid());
            #[cfg(target_os = "linux")]
            terminate_linux_candle_tree(child.pid());
        },
        |sidecar, ctx| {
            // Dispatches `main` to `run_worker()` with `auto`, which runs the
            // multi-GPU supervisor (`supervise_auto_workers`) and advertises the
            // candle image/video capabilities on every discovered GPU.
            let mut command = candle_worker_mode_env()
                .into_iter()
                .fold(sidecar, |command, (name, value)| command.env(name, value))
                .env("SCENEWORKS_WORKER_ID", ctx.worker_id)
                .env("SCENEWORKS_API_URL", ctx.api_url)
                // Parent-death watchdog (run_worker() honours this): a force-quit
                // self-terminates the worker so its multi-GB model + CUDA context
                // isn't orphaned.
                .env("SCENEWORKS_PARENT_PID", std::process::id().to_string())
                .env(
                    "SCENEWORKS_DATA_DIR",
                    resolved_data_dir().to_string_lossy().to_string(),
                )
                .env(
                    "SCENEWORKS_CONFIG_DIR",
                    config_dir().to_string_lossy().to_string(),
                );
            command = inject_huggingface_cache_env(command, ctx.hf_home);
            // cudarc dynamic-linking `LoadLibrary`s the CUDA runtime DLLs by name;
            // prepend the bundled redist dir to this worker's PATH so they resolve
            // without a CUDA Toolkit on the machine (sc-5560).
            #[cfg(target_os = "windows")]
            if let Some(cuda_dir) = resolve_bundled_cuda_dir(ctx.app) {
                let existing = std::env::var_os("PATH").unwrap_or_default();
                let mut paths = vec![cuda_dir.clone()];
                paths.extend(std::env::split_paths(&existing));
                if let Ok(joined) = std::env::join_paths(paths) {
                    command = command.env("PATH", joined);
                }
                // The candle worker's `ort` (onnxruntime) paths — DWPose pose_detect
                // (sc-5496), then YOLO / Real-ESRGAN (sc-5498/5499, epic 5482) — point
                // `ort` at the bundled CUDA-enabled onnxruntime and tell the worker where
                // the CUDA-12 runtime + cuDNN-9 DLLs live, so its CUDA execution provider
                // engages instead of falling back to CPU. The off-Mac analogue of the
                // macOS CoreML `ORT_DYLIB_PATH` wiring. The `cuda` resource dir holds the
                // version-matched CUDA-12 runtime + cuDNN-9 (staged by build-sidecar.mjs);
                // `ort_cuda::preload_cuda_dylibs` preloads them + puts the dir on the
                // loader search path so cuDNN's lazily-loaded sub-engine DLLs resolve.
                if let Some(ort_dylib) = resolve_bundled_onnxruntime(ctx.app) {
                    let cuda = cuda_dir.to_string_lossy().to_string();
                    command = command
                        .env("ORT_DYLIB_PATH", ort_dylib)
                        .env("SCENEWORKS_ORT_CUDA_DIR", &cuda)
                        .env("SCENEWORKS_ORT_CUDNN_DIR", &cuda);
                }
            }
            // Linux uses the same candle/ORT contract, with the dynamic linker's
            // LD_LIBRARY_PATH in place of the Windows DLL-search PATH.
            #[cfg(target_os = "linux")]
            if let Some(runtime) = linux_candle_runtime() {
                command = inject_linux_candle_runtime_env(command, &runtime);
            }
            // The worker muxes generated video with ffmpeg; point it at the bundled
            // binary when staged (else it falls back to PATH ffmpeg), as spawn_api does.
            if let Some(ffmpeg) = resolve_bundled_ffmpeg(ctx.app) {
                command = command.env("SCENEWORKS_FFMPEG", ffmpeg);
            }
            if let Some(token) = crate::settings::read_hf_token()? {
                command = command.env("HF_TOKEN", token);
            }
            if let Some(credentials) = crate::settings::credentials_env_json()? {
                command = command.env("SCENEWORKS_CREDENTIALS", credentials);
            }
            // LAN mode (epic 4484): the API now requires the password as an access
            // token, so the candle worker must send it on every API call or it'd be
            // 401'd. `None` in the default loopback mode (no-op unless LAN is on).
            if let Some(token) = lan_access_token() {
                command = command.env("SCENEWORKS_ACCESS_TOKEN", token);
            }
            Ok(command)
        },
    );
}

/// The `Managed` slot holding the candle worker child (sc-13615). Named fn for
/// the same higher-ranked-lifetime coercion reason as `mlx_worker_slot`.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn candle_worker_slot(managed: &Managed) -> &Mutex<Option<CommandChild>> {
    &managed.candle_worker
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
}

/// File holding this launch's sidecar PIDs, used to reap orphans left by a prior
/// unclean exit. Linux treats it as mutable state under `XDG_STATE_HOME`; other
/// platforms retain the existing application-support location.
fn sidecar_pidfile() -> PathBuf {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        linux_desktop_paths().sidecar_pidfile()
    }
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    {
        app_support_dir().join("desktop-sidecars.json")
    }
}

/// Persist the current sidecar PIDs (best effort, atomic via temp+rename).
fn write_sidecar_pidfile(pids: &SidecarPids) {
    let path = sidecar_pidfile();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(json) = serde_json::to_vec_pretty(pids) else {
        return;
    };
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &json).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

fn record_api_pid(app: &AppHandle, pid: u32) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.api = Some(pid);
    write_sidecar_pidfile(&pids);
}

/// Windows sidecar-tree containment (sc-11946).
///
/// The API sidecar can spawn child processes. If the desktop dies without cleanly reaping
/// that subtree — a clean API self-exit orphans it, and a force-quit races the `taskkill /T`
/// walk — an orphaned child keeps the API's listening socket handle it inherited, pinning the
/// port so the next launch fails to bind ("local API stopped unexpectedly" / AddrInUse).
///
/// We create ONE process-lifetime Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and
/// assign the API sidecar to it. Descendants a job member spawns join the job automatically
/// (nested jobs, Win8+), so the whole API subtree is in it. When the
/// desktop process ends for ANY reason — graceful exit, panic, or hard kill — the OS closes
/// our handle and terminates every process still in the job. Nothing can be orphaned. This is
/// the OS-enforced backstop the pidfile `taskkill /T` reaping ([`kill_pid`]) can't guarantee
/// once the tracked parent has already exited.
///
/// Best-effort: any failure is logged and left to the existing reaping, never fatal.
#[cfg(windows)]
mod sidecar_job {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, FALSE, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    /// A raw job handle held for the process's whole life. `HANDLE` is a raw pointer and thus
    /// not `Send`/`Sync`; a job handle is safe to share across threads, so we assert it.
    struct SharedJob(HANDLE);
    // SAFETY: a Windows job-object handle is a plain kernel handle with no thread affinity.
    unsafe impl Send for SharedJob {}
    unsafe impl Sync for SharedJob {}

    /// The one kill-on-close job. Never dropped explicitly — the handle closes when the
    /// process exits, which is exactly when we want the job's KILL_ON_JOB_CLOSE to fire.
    static JOB: OnceLock<SharedJob> = OnceLock::new();

    fn job_handle() -> Option<HANDLE> {
        let raw = JOB.get_or_init(|| SharedJob(create())).0;
        (!raw.is_null()).then_some(raw)
    }

    fn create() -> HANDLE {
        // SAFETY: FFI to the Job Object APIs. The limit-info struct is fully zeroed and only
        // the documented KILL_ON_JOB_CLOSE flag is set; on any failure the handle is closed and
        // a null handle returned so callers degrade to the existing taskkill reaping.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() || job == INVALID_HANDLE_VALUE {
                return std::ptr::null_mut();
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                CloseHandle(job);
                return std::ptr::null_mut();
            }
            job
        }
    }

    /// Assign sidecar process `pid` to the shared kill-on-close job.
    pub(super) fn confine(pid: u32) {
        let Some(job) = job_handle() else {
            tracing::warn!(
                event = "sidecar_job_unavailable",
                pid,
                "could not create kill-on-close job; relying on taskkill reaping"
            );
            return;
        };
        // SAFETY: OpenProcess with exactly the rights AssignProcessToJobObject needs; the
        // process handle is closed immediately after the (single) assignment call.
        unsafe {
            let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, pid);
            if process.is_null() {
                tracing::warn!(event = "sidecar_job_open_failed", pid, "OpenProcess failed");
                return;
            }
            let assigned = AssignProcessToJobObject(job, process) != 0;
            CloseHandle(process);
            if assigned {
                tracing::info!(
                    event = "sidecar_job_assigned",
                    pid,
                    "API sidecar confined to kill-on-close job"
                );
            } else {
                tracing::warn!(
                    event = "sidecar_job_assign_failed",
                    pid,
                    "assign to job failed"
                );
            }
        }
    }
}

/// Clear the recorded API PID after the sidecar exits unexpectedly (F-128, sc-8930), so
/// the next launch's `reap_stale_sidecars` doesn't try to reap a PID that already died
/// (or, worse, a recycled one). Paired with clearing the in-memory `Managed.api` slot so
/// a Retry re-spawns the API.
fn record_api_pid_cleared(app: &AppHandle) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.api = None;
    write_sidecar_pidfile(&pids);
}

#[cfg(target_os = "macos")]
fn record_mlx_worker_pid(app: &AppHandle, pid: Option<u32>) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.mlx_worker = pid;
    write_sidecar_pidfile(&pids);
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn record_candle_worker_pid(app: &AppHandle, pid: Option<u32>) {
    let state = app.state::<Managed>();
    let mut pids = state.pids.lock().expect("pids lock");
    pids.candle_worker = pid;
    write_sidecar_pidfile(&pids);
}

/// True when `pid` is one of our sidecars (not a recycled, unrelated PID). The
/// command line must reference the API binary. The native GPU worker
/// (`sceneworks-rust-worker`) exits on its own when its parent/API is gone, so
/// only the API sidecar needs identity-checked reaping.
#[cfg(unix)]
fn is_our_sidecar(pid: u32) -> bool {
    let Ok(output) = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let command = String::from_utf8_lossy(&output.stdout);
    command.contains("sceneworks-api")
}

#[cfg(windows)]
fn is_our_sidecar(pid: u32) -> bool {
    // tasklist exposes the image name (sceneworks-api.exe) but not arguments; the
    // native GPU worker exits on its own when its parent/API is gone, so only the
    // API needs reaping on Windows.
    let Ok(output) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    else {
        return false;
    };
    String::from_utf8_lossy(&output.stdout).contains("sceneworks-api")
}

#[cfg(unix)]
fn worker_shutdown_grace_seconds() -> u64 {
    std::env::var("SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(10)
        .clamp(1, 30)
}

/// Direct children of every thread in a Linux process. A Tokio worker thread can
/// call `Command::spawn`, and Linux records that child in the spawning thread's
/// `/proc/<pid>/task/<tid>/children` file rather than necessarily in the thread
/// group leader's file, so every task must be inspected.
#[cfg(any(target_os = "linux", test))]
fn linux_direct_children(proc_root: &Path, parent: u32) -> Vec<u32> {
    let task_dir = proc_root.join(parent.to_string()).join("task");
    let Ok(tasks) = std::fs::read_dir(task_dir) else {
        return Vec::new();
    };
    let mut task_ids = tasks
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .collect::<Vec<_>>();
    task_ids.sort_unstable();

    let mut children = Vec::new();
    for task_id in task_ids {
        let path = proc_root
            .join(parent.to_string())
            .join("task")
            .join(task_id.to_string())
            .join("children");
        if let Ok(value) = std::fs::read_to_string(path) {
            children.extend(
                value
                    .split_whitespace()
                    .filter_map(|value| value.parse::<u32>().ok()),
            );
        }
    }
    children.sort_unstable();
    children.dedup();
    children
}

/// Linux `/proc` descendant snapshot in post-order (deepest children first).
/// Candle children watch the desktop PID, not the supervisor PID, so killing
/// only the supervisor while the desktop stays alive would orphan them.
#[cfg(any(target_os = "linux", test))]
fn linux_process_descendants_from(proc_root: &Path, root: u32) -> Vec<u32> {
    fn visit(
        proc_root: &Path,
        parent: u32,
        seen: &mut std::collections::HashSet<u32>,
        out: &mut Vec<u32>,
    ) {
        for child in linux_direct_children(proc_root, parent) {
            if seen.insert(child) {
                visit(proc_root, child, seen, out);
                out.push(child);
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut descendants = Vec::new();
    visit(proc_root, root, &mut seen, &mut descendants);
    descendants
}

#[cfg(target_os = "linux")]
fn linux_process_descendants(root: u32) -> Vec<u32> {
    linux_process_descendants_from(Path::new("/proc"), root)
}

/// Gracefully stop a Linux candle supervisor and every descendant, then
/// SIGKILL any survivor after the configured worker grace period.
#[cfg(target_os = "linux")]
fn terminate_linux_candle_tree(root: u32) {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let root_pid = Pid::from_raw(root as i32);
    let mut descendants = linux_process_descendants(root);
    let _ = kill(root_pid, Signal::SIGTERM);
    for &pid in &descendants {
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
    }

    let deadline =
        Instant::now() + Duration::from_secs(worker_shutdown_grace_seconds().saturating_add(1));
    while Instant::now() < deadline {
        // Capture children spawned in the narrow signal-delivery race and retain
        // them even if the supervisor exits and Linux reparents them meanwhile.
        for pid in linux_process_descendants(root) {
            if !descendants.contains(&pid) {
                descendants.push(pid);
                let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
            }
        }
        if !pid_alive(root) && !descendants.iter().copied().any(pid_alive) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Descendants are post-ordered, so force-kill leaves before parents.
    for pid in descendants {
        if pid_alive(pid) {
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        }
    }
    if pid_alive(root) {
        let _ = kill(root_pid, Signal::SIGKILL);
    }
}

/// SIGTERM then SIGKILL a confirmed-stale sidecar.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    let target = nix::unistd::Pid::from_raw(pid as i32);
    let _ = nix::sys::signal::kill(target, nix::sys::signal::Signal::SIGTERM);
    for _ in 0..20 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = nix::sys::signal::kill(target, nix::sys::signal::Signal::SIGKILL);
}

#[cfg(windows)]
fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output();
}

/// Kill sidecars left running by a prior unclean exit before spawning fresh
/// ones. Without this, a crash/force-quit (which skips `begin_shutdown`) leaves
/// orphaned API processes that accumulate, hold ports, and contend on jobs.db.
/// Each recorded PID is identity-checked so a recycled PID is never killed.
pub fn reap_stale_sidecars() {
    let path = sidecar_pidfile();
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let pids: SidecarPids = serde_json::from_slice(&bytes).unwrap_or_default();
    for (pid, candle_tree) in [
        (pids.api, false),
        (pids.mlx_worker, false),
        (pids.candle_worker, true),
    ]
    .into_iter()
    .filter_map(|(pid, candle_tree)| pid.map(|pid| (pid, candle_tree)))
    {
        if is_our_sidecar(pid) {
            #[cfg(target_os = "linux")]
            if candle_tree {
                terminate_linux_candle_tree(pid);
                continue;
            }
            #[cfg(not(target_os = "linux"))]
            let _ = candle_tree;
            kill_pid(pid);
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Ordered PIDs for Unix graceful shutdown: every native GPU supervisor first,
/// then the API. Kept pure so Linux candle PID participation cannot regress
/// unnoticed on non-Linux test hosts.
#[cfg(any(unix, test))]
fn unix_shutdown_pids(
    mlx_worker: Option<u32>,
    candle_worker: Option<u32>,
    api: Option<u32>,
) -> Vec<u32> {
    [mlx_worker, candle_worker, api]
        .into_iter()
        .flatten()
        .collect()
}

/// Begin graceful shutdown: stop the GPU worker (MLX on macOS, candle on
/// Windows/Linux) then the API sidecar.
/// On Unix this sends SIGTERM and waits up to the grace period before
/// force-killing; on Windows it force-kills (CTRL_BREAK handling is a
/// Windows-session refinement). Returns true if shutdown was initiated (caller
/// should prevent the immediate exit), false if it was already in progress.
pub fn begin_shutdown(app: &AppHandle) -> bool {
    let managed = app.state::<Managed>();
    if managed.shutting_down.swap(true, Ordering::SeqCst) {
        return false;
    }
    let mlx_worker = managed.mlx_worker.lock().expect("mlx worker lock").take();
    let candle_worker = managed
        .candle_worker
        .lock()
        .expect("candle worker lock")
        .take();
    let api_child = managed.api.lock().expect("api lock").take();
    // Take the credential IPC handle so its socket file can be unlinked on a graceful
    // quit (sc-5891); see the clean-exit block below. macOS-only (the socket is too).
    #[cfg(target_os = "macos")]
    let cred_ipc = managed.cred_ipc.lock().expect("cred_ipc lock").take();
    let handle = app.clone();
    std::thread::spawn(move || {
        #[cfg(target_os = "linux")]
        if let Some(child) = candle_worker.as_ref() {
            // Stop/reap the auto supervisor's full descendant tree while the
            // desktop is still alive, before the API is signalled.
            terminate_linux_candle_tree(child.pid());
        }
        #[cfg(unix)]
        {
            let grace = worker_shutdown_grace_seconds();
            let mlx_worker_pid = mlx_worker.as_ref().map(CommandChild::pid);
            let candle_worker_pid = candle_worker.as_ref().map(CommandChild::pid);
            let api_pid = api_child.as_ref().map(CommandChild::pid);
            let shutdown_pids = unix_shutdown_pids(mlx_worker_pid, candle_worker_pid, api_pid);
            // SIGTERM the workers first, then the API.
            for &pid in &shutdown_pids {
                let _ = nix::sys::signal::kill(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                );
            }
            let deadline = Instant::now() + Duration::from_secs(grace);
            while Instant::now() < deadline {
                if !shutdown_pids.iter().copied().any(pid_alive) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        // Force-kill anything still alive.
        if let Some(child) = mlx_worker {
            let _ = child.kill();
        }
        // The candle `auto` worker
        // is a supervisor that spawns one child per GPU plus a CPU child;
        // CommandChild::kill() (TerminateProcess) reaps ONLY the supervisor and
        // orphans those children — the exact leak that left N worker processes
        // running after every quit. Tree-kill the whole group by PID instead
        // (taskkill /T /F, the same reap path `reap_stale_sidecars` uses). The
        // worker's parent-death watchdog is the belt to this suspenders for the
        // force-quit/crash path that skips this teardown entirely.
        if let Some(child) = candle_worker {
            #[cfg(windows)]
            kill_pid(child.pid());
            #[cfg(target_os = "linux")]
            terminate_linux_candle_tree(child.pid());
            #[cfg(not(any(windows, target_os = "linux")))]
            let _ = child.kill();
        }
        if let Some(child) = api_child {
            let _ = child.kill();
        }
        // Clean exit: drop the pidfile so the next launch doesn't try to reap
        // PIDs we already terminated.
        let _ = std::fs::remove_file(sidecar_pidfile());
        // Unlink the credential IPC socket (sc-5891) so a graceful quit doesn't leave a
        // stale `cred-ipc.sock` in the data dir. A crash/force-quit skips this, but the
        // next launch's bind reaps it (`cred_ipc::start` removes a stale socket first).
        #[cfg(target_os = "macos")]
        if let Some(cred_ipc) = cred_ipc {
            let _ = std::fs::remove_file(&cred_ipc.socket);
        }
        handle.exit(0);
    });
    true
}

/// True while any `sceneworks-api` sidecar image is still running. The auto-update
/// install gate polls this so the NSIS installer only overwrites `sceneworks-api.exe`
/// once every sidecar has released its lock on the binary — the API, the candle `auto`
/// worker supervisor, and its per-GPU + CPU children are all this same image.
#[cfg(windows)]
fn api_sidecar_running() -> bool {
    use std::os::windows::process::CommandExt;
    // CREATE_NO_WINDOW: this is polled in a tight loop during the update, so keep a
    // console window from flashing on every probe.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let Ok(output) = std::process::Command::new("tasklist")
        .args([
            "/FI",
            "IMAGENAME eq sceneworks-api.exe",
            "/FO",
            "CSV",
            "/NH",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
    else {
        // If the probe itself fails, don't spin — treat the binary as clear and let the
        // installer's own retry/abort dialog handle any lingering lock.
        return false;
    };
    String::from_utf8_lossy(&output.stdout).contains("sceneworks-api")
}

/// Non-Windows replaces a running executable in place (the swap unlinks the old inode,
/// which the live process keeps using), so an update never waits on the binary being
/// unlocked — the wait loop that calls this exits immediately.
#[cfg(not(windows))]
fn api_sidecar_running() -> bool {
    false
}

/// Stop the API + GPU-worker sidecars for an in-place auto-update and BLOCK until the
/// `sceneworks-api` binary is no longer running, so the installer can overwrite it
/// (sc-11015).
///
/// The Windows update path is why this exists: `tauri-plugin-updater`'s NSIS `/UPDATE`
/// terminates only the main app binary (`SceneWorks.exe`), not the `sceneworks-api.exe`
/// sidecars this shell spawned — the API, the candle `auto` worker supervisor, and its
/// per-GPU + CPU children. Any of them still holding the binary open makes NSIS abort
/// with "Error opening file for writing: …\sceneworks-api.exe". We tear them down and
/// wait for the lock to release before handing off to the installer.
///
/// Unlike [`begin_shutdown`] this does NOT exit the app — the shell must stay alive to
/// launch the installer (which relaunches into the new build on Windows, or which the
/// caller follows with `restart()` off Windows). Latches `shutting_down` so the worker
/// supervisors stop respawning and a sidecar exit doesn't surface the "API stopped
/// unexpectedly" error dialog mid-update.
pub fn stop_sidecars_for_update(app: &AppHandle) {
    let managed = app.state::<Managed>();
    managed.shutting_down.store(true, Ordering::SeqCst);

    let mlx_worker = managed.mlx_worker.lock().expect("mlx worker lock").take();
    let candle_worker = managed
        .candle_worker
        .lock()
        .expect("candle worker lock")
        .take();
    let api_child = managed.api.lock().expect("api lock").take();

    // Tear down the candle `auto` supervisor's complete per-GPU + CPU
    // descendant tree. A root-only kill can orphan those workers because they
    // watch the still-running desktop PID.
    if let Some(child) = candle_worker {
        #[cfg(windows)]
        kill_pid(child.pid());
        #[cfg(target_os = "linux")]
        terminate_linux_candle_tree(child.pid());
        #[cfg(not(any(windows, target_os = "linux")))]
        let _ = child.kill();
    }
    if let Some(child) = mlx_worker {
        let _ = child.kill();
    }
    if let Some(child) = api_child {
        let _ = child.kill();
    }

    // These PIDs are being terminated on purpose; drop the pidfile so the next launch
    // doesn't try to reap them (or a since-recycled PID).
    let _ = std::fs::remove_file(sidecar_pidfile());

    // Wait for the OS to report no `sceneworks-api.exe` still running, so its file
    // handle is released before the installer overwrites it. Bounded, so a wedged
    // process can't hang the update forever — past the deadline the installer's own
    // retry/abort dialog takes over. Returns immediately off Windows (no lock to wait
    // on).
    let deadline = Instant::now() + Duration::from_secs(15);
    while api_sidecar_running() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Minimum NVIDIA display driver for the bundled CUDA 12.9 runtime (sc-3676 /
/// sc-5560): the floor that supports it and forward-JITs the compute_80 PTX.
#[cfg(target_os = "windows")]
const MIN_NVIDIA_DRIVER: f64 = 576.02;

#[cfg(target_os = "windows")]
const CUDA_REQUIREMENT: &str = "SceneWorks on Windows requires an NVIDIA (CUDA) GPU. \
    No NVIDIA GPU was detected — SceneWorks needs an NVIDIA GPU with driver 576.02 or \
    newer (there is no CPU or AMD fallback).";

/// Decide the preflight verdict from `nvidia-smi --query-gpu=name,driver_version`
/// output (`None` = nvidia-smi missing/failed). Pure so it's unit-testable; the IO
/// lives in `cuda_preflight`. `Ok(())` when a usable GPU is present; `Err(message)`
/// with a clear requirement otherwise (no GPU, or a driver below the floor).
#[cfg(target_os = "windows")]
fn evaluate_nvidia_preflight(smi_output: Option<&str>) -> Result<(), String> {
    let Some(line) =
        smi_output.and_then(|out| out.lines().map(str::trim).find(|line| !line.is_empty()))
    else {
        return Err(CUDA_REQUIREMENT.to_owned());
    };
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next().unwrap_or("");
    let driver = parts.next().unwrap_or("");
    // Block on a too-old driver; if the version is unparseable, don't block on it
    // (the GPU is present — let the worker surface any deeper issue).
    if let Ok(version) = driver.parse::<f64>() {
        if version < MIN_NVIDIA_DRIVER {
            return Err(format!(
                "SceneWorks on Windows requires NVIDIA driver {MIN_NVIDIA_DRIVER} or newer \
                 (found {driver} on {name}). Update your NVIDIA driver to continue."
            ));
        }
    }
    Ok(())
}

/// Windows CUDA preflight (sc-5561). SceneWorks generation off-Mac is CUDA-only —
/// no CPU/AMD fallback — so a machine without an NVIDIA GPU + an adequate driver
/// can run neither candle nor the Python torch worker's cu128 wheels. Probe
/// `nvidia-smi` for a GPU + driver version and return a clear, actionable error so
/// the app says "requires an NVIDIA GPU" up front instead of provisioning a venv and
/// then dead-polling jobs it can never run. `Ok(())` when a usable GPU is present.
#[cfg(target_os = "windows")]
fn cuda_preflight() -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    let mut command = std::process::Command::new("nvidia-smi");
    command.args([
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]);
    // Don't flash a console window when probing from the GUI app.
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let stdout = match command.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        // Missing (no NVIDIA driver) or errored → treat as no usable GPU.
        _ => return Err(CUDA_REQUIREMENT.to_owned()),
    };
    evaluate_nvidia_preflight(Some(&stdout))
}

#[cfg(any(target_os = "linux", all(test, target_os = "windows")))]
const MIN_LINUX_NVIDIA_DRIVER: (u32, u32, u32) = (575, 51, 3);

#[cfg(any(target_os = "linux", all(test, target_os = "windows")))]
fn parse_driver_version(value: &str) -> Option<(u32, u32, u32)> {
    let mut parts = value.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next().unwrap_or("0").parse().ok()?,
    ))
}

/// Linux counterpart of the Windows CUDA preflight. CUDA 12.9 requires the
/// 575-series Linux driver; missing `nvidia-smi`, no GPU rows, and an old driver
/// map to actionable setup errors before the multi-GB first-run download.
#[cfg(any(target_os = "linux", all(test, target_os = "windows")))]
fn evaluate_linux_nvidia_preflight(smi_output: Option<&str>) -> Result<(), String> {
    let Some(line) =
        smi_output.and_then(|out| out.lines().map(str::trim).find(|line| !line.is_empty()))
    else {
        return Err(
            "SceneWorks on Linux requires an NVIDIA GPU with driver 575.51.03 or newer. \
             No NVIDIA GPU was detected (nvidia-smi is missing or returned no devices); \
             the GPU worker will remain disabled."
                .to_owned(),
        );
    };
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next().unwrap_or("NVIDIA GPU");
    let driver = parts.next().unwrap_or("");
    if let Some(version) = parse_driver_version(driver) {
        if version < MIN_LINUX_NVIDIA_DRIVER {
            return Err(format!(
                "SceneWorks on Linux requires NVIDIA driver 575.51.03 or newer for CUDA 12.9 \
                 (found {driver} on {name}). Update the NVIDIA driver; the GPU worker will \
                 remain disabled."
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn linux_cuda_preflight() -> Result<(), String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let stdout = match output {
        Ok(output) if output.status.success() => {
            Some(String::from_utf8_lossy(&output.stdout).into_owned())
        }
        _ => None,
    };
    evaluate_linux_nvidia_preflight(stdout.as_deref())
}

/// Apple-Silicon Metal preflight (sc-8411): the macOS counterpart of the Windows
/// `cuda_preflight`. The desktop crate doesn't link MLX, so probe by re-launching the
/// bundled `sceneworks-api` sidecar in its one-shot `SCENEWORKS_GPU_CHECK=1` mode — a
/// tiny MLX op (1-element astype + eval) that fails exactly as a real job would if this
/// machine can't acquire a Metal GPU (a headless/SSH session, a wedged GPU). Running it
/// as the same sidecar binary means the probe sees the identical spawn context the
/// worker will. `Ok(())` when usable; `Err(message)` is the sidecar's user-facing reason
/// (relayed verbatim onto the setup screen), or a fallback if the probe produced none.
#[cfg(target_os = "macos")]
async fn metal_preflight(app: &AppHandle) -> Result<(), String> {
    let mut command = app
        .shell()
        .sidecar("sceneworks-api")
        .map_err(|error| format!("locate api for GPU check: {error}"))?
        .env("SCENEWORKS_GPU_CHECK", "1");
    // The probe's `astype`+`eval` dispatches a real MLX kernel, which loads MLX's
    // Metal shader library — so it needs the bundled metallib just like the API
    // sidecar and MLX worker spawns (sc-10349). This runs FIRST on startup, before
    // spawn_api/gate_window, so on a clean Mac (no ~/.cache/pmetal, no build tree) a
    // preflight without this env fails with "Failed to load the default metallib.
    // library not found" and the setup screen relays it — stranding every fresh
    // install on the first screen (sc-10353). Keeps the probe's spawn context
    // identical to the worker's, as this fn's own contract states.
    if let Some(metallib) = resolve_bundled_metallib(app) {
        command = command.env("PMETAL_METALLIB_PATH", metallib);
    }
    let output = command
        .output()
        .await
        .map_err(|error| format!("run GPU check: {error}"))?;
    if output.status.code() == Some(0) {
        return Ok(());
    }
    let message = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if message.is_empty() {
        // The probe died before printing its reason (e.g. an unexpected crash / signal).
        // Surface a usable fallback rather than a blank error.
        return Err(
            "SceneWorks can't initialize the Metal GPU on this Mac. It requires \
                    Apple Silicon with GPU access; running over SSH or in a headless \
                    session is not supported. Try opening SceneWorks on the Mac itself, \
                    or reboot and reopen."
                .to_owned(),
        );
    }
    Err(message)
}

async fn run_startup(app: AppHandle) {
    // Provide the builtin model catalog the rust-api/worker expect before they
    // start, so Model Manager is populated and native video resources resolve.
    // Mandatory: abort (rather than start a half-working app) if it can't be written.
    if let Err(error) = seed_builtin_manifests() {
        emit(&app, "error", format!("Setup failed: {error}"), true);
        return;
    }
    // CUDA-only on Windows (sc-5561): fail fast with a clear requirement message on a
    // machine with no NVIDIA GPU / too-old driver, BEFORE the multi-GB redist download,
    // rather than silently failing or dead-polling a job later. The setup page renders
    // this `error` event (apps/desktop/ui/index.html). The off-Mac desktop is candle/
    // CUDA-only now, so this always runs on Windows (the old "is the redist bundled?"
    // gate can't be used — the redist isn't downloaded yet at this point, and there's no
    // candle feature on the desktop crate to gate on; failing fast before a 2.7 GB
    // download is the whole point).
    #[cfg(target_os = "windows")]
    if let Err(error) = cuda_preflight() {
        emit(&app, "error", error, true);
        return;
    }
    #[cfg(target_os = "linux")]
    if let Err(error) = linux_cuda_preflight() {
        emit(&app, "error", error, true);
        return;
    }
    // First-run GPU runtime provisioning (Windows candle build): the CUDA runtime +
    // cuDNN + onnxruntime-gpu DLLs are no longer bundled (they exceeded NSIS's ~2 GB
    // datablock limit), so download them once into %APPDATA%\SceneWorks\gpu-runtime and
    // resolve them from there (cuda_provision.rs). Idempotent + cached on later runs via
    // a version marker; emits per-component progress to the setup screen. On failure,
    // surface it and abort (same slot the removed Python-venv provisioning used).
    #[cfg(target_os = "windows")]
    if let Err(error) = crate::cuda_provision::provision(&app).await {
        emit(
            &app,
            "error",
            format!(
                "GPU runtime setup failed: {error}. To install on a disconnected machine, \
                 pre-stage the runtime and set SCENEWORKS_GPU_RUNTIME_DIR — see \
                 docs/offline-install.md."
            ),
            true,
        );
        return;
    }
    #[cfg(target_os = "linux")]
    if let Err(error) = crate::linux_cuda_provision::provision(&app).await {
        emit(
            &app,
            "error",
            format!(
                "Linux GPU runtime setup failed: {error}. The GPU worker will remain disabled. \
                 For an offline install, pre-stage a provisioned Linux gpu-runtime directory \
                 and set SCENEWORKS_GPU_RUNTIME_DIR; see apps/desktop/docs/offline-install.md."
            ),
            true,
        );
        return;
    }
    // Apple-Silicon Metal preflight (sc-8411): fail fast with a clear, actionable
    // message on the setup screen if this Mac can't acquire a Metal GPU (a headless/SSH
    // session, a wedged GPU), BEFORE spawning the worker and loading a multi-GB model —
    // the macOS counterpart of the Windows `cuda_preflight` above. Without it the first
    // GPU op fails deep inside a model load with a raw MLX C++ assertion (and a leaked
    // CI build path). The setup page renders this `error` event.
    #[cfg(target_os = "macos")]
    if let Err(error) = metal_preflight(&app).await {
        emit(&app, "error", error, true);
        return;
    }
    // Linux credentials are eagerly handed to the native worker. If Secret Service
    // is unavailable/locked, stop on the visible setup screen with the actionable
    // keyring error instead of spawning a worker without HF/service tokens and
    // producing an opaque downstream 401.
    if let Err(error) = crate::settings::validate_worker_credentials() {
        emit(&app, "error", error, true);
        return;
    }
    // No Python venv on ANY platform: macOS went MLX-only (epic 3482, sc-3492/sc-3493)
    // and off-Mac went candle-only (epic 5483 Phase 7, sc-5563), so first run starts
    // straight on the native engine with no Python provisioning step.
    // Spawn the API only once across retries.
    if app
        .state::<Managed>()
        .api
        .lock()
        .expect("api lock")
        .is_some()
    {
        return;
    }
    emit(&app, "starting", "Starting the local engine…", false);
    if let Err(error) = spawn_api(&app) {
        emit(&app, "error", error, true);
        return;
    }
    gate_window(app);
}

/// Frontend entry point (called on setup-screen load and on retry). Kicks off
/// provisioning + startup; guarded so concurrent invocations are ignored.
#[tauri::command]
pub async fn start_setup(app: AppHandle) {
    {
        let state = app.state::<Managed>();
        if state.running.swap(true, Ordering::SeqCst) {
            return;
        }
    }
    run_startup(app.clone()).await;
    app.state::<Managed>()
        .running
        .store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod path_tests {
    use super::{huggingface_cache_env, select_huggingface_home, LinuxDesktopPaths};
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::path::PathBuf;

    fn resolve(values: &[(&str, &str)]) -> Result<LinuxDesktopPaths, String> {
        let env = values
            .iter()
            .map(|(name, value)| ((*name).to_owned(), OsString::from(value)))
            .collect::<HashMap<_, _>>();
        LinuxDesktopPaths::resolve(|name| env.get(name).cloned())
    }

    #[test]
    fn linux_xdg_overrides_cover_every_desktop_managed_location() {
        let paths = resolve(&[
            ("HOME", "relative-home"),
            ("XDG_DATA_HOME", "/mnt/xdg-data"),
            ("XDG_CONFIG_HOME", "/mnt/xdg-config"),
            ("XDG_CACHE_HOME", "/mnt/xdg-cache"),
            ("XDG_STATE_HOME", "/mnt/xdg-state"),
        ])
        .expect("absolute XDG overrides resolve");

        assert_eq!(paths.data_dir(), PathBuf::from("/mnt/xdg-data/SceneWorks"));
        assert_eq!(
            paths.config_dir(),
            PathBuf::from("/mnt/xdg-config/SceneWorks")
        );
        assert_eq!(paths.cache_dir, PathBuf::from("/mnt/xdg-cache/SceneWorks"));
        assert_eq!(paths.state_dir, PathBuf::from("/mnt/xdg-state/SceneWorks"));

        // Exact call surfaces owned by the desktop and its sidecars.
        assert_eq!(
            paths.settings_file(),
            PathBuf::from("/mnt/xdg-config/SceneWorks/settings.json")
        );
        assert_eq!(
            paths.config_dir().join("manifests"),
            PathBuf::from("/mnt/xdg-config/SceneWorks/manifests")
        );
        assert_eq!(
            paths.data_dir().join("cache").join("jobs.db"),
            PathBuf::from("/mnt/xdg-data/SceneWorks/cache/jobs.db")
        );
        assert_eq!(
            paths.logs_dir(),
            PathBuf::from("/mnt/xdg-state/SceneWorks/logs")
        );
        assert_eq!(
            paths.gpu_runtime_dir(),
            PathBuf::from("/mnt/xdg-data/SceneWorks/gpu-runtime")
        );
        assert_eq!(
            paths.huggingface_home(),
            PathBuf::from("/mnt/xdg-cache/SceneWorks/huggingface")
        );
        assert_eq!(
            paths.sidecar_pidfile(),
            PathBuf::from("/mnt/xdg-state/SceneWorks/desktop-sidecars.json")
        );
    }

    #[test]
    fn linux_xdg_fallbacks_are_home_scoped_and_never_relative() {
        let paths = resolve(&[("HOME", "/home/alice")]).expect("absolute HOME resolves");
        assert_eq!(
            paths.data_dir(),
            PathBuf::from("/home/alice/.local/share/SceneWorks")
        );
        assert_eq!(
            paths.config_dir(),
            PathBuf::from("/home/alice/.config/SceneWorks")
        );
        assert_eq!(
            paths.cache_dir,
            PathBuf::from("/home/alice/.cache/SceneWorks")
        );
        assert_eq!(
            paths.state_dir,
            PathBuf::from("/home/alice/.local/state/SceneWorks")
        );
    }

    #[test]
    fn linux_paths_fail_without_absolute_xdg_or_home_even_with_relative_tmpdir() {
        // The XDG spec requires absolute override values. `TMPDIR` is
        // intentionally relative to reproduce the review finding: resolution
        // must fail before any accessor can turn it into a launch-CWD write.
        let error = resolve(&[
            ("HOME", "relative-home"),
            ("XDG_DATA_HOME", "relative-data"),
            ("XDG_CONFIG_HOME", ""),
            ("TMPDIR", "relative-tmp"),
        ])
        .expect_err("missing absolute XDG/HOME bases must stop startup");
        assert!(
            error.contains("XDG_DATA_HOME") && error.contains("HOME"),
            "unexpected resolution error: {error}"
        );

        let absent = resolve(&[("TMPDIR", "relative-tmp")])
            .expect_err("missing HOME and XDG bases must stop startup");
        assert!(
            absent.contains("XDG_DATA_HOME") && absent.contains("HOME"),
            "unexpected resolution error: {absent}"
        );
    }

    #[test]
    fn linux_hf_home_ignores_relative_ambient_and_legacy_overrides() {
        let shared = PathBuf::from("/home/alice/.cache/SceneWorks/huggingface");
        assert_eq!(
            select_huggingface_home(
                Some("relative-ambient"),
                Some("/mnt/models/huggingface"),
                shared.clone(),
                true,
            ),
            PathBuf::from("/mnt/models/huggingface")
        );
        assert_eq!(
            select_huggingface_home(
                Some("/mnt/ambient/huggingface"),
                Some("/mnt/persisted/huggingface"),
                shared.clone(),
                true,
            ),
            PathBuf::from("/mnt/ambient/huggingface")
        );
        assert_eq!(
            select_huggingface_home(
                Some("./ambient"),
                Some("persisted-relative"),
                shared.clone(),
                true,
            ),
            shared
        );
    }

    #[test]
    fn non_linux_hf_home_preserves_relative_ambient_override() {
        assert_eq!(
            select_huggingface_home(
                Some("relative-ambient"),
                Some("relative-persisted"),
                PathBuf::from("shared"),
                false,
            ),
            PathBuf::from("relative-ambient")
        );
        assert_eq!(
            huggingface_cache_env("relative-ambient", false),
            vec![("HF_HOME", "relative-ambient".to_owned())]
        );
    }

    #[test]
    fn linux_hf_cache_env_pins_every_hub_override_to_absolute_xdg_home() {
        assert_eq!(
            huggingface_cache_env("/mnt/cache/SceneWorks/huggingface", true),
            vec![
                ("HF_HOME", "/mnt/cache/SceneWorks/huggingface".to_owned()),
                (
                    "HF_HUB_CACHE",
                    "/mnt/cache/SceneWorks/huggingface/hub".to_owned()
                ),
                (
                    "HUGGINGFACE_HUB_CACHE",
                    "/mnt/cache/SceneWorks/huggingface/hub".to_owned()
                ),
            ]
        );
    }
}

#[cfg(test)]
mod linux_candle_tests {
    use super::{
        candle_worker_mode_env, linux_process_descendants_from, prepend_loader_paths,
        resolve_linux_candle_runtime, select_worker_supervisor, unix_shutdown_pids,
        DesktopPlatform, SidecarPids, WorkerSupervisor,
    };
    use std::path::{Path, PathBuf};

    fn runtime_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sceneworks-sc-10375-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ))
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("library parent"))
            .expect("create library dir");
        std::fs::write(path, b"sentinel").expect("write library sentinel");
    }

    fn write_proc_children(proc_root: &Path, pid: u32, tid: u32, children: &str) {
        let path = proc_root
            .join(pid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("children");
        std::fs::create_dir_all(path.parent().expect("proc task parent"))
            .expect("create fake proc task");
        std::fs::write(path, children).expect("write fake proc children");
    }

    #[test]
    fn linux_runtime_gate_requires_ort_cuda_and_cudnn() {
        let root = runtime_root("presence");
        assert!(
            resolve_linux_candle_runtime(&root).is_none(),
            "pre-provision state must stay dormant"
        );

        touch(&root.join("onnxruntime/capi/libonnxruntime.so.1.26.0"));
        touch(&root.join("cuda/lib64/libcudart.so.12"));
        assert!(
            resolve_linux_candle_runtime(&root).is_none(),
            "a partial runtime must not start a crash-looping worker"
        );

        touch(&root.join("cublas/lib/libcublas.so.12"));
        touch(&root.join("cublas/lib/libcublasLt.so.12"));
        touch(&root.join("curand/lib/libcurand.so.10"));
        touch(&root.join("cuda_nvrtc/lib/libnvrtc.so.12"));
        touch(&root.join("cudnn/lib/libcudnn.so.9"));
        touch(&root.join("cufft/lib/libcufft.so.11"));
        touch(&root.join("nvjitlink/lib/libnvJitLink.so.12"));
        let runtime =
            resolve_linux_candle_runtime(&root).expect("all required sentinels are present");
        assert_eq!(
            runtime.ort_dylib,
            root.join("onnxruntime/capi/libonnxruntime.so.1.26.0")
        );
        assert_eq!(runtime.cuda_dir, root.join("cuda/lib64"));
        assert_eq!(runtime.cudnn_dir, root.join("cudnn/lib"));
        assert_eq!(
            runtime.loader_dirs,
            vec![
                root.join("onnxruntime/capi"),
                root.join("cudnn/lib"),
                root.join("cufft/lib"),
                root.join("nvjitlink/lib"),
                root.join("cuda_nvrtc/lib"),
                root.join("cublas/lib"),
                root.join("curand/lib"),
                root.join("cuda/lib64"),
            ]
        );

        std::fs::remove_dir_all(root).expect("remove isolated test runtime");
    }

    #[test]
    fn linux_loader_path_prepends_runtime_and_deduplicates_inherited_entries() {
        let ort = PathBuf::from("/runtime/onnxruntime/capi");
        let cuda = PathBuf::from("/runtime/cuda/lib64");
        let system = PathBuf::from("/usr/local/cuda/lib64");
        assert_eq!(
            prepend_loader_paths(&[ort.clone(), cuda.clone()], [system.clone(), cuda.clone()]),
            vec![ort, cuda, system]
        );
    }

    #[test]
    fn supervisor_selection_ungates_linux_only_after_runtime_presence() {
        assert_eq!(
            select_worker_supervisor(DesktopPlatform::Macos, false),
            WorkerSupervisor::Mlx
        );
        assert_eq!(
            select_worker_supervisor(DesktopPlatform::Windows, true),
            WorkerSupervisor::Candle
        );
        assert_eq!(
            select_worker_supervisor(DesktopPlatform::Linux, true),
            WorkerSupervisor::Candle
        );
        assert_eq!(
            select_worker_supervisor(DesktopPlatform::Linux, false),
            WorkerSupervisor::Dormant
        );
        assert_eq!(
            select_worker_supervisor(DesktopPlatform::Other, true),
            WorkerSupervisor::Dormant
        );
    }

    #[test]
    fn candle_supervisor_uses_worker_only_auto_backend_contract() {
        assert_eq!(
            candle_worker_mode_env(),
            [
                ("SCENEWORKS_WORKER_ONLY", "1"),
                ("SCENEWORKS_GPU_ID", "auto"),
                ("SCENEWORKS_BACKEND_CANDLE_ENABLED", "true"),
            ]
        );
    }

    #[test]
    fn linux_shutdown_orders_candle_supervisor_before_api() {
        assert_eq!(
            unix_shutdown_pids(None, Some(202), Some(303)),
            vec![202, 303]
        );
        assert_eq!(
            unix_shutdown_pids(Some(101), Some(202), Some(303)),
            vec![101, 202, 303]
        );
    }

    #[test]
    fn candle_pid_survives_pidfile_round_trip() {
        let pids = SidecarPids {
            api: Some(101),
            mlx_worker: None,
            candle_worker: Some(202),
        };
        let encoded = serde_json::to_vec(&pids).expect("serialize pids");
        let decoded: SidecarPids = serde_json::from_slice(&encoded).expect("deserialize pids");
        assert_eq!(decoded.api, Some(101));
        assert_eq!(decoded.mlx_worker, None);
        assert_eq!(decoded.candle_worker, Some(202));
    }

    #[test]
    fn linux_candle_tree_teardown_is_wired_to_every_lifecycle_path() {
        let source = include_str!("setup.rs");
        // Assemble the call so this test's own source does not count itself.
        let call = ["terminate_linux_", "candle_tree(child.pid());"].concat();
        assert!(
            source.matches(&call).count() >= 4,
            "restart, stale-spawn cleanup, shutdown and update teardown must all \
             use descendant-aware Linux candle cleanup"
        );
        let stale_call = ["terminate_linux_", "candle_tree(pid);"].concat();
        assert!(
            source.contains(&stale_call),
            "pidfile reaping must use descendant-aware cleanup for candle"
        );
        let linux_cfg = [
            r#"#[cfg(any(target_os = "windows", "#,
            r#"target_os = "linux"))]"#,
        ]
        .concat();
        assert!(
            source.contains(&linux_cfg),
            "the candle supervisor/PID slots must compile on Linux as well as Windows"
        );
    }

    #[test]
    fn linux_descendants_include_children_spawned_by_non_leader_threads() {
        let proc_root = runtime_root("proc");
        // PID 100's leader spawned 200, while Tokio-style worker TID 101
        // spawned 300. PID 200 in turn spawned 400 from its own worker thread.
        write_proc_children(&proc_root, 100, 100, "200\n");
        write_proc_children(&proc_root, 100, 101, "300\n");
        write_proc_children(&proc_root, 200, 200, "");
        write_proc_children(&proc_root, 200, 201, "400\n");
        write_proc_children(&proc_root, 300, 300, "");
        write_proc_children(&proc_root, 400, 400, "");

        assert_eq!(
            linux_process_descendants_from(&proc_root, 100),
            vec![400, 200, 300],
            "post-order discovery must include every task's recursively spawned children"
        );

        std::fs::remove_dir_all(proc_root).expect("remove isolated fake proc tree");
    }
}

#[cfg(all(test, target_os = "windows"))]
mod preflight_tests {
    use super::{evaluate_nvidia_preflight, MIN_NVIDIA_DRIVER};

    #[test]
    fn no_nvidia_smi_output_requires_an_nvidia_gpu() {
        // nvidia-smi missing/failed (None) or empty output → requirement error.
        assert!(evaluate_nvidia_preflight(None).is_err());
        assert!(evaluate_nvidia_preflight(Some("")).is_err());
        assert!(evaluate_nvidia_preflight(Some("   \n")).is_err());
    }

    #[test]
    fn adequate_driver_passes() {
        assert!(evaluate_nvidia_preflight(Some("NVIDIA RTX PRO 6000, 576.02\n")).is_ok());
        assert!(evaluate_nvidia_preflight(Some("NVIDIA GeForce RTX 4090, 597.36")).is_ok());
    }

    #[test]
    fn too_old_driver_is_rejected_with_the_floor() {
        let verdict = evaluate_nvidia_preflight(Some("NVIDIA GeForce RTX 3090, 560.94"));
        let message = verdict.expect_err("a sub-576.02 driver must fail preflight");
        assert!(message.contains(&MIN_NVIDIA_DRIVER.to_string()));
        assert!(message.contains("560.94"));
    }

    #[test]
    fn unparseable_driver_does_not_block_a_present_gpu() {
        // The GPU is present; an odd version string shouldn't hard-block startup.
        assert!(evaluate_nvidia_preflight(Some("NVIDIA RTX, not-a-version")).is_ok());
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod linux_preflight_tests {
    use super::evaluate_linux_nvidia_preflight;

    #[test]
    fn absent_linux_nvidia_runtime_maps_to_dormant_message() {
        for output in [None, Some(""), Some(" \n")] {
            let error = evaluate_linux_nvidia_preflight(output).expect_err("missing GPU must fail");
            assert!(error.contains("575.51.03"));
            assert!(error.contains("remain disabled"));
        }
    }

    #[test]
    fn linux_driver_floor_is_compared_by_version_components() {
        assert!(evaluate_linux_nvidia_preflight(Some("NVIDIA RTX 4090, 575.51.03\n")).is_ok());
        assert!(evaluate_linux_nvidia_preflight(Some("NVIDIA RTX 4090, 580.1.0\n")).is_ok());
        let error = evaluate_linux_nvidia_preflight(Some("NVIDIA RTX 4090, 575.50.99\n"))
            .expect_err("old driver must fail");
        assert!(error.contains("575.50.99"));
        assert!(error.contains("CUDA 12.9"));
    }

    #[test]
    fn unusual_linux_driver_text_defers_to_runtime_instead_of_false_negative() {
        assert!(
            evaluate_linux_nvidia_preflight(Some("NVIDIA Datacenter GPU, vendor-build")).is_ok()
        );
    }
}

/// Opt-in real-host validation seam for Linux packaging/CI. Normal tests remain
/// fixture-only and never need an NVIDIA host or the multi-GB runtime.
#[cfg(all(test, target_os = "linux"))]
mod linux_runtime_smoke_tests {
    use super::{
        find_linux_shared_object, linux_candle_runtime, linux_cuda_preflight, prepend_loader_paths,
    };

    #[test]
    #[ignore = "manual/CI: requires a provisioned Linux NVIDIA runtime"]
    fn provisioned_runtime_resolves_every_onnxruntime_dependency() {
        linux_cuda_preflight().expect("NVIDIA driver preflight");
        let runtime = linux_candle_runtime().expect("complete XDG gpu-runtime");
        let provider = runtime
            .loader_dirs
            .iter()
            .find_map(|dir| find_linux_shared_object(dir, "libonnxruntime_providers_cuda.so"))
            .expect("onnxruntime CUDA provider");
        let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let loader_paths =
            prepend_loader_paths(&runtime.loader_dirs, std::env::split_paths(&inherited));
        let joined = std::env::join_paths(loader_paths).expect("join LD_LIBRARY_PATH");
        let output = std::process::Command::new("ldd")
            .arg(provider)
            .env("LD_LIBRARY_PATH", joined)
            .output()
            .expect("run ldd");
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "ldd failed:\n{report}");
        assert!(
            !report.contains("not found"),
            "onnxruntime CUDA dependency missing:\n{report}"
        );
    }
}

#[cfg(test)]
mod bind_tests {
    use super::{decide_api_bind_env, parse_listening_port, ApiBindEnv};
    use crate::settings::DEFAULT_REMOTE_PORT;

    // epic 4484 stories 2/3/10: the API sidecar launch-env selector.

    #[test]
    fn disabled_binds_loopback_dynamic_no_token() {
        // Default (remote access off): byte-for-byte today's behavior.
        let env = decide_api_bind_env(false, Some(9000), Some("hunter2".to_owned()));
        assert_eq!(
            env,
            ApiBindEnv {
                host: "127.0.0.1",
                port: "0".to_owned(),
                access_token: None,
                fixed_port: None,
                warning: None,
            }
        );
    }

    #[test]
    fn enabled_with_password_binds_open_fixed_port_with_token() {
        let env = decide_api_bind_env(true, Some(8910), Some("  swordfish  ".to_owned()));
        assert_eq!(env.host, "0.0.0.0");
        assert_eq!(env.port, "8910");
        // Token is the trimmed password; it is also handed to the GPU worker(s).
        assert_eq!(env.access_token.as_deref(), Some("swordfish"));
        assert_eq!(env.fixed_port, Some(8910));
        assert!(env.warning.is_none());
    }

    #[test]
    fn enabled_without_port_uses_the_default_suggestion() {
        let env = decide_api_bind_env(true, None, Some("pw".to_owned()));
        assert_eq!(env.host, "0.0.0.0");
        assert_eq!(env.port, DEFAULT_REMOTE_PORT.to_string());
        assert_eq!(env.fixed_port, Some(DEFAULT_REMOTE_PORT));
    }

    #[test]
    fn enabled_without_password_fails_closed_to_loopback() {
        // Security invariant: never bind non-loopback without a password (story 3).
        let env = decide_api_bind_env(true, Some(8787), None);
        assert_eq!(env.host, "127.0.0.1");
        assert_eq!(env.port, "0");
        assert!(env.access_token.is_none());
        assert!(env.fixed_port.is_none());
        assert!(
            env.warning.is_some(),
            "missing-password must surface a warning"
        );
    }

    #[test]
    fn enabled_with_blank_password_fails_closed() {
        // A whitespace-only password is treated as absent → loopback, never an empty
        // token on an open bind.
        let env = decide_api_bind_env(true, Some(8787), Some("   ".to_owned()));
        assert_eq!(env.host, "127.0.0.1");
        assert!(env.access_token.is_none());
        assert!(env.warning.is_some());
    }

    /// Story 3: the desktop must NEVER set `SCENEWORKS_ALLOW_OPEN_BIND` (the server's
    /// open-bind refusal must remain the backstop). Assert the var never appears as a
    /// double-quoted string literal anywhere in the desktop crate — i.e. it is never
    /// passed to `.env("…")` / `set_var("…")`. (Explanatory comments referencing the
    /// var in prose or `backticks` are allowed and don't match.) The needle is
    /// assembled from parts so this test's own source doesn't trip the `include_str!`
    /// scan of this very file.
    ///
    /// F-128 (sc-8930): scans ALL of the crate's source modules, not just four — the
    /// invariant is only enforced where it's checked, so a `.env("SCENEWORKS_ALLOW_
    /// OPEN_BIND", …)` added to `update.rs` / `cred_ipc.rs` / `cuda_provision.rs` would
    /// previously have slipped past. Keep this list in lockstep with `apps/desktop/src`.
    #[test]
    fn desktop_never_sets_allow_open_bind() {
        let needle = concat!("\"SCENEWORKS_", "ALLOW_OPEN_BIND\"");
        // Every .rs module in apps/desktop/src (cuda_provision.rs is Windows-gated at
        // compile time, but include_str! reads its source on any host, so its bind env
        // is scanned everywhere too).
        for source in [
            include_str!("setup.rs"),
            include_str!("settings.rs"),
            include_str!("main.rs"),
            include_str!("net.rs"),
            include_str!("update.rs"),
            include_str!("cred_ipc.rs"),
            include_str!("cuda_provision.rs"),
        ] {
            assert!(
                !source.contains(needle),
                "desktop crate must never set the open-bind override env var"
            );
        }
    }

    #[test]
    fn parse_listening_port_handles_both_markers() {
        // Loopback (dynamic) startup line.
        assert_eq!(
            parse_listening_port("api_listening address=127.0.0.1:54321 ..."),
            Some(54321)
        );
        // LAN (0.0.0.0/fixed) startup line.
        assert_eq!(
            parse_listening_port("api_listening address=0.0.0.0:8787 ..."),
            Some(8787)
        );
        assert_eq!(parse_listening_port("nothing to see here"), None);
    }

    /// F-127 (sc-8929): the port is read ONLY from the `api_listening` event's
    /// `address=` field. An earlier diagnostic line that merely mentions a loopback
    /// `host:port` (a health probe, the credential socket, etc.) must NOT seed a port —
    /// doing so previously pointed window-gating at a dead port for the full 30 s.
    #[test]
    fn parse_listening_port_ignores_unrelated_lines_with_a_host_port() {
        // A diagnostic line with a loopback host:port but no api_listening marker.
        assert_eq!(
            parse_listening_port("probing health at 127.0.0.1:9999 before start"),
            None
        );
        // A line that mentions the event but whose host:port isn't in the address field
        // (e.g. an incidental reference) doesn't get mis-parsed from the wrong token.
        assert_eq!(
            parse_listening_port("api_listening address=127.0.0.1:54321 (peer 127.0.0.1:1)"),
            Some(54321)
        );
        // Real logfmt-style line with fields before/after address= still parses.
        assert_eq!(
            parse_listening_port(
                "2026-07-04 event=api_listening address=127.0.0.1:41234 msg=\"listening\""
            ),
            Some(41234)
        );
    }

    /// The packaged sidecar's stdout is a pipe, so the API logs JSON
    /// (`"address":"HOST:PORT"`), not the logfmt `address=HOST:PORT` a TTY/dev run
    /// emits. Keying only on `address=` (the sc-8929 regression) failed to discover
    /// the port on a default loopback launch, so window-gating dead-polled until the
    /// 30 s timeout. Parse the real JSON envelope the desktop reader actually sees.
    #[test]
    fn parse_listening_port_handles_json_stdout() {
        // Loopback (OS-assigned) — the default packaged first-run path.
        assert_eq!(
            parse_listening_port(
                r#"{"message":"SceneWorks API listening","event":"api_listening","address":"127.0.0.1:60294","level":"info","reportedAt":"2026-07-04T15:20:19Z"}"#
            ),
            Some(60294)
        );
        // LAN (0.0.0.0/fixed) JSON envelope parses too.
        assert_eq!(
            parse_listening_port(
                r#"{"event":"api_listening","address":"0.0.0.0:8787","level":"info"}"#
            ),
            Some(8787)
        );
        // A JSON line that is NOT the listening event contributes no port.
        assert_eq!(
            parse_listening_port(
                r#"{"event":"utility_worker_inprocess","apiUrl":"http://127.0.0.1:60294"}"#
            ),
            None
        );
    }
}

/// sc-13605: the API-crash → Retry supervisor-thread leak.
///
/// `handle_api_exit` clears the API slot so Retry re-runs `spawn_api` +
/// `gate_window`. The old `gate_window` spawned a worker supervisor thread
/// unconditionally, so every crash → Retry cycle stacked another supervisor on
/// the single worker slot; the stale ones kept respawning a worker pointed at
/// the dead port captured at their spawn. These tests pin the halves of the fix
/// — the sole-supervisor guard, the per-iteration port re-read, and the
/// post-spawn correlated-death recheck — as pure decision logic, without
/// spawning real threads.
#[cfg(test)]
mod supervisor_tests {
    use super::{
        current_api_url, resolve_supervisor_action, verify_spawned_target, Managed, SpawnVerdict,
        SupervisorAction, SupervisorSlot,
    };
    use std::sync::atomic::Ordering;

    /// Discriminating test: simulate the API-crash → Retry cycle N times and
    /// assert only ONE supervisor ever starts. Each `gate_window` attempts to
    /// acquire the sole-supervisor slot before starting its loop; the live
    /// supervisor survives an API crash (it re-reads the new port rather than
    /// exiting), so it never releases the slot and every Retry is refused.
    /// Against the pre-sc-13605 unconditional spawn this count would be `1 + N`,
    /// so this test fails on the old behavior.
    #[test]
    fn only_one_supervisor_starts_across_repeated_api_crash_retries() {
        let slot = SupervisorSlot::default();
        let mut started = 0;
        // Initial startup: gate_window claims the free slot and starts.
        if slot.try_acquire() {
            started += 1;
        }
        // Ten API-crash → Retry cycles. The live supervisor has NOT released the
        // slot (it survives each crash and re-reads the fresh port), so every
        // retry's acquire must be refused.
        for _ in 0..10 {
            if slot.try_acquire() {
                started += 1;
            }
        }
        assert_eq!(
            started, 1,
            "exactly one supervisor may start across retries; the old unconditional \
             spawn would have started 11"
        );
    }

    /// A supervisor that legitimately exits (e.g. a sidecar-locate failure)
    /// releases the slot via its `SupervisorLease` Drop, so the NEXT
    /// `gate_window` can start a fresh supervisor instead of being wedged out
    /// forever by a stuck liveness flag.
    #[test]
    fn released_slot_admits_a_fresh_supervisor() {
        let slot = SupervisorSlot::default();
        assert!(slot.try_acquire(), "first acquire wins the free slot");
        assert!(
            !slot.try_acquire(),
            "a second acquire while live is refused"
        );
        slot.release();
        assert!(
            slot.try_acquire(),
            "after the live supervisor exits and releases, a new one may start"
        );
    }

    /// Port re-read: the supervisor derives the worker URL from the CURRENT
    /// `Managed.api_port` every iteration, not a value captured at spawn. Model a
    /// Retry that rebinds the API to a NEW port and assert the derived URL
    /// follows it — the exact behavior the old captured-`api_url` code got wrong
    /// (it kept pointing the respawned worker at the dead original port).
    #[test]
    fn worker_url_follows_the_current_api_port_after_a_retry() {
        let managed = Managed::default();
        // No port discovered yet → no URL (the supervisor parks).
        assert_eq!(current_api_url(&managed), None);
        // Initial API bind publishes a port.
        *managed.api_port.lock().expect("api port lock") = Some(50111);
        assert_eq!(
            current_api_url(&managed).as_deref(),
            Some("http://127.0.0.1:50111")
        );
        // An API crash clears the port (handle_api_exit); the Retry rebinds a NEW one.
        *managed.api_port.lock().expect("api port lock") = None;
        assert_eq!(current_api_url(&managed), None);
        *managed.api_port.lock().expect("api port lock") = Some(50222);
        // The re-read yields the NEW port's URL. A URL captured at spawn would
        // still read 50111 — the dead port the old supervisor churned against.
        assert_eq!(
            current_api_url(&managed).as_deref(),
            Some("http://127.0.0.1:50222")
        );
    }

    /// Loop-level port re-read (F-036 issue 2): the supervisor's per-attempt
    /// target comes from `resolve_supervisor_action` — the *same* seam the loop
    /// body calls every iteration (top of loop AND the post-spawn recheck) — which
    /// re-reads the live `Managed.api_port`. Drive successive attempts across a
    /// port change and assert the resolved `Spawn` target follows the port. A loop
    /// that captured the port (or a `resolve_supervisor_action` that memoized it)
    /// would resolve the OLD port on the second attempt and fail here — the
    /// captured-vs-reread regression this discriminates.
    #[test]
    fn supervisor_resolves_the_live_port_on_each_attempt() {
        let managed = Managed::default();
        // Attempt with no port yet → park, never spawn (post-crash, pre-Retry).
        assert_eq!(
            resolve_supervisor_action(&managed),
            SupervisorAction::WaitForPort
        );
        // Attempt 1: API bound at port A.
        *managed.api_port.lock().expect("api port lock") = Some(50111);
        assert_eq!(
            resolve_supervisor_action(&managed),
            SupervisorAction::Spawn("http://127.0.0.1:50111".to_owned())
        );
        // API crash + Retry rebinds a NEW port B; the very next attempt follows it.
        *managed.api_port.lock().expect("api port lock") = Some(50222);
        assert_eq!(
            resolve_supervisor_action(&managed),
            SupervisorAction::Spawn("http://127.0.0.1:50222".to_owned())
        );
        // A latched shutdown wins over any port → Exit.
        managed.shutting_down.store(true, Ordering::SeqCst);
        assert_eq!(resolve_supervisor_action(&managed), SupervisorAction::Exit);
    }

    /// Correlated-death verdict (F-036 issue 1): a worker just stored against
    /// port A must be kept only while A is still current; if the target moved to
    /// B or cleared in the pre-store window (where `handle_api_exit`'s
    /// `restart_gpu_worker` kill can miss the not-yet-stored child) it is killed
    /// and the loop retries, and if a shutdown latched it is killed and the loop
    /// exits.
    #[test]
    fn post_store_recheck_classifies_the_spawned_target() {
        let a = "http://127.0.0.1:50111";
        let b = "http://127.0.0.1:50222";
        // Target unchanged → keep and block on the worker's events.
        assert_eq!(
            verify_spawned_target(a, &SupervisorAction::Spawn(a.to_owned())),
            SpawnVerdict::Keep
        );
        // Port changed under us → kill + retry (the residual race this closes).
        assert_eq!(
            verify_spawned_target(a, &SupervisorAction::Spawn(b.to_owned())),
            SpawnVerdict::KillAndRetry
        );
        // Port cleared (crash, Retry not yet) → kill + retry, then the loop parks.
        assert_eq!(
            verify_spawned_target(a, &SupervisorAction::WaitForPort),
            SpawnVerdict::KillAndRetry
        );
        // Shutdown latched meanwhile → kill + exit.
        assert_eq!(
            verify_spawned_target(a, &SupervisorAction::Exit),
            SpawnVerdict::KillAndExit
        );
    }

    /// End-to-end (pure) of the correlated-death window through the *live*
    /// `Managed` seam: the loop resolves the target for this attempt (port A) and
    /// spawns against it, then a crash+Retry moves the port to B before the store
    /// completes, so the post-store recheck — reading the SAME live state the loop
    /// reads — returns `KillAndRetry`, and the retry then resolves B. Fails if the
    /// recheck used a captured target instead of re-reading.
    #[test]
    fn post_store_recheck_repoints_a_worker_left_on_a_dead_port() {
        let managed = Managed::default();
        *managed.api_port.lock().expect("api port lock") = Some(50111);
        let SupervisorAction::Spawn(spawned) = resolve_supervisor_action(&managed) else {
            panic!("expected a spawn target at port A");
        };
        assert_eq!(spawned, "http://127.0.0.1:50111");
        // Correlated death: the API crashed (port cleared) and a Retry rebound B,
        // all in the window before the child was stored — so restart_gpu_worker's
        // kill missed it. The post-store recheck re-reads and finds it stale.
        *managed.api_port.lock().expect("api port lock") = Some(50222);
        assert_eq!(
            verify_spawned_target(&spawned, &resolve_supervisor_action(&managed)),
            SpawnVerdict::KillAndRetry
        );
        // The retry resolves the NEW port, so the respawn targets B, not A.
        assert_eq!(
            resolve_supervisor_action(&managed),
            SupervisorAction::Spawn("http://127.0.0.1:50222".to_owned())
        );
    }
}
