use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::Notify;

use crate::{
    sweep_stale_uploads_cancellable, KEYPOINT_UPLOADS_CACHE_DIR, POSE_UPLOADS_CACHE_DIR,
    STALE_UPLOAD_SECONDS,
};

pub(crate) const BACKGROUND_MAINTENANCE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy)]
pub(crate) enum StartupCriticality {
    BindCritical,
    ReadinessCritical,
    BackgroundSafe,
}

impl StartupCriticality {
    fn as_str(self) -> &'static str {
        match self {
            Self::BindCritical => "bind-critical",
            Self::ReadinessCritical => "readiness-critical",
            Self::BackgroundSafe => "background-safe",
        }
    }
}

/// One-shot startup phase timer. `Drop` keeps failed phases observable without
/// changing their existing error propagation.
pub(crate) struct StartupPhaseTimer {
    phase: &'static str,
    criticality: StartupCriticality,
    started: Instant,
}

impl StartupPhaseTimer {
    pub(crate) fn start(phase: &'static str, criticality: StartupCriticality) -> Self {
        Self {
            phase,
            criticality,
            started: Instant::now(),
        }
    }
}

impl Drop for StartupPhaseTimer {
    fn drop(&mut self) {
        tracing::info!(
            event = "startup_phase_duration",
            phase = self.phase,
            criticality = self.criticality.as_str(),
            duration_ms = self.started.elapsed().as_secs_f64() * 1000.0,
            "SceneWorks API startup phase completed"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaintenanceStatus {
    Pending,
    Running,
    Complete,
    Failed,
    TimedOutCancelling,
    TimedOut,
    Cancelled,
}

impl MaintenanceStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Complete => "complete",
            Self::Failed => "failed",
            Self::TimedOutCancelling => "timed-out-cancelling",
            Self::TimedOut => "timed-out",
            Self::Cancelled => "cancelled",
        }
    }

    #[cfg(test)]
    fn is_terminal(self) -> bool {
        !matches!(
            self,
            Self::Pending | Self::Running | Self::TimedOutCancelling
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct MaintenanceState {
    status: MaintenanceStatus,
    removed_uploads: usize,
    failure_count: usize,
}

impl MaintenanceState {
    fn pending() -> Self {
        Self {
            status: MaintenanceStatus::Pending,
            removed_uploads: 0,
            failure_count: 0,
        }
    }

    fn complete() -> Self {
        Self {
            status: MaintenanceStatus::Complete,
            removed_uploads: 0,
            failure_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StartupMaintenanceResponse {
    pub(crate) phase: &'static str,
    pub(crate) criticality: &'static str,
    pub(crate) status: &'static str,
    pub(crate) removed_uploads: usize,
    pub(crate) failure_count: usize,
}

struct StartupMaintenanceInner {
    state: Mutex<MaintenanceState>,
    started: AtomicBool,
    cancelled: Arc<AtomicBool>,
    changed: Notify,
}

impl Drop for StartupMaintenanceInner {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Single-flight lifecycle for the only background-safe startup work: stale
/// upload staging sweeps. Project migrations, reserved libraries, job recovery,
/// and orphan pruning remain readiness-critical and synchronous.
#[derive(Clone)]
pub(crate) struct StartupMaintenance {
    inner: Arc<StartupMaintenanceInner>,
}

impl StartupMaintenance {
    pub(crate) fn pending() -> Self {
        Self::new(MaintenanceState::pending())
    }

    pub(crate) fn complete() -> Self {
        let maintenance = Self::new(MaintenanceState::complete());
        maintenance.inner.started.store(true, Ordering::Release);
        maintenance
    }

    fn new(state: MaintenanceState) -> Self {
        Self {
            inner: Arc::new(StartupMaintenanceInner {
                state: Mutex::new(state),
                started: AtomicBool::new(false),
                cancelled: Arc::new(AtomicBool::new(false)),
                changed: Notify::new(),
            }),
        }
    }

    /// Start exactly one bounded blocking sweep. Returns false when another
    /// caller already started it.
    pub(crate) fn start(&self, data_dir: PathBuf, timeout: Duration) -> bool {
        if self
            .inner
            .started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.set_state(MaintenanceState {
            status: MaintenanceStatus::Running,
            removed_uploads: 0,
            failure_count: 0,
        });

        let maintenance = self.clone();
        let cancelled = self.inner.cancelled.clone();
        tokio::spawn(async move {
            let _phase =
                StartupPhaseTimer::start("upload_sweeps", StartupCriticality::BackgroundSafe);
            let worker_cancelled = cancelled.clone();
            let mut worker = tokio::task::spawn_blocking(move || {
                run_upload_sweeps(&data_dir, worker_cancelled.as_ref())
            });
            match tokio::time::timeout(timeout, &mut worker).await {
                Ok(Ok(result)) if result.cancelled => {
                    maintenance.set_state(MaintenanceState {
                        status: MaintenanceStatus::Cancelled,
                        removed_uploads: result.removed_uploads,
                        failure_count: result.failure_count,
                    });
                }
                Ok(Ok(result)) => {
                    let status = if result.failure_count == 0 {
                        MaintenanceStatus::Complete
                    } else {
                        MaintenanceStatus::Failed
                    };
                    maintenance.set_state(MaintenanceState {
                        status,
                        removed_uploads: result.removed_uploads,
                        failure_count: result.failure_count,
                    });
                    if result.failure_count > 0 {
                        tracing::warn!(
                            event = "startup_background_maintenance_failed",
                            phase = "upload_sweeps",
                            criticality = StartupCriticality::BackgroundSafe.as_str(),
                            failure_count = result.failure_count,
                            "background startup maintenance completed with failures"
                        );
                    }
                }
                Ok(Err(error)) => {
                    maintenance.set_state(MaintenanceState {
                        status: MaintenanceStatus::Failed,
                        removed_uploads: 0,
                        failure_count: 1,
                    });
                    tracing::warn!(
                        event = "startup_background_maintenance_failed",
                        phase = "upload_sweeps",
                        criticality = StartupCriticality::BackgroundSafe.as_str(),
                        failure_count = 1,
                        error = %error,
                        "background startup maintenance task failed"
                    );
                }
                Err(_) => {
                    cancelled.store(true, Ordering::Release);
                    maintenance.set_state(MaintenanceState {
                        status: MaintenanceStatus::TimedOutCancelling,
                        removed_uploads: 0,
                        failure_count: 1,
                    });
                    tracing::warn!(
                        event = "startup_background_maintenance_timed_out",
                        phase = "upload_sweeps",
                        criticality = StartupCriticality::BackgroundSafe.as_str(),
                        timeout_ms = timeout.as_millis(),
                        "background startup maintenance exceeded its deadline; cancellation pending"
                    );
                    // `spawn_blocking` cannot be force-aborted. Retain and observe
                    // the worker so health never claims terminal completion while
                    // an in-flight network filesystem call may still mutate disk.
                    let (removed_uploads, failure_count) = match worker.await {
                        Ok(result) => (result.removed_uploads, result.failure_count.max(1)),
                        Err(error) => {
                            tracing::warn!(
                                event = "startup_background_maintenance_cancel_failed",
                                phase = "upload_sweeps",
                                error = %error,
                                "timed-out background maintenance task failed while cancelling"
                            );
                            (0, 1)
                        }
                    };
                    maintenance.set_state(MaintenanceState {
                        status: MaintenanceStatus::TimedOut,
                        removed_uploads,
                        failure_count,
                    });
                }
            }
        });
        true
    }

    pub(crate) fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn snapshot(&self) -> StartupMaintenanceResponse {
        let state = *self.inner.state.lock();
        StartupMaintenanceResponse {
            phase: "upload_sweeps",
            criticality: StartupCriticality::BackgroundSafe.as_str(),
            status: state.status.as_str(),
            removed_uploads: state.removed_uploads,
            failure_count: state.failure_count,
        }
    }

    #[cfg(test)]
    pub(crate) async fn wait_for_terminal(&self) -> StartupMaintenanceResponse {
        loop {
            let notified = self.inner.changed.notified();
            if self.inner.state.lock().status.is_terminal() {
                return self.snapshot();
            }
            notified.await;
        }
    }

    fn set_state(&self, state: MaintenanceState) {
        *self.inner.state.lock() = state;
        self.inner.changed.notify_waiters();
    }
}

#[derive(Debug, Default)]
struct UploadSweepResult {
    removed_uploads: usize,
    failure_count: usize,
    cancelled: bool,
}

fn run_upload_sweeps(data_dir: &Path, cancelled: &AtomicBool) -> UploadSweepResult {
    #[cfg(test)]
    if let Some(hook) = take_test_hook(data_dir) {
        hook.reached.notify_one();
        while !cancelled.load(Ordering::Acquire) && !hook.released.load(Ordering::Acquire) {
            std::thread::park_timeout(Duration::from_millis(5));
        }
        if hook.fail {
            return UploadSweepResult {
                failure_count: 1,
                cancelled: cancelled.load(Ordering::Acquire),
                ..Default::default()
            };
        }
    }

    let mut result = UploadSweepResult::default();
    let cutoff = std::time::SystemTime::now() - Duration::from_secs(STALE_UPLOAD_SECONDS);
    for (area, subdir) in [
        ("lora", "lora-uploads"),
        ("pose", POSE_UPLOADS_CACHE_DIR),
        ("keypoint", KEYPOINT_UPLOADS_CACHE_DIR),
        ("asset", "uploads"),
    ] {
        if cancelled.load(Ordering::Acquire) {
            result.cancelled = true;
            break;
        }
        match sweep_stale_uploads_cancellable(data_dir, subdir, cutoff, || {
            !cancelled.load(Ordering::Acquire)
        }) {
            Ok(removed) => result.removed_uploads = result.removed_uploads.saturating_add(removed),
            Err(error) => {
                result.failure_count = result.failure_count.saturating_add(1);
                tracing::warn!(
                    event = "stale_upload_sweep_failed",
                    area,
                    error = %error,
                    "background stale upload sweep failed"
                );
            }
        }
        if cancelled.load(Ordering::Acquire) {
            result.cancelled = true;
            break;
        }
    }
    result
}

#[cfg(test)]
#[derive(Clone)]
struct StartupMaintenanceTestHookInner {
    reached: Arc<Notify>,
    released: Arc<AtomicBool>,
    fail: bool,
}

#[cfg(test)]
static TEST_HOOKS: std::sync::OnceLock<
    Mutex<std::collections::HashMap<PathBuf, StartupMaintenanceTestHookInner>>,
> = std::sync::OnceLock::new();

#[cfg(test)]
pub(crate) struct StartupMaintenanceTestHook {
    data_dir: PathBuf,
    inner: StartupMaintenanceTestHookInner,
}

#[cfg(test)]
impl StartupMaintenanceTestHook {
    pub(crate) async fn wait_until_reached(&self) {
        self.inner.reached.notified().await;
    }

    pub(crate) fn release(&self) {
        self.inner.released.store(true, Ordering::Release);
    }
}

#[cfg(test)]
impl Drop for StartupMaintenanceTestHook {
    fn drop(&mut self) {
        self.release();
        TEST_HOOKS
            .get_or_init(Default::default)
            .lock()
            .remove(&self.data_dir);
    }
}

#[cfg(test)]
pub(crate) fn test_startup_maintenance_hook(
    data_dir: PathBuf,
    fail: bool,
) -> StartupMaintenanceTestHook {
    let inner = StartupMaintenanceTestHookInner {
        reached: Arc::new(Notify::new()),
        released: Arc::new(AtomicBool::new(false)),
        fail,
    };
    assert!(
        TEST_HOOKS
            .get_or_init(Default::default)
            .lock()
            .insert(data_dir.clone(), inner.clone())
            .is_none(),
        "startup maintenance hook already registered"
    );
    StartupMaintenanceTestHook { data_dir, inner }
}

#[cfg(test)]
fn take_test_hook(data_dir: &Path) -> Option<StartupMaintenanceTestHookInner> {
    TEST_HOOKS
        .get_or_init(Default::default)
        .lock()
        .remove(data_dir)
}
