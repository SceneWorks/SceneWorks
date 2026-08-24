//! Process recovery for the MiniMax-H3 MLX I2V Metal-watchdog failure (sc-21027).
//!
//! The inference provider labels its two FL2VA-only force boundaries with a stable
//! `MiniMax-H3 MLX I2V` marker. That lets this worker quarantine only the process instance whose
//! Metal command queue failed, without changing T2V, another model family, Candle, or an ordinary
//! MLX allocation failure. A poisoned process must not submit again: the worker loop consumes one
//! recycle request before its next claim and exits cleanly for the existing auto-worker supervisor
//! to replace it.
//!
//! Metal may report `SubmissionsIgnored` concurrently with, or just after, the actionable watchdog
//! timeout. All observations share one mutex so a secondary diagnostic can never replace the first
//! timeout. If scheduling makes the secondary visible first, the later timeout fills the still-empty
//! primary slot; the secondary is never promoted to an OOM or an invented timeout.

use std::future::Future;
use std::sync::{Mutex, PoisonError};

const H3_MLX_I2V_PHASE_MARKER: &str = "minimax-h3 mlx i2v";
const METAL_TIMEOUT_ENUM: &str = "kiogpucommandbuffercallbackerrortimeout";
const METAL_TIMEOUT_TEXT: &str = "caused gpu timeout error";
const SUBMISSIONS_IGNORED: &str = "submissionsignored";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalSignal {
    WatchdogTimeout,
    SubmissionsIgnored,
}

fn classify(diagnostic: &str) -> Option<MetalSignal> {
    let normalized = diagnostic.to_ascii_lowercase();
    if !normalized.contains(H3_MLX_I2V_PHASE_MARKER) {
        return None;
    }
    if normalized.contains(METAL_TIMEOUT_ENUM) || normalized.contains(METAL_TIMEOUT_TEXT) {
        Some(MetalSignal::WatchdogTimeout)
    } else if normalized.contains(SUBMISSIONS_IGNORED) {
        Some(MetalSignal::SubmissionsIgnored)
    } else {
        None
    }
}

#[derive(Default)]
struct Inner {
    first_timeout: Option<String>,
    first_secondary: Option<String>,
    quarantined: bool,
    terminal_failure_persisted: bool,
    recycle_started: bool,
}

/// State owned by one OS process. Instantiable so lifecycle tests can model the poisoned process
/// and its fresh replacement without mutating the production singleton.
pub(crate) struct MlxWorkerRecovery {
    inner: Mutex<Inner>,
}

impl MlxWorkerRecovery {
    pub(crate) const fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                first_timeout: None,
                first_secondary: None,
                quarantined: false,
                terminal_failure_persisted: false,
                recycle_started: false,
            }),
        }
    }

    /// Record one family-scoped diagnostic. Returns the truthful job failure when the diagnostic
    /// proves the process is poisoned, otherwise `None` and leaves claim eligibility untouched.
    pub(crate) fn observe(&self, diagnostic: &str) -> Option<String> {
        let signal = classify(diagnostic)?;
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        match signal {
            MetalSignal::WatchdogTimeout => {
                inner
                    .first_timeout
                    .get_or_insert_with(|| diagnostic.to_owned());
            }
            MetalSignal::SubmissionsIgnored => {
                inner
                    .first_secondary
                    .get_or_insert_with(|| diagnostic.to_owned());
            }
        }
        inner.quarantined = true;
        Some(job_failure(&inner))
    }

    /// Whether this process may enter the next claim. Once false it never becomes true again; only
    /// constructing a replacement process (and therefore a fresh state instance) restores it.
    pub(crate) fn can_claim(&self) -> bool {
        !self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .quarantined
    }

    /// Arm recycling only after the active job's truthful failure reached the API. Quarantine is
    /// immediate, but a transient terminal-write failure must never turn into a clean child exit
    /// whose supervisor quite correctly has no abnormal death to attribute.
    pub(crate) fn note_terminal_failure_persisted(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if inner.quarantined {
            inner.terminal_failure_persisted = true;
        }
    }

    /// Current quarantine text even before recycling is armed. The loop uses this only as a
    /// no-claim backstop when shutdown interrupted the terminal-write retry.
    pub(crate) fn quarantine_reason(&self) -> Option<String> {
        let inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        inner.quarantined.then(|| status_reason(&inner))
    }

    /// Consume the process's one recycle transition. The caller clean-exits on `Some`; repeated
    /// loop checks return `None`, preventing duplicate restart requests or duplicate attribution.
    pub(crate) fn begin_recycle(&self) -> Option<RecycleRequest> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if !inner.quarantined || !inner.terminal_failure_persisted || inner.recycle_started {
            return None;
        }
        inner.recycle_started = true;
        Some(RecycleRequest {
            reason: status_reason(&inner),
            first_timeout: inner.first_timeout.clone(),
            saw_submissions_ignored: inner.first_secondary.is_some(),
        })
    }
}

fn job_failure(inner: &Inner) -> String {
    match (&inner.first_timeout, &inner.first_secondary) {
        (Some(timeout), Some(_)) => format!(
            "{timeout} This first Metal watchdog timeout poisoned the MLX worker process. A later \
             kIOGPUCommandBufferCallbackErrorSubmissionsIgnored diagnostic is secondary fallout, \
             not an out-of-memory error. The active job failed and this process will recycle \
             before another claim."
        ),
        (Some(timeout), None) => format!(
            "{timeout} This Metal watchdog timeout poisoned the MLX worker process. The active job \
             failed and this process will recycle before another claim."
        ),
        (None, Some(secondary)) => format!(
            "{secondary} The driver is ignoring this process's Metal submissions after an earlier \
             GPU fault. SubmissionsIgnored is secondary fallout, not an out-of-memory error; no \
             actionable watchdog timeout was captured by this process. The active job failed and \
             this process will recycle before another claim."
        ),
        (None, None) => unreachable!("a recovery message requires a classified diagnostic"),
    }
}

fn status_reason(inner: &Inner) -> String {
    match &inner.first_timeout {
        Some(timeout) => format!(
            "The MLX process is quarantined after the first MiniMax-H3 I2V Metal watchdog timeout \
             and is recycling before another claim. Original failure: {timeout}"
        ),
        None => "The MLX process is quarantined after \
                 kIOGPUCommandBufferCallbackErrorSubmissionsIgnored and is recycling before \
                 another claim. The originating Metal failure was not captured by this process."
            .to_owned(),
    }
}

pub(crate) struct RecycleRequest {
    pub(crate) reason: String,
    pub(crate) first_timeout: Option<String>,
    pub(crate) saw_submissions_ignored: bool,
}

/// Result of one injected active-job terminal persistence attempt.
pub(crate) enum TerminalPersistenceAttempt {
    /// The truthful failure reached the API. Recycling may now be armed.
    Persisted,
    /// The write failed transiently. Remain quarantined and retry without claiming.
    Retry,
    /// Shutdown interrupted retrying. Remain quarantined and do not arm recycling.
    Stop,
}

/// Drive the terminal-write lifecycle through an injected async attempt.
///
/// Production injects the real `fail_job` POST plus unhealthy heartbeat/backoff. Tests inject
/// deterministic results, which pins the load-bearing ordering: the first failure cannot restore
/// claim eligibility or arm recycling, and only a later successful POST advances the state.
pub(crate) async fn persist_terminal_failure_with<P, Fut>(
    state: &MlxWorkerRecovery,
    mut attempt: P,
) -> bool
where
    P: FnMut() -> Fut,
    Fut: Future<Output = TerminalPersistenceAttempt>,
{
    loop {
        match attempt().await {
            TerminalPersistenceAttempt::Persisted => {
                state.note_terminal_failure_persisted();
                return true;
            }
            TerminalPersistenceAttempt::Retry => {}
            TerminalPersistenceAttempt::Stop => return false,
        }
    }
}

/// The process-global state consulted by the job terminal seam and the next claim turn.
pub(crate) fn global() -> &'static MlxWorkerRecovery {
    static STATE: MlxWorkerRecovery = MlxWorkerRecovery::new();
    &STATE
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    const TIMEOUT: &str = "MiniMax-H3 MLX I2V keyframe VAE conditioning failed: MLX op failed: \
        [METAL] Command buffer execution failed: Caused GPU Timeout Error \
        (00000002:kIOGPUCommandBufferCallbackErrorTimeout)";
    const LATER_TIMEOUT: &str = "MiniMax-H3 MLX I2V grounded Qwen3-VL vision/text conditioning \
        failed: kIOGPUCommandBufferCallbackErrorTimeout: a different timeout";
    const IGNORED: &str = "MiniMax-H3 MLX I2V keyframe VAE conditioning failed: MLX op failed: \
        Ignored (for causing prior/excessive GPU errors) \
        (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)";

    fn first_timeout(state: &MlxWorkerRecovery) -> Option<String> {
        state
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .first_timeout
            .clone()
    }

    #[test]
    fn first_timeout_survives_later_secondary_and_later_primary_errors() {
        let state = MlxWorkerRecovery::new();
        state.observe(TIMEOUT).expect("timeout must poison");
        let detail = state.observe(IGNORED).expect("fallout must remain poison");
        state
            .observe(LATER_TIMEOUT)
            .expect("later timeout remains poison");

        assert_eq!(first_timeout(&state).as_deref(), Some(TIMEOUT));
        assert!(detail.starts_with(TIMEOUT), "primary was hidden: {detail}");
        assert!(detail.contains("secondary fallout"), "{detail}");
        assert!(!detail.contains("ran out of memory"), "{detail}");
    }

    #[test]
    fn secondary_before_primary_never_becomes_the_primary_error() {
        let state = MlxWorkerRecovery::new();
        state.observe(IGNORED).expect("ignored submission poisons");
        assert_eq!(first_timeout(&state), None);

        let detail = state.observe(TIMEOUT).expect("timeout must poison");
        assert_eq!(first_timeout(&state).as_deref(), Some(TIMEOUT));
        assert!(detail.starts_with(TIMEOUT), "primary was hidden: {detail}");
        assert!(detail.contains("secondary fallout"), "{detail}");
    }

    #[test]
    fn concurrent_timeout_and_secondary_observations_converge_on_the_timeout() {
        let state = Arc::new(MlxWorkerRecovery::new());
        let start = Arc::new(Barrier::new(3));
        let timeout_thread = {
            let state = Arc::clone(&state);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                state.observe(TIMEOUT)
            })
        };
        let secondary_thread = {
            let state = Arc::clone(&state);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                state.observe(IGNORED)
            })
        };
        start.wait();
        timeout_thread.join().expect("timeout observer");
        secondary_thread.join().expect("secondary observer");

        assert_eq!(first_timeout(&state).as_deref(), Some(TIMEOUT));
        state.note_terminal_failure_persisted();
        let request = state.begin_recycle().expect("one recycle");
        assert!(request.reason.contains(TIMEOUT), "{}", request.reason);
        assert!(request.saw_submissions_ignored);
    }

    #[tokio::test]
    async fn failed_terminal_post_stays_quarantined_then_success_arms_one_recycle() {
        use std::cell::Cell;

        let state = MlxWorkerRecovery::new();
        state.observe(TIMEOUT).expect("timeout must poison");
        let attempts = Cell::new(0_u32);

        let persisted = persist_terminal_failure_with(&state, || {
            let attempt = attempts.get() + 1;
            attempts.set(attempt);
            if attempt == 1 {
                assert!(!state.can_claim(), "failed POST must not reopen claims");
                assert!(
                    state.begin_recycle().is_none(),
                    "failed POST must not arm the clean-exit recycle"
                );
                std::future::ready(TerminalPersistenceAttempt::Retry)
            } else {
                assert!(!state.can_claim(), "retry must remain quarantined");
                assert!(
                    state.begin_recycle().is_none(),
                    "recycle must still be unarmed immediately before POST success"
                );
                std::future::ready(TerminalPersistenceAttempt::Persisted)
            }
        })
        .await;

        assert!(persisted, "the second terminal POST succeeded");
        assert_eq!(attempts.get(), 2, "one failure was retried exactly once");
        assert!(!state.can_claim(), "success arms recycle, not reuse");
        assert!(state.begin_recycle().is_some(), "success arms recycling");
        assert!(state.begin_recycle().is_none(), "recycle remains one-shot");
    }

    #[test]
    fn quarantine_refuses_the_old_process_and_replacement_runs_the_known_good_job() {
        fn run_known_good_small_job(
            process: &MlxWorkerRecovery,
            run_count: &mut usize,
        ) -> Result<(), &'static str> {
            if !process.can_claim() {
                return Err("process is quarantined");
            }
            *run_count += 1;
            Ok(())
        }

        let poisoned_process = MlxWorkerRecovery::new();
        assert!(poisoned_process.can_claim());
        poisoned_process
            .observe(TIMEOUT)
            .expect("timeout must poison");
        assert!(!poisoned_process.can_claim());
        assert!(
            poisoned_process.begin_recycle().is_none(),
            "recycling must wait for the truthful active-job terminal write"
        );

        poisoned_process.note_terminal_failure_persisted();
        let recycle = poisoned_process.begin_recycle().expect("first recycle");
        assert_eq!(recycle.first_timeout.as_deref(), Some(TIMEOUT));
        assert!(
            poisoned_process.begin_recycle().is_none(),
            "one recycle only"
        );

        let replacement_process = MlxWorkerRecovery::new();
        let mut small_job_runs = 0;
        assert_eq!(
            run_known_good_small_job(&poisoned_process, &mut small_job_runs),
            Err("process is quarantined")
        );
        run_known_good_small_job(&replacement_process, &mut small_job_runs)
            .expect("fresh process runs the known-good small job");
        assert_eq!(
            small_job_runs, 1,
            "the known-good small job must run only in the replacement process"
        );
    }

    #[test]
    fn t2v_other_families_and_ooms_do_not_quarantine_the_worker() {
        let state = MlxWorkerRecovery::new();
        for diagnostic in [
            "MiniMax-H3 MLX T2V failed: kIOGPUCommandBufferCallbackErrorTimeout",
            "Wan MLX I2V failed: kIOGPUCommandBufferCallbackErrorTimeout",
            "MiniMax-H3 MLX I2V keyframe VAE conditioning failed: [metal::malloc] out of memory",
        ] {
            assert_eq!(state.observe(diagnostic), None, "{diagnostic}");
        }
        assert!(state.can_claim());
        assert!(state.begin_recycle().is_none());
    }
}
