//! Metal command-queue poisoning: detection, scope discrimination, containment (sc-20572).
//!
//! # What this is about
//!
//! When a Metal command buffer fails badly enough, the macOS GPU driver stops accepting the
//! offending client's submissions altogether and completes every later command buffer with
//! `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored` — literally *"ignored for causing
//! prior/excessive GPU errors"*. mlx-rs surfaces that as an `Exception`, which the engine
//! `.unwrap()`s, which the model-cache seam contains as a panic
//! ([`crate::cache_thread::run_cached_with_access_after_cold_admission`]).
//!
//! Two things were wrong with how that was contained, both measured in production on 2026-08-19:
//!
//! 1. **It was reported as an out-of-memory condition.** `SubmissionsIgnored` is *never* the first
//!    error — it is the driver refusing a channel it already failed something on. The OOM guess is
//!    right for the FIRST failure (a `[metal::malloc]` throw really is an allocation failure) and
//!    wrong for every ignored submission after it, and it sends diagnosis to the wrong place. So
//!    this class is attributed on its own terms, and the process's *first* contained failure — the
//!    one that actually explains the cascade — is surfaced alongside it.
//! 2. **It claimed the cached generator was reset, as though that recovered anything.** Dropping
//!    the resident model does not un-poison a command queue: the driver's refusal is keyed on the
//!    process, not on any allocation the cache holds. Measured: ~8 back-to-back jobs (bernini →
//!    boogu → krea_2_raw) each hit the containment, each reported a successful reset, and each
//!    failed identically over 2.5 minutes, until libc++abi hit an uncaught `std::runtime_error` and
//!    `abort()`ed the worker. The process respawn is what actually recovered it. So a reset is
//!    never the response here: the resident model is left alone, every subsequent job is refused
//!    WITHOUT touching the GPU, and the worker exits for its supervisor to restart.
//!
//! # The two scopes, and why the message must not pick a reboot on its own
//!
//! The identical error string covers two situations that look the same in the log:
//!
//! * **Process-poisoned** — this process blew its own allocation; other Metal clients keep
//!   rendering; a process restart clears it, and **no reboot is involved**.
//! * **Host-wedged** — the GPU client class itself is wedged (the `pkill`-mid-render trigger seen
//!   on 2026-08-17); fresh processes fail too, and only a reboot clears it.
//!
//! The authoritative discriminator is cross-process and therefore outside this worker: *is a
//! DIFFERENT MLX process completing GPU work?* On 08-19 a concurrent real-weights run was logging
//! healthy forwards throughout, which is what proved the scope was per-process. This module
//! therefore never *asserts* a reboot. It reports the evidence it actually holds — whether THIS
//! process had driven the GPU before the poisoning — and, when that evidence points away from this
//! process, it hands the operator the cross-process check to run before concluding anything about
//! the host.
//!
//! # Why both scopes still restart the worker
//!
//! The restart is the only recovery available in-process, it is what demonstrably fixed the
//! measured incident, and it doubles as the operator-visible discriminator: a fresh process that
//! fails the same way is the "fresh processes fail too" evidence for a host wedge. Parking the
//! worker alive-but-idle on the host-suspected branch instead would strand it on any
//! misclassification, and the classification here is deliberately evidence-poor (see
//! [`GpuPoisonState::note_gpu_work_completed`]). The scope changes what the operator is TOLD, not
//! whether the worker comes back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

/// The Metal command-buffer callback error the driver reports once it is refusing a process's
/// submissions: `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`. Matched case-insensitively on
/// the distinctive tail so the full enum name, a bare `SubmissionsIgnored`, and any surrounding
/// mlx-rs `Exception(...)` framing all hit.
const SUBMISSIONS_IGNORED: &str = "submissionsignored";

/// Which scope the evidence points at for an observed command-queue poisoning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PoisonScope {
    /// This process had already driven the GPU (completed work, or produced a contained fault of
    /// its own) before the driver started ignoring it. The poisoned submission channel is therefore
    /// this process's, and a restart clears it. No reboot.
    Process,
    /// This process had driven no GPU work at all before the driver started ignoring it, so the
    /// errors being blamed came from before it started. A restart is not known to clear this; the
    /// operator must run the cross-process discriminator before concluding the host needs a reboot.
    HostSuspected,
}

impl PoisonScope {
    /// Stable token for the structured log field.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::HostSuspected => "host_suspected",
        }
    }
}

/// What a contained panic turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ContainedPanic {
    /// The driver is ignoring this process's submissions. Carries the fully-composed user-facing
    /// text, so no caller has to re-derive the wording (or re-derive it differently).
    PoisonedQueue { scope: PoisonScope, message: String },
    /// Anything else — an ordinary engine panic, including a genuine first-failure Metal OOM. The
    /// caller keeps its existing containment (evict the resident, report with its own prefix).
    Unclassified,
}

/// Whether `payload` is the driver's ignored-submission report.
fn is_submissions_ignored(payload: &str) -> bool {
    payload.to_ascii_lowercase().contains(SUBMISSIONS_IGNORED)
}

/// Per-process state backing the detection. Instantiable so the decisions are unit-testable against
/// a fresh instance instead of only through the process-global [`global`].
pub(crate) struct GpuPoisonState {
    drove_gpu: AtomicBool,
    first_failure: Mutex<Option<String>>,
    scope: Mutex<Option<PoisonScope>>,
}

impl GpuPoisonState {
    pub(crate) const fn new() -> Self {
        Self {
            drove_gpu: AtomicBool::new(false),
            first_failure: Mutex::new(None),
            scope: Mutex::new(None),
        }
    }

    /// Record that this process has successfully completed GPU work.
    ///
    /// This is the whole basis of the scope discrimination, so it is deliberately conservative:
    /// only a model-cache run that returned `Ok` sets it. A path that drives MLX outside those
    /// seams (LoRA training, for instance) does not, which can leave a genuinely process-scoped
    /// poisoning classified as [`PoisonScope::HostSuspected`]. That direction is the safe one — the
    /// host-suspected text tells the operator the restart is being made precisely so they can see
    /// whether a fresh process gets through, and both scopes restart either way.
    pub(crate) fn note_gpu_work_completed(&self) {
        self.drove_gpu.store(true, Ordering::SeqCst);
    }

    /// Classify one contained panic payload, recording whatever it implies for later jobs.
    ///
    /// Non-poison faults count as this process having driven the GPU: a contained engine panic is
    /// exactly the kind of "prior GPU error" a later `SubmissionsIgnored` is blaming, so a process
    /// that produced one owns the poisoning that follows. (A contained panic that had nothing to do
    /// with the GPU also sets it. That biases toward [`PoisonScope::Process`], the less alarming
    /// verdict, and it is corrected by the restart: the fresh process starts with no such history.)
    pub(crate) fn record_contained_panic(&self, payload: &str) -> ContainedPanic {
        let first_failure = self.record_first_failure(payload);
        if !is_submissions_ignored(payload) {
            self.drove_gpu.store(true, Ordering::SeqCst);
            return ContainedPanic::Unclassified;
        }
        let observed = if self.drove_gpu.load(Ordering::SeqCst) {
            PoisonScope::Process
        } else {
            PoisonScope::HostSuspected
        };
        // The FIRST verdict latches. Everything after the poisoning happens in a process the driver
        // is already refusing, so later faults carry no evidence about how it started — and the
        // `drove_gpu` flag they would be read against is meaningless by then.
        let mut latch = self.scope.lock().unwrap_or_else(PoisonError::into_inner);
        let scope = *latch.get_or_insert(observed);
        ContainedPanic::PoisonedQueue {
            scope,
            message: poisoned_queue_message(scope, first_failure.as_deref(), payload),
        }
    }

    /// The latched scope, once a poisoning has been observed. `None` until then.
    pub(crate) fn poisoned(&self) -> Option<PoisonScope> {
        *self.scope.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The latched scope paired with the text an operator should read, once a poisoning has been
    /// observed. The worker loop wants both — the scope as a log field, the text as the `Unhealthy`
    /// status reason — and taking them together keeps them from being read a turn apart.
    pub(crate) fn poisoned_status(&self) -> Option<(PoisonScope, String)> {
        let scope = self.poisoned()?;
        Some((
            scope,
            refusal_message(scope, self.first_failure().as_deref()),
        ))
    }

    /// User-facing text for a job arriving AFTER the latch: it is refused outright rather than
    /// handed to a GPU that will only produce another ignored submission.
    pub(crate) fn refusal_message(&self) -> Option<String> {
        Some(self.poisoned_status()?.1)
    }

    /// The first contained failure this process saw — the one that actually explains the cascade.
    pub(crate) fn first_failure(&self) -> Option<String> {
        self.first_failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Store `payload` as the first contained failure if nothing is stored yet, and return whatever
    /// is stored (which is `payload` itself on the first call).
    fn record_first_failure(&self, payload: &str) -> Option<String> {
        let mut first = self
            .first_failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Some(first.get_or_insert_with(|| payload.to_owned()).clone())
    }
}

/// What the operator should do about a poisoning of this scope. Shared by every message this module
/// composes so the job error, the refusal and the worker's status reason cannot drift apart.
fn scope_remedy(scope: PoisonScope) -> &'static str {
    match scope {
        PoisonScope::Process => {
            "This worker had already driven the GPU before the poisoning, so the refused submission \
             channel is this process's own: it is restarting now, which clears it. No reboot is \
             needed."
        }
        PoisonScope::HostSuspected => {
            "This worker had completed no GPU work before the poisoning, so the errors the driver \
             is refusing over came from before it started and a restart may not clear them. The \
             worker is restarting so you can see whether a fresh process gets through. If it fails \
             the same way again, check whether a DIFFERENT MLX process is still completing GPU work \
             before concluding the host needs a reboot: if another process renders fine the scope is \
             per-process, but if every fresh process fails then the GPU client class is wedged \
             host-wide and only a reboot clears it."
        }
    }
}

/// The user-facing text for a job refused because the latch is already set.
fn refusal_message(scope: PoisonScope, first_failure: Option<&str>) -> String {
    let mut message = String::from(
        "MLX generation was refused without touching the GPU: the driver is ignoring this worker's \
         Metal command submissions after prior GPU errors \
         (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored), so any work handed to it now would \
         fail the same way. ",
    );
    message.push_str(scope_remedy(scope));
    if let Some(first) = first_failure {
        message.push_str(" The first failure in this worker process was: ");
        message.push_str(first);
    }
    message
}

/// The user-facing error for the job that ran into the poisoning.
///
/// `first_failure` is this process's first contained failure; it is surfaced only when it differs
/// from `payload`, because that is exactly the case where the cascade is hiding the real cause.
fn poisoned_queue_message(
    scope: PoisonScope,
    first_failure: Option<&str>,
    payload: &str,
) -> String {
    let mut message = String::from(
        "MLX generation could not run: the GPU driver is ignoring this worker's Metal command \
         submissions after prior GPU errors (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored). \
         That is the cascade, not the original fault — an ignored submission is never the first \
         error and is not an out-of-memory condition, and resetting the cached generator cannot \
         clear a poisoned command queue. ",
    );
    message.push_str(scope_remedy(scope));
    match first_failure.filter(|first| *first != payload) {
        Some(first) => {
            message.push_str(" The first failure in this worker process was: ");
            message.push_str(first);
        }
        None => message.push_str(
            " No earlier failure was recorded in this worker process, so the errors being refused \
             over are not ones it reported.",
        ),
    }
    message.push_str(" Driver error: ");
    message.push_str(payload);
    message
}

/// The process-global state the production seams consult.
pub(crate) fn global() -> &'static GpuPoisonState {
    static STATE: GpuPoisonState = GpuPoisonState::new();
    &STATE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape mlx-rs hands the containment, from the 2026-08-19 production log.
    const IGNORED: &str =
        "called `Result::unwrap()` on an `Err` value: Exception(\"[METAL] Command \
                           buffer execution failed: Ignored (for causing prior/excessive GPU \
                           errors) \
                           (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)\")";
    const METAL_OOM: &str = "[metal::malloc] Attempting to allocate 41231237120 bytes which is \
                             greater than the maximum allowed buffer size";

    fn poisoned_message(state: &GpuPoisonState, payload: &str) -> String {
        match state.record_contained_panic(payload) {
            ContainedPanic::PoisonedQueue { message, .. } => message,
            ContainedPanic::Unclassified => panic!("expected a poisoned-queue classification"),
        }
    }

    fn scope_of(state: &GpuPoisonState, payload: &str) -> PoisonScope {
        match state.record_contained_panic(payload) {
            ContainedPanic::PoisonedQueue { scope, .. } => scope,
            ContainedPanic::Unclassified => panic!("expected a poisoned-queue classification"),
        }
    }

    #[test]
    fn an_ordinary_engine_panic_is_not_a_poisoned_queue_and_does_not_latch() {
        let state = GpuPoisonState::new();
        assert_eq!(
            state.record_contained_panic(METAL_OOM),
            ContainedPanic::Unclassified
        );
        assert_eq!(
            state.poisoned(),
            None,
            "a first-failure Metal OOM must not make the worker refuse later jobs"
        );
        assert_eq!(state.refusal_message(), None);
    }

    #[test]
    fn ignored_submissions_after_completed_gpu_work_are_scoped_to_this_process() {
        let state = GpuPoisonState::new();
        state.note_gpu_work_completed();
        assert_eq!(scope_of(&state, IGNORED), PoisonScope::Process);
        assert_eq!(state.poisoned(), Some(PoisonScope::Process));
    }

    #[test]
    fn ignored_submissions_in_a_process_that_never_drove_the_gpu_suspect_the_host() {
        let state = GpuPoisonState::new();
        assert_eq!(scope_of(&state, IGNORED), PoisonScope::HostSuspected);
        assert_eq!(state.poisoned(), Some(PoisonScope::HostSuspected));
    }

    #[test]
    fn an_earlier_contained_fault_scopes_a_later_poisoning_to_this_process() {
        let state = GpuPoisonState::new();
        assert_eq!(
            state.record_contained_panic(METAL_OOM),
            ContainedPanic::Unclassified
        );
        assert_eq!(
            scope_of(&state, IGNORED),
            PoisonScope::Process,
            "the fault this process already produced is the 'prior GPU error' being refused over"
        );
    }

    #[test]
    fn the_cascade_message_surfaces_the_first_failure_not_just_the_ignored_submission() {
        let state = GpuPoisonState::new();
        state.record_contained_panic(METAL_OOM);
        let message = poisoned_message(&state, IGNORED);
        assert!(
            message.contains("The first failure in this worker process was:"),
            "{message}"
        );
        assert!(message.contains("41231237120 bytes"), "{message}");
        assert_eq!(state.first_failure().as_deref(), Some(METAL_OOM));
    }

    #[test]
    fn a_poisoning_with_no_earlier_fault_says_so_instead_of_repeating_itself() {
        let state = GpuPoisonState::new();
        let message = poisoned_message(&state, IGNORED);
        assert!(
            message.contains("No earlier failure was recorded in this worker process"),
            "{message}"
        );
        assert!(
            !message.contains("The first failure in this worker process was:"),
            "the ignored submission must not be presented as the cause of itself: {message}"
        );
    }

    #[test]
    fn no_composed_message_blames_memory_or_claims_a_reset_recovered_anything() {
        for scope in [PoisonScope::Process, PoisonScope::HostSuspected] {
            let state = GpuPoisonState::new();
            if scope == PoisonScope::Process {
                state.note_gpu_work_completed();
            }
            let job = poisoned_message(&state, IGNORED);
            let refusal = state.refusal_message().expect("latched");
            for message in [job, refusal] {
                assert!(
                    !message.contains("ran out of memory"),
                    "{scope:?} attribution still blames memory: {message}"
                );
                assert!(
                    !message.contains("generator was reset"),
                    "{scope:?} attribution still claims a reset: {message}"
                );
            }
        }
    }

    #[test]
    fn the_process_scoped_remedy_rules_a_reboot_out_and_the_host_scoped_one_gates_it() {
        let process = GpuPoisonState::new();
        process.note_gpu_work_completed();
        let process_message = poisoned_message(&process, IGNORED);
        assert!(
            process_message.contains("No reboot is needed"),
            "{process_message}"
        );

        let host = GpuPoisonState::new();
        let host_message = poisoned_message(&host, IGNORED);
        assert!(
            host_message.contains("DIFFERENT MLX process"),
            "the host-suspected remedy must hand over the cross-process discriminator: \
             {host_message}"
        );
        assert!(
            host_message.contains("before concluding the host needs a reboot"),
            "the discriminator must gate the reboot, not follow it: {host_message}"
        );
    }

    #[test]
    fn the_first_verdict_latches_and_later_faults_cannot_re_scope_it() {
        let state = GpuPoisonState::new();
        assert_eq!(scope_of(&state, IGNORED), PoisonScope::HostSuspected);
        // A later job on the poisoned queue looks identical, and by then the process has "driven
        // the GPU" only in the sense that it produced faults on a channel already being refused.
        state.note_gpu_work_completed();
        assert_eq!(scope_of(&state, IGNORED), PoisonScope::HostSuspected);
        assert_eq!(state.poisoned(), Some(PoisonScope::HostSuspected));
    }

    #[test]
    fn a_latched_worker_refuses_later_jobs_without_touching_the_gpu() {
        let state = GpuPoisonState::new();
        state.note_gpu_work_completed();
        state.record_contained_panic(IGNORED);
        let refusal = state.refusal_message().expect("latched");
        assert!(
            refusal.contains("refused without touching the GPU"),
            "{refusal}"
        );
        assert!(refusal.contains("No reboot is needed"), "{refusal}");
        assert_eq!(
            state.poisoned_status(),
            Some((PoisonScope::Process, refusal)),
            "the worker loop must read the same scope and text the refusal reports"
        );
    }

    #[test]
    fn the_driver_error_is_recognized_by_its_enum_name_in_any_case() {
        for payload in [
            "kIOGPUCommandBufferCallbackErrorSubmissionsIgnored",
            "00000004:KIOGPUCOMMANDBUFFERCALLBACKERRORSUBMISSIONSIGNORED",
            "submissionsignored",
            IGNORED,
        ] {
            assert!(is_submissions_ignored(payload), "{payload}");
        }
        for payload in [
            METAL_OOM,
            "[METAL] Command buffer execution failed: Caused GPU Timeout Error",
            "submissions ignored",
        ] {
            assert!(!is_submissions_ignored(payload), "{payload}");
        }
    }
}
