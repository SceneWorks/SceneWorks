//! Worker error type ([`WorkerError`]), its `From` conversions, and the [`WorkerResult`] alias.
use super::*;

thread_local! {
    // A typed provider fault is terminal for this owner thread. Destructors below the
    // cache owner's return path must not submit another command to the failed device.
    static OWNER_PROCESS_POISONED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn owner_process_poisoned() -> bool {
    OWNER_PROCESS_POISONED.with(std::cell::Cell::get)
}

/// The one gate for worker-side device work that is not the job itself: cleanup, cache release,
/// eviction drops and scope teardown. False while unwinding, once this thread classified a typed
/// provider fault, or once the process is quarantined for one (latched on any thread). Every
/// device-touching destructor or release the worker owns checks it; when it is false the object is
/// forgotten instead and OS process retirement releases it.
pub(crate) fn owner_device_cleanup_allowed() -> bool {
    if std::thread::panicking() || owner_process_poisoned() {
        return false;
    }
    #[cfg(not(test))]
    if !crate::mlx_worker_recovery::global().can_claim() {
        return false;
    }
    true
}

/// A device-owning value (a loaded model) whose drop goes through [`release_or_forget`]. An
/// uncached producer holds its model in one so a typed fault classified mid-run never lets the
/// model's destructor touch the poisoned device on the way out.
pub(crate) struct DeviceOwned<T>(Option<T>);

impl<T> DeviceOwned<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T> std::ops::Deref for DeviceOwned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
            .as_ref()
            .expect("device-owned value is present until drop")
    }
}

impl<T> std::ops::DerefMut for DeviceOwned<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0
            .as_mut()
            .expect("device-owned value is present until drop")
    }
}

impl<T> Drop for DeviceOwned<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            release_or_forget(value);
        }
    }
}

/// Drop `value` when device cleanup is still allowed, otherwise forget it (see
/// [`owner_device_cleanup_allowed`]).
pub(crate) fn release_or_forget<T>(value: T) {
    if owner_device_cleanup_allowed() {
        drop(value);
    } else {
        std::mem::forget(value);
    }
}

fn mark_owner_process_poisoned() {
    OWNER_PROCESS_POISONED.with(|poisoned| poisoned.set(true));
}

/// Latch a typed provider fault process-wide at the moment it is classified, on the producer's own
/// thread. The cache boundary latches its faults too, but an uncached `spawn_blocking` producer has
/// no such boundary: when a progress/heartbeat POST failure made the consumer abandon it after the
/// cancel grace, its later `ProcessPoisoned` reached nobody and the worker kept claiming jobs on the
/// poisoned device. Latched here, the quarantine stands regardless of any consumer, and the idle
/// worker loop adopts and recycles it. `observe_process_fault` keeps the first detail, so the cache
/// boundary and `handle_job_failure` observing the same fault again are no-ops.
fn latch_process_fault(detail: &str) {
    #[cfg(not(test))]
    crate::mlx_worker_recovery::global().observe_process_fault(detail);
    // Tests never quarantine the shared test binary's production singleton; they read the latch.
    #[cfg(test)]
    test_latched_process_faults()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(detail.to_owned());
}

#[cfg(test)]
pub(crate) fn test_latched_process_faults() -> &'static std::sync::Mutex<Vec<String>> {
    static LATCHED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    &LATCHED
}

#[derive(Debug)]
pub enum WorkerError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    ProjectStore(ProjectStoreError),
    Api {
        status: StatusCode,
        detail: String,
        code: Option<String>,
    },
    InvalidPayload(String),
    Validation(sceneworks_core::request_validation::ValidationRefusal),
    ReplayUnavailable(sceneworks_core::execution_receipt::ReplayUnavailable),
    /// The configured external model library holding an installed model cannot currently be
    /// proven present (disconnected, or a different physical volume now occupies its path).
    /// Distinct from [`Self::InvalidPayload`]/[`Self::Engine`] so a disconnect is never surfaced
    /// as a raw loader ENOENT and never triggers a silent re-download of installed state.
    ExternalLibraryUnavailable(String),
    Engine(String),
    /// The owner executor caught a panic or an explicitly typed fatal fault. Its backend state
    /// cannot be touched again in this process; terminal accounting keeps this original detail.
    ProcessPoisoned(String),
    Canceled(String),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::ProjectStore(error) => write!(formatter, "{error}"),
            Self::Api {
                status,
                detail,
                code,
            } => match code {
                Some(code) => write!(formatter, "API {status} ({code}): {detail}"),
                None => write!(formatter, "API {status}: {detail}"),
            },
            Self::InvalidPayload(detail) => formatter.write_str(detail),
            Self::Validation(error) => error.fmt(formatter),
            Self::ReplayUnavailable(reason) => write!(formatter, "{}", reason.reason),
            Self::ExternalLibraryUnavailable(detail) => write!(
                formatter,
                "[{}] {detail}",
                sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE
            ),
            Self::Engine(detail) => formatter.write_str(detail),
            Self::ProcessPoisoned(detail) => formatter.write_str(detail),
            Self::Canceled(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<reqwest::Error> for WorkerError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<std::io::Error> for WorkerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for WorkerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<ProjectStoreError> for WorkerError {
    fn from(value: ProjectStoreError) -> Self {
        Self::ProjectStore(value)
    }
}

pub(crate) fn task_join_error(label: &str, error: tokio::task::JoinError) -> WorkerError {
    WorkerError::Io(std::io::Error::other(format!("{label}: {error}")))
}

/// Classify a [`gen_core::Error`] surfacing from an engine load / generate call into a
/// [`WorkerError`], distinguishing a user-actionable capability gap from a generic internal failure.
///
/// A [`gen_core::Error::Unsupported`] is the engine's typed "the request asked for something this
/// engine/backend cannot do" signal — e.g. `reject_loha_on_packed` on quantized Wan, which carries
/// an actionable steer-to-bf16 message (sc-10051, epic 10043). We surface it as
/// [`WorkerError::InvalidPayload`] (the worker's user-facing / validation class) with the engine's
/// message text INTACT, rather than burying it in an opaque [`WorkerError::Engine`]. Typed
/// cancellation stays [`WorkerError::Canceled`]; everything else stays [`WorkerError::Engine`].
/// `context` prefixes the message so the origin (load vs generate)
/// stays legible, matching the existing `format!("{context}: {error}")` seams.
///
/// ## Driver-class CUDA failures (sc-16247, GH #1966)
///
/// A CUDA driver/userspace version mismatch used to reach the user's screen verbatim: this
/// function's message becomes `fail_job`'s failure detail (`lib.rs`), which `QueueScreen.jsx`
/// renders as `job.error`, so GH #1966 read
/// `candle krea_2_turbo load failed: backend op failed: DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
/// "system has unsupported display driver / cuda driver combination")` — with no hint that the fix
/// is on the reporter's machine, not in ours. When the engine error carries a known `CUDA_ERROR_*`
/// token, the actionable host-side remedy now LEADS the message and the raw error is appended for
/// the logs. The token table is shared with [`crate::cuda_preflight`] so the same failure reads
/// identically whether it is caught at startup or escapes into a running job; see
/// [`crate::preflight::cuda_driver_error_guidance`] for why this matches on the string rather than
/// downcasting.
///
/// Matched on the whole stringified error rather than only `Error::Backend`, because a provider is
/// free to surface a driver failure as `Error::Msg` — the token is what identifies it, not the
/// variant it arrived in.
///
/// These stay [`WorkerError::Engine`] deliberately. A driver mismatch is not an invalid payload,
/// and the variant carries no behaviour to change: every non-test site that branches on a
/// `WorkerError` variant during job completion special-cases only `Canceled` (`lib.rs`) — `Engine`
/// and `InvalidPayload` both fall through to the identical `fail_job` call and terminate the job.
/// There is no retry semantics attached to the distinction, so promoting the variant would add an
/// unread distinction; the defect was the message, and the message is what changed.
pub(crate) fn classify_engine_error(context: &str, error: gen_core::Error) -> WorkerError {
    match error {
        gen_core::Error::Canceled => WorkerError::Canceled(format!("{context}: {error}")),
        // A provider has already classified a fatal process state. Keep its original detail
        // byte-for-byte so the owner and terminal paths can retire without string inference.
        gen_core::Error::ProcessPoisoned(detail) => {
            mark_owner_process_poisoned();
            latch_process_fault(&detail);
            WorkerError::ProcessPoisoned(detail)
        }
        gen_core::Error::Unsupported(_) => {
            WorkerError::InvalidPayload(format!("{context}: {error}"))
        }
        other => {
            let detail = other.to_string();
            match crate::preflight::cuda_driver_error_guidance(&detail) {
                Some(guidance) => WorkerError::Engine(format!(
                    "{context}: {guidance}\n\nUnderlying CUDA error: {detail}"
                )),
                None => WorkerError::Engine(format!("{context}: {detail}")),
            }
        }
    }
}

/// The fatal process fault a finished task returned, if any. A consumer checks it BEFORE its
/// user-cancel branch: a cancel-shaped terminal would skip the quarantine, terminal persistence
/// and recycle that `handle_job_failure` gives `ProcessPoisoned`, leaving the worker free to claim
/// another GPU job on poisoned state.
pub(crate) fn process_fault<T>(result: &WorkerResult<T>) -> Option<WorkerError> {
    match result {
        Err(WorkerError::ProcessPoisoned(detail)) => {
            Some(WorkerError::ProcessPoisoned(detail.clone()))
        }
        _ => None,
    }
}

/// A provider error type that can carry a typed `ProcessPoisoned` and bridges it 1:1 into
/// [`gen_core::Error`]. Exactly the three provider error contracts with that variant implement it,
/// so [`generation_error`] / [`provider_error`] cannot be applied to an error that never carries it.
pub(crate) trait ProviderError: Into<gen_core::Error> + std::fmt::Display {
    fn is_process_poisoned(&self) -> bool;
}

impl ProviderError for gen_core::Error {
    fn is_process_poisoned(&self) -> bool {
        matches!(self, Self::ProcessPoisoned(_))
    }
}

#[cfg(target_os = "macos")]
impl ProviderError for runtime_macos::media::Error {
    fn is_process_poisoned(&self) -> bool {
        matches!(self, Self::ProcessPoisoned(_))
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl ProviderError for runtime_cuda::media::CandleError {
    fn is_process_poisoned(&self) -> bool {
        matches!(self, Self::ProcessPoisoned(_))
    }
}

/// THE precedence rule for a provider result and the result of its cleanup (a scope `finish`, a
/// teardown): a typed `ProcessPoisoned` on either side wins over every other outcome; otherwise the
/// primary error wins; otherwise the cleanup error fails an otherwise successful run.
pub(crate) fn combine_provider_outcomes<T, E: ProviderError>(
    primary: Result<T, E>,
    cleanup: Result<(), E>,
) -> Result<T, E> {
    match (primary, cleanup) {
        (Err(primary), _) if primary.is_process_poisoned() => Err(primary),
        (_, Err(cleanup)) if cleanup.is_process_poisoned() => Err(cleanup),
        (Err(primary), _) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Ok(value), Ok(())) => Ok(value),
    }
}

/// [`combine_provider_outcomes`] for worker results: a `ProcessPoisoned` on either side wins;
/// two ordinary failures are merged by `both` (each site keeps its existing message shape).
pub(crate) fn combine_outcomes<T>(
    primary: WorkerResult<T>,
    cleanup: WorkerResult<()>,
    both: impl FnOnce(WorkerError, WorkerError) -> WorkerError,
) -> WorkerResult<T> {
    match (primary, cleanup) {
        (Err(primary @ WorkerError::ProcessPoisoned(_)), _) => Err(primary),
        (_, Err(cleanup @ WorkerError::ProcessPoisoned(_))) => Err(cleanup),
        (Err(primary), Err(cleanup)) => Err(both(primary, cleanup)),
        (Err(primary), Ok(())) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Ok(value), Ok(())) => Ok(value),
    }
}

/// [`finish_scope`] on a path that is already leaving with its own outcome (a cancel or an
/// ordinary failure): an ordinary cleanup error must not mask that outcome, but a typed
/// `ProcessPoisoned` from the finish outranks it and is classified (and latched) here.
pub(crate) fn finish_scope_after_failure<S, E: ProviderError>(
    scope: S,
    finish: impl FnOnce(&mut S) -> Result<(), E>,
    context: &str,
) -> WorkerResult<()> {
    match finish_scope(scope, finish) {
        Err(error) if error.is_process_poisoned() => Err(generation_error(context, error)),
        _ => Ok(()),
    }
}

/// Run a provider scope's terminal `finish` and dispose of the scope under the one rule: when the
/// finish reports a typed `ProcessPoisoned`, or device cleanup is otherwise no longer allowed, the
/// scope is forgotten rather than dropped — its destructor may submit more device work.
pub(crate) fn finish_scope<S, E: ProviderError>(
    mut scope: S,
    finish: impl FnOnce(&mut S) -> Result<(), E>,
) -> Result<(), E> {
    let finished = finish(&mut scope);
    if finished
        .as_ref()
        .is_err_and(ProviderError::is_process_poisoned)
    {
        std::mem::forget(scope);
    } else {
        release_or_forget(scope);
    }
    finished
}

/// Map a provider failure on a device path onto the worker's error classes. `E` is any provider
/// error that bridges into [`gen_core::Error`] (`gen_core::Error` itself, `mlx_gen::Error`,
/// `CandleError`, …); each bridge carries `ProcessPoisoned` across 1:1. A poisoned process keeps its
/// typed recovery class through [`classify_engine_error`], which also marks owner cleanup unsafe on
/// this thread. Every other failure is an engine error under `context` whose text is the provider
/// error's own `Display`, byte-identical to the `format!("{context}: {error}")` seams this replaces.
pub(crate) fn generation_error<E: ProviderError>(context: &str, error: E) -> WorkerError {
    provider_error(error, |detail| {
        WorkerError::Engine(format!("{context}: {detail}"))
    })
}

/// [`generation_error`] for a lane whose ordinary failure has its own shape: `ordinary` receives the
/// provider error's `Display` text and builds the error exactly as the lane always has. Only a
/// typed `ProcessPoisoned` bypasses it.
pub(crate) fn provider_error<E: ProviderError>(
    error: E,
    ordinary: impl FnOnce(String) -> WorkerError,
) -> WorkerError {
    let detail = error.to_string();
    match error.into() {
        poisoned @ gen_core::Error::ProcessPoisoned(_) => {
            classify_engine_error("provider failure", poisoned)
        }
        _ => ordinary(detail),
    }
}

/// Prepend the actionable host-side remedy to a terminal job-failure detail when it describes a
/// driver-class CUDA failure, leaving everything else untouched (sc-16247).
///
/// The backstop for [`classify_engine_error`]. AC3 of sc-16247 names that function, and the lane
/// GH #1966 reported does route through it — but roughly 35 other load seams build
/// `WorkerError::Engine(format!("… load failed: {error}"))` directly (`image_jobs/*_candle.rs`,
/// `instantid.rs`, `training_jobs.rs`, …), and a host driver mismatch breaks all of them the same
/// way. Converting 36 call sites would be a large mechanical churn across 20+ files for the same
/// result; annotating once at the job-failure tail, where every terminal detail passes, covers
/// them all.
///
/// **Idempotent**, because a detail that already came through `classify_engine_error` carries the
/// guidance: if the remedy text is present the string is returned unchanged, so the message is
/// never doubled.
pub(crate) fn annotate_cuda_driver_failure(detail: &str) -> String {
    match crate::preflight::cuda_driver_error_guidance(detail) {
        Some(guidance) if !detail.contains(guidance) => {
            format!("{guidance}\n\nUnderlying CUDA error: {detail}")
        }
        _ => detail.to_owned(),
    }
}

pub type WorkerResult<T> = Result<T, WorkerError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn poisoned(detail: &str) -> gen_core::Error {
        gen_core::Error::ProcessPoisoned(detail.into())
    }

    /// The one precedence rule: poison on either side wins over every other outcome; otherwise the
    /// primary error, then the cleanup error.
    #[test]
    fn a_poisoned_result_wins_every_combination() {
        let ordinary = || gen_core::Error::Unsupported("ordinary".into());
        let poison_cleanup = combine_provider_outcomes(Ok::<u8, _>(1), Err(poisoned("cleanup")));
        assert!(
            matches!(poison_cleanup, Err(gen_core::Error::ProcessPoisoned(ref d)) if d == "cleanup")
        );
        let poison_over_cancel = combine_provider_outcomes::<u8, _>(
            Err(gen_core::Error::Canceled),
            Err(poisoned("cleanup")),
        );
        assert!(matches!(
            poison_over_cancel,
            Err(gen_core::Error::ProcessPoisoned(_))
        ));
        let poison_over_error =
            combine_provider_outcomes::<u8, _>(Err(ordinary()), Err(poisoned("cleanup")));
        assert!(matches!(
            poison_over_error,
            Err(gen_core::Error::ProcessPoisoned(_))
        ));
        let primary_poison =
            combine_provider_outcomes::<u8, _>(Err(poisoned("primary")), Err(ordinary()));
        assert!(
            matches!(primary_poison, Err(gen_core::Error::ProcessPoisoned(ref d)) if d == "primary")
        );
        let primary_error =
            combine_provider_outcomes::<u8, _>(Err(ordinary()), Err(gen_core::Error::Canceled));
        assert!(matches!(
            primary_error,
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            combine_provider_outcomes(Ok::<u8, _>(1), Err(ordinary())),
            Err(gen_core::Error::Unsupported(_))
        ));
        assert!(matches!(
            combine_provider_outcomes(Ok::<u8, gen_core::Error>(1), Ok(())),
            Ok(1)
        ));

        let merged = combine_outcomes::<u8>(
            Err(WorkerError::Engine("primary".into())),
            Err(WorkerError::Engine("index".into())),
            |a, b| WorkerError::Engine(format!("{a}; {b}")),
        );
        assert!(
            matches!(merged, Err(WorkerError::Engine(ref d)) if d.contains("primary") && d.contains("index"))
        );
        let poison_cleanup = combine_outcomes::<u8>(
            Err(WorkerError::Engine("primary".into())),
            Err(WorkerError::ProcessPoisoned("cleanup".into())),
            |_, _| unreachable!("poison is never merged away"),
        );
        assert!(
            matches!(poison_cleanup, Err(WorkerError::ProcessPoisoned(ref d)) if d == "cleanup")
        );
        let poison_primary = combine_outcomes::<u8>(
            Err(WorkerError::ProcessPoisoned("primary".into())),
            Err(WorkerError::Engine("index".into())),
            |_, _| unreachable!("poison is never merged away"),
        );
        assert!(
            matches!(poison_primary, Err(WorkerError::ProcessPoisoned(ref d)) if d == "primary")
        );
    }

    struct DropCount(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// A scope whose finish reports poison is forgotten; an ordinary one is dropped. A device-owned
    /// value is dropped normally and forgotten once this thread classified a typed fault.
    #[test]
    fn poisoned_scopes_and_device_owners_are_forgotten_not_dropped() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        std::thread::spawn(|| {
            let drops = Arc::new(AtomicUsize::new(0));
            drop(DeviceOwned::new(DropCount(Arc::clone(&drops))));
            assert_eq!(drops.load(Ordering::SeqCst), 1, "a healthy owner drops");
            let drops = Arc::new(AtomicUsize::new(0));
            let finished = finish_scope(DropCount(Arc::clone(&drops)), |_| {
                Err::<(), _>(gen_core::Error::Unsupported("ordinary".into()))
            });
            assert!(finished.is_err());
            assert_eq!(drops.load(Ordering::SeqCst), 1, "an ordinary finish drops the scope");
            let finished =
                finish_scope(DropCount(Arc::clone(&drops)), |_| Err::<(), _>(poisoned("finish")));
            assert!(matches!(finished, Err(gen_core::Error::ProcessPoisoned(_))));
            assert_eq!(drops.load(Ordering::SeqCst), 1, "a poisoned finish forgets the scope");
            assert!(
                finish_scope_after_failure(DropCount(Arc::clone(&drops)), |_| {
                    Err::<(), _>(gen_core::Error::Unsupported("ordinary".into()))
                }, "cleanup")
                .is_ok(),
                "an ordinary cleanup error never masks the outcome being reported"
            );
            assert!(matches!(
                finish_scope_after_failure(DropCount(Arc::clone(&drops)), |_| Err::<(), _>(poisoned("late")), "cleanup"),
                Err(WorkerError::ProcessPoisoned(ref d)) if d == "late"
            ));
            assert_eq!(drops.load(Ordering::SeqCst), 2);

            let before = drops.load(Ordering::SeqCst);
            // The `finish_scope_after_failure` above classified a typed fault on this thread.
            assert!(owner_process_poisoned());
            drop(DeviceOwned::new(DropCount(Arc::clone(&drops))));
            assert_eq!(drops.load(Ordering::SeqCst), before, "a poisoned owner is forgotten");
        })
        .join()
        .expect("fault thread");
    }

    // Literal typed errors only: no runtime, source lookup, model or device work.
    #[test]
    fn typed_engine_cancellation_remains_cancellation_at_checked_preload_boundaries() {
        for context in [
            "MOSS preload canceled",
            "MOSS adapter preload canceled",
            "MOSS preload facts failed",
        ] {
            let classified = classify_engine_error(context, gen_core::Error::Canceled);
            assert!(
                matches!(&classified, WorkerError::Canceled(_)),
                "{classified:?}"
            );
            assert_eq!(classified.to_string(), format!("{context}: cancelled"));
        }
        // The typed contract must not turn arbitrary failures containing this word into success/cancel.
        let untyped =
            classify_engine_error("MOSS preload", gen_core::Error::Msg("cancelled".into()));
        assert!(matches!(untyped, WorkerError::Engine(_)));
    }

    // sc-10051 / epic 10043: at the new mlx-gen rev, the Wan load path rejects a LoHa adapter on a
    // packed (quantized) base with a typed `gen_core::Error::Unsupported` carrying an actionable
    // steer-to-bf16 message. The worker's load seam (`generator_cache::with_generator`) routes that
    // error through `classify_engine_error`, which must surface it as a USER-FACING validation error
    // (`WorkerError::InvalidPayload`) — distinct from an opaque internal `Engine` failure — WITHOUT
    // swallowing the engine's message text. This injects the typed error at the seam (no real
    // checkpoint) exactly as `crate::inference_runtime::load` would return it.
    #[test]
    fn unsupported_engine_error_surfaces_as_user_facing_validation_with_message_intact() {
        let steer = "LoHa adapters require a bf16 base; this model is quantized (q4). \
                     Switch to the bf16 tier to use this adapter.";
        let engine_error = gen_core::Error::Unsupported(steer.to_owned());

        let classified = classify_engine_error("video load failed", engine_error);

        let WorkerError::InvalidPayload(message) = classified else {
            panic!(
                "an Unsupported engine error must map to WorkerError::InvalidPayload \
                 (user-facing), got {classified:?}"
            );
        };
        // The actionable steer-to-bf16 guidance must reach the user verbatim.
        assert!(
            message.contains(steer),
            "the engine's steer-to-bf16 message must survive intact, got: {message}"
        );
        // And the origin context must be preserved for legibility.
        assert!(
            message.contains("video load failed"),
            "the load-vs-generate context must be preserved, got: {message}"
        );
    }

    // A non-Unsupported engine failure stays an opaque internal `Engine` error (not user-facing),
    // so the classification genuinely distinguishes capability gaps from generic failures.
    #[test]
    fn non_unsupported_engine_error_stays_internal_engine() {
        let engine_error =
            gen_core::Error::MissingTensor("wan.transformer.blocks.0.attn".to_owned());

        let classified = classify_engine_error("video load failed", engine_error);

        assert!(
            matches!(classified, WorkerError::Engine(_)),
            "a generic engine failure must stay WorkerError::Engine, got {classified:?}"
        );
    }

    #[test]
    fn typed_provider_process_fault_keeps_original_detail() {
        let error = classify_engine_error(
            "FLUX.1 generation failed",
            gen_core::Error::ProcessPoisoned("first device fault".into()),
        );
        assert!(
            matches!(error, WorkerError::ProcessPoisoned(ref detail) if detail == "first device fault")
        );
        assert!(owner_process_poisoned());
    }

    /// sc-16247 / GH #1966: the reported failure end-to-end. A Bazzite host whose kernel module
    /// and `libcuda.so` were out of step raised `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` deep inside a
    /// Krea 2 Turbo load; the load seam (`generator_cache::with_generator`) routes that through
    /// here, and the resulting string is what `fail_job` stores and `QueueScreen.jsx` renders. It
    /// must lead with the host-side remedy instead of a raw `DriverError(...)` dump. Injected at
    /// the seam exactly as `crate::inference_runtime::load` would return it — no GPU needed.
    #[test]
    fn cuda_driver_mismatch_surfaces_the_host_side_remedy_not_a_raw_driver_error() {
        let raw = "DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, \
                   \"system has unsupported display driver / cuda driver combination\")";
        let engine_error = gen_core::Error::backend(std::io::Error::other(raw));

        let classified = classify_engine_error("candle krea_2_turbo load failed", engine_error);

        let WorkerError::Engine(message) = classified else {
            panic!("a driver-class CUDA failure stays WorkerError::Engine, got {classified:?}");
        };
        // The actionable part: the user must be told this is their machine's driver state and
        // that a reboot is what fixes it. Asserting on the CONTENT, not merely that it changed.
        assert!(
            message.contains("kernel driver") && message.contains("reboot"),
            "the message must name the host-side remedy, got: {message}"
        );
        // Bazzite/Silverblue is the reported environment and the reason a long uptime doesn't
        // imply an up-to-date module, so the immutable-distro case is called out by name.
        assert!(
            message.contains("Bazzite"),
            "the immutable-distro case must be covered, got: {message}"
        );
        // The raw driver error still has to reach the logs.
        assert!(
            message.contains("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
            "the underlying CUDA error must be preserved, got: {message}"
        );
        // And the origin context must survive, as it does for every other classification.
        assert!(
            message.contains("candle krea_2_turbo load failed"),
            "the load-vs-generate context must be preserved, got: {message}"
        );
    }

    /// The rest of the driver-class family (sc-16247 AC 3) reaches the same treatment, and each
    /// gets its OWN text — a shared fallback would pass a weaker version of this test.
    #[test]
    fn the_whole_driver_class_family_is_classified_with_distinct_messages() {
        let mut seen = Vec::new();
        for token in [
            "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
            "CUDA_ERROR_INSUFFICIENT_DRIVER",
            "CUDA_ERROR_SYSTEM_NOT_READY",
            "CUDA_ERROR_NO_DEVICE",
        ] {
            let engine_error = gen_core::Error::backend(std::io::Error::other(format!(
                "DriverError({token}, \"...\")"
            )));
            let classified = classify_engine_error("candle load failed", engine_error);
            let WorkerError::Engine(message) = classified else {
                panic!("{token} must stay WorkerError::Engine, got {classified:?}");
            };
            assert!(
                message.contains(token),
                "{token} must be preserved in the message for the logs, got: {message}"
            );
            // The rewrite must actually have happened. Without this the test passes on the
            // ORIGINAL raw pass-through: each message would still be "distinct" below, purely
            // because the embedded token differs. Requiring the separator pins the shape —
            // guidance first, raw error after — rather than merely "the strings differ".
            let (guidance, underlying) = message
                .split_once("\n\nUnderlying CUDA error:")
                .unwrap_or_else(|| {
                    panic!("{token} was not rewritten with guidance + a preserved raw error: {message}")
                });
            assert!(
                underlying.contains(token),
                "{token} must appear in the preserved tail, not only the guidance: {message}"
            );
            let guidance = guidance.to_owned();
            assert!(
                !seen.contains(&guidance),
                "{token} reuses another code's guidance; the family must not collapse"
            );
            seen.push(guidance);
        }
        assert_eq!(seen.len(), 4, "all four driver-class codes must be covered");
    }

    /// The job-failure-tail backstop (sc-16247) for the ~35 load seams that build
    /// `WorkerError::Engine` directly instead of going through `classify_engine_error`. Must
    /// annotate those, must leave unrelated failures alone, and must NOT double-annotate a detail
    /// that already came through the classifier.
    #[test]
    fn the_job_failure_tail_annotates_unclassified_driver_errors_exactly_once() {
        // A lane that bypassed `classify_engine_error` (e.g. `image_jobs/krea_edit_candle.rs`).
        let raw = "Krea edit load failed: backend op failed: \
                   DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, \"...\")";
        let annotated = annotate_cuda_driver_failure(raw);
        assert!(
            annotated.contains("reboot") && annotated.contains(raw),
            "a direct-construction lane must still get the remedy, got: {annotated}"
        );

        // Already classified: running it through the tail must change nothing.
        let classified = classify_engine_error(
            "candle krea_2_turbo load failed",
            gen_core::Error::backend(std::io::Error::other(
                "DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, \"...\")",
            )),
        )
        .to_string();
        assert_eq!(
            annotate_cuda_driver_failure(&classified),
            classified,
            "an already-classified detail must not be annotated twice"
        );

        // Unrelated failures pass through byte-for-byte.
        for untouched in ["disk full", "missing tensor: unet.x", ""] {
            assert_eq!(annotate_cuda_driver_failure(untouched), untouched);
        }
    }

    /// The negative control that keeps the rewrite honest: a CUDA OOM is a CUDA error but NOT a
    /// driver-stack problem, so it must pass through with its original text. Without this, a
    /// table that matched `CUDA_ERROR_` as a prefix would tell an out-of-VRAM user to reboot.
    #[test]
    fn a_cuda_oom_is_not_rewritten_as_a_driver_problem() {
        let raw = "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")";
        let engine_error = gen_core::Error::backend(std::io::Error::other(raw));

        let classified = classify_engine_error("candle wan load failed", engine_error);

        let WorkerError::Engine(message) = classified else {
            panic!("a CUDA OOM stays WorkerError::Engine, got {classified:?}");
        };
        assert_eq!(
            message,
            format!("candle wan load failed: backend op failed: {raw}"),
            "a non-driver CUDA failure must pass through untouched"
        );
    }
}
