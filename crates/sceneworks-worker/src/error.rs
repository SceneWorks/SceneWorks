//! Worker error type ([`WorkerError`]), its `From` conversions, and the [`WorkerResult`] alias.
use super::*;

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
    ExternalLibraryUnavailable(String),
    Engine(String),
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
            Self::ExternalLibraryUnavailable(detail) => write!(
                formatter,
                "[{}] {detail}",
                sceneworks_core::model_artifacts::external_library::EXTERNAL_LIBRARY_UNAVAILABLE_CODE
            ),
            Self::Engine(detail) => formatter.write_str(detail),
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
/// message text INTACT, rather than burying it in an opaque [`WorkerError::Engine`]. Everything else
/// stays [`WorkerError::Engine`]. `context` prefixes the message so the origin (load vs generate)
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
