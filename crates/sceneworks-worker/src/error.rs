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
fn geometry_refusal_event(
    context: &str,
    reason: &str,
    requested_width: u32,
    requested_height: u32,
    alternative: Option<(u32, u32)>,
) -> Value {
    json!({
        "event": "generation_geometry_refused",
        "context": context,
        "refusalReason": reason,
        "requestedGeometry": {
            "width": requested_width,
            "height": requested_height,
        },
        "verifiedAlternative": alternative
            .map(|(width, height)| json!({ "width": width, "height": height })),
    })
}

pub(crate) fn classify_engine_error(context: &str, error: gen_core::Error) -> WorkerError {
    match error {
        gen_core::Error::Unsupported(_) => {
            WorkerError::InvalidPayload(format!("{context}: {error}"))
        }
        gen_core::Error::GeometryRefused {
            reason,
            requested_width,
            requested_height,
            alternative,
        } => {
            emit_event_value(
                Level::WARN,
                geometry_refusal_event(
                    context,
                    &reason,
                    requested_width,
                    requested_height,
                    alternative,
                ),
            );
            let verified_alternative = alternative.map_or_else(
                || "none available".to_owned(),
                |(width, height)| format!("{width}x{height}"),
            );
            WorkerError::InvalidPayload(format!(
                "{context}: geometry refused: {reason}; requested \
                 {requested_width}x{requested_height}; verified alternative: \
                 {verified_alternative}"
            ))
        }
        other => WorkerError::Engine(format!("{context}: {other}")),
    }
}

pub type WorkerResult<T> = Result<T, WorkerError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::{Event, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::Layer;

    #[derive(Default)]
    struct PayloadVisitor {
        payload: Option<Value>,
    }

    impl Visit for PayloadVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            if field.name() == "sw_payload" {
                self.payload = serde_json::from_str(&format!("{value:?}")).ok();
            }
        }
    }

    #[derive(Clone, Default)]
    struct EventCapture(Arc<Mutex<Vec<(Level, Value)>>>);

    impl<S: Subscriber> Layer<S> for EventCapture {
        fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
            let mut visitor = PayloadVisitor::default();
            event.record(&mut visitor);
            if let Some(payload) = visitor.payload {
                self.0
                    .lock()
                    .expect("event capture lock")
                    .push((*event.metadata().level(), payload));
            }
        }
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
    fn geometry_refusal_emits_current_none_alternative_and_preserves_user_fields() {
        let captured = EventCapture::default();
        let observed = captured.0.clone();
        let subscriber = tracing_subscriber::registry().with(captured);
        let classified = tracing::subscriber::with_default(subscriber, || {
            classify_engine_error(
                "Krea control generation failed",
                gen_core::Error::GeometryRefused {
                    reason: "measured infeasible".to_owned(),
                    requested_width: 1536,
                    requested_height: 1024,
                    alternative: None,
                },
            )
        });

        let events = observed.lock().expect("event capture lock");
        assert_eq!(events.len(), 1, "the refusal must emit exactly one event");
        let (level, event) = &events[0];
        assert_eq!(*level, Level::WARN);
        assert_eq!(event["event"], "generation_geometry_refused");
        assert_eq!(event["refusalReason"], "measured infeasible");
        assert_eq!(event["requestedGeometry"]["width"], 1536);
        assert_eq!(event["requestedGeometry"]["height"], 1024);
        assert_eq!(event["verifiedAlternative"], Value::Null);

        let WorkerError::InvalidPayload(message) = classified else {
            panic!("typed geometry refusal must be user-facing: {classified:?}");
        };
        assert!(message.contains("requested 1536x1024"));
        assert!(message.contains("verified alternative: none available"));
    }
}
