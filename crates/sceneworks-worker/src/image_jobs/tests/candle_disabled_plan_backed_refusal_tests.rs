use super::prepare_candle_image_route;
use super::super::ImageRequest;
use crate::Settings;
use crate::WorkerError;
use serde_json::json;

fn request(entry: serde_json::Value) -> ImageRequest {
    ImageRequest::from_payload(
        json!({
            "projectId": "project-1",
            "model": "imported_krea_abc123",
            "mode": "text_to_image",
            "prompt": "a fox",
            "modelManifestEntry": entry
        })
        .as_object()
        .expect("payload object"),
    )
}

fn candle_disabled_settings() -> Settings {
    Settings {
        backend_candle_enabled: false,
        ..crate::test_env::offline_settings()
    }
}

#[test]
fn a_plan_backed_image_request_refuses_instead_of_stubbing_when_candle_is_disabled() {
    let settings = candle_disabled_settings();
    let routed = prepare_candle_image_route(
        &request(json!({
            "id": "imported_krea_abc123",
            "family": "krea_2",
            "importPlan": { "checkpointId": "ckpt_krea_abc123" }
        })),
        &settings,
    );
    // `PreparedCandleImageRoute` is not `Debug`, so match rather than `expect_err`.
    let error = match routed {
        Err(error) => error,
        Ok(_) => panic!("a plan-backed entry must never fall through to the stub generator"),
    };
    match error {
        WorkerError::InvalidPayload(message) => assert!(
            message.contains("[checkpoint-plan:no-adapter-binding]")
                && message.contains("\"ckpt_krea_abc123\""),
            "the refusal must name the checkpoint: {message}"
        ),
        other => panic!("expected an InvalidPayload refusal, got {other:?}"),
    }

    // Control: the identical entry WITHOUT a plan still takes the early-out unchanged.
    let control = prepare_candle_image_route(
        &request(json!({ "id": "krea_2", "family": "krea_2" })),
        &settings,
    );
    assert!(
        matches!(control, Ok(None)),
        "the refusal must be keyed on the plan, not on the disabled backend"
    );
}
