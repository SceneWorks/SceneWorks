use super::*;
use serde_json::json;

fn request_with(manifest_entry: Value, advanced: Value) -> ImageRequest {
    let payload = json!({
        "projectId": "p",
        "model": "krea_2_turbo",
        "modelManifestEntry": manifest_entry,
        "advanced": advanced,
    });
    ImageRequest::from_payload(payload.as_object().expect("payload is an object"))
}

#[test]
fn gain_resolves_through_ui_nesting_not_top_level() {
    // Correct nesting → the user value flows through, clamped to the GPU-validated band.
    let declared = request_with(
        json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0, "min": 0.25, "max": 1.75 } } }),
        json!({ "textStyleGain": 1.75 }),
    );
    assert_eq!(resolve_text_style_gain(&declared), Some(1.75));

    // Slider declared but user left it at default (web omits the key) → Some(1.0), a byte-exact
    // engine no-op — NOT None, so the field is still wired through.
    let defaulted = request_with(
        json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0 } } }),
        json!({}),
    );
    assert_eq!(resolve_text_style_gain(&defaulted), Some(1.0));

    // Out-of-range user value is clamped to [0.25, 1.75].
    let hot = request_with(
        json!({ "id": "krea_2_turbo", "ui": { "textStyleGain": { "default": 1.0 } } }),
        json!({ "textStyleGain": 9.0 }),
    );
    assert_eq!(resolve_text_style_gain(&hot), Some(1.75));

    // MUTATION CHECK (the sc-12008 bug): the slider object at the manifest TOP LEVEL with no `ui`
    // block must NOT resolve — guards against regressing to `.get("textStyleGain")`.
    let top_level_only = request_with(
        json!({ "id": "krea_2_turbo", "textStyleGain": { "default": 1.0 } }),
        json!({ "textStyleGain": 1.75 }),
    );
    assert_eq!(resolve_text_style_gain(&top_level_only), None);

    // A model that doesn't declare the slider (no `ui.textStyleGain`) self-gates to None even when
    // the client sends an `advanced.textStyleGain` — the manifest is the gate.
    let undeclared = request_with(
        json!({ "id": "sana_1600m", "ui": { "img2img": true } }),
        json!({ "textStyleGain": 1.75 }),
    );
    assert_eq!(resolve_text_style_gain(&undeclared), None);
}
