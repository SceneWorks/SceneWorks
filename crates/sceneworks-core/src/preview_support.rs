//! The generated live-preview support catalog (sc-16965, epic 16948) — the served half of the
//! two-stage engine-capability pipeline.
//!
//! `gen_core::Capabilities::supports_preview` (sc-16677) is the only weights-free way for a
//! consumer to tell "this route cannot live-preview" from "it can, but no frame has arrived yet".
//! Nothing in SceneWorks read it before this story, so `WorkerProgressCard` rendered both states
//! identically and a user waiting through model load + text encode had no way to know whether to
//! keep waiting.
//!
//! # Why this is a checked-in manifest and not a runtime lookup
//!
//! The process that serves `GET /api/v1/models` is not always the process that links an engine.
//! `docker/rust.Dockerfile:126` builds `sceneworks-rust-api --features embed-web` with **no**
//! `backend-candle`, while `:233` builds the worker with it — separate images. A serve-time
//! `registry.generators()` lookup would answer truthfully on desktop and report "nothing supports
//! preview" on every server: a *wrong* answer, not a missing one, in the topology least exercised
//! locally. And this crate, which owns the manifests, has no engine dependency at all and could
//! never derive it.
//!
//! So the flag is dumped at build time on a lane that links engines
//! (`sceneworks_worker::engine_capability_facts`, stage 1) and derived into
//! `config/manifests/builtin.preview-support.jsonc` by `apps/web/scripts/generate-preview-support.mjs`
//! (stage 2), whose vitest drift guard re-derives it on every PR. This module only embeds and reads
//! that artifact.
//!
//! # Engine-keyed, never a boolean
//!
//! `supports_streaming` is a property of the model, so the audio catalog ships one flag. Preview
//! support is a property of **the engine that runs it**, and it does not collapse when epic 16948
//! completes: SenseNova-U1 is candle-only and MLX never wired it, so at least one route stays split
//! permanently. Hence `preview.byBackend`: `{"candle": true}`, `{"mlx": true, "candle": false}`, …
//!
//! A backend absent from a model's map means **unknown**, never false — that backend either never
//! registered the engine, or its facts file has not been dumped on this checkout. Consumers render
//! unknown exactly as they did before this story.
//!
//! `include_str!` reaches `config/manifests/` outside this crate, which `docker/rust.Dockerfile:81`
//! `COPY config ./config`s wholesale — the same reason [`crate::builtin_manifests`] can do it.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::Value;

/// The generated catalog, embedded at compile time from the canonical repo copy. Regenerate with
/// `npm run gen:preview-support` (apps/web) — never hand-edit.
pub const PREVIEW_SUPPORT_MANIFEST: &str =
    include_str!("../../../config/manifests/builtin.preview-support.jsonc");

/// The catalog key rust-api stamps onto each model entry.
pub const PREVIEW_FIELD: &str = "preview";
/// The per-backend map inside it.
pub const BY_BACKEND_FIELD: &str = "byBackend";

/// `model id -> { backend -> supports_preview }`, parsed once.
type PreviewTable = BTreeMap<String, BTreeMap<String, bool>>;

static TABLE: OnceLock<Result<PreviewTable, String>> = OnceLock::new();

fn parse(contents: &str) -> Result<PreviewTable, String> {
    let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(contents))
        .map_err(|error| format!("builtin.preview-support.jsonc is malformed: {error}"))?;
    let models = manifest
        .get("models")
        .and_then(Value::as_object)
        .ok_or_else(|| "builtin.preview-support.jsonc has no models object".to_owned())?;

    let mut table = PreviewTable::new();
    for (model_id, by_backend) in models {
        let entries = by_backend.as_object().ok_or_else(|| {
            format!("builtin.preview-support.jsonc: {model_id} must map backends to booleans")
        })?;
        let mut parsed = BTreeMap::new();
        for (backend, supported) in entries {
            let supported = supported.as_bool().ok_or_else(|| {
                format!(
                    "builtin.preview-support.jsonc: {model_id}.{backend} must be a boolean, got \
                     {supported}"
                )
            })?;
            parsed.insert(backend.clone(), supported);
        }
        table.insert(model_id.clone(), parsed);
    }
    Ok(table)
}

fn table() -> Result<&'static PreviewTable, &'static str> {
    TABLE
        .get_or_init(|| parse(PREVIEW_SUPPORT_MANIFEST))
        .as_ref()
        .map_err(String::as_str)
}

/// Whether `model_id` advertises live denoise preview on `backend`.
///
/// `None` = **unknown**: either the model has no engine on that backend, or that backend's stage-1
/// facts file has not been dumped. Never conflate it with `Some(false)` ("this route is wired and
/// does not preview") — the whole point of the story is that those are different states.
pub fn supports_preview(model_id: &str, backend: &str) -> Option<bool> {
    table().ok()?.get(model_id)?.get(backend).copied()
}

/// Stamp `preview: { byBackend: { … } }` onto one catalog entry, if the generated table knows it.
///
/// Additive and total: a model absent from the table gets no `preview` key at all, so every
/// pre-existing consumer is untouched and the UI's "unknown" branch keeps rendering as before. Any
/// parse failure is swallowed here — the flag is decorative discoverability, and a malformed
/// generated artifact must never take down `/api/v1/models`. The vitest drift guard is what makes
/// the artifact correct; this is the serving seam, not a second validator.
pub fn apply_to_model_entry(object: &mut serde_json::Map<String, Value>) {
    let Ok(table) = table() else { return };
    let Some(model_id) = object.get("id").and_then(Value::as_str) else {
        return;
    };
    let Some(by_backend) = table.get(model_id) else {
        return;
    };
    let map = by_backend
        .iter()
        .map(|(backend, supported)| (backend.clone(), Value::Bool(*supported)))
        .collect::<serde_json::Map<String, Value>>();
    object.insert(
        PREVIEW_FIELD.to_owned(),
        serde_json::json!({ BY_BACKEND_FIELD: Value::Object(map) }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_manifest_parses_and_is_not_empty() {
        let table = table().expect("the generated preview-support manifest parses");
        // The stage-1 dumper refuses an empty registry and the stage-2 guard refuses an empty facts
        // file, so an empty table here would mean the embedded artifact was hand-emptied.
        assert!(
            table.len() > 40,
            "the generated table should cover the whole MODEL_TABLE, got {}",
            table.len()
        );
    }

    #[test]
    fn reads_the_current_wired_candle_routes_as_supporting() {
        // Representative truth at inference pin 5b6d6aa: Krea, Qwen-Image, and SDXL all advertise
        // live preview. Keep SDXL here because its provider gained preview after this contract's
        // original pre-5b6 fixture was written.
        assert_eq!(supports_preview("krea_2_turbo", "candle"), Some(true));
        assert_eq!(supports_preview("krea_2_raw", "candle"), Some(true));
        assert_eq!(supports_preview("qwen_image", "candle"), Some(true));
        assert_eq!(supports_preview("sdxl", "candle"), Some(true));
    }

    #[test]
    fn flux2_dev_now_advertises_its_wired_preview_route() {
        assert_eq!(supports_preview("flux2_dev", "candle"), Some(true));
    }

    // The distinction the whole story exists for: a wired route that does NOT preview answers
    // `Some(false)`, and an unknown backend/model answers `None`. Collapsing either into the other
    // reintroduces the bug.
    #[test]
    fn distinguishes_wired_false_from_unknown() {
        // Bernini is registered in the current Candle facts but does not advertise preview.
        // This is a declared false route, not an inference from a provider being absent.
        assert_eq!(supports_preview("bernini_image", "candle"), Some(false));
        // A backend with no facts file is UNKNOWN, not false. Asserted against a name that can
        // never be dumped rather than against `mlx`: `mlx` is exactly the backend the macOS lane is
        // expected to add, and pinning "mlx is absent" here would turn that follow-up into a red
        // test with a comment claiming the absence was intentional.
        assert_eq!(supports_preview("bernini_image", "no_such_backend"), None);
        assert_eq!(supports_preview("not_a_model", "candle"), None);
    }

    #[test]
    fn stamps_by_backend_onto_a_known_catalog_entry() {
        let mut entry = serde_json::json!({ "id": "krea_2_turbo", "type": "image" });
        apply_to_model_entry(entry.as_object_mut().unwrap());
        assert_eq!(entry["preview"]["byBackend"]["candle"], Value::Bool(true));
    }

    #[test]
    fn leaves_an_unknown_entry_completely_untouched() {
        let mut entry = serde_json::json!({ "id": "external_base_whatever", "type": "image" });
        let before = entry.clone();
        apply_to_model_entry(entry.as_object_mut().unwrap());
        assert_eq!(
            entry, before,
            "a model the table does not know must get NO preview key — absence is `unknown`, and \
             inventing `false` would let the UI claim a route cannot preview when it has no opinion"
        );
    }

    #[test]
    fn rejects_a_non_boolean_flag() {
        let error = parse(r#"{ "models": { "sdxl": { "candle": "yes" } } }"#)
            .expect_err("a non-boolean flag is malformed");
        assert!(error.contains("sdxl.candle"), "got: {error}");
    }
}
