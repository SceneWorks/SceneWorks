//! Live-preview support projected from the build-validated Feature export in ModelSpec.
//!
//! `gen_core::Capabilities::supports_preview` (sc-16677) is the only weights-free way for a
//! consumer to tell "this route cannot live-preview" from "it can, but no frame has arrived yet".
//! Nothing in SceneWorks read it before this story, so `WorkerProgressCard` rendered both states
//! identically and a user waiting through model load + text encode had no way to know whether to
//! keep waiting.
//!
//! # Why the Feature export is built rather than queried at request time
//!
//! The process that serves `GET /api/v1/models` is not always the process that links an engine.
//! `docker/rust.Dockerfile:126` builds `sceneworks-rust-api --features embed-web` with **no**
//! `backend-candle`, while `:233` builds the worker with it — separate images. A serve-time
//! `registry.generators()` lookup would answer truthfully on desktop and report "nothing supports
//! preview" on every server: a *wrong* answer, not a missing one, in the topology least exercised
//! locally. And this crate, which owns the manifests, has no engine dependency at all and could
//! never derive it.
//!
//! The engine-linked lane dumps stage-1 facts, and the stage-2 preview export is joined into
//! ModelSpec at build time. Historical job aliases without catalog rows are compiled separately.
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
use serde_json::Value;

/// The generated stage-2 input is embedded only in tests. Production reads the compiled Feature
/// export, never a parallel runtime fallback table.
#[cfg(test)]
pub const PREVIEW_SUPPORT_MANIFEST: &str =
    include_str!("../../../config/manifests/builtin.preview-support.jsonc");

/// The catalog key rust-api stamps onto each model entry.
pub const PREVIEW_FIELD: &str = "preview";
/// The per-backend map inside it.
pub const BY_BACKEND_FIELD: &str = "byBackend";

#[cfg(test)]
fn parse(
    contents: &str,
) -> Result<std::collections::BTreeMap<String, std::collections::BTreeMap<String, bool>>, String> {
    let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(contents))
        .map_err(|error| format!("builtin.preview-support.jsonc is malformed: {error}"))?;
    let models = manifest
        .get("models")
        .and_then(Value::as_object)
        .ok_or_else(|| "builtin.preview-support.jsonc has no models object".to_owned())?;

    let mut table = std::collections::BTreeMap::new();
    for (model_id, by_backend) in models {
        let entries = by_backend.as_object().ok_or_else(|| {
            format!("builtin.preview-support.jsonc: {model_id} must map backends to booleans")
        })?;
        let mut parsed = std::collections::BTreeMap::new();
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

/// Whether `model_id` advertises live denoise preview on `backend`.
///
/// `None` = **unknown**: either the model has no engine on that backend, or that backend's stage-1
/// facts file has not been dumped. Never conflate it with `Some(false)` ("this route is wired and
/// does not preview") — the whole point of the story is that those are different states.
pub fn supports_preview(model_id: &str, backend: &str) -> Option<bool> {
    crate::registry::preview_feature(model_id)?
        .get(backend)
        .copied()
}

/// Stamp `preview: { byBackend: { … } }` onto one catalog entry from the Feature export.
///
/// A model absent from the export gets no `preview` key, preserving the unknown branch.
pub fn apply_to_model_entry(object: &mut serde_json::Map<String, Value>) {
    let Some(model_id) = object.get("id").and_then(Value::as_str) else {
        return;
    };
    let Some(by_backend) = crate::registry::preview_feature(model_id) else {
        return;
    };
    object.insert(
        PREVIEW_FIELD.to_owned(),
        serde_json::json!({ BY_BACKEND_FIELD: by_backend }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_manifest_parses_and_is_not_empty() {
        let table =
            parse(PREVIEW_SUPPORT_MANIFEST).expect("the generated preview-support manifest parses");
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

    // sc-17593. The audio registry is a SEPARATE `ProviderRegistry` that nothing dumped until that
    // story, so every audio route answered `None` — unknown — on every platform, which the card
    // renders identically to "wired and does not preview". These are the ids the audio jobs route
    // on (`audio_jobs.rs` hands the payload's `model` straight to `inference_runtime::load_audio`,
    // which is why the audio join needs no MODEL_TABLE row).
    #[test]
    fn answers_for_audio_routes_instead_of_leaving_them_unknown() {
        for model in [
            "kokoro_82m",
            "chatterbox_tts",
            "acestep_v15_turbo",
            "moss_sfx_v2",
            "moss_tts_realtime",
            "moss_ttsd_v05",
            // Both MMAudio routes are typed `"utility"` in builtin.models.jsonc but are ordinary
            // audio Generators in the registry, so they get a real answer like any other. Named
            // here because the manifest `type` is NOT what decides: the audio join is the identity
            // over the registry, so a served entry answers whatever its type field says.
            "mmaudio_small_16k",
            "mmaudio_large_44k",
        ] {
            assert_eq!(
                supports_preview(model, "candle"),
                Some(false),
                "{model} must answer Some(false) — no audio generator emits PreviewSink frames at \
                 the current pin, and `None` here would be the pre-sc-17593 hole rather than a fact"
            );
        }
        // Only GENERATORS carry `Capabilities`, so the audio lane's transform/embedder providers
        // have no `supports_preview` to dump and must stay unknown rather than be given a false.
        assert_eq!(supports_preview("openvoice_v2", "candle"), None);
        assert_eq!(supports_preview("chatterbox_ve", "candle"), None);
    }

    // The distinction the whole story exists for: a wired route that does NOT preview answers
    // `Some(false)`, and an unknown backend/model answers `None`. Collapsing either into the other
    // reintroduces the bug.
    #[test]
    fn distinguishes_wired_false_from_unknown() {
        // Bernini Image is registered in the current Candle facts but does not advertise preview.
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
