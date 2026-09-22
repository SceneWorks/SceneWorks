//! Transparency (RGBA output) request adapter — sc-24113, epic 24107.
//!
//! # Why this is its own module
//!
//! Qwen-Image 2.1's VAE can decode straight to four channels
//! (`QwenImage21Vae::decode_rgba` → NCHW `[1, 4, H, W]`), and sc-24111 already widened the whole
//! SceneWorks EGRESS path to carry that: [`crate::image_jobs::GeneratedPixels::from_engine_buffer`]
//! types a buffer by its channel count, `workflow_png` writes RGBA, the upscale/detail passes route
//! the alpha plane around the 3-channel models, and the Image Editor composites and exports without
//! flattening onto black.
//!
//! What was missing is the INGRESS half: a way for the user to ask for it. That is this module, and
//! it is deliberately the ONLY place in the worker that spells the names the S4 RGBA contract
//! (inference sc-24111, PR #1009) turns on:
//!
//! * [`DESCRIPTOR_ALPHA_CAPABILITY`] — `Capabilities::supports_alpha_output`, false by default and
//!   true only on `qwen_image_2_1`, on both backends.
//! * [`REQUEST_OUTPUT_CHANNELS`] — `GenerationRequest::output_channels`, a
//!   `gen_core::OutputChannels { Rgb (default), Rgba }`. It is an ENUM, not a channel count: the
//!   default is byte-identical to every pre-S4 render.
//! * [`CONDITIONING_REFERENCE_RGBA`] — `Conditioning::ReferenceRgba`, for a reference that CARRIES
//!   alpha. See *Alpha-carrying references* below.
//!
//! # Transparency is PROMPT-DRIVEN; the toggle only opens the surface
//!
//! The single most important thing this module encodes, because it is the thing a reader will
//! assume wrongly: **there is no transparency flag or mode upstream.** The 2.1 VAE is ALWAYS
//! four-channel. `output_channels: Rgba` does not ask the model for a transparent background — it
//! asks for the fourth channel to survive the decode instead of being composited over white.
//! Whether that channel actually contains a cut-out is decided by the PROMPT, using the model
//! card's own convention ("the background is transparent" / "This is an RGBA image with
//! transparency").
//!
//! So the toggle alone can hand a user a fully opaque RGBA PNG and look broken. It is paired with
//! [`TRANSPARENCY_PROMPT_HINT`], which the UI offers as visible, editable wording the user adds to
//! their prompt — never appended silently, because a prompt the user cannot see is a prompt they
//! cannot fix.
//!
//! # Alpha-carrying references
//!
//! A reference that already has an alpha channel must travel as `Conditioning::ReferenceRgba`, not
//! as a flattened RGB `Reference`: the VAE encodes all four channels (the vision tower separately
//! gets it composited over white), so **a flattened reference is a DIFFERENT request**. That is
//! what makes transparent-layer editing work at all — the Image Editor's own cut-out output can go
//! straight back in as a reference without losing its alpha. Alpha is STRAIGHT (un-premultiplied)
//! and `A=0` does NOT zero RGB, so a consumer that must flatten has to COMPOSITE, never drop the
//! fourth byte.
//!
//! # What is real today and what is not
//!
//! Real today: the user-facing toggle, the prompt-convention pairing, validation, the refusal when
//! the engine cannot serve it, the round-trip through the job payload, and the whole egress path
//! that turns four channels into a transparent PNG.
//!
//! Not real until the epic's terminal pin bump: the request field reaching the provider. This
//! branch pins an inference revision that predates S4, so the pinned `Capabilities` has no
//! `supports_alpha_output` member and the pinned `GenerationRequest` no `output_channels` field.
//! [`engine_advertises_alpha_output`] therefore answers from a pinned id list rather than by
//! reading the descriptor; that function's BODY is the one-line swap, and its callers, its tests
//! and its refusal text are already written against the final meaning.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::{WorkerError, WorkerResult};

/// `Capabilities::supports_alpha_output` — the engine flag that says this model can emit alpha.
pub(crate) const DESCRIPTOR_ALPHA_CAPABILITY: &str = "supports_alpha_output";

/// `GenerationRequest::output_channels` — the request field carrying the output surface.
pub(crate) const REQUEST_OUTPUT_CHANNELS: &str = "output_channels";

/// `gen_core::OutputChannels::Rgb` — the default, byte-identical to every pre-S4 render.
pub(crate) const OUTPUT_CHANNELS_RGB: &str = "rgb";
/// `gen_core::OutputChannels::Rgba` — the fourth channel survives the decode.
pub(crate) const OUTPUT_CHANNELS_RGBA: &str = "rgba";

/// `Conditioning::ReferenceRgba` — the carrier for a reference that has its own alpha channel.
///
/// Sending such a reference flattened to RGB is a DIFFERENT request, not a lossy version of the
/// same one, because the VAE encodes all four channels.
pub(crate) const CONDITIONING_REFERENCE_RGBA: &str = "ReferenceRgba";

/// The `advanced` key the Studio/Editor set when the user turns the transparency toggle on.
///
/// A SceneWorks request axis, not an engine field: the engine has no transparency concept at all
/// (see the module docs). This is the app's record of what the USER asked for, which is what the
/// recipe replays.
pub(crate) const ADVANCED_TRANSPARENT_BACKGROUND: &str = "transparentBackground";

/// The model card's own wording for asking 2.1 for a transparent background.
///
/// Recorded here as part of the contract; the COMPOSITION lives in `apps/web/src/qwenAlpha.js`
/// because the prompt is assembled client-side, and `the_s4_contract_names_are_the_ones_the_web_half_mirrors`
/// pins the two copies together. Offered to the user as editable text beside the toggle, never
/// appended silently: transparency is prompt-driven (module docs), so a toggle with no wording
/// behind it produces an opaque RGBA image and looks like a bug, and a prompt the user cannot see
/// is one they cannot fix.
#[allow(dead_code)]
pub(crate) const TRANSPARENCY_PROMPT_HINT: &str =
    "The background is transparent. This is an RGBA image with transparency.";

/// Opaque RGB — three channels. Every lane before sc-24113, byte-identical.
pub(crate) const CHANNELS_RGB: u8 = 3;
/// Native transparency — four channels, straight through to an RGBA PNG.
pub(crate) const CHANNELS_RGBA: u8 = 4;

/// SceneWorks model ids known to decode to four channels at THIS pin.
///
/// Keyed on the SceneWorks catalog id rather than the engine id because that is what the request
/// carries at the funnel this is checked from. For `qwen_image_2_1` the two are the same string.
///
/// This list is the stand-in for a descriptor read, and it exists for exactly one reason: the
/// inference revision this branch pins predates `mlx-gen-qwen-image-2-1`, so there is no
/// `Capabilities::supports_alpha_output` to read and no provider to read it from. Writing a
/// hand-list is honest about that; silently defaulting to "everything supports it" would not be.
///
/// At the epic's terminal pin bump [`engine_advertises_alpha_output`] becomes the descriptor read
/// and this constant goes away. Until then a model added here without an RGBA VAE would produce a
/// three-channel buffer and the egress path would simply type it `Rgb` — a wrong toggle, not a
/// crash.
const PINNED_ALPHA_CAPABLE_ENGINES: &[&str] = &["qwen_image_2_1"];

/// Does this engine advertise alpha output?
///
/// ⚠️ **THE ONE-LINE SWAP.** At the terminal pin bump the body becomes
/// `descriptor.capabilities.supports_alpha_output` and the signature takes the resolved descriptor
/// instead of the id. Every caller, test and message below is already written against that
/// meaning.
pub(crate) fn engine_advertises_alpha_output(model_id: &str) -> bool {
    PINNED_ALPHA_CAPABLE_ENGINES.contains(&model_id)
}

/// Did the user ask for a transparent background?
///
/// Reads the SceneWorks request axis, not the engine field. Absent, null, `false` and any
/// non-boolean all mean "no": a toggle that arrives malformed must not silently change what the
/// render emits.
pub(crate) fn transparency_requested(advanced: &Map<String, Value>) -> bool {
    advanced
        .get(ADVANCED_TRANSPARENT_BACKGROUND)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Resolve the output-channel count for a render, or refuse by name.
///
/// * toggle off → [`CHANNELS_RGB`], for every engine, byte-identical to every lane before this.
/// * toggle on + engine advertises alpha → [`CHANNELS_RGBA`].
/// * toggle on + engine does NOT advertise alpha → a typed refusal naming the engine.
///
/// The third case is a refusal rather than a silent downgrade on purpose. A user who asked for a
/// cut-out and got an opaque render on a white background has to discover that by looking; a job
/// that fails saying the model cannot do it is actionable. This mirrors the steps-floor decision in
/// `create_image_job` (reject, never clamp).
pub(crate) fn resolve_output_channels(
    model_id: &str,
    advanced: &Map<String, Value>,
) -> WorkerResult<u8> {
    if !transparency_requested(advanced) {
        return Ok(CHANNELS_RGB);
    }
    if engine_advertises_alpha_output(model_id) {
        return Ok(CHANNELS_RGBA);
    }
    Err(WorkerError::InvalidPayload(format!(
        "{model_id} cannot render a transparent background — it emits opaque RGB only. Turn \
         transparency off, or pick a model that advertises `{DESCRIPTOR_ALPHA_CAPABILITY}`."
    )))
}

/// Stamp the resolved output surface onto a render's `rawSettings`, so the request round-trips.
///
/// [`CHANNELS_RGB`] stamps NOTHING. Every render before sc-24113 is opaque, and writing
/// `transparentBackground: false` onto all of them would churn every stored recipe and every
/// workflow PNG in the app for a field that means "the default". Only a transparency request
/// leaves a mark, which is also what makes the recipe replay it.
pub(crate) fn record_requested_channels(
    raw_settings: &mut serde_json::Map<String, Value>,
    channels: u8,
) {
    if channels == CHANNELS_RGB {
        return;
    }
    raw_settings.insert(
        ADVANCED_TRANSPARENT_BACKGROUND.to_owned(),
        Value::from(true),
    );
    raw_settings.insert(
        REQUEST_OUTPUT_CHANNELS.to_owned(),
        Value::from(output_channels_variant(channels)),
    );
}

/// The `gen_core::OutputChannels` variant a resolved channel count names.
///
/// An ENUM, not the count. The count is how SceneWorks reasons about the buffer it gets BACK
/// (`GeneratedPixels` types by channels); the request says `Rgb` or `Rgba`. Keeping both and
/// converting here is what stops the two meanings being confused at a call site.
pub(crate) fn output_channels_variant(channels: u8) -> &'static str {
    if channels == CHANNELS_RGBA {
        OUTPUT_CHANNELS_RGBA
    } else {
        OUTPUT_CHANNELS_RGB
    }
}

/// The engine-request fragment for a resolved output surface, as a name→value map.
///
/// Returns an EMPTY map for [`CHANNELS_RGB`] rather than `{output_channels: "rgb"}`: `Rgb` IS the
/// contract's default, so an opaque render must put nothing new on the wire and every existing
/// lane stays byte-identical.
///
/// Returned as a map rather than set directly on `GenerationRequest` because the pinned request
/// struct has no `output_channels` field yet (see the module docs). At the pin bump the call site
/// assigns `gen_core::OutputChannels::Rgba` instead of merging this — the DECISION this computes
/// does not change either way.
// Unused OUTSIDE tests at this pin, and deliberately kept: it is the computed value the pin bump
// assigns to the engine request, and the one place the field name meets its variant.
#[allow(dead_code)]
pub(crate) fn output_channels_request_fragment(channels: u8) -> BTreeMap<&'static str, Value> {
    let mut fragment = BTreeMap::new();
    if channels != CHANNELS_RGB {
        fragment.insert(
            REQUEST_OUTPUT_CHANNELS,
            Value::from(output_channels_variant(channels)),
        );
    }
    fragment
}

/// Which conditioning carrier a reference image belongs in.
///
/// A reference that CARRIES alpha must travel as `Conditioning::ReferenceRgba`; one without alpha
/// is an ordinary `Reference`. This is not an optimisation: the VAE encodes all four channels, so
/// flattening an alpha-carrying reference to RGB produces a DIFFERENT request, and it is exactly
/// the case transparent-layer editing depends on — the editor's own cut-out going straight back in
/// as a reference.
///
/// `has_alpha` is the DECODED image's own answer (`image::DynamicColor::has_alpha`, which is what
/// `image_jobs::split_alpha` already reads), not a guess from the file extension: a PNG with no
/// alpha channel is an ordinary reference, and an opaque alpha channel is still alpha as far as the
/// carrier is concerned — `A=255` is byte-identical through the RGBA path, so classifying by
/// CHANNELS rather than by content keeps the choice total and cheap.
// sc-24110 gave this its call site: `image_jobs::build_qwen_image_2_1_conditioning` classifies
// every resolved reference through this function, so the two halves of the story cannot drift on
// what an alpha-carrying reference becomes. The `ReferenceRgba` ARM still cannot be CONSTRUCTED —
// the variant is not in the pinned `gen_core` — so that is what the builder refuses on, by name,
// rather than flattening; see `qwen_image_2_1_rgba_reference_is_pending_the_pin`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn reference_conditioning_kind(has_alpha: bool) -> &'static str {
    if has_alpha {
        CONDITIONING_REFERENCE_RGBA
    } else {
        "Reference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn advanced(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn an_absent_or_false_toggle_asks_for_three_channels_on_every_engine() {
        for payload in [json!({}), json!({ "transparentBackground": false })] {
            let advanced = advanced(payload);
            assert!(!transparency_requested(&advanced));
            for engine in ["qwen_image_2_1", "flux_dev", "z_image"] {
                assert_eq!(
                    resolve_output_channels(engine, &advanced).unwrap(),
                    CHANNELS_RGB,
                    "{engine}"
                );
            }
        }
    }

    #[test]
    fn a_malformed_toggle_is_read_as_off_rather_than_as_on() {
        // A string, a number and a null all mean "no". The alternative — treating any present key
        // as truthy — would let a stale or mistyped client silently change what the render emits.
        for payload in [
            json!({ "transparentBackground": "true" }),
            json!({ "transparentBackground": 1 }),
            json!({ "transparentBackground": Value::Null }),
        ] {
            let advanced = advanced(payload);
            assert!(!transparency_requested(&advanced));
            assert_eq!(
                resolve_output_channels("qwen_image_2_1", &advanced).unwrap(),
                CHANNELS_RGB
            );
        }
    }

    #[test]
    fn transparency_on_an_alpha_capable_engine_resolves_to_four_channels() {
        let advanced = advanced(json!({ "transparentBackground": true }));
        assert!(engine_advertises_alpha_output("qwen_image_2_1"));
        assert_eq!(
            resolve_output_channels("qwen_image_2_1", &advanced).unwrap(),
            CHANNELS_RGBA
        );
    }

    #[test]
    fn transparency_on_an_opaque_engine_is_refused_by_name_not_silently_dropped() {
        let advanced = advanced(json!({ "transparentBackground": true }));
        let error = resolve_output_channels("flux_dev", &advanced).unwrap_err();
        let message = error.to_string();
        // The refusal must name the engine that cannot serve it AND the capability flag, so the
        // message stays actionable after the flag is renamed at the pin bump.
        assert!(message.contains("flux_dev"), "{message}");
        assert!(message.contains(DESCRIPTOR_ALPHA_CAPABILITY), "{message}");
    }

    #[test]
    fn an_opaque_render_puts_nothing_new_on_the_wire() {
        assert!(output_channels_request_fragment(CHANNELS_RGB).is_empty());
    }

    #[test]
    fn a_transparent_render_carries_the_rgba_variant_not_a_channel_count() {
        // `output_channels` is an ENUM (`OutputChannels { Rgb, Rgba }`), not a number. Sending 4
        // would not deserialize; sending "rgb" would be a no-op field the contract already defaults.
        let fragment = output_channels_request_fragment(CHANNELS_RGBA);
        assert_eq!(fragment.len(), 1);
        assert_eq!(
            fragment.get(REQUEST_OUTPUT_CHANNELS),
            Some(&Value::from("rgba"))
        );
        assert_eq!(output_channels_variant(CHANNELS_RGBA), OUTPUT_CHANNELS_RGBA);
        assert_eq!(output_channels_variant(CHANNELS_RGB), OUTPUT_CHANNELS_RGB);
    }

    #[test]
    fn an_alpha_carrying_reference_takes_the_rgba_carrier() {
        // The VAE encodes all four channels, so a flattened alpha reference is a DIFFERENT request
        // — not a lossy version of the same one. This is what transparent-layer editing rests on.
        assert_eq!(
            reference_conditioning_kind(true),
            CONDITIONING_REFERENCE_RGBA
        );
        assert_eq!(reference_conditioning_kind(false), "Reference");
    }

    #[test]
    fn an_opaque_render_leaves_no_mark_on_its_recipe() {
        // Every render in the app before sc-24113 is opaque. Stamping `transparentBackground: false`
        // onto all of them would churn every stored recipe and workflow PNG for a default.
        let mut raw_settings = Map::new();
        record_requested_channels(&mut raw_settings, CHANNELS_RGB);
        assert!(raw_settings.is_empty());
    }

    #[test]
    fn a_transparency_request_round_trips_through_the_recipe() {
        // This is what makes a re-run reproduce a cut-out instead of quietly re-rendering it
        // opaque: the recipe carries what was ASKED FOR, under both names.
        let mut raw_settings = Map::new();
        record_requested_channels(&mut raw_settings, CHANNELS_RGBA);
        assert_eq!(
            raw_settings.get(ADVANCED_TRANSPARENT_BACKGROUND),
            Some(&Value::from(true))
        );
        assert_eq!(
            raw_settings.get(REQUEST_OUTPUT_CHANNELS),
            Some(&Value::from("rgba"))
        );
        // ... and replaying that recipe re-resolves to four channels, closing the loop.
        let replayed = advanced(Value::Object(raw_settings));
        assert!(transparency_requested(&replayed));
        assert_eq!(
            resolve_output_channels("qwen_image_2_1", &replayed).unwrap(),
            CHANNELS_RGBA
        );
    }

    #[test]
    fn the_s4_contract_names_are_the_ones_the_web_half_mirrors() {
        // The literals this module exists to centralize, transcribed from the S4 RGBA contract
        // (inference sc-24111, PR #1009). Asserted here so a rename fails LOUDLY in one place with
        // the mirror named, rather than drifting away from `apps/web/src/qwenAlpha.js`.
        assert_eq!(DESCRIPTOR_ALPHA_CAPABILITY, "supports_alpha_output");
        assert_eq!(REQUEST_OUTPUT_CHANNELS, "output_channels");
        assert_eq!(OUTPUT_CHANNELS_RGB, "rgb");
        assert_eq!(OUTPUT_CHANNELS_RGBA, "rgba");
        assert_eq!(CONDITIONING_REFERENCE_RGBA, "ReferenceRgba");
        // The wording the web half composes; the two copies must stay identical.
        assert_eq!(
            TRANSPARENCY_PROMPT_HINT,
            "The background is transparent. This is an RGBA image with transparency."
        );
        // OURS, and stable: the engine has no transparency concept, so this is the app's record of
        // what the user asked for.
        assert_eq!(ADVANCED_TRANSPARENT_BACKGROUND, "transparentBackground");
    }
}
