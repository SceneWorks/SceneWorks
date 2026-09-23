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
//! # Where each half lands
//!
//! The toggle is resolved against the engine's own descriptor
//! ([`engine_advertises_alpha_output`]), validated and refused by name when the engine cannot serve
//! it, round-tripped through the recipe ([`record_requested_channels`]), and assigned onto the
//! engine request as `gen_core::OutputChannels::Rgba` ([`request_output_channels`]). The generic
//! image lane then receives `GenerationOutput::ImagesRgba`, and the egress path types those four
//! channels as an RGBA PNG. An alpha-carrying reference travels as `Conditioning::ReferenceRgba`
//! (see `image_jobs::build_qwen_image_2_1_conditioning`).

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
// Read only by `reference_conditioning_kind`, whose callers are backend lanes.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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

/// Does the engine a SceneWorks model id resolves to advertise alpha output?
///
/// Read from the REGISTERED descriptor, never from a hand-list: a model id that resolves to no
/// linked engine (an unregistered id, or a build with no engine backend at all) cannot emit alpha,
/// so it answers `false` and a transparency request against it is refused by name.
pub(crate) fn engine_advertises_alpha_output(model_id: &str) -> bool {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        crate::engines::mlx_model(model_id)
            .is_some_and(|model| model.descriptor.capabilities.supports_alpha_output)
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = model_id;
        false
    }
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

/// The `gen_core::OutputChannels` a resolved channel count assigns onto the engine request.
///
/// [`CHANNELS_RGB`] is `OutputChannels::Rgb`, the contract's `Default`, so an opaque render puts
/// nothing new on the wire and every existing lane stays byte-identical.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn request_output_channels(channels: u8) -> gen_core::OutputChannels {
    if channels == CHANNELS_RGBA {
        gen_core::OutputChannels::Rgba
    } else {
        gen_core::OutputChannels::Rgb
    }
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
// `image_jobs::build_qwen_image_2_1_conditioning` classifies every resolved reference through this
// function, so the two halves of the epic cannot drift on what an alpha-carrying reference becomes.
#[cfg(any(
    test,
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

    // Read from the REGISTERED descriptor, so it needs a linked engine backend.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
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

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn an_opaque_render_puts_nothing_new_on_the_wire() {
        // `Rgb` is the contract's Default, so the request an opaque render builds is unchanged.
        assert_eq!(
            request_output_channels(CHANNELS_RGB),
            gen_core::OutputChannels::default()
        );
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn a_transparent_render_carries_the_rgba_variant_not_a_channel_count() {
        // `output_channels` is an ENUM (`OutputChannels { Rgb, Rgba }`), not a number.
        assert_eq!(
            request_output_channels(CHANNELS_RGBA),
            gen_core::OutputChannels::Rgba
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

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
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

    /// The capability is the DESCRIPTOR's answer on every registered engine, not a hand-list:
    /// exactly the engines whose descriptor sets `supports_alpha_output` resolve to four channels,
    /// and `qwen_image_2_1` is among them at the pinned inference revision.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn alpha_capability_is_read_from_every_registered_descriptor() {
        let mut capable = Vec::new();
        for row in crate::engines::MODEL_TABLE {
            let Some(model) = crate::engines::mlx_model(row.sceneworks_id) else {
                continue;
            };
            assert_eq!(
                engine_advertises_alpha_output(row.sceneworks_id),
                model.descriptor.capabilities.supports_alpha_output,
                "{}",
                row.sceneworks_id
            );
            if model.descriptor.capabilities.supports_alpha_output {
                capable.push(row.sceneworks_id);
            }
        }
        assert!(
            capable.contains(&"qwen_image_2_1"),
            "the pinned 2.1 provider advertises alpha output: {capable:?}"
        );
        assert!(!engine_advertises_alpha_output("no_such_model_xyz"));

        // The catalog's `supportsAlphaOutput` (what the web reads to offer the toggle) is a
        // hand-written MIRROR of this descriptor bit, so it is held to it here: a model whose
        // manifest offers the toggle must resolve to an engine that can serve it, and vice versa.
        for entry in crate::tests::builtin_models_manifest() {
            let Some(id) = entry.get("id").and_then(Value::as_str) else {
                continue;
            };
            if crate::engines::mlx_model(id).is_none() {
                continue;
            }
            let declared = entry.get("supportsAlphaOutput").and_then(Value::as_bool) == Some(true);
            assert_eq!(
                declared,
                engine_advertises_alpha_output(id),
                "{id}: manifest supportsAlphaOutput disagrees with the descriptor's \
                 supports_alpha_output"
            );
        }
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
