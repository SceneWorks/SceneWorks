//! The A1111 `parameters` rendering of a sanitized envelope (sc-15957, epic 15945).
//!
//! [`crate::workflow_share`] owns the contract, [`crate::workflow_png`] owns the PNG plumbing, and
//! this module owns one narrow question: **what does a third-party gallery get to display?**
//!
//! # This is a display feature, not an interop feature
//!
//! Civitai and most image viewers parse the de-facto `parameters` text chunk that AUTOMATIC1111's
//! WebUI popularised. They *display* generation settings; they do not execute them. That is what
//! makes this worth doing for models those tools have never heard of — a Krea or FLUX.2 image posted
//! to a gallery shows its prompt, seed and size instead of arriving as opaque pixels.
//!
//! Judge it on that. Nothing here is a replay format: [`crate::workflow_share`]'s
//! `sceneworks:workflow` envelope is, and it is the one a SceneWorks install reads back.
//!
//! # Explicitly not ComfyUI's format
//!
//! ComfyUI embeds `workflow` / `prompt` chunks holding a serialized **node graph**. SceneWorks has
//! no node graph, and emitting a fake one would misrepresent the product. This is the A1111 text
//! convention and nothing else.
//!
//! # Written from the sanitized envelope, never from the raw payload
//!
//! [`parameters_text`] takes a [`WorkflowShare`] — the already-allow-listed, already-reduced,
//! already-path-guarded envelope. It is a second *rendering* of the same data, not a second, laxer
//! channel: a field the sc-15946 allow-list withholds cannot appear here, because this function
//! never sees it. `the_parameters_chunk_cannot_carry_a_withheld_field` in
//! `crates/sceneworks-core/tests/workflow_parameters.rs` seeds a withheld key into a raw payload and
//! proves it is absent from both chunks.
//!
//! The sc-15953 setting therefore governs both chunks together, for free: the writer is handed
//! `None` when embedding is off, and neither chunk is written.
//!
//! # Omit rather than approximate
//!
//! The discipline of the whole story. A guessed mapping ships misleading data to a *public gallery*,
//! which is worse than a shorter trailer, and nobody downstream can tell a guess from a fact. So
//! every A1111 field is either an exact restatement of something the envelope holds or is left out.
//! [`SETTINGS_FIELDS`] is the decision record, and the omissions are argued in
//! [`the omitted fields`](#what-is-deliberately-not-here) below.
//!
//! # What is deliberately not here
//!
//! | SceneWorks has | A1111 has | Why it is omitted |
//! | --- | --- | --- |
//! | `advanced.phases` (multi-phase Krea) | nothing | A schedule of N (steps, guidance) pairs has no single-number form. Its presence also suppresses `Steps` and `CFG scale`, because a top-level number beside a multi-phase schedule describes a run that did not happen. |
//! | `advanced.textStyleGain` | nothing | A Krea tap-reweight gain. No field means it, and inventing one names a knob no reader can interpret. |
//! | `advanced.scheduler`, `schedulerShift` | `Schedule type` | A1111's `Schedule type` is a NOISE schedule (Karras, Exponential). Ours is a diffusers scheduler class, which is closer to A1111's *sampler*. Two plausible mappings and no correct one. |
//! | `advanced.strength` | `Denoising strength` | The sense is lane-dependent — the candle fork lane inverts it (higher is *closer* to the reference). A number whose direction is ambiguous is worse than no number. |
//! | `upscale.*` | `Hires upscale` / `Hires upscaler` | A1111's hires fix is a specific latent second pass. Ours is a separate post-pass on the decoded image. Same words, different pipeline. |
//! | `loras[]` | `<lora:name:weight>` in the prompt | A1111 carries LoRAs by rewriting the prompt. Editing the user's prompt to smuggle a resource list in is fabrication — the prompt in this chunk is the prompt they typed. |
//! | `count` | `Batch size` | Ours is how many images the run asked for; A1111's is the sampler's parallel batch. Near-synonyms that are not the same number. |
//! | `advanced.controlMode` / `controlScale` | ControlNet extension fields | The ControlNet trailer is a whole sub-format with per-unit indices and model hashes. A partial emission of it reads as a full one. |
//! | `advanced.faceRestore` | `Face restoration: CodeFormer` | A1111's field names the *engine*. Ours is a boolean and we have no engine name to give. |
//! | `stylePreset`, `styleId`, `fitMode`, `mode`, `inputs[]` | nothing | No field, near or far. |
//! | quant tier, attention kernel, GPU | nothing | Never in the envelope at all — the allow-list withholds them, so this rendering could not see them if it wanted to. |
//!
//! # Encoding is decided by [`crate::workflow_png`], not here
//!
//! This module produces text. Whether that text rides in a `tEXt` or an `iTXt` chunk is a PNG
//! question and is answered where the chunk is framed; see `parameters_chunk` there.

use serde_json::Value;

use crate::workflow_share::WorkflowShare;

/// The PNG text keyword the A1111 convention publishes under.
///
/// Unprefixed, unlike [`crate::workflow_png::WORKFLOW_CHUNK_KEYWORD`], and deliberately so: the
/// whole value of this chunk is that a reader who has never heard of SceneWorks finds it under the
/// name it already looks for. Namespacing it would make it correct and useless.
pub const PARAMETERS_CHUNK_KEYWORD: &str = "parameters";

/// The label A1111 puts in front of the negative prompt, on its own line.
///
/// Public because it is the token a parser splits on, and
/// `crates/sceneworks-core/tests/workflow_parameters.rs` implements that parser against this
/// constant rather than against a copy of the string.
pub const NEGATIVE_PROMPT_LABEL: &str = "Negative prompt: ";

/// Every key that may appear on the trailing settings line, in emission order.
///
/// The decision record for "omit rather than approximate": a key is here because the envelope holds
/// something that *is* that field, not something that resembles it. The module docs list what was
/// left out and why. `the_settings_line_emits_exactly_the_declared_keys` pins this list against the
/// rendered output of a fully-populated envelope, so a key added to the renderer without being
/// declared here fails, and a key declared here that nothing can emit fails too.
pub const SETTINGS_FIELDS: &[(&str, &str)] = &[
    (
        "Steps",
        "`advanced.steps`, when it is a whole number of at least one and no multi-phase schedule \
         overrides it.",
    ),
    (
        "Sampler",
        "`advanced.sampler`, verbatim — except the literal `default`, which names no sampler and is \
         omitted.",
    ),
    (
        "CFG scale",
        "`advanced.guidanceScale`, only when `advanced.guidanceMethod` is absent. The studio emits \
         that key only for a method that is NOT plain CFG, so its presence means the number is not \
         a CFG scale.",
    ),
    (
        "Seed",
        "`seed`, verbatim — the seed of THIS image, which is the only one the envelope carries.",
    ),
    (
        "Size",
        "`width` x `height`, only when they are the dimensions of the file being written.",
    ),
    (
        "Model",
        "`model`, the catalog slug, verbatim. A1111's companion `Model hash` is omitted: we have no \
         checkpoint hash to give and inventing one would be a claim about bytes we never read.",
    ),
    (
        "Version",
        "`producer.version` off the envelope's own producer block, so the two chunks in one file \
         cannot disagree about which build wrote it.",
    ),
];

/// Render `share` as the A1111 `parameters` text for an image encoded at `encoded` pixels.
///
/// ```text
/// <prompt>
/// Negative prompt: <negative>
/// Steps: N, Sampler: X, CFG scale: N, Seed: N, Size: WxH, Model: <slug>, Version: <version>
/// ```
///
/// The prompt is always the first line, even when it is empty, because the layout is positional:
/// every parser in the wild treats everything before the `Negative prompt:` line (or before the
/// trailing settings line, when there is no negative) as the prompt. The negative line is omitted
/// entirely when there is no negative prompt, which is what A1111 itself does. The settings line is
/// omitted when nothing on it could be emitted exactly.
///
/// # Why `encoded` is a parameter
///
/// `Size` is the one field where the envelope and the file can honestly disagree. `width` / `height`
/// are the geometry the run *asked for*, and the inline-upscale variant deliberately keeps the base
/// render's numbers (`upscaled_workflow_share` in the worker explains why) while the PNG it is
/// written into is larger. A1111 papers over the same gap with `Hires upscale: 2`, which we do not
/// emit — so a reader would see `Size: 1024x1024` stamped on a 2048² image with nothing to explain
/// it.
///
/// So `Size` is emitted only when the envelope's geometry IS the encoded image's. That makes it a
/// statement about the file rather than an approximation of it, and it costs the field only on the
/// derived-pass images where it would have been wrong. The base render — every ordinary generation —
/// matches by construction, because `write_image_asset` builds its envelope from the same width and
/// height it hands the encoder.
#[must_use]
pub fn parameters_text(share: &WorkflowShare, encoded: (u32, u32)) -> String {
    let mut out = share.prompt.clone();
    if !share.negative_prompt.is_empty() {
        out.push('\n');
        out.push_str(NEGATIVE_PROMPT_LABEL);
        out.push_str(&share.negative_prompt);
    }
    let settings = settings_line(share, encoded);
    if !settings.is_empty() {
        out.push('\n');
        out.push_str(&settings);
    }
    out
}

/// The trailing `Key: value, Key: value` line, or empty when nothing maps exactly.
fn settings_line(share: &WorkflowShare, encoded: (u32, u32)) -> String {
    let advanced = &share.advanced;
    // A multi-phase run has no single step count and no single guidance value. Emitting the
    // top-level ones beside a schedule that overrode them would describe a run that did not happen,
    // and a gallery reader has no way to tell. So both are suppressed together — the presence of the
    // schedule is the signal, and `omitted` carries it too when the schedule itself was too large to
    // record, which is the same fact arriving by the other door.
    let multi_phase = advanced.contains_key("phases")
        || share
            .omitted
            .iter()
            .any(|name| name == crate::workflow_share::OMITTED_PHASES);

    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(SETTINGS_FIELDS.len());
    let mut push = |key: &'static str, value: String| pairs.push((key, value));

    if !multi_phase {
        if let Some(steps) = advanced.get("steps").and_then(whole_number) {
            if steps >= 1 {
                push("Steps", steps.to_string());
            }
        }
    }
    if let Some(sampler) = advanced.get("sampler").and_then(Value::as_str) {
        let sampler = sampler.trim();
        // `default` is a real value in our vocabulary and means "whatever the engine picks". It is
        // not the name of a sampler, and writing it into a field a gallery renders as one would
        // invent a sampler that does not exist.
        if !sampler.is_empty() && sampler != "default" {
            push("Sampler", sampler.to_owned());
        }
    }
    // `guidanceMethod` is emitted by the studio builder ONLY when the method is not plain CFG
    // (`imageJobAdvanced.js` drops the `cfg` no-op), so its presence is exactly the signal that this
    // number is not a CFG scale.
    if !multi_phase && !advanced.contains_key("guidanceMethod") {
        if let Some(scale) = advanced.get("guidanceScale").and_then(finite_number) {
            push("CFG scale", format_number(scale));
        }
    }
    if let Some(seed) = share.seed {
        push("Seed", seed.to_string());
    }
    if share.width == Some(encoded.0) && share.height == Some(encoded.1) {
        push("Size", format!("{}x{}", encoded.0, encoded.1));
    }
    if !share.model.is_empty() {
        push("Model", share.model.clone());
    }
    // Fed from the envelope's own producer block rather than from `PRODUCER_VERSION` directly, so
    // the two chunks in one file cannot disagree about which build wrote it. An envelope whose
    // producer block was reduced to empty (a version that is not strict semver, on the read side)
    // therefore omits the field rather than substituting this build's.
    if !share.producer.version.is_empty() {
        push("Version", share.producer.version.clone());
    }

    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}: {}", quote(&value)))
        .collect::<Vec<String>>()
        .join(", ")
}

/// A1111's own quoting rule, mirrored.
///
/// `modules/generation_parameters_copypaste.py` JSON-quotes a value containing a comma, a newline or
/// a colon, because all three are the separators the trailing line is parsed with, and its reader
/// un-quotes on the way back. Without it a sampler or model slug containing a comma would silently
/// split into two fields for every reader in the wild.
fn quote(value: &str) -> String {
    if value.contains(',') || value.contains('\n') || value.contains(':') {
        // Infallible for a `str`; the fallback is unreachable and is a fallback rather than an
        // `expect` because a panic in a metadata renderer would fail a generation that succeeded.
        serde_json::to_string(value).unwrap_or_else(|_| value.to_owned())
    } else {
        value.to_owned()
    }
}

/// A JSON number that is a non-negative whole number, whichever way it was spelled.
///
/// `8` and `8.0` are the same step count to a user and arrive differently depending on which builder
/// wrote the payload, so both are read. `8.5` is not a step count and is dropped rather than
/// rounded — rounding is exactly the kind of quiet approximation this module refuses.
fn whole_number(value: &Value) -> Option<u64> {
    if let Some(integer) = value.as_u64() {
        return Some(integer);
    }
    let float = value.as_f64()?;
    if !float.is_finite() || float < 0.0 || float.fract() != 0.0 || float > 4_294_967_295.0 {
        return None;
    }
    // Exact: the guards above put `float` on a whole number inside u32's range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(float as u64)
}

/// A JSON number that is finite. NaN and the infinities are dropped: they render as `NaN` / `inf`,
/// which is not a guidance scale and would be displayed as one.
fn finite_number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

/// `4.0` as `4`, `3.5` as `3.5`.
///
/// A trailing `.0` is what `f64`'s own `Display` produces and is not what any A1111 emitter writes,
/// so a reader diffing two files would see a difference that is not one.
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{:.0}", value)
    } else {
        format!("{value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_share::parse_workflow_share_json;

    fn envelope(extra: serde_json::Value) -> WorkflowShare {
        let mut object = serde_json::json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": {
                "name": "SceneWorks",
                "url": "https://github.com/SceneWorks/SceneWorks",
                "version": "0.8.1"
            },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "a lighthouse in heavy fog",
        });
        let map = object.as_object_mut().expect("an object");
        for (key, value) in extra.as_object().expect("extra is an object") {
            map.insert(key.clone(), value.clone());
        }
        parse_workflow_share_json(&object.to_string()).expect("the fixture parses")
    }

    #[test]
    fn the_layout_is_the_a1111_one() {
        let share = envelope(serde_json::json!({
            "negativePrompt": "text, watermark",
            "seed": 880_412,
            "width": 1024,
            "height": 768,
            "advanced": { "steps": 28, "sampler": "euler", "guidanceScale": 3.5 },
        }));
        assert_eq!(
            parameters_text(&share, (1024, 768)),
            "a lighthouse in heavy fog\n\
             Negative prompt: text, watermark\n\
             Steps: 28, Sampler: euler, CFG scale: 3.5, Seed: 880412, Size: 1024x768, \
             Model: z_image_turbo, Version: 0.8.1"
        );
    }

    #[test]
    fn an_empty_negative_prompt_omits_its_whole_line() {
        // A1111 does the same. An empty `Negative prompt:` line is a claim that the run had one.
        let share = envelope(serde_json::json!({ "seed": 7 }));
        let text = parameters_text(&share, (1024, 768));
        assert!(!text.contains(NEGATIVE_PROMPT_LABEL), "{text:?}");
        assert_eq!(
            text,
            "a lighthouse in heavy fog\nSeed: 7, Model: z_image_turbo, Version: 0.8.1"
        );
    }

    #[test]
    fn a_multi_phase_run_omits_steps_and_cfg_rather_than_picking_one() {
        // The AC's named case. A Krea multi-phase schedule is N (steps, guidance) pairs; a single
        // number beside it describes a run that did not happen.
        let share = envelope(serde_json::json!({
            "seed": 4,
            "advanced": {
                "steps": 28,
                "guidanceScale": 3.5,
                "sampler": "euler",
                "textStyleGain": 1.4,
                "phases": [
                    { "steps": 12, "guidance": 4.0 },
                    { "steps": 16, "guidance": 2.0 }
                ]
            },
        }));
        let text = parameters_text(&share, (1024, 768));
        assert!(!text.contains("Steps:"), "{text:?}");
        assert!(!text.contains("CFG scale:"), "{text:?}");
        // And nothing invented a field for the schedule or the tap-reweight gain.
        assert!(
            !text.contains("phases") && !text.contains("1.4"),
            "{text:?}"
        );
        // The fields that DO map exactly still travel.
        assert!(
            text.contains("Sampler: euler") && text.contains("Seed: 4"),
            "{text:?}"
        );
    }

    #[test]
    fn a_dropped_phase_schedule_suppresses_them_too() {
        // The schedule can arrive as an `omitted` marker instead of as a key, when it was over the
        // recording cap. Same fact, other door.
        let share = envelope(serde_json::json!({
            "advanced": { "steps": 28, "guidanceScale": 3.5 },
            "omitted": ["advanced.phases"],
        }));
        let text = parameters_text(&share, (1024, 768));
        assert!(
            !text.contains("Steps:") && !text.contains("CFG scale:"),
            "{text:?}"
        );
    }

    #[test]
    fn a_non_cfg_guidance_method_omits_the_cfg_scale() {
        // `imageJobAdvanced.js` emits `guidanceMethod` only when it is NOT the `cfg` no-op, so its
        // presence means the number is a CFG++ scale and labelling it `CFG scale` is a guess.
        let share = envelope(serde_json::json!({
            "advanced": { "guidanceScale": 3.5, "guidanceMethod": "cfg_pp" },
        }));
        assert!(!parameters_text(&share, (1, 1)).contains("CFG scale"));

        let plain = envelope(serde_json::json!({ "advanced": { "guidanceScale": 3.5 } }));
        assert!(parameters_text(&plain, (1, 1)).contains("CFG scale: 3.5"));
    }

    #[test]
    fn a_default_sampler_names_no_sampler_at_all() {
        let share = envelope(serde_json::json!({ "advanced": { "sampler": "default" } }));
        assert!(!parameters_text(&share, (1, 1)).contains("Sampler"));
    }

    #[test]
    fn size_is_a_statement_about_the_file() {
        let share = envelope(serde_json::json!({ "width": 1024, "height": 1024 }));
        assert!(parameters_text(&share, (1024, 1024)).contains("Size: 1024x1024"));
        // The upscaled variant: the envelope keeps the base render's geometry on purpose, so the
        // field would be a lie about the file it is stamped on.
        assert!(!parameters_text(&share, (2048, 2048)).contains("Size"));
        // And an envelope with no geometry says nothing rather than guessing from the pixels.
        let bare = envelope(serde_json::json!({}));
        assert!(!parameters_text(&bare, (1024, 1024)).contains("Size"));
    }

    #[test]
    fn steps_are_a_whole_number_or_nothing() {
        for (value, expected) in [
            (serde_json::json!(28), Some("Steps: 28")),
            (serde_json::json!(28.0), Some("Steps: 28")),
            (serde_json::json!(28.5), None),
            (serde_json::json!(0), None),
            (serde_json::json!("28"), None),
        ] {
            let share = envelope(serde_json::json!({ "advanced": { "steps": value } }));
            let text = parameters_text(&share, (1, 1));
            match expected {
                Some(fragment) => assert!(text.contains(fragment), "{value} -> {text:?}"),
                None => assert!(!text.contains("Steps"), "{value} -> {text:?}"),
            }
        }
    }

    #[test]
    fn a_value_carrying_a_separator_is_quoted_the_way_a1111_quotes_it() {
        // Otherwise a comma inside a model slug splits one field into two for every reader.
        assert_eq!(quote("euler"), "euler");
        assert_eq!(quote("euler, ancestral"), "\"euler, ancestral\"");
        assert_eq!(quote("a:b"), "\"a:b\"");
        assert_eq!(quote("two\nlines"), "\"two\\nlines\"");
    }

    #[test]
    fn integral_guidance_loses_its_trailing_zero() {
        assert_eq!(format_number(4.0), "4");
        assert_eq!(format_number(3.5), "3.5");
        assert_eq!(format_number(0.0), "0");
    }

    #[test]
    fn the_version_is_the_envelopes_own_producer_version() {
        // The AC: the two chunks cannot disagree about which build wrote the file.
        let share = envelope(serde_json::json!({}));
        assert_eq!(share.producer.version, "0.8.1");
        assert!(parameters_text(&share, (1, 1)).ends_with("Version: 0.8.1"));
    }
}
