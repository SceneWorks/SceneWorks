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
//! | `advanced.phases` (multi-phase Krea) | `Steps` only | The exact total is the sum of the contiguous phase step counts. Guidance remains omitted because it varies by phase. |
//! | `advanced.textStyleGain` | nothing | A Krea tap-reweight gain. No field means it, and inventing one names a knob no reader can interpret. |
//! | `advanced.scheduler`, `schedulerShift` | `Schedule type` | A1111's `Schedule type` is a NOISE schedule (Karras, Exponential). Ours is a diffusers scheduler class, which is closer to A1111's *sampler*. Two plausible mappings and no correct one. |
//! | `advanced.strength` | `Denoising strength` | The sense is lane-dependent — the candle fork lane inverts it (higher is *closer* to the reference). A number whose direction is ambiguous is worse than no number. |
//! | `upscale.*` | `Hires upscale` / `Hires upscaler` | A1111's hires fix is a specific latent second pass. Ours is a separate post-pass on the decoded image. Same words, different pipeline. |
//! | unresolved or unhashed `loras[]` | `<lora:name:weight>` / `Lora hashes` | A readable name is not identity. Only worker-proven adapter bytes are rendered as gallery resources; unresolved recipe hints remain only in the authoritative envelope. |
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

/// The fewest `Key: value` pairs a trailing line may carry and still be read as one.
///
/// A1111's `parse_generation_parameters` pops the last line, counts the pairs its `re_param_code`
/// finds, and puts the line back into the PROMPT when there are fewer than three. Civitai and the
/// rest of the ecosystem's parsers inherit the rule. So this is not our threshold and not a style
/// choice: it is the boundary between a settings line and a line of prompt text, for every reader
/// this chunk is written for.
///
/// [`settings_line`] treats it as a floor and withholds the whole line below it — see the reasoning
/// there. Public so the corpus in `crates/sceneworks-core/tests/workflow_parameters.rs` can pin the
/// lanes against it; that file's `parse_a1111` keeps its own literal threshold, because a parser
/// that took its rule from the renderer it is measuring would agree with it by construction.
/// `the_local_parser_reads_a_real_a1111_block` joins the two.
pub const SETTINGS_PAIR_FLOOR: usize = 3;

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
        "`advanced.steps` for a single-phase run, or the exact sum of `advanced.phases[].steps` for \
         a recorded multi-phase schedule.",
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
        "`model`, the safe readable catalog slug, verbatim.",
    ),
    (
        "Model hash",
        "`modelHash`, only when it is the exact SHA-256 of the executed checkpoint and `Model` is \
         also present. Civitai uses the pair to resolve its linked model-version resource.",
    ),
    (
        "Lora hashes",
        "A quoted name-to-SHA-256 map for every worker-proven adapter. Each key is the same safe, \
         readable key used by its `<lora:key:weight>` prompt tag.",
    ),
    (
        "Version",
        "`producer.version` off the envelope's own producer block, so the two chunks in one file \
         cannot disagree about which build wrote it.",
    ),
    (
        "software",
        "`producer.name` from the trusted envelope producer block. Civitai displays this canonical \
         lowercase field as the generating-software badge.",
    ),
];

/// Render `share` as the A1111 `parameters` text for an image encoded at `encoded` pixels.
///
/// ```text
/// <prompt>
/// Negative prompt: <negative>
/// Steps: N, Sampler: X, CFG scale: N, Seed: N, Size: WxH, Model: <slug>, Version: <version>, software: SceneWorks
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
///
/// # Two limits the format has and we inherit
///
/// Both are properties of the convention rather than of this renderer, and neither is worth
/// diverging from it over — a reader that has to guess which dialect it is holding is worse off than
/// one that occasionally mis-splits.
///
/// * **A prompt containing a line that begins `Negative prompt:` will mis-split.** A1111 has the
///   same behaviour on its own output; the boundary is positional and there is no escape for it.
/// * **A settings line of fewer than three pairs is prompt text to A1111's parser.** Inherited, but
///   not left to chance: [`SETTINGS_PAIR_FLOOR`] omits the line entirely rather than emitting one a
///   gallery would append to the prompt. Every shipping lane clears the floor on its own — the
///   thinnest, a standalone upscale, emits `Seed` + `Model` + `Version` + `software`, one field
///   above the floor — and `the_settings_line_clears_the_pair_floor_a_gallery_parses_with` in
///   `crates/sceneworks-core/tests/workflow_parameters.rs` pins the lane SHAPES rather than assuming
///   a well-populated one.
///
/// The authoritative recipe is unaffected by either: it is the `sceneworks:workflow` chunk, and this
/// block is for display.
#[must_use]
pub fn parameters_text(share: &WorkflowShare, encoded: (u32, u32)) -> String {
    let mut out = share.prompt.clone();
    for (index, lora) in share.loras.iter().enumerate() {
        let (Some(name), Some(weight), Some(_hash)) = (&lora.name, lora.weight, &lora.hash) else {
            continue;
        };
        if !civitai_lora_key(name) || weight < 0.0 || !lora_weight_is_fixed(share, index, weight) {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&format!("<lora:{name}:{}>", format_number(weight)));
    }
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

/// The trailing `Key: value, Key: value` line, or empty when nothing maps exactly — or when what
/// maps is too little for a gallery to recognize as a settings line at all.
fn settings_line(share: &WorkflowShare, encoded: (u32, u32)) -> String {
    let advanced = &share.advanced;
    // A multi-phase run has one exact total step count (the sum of its contiguous segments), but no
    // single guidance value. The top-level values were overridden, so never use them for a phase run.
    // An `omitted` phase marker means the schedule was too large to record and therefore cannot be
    // summed exactly here.
    let multi_phase = advanced.contains_key("phases")
        || share
            .omitted
            .iter()
            .any(|name| name == crate::workflow_share::OMITTED_PHASES);

    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(SETTINGS_FIELDS.len());
    let mut push = |key: &'static str, value: String| pairs.push((key, value));

    let exact_steps = if multi_phase {
        advanced
            .get("phases")
            .and_then(Value::as_array)
            .and_then(|phases| {
                phases.iter().try_fold(0_u64, |total, phase| {
                    let steps = phase.get("steps").and_then(whole_number)?;
                    (steps >= 1).then_some(total.checked_add(steps)?)
                })
            })
    } else {
        advanced.get("steps").and_then(whole_number)
    };
    if let Some(steps) = exact_steps.filter(|steps| *steps >= 1) {
        // Civitai's Automatic parser requires a `Steps` pair before it will inspect `Lora hashes`.
        // A phase schedule's exact step count is the sum of its contiguous denoise segments.
        push("Steps", steps.to_string());
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
        if let Some(model_hash) = &share.model_hash {
            push("Model hash", model_hash.clone());
        }
    }
    let lora_hashes = share
        .loras
        .iter()
        .filter_map(|lora| {
            let name = lora.name.as_deref().filter(|name| civitai_lora_key(name))?;
            let hash = lora.hash.as_deref()?;
            Some(format!("{name}: {hash}"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    if !lora_hashes.is_empty() {
        push("Lora hashes", lora_hashes);
    }
    // Fed from the envelope's own producer block rather than from `PRODUCER_VERSION` directly, so
    // the two chunks in one file cannot disagree about which build wrote it. An envelope whose
    // producer block was reduced to empty (a version that is not strict semver, on the read side)
    // therefore omits the field rather than substituting this build's.
    if !share.producer.version.is_empty() {
        push("Version", share.producer.version.clone());
    }
    if !share.producer.name.is_empty() {
        push("software", share.producer.name.clone());
    }

    // Below the floor the line is not a settings line to anything that reads this chunk — it is a
    // trailing line of the PROMPT. Emitting it anyway would not "give the reader two fields": it
    // would make a gallery display `a lighthouse\nSeed: 880412, Version: 0.8.1` as the prompt, which
    // is a wrong prompt rather than a thin settings block. Omitting the whole line leaves the prompt
    // correct and the settings absent, and the absent settings were never legible in the first place.
    //
    // This is the same "omit rather than approximate" rule the module is built on, applied one level
    // up: the fields are individually exact and the LINE is still not a statement a reader can act
    // on, so the line goes. `sceneworks:workflow` is unaffected — the authoritative recipe rides in
    // the other chunk and carries every one of these fields regardless.
    if pairs.len() < SETTINGS_PAIR_FLOOR {
        return String::new();
    }
    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}: {}", quote(&value)))
        .collect::<Vec<String>>()
        .join(", ")
}

/// Civitai's Automatic parser recognizes this deliberately narrow key alphabet in prompt tags.
/// The digest, not this display key, resolves the resource; keeping the key narrow merely ensures
/// the prompt tag and `Lora hashes` map associate with each other.
fn civitai_lora_key(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

/// A phase schedule may vary one adapter's weight across denoise phases. Such a resource still gets
/// a hash (and therefore an attribution card), but no single prompt-tag weight is emitted unless
/// the adapter is active in every phase and every explicit override agrees with the top-level
/// weight. An omitted adapter means that phase runs base-only, so it is not a fixed-weight run.
fn lora_weight_is_fixed(share: &WorkflowShare, index: usize, weight: f64) -> bool {
    let Some(phases) = share.advanced.get("phases").and_then(Value::as_array) else {
        return true;
    };
    !phases.is_empty()
        && phases.iter().all(|phase| {
            phase
                .get("loras")
                .and_then(Value::as_array)
                .is_some_and(|loras| {
                    let mut matches = loras.iter().filter(|entry| {
                        entry.get("index").and_then(Value::as_u64) == Some(index as u64)
                    });
                    let fixed = matches.next().is_some_and(|entry| {
                        entry
                            .get("weight")
                            .filter(|value| !value.is_null())
                            .map_or(true, |value| {
                                finite_number(value)
                                    .is_some_and(|phase_weight| phase_weight == weight)
                            })
                    });
                    fixed && matches.next().is_none()
                })
        })
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
             Model: z_image_turbo, Version: 0.8.1, software: SceneWorks"
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
            "a lighthouse in heavy fog\nSeed: 7, Model: z_image_turbo, Version: 0.8.1, software: SceneWorks"
        );
    }

    #[test]
    fn a_multi_phase_run_emits_the_exact_total_steps_but_no_single_cfg() {
        // Steps are contiguous segments of one denoise schedule, so their sum is exact. Guidance
        // varies by phase and still has no honest single-number representation.
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
        assert!(text.contains("Steps: 28"), "{text:?}");
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
        //
        // Carries a seed so this also proves the version remains the envelope's value when ordinary
        // numeric settings share the same gallery-readable line.
        let share = envelope(serde_json::json!({ "seed": 7 }));
        assert_eq!(share.producer.version, "0.8.1");
        assert!(parameters_text(&share, (1, 1)).contains("Version: 0.8.1, software: SceneWorks"));
    }

    #[test]
    fn the_resource_line_meets_the_pair_floor_without_numeric_settings() {
        // Model + Version + software is exactly A1111's three-pair recognition floor. A sparse
        // recipe can therefore identify its model and generator without becoming prompt text.
        let resources = envelope(serde_json::json!({}));
        assert_eq!(
            parameters_text(&resources, (1, 1)),
            "a lighthouse in heavy fog\nModel: z_image_turbo, Version: 0.8.1, software: SceneWorks"
        );

        // Additional exact fields join the same line; the guard remains a floor, not a ceiling.
        let with_seed = envelope(serde_json::json!({ "seed": 7 }));
        assert_eq!(
            parameters_text(&with_seed, (1, 1)),
            "a lighthouse in heavy fog\nSeed: 7, Model: z_image_turbo, Version: 0.8.1, software: SceneWorks"
        );
        assert_eq!(SETTINGS_PAIR_FLOOR, 3, "A1111's threshold, not ours");
    }
}
