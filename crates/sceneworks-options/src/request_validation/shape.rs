//! Shared payload shape and cross-field policy. No database, device or filesystem access.
use super::ValidationRefusal as ApiError;
use crate::video_request::{classify_reference_set, ReferenceSetVerdict, VIDEO_JOB_MODES};
use crate::{contracts::JsonObject, generation_contracts::*, MAX_PROMPT_CHARS};
use serde::Serialize;
use serde_json::Value;
pub const MAX_ADVANCED_JSON_BYTES: usize = 64 * 1024;
pub const MAX_IMAGE_DIMENSION: u32 = 4096;
pub const SHAPE_MIN_IMAGE_DIMENSION: u32 = 16;
pub const MAX_VIDEO_DIMENSION: u32 = 1920;
pub const MAX_VIDEO_REFERENCE_ASSET_IDS: usize = 9;
pub const MAX_VIDEO_SOURCE_CLIP_ASSET_IDS: usize = 8;
pub const MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS: usize = 3;
pub const AUDIO_ICL_MODES: &[&str] = &["single", "dual"];
pub const AUDIO_QUANT_TIERS: &[&str] = &["bf16", "q8", "q4"];
pub const AUDIO_ICL_DEFAULT_END_SECS: f32 = 30.0;
pub const AUDIO_OUTPUT_LIMITERS: &[&str] = &["clamp", "rescale"];
pub const AUDIO_EDIT_MODES: &[&str] = &["inpaint", "repaint", "extend", "cover"];
pub fn validate_prompt_extras(
    negative_prompt: &str,
    advanced: &impl Serialize,
) -> Result<(), ApiError> {
    if negative_prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "negativePrompt must be at most {MAX_PROMPT_CHARS} characters"
        )));
    }
    // Serialize once to measure the on-the-wire size of the pass-through bag.
    let advanced_bytes = serde_json::to_vec(advanced)
        .map(|bytes| bytes.len())
        .unwrap_or(0);
    if advanced_bytes > MAX_ADVANCED_JSON_BYTES {
        return Err(ApiError::bad_request(format!(
            "advanced settings must serialize to at most {MAX_ADVANCED_JSON_BYTES} bytes"
        )));
    }
    Ok(())
}

pub fn validate_image_prompt_extras(
    negative_prompt: &str,
    advanced: &JsonObject,
) -> Result<(), ApiError> {
    validate_image_pose_count(advanced)?;
    let Some(poses) = advanced.get("poses").and_then(Value::as_array) else {
        return validate_prompt_extras(negative_prompt, advanced);
    };
    // 16 KiB per supported output accommodates 128 whole-body points with three f64 coordinates
    // each, JSON framing and the pose id. This remains below the route's total request-body cap.
    const MAX_POSE_JSON_BYTES: usize = crate::MAX_JOB_POSES * 16 * 1024;
    if serde_json::to_vec(poses)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
        > MAX_POSE_JSON_BYTES
    {
        return Err(ApiError::bad_request(format!(
            "pose data must serialize to at most {MAX_POSE_JSON_BYTES} bytes; select fewer poses"
        )));
    }
    let settings: std::collections::BTreeMap<_, _> = advanced
        .iter()
        .filter(|(key, _)| key.as_str() != "poses")
        .collect();
    validate_prompt_extras(negative_prompt, &settings)
}

pub fn validate_model_id(model: &str) -> Result<(), ApiError> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request("model is required"));
    }
    let mut components = std::path::Path::new(trimmed).components();
    let single_normal = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if !single_normal || trimmed.contains('/') || trimmed.contains('\\') {
        return Err(ApiError::bad_request(
            "model must be a plain model id (no path separators or '..')",
        ));
    }
    Ok(())
}

pub fn validate_image_job<A: Serialize, M: Serialize>(
    payload: &ImageJobRequest<A, M>,
) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    if (payload.moss_image_advanced.is_some() || payload.advanced.contains_key("mossImageAdvanced"))
        && (payload.mode != "edit_image" || !crate::is_moss_image(&payload.model))
    {
        return Err(ApiError::bad_request(
            "mossImageAdvanced tensor/output controls require explicit MOSS edit_image mode",
        ));
    }
    let positive_tensor = payload.mode == "edit_image"
        && crate::is_moss_image(&payload.model)
        && (payload.moss_image_advanced.as_ref().is_some_and(|a| {
            serde_json::to_value(a)
                .ok()
                .and_then(|v| v.get("positive").cloned())
                .is_some_and(|v| !v.is_null())
        }) || payload
            .advanced
            .get("mossImageAdvanced")
            .and_then(|v| v.get("positive"))
            .is_some_and(|v| !v.is_null()));
    if (!positive_tensor && payload.prompt.is_empty())
        || payload.prompt.chars().count() > MAX_PROMPT_CHARS
    {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    // A native prompt batch (sc-25681): every prompt obeys the single prompt's length bound. Its
    // shape against the model (declared cap, `prompt == prompts[0]`, `prompts × count`) is judged
    // in `create_image_job` against the canonical catalog entry.
    if payload
        .prompts
        .iter()
        .any(|prompt| prompt.is_empty() || prompt.chars().count() > MAX_PROMPT_CHARS)
    {
        return Err(ApiError::bad_request(
            "each of prompts must be between 1 and 4000 characters",
        ));
    }
    validate_image_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    // sc-26241: a FLUX 3 request names exactly one of its packages, and the publisher FP8R tier
    // exists only for the model that ships it. Which backend and worker can load it is routing's.
    if payload.loras.len() > crate::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            crate::MAX_JOB_LORAS
        )));
    }
    if ![
        "text_to_image",
        "edit_image",
        "character_image",
        "style_variations",
    ]
    .contains(&payload.mode.as_str())
    {
        return Err(ApiError::bad_request("Unsupported image mode"));
    }
    // Only a *named* count is bounded here: an omitted one resolves to the model's declared
    // `defaults.count` in `create_image_job`, like the size below.
    if let Some(count) = payload.count {
        if let Some(error) =
            (!(1..=8).contains(&count)).then(|| "count must be between 1 and 8".to_owned())
        {
            return Err(ApiError::bad_request(error));
        }
    }
    // Only a *named* dimension is bounded here: an omitted side is resolved from the model's
    // declared `defaults.resolution` in `create_image_job` (sc-12400), the same shape as the video
    // route's duration/fps/size.
    //
    // sc-24113 lowers the FLOOR of this shape check from 256 to 32 and nothing else. This gate runs
    // before the model is known, so — exactly like the video route's `1..=30` duration blanket — it
    // is a payload-sanity outer bound, not the model's envelope. It had to widen because
    // Qwen-Image 2.1 genuinely renders from 32 px per side, which the old blanket refused outright
    // with no model ever getting a say.
    //
    // Nothing is loosened for any other model: `create_image_job` immediately re-checks the
    // resolved geometry against the model's own `limits.minDimension` / `maxDimension` /
    // `requiresDimensionsMultipleOf`, and an image model that declares NO floor falls back to
    // `HISTORICAL_MIN_IMAGE_DIMENSION` — the same 256 this line used to enforce. So a 64-px request
    // to any pre-existing model still 400s, just one layer down and with the model named.
    // Reference output can be one 16px token wide. A preset's model is not canonical yet, so
    // allow that outer shape for reference-bearing presets too; the post-preset model/family
    // validation still enforces the actual floor and never loosens another model's envelope.
    let has_reference = !payload.reference_asset_ids.is_empty()
        || payload
            .reference_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        || payload
            .source_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty());
    let shape_floor = if has_reference
        && (matches!(payload.model.as_str(), "flux3_klein" | "flux3_klein_base")
            || payload.recipe_preset_id.is_some())
    {
        16
    } else {
        SHAPE_MIN_IMAGE_DIMENSION
    };
    if let Some(width) = payload.width {
        validate_dimension_in(width, "width", shape_floor, MAX_IMAGE_DIMENSION)?;
    }
    if let Some(height) = payload.height {
        validate_dimension_in(height, "height", shape_floor, MAX_IMAGE_DIMENSION)?;
    }
    payload.upscale.validate().map_err(ApiError::bad_request)?;

    Ok(())
}

pub fn validate_image_pose_count(advanced: &JsonObject) -> Result<(), ApiError> {
    let Some(poses) = advanced.get("poses").and_then(Value::as_array) else {
        return Ok(());
    };
    if poses.len() > crate::MAX_JOB_POSES {
        return Err(ApiError::bad_request(format!(
            "advanced.poses must contain at most {} entries; each pose renders one image",
            crate::MAX_JOB_POSES
        )));
    }
    Ok(())
}

pub fn validate_video_job<A>(payload: &VideoJobRequest<A>) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    if payload.loras.len() > crate::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            crate::MAX_JOB_LORAS
        )));
    }
    if !VIDEO_JOB_MODES.contains(&payload.mode.as_str()) {
        return Err(ApiError::bad_request("Unsupported video mode"));
    }
    if payload
        .reference_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "referenceAssetIds must not contain blank ids",
        ));
    }
    if payload.reference_asset_ids.len() > MAX_VIDEO_REFERENCE_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAssetIds must contain at most {MAX_VIDEO_REFERENCE_ASSET_IDS} ids"
        )));
    }
    if payload
        .source_clip_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "sourceClipAssetIds must not contain blank ids",
        ));
    }
    if payload.source_clip_asset_ids.len() > MAX_VIDEO_SOURCE_CLIP_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "sourceClipAssetIds must contain at most {MAX_VIDEO_SOURCE_CLIP_ASSET_IDS} ids"
        )));
    }
    // The audio references (sc-17160), rejected blank-first and bounded exactly like the two
    // id lists above — same order, same wording — because "consistent with the existing lists"
    // is the contract a caller reads off one list and applies to the next.
    if payload
        .reference_audio_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "referenceAudioAssetIds must not contain blank ids",
        ));
    }
    if payload.reference_audio_asset_ids.len() > MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAudioAssetIds must contain at most {MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS} ids"
        )));
    }
    // There is deliberately NO combined blanket here, only the three per-list ones above. 12 is
    // MiniMax-H3's number, not a product-wide truth: today's caps admit 8 images AND 8 clips on
    // one request, so a blanket 12 would refuse a 16-file shape this route accepts right now —
    // narrowing every existing video model, which is precisely what the cap change must not do.
    // The combined budget is a per-model fact and lives with the model, as
    // `limits.maxCombinedReferenceAssets` (see `create_video_job`).
    // Only a *named* duration is bounded here, for the same reason as fps below: an omitted one is
    // resolved from the model's declared `defaults.duration` in `create_video_job`. This blanket
    // stays a payload-sanity check; the model's own `limits.hardMaxDuration` is enforced there
    // (sc-12297 / sc-12400).
    if let Some(duration) = payload.duration.as_ref() {
        let duration = duration
            .as_f64()
            .ok_or_else(|| ApiError::bad_request("duration must be a number between 1 and 30"))?;
        if !duration.is_finite() || !(1.0..=30.0).contains(&duration) {
            return Err(ApiError::bad_request("duration must be between 1 and 30"));
        }
    }
    // Only a *named* fps is bounded here: an omitted one is resolved from the model's declared
    // `defaults.fps` in `create_video_job`, which is the first point that knows the model (this
    // runs before preset expansion, so the model here may be stale — sc-12300). This blanket stays
    // a payload-sanity check; the model's own `limits.fps` menu is enforced there (sc-12347).
    if let Some(fps) = payload.fps {
        if !(1..=60).contains(&fps) {
            return Err(ApiError::bad_request("fps must be between 1 and 60"));
        }
    }
    // Only a *named* dimension is bounded here, like duration and fps above: an omitted side is
    // resolved from the model's declared `defaults.resolution` in `create_video_job` (sc-12400).
    if let Some(width) = payload.width {
        validate_dimension(width, "width", MAX_VIDEO_DIMENSION)?;
    }
    if let Some(height) = payload.height {
        validate_dimension(height, "height", MAX_VIDEO_DIMENSION)?;
    }
    match payload.mode.as_str() {
        "image_to_video"
            if payload.source_asset_id.is_none()
                // Shape validation cannot identify an arbitrary imported model id. Defer
                // last-only compatibility to the resolved-entry validator shared with replay.
                && !payload
                    .last_frame_asset_id
                    .as_deref()
                    .is_some_and(|id| !id.trim().is_empty()) =>
        {
            Err(ApiError::bad_request(
                "Image to Video requires a source image.",
            ))
        }
        "first_last_frame"
            if payload.source_asset_id.is_none() || payload.last_frame_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "First/Last Frame requires first and last image assets.",
            ))
        }
        "extend_clip" if payload.source_clip_asset_id.is_none() => {
            Err(ApiError::bad_request("Extend Clip requires a source clip."))
        }
        "video_bridge"
            if payload.source_clip_asset_id.is_none()
                || payload.bridge_right_clip_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "Bridge generation requires left and right source clips.",
            ))
        }
        "replace_person" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a source clip.",
        )),
        "replace_person" if payload.person_track_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a selected person track.",
        )),
        "replace_person" if payload.character_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a Character.",
        )),
        // Bernini editing / reference-driven video modes (sc-4703): each requires its
        // source media so the worker never falls through to an unconditioned t2v render.
        "video_to_video" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Video to Video requires a source clip.",
        )),
        // `reference_to_video` requires at least one VISUAL reference — an image or a video clip.
        // Audio references are admitted alongside them and never instead of them.
        //
        // Two corrections in one line. sc-17159 widened this from `reference_asset_ids.is_empty()`
        // because Bernini, the only r2v model at the time, takes images alone, so "at least one
        // reference image" and "at least one reference" were the same sentence — true for the
        // clips MiniMax-H3 Ref2VA added, and WRONG for the audio it added at the same time.
        //
        // sc-19574 settled it against the reference implementation rather than by argument.
        // diffusers `MiniMaxH3` (upstream PR #14355, `0.40.0.dev0 @ 7564fb01`) states the rule on
        // `MiniMaxH3AudioReference` — "never on its own — an audio reference has to be paired with
        // at least one image or video reference. It never reaches the conditioner and is encoded by
        // the audio VAE alone" — and ENFORCES it in `before_encoder.py`:
        //
        //     if set(kinds) == {"audio"}:
        //         raise ValueError("An audio reference has to be paired with at least one image or
        //                           video reference and cannot be used on its own.")
        //
        // So the engine was right and this layer was wrong: an audio-only set leaves the visual
        // conditioner with nothing to read, and the render it would produce is unconditioned. The
        // worker refuses it too (sc-19508, `minimax_h3_validate_partition`); refusing it HERE is
        // what makes the user find out at submission instead of after a queued job fails.
        //
        // The rule itself is `crate::video_request::classify_reference_set`, which the
        // MCP tool and the worker call too — one predicate, three layers, so they cannot drift back
        // into disagreement. Only the WORDING is this layer's. It does NOT loosen Bernini: its own
        // conditioning assembly (`resolve_bernini_conditioning`, both lanes) still refuses an r2v
        // with no `referenceAssetIds`, naming bernini — the model-specific half of the requirement
        // belongs with the model, exactly like `limits.maxReferenceAssets`.
        "reference_to_video"
            if classify_reference_set(
                payload.reference_asset_ids.len(),
                payload.source_clip_asset_ids.len(),
                payload.reference_audio_asset_ids.len(),
            ) != ReferenceSetVerdict::Conditionable =>
        {
            Err(ApiError::bad_request(
                "Reference to Video requires at least one reference image or video clip. Audio \
                 references condition the soundtrack and cannot be the only reference.",
            ))
        }
        "reference_video_to_video" if payload.source_clip_asset_id.is_none() => Err(
            ApiError::bad_request("Reference + Video requires a source clip."),
        ),
        "reference_video_to_video" if payload.reference_asset_ids.is_empty() => Err(
            ApiError::bad_request("Reference + Video requires at least one reference image."),
        ),
        // Bernini multi-source-video modes (sc-5425): mv2v blends multiple source clips;
        // ads2v edits a source clip using a reference video + reference images. Each
        // requires its full media set so the worker never falls through to an
        // unconditioned render.
        "multi_video_to_video" if payload.source_clip_asset_ids.len() < 2 => Err(
            ApiError::bad_request("Multi-Clip → Video requires at least two source clips."),
        ),
        "ads2v" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Source + Reference Video requires a source clip.",
        )),
        "ads2v" if payload.reference_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Source + Reference Video requires a reference video.",
        )),
        "ads2v" if payload.reference_asset_ids.is_empty() => Err(ApiError::bad_request(
            "Source + Reference Video requires at least one reference image.",
        )),
        // SCAIL-2 standalone character animation (sc-5449): the same required-media contract the
        // worker's `resolve_scail2_conditioning` enforces — the character is `referenceAssetIds[0]`
        // (preferred) or the i2v `sourceAssetId`, and the motion comes from `sourceClipAssetId`.
        // Both are hard engine inputs (`Reference` + `ControlClip`), so a missing one is a rejected
        // enqueue rather than a job that fails minutes later inside the worker.
        "animate_character" if payload.source_clip_asset_id.is_none() => Err(
            ApiError::bad_request("Animate Character requires a driving video."),
        ),
        "animate_character"
            if payload.reference_asset_ids.is_empty() && payload.source_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "Animate Character requires a reference character image.",
            ))
        }
        _ => Ok(()),
    }
}

pub fn validate_audio_job(payload: &AudioJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    // A multi-speaker request (sc-13676) carries the text in its `script`, so the `prompt` may be
    // empty then; a single-voice request still requires a prompt. The prompt length ceiling always
    // applies. This keeps a single-voice request (script: None) byte-for-byte on the original path.
    let has_script = payload
        .script
        .as_deref()
        .is_some_and(|segments| !segments.is_empty());
    if payload.prompt.chars().count() > MAX_PROMPT_CHARS
        || (payload.prompt.trim().is_empty() && !has_script)
    {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters (or provide a multi-speaker script)",
        ));
    }
    // Multi-speaker dialogue script (sc-13676): when present, every segment must carry non-empty text,
    // and the segment text totals plus the speaker/style labels are bounded so a runaway payload can't
    // slip past the prompt-size floor. The model's advertised `audio.maxSpeakers` is the real
    // speaker-count gate the generator's `validate` applies at the gen-core floor (a script sent to a
    // model that does not advertise `supports_multi_speaker` is a typed Unsupported there); this only
    // rejects a structurally malformed script up front.
    if let Some(segments) = payload.script.as_deref() {
        if segments.is_empty() {
            return Err(ApiError::bad_request(
                "script must have at least one segment",
            ));
        }
        let mut total = 0usize;
        for segment in segments {
            if segment.text.trim().is_empty() {
                return Err(ApiError::bad_request(
                    "each script segment must have non-empty text",
                ));
            }
            total += segment.text.chars().count();
            if segment
                .speaker
                .as_deref()
                .is_some_and(|speaker| speaker.chars().count() > 64)
            {
                return Err(ApiError::bad_request(
                    "script segment speaker label must be at most 64 characters",
                ));
            }
            if segment
                .style
                .as_deref()
                .is_some_and(|style| style.chars().count() > 64)
            {
                return Err(ApiError::bad_request(
                    "script segment style must be at most 64 characters",
                ));
            }
        }
        if total > MAX_PROMPT_CHARS {
            return Err(ApiError::bad_request(format!(
                "script text must total at most {MAX_PROMPT_CHARS} characters"
            )));
        }
    }
    // Reuse the shared negative-prompt + `advanced`-size guard. Music models (ACE-Step) can carry a
    // negative prompt when they advertise support; the size guard bounds it exactly like image/video.
    validate_prompt_extras(
        payload.negative_prompt.as_deref().unwrap_or(""),
        &payload.advanced,
    )?;
    if let Some(target) = payload.target_duration_secs {
        if !target.is_finite() || !(0.0..=300.0).contains(&target) {
            return Err(ApiError::bad_request(
                "targetDurationSecs must be between 0 and 300",
            ));
        }
    }
    // Music describe-the-music sub-block (ACE-Step, sc-13410). BPM must be a real positive tempo; the
    // model's own `validate` re-checks it (finite & > 0), so this is a blanket sanity bound. key/lyrics
    // are free-form — bound their length so a runaway payload can't slip past the prompt-size floor.
    if let Some(bpm) = payload.bpm {
        if !bpm.is_finite() || !(0.0..=1000.0).contains(&bpm) || bpm <= 0.0 {
            return Err(ApiError::bad_request("bpm must be between 0 and 1000"));
        }
    }
    if payload
        .musical_key
        .as_deref()
        .is_some_and(|key| key.chars().count() > 64)
    {
        return Err(ApiError::bad_request(
            "musicalKey must be at most 64 characters",
        ));
    }
    if payload
        .lyrics
        .as_deref()
        .is_some_and(|lyrics| lyrics.chars().count() > MAX_PROMPT_CHARS)
    {
        return Err(ApiError::bad_request(format!(
            "lyrics must be at most {MAX_PROMPT_CHARS} characters"
        )));
    }
    // Extend/edit source band (Conditioning::AudioEdit, sc-13410). Source id + edit mode are a pair: one
    // without the other is a malformed edit request. The mode must be a known token (the model's
    // advertised `audio.editModes` is the real gate the generator's `validate` applies — this only
    // rejects a garbage token up front). Region seconds must be finite and well-ordered; strength is a
    // 0..=1 weight.
    validate_audio_edit_fields(payload)?;
    // Recommendations remain model defaults. Reject non-finite guidance and a non-positive
    // flow shift; the provider owns any additional mathematical restrictions.
    if let Some(guidance) = payload.guidance {
        if !guidance.is_finite() {
            return Err(ApiError::bad_request("guidance (CFG scale) must be finite"));
        }
    }
    if payload
        .scheduler_shift
        .is_some_and(|shift| !shift.is_finite() || shift <= 0.0)
    {
        return Err(ApiError::bad_request(
            "schedulerShift must be finite and greater than 0",
        ));
    }
    if let Some(steps) = payload.steps {
        if !(1..=10_000).contains(&steps) {
            return Err(ApiError::bad_request("steps must be between 1 and 10000"));
        }
    }
    // Voice Clone (sc-13411 C4). The match strength overrides OpenVoice V2's posterior-sampling
    // temperature τ — bounded to a blanket sane 0..=1 range here (the UI-facing knob range); the
    // converter re-checks it (finite, >= 0). `baseModel`, when supplied, must be a well-formed model id
    // (the route additionally asserts it is a `type: "audio"` model). `referenceAudioAssetId` is a plain
    // library asset id resolved (project-scoped) by the worker; no membership check belongs here.
    if let Some(strength) = payload.match_strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(ApiError::bad_request(
                "matchStrength must be between 0 and 1",
            ));
        }
    }
    validate_model_id(&payload.base_model)?;
    validate_audio_song_fields(payload)?;
    Ok(())
}

pub fn validate_audio_song_fields(payload: &AudioJobRequest) -> Result<(), ApiError> {
    if let Some(segments) = payload.segments {
        if !(1..=100).contains(&segments) {
            return Err(ApiError::bad_request("segments must be between 1 and 100"));
        }
    }
    if let Some(tokens) = payload.max_new_tokens_per_segment {
        // YuE's stage-1 context is 16384 positions and each segment keeps `16384 - budget - 1` of
        // them for its prompt, which must be at least 1 — so 16382 is the largest budget the engine
        // accepts. Refused here rather than after admission, the ICL encode and the 7B load.
        if !(1..=16_382).contains(&tokens) {
            return Err(ApiError::bad_request(
                "maxNewTokensPerSegment must be between 1 and 16382",
            ));
        }
    }
    if let Some(penalty) = payload.repetition_penalty {
        if !penalty.is_finite() || penalty <= 0.0 || penalty > 10.0 {
            return Err(ApiError::bad_request(
                "repetitionPenalty must be greater than 0 and at most 10",
            ));
        }
    }
    if payload.guidance_enabled == Some(false) && payload.guidance.is_some() {
        return Err(ApiError::bad_request(
            "guidance (CFG scale) cannot be set while guidanceEnabled is false",
        ));
    }
    if let Some(tier) = payload.quant_tier.as_deref() {
        if !AUDIO_QUANT_TIERS.contains(&tier.trim().to_lowercase().as_str()) {
            return Err(ApiError::bad_request(format!(
                "quantTier must be one of {AUDIO_QUANT_TIERS:?}"
            )));
        }
    }
    if let Some(limiter) = payload.output_limiter.as_deref() {
        if !AUDIO_OUTPUT_LIMITERS.contains(&limiter.trim().to_lowercase().as_str()) {
            return Err(ApiError::bad_request(format!(
                "outputLimiter must be one of {AUDIO_OUTPUT_LIMITERS:?}"
            )));
        }
    }
    let present = |value: &Option<String>| value.as_deref().is_some_and(|id| !id.trim().is_empty());
    let single = present(&payload.icl_reference_asset_id);
    let vocal = present(&payload.icl_vocal_asset_id);
    let instrumental = present(&payload.icl_instrumental_asset_id);
    let window = payload.icl_start_secs.is_some() || payload.icl_end_secs.is_some();
    match payload
        .icl_mode
        .as_deref()
        .map(|mode| mode.trim().to_lowercase())
        .filter(|mode| !mode.is_empty())
        .as_deref()
    {
        None => {
            if single || vocal || instrumental || window {
                return Err(ApiError::bad_request(
                    "ICL reference fields need an iclMode (\"single\" or \"dual\")",
                ));
            }
        }
        Some("single") => {
            if !single || vocal || instrumental {
                return Err(ApiError::bad_request(
                    "iclMode \"single\" takes exactly iclReferenceAssetId (no vocal/instrumental ids)",
                ));
            }
        }
        Some("dual") => {
            if single || !vocal || !instrumental {
                return Err(ApiError::bad_request(
                    "iclMode \"dual\" takes both iclVocalAssetId and iclInstrumentalAssetId (no \
                     iclReferenceAssetId)",
                ));
            }
        }
        Some(_) => {
            return Err(ApiError::bad_request(format!(
                "iclMode must be one of {AUDIO_ICL_MODES:?}"
            )));
        }
    }
    for (field, value) in [
        ("iclStartSecs", payload.icl_start_secs),
        ("iclEndSecs", payload.icl_end_secs),
    ] {
        if let Some(secs) = value {
            if !secs.is_finite() || !(0.0..=3600.0).contains(&secs) {
                return Err(ApiError::bad_request(format!(
                    "{field} must be between 0 and 3600"
                )));
            }
        }
    }
    // A window with only a start ends at upstream YuE's `prompt_end_time` default (30 s), so a start
    // at or past 30 s with no end is as ill-ordered as an explicit end <= start.
    if let Some(start) = payload.icl_start_secs {
        let end = payload.icl_end_secs.unwrap_or(AUDIO_ICL_DEFAULT_END_SECS);
        if end <= start {
            return Err(ApiError::bad_request(format!(
                "iclEndSecs must be greater than iclStartSecs (the window ends at \
                 {AUDIO_ICL_DEFAULT_END_SECS} s when iclEndSecs is omitted)"
            )));
        }
    }
    if payload.icl_mode.is_some()
        && (present(&payload.reference_audio_asset_id) || present(&payload.source_audio_asset_id))
    {
        return Err(ApiError::bad_request(
            "an ICL reference cannot be combined with referenceAudioAssetId or sourceAudioAssetId",
        ));
    }
    Ok(())
}

pub fn validate_audio_job_for_model(
    payload: &AudioJobRequest,
    entry: &Value,
) -> Result<(), ApiError> {
    let model = &payload.model;
    let audio = entry.get("audio");
    let flag = |key: &str| {
        audio
            .and_then(|audio| audio.get(key))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    if payload.scheduler_shift.is_some() && !flag("supportsSchedulerShift") {
        return Err(ApiError::bad_request(format!(
            "{model} does not take a schedulerShift"
        )));
    }
    let segmented = flag("supportsSegmentedLyrics");
    let repetition = flag("supportsRepetitionPenalty");
    let region = flag("supportsReferenceRegion");
    let reference_audio = audio
        .and_then(|audio| audio.get("conditioning"))
        .and_then(Value::as_array)
        .is_some_and(|kinds| kinds.iter().any(|kind| kind == "ReferenceAudio"));
    let icl_model = segmented && reference_audio;

    if (payload.segments.is_some() || payload.max_new_tokens_per_segment.is_some()) && !segmented {
        return Err(ApiError::bad_request(format!(
            "{model} does not render segmented lyrics (segments / maxNewTokensPerSegment)"
        )));
    }
    if payload.output_limiter.is_some() && !flag("supportsOutputLimiter") {
        return Err(ApiError::bad_request(format!(
            "{model} does not take an outputLimiter"
        )));
    }
    if payload.repetition_penalty.is_some() && !repetition {
        return Err(ApiError::bad_request(format!(
            "{model} does not take a repetitionPenalty"
        )));
    }
    if payload.icl_mode.is_some() && !icl_model {
        return Err(ApiError::bad_request(format!(
            "iclMode needs an in-context-learning (`_icl`) checkpoint; {model} is not one"
        )));
    }
    if (payload.icl_start_secs.is_some() || payload.icl_end_secs.is_some()) && !region {
        return Err(ApiError::bad_request(format!(
            "{model} does not take a reference window (iclStartSecs / iclEndSecs)"
        )));
    }
    // An `_icl` checkpoint WITHOUT a reference is a plain prompt run, exactly as upstream YuE allows
    // (epic R1); only the reverse — ICL fields on a CoT checkpoint — is refused above.
    if segmented {
        if payload
            .lyrics
            .as_deref()
            .is_none_or(|lyrics| lyrics.trim().is_empty())
        {
            return Err(ApiError::bad_request(format!(
                "{model} sings structured lyrics — lyrics are required"
            )));
        }
        let refused = [
            ("steps", payload.steps.is_some()),
            ("targetDurationSecs", payload.target_duration_secs.is_some()),
            ("bpm", payload.bpm.is_some()),
            ("musicalKey", payload.musical_key.is_some()),
            ("voice", payload.voice.is_some()),
            ("script", payload.script.is_some()),
            (
                "sourceAudioAssetId",
                payload.source_audio_asset_id.is_some(),
            ),
            (
                "referenceAudioAssetId",
                payload.reference_audio_asset_id.is_some(),
            ),
        ];
        if let Some((field, _)) = refused.iter().find(|(_, present)| *present) {
            return Err(ApiError::bad_request(format!(
                "{field} is not a control of {model} (put tempo, key and voice in the genre tags; \
                 song length follows the lyrics and segments)"
            )));
        }
        // YuE accepts nonnegative scales; 0..=1 uses its native CFG-off path.
        if payload.guidance.is_some_and(|guidance| guidance < 0.0) {
            return Err(ApiError::bad_request(format!(
                "{model}: guidance must be nonnegative"
            )));
        }
    } else if payload.guidance_enabled.is_some() {
        // The on/off switch is the segmented-song contract (a `0.0` scale reads as "no CFG" there);
        // on any other model a `0.0` scale is a range error, so refuse the switch up front.
        return Err(ApiError::bad_request(format!(
            "{model} does not take guidanceEnabled (set a guidance scale instead)"
        )));
    }
    if let Some(tier) = payload.quant_tier.as_deref() {
        let tier = tier.trim().to_lowercase();
        let offered = entry
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(|downloads| {
                downloads.iter().any(|download| {
                    download.get("coRequisite").and_then(Value::as_bool) != Some(true)
                        && download
                            .get("variant")
                            .and_then(Value::as_str)
                            .is_some_and(|variant| variant.eq_ignore_ascii_case(&tier))
                })
            });
        if !offered {
            return Err(ApiError::bad_request(format!(
                "{model} does not ship a {tier} tier"
            )));
        }
    }
    Ok(())
}

pub fn validate_audio_edit_fields(payload: &AudioJobRequest) -> Result<(), ApiError> {
    let has_source = payload
        .source_audio_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let has_mode = payload
        .edit_mode
        .as_deref()
        .is_some_and(|mode| !mode.trim().is_empty());
    // A source without a mode (or vice versa) is a malformed edit request — one names WHAT to edit, the
    // other HOW. Reject the half-specified pair rather than silently dropping the edit.
    if has_source != has_mode {
        return Err(ApiError::bad_request(
            "an audio edit needs both sourceAudioAssetId and editMode (or neither)",
        ));
    }
    if let Some(mode) = payload
        .edit_mode
        .as_deref()
        .filter(|m| !m.trim().is_empty())
    {
        // Match the worker's case handling (`edit_mode.map(|m| m.to_lowercase())` at deserialize,
        // then `parse_audio_edit_mode`): the token is case-insensitive, so lowercase before the
        // membership check. Otherwise a mixed-case value like "Extend" is 400'd here even though the
        // worker would accept it — the two validation seams must agree.
        if !AUDIO_EDIT_MODES.contains(&mode.to_lowercase().as_str()) {
            return Err(ApiError::bad_request(format!(
                "editMode must be one of {AUDIO_EDIT_MODES:?}"
            )));
        }
    }
    for (field, value) in [
        ("editRegionStartSecs", payload.edit_region_start_secs),
        ("editRegionEndSecs", payload.edit_region_end_secs),
    ] {
        if let Some(secs) = value {
            if !secs.is_finite() || !(0.0..=3600.0).contains(&secs) {
                return Err(ApiError::bad_request(format!(
                    "{field} must be between 0 and 3600"
                )));
            }
        }
    }
    if let (Some(start), Some(end)) = (payload.edit_region_start_secs, payload.edit_region_end_secs)
    {
        if end <= start {
            return Err(ApiError::bad_request(
                "editRegionEndSecs must be greater than editRegionStartSecs",
            ));
        }
    }
    if let Some(strength) = payload.edit_strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(ApiError::bad_request(
                "editStrength must be between 0 and 1",
            ));
        }
    }
    Ok(())
}

pub fn validate_dimension(value: u32, field: &'static str, max: u32) -> Result<(), ApiError> {
    validate_dimension_in(value, field, 256, max)
}

pub fn validate_dimension_in(
    value: u32,
    field: &'static str,
    min: u32,
    max: u32,
) -> Result<(), ApiError> {
    if !(min..=max).contains(&value) {
        return Err(ApiError::bad_request(format!(
            "{field} must be between {min} and {max}"
        )));
    }
    Ok(())
}

pub fn validate_video_reference_asset_ids_payload(
    payload: &JsonObject,
    model_manifest_entry: &Value,
) -> Result<(), ApiError> {
    if payload.get("mode").and_then(Value::as_str) == Some("image_to_video")
        && !payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty())
    {
        let imported_partition = model_manifest_entry
            .get("minimaxH3Partition")
            .and_then(Value::as_str);
        if !crate::h3_keyframes::accepts_last_only_resolved_image_to_video(
            model_manifest_entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            "image_to_video",
            payload
                .get("lastFrameAssetId")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.trim().is_empty()),
            imported_partition,
        ) {
            return Err(ApiError::bad_request(
                "Image to Video requires a source image.",
            ));
        }
    }
    let reference_asset_ids = match payload.get("referenceAssetIds") {
        None => &[][..],
        Some(Value::Array(values)) => values.as_slice(),
        Some(_) => {
            return Err(ApiError::bad_request(
                "referenceAssetIds must be an array of string ids",
            ));
        }
    };
    for value in reference_asset_ids {
        let id = value.as_str().ok_or_else(|| {
            ApiError::bad_request("referenceAssetIds must contain only string ids")
        })?;
        if id.trim().is_empty() {
            return Err(ApiError::bad_request(
                "referenceAssetIds must not contain blank ids",
            ));
        }
        if id != id.trim() {
            return Err(ApiError::bad_request(
                "referenceAssetIds must not contain leading or trailing whitespace",
            ));
        }
    }
    if reference_asset_ids.len() > MAX_VIDEO_REFERENCE_ASSET_IDS {
        return Err(ApiError::bad_request(format!(
            "referenceAssetIds must contain at most {MAX_VIDEO_REFERENCE_ASSET_IDS} ids"
        )));
    }

    let model = match payload.get("model") {
        None => "",
        Some(Value::String(model)) if model == model.trim() => model.as_str(),
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request(
                "model must not contain leading or trailing whitespace",
            ));
        }
        Some(_) => return Err(ApiError::bad_request("model must be a string")),
    };
    let mode = match payload.get("mode") {
        None => "",
        Some(Value::String(mode)) if mode == mode.trim() => mode.as_str(),
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request(
                "mode must not contain leading or trailing whitespace",
            ));
        }
        Some(_) => return Err(ApiError::bad_request("mode must be a string")),
    };
    let reference_count = reference_asset_ids.len();
    if model == "scail2_14b"
        && mode == "animate_character"
        && reference_count > crate::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
    {
        return Err(ApiError::bad_request(format!(
            "SCAIL-2 Animate Character supports at most {} reference characters.",
            crate::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
        )));
    }
    if model == "scail2_14b"
        && mode == "animate_character"
        && reference_count > 1
        && !model_manifest_entry
            .get("ui")
            .and_then(|ui| ui.get("scail2MultiReference"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(ApiError::bad_request(
            "SCAIL-2 Animate Character multi-reference is unavailable until the paired inference descriptor is installed.",
        ));
    }
    Ok(())
}

pub fn validate_vqa_job(payload: &VqaJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.source_asset_id.trim().is_empty() {
        return Err(ApiError::bad_request("sourceAssetId is required"));
    }
    let question = payload.question.trim();
    if question.is_empty() || question.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "question must be between 1 and {MAX_PROMPT_CHARS} characters"
        )));
    }
    validate_prompt_extras("", &payload.advanced)?;
    if !(16..=2048).contains(&payload.max_new_tokens) {
        return Err(ApiError::bad_request(
            "maxNewTokens must be between 16 and 2048",
        ));
    }
    Ok(())
}
pub fn validate_interleave_job(payload: &InterleaveJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.trim().is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "prompt must be between 1 and {MAX_PROMPT_CHARS} characters"
        )));
    }
    validate_prompt_extras("", &payload.advanced)?;
    // Upstream interleave_gen caps the run at 10 generated images.
    if !(1..=10).contains(&payload.max_images) {
        return Err(ApiError::bad_request("maxImages must be between 1 and 10"));
    }
    if payload
        .source_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "sourceAssetIds must not contain blank ids",
        ));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
}
const MIN_VECTOR_SVG_BYTES: u32 = 1024;
const MAX_VECTOR_SVG_BYTES: u32 = 256 * 1024;
pub fn validate_vector_request(payload: &VectorRequest) -> Result<(), ApiError> {
    if payload.project_id.trim().is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    let prompt = payload.prompt.trim();
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "prompt must not exceed {MAX_PROMPT_CHARS} characters"
        )));
    }
    match payload.mode {
        VectorMode::ImageToSvg => {
            if payload
                .source_asset_id
                .as_deref()
                .is_none_or(|id| id.trim().is_empty())
            {
                return Err(ApiError::bad_request(
                    "sourceAssetId is required for image_to_svg",
                ));
            }
        }
        VectorMode::TextToSvg => {
            if prompt.is_empty() {
                return Err(ApiError::bad_request("prompt is required for text_to_svg"));
            }
            if payload.source_asset_id.is_some() {
                return Err(ApiError::bad_request(
                    "sourceAssetId is not accepted for text_to_svg",
                ));
            }
        }
    }
    let sampling = &payload.sampling;
    if !sampling.temperature.is_finite() || !(0.0..=2.0).contains(&sampling.temperature) {
        return Err(ApiError::bad_request(
            "sampling.temperature must be between 0 and 2",
        ));
    }
    if !sampling.top_p.is_finite() || !(0.0 < sampling.top_p && sampling.top_p <= 1.0) {
        return Err(ApiError::bad_request(
            "sampling.topP must be greater than 0 and at most 1",
        ));
    }
    if sampling.top_k > 1_000 {
        return Err(ApiError::bad_request("sampling.topK must be at most 1000"));
    }
    if !sampling.repetition_penalty.is_finite()
        || !(0.1..=2.0).contains(&sampling.repetition_penalty)
    {
        return Err(ApiError::bad_request(
            "sampling.repetitionPenalty must be between 0.1 and 2",
        ));
    }
    if sampling.repetition_context > 32_768 {
        return Err(ApiError::bad_request(
            "sampling.repetitionContext must be at most 32768",
        ));
    }
    let budget = &payload.detail_budget;
    if !(16..=32_768).contains(&budget.max_new_tokens) {
        return Err(ApiError::bad_request(
            "detailBudget.maxNewTokens must be between 16 and 32768",
        ));
    }
    if !(MIN_VECTOR_SVG_BYTES..=MAX_VECTOR_SVG_BYTES).contains(&budget.max_svg_bytes) {
        return Err(ApiError::bad_request(format!(
            "detailBudget.maxSvgBytes must be between {MIN_VECTOR_SVG_BYTES} and {MAX_VECTOR_SVG_BYTES}"
        )));
    }
    if !(1_000..=MAX_VECTOR_WALL_TIME_MS).contains(&budget.max_wall_time_ms) {
        return Err(ApiError::bad_request(format!(
            "detailBudget.maxWallTimeMs must be between 1000 and {MAX_VECTOR_WALL_TIME_MS}"
        )));
    }
    Ok(())
}
pub fn validate_vector_model_manifest(
    model_id: &str,
    mode: VectorMode,
    manifest: &Value,
) -> Result<(), ApiError> {
    if manifest.get("type").and_then(Value::as_str) != Some("vector") {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} is not a vector model (type: \"vector\" required)"
        )));
    }
    let adapter = manifest
        .get("adapter")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|adapter| !adapter.is_empty())
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Model {model_id} has no declared vector provider adapter"
            ))
        })?;
    let supported = manifest
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|capabilities| {
            capabilities
                .iter()
                .any(|capability| capability.as_str() == Some(mode.as_str()))
        });
    if !supported {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} provider {adapter} does not declare the {} capability; the request was not queued",
            mode.as_str()
        )));
    }
    Ok(())
}
pub fn validate_vector_provider_request(
    payload: &VectorRequest,
    model: &Value,
) -> Result<(), ApiError> {
    let vector = model.get("vector").and_then(Value::as_object);
    if payload.mode == VectorMode::ImageToSvg
        && !payload.prompt.trim().is_empty()
        && vector
            .and_then(|value| value.get("acceptsTextGuidance"))
            .and_then(Value::as_bool)
            == Some(false)
    {
        return Err(ApiError::bad_request(format!(
            "Model {} does not accept text guidance for image_to_svg",
            payload.model
        )));
    }
    for (field, requested) in [
        (
            "maxNewTokens",
            u64::from(payload.detail_budget.max_new_tokens),
        ),
        (
            "maxSvgBytes",
            u64::from(payload.detail_budget.max_svg_bytes),
        ),
        ("maxWallTimeMs", payload.detail_budget.max_wall_time_ms),
    ] {
        if let Some(limit) = vector
            .and_then(|value| value.get(field))
            .and_then(Value::as_u64)
        {
            if requested > limit {
                return Err(ApiError::bad_request(format!(
                    "detailBudget.{field} exceeds the selected provider limit of {limit}"
                )));
            }
        }
    }
    Ok(())
}
