use super::*;

pub(super) fn requested_gpu_or_auto(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "auto".to_owned()
    } else {
        trimmed.to_owned()
    }
}

pub(super) fn option_str_is_empty(value: Option<&str>) -> bool {
    value.map(str::trim).unwrap_or_default().is_empty()
}

pub(super) fn number_to_f64(
    number: &serde_json::Number,
    field: &'static str,
) -> Result<f64, ApiError> {
    number
        .as_f64()
        .ok_or_else(|| ApiError::bad_request(format!("Invalid numeric value for {field}")))
}

pub(super) fn optional_number_to_f64(
    number: Option<&serde_json::Number>,
    field: &'static str,
) -> Result<Option<f64>, ApiError> {
    number.map(|value| number_to_f64(value, field)).transpose()
}

/// The only output heights `POST /api/v1/projects/:id/timelines/:id/exports` admits. One
/// declaration (sc-22710): the film harness picks an export resolution before it dispatches, and a
/// second hand-copied list would turn a pre-dispatch diagnostic into a 400 at enqueue on drift.
pub(crate) const TIMELINE_EXPORT_RESOLUTIONS: &[u32] = &[640, 720, 1024, 1280];

pub(super) fn validate_timeline_export(payload: &TimelineExportRequest) -> Result<(), ApiError> {
    if !TIMELINE_EXPORT_RESOLUTIONS.contains(&payload.resolution) {
        return Err(ApiError::bad_request(
            "Resolution must be one of 640, 720, 1024, or 1280.",
        ));
    }
    if !(1..=60).contains(&payload.fps) {
        return Err(ApiError::bad_request("FPS must be between 1 and 60"));
    }
    Ok(())
}

pub(super) fn validate_frame_extract(payload: &FrameExtractRequest) -> Result<(), ApiError> {
    if !payload.playhead_seconds.is_finite() || payload.playhead_seconds < 0.0 {
        return Err(ApiError::bad_request(
            "playheadSeconds must be greater than or equal to 0",
        ));
    }
    if ![
        "reuse",
        "first_frame",
        "last_frame",
        "video_studio",
        "image_studio",
        "bridge",
        "extension",
    ]
    .contains(&payload.intended_use.as_str())
    {
        return Err(ApiError::bad_request("Unsupported intendedUse"));
    }
    Ok(())
}

pub(super) fn validate_person_detection_job(
    payload: &PersonDetectionJobRequest,
) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload
        .source_timestamp
        .is_some_and(|timestamp| !timestamp.is_finite() || timestamp < 0.0)
    {
        return Err(ApiError::bad_request(
            "sourceTimestamp must be greater than or equal to 0",
        ));
    }
    Ok(())
}

pub(super) fn validate_person_track_job(payload: &PersonTrackJobRequest) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload.representative_frame_asset_id.is_empty() {
        return Err(ApiError::bad_request(
            "Representative frame asset is required",
        ));
    }
    if payload.track_name.is_empty() || payload.track_name.chars().count() > 120 {
        return Err(ApiError::bad_request(
            "trackName must be between 1 and 120 characters",
        ));
    }
    if !payload.detection.contains_key("id") {
        return Err(ApiError::bad_request(
            "Selected detection metadata is required",
        ));
    }
    Ok(())
}

/// sc-8884 (F-082): `negativePrompt` and the free-form `advanced` bag previously escaped
/// all length validation (only `prompt` was capped), so an oversized field was persisted
/// to jobs.db and re-serialized to every SSE subscriber on each status change. Cap the
/// negative prompt at the same char limit as `prompt` and bound `advanced`'s serialized
/// size. Shared by `validate_image_job` / `validate_video_job`.
pub(super) fn validate_prompt_extras(
    negative_prompt: &str,
    advanced: &JsonObject,
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

pub(super) const PROMPT_ENHANCE_MAX_TEMPERATURE: f64 = 2.0;
pub(super) const PROMPT_ENHANCEMENT_FACT_KEY: &str = "promptEnhancement";

/// Validate the typed, bounded part of the FLUX.2 prompt-enhancement request at every image enqueue
/// boundary. `advanced` is otherwise intentionally extensible, but these fields cross into a native
/// LLM sampler and must never inherit the old truthy/coercing behavior.
pub(super) fn validate_prompt_enhancement_fields(
    model_id: &str,
    advanced: &JsonObject,
) -> Result<bool, ApiError> {
    let max_tokens = sceneworks_worker::prompt_enhancement_max_tokens(model_id);
    if advanced.contains_key(PROMPT_ENHANCEMENT_FACT_KEY) {
        return Err(ApiError::bad_request(format!(
            "advanced.{PROMPT_ENHANCEMENT_FACT_KEY} is worker-owned and cannot be supplied by a client"
        )));
    }
    let enabled = match advanced.get("enhancePrompt") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt must be a boolean",
            ));
        }
    };
    if let Some(value) = advanced.get("enhanceTemperature") {
        let temperature = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                ApiError::bad_request("advanced.enhanceTemperature must be a finite number")
            })?;
        if !(0.0..=PROMPT_ENHANCE_MAX_TEMPERATURE).contains(&temperature) {
            return Err(ApiError::bad_request(format!(
                "advanced.enhanceTemperature must be between 0 and {PROMPT_ENHANCE_MAX_TEMPERATURE}"
            )));
        }
        if !enabled {
            return Err(ApiError::bad_request(
                "advanced.enhanceTemperature requires advanced.enhancePrompt=true",
            ));
        }
    }
    if let Some(value) = advanced.get("enhanceMaxTokens") {
        let tokens = value
            .as_u64()
            .ok_or_else(|| ApiError::bad_request("advanced.enhanceMaxTokens must be an integer"))?;
        if !(1..=max_tokens).contains(&tokens) {
            return Err(ApiError::bad_request(format!(
                "advanced.enhanceMaxTokens must be between 1 and {max_tokens}"
            )));
        }
        if !enabled {
            return Err(ApiError::bad_request(
                "advanced.enhanceMaxTokens requires advanced.enhancePrompt=true",
            ));
        }
    }
    Ok(enabled)
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(super) fn prompt_enhancement_payload_string(payload: &JsonObject, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(super) fn prompt_enhancement_payload_references(payload: &JsonObject) -> bool {
    prompt_enhancement_payload_string(payload, "referenceAssetId")
        || payload
            .get("referenceAssetIds")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str().is_some_and(|value| !value.trim().is_empty()))
            })
}

pub(super) fn validate_prompt_enhancement_route(payload: &JsonObject) -> Result<(), ApiError> {
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("text_to_image");
    // The API may be a CPU-only Docker service serving a native worker. Its own compiled
    // inference backends are not evidence of that worker's capabilities. FLUX 3 validates shape
    // here; the selected worker independently requires an enabled native backend and installed PUP.
    if matches!(
        payload.get("model").and_then(Value::as_str),
        Some("flux3_klein" | "flux3_klein_base")
    ) {
        // Generation references are AE-only conditioning; the existing separate PUP consumes the
        // ORIGINAL ordered images. The same pure shape policy validates both modes before enqueue.
        sceneworks_core::image_request::flux3::parse(payload).map_err(ApiError::bad_request)?;
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    if !matches!(
        mode,
        "text_to_image" | "edit_image" | "character_image" | "style_variations"
    ) {
        return Err(ApiError::bad_request(format!(
            "advanced.enhancePrompt on MLX does not support image mode {mode}"
        )));
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if !matches!(mode, "text_to_image" | "edit_image") {
        return Err(ApiError::bad_request(format!(
            "advanced.enhancePrompt on Candle supports only text_to_image and edit_image; mode {mode} is unsupported"
        )));
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = payload;
        Err(ApiError::bad_request(
            "advanced.enhancePrompt requires a native MLX or Candle image backend",
        ))
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let has_references = prompt_enhancement_payload_references(payload);
        let has_edit_input =
            has_references || prompt_enhancement_payload_string(payload, "sourceAssetId");
        if mode == "text_to_image" && has_edit_input {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt text_to_image cannot include source or reference image assets",
            ));
        }
        if mode == "edit_image" && !has_edit_input {
            return Err(ApiError::bad_request(
                "advanced.enhancePrompt edit_image requires a source or reference image asset",
            ));
        }
        if matches!(mode, "character_image" | "style_variations") && !has_references {
            return Err(ApiError::bad_request(format!(
                "advanced.enhancePrompt {mode} requires a reference image asset"
            )));
        }
        Ok(())
    }
}

/// Validate the canonical, post-preset image payload. Enhancement is deliberately scoped to the
/// actual native backend route: MLX owns base/edit/character plus the defensive legacy style alias,
/// while Candle owns only base and its bespoke `edit_image` lane. FLUX 3 Image Klein uses the same
/// optional upsampler for text or ordered reference conditioning; FLUX.2-Klein remains unsupported. Neither
/// grants enhancement to strict control or backendless builds,
/// or a reference-bearing mode that would fall through to a plain base render. Keeping this check
/// on the final payload also covers presets and retry/duplicate's shallow-merged canonical payload.
pub(super) fn validate_prompt_enhancement_payload(payload: &JsonObject) -> Result<(), ApiError> {
    let empty = JsonObject::new();
    let advanced = payload
        .get("advanced")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !validate_prompt_enhancement_fields(model, advanced)? {
        return Ok(());
    }
    if !matches!(model, "flux2_dev" | "flux3_klein" | "flux3_klein_base") {
        return Err(ApiError::bad_request(
            "advanced.enhancePrompt is supported only by FLUX.2-dev and FLUX 3 Image Klein; FLUX.2-Klein and other models reject it",
        ));
    }
    let strict_control = advanced
        .get("poses")
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty())
        || advanced.contains_key("controlWeights")
        || advanced.contains_key("controlImage")
        || advanced.contains_key("controlMode");
    if strict_control {
        return Err(ApiError::bad_request(
            "advanced.enhancePrompt cannot be combined with strict control",
        ));
    }
    validate_prompt_enhancement_route(payload)
}

/// Reject a `model` id that is not a safe single path component (F-003 / sc-11159).
///
/// The id flows verbatim from the untrusted job payload into the worker's asset
/// filename — `write_image_asset` / `write_upscaled_asset` / `VideoPlan::new` each
/// `format!(".._{model}_..")` a project-relative path, then `create_dir_all` +
/// atomic-rename the file into place. A `model` containing `../`, `..\`, or an
/// absolute path would therefore traverse out of the project dir and hand a remote
/// caller an arbitrary-write primitive. The worker now slugifies the id as a last
/// line of defense, but the API is the first gate and rejects such ids outright,
/// mirroring the worker's `safe_weight_filename` posture for payload-supplied
/// filename components.
///
/// SCOPE (sc-11159): this enforces *path-safety only*, NOT catalog membership. The
/// worker's stub lane deliberately serves uncatalogued model ids for dev/testing,
/// and the API test harness (which does not seed the shipped model manifests)
/// routinely enqueues real-but-uncatalogued ids and expects success, so a catalog
/// rejection here would break that legitimate lane. Path-safety alone fully closes
/// the traversal/arbitrary-write vulnerability: an uncatalogued-but-path-safe id
/// can never escape the project dir.
pub(super) fn validate_model_id(model: &str) -> Result<(), ApiError> {
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

pub(super) fn validate_image_job(payload: &ImageJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    validate_model_id(&payload.model)?;
    if (payload.moss_image_advanced.is_some()
        || payload
            .advanced
            .contains_key(sceneworks_core::moss_image_advanced::KEY))
        && (payload.mode != "edit_image" || !sceneworks_core::moss_image::is_moss(&payload.model))
    {
        return Err(ApiError::bad_request(
            "mossImageAdvanced tensor/output controls require explicit MOSS edit_image mode",
        ));
    }
    let positive_tensor = payload.mode == "edit_image"
        && sceneworks_core::moss_image::is_moss(&payload.model)
        && (payload
            .moss_image_advanced
            .as_ref()
            .is_some_and(|a| a.positive.is_some())
            || payload
                .advanced
                .get(sceneworks_core::moss_image_advanced::KEY)
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
    validate_prompt_extras(&payload.negative_prompt, &payload.advanced)?;
    validate_prompt_enhancement_fields(&payload.model, &payload.advanced)?;
    // sc-26241: a FLUX 3 request names exactly one of its packages, and the publisher FP8R tier
    // exists only for the model that ships it. Which backend and worker can load it is routing's.
    if let Some(reason) =
        sceneworks_core::image_request::image_quant_tier_refusal(&payload.model, &payload.advanced)
    {
        return Err(ApiError::bad_request(reason));
    }
    validate_image_pose_count(&payload.advanced)?;
    if payload.loras.len() > sceneworks_core::lora_family::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            sceneworks_core::lora_family::MAX_JOB_LORAS
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
        if let Some(error) = sceneworks_core::image_request::image_count_error(count) {
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
        && (sceneworks_core::image_request::flux3::is_model(&payload.model)
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

/// Bound strict-pose fan-out at the image-job creation boundary. Each `advanced.poses` entry
/// renders one image, so this is an output-count contract rather than a JSON-size guard.
///
/// Existing stored jobs are immutable and remain readable. Retry/duplicate pass their merged
/// payload through this same helper in `jobs.rs`, so a legacy over-limit job must be reduced before
/// it can create new work instead of silently reproducing the old unbounded behavior.
pub(super) fn validate_image_pose_count(advanced: &JsonObject) -> Result<(), ApiError> {
    let Some(poses) = advanced.get("poses").and_then(Value::as_array) else {
        return Ok(());
    };
    if poses.len() > sceneworks_core::image_request::MAX_JOB_POSES {
        return Err(ApiError::bad_request(format!(
            "advanced.poses must contain at most {} entries; each pose renders one image",
            sceneworks_core::image_request::MAX_JOB_POSES
        )));
    }
    Ok(())
}

pub(super) fn validate_character_test_job(payload: &CharacterTestRequest) -> Result<(), ApiError> {
    if payload.prompt.is_empty() || payload.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::bad_request(format!(
            "prompt must be between 1 and {MAX_PROMPT_CHARS} characters"
        )));
    }
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
}

/// Every `mode` a `POST /api/v1/video/jobs` request may name — the enqueue allow-list, and a
/// SEPARATE reachability gate from the catalog: a mode absent HERE 400s with "Unsupported video
/// mode" no matter what the model's manifest `capabilities` declare, what `VIDEO_UI_MODES` offers,
/// or what the worker can render.
///
/// It is a named `const` rather than the inline array literal it was so a test can enumerate the
/// REAL list. A guard that retypes the modes asserts against its own copy and stays green while
/// this list drifts — the false-green shape that let GH #2074 ship. `every_declared_video_capability_is_submittable`
/// (tests/jobs.rs) reads this constant and the shipped manifest, so a new family's capability that
/// is not admitted here is RED at the source.
///
/// ⚠️ Adding a family means checking SIX surfaces, not one (sc-17159): manifest `capabilities`,
/// [`sceneworks_core::jobs_store::routing`]'s `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible` +
/// `VIDEO_MODEL_CAPS`, the candle claim gate, THIS list plus the per-mode required-asset `match`
/// below it, and the worker's dispatch arm.
pub(crate) use sceneworks_core::video_request::VIDEO_JOB_MODES;

pub(super) fn validate_video_job(payload: &VideoJobRequest) -> Result<(), ApiError> {
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
    if payload.loras.len() > sceneworks_core::lora_family::MAX_JOB_LORAS {
        return Err(ApiError::bad_request(format!(
            "loras must contain at most {} entries",
            sceneworks_core::lora_family::MAX_JOB_LORAS
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
        // The rule itself is `sceneworks_core::video_request::classify_reference_set`, which the
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

/// Payload-sanity validation for a `POST /api/v1/audio/jobs` request (sc-13404), the audio analogue
/// of [`validate_video_job`]. Bounds the script prompt + the `advanced` bag exactly as the image/
/// video validators do; the model's own advertised audio surface (voices / languages / max
/// duration) is enforced by the shared gen-core validation floor at generate time, so a *named*
/// duration is only bounded to the sane blanket here (the model's `audio.maxDurationSecs` is the
/// real cap the worker applies). Voice/language are not allow-listed here — an unknown value is
/// rejected by the generator's `validate`, which owns the per-model surface.
pub(super) fn validate_audio_job(payload: &AudioJobRequest) -> Result<(), ApiError> {
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
    // Diffusion-audio sampling knobs (Sound FX / MOSS-SoundEffect, sc-13409). Bounded to a blanket
    // sane range here — the same "API blanket vs. model's real cap" split the duration uses: the
    // generator's `validate` owns the per-model guidance range (MOSS: 1.0..=20.0) and step ceiling
    // (MOSS: 1000), so this only rejects nonsense (NaN / non-positive / absurdly large) up front.
    if let Some(guidance) = payload.guidance {
        if !guidance.is_finite() || !(0.0..=100.0).contains(&guidance) {
            return Err(ApiError::bad_request(
                "guidance (CFG scale) must be between 0 and 100",
            ));
        }
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

/// The ICL reference modes a YuE `_icl` checkpoint accepts (sc-19384): one mixed clip, or a
/// vocal + instrumental pair.
pub(super) const AUDIO_ICL_MODES: &[&str] = &["single", "dual"];
/// The weight tiers an audio model can ship as physical per-tier downloads (sc-19384).
pub(super) const AUDIO_QUANT_TIERS: &[&str] = &["bf16", "q8", "q4"];
/// Upstream YuE's `prompt_end_time` default: the ICL window end when only a start is given.
pub(super) const AUDIO_ICL_DEFAULT_END_SECS: f32 = 30.0;
/// The output limiters a segmented-song model applies to its stems (YuE `save_audio`, sc-19384).
pub(super) const AUDIO_OUTPUT_LIMITERS: &[&str] = &["clamp", "rescale"];

/// Payload-shape floor for the segmented-song / ICL / tier fields (YuE, sc-19384), before the model
/// is known: numeric knobs finite and in a sane blanket range, guidance on/off coherent with the
/// scale, the ICL mode a known token with exactly its own asset ids, and the reference window
/// well-ordered. [`validate_audio_job_for_model`] then gates each field on the model's declared
/// capability once the manifest entry is resolved.
pub(super) fn validate_audio_song_fields(payload: &AudioJobRequest) -> Result<(), ApiError> {
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

/// Gate the segmented-song / ICL / tier fields on the resolved model's DECLARED audio capabilities
/// (sc-19384) — the manifest mirrors the engine's `Capabilities` flags, so a request the engine
/// would refuse is a 400 here instead of a failed job. In particular an ICL mode is reachable only
/// on an in-context-learning checkpoint (`audio.supportsSegmentedLyrics` + `ReferenceAudio`
/// conditioning — the YuE `_icl` variants); such a checkpoint may also run WITHOUT one, as a plain
/// prompt run, as upstream allows. A segmented-lyrics
/// model also refuses the knobs it does not read (it sings lyrics to genre tags; length follows the
/// lyrics and `segments`) rather than silently dropping them.
pub(crate) fn validate_audio_job_for_model(
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
        // A segmented-song model reads a scale in 0..=1 as "guidance off", so an explicit ON scale
        // must be above 1 — turn guidance off with guidanceEnabled: false instead.
        if payload.guidance_enabled != Some(false) && payload.guidance.is_some_and(|g| g <= 1.0) {
            return Err(ApiError::bad_request(format!(
                "{model}: a guidance scale must be greater than 1 (set guidanceEnabled: false to \
                 turn guidance off)"
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

/// The edit operations the audio route accepts up front (the union across audio models — each model's
/// advertised `audio.editModes` is the real gate the generator's `validate` applies). Matches the
/// [`gen_core::AudioEditMode`] variant set.
pub(super) const AUDIO_EDIT_MODES: &[&str] = &["inpaint", "repaint", "extend", "cover"];

/// Sanity-bound the extend/edit source-band fields on an [`AudioJobRequest`] (Conditioning::AudioEdit,
/// sc-13410). This is the API blanket floor: the source id + edit mode must be a well-formed pair, the
/// mode a known token, the region seconds finite/ordered, and the strength a 0..=1 weight. The
/// per-model gates (mode ∈ advertised `audio.editModes`, region inside the clip, 48 kHz source) belong
/// to the worker/provider, which knows the model surface and the loaded clip.
pub(super) fn validate_audio_edit_fields(payload: &AudioJobRequest) -> Result<(), ApiError> {
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

/// Upper bound for image width/height. A backstop only — per-model resolution is
/// governed by manifest `limits.resolutions` + the UI. Covers SenseNova-U1's
/// largest trained bucket (3456) with headroom; video uses its own lower cap.
pub(super) const MAX_IMAGE_DIMENSION: u32 = 4096;

/// Payload-sanity FLOOR for a named image width/height, before the model is known (sc-24113).
///
/// 16 because MOSS-Image accepts 16 px per side (Qwen-Image 2.1 starts at 32) and this
/// check runs before preset/model resolution. It is the outer bound only — see
/// the shared core image-request boundary for the per-model floor that still applies.
pub(super) const SHAPE_MIN_IMAGE_DIMENSION: u32 = 16;

/// Upper bound for video width/height — a lower backstop than images, matching
/// the cap enforced when validating a video job request.
pub(super) const MAX_VIDEO_DIMENSION: u32 = 1920;
// ---------------------------------------------------------------------------
// Reference-media payload-sanity blankets (sc-17160).
//
// These three are the OUTER bound — "no video model in the product accepts more than
// this" — and play exactly the role the `1..=30` duration and `1..=60` fps blankets do
// a few lines up in `validate_video_job`: they run BEFORE the model is known (a recipe
// preset can still replace it — sc-12300), so they cannot be the per-model answer.
//
// The BINDING per-model cap is `sceneworks_core::video_request::reference_limit_error`,
// enforced in `create_video_job` against the resolved manifest entry, and mirrored in the
// worker's `video_preflight`. Its defaults are 8 images / 8 clips / 0 audio / no combined
// ceiling, so raising the image blanket from 8 to 9 here does NOT hand a 9th reference to
// any already-shipped model — bernini and every other family still refuse it, one layer
// down, on `limits.maxReferenceAssets`.
//
// The COMBINED cap has no blanket at all, only a per-model declaration, for the same reason
// in the opposite direction: see the note where the per-list checks end.
// ---------------------------------------------------------------------------

/// 9 — MiniMax-H3 Ref2VA's image-reference ceiling, the largest of any shipped video model
/// (every other family stops at [`sceneworks_core::video_request::DEFAULT_MAX_REFERENCE_ASSETS`]).
pub(super) const MAX_VIDEO_REFERENCE_ASSET_IDS: usize = 9;
pub(super) const MAX_VIDEO_SOURCE_CLIP_ASSET_IDS: usize = 8;
/// 3 — Ref2VA's audio-reference ceiling. No model declares more; models that declare
/// nothing take none at all (`DEFAULT_MAX_REFERENCE_AUDIO_ASSETS` is 0).
pub(super) const MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS: usize = 3;

/// Validate the exact model/mode spelling and reference array every video creation boundary will
/// persist. The typed submit route and retry/duplicate all call this after preset/patch merging and
/// current manifest resolution, so replay cannot drift into the worker parser's deliberately
/// tolerant behavior (which trims strings and drops blank/non-string list entries for legacy reads).
///
/// This helper is validation-only: it never trims, filters, sorts, or truncates, preserving every
/// accepted id byte-for-byte and in caller order. SCAIL-2 additionally consumes strict ordered
/// Reference/Mask pairs, has six source positions, and may expose multiple characters only once the
/// current server-owned entry says the paired inference descriptor is installed.
pub(crate) fn validate_video_reference_asset_ids_payload(
    payload: &JsonObject,
    model_manifest_entry: &Value,
) -> Result<(), ApiError> {
    if payload.get("mode").and_then(Value::as_str) == Some("image_to_video")
        && !payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty())
    {
        use sceneworks_core::minimax_h3_import::{
            entry_is_imported_minimax_h3, MiniMaxH3Partition,
        };
        let imported_partition = model_manifest_entry
            .as_object()
            .filter(|entry| entry_is_imported_minimax_h3(entry))
            .and_then(|entry| MiniMaxH3Partition::from_entry(entry).ok());
        if !sceneworks_core::h3_keyframes::accepts_last_only_resolved_image_to_video(
            model_manifest_entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            "image_to_video",
            payload
                .get("lastFrameAssetId")
                .and_then(Value::as_str)
                .is_some_and(|id| !id.trim().is_empty()),
            imported_partition.map(|partition| partition.as_str()),
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
        && reference_count > sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
    {
        return Err(ApiError::bad_request(format!(
            "SCAIL-2 Animate Character supports at most {} reference characters.",
            sceneworks_core::video_request::MAX_SCAIL2_REFERENCE_CHARACTERS
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

pub(super) fn validate_dimension(
    value: u32,
    field: &'static str,
    max: u32,
) -> Result<(), ApiError> {
    validate_dimension_in(value, field, 256, max)
}

/// [`validate_dimension`] with an explicit floor (sc-24113).
///
/// Every caller but the image-generation shape check passes the historical 256 through the wrapper
/// above and is byte-identical. The image route passes [`SHAPE_MIN_IMAGE_DIMENSION`] because its
/// real floor is per-model and is applied once the manifest entry is resolved.
pub(super) fn validate_dimension_in(
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

pub(super) fn required_string_field<'a>(
    payload: &'a Value,
    field: &str,
) -> Result<&'a str, ApiError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))
}

pub(super) fn optional_f64_field(payload: &Value, field: &str) -> Option<f64> {
    payload.get(field).and_then(Value::as_f64)
}

pub(super) fn required_finite_f64_field(payload: &Value, field: &str) -> Result<f64, ApiError> {
    let value = optional_f64_field(payload, field)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))?;
    if !value.is_finite() {
        return Err(ApiError::bad_request(format!(
            "Invalid numeric value for {field}"
        )));
    }
    Ok(value)
}
