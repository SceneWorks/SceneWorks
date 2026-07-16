use super::*;

pub(crate) async fn create_image_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ImageJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_image_job(&payload)?;
    let job_type = if payload.mode == "edit_image" {
        JobType::ImageEdit
    } else {
        JobType::ImageGenerate
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    // One request-scoped catalog snapshot threaded through preset expansion + LoRA
    // validation so the per-model/per-LoRA filesystem install-state probes run once per
    // job-create instead of 2–3× (sc-8819, F-017).
    let catalogs = JobCatalogSnapshot::default();
    apply_recipe_preset_to_image_payload(&state, &payload, &mut job_payload, &catalogs).await?;
    // Ideogram 4 headless/API parity (sc-6519, fully async per sc-9120): a plain-text Ideogram 4 job
    // needs its prompt expanded into a rich JSON caption via the magic-prompt utility model — the same
    // separate prompt_refine job the web runs (sc-6501) — or it stochastically renders the safety-filter
    // placeholder. Rather than block the POST on that expansion, we detect the need here, create the
    // image job IMMEDIATELY in a non-claimable `pending_caption` status, and let a background task run
    // the expansion and rewrite the prompt before promoting the job to `queued`. A no-op (→ normal
    // `queued` create) for every other model, an already-structured caption, or an image-conditioned
    // edit. The worker's format-guard + reseed net remains the fallback if the expansion is unavailable.
    let caption_request = crate::ideogram::caption_request_for_ideogram(&job_payload);
    // Keyed off the POST-preset job_payload["model"], NOT the DTO's payload.model — see the
    // matching note in create_video_job (sc-12300). apply_recipe_preset_to_image_payload above
    // may have replaced the model with the preset's own when the caller omitted one, which
    // leaves payload.model stale and would resolve the DEFAULT model's entry.
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(payload.model.as_str())
        .to_owned();
    let model_manifest_entry = resolve_model_manifest_entry(&state, &model_id).await?;
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    validate_job_lora_compatibility_with(
        &state,
        Some(&payload.project_id),
        &mut job_payload,
        false,
        Some(&catalogs),
    )
    .await?;
    // Resolve a selected control overlay id → its installed `.safetensors` path (sc-10165, B4), so a
    // ControlNet the user picked in the Studio's ControlPanel is loadable by the worker strict-control
    // lane. A no-op unless `advanced.controlWeights.overlayId` is set.
    crate::control_overlays::resolve_control_overlay_selection(
        &state,
        Some(&payload.project_id),
        &mut job_payload,
    )
    .await?;
    if payload.seed.is_none() {
        let count = job_payload
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(payload.count);
        job_payload.insert("seeds".to_owned(), random_image_seeds(count));
    }
    // Create in `pending_caption` when an async caption is pending, else the default `queued`. The
    // POST returns 201 immediately either way — it never waits on the expansion (sc-9120).
    let initial_status = caption_request.as_ref().map(|_| JobStatus::PendingCaption);
    let job = create_generation_job_with_status(
        state.clone(),
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
        initial_status,
    )
    .await?;
    // Kick off the async expansion + promotion AFTER the job row exists. The watcher always leaves the
    // job claimable (rewritten to the rich caption, or degraded to the original prompt), and recovers to
    // `queued` on an API restart if it is lost mid-flight, so the job can never sit un-claimable.
    if let Some(caption_request) = caption_request {
        crate::ideogram::spawn_ideogram_caption_watcher(state, job.id.clone(), caption_request);
    }
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn create_vqa_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VqaJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_vqa_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageVqa,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) fn validate_vqa_job(payload: &VqaJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.source_asset_id.trim().is_empty() {
        return Err(ApiError::bad_request("sourceAssetId is required"));
    }
    let question = payload.question.trim();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "question must be between 1 and 4000 characters",
        ));
    }
    if !(16..=2048).contains(&payload.max_new_tokens) {
        return Err(ApiError::bad_request(
            "maxNewTokens must be between 16 and 2048",
        ));
    }
    Ok(())
}

pub(crate) async fn create_interleave_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<InterleaveJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_interleave_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageInterleave,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) fn validate_interleave_job(payload: &InterleaveJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.trim().is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
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

/// Request-scoped, lazily-memoized snapshot of the model and LoRA catalogs (sc-8819,
/// F-017). A single preset-backed `POST /image/jobs` (or `/video/jobs`) fans out into
/// `recipe_preset_catalog`, `merge_preset_loras_into_payload`, and
/// `validate_job_lora_compatibility`, each of which formerly re-assembled
/// `model_catalog`/`lora_catalog` from scratch — re-running the per-model install-state
/// probes (recursive HF-cache walks, `model_is_installed`, `mlx_catalog_status`) 2–3×
/// over the whole catalog per submit. Threading one snapshot through those seams makes
/// each catalog's filesystem probing run exactly once per job-create.
///
/// This is deliberately request-scoped rather than a shared TTL cache: the snapshot lives
/// only for the duration of one job-create, so there is no staleness window and the
/// catalog contents seen by preset expansion and by LoRA validation are guaranteed
/// identical. It memoizes per `project_id` (constant within a single job-create), so both
/// the `project_id`-scoped and the `None` (no-project) catalog reads are covered.
#[derive(Default)]
pub(crate) struct JobCatalogSnapshot {
    models: tokio::sync::OnceCell<Vec<Value>>,
    loras_by_project: tokio::sync::Mutex<HashMap<Option<String>, Arc<Vec<Value>>>>,
}

impl JobCatalogSnapshot {
    /// The model catalog, built once per request and reused thereafter. Identical output
    /// to a direct `model_catalog(state)` call.
    pub(crate) async fn models(&self, state: &AppState) -> Result<&[Value], ApiError> {
        let models = self
            .models
            .get_or_try_init(|| async { model_catalog(state).await })
            .await?;
        Ok(models.as_slice())
    }

    /// The LoRA catalog for `project_id`, built once per (request, project) and reused
    /// thereafter. Identical output to a direct `lora_catalog(state, project_id)` call.
    pub(crate) async fn loras(
        &self,
        state: &AppState,
        project_id: Option<&str>,
    ) -> Result<Arc<Vec<Value>>, ApiError> {
        let key = project_id.map(str::to_owned);
        let mut guard = self.loras_by_project.lock().await;
        if let Some(existing) = guard.get(&key) {
            return Ok(existing.clone());
        }
        let loras = Arc::new(lora_catalog(state, project_id).await?);
        guard.insert(key, loras.clone());
        Ok(loras)
    }
}

pub(crate) async fn apply_recipe_preset_to_image_payload(
    state: &AppState,
    payload: &ImageJobRequest,
    job_payload: &mut JsonObject,
    snapshot: &JobCatalogSnapshot,
) -> Result<(), ApiError> {
    let Some(preset_id) = payload.recipe_preset_id.as_deref() else {
        return Ok(());
    };
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    let presets =
        recipe_preset_catalog_with(state, Some(&payload.project_id), Some(snapshot)).await?;
    let preset = presets
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(preset_id))
        .ok_or_else(|| ApiError::bad_request("Recipe preset not found"))?;

    // Submitting a job with a preset is the strong "used" signal, and the one place the
    // backend already sees the resolved preset id — stamp lastUsedAt now (sc-10520).
    stamp_recipe_preset_used(state, preset_id).await;

    let expanded_prompt = if payload.preset_prompt_resolved_client_side.unwrap_or(false) {
        // The studio already composed the full preset-stack prompt client-side; take it
        // verbatim so we don't double-fold this preset's prefix/suffix (epic 11949).
        payload.prompt.clone()
    } else {
        preset_prompt(&payload.prompt, preset)
    };
    job_payload.insert("prompt".to_owned(), Value::String(expanded_prompt));
    if payload.model == default_image_model() {
        if let Some(model) = preset
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
    }
    // Render defaults (count/resolution/negativePrompt) are intentionally NOT
    // applied here — the studio seeds those into the form from the preset and the
    // user can override them, so the submitted values are authoritative.
    job_payload.insert(
        "stylePreset".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    merge_preset_loras_into_payload(
        state,
        &payload.project_id,
        preset_id,
        preset,
        job_payload,
        Some(snapshot),
        payload.preset_loras_resolved_client_side.unwrap_or(false),
    )
    .await
}

/// Prepend a preset's declared LoRAs to whatever LoRAs the client already sent,
/// skipping ids that are already present. Records ids the catalog can't resolve
/// under advanced.presetMissingLoras and stamps advanced.recipePresetId. Shared
/// by the image and video job paths so preset-LoRA semantics stay identical.
///
/// When `client_resolved` is set (the web studio seeds a selected preset's LoRAs
/// straight into `loras` and sends presetLorasResolvedClientSide), the client is
/// authoritative for the preset's LoRAs — including weight edits and removals — so
/// the merge is skipped and `loras` is left exactly as sent; only the advanced
/// recipePresetId stamp is applied. Headless/API clients that send only
/// recipePresetId leave the flag unset and get the server-side merge.
pub(crate) async fn merge_preset_loras_into_payload(
    state: &AppState,
    project_id: &str,
    preset_id: &str,
    preset: &Value,
    job_payload: &mut JsonObject,
    snapshot: Option<&JobCatalogSnapshot>,
    client_resolved: bool,
) -> Result<(), ApiError> {
    // Stamp the resolved preset id onto advanced regardless of who owns the LoRAs.
    let advanced = job_payload
        .entry("advanced".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !advanced.is_object() {
        *advanced = Value::Object(JsonObject::new());
    }
    let advanced = advanced
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?;
    advanced.insert(
        "recipePresetId".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    advanced.remove("recipePresetName");
    advanced.remove("recipePresetPrompt");

    // Client owns the preset LoRAs — leave `loras` untouched. There's no server-resolved
    // "missing" set in this path (the studio only seeds LoRAs it can actually apply), so
    // clear any stale marker and return.
    if client_resolved {
        advanced.remove("presetMissingLoras");
        return Ok(());
    }

    // Reuse the request-scoped LoRA catalog snapshot when threaded (sc-8819), else build
    // fresh. Both paths see identical catalog contents.
    let loras = match snapshot {
        Some(snapshot) => snapshot.loras(state, Some(project_id)).await?,
        None => Arc::new(lora_catalog(state, Some(project_id)).await?),
    };
    let existing_lora_ids = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut seen_lora_ids = existing_lora_ids;
    let mut preset_loras = Vec::new();
    let mut missing_lora_ids = Vec::new();
    for preset_lora in recipe_preset_loras(preset) {
        let Some(lora_id) = preset_lora_id(&preset_lora) else {
            continue;
        };
        let Some(lora) = loras
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(lora_id))
        else {
            missing_lora_ids.push(Value::String(lora_id.to_owned()));
            continue;
        };
        if seen_lora_ids.iter().any(|seen_id| seen_id == lora_id) {
            continue;
        }
        preset_loras.push(serialize_preset_lora(lora, &preset_lora, lora_id));
        seen_lora_ids.push(lora_id.to_owned());
    }

    // Re-borrow advanced (the stamp borrow above has ended) to record any unresolved ids.
    let advanced = job_payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?;
    if missing_lora_ids.is_empty() {
        advanced.remove("presetMissingLoras");
    } else {
        advanced.insert(
            "presetMissingLoras".to_owned(),
            Value::Array(missing_lora_ids),
        );
    }

    let user_loras = job_payload
        .remove("loras")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    preset_loras.extend(user_loras);
    job_payload.insert("loras".to_owned(), Value::Array(preset_loras));
    Ok(())
}

pub(crate) fn parse_recipe_preset_resolution(value: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(ApiError::bad_request(
            "Recipe preset resolution must use WIDTHxHEIGHT",
        ));
    };
    let width = width
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset width must be a number"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset height must be a number"))?;
    Ok((width, height))
}

/// Server-side expansion of a video job's recipe preset, mirroring
/// apply_recipe_preset_to_image_payload: the client sends the raw prompt plus
/// recipePresetId and the server folds in the preset's prompt prefix/suffix,
/// model, and LoRAs. Render defaults (duration/fps/resolution/quality/
/// negativePrompt) are intentionally NOT applied here — the studio seeds those
/// into the form from the preset and the user can override them, so the
/// submitted values are authoritative.
pub(crate) async fn apply_recipe_preset_to_video_payload(
    state: &AppState,
    payload: &VideoJobRequest,
    job_payload: &mut JsonObject,
    snapshot: &JobCatalogSnapshot,
) -> Result<(), ApiError> {
    let Some(preset_id) = payload.recipe_preset_id.as_deref() else {
        return Ok(());
    };
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    let presets =
        recipe_preset_catalog_with(state, Some(&payload.project_id), Some(snapshot)).await?;
    let preset = presets
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(preset_id))
        .ok_or_else(|| ApiError::bad_request("Recipe preset not found"))?;

    // Submitting a job with a preset is the strong "used" signal, and the one place the
    // backend already sees the resolved preset id — stamp lastUsedAt now (sc-10520).
    stamp_recipe_preset_used(state, preset_id).await;

    let expanded_prompt = if payload.preset_prompt_resolved_client_side.unwrap_or(false) {
        // The studio already composed the full preset-stack prompt client-side; take it
        // verbatim so we don't double-fold this preset's prefix/suffix (epic 11949).
        payload.prompt.clone()
    } else {
        preset_prompt(&payload.prompt, preset)
    };
    job_payload.insert("prompt".to_owned(), Value::String(expanded_prompt));
    if payload.model == default_video_model() {
        if let Some(model) = preset
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
    }
    merge_preset_loras_into_payload(
        state,
        &payload.project_id,
        preset_id,
        preset,
        job_payload,
        Some(snapshot),
        payload.preset_loras_resolved_client_side.unwrap_or(false),
    )
    .await
}

pub(crate) async fn create_video_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VideoJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_video_job(&payload)?;
    let job_type = match payload.mode.as_str() {
        "extend_clip" => JobType::VideoExtend,
        "video_bridge" => JobType::VideoBridge,
        "replace_person" => JobType::PersonReplace,
        _ => JobType::VideoGenerate,
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    // One request-scoped catalog snapshot threaded through preset expansion + LoRA
    // validation so the per-model/per-LoRA filesystem install-state probes run once per
    // job-create instead of 2–3× (sc-8819, F-017).
    let catalogs = JobCatalogSnapshot::default();
    apply_recipe_preset_to_video_payload(&state, &payload, &mut job_payload, &catalogs).await?;
    // Resolve the model manifest entry here so the GPU worker never re-parses
    // builtin/user.models.jsonc itself — Rust owns manifest parsing/merging
    // (story 1653). An unknown model resolves to {}, matching the worker's
    // existing fallback to the model's default repo.
    //
    // Keyed off the POST-preset job_payload["model"], NOT the DTO's payload.model:
    // apply_recipe_preset_to_video_payload above may have replaced the model with the
    // preset's own when the caller omitted one, which leaves payload.model stale (sc-12300).
    // Resolving from the stale id enqueued the overridden model id alongside the DEFAULT
    // model's entry — wrong repo/paths/quant, and wrong `limits`, which normalized_dimensions
    // honors for the dimension floor (sc-11993). Mirrors how validate_job_lora_compatibility_with
    // below already reads the model from job_payload.
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(payload.model.as_str())
        .to_owned();
    let model_manifest_entry = resolve_model_manifest_entry(&state, &model_id).await?;
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    validate_job_lora_compatibility_with(
        &state,
        Some(&payload.project_id),
        &mut job_payload,
        false,
        Some(&catalogs),
    )
    .await?;
    let job = create_generation_job(
        state,
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// The typed route that owns `job_type`, or `None` for every job type the generic
/// `POST /api/v1/jobs` legitimately serves (`image_upscale`, `image_detail`,
/// `model_download`, …). The guard list for `create_job` (sc-12305).
///
/// This is exactly the set of job types produced by [`create_image_job`] /
/// [`create_video_job`] — the only two routes in the tree that resolve a model's merged
/// manifest entry and inject it as `modelManifestEntry`. Enqueued raw through the generic
/// route, such a job carries no entry at all: the worker falls back to the model's default
/// repo/knobs, and on the video lane `VideoRequest::from_payload` misses
/// `limits.requiresDimensionsMultipleOf` and falls back to ÷32 — silently rendering
/// Mochi's native (and only trained) 848x480 as 832x480, a rewrite the engine's own ÷16
/// check cannot catch because `832 % 16 == 0`. The video geometry is the *silent* failure
/// (see `mochi_without_manifest_entry_silently_loses_its_native_bucket`); the image lane
/// reads the same entry for its family/repo knobs, where a miss surfaces sooner.
///
/// Rejecting is deliberate over resolving the entry here. The manifest entry is one of
/// several things these routes do: they also validate the request (`validate_video_job` —
/// projectId, model id, prompt bounds, mode allowlist), expand recipe presets (which can
/// *replace* the model — sc-12300), check LoRA compatibility, and map `mode` to the job
/// type. Filling in only the entry would leave a path that renders at the right geometry
/// while skipping every one of those, which is a subtler trap than the one being closed.
/// One door per generation job type.
///
/// Keep in step with [`create_image_job`] / [`create_video_job`]: a new generation route
/// that injects a `modelManifestEntry` belongs here too. (`image_vqa` / `image_interleave`
/// have typed routes but resolve no manifest entry, so they are deliberately absent.)
pub(crate) fn typed_generation_route(job_type: &JobType) -> Option<&'static str> {
    match job_type {
        JobType::ImageGenerate | JobType::ImageEdit => Some("/api/v1/image/jobs"),
        JobType::VideoGenerate
        | JobType::VideoExtend
        | JobType::VideoBridge
        | JobType::PersonReplace => Some("/api/v1/video/jobs"),
        _ => None,
    }
}
