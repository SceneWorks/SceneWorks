use super::*;

/// The single backend this API instance can actually enqueue for — the one authority every
/// enqueue-time capability gate must key off.
///
/// macOS runs the MLX worker, *unless* the operator forced the Candle one
/// (`SCENEWORKS_CANDLE_REQUIRED` → [`Settings::candle_required`]); every other platform is
/// Candle-only. A bare `cfg!(target_os = "macos")` is wrong in precisely the mode the setting
/// exists for: the gate would validate the request against MLX's capability lists while the job
/// executes on Candle, so an MLX-only choice is admitted and then fails on the worker, and a
/// Candle-valid choice is 400'd (sc-18420). One derivation, one place.
///
/// ## Scope: enqueue gates only — NOT the catalog advertisement lanes
///
/// `models.rs`'s `apply_imported_lora_advertisement`, `apply_imported_provider_surface` and the
/// `mlx_catalog_status` probe keep their own bare `cfg!(target_os = "macos")` deliberately. They
/// answer a DIFFERENT question: *which engines does this BUILD link* (macOS links MLX and no candle
/// engine; Windows/Linux/Docker link candle and no MLX), and the MLX tier probe additionally reads
/// convert-output directories that only exist on macOS. This function answers *which single backend
/// does this INSTANCE route a job to*, which `candle_required` can move at runtime.
///
/// Three reasons not to unify them:
/// 1. Those sites need a LANE PAIR, consulted ASYMMETRICALLY — a withdrawal crosses lanes, a
///    positive advertisement never does. Collapsing that into this one-hot routing answer would
///    break the invariant `apply_imported_lora_advertisement` documents.
/// 2. A macOS build links no candle engine, so deriving the advertisement from `candle_required`
///    would publish verdicts for an engine this binary cannot run.
/// 3. Routing the `mlx_catalog_status` probe through here would HIDE `mlxTiers` and its per-tier
///    state from the Studio on macOS under `candle_required`, for tier directories that are
///    genuinely on disk — a regression, not a unification.
///
/// The residual skew is real but unshipped: on macOS the desktop wrapper sets
/// `SCENEWORKS_CANDLE_REQUIRED` only under `cfg(not(target_os = "macos"))` and
/// [`Settings::candle_required`] is documented "Absent on macOS", so reaching it takes a manual
/// operator override. Under that override the catalog still advertises MLX-derived imported verdicts
/// while these gates refuse against candle. Closing it properly means making the advertisement lane
/// pair a runtime DEPLOYMENT fact (including remote candle workers registered with this API), not
/// re-pointing it at this function.
pub(crate) fn enqueue_backend(state: &AppState) -> &'static str {
    if !cfg!(target_os = "macos") || state.settings.candle_required {
        "candle"
    } else {
        "mlx"
    }
}

/// Validate `advanced.decoder` against the descriptor-derived options on the exact post-preset model
/// row. This is an enqueue-time fail-closed gate: z48/video/unknown models have no compatible option,
/// and a soft donor that is not installed cannot produce a job that will fail later on the worker.
///
/// `active_backend` must come from [`enqueue_backend`] — never a hand-rolled `cfg!` — so the list
/// consulted here is the list the worker that runs this job will actually offer.
pub(crate) fn validate_selected_decoder_for_manifest(
    active_backend: &str,
    job_payload: &JsonObject,
    model_manifest_entry: &Value,
) -> Result<(), ApiError> {
    let Some(raw) = job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("decoder"))
    else {
        return Ok(());
    };
    let decoder_id = match raw {
        Value::Null => return Ok(()),
        Value::String(value) if value.trim().is_empty() || value == "native" => return Ok(()),
        Value::String(value) => value.as_str(),
        _ => {
            return Err(ApiError::bad_request(
                "advanced.decoder must be a decoder id string (or 'native')",
            ))
        }
    };
    if job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("usePid"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(ApiError::bad_request(
            "advanced.decoder cannot be combined with advanced.usePid; select exactly one decoder",
        ));
    }

    // The catalog carries capability facts for every deployable backend, but this API instance can
    // enqueue only for the backend it actually routes to. Looking across every list would let a Linux
    // candle deployment accept an MLX-only choice and defer the rejection to the worker.
    let matching: Vec<&Value> = model_manifest_entry
        .get(sceneworks_core::decoder_support::DECODERS_FIELD)
        .and_then(|decoders| decoders.get(sceneworks_core::decoder_support::BY_BACKEND_FIELD))
        .and_then(Value::as_object)
        .and_then(|by_backend| by_backend.get(active_backend))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|option| option.get("id").and_then(Value::as_str) == Some(decoder_id))
        .collect();
    if matching.is_empty() {
        let model_id = model_manifest_entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("selected model");
        return Err(ApiError::bad_request(format!(
            "decoder '{decoder_id}' is not compatible with {model_id}; z48 and unsupported providers fail closed"
        )));
    }
    if !matching
        .iter()
        .any(|option| option.get("available").and_then(Value::as_bool) == Some(true))
    {
        return Err(ApiError::bad_request(format!(
            "decoder '{decoder_id}' is not installed; install or repair its standalone pinned component before submitting"
        )));
    }
    Ok(())
}

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
    // Style-catalog fold (sc-13134): a headless/MCP client that sends a `styleId` + raw prompt gets
    // the SAME `Subject:`/`Style:` composition the web app does. Runs AFTER the preset fold so it
    // wraps the preset-composed prompt exactly as the web's composeStyledPrompt is the LAST wrap; a
    // no-op for a web request (which sends the already-composed prompt + presetPromptResolvedClientSide
    // and no top-level styleId), so a web-composed prompt is never double-folded.
    crate::styles::apply_style_to_image_payload(&state, &payload, &mut job_payload).await?;
    // Prompt enhancement is route-specific, so validate the canonical post-preset model rather
    // than trusting the DTO's pre-expansion model. This is also where client attempts to forge the
    // worker-owned report block are refused.
    validate_prompt_enhancement_payload(&job_payload)?;
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
    // The shared canonicalizer reads the final payload and also owns retry/duplicate, so no
    // alternate image boundary can retain caller-supplied manifest metadata.
    let mut model_manifest_entry =
        crate::jobs::canonicalize_image_model_payload(&state, &job_type, &mut job_payload)
            .await?
            .ok_or_else(|| {
                ApiError::internal("image model canonicalization returned no catalog entry")
            })?;
    // Overlay the authored text-encoder selection onto the canonical entry and re-stamp it, so the
    // worker sees the resolved encoder rather than the catalog default. The decoder gate then runs
    // against the same server-owned entry.
    resolve_selected_image_text_encoder(&state, &job_payload, &model_id, &mut model_manifest_entry)
        .await?;
    validate_selected_decoder_for_manifest(
        enqueue_backend(&state),
        &job_payload,
        &model_manifest_entry,
    )?;
    job_payload.insert(
        "modelManifestEntry".to_owned(),
        model_manifest_entry.clone(),
    );
    // The model's declared `defaults.resolution`, keyed off the canonical post-preset entry for the same
    // reason the video route's gates are (sc-12300). The image half of the dead-`defaults.*` sweep:
    // the web honors this key (`ImageStudio.jsx:215`) but Rust did not, so a caller that named no
    // size rendered a blanket 1024x1024 — HALF the declared 2048x2048 on the four sensenova_u1_8b
    // variants, the text/infographic family where resolution is the whole point, and 1024 instead of
    // chroma1_flash's declared 768. Silent: geometry coerces, so nothing ever errored (sc-12400).
    //
    // Written back per side only when the caller named none, so a caller's own size is untouched.
    if let Some(entry) = model_manifest_entry.as_object() {
        let (default_width, default_height) =
            image_default_resolution(entry).unwrap_or((1024, 1024));
        if !job_payload.contains_key("width") {
            job_payload.insert("width".to_owned(), Value::from(default_width));
        }
        if !job_payload.contains_key("height") {
            job_payload.insert("height".to_owned(), Value::from(default_height));
        }
        // `defaults.count`, the resolution key's other half — 29 of the 45 image models declare
        // `count: 1` against this layer's blanket 4. Reading only the geometry made the bare call
        // WORSE, not better: sensenova_u1_8b declares `2048x2048` AND `count: 1`, so honoring the
        // size alone took `generate_image(model = "sensenova_u1_8b", prompt = …)` from 4x1024² to
        // 4x2048² — 4x the pixels — where the model asks for 1x2048², i.e. the original cost at the
        // correct geometry. One intent, two keys; reading half of it is what over-charged.
        if !job_payload.contains_key("count") {
            let count = image_default_count(entry).unwrap_or(4);
            job_payload.insert("count".to_owned(), Value::from(count));
        }
    }
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
    validate_imported_submission(&state, &model_id, &job_payload)?;
    if payload.seed.is_none() {
        // `job_payload["count"]` is the resolved count — the block above writes the model's declared
        // `defaults.count` whenever the caller named none, so the seed batch matches what actually
        // renders. The fallback only survives for a manifest entry that is not an object (an
        // unknown model resolves to `{}`, which IS one), where the caller's own count, then the
        // blanket, is all there is.
        let count = job_payload
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or_else(|| payload.count.unwrap_or_else(default_image_count));
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

const MIN_VECTOR_SVG_BYTES: u32 = 1_024;
const MAX_VECTOR_SVG_BYTES: u32 = 256 * 1_024;

fn validate_vector_request(payload: &VectorRequest) -> Result<(), ApiError> {
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
    if !(1_000..=600_000).contains(&budget.max_wall_time_ms) {
        return Err(ApiError::bad_request(
            "detailBudget.maxWallTimeMs must be between 1000 and 600000",
        ));
    }
    Ok(())
}

fn validate_vector_model_manifest(
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

fn validate_vector_model_availability(
    model_id: &str,
    backend: &str,
    model: &Value,
) -> Result<(), ApiError> {
    let provider = model
        .pointer(&format!("/vector/providers/{backend}"))
        .and_then(Value::as_object);
    if provider
        .and_then(|value| value.get("available"))
        .and_then(Value::as_bool)
        != Some(true)
    {
        let reason = provider
            .and_then(|value| value.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or("provider_not_linked");
        return Err(ApiError::typed(
            StatusCode::CONFLICT,
            format!("Model {model_id} is unavailable on the {backend} vector backend"),
            "vector_backend_unavailable",
            json!({ "reason": reason, "modelId": model_id, "backend": backend }),
        ));
    }
    let install_state = model
        .get("installState")
        .and_then(Value::as_str)
        .unwrap_or("missing");
    let cache_state = model
        .get("cacheState")
        .and_then(Value::as_str)
        .unwrap_or("missing");
    if install_state != "installed" || cache_state != "complete" {
        let reason = if cache_state == "incomplete" {
            "model_incomplete"
        } else {
            "model_missing"
        };
        return Err(ApiError::typed(
            StatusCode::CONFLICT,
            format!(
                "Model {model_id} is not completely installed; download or repair it in Model Manager before submitting"
            ),
            "vector_model_unavailable",
            json!({
                "reason": reason,
                "modelId": model_id,
                "installState": install_state,
                "cacheState": cache_state,
                "downloadable": model.get("downloadable").and_then(Value::as_bool).unwrap_or(false),
                "repairAvailable": model.get("repairAvailable").and_then(Value::as_bool).unwrap_or(false),
                "missingRequiredFiles": model.get("missingRequiredFiles").cloned().unwrap_or_else(|| json!([])),
            }),
        ));
    }
    Ok(())
}

fn validate_vector_provider_request(
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

async fn validate_vector_source_asset(
    state: AppState,
    project_id: String,
    source_asset_id: String,
) -> Result<(), ApiError> {
    let (asset, media_path) = project_call(state, move |store| {
        let asset = store.get_asset(&project_id, &source_asset_id)?;
        let media_path = store.resolve_asset_media_path(&project_id, &source_asset_id)?;
        Ok((asset, media_path))
    })
    .await?;
    let media_type = asset
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mime = asset
        .pointer("/file/mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if media_type != "image" || !mime.starts_with("image/") || mime == "image/svg+xml" {
        return Err(ApiError::bad_request(
            "sourceAssetId must name a raster image owned by projectId",
        ));
    }
    if !media_path.is_file() {
        return Err(ApiError::bad_request(
            "sourceAssetId media is missing from its project",
        ));
    }
    Ok(())
}

/// Create a typed Vector Studio job. The API resolves all caller-owned identifiers and model
/// capability facts before enqueue; the worker alone owns provider streaming, SVG sanitization,
/// preview rasterization, and atomic publication.
pub(crate) async fn create_vector_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VectorRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_vector_request(&payload)?;
    if let Some(source_asset_id) = payload.source_asset_id.clone() {
        validate_vector_source_asset(state.clone(), payload.project_id.clone(), source_asset_id)
            .await?;
    }
    let model_manifest_entry = crate::models::model_catalog(&state)
        .await?
        .into_iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(payload.model.as_str()))
        .unwrap_or_else(|| json!({}));
    validate_vector_model_manifest(&payload.model, payload.mode, &model_manifest_entry)?;
    validate_vector_provider_request(&payload, &model_manifest_entry)?;
    validate_vector_model_availability(
        &payload.model,
        enqueue_backend(&state),
        &model_manifest_entry,
    )?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = payload.project_id.clone();
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    let job = create_generation_job(
        state,
        JobType::VectorGenerate,
        Some(project_id),
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

const VECTOR_PROMPT_WORKFLOW_KIND: &str = "create_from_prompt";
const VECTOR_WORKFLOW_POLL_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Clone, Copy)]
enum VectorWorkflowReplayRelation<'a> {
    Fresh,
    Retry {
        source_job_id: &'a str,
        attempts: u32,
    },
    Duplicate {
        duplicate_of_job_id: &'a str,
    },
}

fn typed_vector_workflow_error(code: &'static str, detail: impl Into<String>) -> ApiError {
    ApiError::typed(
        StatusCode::CONFLICT,
        detail.into(),
        code,
        json!({ "workflow": VECTOR_PROMPT_WORKFLOW_KIND }),
    )
}

/// Exactly one immutable primary artifact identity is required. Imported/path-backed rows and
/// multi-revision tier sets deliberately fail closed: a prompt workflow must be replayable without
/// guessing which bytes the first stage meant.
pub(crate) fn authoritative_workflow_revision(model: &Value) -> Result<String, ApiError> {
    let primary_downloads = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| download.get("coRequisite").and_then(Value::as_bool) != Some(true))
        .collect::<Vec<_>>();
    let revisions = primary_downloads
        .iter()
        .filter_map(|download| download.get("revision").and_then(Value::as_str))
        .collect::<std::collections::BTreeSet<_>>();
    let every_revision_is_immutable = primary_downloads.iter().all(|download| {
        download
            .get("revision")
            .and_then(Value::as_str)
            .is_some_and(|revision| {
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            })
    });
    if primary_downloads.is_empty() || !every_revision_is_immutable || revisions.len() != 1 {
        return Err(typed_vector_workflow_error(
            "vector_workflow_artifact_ambiguous",
            "Create from Prompt requires one immutable authoritative revision for each stage.",
        ));
    }
    Ok(revisions
        .into_iter()
        .next()
        .expect("one immutable primary revision")
        .to_owned())
}

fn validate_workflow_revision(
    stage: &str,
    actual: &str,
    expected: Option<&str>,
) -> Result<(), ApiError> {
    if expected.is_some_and(|expected| expected != actual) {
        return Err(ApiError::typed(
            StatusCode::CONFLICT,
            format!(
                "The {stage} model revision changed since this workflow was recorded; replay was not queued."
            ),
            "vector_workflow_revision_drift",
            json!({ "stage": stage, "expectedRevision": expected, "actualRevision": actual }),
        ));
    }
    Ok(())
}

fn validate_raster_workflow_model(
    state: &AppState,
    model_id: &str,
    model: &Value,
) -> Result<String, ApiError> {
    if model.get("type").and_then(Value::as_str) != Some("image")
        || !model
            .get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|capabilities| {
                capabilities
                    .iter()
                    .any(|capability| capability.as_str() == Some("text_to_image"))
            })
    {
        return Err(typed_vector_workflow_error(
            "vector_workflow_raster_unsupported",
            format!("Model {model_id} does not declare text_to_image."),
        ));
    }
    let backend_support = if enqueue_backend(state) == "mlx" {
        model.pointer("/macSupport/supported")
    } else {
        model.pointer("/candleSupport/supported")
    };
    if backend_support.and_then(Value::as_bool) != Some(true)
        || model.get("usable").and_then(Value::as_bool) == Some(false)
    {
        return Err(typed_vector_workflow_error(
            "vector_workflow_raster_unclaimable",
            format!("Model {model_id} is not claimable by the current raster backend."),
        ));
    }
    if model.get("installState").and_then(Value::as_str) != Some("installed")
        || model.get("cacheState").and_then(Value::as_str) != Some("complete")
    {
        return Err(typed_vector_workflow_error(
            "vector_workflow_raster_unavailable",
            format!("Model {model_id} must be completely installed in Model Manager."),
        ));
    }
    authoritative_workflow_revision(model)
}

fn workflow_stage<'a>(payload: &'a JsonObject, stage: &str) -> Result<&'a Value, ApiError> {
    payload
        .get("workflow")
        .and_then(|workflow| workflow.get(stage))
        .ok_or_else(|| ApiError::internal(format!("vector workflow is missing {stage}")))
}

async fn create_vector_prompt_workflow_internal(
    state: AppState,
    payload: VectorPromptWorkflowRequest,
    relation: VectorWorkflowReplayRelation<'_>,
) -> Result<JobSnapshot, ApiError> {
    let raster_request: ImageJobRequest = serde_json::from_value(json!({
        "projectId": payload.project_id,
        "projectName": payload.project_name,
        "mode": "text_to_image",
        "prompt": payload.prompt,
        "negativePrompt": payload.negative_prompt,
        "model": payload.raster_model,
        "count": 1,
        "seed": payload.seed,
        "width": payload.width,
        "height": payload.height,
        "requestedGpu": payload.requested_gpu,
    }))
    .map_err(|error| ApiError::bad_request(format!("Invalid raster stage: {error}")))?;
    validate_image_job(&raster_request)?;

    let catalog = crate::models::model_catalog(&state).await?;
    let raster_model = catalog
        .iter()
        .find(|model| {
            model.get("id").and_then(Value::as_str) == Some(payload.raster_model.as_str())
        })
        .cloned()
        .unwrap_or_else(|| json!({}));
    let vector_model = catalog
        .iter()
        .find(|model| {
            model.get("id").and_then(Value::as_str) == Some(payload.vector_model.as_str())
        })
        .cloned()
        .unwrap_or_else(|| json!({}));
    let raster_revision =
        validate_raster_workflow_model(&state, &payload.raster_model, &raster_model)?;
    let vector_revision = authoritative_workflow_revision(&vector_model)?;
    validate_workflow_revision(
        "raster",
        &raster_revision,
        payload.expected_raster_revision.as_deref(),
    )?;
    validate_workflow_revision(
        "vector",
        &vector_revision,
        payload.expected_vector_revision.as_deref(),
    )?;
    validate_vector_model_manifest(&payload.vector_model, VectorMode::ImageToSvg, &vector_model)?;
    let pending_vector_request = VectorRequest {
        project_id: payload.project_id.clone(),
        project_name: payload.project_name.clone(),
        mode: VectorMode::ImageToSvg,
        model: payload.vector_model.clone(),
        source_asset_id: Some("workflow_pending_source".to_owned()),
        // StarVector-1B does not accept guidance. The disclosed prompt belongs only to the raster
        // stage; forwarding it here would falsely turn the composition into native text-to-SVG.
        prompt: String::new(),
        sampling: payload.sampling.clone(),
        detail_budget: payload.detail_budget.clone(),
        requested_gpu: payload.requested_gpu.clone(),
    };
    validate_vector_request(&pending_vector_request)?;
    validate_vector_provider_request(&pending_vector_request, &vector_model)?;
    validate_vector_model_availability(
        &payload.vector_model,
        enqueue_backend(&state),
        &vector_model,
    )?;

    let parent_id = format!("job_{}", uuid::Uuid::new_v4().simple());
    let workflow_id = format!("vwf_{}", uuid::Uuid::new_v4().simple());
    let mut parent_payload = to_json_object(&pending_vector_request)?;
    parent_payload.remove("requestedGpu");
    parent_payload.remove("sourceAssetId");
    parent_payload.insert("modelManifestEntry".to_owned(), vector_model.clone());
    parent_payload.insert(
        "workflow".to_owned(),
        json!({
            "kind": VECTOR_PROMPT_WORKFLOW_KIND,
            "disclosure": "raster_to_vector",
            "id": workflow_id,
            "parentJobId": parent_id,
            "childJobId": Value::Null,
            "intermediateAssetId": Value::Null,
            "intermediateVisibility": "hidden_retained_on_success",
            "rasterStage": {
                "model": payload.raster_model,
                "revision": raster_revision,
                "prompt": payload.prompt,
                "negativePrompt": payload.negative_prompt,
                "seed": payload.seed,
                "width": payload.width,
                "height": payload.height,
                "count": 1,
            },
            "vectorStage": {
                "model": payload.vector_model,
                "revision": vector_revision,
                "mode": "image_to_svg",
                "sampling": payload.sampling,
                "detailBudget": payload.detail_budget,
            },
        }),
    );
    let (source_job_id, duplicate_of_job_id, attempts) = match relation {
        VectorWorkflowReplayRelation::Fresh => (None, None, 1),
        VectorWorkflowReplayRelation::Retry {
            source_job_id,
            attempts,
        } => (Some(source_job_id.to_owned()), None, attempts),
        VectorWorkflowReplayRelation::Duplicate {
            duplicate_of_job_id,
        } => (None, Some(duplicate_of_job_id.to_owned()), 1),
    };
    let parent = store_call(state.clone(), {
        let parent_id = parent_id.clone();
        let project_id = payload.project_id.clone();
        let project_name = payload.project_name.clone();
        let requested_gpu = payload.requested_gpu.clone();
        let parent_payload = parent_payload.clone();
        move |store, _timeout| {
            store.create_job_with_id(
                parent_id,
                CreateJob {
                    job_type: JobType::VectorGenerate,
                    project_id: Some(project_id),
                    project_name,
                    payload: parent_payload,
                    requested_gpu,
                    source_job_id,
                    duplicate_of_job_id,
                    attempts,
                    initial_status: Some(JobStatus::PendingWorkflow),
                },
            )
        }
    })
    .await?;
    publish(&state, "job.updated", &parent);
    publish_queue(&state).await?;

    let mut child_request = raster_request;
    child_request.workflow_parent_id = Some(parent_id.clone());
    child_request.workflow_id = Some(workflow_id.clone());
    let child = match create_image_job(State(state.clone()), ApiJson(child_request)).await {
        Ok((_status, Json(child))) => child,
        Err(error) => {
            let _ = store_call(state.clone(), {
                let parent_id = parent_id.clone();
                let detail = error.detail.clone();
                move |store, _timeout| {
                    store
                        .terminate_pending_workflow_job(
                            &parent_id,
                            JobStatus::Failed,
                            "Raster stage could not be queued.",
                            Some(&detail),
                        )
                        .map(|transition| transition.job)
                }
            })
            .await;
            return Err(error);
        }
    };
    let mut linked_payload = parent.payload.clone();
    if let Some(workflow) = linked_payload
        .get_mut("workflow")
        .and_then(Value::as_object_mut)
    {
        workflow.insert("childJobId".to_owned(), Value::String(child.id.clone()));
        if let Some(seed) = child
            .payload
            .get("seeds")
            .and_then(Value::as_array)
            .and_then(|seeds| seeds.first())
            .cloned()
        {
            workflow
                .get_mut("rasterStage")
                .and_then(Value::as_object_mut)
                .expect("raster stage object")
                .insert("seed".to_owned(), seed);
        }
    }
    let linked = store_call(state.clone(), {
        let parent_id = parent_id.clone();
        move |store, _timeout| store.update_pending_workflow_payload(&parent_id, linked_payload)
    })
    .await?;
    if !linked.changed {
        let _ = store_call(state.clone(), {
            let child_id = child.id.clone();
            move |store, _timeout| store.cancel_job(&child_id)
        })
        .await;
        return Ok(linked.job);
    }
    publish(&state, "job.updated", &linked.job);
    spawn_vector_prompt_workflow_coordinator(state, parent_id);
    Ok(linked.job)
}

pub(crate) async fn create_vector_prompt_workflow(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VectorPromptWorkflowRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job =
        create_vector_prompt_workflow_internal(state, payload, VectorWorkflowReplayRelation::Fresh)
            .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

fn vector_prompt_workflow(payload: &JsonObject) -> Option<&serde_json::Map<String, Value>> {
    payload
        .get("workflow")
        .and_then(Value::as_object)
        .filter(|workflow| {
            workflow.get("kind").and_then(Value::as_str) == Some(VECTOR_PROMPT_WORKFLOW_KIND)
        })
}

async fn cleanup_vector_prompt_intermediate(
    state: &AppState,
    parent: &JobSnapshot,
) -> Result<bool, ApiError> {
    let Some(workflow) = vector_prompt_workflow(&parent.payload) else {
        return Ok(false);
    };
    let Some(asset_id) = workflow
        .get("intermediateAssetId")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return Ok(false);
    };
    cleanup_named_vector_prompt_intermediate(state, parent, &asset_id).await
}

/// Cleanup by the asset id held by the coordinator, including the narrow race where the child
/// asset has been ownership-stamped but cancel wins before that id is attached to the parent row.
async fn cleanup_named_vector_prompt_intermediate(
    state: &AppState,
    parent: &JobSnapshot,
    asset_id: &str,
) -> Result<bool, ApiError> {
    let Some(project_id) = parent.project_id.clone() else {
        return Ok(false);
    };
    let Some(workflow_id) = vector_prompt_workflow(&parent.payload)
        .and_then(|workflow| workflow.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return Ok(false);
    };
    let parent_id = parent.id.clone();
    let asset_id = asset_id.to_owned();
    project_call(state.clone(), move |store| {
        store.purge_vector_workflow_intermediate(&project_id, &asset_id, &workflow_id, &parent_id)
    })
    .await
}

/// Cancel the ordinary raster child when its composed parent is canceled. Cleanup remains
/// ownership-checked and idempotent, so this is safe before, during, or after child publication.
pub(crate) async fn cascade_cancel_vector_prompt_workflow(
    state: &AppState,
    parent: &JobSnapshot,
) -> Result<(), ApiError> {
    let Some(workflow) = vector_prompt_workflow(&parent.payload) else {
        return Ok(());
    };
    if let Some(child_id) = workflow
        .get("childJobId")
        .and_then(Value::as_str)
        .map(str::to_owned)
    {
        let child = store_call(state.clone(), move |store, _timeout| {
            store.cancel_job(&child_id)
        })
        .await?;
        publish(state, "job.updated", &child);
    }
    let _ = cleanup_vector_prompt_intermediate(state, parent).await?;
    Ok(())
}

fn spawn_vector_prompt_workflow_coordinator(state: AppState, parent_id: String) {
    tokio::spawn(async move {
        if let Err(error) = run_vector_prompt_workflow_coordinator(&state, &parent_id).await {
            tracing::error!(
                event = "vector_prompt_workflow_coordinator_failed",
                %parent_id,
                error = %error.detail,
                "prompt-to-vector workflow coordinator stopped"
            );
        }
    });
}

/// Resume every disclosed workflow visible in the retained jobs window. Pending parents continue
/// their child watch; queued/active parents continue terminal cleanup; terminal failures get one
/// idempotent ownership-checked cleanup attempt.
pub(crate) fn spawn_vector_prompt_workflow_recovery(state: AppState) {
    tokio::spawn(async move {
        let jobs = match store_call(state.clone(), |store, _timeout| {
            store.list_vector_prompt_workflow_jobs_for_recovery()
        })
        .await
        {
            Ok(jobs) => jobs,
            Err(error) => {
                tracing::error!(event = "vector_workflow_recovery_scan_failed", error = %error.detail);
                return;
            }
        };
        for job in jobs {
            if vector_prompt_workflow(&job.payload).is_some() {
                spawn_vector_prompt_workflow_coordinator(state.clone(), job.id);
            }
        }
    });
}

async fn terminate_vector_prompt_parent(
    state: &AppState,
    parent: &JobSnapshot,
    status: JobStatus,
    message: &str,
    error: &str,
) -> Result<JobSnapshot, ApiError> {
    let transition = store_call(state.clone(), {
        let parent_id = parent.id.clone();
        let message = message.to_owned();
        let error = error.to_owned();
        move |store, _timeout| {
            store.terminate_pending_workflow_job(&parent_id, status, &message, Some(&error))
        }
    })
    .await?;
    if transition.changed {
        publish(state, "job.updated", &transition.job);
        publish_queue(state).await?;
    }
    Ok(transition.job)
}

async fn prepare_vector_parent_from_intermediate(
    state: &AppState,
    parent: &JobSnapshot,
    child: &JobSnapshot,
    asset_id: &str,
) -> Result<JsonObject, ApiError> {
    let project_id = parent
        .project_id
        .clone()
        .ok_or_else(|| ApiError::internal("vector workflow parent has no project"))?;
    let workflow = vector_prompt_workflow(&parent.payload)
        .ok_or_else(|| ApiError::internal("vector workflow metadata is missing"))?;
    let workflow_id = workflow
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("vector workflow id is missing"))?
        .to_owned();
    let vector_stage = workflow_stage(&parent.payload, "vectorStage")?;
    let vector_model_id = vector_stage
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("vector workflow model is missing"))?
        .to_owned();
    let expected_revision = vector_stage
        .get("revision")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("vector workflow revision is missing"))?
        .to_owned();
    let model = crate::models::model_catalog(state)
        .await?
        .into_iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(vector_model_id.as_str()))
        .unwrap_or_else(|| json!({}));
    let actual_revision = authoritative_workflow_revision(&model)?;
    validate_workflow_revision("vector", &actual_revision, Some(&expected_revision))?;
    validate_vector_model_manifest(&vector_model_id, VectorMode::ImageToSvg, &model)?;
    validate_vector_model_availability(&vector_model_id, enqueue_backend(state), &model)?;
    validate_vector_source_asset(state.clone(), project_id.clone(), asset_id.to_owned()).await?;

    let child_id = child.id.clone();
    let marked = project_call(state.clone(), {
        let project_id = project_id.clone();
        let asset_id = asset_id.to_owned();
        let parent_id = parent.id.clone();
        let workflow_id = workflow_id.clone();
        move |store| {
            store.mark_vector_workflow_intermediate(
                &project_id,
                &asset_id,
                &workflow_id,
                &parent_id,
                &child_id,
            )
        }
    })
    .await?;
    if marked.get("type").and_then(Value::as_str) != Some("image") {
        return Err(ApiError::internal(
            "workflow intermediate changed type while being retained",
        ));
    }

    let mut resolved = parent.payload.clone();
    resolved.insert(
        "sourceAssetId".to_owned(),
        Value::String(asset_id.to_owned()),
    );
    resolved.insert("modelManifestEntry".to_owned(), model);
    let workflow = resolved
        .get_mut("workflow")
        .and_then(Value::as_object_mut)
        .expect("workflow object");
    workflow.insert(
        "intermediateAssetId".to_owned(),
        Value::String(asset_id.to_owned()),
    );
    workflow
        .get_mut("rasterStage")
        .and_then(Value::as_object_mut)
        .expect("raster stage")
        .insert("jobId".to_owned(), Value::String(child.id.clone()));
    Ok(resolved)
}

async fn run_vector_prompt_workflow_coordinator(
    state: &AppState,
    parent_id: &str,
) -> Result<(), ApiError> {
    let mut completed_without_asset_polls = 0u32;
    loop {
        let parent = store_call(state.clone(), {
            let parent_id = parent_id.to_owned();
            move |store, _timeout| store.get_job(&parent_id)
        })
        .await?;
        if vector_prompt_workflow(&parent.payload).is_none() {
            return Ok(());
        }
        if matches!(parent.status, JobStatus::Completed) {
            return Ok(());
        }
        if matches!(
            parent.status,
            JobStatus::Failed | JobStatus::Canceled | JobStatus::Interrupted
        ) {
            let _ = cleanup_vector_prompt_intermediate(state, &parent).await?;
            return Ok(());
        }
        if parent.status != JobStatus::PendingWorkflow {
            tokio::time::sleep(VECTOR_WORKFLOW_POLL_INTERVAL).await;
            continue;
        }
        let child_id = vector_prompt_workflow(&parent.payload)
            .and_then(|workflow| workflow.get("childJobId"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(child_id) = child_id else {
            tokio::time::sleep(VECTOR_WORKFLOW_POLL_INTERVAL).await;
            continue;
        };
        let child = store_call(state.clone(), {
            let child_id = child_id.clone();
            move |store, _timeout| store.get_job(&child_id)
        })
        .await?;
        match child.status {
            JobStatus::Completed => {
                let asset_id = child
                    .result
                    .get("assetIds")
                    .and_then(Value::as_array)
                    .and_then(|ids| ids.first())
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let Some(asset_id) = asset_id else {
                    // Terminal progress is accepted before its durable asset side effect completes.
                    // Give that idempotent recovery handoff time to publish the child sidecar.
                    completed_without_asset_polls += 1;
                    if completed_without_asset_polls < 300 {
                        tokio::time::sleep(VECTOR_WORKFLOW_POLL_INTERVAL).await;
                        continue;
                    }
                    let terminal = terminate_vector_prompt_parent(
                        state,
                        &parent,
                        JobStatus::Failed,
                        "Raster stage completed without a publishable asset.",
                        "vector_workflow_raster_asset_missing",
                    )
                    .await?;
                    let _ = cleanup_vector_prompt_intermediate(state, &terminal).await?;
                    return Ok(());
                };
                match prepare_vector_parent_from_intermediate(state, &parent, &child, &asset_id)
                    .await
                {
                    Ok(payload) => {
                        let transition = store_call(state.clone(), {
                            let parent_id = parent.id.clone();
                            move |store, _timeout| {
                                store.promote_pending_workflow_job(&parent_id, payload)
                            }
                        })
                        .await?;
                        if transition.changed {
                            publish(state, "job.updated", &transition.job);
                            publish_queue(state).await?;
                        } else {
                            let _ = cleanup_named_vector_prompt_intermediate(
                                state,
                                &transition.job,
                                &asset_id,
                            )
                            .await?;
                            return Ok(());
                        }
                    }
                    Err(error) => {
                        let terminal = terminate_vector_prompt_parent(
                            state,
                            &parent,
                            JobStatus::Failed,
                            "Vector stage became unavailable before dispatch.",
                            &error.detail,
                        )
                        .await?;
                        let _ = cleanup_vector_prompt_intermediate(state, &terminal).await?;
                        return Ok(());
                    }
                }
            }
            JobStatus::Failed | JobStatus::Interrupted => {
                let terminal = terminate_vector_prompt_parent(
                    state,
                    &parent,
                    JobStatus::Failed,
                    "Raster stage failed; vectorization was not dispatched.",
                    child
                        .error
                        .as_deref()
                        .unwrap_or("vector_workflow_raster_failed"),
                )
                .await?;
                let _ = cleanup_vector_prompt_intermediate(state, &terminal).await?;
                return Ok(());
            }
            JobStatus::Canceled => {
                let terminal = terminate_vector_prompt_parent(
                    state,
                    &parent,
                    JobStatus::Canceled,
                    "Raster stage was canceled; vectorization was not dispatched.",
                    "vector_workflow_raster_canceled",
                )
                .await?;
                let _ = cleanup_vector_prompt_intermediate(state, &terminal).await?;
                return Ok(());
            }
            _ => {}
        }
        tokio::time::sleep(VECTOR_WORKFLOW_POLL_INTERVAL).await;
    }
}

pub(crate) async fn replay_vector_prompt_workflow(
    state: AppState,
    job_id: &str,
    duplicate: bool,
    requested_gpu: Option<String>,
    has_payload_changes: bool,
) -> Result<Option<JobSnapshot>, ApiError> {
    let original = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await?;
    let Some(workflow) = vector_prompt_workflow(&original.payload) else {
        return Ok(None);
    };
    if has_payload_changes {
        let operation = if duplicate { "duplicate" } else { "retry" };
        return Err(ApiError::bad_request(format!(
            "Create-from-Prompt {operation} does not accept payloadChanges; replay the recorded two-stage recipe."
        )));
    }
    let raster = workflow
        .get("rasterStage")
        .ok_or_else(|| ApiError::internal("workflow raster stage is missing"))?;
    let vector = workflow
        .get("vectorStage")
        .ok_or_else(|| ApiError::internal("workflow vector stage is missing"))?;
    let request: VectorPromptWorkflowRequest = serde_json::from_value(json!({
        "projectId": original.project_id,
        "projectName": original.project_name,
        "prompt": raster.get("prompt"),
        "negativePrompt": raster.get("negativePrompt"),
        "rasterModel": raster.get("model"),
        "vectorModel": vector.get("model"),
        "seed": raster.get("seed"),
        "width": raster.get("width"),
        "height": raster.get("height"),
        "sampling": vector.get("sampling"),
        "detailBudget": vector.get("detailBudget"),
        "requestedGpu": requested_gpu.unwrap_or(original.requested_gpu.clone()),
        "expectedRasterRevision": raster.get("revision"),
        "expectedVectorRevision": vector.get("revision"),
    }))
    .map_err(|error| {
        ApiError::internal(format!("stored vector workflow cannot replay: {error}"))
    })?;
    let relation = if duplicate {
        VectorWorkflowReplayRelation::Duplicate {
            duplicate_of_job_id: &original.id,
        }
    } else {
        if original.attempts >= sceneworks_core::jobs_store::MAX_JOB_ATTEMPTS {
            return Err(ApiError::bad_request("Job retry limit reached."));
        }
        VectorWorkflowReplayRelation::Retry {
            source_job_id: &original.id,
            attempts: original.attempts + 1,
        }
    };
    create_vector_prompt_workflow_internal(state, request, relation)
        .await
        .map(Some)
}

/// Refuse every imported request shape that the selected backend cannot execute. The exact stamped
/// source shape and operation select one provider registration; family identity alone never admits
/// a request. Builtins retain their id-keyed routing and are out of this family gate.
pub(crate) fn validate_imported_submission(
    state: &AppState,
    model_id: &str,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let Some(entry) = payload.get("modelManifestEntry").and_then(Value::as_object) else {
        return Ok(());
    };
    if entry.get("catalogScope").and_then(Value::as_str) == Some("builtin")
        || sceneworks_core::jobs_store::is_builtin_image_model(model_id)
    {
        return Ok(());
    }
    let has_material_control = payload
        .get("advanced")
        .and_then(Value::as_object)
        .is_some_and(sceneworks_core::jobs_store::imported_control_intent_is_material);
    let backend = enqueue_backend(state);
    let candle_required = backend == "candle";
    let family = entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .unwrap_or("unknown");
    if sceneworks_core::jobs_store::imported_image_request_provider_eligible(
        model_id, payload, backend,
    ) {
        return Ok(());
    }
    let feature = if has_material_control
        && !payload
            .get("advanced")
            .and_then(Value::as_object)
            .and_then(|advanced| advanced.get("poses"))
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
    {
        "control image/mode without a supported Pose request"
    } else if payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
    {
        "strict-pose control"
    } else if payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("phases"))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
    {
        "multi-phase denoise"
    } else if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        "image edit"
    } else if payload
        .get("loras")
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
    {
        "LoRA/LoKr adapters"
    } else {
        "requested generation shape"
    };
    let code = if candle_required {
        "candle_unsupported"
    } else if has_material_control {
        "imported_control_unsupported"
    } else {
        "imported_unsupported"
    };
    Err(ApiError::bad_request(format!(
        "{code}: imported {family} {feature} is not supported by the resolved {backend} provider \
         registration for this exact source and operation; the request was not queued"
    )))
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) fn validate_vqa_job(payload: &VqaJobRequest) -> Result<(), ApiError> {
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) fn validate_interleave_job(payload: &InterleaveJobRequest) -> Result<(), ApiError> {
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

/// Request-scoped, lazily-memoized snapshot of the model and LoRA catalogs (sc-8819,
/// F-017). A single preset-backed `POST /image/jobs` (or `/video/jobs`) fans out into
/// `recipe_preset_catalog`, `merge_preset_loras_into_payload`, and
/// `validate_job_lora_compatibility`, each of which formerly re-assembled
/// `model_catalog`/`lora_catalog` from scratch — re-running the per-model install-state
/// probes (recursive HF-cache walks, `model_is_installed`, `mlx_catalog_status`) 2–3×
/// over the whole catalog per submit. Threading one snapshot through those seams limits
/// each request to one model value and one LoRA value; the model value may require no
/// new filesystem probing when the process-shared generation is already warm.
///
/// The request-scoped layer still guarantees that preset expansion and LoRA validation
/// see identical values within one job-create. Its model value is sourced from the
/// process-shared, generation-keyed install-state cache (SC-14784), which coalesces cold
/// callers and is invalidated by model lifecycle completion, model deletion, and model
/// manifest writes. LoRAs remain memoized only per `(request, project_id)`.
#[derive(Default)]
pub(crate) struct JobCatalogSnapshot {
    models: tokio::sync::OnceCell<Vec<Value>>,
    loras_by_project: tokio::sync::Mutex<HashMap<Option<String>, Arc<Vec<Value>>>>,
}

impl JobCatalogSnapshot {
    /// The model catalog, fetched once per request and reused thereafter. The direct
    /// `model_catalog(state)` call may itself reuse the current process-shared generation.
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
    // The mode → job-type mapping lives in core (sc-19570) because the UI-gating probes must build
    // the SAME pair this route enqueues before asking the claim predicates about it. Two copies
    // would let the oracle answer about a job type that is never created.
    let job_type = video_job_type_for_mode(payload.mode.as_str());
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
    // Do not advertise or enqueue the paired source layout until the pinned inference descriptor
    // records it. The worker implementation is deliberately source-ready ahead of that pin, but a
    // direct API client must not be able to send additional references to an older engine that
    // cannot consume them. The catalog flag is absent today and will be set only with the paired
    // descriptor/pin evidence.
    validate_video_reference_asset_ids_payload(&job_payload, &model_manifest_entry)?;
    validate_selected_decoder_for_manifest(
        enqueue_backend(&state),
        &job_payload,
        &model_manifest_entry,
    )?;
    ensure_video_model_available_on_platform(
        &model_id,
        &model_manifest_entry,
        video_job_platform(&state),
    )?;
    // The model's declared `limits.hardMaxDuration`, enforced at enqueue (sc-12297). It had ten
    // declarations and zero readers, so `validate_video_job`'s blanket `1..=30` was the ONLY
    // ceiling: a raw API/MCP/preset-replay caller could ask mochi_1 (cap 5) for 30s @ 30fps, and
    // the resulting 901 frames clear the engine's own `frames % 6 == 1` check — nothing downstream
    // says no. Rejected, never clamped: silently rendering 5s of a 30s request is the same
    // silent-coercion class as the 848→832 rewrite (sc-11993/sc-12294).
    //
    // It lives HERE, not in `validate_video_job`, because this is the first point that holds BOTH
    // halves of the decision: the resolved manifest entry (the cap) and the post-preset `model_id`
    // (whose cap). Reading either from the DTO is precisely sc-12300 — a preset may have replaced
    // the model, leaving `payload.model` stale, which would gate against the DEFAULT model's cap.
    // Duration comes off `job_payload` for the same reason: gate the value actually enqueued.
    // (Presets deliberately do not patch duration today — see `apply_recipe_preset_to_video_payload`
    // — so this reads the DTO's value; it just stops being a latent bug if that ever changes.)
    //
    // The duration is RESOLVED against the model's declared `defaults.duration` first, for the same
    // reason the fps menu is below — and sc-12297 shipped without it, which is the bug sc-12400
    // fixes. The DTO blanket was 6.0, serialized unconditionally, and 6.0 is past the cap of 7 of
    // the 10 shipped video models: a payload that named NO duration was rejected for "asking for
    // 6s", a value the caller never set, with a lever ("shorten the clip") for a field they never
    // touched. Enforcing a manifest constraint requires the layer's own default to be
    // manifest-aware first, or the gate refuses a payload this route constructed itself.
    if let Some(entry) = model_manifest_entry.as_object() {
        let duration = resolve_duration(job_payload.get("duration"), entry);
        if let Some(message) = duration_limit_error(&model_id, duration, entry) {
            return Err(ApiError::bad_request(message));
        }
        // Write back ONLY the resolved default. A duration the caller named is already in the
        // payload verbatim — `validate_video_job` bounded it to 1..=30, so it needs no clamp — and
        // rewriting it would flatten its JSON shape: `duration` is a `ContractNumber`
        // (= `serde_json::Number`), which carries int-vs-float across the wire, so a caller's `10`
        // must not become `10.0`.
        if !job_payload.contains_key("duration") {
            job_payload.insert("duration".to_owned(), contract_number(duration));
        }
    }
    // The model's declared `limits.hardMinSteps` AND `limits.steps`, enforced at enqueue (sc-19426,
    // sc-19502 — one call, because `steps_limit_error` owns both). A FOURTH axis: the
    // three gates around it bound how long the clip is and how much conditioning media it carries,
    // never how it is SAMPLED — so a MiniMax-H3 request at exactly its 5.1667s floor and its one
    // advertised 24 fps, with no references at all, clears every one of them carrying
    // `advanced.steps = 1` — a single Euler jump from pure noise, which is not a fast draft. Until
    // this key existed the constraint could only be written as a manifest comment, and a comment
    // enforces nothing.
    //
    // The unit is MODEL EVALUATIONS everywhere on this seam — `advanced.steps`, `defaults.steps`,
    // `limits.hardMinSteps`, `limits.steps`, and each turbo adapter's declared `sampling.steps`. The
    // MiniMax-H3 engine appends its own terminal sigma grid point (`evaluations + 1`), so no ±1 is
    // applied on this side of the boundary (sc-18726). A gate that read grid points while the worker
    // passed evaluations would be off by one on every model in the family.
    //
    // Rejected, never clamped, for the duration cap's reason: raising the step count for the caller
    // doubles the compute they asked for with no error and no signal.
    //
    // sc-19502 widened this from a floor to also cover an EXACT menu, and the same call site serves
    // both. LTX-2.3 is distilled — 8 baked sigma waypoints, no other renderable count — so an
    // `advanced.steps = 30` request used to 400 late from the candle engine and, on mlx, be accepted
    // and silently rendered at 8 anyway. Both lanes now refuse it, and this gate refuses it here
    // first, before the job is dispatched.
    //
    // Same placement rationale as the three gates above — keyed off the post-preset `model_id` and
    // the resolved entry, with the count read off `job_payload` so the gate judges the value
    // actually enqueued. Unlike duration and fps there is no write-back and no resolved default:
    // `advanced` is a verbatim passthrough map, so an omitted `steps` means the engine picks, which
    // is not a value this gate may invent (see `requested_steps`).
    if let Some(entry) = model_manifest_entry.as_object() {
        if let Some(steps) = job_payload
            .get("advanced")
            .and_then(Value::as_object)
            .and_then(requested_steps)
        {
            if let Some(message) = steps_limit_error(&model_id, steps, entry) {
                return Err(ApiError::bad_request(message));
            }
        }
    }
    // The model's declared `defaults.fps` + `limits.fps`, the other half of `frames = duration ×
    // fps` (sc-12347). The cap above closes the *duration* axis only: a legally-5s mochi_1 request
    // (cap 5 ✓) at 60 fps is 301 frames — double the shipped default's 151 — and `301 % 6 == 1`
    // clears the engine's own check, so nothing downstream refuses it.
    //
    // Resolving the omission is what makes the menu enforceable, not a separate nicety. The DTO
    // used to default fps to a blanket 25, which is off-menu for 7 of the 10 shipped video models,
    // so gating alone would 400 a payload this route constructed itself — and the MCP server omits
    // `fps` whenever its caller does, making that the likeliest non-UI call shape. It was already
    // silently wrong: `generate_video(model = "mochi_1", duration = 5)` rendered at 25 fps, not
    // mochi's declared 30, playing a 30 fps motion prior 20% slow. The web never had this bug —
    // `VideoStudio.jsx:223` reads `defaults.fps` — so this is the API converging on the UI.
    //
    // Same placement rationale as the duration cap: keyed off the post-preset `model_id` and the
    // resolved entry (reading either from the DTO is sc-12300). The RESOLVED rate is written back so
    // the enqueued payload records what was actually rendered — the worker re-resolves identically
    // from the same entry, but a recipe replay reads this row.
    // The model's declared reference-media caps (sc-17160) — `limits.maxReferenceAssets` /
    // `maxSourceClipAssets` / `maxReferenceAudioAssets` / `maxCombinedReferenceAssets`. This is the
    // BINDING half of the caps; `validate_video_job`'s three constants are only the payload-sanity
    // outer bound, for the same reason `1..=30` is for duration — that gate runs before the model is
    // known, and a recipe preset can still replace it (sc-12300).
    //
    // It is what makes raising the image blanket 8 -> 9 for MiniMax-H3 safe for every OTHER video
    // model: `reference_caps` defaults to 8 images / 8 clips / 0 audio / no combined ceiling, so a
    // family that declares nothing behaves byte-for-byte as it did before this key had a reader — a
    // 9th reference to bernini is still refused, just here instead of one layer up. A per-family cap
    // rather than a global bump, which is what the "raising a shared constant affects every video
    // model" warning on the story asks for.
    //
    // Same placement rationale as the duration and fps caps above: keyed off the post-preset
    // `model_id` and the resolved entry. The counts come off the DTO, not `job_payload`, because no
    // preset patches the id lists — they are the caller's media, verbatim.
    if let Some(entry) = model_manifest_entry.as_object() {
        if let Some(message) = reference_limit_error(
            &model_id,
            payload.reference_asset_ids.len(),
            payload.source_clip_asset_ids.len(),
            payload.reference_audio_asset_ids.len(),
            entry,
        ) {
            return Err(ApiError::bad_request(message));
        }
    }
    if let Some(entry) = model_manifest_entry.as_object() {
        let fps = resolve_fps(job_payload.get("fps"), entry);
        if let Some(message) = fps_limit_error(&model_id, fps, entry) {
            return Err(ApiError::bad_request(message));
        }
        job_payload.insert("fps".to_owned(), Value::from(fps));

        // The model's declared `defaults.resolution` — the third dead `defaults.*` key, and the one
        // with no error surface: dimensions COERCE rather than reject, so this was silent by
        // construction. The blanket 768×512 is not in `limits.resolutions` for 8 of the 10 shipped
        // video models, so an MCP `generate_video(model = "mochi_1", prompt = …)` naming no size
        // rendered at 768×512 — a geometry mochi was never trained on — while the web rendered its
        // declared 848×480 from the identical request (`VideoStudio.jsx:234`). Same convergence as
        // fps and duration: the API adopts what the manifest and the dropdown already say.
        //
        // Written back per side only when the caller named none, so a caller's own size is untouched
        // and each side still falls back independently. The stride/area normalization stays in
        // core's `normalized_dimensions` — this records the RESOLVED request, not the final
        // geometry, exactly as it did when the blanket lived in the DTO.
        let (default_width, default_height) = default_resolution(entry).unwrap_or((768, 512));
        if !job_payload.contains_key("width") {
            job_payload.insert("width".to_owned(), Value::from(default_width));
        }
        if !job_payload.contains_key("height") {
            job_payload.insert("height".to_owned(), Value::from(default_height));
        }
    }
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    validate_job_lora_compatibility_with(
        &state,
        Some(&payload.project_id),
        &mut job_payload,
        false,
        Some(&catalogs),
    )
    .await?;
    // **THE NO-LANE GATE (sc-19504).** Last, on the payload actually about to be enqueued, so it
    // judges the post-preset model + the resolved geometry rather than the DTO (sc-12300) — and
    // after the LoRA validation, because a rejected adapter is a better error than "no backend".
    //
    // Every gate above this one asks "is this request well-formed?". This one asks the different
    // question that GH #2074, sc-15328 and sc-19504 all turned on: **will anything ever run it?**
    // A video job no lane claims is not rejected and not failed — it sits `queued` /
    // "Waiting for an available worker." forever, next to an idle worker, with no error and no
    // terminal state. `wan_2_2_i2v_14b` + `first_last_frame` was exactly that: admitted by the
    // `VIDEO_JOB_MODES` allow-list, advertised by the manifest, offered as a Video Studio tab
    // off-Mac, and claimable by neither the MLX engine (the I2V-A14B descriptor declares
    // `conditioning: [Reference]` — no `Keyframe`) nor the candle i2v gate (which requires
    // `mode == "image_to_video"`). Withdrawing the capability stops the TAB; only this stops the
    // HANG, because `VIDEO_JOB_MODES` is global and an MCP / raw-REST / recipe-replay caller can
    // still name any admitted mode against any model.
    //
    // Derived from the real claim predicates `worker_supports_job` consults, never a restated list,
    // so a routing change moves this gate with it — see
    // `video_request_is_claimable_by_any_lane`, which also documents why this is neither a
    // capability gate nor a platform gate.
    if !video_request_is_claimable_by_any_lane(&job_type, &job_payload) {
        let mode = job_payload
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or(payload.mode.as_str());
        return Err(ApiError::bad_request(format!(
            "{model_id} cannot render the \"{mode}\" mode — no backend implements it, so this job \
             would wait for a worker that will never claim it. Choose a mode this model lists in \
             its capabilities, or a model that supports this one."
        )));
    }
    // **THE PLATFORM HALF OF THE SAME DEFECT IS NOT DECIDED HERE (sc-19570).** The gate above is
    // platform-independent and stays that way, so it admits a request some OTHER host's lane would
    // claim — `ltx_2_3` + `image_to_video`, `wan_2_2` + `first_last_frame` and the rest of the
    // measured MLX-only set. Off-Mac nothing can claim those, and the job used to hang forever.
    //
    // sc-19570 first refused them right here with a `400`, which made `POST /api/v1/video/jobs`
    // answer differently on Windows than on macOS for byte-identical bodies. That was ruled out:
    // **an HTTP contract is not platform-dependent.** The status code, response shape and error
    // envelope are the same on every host; what a given machine can actually RENDER is an execution
    // outcome, and it is reported as one.
    //
    // So the platform verdict moved into the job lifecycle —
    // `JobsStore::fail_platform_unreachable_jobs`, run below and again on every claim — and the job
    // reaches a terminal `failed` with a `platform_unreachable:` reason instead of sitting queued.
    // Nothing about the two gates' ORDER or wording is coupled: this one still 400s a mode no lane
    // serves anywhere, on every platform alike.
    let job = create_generation_job(
        state.clone(),
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    // Terminate an unreachable job NOW rather than leaving it for the claim-time sweep. The sweep
    // runs when a worker polls, and the deployments that most need this answer are exactly the ones
    // where no worker ever will (an API-only container, a Windows host whose GPU worker is not
    // installed). Waiting for a poll would reintroduce the hang this story exists to remove, so the
    // response the caller already holds carries the terminal state.
    let job = fail_job_if_platform_unreachable(&state, job).await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) fn video_job_platform(_state: &AppState) -> &'static str {
    #[cfg(test)]
    if let Some(platform) = *_state.video_platform_override.lock() {
        return platform;
    }
    std::env::consts::OS
}

pub(crate) fn ensure_video_model_available_on_platform(
    model_id: &str,
    model_manifest_entry: &Value,
    platform: &str,
) -> Result<(), ApiError> {
    if crate::models::video_model_withdrawn_on_platform(model_id, model_manifest_entry, platform) {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} is available only on macOS and cannot create video jobs on {platform}."
        )));
    }
    Ok(())
}

/// Run the sc-19570 platform-reachability sweep and return `job` as it now stands: unchanged on a
/// Mac and for any pair this host's lane serves, terminal `failed` when nothing here can ever claim
/// it.
///
/// The status code does NOT depend on the verdict — the caller returns `201` either way, on every
/// platform. Only the snapshot's `status` / `error` differ, which is what an execution outcome is.
///
/// It calls the same store method the claim path calls rather than duplicating the decision, so the
/// two entry points can never disagree about which pairs are unreachable.
async fn fail_job_if_platform_unreachable(
    state: &AppState,
    job: JobSnapshot,
) -> Result<JobSnapshot, ApiError> {
    let host_os = state.settings.host_os.clone();
    let failed = store_call(state.clone(), move |store, _timeout| {
        store.fail_platform_unreachable_jobs(&host_os)
    })
    .await?;
    let mut result = job;
    for failed_job in failed {
        crate::jobs::emit_platform_unreachable(&failed_job);
        publish(state, "job.updated", &failed_job);
        if failed_job.id == result.id {
            result = failed_job;
        }
    }
    Ok(result)
}

/// `POST /api/v1/audio/jobs` — the SceneWorks Audio Studio job path (epic 13400 / sc-13404), the
/// audio analogue of [`create_video_job`]. Validates the request, resolves + injects the model's
/// merged manifest entry (so the worker never re-parses the jsonc, exactly as the image/video routes
/// do — sc-1653/sc-12300), asserts the model is an `audio`-type model, and enqueues an
/// `audio_generate` job. The worker builds the `GenerationRequest { audio: Some(AudioParams{..}) }`
/// from this payload and dispatches it through the runtime's candle audio registry to the
/// `Modality::Audio` generator.
///
/// Deliberately simpler than the video route: audio has no recipe-preset / LoRA / duration-fps
/// resolution — its knobs (`voice` / `language` / `targetDurationSecs`) go straight through to the
/// worker's `AudioParams`, and the model's declared duration/voice/language surface is enforced by
/// the shared gen-core validation floor at generate time.
pub(crate) async fn create_audio_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<AudioJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_audio_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    // Resolve the model manifest entry here so the audio worker never re-parses the jsonc — Rust owns
    // manifest parsing/merging (story 1653), mirroring create_image_job / create_video_job. Keyed off
    // the request model (audio has no preset that could replace it, so payload.model is authoritative).
    let model_id = payload.model.clone();
    let model_manifest_entry = resolve_model_manifest_entry(&state, &model_id).await?;
    // The audio route only serves `type: audio` models. An unknown id resolves to `{}` (no type), and
    // a mis-typed id (an image/video model posted here) is rejected up front rather than failing deep
    // in the worker's audio lane with an opaque "no generator registered" — the typed-route contract
    // (a door per media kind). The seeded audio models all declare `type: audio` (sc-13402).
    let entry_type = model_manifest_entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if entry_type != "audio" {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} is not an audio model (type: \"audio\" required)"
        )));
    }
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    // Voice Clone (sc-13411 C4): the two-call chain runs a SECOND model — the base TTS (Kokoro) whose
    // speech the selected converter (OpenVoice V2) re-timbres. Resolve + inject its manifest entry too so
    // the worker resolves BOTH snapshots without re-parsing the jsonc, exactly as the primary model entry
    // is injected. Only when the request actually carries a reference voice (the voice-clone discriminator).
    if payload
        .reference_audio_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty())
    {
        let base_model_id = payload.base_model.clone();
        let base_entry = resolve_model_manifest_entry(&state, &base_model_id).await?;
        let base_type = base_entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if base_type != "audio" {
            return Err(ApiError::bad_request(format!(
                "Base voice model {base_model_id} is not an audio model (type: \"audio\" required)"
            )));
        }
        job_payload.insert("baseModelManifestEntry".to_owned(), base_entry);
    }
    let job = create_generation_job(
        state,
        JobType::AudioGenerate,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// A resolved `duration` in the payload's `ContractNumber` (= `serde_json::Number`) shape: an
/// integral value stays integral.
///
/// `pub(crate)` only so `a_fractional_resolved_duration_keeps_the_manifests_own_decimal` can assert
/// the encoding directly rather than through a whole enqueue round-trip.
///
/// The wire has always carried `"duration": 5`, not `5.0` — `ContractNumber` preserves whatever the
/// caller sent, and every shipped `defaults.duration` is a whole number. Writing an `f32` straight
/// in flattens that (`Value::from(5.0_f32)` is `5.0`), a gratuitous contract change that the
/// payload-normalization tests correctly catch.
pub(crate) fn contract_number(value: f32) -> Value {
    if value.fract() == 0.0 {
        return Value::from(value as i64);
    }
    // A FRACTIONAL default must round-trip to the decimal the manifest declared, not to the f32's
    // binary expansion (sc-17159). `Value::from(f32)` widens to `f64`, so MiniMax-H3's
    // `defaults.duration: 5.1667` was enqueued as `5.1666998863220215` — numerically harmless
    // (`VideoRequest` reads it back as the same f32 and lands on frame 124) but not EQUAL to any of
    // the fourteen values in that model's `limits.durations`, which is the menu the duration
    // dropdown preselects against and the set a recipe replay is compared to. Every video model
    // before this family had a whole-number default, so the fractional branch had never carried a
    // shipped value.
    //
    // `f32::to_string` emits the shortest decimal that reproduces the f32 exactly ("5.1667"), so
    // parsing that back yields the manifest's own number. The fallback keeps the historical shape
    // for any value whose text somehow does not re-parse.
    serde_json::from_str(&value.to_string()).unwrap_or_else(|_| Value::from(value))
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
        JobType::VectorGenerate => Some("/api/v1/image/vectorize/jobs"),
        JobType::VideoGenerate
        | JobType::VideoExtend
        | JobType::VideoBridge
        | JobType::PersonReplace => Some("/api/v1/video/jobs"),
        // The audio route resolves + injects the model's manifest entry (sc-13404), so — like the
        // image/video routes — an `audio_generate` job enqueued raw through the generic
        // `POST /api/v1/jobs` would reach the worker without its entry. One door per generation kind.
        JobType::AudioGenerate => Some("/api/v1/audio/jobs"),
        _ => None,
    }
}

#[cfg(test)]
mod decoder_selection_tests {
    use super::*;

    fn manifest(available: bool) -> Value {
        json!({
            "id": "qwen_image",
            "decoders": { "byBackend": {
                "mlx": [{
                    "id": "wan_2_1_vae",
                    "componentId": "vae",
                    "available": available
                }],
                "candle": []
            } }
        })
    }

    /// A row whose `mlx` and `candle` lists advertise DIFFERENT installed decoders, so which list
    /// the gate consulted is observable from the error alone. The shipped facts happen to declare
    /// no candle decoders at all today, which would make "consulted candle" and "consulted nothing"
    /// indistinguishable — this fixture keeps the assertion about the backend selection itself.
    fn split_manifest() -> Value {
        json!({
            "id": "qwen_image",
            "decoders": { "byBackend": {
                "mlx": [{ "id": "mlx_only_vae", "componentId": "vae", "available": true }],
                "candle": [{ "id": "candle_only_vae", "componentId": "vae", "available": true }]
            } }
        })
    }

    fn payload(value: Value) -> JsonObject {
        value.as_object().expect("payload object").clone()
    }

    #[test]
    fn decoder_submission_is_default_off_and_fails_closed() {
        assert!(validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({})),
            &manifest(false)
        )
        .is_ok());
        assert!(validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({ "advanced": { "decoder": "native" } })),
            &manifest(false),
        )
        .is_ok());
        assert!(validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({ "advanced": { "decoder": "wan_2_1_vae" } })),
            &manifest(false),
        )
        .is_err());
        assert!(validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({ "advanced": { "decoder": "wan_2_1_vae" } })),
            &manifest(true),
        )
        .is_ok());
        // The same installed MLX option on the candle lane: the shipped facts declare no candle
        // decoders, so it fails closed rather than deferring the rejection to the worker.
        assert!(validate_selected_decoder_for_manifest(
            "candle",
            &payload(json!({ "advanced": { "decoder": "wan_2_1_vae" } })),
            &manifest(true),
        )
        .is_err());
        assert!(validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({ "advanced": { "decoder": "wan_2_1_vae" } })),
            &json!({ "id": "wan_2_2_t2v_a14b" }),
        )
        .is_err());
    }

    /// Both directions of the backend selection, against one row that advertises a different
    /// installed decoder per lane. This is the half a bare `cfg!(target_os = "macos")` got wrong
    /// (sc-18420): under `SCENEWORKS_CANDLE_REQUIRED` on macOS the gate consulted the MLX list
    /// while the job ran on candle, admitting the MLX-only id and refusing the candle-valid one.
    #[test]
    fn the_gate_consults_only_the_executing_backends_option_list() {
        for (backend, valid, foreign) in [
            ("mlx", "mlx_only_vae", "candle_only_vae"),
            ("candle", "candle_only_vae", "mlx_only_vae"),
        ] {
            assert!(
                validate_selected_decoder_for_manifest(
                    backend,
                    &payload(json!({ "advanced": { "decoder": valid } })),
                    &split_manifest(),
                )
                .is_ok(),
                "{backend} must admit its own installed option {valid}"
            );
            let error = validate_selected_decoder_for_manifest(
                backend,
                &payload(json!({ "advanced": { "decoder": foreign } })),
                &split_manifest(),
            )
            .expect_err("the other lane's option must not be admitted")
            .detail;
            assert!(
                error.contains("is not compatible with qwen_image"),
                "{backend} got: {error}"
            );
        }
    }

    #[test]
    fn alternate_decoder_and_pid_are_mutually_exclusive() {
        let error = validate_selected_decoder_for_manifest(
            "mlx",
            &payload(json!({
                "advanced": { "decoder": "wan_2_1_vae", "usePid": true }
            })),
            &manifest(true),
        )
        .unwrap_err()
        .detail;
        assert!(error.contains("exactly one decoder"), "got: {error}");
    }
}

/// End-to-end enqueue coverage for an IMPORTED plan-backed Wan checkpoint (epic 20398, sc-20651).
///
/// The unit-level claim predicates live in `sceneworks-core`; this module asserts the thing the
/// user actually experiences — that `POST /api/v1/video/jobs` answers `201` and the job is really
/// on the queue, rather than the `400` the sc-19504 no-lane gate returned for every imported Wan
/// checkpoint while the video claim predicates had no `importPlan` awareness.
#[cfg(test)]
mod plan_backed_video_enqueue_tests {
    use crate::tests::support::*;

    /// The manifest set `test_settings` reads, with `user.models.jsonc` supplied by the caller.
    fn write_manifests(config_dir: &std::path::Path, user_models: &str) {
        std::fs::create_dir_all(config_dir).expect("manifest dir creates");
        for (name, body) in [
            (
                "builtin.models.jsonc",
                r#"{ "schemaVersion": 1, "models": [] }"#,
            ),
            ("user.models.jsonc", user_models),
            (
                "builtin.loras.jsonc",
                r#"{ "schemaVersion": 1, "loras": [] }"#,
            ),
            ("user.loras.jsonc", r#"{ "schemaVersion": 1, "loras": [] }"#),
            (
                "builtin.recipe-presets.jsonc",
                r#"{ "schemaVersion": 1, "presets": [] }"#,
            ),
            (
                "user.recipe-presets.jsonc",
                r#"{ "schemaVersion": 1, "presets": [] }"#,
            ),
        ] {
            std::fs::write(config_dir.join(name), body).expect("manifest writes");
        }
    }

    /// An app whose catalog holds ONE imported `wan-video` model. `import_plan` is spliced in
    /// verbatim so the control case can drop it and change nothing else.
    fn app_with_imported_wan(temp_dir: &tempfile::TempDir, import_plan: &str) -> axum::Router {
        write_manifests(
            &temp_dir.path().join("config/manifests"),
            &format!(
                r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "imported_wan_2_2_abc123",
                    "name": "Imported Wan 2.2",
                    "type": "video",
                    "family": "wan-video"{import_plan}
                  }}]
                }}"#
            ),
        );
        create_app(test_settings(temp_dir)).expect("app creates")
    }

    const IMPORT_PLAN: &str = r#","importPlan": { "checkpointId": "ckpt_wan_abc123" }"#;

    async fn create_project(app: &axum::Router) {
        let (status, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Imported Wan" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "project create: {project}");
    }

    #[tokio::test]
    async fn a_plan_backed_wan_checkpoint_enqueues_a_text_to_video_job() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = app_with_imported_wan(&temp_dir, IMPORT_PLAN);
        create_project(&app).await;

        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "imported_wan_2_2_abc123",
                "mode": "text_to_video",
                "prompt": "a fox in the rain"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "enqueue answered: {job}");
        assert_eq!(job["type"], "video_generate");
        // Not merely accepted — still QUEUED after `fail_job_if_platform_unreachable` has run, so
        // the row is genuinely waiting for a worker rather than terminal-failed on the way out.
        assert_eq!(job["status"], "queued", "{job}");
        // The plan identity the worker's Wan lane resolves through is on the enqueued payload.
        assert_eq!(
            job["payload"]["modelManifestEntry"]["importPlan"]["checkpointId"],
            "ckpt_wan_abc123"
        );

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 1, "the job must be on the queue");
    }

    /// The control that makes the test above an assertion about the PLAN. The identical catalog
    /// entry with no `importPlan` is claimed by no lane — its model id is in no static routed list
    /// — so the no-lane gate 400s it, which is exactly what the plan-backed request used to get.
    #[tokio::test]
    async fn the_same_imported_wan_entry_without_a_plan_is_still_refused_by_the_no_lane_gate() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = app_with_imported_wan(&temp_dir, "");
        create_project(&app).await;

        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "imported_wan_2_2_abc123",
                "mode": "text_to_video",
                "prompt": "a fox in the rain"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "answered: {error}");
        assert!(
            error.to_string().contains("no backend implements it"),
            "the no-lane gate must be what refused it: {error}"
        );

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 0);
    }

    /// A mode the plan-backed Wan lane does not serve is refused at the gate that can still explain
    /// why (a `400`), rather than enqueued and later refused by the worker — the claim predicate and
    /// `resolve_candle_video_route`'s plan-backed arm agree on the served set.
    #[tokio::test]
    async fn a_plan_backed_wan_checkpoint_is_refused_on_a_mode_its_lane_does_not_serve() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = app_with_imported_wan(&temp_dir, IMPORT_PLAN);
        create_project(&app).await;

        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "imported_wan_2_2_abc123",
                "mode": "image_to_video",
                "prompt": "a fox in the rain",
                "sourceAssetId": "asset-1"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "answered: {error}");

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 0);
    }
}
