use super::*;

pub(crate) async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> Result<Json<Vec<JobSnapshot>>, ApiError> {
    if let Some(status) = &query.status {
        if !JOB_STATUSES.contains(&status.as_str()) {
            return Err(ApiError::bad_request("Unsupported job status"));
        }
    }
    let (sweep, jobs) = store_call(state.clone(), move |store, timeout| {
        let sweep = store.mark_stale_workers_interrupted(timeout)?;
        let jobs = store.list_jobs(
            query.project_id.as_deref(),
            query.status.as_deref(),
            query.limit.unwrap_or(100),
        );
        Ok((sweep, jobs))
    })
    .await?;
    handle_stale_sweep(&state, &sweep);
    let jobs = jobs?;
    Ok(Json(jobs))
}

/// Worker → API write of a job's structured generation metrics (epic 10402).
/// Posted once on completion; upserts (merges) into the `generation_metrics`
/// table and echoes the stored block back.
pub(crate) async fn upsert_job_metrics(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<GenerationMetrics>,
) -> Result<Json<GenerationMetrics>, ApiError> {
    let metrics = store_call(state, move |store, _timeout| {
        store.upsert_generation_metrics(&job_id, &payload)?;
        Ok::<GenerationMetrics, JobsStoreError>(payload)
    })
    .await?;
    Ok(Json(metrics))
}

/// Read a single job's structured metrics (epic 10402). Returns `null` (200)
/// when the job never recorded metrics — friendlier for the detail view than a
/// 404 for the common "older job" case.
pub(crate) async fn get_job_metrics(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Option<GenerationMetrics>>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| {
            store.get_generation_metrics(&job_id)
        })
        .await?,
    ))
}

/// Aggregate metrics feed powering the Generation Stats comparison charts
/// (epic 10402). Newest first, filterable by job type / model / quant.
pub(crate) async fn list_metrics(
    State(state): State<AppState>,
    Query(query): Query<MetricsQuery>,
) -> Result<Json<Vec<GenerationMetricsRow>>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| {
            store.list_generation_metrics(
                query.job_type.as_deref(),
                query.model.as_deref(),
                query.quant.as_deref(),
                query.limit.unwrap_or(2000),
            )
        })
        .await?,
    ))
}

pub(crate) async fn create_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<JobCreateRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if matches!(payload.job_type, JobType::CatalogAnalysis) {
        return Err(ApiError::bad_request(
            "catalog_analysis jobs must be created through POST /api/v1/catalogs/:catalog_id/analyze",
        ));
    }
    // A generation job type must be created through its typed route, which resolves the
    // model's merged manifest entry into the payload (and validates the request). This
    // route is the raw queue primitive: it enqueues `job_type` + payload verbatim, so a
    // generation job through this door carries no `modelManifestEntry` and renders at the
    // wrong geometry with no error (sc-12305). See `typed_generation_route`.
    if let Some(route) = typed_generation_route(&payload.job_type) {
        return Err(ApiError::bad_request(format!(
            "{} jobs must be created via POST {route}. That route resolves the model's \
             manifest entry — its repo, quant and geometry limits — and validates the \
             request; POST /api/v1/jobs enqueues the payload verbatim, so the job would run \
             with none of it.",
            payload.job_type.as_str()
        )));
    }
    validate_raw_job_payload(&state, &payload.job_type, &payload.payload).await?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: payload.job_type,
            project_id: payload.project_id,
            project_name: payload.project_name,
            payload: payload.payload,
            requested_gpu: payload.requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn claim_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let mlx_required = state.settings.mlx_required;
    let enforce_unsupported = state.settings.mlx_enforce_unsupported;
    let candle_required = state.settings.candle_required;
    let candle_enforce = state.settings.candle_enforce_unsupported;
    let (stale_sweep, claim_result) = store_call(state.clone(), move |store, timeout| {
        let stale_sweep = store.mark_stale_workers_interrupted(timeout)?;
        let claim_result = (|| {
            // macOS MLX-required (sc-3483): before claiming, fail any MLX-eligible job left
            // stranded because no live `mlx` worker took it within the grace window — reusing
            // the worker timeout as that window. No-op when the flag is off.
            let stranded = store.fail_stranded_mlx_jobs(mlx_required, timeout)?;
            // macOS MLX-required + enforce (sc-3484): fail any queued job the Rust/MLX flow
            // can't run. No-op in warn mode (the default) — the gap is logged at claim instead.
            let unsupported = store.fail_unsupported_mlx_jobs(mlx_required, enforce_unsupported)?;
            // Off-Mac candle-required (sc-5502, epic 5483): the Windows/Linux twins of the two
            // sweeps above — fail any candle-eligible job stranded with no live candle worker, and
            // (enforce) any queued job the candle/CUDA flow can't serve. Both no-op when the flag
            // is off (the default), so normal capability routing is unaffected.
            let candle_stranded = store.fail_stranded_candle_jobs(candle_required, timeout)?;
            let candle_unsupported =
                store.fail_unsupported_candle_jobs(candle_required, candle_enforce)?;
            let (job, decision) = store.claim_next_job_routed(&payload.worker_id, mlx_required)?;
            Ok::<_, JobsStoreError>((
                job,
                decision,
                stranded,
                unsupported,
                candle_stranded,
                candle_unsupported,
            ))
        })();
        Ok((stale_sweep, claim_result))
    })
    .await?;
    handle_stale_sweep(&state, &stale_sweep);
    let (response, decision, stranded, unsupported, candle_stranded, candle_unsupported) =
        claim_result?;
    for job in &stranded {
        emit_mlx_unavailable(job);
        publish(&state, "job.updated", job);
    }
    for (job, reason) in &unsupported {
        emit_mlx_unsupported(job, reason, "enforce");
        publish(&state, "job.updated", job);
    }
    for job in &candle_stranded {
        emit_candle_unavailable(job);
        publish(&state, "job.updated", job);
    }
    for (job, reason) in &candle_unsupported {
        emit_candle_unsupported(job, reason, "enforce");
        publish(&state, "job.updated", job);
    }
    if let Some(decision) = &decision {
        emit_route_decision(decision);
    }
    if let Some(job) = &response {
        // Warn-only (sc-3484): an unsupported job claimed by a non-MLX GPU descriptor on a Mac —
        // log the gap once so the inventory materializes.
        // In enforce mode such a job was already failed above and never reaches here.
        if mlx_required && !enforce_unsupported {
            if let Err(reason) = mac_rust_supported(job) {
                emit_mlx_unsupported(job, &reason, "warn");
            }
        }
        // Warn-only (sc-5502): the off-Mac candle twin — log the gap once if a non-candle GPU
        // descriptor claims it, so the off-Mac port-or-drop inventory materializes. In enforce mode
        // such a job was already failed above and never reaches here.
        if candle_required && !candle_enforce {
            if let Err(reason) = candle_supported(job) {
                emit_candle_unsupported(job, &reason, "warn");
            }
        }
        publish(&state, "job.updated", job);
    }
    if response.is_some()
        || !stale_sweep.workers.is_empty()
        || !stale_sweep.jobs.is_empty()
        || !stranded.is_empty()
        || !unsupported.is_empty()
        || !candle_stranded.is_empty()
        || !candle_unsupported.is_empty()
    {
        // claim_job already ran mark_stale_workers_interrupted above (its own
        // transaction), so refresh the queue WITHOUT sweeping a second time
        // (sc-8889 / F-087). The old publish_queue path swept again on every
        // claim; that second sweep found nothing (the first already interrupted
        // the stale jobs) yet still cost a blocking round-trip.
        publish_queue_skip_sweep(&state).await?;
    }
    Ok(Json(ClaimResponse {
        job: response,
        extra: Default::default(),
    }))
}

/// Emit the macOS `mlx_unsupported` gap event (epic 3482 / sc-3484) as a structured JSON line
/// for the desktop stdout capture + headless `GET /api/v1/logs` buffer (sc-3447/3451/3453).
/// `mode` is `"enforce"` (the job was failed terminal) or `"warn"` (logged while it remains
/// queued for a compatible native worker). The body is the feature-precise [`UnsupportedReason`] — model/feature/detail/
/// suggestedEpic — so the Logs surface and the gap inventory name the exact port-or-drop work.
fn emit_mlx_unsupported(job: &JobSnapshot, reason: &UnsupportedReason, mode: &str) {
    let mut value = serde_json::to_value(reason).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event".to_owned(),
            Value::String("mlx_unsupported".to_owned()),
        );
        object.insert("mode".to_owned(), Value::String(mode.to_owned()));
        object.insert("jobId".to_owned(), Value::String(job.id.clone()));
        object.insert(
            "jobType".to_owned(),
            Value::String(job.job_type.as_str().to_owned()),
        );
    }
    // Through the tracing backbone: the stdout JSON layer feeds the desktop capture
    // and the API's own ring buffer (GET /api/v1/logs) via the session-log layer.
    sceneworks_core::observability::emit_event(tracing::Level::INFO, value);
}

/// Emit the macOS `mlx_unavailable` terminal-routing event as a structured JSON line for
/// the desktop's stdout capture + the headless `GET /api/v1/logs` buffer (sc-3447/3451/3453).
/// Mirrors [`emit_route_decision`]: this is the System → Logs surface that turns "no MLX
/// worker took the job" into a named, actionable line instead of a job silently stuck or
/// entering the legacy MPS compatibility branch (sc-3483). `reason` carries the full actionable
/// error set on the job.
fn emit_mlx_unavailable(job: &JobSnapshot) {
    let model = job.payload.get("model").and_then(Value::as_str);
    sceneworks_core::observability::emit_event(
        tracing::Level::INFO,
        json!({
            "event": "mlx_unavailable",
            "jobId": job.id,
            "jobType": job.job_type.as_str(),
            "model": model,
            "reason": job.error,
        }),
    );
}

/// Emit the off-Mac `candle_unsupported` gap event (sc-5502, epic 5483) — the candle twin of
/// [`emit_mlx_unsupported`]. `mode` is `"enforce"` (the job was failed terminal) or `"warn"`
/// (logged if a non-candle GPU descriptor claimed it). The body is the feature-precise
/// [`UnsupportedReason`] so the Logs surface + the off-Mac gap inventory name the exact
/// port-or-drop work.
fn emit_candle_unsupported(job: &JobSnapshot, reason: &UnsupportedReason, mode: &str) {
    let mut value = serde_json::to_value(reason).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event".to_owned(),
            Value::String("candle_unsupported".to_owned()),
        );
        object.insert("mode".to_owned(), Value::String(mode.to_owned()));
        object.insert("jobId".to_owned(), Value::String(job.id.clone()));
        object.insert(
            "jobType".to_owned(),
            Value::String(job.job_type.as_str().to_owned()),
        );
    }
    sceneworks_core::observability::emit_event(tracing::Level::INFO, value);
}

/// Emit the off-Mac `candle_unavailable` terminal-routing event (sc-5502, epic 5483) — the candle
/// twin of [`emit_mlx_unavailable`]: the System → Logs surface that turns "no candle worker took
/// the job" into a named, actionable line instead of a job silently stuck. `reason` carries the
/// full actionable error set on the job.
fn emit_candle_unavailable(job: &JobSnapshot) {
    let model = job.payload.get("model").and_then(Value::as_str);
    sceneworks_core::observability::emit_event(
        tracing::Level::INFO,
        json!({
            "event": "candle_unavailable",
            "jobId": job.id,
            "jobType": job.job_type.as_str(),
            "model": model,
            "reason": job.error,
        }),
    );
}

/// Emit the GPU routing decision as a structured JSON line on the API's stdout
/// (sc-3449). The desktop wrapper captures this into `api.log` + the in-app Logs buffer,
/// so *which backend ran a job* is explained at claim time rather than inferred from
/// archaeology. Shape mirrors the worker's `emit_worker_event` events
/// (`event` + `reportedAt` + payload).
fn emit_route_decision(decision: &RouteDecision) {
    let mut value = serde_json::to_value(decision).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event".to_owned(),
            Value::String("gpu_route_decision".to_owned()),
        );
    }
    // Emitted through the tracing backbone: the stdout JSON layer reaches the desktop
    // wrapper's capture (sc-3451) + api.log, and the session-log layer records it into
    // the API's own buffer for the headless `GET /api/v1/logs` (sc-3453).
    sceneworks_core::observability::emit_event(tracing::Level::INFO, value);
}

pub(crate) async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| store.get_job(&job_id)).await?,
    ))
}

pub(crate) async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.cancel_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

pub(crate) async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let mut payload = retry_job_request_from_body(request).await?;
    payload.payload_changes = validate_and_canonicalize_merged_generation_payload(
        &state,
        &job_id,
        &payload.payload_changes,
    )
    .await?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.retry_job(
            &job_id,
            RetryJob {
                payload_changes: payload.payload_changes,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn retry_job_request_from_body(request: AxumRequest) -> Result<RetryJobRequest, ApiError> {
    let bytes = to_bytes(request.into_body(), 1024 * 1024)
        .await
        .map_err(|error| {
            ApiError::bad_request(format!("Unable to read retry request body: {error}"))
        })?;
    if bytes.iter().all(|byte| byte.is_ascii_whitespace()) {
        return Ok(RetryJobRequest::default());
    }
    serde_json::from_slice::<RetryJobRequest>(&bytes)
        .map_err(|error| ApiError::bad_request(format!("Invalid retry request body: {error}")))
}

pub(crate) async fn duplicate_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(mut payload): ApiJson<DuplicateJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    payload.payload_changes = validate_and_canonicalize_merged_generation_payload(
        &state,
        &job_id,
        &payload.payload_changes,
    )
    .await?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.duplicate_job(
            &job_id,
            DuplicateJob {
                payload_changes: payload.payload_changes,
                requested_gpu: payload.requested_gpu,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Validate and canonicalize the exact payload a retry/duplicate will enqueue. Existing job
/// payloads are immutable, so reading and merging before the create transaction cannot race with a
/// payload update. Returning the complete merged payload is intentional: the store's shallow merge
/// then persists this canonical object rather than allowing a nested `advanced.controlWeights`
/// replacement to bypass the typed image route's authorization boundary.
async fn validate_and_canonicalize_merged_generation_payload(
    state: &AppState,
    job_id: &str,
    payload_changes: &JsonObject,
) -> Result<JsonObject, ApiError> {
    let job_id = job_id.to_owned();
    let job = store_call(state.clone(), move |store, _timeout| store.get_job(&job_id)).await?;
    let job_type = job.job_type.clone();
    let project_id = job.project_id.clone();
    let mut merged = job.payload;
    merged.extend(payload_changes.clone());
    if generation_job_model_is_path_backed(&job_type) {
        validate_payload_model(&merged)?;
    } else {
        validate_raw_job_payload(state, &job_type, &merged).await?;
    }
    if matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        if let Some(advanced) = merged.get("advanced").and_then(Value::as_object) {
            validate_image_pose_count(advanced)?;
        }
        crate::control_overlays::resolve_control_overlay_selection(
            state,
            project_id.as_deref(),
            &mut merged,
        )
        .await?;
    }
    Ok(merged)
}

/// Jobs created through the raw queue route do not pass a typed request validator. Keep this
/// inventory aligned with worker payload fields that reach filesystem model resolution:
/// `image_upscale` and `prompt_refine` consume `model`; the model-management jobs consume
/// `modelId`, and `model_convert.outputDir` selects its final install location. Other raw job
/// payloads may contain descriptive model metadata, but are deliberately absent unless that field
/// selects a filesystem path.
async fn validate_raw_job_payload(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    if matches!(job_type, JobType::ImageUpscale | JobType::PromptRefine) {
        validate_payload_model(payload)?;
    }
    if matches!(
        job_type,
        JobType::ModelDownload | JobType::ModelImport | JobType::ModelConvert
    ) {
        validate_payload_model_id(payload)?;
    }
    // Licence-acknowledgment gate for the FETCHING job types (sc-17227), keyed on the payload's
    // `repo`/`sourceUrl` rather than on a model id.
    //
    // This route enqueues `job_type` + payload VERBATIM: `run_model_download_job` reads `repo` /
    // `files` / `revision` straight out of the payload with no catalog lookup anywhere in between,
    // and `validate_payload_model_id` above only FORMAT-checks `modelId` — which the payload need
    // not carry at all. So a `model_download` posted here fetched `MiniMaxAI/MiniMax-H3` and was
    // answered 201 while the typed `POST /api/v1/models/:id/download` answered 403 for the same
    // bytes. Rejecting the whole job type instead would break retry/duplicate, which re-validate a
    // stored `model_download` payload through this same function; the repo-keyed check lets an
    // already-authorized download retry (the typed route stamps `licenseAcknowledged` onto the job)
    // while still refusing a fresh unacknowledged one.
    //
    // The LoRA download/import types are in the list because they take the same `repo` + `files`
    // shape through `run_lora_download_job` and would otherwise be the identical bypass wearing a
    // different `job_type`. Their own typed routes (`/loras/:id/download`, `/loras/import`) are NOT
    // gated — no LoRA declares a licence acknowledgment today, so there is nothing for them to
    // enforce yet; see `docs/minimax-h3-use-restriction-safeguards.md` for the boundary.
    if matches!(
        job_type,
        JobType::ModelDownload | JobType::ModelImport | JobType::LoraDownload | JobType::LoraImport
    ) {
        crate::models::ensure_job_payload_license_acknowledged(state, payload).await?;
    }
    if matches!(job_type, JobType::ModelConvert) {
        let output_dir = payload
            .get("outputDir")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("model_convert outputDir must be a string"))?;
        sceneworks_worker::resolve_model_convert_output(&state.settings.data_dir, output_dir)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
    }
    Ok(())
}

/// Generation kinds whose `model` selects weights and therefore reaches worker path resolution.
/// This is deliberately separate from `typed_generation_route`: VQA/interleave have typed routes
/// but need no manifest injection, so the raw-route predicate intentionally excludes them.
fn generation_job_model_is_path_backed(job_type: &JobType) -> bool {
    matches!(
        job_type,
        JobType::ImageGenerate
            | JobType::ImageEdit
            | JobType::ImageVqa
            | JobType::ImageInterleave
            | JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
            | JobType::AudioGenerate
    )
}

fn validate_payload_model(payload: &JsonObject) -> Result<(), ApiError> {
    if let Some(model) = payload.get("model") {
        let model = model
            .as_str()
            .ok_or_else(|| ApiError::bad_request("model must be a string"))?;
        validate_model_id(model)?;
    }
    Ok(())
}

fn validate_payload_model_id(payload: &JsonObject) -> Result<(), ApiError> {
    if let Some(model_id) = payload.get("modelId") {
        let model_id = model_id
            .as_str()
            .ok_or_else(|| ApiError::bad_request("modelId must be a string"))?;
        validate_model_id(model_id)?;
    }
    Ok(())
}

/// Clear completed items from the queue (sc-12231, issue #1556). Soft-hides every
/// terminal (completed / failed / canceled / interrupted) job so it drops off the
/// operator's queue list + counts, optionally scoped to one project via the
/// request body. The job rows are kept so the Generation Stats feed and the
/// generated assets are untouched (see `JobsStore::clear_terminal_jobs`). Returns
/// the cleared ids so the client can prune them from its live queue immediately.
pub(crate) async fn clear_jobs(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ClearJobsRequest>,
) -> Result<Json<ClearJobsResponse>, ApiError> {
    let cleared_ids = store_call(state.clone(), move |store, _timeout| {
        store.clear_terminal_jobs(payload.project_id.as_deref())
    })
    .await?;
    if !cleared_ids.is_empty() {
        publish(&state, "jobs.cleared", &json!({ "ids": &cleared_ids }));
    }
    // Republish the queue so every subscriber's status counts drop the cleared
    // jobs; the acting client also prunes them locally from the returned ids.
    publish_queue(&state).await?;
    Ok(Json(ClearJobsResponse {
        cleared: cleared_ids.len(),
        cleared_ids,
        extra: Default::default(),
    }))
}

/// Cancel every pending (not-yet-started) item in the queue (sc-13448) — the bulk
/// analog of the per-job cancel fast path. A `queued` / `pending_caption` job has no
/// worker to acknowledge the cancel, so each is flipped straight to terminal
/// `canceled` in one pass (see `JobsStore::cancel_pending_jobs`), optionally scoped
/// to one project via the request body (matching the queue's project filter). Active
/// (worker-owned) jobs are left untouched — those cancel one at a time so the owning
/// worker acknowledges. Broadcasts `job.updated` per canceled job so every SSE client
/// flips the card, plus one queue refresh, and returns the updated snapshots so the
/// acting client updates instantly.
pub(crate) async fn cancel_pending_jobs(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<CancelPendingJobsRequest>,
) -> Result<Json<CancelPendingJobsResponse>, ApiError> {
    let jobs = store_call(state.clone(), move |store, _timeout| {
        store.cancel_pending_jobs(payload.project_id.as_deref())
    })
    .await?;
    // Per-job `job.updated` so every subscriber's card flips to Cancelled (the queue
    // summary alone only updates counts, not individual cards), then one queue refresh
    // for the status counts. The pending set is bounded by what a user queued, so the
    // fan-out is small.
    for job in &jobs {
        publish(&state, "job.updated", job);
    }
    publish_queue(&state).await?;
    Ok(Json(CancelPendingJobsResponse {
        canceled: jobs.len(),
        jobs,
        extra: Default::default(),
    }))
}

/// Clear a single completed item from the queue (sc-12231, issue #1556) — the
/// per-card "×" dismiss. Soft-hides one terminal job (see `JobsStore::clear_job`)
/// so it drops off the queue list + counts while its row (and generated assets)
/// stay put. Rejects a non-terminal job with 400 (cancel an in-flight job
/// instead). Returns the updated snapshot and republishes the queue.
pub(crate) async fn clear_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.clear_job(&job_id)
    })
    .await?;
    publish(&state, "jobs.cleared", &json!({ "ids": [job.id.clone()] }));
    publish_queue(&state).await?;
    Ok(Json(job))
}

pub(crate) async fn update_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<ProgressRequest>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let progress = number_to_f64(&payload.progress, "progress")?;
    let eta_seconds = optional_number_to_f64(payload.eta_seconds.as_ref(), "etaSeconds")?;
    let peak_gpu_memory_pct =
        optional_number_to_f64(payload.peak_gpu_memory_pct.as_ref(), "peakGpuMemoryPct")?;
    let peak_gpu_load_pct =
        optional_number_to_f64(payload.peak_gpu_load_pct.as_ref(), "peakGpuLoadPct")?;
    let submitted_result = payload.result;

    // Test-only one-shot rendezvous that makes a competing terminal transition
    // deterministic. Take the hook before awaiting so the competing request is
    // not blocked by the same mutex.
    #[cfg(test)]
    let progress_barrier = state.progress_before_accept_once.lock().take();
    #[cfg(test)]
    if let Some(barrier) = progress_barrier {
        barrier.wait().await;
        barrier.wait().await;
    }

    // Win the authoritative ownership/terminal-state race before performing
    // any async catalog or project writes. A snapshot precheck is insufficient:
    // cancel/sweep/a competing terminal report can commit while those writes
    // await and make the later progress update return 409 after side effects.
    let accepted = store_call(state.clone(), {
        let job_id = job_id.clone();
        let result = submitted_result.clone();
        let worker_id = payload.worker_id.clone();
        let status = payload.status.clone();
        let stage = payload.stage;
        let message = payload.message.clone();
        let error = payload.error.clone();
        let backend = payload.backend.clone();
        move |store, _timeout| {
            store.update_job_progress_with_outcome(
                &job_id,
                ProgressUpdate {
                    status,
                    stage,
                    progress,
                    message,
                    error,
                    result,
                    eta_seconds,
                    peak_gpu_memory_pct,
                    peak_gpu_load_pct,
                    backend,
                    worker_id,
                },
            )
        }
    })
    .await?;
    let status_changed = accepted.previous_status != accepted.job.status;
    let terminal_side_effects_pending = accepted.side_effects_pending;
    let accepted_job = accepted.job.clone();
    let mut publish_job_update = accepted.applied;
    let mut job = accepted.job;

    // Model workers mutate install receipts, imported manifests, or converted
    // output directories before reporting a terminal status. Advance the
    // shared catalog generation for every accepted terminal outcome (including
    // failures/cancellation, which may leave an incomplete cache) so the next
    // `/models`, preset, or job-validation caller observes the new filesystem.
    if accepted.applied && terminal_model_job_changes_catalog(&job.job_type, &job.status) {
        state.model_catalog_cache.invalidate();
    }

    if terminal_side_effects_pending {
        // The request that accepted the terminal state gets the first chance to
        // drain its durable handoff. If it errors or the process dies, the
        // production recovery loop below discovers the same pending row without
        // depending on a worker retry.
        job = match process_pending_terminal_progress_side_effects(&state, &job_id).await {
            Ok(Some(job)) => job,
            Ok(None) => {
                store_call(state.clone(), {
                    let job_id = job_id.clone();
                    move |store, _timeout| store.get_job(&job_id)
                })
                .await?
            }
            Err(error) => {
                // Terminal acceptance already committed before the fallible
                // catalog/project writes. Publish that authoritative transition
                // even though the request returns 500, otherwise the UI retains
                // the previous active job and queue counts until an unrelated
                // transition occurs. A later retry/recovery has no status change,
                // so it publishes only the augmented job snapshot below.
                if status_changed {
                    publish(&state, "job.updated", &job);
                    if let Err(queue_error) = publish_queue(&state).await {
                        tracing::error!(
                            event = "terminal_progress_queue_publish_failed",
                            job_id,
                            status = queue_error.status.as_u16(),
                            detail = %queue_error.detail,
                            "terminal progress committed but its queue refresh failed"
                        );
                    }
                }
                return Err(error);
            }
        };
        publish_job_update |= job != accepted_job;
    } else if accepted.applied {
        job = apply_progress_side_effects(&state, job, false).await?;
    }
    if publish_job_update {
        publish(&state, "job.updated", &job);
    }
    // sc-4203 (F-API-5): workers POST progress per inference step. The queue summary
    // is a full SQLite aggregation plus a stale-worker sweep, serialized and
    // broadcast to every SSE subscriber — but the queue composition only changes when
    // a job's status transitions (queued/running/terminal), not on a percentage tick.
    // Skip the refresh on pure ticks; the stale sweep still runs on worker heartbeats
    // and on every status transition.
    if status_changed {
        publish_queue(&state).await?;
    }
    Ok(Json(job))
}

pub(crate) fn terminal_model_job_changes_catalog(job_type: &JobType, status: &JobStatus) -> bool {
    matches!(
        job_type,
        JobType::ModelDownload | JobType::ModelImport | JobType::ModelConvert
    ) && matches!(
        status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Canceled | JobStatus::Interrupted
    )
}

pub(crate) fn invalidate_model_catalog_for_terminal_jobs(state: &AppState, jobs: &[JobSnapshot]) {
    if jobs
        .iter()
        .any(|job| terminal_model_job_changes_catalog(&job.job_type, &job.status))
    {
        state.model_catalog_cache.invalidate();
    }
}

pub(crate) const PROGRESS_SIDE_EFFECT_RECOVERY_BATCH: usize = 128;
const PROGRESS_SIDE_EFFECT_RECOVERY_INTERVAL: Duration = Duration::from_secs(1);

/// Apply the API-owned writes for one accepted progress snapshot, then fold the
/// derived result back through the store's ownership/status/result CAS.
async fn apply_progress_side_effects(
    state: &AppState,
    job: JobSnapshot,
    clear_terminal_side_effects: bool,
) -> Result<JobSnapshot, ApiError> {
    let job_id = job.id.clone();
    // Start from the store's accepted result because it may have merged
    // accumulated training-sample history into the submitted object.
    let accepted_result = job.result.clone();
    let mut result = accepted_result.clone();
    // On a completing real training run, register the produced adapter as a
    // SceneWorks LoRA after authoritative completion acceptance, then fold the
    // registration outcome into the persisted result (story 1418).
    if matches!(job.status, JobStatus::Completed) {
        if let Some(status) = register_completed_training_lora(state, &job_id).await {
            result.extend(status);
        }
        // A completing ControlTraining run registers its trained overlay into the control-overlay
        // manifest so it is selectable + runnable in generation (sc-10165, B4). Gates on
        // `JobType::ControlTraining`, so exactly one of these two fires per job.
        if let Some(status) = register_completed_control_overlay(state, &job_id).await {
            result.extend(status);
        }
        // A completing FULL base fine-tune registers its trained checkpoint into the MODEL catalog
        // so it is selectable + runnable in generation (sc-15036, epic 14034 F6). Like the pair
        // above it self-gates — on the plan's `outputKind`, which is `base_checkpoint` only for a
        // `networkType: "full"` run — so exactly one of the three registrars fires per job and the
        // LoRA registrar defers rather than mis-registering an 8 GB base checkpoint as an adapter.
        if let Some(status) = register_completed_base_checkpoint(state, &job_id).await {
            result.extend(status);
        }
    }
    // Fail after catalog registration but before project persistence/CAS so
    // the recovery regression covers a genuinely partial side-effect run. The
    // resumed registration must upsert rather than duplicate.
    #[cfg(test)]
    if clear_terminal_side_effects
        && std::mem::take(&mut *state.progress_side_effects_fail_once.lock())
    {
        return Err(ApiError::internal(
            "injected post-acceptance progress side-effect failure",
        ));
    }
    #[cfg(test)]
    if clear_terminal_side_effects
        && state
            .progress_side_effects_fail_job_ids
            .lock()
            .contains(&job_id)
    {
        *state
            .progress_side_effects_attempts
            .lock()
            .entry(job_id.clone())
            .or_default() += 1;
        return Err(ApiError::internal(
            "injected persistent progress side-effect failure",
        ));
    }
    // Persist any generated assets the worker reported as `assetWrites` facts and
    // re-inject the built sidecars into the result so the UI keeps streaming them
    // (story 1656 — Rust is the single project-store writer).
    persist_reported_assets(state, &job_id, &mut result).await?;

    if result == accepted_result && !clear_terminal_side_effects {
        return Ok(job);
    }
    let expected_worker_id = job.worker_id.clone();
    let expected_status = job.status.clone();
    store_call(state.clone(), move |store, _timeout| {
        store.replace_job_result_after_progress(
            &job_id,
            expected_worker_id.as_deref(),
            expected_status,
            &accepted_result,
            &result,
            clear_terminal_side_effects,
        )
    })
    .await
}

/// Claim one durable terminal handoff under the process-wide serializer,
/// re-checking status and owner after taking the lock. The same function serves
/// the accepting request, live background recovery, and startup recovery.
async fn process_pending_terminal_progress_side_effects(
    state: &AppState,
    job_id: &str,
) -> Result<Option<JobSnapshot>, ApiError> {
    let _guard = state.progress_side_effects_lock.lock().await;
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| {
            let current = store.get_job(&job_id)?;
            store.pending_terminal_progress_side_effects(
                &job_id,
                current.worker_id.as_deref(),
                current.status,
            )
        }
    })
    .await?;
    let Some(job) = job else {
        return Ok(None);
    };
    apply_progress_side_effects(state, job, true)
        .await
        .map(Some)
}

/// Drain the batch that is due *now* — the production entry point, called on
/// startup and on the background cadence. Resolving the instant here rather
/// than inside the scan is what lets the drain below be driven from a fixed
/// instant; see it for the behavior and for why.
pub(crate) async fn recover_pending_terminal_progress_side_effects_once(
    state: &AppState,
) -> Result<usize, ApiError> {
    recover_pending_terminal_progress_side_effects_as_of(state, now_unix_seconds()).await
}

/// Drain one bounded batch of durable terminal handoffs, taking those due as of
/// `as_of` (Unix seconds) rather than at the wall clock. Per-job failures are
/// isolated and remain pending for the next cadence; a DB enumeration failure
/// is returned so the lifecycle loop can report it without exiting.
///
/// Production passes `now` via the wrapper above; tests freeze the instant so an
/// assertion about the durable backoff cannot be decided by how long the test
/// took (sc-17640). Only this read side honors `as_of` — a failed attempt is
/// still deferred against real time by the store.
pub(crate) async fn recover_pending_terminal_progress_side_effects_as_of(
    state: &AppState,
    as_of: i64,
) -> Result<usize, ApiError> {
    let ids = store_call(state.clone(), move |store, _timeout| {
        store.pending_terminal_progress_side_effect_job_ids_as_of(
            as_of,
            PROGRESS_SIDE_EFFECT_RECOVERY_BATCH,
        )
    })
    .await?;
    let mut recovered = 0;
    for job_id in ids {
        match process_pending_terminal_progress_side_effects(state, &job_id).await {
            Ok(Some(job)) => {
                publish(state, "job.updated", &job);
                recovered += 1;
            }
            Ok(None) => {}
            Err(error) => {
                let deferred = store_call(state.clone(), {
                    let job_id = job_id.clone();
                    move |store, _timeout| {
                        store.defer_pending_terminal_progress_side_effects(&job_id)
                    }
                })
                .await;
                match deferred {
                    Ok(retry_deferred) => tracing::warn!(
                        event = "terminal_progress_side_effect_recovery_failed",
                        job_id,
                        status = error.status.as_u16(),
                        detail = %error.detail,
                        retry_deferred,
                        "terminal progress side-effect recovery remains pending"
                    ),
                    Err(defer_error) => tracing::error!(
                        event = "terminal_progress_side_effect_retry_schedule_failed",
                        job_id,
                        status = error.status.as_u16(),
                        detail = %error.detail,
                        defer_status = defer_error.status.as_u16(),
                        defer_detail = %defer_error.detail,
                        "terminal progress side-effect failed and its retry could not be scheduled"
                    ),
                }
            }
        }
    }
    Ok(recovered)
}

/// Production lifecycle task: run immediately on API startup, then continue at
/// a short cadence so a post-acceptance 500 recovers even while the process and
/// worker remain alive. The durable DB bit survives a crash/restart.
pub(crate) fn spawn_terminal_progress_side_effect_recovery(
    state: AppState,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if let Err(error) = recover_pending_terminal_progress_side_effects_once(&state).await {
                tracing::warn!(
                    event = "terminal_progress_side_effect_recovery_scan_failed",
                    status = error.status.as_u16(),
                    detail = %error.detail,
                    "could not scan pending terminal progress side effects"
                );
            }
            tokio::time::sleep(PROGRESS_SIDE_EFFECT_RECOVERY_INTERVAL).await;
        }
    })
}

/// Persist the generated assets a worker reports as `assetWrites` facts in its
/// progress result, then re-inject the built sidecars into `result.assets` /
/// `result.assetIds` so ImageStudio's live preview and the library refresh keep
/// streaming (story 1656). Idempotent: re-applied progress updates upsert the
/// same rows/files. No-op when there are no `assetWrites` (status-only updates,
/// or job types that still write their own assets).
pub(crate) async fn persist_reported_assets(
    state: &AppState,
    job_id: &str,
    result: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(asset_writes) = result.get("assetWrites").and_then(Value::as_array) else {
        return Ok(());
    };
    if asset_writes.is_empty() {
        return Ok(());
    }
    let asset_writes = asset_writes.clone();
    let generation_set_id = result
        .get("generationSetId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let generation_set = result.get("generationSet").cloned();
    // The job row is authoritative for the project id (never the worker payload).
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await?;
    let Some(project_id) = job.project_id.clone() else {
        return Ok(());
    };
    let job_id_owned = job_id.to_owned();
    let built = project_call(state.clone(), move |store| {
        if let Some(generation_set) = generation_set.as_ref() {
            store.write_generation_set(
                &project_id,
                &job_id_owned,
                generation_set,
                asset_writes.first(),
            )?;
        }
        let mut built = Vec::with_capacity(asset_writes.len());
        for fact in &asset_writes {
            built.push(store.persist_generated_asset(
                &project_id,
                &job_id_owned,
                &generation_set_id,
                fact,
            )?);
        }
        Ok(built)
    })
    .await?;
    let asset_ids: Vec<Value> = built
        .iter()
        .filter_map(|asset| asset.get("id").cloned())
        .collect();
    result.insert("assets".to_owned(), Value::Array(built));
    result.insert("assetIds".to_owned(), Value::Array(asset_ids));
    result.remove("assetWrites");
    result.remove("generationSet");
    Ok(())
}

/// Attempts LoRA registration for a job reporting completion, returning result
/// fields that describe the outcome — or `None` when the job is not a real
/// training run with a staged output. Never errors the progress update: a
/// registration failure is logged and surfaced via `loraRegistered: false` +
/// `loraRegistrationError` so the trained output is not silently lost.
pub(crate) async fn register_completed_training_lora(
    state: &AppState,
    job_id: &str,
) -> Option<JsonObject> {
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .ok()?;
    if !matches!(job.job_type, JobType::LoraTrain) {
        return None;
    }
    match register_trained_lora(state, &job).await {
        Ok(None) => None,
        Ok(Some((lora_id, manifest_path))) => {
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(true));
            status.insert("loraId".to_owned(), Value::String(lora_id));
            status.insert(
                "loraManifestPath".to_owned(),
                Value::String(manifest_path.display().to_string()),
            );
            Some(status)
        }
        Err(error) => {
            tracing::error!(
                event = "lora_registration_failed",
                jobId = %job.id,
                detail = %error.detail,
                "failed to register trained LoRA"
            );
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(false));
            status.insert(
                "loraRegistrationError".to_owned(),
                Value::String(error.detail),
            );
            Some(status)
        }
    }
}

/// Registers a completed real training run's output as a normal SceneWorks LoRA,
/// returning the registered `(lora_id, manifest_path)` or `None` when there is
/// nothing to register (a dry run, or a job without a staged entry).
///
/// Security: the manifest path and output directory are recomputed from the
/// run's scope, owning project, and a validated LoRA id — never from the
/// (mutable) job payload — so a crafted or duplicated `lora_train` job cannot
/// redirect the manifest write outside the two canonical LoRA manifests
/// (`config_dir/manifests/user.loras.jsonc` or `<project>/loras/manifest.jsonc`).
/// A run whose adapter is missing under the recomputed output dir registers
/// nothing, so a failed/canceled/unwritten job never leaves a broken entry. The
/// entry shows up in `/api/v1/loras` and is selectable in the Studio (Image or
/// Video Studio, by LoRA family).
pub(crate) async fn register_trained_lora(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<Option<(String, PathBuf)>, ApiError> {
    if job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Ok(None);
    }
    // sc-15036 — a FULL base fine-tune does not produce an adapter at all. It writes a diffusers
    // transformer component directory (a full fine-tuned base checkpoint, ~8 GB) into a DIFFERENT
    // tree, and it belongs in the model catalog, not the LoRA library. `register_completed_base_\
    // checkpoint` owns it; defer here so exactly one registrar claims each job. Keyed on the plan's
    // `outputKind` — the one discriminator `sceneworks_core::training::training_output_kind` stamps
    // — not on a second reading of `networkType`.
    if job_plan_is_base_checkpoint(job) {
        return Ok(None);
    }
    let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(None);
    };
    // Derive the security-sensitive fields from the entry but trust nothing: the
    // scope is validated by `resolve_training_output_location`, and the id must be
    // a safe single path component before it can name an output dir / manifest.
    let scope = manifest_entry
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("project")
        .to_owned();
    let lora_id = manifest_entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("Training manifest entry requires an id"))?
        .to_owned();
    validate_lora_id_component(&lora_id)?;

    // Recompute the output dir and manifest path from trusted inputs; the job
    // payload's own manifest/output paths are deliberately ignored.
    let (output_dir, manifest_path) =
        resolve_training_output_location(state, &scope, job.project_id.as_deref(), &lora_id)
            .await?;
    // Register the adapter file(s) the plan declared, validated as plain in-tree
    // files that exist under the recomputed output dir. Using the declared final
    // name (not the first `.safetensors` on disk) means a step checkpoint sharing
    // the directory is never registered in place of the final adapter, while the
    // validation still rejects any `..`-traversing name a crafted payload injects.
    let Some(files) = trusted_adapter_files(manifest_entry.get("files"), &output_dir) else {
        return Err(ApiError::internal(format!(
            "No declared trained adapter found under {}; skipping LoRA registration",
            output_dir.display()
        )));
    };

    // Overwrite the security-sensitive fields with the trusted values, keeping
    // the descriptive metadata (name, family, triggerWords, baseModel,
    // provenance) the submit step captured. `source.path` stays relative so
    // `normalize_lora_entry` resolves it under the scope root.
    let mut entry = manifest_entry;
    entry.insert("id".to_owned(), Value::String(lora_id.clone()));
    entry.insert("scope".to_owned(), Value::String(scope));
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("loras/{lora_id}") }),
    );
    entry.insert(
        "files".to_owned(),
        Value::Array(files.into_iter().map(Value::String).collect()),
    );
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));

    let upsert_id = lora_id.clone();
    mutate_manifest_entries(state, &manifest_path, "loras", move |entries| {
        // Replace any prior entry with this id (re-run) so provenance refreshes
        // without duplicating, preserving the original createdAt.
        let created_at = entries
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(upsert_id.as_str()))
            .and_then(|item| item.get("createdAt").cloned());
        let mut entries = entries
            .into_iter()
            .filter(|item| item.get("id").and_then(Value::as_str) != Some(upsert_id.as_str()))
            .collect::<Vec<_>>();
        let mut entry = entry;
        if let Some(created_at) = created_at {
            entry.insert("createdAt".to_owned(), created_at);
        }
        entries.push(Value::Object(entry));
        Ok((entries, ()))
    })
    .await?;
    Ok(Some((lora_id, manifest_path)))
}

/// Whether a training job's resolved plan declares it produces a FULL base checkpoint (sc-15036).
///
/// Reads the plan's own `target.outputKind` — the single discriminator
/// `sceneworks_core::training::training_output_kind` stamps at submit — rather than re-deriving the
/// answer from `advanced.networkType` a second time here. The expected token comes from the enum
/// itself, so renaming the variant's wire value cannot silently orphan this gate. Fails closed: a
/// job with no plan, or any other kind, is not a base checkpoint.
fn job_plan_is_base_checkpoint(job: &JobSnapshot) -> bool {
    job.payload
        .get("plan")
        .and_then(|plan| plan.get("target"))
        .and_then(|target| target.get("outputKind"))
        .and_then(Value::as_str)
        == Some(sceneworks_core::training::TrainingOutputKind::BaseCheckpoint.as_str())
}

/// The model-catalog analog of [`register_completed_training_lora`] (sc-15036, epic 14034 F6).
///
/// A completing FULL base fine-tune leaves a diffusers transformer component directory that neither
/// existing registrar has a shape for: the LoRA registry offers adapters to stack onto a base, and
/// this artifact IS a base. Without this the run trains, converges, writes a valid checkpoint — and
/// the checkpoint cannot be selected at generation, which is not a delivered capability.
///
/// Registers it into the user MODEL catalog, which is also what makes it appear without a restart:
/// `mutate_manifest_entries` invalidates `state.model_catalog_cache` on any `"models"` write, and
/// the web already refetches `/api/v1/models` on a completed `lora_train` job. (Note the
/// `terminal_model_job_changes_catalog` hook is NOT the right seam for this — it fires in
/// `update_job_progress` BEFORE `apply_progress_side_effects` runs, so it would invalidate a
/// generation older than the write it is meant to publish.)
///
/// Never errors the progress update: a failure is logged and surfaced via
/// `baseCheckpointRegistered: false` + `baseCheckpointRegistrationError` so the trained checkpoint is
/// not silently lost.
pub(crate) async fn register_completed_base_checkpoint(
    state: &AppState,
    job_id: &str,
) -> Option<JsonObject> {
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .ok()?;
    if !matches!(job.job_type, JobType::LoraTrain) || !job_plan_is_base_checkpoint(&job) {
        return None;
    }
    match register_trained_base_checkpoint(state, &job).await {
        Ok(None) => None,
        Ok(Some((model_id, manifest_path))) => {
            let mut status = JsonObject::new();
            status.insert("baseCheckpointRegistered".to_owned(), Value::Bool(true));
            status.insert("baseCheckpointModelId".to_owned(), Value::String(model_id));
            status.insert(
                "baseCheckpointManifestPath".to_owned(),
                Value::String(manifest_path.display().to_string()),
            );
            Some(status)
        }
        Err(error) => {
            tracing::error!(
                event = "base_checkpoint_registration_failed",
                job_id,
                status = error.status.as_u16(),
                detail = %error.detail,
                "trained base checkpoint could not be registered into the model catalog"
            );
            let mut status = JsonObject::new();
            status.insert("baseCheckpointRegistered".to_owned(), Value::Bool(false));
            status.insert(
                "baseCheckpointRegistrationError".to_owned(),
                Value::String(error.detail.clone()),
            );
            Some(status)
        }
    }
}

/// Upsert the completed full fine-tune's checkpoint into the user model catalog (sc-15036).
///
/// Same trusted-input discipline as [`register_trained_lora`]: the id is validated to a safe single
/// path component, and the output dir + manifest path are RECOMPUTED from it rather than read back
/// from the mutable job payload. Two things differ, both because the artifact is a model:
///
/// * the payload's declared `files` are ignored entirely — [`trusted_base_checkpoint_files`]
///   validates the recomputed directory's SHAPE (both component files present) instead, which is
///   what `trusted_adapter_files`'s `is_file()` contract cannot express; and
/// * `paths.model` is stamped with the recomputed absolute directory, so the catalog's install-state
///   derivation and the worker's render lane resolve the same path the API just validated.
///
/// `apply_model_manifest_defaults` then fills the family surface (`adapter`, `capabilities`,
/// resolutions/defaults, `loraCompatibility`) exactly as the model-import job does, so a fine-tune
/// arrives in the Studio with its base's surface rather than a bare 4-option resolution list.
pub(crate) async fn register_trained_base_checkpoint(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<Option<(String, PathBuf)>, ApiError> {
    if job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Ok(None);
    }
    let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(None);
    };
    let model_id = manifest_entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("Training manifest entry requires an id"))?
        .to_owned();
    validate_lora_id_component(&model_id)?;
    let (output_dir, manifest_path) = resolve_finetune_output_location(state, &model_id);

    let Some(files) = trusted_base_checkpoint_files(&output_dir) else {
        return Err(ApiError::internal(format!(
            "No complete fine-tuned base checkpoint found under {} (expected {} and {}); skipping \
             model registration",
            output_dir.display(),
            sceneworks_core::base_weights::MAGE_FLOW_TRANSFORMER_CONFIG_FILE,
            sceneworks_core::base_weights::MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE,
        )));
    };

    // The catalog derives `installState` for a non-downloadable entry from the SceneWorks install
    // marker in `paths.model` (`model_is_installed`), exactly as it does for an imported model. A
    // fine-tune has no download to leave one, so write it here — without it the entry lands with
    // `installState: "missing"` and `modelInstallComplete` drops it from every picker, i.e. the
    // model would be registered and still not selectable.
    tokio::fs::write(
        output_dir.join(".sceneworks-download-complete.json"),
        serde_json::to_vec_pretty(&json!({
            "modelId": model_id,
            "provider": "training",
            "jobId": job.id,
            "completedAt": now_rfc3339(),
        }))
        .map_err(|error| ApiError::internal(error.to_string()))?,
    )
    .await
    .map_err(|error| {
        ApiError::internal(format!(
            "Could not mark the fine-tuned checkpoint at {} as installed: {error}",
            output_dir.display()
        ))
    })?;

    let model_type = manifest_entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("image")
        .to_owned();
    let family = manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::to_owned);

    let mut entry = manifest_entry;
    entry.insert("id".to_owned(), Value::String(model_id.clone()));
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("models/finetunes/{model_id}") }),
    );
    entry.insert(
        "files".to_owned(),
        Value::Array(files.into_iter().map(Value::String).collect()),
    );
    entry.insert(
        "paths".to_owned(),
        json!({ "model": output_dir.display().to_string() }),
    );
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
    sceneworks_core::lora_family::apply_model_manifest_defaults(
        &mut entry,
        &model_type,
        family.as_deref(),
    );
    // sc-15328 — a fine-tuned checkpoint must not ADVERTISE adapters it cannot render.
    //
    // `apply_model_manifest_defaults` synthesizes `loraCompatibility.families = [family]` from the
    // family token alone, which is right for an imported sibling of a builtin but wrong here: the
    // fine-tuned lane refuses adapters on every backend (`mlx_gen_mage::load_finetuned`, and the
    // `!has_loras` term in `imported_image_request_family_eligible`). Left advertised, the API
    // accepted the job and NO worker could claim it — it queued forever with no error.
    //
    // 🔴 This removal does NOT by itself produce the rejection sc-15328 claimed for it, and for two
    // years it did not: `families_from_value_chain` (lib.rs) falls back to the top-level `family`
    // key — which this entry still carries, and must, for routing — so `model_lora_families` kept
    // returning `["mage-flow"]`, the "has no declared LoRA families" branch was never taken, and the
    // lane went on hanging. Removing the key here is kept only so the STORED entry carries no false
    // promise; the enforcement is `models::apply_imported_lora_advertisement`, which withdraws the
    // advertisement on the catalog projection every read goes through (an explicit EMPTY families
    // array, which is non-null and so actually defeats that fallback).
    //
    // Whether a fine-tune SHOULD accept adapters is a separate product question (sc-15334) — this
    // only guarantees that what is advertised is what can actually run.
    entry.remove("loraCompatibility");

    let upsert_id = model_id.clone();
    mutate_manifest_entries(state, &manifest_path, "models", move |entries| {
        // Replace any prior entry with this id (a re-run of the same job) so provenance refreshes
        // without duplicating, preserving the original createdAt — the LoRA registrar's contract.
        let created_at = entries
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(upsert_id.as_str()))
            .and_then(|item| item.get("createdAt").cloned());
        let mut entries = entries
            .into_iter()
            .filter(|item| item.get("id").and_then(Value::as_str) != Some(upsert_id.as_str()))
            .collect::<Vec<_>>();
        let mut entry = entry;
        if let Some(created_at) = created_at {
            entry.insert("createdAt".to_owned(), created_at);
        }
        entries.push(Value::Object(entry));
        Ok((entries, ()))
    })
    .await?;
    Ok(Some((model_id, manifest_path)))
}

/// The control-overlay analog of [`register_completed_training_lora`] (sc-10165, epic 10159 B4). A
/// completing `ControlTraining` run leaves a trained overlay in its output dir that nothing yet
/// registers (`register_completed_training_lora` is `LoraTrain`-only), so a Krea ControlNet was produced
/// but not usable in generation. This registers it into the control-overlay manifest. Never errors the
/// progress update: a failure is logged and surfaced via `controlOverlayRegistered: false` +
/// `controlOverlayRegistrationError` so the trained output is not silently lost.
pub(crate) async fn register_completed_control_overlay(
    state: &AppState,
    job_id: &str,
) -> Option<JsonObject> {
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .ok()?;
    if !matches!(job.job_type, JobType::ControlTraining) {
        return None;
    }
    match register_trained_control_overlay(state, &job).await {
        Ok(None) => None,
        Ok(Some((overlay_id, manifest_path))) => {
            let mut status = JsonObject::new();
            status.insert("controlOverlayRegistered".to_owned(), Value::Bool(true));
            status.insert("controlOverlayId".to_owned(), Value::String(overlay_id));
            status.insert(
                "controlOverlayManifestPath".to_owned(),
                Value::String(manifest_path.display().to_string()),
            );
            Some(status)
        }
        Err(error) => {
            tracing::error!(
                event = "control_overlay_registration_failed",
                jobId = %job.id,
                detail = %error.detail,
                "failed to register trained control overlay"
            );
            let mut status = JsonObject::new();
            status.insert("controlOverlayRegistered".to_owned(), Value::Bool(false));
            status.insert(
                "controlOverlayRegistrationError".to_owned(),
                Value::String(error.detail),
            );
            Some(status)
        }
    }
}

/// Registers a completed control-training run's overlay as a SceneWorks control overlay, returning the
/// registered `(overlay_id, manifest_path)` or `None` when there is nothing to register (a dry run, or a
/// job without a staged entry).
///
/// Security: mirrors [`register_trained_lora`] exactly — the manifest path and output dir are recomputed
/// from the run's scope, owning project, and a validated overlay id (never the mutable job payload), so a
/// crafted or duplicated job cannot redirect the write outside the two canonical control-overlay
/// manifests (`config_dir/manifests/user.control_overlays.jsonc` or
/// `<project>/control-overlays/manifest.jsonc`). A run whose overlay is missing under the recomputed dir
/// registers nothing. The entry lands in the `controlOverlays` field and is selectable as a ControlNet
/// for its backbone in the Studio.
pub(crate) async fn register_trained_control_overlay(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<Option<(String, PathBuf)>, ApiError> {
    if job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Ok(None);
    }
    let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(None);
    };
    let scope = manifest_entry
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("project")
        .to_owned();
    let overlay_id = manifest_entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("Control training manifest entry requires an id"))?
        .to_owned();
    validate_lora_id_component(&overlay_id)?;

    // Recompute the output dir + manifest path from trusted inputs; the payload's own paths are ignored.
    let (output_dir, manifest_path) = resolve_control_overlay_output_location(
        state,
        &scope,
        job.project_id.as_deref(),
        &overlay_id,
    )
    .await?;
    let Some(files) = trusted_adapter_files(manifest_entry.get("files"), &output_dir) else {
        return Err(ApiError::internal(format!(
            "No declared trained overlay found under {}; skipping control-overlay registration",
            output_dir.display()
        )));
    };

    // Overwrite the security-sensitive fields with trusted values, keeping the descriptive control
    // metadata (name, controlType, controlEngine, backbone, baseModel, kind, provenance) the submit step
    // captured. `source.path` stays relative so the catalog resolves it under the scope root.
    let mut entry = manifest_entry;
    entry.insert("id".to_owned(), Value::String(overlay_id.clone()));
    entry.insert("scope".to_owned(), Value::String(scope));
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("control-overlays/{overlay_id}") }),
    );
    entry.insert(
        "files".to_owned(),
        Value::Array(files.into_iter().map(Value::String).collect()),
    );
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));

    let upsert_id = overlay_id.clone();
    mutate_manifest_entries(state, &manifest_path, "controlOverlays", move |entries| {
        // Replace any prior entry with this id (re-run) so provenance refreshes without duplicating,
        // preserving the original createdAt.
        let created_at = entries
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(upsert_id.as_str()))
            .and_then(|item| item.get("createdAt").cloned());
        let mut entries = entries
            .into_iter()
            .filter(|item| item.get("id").and_then(Value::as_str) != Some(upsert_id.as_str()))
            .collect::<Vec<_>>();
        let mut entry = entry;
        if let Some(created_at) = created_at {
            entry.insert("createdAt".to_owned(), created_at);
        }
        entries.push(Value::Object(entry));
        Ok((entries, ()))
    })
    .await?;
    Ok(Some((overlay_id, manifest_path)))
}
