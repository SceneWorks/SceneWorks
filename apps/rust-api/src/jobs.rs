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
    Ok(Json(public_job_snapshots(jobs)))
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
    ApiJson(mut payload): ApiJson<JobCreateRequest>,
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
    canonicalize_image_model_payload(&state, &payload.job_type, &mut payload.payload).await?;
    crate::model_sources::ensure_runtime_model_sources(
        &state,
        &payload.job_type,
        &mut payload.payload,
    )
    .await?;
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

pub(crate) async fn claim_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let mlx_required = state.settings.mlx_required;
    let enforce_unsupported = state.settings.mlx_enforce_unsupported;
    let candle_required = state.settings.candle_required;
    let candle_enforce = state.settings.candle_enforce_unsupported;
    let host_os = state.settings.host_os.clone();
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
            // Platform reachability (sc-19570): fail any queued video job whose mode no lane on
            // THIS host can ever claim. Unlike the four sweeps above it takes no flag and no grace
            // window — the gap is structural, not transient, and every one of those four declines
            // to touch this job (the stranded sweeps bail the moment a live worker of their own
            // kind exists; both unsupported sweeps default to warn), which is why it hung.
            // `POST /api/v1/video/jobs` runs the same sweep inline so the hang closes even where no
            // worker ever polls; this arm covers the raw `POST /api/v1/jobs`, retry and duplicate
            // paths that never pass through that route.
            let platform_unreachable = store.fail_platform_unreachable_jobs(&host_os)?;
            let (job, decision) = store.claim_next_job_routed(&payload.worker_id, mlx_required)?;
            Ok::<_, JobsStoreError>((
                job,
                decision,
                stranded,
                unsupported,
                candle_stranded,
                candle_unsupported,
                platform_unreachable,
            ))
        })();
        Ok((stale_sweep, claim_result))
    })
    .await?;
    handle_stale_sweep(&state, &stale_sweep);
    let (
        response,
        decision,
        stranded,
        unsupported,
        candle_stranded,
        candle_unsupported,
        platform_unreachable,
    ) = claim_result?;
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
    for job in &platform_unreachable {
        emit_platform_unreachable(job);
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
        || !platform_unreachable.is_empty()
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

/// Emit the `platform_unreachable` terminal-routing event (sc-19570) — the System → Logs surface
/// for a video job whose mode has no lane on this host at all.
///
/// A SEPARATE event from `mlx_unavailable` / `candle_unavailable` on purpose, not a fifth `mode` on
/// one of them: those two say a worker of the right kind failed to check in, which is transient and
/// operational ("confirm the worker is running"). This one says no such worker can exist here, and
/// the only remedy is a different model or a different mode. Collapsing them would send an operator
/// looking for a process that is not missing.
///
/// `pub(crate)` because the video enqueue route emits it too — `POST /api/v1/video/jobs` runs the
/// same sweep inline so the terminal state does not wait on a worker poll.
pub(crate) fn emit_platform_unreachable(job: &JobSnapshot) {
    let model = job.payload.get("model").and_then(Value::as_str);
    let mode = job.payload.get("mode").and_then(Value::as_str);
    sceneworks_core::observability::emit_event(
        tracing::Level::INFO,
        json!({
            "event": "platform_unreachable",
            "jobId": job.id,
            "jobType": job.job_type.as_str(),
            "model": model,
            "mode": mode,
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
    Ok(Json(public_job_snapshot(
        store_call(state, move |store, _timeout| store.get_job(&job_id)).await?,
    )))
}

pub(crate) async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.cancel_job(&job_id)
    })
    .await?;
    crate::generation::cascade_cancel_vector_prompt_workflow(&state, &job).await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(public_job_snapshot(job)))
}

pub(crate) async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let mut payload = retry_job_request_from_body(request).await?;
    if let Some(job) = crate::generation::replay_vector_prompt_workflow(
        state.clone(),
        &job_id,
        false,
        None,
        !payload.payload_changes.is_empty(),
    )
    .await?
    {
        return Ok((StatusCode::CREATED, Json(public_job_snapshot(job))));
    }
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
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
    if let Some(job) = crate::generation::replay_vector_prompt_workflow(
        state.clone(),
        &job_id,
        true,
        payload.requested_gpu.clone(),
        !payload.payload_changes.is_empty(),
    )
    .await?
    {
        return Ok((StatusCode::CREATED, Json(public_job_snapshot(job))));
    }
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
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// The character-route inline LoRA links a job's PERSISTED payload already carried — the ONLY
/// adapters a retry/duplicate of that job may re-validate as inline. Empty for every other job.
///
/// `characters.rs`'s test-job route is the one image-generation boundary that validates with inline
/// LoRAs allowed. A character's attached adapters are inline links, not catalog rows:
/// `character_store::attach_lora` mints `id: "character_lora_<hex>"` with `category: "character"` and
/// a `sourcePath`/`projectPath`, copies the file into the project, and registers it in NO LoRA
/// catalog. Re-validating that set as catalog-backed refuses it with "LoRA not found" — which broke
/// even a no-op retry of a character test job when this boundary's gate was first mirrored.
///
/// ## Why the link SHAPE and not the `characterId` / `mode` markers
///
/// Both of those look server-stamped but are caller-settable: `ImageJobRequest` exposes
/// `character_id` and `mode`, so `POST /api/v1/image/jobs { mode: "character_image", loras: [] }` is
/// an ordinary image job that create admits (the LoRA gate no-ops on an empty set) while bearing
/// both markers. Deriving permission from them would let that job's retry attach an arbitrary
/// path-bearing adapter and have it accepted as "inline".
///
/// The link shape cannot be forged the same way, because it can only have been PERSISTED by a
/// boundary that already allowed inline LoRAs:
/// - `create_image_job` / `create_video_job` validate with `allow_inline_loras = false`, so a
///   non-catalog adapter is refused before any job row exists.
/// - `POST /api/v1/jobs` refuses image/video generation job types outright (`typed_generation_route`).
/// - retry/duplicate reach this function, which is where the permission is being decided.
///
/// So the only way a persisted `image_generate` payload holds a character link is the character
/// route. Requiring a non-empty `characterId` as well costs nothing (that route stamps it onto the
/// same payload it writes the links into) and keeps the predicate honest about what it identifies.
///
/// An empty persisted set deliberately yields an empty permit: the gate no-ops on it anyway, and a
/// character job whose character had no adapters must not become a hole through which a retry can add
/// inline ones — attaching them to the character is the supported path.
fn persisted_character_inline_loras(persisted_payload: &JsonObject) -> Vec<Value> {
    let has_character = persisted_payload
        .get("characterId")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty());
    if !has_character {
        return Vec::new();
    }
    persisted_payload
        .get("loras")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|lora| {
            lora.get("category").and_then(Value::as_str) == Some("character")
                || lora
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id.starts_with("character_lora_"))
        })
        .cloned()
        .collect()
}

/// Whether `candidate` IS one of the persisted character links in `permitted`, and therefore may be
/// hydrated inline.
///
/// Identity is the link id AND agreement on every path field the candidate carries. Matching on the
/// id alone would leave the narrowing hollow: `normalize_inline_job_lora` passes a caller's object
/// through almost verbatim, so replaying a persisted link id with a swapped `sourcePath` would carry
/// an arbitrary file into the enqueued payload. A candidate that omits a path field is still a match
/// — it is asking for the persisted link, not redirecting it.
fn matches_permitted_inline_lora(candidate: &Value, permitted: &[Value]) -> bool {
    let Some(candidate_id) = job_lora_id(candidate) else {
        return false;
    };
    permitted.iter().any(|link| {
        if job_lora_id(link) != Some(candidate_id) {
            return false;
        }
        ["sourcePath", "projectPath"].iter().all(|field| {
            match (
                candidate.get(*field).and_then(Value::as_str),
                link.get(*field).and_then(Value::as_str),
            ) {
                (Some(requested), Some(persisted)) => requested == persisted,
                // The candidate names no path for this field, so it cannot redirect it.
                (None, _) => true,
                // The candidate names a path the persisted link does not have at all.
                (Some(_), None) => false,
            }
        })
    })
}

/// Re-validate a merged retry/duplicate payload's `loras`, granting inline hydration ONLY to the
/// adapters the persisted payload already carried (`permitted_inline`) and requiring catalog backing
/// for everything else — including any ADDITION to a genuine character job's set.
///
/// A single `allow_inline_loras = true` would have covered the whole merged array, so a retry of a
/// real character job could swap in arbitrary inline path-bearing adapters. Splitting the array is
/// verdict-preserving: `validate_lora_specs_for_model` decides each attached adapter independently
/// (no cross-adapter state), so validating two sub-arrays yields exactly the per-adapter verdicts one
/// pass over the whole array would, and the original order is restored afterwards.
async fn validate_merged_job_loras(
    state: &AppState,
    project_id: Option<&str>,
    merged: &mut JsonObject,
    permitted_inline: &[Value],
) -> Result<(), ApiError> {
    // No inline permit (every job but a character test job's replay): one catalog-only pass, which
    // is the create path's own posture.
    if permitted_inline.is_empty() {
        return validate_job_lora_compatibility(state, project_id, merged, false).await;
    }
    let Some(loras) = merged
        .get("loras")
        .and_then(Value::as_array)
        .filter(|loras| !loras.is_empty())
        .cloned()
    else {
        return Ok(());
    };

    // Partition, remembering each adapter's slot so the normalized array keeps the caller's order.
    let mut inline_slots = Vec::new();
    let mut catalog_slots = Vec::new();
    for (slot, lora) in loras.iter().enumerate() {
        if matches_permitted_inline_lora(lora, permitted_inline) {
            inline_slots.push((slot, lora.clone()));
        } else {
            catalog_slots.push((slot, lora.clone()));
        }
    }

    let mut normalized = vec![Value::Null; loras.len()];
    for (allow_inline, slots) in [(true, inline_slots), (false, catalog_slots)] {
        if slots.is_empty() {
            continue;
        }
        let mut probe = merged.clone();
        probe.insert(
            "loras".to_owned(),
            Value::Array(slots.iter().map(|(_, lora)| lora.clone()).collect()),
        );
        validate_job_lora_compatibility(state, project_id, &mut probe, allow_inline).await?;
        let validated = probe
            .get("loras")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::internal("LoRA validation dropped the adapter array"))?;
        // `validate_lora_specs_for_model` may legitimately SKIP an entry (an unusable spec that
        // `hydrate_lora_spec` returns `None` for), so pair by position over what came back rather
        // than assuming a 1:1 mapping.
        for ((slot, _), value) in slots.iter().zip(validated) {
            normalized[*slot] = value.clone();
        }
    }
    merged.insert(
        "loras".to_owned(),
        Value::Array(normalized.into_iter().filter(|v| !v.is_null()).collect()),
    );
    Ok(())
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
    // Resolved from the PERSISTED payload, BEFORE the merge below, because inline-LoRA permission is
    // a property of how the job was originally created and `payload_changes` must not be able to
    // mint it. Carries the specific adapters permitted, not a blanket flag — see
    // [`persisted_character_inline_loras`] and [`validate_merged_job_loras`].
    let permitted_inline_loras = persisted_character_inline_loras(&merged);
    merged.extend(payload_changes.clone());
    if generation_job_model_is_path_backed(&job_type) {
        validate_payload_model(&merged)?;
    } else {
        validate_raw_job_payload(state, &job_type, &merged).await?;
    }
    let canonical_model_entry =
        canonicalize_image_model_payload(state, &job_type, &mut merged).await?;
    if matches!(
        job_type,
        JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
    ) {
        let model_id = merged
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("model must be a string"))?;
        // A retry/duplicate can replace `model`, so re-resolve the authoritative entry instead of
        // trusting the original job's modelManifestEntry. This is the replay counterpart to the
        // typed `/video/jobs` pre-enqueue platform gate.
        let model_manifest_entry = resolve_model_manifest_entry(state, model_id).await?;
        // Retry/duplicate are video creation boundaries too. Validate the exact shallow-merged
        // reference array against the CURRENT server-owned entry before stamping it: malformed
        // arrays must not be cleaned by VideoRequest's tolerant parser, and a legacy multi-ref row
        // must not bypass today's descriptor gate simply because its stored entry is rebuilt here.
        validate_video_reference_asset_ids_payload(&merged, &model_manifest_entry)?;
        crate::generation::ensure_video_model_available_on_platform(
            model_id,
            &model_manifest_entry,
            crate::generation::video_job_platform(state),
        )?;
        merged.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    }
    if matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        if let Some(advanced) = merged.get("advanced").and_then(Value::as_object) {
            validate_image_pose_count(advanced)?;
        }
        // PRECEDENCE DIVERGENCE, deliberate: `create_image_job` resolves the control overlay AFTER
        // its LoRA gate, this boundary resolves it BEFORE. Acceptance is equivalent — the two read
        // disjoint fields (`advanced.controlWeights.overlayId` vs the top-level `loras` array) and
        // neither can change the other's verdict — so no payload is admitted here that create would
        // refuse, or vice versa. Only the FIRST error reported for a payload that is invalid on both
        // axes differs. Left as-is rather than reordered: this call predates the gate mirror and the
        // existing ordering is pinned by the sc-13639 control-weights reauthorization tests.
        crate::control_overlays::resolve_control_overlay_selection(
            state,
            project_id.as_deref(),
            &mut merged,
        )
        .await?;
        validate_prompt_enhancement_payload(&merged)?;

        // Retry and duplicate are image-job creation boundaries too. The canonicalizer above
        // already discarded any persisted or caller-supplied path-bearing resolution and rebuilt
        // `modelManifestEntry` from the current catalog plus the authored opaque id; re-resolve
        // the authored text-encoder selection against that rebuilt entry so a removed/retargeted
        // choice fails before the queue transaction.
        if let Some(mut manifest_entry) = canonical_model_entry {
            let model_id = merged
                .get("model")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::bad_request("model must be a string"))?
                .to_owned();
            resolve_selected_image_text_encoder(state, &merged, &model_id, &mut manifest_entry)
                .await?;
            // The `create_image_job` capability gates this boundary did not run, applied to the
            // SAME merged object the queue transaction will persist (sc-18420). `payload_changes`
            // is a SHALLOW merge, so both `advanced` and the top-level `loras` array arrive
            // REPLACED WHOLESALE: without these, a retry/duplicate could enqueue a decoder+`usePid`
            // pair, an uninstalled or wrong-backend decoder, a family-incompatible or uninstalled
            // LoRA set, or an imported request shape — every one of which the create path 400s, and
            // all of them exactly the combinations this boundary's own doc comment claims it
            // re-validates.
            //
            // Order mirrors `create_image_job`: the decoder gate reads the rebuilt entry directly,
            // the entry is then stamped, the LoRA gate runs (it also NORMALIZES `loras` in place,
            // and the canonical object returned from here is what gets persisted), and the imported
            // gate runs last so it sees the server-owned row and the normalized adapter list rather
            // than anything the caller sent.
            crate::generation::validate_selected_decoder_for_manifest(
                crate::generation::enqueue_backend(state),
                &merged,
                &manifest_entry,
            )?;
            merged.insert("modelManifestEntry".to_owned(), manifest_entry);
            validate_merged_job_loras(
                state,
                project_id.as_deref(),
                &mut merged,
                &permitted_inline_loras,
            )
            .await?;
            crate::generation::validate_imported_submission(state, &model_id, &merged)?;
        }
    } else if matches!(
        job_type,
        JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
    ) {
        // The video half of the same bypass — `create_video_job` runs both of these too.
        // `canonicalize_image_model_payload` is image-only, so there is no rebuilt entry to gate
        // against here; resolve the catalog row for the merged model exactly as `create_video_job`
        // does and gate on that. Read-only on purpose — this closes the bypass without taking on
        // video's separate entry-canonicalization question.
        //
        // The decoder gate is keyed off an actually-present `advanced.decoder`, which is precisely
        // when it stops being a no-op, so the overwhelmingly common decoder-less replay costs no
        // extra catalog resolution. The LoRA gate needs no such guard: it returns before touching a
        // catalog when `loras` is absent or empty.
        let selects_decoder = merged
            .get("advanced")
            .and_then(Value::as_object)
            .is_some_and(|advanced| advanced.contains_key("decoder"));
        if let Some(model_id) = merged
            .get("model")
            .and_then(Value::as_str)
            .filter(|_| selects_decoder)
        {
            let entry = crate::models::resolve_model_manifest_entry(state, model_id).await?;
            crate::generation::validate_selected_decoder_for_manifest(
                crate::generation::enqueue_backend(state),
                &merged,
                &entry,
            )?;
        }
        validate_merged_job_loras(
            state,
            project_id.as_deref(),
            &mut merged,
            &permitted_inline_loras,
        )
        .await?;
    }
    Ok(merged)
}

/// Resolve and stamp the authoritative catalog entry at every image-job creation boundary.
///
/// The typed image route, raw Batch Detail route, retry, and duplicate all reach this seam. That is
/// security-sensitive for imported/custom models: scheduling uses the entry's family and installed
/// path hints, and the native workers then confine the resulting path before opening it. A caller may
/// choose the catalog model id, but may never replace the server-owned entry that proves what that id
/// means. Keeping this post-merge also prevents `payloadChanges` from reopening that trust boundary.
///
/// Historical raw `image_detail` jobs without an explicit model remain untouched. They predate model
/// metadata hydration and are used by the public queue/claim contract; stamping an empty entry and a
/// tier selector into that shape neither helps the worker nor preserves the contract. The shipped
/// Batch Detail request names `realvisxl`, so a real request resolves a non-empty entry below.
pub(crate) async fn canonicalize_image_model_payload(
    state: &AppState,
    job_type: &JobType,
    payload: &mut JsonObject,
) -> Result<Option<Value>, ApiError> {
    if !matches!(
        job_type,
        JobType::ImageGenerate | JobType::ImageEdit | JobType::ImageDetail
    ) {
        return Ok(None);
    }

    let Some(model_id) = payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_owned)
    else {
        if matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
            return Err(ApiError::bad_request("model is required"));
        }
        reject_image_detail_packed_tier(payload)?;
        // Never forward CATALOG metadata that cannot be tied to an explicit catalog id — a forged
        // entry is the thing this drop exists to destroy, and the established `{}` contract stays
        // exact.
        //
        // The server-resolved text encoder is the one exception, and it is not an exception to the
        // rule so much as a different kind of value: `resolvedTextEncoder` is written by the API
        // itself when it resolves the client's opaque option id, it is worker-private rather than
        // client-authored, and the worker claim is required to retain it (sc-18314). Dropping it
        // here would silently strip a resolution the client never supplied and could not forge,
        // and would leave the public projection with nothing to key its path redaction on — so the
        // private path would then survive in any sibling payload field instead of being scrubbed.
        // Keep only that sub-object; everything else in the entry still goes.
        let resolved_text_encoder = payload
            .get_mut("modelManifestEntry")
            .and_then(Value::as_object_mut)
            .and_then(|entry| entry.remove("resolvedTextEncoder"));
        payload.remove("modelManifestEntry");
        if let Some(resolution) = resolved_text_encoder {
            payload.insert(
                "modelManifestEntry".to_owned(),
                Value::Object(serde_json::Map::from_iter([(
                    "resolvedTextEncoder".to_owned(),
                    resolution,
                )])),
            );
        }
        return Ok(None);
    };
    validate_model_id(&model_id)?;

    let model_manifest_entry = project_image_manifest_for_worker(
        crate::models::resolve_model_manifest_entry(state, &model_id).await?,
    );
    if matches!(job_type, JobType::ImageDetail)
        && !model_manifest_entry
            .as_object()
            .is_some_and(|entry| !entry.is_empty())
    {
        return Err(ApiError::bad_request(format!(
            "image_detail model '{model_id}' was not found in the model catalog"
        )));
    }

    // Overwrite rather than merge. The selected id was path-confined above; imported/custom
    // workers independently confine the authoritative entry's `modelPath` / `paths.model` before
    // opening it, preserving the established two-boundary defense.
    payload.insert(
        "modelManifestEntry".to_owned(),
        model_manifest_entry.clone(),
    );
    if matches!(job_type, JobType::ImageDetail) {
        canonicalize_image_detail_dense_tier(payload)?;
    }
    Ok(Some(model_manifest_entry))
}

/// Project catalog-only, request-scoped components out of the generic image worker payload.
///
/// The shared SDXL OpenPose checkpoint is a soft install companion so Model Manager can provision
/// and repair it alongside each supported backbone. It is not a descriptor-required component and
/// must never be staged by ordinary txt2img, edit, or Batch Detail jobs. The dedicated
/// `sdxl_control` pose route owns its strict authority tuple and resolves that exact component
/// synchronously from the pinned cache before load, so forwarding this catalog row to unrelated
/// jobs only broadens their worker-visible artifact set without helping the pose route.
///
/// Keep every other soft component: selected decoders and other request-specific features resolve
/// their own authored component ids from this worker-private entry.
fn project_image_manifest_for_worker(mut entry: Value) -> Value {
    if let Some(downloads) = entry.get_mut("downloads").and_then(Value::as_array_mut) {
        downloads.retain(|download| {
            !(download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                && download.get("required").and_then(Value::as_str) == Some("soft")
                && download.get("componentId").and_then(Value::as_str)
                    == Some("controlnet_openpose"))
        });
    }
    entry
}

#[cfg(not(target_os = "macos"))]
fn canonicalize_image_detail_dense_tier(payload: &mut JsonObject) -> Result<(), ApiError> {
    // Candle's SDXL detail provider supports only the dense bf16 base. Batch Detail does not expose
    // a tier picker, so the API owns this selector. Reject all three packed carriers the image
    // product surface can emit instead of silently converting a caller's explicit request.
    reject_image_detail_packed_tier(payload)?;
    let advanced = payload
        .entry("advanced".to_owned())
        .or_insert_with(|| json!({}));
    let advanced = advanced
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("image_detail advanced must be an object"))?;

    advanced.remove("convRot");
    advanced.remove("quantTier");
    advanced.remove("mlxQuantizeExplicit");
    advanced.insert("mlxQuantize".to_owned(), Value::from(0));
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn reject_image_detail_packed_tier(payload: &JsonObject) -> Result<(), ApiError> {
    let Some(advanced) = payload.get("advanced").and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(value) = advanced.get("mlxQuantize") {
        let bits = if value.is_null() {
            0
        } else {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
                .ok_or_else(|| {
                    ApiError::bad_request("image_detail advanced.mlxQuantize must be an integer")
                })?
        };
        if bits > 0 {
            return Err(dense_image_detail_error());
        }
    }
    if let Some(value) = advanced.get("convRot") {
        match value {
            Value::Bool(false) | Value::Null => {}
            Value::Bool(true) => return Err(dense_image_detail_error()),
            _ => {
                return Err(ApiError::bad_request(
                    "image_detail advanced.convRot must be a boolean",
                ))
            }
        }
    }
    if let Some(value) = advanced.get("quantTier") {
        let dense = value.is_null()
            || value.as_str().is_some_and(|tier| {
                tier.trim().is_empty() || tier.trim().eq_ignore_ascii_case("bf16")
            });
        if !dense {
            return Err(dense_image_detail_error());
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn canonicalize_image_detail_dense_tier(_payload: &mut JsonObject) -> Result<(), ApiError> {
    // Batch Detail's MLX route retains its established platform-specific quant semantics.
    Ok(())
}

#[cfg(target_os = "macos")]
fn reject_image_detail_packed_tier(_payload: &JsonObject) -> Result<(), ApiError> {
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn dense_image_detail_error() -> ApiError {
    ApiError::bad_request(
        "Candle image_detail requires the dense bf16 model tier; packed quant selectors are unsupported",
    )
}

/// Jobs created through the raw queue route do not pass a typed request validator. Keep this
/// inventory aligned with worker payload fields that reach filesystem model resolution:
/// `image_upscale`, `image_detail`, and `prompt_refine` consume `model`; the model-management jobs
/// consume `modelId`, and `model_convert.outputDir` selects its final install location. Other raw
/// job payloads may contain descriptive model metadata, but are deliberately absent unless that
/// field selects a filesystem path.
///
/// `async` since sc-17227: the licence-acknowledgment check on the fetching job types awaits the
/// catalog. Both call sites (`create_job`, and the retry/duplicate path) must `.await` it.
async fn validate_raw_job_payload(
    state: &AppState,
    job_type: &JobType,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    if matches!(
        job_type,
        JobType::ImageUpscale | JobType::ImageDetail | JobType::PromptRefine
    ) {
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
    // different `job_type`. `/loras/import` applies the SAME predicate on its own typed route
    // (`queue_lora_import_job`) — it fetches whatever repo the caller names and never consults the
    // LoRA catalog for it, so what the catalog happens to declare has no bearing on what that route
    // can reach. `/loras/:id/download` now applies the SAME predicate on its own typed route too
    // (`create_lora_download_job`, `apps/rust-api/src/loras.rs`). The reasoning previously recorded
    // here for exempting it — that it resolves the repo FROM the catalog entry named by the path id
    // and 404s an unknown id, so "a caller cannot point it at a repo" — answered a different
    // question, and sc-17227 overturned it: who CHOOSES the repo is not who is bound by its licence.
    // A catalog LoRA whose `source.repo` names a repo a `requiresLicenseAcknowledgment` model
    // declares was fetched there with no acknowledgment, while the identical `lora_download` job
    // posted to THIS route was answered 403 — the asymmetry, not the reachability, was the defect.
    //
    // `model_convert` is here because it is a fetching job type too, and less obviously so: it
    // names no `repo`, but `resolve_convert_plan`'s LTX arm hands the payload's `baseRepo` to
    // `ensure_ltx_upscaler_cached` → `ensure_hf_files_cached`, and `upscalerFile` is a GLOB — so
    // `"**"` downloads the entire named repo. Adding the job type alone would have been inert;
    // `ensure_job_payload_license_acknowledged` reads `baseRepo`/`sourceRepo` as well as `repo`
    // (`LICENSE_GATED_REPO_PAYLOAD_KEYS`), which is what makes this line bite.
    if matches!(
        job_type,
        JobType::ModelDownload
            | JobType::ModelImport
            | JobType::ModelConvert
            | JobType::LoraDownload
            | JobType::LoraImport
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
/// analog of the per-job cancel fast path. A `queued`, `pending_caption`, or `pending_workflow` job has no
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
    let mut cascade_error = None;
    // Per-job `job.updated` so every subscriber's card flips to Cancelled (the queue
    // summary alone only updates counts, not individual cards), then one queue refresh
    // for the status counts. The pending set is bounded by what a user queued, so the
    // fan-out is small.
    for job in &jobs {
        if let Err(error) =
            crate::generation::cascade_cancel_vector_prompt_workflow(&state, job).await
        {
            cascade_error.get_or_insert(error);
        }
        publish(&state, "job.updated", job);
    }
    publish_queue(&state).await?;
    if let Some(error) = cascade_error {
        return Err(error);
    }
    Ok(Json(CancelPendingJobsResponse {
        canceled: jobs.len(),
        jobs: public_job_snapshots(jobs),
        extra: Default::default(),
    }))
}

/// Move selected not-yet-started jobs to the front of the worker queue. The store applies the
/// change under the same immediate transaction used for claims, so a worker either claims a job
/// first (and it is ignored here) or observes its new rank — there is no preemption race. Prompt
/// refinement uses the same rank mechanism automatically when it is created.
pub(crate) async fn prioritize_jobs(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<PrioritizeJobsRequest>,
) -> Result<Json<PrioritizeJobsResponse>, ApiError> {
    const MAX_PRIORITY_SELECTION: usize = 500;
    if payload.job_ids.is_empty() {
        return Err(ApiError::bad_request("Select at least one queued job"));
    }
    if payload.job_ids.len() > MAX_PRIORITY_SELECTION {
        return Err(ApiError::bad_request(format!(
            "At most {MAX_PRIORITY_SELECTION} jobs can be prioritized at once"
        )));
    }

    let job_ids = payload
        .job_ids
        .into_iter()
        .filter(|job_id| !job_id.trim().is_empty())
        .collect::<Vec<_>>();
    if job_ids.is_empty() {
        return Err(ApiError::bad_request("Select at least one queued job"));
    }

    let jobs = store_call(state.clone(), move |store, _timeout| {
        store.prioritize_jobs(&job_ids)
    })
    .await?;
    for job in &jobs {
        publish(&state, "job.updated", job);
    }
    if !jobs.is_empty() {
        publish_queue(&state).await?;
    }
    Ok(Json(PrioritizeJobsResponse {
        prioritized: jobs.len(),
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
    Ok(Json(public_job_snapshot(job)))
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
    Ok(Json(public_job_snapshot(job)))
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
    let mut asset_writes = asset_writes.clone();
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
    stamp_vector_workflow_asset_writes(&job.job_type, &job.payload, &job.id, &mut asset_writes);
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

/// Replace any worker-authored ownership marker with the server-authored relationship persisted
/// on the ordinary raster child. This makes the intermediate hidden and cleanup-owned in its first
/// sidecar write, rather than exposing it until the workflow coordinator's next poll.
pub(crate) fn stamp_vector_workflow_asset_writes(
    job_type: &JobType,
    payload: &JsonObject,
    job_id: &str,
    asset_writes: &mut [Value],
) {
    const OWNERSHIP_KEY: &str = "vectorWorkflowOwnership";
    for fact in asset_writes.iter_mut() {
        if let Some(object) = fact.as_object_mut() {
            object.remove(OWNERSHIP_KEY);
        }
    }
    if !matches!(job_type, JobType::ImageGenerate) {
        return;
    }
    let Some(parent_job_id) = payload
        .get("workflowParentId")
        .and_then(Value::as_str)
        .filter(|id| valid_server_workflow_id(id, "job_"))
    else {
        return;
    };
    let Some(workflow_id) = payload
        .get("workflowId")
        .and_then(Value::as_str)
        .filter(|id| valid_server_workflow_id(id, "vwf_"))
    else {
        return;
    };
    if !valid_server_workflow_id(job_id, "job_") {
        return;
    }
    let ownership = json!({
        "role": "retained_intermediate",
        "publication": "unpublished",
        "workflowId": workflow_id,
        "parentJobId": parent_job_id,
        "childJobId": job_id,
        "hidden": true,
    });
    for fact in asset_writes {
        if let Some(object) = fact.as_object_mut() {
            object.insert(OWNERSHIP_KEY.to_owned(), ownership.clone());
        }
    }
}

fn valid_server_workflow_id(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
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
    entry.insert(
        "importSourceShape".to_owned(),
        Value::String("transformer_directory".to_owned()),
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

#[cfg(test)]
mod image_manifest_projection_tests {
    use super::*;

    #[test]
    fn worker_projection_removes_only_the_exact_soft_openpose_component() {
        let primary = json!({
            "provider": "huggingface",
            "repo": "SceneWorks/sdxl-base-mlx",
            "files": ["q4/*"]
        });
        let hard = json!({
            "coRequisite": true,
            "componentId": "vae_fp16_fix",
            "repo": "madebyollin/sdxl-vae-fp16-fix"
        });
        let selected_vae = json!({
            "coRequisite": true,
            "required": "soft",
            "componentId": "vae",
            "repo": "operator/selected-vae",
            "files": ["vae.safetensors"]
        });
        let openpose = json!({
            "coRequisite": true,
            "required": "soft",
            "componentId": "controlnet_openpose",
            "repo": "xinsir/controlnet-openpose-sdxl-1.0",
            "files": ["diffusion_pytorch_model.safetensors"]
        });
        let entry = json!({
            "id": "projection-probe",
            "downloads": [
                primary.clone(),
                hard.clone(),
                selected_vae.clone(),
                openpose
            ],
            "ui": { "description": "preserved metadata" }
        });

        assert_eq!(
            project_image_manifest_for_worker(entry),
            json!({
                "id": "projection-probe",
                "downloads": [primary, hard, selected_vae],
                "ui": { "description": "preserved metadata" }
            }),
            "projection must remove only required:soft/controlnet_openpose and preserve every other Value exactly"
        );
    }
}
