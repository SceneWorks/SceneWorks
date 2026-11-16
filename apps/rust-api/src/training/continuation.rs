use super::*;
use sceneworks_core::training::{TrainingOutputKind, TrainingPlan};

#[derive(Deserialize)]
pub(crate) struct TrainingRunsQuery {
    offset: Option<u32>,
    limit: Option<u32>,
}

/// This history includes queue-cleared runs; it never adds them back to the queue.
pub(crate) async fn list_completed_training_runs(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<TrainingRunsQuery>,
) -> Result<Json<Value>, ApiError> {
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(25).clamp(1, 100);
    let mut jobs = store_call(state, move |store, _| {
        store.list_completed_lora_runs(&project_id, offset, limit + 1)
    })
    .await?;
    let has_more = jobs.len() > limit as usize;
    jobs.truncate(limit as usize);
    Ok(Json(json!({
        "jobs": public_job_snapshots(jobs),
        "nextOffset": has_more.then_some(offset.saturating_add(limit)),
    })))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ContinueTrainingRequest {
    additional_steps: u32,
}

/// A continuation is an independent job and adapter. All inputs come from the
/// immutable completed plan; the caller can change only the additional step count.
pub(crate) async fn continue_training_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(request): ApiJson<ContinueTrainingRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if request.additional_steps == 0 {
        return Err(ApiError::bad_request(
            "Additional steps must be a positive whole number.",
        ));
    }
    let source = store_call(state.clone(), {
        let job_id = job_id.clone();
        move |store, _| store.get_job(&job_id)
    })
    .await?;
    if source.job_type != JobType::LoraTrain
        || source.status != JobStatus::Completed
        || source.payload.get("dryRun").and_then(Value::as_bool) != Some(false)
    {
        return Err(ApiError::bad_request(
            "Continue training requires a completed LoRA training run.",
        ));
    }
    let mut plan: TrainingPlan =
        serde_json::from_value(source.payload.get("plan").cloned().unwrap_or(Value::Null))
            .map_err(|error| {
                ApiError::bad_request(format!("Invalid source training plan: {error}"))
            })?;
    if plan.target.output_kind != TrainingOutputKind::Lora {
        return Err(ApiError::bad_request(
            "Only LoRA outputs can be continued with this action.",
        ));
    }
    let completed_steps = match source.result.get("stepsCompleted").and_then(Value::as_u64) {
        Some(steps) => u32::try_from(steps)
            .map_err(|_| ApiError::bad_request("Invalid completed step count."))?,
        None => plan.config.steps,
    };
    if completed_steps == 0 {
        return Err(ApiError::bad_request(
            "This run completed no training steps.",
        ));
    }
    if completed_steps % plan.config.gradient_accumulation.max(1) != 0 {
        return Err(ApiError::bad_request(
            "This run finished inside a gradient accumulation window. Continuing its flushed optimizer state is not supported.",
        ));
    }
    let total_steps = completed_steps
        .checked_add(request.additional_steps)
        .ok_or_else(|| ApiError::bad_request("Total training steps exceed the supported range."))?;
    plan.config.steps = total_steps;
    let registry = effective_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == plan.target.target_id)
        .ok_or_else(|| ApiError::bad_request("This training target is no longer available."))?;
    // Stored Iris plans include paths resolved by the API. Validate the client-shaped
    // options again, then resolve this continuation's immediate parent below.
    if let Some(objective) = sceneworks_core::training::iris::iris_objective(target) {
        sceneworks_core::training::iris::strip_api_owned_paths_for_continuation(
            &mut plan.config,
            objective,
        );
    }
    sceneworks_core::training::validate_training_config_for_target(target, &plan.config)
        .map_err(training_plan_error_to_api_error)?;
    let project_id = source
        .project_id
        .clone()
        .ok_or_else(|| ApiError::bad_request("The source training project is missing."))?;
    let scope = sceneworks_core::resolved_options::TrainingExecutionOptions::for_config(
        &plan.config,
        &plan.target.kernel,
        &plan.target.base_model,
    )
    .map_err(ApiError::bad_request)?
    .common
    .output_scope;
    validate_lora_id_component(&plan.output.lora_id)?;
    let (source_dir, _) =
        resolve_training_output_location(&state, &scope, Some(&project_id), &plan.output.lora_id)
            .await?;
    let original_dir = sceneworks_worker::resolve_training_output_dir_under(
        &state.settings.data_dir,
        &plan.output.output_dir,
        "Source training outputDir",
    )
    .map_err(|error| ApiError::bad_request(error.to_string()))?;
    if source_dir.canonicalize().ok().as_ref() != Some(&original_dir) {
        return Err(ApiError::bad_request(
            "The source training output identity is inconsistent.",
        ));
    }
    let source_output = plan.output.output_dir.clone();
    let output_in_use = store_call(state.clone(), move |store, _| {
        for status in [
            JobStatus::Queued,
            JobStatus::Preparing,
            JobStatus::Downloading,
            JobStatus::LoadingModel,
            JobStatus::Running,
            JobStatus::Saving,
        ] {
            if store
                .list_jobs(None, Some(status.as_str()), 500)?
                .iter()
                .any(|job| {
                    job.payload
                        .get("plan")
                        .and_then(|plan| plan.pointer("/output/outputDir"))
                        .and_then(Value::as_str)
                        == Some(source_output.as_str())
                })
            {
                return Ok(true);
            }
        }
        Ok(false)
    })
    .await?;
    if output_in_use {
        return Err(ApiError::bad_request(
            "Another run is using the source training output; wait for it to finish.",
        ));
    }
    let parent_lora_id = plan.output.lora_id.clone();
    let fresh_job_id = format!("job_{}", Uuid::new_v4().simple());
    let fresh_lora_id = format!("lora_{}", Uuid::new_v4().simple());
    let (output_dir, _) =
        resolve_training_output_location(&state, &scope, Some(&project_id), &fresh_lora_id).await?;
    let name = source
        .payload
        .get("outputName")
        .and_then(Value::as_str)
        .unwrap_or("LoRA");
    let output_name = format!("{name} — {total_steps} steps");
    let link = json!({
        "sourceJobId": job_id, "sourceLoraId": parent_lora_id,
        "initialStep": completed_steps, "additionalSteps": request.additional_steps,
        "totalSteps": total_steps, "lrSchedulePolicy": "extended_total"
    });
    plan.job_id = fresh_job_id.clone();
    plan.output.lora_id = fresh_lora_id.clone();
    plan.output.output_dir = output_dir.display().to_string();
    plan.config.advanced.insert("resume".into(), json!(true));
    let is_iris = sceneworks_core::training::iris::is_iris_training_kernel(&plan.target.kernel);
    if is_iris {
        if let Some(objective) = sceneworks_core::training::iris::iris_objective(target) {
            sceneworks_core::training::iris::prepare_iris_continuation_options(
                &mut plan.config,
                objective,
            )
            .map_err(training_plan_error_to_api_error)?;
        }
        let root = sceneworks_worker::iris_checkpoint_root(&original_dir, &plan.output.file_name);
        let terminal = root.join(format!(
            "step_{:08}",
            completed_steps / plan.config.gradient_accumulation.max(1)
        ));
        if !terminal.join("state.json").is_file() {
            return Err(ApiError::bad_request(
                "This Iris run has no recovery checkpoint at its completed step.",
            ));
        }
        plan.config
            .advanced
            .insert("resumeFromJobId".into(), json!(job_id));
        crate::iris_training::resolve_iris_run_sources(&state, &project_id, &mut plan).await?;
        if let Some(objective) = sceneworks_core::training::iris::iris_objective(target) {
            sceneworks_core::training::iris::pin_iris_continuation_checkpoint(
                &mut plan.config,
                objective,
                &terminal.display().to_string(),
            )
            .map_err(training_plan_error_to_api_error)?;
        }
    }
    plan.provenance.source_job_id = fresh_job_id.clone();
    plan.provenance.output_lora_id = fresh_lora_id.clone();
    plan.provenance.created_at = now_rfc3339();
    plan.provenance
        .extra
        .insert("continuation".into(), link.clone());
    plan.provenance.config_snapshot = serde_json::to_value(&plan.config)
        .map_err(|error| ApiError::internal(error.to_string()))?
        .as_object()
        .cloned()
        .ok_or_else(|| ApiError::internal("Training config must be an object."))?;
    let iris_receipts =
        crate::iris_curricula::check_iris_submit(&state, &project_id, target, &plan, false).await?;
    let mut payload = source.payload.clone();
    let entry = payload
        .get_mut("manifestEntry")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| ApiError::bad_request("Source LoRA output record is missing."))?;
    entry.insert("id".into(), json!(fresh_lora_id));
    entry.insert("name".into(), json!(output_name));
    entry.insert(
        "source".into(),
        json!({"provider":"training", "path":format!("loras/{fresh_lora_id}")}),
    );
    entry.insert("createdAt".into(), json!(plan.provenance.created_at));
    entry.insert("updatedAt".into(), json!(plan.provenance.created_at));
    let provenance = entry
        .entry("provenance")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Source LoRA provenance is invalid."))?;
    provenance.insert("trainingJobId".into(), json!(fresh_job_id));
    provenance.insert("configSnapshot".into(), json!(plan.config));
    provenance.insert("continuation".into(), link.clone());
    provenance.insert("createdAt".into(), json!(plan.provenance.created_at));
    if is_iris {
        // This also changes paths inside Iris adapter exports.
        crate::iris_training::shape_iris_manifest_entry(
            entry,
            &output_dir,
            &plan.output.file_name,
            false,
            &plan.target.base_model,
        );
        if let Some(receipts) = &iris_receipts {
            entry.insert(
                crate::iris_curricula::IRIS_RECEIPTS_KEY.to_owned(),
                receipts.clone(),
            );
        }
    }
    if let Some(receipts) = iris_receipts {
        payload.insert(
            crate::iris_curricula::IRIS_RECEIPTS_KEY.to_owned(),
            receipts,
        );
    }
    payload.insert("continuation".into(), link);
    payload.insert("outputName".into(), json!(output_name));
    payload.insert("plan".into(), json!(plan));
    payload.insert("dryRun".into(), json!(false));
    // Do admission before materializing any copied state.
    let requested_gpu = source.requested_gpu.clone();
    portable_training_binding(&state, &JobType::LoraTrain, &payload, &requested_gpu).await?;
    if !is_iris {
        let source_dir = original_dir;
        let destination = output_dir.clone();
        let file_name = plan.output.file_name.clone();
        tokio::task::spawn_blocking(move || {
            sceneworks_worker::fork_completed_training_checkpoint(
                &source_dir,
                &destination,
                &file_name,
                completed_steps,
                total_steps,
            )
        })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    }
    if let Some(plan) = payload.get("plan") {
        write_training_config_snapshot(&output_dir, plan).await;
    }
    let cleanup_state = state.clone();
    let cleanup_job_id = fresh_job_id.clone();
    let result = enqueue_training_submission(
        state,
        PreparedTrainingSubmission {
            job_id: fresh_job_id,
            project_id,
            project_name: source.project_name.unwrap_or_default(),
            job_payload: payload,
            requested_gpu,
            training_job_type: JobType::LoraTrain,
            source_job_id: Some(job_id),
        },
    )
    .await;
    if result.is_err() && !is_iris {
        // Publication can fail after insertion. Never delete a queued run's recovery state.
        let not_queued = store_call(cleanup_state, move |store, _| {
            Ok(matches!(
                store.get_job(&cleanup_job_id),
                Err(sceneworks_core::jobs_store::JobsStoreError::NotFound(_))
            ))
        })
        .await
        .unwrap_or(false);
        if not_queued {
            let _ = tokio::fs::remove_dir_all(&output_dir).await;
        }
    }
    result
}
