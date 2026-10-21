//! The API half of Iris-3B curricula, REPA, holdout validation and distributed training (story
//! sc-25689, epic sc-25678), on top of the S7/S8 generation-training surface (`iris_training`):
//!
//! * `GET /api/v1/training/iris/stage-presets` — the four official stages (`stage1_256 →
//!   stage2_512 → stage3_1024 → sft_1024`) as editable configs, translated by the ENGINE from its
//!   vendored upstream YAML and normalized onto the offered surface (the worker crate);
//! * `POST /api/v1/training/iris/plan-check` — the pre-launch view of a config: the global batch
//!   (`batch × ranks × accumulation`), the topology, the declared memory floor per device, the REPA
//!   teacher's install state, and — when the backbone is installed and this build links an engine —
//!   the engine's own resolution with its typed refusal (invalid mesh, global-batch mismatch, …);
//! * `GET /api/v1/training/jobs/:job_id/iris-metrics` — the run's `<checkpoints>/metrics.jsonl`
//!   (run header, logged steps, holdout validation, block grad norms), read from the output
//!   directory recomputed from the job's trusted ids;
//! * submit-time checks + receipts ([`check_iris_submit`]): the curriculum lineage of a parent run
//!   (a continuing stage names the run of its parent stage and continues it as a new phase), the
//!   REPA drop that a replicated full-optimizer resume refuses, the teacher install, the engine plan
//!   resolution, and the receipts (stage, parent, data identity, topology) recorded on the job and
//!   on the registered output.

use super::*;

use sceneworks_core::training::iris::curricula::{
    self, iris_curriculum_receipt, iris_dist_world, iris_repa_enabled, IRIS_REPA_TEACHER_MODEL_ID,
    IRIS_REPA_TEACHER_REPO, IRIS_REPA_TEACHER_REVISION, IRIS_STAGES,
};
use sceneworks_core::training::iris::{
    iris_options, is_iris_training_kernel, IrisObjective, IRIS_OPTIONS_KEY, RESUME_FROM_JOB_KEY,
};
use sceneworks_core::training::{TrainingConfig, TrainingPlan, TrainingTarget};

/// The job-payload and manifest-entry key carrying an Iris run's receipts.
pub(crate) const IRIS_RECEIPTS_KEY: &str = "irisReceipts";
/// Upper bound on the metrics bytes one request reads (the tail of the file).
const METRICS_READ_CAP: u64 = 8 * 1024 * 1024;

/// `GET /api/v1/training/iris/stage-presets`.
pub(crate) async fn iris_stage_presets() -> Result<Json<Value>, ApiError> {
    let presets = sceneworks_worker::iris_stage_presets().map_err(ApiError::internal)?;
    Ok(Json(json!({ "presets": presets })))
}

fn iris_target(target_id: &str) -> Result<TrainingTarget, ApiError> {
    sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == target_id && is_iris_training_kernel(&target.kernel))
        .ok_or_else(|| {
            ApiError::bad_request(format!("{target_id} is not an Iris training target."))
        })
}

/// The objective of an Iris target.
fn target_objective(target: &TrainingTarget) -> Option<IrisObjective> {
    IrisObjective::from_base_model(&target.base_model)
}

/// The typed install refusal of an Iris target's run (S7/S8): what it needs that is not installed,
/// judged per task resource (`None` = ready).
fn install_refusal(
    state: &AppState,
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Option<sceneworks_worker::IrisInstallRefusal> {
    target_objective(target).and_then(|objective| {
        sceneworks_worker::iris_training_install_refusal(
            &state.settings.data_dir,
            objective,
            config,
        )
    })
}

/// An Iris install refusal as the typed `iris_training_refused` API error.
pub(crate) fn iris_install_refused(refusal: sceneworks_worker::IrisInstallRefusal) -> ApiError {
    refused(refusal.kind, refusal.message)
}

/// The installed backbone directory of an Iris target when its `config.yaml` is present (the engine
/// resolution reads it); `None` when the base is not installed. Generation reads the pinned
/// generation backbone itself (root `config.yaml` + `model.safetensors`), never the repo-level
/// marker a task export satisfies (S7).
fn installed_backbone(state: &AppState, target: &TrainingTarget) -> Option<PathBuf> {
    let data_dir = &state.settings.data_dir;
    if target_objective(target) == Some(IrisObjective::Generation) {
        return sceneworks_worker::iris_generation_backbone_dir(data_dir);
    }
    if !matches!(
        crate::training::training_base_model_status(data_dir, target),
        crate::training::TrainingBaseStatus::Ready
    ) {
        return None;
    }
    let dir = PathBuf::from(crate::training::resolve_base_model_path(target, data_dir));
    dir.join("config.yaml").is_file().then_some(dir)
}

fn teacher_status(state: &AppState) -> Value {
    json!({
        "modelId": IRIS_REPA_TEACHER_MODEL_ID,
        "repo": IRIS_REPA_TEACHER_REPO,
        "revision": IRIS_REPA_TEACHER_REVISION,
        "license": "Apache-2.0",
        "installed": sceneworks_worker::installed_repa_teacher(&state.settings.data_dir).is_some(),
    })
}

/// The topology refusal of a config against the GPU workers this API has registered (sc-25689): a
/// job runs on ONE worker, so more ranks than the biggest worker's devices need `share_device`.
/// Judged on every platform; the device count is what the workers registered (their `gpuId`s), 1
/// when none has registered yet.
async fn topology_refusal(
    state: &AppState,
    config: &TrainingConfig,
) -> Result<Option<String>, ApiError> {
    let workers = store_call(state.clone(), |store, _timeout| store.list_workers()).await?;
    let devices = curricula::max_worker_devices(
        workers
            .iter()
            .filter(|worker| !matches!(worker.status, WorkerStatus::Offline))
            .map(|worker| worker.gpu_id.as_str()),
    );
    let noun = if cfg!(target_os = "macos") {
        "Mac GPU"
    } else {
        "GPU"
    };
    Ok(curricula::iris_topology_refusal(config, devices, noun))
}

fn refusal(kind: &str, message: impl Into<String>) -> Value {
    json!({ "kind": kind, "message": message.into() })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlanCheckRequest {
    target_id: String,
    config: TrainingConfig,
}

/// The engine's resolution of a config against `backbone` (native builds), as `Ok(json)` or the
/// typed refusal; `None` when this build links no engine.
fn engine_check(backbone: &FsPath, config: &TrainingConfig) -> Option<Result<Value, Value>> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        Some(
            sceneworks_worker::resolve_iris_config(backbone, config)
                .map(|resolved| resolved.to_json())
                .map_err(|refused| refused.to_json()),
        )
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = (backbone, config);
        None
    }
}

/// `POST /api/v1/training/iris/plan-check` — what the run would be, and the first refusal, before
/// launch. Always 200 for a well-formed request: the Studio renders `ok` / `refusal`.
pub(crate) async fn iris_plan_check(
    State(state): State<AppState>,
    Json(request): Json<PlanCheckRequest>,
) -> Result<Json<Value>, ApiError> {
    let target = iris_target(&request.target_id)?;
    let config = request.config;
    let options = iris_options(&config).unwrap_or_default();
    let (world_size, share_device, timeout_secs) = iris_dist_world(&options);
    let global_batch = curricula::iris_global_batch(&config, world_size);
    let teacher = teacher_status(&state);
    let mut first_refusal: Option<Value> = None;

    if let Err(error) =
        sceneworks_core::training::validate_training_config_for_target(&target, &config)
    {
        first_refusal = Some(refusal("invalid", error.to_string()));
    }
    // The run's own install (S7): a missing backbone / text encoder / task export is a typed
    // refusal here, so the engine resolution below never silently skips for want of a backbone.
    if first_refusal.is_none() {
        if let Some(missing) = install_refusal(&state, &target, &config) {
            first_refusal = Some(refusal(missing.kind, missing.message));
        }
    }
    if first_refusal.is_none()
        && iris_repa_enabled(&options)
        && teacher["installed"] != Value::Bool(true)
    {
        first_refusal = Some(refusal(
            "missing_teacher",
            "REPA needs the DINOv2-B/14 teacher — install \"Iris REPA Teacher — DINOv2-B/14\" in \
             the Model Manager.",
        ));
    }
    // The topology every platform can judge before launch (the registered workers' devices); the
    // memory floor only where this API can read the device (the co-located Mac) — off-Mac the
    // worker that claims the job reads its own VRAM.
    if first_refusal.is_none() {
        if let Some(message) = topology_refusal(&state, &config).await? {
            first_refusal = Some(refusal("unsupported_topology", message));
        }
    }
    #[cfg(target_os = "macos")]
    {
        if first_refusal.is_none() {
            if let Some(message) =
                sceneworks_worker::iris_plan_memory_error(&target.base_model, &config)
            {
                first_refusal = Some(refusal("memory", message));
            }
        }
    }
    let mut resolved = Value::Null;
    let mut engine_checked = false;
    if first_refusal.is_none() {
        if let Some(backbone) = installed_backbone(&state, &target) {
            if let Some(outcome) = engine_check(&backbone, &config) {
                engine_checked = true;
                match outcome {
                    Ok(plan) => resolved = plan,
                    Err(refused) => first_refusal = Some(refused),
                }
            }
        }
    }
    let lane = if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    };
    Ok(Json(json!({
        "ok": first_refusal.is_none(),
        "refusal": first_refusal,
        "globalBatch": global_batch,
        "expectedGlobalBatch": options.get("expected_global_batch").cloned().unwrap_or(Value::Null),
        "worldSize": world_size,
        "shareDevice": share_device,
        "timeoutSecs": timeout_secs,
        "memory": iris_rank_memory_json(lane, &config),
        "teacher": teacher,
        "receipt": iris_curriculum_receipt(&config),
        "engineChecked": engine_checked,
        "resolved": resolved,
        "stages": IRIS_STAGES
            .iter()
            .map(|(id, parent)| json!({ "id": id, "parentStage": parent }))
            .collect::<Vec<_>>(),
    })))
}

/// One rank's floor on `lane` for the curriculum check: GiB, with the provenance of the row it was
/// read from (never a blanket "measured" flag). A null floor = no row covers the shape (refused).
fn iris_rank_memory_json(lane: &str, config: &TrainingConfig) -> Value {
    let floor = curricula::iris_rank_memory_floor(lane, config);
    json!({
        "lane": lane,
        "unit": sceneworks_core::training::iris::floors::FLOOR_UNIT,
        "perRankFloorGib": floor.map(|floor| floor.gib),
        "provenance": floor.map(|floor| floor.provenance),
    })
}

fn plan_stage(plan: &TrainingPlan) -> Option<String> {
    plan.config
        .advanced
        .get(IRIS_OPTIONS_KEY)
        .and_then(|iris| iris.get("stage"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn refused(kind: &'static str, message: impl Into<String>) -> ApiError {
    let message = message.into();
    ApiError::typed(
        StatusCode::BAD_REQUEST,
        message.clone(),
        "iris_training_refused",
        json!({ "kind": kind, "message": message }),
    )
}

/// The submit-time Iris S11 checks of a built plan, run after the run sources are resolved.
/// Returns the receipts recorded on the job and on the registered output (`None` for a non-Iris
/// plan).
pub(crate) async fn check_iris_submit(
    state: &AppState,
    project_id: &str,
    target: &TrainingTarget,
    plan: &TrainingPlan,
    dry_run: bool,
) -> Result<Option<Value>, ApiError> {
    if !is_iris_training_kernel(&plan.target.kernel) {
        return Ok(None);
    }
    let options =
        iris_options(&plan.config).map_err(|error| ApiError::bad_request(error.to_string()))?;
    let stage = options.get("stage").and_then(Value::as_str);
    let parent_stage = options.get("parent_stage").and_then(Value::as_str);
    let resume_job = plan
        .config
        .advanced
        .get(RESUME_FROM_JOB_KEY)
        .and_then(Value::as_str);

    // ---- curriculum lineage: a continuing stage names the run of its parent stage ------------
    if let (Some(stage), Some(parent), None) = (stage, parent_stage, resume_job) {
        return Err(refused(
            "lineage",
            format!(
                "Stage {stage} continues {parent}: choose the {parent} run it continues (its newest \
                 checkpoint is resumed as a new phase), or clear the parent stage to train {stage} \
                 on its own."
            ),
        ));
    }
    if let Some(job_id) = resume_job {
        let (_, previous, _) = crate::iris_training::previous_iris_run(
            state,
            project_id,
            job_id,
            RESUME_FROM_JOB_KEY,
            &plan.target.base_model,
        )
        .await?;
        let previous_stage = plan_stage(&previous);
        if let Some(stage) = stage {
            let policy = options
                .get("resume_data_policy")
                .and_then(Value::as_str)
                .unwrap_or("exact");
            match previous_stage.as_deref() {
                Some(saved) if saved == stage => {}
                Some(saved) if Some(saved) == parent_stage => {
                    if policy != "new_phase" {
                        return Err(refused(
                            "lineage",
                            format!(
                                "{job_id} trained stage {saved}; stage {stage} continues it as a \
                                 new phase — set the resume data policy to new_phase."
                            ),
                        ));
                    }
                }
                saved => {
                    return Err(refused(
                        "lineage",
                        format!(
                            "{job_id} trained {}; stage {stage} continues {} (or restarts \
                             itself).",
                            saved
                                .map(|saved| format!("stage {saved}"))
                                .unwrap_or_else(|| "no curriculum stage".to_owned()),
                            parent_stage.unwrap_or("no parent stage")
                        ),
                    ))
                }
            }
        }
        // A projector the resumed run drops: a replicated full-optimizer resume refuses the changed
        // parameter routing (the engine's `check_resume`, upstream's `_validate_optimizer_layout`).
        let previous_options = iris_options(&previous.config).unwrap_or_default();
        let strategy = options
            .get("dist")
            .and_then(|dist| dist.get("strategy"))
            .and_then(Value::as_str)
            .unwrap_or("ddp");
        let core_resume = options.get("resume_optimizer").and_then(Value::as_str) == Some("core");
        if iris_repa_enabled(&previous_options)
            && !iris_repa_enabled(&options)
            && strategy == "ddp"
            && !core_resume
        {
            return Err(refused(
                "unsupported",
                format!(
                    "{job_id} trained a REPA projector (and its optimizer state) that this run \
                     drops. A full optimizer resume of a replicated (ddp) run refuses the changed \
                     parameter routing, as upstream does — set resume_optimizer to core to resume \
                     the backbone's optimizer state without the projector."
                ),
            ));
        }
    }

    // ---- the run's own install (S7): typed, before the engine resolution needs the backbone -------
    if !dry_run {
        if let Some(missing) = install_refusal(state, target, &plan.config) {
            return Err(iris_install_refused(missing));
        }
    }

    // ---- the REPA teacher ----------------------------------------------------------------------
    if iris_repa_enabled(&options)
        && !dry_run
        && sceneworks_worker::installed_repa_teacher(&state.settings.data_dir).is_none()
    {
        return Err(refused(
            "missing_teacher",
            format!(
                "REPA needs the DINOv2-B/14 teacher ({IRIS_REPA_TEACHER_REPO}) — install \"Iris \
                 REPA Teacher — DINOv2-B/14\" in the Model Manager. Nothing is downloaded during \
                 training."
            ),
        ));
    }

    // ---- the topology the registered workers can host (every platform) ---------------------------
    if !dry_run {
        if let Some(message) = topology_refusal(state, &plan.config).await? {
            return Err(refused("unsupported_topology", message));
        }
    }

    // ---- the engine's own resolution ------------------------------------------------------------
    #[cfg_attr(
        not(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        )),
        allow(unused_mut)
    )]
    let mut resolved = Value::Null;
    if !dry_run {
        if let Some(backbone) = installed_backbone(state, target) {
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            match sceneworks_worker::resolve_iris_plan(&backbone, plan) {
                Ok(plan) => resolved = plan.to_json(),
                Err(error) => {
                    let message = error.message();
                    return Err(ApiError::typed(
                        StatusCode::BAD_REQUEST,
                        message.clone(),
                        "iris_training_refused",
                        json!({ "kind": error.kind, "message": message }),
                    ));
                }
            }
            #[cfg(not(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            )))]
            let _ = backbone;
        }
    }

    // ---- receipts --------------------------------------------------------------------------------
    let mut receipt = iris_curriculum_receipt(&plan.config);
    if let Some(object) = receipt.as_object_mut() {
        object.insert(
            "dataset".to_owned(),
            json!({
                "id": plan.provenance.dataset_id,
                "version": plan.provenance.dataset_version,
                "items": plan.dataset.items.len(),
                "holdoutItems": plan
                    .dataset
                    .items
                    .iter()
                    .filter(|item| {
                        item.extra.get(curricula::ITEM_SPLIT_KEY).and_then(Value::as_str)
                            == Some(curricula::ITEM_SPLIT_HOLDOUT)
                    })
                    .count(),
            }),
        );
        if !resolved.is_null() {
            for key in [
                "dataIdentity",
                "stateIdentity",
                "shapePolicy",
                "dist",
                "globalBatch",
            ] {
                if let Some(value) = resolved.get(key) {
                    object.insert(format!("resolved{}", capitalize(key)), value.clone());
                }
            }
        }
    }
    Ok(Some(receipt))
}

fn capitalize(key: &str) -> String {
    let mut chars = key.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// `GET /api/v1/training/jobs/:job_id/iris-metrics` — the run's metrics file, grouped.
pub(crate) async fn iris_training_metrics(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let job = store_call(state.clone(), {
        let job_id = job_id.clone();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await?;
    let plan: TrainingPlan = job
        .payload
        .get("plan")
        .cloned()
        .and_then(|plan| serde_json::from_value(plan).ok())
        .ok_or_else(|| ApiError::bad_request(format!("{job_id} has no training plan.")))?;
    if !is_iris_training_kernel(&plan.target.kernel) {
        return Err(ApiError::bad_request(format!(
            "{job_id} is not an Iris training run."
        )));
    }
    let output_dir = crate::iris_training::trusted_previous_output_dir(&state, &job).await?;
    crate::iris_training::file_stem(&plan.output.file_name)
        .ok_or_else(|| ApiError::bad_request(format!("{job_id} has no valid output name.")))?;
    let path = sceneworks_worker::iris_checkpoint_root(&output_dir, &plan.output.file_name)
        .join("metrics.jsonl");
    let text = match tokio::fs::File::open(&path).await {
        Ok(mut file) => {
            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let length = file.metadata().await.map(|meta| meta.len()).unwrap_or(0);
            let start = length.saturating_sub(METRICS_READ_CAP);
            file.seek(std::io::SeekFrom::Start(start))
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            if start > 0 {
                // Drop the partial first line of a tail read.
                text.split_once('\n')
                    .map(|(_, rest)| rest.to_owned())
                    .unwrap_or_default()
            } else {
                text
            }
        }
        Err(_) => String::new(),
    };
    let (mut run, mut steps, mut validation, mut grad_norms) =
        (Value::Null, Vec::new(), Vec::new(), Vec::new());
    for record in text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
    {
        match record.get("kind").and_then(Value::as_str) {
            Some("run") => run = record,
            Some("step") => steps.push(record),
            Some("validation") => validation.push(record),
            Some("grad_norms") => grad_norms.push(record),
            _ => {}
        }
    }
    Ok(Json(json!({
        "jobId": job_id,
        "available": !text.is_empty(),
        "run": run,
        "steps": steps,
        "validation": validation,
        "gradNorms": grad_norms,
    })))
}
