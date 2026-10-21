//! The API half of Iris-3B generation training (epic sc-25678; S7 sc-25685 MLX, S8 sc-25686
//! Candle/CUDA): what `create_training_job` and the completion registrars need beyond the generic
//! native-training path every family shares.
//!
//! * the pre-launch **data walk** endpoint — the trainer's own unused-tail condition
//!   (`gen_core::iris::train::DataWalk`, via the worker crate), so the Studio can warn before Start;
//! * **resume / new phase / load-from** — a client names a previous Iris training JOB, never a
//!   path; the run's checkpoint root or export is recomputed from that job's trusted ids exactly as
//!   the registrars recompute an output dir, and stamped into the engine options bag;
//! * the **full-model export** registration — `<output_dir>/<stem>/{config.yaml, model.safetensors}`
//!   becomes a user image model carrying its Iris task identity (`trainedCheckpoint`), selectable
//!   for generation without file surgery;
//! * the generation-time **alias** — a request for such a checkpoint runs the `iris_3b` generation
//!   engine on the checkpoint directory (`modelPath`) with the generation install's text encoder,
//!   so routing, admission, the co-requisite staging and the worker lane are the builtin ones.

use super::*;
use crate::training::is_plain_relative_file;

use sceneworks_core::training::iris::{
    is_iris_training_kernel, IrisObjective, IRIS_BASE_MODEL, IRIS_FAMILY, IRIS_GENERATION_TASK,
    LOAD_FROM_JOB_KEY, RESUME_FROM_JOB_KEY,
};
use sceneworks_core::training::TrainingPlan;

/// The catalog-entry block that marks a user model as a trained checkpoint of a builtin generation
/// engine (the Iris full-model export).
pub(crate) const TRAINED_CHECKPOINT_KEY: &str = "trainedCheckpoint";
/// Payload key recording which catalog model a generation request named when it was aliased onto
/// its engine (provenance + retry/duplicate re-canonicalization).
pub(crate) const CHECKPOINT_MODEL_ID_KEY: &str = "checkpointModelId";
const BACKBONE_CONFIG_FILE: &str = "config.yaml";
const BACKBONE_WEIGHTS_FILE: &str = "model.safetensors";
const CHECKPOINT_STATE_FILE: &str = "state.json";
/// The third file of a depth / restoration task export (`gen_core::iris::downstream`).
const EMPTY_PROMPT_FILE: &str = "empty_prompt.safetensors";

/// The objective of a plan (`None` for a non-Iris plan).
fn plan_objective(plan: &TrainingPlan) -> Option<IrisObjective> {
    is_iris_training_kernel(&plan.target.kernel)
        .then(|| IrisObjective::from_base_model(&plan.target.base_model))
        .flatten()
}

/// The files of an objective's exported model directory.
fn export_files(objective: IrisObjective) -> &'static [&'static str] {
    match objective {
        IrisObjective::Generation => &[BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE],
        _ => &[
            BACKBONE_CONFIG_FILE,
            BACKBONE_WEIGHTS_FILE,
            EMPTY_PROMPT_FILE,
        ],
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DataWalkQuery {
    pub(crate) target_id: String,
    pub(crate) items: usize,
    #[serde(default = "default_batch_size")]
    pub(crate) batch_size: usize,
}

fn default_batch_size() -> usize {
    1
}

/// `GET /api/v1/training/data-walk?targetId=&items=&batchSize=` — the trainer's data walk for a
/// dataset of `items` at `batchSize`, so the Studio can show the unused tail BEFORE launch. Only the
/// Iris target walks this way (`applies: false` for every other target).
pub(crate) async fn training_data_walk(
    Query(query): Query<DataWalkQuery>,
) -> Result<Json<Value>, ApiError> {
    let registry = sceneworks_core::training::builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == query.target_id)
        .ok_or_else(|| {
            ApiError::bad_request(format!("Unknown training target: {}", query.target_id))
        })?;
    if !is_iris_training_kernel(&target.kernel) {
        return Ok(Json(json!({ "applies": false })));
    }
    if query.batch_size == 0 {
        return Err(ApiError::bad_request("batchSize must be at least 1."));
    }
    let Some(report) = sceneworks_worker::iris_data_walk_report(query.items, query.batch_size)
    else {
        return Ok(Json(json!({
            "applies": true,
            "items": query.items,
            "covered": 0,
            "unusedItems": 0,
            "chunks": 640,
            "warning": null,
        })));
    };
    let mut value =
        serde_json::to_value(&report).map_err(|error| ApiError::internal(error.to_string()))?;
    if let Some(object) = value.as_object_mut() {
        object.insert("applies".to_owned(), Value::Bool(true));
    }
    Ok(Json(value))
}

fn plan_network_type(plan: &TrainingPlan) -> String {
    let default = match plan_objective(plan) {
        Some(IrisObjective::Depth | IrisObjective::Restoration) => "full",
        _ => "lora",
    };
    plan.config
        .advanced
        .get("networkType")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_ascii_lowercase()
}

pub(crate) fn file_stem(file_name: &str) -> Option<String> {
    if !is_plain_relative_file(file_name) {
        return None;
    }
    FsPath::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
}

/// The output directory a previous training job wrote, RECOMPUTED from its trusted ids (the same
/// discipline as the completion registrars): never read back from the mutable job payload's paths.
pub(crate) async fn trusted_previous_output_dir(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<PathBuf, ApiError> {
    let entry = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("The previous training run has no output record."))?;
    let id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("The previous training run has no output id."))?;
    validate_lora_id_component(id)?;
    let full = job
        .payload
        .get("plan")
        .and_then(|plan| plan.get("target"))
        .and_then(|target| target.get("outputKind"))
        .and_then(Value::as_str)
        == Some(sceneworks_core::training::TrainingOutputKind::BaseCheckpoint.as_str());
    if full {
        return Ok(resolve_finetune_output_location(state, id).0);
    }
    let scope = entry
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("project");
    Ok(
        resolve_training_output_location(state, scope, job.project_id.as_deref(), id)
            .await?
            .0,
    )
}

/// The Iris training plan of a `LoraTrain` job (`None` for every other job).
fn iris_training_plan(job: &JobSnapshot) -> Option<TrainingPlan> {
    if !matches!(job.job_type, JobType::LoraTrain) {
        return None;
    }
    let plan: TrainingPlan = serde_json::from_value(job.payload.get("plan")?.clone()).ok()?;
    is_iris_training_kernel(&plan.target.kernel).then_some(plan)
}

/// The output id (the registered model / adapter id, and the output directory's name) a training
/// job writes and registers under.
fn training_output_id(job: &JobSnapshot) -> Option<&str> {
    job.payload
        .get("manifestEntry")
        .and_then(|entry| entry.get("id"))
        .and_then(Value::as_str)
}

fn output_in_use(message: String) -> ApiError {
    ApiError::typed(
        StatusCode::BAD_REQUEST,
        message.clone(),
        "iris_training_refused",
        json!({ "kind": "output_in_use", "message": message }),
    )
}

/// Replace every occurrence of `from` inside the strings of `value` (an output id is
/// `lora_<32 hex>` / `finetune_<32 hex>`, so it never occurs by accident).
fn replace_in_strings(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(text) if text.contains(from) => *text = text.replace(from, to),
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| replace_in_strings(item, from, to)),
        Value::Object(object) => object
            .values_mut()
            .for_each(|item| replace_in_strings(item, from, to)),
        _ => {}
    }
}

/// The guard on rerunning an Iris training job (S9a).
///
/// `retry_job` / `duplicate_job` copy a training job's payload verbatim — its plan's output
/// directory AND its output id (`manifestEntry.id`), which is the id the completion registrar
/// registers the export under. One output id is one run: two jobs sharing it write the same export
/// and register over the same model / adapter, whatever the trainer does with an earlier run's
/// checkpoints (since the sc-25690 inference pin the trainer moves a foreign run's checkpoints aside
/// into the one sibling `<root>.prior` — `retire_foreign_checkpoints` — instead of resuming them, so a fresh
/// run in a used directory no longer mixes checkpoints, but the export and registration still
/// collide). So:
///
/// * a **duplicate** is a NEW run: it gets a fresh output id and directory (`Some(changes)` — the
///   rewritten `plan` + `manifestEntry` to merge over the copied payload), like YuE2's
///   `refresh_block_for_duplicate`;
/// * a **retry** is another attempt of the SAME run under the same output id, so it is refused
///   while any queued / running job writes that output id (the source itself included), and once
///   the run has a complete checkpoint (resume it instead) or an export (it finished — duplicate it
///   or submit a new run). A run that died before its first complete checkpoint retries as before.
pub(crate) async fn guard_iris_training_rerun(
    state: &AppState,
    job: &JobSnapshot,
    duplicate: bool,
) -> Result<Option<JsonObject>, ApiError> {
    let Some(plan) = iris_training_plan(job) else {
        return Ok(None);
    };
    let Some(output_id) = training_output_id(job).map(str::to_owned) else {
        return Ok(None);
    };
    validate_lora_id_component(&output_id)?;
    if duplicate {
        let prefix = output_id
            .split_once('_')
            .map(|(prefix, _)| prefix)
            .unwrap_or("lora");
        let fresh_id = format!("{prefix}_{}", Uuid::new_v4().simple());
        let mut fresh = job.clone();
        for key in ["plan", "manifestEntry"] {
            if let Some(value) = fresh.payload.get_mut(key) {
                replace_in_strings(value, &output_id, &fresh_id);
            }
        }
        // The directory the registrar will recompute from the fresh id must be the one the plan
        // now names, or the run would write where nothing registers it.
        let fresh_dir = trusted_previous_output_dir(state, &fresh).await?;
        let planned_dir = fresh
            .payload
            .get("plan")
            .and_then(|plan| plan.pointer("/output/outputDir"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        if planned_dir.as_deref() != Some(fresh_dir.as_path()) {
            return Err(ApiError::internal(format!(
                "duplicating training job {}: its output folder does not follow its output id",
                job.id
            )));
        }
        let mut changes = JsonObject::new();
        for key in ["plan", "manifestEntry"] {
            if let Some(value) = fresh.payload.remove(key) {
                changes.insert(key.to_owned(), value);
            }
        }
        if let Some(plan) = changes.get("plan") {
            crate::training::write_training_config_snapshot(&fresh_dir, plan).await;
        }
        return Ok(Some(changes));
    }

    // ---- retry: the same output id -----------------------------------------------------------
    let active = store_call(state.clone(), {
        let project_id = job.project_id.clone();
        move |store, _timeout| store.list_jobs(project_id.as_deref(), None, 500)
    })
    .await?;
    if let Some(writer) = active.iter().find(|other| {
        matches!(other.job_type, JobType::LoraTrain)
            && !matches!(
                other.status,
                JobStatus::Completed
                    | JobStatus::Failed
                    | JobStatus::Canceled
                    | JobStatus::Interrupted
            )
            && training_output_id(other) == Some(output_id.as_str())
    }) {
        return Err(output_in_use(format!(
            "Training job {} is still queued or running into this run's output ({output_id}); \
             wait for it to finish or cancel it before retrying. Duplicate the run to train a \
             separate copy.",
            writer.id
        )));
    }
    let Ok(output_dir) = trusted_previous_output_dir(state, job).await else {
        return Ok(None);
    };
    let Some(stem) = file_stem(&plan.output.file_name) else {
        return Ok(None);
    };
    if newest_complete_checkpoint(&sceneworks_worker::iris_checkpoint_root(
        &output_dir,
        &plan.output.file_name,
    ))
    .is_some()
    {
        return Err(output_in_use(format!(
            "Training run {} already saved a complete checkpoint; a retry would start the run \
             over under the same output. Resume it (or start a new phase from it) in the Training \
             Studio, or duplicate it to train a separate copy.",
            job.id
        )));
    }
    if output_dir.join(&stem).exists() || output_dir.join(&plan.output.file_name).exists() {
        return Err(output_in_use(format!(
            "Training run {} already exported its result; a retry would overwrite it and its \
             registration. Duplicate the run (a duplicate trains under its own output) or submit a \
             new run.",
            job.id
        )));
    }
    Ok(None)
}

/// A complete checkpoint (`step_XXXXXXXX/state.json`) under a run's checkpoint root, newest first.
fn newest_complete_checkpoint(root: &FsPath) -> Option<PathBuf> {
    let mut steps: Vec<(u64, PathBuf)> = std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            let step = name.strip_prefix("step_")?.parse::<u64>().ok()?;
            let path = entry.path();
            path.join(CHECKPOINT_STATE_FILE)
                .is_file()
                .then_some((step, path))
        })
        .collect();
    steps.sort();
    steps.pop().map(|(_, path)| path)
}

/// The checkpoint directories of the Iris training run a catalog entry (a trained adapter, or a
/// full / depth / restoration export) was registered from, under its run directory `run_dir`: the
/// checkpoint root and the single `.prior` sibling the trainers retire foreign or abandoned
/// checkpoints into (sc-25690), named by the engine contract
/// ([`sceneworks_worker::iris_run_checkpoint_dirs`]). Deleting the trained model or adapter deletes
/// them by name. Empty for every entry that is not an Iris training output.
pub(crate) fn iris_run_checkpoint_paths(entry: &Value, run_dir: &FsPath) -> Vec<PathBuf> {
    let Some(object) = entry.as_object() else {
        return Vec::new();
    };
    let trained = object
        .get("source")
        .and_then(|source| source.get("provider"))
        .and_then(Value::as_str)
        == Some("training");
    let iris = object.contains_key("irisTask") || object.contains_key(TRAINED_CHECKPOINT_KEY);
    if !trained || !iris || object.get("catalogScope").and_then(Value::as_str) == Some("builtin") {
        return Vec::new();
    }
    // A full export is the directory `<run_dir>/<stem>` (`paths.model`); an adapter is the file
    // the run named (`files[0]`). Either way the engine keyed its checkpoint root on that stem.
    let file_name = object
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .and_then(|export| FsPath::new(export).file_name()?.to_str())
        .map(|stem| format!("{stem}.safetensors"))
        .or_else(|| {
            object
                .get("files")?
                .as_array()?
                .first()?
                .as_str()
                .map(str::to_owned)
        });
    match file_name.filter(|name| is_plain_relative_file(name)) {
        Some(file_name) => sceneworks_worker::iris_run_checkpoint_dirs(run_dir, &file_name).into(),
        None => Vec::new(),
    }
}

/// Load and vet the previous Iris training job a request names.
pub(crate) async fn previous_iris_run(
    state: &AppState,
    project_id: &str,
    job_id: &str,
    key: &str,
    base_model: &str,
) -> Result<(JobSnapshot, TrainingPlan, PathBuf), ApiError> {
    if !job_id.starts_with("job_")
        || !job_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return Err(ApiError::bad_request(format!(
            "advanced.{key} must name a training job id."
        )));
    }
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .map_err(|_| ApiError::bad_request(format!("advanced.{key}: no training job {job_id}.")))?;
    if !matches!(job.job_type, JobType::LoraTrain) || job.project_id.as_deref() != Some(project_id)
    {
        return Err(ApiError::bad_request(format!(
            "advanced.{key}: {job_id} is not a training run of this project."
        )));
    }
    if !matches!(
        job.status,
        JobStatus::Completed | JobStatus::Canceled | JobStatus::Failed | JobStatus::Interrupted
    ) {
        return Err(ApiError::bad_request(format!(
            "advanced.{key}: {job_id} is still running; wait for it to finish or cancel it."
        )));
    }
    let plan: TrainingPlan = job
        .payload
        .get("plan")
        .cloned()
        .and_then(|plan| serde_json::from_value(plan).ok())
        .ok_or_else(|| {
            ApiError::bad_request(format!("advanced.{key}: {job_id} has no training plan."))
        })?;
    // A run continues / starts from a run of the SAME objective (generation, depth, restoration).
    if !is_iris_training_kernel(&plan.target.kernel) || plan.target.base_model != base_model {
        let task =
            IrisObjective::from_base_model(base_model).map_or("", |objective| objective.task());
        return Err(ApiError::bad_request(format!(
            "advanced.{key}: {job_id} is not an Iris 3B {task} training run."
        )));
    }
    let output_dir = trusted_previous_output_dir(state, &job).await?;
    Ok((job, plan, output_dir))
}

/// Resolve a real Iris run's `resumeFromJobId` / `loadFromJobId` into the engine options bag's
/// `resume_from` / `load_from` (absolute paths SceneWorks derived and validated). Runs AFTER the plan
/// is built: the client config never carries a path (the submit validation refuses one).
pub(crate) async fn resolve_iris_run_sources(
    state: &AppState,
    project_id: &str,
    plan: &mut TrainingPlan,
) -> Result<(), ApiError> {
    let Some(objective) = plan_objective(plan) else {
        return Ok(());
    };
    let options_key = objective.options_key();
    let base_model = plan.target.base_model.clone();
    let resume_job = plan
        .config
        .advanced
        .get(RESUME_FROM_JOB_KEY)
        .and_then(Value::as_str)
        .map(str::to_owned);
    let load_job = plan
        .config
        .advanced
        .get(LOAD_FROM_JOB_KEY)
        .and_then(Value::as_str)
        .map(str::to_owned);
    if resume_job.is_none() && load_job.is_none() {
        return Ok(());
    }
    let network_type = plan_network_type(plan);
    let mut options = match plan.config.advanced.remove(options_key) {
        Some(Value::Object(map)) => map,
        _ => JsonObject::new(),
    };

    if let Some(job_id) = resume_job {
        let (_, previous, output_dir) =
            previous_iris_run(state, project_id, &job_id, RESUME_FROM_JOB_KEY, &base_model).await?;
        if plan_network_type(&previous) != network_type {
            return Err(ApiError::bad_request(format!(
                "advanced.{RESUME_FROM_JOB_KEY}: {job_id} trained a {} run; a resume continues the \
                 same kind of run ({} requested).",
                plan_network_type(&previous),
                network_type
            )));
        }
        file_stem(&previous.output.file_name).ok_or_else(|| {
            ApiError::bad_request(format!(
                "advanced.{RESUME_FROM_JOB_KEY}: {job_id} has no valid output name."
            ))
        })?;
        let root = sceneworks_worker::iris_checkpoint_root(&output_dir, &previous.output.file_name);
        if newest_complete_checkpoint(&root).is_none() {
            return Err(ApiError::bad_request(format!(
                "advanced.{RESUME_FROM_JOB_KEY}: {job_id} saved no complete checkpoint to resume \
                 from (it stopped before its first checkpoint)."
            )));
        }
        options.insert(
            "resume_from".to_owned(),
            Value::String(root.display().to_string()),
        );
        plan.config
            .advanced
            .insert("resume".to_owned(), Value::Bool(true));
    }

    if let Some(job_id) = load_job {
        let (_, previous, output_dir) =
            previous_iris_run(state, project_id, &job_id, LOAD_FROM_JOB_KEY, &base_model).await?;
        if plan_network_type(&previous) != "full" {
            return Err(ApiError::bad_request(format!(
                "advanced.{LOAD_FROM_JOB_KEY}: {job_id} trained an adapter; a full run starts \
                 only from a previous full run's weights."
            )));
        }
        let stem = file_stem(&previous.output.file_name).ok_or_else(|| {
            ApiError::bad_request(format!(
                "advanced.{LOAD_FROM_JOB_KEY}: {job_id} has no valid output name."
            ))
        })?;
        let export = output_dir.join(&stem);
        let source = if export.join(BACKBONE_WEIGHTS_FILE).is_file() {
            export
        } else {
            newest_complete_checkpoint(&sceneworks_worker::iris_checkpoint_root(
                &output_dir,
                &previous.output.file_name,
            ))
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "advanced.{LOAD_FROM_JOB_KEY}: {job_id} left neither an export nor a \
                         complete checkpoint to start from."
                ))
            })?
        };
        options.insert(
            "load_from".to_owned(),
            Value::String(source.display().to_string()),
        );
    }
    plan.config
        .advanced
        .insert(options_key.to_owned(), Value::Object(options));
    Ok(())
}

/// Restoration's synthetic-degradation overrides (`iris_restore.degradation`), through the engine's
/// own parser against the run's crop — refused at submit, not after the weights load. Aligned-pair
/// datasets carry only `use_hflip` (the core dataset contract refuses the rest).
pub(crate) fn check_restore_degradation_overrides(plan: &TrainingPlan) -> Result<(), ApiError> {
    let Some(degradation) = plan
        .config
        .advanced
        .get(sceneworks_core::training::iris::restore::RESTORE_OPTIONS_KEY)
        .and_then(|options| options.get("degradation"))
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    sceneworks_worker::check_restore_degradation(degradation, plan.config.resolution).map_err(
        |error| ApiError::bad_request(format!("advanced.iris_restore.degradation: {error}")),
    )
}

/// The exported full model's directory inside a full run's output dir (`<output_dir>/<stem>`).
pub(crate) fn iris_full_export_dir(output_dir: &FsPath, file_name: &str) -> Option<PathBuf> {
    file_stem(file_name).map(|stem| output_dir.join(stem))
}

/// Shape the submit-time (descriptive) manifest entry for an Iris run: an adapter carries its task
/// identity; a full run (every depth / restoration run) describes the exported model directory
/// instead of Mage's transformer tree.
pub(crate) fn shape_iris_manifest_entry(
    entry: &mut JsonObject,
    output_dir: &FsPath,
    file_name: &str,
    is_full: bool,
    base_model: &str,
) {
    let objective = IrisObjective::from_base_model(base_model).unwrap_or(IrisObjective::Generation);
    entry.insert(
        "irisTask".to_owned(),
        Value::String(objective.task().to_owned()),
    );
    if !is_full {
        return;
    }
    entry.remove("importSourceShape");
    if objective != IrisObjective::Generation {
        // A task model belongs to its task surface (depth / restoration), not the image pickers.
        entry.remove("triggerWords");
        entry.insert("type".to_owned(), Value::String("utility".to_owned()));
    }
    if let (Some(export), Some(stem)) = (
        iris_full_export_dir(output_dir, file_name),
        file_stem(file_name),
    ) {
        entry.insert(
            "paths".to_owned(),
            json!({ "model": export.display().to_string() }),
        );
        entry.insert(
            "files".to_owned(),
            json!(export_files(objective)
                .iter()
                .map(|file| format!("{stem}/{file}"))
                .collect::<Vec<_>>()),
        );
    }
    entry.insert(
        TRAINED_CHECKPOINT_KEY.to_owned(),
        trained_checkpoint_block(objective),
    );
}

fn trained_checkpoint_block(objective: IrisObjective) -> Value {
    json!({
        "engineModel": objective.base_model(),
        "family": IRIS_FAMILY,
        "task": objective.task(),
        "baseModel": objective.base_model(),
        "layout": match objective {
            IrisObjective::Generation => "iris_backbone_directory",
            IrisObjective::Depth => "iris_depth_export",
            IrisObjective::Restoration => "iris_upscaler_export",
        }
    })
}

/// The objective of ANY trained Iris model entry (generation, depth or restoration): the engine whose
/// surface it carries and the task pickers that list it. Builtin entries never qualify.
pub(crate) fn trained_task_checkpoint(entry: &JsonObject) -> Option<IrisObjective> {
    if entry.get("catalogScope").and_then(Value::as_str) == Some("builtin") {
        return None;
    }
    let block = entry.get(TRAINED_CHECKPOINT_KEY)?.as_object()?;
    let objective = IrisObjective::from_base_model(block.get("engineModel")?.as_str()?)?;
    (block.get("task").and_then(Value::as_str) == Some(objective.task())).then_some(objective)
}

/// Whether a user model entry is a trained checkpoint of a builtin generation engine — its engine
/// model id. Builtin entries are never aliased.
pub(crate) fn trained_checkpoint_engine_model(entry: &JsonObject) -> Option<&str> {
    if entry.get("catalogScope").and_then(Value::as_str) == Some("builtin") {
        return None;
    }
    let block = entry.get(TRAINED_CHECKPOINT_KEY)?.as_object()?;
    let engine = block.get("engineModel")?.as_str()?;
    (engine == IRIS_BASE_MODEL
        && block.get("task").and_then(Value::as_str) == Some(IRIS_GENERATION_TASK))
    .then_some(engine)
}

/// The generation surface a trained checkpoint renders with — what its ENGINE declares. Never
/// stored on the user entry: [`overlay_trained_checkpoint_surface`] reads it from the engine's
/// catalog entry every time the catalog is read, so a registered export follows later changes to
/// the engine entry (new adapter families, limits, capabilities) instead of a registration-time
/// snapshot.
const ENGINE_SURFACE_KEYS: [&str; 11] = [
    "type",
    "adapter",
    "capabilities",
    "image",
    "defaults",
    "limits",
    "loraCompatibility",
    "macOnly",
    "mlx",
    "candle",
    // The depth task's job-admission declaration (sc-25682): a trained depth model is the same
    // architecture, so it carries its engine's declaration.
    "depth",
];

/// Overlay a trained checkpoint's engine surface ([`ENGINE_SURFACE_KEYS`]) from its engine's
/// builtin entry — found in `builtin` (the configured manifest), else the compiled-in manifest.
/// A no-op for any entry that is not a trained checkpoint (generation, depth or restoration).
pub(crate) fn overlay_trained_checkpoint_surface(
    entry: &mut Value,
    builtin: &[Value],
) -> Result<(), ApiError> {
    let Some(object) = entry.as_object_mut() else {
        return Ok(());
    };
    let Some(engine) = trained_task_checkpoint(object).map(IrisObjective::base_model) else {
        return Ok(());
    };
    let is_engine = |candidate: &Value| candidate.get("id").and_then(Value::as_str) == Some(engine);
    let engine_entry = match builtin.iter().find(|candidate| is_engine(candidate)) {
        Some(found) => Some(found.clone()),
        None => crate::models::embedded_builtin_catalog_entry(is_engine)?,
    };
    let Some(engine_entry) = engine_entry.as_ref().and_then(Value::as_object) else {
        return Ok(());
    };
    for key in ENGINE_SURFACE_KEYS {
        match engine_entry.get(key) {
            Some(value) => {
                object.insert(key.to_owned(), value.clone());
            }
            None => {
                object.remove(key);
            }
        }
    }
    Ok(())
}

/// Register a completed (or exported-on-cancel) Iris FULL run's export as a user model carrying its
/// task identity: a generation backbone (an image model), or a depth / restoration task export (a
/// utility the depth / restoration job paths select). Same trusted-input discipline as the Mage
/// registrar: the directory is recomputed from the validated id + the plan's validated file stem,
/// and its shape is checked on disk.
pub(crate) fn iris_full_checkpoint_entry(
    job: &JobSnapshot,
    manifest_entry: JsonObject,
    output_dir: &FsPath,
    model_id: &str,
) -> Result<JsonObject, ApiError> {
    let plan = job.payload.get("plan");
    let file_name = plan
        .and_then(|plan| plan.get("output"))
        .and_then(|output| output.get("fileName"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let objective = plan
        .and_then(|plan| plan.get("target"))
        .and_then(|target| target.get("baseModel"))
        .and_then(Value::as_str)
        .and_then(IrisObjective::from_base_model)
        .unwrap_or(IrisObjective::Generation);
    let export = iris_full_export_dir(output_dir, file_name).ok_or_else(|| {
        ApiError::internal("The Iris full run has no valid output name; skipping registration")
    })?;
    let files = export_files(objective);
    if let Some(missing) = files.iter().find(|file| !export.join(file).is_file()) {
        return Err(ApiError::internal(format!(
            "No complete Iris {} export under {} (missing {missing}); skipping model registration",
            objective.task(),
            export.display()
        )));
    }
    let mut entry = manifest_entry;
    // The engine surface is overlaid at catalog read time ([`overlay_trained_checkpoint_surface`]);
    // only the identity is stored here.
    for key in ENGINE_SURFACE_KEYS {
        entry.remove(key);
    }
    let name = entry
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(model_id)
        .to_owned();
    let (entry_type, description, recommended) = match objective {
        IrisObjective::Generation => (
            "image",
            format!(
                "Iris 3B full fine-tune trained in SceneWorks (job {}). Renders with the Iris 3B \
                 generation engine and the installed Qwen3-VL text encoder.",
                job.id
            ),
            "text_to_image",
        ),
        IrisObjective::Depth => (
            "utility",
            format!(
                "Iris 3B depth model trained in SceneWorks (job {}). Pick it in the Image Editor or \
                 Library depth action; it needs no text encoder.",
                job.id
            ),
            "depth",
        ),
        IrisObjective::Restoration => (
            "utility",
            format!(
                "Iris 3B restoration model trained in SceneWorks (job {}). Pick it wherever Iris \
                 restores (Image Studio post-processing, Image Editor, Library).",
                job.id
            ),
            "image_upscale",
        ),
    };
    entry.insert("type".to_owned(), Value::String(entry_type.to_owned()));
    entry.insert(
        "ui".to_owned(),
        json!({
            "label": name,
            "description": description,
            "recommendedFor": [recommended]
        }),
    );
    entry.insert("id".to_owned(), Value::String(model_id.to_owned()));
    entry.insert("family".to_owned(), Value::String(IRIS_FAMILY.to_owned()));
    entry.insert(
        "baseModel".to_owned(),
        Value::String(objective.base_model().to_owned()),
    );
    entry.insert(
        "irisTask".to_owned(),
        Value::String(objective.task().to_owned()),
    );
    entry.insert(
        TRAINED_CHECKPOINT_KEY.to_owned(),
        trained_checkpoint_block(objective),
    );
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("models/finetunes/{model_id}") }),
    );
    entry.insert("files".to_owned(), json!(files));
    entry.insert(
        "paths".to_owned(),
        json!({ "model": export.display().to_string() }),
    );
    entry.remove("importSourceShape");
    entry.remove("downloads");
    entry.remove("triggerWords");
    entry.insert("scope".to_owned(), Value::String("global".to_owned()));
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
    Ok(entry)
}

/// Refuse a depth / restoration job naming a model that is not a registered trained model of that
/// task (the worker re-checks the export's own task identity before it loads).
pub(crate) async fn validate_trained_task_model(
    state: &AppState,
    model_id: &str,
    objective: IrisObjective,
) -> Result<(), ApiError> {
    let entry = crate::models::resolve_model_manifest_entry(state, model_id)
        .await
        .map_err(|_| {
            ApiError::bad_request(format!("irisModel: no model {model_id} in the catalog"))
        })?;
    match entry.as_object().and_then(trained_task_checkpoint) {
        Some(found) if found == objective => Ok(()),
        _ => Err(ApiError::bad_request(format!(
            "irisModel: {model_id} is not a trained Iris {} model",
            objective.task()
        ))),
    }
}

/// The export directory a registered Iris checkpoint loads from, confined under the app's
/// `models/finetunes` tree.
fn confined_checkpoint_dir(entry: &JsonObject, data_dir: &FsPath) -> Result<PathBuf, ApiError> {
    let raw = entry
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| ApiError::bad_request("The trained checkpoint has no model directory."))?;
    let path = std::fs::canonicalize(raw).map_err(|_| {
        ApiError::bad_request(format!(
            "The trained checkpoint's model directory {raw} is missing."
        ))
    })?;
    let root = std::fs::canonicalize(data_dir.join("models").join("finetunes")).map_err(|_| {
        ApiError::bad_request("The trained checkpoint is not inside the app's model store.")
    })?;
    if !path.starts_with(&root)
        || !path.join(BACKBONE_CONFIG_FILE).is_file()
        || !path.join(BACKBONE_WEIGHTS_FILE).is_file()
    {
        return Err(ApiError::bad_request(
            "The trained checkpoint's model directory is not a complete Iris export inside the \
             app's model store.",
        ));
    }
    Ok(path)
}

/// Alias a generation request for a trained Iris checkpoint onto the `iris_3b` engine: the payload
/// names `iris_3b` (routing, admission and the worker lane are the builtin ones), the manifest entry
/// is the generation install's (so its text-encoder co-requisite is staged) plus `modelPath` = the
/// checkpoint's export, and `checkpointModelId` records what the user picked. `entry` is the
/// resolved entry of `requested`; returns the replacement entry, or `None` when it is not a trained
/// checkpoint.
pub(crate) async fn alias_trained_checkpoint(
    state: &AppState,
    requested: &str,
    entry: &Value,
    payload: &mut JsonObject,
) -> Result<Option<Value>, ApiError> {
    let Some(object) = entry.as_object() else {
        return Ok(None);
    };
    let Some(engine_model) = trained_checkpoint_engine_model(object) else {
        return Ok(None);
    };
    let checkpoint_dir = confined_checkpoint_dir(object, &state.settings.data_dir)?;
    let mut engine_entry = crate::jobs::project_image_manifest_for_worker(
        crate::models::resolve_model_manifest_entry(state, engine_model).await?,
    );
    let Some(engine_object) = engine_entry.as_object_mut() else {
        return Err(ApiError::internal(format!(
            "the {engine_model} catalog entry is missing"
        )));
    };
    engine_object.insert(
        "modelPath".to_owned(),
        Value::String(checkpoint_dir.display().to_string()),
    );
    let mut block = object
        .get(TRAINED_CHECKPOINT_KEY)
        .cloned()
        .unwrap_or_else(|| trained_checkpoint_block(IrisObjective::Generation));
    if let Some(block) = block.as_object_mut() {
        block.insert("modelId".to_owned(), Value::String(requested.to_owned()));
    }
    engine_object.insert(TRAINED_CHECKPOINT_KEY.to_owned(), block);
    payload.insert("model".to_owned(), Value::String(engine_model.to_owned()));
    payload.insert(
        CHECKPOINT_MODEL_ID_KEY.to_owned(),
        Value::String(requested.to_owned()),
    );
    Ok(Some(engine_entry))
}
