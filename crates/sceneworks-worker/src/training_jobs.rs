//! Native MLX LoRA/LoKr training jobs (epic 3039) — the training analog of
//! [`image_jobs`](crate::image_jobs)/[`video_jobs`](crate::video_jobs).
//!
//! Parses a `lora_train` job into the Rust-resolved [`TrainingPlan`], then either
//! validates it (dry run) or maps it onto a [`gen_core::TrainingRequest`] and drives
//! `crate::inference_runtime::load_trainer(id, &LoadSpec).train(req, on_progress)` — exactly as the
//! image path maps `ImageRequest` → `GenerationRequest` and calls `Generator::generate`.
//! The engine writes the adapter to the plan's `output.outputDir`; the API registers
//! it from the staged `manifestEntry` + the files on disk (apps/rust-api jobs.rs
//! `register_trained_lora`), so the streamed `result` here is informational/UI only.
//!
//! Routing (sc-3049, sc-7817): the API sends native-trainable families
//! (`z_image_lora`/`sdxl_lora`/`kolors_lora`/`lens_lora`/`krea_lora`/`sd3_lora`/`wan_lora`/
//! `wan_moe_lora`/`ltx_mlx_lora`)
//! here (`jobs_store::training_job_is_mlx_eligible` on Mac, `…_is_candle_eligible` off-Mac).
//! `kolors_lora` joined the native trainers in sc-4732 (engine trainer sc-4568); `lens_lora` in
//! sc-5180; `krea_lora` in sc-7577/7578; `sd3_lora` (Large + MMDiT-X Medium training bases) in
//! sc-7884 (engine trainer sc-7883/7885), native-MLX/Apple-Silicon only. The dry-run validator is
//! cross-platform. The real run executes in-process on the macOS
//! MLX engine OR — the off-Mac cutover (sc-7817, epic 5164) — on the candle Windows/CUDA + Linux
//! engine, for the five families with a candle trainer (`sdxl`/`z_image_turbo`/`lens`/the Krea 2 Raw
//! 12B DiT `krea_2_raw` (sc-8614)/the Wan A14B **T2V** `wan2_2_t2v_14b`). The Python torch trainer
//! has no Krea path at all (Krea is Rust-only — mlx or candle); it stays the fallback for everything
//! else off-Mac with no candle trainer (Kolors, LTX, the dense Wan 5B, the Wan I2V A14B) and the
//! cross-platform default until candle is the default off-Mac worker.

use super::*;
use sceneworks_core::training::{TrainingPlan, TRAINING_PLAN_VERSION};

// epic 3720 (sc-3724): the backend-neutral training contract types come from `gen_core`, which is
// tensor-free and links on every target. The import is gated to the backends that actually run a
// trainer — macOS (mlx-gen) OR the candle Windows/CUDA + Linux/NVIDIA lane (sc-7817) — only so the
// non-training default build doesn't carry an unused-import warning.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    CancelFlag, LoadSpec, LrSchedule, NetworkType, Precision, TrainingConfig, TrainingItem,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};

/// Run a `lora_train` job: parse the resolved plan, then either validate it
/// (dry run, the default) or execute real training. The dry-run/execute split was
/// ported from the retired Python worker's `run_lora_train_job`.
pub(crate) async fn run_lora_train_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let plan = parse_plan(&job.payload)?;
    let dry_run = job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if dry_run {
        run_training_dry_run(api, settings, job, &plan).await
    } else {
        run_training_execution(api, settings, job, &plan).await
    }
}

/// Deserialize the Rust-resolved plan stamped into the job payload at submit time
/// (apps/rust-api training.rs). The plan round-trips through `TrainingPlan`, so a
/// payload missing/garbling it is a hard error (never a silent no-op).
pub(crate) fn parse_plan(payload: &JsonObject) -> WorkerResult<TrainingPlan> {
    let plan = payload.get("plan").ok_or_else(|| {
        WorkerError::InvalidPayload("Training job payload is missing a resolved plan.".to_owned())
    })?;
    serde_json::from_value(plan.clone())
        .map_err(|error| WorkerError::InvalidPayload(format!("Invalid training plan: {error}")))
}

/// Validate a resolved plan the way the Python `validate_training_plan` does:
/// reject an unknown plan version, an empty dataset, or missing dataset images.
/// Shared by the dry-run validator and the real run so both reject the same inputs.
fn validate_training_plan(settings: &Settings, plan: &TrainingPlan) -> WorkerResult<()> {
    if plan.plan_version != TRAINING_PLAN_VERSION {
        return Err(WorkerError::InvalidPayload(format!(
            "Unsupported training plan version {}; this worker understands version {}.",
            plan.plan_version, TRAINING_PLAN_VERSION
        )));
    }
    if plan.dataset.items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Training plan dataset has no items to train on.".to_owned(),
        ));
    }
    normalize_app_managed_model_path(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )?;
    resolve_training_output_dir(settings, &plan.output.output_dir, "Training outputDir")?;
    // sc-8878 (F-076): the client-supplied `output.file_name` is joined under the confined
    // `output_dir` by the engine trainer (`TrainingRequest.file_name`), but only the preview-sample
    // stem was ever sanitized — a `../…`/absolute `file_name` would escape the confined output dir
    // and write the adapter anywhere the worker can. Confine it to a single plain path component
    // (no separators, no `..`, not absolute) with the same primitive the strict-control lanes use
    // for payload-supplied weight filenames (`safe_weight_filename`). Shared by the dry run and the
    // real run (both call this), so both reject the same forged filename before any join.
    safe_weight_filename(&plan.output.file_name, "Training fileName")?;
    let mut missing = Vec::new();
    for item in &plan.dataset.items {
        let image_path = resolve_dataset_item_path(
            settings,
            &plan.dataset.root_path,
            &item.image_path,
            "Training dataset imagePath",
        )?;
        if !image_path.exists() {
            missing.push(image_path.display().to_string());
        }
    }
    if !missing.is_empty() {
        let preview = missing
            .iter()
            .take(3)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WorkerError::InvalidPayload(format!(
            "{} dataset image(s) are missing on the worker, e.g. {preview}.",
            missing.len()
        )));
    }
    Ok(())
}

/// Dry-run: validate the plan and report what a real run would produce, with no
/// model load or training (so a GPU worker without the engine still validates).
/// Cross-platform — the validator and summary touch only the plan + the dataset
/// images on disk. Mirrors the Python `_run_lora_train_dry_run`.
async fn run_training_dry_run(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
) -> WorkerResult<()> {
    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Validating training plan.",
            None,
            backend,
        ),
    )
    .await?;
    validate_training_plan(settings, plan)?;
    let item_count = plan.dataset.items.len();
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            &format!("Checked {item_count} dataset item(s)."),
            None,
            backend,
        ),
    )
    .await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Dry run validated {item_count} dataset item(s); training plan is ready."),
            Some(dry_run_summary(settings, plan)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// The dry-run completion summary (keys mirror the Python `dry_run_training_summary`
/// so the Training Studio reads an identical shape regardless of which worker runs it).
fn dry_run_summary(settings: &Settings, plan: &TrainingPlan) -> JsonObject {
    let base_model_installed = normalize_app_managed_model_path(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )
    .is_ok_and(|path| path.exists());
    let mut summary = JsonObject::new();
    summary.insert("mode".to_owned(), json!("dry_run"));
    summary.insert("validated".to_owned(), json!(true));
    summary.insert("dryRun".to_owned(), json!(true));
    summary.insert(
        "datasetItemCount".to_owned(),
        json!(plan.dataset.items.len()),
    );
    summary.insert("datasetId".to_owned(), json!(plan.dataset.dataset_id));
    summary.insert(
        "datasetVersion".to_owned(),
        json!(plan.dataset.dataset_version),
    );
    summary.insert("targetId".to_owned(), json!(plan.target.target_id));
    summary.insert("kernel".to_owned(), json!(plan.target.kernel));
    summary.insert("loraId".to_owned(), json!(plan.output.lora_id));
    summary.insert("outputDir".to_owned(), json!(plan.output.output_dir));
    summary.insert("fileName".to_owned(), json!(plan.output.file_name));
    summary.insert("baseModel".to_owned(), json!(plan.target.base_model));
    summary.insert(
        "baseModelRepo".to_owned(),
        json!(plan.target.base_model_repo),
    );
    summary.insert(
        "baseModelPath".to_owned(),
        json!(plan.target.base_model_path),
    );
    summary.insert("baseModelInstalled".to_owned(), json!(base_model_installed));
    summary.insert("planVersion".to_owned(), json!(plan.plan_version));
    summary
}

/// A `lora_train` progress update with the worker's backend label (mirrors
/// `image_jobs::image_progress`). LoRA training keeps `status: running` across the
/// caching/training/checkpointing/saving stages; only the final update is `completed`.
pub(crate) fn training_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

// --------------------------------------------------------------------------- #
// Real training — the in-process native engine: macOS / Apple-Silicon (mlx-gen) OR the candle
// Windows/CUDA + Linux/NVIDIA lane (sc-7817). The whole section is backend-neutral (it drives the
// `gen_core::Trainer` contract via `load_trainer`), so it compiles on whichever backend is linked;
// the registry resolves the trainer by id. The capability is advertised only when a backend with a
// registered trainer is enabled, so on a build with neither this section is cfg'd out and the stub
// at the bottom fails loudly.
// --------------------------------------------------------------------------- #

/// Map a resolved plan's `(kernel, baseModel)` onto the trainer registry id (the trainer id matches
/// the generator id of the same base model). Backend-neutral: the registry resolves it to the mlx
/// trainer on macOS or the candle trainer off-Mac. Wan splits by the base model variant: the dense
/// TI2V-5B (`wan_lora`) vs the two A14B MoE variants (`wan_moe_lora` + the T2V/I2V base model).
/// `None` for a family with no native trainer at all — those never route here, but the mapping fails
/// loudly if one does. NB the candle registry only holds a subset of these ids (`sdxl`,
/// `z_image_turbo`, `lens`, `wan2_2_t2v_14b`); the API's `training_job_is_candle_eligible` gate keeps
/// the candle-untrained ids (Kolors/LTX/Wan-5B/Wan-I2V) off the candle worker, and `load_trainer`
/// fails loudly as the backstop if one ever slips through.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn engine_trainer_id(plan: &TrainingPlan) -> Option<&'static str> {
    match plan.target.kernel.as_str() {
        "z_image_lora" => Some("z_image_turbo"),
        "sdxl_lora" => Some("sdxl"),
        // Kolors is an SDXL U-Net under a ChatGLM3-6B encoder; the engine registers its
        // LoRA/LoKr trainer under the same id as its generator (`"kolors"`), sc-4568.
        "kolors_lora" => Some("kolors"),
        // Lens trains the base (non-distilled) Lens DiT — the `SceneWorks/Lens` diffusers rehost
        // since sc-8797 (microsoft/Lens is dead); the engine registers its LoRA/LoKr trainer
        // under the base generator id `"lens"` (arch-identical to lens_turbo), sc-5148.
        "lens_lora" => Some("lens"),
        // Krea trains the undistilled `krea/Krea-2-Raw` DiT; the engine registers its LoRA/LoKr
        // trainer under the base id `"krea_2_raw"` (arch-identical to krea_2_turbo), sc-7577/7578.
        "krea_lora" => Some("krea_2_raw"),
        // Krea pose-ControlNet: trains a control branch on the frozen Krea base; the engine registers
        // its control trainer under `"krea_2_control"` (candle-gen-krea, sc-10163 / epic 10159 B2).
        "krea_control" => Some("krea_2_control"),
        // SD3.5 (epic 7841, T3 sc-7884): the engine registers its LoRA/LoKr trainer under the same
        // id as the inference generator of the training base — `sd3_5_large` (T2 sc-7883) and the
        // MMDiT-X `sd3_5_medium` (T4 sc-7885). The trained adapter records `family: sd3` and applies
        // back at that base (Large also covers the family-arch-identical `sd3_5_large_turbo`) via the
        // `apply_sd3_adapters` seam — no base-model gating (family-match, like Krea Raw→Turbo).
        "sd3_lora" => match plan.target.base_model.as_str() {
            "sd3_5_large" => Some("sd3_5_large"),
            "sd3_5_medium" => Some("sd3_5_medium"),
            _ => None,
        },
        "ltx_mlx_lora" => Some("ltx_2_3"),
        // Dense Wan2.2-TI2V-5B.
        "wan_lora" => Some("wan2_2_ti2v_5b"),
        // A14B dual-expert MoE; the T2V/I2V base model picks the trainer.
        "wan_moe_lora" => match plan.target.base_model.as_str() {
            "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
            "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
            _ => None,
        },
        // Anima (epic 10512, sc-10522): the `mlx-gen-anima` LoRA/LoKr trainer registers under the same
        // id as the inference generator of the training base. All three variants share one architecture,
        // so a LoRA trained on any applies back to every variant via `apply_anima_adapters`; the base
        // model selects which variant's dense weights the trainer loads (default `anima_base`).
        "anima_lora" => match plan.target.base_model.as_str() {
            "anima_aesthetic" => Some("anima_aesthetic"),
            "anima_turbo" => Some("anima_turbo"),
            _ => Some("anima_base"),
        },
        _ => None,
    }
}

/// Read an `advanced` field as a string, trimmed and non-empty, else `default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_str(advanced: &JsonObject, key: &str, default: &str) -> String {
    advanced
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

/// Read an `advanced` field as an f32, else `default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_f32(advanced: &JsonObject, key: &str, default: f32) -> f32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

/// Read an `advanced` field as a u32, else `default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_u32(advanced: &JsonObject, key: &str, default: u32) -> u32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as u32)
        .unwrap_or(default)
}

/// Read an `advanced` field as a bool (accepting a JSON bool or a `"true"`/`"false"`
/// string), else `default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_bool(advanced: &JsonObject, key: &str, default: bool) -> bool {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_bool()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(default)
}

/// Normalize the advanced `mixedPrecision` string onto the engine's `train_dtype`
/// domain, which is exactly `{"bf16", "f32"}` (sc-4887). Only an explicit `"bf16"`
/// (case-insensitive) selects bf16; every other value — `"fp16"`, `"no"`, empty,
/// anything unrecognized — falls back to full-precision `"f32"`, matching the
/// engine's own "unrecognized ⇒ f32" rule. The *absent*-key default is applied by
/// the caller (`"bf16"`), so a plan that omits the key keeps the OOM fix on.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn normalize_train_dtype(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "bf16" => "bf16".to_owned(),
        _ => "f32".to_owned(),
    }
}

/// Map the SceneWorks `TrainingConfig` (plan `config` + its free-form `advanced`
/// bag) onto the engine's typed [`gen_core::TrainingConfig`]. The optimizer string is
/// passed verbatim — the engine normalizes aliases (`adamw8bit`→`adamw`,
/// `prodigyopt`→`prodigy`). An empty `loraTargetModules` lets the family trainer use
/// its default target set.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn map_training_config(config: &sceneworks_core::training::TrainingConfig) -> TrainingConfig {
    let advanced = &config.advanced;
    let lora_target_modules = advanced
        .get("loraTargetModules")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    TrainingConfig {
        rank: config.rank,
        alpha: config.alpha as f32,
        learning_rate: config.learning_rate.as_f64().unwrap_or(1e-4) as f32,
        steps: config.steps,
        batch_size: config.batch_size,
        gradient_accumulation: config.gradient_accumulation,
        resolution: config.resolution,
        save_every: config.save_every,
        seed: config.seed.max(0) as u64,
        optimizer: config.optimizer.clone(),
        weight_decay: advanced_f32(advanced, "weightDecay", 0.0),
        lr_scheduler: LrSchedule::parse(&advanced_str(advanced, "lrScheduler", "constant")),
        lr_warmup_steps: advanced_u32(advanced, "lrWarmupSteps", 0),
        network_type: NetworkType::parse(&advanced_str(advanced, "networkType", "lora")),
        decompose_factor: advanced
            .get("decomposeFactor")
            .and_then(Value::as_i64)
            .map(|value| value as i32)
            .unwrap_or(-1),
        lora_target_modules,
        timestep_type: advanced_str(advanced, "timestepType", "sigmoid"),
        timestep_bias: advanced_str(advanced, "timestepBias", "balanced"),
        loss_type: advanced_str(advanced, "lossType", "mse"),
        // Training compute dtype — the primary OOM fix (sc-4887). bf16 halves the
        // activation working set and drops the 1024² z-image first-step peak 135 → ~44 GB,
        // so the run survives. This mapping builds the engine config field-by-field (no
        // `..Default::default()`), so the engine's "bf16" default never reaches here — it
        // MUST be set explicitly. Sourced from the advanced `mixedPrecision` key (presets
        // already carry "bf16"); the engine only supports bf16/f32, so anything else
        // (incl. "fp16", "no", empty) normalizes to "f32". Absent → "bf16" (keep the fix on).
        train_dtype: normalize_train_dtype(&advanced_str(advanced, "mixedPrecision", "bf16")),
        // Honor the "Gradient Checkpointing" UI checkbox on the Rust path (sc-4881) — an
        // extra lever for smaller machines / higher resolution on top of the bf16 fix.
        // Previously dropped here, so the engine always ran at its `false` default. Absent
        // (legacy payloads) preserves that default.
        gradient_checkpointing: advanced_bool(advanced, "gradientCheckpointing", false),
        trigger_word: config.trigger_word.clone(),
        // sc-5637 — preview-sample cadence. The SceneWorks config + UI always supply these (presets
        // set `sampleEvery`/`sampleSteps`/`sampleGuidanceScale`; the submit derives `samplePrompts`
        // from the trigger word), but the engine config is built field-by-field (no `..Default`), so
        // they must be mapped explicitly or the family trainer never samples. Absent `sampleEvery`
        // (legacy payloads) → 0 → sampling stays off, exactly as before this fix.
        sample_every: advanced_u32(advanced, "sampleEvery", 0),
        sample_steps: advanced_u32(advanced, "sampleSteps", 20),
        sample_guidance_scale: advanced_f32(advanced, "sampleGuidanceScale", 1.0),
        // Mid-schedule resume (gen-core sc-9560 / F-125): not yet surfaced on the SceneWorks
        // training path — default `false` preserves the current from-scratch behavior. Wiring the
        // resume toggle from the plan's `advanced` bag is a separate training-feature story.
        resume: false,
        // ControlNet control type (sc-10163) — set by a control-branch target's `advanced.controlType`
        // (e.g. "pose"); absent for LoRA/LoKr targets ⇒ None. Drives the control trainer's overlay
        // `kind` metadata and is required by its validate; ignored by LoRA trainers.
        control_type: advanced
            .get("controlType")
            .and_then(Value::as_str)
            .map(str::to_owned),
        // sc-8671 — the engine renders one preview per prompt, capped at `sampleCount` (default 4 =
        // the historical fixed `[:4]` cap). The UI always supplies `samplePrompts` (prefilled from the
        // trigger phrase); an absent/empty pool ⇒ no previews, exactly as before (sampling is gated by
        // `sampleEvery` anyway). The pool is truncated, never padded — to get more previews, add prompts.
        sample_prompts: {
            let pool = advanced
                .get("samplePrompts")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            resolve_sample_prompts(
                pool,
                advanced_u32(advanced, "sampleCount", DEFAULT_SAMPLE_COUNT),
            )
        },
    }
}

/// Default number of preview images rendered per sample step when the plan doesn't set
/// `advanced.sampleCount`. Matches the four default prompts the UI prefills, so legacy payloads
/// preview exactly as before this knob existed (sc-8671).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const DEFAULT_SAMPLE_COUNT: u32 = 4;

/// Cap a prompt pool at `count` (one preview per prompt, truncated — never padded). An empty pool or
/// `count == 0` yields no samples (sampling stays off). The UI owns the default prompts (sc-8671).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_sample_prompts(pool: Vec<String>, count: u32) -> Vec<String> {
    pool.into_iter().take(count as usize).collect()
}

/// Whether the candle (Windows/CUDA + Linux/NVIDIA) backend MUST run this family with gradient
/// checkpointing, because a dense backward over its base DiT OOMs. UNLIKE MLX/torch, candle's matmul
/// backward materializes a gradient for the FROZEN base weight too, so a dense backward over a
/// multi-billion-parameter DiT OOMs even alone on a 96 GB card (epic 5164: the candle Z-Image trainer
/// got checkpointing in sc-5246, and the Wan A14B trainer needs it too). Scoped to exactly the
/// candle-trainable big-DiT families: Z-Image and the Wan A14B **T2V** MoE — the same set
/// `jobs_store::training_job_is_candle_eligible` gates the MoE to (only `wan_2_2_t2v_14b` has a candle
/// trainer; the I2V A14B / dense 5B stay on torch). Krea 2 Raw is a 12B DiT (epic 7565 P4, sc-8614) —
/// the same dense-backward OOM class, so it is forced too (the sc-7900 big-DiT backstop). SDXL's
/// smaller U-Net fits a dense backward (the `candle_sdxl_real_weights` smoke trains it with
/// checkpointing off) and Lens is small, so neither is forced — both honor the plan's value.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_requires_gradient_checkpointing(plan: &TrainingPlan) -> bool {
    match plan.target.kernel.as_str() {
        "z_image_lora" => true,
        // Krea 2 Raw is a 12B DiT — a dense backward materializes ~another model of frozen-weight
        // grads on candle and OOMs, so force checkpointing on (sc-8614 / sc-7900 backstop).
        "krea_lora" => true,
        // Krea pose-ControlNet trains a control branch on the same frozen 12B DiT — same OOM risk,
        // so force checkpointing on (sc-10163).
        "krea_control" => true,
        "wan_moe_lora" => plan.target.base_model == "wan_2_2_t2v_14b",
        _ => false,
    }
}

/// Apply backend-specific safety overrides to the mapped engine [`TrainingConfig`] before it reaches
/// the trainer. On candle, force gradient checkpointing on for the big-DiT families that can't do a
/// dense backward (see [`candle_requires_gradient_checkpointing`]): every preset already defaults the
/// cross-platform "Gradient Checkpointing" UI box ON, but the user can turn it OFF — harmless on the
/// macOS MLX path (bf16 alone fits) yet a guaranteed OOM on candle for these families — and a thin /
/// legacy submit that omits the key leaves the worker's `map_training_config` default (`false`) in
/// force. Forcing it here makes a real off-Mac Z-Image / Krea Raw / Wan A14B-T2V run impossible to configure
/// into a dense-backward OOM. On macOS this is the identity (the MLX path tolerates a dense bf16
/// backward), so the mapped config passes through unchanged.
#[cfg(target_os = "macos")]
fn finalize_training_config(config: TrainingConfig, _plan: &TrainingPlan) -> TrainingConfig {
    config
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn finalize_training_config(mut config: TrainingConfig, plan: &TrainingPlan) -> TrainingConfig {
    if candle_requires_gradient_checkpointing(plan) && !config.gradient_checkpointing {
        tracing::info!(
            event = "candle_training_force_gradient_checkpointing",
            kernel = %plan.target.kernel,
            base_model = %plan.target.base_model,
            "candle big-DiT training family requires gradient checkpointing; overriding the plan's \
             gradientCheckpointing=false to avoid a dense-backward CUDA OOM"
        );
        config.gradient_checkpointing = true;
    }
    config
}

/// One progress event streamed from the blocking training thread to the async side.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
enum TrainEvent {
    Progress(TrainingProgress),
    Done(TrainingOutput),
}

/// The trainer's `LoadSpec::text_encoder` override for `engine_id`, or `None` to keep the engine's
/// own resolution. Only LTX-2.3 needs one: its Gemma-3 TE lives OUTSIDE the weights dir (the turnkey
/// bundle ships it as a sibling `gemma/`, sc-5608), so — mirroring the inference path (sc-8827) —
/// resolve the bundled sibling and thread it on, letting a self-contained install train without a
/// separate `mlx-community/gemma-3-12b-it-bf16` download (sc-9989). `None` for every other family (TE
/// lives inside the weights dir) and for a legacy LTX conversion with no sibling or an operator
/// `$LTX_GEMMA_DIR` (the engine's env/HF-cache fallback stays in force). LTX training is mlx-only, so
/// off-Mac this is always `None`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn training_text_encoder(engine_id: &str, weights_dir: &std::path::Path) -> Option<WeightsSource> {
    #[cfg(target_os = "macos")]
    if engine_id == "ltx_2_3" {
        return crate::video_jobs::resolve_bundled_ltx_gemma_dir(weights_dir)
            .map(WeightsSource::Dir);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (engine_id, weights_dir);
    None
}

/// The shipped `builtin.models.jsonc` entry for `model_id`, parsed from the embedded manifest — the
/// source of truth for a base model's pinned `coRequisite` downloads (the app seeds its live catalog
/// from these bytes). The training plan carries the base model id, not its manifest entry, so the trainer
/// resolves components off this. `None` when the id is not a builtin model.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn builtin_model_manifest_entry(model_id: &str) -> Option<serde_json::Value> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)?;
    let manifest: serde_json::Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).ok()?;
    manifest
        .get("models")?
        .as_array()?
        .iter()
        .find(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(model_id))
        .cloned()
}

/// Resolve a trainer's caller-staged components (epic 13657, sc-13682) from the base model's pinned
/// `coRequisite` downloads, so the [`LoadSpec`] handed to [`crate::inference_runtime::load_trainer`]
/// carries them for the engine's load-time `require_component` gate. Driven by the registered GENERATOR
/// descriptor's `required_components`: SDXL's candle trainer consumes exactly the SAME three ids its
/// generator advertises (the CLIP-L/bigG tokenizers + fp16-fix VAE, inference sc-13663), so the generator
/// descriptor is the correct driver at this pin. Empty for every engine that advertises none — INCLUDING
/// the self-contained macOS MLX SDXL trainer (descriptor `&[]`), so this never reads the manifest on
/// macOS. All-or-nothing: a missing component fails the job with an actionable error BEFORE the trainer
/// load (never a mid-train hub fetch).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_trainer_components(
    engine_id: &str,
    base_model_id: &str,
    settings: &Settings,
) -> WorkerResult<std::collections::BTreeMap<String, WeightsSource>> {
    let Some(descriptor) = crate::inference_runtime::media_descriptor(engine_id) else {
        return Ok(std::collections::BTreeMap::new());
    };
    if descriptor.required_components.is_empty() {
        return Ok(std::collections::BTreeMap::new());
    }
    let manifest_entry = builtin_model_manifest_entry(base_model_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "training base model '{base_model_id}' has no builtin catalog entry to resolve its \
             '{engine_id}' components from"
        ))
    })?;
    crate::model_jobs::resolve_co_requisites(&descriptor, &manifest_entry, settings)
}

/// Execute a real training run on the in-process native engine (mlx-gen on macOS, candle-gen
/// off-Mac via `backend-candle`, sc-7817). Loads the (frozen) base model via a [`LoadSpec`] (exactly
/// as inference's `load_engine`), runs the family trainer on a blocking thread, streams staged
/// progress, honors cancellation via the engine's [`CancelFlag`], and reports the produced adapter.
/// The adapter is written by the engine into the plan's `output.outputDir`; the API registers it
/// from the staged manifest entry. Backend-neutral: `load_trainer(engine_id, …)` resolves whichever
/// backend's trainer is registered under `engine_id`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_training_execution(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
) -> WorkerResult<()> {
    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing LoRA training.",
            None,
            backend,
        ),
    )
    .await?;

    validate_training_plan(settings, plan)?;

    let engine_id = engine_trainer_id(plan).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "No native trainer for kernel '{}' (base model '{}').",
            plan.target.kernel, plan.target.base_model
        ))
    })?;

    let weights_dir = resolve_app_managed_model_dir(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )?;

    // LTX-2.3's Gemma-3 text encoder lives OUTSIDE `weights_dir` — thread the bundled sibling onto the
    // trainer `LoadSpec` so a self-contained install trains without a separate gemma download (sc-9989).
    let ltx_text_encoder = training_text_encoder(engine_id, &weights_dir);

    // Caller-staged model components (epic 13657, sc-13682): the base model's pinned `coRequisite`
    // downloads the trainer's engine consumes — SDXL's CLIP-L/bigG tokenizers + fp16-fix VAE on candle.
    // Resolved BEFORE the blocking thread (a missing one fails the job with an actionable error naming the
    // component id + repo), then moved into the closure and folded onto the trainer `LoadSpec`. Empty for
    // the self-contained macOS MLX SDXL trainer (its descriptor advertises none), so a no-op on macOS.
    let train_components =
        resolve_trainer_components(engine_id, &plan.target.base_model, settings)?;

    let output_dir =
        resolve_training_output_dir(settings, &plan.output.output_dir, "Training outputDir")?;
    tokio::fs::create_dir_all(&output_dir).await?;

    let items: Vec<TrainingItem> = plan
        .dataset
        .items
        .iter()
        .map(|item| {
            Ok(TrainingItem {
                image_path: resolve_dataset_item_path(
                    settings,
                    &plan.dataset.root_path,
                    &item.image_path,
                    "Training dataset imagePath",
                )?,
                caption: item.caption.clone(),
                // ControlNet training: resolve the per-item conditioning image the same way as the
                // target (None for LoRA — the control trainer's validate rejects a missing one).
                control_image_path: item
                    .control_image_path
                    .as_ref()
                    .map(|p| {
                        resolve_dataset_item_path(
                            settings,
                            &plan.dataset.root_path,
                            p,
                            "Training dataset controlImagePath",
                        )
                    })
                    .transpose()?,
            })
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    // Apply backend-specific safety overrides before the config reaches the engine: candle can't run
    // a dense backward over the big-DiT families without a CUDA OOM, so `finalize_training_config`
    // forces gradient checkpointing on for them (Z-Image, Wan A14B-T2V) regardless of the plan value.
    // On macOS this is the identity. See its doc comment for the why.
    let config = finalize_training_config(map_training_config(&plan.config), plan);
    let total_steps = config.steps;
    let file_name = plan.output.file_name.clone();
    let trigger_words = plan.output.trigger_words.clone();

    check_cancel(api, &job.id, "LoRA training canceled before it started.").await?;

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<TrainEvent>(64);

    let blocking = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            // Load precision tracks the requested `train_dtype` (advanced `mixedPrecision`, default
            // bf16). The MLX Lens trainer loads its DiT at this precision and enforces `train_dtype`
            // against it (sc-5148), so f32 training must load at Fp32; the cast-based MLX trainers
            // (z-image/kolors/sdxl/wan/ltx) load dense and ignore `precision`, so this is inert for
            // them — the default bf16 path is byte-identical to before. The candle trainers are lazy
            // (sc-7817): they build the frozen base inside `train()` at the request's `train_dtype`,
            // so `LoadSpec.precision` is likewise inert for them and this mapping stays harmless.
            let load_precision = if config.train_dtype.trim().eq_ignore_ascii_case("f32") {
                Precision::Fp32
            } else {
                Precision::Bf16
            };
            let spec = LoadSpec {
                precision: load_precision,
                // LTX-2.3's bundled Gemma-3 TE (sc-9989); `None` for every other family (TE lives
                // inside `weights_dir`) and for legacy/env-override LTX installs.
                text_encoder: ltx_text_encoder,
                ..LoadSpec::new(WeightsSource::Dir(weights_dir))
            };
            // Fold the caller-staged components (epic 13657, sc-13682) resolved above onto the spec — the
            // trainer's load-time gate requires them (SDXL on candle); an empty map is a no-op otherwise.
            let spec = train_components
                .into_iter()
                .fold(spec, |spec, (id, source)| spec.with_component(id, source));
            let mut trainer =
                crate::inference_runtime::load_trainer(engine_id, &spec).map_err(|error| {
                    WorkerError::Engine(format!("{engine_id} trainer load failed: {error}"))
                })?;
            let request = TrainingRequest {
                items,
                config,
                output_dir,
                file_name,
                trigger_words,
                cancel,
            };
            trainer.validate(&request).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "{engine_id} trainer rejected the plan: {error}"
                ))
            })?;
            // If the consumer loop has returned early (a POST failure / 409 dropped `rx`), the
            // channel is closed. Detect that here and trip the engine cancel flag so the trainer
            // bails at its next cooperative check instead of silently running to completion on a
            // job nobody is listening to (sc-8804, F-003 — the swallowed-closed-channel leak). The
            // `progress_cancel` clone is the same flag threaded into the `TrainingRequest`.
            let progress_cancel = request.cancel.clone();
            let mut on_progress = |progress: TrainingProgress| {
                if tx.blocking_send(TrainEvent::Progress(progress)).is_err() {
                    progress_cancel.cancel();
                }
            };
            let output = trainer
                .train(&request, &mut on_progress)
                .map_err(|error| WorkerError::Engine(format!("training failed: {error}")))?;
            let _ = tx.blocking_send(TrainEvent::Done(output));
            Ok(())
        })
    };

    consume_training_events(
        api,
        settings,
        job,
        plan,
        backend,
        total_steps,
        rx,
        cancel,
        blocking,
    )
    .await
}

/// First-detection handling for the in-loop training cancel poller (sc-5516): trip the engine
/// `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate progress bar —
/// `running` + fraction 0.0 renders the "Working" animation, not a backward jump). The terminal
/// `Canceled` is posted only after the blocking training actually stops (see
/// `consume_training_events`), so the worker row — and therefore the next queued job — is not freed
/// until the GPU is genuinely idle, and the UI honestly shows "Cancelling…" until completion.
/// Best-effort: a failed status update here is non-fatal because the post-run terminal write is
/// what ultimately frees the worker. Mirrors the image path's `begin_image_cancel` (sc-5515).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn begin_training_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        training_progress(
            JobStatus::Running,
            ProgressStage::Training,
            0.0,
            "Cancelling — finishing the current step…",
            None,
            backend,
        ),
    )
    .await;
}

/// The preview-sample plumbing (sc-5637), lifted out of [`consume_training_events`] into one cohesive
/// unit (sc-8921, F-119): the resolved sample config + output/project/stem paths, plus the accumulated
/// cumulative (`all_samples`) and this-cadence (`latest_samples`) record lists the Training Studio
/// renders. The engine streams `TrainingProgress::Sample` events carrying a decoded RGB bitmap; each is
/// persisted as a PNG project asset off the async thread and the updated lists are streamed live. All
/// best-effort — a persistence hiccup logs and is skipped, never failing the run. Owning this state
/// here shrinks the event loop to `persister.persist(...)` and keeps the Done handler's final-result
/// fold reading the same accumulated samples.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct SamplePersister {
    cfg: TrainingConfig,
    output_dir: Option<PathBuf>,
    project_root: Option<PathBuf>,
    stem: String,
    all_samples: Vec<Value>,
    latest_step: u32,
    latest_samples: Vec<Value>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl SamplePersister {
    /// Resolve the sample config + destination paths for a run. `output_dir`/`project_root` are
    /// resolved leniently (already validated upstream in `run_training_execution`); if either is
    /// unavailable, samples simply don't render but training is unaffected.
    fn new(settings: &Settings, job: &JobSnapshot, plan: &TrainingPlan) -> Self {
        let output_dir =
            resolve_training_output_dir(settings, &plan.output.output_dir, "Training outputDir")
                .ok();
        let project_root = job.project_id.as_ref().and_then(|project_id| {
            ProjectStore::new(settings.data_dir.clone(), "worker")
                .get_project(project_id)
                .ok()
                // Canonicalize the project root with the SAME normalization the output
                // path went through (`resolve_training_output_dir` -> `normalize_app_managed_path`
                // -> `normalize_existing_or_absolute`). The project store persists `path`
                // lexically (`data_dir.join(...).display()`), so on macOS it stays `/var/...`
                // while `output_dir` resolves to `/private/var/...`; without this, the
                // `strip_prefix(project_root)` in `write_training_sample` fails and every
                // sample record silently loses its `relativePath` (sc-9812 / F-075 follow-up).
                // The project dir always exists here (`find_project_path` requires it), so
                // this canonicalizes fully; fall back to the lexical path if resolution fails.
                .map(|project| {
                    let lexical = PathBuf::from(project.path);
                    normalize_existing_or_absolute(&lexical).unwrap_or(lexical)
                })
        });
        let stem = Path::new(&plan.output.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_owned();
        Self {
            cfg: map_training_config(&plan.config),
            output_dir,
            project_root,
            stem,
            all_samples: Vec::new(),
            latest_step: 0,
            latest_samples: Vec::new(),
        }
    }

    /// Persist one `TrainingProgress::Sample` preview and stream the updated cumulative + this-cadence
    /// lists. The PNG encode + atomic rename are blocking, so the owned `image` (no buffer clone) +
    /// owned path/stem/prompt move into a `spawn_blocking` (sc-8909 / F-107). A write failure logs and
    /// is skipped — the run continues.
    #[allow(clippy::too_many_arguments)]
    async fn persist(
        &mut self,
        api: &ApiClient,
        job: &JobSnapshot,
        backend: &str,
        total_steps: u32,
        step: u32,
        index: u32,
        total: u32,
        prompt: String,
        image: gen_core::Image,
    ) -> WorkerResult<()> {
        let output_dir = self.output_dir.clone();
        let project_root = self.project_root.clone();
        let stem = self.stem.clone();
        let sample_steps = self.cfg.sample_steps;
        let sample_guidance_scale = self.cfg.sample_guidance_scale;
        let persisted = tokio::task::spawn_blocking(move || {
            write_training_sample(
                output_dir.as_deref(),
                project_root.as_deref(),
                &stem,
                step,
                index,
                &prompt,
                image,
                sample_steps,
                sample_guidance_scale,
            )
        })
        .await
        .map_err(|error| task_join_error("training preview persist task", error))?;
        match persisted {
            Ok(record) => {
                let record = Value::Object(record);
                if step != self.latest_step {
                    self.latest_step = step;
                    self.latest_samples.clear();
                }
                self.all_samples.push(record.clone());
                self.latest_samples.push(record);
                let result = training_samples_result(
                    &self.all_samples,
                    &self.latest_samples,
                    &self.cfg.sample_prompts,
                    self.cfg.sample_steps,
                    self.cfg.sample_guidance_scale,
                );
                update_job(
                    api,
                    &job.id,
                    training_progress(
                        JobStatus::Running,
                        ProgressStage::Training,
                        train_fraction(step, total_steps.max(step)),
                        &format!("Rendered preview {index}/{total} at step {step}."),
                        Some(result),
                        backend,
                    ),
                )
                .await?;
            }
            Err(error) => tracing::warn!(
                event = "training_preview_persist_failed",
                step,
                index,
                error = %error,
                "worker failed to persist training preview — skipping, training continues"
            ),
        }
        Ok(())
    }
}

/// Consume training events from the blocking thread: stream staged progress, poll
/// cancel ~every 2s (draining after a cancel so the blocking sender never blocks),
/// and on the final `Done` event report completion with the result the UI shows.
/// Mirrors `image_jobs::consume_gen_events`.
///
/// sc-9541 (F-034 follow-up): this deliberately does NOT fold into the shared
/// [`analysis_jobs_common::run_batched_analysis_job`] scaffold (which the CLIP + face analysis jobs
/// share). That scaffold abstracts "N homogeneous items, each a bare `usize` index → one record `R`,
/// all ending in ONE sidecar POST"; training shares none of those axes — its channel carries a rich
/// `TrainEvent`/`TrainingProgress` variant tree (not a `usize`), its progress spans five kernel bands
/// via [`map_training_progress`] (not the analysis single-ramp `item_progress`), its `Sample` events
/// have file-persistence side effects (`SamplePersister`), its cancel posts an interim non-terminal
/// "Cancelling…" update (`begin_training_cancel`, the analysis loop trips a bare flag), and it emits an
/// in-place `Done` result with NO sidecar POST (the engine writes the adapter; the API registers it
/// separately). Forcing training through the analysis scaffold would require making it generic over an
/// event type + an event→progress mapper + a stateful side-effect hook + a no-op sidecar branch —
/// distorting the clean analysis-only seam for more complexity than the duplication it removes. The
/// only genuinely shared machinery (the `CancelJoinGuard` + bounded-join teardown + deferred-terminal
/// cancel, sc-8804/8917) is already reconciled across both by convention, not code sharing.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn consume_training_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
    backend: &str,
    total_steps: u32,
    mut rx: tokio::sync::mpsc::Receiver<TrainEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
) -> WorkerResult<()> {
    // Bind the blocking training task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` below returns early on a transient POST failure or a 409 (stale-sweep
    // reclaim); on that early return this guard trips the engine `CancelFlag` and aborts the
    // still-running training thread instead of leaving it burning GPU memory alongside the next
    // claimed job. `cancel` is kept alongside (it's `Clone`) for the in-loop `begin_training_cancel`
    // pollers; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    let mut checkpoints: Vec<Value> = Vec::new();

    // sc-5637 — preview-sample plumbing, now owned by `SamplePersister` (sc-8921): the resolved sample
    // config + destination paths + the accumulated cumulative/this-cadence record lists. The event
    // loop's `Sample` arm delegates to `samples.persist(...)` and the `Done` arm reads
    // `samples.all_samples` / `samples.cfg` for the final result fold.
    let mut samples = SamplePersister::new(settings, job, plan);

    let mut heartbeat_interval = tokio::time::interval(progress_report_interval(settings));
    heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Run the event loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
            let event = tokio::select! {
                maybe_event = rx.recv() => match maybe_event {
                    Some(event) => event,
                    None => break,
                },
                _ = heartbeat_interval.tick() => {
                    // Keep the worker's heartbeat alive during long silent gaps between engine events
                    // — e.g. a slow checkpoint disk-write + preview-sampling pass for a large model
                    // (Krea2). Heartbeats otherwise ride only on incoming progress events (below), so a
                    // quiet stretch past the API's 90s worker-timeout lets the stale-sweep mark this
                    // still-running job `interrupted`, and the next progress post is then 409'd
                    // (sc-8390; same class as the inline-upscale sc-8200). Also poll cancel here so a
                    // cancel requested during such a gap is honored without awaiting the next event.
                    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll (a local flag read) so a quit stops training at the next step / event gap.
                    if !canceled && (shutdown_requested() || cancel_requested_peek(api, &job.id).await)
                    {
                        begin_training_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                    }
                    continue;
                }
            };
            if canceled {
                continue; // drain remaining events so the blocking sender never blocks.
            }
            match event {
                TrainEvent::Progress(progress) => {
                    // Poll cancel on the long training band only (cheap stages fly by). A process
                    // shutdown (sc-9618) short-circuits the throttle + API poll so a quit stops on the
                    // very next training step, not just at the heartbeat-tick gap above.
                    if matches!(progress, TrainingProgress::Training { .. })
                        && (shutdown_requested()
                            || (last_cancel_check.elapsed() >= Duration::from_secs(2) && {
                                last_cancel_check = Instant::now();
                                cancel_requested_peek(api, &job.id).await
                            }))
                    {
                        begin_training_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                        continue;
                    }
                    if let TrainingProgress::Checkpoint { step } = progress {
                        checkpoints.push(json!({ "step": step }));
                    }
                    // sc-5637 — a preview sample: persist it + stream the updated cumulative/this-cadence
                    // lists so Training Studio shows it live. Handled here (not in `map_training_progress`)
                    // because it writes a file + carries a result payload; delegated to `SamplePersister`
                    // (sc-8921). Best-effort: a write failure logs and is skipped, never failing the run.
                    if let TrainingProgress::Sample {
                        step,
                        index,
                        total,
                        prompt,
                        image,
                    } = progress
                    {
                        samples
                            .persist(
                                api,
                                job,
                                backend,
                                total_steps,
                                step,
                                index,
                                total,
                                prompt,
                                image,
                            )
                            .await?;
                        continue;
                    }
                    let (status, stage, fraction, message) =
                        map_training_progress(progress, total_steps);
                    update_job(
                        api,
                        &job.id,
                        training_progress(status, stage, fraction, &message, None, backend),
                    )
                    .await?;
                    // No per-event heartbeat here: the `heartbeat_interval.tick()` arm above already
                    // pings `Busy` on the shared 5–15 s interval, keeping `last_seen` fresh independent of
                    // event cadence (posting progress does not refresh it). A heartbeat per progress event
                    // just doubled the API round-trips, throttling GPU stepping to API latency (sc-8917,
                    // F-115) with no keepalive benefit.
                }
                TrainEvent::Done(output) => {
                    let result = training_result(
                        plan,
                        &output,
                        &checkpoints,
                        &samples.all_samples,
                        &samples.cfg.sample_prompts,
                        samples.cfg.sample_steps,
                        samples.cfg.sample_guidance_scale,
                        backend,
                    );
                    update_job(
                        api,
                        &job.id,
                        training_progress(
                            JobStatus::Completed,
                            ProgressStage::Completed,
                            1.0,
                            &format!("Trained LoRA saved as {}.", plan.output.file_name),
                            Some(result),
                            backend,
                        ),
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // The event loop exited cleanly (channel closed after `Done`/cancel-drain), so the blocking
    // task is finished or finishing — reclaim its handle (disarming the drop-guard) and join it.
    let task_result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("training task join", error))?;
    if canceled {
        // Training has actually stopped now, so post the TERMINAL Canceled here (not at the
        // earlier cancel poll, which only tripped the flag + showed "Cancelling…"). This terminal
        // write is what frees the worker row (`jobs_store::update_job_progress`), so it lands as
        // the worker returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free (sc-5516; mirrors the image path sc-5515).
        let message = "LoRA training canceled by user.";
        update_job(
            api,
            &job.id,
            training_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    task_result
}

/// Map an engine [`TrainingProgress`] event onto a job `(status, stage, fraction,
/// message)`. The fractions follow the kernel's bands (prepare 0–.08, load .08–.18,
/// cache .18–.32, train/checkpoint .32–.92, save .92–1.0) so the UI's existing
/// `caching_latents`/`training`/`checkpointing`/`saving` stages light up unchanged.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn map_training_progress(
    progress: TrainingProgress,
    total_steps: u32,
) -> (JobStatus, ProgressStage, f64, String) {
    match progress {
        TrainingProgress::Preparing => (
            JobStatus::Running,
            ProgressStage::Preparing,
            0.06,
            "Preparing dataset.".to_owned(),
        ),
        TrainingProgress::LoadingModel => (
            JobStatus::Running,
            ProgressStage::LoadingModel,
            0.12,
            "Loading base model.".to_owned(),
        ),
        TrainingProgress::Caching { current, total } => {
            let span = if total == 0 {
                0.0
            } else {
                0.14 * (current as f64 / total as f64)
            };
            (
                JobStatus::Running,
                ProgressStage::CachingLatents,
                0.18 + span,
                format!("Caching dataset latents ({current}/{total})."),
            )
        }
        TrainingProgress::Training { step, total, loss } => (
            JobStatus::Running,
            ProgressStage::Training,
            train_fraction(step, total),
            format!("Training step {step} of {total} (loss {loss:.4})."),
        ),
        TrainingProgress::Checkpoint { step } => (
            JobStatus::Running,
            ProgressStage::Checkpointing,
            train_fraction(step, total_steps.max(step)),
            format!("Saved checkpoint at step {step}."),
        ),
        // sc-5637 — Sample events are intercepted in `consume_training_events` (they write a project
        // asset + stream the sample list) and never reach this mapper; the arm exists only to keep the
        // match exhaustive against the additive contract variant.
        TrainingProgress::Sample { .. } => {
            unreachable!("TrainingProgress::Sample is handled before map_training_progress")
        }
        TrainingProgress::Saving => (
            JobStatus::Running,
            ProgressStage::Saving,
            0.94,
            "Saving adapter.".to_owned(),
        ),
    }
}

/// Scale a training micro-step into the 0.32–0.92 training band.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn train_fraction(step: u32, total: u32) -> f64 {
    if total == 0 {
        return 0.32;
    }
    0.32 + 0.60 * (step as f64 / total as f64).clamp(0.0, 1.0)
}

/// Build the completion `result` the Training Studio reads (keys mirror the Python
/// trainer's `_result`). LoRA registration is driven by the staged `manifestEntry` +
/// the on-disk adapter (apps/rust-api `register_trained_lora`), not this result, so
/// this is informational/UI metadata.
/// Build the `result` payload carrying the accumulated preview samples (sc-5637). Streamed on each
/// rendered preview and folded into the final completion result, in the exact shape the Training
/// Studio reads: `trainingSamples` (cumulative), `latestTrainingSamples` (this cadence),
/// `samplePrompts` (for labels), and `sampleSettings` (steps + guidance + source). Mirrors the
/// Python trainer's `_result` sample keys.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn training_samples_result(
    all_samples: &[Value],
    latest_samples: &[Value],
    sample_prompts: &[String],
    sample_steps: u32,
    sample_guidance_scale: f32,
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("trainingSamples".to_owned(), json!(all_samples));
    result.insert("latestTrainingSamples".to_owned(), json!(latest_samples));
    result.insert("samplePrompts".to_owned(), json!(sample_prompts));
    result.insert(
        "sampleSettings".to_owned(),
        json!({
            "numInferenceSteps": sample_steps,
            "guidanceScale": sample_guidance_scale,
            "sampleSource": "live_adapter",
        }),
    );
    result
}

/// Persist one preview sample (sc-5637) as a PNG project asset and return the record the Training
/// Studio renders. The on-disk layout mirrors the Python trainer:
/// `<output_dir>/samples/step-NNNNNN/<stem>-stepNNNNNN-<index>.png`. `relativePath` is project-root-
/// relative (the UI resolves it as `/api/v1/projects/<id>/files/<relativePath>`); it is omitted when
/// the project root is unknown (the absolute `path` is still recorded for debugging). PNG encoding +
/// the atomic temp-then-rename mirror `image_jobs::write_image_asset`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
fn write_training_sample(
    output_dir: Option<&Path>,
    project_root: Option<&Path>,
    stem: &str,
    step: u32,
    index: u32,
    prompt: &str,
    image: gen_core::Image,
    sample_steps: u32,
    sample_guidance_scale: f32,
) -> WorkerResult<JsonObject> {
    let output_dir = output_dir.ok_or_else(|| {
        WorkerError::Engine("training preview: output directory is unavailable".to_owned())
    })?;
    let dir = output_dir.join("samples").join(format!("step-{step:06}"));
    std::fs::create_dir_all(&dir)?;
    let filename = format!("{stem}-step{step:06}-{index}.png");
    let path = dir.join(&filename);
    // Take `image` by value (sc-8909 / F-107) — the caller no longer needs the buffer, so the full
    // RGB `pixels.clone()` is dropped and the raw buffer is moved straight into the encoder.
    let rgb =
        image::RgbImage::from_raw(image.width, image.height, image.pixels).ok_or_else(|| {
            WorkerError::Engine("training preview: image buffer size mismatch".into())
        })?;
    let temp_path = path.with_extension("tmp.png");
    rgb.save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let mut record = JsonObject::new();
    record.insert("step".to_owned(), json!(step));
    record.insert("prompt".to_owned(), json!(prompt));
    record.insert("path".to_owned(), json!(path.display().to_string()));
    if let Some(relative) = project_root.and_then(|root| path.strip_prefix(root).ok()) {
        record.insert(
            "relativePath".to_owned(),
            json!(relative.to_string_lossy().replace('\\', "/")),
        );
    }
    record.insert("sampleSource".to_owned(), json!("live_adapter"));
    record.insert("numInferenceSteps".to_owned(), json!(sample_steps));
    record.insert("guidanceScale".to_owned(), json!(sample_guidance_scale));
    record.insert("createdAt".to_owned(), json!(now_rfc3339()));
    Ok(record)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
fn training_result(
    plan: &TrainingPlan,
    output: &TrainingOutput,
    checkpoints: &[Value],
    all_samples: &[Value],
    sample_prompts: &[String],
    sample_steps: u32,
    sample_guidance_scale: f32,
    backend: &str,
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("mode".to_owned(), json!("train"));
    result.insert("kernel".to_owned(), json!(plan.target.kernel));
    result.insert("loraId".to_owned(), json!(plan.output.lora_id));
    result.insert("outputDir".to_owned(), json!(plan.output.output_dir));
    result.insert("fileName".to_owned(), json!(plan.output.file_name));
    result.insert(
        "outputPath".to_owned(),
        json!(output.adapter_path.display().to_string()),
    );
    result.insert("format".to_owned(), json!(plan.output.format));
    result.insert("datasetId".to_owned(), json!(plan.dataset.dataset_id));
    result.insert(
        "datasetVersion".to_owned(),
        json!(plan.dataset.dataset_version),
    );
    result.insert(
        "datasetItemCount".to_owned(),
        json!(plan.dataset.items.len()),
    );
    result.insert("targetId".to_owned(), json!(plan.target.target_id));
    result.insert("baseModel".to_owned(), json!(plan.target.base_model));
    result.insert("steps".to_owned(), json!(plan.config.steps));
    result.insert("stepsCompleted".to_owned(), json!(output.steps));
    result.insert("finalLoss".to_owned(), json!(output.final_loss));
    result.insert("checkpoints".to_owned(), json!(checkpoints));
    result.insert("rank".to_owned(), json!(plan.config.rank));
    result.insert("alpha".to_owned(), json!(plan.config.alpha));
    result.insert("resolution".to_owned(), json!(plan.config.resolution));
    result.insert("triggerWords".to_owned(), json!(plan.output.trigger_words));
    result.insert("planVersion".to_owned(), json!(plan.plan_version));
    // Record the REAL backend that ran the training (mlx on macOS, candle off-Mac, cpu fallback),
    // not a hardcoded "mlx" — off-Mac candle runs otherwise stamped the wrong provenance (sc-8916,
    // F-114).
    result.insert("backend".to_owned(), json!(backend));
    // sc-5637 — fold the preview samples into the final result so they persist on the completed job
    // (the streamed updates are transient). `latestTrainingSamples` is left empty on completion (the
    // UI unions `trainingSamples` + `latestTrainingSamples`, so the cumulative list is sufficient).
    for (key, value) in training_samples_result(
        all_samples,
        &[],
        sample_prompts,
        sample_steps,
        sample_guidance_scale,
    ) {
        result.insert(key, value);
    }
    result
}

/// With no native trainer backend linked (not macOS/mlx-gen, and `backend-candle` off) the
/// `lora_train_execute` capability is never advertised, so a real run can never be claimed here.
/// Fail loudly if one is. Off-Mac with `backend-candle` the real `run_training_execution` above is
/// compiled instead (sc-7817).
#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
async fn run_training_execution(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
    _plan: &TrainingPlan,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Native LoRA training requires the macOS MLX engine or a `backend-candle` build; this \
         worker has neither linked and cannot execute it."
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every test that mutates the process-global `HF_HUB_CACHE` goes through the crate-wide env
    // seam. This module used to own a module-level mutex, on the reasoning that per-function
    // `static`s are distinct locks that do not serialize against each other (the intermittent "must
    // be inside an app-managed directory" flake on the macOS nax-worker lane). That reasoning was
    // right and simply did not go far enough: the whole crate's tests share ONE process, so a
    // per-MODULE lock is distinct from `video_jobs`' and `image_jobs`' in exactly the same way, and
    // those modules write `HF_HUB_CACHE` too (sc-12380).
    use crate::test_env::EnvVars;

    /// sc-9989: only LTX-2.3 gets a trainer `LoadSpec::text_encoder` override, and only from the
    /// bundled sibling `gemma/` (the self-contained turnkey install). Every other family's TE lives
    /// inside the weights dir → `None`. An operator `$LTX_GEMMA_DIR` is the intended passthrough → the
    /// engine reads the env var itself, so the override is `None`.
    #[cfg(target_os = "macos")]
    #[test]
    fn training_text_encoder_gates_on_engine_and_bundle() {
        let root = std::env::temp_dir().join(format!(
            "sw_ltx_train_te_{}_{}",
            std::process::id(),
            line!()
        ));
        let (tier, gemma) = (root.join("q4"), root.join("gemma"));
        std::fs::create_dir_all(&tier).unwrap();
        std::fs::create_dir_all(&gemma).unwrap();

        // Non-LTX engines never resolve an external TE (theirs lives inside the weights dir).
        assert!(training_text_encoder("kolors", &tier).is_none());
        assert!(training_text_encoder("sdxl", &tier).is_none());

        // LTX with a bundled `gemma/` sibling → threads it (unless an operator `$LTX_GEMMA_DIR` is set,
        // in which case the engine reads the env var and the override is `None`).
        let te = training_text_encoder("ltx_2_3", &tier);
        if std::env::var_os("LTX_GEMMA_DIR").is_none() {
            assert!(
                matches!(&te, Some(WeightsSource::Dir(p)) if *p == gemma),
                "expected the bundled gemma sibling to be threaded onto the trainer LoadSpec"
            );
        }

        // LTX with no `gemma/` sibling → `None` (engine falls back to its env/HF-cache path).
        let bare = root.join("no_sibling").join("q4");
        std::fs::create_dir_all(&bare).unwrap();
        assert!(training_text_encoder("ltx_2_3", &bare).is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    fn test_settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        }
    }

    /// A complete resolved plan as the API serializes it, parameterized by the
    /// fields the worker glue reads. `baseModelPath` is a path that does not exist,
    /// so `baseModelInstalled` is false unless a test overrides it.
    fn plan_json(
        data_dir: &Path,
        kernel: &str,
        base_model: &str,
        network_type: &str,
        image_paths: &[&str],
    ) -> Value {
        let dataset_root = data_dir.join("datasets").join("ds-1");
        let items: Vec<Value> = image_paths
            .iter()
            .map(|path| json!({ "imagePath": path, "caption": "a photo of mychar" }))
            .collect();
        json!({
            "schemaVersion": 1,
            "planVersion": 1,
            "jobId": "job-1",
            "target": {
                "targetId": format!("{kernel}_target"),
                "kernel": kernel,
                "family": "test",
                "modality": "image",
                "outputKind": "lora",
                "baseModel": base_model,
                "baseModelPath": data_dir.join("models").join("base-missing").display().to_string()
            },
            "dataset": {
                "datasetId": "ds-1",
                "datasetVersion": 1,
                "rootPath": dataset_root.display().to_string(),
                "items": items
            },
            "config": {
                "rank": 16,
                "alpha": 32,
                "learningRate": 0.0001,
                "steps": 1000,
                "batchSize": 1,
                "gradientAccumulation": 2,
                "resolution": 1024,
                "saveEvery": 250,
                "seed": 42,
                "optimizer": "adamw8bit",
                "advanced": {
                    "networkType": network_type,
                    "lrScheduler": "cosine",
                    "lrWarmupSteps": 50,
                    "weightDecay": 0.01,
                    "decomposeFactor": 8,
                    "loraTargetModules": ["to_q", "to_k"],
                    "timestepType": "sigmoid",
                    "timestepBias": "high_noise",
                    "lossType": "mse"
                }
            },
            "output": {
                "loraId": "lora-1",
                "outputDir": data_dir.join("loras").join("lora-1").display().to_string(),
                "fileName": "lora.safetensors",
                "format": "safetensors",
                "triggerWords": ["mychar"]
            },
            "provenance": {
                "datasetId": "ds-1",
                "datasetVersion": 1,
                "targetId": format!("{kernel}_target"),
                "baseModel": base_model,
                "configSnapshot": {},
                "outputLoraId": "lora-1",
                "sourceJobId": "job-1",
                "createdAt": "2026-06-06T00:00:00Z"
            }
        })
    }

    fn parse(value: Value) -> TrainingPlan {
        serde_json::from_value(value).expect("plan deserializes")
    }

    /// sc-4887: only an explicit bf16 selects bf16; every other value (incl. the
    /// engine-unsupported fp16, "no", empty) falls back to full-precision f32.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn normalize_train_dtype_only_bf16_selects_bf16() {
        assert_eq!(normalize_train_dtype("bf16"), "bf16");
        assert_eq!(normalize_train_dtype("BF16"), "bf16");
        assert_eq!(normalize_train_dtype("  bf16 "), "bf16");
        assert_eq!(normalize_train_dtype("fp16"), "f32");
        assert_eq!(normalize_train_dtype("no"), "f32");
        assert_eq!(normalize_train_dtype(""), "f32");
    }

    /// sc-4881 / sc-4887: the two OOM-fix levers must reach the engine config. The
    /// mapping builds it field-by-field (no `..Default::default()`), so a dropped
    /// field silently reverts to the wrong value — exactly the bug this story fixes.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn map_training_config_wires_train_dtype_and_gradient_checkpointing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let image = image.display().to_string();

        // Absent keys: the OOM fix stays on (bf16) and checkpointing stays off, so a
        // legacy plan that omits both still trains under the safe default.
        let default_plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image],
        ));
        let mapped = map_training_config(&default_plan.config);
        assert_eq!(mapped.train_dtype, "bf16");
        assert!(!mapped.gradient_checkpointing);

        // Explicit values flow through; a non-bf16 precision normalizes to f32.
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image],
        );
        value["config"]["advanced"]["mixedPrecision"] = json!("fp16");
        value["config"]["advanced"]["gradientCheckpointing"] = json!(true);
        let mapped = map_training_config(&parse(value).config);
        assert_eq!(mapped.train_dtype, "f32");
        assert!(mapped.gradient_checkpointing);
    }

    /// sc-7817 follow-up: the candle backend OOMs on a dense backward over the big-DiT training
    /// families (Z-Image, Wan A14B-T2V), so `finalize_training_config` must force gradient
    /// checkpointing on for them even when the resolved plan turns it off — a user un-checking the
    /// cross-platform "Gradient Checkpointing" UI box, or a thin submit that omits the key (the
    /// worker default is `false`). SDXL's smaller U-Net fits a dense backward, so its plan value is
    /// honored. Candle-only (the override is a candle-backend invariant); run with
    /// `cargo test -p sceneworks-worker --lib --features backend-candle`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn finalize_training_config_forces_checkpointing_for_candle_big_dit_families() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let image = image.display().to_string();

        // Map + finalize a plan with `gradientCheckpointing` explicitly OFF, and report whether the
        // engine config the worker would actually send has checkpointing on.
        let finalized_checkpointing = |kernel: &str, base_model: &str| -> bool {
            let mut value = plan_json(dir.path(), kernel, base_model, "lora", &[&image]);
            value["config"]["advanced"]["gradientCheckpointing"] = json!(false);
            let plan = parse(value);
            finalize_training_config(map_training_config(&plan.config), &plan)
                .gradient_checkpointing
        };

        // The candle-eligible big DiTs are forced on despite the plan's `false`, so a real off-Mac
        // run can't be configured into a dense-backward OOM.
        assert!(
            finalized_checkpointing("z_image_lora", "z_image_turbo"),
            "z-image forces gradient checkpointing on candle"
        );
        assert!(
            finalized_checkpointing("wan_moe_lora", "wan_2_2_t2v_14b"),
            "wan A14B T2V forces gradient checkpointing on candle"
        );
        // SDXL fits a dense backward, so its plan value (off) is honored — never forced on.
        assert!(
            !finalized_checkpointing("sdxl_lora", "sdxl"),
            "sdxl honors the plan (its U-Net fits dense)"
        );

        // A plan that already requests checkpointing is unchanged (the override only flips false→true).
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image],
        );
        value["config"]["advanced"]["gradientCheckpointing"] = json!(true);
        let plan = parse(value);
        assert!(
            finalize_training_config(map_training_config(&plan.config), &plan)
                .gradient_checkpointing,
            "an explicit gradientCheckpointing=true is preserved"
        );
    }

    #[test]
    fn validate_rejects_unknown_plan_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["planVersion"] = json!(999);
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("version mismatch rejected");
        assert!(error
            .to_string()
            .contains("Unsupported training plan version"));
    }

    #[test]
    fn validate_rejects_empty_dataset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("empty dataset rejected");
        assert!(error.to_string().contains("no items"));
    }

    #[test]
    fn validate_rejects_missing_image() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("missing.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("missing image rejected");
        assert!(error.to_string().contains("missing on the worker"));
    }

    #[test]
    fn validate_accepts_present_images() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        assert!(validate_training_plan(&settings, &plan).is_ok());
    }

    #[test]
    fn validate_rejects_image_outside_dataset_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("other").join("image.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("outside image rejected");
        assert!(error.to_string().contains("Training dataset imagePath"));
    }

    #[test]
    fn validate_rejects_base_model_outside_data_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("image.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["target"]["baseModelPath"] = json!("/tmp/sceneworks-outside-base");
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("outside base model rejected");
        assert!(error.to_string().contains("Training baseModelPath"));
    }

    /// The base model is a read-only weights source the rust-api resolves from the
    /// shared Hugging Face hub cache, which the desktop points outside `data_dir`
    /// via `HF_HOME`. Such a path must be accepted even though it is not under the
    /// app data dir (regression for the z_image_turbo "must be inside an
    /// app-managed directory" training failure). Serialized so the `HF_HUB_CACHE`
    /// mutation can't race other env-reading tests.
    #[test]
    fn validate_accepts_base_model_in_hf_cache_outside_data_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let hf_cache = tempfile::tempdir().expect("hf cache tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");

        // The base model lives under the HF hub cache (outside data_dir), exactly
        // as `resolve_base_model_path` returns for an HF-cache-resident model.
        let base_model = hf_cache
            .path()
            .join("models--Tongyi-MAI--Z-Image-Turbo")
            .join("snapshots")
            .join("abc123");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["target"]["baseModelPath"] = json!(base_model.display().to_string());

        let result = {
            let _env = EnvVars::set(&[("HF_HUB_CACHE", hf_cache.path().to_str().expect("utf-8"))]);
            validate_training_plan(&settings, &parse(value))
        };

        assert!(
            result.is_ok(),
            "HF-cache base model should validate: {result:?}"
        );
    }

    /// The REAL training run (`run_training_execution`) resolves the base model
    /// weights via `resolve_app_managed_model_dir`, a path separate from the
    /// dry-run validator. It must also accept an HF-cache-resident model dir, or
    /// the dry run passes while the real run fails with "must be inside an
    /// app-managed directory" (the z_image_turbo regression: dry run completed,
    /// real run rejected the same `~/.cache/huggingface` snapshot).
    #[test]
    fn resolve_app_managed_model_dir_accepts_hf_cache_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let hf_cache = tempfile::tempdir().expect("hf cache tempdir");
        let settings = test_settings(dir.path());

        // A real, existing snapshot dir under the HF hub cache (outside data_dir),
        // exactly what `resolve_base_model_path` hands the worker.
        let snapshot = hf_cache
            .path()
            .join("models--Tongyi-MAI--Z-Image-Turbo")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let resolved = {
            let _env = EnvVars::set(&[("HF_HUB_CACHE", hf_cache.path().to_str().expect("utf-8"))]);
            resolve_app_managed_model_dir(
                &settings,
                &snapshot.display().to_string(),
                "Training baseModelPath",
            )
        };

        assert!(
            resolved.is_ok(),
            "HF-cache model dir should resolve for the real run: {resolved:?}"
        );
    }

    /// sc-8878 (F-076): a forged `output.file_name` carrying a path separator / `..` / an absolute
    /// path must be rejected before the engine joins it under the confined output dir — otherwise the
    /// adapter write escapes the app-managed output root. The plan is otherwise valid (present dataset
    /// image, in-root output dir), so only the malicious file name trips the guard.
    #[test]
    fn validate_rejects_traversal_file_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");
        for bad in [
            "../evil.safetensors",
            "../../etc/passwd",
            "sub/evil.safetensors",
            "/tmp/evil.safetensors",
            "..",
        ] {
            let mut value = plan_json(
                dir.path(),
                "z_image_lora",
                "z_image_turbo",
                "lora",
                &[&image.display().to_string()],
            );
            value["output"]["fileName"] = json!(bad);
            let error = validate_training_plan(&settings, &parse(value))
                .expect_err(&format!("traversal file name '{bad}' must be rejected"));
            assert!(
                error.to_string().contains("Training fileName"),
                "rejection for '{bad}' should name the field: {error}"
            );
        }
        // A plain single-component file name still validates.
        let mut ok = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        ok["output"]["fileName"] = json!("my_style.safetensors");
        assert!(
            validate_training_plan(&settings, &parse(ok)).is_ok(),
            "a plain file name should validate"
        );
    }

    #[test]
    fn validate_rejects_output_dir_outside_app_lora_or_model_roots() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("image.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["output"]["outputDir"] =
            json!(dir.path().join("tmp").join("lora-1").display().to_string());
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("outside output dir rejected");
        assert!(error.to_string().contains("Training outputDir"));
    }

    /// Project-scoped training (the default) writes to the owning project's tree,
    /// `<data>/projects/<slug>.sceneworks/loras/<lora_id>`, which is under the app
    /// data dir but not under `<data>/loras` or `<data>/models`. The worker must
    /// accept it (regression for the "Training outputDir must be inside an
    /// app-managed directory" failure on project-scoped runs).
    #[test]
    fn validate_accepts_project_scoped_output_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["output"]["outputDir"] = json!(dir
            .path()
            .join("projects")
            .join("my-character.sceneworks")
            .join("loras")
            .join("lora-1")
            .display()
            .to_string());
        assert!(
            validate_training_plan(&settings, &parse(value)).is_ok(),
            "project-scoped output dir should validate"
        );
    }

    #[test]
    fn dry_run_summary_carries_plan_facts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image_a = dir.path().join("datasets").join("ds-1").join("a.png");
        let image_b = dir.path().join("datasets").join("ds-1").join("b.png");
        let plan = parse(plan_json(
            dir.path(),
            "sdxl_lora",
            "sdxl",
            "lora",
            &[
                &image_a.display().to_string(),
                &image_b.display().to_string(),
            ],
        ));
        let summary = dry_run_summary(&settings, &plan);
        assert_eq!(summary.get("mode").unwrap(), "dry_run");
        assert_eq!(summary.get("kernel").unwrap(), "sdxl_lora");
        assert_eq!(summary.get("datasetItemCount").unwrap(), 2);
        assert_eq!(summary.get("loraId").unwrap(), "lora-1");
        assert_eq!(summary.get("fileName").unwrap(), "lora.safetensors");
        // The placeholder base path does not exist.
        assert_eq!(summary.get("baseModelInstalled").unwrap(), false);
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn engine_trainer_id_maps_mlx_native_families_and_rejects_the_rest() {
        let cases: &[(&str, &str, Option<&str>)] = &[
            ("z_image_lora", "z_image_turbo", Some("z_image_turbo")),
            ("sdxl_lora", "sdxl", Some("sdxl")),
            // Illustrious v1.0/v2.0 (epic 10609) share the `sdxl_lora` kernel — vanilla SDXL —
            // so both base models resolve to the `sdxl` engine trainer with no branch here,
            // unlike `sd3_lora`/`anima_lora` which split on base_model. Pin that.
            ("sdxl_lora", "illustrious_xl_v1", Some("sdxl")),
            ("sdxl_lora", "illustrious_xl_v2", Some("sdxl")),
            ("ltx_mlx_lora", "ltx_2_3", Some("ltx_2_3")),
            ("wan_lora", "wan_2_2", Some("wan2_2_ti2v_5b")),
            ("wan_moe_lora", "wan_2_2_t2v_14b", Some("wan2_2_t2v_14b")),
            ("wan_moe_lora", "wan_2_2_i2v_14b", Some("wan2_2_i2v_14b")),
            // Kolors gained a native mlx-gen trainer (sc-4568) and now routes here (sc-4732);
            // the trainer registers under the generator id `"kolors"`.
            ("kolors_lora", "kolors", Some("kolors")),
            // Lens gained a native mlx-gen trainer (sc-5148) and now routes here (sc-5180); the
            // trainer registers under the base generator id `"lens"`.
            ("lens_lora", "lens", Some("lens")),
            // Krea trains the undistilled Raw DiT; the engine trainer registers under the base id
            // `"krea_2_raw"` (sc-7577/7578).
            ("krea_lora", "krea_2_raw", Some("krea_2_raw")),
            // SD3.5 (sc-7884): the `sd3_lora` kernel splits by training base — the engine trainer
            // registers under the inference-generator id of each base.
            ("sd3_lora", "sd3_5_large", Some("sd3_5_large")),
            ("sd3_lora", "sd3_5_medium", Some("sd3_5_medium")),
            // Anima (sc-10522): the `anima_lora` kernel maps to the trainer registered under the
            // inference-generator id of the training-base variant; `anima_base` is the default.
            ("anima_lora", "anima_base", Some("anima_base")),
            ("anima_lora", "anima_aesthetic", Some("anima_aesthetic")),
            ("anima_lora", "anima_turbo", Some("anima_turbo")),
            // Unknown SD3.5 base model variant (e.g. Turbo is NOT a training base).
            ("sd3_lora", "sd3_5_large_turbo", None),
            // Unknown A14B base model variant.
            ("wan_moe_lora", "wan_2_2_mystery", None),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        for (kernel, base_model, expected) in cases {
            let plan = parse(plan_json(
                dir.path(),
                kernel,
                base_model,
                "lora",
                &[&image.display().to_string()],
            ));
            assert_eq!(
                engine_trainer_id(&plan),
                *expected,
                "kernel={kernel} base_model={base_model}"
            );
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn map_training_config_reads_advanced_and_passes_optimizer_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lokr",
            &[&image.display().to_string()],
        ));
        let cfg = map_training_config(&plan.config);
        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha as u32, 32);
        assert_eq!(cfg.steps, 1000);
        assert_eq!(cfg.gradient_accumulation, 2);
        assert_eq!(cfg.seed, 42);
        // The optimizer alias is passed verbatim; the engine normalizes it.
        assert_eq!(cfg.optimizer, "adamw8bit");
        assert!((cfg.weight_decay - 0.01).abs() < 1e-6);
        assert!((cfg.learning_rate - 0.0001).abs() < 1e-6);
        assert_eq!(cfg.lr_warmup_steps, 50);
        assert_eq!(cfg.decompose_factor, 8);
        assert!(matches!(cfg.network_type, NetworkType::Lokr));
        assert!(matches!(cfg.lr_scheduler, LrSchedule::Cosine));
        assert_eq!(
            cfg.lora_target_modules,
            vec!["to_q".to_owned(), "to_k".to_owned()]
        );
        assert_eq!(cfg.timestep_bias, "high_noise");
        assert_eq!(cfg.trigger_word, None);
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn map_training_config_defaults_when_advanced_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "sdxl_lora",
            "sdxl",
            "lora",
            &[&image.display().to_string()],
        );
        value["config"]["advanced"] = json!({});
        let plan = parse(value);
        let cfg = map_training_config(&plan.config);
        assert!(matches!(cfg.network_type, NetworkType::Lora));
        assert!(matches!(cfg.lr_scheduler, LrSchedule::Constant));
        assert_eq!(cfg.lr_warmup_steps, 0);
        assert_eq!(cfg.decompose_factor, -1);
        assert!(cfg.lora_target_modules.is_empty());
        assert_eq!(cfg.timestep_type, "sigmoid");
        assert_eq!(cfg.loss_type, "mse");
        // sc-5637 — absent sample keys ⇒ sampling OFF (sample_every 0), so a legacy plan that omits
        // them trains exactly as before (no previews) rather than erroring.
        assert_eq!(cfg.sample_every, 0);
        assert!(cfg.sample_prompts.is_empty());
    }

    /// sc-5637 — the preview-sample config must reach the engine. The mapping builds the engine config
    /// field-by-field (no `..Default`), so a dropped sample field silently disables previews — exactly
    /// the gap this story fixes. Asserts `sampleEvery`/`sampleSteps`/`sampleGuidanceScale`/`samplePrompts`
    /// flow through from the plan's `advanced` bag.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn map_training_config_wires_sample_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["config"]["advanced"]["sampleEvery"] = json!(250);
        value["config"]["advanced"]["sampleSteps"] = json!(8);
        value["config"]["advanced"]["sampleGuidanceScale"] = json!(1.5);
        value["config"]["advanced"]["samplePrompts"] =
            json!(["mychar, studio portrait", "mychar, full body"]);
        let cfg = map_training_config(&parse(value).config);
        assert_eq!(cfg.sample_every, 250);
        assert_eq!(cfg.sample_steps, 8);
        assert!((cfg.sample_guidance_scale - 1.5).abs() < 1e-6);
        assert_eq!(
            cfg.sample_prompts,
            vec![
                "mychar, studio portrait".to_owned(),
                "mychar, full body".to_owned()
            ]
        );
    }

    /// sc-8671 — `sampleCount` caps how many previews render (one per prompt, truncated, never padded).
    /// A count below the pool size truncates; a count at/above it leaves the pool intact.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn map_training_config_caps_sample_prompts_at_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["config"]["advanced"]["samplePrompts"] = json!(["p1", "p2", "p3"]);
        value["config"]["advanced"]["sampleCount"] = json!(2);
        let capped = map_training_config(&parse(value.clone()).config);
        assert_eq!(
            capped.sample_prompts,
            vec!["p1".to_owned(), "p2".to_owned()]
        );

        // Count >= pool size leaves every prompt (no padding/duplication).
        value["config"]["advanced"]["sampleCount"] = json!(9);
        let uncapped = map_training_config(&parse(value).config);
        assert_eq!(
            uncapped.sample_prompts,
            vec!["p1".to_owned(), "p2".to_owned(), "p3".to_owned()]
        );
    }

    /// sc-5637 — `write_training_sample` must persist the PNG under
    /// `<output_dir>/samples/step-NNNNNN/<stem>-stepNNNNNN-<index>.png` and return a record carrying
    /// the exact shape the Training Studio reads (`step`/`prompt`/`relativePath`/`numInferenceSteps`/
    /// `guidanceScale`/`sampleSource`), with `relativePath` resolved against the project root. Validates
    /// the worker persistence deterministically (no model weights), so the chain engine→worker→UI shape
    /// is covered without a real-weight run.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn write_training_sample_writes_png_and_project_relative_record() {
        let project_root = tempfile::tempdir().expect("project root tempdir");
        let output_dir = project_root
            .path()
            .join("training")
            .join("loras")
            .join("lora-1");
        // A 4×2 solid-red RGB bitmap (pixels.len() == 4*2*3).
        let image = gen_core::Image {
            width: 4,
            height: 2,
            pixels: vec![255u8, 0, 0]
                .into_iter()
                .cycle()
                .take(4 * 2 * 3)
                .collect(),
        };
        let record = write_training_sample(
            Some(output_dir.as_path()),
            Some(project_root.path()),
            "mychar",
            250,
            2,
            "mychar, studio portrait",
            image,
            8,
            1.5,
        )
        .expect("sample persists");

        let png = output_dir
            .join("samples")
            .join("step-000250")
            .join("mychar-step000250-2.png");
        assert!(png.exists(), "preview PNG written to {}", png.display());
        assert_eq!(record.get("step").unwrap(), 250);
        assert_eq!(record.get("prompt").unwrap(), "mychar, studio portrait");
        assert_eq!(record.get("sampleSource").unwrap(), "live_adapter");
        assert_eq!(record.get("numInferenceSteps").unwrap(), 8);
        assert!((record.get("guidanceScale").unwrap().as_f64().unwrap() - 1.5).abs() < 1e-6);
        assert_eq!(
            record.get("relativePath").unwrap(),
            "training/loras/lora-1/samples/step-000250/mychar-step000250-2.png"
        );
        // The PNG must decode back to the source dimensions.
        let decoded = image::open(&png).expect("re-open png").to_rgb8();
        assert_eq!((decoded.width(), decoded.height()), (4, 2));
    }

    /// sc-9812 (F-075 follow-up) — INTEGRATION-level guard for the caller-symmetry regression the
    /// unit test above cannot catch. `SamplePersister::new` derives its two `strip_prefix` operands
    /// from DIFFERENT provenance: `output_dir` from `resolve_training_output_dir`
    /// (`normalize_app_managed_path` -> `normalize_existing_or_absolute`, canonicalized) and
    /// `project_root` from the project store (`get_project(...).path`, persisted LEXICALLY as
    /// `data_dir.join(...).display()`). On macOS the app data dir under `/var/...` canonicalizes to
    /// `/private/var/...`, so before the fix the two forms diverged and `write_training_sample`'s
    /// `strip_prefix` dropped `relativePath` from every project-scoped sample record. We build the
    /// project root the way `project_store` does (a real `create_project`), NOT a hand-matched
    /// lexical pair, and assert the record's `relativePath` is populated and correct. This FAILS
    /// before the `SamplePersister::new` canonicalization and PASSES after.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn sample_persister_populates_relative_path_for_project_store_provisioned_root() {
        let data_dir = tempfile::tempdir().expect("data dir tempdir");
        let settings = test_settings(data_dir.path());

        // Provision the project exactly as production does: the store persists `path`
        // lexically (`project_path.display().to_string()`), which on macOS keeps the
        // `/var/...` form even though the real path resolves under `/private/var/...`.
        let store = ProjectStore::new(settings.data_dir.to_path_buf(), "test");
        let project = store
            .create_project("Char Studio")
            .expect("project provisions");

        // Project-scoped output lives under the owning project's tree, the default scope.
        // `resolve_training_output_dir` (via `normalize_app_managed_path`) canonicalizes it,
        // so this is the operand whose form diverges from the lexical `project.path`.
        let output_dir = PathBuf::from(&project.path).join("loras").join("lora-1");

        let mut plan_value = plan_json(
            data_dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &["x.png"],
        );
        plan_value["output"]["outputDir"] = json!(output_dir.display().to_string());
        plan_value["output"]["fileName"] = json!("mychar.safetensors");
        let plan = parse(plan_value);

        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-1",
            "type": "train_lora",
            "status": "running",
            "projectId": project.id,
            "projectName": "Char Studio",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": "test-worker",
            "progress": 0.0,
            "stage": "queued",
            "message": "queued",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 1,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-07-04T00:00:00Z",
            "updatedAt": "2026-07-04T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        }))
        .expect("job snapshot deserializes");

        // Drive the REAL provenance — this is where the two operands are resolved.
        let persister = SamplePersister::new(&settings, &job, &plan);
        let resolved_output = persister
            .output_dir
            .clone()
            .expect("output dir resolves under the projects tree");
        let resolved_project_root = persister
            .project_root
            .clone()
            .expect("project root resolves from the store");

        let image = gen_core::Image {
            width: 4,
            height: 2,
            pixels: vec![255u8, 0, 0]
                .into_iter()
                .cycle()
                .take(4 * 2 * 3)
                .collect(),
        };
        let record = write_training_sample(
            Some(resolved_output.as_path()),
            Some(resolved_project_root.as_path()),
            &persister.stem,
            250,
            2,
            "mychar, studio portrait",
            image,
            8,
            1.5,
        )
        .expect("sample persists");

        // The regression: before the fix, `strip_prefix(project_root)` fails on macOS and this
        // key is absent. It must be present and project-root-relative (the shape the Training
        // Studio resolves as `/api/v1/projects/<id>/files/<relativePath>`).
        let relative = record
            .get("relativePath")
            .and_then(|value| value.as_str())
            .expect("relativePath is populated for a project-scoped sample");
        assert!(
            !relative.is_empty(),
            "relativePath must be non-empty, got {relative:?}"
        );
        assert_eq!(
            relative, "loras/lora-1/samples/step-000250/mychar-step000250-2.png",
            "relativePath must be project-root-relative"
        );
    }

    /// Real-weights smoke (sc-4732 + sc-4764): load the Kolors trainer from the installed
    /// `Kwai-Kolors/Kolors-diffusers` snapshot and run two LoRA micro-steps on a one-image dataset.
    /// Proves the worker links `mlx-gen-kolors` (the `load_trainer("kolors", …)` registration), the
    /// snapshot's overlaid `tokenizer/tokenizer.json` (sc-4764) lets the trainer construct, and a
    /// real step runs (finite loss) + writes an adapter on the Mac GPU. The trainer loads the base
    /// at **f32** (engine choice for clean autograd), so this is memory-heavy. Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored kolors_real_weights_trains --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Kolors weights (+ tokenizer.json overlay) + Metal device; f32 base is heavy"]
    fn kolors_real_weights_trains_a_lora_step() {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME set"));
        let snapshot = std::fs::read_dir(
            home.join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots"),
        )
        .expect("kolors snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a kolors snapshot dir");
        assert!(
            snapshot.join("tokenizer").join("tokenizer.json").exists(),
            "kolors snapshot is missing the overlaid tokenizer.json (sc-4764)"
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(512, 512, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_owned(),
            resolution: 512,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            resume: false,
            control_type: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
                control_image_path: None,
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: "kolors_smoke.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            "kolors",
            &LoadSpec::new(WeightsSource::Dir(snapshot)),
        )
        .expect("kolors trainer loads (tokenizer.json present)");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training runs a step");

        eprintln!(
            "[kolors-train-smoke] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(output.steps >= 1, "expected at least one micro-step");
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        assert!(
            output_dir.join("kolors_smoke.safetensors").exists(),
            "trained adapter was written"
        );
    }

    /// sc-5180 — the Lens training cutover's worker smoke: load the Lens trainer from the installed
    /// `SceneWorks/Lens` snapshot (the training-base diffusers rehost; the original `microsoft/Lens`
    /// is dead, sc-8797) and run two LoRA micro-steps on a one-image dataset. Proves the
    /// worker LINKS `mlx-gen-lens` (the `load_trainer("lens", …)` trainer registration survives
    /// linker GC — the dead-strip gotcha that bit Kolors) and a real step runs (finite loss) + writes
    /// an adapter through the full worker path. The trainer loads the gpt-oss encoder Q8 (~12 GB) +
    /// the DiT bf16 (res 64 keeps the per-step graph tiny — the encoder load is the heavy part).
    /// Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored lens_real_weights_trains --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights + Metal device; loads the 20B gpt-oss encoder (Q8)"]
    fn lens_real_weights_trains_a_lora_step() {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME set"));
        let snapshot = std::fs::read_dir(
            home.join(".cache/huggingface/hub/models--SceneWorks--Lens/snapshots"),
        )
        .expect("lens snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir() && path.join("transformer").is_dir())
        .expect("a SceneWorks/Lens snapshot dir");

        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(256, 256, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_owned(),
            resolution: 64,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            resume: false,
            control_type: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
                control_image_path: None,
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: "lens_smoke.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            "lens",
            &LoadSpec::new(WeightsSource::Dir(snapshot)),
        )
        .expect("lens trainer loads from the explicit runtime catalog");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training runs a step");

        eprintln!(
            "[lens-train-smoke] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(output.steps >= 1, "expected at least one micro-step");
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        assert!(
            output_dir.join("lens_smoke.safetensors").exists(),
            "trained adapter was written"
        );
    }

    /// Real-weights production-scale smoke (sc-4881 / sc-4874+4886+4887, Part A4): load the
    /// z-image trainer from the installed `Tongyi-MAI/Z-Image-Turbo` snapshot and run two LoRA
    /// micro-steps **at resolution 1024 with `train_dtype="bf16"`** — the exact configuration
    /// that SIGKILL-OOM'd the worker before this fix (the 1024² first step materialized ~135 GB,
    /// over the 128 GB unified-memory budget). The image *count* doesn't change the first-step peak (batch 1;
    /// the peak is the per-step forward graph), so a one-image dataset faithfully reproduces the
    /// memory profile of the 221-image production run. Passing step 1 with a finite loss proves
    /// bf16 brings the peak under budget through the **full worker path** (`map_training_config`
    /// → `load_trainer` → `train`), not just the engine isolation tests. `gradient_checkpointing`
    /// stays off here to prove bf16 alone is sufficient. Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored z_image_1024_bf16 --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Z-Image-Turbo weights + Metal device; loads the full model and peaks ~44 GB at 1024"]
    fn z_image_1024_bf16_trains_past_the_first_step() {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME set"));
        let snapshot = std::fs::read_dir(
            home.join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots"),
        )
        .expect("z-image snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a z-image snapshot dir");

        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(1024, 1024, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            // The fix under test: bf16 forward, checkpointing OFF — bf16 alone must suffice.
            gradient_checkpointing: false,
            train_dtype: "bf16".to_owned(),
            resolution: 1024,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            resume: false,
            control_type: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
                control_image_path: None,
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: "z_image_1024_smoke.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot)),
        )
        .expect("z-image trainer loads");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training survives the 1024 first step (no OOM)");

        eprintln!(
            "[z-image-1024-bf16-smoke] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(
            output.steps >= 1,
            "expected at least one micro-step past step 1"
        );
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        assert!(
            output_dir.join("z_image_1024_smoke.safetensors").exists(),
            "trained adapter was written"
        );
    }

    // ── sc-7817: candle (Windows/CUDA + Linux/NVIDIA) real-weight training smokes ──────────────────
    // The off-Mac twins of the macOS real-weight smokes above. They prove the CUDA bundle explicitly
    // includes each candle trainer, resolves it via `crate::inference_runtime::load_trainer(id, …)`,
    // and runs a real step on the CUDA GPU through
    // the full worker path. Release build only (the GPU smokes hit the debug_assert dup-id panic in
    // debug, per the candle CI-gap note).

    /// Resolve a snapshot directory for an HF repo from the local Hugging Face hub cache, the way
    /// `resolve_base_model_path` hands the worker a base model dir. Honors `HF_HUB_CACHE`, then
    /// `HF_HOME/hub`, then `%USERPROFILE%\.cache\huggingface\hub` (the Windows default). `repo_dir`
    /// is the hub's `models--<org>--<name>` directory; `require` (when non-empty) filters snapshots
    /// to one containing that child, so a partial download is skipped.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    fn candle_hf_snapshot(repo_dir: &str, require: &str) -> std::path::PathBuf {
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HF_HOME").map(|h| std::path::PathBuf::from(h).join("hub"))
            })
            .or_else(|| {
                std::env::var_os("USERPROFILE").map(|p| {
                    std::path::PathBuf::from(p)
                        .join(".cache")
                        .join("huggingface")
                        .join("hub")
                })
            })
            .expect("an HF hub cache (HF_HUB_CACHE / HF_HOME / %USERPROFILE%) must be resolvable");
        let snapshots = hub.join(repo_dir).join("snapshots");
        std::fs::read_dir(&snapshots)
            .unwrap_or_else(|error| panic!("read {}: {error}", snapshots.display()))
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir() && (require.is_empty() || path.join(require).exists()))
            .unwrap_or_else(|| {
                panic!(
                    "no snapshot under {} containing {require:?}",
                    snapshots.display()
                )
            })
    }

    /// Shared body for the candle real-weight training smokes: two LoRA micro-steps on a one-image
    /// swatch dataset, asserting a finite loss and a written adapter — the candle twin of the macOS
    /// `z_image_1024_bf16_trains_past_the_first_step` / `kolors_real_weights_*` smokes.
    ///
    /// `gradient_checkpointing` matters per-family on candle: UNLIKE MLX/torch, candle's matmul
    /// backward materializes a gradient for the FROZEN base weight too, so a dense backward over a
    /// multi-billion-param DiT (Z-Image, Krea Raw 12B, Wan) OOMs even at 512² on a 96 GB card — checkpointing
    /// (sc-5246) recomputes activations and avoids retaining those frozen-weight grads. SDXL's smaller
    /// U-Net fits dense; Z-Image needs checkpointing on. (A real training job carries this from its
    /// preset; the smoke sets it explicitly per family.)
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    fn candle_real_weights_smoke(
        engine_id: &str,
        snapshot: &std::path::Path,
        resolution: u32,
        gradient_checkpointing: bool,
        file_name: &str,
    ) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(resolution, resolution, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing,
            train_dtype: "bf16".to_owned(),
            resolution,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            resume: false,
            control_type: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
                control_image_path: None,
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: file_name.to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            engine_id,
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .unwrap_or_else(|error| {
            panic!("candle {engine_id} trainer loads from the runtime catalog: {error}")
        });
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training runs a step");

        eprintln!(
            "[candle-train-smoke {engine_id}] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(output.steps >= 1, "expected at least one micro-step");
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        // Assert on the path the trainer actually wrote (`output.adapter_path`), not
        // `output_dir.join(file_name)` — the Wan MoE trainer emits a `{stem}.high_noise` /
        // `{stem}.low_noise` PAIR (adapter_path points at the high-noise expert), so a bare `file_name`
        // check is wrong for it; the single-adapter families write exactly `file_name`.
        assert!(
            output.adapter_path.exists(),
            "trained adapter was written at {}",
            output.adapter_path.display()
        );
    }

    /// sc-7817 — the candle SDXL training smoke. Loads `load_trainer("sdxl", …)` from the installed
    /// `stabilityai/stable-diffusion-xl-base-1.0` diffusers snapshot (`text_encoder/ text_encoder_2/
    /// unet/`) and trains two LoRA micro-steps; the smaller U-Net fits a dense backward (no
    /// checkpointing). GPU-validated on RTX PRO 6000 (2 steps, finite loss ~0.02, adapter written).
    /// Run on demand on the RTX box (one smoke per GPU — two full models at once OOM):
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored candle_sdxl_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real SDXL weights + a CUDA device; run one smoke per GPU (two full models OOM)"]
    fn candle_sdxl_real_weights_trains_a_lora_step() {
        let snapshot =
            candle_hf_snapshot("models--stabilityai--stable-diffusion-xl-base-1.0", "unet");
        candle_real_weights_smoke(
            "sdxl",
            &snapshot,
            512,
            false,
            "candle_sdxl_smoke.safetensors",
        );
    }

    /// sc-10618 — the candle (CUDA) twin of the macOS Illustrious train smoke. Points the config-blind
    /// `load_trainer("sdxl", …)` at an Illustrious turnkey's dense `bf16/` tier (the tier
    /// `resolve_base_model_path` trains from) and runs a LoRA micro-step, proving the candle SDXL trainer
    /// trains the Illustrious base on CUDA exactly as it trains stock SDXL — Illustrious differs only in
    /// its base weights (the gen crates are config-blind). Upstream ships a single-file LDM (no plain
    /// diffusers repo), so point `ILL_CANDLE_BF16_DIR` at the turnkey's `bf16/` tier on the RTX box:
    /// `ILL_CANDLE_BF16_DIR=/hfcache/models--SceneWorks--illustrious-xl-v1-mlx/snapshots/<rev>/bf16 \
    ///   cargo test -p sceneworks-worker --lib --features backend-candle --release -- \
    ///   --ignored candle_illustrious_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs an Illustrious turnkey bf16 tier + a CUDA device; set ILL_CANDLE_BF16_DIR"]
    fn candle_illustrious_real_weights_trains_a_lora_step() {
        let Some(dir) = std::env::var_os("ILL_CANDLE_BF16_DIR")
            .map(std::path::PathBuf::from)
            .filter(|p| p.join("unet").is_dir())
        else {
            eprintln!(
                "ILL_CANDLE_BF16_DIR unset or missing unet/ — skipping candle Illustrious smoke"
            );
            return;
        };
        candle_real_weights_smoke(
            "sdxl",
            &dir,
            512,
            false,
            "candle_illustrious_lora.safetensors",
        );
    }

    /// sc-7817 — the candle Z-Image training smoke. Loads `load_trainer("z_image_turbo", …)` from the
    /// installed `Tongyi-MAI/Z-Image-Turbo` snapshot and trains two LoRA micro-steps with
    /// **gradient checkpointing ON** — REQUIRED on candle: the Z-Image DiT's dense backward OOMs even
    /// at 512² on a 96 GB card because candle materializes a grad for the frozen base weight too
    /// (sc-5246; the Wan-14B finding). Run on demand on the RTX box (one smoke per GPU):
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored candle_z_image_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real Z-Image-Turbo weights + a CUDA device; run one smoke per GPU (two full models OOM)"]
    fn candle_z_image_real_weights_trains_a_lora_step() {
        let snapshot = candle_hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo", "");
        candle_real_weights_smoke(
            "z_image_turbo",
            &snapshot,
            512,
            true,
            "candle_z_image_smoke.safetensors",
        );
    }

    /// sc-8671 GPU-VAL — proves `sampleCount` caps the rendered previews end-to-end on the candle
    /// path: a 3-prompt pool with `sampleCount=2` must (a) reach the engine as exactly 2 prompts via
    /// the real `map_training_config` mapping, and (b) render exactly 2 preview PNGs per sample step
    /// (the engine renders one image per prompt), which `write_training_sample` persists to
    /// `<out>/samples/step-NNNNNN/`. Trains Z-Image-Turbo (checkpointing ON, required on candle) with
    /// `sampleEvery=1`, `sampleSteps=4` (fast preview). Needs candle-gen ≥ #1008 (the preview-sample
    /// emission). Run on the RTX box:
    /// `HF_HUB_CACHE=D:\hfcache CUDA_VISIBLE_DEVICES=0 cargo test -p sceneworks-worker --lib --features backend-candle --release -- --ignored candle_sample_count_caps_previews --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real Z-Image-Turbo weights + a CUDA device; GPU-val for sc-8671 sampleCount cap"]
    fn candle_sample_count_caps_previews() {
        let snapshot = candle_hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo", "");
        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(512, 512, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        // Drive the REAL merged mapping: a 3-prompt pool capped at sampleCount=2 must map to exactly
        // 2 engine prompts (one preview per prompt), proving training_jobs::resolve_sample_prompts.
        let pool = ["alpha portrait", "beta full body", "gamma outdoors"];
        let mut value = plan_json(
            tmp.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image_path.display().to_string()],
        );
        value["config"]["advanced"]["sampleEvery"] = json!(1);
        value["config"]["advanced"]["sampleSteps"] = json!(4);
        value["config"]["advanced"]["sampleGuidanceScale"] = json!(1.0);
        value["config"]["advanced"]["sampleCount"] = json!(2);
        value["config"]["advanced"]["samplePrompts"] = json!(pool);
        value["config"]["advanced"]["gradientCheckpointing"] = json!(true);
        let mut config = map_training_config(&parse(value).config);
        assert_eq!(
            config.sample_prompts,
            vec!["alpha portrait".to_owned(), "beta full body".to_owned()],
            "sampleCount=2 must cap the 3-prompt pool to the first 2 prompts"
        );
        // Speed overrides for the smoke — keep the mapped sample_* fields (the thing under test).
        // steps=2 (not 1): the engine skips sampling on the final step, so a single step never previews.
        config.steps = 2;
        config.rank = 4;
        config.alpha = 4.0;
        config.batch_size = 1;
        config.resolution = 512;
        config.train_dtype = "bf16".to_owned();

        let stem = "zimage_sample_cap";
        // Capture the preview params before `config` moves into the request.
        let preview_steps = config.sample_steps;
        let preview_guidance = config.sample_guidance_scale;
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
                control_image_path: None,
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: format!("{stem}.safetensors"),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot.to_path_buf())),
        )
        .expect("candle z_image trainer loads");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");

        // Persist each preview exactly as the worker does, recording (step, prompt) for assertions.
        let mut rendered: Vec<(u32, u32, String)> = Vec::new();
        trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Sample {
                    step,
                    index,
                    prompt,
                    image,
                    ..
                } = progress
                {
                    write_training_sample(
                        Some(&output_dir),
                        None,
                        stem,
                        step,
                        index,
                        &prompt,
                        image,
                        preview_steps,
                        preview_guidance,
                    )
                    .expect("preview PNG persists");
                    rendered.push((step, index, prompt));
                }
            })
            .expect("training + sampling runs");

        eprintln!("[sc-8671 GPU-val] rendered previews: {rendered:?}");
        assert!(
            !rendered.is_empty(),
            "at least one sample step must render (sampleEvery=1)"
        );
        // Per sample step: exactly 2 previews (== sampleCount), never 3 — and only the first 2 prompts.
        use std::collections::BTreeMap;
        let mut per_step: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for (step, _index, prompt) in &rendered {
            per_step.entry(*step).or_default().push(prompt.clone());
        }
        for (step, prompts) in &per_step {
            assert_eq!(
                prompts.len(),
                2,
                "step {step} must render exactly sampleCount=2 previews, got {prompts:?}"
            );
            assert!(
                prompts.contains(&"alpha portrait".to_owned())
                    && prompts.contains(&"beta full body".to_owned()),
                "step {step} must use the first 2 pool prompts, got {prompts:?}"
            );
            assert!(
                !prompts.contains(&"gamma outdoors".to_owned()),
                "the 3rd pool prompt must be capped out at step {step}"
            );
        }
        // Each rendered preview must exist on disk in the worker's layout. The engine numbers samples
        // 1-based within a step, so check the indices actually emitted rather than assuming 0-based.
        for (step, index, _prompt) in &rendered {
            let png = output_dir
                .join("samples")
                .join(format!("step-{step:06}"))
                .join(format!("{stem}-step{step:06}-{index}.png"));
            assert!(png.exists(), "preview PNG missing: {}", png.display());
        }
    }

    /// sc-7817 — the candle Wan A14B **T2V** training smoke. Loads `load_trainer("wan2_2_t2v_14b", …)`
    /// from the installed `Wan-AI/Wan2.2-T2V-A14B-Diffusers` snapshot (`tokenizer/ text_encoder/
    /// transformer/ transformer_2/ vae/`) and trains two LoRA micro-steps — the MoE alternates one
    /// high-noise expert step + one low-noise. Gradient checkpointing is REQUIRED: the two 14B experts
    /// are the working set (~56 GB bf16), so a dense backward materializes ~another model of
    /// frozen-weight grads on candle and OOMs (the Wan-14B finding). Run on demand (one smoke per GPU):
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored candle_wan_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real Wan2.2-T2V-A14B weights + a CUDA device; run one smoke per GPU (two 14B experts)"]
    fn candle_wan_real_weights_trains_a_lora_step() {
        let snapshot =
            candle_hf_snapshot("models--Wan-AI--Wan2.2-T2V-A14B-Diffusers", "transformer_2");
        candle_real_weights_smoke(
            "wan2_2_t2v_14b",
            &snapshot,
            256,
            true,
            "candle_wan_smoke.safetensors",
        );
    }

    /// sc-7817 — the candle Lens training smoke. Loads `load_trainer("lens", …)` from the installed
    /// `SceneWorks/Lens` snapshot (`tokenizer/ text_encoder/ transformer/ vae/` — the training-base
    /// diffusers rehost; the original `microsoft/Lens` is dead, sc-8797) and trains two LoRA
    /// micro-steps at resolution 64 (the gpt-oss-20b encoder load dominates; the per-step DiT graph
    /// stays tiny). Checkpointing on for the large 48-layer MMDiT. Run on demand (one smoke per GPU):
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored candle_lens_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights + a CUDA device; loads the gpt-oss-20b encoder"]
    fn candle_lens_real_weights_trains_a_lora_step() {
        let snapshot = candle_hf_snapshot("models--SceneWorks--Lens", "transformer");
        candle_real_weights_smoke("lens", &snapshot, 64, true, "candle_lens_smoke.safetensors");
    }

    /// sc-8614 — the candle Krea 2 training smoke. Loads `load_trainer("krea_2_raw", …)` from the
    /// installed `krea/Krea-2-Raw` diffusers snapshot (`transformer/ text_encoder/ vae/`) and trains
    /// two LoRA micro-steps at 256² with **gradient checkpointing ON** — REQUIRED on candle: the Raw
    /// 12B DiT's dense backward OOMs because candle materializes a grad for the frozen base weight too
    /// (the Z-Image/Wan-14B finding). The adapter records `family: krea_2` and applies at Krea 2 Turbo
    /// inference (family match, no base-model gating). Needs the `krea/Krea-2-Raw` base model
    /// installed (the sibling catalog story sc-8613 ships the off-Mac download). Run on demand on the
    /// RTX box (one smoke per GPU):
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored candle_krea_real_weights --nocapture`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real krea/Krea-2-Raw weights + a CUDA device; run one smoke per GPU (12B DiT)"]
    fn candle_krea_real_weights_trains_a_lora_step() {
        let snapshot = candle_hf_snapshot("models--krea--Krea-2-Raw", "transformer");
        candle_real_weights_smoke(
            "krea_2_raw",
            &snapshot,
            256,
            true,
            "candle_krea_smoke.safetensors",
        );
    }

    /// sc-10163 (epic 10159 B2) — the candle Krea pose-**ControlNet** training smoke. Drives the exact
    /// studio path — `crate::inference_runtime::load_trainer("krea_2_control", …).train(req)` — over real (target, pose
    /// skeleton, caption) triples from the COCO-pose dataset, proving the control trainer is reachable
    /// through the worker's registry, trains a control branch end-to-end on the frozen 12B Krea base
    /// (checkpointing forced on), and emits an overlay whose meta `kind` is derived from `control_type`
    /// (sc-10722). Uses the S0-validated Krea-2-Turbo base. Run on demand on the RTX box (one per GPU):
    /// `CONTROL_DATA=D:\sceneworks-pose-controlnet\spike cargo test -p sceneworks-worker --lib \
    ///    --features backend-candle --release -- --ignored candle_krea_control_real_weights --nocapture`
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "needs real Krea-2-Turbo weights + the COCO-pose dataset + a CUDA device (12B DiT). Set CONTROL_DATA"]
    fn candle_krea_control_real_weights_trains_a_branch_step() {
        let snapshot = candle_hf_snapshot("models--krea--Krea-2-Turbo", "transformer");
        let data = std::path::PathBuf::from(
            std::env::var("CONTROL_DATA").expect("set CONTROL_DATA to the pose dataset dir"),
        );
        let steps = std::env::var("KC_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8u32);
        let max_items = std::env::var("KC_ITEMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32usize);

        // Read the control manifest → (target, control image, caption) triples. The A2 folder-ingest
        // pipeline (sc-10161) emits the canonical `control` key + `kind`; the older COCO-pose spike
        // datasets keyed the condition as `pose`, so accept `control` first and fall back to `pose`.
        let manifest =
            std::fs::read_to_string(data.join("manifest.jsonl")).expect("read manifest.jsonl");
        let items: Vec<TrainingItem> = manifest
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(max_items)
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("manifest row");
                let field = |k: &str| {
                    v.get(k)
                        .and_then(|s| s.as_str())
                        .unwrap_or_else(|| panic!("manifest row missing {k}"))
                        .to_string()
                };
                let control = v
                    .get("control")
                    .or_else(|| v.get("pose"))
                    .and_then(|s| s.as_str())
                    .unwrap_or_else(|| panic!("manifest row missing control/pose"))
                    .to_string();
                TrainingItem::with_control(
                    data.join(field("target")),
                    field("caption"),
                    data.join(control),
                )
            })
            .collect();
        assert!(!items.is_empty(), "no items in {}", data.display());
        eprintln!("krea_control smoke: {} items, {steps} steps", items.len());

        let output_dir = std::env::temp_dir().join("krea-control-smoke");
        std::fs::create_dir_all(&output_dir).unwrap();
        let config = TrainingConfig {
            steps,
            resolution: 512,
            gradient_accumulation: 1,
            gradient_checkpointing: true,
            train_dtype: "bf16".to_owned(),
            control_type: Some("pose".to_owned()),
            save_every: 0,
            seed: 42,
            ..Default::default()
        };
        let request = TrainingRequest {
            items,
            config,
            output_dir: output_dir.clone(),
            file_name: "krea_pose_control.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = crate::inference_runtime::load_trainer(
            "krea_2_control",
            &LoadSpec::new(WeightsSource::Dir(snapshot)),
        )
        .expect("candle krea_2_control trainer loads (candle_gen_krea linked)");
        trainer
            .validate(&request)
            .expect("control trainer accepts the request");

        let mut last_loss = f32::NAN;
        let out = trainer
            .train(&request, &mut |progress| match progress {
                TrainingProgress::Caching { current, total }
                    if current == 1 || current == total =>
                {
                    eprintln!("caching {current}/{total}");
                }
                TrainingProgress::Training { step, total, loss } => {
                    last_loss = loss;
                    eprintln!("step {step}/{total} loss {loss:.5}");
                }
                _ => {}
            })
            .expect("control training runs");

        eprintln!(
            "DONE overlay={} steps={} final_loss={} (last {last_loss:.5})",
            out.adapter_path.display(),
            out.steps,
            out.final_loss
        );
        assert!(out.adapter_path.exists(), "overlay written");
        assert!(out.final_loss.is_finite(), "final loss finite");
        // The overlay meta records the control `kind` derived from control_type (sc-10722).
        let meta = std::fs::read_to_string(out.adapter_path.with_extension("json"))
            .expect("overlay meta sidecar");
        assert!(
            meta.contains("pose_control_branch"),
            "overlay kind must derive from control_type: {meta}"
        );
    }
}
