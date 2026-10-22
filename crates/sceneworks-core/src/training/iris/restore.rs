//! Iris-3B RESTORATION training (epic sc-25678, S10 sc-25688) — the SceneWorks surface of the
//! native `iris_3b_restore` trainers (`mlx-gen-iris` / `candle-gen-iris`): the paper's one-step
//! adversarial recipe (arXiv 2610.09450v1 §5.3, Eq. 6 — `‖x̂ − y‖² + 5·LPIPS + 0.5·L_adv` with a frozen
//! OpenCLIP ConvNeXt-XXL critic trunk), fine-tuning the `upscaler/` export.
//!
//! The engine contract is `gen_core::iris::restoration_train`: the shared `TrainingConfig` plus the
//! `model_options["iris_restore"]` bag (`RestoreOptions`, unknown keys refused) and, per item, an
//! optional `model_options["degraded_image"]` (an aligned degraded input; absent on every item =
//! seeded synthetic second-order degradation of clean images).
//!
//! * [`iris_restore_target`] — defaults = the paper's recipe (`recipe_training_config`: a FULL
//!   fine-tune, AdamW lr 1e-5 / weight decay 0.01 for generator AND critic, constant LR, 1024²
//!   crops, EMA 0.999 scored); LoRA / LoKr (HYPIR's own arm) is an explicit choice, never the default;
//! * [`validate_restore_config`] — the request-only refusals, incl. the step unit: one step is a
//!   generator window plus a critic window, so `steps`, `saveEvery`, `sampleEvery` and
//!   `lrWarmupSteps` are whole multiples of `2 × gradientAccumulation` micro-batches;
//! * [`validate_restore_dataset`] — all aligned pairs or all clean images;
//! * the memory floors, GiB with per-row provenance ([`restore_memory_floor`]).
//!
//! The run reads three TRAINING-ONLY resources (VGG16 + LPIPS lin heads, the OpenCLIP critic trunk),
//! each its own pinned Model Manager install ([`RESTORE_TRAINING_RESOURCE_MODELS`]) — never pulled
//! for restoration inference.

use serde_json::{json, Map, Value};

use super::super::{
    object, ContractNumber, ExtraFields, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlanError, TrainingTarget, TrainingTargetLimitError,
};
use super::floors::{
    IrisMemoryFloor, Peak, CAMPAIGN_CUDA_CARD, CANDLE_RESERVE_GIB, MLX_RESERVE_GIB,
};
use super::{
    boolean, finite_in, invalid, one_of, validate_micro_steps, whole, IRIS_BASE_MODEL_REPO,
    IRIS_FAMILY, IRIS_TRAINING_KERNEL, LOAD_FROM_JOB_KEY, RESUME_FROM_JOB_KEY,
};
use crate::store_util::is_safe_relative_path;

/// The restoration training target id (the engine trainer id and the restoration task install).
pub const IRIS_RESTORE_TARGET_ID: &str = "iris_3b_restore";
/// The catalog model a restoration run trains: the restoration task install (`upscaler/`).
pub const IRIS_RESTORE_BASE_MODEL: &str = "iris_3b_restore";
/// The task a restoration artifact belongs to (the engine's `irisTask` stamp).
pub const IRIS_RESTORATION_TASK: &str = "restoration";
/// The engine options bag key (`gen_core::iris::restoration_train::OPTIONS_KEY`).
pub const RESTORE_OPTIONS_KEY: &str = "iris_restore";
/// `TrainingDatasetItem.extra` key naming an item's aligned degraded input (a dataset-relative path).
/// The worker resolves it into the engine's `model_options["degraded_image"]`.
pub const ITEM_DEGRADED_KEY: &str = "irisDegradedImage";
/// The paper's training square (§5.3, "1024² crops").
pub const RECIPE_RESOLUTION: u32 = 1024;

/// The three training-only resources, as Model Manager catalog ids, in the engine's component order
/// (`lpips_vgg16`, `lpips_lin`, `critic_trunk`). `(catalog id, engine component id)`.
pub const RESTORE_TRAINING_RESOURCE_MODELS: [(&str, &str); 3] = [
    ("iris_restore_train_vgg16", "lpips_vgg16"),
    ("iris_restore_train_lpips_lin", "lpips_lin"),
    ("iris_restore_train_critic_trunk", "critic_trunk"),
];

/// Every engine restoration option a client may set: the engine's 26 `RestoreOptions` keys minus
/// `resume_from` (SceneWorks resolves it from a trusted previous job).
pub const RESTORE_CLIENT_OPTION_KEYS: [&str; 25] = [
    "loss_l2",
    "loss_lpips",
    "loss_gan",
    "gan_real_label",
    "lr_critic",
    "betas",
    "critic_betas",
    "critic_weight_decay",
    "gradient_clip",
    "critic_gradient_clip",
    "ema_enabled",
    "ema_decay",
    "crop",
    "degradation",
    "critic_dtype",
    "init",
    "num_epochs",
    "nan_loss_tolerance",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_data_policy",
    "export_weights",
    "export_dtype",
    "preview_weights",
    "lora_init",
];

/// The restoration objective's micro-step fields (whole `2 × gradientAccumulation` steps).
pub const RESTORE_MICRO_STEP_FIELDS: [(&str, bool); 4] = [
    ("steps", true),
    ("saveEvery", true),
    ("sampleEvery", false),
    ("lrWarmupSteps", false),
];

const UNREAD_ADVANCED_KEYS: [&str; 6] = [
    "samplePrompts",
    "sampleSteps",
    "sampleGuidanceScale",
    "sampleCount",
    "timestepType",
    "timestepBias",
];

// ---- Measured inputs of the restoration full fine-tune rows ----------------------------------------
// The ONLY numbers to edit when a campaign re-measures: whole-process peaks of the paper's full
// recipe (batch 1, bf16, EMA), DECIMAL GB as the probe printed them.
//
// MLX (nax-macos-2; inference PR 1238 @b1c04bd0d, runs 38312190794 and 38318086154): the ACTIVE
// peak with the buffer cache DISABLED, steady by step 2 — production runs keep the cache, so their
// footprint can rise to the allocator's GC ceiling (0.95 × the recommended working set) without
// that being need. At a 1024² crop the run was stopped by its guard at a 116.47 GB footprint with
// the cache off (live need above 115 GB): a LOWER bound, not a peak.
//
// Candle (RTX PRO 6000 card 1, 95.59 GiB; inference run 38393679427 @1449045c8, after the
// training-tape memory fix of inference #1242): the LIVE pool peak over six optimizer steps, steady
// from step 2. A 1024² crop without activation checkpointing needs 142.96 GB and completed only by
// spilling 42.5 GB to system memory — a measured need that does not fit that card; under full
// activation checkpointing (`gradientCheckpointing`, new at that pin) it peaks at 86.47 GB.
const MLX_RESTORE_FULL_512_PEAK: Peak = Peak::DecimalGb(71.57);
const MLX_RESTORE_FULL_1024_LOWER_BOUND: Peak = Peak::DecimalGb(116.47);
// Under full activation checkpointing (`gradientCheckpointing`; MLX since inference #1243):
// inference run 38405154244 on nax-macos-2 — active peak, cache off, six steps, steady. The 1024²
// crop COMPLETES under it (77.25 GB); without it only the lower bound above is known.
const MLX_RESTORE_FULL_AC_512_PEAK: Peak = Peak::DecimalGb(63.21);
const MLX_RESTORE_FULL_AC_1024_PEAK: Peak = Peak::DecimalGb(77.25);
const CANDLE_RESTORE_FULL_256_PEAK: Peak = Peak::DecimalGb(76.15);
const CANDLE_RESTORE_FULL_512_PEAK: Peak = Peak::DecimalGb(87.36);
const CANDLE_RESTORE_FULL_1024_SPILL_PEAK: Peak = Peak::DecimalGb(142.96);
const CANDLE_RESTORE_FULL_AC_1024_PEAK: Peak = Peak::DecimalGb(86.47);

const MLX_RESTORE_FULL_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_RESTORE_FULL_512_PEAK, MLX_RESERVE_GIB);
const MLX_RESTORE_FULL_256: IrisMemoryFloor =
    IrisMemoryFloor::declared(92).bounded_by(MLX_RESTORE_FULL_512);
const MLX_RESTORE_FULL_AC_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_RESTORE_FULL_AC_512_PEAK, MLX_RESERVE_GIB);
const MLX_RESTORE_FULL_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_RESTORE_FULL_AC_1024_PEAK, MLX_RESERVE_GIB);
const CANDLE_RESTORE_FULL_256: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_RESTORE_FULL_256_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
const CANDLE_RESTORE_FULL_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_RESTORE_FULL_512_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
const CANDLE_RESTORE_FULL_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_RESTORE_FULL_AC_1024_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());

/// Memory floors, **GiB** ([`super::floors`]), per `(lane, artifact, micro-batch pixels ≤, floor)`
/// with a micro-batch's pixels = `batchSize × resolution²`; each row carries its own provenance.
/// `full_ac` is the FULL fine-tune under full activation checkpointing
/// (`advanced.gradientCheckpointing`), on the lanes whose trainer implements it
/// ([`super::iris_gradient_checkpointing_lanes`]).
///
/// * MEASURED — the full fine-tune at 262 144 px on both lanes and at 65 536 px on Candle;
///   `full_ac` at 1 048 576 px on both lanes and at 262 144 px on MLX: the `*_PEAK` constants above
///   converted to GiB, plus the
///   lane's 2 GiB reserve, rounded up; a Candle row that ran without spilling never exceeds the
///   card that held its run.
/// * MEASURED WITH SPILL — Candle's full fine-tune at 1 048 576 px without checkpointing: the
///   measured need, which the card met only by spilling to system memory; the row refuses that
///   card.
/// * LOWER BOUND — MLX's full fine-tune at 1 048 576 px: declared above the measured lower bound.
/// * EXTRAPOLATED — the MLX full row at 65 536 px, bounded by the measured 262 144 px row (a
///   smaller crop never needs more; its earlier 92 was declared from an 87.5 GB process footprint,
///   cache included); and the `full_ac` rows with no run of their own, each the lower of the plain
///   row at that size and the lane's next measured AC row up.
/// * DECLARED — every adapter row: LoRA rank 8 at a 256² crop peaked at 25.2 GB on MLX (run
///   38140601912) and 29.3 GB of VRAM on CUDA (run 38137202298), padded (Candle adding the
///   reserve); adapter rows above 65 536 px scale the activation, LPIPS and critic terms
///   conservatively.
///
/// A micro-batch above the last row is refused rather than admitted unbounded.
const RESTORE_FLOORS_GIB: [(&str, &str, u64, IrisMemoryFloor); 18] = [
    ("mlx", "full", 65_536, MLX_RESTORE_FULL_256),
    ("mlx", "full", 262_144, MLX_RESTORE_FULL_512),
    (
        "mlx",
        "full",
        1_048_576,
        IrisMemoryFloor::above_lower_bound(MLX_RESTORE_FULL_1024_LOWER_BOUND, 128),
    ),
    (
        "mlx",
        "full_ac",
        65_536,
        IrisMemoryFloor::bound_from(MLX_RESTORE_FULL_256).bounded_by(MLX_RESTORE_FULL_AC_512),
    ),
    ("mlx", "full_ac", 262_144, MLX_RESTORE_FULL_AC_512),
    ("mlx", "full_ac", 1_048_576, MLX_RESTORE_FULL_AC_1024),
    ("mlx", "adapter", 65_536, IrisMemoryFloor::declared(28)),
    ("mlx", "adapter", 262_144, IrisMemoryFloor::declared(36)),
    ("mlx", "adapter", 1_048_576, IrisMemoryFloor::declared(56)),
    ("candle", "full", 65_536, CANDLE_RESTORE_FULL_256),
    ("candle", "full", 262_144, CANDLE_RESTORE_FULL_512),
    (
        "candle",
        "full",
        1_048_576,
        IrisMemoryFloor::measured_with_spill(
            CANDLE_RESTORE_FULL_1024_SPILL_PEAK,
            CANDLE_RESERVE_GIB,
            CAMPAIGN_CUDA_CARD.gib(),
        ),
    ),
    (
        "candle",
        "full_ac",
        65_536,
        IrisMemoryFloor::bound_from(CANDLE_RESTORE_FULL_256)
            .bounded_by(CANDLE_RESTORE_FULL_AC_1024),
    ),
    (
        "candle",
        "full_ac",
        262_144,
        IrisMemoryFloor::bound_from(CANDLE_RESTORE_FULL_512)
            .bounded_by(CANDLE_RESTORE_FULL_AC_1024),
    ),
    ("candle", "full_ac", 1_048_576, CANDLE_RESTORE_FULL_AC_1024),
    ("candle", "adapter", 65_536, IrisMemoryFloor::declared(32)),
    ("candle", "adapter", 262_144, IrisMemoryFloor::declared(40)),
    (
        "candle",
        "adapter",
        1_048_576,
        IrisMemoryFloor::declared(60),
    ),
];

fn network_type(config: &TrainingConfig) -> String {
    config
        .advanced
        .get("networkType")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("full")
        .to_ascii_lowercase()
}

/// The floor row (GiB + provenance) of a restoration run on `lane`, or `None` above the table / for
/// an unknown lane (callers refuse). A FULL run under activation checkpointing reads the lane's
/// `full_ac` rows where it has them.
pub fn restore_memory_floor(lane: &str, config: &TrainingConfig) -> Option<IrisMemoryFloor> {
    let full = network_type(config) == "full";
    let side = u64::from(config.resolution);
    let pixels = u64::from(config.batch_size.max(1)) * side * side;
    let lookup = |artifact: &str| {
        RESTORE_FLOORS_GIB
            .iter()
            .filter(|(row_lane, row_artifact, row_pixels, _)| {
                *row_lane == lane && *row_artifact == artifact && *row_pixels >= pixels
            })
            .min_by_key(|(_, _, row_pixels, _)| *row_pixels)
            .map(|(_, _, _, floor)| *floor)
    };
    if !full {
        return lookup("adapter");
    }
    super::iris_gradient_checkpointing(config)
        .then(|| lookup("full_ac"))
        .flatten()
        .or_else(|| lookup("full"))
}

fn memory_floors_json() -> Value {
    super::floors::memory_floors_json(
        Some("microBatchPixels"),
        "Keyed by artifact and the pixels of one micro-batch (batch size × crop²). Measured rows \
         are the whole-process peak of the paper's full recipe at batch 1, converted to GiB, plus \
         a 2 GiB reserve; `full_ac` is the full fine-tune under gradient (activation) \
         checkpointing on the lanes that implement it. MLX peaks are the ACTIVE peak with MLX's \
         buffer cache disabled; production runs keep the cache, so a process footprint can rise \
         to the allocator's GC ceiling (0.95 × the recommended working set) without that being \
         need. Candle peaks are the live pool peak over six optimizer steps (steady from step 2) on a 95.59 \
         GiB card; a row marked measured_spill is a need the run met only by spilling to system \
         memory, so it does not fit that card.",
        RESTORE_FLOORS_GIB
            .iter()
            .map(|(lane, artifact, pixels, floor)| {
                (
                    *lane,
                    vec![(*artifact).to_owned(), pixels.to_string()],
                    *floor,
                )
            }),
    )
}

/// The Iris-3B restoration training target.
pub fn iris_restore_target() -> TrainingTarget {
    let mut extra = ExtraFields::new();
    extra.insert("memoryFloors".to_owned(), memory_floors_json());
    extra.insert("irisObjective".to_owned(), json!(IRIS_RESTORATION_TASK));
    extra.insert(
        "trainingResources".to_owned(),
        json!(RESTORE_TRAINING_RESOURCE_MODELS
            .iter()
            .map(|(model, component)| json!({ "modelId": model, "component": component }))
            .collect::<Vec<_>>()),
    );
    TrainingTarget {
        id: IRIS_RESTORE_TARGET_ID.to_owned(),
        name: "Iris 3B Restoration".to_owned(),
        modality: TrainingModality::Image,
        // Every arm exports the full `upscaler/` layout (an adapter run is merged at export), so the
        // artifact is always a restoration MODEL, never a stackable adapter.
        output_kind: TrainingOutputKind::BaseCheckpoint,
        family: IRIS_FAMILY.to_owned(),
        base_model: IRIS_RESTORE_BASE_MODEL.to_owned(),
        base_model_repo: Some(IRIS_BASE_MODEL_REPO.to_owned()),
        kernel: IRIS_TRAINING_KERNEL.to_owned(),
        defaults: TrainingConfig {
            // Read only by the LoRA / LoKr arm.
            rank: 16,
            alpha: ContractNumber::from(16),
            learning_rate: ContractNumber::from_f64(1e-5).expect("1e-5 is finite"),
            // MICRO-batches: one step = a generator window + a critic window (2 × accumulation).
            steps: 10_000,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: RECIPE_RESOLUTION,
            save_every: 1_000,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                // The paper's recipe is a FULL fine-tune; LoRA/LoKr is an explicit choice.
                "networkType": "full",
                "decomposeFactor": -1,
                "lossType": "mse",
                "weightDecay": 0.01,
                "lrScheduler": "constant",
                "lrWarmupSteps": 0,
                "gradientCheckpointing": false,
                "sampleEvery": 1_000,
                "outputScope": "project",
                "requestedGpu": "auto",
                // The engine `model_options["iris_restore"]` bag: the recipe defaults (Eq. 6 weights,
                // HYPIR's betas / clipping / EMA, the paper's random 1024² crops) except
                // `keep_last_checkpoints` (a full checkpoint is ~60 GB with the critic).
                "iris_restore": {
                    "loss_l2": 1.0,
                    "loss_lpips": 5.0,
                    "loss_gan": 0.5,
                    "gan_real_label": 0.8,
                    "betas": [0.9, 0.999],
                    "gradient_clip": 1.0,
                    "critic_gradient_clip": 1.0,
                    "ema_enabled": true,
                    "ema_decay": 0.999,
                    "crop": "random",
                    "critic_dtype": "bf16",
                    "init": "export",
                    "nan_loss_tolerance": 20,
                    "keep_last_checkpoints": 2,
                    "milestone_steps": [],
                    "export_dtype": "fp32"
                }
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [1, 128],
            "alpha": [1, 128],
            "steps": [2, 2_000_000],
            "batchSize": [1, 16],
            // Multiples of the backbone's 16-pixel patch; the paper trains at 1024.
            "resolutions": [256, 512, 1024],
            "optimizers": ["adamw"],
            "networkTypes": ["full", "lora", "lokr"],
            "lrSchedulers": ["constant", "cosine"],
            "lossTypes": ["mse"],
            "mixedPrecisions": ["bf16", "f32"],
            // Activation checkpointing through the shared toggle, on the lanes whose restoration
            // trainer implements it (never pinned off: a lane that lacks it refuses typed before load).
            "gradientCheckpointingLanes":
                super::iris_gradient_checkpointing_lanes(super::IrisObjective::Restoration),
            "microStepFields": ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
            "microStepMultiple": "2xGradientAccumulation",
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Iris 3B Restoration",
            "description": "Fine-tune Iris-3B restoration with the paper's adversarial recipe (pixel + \
                            LPIPS + ConvNeXt-critic GAN loss): a full fine-tune by default, or a LoRA / \
                            LoKr merged into the export. Train on clean images (reproducible synthetic \
                            degradation) or on aligned degraded/clean pairs. Writes a restoration model \
                            you can pick wherever Iris restores. Apple Silicon (MLX) and NVIDIA \
                            (Candle/CUDA).",
            "recommendedFor": ["image_upscale"],
            "datasetModality": "image",
            "objective": IRIS_RESTORATION_TASK
        })),
        extra,
    }
}

/// The restoration options bag of a config (`advanced.iris_restore`), or an empty map.
pub fn restore_options(config: &TrainingConfig) -> Result<Map<String, Value>, TrainingPlanError> {
    match config.advanced.get(RESTORE_OPTIONS_KEY) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err(invalid(
            "advanced.iris_restore must be an object of Iris restoration training options.",
        )),
    }
}

fn pair_in_unit(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| {
            invalid(format!(
                "advanced.iris_restore.{key} must be a pair [b1, b2]."
            ))
        })?;
    if pair.iter().all(|beta| {
        beta.as_f64()
            .is_some_and(|beta| beta.is_finite() && (0.0..1.0).contains(&beta))
    }) {
        Ok(())
    } else {
        Err(invalid(format!(
            "advanced.iris_restore.{key} must both be in [0, 1)."
        )))
    }
}

fn validate_option(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    let at_least_zero = |key: &str| {
        finite_in(key, value, |x| x >= 0.0, "at least 0")
            .map(drop)
            .map_err(|error| rename(error, key))
    };
    match key {
        "loss_l2" | "loss_lpips" | "loss_gan" | "lr_critic" | "critic_weight_decay" => {
            at_least_zero(key)
        }
        "gan_real_label" | "ema_decay" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1")
                .map(drop)
                .map_err(|error| rename(error, key))
        }
        "gradient_clip" | "critic_gradient_clip" => {
            finite_in(key, value, |x| x > 0.0, "greater than 0")
                .map(drop)
                .map_err(|error| rename(error, key))
        }
        "betas" | "critic_betas" => pair_in_unit(key, value),
        "ema_enabled" => boolean(key, value).map(drop).map_err(|e| rename(e, key)),
        "crop" => one_of(key, value, &["random", "center", "none"]).map_err(|e| rename(e, key)),
        "critic_dtype" => one_of(key, value, &["bf16", "fp32"]).map_err(|e| rename(e, key)),
        "init" => one_of(key, value, &["export", "parent"]).map_err(|e| rename(e, key)),
        "lora_init" => one_of(key, value, &["gaussian", "kaiming"]).map_err(|e| rename(e, key)),
        "resume_data_policy" => {
            one_of(key, value, &["exact", "new_phase"]).map_err(|e| rename(e, key))
        }
        "export_weights" | "preview_weights" => {
            one_of(key, value, &["ema", "raw"]).map_err(|e| rename(e, key))
        }
        "export_dtype" => one_of(key, value, &["fp32", "bf16"]).map_err(|e| rename(e, key)),
        "num_epochs" => whole(key, value, 1).map(drop).map_err(|e| rename(e, key)),
        "nan_loss_tolerance" | "keep_last_checkpoints" => {
            whole(key, value, 0).map(drop).map_err(|e| rename(e, key))
        }
        "milestone_steps" => value
            .as_array()
            .filter(|steps| {
                steps
                    .iter()
                    .all(|step| step.as_u64().is_some_and(|s| s > 0))
            })
            .map(drop)
            .ok_or_else(|| {
                invalid("advanced.iris_restore.milestone_steps must be a list of positive steps.")
            }),
        // Real-ESRGAN second-order degradation keys (HYPIR `sd2_train.yaml` names). The engine's own
        // parser judges the keys and ranges (the API runs it at submit); here only the shape.
        "degradation" => value.as_object().map(drop).ok_or_else(|| {
            invalid("advanced.iris_restore.degradation must be an object of degradation keys.")
        }),
        _ => Err(invalid(format!(
            "advanced.iris_restore.{key} is not an Iris restoration training option."
        ))),
    }
}

/// The shared helpers name the `iris` bag; restate their messages for `iris_restore`.
fn rename(error: TrainingPlanError, key: &str) -> TrainingPlanError {
    match error {
        TrainingPlanError::InvalidConfig(message) => {
            TrainingPlanError::InvalidConfig(message.replace(
                &format!("advanced.iris.{key}"),
                &format!("advanced.iris_restore.{key}"),
            ))
        }
        TrainingPlanError::TargetLimit(TrainingTargetLimitError::UnsupportedValue {
            value,
            allowed,
            ..
        }) => TrainingPlanError::TargetLimit(TrainingTargetLimitError::UnsupportedValue {
            field: format!("iris_restore.{key}"),
            value,
            allowed,
        }),
        other => other,
    }
}

/// Submit-time validation of a restoration config.
pub fn validate_restore_config(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let per_step = 2 * u64::from(config.gradient_accumulation.max(1));
    validate_micro_steps(config, per_step, &RESTORE_MICRO_STEP_FIELDS)?;
    if config
        .trigger_word
        .as_deref()
        .is_some_and(|word| !word.trim().is_empty())
    {
        return Err(invalid(
            "The restorer takes no prompt, so a restoration run takes no trigger word.",
        ));
    }
    for key in UNREAD_ADVANCED_KEYS {
        let present = match config.advanced.get(key) {
            None | Some(Value::Null) => false,
            Some(Value::Array(values)) => !values.is_empty(),
            Some(Value::String(text)) => !text.trim().is_empty(),
            Some(_) => true,
        };
        if present {
            return Err(invalid(format!(
                "advanced.{key} is not a restoration training control (the restorer is one-step at \
                 its export's fixed sigma, and its previews restore the first dataset item)."
            )));
        }
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY) {
        return Err(invalid(
            "A restoration run starts from the restoration export (or the generation backbone with \
             init: parent); continue a previous run with resumeFromJobId (new_phase for a fresh \
             phase) instead of loadFromJobId.",
        ));
    }
    let options = restore_options(config)?;
    for (key, value) in &options {
        if key == "resume_from" {
            return Err(invalid(format!(
                "advanced.iris_restore.resume_from is set by SceneWorks from a previous run; name \
                 the run with advanced.{RESUME_FROM_JOB_KEY} instead."
            )));
        }
        if !RESTORE_CLIENT_OPTION_KEYS.contains(&key.as_str()) {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: RESTORE_OPTIONS_KEY.to_owned(),
                    value: key.clone(),
                    allowed: RESTORE_CLIENT_OPTION_KEYS
                        .iter()
                        .map(|key| (*key).to_owned())
                        .collect(),
                },
            ));
        }
        if !value.is_null() {
            validate_option(key, value)?;
        }
    }
    let full = network_type(config) == "full";
    if full
        && options
            .get("lora_init")
            .is_some_and(|value| !value.is_null())
    {
        return Err(invalid(
            "advanced.iris_restore.lora_init initializes a LoRA / LoKr adapter; a full fine-tune \
             has none.",
        ));
    }
    let ema_enabled = options
        .get("ema_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if !ema_enabled {
        if options
            .get("ema_decay")
            .is_some_and(|value| !value.is_null())
        {
            return Err(invalid(
                "advanced.iris_restore.ema_decay needs the EMA enabled (ema_enabled).",
            ));
        }
        for key in ["export_weights", "preview_weights"] {
            if options.get(key).and_then(Value::as_str) == Some("ema") {
                return Err(invalid(format!(
                    "advanced.iris_restore.{key} = ema needs the EMA enabled (ema_enabled)."
                )));
            }
        }
    }
    if !config.advanced.contains_key(RESUME_FROM_JOB_KEY)
        && options
            .get("resume_data_policy")
            .is_some_and(|value| !value.is_null())
    {
        return Err(invalid(format!(
            "advanced.iris_restore.resume_data_policy applies only when resuming a previous run \
             (advanced.{RESUME_FROM_JOB_KEY})."
        )));
    }
    Ok(())
}

/// One item's aligned degraded input (`extra.irisDegradedImage`), validated.
pub fn validate_item_degraded(extra: &ExtraFields) -> Result<(), String> {
    match extra.get(ITEM_DEGRADED_KEY) {
        None => Ok(()),
        Some(Value::String(path)) if is_safe_relative_path(path) => Ok(()),
        Some(_) => {
            Err("Item irisDegradedImage must be a path relative to the dataset root.".to_owned())
        }
    }
}

/// A run is either all aligned degraded/clean pairs or all clean images (synthetic degradation);
/// aligned pairs are never synthetically degraded, so only `use_hflip` applies to them.
pub fn validate_restore_dataset(
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_degraded(&item.extra).map_err(invalid)?;
    }
    let paired = dataset
        .items
        .iter()
        .filter(|item| item.extra.contains_key(ITEM_DEGRADED_KEY))
        .count();
    if paired != 0 && paired != dataset.items.len() {
        return Err(invalid(format!(
            "{paired} of {} dataset items carry an aligned degraded image and the rest do not — a \
             restoration run is either all aligned pairs or all clean images (synthetic \
             degradation). Attach the missing degraded images or remove them all.",
            dataset.items.len()
        )));
    }
    if paired != 0 {
        let options = restore_options(config)?;
        if let Some(Value::Object(degradation)) = options.get("degradation") {
            if let Some(key) = degradation.keys().find(|key| *key != "use_hflip") {
                return Err(invalid(format!(
                    "advanced.iris_restore.degradation.{key} configures SYNTHETIC degradation; an \
                     aligned-pair dataset is not degraded (only use_hflip applies)."
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TrainingConfig {
        iris_restore_target().defaults
    }

    #[test]
    fn the_defaults_are_the_papers_full_fine_tune_recipe() {
        let defaults = config();
        assert_eq!(defaults.advanced["networkType"], "full");
        assert_eq!(defaults.learning_rate.as_f64(), Some(1e-5));
        assert_eq!(defaults.advanced["weightDecay"], json!(0.01));
        assert_eq!(defaults.advanced["lrScheduler"], "constant");
        assert_eq!(defaults.advanced["lrWarmupSteps"], 0);
        assert_eq!(defaults.resolution, 1024);
        assert_eq!(defaults.optimizer, "adamw");
        assert_eq!(defaults.advanced["iris_restore"]["ema_enabled"], true);
        validate_restore_config(&defaults).expect("the recipe validates");
        // LoRA is offered, not the default.
        let target = iris_restore_target();
        assert!(target.limits["networkTypes"]
            .as_array()
            .unwrap()
            .contains(&json!("lora")));
    }

    #[test]
    fn the_step_unit_is_a_generator_plus_a_critic_window() {
        let mut config = config();
        config.gradient_accumulation = 2;
        // 4 = 2 × 2: whole steps.
        config.steps = 4000;
        config.save_every = 400;
        config.advanced.insert("sampleEvery".into(), json!(400));
        validate_restore_config(&config).expect("whole steps");
        config.save_every = 402;
        let error = validate_restore_config(&config).expect_err("402 is not a multiple of 4");
        assert!(matches!(
            error,
            TrainingPlanError::TargetLimit(TrainingTargetLimitError::NotMultipleOf {
                multiple: 4,
                ..
            })
        ));
        let mut odd = iris_restore_target().defaults;
        odd.steps = 9_999;
        assert!(
            validate_restore_config(&odd).is_err(),
            "accumulation 1 ⇒ even"
        );
    }

    #[test]
    fn request_only_refusals() {
        let refused = |mutate: &dyn Fn(&mut TrainingConfig)| {
            let mut config = config();
            mutate(&mut config);
            validate_restore_config(&config).expect_err("refused")
        };
        refused(&|config| config.advanced["iris_restore"]["prediction"] = json!("v"));
        refused(&|config| config.advanced["iris_restore"]["resume_from"] = json!("/x"));
        refused(&|config| config.advanced["iris_restore"]["crop"] = json!("tile"));
        refused(&|config| config.advanced["iris_restore"]["lora_init"] = json!("gaussian"));
        refused(&|config| {
            config.advanced["iris_restore"]["ema_enabled"] = json!(false);
            config.advanced["iris_restore"]["export_weights"] = json!("ema");
        });
        refused(&|config| {
            config.advanced.insert("samplePrompts".into(), json!(["x"]));
        });
        refused(&|config| {
            config
                .advanced
                .insert("loadFromJobId".into(), json!("job_1"));
        });
        refused(&|config| config.trigger_word = Some("x".into()));
        refused(&|config| config.advanced["iris_restore"]["resume_data_policy"] = json!("exact"));
        let error =
            refused(&|config| config.advanced["iris_restore"]["critic_dtype"] = json!("f16"));
        assert!(
            error.to_string().contains("iris_restore.critic_dtype"),
            "{error}"
        );
        // The LoRA arm with its init.
        let mut lora = config();
        lora.advanced.insert("networkType".into(), json!("lora"));
        lora.advanced["iris_restore"]["lora_init"] = json!("kaiming");
        validate_restore_config(&lora).expect("HYPIR's LoRA arm");
    }

    #[test]
    fn floors_distinguish_the_full_recipe_from_an_adapter() {
        let row = |lane: &str, artifact: &str, pixels: u64| {
            RESTORE_FLOORS_GIB
                .iter()
                .find(|(row_lane, row_artifact, row_pixels, _)| {
                    *row_lane == lane && *row_artifact == artifact && *row_pixels == pixels
                })
                .map(|(_, _, _, floor)| *floor)
        };
        let mut config = config();
        // The recipe (full, 1024², batch 1) is the 1 048 576-pixel row.
        assert_eq!(
            restore_memory_floor("mlx", &config),
            row("mlx", "full", 1_048_576)
        );
        config.resolution = 256;
        for lane in ["mlx", "candle"] {
            assert_eq!(
                restore_memory_floor(lane, &config),
                row(lane, "full", 65_536)
            );
        }
        let full = restore_memory_floor("mlx", &config).unwrap();
        config.advanced.insert("networkType".into(), json!("lora"));
        for lane in ["mlx", "candle"] {
            assert_eq!(
                restore_memory_floor(lane, &config),
                row(lane, "adapter", 65_536)
            );
        }
        assert!(
            restore_memory_floor("mlx", &config).unwrap().gib < full.gib,
            "an adapter needs far less than the full recipe"
        );
        config.resolution = 1024;
        config.batch_size = 2;
        assert_eq!(
            restore_memory_floor("mlx", &config),
            None,
            "beyond the table"
        );
    }

    /// sc-25690: provenance is per row. The full rows measured at their own shape say measured; the
    /// 1024² full rows — whose runs were stopped before their peak — say lower bound and sit above
    /// it; the adapter rows say declared. The published table carries the same, with no
    /// table-wide `measured` flag, and floors only grow with the micro-batch.
    #[test]
    fn floor_rows_publish_their_own_provenance() {
        use super::super::floors::{ceil_gib, FloorProvenance};
        let published = &iris_restore_target().extra["memoryFloors"];
        assert_eq!(published["unit"], json!("GiB"));
        assert!(published.get("measured").is_none());
        for (lane, artifact, pixels, floor) in RESTORE_FLOORS_GIB {
            let expected = match (lane, artifact, pixels) {
                (_, "adapter", _) => FloorProvenance::Declared,
                ("mlx", "full", 65_536) => FloorProvenance::Extrapolated,
                ("mlx", "full", 1_048_576) => FloorProvenance::LowerBound,
                ("candle", "full", 1_048_576) => FloorProvenance::MeasuredSpill,
                (_, "full_ac", 65_536) | ("candle", "full_ac", 262_144) => {
                    FloorProvenance::Extrapolated
                }
                _ => FloorProvenance::Measured,
            };
            assert_eq!(floor.provenance, expected, "{lane} {artifact} {pixels}");
            assert_eq!(
                published["provenance"][lane][artifact][pixels.to_string()],
                json!(expected.as_str())
            );
            assert_eq!(
                published["lanes"][lane][artifact][pixels.to_string()],
                json!(floor.gib)
            );
        }
        for (lane, bound) in [
            ("mlx", MLX_RESTORE_FULL_1024_LOWER_BOUND),
            // Candle's is a measured need, not a bound: the row sits above the card it spilled off.
            ("candle", CAMPAIGN_CUDA_CARD),
        ] {
            let floor = RESTORE_FLOORS_GIB
                .iter()
                .find(|row| row.0 == lane && row.1 == "full" && row.2 == 1_048_576)
                .unwrap()
                .3;
            assert!(floor.gib > ceil_gib(bound.gib()), "{lane}: {floor:?}");
        }
        for lane in ["mlx", "candle"] {
            for artifact in ["full", "full_ac", "adapter"] {
                let rows: Vec<_> = RESTORE_FLOORS_GIB
                    .iter()
                    .filter(|row| row.0 == lane && row.1 == artifact)
                    .collect();
                assert!(
                    rows.windows(2)
                        .all(|pair| pair[0].2 < pair[1].2 && pair[0].3.gib <= pair[1].3.gib),
                    "{lane} {artifact}: {rows:?}"
                );
            }
        }
        let basis = published["basis"].as_str().unwrap();
        assert!(!basis.contains("no measurement campaign"), "{basis}");
        assert!(
            basis.contains("Declared above a measured lower bound: mlx full 1048576"),
            "{basis}"
        );
    }

    /// sc-25690: a FULL run under gradient (activation) checkpointing reads the lane's `full_ac`
    /// rows — Candle's 1024² crop, which does not fit a 96 GB card without it, fits with it — an
    /// adapter run never does, and a lane with no such rows keeps reading the plain ones.
    #[test]
    fn a_checkpointed_full_run_reads_the_full_ac_rows_where_the_lane_has_them() {
        use super::super::floors::FloorProvenance;
        let mut config = config();
        let card = CAMPAIGN_CUDA_CARD.gib();
        let plain = restore_memory_floor("candle", &config).unwrap();
        let mlx_plain = restore_memory_floor("mlx", &config).unwrap();
        assert_eq!(plain.provenance, FloorProvenance::MeasuredSpill);
        assert!(f64::from(plain.gib) > card, "{plain:?}");
        config
            .advanced
            .insert("gradientCheckpointing".into(), json!(true));
        let checkpointed = restore_memory_floor("candle", &config).unwrap();
        assert_eq!(checkpointed, CANDLE_RESTORE_FULL_AC_1024);
        assert!(f64::from(checkpointed.gib) <= card, "{checkpointed:?}");
        // MLX (inference #1243): the 1024² crop, only a lower bound without checkpointing, has a
        // measured row with it — and that row fits a 128 GiB Mac after the OS reserve.
        assert_eq!(mlx_plain.provenance, FloorProvenance::LowerBound);
        assert_eq!(
            restore_memory_floor("mlx", &config),
            Some(MLX_RESTORE_FULL_AC_1024)
        );
        let checkpointed_mlx = restore_memory_floor("mlx", &config).unwrap();
        assert!(checkpointed_mlx.gib + 4 <= 128, "{checkpointed_mlx:?}");
        assert_eq!(restore_memory_floor("cpu", &config), None);
        config.advanced.insert("networkType".into(), json!("lora"));
        assert_eq!(
            restore_memory_floor("candle", &config).map(|floor| floor.provenance),
            Some(FloorProvenance::Declared),
            "an adapter run reads the adapter rows whatever the toggle says"
        );
        for (lane, artifact, pixels, floor) in RESTORE_FLOORS_GIB {
            if artifact == "full_ac" {
                let plain = RESTORE_FLOORS_GIB
                    .iter()
                    .find(|row| row.0 == lane && row.1 == "full" && row.2 == pixels)
                    .unwrap()
                    .3;
                assert!(floor.gib <= plain.gib, "{lane} {pixels}");
            }
        }
        let limits = &iris_restore_target().limits;
        assert_eq!(
            limits["gradientCheckpointingLanes"],
            json!(["candle", "mlx"])
        );
        assert!(limits.get("gradientCheckpointing").is_none());
    }

    #[test]
    fn a_run_is_all_pairs_or_all_clean_images() {
        let dataset = |paired: &[bool]| -> TrainingDataset {
            let items: Vec<Value> = paired
                .iter()
                .enumerate()
                .map(|(index, paired)| {
                    let mut item = json!({
                        "id": format!("i{index}"), "path": format!("images/{index}.png"),
                        "displayName": format!("i{index}"),
                        "caption": {"text": "", "source": "manual", "triggerWords": []},
                        "addedAt": "2026-10-11T00:00:00Z"
                    });
                    if *paired {
                        item["irisDegradedImage"] = json!(format!("sidecars/{index}.lq.png"));
                    }
                    item
                })
                .collect();
            serde_json::from_value(json!({
                "schemaVersion": 1, "id": "ds", "version": 1, "projectId": "p", "name": "ds",
                "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
                "updatedAt": "2026-10-11T00:00:00Z", "items": items
            }))
            .unwrap()
        };
        validate_restore_dataset(&config(), &dataset(&[false, false])).unwrap();
        validate_restore_dataset(&config(), &dataset(&[true, true])).unwrap();
        assert!(validate_restore_dataset(&config(), &dataset(&[true, false])).is_err());
        let mut synthetic_knob = config();
        synthetic_knob.advanced["iris_restore"]["degradation"] = json!({"noise_range": [1, 20]});
        assert!(validate_restore_dataset(&synthetic_knob, &dataset(&[true, true])).is_err());
        validate_restore_dataset(&synthetic_knob, &dataset(&[false])).unwrap();
        let mut flip = config();
        flip.advanced["iris_restore"]["degradation"] = json!({"use_hflip": false});
        validate_restore_dataset(&flip, &dataset(&[true])).unwrap();
    }
}
