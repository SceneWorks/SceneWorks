//! Iris-3B DEPTH training (epic sc-25678, S9 sc-25687) — the SceneWorks surface of the native
//! `iris_3b_depth` trainers (`mlx-gen-iris` / `candle-gen-iris`): the paper's supervised
//! direct-regression fine-tune (arXiv 2610.09450v1 §5.2) that produces a `depth/` export.
//!
//! The engine contract is `gen_core::iris::train::depth`: the shared `TrainingConfig` (a FULL
//! fine-tune, `loss_type = "l1"`) plus the `model_options["iris"]` bag whose unknown keys the trainer
//! refuses (`DEPTH_OPTION_KEYS`), and per-item `model_options["depth"] = {path, mask_path?, scale?,
//! min_depth?, max_depth?}`. This module owns the SceneWorks half so the Training Studio offers
//! exactly what the trainer accepts:
//!
//! * [`iris_depth_target`] — the target, defaulting to the paper recipe (AdamW lr 3e-5, weight decay
//!   0.01, constant LR, no warmup, clip 1.0, β (0.9, 0.999), raw-weights export, 10K optimizer steps
//!   at a global batch of 32). The global batch is offered as `batchSize 1 × gradientAccumulation 32`
//!   — the same 32 examples per optimizer step that fits a single machine (every micro-batch
//!   normalizes its own valid pixels; a larger `batchSize` is a submit away);
//! * [`validate_depth_config`] — the submit-time twin of `DepthTrainPlan::resolve`'s request-only
//!   refusals;
//! * [`validate_depth_dataset`] — every item is an aligned example (`extra.irisDepth`), the sidecar
//!   files live under the dataset root;
//! * the memory floors, GiB with per-row provenance ([`depth_memory_floor`]).
//!
//! Where the run starts (`advanced.depthInit`): `export` (the installed `depth/` export — the
//! release's own depth model, fine-tuned further; no text encoder) or `parent` (the paper's start:
//! the GENERATION backbone, input projections widened, conditioned on the empty prompt — read from
//! the installed depth export's `empty_prompt.safetensors` when present, else encoded ONCE by the
//! generation install's text encoder).

use serde_json::{json, Value};

use super::super::{
    object, ContractNumber, ExtraFields, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlanError, TrainingTarget, TrainingTargetLimitError,
};
use super::floors::{
    IrisMemoryFloor, Peak, CAMPAIGN_CUDA_CARD, CANDLE_RESERVE_GIB, MLX_RESERVE_GIB,
};
use super::{
    boolean, finite_in, invalid, iris_options, one_of, validate_micro_steps, whole,
    IRIS_BASE_MODEL_REPO, IRIS_FAMILY, IRIS_OPTIONS_KEY, IRIS_TRAINING_KERNEL, LOAD_FROM_JOB_KEY,
    RESUME_FROM_JOB_KEY,
};
use crate::store_util::is_safe_relative_path;

/// The depth training target id (the same id as the engine trainer and the depth task install).
pub const IRIS_DEPTH_TARGET_ID: &str = "iris_3b_depth";
/// The catalog model (and engine trainer id) a depth run trains: the depth task install.
pub const IRIS_DEPTH_BASE_MODEL: &str = "iris_3b_depth";
/// The task a depth artifact belongs to (the engine's `irisTask` stamp).
pub const IRIS_DEPTH_TASK: &str = "depth";
/// `TrainingConfig.advanced` key choosing where a depth run starts: `export` | `parent`.
pub const DEPTH_INIT_KEY: &str = "depthInit";
pub const DEPTH_INIT_EXPORT: &str = "export";
pub const DEPTH_INIT_PARENT: &str = "parent";
/// `TrainingDatasetItem.extra` key carrying an item's aligned depth target
/// (`{path, maskPath?, scale?, minDepth?, maxDepth?}`, dataset-relative paths). The worker resolves
/// it into the engine's `model_options["depth"]`.
pub const ITEM_DEPTH_KEY: &str = "irisDepth";
/// The paper's Hypersim native training size `[height, width]`.
pub const DEFAULT_DEPTH_SIZE: [u32; 2] = [480, 640];
/// Training-size bounds (each side a multiple of the backbone's 16-pixel patch).
pub const DEPTH_SIZE_MIN: u32 = 64;
pub const DEPTH_SIZE_MAX: u32 = 2048;
pub const DEPTH_SIZE_MULTIPLE: u32 = 16;
/// Paper §5.2: 10K optimizer steps at a global batch of 32 (offered as 1 × 32 accumulation).
pub const PAPER_OPTIMIZER_STEPS: u32 = 10_000;
pub const PAPER_GLOBAL_BATCH: u32 = 32;

/// Every engine depth option a client may set: the engine's `DEPTH_OPTION_KEYS` (32) minus the two
/// PATH options SceneWorks resolves from a trusted previous job.
pub const DEPTH_CLIENT_OPTION_KEYS: [&str; 30] = [
    "depth_size",
    "resize",
    "crop",
    "hflip_prob",
    "gradient_weight",
    "low_percentile",
    "high_percentile",
    "clip_target",
    "loss_reduction",
    "invalid_examples",
    "min_valid_pixels",
    "reducer_init",
    "betas",
    "gradient_clip",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_items",
    "preview_weights",
    "export_weights",
];

/// The depth objective's micro-step fields (whole `gradientAccumulation` windows).
pub const DEPTH_MICRO_STEP_FIELDS: [(&str, bool); 4] = [
    ("steps", true),
    ("saveEvery", true),
    ("sampleEvery", false),
    ("lrWarmupSteps", false),
];

/// Advanced keys of the generic Studio a depth run has no use for: the depth previews predict
/// dataset items (no prompt, no sampler), and the run is a full fine-tune (no adapter targets).
const UNREAD_ADVANCED_KEYS: [&str; 7] = [
    "samplePrompts",
    "sampleSteps",
    "sampleGuidanceScale",
    "sampleCount",
    "loraTargetModules",
    "timestepType",
    "timestepBias",
];

// ---- Measured inputs of the depth rows -------------------------------------------------------------
// The ONLY numbers to edit when a campaign re-measures: whole-process peaks, in the unit the probe
// printed (DECIMAL GB). sc-25690 terminal campaign, `train_memory_probe`, full fine-tune, batch 1.
//
// MLX (nax-macos-2; inference PR 1238 @b1c04bd0d, runs 38312190794 and 38318086154) at 512² and
// 1024²: the ACTIVE peak with MLX's buffer cache DISABLED — steady by step 2, flat through step 6.
// Production runs keep the cache, so their process footprint can rise to the allocator's GC ceiling
// (0.95 × the recommended working set) without that being need.
//
// Candle (RTX PRO 6000 card 1, 95.59 GiB; inference run 38393679427 @1449045c8, after the
// training-tape memory fix of inference #1242): the LIVE pool peak over six optimizer steps, steady
// from step 2. 1024² without activation checkpointing needs 116.77 GB and completed only by
// spilling 17.2 GB to system memory — a measured need that does not fit that card; under full
// activation checkpointing (`gradientCheckpointing`, new at that pin) 1024² peaks at 70.56 GB.
const MLX_DEPTH_512_PEAK: Peak = Peak::DecimalGb(53.60);
const MLX_DEPTH_1024_PEAK: Peak = Peak::DecimalGb(92.58);
// Under full activation checkpointing (`gradientCheckpointing`; MLX since inference #1243):
// inference run 38405154244 on nax-macos-2 — active peak, cache off, six steps, steady.
const MLX_DEPTH_AC_1024_PEAK: Peak = Peak::DecimalGb(63.01);
const CANDLE_DEPTH_512_PEAK: Peak = Peak::DecimalGb(70.41);
const CANDLE_DEPTH_1024_SPILL_PEAK: Peak = Peak::DecimalGb(116.77);
const CANDLE_DEPTH_AC_1024_PEAK: Peak = Peak::DecimalGb(70.56);
const PIXELS_512: f64 = 262_144.0;
const PIXELS_1024: f64 = 1_048_576.0;

/// The estimate (GiB, before the reserve) at `pixels` BETWEEN the two measured sizes of a lane: the
/// chord through them. The peak grows faster than linearly in pixels (attention), so the chord
/// sits at or above the true peak inside the interval — and below it outside, which is why nothing
/// under 512² is extrapolated with it.
const fn chord_gib(at_512: Peak, at_1024: Peak, pixels: f64) -> f64 {
    assert!(pixels >= PIXELS_512 && pixels <= PIXELS_1024);
    let low = at_512.gib();
    low + (at_1024.gib() - low) * (pixels - PIXELS_512) / (PIXELS_1024 - PIXELS_512)
}

const fn mlx_depth_estimate_gib(pixels: f64) -> f64 {
    chord_gib(MLX_DEPTH_512_PEAK, MLX_DEPTH_1024_PEAK, pixels)
}

const fn candle_depth_estimate_gib(pixels: f64) -> f64 {
    chord_gib(CANDLE_DEPTH_512_PEAK, CANDLE_DEPTH_1024_SPILL_PEAK, pixels)
}

const MLX_DEPTH_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_DEPTH_512_PEAK, MLX_RESERVE_GIB);
const MLX_DEPTH_16K: IrisMemoryFloor = IrisMemoryFloor::declared(64).bounded_by(MLX_DEPTH_512);
const MLX_DEPTH_307K: IrisMemoryFloor =
    IrisMemoryFloor::extrapolated(mlx_depth_estimate_gib(307_200.0), MLX_RESERVE_GIB);
const MLX_DEPTH_590K: IrisMemoryFloor =
    IrisMemoryFloor::extrapolated(mlx_depth_estimate_gib(589_824.0), MLX_RESERVE_GIB);
const MLX_DEPTH_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_DEPTH_AC_1024_PEAK, MLX_RESERVE_GIB);
const CANDLE_DEPTH_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_DEPTH_512_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
const CANDLE_DEPTH_16K: IrisMemoryFloor =
    IrisMemoryFloor::declared(68).bounded_by(CANDLE_DEPTH_512);
const CANDLE_DEPTH_307K: IrisMemoryFloor =
    IrisMemoryFloor::extrapolated(candle_depth_estimate_gib(307_200.0), CANDLE_RESERVE_GIB);
const CANDLE_DEPTH_590K: IrisMemoryFloor =
    IrisMemoryFloor::extrapolated(candle_depth_estimate_gib(589_824.0), CANDLE_RESERVE_GIB);
const CANDLE_DEPTH_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_DEPTH_AC_1024_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());

/// Memory floors, **GiB** ([`super::floors`]), per `(lane, artifact, micro-batch pixels ≤, floor)`,
/// each with its own provenance. A micro-batch's pixels are `batchSize × height × width` of the
/// training size. `full_ac` is the run under full activation checkpointing
/// (`advanced.gradientCheckpointing`), on the lanes whose trainer implements it
/// ([`super::iris_gradient_checkpointing_lanes`]); every other run reads `full`.
///
/// * MEASURED — MLX and Candle at 262 144 px (512²), MLX at 1 048 576 px (1024²), `full_ac` at
///   1 048 576 px on both lanes: the peaks above converted to GiB, plus the lane's 2 GiB reserve,
///   rounded up; a
///   Candle row that ran without spilling never exceeds the card that held its run.
/// * MEASURED WITH SPILL — Candle 1 048 576 px without checkpointing: the measured need, which the
///   card met only by spilling to system memory; the row refuses that card.
/// * EXTRAPOLATED — 307 200 / 589 824 px on both lanes (the lane's own chord, [`chord_gib`], + the
///   reserve); the 16 384 px rows, which the measured 512² row bounds (a smaller run never needs
///   more than a larger one; the MLX 64 was declared from a 59.0 GB process footprint at 128²,
///   cache included); and the `full_ac` rows under 1 048 576 px on both lanes, each the lower of
///   the plain row at that size and the lane's measured AC 1024² row.
/// * DECLARED — Candle 16 384 px where its declaration (68: a full fine-tune at 128×128 peaked at
///   64.6 GB of VRAM before the memory fix, padded) is not above the measured 512² row.
///
/// A micro-batch above the last row is refused rather than admitted unbounded.
const DEPTH_FLOORS_GIB: [(&str, &str, u64, IrisMemoryFloor); 20] = [
    ("mlx", "full", 16_384, MLX_DEPTH_16K),
    ("mlx", "full", 262_144, MLX_DEPTH_512),
    ("mlx", "full", 307_200, MLX_DEPTH_307K),
    ("mlx", "full", 589_824, MLX_DEPTH_590K),
    (
        "mlx",
        "full",
        1_048_576,
        IrisMemoryFloor::measured(MLX_DEPTH_1024_PEAK, MLX_RESERVE_GIB),
    ),
    (
        "mlx",
        "full_ac",
        16_384,
        IrisMemoryFloor::bound_from(MLX_DEPTH_16K).bounded_by(MLX_DEPTH_AC_1024),
    ),
    (
        "mlx",
        "full_ac",
        262_144,
        IrisMemoryFloor::bound_from(MLX_DEPTH_512).bounded_by(MLX_DEPTH_AC_1024),
    ),
    (
        "mlx",
        "full_ac",
        307_200,
        IrisMemoryFloor::bound_from(MLX_DEPTH_307K).bounded_by(MLX_DEPTH_AC_1024),
    ),
    (
        "mlx",
        "full_ac",
        589_824,
        IrisMemoryFloor::bound_from(MLX_DEPTH_590K).bounded_by(MLX_DEPTH_AC_1024),
    ),
    ("mlx", "full_ac", 1_048_576, MLX_DEPTH_AC_1024),
    ("candle", "full", 16_384, CANDLE_DEPTH_16K),
    ("candle", "full", 262_144, CANDLE_DEPTH_512),
    ("candle", "full", 307_200, CANDLE_DEPTH_307K),
    ("candle", "full", 589_824, CANDLE_DEPTH_590K),
    (
        "candle",
        "full",
        1_048_576,
        IrisMemoryFloor::measured_with_spill(
            CANDLE_DEPTH_1024_SPILL_PEAK,
            CANDLE_RESERVE_GIB,
            CAMPAIGN_CUDA_CARD.gib(),
        ),
    ),
    (
        "candle",
        "full_ac",
        16_384,
        IrisMemoryFloor::bound_from(CANDLE_DEPTH_16K).bounded_by(CANDLE_DEPTH_AC_1024),
    ),
    (
        "candle",
        "full_ac",
        262_144,
        IrisMemoryFloor::bound_from(CANDLE_DEPTH_512).bounded_by(CANDLE_DEPTH_AC_1024),
    ),
    (
        "candle",
        "full_ac",
        307_200,
        IrisMemoryFloor::bound_from(CANDLE_DEPTH_307K).bounded_by(CANDLE_DEPTH_AC_1024),
    ),
    (
        "candle",
        "full_ac",
        589_824,
        IrisMemoryFloor::bound_from(CANDLE_DEPTH_590K).bounded_by(CANDLE_DEPTH_AC_1024),
    ),
    ("candle", "full_ac", 1_048_576, CANDLE_DEPTH_AC_1024),
];

/// The training size `[height, width]` a depth config runs at: `iris.depth_size`, else the square
/// `resolution`.
pub fn depth_training_size(config: &TrainingConfig) -> (u32, u32) {
    let size = config
        .advanced
        .get(IRIS_OPTIONS_KEY)
        .and_then(|options| options.get("depth_size"))
        .and_then(Value::as_array)
        .and_then(|pair| {
            let side = |value: &Value| value.as_u64().and_then(|value| u32::try_from(value).ok());
            match pair.as_slice() {
                [h, w] => Some((side(h)?, side(w)?)),
                _ => None,
            }
        });
    size.unwrap_or((config.resolution, config.resolution))
}

/// The pixels of one micro-batch (`batchSize × height × width`).
pub fn depth_micro_batch_pixels(config: &TrainingConfig) -> u64 {
    let (height, width) = depth_training_size(config);
    u64::from(config.batch_size.max(1)) * u64::from(height) * u64::from(width)
}

/// The floor row (GiB + provenance) of a depth run on `lane`, or `None` above the largest row /
/// for an unknown lane (callers refuse, never admit unbounded). A run under activation
/// checkpointing reads the lane's `full_ac` rows where it has them.
pub fn depth_memory_floor(lane: &str, config: &TrainingConfig) -> Option<IrisMemoryFloor> {
    let pixels = depth_micro_batch_pixels(config);
    let lookup = |artifact: &str| {
        DEPTH_FLOORS_GIB
            .iter()
            .filter(|(row_lane, row_artifact, row_pixels, _)| {
                *row_lane == lane && *row_artifact == artifact && *row_pixels >= pixels
            })
            .min_by_key(|(_, _, row_pixels, _)| *row_pixels)
            .map(|(_, _, _, floor)| *floor)
    };
    super::iris_gradient_checkpointing(config)
        .then(|| lookup("full_ac"))
        .flatten()
        .or_else(|| lookup("full"))
}

fn memory_floors_json() -> Value {
    super::floors::memory_floors_json(
        Some("microBatchPixels"),
        "Keyed by the pixels of one micro-batch (batch size × training height × width); `full_ac` \
         holds the rows of a run under gradient (activation) checkpointing on the lanes that \
         implement it. Evidence is the whole-process peak of a real full fine-tune of the 3B \
         depth model at batch 1, converted to GiB, plus a 2 GiB reserve. MLX peaks are the ACTIVE \
         peak with MLX's buffer cache disabled; production runs keep the cache, so a process \
         footprint can rise to the allocator's GC ceiling (0.95 × the recommended working set) \
         without that being need. Candle peaks are the live pool peak over six optimizer steps (steady from step 2) on a 95.59 \
         GiB card; a row marked measured_spill is a need the run met only by spilling to system \
         memory, so it does not fit that card.",
        DEPTH_FLOORS_GIB.iter().map(|(lane, artifact, pixels, floor)| {
            let mut path = vec![pixels.to_string()];
            if *artifact != "full" {
                path.insert(0, (*artifact).to_owned());
            }
            (*lane, path, *floor)
        }),
    )
}

/// The Iris-3B depth training target: a full fine-tune that writes a `depth/` export.
pub fn iris_depth_target() -> TrainingTarget {
    let mut extra = ExtraFields::new();
    extra.insert("memoryFloors".to_owned(), memory_floors_json());
    extra.insert("irisObjective".to_owned(), json!(IRIS_DEPTH_TASK));
    extra.insert(
        "depthTraining".to_owned(),
        json!({
            "sizeMin": DEPTH_SIZE_MIN,
            "sizeMax": DEPTH_SIZE_MAX,
            "sizeMultiple": DEPTH_SIZE_MULTIPLE,
            "inits": [DEPTH_INIT_EXPORT, DEPTH_INIT_PARENT],
            "depthFormats": ["npy", "png8", "png16"],
            "paperOptimizerSteps": PAPER_OPTIMIZER_STEPS,
            "paperGlobalBatch": PAPER_GLOBAL_BATCH,
        }),
    );
    let accumulation = PAPER_GLOBAL_BATCH;
    let cadence = 1_000 * accumulation;
    TrainingTarget {
        id: IRIS_DEPTH_TARGET_ID.to_owned(),
        name: "Iris 3B Depth".to_owned(),
        modality: TrainingModality::Image,
        // Always a full model (the `depth/` export) — a model-catalog entry, never an adapter.
        output_kind: TrainingOutputKind::BaseCheckpoint,
        family: IRIS_FAMILY.to_owned(),
        base_model: IRIS_DEPTH_BASE_MODEL.to_owned(),
        base_model_repo: Some(IRIS_BASE_MODEL_REPO.to_owned()),
        kernel: IRIS_TRAINING_KERNEL.to_owned(),
        defaults: TrainingConfig {
            // No adapter: rank/alpha are not read by a full fine-tune (the Studio hides them).
            rank: 1,
            alpha: ContractNumber::from(1),
            learning_rate: ContractNumber::from_f64(3e-5).expect("3e-5 is finite"),
            // MICRO-steps: the paper's 10K optimizer steps × 32 accumulation.
            steps: PAPER_OPTIMIZER_STEPS * accumulation,
            batch_size: 1,
            gradient_accumulation: accumulation,
            // The square fallback size; the run trains at `iris.depth_size`, whose larger side this
            // mirrors (validated).
            resolution: DEFAULT_DEPTH_SIZE[1],
            save_every: cadence,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "networkType": "full",
                "lossType": "l1",
                "weightDecay": 0.01,
                "lrScheduler": "constant",
                "lrWarmupSteps": 0,
                "gradientCheckpointing": false,
                "sampleEvery": cadence,
                "outputScope": "project",
                "requestedGpu": "auto",
                "depthInit": DEPTH_INIT_EXPORT,
                // The engine `model_options["iris"]` bag: the trainer's defaults (paper §5.2 where it
                // speaks) except `keep_last_checkpoints` (a full checkpoint is ~48 GB, so keep-all
                // would fill a disk).
                "iris": {
                    "depth_size": DEFAULT_DEPTH_SIZE,
                    "resize": "cover",
                    "crop": "center",
                    "hflip_prob": 0.5,
                    "gradient_weight": 5.0,
                    "low_percentile": 2.0,
                    "high_percentile": 98.0,
                    "clip_target": false,
                    "loss_reduction": "valid_pixels",
                    "invalid_examples": "reject",
                    "min_valid_pixels": 2,
                    "betas": [0.9, 0.999],
                    "gradient_clip": 1.0,
                    "ema_enabled": false,
                    "nan_loss_tolerance": 20,
                    "keep_last_checkpoints": 2,
                    "milestone_steps": [],
                    "preview_items": 4,
                    "preview_weights": "raw",
                    "export_weights": "raw"
                }
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "steps": [1, 3_200_000],
            "batchSize": [1, 64],
            "optimizers": ["adamw", "muon"],
            "networkTypes": ["full"],
            "lrSchedulers": ["constant", "cosine"],
            "lossTypes": ["l1"],
            "mixedPrecisions": ["bf16", "f32"],
            // Activation checkpointing through the shared toggle, on the lanes whose depth trainer
            // implements it (never pinned off: a lane that lacks it refuses typed before load).
            "gradientCheckpointingLanes":
                super::iris_gradient_checkpointing_lanes(super::IrisObjective::Depth),
            "microStepFields": ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Iris 3B Depth",
            "description": "Fine-tune Iris-3B monocular depth on aligned RGB + numerical depth (+ \
                            validity mask) examples — the paper's recipe (full fine-tune, masked L1 \
                            + gradient loss, AdamW 3e-5). Writes a depth model you can pick in the \
                            Image Editor and Library depth actions. Apple Silicon (MLX) and NVIDIA \
                            (Candle/CUDA).",
            "recommendedFor": ["depth"],
            "datasetModality": "image",
            "objective": IRIS_DEPTH_TASK
        })),
        extra,
    }
}

fn depth_size_value(value: &Value) -> Result<(u32, u32), TrainingPlanError> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| invalid("advanced.iris.depth_size must be [height, width]."))?;
    let side = |value: &Value| -> Result<u32, TrainingPlanError> {
        let side = value
            .as_u64()
            .and_then(|side| u32::try_from(side).ok())
            .ok_or_else(|| invalid("advanced.iris.depth_size sides must be integers."))?;
        if !(DEPTH_SIZE_MIN..=DEPTH_SIZE_MAX).contains(&side)
            || !side.is_multiple_of(DEPTH_SIZE_MULTIPLE)
        {
            return Err(invalid(format!(
                "advanced.iris.depth_size sides must be multiples of {DEPTH_SIZE_MULTIPLE} between \
                 {DEPTH_SIZE_MIN} and {DEPTH_SIZE_MAX} (got {side})."
            )));
        }
        Ok(side)
    };
    Ok((side(&pair[0])?, side(&pair[1])?))
}

fn validate_option(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    match key {
        "depth_size" => depth_size_value(value).map(drop),
        "resize" => one_of(key, value, &["cover", "none"]),
        "crop" => one_of(key, value, &["center", "random"]),
        "loss_reduction" => one_of(key, value, &["valid_pixels", "per_image"]),
        "invalid_examples" => one_of(key, value, &["reject", "skip"]),
        "reducer_init" => one_of(key, value, &["torch_default", "channel_mean"]),
        "muon_adjust_lr" => one_of(key, value, &["rms_norm", "spectral_norm", "none"]),
        "auto_lr" => one_of(key, value, &["none", "sqrt", "linear"]),
        "resume_data_policy" => one_of(key, value, &["exact", "new_phase"]),
        "preview_weights" | "export_weights" => one_of(key, value, &["raw", "ema"]),
        "hflip_prob" | "ema_decay" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "gradient_weight" => finite_in(key, value, |x| x >= 0.0, "at least 0").map(drop),
        "low_percentile" | "high_percentile" => finite_in(
            key,
            value,
            |x| (0.0..=100.0).contains(&x),
            "between 0 and 100",
        )
        .map(drop),
        "gradient_clip" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "muon_momentum" => {
            finite_in(key, value, |x| (0.0..1.0).contains(&x), "in [0, 1)").map(drop)
        }
        "min_valid_pixels" => whole(key, value, 2).map(drop),
        "base_batch_size" | "num_epochs" => whole(key, value, 1).map(drop),
        "nan_loss_tolerance" | "keep_last_checkpoints" | "preview_items" => {
            whole(key, value, 0).map(drop)
        }
        "clip_target" | "muon_nesterov" | "ema_enabled" | "override_lr_on_resume" => {
            boolean(key, value).map(drop)
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
                invalid("advanced.iris.milestone_steps must be a list of positive optimizer steps.")
            }),
        "betas" => {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| invalid("advanced.iris.betas must be a pair [beta1, beta2]."))?;
            if pair.iter().all(|beta| {
                beta.as_f64()
                    .is_some_and(|beta| beta.is_finite() && (0.0..1.0).contains(&beta))
            }) {
                Ok(())
            } else {
                Err(invalid("advanced.iris.betas must both be in [0, 1)."))
            }
        }
        _ => Err(invalid(format!(
            "advanced.iris.{key} is not an Iris depth training option."
        ))),
    }
}

/// The run's start (`advanced.depthInit`), validated.
pub fn depth_init(config: &TrainingConfig) -> Result<&str, TrainingPlanError> {
    match config.advanced.get(DEPTH_INIT_KEY) {
        None | Some(Value::Null) => Ok(DEPTH_INIT_EXPORT),
        Some(Value::String(init)) if init == DEPTH_INIT_EXPORT => Ok(DEPTH_INIT_EXPORT),
        Some(Value::String(init)) if init == DEPTH_INIT_PARENT => Ok(DEPTH_INIT_PARENT),
        Some(other) => Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedValue {
                field: DEPTH_INIT_KEY.to_owned(),
                value: other
                    .as_str()
                    .map_or_else(|| other.to_string(), str::to_owned),
                allowed: vec![DEPTH_INIT_EXPORT.to_owned(), DEPTH_INIT_PARENT.to_owned()],
            },
        )),
    }
}

/// Submit-time validation of a depth config: everything `DepthTrainPlan::resolve` would refuse from
/// the request alone, as typed 400s.
pub fn validate_depth_config(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    validate_micro_steps(
        config,
        u64::from(config.gradient_accumulation.max(1)),
        &DEPTH_MICRO_STEP_FIELDS,
    )?;
    if config
        .trigger_word
        .as_deref()
        .is_some_and(|word| !word.trim().is_empty())
    {
        return Err(invalid(
            "The depth task is conditioned on the empty prompt; it takes no trigger word.",
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
                "advanced.{key} is not a depth training control (depth previews predict dataset \
                 items with no prompt or sampler, and the run is a full fine-tune)."
            )));
        }
    }
    let init = depth_init(config)?;
    let options = iris_options(config)?;
    for (key, value) in &options {
        if super::IRIS_API_OWNED_OPTION_KEYS.contains(&key.as_str()) {
            return Err(invalid(format!(
                "advanced.iris.{key} is set by SceneWorks from a previous run; name the run with \
                 advanced.{} instead.",
                if key == "resume_from" {
                    RESUME_FROM_JOB_KEY
                } else {
                    LOAD_FROM_JOB_KEY
                }
            )));
        }
        if !DEPTH_CLIENT_OPTION_KEYS.contains(&key.as_str()) {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: "iris".to_owned(),
                    value: key.clone(),
                    allowed: DEPTH_CLIENT_OPTION_KEYS
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
    // The training size: `resolution` mirrors the larger side so it never says something else.
    if let Some(size) = options.get("depth_size").filter(|value| !value.is_null()) {
        let (height, width) = depth_size_value(size)?;
        if config.resolution != height.max(width) {
            return Err(invalid(format!(
                "resolution ({}) must equal the larger side of advanced.iris.depth_size \
                 ({height}×{width}); the depth run trains at depth_size.",
                config.resolution
            )));
        }
    } else if !config.resolution.is_multiple_of(DEPTH_SIZE_MULTIPLE)
        || !(DEPTH_SIZE_MIN..=DEPTH_SIZE_MAX).contains(&config.resolution)
    {
        return Err(invalid(format!(
            "resolution must be a multiple of {DEPTH_SIZE_MULTIPLE} between {DEPTH_SIZE_MIN} and \
             {DEPTH_SIZE_MAX}."
        )));
    }
    let low = options
        .get("low_percentile")
        .and_then(Value::as_f64)
        .unwrap_or(2.0);
    let high = options
        .get("high_percentile")
        .and_then(Value::as_f64)
        .unwrap_or(98.0);
    if low >= high {
        return Err(invalid(
            "advanced.iris.low_percentile must be below advanced.iris.high_percentile.",
        ));
    }
    if options
        .get("reducer_init")
        .is_some_and(|value| !value.is_null())
        && init != DEPTH_INIT_PARENT
    {
        return Err(invalid(
            "advanced.iris.reducer_init initializes the depth head of a run that starts from the \
             generation backbone (depthInit: parent); a run from the depth export already has one.",
        ));
    }
    let ema_enabled = options
        .get("ema_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if options
        .get("ema_decay")
        .is_some_and(|value| !value.is_null())
        && !ema_enabled
    {
        return Err(invalid(
            "advanced.iris.ema_decay needs the EMA enabled (advanced.iris.ema_enabled).",
        ));
    }
    for key in ["preview_weights", "export_weights"] {
        if options.get(key).and_then(Value::as_str) == Some("ema") && !ema_enabled {
            return Err(invalid(format!(
                "advanced.iris.{key} = ema needs the EMA enabled (advanced.iris.ema_enabled)."
            )));
        }
    }
    let resuming = config.advanced.contains_key(RESUME_FROM_JOB_KEY);
    if !resuming {
        for key in ["resume_data_policy", "override_lr_on_resume"] {
            if options.get(key).is_some_and(|value| !value.is_null()) {
                return Err(invalid(format!(
                    "advanced.iris.{key} applies only when resuming a previous run \
                     (advanced.{RESUME_FROM_JOB_KEY})."
                )));
            }
        }
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY) {
        if resuming {
            return Err(invalid(
                "Resume continues a run's full state; loadFromJobId starts a fresh run from its \
                 weights. Choose one.",
            ));
        }
        if init != DEPTH_INIT_EXPORT {
            return Err(invalid(
                "loadFromJobId starts from a previous depth run's weights; it replaces the \
                 generation-backbone start (depthInit: parent). Use depthInit: export.",
            ));
        }
    }
    Ok(())
}

/// One item's aligned depth target, validated (`extra.irisDepth`).
pub fn validate_item_depth(extra: &ExtraFields) -> Result<(), String> {
    let Some(value) = extra.get(ITEM_DEPTH_KEY) else {
        return Ok(());
    };
    let map = value.as_object().ok_or_else(|| {
        "Item depth must be an object {path, maskPath?, scale?, minDepth?, maxDepth?}.".to_owned()
    })?;
    for key in map.keys() {
        if !matches!(
            key.as_str(),
            "path" | "maskPath" | "scale" | "minDepth" | "maxDepth"
        ) {
            return Err(format!(
                "Item depth key {key:?} is not part of the depth target."
            ));
        }
    }
    let path = |key: &str, required: bool| -> Result<(), String> {
        match map.get(key) {
            None | Some(Value::Null) if !required => Ok(()),
            Some(Value::String(path)) if is_safe_relative_path(path) => {
                let lower = path.to_ascii_lowercase();
                if lower.ends_with(".npy") || lower.ends_with(".png") {
                    Ok(())
                } else {
                    Err(format!(
                        "Item depth {key} must be a .npy or grayscale .png file (got {path:?})."
                    ))
                }
            }
            _ => Err(format!(
                "Item depth {key} must be a path relative to the dataset root."
            )),
        }
    };
    path("path", true)?;
    path("maskPath", false)?;
    let number = |key: &str| -> Result<Option<f64>, String> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .filter(|number| number.is_finite())
                .map(Some)
                .ok_or_else(|| format!("Item depth {key} must be a finite number.")),
        }
    };
    if number("scale")?.is_some_and(|scale| scale <= 0.0) {
        return Err("Item depth scale must be greater than 0.".to_owned());
    }
    let min_depth = number("minDepth")?.unwrap_or(0.0);
    if min_depth < 0.0 {
        return Err("Item depth minDepth must be at least 0.".to_owned());
    }
    if number("maxDepth")?.is_some_and(|max| max <= min_depth) {
        return Err("Item depth maxDepth must exceed minDepth.".to_owned());
    }
    Ok(())
}

/// Every dataset item must be an aligned example: an RGB image plus its depth target. The files'
/// alignment, formats and validity are checked by the engine's own readers (the dataset check
/// endpoint and the trainer's dataset pass).
pub fn validate_depth_dataset(dataset: &TrainingDataset) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_depth(&item.extra).map_err(invalid)?;
    }
    let missing: Vec<&str> = dataset
        .items
        .iter()
        .filter(|item| !item.extra.contains_key(ITEM_DEPTH_KEY))
        .map(|item| item.display_name.as_str())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(invalid(format!(
        "{} dataset item(s) have no depth target: {}{}. Attach each image's depth map (.npy or \
         8/16-bit grayscale .png) before training depth.",
        missing.len(),
        missing
            .iter()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .join(", "),
        if missing.len() > 5 { ", …" } else { "" }
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TrainingConfig {
        iris_depth_target().defaults
    }

    #[test]
    fn the_client_allow_list_is_the_engine_list_minus_the_path_options() {
        // The engine's `DEPTH_OPTION_KEYS` (32) = these 30 + `load_from` + `resume_from`.
        assert_eq!(DEPTH_CLIENT_OPTION_KEYS.len() + 2, 32);
        let mut sorted = DEPTH_CLIENT_OPTION_KEYS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), DEPTH_CLIENT_OPTION_KEYS.len());
    }

    #[test]
    fn the_defaults_are_the_paper_recipe_and_validate() {
        let target = iris_depth_target();
        let defaults = &target.defaults;
        assert_eq!(defaults.learning_rate.as_f64(), Some(3e-5));
        assert_eq!(defaults.advanced["weightDecay"], json!(0.01));
        assert_eq!(defaults.advanced["lrScheduler"], "constant");
        assert_eq!(defaults.advanced["lrWarmupSteps"], 0);
        assert_eq!(defaults.advanced["networkType"], "full");
        assert_eq!(defaults.advanced["lossType"], "l1");
        assert_eq!(defaults.optimizer, "adamw");
        // 10K optimizer steps at a global batch of 32.
        assert_eq!(defaults.batch_size * defaults.gradient_accumulation, 32);
        assert_eq!(defaults.steps / defaults.gradient_accumulation, 10_000);
        assert_eq!(target.output_kind, TrainingOutputKind::BaseCheckpoint);
        validate_depth_config(defaults).expect("the defaults validate");
    }

    #[test]
    fn request_only_refusals_are_typed_400s() {
        let refused = |mutate: &dyn Fn(&mut TrainingConfig)| {
            let mut config = config();
            mutate(&mut config);
            validate_depth_config(&config).expect_err("refused")
        };
        // Micro-step rule.
        let error = refused(&|config| config.save_every = 33);
        assert!(matches!(
            error,
            TrainingPlanError::TargetLimit(TrainingTargetLimitError::NotMultipleOf { .. })
        ));
        // Unknown and API-owned options.
        refused(&|config| config.advanced["iris"]["prediction"] = json!("v"));
        refused(&|config| config.advanced["iris"]["load_from"] = json!("/tmp/x"));
        // Domains.
        refused(&|config| config.advanced["iris"]["invalid_examples"] = json!("ignore"));
        refused(&|config| config.advanced["iris"]["depth_size"] = json!([480, 650]));
        refused(&|config| config.advanced["iris"]["min_valid_pixels"] = json!(1));
        refused(&|config| {
            config.advanced["iris"]["low_percentile"] = json!(60.0);
            config.advanced["iris"]["high_percentile"] = json!(40.0);
        });
        // The size mirror.
        refused(&|config| config.resolution = 512);
        // Generic controls the depth objective does not read.
        refused(&|config| {
            config
                .advanced
                .insert("samplePrompts".into(), json!(["a cat"]));
        });
        refused(&|config| config.trigger_word = Some("tok".into()));
        // The reducer initializes a parent start only; EMA-dependent selections need EMA.
        refused(&|config| config.advanced["iris"]["reducer_init"] = json!("channel_mean"));
        refused(&|config| config.advanced["iris"]["export_weights"] = json!("ema"));
        refused(&|config| config.advanced["iris"]["ema_decay"] = json!(0.99));
        refused(&|config| config.advanced["iris"]["resume_data_policy"] = json!("new_phase"));
        refused(&|config| {
            config.advanced.insert("depthInit".into(), json!("parent"));
            config
                .advanced
                .insert("loadFromJobId".into(), json!("job_1"));
        });
        refused(&|config| {
            config.advanced.insert("depthInit".into(), json!("random"));
        });
        // ...and the matching accepted shapes.
        let mut parent = config();
        parent.advanced.insert("depthInit".into(), json!("parent"));
        parent.advanced["iris"]["reducer_init"] = json!("channel_mean");
        validate_depth_config(&parent).expect("a parent start may choose the reducer init");
        let mut ema = config();
        ema.advanced["iris"]["ema_enabled"] = json!(true);
        ema.advanced["iris"]["export_weights"] = json!("ema");
        validate_depth_config(&ema).expect("EMA export with EMA on");
        let mut skip = config();
        skip.advanced["iris"]["invalid_examples"] = json!("skip");
        validate_depth_config(&skip).expect("skip is the other explicit policy");
    }

    #[test]
    fn floors_follow_the_micro_batch_and_refuse_beyond_the_table() {
        let mut config = config();
        // 480×640 × 1 = 307 200 px: its own row on each lane; 128² is the first, and no higher.
        let row = |lane: &str, artifact: &str, pixels: u64| {
            DEPTH_FLOORS_GIB
                .iter()
                .find(|row| row.0 == lane && row.1 == artifact && row.2 == pixels)
                .map(|row| row.3)
        };
        for lane in ["mlx", "candle"] {
            assert_eq!(
                depth_memory_floor(lane, &config),
                row(lane, "full", 307_200)
            );
        }
        let default_shape = depth_memory_floor("candle", &config).unwrap();
        config.advanced["iris"]["depth_size"] = json!([128, 128]);
        config.resolution = 128;
        for lane in ["mlx", "candle"] {
            assert_eq!(depth_memory_floor(lane, &config), row(lane, "full", 16_384));
        }
        assert!(depth_memory_floor("candle", &config).unwrap().gib <= default_shape.gib);
        // 512² reads the row measured at exactly that shape, not the next one up.
        config.advanced["iris"]["depth_size"] = json!([512, 512]);
        config.resolution = 512;
        assert_eq!(depth_memory_floor("mlx", &config), Some(MLX_DEPTH_512));
        assert_eq!(
            depth_memory_floor("candle", &config),
            Some(CANDLE_DEPTH_512)
        );
        // The paper's batch 32 in one micro-batch is beyond every row.
        let mut big = iris_depth_target().defaults;
        big.batch_size = 32;
        big.gradient_accumulation = 1;
        assert_eq!(depth_memory_floor("mlx", &big), None);
        assert_eq!(depth_memory_floor("cpu", &config), None);
    }

    /// sc-25690: a run under gradient (activation) checkpointing reads the lane's `full_ac` rows —
    /// Candle 1024², which does not fit a 96 GB card without it, fits with it — and a lane with no
    /// such rows keeps reading the plain ones (it refuses the toggle before load anyway).
    #[test]
    fn a_checkpointed_run_reads_the_full_ac_rows_where_the_lane_has_them() {
        use super::super::floors::FloorProvenance;
        let mut config = config();
        config.advanced["iris"]["depth_size"] = json!([1024, 1024]);
        config.resolution = 1024;
        let plain = depth_memory_floor("candle", &config).unwrap();
        let mlx_plain = depth_memory_floor("mlx", &config).unwrap();
        assert_eq!(plain.provenance, FloorProvenance::MeasuredSpill);
        let card = CAMPAIGN_CUDA_CARD.gib();
        assert!(f64::from(plain.gib) > card, "{plain:?}");
        config
            .advanced
            .insert("gradientCheckpointing".into(), json!(true));
        let checkpointed = depth_memory_floor("candle", &config).unwrap();
        assert_eq!(checkpointed, CANDLE_DEPTH_AC_1024);
        assert_eq!(checkpointed.provenance, FloorProvenance::Measured);
        assert!(f64::from(checkpointed.gib) <= card, "{checkpointed:?}");
        // MLX (inference #1243): its own measured `full_ac` row, well under the plain one.
        assert_eq!(depth_memory_floor("mlx", &config), Some(MLX_DEPTH_AC_1024));
        assert!(MLX_DEPTH_AC_1024.gib < mlx_plain.gib);
        // A lane with no `full_ac` rows keeps reading its plain ones.
        assert_eq!(depth_memory_floor("cpu", &config), None);
        // Checkpointing never raises a floor, at any row.
        for (lane, artifact, pixels, floor) in DEPTH_FLOORS_GIB {
            if artifact == "full_ac" {
                let plain = DEPTH_FLOORS_GIB
                    .iter()
                    .find(|row| row.0 == lane && row.1 == "full" && row.2 == pixels)
                    .unwrap()
                    .3;
                assert!(floor.gib <= plain.gib, "{lane} {pixels}");
            }
        }
        // The target publishes the lanes, and does not pin the toggle off.
        let limits = &iris_depth_target().limits;
        assert_eq!(
            limits["gradientCheckpointingLanes"],
            json!(["candle", "mlx"])
        );
        assert!(limits.get("gradientCheckpointing").is_none());
    }

    /// sc-25690: floors only grow with the micro-batch, each row says what it rests on — measured
    /// at its own shape, measured by spilling, extrapolated, or declared — and the published
    /// provenance carries the same per row, with no table-wide `measured` flag.
    #[test]
    fn floor_rows_are_monotonic_and_publish_their_own_provenance() {
        use super::super::floors::FloorProvenance;
        for lane in ["mlx", "candle"] {
            for artifact in ["full", "full_ac"] {
                let rows: Vec<_> = DEPTH_FLOORS_GIB
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
        // Each lane's extrapolation is the chord through its own two measured sizes.
        assert_eq!(mlx_depth_estimate_gib(PIXELS_512), MLX_DEPTH_512_PEAK.gib());
        assert!((mlx_depth_estimate_gib(PIXELS_1024) - MLX_DEPTH_1024_PEAK.gib()).abs() < 1e-9);
        assert_eq!(
            candle_depth_estimate_gib(PIXELS_512),
            CANDLE_DEPTH_512_PEAK.gib()
        );
        assert!(
            (candle_depth_estimate_gib(PIXELS_1024) - CANDLE_DEPTH_1024_SPILL_PEAK.gib()).abs()
                < 1e-9
        );
        let published = &iris_depth_target().extra["memoryFloors"];
        assert_eq!(published["unit"], json!("GiB"));
        assert!(published.get("measured").is_none());
        for (lane, artifact, pixels, floor) in DEPTH_FLOORS_GIB {
            let expected = match (lane, artifact, pixels) {
                (_, "full", 262_144) | ("mlx", "full", 1_048_576) => FloorProvenance::Measured,
                (_, "full_ac", 1_048_576) => FloorProvenance::Measured,
                ("candle", "full", 1_048_576) => FloorProvenance::MeasuredSpill,
                ("candle", _, 16_384) => FloorProvenance::Declared,
                _ => FloorProvenance::Extrapolated,
            };
            assert_eq!(floor.provenance, expected, "{lane} {artifact} {pixels}");
            let (values, provenance) = if artifact == "full" {
                (&published["lanes"][lane], &published["provenance"][lane])
            } else {
                (
                    &published["lanes"][lane][artifact],
                    &published["provenance"][lane][artifact],
                )
            };
            assert_eq!(provenance[pixels.to_string()], json!(expected.as_str()));
            assert_eq!(values[pixels.to_string()], json!(floor.gib));
        }
        let basis = published["basis"].as_str().unwrap();
        assert!(!basis.contains("no measurement campaign"), "{basis}");
        assert!(
            basis.contains("Measured at the row's own shape: mlx 262144; mlx 1048576; mlx full_ac 1048576; candle 262144; candle full_ac 1048576.")
                && basis.contains("completed only by spilling to system memory): candle 1048576.")
                && basis.contains("ACTIVE peak with MLX's buffer cache disabled"),
            "{basis}"
        );
    }

    #[test]
    fn item_depth_targets_are_validated_at_the_store_boundary() {
        let item = |value: Value| {
            let mut extra = ExtraFields::new();
            extra.insert(ITEM_DEPTH_KEY.to_owned(), value);
            validate_item_depth(&extra)
        };
        item(json!({"path": "sidecars/a.depth.npy"})).unwrap();
        item(
            json!({"path": "sidecars/a.depth.png", "maskPath": "sidecars/a.mask.png",
                    "scale": 0.001, "minDepth": 0.1, "maxDepth": 80}),
        )
        .unwrap();
        assert!(item(json!({"path": "../escape.npy"})).is_err());
        assert!(item(json!({"path": "sidecars/a.exr"})).is_err());
        assert!(item(json!({"mask": "x"})).is_err());
        assert!(item(json!({"path": "a.npy", "scale": 0})).is_err());
        assert!(item(json!({"path": "a.npy", "minDepth": 5, "maxDepth": 5})).is_err());
    }
}
