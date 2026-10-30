//! Iris-3B generation training — the SceneWorks product surface of the native MLX (sc-25685) and
//! Candle/CUDA (sc-25686) trainers registered under `iris_3b` (epic sc-25678).
//!
//! The engine contract is `gen_core_iris::train`: the shared `TrainingConfig` plus the
//! `model_options["iris"]` options bag, whose unknown keys the trainer REFUSES. This module owns
//! the SceneWorks half of that contract so the Training Studio offers exactly what the trainer
//! accepts and nothing it refuses (E3):
//!
//! * [`iris_generation_target`] — the one training target. Its `limits` enumerate every choice the
//!   trainer honors (optimizer `adamw | muon`, LR scheduler `constant | cosine`, timestep sampler
//!   `logit_normal | uniform`, train dtype `bf16 | fp16 | f32`, network `lora | lokr | full`) and pin the
//!   refused ones closed (timestep bias `balanced`, loss `mse`, no gradient checkpointing).
//! * [`validate_iris_training_config`] — the submit-time twin of `IrisTrainPlan::resolve`'s
//!   request-only refusals, so a bad request is a typed 400 rather than a failure after the
//!   weights load. Notably the MICRO-STEP rule: the shared contract counts `steps`, `saveEvery`,
//!   `sampleEvery` and `lrWarmupSteps` in micro-steps, and the trainer lands checkpoints, previews
//!   and the warmup on optimizer-step boundaries, so each must be a whole number of
//!   `gradientAccumulation` windows ([`TrainingTargetLimitError::NotMultipleOf`]).
//! * [`validate_iris_dataset_captions`] / [`validate_item_captions`] — per-item named captions
//!   (`TrainingItem.model_options["captions"] = {field: text}`, upstream's per-sample `info`
//!   dict) and the `caption_field(s)` selection that reads them.
//! * [`iris_training_memory_floor_gib`] / [`iris_generation_memory_floor`] — the per-lane floors, in
//!   GiB, each row with its own provenance ([`floors`]: measured / extrapolated / lower bound /
//!   declared), so a routed Iris training lane is never admitted unbounded.
//!
//! The S11 surface of GENERATION training (sc-25689) — official stage presets, bucket/area shapes,
//! the RoPE law, REPA / iREPA, activation-checkpointing policies, holdout validation, fp16 and
//! distributed execution — lives in [`curricula`].
//!
//! The same `iris_train` kernel serves the two downstream TASK trainers too, keyed on the target's
//! base model ([`IrisObjective`]): depth ([`depth`], S9 sc-25687) and restoration ([`restore`], S10
//! sc-25688). Everything kernel-wide (durable cancel, resume / new phase, the memory-floor gate, the
//! data walk) applies to all three; the request surface, the dataset contract and the floors are per
//! objective.

pub mod curricula;
pub mod depth;
pub mod floors;
pub mod restore;

use serde_json::{json, Map, Value};

use floors::{IrisMemoryFloor, Peak, CAMPAIGN_CUDA_CARD, CANDLE_RESERVE_GIB, MLX_RESERVE_GIB};

use super::{
    object, ContractNumber, ExtraFields, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlanError, TrainingTarget, TrainingTargetLimitError,
};

/// The SceneWorks training kernel id the router and worker key the Iris trainer on.
pub const IRIS_TRAINING_KERNEL: &str = "iris_train";
/// The one Iris training target id.
pub const IRIS_TRAINING_TARGET_ID: &str = "iris_3b_generation";
/// The catalog model (and engine trainer id) the target trains: the generation task install.
pub const IRIS_BASE_MODEL: &str = "iris_3b";
/// The upstream backbone repo the generation install pulls.
pub const IRIS_BASE_MODEL_REPO: &str = "speridlabs/iris-3b";
/// The SceneWorks LoRA family an Iris adapter registers under (the engine stamps `family = "iris"`).
pub const IRIS_FAMILY: &str = "iris";
/// The task an Iris generation artifact belongs to (the engine's `irisTask` stamp).
pub const IRIS_GENERATION_TASK: &str = "generation";
/// `TrainingConfig.advanced` key carrying the engine's `model_options["iris"]` bag verbatim.
pub const IRIS_OPTIONS_KEY: &str = "iris";
/// `TrainingDatasetItem.extra` key carrying an item's named captions — passed through to the
/// engine `TrainingItem.model_options["captions"]` (`gen_core_iris::train::ITEM_CAPTIONS_KEY`).
pub const ITEM_CAPTIONS_KEY: &str = "captions";
/// Request key naming a previous Iris training job to RESUME (its newest durable checkpoint). The
/// API resolves it to the engine's `resume_from` path; a client never sends a path.
pub const RESUME_FROM_JOB_KEY: &str = "resumeFromJobId";
/// Request key naming a previous FULL Iris training job whose weights seed a full run
/// (`init: load_from`). The API resolves it to the engine's `load_from` path.
pub const LOAD_FROM_JOB_KEY: &str = "loadFromJobId";
/// Upstream's most preview prompts per cadence (`gen_core_iris::train::PREVIEW_PROMPT_CAP`).
pub const IRIS_PREVIEW_PROMPT_CAP: u32 = 4;

/// The micro-step fields that must be whole `gradientAccumulation` windows. `(request field, where it
/// lives)`; `true` = a top-level `TrainingConfig` field, `false` = an `advanced` key.
pub const IRIS_MICRO_STEP_FIELDS: [(&str, bool); 4] = [
    ("steps", true),
    ("saveEvery", true),
    ("sampleEvery", false),
    ("lrWarmupSteps", false),
];

/// Every engine option a client may set. The engine's full allow-list
/// (`gen_core_iris::train::IrisTrainOptions`) minus the two PATH options the API resolves from a
/// trusted job id ([`IRIS_API_OWNED_OPTION_KEYS`]).
pub const IRIS_CLIENT_OPTION_KEYS: [&str; 33] = [
    "init",
    "prediction",
    "flow_shift",
    "shift_law",
    "shift_base_tokens",
    "logit_mean",
    "logit_std",
    "x_pred_sigma_min",
    "text_dropout",
    "gradient_clip",
    "betas",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "caption_field",
    "caption_fields",
    "on_caption_overflow",
    "text_conditioning",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_weights",
    "preview_negative_prompt",
    "export_weights",
    "export_dtype",
    "upstream_validation_prompts",
];

/// Engine options that carry an on-disk path. Never client-authored: the API derives them from a
/// previous job ([`RESUME_FROM_JOB_KEY`] / [`LOAD_FROM_JOB_KEY`]) after validating it.
pub const IRIS_API_OWNED_OPTION_KEYS: [&str; 2] = ["resume_from", "load_from"];

/// Whether `kernel` is an Iris trainer (generation, depth or restoration — see [`IrisObjective`]).
pub fn is_iris_training_kernel(kernel: &str) -> bool {
    kernel == IRIS_TRAINING_KERNEL
}

/// What an Iris training run trains, keyed on its target's base model (the task install).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrisObjective {
    /// `iris_3b` — LoRA / LoKr adapters and full fine-tunes of the generation backbone.
    Generation,
    /// `iris_3b_depth` — the supervised depth fine-tune (a `depth/` export).
    Depth,
    /// `iris_3b_restore` — the adversarial restoration fine-tune (an `upscaler/` export).
    Restoration,
}

impl IrisObjective {
    /// The objective of an Iris target / plan base model, `None` for any other model.
    pub fn from_base_model(base_model: &str) -> Option<Self> {
        match base_model {
            IRIS_BASE_MODEL => Some(Self::Generation),
            depth::IRIS_DEPTH_BASE_MODEL => Some(Self::Depth),
            restore::IRIS_RESTORE_BASE_MODEL => Some(Self::Restoration),
            _ => None,
        }
    }

    /// The task identity its artifacts carry (`irisTask`).
    pub fn task(self) -> &'static str {
        match self {
            Self::Generation => IRIS_GENERATION_TASK,
            Self::Depth => depth::IRIS_DEPTH_TASK,
            Self::Restoration => restore::IRIS_RESTORATION_TASK,
        }
    }

    /// The catalog model (engine trainer id) it trains.
    pub fn base_model(self) -> &'static str {
        match self {
            Self::Generation => IRIS_BASE_MODEL,
            Self::Depth => depth::IRIS_DEPTH_BASE_MODEL,
            Self::Restoration => restore::IRIS_RESTORE_BASE_MODEL,
        }
    }

    /// The `TrainingConfig.advanced` key of its engine options bag.
    pub fn options_key(self) -> &'static str {
        match self {
            Self::Generation | Self::Depth => IRIS_OPTIONS_KEY,
            Self::Restoration => restore::RESTORE_OPTIONS_KEY,
        }
    }
}

/// The objective of an Iris training target, `None` for a non-Iris target.
pub fn iris_objective(target: &TrainingTarget) -> Option<IrisObjective> {
    if !is_iris_training_kernel(&target.kernel) {
        return None;
    }
    IrisObjective::from_base_model(&target.base_model)
}

/// The floor (GiB, with its provenance) of an Iris run of `objective` with `config` on `lane`, or
/// `None` when no floor row covers that shape (callers refuse rather than admit unbounded).
pub fn iris_objective_memory_floor(
    objective: IrisObjective,
    lane: &str,
    config: &TrainingConfig,
) -> Option<IrisMemoryFloor> {
    match objective {
        IrisObjective::Generation => iris_generation_memory_floor(
            lane,
            advanced_string(config, "networkType").unwrap_or("lora"),
            config.resolution,
            iris_full_activation_checkpointing(config),
        ),
        IrisObjective::Depth => depth::depth_memory_floor(lane, config),
        IrisObjective::Restoration => restore::restore_memory_floor(lane, config),
    }
}

/// The lanes whose trainer implements activation checkpointing for `objective` through the shared
/// `advanced.gradientCheckpointing` toggle (the engine's `TrainingConfig::gradient_checkpointing`,
/// which the depth and restoration trainers map to the `full` policy).
///
/// The engine publishes NO declaration of this — its trainer descriptor has no such flag, and a
/// lane that has not wired the regions refuses in `Trainer::validate`, which needs the weights
/// loaded. So this is an explicit table of what each lane's trainer accepts at the pinned inference
/// revision (Candle since inference #1242, MLX since #1243). The real-weight depth / restoration
/// training smokes ask the lane's own trainer and FAIL when its answer differs from this table, so
/// the table cannot drift from the engine unnoticed.
///
/// Generation has none: its control is the richer `advanced.iris.activation_checkpointing` policy,
/// and its trainer refuses the shared toggle on both lanes.
pub fn iris_gradient_checkpointing_lanes(objective: IrisObjective) -> &'static [&'static str] {
    match objective {
        IrisObjective::Generation => &[],
        IrisObjective::Depth | IrisObjective::Restoration => &["candle", "mlx"],
    }
}

/// Whether a run asks for the shared gradient (activation) checkpointing toggle.
pub fn iris_gradient_checkpointing(config: &TrainingConfig) -> bool {
    config.advanced.get("gradientCheckpointing") == Some(&Value::Bool(true))
}

/// `Some(message)` when a depth / restoration run asks for gradient checkpointing on a lane whose
/// trainer does not implement it for that task — a typed refusal before load, never a knob the
/// trainer would refuse after the weights are in (or silently ignore).
pub fn iris_gradient_checkpointing_refusal(
    objective: IrisObjective,
    lane: &str,
    config: &TrainingConfig,
) -> Option<String> {
    if objective == IrisObjective::Generation || !iris_gradient_checkpointing(config) {
        return None;
    }
    let lanes = iris_gradient_checkpointing_lanes(objective);
    (!lanes.contains(&lane)).then(|| {
        let backend = |lane: &str| {
            match lane {
                "mlx" => "Apple Silicon (MLX)",
                "candle" => "NVIDIA (Candle/CUDA)",
                other => other,
            }
            .to_owned()
        };
        let supported = if lanes.is_empty() {
            "no backend implements it for this objective yet".to_owned()
        } else {
            format!(
                "it is implemented on {}",
                lanes
                    .iter()
                    .map(|lane| backend(lane))
                    .collect::<Vec<_>>()
                    .join(" and ")
            )
        };
        format!(
            "Gradient (activation) checkpointing is not implemented for Iris 3B {} training on \
             {}; {supported}. Turn gradient checkpointing off to run here.",
            objective.task(),
            backend(lane)
        )
    })
}

/// A short description of the run shape a floor was declared for (refusal messages).
pub fn iris_objective_shape(objective: IrisObjective, config: &TrainingConfig) -> String {
    let network = advanced_string(config, "networkType").unwrap_or(match objective {
        IrisObjective::Generation => "lora",
        _ => "full",
    });
    let artifact = if network.eq_ignore_ascii_case("full") {
        "a full fine-tune"
    } else {
        "an adapter (LoRA/LoKr) run"
    };
    match objective {
        IrisObjective::Generation => format!("{artifact} at {}px", config.resolution),
        IrisObjective::Depth => {
            let (height, width) = depth::depth_training_size(config);
            format!(
                "a depth full fine-tune at {height}×{width}, batch {}",
                config.batch_size.max(1)
            )
        }
        IrisObjective::Restoration => format!(
            "a restoration {} at a {}² crop, batch {}",
            artifact.trim_start_matches("a ").trim_start_matches("an "),
            config.resolution,
            config.batch_size.max(1)
        ),
    }
}

// ---- Measured inputs of the generation full fine-tune rows --------------------------------------
// The ONLY numbers to edit when a campaign re-measures: each is the whole-process peak its probe
// printed, in the unit it printed it (`Peak`). Every dependent row below is computed from them.
//
// sc-25690 terminal campaign, `train_memory_probe`: full fine-tune, AdamW, bf16 over f32 masters,
// EMA on, cached text conditioning, batch 1, whole-process peak; summaries print DECIMAL GB.
//
// MLX (nax-macos-2; inference PR 1238 @b1c04bd0d, runs 38312190794 two steps + export and
// 38318086154 six steps): the ACTIVE peak with MLX's buffer cache DISABLED — what the run needs.
// Steady by optimizer step 2 and flat through step 6; the process footprint with the cache off
// agrees within 0.5 GB. Production runs keep the cache, so their footprint can rise to the
// allocator's GC ceiling (0.95 × the recommended working set, 109.68 GB on that Mac) without that
// being need — which is what the campaign's first MLX numbers were.
//
// Candle (RTX PRO 6000 card 1, 97,887 MiB = 95.59 GiB; inference run 38393679427 @1449045c8,
// after the training-tape memory fix of inference #1242): the LIVE pool peak over six optimizer
// steps, steady from step 2. 1024² without activation checkpointing needs 128.75 GB: it completed
// only by spilling 29.2 GB to system memory (115–157 s/step) — a measured need that does NOT fit
// that card.
const MLX_FULL_512_PEAK: Peak = Peak::DecimalGb(65.57);
const MLX_FULL_1024_PEAK: Peak = Peak::DecimalGb(104.49);
const MLX_FULL_AC_1024_PEAK: Peak = Peak::DecimalGb(74.88);
const CANDLE_FULL_512_PEAK: Peak = Peak::DecimalGb(82.41);
const CANDLE_FULL_1024_SPILL_PEAK: Peak = Peak::DecimalGb(128.75);
const CANDLE_FULL_AC_512_PEAK: Peak = Peak::DecimalGb(69.19);
const CANDLE_FULL_AC_1024_PEAK: Peak = Peak::DecimalGb(82.58);

const MLX_FULL_512: IrisMemoryFloor = IrisMemoryFloor::measured(MLX_FULL_512_PEAK, MLX_RESERVE_GIB);
const MLX_FULL_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_FULL_1024_PEAK, MLX_RESERVE_GIB);
const MLX_FULL_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(MLX_FULL_AC_1024_PEAK, MLX_RESERVE_GIB);
const CANDLE_FULL_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_FULL_512_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
const CANDLE_FULL_1024: IrisMemoryFloor = IrisMemoryFloor::measured_with_spill(
    CANDLE_FULL_1024_SPILL_PEAK,
    CANDLE_RESERVE_GIB,
    CAMPAIGN_CUDA_CARD.gib(),
);
const CANDLE_FULL_AC_512: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_FULL_AC_512_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
const CANDLE_FULL_AC_1024: IrisMemoryFloor =
    IrisMemoryFloor::measured(CANDLE_FULL_AC_1024_PEAK, CANDLE_RESERVE_GIB)
        .held_on(CAMPAIGN_CUDA_CARD.gib());
/// The 256² full rows are declared (80) from the 256² evidence below, and never above the measured
/// 512² row (a smaller run never needs more than a larger one).
const MLX_FULL_256: IrisMemoryFloor = IrisMemoryFloor::declared(80).bounded_by(MLX_FULL_512);
const CANDLE_FULL_256: IrisMemoryFloor = IrisMemoryFloor::declared(80).bounded_by(CANDLE_FULL_512);

/// Memory floors, **GiB** ([`floors`]), per `(lane, artifact, resolution)`, each with its own
/// provenance. `full_ac` is a FULL run under activation checkpointing
/// (`advanced.iris.activation_checkpointing` = `full` / `selective_op`,
/// [`iris_generation_memory_floor`]); every other full run reads `full`.
///
/// * MEASURED — `full` 512 / 1024 and `full_ac` 1024 on MLX; `full` 512 and `full_ac` 512 / 1024
///   on Candle: the `*_PEAK` constants above, converted to GiB, plus the lane's 2 GiB reserve
///   ([`floors::MLX_RESERVE_GIB`] / [`CANDLE_RESERVE_GIB`]), rounded up; a Candle row that ran
///   without spilling never exceeds the card that held its run.
/// * MEASURED WITH SPILL — Candle `full` 1024 ([`CANDLE_FULL_1024`]): the measured need, which the
///   card met only by spilling to system memory; the row refuses that card.
/// * EXTRAPOLATED — the `full` 256 rows where the measured 512 row bounds them; the `full_ac` rows
///   with no run of their own, each the lower of the no-AC row at that size and the next measured
///   AC row up (checkpointing only lowers a peak, and a smaller run never needs more than a larger
///   one).
/// * DECLARED — the adapter rows: inference real-weight runs on the 3B backbone, 256², batch 1,
///   bf16 — MLX LoRA peak 15.3 GB, Candle LoRA peak 21.8 GB (decimal) — padded up (Candle adds the
///   2 GiB reserve); larger adapter rows scale the activation term conservatively.
///
/// A machine under its floor is refused before load, never admitted unbounded.
const IRIS_MEMORY_FLOORS_GIB: [(&str, &str, u32, IrisMemoryFloor); 18] = [
    ("mlx", "adapter", 256, IrisMemoryFloor::declared(16)),
    ("mlx", "adapter", 512, IrisMemoryFloor::declared(24)),
    ("mlx", "adapter", 1024, IrisMemoryFloor::declared(40)),
    ("mlx", "full", 256, MLX_FULL_256),
    ("mlx", "full", 512, MLX_FULL_512),
    ("mlx", "full", 1024, MLX_FULL_1024),
    (
        "mlx",
        "full_ac",
        256,
        IrisMemoryFloor::bound_from(MLX_FULL_256).bounded_by(MLX_FULL_AC_1024),
    ),
    (
        "mlx",
        "full_ac",
        512,
        IrisMemoryFloor::bound_from(MLX_FULL_512).bounded_by(MLX_FULL_AC_1024),
    ),
    ("mlx", "full_ac", 1024, MLX_FULL_AC_1024),
    ("candle", "adapter", 256, IrisMemoryFloor::declared(24)),
    ("candle", "adapter", 512, IrisMemoryFloor::declared(32)),
    ("candle", "adapter", 1024, IrisMemoryFloor::declared(48)),
    ("candle", "full", 256, CANDLE_FULL_256),
    ("candle", "full", 512, CANDLE_FULL_512),
    ("candle", "full", 1024, CANDLE_FULL_1024),
    (
        "candle",
        "full_ac",
        256,
        IrisMemoryFloor::bound_from(CANDLE_FULL_256).bounded_by(CANDLE_FULL_AC_512),
    ),
    ("candle", "full_ac", 512, CANDLE_FULL_AC_512),
    ("candle", "full_ac", 1024, CANDLE_FULL_AC_1024),
];

/// The memory floor (GiB) an Iris run of `network_type` at `resolution` needs on `lane` (`"mlx"`
/// unified memory, `"candle"` device VRAM). A resolution between two rows takes the next row up
/// (conservative); `None` above the largest row or for an unknown lane — which callers treat as a
/// refusal, never as "unbounded".
pub fn iris_training_memory_floor_gib(
    lane: &str,
    network_type: &str,
    resolution: u32,
) -> Option<u32> {
    iris_generation_memory_floor(lane, network_type, resolution, false).map(|floor| floor.gib)
}

/// Whether a run's `advanced.iris.activation_checkpointing` recomputes every region (`full`, or
/// `selective_op`, which natively enters and recomputes every region too) — the shape the
/// `full_ac` floor rows were measured at. `selective_layer` and `none` read the plain rows.
pub fn iris_full_activation_checkpointing(config: &TrainingConfig) -> bool {
    matches!(
        config
            .advanced
            .get(IRIS_OPTIONS_KEY)
            .and_then(|options| options.get("activation_checkpointing"))
            .and_then(Value::as_str),
        Some("full" | "selective_op")
    )
}

/// The floor row (GiB + provenance) of a generation run, with the run's activation checkpointing:
/// a FULL run that recomputes every region reads the `full_ac` rows.
pub fn iris_generation_memory_floor(
    lane: &str,
    network_type: &str,
    resolution: u32,
    full_activation_checkpointing: bool,
) -> Option<IrisMemoryFloor> {
    let artifact = if network_type.trim().eq_ignore_ascii_case("full") {
        if full_activation_checkpointing {
            "full_ac"
        } else {
            "full"
        }
    } else {
        "adapter"
    };
    IRIS_MEMORY_FLOORS_GIB
        .iter()
        .filter(|(row_lane, row_artifact, row_resolution, _)| {
            *row_lane == lane && *row_artifact == artifact && *row_resolution >= resolution
        })
        .min_by_key(|(_, _, row_resolution, _)| *row_resolution)
        .map(|(_, _, _, floor)| *floor)
}

/// The floor table as published on the target (`memoryFloors`), so the Studio can say what a run
/// needs — and what that number rests on — before launch.
fn memory_floors_json() -> Value {
    floors::memory_floors_json(
        None,
        "Keyed by artifact and training resolution; `full_ac` is a full fine-tune under full \
         activation checkpointing. Measured rows are the whole-process peak of a real full \
         fine-tune at batch 1, converted to GiB, plus a 2 GiB reserve. MLX peaks are the ACTIVE peak with MLX's buffer cache disabled; production runs keep the \
         cache, so a process footprint can rise to the allocator's GC ceiling (0.95 × the \
         recommended working set) without that being need. Candle peaks are the live pool peak over six optimizer steps (steady from step 2) on a 95.59 \
         GiB card; a row marked measured_spill is a need the run met only by spilling to system \
         memory, so it does not fit that card.",
        IRIS_MEMORY_FLOORS_GIB
            .iter()
            .map(|(lane, artifact, resolution, floor)| {
                (
                    *lane,
                    vec![(*artifact).to_owned(), resolution.to_string()],
                    *floor,
                )
            }),
    )
}

/// The Iris-3B generation training target: random-init / weights-init FULL training and LoRA /
/// LoKr adapters over the frozen upstream objective, on the MLX and Candle trainers.
pub fn iris_generation_target() -> TrainingTarget {
    let mut extra = ExtraFields::new();
    extra.insert("memoryFloors".to_owned(), memory_floors_json());
    TrainingTarget {
        id: IRIS_TRAINING_TARGET_ID.to_owned(),
        name: "Iris 3B".to_owned(),
        modality: TrainingModality::Image,
        // `networkType: "full"` turns a run into a base checkpoint (`training_output_kind`).
        output_kind: TrainingOutputKind::Lora,
        family: IRIS_FAMILY.to_owned(),
        base_model: IRIS_BASE_MODEL.to_owned(),
        base_model_repo: Some(IRIS_BASE_MODEL_REPO.to_owned()),
        kernel: IRIS_TRAINING_KERNEL.to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            // MICRO-steps (the shared unit); all four cadences are multiples of the default
            // accumulation of 1.
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            // Upstream's fixed-square `data.image_size`; 256 is the release's first stage and the
            // resolution every real-weight training run so far used.
            resolution: 256,
            save_every: 250,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                // bf16 autocast over f32 masters (upstream `mixed_precision: bf16`); `f32` is `no`.
                "mixedPrecision": "bf16",
                "networkType": "lora",
                "decomposeFactor": -1,
                // The backbone's own `flow.timestep_sampler` (logit-normal).
                "timestepType": "logit_normal",
                // The only bias/loss the objective has; pinned so the generic controls cannot
                // offer a refused value.
                "timestepBias": "balanced",
                "lossType": "mse",
                "weightDecay": 0.0,
                "lrScheduler": "constant",
                "lrWarmupSteps": 0,
                // Not implemented by the native trainer (it refuses `gradient_checkpointing`).
                "gradientCheckpointing": false,
                "sampleEvery": 250,
                "sampleSteps": 30,
                "sampleGuidanceScale": 3.0,
                "sampleCount": IRIS_PREVIEW_PROMPT_CAP,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto",
                // The engine `model_options["iris"]` bag. Values are upstream's defaults except
                // `text_conditioning` (bit-identical to on-the-fly, but frees the 8.9 GB encoder
                // before the optimizer state is built) and `keep_last_checkpoints` (a full-model
                // checkpoint is ~48 GB, so keep-all would fill a disk).
                "iris": {
                    "init": "weights",
                    "text_conditioning": "cached",
                    "text_dropout": 0.1,
                    "gradient_clip": 0.5,
                    "ema_enabled": true,
                    "ema_decay": 0.9999,
                    "on_caption_overflow": "warn",
                    "caption_field": "caption",
                    "caption_fields": [],
                    "keep_last_checkpoints": 2,
                    "milestone_steps": [],
                    "nan_loss_tolerance": 20,
                    "betas": [0.9, 0.95],
                    "auto_lr": "none",
                    "base_batch_size": 256,
                    "preview_negative_prompt": "",
                    "upstream_validation_prompts": false,
                    "export_dtype": "fp32"
                }
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [1, 128],
            "alpha": [1, 128],
            // The official curriculum's horizons are absolute optimizer steps (SFT ends at 665,000).
            "steps": [1, 5000000],
            "batchSize": [1, 64],
            // Multiples of the backbone's patch size (16); never silently changed by the trainer.
            "resolutions": [256, 512, 1024],
            // The trainer renders at most four preview prompts per cadence.
            "sampleCount": [1, IRIS_PREVIEW_PROMPT_CAP],
            "optimizers": ["adamw", "muon"],
            "networkTypes": ["lora", "lokr", "full"],
            "lrSchedulers": ["constant", "cosine"],
            "timestepTypes": ["logit_normal", "uniform"],
            "timestepBiases": ["balanced"],
            "lossTypes": ["mse"],
            // fp16 = accelerate's dynamic loss scaling (MLX and CUDA; Candle's CPU backend refuses it).
            "mixedPrecisions": ["bf16", "fp16", "f32"],
            "gradientCheckpointing": false,
            "microStepFields": ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
            "inits": ["weights", "random"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Iris 3B",
            "description": "Train Iris-3B generation: a LoRA or LoKr adapter, or a full fine-tune from \
                            the released weights or from random init. Pixel-space flow matching on \
                            the frozen Qwen3-VL text encoder, on Apple Silicon (MLX) and NVIDIA \
                            (Candle/CUDA).",
            "recommendedFor": ["character", "style", "concept"],
            "datasetModality": "image"
        })),
        extra,
    }
}

/// Every micro-step field of `fields` present in `config` must be a whole number of `multiple`
/// micro-steps (an optimizer-step boundary — or, for restoration, a generator + critic step).
fn validate_micro_steps(
    config: &TrainingConfig,
    multiple: u64,
    fields: &[(&str, bool)],
) -> Result<(), TrainingPlanError> {
    for (field, top_level) in fields {
        let value = if *top_level {
            Some(u64::from(match *field {
                "steps" => config.steps,
                "saveEvery" => config.save_every,
                _ => unreachable!("only steps/saveEvery are top-level micro-step fields"),
            }))
        } else {
            advanced_u64(config, field)?
        };
        if let Some(value) = value {
            if !value.is_multiple_of(multiple) {
                return Err(TrainingPlanError::TargetLimit(
                    TrainingTargetLimitError::NotMultipleOf {
                        field: (*field).to_owned(),
                        value,
                        multiple,
                    },
                ));
            }
        }
    }
    Ok(())
}

/// The per-objective dataset contract: generation items answer the caption selection; depth items
/// are aligned examples; restoration items are all pairs or all clean images.
pub fn validate_iris_dataset(
    target: &TrainingTarget,
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    match IrisObjective::from_base_model(&target.base_model) {
        Some(IrisObjective::Depth) => depth::validate_depth_dataset(dataset),
        Some(IrisObjective::Restoration) => restore::validate_restore_dataset(config, dataset),
        Some(IrisObjective::Generation) | None => validate_iris_dataset_captions(config, dataset),
    }
}

/// The store-boundary check of an item's Iris extras: named captions, a depth target, a degraded
/// input.
pub fn validate_item_iris_extras(extra: &ExtraFields) -> Result<(), String> {
    validate_item_captions(extra)?;
    depth::validate_item_depth(extra)?;
    restore::validate_item_degraded(extra)
}

fn invalid(detail: impl Into<String>) -> TrainingPlanError {
    TrainingPlanError::InvalidConfig(detail.into())
}

fn advanced_u64(config: &TrainingConfig, key: &str) -> Result<Option<u64>, TrainingPlanError> {
    match config.advanced.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| invalid(format!("{key} must be a non-negative integer."))),
    }
}

fn advanced_string<'a>(config: &'a TrainingConfig, key: &str) -> Option<&'a str> {
    config
        .advanced
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// The Iris options bag of a config (`advanced.iris`), or an empty map when absent.
pub fn iris_options(config: &TrainingConfig) -> Result<Map<String, Value>, TrainingPlanError> {
    match config.advanced.get(IRIS_OPTIONS_KEY) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err(invalid(
            "advanced.iris must be an object of Iris training options.",
        )),
    }
}

fn one_of(key: &str, value: &Value, allowed: &[&str]) -> Result<(), TrainingPlanError> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a string.")))?;
    if allowed.contains(&text) {
        Ok(())
    } else {
        Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedValue {
                field: format!("iris.{key}"),
                value: text.to_owned(),
                allowed: allowed.iter().map(|value| (*value).to_owned()).collect(),
            },
        ))
    }
}

fn finite_in(
    key: &str,
    value: &Value,
    accept: impl Fn(f64) -> bool,
    expect: &str,
) -> Result<f64, TrainingPlanError> {
    let number = value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a finite number.")))?;
    if accept(number) {
        Ok(number)
    } else {
        Err(invalid(format!("advanced.iris.{key} must be {expect}.")))
    }
}

fn whole(key: &str, value: &Value, minimum: u64) -> Result<u64, TrainingPlanError> {
    let number = value.as_u64().ok_or_else(|| {
        invalid(format!(
            "advanced.iris.{key} must be a non-negative integer."
        ))
    })?;
    if number < minimum {
        return Err(invalid(format!(
            "advanced.iris.{key} must be at least {minimum}."
        )));
    }
    Ok(number)
}

fn boolean(key: &str, value: &Value) -> Result<bool, TrainingPlanError> {
    value
        .as_bool()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be true or false.")))
}

fn string_list(key: &str, value: &Value) -> Result<Vec<String>, TrainingPlanError> {
    value
        .as_array()
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a list of strings.")))
}

/// Validate one `advanced.iris` value against the engine's accepted domain.
fn validate_option(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    match key {
        "init" => one_of(key, value, &["weights", "random"]),
        "prediction" => one_of(key, value, &["v", "x"]),
        "shift_law" => one_of(key, value, &["none", "sd3", "flux"]),
        "muon_adjust_lr" => one_of(key, value, &["rms_norm", "spectral_norm", "none"]),
        "auto_lr" => one_of(key, value, &["none", "sqrt", "linear"]),
        "on_caption_overflow" => one_of(key, value, &["warn", "error", "silent"]),
        "text_conditioning" => one_of(key, value, &["on_the_fly", "cached"]),
        "resume_data_policy" => one_of(key, value, &["exact", "new_phase"]),
        "preview_weights" | "export_weights" => one_of(key, value, &["raw", "ema"]),
        "export_dtype" => one_of(key, value, &["fp32", "bf16"]),
        "flow_shift" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "logit_mean" | "logit_std" | "x_pred_sigma_min" => {
            finite_in(key, value, |_| true, "a number").map(drop)
        }
        "text_dropout" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "gradient_clip" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "ema_decay" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "muon_momentum" => {
            finite_in(key, value, |x| (0.0..1.0).contains(&x), "in [0, 1)").map(drop)
        }
        "shift_base_tokens" | "base_batch_size" | "num_epochs" => whole(key, value, 1).map(drop),
        "nan_loss_tolerance" | "keep_last_checkpoints" => whole(key, value, 0).map(drop),
        "muon_nesterov" | "ema_enabled" | "override_lr_on_resume" => boolean(key, value).map(drop),
        "upstream_validation_prompts" => curricula::validate_upstream_validation_prompts(value),
        "caption_field" => {
            let field = value
                .as_str()
                .map(str::trim)
                .filter(|field| !field.is_empty())
                .ok_or_else(|| {
                    invalid("advanced.iris.caption_field must be a non-empty string.")
                })?;
            if field.len() > 64 {
                return Err(invalid(
                    "advanced.iris.caption_field is longer than 64 characters.",
                ));
            }
            Ok(())
        }
        "caption_fields" => {
            let fields = string_list(key, value)?;
            if fields
                .iter()
                .any(|field| field.trim().is_empty() || field.len() > 64)
            {
                return Err(invalid(
                    "advanced.iris.caption_fields entries must be non-empty names of at most 64 \
                     characters.",
                ));
            }
            Ok(())
        }
        "preview_negative_prompt" => value
            .as_str()
            .map(drop)
            .ok_or_else(|| invalid("advanced.iris.preview_negative_prompt must be a string.")),
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
        _ => curricula::validate_curricula_option(key, value).unwrap_or_else(|| {
            Err(invalid(format!(
                "advanced.iris.{key} is not an Iris training option."
            )))
        }),
    }
}

/// Submit-time validation of an Iris training config: the target-local refusals the trainer would
/// otherwise raise only after loading the weights, as typed 400s.
pub fn validate_iris_training_config(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    match IrisObjective::from_base_model(&target.base_model) {
        Some(IrisObjective::Depth) => return depth::validate_depth_config(config),
        Some(IrisObjective::Restoration) => return restore::validate_restore_config(config),
        Some(IrisObjective::Generation) | None => {}
    }
    // ---- micro-step multiples -----------------------------------------------------------------
    validate_micro_steps(
        config,
        u64::from(config.gradient_accumulation.max(1)),
        &IRIS_MICRO_STEP_FIELDS,
    )?;

    // ---- the options bag ----------------------------------------------------------------------
    let options = iris_options(config)?;
    for (key, value) in &options {
        if IRIS_API_OWNED_OPTION_KEYS.contains(&key.as_str()) {
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
        if !IRIS_CLIENT_OPTION_KEYS.contains(&key.as_str())
            && !curricula::IRIS_CURRICULA_OPTION_KEYS.contains(&key.as_str())
        {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: "iris".to_owned(),
                    value: key.clone(),
                    allowed: IRIS_CLIENT_OPTION_KEYS
                        .iter()
                        .chain(curricula::IRIS_CURRICULA_OPTION_KEYS.iter())
                        .map(|key| (*key).to_owned())
                        .collect(),
                },
            ));
        }
        if !value.is_null() {
            validate_option(key, value)?;
        }
    }

    let network_type = advanced_string(config, "networkType").unwrap_or("lora");
    let full = network_type.eq_ignore_ascii_case("full");
    let init = options
        .get("init")
        .and_then(Value::as_str)
        .unwrap_or("weights");
    if init == "random" && !full {
        return Err(invalid(
            "Random initialization trains a full model; an Iris adapter (LoRA/LoKr) trains on the \
             released weights. Choose the full fine-tune network type or weights init.",
        ));
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY) {
        if !full {
            return Err(invalid(
                "Starting from a previous run's weights (loadFromJobId) applies to a full \
                 fine-tune only.",
            ));
        }
        if init == "random" {
            return Err(invalid(
                "loadFromJobId starts from a previous run's weights; it cannot be combined with \
                 random initialization.",
            ));
        }
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY)
        && config.advanced.contains_key(RESUME_FROM_JOB_KEY)
    {
        return Err(invalid(
            "Resume continues a run's full state; loadFromJobId starts a fresh run from its \
             weights. Choose one.",
        ));
    }
    let ema_enabled = options
        .get("ema_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    for key in ["preview_weights", "export_weights"] {
        if options.get(key).and_then(Value::as_str) == Some("ema") && !ema_enabled {
            return Err(invalid(format!(
                "advanced.iris.{key} = ema needs the EMA enabled (advanced.iris.ema_enabled)."
            )));
        }
    }

    // ---- previews -----------------------------------------------------------------------------
    let upstream_prompts = curricula::upstream_prompts_on(&options);
    let prompt_count = config
        .advanced
        .get("samplePrompts")
        .and_then(Value::as_array)
        .map_or(0, |prompts| {
            prompts
                .iter()
                .filter(|prompt| {
                    prompt
                        .as_str()
                        .is_some_and(|prompt| !prompt.trim().is_empty())
                })
                .count()
        });
    if upstream_prompts && prompt_count > 0 {
        return Err(invalid(
            "Upstream validation prompts render Iris' seven default prompts in place of your own \
             sample prompts — clear the sample prompts or turn the upstream prompts off.",
        ));
    }
    let sample_every = advanced_u64(config, "sampleEvery")?.unwrap_or(0);
    let sample_steps = advanced_u64(config, "sampleSteps")?.unwrap_or(20);
    if sample_every > 0 && (prompt_count > 0 || upstream_prompts) && sample_steps == 0 {
        return Err(invalid(
            "sampleSteps must be at least 1 when previews are on.",
        ));
    }
    let previews_on = sample_every > 0 && (prompt_count > 0 || upstream_prompts);
    curricula::validate_curricula_config(config, &options, previews_on)?;

    // ---- the resume knobs only mean something on a resume --------------------------------------
    if !config.advanced.contains_key(RESUME_FROM_JOB_KEY) {
        for key in ["resume_data_policy", "override_lr_on_resume"] {
            if options.get(key).is_some_and(|value| !value.is_null()) {
                return Err(invalid(format!(
                    "advanced.iris.{key} applies only when resuming a previous run \
                     (advanced.{RESUME_FROM_JOB_KEY})."
                )));
            }
        }
    }
    let _ = target;
    Ok(())
}

/// Validate a dataset item's per-item named captions (`extra.captions = {field: text}`).
pub fn validate_item_captions(extra: &ExtraFields) -> Result<(), String> {
    let Some(value) = extra.get(ITEM_CAPTIONS_KEY) else {
        return Ok(());
    };
    let map = value
        .as_object()
        .ok_or_else(|| "Item captions must be an object of {field: caption text}.".to_owned())?;
    for (field, text) in map {
        let trimmed = field.trim();
        if trimmed.is_empty() || trimmed != field || field.len() > 64 {
            return Err(format!(
                "Item caption field names must be non-empty, untrimmed-whitespace-free, and at most \
                 64 characters (got {field:?})."
            ));
        }
        if !text.is_string() {
            return Err(format!("Item caption field {field:?} must be text."));
        }
    }
    Ok(())
}

fn item_has_caption_field(item: &super::TrainingDatasetItem, field: &str) -> bool {
    let named = item
        .extra
        .get(ITEM_CAPTIONS_KEY)
        .and_then(Value::as_object)
        .and_then(|captions| captions.get(field))
        .and_then(Value::as_str)
        .is_some_and(|text| !text.trim().is_empty());
    named || (field == "caption" && !item.caption.text.trim().is_empty())
}

/// Every item must be able to answer the run's caption selection, or the trainer would stop
/// mid-epoch on the first item without it (`select_caption`). With `caption_fields` set an item needs
/// one of them OR the `caption_field` fallback; otherwise it needs `caption_field` itself
/// (`"caption"` reads the item's ordinary caption).
pub fn validate_iris_dataset_captions(
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_captions(&item.extra).map_err(invalid)?;
    }
    let options = iris_options(config)?;
    let caption_field = options
        .get("caption_field")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .unwrap_or("caption");
    let caption_fields: Vec<String> = options
        .get("caption_fields")
        .and_then(Value::as_array)
        .map(|fields| {
            fields
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let missing: Vec<&str> = dataset
        .items
        .iter()
        .filter(|item| {
            let any_listed = caption_fields
                .iter()
                .any(|field| item_has_caption_field(item, field));
            !(any_listed || item_has_caption_field(item, caption_field))
        })
        .map(|item| item.display_name.as_str())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let wanted = if caption_fields.is_empty() {
        format!("caption field '{caption_field}'")
    } else {
        format!(
            "any of caption fields {:?} (or the fallback '{caption_field}')",
            caption_fields
        )
    };
    Err(invalid(format!(
        "{} dataset item(s) have no {wanted}: {}{}. Add the caption or pick a field every item \
         carries.",
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

    #[test]
    fn the_client_allow_list_is_the_engine_list_minus_the_path_options() {
        // The engine's `OPTION_KEYS` (33) + `OPTION_KEYS_EXTRA` (2) = 35 keys; SceneWorks owns the
        // two path keys, the client may set the other 33 — no more, no fewer.
        assert_eq!(
            IRIS_CLIENT_OPTION_KEYS.len() + IRIS_API_OWNED_OPTION_KEYS.len(),
            35
        );
        for key in IRIS_API_OWNED_OPTION_KEYS {
            assert!(!IRIS_CLIENT_OPTION_KEYS.contains(&key));
        }
        let mut sorted = IRIS_CLIENT_OPTION_KEYS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), IRIS_CLIENT_OPTION_KEYS.len(), "no duplicates");
    }

    #[test]
    fn a_floor_row_covers_every_offered_lane_artifact_and_resolution() {
        let target = iris_generation_target();
        let resolutions: Vec<u32> = target.limits["resolutions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        for lane in ["mlx", "candle"] {
            for network in ["lora", "lokr", "full"] {
                for resolution in &resolutions {
                    assert!(
                        iris_training_memory_floor_gib(lane, network, *resolution).is_some(),
                        "{lane}/{network}/{resolution} has no floor row"
                    );
                }
            }
        }
        // The declared adapter rows, and full far above adapters.
        assert_eq!(iris_training_memory_floor_gib("mlx", "lora", 256), Some(16));
        assert_eq!(
            iris_training_memory_floor_gib("candle", "lokr", 256),
            Some(24)
        );
        assert!(
            iris_training_memory_floor_gib("mlx", "full", 256).unwrap()
                > iris_training_memory_floor_gib("mlx", "lora", 1024).unwrap()
        );
        // Between rows takes the next row up; beyond the table and unknown lanes are refusals.
        assert_eq!(iris_training_memory_floor_gib("mlx", "lora", 384), Some(24));
        assert_eq!(iris_training_memory_floor_gib("mlx", "lora", 2048), None);
        assert_eq!(iris_training_memory_floor_gib("cpu", "lora", 256), None);
        assert_eq!(
            target.extra["memoryFloors"]["lanes"]["candle"]["adapter"]["256"],
            json!(24)
        );
    }

    /// sc-25690: the shared gradient-checkpointing toggle is a depth / restoration control on the
    /// lanes whose trainer implements it — both native lanes at this pin; a lane outside the table
    /// is refused typed, naming where it works.
    #[test]
    fn task_gradient_checkpointing_is_refused_on_a_lane_that_lacks_it() {
        let mut config = depth::iris_depth_target().defaults;
        for objective in [IrisObjective::Depth, IrisObjective::Restoration] {
            assert_eq!(
                iris_gradient_checkpointing_lanes(objective),
                ["candle", "mlx"]
            );
            // Off: nothing to refuse on any lane.
            for lane in ["mlx", "candle", "cpu"] {
                assert_eq!(
                    iris_gradient_checkpointing_refusal(objective, lane, &config),
                    None
                );
            }
        }
        config
            .advanced
            .insert("gradientCheckpointing".into(), json!(true));
        assert!(iris_gradient_checkpointing(&config));
        for objective in [IrisObjective::Depth, IrisObjective::Restoration] {
            for lane in ["mlx", "candle", "cpu"] {
                let refusal = iris_gradient_checkpointing_refusal(objective, lane, &config);
                assert_eq!(
                    refusal.is_none(),
                    iris_gradient_checkpointing_lanes(objective).contains(&lane),
                    "{objective:?} on {lane}: {refusal:?}"
                );
            }
            let refusal =
                iris_gradient_checkpointing_refusal(objective, "cpu", &config).expect("refused");
            assert!(
                refusal.contains(objective.task())
                    && refusal.contains("training on cpu")
                    && refusal
                        .contains("implemented on NVIDIA (Candle/CUDA) and Apple Silicon (MLX)"),
                "{refusal}"
            );
        }
        // Generation's control is `iris.activation_checkpointing`; this gate is not its.
        assert!(iris_gradient_checkpointing_lanes(IrisObjective::Generation).is_empty());
        assert_eq!(
            iris_gradient_checkpointing_refusal(IrisObjective::Generation, "mlx", &config),
            None
        );
    }

    /// sc-25690: provenance is PER ROW and everything published derives from it — the measured
    /// full rows say measured, the Candle 1024² row no card here holds says extrapolated, the
    /// adapter rows say declared; there is no table-wide `measured` flag to disagree with them.
    #[test]
    fn every_floor_row_publishes_its_own_provenance() {
        use floors::FloorProvenance::{Declared, Measured, MeasuredSpill};
        let floor = |lane, artifact, resolution| {
            IRIS_MEMORY_FLOORS_GIB
                .iter()
                .find(|row| row.0 == lane && row.1 == artifact && row.2 == resolution)
                .map(|row| row.3)
                .unwrap()
        };
        for lane in ["mlx", "candle"] {
            for resolution in [256, 512, 1024] {
                assert_eq!(floor(lane, "adapter", resolution).provenance, Declared);
            }
            assert_eq!(floor(lane, "full", 512).provenance, Measured, "{lane}");
            assert_eq!(floor(lane, "full_ac", 1024).provenance, Measured, "{lane}");
            // Bounds borrowed from another shape's measurement are never called measured.
            assert_ne!(floor(lane, "full_ac", 256).provenance, Measured, "{lane}");
            assert_ne!(floor(lane, "full", 256).provenance, Measured, "{lane}");
        }
        assert_ne!(floor("mlx", "full_ac", 512).provenance, Measured);
        assert_eq!(floor("candle", "full_ac", 512).provenance, Measured);
        assert_eq!(floor("mlx", "full", 1024).provenance, Measured);
        // Candle 1024² without checkpointing was measured, but only by spilling off the card.
        assert_eq!(floor("candle", "full", 1024).provenance, MeasuredSpill);

        let published = &iris_generation_target().extra["memoryFloors"];
        assert_eq!(published["unit"], json!("GiB"));
        assert!(published.get("measured").is_none(), "{published:#}");
        for (lane, artifact, resolution, row) in IRIS_MEMORY_FLOORS_GIB {
            assert_eq!(
                published["provenance"][lane][artifact][resolution.to_string()],
                json!(row.provenance.as_str()),
                "{lane} {artifact} {resolution}"
            );
            assert_eq!(
                published["lanes"][lane][artifact][resolution.to_string()],
                json!(row.gib)
            );
        }
        let basis = published["basis"].as_str().unwrap();
        assert!(basis.contains("Floors are GiB"), "{basis}");
        assert!(
            basis.contains("completed only by spilling to system memory): candle full 1024."),
            "{basis}"
        );
    }

    /// sc-25690: the measured rows are the probe peaks converted from the unit the probe printed
    /// (decimal GB ÷ 1.073741824), plus the Candle reserve, rounded UP — and a floor never shrinks
    /// as the resolution grows, nor does checkpointing ever raise one.
    #[test]
    fn measured_rows_are_the_converted_peaks_and_the_table_is_monotonic() {
        use floors::{ceil_gib, DECIMAL_GB_PER_GIB};
        let gib = |peak: Peak| match peak {
            Peak::DecimalGb(gb) => gb / DECIMAL_GB_PER_GIB,
            other => other.gib(),
        };
        let mlx = |peak: Peak| ceil_gib(gib(peak) + f64::from(MLX_RESERVE_GIB));
        assert_eq!(MLX_FULL_512.gib, mlx(MLX_FULL_512_PEAK));
        assert_eq!(MLX_FULL_1024.gib, mlx(MLX_FULL_1024_PEAK));
        assert_eq!(MLX_FULL_AC_1024.gib, mlx(MLX_FULL_AC_1024_PEAK));
        let card = CAMPAIGN_CUDA_CARD.gib() as u32;
        assert_eq!(
            CANDLE_FULL_512.gib,
            ceil_gib(gib(CANDLE_FULL_512_PEAK) + f64::from(CANDLE_RESERVE_GIB)).min(card)
        );
        assert_eq!(
            CANDLE_FULL_AC_1024.gib,
            ceil_gib(gib(CANDLE_FULL_AC_1024_PEAK) + f64::from(CANDLE_RESERVE_GIB)).min(card)
        );
        assert_eq!(
            CANDLE_FULL_AC_512.gib,
            ceil_gib(gib(CANDLE_FULL_AC_512_PEAK) + f64::from(CANDLE_RESERVE_GIB)).min(card)
        );
        // Candle 1024² without checkpointing did not fit the campaign card: its row is the measured
        // need, above the card, never capped to it.
        assert_eq!(
            CANDLE_FULL_1024.gib,
            ceil_gib(gib(CANDLE_FULL_1024_SPILL_PEAK) + f64::from(CANDLE_RESERVE_GIB))
        );
        assert!(CANDLE_FULL_1024.gib > card, "{CANDLE_FULL_1024:?}");
        for lane in ["mlx", "candle"] {
            for artifact in ["adapter", "full", "full_ac"] {
                let rows: Vec<u32> = [256, 512, 1024]
                    .into_iter()
                    .map(|resolution| {
                        iris_generation_memory_floor(
                            lane,
                            if artifact == "adapter" {
                                "lora"
                            } else {
                                "full"
                            },
                            resolution,
                            artifact == "full_ac",
                        )
                        .unwrap()
                        .gib
                    })
                    .collect();
                assert!(
                    rows.windows(2).all(|pair| pair[0] <= pair[1]),
                    "{lane} {artifact}: {rows:?}"
                );
            }
            for resolution in [256, 512, 1024] {
                let plain = iris_generation_memory_floor(lane, "full", resolution, false).unwrap();
                let checkpointed =
                    iris_generation_memory_floor(lane, "full", resolution, true).unwrap();
                assert!(checkpointed.gib <= plain.gib, "{lane} {resolution}");
            }
        }
    }

    /// sc-25690: a full 1024² run under full activation checkpointing reads the measured `full_ac`
    /// row (it fits a 96 GB card on Candle), one without reads the higher plain row, and an adapter
    /// run ignores the policy.
    #[test]
    fn full_runs_read_the_activation_checkpointing_floor_rows() {
        let mut config = iris_generation_target().defaults;
        config.resolution = 1024;
        config.advanced.insert("networkType".into(), json!("full"));
        assert!(!iris_full_activation_checkpointing(&config));
        let plain = iris_objective_memory_floor(IrisObjective::Generation, "candle", &config);
        config.advanced["iris"]["activation_checkpointing"] = json!("full");
        assert!(iris_full_activation_checkpointing(&config));
        let checkpointed =
            iris_objective_memory_floor(IrisObjective::Generation, "candle", &config);
        assert_eq!(
            (plain, checkpointed),
            (Some(CANDLE_FULL_1024), Some(CANDLE_FULL_AC_1024))
        );
        assert!(checkpointed.unwrap().gib < plain.unwrap().gib);
        assert!(
            f64::from(checkpointed.unwrap().gib) <= CAMPAIGN_CUDA_CARD.gib(),
            "the checkpointed 1024² row fits the card that ran it"
        );
        assert_eq!(
            iris_objective_memory_floor(IrisObjective::Generation, "mlx", &config),
            Some(MLX_FULL_AC_1024)
        );
        config.advanced["iris"]["activation_checkpointing"] = json!("selective_layer");
        assert!(!iris_full_activation_checkpointing(&config));
        assert_eq!(
            iris_generation_memory_floor("mlx", "lora", 1024, true).map(|floor| floor.gib),
            iris_training_memory_floor_gib("mlx", "lora", 1024)
        );
        assert_eq!(
            iris_generation_target().extra["memoryFloors"]["lanes"]["mlx"]["full_ac"]["1024"],
            json!(MLX_FULL_AC_1024.gib)
        );
    }
}
