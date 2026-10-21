//! Iris-3B curricula, representation alignment, validation and distributed execution — the
//! SceneWorks half of the S11 trainer contract (story sc-25689, epic sc-25678).
//!
//! The engine contract is `gen_core::iris::train` at the pinned inference revision: the options bag
//! (`model_options["iris"]`) gains the curriculum (`stage`, `parent_stage`, `run_name`, `tracking`),
//! data-shape (`shape_policy`, `aspect_ratio_bucket`, `shape_align`, `shape_max_ratio`), RoPE law
//! (`rope_aspect`), representation alignment (`repa`), activation checkpointing
//! (`activation_checkpointing`, `ac_selective_every`), holdout validation (`val_every_steps`,
//! `val_samples`), logging (`log_every`, `log_block_grad_norms`), retention
//! (`save_every_epochs`), resume (`resume_optimizer`), global-batch (`expected_global_batch`,
//! `scale_warmup_by_world`), `perf.lazy_sync`, preview (`preview_at_step_one`, `preview_order`) and
//! distribution (`dist`) keys. This module owns which of them a CLIENT may set and the domain each
//! accepts, so the Training Studio offers exactly what the trainer honors (E3):
//!
//! * offered: every key above. NOT offered (and refused by [`super::validate_iris_training_config`]):
//!   `data_type` (`shape_policy` carries the same choice), `architecture` / `text_encoder` /
//!   `num_train_timesteps` (resource identity checks of a translated upstream YAML — the provisioned
//!   backbone IS the architecture), the `perf` torch flags other than `lazy_sync` (mapped: they never
//!   change the update), `dist.ddp.*` (DDP bucketing has no numeric effect), and every
//!   `dist.world` key except `world_size`, `share_device` and the three wait bounds
//!   (`timeout_secs`, `rendezvous_timeout_secs`, `cancel_grace_secs`; each at least 1 s, as the
//!   engine refuses 0) — rank numbering, rendezvous and transport belong to the worker that
//!   launches the ranks.
//! * `repa` offers the release teacher only: `dinov2_vitb14` from the separately installed,
//!   pinned `facebook/dinov2-base` training resource ([`IRIS_REPA_TEACHER_MODEL_ID`]). The teacher
//!   path is never client-authored (the worker stages it as the `repa_teacher` component).
//! * holdout items are dataset items whose `extra.split` is `"holdout"` (the engine's
//!   `TrainingItem.model_options.split`); they never enter the training walk.
//! * the declared memory floor of a run adds the REPA teacher and multiplies by the ranks that
//!   share one device ([`iris_run_memory_floor`]).

use serde_json::{Map, Value};

use super::floors::IrisMemoryFloor;
use super::{
    boolean, finite_in, invalid, iris_full_activation_checkpointing, iris_generation_memory_floor,
    iris_options, one_of, string_list, whole, RESUME_FROM_JOB_KEY,
};
use crate::training::{
    TrainingConfig, TrainingDataset, TrainingPlanError, TrainingTargetLimitError,
};

/// The S11 option keys a client may set (the engine's `OPTION_KEYS_S11` minus `data_type`,
/// `architecture`, `text_encoder` and `num_train_timesteps`).
pub const IRIS_CURRICULA_OPTION_KEYS: [&str; 24] = [
    "stage",
    "parent_stage",
    "run_name",
    "tracking",
    "shape_policy",
    "aspect_ratio_bucket",
    "shape_align",
    "shape_max_ratio",
    "rope_aspect",
    "repa",
    "activation_checkpointing",
    "ac_selective_every",
    "expected_global_batch",
    "scale_warmup_by_world",
    "dist",
    "resume_optimizer",
    "save_every_epochs",
    "log_every",
    "log_block_grad_norms",
    "val_every_steps",
    "val_samples",
    "perf",
    "preview_at_step_one",
    "preview_order",
];

/// The engine's S11 keys SceneWorks deliberately does not offer (see the module docs).
pub const IRIS_CURRICULA_UNOFFERED_KEYS: [&str; 4] = [
    "data_type",
    "architecture",
    "text_encoder",
    "num_train_timesteps",
];

/// The four official generation stages and the stage each continues (`upstream::STAGE_PRESETS`).
pub const IRIS_STAGES: [(&str, Option<&str>); 4] = [
    ("stage1_256", None),
    ("stage2_512", Some("stage1_256")),
    ("stage3_1024", Some("stage2_512")),
    ("sft_1024", Some("stage3_1024")),
];

/// `shapes::ShapePolicy` names.
pub const SHAPE_POLICIES: [&str; 3] = ["fixed", "bucket", "area"];
/// `shapes::BUCKET_TABLES`.
pub const BUCKET_TABLES: [&str; 2] = ["shared21-512", "shared21-1024"];
/// `ac::POLICIES`.
pub const AC_POLICIES: [&str; 4] = ["none", "full", "selective_op", "selective_layer"];
/// The image RoPE laws (`rope_aspect`).
pub const ROPE_ASPECTS: [&str; 2] = ["square", "isotropic"];
/// `repa.variant`.
pub const REPA_VARIANTS: [&str; 2] = ["repa", "irepa"];
/// The `repa` keys a client may set: the release teacher is fixed (`teacher_source: hub`,
/// `teacher_hub: facebookresearch/dinov2`, `teacher: dinov2_vitb14`, `teacher_dim: 768`,
/// `teacher_patch_size: 14`) and its path is the staged `repa_teacher` component.
pub const REPA_CLIENT_KEYS: [&str; 7] = [
    "weight",
    "variant",
    "spatial_norm_gamma",
    "proj_hidden_dim",
    "teacher_image_size",
    "teacher_match_student",
    "layer",
];
/// `dist.strategy`.
pub const DIST_STRATEGIES: [&str; 2] = ["ddp", "fsdp2"];
/// `dist.sharding`.
pub const DIST_SHARDINGS: [&str; 4] = ["full", "grad_op", "hybrid", "hybrid_grad_op"];
/// `dist.param_dtype`.
pub const DIST_PARAM_DTYPES: [&str; 3] = ["bf16", "fp16", "fp32"];
/// `dist.reduce_dtype` (the engine refuses an fp16 reduction).
pub const DIST_REDUCE_DTYPES: [&str; 2] = ["fp32", "bf16"];
/// `dist.checkpoint_io` on one host (`per_node` is `rank0` on a single node, so it is not offered).
pub const DIST_CHECKPOINT_IO: [&str; 3] = ["auto", "rank0", "dcp"];
/// The `dist` keys a client may set.
pub const DIST_CLIENT_KEYS: [&str; 8] = [
    "strategy",
    "sharding",
    "mesh",
    "param_dtype",
    "reduce_dtype",
    "reshard_after_forward",
    "checkpoint_io",
    "world",
];
/// The `dist.world` keys a client may set; the launching worker owns every other world key.
pub const DIST_WORLD_CLIENT_KEYS: [&str; 5] = [
    "world_size",
    "share_device",
    "timeout_secs",
    "rendezvous_timeout_secs",
    "cancel_grace_secs",
];
/// The `dist.world` wait bounds a client may set (seconds). The engine refuses `0` for each
/// (`gen_core::iris::train::dist`, sc-25690: a zero cancel grace would end the cancel agreement's
/// own wait at its first poll), so each is a whole number of at least 1.
pub const DIST_WORLD_WAIT_KEYS: [&str; 3] = [
    "timeout_secs",
    "rendezvous_timeout_secs",
    "cancel_grace_secs",
];
/// The most ranks one SceneWorks job launches.
pub const MAX_IRIS_RANKS: u64 = 16;
/// `dist::DEFAULT_TIMEOUT_SECS` (upstream's process-group timeout).
pub const DEFAULT_DIST_TIMEOUT_SECS: u64 = 5400;

/// The dataset-item extra key (and engine `TrainingItem.model_options` key) assigning an item to the
/// holdout set (`gen_core::iris::train::holdout::ITEM_SPLIT_KEY`).
pub const ITEM_SPLIT_KEY: &str = "split";
pub const ITEM_SPLIT_HOLDOUT: &str = "holdout";
pub const ITEM_SPLIT_TRAIN: &str = "train";

/// The catalog id of the REPA teacher TRAINING resource (`facebook/dinov2-base`, the Hugging Face
/// mirror of the release teacher `dinov2_vitb14`). Installed separately in the Model Manager, never
/// part of any inference closure.
pub const IRIS_REPA_TEACHER_MODEL_ID: &str = "iris_repa_teacher_dinov2_base";
/// `gen_core::iris::train::repa::DINOV2_BASE_HF_REPO` / `DINOV2_BASE_HF_REVISION`.
pub const IRIS_REPA_TEACHER_REPO: &str = "facebook/dinov2-base";
pub const IRIS_REPA_TEACHER_REVISION: &str = "f9e44c814b77203eaa57a6bdbbd535f21ede1415";
/// `gen_core::iris::train::repa::TEACHER_WEIGHTS_FILE`.
pub const IRIS_REPA_TEACHER_FILE: &str = "model.safetensors";
/// The pinned file's exact size and sha256 (the Hugging Face LFS pointer at the pinned revision).
pub const IRIS_REPA_TEACHER_SIZE_BYTES: u64 = 346_345_912;
pub const IRIS_REPA_TEACHER_SHA256: &str =
    "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841";
/// The only teacher the provisioned resource is.
pub const IRIS_REPA_TEACHER_ENTRYPOINT: &str = "dinov2_vitb14";

/// Declared (NOT measured) extra GiB a REPA run holds on top of its run floor: the frozen fp32
/// DINOv2-B/14 (0.35 GB), the projector + its AdamW state and the teacher's activations. Evidence:
/// the MLX stage-1 curriculum with REPA (full model, 256²) peaked at 68.2 GB, inside the 80 GB full
/// floor; an adapter run's floor has no such slack, so the teacher is added explicitly.
pub const REPA_EXTRA_FLOOR_GIB: u32 = 4;

fn validate_stage_id(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a string.")))?;
    let ok = !text.is_empty()
        && text.len() <= 64
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if ok {
        Ok(())
    } else {
        Err(invalid(format!(
            "advanced.iris.{key} must be a stage id of at most 64 letters, digits, '_', '-' or '.'."
        )))
    }
}

fn object<'a>(key: &str, value: &'a Value) -> Result<&'a Map<String, Value>, TrainingPlanError> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be an object.")))
}

fn refuse_unknown(
    path: &str,
    map: &Map<String, Value>,
    known: &[&str],
) -> Result<(), TrainingPlanError> {
    for key in map.keys() {
        if !known.contains(&key.as_str()) {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: format!("iris.{path}"),
                    value: key.clone(),
                    allowed: known.iter().map(|key| (*key).to_owned()).collect(),
                },
            ));
        }
    }
    Ok(())
}

fn validate_tracking(value: &Value) -> Result<(), TrainingPlanError> {
    let map = object("tracking", value)?;
    refuse_unknown("tracking", map, &["project", "tags"])?;
    if let Some(project) = map.get("project").filter(|v| !v.is_null()) {
        let ok = project.as_str().is_some_and(|text| text.len() <= 200);
        if !ok {
            return Err(invalid(
                "advanced.iris.tracking.project must be text of at most 200 characters.",
            ));
        }
    }
    if let Some(tags) = map.get("tags").filter(|v| !v.is_null()) {
        let tags = string_list("tracking.tags", tags)?;
        if tags.len() > 32
            || tags
                .iter()
                .any(|tag| tag.trim().is_empty() || tag.len() > 64)
        {
            return Err(invalid(
                "advanced.iris.tracking.tags must be at most 32 non-empty tags of at most 64 \
                 characters.",
            ));
        }
    }
    Ok(())
}

fn validate_repa(value: &Value) -> Result<(), TrainingPlanError> {
    let map = object("repa", value)?;
    refuse_unknown("repa", map, &REPA_CLIENT_KEYS)?;
    for (key, value) in map {
        if value.is_null() {
            continue;
        }
        let path = format!("repa.{key}");
        match key.as_str() {
            "weight" => finite_in(&path, value, |x| x >= 0.0, "at least 0").map(drop)?,
            "variant" => one_of(&path, value, &REPA_VARIANTS)?,
            "spatial_norm_gamma" => finite_in(&path, value, |_| true, "a number").map(drop)?,
            "proj_hidden_dim" => whole(&path, value, 1).map(drop)?,
            // One teacher patch (14 px) is the smallest teacher grid.
            "teacher_image_size" => whole(&path, value, 14).map(drop)?,
            "teacher_match_student" => boolean(&path, value).map(drop)?,
            "layer" => whole(&path, value, 1).map(drop)?,
            _ => unreachable!("refuse_unknown admitted only REPA_CLIENT_KEYS"),
        }
    }
    Ok(())
}

fn validate_perf(value: &Value) -> Result<(), TrainingPlanError> {
    let map = object("perf", value)?;
    refuse_unknown("perf", map, &["lazy_sync"])?;
    if let Some(lazy) = map.get("lazy_sync").filter(|v| !v.is_null()) {
        boolean("perf.lazy_sync", lazy)?;
    }
    Ok(())
}

fn validate_dist(value: &Value) -> Result<(), TrainingPlanError> {
    let map = object("dist", value)?;
    refuse_unknown("dist", map, &DIST_CLIENT_KEYS)?;
    for (key, value) in map {
        if value.is_null() {
            continue;
        }
        let path = format!("dist.{key}");
        match key.as_str() {
            "strategy" => one_of(&path, value, &DIST_STRATEGIES)?,
            "sharding" => one_of(&path, value, &DIST_SHARDINGS)?,
            "param_dtype" => one_of(&path, value, &DIST_PARAM_DTYPES)?,
            "reduce_dtype" => one_of(&path, value, &DIST_REDUCE_DTYPES)?,
            "checkpoint_io" => one_of(&path, value, &DIST_CHECKPOINT_IO)?,
            "reshard_after_forward" => boolean(&path, value).map(drop)?,
            "mesh" => {
                let mesh = object("dist.mesh", value)?;
                refuse_unknown("dist.mesh", mesh, &["dp_shard", "dp_replicate"])?;
                for (axis, size) in mesh {
                    if !size.is_null() {
                        whole(&format!("dist.mesh.{axis}"), size, 0)?;
                    }
                }
            }
            "world" => {
                let world = object("dist.world", value)?;
                refuse_unknown("dist.world", world, &DIST_WORLD_CLIENT_KEYS)?;
                if let Some(size) = world.get("world_size").filter(|v| !v.is_null()) {
                    let size = whole("dist.world.world_size", size, 1)?;
                    if size > MAX_IRIS_RANKS {
                        return Err(invalid(format!(
                            "advanced.iris.dist.world.world_size must be at most {MAX_IRIS_RANKS}."
                        )));
                    }
                }
                if let Some(share) = world.get("share_device").filter(|v| !v.is_null()) {
                    boolean("dist.world.share_device", share)?;
                }
                for key in DIST_WORLD_WAIT_KEYS {
                    if let Some(bound) = world.get(key).filter(|v| !v.is_null()) {
                        whole(&format!("dist.world.{key}"), bound, 1)?;
                    }
                }
            }
            _ => unreachable!("refuse_unknown admitted only DIST_CLIENT_KEYS"),
        }
    }
    Ok(())
}

/// Validate one S11 option value against the domain the engine accepts. `None` when `key` is not
/// an S11 key (the caller's own validation then decides).
pub(super) fn validate_curricula_option(
    key: &str,
    value: &Value,
) -> Option<Result<(), TrainingPlanError>> {
    let result = match key {
        "stage" | "parent_stage" => validate_stage_id(key, value),
        "run_name" => value
            .as_str()
            .filter(|text| !text.trim().is_empty() && text.len() <= 200)
            .map(drop)
            .ok_or_else(|| {
                invalid("advanced.iris.run_name must be non-empty text of at most 200 characters.")
            }),
        "tracking" => validate_tracking(value),
        "shape_policy" => one_of(key, value, &SHAPE_POLICIES),
        "aspect_ratio_bucket" => one_of(key, value, &BUCKET_TABLES),
        "shape_align" => whole(key, value, 1).map(drop),
        "shape_max_ratio" => finite_in(key, value, |x| x >= 1.0, "at least 1").map(drop),
        "rope_aspect" => one_of(key, value, &ROPE_ASPECTS),
        "repa" => validate_repa(value),
        "activation_checkpointing" => one_of(key, value, &AC_POLICIES),
        "ac_selective_every" | "expected_global_batch" | "log_every" | "preview_order" => {
            whole(key, value, 1).map(drop)
        }
        "val_every_steps" | "val_samples" | "save_every_epochs" => whole(key, value, 0).map(drop),
        "scale_warmup_by_world" | "log_block_grad_norms" | "preview_at_step_one" => {
            boolean(key, value).map(drop)
        }
        "resume_optimizer" => one_of(key, value, &["full", "core"]),
        "perf" => validate_perf(value),
        "dist" => validate_dist(value),
        _ => return None,
    };
    Some(result)
}

/// `upstream_validation_prompts`: `true`/`false` (upstream's seven defaults) or a translated
/// config's own prompt list (rendered uncapped).
pub(super) fn validate_upstream_validation_prompts(value: &Value) -> Result<(), TrainingPlanError> {
    match value {
        Value::Bool(_) => Ok(()),
        Value::Array(prompts) => {
            if prompts.is_empty()
                || prompts.len() > 64
                || prompts
                    .iter()
                    .any(|prompt| prompt.as_str().is_none_or(|text| text.trim().is_empty()))
            {
                Err(invalid(
                    "advanced.iris.upstream_validation_prompts must be true/false or a list of 1–64 \
                     non-empty prompts.",
                ))
            } else {
                Ok(())
            }
        }
        _ => Err(invalid(
            "advanced.iris.upstream_validation_prompts must be true/false or a list of prompts.",
        )),
    }
}

/// Whether `upstream_validation_prompts` turns upstream's prompts on (true or a list).
pub(super) fn upstream_prompts_on(options: &Map<String, Value>) -> bool {
    match options.get("upstream_validation_prompts") {
        Some(Value::Bool(on)) => *on,
        Some(Value::Array(prompts)) => !prompts.is_empty(),
        _ => false,
    }
}

/// The topology a config asks for: `(world_size, share_device, timeout_secs)`.
pub fn iris_dist_world(options: &Map<String, Value>) -> (u64, bool, u64) {
    let world = options
        .get("dist")
        .and_then(|dist| dist.get("world"))
        .and_then(Value::as_object);
    let world_size = world
        .and_then(|world| world.get("world_size"))
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1);
    let share_device = world
        .and_then(|world| world.get("share_device"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let timeout = world
        .and_then(|world| world.get("timeout_secs"))
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_DIST_TIMEOUT_SECS);
    (world_size, share_device, timeout)
}

/// The world size a config asks for (1 when it names none).
pub fn iris_world_size(config: &TrainingConfig) -> u64 {
    iris_options(config)
        .map(|options| iris_dist_world(&options).0)
        .unwrap_or(1)
}

/// Whether the config trains with REPA (`repa.weight > 0`).
pub fn iris_repa_enabled(options: &Map<String, Value>) -> bool {
    options
        .get("repa")
        .and_then(|repa| repa.get("weight"))
        .and_then(Value::as_f64)
        .is_some_and(|weight| weight > 0.0)
}

fn opt_str<'a>(options: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    options.get(key).and_then(Value::as_str)
}

fn present(options: &Map<String, Value>, key: &str) -> bool {
    options.get(key).is_some_and(|value| !value.is_null())
}

/// The global batch of a config (`batch_size × world_size × gradient_accumulation`) — the engine's
/// `dist::effective_batch`.
pub fn iris_global_batch(config: &TrainingConfig, world_size: u64) -> u64 {
    u64::from(config.batch_size.max(1))
        * world_size
        * u64::from(config.gradient_accumulation.max(1))
}

/// Cross-field validation of the S11 options: every knob that would be refused by the trainer or
/// would do nothing in this run is a typed 400 here.
pub(super) fn validate_curricula_config(
    config: &TrainingConfig,
    options: &Map<String, Value>,
    previews_on: bool,
) -> Result<(), TrainingPlanError> {
    let resuming = config.advanced.contains_key(RESUME_FROM_JOB_KEY);

    // ---- curriculum lineage ------------------------------------------------------------------
    if present(options, "parent_stage") && !present(options, "stage") {
        return Err(invalid(
            "advanced.iris.parent_stage names the stage this run continues; set advanced.iris.stage \
             too.",
        ));
    }
    if let (Some(stage), Some(parent)) =
        (opt_str(options, "stage"), opt_str(options, "parent_stage"))
    {
        if let Some((_, official)) = IRIS_STAGES.iter().find(|(id, _)| *id == stage) {
            if *official != Some(parent) {
                return Err(invalid(format!(
                    "advanced.iris.parent_stage: the official stage {stage} continues {}, not \
                     {parent}.",
                    official.unwrap_or("nothing (it is the first stage)")
                )));
            }
        }
        if stage == parent {
            return Err(invalid(
                "advanced.iris.parent_stage must differ from the stage.",
            ));
        }
    }
    if !resuming && present(options, "resume_optimizer") {
        return Err(invalid(format!(
            "advanced.iris.resume_optimizer applies only when resuming a previous run \
             (advanced.{RESUME_FROM_JOB_KEY})."
        )));
    }

    // ---- data shapes -------------------------------------------------------------------------
    let policy = opt_str(options, "shape_policy").unwrap_or("fixed");
    if present(options, "aspect_ratio_bucket") && policy != "bucket" {
        return Err(invalid(
            "advanced.iris.aspect_ratio_bucket applies to the bucket shape policy only.",
        ));
    }
    for key in ["shape_align", "shape_max_ratio"] {
        if present(options, key) && policy != "area" {
            return Err(invalid(format!(
                "advanced.iris.{key} applies to the area shape policy only."
            )));
        }
    }

    // ---- REPA --------------------------------------------------------------------------------
    if let Some(repa) = options.get("repa").and_then(Value::as_object) {
        let on = repa
            .get("weight")
            .and_then(Value::as_f64)
            .is_some_and(|weight| weight > 0.0);
        if !on
            && repa
                .iter()
                .any(|(key, value)| key != "weight" && !value.is_null())
        {
            return Err(invalid(
                "REPA is off (advanced.iris.repa.weight is 0): its other settings would do nothing. \
                 Set a weight or clear them.",
            ));
        }
        if repa.get("variant").and_then(Value::as_str) != Some("irepa")
            && repa.get("spatial_norm_gamma").is_some_and(|v| !v.is_null())
        {
            return Err(invalid(
                "advanced.iris.repa.spatial_norm_gamma applies to the irepa variant only.",
            ));
        }
    }

    // ---- activation checkpointing ------------------------------------------------------------
    if present(options, "ac_selective_every")
        && opt_str(options, "activation_checkpointing") != Some("selective_layer")
    {
        return Err(invalid(
            "advanced.iris.ac_selective_every applies to activation_checkpointing = \
             selective_layer only.",
        ));
    }

    // ---- validation / previews ---------------------------------------------------------------
    let val_every = options
        .get("val_every_steps")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if present(options, "val_samples") && val_every == 0 {
        return Err(invalid(
            "advanced.iris.val_samples applies only when holdout validation runs \
             (val_every_steps > 0).",
        ));
    }
    if !previews_on {
        for key in ["preview_at_step_one", "preview_order"] {
            if present(options, key) {
                return Err(invalid(format!(
                    "advanced.iris.{key} shapes the training previews, which are off for this run \
                     (no sample prompts, no upstream validation prompts, or sampleEvery = 0)."
                )));
            }
        }
    }

    // ---- distribution ------------------------------------------------------------------------
    let dist = options.get("dist").and_then(Value::as_object);
    let strategy = dist
        .and_then(|dist| dist.get("strategy"))
        .and_then(Value::as_str)
        .unwrap_or("ddp");
    let sharding = dist
        .and_then(|dist| dist.get("sharding"))
        .and_then(Value::as_str)
        .unwrap_or("full");
    let mesh = dist
        .and_then(|dist| dist.get("mesh"))
        .and_then(Value::as_object);
    let mesh_axis = |axis: &str| {
        mesh.and_then(|mesh| mesh.get(axis))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let (dp_shard, dp_replicate) = (mesh_axis("dp_shard"), mesh_axis("dp_replicate"));
    let (world_size, _, _) = iris_dist_world(options);
    if strategy == "ddp" {
        if sharding != "full" {
            return Err(invalid(format!(
                "advanced.iris.dist.sharding applies to fsdp2 only; strategy ddp shards nothing \
                 (got {sharding})."
            )));
        }
        if dp_shard != 0 || dp_replicate != 0 {
            return Err(invalid("advanced.iris.dist.mesh applies to fsdp2 only."));
        }
        if dist
            .and_then(|dist| dist.get("checkpoint_io"))
            .and_then(Value::as_str)
            == Some("dcp")
        {
            return Err(invalid(
                "advanced.iris.dist.checkpoint_io = dcp writes per-rank shards and needs the \
                 sharded strategy (fsdp2).",
            ));
        }
        for key in ["param_dtype", "reshard_after_forward"] {
            if dist
                .and_then(|dist| dist.get(key))
                .is_some_and(|v| !v.is_null())
            {
                return Err(invalid(format!(
                    "advanced.iris.dist.{key} applies to the sharded strategy (fsdp2) only."
                )));
            }
        }
    } else {
        if options
            .get("log_block_grad_norms")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(invalid(
                "advanced.iris.log_block_grad_norms needs whole per-parameter gradients; fsdp2 \
                 holds only local shards. Use ddp or turn the block norms off.",
            ));
        }
        if opt_str(options, "resume_optimizer") == Some("core") {
            return Err(invalid(
                "advanced.iris.dist.strategy = fsdp2 requires resume_optimizer = full.",
            ));
        }
        let hybrid = sharding.starts_with("hybrid");
        if !hybrid {
            if !(dp_shard == 0 || dp_shard == world_size) || dp_replicate > 1 {
                return Err(invalid(format!(
                    "sharding = {sharding} shards over every rank, so dist.mesh.dp_shard must be 0 \
                     or {world_size} and dist.mesh.dp_replicate 0 or 1 (got {dp_shard} × \
                     {dp_replicate}); use hybrid sharding to replicate across groups."
                )));
            }
        } else {
            if dp_shard == 0 {
                return Err(invalid(
                    "hybrid sharding needs dist.mesh.dp_shard (the ranks one sharded copy spans).",
                ));
            }
            let replicate = if dp_replicate == 0 {
                world_size / dp_shard
            } else {
                dp_replicate
            };
            if replicate * dp_shard != world_size {
                return Err(invalid(format!(
                    "The mesh does not tile the world: dp_replicate {replicate} × dp_shard \
                     {dp_shard} ≠ {world_size} ranks."
                )));
            }
        }
    }
    if let Some(expected) = options.get("expected_global_batch").and_then(Value::as_u64) {
        let actual = iris_global_batch(config, world_size);
        if actual != expected {
            return Err(invalid(format!(
                "Global batch {actual} (batch size {} × {world_size} rank(s) × gradient \
                 accumulation {}) does not match the expected global batch {expected}. Change the \
                 batch size, the ranks or the accumulation, or clear the expected global batch.",
                config.batch_size.max(1),
                config.gradient_accumulation.max(1)
            )));
        }
    }
    Ok(())
}

/// Validate a dataset item's `split` extra (`"train"` | `"holdout"`).
pub fn validate_item_split(extra: &crate::training::ExtraFields) -> Result<(), String> {
    match extra.get(ITEM_SPLIT_KEY) {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(split)) if split == ITEM_SPLIT_TRAIN || split == ITEM_SPLIT_HOLDOUT => {
            Ok(())
        }
        Some(other) => Err(format!(
            "Item split must be \"{ITEM_SPLIT_TRAIN}\" or \"{ITEM_SPLIT_HOLDOUT}\" (got {other})."
        )),
    }
}

/// Whether a dataset item is in the holdout set.
pub fn is_holdout_item(item: &crate::training::TrainingDatasetItem) -> bool {
    item.extra.get(ITEM_SPLIT_KEY).and_then(Value::as_str) == Some(ITEM_SPLIT_HOLDOUT)
}

/// The holdout contract of a run's dataset: every split value is valid, something is left to train
/// on, and holdout validation (`val_every_steps > 0`) has holdout items to evaluate.
pub fn validate_iris_holdout(
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_split(&item.extra).map_err(invalid)?;
    }
    let holdout = dataset
        .items
        .iter()
        .filter(|item| is_holdout_item(item))
        .count();
    if holdout == dataset.items.len() && holdout > 0 {
        return Err(invalid(
            "Every dataset item is marked holdout, so nothing is left to train on.",
        ));
    }
    let val_every = iris_options(config)?
        .get("val_every_steps")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if val_every > 0 && holdout == 0 {
        return Err(invalid(
            "Holdout validation (val_every_steps) needs at least one dataset item marked holdout.",
        ));
    }
    Ok(())
}

/// The floor row (GiB + provenance) of ONE rank of a generation run: its artifact × resolution row
/// plus the REPA teacher when alignment is on. `None` when no row covers the shape (refuse).
pub fn iris_rank_memory_floor(lane: &str, config: &TrainingConfig) -> Option<IrisMemoryFloor> {
    iris_run_floor(lane, config, 1)
}

/// The run's floor row with `ranks` ranks on one device (the REPA teacher added per rank).
fn iris_run_floor(lane: &str, config: &TrainingConfig, ranks: u64) -> Option<IrisMemoryFloor> {
    let network = config
        .advanced
        .get("networkType")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("lora");
    let base = iris_generation_memory_floor(
        lane,
        network,
        config.resolution,
        iris_full_activation_checkpointing(config),
    )?;
    let options = iris_options(config).ok()?;
    let extra = if iris_repa_enabled(&options) {
        REPA_EXTRA_FLOOR_GIB
    } else {
        0
    };
    base.for_run(u32::try_from(ranks).ok()?, extra)
}

/// How many ranks share each device when `world_size` ranks run on `devices` devices (ranks are
/// spread round-robin, so the busiest device holds the ceiling).
pub fn ranks_per_device(world_size: u64, devices: u64) -> u64 {
    world_size.max(1).div_ceil(devices.max(1))
}

/// The floor row (GiB + provenance) one DEVICE needs for a run: one rank's floor times the ranks
/// sharing the busiest device. Only a single rank without the teacher IS a table row; anything
/// else is arithmetic on one, so a measured row becomes an extrapolation. Never admitted unbounded:
/// `None` refuses.
pub fn iris_run_memory_floor(
    lane: &str,
    config: &TrainingConfig,
    devices: u64,
) -> Option<IrisMemoryFloor> {
    iris_run_floor(
        lane,
        config,
        ranks_per_device(iris_world_size(config), devices),
    )
}

/// How many accelerators a worker with `gpu_id` owns (`"1"` → 1, `"0,1"` → 2, `"mlx"` → 1; the
/// CPU utility worker owns none).
pub fn worker_device_count(gpu_id: &str) -> u64 {
    gpu_id
        .split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty() && *id != "cpu" && *id != "auto")
        .count() as u64
}

/// The most devices any one registered GPU worker owns — the largest world a job can get without
/// `share_device`, since a job runs on the one worker that claims it. `1` when no GPU worker is
/// registered (every SceneWorks GPU worker owns at least one device).
pub fn max_worker_devices<'a>(gpu_ids: impl IntoIterator<Item = &'a str>) -> u64 {
    gpu_ids
        .into_iter()
        .map(worker_device_count)
        .max()
        .unwrap_or(0)
        .max(1)
}

/// The topology refusal of a run on a worker that sees `devices` accelerators: more ranks than
/// devices need the explicit `dist.world.share_device` opt-in (several ranks on one device — the
/// same math, shared memory and compute). `None` = admissible.
pub fn iris_topology_refusal(
    config: &TrainingConfig,
    devices: u64,
    device_noun: &str,
) -> Option<String> {
    let options = iris_options(config).ok()?;
    let (world_size, share_device, _) = iris_dist_world(&options);
    (world_size > devices.max(1) && !share_device).then(|| {
        format!(
            "This run asks for {world_size} ranks but the worker that runs it owns {} {device_noun}\
             {}. Each rank needs its own {device_noun}, or turn on \"share device\" \
             (dist.world.share_device) to run several ranks on one {device_noun} — the same \
             training, with the memory and compute shared.",
            devices.max(1),
            if devices.max(1) == 1 { "" } else { "s" }
        )
    })
}

/// The receipts an Iris S11 run records on its job and on the registered output: the curriculum
/// stage/parent, the data identity inputs and the topology. Engine-resolved values (the resolved
/// data identity, the dist summary) are added by the API from the engine plan.
pub fn iris_curriculum_receipt(config: &TrainingConfig) -> Value {
    let options = iris_options(config).unwrap_or_default();
    let (world_size, share_device, timeout) = iris_dist_world(&options);
    let dist = options.get("dist").and_then(Value::as_object);
    let dist_str = |key: &str, default: &str| {
        dist.and_then(|dist| dist.get(key))
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_owned()
    };
    serde_json::json!({
        "stage": options.get("stage").cloned().unwrap_or(Value::Null),
        "parentStage": options.get("parent_stage").cloned().unwrap_or(Value::Null),
        "parentRunJobId": config.advanced.get(RESUME_FROM_JOB_KEY).cloned().unwrap_or(Value::Null),
        "runName": options.get("run_name").cloned().unwrap_or(Value::Null),
        "shapePolicy": options.get("shape_policy").cloned().unwrap_or(Value::String("fixed".into())),
        "repa": iris_repa_enabled(&options).then(|| options.get("repa").cloned()).flatten(),
        "activationCheckpointing": options
            .get("activation_checkpointing")
            .cloned()
            .unwrap_or(Value::String("none".into())),
        "dist": {
            "strategy": dist_str("strategy", "ddp"),
            "sharding": dist_str("sharding", "full"),
            "worldSize": world_size,
            "shareDevice": share_device,
            "timeoutSecs": timeout,
            "globalBatch": iris_global_batch(config, world_size),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::ExtraFields;
    use serde_json::json;

    fn config_with(iris: Value) -> TrainingConfig {
        let mut config = crate::training::iris::iris_generation_target().defaults;
        config.advanced.insert("iris".into(), iris);
        config
    }

    fn check(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
        crate::training::iris::validate_iris_training_config(
            &crate::training::iris::iris_generation_target(),
            config,
        )
    }

    #[test]
    fn every_offered_curricula_key_is_validated_and_the_unoffered_are_refused() {
        for key in IRIS_CURRICULA_OPTION_KEYS {
            assert!(
                validate_curricula_option(key, &Value::Null).is_some(),
                "{key} has no validator"
            );
        }
        for key in IRIS_CURRICULA_UNOFFERED_KEYS {
            let config = config_with(json!({ key: "pixel" }));
            let error = check(&config).expect_err(key);
            assert!(
                format!("{error}").contains(key)
                    || matches!(error, TrainingPlanError::TargetLimit(_)),
                "{key}"
            );
        }
    }

    #[test]
    fn curricula_values_outside_the_engine_domain_are_refused() {
        for bad in [
            json!({"shape_policy": "square"}),
            json!({"shape_policy": "bucket", "aspect_ratio_bucket": "shared21-2048"}),
            json!({"rope_aspect": "diagonal"}),
            json!({"activation_checkpointing": "block"}),
            json!({"repa": {"weight": -1.0}}),
            json!({"repa": {"weight": 0.5, "teacher": "dinov2_vitl14"}}),
            json!({"repa": {"weight": 0.5, "teacher_weights": "/etc/x"}}),
            json!({"perf": {"compile": true}}),
            json!({"dist": {"strategy": "fsdp"}}),
            json!({"dist": {"world": {"rank": 1}}}),
            json!({"dist": {"world": {"world_size": 99}}}),
            // sc-25690: the engine refuses a zero wait bound (a zero cancel grace would end the
            // cancel agreement's own wait at its first poll).
            json!({"dist": {"world": {"timeout_secs": 0}}}),
            json!({"dist": {"world": {"rendezvous_timeout_secs": 0}}}),
            json!({"dist": {"world": {"cancel_grace_secs": 0}}}),
            json!({"dist": {"world": {"cancel_grace_secs": 1.5}}}),
            json!({"dist": {"ddp": {"bucket_cap_mb": 50}}}),
            json!({"dist": {"reduce_dtype": "fp16"}}),
            json!({"stage": "../x"}),
        ] {
            assert!(
                check(&config_with(bad.clone())).is_err(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn settings_that_would_do_nothing_are_refused() {
        for (bad, why) in [
            (
                json!({"aspect_ratio_bucket": "shared21-512"}),
                "bucket table without bucket policy",
            ),
            (
                json!({"shape_policy": "bucket", "shape_align": 32}),
                "align without area",
            ),
            (
                json!({"repa": {"weight": 0.0, "variant": "irepa"}}),
                "repa settings with weight 0",
            ),
            (
                json!({"repa": {"weight": 0.5, "spatial_norm_gamma": 0.5}}),
                "gamma without irepa",
            ),
            (
                json!({"ac_selective_every": 3}),
                "every without selective_layer",
            ),
            (json!({"val_samples": 4}), "val_samples without validation"),
            (
                json!({"resume_optimizer": "core"}),
                "resume_optimizer without a resume",
            ),
            (
                json!({"parent_stage": "stage1_256"}),
                "parent without stage",
            ),
            (
                json!({"stage": "stage2_512", "parent_stage": "stage3_1024"}),
                "wrong official parent",
            ),
            (
                json!({"dist": {"sharding": "hybrid"}}),
                "sharding under ddp",
            ),
            (json!({"dist": {"checkpoint_io": "dcp"}}), "dcp under ddp"),
            (
                json!({"dist": {"param_dtype": "fp32"}}),
                "param_dtype under ddp",
            ),
            (
                json!({"dist": {"strategy": "fsdp2"}, "log_block_grad_norms": true}),
                "block norms under fsdp2",
            ),
            (
                json!({"dist": {"strategy": "fsdp2", "sharding": "hybrid", "mesh": {"dp_shard": 3}, "world": {"world_size": 4, "share_device": true}}}),
                "a mesh that does not tile",
            ),
            (
                json!({"expected_global_batch": 1024}),
                "global batch mismatch",
            ),
        ] {
            assert!(check(&config_with(bad.clone())).is_err(), "{why}: {bad}");
        }
        // The previews knobs need previews on; the default target has no sample prompts.
        let mut quiet = config_with(json!({"preview_at_step_one": true}));
        quiet.advanced.insert("sampleEvery".into(), json!(0));
        assert!(check(&quiet).is_err());
    }

    #[test]
    fn a_full_s11_request_passes_and_the_global_batch_rule_is_exact() {
        let mut config = config_with(json!({
            "stage": "stage2_512",
            "parent_stage": "stage1_256",
            "run_name": "iris stage 2",
            "tracking": {"project": "iris", "tags": ["s2"]},
            "shape_policy": "bucket",
            "aspect_ratio_bucket": "shared21-512",
            "rope_aspect": "isotropic",
            "repa": {"weight": 0.5, "variant": "irepa", "spatial_norm_gamma": 0.7, "layer": 10},
            "activation_checkpointing": "selective_layer",
            "ac_selective_every": 2,
            "val_every_steps": 50,
            "val_samples": 8,
            "log_every": 10,
            "save_every_epochs": 1,
            "expected_global_batch": 4,
            "scale_warmup_by_world": true,
            "perf": {"lazy_sync": true},
            "dist": {"strategy": "fsdp2", "sharding": "full", "world": {"world_size": 2, "share_device": true, "timeout_secs": 600, "rendezvous_timeout_secs": 300, "cancel_grace_secs": 60}},
        }));
        config.batch_size = 2;
        config.resolution = 512;
        check(&config).expect("a complete S11 request is accepted");
        assert_eq!(iris_global_batch(&config, 2), 4);
        config.gradient_accumulation = 2;
        config.steps = 1000;
        config.save_every = 250;
        config.advanced.insert("sampleEvery".into(), json!(250));
        assert!(
            check(&config).is_err(),
            "accumulation 2 makes the global batch 8 ≠ 4"
        );
    }

    #[test]
    fn holdout_items_are_validated_against_the_validation_cadence() {
        let item = |id: &str, split: Option<&str>| {
            let mut extra = ExtraFields::new();
            if let Some(split) = split {
                extra.insert(ITEM_SPLIT_KEY.into(), json!(split));
            }
            serde_json::from_value::<crate::training::TrainingDatasetItem>(json!({
                "id": id, "path": format!("images/{id}.png"), "displayName": id,
                "caption": {"text": "a cat", "source": "manual", "triggerWords": []}, "addedAt": "x",
            }))
            .map(|mut item| {
                item.extra = extra;
                item
            })
            .unwrap()
        };
        let dataset = |items| -> TrainingDataset {
            let mut dataset: TrainingDataset = serde_json::from_value(json!({
                "schemaVersion": 1, "id": "ds", "name": "ds", "version": 1, "modality": "image",
                "status": "ready", "items": [], "createdAt": "x", "updatedAt": "x",
            }))
            .unwrap();
            dataset.items = items;
            dataset
        };
        let validating = config_with(json!({"val_every_steps": 10}));
        let plain = config_with(json!({}));
        assert!(validate_iris_holdout(&validating, &dataset(vec![item("a", None)])).is_err());
        validate_iris_holdout(
            &validating,
            &dataset(vec![item("a", None), item("b", Some("holdout"))]),
        )
        .unwrap();
        assert!(validate_iris_holdout(&plain, &dataset(vec![item("a", Some("holdout"))])).is_err());
        assert!(validate_iris_holdout(&plain, &dataset(vec![item("a", Some("val"))])).is_err());
        validate_iris_holdout(&plain, &dataset(vec![item("a", Some("train"))])).unwrap();
    }

    #[test]
    fn the_largest_world_without_sharing_is_the_biggest_workers_device_count() {
        assert_eq!(worker_device_count("1"), 1);
        assert_eq!(worker_device_count("0, 1"), 2);
        assert_eq!(worker_device_count("mlx"), 1);
        assert_eq!(worker_device_count("cpu"), 0);
        assert_eq!(max_worker_devices(["cpu", "0", "1"]), 1);
        assert_eq!(max_worker_devices(["cpu", "0,1", "2"]), 2);
        assert_eq!(max_worker_devices([]), 1);
        assert_eq!(max_worker_devices(["cpu"]), 1);
    }

    #[test]
    fn the_memory_floor_adds_the_teacher_and_multiplies_by_ranks_per_device() {
        use super::super::floors::FloorProvenance;
        let plain = config_with(json!({}));
        let base = iris_rank_memory_floor("mlx", &plain).unwrap();
        assert_eq!(
            Some(base),
            iris_generation_memory_floor("mlx", "lora", plain.resolution, false)
        );
        let repa = config_with(json!({"repa": {"weight": 0.5}}));
        assert_eq!(
            iris_rank_memory_floor("mlx", &repa).map(|floor| floor.gib),
            Some(base.gib + REPA_EXTRA_FLOOR_GIB)
        );
        let two = config_with(json!({"dist": {"world": {"world_size": 2, "share_device": true}}}));
        let one_rank = iris_rank_memory_floor("candle", &two).unwrap().gib;
        assert_eq!(
            iris_run_memory_floor("candle", &two, 1).map(|floor| floor.gib),
            Some(2 * one_rank)
        );
        assert_eq!(
            iris_run_memory_floor("candle", &two, 2).map(|floor| floor.gib),
            Some(one_rank)
        );
        // sc-25690: a measured row times two ranks, or plus the teacher, is arithmetic on a
        // measurement — never reported as measured; one rank without the teacher is the row.
        let mut full = config_with(json!({}));
        full.resolution = 512;
        full.advanced.insert("networkType".into(), json!("full"));
        let row = iris_run_memory_floor("mlx", &full, 1).unwrap();
        assert_eq!(row.provenance, FloorProvenance::Measured);
        full.advanced["iris"]["repa"] = json!({"weight": 0.5});
        let with_teacher = iris_run_memory_floor("mlx", &full, 1).unwrap();
        assert_eq!(with_teacher.gib, row.gib + REPA_EXTRA_FLOOR_GIB);
        assert_eq!(with_teacher.provenance, FloorProvenance::Extrapolated);
        assert_eq!(ranks_per_device(3, 2), 2);
        assert!(iris_topology_refusal(&two, 1, "GPU").is_none());
        let unshared = config_with(json!({"dist": {"world": {"world_size": 2}}}));
        let refusal = iris_topology_refusal(&unshared, 1, "GPU").expect("refused");
        assert!(refusal.contains("share device"), "{refusal}");
        assert!(iris_topology_refusal(&unshared, 2, "GPU").is_none());
    }
}
