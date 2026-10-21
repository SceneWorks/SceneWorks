//! Worker-side seams of Iris-3B curricula, REPA, holdout validation and fp16 (story sc-25689, epic
//! sc-25678). Everything here reads the ENGINE's own contract (`gen_core::iris::train` at the
//! pinned inference revision), so SceneWorks never re-derives what the trainer decides:
//!
//! * [`iris_stage_presets`] — the four official stages (`stage1_256 → stage2_512 → stage3_1024 →
//!   sft_1024`) translated by `upstream::stage_preset_config` and normalized onto the SceneWorks
//!   offered surface (`sceneworks_core::training::iris::curricula`): editable, carrying their stage,
//!   parent stage and data-shaping identity;
//! * [`resolve_iris_plan`] — `IrisTrainPlan::resolve` of a SceneWorks plan against the installed
//!   backbone's `config.yaml`: the engine's typed refusals (an invalid mesh, a global-batch mismatch,
//!   a REPA capture block outside the trunk, a shape policy the patch cannot tile, …) before
//!   anything loads, and the resolved receipts (data identity, topology, shape policy) recorded on
//!   the run;
//! * the REPA teacher TRAINING resource — `facebook/dinov2-base` at the engine's pinned revision,
//!   installed separately in the Model Manager, checked (exact size + sha256) before every run and
//!   staged as the `repa_teacher` load component ([`repa_teacher_component`]); never downloaded at
//!   training time and never part of an inference closure;
//! * fp16 — the Iris trainers run accelerate's dynamic loss scaling, so `mixedPrecision: fp16`
//!   reaches the engine verbatim ([`apply_iris_train_dtype`]) instead of the shared f32 fallback.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use sceneworks_core::training::iris::curricula::{
    self, IRIS_REPA_TEACHER_FILE, IRIS_REPA_TEACHER_REPO, IRIS_REPA_TEACHER_REVISION,
    IRIS_REPA_TEACHER_SHA256, IRIS_REPA_TEACHER_SIZE_BYTES,
};
use sceneworks_core::training::iris::{is_iris_training_kernel, IRIS_OPTIONS_KEY};
use sceneworks_core::training::TrainingPlan;

/// The pinned `(size, sha256)` of the teacher file. Tests stage a small stand-in teacher through
/// [`test_teacher_integrity`]; production always checks the real pin.
fn pinned_teacher_integrity() -> (u64, String) {
    #[cfg(test)]
    if let Some(pin) = TEST_TEACHER_PIN.lock().unwrap().clone() {
        return pin;
    }
    (
        IRIS_REPA_TEACHER_SIZE_BYTES,
        IRIS_REPA_TEACHER_SHA256.to_owned(),
    )
}

#[cfg(test)]
static TEST_TEACHER_PIN: std::sync::Mutex<Option<(u64, String)>> = std::sync::Mutex::new(None);
/// Serializes the tests that depend on the pin (the override is process-wide).
#[cfg(test)]
static TEST_TEACHER_PIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Pin the teacher integrity to `bytes` for the guard's lifetime (tests only); `None` holds the
/// real pin. Tests that read the pin serialize on the guard.
#[cfg(test)]
pub(crate) fn test_teacher_integrity(bytes: Option<&[u8]>) -> TestTeacherPin {
    use sha2::{Digest, Sha256};
    let lock = TEST_TEACHER_PIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *TEST_TEACHER_PIN.lock().unwrap() =
        bytes.map(|bytes| (bytes.len() as u64, format!("{:x}", Sha256::digest(bytes))));
    TestTeacherPin { _lock: lock }
}

#[cfg(test)]
pub(crate) struct TestTeacherPin {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for TestTeacherPin {
    fn drop(&mut self) {
        *TEST_TEACHER_PIN.lock().unwrap() = None;
    }
}

/// The pinned snapshot directory of the REPA teacher in the app's model source library, when its
/// weights file is present at the exact pinned size (the cheap install check the API uses; the
/// worker additionally hashes the file before a run).
pub fn installed_repa_teacher(data_dir: &Path) -> Option<PathBuf> {
    let dir = crate::model_jobs::huggingface_pinned_snapshot_dir(
        data_dir,
        IRIS_REPA_TEACHER_REPO,
        IRIS_REPA_TEACHER_REVISION,
    )?;
    let size = std::fs::metadata(dir.join(IRIS_REPA_TEACHER_FILE))
        .ok()
        .map(|meta| meta.len())?;
    (size == pinned_teacher_integrity().0).then_some(dir)
}

/// The integrity check of a staged teacher file: the exact pinned size, then the pinned sha256.
pub fn verify_repa_teacher_file(path: &Path) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let (pinned_size, pinned_sha256) = pinned_teacher_integrity();
    let size = std::fs::metadata(path)
        .map_err(|error| format!("the REPA teacher {} is unreadable: {error}", path.display()))?
        .len();
    if size != pinned_size {
        return Err(format!(
            "the REPA teacher {} is {size} bytes, not the pinned {pinned_size} \
             ({IRIS_REPA_TEACHER_REPO} @ {IRIS_REPA_TEACHER_REVISION}); reinstall it in the Model \
             Manager",
            path.display()
        ));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("the REPA teacher {} is unreadable: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("reading the REPA teacher failed: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != pinned_sha256 {
        return Err(format!(
            "the REPA teacher {} has sha256 {digest}, not the pinned {pinned_sha256}; reinstall \
             it in the Model Manager",
            path.display()
        ));
    }
    Ok(())
}

/// The `repa_teacher` load component of an Iris run that trains with REPA: the verified pinned
/// teacher snapshot, or a typed refusal naming the Model Manager install. `None` for any other run.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn repa_teacher_component(
    settings: &crate::settings::Settings,
    plan: &TrainingPlan,
) -> crate::error::WorkerResult<Option<gen_core::WeightsSource>> {
    if !is_iris_training_kernel(&plan.target.kernel) {
        return Ok(None);
    }
    let options = sceneworks_core::training::iris::iris_options(&plan.config)
        .map_err(|error| crate::error::WorkerError::InvalidPayload(error.to_string()))?;
    if !curricula::iris_repa_enabled(&options) {
        return Ok(None);
    }
    let dir = installed_repa_teacher(&settings.data_dir).ok_or_else(|| {
        crate::error::WorkerError::InvalidPayload(format!(
            "REPA needs the DINOv2 teacher ({IRIS_REPA_TEACHER_REPO} @ {IRIS_REPA_TEACHER_REVISION}) \
             — install \"Iris REPA Teacher — DINOv2-B/14\" in the Model Manager first. Nothing is \
             downloaded during training."
        ))
    })?;
    verify_repa_teacher_file(&dir.join(IRIS_REPA_TEACHER_FILE))
        .map_err(crate::error::WorkerError::InvalidPayload)?;
    Ok(Some(gen_core::WeightsSource::Dir(dir)))
}

/// The engine `train_dtype` of an Iris run: `fp16` reaches the trainer verbatim (accelerate's
/// dynamic loss scaling, `MixedPrecision::Fp16`); every other kernel keeps the shared mapping.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn apply_iris_train_dtype(
    mut config: gen_core::TrainingConfig,
    plan: &TrainingPlan,
) -> gen_core::TrainingConfig {
    if is_iris_training_kernel(&plan.target.kernel)
        && plan
            .config
            .advanced
            .get("mixedPrecision")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("fp16"))
    {
        config.train_dtype = "fp16".to_owned();
    }
    config
}

/// A typed refusal of an Iris plan by the engine's own resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct IrisPlanRefusal {
    /// `dist` failures keep their kind (`invalid_mesh`, `global_batch_mismatch`, `split_world`,
    /// `rank_lost`, `desync`, `unsupported`); everything else is `invalid`.
    pub kind: String,
    pub message: String,
}

impl IrisPlanRefusal {
    pub fn message(&self) -> String {
        self.message.clone()
    }

    pub fn to_json(&self) -> Value {
        json!({ "kind": self.kind, "message": self.message })
    }
}

/// The engine-resolved receipts of an Iris plan.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedIrisPlan(Value);

impl ResolvedIrisPlan {
    pub fn to_json(&self) -> Value {
        self.0.clone()
    }
}

/// The kind name of a typed distributed failure.
pub fn dist_error_kind(error: &gen_core::iris::train::dist::DistError) -> &'static str {
    use gen_core::iris::train::dist::DistError;
    match error {
        DistError::InvalidMesh(_) => "invalid_mesh",
        DistError::GlobalBatchMismatch { .. } => "global_batch_mismatch",
        DistError::SplitWorld(_) => "split_world",
        DistError::RankLost { .. } => "rank_lost",
        DistError::Desync { .. } => "desync",
        DistError::Unsupported(_) => "unsupported",
    }
}

/// Resolve a SceneWorks Iris plan with the engine (`IrisTrainPlan::resolve`) against the backbone
/// in `backbone_dir`. Adapter targets are not resolved (that needs the weights); everything else the
/// trainer refuses from the request alone is refused here, typed.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub fn resolve_iris_plan(
    backbone_dir: &Path,
    plan: &TrainingPlan,
) -> Result<ResolvedIrisPlan, IrisPlanRefusal> {
    let request = gen_core::TrainingRequest {
        items: plan
            .dataset
            .items
            .iter()
            .map(|item| gen_core::TrainingItem {
                image_path: PathBuf::from(&item.image_path),
                caption: item.caption.clone(),
                control_image_path: None,
                model_options: item
                    .extra
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            })
            .collect(),
        config: crate::training_jobs::engine_training_config(plan),
        output_dir: PathBuf::from(&plan.output.output_dir),
        file_name: plan.output.file_name.clone(),
        trigger_words: plan.output.trigger_words.clone(),
        cancel: gen_core::CancelFlag::new(),
    };
    resolve_iris_request(backbone_dir, &request)
}

/// [`resolve_iris_plan`] for a bare SceneWorks config (the Studio's pre-launch check, before a
/// dataset or a plan exists): the same engine config mapping, no items.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub fn resolve_iris_config(
    backbone_dir: &Path,
    config: &sceneworks_core::training::TrainingConfig,
) -> Result<ResolvedIrisPlan, IrisPlanRefusal> {
    let mut engine = crate::training_jobs::map_engine_config(config);
    if config
        .advanced
        .get("mixedPrecision")
        .and_then(Value::as_str)
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("fp16"))
    {
        engine.train_dtype = "fp16".to_owned();
    }
    let request = gen_core::TrainingRequest {
        items: Vec::new(),
        config: engine,
        output_dir: PathBuf::new(),
        file_name: "plan-check.safetensors".to_owned(),
        trigger_words: Vec::new(),
        cancel: gen_core::CancelFlag::new(),
    };
    resolve_iris_request(backbone_dir, &request)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_iris_request(
    backbone_dir: &Path,
    request: &gen_core::TrainingRequest,
) -> Result<ResolvedIrisPlan, IrisPlanRefusal> {
    use gen_core::iris::train::{IrisTrainPlan, TrainFlowDefaults};
    let refuse = |error: gen_core::Error| {
        let kind = gen_core::iris::train::dist::dist_error(&error)
            .map(dist_error_kind)
            .unwrap_or("invalid");
        IrisPlanRefusal {
            kind: kind.to_owned(),
            message: format!("Iris 3B training refused this configuration: {error}"),
        }
    };
    let config = gen_core::iris::IrisConfig::from_dir(backbone_dir).map_err(refuse)?;
    let flow = TrainFlowDefaults::from_dir(backbone_dir).map_err(refuse)?;
    let resolved = IrisTrainPlan::resolve(request, &config, &flow, &[]).map_err(refuse)?;
    let (train, holdout) =
        gen_core::iris::train::holdout::split_items(&request.items).map_err(refuse)?;
    let global_batch = resolved.batch_size * resolved.dist.world.world_size * resolved.grad_accum;
    let mut dist = resolved.dist.identity();
    if let Some(object) = dist.as_object_mut() {
        object.insert("summary".to_owned(), json!(resolved.dist.summary()));
        object.insert(
            "shareDevice".to_owned(),
            json!(resolved.dist.world.share_device),
        );
        object.insert(
            "timeoutSecs".to_owned(),
            json!(resolved.dist.world.timeout.as_secs()),
        );
    }
    Ok(ResolvedIrisPlan(json!({
        "stage": resolved.stage,
        "parentStage": resolved.parent_stage,
        "runName": resolved.run_name,
        "globalBatch": global_batch,
        "expectedGlobalBatch": resolved.expected_global_batch,
        "batchSize": resolved.batch_size,
        "gradientAccumulation": resolved.grad_accum,
        "worldSize": resolved.dist.world.world_size,
        "dist": dist,
        "shapePolicy": resolved.shape_policy_id,
        "ropeAspect": resolved.rope_aspect,
        "repa": resolved.repa.as_ref().map(|repa| json!({
            "weight": repa.weight,
            "variant": repa.variant.as_str(),
            "layer": repa.layer,
            "teacher": repa.teacher.entrypoint,
            "projector": repa.projector_identity(),
        })),
        "activationCheckpointing": resolved.ac.name(),
        "holdout": {
            "everySteps": resolved.holdout.every,
            "samples": resolved.holdout.samples,
            "items": holdout.len(),
            "trainItems": train.len(),
        },
        "mixedPrecision": format!("{:?}", resolved.mixed_precision).to_ascii_lowercase(),
        "dataIdentity": resolved.data_identity(),
        "stateIdentity": resolved.state_identity(),
        "lazySync": resolved.lazy_sync,
        "scaleWarmupByWorld": resolved.scale_warmup_by_world,
    })))
}

/// The provisioned teacher's fixed `repa` keys, which a preset must leave at these values.
const REPA_FIXED_KEYS: [(&str, &str); 4] = [
    ("teacher_source", "hub"),
    ("teacher_hub", "facebookresearch/dinov2"),
    ("teacher", curricula::IRIS_REPA_TEACHER_ENTRYPOINT),
    ("teacher_weights", ""),
];

fn round_f32(value: f32) -> f64 {
    format!("{value}")
        .parse::<f64>()
        .unwrap_or(f64::from(value))
}

/// Normalize a translated upstream options bag onto the SceneWorks offered surface: drop what
/// SceneWorks does not offer (mapped torch flags, the fixed teacher identity, DDP bucketing) and
/// every knob the run's own choices make inert, so the preset validates exactly as a hand-made
/// config would. Refuses a preset that would need something SceneWorks cannot provision.
fn normalize_preset_options(
    mut bag: Map<String, Value>,
    has_parent: bool,
) -> Result<Map<String, Value>, String> {
    // `shape_policy: ""` is upstream's "derive from data.type" (multiscale → bucket, else fixed);
    // SceneWorks offers the policy itself, so resolve it here.
    let multiscale = bag.get("data_type").and_then(Value::as_str) == Some("pixel_multiscale");
    if bag
        .get("shape_policy")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        bag.insert(
            "shape_policy".to_owned(),
            json!(if multiscale { "bucket" } else { "fixed" }),
        );
    }
    for key in curricula::IRIS_CURRICULA_UNOFFERED_KEYS {
        match (key, bag.remove(key)) {
            (_, None) | (_, Some(Value::Null)) => {}
            ("data_type", Some(Value::String(kind))) if kind.starts_with("pixel") => {}
            (key, Some(value)) => {
                return Err(format!(
                    "the preset sets {key} = {value}, which SceneWorks does not offer"
                ))
            }
        }
    }
    if let Some(tracking) = bag.get_mut("tracking").and_then(Value::as_object_mut) {
        tracking.remove("report_to");
        tracking.retain(|_, value| !value.is_null());
    }
    if let Some(perf) = bag.get_mut("perf").and_then(Value::as_object_mut) {
        perf.retain(|key, _| key == "lazy_sync");
    }
    if let Some(dist) = bag.get_mut("dist").and_then(Value::as_object_mut) {
        dist.remove("ddp");
        if dist.get("checkpoint_io").and_then(Value::as_str) == Some("per_node") {
            dist.insert("checkpoint_io".to_owned(), json!("rank0"));
        }
        if dist.get("strategy").and_then(Value::as_str) != Some("fsdp2") {
            for key in ["sharding", "mesh", "param_dtype", "reshard_after_forward"] {
                dist.remove(key);
            }
        }
    }
    let repa_on = bag
        .get("repa")
        .and_then(|repa| repa.get("weight"))
        .and_then(Value::as_f64)
        .is_some_and(|weight| weight > 0.0);
    if !repa_on {
        bag.remove("repa");
    } else if let Some(repa) = bag.get_mut("repa").and_then(Value::as_object_mut) {
        for (key, fixed) in REPA_FIXED_KEYS {
            match repa.remove(key) {
                None | Some(Value::Null) => {}
                Some(Value::String(value)) if value == fixed => {}
                Some(other) => {
                    return Err(format!(
                        "the preset's REPA teacher {key} = {other} is not the provisioned \
                         {IRIS_REPA_TEACHER_REPO} teacher"
                    ))
                }
            }
        }
        for (key, fixed) in [("teacher_dim", 768), ("teacher_patch_size", 14)] {
            match repa.remove(key) {
                None | Some(Value::Null) => {}
                Some(value) if value.as_u64() == Some(fixed) => {}
                Some(other) => {
                    return Err(format!(
                        "the preset's REPA {key} = {other} does not match the provisioned teacher"
                    ))
                }
            }
        }
        if repa.get("variant").and_then(Value::as_str) != Some("irepa") {
            repa.remove("spatial_norm_gamma");
        }
    }
    match bag
        .get("shape_policy")
        .and_then(Value::as_str)
        .unwrap_or("fixed")
    {
        "bucket" => {
            bag.remove("shape_align");
            bag.remove("shape_max_ratio");
        }
        "area" => {
            bag.remove("aspect_ratio_bucket");
        }
        _ => {
            bag.remove("aspect_ratio_bucket");
            bag.remove("shape_align");
            bag.remove("shape_max_ratio");
        }
    }
    if bag.get("activation_checkpointing").and_then(Value::as_str) != Some("selective_layer") {
        bag.remove("ac_selective_every");
    }
    if bag
        .get("val_every_steps")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        bag.remove("val_every_steps");
        bag.remove("val_samples");
    }
    if !has_parent {
        // A first stage starts fresh: the resume knobs would do nothing.
        for key in [
            "resume_data_policy",
            "override_lr_on_resume",
            "resume_optimizer",
        ] {
            bag.remove(key);
        }
    }
    if bag.get("shape_policy").and_then(Value::as_str) == Some("fixed") {
        bag.remove("shape_policy");
    }
    bag.retain(|_, value| !value.is_null());
    Ok(bag)
}

/// The four official stage presets as editable SceneWorks configs (`[{ id, parentStage, config }]`),
/// from the engine's vendored upstream YAML. The config is the Iris target's defaults overlaid with
/// the translated stage (a full fine-tune, as upstream trains).
pub fn iris_stage_presets() -> Result<Vec<Value>, String> {
    use gen_core::iris::train::upstream::{stage_preset_config, STAGE_PRESETS};
    let target = sceneworks_core::training::iris::iris_generation_target();
    STAGE_PRESETS
        .iter()
        .map(|preset| {
            let engine = stage_preset_config(preset.id, gen_core::TrainingConfig::default())
                .map_err(|error| format!("stage preset {}: {error}", preset.id))?;
            let translated = engine
                .model_options
                .get(IRIS_OPTIONS_KEY)
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let translated = normalize_preset_options(translated, preset.parent.is_some())
                .map_err(|error| format!("stage preset {}: {error}", preset.id))?;
            let mut config = target.defaults.clone();
            config.learning_rate = sceneworks_core::contracts::ContractNumber::from_f64(round_f32(
                engine.learning_rate,
            ))
            .ok_or_else(|| format!("stage preset {}: a non-finite LR", preset.id))?;
            config.steps = engine.steps;
            config.batch_size = engine.batch_size;
            config.gradient_accumulation = engine.gradient_accumulation;
            config.resolution = engine.resolution;
            config.save_every = engine.save_every;
            config.seed = i64::try_from(engine.seed).unwrap_or(i64::MAX);
            config.optimizer = engine.optimizer.clone();
            let advanced = &mut config.advanced;
            let dtype = match engine.train_dtype.as_str() {
                "no" | "f32" | "fp32" => "f32",
                other => other,
            };
            advanced.insert("mixedPrecision".to_owned(), json!(dtype));
            advanced.insert(
                "networkType".to_owned(),
                json!(if engine.full_finetune { "full" } else { "lora" }),
            );
            advanced.insert(
                "lrScheduler".to_owned(),
                json!(match engine.lr_scheduler {
                    gen_core::LrSchedule::Cosine => "cosine",
                    _ => "constant",
                }),
            );
            advanced.insert("lrWarmupSteps".to_owned(), json!(engine.lr_warmup_steps));
            advanced.insert(
                "weightDecay".to_owned(),
                json!(round_f32(engine.weight_decay)),
            );
            advanced.insert("timestepType".to_owned(), json!(engine.timestep_type));
            advanced.insert("sampleEvery".to_owned(), json!(engine.sample_every));
            advanced.insert("sampleSteps".to_owned(), json!(engine.sample_steps));
            advanced.insert(
                "sampleGuidanceScale".to_owned(),
                json!(round_f32(engine.sample_guidance_scale)),
            );
            // Upstream renders its validation prompts (the opt-in), not user sample prompts.
            advanced.remove("sampleCount");
            advanced.insert("samplePrompts".to_owned(), json!([]));
            let mut bag = advanced
                .get(IRIS_OPTIONS_KEY)
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            bag.extend(translated);
            advanced.insert(IRIS_OPTIONS_KEY.to_owned(), Value::Object(bag));
            Ok(json!({
                "id": preset.id,
                "parentStage": preset.parent,
                "config": config,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::training::iris::curricula as core;

    /// The SceneWorks mirror of the engine's closed sets is the engine's, value for value.
    #[test]
    fn the_core_mirror_matches_the_engine_contract() {
        use gen_core::iris::train::{ac, holdout, repa, shapes};
        assert_eq!(core::BUCKET_TABLES, shapes::BUCKET_TABLES);
        let mut policies = core::AC_POLICIES.to_vec();
        let mut engine = ac::POLICIES.to_vec();
        policies.sort_unstable();
        engine.sort_unstable();
        assert_eq!(policies, engine);
        assert_eq!(core::ITEM_SPLIT_KEY, holdout::ITEM_SPLIT_KEY);
        assert_eq!(core::IRIS_REPA_TEACHER_REPO, repa::DINOV2_BASE_HF_REPO);
        assert_eq!(
            core::IRIS_REPA_TEACHER_REVISION,
            repa::DINOV2_BASE_HF_REVISION
        );
        assert_eq!(core::IRIS_REPA_TEACHER_FILE, repa::TEACHER_WEIGHTS_FILE);
        assert_eq!(core::IRIS_REPA_TEACHER_ENTRYPOINT, repa::DEFAULT_TEACHER);
        assert_eq!(
            core::DEFAULT_DIST_TIMEOUT_SECS,
            gen_core::iris::train::dist::DEFAULT_TIMEOUT_SECS
        );
        let stages: Vec<(&str, Option<&str>)> = gen_core::iris::train::upstream::STAGE_PRESETS
            .iter()
            .map(|preset| (preset.id, preset.parent))
            .collect();
        assert_eq!(stages, core::IRIS_STAGES.to_vec());
        for shape in core::SHAPE_POLICIES {
            assert!(
                shapes::ShapePolicy::build(shape, 512, 16, "shared21-512", 32, 4.0).is_ok(),
                "{shape}"
            );
        }
    }

    /// Every offered key is one the engine's option parser accepts, and every engine key SceneWorks
    /// leaves out is named as deliberately unoffered — so a new engine key cannot slip by unnoticed.
    #[test]
    fn every_offered_key_is_an_engine_key() {
        let mut config = gen_core::TrainingConfig::default();
        for key in core::IRIS_CURRICULA_OPTION_KEYS
            .iter()
            .chain(sceneworks_core::training::iris::IRIS_CLIENT_OPTION_KEYS.iter())
        {
            let mut bag = Map::new();
            bag.insert((*key).to_owned(), Value::Null);
            config
                .model_options
                .insert(IRIS_OPTIONS_KEY.to_owned(), Value::Object(bag));
            gen_core::iris::train::IrisTrainOptions::from_config(&config)
                .unwrap_or_else(|error| panic!("{key} is not an engine option: {error}"));
        }
        let mut bag = Map::new();
        bag.insert("not_an_option".to_owned(), Value::Null);
        config
            .model_options
            .insert(IRIS_OPTIONS_KEY.to_owned(), Value::Object(bag));
        assert!(gen_core::iris::train::IrisTrainOptions::from_config(&config).is_err());
    }

    #[test]
    fn the_stage_presets_are_the_engines_translated_and_validate_on_the_offered_surface() {
        let presets = iris_stage_presets().expect("the four official stages translate");
        let ids: Vec<&str> = presets.iter().map(|p| p["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["stage1_256", "stage2_512", "stage3_1024", "sft_1024"]);
        let target = sceneworks_core::training::iris::iris_generation_target();
        for preset in &presets {
            let config: sceneworks_core::training::TrainingConfig =
                serde_json::from_value(preset["config"].clone()).unwrap();
            let iris = &config.advanced["iris"];
            assert_eq!(iris["stage"], preset["id"]);
            assert_eq!(iris["parent_stage"], preset["parentStage"]);
            assert_eq!(config.advanced["networkType"], json!("full"));
            // Upstream's 64-GPU recipe: an expected global batch and an fsdp2 mesh the user edits.
            assert!(iris["expected_global_batch"].as_u64().unwrap() >= 512);
            assert_eq!(iris["dist"]["strategy"], json!("fsdp2"));
            assert!(iris["perf"]
                .as_object()
                .unwrap()
                .keys()
                .all(|k| k == "lazy_sync"));
            assert!(iris.get("data_type").is_none());
            // Applied verbatim on one host the preset is refused, typed (its 8-rank hybrid mesh
            // and 64-GPU global batch do not fit one rank) — the Studio shows that before launch;
            // with the topology edited it validates.
            let mut verbatim = config.clone();
            if preset["parentStage"].is_string() {
                // A continuing stage names its parent run (the API resolves it to the checkpoint).
                verbatim
                    .advanced
                    .insert("resumeFromJobId".to_owned(), json!("job_parent"));
            }
            let refused =
                sceneworks_core::training::validate_training_config_for_target(&target, &verbatim)
                    .expect_err("the 64-GPU recipe does not fit one rank");
            let refused = format!("{refused}");
            assert!(
                refused.contains("mesh") || refused.contains("global batch"),
                "{refused}"
            );
            let mut edited = config.clone();
            let bag = edited
                .advanced
                .get_mut("iris")
                .and_then(Value::as_object_mut)
                .unwrap();
            bag.remove("expected_global_batch");
            bag.insert(
                "dist".to_owned(),
                json!({"strategy": "fsdp2", "sharding": "full"}),
            );
            if preset["parentStage"].is_string() {
                edited
                    .advanced
                    .insert("resumeFromJobId".to_owned(), json!("job_parent"));
            }
            sceneworks_core::training::validate_training_config_for_target(&target, &edited)
                .unwrap_or_else(|error| panic!("{}: {error}", preset["id"]));
        }
        // Stage 1 trains with REPA on the square RoPE law; stage 2 drops REPA, buckets at 512 and
        // continues stage 1 as a new phase.
        let s1 = &presets[0]["config"]["advanced"]["iris"];
        assert_eq!(s1["repa"]["weight"], json!(0.5));
        assert!(s1["repa"].get("teacher").is_none());
        assert_eq!(s1["rope_aspect"], json!("square"));
        assert!(s1.get("resume_data_policy").is_none());
        let s2 = &presets[1]["config"]["advanced"]["iris"];
        assert!(s2.get("repa").is_none());
        assert_eq!(s2["shape_policy"], json!("bucket"));
        assert_eq!(s2["aspect_ratio_bucket"], json!("shared21-512"));
        assert_eq!(s2["resume_data_policy"], json!("new_phase"));
        assert_eq!(presets[1]["config"]["resolution"], json!(512));
    }

    #[test]
    fn a_preset_naming_an_unprovisioned_teacher_is_refused() {
        let bag = json!({"repa": {"weight": 0.5, "teacher": "dinov2_vitl14"}});
        let error = normalize_preset_options(bag.as_object().unwrap().clone(), false)
            .expect_err("only the provisioned teacher");
        assert!(error.contains("teacher"), "{error}");
        let ok = json!({"repa": {"weight": 0.5, "teacher": "dinov2_vitb14", "teacher_dim": 768,
                                  "spatial_norm_gamma": 1.0, "variant": "repa"},
                        "shape_policy": "fixed", "aspect_ratio_bucket": "shared21-512",
                        "perf": {"compile": true, "lazy_sync": false},
                        "dist": {"strategy": "ddp", "sharding": "full", "ddp": {}}});
        let normalized = normalize_preset_options(ok.as_object().unwrap().clone(), false).unwrap();
        assert_eq!(
            normalized["repa"],
            json!({"weight": 0.5, "variant": "repa"})
        );
        assert!(normalized.get("aspect_ratio_bucket").is_none());
        assert_eq!(normalized["perf"], json!({"lazy_sync": false}));
        assert_eq!(normalized["dist"], json!({"strategy": "ddp"}));
    }

    #[test]
    fn the_teacher_integrity_check_reads_size_then_sha256() {
        let _real_pin = test_teacher_integrity(None);
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(IRIS_REPA_TEACHER_FILE);
        std::fs::write(&file, b"not the teacher").unwrap();
        let error = verify_repa_teacher_file(&file).expect_err("wrong size");
        assert!(error.contains("bytes"), "{error}");
        // Right size, wrong bytes: the sha256 refuses (sparse file, never a real 346 MB write).
        let handle = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
        handle.set_len(IRIS_REPA_TEACHER_SIZE_BYTES).unwrap();
        drop(handle);
        let error = verify_repa_teacher_file(&file).expect_err("wrong bytes");
        assert!(error.contains("sha256"), "{error}");
    }
}
