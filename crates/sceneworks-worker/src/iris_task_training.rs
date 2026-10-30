//! Worker seams of Iris-3B DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) training (epic
//! sc-25678).
//!
//! Both run through the generic native training executor (`training_jobs`) on the `iris_train`
//! kernel, exactly like the generation trainer; this module carries what that executor cannot know:
//!
//! * the **request shape** — a task item is an RGB image plus its sidecar(s), resolved under the
//!   dataset root into the engine's per-item `model_options` (`depth` / `degraded_image`); the
//!   engine options bag is the ONLY `model_options` key (both trainers refuse any other); captions
//!   are not read (the tasks run on the empty prompt / no prompt);
//! * the **load inputs** — the task export the run fine-tunes (cache-only, pinned), the empty-prompt
//!   source of a depth run from the generation backbone (the installed depth export's
//!   `empty_prompt.safetensors`, else the generation install's text encoder, encoded once), and the
//!   three pinned TRAINING-ONLY resources of a restoration run, each integrity-checked
//!   (`verify_resource`: size + sha256) before anything loads — nothing downloads at training time;
//! * the **pre-launch views** the API serves: install availability, sidecar format/alignment checks
//!   and a per-item dataset report under the run's invalid-example policy — all through the engine's
//!   own readers (`gen_core_iris::train::depth`, `gen_core_iris::restoration_train`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use gen_core::{TrainingConfig, TrainingItem, TrainingRequest, WeightsSource};
use gen_core_iris::downstream::{TaskExport, EMPTY_PROMPT_FILE};
use gen_core_iris::restoration::RestorationResources;
use gen_core_iris::restoration_train::{
    self as restore_train, TrainingResource, TRAINING_RESOURCES,
};
use gen_core_iris::train::depth as depth_train;
use gen_core_iris::{
    IrisConfig, IrisTask, BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE, UPSTREAM_WEIGHTS_REPO,
    UPSTREAM_WEIGHTS_REVISION,
};
use sceneworks_core::training::iris::depth::{
    depth_init, DEPTH_INIT_PARENT, ITEM_DEPTH_KEY as SW_ITEM_DEPTH_KEY,
};
use sceneworks_core::training::iris::restore::{
    restore_options, ITEM_DEGRADED_KEY as SW_ITEM_DEGRADED_KEY, RESTORE_TRAINING_RESOURCE_MODELS,
};
use sceneworks_core::training::iris::{iris_options, IrisObjective};
use serde_json::{json, Map, Value};

/// The task export folders inside the pinned `speridlabs/iris-3b` snapshot.
const DEPTH_SUBDIR: &str = "depth";
const RESTORE_SUBDIR: &str = "upscaler";
const EXPORT_FILES: [&str; 3] = [
    BACKBONE_CONFIG_FILE,
    BACKBONE_WEIGHTS_FILE,
    EMPTY_PROMPT_FILE,
];

/// The objective of a DOWNSTREAM-task Iris plan (depth / restoration), `None` for generation and
/// every other kernel.
pub fn task_objective(kernel: &str, base_model: &str) -> Option<IrisObjective> {
    if !sceneworks_core::training::iris::is_iris_training_kernel(kernel) {
        return None;
    }
    match IrisObjective::from_base_model(base_model)? {
        IrisObjective::Generation => None,
        objective => Some(objective),
    }
}

fn pinned_snapshot(data_dir: &Path, repo: &str, revision: &str) -> Option<PathBuf> {
    crate::model_jobs::huggingface_pinned_snapshot_dir(data_dir, repo, revision)
}

/// A complete task export folder (`depth/` or `upscaler/`) at the pinned revision, or `None`.
fn installed_export(data_dir: &Path, subdir: &str) -> Option<PathBuf> {
    let dir =
        pinned_snapshot(data_dir, UPSTREAM_WEIGHTS_REPO, UPSTREAM_WEIGHTS_REVISION)?.join(subdir);
    EXPORT_FILES
        .iter()
        .all(|file| dir.join(file).is_file())
        .then_some(dir)
}

/// The generation backbone directory (`config.yaml` + `model.safetensors` at the snapshot root).
fn installed_backbone(data_dir: &Path) -> Option<PathBuf> {
    let dir = pinned_snapshot(data_dir, UPSTREAM_WEIGHTS_REPO, UPSTREAM_WEIGHTS_REVISION)?;
    [BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE]
        .iter()
        .all(|file| dir.join(file).is_file())
        .then_some(dir)
}

/// The generation install's Qwen3-VL text encoder snapshot (its pinned `text_encoder`
/// co-requisite), when installed.
fn installed_generation_text_encoder(data_dir: &Path) -> Option<PathBuf> {
    let entry = crate::training_jobs::builtin_model_manifest_entry("iris_3b")?;
    let download = entry
        .get("downloads")?
        .as_array()?
        .iter()
        .find(|download| {
            download.get("componentId").and_then(Value::as_str)
                == Some(gen_core_iris::TEXT_ENCODER_COMPONENT)
        })?;
    let snapshot = pinned_snapshot(
        data_dir,
        download.get("repo")?.as_str()?,
        download.get("revision")?.as_str()?,
    )?;
    let files = download.get("files")?.as_array()?;
    files
        .iter()
        .filter_map(Value::as_str)
        .all(|pattern| snapshot_has(&snapshot, pattern))
        .then_some(snapshot)
}

/// Whether a snapshot holds a declared download entry: a literal file, or — for a `prefix*suffix`
/// glob (the encoder's `model-*.safetensors` shards) — at least one matching file in that directory.
fn snapshot_has(snapshot: &Path, pattern: &str) -> bool {
    let Some((prefix, suffix)) = pattern.split_once('*') else {
        return snapshot.join(pattern).is_file();
    };
    let (dir, prefix) = match prefix.rsplit_once('/') {
        Some((dir, prefix)) => (snapshot.join(dir), prefix),
        None => (snapshot.to_path_buf(), prefix),
    };
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.len() >= prefix.len() + suffix.len()
                && name.starts_with(prefix)
                && name.ends_with(suffix)
                && entry.path().is_file()
        })
    })
}

/// The pinned training resources a restoration run stages. Tests substitute fixtures with their own
/// pins (a real 4.8 GB trunk cannot be staged in a unit test); production is the engine's table.
fn restore_training_resources() -> Vec<TrainingResource> {
    #[cfg(test)]
    if let Some(resources) = TEST_TRAINING_RESOURCES.lock().unwrap().clone() {
        return resources;
    }
    TRAINING_RESOURCES.to_vec()
}

#[cfg(test)]
pub(crate) static TEST_TRAINING_RESOURCES: std::sync::Mutex<Option<Vec<TrainingResource>>> =
    std::sync::Mutex::new(None);

/// A pinned training resource's file in the HF cache.
fn installed_resource(data_dir: &Path, resource: &TrainingResource) -> Option<PathBuf> {
    let path = pinned_snapshot(data_dir, resource.repo, resource.revision)?.join(resource.file);
    path.is_file().then_some(path)
}

fn catalog_id_for(component: &str) -> &'static str {
    RESTORE_TRAINING_RESOURCE_MODELS
        .iter()
        .find(|(_, candidate)| *candidate == component)
        .map(|(model, _)| *model)
        .unwrap_or("iris_restore_train_resources")
}

fn sw_config_to_engine(config: &sceneworks_core::training::TrainingConfig) -> TrainingConfig {
    let advanced = &config.advanced;
    let string = |key: &str, default: &str| {
        advanced
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    let u32_of = |key: &str| {
        advanced
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0)
    };
    let network = string("networkType", "full");
    TrainingConfig {
        rank: config.rank,
        alpha: sceneworks_core::training::training_alpha_f32(&config.alpha)
            .expect("training target preflight validates alpha before native mapping"),
        learning_rate: config.learning_rate.as_f64().unwrap_or(1e-5) as f32,
        steps: config.steps,
        batch_size: config.batch_size,
        gradient_accumulation: config.gradient_accumulation,
        resolution: config.resolution,
        save_every: config.save_every,
        seed: config.seed.max(0) as u64,
        optimizer: config.optimizer.clone(),
        weight_decay: advanced
            .get("weightDecay")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32,
        lr_scheduler: gen_core::LrSchedule::parse(&string("lrScheduler", "constant")),
        lr_warmup_steps: u32_of("lrWarmupSteps"),
        network_type: gen_core::NetworkType::parse(&network),
        full_finetune: network.eq_ignore_ascii_case("full"),
        loss_type: string("lossType", "mse"),
        sample_every: u32_of("sampleEvery"),
        ..TrainingConfig::default()
    }
}

/// Shape the engine config of a task run: its options bag is the ONLY `model_options` key (both
/// task trainers refuse any other), and nothing prompt-shaped travels (no prompt, no previews
/// prompts, no adapter targets for a full depth run).
pub(crate) fn shape_task_config(
    mut config: TrainingConfig,
    objective: IrisObjective,
) -> TrainingConfig {
    let key = objective.options_key();
    let bag = config
        .model_options
        .get(key)
        .cloned()
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(Map::new()));
    config.model_options = Map::from_iter([(key.to_owned(), bag)]);
    config.sample_prompts.clear();
    config.trigger_word = None;
    config.control_type = None;
    if objective == IrisObjective::Depth {
        config.lora_target_modules.clear();
        config.loss_type = "l1".to_owned();
    }
    config
}

/// A plain dataset-relative path: non-empty, relative, no `..`, no backslash.
fn safe_relative(relative: &str) -> bool {
    !relative.is_empty()
        && !relative.contains('\\')
        && Path::new(relative)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

fn resolve_under(root: &Path, relative: &str, label: &str) -> Result<PathBuf, String> {
    if !safe_relative(relative) {
        return Err(format!(
            "{label} {relative:?} is not a dataset-relative path."
        ));
    }
    let mut path = root.to_path_buf();
    for component in Path::new(relative).components() {
        path.push(component);
    }
    if !path.is_file() {
        return Err(format!("{label} {} does not exist.", path.display()));
    }
    Ok(path)
}

/// One task item → the engine `TrainingItem`: the RGB image, an empty caption (not read), and its
/// sidecar(s) resolved under the dataset root into the engine's per-item option.
pub fn task_item(
    objective: IrisObjective,
    dataset_root: &Path,
    image_path: PathBuf,
    extra: &sceneworks_core::contracts::ExtraFields,
) -> Result<TrainingItem, String> {
    let mut item = TrainingItem::captioned(image_path, String::new());
    match objective {
        IrisObjective::Depth => {
            let depth = extra
                .get(SW_ITEM_DEPTH_KEY)
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    format!(
                        "dataset item {} has no depth target",
                        item.image_path.display()
                    )
                })?;
            let mut engine = Map::new();
            let path = depth
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default();
            engine.insert(
                "path".to_owned(),
                json!(resolve_under(dataset_root, path, "Depth map")?),
            );
            if let Some(mask) = depth.get("maskPath").and_then(Value::as_str) {
                engine.insert(
                    "mask_path".to_owned(),
                    json!(resolve_under(dataset_root, mask, "Depth mask")?),
                );
            }
            for (from, to) in [
                ("scale", "scale"),
                ("minDepth", "min_depth"),
                ("maxDepth", "max_depth"),
            ] {
                if let Some(number) = depth.get(from).filter(|value| value.is_number()) {
                    engine.insert(to.to_owned(), number.clone());
                }
            }
            item.model_options.insert(
                depth_train::ITEM_DEPTH_KEY.to_owned(),
                Value::Object(engine),
            );
        }
        IrisObjective::Restoration => {
            if let Some(degraded) = extra.get(SW_ITEM_DEGRADED_KEY).and_then(Value::as_str) {
                item.model_options.insert(
                    restore_train::ITEM_DEGRADED_KEY.to_owned(),
                    json!(resolve_under(dataset_root, degraded, "Degraded image")?),
                );
            }
        }
        IrisObjective::Generation => {}
    }
    Ok(item)
}

/// What the task trainer loads: `spec.weights` and the caller-staged components.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskLoadInputs {
    pub weights: PathBuf,
    pub components: BTreeMap<String, WeightsSource>,
}

/// A typed install refusal of an Iris training run: `kind` names the missing resource class
/// (`missing_backbone`, `missing_text_encoder`, `missing_export`, `missing_training_resources`) and
/// `message` the actionable fix. The API returns it as an `iris_training_refused` error at
/// plan-check and submit; the worker repeats it before anything loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrisInstallRefusal {
    pub kind: &'static str,
    pub message: String,
}

impl IrisInstallRefusal {
    fn new(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// The generation backbone directory of the pinned `iris_3b` install (`config.yaml` +
/// `model.safetensors` at the snapshot ROOT — never a task export's files), when installed.
pub fn iris_generation_backbone_dir(data_dir: &Path) -> Option<PathBuf> {
    installed_backbone(data_dir)
}

/// What an Iris training run of `objective` needs that is not installed (`None` = every resource it
/// needs is installed), judged PER TASK RESOURCE at the pinned revisions — never off the shared
/// repo's install marker, so a depth/ or upscaler/ export never makes generation look ready.
///
/// * generation: the root backbone (`config.yaml` + `model.safetensors`) AND the generation
///   install's Qwen3-VL text encoder (the trainer encodes captions with it);
/// * depth: its `depth/` export, or (`depthInit: parent`) the backbone + an empty-prompt source;
/// * restoration: its `upscaler/` export, the pinned training-only resources, and (`init: parent`)
///   the backbone.
pub fn iris_training_install_refusal(
    data_dir: &Path,
    objective: IrisObjective,
    config: &sceneworks_core::training::TrainingConfig,
) -> Option<IrisInstallRefusal> {
    match objective {
        IrisObjective::Generation => {
            if installed_backbone(data_dir).is_none() {
                return Some(IrisInstallRefusal::new(
                    "missing_backbone",
                    format!(
                        "Iris 3B is not installed: training needs its generation backbone \
                         (config.yaml + model.safetensors of {UPSTREAM_WEIGHTS_REPO} at \
                         {UPSTREAM_WEIGHTS_REVISION}) — install \"Iris 3B\" in Models. A depth or \
                         restoration install does not provide it."
                    ),
                ));
            }
            installed_generation_text_encoder(data_dir)
                .is_none()
                .then(|| {
                    IrisInstallRefusal::new(
                    "missing_text_encoder",
                    "Iris 3B training encodes its captions with the Qwen3-VL text encoder, which \
                     is not installed — repair \"Iris 3B\" in Models (its text encoder \
                     co-requisite). Nothing is downloaded during training.",
                )
                })
        }
        IrisObjective::Depth => {
            let parent = depth_init(config).ok() == Some(DEPTH_INIT_PARENT);
            if parent {
                if installed_backbone(data_dir).is_none() {
                    return Some(IrisInstallRefusal::new(
                        "missing_backbone",
                        "A depth run from the generation backbone (depthInit: parent) needs \"Iris \
                         3B\" (the generation install) — install it in Models.",
                    ));
                }
                // Its empty-prompt conditioning: the depth export's shipped file, else the
                // generation install's text encoder (encoded once).
                if installed_export(data_dir, DEPTH_SUBDIR).is_none()
                    && installed_generation_text_encoder(data_dir).is_none()
                {
                    return Some(IrisInstallRefusal::new(
                        "missing_text_encoder",
                        "A depth run from the generation backbone needs the empty-prompt \
                         conditioning: install \"Iris 3B Depth\" (its empty_prompt.safetensors) or \
                         the Iris 3B text encoder.",
                    ));
                }
                None
            } else {
                installed_export(data_dir, DEPTH_SUBDIR).is_none().then(|| {
                    IrisInstallRefusal::new(
                        "missing_export",
                        "Iris 3B Depth is not installed. Install \"Iris 3B Depth\" in Models — a \
                         depth run fine-tunes its depth/ export (or choose depthInit: parent to \
                         start from the generation backbone).",
                    )
                })
            }
        }
        IrisObjective::Restoration => {
            if installed_export(data_dir, RESTORE_SUBDIR).is_none() {
                return Some(IrisInstallRefusal::new(
                    "missing_export",
                    "Iris 3B Restoration is not installed. Install \"Iris 3B Restoration\" in \
                     Models — a restoration run fine-tunes its upscaler/ export.",
                ));
            }
            let missing: Vec<String> = restore_training_resources()
                .iter()
                .filter(|resource| installed_resource(data_dir, resource).is_none())
                .map(|resource| {
                    format!("{} ({})", catalog_id_for(resource.component), resource.name)
                })
                .collect();
            if !missing.is_empty() {
                return Some(IrisInstallRefusal::new(
                    "missing_training_resources",
                    format!(
                        "Restoration training needs its training-only resources installed in \
                         Models: {}. They are never used for restoration itself.",
                        missing.join(", ")
                    ),
                ));
            }
            let parent = restore_options(config)
                .ok()
                .and_then(|options| {
                    options
                        .get("init")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .as_deref()
                == Some("parent");
            (parent && installed_backbone(data_dir).is_none()).then(|| {
                IrisInstallRefusal::new(
                    "missing_backbone",
                    "init: parent starts the restorer from the generation backbone — install \
                     \"Iris 3B\" in Models.",
                )
            })
        }
    }
}

/// [`iris_training_install_refusal`]'s message (`None` = every resource the run needs is
/// installed). The worker repeats it before load.
pub fn iris_task_training_unavailable(
    data_dir: &Path,
    objective: IrisObjective,
    config: &sceneworks_core::training::TrainingConfig,
) -> Option<String> {
    iris_training_install_refusal(data_dir, objective, config).map(|refusal| refusal.message)
}

/// Resolve (and integrity-check) what a task trainer loads. Blocking: a restoration run hashes the
/// three training resources against their pins.
pub(crate) fn task_load_inputs(
    settings: &crate::Settings,
    objective: IrisObjective,
    config: &sceneworks_core::training::TrainingConfig,
) -> Result<TaskLoadInputs, String> {
    let data_dir = &settings.data_dir;
    if let Some(refusal) = iris_task_training_unavailable(data_dir, objective, config) {
        return Err(refusal);
    }
    let mut components = BTreeMap::new();
    let weights = match objective {
        IrisObjective::Generation => return Err("not a task objective".to_owned()),
        IrisObjective::Depth => {
            if depth_init(config).ok() == Some(DEPTH_INIT_PARENT) {
                let backbone = installed_backbone(data_dir).expect("checked above");
                match installed_export(data_dir, DEPTH_SUBDIR) {
                    // Prefer the shipped conditioning: no text encoder at all.
                    Some(export) => {
                        components.insert(
                            depth_train::EMPTY_PROMPT_COMPONENT.to_owned(),
                            WeightsSource::File(export.join(EMPTY_PROMPT_FILE)),
                        );
                    }
                    // Else encode `""` once with the generation install's text encoder.
                    None => {
                        let encoder =
                            installed_generation_text_encoder(data_dir).expect("checked above");
                        components.insert(
                            gen_core_iris::TEXT_ENCODER_COMPONENT.to_owned(),
                            WeightsSource::Dir(encoder),
                        );
                    }
                }
                backbone
            } else {
                installed_export(data_dir, DEPTH_SUBDIR).expect("checked above")
            }
        }
        IrisObjective::Restoration => {
            for resource in restore_training_resources().iter() {
                let path = installed_resource(data_dir, resource).expect("checked above");
                restore_train::verify_resource(&path, resource).map_err(|error| {
                    format!(
                        "{error}. Delete and re-install \"{}\" in Models.",
                        catalog_id_for(resource.component)
                    )
                })?;
                components.insert(resource.component.to_owned(), WeightsSource::File(path));
            }
            let parent = restore_options(config)
                .ok()
                .and_then(|options| {
                    options
                        .get("init")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .as_deref()
                == Some("parent");
            if parent {
                components.insert(
                    restore_train::PARENT_COMPONENT.to_owned(),
                    WeightsSource::Dir(installed_backbone(data_dir).expect("checked above")),
                );
            }
            installed_export(data_dir, RESTORE_SUBDIR).expect("checked above")
        }
    };
    Ok(TaskLoadInputs {
        weights,
        components,
    })
}

/// The export directory of a TRAINED task model (a registered depth / restoration training run) the
/// depth or restoration job was asked to use. Confined by construction: the id is one plain path
/// component under `<data>/models/finetunes/`, the export is the run directory's one complete task
/// export (`config.yaml`, `model.safetensors`, `empty_prompt.safetensors`), its canonical path stays
/// inside that run directory, and its `config.yaml` must declare `task` — a model of another task is
/// a typed refusal, never a silent substitute.
pub fn trained_task_export_dir(
    data_dir: &Path,
    model_id: &str,
    task: IrisTask,
) -> Result<PathBuf, String> {
    let label = match task {
        IrisTask::Depth => "depth",
        _ => "restoration",
    };
    if !sceneworks_core::image_restoration::is_plain_model_id(model_id) {
        return Err(format!(
            "{model_id:?} is not a trained Iris {label} model id"
        ));
    }
    let run = data_dir.join("models").join("finetunes").join(model_id);
    let run = std::fs::canonicalize(&run).map_err(|_| {
        format!(
            "the trained {label} model {model_id} is not installed (its training output is \
             missing) — pick another model or retrain it"
        )
    })?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&run)
        .map_err(|error| format!("read {}: {error}", run.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && EXPORT_FILES.iter().all(|file| path.join(file).is_file()))
        .collect();
    candidates.sort();
    let mut last_error = None;
    for dir in candidates {
        let Ok(canonical) = std::fs::canonicalize(&dir) else {
            continue;
        };
        if !canonical.starts_with(&run) {
            continue;
        }
        let checked = match task {
            IrisTask::Depth => {
                TaskExport::from_dir(&canonical, IrisTask::Depth, depth_train::DEPTH_TRAINER_ID)
                    .map(drop)
            }
            _ => RestorationResources::from_dir(&canonical, restore_train::TRAINER_ID).map(drop),
        };
        match checked {
            Ok(()) => return Ok(canonical),
            Err(error) => last_error = Some(error.to_string()),
        }
    }
    Err(match last_error {
        Some(error) => {
            format!("the trained model {model_id} is not an Iris {label} export: {error}")
        }
        None => format!(
            "the trained {label} model {model_id} has no complete export (config.yaml, \
             model.safetensors, empty_prompt.safetensors)"
        ),
    })
}

/// Check one depth/mask sidecar with the engine's own reader (`.npy` or 8/16-bit grayscale `.png`),
/// and its alignment with the item's image (same `W × H`). Returns the plane's `(width, height)`.
pub fn check_depth_sidecar(path: &Path, image: &Path) -> Result<(usize, usize), String> {
    let plane = depth_train::read_plane(path).map_err(|error| error.to_string())?;
    let (width, height) = image::image_dimensions(image)
        .map_err(|error| format!("the item's image is not readable: {error}"))?;
    if (plane.width, plane.height) != (width as usize, height as usize) {
        return Err(format!(
            "the plane is {}×{} but the item's image is {width}×{height} — RGB, depth and mask must \
             be aligned pixel for pixel",
            plane.width, plane.height
        ));
    }
    Ok((plane.width, plane.height))
}

/// Check one aligned degraded input: readable, and the clean image an exact integer multiple of it on
/// both axes (the engine's pair rule).
pub fn check_degraded_sidecar(path: &Path, image: &Path) -> Result<u32, String> {
    let (lw, lh) = image::image_dimensions(path)
        .map_err(|error| format!("the degraded image is not readable: {error}"))?;
    let (w, h) = image::image_dimensions(image)
        .map_err(|error| format!("the item's image is not readable: {error}"))?;
    if lw == 0 || lh == 0 || w % lw != 0 || h % lh != 0 || w / lw != h / lh {
        return Err(format!(
            "the clean image ({w}×{h}) and the degraded image ({lw}×{lh}) are not an aligned \
             integer-scale pair"
        ));
    }
    Ok(w / lw)
}

/// The engine's degradation parser over a request's `degradation` overrides (HYPIR key names) and the
/// crop it must hold — the submit-time twin of the trainer's refusal.
pub fn check_restore_degradation(degradation: &Value, crop: u32) -> Result<(), String> {
    let config = restore_train::parse_degradation(Some(degradation)).map_err(|e| e.to_string())?;
    config.check_crop(crop as usize).map_err(|e| e.to_string())
}

/// One item of a pre-launch dataset report.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetCheckItem {
    pub item_id: String,
    pub display_name: String,
    /// `ok` | `invalid` (the policy decides: reject refuses the run, skip drops the item) | `error`
    /// (unreadable / misaligned — always refused).
    pub status: &'static str,
    pub reason: Option<String>,
}

/// The pre-launch dataset report of a task run: every item checked by the engine's own readers under
/// the config the run would use (the trainer repeats exactly this pass before training).
pub fn iris_task_dataset_report(
    data_dir: &Path,
    objective: IrisObjective,
    config: &sceneworks_core::training::TrainingConfig,
    dataset: &sceneworks_core::training::TrainingDataset,
    dataset_root: &Path,
) -> Value {
    let engine_config = sw_config_to_engine(config);
    let mut engine_config = engine_config;
    let key = objective.options_key();
    let bag = match objective {
        IrisObjective::Restoration => restore_options(config).ok().map(Value::Object),
        _ => iris_options(config).ok().map(Value::Object),
    }
    .unwrap_or_else(|| json!({}));
    engine_config.model_options = Map::from_iter([(key.to_owned(), bag)]);
    let engine_config = shape_task_config(engine_config, objective);
    let mut items = Vec::with_capacity(dataset.items.len());
    let mut engine_items = Vec::with_capacity(dataset.items.len());
    for item in &dataset.items {
        let image = {
            let mut path = dataset_root.to_path_buf();
            for component in Path::new(&item.path).components() {
                path.push(component);
            }
            path
        };
        match task_item(objective, dataset_root, image, &item.extra) {
            Ok(engine_item) => engine_items.push(Some(engine_item)),
            Err(reason) => {
                items.push(DatasetCheckItem {
                    item_id: item.id.clone(),
                    display_name: item.display_name.clone(),
                    status: "error",
                    reason: Some(reason),
                });
                engine_items.push(None);
            }
        }
    }
    let request = TrainingRequest {
        items: engine_items.iter().flatten().cloned().collect(),
        config: engine_config,
        output_dir: PathBuf::new(),
        file_name: "check.safetensors".to_owned(),
        trigger_words: Vec::new(),
        cancel: gen_core::CancelFlag::new(),
    };
    let mut policy = Value::Null;
    let mut config_error: Option<String> = None;
    match objective {
        IrisObjective::Depth => {
            let model_config = installed_export(data_dir, DEPTH_SUBDIR)
                .and_then(|dir| {
                    TaskExport::from_dir(&dir, IrisTask::Depth, depth_train::DEPTH_TRAINER_ID).ok()
                })
                .map(|export| export.config)
                .or_else(|| {
                    installed_backbone(data_dir)
                        .and_then(|dir| {
                            std::fs::read_to_string(dir.join(BACKBONE_CONFIG_FILE)).ok()
                        })
                        .and_then(|text| IrisConfig::parse(&text).ok())
                });
            match model_config
                .ok_or_else(|| {
                    "Install Iris 3B Depth (or Iris 3B) to check a depth dataset.".to_owned()
                })
                .and_then(|model| {
                    depth_train::DepthTrainPlan::resolve(&request, &model)
                        .map_err(|error| error.to_string())
                }) {
                Ok(plan) => {
                    policy = json!(match plan.invalid {
                        depth_train::InvalidPolicy::Reject => "reject",
                        depth_train::InvalidPolicy::Skip => "skip",
                    });
                    let mut engine = request.items.iter();
                    for (index, slot) in engine_items.iter().enumerate() {
                        let Some(_) = slot else { continue };
                        let engine_item = engine.next().expect("one engine item per slot");
                        let source = &dataset.items[index];
                        let (status, reason) = match depth_train::load_depth_sample(engine_item)
                            .map_err(|error| error.to_string())
                            .and_then(|sample| {
                                depth_train::covered_size(
                                    sample.height,
                                    sample.width,
                                    &plan.augment,
                                )
                                .map_err(|error| error.to_string())?;
                                Ok(sample)
                            }) {
                            Err(reason) => ("error", Some(reason)),
                            Ok(sample) => match depth_train::invalid_reason(
                                &sample,
                                &plan.loss,
                                plan.min_valid_pixels,
                            ) {
                                None => ("ok", None),
                                Some(reason) => ("invalid", Some(reason)),
                            },
                        };
                        items.push(DatasetCheckItem {
                            item_id: source.id.clone(),
                            display_name: source.display_name.clone(),
                            status,
                            reason,
                        });
                    }
                }
                Err(error) => config_error = Some(error),
            }
        }
        IrisObjective::Restoration => {
            match installed_export(data_dir, RESTORE_SUBDIR)
                .ok_or_else(|| {
                    "Install Iris 3B Restoration to check a restoration dataset.".to_owned()
                })
                .and_then(|dir| {
                    RestorationResources::from_dir(&dir, restore_train::TRAINER_ID)
                        .map_err(|error| error.to_string())
                })
                .and_then(|export| {
                    let has_parent = restore_options(config)
                        .ok()
                        .and_then(|options| options.get("init").cloned())
                        == Some(json!("parent"));
                    restore_train::RestoreTrainPlan::resolve(&request, &export, &[], has_parent)
                        .map_err(|error| error.to_string())
                }) {
                Ok(plan) => {
                    policy = json!(match plan.source {
                        restore_train::PairSource::Paired { .. } => "pairs",
                        restore_train::PairSource::Synthetic(_) => "synthetic",
                    });
                    let mut engine = request.items.iter();
                    for (index, slot) in engine_items.iter().enumerate() {
                        let Some(_) = slot else { continue };
                        let engine_item = engine.next().expect("one engine item per slot");
                        let source = &dataset.items[index];
                        let (status, reason) =
                            match restore_train::validate_item(engine_item, &plan) {
                                Ok(()) => ("ok", None),
                                Err(error) => ("error", Some(error.to_string())),
                            };
                        items.push(DatasetCheckItem {
                            item_id: source.id.clone(),
                            display_name: source.display_name.clone(),
                            status,
                            reason,
                        });
                    }
                }
                Err(error) => config_error = Some(error),
            }
        }
        IrisObjective::Generation => {}
    }
    let count = |status: &str| items.iter().filter(|item| item.status == status).count();
    let (ok, invalid, errors) = (count("ok"), count("invalid"), count("error"));
    let trainable = match policy.as_str() {
        Some("skip") => errors == 0 && ok > 0,
        _ => config_error.is_none() && errors == 0 && invalid == 0 && ok > 0,
    };
    json!({
        "objective": objective.task(),
        "policy": policy,
        "configError": config_error,
        "ok": ok,
        "invalid": invalid,
        "errors": errors,
        "trainable": trainable && config_error.is_none(),
        "items": items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_npy(path: &Path, h: usize, w: usize, f: impl Fn(usize, usize) -> f32) {
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}")
                .into_bytes();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        for y in 0..h {
            for x in 0..w {
                bytes.extend_from_slice(&f(y, x).to_le_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn sidecars_are_checked_by_the_engine_readers_for_format_and_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("rgb.png");
        image::RgbImage::new(32, 16).save(&image).unwrap();
        // .npy at the image size: aligned.
        let npy = dir.path().join("d.npy");
        write_npy(&npy, 16, 32, |y, _| 1.0 + y as f32);
        assert_eq!(check_depth_sidecar(&npy, &image).unwrap(), (32, 16));
        // 16-bit grayscale PNG: aligned.
        let png16 = dir.path().join("d16.png");
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(32, 16, image::Luma([1000]))
            .save(&png16)
            .unwrap();
        check_depth_sidecar(&png16, &image).unwrap();
        // 8-bit mask PNG.
        let mask = dir.path().join("m.png");
        image::GrayImage::from_pixel(32, 16, image::Luma([255]))
            .save(&mask)
            .unwrap();
        check_depth_sidecar(&mask, &image).unwrap();
        // Misaligned and colour planes are refused.
        let small = dir.path().join("small.npy");
        write_npy(&small, 8, 16, |_, _| 1.0);
        assert!(check_depth_sidecar(&small, &image)
            .unwrap_err()
            .contains("aligned pixel for pixel"));
        let rgb_plane = dir.path().join("rgb_plane.png");
        image::RgbImage::new(32, 16).save(&rgb_plane).unwrap();
        assert!(check_depth_sidecar(&rgb_plane, &image).is_err());

        // A ×4 aligned pair, and a misaligned one.
        let lq = dir.path().join("lq.png");
        image::RgbImage::new(8, 4).save(&lq).unwrap();
        assert_eq!(check_degraded_sidecar(&lq, &image).unwrap(), 4);
        let skewed = dir.path().join("skewed.png");
        image::RgbImage::new(8, 8).save(&skewed).unwrap();
        assert!(check_degraded_sidecar(&skewed, &image).is_err());
    }

    #[test]
    fn task_items_resolve_sidecars_under_the_dataset_root_into_the_engine_shape() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("sidecars")).unwrap();
        for file in [
            "sidecars/a.depth.npy",
            "sidecars/a.mask.png",
            "sidecars/a.degraded.png",
        ] {
            std::fs::write(root.path().join(file), b"x").unwrap();
        }
        let image = root.path().join("images/a.png");
        let extra: sceneworks_core::contracts::ExtraFields = serde_json::from_value(json!({
            "irisDepth": {"path": "sidecars/a.depth.npy", "maskPath": "sidecars/a.mask.png",
                          "scale": 0.001, "maxDepth": 80.0},
            "irisDegradedImage": "sidecars/a.degraded.png",
            "captions": {"short": "ignored"}
        }))
        .unwrap();
        let item = task_item(IrisObjective::Depth, root.path(), image.clone(), &extra).unwrap();
        assert_eq!(item.caption, "");
        assert_eq!(item.model_options.len(), 1, "only the depth target travels");
        let depth = &item.model_options["depth"];
        assert_eq!(
            depth["path"],
            json!(root.path().join("sidecars").join("a.depth.npy"))
        );
        assert_eq!(
            depth["mask_path"],
            json!(root.path().join("sidecars").join("a.mask.png"))
        );
        assert_eq!(depth["scale"], json!(0.001));
        assert_eq!(depth["max_depth"], json!(80.0));
        assert!(depth.get("min_depth").is_none());

        let pair = task_item(
            IrisObjective::Restoration,
            root.path(),
            image.clone(),
            &extra,
        )
        .unwrap();
        assert_eq!(pair.model_options.len(), 1);
        assert_eq!(
            pair.model_options["degraded_image"],
            json!(root.path().join("sidecars").join("a.degraded.png"))
        );
        let clean = task_item(
            IrisObjective::Restoration,
            root.path(),
            image.clone(),
            &Default::default(),
        )
        .unwrap();
        assert!(
            clean.model_options.is_empty(),
            "a clean image degrades synthetically"
        );

        // A traversal or a missing file is refused.
        let escape: sceneworks_core::contracts::ExtraFields =
            serde_json::from_value(json!({"irisDepth": {"path": "../x.npy"}})).unwrap();
        assert!(task_item(IrisObjective::Depth, root.path(), image.clone(), &escape).is_err());
        let gone: sceneworks_core::contracts::ExtraFields =
            serde_json::from_value(json!({"irisDepth": {"path": "sidecars/none.npy"}})).unwrap();
        assert!(task_item(IrisObjective::Depth, root.path(), image, &gone).is_err());
    }

    #[test]
    fn the_task_config_carries_only_its_options_bag() {
        let config = TrainingConfig {
            model_options: serde_json::from_value(json!({
                "iris": {"invalid_examples": "skip"},
                "networkType": "full",
                "sampleEvery": 64,
                "depthInit": "export"
            }))
            .unwrap(),
            sample_prompts: vec!["x".into()],
            ..TrainingConfig::default()
        };
        let shaped = shape_task_config(config, IrisObjective::Depth);
        assert_eq!(
            Value::Object(shaped.model_options.clone()),
            json!({"iris": {"invalid_examples": "skip"}})
        );
        assert!(shaped.sample_prompts.is_empty());
        assert_eq!(shaped.loss_type, "l1");

        let restore = TrainingConfig {
            model_options: serde_json::from_value(
                json!({"iris_restore": {"crop": "center"}, "x": 1}),
            )
            .unwrap(),
            ..TrainingConfig::default()
        };
        let shaped = shape_task_config(restore, IrisObjective::Restoration);
        assert_eq!(
            Value::Object(shaped.model_options),
            json!({"iris_restore": {"crop": "center"}})
        );
    }

    #[test]
    fn the_pinned_restoration_resources_are_the_engines_three() {
        let components: Vec<&str> = TRAINING_RESOURCES.iter().map(|r| r.component).collect();
        let catalog: Vec<&str> = RESTORE_TRAINING_RESOURCE_MODELS
            .iter()
            .map(|(_, component)| *component)
            .collect();
        assert_eq!(components, catalog);
    }

    const DEPTH_CONFIG: &str = include_str!("iris_restore/fixtures/depth_config.yaml");
    const UPSCALER_CONFIG: &str = include_str!("iris_restore/fixtures/upscaler_config.yaml");

    fn isolate() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn stage_export_dir(dir: &Path, config: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.yaml"), config).unwrap();
        let task = if config == DEPTH_CONFIG {
            IrisTask::Depth
        } else {
            IrisTask::Restoration
        };
        let weights = crate::iris_test_weights::task_weights(task);
        for file in ["model.safetensors", "empty_prompt.safetensors"] {
            std::fs::write(dir.join(file), &weights).unwrap();
        }
    }

    fn pinned_export(data: &Path, subdir: &str) -> PathBuf {
        sceneworks_core::hf_home::huggingface_repo_cache_path(data, UPSTREAM_WEIGHTS_REPO)
            .unwrap()
            .join("snapshots")
            .join(UPSTREAM_WEIGHTS_REVISION)
            .join(subdir)
    }

    /// A trained model is selected by id and confined to its own run directory; its export must be
    /// the asked-for task (a restoration model is never a depth model), and an id is one component.
    #[test]
    fn trained_task_exports_resolve_by_id_with_their_task_checked() {
        let data = tempfile::tempdir().unwrap();
        let run = data.path().join("models/finetunes/finetune_depth1");
        stage_export_dir(&run.join("my_depth"), DEPTH_CONFIG);
        // The run's checkpoints and previews sit beside the export and are never candidates.
        std::fs::create_dir_all(run.join("my_depth.checkpoints/step_00000002")).unwrap();
        let found =
            trained_task_export_dir(data.path(), "finetune_depth1", IrisTask::Depth).unwrap();
        assert!(found.ends_with("my_depth"), "{}", found.display());
        let wrong = trained_task_export_dir(data.path(), "finetune_depth1", IrisTask::Restoration)
            .expect_err("a depth export is not a restoration model");
        assert!(wrong.contains("not an Iris restoration export"), "{wrong}");
        let restore_run = data.path().join("models/finetunes/finetune_restore1");
        stage_export_dir(&restore_run.join("r"), UPSCALER_CONFIG);
        trained_task_export_dir(data.path(), "finetune_restore1", IrisTask::Restoration).unwrap();
        assert!(trained_task_export_dir(data.path(), "../escape", IrisTask::Depth).is_err());
        assert_eq!(
            trained_task_export_dir(data.path(), "finetune_none", IrisTask::Depth).unwrap_err(),
            "the trained depth model finetune_none is not installed (its training output is \
             missing) — pick another model or retrain it"
        );
        std::fs::remove_file(run.join("my_depth/empty_prompt.safetensors")).unwrap();
        assert_eq!(
            trained_task_export_dir(data.path(), "finetune_depth1", IrisTask::Depth).unwrap_err(),
            "the trained depth model finetune_depth1 has no complete export (config.yaml, \
             model.safetensors, empty_prompt.safetensors)"
        );
    }

    fn depth_dataset(
        root: &Path,
        items: &[(&str, Value)],
    ) -> sceneworks_core::training::TrainingDataset {
        let entries: Vec<Value> = items
            .iter()
            .map(|(id, depth)| {
                json!({
                    "id": id, "path": format!("images/{id}.png"), "displayName": id,
                    "caption": {"text": "", "source": "manual", "triggerWords": []},
                    "addedAt": "2026-10-11T00:00:00Z", "irisDepth": depth
                })
            })
            .collect();
        let _ = root;
        serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds", "version": 1, "projectId": "p", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": entries
        }))
        .unwrap()
    }

    /// The pre-launch depth report runs the trainer's own per-item pass: an aligned ramp is ok, a
    /// constant plane (no relative normalization) is INVALID — what the run's explicit policy decides
    /// — and a misaligned map is an ERROR either way; a validity mask removes pixels from the count.
    #[test]
    fn the_depth_dataset_report_applies_the_invalid_example_policy() {
        let _env = isolate();
        let data = tempfile::tempdir().unwrap();
        stage_export_dir(&pinned_export(data.path(), "depth"), DEPTH_CONFIG);
        let root = data.path().join("ds");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::create_dir_all(root.join("sidecars")).unwrap();
        for id in ["ramp", "flat", "skewed", "masked", "void", "unmasked"] {
            image::RgbImage::new(64, 48)
                .save(root.join(format!("images/{id}.png")))
                .unwrap();
        }
        write_npy(&root.join("sidecars/ramp.npy"), 48, 64, |y, _| {
            1.0 + y as f32
        });
        write_npy(&root.join("sidecars/flat.npy"), 48, 64, |_, _| 3.0);
        write_npy(&root.join("sidecars/skewed.npy"), 32, 64, |y, _| {
            1.0 + y as f32
        });
        write_npy(&root.join("sidecars/masked.npy"), 48, 64, |y, _| {
            1.0 + y as f32
        });
        // NO valid depth pixel at all (sc-25690): a plane of zeros, and a usable ramp whose mask
        // excludes every pixel.
        write_npy(&root.join("sidecars/void.npy"), 48, 64, |_, _| 0.0);
        write_npy(&root.join("sidecars/unmasked.npy"), 48, 64, |y, _| {
            1.0 + y as f32
        });
        image::GrayImage::new(64, 48)
            .save(root.join("sidecars/unmasked.mask.png"))
            .unwrap();
        // A mask leaving a single valid pixel: under the default min_valid_pixels (2) — invalid.
        image::GrayImage::from_fn(64, 48, |x, y| {
            image::Luma([u8::from(x == 0 && y == 0) * 255])
        })
        .save(root.join("sidecars/masked.mask.png"))
        .unwrap();
        let dataset = depth_dataset(
            &root,
            &[
                ("ramp", json!({"path": "sidecars/ramp.npy"})),
                ("flat", json!({"path": "sidecars/flat.npy"})),
                ("skewed", json!({"path": "sidecars/skewed.npy"})),
                (
                    "masked",
                    json!({"path": "sidecars/masked.npy", "maskPath": "sidecars/masked.mask.png"}),
                ),
                ("void", json!({"path": "sidecars/void.npy"})),
                (
                    "unmasked",
                    json!({"path": "sidecars/unmasked.npy", "maskPath": "sidecars/unmasked.mask.png"}),
                ),
            ],
        );
        let mut config = sceneworks_core::training::iris::depth::iris_depth_target().defaults;
        config.advanced["iris"]["depth_size"] = json!([32, 32]);
        config.resolution = 32;
        let report =
            iris_task_dataset_report(data.path(), IrisObjective::Depth, &config, &dataset, &root);
        assert_eq!(report["policy"], "reject", "{report:#}");
        assert_eq!(report["configError"], Value::Null, "{report:#}");
        let status = |id: &str| {
            report["items"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["itemId"] == id)
                .map(|item| item["status"].clone())
                .unwrap()
        };
        assert_eq!(status("ramp"), "ok");
        assert_eq!(status("flat"), "invalid");
        assert_eq!(status("skewed"), "error");
        assert_eq!(status("masked"), "invalid");
        for id in ["void", "unmasked"] {
            let item = report["items"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["itemId"] == id)
                .unwrap();
            assert_eq!(item["status"], "invalid", "{item:#}");
            assert_eq!(
                item["reason"], "0 valid depth pixels (min_valid_pixels = 2)",
                "{item:#}"
            );
        }
        assert_eq!(
            (report["ok"].as_u64(), report["invalid"].as_u64()),
            (Some(1), Some(4))
        );
        // A dataset of nothing but no-valid-pixel items is not trainable under either policy.
        let void_only = depth_dataset(
            &root,
            &[
                ("void", json!({"path": "sidecars/void.npy"})),
                (
                    "unmasked",
                    json!({"path": "sidecars/unmasked.npy", "maskPath": "sidecars/unmasked.mask.png"}),
                ),
            ],
        );
        for policy in ["reject", "skip"] {
            let mut config = config.clone();
            config.advanced["iris"]["invalid_examples"] = json!(policy);
            let report = iris_task_dataset_report(
                data.path(),
                IrisObjective::Depth,
                &config,
                &void_only,
                &root,
            );
            assert_eq!(
                (report["ok"].as_u64(), report["invalid"].as_u64()),
                (Some(0), Some(2)),
                "{policy}: {report:#}"
            );
            assert_eq!(report["trainable"], false, "{policy}: {report:#}");
        }
        assert_eq!(report["trainable"], false);

        // `skip` drops the invalid items — but a misaligned one is still refused.
        config.advanced["iris"]["invalid_examples"] = json!("skip");
        let skip =
            iris_task_dataset_report(data.path(), IrisObjective::Depth, &config, &dataset, &root);
        assert_eq!(skip["policy"], "skip");
        assert_eq!(skip["trainable"], false, "{skip:#}");
        let aligned = depth_dataset(
            &root,
            &[
                ("ramp", json!({"path": "sidecars/ramp.npy"})),
                ("flat", json!({"path": "sidecars/flat.npy"})),
            ],
        );
        let skip =
            iris_task_dataset_report(data.path(), IrisObjective::Depth, &config, &aligned, &root);
        assert_eq!(skip["trainable"], true, "{skip:#}");
        config.advanced["iris"]["invalid_examples"] = json!("reject");
        let reject =
            iris_task_dataset_report(data.path(), IrisObjective::Depth, &config, &aligned, &root);
        assert_eq!(
            reject["trainable"], false,
            "reject refuses the flat item: {reject:#}"
        );
    }

    /// The restoration report: aligned ×4 pairs pass the engine's pair rule; a misaligned pair is an
    /// error; without the restoration install the check says what to install.
    #[test]
    fn the_restoration_dataset_report_checks_pairs_with_the_engine_rule() {
        let _env = isolate();
        let data = tempfile::tempdir().unwrap();
        let root = data.path().join("ds");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::create_dir_all(root.join("sidecars")).unwrap();
        for id in ["a", "b"] {
            image::RgbImage::new(256, 256)
                .save(root.join(format!("images/{id}.png")))
                .unwrap();
        }
        image::RgbImage::new(64, 64)
            .save(root.join("sidecars/a.png"))
            .unwrap();
        image::RgbImage::new(60, 64)
            .save(root.join("sidecars/b.png"))
            .unwrap();
        let dataset: sceneworks_core::training::TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds", "version": 1, "projectId": "p", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": [
                {"id": "a", "path": "images/a.png", "displayName": "a",
                 "caption": {"text": "", "source": "manual", "triggerWords": []},
                 "addedAt": "2026-10-11T00:00:00Z", "irisDegradedImage": "sidecars/a.png"},
                {"id": "b", "path": "images/b.png", "displayName": "b",
                 "caption": {"text": "", "source": "manual", "triggerWords": []},
                 "addedAt": "2026-10-11T00:00:00Z", "irisDegradedImage": "sidecars/b.png"}
            ]
        }))
        .unwrap();
        let mut config = sceneworks_core::training::iris::restore::iris_restore_target().defaults;
        config.resolution = 256;
        let missing = iris_task_dataset_report(
            data.path(),
            IrisObjective::Restoration,
            &config,
            &dataset,
            &root,
        );
        assert!(missing["configError"]
            .as_str()
            .unwrap()
            .contains("Install Iris 3B Restoration"));
        stage_export_dir(&pinned_export(data.path(), "upscaler"), UPSCALER_CONFIG);
        let report = iris_task_dataset_report(
            data.path(),
            IrisObjective::Restoration,
            &config,
            &dataset,
            &root,
        );
        assert_eq!(report["policy"], "pairs", "{report:#}");
        assert_eq!(report["items"][0]["status"], "ok", "{report:#}");
        assert_eq!(report["items"][1]["status"], "error", "{report:#}");
        assert_eq!(report["trainable"], false);

        // The Training Studio's fixture IS this report (only the engine's reason wording, which
        // carries temp paths, is normalized): the web summary is tested against the real shape.
        let normalized = |mut value: Value| {
            for item in value["items"].as_array_mut().unwrap() {
                if !item["reason"].is_null() {
                    item["reason"] = json!("<reason>");
                }
            }
            value
        };
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../apps/web/src/training/irisDatasetReport.fixture.json"
        ))
        .unwrap();
        assert_eq!(normalized(report), normalized(fixture));
    }

    /// The upload-time pair check agrees with the engine's own per-item pass (`validate_item`, run by
    /// the dataset report) on every crop-compatible pair: aligned ×1/×2/×4 pairs pass both, a skewed
    /// or non-integer pair fails both. (Whether a ×f pair also holds the run's crop depends on the
    /// run config, so only the report — which has it — judges that.)
    #[test]
    fn the_upload_pair_check_agrees_with_the_engine_item_check() {
        let _env = isolate();
        let data = tempfile::tempdir().unwrap();
        stage_export_dir(&pinned_export(data.path(), "upscaler"), UPSCALER_CONFIG);
        let root = data.path().join("ds");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::create_dir_all(root.join("sidecars")).unwrap();
        let cases: [((u32, u32), (u32, u32)); 8] = [
            ((256, 256), (256, 256)),
            ((256, 256), (128, 128)),
            ((256, 256), (64, 64)),
            ((512, 256), (128, 64)),
            ((256, 256), (60, 64)),
            ((256, 256), (128, 64)),
            ((512, 256), (128, 128)),
            ((256, 256), (96, 96)),
        ];
        let mut items = Vec::new();
        let mut upload = Vec::new();
        for (index, ((w, h), (lw, lh))) in cases.iter().enumerate() {
            let id = format!("p{index}");
            image::RgbImage::new(*w, *h)
                .save(root.join(format!("images/{id}.png")))
                .unwrap();
            image::RgbImage::new(*lw, *lh)
                .save(root.join(format!("sidecars/{id}.png")))
                .unwrap();
            upload.push(
                check_degraded_sidecar(
                    &root.join(format!("sidecars/{id}.png")),
                    &root.join(format!("images/{id}.png")),
                )
                .is_ok(),
            );
            items.push(json!({
                "id": id, "path": format!("images/{id}.png"), "displayName": id,
                "caption": {"text": "", "source": "manual", "triggerWords": []},
                "addedAt": "2026-10-11T00:00:00Z", "irisDegradedImage": format!("sidecars/{id}.png")
            }));
        }
        let dataset: sceneworks_core::training::TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds", "version": 1, "projectId": "p", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": items
        }))
        .unwrap();
        let mut config = sceneworks_core::training::iris::restore::iris_restore_target().defaults;
        config.resolution = 256;
        let report = iris_task_dataset_report(
            data.path(),
            IrisObjective::Restoration,
            &config,
            &dataset,
            &root,
        );
        assert_eq!(report["policy"], "pairs", "{report:#}");
        let engine: Vec<bool> = report["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["status"] == "ok")
            .collect();
        assert_eq!(
            upload,
            [true, true, true, true, false, false, false, false],
            "{report:#}"
        );
        assert_eq!(upload, engine, "{report:#}");
    }

    /// The restoration target's defaults ARE the engine's published recipe, read back through the
    /// worker's own SceneWorks → engine config translation, and its options bag carries the engine's
    /// recipe constants.
    #[test]
    fn restoration_defaults_are_the_engine_recipe() {
        let target = sceneworks_core::training::iris::restore::iris_restore_target();
        let engine = sw_config_to_engine(&target.defaults);
        let recipe = restore_train::recipe_training_config();
        assert_eq!(engine.learning_rate, restore_train::RECIPE_LR);
        assert_eq!(engine.learning_rate, recipe.learning_rate);
        assert_eq!(engine.weight_decay, restore_train::RECIPE_WEIGHT_DECAY);
        assert_eq!(engine.weight_decay, recipe.weight_decay);
        assert_eq!(engine.resolution, restore_train::RECIPE_RESOLUTION);
        assert_eq!(
            sceneworks_core::training::iris::restore::RECIPE_RESOLUTION,
            restore_train::RECIPE_RESOLUTION
        );
        assert_eq!(target.defaults.advanced["networkType"], json!("full"));
        assert!(engine.full_finetune && recipe.full_finetune);
        assert_eq!(engine.lr_scheduler, recipe.lr_scheduler);
        assert_eq!(engine.lr_scheduler, gen_core::LrSchedule::Constant);
        assert_eq!(engine.lr_warmup_steps, recipe.lr_warmup_steps);
        assert_eq!(engine.lr_warmup_steps, 0);
        assert_eq!(engine.optimizer, recipe.optimizer);
        assert_eq!(engine.optimizer, "adamw");
        let bag = &target.defaults.advanced["iris_restore"];
        let weights = restore_train::RECIPE_LOSS_WEIGHTS;
        assert_eq!(bag["loss_l2"], json!(weights.l2));
        assert_eq!(bag["loss_lpips"], json!(weights.lpips));
        assert_eq!(bag["loss_gan"], json!(weights.gan));
        assert_eq!(bag["gan_real_label"], json!(restore_train::GAN_REAL_LABEL));
        let (b1, b2) = restore_train::RECIPE_BETAS;
        assert_eq!(bag["betas"], json!([b1, b2]));
        assert_eq!(bag["gradient_clip"], json!(restore_train::RECIPE_CLIP));
        assert_eq!(
            bag["critic_gradient_clip"],
            json!(restore_train::RECIPE_CLIP)
        );
        assert_eq!(bag["ema_decay"], json!(restore_train::RECIPE_EMA_DECAY));
    }

    /// Legacy task targets retain integral alpha semantics after the product surface moved to a
    /// JSON number for MOSS fractions. Mutation: round or drop the checked bridge ⇒ red.
    #[test]
    fn task_alpha_maps_integral_contract_numbers_exactly() {
        let mut config = sceneworks_core::training::iris::restore::iris_restore_target().defaults;
        config.alpha = serde_json::Number::from(7);
        assert_eq!(sw_config_to_engine(&config).alpha, 7.0);
    }

    /// The depth target's defaults ARE the engine's paper recipe (`apply_paper_recipe`): the global
    /// batch and optimizer-step count survive the batch × accumulation split.
    #[test]
    fn depth_defaults_are_the_engine_paper_recipe() {
        use sceneworks_core::training::iris::depth as sw_depth;
        let target = sw_depth::iris_depth_target();
        let defaults = &target.defaults;
        let engine = shape_task_config(sw_config_to_engine(defaults), IrisObjective::Depth);
        let mut recipe = TrainingConfig::default();
        depth_train::apply_paper_recipe(&mut recipe);
        assert_eq!(engine.learning_rate, depth_train::PAPER_LEARNING_RATE);
        assert_eq!(engine.learning_rate, recipe.learning_rate);
        assert_eq!(engine.weight_decay, depth_train::PAPER_WEIGHT_DECAY);
        assert_eq!(engine.weight_decay, recipe.weight_decay);
        assert_eq!(engine.lr_scheduler, recipe.lr_scheduler);
        assert_eq!(engine.lr_warmup_steps, recipe.lr_warmup_steps);
        assert_eq!(engine.optimizer, recipe.optimizer);
        assert!(engine.full_finetune && recipe.full_finetune);
        assert_eq!(defaults.advanced["lossType"], json!(recipe.loss_type));
        assert_eq!(engine.loss_type, recipe.loss_type);
        assert_eq!(
            defaults.batch_size * defaults.gradient_accumulation,
            depth_train::PAPER_GLOBAL_BATCH
        );
        assert_eq!(
            recipe.batch_size * recipe.gradient_accumulation,
            depth_train::PAPER_GLOBAL_BATCH
        );
        assert_eq!(defaults.steps % defaults.gradient_accumulation, 0);
        assert_eq!(
            defaults.steps / defaults.gradient_accumulation,
            depth_train::PAPER_STEPS
        );
        assert_eq!(recipe.steps, depth_train::PAPER_STEPS);
        assert_eq!(
            sw_depth::PAPER_GLOBAL_BATCH,
            depth_train::PAPER_GLOBAL_BATCH
        );
        assert_eq!(sw_depth::PAPER_OPTIMIZER_STEPS, depth_train::PAPER_STEPS);
        let bag = &defaults.advanced["iris"];
        assert_eq!(
            bag["gradient_weight"],
            json!(depth_train::DEFAULT_GRADIENT_WEIGHT)
        );
        assert_eq!(
            bag["low_percentile"],
            json!(depth_train::DEFAULT_LOW_PERCENTILE)
        );
        assert_eq!(
            bag["high_percentile"],
            json!(depth_train::DEFAULT_HIGH_PERCENTILE)
        );
        let (b1, b2) = depth_train::DEFAULT_BETAS;
        assert_eq!(bag["betas"], json!([b1, b2]));
        assert_eq!(
            bag["gradient_clip"],
            json!(depth_train::DEFAULT_GRADIENT_CLIP)
        );
        assert_eq!(bag["hflip_prob"], json!(depth_train::DEFAULT_HFLIP_PROB));
        assert_eq!(
            bag["min_valid_pixels"],
            json!(depth_train::MIN_VALID_PIXELS_FLOOR)
        );
        assert_eq!(
            bag["preview_items"],
            json!(depth_train::DEFAULT_PREVIEW_ITEMS)
        );
    }

    /// The client allow-lists are the engines' option keys minus the path options SceneWorks resolves
    /// from a trusted previous job — as SETS, so a key added to (or dropped from) either side fails.
    #[test]
    fn the_client_option_allow_lists_are_the_engines_keys() {
        use std::collections::BTreeSet;
        let client: BTreeSet<&str> =
            sceneworks_core::training::iris::depth::DEPTH_CLIENT_OPTION_KEYS
                .iter()
                .copied()
                .chain(["load_from", "resume_from"])
                .collect();
        let engine: BTreeSet<&str> = depth_train::DEPTH_OPTION_KEYS.iter().copied().collect();
        assert_eq!(client, engine);

        // The restoration engine keeps its list private: it must ACCEPT every client key, and its
        // refusal of an unknown key names the full list, which must be the client keys + resume_from.
        let restore_client = sceneworks_core::training::iris::restore::RESTORE_CLIENT_OPTION_KEYS;
        let bag = |keys: &[&str]| TrainingConfig {
            model_options: Map::from_iter([(
                restore_train::OPTIONS_KEY.to_owned(),
                Value::Object(
                    keys.iter()
                        .map(|key| ((*key).to_owned(), Value::Null))
                        .collect(),
                ),
            )]),
            ..TrainingConfig::default()
        };
        restore_train::RestoreOptions::from_config(&bag(&restore_client))
            .expect("the engine accepts every client restoration key");
        let refusal = restore_train::RestoreOptions::from_config(&bag(&["not_an_option"]))
            .expect_err("an unknown key is refused")
            .to_string();
        let known = refusal
            .split_once("(known: ")
            .and_then(|(_, rest)| rest.strip_suffix(')'))
            .unwrap_or_else(|| panic!("the refusal names the known keys: {refusal}"));
        let engine: BTreeSet<&str> = known.split(", ").collect();
        let client: BTreeSet<&str> = restore_client
            .iter()
            .copied()
            .chain(["resume_from"])
            .collect();
        assert_eq!(client, engine);
    }

    /// Each training-only resource's Model Manager entry downloads exactly the engine's pinned file:
    /// the worker stages it by the ENGINE's repo/revision, so a drifted catalog entry would install
    /// a file the run then cannot find.
    #[test]
    fn the_training_resource_catalog_entries_are_the_engines_pins() {
        for (catalog_id, component) in RESTORE_TRAINING_RESOURCE_MODELS {
            let resource = TRAINING_RESOURCES
                .iter()
                .find(|resource| resource.component == component)
                .unwrap_or_else(|| panic!("{component} is an engine training resource"));
            let entry = crate::training_jobs::builtin_model_manifest_entry(catalog_id)
                .unwrap_or_else(|| panic!("{catalog_id} is in the builtin catalog"));
            assert_eq!(entry["componentOnly"], json!(true), "{catalog_id}");
            let downloads = entry["downloads"].as_array().unwrap();
            assert_eq!(downloads.len(), 1, "{catalog_id}");
            let download = &downloads[0];
            assert_eq!(download["repo"], json!(resource.repo), "{catalog_id}");
            assert_eq!(
                download["revision"],
                json!(resource.revision),
                "{catalog_id}"
            );
            assert_eq!(download["files"], json!([resource.file]), "{catalog_id}");
            assert_eq!(
                download["estimatedSizeBytes"],
                json!(resource.bytes),
                "{catalog_id}"
            );
        }
    }

    /// The engine's degradation parser judges the synthetic-degradation overrides.
    #[test]
    fn degradation_overrides_go_through_the_engine_parser() {
        check_restore_degradation(&json!({"use_hflip": false}), 1024).unwrap();
        assert!(check_restore_degradation(&json!({"not_a_key": 1}), 1024).is_err());
        assert!(check_restore_degradation(&json!({"resize_back": false}), 1024).is_err());
    }
}
