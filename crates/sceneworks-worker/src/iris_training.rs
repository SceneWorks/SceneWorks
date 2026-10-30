//! Worker-side seams of Iris-3B generation training (epic sc-25678; S7 sc-25685 MLX, S8 sc-25686
//! Candle/CUDA).
//!
//! The trainer itself is `load_trainer("iris_3b", …)` — reached through the generic native training
//! executor in `training_jobs`, exactly like every other family. This module carries only what that
//! executor cannot know about Iris:
//!
//! * the **unused-tail** report — upstream's single-device ranged sampler walks only the first
//!   `640 · ⌊n / 640⌋` items of an `n > 640` dataset that is not a multiple of 640 (641–1279 items
//!   train on 640). The trainer prints that on stderr only, so SceneWorks computes it from the SAME
//!   `gen_core_iris::train::DataWalk` the trainer walks and surfaces it before launch (the API's
//!   data-walk endpoint) and in the job (progress message, structured log event and result);
//! * the **memory floor** gate — a routed Iris training lane is never admitted unbounded: a host
//!   under its floor row (`sceneworks_core::training::iris::floors`, GiB against a capacity in GiB)
//!   is refused before the model loads (and so is a host whose capacity cannot be read). The
//!   refusal names the row's own provenance — measured, extrapolated, lower bound or declared.

use std::path::{Path, PathBuf};

use gen_core_iris::train::{DataWalk, CANONICAL_CHUNKS};
use serde::Serialize;

/// An Iris training run's checkpoint root, `<output_dir>/<stem>.checkpoints` — the engine's own
/// `gen_core_iris::train::checkpoint_root`, so resume / load-from / metrics read exactly where the
/// trainers write.
pub fn iris_checkpoint_root(output_dir: &Path, file_name: &str) -> PathBuf {
    gen_core_iris::train::checkpoint_root(output_dir, file_name)
}

/// Every checkpoint directory an Iris training run leaves beside its export (sc-25690): the
/// checkpoint root and the single sibling `<root>.prior` the trainers retire foreign or abandoned
/// checkpoints into (`gen_core_iris::train::lineage::prior_checkpoint_root`). Both are part of the
/// run's on-disk footprint, so whatever deletes a run's files deletes these too — taken from the
/// engine contract, never re-derived here.
pub fn iris_run_checkpoint_dirs(output_dir: &Path, file_name: &str) -> [PathBuf; 2] {
    [
        iris_checkpoint_root(output_dir, file_name),
        gen_core_iris::train::lineage::prior_checkpoint_root(output_dir, file_name),
    ]
}

/// The pre-launch view of the trainer's data walk for a dataset of `items` at `batch_size`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IrisDataWalkReport {
    pub items: usize,
    pub covered: usize,
    pub unused_items: usize,
    pub batch_size: usize,
    pub chunks: usize,
    /// The trainer's own run-start warning text (`DataWalk::unused_tail_warning`), or `None` when
    /// every item is trained on.
    pub warning: Option<String>,
}

/// Compute the Iris data walk for `items` dataset items at `batch_size`. `None` for an empty
/// dataset or a zero batch (the trainer refuses both before walking).
pub fn iris_data_walk_report(items: usize, batch_size: usize) -> Option<IrisDataWalkReport> {
    let walk = DataWalk::new(items, batch_size).ok()?;
    Some(IrisDataWalkReport {
        items: walk.items,
        covered: walk.covered,
        unused_items: walk.unused_items(),
        batch_size: walk.batch_size,
        chunks: CANONICAL_CHUNKS,
        warning: walk.unused_tail_warning(),
    })
}

/// The memory lane this build trains on: `"mlx"` (Apple-Silicon unified memory) or `"candle"`
/// (NVIDIA VRAM).
pub(crate) fn training_lane() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    }
}

/// The accelerator a rank occupies on `lane` (refusal text).
pub(crate) fn device_noun(lane: &str) -> &'static str {
    if lane == "mlx" {
        "Mac GPU"
    } else {
        "GPU"
    }
}

/// The admission decision for a whole Iris run (sc-25689) on a worker with `devices` accelerators:
/// the floor of ONE rank (+ the REPA teacher) times the ranks sharing the busiest device, against
/// that device's capacity. Missing evidence refuses, exactly like [`iris_memory_refusal`].
pub(crate) fn iris_run_memory_refusal(
    lane: &str,
    config: &sceneworks_core::training::TrainingConfig,
    devices: u64,
    capacity_gib: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::curricula;
    let world = curricula::iris_world_size(config);
    let ranks = curricula::ranks_per_device(world, devices);
    let network_type = config
        .advanced
        .get("networkType")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("lora");
    let floor = curricula::iris_run_memory_floor(lane, config, devices);
    let repa = sceneworks_core::training::iris::iris_options(config)
        .map(|options| curricula::iris_repa_enabled(&options))
        .unwrap_or(false);
    let mut shape = run_shape(network_type, config.resolution);
    if repa {
        shape.push_str(" with REPA (+ the DINOv2 teacher)");
    }
    if ranks > 1 {
        shape.push_str(&format!(
            " × {ranks} ranks sharing one {}",
            device_noun(lane)
        ));
    }
    let advice = match activation_checkpointing_hint(lane, config, devices, capacity_gib) {
        Some(hint) => format!("{hint} Otherwise: {GENERATION_ADVICE}"),
        None => GENERATION_ADVICE.to_owned(),
    };
    memory_refusal_for_floor(lane, floor, &shape, capacity_gib, "", &advice)
}

const GENERATION_ADVICE: &str = "lower the training resolution, train a LoRA/LoKr adapter instead \
                                 of a full fine-tune, run fewer ranks on one device, or run on a \
                                 machine with more memory.";
const TASK_ADVICE: &str = "lower the batch size or the training size, train a LoRA/LoKr where the \
                           objective offers one, or run on a machine with more memory.";
const TASK_NO_ROW: &str =
    " Lower the batch size or the training size (raise gradient accumulation \
                           to keep the global batch), or train a LoRA/LoKr where the objective \
                           offers one.";

/// sc-25690: a FULL generation run refused at its plain row is told when the SAME run under
/// activation checkpointing (`advanced.iris.activation_checkpointing` = `full` or `selective_op`,
/// the `full_ac` rows) fits this machine — the one change that admits it without giving up the
/// resolution or the full fine-tune. `None` when the run already checkpoints every region, is not
/// a full run, or would still not fit.
fn activation_checkpointing_hint(
    lane: &str,
    config: &sceneworks_core::training::TrainingConfig,
    devices: u64,
    capacity_gib: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::{
        curricula, iris_full_activation_checkpointing, IRIS_OPTIONS_KEY,
    };
    let full = config
        .advanced
        .get("networkType")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|network| network.trim().eq_ignore_ascii_case("full"));
    if !full || iris_full_activation_checkpointing(config) {
        return None;
    }
    let mut checkpointed = config.clone();
    let options = checkpointed
        .advanced
        .entry(IRIS_OPTIONS_KEY.to_owned())
        .or_insert_with(|| serde_json::json!({}));
    options
        .as_object_mut()?
        .insert("activation_checkpointing".to_owned(), "full".into());
    let floor = curricula::iris_run_memory_floor(lane, &checkpointed, devices)?;
    under_floor(lane, floor.gib, capacity_gib?)
        .is_none()
        .then(|| {
            format!(
                "With activation checkpointing (Memory → Activation checkpointing = full or \
                 selective_op) the same run needs {} GiB ({}) and fits this machine, at the cost \
                 of slower steps.",
                floor.gib,
                floor.provenance.describe()
            )
        })
}

fn run_shape(network_type: &str, resolution: u32) -> String {
    let artifact = if network_type.trim().eq_ignore_ascii_case("full") {
        "a full fine-tune"
    } else {
        "an adapter (LoRA/LoKr) run"
    };
    format!("{artifact} at {resolution}px")
}

fn capacity_noun(lane: &str) -> &'static str {
    if lane == "mlx" {
        "unified memory"
    } else {
        "GPU memory (VRAM)"
    }
}

/// The admission decision for an Iris run against a host of `capacity_gib` on `lane`:
/// `Some(message)` to refuse, `None` to admit. Missing evidence REFUSES (the inverse of the
/// generation fit gate's fail-open): the floors are the only bound a training lane has, so an
/// unreadable host or a shape no row covers is never admitted unbounded.
#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
pub(crate) fn iris_memory_refusal(
    lane: &str,
    network_type: &str,
    resolution: u32,
    capacity_gib: Option<f64>,
) -> Option<String> {
    let floor = sceneworks_core::training::iris::iris_generation_memory_floor(
        lane,
        network_type,
        resolution,
        false,
    );
    memory_refusal_for_floor(
        lane,
        floor,
        &run_shape(network_type, resolution),
        capacity_gib,
        "",
        GENERATION_ADVICE,
    )
}

/// `Some((usable, reserve))` when a host of `capacity_gib` (GiB) is under a floor of `floor_gib`
/// (GiB) on `lane`. Unified memory is shared with macOS and the other apps, so an MLX floor is
/// judged against the pool minus the same OS/app reserve the MLX generation gate holds out. (A
/// Candle floor already folds its allocator reserve into the row; dedicated VRAM has no OS share.)
fn under_floor(lane: &str, floor_gib: u32, capacity_gib: f64) -> Option<(f64, f64)> {
    let reserve = if lane == "mlx" {
        crate::mlx_fit_gate::OS_APP_RESERVE_GB
    } else {
        0.0
    };
    let usable = capacity_gib - reserve;
    (usable + 0.05 < f64::from(floor_gib)).then_some((usable, reserve))
}

/// The refusal of a run described by `shape` whose resolved floor row is `floor`, or `None` to
/// admit. Floor and capacity are both GiB; the message names the row's own provenance.
fn memory_refusal_for_floor(
    lane: &str,
    floor: Option<sceneworks_core::training::iris::floors::IrisMemoryFloor>,
    shape: &str,
    capacity_gib: Option<f64>,
    no_row_advice: &str,
    advice: &str,
) -> Option<String> {
    let noun = capacity_noun(lane);
    let Some(floor) = floor else {
        return Some(format!(
            "Iris 3B training declares no memory floor for {shape} on the {lane} lane, so it is \
             refused rather than admitted without a bound.{no_row_advice}"
        ));
    };
    let needs = format!(
        "Iris 3B training needs at least {} GiB of {noun} for {shape} ({})",
        floor.gib,
        floor.provenance.describe()
    );
    let Some(capacity) = capacity_gib else {
        return Some(format!(
            "{needs}, but this machine's {noun} could not be read, so the run is refused rather \
             than admitted without a bound."
        ));
    };
    let (usable, reserve) = under_floor(lane, floor.gib, capacity)?;
    let reserve_note = if reserve > 0.0 {
        format!(
            " ({:.1} GiB once the {} GiB macOS/app reserve is held out)",
            usable.max(0.0),
            reserve.round() as i64
        )
    } else {
        String::new()
    };
    Some(format!(
        "{needs}, but this machine has {capacity:.1} GiB{reserve_note}. To run it: {advice}"
    ))
}

/// [`iris_memory_refusal`] for any Iris objective: the plan's base model selects the objective's
/// floor table (generation: network type × resolution; depth: micro-batch pixels of the training
/// size; restoration: artifact × micro-batch crop pixels). Same refusal discipline: no row or no
/// capacity reading refuses; an MLX floor is judged against unified memory minus the macOS/app
/// reserve.
pub(crate) fn iris_plan_memory_refusal(
    lane: &str,
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
    devices: u64,
    capacity_gib: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::{
        iris_objective_memory_floor, iris_objective_shape, IrisObjective,
    };
    let objective = IrisObjective::from_base_model(base_model)?;
    if objective == IrisObjective::Generation {
        // sc-25689: the whole run — one rank's floor (+ the REPA teacher) times the ranks sharing
        // the busiest of `devices`.
        return iris_run_memory_refusal(lane, config, devices, capacity_gib);
    }
    // sc-25690: the shared gradient-checkpointing toggle is honoured only on the lanes whose task
    // trainer implements it; elsewhere it is refused here, before load, never after the weights
    // are in.
    if let Some(refusal) = sceneworks_core::training::iris::iris_gradient_checkpointing_refusal(
        objective, lane, config,
    ) {
        return Some(refusal);
    }
    let advice = match task_checkpointing_hint(objective, lane, config, capacity_gib) {
        Some(hint) => format!("{hint} Otherwise: {TASK_ADVICE}"),
        None => TASK_ADVICE.to_owned(),
    };
    memory_refusal_for_floor(
        lane,
        iris_objective_memory_floor(objective, lane, config),
        &iris_objective_shape(objective, config),
        capacity_gib,
        TASK_NO_ROW,
        &advice,
    )
}

/// sc-25690: a depth / restoration run refused at its plain row is told when the SAME run under
/// gradient (activation) checkpointing fits this machine — on a lane whose task trainer implements
/// it ([`sceneworks_core::training::iris::iris_gradient_checkpointing_lanes`]), and only when the
/// lane's `full_ac` row would actually admit it. `None` when the toggle is already on, the lane
/// lacks it, or the run would still not fit (an adapter run reads the same row either way).
fn task_checkpointing_hint(
    objective: sceneworks_core::training::iris::IrisObjective,
    lane: &str,
    config: &sceneworks_core::training::TrainingConfig,
    capacity_gib: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::{
        iris_gradient_checkpointing, iris_gradient_checkpointing_lanes, iris_objective_memory_floor,
    };
    if iris_gradient_checkpointing(config)
        || !iris_gradient_checkpointing_lanes(objective).contains(&lane)
    {
        return None;
    }
    let mut checkpointed = config.clone();
    checkpointed
        .advanced
        .insert("gradientCheckpointing".to_owned(), true.into());
    let floor = iris_objective_memory_floor(objective, lane, &checkpointed)?;
    under_floor(lane, floor.gib, capacity_gib?)
        .is_none()
        .then(|| {
            format!(
                "With gradient checkpointing on (Training Studio → Gradient checkpointing) the \
                 same run needs {} GiB ({}) and fits this machine, at the cost of slower steps.",
                floor.gib,
                floor.provenance.describe()
            )
        })
}

/// The submit-time twin of [`iris_plan_memory_refusal`] for the co-located macOS API. `None` off
/// macOS (a Candle worker's card is not visible to the API; the worker gate decides there).
pub fn iris_plan_memory_error(
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        iris_plan_memory_refusal("mlx", base_model, config, 1, mlx_training_capacity_gb())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (base_model, config);
        None
    }
}

/// This Mac's unified-memory training budget (GiB), honoring the `SCENEWORKS_MLX_MEMORY_CAP_GB`
/// small-Mac emulation. `None` off macOS.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_training_capacity_gb() -> Option<f64> {
    crate::mlx_fit_gate::resolve_budget(
        crate::mlx_fit_gate::probe_total_unified_memory_gib(),
        crate::mlx_fit_gate::mlx_memory_cap_gb(),
    )
    .map(|budget| budget.total_gb)
}

/// The VRAM (GB) one rank can count on, on the NVIDIA card(s) this worker owns, honoring the
/// `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card emulation. A worker configured with several GPU ids
/// (`SCENEWORKS_GPU_ID=0,1`) is judged per card: a rank lands on any of them, so the smallest
/// card is the bound. `Err` names a card whose memory could not be read (refused, never
/// admitted without a bound).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn candle_training_capacity_gb(gpu_id: &str) -> Result<f64, String> {
    worker_card_capacity_gb(gpu_id, |id| async move {
        crate::vram_gate::apply_vram_cap(
            crate::gpu::nvidia_vram_budget_gb(&id).await,
            crate::vram_gate::cuda_vram_cap_gb(),
        )
        .map(|budget| budget.total_gb)
    })
    .await
}

/// The GPU ids a worker's `gpu_id` names (`"1"`, `"0,1"`).
pub(crate) fn worker_gpu_ids(gpu_id: &str) -> Vec<String> {
    gpu_id
        .split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty() && *id != "cpu" && *id != "auto")
        .map(str::to_owned)
        .collect()
}

/// [`candle_training_capacity_gb`] over an injectable per-card probe (`id → total GB`).
#[cfg_attr(
    not(any(test, all(not(target_os = "macos"), feature = "backend-candle"))),
    allow(dead_code)
)]
pub(crate) async fn worker_card_capacity_gb<F, Fut>(gpu_id: &str, probe: F) -> Result<f64, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Option<f64>>,
{
    let ids = worker_gpu_ids(gpu_id);
    if ids.is_empty() {
        return Err(format!(
            "Iris 3B training needs an NVIDIA GPU, but this worker's GPU id is {gpu_id:?}, so the \
             run is refused rather than admitted without a memory bound."
        ));
    }
    let mut smallest: Option<f64> = None;
    let mut unreadable = Vec::new();
    for id in &ids {
        match probe(id.clone()).await {
            Some(total) => smallest = Some(smallest.map_or(total, |gb: f64| gb.min(total))),
            None => unreadable.push(id.clone()),
        }
    }
    if !unreadable.is_empty() {
        return Err(format!(
            "Iris 3B training could not read the memory of GPU {} (this worker's GPUs: {}), so the \
             run is refused rather than admitted without a memory bound. Check that nvidia-smi \
             reports {}.",
            unreadable.join(", "),
            ids.join(", "),
            if unreadable.len() == 1 { "that card" } else { "those cards" }
        ));
    }
    smallest.ok_or_else(|| "no GPU memory reading".to_owned())
}

/// The submit-time twin of the worker's gate for the co-located macOS API (the training analogue of
/// `full_finetune_memory_error`): `Some(message)` when this Mac is under the declared floor. `None`
/// off macOS — a Candle worker's card is not visible to the API, so the worker gate decides there.
pub fn iris_training_memory_error(network_type: &str, resolution: u32) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        iris_memory_refusal("mlx", network_type, resolution, mlx_training_capacity_gb())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (network_type, resolution);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The upstream ranged walk: `n ≤ 640` uses everything; 641–1279 items train on 640; exact
    /// multiples use everything. These are the trainer's own numbers (`gen_core` `DataWalk`), not a
    /// SceneWorks re-derivation.
    #[test]
    fn the_unused_tail_matches_the_trainers_ranged_walk() {
        let full = iris_data_walk_report(640, 1).unwrap();
        assert_eq!((full.covered, full.unused_items), (640, 0));
        assert!(full.warning.is_none());

        let small = iris_data_walk_report(12, 4).unwrap();
        assert_eq!((small.covered, small.unused_items), (12, 0));
        assert!(small.warning.is_none());

        for (items, unused) in [(641, 1), (1000, 360), (1279, 639)] {
            let report = iris_data_walk_report(items, 1).unwrap();
            assert_eq!(report.covered, 640, "{items} items");
            assert_eq!(report.unused_items, unused, "{items} items");
            let warning = report.warning.expect("an unused tail warns");
            assert!(
                warning.contains(&format!("{unused} of {items}")),
                "the trainer's text names the tail: {warning}"
            );
        }
        let doubled = iris_data_walk_report(1280, 2).unwrap();
        assert_eq!((doubled.covered, doubled.unused_items), (1280, 0));

        assert!(iris_data_walk_report(0, 1).is_none());
        assert!(iris_data_walk_report(10, 0).is_none());
    }

    /// A multi-GPU worker (`SCENEWORKS_GPU_ID=0,1`) is judged per card — the smallest card bounds
    /// a rank — and a card nvidia-smi cannot read is named in the refusal.
    #[tokio::test]
    async fn a_multi_gpu_worker_is_judged_by_its_smallest_readable_card() {
        let inventory = |cards: Vec<(&'static str, Option<f64>)>| {
            move |id: String| {
                let found = cards
                    .iter()
                    .find(|(card, _)| *card == id)
                    .and_then(|(_, gb)| *gb);
                async move { found }
            }
        };
        let both = inventory(vec![("0", Some(96.0)), ("1", Some(48.0))]);
        assert_eq!(worker_card_capacity_gb("0,1", &both).await, Ok(48.0));
        assert_eq!(worker_card_capacity_gb(" 1 ", &both).await, Ok(48.0));
        assert_eq!(worker_card_capacity_gb("0", &both).await, Ok(96.0));
        let error = worker_card_capacity_gb("0,7", &both)
            .await
            .expect_err("card 7 is not in the inventory");
        assert!(error.contains("GPU 7") && error.contains("0, 7"), "{error}");
        assert!(worker_card_capacity_gb("cpu", &both).await.is_err());
        assert_eq!(
            worker_gpu_ids("0, 1,cpu"),
            vec!["0".to_owned(), "1".to_owned()]
        );
    }

    #[test]
    fn the_memory_gate_refuses_under_the_floor_and_without_evidence() {
        // MLX LoRA at 256: floor 16 GiB, judged against unified memory minus the 4 GiB OS/app
        // reserve — a 16 GiB Mac (12 usable) is refused, a 24 GiB Mac (20 usable) admitted.
        let sixteen = iris_memory_refusal("mlx", "lora", 256, Some(16.0))
            .expect("a 16 GiB Mac is under the floor once the OS/app reserve is held out");
        assert!(sixteen.contains("reserve"), "{sixteen}");
        assert!(iris_memory_refusal("mlx", "lora", 256, Some(24.0)).is_none());
        assert!(iris_memory_refusal("mlx", "lora", 256, Some(64.0)).is_none());
        let under = iris_memory_refusal("mlx", "lokr", 256, Some(8.0)).expect("8 GiB is under");
        assert!(
            under.contains("16 GiB") && under.contains("unified memory"),
            "{under}"
        );
        // Floor and capacity are stated in the unit they are compared in.
        assert!(
            under.contains("this machine has 8.0 GiB") && !under.contains(" GB"),
            "{under}"
        );
        // Candle names VRAM, and a full run's floor is far above an adapter's.
        let full = iris_memory_refusal("candle", "full", 256, Some(48.0)).expect("48 GiB < full");
        assert!(
            full.contains("VRAM") && full.contains("full fine-tune"),
            "{full}"
        );
        assert!(iris_memory_refusal("candle", "lora", 256, Some(48.0)).is_none());
        // No reading, no row: refused, never admitted unbounded.
        assert!(iris_memory_refusal("mlx", "lora", 256, None).is_some());
        assert!(iris_memory_refusal("mlx", "lora", 4096, Some(512.0)).is_some());
    }

    fn generation_config(
        network: &str,
        resolution: u32,
        iris: serde_json::Value,
    ) -> sceneworks_core::training::TrainingConfig {
        let mut config = sceneworks_core::training::iris::iris_generation_target().defaults;
        config.resolution = resolution;
        config
            .advanced
            .insert("networkType".into(), serde_json::json!(network));
        config.advanced.insert("iris".into(), iris);
        config
    }

    /// sc-25690 (the unit mismatch): floors are GiB, like the capacity they are compared to. The
    /// campaign card — an RTX PRO 6000, 97,887 MiB = 95.59 GiB, which `nvidia_vram_budget_gb`
    /// reports as `memory_total_mb / 1024` — measurably held the Candle full 512² generation run,
    /// the checkpointed 1024² run and the restoration full fine-tunes at 256² and 512², so the gate
    /// admits it for each; and it still refuses the 1024² shapes that ran out of memory on it.
    #[test]
    fn the_card_that_held_a_measured_candle_run_is_admitted_for_it() {
        use sceneworks_core::training::iris::floors::CAMPAIGN_CUDA_CARD;
        use sceneworks_core::training::iris::restore;
        let card = 97_887.0 / 1024.0;
        assert_eq!(card, CAMPAIGN_CUDA_CARD.gib());
        assert_eq!(format!("{card:.2}"), "95.59");
        let json = serde_json::json!({});

        let full_512 = generation_config("full", 512, json.clone());
        assert_eq!(
            iris_plan_memory_refusal("candle", "iris_3b", &full_512, 1, Some(card)),
            None,
            "the card that ran the full 512² fine-tune must not be refused for it"
        );
        let full_1024_ac = generation_config(
            "full",
            1024,
            serde_json::json!({"activation_checkpointing": "full"}),
        );
        assert_eq!(
            iris_plan_memory_refusal("candle", "iris_3b", &full_1024_ac, 1, Some(card)),
            None
        );
        let full_1024 = generation_config("full", 1024, json);
        let spilled = iris_plan_memory_refusal("candle", "iris_3b", &full_1024, 1, Some(card))
            .expect("1024² without checkpointing met its need only by spilling off this card");
        assert!(
            spilled.contains("a measured need") && spilled.contains("spilling to system memory"),
            "{spilled}"
        );

        // Restoration and depth: the shapes that ran on the card are admitted; 1024² without
        // checkpointing (a measured need above the card) is refused, and admitted WITH it.
        let mut restore = restore::iris_restore_target().defaults;
        for (side, admitted) in [(256, true), (512, true), (1024, false)] {
            restore.resolution = side;
            assert_eq!(
                iris_plan_memory_refusal("candle", "iris_3b_restore", &restore, 1, Some(card))
                    .is_none(),
                admitted,
                "restoration full fine-tune at {side}²"
            );
        }
        let mut depth = sceneworks_core::training::iris::depth::iris_depth_target().defaults;
        for (side, admitted) in [(512, true), (1024, false)] {
            depth.resolution = side;
            depth.advanced["iris"]["depth_size"] = serde_json::json!([side, side]);
            assert_eq!(
                iris_plan_memory_refusal("candle", "iris_3b_depth", &depth, 1, Some(card))
                    .is_none(),
                admitted,
                "depth full fine-tune at {side}²"
            );
        }
        for (base, config) in [
            ("iris_3b_restore", &mut restore),
            ("iris_3b_depth", &mut depth),
        ] {
            config
                .advanced
                .insert("gradientCheckpointing".into(), serde_json::json!(true));
            assert_eq!(
                iris_plan_memory_refusal("candle", base, config, 1, Some(card)),
                None,
                "{base} at 1024² under gradient checkpointing ran on this card"
            );
        }
    }

    /// sc-25690: gradient (activation) checkpointing for depth / restoration is a per-lane
    /// capability. On a lane whose trainer implements it, a run refused at its plain row is told
    /// the toggle admits it — exactly when the `full_ac` row fits — and a run with the toggle on
    /// reads that row. On a lane without it, the toggle is refused typed before load and no hint
    /// is offered.
    #[test]
    fn task_gradient_checkpointing_is_offered_and_gated_per_lane() {
        use sceneworks_core::training::iris::{
            depth, iris_gradient_checkpointing_lanes, iris_objective_memory_floor, restore,
            IrisObjective,
        };
        let mut depth = depth::iris_depth_target().defaults;
        depth.resolution = 1024;
        depth.advanced["iris"]["depth_size"] = serde_json::json!([1024, 1024]);
        let restore = restore::iris_restore_target().defaults; // the paper's 1024² crop, full
        for (objective, base, config) in [
            (IrisObjective::Depth, "iris_3b_depth", &depth),
            (IrisObjective::Restoration, "iris_3b_restore", &restore),
        ] {
            let mut on = config.clone();
            on.advanced
                .insert("gradientCheckpointing".into(), serde_json::json!(true));
            for lane in ["mlx", "candle"] {
                let supported = iris_gradient_checkpointing_lanes(objective).contains(&lane);
                let plain = iris_objective_memory_floor(objective, lane, config).unwrap();
                let checkpointed = iris_objective_memory_floor(objective, lane, &on).unwrap();
                // A capacity that holds the checkpointed row but not the plain one.
                let reserve = if lane == "mlx" {
                    crate::mlx_fit_gate::OS_APP_RESERVE_GB
                } else {
                    0.0
                };
                if supported {
                    assert!(checkpointed.gib < plain.gib, "{base} {lane}");
                    let between = f64::from(checkpointed.gib) + reserve;
                    let refusal = iris_plan_memory_refusal(lane, base, config, 1, Some(between))
                        .expect("under the plain row");
                    assert!(
                        refusal.contains("With gradient checkpointing on")
                            && refusal.contains(&format!("needs {} GiB", checkpointed.gib))
                            && refusal.contains("fits this machine"),
                        "{base} {lane}: {refusal}"
                    );
                    // Not when even the checkpointed row would not fit.
                    let short =
                        iris_plan_memory_refusal(lane, base, config, 1, Some(between - 1.0))
                            .expect("under both rows");
                    assert!(!short.contains("gradient checkpointing"), "{short}");
                    // With the toggle on the run is admitted at that capacity, and a refusal of it
                    // (one GiB short) offers nothing more.
                    assert_eq!(
                        iris_plan_memory_refusal(lane, base, &on, 1, Some(between)),
                        None,
                        "{base} {lane}"
                    );
                    let still = iris_plan_memory_refusal(lane, base, &on, 1, Some(between - 1.0))
                        .expect("one GiB short of the checkpointed row");
                    assert!(!still.contains("With gradient checkpointing on"), "{still}");
                } else {
                    // The lane's trainer refuses it: no row to offer, no hint, and the toggle
                    // itself is refused before load however much memory there is.
                    assert_eq!(checkpointed, plain, "{base} {lane}");
                    let refusal = iris_plan_memory_refusal(lane, base, config, 1, Some(8.0))
                        .expect("8 GiB is under");
                    assert!(!refusal.contains("gradient checkpointing"), "{refusal}");
                    let toggled = iris_plan_memory_refusal(lane, base, &on, 1, Some(4096.0))
                        .expect("the toggle is refused on this lane");
                    assert!(
                        toggled.contains("checkpointing is not implemented")
                            && toggled.contains("Turn gradient checkpointing off"),
                        "{toggled}"
                    );
                }
            }
        }
        // A restoration ADAPTER run reads the adapter row either way: nothing to offer.
        let mut lora = restore.clone();
        lora.advanced
            .insert("networkType".into(), serde_json::json!("lora"));
        let refusal = iris_plan_memory_refusal("candle", "iris_3b_restore", &lora, 1, Some(8.0))
            .expect("8 GiB is under");
        assert!(!refusal.contains("gradient checkpointing"), "{refusal}");
    }

    /// sc-25690: a refusal names the provenance of the row it read — measured, extrapolated, lower
    /// bound or declared — never one blanket phrase for every row.
    #[test]
    fn a_refusal_names_the_provenance_of_its_own_row() {
        use sceneworks_core::training::iris::{depth, restore};
        let refusal = |base: &str, config: &sceneworks_core::training::TrainingConfig| {
            iris_plan_memory_refusal("candle", base, config, 1, Some(8.0)).expect("8 GiB is under")
        };
        let no_iris = serde_json::json!({});
        let measured = refusal("iris_3b", &generation_config("full", 512, no_iris.clone()));
        assert!(measured.contains("(a measured floor)"), "{measured}");
        assert!(!measured.contains("not yet measured"), "{measured}");
        // 256² is bounded by the measured 512² row: an extrapolation, never called measured.
        let extrapolated = refusal("iris_3b", &generation_config("full", 256, no_iris.clone()));
        assert!(
            extrapolated.contains("extrapolated from measured runs"),
            "{extrapolated}"
        );
        // 1024² without checkpointing was measured only by spilling off the card.
        let spilled = refusal("iris_3b", &generation_config("full", 1024, no_iris.clone()));
        assert!(
            spilled
                .contains("(a measured need: on a 96 GB card the run completed only by spilling"),
            "{spilled}"
        );
        let declared = refusal("iris_3b", &generation_config("lora", 256, no_iris.clone()));
        assert!(
            declared.contains("(a declared floor, not yet measured)"),
            "{declared}"
        );
        // A measured row plus the REPA teacher, or times two ranks, is arithmetic on a measurement.
        let with_teacher = refusal(
            "iris_3b",
            &generation_config("full", 512, serde_json::json!({"repa": {"weight": 0.5}})),
        );
        assert!(
            with_teacher.contains("extrapolated from measured runs"),
            "{with_teacher}"
        );
        // Depth (Candle, its default 480×640): extrapolated. Restoration full at 1024²: the runs
        // were stopped before their peak, so only a lower bound is measured.
        let depth = refusal("iris_3b_depth", &depth::iris_depth_target().defaults);
        assert!(depth.contains("extrapolated from measured runs"), "{depth}");
        let mut restore = restore::iris_restore_target().defaults;
        restore.resolution = 1024;
        assert!(refusal("iris_3b_restore", &restore).contains("(a measured need"));
        let bound = iris_plan_memory_refusal("mlx", "iris_3b_restore", &restore, 1, Some(8.0))
            .expect("8 GiB is under");
        assert!(bound.contains("above a measured lower bound"), "{bound}");
        restore.resolution = 512;
        let restore_measured = refusal("iris_3b_restore", &restore);
        assert!(
            restore_measured.contains("(a measured floor)"),
            "{restore_measured}"
        );
        // An unreadable host is refused with the same provenance, and without a capacity.
        let unread = iris_plan_memory_refusal("candle", "iris_3b_restore", &restore, 1, None)
            .expect("no reading refuses");
        assert!(
            unread.contains("(a measured floor)") && unread.contains("could not be read"),
            "{unread}"
        );
    }

    /// sc-25690: a full 1024² run refused at its plain row is told that activation checkpointing
    /// admits it — exactly when the `full_ac` row fits this machine — and not otherwise.
    #[test]
    fn a_refused_full_run_is_told_when_activation_checkpointing_would_admit_it() {
        use sceneworks_core::training::iris::iris_generation_memory_floor;
        let plain = iris_generation_memory_floor("candle", "full", 1024, false).unwrap();
        let checkpointed = iris_generation_memory_floor("candle", "full", 1024, true).unwrap();
        assert!(checkpointed.gib < plain.gib);
        let config = generation_config("full", 1024, serde_json::json!({}));
        // Between the two rows: refused, and the checkpointing option is named with its floor.
        let between = f64::from(checkpointed.gib);
        let refusal = iris_plan_memory_refusal("candle", "iris_3b", &config, 1, Some(between))
            .expect("under the plain row");
        assert!(
            refusal.contains("activation checkpointing")
                && refusal.contains("selective_op")
                && refusal.contains(&format!("needs {} GiB", checkpointed.gib))
                && refusal.contains("fits this machine"),
            "{refusal}"
        );
        // Under the checkpointed row too: no hint (it would not admit).
        let too_small = iris_plan_memory_refusal(
            "candle",
            "iris_3b",
            &config,
            1,
            Some(f64::from(checkpointed.gib) - 1.0),
        )
        .expect("under both rows");
        assert!(
            !too_small.contains("activation checkpointing"),
            "{too_small}"
        );
        // Already checkpointing every region, and an adapter run: nothing to suggest.
        for policy in ["full", "selective_op"] {
            let already = generation_config(
                "full",
                1024,
                serde_json::json!({"activation_checkpointing": policy}),
            );
            let refused = iris_plan_memory_refusal("candle", "iris_3b", &already, 1, Some(8.0))
                .expect("8 GiB is under");
            assert!(!refused.contains("fits this machine"), "{refused}");
        }
        let adapter = generation_config("lora", 1024, serde_json::json!({}));
        let refused = iris_plan_memory_refusal("candle", "iris_3b", &adapter, 1, Some(8.0))
            .expect("8 GiB is under");
        assert!(!refused.contains("activation checkpointing"), "{refused}");
        // `selective_layer` reads the plain row, so it gets the hint as `none` does.
        let selective = generation_config(
            "full",
            1024,
            serde_json::json!({"activation_checkpointing": "selective_layer"}),
        );
        assert!(
            iris_plan_memory_refusal("candle", "iris_3b", &selective, 1, Some(between))
                .expect("under the plain row")
                .contains("fits this machine")
        );
        // MLX: the hint is judged after the OS/app reserve, like the floor itself.
        let mlx_ac = iris_generation_memory_floor("mlx", "full", 1024, true).unwrap();
        let mlx_plain = iris_generation_memory_floor("mlx", "full", 1024, false).unwrap();
        assert!(mlx_ac.gib < mlx_plain.gib);
        let reserve = crate::mlx_fit_gate::OS_APP_RESERVE_GB;
        let fits = iris_plan_memory_refusal(
            "mlx",
            "iris_3b",
            &config,
            1,
            Some(f64::from(mlx_ac.gib) + reserve),
        )
        .expect("under the plain MLX row");
        assert!(fits.contains("fits this machine"), "{fits}");
        let short = iris_plan_memory_refusal(
            "mlx",
            "iris_3b",
            &config,
            1,
            Some(f64::from(mlx_ac.gib) + reserve - 1.0),
        )
        .expect("under both MLX rows once the reserve is held out");
        assert!(!short.contains("fits this machine"), "{short}");
    }

    /// The depth and restoration floors (sc-25687 / sc-25688): per objective and lane, with the MLX
    /// OS reserve, refusing beyond the table and without a capacity reading. Each boundary is the
    /// row's own value (GiB), so a re-measured row moves the boundary with it.
    #[test]
    fn task_objective_floors_gate_per_lane() {
        use sceneworks_core::training::iris::{depth, restore};
        let reserve = crate::mlx_fit_gate::OS_APP_RESERVE_GB;
        let depth = depth::iris_depth_target().defaults;
        // 480×640, batch 1.
        let mlx = f64::from(depth::depth_memory_floor("mlx", &depth).unwrap().gib);
        let candle = f64::from(depth::depth_memory_floor("candle", &depth).unwrap().gib);
        assert!(
            iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, 1, Some(mlx + reserve))
                .is_none()
        );
        let under =
            iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, 1, Some(mlx + reserve - 1.0))
                .expect("one GiB under the floor once the reserve is held out");
        assert!(
            under.contains(&format!("{mlx} GiB")) && under.contains("reserve"),
            "{under}"
        );
        assert!(
            iris_plan_memory_refusal("candle", "iris_3b_depth", &depth, 1, Some(candle)).is_none()
        );
        assert!(
            iris_plan_memory_refusal("candle", "iris_3b_depth", &depth, 1, Some(candle - 1.0))
                .is_some()
        );
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, 1, None).is_some());
        let mut big = depth.clone();
        big.batch_size = 32;
        big.gradient_accumulation = 1;
        let refused = iris_plan_memory_refusal("mlx", "iris_3b_depth", &big, 1, Some(512.0))
            .expect("no row for 32 × 480×640");
        assert!(refused.contains("declares no memory floor"), "{refused}");

        let mut restore = restore::iris_restore_target().defaults;
        restore.resolution = 256;
        // The full recipe at 256².
        let mlx = f64::from(restore::restore_memory_floor("mlx", &restore).unwrap().gib);
        assert!(iris_plan_memory_refusal(
            "mlx",
            "iris_3b_restore",
            &restore,
            1,
            Some(mlx + reserve)
        )
        .is_none());
        assert!(iris_plan_memory_refusal(
            "mlx",
            "iris_3b_restore",
            &restore,
            1,
            Some(mlx + reserve - 1.0)
        )
        .is_some());
        restore
            .advanced
            .insert("networkType".into(), serde_json::json!("lora"));
        let mlx_lora = f64::from(restore::restore_memory_floor("mlx", &restore).unwrap().gib);
        let candle_lora = restore::restore_memory_floor("candle", &restore)
            .unwrap()
            .gib;
        assert!(iris_plan_memory_refusal(
            "mlx",
            "iris_3b_restore",
            &restore,
            1,
            Some(mlx_lora + reserve)
        )
        .is_none());
        let lora = iris_plan_memory_refusal(
            "candle",
            "iris_3b_restore",
            &restore,
            1,
            Some(f64::from(candle_lora) - 1.0),
        )
        .expect("one GiB under the adapter floor");
        assert!(
            lora.contains("VRAM") && lora.contains(&format!("{candle_lora} GiB")),
            "{lora}"
        );
        // Non-Iris base models are not this gate's.
        assert!(iris_plan_memory_refusal("mlx", "sdxl", &restore, 1, Some(1.0)).is_none());
    }
}

/// The REAL training job path — `training_jobs::run_training_execution_using`: plan preflight, the
/// Iris memory floor + data walk, base/encoder resolution, the blocking drive, the progress/cancel
/// consumer and the completion/cancel result — with only the trainer injected. A plan built by the
/// real `sceneworks_core` target registry is what the job executes, so these pin that every
/// offered option reaches `Trainer::train`, and what the job API sees, per backend.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod job_tests {
    use crate::api_client::ApiClient;
    use crate::error::{WorkerError, WorkerResult};
    use axum::extract::{Path as AxumPath, State};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use gen_core::{
        LoadSpec, Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
        WeightsSource,
    };
    use sceneworks_core::contracts::JobSnapshot;
    use sceneworks_core::training::{
        build_training_plan, builtin_training_targets, BuildTrainingPlan, TrainingDataset,
        TrainingPlan,
    };
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const BACKBONE_REPO: &str = "speridlabs/iris-3b";
    const BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
    const TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
    const TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Behavior {
        /// Run to completion: progress, a checkpoint, the export.
        Completes,
        /// Wait for the request's cancel flag, then stop the way the trainer does BEFORE its first
        /// optimizer step: the typed cancellation, nothing written.
        CancelBeforeFirstStep,
        /// Report a step, wait for the cancel flag, then publish the checkpoint + export and return
        /// `Ok` — the trainer's contract for a cancel after the first optimizer step.
        CancelAfterFirstStep,
        /// Train a step, publish a checkpoint, then fail at the export the way the engine does (its
        /// publication is atomic — `gen_core_iris::train::lineage` — so no export is left at the
        /// final path): the typed error, the checkpoint kept, nothing exported.
        FailsExport,
        /// Run the ENGINE's own depth dataset pass (`depth_dataset_pass`, the first thing both depth
        /// trainers do) on the request the worker shaped, then complete if it accepts the dataset.
        DepthDatasetPass,
    }

    /// The release depth architecture the engine's depth plan resolves against.
    const DEPTH_CONFIG: &str = include_str!("iris_restore/fixtures/depth_config.yaml");

    #[derive(Default)]
    struct Seen {
        request: Option<TrainingRequest>,
        weights: Option<PathBuf>,
        text_encoder: Option<PathBuf>,
        teacher: Option<PathBuf>,
        /// Every staged component as `(id, path)` (depth / restoration task runs).
        components: Vec<(String, PathBuf)>,
        loads: u32,
    }

    struct FakeIrisTrainer {
        descriptor: TrainerDescriptor,
        behavior: Behavior,
        seen: Arc<Mutex<Seen>>,
        /// The job API's `cancelRequested`: the fake "user" clicks Cancel only once the trainer is
        /// mid-run, so the cancel reaches the training loop rather than the pre-start check.
        user_cancel: Arc<std::sync::atomic::AtomicBool>,
    }

    fn write_export(req: &TrainingRequest) -> PathBuf {
        std::fs::create_dir_all(&req.output_dir).unwrap();
        let checkpoint = req.output_dir.join(format!(
            "{}.checkpoints/step_00000001",
            Path::new(&req.file_name)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
        ));
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("state.json"), b"{}").unwrap();
        let adapter = req.output_dir.join(&req.file_name);
        std::fs::write(&adapter, b"adapter").unwrap();
        adapter
    }

    impl Trainer for FakeIrisTrainer {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &TrainingRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn train(
            &mut self,
            req: &TrainingRequest,
            on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> gen_core::Result<TrainingOutput> {
            self.seen.lock().unwrap().request = Some(req.clone());
            on_progress(TrainingProgress::Preparing);
            if self.behavior == Behavior::DepthDatasetPass {
                use gen_core_iris::train::depth as depth_train;
                let model = gen_core_iris::IrisConfig::parse(DEPTH_CONFIG)?;
                let plan = depth_train::DepthTrainPlan::resolve(req, &model)?;
                depth_train::depth_dataset_pass(req, &plan, |_| {})?;
            }
            on_progress(TrainingProgress::LoadingModel);
            on_progress(TrainingProgress::Caching {
                current: 1,
                total: 1,
            });
            let user_cancel = Arc::clone(&self.user_cancel);
            let click_cancel = || user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            let wait_for_cancel = || {
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                req.cancel.is_cancelled()
            };
            match self.behavior {
                Behavior::FailsExport => {
                    on_progress(TrainingProgress::Training {
                        step: 1,
                        total: req.config.steps,
                        loss: 0.5,
                    });
                    on_progress(TrainingProgress::Checkpoint { step: 1 });
                    on_progress(TrainingProgress::Saving);
                    // The checkpoint is durable; the export never reaches its final path.
                    let exported = write_export(req);
                    std::fs::remove_file(exported).unwrap();
                    Err(gen_core::Error::Msg(
                        "iris: export failed: write model.safetensors: No space left on device \
                         (os error 28)"
                            .to_owned(),
                    ))
                }
                Behavior::Completes | Behavior::DepthDatasetPass => {
                    let accumulation = req.config.gradient_accumulation.max(1);
                    on_progress(TrainingProgress::Training {
                        step: accumulation,
                        total: req.config.steps,
                        loss: 0.25,
                    });
                    on_progress(TrainingProgress::Checkpoint { step: accumulation });
                    on_progress(TrainingProgress::Saving);
                    let path = write_export(req);
                    Ok(TrainingOutput {
                        adapter_path: path,
                        steps: req.config.steps,
                        final_loss: 0.25,
                    })
                }
                Behavior::CancelBeforeFirstStep => {
                    // Outlive the consumer's 2 s cancel-poll throttle with a Training event so the
                    // poll fires, but never complete an optimizer step.
                    std::thread::sleep(Duration::from_millis(2_300));
                    click_cancel();
                    on_progress(TrainingProgress::Training {
                        step: 0,
                        total: req.config.steps,
                        loss: f32::NAN,
                    });
                    assert!(wait_for_cancel(), "the worker tripped the cancel flag");
                    Err(gen_core::Error::Canceled)
                }
                Behavior::CancelAfterFirstStep => {
                    on_progress(TrainingProgress::Training {
                        step: 1,
                        total: req.config.steps,
                        loss: 0.5,
                    });
                    std::thread::sleep(Duration::from_millis(2_300));
                    click_cancel();
                    on_progress(TrainingProgress::Training {
                        step: 2,
                        total: req.config.steps,
                        loss: 0.4,
                    });
                    assert!(wait_for_cancel(), "the worker tripped the cancel flag");
                    let path = write_export(req);
                    Ok(TrainingOutput {
                        adapter_path: path,
                        steps: 2,
                        final_loss: 0.4,
                    })
                }
            }
        }
    }

    type Posts = Arc<Mutex<Vec<Value>>>;
    type CancelSwitch = Arc<std::sync::atomic::AtomicBool>;

    fn requested(cancel: &CancelSwitch) -> bool {
        cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn spawn_job_api(cancel_requested: CancelSwitch) -> (String, Posts) {
        async fn job_route(
            State((_, cancel)): State<(Posts, CancelSwitch)>,
            AxumPath(job_id): AxumPath<String>,
        ) -> Response {
            Json(crate::tests::job_snapshot_json(&job_id, requested(&cancel))).into_response()
        }
        async fn progress_route(
            State((posts, cancel)): State<(Posts, CancelSwitch)>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            posts.lock().unwrap().push(body);
            Json(crate::tests::job_snapshot_json(&job_id, requested(&cancel))).into_response()
        }
        async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
            Json(json!({
                "id": worker_id, "gpuId": "cpu", "gpuName": null, "status": "busy",
                "currentJobId": "job-iris-train", "capabilities": [], "loadedModels": [],
                "registeredAt": "2026-07-01T00:00:00Z", "lastSeenAt": "2026-07-01T00:00:00Z"
            }))
            .into_response()
        }
        let posts: Posts = Arc::default();
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_route))
            .route("/api/v1/jobs/:job_id/progress", post(progress_route))
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(heartbeat_route),
            )
            .with_state((posts.clone(), cancel_requested));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        crate::load_progress::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), posts)
    }

    fn stage(root: &Path, repo: &str, revision: &str, file: &str) -> PathBuf {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .unwrap()
            .join("snapshots")
            .join(revision);
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"x").unwrap();
        let refs = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .unwrap()
            .join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), revision).unwrap();
        snapshot
    }

    struct Install {
        backbone: PathBuf,
        text_encoder: PathBuf,
    }

    fn stage_install(data: &Path) -> Install {
        let mut backbone = PathBuf::new();
        for file in ["config.yaml", "model.safetensors"] {
            backbone = stage(data, BACKBONE_REPO, BACKBONE_REVISION, file);
        }
        // The worker resolves the engine plan against the backbone's real architecture (sc-25689).
        std::fs::write(
            backbone.join("config.yaml"),
            crate::iris_test_weights::RELEASE_CONFIG_YAML,
        )
        .unwrap();
        let mut text_encoder = PathBuf::new();
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            text_encoder = stage(data, TE_REPO, TE_REVISION, file);
        }
        Install {
            backbone,
            text_encoder,
        }
    }

    /// A dataset of `items` images under the data dir, the first carrying named captions.
    fn dataset(data: &Path, items: usize) -> (TrainingDataset, PathBuf) {
        let root = data.join("projects/p1/training/datasets/ds1");
        std::fs::create_dir_all(root.join("images")).unwrap();
        let entries: Vec<Value> = (0..items)
            .map(|index| {
                let path = format!("images/item{index}.png");
                std::fs::write(root.join(&path), b"png").unwrap();
                let mut item = json!({
                    "id": format!("item{index}"),
                    "path": path,
                    "displayName": format!("item{index}"),
                    "caption": {"text": format!("a photo {index}"), "source": "manual", "triggerWords": []},
                    "addedAt": "2026-10-11T00:00:00Z",
                    "captions": {"short": format!("short {index}"), "tags": "red, square"}
                });
                if index > 0 {
                    item.as_object_mut().unwrap().remove("captions");
                    item["captions"] = json!({"short": format!("short {index}")});
                }
                item
            })
            .collect();
        let dataset: TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds1", "version": 3, "projectId": "p1", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": entries
        }))
        .unwrap();
        (dataset, root)
    }

    /// Build the plan the API would, from the REAL target registry, with `advanced` merged over the
    /// target defaults.
    fn plan(
        data: &Path,
        backbone: &Path,
        items: usize,
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        plan_with_dataset(data, backbone, items, |_| {}, configure)
    }

    /// [`plan`] with the dataset adjusted before the plan is built (e.g. holdout items).
    fn plan_with_dataset(
        data: &Path,
        backbone: &Path,
        items: usize,
        adjust: impl FnOnce(&mut TrainingDataset),
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        let target = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.kernel == "iris_train")
            .expect("the Iris target is registered");
        let mut config = target.defaults.clone();
        configure(&mut config);
        let (mut dataset, root) = dataset(data, items);
        adjust(&mut dataset);
        let output = data.join("loras/lora_iris");
        build_training_plan(BuildTrainingPlan {
            job_id: "job-iris-train",
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "lora_iris",
            base_model_path: backbone.display().to_string(),
            dataset_root: &root,
            output_dir: &output,
            file_name: "iris_style.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        })
        .expect("the plan builds")
    }

    struct Run {
        outcome: WorkerResult<()>,
        posts: Vec<Value>,
        seen: Seen,
    }

    /// Isolate the HF cache env and pin the emulated memory cap for the WHOLE test — staging the
    /// install resolves the cache path too, so the guard must be held before anything is staged.
    fn isolated_env(cap_gb: &str) -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            #[cfg(target_os = "macos")]
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, cap_gb),
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            (crate::vram_gate::CUDA_VRAM_CAP_ENV, cap_gb),
        ])
    }

    fn run(behavior: Behavior, plan: TrainingPlan, data: &Path) -> Run {
        run_as("iris_3b", behavior, plan, data)
    }

    fn run_as(
        expected_engine: &'static str,
        behavior: Behavior,
        plan: TrainingPlan,
        data: &Path,
    ) -> Run {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            // Off-Mac the Iris memory gate reads the worker's NVIDIA card; `isolated_env` caps it
            // (`SCENEWORKS_CUDA_VRAM_CAP_GB`), so the bound is the test's, not the runner's.
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                settings.gpu_id = "0".to_owned();
            }
            settings.api_url = base_url;
            settings.data_dir = data.to_path_buf();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let descriptor = crate::inference_runtime::trainer_descriptor(expected_engine)
                .expect("this backend links the Iris trainer");
            let loader_seen = Arc::clone(&seen);
            let user_cancel = Arc::clone(&user_cancel);
            let loader = move |engine: &str, spec: &LoadSpec| {
                assert_eq!(engine, expected_engine);
                let mut seen = loader_seen.lock().unwrap();
                seen.loads += 1;
                for (id, source) in &spec.components {
                    let path = match source {
                        WeightsSource::Dir(path) | WeightsSource::File(path) => path.clone(),
                    };
                    seen.components.push((id.clone(), path));
                }
                if let WeightsSource::Dir(dir) = &spec.weights {
                    seen.weights = Some(dir.clone());
                }
                if let Some(WeightsSource::Dir(dir)) = spec.components.get("text_encoder") {
                    seen.text_encoder = Some(dir.clone());
                }
                if let Some(WeightsSource::Dir(dir)) = spec.components.get("repa_teacher") {
                    seen.teacher = Some(dir.clone());
                }
                Ok(Box::new(FakeIrisTrainer {
                    descriptor,
                    behavior,
                    seen: Arc::clone(&loader_seen),
                    user_cancel,
                }) as Box<dyn Trainer>)
            };
            let outcome = tokio::time::timeout(
                Duration::from_secs(90),
                crate::training_jobs::run_training_execution_using(
                    &api, &settings, &job, &plan, loader,
                ),
            )
            .await
            .expect("the training job finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        let seen = std::mem::take(&mut *seen.lock().unwrap());
        Run {
            outcome,
            posts,
            seen,
        }
    }

    fn terminal(posts: &[Value]) -> &Value {
        posts.last().expect("the job posted progress")
    }

    /// Every option the Studio offers reaches the trainer's request — the shared fields mapped onto
    /// `TrainingConfig`, the Iris bag verbatim as `model_options["iris"]` — and the trainer is loaded
    /// from the installed backbone with the generation install's text encoder staged.
    #[test]
    fn every_offered_option_reaches_the_trainer_and_the_export_completes() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let iris = json!({
            "init": "weights",
            "prediction": "v",
            "flow_shift": 3.0,
            "shift_law": "sd3",
            "shift_base_tokens": 256,
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "x_pred_sigma_min": 0.05,
            "text_dropout": 0.15,
            "gradient_clip": 1.0,
            "betas": [0.9, 0.99],
            "muon_momentum": 0.9,
            "muon_nesterov": false,
            "muon_adjust_lr": "spectral_norm",
            "auto_lr": "sqrt",
            "base_batch_size": 64,
            "ema_enabled": true,
            "ema_decay": 0.999,
            "num_epochs": 3,
            "nan_loss_tolerance": 5,
            "caption_field": "caption",
            "caption_fields": ["short", "tags"],
            "on_caption_overflow": "error",
            "text_conditioning": "on_the_fly",
            "keep_last_checkpoints": 1,
            "milestone_steps": [10, 20],
            "preview_weights": "ema",
            "preview_negative_prompt": "blurry",
            "export_weights": "ema",
            "export_dtype": "bf16",
            "upstream_validation_prompts": false
        });
        let plan = plan(&data, &install.backbone, 3, |config| {
            config.optimizer = "muon".to_owned();
            config.gradient_accumulation = 2;
            config.steps = 40;
            config.save_every = 10;
            config.resolution = 512;
            config.batch_size = 2;
            config.rank = 8;
            config.alpha = sceneworks_core::contracts::ContractNumber::from(4);
            let advanced = &mut config.advanced;
            advanced.insert("networkType".into(), json!("lokr"));
            advanced.insert("decomposeFactor".into(), json!(4));
            advanced.insert("lrScheduler".into(), json!("cosine"));
            advanced.insert("lrWarmupSteps".into(), json!(6));
            advanced.insert("timestepType".into(), json!("uniform"));
            advanced.insert("mixedPrecision".into(), json!("f32"));
            advanced.insert("weightDecay".into(), json!(0.01));
            advanced.insert("sampleEvery".into(), json!(20));
            advanced.insert("sampleSteps".into(), json!(12));
            advanced.insert("sampleGuidanceScale".into(), json!(2.5));
            advanced.insert("sampleCount".into(), json!(2));
            advanced.insert(
                "samplePrompts".into(),
                json!(["a red fox", "a blue bird", "a third prompt"]),
            );
            advanced.insert("iris".into(), iris.clone());
        });
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        // The trainer checkpointed, but an Iris checkpoint is not a generic resume bundle: no post
        // offers the Queue's retry-resume (sc-25785) — the run resumes from the Training Studio.
        assert!(
            run.posts
                .iter()
                .all(|post| post["result"].get("resumeCheckpointStep").is_none()),
            "{:#?}",
            run.posts
        );

        assert_eq!(run.seen.loads, 1);
        assert_eq!(
            run.seen.weights.as_deref(),
            Some(install.backbone.as_path())
        );
        assert_eq!(
            run.seen.text_encoder.as_deref(),
            Some(install.text_encoder.as_path()),
            "the generation install's Qwen3-VL encoder is the trainer's text_encoder component"
        );
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert_eq!(config.optimizer, "muon");
        assert_eq!(config.lr_scheduler, gen_core::LrSchedule::Cosine);
        assert_eq!(config.lr_warmup_steps, 6);
        assert_eq!(config.timestep_type, "uniform");
        assert_eq!(config.timestep_bias, "balanced");
        assert_eq!(config.loss_type, "mse");
        assert_eq!(config.train_dtype, "f32");
        assert_eq!(config.gradient_accumulation, 2);
        assert_eq!((config.steps, config.save_every), (40, 10));
        assert_eq!((config.resolution, config.batch_size), (512, 2));
        assert_eq!((config.rank, config.alpha), (8, 4.0));
        assert_eq!(config.network_type, gen_core::NetworkType::Lokr);
        assert_eq!(config.decompose_factor, 4);
        assert!(!config.full_finetune);
        assert!(!config.gradient_checkpointing);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
        assert_eq!(
            (config.sample_every, config.sample_steps),
            (20, 12),
            "preview cadence + steps"
        );
        assert!((config.sample_guidance_scale - 2.5).abs() < 1e-6);
        assert_eq!(
            config.sample_prompts,
            vec!["a red fox".to_owned(), "a blue bird".to_owned()],
            "the pool is capped at sampleCount"
        );
        assert!(!config.resume);
        assert_eq!(
            config.model_options.get("iris"),
            Some(&iris),
            "the Iris options bag reaches the trainer verbatim"
        );
        // Per-item named captions ride on the item.
        assert_eq!(request.items.len(), 3);
        assert_eq!(
            request.items[0].model_options.get("captions"),
            Some(&json!({"short": "short 0", "tags": "red, square"}))
        );
        assert_eq!(request.file_name, "iris_style.safetensors");
        assert!(request.output_dir.ends_with("loras/lora_iris"));

        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        let result = &done["result"];
        assert_eq!(result["kernel"], "iris_train");
        assert_eq!(result["networkType"], "lokr");
        assert_eq!(result["dataWalk"]["unusedItems"], 0);
        assert!(result.get("warnings").is_none(), "no tail ⇒ no warning");
        assert!(result.get("trainedArtifactExported").is_none());
    }

    /// The trainer's unused tail (641 items ⇒ 640 trained) reaches the job: a progress message, the
    /// result's `warnings` and `dataWalk`, before the trainer runs.
    #[test]
    fn the_unused_tail_is_reported_in_the_job() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 641, |_| {});
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let warned = run
            .posts
            .iter()
            .position(|post| {
                post["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("1 of 641"))
            })
            .expect("the unused tail is posted as a job message");
        assert_eq!(run.posts[warned]["status"], "preparing");
        let result = &terminal(&run.posts)["result"];
        assert_eq!(result["dataWalk"]["covered"], 640);
        assert_eq!(result["dataWalk"]["unusedItems"], 1);
        assert!(result["warnings"][0]
            .as_str()
            .unwrap()
            .contains("never trained on"));
    }

    /// S11: a warm generator (an `iris_3b` render keeps its backbone + text encoder resident for the
    /// idle window) or a resident restoration transform is evicted BEFORE the trainer loads its own
    /// backbone — never a second backbone on top of the first.
    #[test]
    fn an_iris_training_run_evicts_a_resident_generator_or_transform_before_loading() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let resident_weights = tempfile::tempdir().unwrap();
        let caches = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        for generator in [true, false] {
            caches.block_on(async {
                if generator {
                    crate::generator_cache::load_stub_into_global_cache(resident_weights.path())
                        .await
                        .unwrap();
                    assert!(crate::generator_cache::global_generator_resident()
                        .await
                        .is_some());
                } else {
                    crate::transform_cache::load_stub_into_global_cache(resident_weights.path())
                        .await
                        .unwrap();
                    assert!(crate::transform_cache::global_transform_resident().await);
                }
            });
            let plan = plan(&data, &install.backbone, 2, |_| {});
            let run = run(Behavior::Completes, plan, &data);
            assert!(run.outcome.is_ok(), "{:?}", run.outcome);
            assert_eq!(run.seen.loads, 1);
            let resident_after = caches.block_on(async {
                (
                    crate::generator_cache::global_generator_resident()
                        .await
                        .is_some(),
                    crate::transform_cache::global_transform_resident().await,
                )
            });
            assert_eq!(
                resident_after,
                (false, false),
                "the resident {} must be evicted by the training run",
                if generator { "generator" } else { "transform" }
            );
        }
    }

    /// S7: generation training needs the pinned ROOT backbone and the Qwen3-VL text encoder —
    /// a depth/ export alone, or a backbone without its encoder, is refused typed before load.
    #[test]
    fn a_generation_run_without_its_backbone_or_text_encoder_is_refused_before_load() {
        let _env = isolated_env("64");
        let objective = sceneworks_core::training::iris::IrisObjective::Generation;
        let config = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.kernel == "iris_train" && target.base_model == "iris_3b")
            .unwrap()
            .defaults;

        // Only the depth export installed: the repo exists, the generation backbone does not.
        let only_depth = tempfile::tempdir().unwrap();
        for file in [
            "config.yaml",
            "model.safetensors",
            "empty_prompt.safetensors",
        ] {
            stage(
                only_depth.path(),
                BACKBONE_REPO,
                BACKBONE_REVISION,
                &format!("depth/{file}"),
            );
        }
        let refusal = crate::iris_task_training::iris_training_install_refusal(
            only_depth.path(),
            objective,
            &config,
        )
        .expect("a depth export is not a generation install");
        assert_eq!(refusal.kind, "missing_backbone", "{refusal:?}");

        // The backbone without its text encoder.
        let no_encoder = tempfile::tempdir().unwrap();
        for file in ["config.yaml", "model.safetensors"] {
            stage(no_encoder.path(), BACKBONE_REPO, BACKBONE_REVISION, file);
        }
        let refusal = crate::iris_task_training::iris_training_install_refusal(
            no_encoder.path(),
            objective,
            &config,
        )
        .expect("the text encoder is required");
        assert_eq!(refusal.kind, "missing_text_encoder", "{refusal:?}");

        // A full install is ready; the worker run refuses the encoder-less one before the loader.
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        assert_eq!(
            crate::iris_task_training::iris_training_install_refusal(&data, objective, &config),
            None
        );
        let encoder_repo = install.text_encoder.parent().unwrap().parent().unwrap();
        std::fs::remove_dir_all(encoder_repo).unwrap();
        let plan = plan(&data, &install.backbone, 2, |_| {});
        let run = run(Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("text encoder"), "{message}")
            }
            other => panic!("an encoder-less install must be refused, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0, "refused before the loader ran");
    }

    /// A host under the declared floor is refused BEFORE the loader runs; nothing is written.
    #[test]
    fn a_host_under_the_declared_floor_is_refused_before_load() {
        let _env = isolated_env("48");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |config| {
            config.advanced.insert("networkType".into(), json!("full"));
        });
        let run = run(Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("full fine-tune"), "{message}")
            }
            other => panic!("a 48 GB host must be refused for a full run, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0, "refused before the loader ran");
        assert!(!data.join("loras/lora_iris/iris_style.safetensors").exists());
    }

    /// A cancel honoured before the first optimizer step ends Canceled with no result and nothing
    /// published.
    #[test]
    fn a_cancel_before_the_first_step_publishes_nothing() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |_| {});
        let run = run(Behavior::CancelBeforeFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert!(last["result"].is_null(), "{last:#}");
        assert!(!data.join("loras/lora_iris/iris_style.safetensors").exists());
    }

    /// A cancel honoured after the first optimizer step: the trainer published the checkpoint and
    /// the export, so the terminal Canceled carries the result the API registers from.
    #[test]
    fn a_cancel_after_the_first_step_reports_the_published_export() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |_| {});
        let run = run(Behavior::CancelAfterFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        let result = &last["result"];
        assert_eq!(result["trainedArtifactExported"], true, "{last:#}");
        assert_eq!(result["stepsCompleted"], 2);
        assert!(result["outputPath"]
            .as_str()
            .unwrap()
            .ends_with("iris_style.safetensors"));
        assert!(last["message"].as_str().unwrap().contains("step 2"));
        assert!(data
            .join("loras/lora_iris/iris_style.safetensors")
            .is_file());
    }

    /// Only Iris finishes a cancel durably. Another trainer that returns `Ok` after a cancel (SDXL
    /// saves a partial adapter once a step has run) still ends a plain, result-less Canceled, so the
    /// API registers nothing for it — the pre-Iris behaviour (E8). Drives the REAL event consumer.
    #[test]
    fn a_non_iris_trainer_returning_ok_after_cancel_publishes_no_result() {
        use crate::training_jobs::{consume_training_events, TrainEvent};
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let mut plan = plan(&data, &install.backbone, 2, |_| {});
        plan.target.kernel = "sdxl_lora".to_owned();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            // Off-Mac the Iris memory gate reads the worker's NVIDIA card; `isolated_env` caps it
            // (`SCENEWORKS_CUDA_VRAM_CAP_GB`), so the bound is the test's, not the runner's.
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                settings.gpu_id = "0".to_owned();
            }
            settings.api_url = base_url;
            settings.data_dir = data.clone();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let (tx, rx) = tokio::sync::mpsc::channel(16);
            let cancel = gen_core::CancelFlag::new();
            let flag = cancel.clone();
            let adapter = data.join("loras/lora_iris/partial.safetensors");
            let blocking = crate::load_progress::spawn_blocking(move || {
                let training = |step| {
                    TrainEvent::Progress(TrainingProgress::Training {
                        step,
                        total: 40,
                        loss: 0.5,
                    })
                };
                let _ = tx.blocking_send(training(1));
                // Outlive the consumer's 2 s cancel-poll throttle, then "click Cancel".
                std::thread::sleep(Duration::from_millis(2_300));
                user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.blocking_send(training(2));
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !flag.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                assert!(flag.is_cancelled(), "the worker tripped the cancel flag");
                let _ = tx.blocking_send(TrainEvent::Done(TrainingOutput {
                    adapter_path: adapter,
                    steps: 2,
                    final_loss: 0.4,
                }));
                Ok(())
            });
            let outcome = tokio::time::timeout(
                Duration::from_secs(90),
                consume_training_events(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    gen_core::TrainingConfig::default(),
                    "mlx",
                    40,
                    None,
                    rx,
                    cancel,
                    blocking,
                    serde_json::Map::new(),
                ),
            )
            .await
            .expect("the consumer finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        assert!(
            matches!(outcome, Err(WorkerError::Canceled(_))),
            "{outcome:?}"
        );
        let last = terminal(&posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert!(last["result"].is_null(), "{last:#}");
    }

    // ------------------------------------------------------------------------------------------
    // Iris DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) task runs through the same executor.
    // ------------------------------------------------------------------------------------------

    /// Serializes the tests that substitute the pinned training-resource table.
    static RESOURCE_TABLE: Mutex<()> = Mutex::new(());

    fn stage_bytes(root: &Path, repo: &str, revision: &str, file: &str, bytes: &[u8]) -> PathBuf {
        let path = stage(root, repo, revision, file).join(file);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// The task export folder (`depth/` or `upscaler/`) of the pinned snapshot.
    fn stage_task_export(data: &Path, subdir: &str) -> PathBuf {
        let mut dir = PathBuf::new();
        for file in [
            "config.yaml",
            "model.safetensors",
            "empty_prompt.safetensors",
        ] {
            dir = stage(
                data,
                BACKBONE_REPO,
                BACKBONE_REVISION,
                &format!("{subdir}/{file}"),
            );
        }
        dir.join(subdir)
    }

    /// A task dataset of `items` images; each depth item has its depth map (+ a mask on item 0),
    /// each restoration item its aligned degraded input when `paired`.
    fn task_dataset(
        data: &Path,
        objective: &str,
        items: usize,
        paired: bool,
    ) -> (TrainingDataset, PathBuf) {
        let root = data.join("projects/p1/training/datasets/ds_task");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::create_dir_all(root.join("sidecars")).unwrap();
        let entries: Vec<Value> = (0..items)
            .map(|index| {
                let path = format!("images/item{index}.png");
                std::fs::write(root.join(&path), b"png").unwrap();
                let mut item = json!({
                    "id": format!("item{index}"), "path": path,
                    "displayName": format!("item{index}"),
                    // A caption the task runs never read (and never send).
                    "caption": {"text": "ignored caption", "source": "manual", "triggerWords": []},
                    "addedAt": "2026-10-11T00:00:00Z"
                });
                if objective == "depth" {
                    let depth = format!("sidecars/item{index}.depth.npy");
                    std::fs::write(root.join(&depth), b"npy").unwrap();
                    item["irisDepth"] = json!({"path": depth, "scale": 0.001, "maxDepth": 80.0});
                    if index == 0 {
                        std::fs::write(root.join("sidecars/item0.mask.png"), b"png").unwrap();
                        item["irisDepth"]["maskPath"] = json!("sidecars/item0.mask.png");
                    }
                } else if paired {
                    let degraded = format!("sidecars/item{index}.degraded.png");
                    std::fs::write(root.join(&degraded), b"png").unwrap();
                    item["irisDegradedImage"] = json!(degraded);
                }
                item
            })
            .collect();
        let dataset: TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds_task", "version": 1, "projectId": "p1", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": entries
        }))
        .unwrap();
        (dataset, root)
    }

    fn task_plan(
        data: &Path,
        base_model: &str,
        dataset: (TrainingDataset, PathBuf),
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        let target = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.base_model == base_model)
            .expect("the Iris task target is registered");
        let mut config = target.defaults.clone();
        configure(&mut config);
        let backbone =
            sceneworks_core::hf_home::huggingface_repo_cache_path(data, BACKBONE_REPO).unwrap();
        let output = data.join("models/finetunes/finetune_task");
        build_training_plan(BuildTrainingPlan {
            job_id: "job-iris-train",
            target: &target,
            dataset: &dataset.0,
            config,
            preset: None,
            lora_id: "finetune_task",
            base_model_path: backbone.display().to_string(),
            dataset_root: &dataset.1,
            output_dir: &output,
            file_name: "task_model.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        })
        .expect("the task plan builds")
    }

    /// Every depth option the Studio offers reaches the depth trainer: the shared fields (a FULL
    /// fine-tune, L1) and the `iris` bag verbatim as the ONLY `model_options` key; each item is its
    /// image + its depth target resolved under the dataset root, with no caption. The run loads the
    /// installed `depth/` export alone — no text encoder, no generation backbone.
    #[test]
    fn every_depth_option_reaches_the_depth_trainer_from_the_depth_export() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let export = stage_task_export(&data, "depth");
        let iris = json!({
            "depth_size": [352, 1216], "resize": "none", "crop": "random", "hflip_prob": 0.25,
            "gradient_weight": 4.0, "low_percentile": 1.0, "high_percentile": 99.0,
            "clip_target": true, "loss_reduction": "per_image", "invalid_examples": "skip",
            "min_valid_pixels": 16, "betas": [0.9, 0.99], "gradient_clip": 0.5,
            "muon_momentum": 0.9, "muon_nesterov": false, "muon_adjust_lr": "none",
            "auto_lr": "sqrt", "base_batch_size": 64, "ema_enabled": true, "ema_decay": 0.999,
            "num_epochs": 2, "nan_loss_tolerance": 3, "keep_last_checkpoints": 1,
            "milestone_steps": [5], "preview_items": 2, "preview_weights": "ema",
            "export_weights": "ema"
        });
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 2, false),
            |c| {
                c.optimizer = "muon".to_owned();
                c.gradient_accumulation = 4;
                c.steps = 40;
                c.save_every = 8;
                c.resolution = 1216;
                c.advanced.insert("sampleEvery".into(), json!(8));
                c.advanced.insert("lrWarmupSteps".into(), json!(4));
                c.advanced.insert("lrScheduler".into(), json!("cosine"));
                c.advanced.insert("mixedPrecision".into(), json!("f32"));
                c.advanced.insert("iris".into(), iris.clone());
            },
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(run.seen.weights.as_deref(), Some(export.as_path()));
        assert!(run.seen.components.is_empty(), "{:?}", run.seen.components);
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert!(config.full_finetune);
        assert_eq!(config.loss_type, "l1");
        assert_eq!(config.optimizer, "muon");
        assert_eq!(
            (config.steps, config.save_every, config.sample_every),
            (40, 8, 8)
        );
        assert_eq!(
            (config.gradient_accumulation, config.lr_warmup_steps),
            (4, 4)
        );
        assert_eq!(config.train_dtype, "f32");
        assert_eq!(config.lr_scheduler, gen_core::LrSchedule::Cosine);
        assert!(config.sample_prompts.is_empty() && config.trigger_word.is_none());
        assert_eq!(
            Value::Object(config.model_options.clone()),
            json!({ "iris": iris }),
            "the depth bag is the only model option"
        );
        assert!(request.trigger_words.is_empty());
        let root = data.join("projects/p1/training/datasets/ds_task");
        let item = &request.items[0];
        assert_eq!(item.caption, "", "the depth task reads no caption");
        assert_eq!(
            Value::Object(item.model_options.clone()),
            json!({"depth": {
                "path": root.join("sidecars").join("item0.depth.npy"),
                "mask_path": root.join("sidecars").join("item0.mask.png"),
                "scale": 0.001, "max_depth": 80.0
            }})
        );
        assert!(request.items[1].model_options["depth"]
            .get("mask_path")
            .is_none());
        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        assert_eq!(done["result"]["dataWalk"]["items"], 2);
    }

    /// The paper's start (`depthInit: parent`): the generation backbone, conditioned on the depth
    /// export's SHIPPED empty prompt (no text encoder at all); without a depth install, the generation
    /// install's text encoder encodes it once.
    #[test]
    fn a_parent_depth_run_prefers_the_shipped_empty_prompt_over_the_encoder() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let export = stage_task_export(&data, "depth");
        let parent = |c: &mut sceneworks_core::training::TrainingConfig| {
            c.gradient_accumulation = 1;
            c.steps = 4;
            c.save_every = 2;
            c.advanced.insert("sampleEvery".into(), json!(2));
            c.advanced.insert("depthInit".into(), json!("parent"));
            c.advanced["iris"]["reducer_init"] = json!("channel_mean");
        };
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            parent,
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(
            run.seen.weights.as_deref(),
            Some(install.backbone.as_path())
        );
        assert_eq!(
            run.seen.components,
            vec![(
                "empty_prompt".to_owned(),
                export.join("empty_prompt.safetensors")
            )]
        );

        // No depth install: the generation text encoder (its complete pinned closure), once.
        std::fs::remove_dir_all(&export).unwrap();
        let entry = crate::training_jobs::builtin_model_manifest_entry("iris_3b").unwrap();
        let encoder = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|download| download["componentId"] == "text_encoder")
            .unwrap();
        // The declared shard glob resolves against the shard actually present.
        for file in encoder["files"].as_array().unwrap() {
            let file = file.as_str().unwrap().replace('*', "00001-of-00002");
            stage(&data, TE_REPO, TE_REVISION, &file);
        }
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            parent,
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(
            run.seen.components,
            vec![("text_encoder".to_owned(), install.text_encoder.clone())]
        );
    }

    /// Without the depth export an `export` run is refused BEFORE load, naming the install.
    #[test]
    fn a_depth_run_without_its_install_is_refused_before_load() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_install(&data);
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |c| {
                c.gradient_accumulation = 1;
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(
                    message.contains("Iris 3B Depth is not installed"),
                    "{message}"
                )
            }
            other => panic!("refused before load, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    /// sc-25690: the shared gradient-checkpointing toggle on a depth run, through the real job
    /// path. On a lane whose depth trainer implements it the flag reaches the trainer as the
    /// engine's `gradient_checkpointing`; on a lane that lacks it the job is refused before the
    /// loader runs, naming where it is implemented.
    #[test]
    fn the_depth_checkpointing_toggle_reaches_the_trainer_only_on_a_lane_that_implements_it() {
        use sceneworks_core::training::iris::{iris_gradient_checkpointing_lanes, IrisObjective};
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |config| {
                config
                    .advanced
                    .insert("gradientCheckpointing".into(), json!(true));
            },
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        let lane = crate::iris_training::training_lane();
        if iris_gradient_checkpointing_lanes(IrisObjective::Depth).contains(&lane) {
            assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
            let request = run.seen.request.expect("the trainer ran");
            assert!(request.config.gradient_checkpointing);
        } else {
            match &run.outcome {
                Err(WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("checkpointing is not implemented for Iris 3B depth training")
                        && message.contains("implemented on NVIDIA (Candle/CUDA)"),
                    "{message}"
                ),
                other => panic!("the toggle must be refused before load on {lane}: {other:?}"),
            }
            assert_eq!(run.seen.loads, 0, "refused before the loader ran");
        }
    }

    /// The declared depth floor refuses an undersized host before load (MLX: unified memory minus
    /// the OS reserve; CUDA: VRAM).
    #[test]
    fn a_host_under_the_depth_floor_is_refused_before_load() {
        let _env = isolated_env("48");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |_| {},
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => assert!(
                message.contains("depth full fine-tune at 480×640") && message.contains("GiB"),
                "{message}"
            ),
            other => panic!("a 48 GiB host must be refused for the 480×640 depth run: {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    /// A depth cancel after the first optimizer step publishes the export like a generation run
    /// (Iris-only durable cancel): the terminal Canceled carries the result the API registers from.
    #[test]
    fn a_depth_cancel_after_the_first_step_reports_the_published_export() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |c| {
                c.gradient_accumulation = 1;
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_depth", Behavior::CancelAfterFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert_eq!(last["result"]["trainedArtifactExported"], true, "{last:#}");
    }

    /// Every file under `root` with its bytes (sorted) — a run's SOURCE, compared before and after.
    fn tree(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut files = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push((path.clone(), std::fs::read(&path).unwrap()));
                }
            }
        }
        files.sort();
        files
    }

    /// A REAL depth dataset the engine's readers decode (64x48 RGB + aligned `.npy` depth): item 0
    /// has a usable depth ramp and a validity mask, item 1 a depth plane and no mask.
    fn real_depth_dataset(
        data: &Path,
        mask_value: u8,
        item1_depth: impl Fn(u32) -> f32,
    ) -> (TrainingDataset, PathBuf) {
        let (dataset, root) = task_dataset(data, "depth", 2, false);
        let npy = |f: &dyn Fn(u32) -> f32| {
            let values: Vec<f32> = (0..48u32)
                .flat_map(|y| (0..64).map(move |_| y))
                .map(f)
                .collect();
            sceneworks_core::depth_map::encode_npy_f32(64, 48, &values).unwrap()
        };
        for index in 0..2 {
            image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([x as u8, y as u8, 7]))
                .save(root.join(format!("images/item{index}.png")))
                .unwrap();
        }
        // Stored millimetres under the dataset's `scale: 0.001` / `maxDepth: 80`.
        std::fs::write(
            root.join("sidecars/item0.depth.npy"),
            npy(&|y| 1000.0 + 100.0 * y as f32),
        )
        .unwrap();
        image::GrayImage::from_pixel(64, 48, image::Luma([mask_value]))
            .save(root.join("sidecars/item0.mask.png"))
            .unwrap();
        std::fs::write(root.join("sidecars/item1.depth.npy"), npy(&item1_depth)).unwrap();
        (dataset, root)
    }

    fn small_depth(c: &mut sceneworks_core::training::TrainingConfig) {
        c.gradient_accumulation = 1;
        c.steps = 4;
        c.save_every = 2;
        c.resolution = 64;
        c.advanced.insert("sampleEvery".into(), json!(2));
        c.advanced["iris"]["depth_size"] = json!([64, 64]);
    }

    /// Epic AT6 (sc-25690) — depth training on a dataset with NO valid depth pixel (item 0's mask
    /// excludes every pixel; item 1's depth is 0 everywhere). The request the worker shapes reaches
    /// the engine's own dataset pass, which refuses under BOTH invalid-example policies before any
    /// step: the job fails with the engine's reason, posts no result (so the API registers nothing),
    /// writes no export, and leaves the dataset and the installed export byte-identical. A dataset
    /// with valid pixels passes the same pass and completes (the refusal is the data's, not the
    /// fixture's).
    #[test]
    fn a_depth_dataset_with_no_valid_depth_pixels_is_refused_and_publishes_nothing() {
        let _env = isolated_env("128");
        for (policy, reason) in [
            ("reject", "0 valid depth pixels (min_valid_pixels = 2)"),
            ("skip", "no item of the dataset is a valid depth example"),
        ] {
            let data_holder = tempfile::tempdir().unwrap();
            let data = data_holder.path().canonicalize().unwrap();
            let export = stage_task_export(&data, "depth");
            let (dataset, root) = real_depth_dataset(&data, 0, |_| 0.0);
            let plan = task_plan(&data, "iris_3b_depth", (dataset, root.clone()), |c| {
                small_depth(c);
                c.advanced["iris"]["invalid_examples"] = json!(policy);
            });
            let (dataset_before, export_before) = (tree(&root), tree(&export));
            let run = run_as("iris_3b_depth", Behavior::DepthDatasetPass, plan, &data);
            let error = match &run.outcome {
                Err(WorkerError::Engine(message)) => message.clone(),
                other => panic!("{policy}: expected an engine refusal, got {other:?}"),
            };
            assert!(
                error.contains("training failed") && error.contains(reason),
                "{policy}: {error}"
            );
            if policy == "reject" {
                assert!(error.contains("item0.png"), "the item is named: {error}");
            }
            let request = run.seen.request.as_ref().expect("the trainer was reached");
            assert_eq!(request.items.len(), 2);
            assert!(
                run.posts
                    .iter()
                    .all(|post| post["status"] != "completed" && post["result"].is_null()),
                "{policy}: no completion and no result: {:#?}",
                run.posts
            );
            assert!(
                run.posts.iter().all(|post| post["stage"] != "training"),
                "{policy}: no optimizer step ran: {:#?}",
                run.posts
            );
            assert_eq!(
                tree(&data.join("models/finetunes")),
                Vec::new(),
                "{policy}: nothing was exported"
            );
            assert_eq!(tree(&root), dataset_before, "{policy}: dataset untouched");
            assert_eq!(tree(&export), export_before, "{policy}: install untouched");
        }

        // The control: valid pixels pass the same engine pass and the run completes.
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            real_depth_dataset(&data, 255, |y| 2000.0 + 50.0 * y as f32),
            small_depth,
        );
        let run = run_as("iris_3b_depth", Behavior::DepthDatasetPass, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(terminal(&run.posts)["status"], "completed");
    }

    /// Epic AT6 (sc-25690) — a FAILED export, for every Iris training objective (generation full
    /// fine-tune, generation LoRA and LoKr, depth, restoration): the job ends in the engine's error
    /// with no completion and no result on any post (the API registers a trained artifact only from
    /// a result), no export exists at the output path, the durable checkpoint the trainer published
    /// before the failure is kept (the run stays resumable), and the run's sources — its dataset
    /// and its installed weights — are byte-identical.
    #[test]
    fn a_failed_export_reports_the_failure_and_registers_nothing_for_every_objective() {
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        for objective in [
            "generation_full",
            "generation_lora",
            "generation_lokr",
            "depth",
            "restoration",
        ] {
            let data_holder = tempfile::tempdir().unwrap();
            let data = data_holder.path().canonicalize().unwrap();
            let install = stage_install(&data);
            let small = |c: &mut sceneworks_core::training::TrainingConfig| {
                c.gradient_accumulation = 1;
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            };
            let (engine, plan, output, dataset_root) = match objective {
                "depth" => {
                    stage_task_export(&data, "depth");
                    let dataset = task_dataset(&data, "depth", 2, false);
                    let root = dataset.1.clone();
                    (
                        "iris_3b_depth",
                        task_plan(&data, "iris_3b_depth", dataset, small),
                        data.join("models/finetunes/finetune_task"),
                        root,
                    )
                }
                "restoration" => {
                    stage_task_export(&data, "upscaler");
                    let resources = vec![
                        fixture_resource(
                            &data,
                            "lpips_vgg16",
                            "fixture/vgg",
                            "model.safetensors",
                            b"vgg",
                        ),
                        fixture_resource(
                            &data,
                            "lpips_lin",
                            "fixture/lin",
                            "lin.safetensors",
                            b"lin",
                        ),
                        fixture_resource(
                            &data,
                            "critic_trunk",
                            "fixture/trunk",
                            "trunk.bin",
                            b"trunk",
                        ),
                    ];
                    *crate::iris_task_training::TEST_TRAINING_RESOURCES
                        .lock()
                        .unwrap() = Some(resources);
                    let dataset = task_dataset(&data, "restoration", 2, true);
                    let root = dataset.1.clone();
                    (
                        "iris_3b_restore",
                        task_plan(&data, "iris_3b_restore", dataset, |c| {
                            small(c);
                            c.resolution = 512;
                        }),
                        data.join("models/finetunes/finetune_task"),
                        root,
                    )
                }
                generation => {
                    let network = generation.trim_start_matches("generation_");
                    (
                        "iris_3b",
                        plan(&data, &install.backbone, 2, |c| {
                            small(c);
                            c.advanced.insert("networkType".into(), json!(network));
                        }),
                        data.join("loras/lora_iris"),
                        data.join("projects/p1/training/datasets/ds1"),
                    )
                }
            };
            let file_name = plan.output.file_name.clone();
            let hub = sceneworks_core::hf_home::huggingface_repo_cache_path(&data, BACKBONE_REPO)
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            let (dataset_before, installs_before) = (tree(&dataset_root), tree(&hub));
            assert!(!dataset_before.is_empty() && !installs_before.is_empty());

            let run = run_as(engine, Behavior::FailsExport, plan, &data);
            *crate::iris_task_training::TEST_TRAINING_RESOURCES
                .lock()
                .unwrap() = None;

            let error = match &run.outcome {
                Err(WorkerError::Engine(message)) => message.clone(),
                other => panic!("{objective}: expected the export failure, got {other:?}"),
            };
            assert!(
                error.contains("training failed") && error.contains("export failed"),
                "{objective}: {error}"
            );
            assert!(run.seen.request.is_some(), "{objective}: the trainer ran");
            assert!(
                run.posts.iter().any(|post| post["stage"] == "training"),
                "{objective}: a step was reported before the failure: {:#?}",
                run.posts
            );
            assert!(
                run.posts
                    .iter()
                    .all(|post| post["status"] != "completed" && post["result"].is_null()),
                "{objective}: no completion and no result (nothing to register): {:#?}",
                run.posts
            );
            let stem = file_name.trim_end_matches(".safetensors");
            assert!(
                !output.join(&file_name).exists() && !output.join(stem).exists(),
                "{objective}: no export at the output path"
            );
            let left: Vec<PathBuf> = tree(&output).into_iter().map(|(path, _)| path).collect();
            assert_eq!(
                left,
                [output.join(format!("{stem}.checkpoints/step_00000001/state.json"))],
                "{objective}: only the durable checkpoint is left beside the missing export"
            );
            assert_eq!(
                tree(&dataset_root),
                dataset_before,
                "{objective}: dataset untouched"
            );
            assert_eq!(
                tree(&hub),
                installs_before,
                "{objective}: installs untouched"
            );
        }
    }

    fn fixture_resource(
        data: &Path,
        component: &'static str,
        repo: &'static str,
        file: &'static str,
        bytes: &[u8],
    ) -> gen_core_iris::restoration_train::TrainingResource {
        use sha2::Digest;
        let revision = "0123456789abcdef0123456789abcdef01234567";
        stage_bytes(data, repo, revision, file, bytes);
        gen_core_iris::restoration_train::TrainingResource {
            component,
            name: component,
            url: "https://example.invalid/fixture",
            repo,
            revision,
            file,
            sha256: Box::leak(format!("{:x}", sha2::Sha256::digest(bytes)).into_boxed_str()),
            bytes: bytes.len() as u64,
            license: "fixture",
        }
    }

    /// Every restoration option reaches the trainer: HYPIR's LoRA arm (the recipe's full fine-tune
    /// is the default), the `iris_restore` bag as the ONLY model option, aligned pairs as each item's
    /// `degraded_image`, the three training-only resources staged as components after their pinned
    /// size + sha256 verify, and the generation backbone as `parent_backbone` under `init: parent`.
    #[test]
    fn every_restoration_option_reaches_the_trainer_with_verified_resources() {
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let export = stage_task_export(&data, "upscaler");
        let resources = vec![
            fixture_resource(
                &data,
                "lpips_vgg16",
                "fixture/vgg",
                "model.safetensors",
                b"vgg",
            ),
            fixture_resource(&data, "lpips_lin", "fixture/lin", "lin.safetensors", b"lin"),
            fixture_resource(
                &data,
                "critic_trunk",
                "fixture/trunk",
                "trunk.bin",
                b"trunk",
            ),
        ];
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = Some(resources.clone());
        let bag = json!({
            "loss_l2": 1.0, "loss_lpips": 4.0, "loss_gan": 0.25, "gan_real_label": 0.9,
            "lr_critic": 2e-5, "betas": [0.9, 0.99], "critic_betas": [0.5, 0.99],
            "critic_weight_decay": 0.0, "gradient_clip": 0.5, "critic_gradient_clip": 2.0,
            "ema_enabled": true, "ema_decay": 0.99, "crop": "center",
            "degradation": {"use_hflip": false}, "critic_dtype": "fp32", "init": "parent",
            "num_epochs": 3, "nan_loss_tolerance": 5, "keep_last_checkpoints": 1,
            "milestone_steps": [2], "export_weights": "raw", "export_dtype": "bf16",
            "preview_weights": "ema", "lora_init": "kaiming"
        });
        let plan = task_plan(
            &data,
            "iris_3b_restore",
            task_dataset(&data, "restoration", 2, true),
            |c| {
                c.rank = 8;
                c.alpha = sceneworks_core::contracts::ContractNumber::from(4);
                c.gradient_accumulation = 2;
                c.steps = 40;
                c.save_every = 8;
                c.resolution = 512;
                c.advanced.insert("networkType".into(), json!("lokr"));
                c.advanced.insert("decomposeFactor".into(), json!(4));
                c.advanced.insert("sampleEvery".into(), json!(8));
                c.advanced.insert("iris_restore".into(), bag.clone());
            },
        );
        let run = run_as("iris_3b_restore", Behavior::Completes, plan, &data);
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = None;
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(run.seen.weights.as_deref(), Some(export.as_path()));
        let mut components = run.seen.components.clone();
        components.sort();
        let resource_path = |resource: &gen_core_iris::restoration_train::TrainingResource| {
            crate::model_jobs::huggingface_pinned_snapshot_dir(
                &data,
                resource.repo,
                resource.revision,
            )
            .unwrap()
            .join(resource.file)
        };
        assert_eq!(
            components,
            vec![
                ("critic_trunk".to_owned(), resource_path(&resources[2])),
                ("lpips_lin".to_owned(), resource_path(&resources[1])),
                ("lpips_vgg16".to_owned(), resource_path(&resources[0])),
                ("parent_backbone".to_owned(), install.backbone.clone()),
            ]
        );
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert!(!config.full_finetune);
        assert_eq!(config.network_type, gen_core::NetworkType::Lokr);
        assert_eq!(
            (config.rank, config.alpha, config.decompose_factor),
            (8, 4.0, 4)
        );
        assert_eq!(
            (config.steps, config.save_every, config.sample_every),
            (40, 8, 8)
        );
        assert_eq!(config.resolution, 512);
        assert_eq!(config.loss_type, "mse");
        assert!(config.sample_prompts.is_empty());
        assert_eq!(
            Value::Object(config.model_options.clone()),
            json!({ "iris_restore": bag })
        );
        let root = data.join("projects/p1/training/datasets/ds_task");
        assert_eq!(request.items[1].caption, "");
        assert_eq!(
            Value::Object(request.items[1].model_options.clone()),
            json!({"degraded_image": root.join("sidecars").join("item1.degraded.png")})
        );
    }

    /// A staged training resource that does not match its pin (here: the real pins against stand-in
    /// bytes) is refused BEFORE load, naming the Model Manager install to repair.
    #[test]
    fn a_restoration_resource_failing_its_pin_is_refused_before_load() {
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "upscaler");
        for resource in gen_core_iris::restoration_train::TRAINING_RESOURCES {
            stage(&data, resource.repo, resource.revision, resource.file);
        }
        let plan = task_plan(
            &data,
            "iris_3b_restore",
            task_dataset(&data, "restoration", 1, false),
            |c| {
                c.resolution = 256;
                c.advanced.insert("networkType".into(), json!("lora"));
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_restore", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => assert!(
                message.contains("re-install") && message.contains("iris_restore_train_vgg16"),
                "{message}"
            ),
            other => panic!("a resource failing its pin must be refused: {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);

        // A missing resource is named before anything is hashed.
        let missing = sceneworks_core::hf_home::huggingface_repo_cache_path(
            &data,
            gen_core_iris::restoration_train::CRITIC_TRUNK_RESOURCE.repo,
        )
        .unwrap();
        std::fs::remove_dir_all(missing).unwrap();
        let refusal = crate::iris_task_training::iris_task_training_unavailable(
            &data,
            sceneworks_core::training::iris::IrisObjective::Restoration,
            &builtin_training_targets()
                .targets
                .into_iter()
                .find(|t| t.base_model == "iris_3b_restore")
                .unwrap()
                .defaults,
        )
        .expect("the trunk is missing");
        assert!(
            refusal.contains("iris_restore_train_critic_trunk"),
            "{refusal}"
        );
    }

    // ---- S11 (sc-25689): curricula, REPA, holdout, fp16 and multi-rank execution -------------

    const TEACHER_BYTES: &[u8] = b"a stand-in dinov2-base model.safetensors";

    fn stage_teacher(data: &Path) -> PathBuf {
        let snapshot = stage(
            data,
            sceneworks_core::training::iris::curricula::IRIS_REPA_TEACHER_REPO,
            sceneworks_core::training::iris::curricula::IRIS_REPA_TEACHER_REVISION,
            "model.safetensors",
        );
        std::fs::write(snapshot.join("model.safetensors"), TEACHER_BYTES).unwrap();
        snapshot
    }

    /// Every S11 option the Studio offers reaches the trainer: the curriculum, shape, RoPE, REPA,
    /// activation-checkpointing, validation, logging, retention, global-batch and preview keys in
    /// the bag verbatim, fp16 as the engine dtype, the holdout flag on its item, and the REPA
    /// teacher staged as the `repa_teacher` component after its integrity check.
    #[test]
    fn every_offered_s11_option_reaches_the_trainer_with_the_teacher_staged() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let teacher = stage_teacher(&data);
        let iris = json!({
            "stage": "stage1_256",
            "run_name": "iris s1",
            "tracking": {"project": "iris", "tags": ["s1"]},
            "shape_policy": "area",
            "shape_align": 32,
            "shape_max_ratio": 2.0,
            "rope_aspect": "square",
            "repa": {"weight": 0.5, "variant": "irepa", "spatial_norm_gamma": 0.8,
                     "proj_hidden_dim": 512, "teacher_image_size": 224,
                     "teacher_match_student": true, "layer": 10},
            "activation_checkpointing": "selective_layer",
            "ac_selective_every": 3,
            "val_every_steps": 5,
            "val_samples": 2,
            "log_every": 5,
            "log_block_grad_norms": true,
            "save_every_epochs": 1,
            "expected_global_batch": 4,
            "scale_warmup_by_world": false,
            "perf": {"lazy_sync": true},
            "dist": {"strategy": "ddp"},
            "preview_at_step_one": true,
            "preview_order": 1,
            "upstream_validation_prompts": ["a lighthouse at dusk"]
        });
        let plan = plan_with_dataset(
            &data,
            &install.backbone,
            3,
            |dataset| {
                dataset.items[2]
                    .extra
                    .insert("split".into(), json!("holdout"));
            },
            |config| {
                config.batch_size = 2;
                config.gradient_accumulation = 2;
                config.steps = 40;
                config.save_every = 10;
                config
                    .advanced
                    .insert("mixedPrecision".into(), json!("fp16"));
                config.advanced.insert("sampleEvery".into(), json!(10));
                config.advanced.insert("samplePrompts".into(), json!([]));
                config.advanced.insert("iris".into(), iris.clone());
            },
        );
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(
            run.seen.teacher.as_deref(),
            Some(teacher.as_path()),
            "the verified teacher is the trainer's repa_teacher component"
        );
        let request = run.seen.request.expect("the trainer ran");
        assert_eq!(request.config.train_dtype, "fp16");
        assert_eq!(request.config.model_options.get("iris"), Some(&iris));
        assert_eq!(
            request.items[2].model_options.get("split"),
            Some(&json!("holdout"))
        );
        assert!(request.items[0].model_options.get("split").is_none());
        // The engine-resolved receipts ride on the result.
        let result = &terminal(&run.posts)["result"];
        let resolved = &result["irisPlan"];
        assert_eq!(resolved["stage"], json!("stage1_256"));
        assert_eq!(resolved["globalBatch"], json!(4));
        assert_eq!(
            resolved["activationCheckpointing"],
            json!("selective_layer")
        );
        assert_eq!(resolved["repa"]["variant"], json!("irepa"));
        assert_eq!(resolved["holdout"]["items"], json!(1));
        assert_eq!(resolved["mixedPrecision"], json!("fp16"));
        assert!(resolved["shapePolicy"]
            .as_str()
            .unwrap()
            .starts_with("area:"));
        assert!(resolved["dataIdentity"]
            .as_str()
            .unwrap()
            .contains("shape=area"));
    }

    /// sc-25690: the real-weight FULL fine-tune smokes' OWN configs
    /// (`iris_training_gpu_smoke::CancelResumeArm::FullStage1` and `restore_smoke_config(true)`)
    /// through this real job path with only the trainer faked — the memory floor at the full rows,
    /// the REPA teacher / restoration resources staged, the engine plan resolved — for the first
    /// run and for its resumed twin. A real-weight dispatch then starts from a config the job path
    /// is known to admit and hand to the trainer as a FULL fine-tune of the official stage.
    #[test]
    fn the_full_fine_tune_smoke_configs_reach_the_trainer_through_the_job_path() {
        use crate::iris_training_gpu_smoke::{
            apply_resume_translation, restore_smoke_config, CancelResumeArm,
        };
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let teacher = stage_teacher(&data);

        let smoke = CancelResumeArm::FullStage1.config();
        for resumed in [false, true] {
            let mut plan = plan(&data, &install.backbone, 4, |config| {
                *config = smoke.clone()
            });
            if resumed {
                let root = data.join("loras/lora_iris/iris_style.checkpoints");
                std::fs::create_dir_all(&root).unwrap();
                apply_resume_translation(&mut plan.config, &root);
            }
            let run = run(Behavior::Completes, plan, &data);
            assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
            assert_eq!(run.seen.teacher.as_deref(), Some(teacher.as_path()));
            let request = run.seen.request.expect("the trainer ran");
            assert!(request.config.full_finetune, "a FULL fine-tune");
            assert_eq!(request.config.optimizer, "muon");
            assert_eq!(
                (request.config.batch_size, request.config.resolution),
                (1, 256)
            );
            let options = &request.config.model_options["iris"];
            assert_eq!(options["stage"], json!("stage1_256"));
            assert!(options["repa"]["weight"].as_f64() > Some(0.0));
            assert_eq!(
                options.get("resume_from").is_some(),
                resumed,
                "only the resumed run carries the checkpoint root: {options:#}"
            );
            let resolved = &terminal(&run.posts)["result"]["irisPlan"];
            assert_eq!(resolved["stage"], json!("stage1_256"));
            assert!(resolved["repa"].is_object(), "{resolved:#}");
        }

        stage_task_export(&data, "upscaler");
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = Some(vec![
            fixture_resource(
                &data,
                "lpips_vgg16",
                "fixture/vgg",
                "model.safetensors",
                b"vgg",
            ),
            fixture_resource(&data, "lpips_lin", "fixture/lin", "lin.safetensors", b"lin"),
            fixture_resource(
                &data,
                "critic_trunk",
                "fixture/trunk",
                "trunk.bin",
                b"trunk",
            ),
        ]);
        let plan = task_plan(
            &data,
            "iris_3b_restore",
            task_dataset(&data, "restoration", 2, false),
            |config| *config = restore_smoke_config(true),
        );
        let run = run_as("iris_3b_restore", Behavior::Completes, plan, &data);
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = None;
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        let request = run.seen.request.expect("the restoration trainer ran");
        assert!(request.config.full_finetune, "the paper's FULL fine-tune");
        assert_eq!(request.config.resolution, 256);
    }

    /// REPA without the installed teacher is refused before the loader runs, naming the Model
    /// Manager install; a teacher whose bytes are not the pinned ones is refused the same way.
    #[test]
    fn repa_needs_the_installed_verified_teacher() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let repa = |config: &mut sceneworks_core::training::TrainingConfig| {
            config
                .advanced
                .insert("iris".into(), json!({"repa": {"weight": 0.5}}));
        };
        let run_missing = run(
            Behavior::Completes,
            plan(&data, &install.backbone, 2, repa),
            &data,
        );
        match &run_missing.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("Model Manager"), "{message}")
            }
            other => panic!("REPA without a teacher must be refused, got {other:?}"),
        }
        assert_eq!(run_missing.seen.loads, 0);
        let teacher = stage_teacher(&data);
        let mut tampered = TEACHER_BYTES.to_vec();
        tampered[0] ^= 1;
        std::fs::write(teacher.join("model.safetensors"), &tampered).unwrap();
        let run_tampered = run(
            Behavior::Completes,
            plan(&data, &install.backbone, 2, repa),
            &data,
        );
        match &run_tampered.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("sha256"), "{message}")
            }
            other => panic!("a tampered teacher must be refused, got {other:?}"),
        }
        assert_eq!(run_tampered.seen.loads, 0);
    }

    /// The engine's own plan resolution refuses what only it knows (a REPA capture block outside
    /// the 24-block trunk) before anything loads.
    #[test]
    fn the_engine_plan_refusals_arrive_before_load() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        stage_teacher(&data);
        let plan = plan(&data, &install.backbone, 2, |config| {
            config
                .advanced
                .insert("iris".into(), json!({"repa": {"weight": 0.5, "layer": 99}}));
        });
        let run = run(Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("capture block"), "{message}")
            }
            other => panic!("an out-of-trunk REPA layer must be refused, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    fn fleet_records_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    /// Drive the REAL job path with fake rank processes. The in-process loader must never run.
    fn run_fleet(
        behavior: &'static str,
        plan: TrainingPlan,
        data: &Path,
        records: &Path,
        cancel_on_training: bool,
    ) -> Run {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            if cancel_on_training {
                let posts = Arc::clone(&posts);
                let user_cancel = Arc::clone(&user_cancel);
                crate::load_progress::spawn(async move {
                    loop {
                        let training = posts.lock().unwrap().iter().any(|post: &Value| {
                            post["message"]
                                .as_str()
                                .is_some_and(|message| message.starts_with("Training step"))
                        });
                        if training {
                            user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                });
            }
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            // Off-Mac the Iris memory gate reads the worker's NVIDIA card; `isolated_env` caps it
            // (`SCENEWORKS_CUDA_VRAM_CAP_GB`), so the bound is the test's, not the runner's.
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            {
                settings.gpu_id = "0".to_owned();
            }
            settings.api_url = base_url;
            settings.data_dir = data.to_path_buf();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let loader = |_: &str, _: &LoadSpec| -> gen_core::Result<Box<dyn Trainer>> {
                panic!("a multi-rank run never loads a trainer in the worker process")
            };
            let outcome = tokio::time::timeout(
                Duration::from_secs(120),
                crate::training_jobs::run_training_execution_drive(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    loader,
                    Arc::new(crate::iris_dist::test_support::FakeRankLauncher {
                        behavior,
                        record_dir: records.to_path_buf(),
                    }),
                ),
            )
            .await
            .expect("the training job finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        for record in crate::iris_dist::test_support::records(records) {
            assert!(
                !crate::iris_dist::test_support::alive(record["pid"].as_u64().unwrap()),
                "rank {} outlived the job",
                record["rank"]
            );
        }
        Run {
            outcome,
            posts,
            seen: Seen::default(),
        }
    }

    fn two_ranks(
        share_device: bool,
    ) -> impl FnOnce(&mut sceneworks_core::training::TrainingConfig) {
        move |config| {
            config.advanced.insert(
                "iris".into(),
                json!({"dist": {"strategy": "ddp",
                                "world": {"world_size": 2, "share_device": share_device}}}),
            );
        }
    }

    /// A two-rank run: the worker coordinates two rank processes, rank 0's progress drives the job
    /// and its export completes it, with the topology recorded on the result.
    #[test]
    fn a_two_rank_run_is_coordinated_and_rank_zero_completes_the_job() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let plan = plan(&data, &install.backbone, 2, two_ranks(true));
        let run = run_fleet("complete", plan, &data, records.path(), false);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(
            crate::iris_dist::test_support::records(records.path()).len(),
            2
        );
        assert!(run.posts.iter().any(|post| post["message"]
            .as_str()
            .is_some_and(|message| message.contains("Training step 4 of 4"))));
        assert!(
            !run.posts.iter().any(|post| post["message"]
                .as_str()
                .is_some_and(|message| message.contains("step 99"))),
            "rank 1's progress is not the job's"
        );
        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        let dist = &done["result"]["irisDistributed"];
        assert_eq!(dist["worldSize"], json!(2));
        assert_eq!(dist["shareDevice"], json!(true));
        assert_eq!(dist["topology"], json!("single_host"));
        assert_eq!(done["result"]["irisPlan"]["globalBatch"], json!(2));
        assert_eq!(done["result"]["irisPlan"]["dist"]["world_size"], json!(2));
    }

    /// More ranks than the worker's devices without the share-device opt-in is a typed refusal
    /// before any rank process starts; two ranks on one device need twice the declared floor.
    #[test]
    fn unsupported_topologies_are_refused_before_any_rank_starts() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "complete",
            plan(&data, &install.backbone, 2, two_ranks(false)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("share device"), "{message}")
            }
            other => panic!("2 ranks on 1 device without share_device: {other:?}"),
        }
        assert!(crate::iris_dist::test_support::records(records.path()).is_empty());
        drop(_env);
        // One adapter rank fits a 32 GB host; two on its one device do not.
        let _small = isolated_env("32");
        let run = run_fleet(
            "complete",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("2 ranks"), "{message}")
            }
            other => panic!("two ranks need twice the floor: {other:?}"),
        }
        assert!(crate::iris_dist::test_support::records(records.path()).is_empty());
    }

    /// A rank that dies fails the job with a typed message naming it; the survivor is stopped.
    #[test]
    fn a_lost_rank_fails_the_job_typed() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "rank1_dies",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::Engine(message)) => {
                assert!(message.contains("[rank_lost] on rank 1 of 2"), "{message}")
            }
            other => panic!("a lost rank must fail the job: {other:?}"),
        }
    }

    /// A user cancel reaches every rank; rank 0 finishes durably, so the job ends Canceled with the
    /// export reported for registration.
    #[test]
    fn a_user_cancel_fans_out_to_every_rank() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "wait_cancel",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            true,
        );
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        for rank in 0..2 {
            assert!(
                records.path().join(format!("cancel-{rank}")).is_file(),
                "rank {rank} saw the cancel"
            );
        }
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert_eq!(last["result"]["trainedArtifactExported"], json!(true));
    }

    /// The rank process itself: the SAME preflight and trainer inputs as the job, its world stamped
    /// onto the request, progress and the export written as protocol lines, and a typed distributed
    /// failure reported with its kind.
    #[test]
    fn the_rank_process_trains_with_its_world_stamped() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, two_ranks(true));
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        // Off-Mac the Iris memory gate reads the worker's NVIDIA card; `isolated_env` caps it
        // (`SCENEWORKS_CUDA_VRAM_CAP_GB`), so the bound is the test's, not the runner's.
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            settings.gpu_id = "0".to_owned();
        }
        settings.data_dir = data.clone();
        let spec = crate::iris_dist::RankSpec {
            job_id: "job-iris-train".to_owned(),
            plan,
        };
        let world = crate::iris_dist::RankWorld {
            rank: 1,
            world_size: 2,
            master_addr: "127.0.0.1".to_owned(),
            master_port: 41000,
        };
        let seen = Arc::new(Mutex::new(Seen::default()));
        let descriptor = crate::inference_runtime::trainer_descriptor("iris_3b").unwrap();
        let loader_seen = Arc::clone(&seen);
        let loader = move |_: &str, spec: &LoadSpec| {
            if let Some(WeightsSource::Dir(dir)) = spec.components.get("text_encoder") {
                loader_seen.lock().unwrap().text_encoder = Some(dir.clone());
            }
            Ok(Box::new(FakeIrisTrainer {
                descriptor,
                behavior: Behavior::Completes,
                seen: Arc::clone(&loader_seen),
                user_cancel: Arc::default(),
            }) as Box<dyn Trainer>)
        };
        let mut out = Vec::new();
        let code = crate::iris_dist::run_rank_process_using(
            &spec,
            &world,
            &settings,
            loader,
            std::io::empty(),
            &mut out,
        );
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&out));
        let seen = seen.lock().unwrap();
        let request = seen.request.as_ref().expect("the rank trained");
        let stamped = &request.config.model_options["iris"]["dist"]["world"];
        assert_eq!(stamped["rank"], json!(1));
        assert_eq!(stamped["master_port"], json!(41000));
        assert_eq!(stamped["share_device"], json!(true));
        assert_eq!(
            seen.text_encoder.as_deref(),
            Some(install.text_encoder.as_path())
        );
        let lines: Vec<_> = String::from_utf8(out)
            .unwrap()
            .lines()
            .filter_map(crate::iris_dist::parse_message)
            .collect();
        assert!(matches!(
            lines.last(),
            Some(crate::iris_dist::RankMessage::Done { steps: 1000, .. })
        ));
        assert!(lines.iter().any(|line| matches!(
            line,
            crate::iris_dist::RankMessage::Progress {
                progress: crate::iris_dist::WireProgress::Training { .. }
            }
        )));

        // A launcher world that disagrees with the plan is a split world, reported typed.
        let wrong = crate::iris_dist::RankWorld {
            world_size: 3,
            ..world
        };
        let mut out = Vec::new();
        let code = crate::iris_dist::run_rank_process_using(
            &spec,
            &wrong,
            &settings,
            |_: &str, _: &LoadSpec| -> gen_core::Result<Box<dyn Trainer>> {
                panic!("a split world never loads")
            },
            std::io::empty(),
            &mut out,
        );
        assert_eq!(code, 1);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("split world"), "{text}");
    }
}
