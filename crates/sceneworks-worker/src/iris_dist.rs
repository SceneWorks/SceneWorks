//! Multi-rank Iris-3B training on ONE host (story sc-25689, epic sc-25678).
//!
//! The Iris trainers are multi-PROCESS (`gen_core_iris::train::dist`: one process per rank, a
//! host TCP collective with bounded waits, typed `DistError`s). SceneWorks hosts such a run on the
//! worker that claimed the job:
//!
//! * the worker becomes the **coordinator**: it never loads a trainer, it launches `world_size` rank
//!   processes of THIS executable ([`CurrentExeLauncher`]; both the API binary and the standalone
//!   worker binary re-enter through [`iris_rank_entry`] before anything else runs), each with the
//!   torchrun-style world in its environment (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
//!   `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), the per-job rendezvous secret in
//!   `SCENEWORKS_DIST_TOKEN` (never in the persisted config) and the plan in a private spec file;
//! * **rank 0's** progress (steps, losses, checkpoints, previews, the export) feeds the job's
//!   ordinary event consumer — so progress, SSE, previews, cancel and registration are the
//!   single-process path's;
//! * **cancel** fans out to every rank (a `cancel` line on its stdin trips its engine flag; the
//!   trainers agree on it collectively, so a cancel after the first optimizer step still ends with
//!   rank 0's durable checkpoint + export);
//! * a **rank that fails** (a typed `DistError`, an engine error, or a process that dies) fails the
//!   job with a typed message naming the rank and the kind; the surviving ranks are told to stop
//!   and are killed after a bounded grace, and every rank process is reaped — no orphans (a rank
//!   whose coordinator disappears sees EOF on its stdin, cancels, and exits within a bound).
//!
//! Topology: the ranks run on the devices THIS worker owns. A SceneWorks GPU worker owns one
//! device (the supervisor starts one worker per GPU, each with `CUDA_VISIBLE_DEVICES` = its GPU;
//! the Mac has one Metal device), so `world_size > devices` needs the explicit
//! `dist.world.share_device` opt-in, and multi-GPU ranks need a worker configured with several
//! GPUs (`SCENEWORKS_GPU_ID=0,1` + `CUDA_VISIBLE_DEVICES=0,1`). Multi-host runs are not hosted: the
//! scheduler has no multi-host job concept (see the story's closeout).

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use sceneworks_core::training::iris::curricula;
use sceneworks_core::training::iris::{is_iris_training_kernel, IRIS_OPTIONS_KEY};
use sceneworks_core::training::TrainingPlan;

/// The env var naming a rank process's spec file. Its presence turns this executable into a rank.
pub const RANK_SPEC_ENV: &str = "SCENEWORKS_IRIS_RANK_SPEC";
/// The prefix of every protocol line a rank writes on stdout (anything else on stdout is noise).
pub const RANK_LINE_MARKER: &str = "@@sceneworks-iris-rank@@ ";
/// The stdin line that cancels a rank.
pub const CANCEL_LINE: &str = "cancel";
/// How long the other ranks get to stop once the fleet has a terminal outcome (a failure, or rank 0
/// finished) before they are killed.
const PEER_STOP_GRACE: Duration = Duration::from_secs(120);
/// How long a canceled fleet gets to stop (rank 0's durable checkpoint + export of a full model
/// is the slow part) before the remaining ranks are killed.
const CANCEL_GRACE: Duration = Duration::from_secs(600);
/// How long an orphaned rank (its coordinator's stdin closed) keeps running after it cancels, before
/// it exits on its own.
const ORPHAN_EXIT_BOUND: Duration = Duration::from_secs(600);

/// The world size the plan asks for (1 for every non-Iris plan).
pub(crate) fn fleet_world_size(plan: &TrainingPlan) -> usize {
    // Distribution is a generation-objective option (the depth / restoration task trainers are
    // single-process).
    if !is_iris_training_kernel(&plan.target.kernel)
        || sceneworks_core::training::iris::IrisObjective::from_base_model(&plan.target.base_model)
            != Some(sceneworks_core::training::iris::IrisObjective::Generation)
    {
        return 1;
    }
    curricula::iris_world_size(&plan.config) as usize
}

/// How many accelerators THIS worker owns: one on Apple Silicon (one Metal device) and one per id
/// in the worker's `gpu_id` on NVIDIA (the supervisor's per-GPU workers own exactly one).
pub(crate) fn visible_training_devices(settings: &crate::settings::Settings) -> u64 {
    if cfg!(target_os = "macos") {
        return 1;
    }
    (crate::iris_training::worker_gpu_ids(&settings.gpu_id).len() as u64).max(1)
}

/// The topology receipt recorded on the job result of a multi-rank run.
pub(crate) fn fleet_receipt(plan: &TrainingPlan, world_size: usize) -> Value {
    let mut receipt = curricula::iris_curriculum_receipt(&plan.config)["dist"].clone();
    if let Some(object) = receipt.as_object_mut() {
        object.insert("topology".to_owned(), json!("single_host"));
        object.insert("transport".to_owned(), json!("tcp"));
        object.insert("rankProcesses".to_owned(), json!(world_size));
    }
    receipt
}

// =============================================================================================
// The wire protocol (rank → coordinator)
// =============================================================================================

/// One rank's world, as its launcher hands it over (torchrun's environment contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RankWorld {
    pub rank: usize,
    pub world_size: usize,
    pub master_addr: String,
    pub master_port: u16,
}

impl RankWorld {
    pub(crate) fn env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("RANK", self.rank.to_string()),
            ("LOCAL_RANK", self.rank.to_string()),
            ("WORLD_SIZE", self.world_size.to_string()),
            ("LOCAL_WORLD_SIZE", self.world_size.to_string()),
            ("GROUP_RANK", "0".to_owned()),
            ("MASTER_ADDR", self.master_addr.clone()),
            ("MASTER_PORT", self.master_port.to_string()),
        ]
    }

    pub(crate) fn from_env() -> Result<Self, String> {
        let var = |key: &str| {
            std::env::var(key).map_err(|_| format!("the rank environment is missing {key}"))
        };
        let number = |key: &str| -> Result<usize, String> {
            var(key)?
                .trim()
                .parse()
                .map_err(|_| format!("{key} is not a rank number"))
        };
        Ok(Self {
            rank: number("RANK")?,
            world_size: number("WORLD_SIZE")?,
            master_addr: var("MASTER_ADDR")?,
            master_port: var("MASTER_PORT")?
                .trim()
                .parse()
                .map_err(|_| "MASTER_PORT is not a port".to_owned())?,
        })
    }
}

/// The plan a rank trains (written by the coordinator into a private directory).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RankSpec {
    pub job_id: String,
    pub plan: TrainingPlan,
}

/// A typed distributed failure on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WireDistError {
    pub kind: String,
    pub detail: String,
}

/// One `TrainingProgress` on the wire (a preview's RGB pixels base64-encoded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub(crate) enum WireProgress {
    Preparing,
    LoadingModel,
    Caching {
        current: u32,
        total: u32,
    },
    Training {
        step: u32,
        total: u32,
        loss: f32,
    },
    Checkpoint {
        step: u32,
    },
    Sample {
        step: u32,
        index: u32,
        total: u32,
        prompt: String,
        width: u32,
        height: u32,
        pixels: String,
    },
    Saving,
}

impl WireProgress {
    pub(crate) fn from_progress(progress: &gen_core::TrainingProgress) -> Self {
        use gen_core::TrainingProgress as P;
        match progress {
            P::Preparing => Self::Preparing,
            P::LoadingModel => Self::LoadingModel,
            P::Caching { current, total } => Self::Caching {
                current: *current,
                total: *total,
            },
            P::Training { step, total, loss } => Self::Training {
                step: *step,
                total: *total,
                loss: *loss,
            },
            P::Checkpoint { step } => Self::Checkpoint { step: *step },
            P::Sample {
                step,
                index,
                total,
                prompt,
                image,
            } => Self::Sample {
                step: *step,
                index: *index,
                total: *total,
                prompt: prompt.clone(),
                width: image.width,
                height: image.height,
                pixels: base64::engine::general_purpose::STANDARD.encode(&image.pixels),
            },
            P::Saving => Self::Saving,
        }
    }

    pub(crate) fn into_progress(self) -> Option<gen_core::TrainingProgress> {
        use gen_core::TrainingProgress as P;
        Some(match self {
            Self::Preparing => P::Preparing,
            Self::LoadingModel => P::LoadingModel,
            Self::Caching { current, total } => P::Caching { current, total },
            Self::Training { step, total, loss } => P::Training { step, total, loss },
            Self::Checkpoint { step } => P::Checkpoint { step },
            Self::Sample {
                step,
                index,
                total,
                prompt,
                width,
                height,
                pixels,
            } => P::Sample {
                step,
                index,
                total,
                prompt,
                image: gen_core::Image {
                    width,
                    height,
                    pixels: base64::engine::general_purpose::STANDARD
                        .decode(pixels)
                        .ok()?,
                },
            },
            Self::Saving => P::Saving,
        })
    }
}

/// One protocol message a rank writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum RankMessage {
    Progress {
        progress: WireProgress,
    },
    Done {
        adapter_path: String,
        steps: u32,
        final_loss: f32,
    },
    Failed {
        message: String,
        dist: Option<WireDistError>,
        canceled: bool,
    },
}

pub(crate) fn write_message(out: &mut dyn Write, message: &RankMessage) {
    if let Ok(line) = serde_json::to_string(message) {
        let _ = writeln!(out, "{RANK_LINE_MARKER}{line}");
        let _ = out.flush();
    }
}

/// Parse a protocol line. The marker may follow other output on the same line (a library that
/// printed without a newline), so it is searched for, not required at the start.
pub(crate) fn parse_message(line: &str) -> Option<RankMessage> {
    let start = line.find(RANK_LINE_MARKER)? + RANK_LINE_MARKER.len();
    serde_json::from_str(line[start..].trim_end()).ok()
}

/// Stamp this rank's place in the world onto the request's options bag (`dist.world`): the client's
/// `world_size` / `share_device` and wait bounds (`timeout_secs`, `rendezvous_timeout_secs`,
/// `cancel_grace_secs`) stay; the rank numbering and rendezvous are the launcher's. A world size that disagrees with the plan is refused (a split world).
pub(crate) fn apply_rank_world(
    model_options: &mut Map<String, Value>,
    world: &RankWorld,
) -> Result<(), String> {
    let iris = model_options
        .entry(IRIS_OPTIONS_KEY.to_owned())
        .or_insert_with(|| json!({}));
    let iris = iris
        .as_object_mut()
        .ok_or("the Iris options bag is not an object")?;
    let dist = iris.entry("dist").or_insert_with(|| json!({}));
    let dist = dist.as_object_mut().ok_or("dist is not an object")?;
    let world_entry = dist.entry("world").or_insert_with(|| json!({}));
    let world_entry = world_entry
        .as_object_mut()
        .ok_or("dist.world is not an object")?;
    let planned = world_entry
        .get("world_size")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    if planned != world.world_size as u64 {
        return Err(format!(
            "the launcher started a {}-rank world but the plan asks for {planned}",
            world.world_size
        ));
    }
    if world.rank >= world.world_size {
        return Err(format!(
            "rank {} is outside the {}-rank world",
            world.rank, world.world_size
        ));
    }
    for (key, value) in [
        ("rank", json!(world.rank)),
        ("local_rank", json!(world.rank)),
        ("world_size", json!(world.world_size)),
        ("local_world_size", json!(world.world_size)),
        ("node_rank", json!(0)),
        ("num_nodes", json!(1)),
        ("master_addr", json!(world.master_addr)),
        ("master_port", json!(world.master_port)),
        ("transport", json!("tcp")),
    ] {
        world_entry.insert(key.to_owned(), value);
    }
    Ok(())
}

// =============================================================================================
// The rank process
// =============================================================================================

/// The rank-process entry point. Call it FIRST in every binary that can host the worker: `None` when
/// this process is not a rank (no [`RANK_SPEC_ENV`]); otherwise it trains one rank to completion
/// and returns the process exit code (0 trained, 3 canceled before the first step, 1 failed).
pub fn iris_rank_entry() -> Option<i32> {
    let spec_path = PathBuf::from(std::env::var_os(RANK_SPEC_ENV)?);
    // The rank's own diagnostics (stderr, inherited by the worker's log) use the worker's format.
    sceneworks_core::observability::init_logging();
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        let thread = std::thread::Builder::new()
            .name("iris-rank".to_owned())
            .stack_size(crate::WORKER_ENTRY_STACK_BYTES)
            .spawn(move || {
                let settings = crate::settings::Settings::from_env();
                let mut out = std::io::stdout();
                match (read_spec(&spec_path), RankWorld::from_env()) {
                    (Ok(spec), Ok(world)) => run_rank_process_with(
                        &spec,
                        &world,
                        &settings,
                        crate::inference_runtime::load_trainer,
                        std::io::stdin(),
                        &mut out,
                        true,
                    ),
                    (Err(error), _) | (_, Err(error)) => {
                        write_message(
                            &mut out,
                            &RankMessage::Failed {
                                message: error,
                                dist: None,
                                canceled: false,
                            },
                        );
                        1
                    }
                }
            });
        Some(match thread.map(|handle| handle.join()) {
            Ok(Ok(code)) => code,
            _ => 1,
        })
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        let _ = spec_path;
        write_message(
            &mut std::io::stdout(),
            &RankMessage::Failed {
                message: "this build links no training engine, so it cannot run an Iris rank"
                    .to_owned(),
                dist: None,
                canceled: false,
            },
        );
        Some(1)
    }
}

fn read_spec(path: &Path) -> Result<RankSpec, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read the rank spec {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid rank spec: {error}"))
}

/// Train one rank: the SAME weights-free preflight and trainer inputs as the single-process path,
/// the rank's world stamped onto the request, progress and the outcome written as protocol lines on
/// `out`, cancellation read from `input` (a `cancel` line, or EOF when the coordinator is gone).
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
pub(crate) fn run_rank_process_using(
    spec: &RankSpec,
    world: &RankWorld,
    settings: &crate::settings::Settings,
    load_trainer: impl FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Trainer>>,
    input: impl Read + Send + 'static,
    out: &mut dyn Write,
) -> i32 {
    run_rank_process_with(spec, world, settings, load_trainer, input, out, false)
}

/// [`run_rank_process_using`], optionally exiting the PROCESS when the coordinator disappears (the
/// real rank entry does; an in-process test must not).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn run_rank_process_with(
    spec: &RankSpec,
    world: &RankWorld,
    settings: &crate::settings::Settings,
    load_trainer: impl FnOnce(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Trainer>>,
    input: impl Read + Send + 'static,
    out: &mut dyn Write,
    exit_when_orphaned: bool,
) -> i32 {
    let fail = |out: &mut dyn Write, message: String| {
        write_message(
            out,
            &RankMessage::Failed {
                message,
                dist: None,
                canceled: false,
            },
        );
        1
    };
    let prepared = match crate::training_jobs::preflight_training_run(settings, &spec.plan, false) {
        Ok(prepared) => prepared,
        Err(error) => return fail(out, error.to_string()),
    };
    let crate::training_jobs::PreparedTrainingRun {
        engine_id,
        mut request,
        prepared_inputs,
    } = prepared;
    let finish = |request: gen_core::TrainingRequest, inputs| {
        drop(request);
        crate::training_jobs::finish_prepared_input_operation(Ok(()), inputs)
    };
    if let Err(error) = apply_rank_world(&mut request.config.model_options, world) {
        let _ = finish(request, prepared_inputs);
        return fail(out, format!("iris distributed: split world: {error}"));
    }
    let inputs =
        match crate::training_jobs::TrainerLoadInputs::resolve(settings, &spec.plan, engine_id) {
            Ok(inputs) => inputs,
            Err(error) => {
                let _ = finish(request, prepared_inputs);
                return fail(out, error.to_string());
            }
        };

    // Cancellation: a `cancel` line, or EOF (the coordinator is gone — stop, and do not outlive it).
    let cancel = request.cancel.clone();
    std::thread::Builder::new()
        .name("iris-rank-cancel".to_owned())
        .spawn(move || {
            for line in BufReader::new(input).lines() {
                match line {
                    Ok(line) if line.trim() == CANCEL_LINE => cancel.cancel(),
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            cancel.cancel();
            if exit_when_orphaned {
                std::thread::sleep(ORPHAN_EXIT_BOUND);
                std::process::exit(4);
            }
        })
        .ok();

    let spec_load = inputs.load_spec(engine_id, &request.config.train_dtype);
    let outcome = (|| {
        let mut trainer = load_trainer(engine_id, &spec_load)?;
        trainer.validate(&request)?;
        trainer.train(&request, &mut |progress| {
            write_message(
                out,
                &RankMessage::Progress {
                    progress: WireProgress::from_progress(&progress),
                },
            )
        })
    })();
    let _ = finish(request, prepared_inputs);
    match outcome {
        Ok(output) => {
            write_message(
                out,
                &RankMessage::Done {
                    adapter_path: output.adapter_path.display().to_string(),
                    steps: output.steps,
                    final_loss: output.final_loss,
                },
            );
            0
        }
        Err(error) => {
            let canceled = matches!(error, gen_core::Error::Canceled);
            let dist = gen_core_iris::train::dist::dist_error(&error).map(|typed| WireDistError {
                kind: crate::iris_curricula::dist_error_kind(typed).to_owned(),
                detail: typed.to_string(),
            });
            write_message(
                out,
                &RankMessage::Failed {
                    message: error.to_string(),
                    dist,
                    canceled,
                },
            );
            if canceled {
                3
            } else {
                1
            }
        }
    }
}

// =============================================================================================
// The coordinator
// =============================================================================================

/// How the coordinator starts a rank process: the command that re-enters an executable whose entry
/// point calls [`iris_rank_entry`] first. The fleet adds the rank's environment and pipes.
pub(crate) trait RankLauncher: Send + Sync {
    fn command(&self) -> Command;
}

/// The production launcher: this very executable.
pub(crate) struct CurrentExeLauncher;

impl RankLauncher for CurrentExeLauncher {
    fn command(&self) -> Command {
        Command::new(std::env::current_exe().unwrap_or_else(|_| PathBuf::from("sceneworks-worker")))
    }
}

struct RankProcess {
    rank: usize,
    child: Child,
    stdin: Option<ChildStdin>,
    eof: bool,
    status: Option<ExitStatus>,
    done: Option<RankMessage>,
    failure: Option<RankFailure>,
}

#[derive(Debug, Clone)]
struct RankFailure {
    rank: usize,
    kind: String,
    message: String,
    canceled: bool,
    /// The process ended without reporting (it died, or was killed by SceneWorks).
    synthetic: bool,
}

/// Every rank process of one job. Dropping it kills and reaps whatever is still running.
struct Fleet {
    ranks: Vec<RankProcess>,
}

impl Fleet {
    fn send_cancel(&mut self) {
        for rank in &mut self.ranks {
            if let Some(stdin) = rank.stdin.as_mut() {
                let _ = writeln!(stdin, "{CANCEL_LINE}");
                let _ = stdin.flush();
            }
        }
    }

    fn kill_running(&mut self) {
        for rank in &mut self.ranks {
            if rank.status.is_none() {
                let _ = rank.child.kill();
            }
        }
    }
}

impl Drop for Fleet {
    fn drop(&mut self) {
        for rank in &mut self.ranks {
            if rank.status.is_none() {
                let _ = rank.child.kill();
                let _ = rank.child.wait();
            }
        }
    }
}

enum FleetEvent {
    Message(usize, RankMessage),
    Eof(usize),
}

/// Tunables of a fleet drive (tests shorten the graces).
pub(crate) struct FleetTimings {
    /// How long the other ranks get once the fleet has failed or rank 0 finished.
    pub peer_stop_grace: Duration,
    /// How long the fleet gets to wind down after a cancel (rank 0 may still publish its durable
    /// checkpoint + export) before every remaining rank is killed.
    pub cancel_grace: Duration,
}

impl Default for FleetTimings {
    fn default() -> Self {
        Self {
            peer_stop_grace: PEER_STOP_GRACE,
            cancel_grace: CANCEL_GRACE,
        }
    }
}

fn free_loopback_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

/// Coordinate `world_size` rank processes of an Iris plan until they finish: rank 0's events go to
/// `tx`, `cancel` fans out, and the outcome is rank 0's export or a typed failure. Blocking.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn drive_rank_fleet(
    settings: &crate::settings::Settings,
    job_id: &str,
    plan: &TrainingPlan,
    launcher: &dyn RankLauncher,
    tx: &tokio::sync::mpsc::Sender<crate::training_jobs::TrainEvent>,
    cancel: &gen_core::CancelFlag,
) -> crate::error::WorkerResult<()> {
    drive_rank_fleet_with(
        settings,
        job_id,
        plan,
        launcher,
        tx,
        cancel,
        &FleetTimings::default(),
    )
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn drive_rank_fleet_with(
    settings: &crate::settings::Settings,
    job_id: &str,
    plan: &TrainingPlan,
    launcher: &dyn RankLauncher,
    tx: &tokio::sync::mpsc::Sender<crate::training_jobs::TrainEvent>,
    cancel: &gen_core::CancelFlag,
    timings: &FleetTimings,
) -> crate::error::WorkerResult<()> {
    use crate::error::WorkerError;
    use crate::training_jobs::TrainEvent;

    let world_size = fleet_world_size(plan);
    // The spec lives in a private (0700) directory under the data dir; it holds the plan only.
    let tmp_root = settings.data_dir.join("tmp");
    std::fs::create_dir_all(&tmp_root)?;
    let spec_dir = tempfile::Builder::new()
        .prefix("iris-ranks-")
        .tempdir_in(&tmp_root)?;
    let spec_path = spec_dir.path().join("rank-spec.json");
    std::fs::write(
        &spec_path,
        serde_json::to_vec(&RankSpec {
            job_id: job_id.to_owned(),
            plan: plan.clone(),
        })
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?,
    )?;
    let master_port = free_loopback_port()?;
    // The per-job rendezvous secret: in the ranks' environment only, never in a persisted file.
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );

    let (events, inbox) = std::sync::mpsc::channel::<FleetEvent>();
    let mut fleet = Fleet { ranks: Vec::new() };
    for rank in 0..world_size {
        let world = RankWorld {
            rank,
            world_size,
            master_addr: "127.0.0.1".to_owned(),
            master_port,
        };
        let mut command = launcher.command();
        command
            .envs(world.env())
            .env(RANK_SPEC_ENV, &spec_path)
            .env(gen_core_iris::train::dist::DIST_TOKEN_ENV, &token)
            // The rank resolves the plan's paths against THIS worker's settings, whatever the
            // ambient environment says (a test worker, an embedded worker with defaulted dirs).
            .env("SCENEWORKS_DATA_DIR", &settings.data_dir)
            .env("SCENEWORKS_CONFIG_DIR", &settings.config_dir)
            .env("SCENEWORKS_GPU_ID", &settings.gpu_id)
            .env(
                "SCENEWORKS_BACKEND_MLX_ENABLED",
                if settings.backend_mlx_enabled {
                    "1"
                } else {
                    "0"
                },
            )
            .env(
                "SCENEWORKS_BACKEND_CANDLE_ENABLED",
                if settings.backend_candle_enabled {
                    "1"
                } else {
                    "0"
                },
            )
            // A rank never talks to the API.
            .env_remove("SCENEWORKS_ACCESS_TOKEN")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                // `fleet` drops here: every rank already started is killed and reaped.
                return Err(WorkerError::Engine(format!(
                    "Iris distributed training could not start rank {rank} of {world_size}: \
                     {error}"
                )));
            }
        };
        let stdout = child.stdout.take().expect("piped stdout");
        let stdin = child.stdin.take();
        let sender = events.clone();
        std::thread::Builder::new()
            .name(format!("iris-rank-{rank}-reader"))
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    match parse_message(&line) {
                        Some(message) => {
                            if sender.send(FleetEvent::Message(rank, message)).is_err() {
                                return;
                            }
                        }
                        None => tracing::info!(rank, "iris rank output: {line}"),
                    }
                }
                let _ = sender.send(FleetEvent::Eof(rank));
            })?;
        fleet.ranks.push(RankProcess {
            rank,
            child,
            stdin,
            eof: false,
            status: None,
            done: None,
            failure: None,
        });
        tracing::info!(
            event = "iris_rank_started",
            job_id,
            rank,
            world_size,
            master_port,
            "started Iris rank process"
        );
    }
    drop(events);

    let mut cancel_sent = false;
    let mut stop_deadline: Option<Instant> = None;
    let mut killed = false;
    let mut consumer_gone = false;
    loop {
        if cancel.is_cancelled() && !cancel_sent {
            fleet.send_cancel();
            cancel_sent = true;
            // The trainers agree on a cancel only at a collective; a rank still in the rendezvous
            // (`World::connect` retries until `timeout_secs`, 90 min by default) never checks it.
            // Bound the wind-down: long enough for rank 0's durable checkpoint + export, never the
            // transport timeout.
            stop_deadline.get_or_insert(Instant::now() + timings.cancel_grace);
        }
        if fleet.ranks.iter().all(|rank| rank.status.is_some()) {
            break;
        }
        if let Some(deadline) = stop_deadline {
            if Instant::now() >= deadline && !killed {
                fleet.kill_running();
                killed = true;
            }
        }
        match inbox.recv_timeout(Duration::from_millis(100)) {
            Ok(FleetEvent::Message(rank, message)) => match message {
                RankMessage::Progress { progress } => {
                    if rank == 0 && !consumer_gone {
                        if let Some(progress) = progress.into_progress() {
                            if tx.blocking_send(TrainEvent::Progress(progress)).is_err() {
                                // Nobody listens to this job any more: stop every rank.
                                consumer_gone = true;
                                cancel.cancel();
                            }
                        }
                    }
                }
                done @ RankMessage::Done { .. } => {
                    fleet.ranks[rank].done = Some(done);
                    if rank == 0 {
                        stop_deadline.get_or_insert(Instant::now() + timings.peer_stop_grace);
                    }
                }
                RankMessage::Failed {
                    message,
                    dist,
                    canceled,
                } => {
                    let kind = dist
                        .as_ref()
                        .map(|dist| dist.kind.clone())
                        .unwrap_or_else(|| {
                            if canceled {
                                "canceled".to_owned()
                            } else {
                                "error".to_owned()
                            }
                        });
                    fleet.ranks[rank].failure = Some(RankFailure {
                        rank,
                        kind,
                        message,
                        canceled,
                        synthetic: false,
                    });
                    if !cancel_sent {
                        fleet.send_cancel();
                        cancel_sent = true;
                    }
                    // A canceled rank winds down within the cancel grace; any other failure stops
                    // the fleet within the peer grace.
                    stop_deadline.get_or_insert(
                        Instant::now()
                            + if canceled {
                                timings.cancel_grace
                            } else {
                                timings.peer_stop_grace
                            },
                    );
                }
            },
            Ok(FleetEvent::Eof(rank)) => fleet.ranks[rank].eof = true,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            // Every reader thread ended: only the reaping below remains — do not spin.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                std::thread::sleep(Duration::from_millis(100))
            }
        }
        // Reap exited ranks; one that ended without reporting is a lost rank.
        for index in 0..fleet.ranks.len() {
            let rank = &mut fleet.ranks[index];
            if rank.status.is_some() || !rank.eof {
                continue;
            }
            if let Ok(Some(status)) = rank.child.try_wait() {
                rank.status = Some(status);
                rank.stdin = None;
                if rank.done.is_none() && rank.failure.is_none() {
                    rank.failure = Some(RankFailure {
                        rank: rank.rank,
                        kind: "rank_lost".to_owned(),
                        message: if killed {
                            format!(
                                "rank {} did not stop within {}s of the fleet's outcome and was \
                                 killed",
                                rank.rank,
                                timings.peer_stop_grace.as_secs()
                            )
                        } else {
                            format!(
                                "rank {} exited ({status}) without reporting a result",
                                rank.rank
                            )
                        },
                        canceled: false,
                        synthetic: true,
                    });
                    if !killed {
                        if !cancel_sent {
                            fleet.send_cancel();
                            cancel_sent = true;
                        }
                        stop_deadline.get_or_insert(Instant::now() + timings.peer_stop_grace);
                    }
                }
            }
        }
    }
    let ranks = std::mem::take(&mut fleet.ranks);
    drop(fleet);
    drop(spec_dir);

    tracing::info!(
        event = "iris_rank_fleet_finished",
        job_id,
        world_size,
        "every Iris rank process exited"
    );
    // Rank 0's export is the run's outcome (the trainers agree collectively, so the other ranks
    // finish the same steps).
    if let Some(RankMessage::Done {
        adapter_path,
        steps,
        final_loss,
    }) = ranks.first().and_then(|rank| rank.done.clone())
    {
        let _ = tx.blocking_send(TrainEvent::Done(gen_core::TrainingOutput {
            adapter_path: PathBuf::from(adapter_path),
            steps,
            final_loss,
        }));
        return Ok(());
    }
    let failures: Vec<RankFailure> = ranks.iter().filter_map(|r| r.failure.clone()).collect();
    if cancel.is_cancelled() && failures.iter().all(|f| f.canceled || f.synthetic) {
        return Err(WorkerError::Engine(
            "training failed: the run was canceled before its first optimizer step".to_owned(),
        ));
    }
    Err(WorkerError::Engine(fleet_failure_message(
        &failures, world_size,
    )))
}

/// The typed failure of a fleet: the root cause first — a rank's own refusal or error, then a rank
/// that died, then a survivor's `rank_lost` report.
fn fleet_failure_message(failures: &[RankFailure], world_size: usize) -> String {
    let rank_of = |failure: &RankFailure| match failure.kind.as_str() {
        "rank_lost" | "canceled" if !failure.synthetic => 2,
        _ if failure.synthetic => 1,
        _ => 0,
    };
    let root = failures
        .iter()
        .filter(|failure| !failure.canceled)
        .min_by_key(|failure| (rank_of(failure), failure.rank));
    match root {
        Some(root) => format!(
            "Iris distributed training failed [{}] on rank {} of {world_size}: {}. Every rank \
             process was stopped.",
            root.kind, root.rank, root.message
        ),
        None => format!(
            "Iris distributed training failed: the {world_size} rank processes stopped without \
             a result."
        ),
    }
}

/// Fake rank processes for the fleet tests: the test binary itself, re-entered on the ignored
/// [`tests::fake_iris_rank_process`] test, which plays one rank according to
/// `SCENEWORKS_FAKE_RANK_BEHAVIOR` and records its environment under `SCENEWORKS_FAKE_RANK_RECORD`.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
pub(crate) mod test_support {
    use super::*;

    pub(crate) const BEHAVIOR_ENV: &str = "SCENEWORKS_FAKE_RANK_BEHAVIOR";
    pub(crate) const RECORD_ENV: &str = "SCENEWORKS_FAKE_RANK_RECORD";

    pub(crate) struct FakeRankLauncher {
        pub behavior: &'static str,
        pub record_dir: PathBuf,
    }

    impl RankLauncher for FakeRankLauncher {
        fn command(&self) -> Command {
            let mut command = Command::new(std::env::current_exe().expect("the test binary"));
            command
                .args([
                    "iris_dist::tests::fake_iris_rank_process",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(BEHAVIOR_ENV, self.behavior)
                .env(RECORD_ENV, &self.record_dir);
            command
        }
    }

    /// Real rank processes for the real-weight smokes: this test binary re-entered on the ignored
    /// [`tests::real_iris_rank_process`], which runs the PRODUCTION rank entry.
    pub(crate) struct TestBinaryRankLauncher;

    impl RankLauncher for TestBinaryRankLauncher {
        fn command(&self) -> Command {
            let mut command = Command::new(std::env::current_exe().expect("the test binary"));
            command.args([
                "iris_dist::tests::real_iris_rank_process",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]);
            command
        }
    }

    /// The env var naming the directory a REAL rank process records itself in (`rank-<r>.json`:
    /// its rank and its own process id) — how a smoke learns which process was which rank from the
    /// ranks themselves rather than from the plan it handed them.
    pub(crate) const REAL_RANK_RECORD_ENV: &str = "SCENEWORKS_IRIS_REAL_RANK_RECORD";

    /// [`TestBinaryRankLauncher`] whose ranks record themselves under `record_dir`.
    pub(crate) struct RecordingRankLauncher {
        pub record_dir: PathBuf,
    }

    impl RankLauncher for RecordingRankLauncher {
        fn command(&self) -> Command {
            let mut command = TestBinaryRankLauncher.command();
            command.env(REAL_RANK_RECORD_ENV, &self.record_dir);
            command
        }
    }

    /// Record this process as the rank its environment names (no-op without the record dir).
    pub(crate) fn record_real_rank() {
        let (Some(dir), Ok(rank)) = (
            std::env::var_os(REAL_RANK_RECORD_ENV),
            std::env::var("RANK"),
        ) else {
            return;
        };
        let Ok(rank) = rank.trim().parse::<u64>() else {
            return;
        };
        let record = json!({ "rank": rank, "pid": std::process::id() });
        let _ = std::fs::write(
            PathBuf::from(dir).join(format!("rank-{rank}.json")),
            record.to_string(),
        );
    }

    /// The records the fake ranks wrote (`rank-<r>.json`), by rank.
    pub(crate) fn records(dir: &Path) -> Vec<Value> {
        let mut records: Vec<Value> = std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with("rank-"))
                    .filter_map(|entry| std::fs::read(entry.path()).ok())
                    .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
                    .collect()
            })
            .unwrap_or_default();
        records.sort_by_key(|record: &Value| record["rank"].as_u64());
        records
    }

    /// Whether a recorded rank process is still alive (no orphans may survive a drive).
    pub(crate) fn alive(pid: u64) -> bool {
        #[cfg(unix)]
        {
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            false
        }
    }
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod tests {
    use super::*;

    #[test]
    fn the_wire_round_trips_every_progress_event_and_ignores_noise() {
        use gen_core::TrainingProgress as P;
        for progress in [
            P::Preparing,
            P::LoadingModel,
            P::Caching {
                current: 1,
                total: 4,
            },
            P::Training {
                step: 3,
                total: 9,
                loss: 0.25,
            },
            P::Checkpoint { step: 3 },
            P::Sample {
                step: 3,
                index: 1,
                total: 2,
                prompt: "a fox".to_owned(),
                image: gen_core::Image {
                    width: 2,
                    height: 1,
                    pixels: vec![1, 2, 3, 4, 5, 6],
                },
            },
            P::Saving,
        ] {
            let mut out = Vec::new();
            write_message(
                &mut out,
                &RankMessage::Progress {
                    progress: WireProgress::from_progress(&progress),
                },
            );
            let line = String::from_utf8(out).unwrap();
            let Some(RankMessage::Progress { progress: wire }) = parse_message(&line) else {
                panic!("{line}")
            };
            assert_eq!(wire.into_progress(), Some(progress));
        }
        assert!(parse_message("iris training: epoch 1 | step 1").is_none());
        assert!(parse_message(&format!("{RANK_LINE_MARKER}not json")).is_none());
    }

    #[test]
    fn the_rank_world_is_stamped_without_touching_the_client_choices() {
        let mut options = Map::new();
        options.insert(
            "iris".to_owned(),
            json!({"dist": {"strategy": "ddp", "world": {"world_size": 2, "share_device": true,
                    "timeout_secs": 60}}}),
        );
        let world = RankWorld {
            rank: 1,
            world_size: 2,
            master_addr: "127.0.0.1".to_owned(),
            master_port: 40123,
        };
        apply_rank_world(&mut options, &world).unwrap();
        let stamped = &options["iris"]["dist"]["world"];
        assert_eq!(stamped["rank"], json!(1));
        assert_eq!(stamped["local_rank"], json!(1));
        assert_eq!(stamped["local_world_size"], json!(2));
        assert_eq!(stamped["num_nodes"], json!(1));
        assert_eq!(stamped["master_port"], json!(40123));
        assert_eq!(stamped["transport"], json!("tcp"));
        assert_eq!(stamped["share_device"], json!(true));
        assert_eq!(stamped["timeout_secs"], json!(60));
        assert_eq!(options["iris"]["dist"]["strategy"], json!("ddp"));
        // The engine's own world parser accepts what the launcher stamped.
        let config = gen_core::TrainingConfig {
            model_options: options.clone(),
            ..Default::default()
        };
        let plan = gen_core_iris::train::dist::parse_dist(
            config.model_options["iris"].get("dist"),
            false,
            true,
        )
        .expect("the engine parses the stamped world");
        assert_eq!((plan.world.rank, plan.world.world_size), (1, 2));
        assert!(plan.world.share_device);
        // A launcher world that disagrees with the plan is refused.
        let wrong = RankWorld {
            world_size: 3,
            ..world.clone()
        };
        assert!(apply_rank_world(&mut options, &wrong).is_err());
        let env: Map<String, Value> = world
            .env()
            .into_iter()
            .map(|(key, value)| (key.to_owned(), json!(value)))
            .collect();
        assert_eq!(env["RANK"], json!("1"));
        assert_eq!(env["WORLD_SIZE"], json!("2"));
        assert_eq!(env["MASTER_PORT"], json!("40123"));
    }

    #[test]
    fn the_failure_names_the_root_cause_before_a_survivors_rank_lost() {
        let failure = |rank, kind: &str, synthetic| RankFailure {
            rank,
            kind: kind.to_owned(),
            message: format!("{kind} on {rank}"),
            canceled: false,
            synthetic,
        };
        let message = fleet_failure_message(
            &[
                failure(0, "rank_lost", false),
                failure(1, "rank_lost", true),
            ],
            2,
        );
        assert!(message.contains("on rank 1 of 2"), "{message}");
        let message = fleet_failure_message(
            &[
                failure(0, "rank_lost", false),
                failure(1, "invalid_mesh", false),
            ],
            2,
        );
        assert!(message.contains("[invalid_mesh] on rank 1"), "{message}");
    }

    #[test]
    fn devices_count_the_workers_own_gpus() {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.gpu_id = "0".to_owned();
        assert_eq!(visible_training_devices(&settings), 1);
        settings.gpu_id = "0,1".to_owned();
        let expected = if cfg!(target_os = "macos") { 1 } else { 2 };
        assert_eq!(visible_training_devices(&settings), expected);
    }

    // ---- fake rank processes -----------------------------------------------------------------

    /// One fake rank (see [`super::test_support`]). Exits through `std::process::exit` with the
    /// rank's code; when not launched as a rank it does nothing.
    #[test]
    #[ignore = "a fake Iris rank process, launched by the rank-fleet tests"]
    fn fake_iris_rank_process() {
        use super::test_support::{BEHAVIOR_ENV, RECORD_ENV};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        if std::env::var_os(RANK_SPEC_ENV).is_none() {
            return;
        }
        let world = RankWorld::from_env().expect("the launcher set the world");
        let behavior = std::env::var(BEHAVIOR_ENV).unwrap_or_default();
        let record_dir = PathBuf::from(std::env::var_os(RECORD_ENV).expect("a record dir"));
        let spec = read_spec(Path::new(&std::env::var_os(RANK_SPEC_ENV).unwrap()));
        let token = std::env::var(gen_core_iris::train::dist::DIST_TOKEN_ENV).ok();
        std::fs::write(
            record_dir.join(format!("rank-{}.json", world.rank)),
            serde_json::to_vec(&json!({
                "pid": std::process::id(),
                "rank": world.rank,
                "worldSize": world.world_size,
                "localRank": std::env::var("LOCAL_RANK").ok(),
                "localWorldSize": std::env::var("LOCAL_WORLD_SIZE").ok(),
                "masterAddr": world.master_addr,
                "masterPort": world.master_port,
                "token": token,
                "accessToken": std::env::var("SCENEWORKS_ACCESS_TOKEN").ok(),
                "dataDir": std::env::var("SCENEWORKS_DATA_DIR").ok(),
                "specJob": spec.as_ref().ok().map(|spec| spec.job_id.clone()),
                "specKernel": spec.as_ref().ok().map(|spec| spec.plan.target.kernel.clone()),
            }))
            .unwrap(),
        )
        .unwrap();
        let canceled = Arc::new(AtomicBool::new(false));
        {
            let canceled = Arc::clone(&canceled);
            std::thread::spawn(move || {
                for line in BufReader::new(std::io::stdin()).lines() {
                    match line {
                        Ok(line) if line.trim() == CANCEL_LINE => {
                            canceled.store(true, Ordering::SeqCst)
                        }
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            });
        }
        let wait_cancel = |bound: Duration| {
            let deadline = Instant::now() + bound;
            while !canceled.load(Ordering::SeqCst) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            canceled.load(Ordering::SeqCst)
        };
        let mut out = std::io::stdout();
        let progress = |out: &mut dyn Write, progress: gen_core::TrainingProgress| {
            write_message(
                out,
                &RankMessage::Progress {
                    progress: WireProgress::from_progress(&progress),
                },
            )
        };
        let done = |out: &mut dyn Write, steps: u32| {
            write_message(
                out,
                &RankMessage::Done {
                    adapter_path: record_dir.join("adapter.safetensors").display().to_string(),
                    steps,
                    final_loss: 0.125,
                },
            )
        };
        let code = match (behavior.as_str(), world.rank) {
            ("complete", 0) => {
                progress(&mut out, gen_core::TrainingProgress::Preparing);
                progress(
                    &mut out,
                    gen_core::TrainingProgress::Training {
                        step: 4,
                        total: 4,
                        loss: 0.125,
                    },
                );
                progress(&mut out, gen_core::TrainingProgress::Checkpoint { step: 4 });
                progress(&mut out, gen_core::TrainingProgress::Saving);
                done(&mut out, 4);
                0
            }
            ("complete", _) => {
                // A non-main rank's progress is never forwarded.
                progress(
                    &mut out,
                    gen_core::TrainingProgress::Training {
                        step: 99,
                        total: 99,
                        loss: 9.0,
                    },
                );
                done(&mut out, 4);
                0
            }
            ("wait_cancel", rank) => {
                // Keep stepping until the cancel arrives, then finish durably (rank 0 exports).
                let mut step = 1;
                while !canceled.load(Ordering::SeqCst) && step < 600 {
                    if rank == 0 {
                        progress(
                            &mut out,
                            gen_core::TrainingProgress::Training {
                                step,
                                total: 1000,
                                loss: 0.5,
                            },
                        );
                    }
                    step += 1;
                    std::thread::sleep(Duration::from_millis(100));
                }
                if canceled.load(Ordering::SeqCst) {
                    std::fs::write(record_dir.join(format!("cancel-{rank}")), b"1").unwrap();
                }
                done(&mut out, step);
                0
            }
            ("rank1_dies", 1) => 101,
            ("rank1_dies", _) => {
                progress(&mut out, gen_core::TrainingProgress::Preparing);
                // The survivor's transport notices the lost peer (here: SceneWorks' cancel).
                assert!(
                    wait_cancel(Duration::from_secs(60)),
                    "the coordinator cancels"
                );
                write_message(
                    &mut out,
                    &RankMessage::Failed {
                        message: "iris distributed: rank 1 lost during all_reduce (closed)"
                            .to_owned(),
                        dist: Some(WireDistError {
                            kind: "rank_lost".to_owned(),
                            detail: "rank 1 lost during all_reduce".to_owned(),
                        }),
                        canceled: false,
                    },
                );
                1
            }
            ("refused_then_hang", 0) => {
                write_message(
                    &mut out,
                    &RankMessage::Failed {
                        message: "iris distributed: invalid mesh: dp_shard=3 does not tile"
                            .to_owned(),
                        dist: Some(WireDistError {
                            kind: "invalid_mesh".to_owned(),
                            detail: "dp_shard=3 does not tile".to_owned(),
                        }),
                        canceled: false,
                    },
                );
                1
            }
            ("ignore_cancel", _) => {
                // Stuck before any collective (e.g. waiting in the rendezvous): never answers a
                // cancel. Only the fleet's bounded wind-down ends it.
                std::thread::sleep(Duration::from_secs(300));
                0
            }
            ("refused_then_hang", _) => {
                // Ignores the cancel: only a kill ends it.
                std::thread::sleep(Duration::from_secs(300));
                0
            }
            (other, _) => panic!("unknown fake rank behavior {other:?}"),
        };
        std::process::exit(code);
    }

    /// One REAL rank (see [`super::test_support::TestBinaryRankLauncher`]): the production entry,
    /// in a process of its own. Does nothing when not launched as a rank.
    #[test]
    #[ignore = "a real Iris rank process, launched by the real-weight training smokes"]
    fn real_iris_rank_process() {
        super::test_support::record_real_rank();
        if let Some(code) = iris_rank_entry() {
            std::process::exit(code);
        }
    }

    fn fleet_plan(data: &Path, iris: Value) -> TrainingPlan {
        use sceneworks_core::training::{build_training_plan, BuildTrainingPlan, TrainingDataset};
        let target = sceneworks_core::training::iris::iris_generation_target();
        let mut config = target.defaults.clone();
        config.advanced.insert("iris".to_owned(), iris);
        let root = data.join("datasets/ds1");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::write(root.join("images/a.png"), b"png").unwrap();
        let dataset: TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds1", "version": 1, "name": "ds", "modality": "image",
            "status": "ready", "createdAt": "x", "updatedAt": "x",
            "items": [{"id": "a", "path": "images/a.png", "displayName": "a", "addedAt": "x",
                       "caption": {"text": "a cat", "source": "manual", "triggerWords": []}}]
        }))
        .unwrap();
        build_training_plan(BuildTrainingPlan {
            job_id: "job-fleet",
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "lora_fleet",
            base_model_path: data.join("backbone").display().to_string(),
            dataset_root: &root,
            output_dir: &data.join("loras/lora_fleet"),
            file_name: "fleet.safetensors".to_owned(),
            created_at: "x".to_owned(),
        })
        .expect("the plan builds")
    }

    struct Drive {
        outcome: crate::error::WorkerResult<()>,
        events: Vec<crate::training_jobs::TrainEvent>,
        records: Vec<Value>,
        /// The ranks that recorded receiving the cancel line.
        canceled_ranks: Vec<usize>,
        elapsed: Duration,
    }

    fn drive(behavior: &'static str, world: u64, cancel_after: Option<Duration>) -> Drive {
        let data = tempfile::tempdir().unwrap();
        let records = tempfile::tempdir().unwrap();
        let plan = fleet_plan(
            data.path(),
            json!({"dist": {"strategy": "ddp", "world": {"world_size": world, "share_device": true}}}),
        );
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data.path().to_path_buf();
        let launcher = super::test_support::FakeRankLauncher {
            behavior,
            record_dir: records.path().to_path_buf(),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let cancel = gen_core::CancelFlag::new();
        if let Some(after) = cancel_after {
            let cancel = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(after);
                cancel.cancel();
            });
        }
        let started = Instant::now();
        let outcome = drive_rank_fleet_with(
            &settings,
            "job-fleet",
            &plan,
            &launcher,
            &tx,
            &cancel,
            &FleetTimings {
                peer_stop_grace: Duration::from_secs(2),
                cancel_grace: Duration::from_secs(3),
            },
        );
        let elapsed = started.elapsed();
        drop(tx);
        let mut events = Vec::new();
        while let Some(event) = rx.blocking_recv() {
            events.push(event);
        }
        let canceled_ranks = (0..world as usize)
            .filter(|rank| records.path().join(format!("cancel-{rank}")).is_file())
            .collect();
        let records = super::test_support::records(records.path());
        for record in &records {
            assert!(
                !super::test_support::alive(record["pid"].as_u64().unwrap()),
                "rank {} outlived the drive",
                record["rank"]
            );
        }
        assert!(
            std::fs::read_dir(data.path().join("tmp"))
                .map(|entries| entries.count())
                .unwrap_or(0)
                == 0,
            "the private rank spec dir is removed"
        );
        Drive {
            outcome,
            events,
            records,
            canceled_ranks,
            elapsed,
        }
    }

    #[test]
    fn every_rank_gets_its_world_and_rank_zero_drives_the_job() {
        use crate::training_jobs::TrainEvent;
        let run = drive("complete", 3, None);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(run.records.len(), 3, "{:#?}", run.records);
        let port = run.records[0]["masterPort"].clone();
        let token = run.records[0]["token"].clone();
        assert!(token.as_str().is_some_and(|token| token.len() == 64));
        for (rank, record) in run.records.iter().enumerate() {
            assert_eq!(record["rank"], json!(rank));
            assert_eq!(record["worldSize"], json!(3));
            assert_eq!(record["localRank"], json!(rank.to_string()));
            assert_eq!(record["localWorldSize"], json!("3"));
            assert_eq!(record["masterAddr"], json!("127.0.0.1"));
            assert_eq!(record["masterPort"], port, "one rendezvous");
            assert_eq!(
                record["token"], token,
                "one per-job secret, in the environment"
            );
            assert_eq!(
                record["accessToken"],
                Value::Null,
                "ranks never get the API token"
            );
            assert_eq!(record["specJob"], json!("job-fleet"));
            assert_eq!(record["specKernel"], json!("iris_train"));
            assert!(
                record["dataDir"]
                    .as_str()
                    .is_some_and(|dir| !dir.is_empty()),
                "the rank gets the worker's data dir"
            );
        }
        // Rank 0's events only (the other ranks' progress is never forwarded), then the export.
        let trained: Vec<u32> = run
            .events
            .iter()
            .filter_map(|event| match event {
                TrainEvent::Progress(gen_core::TrainingProgress::Training { step, .. }) => {
                    Some(*step)
                }
                _ => None,
            })
            .collect();
        assert_eq!(trained, vec![4]);
        match run.events.last() {
            Some(TrainEvent::Done(output)) => {
                assert_eq!(output.steps, 4);
                assert!(output.adapter_path.ends_with("adapter.safetensors"));
            }
            _ => panic!("the run ends with rank 0's export"),
        }
    }

    #[test]
    fn a_cancel_reaches_every_rank_and_rank_zero_finishes_durably() {
        use crate::training_jobs::TrainEvent;
        let run = drive("wait_cancel", 2, Some(Duration::from_millis(800)));
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert!(matches!(run.events.last(), Some(TrainEvent::Done(_))));
        assert_eq!(
            run.canceled_ranks,
            vec![0, 1],
            "every rank received the cancel"
        );
        assert!(run.elapsed < Duration::from_secs(50), "{:?}", run.elapsed);
    }

    /// A cancel bounds the wind-down: ranks that never answer it (stuck in the rendezvous, which
    /// retries for the 90-minute transport timeout) are killed after the cancel grace and reaped.
    #[test]
    fn a_cancel_never_waits_out_the_transport_timeout() {
        let run = drive("ignore_cancel", 2, Some(Duration::from_millis(500)));
        assert!(run.outcome.is_err(), "{:?}", run.outcome);
        assert_eq!(run.records.len(), 2);
        // 0.5 s + the 3 s cancel grace + reaping; the ranks themselves would sleep 300 s.
        assert!(run.elapsed < Duration::from_secs(30), "{:?}", run.elapsed);
    }

    #[test]
    fn a_rank_that_dies_fails_the_job_typed_and_the_survivor_is_stopped() {
        let run = drive("rank1_dies", 2, None);
        let message = match run.outcome {
            Err(crate::error::WorkerError::Engine(message)) => message,
            other => panic!("a lost rank must fail the job, got {other:?}"),
        };
        assert!(message.contains("[rank_lost] on rank 1 of 2"), "{message}");
        assert!(message.contains("without reporting"), "{message}");
    }

    #[test]
    fn a_refusal_names_its_kind_and_a_rank_ignoring_the_stop_is_killed() {
        let run = drive("refused_then_hang", 2, None);
        let message = match run.outcome {
            Err(crate::error::WorkerError::Engine(message)) => message,
            other => panic!("the refusal must fail the job, got {other:?}"),
        };
        assert!(
            message.contains("[invalid_mesh] on rank 0 of 2"),
            "{message}"
        );
        // Bounded: the hanging rank was killed after the 2 s grace, not waited out (300 s).
        assert!(run.elapsed < Duration::from_secs(60), "{:?}", run.elapsed);
    }
}
