//! `film-harness` — plan, compile and render a production plan into a SceneWorks sequence through a
//! running SceneWorks API (epic 22708, sc-22710, sc-22711, sc-22713, sc-22714).
//!
//! ```text
//! film-harness plan         --brief BRIEF.json --references REFERENCES.json --out DIR [--api URL]
//! film-harness compile      --plan PLAN.json --references REFERENCES.json --out DIR [--api URL]
//! film-harness validate     --plan PLAN.json --references REFERENCES.json [--api URL]
//! film-harness run          --plan PLAN.json --references REFERENCES.json [--api URL] [--shots SH010,SH020]
//!                           [--project-id ID] [--out DIR] [--poll-seconds N] [--no-export]
//!                           [--skip-install-check]
//! film-harness resume       --out DIR [--api URL] [--poll-seconds N] [--no-export]
//! film-harness replace-take --out DIR --shot SH030 [--reason TEXT] [--export]
//! film-harness review       --out DIR [--shots SH010,SH020] [--review-plan FILE] [--api URL]
//! film-harness accept-take  --out DIR --shot SH030 [--reason TEXT]
//! film-harness reject-take  --out DIR --shot SH030 --reason TEXT
//! film-harness request-repair --out DIR --shot SH030 [--reason TEXT] [--export]
//! film-harness review-eval  --set LABELS.jsonc [--out DIR] [--media-root DIR] [--api URL]
//! film-harness review-fixtures --set LABELS.jsonc [--media-root DIR]
//! film-harness cancel       --out DIR
//! film-harness status       --out DIR
//! film-harness trim         --run RUN.json --shot SH010 [--source-in S] [--source-out S]
//! film-harness reorder      --run RUN.json --order SH020,SH010
//! film-harness swap-take    --run RUN.json --shot SH010 --asset asset_...
//! film-harness fixture-images --out DIR
//! film-harness fixture-sound  --out DIR
//! ```
//!
//! The intended loop is `plan` -> read and edit `plan.json` -> `compile` -> `validate` -> `run`.
//! `plan` drives the LOCAL LLM through the shipped `prompt_refine` seam; it writes the plan and the
//! compiled per-shot requests as two versioned documents and touches nothing else. A hand-authored
//! plan skips straight to `validate`/`run` exactly as before.
//!
//! The review commands (sc-22714) read a rendered take back through the EXISTING understanding
//! seams — the `frame_extract` job for timestamped frame evidence, then the `image_vqa` job
//! (SenseNova-U1-8B) for one declared question per frame — and write an observed-state document
//! per take under `<out>/reviews/`. They add no model and no job type. **Review is assistive, not
//! quality assurance**: it flags things for a person, and only `accept-take`, `reject-take` and
//! `request-repair` change anything.
//!
//! `run` needs a SceneWorks API with a registered GPU worker (`video_generate`) and a utility
//! worker (`timeline_export`, e.g. `SCENEWORKS_RUN_UTILITY_INPROCESS=1`). It creates nothing until
//! the plan, the reference pack, the model's catalog entry and the host all validate; the run
//! record (`run.json`) is written under `--out` on every path, including refusal and a
//! transport/API failure partway through. Exit codes: 0 on a completed run, 2 when the plan was
//! refused before dispatch, 3 when the run stopped on a limit or a shot failed, 1 on a
//! transport/API/io error.
//!
//! Ctrl-C cancels the in-flight job through the API, stops dispatching and writes the record,
//! rather than orphaning a render. `film-harness cancel --out DIR` does the same from another
//! shell. A cancel is RESUMABLE: `film-harness resume --out DIR` picks the run back up, reusing
//! every take that finished and adopting every job still in flight (sc-22711).
//!
//! One controller per run directory: nothing locks `run.json`, so `run`, `resume` and
//! `replace-take` must not be held against the same `--out` at the same time.
//!
//! Two different things can change which take a shot carries, and they are SEPARATE verbs:
//!
//! * `replace-take` (sc-22711) is the GENERATION side — it rejects the take a shot is carrying and
//!   renders exactly one more for it, under the run's remaining budget.
//! * `trim` / `reorder` / `swap-take` (sc-22712) are the EDIT side — they change an assembled
//!   sequence without re-rendering anything: they read the run record, edit the SAVED timeline
//!   through the same API the editor uses, re-lay the sequence, and write the record back.
//!   `swap-take` swaps in an asset that already exists. `--export` re-runs the MP4 export.
//! * `request-repair` (sc-22714) is the GENERATION side too: it is `replace-take` with a review's
//!   mismatch flags folded into the recorded reason, so it renders exactly one more take. It is
//!   NOT `swap-take` — a reviewer's reading is never a reason to point an item at a take the run
//!   already has.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sceneworks_core::film_plan::{RunOutcome, RunRecord, RunState};
use sceneworks_core::film_review::{format_eval_report, ASSISTIVE_NOTICE};
use sceneworks_rust_api::film_harness::review::{
    self, Decision, EvalOptions, ReviewOptions, ScriptedVision, VqaVision,
};
use sceneworks_rust_api::film_harness::{
    self, ApiTransport, EditOptions, HarnessError, HttpTransport, ResumeOptions, RunControl,
    RunOptions, TimelineEdit, FIXTURE_REFERENCES, FIXTURE_SOUNDS,
};
use sceneworks_rust_api::film_planner::{
    self, PlannerOptions, SceneWorksLlm, DEFAULT_LLM_JOB_TIMEOUT, DEFAULT_MAX_REPAIR_ROUNDS,
};

const USAGE: &str = "\
film-harness — plan, compile and render a production plan into a SceneWorks sequence

USAGE:
  film-harness plan         --brief BRIEF.json --references REFERENCES.json --out DIR [OPTIONS]
  film-harness compile      --plan PLAN.json --references REFERENCES.json --out DIR [OPTIONS]
  film-harness validate     --plan PLAN.json --references REFERENCES.json [--api URL] [--shots IDS]
  film-harness run          --plan PLAN.json --references REFERENCES.json [OPTIONS]
  film-harness resume       --out DIR [--api URL] [--poll-seconds N] [--no-export]
  film-harness replace-take --out DIR --shot SHxxx [--reason TEXT] [--export] [--api URL]
  film-harness review       --out DIR [--shots IDS] [--review-plan FILE] [--api URL] [--poll-seconds N]
  film-harness accept-take  --out DIR --shot SHxxx [--reason TEXT]
  film-harness reject-take  --out DIR --shot SHxxx --reason TEXT
  film-harness request-repair --out DIR --shot SHxxx [--reason TEXT] [--export] [--api URL]
  film-harness review-eval  --set LABELS.jsonc [--out DIR] [--media-root DIR] [--project-id ID] [--api URL]
  film-harness review-fixtures --set LABELS.jsonc [--media-root DIR]
  film-harness cancel       --out DIR
  film-harness status       --out DIR
  film-harness trim         --run RUN.json --shot ID [--source-in S] [--source-out S] [EDIT OPTIONS]
  film-harness reorder      --run RUN.json --order A,B,C                              [EDIT OPTIONS]
  film-harness swap-take    --run RUN.json --shot ID --asset ASSET_ID                 [EDIT OPTIONS]
  film-harness fixture-images --out DIR
  film-harness fixture-sound  --out DIR

OPTIONS (plan / compile):
  --brief BRIEF.json     The brief to plan from (plan); re-checked for dropped beats (compile)
  --out DIR              Where plan.json and compiled.json are written
  --max-repair-rounds N  Repair rounds after the first draft (default 2, ceiling 5)
  --no-refine            Compile the plan's own prompts instead of running prompt refinement
  --prompt-guide FILE    Model prompt guide forwarded on each rewrite (default: the guide the
                         catalog entry names, when it is on disk in this checkout)
  --force                Replace an existing plan.json that differs from the generated one
  --llm-timeout-seconds N  Give up on one LLM job after N seconds (default 1200)

OPTIONS (run):
  --api URL              SceneWorks API base URL (default http://127.0.0.1:8000, or $SCENEWORKS_API_URL)
  --token TOKEN          API token (default $SCENEWORKS_ACCESS_TOKEN; sent as X-SceneWorks-Token)
  --compiled FILE        Compiled requests to dispatch (default: compiled.json beside the plan)
  --shots A,B            Render only these shot ids, in plan order (default: every shot)
  --project-id ID        Reuse an existing project instead of creating one named after the plan
  --out DIR              Run record directory (default film-harness-runs/<utc-timestamp>)
  --poll-seconds N       Job polling cadence in seconds (default 5)
  --no-export            Skip the timeline assembly and MP4 export
  --skip-install-check   Do not refuse a model/tier the catalog reports as not installed

EDIT OPTIONS (trim / reorder / swap-take):
  --run RUN.json         The run record to edit. It names the project and the timeline, and is
                         rewritten in place with the edited sequence.
  --api URL / --token    As above
  --export               Re-export the MP4 after the edit (default: edit the timeline only)
  --poll-seconds N       Job polling cadence in seconds (default 5)

resume       picks a run up from its record: finished takes are reused, jobs still in flight are
             adopted, and what is left of the plan's wall-clock budget and per-shot attempt cap is
             what bounds it. The selected take of every shot is left alone.
replace-take RE-RENDERS: it rejects the take a shot is carrying and renders exactly ONE more for
             that shot, with --reason recorded beside it. Other shots' takes, jobs and assets are
             untouched; shots that declared a dependency on it are flagged needs_review, never
             re-rendered. Without --export the existing export is only marked stale.
swap-take    does NOT render: it swaps an asset that already exists into the assembled sequence.
             Use replace-take to make a new take, swap-take to choose a different existing one.
cancel       asks a run in another shell to stop; status prints what a record says without touching
             the API. A directory with no run.json in it is refused, not created.
review       samples the selected take of each shot at the review plan's declared positions (the
             frame_extract job), puts each declared question to the image_vqa job (SenseNova-U1-8B)
             and writes an observed-state document per take under <out>/reviews/. It renders
             nothing, moves no selection and writes no intended state. Bounded by the review plan's
             own limits; a review that runs out keeps its partial evidence and says why it stopped.
accept-take  records that a person looked and is happy: clears that shot's needsReview flags and
             leaves the selection alone. No API, no render.
reject-take  records that a person rejects the take: the take, its job and its asset are KEPT, the
             selection is cleared, the shots that declared a dependency on it are flagged
             needsReview and the export is marked stale. Nothing is re-rendered.
request-repair  ONE bounded repair attempt through replace-take, with the review's own mismatch
             flags folded into the recorded reason. It does not loop and it never feeds an
             observation back into the prompt.
review-eval  scores the reviewer against a fixed labeled set of correct and deliberately broken
             takes and reports detections, misses, false alarms, abstentions and overclaims.
review-fixtures writes the placeholder frames a labeled set names.

Every edit re-lays the whole sequence: picture items stay contiguous in cut order, each dialogue
clip keeps its offset from the start of its own shot, and the ambience/music beds re-span the new
duration without restarting at any cut.

ONE CONTROLLER PER RUN DIRECTORY: run, resume and replace-take each rewrite --out/run.json as they
go and nothing locks it, so two held against the same directory at once interleave their writes.
The idempotency keys make a SEQUENTIAL replay safe; they are not a lock. `run` refuses a directory
that already holds a record — use resume, or a different --out.

REVIEW IS ASSISTIVE, NOT QUALITY ASSURANCE. A local vision model both misses real faults and flags
correct takes; nothing it says approves, rejects or conditions anything.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("film-harness: cannot start runtime: {error}");
            return ExitCode::from(1);
        }
    };
    runtime.block_on(main_async(args))
}

async fn main_async(args: Vec<String>) -> ExitCode {
    let Some(command) = args.first().map(String::as_str) else {
        eprint!("{USAGE}");
        return ExitCode::from(1);
    };
    match command {
        "validate" | "run" | "plan" | "compile" => {}
        "resume" | "replace-take" | "request-repair" => {
            return record_command(command, &args[1..]).await
        }
        "review" => return review_command(&args[1..]).await,
        "accept-take" | "reject-take" => return decide_command(command, &args[1..]),
        "review-eval" => return review_eval_command(&args[1..]).await,
        "review-fixtures" => return review_fixtures_command(&args[1..]),
        "cancel" => return cancel_command(&args[1..]),
        "status" => return status_command(&args[1..]),
        "fixture-images" => return fixture_images(&args[1..]),
        "fixture-sound" => return fixture_sound(&args[1..]),
        "trim" | "reorder" | "swap-take" => return edit(command, &args[1..]).await,
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("film-harness: unknown command {other:?}\n\n{USAGE}");
            return ExitCode::from(1);
        }
    }
    let parsed = match parse_options(command, &args[1..]) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("film-harness: {message}\n\n{USAGE}");
            return ExitCode::from(1);
        }
    };
    let transport = match guarded_transport(&parsed.api_url, parsed.token.clone()) {
        Ok(transport) => transport,
        Err(code) => return code,
    };
    if command == "plan" || command == "compile" {
        return plan_or_compile(command, &transport, &parsed).await;
    }
    if command == "validate" {
        return match film_harness::validate(Some(&transport), &parsed.options).await {
            Ok((plan, pack)) => {
                println!(
                    "plan {:?} v{} ({} shots) and reference pack {:?} v{} ({} references) validate \
                     against {} on {}",
                    plan.id,
                    plan.version,
                    plan.shots.len(),
                    pack.id,
                    pack.version,
                    pack.references.len(),
                    plan.model.id,
                    parsed.api_url
                );
                ExitCode::SUCCESS
            }
            Err(error) => report_error(error),
        };
    }
    // SIGINT cancels the in-flight job through the API and still writes the record. Interrupting a
    // 45-minute render otherwise leaves it running on the GPU with nothing to say it happened,
    // which is the opposite of what the plan's cancellation limits exist for.
    // The run watches its OWN directory too, so `film-harness cancel --out DIR` from another shell
    // reaches it without a shared handle (sc-22711).
    let control = parsed.control.clone();
    let signal = spawn_interrupt_handler(control.clone());
    let outcome = film_harness::run_with_control(&transport, &parsed.options, &control).await;
    signal.finished();
    match outcome {
        Ok(record) => {
            print_record(&record, &parsed.options.out_dir);
            exit_code_for(&record)
        }
        Err(error) => report_error(error),
    }
}

/// The ONE way this binary reaches an API. Every command that dispatches work — run, resume,
/// replace-take, request-repair, review, review-eval, plan, compile, and an edit — builds its
/// transport here, and the local-only rule (`film_planner::local_only_guard`, E1) runs FIRST: a
/// hosted endpoint or a hosted-LLM credential in the environment is refused (exit 2) before a
/// single request leaves this machine. `plan`/`compile` used to be the only commands that checked;
/// `run --api https://…` did not (sc-22715). A source-text test in `film_planner` pins the count
/// of `HttpTransport::new` calls in this file to exactly this one.
fn guarded_transport(api_url: &str, token: Option<String>) -> Result<HttpTransport, ExitCode> {
    if let Err(error) = film_planner::local_only_guard(api_url) {
        return Err(report_error(error));
    }
    HttpTransport::new(api_url, token).map_err(|error| {
        eprintln!("film-harness: {error}");
        ExitCode::from(1)
    })
}

fn report_error(error: HarnessError) -> ExitCode {
    eprintln!("film-harness: {error}");
    match error {
        HarnessError::Validation(_) | HarnessError::Refused(_) => ExitCode::from(2),
        _ => ExitCode::from(1),
    }
}

struct Parsed {
    api_url: String,
    token: Option<String>,
    control: RunControl,
    options: RunOptions,
    planner: PlannerOptions,
    /// `--plan` as given, so `compile` can tell "no plan named" from "the default".
    plan_given: Option<PathBuf>,
}

/// Generate a plan from a brief, or recompile an existing (possibly edited) one. Both drive the
/// local LLM through the shipped `prompt_refine` seam; neither creates a job, a project or an asset.
async fn plan_or_compile(command: &str, transport: &HttpTransport, parsed: &Parsed) -> ExitCode {
    let llm = SceneWorksLlm::new(
        transport as &dyn ApiTransport,
        parsed.options.poll_interval,
        parsed.planner.job_timeout,
    );
    let result = if command == "plan" {
        film_planner::generate(transport, &llm, &parsed.planner).await
    } else {
        let Some(plan_path) = parsed.plan_given.clone() else {
            eprintln!("film-harness: compile needs --plan PLAN.json\n\n{USAGE}");
            return ExitCode::from(1);
        };
        film_planner::compile_existing(transport, &llm, &parsed.planner, &plan_path).await
    };
    match result {
        Ok(artifacts) => {
            println!(
                "plan {:?} v{} ({} shots, {} repair round(s)) written to {}",
                artifacts.plan.id,
                artifacts.plan.version,
                artifacts.plan.shots.len(),
                artifacts.repair_rounds,
                artifacts.plan_path.display()
            );
            println!(
                "{} compiled request(s) for {} written to {}",
                artifacts.compiled.requests.len(),
                artifacts.compiled.model.id,
                artifacts.compiled_path.display()
            );
            for request in &artifacts.compiled.requests {
                println!(
                    "  {:<8} {:<16} {:>7.4}s {}x{} {:?} prompt={} chars",
                    request.shot_id,
                    request.mode,
                    request.duration_seconds,
                    request.width,
                    request.height,
                    request.prompt_source,
                    request.prompt.chars().count()
                );
            }
            println!(
                "edit {} by hand if you want to change it, then re-run `film-harness compile` and \
                 `film-harness validate`",
                artifacts.plan_path.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => report_error(error),
    }
}

fn parse_options(command: &str, args: &[String]) -> Result<Parsed, String> {
    let mut plan: Option<PathBuf> = None;
    let mut brief: Option<PathBuf> = None;
    let mut compiled: Option<PathBuf> = None;
    let mut references: Option<PathBuf> = None;
    let mut max_repair_rounds = DEFAULT_MAX_REPAIR_ROUNDS;
    let mut refine_prompts = true;
    let mut prompt_guide: Option<PathBuf> = None;
    let mut force = false;
    let mut llm_timeout = DEFAULT_LLM_JOB_TIMEOUT;
    let mut api_url = std::env::var("SCENEWORKS_API_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let mut token = std::env::var("SCENEWORKS_ACCESS_TOKEN").ok();
    let mut shots: Option<Vec<String>> = None;
    let mut project_id: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut poll_seconds = 5_u64;
    let mut export = true;
    let mut require_installed = true;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--plan" => plan = Some(PathBuf::from(value()?)),
            "--brief" => brief = Some(PathBuf::from(value()?)),
            "--compiled" => compiled = Some(PathBuf::from(value()?)),
            "--max-repair-rounds" => {
                max_repair_rounds = value()?
                    .parse::<u32>()
                    .map_err(|error| format!("--max-repair-rounds: {error}"))?
            }
            "--no-refine" => refine_prompts = false,
            "--prompt-guide" => prompt_guide = Some(PathBuf::from(value()?)),
            "--force" => force = true,
            "--llm-timeout-seconds" => {
                llm_timeout = Duration::from_secs(
                    value()?
                        .parse::<u64>()
                        .map_err(|error| format!("--llm-timeout-seconds: {error}"))?
                        .max(1),
                )
            }
            "--references" => references = Some(PathBuf::from(value()?)),
            "--api" => api_url = value()?,
            "--token" => token = Some(value()?),
            "--shots" => {
                shots = Some(
                    value()?
                        .split(',')
                        .map(str::trim)
                        .filter(|id| !id.is_empty())
                        .map(str::to_owned)
                        .collect(),
                )
            }
            "--project-id" => project_id = Some(value()?),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--poll-seconds" => {
                poll_seconds = value()?
                    .parse::<u64>()
                    .map_err(|error| format!("--poll-seconds: {error}"))?
                    .max(1)
            }
            "--no-export" => export = false,
            "--skip-install-check" => require_installed = false,
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    let reference_pack_path = references.ok_or("--references is required")?;
    let plan_given = plan.clone();
    let out_dir = match out {
        Some(out) => out,
        None if command == "plan" || command == "compile" => {
            return Err(format!("{command} needs --out DIR"))
        }
        None => {
            let stamp: String = sceneworks_core::time::utc_now()
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
                .collect();
            PathBuf::from("film-harness-runs").join(stamp)
        }
    };
    // `plan` writes the plan it is about to generate; every other command reads one.
    let plan_path = match (command, plan) {
        (_, Some(path)) => path,
        ("plan", None) => out_dir.join("plan.json"),
        _ => return Err("--plan is required".to_owned()),
    };
    let brief_path = match (command, brief) {
        (_, Some(path)) => path,
        ("plan", None) => return Err("plan needs --brief BRIEF.json".to_owned()),
        // `compile` re-checks beat coverage when a brief sits beside the plan; absent is fine.
        _ => plan_path
            .parent()
            .map(|dir| dir.join("brief.json"))
            .unwrap_or_else(|| PathBuf::from("brief.json")),
    };
    // A `run` watches its own directory, which is how `film-harness cancel --out DIR` in another
    // shell reaches it (sc-22711). A stale sentinel from a previous controller is cleared first.
    let control = RunControl::watching(&out_dir);
    film_harness::clear_cancel_request(&out_dir).map_err(|error| error.to_string())?;
    Ok(Parsed {
        api_url: api_url.clone(),
        token,
        control,
        planner: PlannerOptions {
            brief_path,
            reference_pack_path: reference_pack_path.clone(),
            out_dir: out_dir.clone(),
            max_repair_rounds,
            refine_prompts,
            prompt_guide_path: prompt_guide,
            require_installed,
            api_url,
            force,
            poll_interval: Duration::from_secs(poll_seconds),
            job_timeout: llm_timeout,
        },
        plan_given,
        options: RunOptions {
            plan_path,
            reference_pack_path,
            compiled_path: compiled,
            project_id,
            shot_ids: shots,
            out_dir,
            poll_interval: Duration::from_secs(poll_seconds),
            export,
            require_installed,
        },
    })
}

fn fixture_images(args: &[String]) -> ExitCode {
    let mut out: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = iter.next().map(PathBuf::from),
            other => {
                eprintln!("film-harness: unknown option {other:?}\n\n{USAGE}");
                return ExitCode::from(1);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("film-harness: fixture-images needs --out DIR");
        return ExitCode::from(1);
    };
    match film_harness::write_fixture_images(&out) {
        Ok(paths) => {
            for path in paths {
                println!("{}", path.display());
            }
            println!(
                "{} plates written ({} roles)",
                FIXTURE_REFERENCES.len(),
                FIXTURE_REFERENCES
                    .iter()
                    .map(|(role, _)| *role)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("film-harness: {error}");
            ExitCode::from(1)
        }
    }
}

/// Cancel the in-flight run on SIGINT or SIGTERM, and exit on a second one.
///
/// The second listener is not optional: once a signal stream has been awaited, tokio owns that
/// signal for the rest of the process, so without it a second Ctrl-C would be swallowed and the
/// operator would have no way out but another signal.
///
/// SIGTERM is handled exactly as Ctrl-C (sc-22715 evaluation): a plain `kill <pid>` of the
/// controller used to take the crash path — the process died with the record left `running`, the
/// in-flight job unmentioned, and the operator's intent (stop this run) unrecorded. The record
/// was still resumable, which is what `resume` is for, but a signal the operator sent on purpose
/// deserves the cancel path: the job is cancelled through the API and the record says `canceled`.
///
/// The streams are registered HERE rather than inside the task, so they exist before the command
/// starts: a signal arriving in the gap between the spawn and the task's first poll is buffered by
/// the stream instead of being delivered to nothing.
fn spawn_interrupt_handler(control: RunControl) -> StopWatch {
    let (done, finished) = tokio::sync::oneshot::channel();
    let signals = StopSignals::listen();
    tokio::spawn(watch_stop_signals(control, signals, finished, |code| {
        std::process::exit(code)
    }));
    StopWatch { done }
}

/// The handle a command holds while its run is in flight.
///
/// [`StopWatch::finished`] tells the watcher the run is over. It is NOT an abort (sc-22715):
/// aborting the task left the process-wide handlers tokio installs with nothing consuming them, so
/// for the whole tail of the process — printing the record, the last flush — a plain `kill <pid>`
/// was swallowed and did nothing at all. The watcher stays alive and stands in for the default
/// disposition instead.
struct StopWatch {
    done: tokio::sync::oneshot::Sender<()>,
}

impl StopWatch {
    fn finished(self) {
        // The receiver only ever goes away with the task, which is the same message.
        let _ = self.done.send(());
    }
}

/// What the in-flight half of [`watch_stop_signals`] ended on.
enum InFlight {
    /// The run finished on its own; the process is now in its tail.
    RunOver,
    /// A second stop signal arrived: give up on the record and leave with this code.
    Exit(i32),
    /// No stop signal can be listened for at all, so there is nothing left to watch.
    Deaf,
}

/// Watch for stop signals while the run is in flight, then go on watching for the tail of the
/// process.
///
/// `exit` is the process exit, taken as an argument so both paths that reach it can be asserted
/// without ending the test binary.
async fn watch_stop_signals(
    control: RunControl,
    mut signals: StopSignals,
    finished: tokio::sync::oneshot::Receiver<()>,
    exit: impl Fn(i32),
) {
    match in_flight(&control, &mut signals, finished).await {
        InFlight::Exit(code) => return exit(code),
        InFlight::Deaf => return,
        InFlight::RunOver => {}
    }
    // The run is over and this process is finishing. The handlers tokio installed stay installed
    // for the life of the process — dropping the streams does not restore the default disposition
    // — so with nothing consuming a signal here, `kill <pid>` would do nothing for the rest of the
    // process. Stand in for the default disposition: terminate, with the code a shell reports.
    if let Some(signal) = signals.next().await {
        exit(signal.exit_code());
    }
}

/// The first half: cancel on the first signal, give up on the second, stand down when the run ends.
async fn in_flight(
    control: &RunControl,
    signals: &mut StopSignals,
    mut finished: tokio::sync::oneshot::Receiver<()>,
) -> InFlight {
    tokio::select! {
        _ = &mut finished => return InFlight::RunOver,
        first = signals.next() => {
            if first.is_none() {
                return InFlight::Deaf;
            }
        }
    }
    eprintln!(
        "film-harness: interrupt received — canceling the in-flight job and writing the run \
         record; interrupt again to exit now (the render keeps going)"
    );
    control.cancel();
    tokio::select! {
        _ = &mut finished => InFlight::RunOver,
        second = signals.next() => match second {
            Some(signal) => {
                eprintln!(
                    "film-harness: second interrupt — exiting without a run record; the worker \
                     may still be rendering (cancel it in the job list)"
                );
                InFlight::Exit(signal.exit_code())
            }
            None => InFlight::Deaf,
        },
    }
}

/// Which signal stopped the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopSignal {
    Interrupt,
    Terminate,
}

impl StopSignal {
    /// The exit code a shell reports for a process this signal killed (`128 + signo`).
    fn exit_code(self) -> i32 {
        match self {
            Self::Interrupt => 130,
            Self::Terminate => 143,
        }
    }
}

/// The signals that mean "stop this run": SIGINT (Ctrl-C) everywhere, and SIGTERM (`kill`) on
/// unix. Registering the streams once, up front, is what lets a second signal be observed at all.
///
/// BOTH streams are held here (sc-22715). SIGINT used to be `tokio::signal::ctrl_c()`, created
/// afresh on every call and dropped again whenever the SIGTERM arm of the `select!` won — and a
/// signal that arrives while no stream is registered for it is delivered to nothing, so a SIGINT
/// in the gap between the first `next()` returning and the second being awaited was lost. A
/// `Signal` held across both calls buffers it instead.
struct StopSignals {
    #[cfg(unix)]
    interrupt: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    terminate: Option<tokio::signal::unix::Signal>,
}

impl StopSignals {
    fn listen() -> Self {
        Self {
            #[cfg(unix)]
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .ok(),
            #[cfg(unix)]
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .ok(),
        }
    }

    /// Resolves on the next SIGINT or SIGTERM; `None` only if neither can be listened for.
    async fn next(&mut self) -> Option<StopSignal> {
        #[cfg(unix)]
        {
            match (self.interrupt.as_mut(), self.terminate.as_mut()) {
                (Some(interrupt), Some(terminate)) => tokio::select! {
                    signal = interrupt.recv() => signal.map(|()| StopSignal::Interrupt),
                    signal = terminate.recv() => signal.map(|()| StopSignal::Terminate),
                },
                (Some(interrupt), None) => interrupt.recv().await.map(|()| StopSignal::Interrupt),
                (None, Some(terminate)) => terminate.recv().await.map(|()| StopSignal::Terminate),
                // Neither stream registered: fall back to the portable listener rather than going
                // deaf, and report it as the Ctrl-C it is.
                (None, None) => tokio::signal::ctrl_c()
                    .await
                    .ok()
                    .map(|()| StopSignal::Interrupt),
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .ok()
                .map(|()| StopSignal::Interrupt)
        }
    }
}

#[cfg(all(test, unix))]
mod stop_signal_tests {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use tokio::sync::{Mutex, MutexGuard};

    use sceneworks_rust_api::film_harness::RunControl;

    use super::{watch_stop_signals, StopSignal, StopSignals};

    /// Signal handlers are PROCESS-wide and tokio delivers a raised signal to every registered
    /// stream, so two of these running at once would read each other's signals. The test harness
    /// runs tests on threads of one process, so they take turns.
    ///
    /// An async mutex, because the guard is deliberately held across the awaits that raise and
    /// observe the signals — which is the whole point of taking the turn.
    async fn one_at_a_time() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().await
    }

    fn raise(signal: nix::sys::signal::Signal) {
        nix::sys::signal::raise(signal).expect("raise");
    }

    /// A `kill <pid>` (SIGTERM) reaches the same listener as Ctrl-C. The stream is registered
    /// before the signal is raised, so the raise is observed rather than terminating the test
    /// binary — which is exactly the difference between the cancel path and the crash path.
    #[tokio::test]
    async fn sigterm_is_a_stop_signal() {
        let _serial = one_at_a_time().await;
        let mut signals = StopSignals::listen();
        assert!(signals.terminate.is_some(), "SIGTERM stream registered");
        raise(nix::sys::signal::Signal::SIGTERM);
        let observed = tokio::time::timeout(Duration::from_secs(5), signals.next())
            .await
            .expect("the signal is observed within 5s");
        assert_eq!(observed, Some(StopSignal::Terminate));
    }

    /// A SIGINT that arrives while nothing is awaiting one is still the operator's SECOND
    /// interrupt when it is next asked for (sc-22715).
    ///
    /// `next()` used to build a fresh `tokio::signal::ctrl_c()` inside its `select!`, so the SIGINT
    /// stream existed only for the duration of one call: whenever the SIGTERM arm won, the stream
    /// was dropped, and a SIGINT raised before the next call had nothing registered to deliver it
    /// to. The second Ctrl-C — the operator's only way out of a 45-minute render — was swallowed
    /// and the call hung until a third signal. Holding the `Signal` in the struct buffers it.
    ///
    /// The `sleep` below is load-bearing, not padding: raising a signal only sets a pending flag,
    /// and tokio's signal driver turns that into a delivery on a later turn of the runtime. Without
    /// an await in the gap, the driver cannot run until the next `select!` has ALREADY created its
    /// short-lived stream, and the old code caught the signal by accident. The sleep puts the
    /// delivery where it really lands in a multi-threaded runtime — while nothing is registered.
    #[tokio::test]
    async fn a_signal_that_lands_between_two_calls_is_still_observed() {
        let _serial = one_at_a_time().await;
        let mut signals = StopSignals::listen();
        assert!(signals.interrupt.is_some(), "SIGINT stream registered");
        raise(nix::sys::signal::Signal::SIGTERM);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), signals.next())
                .await
                .expect("the first signal is observed within 5s"),
            Some(StopSignal::Terminate)
        );
        // Nothing is awaiting a signal at this instant — exactly the gap between the cancel being
        // recorded and the second listener being awaited.
        raise(nix::sys::signal::Signal::SIGINT);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), signals.next())
                .await
                .expect("the interrupt raised between the two calls is not lost"),
            Some(StopSignal::Interrupt)
        );
    }

    /// Once the run is over, a `kill <pid>` still terminates the process (sc-22715).
    ///
    /// The watcher used to be `abort()`ed the moment the command returned, leaving the handlers
    /// tokio installed with nothing consuming them: SIGTERM's default disposition was gone, and for
    /// the whole tail of the process — printing the record, the last flush — a plain `kill` did
    /// nothing whatsoever. The watcher now stays and stands in for the default disposition.
    #[tokio::test]
    async fn a_signal_after_the_run_is_over_still_terminates() {
        let _serial = one_at_a_time().await;
        let signals = StopSignals::listen();
        let (done, finished) = tokio::sync::oneshot::channel();
        let code = Arc::new(AtomicI32::new(0));
        let recorder = Arc::clone(&code);
        let control = RunControl::new();
        let cancelled = control.clone();
        let watcher = tokio::spawn(watch_stop_signals(
            control,
            signals,
            finished,
            move |value| {
                recorder.store(value, Ordering::SeqCst);
            },
        ));
        // The run is over: this is what replaced `signal.abort()`.
        done.send(()).expect("the watcher is listening");
        // Raised until the watcher has reached its tail, because the hand-off is a task switch.
        let observed = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                raise(nix::sys::signal::Signal::SIGTERM);
                if code.load(Ordering::SeqCst) != 0 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await;
        assert!(
            observed.is_ok(),
            "a SIGTERM after the run finished was swallowed: the process would ignore `kill`"
        );
        assert_eq!(
            code.load(Ordering::SeqCst),
            143,
            "the tail reports what a shell reports for a SIGTERM (128 + 15)"
        );
        assert!(
            !cancelled.is_canceled(),
            "the run is already over, so the tail must not pretend it cancelled anything"
        );
        watcher.await.expect("the watcher exits cleanly");
    }
}

fn exit_code_for(record: &RunRecord) -> ExitCode {
    match record.outcome {
        RunOutcome::Completed => ExitCode::SUCCESS,
        RunOutcome::Rejected => ExitCode::from(2),
        _ => ExitCode::from(3),
    }
}

/// One screen of what a run record says: per-shot selection, attempts, review flags, the export,
/// and — the part an operator acts on — whether the run can be resumed and why it stopped.
fn print_record(record: &RunRecord, out_dir: &std::path::Path) {
    println!(
        "run {} {} — {:?} ({:.0}s); record at {}",
        record.run_id,
        match record.state {
            RunState::Running => "RUNNING",
            RunState::Finished => "finished",
        },
        record.outcome,
        record.elapsed_seconds,
        out_dir.join(film_harness::RUN_RECORD_FILE).display()
    );
    for shot in &record.shots {
        let selected = shot.selected();
        println!(
            "  {:<8} {:<15} attempts={} selected={} job={} asset={}{}",
            shot.shot_id,
            format!("{:?}", shot.outcome),
            shot.attempts.len(),
            shot.selected_attempt
                .map(|attempt| attempt.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            selected
                .or_else(|| shot.attempts.last())
                .and_then(|attempt| attempt.job_id.as_deref())
                .unwrap_or("-"),
            selected
                .and_then(|attempt| attempt.take.as_ref())
                .map(|take| take.asset_id.as_str())
                .unwrap_or("-"),
            shot.attempts
                .last()
                .and_then(|attempt| attempt.error.as_deref())
                .map(|error| format!("  error: {error}"))
                .unwrap_or_default()
        );
        for attempt in shot.attempts.iter().filter(|a| a.rejection.is_some()) {
            let rejection = attempt.rejection.as_ref().expect("filtered");
            println!(
                "           rejected attempt {} ({}): {}",
                attempt.attempt, rejection.at, rejection.reason
            );
        }
        for flag in &shot.needs_review {
            println!("           NEEDS REVIEW: {}", flag.reason);
        }
        print!("{}", review::format_shot_reviews(shot));
    }
    if let Some(export) = &record.export {
        println!(
            "  export   {:<15} job={} asset={} path={}{}{}",
            export.status,
            export.job_id,
            export.asset_id.as_deref().unwrap_or("-"),
            export.render_path.as_deref().unwrap_or("-"),
            if export.stale { "  STALE" } else { "" },
            export
                .error
                .as_deref()
                .map(|error| format!("  error: {error}"))
                .unwrap_or_default()
        );
    }
    if let Some(stop) = &record.stop {
        println!(
            "  stopped: {} — {}\n  {}",
            stop.reason,
            stop.detail,
            if stop.resumable {
                "resumable: `film-harness resume --out DIR`"
            } else {
                "terminal: this run will not dispatch again"
            }
        );
    }
}

/// `resume` and `replace-take`: both continue an existing record through the API.
async fn record_command(command: &str, args: &[String]) -> ExitCode {
    let parsed = match parse_record_options(command, args) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("film-harness: {message}\n\n{USAGE}");
            return ExitCode::from(1);
        }
    };
    let transport = match guarded_transport(&parsed.api_url, parsed.token.clone()) {
        Ok(transport) => transport,
        Err(code) => return code,
    };
    let signal = spawn_interrupt_handler(parsed.options.control.clone());
    let result = match command {
        "resume" => film_harness::resume(&transport, &parsed.options).await,
        other => {
            let Some(shot) = parsed.shot.as_deref() else {
                eprintln!("film-harness: {other} needs --shot SHxxx\n\n{USAGE}");
                return ExitCode::from(1);
            };
            if other == "request-repair" {
                review::request_repair(&transport, &parsed.options, shot, &parsed.reason).await
            } else {
                film_harness::replace_take(&transport, &parsed.options, shot, &parsed.reason).await
            }
        }
    };
    signal.finished();
    match result {
        Ok(record) => {
            print_record(&record, &parsed.options.out_dir);
            exit_code_for(&record)
        }
        Err(error) => report_error(error),
    }
}

/// `review`: sample the selected takes and put the review plan's questions to the vision model.
async fn review_command(args: &[String]) -> ExitCode {
    let Some(out_dir) = flag_value(args, "--out") else {
        eprintln!("film-harness: review needs --out DIR\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let api_url = flag_value(args, "--api")
        .or_else(|| std::env::var("SCENEWORKS_API_URL").ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let token =
        flag_value(args, "--token").or_else(|| std::env::var("SCENEWORKS_ACCESS_TOKEN").ok());
    let poll_seconds = match flag_value(args, "--poll-seconds")
        .map(|value| value.parse::<u64>())
        .transpose()
    {
        Ok(value) => value.unwrap_or(3).max(1),
        Err(error) => {
            eprintln!("film-harness: --poll-seconds: {error}");
            return ExitCode::from(1);
        }
    };
    let mut options = ReviewOptions::new(PathBuf::from(&out_dir));
    options.review_plan_path = flag_value(args, "--review-plan").map(PathBuf::from);
    options.poll_interval = Duration::from_secs(poll_seconds);
    options.shot_ids = flag_value(args, "--shots")
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if let Err(error) = film_harness::clear_cancel_request(&options.out_dir) {
        eprintln!("film-harness: {error}");
        return ExitCode::from(1);
    }
    let transport = match guarded_transport(&api_url, token) {
        Ok(transport) => transport,
        Err(code) => return code,
    };
    // The review document's own declared bounds, read BEFORE anything is dispatched: the preflight
    // checks the host against `limits.maxMemoryGb`, so it has to know what the document asks for.
    let limits = match review::review_limits(&options.out_dir, options.review_plan_path.as_deref())
    {
        Ok(limits) => limits,
        Err(error) => return report_error(error),
    };
    let signal = spawn_interrupt_handler(options.control.clone());
    let vision = VqaVision::new(&transport, options.poll_interval, options.control.clone());
    for check in [
        vision.preflight(limits).await,
        vision.preflight_model().await,
    ] {
        if let Err(error) = check {
            signal.finished();
            return report_error(error);
        }
    }
    let result = review::review(&transport, &options, &vision).await;
    signal.finished();
    match result {
        Ok(record) => {
            print_record(&record, &options.out_dir);
            println!("\n{ASSISTIVE_NOTICE}");
            let flagged = record
                .shots
                .iter()
                .filter_map(|shot| shot.latest_review())
                .any(|review| review.actionable_flags > 0);
            if flagged {
                ExitCode::from(3)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => report_error(error),
    }
}

/// `accept-take` / `reject-take`: a human decision, recorded. No API, no render.
fn decide_command(command: &str, args: &[String]) -> ExitCode {
    let (Some(out_dir), Some(shot)) = (flag_value(args, "--out"), flag_value(args, "--shot"))
    else {
        eprintln!("film-harness: {command} needs --out DIR --shot SHxxx\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let decision = if command == "accept-take" {
        Decision::Accept
    } else {
        Decision::Reject
    };
    let reason = flag_value(args, "--reason").unwrap_or_default();
    if decision == Decision::Reject && reason.trim().is_empty() {
        eprintln!(
            "film-harness: reject-take needs --reason TEXT — the reason is what the next person \
             (and the repair) reads\n\n{USAGE}"
        );
        return ExitCode::from(1);
    }
    let reason = if reason.trim().is_empty() {
        "accepted by hand".to_owned()
    } else {
        reason
    };
    let out_dir = PathBuf::from(out_dir);
    match review::decide_take(&out_dir, &shot, decision, &reason) {
        Ok(record) => {
            print_record(&record, &out_dir);
            ExitCode::SUCCESS
        }
        Err(error) => report_error(error),
    }
}

/// `review-eval`: score the reviewer against a fixed labeled set.
async fn review_eval_command(args: &[String]) -> ExitCode {
    let Some(set) = flag_value(args, "--set") else {
        eprintln!("film-harness: review-eval needs --set LABELS.jsonc\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let out_dir =
        flag_value(args, "--out").unwrap_or_else(|| "film-harness-review-eval".to_owned());
    let mut options = EvalOptions::new(PathBuf::from(set), PathBuf::from(out_dir));
    options.media_root = flag_value(args, "--media-root").map(PathBuf::from);
    options.project_id = flag_value(args, "--project-id");
    if let Some(seconds) = flag_value(args, "--poll-seconds").and_then(|v| v.parse::<u64>().ok()) {
        options.poll_interval = Duration::from_secs(seconds.max(1));
    }
    // `--scripted` runs the whole evaluation with no model at all. It measures the harness, not the
    // reviewer, and every document it writes says `realModelInference: false`.
    let scripted = args.iter().any(|arg| arg == "--scripted");
    let api_url = flag_value(args, "--api")
        .or_else(|| std::env::var("SCENEWORKS_API_URL").ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let token =
        flag_value(args, "--token").or_else(|| std::env::var("SCENEWORKS_ACCESS_TOKEN").ok());
    let transport = match guarded_transport(&api_url, token) {
        Ok(transport) => transport,
        Err(code) => return code,
    };
    let signal = spawn_interrupt_handler(options.control.clone());
    let scripted_backend = ScriptedVision::new();
    let vqa_backend = VqaVision::new(&transport, options.poll_interval, options.control.clone());
    if !scripted {
        let limits = match review::eval_review_limits(&options.set_path) {
            Ok(limits) => limits,
            Err(error) => {
                signal.finished();
                return report_error(error);
            }
        };
        for check in [
            vqa_backend.preflight(limits).await,
            vqa_backend.preflight_model().await,
        ] {
            if let Err(error) = check {
                signal.finished();
                return report_error(error);
            }
        }
    }
    let vision: &dyn review::ReviewVision = if scripted {
        &scripted_backend
    } else {
        &vqa_backend
    };
    let result = review::review_eval(&transport, &options, vision).await;
    signal.finished();
    match result {
        Ok(results) => {
            print!("{}", format_eval_report(&results));
            println!(
                "results: {}",
                options.out_dir.join("review-eval.json").display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => report_error(error),
    }
}

/// `review-fixtures`: write the placeholder frames a labeled set names.
fn review_fixtures_command(args: &[String]) -> ExitCode {
    let Some(set) = flag_value(args, "--set") else {
        eprintln!("film-harness: review-fixtures needs --set LABELS.jsonc\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let media_root = flag_value(args, "--media-root").map(PathBuf::from);
    match review::write_review_fixture_frames(&PathBuf::from(set), media_root.as_deref()) {
        Ok(paths) => {
            for path in &paths {
                println!("{}", path.display());
            }
            println!(
                "{} placeholder frame(s) written — flat plates, not footage; point review-eval at \
                 real media to measure anything",
                paths.len()
            );
            ExitCode::SUCCESS
        }
        Err(error) => report_error(error),
    }
}

fn cancel_command(args: &[String]) -> ExitCode {
    let Some(out_dir) = flag_value(args, "--out") else {
        eprintln!("film-harness: cancel needs --out DIR\n\n{USAGE}");
        return ExitCode::from(1);
    };
    match film_harness::request_cancel(&PathBuf::from(out_dir)) {
        Ok(path) => {
            println!(
                "cancel requested ({}); the run stops within one poll interval and its record says \
                 whether it can be resumed",
                path.display()
            );
            ExitCode::SUCCESS
        }
        // A mistyped --out is refused (exit 2) rather than reported as a cancel nobody receives.
        Err(error) => report_error(error),
    }
}

fn status_command(args: &[String]) -> ExitCode {
    let Some(out_dir) = flag_value(args, "--out") else {
        eprintln!("film-harness: status needs --out DIR\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let out_dir = PathBuf::from(out_dir);
    match film_harness::read_run_record(&out_dir) {
        Ok(record) => {
            print_record(&record, &out_dir);
            if record.is_resumable() {
                ExitCode::from(3)
            } else {
                exit_code_for(&record)
            }
        }
        Err(error) => report_error(error),
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

struct ParsedRecord {
    api_url: String,
    token: Option<String>,
    shot: Option<String>,
    reason: String,
    options: ResumeOptions,
}

fn parse_record_options(command: &str, args: &[String]) -> Result<ParsedRecord, String> {
    let mut out: Option<PathBuf> = None;
    let mut api_url = std::env::var("SCENEWORKS_API_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let mut token = std::env::var("SCENEWORKS_ACCESS_TOKEN").ok();
    let mut shot: Option<String> = None;
    let mut reason = String::new();
    let mut poll_seconds = 5_u64;
    // `resume` finishes the run, so it exports by default. `replace-take` re-renders ONE shot; the
    // export is other work, so it only happens when asked for and is otherwise marked stale.
    let mut export = command == "resume";
    let mut require_installed = true;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--api" => api_url = value()?,
            "--token" => token = Some(value()?),
            "--shot" => shot = Some(value()?),
            "--reason" => reason = value()?,
            "--poll-seconds" => {
                poll_seconds = value()?
                    .parse::<u64>()
                    .map_err(|error| format!("--poll-seconds: {error}"))?
                    .max(1)
            }
            "--no-export" => export = false,
            "--export" => export = true,
            "--skip-install-check" => require_installed = false,
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    let out_dir = out.ok_or("--out is required")?;
    if reason.trim().is_empty() {
        reason = if command == "request-repair" {
            "repair requested by hand".to_owned()
        } else {
            "replaced by hand".to_owned()
        };
    }
    let mut options = ResumeOptions::new(out_dir);
    options.poll_interval = Duration::from_secs(poll_seconds);
    options.export = export;
    options.require_installed = require_installed;
    Ok(ParsedRecord {
        api_url,
        token,
        shot,
        reason,
        options,
    })
}

/// `trim` / `reorder` / `swap-take` — edit an assembled sequence in place (sc-22712).
async fn edit(command: &str, args: &[String]) -> ExitCode {
    let mut run_record: Option<PathBuf> = None;
    let mut api_url = std::env::var("SCENEWORKS_API_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let mut token = std::env::var("SCENEWORKS_ACCESS_TOKEN").ok();
    let mut shot: Option<String> = None;
    let mut asset: Option<String> = None;
    let mut order: Option<Vec<String>> = None;
    let mut source_in: Option<f64> = None;
    let mut source_out: Option<f64> = None;
    let mut export = false;
    let mut poll_seconds = 5_u64;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        let parsed = (|| -> Result<(), String> {
            match arg.as_str() {
                "--run" => run_record = Some(PathBuf::from(value()?)),
                "--api" => api_url = value()?,
                "--token" => token = Some(value()?),
                "--shot" => shot = Some(value()?),
                "--asset" => asset = Some(value()?),
                "--order" => {
                    order = Some(
                        value()?
                            .split(',')
                            .map(str::trim)
                            .filter(|id| !id.is_empty())
                            .map(str::to_owned)
                            .collect(),
                    )
                }
                "--source-in" => {
                    source_in = Some(
                        value()?
                            .parse::<f64>()
                            .map_err(|error| format!("--source-in: {error}"))?,
                    )
                }
                "--source-out" => {
                    source_out = Some(
                        value()?
                            .parse::<f64>()
                            .map_err(|error| format!("--source-out: {error}"))?,
                    )
                }
                "--export" => export = true,
                "--poll-seconds" => {
                    poll_seconds = value()?
                        .parse::<u64>()
                        .map_err(|error| format!("--poll-seconds: {error}"))?
                        .max(1)
                }
                other => return Err(format!("unknown option {other:?}")),
            }
            Ok(())
        })();
        if let Err(message) = parsed {
            eprintln!("film-harness: {message}\n\n{USAGE}");
            return ExitCode::from(1);
        }
    }

    let Some(run_record_path) = run_record else {
        eprintln!("film-harness: {command} needs --run RUN.json\n\n{USAGE}");
        return ExitCode::from(1);
    };
    let edit = match command {
        "trim" => {
            let Some(shot_id) = shot else {
                eprintln!("film-harness: trim needs --shot ID\n\n{USAGE}");
                return ExitCode::from(1);
            };
            if source_in.is_none() && source_out.is_none() {
                eprintln!("film-harness: trim needs --source-in and/or --source-out\n\n{USAGE}");
                return ExitCode::from(1);
            }
            TimelineEdit::Trim {
                shot_id,
                source_in,
                source_out,
            }
        }
        "reorder" => {
            let Some(shot_ids) = order.filter(|ids| !ids.is_empty()) else {
                eprintln!("film-harness: reorder needs --order A,B,C\n\n{USAGE}");
                return ExitCode::from(1);
            };
            TimelineEdit::Reorder { shot_ids }
        }
        _ => {
            let (Some(shot_id), Some(asset_id)) = (shot, asset) else {
                eprintln!(
                    "film-harness: swap-take needs --shot ID and --asset ASSET_ID\n\n{USAGE}"
                );
                return ExitCode::from(1);
            };
            TimelineEdit::SwapTake { shot_id, asset_id }
        }
    };

    let transport = match guarded_transport(&api_url, token) {
        Ok(transport) => transport,
        Err(code) => return code,
    };
    let options = EditOptions {
        run_record_path: run_record_path.clone(),
        export,
        poll_interval: Duration::from_secs(poll_seconds),
    };
    match film_harness::edit_timeline(&transport, &options, edit).await {
        Ok(record) => {
            let Some(timeline) = &record.timeline else {
                eprintln!(
                    "film-harness: the edit left no timeline in {}",
                    run_record_path.display()
                );
                return ExitCode::from(1);
            };
            println!(
                "{command} applied to timeline {} ({:.3}s, {} picture items, {} tracks); record at {}",
                timeline.timeline_id,
                timeline.duration_seconds,
                timeline.items.len(),
                timeline.tracks.len(),
                run_record_path.display()
            );
            for item in &timeline.items {
                println!(
                    "  {:<8} {:>7.3}..{:<7.3} source {:.3}..{:.3} asset={} generatedAudio={}",
                    item.shot_id.as_deref().unwrap_or("-"),
                    item.timeline_start,
                    item.timeline_end,
                    item.source_in,
                    item.source_out,
                    item.asset_id,
                    item.generated_audio
                        .map(|policy| policy.as_timeline_str())
                        .unwrap_or("-")
                );
            }
            for track in timeline.tracks.iter().filter(|track| track.kind == "audio") {
                println!(
                    "  {:<14} gain={:.2} muted={} items={}",
                    track.role,
                    track.gain,
                    track.muted,
                    track.items.len()
                );
            }
            if let Some(export) = &record.export {
                println!(
                    "  export   {:<15} job={} asset={} path={}",
                    export.status,
                    export.job_id,
                    export.asset_id.as_deref().unwrap_or("-"),
                    export.render_path.as_deref().unwrap_or("-")
                );
                if export.status != "completed" {
                    return ExitCode::from(3);
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => report_error(error),
    }
}

/// `fixture-sound --out DIR` — write the deterministic placeholder clips the fixture pack's sound
/// roles resolve against (sc-22712).
fn fixture_sound(args: &[String]) -> ExitCode {
    let mut out: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = iter.next().map(PathBuf::from),
            other => {
                eprintln!("film-harness: unknown option {other:?}\n\n{USAGE}");
                return ExitCode::from(1);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("film-harness: fixture-sound needs --out DIR");
        return ExitCode::from(1);
    };
    match film_harness::write_fixture_sound(&out) {
        Ok(paths) => {
            for path in paths {
                println!("{}", path.display());
            }
            println!(
                "{} clips written at {} Hz ({})",
                FIXTURE_SOUNDS.len(),
                film_harness::FIXTURE_SOUND_RATE,
                FIXTURE_SOUNDS
                    .iter()
                    .map(|(role, seconds, hz, _)| format!("{role} {seconds}s @{hz}Hz"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("film-harness: {error}");
            ExitCode::from(1)
        }
    }
}
