//! `film-harness` — plan, compile and render a production plan into a SceneWorks sequence through a
//! running SceneWorks API (epic 22708, sc-22710 + sc-22713).
//!
//! ```text
//! film-harness plan     --brief BRIEF.json --references REFERENCES.json --out DIR [--api URL]
//! film-harness compile  --plan PLAN.json --references REFERENCES.json --out DIR [--api URL]
//! film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL]
//! film-harness run      --plan PLAN.json --references REFERENCES.json [--api URL] [--shots SH010,SH020]
//!                       [--project-id ID] [--out DIR] [--poll-seconds N] [--no-export]
//!                       [--skip-install-check]
//! film-harness fixture-images --out DIR
//! ```
//!
//! The intended loop is `plan` -> read and edit `plan.json` -> `compile` -> `validate` -> `run`.
//! `plan` drives the LOCAL LLM through the shipped `prompt_refine` seam; it writes the plan and the
//! compiled per-shot requests as two versioned documents and touches nothing else. A hand-authored
//! plan skips straight to `validate`/`run` exactly as before.
//!
//! `run` needs a SceneWorks API with a registered GPU worker (`video_generate`) and a utility
//! worker (`timeline_export`, e.g. `SCENEWORKS_RUN_UTILITY_INPROCESS=1`). It creates nothing until
//! the plan, the reference pack, the model's catalog entry and the host all validate; the run
//! record (`run.json`) is written under `--out` on every path, including refusal. Exit codes: 0 on
//! a completed run, 2 when the plan was refused before dispatch, 3 when the run stopped on a limit
//! or a shot failed, 1 on a transport/API/io error.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sceneworks_core::film_plan::RunOutcome;
use sceneworks_rust_api::film_harness::{
    self, ApiTransport, HarnessError, HttpTransport, RunOptions, FIXTURE_REFERENCES,
};
use sceneworks_rust_api::film_planner::{
    self, PlannerOptions, SceneWorksLlm, DEFAULT_LLM_JOB_TIMEOUT, DEFAULT_MAX_REPAIR_ROUNDS,
};

const USAGE: &str = "\
film-harness — plan, compile and render a production plan into a SceneWorks sequence

USAGE:
  film-harness plan     --brief BRIEF.json --references REFERENCES.json --out DIR [OPTIONS]
  film-harness compile  --plan PLAN.json --references REFERENCES.json --out DIR [OPTIONS]
  film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL] [--shots IDS]
  film-harness run      --plan PLAN.json --references REFERENCES.json [OPTIONS]
  film-harness fixture-images --out DIR

OPTIONS (plan / compile):
  --brief BRIEF.json     The brief to plan from (plan); re-checked for dropped beats (compile)
  --out DIR              Where plan.json and compiled.json are written
  --max-repair-rounds N  Repair rounds after the first draft (default 2, ceiling 5)
  --no-refine            Compile the plan's own prompts instead of running prompt refinement
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
        "fixture-images" => return fixture_images(&args[1..]),
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
    let transport = match HttpTransport::new(&parsed.api_url, parsed.token.clone()) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("film-harness: {error}");
            return ExitCode::from(1);
        }
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
    match film_harness::run(&transport, &parsed.options).await {
        Ok(record) => {
            let record_path = parsed.options.out_dir.join("run.json");
            println!(
                "run {} finished: {:?} ({:.0}s); record at {}",
                record.run_id,
                record.outcome,
                record.elapsed_seconds,
                record_path.display()
            );
            for shot in &record.shots {
                let last = shot.attempts.last();
                println!(
                    "  {:<8} {:<15} attempts={} job={} asset={}{}",
                    shot.shot_id,
                    format!("{:?}", shot.outcome),
                    shot.attempts.len(),
                    last.and_then(|attempt| attempt.job_id.as_deref())
                        .unwrap_or("-"),
                    last.and_then(|attempt| attempt.take.as_ref())
                        .map(|take| take.asset_id.as_str())
                        .unwrap_or("-"),
                    last.and_then(|attempt| attempt.error.as_deref())
                        .map(|error| format!("  error: {error}"))
                        .unwrap_or_default()
                );
            }
            if let Some(export) = &record.export {
                println!(
                    "  export   {:<15} job={} asset={} path={}{}",
                    export.status,
                    export.job_id,
                    export.asset_id.as_deref().unwrap_or("-"),
                    export.render_path.as_deref().unwrap_or("-"),
                    export
                        .error
                        .as_deref()
                        .map(|error| format!("  error: {error}"))
                        .unwrap_or_default()
                );
            }
            match record.outcome {
                RunOutcome::Completed => ExitCode::SUCCESS,
                RunOutcome::Rejected => ExitCode::from(2),
                _ => ExitCode::from(3),
            }
        }
        Err(error) => report_error(error),
    }
}

fn report_error(error: HarnessError) -> ExitCode {
    eprintln!("film-harness: {error}");
    match error {
        HarnessError::Validation(_) => ExitCode::from(2),
        _ => ExitCode::from(1),
    }
}

struct Parsed {
    api_url: String,
    token: Option<String>,
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
    Ok(Parsed {
        api_url: api_url.clone(),
        token,
        planner: PlannerOptions {
            brief_path,
            reference_pack_path: reference_pack_path.clone(),
            out_dir: out_dir.clone(),
            max_repair_rounds,
            refine_prompts,
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
