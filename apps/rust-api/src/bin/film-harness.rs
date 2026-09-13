//! `film-harness` — render a hand-authored production plan into a SceneWorks sequence through a
//! running SceneWorks API (epic 22708, sc-22710).
//!
//! ```text
//! film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL]
//! film-harness run      --plan PLAN.json --references REFERENCES.json [--api URL] [--shots SH010,SH020]
//!                       [--project-id ID] [--out DIR] [--poll-seconds N] [--no-export]
//!                       [--skip-install-check]
//! film-harness fixture-images --out DIR
//! ```
//!
//! `run` needs a SceneWorks API with a registered GPU worker (`video_generate`) and a utility
//! worker (`timeline_export`, e.g. `SCENEWORKS_RUN_UTILITY_INPROCESS=1`). It creates nothing until
//! the plan, the reference pack, the model's catalog entry and the host all validate; the run
//! record (`run.json`) is written under `--out` on every path, including refusal and a
//! transport/API failure partway through. Exit codes: 0 on a completed run, 2 when the plan was
//! refused before dispatch, 3 when the run stopped on a limit or a shot failed, 1 on a
//! transport/API/io error.
//!
//! Ctrl-C cancels the in-flight job through the API, stops dispatching and writes the record with
//! `outcome: failed` and a "canceled by operator" diagnostic, rather than orphaning a render.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sceneworks_core::film_plan::RunOutcome;
use sceneworks_rust_api::film_harness::{
    self, HarnessError, HttpTransport, RunOptions, FIXTURE_REFERENCES,
};

const USAGE: &str = "\
film-harness — render a hand-authored production plan into a SceneWorks sequence

USAGE:
  film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL] [--shots IDS]
  film-harness run      --plan PLAN.json --references REFERENCES.json [OPTIONS]
  film-harness fixture-images --out DIR

OPTIONS (run):
  --api URL              SceneWorks API base URL (default http://127.0.0.1:8000, or $SCENEWORKS_API_URL)
  --token TOKEN          API token (default $SCENEWORKS_ACCESS_TOKEN; sent as X-SceneWorks-Token)
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
        "validate" | "run" => {}
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
    let parsed = match parse_options(&args[1..]) {
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
    let control = film_harness::RunControl::new();
    let signal = tokio::spawn({
        let control = control.clone();
        async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!(
                    "film-harness: interrupt received — canceling the in-flight job and writing \
                     the run record; interrupt again to exit now (the render keeps going)"
                );
                control.cancel();
                // The second listener is not optional: once `ctrl_c()` has been awaited, tokio owns
                // SIGINT for the rest of the process, so without this a second Ctrl-C would be
                // swallowed and the operator would have no way out but another signal.
                if tokio::signal::ctrl_c().await.is_ok() {
                    eprintln!(
                        "film-harness: second interrupt — exiting without a run record; the \
                         worker may still be rendering (cancel it in the job list)"
                    );
                    std::process::exit(130);
                }
            }
        }
    });
    let outcome = film_harness::run_with_control(&transport, &parsed.options, &control).await;
    signal.abort();
    match outcome {
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
}

fn parse_options(args: &[String]) -> Result<Parsed, String> {
    let mut plan: Option<PathBuf> = None;
    let mut references: Option<PathBuf> = None;
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
    let plan_path = plan.ok_or("--plan is required")?;
    let reference_pack_path = references.ok_or("--references is required")?;
    let out_dir = out.unwrap_or_else(|| {
        let stamp: String = sceneworks_core::time::utc_now()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        PathBuf::from("film-harness-runs").join(stamp)
    });
    Ok(Parsed {
        api_url,
        token,
        options: RunOptions {
            plan_path,
            reference_pack_path,
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
