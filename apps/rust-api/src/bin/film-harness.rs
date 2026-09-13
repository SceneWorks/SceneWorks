//! `film-harness` — render a hand-authored production plan into a SceneWorks sequence through a
//! running SceneWorks API (epic 22708, sc-22710).
//!
//! ```text
//! film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL]
//! film-harness run      --plan PLAN.json --references REFERENCES.json [--api URL] [--shots SH010,SH020]
//!                       [--project-id ID] [--out DIR] [--poll-seconds N] [--no-export]
//!                       [--skip-install-check]
//! film-harness trim         --run RUN.json --shot SH010 [--source-in S] [--source-out S]
//! film-harness reorder      --run RUN.json --order SH020,SH010
//! film-harness replace-take --run RUN.json --shot SH010 --asset asset_...
//! film-harness fixture-images --out DIR
//! film-harness fixture-sound  --out DIR
//! ```
//!
//! `run` needs a SceneWorks API with a registered GPU worker (`video_generate`) and a utility
//! worker (`timeline_export`, e.g. `SCENEWORKS_RUN_UTILITY_INPROCESS=1`). It creates nothing until
//! the plan, the reference pack, the model's catalog entry and the host all validate; the run
//! record (`run.json`) is written under `--out` on every path, including refusal. Exit codes: 0 on
//! a completed run, 2 when the plan was refused before dispatch, 3 when the run stopped on a limit
//! or a shot failed, 1 on a transport/API/io error.
//!
//! The three edit subcommands (sc-22712) change an assembled sequence without re-rendering
//! anything: they read the run record, edit the SAVED timeline through the same API the editor
//! uses, re-lay the sequence, and write the record back. `--export` re-runs the MP4 export.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sceneworks_core::film_plan::RunOutcome;
use sceneworks_rust_api::film_harness::{
    self, EditOptions, HarnessError, HttpTransport, RunOptions, TimelineEdit, FIXTURE_REFERENCES,
    FIXTURE_SOUNDS,
};

const USAGE: &str = "\
film-harness — render a hand-authored production plan into a SceneWorks sequence

USAGE:
  film-harness validate --plan PLAN.json --references REFERENCES.json [--api URL] [--shots IDS]
  film-harness run      --plan PLAN.json --references REFERENCES.json [OPTIONS]
  film-harness trim         --run RUN.json --shot ID [--source-in S] [--source-out S] [EDIT OPTIONS]
  film-harness reorder      --run RUN.json --order A,B,C                              [EDIT OPTIONS]
  film-harness replace-take --run RUN.json --shot ID --asset ASSET_ID                 [EDIT OPTIONS]
  film-harness fixture-images --out DIR
  film-harness fixture-sound  --out DIR

OPTIONS (run):
  --api URL              SceneWorks API base URL (default http://127.0.0.1:8000, or $SCENEWORKS_API_URL)
  --token TOKEN          API token (default $SCENEWORKS_ACCESS_TOKEN; sent as X-SceneWorks-Token)
  --shots A,B            Render only these shot ids, in plan order (default: every shot)
  --project-id ID        Reuse an existing project instead of creating one named after the plan
  --out DIR              Run record directory (default film-harness-runs/<utc-timestamp>)
  --poll-seconds N       Job polling cadence in seconds (default 5)
  --no-export            Skip the timeline assembly and MP4 export
  --skip-install-check   Do not refuse a model/tier the catalog reports as not installed

EDIT OPTIONS (trim / reorder / replace-take):
  --run RUN.json         The run record to edit. It names the project and the timeline, and is
                         rewritten in place with the edited sequence.
  --api URL / --token    As above
  --export               Re-export the MP4 after the edit (default: edit the timeline only)
  --poll-seconds N       Job polling cadence in seconds (default 5)

Every edit re-lays the whole sequence: picture items stay contiguous in cut order, each dialogue
clip keeps its offset from the start of its own shot, and the ambience/music beds re-span the new
duration without restarting at any cut.
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
        "fixture-sound" => return fixture_sound(&args[1..]),
        "trim" | "reorder" | "replace-take" => return edit(command, &args[1..]).await,
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

/// `trim` / `reorder` / `replace-take` — edit an assembled sequence in place (sc-22712).
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
                    "film-harness: replace-take needs --shot ID and --asset ASSET_ID\n\n{USAGE}"
                );
                return ExitCode::from(1);
            };
            TimelineEdit::ReplaceTake { shot_id, asset_id }
        }
    };

    let transport = match HttpTransport::new(&api_url, token) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("film-harness: {error}");
            return ExitCode::from(1);
        }
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
