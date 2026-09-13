//! `film-harness` — render a hand-authored production plan into a SceneWorks sequence through a
//! running SceneWorks API (epic 22708, sc-22710).
//!
//! ```text
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
//! film-harness fixture-images --out DIR
//! ```
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

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use sceneworks_core::film_plan::{RunOutcome, RunRecord, RunState};
use sceneworks_core::film_review::{format_eval_report, ASSISTIVE_NOTICE};
use sceneworks_rust_api::film_harness::review::{
    self, Decision, EvalOptions, ReviewOptions, ScriptedVision, VqaVision,
};
use sceneworks_rust_api::film_harness::{
    self, HarnessError, HttpTransport, ResumeOptions, RunControl, RunOptions, FIXTURE_REFERENCES,
};

const USAGE: &str = "\
film-harness — render a hand-authored production plan into a SceneWorks sequence

USAGE:
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

resume       picks a run up from its record: finished takes are reused, jobs still in flight are
             adopted, and what is left of the plan's wall-clock budget and per-shot attempt cap is
             what bounds it. The selected take of every shot is left alone.
replace-take rejects the take a shot is carrying and renders exactly ONE more for that shot, with
             --reason recorded beside it. Other shots' takes, jobs and assets are untouched; shots
             that declared a dependency on it are flagged needs_review, never re-rendered. Without
             --export the existing export is only marked stale.
cancel       asks a run in another shell to stop; status prints what a record says without touching
             the API.

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
        "validate" | "run" => {}
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
    // The run watches its OWN directory too, so `film-harness cancel --out DIR` from another shell
    // reaches it without a shared handle (sc-22711).
    let control = parsed.control.clone();
    let signal = spawn_interrupt_handler(control.clone());
    let outcome = film_harness::run_with_control(&transport, &parsed.options, &control).await;
    signal.abort();
    match outcome {
        Ok(record) => {
            print_record(&record, &parsed.options.out_dir);
            exit_code_for(&record)
        }
        Err(error) => report_error(error),
    }
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
    let control = RunControl::watching(&out_dir);
    film_harness::clear_cancel_request(&out_dir).map_err(|error| error.to_string())?;
    Ok(Parsed {
        api_url,
        token,
        control,
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

/// Cancel the in-flight run on SIGINT, and exit on a second one.
///
/// The second listener is not optional: once `ctrl_c()` has been awaited, tokio owns SIGINT for the
/// rest of the process, so without it a second Ctrl-C would be swallowed and the operator would
/// have no way out but another signal.
fn spawn_interrupt_handler(control: RunControl) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!(
                "film-harness: interrupt received — canceling the in-flight job and writing the \
                 run record; interrupt again to exit now (the render keeps going)"
            );
            control.cancel();
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!(
                    "film-harness: second interrupt — exiting without a run record; the worker \
                     may still be rendering (cancel it in the job list)"
                );
                std::process::exit(130);
            }
        }
    })
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
    let transport = match HttpTransport::new(&parsed.api_url, parsed.token.clone()) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("film-harness: {error}");
            return ExitCode::from(1);
        }
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
    signal.abort();
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
    let transport = match HttpTransport::new(&api_url, token) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("film-harness: {error}");
            return ExitCode::from(1);
        }
    };
    let signal = spawn_interrupt_handler(options.control.clone());
    let vision = VqaVision::new(&transport, options.poll_interval, options.control.clone());
    if let Err(error) = vision.preflight().await {
        signal.abort();
        return report_error(error);
    }
    let result = review::review(&transport, &options, &vision).await;
    signal.abort();
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
    let transport = match HttpTransport::new(&api_url, token) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("film-harness: {error}");
            return ExitCode::from(1);
        }
    };
    let signal = spawn_interrupt_handler(options.control.clone());
    let scripted_backend = ScriptedVision::new();
    let vqa_backend = VqaVision::new(&transport, options.poll_interval, options.control.clone());
    if !scripted {
        if let Err(error) = vqa_backend.preflight().await {
            signal.abort();
            return report_error(error);
        }
    }
    let vision: &dyn review::ReviewVision = if scripted {
        &scripted_backend
    } else {
        &vqa_backend
    };
    let result = review::review_eval(&transport, &options, vision).await;
    signal.abort();
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
        Err(error) => {
            eprintln!("film-harness: {error}");
            ExitCode::from(1)
        }
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
