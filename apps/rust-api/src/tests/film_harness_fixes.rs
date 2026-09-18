//! Feature-end fixes for the local filmmaking harness (epic 22708, sc-22715): one test per finding
//! of the feature-end review, each written so that it FAILS on the code the finding was about.
//! Same fixtures and fake worker as `film_harness.rs`; the real routes in-process throughout.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sceneworks_core::film_plan::{RunOutcome, RunRecord, RunState, ShotOutcome};
use serde_json::{json, Value};

use crate::film_harness::review::{self, Decision};
use crate::film_harness::{
    self, ControllerLease, EditOptions, HarnessError, ResumeOptions, RunControl, RunOptions,
    TimelineEdit,
};
use crate::film_planner;
use crate::tests::film_harness::{
    close, draft_text, fast, findings_of, full_draft, harness_record, items_of, planner_llm,
    planner_options, saved_timeline, set_plan_replies, summary, Harness, RunningHook,
    VideoBehavior, BRIEF_FIXTURE, FAKE_REFINE_PEAK_BYTES, FIXTURE_DIR,
};
use crate::tests::support::request;

fn edit_options(harness: &Harness, export: bool) -> EditOptions {
    EditOptions {
        run_record_path: harness.out_dir().join("run.json"),
        export,
        poll_interval: Duration::from_millis(250),
    }
}

fn picture_order(saved: &Value) -> Vec<String> {
    items_of(saved, "track_main")
        .iter()
        .map(|item| {
            item["filmHarness"]["shotId"]
                .as_str()
                .expect("a harness picture item names its shot")
                .to_owned()
        })
        .collect()
}

fn picture_item<'a>(saved: &'a Value, shot_id: &str) -> &'a Value {
    items_of(saved, "track_main")
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!(shot_id))
        .unwrap_or_else(|| panic!("{shot_id} is on the saved picture track"))
}

fn history_sources(item: &Value) -> Vec<&str> {
    item["versionHistory"]
        .as_array()
        .expect("version history")
        .iter()
        .filter_map(|entry| entry["source"].as_str())
        .collect()
}

fn edit_kinds(record: &RunRecord) -> Vec<String> {
    record
        .timeline
        .as_ref()
        .expect("timeline")
        .edits
        .iter()
        .map(|edit| edit.kind.clone())
        .collect()
}

fn selected_asset(record: &RunRecord, shot_id: &str) -> String {
    record
        .shot(shot_id)
        .and_then(|shot| shot.selected())
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone())
        .unwrap_or_else(|| panic!("{shot_id} has a selected take"))
}

/// Make the record on disk look like a controller that died mid-run: `running`, no stop.
fn simulate_crash(harness: &Harness) {
    harness.edit_run_record(|record| {
        record["state"] = json!("running");
        record.as_object_mut().unwrap().remove("stop");
        record.as_object_mut().unwrap().remove("finishedAt");
    });
}

// ---------------------------------------------------------------------------------------------
// [blocker] AT2/E4/E2 — re-assembly merges into the saved sequence instead of rebuilding it
// ---------------------------------------------------------------------------------------------

/// The probe from the feature-end review, as a regression test: run → trim SH010 1.0..3.0 →
/// reorder [SH020, SH010] → `replace-take SH020`. Before the fix the replacement rebuilt every
/// picture item from the selection — order back to plan order, SH010 back to 0.0..5.1667, one
/// `original` history entry — while `timeline.edits` and the decision log still said "trimmed,
/// reordered". Now only SH020's item changes.
#[tokio::test]
async fn a_replacement_merges_into_the_edited_sequence_instead_of_rebuilding_it() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let project_id = record.project_id.clone().expect("project");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let old_sh020 = selected_asset(&record, "SH020");
    let sh010_asset = selected_asset(&record, "SH010");

    film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, false),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(1.0),
            source_out: Some(3.0),
        },
    )
    .await
    .expect("trim applies");
    film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, false),
        TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");

    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "the parcel is the wrong red",
    )
    .await
    .expect("replacement runs");
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));
    let new_sh020 = selected_asset(&after, "SH020");
    assert_ne!(new_sh020, old_sh020);

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(
        picture_order(&saved),
        vec!["SH020", "SH010"],
        "the reorder survived the replacement: {saved}"
    );
    let sh010 = picture_item(&saved, "SH010");
    assert!(
        close(sh010["sourceIn"].as_f64().unwrap(), 1.0)
            && close(sh010["sourceOut"].as_f64().unwrap(), 3.0),
        "the trim survived the replacement: {sh010}"
    );
    assert!(
        close(sh010["timelineStart"].as_f64().unwrap(), 5.1667)
            && close(sh010["timelineEnd"].as_f64().unwrap(), 5.1667 + 2.0),
        "SH010 is still the trimmed 2s after the re-laid SH020: {sh010}"
    );
    assert_eq!(
        sh010["assetId"],
        json!(sh010_asset),
        "SH010's take is untouched"
    );
    assert_eq!(
        history_sources(sh010),
        vec!["original"],
        "an item whose take did not change keeps its history as it was: {sh010}"
    );
    let sh020 = picture_item(&saved, "SH020");
    assert_eq!(sh020["assetId"], json!(new_sh020));
    assert_eq!(sh020["currentVersionAssetId"], json!(new_sh020));
    assert_eq!(
        history_sources(sh020),
        vec!["original", "replacement"],
        "the replaced item's history GROWS rather than restarting: {sh020}"
    );
    assert_eq!(
        sh020["versionHistory"][0]["assetId"],
        json!(old_sh020),
        "the rejected take stays addressable from the item: {sh020}"
    );
    assert!(
        close(sh020["sourceIn"].as_f64().unwrap(), 0.0)
            && close(sh020["sourceOut"].as_f64().unwrap(), 5.1667),
        "the replaced item's source range is reset to the new take: {sh020}"
    );
    assert_eq!(
        edit_kinds(&after),
        vec!["trim", "reorder"],
        "the edits are still recorded, and the sequence still honours them"
    );
    let recorded_order: Vec<&str> = after
        .timeline
        .as_ref()
        .unwrap()
        .items
        .iter()
        .filter_map(|item| item.shot_id.as_deref())
        .collect();
    assert_eq!(recorded_order, vec!["SH020", "SH010"]);
    assert!(
        after.export.as_ref().is_some_and(|export| export.stale),
        "without --export the existing MP4 is flagged stale"
    );

    // A `swap-take` onto a FOREIGN asset (an imported plate, not a take of this run) used to be
    // reverted to the old take by the next replacement. It stays.
    let plate = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("plate imported")
        .asset_id
        .clone();
    film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, false),
        TimelineEdit::SwapTake {
            shot_id: "SH010".to_owned(),
            asset_id: plate.clone(),
        },
    )
    .await
    .expect("swap applies");
    let again = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "still the wrong red",
    )
    .await
    .expect("second replacement runs");
    assert_eq!(again.outcome, RunOutcome::Completed, "{}", summary(&again));
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let sh010 = picture_item(&saved, "SH010");
    assert_eq!(
        sh010["assetId"],
        json!(plate),
        "a swap onto a foreign asset survives a later replacement: {sh010}"
    );
    assert_eq!(history_sources(sh010), vec!["original", "replacement"]);
    assert_eq!(
        history_sources(picture_item(&saved, "SH020")),
        vec!["original", "replacement", "replacement"]
    );
    assert_eq!(picture_order(&saved), vec!["SH020", "SH010"]);
    assert_eq!(edit_kinds(&again), vec!["trim", "reorder", "swap_take"]);
}

/// The other path into `assemble_timeline`: a `resume` after an edit. A run whose export failed
/// is edited (trim, reorder) and then resumed to redo the export — the re-assembly must keep the
/// edited sequence, and the re-export must be of THAT sequence.
#[tokio::test]
async fn a_resume_after_an_edit_keeps_the_edited_sequence() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    harness.script.lock().export_fails = true;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a failed export still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    assert_eq!(
        record.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("export_failed")
    );
    let project_id = record.project_id.clone().expect("project");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();

    film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, false),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(1.0),
            source_out: Some(3.0),
        },
    )
    .await
    .expect("trim applies");
    film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, false),
        TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");

    harness.script.lock().export_fails = false;
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    let export = resumed.export.as_ref().expect("export");
    assert_eq!(export.status, "completed");
    assert!(!export.stale);

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(picture_order(&saved), vec!["SH020", "SH010"], "{saved}");
    let sh010 = picture_item(&saved, "SH010");
    assert!(
        close(sh010["sourceIn"].as_f64().unwrap(), 1.0)
            && close(sh010["sourceOut"].as_f64().unwrap(), 3.0),
        "the trim survived the resume: {sh010}"
    );
    assert!(close(saved["duration"].as_f64().unwrap(), 5.1667 + 2.0));
    for shot_id in ["SH010", "SH020"] {
        assert_eq!(
            history_sources(picture_item(&saved, shot_id)),
            vec!["original"]
        );
    }
    assert_eq!(edit_kinds(&resumed), vec!["trim", "reorder"]);
    assert!(
        close(
            resumed.timeline.as_ref().unwrap().duration_seconds,
            5.1667 + 2.0
        ),
        "the record describes the edited sequence, not a rebuilt one"
    );
}

// ---------------------------------------------------------------------------------------------
// [major] E4/E5 — a replacement runs outside the run's automatic budgets
// ---------------------------------------------------------------------------------------------

/// A `replace-take` is charged to `humanRequestedElapsedSeconds`, never to `elapsedSeconds`: the
/// probe in the review saw 4.28 s → 5.76 s after one replacement, and a run with less left than
/// the replacement took then refused its own `resume` with "raise limits.maxRunSeconds".
///
/// The run here stopped on a failed export (resumable), and the replacement does not re-export —
/// so the `export_failed` stop must survive the replacement too, or there is nothing to resume.
#[tokio::test]
async fn a_replacement_is_booked_as_human_time_and_leaves_the_run_budget_alone() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    harness.script.lock().export_fails = true;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a failed export still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    assert_eq!(
        record.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("export_failed")
    );
    assert_eq!(record.human_requested_elapsed_seconds, 0.0);

    // Book the run as having five seconds of its budget left, as a long first controller would.
    let booked = 595.0;
    harness.edit_run_record(|record| record["elapsedSeconds"] = json!(booked));
    // The replacement takes longer than what is left.
    harness.script.lock().behaviors.push((
        "SH010".to_owned(),
        VideoBehavior::Complete {
            delay_secs: 6,
            peak_pct: 40.0,
        },
    ));
    harness.script.lock().behaviors.rotate_right(1);
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "prefer another",
    )
    .await
    .expect("a replacement is not bounded by what is left of the run budget");
    assert_eq!(after.shot("SH010").unwrap().selected_attempt, Some(2));
    assert_eq!(
        after.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("export_failed"),
        "a replacement that did not re-export leaves the export still owed: {}",
        summary(&after)
    );
    assert!(after.is_resumable());

    let on_disk = harness_record(&harness);
    assert!(
        close(on_disk.elapsed_seconds, booked),
        "the run's automatic wall-clock is untouched by a human-requested attempt: {}",
        on_disk.elapsed_seconds
    );
    assert!(
        on_disk.human_requested_elapsed_seconds >= 6.0,
        "the replacement is booked as human-requested time: {}",
        on_disk.human_requested_elapsed_seconds
    );

    // The run's own `resume` is still admitted: it has the five seconds it had before, and the
    // export it owes fits in them.
    harness.script.lock().export_fails = false;
    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("the resume is not refused for a budget the replacement did not spend");
    assert!(
        resumed
            .decisions
            .iter()
            .any(|decision| decision.action == "resume"
                && decision
                    .detail
                    .contains("5s of the plan's 600s budget left")),
        "{:#?}",
        resumed.decisions
    );
    assert_ne!(
        resumed.outcome,
        RunOutcome::StoppedRunBudget,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert!(
        harness_record(&harness).human_requested_elapsed_seconds >= 6.0,
        "a resume inherits the human-requested spend rather than resetting it"
    );
}

/// A replacement's `--export` is bounded by the export's own per-job budget and NOT by a
/// synthetic run deadline (`started + maxShotSeconds`, which the render had already spent most
/// of): an overrun is `export_failed` / resumable, never `run_budget` / terminal.
#[tokio::test]
async fn a_replacements_export_overrun_is_export_failed_and_resumable_not_run_budget() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 4, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, Some(&["SH010"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );

    harness.script.lock().export_hangs = true;
    let after = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "again, with export",
    )
    .await
    .expect("a replacement whose export overruns still returns its record");
    assert_eq!(after.outcome, RunOutcome::Failed, "{}", summary(&after));
    let stop = after.stop.as_ref().expect("a stop reason");
    assert_eq!(
        stop.reason, "export_failed",
        "an export overrun is an export failure, not the run's budget: {stop:?}"
    );
    assert!(stop.resumable, "{stop:?}");
    let export = after.export.as_ref().expect("export recorded");
    assert_eq!(export.status, "timed_out");
    assert!(
        export
            .error
            .as_deref()
            .is_some_and(|error| error.contains("per-job budget of 4s")),
        "{export:?}"
    );
    assert_eq!(after.shot("SH010").unwrap().selected_attempt, Some(2));
    assert!(after.human_requested_elapsed_seconds >= 4.0);
    assert!(after.is_resumable());
}

// ---------------------------------------------------------------------------------------------
// [major] E5/E4 — every preflight counts LIVE workers only
// ---------------------------------------------------------------------------------------------

/// A stale `offline` row still advertising `video_generate` / `timeline_export` / `prompt_refine`
/// fooled the run and planner preflights (the review preflight had already learned this). Now one
/// shared rule refuses all three, naming the row.
#[tokio::test]
async fn every_preflight_ignores_a_stale_worker_row_that_still_advertises_the_capability() {
    let harness = Harness::start(false, Vec::new()).await;
    let (status, _) = request(
        harness.app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "stale-gpu",
            "gpuId": "mlx",
            "gpuName": "Apple M-series (stale)",
            "capabilities": ["video_generate", "timeline_export", "prompt_refine"],
            "loadedModels": [],
            "utilization": { "memoryTotalMb": 128 * 1024 },
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let heartbeat = |worker_status: &'static str| {
        let app = harness.app.clone();
        async move {
            let (status, _) = request(
                app,
                "POST",
                "/api/v1/workers/stale-gpu/heartbeat",
                json!({
                    "status": worker_status, "loadedModels": [],
                    "utilization": { "memoryTotalMb": 128 * 1024 },
                }),
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::OK);
        }
    };
    heartbeat("offline").await;

    // The run preflight: both the render worker and the export worker.
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        None,
    );
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &options)
            .await
            .expect_err("an offline worker cannot render or export"),
    );
    assert!(
        findings.iter().any(|finding| finding
            .contains("no live registered worker advertises video_generate")
            && finding.contains("stale-gpu (offline)")),
        "{findings:?}"
    );
    assert!(
        findings.iter().any(|finding| finding
            .contains("no live registered worker advertises timeline_export")
            && finding.contains("stale-gpu (offline)")),
        "{findings:?}"
    );

    // The planner preflight.
    let findings = findings_of(
        film_planner::generate(
            &harness.transport,
            &planner_llm(&harness),
            &planner_options(&harness, "stale"),
        )
        .await
        .expect_err("an offline worker cannot plan"),
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("no live registered worker advertises prompt_refine")
            && findings[0].contains("stale-gpu (offline)"),
        "{findings:?}"
    );
    assert!(harness.jobs().await.is_empty(), "nothing was queued");

    // The same row, live, passes the run preflight.
    heartbeat("idle").await;
    film_harness::validate(Some(&harness.transport), &options)
        .await
        .expect("an idle worker advertising both is accepted");
}

// ---------------------------------------------------------------------------------------------
// [minor] E3 — dropped audio layers reach the run record
// ---------------------------------------------------------------------------------------------

/// The layers a `timeline_export` mixed WITHOUT are in its result and copied into `run.json`'s
/// export entry, so "why is the music missing" is answerable from the record. The real worker's
/// `droppedAudioLayers` is exercised in `a_real_timeline_export_mixes_the_harness_four_track_sequence`;
/// this proves the record side against the fake.
#[tokio::test]
async fn the_run_record_carries_the_audio_layers_the_export_dropped() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let dropped = json!({
        "assetId": "asset_gone",
        "trackId": "track_music",
        "role": "music",
        "generated": false,
        "reason": "asset_missing",
    });
    harness.script.lock().export_dropped_layers = vec![dropped.clone()];
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let export = record.export.as_ref().expect("export");
    assert_eq!(export.dropped_audio_layers, vec![dropped.clone()]);
    let on_disk = harness.run_record();
    assert_eq!(on_disk["export"]["droppedAudioLayers"], json!([dropped]));
    assert_eq!(on_disk["export"]["status"], "completed");
}

#[tokio::test]
async fn an_explicit_export_records_the_exact_job_and_never_redispatches_while_running() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let mut run_options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    run_options.export = false;
    let record = film_harness::run(&harness.transport, &run_options)
        .await
        .expect("shot run assembles without exporting");
    assert!(record.timeline.is_some());
    assert!(
        record.export.is_none(),
        "shot rendering never exports implicitly"
    );

    let mut options = ResumeOptions::new(harness.out_dir());
    options.poll_interval = Duration::from_millis(25);
    let (running, task) = film_harness::start_explicit_export(&harness.transport, &options)
        .await
        .expect("explicit export starts");
    assert_eq!(running.export.as_ref().unwrap().job_id, task.job_id);
    assert_eq!(running.export.as_ref().unwrap().status, "running");
    assert_eq!(
        harness
            .jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "timeline_export")
            .count(),
        1
    );

    let (adopted, same_task) = film_harness::start_explicit_export(&harness.transport, &options)
        .await
        .expect("second controller adopts the running export");
    assert_eq!(same_task.job_id, task.job_id);
    assert_eq!(adopted.export.as_ref().unwrap().job_id, task.job_id);
    assert_eq!(
        harness
            .jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "timeline_export")
            .count(),
        1,
        "adoption must not create a second job"
    );

    let completed = film_harness::finish_explicit_export(&harness.transport, &options, &task)
        .await
        .expect("explicit export settles");
    let export = completed.export.as_ref().unwrap();
    assert_eq!(export.status, "completed");
    assert!(export.asset_id.is_some());
    assert!(!export.stale);
}

#[tokio::test]
async fn a_canceled_incremental_run_can_export_its_saved_cut_without_rendering_more_shots() {
    let harness = Harness::start(
        true,
        vec![
            (
                "SH010",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            ),
            ("SH020", VideoBehavior::Hang),
        ],
    )
    .await;
    let mut run_options = harness.options(
        harness.edited_plan(|plan| {
            plan["limits"] = json!({
                "maxRunSeconds": 600, "maxShotSeconds": 600,
                "maxAttemptsPerShot": 2, "maxMemoryGb": 96
            });
        }),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    run_options.export = false;
    let control = RunControl::watching(&harness.out_dir());
    harness.script.lock().running_hook = Some((
        "SH020".to_owned(),
        RunningHook::WriteCancelSentinel(harness.out_dir()),
    ));
    let canceled = film_harness::run_with_control(&harness.transport, &run_options, &control)
        .await
        .expect("the canceled run keeps its record");
    assert_eq!(
        canceled.outcome,
        RunOutcome::Canceled,
        "{}",
        summary(&canceled)
    );
    assert!(
        canceled.timeline.is_some(),
        "SH010 produced an incremental cut"
    );
    let video_jobs = harness.api_video_job_count().await;

    let _lease = ControllerLease::acquire_new_action(&harness.out_dir(), "api:export")
        .expect("the separately authorized export acquires the run");
    let mut options = ResumeOptions::new(harness.out_dir());
    options.poll_interval = Duration::from_millis(25);
    let (_, task) = film_harness::start_explicit_export(&harness.transport, &options)
        .await
        .expect("the saved cut export starts");
    let exported = film_harness::finish_explicit_export(&harness.transport, &options, &task)
        .await
        .expect("the saved cut export settles");
    assert_eq!(exported.export.as_ref().unwrap().status, "completed");
    assert_eq!(
        harness.api_video_job_count().await,
        video_jobs,
        "exporting the retained cut must not render another shot"
    );
}

#[tokio::test]
async fn a_fresh_export_cancel_is_honored_and_an_explicit_retry_dispatches_exactly_once() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let mut run_options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    run_options.export = false;
    film_harness::run(&harness.transport, &run_options)
        .await
        .expect("shot run assembles without exporting");
    harness.script.lock().export_hangs = true;

    let lease = ControllerLease::acquire_new_action(&harness.out_dir(), "api:export:first")
        .expect("first export acquires");
    let mut first_options = ResumeOptions::new(harness.out_dir());
    first_options.poll_interval = Duration::from_millis(25);
    let (_, first_task) = film_harness::start_explicit_export(&harness.transport, &first_options)
        .await
        .expect("first export starts");
    film_harness::request_cancel(&harness.out_dir())
        .expect("a fresh cancel reaches the active export");
    let canceled =
        film_harness::finish_explicit_export(&harness.transport, &first_options, &first_task)
            .await
            .expect("freshly canceled export settles");
    assert_eq!(
        canceled.export.as_ref().unwrap().status,
        "canceled_by_operator",
        "a cancel arriving after the new action begins remains effective"
    );
    drop(lease);

    harness.script.lock().export_hangs = false;
    let _retry_lease = ControllerLease::acquire_new_action(&harness.out_dir(), "api:export:retry")
        .expect("retry acquires");
    let mut retry_options = ResumeOptions::new(harness.out_dir());
    retry_options.poll_interval = Duration::from_millis(25);
    let (_, retry_task) = film_harness::start_explicit_export(&harness.transport, &retry_options)
        .await
        .expect("retry starts");
    assert_ne!(retry_task.job_id, first_task.job_id);
    let retried =
        film_harness::finish_explicit_export(&harness.transport, &retry_options, &retry_task)
            .await
            .expect("retry settles");
    assert_eq!(retried.export.as_ref().unwrap().status, "completed");
    assert_eq!(
        harness
            .jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "timeline_export")
            .count(),
        2,
        "the retry adds exactly one export job"
    );
}

#[tokio::test]
async fn a_supported_sfx_bed_is_imported_with_provenance_and_placed_on_an_editable_track() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let plan = harness.edited_plan(|plan| {
        plan["sound"] = json!({
            "generatedAudio": "mute",
            "dialogue": {"gain": 1.0, "muted": false},
            "sfx": [{
                "role": "door_close", "gain": 0.65, "muted": false,
                "startSeconds": 0.5, "sourceInSeconds": 0.1,
                "fadeInSeconds": 0.05, "fadeOutSeconds": 0.1
            }]
        });
    });
    let pack = harness.edited_pack(|pack| {
        pack["sound"] = json!([{
            "role": "door_close", "kind": "sfx",
            "file": "sound/door_close.wav", "description": "Door close"
        }]);
    });
    let pack_dir = pack.parent().unwrap();
    std::fs::create_dir_all(pack_dir.join("sound")).unwrap();
    std::fs::copy(
        Path::new(FIXTURE_DIR).join("sound/workshop_room_tone.wav"),
        pack_dir.join("sound/door_close.wav"),
    )
    .unwrap();
    let mut options = harness.options(plan, pack, Some(&["SH010"]));
    options.export = false;
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("sfx film run completes");
    assert!(record
        .sound
        .iter()
        .any(|sound| sound.role == "door_close" && sound.kind == "sfx"));
    let track = record
        .timeline
        .as_ref()
        .unwrap()
        .tracks
        .iter()
        .find(|track| track.role == "sfx")
        .expect("sfx track");
    assert_eq!(track.track_id, "track_sfx_0");
    assert_eq!(track.gain, 0.65);
    assert_eq!(track.items.len(), 1);
    assert_eq!(track.items[0].timeline_start, 0.5);
    assert!(
        record.export.is_none(),
        "audio placement never implicitly exports"
    );
}

// ---------------------------------------------------------------------------------------------
// [minor] E2 — an edit persists like every other controller and leaves the run's stop alone
// ---------------------------------------------------------------------------------------------

/// `edit_timeline` wrote `run.json` with a plain `fs::write`, never refreshed the project's copy,
/// and on `--export` overwrote `outcome` regardless of the prior stop. Now it goes through
/// `persist_record` and only an `export_failed` stop (which the re-export resolves) moves.
#[tokio::test]
async fn an_edit_persists_atomically_mirrors_the_projects_copy_and_leaves_the_runs_stop_alone() {
    // A run that stopped for a reason a trim does not change: SH020 exhausted its one attempt.
    let harness = Harness::start(
        true,
        vec![
            (
                "SH010",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            ),
            ("SH020", VideoBehavior::FailAlways),
        ],
    )
    .await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a run with a failed shot still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    assert_eq!(
        record.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("attempts_exhausted")
    );
    assert!(record
        .export
        .as_ref()
        .is_some_and(|export| export.status == "completed"));
    let project_path = PathBuf::from(record.project_path.as_deref().expect("project path"));
    let mirror = project_path
        .join("film-harness")
        .join(&record.run_id)
        .join("run.json");
    assert!(mirror.is_file(), "the run writes the project's copy");

    let edited = film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, true),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(0.5),
            source_out: Some(2.0),
        },
    )
    .await
    .expect("trim --export applies");
    assert_eq!(
        edited.outcome,
        RunOutcome::Failed,
        "a successful re-export does not re-classify a run that stopped on attempts_exhausted"
    );
    assert_eq!(
        edited.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("attempts_exhausted")
    );
    assert!(edited
        .export
        .as_ref()
        .is_some_and(|export| export.status == "completed" && !export.stale));
    assert!(
        edited.human_requested_elapsed_seconds > 0.0,
        "an edit's re-export is human-requested time"
    );

    // The project's copy says the same thing as the run directory's, and nothing was left half
    // written: the temp file `write_atomically` renames from is gone.
    let on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(&mirror).unwrap()).expect("mirror parses");
    assert_eq!(on_disk["timeline"]["edits"][0]["kind"], "trim", "{on_disk}");
    assert_eq!(on_disk["stop"]["reason"], "attempts_exhausted");
    assert_eq!(
        on_disk,
        harness.run_record(),
        "the project's copy is the run directory's record"
    );
    assert!(
        !harness.out_dir().join("run.json.tmp").exists(),
        "the atomic write leaves no temp file behind"
    );

    // An `export_failed` stop IS about the export an edit just redid, so a successful re-export
    // clears it; a re-export that fails records `export_failed` (resumable) itself.
    let harness = Harness::start(true, fast(&["SH010"])).await;
    harness.script.lock().export_fails = true;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("record");
    assert_eq!(
        record.stop.as_ref().map(|stop| stop.reason.as_str()),
        Some("export_failed")
    );
    harness.script.lock().export_fails = false;
    let edited = film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, true),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(0.5),
            source_out: None,
        },
    )
    .await
    .expect("trim --export applies");
    assert_eq!(
        edited.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&edited)
    );
    assert!(edited.stop.is_none(), "{:?}", edited.stop);

    harness.script.lock().export_fails = true;
    let edited = film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, true),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(1.0),
            source_out: None,
        },
    )
    .await
    .expect("a failed re-export still returns the record");
    assert_eq!(edited.outcome, RunOutcome::Failed, "{}", summary(&edited));
    let stop = edited.stop.as_ref().expect("stop");
    assert_eq!(stop.reason, "export_failed");
    assert!(stop.resumable);
    assert_eq!(
        edited.export.as_ref().map(|export| export.status.as_str()),
        Some("failed")
    );
}

// ---------------------------------------------------------------------------------------------
// TERMINAL_READINESS (a) — the LTX-2.5 menus are exercised before the first LTX plan is
// ---------------------------------------------------------------------------------------------

/// The checked-in LTX-2.5 plan validates against the LIVE catalog entry, and an off-menu fps or
/// duration for LTX is refused by name — so the first real LTX run is not the first time those
/// menus are read.
#[tokio::test]
async fn the_ltx_2_5_plan_validates_against_the_live_catalog_and_off_menu_values_are_refused_by_name(
) {
    let harness = Harness::start(true, Vec::new()).await;
    let plan_path = Path::new(FIXTURE_DIR).join("plan.ltx25.jsonc");
    let options = harness.options(plan_path.clone(), harness.fixture_pack(), None);
    let (plan, _) = film_harness::validate(Some(&harness.transport), &options)
        .await
        .expect("the LTX-2.5 plan validates against the live catalog");
    assert_eq!(plan.model.id, "ltx_2_5");
    assert_eq!(plan.model.tier.as_deref(), Some("q4"));
    assert_eq!(plan.model.fps, Some(25));
    assert_eq!(plan.shots.len(), 6);
    assert!(plan
        .shots
        .iter()
        .all(|shot| close(shot.target_duration_seconds, 6.0)));
    // Same beats, same roles, same sound as the H3 plan — only the model block differs.
    let h3 = sceneworks_core::film_plan::read_plan_file(&Path::new(FIXTURE_DIR).join("plan.jsonc"))
        .expect("h3 plan reads");
    let ids = |plan: &sceneworks_core::film_plan::ProductionPlan| {
        plan.shots
            .iter()
            .map(|shot| shot.id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&plan), ids(&h3));
    for (ltx, h3) in plan.shots.iter().zip(&h3.shots) {
        assert_eq!(ltx.continuity_roles, h3.continuity_roles, "{}", ltx.id);
        assert_eq!(
            ltx.dialogue_clip.is_some(),
            h3.dialogue_clip.is_some(),
            "{}",
            ltx.id
        );
    }

    let text = std::fs::read_to_string(&plan_path).unwrap();
    let mut off_fps: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text)).unwrap();
    off_fps["model"]["fps"] = json!(23);
    let off_fps_path = harness.temp_dir.path().join("ltx-off-fps.json");
    std::fs::write(
        &off_fps_path,
        serde_json::to_string_pretty(&off_fps).unwrap(),
    )
    .unwrap();
    let mut options = harness.options(off_fps_path, harness.fixture_pack(), None);
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &options)
            .await
            .expect_err("23 fps is off LTX-2.5's menu"),
    );
    assert!(
        findings.iter().any(|finding| finding.contains("model.fps")
            && finding.contains("ltx_2_5")
            && finding.contains("23 fps")),
        "{findings:?}"
    );

    let mut off_duration: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text)).unwrap();
    off_duration["shots"][2]["targetDurationSeconds"] = json!(5.1667);
    let off_duration_path = harness.temp_dir.path().join("ltx-off-duration.json");
    std::fs::write(
        &off_duration_path,
        serde_json::to_string_pretty(&off_duration).unwrap(),
    )
    .unwrap();
    options.plan_path = off_duration_path;
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &options)
            .await
            .expect_err("H3's 5.1667s is off LTX-2.5's menu"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("[SH030] targetDurationSeconds")
                && finding.contains("ltx_2_5")
                && finding.contains("menu")),
        "{findings:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// (A) resume refuses a changed reference pack and changed compiled requests
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn resume_refuses_a_reference_pack_that_changed_under_it() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan_path = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 1, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
        });
    });
    let pack_path = harness.fixture_pack_without_sound();
    let options = harness.options(plan_path, pack_path.clone(), Some(&["SH010"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("record");
    assert_ne!(record.outcome, RunOutcome::Completed);

    let text = std::fs::read_to_string(&pack_path).unwrap();
    let mut pack: Value = serde_json::from_str(&text).unwrap();
    pack["description"] = json!("edited after the run started");
    std::fs::write(&pack_path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
    std::fs::remove_file(harness.out_dir().join("references.json")).unwrap();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("an edited pack is a new run");
    let message = format!("{error}");
    assert!(matches!(error, HarnessError::Refused(_)), "{error}");
    assert!(
        message.contains("the reference pack changed since run")
            && message.contains("a changed reference pack is a new run, not a resume"),
        "{message}"
    );
}

#[tokio::test]
async fn resume_refuses_compiled_requests_that_changed_under_it() {
    let harness = Harness::start(true, vec![]).await;
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(2);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(2);
    brief["targetTotalSeconds"] = json!({ "min": 10.0, "max": 12.0 });
    let brief_path = harness.temp_dir.path().join("two-beat-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "compiled-hash");
    options.brief_path = brief_path;
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        // The harness's own copy, never the checked-in directory: a run writes its synthesized
        // clips beside the pack (sc-23404).
        reference_pack_path: harness.fixture_pack(),
        compiled_path: None,
        project_id: None,
        shot_ids: None,
        out_dir: harness.out_dir(),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let record = film_harness::run(&harness.transport, &run_options)
        .await
        .expect("the generated plan runs");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let recorded_sha = record
        .compiled
        .as_ref()
        .expect("compiled recorded")
        .sha256
        .clone();

    // Edit the compiled document the run dispatched from, and remove the run's own copy so the
    // SOURCE is what gets re-read.
    let text = std::fs::read_to_string(&artifacts.compiled_path).unwrap();
    let mut compiled: Value = serde_json::from_str(&text).unwrap();
    compiled["requests"][0]["prompt"] = json!("an entirely different prompt");
    std::fs::write(
        &artifacts.compiled_path,
        serde_json::to_string_pretty(&compiled).unwrap(),
    )
    .unwrap();
    std::fs::remove_file(harness.out_dir().join("compiled.json")).unwrap();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("edited compiled requests are a new run");
    let message = format!("{error}");
    assert!(matches!(error, HarnessError::Refused(_)), "{error}");
    assert!(
        message.contains("the compiled requests changed since run")
            && message.contains(&format!("recorded {recorded_sha}, found "))
            && message.contains("recompile and start a new run"),
        "{message}"
    );
    assert!(
        !message.contains("  "),
        "the refusal reads as one line, not a botched continuation: {message:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// (D) `plan --force`
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn plan_refuses_to_overwrite_a_differing_plan_json_unless_forced() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let mut options = planner_options(&harness, "forced");
    let first = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the first plan writes");
    let generated = std::fs::read_to_string(&first.plan_path).unwrap();

    // The same plan again is not a conflict.
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("an identical plan.json is simply rewritten");

    // A hand edit is: the file IS the correction surface, and it is never silently replaced.
    let edited = generated.replace(
        "Courier at the Workshop",
        "Courier at the Workshop (edited)",
    );
    assert_ne!(edited, generated);
    std::fs::write(&first.plan_path, &edited).unwrap();
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect_err("a differing plan.json is refused");
    let message = format!("{error}");
    let HarnessError::PlannerExecutionFailure { source, executions } = error else {
        panic!("the refused write must retain the spent planner receipt: {message}");
    };
    assert!(matches!(*source, HarnessError::Io(_)), "{source}");
    assert_eq!(message, source.to_string());
    assert_eq!(executions.len(), 1);
    let execution = &executions[0];
    let plan_job_ids: Vec<String> = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, payload)| kind == "prompt_refine" && payload["task"] == "film_plan")
        .map(|(_, id, _)| id.clone())
        .collect();
    assert_eq!(plan_job_ids.len(), 3);
    assert_eq!(execution.job_id.as_ref(), plan_job_ids.last());
    assert_eq!(execution.provider, "native");
    assert_eq!(execution.model, "fixture/model-keyed-refiner");
    assert_eq!(execution.request_timeout_seconds, Some(30));
    assert!(execution
        .duration_seconds
        .is_some_and(|seconds| seconds >= 0.0));
    assert_eq!(execution.failure_code, None);
    assert!(
        message.contains("already exists and differs") && message.contains("--force"),
        "{message}"
    );
    assert_eq!(
        std::fs::read_to_string(&first.plan_path).unwrap(),
        edited,
        "the refused write left the hand edit exactly as it was"
    );

    // `--force` replaces it.
    options.force = true;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("--force replaces the edited plan");
    assert_eq!(
        std::fs::read_to_string(&first.plan_path).unwrap(),
        generated
    );
}

// ---------------------------------------------------------------------------------------------
// (E) the attempt cap across a resume, and the human attempt that never counts toward it
// ---------------------------------------------------------------------------------------------

/// maxAttemptsPerShot 1, one automatic failure, a crash, a resume: the resume hands out no fresh
/// budget — the shot is not re-dispatched and the run closes `attempts_exhausted`, terminal.
#[tokio::test]
async fn an_attempt_cap_spent_before_a_crash_gives_the_resume_no_fresh_budget() {
    let harness = Harness::start(
        true,
        vec![
            ("SH010", VideoBehavior::FailAlways),
            (
                "SH020",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            ),
        ],
    )
    .await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    assert_eq!(record.shot("SH010").unwrap().attempts.len(), 1);
    assert_eq!(harness.video_job_count(), 2);

    simulate_crash(&harness);
    assert!(harness_record(&harness).is_resumable());
    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("a crashed run is resumed");
    assert_eq!(
        harness.video_job_count(),
        2,
        "the resume dispatched a fresh attempt for a shot whose cap was already spent\n{}",
        summary(&resumed)
    );
    assert_eq!(resumed.shot("SH010").unwrap().attempts.len(), 1);
    assert_eq!(resumed.shot("SH010").unwrap().outcome, ShotOutcome::Failed);
    assert_eq!(resumed.outcome, RunOutcome::Failed, "{}", summary(&resumed));
    let stop = resumed.stop.as_ref().expect("stop");
    assert_eq!(stop.reason, "attempts_exhausted");
    assert!(!stop.resumable);
    assert_eq!(resumed.state, RunState::Finished);
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("terminal");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("attempts_exhausted")),
        "{error}"
    );
}

/// A `replace-take` attempt is a human decision, not a retry: after the automatic attempts have
/// spent the cap it still runs, and it never fills the cap for a LATER automatic retry either way.
/// Two runs with the same history (fail, ok, human replacement, human rejection, crash) differ
/// only in the cap: at 2 the resume dispatches nothing — the two automatic attempts filled it —
/// and at 3 it dispatches one more, which it could not if the human attempt were the third.
#[tokio::test]
async fn a_human_replacement_is_not_counted_against_the_attempt_cap_on_a_later_retry() {
    for (cap, expect_dispatch) in [(2_u32, false), (3_u32, true)] {
        let harness = Harness::start(true, vec![("SH010", VideoBehavior::FailFirst)]).await;
        let (plan, pack) = harness.minimal_documents(json!({
            "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": cap, "maxMemoryGb": 96
        }));
        let options = harness.options(plan, pack, Some(&["SH010"]));
        let record = film_harness::run(&harness.transport, &options)
            .await
            .expect("record");
        assert_eq!(
            record.outcome,
            RunOutcome::Completed,
            "cap {cap}: {}",
            summary(&record)
        );
        assert_eq!(record.shot("SH010").unwrap().automatic_attempts(), 2);
        assert_eq!(harness.video_job_count(), 2);

        let mut resume_options = harness.resume_options();
        resume_options.export = false;
        let replaced = film_harness::replace_take(
            &harness.transport,
            &resume_options,
            "SH010",
            "another, please",
        )
        .await
        .expect("a replacement never respects the cap");
        assert_eq!(replaced.shot("SH010").unwrap().selected_attempt, Some(3));
        assert_eq!(replaced.shot("SH010").unwrap().automatic_attempts(), 2);
        assert_eq!(harness.video_job_count(), 3);

        // The person rejects the replacement too, then the controller dies.
        review::decide_take(&harness.out_dir(), "SH010", Decision::Reject, "no better")
            .expect("rejection records");
        simulate_crash(&harness);
        let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
            .await
            .expect("the crashed run resumes");
        let dispatched = harness.video_job_count() - 3;
        assert_eq!(
            dispatched,
            usize::from(expect_dispatch),
            "cap {cap}: the human attempt must not count toward the automatic cap\n{}",
            summary(&resumed)
        );
        if expect_dispatch {
            assert_eq!(resumed.shot("SH010").unwrap().selected_attempt, Some(4));
            assert_eq!(resumed.shot("SH010").unwrap().automatic_attempts(), 3);
            assert_eq!(
                resumed.outcome,
                RunOutcome::Completed,
                "{}",
                summary(&resumed)
            );
        } else {
            assert_eq!(resumed.shot("SH010").unwrap().selected_attempt, None);
            assert_eq!(
                resumed.stop.as_ref().map(|stop| stop.reason.as_str()),
                Some("attempts_exhausted"),
                "{}",
                summary(&resumed)
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (F) the repair-round ceiling bounds the loop whatever the caller asks
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_repair_loop_ceiling_bounds_a_never_converging_planner_whatever_the_caller_asks() {
    let harness = Harness::start(true, vec![]).await;
    let mut short = full_draft();
    short["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    set_plan_replies(&harness, vec![draft_text(&short)]);
    let mut options = planner_options(&harness, "ceiling");
    options.max_repair_rounds = 99;
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("a draft that never covers the beat is refused"),
    );
    assert_eq!(
        harness.script.lock().plan_calls,
        1 + film_planner::MAX_REPAIR_ROUNDS_CEILING as usize,
        "the draft plus exactly the ceiling's rounds, not the 99 asked for"
    );
    assert!(
        findings.iter().any(|finding| finding.contains(&format!(
            "did not produce a valid plan within {} repair round",
            film_planner::MAX_REPAIR_ROUNDS_CEILING
        ))),
        "{findings:?}"
    );
    let rejected = std::fs::read_to_string(options.out_dir.join("planner-rejected.txt"))
        .expect("the refused answer is written out");
    assert!(
        rejected.contains(&format!(
            "refused after {} round(s) of repair",
            film_planner::MAX_REPAIR_ROUNDS_CEILING
        )),
        "{rejected}"
    );
    assert!(!options.out_dir.join("plan.json").exists());
}

// ---------------------------------------------------------------------------------------------
// (E3/E6) the REAL timeline_export over a harness-assembled four-track sequence
// ---------------------------------------------------------------------------------------------

/// Decode `[start, start + seconds)` of `path` to mono 48 kHz PCM, the way the worker's measured
/// mix tests do.
fn decode_window(path: &Path, start: f64, seconds: f64) -> Vec<f64> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-ss",
            &format!("{start:.3}"),
            "-i",
            &path.display().to_string(),
            "-t",
            &format!("{seconds:.3}"),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "s16le",
            "-",
        ])
        .output()
        .expect("ffmpeg decodes the export");
    output
        .stdout
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f64 / 32768.0)
        .collect()
}

/// Goertzel magnitude of `frequency` in `samples`, normalised by length.
fn tone_level(samples: &[f64], frequency: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let omega = 2.0 * std::f64::consts::PI * frequency / 48_000.0;
    let coefficient = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for sample in samples {
        let s0 = sample + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coefficient * s1 * s2;
    power.max(0.0).sqrt() / samples.len() as f64
}

/// The `Stream #…: Audio:` / `Video:` lines ffmpeg prints for `path`.
fn stream_kinds(path: &Path) -> Vec<String> {
    let output = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-i", &path.display().to_string()])
        .output()
        .expect("ffmpeg probes the export");
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter(|line| line.contains("Stream #"))
        .map(|line| {
            if line.contains(": Video:") {
                "video".to_owned()
            } else if line.contains(": Audio:") {
                "audio".to_owned()
            } else {
                "other".to_owned()
            }
        })
        .collect()
}

/// The REAL `timeline_export` job code path — `run_timeline_export_job` → `render` →
/// `finalize` (two-pass mux) — over the four-track sequence the harness assembled through the real
/// routes, run by the real utility worker loop against this test's in-process API over loopback.
/// The fake worker renders the takes (as REAL clips, each with its own 900 Hz tone) and leaves
/// `timeline_export` to the real worker. Every claim is measured off the exported file's samples.
///
/// Gated exactly as the worker's measured mix tests are: skips without ffmpeg locally, asserts
/// under `SCENEWORKS_REQUIRE_FFMPEG`, which CI sets on the Linux lane.
#[tokio::test]
async fn a_real_timeline_export_mixes_the_harness_four_track_sequence() {
    if !crate::tests::film_harness::ffmpeg_reachable() {
        return;
    }
    let harness = Harness::start(false, Vec::new()).await;
    // The fake renders takes as real clips and does NOT advertise `timeline_export`.
    {
        let mut script = harness.script.lock();
        script.real_takes = true;
        // Everything the harness drives EXCEPT `timeline_export`, which is the one job this test
        // hands to the REAL utility worker. `audio_generate` stays on the fake (sc-23404): the
        // fixture's dialogue is synthesized, and the point here is the real ffmpeg MIX, not a real
        // Kokoro decode.
        script.capabilities = Some(vec![
            "video_generate",
            "frame_extract",
            "image_vqa",
            "prompt_refine",
            "audio_generate",
        ]);
        script.behaviors = fast(&["SH010", "SH020"])
            .into_iter()
            .map(|(id, behavior)| (id.to_owned(), behavior))
            .collect();
    }
    harness.spawn_worker().await;

    // The in-process API on a loopback port, for the real worker's `reqwest` client.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback port");
    let port = listener.local_addr().expect("address").port();
    let app = harness.app.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .expect("the API serves");
    });
    let data_dir = harness.temp_dir.path().join("data");
    let worker_settings = sceneworks_worker::Settings {
        api_url: format!("http://127.0.0.1:{port}"),
        access_token: None,
        data_dir: data_dir.clone(),
        resolved_cache: Default::default(),
        config_dir: harness.temp_dir.path().join("config"),
        worker_id: "real-utility-worker".to_owned(),
        gpu_id: "cpu".to_owned(),
        is_child_worker: true,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 1,
        huggingface_base_url: "http://127.0.0.1:9".to_owned(),
        huggingface_token: None,
        credentials: Vec::new(),
        max_lora_url_bytes: 1 << 30,
        max_model_url_bytes: 1 << 30,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: false,
        backend_candle_enabled: false,
        external_model_roots: Vec::new(),
        gpu_memory_limit_bytes: 0,
    };
    let worker = tokio::spawn(sceneworks_worker::run_worker_loop(worker_settings));
    // Wait for the real worker to register with `timeline_export`, bounded.
    let mut registered = false;
    for _ in 0..400 {
        let (_, workers) =
            request(harness.app.clone(), "GET", "/api/v1/workers", Value::Null).await;
        if workers.as_array().is_some_and(|rows| {
            rows.iter().any(|row| {
                row["id"] == "real-utility-worker"
                    && row["capabilities"]
                        .as_array()
                        .is_some_and(|caps| caps.iter().any(|cap| cap == "timeline_export"))
            })
        }) {
            registered = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(
        registered,
        "the real utility worker did not register within 10s"
    );

    // SH010 + SH020 with the fixture's sound: a line 1.2s into SH020, two beds across the cut.
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let export = record.export.as_ref().expect("export");
    assert_eq!(export.status, "completed", "{export:?}");
    assert!(export.dropped_audio_layers.is_empty(), "{export:?}");
    let project_id = record.project_id.clone().unwrap();
    let (_, project) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}"),
        Value::Null,
    )
    .await;
    let project_path = PathBuf::from(project["path"].as_str().expect("project path"));
    let render_path = project_path.join(export.render_path.as_deref().expect("render path"));
    assert!(render_path.is_file(), "{}", render_path.display());

    // One video stream and one audio stream, from the two-pass mux.
    let kinds = stream_kinds(&render_path);
    assert_eq!(kinds, vec!["video", "audio"], "nb_streams: {kinds:?}");
    let (_, asset) = request(
        harness.app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/assets/{}",
            export.asset_id.as_deref().unwrap()
        ),
        Value::Null,
    )
    .await;
    let total = 2.0 * 5.1667;
    let duration = asset["file"]["duration"].as_f64().expect("render duration");
    assert!(
        (duration - total).abs() < 0.1,
        "the export is exactly the picture's length: {duration} vs {total}"
    );
    assert_eq!(asset["file"]["hasAudio"], true, "{asset}");
    let layers = asset["recipe"]["rawAdapterSettings"]["audioLayers"]
        .as_array()
        .expect("audio layers in the sidecar");
    let roles: Vec<&str> = layers
        .iter()
        .filter_map(|layer| layer["role"].as_str())
        .collect();
    assert!(
        roles.contains(&"dialogue") && roles.contains(&"ambience") && roles.contains(&"music"),
        "{roles:?}"
    );
    assert_eq!(
        asset["recipe"]["rawAdapterSettings"]["droppedAudioLayers"],
        json!([]),
        "{asset}"
    );
    // The whole file decodes to audio of the picture's length.
    let all = decode_window(&render_path, 0.0, total + 1.0);
    assert!(
        (all.len() as f64 / 48_000.0 - total).abs() < 0.15,
        "decoded audio length {}s vs picture {total}s",
        all.len() as f64 / 48_000.0
    );

    // Bed continuity across the cut at 5.1667s: the 100 Hz room tone and the 250 Hz theme are
    // present on both sides at the same level (one continuous bed each, not restarted).
    let before = decode_window(&render_path, 4.6, 0.4);
    let after = decode_window(&render_path, 5.4, 0.4);
    // The fixture beds are quiet by design (triangle waves at amplitude 2600 / 3600 of 32768,
    // mixed at 0.35 / 0.2): a few thousandths of full scale on the probe, and clearly above the
    // silence floor a muted or missing bed reads as (< 0.0005 measured).
    for (label, hz) in [("ambience", 100.0), ("music", 250.0)] {
        let (b, a) = (tone_level(&before, hz), tone_level(&after, hz));
        assert!(
            b > 0.002 && a > 0.002,
            "{label} must span the cut: {b:.5} before, {a:.5} after"
        );
        assert!(
            (b - a).abs() < b * 0.35,
            "{label} must not step at the cut: {b:.5} -> {a:.5}"
        );
    }
    // The line sits 1.2s into SH020 — 6.37s..7.87s in the file, the 1.5s the fake speaks that
    // text for — and nowhere else.
    let line_inside = tone_level(&decode_window(&render_path, 6.7, 0.4), 400.0);
    let line_outside = tone_level(&decode_window(&render_path, 2.0, 0.4), 400.0);
    assert!(
        line_inside > 0.005,
        "the dialogue line must be audible inside its slot: {line_inside:.5}"
    );
    assert!(
        line_outside < line_inside / 10.0,
        "the dialogue line must be absent outside its slot: {line_outside:.5} vs {line_inside:.5}"
    );
    // The takes' own 900 Hz audio is muted by the plan's `generatedAudio: mute`.
    for start in [2.0, 6.7] {
        let generated = tone_level(&decode_window(&render_path, start, 0.4), 900.0);
        assert!(
            generated < 0.0005,
            "generated clip audio must be muted at {start}s: {generated:.5}"
        );
    }

    // A placed clip whose asset is gone is DROPPED from the mix and reported, by name and reason,
    // in the export result — and from there in the run record and the sidecar (E3).
    let timeline_id = record.timeline.as_ref().unwrap().timeline_id.clone();
    let mut timeline = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let tracks = timeline["tracks"].as_array_mut().unwrap();
    let editor_track = tracks
        .iter_mut()
        .find(|track| track["id"] == "track_audio")
        .expect("the editor's default audio track is kept");
    editor_track["items"] = json!([{
        "id": "item_editor_missing",
        "trackId": "track_audio",
        "assetId": "asset_that_does_not_exist",
        "type": "audio",
        "displayName": "editor clip",
        "sourceIn": 0.0,
        "sourceOut": 1.0,
        "timelineStart": 0.5,
        "timelineEnd": 1.5,
        "speed": 1.0,
        "fit": "fit",
        "volume": 1.0,
    }]);
    let (status, _) = request(
        harness.app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": timeline }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let edited = film_harness::edit_timeline(
        &harness.transport,
        &edit_options(&harness, true),
        TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(0.5),
            source_out: None,
        },
    )
    .await
    .expect("trim --export re-exports through the real worker");
    let export = edited.export.as_ref().expect("export");
    assert_eq!(export.status, "completed", "{export:?}");
    assert_eq!(export.dropped_audio_layers.len(), 1, "{export:?}");
    assert_eq!(
        export.dropped_audio_layers[0]["assetId"],
        "asset_that_does_not_exist"
    );
    assert_eq!(export.dropped_audio_layers[0]["reason"], "asset_missing");
    assert_eq!(export.dropped_audio_layers[0]["trackId"], "track_audio");
    assert_eq!(
        harness.run_record()["export"]["droppedAudioLayers"][0]["reason"],
        "asset_missing"
    );

    worker.abort();
    server.abort();
}

// ---------------------------------------------------------------------------------------------
// (B) helper alignment is asserted structurally: both `ffmpeg_reachable` helpers share one rule
// ---------------------------------------------------------------------------------------------

#[test]
fn both_ffmpeg_reachable_helpers_fall_back_to_the_path_probe_when_the_override_is_broken() {
    let api = include_str!("film_harness.rs");
    let api_body = api
        .split("pub(crate) fn ffmpeg_reachable() -> bool {")
        .nth(1)
        .expect("helper exists")
        .split("\n}\n")
        .next()
        .expect("helper ends");
    // A set-but-broken SCENEWORKS_FFMPEG must NOT decide "unreachable" on its own: the override
    // only counts when it names a real file, and otherwise the PATH probe decides.
    assert!(
        api_body.contains(".exists())") && api_body.contains("Command::new(\"ffmpeg\")"),
        "{api_body}"
    );
    assert!(
        !api_body
            .contains("Ok(path) if !path.trim().is_empty() => Path::new(path.trim()).is_file(),"),
        "the old set-but-broken short-circuit is gone"
    );
    assert_eq!(FAKE_REFINE_PEAK_BYTES, 9_000_000_000);
}

// ---------------------------------------------------------------------------------------------
// sc-22715 evaluation finding — a replacement dropped every dialogue line from the sequence
// ---------------------------------------------------------------------------------------------

/// `replace-take` re-assembled the timeline without adopting the run's sound assets, so the merge
/// re-derived an EMPTY dialogue track over the saved one and every line the run had placed was
/// gone from the sequence and the next export (seen on the 2026-09-14 evaluation film: three lines
/// after the run, none after the first `request-repair`). The beds only survived because a bed
/// track with no asset is skipped and then carried over as a track the harness does not own.
///
/// The sound-carrying fixture pack; SH020 places the courier's line at +1.2 s. The editor turns
/// that line down before the replacement, which also exercises the Step-0 merge rule.
#[tokio::test]
async fn a_replacement_keeps_the_dialogue_lines_the_run_placed() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let project_id = record.project_id.clone().expect("project");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let lines = items_of(&saved, "track_dialogue");
    assert_eq!(lines.len(), 1, "SH020's line is placed by the run: {saved}");
    let line_id = lines[0]["id"].as_str().expect("item id").to_owned();

    // The editor turns the line down and gives it fades.
    let mut edited = saved.clone();
    let line = edited["tracks"]
        .as_array_mut()
        .expect("tracks")
        .iter_mut()
        .find(|track| track["id"] == json!("track_dialogue"))
        .expect("dialogue track")["items"]
        .as_array_mut()
        .expect("items")
        .iter_mut()
        .find(|item| item["id"] == json!(line_id))
        .expect("the line");
    line["volume"] = json!(0.4);
    line["fadeInSeconds"] = json!(0.25);
    line["fadeOutSeconds"] = json!(0.5);
    let (status, body) = request(
        harness.app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": edited }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");

    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "the courier is the wrong person",
    )
    .await
    .expect("replacement runs");
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let lines = items_of(&saved, "track_dialogue");
    let kept = lines
        .iter()
        .find(|item| item["id"] == json!(line_id))
        .unwrap_or_else(|| {
            panic!("the replacement dropped SH020's dialogue line from the sequence: {saved}")
        });
    let sh020_start = picture_item(&saved, "SH020")["timelineStart"]
        .as_f64()
        .expect("start");
    assert!(
        close(kept["timelineStart"].as_f64().unwrap(), sh020_start + 1.2),
        "the line still sits at its shot's start plus its offset: {kept}"
    );
    assert_eq!(
        kept["assetId"],
        json!(
            record
                .sound
                .iter()
                .find(|clip| clip.role == "courier_line")
                .expect("the line's clip")
                .asset_id
        ),
        "the clip is the one the run imported, adopted rather than re-uploaded: {kept}"
    );
    assert_eq!(
        kept["volume"],
        json!(0.4),
        "the editor's volume survives: {kept}"
    );
    assert_eq!(kept["fadeInSeconds"], json!(0.25));
    assert_eq!(kept["fadeOutSeconds"], json!(0.5));
    assert_eq!(
        after.sound.len(),
        record.sound.len(),
        "the replacement adopted the clips instead of importing them again: {}",
        summary(&after)
    );
    for track_id in ["track_ambience", "track_music"] {
        assert_eq!(
            items_of(&saved, track_id).len(),
            1,
            "{track_id} still carries its bed: {saved}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// sc-22715 adversarial review — a replacement on one shot changed another shot's line, and the
// whole run's verdict
// ---------------------------------------------------------------------------------------------

/// A run of SH050 + SH060 — the two shots of the shipped plan that both carry a dialogue line —
/// with SH050's take rejected by hand afterwards. The run itself completed; the rejection is a
/// human decision taken over a finished film, which is what the 2026-09-14 evaluation did.
async fn completed_run_with_sh050_rejected() -> (Harness, RunRecord) {
    let harness = Harness::start(true, fast(&["SH050", "SH060"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH050", "SH060"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    review::decide_take(
        &harness.out_dir(),
        "SH050",
        Decision::Reject,
        "a different room from SH040",
    )
    .expect("the rejection records");
    (harness, record)
}

/// A shot whose take was rejected keeps its LINE as long as it keeps its place in the cut
/// (sc-22715 adversarial review).
///
/// The picture track is MERGED onto the saved sequence, so a rejected shot's item stays and the
/// shot is still on screen. The audio tracks were re-derived from `selected_takes()`, which no
/// longer names that shot — so the next re-assembly, triggered by a replacement on a completely
/// different shot, silently dropped its dialogue. The evaluation film lost SH050's line to a
/// `replace-take` on SH040 that way (`run.json.v1` three lines, `v6` two), and the delivered MP4
/// is permanently missing it while SH050 itself plays at 19.67–24.83 s.
#[tokio::test]
async fn a_rejected_shots_line_survives_a_replacement_on_another_shot() {
    let (harness, record) = completed_run_with_sh050_rejected().await;
    let project_id = record.project_id.clone().expect("project");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let before = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let placed: Vec<String> = items_of(&before, "track_dialogue")
        .iter()
        .map(|item| item["filmHarness"]["shotId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        placed,
        vec!["SH050".to_owned(), "SH060".to_owned()],
        "the run places a line for each of the two shots that declare one: {before}"
    );

    // A replacement for the OTHER shot. Nothing about SH050 is named anywhere in this call.
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH060",
        "the recipient is wrong",
    )
    .await
    .expect("the replacement runs");

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let kept: Vec<String> = items_of(&saved, "track_dialogue")
        .iter()
        .map(|item| item["filmHarness"]["shotId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        kept,
        vec!["SH050".to_owned(), "SH060".to_owned()],
        "a replacement on SH060 dropped SH050's line even though SH050 is still in the cut: \
         {saved}"
    );
    // The line is still against its own beat, which is the only thing that makes keeping it worth
    // anything: SH050's clip sits at its shot's start plus the plan's 2.6 s offset.
    let sh050_start = picture_item(&saved, "SH050")["timelineStart"]
        .as_f64()
        .expect("start");
    let items = items_of(&saved, "track_dialogue");
    let line = items
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!("SH050"))
        .expect("SH050's line");
    assert!(
        close(line["timelineStart"].as_f64().unwrap(), sh050_start + 2.6),
        "SH050's line still sits against its beat: {line}"
    );
    assert_eq!(
        after.shot("SH050").unwrap().selected_attempt,
        None,
        "keeping the line must not quietly re-select the take the human rejected: {}",
        summary(&after)
    );
}

/// A replacement decides ONE shot; it does not re-open the verdict the run reached (sc-22715
/// adversarial review).
///
/// `finish_replacement` fell through to `Session::finish`, whose `all_rendered` is computed over
/// EVERY selected shot — and a `reject-take` leaves exactly one shot with no selection. So a
/// successful replacement of an unrelated shot re-derived the run's outcome and turned `completed`
/// into `failed` / `attempts_exhausted` / `resumable: false`. The evaluation's final record reads
/// that way while its own snapshots v1–v5 read `completed`; the replacement that flipped it was on
/// SH040 and never touched SH050's state at all.
#[tokio::test]
async fn a_replacement_does_not_re_judge_the_run_it_was_asked_about() {
    let (harness, record) = completed_run_with_sh050_rejected().await;
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH060",
        "the recipient is wrong",
    )
    .await
    .expect("the replacement runs");

    assert_eq!(
        after.outcome,
        RunOutcome::Completed,
        "a replacement on SH060 re-judged the run over SH050's rejected take: {}",
        summary(&after)
    );
    assert!(
        after.stop.is_none(),
        "a completed run gains no stop from a replacement that succeeded: {}",
        summary(&after)
    );
    assert_eq!(
        after.shot("SH050").unwrap().selected_attempt,
        None,
        "SH050's own state is untouched: {}",
        summary(&after)
    );
    assert_eq!(after.shot("SH050").unwrap().outcome, ShotOutcome::Rendered);
    assert_eq!(
        after.shot("SH060").unwrap().selected_attempt,
        Some(2),
        "SH060 is the shot this invocation decided: {}",
        summary(&after)
    );
    // The verdict standing does not mean the MP4 does. Two separate acts moved the sequence out
    // from under it — the rejection and the replacement — and neither may leave the export
    // claiming to be current; that flag is the ONLY thing that stops `completed` from reading as
    // "there is a finished film matching this record".
    assert!(
        after.export.as_ref().is_some_and(|export| export.stale),
        "the export the run made no longer matches the sequence: {}",
        summary(&after)
    );
    assert_eq!(after.state, RunState::Finished, "{}", summary(&after));
    // The record on disk says the same thing the returned value does.
    let persisted = harness_record(&harness);
    assert_eq!(persisted.outcome, RunOutcome::Completed);
    assert!(persisted.stop.is_none(), "{}", summary(&persisted));
    let _ = record;
}

/// Rejecting a take says, in the record, that the shot stays in the cut carrying it (sc-22715
/// adversarial review).
///
/// The failed-replacement path already writes that note. The reject path left the shot `rendered`
/// with `selectedAttempt: null` and nothing anywhere saying that the timeline and the exported MP4
/// still show the take the human threw away — which is exactly what a reader of the evaluation's
/// record could not tell about SH050.
#[tokio::test]
async fn rejecting_a_take_records_that_the_shot_stays_in_the_cut_carrying_it() {
    let (harness, _) = completed_run_with_sh050_rejected().await;
    let record = harness_record(&harness);
    let note = record
        .decisions
        .iter()
        .filter(|decision| decision.shot_id.as_deref() == Some("SH050"))
        .find(|decision| decision.detail.contains("REJECTED take"))
        .unwrap_or_else(|| {
            panic!(
                "nothing in the record says SH050 stays in the sequence carrying the take that \
                 was rejected: {:#?}",
                record.decisions
            )
        });
    assert!(
        note.detail.contains("stays in the sequence")
            && note.detail.contains("replace-take --shot SH050"),
        "the note has to say what the cut shows AND what changes it: {}",
        note.detail
    );
}

// ---------------------------------------------------------------------------------------------
// sc-23406 (S5) — the turbo courier plan
// ---------------------------------------------------------------------------------------------

/// `plan.v2.turbo.jsonc` validates and compiles against the LIVE catalog, and it is the SAME FILM
/// as `plan.v2.jsonc` in every respect but its LoRA selection — asserted field by field rather
/// than trusted, because the two files are maintained side by side and a drift between them makes
/// the turbo-vs-base comparison they exist for meaningless.
///
/// All six shots resolve to `minimax_h3_ref`, so all six carry the ref2v turbo and NONE carries the
/// base-partition one — even though the plan declares both. That is the per-partition resolution:
/// the base entry is declared for the family and simply never dispatched by this plan.
#[tokio::test]
async fn the_turbo_plan_puts_the_ref2v_recipe_on_every_shot_and_the_base_recipe_on_none() {
    let harness = Harness::start(true, Vec::new()).await;
    let turbo_path = Path::new(FIXTURE_DIR).join("plan.v2.turbo.jsonc");
    let v2_path = Path::new(FIXTURE_DIR).join("plan.v2.jsonc");

    let (plan, _) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(turbo_path.clone(), harness.fixture_pack(), None),
    )
    .await
    .expect("plan.v2.turbo.jsonc validates against the live catalog");
    assert_eq!(
        plan.model.loras,
        vec!["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"],
        "the regime is declared once on the family"
    );

    // The same film as its sibling apart from `model.loras`. Compared on the parsed documents, so
    // a comment-only difference is not a false failure and a substantive one cannot hide.
    let (base, _) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(v2_path, harness.fixture_pack(), None),
    )
    .await
    .expect("plan.v2.jsonc validates");
    assert!(base.model.loras.is_empty(), "the sibling declares none");
    let mut stripped = plan.clone();
    stripped.model.loras.clear();
    assert_eq!(
        stripped, base,
        "plan.v2.turbo.jsonc differs from plan.v2.jsonc in model.loras and nothing else"
    );

    // Compile: every one of the six requests is on the reference partition and carries the ref2v
    // recipe at the schedule the catalog declares for it — and the base-partition adapter reaches
    // nothing, because no shot of this plan resolves to the base checkpoint.
    let options = planner_options(&harness, "plan-v2-turbo");
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &turbo_path,
    )
    .await
    .expect("plan.v2.turbo.jsonc compiles");
    assert_eq!(artifacts.compiled.requests.len(), 6);
    for request in &artifacts.compiled.requests {
        assert_eq!(request.model, "minimax_h3_ref", "{}", request.shot_id);
        assert_eq!(
            request.loras,
            vec!["minimax_h3_ref2v_turbo_4step"],
            "{}: the ref2v recipe, and only it",
            request.shot_id
        );
        assert!(
            !request
                .loras
                .contains(&"minimax_h3_turbo_4step_v01".to_owned()),
            "{}: the base-partition adapter must reach no reference shot",
            request.shot_id
        );
        assert_eq!(request.effective_steps, Some(4), "{}", request.shot_id);
        assert_eq!(
            request.turbo_scheduler_shift,
            Some(12.0),
            "{}",
            request.shot_id
        );
        assert_eq!(
            request.steps, None,
            "{}: the plan sets no override, so the recipe governs",
            request.shot_id
        );
    }
}

/// The MIXED case the shipped plans do not cover: one plan, one LoRA list, shots on BOTH
/// partitions — each getting the adapter its own checkpoint was distilled for.
///
/// In code rather than as a fourth checked-in plan, because `plan.ref.jsonc` already IS the mixed
/// fixture and a second copy of it that differed only in `model.loras` would be one more document
/// to keep in step with the other three.
#[tokio::test]
async fn a_mixed_plan_gets_the_right_recipe_on_each_partition() {
    let harness = Harness::start(true, Vec::new()).await;
    let plan_path = harness.mixed_partition_plan(|plan| {
        plan["model"]["loras"] =
            serde_json::json!(["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"]);
    });
    let options = planner_options(&harness, "plan-mixed-turbo");
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &plan_path,
    )
    .await
    .expect("the mixed plan compiles");
    let request = |shot_id: &str| {
        artifacts
            .compiled
            .request(shot_id)
            .unwrap_or_else(|| panic!("no compiled request for {shot_id}"))
    };
    let referenced = request("SH010");
    let plain = request("SH020");
    assert_eq!(referenced.model, "minimax_h3_ref");
    assert_eq!(referenced.loras, vec!["minimax_h3_ref2v_turbo_4step"]);
    assert_eq!(plain.model, "minimax_h3");
    assert_eq!(plain.loras, vec!["minimax_h3_turbo_4step_v01"]);
    // Both recipes declare the same (4, 12.0) schedule — the reason `plan.v2.turbo.jsonc` pairs
    // the v0.1 files rather than the 8-step one — so the two halves of a mixed film are comparable.
    assert_eq!(referenced.effective_steps, plain.effective_steps);
    assert_eq!(referenced.effective_steps, Some(4));
    assert_eq!(
        referenced.turbo_scheduler_shift,
        plain.turbo_scheduler_shift
    );
}

// ---------------------------------------------------------------------------------------------
// sc-23405 (S4) — the reference-conditioned courier plan
// ---------------------------------------------------------------------------------------------

/// AC1. `plan.v2.jsonc` validates and compiles against the LIVE catalog with every one of its six
/// shots resolved to `minimax_h3_ref`, every beat of the shipped brief covered and every role that
/// brief requires bound — and `plan.jsonc` still validates unchanged beside it as the phase-1
/// no-reference baseline, every request on `minimax_h3`.
///
/// The pack here is the checked-in stand-in (deterministic placeholder plates). It declares exactly
/// the roles a GENERATED pack declares, so what this proves is the plumbing — the right partition,
/// the right payload, the right coverage — which is the part a test can prove. Likeness needs real
/// plates and a GPU.
#[tokio::test]
async fn the_reference_plan_resolves_every_shot_to_the_reference_partition_and_covers_the_brief() {
    let harness = Harness::start(true, Vec::new()).await;
    let v2_path = Path::new(FIXTURE_DIR).join("plan.v2.jsonc");
    let baseline_path = Path::new(FIXTURE_DIR).join("plan.jsonc");

    // `validate` first — the same command a human runs, against the live catalog entries.
    let (plan, pack) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(v2_path.clone(), harness.fixture_pack(), None),
    )
    .await
    .expect("plan.v2.jsonc validates against the live catalog");
    assert_eq!(plan.id, "courier-workshop-v2");
    assert_eq!(plan.shots.len(), 6);
    assert_eq!(pack.references.len(), 7);
    for shot in &plan.shots {
        assert_eq!(shot.conditioning.mode, "reference_to_video", "{}", shot.id);
        assert!(
            shot.conditioning
                .reference_roles
                .contains(&"workshop_location".to_owned())
                && shot
                    .conditioning
                    .reference_roles
                    .contains(&"red_parcel".to_owned()),
            "every shot is conditioned on the one approved room and the one approved parcel: {} \
             binds {:?}",
            shot.id,
            shot.conditioning.reference_roles
        );
    }

    // Beat coverage and the brief's REQUIRED ROLES, by identity, through the `beatId` each shot
    // carries. `compile_existing` refuses on these findings too (it picks the sibling brief up);
    // asserting them here says which rule holds rather than only that compiling succeeded.
    let brief = sceneworks_core::film_planner::read_brief_file(Path::new(BRIEF_FIXTURE))
        .expect("the shipped brief reads");
    let coverage = sceneworks_core::film_planner::plan_coverage_findings(&brief, &plan, &pack);
    assert!(
        coverage.is_empty(),
        "{:?}",
        coverage
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
    );
    let claimed: Vec<&str> = plan
        .shots
        .iter()
        .filter_map(|shot| shot.beat_id.as_deref())
        .collect();
    assert_eq!(
        claimed,
        vec![
            "arrival",
            "approach",
            "handover",
            "departure",
            "discovery",
            "opening"
        ]
    );

    // Compile: every request resolves to the reference partition, says why, and keeps the plan's
    // role order.
    let mut options = planner_options(&harness, "plan-v2");
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &v2_path,
    )
    .await
    .expect("plan.v2.jsonc compiles");
    assert_eq!(artifacts.compiled.requests.len(), 6);
    // The plan still declares the FAMILY once; only the requests resolve.
    assert_eq!(artifacts.compiled.model.id, "minimax_h3");
    for request in &artifacts.compiled.requests {
        assert_eq!(request.model, "minimax_h3_ref", "{}", request.shot_id);
        assert_eq!(request.mode, "reference_to_video", "{}", request.shot_id);
        assert!(
            request.partition_reason.contains("minimax_h3_ref"),
            "{}: {}",
            request.shot_id,
            request.partition_reason
        );
        let shot = plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("every request is a shot of the plan");
        assert_eq!(
            request.reference_roles, shot.conditioning.reference_roles,
            "{}",
            request.shot_id
        );
        assert!(
            request.reference_roles.len() <= 9,
            "{}: minimax_h3_ref declares maxReferenceAssets 9",
            request.shot_id
        );
    }

    // The phase-1 baseline is UNCHANGED: still validates, still entirely on the base checkpoint,
    // and still the same film shot for shot — which is what makes the two comparable.
    let (baseline, _) = film_harness::validate(
        Some(&harness.transport),
        &harness.options(baseline_path.clone(), harness.fixture_pack(), None),
    )
    .await
    .expect("plan.jsonc still validates");
    assert_eq!(baseline.id, "courier-workshop");
    assert!(
        baseline
            .shots
            .iter()
            .all(|shot| shot.conditioning.reference_roles.is_empty()),
        "the baseline binds no reference roles at all"
    );
    let ids = |plan: &sceneworks_core::film_plan::ProductionPlan| {
        plan.shots
            .iter()
            .map(|shot| shot.id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&plan), ids(&baseline), "the same six shots, same ids");
    for (v2, phase1) in plan.shots.iter().zip(&baseline.shots) {
        assert!(close(
            v2.target_duration_seconds,
            phase1.target_duration_seconds
        ));
        assert_eq!(v2.start_state, phase1.start_state, "{}", v2.id);
        assert_eq!(v2.end_state, phase1.end_state, "{}", v2.id);
        assert_eq!(v2.dialogue, phase1.dialogue, "{}", v2.id);
        assert_eq!(
            v2.dialogue_clip.as_ref().map(|clip| clip.role.as_str()),
            phase1.dialogue_clip.as_ref().map(|clip| clip.role.as_str()),
            "{}",
            v2.id
        );
    }

    options.out_dir = harness.temp_dir.path().join("plan-baseline");
    let baseline_artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &baseline_path,
    )
    .await
    .expect("plan.jsonc compiles");
    assert!(
        baseline_artifacts
            .compiled
            .requests
            .iter()
            .all(|request| request.model == "minimax_h3"),
        "{:?}",
        baseline_artifacts
            .compiled
            .requests
            .iter()
            .map(|request| (request.shot_id.clone(), request.model.clone()))
            .collect::<Vec<_>>()
    );
}

/// Controlled completion through real routes: an operator edits/deletes A while B still renders.
/// Delivery B must preserve that cut, and replaying A must respect the stored deletion.
#[tokio::test]
async fn incremental_film_delivery_preserves_concurrent_cut_and_tombstones() {
    let harness = Harness::start(
        true,
        vec![
            (
                "SH010",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 20.0,
                },
            ),
            (
                "SH020",
                VideoBehavior::Complete {
                    delay_secs: 2,
                    peak_pct: 20.0,
                },
            ),
        ],
    )
    .await;
    let mut options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    options.export = false;
    let second_started = Arc::new(tokio::sync::Notify::new());
    harness.script.lock().running_hook = Some((
        "SH020".to_owned(),
        RunningHook::Notify(second_started.clone()),
    ));
    // Register the waiter before the run can reach the hook, so the lifecycle edge cannot be lost.
    // Keep polling the controller until that edge; if it exits first, report its real result rather
    // than timing out while polling shared fixture state. Once the worker reports SH020 running,
    // leave the controller future unpolled while the editor's concurrent save lands.
    let second_started = second_started.notified();
    let run = film_harness::run(&harness.transport, &options);
    tokio::pin!(second_started);
    tokio::pin!(run);
    tokio::select! {
        () = &mut second_started => {},
        result = &mut run => panic!("incremental run ended before SH020 started: {result:?}"),
    }

    let record = harness.run_record();
    let project_id = record["projectId"].as_str().unwrap();
    let timeline_id = record["timeline"]["timelineId"]
        .as_str()
        .expect("first shot delivered before second starts");
    let original = saved_timeline(&harness.app, project_id, timeline_id).await;
    assert_eq!(picture_order(&original), vec!["SH010"]);
    let mut deleted = original.clone();
    deleted["tracks"][0]["items"] = json!([]);
    deleted["tracks"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|t| t["kind"] == "audio")
        .unwrap()["gain"] = json!(0.35);
    let (status, saved) = request(
        harness.app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({"timeline":deleted,"expectedRevision":original["revision"]}),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{saved}");
    assert!(
        !saved["filmAssembly"]["runs"][record["runId"].as_str().unwrap()]["deletedShots"]["SH010"]
            .is_null()
    );

    let record = run.await.expect("incremental run completes");
    let project_id = record.project_id.as_deref().unwrap();
    let timeline_id = &record.timeline.as_ref().unwrap().timeline_id;
    let saved = saved_timeline(&harness.app, project_id, timeline_id).await;
    assert_eq!(picture_order(&saved), vec!["SH020"]);
    assert_eq!(
        saved["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["kind"] == "audio")
            .unwrap()["gain"],
        0.35
    );
    assert!(record.export.is_none());
    assert!(!harness
        .script
        .lock()
        .claimed
        .iter()
        .any(|(kind, _, _)| kind == "timeline_export"));
    assert_eq!(record.timeline.as_ref().unwrap().items.len(), 1);
}
