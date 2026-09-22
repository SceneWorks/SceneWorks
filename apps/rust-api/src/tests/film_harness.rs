//! End-to-end tests for the local filmmaking harness (sc-22710): the REAL API routes in-process
//! (`create_app_with_state`), a scripted fake worker that claims the jobs through the worker API
//! exactly as the GPU worker would, and the harness driving both through `ApiTransport`.
//!
//! What the fake worker replaces is only the render: it claims `video_generate` / `timeline_export`
//! jobs, writes a placeholder file where the real worker would write the MP4, and reports the same
//! `assetWrites` / `assetIds` result shapes the real worker reports. Asset persistence, timeline
//! validation, export dispatch and every enqueue gate are the production code paths.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use parking_lot::Mutex;
use sceneworks_core::film_compile::InsertedTextPlacement;
use sceneworks_core::film_plan::{RunOutcome, RunRecord, RunState, ShotOutcome};
use sceneworks_core::film_workspace::FilmDraft;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::film_harness::{
    self, ApiRequest, ApiResponse, ApiTransport, BytesResponse, BytesTransportFuture, HarnessError,
    RequestBody, ResumeOptions, RunControl, RunOptions, TransportFuture, FIXTURE_REFERENCES,
};
use crate::film_planner;
use crate::tests::support::{create_app_with_state, request, test_settings};

pub(crate) const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop"
);

/// [`ApiTransport`] over the in-process router: the same `oneshot` driver every route test uses.
pub(crate) struct RouterTransport {
    pub(crate) app: axum::Router,
}

#[test]
fn a_second_new_run_controller_is_refused_while_the_first_holds_the_directory() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let directory = temporary.path().join("run");
    let first = film_harness::ControllerLease::acquire(&directory, "api:first")
        .expect("first controller acquires");
    let refused = film_harness::ControllerLease::acquire(&directory, "cli:second")
        .expect_err("a competing controller must be refused");
    assert!(refused.to_string().contains("another controller"));
    drop(first);
    film_harness::ControllerLease::acquire(&directory, "cli:after")
        .expect("the lease releases when its owner finishes");
}

#[test]
fn a_new_action_cannot_consume_the_cancel_request_for_a_competing_controller() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let directory = temporary.path().join("run");
    std::fs::create_dir_all(&directory).expect("run dir");
    std::fs::write(directory.join("run.json"), "{}").expect("run record");
    let active = film_harness::ControllerLease::acquire(&directory, "api:active")
        .expect("active controller acquires");
    let sentinel = film_harness::request_cancel(&directory).expect("active controller is canceled");

    let refused = film_harness::ControllerLease::acquire_new_action(&directory, "api:new-action")
        .expect_err("a competing action cannot acquire the run");
    assert!(refused.to_string().contains("another controller"));
    assert!(
        sentinel.exists(),
        "the refused action must not consume the active controller's cancel"
    );
    film_harness::ControllerLease::acquire_new_action_for_api(
        &directory,
        "api:competing-resume",
        film_harness::FilmControllerShutdown::default(),
    )
    .expect_err("API action also loses the lease before consuming cancellation");
    assert!(sentinel.exists());
    drop(active);
}

#[test]
fn an_unlocked_crash_metadata_file_is_recovered_without_an_age_guess() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let directory = temporary.path().join("run");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(film_harness::CONTROLLER_LOCK_FILE),
        "owner=dead-process\npid=999999\nacquiredAt=2000-01-01T00:00:00Z\n",
    )
    .unwrap();
    assert!(!film_harness::ControllerLease::is_active(&directory).unwrap());
    let lease = film_harness::ControllerLease::acquire_interrupted(&directory, "startup-adopt")
        .expect("the interrupted lease can be inspected")
        .expect("the OS-released lock is recoverable regardless of metadata age");
    assert!(film_harness::ControllerLease::is_active(&directory).unwrap());
    drop(lease);
    assert!(!film_harness::ControllerLease::is_active(&directory).unwrap());
    assert!(
        film_harness::ControllerLease::acquire_interrupted(&directory, "startup-again")
            .unwrap()
            .is_none(),
        "a clean release erases the crash marker"
    );
}

#[tokio::test]
async fn every_run_mutation_refuses_a_competing_api_or_cli_controller() {
    use crate::film_harness::review::{self, Decision, ReviewOptions, ScriptedVision};
    use crate::film_harness::{EditOptions, TimelineEdit};

    let temporary = tempfile::tempdir().expect("temp dir");
    let out_dir = temporary.path().join("run");
    let held = film_harness::ControllerLease::acquire(&out_dir, "api:active").unwrap();
    let (app, _) = create_app_with_state(test_settings(&temporary)).expect("app creates");
    let transport = RouterTransport { app };
    let options = ResumeOptions::new(out_dir.clone());
    let edit = EditOptions {
        run_record_path: out_dir.join(film_harness::RUN_RECORD_FILE),
        export: false,
        poll_interval: Duration::from_millis(1),
    };

    let errors = [
        film_harness::resume(&transport, &options)
            .await
            .unwrap_err(),
        film_harness::replace_take(&transport, &options, "SH010", "replace")
            .await
            .unwrap_err(),
        film_harness::edit_timeline(
            &transport,
            &edit,
            TimelineEdit::Reorder { shot_ids: vec![] },
        )
        .await
        .unwrap_err(),
        review::review(
            &transport,
            &ReviewOptions::new(out_dir.clone()),
            &ScriptedVision::new(),
        )
        .await
        .unwrap_err(),
        review::request_repair(&transport, &options, "SH010", "repair")
            .await
            .unwrap_err(),
        review::decide_take(&out_dir, "SH010", Decision::Accept, "accept").unwrap_err(),
    ];
    for error in errors {
        assert!(error.to_string().contains("another controller"), "{error}");
    }
    drop(held);
}

#[tokio::test]
async fn api_startup_adopts_a_surviving_workers_exact_film_job_without_redispatch() {
    let harness = Harness::start_http(
        true,
        vec![(
            "SH010",
            VideoBehavior::Complete {
                delay_secs: 3,
                peak_pct: 40.0,
            },
        )],
    )
    .await;
    let project = harness
        .state
        .project_store
        .create_project("Startup adoption")
        .expect("project creates");
    let draft_id = "film_draft_restart";
    let mut draft = FilmDraft::manual_one_shot(&project.id, draft_id, "Startup adoption");
    draft.production_plan.shots[0].beat = "A courier crosses the workshop.".to_owned();
    draft.production_plan.shots[0].prompt =
        "A courier crosses a quiet workshop carrying a red parcel.".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    harness
        .state
        .project_store
        .create_film_draft_document(&project.id, draft)
        .expect("draft creates");
    let locator_id = "filmrun_restart";
    harness
        .state
        .project_store
        .create_film_run(
            &project.id,
            locator_id,
            draft_id,
            vec!["SH010".to_owned()],
            None,
        )
        .expect("run locator creates");
    let files = harness
        .state
        .project_store
        .film_run_files(&project.id, locator_id)
        .expect("run files resolve");
    let options = RunOptions {
        plan_path: files.plan,
        reference_pack_path: files.reference_pack,
        compiled_path: None,
        project_id: Some(project.id.clone()),
        shot_ids: Some(vec!["SH010".to_owned()]),
        out_dir: files.directory.clone(),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let running = Arc::new(tokio::sync::Notify::new());
    harness.script.lock().running_hook =
        Some(("SH010".to_owned(), RunningHook::Notify(running.clone())));
    let reached_running = running.notified();
    tokio::pin!(reached_running);
    let controller_app = harness.app.clone();
    let lease = film_harness::ControllerLease::acquire_for_api(
        &files.directory,
        "api:filmrun_restart",
        harness.state.film_controller_shutdown.clone(),
    )
    .expect("API controller lease acquires");
    let mut controller = tokio::spawn(async move {
        film_harness::run_with_control_and_lease(
            &RouterTransport {
                app: controller_app,
            },
            &options,
            &film_harness::RunControl::default(),
            lease,
        )
        .await
    });
    tokio::select! {
        () = &mut reached_running => {}
        result = &mut controller => {
            panic!("controller ended before the fake worker reached running: {result:?}");
        }
    }
    let original_job_id = harness
        .script
        .lock()
        .claimed
        .iter()
        .find(|(kind, _, _)| kind == "video_generate")
        .map(|(_, job_id, _)| job_id.clone())
        .expect("video job was claimed");

    // The production SIGTERM path sets shutdown intent before Axum drains. Simulate the runtime
    // then dropping this detached controller while the separately hosted worker continues the
    // exact job. Unlike an ordinary handled controller exit, this Drop must retain its marker.
    harness.state.film_controller_shutdown.request();
    controller.abort();
    assert!(
        controller
            .await
            .expect_err("controller aborts")
            .is_cancelled(),
        "the original controller must be gone before startup adopts it"
    );
    let marker = std::fs::read_to_string(files.directory.join(film_harness::CONTROLLER_LOCK_FILE))
        .expect("controller marker remains readable");
    assert!(
        marker
            .lines()
            .any(|line| line == "owner=api:filmrun_restart"),
        "graceful API shutdown must retain active controller ownership: {marker:?}"
    );
    let interrupted = harness
        .state
        .jobs_store
        .mark_interrupted_on_startup()
        .expect("API startup recovery succeeds");
    assert!(
        interrupted.is_empty(),
        "the worker-owned render survives API startup: {interrupted:?}"
    );
    let active = harness
        .state
        .jobs_store
        .get_job(&original_job_id)
        .expect("active job loads");
    assert_eq!(
        active.status,
        sceneworks_core::contracts::JobStatus::Running
    );
    assert_eq!(active.worker_id.as_deref(), Some(WORKER_ID));

    let mut restarted_state = harness.state.clone();
    restarted_state.film_controller_shutdown = film_harness::FilmControllerShutdown::default();
    let mut startup =
        crate::film_lifecycle::spawn_film_startup_reconciliation_for_fake_worker(restarted_state);
    startup.scan.await.expect("startup scan joins");
    startup
        .controller_results
        .recv()
        .await
        .expect("startup scan launches the film controller")
        .unwrap_or_else(|error| panic!("startup film controller failed: {error}"));
    let record = film_harness::read_run_record(&files.directory)
        .expect("startup run record remains readable");
    assert_eq!(record.state, RunState::Finished, "{}", summary(&record));

    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(harness.video_job_count(), 1, "startup must not redispatch");
    assert_eq!(harness.api_video_job_count().await, 1);
    let selected = record
        .shot("SH010")
        .and_then(|shot| shot.selected())
        .expect("startup adopts the completed take");
    assert_eq!(selected.job_id.as_deref(), Some(original_job_id.as_str()));
}

#[tokio::test]
async fn api_startup_leaves_a_cleanly_released_failed_run_for_explicit_resume() {
    let harness = Harness::start_http(true, vec![]).await;
    let project = harness
        .state
        .project_store
        .create_project("Explicit resume")
        .expect("project creates");
    let draft_id = "film_draft_explicit_resume";
    let mut draft = FilmDraft::manual_one_shot(&project.id, draft_id, "Explicit resume");
    draft.production_plan.shots[0].beat = "A courier crosses the workshop.".to_owned();
    draft.production_plan.shots[0].prompt =
        "A courier crosses a quiet workshop carrying a red parcel.".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    harness
        .state
        .project_store
        .create_film_draft_document(&project.id, draft)
        .expect("draft creates");
    let locator_id = "filmrun_explicit_resume";
    harness
        .state
        .project_store
        .create_film_run(
            &project.id,
            locator_id,
            draft_id,
            vec!["SH010".to_owned()],
            None,
        )
        .expect("run locator creates");
    let files = harness
        .state
        .project_store
        .film_run_files(&project.id, locator_id)
        .expect("run files resolve");
    let options = RunOptions {
        plan_path: files.plan.clone(),
        reference_pack_path: files.reference_pack.clone(),
        compiled_path: None,
        project_id: Some(project.id.clone()),
        shot_ids: Some(vec!["SH010".to_owned()]),
        out_dir: files.directory.clone(),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::Before)
        .on_post_route("/api/v1/video/jobs");
    let lease = film_harness::ControllerLease::acquire_for_api(
        &files.directory,
        "api:filmrun_explicit_resume",
        harness.state.film_controller_shutdown.clone(),
    )
    .expect("API controller lease acquires");
    film_harness::run_with_control_and_lease(
        &transport,
        &options,
        &film_harness::RunControl::default(),
        lease,
    )
    .await
    .expect_err("the injected transport loss stops the controller");
    let failed = film_harness::read_run_record(&files.directory).expect("failed record persists");
    assert_eq!(failed.state, RunState::Running, "{}", summary(&failed));
    assert_eq!(failed.outcome, RunOutcome::Failed, "{}", summary(&failed));
    assert!(failed.is_resumable(), "{}", summary(&failed));
    assert_eq!(harness.api_video_job_count().await, 0);
    assert_eq!(
        std::fs::read_to_string(files.directory.join(film_harness::CONTROLLER_LOCK_FILE)).unwrap(),
        "",
        "an ordinary handled API controller failure clears ownership"
    );

    let mut startup = crate::film_lifecycle::spawn_film_startup_reconciliation_for_fake_worker(
        harness.state.clone(),
    );
    startup.scan.await.expect("startup scan joins");
    assert!(
        startup.controller_results.recv().await.is_none(),
        "a cleanly released failed controller must wait for explicit resume"
    );
    assert_eq!(harness.api_video_job_count().await, 0);

    let mut resume = ResumeOptions::new(files.directory.clone());
    resume.poll_interval = Duration::from_millis(100);
    resume.export = false;
    resume.require_installed = false;
    let completed = film_harness::resume(&harness.transport, &resume)
        .await
        .expect("the operator can explicitly resume the failed run");
    assert_eq!(
        completed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&completed)
    );
    assert_eq!(harness.api_video_job_count().await, 1);
}

#[tokio::test]
async fn completion_assembles_one_stable_clip_without_dispatching_an_export() {
    let harness = Harness::start(true, vec![]).await;
    let mut draft = FilmDraft::manual_one_shot("project-film", "film-draft", "Manual film");
    draft.production_plan.shots[0].prompt = "a courier crosses a quiet workshop".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    draft.production_plan.shots[0].beat = "The courier crosses the workshop".to_owned();
    let document_dir = harness.temp_dir.path().join("reference-free-film");
    std::fs::create_dir_all(&document_dir).expect("document directory");
    let plan_path = document_dir.join("plan.json");
    let pack_path = document_dir.join("references.json");
    std::fs::write(
        &plan_path,
        serde_json::to_vec_pretty(&draft.production_plan).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &pack_path,
        serde_json::to_vec_pretty(&draft.reference_pack).unwrap(),
    )
    .unwrap();
    let mut options = harness.options(plan_path, pack_path, Some(&["SH010"]));
    options.export = false;
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("controlled worker completion succeeds");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let compiled = record.compiled.as_ref().expect("compiled request document");
    assert!(Path::new(&compiled.path).is_file());
    assert!(record.export.is_none(), "assembly must not start export");
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    assert_eq!(timeline.items.len(), 1);
    let project_id = record.project_id.as_deref().expect("project");
    let saved = saved_timeline(&harness.app, project_id, &timeline.timeline_id).await;
    assert_eq!(saved["revision"], 2);
    assert_eq!(
        saved["filmAssembly"]["runs"][&record.run_id]["shotOrder"],
        json!(["SH010"])
    );
    let item = &saved["tracks"][0]["items"][0];
    assert_eq!(item["filmHarness"]["runId"], record.run_id);
    assert_eq!(item["filmHarness"]["shotId"], "SH010");
    assert_eq!(item["filmHarness"]["plannedIndex"], 0);
    assert_eq!(item["filmHarness"]["selectedTakeAssetId"], item["assetId"]);
    assert!(item["filmHarness"]["jobId"].as_str().is_some());

    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("repeating the same run directory is refused");
    assert!(error.to_string().contains("already holds run"));
    let (_, timelines) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/timelines"),
        Value::Null,
    )
    .await;
    assert_eq!(
        timelines.as_array().unwrap().len(),
        1,
        "replay made no duplicate timeline"
    );
}

#[tokio::test]
async fn long_legal_asset_identity_is_tagged_through_real_routes_without_losing_provenance() {
    let harness = Harness::start(true, vec![]).await;
    let (plan_path, pack_path) = harness.minimal_documents(json!({
        "maxRunSeconds": 120,
        "maxShotSeconds": 30,
        "maxAttemptsPerShot": 1,
        "maxMemoryGb": 96
    }));
    let shared_role_prefix = "r".repeat(63);
    let first_role = format!("{shared_role_prefix}a");
    let second_role = format!("{shared_role_prefix}b");
    let sound_role = format!("s{}", "s".repeat(63));
    let pack_id = format!("film_{}", "p".repeat(59));
    assert_eq!(first_role.len(), 64);
    assert_eq!(second_role.len(), 64);
    assert_eq!(sound_role.len(), 64);
    assert_eq!(pack_id.len(), 64);

    let mut pack: Value = serde_json::from_slice(&std::fs::read(&pack_path).unwrap()).unwrap();
    pack["id"] = json!(pack_id);
    pack["references"][0]["role"] = json!(first_role);
    let mut second = pack["references"][0].clone();
    second["role"] = json!(second_role);
    // Its OWN plate, copied beside the first. This test is about the LENGTH of the ids the harness
    // stamps onto real assets, so it needs two assets; two roles on one file would be one asset
    // under one `<Picture N>` (sc-24024), which `film_harness_anchoring.rs` covers on purpose.
    let plate_dir = pack_path.parent().unwrap().join("references");
    std::fs::copy(
        plate_dir.join("workshop_plate.png"),
        plate_dir.join("workshop_plate_b.png"),
    )
    .expect("the second plate copies");
    second["file"] = json!("references/workshop_plate_b.png");
    pack["references"].as_array_mut().unwrap().push(second);
    let sound = ffmpeg_reachable();
    if sound {
        let sound_dir = pack_path.parent().unwrap().join("sound");
        std::fs::create_dir_all(&sound_dir).unwrap();
        std::fs::copy(
            Path::new(FIXTURE_DIR).join("sound/workshop_room_tone.wav"),
            sound_dir.join("room-tone.wav"),
        )
        .unwrap();
        pack["sound"] = json!([{
            "role": sound_role,
            "kind": "ambience",
            "file": "sound/room-tone.wav"
        }]);
    }
    std::fs::write(&pack_path, serde_json::to_vec_pretty(&pack).unwrap()).unwrap();

    let mut plan: Value = serde_json::from_slice(&std::fs::read(&plan_path).unwrap()).unwrap();
    for shot in plan["shots"].as_array_mut().unwrap() {
        shot["continuityRoles"] = json!([first_role, second_role]);
    }
    if sound {
        plan["sound"] = json!({
            "generatedAudio": "mute",
            "ambience": { "role": sound_role }
        });
    }
    std::fs::write(&plan_path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();

    let mut options = harness.options(plan_path, pack_path, Some(&["SH010"]));
    options.export = false;
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("the project store accepts every harness-generated tag");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert!(record.run_id.starts_with("run_"), "{}", record.run_id);
    assert_eq!(record.run_id.len(), 36, "a UUID-backed production run id");

    let project_id = record.project_id.as_deref().expect("project");
    let (_, assets) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    let references: Vec<&Value> = assets
        .as_array()
        .unwrap()
        .iter()
        .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
        .collect();
    assert_eq!(references.len(), 2, "{assets:#}");
    let mut role_tags = Vec::new();
    let mut pack_tags = Vec::new();
    for asset in references {
        let provenance = &asset["extra"]["filmHarness"];
        assert_eq!(provenance["runId"], record.run_id);
        assert_eq!(provenance["referencePackId"], pack_id);
        assert!(
            provenance["role"] == first_role || provenance["role"] == second_role,
            "{provenance}"
        );
        let tags: Vec<&str> = asset["tags"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(tags.iter().all(|tag| tag.len() <= 40), "{tags:?}");
        assert!(tags.contains(&"film-harness-reference"), "{tags:?}");
        role_tags.push(
            tags.iter()
                .find(|tag| tag.starts_with("role:h:"))
                .expect("long role has a bounded facet")
                .to_string(),
        );
        pack_tags.push(
            tags.iter()
                .find(|tag| tag.starts_with("pack:h:"))
                .expect("long pack id has a bounded facet")
                .to_string(),
        );
    }
    role_tags.sort();
    role_tags.dedup();
    pack_tags.sort();
    pack_tags.dedup();
    assert_eq!(role_tags.len(), 2, "role collision: {role_tags:?}");
    assert_eq!(pack_tags.len(), 1, "one pack identity: {pack_tags:?}");

    if sound {
        let sounds: Vec<&Value> = assets
            .as_array()
            .unwrap()
            .iter()
            .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "sound")
            .collect();
        assert_eq!(sounds.len(), 1, "{assets:#}");
        let provenance = &sounds[0]["extra"]["filmHarness"];
        assert_eq!(provenance["runId"], record.run_id);
        assert_eq!(provenance["referencePackId"], pack_id);
        assert_eq!(provenance["role"], sound_role);
        let tags: Vec<&str> = sounds[0]["tags"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert!(tags.iter().all(|tag| tag.len() <= 40), "{tags:?}");
        assert!(tags.contains(&"film-harness-sound"), "{tags:?}");
        assert!(
            tags.iter().any(|tag| tag.starts_with("role:h:")),
            "{tags:?}"
        );
        assert!(
            tags.iter().any(|tag| tag.starts_with("pack:h:")),
            "{tags:?}"
        );
    }

    let take_payload = harness
        .script
        .lock()
        .claimed
        .iter()
        .find(|(kind, _, _)| kind == "video_generate")
        .map(|(_, _, payload)| payload.clone())
        .expect("one take was dispatched");
    assert_eq!(
        take_payload["advanced"]["filmHarness"]["runId"],
        record.run_id
    );
    assert_eq!(
        take_payload["advanced"]["filmHarness"]["idempotencyKey"],
        format!("{}:SH010:a1", record.run_id)
    );
}

impl ApiTransport for RouterTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        let app = self.app.clone();
        Box::pin(async move {
            let mut builder = Request::builder()
                .method(request.method)
                .uri(request.path.clone());
            let body = match request.body {
                RequestBody::None => Body::empty(),
                RequestBody::Json(value) => {
                    builder = builder.header("content-type", "application/json");
                    Body::from(value.to_string())
                }
                RequestBody::Multipart { boundary, bytes } => {
                    builder = builder.header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    );
                    Body::from(bytes)
                }
            };
            let response = app
                .oneshot(builder.body(body).expect("request builds"))
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let body = if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap_or(Value::Null)
            };
            Ok(ApiResponse { status, body })
        })
    }

    /// The bytes half: the same `oneshot` driver, without the JSON parse — what `make-references`
    /// downloads a rendered plate through (sc-23403), and what a synthesized dialogue clip is
    /// fetched through (sc-23404).
    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        let app = self.app.clone();
        Box::pin(async move {
            let request = Request::builder()
                .method("GET")
                .uri(path)
                .body(Body::empty())
                .expect("request builds");
            let response = app
                .oneshot(request)
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?
                .to_vec();
            Ok(BytesResponse { status, bytes })
        })
    }
}

#[tokio::test]
async fn film_document_preflight_compiles_selected_shots_and_rejects_a_stale_compile() {
    let _env = crate::tests::support::isolate_hf_cache();
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(
        harness.app.clone(),
        "/api/v1/models",
        only_the_base_partition_is_installed,
    );
    let mut draft = FilmDraft::manual_one_shot("project-film", "film-preflight", "Preflight");
    draft.production_plan.shots[0].prompt = "A courier crosses a quiet workshop.".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    let selected = vec!["SH010".to_owned()];

    let ready = film_harness::preflight_documents(
        &transport,
        &draft.production_plan,
        &draft.reference_pack,
        None,
        Some(&selected),
        true,
    )
    .await
    .expect("preflight resolves through the live route contract");
    assert!(ready.valid, "{:?}", ready.findings);
    assert_eq!(ready.compiled.as_ref().unwrap().requests.len(), 1);
    assert!(ready
        .capabilities
        .as_ref()
        .unwrap()
        .modes
        .contains(&"text_to_video".to_owned()));

    let compiled = ready.compiled.unwrap();
    draft.production_plan.shots[0].prompt =
        "The edited prompt must invalidate the compile.".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    let stale = film_harness::preflight_documents(
        &transport,
        &draft.production_plan,
        &draft.reference_pack,
        Some(compiled),
        Some(&selected),
        true,
    )
    .await
    .expect("staleness is a finding, not a transport failure");
    assert!(!stale.valid);
    assert!(stale
        .findings
        .iter()
        .any(|finding| finding.field == "compiled.planSha256"));
}

/// An [`ApiTransport`] over the in-process router that logs every file DOWNLOAD, and can answer
/// `GET /api/v1/projects…` with the project's `path` relocated to a directory this process cannot
/// read.
///
/// Relocated, that is how a REMOTE API host looks from the controller (sc-23404): `--api` may name
/// a private-network address, a `.local` name or a bare hostname, and the API host "may be a
/// different machine" (docs/film-harness.md), whose project directory is simply not on this
/// filesystem. Not relocated, it is the loopback case, unchanged. Everything else — the routes, the
/// job table, the assets — is the real in-process API either way.
pub(crate) struct CountingTransport {
    inner: RouterTransport,
    /// Where the project documents claim their directories are, or `None` to leave them alone.
    relocate_to: Option<PathBuf>,
    pub(crate) downloads: Arc<Mutex<Vec<String>>>,
}

impl CountingTransport {
    /// The API is on THIS machine: its project directory really is where it says it is.
    pub(crate) fn local(app: axum::Router) -> Self {
        Self {
            inner: RouterTransport { app },
            relocate_to: None,
            downloads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The API is on ANOTHER machine: every project directory it names is unreadable here.
    pub(crate) fn remote(app: axum::Router, unreachable_root: PathBuf) -> Self {
        Self {
            relocate_to: Some(unreachable_root),
            ..Self::local(app)
        }
    }
}

impl ApiTransport for CountingTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        // Only the project documents carry a host-local `path` — the list, one project, and the
        // CREATE that answers with the freshly made one, which is the response `ensure_project`
        // reads on a first run. Narrowed by depth so a nested route (`…/projects/{id}/assets`,
        // five segments) is passed through untouched.
        let relocate_to = self.relocate_to.clone().filter(|_| {
            request.path.starts_with("/api/v1/projects") && request.path.matches('/').count() <= 4
        });
        let inner = self.inner.call(request);
        Box::pin(async move {
            let mut response = inner.await?;
            if let Some(root) = relocate_to.filter(|_| (200..300).contains(&response.status)) {
                let relocate = |project: &mut Value| {
                    if let Some(id) = project.get("id").and_then(Value::as_str) {
                        let path = root.join(id);
                        project["path"] = json!(path.to_string_lossy());
                    }
                };
                match &mut response.body {
                    Value::Array(projects) => projects.iter_mut().for_each(relocate),
                    project @ Value::Object(_) => relocate(project),
                    _ => {}
                }
            }
            Ok(response)
        })
    }

    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        self.downloads.lock().push(path.clone());
        self.inner.get_bytes(path)
    }
}

/// An [`ApiTransport`] wrapping [`RouterTransport`] that can fail or rewrite chosen responses, so
/// a test can inject the failures a live API can produce (a 500 mid-run, a two-phase asset handoff
/// that never finishes) without changing production code to make itself testable.
struct ScriptedTransport {
    inner: RouterTransport,
    /// `(method, path substring)` → status to answer with instead of calling the router.
    failures: Vec<(&'static str, String, u16)>,
    /// A path substring and the rewrite applied to the successful responses it matches.
    rewrite: Option<ResponseRewrite>,
}

/// A path substring plus the mutation applied to every successful response whose path contains it.
type ResponseRewrite = (String, fn(&mut Value));

impl ScriptedTransport {
    fn failing(app: axum::Router, method: &'static str, path: &str, status: u16) -> Self {
        Self {
            inner: RouterTransport { app },
            failures: vec![(method, path.to_owned(), status)],
            rewrite: None,
        }
    }

    fn rewriting(app: axum::Router, path: &str, rewrite: fn(&mut Value)) -> Self {
        Self {
            inner: RouterTransport { app },
            failures: Vec::new(),
            rewrite: Some((path.to_owned(), rewrite)),
        }
    }
}

impl ApiTransport for ScriptedTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        if let Some((_, _, status)) = self
            .failures
            .iter()
            .find(|(method, path, _)| *method == request.method && request.path.contains(path))
        {
            let status = *status;
            return Box::pin(async move {
                Ok(ApiResponse {
                    status,
                    body: json!({ "detail": "injected failure" }),
                })
            });
        }
        let rewrite = self
            .rewrite
            .as_ref()
            .filter(|(path, _)| request.path.contains(path.as_str()))
            .map(|(_, rewrite)| *rewrite);
        let inner = self.inner.call(request);
        Box::pin(async move {
            let mut response = inner.await?;
            if let Some(rewrite) = rewrite {
                if (200..300).contains(&response.status) {
                    rewrite(&mut response.body);
                }
            }
            Ok(response)
        })
    }

    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        self.inner.get_bytes(path)
    }
}

/// How the fake worker treats one video job, keyed by the shot id the harness stamps into
/// `advanced.filmHarness.shotId`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum VideoBehavior {
    /// Complete after `delay`, reporting `peak_pct` as the observed peak in its metrics block.
    Complete { delay_secs: u64, peak_pct: f64 },
    /// Fail once (first attempt), then complete.
    FailFirst,
    /// Fail every time — a shot no retry will rescue.
    FailAlways,
    /// Never complete; honour a cancel request by reporting `canceled`.
    Hang,
    /// Never complete and never honour a cancel — a worker whose cooperative checkpoint is minutes
    /// away, or one wedged in a command buffer. The job stays `running` forever.
    HangIgnoringCancel,
}

#[derive(Debug, Clone)]
pub(crate) enum RunningHook {
    CancelControl(RunControl),
    WriteCancelSentinel(PathBuf),
    Notify(Arc<tokio::sync::Notify>),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WorkerScript {
    pub(crate) behaviors: Vec<(String, VideoBehavior)>,
    /// Run this test action at the exact fake-worker transition where the named shot becomes
    /// running. Cancellation tests use this lifecycle hook instead of racing setup and earlier
    /// shots with a wall-clock polling timeout.
    pub(crate) running_hook: Option<(String, RunningHook)>,
    /// Completed video jobs, recorded only after terminal progress, asset persistence and metrics
    /// have all landed. The notify turns adoption tests into event-driven barriers instead of a
    /// fixed wall-clock polling window.
    settled_video_jobs: Vec<(String, String)>,
    settled_video_jobs_changed: Arc<tokio::sync::Notify>,
    /// Jobs the fake worker has claimed, in order: (type, job id, payload).
    pub(crate) claimed: Vec<(String, String, Value)>,
    failed_once: Vec<String>,
    /// Make `timeline_export` jobs fail, for the failed-export resume path (sc-22711).
    pub(crate) export_fails: bool,
    /// Make `timeline_export` jobs hang until cancelled (honouring the cancel), for the
    /// export-overrun path (sc-22715).
    pub(crate) export_hangs: bool,
    /// `droppedAudioLayers` the fake export reports in its result (sc-22715).
    pub(crate) export_dropped_layers: Vec<Value>,
    /// Delay before the fake `image_vqa` job answers, honouring a cancel meanwhile (sc-22715) —
    /// what lets a test spend a review's `maxSeconds` or `maxAnswerSeconds`.
    pub(crate) vqa_delay: Option<Duration>,
    /// Delay before the fake `frame_extract` job writes its frame, honouring a cancel meanwhile
    /// (sc-22715) — what lets a test spend a review's `maxSeconds` DURING an extraction.
    pub(crate) frame_delay: Option<Duration>,
    /// Publish the extraction's running transition for tests that synchronize on active work.
    pub(crate) frame_reports_running: bool,
    /// Capabilities the fake registers with, when a test needs it to leave one to a REAL worker
    /// (`None` advertises every job type the harness drives).
    pub(crate) capabilities: Option<Vec<&'static str>>,
    /// Write REAL clips (through ffmpeg) for completed takes instead of placeholder bytes, so a
    /// real `timeline_export` can render them (sc-22715).
    pub(crate) real_takes: bool,
    /// sc-22714 VQA answers, keyed by the `[questionId@frameId]` tag the reviewer stamps onto
    /// every question. Most specific first: `questionId@frameId`, then `questionId`, then
    /// `vqa_fallback`. An unmatched question answers "I cannot tell", so a test that forgets one
    /// records it as UNOBSERVED rather than as silent agreement.
    pub(crate) vqa_answers: std::collections::BTreeMap<String, String>,
    /// Fail every `image_vqa` job, for the "the backend answered nothing at all" path.
    pub(crate) vqa_fails: bool,
    /// sc-23403: roles (`advanced.filmHarness.role`) whose `image_generate` job fails outright.
    pub(crate) image_fails: Vec<String>,
    /// sc-23403: roles whose `image_generate` job hangs until it is cancelled, so a test can spend
    /// the spec's own `maxJobSeconds` on one.
    pub(crate) image_hangs: Vec<String>,
    /// sc-23403: roles whose `image_generate` job hangs and IGNORES `cancelRequested` — the image
    /// lane's [`VideoBehavior::HangIgnoringCancel`]. A render still on the GPU after the cancel
    /// grace is the one state in which no further plate may be dispatched beside it.
    pub(crate) image_ignores_cancel: Vec<String>,
    /// sc-23403: peak the fake image worker reports in its metrics block, as a percentage of host
    /// memory. `None` reports the ordinary small peak.
    pub(crate) image_peak_pct: Option<f64>,
    /// Questions the fake worker has been asked, in order: (tag, question).
    pub(crate) vqa_asked: Vec<(String, String)>,
    /// Replies the fake worker returns for `prompt_refine` jobs whose task is `film_plan`, in
    /// order. The last one repeats once the list runs out, which is what lets a test prove the
    /// repair loop STOPS rather than looping on a reply that never validates.
    pub(crate) plan_replies: Vec<String>,
    pub(crate) plan_calls: usize,
    /// Reply for the per-shot prompt-refinement (the ordinary rewrite task). `{prompt}` is replaced
    /// by the shot's own prompt.
    pub(crate) refine_template: Option<String>,
    /// sc-24029: the `generation.finishReason` the fake refine result reports. `None` reports
    /// `"stop"`, exactly as the real worker does for a decode that ended on EOS; `Some("length")`
    /// is the truncated-but-non-empty rewrite the planner must refuse.
    pub(crate) refine_finish_reason: Option<String>,
    /// sc-23404: fail every `audio_generate` job — the "the TTS model refused / fell over" path.
    pub(crate) audio_fails: bool,
    /// sc-23404: never complete an `audio_generate` job (honouring a cancel), so a test can spend
    /// the plan's `maxShotSeconds` inside a synthesis rather than inside a render.
    pub(crate) audio_hangs: bool,
    /// sc-23404: `audio_generate` jobs the fake has claimed, in order: (role, payload).
    pub(crate) audio_claimed: Vec<(String, Value)>,
}

impl WorkerScript {
    fn next_plan_reply(&mut self) -> String {
        let index = self
            .plan_calls
            .min(self.plan_replies.len().saturating_sub(1));
        self.plan_calls += 1;
        self.plan_replies
            .get(index)
            .cloned()
            .unwrap_or_else(|| "the planner has nothing to say".to_owned())
    }
}

impl WorkerScript {
    fn behavior_for(&self, shot_id: &str) -> VideoBehavior {
        self.behaviors
            .iter()
            .find(|(id, _)| id == shot_id)
            .map(|(_, behavior)| *behavior)
            .unwrap_or(VideoBehavior::Complete {
                delay_secs: 1,
                peak_pct: 40.0,
            })
    }
}

const WORKER_ID: &str = "fake-mlx-worker";
const HOST_MEMORY_MB: u64 = 128 * 1024;

/// Every job type the harness drives, which the fake advertises unless a test narrows it.
///
/// `image_vqa` (sc-22714) is what the reviewer's questions ride; `frame_extract` is what turns a
/// take into timestamped frame evidence; `prompt_refine` (sc-22713) is the planner seam;
/// `audio_generate` (sc-23404) is the TTS seam a synthesized dialogue line rides; `image_generate`
/// (sc-23403) is the generation seam `make-references` drives. All of them are job types the real
/// worker already advertises (the macOS worker builds the candle audio lane unconditionally, so
/// `audio_generate` is advertised on the mlx lane too).
pub(crate) const FAKE_CAPABILITIES: &[&str] = &[
    "video_generate",
    "timeline_export",
    "frame_extract",
    "image_vqa",
    "prompt_refine",
    "audio_generate",
    // sc-23403: the generation seam `make-references` drives. A real image worker advertises the
    // same capability and answers with the same `assetWrites` fact shape.
    "image_generate",
];

async fn register_fake_worker(app: &axum::Router, capabilities: &[&str]) {
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": WORKER_ID,
            "gpuId": "mlx",
            "gpuName": "Apple M-series (fake)",
            "capabilities": capabilities,
            "loadedModels": [],
            "utilization": { "memoryTotalMb": HOST_MEMORY_MB }
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
}

fn project_path(
    app: &axum::Router,
    project_id: &str,
) -> impl std::future::Future<Output = PathBuf> {
    let app = app.clone();
    let project_id = project_id.to_owned();
    async move {
        let (_, project) = request(
            app,
            "GET",
            &format!("/api/v1/projects/{project_id}"),
            Value::Null,
        )
        .await;
        PathBuf::from(project["path"].as_str().expect("project path"))
    }
}

/// Spawn the scripted worker loop. It registers, then claims and completes jobs until the task is
/// aborted. Budgets in these tests are real seconds: tokio's paused clock is NOT used, because the
/// API's blocking store calls let the auto-advancing clock race ahead of the harness's own
/// `Instant::now()` reads and spend a plan's budget during setup.
fn spawn_fake_worker(
    app: axum::Router,
    script: Arc<Mutex<WorkerScript>>,
) -> (
    tokio::task::JoinHandle<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let (registered_tx, registered_rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(async move {
        let capabilities = script
            .lock()
            .capabilities
            .clone()
            .unwrap_or_else(|| FAKE_CAPABILITIES.to_vec());
        register_fake_worker(&app, &capabilities).await;
        let _ = registered_tx.send(());
        loop {
            let (status, claim) = request(
                app.clone(),
                "POST",
                "/api/v1/jobs/claim",
                json!({ "workerId": WORKER_ID }),
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::OK, "{claim}");
            let Some(job) = claim.get("job").filter(|job| !job.is_null()).cloned() else {
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            };
            let job_id = job["id"].as_str().expect("job id").to_owned();
            let job_type = job["type"].as_str().expect("job type").to_owned();
            script
                .lock()
                .claimed
                .push((job_type.clone(), job_id.clone(), job["payload"].clone()));
            match job_type.as_str() {
                "video_generate" => run_fake_video_job(&app, &script, &job_id, &job).await,
                "timeline_export" => run_fake_export_job(&app, &script, &job_id, &job).await,
                // sc-22714: the two understanding seams the reviewer drives. Neither renders
                // anything — `frame_extract` writes a placeholder still where FFmpeg would, and
                // `image_vqa` answers from the script's table in the shape SenseNova-U1 posts.
                "frame_extract" => run_fake_frame_job(&app, &script, &job_id, &job).await,
                "image_vqa" => run_fake_vqa_job(&app, &script, &job_id, &job).await,
                "prompt_refine" => run_fake_refine_job(&app, &script, &job_id, &job).await,
                // sc-23404: the TTS seam. Writes a deterministic canonical PCM-16 WAV where the
                // real audio worker writes its clip and reports the same `assetWrites` fact, so
                // asset persistence, the audio sidecar and the two-phase result rewrite are
                // production code paths here exactly as they are for a render.
                "audio_generate" => run_fake_audio_job(&app, &script, &job_id, &job).await,
                // sc-23403: the image-generation seam the reference-fixture generator drives.
                "image_generate" => run_fake_image_job(&app, &script, &job_id, &job).await,
                other => panic!("fake worker claimed an unexpected job type {other}"),
            }
        }
    });
    (handle, registered_rx)
}

async fn post_progress(app: &axum::Router, job_id: &str, body: Value) -> Value {
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        body,
    )
    .await;
    assert!(
        status == axum::http::StatusCode::OK || status == axum::http::StatusCode::CONFLICT,
        "progress {job_id}: {status} {response}"
    );
    response
}

async fn run_fake_video_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let shot_id = payload["advanced"]["filmHarness"]["shotId"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let behavior = script.lock().behavior_for(&shot_id);
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    post_progress(
        app,
        job_id,
        json!({
            "status": "running", "stage": "generating", "progress": 0.2,
            "message": "fake render", "workerId": WORKER_ID, "backend": "mlx"
        }),
    )
    .await;
    let running_hook = {
        let mut script = script.lock();
        if script
            .running_hook
            .as_ref()
            .is_some_and(|(target, _)| target == &shot_id)
        {
            script.running_hook.take().map(|(_, hook)| hook)
        } else {
            None
        }
    };
    match running_hook {
        Some(RunningHook::CancelControl(control)) => control.cancel(),
        Some(RunningHook::WriteCancelSentinel(out_dir)) => {
            film_harness::request_cancel(&out_dir)
                .expect("cancel sentinel writes from worker hook");
        }
        Some(RunningHook::Notify(notify)) => notify.notify_one(),
        None => {}
    }
    match behavior {
        VideoBehavior::HangIgnoringCancel => loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
        },
        VideoBehavior::Hang => loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        },
        VideoBehavior::FailAlways => {
            post_progress(
                app,
                job_id,
                json!({
                    "status": "failed", "stage": "failed", "progress": 1,
                    "message": "fake engine fault", "error": "fake engine fault: persistent",
                    "workerId": WORKER_ID
                }),
            )
            .await;
            return;
        }
        VideoBehavior::FailFirst => {
            let first = {
                let mut script = script.lock();
                if script.failed_once.contains(&shot_id) {
                    false
                } else {
                    script.failed_once.push(shot_id.clone());
                    true
                }
            };
            if first {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "failed", "stage": "failed", "progress": 1,
                        "message": "fake engine fault", "error": "fake engine fault: transient",
                        "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
        VideoBehavior::Complete { delay_secs, .. } => {
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
        }
    }
    let peak_pct = match behavior {
        VideoBehavior::Complete { peak_pct, .. } => peak_pct,
        _ => 40.0,
    };
    let peak_memory_bytes = (HOST_MEMORY_MB as f64 * 1024.0 * 1024.0 * peak_pct / 100.0) as u64;
    let asset_id = format!("asset_{}", &job_id.replace('-', "")[..16]);
    let media_rel = format!("assets/videos/{asset_id}.mp4");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/videos")).expect("videos dir");
    let duration = payload["duration"].as_f64().unwrap_or(5.0);
    let fps = payload["fps"].as_u64().unwrap_or(24);
    if script.lock().real_takes {
        // A REAL clip at the requested geometry, timing and fps, with its own 900 Hz tone as the
        // "generated audio" a mute policy must keep out of the export (sc-22715). Built the way the
        // worker's measured mix tests build theirs.
        let out = project_dir.join(&media_rel);
        let width = payload["width"].as_u64().unwrap_or(576);
        let height = payload["height"].as_u64().unwrap_or(320);
        let ok = tokio::task::spawn_blocking(move || {
            std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    &format!("testsrc2=size={width}x{height}:rate={fps}:duration={duration}"),
                    "-f",
                    "lavfi",
                    "-i",
                    &format!("sine=frequency=900:duration={duration}:sample_rate=48000"),
                    "-shortest",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    &out.display().to_string(),
                ])
                .status()
                .is_ok_and(|status| status.success())
        })
        .await
        .expect("ffmpeg task joins");
        assert!(
            ok,
            "the fake worker could not write a real take with ffmpeg"
        );
    } else {
        std::fs::write(project_dir.join(&media_rel), b"not really an mp4").expect("fake mp4");
    }
    let frames = (duration * fps as f64).round() as u64;
    let fact = json!({
        "type": "video",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "video/mp4",
        "width": payload["width"], "height": payload["height"],
        "duration": payload["duration"], "fps": payload["fps"],
        "encodedFrameCount": frames, "encodedDuration": duration, "encodedFps": fps,
        "hasAudio": true,
        "quality": payload["quality"],
        "family": "minimax-h3",
        "seed": payload["seed"].as_i64().unwrap_or(1),
        "displayName": format!("fake take {shot_id}"),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": payload["mode"], "model": payload["model"], "adapter": "fake_minimax_h3",
        "prompt": payload["prompt"], "negativePrompt": "", "loras": [],
        "rawAdapterSettings": { "tier": "q4", "task": if payload["sourceAssetId"].is_string() { "fl2va" } else { "t2va" }, "advanced": payload["advanced"] },
        "sourceAssetId": payload["sourceAssetId"], "lastFrameAssetId": payload["lastFrameAssetId"],
        "fitMode": payload["fitMode"],
        "sourceClipAssetIds": [], "referenceAssetIds": payload["referenceAssetIds"].as_array().cloned().unwrap_or_default(),
        "referenceAudioAssetIds": [],
        "timelineContext": {}
    });
    let genset_id = format!("genset_{}", &job_id.replace('-', "")[..16]);
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake render done", "workerId": WORKER_ID, "backend": "mlx",
            "result": {
                "generationSetId": genset_id,
                "expectedCount": 1,
                "adapter": "fake_minimax_h3",
                "model": payload["model"],
                "generationSet": {
                    "id": genset_id, "mode": payload["mode"], "model": payload["model"],
                    "prompt": payload["prompt"], "negativePrompt": "", "count": 1,
                    "createdAt": sceneworks_core::time::utc_now()
                },
                "assetWrites": [fact]
            }
        }),
    )
    .await;
    // The hardware peak goes where the real worker puts it: a metrics block POSTed to
    // `/api/v1/jobs/:id/metrics` AFTER the terminal progress (sceneworks-worker `lib.rs`
    // `metrics_probe.finish()` → `post_generation_metrics`). No `ProgressRequest` anywhere in the
    // worker ever sets `peakGpuMemoryPct`, so a fake that reported the peak on the progress update
    // would be testing a signal that does not exist on a real render.
    post_generation_metrics(
        app,
        job_id,
        json!({
            "backend": "mlx",
            "totalMs": 1_000,
            "peakMemoryBytes": peak_memory_bytes,
            "peakMemoryPct": peak_pct,
        }),
    )
    .await;
    let settled_changed = {
        let mut script = script.lock();
        script.settled_video_jobs.push((shot_id, job_id.to_owned()));
        script.settled_video_jobs_changed.clone()
    };
    // Retain a permit when the adoption test has not started waiting yet; a broadcast-only notify
    // could land in the controller-crash window and recreate the race this barrier replaces.
    settled_changed.notify_one();
}

/// The `image_generate` job, faked (sc-23403): write a deterministic plate at the REQUESTED
/// geometry where the GPU worker would write its render, and report it as an `assetWrites` fact in
/// the shape `build_image_sidecar_parts` reads. Asset persistence, the sidecar, the recipe, the
/// index, the two-phase result rewrite and the file route the plate is downloaded back through are
/// all production code paths.
async fn run_fake_image_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let role = payload["advanced"]["filmHarness"]["role"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let (fails, hangs, ignores_cancel, peak_pct) = {
        let script = script.lock();
        (
            script.image_fails.contains(&role),
            script.image_hangs.contains(&role),
            script.image_ignores_cancel.contains(&role),
            script.image_peak_pct,
        )
    };
    if fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake image engine fault", "error": "fake image engine fault",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
    if ignores_cancel {
        // A render that keeps the GPU past the cancel grace — the image lane's
        // `VideoBehavior::HangIgnoringCancel`. The job stays `running` forever, so the generator
        // must refuse rather than dispatch a second plate beside a render it cannot stop.
        post_progress(
            app,
            job_id,
            json!({
                "status": "running", "stage": "generating", "progress": 0.2,
                "message": "fake plate, wedged", "workerId": WORKER_ID, "backend": "mlx"
            }),
        )
        .await;
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    if hangs {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
    }
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let width = payload["width"].as_u64().unwrap_or(1024) as u32;
    let height = payload["height"].as_u64().unwrap_or(1024) as u32;
    let seed = payload["seed"]
        .as_i64()
        .or_else(|| payload["seeds"][0].as_i64())
        .unwrap_or(1);
    let asset_id = format!("asset_{}", &job_id.replace('-', "")[..16]);
    let genset_id = format!("genset_{}", &job_id.replace('-', "")[..16]);
    let media_rel = format!("assets/images/{genset_id}/{asset_id}.png");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join(format!("assets/images/{genset_id}")))
        .expect("images dir");
    // A real PNG at the geometry the job asked for: the pack the generator publishes has to hold
    // something a later `validate` and a later import can actually read.
    let tint = [
        (seed.unsigned_abs() % 251) as u8,
        (role.len() as u8).wrapping_mul(17),
        180,
    ];
    std::fs::write(
        project_dir.join(&media_rel),
        film_harness::fixture_plate_png_sized(&role, tint, width, height).expect("plate encodes"),
    )
    .expect("fake plate");
    let fact = json!({
        "type": "image",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": width, "height": height,
        "normalizedWidth": width, "normalizedHeight": height,
        "count": 1,
        "index": 0,
        "seed": seed,
        "family": "krea-2",
        "displayName": format!("fake plate {role}"),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": payload["mode"], "model": payload["model"], "adapter": "fake_mlx_krea",
        "prompt": payload["prompt"], "negativePrompt": payload["negativePrompt"], "loras": [],
        "rawAdapterSettings": { "advanced": payload["advanced"] }
    });
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake plate done", "workerId": WORKER_ID, "backend": "mlx",
            "result": {
                "generationSetId": genset_id,
                "expectedCount": 1,
                "adapter": "fake_mlx_krea",
                "model": payload["model"],
                "generationSet": {
                    "id": genset_id, "mode": payload["mode"], "model": payload["model"],
                    "prompt": payload["prompt"], "negativePrompt": "", "count": 1,
                    "createdAt": sceneworks_core::time::utc_now()
                },
                "assetWrites": [fact]
            }
        }),
    )
    .await;
    let peak_pct = peak_pct.unwrap_or(12.0);
    post_generation_metrics(
        app,
        job_id,
        json!({
            "backend": "mlx",
            "totalMs": 1_000,
            "peakMemoryBytes": (HOST_MEMORY_MB as f64 * 1024.0 * 1024.0 * peak_pct / 100.0) as u64,
            "peakMemoryPct": peak_pct,
        }),
    )
    .await;
}

/// The worker's `post_generation_metrics`: an upsert of the run's metrics block, posted after the
/// job is already terminal.
async fn post_generation_metrics(app: &axum::Router, job_id: &str, metrics: Value) {
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/metrics"),
        metrics,
    )
    .await;
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "metrics {job_id}: {response}"
    );
}

/// The `frame_extract` job, faked: write a placeholder still where FFmpeg would and report it as
/// an `assetWrites` fact, exactly as `run_frame_extract` does. Asset persistence, the sidecar, the
/// index and the two-phase result rewrite are all production code paths (sc-22714).
async fn run_fake_frame_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let reports_running = script.lock().frame_reports_running;
    if reports_running {
        post_progress(
            app,
            job_id,
            json!({
                "status": "running", "stage": "extracting", "progress": 0.1,
                "message": "Extracting frame.", "workerId": WORKER_ID
            }),
        )
        .await;
    }
    let delay = script.lock().frame_delay;
    if let Some(delay) = delay {
        let started = std::time::Instant::now();
        while started.elapsed() < delay {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
    }
    let payload = &job["payload"];
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let timestamp = payload["sourceTimestamp"].as_f64().unwrap_or(0.0);
    let asset_id = format!("asset_frame_{}", &job_id.replace('-', "")[..12]);
    let media_rel = format!("assets/frames/{asset_id}.png");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/frames")).expect("frames dir");
    std::fs::write(
        project_dir.join(&media_rel),
        film_harness::fixture_plate_png("review-frame", [90, 82, 70]).expect("plate encodes"),
    )
    .expect("fake frame");
    let fact = json!({
        "type": "frame",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": 576, "height": 320,
        "displayName": format!("Frame {timestamp:.2}s"),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": "frame_extract", "model": "timeline-frame-extract",
        "adapter": "ffmpeg-frame-extract",
        "prompt": format!("Extract frame at {timestamp:.2}s"),
        "negativePrompt": "", "loras": [],
        "normalizedSettings": {
            "timelineId": payload["timelineId"],
            "timelineItemId": payload["timelineItemId"],
            "playheadSeconds": payload["playheadSeconds"],
            "sourceTimestamp": timestamp,
        },
    });
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Frame saved.", "workerId": WORKER_ID,
            "result": { "assetWrites": [fact], "adapter": "ffmpeg-frame-extract" }
        }),
    )
    .await;
}

/// The `audio_generate` job, faked (sc-23404): write a deterministic canonical PCM-16 WAV where the
/// audio worker writes its clip and report it through the same `assetWrites` fact
/// `audio_jobs::audio_asset_fact` builds. No TTS weights load; asset persistence, the audio sidecar
/// and the two-phase result rewrite are the production code paths.
///
/// The clip's pitch is derived from the requested VOICE and its length from the text, so a test can
/// tell one synthesized line from another by decoding the export — the same trick the fixture's
/// placeholder tones used, now keyed on what was actually asked for rather than on a checked-in
/// file.
async fn run_fake_audio_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let role = payload["advanced"]["filmHarness"]["role"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    script
        .lock()
        .audio_claimed
        .push((role.clone(), payload.clone()));
    post_progress(
        app,
        job_id,
        json!({
            "status": "running", "stage": "generating", "progress": 0.3,
            "message": "fake synthesis", "workerId": WORKER_ID, "backend": "candle"
        }),
    )
    .await;
    if script.lock().audio_fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake tts fault", "error": "fake tts fault: no voice bank",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
    if script.lock().audio_hangs {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
    }
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let prompt = payload["prompt"].as_str().unwrap_or_default().to_owned();
    let voice = payload["voice"].as_str().map(str::to_owned);
    let asset_id = format!("asset_tts_{}", &job_id.replace('-', "")[..12]);
    let media_rel = format!("assets/audios/{asset_id}.wav");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/audios")).expect("audios dir");
    let (hz, seconds) = fake_speech_shape(voice.as_deref(), &prompt);
    let wav = film_harness::fixture_sound_wav(seconds, hz, 9000);
    std::fs::write(project_dir.join(&media_rel), &wav).expect("fake wav");
    let fact = json!({
        "type": "audio",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "audio/wav",
        "duration": seconds,
        "sampleRate": film_harness::FIXTURE_SOUND_RATE,
        "channels": 1,
        "family": "kokoro",
        "displayName": prompt.chars().take(56).collect::<String>(),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": "speech",
        "model": payload["model"],
        "adapter": "fake_kokoro",
        "prompt": prompt,
        "voice": payload["voice"],
        "language": Value::Null,
        "targetDurationSecs": Value::Null,
        "seed": Value::Null,
        "rawAdapterSettings": {
            "model": payload["model"],
            "voice": payload["voice"],
            "sampleRate": film_harness::FIXTURE_SOUND_RATE,
            "advanced": payload["advanced"],
        },
    });
    let genset_id = format!("genset_tts_{}", &job_id.replace('-', "")[..12]);
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake synthesis done", "workerId": WORKER_ID, "backend": "candle",
            "result": {
                "generationSetId": genset_id,
                "expectedCount": 1,
                "adapter": "fake_kokoro",
                "model": payload["model"],
                "generationSet": {
                    "id": genset_id, "mode": "speech", "model": payload["model"],
                    "prompt": payload["prompt"], "count": 1,
                    "createdAt": sceneworks_core::time::utc_now()
                },
                "assetWrites": [fact]
            }
        }),
    )
    .await;
}

/// Pitch and length for one faked synthesized line: a distinct frequency per voice id, and a length
/// that grows with the text so two lines in the same voice are still distinguishable by duration.
pub(crate) fn fake_speech_shape(voice: Option<&str>, text: &str) -> (u32, f64) {
    let hz = match voice {
        Some("am_michael") => 400,
        Some("af_heart") => 500,
        Some(_) => 600,
        None => 700,
    };
    // 25 characters per second, floored at half a second — short enough that a six-shot fixture's
    // lines all fit inside their shots, long enough to measure.
    let seconds = ((text.trim().chars().count() as f64) / 25.0).max(0.5);
    (hz, (seconds * 10.0).round() / 10.0)
}

/// The `image_vqa` job, faked: answer from the script's table in the shape
/// `sensenova_jobs::vqa_result_json` posts. No weights ran, and `realModelInference: false` says
/// so — which is what keeps a scripted review from ever reading as a real one.
async fn run_fake_vqa_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    post_progress(
        app,
        job_id,
        json!({
            "status": "running", "stage": "generating", "progress": 0.6,
            "message": "Analyzing image.", "workerId": WORKER_ID
        }),
    )
    .await;
    let question = job["payload"]["question"].as_str().unwrap_or("").to_owned();
    let tag = question
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']'))
        .map(|(tag, _)| tag.to_owned())
        .unwrap_or_default();
    let delay = script.lock().vqa_delay;
    if let Some(delay) = delay {
        // A slow model: answer after `delay`, unless the reviewer cancels first — which is what a
        // `maxAnswerSeconds` timeout does through the API (sc-22715).
        let started = std::time::Instant::now();
        while started.elapsed() < delay {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
    }
    let (answer, fails) = {
        let mut script = script.lock();
        script.vqa_asked.push((tag.clone(), question.clone()));
        let question_id = tag.split('@').next().unwrap_or("").to_owned();
        let answer = script
            .vqa_answers
            .get(&tag)
            .or_else(|| script.vqa_answers.get(&question_id))
            .or_else(|| script.vqa_answers.get("vqa_fallback"))
            .cloned()
            .unwrap_or_else(|| "I cannot tell from this frame.".to_owned());
        (answer, script.vqa_fails)
    };
    if fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake vqa fault", "error": "fake vqa fault: no weights",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Answer ready.", "workerId": WORKER_ID, "backend": "mlx",
            "result": {
                "answer": answer,
                "question": question,
                "sourceAssetId": job["payload"]["sourceAssetId"],
                "model": job["payload"]["model"],
                "realModelInference": false,
            }
        }),
    )
    .await;
}

/// The scripted stand-in for the native `prompt_refine` worker. It replaces ONLY the decode: the
/// job was created through the real `POST /api/v1/prompts/refine` route, claimed through the real
/// worker API, and its result is read back through the real job snapshot — so the planner is
/// exercised against the seam it will use on the GPU, with the model's answer scripted.
async fn run_fake_refine_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let task = payload["task"].as_str().unwrap_or_default().to_owned();
    let prompt = payload["prompt"].as_str().unwrap_or_default().to_owned();
    let refined = if task == "film_plan" {
        script.lock().next_plan_reply()
    } else {
        script
            .lock()
            .refine_template
            .clone()
            .unwrap_or_else(|| "{prompt}".to_owned())
            .replace("{prompt}", &prompt)
    };
    // The real worker records how the decode ENDED on the success result too (sc-24029), because
    // a rewrite that stopped on `length` is non-empty and therefore completes normally.
    let finish_reason = script
        .lock()
        .refine_finish_reason
        .clone()
        .unwrap_or_else(|| "stop".to_owned());
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake refine done", "workerId": WORKER_ID, "backend": "mlx",
            "result": {
                "originalPrompt": prompt, "refinedPrompt": refined,
                "generation": {
                    "finishReason": finish_reason,
                    "usage": { "promptTokens": 900, "generatedTokens": 1536 },
                    "maxNewTokens": 1536
                },
                "executionIdentity": {
                    "provider": "native", "model": "fixture/model-keyed-refiner",
                    "backend": "fixture", "thinkingMode": "disabled"
                }
            }
        }),
    )
    .await;
    // The real worker's `run_utility_job` posts a metrics block for EVERY job type after its
    // terminal progress, a refine decode included; the planner reads the peak off it (sc-22715).
    post_generation_metrics(
        app,
        job_id,
        json!({
            "backend": "mlx",
            "totalMs": 1_000,
            "peakMemoryBytes": FAKE_REFINE_PEAK_BYTES,
            "peakMemoryPct": 7.0,
        }),
    )
    .await;
}

/// The peak the fake refine job reports, so a test can prove the number in `compiled.json` is
/// the one the metrics route carried rather than something the planner made up.
pub(crate) const FAKE_REFINE_PEAK_BYTES: u64 = 9_000_000_000;

async fn run_fake_export_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    if script.lock().export_hangs {
        // An export that never finishes on its own but honours a cancel — the shape a per-job
        // budget overrun takes (sc-22715).
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
    }
    if script.lock().export_fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake ffmpeg fault", "error": "fake ffmpeg fault: no such codec",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
    let payload = &job["payload"];
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let render_rel = format!(
        "assets/renders/fake-export-{}.mp4",
        &job_id.replace('-', "")[..8]
    );
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/renders")).expect("renders dir");
    std::fs::write(project_dir.join(&render_rel), b"not really an mp4").expect("fake render");
    let asset_id = format!("asset_render_{}", &job_id.replace('-', "")[..12]);
    let dropped = script.lock().export_dropped_layers.clone();
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Timeline MP4 export saved.", "workerId": WORKER_ID,
            "result": {
                "assetIds": [asset_id],
                "assets": [{ "id": asset_id, "type": "render", "file": { "path": render_rel } }],
                "timelineId": payload["timelineId"],
                "renderPath": render_rel,
                "adapter": "ffmpeg_timeline",
                // The real worker reports the layers the mix went without (sc-22715).
                "droppedAudioLayers": dropped
            }
        }),
    )
    .await;
}

pub(crate) struct Harness {
    pub(crate) app: axum::Router,
    pub(crate) state: crate::AppState,
    pub(crate) transport: RouterTransport,
    pub(crate) temp_dir: tempfile::TempDir,
    pub(crate) script: Arc<Mutex<WorkerScript>>,
    /// The fake worker's task, when one runs. Behind a lock so a test that started WITHOUT a
    /// worker can script the fake first and spawn it afterwards (`spawn_worker`).
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    server: Option<tokio::task::JoinHandle<()>>,
}

impl Harness {
    pub(crate) async fn start(with_worker: bool, behaviors: Vec<(&str, VideoBehavior)>) -> Self {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        // `/api/v1/models` serves the manifests under the config dir, so seed the REAL shipped
        // builtin manifest: the harness validates the fixture against `minimax_h3`'s actual
        // declared menus, caps and memory minimum, not a stand-in.
        let manifests_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&manifests_dir).expect("manifest dir creates");
        std::fs::write(
            manifests_dir.join("builtin.models.jsonc"),
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        )
        .expect("builtin models writes");
        let (app, state) =
            create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
        // The fixture's image-conditioned shot is only MLX-routed; judge the enqueue as a Mac on
        // every lane so the suite means the same thing on ubuntu and on the hosted macOS job.
        *state.video_platform_override.lock() = Some("macos");
        let script = Arc::new(Mutex::new(WorkerScript {
            behaviors: behaviors
                .into_iter()
                .map(|(id, behavior)| (id.to_owned(), behavior))
                .collect(),
            ..WorkerScript::default()
        }));
        let (worker, registered) = if with_worker {
            let (worker, registered) = spawn_fake_worker(app.clone(), script.clone());
            (Some(worker), Some(registered))
        } else {
            (None, None)
        };
        if let Some(registered) = registered {
            registered
                .await
                .expect("fake worker stopped before registration completed");
        }
        Self {
            transport: RouterTransport { app: app.clone() },
            app,
            state,
            temp_dir,
            script,
            worker: Mutex::new(worker),
            server: None,
        }
    }

    /// Start the same in-process fixture with an ephemeral loopback listener. API handlers that
    /// call back through the configured MCP URL (review/repair) then exercise their real HTTP
    /// transport while the fake worker remains deterministic and CPU-only.
    pub(crate) async fn start_http(
        with_worker: bool,
        behaviors: Vec<(&str, VideoBehavior)>,
    ) -> Self {
        Self::start_http_with_cancel_boundary(with_worker, behaviors, None).await
    }

    async fn start_http_with_cancel_boundary(
        with_worker: bool,
        behaviors: Vec<(&str, VideoBehavior)>,
        boundary: Option<Arc<CancelBoundaryTransport>>,
    ) -> Self {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let manifests_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&manifests_dir).expect("manifest dir creates");
        std::fs::write(
            manifests_dir.join("builtin.models.jsonc"),
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        )
        .expect("builtin models writes");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener binds");
        let address = listener.local_addr().expect("loopback address");
        let mut settings = test_settings(&temp_dir);
        settings.mcp_api_url = format!("http://{address}");
        let (app, state) = create_app_with_state(settings).expect("app and state create");
        *state.video_platform_override.lock() = Some("macos");
        let mut server_app = app.clone();
        if let Some(boundary) = boundary {
            server_app = server_app.layer(axum::middleware::from_fn(
                move |request: Request<Body>, next: axum::middleware::Next| {
                    let boundary = boundary.clone();
                    async move {
                        let path = request.uri().path().to_owned();
                        let hold = request.method() == "GET"
                            && path.contains(boundary.needle)
                            && boundary.armed.swap(false, Ordering::SeqCst);
                        let mut response = next.run(request).await;
                        // Successful admission fixtures advertise availability without any weights.
                        if path == "/api/v1/models" && response.status().is_success() {
                            let (parts, body) = response.into_parts();
                            let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                            let mut catalog: Value = serde_json::from_slice(&bytes).unwrap();
                            for entry in catalog.as_array_mut().unwrap() {
                                entry["installState"] = json!("installed");
                                if let Some(variants) = entry["variants"].as_array_mut() {
                                    for variant in variants {
                                        variant["installed"] = json!(true);
                                    }
                                }
                            }
                            response = axum::response::Response::from_parts(
                                parts,
                                Body::from(serde_json::to_vec(&catalog).unwrap()),
                            );
                            response
                                .headers_mut()
                                .remove(axum::http::header::CONTENT_LENGTH);
                        }
                        if hold {
                            assert!(response.status().is_success());
                            boundary.reached.notify_one();
                            boundary.release.notified().await;
                        }
                        response
                    }
                },
            ));
        }
        let server = tokio::spawn(async move {
            axum::serve(listener, server_app)
                .await
                .expect("loopback API serves");
        });
        let script = Arc::new(Mutex::new(WorkerScript {
            behaviors: behaviors
                .into_iter()
                .map(|(id, behavior)| (id.to_owned(), behavior))
                .collect(),
            ..WorkerScript::default()
        }));
        let (worker, registered) = if with_worker {
            let (worker, registered) = spawn_fake_worker(app.clone(), script.clone());
            (Some(worker), Some(registered))
        } else {
            (None, None)
        };
        if let Some(registered) = registered {
            registered
                .await
                .expect("fake worker stopped before registration completed");
        }
        Self {
            transport: RouterTransport { app: app.clone() },
            app,
            state,
            temp_dir,
            script,
            worker: Mutex::new(worker),
            server: Some(server),
        }
    }

    /// Start the fake worker AFTER the script has been shaped — for a test that needs the fake
    /// to register with a narrower capability set, or to render real clips (sc-22715). Waits for
    /// the registration exactly as `start(true, …)` does.
    pub(crate) async fn spawn_worker(&self) {
        let (handle, registered) = spawn_fake_worker(self.app.clone(), self.script.clone());
        *self.worker.lock() = Some(handle);
        registered
            .await
            .expect("fake worker stopped before registration completed");
    }

    pub(crate) fn options(
        &self,
        plan_path: PathBuf,
        pack_path: PathBuf,
        shots: Option<&[&str]>,
    ) -> RunOptions {
        RunOptions {
            plan_path,
            reference_pack_path: pack_path,
            compiled_path: None,
            project_id: None,
            shot_ids: shots.map(|ids| ids.iter().map(|id| (*id).to_owned()).collect()),
            out_dir: self.out_dir(),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
        }
    }

    pub(crate) fn out_dir(&self) -> PathBuf {
        self.temp_dir.path().join("run-out")
    }

    /// What `resume` / `replace-take` are driven with in these tests: the same run directory, a
    /// tight poll cadence, and a control only the test can trip (sc-22711).
    pub(crate) fn resume_options(&self) -> ResumeOptions {
        ResumeOptions {
            out_dir: self.out_dir(),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
            control: RunControl::new(),
        }
    }

    /// Resume until the run reaches a state it will not leave on its own, so a test asserts about
    /// the end of the story rather than about how many crashes it took to get there. Bounded: a
    /// resume that makes no progress is a failure, not a retry.
    pub(crate) async fn resume_to_completion(&self) -> RunRecord {
        let mut last = None;
        for round in 0..12 {
            match film_harness::resume(&self.transport, &self.resume_options()).await {
                Ok(record) => {
                    if record.outcome == RunOutcome::Completed || !record.is_resumable() {
                        return record;
                    }
                    last = Some(record);
                }
                Err(error) => panic!("resume {round} failed: {error}"),
            }
        }
        panic!("run never settled after 12 resumes: {last:#?}");
    }

    pub(crate) fn video_job_count(&self) -> usize {
        self.script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "video_generate")
            .count()
    }

    /// Video jobs the API holds, which — unlike the claim log — counts a job the worker has not
    /// picked up yet.
    pub(crate) async fn api_video_job_count(&self) -> usize {
        self.jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "video_generate")
            .count()
    }

    pub(crate) async fn project_count(&self) -> usize {
        let (_, projects) = request(self.app.clone(), "GET", "/api/v1/projects", Value::Null).await;
        projects.as_array().map(Vec::len).unwrap_or_default()
    }

    /// Every timeline the project holds. `POST /timelines` always creates a new row, so this is
    /// what says whether a replay adopted the run's timeline or created a second one.
    async fn timelines(&self, project_id: &str) -> Vec<Value> {
        let (_, timelines) = request(
            self.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines"),
            Value::Null,
        )
        .await;
        timelines.as_array().cloned().unwrap_or_default()
    }

    /// The timelines of whichever project the record names, or none when it named no project.
    async fn timelines_for(&self, record: &RunRecord) -> Vec<Value> {
        match record.project_id.as_deref() {
            Some(project_id) => self.timelines(project_id).await,
            None => Vec::new(),
        }
    }

    pub(crate) fn export_job_count(&self) -> usize {
        self.script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "timeline_export")
            .count()
    }

    /// Rewrite the record on disk, exactly as a test that needs a run to look older than it is has
    /// to: the harness reads `run.json` back on every `resume`.
    pub(crate) fn edit_run_record(&self, edit: impl FnOnce(&mut Value)) {
        let path = self.out_dir().join("run.json");
        let mut record: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("run.json")).expect("json");
        edit(&mut record);
        std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    /// A two-shot plan with a one-reference pack, written into the temp dir.
    ///
    /// The wall-clock-budget tests need the budget to be spent by the RENDER, not by setup:
    /// importing the shipped seven-reference pack through the real import route costs several
    /// seconds in a debug build under a loaded runner, which would otherwise eat a small budget
    /// before the first shot is ever dispatched. Same model, same menus — just less to import.
    pub(crate) fn minimal_documents(&self, limits: Value) -> (PathBuf, PathBuf) {
        let dir = self.temp_dir.path().join("minimal");
        std::fs::create_dir_all(dir.join("references")).expect("minimal dir");
        std::fs::copy(
            Path::new(FIXTURE_DIR).join("references/workshop_plate.png"),
            dir.join("references/workshop_plate.png"),
        )
        .expect("plate copies");
        let shot = |id: &str, depends_on: Value| {
            json!({
                "id": id,
                "beat": format!("{id} beat"),
                "framing": "wide static",
                "prompt": format!("a quiet workshop, shot {id}"),
                "targetDurationSeconds": 5.1667,
                "startState": "before",
                "endState": "after",
                "audio": "Room tone, no music.",
                "conditioning": { "mode": "text_to_video" },
                // Every shot binds at least one approved role (sc-22713): the pack below approves
                // exactly one, and these shots are about the workshop.
                "continuityRoles": ["workshop_plate"],
                "dependsOn": depends_on,
            })
        };
        let plan = json!({
            "schemaVersion": sceneworks_core::film_plan::PLAN_SCHEMA_VERSION,
            "id": "budget-fixture",
            "version": 1,
            "title": "Budget fixture",
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": limits,
            "shots": [
                shot("SH010", json!([])),
                shot("SH020", json!([{ "shotId": "SH010", "kind": "continuity" }])),
            ],
        });
        let pack = json!({
            "schemaVersion": sceneworks_core::film_plan::REFERENCE_PACK_SCHEMA_VERSION,
            "id": "budget-refs",
            "version": 1,
            "references": [
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png" }
            ],
        });
        let plan_path = dir.join("plan.json");
        let pack_path = dir.join("references.json");
        std::fs::write(&plan_path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        std::fs::write(&pack_path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        (plan_path, pack_path)
    }

    pub(crate) async fn jobs(&self) -> Vec<Value> {
        let (_, jobs) = request(self.app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        jobs.as_array().cloned().unwrap_or_default()
    }

    pub(crate) fn fixture_plan(&self) -> PathBuf {
        Path::new(FIXTURE_DIR).join("plan.jsonc")
    }

    /// The shipped pack, verbatim, COPIED into this harness's temp dir with its plates and beds.
    ///
    /// A copy rather than the checked-in path because the pack directory is now written to
    /// (sc-23404): a run speaks its `dialogue` entries and leaves the WAVs beside the beds, so
    /// pointing the tests at `config/film-harness/courier-workshop` would have them write into the
    /// source tree and race each other on one filename under `cargo test`'s parallelism. The
    /// document is byte-for-byte the shipped one, so `reference_pack.sha256` is unchanged; only its
    /// directory moves. `checked_in_fixture_sound_matches_the_generator_byte_for_byte` reads the
    /// shipped path directly, which is where that guarantee belongs.
    pub(crate) fn fixture_pack(&self) -> PathBuf {
        let dir = self.temp_dir.path().join("fixture-pack");
        let path = dir.join("references.jsonc");
        if path.is_file() {
            return path;
        }
        for sub in ["references", "sound"] {
            std::fs::create_dir_all(dir.join(sub)).expect("pack dir");
            for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join(sub)).expect("fixture dir") {
                let entry = entry.expect("directory entry");
                std::fs::copy(entry.path(), dir.join(sub).join(entry.file_name()))
                    .expect("fixture file copies");
            }
        }
        std::fs::copy(Path::new(FIXTURE_DIR).join("references.jsonc"), &path)
            .expect("pack document copies");
        path
    }

    /// The shipped pack with its `sound` array emptied, copied into the temp dir so the relative
    /// `file` paths still resolve. For the lanes with no ffmpeg to transcode an audio upload with.
    pub(crate) fn fixture_pack_without_sound(&self) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
        pack["sound"] = json!([]);
        let dir = self.temp_dir.path().join("pack");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join("references"))
            .expect("fixture references dir")
        {
            let entry = entry.expect("directory entry");
            std::fs::copy(entry.path(), dir.join("references").join(entry.file_name()))
                .expect("plate copies");
        }
        let path = dir.join("references.json");
        std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        path
    }

    /// The shipped pack with every BINDABLE reference unapproved — only the `style` and the `plate`
    /// stay approved — copied into the temp dir so the relative `file` paths still resolve.
    ///
    /// This, and not a pack that approves literally nothing, is the reachable "the pack fills no
    /// reference shot" case, for two reasons found while covering this seam:
    ///
    ///   * the ANCHOR RULE (`film_plan.rs`) makes every shot name at least one APPROVED role in its
    ///     conditioning slots or `continuityRoles`, so on a pack that approves nothing EVERY shot of
    ///     EVERY plan is a finding and no plan can be produced at all; and
    ///   * `brief.requiredBeats[].requiredRoles` must each be approved by the pack, so the shipped
    ///     brief refuses an all-unapproved pack up front, before an envelope is ever built.
    ///
    /// Approving the style and the plate satisfies both — `house_style` is what every shot of the
    /// scripted drafts declares — while approving no SUBJECT a `reference_to_video` shot could bind
    /// (`BINDABLE_REFERENCE_KINDS`: character/prop/location). `sound` is left alone: a pack that
    /// approves no conditioning images still carries its beds, and the lines are what SH020/SH050/
    /// SH060 speak.
    pub(crate) fn fixture_pack_without_bindable_references(&self) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
        let entries = pack["references"]
            .as_array_mut()
            .expect("the pack declares references");
        assert!(!entries.is_empty(), "the shipped pack declares references");
        let mut approved_kinds = Vec::new();
        for entry in entries {
            let entry = entry.as_object_mut().expect("reference entry object");
            let bindable = sceneworks_core::film_plan::BINDABLE_REFERENCE_KINDS
                .contains(&entry["kind"].as_str().expect("every entry declares a kind"));
            // Explicit `false`: `approved` DEFAULTS to true when the key is absent, so removing the
            // key would approve the entry instead of unapproving it.
            entry.insert("approved".to_owned(), json!(!bindable));
            if !bindable {
                approved_kinds.push(entry["kind"].as_str().unwrap().to_owned());
            }
        }
        approved_kinds.sort();
        approved_kinds.dedup();
        assert_eq!(
            approved_kinds,
            vec!["plate".to_owned(), "style".to_owned()],
            "the shipped pack must still leave exactly a style and a plate approved, or this \
             fixture no longer anchors the shots it is used with"
        );
        let dir = self.temp_dir.path().join("pack-unbindable");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join("references"))
            .expect("fixture references dir")
        {
            let entry = entry.expect("directory entry");
            std::fs::copy(entry.path(), dir.join("references").join(entry.file_name()))
                .expect("plate copies");
        }
        std::fs::create_dir_all(dir.join("sound")).expect("sound dir");
        for entry in
            std::fs::read_dir(Path::new(FIXTURE_DIR).join("sound")).expect("fixture sound dir")
        {
            let entry = entry.expect("directory entry");
            std::fs::copy(entry.path(), dir.join("sound").join(entry.file_name()))
                .expect("bed copies");
        }
        let path = dir.join("references.json");
        std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        path
    }

    /// Copy the checked-in fixture into the temp dir with `edit` applied to the parsed plan, so a
    /// test can break one field without touching the shipped documents.
    ///
    /// The plan's SOUND is stripped first (sc-22712). Every caller is testing validation or a
    /// budget, and importing sound costs an ffmpeg transcode per clip — enough real time to spend a
    /// three-second run budget during setup, which is a test measuring the wrong thing. A test that
    /// wants a sound field back sets it inside `edit`.
    pub(crate) fn edited_plan(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_plan()).expect("fixture plan");
        let mut plan: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture plan parses");
        plan.as_object_mut().expect("plan object").remove("sound");
        for shot in plan["shots"].as_array_mut().expect("shots") {
            shot.as_object_mut()
                .expect("shot object")
                .remove("dialogueClip");
        }
        edit(&mut plan);
        let path = self.temp_dir.path().join("plan.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        path
    }

    /// The checked-in pack (and its plates) copied into the temp dir with `edit` applied, so a test
    /// can change an entry without touching the shipped documents.
    /// The checked-in pack with `edit` applied, copied into the temp dir.
    ///
    /// Its SOUND is emptied first, the mirror of `edited_plan` and for the same reason (sc-22712):
    /// every caller is testing something about references, and importing sound needs an ffmpeg
    /// that is not on every lane. Pair it with `edited_plan`, which drops the roles that would
    /// otherwise dangle.
    pub(crate) fn edited_pack(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
        pack["sound"] = json!([]);
        edit(&mut pack);
        let dir = self.temp_dir.path().join("pack");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        for (role, _) in FIXTURE_REFERENCES {
            let name = format!("references/{role}.png");
            std::fs::copy(Path::new(FIXTURE_DIR).join(&name), dir.join(&name))
                .expect("plate copies");
        }
        let path = dir.join("references.json");
        std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        path
    }

    /// The shipped MIXED-PARTITION fixture (sc-23402) copied into the temp dir with `edit`
    /// applied: SH010 binds `courier` + `workshop_location`, SH020 binds nothing. Its sound block is
    /// already absent, so it needs no ffmpeg.
    pub(crate) fn mixed_partition_plan(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(Path::new(FIXTURE_DIR).join("plan.ref.jsonc"))
            .expect("mixed fixture plan");
        let mut plan: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("mixed fixture plan parses");
        edit(&mut plan);
        let path = self.temp_dir.path().join("plan-ref.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        path
    }

    /// Make the shipped MiniMax-H3 turbo adapters INSTALLED on this harness (sc-23406).
    ///
    /// The route refuses an uninstalled LoRA at enqueue, so a test that dispatches one has to make
    /// it installed the way the API decides that question: a catalog entry whose `source.path`
    /// points at a directory holding a readable `.safetensors`. The manifest written here is the
    /// SHIPPED `builtin.loras.jsonc` with only that one key rewritten, so the ids, families,
    /// `modelIds` allowlists and `sampling` recipes under test are the real ones.
    ///
    /// The header carries one inert tensor key on purpose: it must parse (the route reads it) and
    /// must match no family detector (a detected family would be judged against the model's, which
    /// is a different rule from the one this test is about).
    pub(crate) fn install_turbo_loras(&self) {
        let weights_dir = self.temp_dir.path().join("lora-weights");
        std::fs::create_dir_all(&weights_dir).expect("weights dir creates");
        let header = serde_json::json!({
            "inert.weight": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }
        });
        let header_bytes = serde_json::to_vec(&header).expect("header json");
        let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&[0_u8; 4]);
        let file = weights_dir.join("adapter.safetensors");
        std::fs::write(&file, bytes).expect("adapter writes");

        let shipped = include_str!("../../../../config/manifests/builtin.loras.jsonc");
        let mut manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(shipped))
                .expect("the shipped lora manifest parses");
        for lora in manifest["loras"].as_array_mut().expect("loras") {
            let id = lora["id"].as_str().unwrap_or_default().to_owned();
            if id.starts_with("minimax_h3") {
                lora["source"] = serde_json::json!({
                    "path": weights_dir.display().to_string()
                });
            }
        }
        let manifests_dir = self.temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&manifests_dir).expect("manifest dir creates");
        std::fs::write(
            manifests_dir.join("builtin.loras.jsonc"),
            serde_json::to_string_pretty(&manifest).expect("manifest serializes"),
        )
        .expect("lora manifest writes");
    }

    pub(crate) fn run_record(&self) -> Value {
        let text = std::fs::read_to_string(self.temp_dir.path().join("run-out/run.json"))
            .expect("run.json written");
        serde_json::from_str(&text).expect("run.json parses")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.get_mut().take() {
            worker.abort();
        }
        if let Some(server) = self.server.take() {
            server.abort();
        }
    }
}

/// One line per attempt — the shot, status and error of every attempt plus the export — for
/// assertion messages, so a failing run explains itself without the full Debug dump.
pub(crate) fn summary(record: &sceneworks_core::film_plan::RunRecord) -> String {
    let mut lines = vec![format!("outcome={:?}", record.outcome)];
    for shot in &record.shots {
        lines.push(format!("{} {:?}", shot.shot_id, shot.outcome));
        for attempt in &shot.attempts {
            lines.push(format!(
                "  attempt {} job={} status={} error={:?} take={:?}",
                attempt.attempt,
                attempt.job_id.as_deref().unwrap_or("-"),
                attempt.status,
                attempt.error,
                attempt.take.as_ref().map(|take| take.asset_id.as_str())
            ));
        }
    }
    if let Some(export) = &record.export {
        lines.push(format!(
            "export job={} status={} error={:?}",
            export.job_id, export.status, export.error
        ));
    }
    lines.join("\n")
}

#[tokio::test]
async fn two_shot_run_renders_imports_assembles_and_exports_through_the_real_routes() {
    let harness = Harness::start(true, vec![]).await;
    // Sound import transcodes through ffmpeg (`ProjectStore::import_asset`), which is not on every
    // lane — the same posture as `import_asset_admits_audio_and_normalizes_it_to_pcm16_wav`. Where
    // there is no ffmpeg this test runs against the pack's pictures alone, so everything sc-22710
    // established still runs everywhere; the sound assertions below are gated on the same check and
    // `the_assembled_sequence_carries_three_independently_controlled_sound_buses` owns them in full.
    let sound = ffmpeg_reachable();
    let (plan, pack) = if sound {
        (harness.fixture_plan(), harness.fixture_pack())
    } else {
        (
            harness.edited_plan(|_| {}),
            harness.fixture_pack_without_sound(),
        )
    };
    let options = harness.options(plan, pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");

    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.selected_shot_ids, vec!["SH010", "SH020"]);
    let project_id = record.project_id.clone().expect("project created");

    // Every approved reference is a project asset, tagged with its role, addressable on its own.
    assert_eq!(record.references.len(), 7);
    for reference in &record.references {
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!(
                "/api/v1/projects/{project_id}/assets/{}",
                reference.asset_id
            ),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "image");
        assert_eq!(asset["extra"]["filmHarness"]["role"], reference.role);
        assert_eq!(
            asset["extra"]["filmHarness"]["referencePackId"],
            "courier-workshop-refs"
        );
        let tags: Vec<&str> = asset["tags"]
            .as_array()
            .map(|tags| tags.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            tags.contains(&format!("role:{}", reference.role).as_str()),
            "{tags:?}"
        );
    }

    // Two real jobs went through POST /api/v1/video/jobs and produced two persisted video assets.
    let rendered: Vec<_> = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::Rendered)
        .collect();
    assert_eq!(rendered.len(), 2, "{:#?}", record.shots);
    let not_selected = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::NotSelected)
        .count();
    assert_eq!(
        not_selected, 4,
        "the other four shots stay in the record, unrendered"
    );
    let claimed = harness.script.lock().claimed.clone();
    let video_jobs: Vec<_> = claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .collect();
    assert_eq!(video_jobs.len(), 2);
    for shot in &rendered {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let take = attempt.take.as_ref().expect("a take");
        let (status, job) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/jobs/{job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(job["status"], "completed");
        assert_eq!(job["payload"]["model"], "minimax_h3");
        assert_eq!(job["payload"]["advanced"]["mlxQuantize"], 4);
        assert_eq!(
            job["payload"]["advanced"]["filmHarness"]["shotId"],
            shot.shot_id
        );
        assert_eq!(
            job["payload"]["advanced"]["filmHarness"]["runId"],
            record.run_id
        );
        assert_eq!(job["payload"]["fps"], 24);
        assert_eq!(job["payload"]["width"], 576);
        assert_eq!(job["payload"]["height"], 320);
        assert_eq!(job["result"]["assetIds"][0], take.asset_id);
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets/{}", take.asset_id),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "video");
        assert_eq!(
            asset["lineage"]["jobId"], job_id,
            "asset links back to its job"
        );
        assert_eq!(take.backend.as_deref(), Some("mlx"));
        assert_eq!(take.raw_adapter_settings["tier"], "q4");
        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(job["payload"]["mode"], "text_to_video");
                assert!(job["payload"]["sourceAssetId"].is_null());
            }
            "SH020" => {
                assert_eq!(job["payload"]["mode"], "image_to_video");
                let plate = record
                    .references
                    .iter()
                    .find(|reference| reference.role == "workshop_plate")
                    .expect("plate imported");
                assert_eq!(job["payload"]["sourceAssetId"], plate.asset_id);
                assert_eq!(
                    shot.conditioning_assets.first_frame_asset_id.as_deref(),
                    Some(plate.asset_id.as_str())
                );
            }
            other => panic!("unexpected rendered shot {other}"),
        }
    }
    let model = record.model.as_ref().expect("model record");
    assert_eq!(model.id, "minimax_h3");
    assert_eq!(model.tier_requested.as_deref(), Some("q4"));
    assert_eq!(model.backend_observed.as_deref(), Some("mlx"));
    assert_eq!(model.hardware.worker_id.as_deref(), Some(WORKER_ID));
    assert_eq!(model.hardware.host_memory_gb, Some(128.0));
    assert_eq!(
        model.weights.as_ref().and_then(|w| w["repo"].as_str()),
        Some("SceneWorks/minimax-h3-mlx")
    );

    // The timeline holds the two takes back to back and the export ran through timeline_export.
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    assert_eq!(timeline.items.len(), 2);
    assert_eq!(timeline.fps, 24);
    assert_eq!(timeline.aspect_ratio, "16:9");
    assert_eq!(timeline.items[0].shot_id.as_deref(), Some("SH010"));
    assert_eq!(timeline.items[1].shot_id.as_deref(), Some("SH020"));
    assert!((timeline.items[0].timeline_end - 5.1667).abs() < 1e-6);
    assert!((timeline.items[1].timeline_start - 5.1667).abs() < 1e-6);
    let (status, saved) = request(
        harness.app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/timelines/{}",
            timeline.timeline_id
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{saved}");
    let items = saved["tracks"][0]["items"]
        .as_array()
        .expect("main track items");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["assetId"], timeline.items[0].asset_id);
    assert_eq!(
        items[1]["versionHistory"][0]["jobId"],
        rendered[1].attempts[0].job_id.clone().unwrap()
    );
    if sound {
        // The clips this two-shot selection places — both beds and SH020's line, but not the
        // recipient's, whose shots are not in the selection. The detail lives in the dedicated
        // test; this is the shipped fixture's smoke.
        assert_eq!(record.sound.len(), 3, "{:#?}", record.sound);
        let roles: Vec<&str> = timeline
            .tracks
            .iter()
            .filter(|track| track.kind == "audio")
            .map(|track| track.role.as_str())
            .collect();
        assert_eq!(roles, vec!["dialogue", "ambience", "music"], "{roles:?}");
    }
    let export = record.export.as_ref().expect("export ran");
    assert_eq!(export.status, "completed");
    assert!(export.asset_id.is_some());
    assert!(export
        .render_path
        .as_deref()
        .unwrap_or("")
        .starts_with("assets/renders/"));
    let export_jobs: Vec<_> = claimed
        .iter()
        .filter(|(kind, _, _)| kind == "timeline_export")
        .collect();
    assert_eq!(export_jobs.len(), 1);
    assert_eq!(export_jobs[0].2["timelineId"], timeline.timeline_id);
    assert_eq!(export_jobs[0].2["resolution"], 640);
    assert_eq!(export_jobs[0].2["fps"], 24);

    // The run record is on disk beside copies of both documents, and inside the project.
    let on_disk = harness.run_record();
    assert_eq!(on_disk["runId"], record.run_id);
    assert_eq!(on_disk["outcome"], "completed");
    assert_eq!(on_disk["plan"]["id"], "courier-workshop");
    assert_eq!(on_disk["referencePack"]["id"], "courier-workshop-refs");
    assert!(harness.temp_dir.path().join("run-out/plan.json").exists());
    assert!(harness
        .temp_dir
        .path()
        .join("run-out/references.json")
        .exists());
    let project_record = PathBuf::from(record.project_path.as_deref().unwrap())
        .join("film-harness")
        .join(&record.run_id)
        .join("run.json");
    assert!(project_record.exists(), "{}", project_record.display());
}

#[tokio::test]
async fn malformed_plan_is_refused_before_any_job_exists() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        plan["shots"][1]["id"] = json!("SH010");
        plan["shots"][2]["prompt"] = json!("");
        plan["shots"][3]["conditioning"] =
            json!({ "mode": "image_to_video", "referenceRoles": ["courier"] });
        plan["limits"]["maxAttemptsPerShot"] = json!(0);
    });
    let options = harness.options(plan, harness.fixture_pack(), None);
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("malformed plan is refused");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("duplicate shot id")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.starts_with("[SH030] prompt:")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.starts_with("[SH040] conditioning.firstFrameRole:")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.starts_with("[SH040] conditioning.referenceRoles:")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("limits.maxAttemptsPerShot")),
        "{text:?}"
    );
    assert!(
        harness.jobs().await.is_empty(),
        "no job may exist after a refusal"
    );
    let (_, projects) = request(harness.app.clone(), "GET", "/api/v1/projects", Value::Null).await;
    assert!(
        projects.as_array().unwrap().is_empty(),
        "no project is created either"
    );
    let record = harness.run_record();
    assert_eq!(record["outcome"], "rejected");
    assert_eq!(
        record["diagnostics"].as_array().unwrap().len(),
        findings.len()
    );
}

#[tokio::test]
async fn missing_reference_files_and_dangling_roles_are_refused_before_dispatch() {
    let harness = Harness::start(true, vec![]).await;
    // A pack in the temp dir whose files are absent, plus a role the plan never defines.
    let pack_dir = harness.temp_dir.path().join("pack");
    std::fs::create_dir_all(pack_dir.join("references")).unwrap();
    let pack_text = std::fs::read_to_string(harness.fixture_pack()).unwrap();
    let mut pack: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&pack_text)).unwrap();
    pack["references"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["role"] != "workshop_plate");
    let pack_path = pack_dir.join("references.json");
    std::fs::write(&pack_path, pack.to_string()).unwrap();
    // Only some of the remaining files exist.
    for role in ["courier", "recipient"] {
        std::fs::write(pack_dir.join(format!("references/{role}.png")), b"png").unwrap();
    }
    let options = harness.options(harness.fixture_plan(), pack_path, Some(&["SH010", "SH020"]));
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("dangling roles and missing files are refused");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] conditioning.firstFrameRole")
                && m.contains("\"workshop_plate\" is not in reference pack")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("\"red_parcel\"") && m.contains("is missing")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());
}

#[tokio::test]
async fn unsupported_conditioning_and_off_menu_timing_are_refused_against_the_catalog() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        // A negative prompt the model has no axis for. (A shot that binds reference roles is NOT a
        // refusal any more: it resolves to `minimax_h3_ref`, which declares nine — sc-23402, and
        // `a_mixed_plan_dispatches_each_shot_on_its_own_partition` owns that case.)
        plan["shots"][0]["negativePrompt"] = json!("blurry, low quality");
        // Off the fourteen-length menu, and an undeclared canvas.
        plan["shots"][1]["targetDurationSeconds"] = json!(6.0);
        plan["shots"][1]["resolution"] = json!("640x360");
        // Below the model's declared mlx minimum.
        plan["limits"]["maxMemoryGb"] = json!(32);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("catalog-level refusal");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("[SH010] negativePrompt")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] targetDurationSeconds") && m.contains("menu")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] resolution") && m.contains("640x360")),
        "{text:?}"
    );
    assert!(
        // The lane (and so the minimum: mlx 64 GB, candle 43 GB) follows the host the test runs
        // on; the finding's shape is what is pinned here.
        text.iter()
            .any(|m| m.contains("limits.maxMemoryGb") && m.contains("minMemoryGb of")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());
    assert_eq!(harness.run_record()["outcome"], "rejected");
}

/// sc-23402 AC1/AC2. One plan, two checkpoints of one family: the shot that binds reference roles
/// dispatches as `minimax_h3_ref` / `reference_to_video`, the shot that binds none as `minimax_h3`
/// / `text_to_video` with no reference field at all — through the REAL
/// `POST /api/v1/video/jobs` route, with the shipped catalog deciding what each partition declares.
#[tokio::test]
async fn a_mixed_plan_dispatches_each_shot_on_its_own_partition() {
    let harness = Harness::start(true, vec![]).await;
    let plan_path = harness.mixed_partition_plan(|_| {});
    let pack_path = harness.fixture_pack_without_sound();

    // `film-harness compile --no-refine` writes the document a reviewer reads and the run
    // dispatches. Doing it here rather than compiling in memory is the point: `compiled.json` is
    // where per-shot partition resolution has to be visible.
    let mut compile_options = planner_options(&harness, "compile-mixed");
    compile_options.reference_pack_path = pack_path.clone();
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &compile_options,
        &plan_path,
    )
    .await
    .expect("the mixed plan compiles");
    let compiled: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.compiled_path).expect("compiled.json written"),
    )
    .expect("compiled.json parses");

    let mut options = harness.options(plan_path, pack_path, None);
    options.compiled_path = Some(artifacts.compiled_path.clone());
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(
        compiled["model"]["id"], "minimax_h3",
        "the plan declares the family once"
    );
    let request = |shot_id: &str| -> Value {
        compiled["requests"]
            .as_array()
            .expect("requests")
            .iter()
            .find(|request| request["shotId"] == shot_id)
            .cloned()
            .unwrap_or_else(|| panic!("no compiled request for {shot_id}"))
    };
    let referenced = request("SH010");
    assert_eq!(referenced["model"], "minimax_h3_ref");
    assert_eq!(referenced["mode"], "reference_to_video");
    assert_eq!(
        referenced["referenceRoles"],
        json!(["courier", "workshop_location"]),
        "role ORDER is the plan's"
    );
    assert!(referenced["partitionReason"]
        .as_str()
        .is_some_and(|reason| reason.contains("minimax_h3_ref")));
    let plain = request("SH020");
    assert_eq!(plain["model"], "minimax_h3");
    assert_eq!(plain["mode"], "text_to_video");
    assert_eq!(plain["referenceRoles"], json!([]));

    // What the route received, and what the record says about it.
    let location = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_location")
        .expect("location imported");
    let courier = record
        .references
        .iter()
        .find(|reference| reference.role == "courier")
        .expect("courier imported");
    // sc-24026, on the DISPATCH the run actually made: the plan the run read, so the expected tail
    // is derived from the same document the compile was driven from rather than restated here.
    let dispatched_plan = sceneworks_core::film_plan::read_plan_file(&options.plan_path)
        .expect("the run's plan re-reads");

    for shot in &record.shots {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let (status, job) = request_job(&harness, &job_id).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{job}");

        // EVERY shot, on BOTH partitions: the body that came off the route ends with this shot's
        // own `Audio:` sentence. Asserted on the job payload rather than on `compiled.json`
        // because the document is only a promise — this is the text the engine was handed.
        let planned = dispatched_plan
            .shots
            .iter()
            .find(|planned| planned.id == shot.shot_id)
            .expect("every dispatched shot is a shot of the plan");
        let prompt = job["payload"]["prompt"]
            .as_str()
            .unwrap_or_else(|| panic!("a dispatched prompt: {}", job["payload"]));
        let audio = format!(
            "Audio: {}",
            sceneworks_core::film_compile::normalized_description(&planned.audio)
        );
        let tail = match planned.dialogue_clip {
            Some(_) => format!(
                "{audio} {}",
                sceneworks_core::film_compile::NO_SPEECH_SENTENCE
            ),
            None => audio,
        };
        assert!(
            prompt.ends_with(&tail),
            "{}: the dispatched prompt must end with {tail:?}: {prompt:?}",
            shot.shot_id
        );

        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(job["payload"]["model"], "minimax_h3_ref");
                assert_eq!(job["payload"]["mode"], "reference_to_video");
                assert_eq!(
                    job["payload"]["referenceAssetIds"],
                    json!([courier.asset_id, location.asset_id]),
                    "the reference assets ride the payload in role order"
                );
                // sc-24023, on the DISPATCH the run actually made: `ensure_shot_records` resolved
                // the conditioning, `work_attempt` posted `to_job_body_with`, and this is the body
                // that came back off the route. The prompt in it must bind each role to the
                // `<Picture N>` whose N is that role's 1-based position in the SAME payload's
                // `referenceAssetIds` — the engine labels the supplied images positionally, so a
                // sentence naming the wrong number renders a confidently wrong shot and no
                // validator, record or reviewer downstream can tell.
                let prompt = job["payload"]["prompt"]
                    .as_str()
                    .unwrap_or_else(|| panic!("a dispatched prompt: {}", job["payload"]));
                let dispatched = job["payload"]["referenceAssetIds"]
                    .as_array()
                    .unwrap_or_else(|| panic!("dispatched assets: {}", job["payload"]));
                let mut cursor = 0usize;
                for (index, (role, asset_id)) in [
                    ("courier", &courier.asset_id),
                    ("workshop_location", &location.asset_id),
                ]
                .iter()
                .enumerate()
                {
                    let number = index + 1;
                    assert_eq!(
                        dispatched[index],
                        json!(asset_id),
                        "{role} is dispatched at position {number} of {dispatched:?}"
                    );
                    // Anchored on the whole sentence opening and scanned forward, so a phrase
                    // occurring inside a pack description cannot stand in for the binding itself
                    // and the sentences must also come out in picture order.
                    let phrase = format!("{} is the ", role.replace(['_', '-'], " "));
                    let role_at = prompt[cursor..]
                        .find(&phrase)
                        .map(|at| at + cursor)
                        .unwrap_or_else(|| panic!("{phrase:?} is never said in {prompt:?}"));
                    cursor = prompt[role_at..]
                        .find(&format!("<Picture {number}>"))
                        .map(|at| at + role_at)
                        .unwrap_or_else(|| {
                            panic!(
                                "{phrase:?} must be bound to <Picture {number}>, the position \
                                 {role}'s asset takes in {dispatched:?}: {prompt:?}"
                            )
                        });
                }
                assert!(
                    !prompt.contains("<Picture 3>"),
                    "the dispatched prompt names a picture this shot never sends: {prompt:?}"
                );
                assert_eq!(attempt.resolved_model_id, "minimax_h3_ref");
                // sc-23402 short edge: this plan names none, so nothing is dispatched and the
                // record keeps the EFFECTIVE value the engine rendered at — its own 2048.
                assert!(
                    job["payload"]["advanced"]
                        .get("referenceImageShortEdge")
                        .is_none(),
                    "{}",
                    job["payload"]["advanced"]
                );
                assert_eq!(attempt.reference_image_short_edge, Some(2048));
                assert_eq!(
                    shot.conditioning_assets.reference_asset_ids,
                    vec![courier.asset_id.clone(), location.asset_id.clone()]
                );
                assert_eq!(
                    job["payload"]["modelManifestEntry"]["id"], "minimax_h3_ref",
                    "the route resolved the reference partition's entry for this shot"
                );
                assert_eq!(
                    attempt.take.as_ref().expect("take").model,
                    "minimax_h3_ref",
                    "the take names the checkpoint that rendered it"
                );
            }
            "SH020" => {
                assert_eq!(job["payload"]["model"], "minimax_h3");
                assert_eq!(job["payload"]["mode"], "text_to_video");
                // The route normalises an absent list to `[]`; the SENT body carries no
                // `referenceAssetIds` key at all (asserted on the compiled request in
                // `film_compile`'s own tests).
                assert_eq!(job["payload"]["referenceAssetIds"], json!([]));
                assert_eq!(
                    job["payload"]["modelManifestEntry"]["id"], "minimax_h3",
                    "the route resolved the base entry for this shot"
                );
                assert_eq!(attempt.resolved_model_id, "minimax_h3");
                assert_eq!(attempt.take.as_ref().expect("take").model, "minimax_h3");
                assert_eq!(
                    attempt.reference_image_short_edge, None,
                    "a base-partition attempt encodes no reference, so it records no short edge"
                );
            }
            other => panic!("unexpected shot {other}"),
        }
        assert!(
            !attempt.partition_reason.is_empty(),
            "{} has no partition reason",
            shot.shot_id
        );
        assert_eq!(
            job["payload"]["advanced"]["filmHarness"]["partitionReason"],
            json!(attempt.partition_reason),
            "the payload and the record carry the same reason"
        );
    }

    // And on disk, where a later reader finds it.
    let on_disk = harness.run_record();
    let attempt = |shot_id: &str| -> Value {
        on_disk["shots"]
            .as_array()
            .expect("shots")
            .iter()
            .find(|shot| shot["shotId"] == shot_id)
            .and_then(|shot| shot["attempts"][0].as_object())
            .map(|attempt| Value::Object(attempt.clone()))
            .unwrap_or_else(|| panic!("no recorded attempt for {shot_id}"))
    };
    assert_eq!(attempt("SH010")["resolvedModelId"], "minimax_h3_ref");
    assert_eq!(attempt("SH020")["resolvedModelId"], "minimax_h3");
    assert!(attempt("SH010")["partitionReason"]
        .as_str()
        .is_some_and(|reason| reason.contains("minimax_h3_ref")));

    // sc-23402 review: the record names the WEIGHTS behind each partition it dispatched on, not
    // only the declared model's row — a mixed run loads a second 18.78 GB `transformer_ref`
    // download and nothing recorded which files produced the reference take.
    let weights = &on_disk["model"]["partitionWeights"];
    let partitions: Vec<&str> = weights
        .as_object()
        .expect("partitionWeights")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        partitions,
        vec!["minimax_h3", "minimax_h3_ref"],
        "both partitions this run loaded: {weights}"
    );
    let files = |partition: &str| -> String { weights[partition]["files"].to_string() };
    assert!(files("minimax_h3").contains("q4/transformer/"), "{weights}");
    assert!(
        files("minimax_h3_ref").contains("q4/transformer_ref/"),
        "the reference partition's own rows, not a copy of the base's: {weights}"
    );
    assert_eq!(
        on_disk["model"]["weights"], weights["minimax_h3"],
        "`weights` stays the DECLARED model's row"
    );
}

/// sc-23402 short edge. A plan that lowers `model.advanced.referenceImageShortEdge` sends it on the
/// REFERENCE shot's `POST /api/v1/video/jobs` body and records the same number on that attempt —
/// while the base-partition shot in the same plan dispatches and records nothing, because it encodes
/// no reference for the knob to size. Through the real route, with the real record on disk.
#[tokio::test]
async fn a_lowered_reference_short_edge_rides_the_reference_shots_payload_and_record() {
    let harness = Harness::start(true, vec![]).await;
    let plan_path = harness.mixed_partition_plan(|plan| {
        plan["model"]["advanced"] = json!({ "referenceImageShortEdge": 1536 });
    });
    let pack_path = harness.fixture_pack_without_sound();
    let mut options = harness.options(plan_path, pack_path, None);
    options.out_dir = harness.temp_dir.path().join("run-out-short-edge");
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );

    for shot in &record.shots {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let (status, job) = request_job(&harness, &job_id).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{job}");
        let advanced = &job["payload"]["advanced"];
        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(job["payload"]["model"], "minimax_h3_ref");
                assert_eq!(
                    advanced["referenceImageShortEdge"],
                    json!(1536),
                    "the route persisted the knob in the job payload: {advanced}"
                );
                assert_eq!(attempt.reference_image_short_edge, Some(1536));
            }
            "SH020" => {
                assert_eq!(job["payload"]["model"], "minimax_h3");
                assert!(
                    advanced.get("referenceImageShortEdge").is_none(),
                    "the base partition has no reference to size: {advanced}"
                );
                assert_eq!(attempt.reference_image_short_edge, None);
            }
            other => panic!("unexpected shot {other}"),
        }
    }

    // And on disk, where a later reader — a comparison against a 2048 run — finds it.
    let on_disk: Value = serde_json::from_str(
        &std::fs::read_to_string(options.out_dir.join("run.json")).expect("run.json written"),
    )
    .expect("run.json parses");
    let attempt = |shot_id: &str| -> Value {
        on_disk["shots"]
            .as_array()
            .expect("shots")
            .iter()
            .find(|shot| shot["shotId"] == shot_id)
            .map(|shot| shot["attempts"][0].clone())
            .unwrap_or_else(|| panic!("no recorded attempt for {shot_id}"))
    };
    assert_eq!(attempt("SH010")["referenceImageShortEdge"], json!(1536));
    assert!(
        attempt("SH020").get("referenceImageShortEdge").is_none(),
        "{}",
        attempt("SH020")
    );
}

/// 🔴 sc-23406. The plan declares its accelerators ONCE on the family; each shot's
/// `POST /api/v1/video/jobs` body carries only the ones its RESOLVED partition was distilled for,
/// plus the plan's `advanced.steps` — and the attempt record on disk agrees with the payload.
///
/// Through the real route with the fake worker, because the route is where this could silently go
/// wrong in three different ways: the LoRA compatibility gate could refuse the pairing, the
/// declared-partition gate could refuse the ref2v adapter on the base checkpoint (it should — and
/// the harness must therefore never send it there), and the payload normalisation could drop the
/// entry shape. A core-only test proves none of those.
#[tokio::test]
async fn the_plans_turbo_loras_ride_each_shots_payload_for_its_own_partition() {
    let harness = Harness::start(true, vec![]).await;
    harness.install_turbo_loras();
    let plan_path = harness.mixed_partition_plan(|plan| {
        plan["model"]["loras"] =
            json!(["minimax_h3_ref2v_turbo_4step", "minimax_h3_turbo_4step_v01"]);
        plan["model"]["advanced"] = json!({ "steps": 6 });
    });
    let pack_path = harness.fixture_pack_without_sound();
    let mut options = harness.options(plan_path, pack_path, None);
    options.out_dir = harness.temp_dir.path().join("run-out-turbo");
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );

    for shot in &record.shots {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let (status, job) = request_job(&harness, &job_id).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{job}");
        let payload = &job["payload"];
        let sent: Vec<&str> = payload["loras"]
            .as_array()
            .map(|loras| {
                loras
                    .iter()
                    .filter_map(|lora| lora["id"].as_str())
                    .collect()
            })
            .unwrap_or_default();
        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(payload["model"], "minimax_h3_ref");
                assert_eq!(
                    sent,
                    vec!["minimax_h3_ref2v_turbo_4step"],
                    "the reference partition takes the ref2v adapter and ONLY that one: {}",
                    payload["loras"]
                );
                assert_eq!(attempt.loras, vec!["minimax_h3_ref2v_turbo_4step"]);
            }
            "SH020" => {
                assert_eq!(payload["model"], "minimax_h3");
                assert_eq!(
                    sent,
                    vec!["minimax_h3_turbo_4step_v01"],
                    "the base partition takes the fl2v adapter — sending the ref2v one here is \
                     what the route's declared-partition gate refuses: {}",
                    payload["loras"]
                );
                assert_eq!(attempt.loras, vec!["minimax_h3_turbo_4step_v01"]);
            }
            other => panic!("unexpected shot {other}"),
        }
        // The route hydrates each entry from the catalog, so the SENT weight survives as the
        // catalog's declared one rather than being dropped.
        assert_eq!(payload["loras"][0]["weight"], json!(1.0), "{payload}");
        // The plan's override rides `advanced.steps` on both partitions and is what the record
        // says ran — over the recipe's own 4.
        assert_eq!(payload["advanced"]["steps"], json!(6), "{payload}");
        assert_eq!(attempt.effective_steps, Some(6), "{}", shot.shot_id);
        assert_eq!(
            attempt.turbo_scheduler_shift,
            Some(12.0),
            "{}: a recipe applied, so its trained video shift is recorded",
            shot.shot_id
        );
    }

    // And on disk, where a later reader comparing this run against a 50-step one finds it.
    let on_disk: Value = serde_json::from_str(
        &std::fs::read_to_string(options.out_dir.join("run.json")).expect("run.json written"),
    )
    .expect("run.json parses");
    let attempt = |shot_id: &str| -> Value {
        on_disk["shots"]
            .as_array()
            .expect("shots")
            .iter()
            .find(|shot| shot["shotId"] == shot_id)
            .map(|shot| shot["attempts"][0].clone())
            .unwrap_or_else(|| panic!("no recorded attempt for {shot_id}"))
    };
    assert_eq!(
        attempt("SH010")["loras"],
        json!(["minimax_h3_ref2v_turbo_4step"])
    );
    assert_eq!(
        attempt("SH020")["loras"],
        json!(["minimax_h3_turbo_4step_v01"])
    );
    assert_eq!(attempt("SH010")["effectiveSteps"], json!(6));
    assert_eq!(attempt("SH010")["turboSchedulerShift"], json!(12.0));
}

/// The same plan with NO override: each shot records the step count its own partition's recipe
/// declares, and a shot whose partition no accelerator reached records the model's own default.
///
/// The three-way distinction is the point — `advanced.steps`, the recipe, the model default are
/// three different sources and a record that could not tell them apart would make a turbo run and
/// a base run indistinguishable after the fact.
#[tokio::test]
async fn a_shot_whose_partition_no_accelerator_reaches_records_the_models_own_step_count() {
    let harness = Harness::start(true, vec![]).await;
    harness.install_turbo_loras();
    // Only the REFERENCE partition's adapter is declared, so SH020 (base) gets none.
    let plan_path = harness.mixed_partition_plan(|plan| {
        plan["model"]["loras"] = json!(["minimax_h3_ref2v_turbo_4step"]);
    });
    let pack_path = harness.fixture_pack_without_sound();
    let mut options = harness.options(plan_path, pack_path, None);
    options.out_dir = harness.temp_dir.path().join("run-out-turbo-partial");
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );

    for shot in &record.shots {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let (_, job) = request_job(&harness, &job_id).await;
        let payload = &job["payload"];
        assert!(
            payload["advanced"].get("steps").is_none(),
            "no plan override ⇒ nothing dispatched; the recipe or the engine default governs: {}",
            payload["advanced"]
        );
        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(attempt.loras, vec!["minimax_h3_ref2v_turbo_4step"]);
                assert_eq!(attempt.effective_steps, Some(4), "the recipe's own count");
                assert_eq!(attempt.turbo_scheduler_shift, Some(12.0));
            }
            "SH020" => {
                assert!(
                    payload
                        .get("loras")
                        .is_none_or(|loras| loras.as_array().is_some_and(|loras| loras.is_empty())),
                    "the ref2v adapter must NOT reach the base checkpoint: {}",
                    payload["loras"]
                );
                assert!(attempt.loras.is_empty());
                assert_eq!(
                    attempt.effective_steps,
                    Some(50),
                    "no recipe applied, so the model's declared default is what ran"
                );
                assert_eq!(attempt.turbo_scheduler_shift, None);
            }
            other => panic!("unexpected shot {other}"),
        }
    }
}

/// sc-23402 AC1, the refusals: too many roles for the RESOLVED partition, and a
/// `reference_to_video` shot binding none.
#[tokio::test]
async fn reference_counts_are_refused_against_the_resolved_partitions_limits() {
    let harness = Harness::start(true, vec![]).await;
    // Ten roles against `minimax_h3_ref`'s declared nine. The pack approves ten, and every one of
    // them is a BINDABLE kind, so the count is the only thing wrong with the plan — a `plate` here
    // would be refused on its kind instead and the count would never be reached.
    let roles: Vec<String> = (0..10).map(|index| format!("extra_prop_{index}")).collect();
    // Each on its OWN plate: `maxReferenceAssets` bounds the IMAGES a request supplies, so ten
    // roles over one file would be one image and inside the cap (sc-24024). The files are copies
    // of the shipped plate under ten names, so `validate_reference_pack_files` can stat them.
    let pack = harness.edited_pack(|pack| {
        let references = pack["references"].as_array_mut().expect("references");
        for role in 0..10 {
            references.push(json!({
                "role": format!("extra_prop_{role}"),
                "kind": "prop",
                "file": format!("references/extra_prop_{role}.png")
            }));
        }
    });
    let plate_dir = pack.parent().unwrap().join("references");
    for role in 0..10 {
        std::fs::copy(
            plate_dir.join("workshop_plate.png"),
            plate_dir.join(format!("extra_prop_{role}.png")),
        )
        .expect("the extra plates copy");
    }
    let plan = harness.mixed_partition_plan(|plan| {
        plan["shots"][0]["conditioning"]["referenceRoles"] = json!(roles);
    });
    let mut options = harness.options(plan, pack.clone(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-too-many-refs");
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("over the cap");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter()
            .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                && m.contains("minimax_h3_ref")
                && m.contains("maxReferenceAssets")
                && m.contains('9')),
        "{text:?}"
    );

    // A reference_to_video shot binding nothing: a contradiction, named with the shot.
    let plan = harness.mixed_partition_plan(|plan| {
        plan["shots"][0]["conditioning"] = json!({ "mode": "reference_to_video" });
    });
    let mut options = harness.options(plan, pack, None);
    options.out_dir = harness.temp_dir.path().join("run-out-no-refs");
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("no references on a reference shot");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter()
            .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                && m.contains("at least one reference role")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());
}

/// A catalog rewrite that reports `minimax_h3` INSTALLED and leaves `minimax_h3_ref` exactly as
/// the host serves it — missing, since no weights are on disk under the test's data dir.
///
/// That split is the whole subject of the install-gate tests below: the base checkpoint downloaded,
/// the separate 18.78 GB reference DiT not. Without the rewrite both partitions read `missing` and
/// the base's own refusal would mask whatever the reference partition's gate did.
fn only_the_base_partition_is_installed(body: &mut Value) {
    let Some(entries) = body.as_array_mut() else {
        return;
    };
    for entry in entries {
        if entry.get("id").and_then(Value::as_str) != Some("minimax_h3") {
            continue;
        }
        entry["installState"] = json!("installed");
        if let Some(variants) = entry.get_mut("variants").and_then(Value::as_array_mut) {
            for variant in variants {
                variant["installed"] = json!(true);
                variant["installState"] = json!("installed");
            }
        }
    }
}

/// The same rewrite, plus the family's REFERENCE partition removed from the catalog entirely
/// (sc-23405): the host that serves `minimax_h3` and has no `minimax_h3_ref` row at all.
///
/// That is the catalog fact the planner's envelope narrows on — a mode whose partition the API does
/// not serve could only ever be refused per shot — and it is the shape a `plan` on a
/// reference-less catalog has to keep working through.
fn only_the_base_partition_exists(body: &mut Value) {
    only_the_base_partition_is_installed(body);
    if let Some(entries) = body.as_array_mut() {
        entries.retain(|entry| entry.get("id").and_then(Value::as_str) != Some("minimax_h3_ref"));
    }
}

/// sc-23402 review. The install/reachability gate runs on the RESOLVED reference partition too, and
/// names it: `plan.ref.jsonc`'s SH010 needs `minimax_h3_ref`, which is a second download with its
/// own install state. Shaped like the base install-gate assertion in
/// `a_plan_is_refused_when_the_host_cannot_run_it`, but driven from the mixed fixture.
#[tokio::test]
async fn the_reference_partitions_install_state_is_gated_and_named() {
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
        only_the_base_partition_is_installed(body);
    });
    let plan = harness.mixed_partition_plan(|_| {});
    let mut options = harness.options(plan, harness.fixture_pack_without_sound(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-ref-install-gate");
    options.require_installed = true;

    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("the reference partition is not installed");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|f| f.field == "model.tier"
            && f.message.contains("minimax_h3_ref")
            && f.message.contains("not installed")),
        "the refusal must NAME the partition that is missing: {findings:?}"
    );
    // And it must not blame the base checkpoint, which this host does have.
    assert!(
        !findings
            .iter()
            .any(|f| f.message.contains("minimax_h3 tier")),
        "{findings:?}"
    );
    assert!(harness.jobs().await.is_empty());
}

/// sc-23402 review, the scope gap. The install gate follows the SELECTION: `--shots SH020` on the
/// mixed fixture dispatches only the base checkpoint, so it must run on a host that never
/// downloaded the reference DiT — even with `--require-installed`. The plan DOCUMENT is still
/// validated whole (SH010's reference entry is still resolved and its caps still judged); only the
/// weights-on-disk demand narrows.
#[tokio::test]
async fn a_shot_filtered_run_does_not_demand_an_unselected_partitions_weights() {
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
        only_the_base_partition_is_installed(body);
    });
    let plan = harness.mixed_partition_plan(|_| {});
    let mut options = harness.options(plan, harness.fixture_pack_without_sound(), Some(&["SH020"]));
    options.out_dir = harness.temp_dir.path().join("run-out-selected-base-only");
    options.require_installed = true;

    let record = film_harness::run(&transport, &options)
        .await
        .expect("a base-only selection runs with the reference DiT absent");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.selected_shot_ids, vec!["SH020".to_owned()]);
    let selected = record
        .shots
        .iter()
        .find(|shot| shot.shot_id == "SH020")
        .expect("SH020 has a shot record");
    assert_eq!(selected.attempts[0].resolved_model_id, "minimax_h3");

    // Only the partition it actually loaded is priced into the record's weights.
    let model = record.model.as_ref().expect("a model record");
    let partitions: Vec<&String> = model.partition_weights.keys().collect();
    assert_eq!(partitions, vec!["minimax_h3"], "{partitions:?}");

    // Selecting the reference shot instead DOES demand it — same plan, same host, same gate.
    let plan = harness.mixed_partition_plan(|_| {});
    let mut options = harness.options(plan, harness.fixture_pack_without_sound(), Some(&["SH010"]));
    options.out_dir = harness.temp_dir.path().join("run-out-selected-reference");
    options.require_installed = true;
    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("the selected shot needs the reference partition");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|f| f.field == "model.tier"
            && f.message.contains("minimax_h3_ref")
            && f.message.contains("not installed")),
        "{findings:?}"
    );
}

/// `GET /api/v1/jobs/{id}`, for the assertions above and for the epic's acceptance tests, which
/// read the body the route actually received rather than the one a compiled request would build.
pub(crate) async fn request_job(
    harness: &Harness,
    job_id: &str,
) -> (axum::http::StatusCode, Value) {
    request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await
}

#[tokio::test]
async fn unknown_model_and_missing_workers_are_refused_before_dispatch() {
    let harness = Harness::start(false, vec![]).await;
    let plan = harness.edited_plan(|plan| plan["model"]["id"] = json!("no_such_model"));
    // Each refusal gets its own run directory: a refused run still writes its record there, and a
    // second `run` over a directory that already holds one is itself refused.
    let mut options = harness.options(plan, harness.fixture_pack(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-unknown-model");
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|f| f.field == "model.id" && f.message.contains("not in this API's model catalog")),
        "{findings:?}"
    );

    // A valid plan with no registered worker: the memory budget cannot be checked and nothing
    // could claim the job, so it is refused rather than queued forever.
    let mut options = harness.options(harness.fixture_plan(), harness.fixture_pack(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-no-worker");
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("video_generate")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("timeline_export")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("cannot be checked")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());

    // The install gate: a tier the catalog reports missing is refused when the gate is on.
    let mut options = harness.options(harness.fixture_plan(), harness.fixture_pack(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-install-gate");
    options.require_installed = true;
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|f| f.field == "model.tier" && f.message.contains("not installed")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn shot_budget_cancels_a_hung_job_and_the_attempt_cap_bounds_retries() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 2, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run finishes with a record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let sh010 = &record.shots[0];
    assert_eq!(sh010.outcome, ShotOutcome::TimedOut);
    assert_eq!(
        sh010.attempts.len(),
        2,
        "exactly the declared attempt cap: {:#?}",
        sh010.attempts
    );
    for attempt in &sh010.attempts {
        assert_eq!(attempt.status, "timed_out");
        assert!(attempt
            .error
            .as_deref()
            .unwrap()
            .contains("per-shot budget of 2s"));
        let job_id = attempt.job_id.as_deref().unwrap();
        let (_, job) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/jobs/{job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(job["status"], "canceled", "{job}");
        assert!(attempt.elapsed_seconds >= 2.0);
    }
    let sh020 = &record.shots[1];
    assert_eq!(
        sh020.outcome,
        ShotOutcome::Rendered,
        "an unaffected shot still renders\n{}",
        summary(&record)
    );
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(
        video_jobs, 3,
        "2 capped attempts + 1 rendered shot, no unbounded retry"
    );
    assert!(
        record.timeline.is_some(),
        "the rendered take is still assembled"
    );
    assert_eq!(harness.run_record()["outcome"], "failed");
}

#[tokio::test]
async fn a_failed_attempt_is_retried_only_up_to_the_declared_cap() {
    let harness = Harness::start(true, vec![("SH020", VideoBehavior::FailFirst)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"]["maxAttemptsPerShot"] = json!(2);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let sh020 = &record.shots[1];
    assert_eq!(sh020.attempts.len(), 2);
    assert_eq!(sh020.attempts[0].status, "failed");
    assert_eq!(
        sh020.attempts[0].error.as_deref(),
        Some("fake engine fault: transient")
    );
    assert_eq!(sh020.attempts[1].status, "completed");
    assert_ne!(sh020.attempts[0].job_id, sh020.attempts[1].job_id);
    assert_eq!(sh020.outcome, ShotOutcome::Rendered);
}

#[tokio::test]
async fn run_budget_stops_dispatch_and_leaves_the_remaining_shots_undispatched() {
    // A job that never finishes on its own, so once dispatched only the run budget can end it.
    // `maxShotSeconds` equals `maxRunSeconds` and the run deadline starts first, so it wins the
    // tie. The run budget deliberately includes setup too: on a loaded runner it may expire before
    // the first dispatch. Both points must stop all later dispatch and leave the same durable,
    // terminal run-budget outcome; the focused PollBounds unit test pins the in-flight precedence
    // without relying on twelve quiet wall-clock seconds.
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 12, "maxShotSeconds": 12, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::StoppedRunBudget,
        "{:#?}",
        record.shots
    );
    let claimed = harness.script.lock().claimed.len();
    match record.shots[0].outcome {
        ShotOutcome::TimedOut => {
            assert_eq!(
                record.shots[0].attempts.len(),
                1,
                "the run budget stops retries too"
            );
            assert!(record.shots[0].attempts[0]
                .error
                .as_deref()
                .unwrap()
                .contains("run exceeded its budget of 12s"));
            assert_eq!(claimed, 1, "only the timed-out first job was dispatched");
        }
        ShotOutcome::NotDispatched => {
            assert!(
                record.shots[0].attempts.is_empty(),
                "a budget exhausted during setup cannot leave a partial attempt"
            );
            assert_eq!(
                claimed, 0,
                "a run whose budget expired during setup cannot dispatch after its deadline"
            );
        }
        outcome => panic!("run budget left the first shot in {outcome:?}"),
    }
    for shot in &record.shots[1..] {
        assert_eq!(shot.outcome, ShotOutcome::NotDispatched, "{shot:?}");
        assert!(shot.attempts.is_empty());
    }
    assert!(record.timeline.is_none());
    assert!(record.export.is_none());
    let stop = record.stop.as_ref().expect("the run records its stop");
    assert_eq!(stop.reason, "run_budget");
    assert!(!stop.resumable, "an exhausted run budget is terminal");
    let persisted = harness.run_record();
    assert_eq!(persisted["outcome"], "stopped_run_budget");
    assert_eq!(persisted["stop"]["reason"], "run_budget");
    assert_eq!(persisted["stop"]["resumable"], false);
}

/// The memory limit reads the SAME signal a real render produces: the job's `generation_metrics`
/// block (`peakMemoryBytes`), posted through `POST /api/v1/jobs/:id/metrics` after the terminal
/// progress. Nothing in this file writes `peakGpuMemoryPct` — a job snapshot field every shipped
/// worker leaves null — so the limit cannot be satisfied by a fabricated signal.
#[tokio::test]
async fn observed_memory_over_budget_stops_new_dispatch() {
    // 90% of the 128 GiB the fake worker reports is 115.2 GiB, over the plan's 96 GB budget.
    let harness = Harness::start(
        true,
        vec![(
            "SH010",
            VideoBehavior::Complete {
                delay_secs: 1,
                peak_pct: 90.0,
            },
        )],
    )
    .await;
    // Sound-free: this test is about the memory budget, and importing the pack's clips transcodes
    // through ffmpeg, which the hosted macOS lane does not have. Same reasoning as `edited_plan`.
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::StoppedMemoryLimit,
        "{}",
        summary(&record)
    );
    assert_eq!(
        record.shots[0].outcome,
        ShotOutcome::Rendered,
        "the finished take is kept\n{}",
        summary(&record)
    );
    let attempt = &record.shots[0].attempts[0];
    assert_eq!(
        attempt.peak_memory_source.as_deref(),
        Some("metrics.peakMemoryBytes"),
        "the limit must read the production signal, not the job snapshot"
    );
    assert!(
        attempt
            .peak_memory_gb
            .is_some_and(|gb| (gb - 115.2).abs() < 0.01),
        "{:?}",
        attempt.peak_memory_gb
    );
    assert_eq!(attempt.peak_gpu_memory_pct, Some(90.0));
    assert_eq!(record.shots[1].outcome, ShotOutcome::NotDispatched);
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(video_jobs, 1);
    assert_eq!(harness.run_record()["outcome"], "stopped_memory_limit");
    // The peak the harness compared is the one the metrics route holds for that job.
    let job_id = attempt.job_id.as_deref().expect("job id");
    let (status, metrics) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/metrics"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(
        metrics["peakMemoryBytes"].as_u64(),
        Some((HOST_MEMORY_MB as f64 * 1024.0 * 1024.0 * 0.90) as u64)
    );
}

/// E2's production record survives the failure it exists to explain: a run that has already
/// created a project and imported assets must still leave a `run.json` when the API fails mid-run.
#[tokio::test]
async fn an_api_failure_mid_run_still_writes_the_run_record() {
    let harness = Harness::start(true, vec![]).await;
    // The tag PATCH is the first write AFTER the project exists and the first reference has been
    // imported, so the run fails with real side effects already on disk.
    let transport = ScriptedTransport::failing(harness.app.clone(), "PATCH", "/tags", 500);
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("the injected 500 fails the run");
    assert!(
        matches!(&error, HarnessError::Api { status: 500, path, .. } if path.contains("/tags")),
        "{error}"
    );

    let record = harness.run_record();
    assert_eq!(record["outcome"], "failed");
    assert!(
        record["projectId"].is_string(),
        "the record names the project the run created: {record}"
    );
    let diagnostics: Vec<String> = record["diagnostics"]
        .as_array()
        .expect("diagnostics")
        .iter()
        .map(|finding| finding["message"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        diagnostics.iter().any(|message| message.contains("500")
            && message.contains("/tags")
            && message.contains("the run stopped on an error")),
        "{diagnostics:?}"
    );
    assert!(harness.temp_dir.path().join("run-out/plan.json").exists());
}

/// A cancel the worker never honours must stop the run, not free the harness to dispatch a second
/// render against the one memory budget the plan declared.
#[tokio::test]
async fn a_cancel_the_worker_ignores_stops_dispatch_instead_of_retrying() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::HangIgnoringCancel)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 2, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run finishes with a record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let sh010 = &record.shots[0];
    assert_eq!(sh010.outcome, ShotOutcome::TimedOut);
    assert_eq!(
        sh010.attempts.len(),
        1,
        "the uncancelable attempt consumes the shot: {:#?}",
        sh010.attempts
    );
    assert!(
        sh010.attempts[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("still running")
            && sh010.attempts[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("stopping dispatch rather than running a second render"),
        "{:?}",
        sh010.attempts[0].error
    );
    assert_eq!(
        record.shots[1].outcome,
        ShotOutcome::NotDispatched,
        "no further shot goes out while the first render is still in flight"
    );
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(video_jobs, 1, "exactly one render was ever dispatched");
    assert_eq!(harness.run_record()["outcome"], "failed");
}

/// A terminal job whose assets the API never finishes persisting is bounded and reported for what
/// it is, instead of being polled until the shot budget expires and blamed on a timeout.
#[tokio::test]
async fn a_terminal_job_whose_assets_never_settle_is_reported_as_unsettled() {
    let harness = Harness::start(true, vec![]).await;
    // Every job snapshot the harness reads is rewritten to look like the two-phase handoff never
    // completed: raw `assetWrites` still in place, no `assets` / `assetIds`.
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/jobs/", |body| {
        if body.get("type").and_then(Value::as_str) == Some("video_generate")
            && body["status"] == "completed"
        {
            body["result"]["assetWrites"] = json!([{ "type": "video" }]);
            if let Some(result) = body["result"].as_object_mut() {
                result.remove("assets");
                result.remove("assetIds");
            }
        }
    });
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 3, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010"]));
    let record = film_harness::run(&transport, &options)
        .await
        .expect("run finishes with a record");
    let attempt = &record.shots[0].attempts[0];
    assert_eq!(attempt.status, "completed", "{}", summary(&record));
    assert!(
        attempt
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("assets never settled"),
        "{:?}",
        attempt.error
    );
    assert_eq!(
        record.shots[0].attempts.len(),
        1,
        "a server-side persistence stall is not retried into a second render"
    );
    assert_eq!(record.shots[0].outcome, ShotOutcome::Failed);
    assert!(
        attempt.elapsed_seconds < 60.0,
        "the wait is bounded, not left to the shot budget: {}",
        attempt.elapsed_seconds
    );
}

/// A timeline with no track to hold the takes is an error, not a silently empty save: the record
/// must never list a sequence the project does not hold.
#[tokio::test]
async fn a_timeline_with_no_video_track_fails_the_run_instead_of_saving_nothing() {
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/timelines", |body| {
        if body.get("tracks").is_some() {
            body["tracks"] = json!([{ "id": "track_audio", "kind": "audio", "items": [] }]);
        }
    });
    // Sound-free for the same reason as the budget test above: nothing here is about sound, and the
    // import transcodes through an ffmpeg the hosted macOS lane does not have.
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("a timeline with nowhere to put the takes stops the run");
    assert!(
        error
            .to_string()
            .contains("has no track_main and no video track"),
        "{error}"
    );
    // E2 again: the record still lands, with the rendered take and the failure both in it.
    let record = harness.run_record();
    assert_eq!(record["outcome"], "failed");
    assert_eq!(record["shots"][0]["outcome"], "rendered");
    assert!(record["timeline"].is_null(), "{record}");
}

/// An unapproved reference is still imported and recorded, but is distinguishable in the record and
/// in the project — AC1 is about APPROVED references staying addressable.
#[tokio::test]
async fn unapproved_references_are_tagged_and_recorded_apart_from_approved_ones() {
    let harness = Harness::start(true, vec![]).await;
    let pack = harness.edited_pack(|pack| {
        for entry in pack["references"].as_array_mut().expect("references") {
            if entry["role"] == "house_style" {
                entry["approved"] = json!(false);
            }
        }
    });
    let options = harness.options(harness.edited_plan(|_| {}), pack, Some(&["SH010"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let project_id = record.project_id.clone().expect("project created");
    let unapproved = record
        .references
        .iter()
        .find(|reference| reference.role == "house_style")
        .expect("the unapproved reference is still imported and recorded");
    assert!(!unapproved.approved);
    assert!(
        record
            .references
            .iter()
            .filter(|reference| reference.role != "house_style")
            .all(|reference| reference.approved),
        "{:#?}",
        record.references
    );
    let tags_for = |asset_id: &str| {
        let app = harness.app.clone();
        let path = format!("/api/v1/projects/{project_id}/assets/{asset_id}");
        async move {
            let (_, asset) = request(app, "GET", &path, Value::Null).await;
            asset["tags"]
                .as_array()
                .map(|tags| {
                    tags.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }
    };
    let tags = tags_for(&unapproved.asset_id).await;
    assert!(
        tags.iter()
            .any(|tag| tag == "film-harness-reference-unapproved"),
        "{tags:?}"
    );
    assert!(
        !tags.iter().any(|tag| tag == "film-harness-reference"),
        "an unapproved plate must not carry the conditioning-eligible tag: {tags:?}"
    );
    let approved = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("approved plate");
    let tags = tags_for(&approved.asset_id).await;
    assert!(
        tags.iter().any(|tag| tag == "film-harness-reference"),
        "{tags:?}"
    );
}

/// Ctrl-C on the binary: the in-flight job is canceled through the API, nothing else is dispatched,
/// and the record still lands with the reason in it.
#[tokio::test]
async fn an_operator_cancel_stops_the_run_and_still_writes_the_record() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let control = film_harness::RunControl::new();
    harness.script.lock().running_hook = Some((
        "SH010".to_owned(),
        RunningHook::CancelControl(control.clone()),
    ));
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("run finishes with a record");
    assert!(
        harness.script.lock().running_hook.is_none(),
        "the fake worker observed SH010 running and consumed the cancellation hook"
    );

    // sc-22711 gives a cancel its own outcome and a RESUMABLE stop, rather than folding it into
    // `failed` with a diagnostic: nothing went wrong, and the run can be picked back up.
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "canceled");
    assert!(stop.resumable, "a cancel leaves work a resume can finish");
    let sh010 = &record.shots[0];
    assert_eq!(sh010.attempts.len(), 1, "{:#?}", sh010.attempts);
    assert_eq!(sh010.attempts[0].status, "canceled_by_operator");
    assert_eq!(sh010.outcome, ShotOutcome::Canceled);
    let job_id = sh010.attempts[0].job_id.as_deref().expect("job id");
    let (_, job) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(job["status"], "canceled", "{job}");
    let (status, _) = request(
        harness.app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let (_, visible_jobs) = request(harness.app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        visible_jobs
            .as_array()
            .unwrap()
            .iter()
            .all(|job| job["id"] != job_id),
        "the canceled attempt is hidden from the queue UI before resume"
    );
    assert_eq!(record.shots[1].outcome, ShotOutcome::NotDispatched);
    let on_disk = harness.run_record();
    assert_eq!(on_disk["outcome"], "canceled");
    assert_eq!(on_disk["state"], "finished");
    assert!(
        on_disk["stop"]["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("canceled"),
        "the record says what stopped it: {}",
        on_disk["stop"]
    );
    assert_eq!(on_disk["stop"]["resumable"], true);
}

#[test]
fn checked_in_fixture_plates_match_the_generator_byte_for_byte() {
    for (role, rgb) in FIXTURE_REFERENCES {
        let path = Path::new(FIXTURE_DIR).join(format!("references/{role}.png"));
        let on_disk = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "{}: {error} (regenerate with `film-harness fixture-images`)",
                path.display()
            )
        });
        let generated = film_harness::fixture_plate_png(role, *rgb).unwrap();
        assert_eq!(
            on_disk,
            generated,
            "{} drifted from the generator; regenerate with `film-harness fixture-images --out {}`",
            path.display(),
            Path::new(FIXTURE_DIR).join("references").display()
        );
    }
    // Every plate the pack names is one the generator writes, and vice versa.
    let pack_text =
        std::fs::read_to_string(Path::new(FIXTURE_DIR).join("references.jsonc")).unwrap();
    let pack = sceneworks_core::film_plan::parse_reference_pack(&pack_text).unwrap();
    let mut pack_roles: Vec<&str> = pack.references.iter().map(|r| r.role.as_str()).collect();
    let mut generated_roles: Vec<&str> = FIXTURE_REFERENCES.iter().map(|(role, _)| *role).collect();
    pack_roles.sort_unstable();
    generated_roles.sort_unstable();
    assert_eq!(pack_roles, generated_roles);
}

#[tokio::test]
async fn validate_subcommand_path_checks_documents_without_an_api() {
    let temp_dir = tempfile::tempdir().unwrap();
    let options = RunOptions {
        plan_path: Path::new(FIXTURE_DIR).join("plan.jsonc"),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        compiled_path: None,
        project_id: None,
        shot_ids: Some(vec!["SH010".into(), "SH999".into()]),
        out_dir: temp_dir.path().to_path_buf(),
        poll_interval: Duration::from_secs(1),
        export: true,
        require_installed: true,
    };
    let error = film_harness::validate(None, &options).await.unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].message.contains("\"SH999\" is not in plan"));
    let mut options = options;
    options.shot_ids = None;
    let (plan, pack) = film_harness::validate(None, &options).await.unwrap();
    assert_eq!(plan.shots.len(), 6);
    assert_eq!(pack.references.len(), 7);
}

// ---------------------------------------------------------------------------------------------
// sc-22711 — durable run state: crash windows, reconciliation, cancellation, take replacement
// ---------------------------------------------------------------------------------------------

/// Where a simulated controller death lands relative to the API call it died on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultMode {
    /// The call never reaches the API: the mutation did not happen.
    Before,
    /// The call reaches the API and succeeds, but the controller never sees the answer. This is the
    /// window that matters — the job (or project, or asset) exists and its id is nowhere in the
    /// record, so only an idempotency key can stop a replay from creating a second one.
    After,
}

/// A transport that stops answering the controller after `die_at` calls. Everything after the fault
/// fails too, because the process it models is gone.
struct FaultTransport {
    inner: RouterTransport,
    die_at: usize,
    mode: FaultMode,
    /// When set, only POSTs whose path contains this needle are counted, which aims the fault at
    /// one specific transition instead of sweeping every call.
    route: Option<&'static str>,
    matched: AtomicUsize,
    dead: AtomicBool,
}

impl FaultTransport {
    fn new(app: axum::Router, die_at: usize, mode: FaultMode) -> Self {
        Self {
            inner: RouterTransport { app },
            die_at,
            mode,
            route: None,
            matched: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
        }
    }

    fn on_post_route(mut self, needle: &'static str) -> Self {
        self.route = Some(needle);
        self
    }

    fn fired(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }
}

impl ApiTransport for FaultTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        if self.fired() {
            return Box::pin(async {
                Err(HarnessError::Transport("controller is gone".to_owned()))
            });
        }
        let targeted = self
            .route
            .is_none_or(|needle| request.method == "POST" && request.path.contains(needle));
        if !targeted || self.matched.fetch_add(1, Ordering::SeqCst) + 1 < self.die_at {
            return self.inner.call(request);
        }
        self.dead.store(true, Ordering::SeqCst);
        let deliver = self.mode == FaultMode::After;
        Box::pin(async move {
            if deliver {
                // The API really performs the write; the controller just never learns the answer.
                let _ = self.inner.call(request).await?;
            }
            Err(HarnessError::Transport(
                "simulated controller death".to_owned(),
            ))
        })
    }

    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        if self.fired() {
            return Box::pin(async {
                Err(HarnessError::Transport("controller is gone".to_owned()))
            });
        }
        self.inner.get_bytes(path)
    }
}

pub(crate) fn fast(shots: &[&str]) -> Vec<(&'static str, VideoBehavior)> {
    const IDS: &[&str] = &["SH010", "SH020", "SH030", "SH040", "SH050", "SH060"];
    IDS.iter()
        .filter(|id| shots.contains(id))
        .map(|id| {
            (
                *id,
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            )
        })
        .collect()
}

/// What identifies one shot's selected take: which attempt, which job, which asset. Compared
/// instead of the whole `AttemptRecord` whenever one side has been through the record on disk — an
/// `elapsedSeconds` measurement does not survive a JSON round trip in its last bit, and none of
/// these assertions are about that.
fn selected_identity(
    record: &RunRecord,
    shot_id: &str,
) -> Option<(u32, Option<String>, Option<String>)> {
    let attempt = record.shot(shot_id)?.selected()?;
    Some((
        attempt.attempt,
        attempt.job_id.clone(),
        attempt.take.as_ref().map(|take| take.asset_id.clone()),
    ))
}

/// The selected take of every shot, as `shot -> (attempt, asset)`. The thing replay must never move.
fn selections(record: &RunRecord) -> Vec<(String, Option<(u32, String)>)> {
    record
        .shots
        .iter()
        .map(|shot| {
            (
                shot.shot_id.clone(),
                shot.selected().and_then(|attempt| {
                    Some((attempt.attempt, attempt.take.as_ref()?.asset_id.clone()))
                }),
            )
        })
        .collect()
}

/// Kill the controller at EVERY point it can die — before and after each API call it makes — and
/// resume from the record it left behind. Whatever the window, the run must finish with exactly the
/// work a clean run does: two video jobs, one project, seven reference assets, and the same take
/// selected for each shot.
#[tokio::test]
async fn a_crash_at_every_window_replays_without_duplicate_work_or_a_moved_take() {
    for mode in [FaultMode::Before, FaultMode::After] {
        let mut window = 1_usize;
        loop {
            assert!(window < 300, "sweep did not terminate ({mode:?})");
            let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
            let transport = FaultTransport::new(harness.app.clone(), window, mode);
            // The two-shot / one-reference documents: every distinct transition a run makes, none
            // of the near-identical repeats seven reference imports would add to the sweep. The
            // multi-reference replay has its own targeted test below.
            let (plan, pack) = harness.minimal_documents(json!({
                "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2,
                "maxMemoryGb": 96
            }));
            let options = harness.options(plan, pack, None);
            let crashed = film_harness::run(&transport, &options).await;
            if !transport.fired() {
                // Past the last call a clean run makes: the sweep has covered every window.
                let record = crashed.expect("a run that never faulted completes");
                assert_eq!(
                    record.outcome,
                    RunOutcome::Completed,
                    "{}",
                    summary(&record)
                );
                assert_eq!(harness.video_job_count(), 2);
                assert!(
                    window > 20,
                    "{mode:?}: only {} windows swept — the run stopped making calls far earlier \
                     than a project + reference + two dispatches + timeline + export should",
                    window - 1
                );
                break;
            }
            crashed.expect_err("the fault stops the controller");

            let context = format!("{mode:?} window {window}");
            match film_harness::read_run_record(&harness.out_dir()) {
                Err(_) => {
                    // Died before the record existed. Nothing was created, so there is nothing to
                    // resume — and nothing to clean up either.
                    assert_eq!(harness.video_job_count(), 0, "{context}");
                    assert_eq!(harness.project_count().await, 0, "{context}");
                }
                Ok(crashed_record) => {
                    assert_eq!(
                        crashed_record.state,
                        RunState::Running,
                        "{context}: a record left by a dead controller must say so"
                    );
                    let before = selections(&crashed_record);
                    let record = harness.resume_to_completion().await;
                    assert_eq!(
                        record.outcome,
                        RunOutcome::Completed,
                        "{context}\n{}",
                        summary(&record)
                    );
                    assert_eq!(record.run_id, crashed_record.run_id, "{context}");
                    assert_eq!(
                        harness.video_job_count(),
                        2,
                        "{context}: replay enqueued duplicate work\n{}",
                        summary(&record)
                    );
                    assert_eq!(harness.project_count().await, 1, "{context}");
                    assert_eq!(
                        harness.timelines_for(&record).await.len(),
                        1,
                        "{context}: replay left the project holding two timelines"
                    );
                    assert_eq!(record.references.len(), 1, "{context}");
                    let mut asset_ids: Vec<&str> = record
                        .references
                        .iter()
                        .map(|reference| reference.asset_id.as_str())
                        .collect();
                    asset_ids.sort_unstable();
                    let unique = asset_ids.len();
                    asset_ids.dedup();
                    assert_eq!(
                        asset_ids.len(),
                        unique,
                        "{context}: a reference was imported twice"
                    );
                    // A take the crashed record had already selected is still the selected take.
                    for (shot_id, selected) in before {
                        if let Some(expected) = selected {
                            let actual = record
                                .shot(&shot_id)
                                .and_then(|shot| shot.selected())
                                .map(|attempt| {
                                    (
                                        attempt.attempt,
                                        attempt.take.as_ref().unwrap().asset_id.clone(),
                                    )
                                });
                            assert_eq!(
                                actual,
                                Some(expected),
                                "{context}: {shot_id}'s selected take moved"
                            );
                        }
                    }
                    for shot_id in ["SH010", "SH020"] {
                        let shot = record.shot(shot_id).expect("shot recorded");
                        assert_eq!(shot.outcome, ShotOutcome::Rendered, "{context} {shot_id}");
                        assert_eq!(
                            shot.attempts.len(),
                            1,
                            "{context} {shot_id}: one attempt, not one per restart"
                        );
                    }
                    let export = record.export.as_ref().expect("export recorded");
                    assert_eq!(export.status, "completed", "{context}");
                    let exports = harness
                        .script
                        .lock()
                        .claimed
                        .iter()
                        .filter(|(kind, _, _)| kind == "timeline_export")
                        .count();
                    assert_eq!(exports, 1, "{context}: the export ran twice");
                }
            }
            window += 1;
        }
    }
}

/// Dying right after a reference upload the record never learned about must not import it twice.
/// Swept over every reference in the shipped seven-role pack, so "the first one" and "the last one"
/// are both covered.
#[tokio::test]
async fn a_reference_imported_but_not_recorded_is_adopted_not_imported_again() {
    for nth in [1_usize, 4, 7] {
        let harness = Harness::start(true, fast(&["SH010"])).await;
        let transport = FaultTransport::new(harness.app.clone(), nth, FaultMode::After)
            .on_post_route("/assets");
        let options = harness.options(
            harness.edited_plan(|_| {}),
            harness.fixture_pack_without_sound(),
            Some(&["SH010"]),
        );
        film_harness::run(&transport, &options)
            .await
            .expect_err("the fault stops the controller");
        assert!(transport.fired(), "reference {nth}");

        let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
        assert_eq!(
            crashed.references.len(),
            nth - 1,
            "reference {nth}: the upload that was answered is the one the record missed"
        );
        let record = harness.resume_to_completion().await;
        assert_eq!(
            record.outcome,
            RunOutcome::Completed,
            "{}",
            summary(&record)
        );
        assert_eq!(record.references.len(), 7, "reference {nth}");

        // Exactly seven reference assets exist in the project — the interrupted upload was adopted
        // by its provenance rather than uploaded a second time.
        let project_id = record.project_id.clone().expect("project");
        let (_, assets) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets"),
            Value::Null,
        )
        .await;
        let references: Vec<&Value> = assets
            .as_array()
            .into_iter()
            .flatten()
            .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
            .collect();
        assert_eq!(
            references.len(),
            7,
            "reference {nth}: a reference was imported twice"
        );
        let mut roles: Vec<&str> = references
            .iter()
            .filter_map(|asset| asset["extra"]["filmHarness"]["role"].as_str())
            .collect();
        roles.sort_unstable();
        roles.dedup();
        assert_eq!(roles.len(), 7, "reference {nth}: duplicate roles");

        // The upload and the tag PATCH are two writes: the ADOPTED asset — the one whose upload the
        // record missed — must end up tagged like every other, or a query for the
        // conditioning-eligible references silently misses it while the record claims it is tagged.
        for asset in &references {
            let role = asset["extra"]["filmHarness"]["role"]
                .as_str()
                .expect("role provenance");
            let tags: Vec<&str> = asset["tags"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            assert!(
                tags.contains(&"film-harness-reference"),
                "reference {nth}: {role} carries no kind tag: {tags:?}"
            );
            assert!(
                tags.contains(&format!("role:{role}").as_str()),
                "reference {nth}: {role} carries no role tag: {tags:?}"
            );
        }
    }
}

#[tokio::test]
async fn resuming_a_finished_run_reuses_every_take_and_enqueues_nothing() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(first.outcome, RunOutcome::Completed, "{}", summary(&first));
    assert!(
        !first.is_resumable(),
        "a completed run is not something to resume"
    );

    // A completed run refuses a resume outright rather than re-dispatching anything.
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("a completed run is not resumable");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("not resumable")),
        "{error}"
    );
    assert_eq!(harness.video_job_count(), 2);
    assert_eq!(selections(&harness_record(&harness)), selections(&first));
}

/// Reading the record back off disk, which is what a separate `film-harness` invocation does.
pub(crate) fn harness_record(harness: &Harness) -> RunRecord {
    film_harness::read_run_record(&harness.out_dir()).expect("run record on disk")
}

/// Block until `shot_id`'s job has finished, its assets are persisted AND its metrics block has
/// landed — everything a resume needs to ADOPT the attempt rather than poll it. Without this a
/// resume races the fake worker and exercises the dispatch path instead of the reconciliation one.
async fn wait_for_settled_shot(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    shot_id: &str,
) -> String {
    let job_id = loop {
        let changed = {
            let script = script.lock();
            if let Some((_, job_id)) = script
                .settled_video_jobs
                .iter()
                .find(|(settled_shot_id, _)| settled_shot_id == shot_id)
            {
                break job_id.clone();
            }
            script.settled_video_jobs_changed.clone().notified_owned()
        };
        changed.await;
    };
    let (_, job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    let (_, metrics) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/metrics"),
        Value::Null,
    )
    .await;
    assert_eq!(job["status"], "completed", "{job}");
    assert!(job["result"]["assets"].is_array(), "{job}");
    assert!(!metrics.is_null(), "metrics missing for {job_id}");
    job_id
}

#[tokio::test]
async fn a_cancel_stops_dispatch_keeps_finished_takes_and_the_run_resumes() {
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
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan,
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    // Trip the cancel once SH010 is safely rendered and SH020 is hanging.
    let control = RunControl::new();
    harness.script.lock().running_hook = Some((
        "SH020".to_owned(),
        RunningHook::CancelControl(control.clone()),
    ));
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    assert!(
        harness.script.lock().running_hook.is_none(),
        "the fake worker observed SH020 running and consumed the cancellation hook"
    );

    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "canceled");
    assert!(stop.resumable, "a cancel leaves work a resume can finish");
    assert!(record.is_resumable());

    // The finished take is kept, the hung job is canceled through the API, and the shot that was
    // never reached was never dispatched.
    let sh010 = record.shot("SH010").expect("SH010 recorded");
    assert_eq!(sh010.outcome, ShotOutcome::Rendered);
    assert_eq!(sh010.selected_attempt, Some(1));
    let sh020 = record.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 1, "a cancel is not a retry");
    assert_eq!(sh020.attempts[0].status, "canceled_by_operator");
    assert_eq!(
        sh020.outcome,
        ShotOutcome::Canceled,
        "a canceled shot is not a failed one — nothing went wrong with it"
    );
    let job_id = sh020.attempts[0].job_id.as_deref().expect("a job id");
    let (_, job) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(job["status"], "canceled", "{job}");
    assert_eq!(
        record.shot("SH030").expect("SH030 recorded").outcome,
        ShotOutcome::NotDispatched
    );
    assert!(record.export.is_none(), "a cancel dispatches no export");
    assert_eq!(harness.run_record()["outcome"], "canceled");

    // Resuming reads the hidden job back by exact id, finishes the run, and never duplicates the
    // completed take: SH020 gets its remaining attempt, SH030 is dispatched, and SH010 is reused.
    harness
        .script
        .lock()
        .behaviors
        .retain(|(id, _)| id != "SH020");
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        resumed.shot("SH010").unwrap().attempts.len(),
        1,
        "SH010 was not re-rendered"
    );
    // Identity, not the whole struct: the resumed record came back through JSON, and an
    // `elapsedSeconds` measurement does not survive that round trip in its last bit.
    assert_eq!(
        selected_identity(&resumed, "SH010"),
        selected_identity(&record, "SH010"),
        "the selected take survived the cancel and the resume"
    );
    assert_eq!(resumed.shot("SH020").unwrap().selected_attempt, Some(2));
    assert!(resumed
        .decisions
        .iter()
        .any(|decision| decision.action == "resume"));
}

#[tokio::test]
async fn a_cancel_sentinel_written_by_another_process_stops_the_run() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    // The control a `film-harness run` builds: it watches its own run directory, which is how
    // `film-harness cancel --out DIR` in another shell reaches it.
    let control = RunControl::watching(&harness.out_dir());
    harness.script.lock().running_hook = Some((
        "SH010".to_owned(),
        RunningHook::WriteCancelSentinel(harness.out_dir()),
    ));
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    assert!(
        harness.script.lock().running_hook.is_none(),
        "the fake worker observed SH010 running and wrote the cancel sentinel"
    );
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    assert!(record.stop.as_ref().expect("stop").resumable);
    assert_eq!(
        record.shot("SH010").unwrap().attempts.len(),
        1,
        "the attempt cap was never spent on a cancel"
    );
    assert!(
        harness.out_dir().join("cancel.requested").exists(),
        "the sentinel is left for the operator to see"
    );

    // The stale sentinel must not cancel the resume the operator asked for.
    harness.script.lock().behaviors.clear();
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert!(!harness.out_dir().join("cancel.requested").exists());
}

#[tokio::test]
async fn an_exhausted_budget_is_terminal_and_refuses_a_resume() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 12, "maxShotSeconds": 12, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(record.outcome, RunOutcome::StoppedRunBudget);
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "run_budget");
    assert!(
        !stop.resumable,
        "resuming an exhausted budget is exactly the unbounded retry the budget exists to prevent"
    );
    assert!(
        stop.detail.contains("maxRunSeconds"),
        "the terminal reason says what to change: {}",
        stop.detail
    );
    let before = harness.video_job_count();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("terminal runs refuse a resume");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("not resumable") && message.contains("run_budget")),
        "{error}"
    );
    assert_eq!(
        harness.video_job_count(),
        before,
        "the refusal dispatched nothing"
    );
}

/// The wall-clock budget bounds the RUN, not one attempt at it. A resume inherits what earlier
/// controllers already spent, which is what stops a restart loop from turning a bounded run into an
/// unbounded one.
#[tokio::test]
async fn a_resume_inherits_the_wall_clock_already_spent() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let control = RunControl::watching(&harness.out_dir());
    harness.script.lock().running_hook = Some((
        "SH010".to_owned(),
        RunningHook::WriteCancelSentinel(harness.out_dir()),
    ));
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .unwrap();
    assert!(
        harness.script.lock().running_hook.is_none(),
        "the fake worker observed SH010 running and wrote the cancel sentinel"
    );
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    assert!(record.is_resumable());
    let dispatched = harness.api_video_job_count().await;

    // Book the run as having spent its whole budget, exactly as a long first controller would have.
    let path = harness.out_dir().join("run.json");
    let mut on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    on_disk["elapsedSeconds"] = json!(record.limits.max_run_seconds);
    std::fs::write(&path, serde_json::to_string_pretty(&on_disk).unwrap()).unwrap();

    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("a spent budget still returns a record");
    assert_eq!(resumed.outcome, RunOutcome::StoppedRunBudget);
    let stop = resumed.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "run_budget");
    assert!(
        !stop.resumable,
        "the budget is spent; another resume would be the unbounded retry it exists to prevent"
    );
    assert_eq!(
        harness.api_video_job_count().await,
        dispatched,
        "a resume with no budget left dispatches nothing"
    );
    assert!(resumed.timeline.is_none(), "and assembles nothing");
    // And it really is terminal now.
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("terminal");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("run_budget")),
        "{error}"
    );
}

#[tokio::test]
async fn resume_refuses_a_plan_that_changed_under_it() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan_path = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 1, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan_path.clone(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_ne!(record.outcome, RunOutcome::Completed);

    // Edit the plan the run was started from, then try to resume onto it.
    let text = std::fs::read_to_string(&plan_path).unwrap();
    let mut plan: Value = serde_json::from_str(&text).unwrap();
    plan["shots"][0]["prompt"] = json!("an entirely different shot");
    std::fs::write(&plan_path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
    // The copy the run kept beside its record would otherwise satisfy the read, so remove it: the
    // point is that the SOURCE no longer hashes to what the run recorded.
    std::fs::remove_file(harness.out_dir().join("plan.json")).unwrap();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("an edited plan is a new run");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("the plan changed since run")),
        "{error}"
    );
}

#[tokio::test]
async fn replacing_a_take_renders_one_more_and_leaves_every_other_shot_untouched() {
    let harness = Harness::start(true, fast(&["SH010", "SH020", "SH030"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        before.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&before)
    );
    assert_eq!(harness.video_job_count(), 3);
    let untouched_before = other_shots_digest(&before, "SH020");
    let references_before = serde_json::to_value(&before.references).unwrap();
    let rejected_asset = before
        .shot("SH020")
        .and_then(|shot| shot.selected())
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone())
        .expect("SH020 had a take");

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

    // Exactly one more job, for SH020 alone.
    assert_eq!(
        harness.video_job_count(),
        4,
        "one replacement attempt, no re-render of anything else\n{}",
        summary(&after)
    );
    assert_eq!(
        other_shots_digest(&after, "SH020"),
        untouched_before,
        "every other shot's record changed under a replacement"
    );
    assert_eq!(
        serde_json::to_value(&after.references).unwrap(),
        references_before,
        "the imported references are untouched"
    );

    // The rejected take is still there, with its reason, beside the one that replaced it.
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 2);
    let rejection = sh020.attempts[0]
        .rejection
        .as_ref()
        .expect("the old take is marked rejected");
    assert_eq!(rejection.reason, "the parcel is the wrong red");
    assert_eq!(
        sh020.attempts[0].take.as_ref().unwrap().asset_id,
        rejected_asset,
        "the rejected take keeps its asset and provenance"
    );
    assert_eq!(sh020.selected_attempt, Some(2));
    assert!(
        sh020.attempts[1].human_requested,
        "a replacement is a decision, not a retry"
    );
    assert_ne!(
        sh020.attempts[1].take.as_ref().unwrap().asset_id,
        rejected_asset
    );

    // The rejected asset is still the project's — a replacement discards nothing.
    let project_id = after.project_id.clone().unwrap();
    let (status, asset) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{rejected_asset}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{asset}");

    // The declared dependent is flagged, and only it; nothing was regenerated for it.
    let sh030 = after.shot("SH030").expect("SH030 recorded");
    assert_eq!(sh030.needs_review.len(), 1, "{:#?}", sh030.needs_review);
    let flag = &sh030.needs_review[0];
    assert_eq!(flag.source_shot_id, "SH020");
    assert_eq!(flag.dependency, "continuity");
    assert!(
        flag.reason.contains("the parcel is the wrong red"),
        "{}",
        flag.reason
    );
    assert_eq!(sh030.attempts.len(), 1, "a flagged shot is not re-rendered");
    assert!(
        after.shot("SH010").unwrap().needs_review.is_empty(),
        "SH020 depends on SH010, not the other way round"
    );

    // The timeline now carries the new take and nothing else moved; the export is flagged stale.
    let timeline = after.timeline.as_ref().expect("timeline");
    let item = timeline
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH020"))
        .expect("SH020 on the timeline");
    assert_eq!(
        item.asset_id,
        sh020.selected().unwrap().take.as_ref().unwrap().asset_id
    );
    for shot_id in ["SH010", "SH030"] {
        let before_item = before
            .timeline
            .as_ref()
            .unwrap()
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .unwrap();
        let after_item = timeline
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .unwrap();
        assert_eq!(before_item, after_item, "{shot_id}'s timeline item moved");
    }
    let export = after.export.as_ref().expect("the export record is kept");
    assert!(export.stale, "a changed take makes the rendered MP4 stale");
    assert_eq!(
        harness
            .script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "timeline_export")
            .count(),
        1,
        "without --export the replacement re-renders nothing"
    );
    assert!(after
        .decisions
        .iter()
        .any(|decision| decision.action == "replace_take"
            && decision.shot_id.as_deref() == Some("SH020")
            && decision.detail.contains("the parcel is the wrong red")));
    // Choosing not to re-export is not a failure: the replacement landed, the MP4 is just stale.
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));
    assert!(after.stop.is_none());
}

/// A failed export is the one stop a plain `resume` is meant to fix, so it must actually dispatch a
/// NEW export job rather than re-adopt the failed one.
#[tokio::test]
async fn a_failed_export_is_resumable_and_the_retry_is_a_new_job() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    harness.script.lock().export_fails = true;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a failed export still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "export_failed");
    assert!(
        stop.resumable,
        "the shots rendered; only the export needs redoing"
    );
    let failed_export = record
        .export
        .as_ref()
        .expect("export recorded")
        .job_id
        .clone();
    assert_eq!(
        record.shot("SH010").unwrap().selected_attempt,
        Some(1),
        "the take is kept"
    );

    harness.script.lock().export_fails = false;
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    let export = resumed.export.as_ref().expect("export recorded");
    assert_ne!(
        export.job_id, failed_export,
        "the resume re-adopted the failed export instead of running a new one"
    );
    assert_eq!(export.status, "completed");
    assert!(export.asset_id.is_some());
    assert_eq!(
        harness.video_job_count(),
        1,
        "resuming to fix the export re-rendered nothing"
    );
}

/// Every shot except `except`, serialized and hashed — the "nothing else moved" assertion.
///
/// Review flags are cleared first, because a flag is precisely the change a replacement IS allowed
/// to make to a dependent. Everything else — attempts, jobs, takes, selection, outcome — must hash
/// identically before and after.
///
/// Every float is rounded to six decimals first. The two records being compared do not come down
/// the same path: the "before" one is the controller's own in-memory record, the "after" one was
/// read back off disk, and serde_json's float parser can land one ULP away from the value it wrote
/// (`elapsedSeconds: 1.599993458` read back as `1.5999934580000001`), which flips the hash while
/// nothing moved. That is the same round trip `selected_identity` above exists to dodge. Six
/// decimals is ten orders of magnitude coarser than the artifact and far finer than any real
/// re-measurement, so a shot that actually moved — a new attempt, job, take, status or selection —
/// still changes the digest.
fn other_shots_digest(record: &RunRecord, except: &str) -> String {
    let mut value = serde_json::to_value(
        record
            .shots
            .iter()
            .filter(|shot| shot.shot_id != except)
            .map(|shot| {
                let mut shot = shot.clone();
                shot.needs_review.clear();
                shot
            })
            .collect::<Vec<_>>(),
    )
    .expect("shots serialize");
    quantize_floats(&mut value);
    let digest = <sha2::Sha256 as sha2::Digest>::digest(value.to_string().as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Round every float in `value`, in place, to six decimal places.
fn quantize_floats(value: &mut Value) {
    match value {
        Value::Number(number) => {
            if let Some(float) = number.as_f64() {
                if !number.is_i64() && !number.is_u64() {
                    let rounded = (float * 1e6).round() / 1e6;
                    if let Some(number) = serde_json::Number::from_f64(rounded) {
                        *value = Value::Number(number);
                    }
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(quantize_floats),
        Value::Object(fields) => fields.values_mut().for_each(quantize_floats),
        _ => {}
    }
}

#[tokio::test]
async fn replacing_a_take_with_export_re_renders_the_timeline_once() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let first_export = before.export.as_ref().expect("export").job_id.clone();

    let after = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "flat lighting",
    )
    .await
    .expect("replacement runs");
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));
    let export = after.export.as_ref().expect("export");
    assert!(!export.stale, "the re-export is current again");
    assert_ne!(export.job_id, first_export, "a new export job ran");
    assert_eq!(export.status, "completed");
    assert_eq!(harness.export_job_count(), 2, "exactly one re-export");
    assert!(after.export_pending.is_none());

    // A SECOND replacement is where excluding only the most recently superseded export breaks: the
    // candidate set still holds the ORIGINAL export job, and adopting it would record the MP4 of
    // the timeline from before BOTH replacements as the current one.
    let again = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "still flat",
    )
    .await
    .expect("second replacement runs");
    let third = again.export.as_ref().expect("export");
    assert_ne!(
        third.job_id,
        first_export,
        "the second re-export adopted the first export\n{}",
        summary(&again)
    );
    assert_ne!(third.job_id, export.job_id);
    assert_eq!(third.status, "completed");
    assert!(!third.stale);
    assert_eq!(
        harness.export_job_count(),
        3,
        "the second replacement ran no export at all\n{}",
        summary(&again)
    );
    assert_eq!(
        again.superseded_export_job_ids,
        vec![first_export, export.job_id.clone()],
        "every export the record has held is excluded, not only the last"
    );
}

#[tokio::test]
async fn a_failed_replacement_keeps_the_rejection_and_does_not_loop() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(before.outcome, RunOutcome::Completed);
    let rejected_asset = before
        .shot("SH020")
        .and_then(|shot| shot.selected())
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone())
        .expect("SH020 had a take");
    // Replace SH020's scripted behavior; `behavior_for` takes the first match, so pushing would
    // be shadowed by the entry the run already used.
    {
        let mut script = harness.script.lock();
        script.behaviors.retain(|(id, _)| id != "SH020");
        script
            .behaviors
            .push(("SH020".to_owned(), VideoBehavior::FailAlways));
    }

    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "reshoot the approach",
    )
    .await
    .expect("a failed replacement still returns its record");
    assert_eq!(
        harness.video_job_count(),
        3,
        "exactly one replacement attempt, no retry loop\n{}",
        summary(&after)
    );
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 2);
    assert!(
        sh020.attempts[0].rejection.is_some(),
        "the human's rejection stands even though the replacement failed"
    );
    assert!(
        sh020.attempts[0].take.is_some(),
        "the failure evidence — and the old take — are preserved"
    );
    assert_eq!(sh020.attempts[1].status, "failed");
    assert_eq!(
        sh020.attempts[1].error.as_deref(),
        Some("fake engine fault: persistent")
    );
    assert_eq!(sh020.selected_attempt, None);
    assert_eq!(sh020.outcome, ShotOutcome::Failed);
    let stop = after.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "replacement_failed");
    assert!(!stop.resumable, "nothing automatic will try again");
    assert!(
        after.shot("SH010").unwrap().needs_review.is_empty(),
        "a failed replacement flags nothing"
    );

    // The shot now has NO selected take, and the timeline was deliberately not rewritten — so the
    // sequence and the MP4 rendered from it still carry the take the human REJECTED. The record has
    // to say so rather than leave `stale: false` claiming the export is current.
    let timeline = after.timeline.as_ref().expect("the timeline is kept");
    let item = timeline
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH020"))
        .expect("SH020 is still in the sequence");
    assert_eq!(
        item.asset_id, rejected_asset,
        "the timeline still names the rejected take"
    );
    let export = after.export.as_ref().expect("the export record is kept");
    assert!(
        export.stale,
        "the MP4 renders a timeline that carries a take the human rejected, so it is NOT current"
    );
    assert!(
        after
            .decisions
            .iter()
            .any(|decision| decision.action == "replace_take"
                && decision.shot_id.as_deref() == Some("SH020")
                && decision.detail.contains("still carry the REJECTED take")),
        "{:#?}",
        after.decisions
    );
    assert!(
        stop.detail.contains("REJECTED take"),
        "the stop says what the delivered sequence actually holds: {}",
        stop.detail
    );
}

#[tokio::test]
async fn replace_take_refuses_a_shot_the_run_does_not_hold() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    for (shot, expected) in [
        ("SH999", "is not a shot in plan"),
        ("SH020", "is not in run"),
    ] {
        let error =
            film_harness::replace_take(&harness.transport, &harness.resume_options(), shot, "nope")
                .await
                .expect_err("refused");
        assert!(
            matches!(&error, HarnessError::Refused(message) if message.contains(expected)),
            "{shot}: {error}"
        );
    }
    assert_eq!(harness.video_job_count(), 1, "a refusal dispatches nothing");
}

/// Replacing a take while the shot still owes an attempt would orphan that attempt's job, so it is
/// refused and the operator is pointed at the command that settles it.
#[tokio::test]
async fn replace_take_refuses_while_the_shot_still_has_an_unsettled_attempt() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    // Die right after the video job is created: the record holds an attempt with no settled status.
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
        .on_post_route("/api/v1/video/jobs");
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    // The API holds the job even though the worker may not have claimed it yet.
    assert_eq!(harness.api_video_job_count().await, 1);

    let error = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "too soon",
    )
    .await
    .expect_err("an unsettled attempt blocks a replacement");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("still has attempt 1 in flight") && message.contains("resume")),
        "{error}"
    );
    assert_eq!(
        harness.api_video_job_count().await,
        1,
        "the refusal dispatched nothing"
    );
    // The record is untouched by the refusal: no rejection, no extra attempt.
    let record = harness_record(&harness);
    let shot = record.shot("SH010").expect("SH010 recorded");
    assert_eq!(shot.attempts.len(), 1);
    assert!(shot.attempts[0].rejection.is_none());

    // Settling it first is exactly what the refusal asked for, and then the replacement works.
    let resumed = harness.resume_to_completion().await;
    assert_eq!(resumed.shot("SH010").unwrap().selected_attempt, Some(1));
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let replaced = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "now it can be replaced",
    )
    .await
    .expect("replacement runs once the attempt settled");
    assert_eq!(replaced.shot("SH010").unwrap().selected_attempt, Some(2));
    assert_eq!(harness.video_job_count(), 2);
}

#[tokio::test]
async fn a_conditioning_dependency_is_flagged_with_its_own_kind() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    // SH020's conditioning is declared as coming out of SH010's take rather than a pack plate.
    let plan = harness.edited_plan(|plan| {
        plan["shots"][1]["dependsOn"] = json!([
            { "shotId": "SH010", "kind": "conditioning", "note": "first frame is SH010's last frame" }
        ]);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after =
        film_harness::replace_take(&harness.transport, &resume_options, "SH010", "wrong door")
            .await
            .expect("replacement runs");
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.needs_review.len(), 1);
    assert_eq!(sh020.needs_review[0].dependency, "conditioning");
    assert!(
        sh020.needs_review[0]
            .reason
            .contains("first frame is SH010's last frame"),
        "{}",
        sh020.needs_review[0].reason
    );
    assert_eq!(
        sh020.attempts.len(),
        1,
        "the flagged shot was not re-rendered"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-22711 review: the windows the first pass left open
// ---------------------------------------------------------------------------------------------

/// A SECOND re-export must dispatch a third job, not adopt the FIRST export.
///
/// Every export the run ever dispatches renders the same timeline, which is the only key the export
/// route gives the harness. Excluding just the most recently superseded job leaves the original
/// export in the candidate set, and it is a completed `timeline_export` job — so the harness would
/// adopt it, record `status: completed, stale: false` with its pre-replacement asset, and the record
/// would claim the delivered MP4 is current while it is the sequence from before both replacements.
#[tokio::test]
async fn a_second_re_export_dispatches_a_new_job_instead_of_adopting_the_first() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let export_one = first.export.as_ref().expect("export").job_id.clone();

    let second = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "flat lighting",
    )
    .await
    .expect("first replacement runs");
    let export_two = second.export.as_ref().expect("export").job_id.clone();
    assert_ne!(export_two, export_one, "the first re-export ran a new job");

    let third = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH020",
        "the parcel is the wrong red",
    )
    .await
    .expect("second replacement runs");
    let export_three = third.export.as_ref().expect("export");
    assert_ne!(
        export_three.job_id,
        export_one,
        "the second re-export adopted the ORIGINAL export — its asset is the timeline from before \
         both replacements\n{}",
        summary(&third)
    );
    assert_ne!(export_three.job_id, export_two);
    assert_eq!(export_three.status, "completed");
    assert!(!export_three.stale);
    assert_eq!(
        harness.export_job_count(),
        3,
        "three exports were asked for, so three must have run\n{}",
        summary(&third)
    );
    // Both superseded exports are remembered, which is what keeps the exclusion cumulative.
    assert_eq!(
        third.superseded_export_job_ids,
        vec![export_one, export_two]
    );
    // And the MP4 the record points at really is the one rendered from the current takes.
    let timeline = third.timeline.as_ref().expect("timeline");
    for shot_id in ["SH010", "SH020"] {
        let selected = third
            .shot(shot_id)
            .and_then(|shot| shot.selected())
            .and_then(|attempt| attempt.take.as_ref())
            .expect("a selected take");
        let item = timeline
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .expect("on the timeline");
        assert_eq!(item.asset_id, selected.asset_id, "{shot_id}");
    }
}

/// `POST /timelines` always creates a NEW row, and the record only learns the timeline id after the
/// PUT — so a controller killed in that window leaves a timeline nothing names. The resume must
/// adopt it by the name it was created under, exactly as it adopts the project.
#[tokio::test]
async fn a_crash_after_the_timeline_was_created_adopts_it_instead_of_creating_a_second() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport =
        FaultTransport::new(harness.app.clone(), 1, FaultMode::After).on_post_route("/timelines");
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    assert!(transport.fired());

    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.timeline.is_none(),
        "the controller died before it could record the timeline"
    );
    let created = harness.timelines_for(&crashed).await;
    assert_eq!(created.len(), 1, "the API really created one: {created:#?}");

    let record = harness.resume_to_completion().await;
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let timelines = harness.timelines_for(&record).await;
    assert_eq!(
        timelines.len(),
        1,
        "the resume created a SECOND timeline instead of adopting the one it already made: \
         {timelines:#?}"
    );
    let timeline = record.timeline.as_ref().expect("timeline recorded");
    assert_eq!(
        timelines[0]["id"].as_str(),
        Some(timeline.timeline_id.as_str())
    );
    assert_eq!(timelines[0]["name"].as_str(), Some(timeline.name.as_str()));
    assert_eq!(timeline.items.len(), 1);
}

/// The memory budget is judged on the EVIDENCE, not on who was watching when it landed: a resume
/// that adopts an over-budget terminal attempt must stop new dispatch exactly as the controller that
/// watched it would have (E5 / AC3).
#[tokio::test]
async fn an_over_budget_peak_adopted_on_a_resume_stops_new_dispatch() {
    // 90% of the 128 GiB the fake worker reports is 115.2 GiB, over the fixture's 96 GB budget.
    let harness = Harness::start(
        true,
        vec![(
            "SH010",
            VideoBehavior::Complete {
                delay_secs: 0,
                peak_pct: 90.0,
            },
        )],
    )
    .await;
    // Die right after the video job POST: the API holds the job, the record does not know its id.
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
        .on_post_route("/api/v1/video/jobs");
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.shot("SH010").expect("SH010").attempts[0]
            .job_id
            .is_none(),
        "the controller never learned the job id, so the resume must reconcile it"
    );
    // Let the render finish before the resume, so the attempt is ADOPTED rather than polled.
    wait_for_settled_shot(&harness.app, &harness.script, "SH010").await;

    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(
        resumed.outcome,
        RunOutcome::StoppedMemoryLimit,
        "an adopted over-budget peak must stop the run\n{}",
        summary(&resumed)
    );
    let stop = resumed.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "memory_limit");
    assert!(
        !stop.resumable,
        "the budget is blown; another pass is not the fix"
    );
    let attempt = &resumed.shot("SH010").expect("SH010").attempts[0];
    assert_eq!(
        attempt.peak_memory_source.as_deref(),
        Some("metrics.peakMemoryBytes")
    );
    assert!(
        attempt
            .peak_memory_gb
            .is_some_and(|gb| (gb - 115.2).abs() < 0.01),
        "{:?}",
        attempt.peak_memory_gb
    );
    assert_eq!(
        resumed.shot("SH010").expect("SH010").outcome,
        ShotOutcome::Rendered,
        "the take that was produced is kept"
    );
    assert_eq!(
        resumed.shot("SH020").expect("SH020").outcome,
        ShotOutcome::NotDispatched,
        "nothing new goes out against a budget the evidence says was blown\n{}",
        summary(&resumed)
    );
    assert_eq!(
        harness.api_video_job_count().await,
        1,
        "the resume dispatched a second render anyway"
    );
}

/// sc-23402 review. A run record written by a PRE-STORY build carries no `resolvedModelId` and no
/// `partitionReason`; `#[serde(default)]` reads them back as `""`.
///
/// The reconcile/adopt paths cloned that empty string straight onto the take they imported, so a
/// phase-1 run directory resumed on this build recorded its adopted take with `model: ""` — losing
/// the only statement of which checkpoint produced the clip. The fallback is the shot's own
/// resolved partition (what the first controller would have written), and the attempt row is
/// backfilled so the record self-heals on the resume that touched it. The mixed fixture is the
/// fixture that can tell the fix apart from `plan.model.id`: SH010 resolves to `minimax_h3_ref`.
#[tokio::test]
async fn an_adopted_attempt_without_a_recorded_partition_falls_back_and_backfills() {
    let harness = Harness::start(true, vec![]).await;
    // Die right after the video job POST: the API holds the job, the record does not know its id,
    // so the resume ADOPTS it through `reconcile_shot` rather than polling one it dispatched.
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
        .on_post_route("/api/v1/video/jobs");
    let options = harness.options(
        harness.mixed_partition_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");

    // Rewrite the record the way a pre-story build wrote it: the two keys absent entirely.
    let path = harness.out_dir().join("run.json");
    let mut on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("record on disk"))
            .expect("record parses");
    for shot in on_disk["shots"].as_array_mut().expect("shots") {
        for attempt in shot["attempts"].as_array_mut().expect("attempts") {
            let attempt = attempt.as_object_mut().expect("attempt");
            attempt.remove("resolvedModelId");
            attempt.remove("partitionReason");
        }
    }
    std::fs::write(&path, serde_json::to_string_pretty(&on_disk).unwrap()).unwrap();
    let stripped = film_harness::read_run_record(&harness.out_dir()).expect("record re-reads");
    let attempt = &stripped.shot("SH010").expect("SH010").attempts[0];
    assert!(
        attempt.resolved_model_id.is_empty() && attempt.job_id.is_none(),
        "the fields really are absent and the job id was never recorded: {attempt:?}"
    );

    // Let the render settle so the resume adopts a COMPLETED job and imports its take.
    wait_for_settled_shot(&harness.app, &harness.script, "SH010").await;
    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("the resume adopts the in-flight job");
    let attempt = &resumed.shot("SH010").expect("SH010").attempts[0];
    let take = attempt.take.as_ref().unwrap_or_else(|| {
        panic!("the adopted attempt has no take\n{}", summary(&resumed));
    });
    // `take_from_result` prefers the asset recipe's own `model` and falls back to the string the
    // adopt path hands it, so this asserts the two agree — the fake worker's recipe carries the id.
    // The ATTEMPT ROW below is the assertion that pins the fallback: it is written from nothing but
    // the adopt path's value, and it is what a reader (and the next resume) reads.
    assert_eq!(
        take.model, "minimax_h3_ref",
        "the adopted take must name the partition that rendered it, not \"\""
    );
    assert_eq!(
        attempt.resolved_model_id, "minimax_h3_ref",
        "and the attempt row is backfilled"
    );
    assert!(
        attempt.partition_reason.contains("minimax_h3_ref"),
        "{}",
        attempt.partition_reason
    );
}

/// `replace-take` decides ONE shot's outcome. Closing the run through the whole-run classifier
/// overwrote a resumable stop with `attempts_exhausted` / `resumable: false`, which permanently
/// blocks the `resume` that was going to render the remaining shots.
#[tokio::test]
async fn replacing_a_take_leaves_a_canceled_runs_resumable_stop_in_place() {
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
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan,
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    let control = RunControl::new();
    harness.script.lock().running_hook = Some((
        "SH020".to_owned(),
        RunningHook::CancelControl(control.clone()),
    ));
    let canceled = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    assert!(
        harness.script.lock().running_hook.is_none(),
        "the fake worker observed SH020 running and consumed the cancellation hook"
    );
    assert_eq!(
        canceled.outcome,
        RunOutcome::Canceled,
        "{}",
        summary(&canceled)
    );
    assert!(canceled.is_resumable());
    assert!(canceled
        .shot("SH020")
        .expect("SH020")
        .selected_attempt
        .is_none());

    // The human replaces the ONE take the run does have. SH020 and SH030 still owe work.
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "flat lighting",
    )
    .await
    .expect("replacement runs");
    assert_eq!(
        after.shot("SH010").expect("SH010").selected_attempt,
        Some(2)
    );
    let stop = after.stop.as_ref().expect("the run's own stop is kept");
    assert_eq!(
        stop.reason,
        "canceled",
        "replacing one take must not re-classify the run\n{}",
        summary(&after)
    );
    assert!(
        stop.resumable,
        "the replacement made a resumable run terminal, so SH020/SH030 can never be rendered\n{}",
        summary(&after)
    );
    assert_eq!(after.outcome, RunOutcome::Canceled);
    assert!(after.is_resumable());

    // And the resume that stop promises really does finish the run, keeping the replacement.
    harness
        .script
        .lock()
        .behaviors
        .retain(|(id, _)| id != "SH020");
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        resumed.shot("SH010").expect("SH010").selected_attempt,
        Some(2),
        "the replacement take survived the resume"
    );
    for shot_id in ["SH020", "SH030"] {
        assert_eq!(
            resumed.shot(shot_id).expect("recorded").outcome,
            ShotOutcome::Rendered,
            "{shot_id}"
        );
    }
}

/// An attempt is recorded BEFORE its job is created — that is what makes the idempotency key work.
/// A controller that died in that window rendered nothing, so the next morning's resume must not
/// charge it the wall clock since: with `maxAttemptsPerShot: 1` that spends the shot's only attempt
/// on a job it dispatches and cancels on the first poll.
#[tokio::test]
async fn an_attempt_whose_job_post_never_landed_is_not_charged_wall_clock() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::Before)
        .on_post_route("/api/v1/video/jobs");
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, Some(&["SH010"]));
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    assert_eq!(
        harness.api_video_job_count().await,
        0,
        "the POST never reached the API"
    );
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    let attempt = &crashed.shot("SH010").expect("SH010").attempts[0];
    assert_eq!(attempt.status, "dispatching");
    assert!(attempt.job_id.is_none());

    // The operator comes back the next morning.
    let yesterday = sceneworks_core::time::format_unix_seconds(
        sceneworks_core::time::now_unix_seconds() - 86_400,
    );
    harness.edit_run_record(|record| {
        record["shots"][0]["attempts"][0]["startedAt"] = json!(yesterday);
    });

    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "an attempt that never dispatched was charged a night's wall clock\n{}",
        summary(&resumed)
    );
    let shot = resumed.shot("SH010").expect("SH010");
    assert_eq!(
        shot.attempts.len(),
        1,
        "the plan allows exactly one automatic attempt"
    );
    assert_eq!(
        shot.attempts[0].status,
        "completed",
        "the attempt timed out against a budget it had never spent\n{}",
        summary(&resumed)
    );
    assert_eq!(shot.selected_attempt, Some(1));
    assert_eq!(shot.outcome, ShotOutcome::Rendered);
    assert_eq!(harness.video_job_count(), 1);
}

/// `run` over a directory that already holds a record would mint a new run id over the old run's
/// takes, decisions and provenance — while `persist_record` keeps the plan/pack copies of the run it
/// just destroyed. `scripts/film-harness-smoke.sh` pins its `--out`, so a second invocation is
/// exactly this (E2).
#[tokio::test]
async fn run_refuses_a_directory_that_already_holds_a_run_record() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(first.outcome, RunOutcome::Completed, "{}", summary(&first));
    let dispatched = harness.video_job_count();

    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("a second run over the same --out is refused");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("already holds run") && message.contains("resume")),
        "{error}"
    );
    // The previous run's record is exactly as it was, and nothing was dispatched against it.
    let on_disk = harness_record(&harness);
    assert_eq!(on_disk.run_id, first.run_id);
    assert_eq!(selections(&on_disk), selections(&first));
    assert_eq!(harness.video_job_count(), dispatched);
    assert_eq!(harness.project_count().await, 1);
}

/// `cancel --out /typo/path` used to create the directory, print "cancel requested" and exit 0 while
/// the real render kept going. A cancel that reaches nothing must say so.
#[test]
fn cancel_refuses_a_directory_that_holds_no_run_record() {
    let temp = tempfile::tempdir().expect("temp dir");
    let missing = temp.path().join("typo").join("run");
    let error = film_harness::request_cancel(&missing).expect_err("there is no run there");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("no run record in")),
        "{error}"
    );
    assert!(
        !missing.exists(),
        "a mistyped --out must not be created on the way to a cancel nobody receives"
    );

    // With a record in it, the sentinel is written as before.
    let held = temp.path().join("held");
    std::fs::create_dir_all(&held).expect("run dir");
    std::fs::write(held.join("run.json"), "{}").expect("record");
    let sentinel = film_harness::request_cancel(&held).expect("a real run directory is cancelable");
    assert!(sentinel.exists());
}

/// Replacing the same upstream take twice is the same unread signal, not two.
#[tokio::test]
async fn replacing_the_same_take_twice_does_not_duplicate_a_review_flag() {
    let harness = Harness::start(true, fast(&["SH010", "SH020", "SH030"])).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    for reason in ["the parcel is the wrong red", "still the wrong red"] {
        film_harness::replace_take(&harness.transport, &resume_options, "SH020", reason)
            .await
            .expect("replacement runs");
    }
    let record = harness_record(&harness);
    let sh030 = record.shot("SH030").expect("SH030 recorded");
    assert_eq!(
        sh030.needs_review.len(),
        1,
        "one standing flag per (source shot, dependency), not one per replacement: {:#?}",
        sh030.needs_review
    );
    assert_eq!(sh030.needs_review[0].source_shot_id, "SH020");
    assert_eq!(sh030.needs_review[0].dependency, "continuity");
}

/// A human can reject a plate between two controllers, and the asset listing hides rejected and
/// trashed assets by default — so the adoption lookup has to ask for them. Otherwise the replay
/// cannot see the reference it already uploaded and imports a second copy, while the record claims
/// one import.
#[tokio::test]
async fn a_reference_rejected_between_controllers_is_still_adopted_not_imported_again() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport =
        FaultTransport::new(harness.app.clone(), 1, FaultMode::After).on_post_route("/assets");
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.references.is_empty(),
        "the upload that was answered is the one the record missed"
    );
    let project_id = crashed.project_id.clone().expect("project created");

    // The human rejects the uploaded plate before the resume.
    let (_, assets) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    let uploaded = assets
        .as_array()
        .into_iter()
        .flatten()
        .find(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
        .and_then(|asset| asset["id"].as_str())
        .expect("the interrupted upload landed")
        .to_owned();
    let (status, _) = request(
        harness.app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{uploaded}/status"),
        json!({ "rejected": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);

    let record = harness.resume_to_completion().await;
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.references.len(), 7);
    let (_, assets) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?includeRejected=true&includeTrashed=true"),
        Value::Null,
    )
    .await;
    let references: Vec<&Value> = assets
        .as_array()
        .into_iter()
        .flatten()
        .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
        .collect();
    assert_eq!(
        references.len(),
        7,
        "the rejected plate was invisible to the adoption lookup, so it was imported twice"
    );
    assert!(
        references
            .iter()
            .any(|asset| asset["id"].as_str() == Some(uploaded.as_str())),
        "the adopted asset is the one that was already uploaded"
    );
}

// -------------------------------------------------------------------------------------------
// Editable picture and continuous sound (sc-22712)
// -------------------------------------------------------------------------------------------
//
// What is proved HERE is the shape of the saved sequence and what the editing commands do to it:
// which track each clip lands on, what it is linked to, and how everything re-times. What the
// exported MP4 actually SOUNDS like is proved where the real ffmpeg runs, in
// `sceneworks_worker::media_jobs::timeline_audio_mix_tests` — an assertion about a timeline is not
// an assertion about audio, and neither one substitutes for the other.

/// Whether an ffmpeg the store can transcode an audio upload with is reachable.
///
/// Sound import goes through `ProjectStore::import_asset` -> `transcode_to_wav_pcm16`, and ffmpeg
/// is not on every lane. Soft-skipping is the posture the sibling store test
/// (`import_asset_admits_audio_and_normalizes_it_to_pcm16_wav`) already takes for exactly this
/// call; `SCENEWORKS_REQUIRE_FFMPEG` turns the skip into a failure on the lane that installs one.
pub(crate) fn ffmpeg_reachable() -> bool {
    // The same rule as `sceneworks_worker::video_jobs::tests::ffmpeg_reachable` (sc-22715): a
    // `SCENEWORKS_FFMPEG` that names a real file is reachable; one that is set but broken falls
    // through to the PATH probe rather than counting as unreachable on its own.
    let configured = std::env::var("SCENEWORKS_FFMPEG")
        .ok()
        .is_some_and(|path| !path.trim().is_empty() && Path::new(path.trim()).exists());
    let reachable = configured
        || std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
    assert!(
        reachable || std::env::var("SCENEWORKS_REQUIRE_FFMPEG").is_err(),
        "SCENEWORKS_REQUIRE_FFMPEG is set but no ffmpeg is reachable, so the film-harness sound \
         tests would have silently reported ok without importing a single clip"
    );
    reachable
}

/// Read the saved timeline document straight from the API — the thing the exporter reads and the
/// editor opens, rather than the run record's description of it.
pub(crate) async fn saved_timeline(
    app: &axum::Router,
    project_id: &str,
    timeline_id: &str,
) -> Value {
    let (status, timeline) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{timeline}");
    timeline
}

pub(crate) fn track_of<'a>(timeline: &'a Value, id: &str) -> &'a Value {
    timeline["tracks"]
        .as_array()
        .expect("tracks")
        .iter()
        .find(|track| track["id"] == json!(id))
        .unwrap_or_else(|| panic!("timeline has no track {id}: {timeline}"))
}

pub(crate) fn items_of<'a>(timeline: &'a Value, track_id: &str) -> &'a Vec<Value> {
    track_of(timeline, track_id)["items"]
        .as_array()
        .unwrap_or_else(|| panic!("track {track_id} has no items"))
}

pub(crate) fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-3
}

/// AC2, on the saved sequence: dialogue, ambience and music are three separately controlled buses,
/// and the beds are placed ONCE rather than per shot.
///
/// Runs on EVERY lane, ffmpeg or not (sc-22715): the fixture clips are canonical PCM-16 WAVs, which
/// the import route now stores without a transcode (`media_convert::is_canonical_pcm16_wav`), so
/// the timeline-document assertions here — the bus rollup, the dialogue offset, the beds placed
/// once, the reorder that keeps the sound — no longer hide behind an ffmpeg skip that reported
/// `ok` on the hosted macOS lane while asserting nothing. What still needs an ffmpeg is the
/// REAL mix, measured in `a_real_timeline_export_mixes_the_harness_four_track_sequence`.
#[tokio::test]
async fn the_assembled_sequence_carries_three_independently_controlled_sound_buses() {
    let harness = Harness::start(true, vec![]).await;
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline = record.timeline.as_ref().expect("timeline assembled");

    // Every clip this selection places is a project asset of type `audio`, tagged with its role —
    // and only those: `recipient_line` belongs to shots SH050/SH060, which this run left out.
    let imported: Vec<&str> = record.sound.iter().map(|clip| clip.role.as_str()).collect();
    assert_eq!(
        imported,
        // Pack order, filtered — not plan order.
        vec!["courier_line", "workshop_room_tone", "main_theme"],
        "{imported:?}"
    );
    for clip in &record.sound {
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets/{}", clip.asset_id),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "audio", "{asset}");
        assert_eq!(asset["extra"]["filmHarness"]["role"], clip.role);
        assert_eq!(asset["extra"]["filmHarness"]["kind"], "sound");
        let tags: Vec<&str> = asset["tags"]
            .as_array()
            .map(|tags| tags.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            tags.contains(&format!("role:{}", clip.role).as_str()),
            "{tags:?}"
        );
    }

    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let total = timeline.duration_seconds;
    assert!(
        close(total, 2.0 * 5.1667),
        "two shots back to back: {total}"
    );

    // Picture: two takes, contiguous, both muting their own audio by default.
    let picture = items_of(&saved, "track_main");
    assert_eq!(picture.len(), 2);
    assert!(close(picture[0]["timelineEnd"].as_f64().unwrap(), 5.1667));
    assert!(close(picture[1]["timelineStart"].as_f64().unwrap(), 5.1667));
    for item in picture {
        assert_eq!(
            item["generatedAudio"], "mute",
            "the fixture's run-level policy is mute: {item}"
        );
    }

    // Dialogue: the plan places a line against SH020 only, at its own offset INTO that shot.
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1, "{dialogue:#?}");
    assert_eq!(dialogue[0]["filmHarness"]["shotId"], "SH020");
    assert!(
        close(dialogue[0]["timelineStart"].as_f64().unwrap(), 5.1667 + 1.2),
        "the line sits 1.2s into SH020, not 1.2s into the sequence: {}",
        dialogue[0]
    );
    // The line is SPOKEN by the run now rather than read off a checked-in tone (sc-23404), so its
    // length is the length of what was said — derived from the same shape the fake synthesizes at,
    // never a number copied here by hand.
    let (_, spoken) =
        fake_speech_shape(Some("am_michael"), "Delivery. I'll leave it on the bench.");
    assert!(
        close(
            dialogue[0]["timelineEnd"].as_f64().unwrap(),
            5.1667 + 1.2 + spoken
        ),
        "the clip is the {spoken}s synthesized line: {}",
        dialogue[0]
    );
    assert_eq!(track_of(&saved, "track_dialogue")["gain"], 1.0);
    assert_eq!(track_of(&saved, "track_dialogue")["muted"], false);

    // Beds: ONE item each, spanning the whole sequence through the cut, at their own gains.
    for (track_id, gain, fade_in, fade_out) in [
        ("track_ambience", 0.35, 1.0, 1.5),
        ("track_music", 0.2, 2.0, 3.0),
    ] {
        let track = track_of(&saved, track_id);
        assert_eq!(track["kind"], "audio");
        assert_eq!(track["muted"], false);
        assert!(
            close(track["gain"].as_f64().unwrap(), gain),
            "{track_id} gain: {track}"
        );
        let items = items_of(&saved, track_id);
        assert_eq!(
            items.len(),
            1,
            "a bed is placed ONCE for the whole sequence — one item per shot is exactly the \
             per-shot restart this design exists to avoid: {items:#?}"
        );
        assert!(close(items[0]["timelineStart"].as_f64().unwrap(), 0.0));
        assert!(
            close(items[0]["timelineEnd"].as_f64().unwrap(), total),
            "the bed must reach the last frame: {} vs {total}",
            items[0]
        );
        assert!(
            close(items[0]["fadeInSeconds"].as_f64().unwrap(), fade_in)
                && close(items[0]["fadeOutSeconds"].as_f64().unwrap(), fade_out),
            "the plan's fades travel with the bed: {}",
            items[0]
        );
        // The source range follows the span, so the whole stretch that plays is asked for.
        assert!(close(
            items[0]["sourceOut"].as_f64().unwrap() - items[0]["sourceIn"].as_f64().unwrap(),
            total
        ));
    }

    // Sound never extends the picture: the timeline's own recomputed duration is the picture's.
    assert!(
        close(saved["duration"].as_f64().unwrap(), total),
        "the store recomputes duration across every track; sound must not stretch it: {}",
        saved["duration"]
    );

    // The run record says the same thing, so run.json alone explains the mix.
    assert_eq!(
        timeline.generated_audio_default,
        sceneworks_core::film_plan::GeneratedAudio::Mute
    );
    let buses: Vec<(&str, f64, bool, usize)> = timeline
        .tracks
        .iter()
        .filter(|track| track.kind == "audio")
        .map(|track| {
            (
                track.role.as_str(),
                track.gain,
                track.muted,
                track.items.len(),
            )
        })
        .collect();
    assert_eq!(
        buses,
        vec![
            ("dialogue", 1.0, false, 1),
            ("ambience", 0.35, false, 1),
            ("music", 0.2, false, 1),
        ],
        "{buses:?}"
    );
    for item in &timeline.items {
        assert_eq!(
            item.generated_audio,
            Some(sceneworks_core::film_plan::GeneratedAudio::Mute),
            "every picture item records the policy the export obeyed: {item:?}"
        );
    }

    // And the sound survives an edit to the picture: put SH020 first, and its line must travel
    // with it while both beds re-span the sequence. This is the half of AC1 that a picture-only
    // assertion misses — a reorder that leaves a line under the wrong shot has kept the shot/asset
    // links and still broken the film.
    film_harness::edit_timeline(
        &harness.transport,
        &film_harness::EditOptions {
            run_record_path: harness.temp_dir.path().join("run-out/run.json"),
            export: false,
            poll_interval: Duration::from_millis(250),
        },
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");
    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let total = saved["duration"].as_f64().expect("duration");
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1);
    assert!(
        close(dialogue[0]["timelineStart"].as_f64().unwrap(), 1.2),
        "SH020 now starts at 0, so its line sits at its own 1.2s offset into it: {}",
        dialogue[0]
    );
    for track_id in ["track_ambience", "track_music"] {
        let items = items_of(&saved, track_id);
        assert_eq!(items.len(), 1, "{track_id} is still ONE continuous bed");
        assert!(close(items[0]["timelineStart"].as_f64().unwrap(), 0.0));
        assert!(
            close(items[0]["timelineEnd"].as_f64().unwrap(), total),
            "{track_id} must still reach the last frame after the edit: {}",
            items[0]
        );
    }
}

/// AC3's policy half: a shot's own setting wins over the run's, and both are recorded.
///
/// Needs no sound files, so it runs on every lane. That the policy is OBEYED — that `mute` really
/// keeps a generated line out of the mix and `include` really brings it in — is measured against a
/// real export in `generated_audio_doubles_with_dialogue_only_when_explicitly_included`.
#[tokio::test]
async fn a_shots_generated_audio_policy_overrides_the_runs_and_both_are_recorded() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        plan["sound"]["generatedAudio"] = json!("mute");
        plan["shots"][1]["generatedAudio"] = json!("include");
    });
    let options = harness.options(
        plan,
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let picture = items_of(&saved, "track_main");
    assert_eq!(
        picture[0]["generatedAudio"], "mute",
        "SH010 inherits the run"
    );
    assert_eq!(
        picture[1]["generatedAudio"], "include",
        "SH020 declared its own"
    );

    // The resolved policy is on the shot record too, so run.json says what the export obeyed
    // without anyone re-deriving it from the plan.
    let resolved: Vec<(&str, &str)> = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::Rendered)
        .map(|shot| {
            (
                shot.shot_id.as_str(),
                shot.intended.generated_audio.as_timeline_str(),
            )
        })
        .collect();
    assert_eq!(
        resolved,
        vec![("SH010", "mute"), ("SH020", "include")],
        "{resolved:?}"
    );
}

/// AC1: trim, reorder and replace-a-take, each keeping the shot -> asset links and re-timing the
/// sequence, and each persisted in both the project timeline and the run record.
#[tokio::test]
async fn trimming_reordering_and_replacing_a_take_keep_links_and_retime_the_sequence() {
    let harness = Harness::start(true, vec![]).await;
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let original: Vec<(String, String)> = record
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .map(|item| {
            (
                item.shot_id.clone().expect("picture items carry a shot"),
                item.asset_id.clone(),
            )
        })
        .collect();

    let edit_options = film_harness::EditOptions {
        run_record_path: harness.temp_dir.path().join("run-out/run.json"),
        export: false,
        poll_interval: Duration::from_millis(250),
    };

    // 1. TRIM. SH010 keeps only 1.0..3.0 of its take, and SH020 slides up to meet it.
    let trimmed = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(1.0),
            source_out: Some(3.0),
        },
    )
    .await
    .expect("trim applies");
    let items = &trimmed.timeline.as_ref().expect("timeline").items;
    assert!(close(items[0].source_in, 1.0) && close(items[0].source_out, 3.0));
    assert!(close(items[0].timeline_start, 0.0) && close(items[0].timeline_end, 2.0));
    assert!(
        close(items[1].timeline_start, 2.0) && close(items[1].timeline_end, 2.0 + 5.1667),
        "the trim must RIPPLE — a cut does not leave a hole: {items:#?}"
    );
    assert_eq!(
        items
            .iter()
            .map(|item| (item.shot_id.clone().unwrap(), item.asset_id.clone()))
            .collect::<Vec<_>>(),
        original,
        "a trim changes timing, never which take a shot points at"
    );

    // 2. REORDER. The takes swap places and keep their own lengths.
    let reordered = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");
    let items = &reordered.timeline.as_ref().expect("timeline").items;
    assert_eq!(
        items
            .iter()
            .map(|item| item.shot_id.clone().unwrap())
            .collect::<Vec<_>>(),
        vec!["SH020", "SH010"]
    );
    assert_eq!(
        items[0].asset_id, original[1].1,
        "SH020 keeps its own take at the head"
    );
    assert_eq!(items[1].asset_id, original[0].1);
    assert!(close(items[0].timeline_start, 0.0) && close(items[0].timeline_end, 5.1667));
    assert!(
        close(items[1].timeline_start, 5.1667) && close(items[1].timeline_end, 5.1667 + 2.0),
        "SH010 is still the trimmed 2s: {items:#?}"
    );

    // 3. REPLACE THE TAKE. Point SH010 at a different asset in the project; the sequence re-times
    //    around the replacement's own length and the version history remembers what it was.
    let replacement = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("plate imported")
        .asset_id
        .clone();
    let replaced = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::SwapTake {
            shot_id: "SH010".to_owned(),
            asset_id: replacement.clone(),
        },
    )
    .await
    .expect("replacement applies");
    let items = &replaced.timeline.as_ref().expect("timeline").items;
    let swapped = items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is still in the sequence");
    assert_eq!(swapped.asset_id, replacement);
    assert_ne!(swapped.asset_id, original[0].1);

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let picture = items_of(&saved, "track_main");
    let swapped_item = picture
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!("SH010"))
        .expect("SH010 on the saved picture track");
    assert_eq!(swapped_item["assetId"], replacement);
    assert_eq!(swapped_item["currentVersionAssetId"], replacement);
    let history: Vec<&str> = swapped_item["versionHistory"]
        .as_array()
        .expect("version history")
        .iter()
        .filter_map(|entry| entry["source"].as_str())
        .collect();
    assert_eq!(
        history,
        vec!["original", "replacement"],
        "the take that was there is still addressable: {swapped_item}"
    );
    assert!(
        swapped_item["versionAssetIds"]
            .as_array()
            .expect("version asset ids")
            .iter()
            .any(|value| value == &json!(replacement)),
        "{swapped_item}"
    );
    // Contiguity survived all three edits.
    let mut cursor = 0.0;
    for item in picture {
        assert!(
            close(item["timelineStart"].as_f64().unwrap(), cursor),
            "picture items must abut: {item} expected start {cursor}"
        );
        cursor = item["timelineEnd"].as_f64().unwrap();
    }

    // Every edit is in the run record on disk, oldest first, with the duration it produced.
    let on_disk = harness.run_record();
    let edits: Vec<&str> = on_disk["timeline"]["edits"]
        .as_array()
        .expect("edits recorded")
        .iter()
        .filter_map(|edit| edit["kind"].as_str())
        .collect();
    assert_eq!(edits, vec!["trim", "reorder", "swap_take"], "{on_disk}");
    assert!(on_disk["timeline"]["edits"][2]["detail"]
        .as_str()
        .unwrap_or_default()
        .contains(&replacement));

    // The edits reach the RUN RECORD's own shape, not only the timeline block (sc-22712 on
    // sc-22711's schema 2). Three things have to follow an edit, and none of them did before the
    // two stories were merged:

    // 1. The decision log carries every edit, in order, with the shot the edit named.
    let decisions: Vec<(&str, Option<&str>)> = on_disk["decisions"]
        .as_array()
        .expect("decisions recorded")
        .iter()
        .filter_map(|decision| {
            let action = decision["action"].as_str()?;
            ["trim", "reorder", "swap_take"]
                .contains(&action)
                .then(|| (action, decision["shotId"].as_str()))
        })
        .collect();
    assert_eq!(
        decisions,
        vec![
            ("trim", Some("SH010")),
            ("reorder", None),
            ("swap_take", Some("SH010")),
        ],
        "every edit is a human decision about the run: {on_disk}"
    );

    // 2. The export these edits ran against is FLAGGED stale — the sequence moved under the MP4,
    //    and none of these edits passed `--export`. It is never silently re-rendered.
    assert_eq!(
        on_disk["export"]["stale"],
        json!(true),
        "an edited sequence leaves the existing export stale: {on_disk}"
    );

    // 3. The asset swapped in here is a reference PLATE, not a take this run rendered, so the saved
    //    cut changes while selectedAttempt continues to record the last generation selection. (The other
    //    branch — a swap onto an asset that IS one of the shot's takes — moves the selection, and
    //    is proved in `swapping_onto_an_existing_take_moves_the_shots_selected_attempt`.)
    let sh010 = on_disk["shots"]
        .as_array()
        .expect("shots recorded")
        .iter()
        .find(|shot| shot["shotId"] == json!("SH010"))
        .expect("SH010 recorded");
    assert_eq!(
        sh010["selectedAttempt"],
        json!(1),
        "a foreign cut asset does not invent or erase generation provenance: {sh010}"
    );

    // A reorder that does not name the whole sequence is refused rather than silently dropping a
    // shot — the one way a "reorder" could quietly become a delete.
    let error = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH010".to_owned()],
        },
    )
    .await
    .expect_err("a partial order is refused");
    let message = error.to_string();
    assert!(
        message.contains("SH020") && message.contains("exactly once"),
        "{message}"
    );
}

/// AC1, bounded: a trim is measured against the take it cuts, not taken on faith.
///
/// An out point past the end of the media used to be accepted in silence. `relayout_timeline` then
/// wrote a `timelineEnd` longer than the file, `render_item_segment` returned the DECLARED duration
/// while `-t` gave ffmpeg a short segment, and the exported picture came out shorter than the saved
/// sequence — sound drifting against picture, and a duration in the render sidecar that no file
/// has. `SwapTake` always measured its asset first; `Trim` now does the same.
#[tokio::test]
async fn a_trim_past_the_end_of_the_take_is_clamped_to_the_takes_real_length() {
    let harness = Harness::start(true, vec![]).await;
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
    let project_id = record.project_id.clone().expect("project created");
    let take_asset = record
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is in the sequence")
        .asset_id
        .clone();
    let (status, asset) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{take_asset}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
    let take_seconds = asset["file"]["duration"]
        .as_f64()
        .expect("the imported take carries a measured duration");

    let edit_options = film_harness::EditOptions {
        run_record_path: harness.temp_dir.path().join("run-out/run.json"),
        export: false,
        poll_interval: Duration::from_millis(250),
    };
    let trimmed = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(0.5),
            // Far past the end of a take this caller never measured.
            source_out: Some(take_seconds * 10.0),
        },
    )
    .await
    .expect("a trim past the end of the take is clamped, not refused");
    let trimmed_item = trimmed
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is still in the sequence")
        .clone();
    assert!(
        close(trimmed_item.source_out, take_seconds),
        "the out point must be clamped to the take's real {take_seconds}s, got {}",
        trimmed_item.source_out
    );
    assert!(
        close(
            trimmed_item.timeline_end - trimmed_item.timeline_start,
            take_seconds - 0.5
        ),
        "and the slot the sequence gives the shot must be the media that exists: {trimmed_item:#?}"
    );

    // The saved timeline says the same thing — this is what the exporter reads.
    let saved = saved_timeline(
        &harness.app,
        &project_id,
        &record.timeline.as_ref().expect("timeline").timeline_id,
    )
    .await;
    let saved_item = items_of(&saved, "track_main")
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!("SH010"))
        .expect("SH010 on the saved picture track")
        .clone();
    assert!(
        close(saved_item["sourceOut"].as_f64().unwrap(), take_seconds),
        "{saved_item}"
    );

    // A range that lands entirely past the end is an error rather than a clamp, and it says how
    // long the take actually is so the caller can pick a real number.
    let error = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(take_seconds + 5.0),
            source_out: Some(take_seconds + 9.0),
        },
    )
    .await
    .expect_err("an in point past the end of the take has no usable range");
    let message = error.to_string();
    assert!(
        message.contains(&format!("{take_seconds:.3}")),
        "the refusal must name the take's real length: {message}"
    );
}

/// A re-layout is not a licence to delete the editor's own clips (sc-22712 review).
///
/// The harness re-places what IT placed — a line follows its shot, a bed re-spans the sequence —
/// and a harness-owned clip with nowhere left to go is dropped because the plan can put it back.
/// A clip the harness did not place is none of its business: it is clamped into the sequence and
/// kept. The shared drop used to apply to both, so any editor clip sitting within 40 ms of the end
/// was destroyed by the next edit, with no diagnostic and no undo.
#[tokio::test]
async fn a_relayout_keeps_an_editor_placed_clip_the_harness_never_put_there() {
    let harness = Harness::start(true, vec![]).await;
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let total = saved["duration"].as_f64().expect("duration");
    let asset_id = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("plate imported")
        .asset_id
        .clone();

    // A clip the editor placed: no `filmHarness` block, sitting right at the end of the sequence
    // and overhanging it.
    let mut edited = saved.clone();
    edited["tracks"]
        .as_array_mut()
        .expect("tracks")
        .iter_mut()
        .find(|track| track["id"] == json!("track_dialogue"))
        .expect("the editor's dialogue lane")["items"]
        .as_array_mut()
        .expect("items")
        .push(json!({
            "id": "editor_clip",
            "trackId": "track_dialogue",
            "assetId": asset_id,
            "type": "audio",
            "displayName": "an editor's own clip",
            "sourceIn": 0.0,
            "sourceOut": 1.0,
            "timelineStart": total - 0.02,
            "timelineEnd": total + 0.5,
            "speed": 1.0,
            "fit": "fit",
            "volume": 1.0,
            "fadeInSeconds": 0.0,
            "fadeOutSeconds": 0.0,
        }));
    let (status, body) = request(
        harness.app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": edited }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");

    // A reorder does not change the sequence's length, so nothing about this clip is stale.
    film_harness::edit_timeline(
        &harness.transport,
        &film_harness::EditOptions {
            run_record_path: harness.temp_dir.path().join("run-out/run.json"),
            export: false,
            poll_interval: Duration::from_millis(250),
        },
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let kept = items_of(&saved, "track_dialogue")
        .iter()
        .find(|item| item["id"] == json!("editor_clip"))
        .unwrap_or_else(|| {
            panic!(
                "the harness deleted a clip it never placed. A re-layout may re-place the \
                 harness's own dialogue, ambience and music items; the editor's belong to the \
                 editor: {saved}"
            )
        })
        .clone();
    assert!(
        close(kept["timelineStart"].as_f64().unwrap(), total - 0.02),
        "an unowned clip keeps where the editor put it: {kept}"
    );
    assert!(
        close(kept["timelineEnd"].as_f64().unwrap(), total),
        "clamped to the sequence, never stretching it: {kept} against {total}"
    );
}

/// The checked-in fixture clips are exactly what the generator writes, the same guarantee
/// `checked_in_fixture_plates_match_the_generator_byte_for_byte` gives the plates.
#[test]
fn checked_in_fixture_sound_matches_the_generator_byte_for_byte() {
    for (role, seconds, hz, amplitude) in film_harness::FIXTURE_SOUNDS {
        let path = Path::new(FIXTURE_DIR)
            .join("sound")
            .join(format!("{role}.wav"));
        let checked_in = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("{} is missing ({error})", path.display());
        });
        let generated = film_harness::fixture_sound_wav(*seconds, *hz, *amplitude);
        assert_eq!(
            checked_in.len(),
            generated.len(),
            "{role}: checked-in clip is a different length from the generator's"
        );
        assert!(
            checked_in == generated,
            "{role}: the checked-in clip no longer matches `fixture_sound_wav`. Regenerate it \
             with `film-harness fixture-sound --out config/film-harness/courier-workshop/sound`."
        );
    }
    // And the generator and the shipped pack agree about which roles are which (sc-23404): every
    // BED is a checked-in file the loop above just verified, every DIALOGUE role is a line the run
    // speaks, and no dialogue tone is left behind on disk pretending to be speech.
    let text = std::fs::read_to_string(Path::new(FIXTURE_DIR).join("references.jsonc"))
        .expect("fixture pack");
    let pack = sceneworks_core::film_plan::parse_reference_pack(&text).expect("pack parses");
    for entry in &pack.sound {
        if entry.kind == "dialogue" {
            assert!(
                entry.is_synthesized(),
                "{}: the fixture's dialogue is spoken by the run, not a checked-in tone",
                entry.role
            );
        } else {
            assert!(!entry.is_synthesized(), "{}", entry.role);
            assert!(
                film_harness::FIXTURE_SOUNDS
                    .iter()
                    .any(|(role, ..)| *role == entry.role),
                "{}: a bed the generator does not write",
                entry.role
            );
        }
    }
    let spoken = pack
        .sound
        .iter()
        .filter(|entry| entry.is_synthesized())
        .count();
    assert_eq!(spoken, 3, "the fixture places three lines");
}

/// The seam between the two ways a shot's take can change (sc-22711 `replace-take` re-renders,
/// sc-22712 `swap-take` re-cuts): swapping the sequence back onto a take the run ALREADY rendered
/// must move the shot's `selectedAttempt` with it.
///
/// The generation side owns `selectedAttempt` and the edit side owns the timeline item, and before
/// the two stories were merged neither told the other anything. A selection left behind is not
/// cosmetic: `status` reports the wrong take, and a subsequent `replace-take` rejects the attempt
/// the sequence no longer shows while leaving the one it does show in the cut.
#[tokio::test]
async fn swapping_onto_an_existing_take_moves_the_shots_selected_attempt() {
    let harness = Harness::start(true, vec![]).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let first_take = record
        .shot("SH010")
        .expect("SH010 recorded")
        .attempts
        .last()
        .and_then(|attempt| attempt.take.as_ref())
        .expect("SH010 rendered")
        .asset_id
        .clone();

    // Re-render SH010, which makes attempt 2 the selected one and leaves attempt 1's take in the
    // project, still addressable.
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "prefer the first one after all",
    )
    .await
    .expect("replacement runs");
    assert_eq!(
        after
            .shot("SH010")
            .expect("SH010 recorded")
            .selected_attempt,
        Some(2),
        "the re-render is what the shot now carries"
    );

    // Now change our mind on the EDIT side: put attempt 1's take back into the cut.
    let swapped = film_harness::edit_timeline(
        &harness.transport,
        &film_harness::EditOptions {
            run_record_path: harness.temp_dir.path().join("run-out/run.json"),
            export: false,
            poll_interval: Duration::from_millis(250),
        },
        film_harness::TimelineEdit::SwapTake {
            shot_id: "SH010".to_owned(),
            asset_id: first_take.clone(),
        },
    )
    .await
    .expect("swap applies");

    assert_eq!(
        swapped
            .shot("SH010")
            .expect("SH010 recorded")
            .selected_attempt,
        Some(1),
        "the selection follows the sequence back onto attempt 1"
    );
    let item = swapped
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is in the sequence");
    assert_eq!(
        item.asset_id, first_take,
        "the record and the sequence name the same take"
    );
}

// ----------------------------------------------------------------------------------------------
// Local planner (sc-22713): brief -> plan -> compiled requests -> dispatch.
//
// The planner drives the SHIPPED LLM seam — `POST /api/v1/prompts/refine`, the `prompt_refine`
// job, the worker's native TextLlm — so these tests create the jobs through the real route and
// script only the model's answer (`run_fake_refine_job`). Nothing here loads weights or touches a
// GPU: what is under test is the request the planner composes, the strictness of the parse, the
// bound on the repair loop, and the conformance of the compiled requests to the INSTALLED manifest.
// ----------------------------------------------------------------------------------------------

pub(crate) const BRIEF_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop/brief.jsonc"
);

/// The six beats the checked-in brief requires, in order.
const BRIEF_BEATS: &[&str] = &[
    "arrival",
    "approach",
    "handover",
    "departure",
    "discovery",
    "opening",
];

/// The roles the checked-in brief declares each beat must show (`requiredBeats[].requiredRoles`).
/// A scripted draft binds them because the validator checks them — the same demand the real
/// planner is held to (sc-22713).
fn beat_roles(beat_id: &str) -> Vec<&'static str> {
    let mut roles = match beat_id {
        "arrival" => vec!["courier", "red_parcel", "workshop_location"],
        "discovery" => vec!["recipient", "red_parcel", "workbench_table"],
        "opening" => vec!["recipient", "red_parcel"],
        // approach / handover / departure, and anything a test invents.
        _ => vec!["courier", "red_parcel", "workbench_table"],
    };
    roles.push("house_style");
    roles
}

/// One shot of a scripted planner draft, on the H3 envelope (24 fps, 576x320, 5.1667s).
fn draft_shot(id: &str, beat_id: &str) -> Value {
    json!({
        "id": id,
        "beatId": beat_id,
        "beat": format!("beat {beat_id}"),
        "framing": "wide static, eye level",
        "prompt": format!("A cluttered woodworking workshop in warm late-afternoon light; {beat_id}."),
        "targetDurationSeconds": 5.1667,
        "startState": "the workshop before this shot",
        "endState": "the workshop after this shot",
        "audio": "Room tone, distant birds. No music.",
        "conditioning": { "mode": "text_to_video" },
        "seed": 22713,
        "continuityRoles": beat_roles(beat_id)
    })
}

/// The same shot, written the way the sc-23405 envelope asks for it: `reference_to_video`, binding
/// the approved roles the beat is about.
///
/// `house_style` stays in `continuityRoles` only. It is a style reference, and MiniMax-H3's Ref2VA
/// treats every bound image as a subject to depict — binding a look as a subject asks for a shot OF
/// the look — which is also why the shipped `plan.v2.jsonc` binds it nowhere.
fn reference_draft_shot(id: &str, beat_id: &str) -> Value {
    let mut shot = draft_shot(id, beat_id);
    let roles: Vec<&str> = beat_roles(beat_id)
        .into_iter()
        .filter(|role| *role != "house_style")
        .collect();
    shot["conditioning"] = json!({ "mode": "reference_to_video", "referenceRoles": roles });
    shot
}

/// A well-formed draft covering every beat of the checked-in brief with every shot bound to the
/// approved roles it depicts — what the planner is asked for when the pack has references.
pub(crate) fn reference_draft() -> Value {
    json!({
        "shots": BRIEF_BEATS
            .iter()
            .enumerate()
            .map(|(index, beat)| reference_draft_shot(&format!("SH{:03}0", index + 1), beat))
            .collect::<Vec<_>>()
    })
}

/// A well-formed draft covering every beat of the checked-in brief.
pub(crate) fn full_draft() -> Value {
    json!({
        "shots": BRIEF_BEATS
            .iter()
            .enumerate()
            .map(|(index, beat)| draft_shot(&format!("SH{:03}0", index + 1), beat))
            .collect::<Vec<_>>()
    })
}

pub(crate) fn draft_text(draft: &Value) -> String {
    serde_json::to_string_pretty(draft).expect("draft serializes")
}

pub(crate) fn planner_options(harness: &Harness, out: &str) -> film_planner::PlannerOptions {
    film_planner::PlannerOptions {
        brief_path: PathBuf::from(BRIEF_FIXTURE),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        out_dir: harness.temp_dir.path().join(out),
        max_repair_rounds: 2,
        refine_prompts: false,
        prompt_guide_path: None,
        require_installed: false,
        require_local_planner: true,
        send_reference_pixels: false,
        // Empty: the in-process transport has no URL. The local-only rule is exercised as a unit
        // test in `film_planner` and end to end below.
        api_url: String::new(),
        force: false,
        poll_interval: Duration::from_millis(50),
        job_timeout: Duration::from_secs(30),
    }
}

pub(crate) fn planner_llm(harness: &Harness) -> film_planner::SceneWorksLlm<'_> {
    film_planner::SceneWorksLlm::new(
        &harness.transport,
        Duration::from_millis(50),
        Duration::from_secs(30),
    )
}

pub(crate) fn set_plan_replies(harness: &Harness, replies: Vec<String>) {
    let mut script = harness.script.lock();
    script.plan_replies = replies;
    script.plan_calls = 0;
}

pub(crate) fn refine_job_payloads(harness: &Harness, plan_task_only: bool) -> Vec<Value> {
    harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(job_type, _, payload)| {
            job_type == "prompt_refine" && (!plan_task_only || payload["task"] == "film_plan")
        })
        .map(|(_, _, payload)| payload.clone())
        .collect()
}

pub(crate) fn findings_of(error: HarnessError) -> Vec<String> {
    match error {
        HarnessError::Validation(findings) | HarnessError::PlannerValidation { findings, .. } => {
            findings.iter().map(ToString::to_string).collect()
        }
        HarnessError::PlannerExecutionFailure { source, .. } => findings_of(*source),
        other => panic!("expected a validation refusal, got {other}"),
    }
}

#[tokio::test]
async fn the_brief_produces_a_plan_the_existing_controller_accepts_unchanged() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    harness.script.lock().refine_template = Some(
        "integrated_multimodal_description: {prompt}\noverall_soundscape: room tone".to_owned(),
    );
    let mut options = planner_options(&harness, "planned");
    options.refine_prompts = true;

    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the planner produces a plan");

    // A complete plan: every beat, every narrative field, timing, camera, reference bindings and
    // the intended start/end state of each shot.
    assert_eq!(artifacts.repair_rounds, 0);
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    assert_eq!(artifacts.plan.id, "courier-workshop-planned");
    assert_eq!(artifacts.plan.model.id, "minimax_h3");
    assert_eq!(artifacts.plan.limits.max_memory_gb, 96.0);
    for shot in &artifacts.plan.shots {
        assert!(!shot.beat.trim().is_empty(), "{shot:?}");
        assert!(!shot.framing.trim().is_empty(), "{shot:?}");
        assert!(!shot.start_state.trim().is_empty(), "{shot:?}");
        assert!(!shot.end_state.trim().is_empty(), "{shot:?}");
        assert!(shot.target_duration_seconds > 0.0, "{shot:?}");
        assert!(
            !shot.continuity_roles.is_empty(),
            "every shot binds canonical reference roles: {shot:?}"
        );
    }

    // The plan is a file, and the SAME validator the hand-authored path uses accepts it unchanged.
    assert_eq!(artifacts.plan_path, options.out_dir.join("plan.json"));
    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        // The harness's own copy, never the checked-in directory: a run writes its synthesized
        // clips beside the pack (sc-23404).
        reference_pack_path: harness.fixture_pack(),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("planned-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let (plan, _) = film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the generated plan validates against the live catalog");
    assert_eq!(plan.shots.len(), BRIEF_BEATS.len());

    // The compiled requests are a second versioned document, on the installed model's envelope.
    let compiled: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.compiled_path).expect("compiled.json written"),
    )
    .expect("compiled.json parses");
    assert_eq!(
        compiled["schemaVersion"],
        sceneworks_core::film_compile::COMPILED_PLAN_SCHEMA_VERSION,
        "sc-23402 bumped this to 2 (`model` is the RESOLVED partition id) and sc-23406 to 3 (a \
         request carries its LoRAs and its step count)"
    );
    assert_eq!(compiled["planId"], "courier-workshop-planned");
    assert_eq!(compiled["model"]["fps"], 24);
    let requests = compiled["requests"].as_array().expect("requests");
    assert_eq!(requests.len(), BRIEF_BEATS.len());
    for request in requests {
        assert_eq!(request["fps"], 24);
        assert_eq!(request["width"], 576);
        assert_eq!(request["height"], 320);
        assert_eq!(request["durationSeconds"], 5.1667);
        assert_eq!(request["mode"], "text_to_video");
        assert_eq!(request["promptSource"], "refined");
        // The refinement produced the dispatched prompt — CONTAINED rather than leading it, because
        // since sc-24025 the compiler's identity text leads every shot that names a continuity
        // role it does not bind, and these shots bind nothing at all. What must still hold is that
        // the refiner's own words survive and that everything ahead of them is recorded inserted
        // text rather than something nobody wrote.
        let prompt = request["prompt"].as_str().unwrap();
        let refined_at = prompt
            .find("integrated_multimodal_description:")
            .unwrap_or_else(|| {
                panic!("the H3 refinement produced the dispatched prompt: {request}")
            });
        let leading: String = request["insertedText"]
            .as_array()
            .expect("insertedText is recorded")
            .iter()
            .filter(|piece| piece["kind"] == "continuity_description")
            .map(|piece| piece["text"].as_str().unwrap_or_default().to_owned())
            .collect();
        assert!(
            !leading.is_empty(),
            "these shots lock continuity roles: {request}"
        );
        assert_eq!(
            prompt[..refined_at].trim(),
            leading.trim(),
            "only the compiler's own identity text precedes the refined prompt: {request}"
        );
        assert!(request.get("negativePrompt").is_none(), "{request}");
        assert!(
            request["referenceRoles"].as_array().unwrap().is_empty(),
            "H3 declares maxReferenceAssets 0: {request}"
        );
    }

    // What the planning cost is PERSISTED beside what it produced (sc-22715): every LLM job, its
    // wall-clock, the peak the metrics route reported, the rounds, and the budget it ran under.
    let planner = compiled["planner"]
        .as_object()
        .expect("compiled.json carries a planner cost block");
    assert_eq!(
        planner["jobIds"].as_array().map(Vec::len),
        Some(1 + BRIEF_BEATS.len()),
        "one plan draft plus one rewrite per shot: {planner:?}"
    );
    assert_eq!(planner["repairRounds"], 0);
    assert!(
        planner["elapsedSeconds"].as_f64().unwrap_or(0.0) > 0.0,
        "{planner:?}"
    );
    assert_eq!(
        planner["peakMemoryBytes"].as_u64(),
        Some(FAKE_REFINE_PEAK_BYTES),
        "the peak is the one the metrics route carried: {planner:?}"
    );
    assert_eq!(planner["plannerMaxMemoryGb"], 24.0);
    assert_eq!(
        artifacts
            .compiled
            .planner
            .as_ref()
            .map(|cost| cost.job_ids.len()),
        Some(1 + BRIEF_BEATS.len())
    );

    // The planner drove the shipped seam: one film_plan job plus one rewrite job per shot, all of
    // them `prompt_refine`, with the target model forwarded (which is what selects the H3 asset).
    let refine_jobs = refine_job_payloads(&harness, false);
    assert_eq!(refine_jobs.len(), 1 + BRIEF_BEATS.len(), "{refine_jobs:?}");
    assert_eq!(refine_jobs[0]["task"], "film_plan");
    assert_eq!(refine_jobs[0]["modelId"], "minimax_h3");
    assert_eq!(refine_jobs[0]["workflow"], "video");
    for payload in &refine_jobs[1..] {
        assert!(payload.get("task").is_none(), "{payload}");
        assert_eq!(payload["modelId"], "minimax_h3");
    }
    // The planning request carried the brief's beats and only APPROVED roles.
    let request = refine_jobs[0]["prompt"].as_str().unwrap();
    for beat in BRIEF_BEATS {
        assert!(request.contains(beat), "{beat} missing from the request");
    }
    assert!(request.contains("workshop_plate (plate)"), "{request}");
}

/// sc-23402 review, E1, as sc-23405 leaves it. A host whose catalog serves NO reference partition
/// plans a text-only film and is never asked for the reference weights.
///
/// sc-23402's first cut resolved the reference partition AND ran the entry-level gate on it for
/// every `plan`, so with the CLI's default `--require-installed` a brief that produces a text-only
/// film refused with "minimax_h3_ref tier q4 is not installed on this host" — demanding an 18.78 GB
/// download the resulting plan could never load. The envelope now DOES widen to that partition when
/// one is on offer (sc-23405), so the property is kept where it belongs: the planner offers
/// reference conditioning only when the catalog serves the partition and the pack can fill it, and
/// where it offers none the film, the envelope and the requests are exactly the phase-1 ones.
#[tokio::test]
async fn planning_without_a_reference_partition_in_the_catalog_stays_on_the_base_checkpoint() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    // The base checkpoint installed, and no `minimax_h3_ref` row in the catalog at all.
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
        only_the_base_partition_exists(body);
    });
    let mut options = planner_options(&harness, "planned-require-installed");
    options.require_installed = true;

    let artifacts = film_planner::generate(&transport, &planner_llm(&harness), &options)
        .await
        .expect("a text-only film plans with the reference partition absent");
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    assert!(
        artifacts
            .plan
            .shots
            .iter()
            .all(|shot| shot.conditioning.reference_roles.is_empty()),
        "with no partition to dispatch them against, reference shots are never offered"
    );
    // The envelope said so in as many words, so the planner was never invited to write one.
    let request = refine_job_payloads(&harness, true)
        .first()
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a planning job was created");
    assert!(request.contains("THIS CHECKPOINT HAS NONE"), "{request}");
    assert!(
        !request.contains("reference_to_video is the DEFAULT"),
        "{request}"
    );
    // Every compiled request stays on the base partition, so nothing here would load the ref DiT.
    assert!(
        artifacts
            .compiled
            .requests
            .iter()
            .all(|request| request.model == "minimax_h3"),
        "{:?}",
        artifacts
            .compiled
            .requests
            .iter()
            .map(|request| request.model.clone())
            .collect::<Vec<_>>()
    );

    // The base partition's own install state is still gated: uninstall it and the SAME planner
    // call refuses, naming it. The gate did not go away, it narrowed to what a plan can load.
    let strict = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |_| {});
    let mut options = planner_options(&harness, "planned-base-missing");
    options.require_installed = true;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let error = film_planner::generate(&strict, &planner_llm(&harness), &options)
        .await
        .expect_err("the base checkpoint is not installed either");
    let findings = findings_of(error);
    assert!(
        findings
            .iter()
            .any(|m| m.contains("minimax_h3") && m.contains("not installed")),
        "{findings:?}"
    );
}

/// sc-23405 AC2, the reference half. With a pack that approves references and a catalog that serves
/// the family's reference partition, the planner is TOLD to bind approved roles on every shot, and
/// the draft that does compiles to `reference_to_video` on `minimax_h3_ref` throughout.
///
/// The draft is scripted rather than decoded, so what this asserts is the two halves the harness
/// owns: the request the planner composes (the envelope's caps and its default mode), and the
/// resolution of what comes back. Whether a real 8B model follows the instruction is the coordinator's
/// real-LLM smoke (`scripts/film-harness-plan-smoke.sh`).
#[tokio::test]
async fn the_planner_binds_approved_roles_on_every_shot_when_the_pack_has_references() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&reference_draft())]);
    let options = planner_options(&harness, "planned-references");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("a reference-binding draft is a plan");
    assert_eq!(artifacts.repair_rounds, 0);
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    for shot in &artifacts.plan.shots {
        assert_eq!(shot.conditioning.mode, "reference_to_video", "{}", shot.id);
        assert!(
            !shot.conditioning.reference_roles.is_empty(),
            "{} binds nothing",
            shot.id
        );
    }
    // Every request resolves to the reference partition, in the draft's own role order.
    for request in &artifacts.compiled.requests {
        assert_eq!(request.model, "minimax_h3_ref", "{}", request.shot_id);
        assert_eq!(request.mode, "reference_to_video", "{}", request.shot_id);
        let shot = artifacts
            .plan
            .shots
            .iter()
            .find(|shot| shot.id == request.shot_id)
            .expect("every request is a shot");
        assert_eq!(
            request.reference_roles, shot.conditioning.reference_roles,
            "{}",
            request.shot_id
        );
    }
    // The plan still declares the FAMILY once — the planner cannot swap the brief's model.
    assert_eq!(artifacts.plan.model.id, "minimax_h3");
    assert_eq!(artifacts.compiled.model.id, "minimax_h3");

    // And the planner was actually told to do this: the REFERENCE partition's cap, the inverted
    // default mode, the standing binding rule and a worked example that models it.
    let request = refine_job_payloads(&harness, true)
        .first()
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a planning job was created");
    assert!(
        request.contains("at most 9 reference IMAGES"),
        "the cap is the REFERENCE partition's maxReferenceAssets, not the base entry's 0: {request}"
    );
    assert!(
        request.contains("reference_to_video is the DEFAULT"),
        "{request}"
    );
    assert!(
        request.contains("An approved reference pack is available"),
        "{request}"
    );
    assert!(
        request.contains("\"conditioning\": { \"mode\": \"reference_to_video\""),
        "the one worked example must model the default, not contradict it: {request}"
    );
}

/// sc-23405, E1 and the install gate. The reference partition the planner's envelope will send
/// every shot to is gated by install state exactly as `validate`/`run` gate the partition a
/// selected shot resolves to — a refusal in seconds naming the partition, rather than a full
/// planning run whose every request needs 18.78 GB that are not on this disk.
///
/// `--skip-install-check` is the documented escape, and it still plans the same film: what the
/// planner writes is decided by the catalog and the pack, never by which weights happen to be here,
/// or the same brief would produce two different films on two machines.
#[tokio::test]
async fn the_reference_partition_the_planner_will_use_is_gated_by_install_state() {
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
        only_the_base_partition_is_installed(body);
    });
    set_plan_replies(&harness, vec![draft_text(&reference_draft())]);
    let mut options = planner_options(&harness, "planned-ref-uninstalled");
    options.require_installed = true;
    let findings = findings_of(
        film_planner::generate(&transport, &planner_llm(&harness), &options)
            .await
            .expect_err("the reference DiT is not on this disk"),
    );
    assert!(
        findings
            .iter()
            .any(|m| m.contains("minimax_h3_ref") && m.contains("not installed")),
        "{findings:?}"
    );
    // Refused BEFORE the first decode: no planning job was ever created.
    assert!(
        refine_job_payloads(&harness, true).is_empty(),
        "the gate runs before a token is spent"
    );

    // `--skip-install-check`: the same catalog, the same pack, the same film.
    set_plan_replies(&harness, vec![draft_text(&reference_draft())]);
    let mut options = planner_options(&harness, "planned-ref-skipped");
    options.require_installed = false;
    let artifacts = film_planner::generate(&transport, &planner_llm(&harness), &options)
        .await
        .expect("--skip-install-check plans the reference film anyway");
    assert!(
        artifacts
            .compiled
            .requests
            .iter()
            .all(|request| request.model == "minimax_h3_ref"),
        "{:?}",
        artifacts
            .compiled
            .requests
            .iter()
            .map(|request| request.model.clone())
            .collect::<Vec<_>>()
    );
}

/// sc-23405 review, E1 at the SEAM. A pack that approves no BINDABLE reference plans a text-only
/// film and is never asked for the reference weights — even though the catalog SERVES the reference
/// partition and that partition is not installed.
///
/// This is the pack half of the property
/// `planning_without_a_reference_partition_in_the_catalog_stays_on_the_base_checkpoint` holds for the
/// catalog half, and it is a separate test because it fails for a different reason. The narrowing
/// that carries it — `narrowed_to_pack` inside `resolve_envelope` — had no harness-level cover: both
/// pack fixtures approve all seven roles, so deleting the call left all 159 film_harness tests green
/// while a pack that fills no reference shot still set `gate_reference` and made
/// `plan --require-installed` demand the 18.78 GB `transformer_ref` for a plan that could never load
/// it. That is the sc-23402/E1 regression, asserted here through the whole planner rather than only
/// on `narrowed_to_pack` directly.
///
/// The pack approves a style and a plate rather than nothing at all, and the brief's `requiredRoles`
/// are stripped, because a pack approving NOTHING is refused before an envelope exists — the anchor
/// rule and `requiredRoles` both demand an approved role. See
/// `fixture_pack_without_bindable_references`. What is left is the case the narrowing actually has
/// to carry: approved references that are not SUBJECTS, so no `reference_to_video` shot could bind
/// one.
#[tokio::test]
async fn a_pack_that_fills_no_reference_shot_plans_a_text_only_film_on_a_reference_serving_catalog()
{
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    // The FULL catalog: `minimax_h3_ref` is present and is NOT installed. With an approving pack
    // this exact transport + `require_installed` refuses by name
    // (`the_reference_partition_the_planner_will_use_is_gated_by_install_state`), which is what
    // makes a plan coming back here evidence that the envelope narrowed.
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
        only_the_base_partition_is_installed(body);
    });
    // `requiredRoles` stripped: they name the subjects this pack deliberately does not approve, and
    // an unmet required role is a refusal on the BRIEF, which would mask the envelope question.
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    for beat in brief["requiredBeats"].as_array_mut().unwrap() {
        beat.as_object_mut().unwrap().remove("requiredRoles");
    }
    let brief_path = harness.temp_dir.path().join("brief-no-required-roles.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();

    let mut options = planner_options(&harness, "planned-unbindable-pack");
    options.brief_path = brief_path;
    options.reference_pack_path = harness.fixture_pack_without_bindable_references();
    options.require_installed = true;

    let artifacts = film_planner::generate(&transport, &planner_llm(&harness), &options)
        .await
        .expect("a pack that fills no reference shot must not demand the reference weights");

    // No shot was offered references, so none binds any.
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    assert!(
        artifacts
            .plan
            .shots
            .iter()
            .all(|shot| shot.conditioning.reference_roles.is_empty()),
        "a pack approving no bindable subject fills no reference shot"
    );
    // The envelope told the planner so in as many words: the phase-1 envelope, not the widened one.
    let request = refine_job_payloads(&harness, true)
        .first()
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a planning job was created");
    assert!(request.contains("THIS CHECKPOINT HAS NONE"), "{request}");
    assert!(
        !request.contains("reference_to_video is the DEFAULT"),
        "{request}"
    );
    // And every compiled request stays on the base partition, so nothing here loads the ref DiT.
    assert!(
        artifacts
            .compiled
            .requests
            .iter()
            .all(|request| request.model == "minimax_h3"),
        "{:?}",
        artifacts
            .compiled
            .requests
            .iter()
            .map(|request| request.model.clone())
            .collect::<Vec<_>>()
    );
}

/// sc-23405 AC2, the enforcement half. `requiredRoles` is enforced exactly as before, through the
/// roles a shot BINDS — conditioning slots included. A reference draft that leaves the parcel out
/// of the handover is a finding that names the role, handed back verbatim to a repair round; a
/// planner that never binds it is refused rather than looped on.
#[tokio::test]
async fn a_reference_draft_that_leaves_a_required_role_unbound_is_repaired_then_refused_by_name() {
    let harness = Harness::start(true, vec![]).await;
    // The handover beat MUST show `red_parcel`; this draft binds it nowhere on that shot.
    let mut unbound = reference_draft();
    for shot in unbound["shots"].as_array_mut().unwrap() {
        if shot["beatId"] != "handover" {
            continue;
        }
        shot["conditioning"] = json!({
            "mode": "reference_to_video",
            "referenceRoles": ["courier", "workbench_table"]
        });
        shot["continuityRoles"] = json!(["courier", "workbench_table", "house_style"]);
    }
    set_plan_replies(
        &harness,
        vec![draft_text(&unbound), draft_text(&reference_draft())],
    );
    let options = planner_options(&harness, "repaired-references");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the repair round binds the parcel");
    assert_eq!(artifacts.repair_rounds, 1);
    let repair = refine_job_payloads(&harness, true)
        .get(1)
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a second planning job was created");
    assert!(
        repair.contains("red_parcel") && repair.contains("handover"),
        "the finding names the unbound role and its beat: {repair}"
    );

    // A planner that never binds it is refused after the declared rounds, naming the role.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&unbound)]);
    let mut options = planner_options(&harness, "exhausted-references");
    options.max_repair_rounds = 0;
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("an uncorrected draft is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|m| m.contains("red_parcel") && m.contains("handover")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn a_dropped_beat_is_repaired_and_the_repair_loop_is_bounded() {
    let harness = Harness::start(true, vec![]).await;
    // The first draft drops "handover"; the repair round returns the whole plan.
    let mut short = full_draft();
    short["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    set_plan_replies(
        &harness,
        vec![draft_text(&short), draft_text(&full_draft())],
    );
    let options = planner_options(&harness, "repaired");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the repair round produces a plan");
    assert_eq!(artifacts.repair_rounds, 1);
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    let cost = artifacts
        .compiled
        .planner
        .as_ref()
        .expect("the compiled document records the planner's cost");
    assert_eq!(cost.repair_rounds, 1, "{cost:?}");
    assert_eq!(
        cost.job_ids.len(),
        2,
        "the draft and its one repair: {cost:?}"
    );
    // The repair round was told exactly what was wrong, and the dropped beat was never accepted.
    let repair = refine_job_payloads(&harness, true)
        .get(1)
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a second planning job was created");
    assert!(repair.contains("Repair round 1 of 2"), "{repair}");
    assert!(
        repair.contains("required beat \"handover\""),
        "the finding is handed back verbatim: {repair}"
    );

    // A planner that never covers the beat is refused after the declared rounds — not looped on.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&short)]);
    let options = planner_options(&harness, "exhausted");
    let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect_err("an uncorrected draft is refused");
    let findings = findings_of(error);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("required beat \"handover\"")),
        "{findings:?}"
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("did not produce a valid plan within 2 repair round")),
        "{findings:?}"
    );
    // Exactly 1 + 2 model calls: the loop is bounded by the declared rounds.
    assert_eq!(harness.script.lock().plan_calls, 3);
    let prompts = refine_job_payloads(&harness, true);
    assert_eq!(
        prompts.len(),
        3,
        "the initial request plus exactly two repairs"
    );
    for (index, payload) in prompts.iter().enumerate().skip(1) {
        let prompt = payload["prompt"]
            .as_str()
            .expect("each refine job records its prompt");
        assert!(
            prompt.contains(&format!("Repair round {index} of 2")),
            "{prompt}"
        );
        let contract = prompt.find("# Output contract").expect("shared contract");
        let checklist = prompt
            .rfind("# Final repair checklist")
            .expect("final repair checklist");
        let finding = prompt
            .rfind("required beat \"handover\"")
            .expect("unchanged draft's exact finding is repeated last");
        assert!(contract < checklist && checklist < finding, "{prompt}");
        assert!(
            prompt.trim_end().ends_with(
                "Return the whole corrected JSON object. Do not return the draft above unchanged."
            ),
            "the repeated invalid output must receive an actionable final instruction: {prompt}"
        );
    }
    // Nothing was written but the diagnosable refusal.
    assert!(!options.out_dir.join("plan.json").exists());
    let rejected = std::fs::read_to_string(options.out_dir.join("planner-rejected.txt"))
        .expect("the refused answer is written out");
    assert!(rejected.contains("\"beatId\": \"arrival\""), "{rejected}");
    // Planning dispatches nothing: no video job was ever created.
    assert!(
        harness
            .jobs()
            .await
            .iter()
            .all(|job| job["type"] == "prompt_refine"),
        "planning created a non-planning job"
    );
}

#[tokio::test]
async fn malformed_and_out_of_envelope_drafts_are_refused_rather_than_coerced() {
    let mut unknown_field = full_draft();
    unknown_field["shots"][0]["cameraLens"] = json!("35mm");
    let mut off_menu = full_draft();
    off_menu["shots"][1]["targetDurationSeconds"] = json!(6.0);
    // A reference mode with nothing bound to it. Since sc-23402 a draft that BINDS reference roles
    // is legitimate — it resolves to the family's reference partition and dispatches there — but a
    // reference shot with no references is still a contradiction the validator names.
    let mut unsupported = full_draft();
    unsupported["shots"][2]["conditioning"] = json!({ "mode": "reference_to_video" });
    let mut unanchored = full_draft();
    unanchored["shots"][3]["continuityRoles"] = json!([]);
    unanchored["shots"][3]["conditioning"] =
        json!({ "mode": "text_to_video", "chainFromShotId": "SH0030" });
    let mut missing_asset = full_draft();
    missing_asset["shots"][4]["conditioning"] =
        json!({ "mode": "image_to_video", "firstFrameRole": "a_plate_nobody_approved" });

    let cases: Vec<(&str, String, &str)> = vec![
        (
            "prose",
            "I'd love to help! Here are some ideas for your film...".to_owned(),
            "not a plan document",
        ),
        ("unknown field", draft_text(&unknown_field), "cameraLens"),
        ("off-menu duration", draft_text(&off_menu), "duration menu"),
        (
            "unsupported conditioning",
            draft_text(&unsupported),
            "requires at least one reference role",
        ),
        (
            "chain cannot replace required continuity roles",
            draft_text(&unanchored),
            "into SH0040's continuityRoles",
        ),
        (
            "missing reference asset",
            draft_text(&missing_asset),
            "is not in reference pack",
        ),
    ];
    for (label, reply, expected) in cases {
        let harness = Harness::start(true, vec![]).await;
        set_plan_replies(&harness, vec![reply]);
        let mut options = planner_options(&harness, "refused");
        options.max_repair_rounds = 0;
        let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .err()
            .unwrap_or_else(|| panic!("{label}: expected a refusal"));
        let findings = findings_of(error);
        assert!(
            findings.iter().any(|finding| finding.contains(expected)),
            "{label}: {findings:?}"
        );
        assert!(!options.out_dir.join("plan.json").exists(), "{label}");
        // One call: with zero repair rounds the planner asks once and stops.
        assert_eq!(harness.script.lock().plan_calls, 1, "{label}");
    }
}

#[tokio::test]
async fn a_hosted_endpoint_refuses_the_planner_before_it_creates_a_job() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let mut options = planner_options(&harness, "hosted");
    options.api_url = "https://api.openai.com/v1".to_owned();
    let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect_err("a hosted endpoint is refused");
    let findings = findings_of(error);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("not this machine or a private-network host"),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0);
    assert!(harness.jobs().await.is_empty(), "no job was created");
}

#[tokio::test]
async fn planning_without_a_local_refiner_is_refused_rather_than_queued_forever() {
    // No worker at all: nothing can run an LLM job, so the planner says so instead of waiting out
    // its timeout on a job nobody will claim.
    let harness = Harness::start(false, vec![]).await;
    let options = planner_options(&harness, "no-refiner");
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("planning without a refiner is refused"),
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("no live registered worker advertises prompt_refine"),
        "{findings:?}"
    );
    assert!(harness.jobs().await.is_empty(), "no job was created");
}

#[tokio::test]
async fn the_plan_is_editable_between_generation_and_dispatch_and_a_stale_compile_is_refused() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let options = planner_options(&harness, "editable");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    // The human edits the plan by hand: a different prompt and a longer legal clip.
    let mut plan: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.plan_path).expect("plan readable"),
    )
    .expect("plan parses");
    plan["shots"][0]["prompt"] = json!("A hand-written prompt the planner never wrote.");
    plan["shots"][0]["targetDurationSeconds"] = json!(8.0);
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();

    // The stale compiled document is refused rather than dispatched.
    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        // The harness's own copy, never the checked-in directory: a run writes its synthesized
        // clips beside the pack (sc-23404).
        reference_pack_path: harness.fixture_pack(),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("editable-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &run_options)
            .await
            .expect_err("a plan edited after the compile is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("has changed since it was compiled")),
        "{findings:?}"
    );

    // Recompiling adopts the edit, and `--no-refine` keeps the hand-written text verbatim.
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &artifacts.plan_path,
    )
    .await
    .expect("the edited plan recompiles");
    let first = &artifacts.compiled.requests[0];
    // Verbatim and intact. NOT `starts_with`: since sc-24025 the compiler's identity text leads a
    // shot that names continuity roles it does not bind, and `--no-refine` promises the authored
    // text is untouched, not that nothing the compiler owns is written around it. So everything
    // ahead of the authored text must be RECORDED inserted text — `contains` alone would permit
    // arbitrary unattributed prose, which is precisely what `--no-refine` forbids.
    const AUTHORED: &str = "A hand-written prompt the planner never wrote.";
    let authored_at = first
        .prompt
        .find(AUTHORED)
        .unwrap_or_else(|| panic!("the hand-written text survives verbatim: {}", first.prompt));
    let leading = first
        .inserted_text
        .iter()
        .filter(|piece| piece.kind.placement() == InsertedTextPlacement::Leading)
        .map(|piece| piece.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(
        first.prompt[..authored_at].trim(),
        leading.trim(),
        "only the compiler's own recorded insertions may precede the authored prompt: {}",
        first.prompt
    );
    assert!(
        first
            .prompt
            .ends_with("Audio: Room tone, distant birds. No music."),
        "{}",
        first.prompt
    );
    assert_eq!(first.duration_seconds, 8.0);
    film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the recompiled plan validates");

    // An edit that deletes a beat is reported against the brief rather than quietly compiled.
    let mut plan: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.plan_path).expect("plan readable"),
    )
    .unwrap();
    plan["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &options,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a deleted beat is reported"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("\"handover\"")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn a_generated_plan_dispatches_its_compiled_prompts_through_the_same_run_path() {
    let harness = Harness::start(true, vec![]).await;
    // Two beats is enough to prove the dispatch path; the run itself is the sc-22710 one.
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
    harness.script.lock().refine_template =
        Some("integrated_multimodal_description: {prompt}".to_owned());
    let mut options = planner_options(&harness, "dispatch");
    options.brief_path = brief_path;
    options.refine_prompts = true;
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        // The harness's own copy, never the checked-in directory: a run writes its synthesized
        // clips beside the pack (sc-23404).
        reference_pack_path: harness.fixture_pack(),
        // Found beside the plan, exactly as a run started from the plan directory would.
        compiled_path: None,
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("dispatch-run"),
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
    assert_eq!(record.shots.len(), 2);
    assert!(record
        .shots
        .iter()
        .all(|shot| shot.outcome == ShotOutcome::Rendered));
    // The run record pins the compiled document it dispatched, not just the plan.
    let compiled_source = record
        .compiled
        .as_ref()
        .expect("compiled document recorded");
    assert_eq!(compiled_source.id, "courier-workshop-planned");
    assert_eq!(compiled_source.sha256.len(), 64);
    assert!(run_options.out_dir.join("compiled.json").is_file());

    // Every dispatched video job carried the COMPILED prompt, not the plan's authored one.
    let dispatched: Vec<Value> = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(job_type, _, _)| job_type == "video_generate")
        .map(|(_, _, payload)| payload.clone())
        .collect();
    assert_eq!(dispatched.len(), 2);
    for payload in &dispatched {
        let shot_id = payload["advanced"]["filmHarness"]["shotId"]
            .as_str()
            .unwrap();
        let request = artifacts
            .compiled
            .request(shot_id)
            .unwrap_or_else(|| panic!("{shot_id} is in the compiled document"));
        assert_eq!(payload["prompt"], json!(request.prompt));
        assert_eq!(payload["duration"], json!(request.duration_seconds));
        assert_eq!(payload["width"], json!(request.width));
        assert_eq!(payload["height"], json!(request.height));
        assert_eq!(
            payload["advanced"]["filmHarness"]["promptSource"],
            "refined"
        );
        assert_eq!(payload["advanced"]["mlxQuantize"], 4);
        // Contained, not leading: the compiler's identity text leads these shots (sc-24025). The
        // payload is asserted equal to `request.prompt` above, so the exact composition is already
        // pinned; what this adds is that the REFINER's words are the ones that reached the route.
        assert!(
            payload["prompt"]
                .as_str()
                .unwrap()
                .contains("integrated_multimodal_description:"),
            "{payload}"
        );
    }
}

#[tokio::test]
async fn a_hand_edited_compiled_request_is_refused_instead_of_dispatched() {
    // The compiled document — not the plan — is what becomes the job body: `execute_run` takes the
    // mode, model, duration, fps, geometry, seed, negative prompt and every role slot straight out
    // of it. An edit here therefore reaches the engine unless something judges THIS document, and
    // 9.0s is the case that would not even fail loudly: it is inside H3's hard bounds, so the
    // engine snaps it up onto the 17n+5 lattice and renders a length the plan never claimed.
    let harness = Harness::start(true, vec![]).await;
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(2);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(2);
    brief["targetTotalSeconds"] = json!({ "min": 10.0, "max": 12.0 });
    let brief_path = harness.temp_dir.path().join("tamper-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();

    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "tamper");
    options.brief_path = brief_path;
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    // Edit ONE field of ONE compiled request, leaving the plan and its sha256 untouched, so the
    // staleness check has nothing to say.
    let mut compiled: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.compiled_path).expect("compiled.json"),
    )
    .expect("compiled.json parses");
    let tampered_shot = compiled["requests"][1]["shotId"]
        .as_str()
        .expect("a second request")
        .to_owned();
    compiled["requests"][1]["durationSeconds"] = json!(9.0);
    std::fs::write(
        &artifacts.compiled_path,
        serde_json::to_string_pretty(&compiled).unwrap(),
    )
    .unwrap();

    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        // The harness's own copy, never the checked-in directory: a run writes its synthesized
        // clips beside the pack (sc-23404).
        reference_pack_path: harness.fixture_pack(),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("tamper-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let findings = findings_of(
        film_harness::run(&harness.transport, &run_options)
            .await
            .expect_err("a hand-edited compiled request is refused"),
    );
    assert!(
        findings.iter().any(|finding| {
            finding.contains(&format!("[{tampered_shot}]"))
                && finding.contains("compiled.durationSeconds")
                && finding.contains("9s")
        }),
        "{findings:?}"
    );
    // Refused BEFORE dispatch: nothing rendered.
    assert!(
        harness
            .jobs()
            .await
            .iter()
            .all(|job| job["type"] == "prompt_refine"),
        "a video job was created for a request that was refused"
    );
    // `validate` refuses it on the same grounds, so the operator sees it without starting a run.
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &run_options)
            .await
            .expect_err("validate refuses it too"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("compiled.durationSeconds")),
        "{findings:?}"
    );

    // Restored, the same documents run.
    compiled["requests"][1]["durationSeconds"] = json!(5.1667);
    std::fs::write(
        &artifacts.compiled_path,
        serde_json::to_string_pretty(&compiled).unwrap(),
    )
    .unwrap();
    film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the untampered document validates");
}

#[tokio::test]
async fn a_brief_the_model_cannot_render_is_refused_before_a_single_decode() {
    // fps 30 is a property of the BRIEF, which every draft copies verbatim — the planner is not
    // allowed to change it. Discovered on the first draft it would cost 1 + rounds full local
    // decodes (minutes each on an 8B) and then blame the planner for its input.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["model"]["fps"] = json!(30);
    let brief_path = harness.temp_dir.path().join("off-menu-fps-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let mut options = planner_options(&harness, "off-menu-fps");
    options.brief_path = brief_path.clone();

    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("an unrenderable brief is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("model.fps") && finding.contains("30 fps")),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0, "a decode was spent");
    assert!(harness.jobs().await.is_empty(), "a job was created");

    // The memory budget is judged the same way, against the lane the API HOST renders on.
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["limits"]["maxMemoryGb"] = json!(8.0);
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("a budget below the model's minimum is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("limits.maxMemoryGb")),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0);

    // And the PLANNER's own budget (sc-22715): the decodes run on the same host, so a
    // `plannerMaxMemoryGb` the host cannot meet is refused before the first token too.
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["limits"]["plannerMaxMemoryGb"] = json!(4096.0);
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("a planner budget above the host's memory is refused"),
    );
    assert!(
        findings.iter().any(
            |finding| finding.contains("limits.plannerMaxMemoryGb") && finding.contains("4096")
        ),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0);
    // A brief that declares no planner budget at all is refused by the document check.
    brief["limits"]
        .as_object_mut()
        .unwrap()
        .remove("plannerMaxMemoryGb");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("an undeclared planner budget is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("brief.limits.plannerMaxMemoryGb")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn the_default_correction_loop_rechecks_beat_coverage_without_a_brief_flag() {
    // The documented loop is `plan --out DIR`, edit `DIR/plan.json`, `compile --plan DIR/plan.json
    // --out DIR`. `compile` looks for a brief beside the plan, so `plan` has to leave one there —
    // otherwise the recompile AC3 asks a human to run performs NO coverage check and a hand edit
    // that deletes a required beat compiles and dispatches silently.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let options = planner_options(&harness, "loop");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");
    let sibling = options.out_dir.join("brief.json");
    assert!(sibling.is_file(), "the brief travels with the plan");
    assert_eq!(
        std::fs::read(&sibling).unwrap(),
        std::fs::read(BRIEF_FIXTURE).unwrap(),
        "the brief is copied byte for byte, comments and all"
    );

    // Now the human deletes a beat's shot, and compiles WITHOUT naming a brief.
    let mut plan: Value =
        serde_json::from_str(&std::fs::read_to_string(&artifacts.plan_path).unwrap()).unwrap();
    plan["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();
    let mut blind = options.clone();
    blind.brief_path = harness.temp_dir.path().join("no-such-brief.json");
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &blind,
            &artifacts.plan_path,
        )
        .await
        .expect_err("the deleted beat is caught by the discovered sibling brief"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("\"handover\"")),
        "{findings:?}"
    );

    // A brief the caller NAMED but that cannot be read is an error, not a silent skip: a typo in
    // `--brief` must not be indistinguishable from "coverage verified".
    let malformed = harness.temp_dir.path().join("malformed-brief.json");
    std::fs::write(&malformed, "{ \"schemaVersion\": 1, ").unwrap();
    let mut named = options.clone();
    named.brief_path = malformed.clone();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &named,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a malformed named brief is an error"),
    );
    assert!(
        findings.iter().any(|finding| finding.contains("brief")),
        "{findings:?}"
    );

    // So is one that parses but is not a brief this build accepts.
    let mut stale: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    stale["schemaVersion"] = json!(99);
    std::fs::write(&malformed, serde_json::to_string_pretty(&stale).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &named,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a stale schema version is an error"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("brief schema version 99")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn the_prompt_guide_reaches_the_rewrite_the_way_video_studio_sends_it() {
    // The per-shot rewrite is only the SAME rewrite the "Refine" button runs if the model's prompt
    // guide rides with it: the web forwards `guide`, and the worker appends it to the H3 system
    // turn under `# Model prompt guide`. The harness cannot fetch it from `--api` (the rust-api
    // serves `/prompt-guides/` only in an `embed-web` build), so it is read from disk.
    let harness = Harness::start(true, vec![]).await;
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(1);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(1);
    brief["targetTotalSeconds"] = json!({ "min": 5.0, "max": 6.0 });
    let brief_path = harness.temp_dir.path().join("one-beat-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let guide_path = harness.temp_dir.path().join("h3-guide.md");
    std::fs::write(&guide_path, "# H3\nWrite one paragraph.").unwrap();

    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "guided");
    options.brief_path = brief_path;
    options.refine_prompts = true;
    options.prompt_guide_path = Some(guide_path);
    film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    let jobs = refine_job_payloads(&harness, false);
    assert_eq!(jobs.len(), 2, "{jobs:?}");
    // The planning turn carries no guide: its system turn is the plan contract, not prompt advice.
    assert!(jobs[0].get("guide").is_none(), "{:?}", jobs[0]);
    let guide = jobs[1]["guide"]
        .as_str()
        .expect("the rewrite carries a guide");
    assert!(
        guide.starts_with("# H3\nWrite one paragraph."),
        "the model's own guide rides with the rewrite, verbatim and first: {guide:?}"
    );
    // And the FILM path's own rules follow it (sc-24029). The guide teaches `<Picture N>` as the
    // way to give a reference a job, which is right for a person writing one prompt against
    // references they chose and wrong here: the compiler assigns and writes every label AFTER the
    // rewrite. Said in the film path's own text rather than in the worker's rewrite asset, which
    // belongs to every caller of `prompt_refine` and is hash-pinned into the StarVector closure.
    // LAST, so it is the last word on a subject the guide above has already spoken on.
    assert!(
        guide.contains("NEVER write an engine media label")
            && guide.contains("`<Picture 1>`")
            && guide.contains("assigned and written by the compiler AFTER your rewrite"),
        "the film path tells the refiner not to write a label: {guide:?}"
    );
    assert!(
        guide
            .find("# Film harness rules")
            .expect("the block is present")
            > guide.find("Write one paragraph.").unwrap(),
        "the film rules come after the model guide they override: {guide:?}"
    );

    // A guide the caller NAMED and that is not there is an error, not a guide-less rewrite.
    let mut missing = options.clone();
    missing.out_dir = harness.temp_dir.path().join("guide-missing");
    missing.prompt_guide_path = Some(harness.temp_dir.path().join("no-such-guide.md"));
    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &missing)
            .await
            .expect_err("an unreadable named guide is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("cannot read the prompt guide")),
        "{findings:?}"
    );
}

#[test]
fn the_checked_in_brief_is_valid_and_matches_the_hand_authored_baseline() {
    let text = std::fs::read_to_string(BRIEF_FIXTURE).expect("brief fixture readable");
    let brief = sceneworks_core::film_planner::parse_brief(&text).expect("brief parses");
    let findings = sceneworks_core::film_planner::validate_brief(&brief);
    assert!(findings.is_empty(), "{findings:?}");
    let ids: Vec<&str> = brief
        .required_beats
        .iter()
        .map(|beat| beat.id.as_str())
        .collect();
    assert_eq!(ids, BRIEF_BEATS);
    // The brief plans the SAME model, tier and canvas the hand-authored baseline renders, so the
    // two plans are comparable, and it carries a DIFFERENT plan id so neither overwrites the other.
    let baseline = sceneworks_core::film_plan::parse_plan(
        &std::fs::read_to_string(Path::new(FIXTURE_DIR).join("plan.jsonc")).unwrap(),
    )
    .unwrap();
    assert_eq!(brief.model, baseline.model);
    // The render-side limits match; `plannerMaxMemoryGb` is a brief-only declaration (a
    // hand-authored plan runs no planner), so it is the one field left out of the comparison.
    assert_eq!(
        sceneworks_core::film_plan::PlanLimits {
            planner_max_memory_gb: None,
            ..brief.limits.clone()
        },
        baseline.limits
    );
    assert!(
        brief.limits.planner_max_memory_gb.is_some(),
        "the checked-in brief declares the planner's memory budget"
    );
    assert_ne!(brief.id, baseline.id);
    // Every beat is coverable inside the model's shortest legal clip and the declared window.
    assert!(brief.required_beats.len() as f64 * 5.1667 >= brief.target_total_seconds.min);
    assert!(brief.max_shots >= brief.required_beats.len());
}

// ---------------------------------------------------------------------------------------------
// sc-23404 — speech dialogue synthesized through the audio job route
// ---------------------------------------------------------------------------------------------

/// A pack whose dialogue entries carry `text`, copied into the temp dir so the clips synthesis
/// writes land somewhere disposable rather than in the checked-in fixture.
///
/// `edit` shapes the parsed pack first, so a test can break one entry (drop its text, move it to
/// the wrong kind) without touching the shipped documents.
fn speech_pack(harness: &Harness, edit: impl FnOnce(&mut Value)) -> PathBuf {
    // `fixture_pack` already gives every harness its own copy of the pack and its media, which is
    // what keeps a run's synthesized clips out of the checked-in fixture; this only rewrites the
    // document in place beside them.
    let path = harness.fixture_pack();
    let text = std::fs::read_to_string(&path).expect("fixture pack");
    let mut pack: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
            .expect("fixture pack parses");
    edit(&mut pack);
    std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
    path
}

/// The synthesis jobs the API holds for one project, read out of the JOB TABLE.
///
/// Not out of the fake worker's claim log: that log is the fake's own bookkeeping and says what a
/// worker picked UP, so a job the harness enqueued that nobody claimed — the thing an
/// "exactly one job was created" assertion most needs to catch — would not appear in it at all.
async fn audio_job_count(harness: &Harness, project_id: &str) -> usize {
    let (status, jobs) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs?projectId={project_id}&limit=100"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{jobs}");
    jobs.as_array()
        .into_iter()
        .flatten()
        .filter(|job| job["type"] == "audio_generate")
        .count()
}

/// Reopen a finished run so `resume` walks the whole pipeline — project, references, sound — the
/// way a crash-resume does, rather than returning at the first "already completed" gate.
fn reopen_for_resume(harness: &Harness) {
    harness.edit_run_record(|record| {
        record["state"] = json!("running");
        record["outcome"] = json!("failed");
        record["export"]["stale"] = json!(true);
        record["stop"] = json!({
            "reason": "export_failed", "detail": "reopened by the test", "resumable": true
        });
    });
}

/// AC1: a `dialogue` entry carrying `text` yields ONE `audio_generate` job through the real route,
/// a WAV in the pack directory, a dialogue-bus item at the shot's offset, and provenance in the run
/// record — model, voice, text, job id and asset id.
///
/// Runs on every lane, ffmpeg or not: the synthesized clip is a canonical PCM-16 WAV, which the
/// import route stores without a transcode (`media_convert::is_canonical_pcm16_wav`), exactly as
/// the fixture's beds are.
#[tokio::test]
async fn a_dialogue_line_with_text_is_synthesized_placed_and_recorded() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
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

    let project_id = record.project_id.clone().expect("project created");
    // ONE synthesis job, for the ONE line this two-shot selection places. The recipient's lines
    // belong to SH050/SH060, which the selection leaves out, so they are never spoken — synthesis
    // follows the same "only what the run PLACES" rule the import does.
    assert_eq!(audio_job_count(&harness, &project_id).await, 1);
    assert_eq!(
        record.synthesized_sound.len(),
        1,
        "{:#?}",
        record.synthesized_sound
    );
    let line = &record.synthesized_sound[0];
    assert_eq!(line.role, "courier_line");
    assert_eq!(line.text, "Delivery. I'll leave it on the bench.");
    assert_eq!(line.model, "kokoro_82m");
    assert_eq!(line.voice.as_deref(), Some("am_michael"));
    assert_eq!(line.status, "completed");
    assert!(line.job_id.is_some(), "{line:#?}");
    assert!(line.asset_id.is_some(), "{line:#?}");
    assert!(line.is_usable(), "{line:#?}");

    // The dispatched body is the real audio route's, and it carries the key a resume adopts by.
    let (_, payload) = harness
        .script
        .lock()
        .audio_claimed
        .first()
        .cloned()
        .expect("the fake claimed the synthesis job");
    assert_eq!(payload["prompt"], "Delivery. I'll leave it on the bench.");
    assert_eq!(payload["model"], "kokoro_82m");
    assert_eq!(payload["voice"], "am_michael");
    assert_eq!(
        payload["advanced"]["filmHarness"]["idempotencyKey"],
        json!(line.idempotency_key)
    );
    assert_eq!(payload["advanced"]["filmHarness"]["role"], "courier_line");
    // The route resolved the model's manifest entry, which is what says this went through
    // `create_audio_job` rather than through a hand-built job row.
    assert_eq!(payload["modelManifestEntry"]["type"], "audio");

    // The WAV is in the PACK directory, under the deterministic name, and it is what was imported.
    let file = line.file.clone().expect("the clip was written");
    assert_eq!(
        file,
        film_harness::synthesized_sound_file("courier_line", &line.text_sha256)
    );
    let on_disk = pack.parent().expect("pack dir").join(&file);
    assert!(on_disk.is_file(), "{} was not written", on_disk.display());
    let imported = record
        .sound
        .iter()
        .find(|clip| clip.role == "courier_line")
        .expect("the spoken line is imported like any other clip");
    assert_eq!(imported.file, file);
    assert_eq!(imported.kind, "dialogue");
    assert_ne!(
        Some(&imported.asset_id),
        line.asset_id.as_ref(),
        "the dialogue bus plays the IMPORTED clip, not the synthesis job's library asset"
    );

    // The imported asset says it was spoken rather than recorded.
    let (status, asset) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{}", imported.asset_id),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
    assert_eq!(asset["type"], "audio", "{asset}");
    assert_eq!(
        asset["extra"]["filmHarness"]["synthesized"], true,
        "{asset}"
    );

    // And it lands on the dialogue bus at the shot's own offset, for the length that was spoken.
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1, "{dialogue:#?}");
    assert_eq!(dialogue[0]["filmHarness"]["shotId"], "SH020");
    assert!(
        close(dialogue[0]["timelineStart"].as_f64().unwrap(), 5.1667 + 1.2),
        "{}",
        dialogue[0]
    );
    let (_, spoken) = fake_speech_shape(Some("am_michael"), &line.text);
    assert!(
        close(
            dialogue[0]["timelineEnd"].as_f64().unwrap(),
            5.1667 + 1.2 + spoken
        ),
        "the item is as long as the line that was actually spoken ({spoken}s): {}",
        dialogue[0]
    );
}

/// E3: the entry's `model` is what gets POSTED and what gets RECORDED, not the route's default.
///
/// Every other synthesis test leaves `model` off the pack entry, so
/// [`sceneworks_core::film_plan::DEFAULT_SOUND_SYNTHESIS_MODEL`] would satisfy them all — a harness
/// that ignored the entry's model and always posted `kokoro_82m` would stay green, and a plan asking
/// for a named voice model would silently get the default one. Naming a second
/// [`sceneworks_core::film_plan::SOUND_SYNTHESIS_MODELS`] entry is what separates the two.
#[tokio::test]
async fn a_dialogue_entry_that_names_a_speech_model_is_synthesized_and_recorded_on_that_model() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |pack| {
        let sound = pack["sound"].as_array_mut().expect("sound");
        let entry = sound
            .iter_mut()
            .find(|entry| entry["role"] == "courier_line")
            .expect("the courier's line is in the pack");
        entry["model"] = json!("chatterbox_tts");
    });
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );

    let line = &record.synthesized_sound[0];
    assert_eq!(line.role, "courier_line");
    assert_eq!(
        line.model, "chatterbox_tts",
        "the record names the model the PACK asked for: {line:#?}"
    );
    let (_, payload) = harness
        .script
        .lock()
        .audio_claimed
        .first()
        .cloned()
        .expect("the fake claimed the synthesis job");
    assert_eq!(
        payload["model"], "chatterbox_tts",
        "the posted job asks for the pack's model, not the route default: {payload}"
    );
    assert_eq!(payload["modelManifestEntry"]["type"], "audio", "{payload}");
    assert!(line.is_usable(), "{line:#?}");
}

/// AC1, second half: a resume after the synthesis completed ADOPTS the clip — no second job, no
/// second asset, the same file.
#[tokio::test]
async fn a_resume_adopts_a_spoken_line_instead_of_speaking_it_again() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let project_id = first.project_id.clone().expect("project");
    assert_eq!(audio_job_count(&harness, &project_id).await, 1);
    let spoken = first.synthesized_sound[0].clone();

    reopen_for_resume(&harness);
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        audio_job_count(&harness, &project_id).await,
        1,
        "the resume must adopt the clip, not speak the line a second time"
    );
    assert_eq!(resumed.synthesized_sound.len(), 1);
    assert_eq!(resumed.synthesized_sound[0], spoken);
    assert_eq!(resumed.sound.len(), first.sound.len());
    assert_eq!(resumed.project_id.as_deref(), Some(project_id.as_str()));
}

/// AC2, first half: a pack entry with neither `text` nor `file`, and `text` on a non-dialogue kind,
/// are refused BEFORE dispatch and by role. Nothing is created — no project, no job.
#[tokio::test]
async fn a_sound_entry_with_no_source_or_a_spoken_bed_is_refused_before_dispatch() {
    /// How one case breaks the pack: drop the courier's line so its entry has no source at all, or
    /// give the room-tone bed a line to speak.
    #[derive(Clone, Copy)]
    enum Break {
        NoSource,
        SpokenBed,
    }
    let cases = [
        (Break::NoSource, "neither `file` nor `text`", "courier_line"),
        (
            Break::SpokenBed,
            "only a `dialogue` entry may carry `text`",
            "workshop_room_tone",
        ),
    ];
    for (case, needle, label) in cases {
        let harness = Harness::start(true, vec![]).await;
        let pack = speech_pack(&harness, |pack| match case {
            Break::NoSource => {
                pack["sound"][0]
                    .as_object_mut()
                    .expect("entry")
                    .remove("text");
            }
            Break::SpokenBed => {
                pack["sound"][3]["text"] = json!("a quiet workshop, distant birds");
            }
        });
        let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
        let error = film_harness::run(&harness.transport, &options)
            .await
            .unwrap_err();
        let HarnessError::Validation(findings) = error else {
            panic!("{label}: expected a validation refusal, got {error}");
        };
        let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
        // The finding names the ENTRY by role, not just the array slot.
        assert!(
            text.iter()
                .any(|message| message.contains(needle) && message.contains(label)),
            "{label}: the finding must name the entry: {text:?}"
        );
        assert!(harness.jobs().await.is_empty(), "{label}");
        assert_eq!(harness.project_count().await, 0, "{label}");
        assert_eq!(harness.run_record()["outcome"], "rejected", "{label}");
    }
}

/// AC2, second half: a synthesis that FAILS stops the run with a resumable reason and leaves the
/// rest of the sequence intact — no render is dispatched, the record says which line and why, and a
/// resume with a worker that can speak finishes the film.
#[tokio::test]
async fn a_failed_synthesis_stops_the_run_resumably_with_the_rest_intact() {
    let harness = Harness::start(true, vec![]).await;
    harness.script.lock().audio_fails = true;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("the run stopped");
    assert_eq!(stop.reason, "dialogue_synthesis_failed", "{stop:?}");
    assert!(stop.resumable, "{stop:?}");
    assert!(
        stop.detail.contains("courier_line"),
        "the stop names the line: {}",
        stop.detail
    );
    assert!(record.is_resumable());

    // Nothing downstream ran: the failure lands BEFORE the first render, which is the point of
    // speaking the lines before the shots.
    assert_eq!(harness.video_job_count(), 0);
    assert!(record.export.is_none());
    let line = &record.synthesized_sound[0];
    assert_eq!(line.role, "courier_line");
    assert_eq!(line.status, "failed");
    assert!(
        line.error
            .as_deref()
            .is_some_and(|error| error.contains("fake tts fault")),
        "{line:#?}"
    );
    assert!(line.asset_id.is_none());

    harness.script.lock().audio_fails = false;
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(resumed.synthesized_sound.len(), 1, "one record per role");
    assert_eq!(resumed.synthesized_sound[0].status, "completed");
    assert_eq!(
        resumed.synthesized_sound[0].attempt, 2,
        "the retry is a NEW attempt under a new key, not a re-poll of the failed job"
    );
    assert_ne!(
        resumed.synthesized_sound[0].idempotency_key,
        line.idempotency_key
    );
    assert_eq!(
        audio_job_count(
            &harness,
            resumed.project_id.as_deref().expect("project created")
        )
        .await,
        2,
        "one attempt that failed, one that spoke"
    );
    assert_eq!(resumed.sound.len(), 3, "{:#?}", resumed.sound);
}

/// A pack that asks for speech on a host with no TTS worker is told so BEFORE the job exists.
///
/// Without this the synthesis would be enqueued, claimed by nobody, and cancelled when the plan's
/// per-job budget ran out — the run would take `maxShotSeconds` to report "not synthesized" instead
/// of reporting "nothing here can speak" in a second. The same posture as the `video_generate` and
/// `image_vqa` preflights, and scoped to the lines still OWED, so a run whose clips are all already
/// spoken needs no TTS worker at all.
#[tokio::test]
async fn speech_with_no_live_audio_worker_is_refused_before_the_job_exists() {
    let harness = Harness::start(false, vec![]).await;
    harness.script.lock().capabilities = Some(vec![
        "video_generate",
        "timeline_export",
        "frame_extract",
        "image_vqa",
        "prompt_refine",
    ]);
    harness.spawn_worker().await;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("the run stopped");
    assert_eq!(stop.reason, "no_audio_worker", "{stop:?}");
    assert!(stop.resumable, "{stop:?}");
    assert!(
        stop.detail.contains("courier_line") && stop.detail.contains("audio_generate"),
        "{}",
        stop.detail
    );
    // Nothing was enqueued and nothing was rendered.
    assert!(
        record.synthesized_sound.is_empty(),
        "{:#?}",
        record.synthesized_sound
    );
    assert_eq!(harness.video_job_count(), 0);
    assert_eq!(harness.jobs().await.len(), 0);
}

/// AC2, the limit: a synthesis that runs past the plan's declared per-job budget stops the run
/// resumably, with nothing rendered.
#[tokio::test]
async fn a_synthesis_that_overruns_its_budget_stops_the_run_resumably() {
    let harness = Harness::start(true, vec![]).await;
    harness.script.lock().audio_hangs = true;
    let pack = speech_pack(&harness, |_| {});
    // The plan's own declared bound, tightened: one second for a synthesis the fake never finishes.
    let plan = {
        let text = std::fs::read_to_string(harness.fixture_plan()).expect("plan");
        let mut plan: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("plan parses");
        plan["limits"]["maxShotSeconds"] = json!(1);
        let path = harness.temp_dir.path().join("tight-plan.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        path
    };
    let options = harness.options(plan, pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("the run stopped");
    assert_eq!(stop.reason, "dialogue_synthesis_failed", "{stop:?}");
    assert!(stop.resumable, "{stop:?}");
    assert!(
        stop.detail.contains("per-job budget"),
        "the stop says which limit: {}",
        stop.detail
    );
    assert_eq!(record.synthesized_sound[0].status, "timed_out");
    assert_eq!(harness.video_job_count(), 0);
}

/// What the synthesis key is keyed on, checked directly rather than through a run.
///
/// The role alone would be stable across restarts too — and would make a re-cast line adopt the job
/// that spoke the old one, so the film would go on saying the wrong thing in the wrong voice. The
/// attempt is the retry axis: without it, a resume after a failure finds the FAILED job under the
/// same key and re-reads the same failure forever instead of speaking the line.
#[test]
fn the_synthesis_key_separates_content_and_attempts_and_is_stable_otherwise() {
    let key = |voice: Option<&str>, text: &str, attempt: u32| {
        film_harness::dialogue_idempotency_key(
            "run_1",
            "courier_line",
            "kokoro_82m",
            voice,
            text,
            attempt,
        )
    };
    let base = key(Some("am_michael"), "Delivery.", 1);
    assert_eq!(base, key(Some("am_michael"), "  Delivery.  ", 1));
    assert!(base.starts_with("run_1:sound:courier_line:"), "{base}");
    assert!(base.ends_with(":a1"), "{base}");
    for different in [
        key(Some("bm_george"), "Delivery.", 1),
        key(None, "Delivery.", 1),
        key(Some("am_michael"), "Delivery, sorry.", 1),
        key(Some("am_michael"), "Delivery.", 2),
        film_harness::dialogue_idempotency_key(
            "run_1",
            "recipient_line",
            "kokoro_82m",
            Some("am_michael"),
            "Delivery.",
            1,
        ),
        film_harness::dialogue_idempotency_key(
            "run_1",
            "courier_line",
            "chatterbox_tts",
            Some("am_michael"),
            "Delivery.",
            1,
        ),
    ] {
        assert_ne!(base, different);
    }
}

/// The clip's name is deterministic in the role and the line, so the same pack run twice writes the
/// same file and a resume finds the one it wrote — and a CHANGED line is a different file rather
/// than a silent overwrite of the one the last export used.
#[test]
fn a_synthesized_clips_name_is_deterministic_in_the_role_and_the_line() {
    let digest = "0123456789abcdef0123";
    assert_eq!(
        film_harness::synthesized_sound_file("courier_line", digest),
        "sound/courier_line.tts-0123456789ab.wav"
    );
    assert_ne!(
        film_harness::synthesized_sound_file("courier_line", digest),
        film_harness::synthesized_sound_file("recipient_line", digest)
    );
}

/// A replacement re-assembles the timeline, and the sound has to survive it.
///
/// `assemble_timeline` derives the sound tracks from the session's clip map, and
/// `merge_harness_audio_track` keeps only the items the harness does NOT own — so a replacement
/// that re-assembled without re-hydrating the clips did not leave the saved dialogue and beds
/// alone, it DELETED them, and the re-export came back as picture with nothing under it. That is
/// the phase-1 evaluation's finding #2, fixed here alongside synthesis because a spoken line is
/// exactly what it silences.
#[tokio::test]
async fn a_replacement_re_assembles_with_the_sound_the_run_already_has() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let project_id = record.project_id.clone().expect("project");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let before = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(items_of(&before, "track_dialogue").len(), 1);
    assert_eq!(items_of(&before, "track_ambience").len(), 1);
    let audio_jobs = audio_job_count(&harness, &project_id).await;

    let replaced = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH020",
        "the parcel is the wrong colour",
    )
    .await
    .expect("the replacement runs");
    assert_eq!(
        replaced.shots[1].outcome,
        ShotOutcome::Rendered,
        "{}",
        summary(&replaced)
    );
    let after = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(
        items_of(&after, "track_dialogue").len(),
        1,
        "the replacement must not silence the film: {:#?}",
        track_of(&after, "track_dialogue")
    );
    assert_eq!(items_of(&after, "track_ambience").len(), 1);
    assert_eq!(items_of(&after, "track_music").len(), 1);
    assert_eq!(
        audio_job_count(&harness, &project_id).await,
        audio_jobs,
        "re-hydrating adopts the spoken line; it does not speak it again"
    );
    assert_eq!(replaced.sound.len(), record.sound.len());
}

/// The synthesized clip reaches the pack over the TRANSPORT, not off the API host's disk.
///
/// Every other media hop the harness makes is HTTP, and `--api` may legitimately name a
/// private-network address, a `.local` name or a bare hostname — the API host "may be a different
/// machine" (docs/film-harness.md). Reading the worker's WAV straight out of the project directory
/// worked only when that directory happened to be on this filesystem; on any other host every
/// synthesis died with an io error. The local read survives as a FAST PATH for the loopback case,
/// which is the second half of this test.
#[tokio::test]
async fn a_synthesized_clip_is_fetched_over_the_file_route_when_the_project_dir_is_not_local() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |_| {});
    // The API's project directories, as this controller would see them across a network: named,
    // and not there.
    let transport = CountingTransport::remote(
        harness.app.clone(),
        harness.temp_dir.path().join("another-machine"),
    );
    let options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "a remote API host must not break synthesis: {}",
        summary(&record)
    );
    let project_id = record.project_id.clone().expect("project");
    let project_path = record.project_path.clone().expect("project path");
    assert!(
        !Path::new(&project_path).is_dir(),
        "the fast path must genuinely be unavailable for this to mean anything: {project_path}"
    );

    // The file route is what carried it, once, for the one line that was spoken.
    let downloads = transport.downloads.lock().clone();
    assert_eq!(downloads.len(), 1, "{downloads:?}");
    assert!(
        downloads[0].starts_with(&format!(
            "/api/v1/projects/{project_id}/files/assets/audios/"
        )) && downloads[0].ends_with(".wav"),
        "{downloads:?}"
    );

    // And the bytes that arrived are the clip the worker wrote, not an empty or truncated file.
    let line = &record.synthesized_sound[0];
    assert_eq!(line.status, "completed", "{line:#?}");
    let file = line.file.clone().expect("the clip was written");
    let written = std::fs::read(pack.parent().expect("pack dir").join(&file))
        .expect("the clip is in the pack");
    let (hz, seconds) = fake_speech_shape(Some("am_michael"), &line.text);
    assert_eq!(
        written,
        film_harness::fixture_sound_wav(seconds, hz, 9000),
        "the downloaded clip must be the WAV the worker wrote"
    );
    // It was imported and placed exactly as a locally-read clip is.
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(items_of(&saved, "track_dialogue").len(), 1, "{saved}");

    // The loopback half: the same run against an API whose project directory IS readable here
    // downloads nothing, because the fast path has the file.
    let local = Harness::start(true, vec![]).await;
    let local_pack = speech_pack(&local, |_| {});
    let local_transport = CountingTransport::local(local.app.clone());
    let local_record = film_harness::run(
        &local_transport,
        &local.options(local.fixture_plan(), local_pack, Some(&["SH010", "SH020"])),
    )
    .await
    .expect("run completes");
    assert_eq!(
        local_record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&local_record)
    );
    assert!(
        Path::new(local_record.project_path.as_deref().expect("project path")).is_dir(),
        "the loopback case has the project directory right here"
    );
    assert!(
        local_transport.downloads.lock().is_empty(),
        "a local project directory is read directly: {:?}",
        local_transport.downloads.lock()
    );
}

/// A `replace-take --export` whose sound could not be re-hydrated must not re-export.
///
/// `ensure_sound` stopping is what says the session's clip map is SHORT, so the re-assembly is
/// deliberately skipped and the saved timeline keeps the take the human just replaced. Exporting
/// anyway renders a fresh MP4 from that stale timeline — and `run_export` writes
/// `ExportRecord { stale: false }` over the `stale: true` the replacement set, so the record would
/// claim the MP4 is current when it carries exactly the material that was rejected.
#[tokio::test]
async fn a_replacement_whose_sound_cannot_be_rehydrated_does_not_re_export() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |_| {});
    let options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
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
    let before = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let exports_before = harness.export_job_count();
    assert!(!record.export.as_ref().expect("exported").stale);

    // Take the spoken line away from the record and from the pack, so the replacement's
    // `ensure_sound` has to speak it again — and make the TTS lane fail, so it cannot.
    let spoken_file = record.synthesized_sound[0]
        .file
        .clone()
        .expect("the clip was written");
    std::fs::remove_file(pack.parent().expect("pack dir").join(&spoken_file))
        .expect("clip removed");
    harness.edit_run_record(|record| {
        let kept: Vec<Value> = record["sound"]
            .as_array()
            .expect("sound")
            .iter()
            .filter(|clip| clip["role"] != json!("courier_line"))
            .cloned()
            .collect();
        record["sound"] = json!(kept);
        record["synthesizedSound"][0]["status"] = json!("failed");
        record["synthesizedSound"][0]["assetId"] = Value::Null;
        record["synthesizedSound"][0]["file"] = Value::Null;
    });
    harness.script.lock().audio_fails = true;

    let replaced = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH020",
        "the parcel is the wrong colour",
    )
    .await
    .expect("the replacement runs");

    // The take landed, and the run stopped on the line it could not re-speak.
    assert_eq!(
        replaced.shots[1].outcome,
        ShotOutcome::Rendered,
        "{}",
        summary(&replaced)
    );
    let stop = replaced.stop.as_ref().expect("the run stopped");
    assert_eq!(stop.reason, "dialogue_synthesis_failed", "{stop:?}");

    // No re-export, and the record still says the MP4 on disk is stale.
    assert_eq!(
        harness.export_job_count(),
        exports_before,
        "a stale timeline must not be rendered into a fresh MP4"
    );
    let export = replaced
        .export
        .as_ref()
        .expect("the first export is recorded");
    assert!(
        export.stale,
        "the export must stay stale when the timeline it came from was not rewritten: {export:#?}"
    );
    // And the saved timeline really was left alone.
    let after = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(after, before, "the saved timeline must be untouched");
}

/// Re-casting a line is a NEW RUN, not a resume — and inside one run a record that no longer
/// matches the pack drops the clip it made.
///
/// The first half is the guard `resume` / `replace-take` already apply: the pack's bytes are hashed
/// at start and a changed document is refused, so an edited line can never reach a running record.
/// The second half is what makes the refusal safe to rely on: a fresh run of the edited pack speaks
/// the NEW line, and — crucially for a pack that PINS `file`, where the clip's name never changes —
/// the stale `sound[]` entry is dropped rather than re-adopted by the import pass.
#[tokio::test]
async fn re_casting_a_line_is_refused_by_resume_and_spoken_by_a_fresh_run() {
    let harness = Harness::start(true, vec![]).await;
    // A PINNED file, so the re-cast cannot be told apart by the clip's name.
    let pack = speech_pack(&harness, |pack| {
        pack["sound"][0]["file"] = json!("sound/courier_line.wav");
    });
    let options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
        Some(&["SH010", "SH020"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(first.outcome, RunOutcome::Completed, "{}", summary(&first));
    assert_eq!(
        first.synthesized_sound[0].file.as_deref(),
        Some("sound/courier_line.wav")
    );
    assert_eq!(
        first.synthesized_sound[0].text,
        "Delivery. I'll leave it on the bench."
    );

    // Re-cast the line in place, at the same pinned path.
    let text = std::fs::read_to_string(&pack).expect("pack");
    let mut document: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text)).expect("parses");
    document["sound"][0]["text"] = json!("Delivery. It's on the bench, then.");
    std::fs::write(&pack, serde_json::to_string_pretty(&document).unwrap()).unwrap();

    // A resume will not have it: the pack no longer hashes to what the run started from.
    reopen_for_resume(&harness);
    let refusal = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .unwrap_err();
    let HarnessError::Refused(message) = refusal else {
        panic!("expected a refusal, got {refusal}");
    };
    assert!(
        message.contains("reference pack") && message.contains("a new run, not a resume"),
        "{message}"
    );

    // A FRESH run of the edited pack speaks the NEW line into the same pinned path — which is what
    // the refusal above sends the operator to do.
    let mut fresh_options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
        Some(&["SH010", "SH020"]),
    );
    fresh_options.out_dir = harness.temp_dir.path().join("recast-out");
    let second = film_harness::run(&harness.transport, &fresh_options)
        .await
        .expect("the re-cast run completes");
    assert_eq!(
        second.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&second)
    );
    assert_eq!(
        second.synthesized_sound[0].text,
        "Delivery. It's on the bench, then."
    );
    assert_eq!(
        second.synthesized_sound[0].file.as_deref(),
        Some("sound/courier_line.wav"),
        "the pinned path is where the re-cast line is written"
    );
    assert_eq!(
        second
            .sound
            .iter()
            .filter(|clip| clip.role == "courier_line")
            .count(),
        1,
        "exactly one clip per role: {:#?}",
        second.sound
    );
    // The line the film now says is as long as the NEW text, measured off the sequence — a re-cast
    // that had been adopted rather than spoken would still be the 1.5s of the old line.
    let project_id = second.project_id.clone().expect("project");
    let timeline_id = second
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1, "{dialogue:#?}");
    let (_, spoken) = fake_speech_shape(Some("am_michael"), &second.synthesized_sound[0].text);
    assert!(
        close(
            dialogue[0]["timelineEnd"].as_f64().unwrap()
                - dialogue[0]["timelineStart"].as_f64().unwrap(),
            spoken
        ),
        "the placed item is as long as the RE-CAST line ({spoken}s): {}",
        dialogue[0]
    );
}

/// A `dialogue` entry carrying BOTH `text` and `file`: synthesis writes into the PINNED path, and
/// everything downstream treats it as the recorded clip at that path.
///
/// This is how a pack keeps a stable, checkable-in name for a line it means to keep — the derived
/// `sound/<role>.tts-<sha>.wav` name is gitignored precisely because it is an output. The
/// `destination` / `imported` interplay is the part worth an end-to-end test rather than a
/// validator unit test: the adoption on resume reads `record.sound` FIRST (so a pack whose clip has
/// been cleaned away still adopts) and only then falls back to the pinned file being on disk.
#[tokio::test]
async fn a_dialogue_entry_with_both_text_and_file_synthesizes_into_the_pinned_path() {
    let harness = Harness::start(true, vec![]).await;
    let pack = speech_pack(&harness, |pack| {
        pack["sound"][0]["file"] = json!("sound/courier_line.wav");
    });
    let pack_dir = pack.parent().expect("pack dir").to_path_buf();
    let pinned = pack_dir.join("sound/courier_line.wav");
    assert!(!pinned.exists(), "the pinned clip does not exist yet");

    let options = harness.options(
        harness.fixture_plan(),
        pack.clone(),
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

    // Synthesis wrote THERE, not under the derived name.
    let line = &record.synthesized_sound[0];
    assert_eq!(line.file.as_deref(), Some("sound/courier_line.wav"));
    assert!(pinned.is_file(), "{} was not written", pinned.display());
    assert!(
        !pack_dir
            .join(film_harness::synthesized_sound_file(
                "courier_line",
                &line.text_sha256
            ))
            .exists(),
        "a pinned `file` replaces the derived name; it does not write both"
    );
    let (hz, seconds) = fake_speech_shape(Some("am_michael"), &line.text);
    assert_eq!(
        std::fs::read(&pinned).expect("pinned clip"),
        film_harness::fixture_sound_wav(seconds, hz, 9000)
    );

    // And it is the clip the import and the bus use.
    let imported = record
        .sound
        .iter()
        .find(|clip| clip.role == "courier_line")
        .expect("imported");
    assert_eq!(imported.file, "sound/courier_line.wav");
    assert_eq!(imported.kind, "dialogue");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    assert_eq!(items_of(&saved, "track_dialogue").len(), 1, "{saved}");
    assert_eq!(audio_job_count(&harness, &project_id).await, 1);

    // `imported` before `destination`: a resume whose pack directory has been CLEANED still adopts
    // the asset the record names rather than speaking the line a second time.
    std::fs::remove_file(&pinned).expect("clip removed");
    reopen_for_resume(&harness);
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        audio_job_count(&harness, &project_id).await,
        1,
        "the record already names the imported clip; nothing is spoken again"
    );
    assert_eq!(resumed.synthesized_sound.len(), 1);
    assert_eq!(
        resumed
            .sound
            .iter()
            .filter(|clip| clip.role == "courier_line")
            .count(),
        1
    );
    assert!(
        !pinned.exists(),
        "adopting must not re-download the clip into the pack"
    );
}

#[tokio::test]
async fn interrupted_human_operations_keep_same_attempt_scope_and_prior_budget_verdict() {
    for exhausted in [false, true] {
        for repair in [false, true] {
            let harness = Harness::start(true, vec![("SH020", VideoBehavior::Hang)]).await;
            harness.script.lock().running_hook = Some((
                "SH020".to_owned(),
                RunningHook::WriteCancelSentinel(harness.out_dir()),
            ));
            let options = harness.options(
                harness.edited_plan(|_| {}),
                harness.fixture_pack_without_sound(),
                Some(&["SH010", "SH020", "SH030"]),
            );
            let mut before = film_harness::run_with_control(
                &harness.transport,
                &options,
                &RunControl::watching(&harness.out_dir()),
            )
            .await
            .expect("canceled run");
            assert_eq!(before.outcome, RunOutcome::Canceled);
            assert!(before.shot("SH010").unwrap().selected_attempt.is_some());
            assert!(before.shot("SH030").unwrap().attempts.is_empty());
            if exhausted {
                before.elapsed_seconds = before.limits.max_run_seconds as f64;
                before.outcome = RunOutcome::StoppedRunBudget;
                before.stop = Some(sceneworks_core::film_plan::RunStop {
                    reason: "run_budget".to_owned(),
                    detail: "Automatic run budget exhausted".to_owned(),
                    resumable: false,
                });
                std::fs::write(
                    harness.out_dir().join("run.json"),
                    serde_json::to_vec_pretty(&before).unwrap(),
                )
                .unwrap();
            }
            let other_before = other_shots_digest(&before, "SH010");
            let jobs_before = harness.api_video_job_count().await;
            harness.script.lock().behaviors.clear();
            let fault = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
                .on_post_route("/api/v1/video/jobs");
            let mut resumed_options = harness.resume_options();
            resumed_options.export = false;
            let result = if repair {
                film_harness::review::request_repair(
                    &fault,
                    &resumed_options,
                    "SH010",
                    "Fix the take",
                )
                .await
            } else {
                film_harness::replace_take(&fault, &resumed_options, "SH010", "Replace the take")
                    .await
            };
            result.expect_err("controller dies after job dispatch, before receiving its id");
            assert!(fault.fired());
            let interrupted = harness_record(&harness);
            let operation = interrupted
                .active_take_operation
                .as_ref()
                .expect("durable operation before dispatch");
            assert_eq!(
                operation.kind,
                if repair { "repair" } else { "replacement" }
            );
            assert_eq!(operation.attempt, 2);
            assert_eq!(operation.prior_outcome, before.outcome);
            assert_eq!(operation.prior_stop, before.stop);
            assert!(interrupted.shot("SH010").unwrap().attempts[1]
                .job_id
                .is_none());
            assert_eq!(harness.api_video_job_count().await, jobs_before + 1);
            let after = film_harness::resume(&harness.transport, &resumed_options)
                .await
                .expect("adopts bounded human operation");
            assert_eq!(
                harness.api_video_job_count().await,
                jobs_before + 1,
                "recovery cannot mint a new job or resume another shot"
            );
            assert_eq!(after.outcome, before.outcome);
            assert_eq!(after.stop, before.stop);
            // The pre-crash record is in memory and recovery reads JSON. Permit only a single
            // representational ULP; any measurable automatic budget charge still fails.
            assert!(
                (after.elapsed_seconds - before.elapsed_seconds).abs()
                    <= before.elapsed_seconds.abs().max(1.0) * f64::EPSILON,
                "human recovery cannot charge the automatic budget: {} -> {}",
                before.elapsed_seconds,
                after.elapsed_seconds
            );
            assert!(after.human_requested_elapsed_seconds > before.human_requested_elapsed_seconds);
            assert!(after.active_take_operation.is_none());
            let shot = after.shot("SH010").unwrap();
            assert_eq!(shot.attempts.len(), 2);
            assert_eq!(shot.selected_attempt, Some(2));
            assert_eq!(shot.attempts[1].idempotency_key, operation.idempotency_key);
            assert!(shot.attempts[1].job_id.is_some());
            assert_eq!(other_shots_digest(&after, "SH010"), other_before);
        }
    }
}

#[path = "film_action_errors.rs"]
mod action_errors;

/// Hold a successful read at the asynchronous boundary where another process can Cancel.
struct CancelBoundaryTransport {
    inner: RouterTransport,
    needle: &'static str,
    armed: AtomicBool,
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

impl CancelBoundaryTransport {
    fn new(app: axum::Router, needle: &'static str) -> Self {
        Self {
            inner: RouterTransport { app },
            needle,
            armed: AtomicBool::new(true),
            reached: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        }
    }

    async fn cancel_at_barrier(&self, directory: &Path) {
        tokio::time::timeout(Duration::from_secs(30), self.reached.notified())
            .await
            .expect("controller reaches read barrier");
        film_harness::request_cancel(directory).expect("fresh cancellation is accepted");
        self.release.notify_one();
    }
}

impl ApiTransport for CancelBoundaryTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        let hold = request.method == "GET"
            && request.path.contains(self.needle)
            && self.armed.swap(false, Ordering::SeqCst);
        Box::pin(async move {
            let response = self.inner.call(request).await?;
            if hold {
                assert_eq!(response.status, 200);
                self.reached.notify_one();
                self.release.notified().await;
            }
            Ok(response)
        })
    }
    fn get_bytes(&self, path: String) -> BytesTransportFuture<'_> {
        self.inner.get_bytes(path)
    }
}

async fn r15_pending_video(harness: &Harness, delivered: bool) {
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 60, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let mut options = harness.options(plan, pack, Some(&["SH010"]));
    options.export = false;
    let fault = FaultTransport::new(
        harness.app.clone(),
        1,
        if delivered {
            FaultMode::After
        } else {
            FaultMode::Before
        },
    )
    .on_post_route("/video/jobs");
    film_harness::run(&fault, &options)
        .await
        .expect_err("crash at video dispatch");
    assert!(fault.fired());
}

#[tokio::test]
async fn r15_fresh_cancel_during_resume_admission_is_not_consumed() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    r15_pending_video(&harness, false).await;
    let transport = CancelBoundaryTransport::new(harness.app.clone(), "/workers");
    let mut options = harness.resume_options();
    options.export = false;
    options.control = RunControl::watching(&harness.out_dir());
    let directory = harness.out_dir();
    let (result, ()) = tokio::join!(
        film_harness::resume(&transport, &options),
        transport.cancel_at_barrier(&directory),
    );
    let record = result.expect("cancellation settles");
    assert!(
        harness.out_dir().join("cancel.requested").exists(),
        "fresh cancel must survive admission"
    );
    assert_eq!(record.outcome, RunOutcome::Canceled);
    assert_eq!(record.shot("SH010").unwrap().outcome, ShotOutcome::Canceled);
    assert_eq!(harness.api_video_job_count().await, 0);
    assert_eq!(record.shot("SH010").unwrap().attempts.len(), 1);
}

#[tokio::test]
async fn r15_cancel_during_video_lookup_prevents_new_dispatch() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 60, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let mut options = harness.options(plan, pack, Some(&["SH010"]));
    options.export = false;
    let transport = CancelBoundaryTransport::new(harness.app.clone(), "/api/v1/jobs?");
    let control = RunControl::watching(&harness.out_dir());
    let directory = harness.out_dir();
    let (result, ()) = tokio::join!(
        film_harness::run_with_control(&transport, &options, &control),
        transport.cancel_at_barrier(&directory),
    );
    assert_eq!(result.unwrap().outcome, RunOutcome::Canceled);
    assert_eq!(
        harness.api_video_job_count().await,
        0,
        "no new POST after cancellation during lookup"
    );
    let pending = film_harness::read_run_record(&directory).unwrap();
    let key = pending.shot("SH010").unwrap().attempts[0]
        .idempotency_key
        .clone();
    let mut retry = harness.resume_options();
    retry.export = false;
    retry.control = RunControl::watching(&directory);
    let resumed = film_harness::resume(&harness.transport, &retry)
        .await
        .unwrap();
    let shot = resumed.shot("SH010").unwrap();
    assert_eq!(
        shot.attempts.len(),
        1,
        "cancellation does not consume another attempt"
    );
    assert_eq!(shot.selected_attempt, Some(1));
    assert_eq!(shot.attempts[0].idempotency_key, key);
    assert_eq!(harness.api_video_job_count().await, 1);
}

#[tokio::test]
async fn r15_api_accepted_resume_replace_and_repair_preserve_cancel_during_admission() {
    for action in ["resume", "review/replace", "review/repair"] {
        let barrier = Arc::new(CancelBoundaryTransport::new(
            axum::Router::new(),
            "/workers",
        ));
        barrier.armed.store(false, Ordering::SeqCst);
        let harness =
            Harness::start_http_with_cancel_boundary(true, fast(&["SH010"]), Some(barrier.clone()))
                .await;
        let project = harness
            .state
            .project_store
            .create_project("R15 acceptance")
            .unwrap();
        let mut draft = FilmDraft::manual_one_shot(&project.id, "draft_r15", "R15");
        draft.production_plan.shots[0].beat = "A courier crosses the workshop.".to_owned();
        draft.production_plan.shots[0].prompt = "A courier crosses a quiet workshop.".to_owned();
        draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
        harness
            .state
            .project_store
            .create_film_draft_document(&project.id, draft)
            .unwrap();
        harness
            .state
            .project_store
            .create_film_run(
                &project.id,
                "run_r15",
                "draft_r15",
                vec!["SH010".to_owned()],
                None,
            )
            .unwrap();
        let files = harness
            .state
            .project_store
            .film_run_files(&project.id, "run_r15")
            .unwrap();
        let options = RunOptions {
            plan_path: files.plan,
            reference_pack_path: files.reference_pack,
            compiled_path: None,
            project_id: Some(project.id.clone()),
            shot_ids: Some(vec!["SH010".to_owned()]),
            out_dir: files.directory.clone(),
            poll_interval: Duration::from_millis(25),
            export: false,
            require_installed: false,
        };
        if action == "resume" {
            let fault = FaultTransport::new(harness.app.clone(), 1, FaultMode::Before)
                .on_post_route("/video/jobs");
            film_harness::run(&fault, &options).await.unwrap_err();
        } else {
            film_harness::run(&harness.transport, &options)
                .await
                .unwrap();
        }
        let before = film_harness::read_run_record(&files.directory).unwrap();
        let jobs_before = harness.api_video_job_count().await;
        // This old marker belongs to the previous controller, not the newly accepted action.
        film_harness::request_cancel(&files.directory).unwrap();
        barrier.armed.store(true, Ordering::SeqCst);
        let base = format!("/api/v1/projects/{}/film-runs/run_r15", project.id);
        let (status, response) = request(
            harness.app.clone(),
            "POST",
            &format!("{base}/{action}"),
            json!({"shotId":"SH010", "reason":"R15"}),
        )
        .await;
        assert_eq!(
            status,
            axum::http::StatusCode::ACCEPTED,
            "{action}: {response}"
        );
        tokio::time::timeout(Duration::from_secs(30), barrier.reached.notified())
            .await
            .unwrap();
        assert!(
            !files.directory.join("cancel.requested").exists(),
            "explicit acceptance consumed the old marker before admission"
        );
        let (status, response) = request(
            harness.app.clone(),
            "POST",
            &format!("{base}/cancel"),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::ACCEPTED, "{response}");
        barrier.release.notify_one();
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if !film_harness::ControllerLease::is_active(&files.directory).unwrap() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("accepted action settles");
        assert!(
            files.directory.join("cancel.requested").exists(),
            "{action} retained fresh cancel"
        );
        assert_eq!(
            harness.api_video_job_count().await,
            jobs_before,
            "{action} dispatched no new video"
        );
        let after = film_harness::read_run_record(&files.directory).unwrap();
        assert_eq!(
            after.shot("SH010").unwrap().attempts.len(),
            before.shot("SH010").unwrap().attempts.len()
        );
        assert_eq!(
            selected_identity(&after, "SH010"),
            selected_identity(&before, "SH010")
        );
        if action != "resume" {
            assert_eq!(
                film_harness::read_action_operation(&files.directory)
                    .unwrap()
                    .unwrap()
                    .status,
                "canceled"
            );
        }
    }
}

#[tokio::test]
async fn r15_startup_pending_and_fresh_cancel_adopt_existing_video_without_redispatch() {
    for (fresh, exhausted) in [(false, false), (true, false), (false, true)] {
        let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
        r15_pending_video(&harness, true).await;
        if exhausted {
            harness.edit_run_record(|record| record["elapsedSeconds"] = json!(600.0));
        }
        let original = harness
            .jobs()
            .await
            .into_iter()
            .find(|job| job["type"] == "video_generate")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_owned();
        std::fs::write(
            harness.out_dir().join(film_harness::CONTROLLER_LOCK_FILE),
            "owner=api:interrupted\npid=999999\n",
        )
        .unwrap();
        if !fresh {
            film_harness::request_cancel(&harness.out_dir()).unwrap();
        }
        let lease =
            film_harness::ControllerLease::acquire_interrupted(&harness.out_dir(), "api:startup")
                .unwrap()
                .unwrap();
        let mut options = harness.resume_options();
        options.export = false;
        options.control = RunControl::watching(&harness.out_dir());
        let transport = CancelBoundaryTransport::new(harness.app.clone(), "/workers");
        let directory = harness.out_dir();
        let (result, ()) = tokio::join!(
            film_harness::resume_with_lease(&transport, &options, lease),
            async {
                tokio::time::timeout(Duration::from_secs(30), transport.reached.notified())
                    .await
                    .unwrap();
                if fresh {
                    film_harness::request_cancel(&directory).unwrap();
                }
                transport.release.notify_one();
            }
        );
        let record = result.unwrap();
        assert!(directory.join("cancel.requested").exists());
        assert_eq!(record.outcome, RunOutcome::Canceled);
        let attempt = &record.shot("SH010").unwrap().attempts[0];
        assert_eq!(attempt.job_id.as_deref(), Some(original.as_str()));
        assert_eq!(attempt.status, "canceled_by_operator");
        assert_eq!(harness.api_video_job_count().await, 1);
        if exhausted {
            let resumed = film_harness::resume(&harness.transport, &options)
                .await
                .unwrap();
            assert_eq!(resumed.outcome, RunOutcome::StoppedRunBudget);
            assert_eq!(
                harness.api_video_job_count().await,
                1,
                "cancellation cannot replenish the spent budget"
            );
        }
    }
}

#[tokio::test]
async fn r15_active_take_resume_retains_cancel_and_authorized_jobless_attempt() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    r15_pending_video(&harness, false).await;
    let mut options = harness.resume_options();
    options.export = false;
    film_harness::resume(&harness.transport, &options)
        .await
        .unwrap();
    let fault =
        FaultTransport::new(harness.app.clone(), 1, FaultMode::Before).on_post_route("/video/jobs");
    film_harness::replace_take(&fault, &options, "SH010", "repair framing")
        .await
        .unwrap_err();
    let before = film_harness::read_run_record(&harness.out_dir()).unwrap();
    let transport = CancelBoundaryTransport::new(harness.app.clone(), "/workers");
    options.control = RunControl::watching(&harness.out_dir());
    let directory = harness.out_dir();
    let (result, ()) = tokio::join!(
        film_harness::resume(&transport, &options),
        transport.cancel_at_barrier(&directory)
    );
    let record = result.unwrap();
    assert_eq!(record.outcome, RunOutcome::Canceled);
    assert!(directory.join("cancel.requested").exists());
    assert_eq!(record.active_take_operation, before.active_take_operation);
    assert_eq!(record.shot("SH010").unwrap().attempts.len(), 2);
    assert_eq!(harness.api_video_job_count().await, 1);
    let mut retry = harness.resume_options();
    retry.export = false;
    let resumed = film_harness::resume(&harness.transport, &retry)
        .await
        .unwrap();
    assert_eq!(resumed.shot("SH010").unwrap().selected_attempt, Some(2));
    assert_eq!(harness.api_video_job_count().await, 2);
}

#[tokio::test]
async fn r15_video_lookup_adopts_and_cancels_existing_job() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    r15_pending_video(&harness, true).await;
    let id = harness.jobs().await[0]["id"].as_str().unwrap().to_owned();
    let transport = CancelBoundaryTransport::new(harness.app.clone(), "/api/v1/jobs?");
    let mut options = harness.resume_options();
    options.export = false;
    options.control = RunControl::watching(&harness.out_dir());
    let directory = harness.out_dir();
    let (result, ()) = tokio::join!(
        film_harness::resume(&transport, &options),
        transport.cancel_at_barrier(&directory)
    );
    let record = result.unwrap();
    assert_eq!(harness.api_video_job_count().await, 1);
    let attempt = &record.shot("SH010").unwrap().attempts[0];
    assert_eq!(attempt.job_id.as_deref(), Some(id.as_str()));
    assert_eq!(attempt.status, "canceled_by_operator");
    assert_eq!(record.outcome, RunOutcome::Canceled);
}

#[tokio::test]
async fn r15_audio_lookup_cancel_prevents_dispatch_or_cancels_exact_existing_job() {
    for existing in [false, true] {
        let harness = Harness::start(true, fast(&["SH010"])).await;
        harness.script.lock().audio_hangs = true;
        let mut options = harness.options(
            harness.fixture_plan(),
            speech_pack(&harness, |_| {}),
            Some(&["SH020"]),
        );
        options.export = false;
        if existing {
            let fault = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
                .on_post_route("/audio/jobs");
            film_harness::run(&fault, &options).await.unwrap_err();
            assert!(fault.fired());
        }
        let original = harness
            .jobs()
            .await
            .into_iter()
            .find(|job| job["type"] == "audio_generate")
            .map(|job| job["id"].as_str().unwrap().to_owned());
        let transport = CancelBoundaryTransport::new(harness.app.clone(), "/api/v1/jobs?");
        let directory = harness.out_dir();
        let control = RunControl::watching(&directory);
        let mut resume = harness.resume_options();
        resume.export = false;
        resume.control = control.clone();
        let (result, ()) = tokio::join!(
            async {
                if existing {
                    film_harness::resume(&transport, &resume).await
                } else {
                    film_harness::run_with_control(&transport, &options, &control).await
                }
            },
            transport.cancel_at_barrier(&directory)
        );
        let record = result.unwrap();
        assert_eq!(record.outcome, RunOutcome::Canceled);
        assert_eq!(
            audio_job_count(&harness, record.project_id.as_deref().unwrap()).await,
            usize::from(existing)
        );
        assert_eq!(harness.api_video_job_count().await, 0);
        assert_eq!(record.synthesized_sound.len(), 1);
        let line = &record.synthesized_sound[0];
        assert_eq!(line.job_id, original);
        assert_eq!(line.attempt, 1);
        if existing {
            assert_eq!(line.status, "canceled_by_operator");
        } else {
            assert!(line.error.as_ref().unwrap().contains("canceled"));
        }
    }
}

#[tokio::test]
async fn r15_both_export_lookup_routes_cancel_without_new_dispatch_and_adopt_existing() {
    for explicit in [false, true] {
        for existing in [false, true] {
            let harness = Harness::start(true, fast(&["SH010"])).await;
            r15_pending_video(&harness, false).await;
            let mut setup = harness.resume_options();
            setup.export = false;
            film_harness::resume(&harness.transport, &setup)
                .await
                .unwrap();
            harness.script.lock().export_hangs = true;
            let fault = FaultTransport::new(
                harness.app.clone(),
                1,
                if existing {
                    FaultMode::After
                } else {
                    FaultMode::Before
                },
            )
            .on_post_route("/exports");
            // Persist a real export intention, with either a lost response or no delivered POST.
            film_harness::start_explicit_export(&fault, &setup)
                .await
                .unwrap_err();
            assert!(fault.fired());
            let original = harness
                .jobs()
                .await
                .into_iter()
                .find(|job| job["type"] == "timeline_export")
                .map(|job| job["id"].as_str().unwrap().to_owned());
            if !explicit {
                harness.edit_run_record(|record| {
                    record["state"] = json!("running");
                    record["outcome"] = json!("failed");
                    record["stop"] = json!({"reason":"export_failed", "detail":"interrupted export", "resumable":true});
                });
            }
            let transport = CancelBoundaryTransport::new(harness.app.clone(), "/api/v1/jobs?");
            let mut options = harness.resume_options();
            options.control = RunControl::watching(&harness.out_dir());
            let directory = harness.out_dir();
            let (result, ()) = tokio::join!(
                async {
                    if explicit {
                        let _lease = film_harness::ControllerLease::acquire_new_action(
                            &directory,
                            "export-r15",
                        )?;
                        let (_, task) =
                            film_harness::start_explicit_export(&transport, &options).await?;
                        film_harness::finish_explicit_export(&transport, &options, &task).await
                    } else {
                        film_harness::resume(&transport, &options).await
                    }
                },
                transport.cancel_at_barrier(&directory)
            );
            assert_eq!(
                harness
                    .jobs()
                    .await
                    .iter()
                    .filter(|job| job["type"] == "timeline_export")
                    .count(),
                usize::from(existing)
            );
            assert_eq!(harness.api_video_job_count().await, 1);
            if explicit && !existing {
                assert!(matches!(result, Err(HarnessError::Canceled(_))));
                assert_eq!(
                    film_harness::read_action_operation(&directory)
                        .unwrap()
                        .unwrap()
                        .status,
                    "canceled"
                );
                assert!(film_harness::read_run_record(&directory)
                    .unwrap()
                    .export_pending
                    .is_some());
            } else {
                let record = result.unwrap();
                if existing {
                    let export = record.export.unwrap();
                    assert_eq!(Some(export.job_id), original);
                    assert_eq!(export.status, "canceled_by_operator");
                } else {
                    assert_eq!(record.outcome, RunOutcome::Canceled);
                }
            }
        }
    }
}

#[tokio::test]
async fn r15_explicit_replace_and_repair_consume_only_previous_cancellation() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    r15_pending_video(&harness, false).await;
    let mut options = harness.resume_options();
    options.export = false;
    film_harness::resume(&harness.transport, &options)
        .await
        .unwrap();
    for repair in [false, true] {
        film_harness::request_cancel(&harness.out_dir()).unwrap();
        options.control = RunControl::watching(&harness.out_dir());
        let record = if repair {
            film_harness::review::request_repair(&harness.transport, &options, "SH010", "repair")
                .await
        } else {
            film_harness::replace_take(&harness.transport, &options, "SH010", "replace").await
        }
        .unwrap();
        assert!(!harness.out_dir().join("cancel.requested").exists());
        assert_eq!(
            record.shot("SH010").unwrap().selected_attempt,
            Some(if repair { 3 } else { 2 })
        );
        assert_eq!(record.shot("SH010").unwrap().automatic_attempts(), 1);
    }
}

// -------------------------------------------------------------------------------------------
// sc-24029 — the feature-end review's findings
// -------------------------------------------------------------------------------------------

/// The shipped fixture plan, pack and reference plates copied somewhere editable, with a compiled
/// document beside them that is current against both.
///
/// Returns the run options the CLI's `validate` takes, so the test edits a DOCUMENT and asks the
/// real subcommand — the path a person is on when they change a pack by hand.
fn editable_fixture_with_compiled(temp: &Path) -> RunOptions {
    let plan_path = temp.join("plan.jsonc");
    let pack_path = temp.join("references.jsonc");
    std::fs::copy(Path::new(FIXTURE_DIR).join("plan.jsonc"), &plan_path).unwrap();
    std::fs::copy(Path::new(FIXTURE_DIR).join("references.jsonc"), &pack_path).unwrap();
    // The plates AND the sound beds: `validate_all` is given the pack's own directory, so every
    // file the pack names has to be beside it or the refusal is about missing media rather than
    // about the compiled document.
    for directory in ["references", "sound"] {
        std::fs::create_dir_all(temp.join(directory)).unwrap();
        for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join(directory)).unwrap() {
            let entry = entry.unwrap();
            if entry.path().is_file() {
                std::fs::copy(entry.path(), temp.join(directory).join(entry.file_name())).unwrap();
            }
        }
    }

    let plan_bytes = std::fs::read(&plan_path).unwrap();
    let plan = sceneworks_core::film_plan::parse_plan_document(
        &String::from_utf8(plan_bytes.clone()).unwrap(),
    )
    .expect("the fixture plan parses");
    let pack = sceneworks_core::film_plan::parse_reference_pack(
        &std::fs::read_to_string(&pack_path).unwrap(),
    )
    .expect("the fixture pack parses");
    let requests: Vec<Value> = plan
        .shots
        .iter()
        .map(|shot| {
            json!({
                "shotId": shot.id,
                "beat": shot.beat,
                "mode": shot.conditioning.mode,
                "model": plan.model.id,
                "prompt": shot.prompt,
                "promptSource": "authored",
                "durationSeconds": shot.target_duration_seconds,
                "fps": 24,
                "width": 576,
                "height": 320
            })
        })
        .collect();
    let compiled = json!({
        "schemaVersion": sceneworks_core::film_compile::COMPILED_PLAN_SCHEMA_VERSION,
        "planId": plan.id,
        "planVersion": plan.version,
        // The CLI hashes the plan FILE's bytes, which is the identity `validate` recomputes.
        "planSha256": crate::film_harness::sha256_hex(&plan_bytes),
        "referencePackId": pack.id,
        "referencePackVersion": pack.version,
        // The pack's identity is the PARSED pack, which is what makes the comment-only edit below
        // a non-event and the description edit a real one.
        "referencePackSha256":
            sceneworks_core::film_compile::reference_pack_sha256(&pack).expect("the pack hashes"),
        "compiledAt": "2026-09-19T00:00:00Z",
        "model": {"id": plan.model.id, "tier": plan.model.tier, "fps": 24, "lane": "mlx"},
        "requests": requests
    });
    let compiled_path = temp.join("compiled.json");
    std::fs::write(
        &compiled_path,
        serde_json::to_vec_pretty(&compiled).unwrap(),
    )
    .unwrap();

    RunOptions {
        plan_path,
        reference_pack_path: pack_path,
        compiled_path: Some(compiled_path),
        project_id: None,
        shot_ids: None,
        out_dir: temp.join("out"),
        poll_interval: Duration::from_millis(10),
        export: false,
        require_installed: false,
    }
}

/// sc-24029, E5/E7. From the CLI too: a compiled document is refused once the pack DOCUMENT's
/// description changes, and is NOT refused by a comment-only edit of the same file.
///
/// The pair is the whole point of hashing the parsed pack rather than the file's bytes. The CLI
/// reads a JSONC document whose comments and spacing belong to its author, while the workspace
/// holds a typed pack that was never a file; a byte hash would give one pack two identities and
/// stale a compile every time somebody wrote a note in it.
#[tokio::test]
async fn a_pack_description_edit_stales_the_cli_compiled_document_and_a_comment_does_not() {
    let temp = tempfile::tempdir().unwrap();
    let options = editable_fixture_with_compiled(temp.path());

    film_harness::validate(None, &options)
        .await
        .expect("the untouched documents validate");

    let original = std::fs::read_to_string(&options.reference_pack_path).unwrap();

    // A COMMENT ONLY. Nothing the compiler reads has changed.
    std::fs::write(
        &options.reference_pack_path,
        format!("// A note from whoever owns this pack.\n{original}"),
    )
    .unwrap();
    film_harness::validate(None, &options)
        .await
        .expect("a comment is not a pack change");

    // A DESCRIPTION. This is text the compiler repeats into every prompt that names the role.
    assert!(original.contains("door camera-left."));
    std::fs::write(
        &options.reference_pack_path,
        original.replace("door camera-left.", "door camera-right."),
    )
    .unwrap();
    let error = film_harness::validate(None, &options)
        .await
        .expect_err("an edited description stales the compile");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings.iter().any(|finding| finding
            .message
            .contains("the reference pack changed since these requests were compiled")
            && finding
                .message
                .contains("recompile, or use authored prompts")),
        "{findings:?}"
    );
}
