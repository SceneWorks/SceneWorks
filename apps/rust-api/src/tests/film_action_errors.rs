use super::*;
use axum::http::StatusCode;

fn located_options(harness: &Harness) -> (String, String, RunOptions) {
    let project = harness
        .state
        .project_store
        .create_project("Action failures")
        .unwrap();
    let draft_id = "film_action_errors";
    let mut draft = FilmDraft::manual_one_shot(&project.id, draft_id, "Action failures");
    draft.production_plan.shots[0].beat = "Courier enters".to_owned();
    draft.production_plan.shots[0].prompt = "A courier enters a quiet workshop.".to_owned();
    draft.production_plan.shots[0].audio = "Room tone. No music.".to_owned();
    harness
        .state
        .project_store
        .create_film_draft_document(&project.id, draft)
        .unwrap();
    let run_id = "filmrun_action_errors";
    harness
        .state
        .project_store
        .create_film_run(
            &project.id,
            run_id,
            draft_id,
            vec!["SH010".to_owned()],
            None,
        )
        .unwrap();
    let files = harness
        .state
        .project_store
        .film_run_files(&project.id, run_id)
        .unwrap();
    let options = RunOptions {
        plan_path: files.plan,
        reference_pack_path: files.reference_pack,
        compiled_path: None,
        project_id: Some(project.id.clone()),
        shot_ids: Some(vec!["SH010".to_owned()]),
        out_dir: files.directory,
        poll_interval: Duration::from_millis(20),
        export: false,
        require_installed: false,
    };
    (project.id, run_id.to_owned(), options)
}

async fn settled_view(harness: &Harness, route: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let (status, view) = request(harness.app.clone(), "GET", route, Value::Null).await;
            assert_eq!(status, StatusCode::OK, "{view}");
            if view["controllerActive"] == false {
                return view;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("accepted action releases its lease")
}

#[tokio::test]
async fn accepted_replace_repair_and_resume_failures_survive_reload_and_allow_explicit_retry() {
    let harness = Harness::start_http(true, vec![]).await;
    let (project_id, run_id, options) = located_options(&harness);
    let completed = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(completed.outcome, RunOutcome::Completed);
    let route = format!("/api/v1/projects/{project_id}/film-runs/{run_id}");
    let mut resume = ResumeOptions::new(options.out_dir.clone());
    resume.require_installed = false;
    resume.export = false;
    resume.poll_interval = Duration::from_millis(20);
    for action in ["replacement", "repair", "resume"] {
        let mut before = film_harness::read_run_record(&options.out_dir).unwrap();
        if action == "resume" {
            before.outcome = RunOutcome::Canceled;
            before.stop = Some(sceneworks_core::film_plan::RunStop {
                reason: "canceled_by_operator".to_owned(),
                detail: "Operator canceled".to_owned(),
                resumable: true,
            });
            std::fs::write(
                options.out_dir.join(film_harness::RUN_RECORD_FILE),
                serde_json::to_vec_pretty(&before).unwrap(),
            )
            .unwrap();
        }
        register_fake_worker(&harness.app, &["frame_extract"]).await;
        let jobs_before = harness.api_video_job_count().await;
        let action_route = match action {
            "replacement" => "review/replace",
            "repair" => "review/repair",
            _ => "resume",
        };
        let (status, accepted) = request(
            harness.app.clone(),
            "POST",
            &format!("{route}/{action_route}"),
            json!({"shotId":"SH010", "reason":"try again"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
        let view = settled_view(&harness, &route).await;
        assert_eq!(view["actionOperation"]["status"], "failed", "{view}");
        assert_eq!(view["actionOperation"]["action"], action);
        assert!(
            view["actionOperation"]["detail"]
                .as_str()
                .unwrap()
                .contains("not installed"),
            "{view}"
        );
        let after = film_harness::read_run_record(&options.out_dir).unwrap();
        assert_eq!(after.outcome, before.outcome);
        assert_eq!(
            serde_json::to_value(&after.stop).unwrap(),
            serde_json::to_value(&before.stop).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&after.shots).unwrap(),
            serde_json::to_value(&before.shots).unwrap()
        );
        assert_eq!(after.elapsed_seconds, before.elapsed_seconds);
        assert_eq!(
            after.human_requested_elapsed_seconds,
            before.human_requested_elapsed_seconds
        );
        assert_eq!(harness.api_video_job_count().await, jobs_before);
        let (_, reloaded) = request(
            harness.app.clone(),
            "GET",
            &format!("{route}/review"),
            Value::Null,
        )
        .await;
        assert_eq!(reloaded["run"]["actionOperation"], view["actionOperation"]);
        // With only the fake's installation check disabled, the independently unavailable
        // worker is refused too, before spending any attempt. The same receipt is served by GET.
        let missing_worker = match action {
            "replacement" => {
                film_harness::replace_take(&harness.transport, &resume, "SH010", "retry").await
            }
            "repair" => {
                film_harness::review::request_repair(&harness.transport, &resume, "SH010", "retry")
                    .await
            }
            _ => film_harness::resume(&harness.transport, &resume).await,
        }
        .unwrap_err();
        assert!(
            missing_worker.to_string().contains("video_generate"),
            "{missing_worker}"
        );
        let reloaded = settled_view(&harness, &route).await;
        assert!(reloaded["actionOperation"]["detail"]
            .as_str()
            .unwrap()
            .contains("video_generate"));
        assert_eq!(harness.api_video_job_count().await, jobs_before);
        register_fake_worker(&harness.app, FAKE_CAPABILITIES).await;
        // The fake owns no model weights. The shared API/CLI controller retries with only that
        // installation check disabled; admission and all dispatch/reconciliation routes stay real.
        match action {
            "replacement" => {
                film_harness::replace_take(&harness.transport, &resume, "SH010", "restored").await
            }
            "repair" => {
                film_harness::review::request_repair(
                    &harness.transport,
                    &resume,
                    "SH010",
                    "restored",
                )
                .await
            }
            _ => film_harness::resume(&harness.transport, &resume).await,
        }
        .unwrap();
        assert_eq!(
            film_harness::read_action_operation(&options.out_dir)
                .unwrap()
                .unwrap()
                .status,
            "completed"
        );
        assert_eq!(
            harness.api_video_job_count().await,
            jobs_before + usize::from(action != "resume")
        );
    }
}

#[tokio::test]
async fn accepted_start_transport_failure_persists_without_a_run_or_dispatch_and_can_retry() {
    let mut harness = Harness::start_http(true, vec![]).await;
    let (project_id, run_id, options) = located_options(&harness);
    let server = harness.server.take().unwrap();
    server.abort();
    let _ = server.await;
    let route = format!("/api/v1/projects/{project_id}/film-runs/{run_id}");
    let (status, body) = request(
        harness.app.clone(),
        "POST",
        &format!("{route}/start"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let view = settled_view(&harness, &route).await;
    assert!(view.get("record").is_none(), "{view}");
    assert_eq!(view["actionOperation"]["action"], "start");
    assert_eq!(view["actionOperation"]["status"], "failed");
    assert!(
        view["actionOperation"]["detail"]
            .as_str()
            .unwrap()
            .contains("transport"),
        "{view}"
    );
    assert_eq!(harness.api_video_job_count().await, 0);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(record.outcome, RunOutcome::Completed);
    assert_eq!(harness.api_video_job_count().await, 1);
    assert_eq!(
        film_harness::read_action_operation(&options.out_dir)
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
}

#[tokio::test]
async fn startup_preflight_failure_preserves_saved_jobs_and_can_be_resumed_explicitly() {
    let harness = Harness::start_http(true, vec![]).await;
    let (project_id, run_id, options) = located_options(&harness);
    let mut before = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    before.state = RunState::Running;
    std::fs::write(
        options.out_dir.join(film_harness::RUN_RECORD_FILE),
        serde_json::to_vec_pretty(&before).unwrap(),
    )
    .unwrap();
    let before = film_harness::read_run_record(&options.out_dir).unwrap();
    std::fs::write(
        options.out_dir.join(film_harness::CONTROLLER_LOCK_FILE),
        "owner=api:dead\npid=999999\n",
    )
    .unwrap();
    register_fake_worker(&harness.app, &["frame_extract"]).await;
    let mut startup = crate::film_lifecycle::spawn_film_startup_reconciliation_for_fake_worker(
        harness.state.clone(),
    );
    startup.scan.await.unwrap();
    let error = startup
        .controller_results
        .recv()
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.contains("video_generate"), "{error}");
    let route = format!("/api/v1/projects/{project_id}/film-runs/{run_id}");
    let view = settled_view(&harness, &route).await;
    assert_eq!(view["actionOperation"]["status"], "failed");
    let after = film_harness::read_run_record(&options.out_dir).unwrap();
    assert_eq!(
        serde_json::to_value(&after).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
    assert_eq!(harness.api_video_job_count().await, 1);
    register_fake_worker(&harness.app, FAKE_CAPABILITIES).await;
    let mut resume = ResumeOptions::new(options.out_dir.clone());
    resume.require_installed = false;
    resume.export = false;
    let record = film_harness::resume(&harness.transport, &resume)
        .await
        .unwrap();
    assert_eq!(record.outcome, RunOutcome::Completed);
    assert_eq!(harness.api_video_job_count().await, 1);
}

#[tokio::test]
async fn missing_model_and_changed_compiled_requests_persist_action_refusals_without_attempts() {
    let harness = Harness::start(true, vec![]).await;
    let (_, _, options) = located_options(&harness);
    film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    let before = film_harness::read_run_record(&options.out_dir).unwrap();
    let missing_model =
        ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/models", |body| {
            *body = json!([])
        });
    let mut resume = ResumeOptions::new(options.out_dir.clone());
    resume.require_installed = false;
    resume.export = false;
    for action in ["replacement", "repair", "resume"] {
        let result = match action {
            "replacement" => {
                film_harness::replace_take(&missing_model, &resume, "SH010", "retry").await
            }
            "repair" => {
                film_harness::review::request_repair(&missing_model, &resume, "SH010", "retry")
                    .await
            }
            _ => film_harness::resume(&missing_model, &resume).await,
        };
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not in this API's model catalog"));
        let receipt = film_harness::read_action_operation(&options.out_dir)
            .unwrap()
            .unwrap();
        assert_eq!(receipt.action, action);
        assert_eq!(receipt.status, "failed");
        let after = film_harness::read_run_record(&options.out_dir).unwrap();
        assert_eq!(after.outcome, before.outcome);
        assert_eq!(
            serde_json::to_value(&after.shots).unwrap(),
            serde_json::to_value(&before.shots).unwrap()
        );
        assert_eq!(harness.api_video_job_count().await, 1);
    }
    std::fs::write(options.out_dir.join("compiled.json"), b"{}").unwrap();
    let error = film_harness::replace_take(&harness.transport, &resume, "SH010", "retry")
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("compiled requests changed"),
        "{error}"
    );
    let receipt = film_harness::read_action_operation(&options.out_dir)
        .unwrap()
        .unwrap();
    assert!(receipt
        .detail
        .unwrap()
        .contains("compiled requests changed"));
    assert_eq!(harness.api_video_job_count().await, 1);
}
