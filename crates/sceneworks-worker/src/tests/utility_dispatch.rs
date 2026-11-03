/// sc-3513: the worker's `JobType::ImageEdit` dispatch arm delegates to
/// `run_image_generate_job` — the engine keys edits on payload model+mode, not job
/// type. Feeding an `image_edit`-typed job into the handler proves it reaches the image
/// pipeline (stopping at the payload's projectId guard) rather than the `run_utility_job`
/// "unsupported job type" default — i.e. plain Image Edit is genuinely handled. The
/// handler never reads `job_type`, so a missing projectId is its first stop (no network).
#[tokio::test]
async fn image_edit_job_dispatches_to_image_generate_handler() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-image-edit-1",
        "type": "image_edit",
        "status": "preparing",
        "projectId": null,
        "projectName": null,
        "payload": { "model": "qwen_image_edit_2511", "mode": "edit_image" },
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": null,
        "progress": 0,
        "stage": "preparing",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-06-07T00:00:00Z",
        "updatedAt": "2026-06-07T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("image_edit job snapshot deserializes");

    let error = super::image_jobs::run_image_generate_job(&api, &settings, &job)
        .await
        .expect_err("missing projectId is rejected by the image handler");
    assert!(
        matches!(&error, WorkerError::InvalidPayload(message) if message.contains("projectId")),
        "expected a projectId payload error from the image handler, got {error:?}",
    );
}

/// sc-8845 (F-043) — a process shutdown mid-job must NOT drop the in-flight future without a
/// terminal write. When the shared shutdown `CancelFlag` is already tripped, `run_placeholder_job`
/// must post a terminal `Canceled` progress update (status `canceled`) and return
/// `WorkerError::Canceled` — a prompt, specific terminal state — instead of running on or leaving
/// the job `running` for the 90s stale sweep to relabel `interrupted`. It must do so even with the
/// user-cancel GET reporting NOT canceled, proving the shutdown flag (not a user cancel) is the
/// trigger. Discriminator: under the old behavior (no shutdown checkpoint) the job would run its
/// first stage and post a `preparing`/`running` update instead of `canceled`.
#[tokio::test]
async fn run_placeholder_job_posts_terminal_canceled_on_shutdown_flag() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let job = placeholder_job_snapshot();

    // Process shutdown already observed by the loop supervisor before the first stage runs.
    let shutdown = gen_core::CancelFlag::new();
    shutdown.cancel();

    let result = super::run_placeholder_job(&api, &settings, &job, &shutdown).await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(_))),
        "a tripped shutdown flag must surface as WorkerError::Canceled, not a completed/failed job"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one progress write is posted — the terminal Canceled — with no work stages first"
    );
    assert_eq!(
        posts[0]["status"], "canceled",
        "the shutdown-during-job write must be the TERMINAL Canceled state (not left running)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .contains("shut down"),
        "the terminal message should attribute the cancel to worker shutdown, not a user cancel"
    );
}

/// sc-8845 (F-043) — control: an UN-tripped shutdown flag must not spuriously cancel a clean run.
/// The placeholder job should proceed past its first stage (posting a non-terminal `preparing`
/// update) when neither a user cancel nor a shutdown is present, so the new shutdown checkpoint
/// can't false-positive a normal job to `canceled`.
#[tokio::test]
async fn run_placeholder_job_proceeds_when_shutdown_flag_untripped() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let job = placeholder_job_snapshot();

    let shutdown = gen_core::CancelFlag::new(); // never tripped

    // Drive only the first stage: the placeholder loop sleeps 1.5s per stage, so a short timeout
    // lets us observe the first non-terminal write without waiting out the whole job.
    let run = super::run_placeholder_job(&api, &settings, &job, &shutdown);
    let _ = tokio::time::timeout(Duration::from_millis(400), run).await;

    let posts = posts.lock().expect("posts lock");
    assert!(
        !posts.is_empty(),
        "a job with no cancel and no shutdown must make progress"
    );
    assert_ne!(
        posts[0]["status"], "canceled",
        "an untripped shutdown flag must NOT cancel a clean run — the first write is a work stage"
    );
}

