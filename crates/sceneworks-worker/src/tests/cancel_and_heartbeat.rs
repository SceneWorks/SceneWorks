
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

/// sc-5516 — `begin_video_cancel` trips the engine cancel flag and posts the
/// cancel acknowledgement as a NON-terminal `running` "Cancelling…" update. The
/// terminal `Canceled` (which frees the worker row) is posted by `generate_video`
/// only after the blocking generation actually stops, so the next queued job waits
/// until the GPU is genuinely free.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn begin_video_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::video_jobs::begin_video_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_video_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal — the terminal Canceled is \
         deferred to actual-stop (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

/// sc-5516 — the training sibling of the above: `begin_training_cancel` trips the
/// flag and acknowledges with a NON-terminal `running` update; the terminal
/// `Canceled` is posted by `consume_training_events` after training stops. Compiled on the macOS MLX
/// path AND the off-Mac candle training lane (sc-7817), where `begin_training_cancel` is also linked.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn begin_training_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::training_jobs::begin_training_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_training_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

// sc-9646 (sc-9595 follow-up) — the streaming SeedVR2 upscale writes the whole upscaled PNG sequence
// to a scratch dir before the final encode, so its disk footprint is unbounded after the sc-8829
// host-RAM cap was removed. `check_seedvr2_output_disk` is a fail-loud-before-decode preflight that
// mirrors the removed RAM guard: estimate the output footprint, compare against a generous fraction
// of the scratch volume's free space, and reject a clip that would fill the disk.

#[test]
fn seedvr2_output_bytes_scales_with_frames_and_pixels() {
    // Raw RGB8 upper bound: 1 frame at 100×100 = 100·100·3 = 30_000 bytes.
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(1, 100, 100),
        30_000
    );
    // Linear in frame count and in each dimension.
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(10, 100, 100),
        300_000
    );
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(1, 200, 100),
        60_000
    );
    // A degenerate zero dimension / frame count yields zero (the guard treats that as "unknown").
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(0, 3840, 2160),
        0
    );
    assert_eq!(
        crate::video_jobs::seedvr2_estimated_output_bytes(100, 0, 2160),
        0
    );
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_disk_guard_passes_a_tiny_clip_and_is_noop_when_unknown() {
    let dir = tempdir().expect("tempdir");
    // A few small frames fit any real scratch volume — must pass.
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 8, 64, 64)
        .expect("a tiny output footprint fits available disk");
    // Unknown frame count / dims short-circuit to Ok (the guard defers to the real count).
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 0, 3840, 2160)
        .expect("zero frames is a no-op guard");
    crate::video_jobs::check_seedvr2_output_disk(dir.path(), 1_000, 0, 0)
        .expect("zero dims is a no-op guard");
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_disk_guard_rejects_an_impossibly_large_clip() {
    let dir = tempdir().expect("tempdir");
    // 100M frames at 4096×4096 ≈ 4.7 exabytes of PNG — no scratch volume holds that, so the guard
    // must fail loud with an actionable message (unless the free-space probe is unavailable, in which
    // case the guard is a documented no-op and we skip the assertion).
    if crate::video_jobs::available_disk_bytes(dir.path()).is_some() {
        let error =
            crate::video_jobs::check_seedvr2_output_disk(dir.path(), 100_000_000, 4096, 4096)
                .expect_err("an exabyte-scale output must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("Not enough disk space"),
            "the error names the disk-space limit: {message}"
        );
        assert!(
            message.contains("Trim the clip"),
            "the error is actionable (suggests trimming): {message}"
        );
    }
}

// sc-8917 (F-115) — the shared batched-analysis scaffold DEFERS the terminal `Canceled` to
// actual-stop, exactly like the training/video pollers. A cancel observed on the heartbeat-tick
// poll must only trip the engine flag (so the still-running embed loop bails at its next per-item
// check); the terminal `Canceled` is posted only AFTER the blocking task has stopped, so the worker
// row isn't freed — and the scheduler isn't told a busy worker is free — while the GPU is still
// finishing the current embed. Regression: the old scaffold called `check_cancel` on the tick, which
// posted the terminal `Canceled` at acknowledgement time.

/// A capture stub for the analysis scaffold: the job GET/heartbeat report a user cancel (so the
/// interval-tick poll trips), every progress POST body is recorded, and the heartbeat route answers.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn spawn_analysis_cancel_stub() -> (String, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::sync::{Arc, Mutex};
    type Posts = Arc<Mutex<Vec<Value>>>;
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        // Report the user cancel so `cancel_requested_peek` on the first (immediate) interval tick
        // trips the flag.
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(posts): State<Posts>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(body): Json<Value>,
    ) -> Response {
        posts.lock().expect("posts lock").push(body);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn heartbeat_route(
        axum::extract::Path(worker_id): axum::extract::Path<String>,
    ) -> Response {
        Json(worker_snapshot_json(&worker_id)).into_response()
    }
    let posts: Posts = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    (format!("http://{address}"), posts)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
async fn batched_analysis_defers_terminal_canceled_until_the_task_stops() {
    let (base_url, posts) = spawn_analysis_cancel_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    // A short heartbeat so the interval fires promptly; the first `interval.tick()` resolves
    // immediately regardless, which is what drives the cancel poll here.
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(job_snapshot_json("job-1", true))
        .expect("job snapshot deserializes");

    let cancel = gen_core::CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<usize>(4);
    // The blocking task models a real embed loop: it sends NO item (so `rx.recv()` stays pending and
    // the immediate interval tick wins the `select!`), waits for the flag, and — once tripped —
    // returns `Canceled` exactly as the CLIP/face loops do at their per-item `is_cancelled()` check.
    let task_cancel = cancel.clone();
    let blocking = tokio::task::spawn_blocking(move || -> super::WorkerResult<Vec<u32>> {
        // Keep `tx` alive until we bail so the channel isn't closed early (which would also break the
        // loop, but via the `None` arm rather than the cancel path we want to exercise).
        let _tx = tx;
        for _ in 0..2_000 {
            if task_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "Analysis canceled by user.".to_owned(),
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(vec![1, 2, 3])
    });

    let cfg = super::AnalysisJobConfig {
        endpoint_suffix: "analysis-embeddings",
        space: "clip-vit-l14",
        cancel_message: "Analysis canceled by user.",
        saving_message: "Saving embeddings.",
        join_error_label: "analysis task join",
        item_message: &|index, total| format!("Analyzed image {} of {}.", index + 1, total),
    };
    let mut records_payload_calls = 0usize;
    let result = super::run_batched_analysis_job(
        &api,
        &settings,
        &job,
        &cfg,
        3,
        "mlx",
        cancel,
        rx,
        blocking,
        |records: &[u32]| {
            records_payload_calls += 1;
            records.iter().map(|r| json!(r)).collect()
        },
        |_records, _stored| unreachable!("a canceled job must not build a completed update"),
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(_))),
        "a canceled analysis job returns WorkerError::Canceled, got {result:?}"
    );
    assert_eq!(
        records_payload_calls, 0,
        "the records → sidecar payload must never be built on cancel (no embeddings are POSTed)"
    );
    let posts = posts.lock().expect("posts lock");
    // Exactly one terminal `canceled`, and it is the FINAL post — never a mid-run acknowledgement.
    let canceled: Vec<&Value> = posts.iter().filter(|p| p["status"] == "canceled").collect();
    assert_eq!(
        canceled.len(),
        1,
        "exactly one terminal canceled is posted, got posts: {posts:?}"
    );
    assert_eq!(
        posts.last().map(|p| &p["status"]),
        Some(&json!("canceled")),
        "the terminal canceled must be the LAST post (deferred to after the task stopped)"
    );
    // The scaffold never reached the saving stage (the records POST) on the cancel path.
    assert!(
        posts.iter().all(|p| p["status"] != "saving"),
        "a canceled job must not post the Saving stage: {posts:?}"
    );
}

// sc-8804 (F-003) — the shared cancel-and-join guard. Every streaming job consumer binds its
// blocking GPU/training task to its `CancelFlag` through `CancelJoinGuard`; on any early return
// (a transient progress/heartbeat POST failure or a 409 stale-sweep reclaim propagating through
// `?`) the guard must trip the flag AND abort the task, so the GPU work stops instead of leaking
// alongside the next claimed job. The happy path reclaims the handle via `into_handle`, disarming
// the guard so a clean completion never aborts.

/// Dropping the guard early (the `?`-return shape) trips the engine cancel flag and aborts the
/// still-running task — the core F-003 defect, tested in isolation.
#[tokio::test]
async fn cancel_join_guard_trips_flag_and_aborts_on_early_drop() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let cancel = gen_core::CancelFlag::new();
    // A task that would run "forever" unless aborted; it flips `ran_to_completion` only if it ever
    // reaches its end. We hold a cancel-flag clone the task polls so a cooperative bail is possible,
    // but the guard's abort is what must stop a task that never checks.
    let completed = Arc::new(AtomicBool::new(false));
    let task_completed = completed.clone();
    let task_cancel = cancel.clone();
    let handle = tokio::task::spawn(async move {
        for _ in 0..1_000 {
            if task_cancel.is_cancelled() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        task_completed.store(true, Ordering::SeqCst);
    });

    {
        let _guard: crate::CancelJoinGuard<gen_core::CancelFlag, ()> =
            crate::CancelJoinGuard::new(cancel.clone(), handle);
        // Simulate the early `?` return: the guard goes out of scope here without `into_handle`.
    }

    assert!(
        cancel.is_cancelled(),
        "dropping the guard early must trip the engine cancel flag"
    );
    // The abort tears the task down; give the runtime a tick to reap it, then confirm it never
    // ran to completion (i.e. it was stopped, not left running).
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !completed.load(Ordering::SeqCst),
        "the guard must abort the task on early drop — it must not run to completion"
    );
}

/// Reclaiming the handle via `into_handle` disarms the guard: a clean completion neither cancels
/// nor aborts, and the task's value is returned intact.
#[tokio::test]
async fn cancel_join_guard_into_handle_disarms_the_guard() {
    let cancel = gen_core::CancelFlag::new();
    let handle = tokio::task::spawn(async { 42_u32 });
    let guard: crate::CancelJoinGuard<gen_core::CancelFlag, u32> =
        crate::CancelJoinGuard::new(cancel.clone(), handle);
    let value = guard.into_handle().await.expect("task joins cleanly");
    assert_eq!(value, 42, "the reclaimed handle yields the task's value");
    assert!(
        !cancel.is_cancelled(),
        "a disarmed guard must never trip the cancel flag on the success path"
    );
}

/// A stub whose worker-heartbeat POST fails (a 409/stale-sweep-class error), while its job GET
/// reports no cancel. `run_blocking_with_heartbeat` posts a `Busy` heartbeat every interval; the
/// failing POST propagates through `?`, which must trip the cancel flag and abort the long task
/// via the guard rather than returning while the task keeps running (sc-8804, F-003).
async fn spawn_failing_heartbeat_stub() -> String {
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn progress_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, false)).into_response()
    }
    async fn heartbeat_route() -> Response {
        // Mimic a 409 the stale-sweep produces once the job is reclaimed — a non-transport API
        // error that heartbeat() propagates (it swallows only transport errors).
        (AxumStatusCode::CONFLICT, "stub heartbeat conflict").into_response()
    }
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state(());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

#[tokio::test]
async fn run_blocking_with_heartbeat_aborts_task_when_heartbeat_post_fails() {
    let base_url = spawn_failing_heartbeat_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    // Shortest interval the clamp allows, so the failing heartbeat fires quickly.
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    // A long "GPU" task that only stops if its cancel flag is tripped — modeling a denoise/training
    // run that would otherwise keep burning the device after the consumer returns.
    let task = tokio::task::spawn(async move {
        for _ in 0..600 {
            if task_cancel.is_cancelled() {
                return Ok::<(), WorkerError>(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(())
    });

    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        Some(cancel.clone()),
        "canceled",
        "f003 test task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        result.is_err(),
        "the failing heartbeat POST must propagate as an error"
    );
    assert!(
        cancel.is_cancelled(),
        "on the POST-failure early return the guard must trip the task's cancel flag (sc-8804)"
    );
}

/// sc-8804 (F-003) — the CORE bounded-join assertion, in the real `spawn_blocking` shape the
/// converted consumers use. `JoinHandle::abort()` is INERT on an already-running blocking task, so
/// a drop-and-run teardown (trip flag, abort, return immediately) would return BEFORE the blocking
/// GPU task observes the flag and winds down — and the worker would claim the next job with the
/// device still busy. The fix's `cancel_and_join` must AWAIT the task, so by the time
/// `run_blocking_with_heartbeat` returns the blocking task has ALREADY wound down.
///
/// Discriminator (mutation check): this test asserts `wound_down == true` AT RETURN TIME. Under the
/// old drop-and-run behavior (abort-only, no awaited join) the flag is tripped but the consumer
/// returns before the `spawn_blocking` thread reaches its next cancel poll, so `wound_down` is still
/// `false` at return — the assertion fails. Only an awaited bounded-join makes it pass. The prior
/// test above used `tokio::task::spawn` (where `abort()` DOES stop the task), which is exactly why
/// it could not catch the blocking-task gap.
#[tokio::test]
async fn run_blocking_with_heartbeat_waits_for_blocking_task_winddown_on_post_failure() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let base_url = spawn_failing_heartbeat_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    // Set by the blocking task ONLY when it observes the cancel flag and winds down. The consumer
    // must not return until this is true.
    let wound_down = Arc::new(AtomicBool::new(false));
    let task_wound_down = wound_down.clone();
    // A genuine `spawn_blocking` task (a synchronous busy loop, like a denoise/detect thread) that
    // polls its cancel flag between "steps". `abort()` cannot stop it — only the flag can — so an
    // awaited join is the only way the consumer can observe its wind-down before returning.
    let task = tokio::task::spawn_blocking(move || {
        for _ in 0..2_000 {
            if task_cancel.is_cancelled() {
                task_wound_down.store(true, Ordering::SeqCst);
                return Ok::<(), WorkerError>(());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Reached only if never canceled: still mark wound-down so a hang is distinguishable.
        task_wound_down.store(true, Ordering::SeqCst);
        Ok(())
    });

    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        Some(cancel.clone()),
        "canceled",
        "f003 blocking winddown task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        result.is_err(),
        "the failing heartbeat POST must propagate as an error"
    );
    assert!(
        cancel.is_cancelled(),
        "the teardown must trip the blocking task's cancel flag (sc-8804)"
    );
    // The load-bearing assertion: the consumer AWAITED the bounded join, so the blocking task has
    // already wound down by the time we get here. A drop-and-run teardown fails this.
    assert!(
        wound_down.load(Ordering::SeqCst),
        "run_blocking_with_heartbeat must bounded-join the blocking task — it must NOT return \
         while the task is still running (sc-8804 F-003); the task had not wound down at return"
    );
}

/// sc-8908 (F-106) — the smart-select SAM3 image ops (box/points) pass `None` for `cancel` by
/// explicit per-engine decision: each is one bounded SAM3 forward pass on one image with no
/// per-step loop for a flag to poll, so there is no seam a `Some(flag)` could interrupt. This test
/// locks the `None`-path contract the decision relies on: a bounded single-shot blocking task run
/// with `cancel = None` resolves to its value, and (unlike the `Some(flag)` paths) the consumer
/// never consults the cancel poll — so no "canceled" is ever posted even mid-run. If someone
/// re-adds a `Some(flag)` that the SAM3 engine can't read, this stays green but the flag is a no-op
/// (the F-106 finding); the guard against that is the documented `None`-caller list at the
/// `run_blocking_with_heartbeat` definition, kept in sync with this call site.
#[tokio::test]
async fn run_blocking_with_heartbeat_none_cancel_returns_single_shot_value() {
    let (base_url, posts) = spawn_no_user_cancel_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    settings.heartbeat_seconds = 5;
    let api = ApiClient::new(&settings);

    // A bounded single-shot blocking task, the smart-select SAM3 shape: one forward pass, returns a
    // value, no cancel flag consulted.
    let task = tokio::task::spawn_blocking(|| Ok::<u32, WorkerError>(42));
    let result = crate::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-1",
        None,
        "",
        "smart-select none-cancel test task",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert_eq!(
        result.expect("the None-cancel single-shot task resolves to its value"),
        42
    );
    // No terminal "canceled" progress was posted — the None path never consults the cancel poll.
    let posted = posts.lock().expect("posts lock");
    assert!(
        posted.iter().all(|body| body
            .get("status")
            .and_then(Value::as_str)
            .is_none_or(|s| s != "canceled")),
        "the None-cancel path must never post a terminal canceled status"
    );
}

/// sc-8804 (F-003) — the child-process leg: `run_ffmpeg` (media_jobs) and the `hf` CLI download
/// (model_jobs) build their `tokio::process::Command` with `kill_on_drop(true)` so a
/// heartbeat/cancel `?` early return reaps the child instead of leaving ffmpeg/`hf` writing partial
/// files. A tokio child is NOT reaped on drop by default; this test locks the mechanism the fix
/// relies on — a `kill_on_drop(true)` child is torn down when its handle is dropped.
#[cfg(unix)]
#[tokio::test]
async fn kill_on_drop_reaps_a_dropped_child() {
    use tokio::process::Command;
    // A child that would run for a minute unless killed. Capture its PID, drop the handle, then
    // confirm the OS no longer has it (kill(pid, 0) fails with ESRCH once reaped).
    let child = Command::new("sleep")
        .arg("60")
        .kill_on_drop(true)
        .spawn()
        .expect("spawn sleep");
    let pid = child.id().expect("child has a pid");
    drop(child);
    // Give the runtime a moment to send the kill and reap the zombie.
    tokio::time::sleep(Duration::from_millis(200)).await;
    // `kill -0` probes existence without signaling; a reaped/gone process yields a non-zero status.
    let alive = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .expect("run kill -0")
        .success();
    assert!(
        !alive,
        "a kill_on_drop(true) child must be reaped when its handle is dropped (pid {pid} still alive)"
    );
}

/// sc-5516 / sc-8805 — the detail sibling: `begin_detail_cancel` trips the engine flag
/// (which the interval-armed detail event loop now reaches within one ~2s poll even while
/// the model is cold-loading or a tile is mid-refine) and acknowledges with a NON-terminal
/// `running` update; the terminal `Canceled` is posted by `run_image_detail_job` only after
/// the blocking refinement actually stops. macOS-only because `image_jobs/detail.rs` is the
/// macOS MLX route (off-Mac `run_image_detail_job` is a hard error stub).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn begin_detail_cancel_trips_flag_and_stays_non_terminal() {
    let (base_url, posts) = spawn_progress_capture_stub().await;
    let mut settings = test_settings(base_url.clone(), None);
    settings.api_url = base_url;
    let api = ApiClient::new(&settings);
    let cancel = gen_core::CancelFlag::new();
    crate::image_jobs::begin_detail_cancel(&api, "job-1", &cancel, "mlx").await;
    assert!(
        cancel.is_cancelled(),
        "begin_detail_cancel must trip the engine cancel flag"
    );
    let posts = posts.lock().expect("posts lock");
    assert_eq!(
        posts.len(),
        1,
        "exactly one acknowledgement update is posted"
    );
    assert_eq!(
        posts[0]["status"], "running",
        "the cancel acknowledgement must stay NON-terminal (sc-5516)"
    );
    assert!(
        posts[0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Cancelling"),
        "the acknowledgement message should read as Cancelling…"
    );
}

/// sc-4176 — on macOS an MLX-routed model whose weights don't resolve must
/// fail loudly with a precise re-download error instead of silently completing
/// with procedural stub output; non-engine model ids keep the stub path.
#[cfg(target_os = "macos")]
mod stub_fallback_gate {
    use super::*;
    use crate::image_jobs::mlx_weights_gap;
    use crate::video_jobs::ensure_video_engine_weights;
    use sceneworks_core::image_request::ImageRequest;
    use sceneworks_core::video_request::VideoRequest;
    use serde_json::Map;

    fn settings_with_empty_data_dir() -> (tempfile::TempDir, Settings) {
        let dir = tempdir().expect("tempdir");
        let mut settings = test_settings("http://127.0.0.1:1".to_owned(), None);
        settings.data_dir = dir.path().to_path_buf();
        (dir, settings)
    }

    fn object(value: serde_json::Value) -> Map<String, serde_json::Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn image_engine_model_without_weights_is_a_precise_error_not_a_stub() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({ "model": "z_image_turbo" })));
        let gap = mlx_weights_gap(&request, &settings).expect("missing weights flagged");
        assert!(gap.contains("z_image_turbo"), "{gap}");
        assert!(gap.contains("Re-download"), "{gap}");
    }

    #[test]
    fn image_non_engine_model_keeps_the_stub_path() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({ "model": "base-model" })));
        assert_eq!(mlx_weights_gap(&request, &settings), None);
    }

    #[test]
    fn image_explicit_model_path_resolves_and_passes_the_gate() {
        let (dir, settings) = settings_with_empty_data_dir();
        let request = ImageRequest::from_payload(&object(json!({
            "model": "z_image_turbo",
            "advanced": { "modelPath": dir.path().to_string_lossy() },
        })));
        assert_eq!(mlx_weights_gap(&request, &settings), None);
    }

    #[test]
    fn image_explicit_model_path_outside_data_dir_is_rejected() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let outside = tempdir().expect("outside tempdir");
        let request = ImageRequest::from_payload(&object(json!({
            "model": "z_image_turbo",
            "advanced": { "modelPath": outside.path().to_string_lossy() },
        })));
        let gap = mlx_weights_gap(&request, &settings).expect("unsafe modelPath rejected");
        assert!(gap.contains("Image modelPath"), "{gap}");
        assert!(gap.contains("app-managed"), "{gap}");
    }

    #[test]
    fn video_engine_model_without_weights_is_a_precise_error_not_a_stub() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "wan_2_2_i2v_14b" })));
        let error = ensure_video_engine_weights(&request, &settings)
            .expect_err("missing Wan weights flagged");
        assert!(
            error.to_string().contains("no MLX weights found"),
            "{error}"
        );
    }

    #[test]
    fn video_svd_without_source_asset_is_an_invalid_payload_error() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "svd" })));
        let error = ensure_video_engine_weights(&request, &settings)
            .expect_err("svd without a source image flagged");
        assert!(error.to_string().contains("source image"), "{error}");
    }

    #[test]
    fn video_non_engine_model_keeps_the_stub_path() {
        let (_dir, settings) = settings_with_empty_data_dir();
        let request = VideoRequest::from_payload(&object(json!({ "model": "stub-model" })));
        ensure_video_engine_weights(&request, &settings).expect("non-engine model passes");
    }
}

// ---------------------------------------------------------------------------
// sc-8807: the shared blocking keepalive (`run_blocking_with_heartbeat`) is the seam the
// SAM2/SAM3 propagate steps now run through. These tests pin the three behaviors the segment
// wiring relies on, with a stand-in blocking task in place of the real (weights-requiring)
// propagate: (a) the worker heartbeat is pinged while the task runs, (b) the API cancel poll
// trips the threaded `CancelFlag`, and (c) the terminal `Canceled` is posted whether the task
// honors the flag itself or completes before the flag is tripped.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct KeepaliveStubState {
    heartbeats: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    progress: std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
}

impl KeepaliveStubState {
    fn new() -> Self {
        Self {
            heartbeats: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            progress: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
}

/// A stub API that always reports `cancel_requested: true`, counts worker heartbeats, and records
/// every job-progress post (so the terminal `Canceled` write is observable).
async fn spawn_keepalive_stub(state: KeepaliveStubState) -> String {
    async fn heartbeat_route(State(state): State<KeepaliveStubState>) -> Response {
        state
            .heartbeats
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // The body does not parse as a WorkerSnapshot; `heartbeat()` tolerates the decode
        // failure as a transport error, so the ping still counts without modeling the snapshot.
        Json(json!({})).into_response()
    }
    async fn job_route(axum::extract::Path(job_id): axum::extract::Path<String>) -> Response {
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    async fn progress_route(
        State(state): State<KeepaliveStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
        Json(payload): Json<Value>,
    ) -> Response {
        state.progress.lock().unwrap().push(payload);
        Json(job_snapshot_json(&job_id, true)).into_response()
    }
    let app = Router::new()
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener has address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serves");
    });
    format!("http://{address}")
}

/// The MLX lane: the keepalive trips the flag, the engine honors it between propagated frames and
/// surfaces `WorkerError::Canceled` from inside the task — the terminal `Canceled` must be posted
/// and the error propagated (never a dangling non-terminal job).
#[tokio::test]
async fn keepalive_posts_terminal_canceled_when_the_task_honors_the_flag() {
    let state = KeepaliveStubState::new();
    let base = spawn_keepalive_stub(state.clone()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base;
    let api = ApiClient::new(&settings);

    let flag = gen_core::CancelFlag::new();
    let task_flag = flag.clone();
    let task = tokio::task::spawn_blocking(move || -> super::WorkerResult<u32> {
        let start = std::time::Instant::now();
        while !task_flag.is_cancelled() {
            if start.elapsed() > Duration::from_secs(30) {
                return Err(WorkerError::Engine("cancel flag never tripped".to_owned()));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // The engine's per-frame cancel contract (gen-core d8038beb) surfaces Canceled itself.
        Err(WorkerError::Canceled(
            "Person segmentation canceled by user.".to_owned(),
        ))
    });
    let result = super::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-8807-mlx",
        Some(flag),
        "Person tracking canceled during segmentation.",
        "sam propagate stand-in",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(ref m)) if m == "Person segmentation canceled by user."),
        "expected the task's Canceled to propagate, got {result:?}"
    );
    assert!(
        state.heartbeats.load(std::sync::atomic::Ordering::SeqCst) >= 1,
        "Busy heartbeat pinged during the blocking task"
    );
    let posts = state.progress.lock().unwrap();
    assert!(
        posts.iter().any(|p| p["status"] == "canceled"),
        "terminal Canceled posted to the job, got {posts:?}"
    );
}

/// A task that completes before the flag is observed: the compute finishes and returns `Ok`
/// before it sees the tripped flag — the keepalive still posts the terminal `Canceled` with
/// its own cancel copy and returns `WorkerError::Canceled`.
#[tokio::test]
async fn keepalive_cancels_the_job_even_when_the_task_cannot_observe_the_flag() {
    let state = KeepaliveStubState::new();
    let base = spawn_keepalive_stub(state.clone()).await;
    let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
    settings.api_url = base;
    let api = ApiClient::new(&settings);

    let flag = gen_core::CancelFlag::new();
    let wait_flag = flag.clone();
    let task = tokio::task::spawn_blocking(move || -> super::WorkerResult<u32> {
        // Wait until the keepalive has tripped the flag (so `canceled` is set before we return),
        // then finish successfully — the completes-before-observing-the-flag shape.
        let start = std::time::Instant::now();
        while !wait_flag.is_cancelled() && start.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(7)
    });
    let result = super::run_blocking_with_heartbeat(
        &api,
        &settings,
        "job-8807-candle",
        Some(flag),
        "Person tracking canceled during segmentation.",
        "candle propagate stand-in",
        crate::no_cancel_ack(),
        task,
    )
    .await;

    assert!(
        matches!(result, Err(WorkerError::Canceled(ref m)) if m == "Person tracking canceled during segmentation."),
        "expected the keepalive's Canceled, got {result:?}"
    );
    let posts = state.progress.lock().unwrap();
    assert!(
        posts.iter().any(|p| p["status"] == "canceled"),
        "terminal Canceled posted to the job, got {posts:?}"
    );
}

/// sc-11184 (F-014) — the Windows stdin-close latch must survive a zero-receiver window.
///
/// `wait_for_parent_stdin_close` (the `#[cfg(not(unix))]` graceful-shutdown signal) fans a single
/// stdin-EOF over a `tokio::sync::watch` channel whose `Sender` lives in a process-lifetime
/// `OnceLock`; receivers exist ONLY while a caller is polling the wait future. Between the poll-phase
/// `select!` and the run_job `select!` `receiver_count()` is 0, and the dedicated reader is
/// single-shot (it breaks on EOF, latches once, exits). The reader originally used
/// `watch::Sender::send`, which returns `Err` WITHOUT storing the value when there is no receiver —
/// so an EOF landing in that window was lost forever and every later waiter blocked on `changed()`
/// indefinitely, defeating F-014's graceful shutdown. The fix trips the latch with `send_replace`,
/// which stores the value unconditionally.
///
/// This asserts the semantics directly (the reader task itself is `#[cfg(not(unix))]` and can't be
/// driven on the macOS/Linux test host): construct the channel EXACTLY as the reader does, drop the
/// initial receiver so `receiver_count() == 0`, trip via `send_replace`, THEN subscribe (mirroring
/// `sender.subscribe()` in the wait fn) and assert the fresh receiver observes `true` immediately via
/// `borrow_and_update()` (the wait fn's entry-borrow), so its wait future is ready with no `changed()`
/// await. It also documents that plain `send` would have failed this scenario.
#[tokio::test]
async fn stdin_close_latch_retained_across_zero_receiver_window() {
    use tokio::sync::watch;

    // Same construction the reader uses: initial `_rx` is dropped, so no receiver is subscribed.
    let (tx, rx) = watch::channel(false);
    drop(rx);
    assert_eq!(
        tx.receiver_count(),
        0,
        "precondition: the zero-receiver window (initial rx dropped, no waiter polling)"
    );

    // Regression guard: the OLD `send` drops the value in this window (returns Err, latch stays
    // false) — this is exactly the missed-wakeup the fix removes.
    assert!(
        tx.send(true).is_err(),
        "watch::Sender::send returns Err with zero receivers — proves the pre-fix bug"
    );
    assert!(
        !*tx.borrow(),
        "and `send` left the latched value UNCHANGED (false) — the lost graceful-shutdown request"
    );

    // The fix: `send_replace` stores the value unconditionally, even with zero receivers.
    tx.send_replace(true);
    assert!(
        *tx.borrow(),
        "send_replace latches `true` regardless of receiver_count"
    );

    // A waiter that subscribes AFTER the trip (the run_job select!'s fresh `subscribe()`) must see
    // the latched value immediately via the entry-borrow, never blocking on `changed()`.
    let mut receiver = tx.subscribe();
    assert!(
        *receiver.borrow_and_update(),
        "a receiver created after the trip observes shutdown immediately (entry-borrow), not a hang"
    );
}

