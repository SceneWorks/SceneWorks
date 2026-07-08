
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

    let error =
        super::image_jobs::run_image_generate_job(&api, &settings, &reqwest::Client::new(), &job)
            .await
            .expect_err("missing projectId is rejected by the image handler");
    assert!(
        matches!(&error, WorkerError::InvalidPayload(message) if message.contains("projectId")),
        "expected a projectId payload error from the image handler, got {error:?}",
    );
}

#[tokio::test]
async fn supervisor_restarts_exited_children_with_backoff_state() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let spec = WorkerSpec {
        worker_id: "worker-gpu-auto-0".to_owned(),
        gpu_id: "0".to_owned(),
    };
    let mut exited = spawn_exit_child();
    for _ in 0..20 {
        if exited.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let mut children = HashMap::from([(
        spec.worker_id.clone(),
        SupervisedChild {
            spec,
            process: exited,
            restart_attempt: 0,
            spawned_at: std::time::Instant::now(),
            next_restart_at: None,
        },
    )]);
    let spawns = std::cell::Cell::new(0_u32);
    let mut spawner = |_settings: &_, _spec: &WorkerSpec| {
        spawns.set(spawns.get() + 1);
        Ok(spawn_sleep_child())
    };

    // Detection tick: the exit is reaped and a backoff deadline is stamped, but the
    // child is not respawned yet because its backoff has not elapsed (sc-8899).
    let t0 = std::time::Instant::now();
    restart_exited_children_at(&settings, &mut children, &mut spawner, t0)
        .await
        .expect("exit is detected");
    assert_eq!(
        spawns.get(),
        0,
        "detection tick does not respawn before backoff"
    );
    {
        let child = children
            .get("worker-gpu-auto-0")
            .expect("exited child stays tracked while backing off");
        assert_eq!(child.restart_attempt, 1);
        assert!(
            child.next_restart_at.is_some(),
            "a backoff deadline is stamped on the exited child"
        );
    }

    // Restart tick past the backoff deadline: the child is respawned exactly once.
    restart_exited_children_at(
        &settings,
        &mut children,
        &mut spawner,
        t0 + Duration::from_secs(30),
    )
    .await
    .expect("child restarts once its backoff elapses");

    assert_eq!(spawns.get(), 1);
    let child = children
        .get_mut("worker-gpu-auto-0")
        .expect("restarted child is tracked");
    assert_eq!(child.restart_attempt, 1);
    assert!(
        child.next_restart_at.is_none(),
        "the backoff deadline clears once the child is respawned"
    );
    assert!(child
        .process
        .try_wait()
        .expect("child status checks")
        .is_none());
    let _ = child.process.start_kill();
    let _ = child.process.wait().await;
}

/// sc-8899 / F-097: one child's restart backoff must not stall the whole
/// supervision tick. A crash-looping child with a long backoff and a healthy
/// sibling that just crashed are handled independently — the sibling restarts on
/// the next eligible tick while the still-backing-off child simply waits its turn.
#[tokio::test]
async fn supervisor_backoff_on_one_child_does_not_stall_another() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);

    // A crash-looping child mid-backoff: already reaped, with a deadline 30 s out.
    let mut looping = spawn_exit_child();
    for _ in 0..20 {
        if looping.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let t0 = std::time::Instant::now();
    let looping_child = SupervisedChild {
        spec: WorkerSpec {
            worker_id: "worker-gpu-auto-0".to_owned(),
            gpu_id: "0".to_owned(),
        },
        process: looping,
        restart_attempt: 6,
        spawned_at: t0,
        next_restart_at: Some(t0 + Duration::from_secs(30)),
    };

    // A healthy sibling that already crashed and whose short backoff is now due.
    let mut sibling = spawn_exit_child();
    for _ in 0..20 {
        if sibling.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let sibling_child = SupervisedChild {
        spec: WorkerSpec {
            worker_id: "worker-gpu-auto-1".to_owned(),
            gpu_id: "1".to_owned(),
        },
        process: sibling,
        restart_attempt: 1,
        spawned_at: t0,
        next_restart_at: Some(t0 + Duration::from_secs(1)),
    };

    let mut children = HashMap::from([
        ("worker-gpu-auto-0".to_owned(), looping_child),
        ("worker-gpu-auto-1".to_owned(), sibling_child),
    ]);
    let mut restarted = Vec::new();
    let mut spawner = |_settings: &_, spec: &WorkerSpec| {
        restarted.push(spec.worker_id.clone());
        Ok(spawn_sleep_child())
    };

    // A single tick at t0 + 5 s: the sibling's 1 s backoff is due while the looping
    // child's 30 s backoff is not. The old inline-sleep design would have blocked the
    // whole tick on whichever child was handled first; the per-child deadline model
    // restarts only the due sibling and leaves the looping child untouched (sc-8899).
    restart_exited_children_at(
        &settings,
        &mut children,
        &mut spawner,
        t0 + Duration::from_secs(5),
    )
    .await
    .expect("tick completes without stalling on the backing-off child");

    assert_eq!(
        restarted,
        vec!["worker-gpu-auto-1".to_owned()],
        "the due sibling restarts while the looping child is still backing off"
    );
    assert!(
        children["worker-gpu-auto-0"].next_restart_at.is_some(),
        "the looping child keeps its unexpired backoff deadline"
    );
    assert!(
        children["worker-gpu-auto-1"].next_restart_at.is_none(),
        "the restarted sibling has its deadline cleared"
    );

    for child in children.values_mut() {
        let _ = child.process.start_kill();
        let _ = child.process.wait().await;
    }
}

/// sc-4282 / F-MLXW-20: a child that ran healthily past the reset threshold
/// before exiting starts its restart backoff fresh, rather than carrying a
/// counter that has saturated upward over many widely-spaced crashes.
#[tokio::test]
async fn supervisor_resets_backoff_after_a_healthy_run() {
    let settings = test_settings("http://127.0.0.1".to_owned(), None);
    let spec = WorkerSpec {
        worker_id: "worker-gpu-auto-0".to_owned(),
        gpu_id: "0".to_owned(),
    };
    let mut exited = spawn_exit_child();
    for _ in 0..20 {
        if exited.try_wait().expect("child status checks").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The counter has ratcheted up over time, but this run stayed alive well past
    // the healthy-uptime threshold (spawned > 6 minutes ago).
    let mut children = HashMap::from([(
        spec.worker_id.clone(),
        SupervisedChild {
            spec,
            process: exited,
            restart_attempt: 7,
            spawned_at: std::time::Instant::now()
                .checked_sub(Duration::from_secs(360))
                .expect("monotonic clock backdates 6 minutes"),
            next_restart_at: None,
        },
    )]);

    // The backoff reset happens when the exit is detected, so a single detection
    // tick is enough to observe it (the respawn itself waits for the backoff).
    restart_exited_children_with_spawner(&settings, &mut children, |_settings, _spec| {
        Ok(spawn_sleep_child())
    })
    .await
    .expect("exit is detected");

    let child = children
        .get_mut("worker-gpu-auto-0")
        .expect("exited child stays tracked while backing off");
    // Reset to 0 on the healthy run, then advanced once for this restart.
    assert_eq!(
        child.restart_attempt, 1,
        "a healthy run resets the backoff counter"
    );
    let _ = child.process.start_kill();
    let _ = child.process.wait().await;
}

/// sc-4881: a child reaped after an uncatchable signal (here SIGKILL, the OOM
/// killer's weapon) is attributed to that signal, while a clean exit reports none.
/// This is the only layer that can observe the death — it's uncatchable in the
/// dying child itself.
#[cfg(unix)]
#[tokio::test]
async fn terminating_signal_distinguishes_signal_death_from_clean_exit() {
    let mut child = spawn_sleep_child();
    let pid = child.id().expect("child has a pid");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid as i32),
        nix::sys::signal::Signal::SIGKILL,
    )
    .expect("SIGKILL delivered");
    let status = child.wait().await.expect("killed child reaped");
    assert_eq!(terminating_signal(&status), Some(9));

    let mut clean = spawn_exit_child();
    let status = clean.wait().await.expect("clean child reaped");
    assert_eq!(terminating_signal(&status), None);
}

#[test]
fn child_died_abnormally_reports_signals_and_non_zero_exits_not_clean_exits() {
    // sc-6320: the supervisor attributes a real FAILURE for an uncatchable signal
    // death OR a non-zero self-exit (e.g. a Rust panic → 101), but a clean exit-0
    // is graceful and must report nothing (else a normal worker shutdown would
    // spuriously fail its job).
    assert!(child_died_abnormally(Some(9), None), "SIGKILL is abnormal");
    assert!(
        child_died_abnormally(None, Some(101)),
        "a panic exit (101) is abnormal"
    );
    assert!(
        child_died_abnormally(None, Some(1)),
        "any non-zero exit is abnormal"
    );
    assert!(
        !child_died_abnormally(None, Some(0)),
        "a clean exit-0 is graceful, not reported"
    );
}

#[tokio::test]
async fn writes_model_install_marker_with_expected_keys() {
    let temp = tempdir().expect("tempdir creates");
    let mut payload = serde_json::Map::new();
    payload.insert("modelId".to_owned(), json!("base-model"));
    payload.insert("modelName".to_owned(), json!("Base Model"));

    write_model_install_marker(temp.path(), &payload, "owner/model", "job-1")
        .await
        .expect("marker writes");

    let marker_path = temp.path().join(INSTALL_MARKER);
    let marker: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(marker_path).await.unwrap()).unwrap();
    assert_eq!(marker["repo"], "owner/model");
    assert_eq!(marker["modelId"], "base-model");
    assert_eq!(marker["modelName"], "Base Model");
    assert_eq!(marker["jobId"], "job-1");
    assert!(marker["completedAt"].as_str().is_some());
}
