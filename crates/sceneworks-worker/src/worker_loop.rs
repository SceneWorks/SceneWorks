//! Worker startup and claim-loop ownership.
use super::*;

pub async fn run_worker_loop(settings: Settings) -> WorkerResult<()> {
    // sc-4482 (epic 3720): log the resolved backend-neutral gen-core contract version at startup
    // so a pin skew that slips past the CI guard (`scripts/check-gen-core-skew.sh`) is
    // diagnosable from one log line. One shared contract version backs every linked backend.
    tracing::info!(
        event = "gen_core_contract_version",
        version = %gen_core::VERSION,
        gpuId = %settings.gpu_id,
        "rust_worker gen-core contract version"
    );
    // sc-7820 (epic 7819): apply the GPU memory ceiling to the MLX runtime once at startup, before
    // any model load. The MLX limit is process-global, so this single call covers generations,
    // upscales, AND LoRA training. When the user configured no ceiling (0) a default derived from
    // physical RAM is applied instead (GitHub #1932) — MLX's own default budget is ~all of unified
    // memory, which starves macOS on a small Mac. No-op on non-macOS/candle builds.
    initialize_worker_gpu_with(
        &settings,
        utility_job_scope(std::env::var("SCENEWORKS_UTILITY_JOBS").ok().as_deref()),
        generator_cache::apply_gpu_memory_limit,
    )?;
    // sc-26610: keep proven weight-seal digests across worker restarts, so a restart does not
    // re-hash every multi-GB weight file before its next load.
    gen_core::ArtifactSeal::persist_proven_digests_in(
        settings.data_dir.join("cache").join("seal-digests"),
    );
    // sc-7825 (epic 7819): on the MLX GPU worker only, publish live MLX memory telemetry to the
    // shared config dir for the Settings readout. Gated to `mlx` so the CPU utility workers (which
    // do no MLX work) don't clobber the file with zeros.
    if settings.gpu_id == "mlx" {
        generator_cache::spawn_gpu_telemetry(settings.config_dir.clone());
    }
    // sc-16247 / sc-16260: probe the CUDA device ONCE before advertising anything, and let the
    // verdict shape what this worker claims to be able to do. `discover_gpu` withholds the whole
    // candle capability block on an unusable GPU, so the registration below advertises only the
    // placeholder set and no generation job can route here.
    let mut health = cuda_startup_health(&settings).await;
    let gpu = discover_gpu(&settings, &health).await;
    let api = ApiClient::new(&settings);
    let http_client = crate::downloads::streaming_download_client();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    let mut lock_failures = 0_u32;
    let mut idle_heartbeat = IdleHeartbeat::new(progress_report_interval(&settings));
    // sc-16260 AC 4: the startup probe is one sample, so an unhealthy worker re-probes on an
    // interval and re-advertises if the host is repaired underneath it. Seeded a full interval
    // out — the startup probe just ran, and re-running it immediately would say nothing new.
    let mut next_gpu_recheck = Instant::now() + GPU_HEALTH_RECHECK;
    // sc-19710 / sc-19706: ONE resolved-cache maintenance slot, shared by the retention checkpoint
    // and the promotion drain. A single slot is deliberate rather than one handle each: a sweep
    // walks and re-hashes eviction candidates while a promotion copies and hashes a whole bundle,
    // and running both at once would have them competing for the same disk and the same entries —
    // retention deciding what fits while promotion is still adding to it.
    //
    // The startup occupant is the retention checkpoint, which recovers the store (finishing any
    // eviction interrupted by a crash) and then enforces retention. It is started, never awaited —
    // the first job claim must not queue behind a recover-plus-retention pass.
    let mut maintenance_task = Some(spawn_retention_checkpoint(settings.data_dir.clone(), true));
    let mut next_retention_checkpoint = Instant::now() + RESOLVED_CACHE_RETENTION_INTERVAL;
    // sc-20636: the other startup reclamation. Spawned blocking and never awaited, for the same
    // reason the retention checkpoint is not: removing an abandoned multi-gigabyte staging tree
    // must not delay the first job claim.
    {
        let data_dir = settings.data_dir.clone();
        crate::load_progress::spawn_blocking(move || match reclaim_import_staging(&data_dir) {
            Ok(0) => {}
            Ok(reclaimed) => tracing::info!(
                event = "import_staging_reclaimed",
                reclaimed,
                "reclaimed abandoned model-import staging directories"
            ),
            Err(error) => tracing::warn!(error = %error, "model-import staging sweep failed"),
        });
    }
    loop {
        // A Metal watchdog timeout poisons the command queue for this OS process. The just-failed
        // job was already made terminal by `run_utility_job`; stop BEFORE `poll_once` can claim a
        // successor, mark this instance unhealthy, and clean-exit. Under `GPU_ID=auto` the existing
        // supervisor observes that single exit and spawns a fresh process with fresh Metal state.
        // Clean exit is intentional: an abnormal-exit attribution would manufacture a second job
        // failure after the truthful first timeout has already been persisted.
        if settings.gpu_id == "mlx" {
            if let Some(recycle) = mlx_worker_recovery::global().begin_recycle() {
                emit_event_value(
                    Level::ERROR,
                    json!({
                        "event": if recycle.process_fault.is_some() {
                            "mlx_owner_executor_poisoned_worker_recycle"
                        } else {
                            "mlx_h3_i2v_poisoned_worker_recycle"
                        },
                        "workerId": settings.worker_id,
                        "gpuId": settings.gpu_id,
                        "firstTimeout": recycle.first_timeout,
                        "processFault": recycle.process_fault,
                        "sawSubmissionsIgnored": recycle.saw_submissions_ignored,
                    }),
                );
                let _ = heartbeat_with_reason(
                    &api,
                    &settings,
                    WorkerStatus::Unhealthy,
                    None,
                    Some(recycle.reason),
                )
                .await;
                return Ok(());
            }
            if let Some(reason) = mlx_worker_recovery::global().quarantine_reason() {
                // This is reachable only when shutdown interrupted the active job's terminal-write
                // retry. Stay unclaimable rather than recycling on an unconfirmed failure or
                // falling through to `poll_once`; the shutdown arm normally resolves immediately.
                let _ = heartbeat_with_reason(
                    &api,
                    &settings,
                    WorkerStatus::Unhealthy,
                    None,
                    Some(reason),
                )
                .await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))) => {}
                    _ = shutdown_signal() => return Ok(()),
                }
                continue;
            }
        }
        if !health.is_usable() && Instant::now() >= next_gpu_recheck {
            next_gpu_recheck = Instant::now() + GPU_HEALTH_RECHECK;
            recheck_gpu_health(&api, &settings, &mut health).await?;
        }
        // sc-8845 (F-043): shutdown is observed ONLY here, around the claim / idle-sleep phase —
        // NOT around full job execution. `poll_once` does no long GPU work (memory-sync, idle
        // heartbeat, the transactional claim POST, and the idle sleep), so racing it against
        // shutdown and dropping it at an await point loses no job state: nothing is claimed on the
        // idle path, and a claimed job is handled below OUTSIDE this select. Before this change the
        // select raced the WHOLE `poll_once` (claim + the entire job) against shutdown, so a
        // graceful quit mid-job dropped the in-flight future at an arbitrary await, left the job
        // `running` until the 90s stale sweep marked it `interrupted`, and killed spawn_blocking GPU
        // work mid-write. Now a mid-job shutdown trips the job's cancel and posts a prompt terminal
        // `Canceled` (see `run_job_with_shutdown`).
        let claim = tokio::select! {
            result = poll_once(&api, &settings, &mut idle_heartbeat, &health) => result,
            _ = shutdown_signal() => {
                // Clean-idle shutdown: no job in flight, so the pre-existing Offline heartbeat +
                // return is preserved exactly.
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        };
        match claim {
            Ok(None) => {
                lock_failures = 0;
                start_idle_maintenance(
                    &settings,
                    &mut maintenance_task,
                    &mut next_retention_checkpoint,
                );
            }
            Ok(Some(job)) => {
                lock_failures = 0;
                // Execute the claimed job WITHOUT racing (and dropping) the whole future against
                // shutdown. `run_job_with_shutdown` supervises execution: on a mid-job shutdown it
                // trips the job's cancel flag, lets the in-flight future wind down (never dropped
                // mid-write), and posts a terminal `Canceled` for the job before returning
                // `ShutdownDuringJob` so the loop exits with the job in a prompt terminal state.
                match run_job_with_shutdown(&api, &settings, &http_client, job).await {
                    // `run_utility_job` already posts a terminal Idle heartbeat at its end, so the
                    // scheduler should treat that as the just-sent one and wait a full interval —
                    // marking it *due* here made the next `poll_once` fire a redundant second Idle
                    // heartbeat right away (sc-8952).
                    JobOutcome::Completed => idle_heartbeat.mark_sent(),
                    JobOutcome::ShutdownDuringJob => {
                        let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                        return Ok(());
                    }
                }
            }
            Err(error) => {
                if wait_after_poll_failure(&api, &settings, error, &mut lock_failures).await {
                    return Ok(());
                }
            }
        }
    }
}

/// Poll the shared maintenance slot without awaiting work on the claim path.
fn start_idle_maintenance(
    settings: &Settings,
    maintenance_task: &mut Option<tokio::task::JoinHandle<()>>,
    next_retention_checkpoint: &mut Instant,
) {
    // Claiming nothing is the proof of idleness both maintenance activities require:
    // no job is in flight, so neither a sweep nor a bundle copy can compete with one.
    // Every artifact lock a sweep takes is non-blocking, so an in-use model is skipped
    // rather than waited on.
    //
    // The handle is polled, never awaited: this arm sits directly on the claim path,
    // so the loop must come straight back round to claim the next job while
    // maintenance is still running. Work is also skipped entirely while a predecessor
    // is in flight, so slow sweeps and slow copies cannot stack up.
    if maintenance_task
        .as_ref()
        .is_some_and(tokio::task::JoinHandle::is_finished)
    {
        *maintenance_task = None;
    }
    if maintenance_task.is_none() {
        if Instant::now() >= *next_retention_checkpoint {
            // Retention wins whenever it is due. The reverse ordering could starve
            // retention indefinitely on a worker with a steady promotion stream, and
            // retention is what keeps the cache inside its size limit — the limit
            // promotion admission itself is judged against.
            //
            // This does NOT make starving promotion impossible, only bounded by an
            // assumption: the next checkpoint is armed when this one is STARTED, so a
            // sweep that runs longer than the interval is already due again when it
            // finishes and sweeps run back to back, with no idle turn left over for a
            // drain. That holds only while sweep duration stays under the interval.
            // It is self-stabilizing rather than a defect — a sweep that slow means
            // the cache is over its limit, which is the condition promotion must not
            // be adding to — so the ordering stands, but the guarantee is
            // "promotion yields to retention", not "promotion always runs".
            *next_retention_checkpoint = Instant::now() + RESOLVED_CACHE_RETENTION_INTERVAL;
            *maintenance_task = Some(spawn_retention_checkpoint(settings.data_dir.clone(), false));
        } else if resolved_cache_promotion::work_pending(&settings.data_dir) {
            // sc-19706: in-memory check only — no store open, no stat — because this
            // sits on the claim path exactly like the checkpoint gate above.
            *maintenance_task = Some(spawn_promotion_drain(settings.clone()));
        }
    }
}

/// Back off between claims; return true only when clean-idle shutdown requests exit.
async fn wait_after_poll_failure(
    api: &ApiClient,
    settings: &Settings,
    error: WorkerError,
    lock_failures: &mut u32,
) -> bool {
    match error {
        error if is_database_locked(&error) => {
            // SQLite claim contention. With busy_timeout + BEGIN IMMEDIATE in the
            // store this should be rare, but back off (instead of hammering at the
            // flat poll interval) and make it visible so an MLX-eligible job lost to
            // lock contention is explained rather than silently losing the claim to another poller.
            *lock_failures = lock_failures.saturating_add(1);
            let delay = retry_delay(settings.poll_seconds, *lock_failures);
            emit_event_value(
                Level::WARN,
                json!({
                    "event": "claim_lock_contention",
                    "workerId": settings.worker_id,
                    "gpuId": settings.gpu_id,
                    "consecutiveFailures": *lock_failures,
                    "retryInSeconds": delay,
                    "error": error.to_string(),
                }),
            );
            // The back-off sleep is a between-jobs wait, so it too must observe shutdown rather
            // than blocking a graceful quit for up to `delay` seconds.
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                _ = shutdown_signal() => {
                    let _ = heartbeat(api, settings, WorkerStatus::Offline, None).await;
                    return true;
                }
            }
        }
        error => {
            *lock_failures = 0;
            tracing::error!(
                event = "rust_worker_poll_failed",
                error = %error,
                "worker claim poll failed"
            );
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))) => {}
                _ = shutdown_signal() => {
                    let _ = heartbeat(api, settings, WorkerStatus::Offline, None).await;
                    return true;
                }
            }
        }
    }
    false
}
