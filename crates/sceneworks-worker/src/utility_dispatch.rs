//! Utility dispatch and terminal failure reconciliation.
use super::*;

pub(crate) fn missing_portable_revision_files(
    binding: &PortableRouteBinding,
    settings: &Settings,
) -> std::collections::BTreeMap<(String, String), Vec<String>> {
    let mut missing = std::collections::BTreeMap::new();
    let library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    for member in binding.models.iter().flat_map(|model| &model.members) {
        let sceneworks_core::model_artifacts::portable_binding::PortableSourceIdentity::Revision {
            identity,
        } = &member.source
        else {
            continue;
        };
        let snapshot = library
            .discover_snapshot(&identity.repository, Some(&identity.revision))
            .ok()
            .map(|(_, path)| path);
        for file in &member.files {
            if !snapshot
                .as_ref()
                .is_some_and(|root| root.join(&file.relative_path).is_file())
            {
                missing
                    .entry((identity.repository.clone(), identity.revision.clone()))
                    .or_insert_with(Vec::new)
                    .push(file.relative_path.to_string_lossy().into_owned());
            }
        }
    }
    for files in missing.values_mut() {
        files.sort();
        files.dedup();
    }
    missing
}

/// Dispatch one claimed job to its handler and reconcile the terminal state.
///
/// `shutdown` (sc-8845, F-043) is the process-shutdown [`CancelFlag`] tripped by
/// `run_job_with_shutdown` when SIGTERM/Ctrl-C arrives mid-job. The caller's
/// bounded-wait-then-terminal-`Canceled` write is what GUARANTEES no job dangles `running` even for a
/// handler that cannot observe the flag. The always-compiled placeholder path threads it directly so
/// the shutdown-during-job behavior is exercised on every target (the GPU handlers are macOS/candle-
/// gated). sc-9618 (F-043 follow-up): the dispatch below is scoped in [`with_shutdown_flag`], binding
/// the flag as a task-local so the per-engine GPU consumer loops (image `consume_gen_events`, video
/// `generate_video`, training `consume_training_events`, the shared `run_batched_analysis_job`, and the
/// image-detail loop) consult it via `shutdown_requested()` at their existing per-step cancel
/// checkpoints — tripping the engine cancel mid-step on quit instead of winding down at the grace
/// window. MLX and candle twins stay in sync because both funnel through those SAME shared consumers.
pub(super) async fn run_utility_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
    portable_model_binding: Option<PortableRouteBinding>,
    shutdown: gen_core::CancelFlag,
) {
    let job_id = job.id.clone();
    load_progress::with_job(
        api,
        &job_id,
        run_utility_job_inner(
            api,
            settings,
            http_client,
            job,
            portable_model_binding,
            shutdown,
        ),
    )
    .await;
}

async fn run_utility_job_inner(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
    portable_model_binding: Option<PortableRouteBinding>,
    shutdown: gen_core::CancelFlag,
) {
    // Bind the process-shutdown flag as a task-local for the whole dispatch (sc-9618, F-043 follow-up)
    // so the per-engine GPU consumer loops awaited below honor it at their per-step cancel checkpoints
    // (via `shutdown_requested()`), stopping a gen/prompt mid-step on quit instead of waiting out the
    // grace window — without threading the flag through every stream-handler signature. The placeholder
    // path keeps its explicit `&shutdown` (it's the always-compiled reference implementation).
    // Per-run metrics probe (epic 10402, sc-10404): reset the MLX peak-memory
    // window and start the background GPU-load/memory sampler before the job
    // runs, so peak memory + peak load cover the whole job. Consumed after the
    // handler returns to POST the hardware/timing block; settings + phase
    // timings are posted separately by the handlers and coalesce-merge server-side.
    let metrics_probe = job_metrics::JobMetricsProbe::start(&settings.gpu_id);
    let result = with_shutdown_flag(shutdown.clone(), async {
        // Cold pinned files are fetched under the frozen route before the byte guard. Existing
        // on-demand handlers may fetch too late: the guard must hash and lease the exact selected
        // closure before any loader starts.
        if let Some(binding) = portable_model_binding.as_ref() {
            binding.validate().map_err(|_| {
                (
                    "Model source unavailable.",
                    WorkerError::InvalidPayload("portable model binding is invalid".to_owned()),
                )
            })?;
            if binding.backend != crate::portable_runtime_backend(settings)
                || binding.platform != std::env::consts::OS
            {
                return Err((
                    "Model source unavailable.",
                    WorkerError::InvalidPayload(
                        "portable model route does not match this worker".to_owned(),
                    ),
                ));
            }
            for ((repo, revision), files) in missing_portable_revision_files(binding, settings) {
                crate::model_jobs::ensure_hf_files_cached(
                    api, settings, &job, &repo, &revision, &files,
                )
                .await
                .map_err(|error| ("Model source unavailable.", error))?;
            }
        }
        // sc-19708: the ONE pre-loader guard. Typed model-source admission happens here for every
        // job type, before any handler runs; handlers never carry availability or cache policy.
        //
        // sc-19707: run it on the BLOCKING pool. The guard walks the resolved-cache journal, stats
        // whole closures, and — when it leases a local bundle — takes the entry's shared artifact
        // lock through the blocking `FileExt::lock_shared`. An evictor holding that lock
        // exclusively would otherwise stall this runtime worker thread, not just this job.
        let guard_job_type = job.job_type.clone();
        let guard_payload = job.payload.clone();
        let guard_settings = settings.clone();
        let guard_binding = portable_model_binding.clone();
        let guard_task = crate::load_progress::spawn_blocking(move || {
            external_library_runtime::RuntimeSourceGuard::begin_with_binding(
                &guard_job_type,
                &guard_payload,
                &guard_settings,
                guard_binding.as_ref(),
            )
        });
        // The guard's admission pass can re-verify a multi-GB resolved bundle, which outlasts the
        // API's stale-worker timeout — the sc-13856 hazard at a new call site. Awaiting the handle
        // bare let the sweep mark a still-healthy worker offline mid-verification (the Krea bf16
        // 90s lost-heartbeat incident), so the wait must pump heartbeats on the progress interval.
        let source_guard = progress::heartbeat_while_blocking(
            api,
            settings,
            &job.id,
            "model source guard",
            guard_task,
        )
        .await
        .map_err(|error| ("Model source unavailable.", error))?;
        let mut dispatch_job = job.clone();
        dispatch_job.payload = source_guard.dispatch_payload().clone();
        let mut dispatch_settings = settings.clone();
        // The typed paths belong to the source guard's verified, immutable attempt snapshot.
        // The guard below outlives this dispatch, retaining its exclusive lease through load.
        dispatch_settings.verified_plan_layers = source_guard.plan_layers();
        let dispatch_result = dispatch_utility_job(
            api,
            &dispatch_settings,
            http_client,
            &dispatch_job,
            &shutdown,
        )
        .await;
        // Success releases the operation-owned source sessions; failure re-probes the exact bound
        // sources so a mid-load disconnect surfaces as the typed unavailable condition instead of
        // a raw loader error (sc-19708). An error with the source still present stays verbatim.
        match dispatch_result {
            Ok(()) => source_guard
                .finish_success()
                .map_err(|error| ("Model source session cleanup failed.", error)),
            Err((message, error)) => Err((message, source_guard.classify_failure(settings, error))),
        }
    })
    .await;
    // Model uploads are retry inputs: the model importer releases them only after durable
    // successful publication. LoRA imports retain their existing cleanup policy.
    if job.job_type == JobType::LoraImport {
        let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    }
    if let Err((message, error)) = result {
        handle_job_failure(api, settings, &job, &shutdown, message, error).await;
    }
    // Capture + POST the run's hardware metrics for every job type — including
    // failed/canceled runs, which still carry a meaningful peak + wall-clock
    // (epic 10402, sc-10404). Best-effort: never fails the job.
    let metrics = metrics_probe.finish().await;
    job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
    // Do not advertise Idle after a poison latch. The next loop turn reports Unhealthy and exits
    // before polling, so no scheduler observation can mistake this process for claimable between
    // the active job's truthful failure and the supervisor's replacement.
    if settings.gpu_id != "mlx" || mlx_worker_recovery::global().can_claim() {
        let _ = heartbeat(api, settings, WorkerStatus::Idle, None).await;
    }
}

/// Route the claimed type without changing its handler or lane-specific failure label.
async fn dispatch_utility_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    shutdown: &gen_core::CancelFlag,
) -> Result<(), (&'static str, WorkerError)> {
    match job.job_type {
        JobType::Placeholder => run_placeholder_job(api, settings, job, shutdown)
            .await
            .map_err(|error| ("Placeholder job failed.", error)),
        // Native MLX image generation, served in-process by the linked mlx-gen
        // engine on the macOS Apple-Silicon GPU worker (epic 3018). Off macOS the
        // capability is never advertised, so this arm is unreachable there.
        JobType::ImageGenerate => run_image_generate_job(api, settings, job)
            .await
            .map_err(|error| ("Image generation failed.", error)),
        // Plain Image Edit (sc-3513): the distinct `image_edit` job type (`mode=edit_image`
        // + `sourceAssetId`, epic 2427) shares the generate handler — it dispatches on
        // payload model+mode (qwen/flux2/sdxl edit streams), not job type. The API only
        // routes MLX-eligible edit models here (jobs_store::image_job_is_mlx_eligible); off
        // macOS the `image_edit` capability is never advertised, so this arm is unreachable.
        JobType::ImageEdit => run_image_generate_job(api, settings, job)
            .await
            .map_err(|error| ("Image edit failed.", error)),
        JobType::VectorGenerate => run_vector_job(api, settings, job)
            .await
            .map_err(|error| ("Vector generation failed.", error)),
        // Native MLX tile-ControlNet detail refine (epic 3041, sc-3060), served in-process
        // by the engine on the macOS Apple-Silicon GPU worker. Off macOS the capability is
        // never advertised, so this arm is unreachable there and the job remains queued.
        JobType::ImageDetail => run_image_detail_job(api, settings, job)
            .await
            .map_err(|error| ("Image detail enhancement failed.", error)),
        // SenseNova-U1 visual question answering + Document Studio interleave (epic 3180,
        // sc-3905). These bypass the `Generator` registry and call the concrete `T2iModel`
        // directly (text / text+images output the `GenerationOutput` contract can't express).
        // On Mac the API routes them here through `understanding_job_is_mlx_eligible`; off-Mac
        // the candle worker advertises the same capabilities and uses the candle handlers below.
        JobType::ImageVqa => run_vqa_job(api, settings, job)
            .await
            .map_err(|error| ("Visual question answering failed.", error)),
        JobType::ImageInterleave => run_interleave_job(api, settings, job)
            .await
            .map_err(|error| ("Interleaved generation failed.", error)),
        // Native MLX video generation, served in-process by the linked mlx-gen engine
        // on the macOS Apple-Silicon GPU worker (epic 3018). sc-3033 ships the runtime
        // + procedural stub; the real Wan (sc-3034) / LTX+audio (sc-3035) models link
        // their provider crates. Off macOS the capability is never advertised, so this
        // arm is unreachable there.
        // The clip-conditioning advanced video modes (epic 3040, sc-3522) share the video
        // generation handler — `run_video_generate_job` dispatches `extend_clip` /
        // `video_bridge` by the request `mode` into the LTX IC-LoRA `VideoClip` path. The API
        // only routes the LTX-eligible jobs here (`video_job_is_mlx_eligible`); off macOS the
        // VideoExtend/VideoBridge capabilities are never advertised, so these arms are
        // unreachable there (the procedural stub would otherwise ignore the conditioning).
        JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
            run_video_generate_job(api, settings, job)
                .await
                .map_err(|error| ("Video generation failed.", error))
        }
        // replace_person → native Wan-VACE (epic 3040, sc-3521): the `PersonReplace` job
        // type (and `video_generate` mode=`replace_person`) shares the video handler, which
        // dispatches on `mode == "replace_person"` to the engine `wan_vace` provider — the
        // native equivalent of the torch `WanVACEPipeline` path. The API routes only
        // MLX-eligible replace_person jobs here (`jobs_store::video_job_is_mlx_eligible`). The
        // off-Mac candle lane serves eligible Wan-VACE/SCAIL-2 replacement; unsupported models
        // are refused and remain queued.
        JobType::PersonReplace => run_video_generate_job(api, settings, job)
            .await
            .map_err(|error| ("Person replacement failed.", error)),
        // Pure audio synthesis (SceneWorks Audio Studio, epic 13400 / sc-13404): Kokoro TTS (and
        // future SFX/music) served in-process by the runtime's candle audio lane
        // (`inference_runtime::load_audio` → `catalog().audio()`). Advertised only by a worker
        // that links the audio registry (the macOS mlx worker, whose `runtime-macos` bundle ships
        // it default-on); a worker without the lane never advertises `audio_generate`, so this arm
        // is unreachable there and the job stays queued for a capable worker.
        JobType::AudioGenerate => run_audio_generate_job(api, settings, job)
            .await
            .map_err(|error| ("Audio generation failed.", error)),
        // Native MLX LoRA/LoKr training (epic 3039, sc-3043/3049), served in-process
        // by the linked mlx-gen engine on the macOS Apple-Silicon GPU worker. The API
        // routes only MLX-native families here (jobs_store::training_job_is_mlx_eligible);
        // unsupported shapes are refused and remain queued for a compatible native trainer.
        // Off macOS the execute capability is advertised only by a linked native trainer.
        JobType::LoraTrain => run_lora_train_job(api, settings, job)
            .await
            .map_err(|error| ("LoRA training failed.", error)),
        // ControlNet Training Studio (epic 10159, sc-10162): render the per-image control
        // condition from the plan's dataset (A1/A2), then train the control branch through the
        // same native executor as LoRA (`krea_control` → `krea_2_control`). Candle-only today; the
        // routing gate keeps it on a candle worker (or the linked mlx build), the stub fails loudly
        // elsewhere.
        JobType::ControlTraining => {
            control_training_jobs::run_control_training_job(api, settings, job)
                .await
                .map_err(|error| ("ControlNet training failed.", error))
        }
        // Native MLX JoyCaption dataset captioning (epic 3550, sc-3556). The API
        // routes only `captioner=joy_caption` jobs here; other captioners remain queued unless
        // another compatible native captioner registers.
        JobType::TrainingCaption => run_training_caption_job(api, settings, job)
            .await
            .map_err(|error| ("Training captioning failed.", error)),
        JobType::DatasetParquetImport => run_dataset_parquet_import_job(api, settings, job)
            .await
            .map_err(|error| ("Parquet dataset import failed.", error)),
        // Dataset Doctor CLIP-embedding analysis (sc-6535): the native MLX or Candle worker embeds
        // every dataset image (clip_vit_l14) and POSTs the content-hash sidecar.
        JobType::DatasetAnalysis => run_dataset_analysis_job(api, settings, job)
            .await
            .map_err(|error| ("Dataset analysis failed.", error)),
        JobType::CatalogAnalysis => run_catalog_analysis_job(api, settings, job)
            .await
            .map_err(|error| ("Catalog analysis failed.", error)),
        // Dataset Doctor face pass (sc-6538): the native SCRFD+ArcFace stack embeds the largest face of
        // each Person-dataset image and POSTs the face sidecar. MLX on Mac (`mlx-gen-face`), candle on
        // the candle lane; off both the handler returns a precise unsupported error.
        JobType::DatasetFaceAnalysis => run_dataset_face_analysis_job(api, settings, job)
            .await
            .map_err(|error| ("Dataset face analysis failed.", error)),
        // Training subject masks (sc-2126): SAM3 "person" segmentation of every dataset image,
        // POSTed to the mask sidecar. Routed by the `image_segment` capability; MLX on Mac, candle
        // off-Mac; off both the handler returns a precise unsupported error.
        JobType::DatasetSubjectMask => run_dataset_subject_mask_job(api, settings, job)
            .await
            .map_err(|error| ("Subject mask generation failed.", error)),
        // On-demand "compare image to another" likeness tool (sc-4415): scores a CANDIDATE asset
        // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer. MLX on Mac,
        // candle off-Mac; off both the handler returns a precise unsupported error. Like the
        // dataset-face pass, the job-type capability is gpu.rs-hardcoded (the face stack has no gen-core
        // registry), so a job stays queued rather than mis-claimed where the stack isn't linked.
        JobType::FaceLikenessCompare => run_face_likeness_compare_job(api, settings, job)
            .await
            .map_err(|error| ("Face likeness compare failed.", error)),
        // Native candle prompt refinement (epic 5095, sc-5525; consolidated onto candle-llm in sc-7404):
        // routes `prompt_refine` to the candle `core_llm::TextLlm` provider (candle-llama, resolved
        // model-first). The candle worker advertises `prompt_refine` only when `backend_candle_enabled`
        // (engines::registry_capabilities from the registered core_llm provider); without a native
        // text provider the capability is not advertised and the job remains queued.
        JobType::PromptRefine => run_prompt_refine_job(api, settings, job)
            .await
            .map_err(|error| ("Prompt refinement failed.", error)),
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
        JobType::LoraDownload => run_lora_download_job(api, settings, http_client, job)
            .await
            .map_err(|error| ("LoRA download failed.", error)),
        JobType::ModelImport => run_model_import_job(api, settings, http_client, job)
            .await
            .map_err(|error| ("Model import failed.", error)),
        JobType::ModelConvert => run_model_convert_job(api, settings, job)
            .await
            .map_err(|error| ("Model conversion failed.", error)),
        JobType::FrameExtract => run_frame_extract_job(api, settings, job)
            .await
            .map_err(|error| ("Frame extraction failed.", error)),
        JobType::TimelineExport => run_timeline_export_job(api, settings, job)
            .await
            .map_err(|error| ("Timeline export failed.", error)),
        JobType::PersonDetect => run_person_detect_job(api, settings, job)
            .await
            .map_err(|error| ("Person detection failed.", error)),
        // DWPose whole-body pose detection (epic 3482, sc-3487 Mac / sc-5496 off-Mac):
        // RTMW via onnxruntime, replacing the Python rtmlib path — CoreML EP on the
        // macOS MLX worker, CUDA EP on the off-Mac candle GPU worker. Available on Mac
        // AND the candle lane; on a candle-disabled box `PoseDetect` is never advertised, so the
        // job remains queued and this falls to the `_` arm only if called defensively.
        #[cfg(native_lane)]
        JobType::PoseDetect => run_pose_detect_job(api, settings, job)
            .await
            .map_err(|error| ("Pose detection failed.", error)),
        // SCRFD 5-point landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac + the candle
        // SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482), served in-process
        // for the Key Point Library. Available on Mac AND the candle lane; on a candle-disabled box
        // `KpsExtract` is never advertised, so the job remains queued and this falls to the `_` arm
        // only if called defensively.
        #[cfg(native_lane)]
        JobType::KpsExtract => run_kps_extract_job(api, settings, job)
            .await
            .map_err(|error| ("Keypoint extraction failed.", error)),
        // Image upscaling, served in-process by `upscale_jobs::run_image_upscale_job`: Real-ESRGAN
        // RRDBNet x2/x4 via onnxruntime/CoreML (epic 3482, sc-3489, Mac) + SeedVR2 one-step diffusion
        // (native MLX on Mac sc-4815 / candle CUDA on Windows sc-5928). Available on Mac AND the
        // Windows/CUDA candle lane; on a build without either lane `ImageUpscale` is never advertised,
        // so the job remains queued. The routing oracle admits only the engines each native lane serves.
        #[cfg(native_lane)]
        JobType::ImageUpscale => run_image_upscale_job(api, settings, job)
            .await
            .map_err(|error| ("Image upscale failed.", error)),
        // Dataset Doctor one-tap upscale (sc-6539): Real-ESRGAN over flagged low-res items, then
        // re-point each via the API. Same engine + worker lanes as image_upscale.
        #[cfg(native_lane)]
        JobType::DatasetUpscale => run_dataset_upscale_job(api, settings, job)
            .await
            .map_err(|error| ("Dataset upscale failed.", error)),
        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt segmentation,
        // served in-process by `segment_jobs::run_image_segment_job` — a box prompt → a binary
        // inpaint mask asset for the Image Editor. Advertised by both native workers; Candle
        // point prompts fail closed because the pinned provider exposes box PVS only.
        #[cfg(native_lane)]
        JobType::ImageSegment => segment_jobs::run_image_segment_job(api, settings, job)
            .await
            .map_err(|error| ("Smart-select segmentation failed.", error)),
        // Iris-3B monocular depth (sc-25682): the `iris_3b_depth` task export on the native MLX
        // worker (macOS) or the Candle worker (off-Mac); a build without either lane never
        // advertises `image_depth`, so the job remains queued.
        #[cfg(native_lane)]
        JobType::ImageDepth => depth_jobs::run_image_depth_job(api, settings, job)
            .await
            .map_err(|error| ("Depth estimation failed.", error)),
        // SeedVR2 video upscaling (epic 4811): one-step super-resolution — native MLX on Mac (sc-4816)
        // / candle CUDA on Windows (sc-5928). SceneWorks' first video upscaler: decodes the source
        // clip, runs the temporal-chunked 5D upscale, re-encodes, and passes the source audio through.
        // Available on Mac + the Windows/CUDA candle lane; elsewhere `VideoUpscale` is never advertised
        // (no torch path), so it falls to the `_` arm and the routing oracle reports it unsupported.
        #[cfg(native_lane)]
        JobType::VideoUpscale => video_jobs::seedvr2::run_video_upscale_job(api, settings, job)
            .await
            .map_err(|error| ("Video upscale failed.", error)),
        JobType::PersonTrack => run_person_track_job(api, settings, job)
            .await
            .map_err(|error| ("Person tracking failed.", error)),
        _ => {
            let result = fail_job(
                api,
                &job.id,
                "No utility exists for this job type.",
                Some(format!(
                    "Unsupported utility job type: {}",
                    job.job_type.as_str()
                )),
            )
            .await;
            result.map_err(|error| ("Utility job failed.", error))
        }
    }
}

/// Reconcile one handler failure before metrics and the final worker heartbeat.
async fn handle_job_failure(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    shutdown: &gen_core::CancelFlag,
    message: &'static str,
    error: WorkerError,
) {
    match error {
        WorkerError::Canceled(_) => {}
        error => {
            // sc-16247: this detail is what `QueueScreen` renders as `job.error`, so it is the
            // last point before a raw `DriverError(...)` reaches the user. `classify_engine_error`
            // already annotates the lanes that route through it (the reported krea_2_turbo path),
            // but ~35 other load seams build `WorkerError::Engine` directly — a host driver
            // problem hits all of them identically. Annotating here catches every one, and is a
            // no-op when the guidance is already present.
            let original_detail = error.to_string();
            let recovery_detail = if settings.gpu_id == "mlx" {
                mlx_worker_recovery::global().observe(&original_detail)
            } else {
                None
            };
            if let Some(detail) = &recovery_detail {
                emit_event_value(
                    Level::ERROR,
                    json!({
                        "event": "mlx_h3_i2v_worker_poisoned",
                        "workerId": settings.worker_id,
                        "gpuId": settings.gpu_id,
                        "jobId": job.id,
                        "originalError": original_detail,
                        "jobError": detail,
                    }),
                );
            }
            let poisoned = recovery_detail.is_some();
            let detail =
                recovery_detail.unwrap_or_else(|| annotate_cuda_driver_failure(&original_detail));
            if poisoned {
                // A clean child exit is not an abnormal death, so the supervisor deliberately
                // will not invent a second terminal attribution for it. Persist the original
                // timeout before arming that exit. Transport/API failures keep this process
                // quarantined and retry without claiming; shutdown cancellation stops the
                // retry and leaves the loop's no-claim backstop armed.
                let mut attempt = 0_u32;
                let _ = mlx_worker_recovery::persist_terminal_failure_with(
                    mlx_worker_recovery::global(),
                    || {
                        attempt = attempt.saturating_add(1);
                        let post_attempt = attempt;
                        let detail = detail.clone();
                        let job_id = job.id.clone();
                        let shutdown = shutdown.clone();
                        async move {
                            let Err(terminal_error) =
                                fail_job(api, &job_id, message, Some(detail)).await
                            else {
                                return mlx_worker_recovery::TerminalPersistenceAttempt::Persisted;
                            };
                            emit_event_value(
                                Level::ERROR,
                                json!({
                                    "event": "mlx_h3_i2v_terminal_failure_retry",
                                    "workerId": settings.worker_id,
                                    "gpuId": settings.gpu_id,
                                    "jobId": &job_id,
                                    "attempt": post_attempt,
                                    "error": terminal_error.to_string(),
                                }),
                            );
                            if shutdown.is_cancelled() {
                                return mlx_worker_recovery::TerminalPersistenceAttempt::Stop;
                            }
                            let _ = heartbeat_with_reason(
                                api,
                                settings,
                                WorkerStatus::Unhealthy,
                                Some(&job_id),
                                mlx_worker_recovery::global().quarantine_reason(),
                            )
                            .await;
                            tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1)))
                                .await;
                            mlx_worker_recovery::TerminalPersistenceAttempt::Retry
                        }
                    },
                )
                .await;
            } else {
                let _ = fail_job(api, &job.id, message, Some(detail)).await;
            }
            tracing::error!(
                event = "utility_job_failed",
                jobId = %job.id,
                error = %error,
                "{message}"
            );
        }
    }
}
