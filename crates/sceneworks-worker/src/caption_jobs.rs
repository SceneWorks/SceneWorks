//! Native dataset captioning (epic 3550, sc-3556 MLX; epic 5095 sc-5098 candle).
//!
//! SceneWorks keeps the existing `training_caption` job contract and result shape,
//! but the in-process Rust worker can serve `captioner=joy_caption` through the
//! backend-neutral `crate::inference_runtime::load_captioner` seam: the macOS `mlx` worker via mlx-gen's
//! JoyCaption provider, and the Windows/CUDA candle worker via candle-gen-joycaption
//! (`--features backend-candle`). `cfg(target_os)` only decides which provider crate registers the
//! captioner; the job flow below is identical. Without either native provider the capability is not
//! advertised and captioning jobs remain queued.

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const JOY_CAPTION_MODEL: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Training captioning canceled by user.";

// Coalesced per-step progress-post cadence (sc-11189, F-016 — the sc-8840 F-038 pattern the refine
// path already uses). The token callback publishes every decoded token into a latest-wins watch
// channel (never blocking decode); the job loop drains only the newest value on this tick, so a
// multi-image dataset emits a handful of `update_job` POSTs per second instead of hundreds of
// sequential per-token POSTs (which also throttled decode to API latency). 250 ms keeps the progress
// bar visibly smooth while bounding API load and decoupling decode speed from API latency.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PROGRESS_POST_INTERVAL: Duration = Duration::from_millis(250);

// epic 3720 (sc-3724): the backend-neutral captioner contract types come from `gen_core`; the
// selected runtime bundle explicitly includes its JoyCaption implementation.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
use gen_core::CAPTION_TRIGGER_WORD_CONFORMANCE;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    apply_caption_trigger_words, CancelFlag, CaptionOptions, CaptionRequest, CaptionSampling,
    Captioner, Image, LoadSpec, Progress, WeightsSource,
};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::training::CaptionMode;

/// The JoyCaption prompt a `subjectOnly` caption job sends (epic 2123, sc-24829). It keeps the
/// caption to what changes between images of one character (clothing, expression, pose,
/// accessories) and leaves out the background and the fixed identity traits, so a character LoRA
/// binds that identity to the trigger words (still prepended after generation) instead of to
/// caption tokens.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) const SUBJECT_ONLY_CAPTION_PROMPT: &str = "Write a short caption that describes only \
the changeable parts of the main subject in this image: their clothing, facial expression, pose, \
and any accessories they wear or hold. Do not describe the background, setting, location, \
lighting, or camera. Do not describe fixed identity traits such as face shape, eye color, hair \
color, skin tone, ethnicity, age, or body type. Do not name the subject.";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct CaptionItem {
    item_id: String,
    image_path: PathBuf,
    trigger_words: Vec<String>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct CaptionJobOptions {
    options: CaptionOptions,
    sampling: CaptionSampling,
}

/// The job's caption mode (sc-24829). An absent `mode` is `default` (jobs queued before modes
/// existed); an unrecognised one is refused rather than captioned as `default`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_job_mode(payload: &JsonObject) -> WorkerResult<CaptionMode> {
    match payload.get("mode") {
        None | Some(Value::Null) => Ok(CaptionMode::Default),
        Some(Value::String(mode)) => match mode.as_str() {
            "default" => Ok(CaptionMode::Default),
            "subjectOnly" => Ok(CaptionMode::SubjectOnly),
            "triggerOnly" => Ok(CaptionMode::TriggerOnly),
            other => Err(WorkerError::InvalidPayload(format!(
                "Unsupported training caption mode {other:?}; use default, subjectOnly, or triggerOnly."
            ))),
        },
        Some(other) => Err(WorkerError::InvalidPayload(format!(
            "Training caption mode must be a string, got {other}."
        ))),
    }
}

/// The prompt actually sent to the captioner for `mode`: the subject-only prompt for
/// `subjectOnly`, otherwise the caller's prompt override (empty means the provider renders its
/// type/length template, today's behaviour). `triggerOnly` never reaches a captioner.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_prompt_for_mode(mode: &CaptionMode, options: &CaptionOptions) -> String {
    match mode {
        CaptionMode::SubjectOnly => SUBJECT_ONLY_CAPTION_PROMPT.to_owned(),
        _ => options.custom_prompt.clone(),
    }
}

/// One stored caption as the `/caption-sidecars` route takes it: auto-sourced, stamped with the
/// mode that produced it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_sidecar_item(
    item_id: &str,
    text: &str,
    trigger_words: &[String],
    mode: &CaptionMode,
) -> Value {
    json!({
        "itemId": item_id,
        "caption": {
            "text": text,
            "source": "auto",
            "triggerWords": trigger_words,
            "mode": mode,
        }
    })
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// The meaningful per-image result streamed on the bounded mpsc channel (sc-11189, F-016). Per-STEP
// token progress no longer rides this channel — it is coalesced through a latest-wins watch channel
// (see `run_training_caption_job`) — so only the finished caption for each image is sent here, one
// send per image, back-pressured by design.
#[derive(Debug)]
struct CaptionedItem {
    index: usize,
    item_id: String,
    text: String,
    trigger_words: Vec<String>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_destination(payload: &JsonObject) -> WorkerResult<(String, String)> {
    Ok((
        required_payload_string(payload, "projectId")?.to_owned(),
        required_payload_string(payload, "datasetId")?.to_owned(),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_training_caption_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    run_training_caption_job_using(api, settings, job, crate::inference_runtime::load_captioner)
        .await
}

/// [`run_training_caption_job`] with the captioner loader injected (sc-24829), so a test drives the
/// REAL job path against a stub [`Captioner`] and can prove a `triggerOnly` job never calls it.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn run_training_caption_job_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    load_captioner: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Captioner>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    if job
        .payload
        .get("captioner")
        .and_then(Value::as_str)
        .unwrap_or_default()
        != "joy_caption"
    {
        return Err(WorkerError::InvalidPayload(
            "Unsupported training captioner for the native worker; use joy_caption.".to_owned(),
        ));
    }

    // Validate the persistence destination before reserving the GPU or loading weights.
    // These fields used to be checked only after inference had completed, wasting a full
    // caption run for a request that could never save its results.
    let (project_id, dataset_id) = caption_destination(&job.payload)?;
    let items = caption_items(settings, &job.payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Training caption job has no items to caption.".to_owned(),
        ));
    }
    let mode = caption_job_mode(&job.payload)?;
    if mode == CaptionMode::TriggerOnly {
        return run_trigger_only_caption_job(api, settings, job, &project_id, &dataset_id, &items)
            .await;
    }
    let options = caption_job_options(&job.payload);
    let model_name_or_path = job
        .payload
        .get("modelNameOrPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(JOY_CAPTION_MODEL)
        .to_owned();
    let weights_dir = resolve_caption_weights_dir(settings, &model_name_or_path)?;
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing training caption job.",
            None,
            backend,
        ),
    )
    .await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.08,
            &format!("Loading JoyCaption model ({}).", backend.to_uppercase()),
            None,
            backend,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    // Per-item captions ride a bounded mpsc (the meaningful per-image results — kept, one send per
    // image, back-pressured by design). Per-STEP token progress is coalesced through a latest-wins
    // **watch** channel instead of a POST per token (sc-11189, F-016 — the sc-8840 F-038 pattern the
    // refine path already uses): the callback publishes `(index, current, total)` non-blocking and
    // latest-wins (`send` never blocks decode and never drops the LATEST value), and the loop below
    // posts only the newest snapshot on a fixed tick. Decode is fully decoupled from API latency and
    // intermediate ticks are coalesced away.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<CaptionedItem>(64);
    let (progress_tx, progress_rx) = tokio::sync::watch::channel::<(usize, u32, u32)>((0, 0, 0));
    let blocking_cancel = cancel.clone();
    let blocking_items = items.clone();
    let blocking_options = options.clone();
    // The active backend label (mlx / candle / cpu), owned into the blocking task so its engine-error
    // strings name the REAL backend instead of a hardcoded "MLX" (sc-8916, F-114).
    let blocking_backend = backend.to_owned();
    let blocking_mode = mode.clone();
    let job_id = job.id.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        emit_event(
            "caption_pipeline_load_start",
            json!({
                "jobId": job_id,
                "engine": JOY_CAPTION_MODEL,
            }),
        );
        let captioner = load_captioner(
            JOY_CAPTION_MODEL,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .map_err(|error| {
            WorkerError::Engine(format!(
                "JoyCaption {} load failed: {error}",
                blocking_backend.to_uppercase()
            ))
        })?;
        emit_event(
            "caption_pipeline_load_complete",
            json!({
                "jobId": job_id,
                "engine": JOY_CAPTION_MODEL,
            }),
        );

        for (index, item) in blocking_items.into_iter().enumerate() {
            if blocking_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let image = load_caption_image(&item.image_path)?;
            let mut request = CaptionRequest {
                image,
                options: blocking_options.options.clone(),
                sampling: blocking_options.sampling,
                trigger_words: item.trigger_words.clone(),
                cancel: blocking_cancel.clone(),
                ..Default::default()
            };
            // The prompt that produced the caption is the mode's; the options' override field
            // mirrors it, as the gen-core contract asks (`custom_prompt` is `prompt`'s source).
            request.prompt = caption_prompt_for_mode(&blocking_mode, &request.options);
            request.options.custom_prompt = request.prompt.clone();
            // sc-24029: JoyCaption decodes up to 4096 tokens on the same mlx-llm KV cache, which
            // grows per token. Scoped to ONE item so each caption's terminal clear runs before the
            // next item's decode begins. MLX's freed-buffer cache is PROCESS-GLOBAL, not per-thread,
            // so that clear also discards buffers a concurrent image render had cached.
            //
            // KNOWN GAP (sc-24029): the interval bound is INERT here today. MLX JoyCaption's
            // `caption()` (inference `crates/media/mlx-gen/mlx-gen-joycaption/src/model.rs`) emits
            // exactly TWO `Progress::Step` events per item — `1/2` before `generate` and `2/2` after
            // it returns — so `note_event` is called twice per item and the 16-event interval is
            // never reached. This site therefore gets ONLY the terminal drop clear, and JoyCaption's
            // cache growth WITHIN one item is bounded by nothing but that end-of-item clear. Closing
            // the gap needs `mlx-gen-joycaption` to emit per-token progress from its own stream
            // callback — an inference-side change plus a pin bump, tracked and being done separately.
            // The wiring stays here so the bound starts working the moment that pin lands.
            let mut cache_bound = crate::mlx_decode_cache::DecodeCacheBound::mlx();
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { current, total } = progress {
                    cache_bound.note_event();
                    // Publish the latest `(index, current, total)` token count into the coalescing
                    // watch channel the loop below reads. `send` is non-blocking and latest-wins —
                    // token decode is NEVER back-pressured by API latency (the F-016 fix). A send
                    // error means every receiver was dropped (the consumer loop returned early on a
                    // POST failure / 409): trip the engine flag so the captioner bails instead of
                    // running unheard (sc-8804, F-003 — the swallowed-closed-channel leak, preserved
                    // verbatim from the old bounded-channel behavior).
                    if progress_tx.send((index, current, total)).is_err() {
                        blocking_cancel.cancel();
                    }
                }
            };
            let output = captioner
                .caption(&request, &mut on_progress)
                .map_err(|error| {
                    WorkerError::Engine(format!(
                        "JoyCaption {} generation failed: {error}",
                        blocking_backend.to_uppercase()
                    ))
                })?;
            // The backend-neutral `gen-core` contract owns trigger-word normalization and complete
            // token/phrase matching, so the MLX and Candle job paths cannot drift.
            let text = apply_caption_trigger_words(&output.text, &item.trigger_words);
            tx.blocking_send(CaptionedItem {
                index,
                item_id: item.item_id,
                text,
                trigger_words: item.trigger_words,
            })
            .map_err(|_| WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))?;
        }
        Ok(())
    });

    // Bind the blocking captioner task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` below returns early on a transient POST failure or a 409 (stale-sweep
    // reclaim); on that early return this guard trips `cancel` and aborts the captioner thread
    // instead of leaving it running on a job nobody is consuming. `cancel` is kept alongside (it's
    // `Clone`) for the in-loop cancel poll; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    // Heartbeat + poll-cancel cadence (the shared 5–15 s worker interval).
    let mut heartbeat_interval = tokio::time::interval(progress_report_interval(settings));
    heartbeat_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Coalesced per-step progress cadence (sc-11189, F-016): a fixed short tick that drains ONLY the
    // newest token count from the watch channel — hundreds of per-token POSTs collapse to a few, and
    // decode is never back-pressured by API latency. Decoupled from the (coarser) heartbeat interval
    // so per-step progress stays smooth without the token stream driving the API.
    let mut progress_interval = tokio::time::interval(PROGRESS_POST_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Skip a redundant post when the token count has not advanced since the last one (the watch holds
    // the same value between ticks), so a stalled decode does not re-POST identical progress.
    let mut last_posted: Option<(usize, u32, u32)> = None;
    // Defer the terminal `Canceled` until the blocking captioner has actually stopped (sc-11189,
    // F-017): a cancel poll here only TRIPS the engine flag (`cancel_requested_peek`, no terminal
    // write) and latches `canceled`; the terminal `Canceled` — which frees the worker row — is posted
    // below only AFTER `into_handle().await` returns, so the scheduler never sees a free worker while
    // the captioner is still on the GPU (mirrors `run_batched_analysis_job` sc-8917/F-115 + the
    // training path). The old path called `check_cancel` on the tick, which posted the terminal
    // `Canceled` at acknowledgement time.
    let mut canceled = false;
    let mut captions = Vec::with_capacity(items.len());
    // Run the stream loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(CaptionedItem { index, item_id, text, trigger_words }) => {
                        captions.push(caption_sidecar_item(&item_id, &text, &trigger_words, &mode));
                        let progress = 0.12 + 0.76 * ((index + 1) as f64 / items.len() as f64);
                        update_job(
                            api,
                            &job.id,
                            caption_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                progress,
                                &format!("Captioned image {} of {}.", index + 1, items.len()),
                                None,
                                backend,
                            ),
                        )
                        .await?;
                    }
                    None => break,
                }
            }
            _ = progress_interval.tick() => {
                // Coalesced per-step progress: post ONLY the latest `(index, current, total)`, and
                // only if it moved since the last post (a stalled decode holds the same value between
                // ticks → no redundant re-POST). Copy the `(usize, u32, u32)` out of the borrow first
                // so the non-`Send` `watch::Ref` is not held across the `.await` (keeps the enclosing
                // job future `Send` for rust-api's `tokio::spawn`).
                let latest = *progress_rx.borrow();
                if let Some((index, current, total)) = next_caption_step_post(latest, last_posted) {
                    update_job(
                        api,
                        &job.id,
                        caption_progress(
                            JobStatus::Running,
                            ProgressStage::Running,
                            caption_step_progress(index, current, total, items.len()),
                            &format!("Captioning image {} of {}.", index + 1, items.len()),
                            None,
                            backend,
                        ),
                    )
                    .await?;
                    last_posted = Some((index, current, total));
                }
            }
            _ = heartbeat_interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API poll
                // (a local flag read) so a quit trips the captioner flag at its next per-item check
                // instead of waiting out the grace window, exactly like a user cancel does. Here we
                // only TRIP the engine flag and latch `canceled`; the terminal `Canceled` is posted
                // below once the task has actually stopped (sc-11189, F-017).
                if !canceled && (shutdown_requested() || cancel_requested_peek(api, &job.id).await) {
                    cancel.cancel();
                    canceled = true;
                }
            }
        }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        // A progress/heartbeat POST failed (transient error or 409 stale-sweep reclaim): trip the
        // flag and bounded-join the captioner before returning so it isn't left running (sc-8804).
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly (channel closed) — reclaim the handle (disarming the drop-guard) and join.
    let join_result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("caption task join", error))?;
    if canceled {
        // The captioner has actually stopped now, so post the TERMINAL `Canceled` here (not at the
        // earlier cancel poll, which only tripped the flag) — this terminal write frees the worker row
        // as the worker returns to its claim loop, so the next queued job waits only until the GPU is
        // genuinely free (sc-11189, F-017; mirrors `run_batched_analysis_job` + the training path).
        mark_job_canceled(api, &job.id, CANCEL_MESSAGE).await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    join_result?;

    save_caption_results(
        api,
        job,
        CaptionSave {
            project_id: &project_id,
            dataset_id: &dataset_id,
            model_name_or_path: &model_name_or_path,
            mode: &mode,
            backend,
        },
        captions,
    )
    .await
}

/// Where and how a finished caption job's captions are saved.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct CaptionSave<'a> {
    project_id: &'a str,
    dataset_id: &'a str,
    model_name_or_path: &'a str,
    mode: &'a CaptionMode,
    backend: &'a str,
}

/// POST the captions to the dataset's `/caption-sidecars` route and complete the job: the shared
/// tail of the captioner and trigger-only paths.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn save_caption_results(
    api: &ApiClient,
    job: &JobSnapshot,
    save: CaptionSave<'_>,
    captions: Vec<Value>,
) -> WorkerResult<()> {
    let CaptionSave {
        project_id,
        dataset_id,
        model_name_or_path,
        mode,
        backend,
    } = save;
    let captioned_count = captions.len();
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            "Saving generated captions.",
            None,
            backend,
        ),
    )
    .await?;
    let sidecars: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars"
            ),
            &json!({ "items": captions }),
        )
        .await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Created captions for {captioned_count} training item(s)."),
            Some(caption_result(
                model_name_or_path,
                dataset_id,
                mode,
                captioned_count,
                sidecars,
            )),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// A `triggerOnly` caption job (sc-24829): every image's caption is exactly its trigger words,
/// written without resolving weights or loading a captioner.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn run_trigger_only_caption_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    dataset_id: &str,
    items: &[CaptionItem],
) -> WorkerResult<()> {
    let mode = CaptionMode::TriggerOnly;
    let captions = trigger_only_captions(items, &mode)?;
    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            &format!(
                "Writing trigger-word captions for {} image(s).",
                items.len()
            ),
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    save_caption_results(
        api,
        job,
        CaptionSave {
            project_id,
            dataset_id,
            model_name_or_path: "",
            mode: &mode,
            backend,
        },
        captions,
    )
    .await
}

/// The trigger-only sidecar items: each caption is the item's trigger words joined by the shared
/// gen-core trigger policy over an empty caption. An item with no trigger words would get an empty
/// caption, so it is refused (the API refuses the same job before queuing it).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn trigger_only_captions(items: &[CaptionItem], mode: &CaptionMode) -> WorkerResult<Vec<Value>> {
    items
        .iter()
        .map(|item| {
            let text = apply_caption_trigger_words("", &item.trigger_words);
            if text.is_empty() {
                return Err(WorkerError::InvalidPayload(format!(
                    "Caption item {} has no trigger words for a triggerOnly caption.",
                    item.item_id
                )));
            }
            Ok(caption_sidecar_item(
                &item.item_id,
                &text,
                &item.trigger_words,
                mode,
            ))
        })
        .collect()
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_training_caption_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "The in-process JoyCaption worker needs the macOS MLX backend or the candle backend \
         (build with --features backend-candle); use the Python torch captioner on this platform."
            .to_owned(),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_result(
    model_name_or_path: &str,
    dataset_id: &str,
    mode: &CaptionMode,
    captioned_count: usize,
    sidecars: Value,
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("captioner".to_owned(), json!("joy_caption"));
    result.insert("mode".to_owned(), json!(mode));
    // A trigger-only job loads no model, so it names none.
    if !model_name_or_path.is_empty() {
        result.insert("modelNameOrPath".to_owned(), json!(model_name_or_path));
    }
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert(
        "datasetVersion".to_owned(),
        sidecars
            .get("dataset")
            .and_then(|dataset| dataset.get("version"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert("captionedItemCount".to_owned(), json!(captioned_count));
    result.insert(
        "sidecars".to_owned(),
        sidecars
            .get("sidecars")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );
    result
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_items(settings: &Settings, payload: &JsonObject) -> WorkerResult<Vec<CaptionItem>> {
    let dataset_root = payload
        .get("datasetRoot")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Caption payload.datasetRoot must be an app-managed dataset path.".to_owned(),
            )
        })?;
    let items = payload
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Caption payload.items must be an array.".to_owned())
        })?;
    items
        .iter()
        .map(|item| {
            let object = item.as_object().ok_or_else(|| {
                WorkerError::InvalidPayload("Caption item must be an object.".to_owned())
            })?;
            let item_id = object
                .get("itemId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("Caption item is missing itemId.".to_owned())
                })?
                .to_owned();
            let image_path = object
                .get("imagePath")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Caption item {item_id} is missing imagePath."
                    ))
                })?;
            let trigger_words = object
                .get("triggerWords")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let image_path = resolve_dataset_item_path(
                settings,
                dataset_root,
                image_path,
                &format!("Caption item {item_id} imagePath"),
            )?;
            Ok(CaptionItem {
                item_id,
                image_path,
                trigger_words,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_job_options(payload: &JsonObject) -> CaptionJobOptions {
    let options = payload
        .get("options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    CaptionJobOptions {
        options: CaptionOptions {
            caption_type: option_string(&options, "captionType", "Descriptive"),
            caption_length: option_string(&options, "captionLength", "long"),
            extra_options: options
                .get("extraOptions")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            name_input: option_string(&options, "nameInput", ""),
            custom_prompt: option_string(&options, "captionPrompt", ""),
            low_vram: options
                .get("lowVram")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        sampling: CaptionSampling {
            temperature: value_f64(options.get("temperature").unwrap_or(&Value::Null), 0.6) as f32,
            top_p: value_f64(options.get("topP").unwrap_or(&Value::Null), 0.9) as f32,
            max_new_tokens: options
                .get("maxNewTokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(256),
            // sc-3963 engine knob: `None` keeps the per-call fresh seed (captions vary across
            // runs, the pre-bump behavior); an explicit `options.seed` reproduces a caption.
            seed: options.get("seed").and_then(Value::as_u64),
            // gen-core d8038beb (sc-7176 pin sync) exposed the JoyCaption repetition penalty that was
            // previously an internal engine constant. `..Default::default()` fills the new
            // `repetition_penalty` (1.05) + `repetition_context` (256) with the shipped values, so
            // captions stay byte-identical to the pre-bump behavior (and any future additive sampling
            // fields keep their engine defaults until the worker wires them).
            ..CaptionSampling::default()
        },
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn option_string(options: &JsonObject, key: &str, default: &str) -> String {
    options
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_caption_weights_dir(
    settings: &Settings,
    model_name_or_path: &str,
) -> WorkerResult<PathBuf> {
    resolve_app_managed_model_dir(settings, model_name_or_path, "JoyCaption model path")
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn load_caption_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("caption image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// The coalescing decision (sc-11189, F-016; mirrors `prompt_refine_jobs::next_progress_post`):
/// given the LATEST `(index, current, total)` token count from the watch channel and the value
/// already posted, return `Some(latest)` when it should be posted (it moved) or `None` when it is a
/// redundant repeat (a stalled decode holds the same value between ticks, so we don't re-POST
/// identical progress). Pure so the coalescing invariant — intermediate repeats dropped, every
/// distinct value posted exactly once — is unit-testable without an API or real weights (the caption
/// test module, like the whole caption path, is gated on the macOS/candle backends).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn next_caption_step_post(
    latest: (usize, u32, u32),
    last_posted: Option<(usize, u32, u32)>,
) -> Option<(usize, u32, u32)> {
    (last_posted != Some(latest)).then_some(latest)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_step_progress(index: usize, current: u32, total: u32, item_count: usize) -> f64 {
    let item_count = item_count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.12 + 0.76 * ((index as f64 + within) / item_count)).min(0.9)
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod tests {
    use super::*;

    fn test_settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        }
    }

    #[test]
    fn caption_job_options_preserve_training_surface() {
        let options = caption_job_options(&serde_json::Map::from_iter([(
            "options".to_owned(),
            json!({
                "captionType": "Straightforward",
                "captionLength": "40",
                "extraOptions": ["Mention lighting."],
                "nameInput": "Mira",
                "captionPrompt": "Use the custom prompt.",
                "temperature": 0.5,
                "topP": 0.8,
                "maxNewTokens": 128,
                "lowVram": true
            }),
        )]));
        assert_eq!(options.options.caption_type, "Straightforward");
        assert_eq!(options.options.caption_length, "40");
        assert_eq!(options.options.extra_options, vec!["Mention lighting."]);
        assert_eq!(options.options.name_input, "Mira");
        assert_eq!(options.options.custom_prompt, "Use the custom prompt.");
        assert!(options.options.low_vram);
        assert_eq!(options.sampling.temperature, 0.5);
        assert_eq!(options.sampling.top_p, 0.8);
        assert_eq!(options.sampling.max_new_tokens, 128);
    }

    #[test]
    fn caption_destination_rejects_missing_ids_before_inference() {
        let complete = serde_json::Map::from_iter([
            ("projectId".to_owned(), json!("project-1")),
            ("datasetId".to_owned(), json!("dataset-1")),
        ]);
        assert_eq!(
            caption_destination(&complete).expect("destination"),
            ("project-1".to_owned(), "dataset-1".to_owned())
        );
        assert!(caption_destination(&serde_json::Map::new()).is_err());
    }

    // ── sc-11189 (F-016): per-step token-progress coalescing (ported from the refine path) ────────

    // The coalescing decision: a value that MOVED is posted; a repeat of the last-posted value is
    // dropped (a stalled decode between ticks does not re-POST identical progress). Mirrors
    // `prompt_refine_jobs::next_progress_post_drops_repeats_and_emits_movement`, extended to the
    // `(index, current, total)` triple the caption path tracks (progress advances across images too).
    #[test]
    fn next_caption_step_post_drops_repeats_and_emits_movement() {
        // First observation (nothing posted yet) is always emitted.
        assert_eq!(next_caption_step_post((0, 5, 256), None), Some((0, 5, 256)));
        // An unchanged value after it was posted is a redundant repeat → dropped.
        assert_eq!(next_caption_step_post((0, 5, 256), Some((0, 5, 256))), None);
        // Token movement within the same image is emitted.
        assert_eq!(
            next_caption_step_post((0, 6, 256), Some((0, 5, 256))),
            Some((0, 6, 256))
        );
        // Advancing to the next image (index moved, token count reset) is emitted.
        assert_eq!(
            next_caption_step_post((1, 1, 256), Some((0, 6, 256))),
            Some((1, 1, 256))
        );
    }

    // The watch channel is latest-wins and non-blocking (the F-016 core): a fast token callback that
    // writes thousands of counts NEVER blocks decode, the consumer reading on a tick sees only the
    // newest value (intermediate ticks coalesced away), and the FINAL value is always readable after
    // the producer finishes — so the terminal per-image step progress can never be lost. Mirrors the
    // refine `watch_channel_coalesces_and_preserves_final_value` test.
    #[test]
    fn caption_step_watch_channel_coalesces_and_preserves_final_value() {
        let (tx, rx) = tokio::sync::watch::channel::<(usize, u32, u32)>((0, 0, 0));
        // Simulate the token callback for image 0: publish every token. `send` is non-blocking and
        // never drops the LATEST value, so a burst faster than any consumer cannot back-pressure it.
        for current in 1..=256u32 {
            tx.send((0, current, 256))
                .expect("receiver alive → send never errors");
        }
        // A consumer reading between bursts sees only the newest value, not each intermediate one.
        assert_eq!(*rx.borrow(), (0, 256, 256));

        // The final value survives the producer dropping (this image finished decoding).
        drop(tx);
        assert_eq!(*rx.borrow(), (0, 256, 256));

        // Feeding the resident value through the coalescing gate emits it once, then drops the repeat.
        let final_value = *rx.borrow();
        assert_eq!(
            next_caption_step_post(final_value, None),
            Some((0, 256, 256))
        );
        assert_eq!(next_caption_step_post(final_value, Some(final_value)), None);
    }

    // `send` signals a closed consumer (all receivers dropped): the token callback maps this to
    // tripping the engine cancel flag (sc-8804, F-003 — the captioner must not run unheard). Prove the
    // error surfaces when the receiver is gone, which is exactly the condition the callback keys on.
    #[test]
    fn caption_step_watch_send_errors_when_all_receivers_dropped() {
        let (tx, rx) = tokio::sync::watch::channel::<(usize, u32, u32)>((0, 0, 0));
        assert!(
            tx.send((0, 1, 256)).is_ok(),
            "send succeeds while a receiver lives"
        );
        drop(rx);
        assert!(
            tx.send((0, 2, 256)).is_err(),
            "send must error once every receiver is dropped (the consumer-gone signal)"
        );
    }

    #[test]
    fn caption_trigger_words_follow_gen_core_conformance_matrix() {
        for case in CAPTION_TRIGGER_WORD_CONFORMANCE {
            let triggers = case
                .trigger_words
                .iter()
                .map(|word| (*word).to_owned())
                .collect::<Vec<_>>();
            assert_eq!(
                apply_caption_trigger_words(case.caption, &triggers),
                case.expected
            );
        }
    }

    #[test]
    fn caption_items_require_ids_and_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        let image_path = dataset_root.join("image.png");
        let payload = serde_json::Map::from_iter([
            (
                "datasetRoot".to_owned(),
                json!(dataset_root.display().to_string()),
            ),
            (
                "items".to_owned(),
                json!([{
                    "itemId": "item_1",
                    "imagePath": image_path.display().to_string(),
                    "triggerWords": ["miraStyle", ""]
                }]),
            ),
        ]);
        let items = caption_items(&settings, &payload).expect("items parse");
        assert_eq!(items[0].item_id, "item_1");
        // sc-9812: path confinement now canonicalizes the deepest existing ancestor
        // before re-appending the (not-yet-created) tail, so the resolved image path
        // is expressed via the canonical tempdir root (on macOS `/var` -> `/private/var`).
        let expected = dir
            .path()
            .canonicalize()
            .expect("tempdir canonicalizes")
            .join("datasets")
            .join("ds-1")
            .join("image.png");
        assert_eq!(items[0].image_path, expected);
        assert_eq!(items[0].trigger_words, vec!["miraStyle"]);
    }

    #[test]
    fn caption_items_reject_paths_outside_dataset_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        let payload = serde_json::Map::from_iter([
            (
                "datasetRoot".to_owned(),
                json!(dataset_root.display().to_string()),
            ),
            (
                "items".to_owned(),
                json!([{
                    "itemId": "item_1",
                    "imagePath": dir.path().join("other.png").display().to_string()
                }]),
            ),
        ]);
        let error = caption_items(&settings, &payload).expect_err("unsafe image path rejected");
        assert!(
            error.to_string().contains("Caption item item_1 imagePath"),
            "{error}"
        );
    }

    // ── sc-24829: subject-only and trigger-only caption modes, driven through the REAL job path ──

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use axum::{
        extract::{Path as AxumPath, State},
        response::{IntoResponse, Response},
        routing::{get, post},
        Json, Router,
    };
    use gen_core::{CaptionCapabilities, CaptionOutput, CaptionerDescriptor};

    /// What the stub API saw: every progress POST and every `/caption-sidecars` body.
    #[derive(Clone, Default)]
    struct CaptionApi {
        progress: Arc<Mutex<Vec<Value>>>,
        sidecars: Arc<Mutex<Vec<Value>>>,
    }

    fn caption_job_value(job_id: &str, payload: Value) -> Value {
        json!({
            "id": job_id, "type": "training_caption", "status": "running",
            "projectId": null, "projectName": null, "payload": payload, "result": {},
            "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker",
            "progress": 0.0, "stage": "queued", "message": "queued", "error": null,
            "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": false,
            "createdAt": "2026-10-04T00:00:00Z", "updatedAt": "2026-10-04T00:00:00Z",
            "startedAt": null, "completedAt": null, "canceledAt": null, "lastHeartbeatAt": null
        })
    }

    async fn spawn_caption_api() -> (String, CaptionApi) {
        async fn job_route(AxumPath(job_id): AxumPath<String>) -> Response {
            Json(caption_job_value(&job_id, json!({}))).into_response()
        }
        async fn progress_route(
            State(state): State<CaptionApi>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            state.progress.lock().expect("progress lock").push(body);
            Json(caption_job_value(&job_id, json!({}))).into_response()
        }
        async fn heartbeat_route() -> Response {
            Json(json!({})).into_response()
        }
        async fn sidecars_route(
            State(state): State<CaptionApi>,
            Json(body): Json<Value>,
        ) -> Response {
            state.sidecars.lock().expect("sidecars lock").push(body);
            Json(json!({ "dataset": { "version": 2 }, "sidecars": [] })).into_response()
        }
        let state = CaptionApi::default();
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_route))
            .route("/api/v1/jobs/:job_id/progress", post(progress_route))
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(heartbeat_route),
            )
            .route(
                "/api/v1/projects/:project_id/training/datasets/:dataset_id/caption-sidecars",
                post(sidecars_route),
            )
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let address = listener.local_addr().expect("listener has address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serves");
        });
        (format!("http://{address}"), state)
    }

    /// A captioner that records the prompt of every request and answers a fixed caption.
    struct RecordingCaptioner {
        descriptor: CaptionerDescriptor,
        prompts: Arc<Mutex<Vec<String>>>,
    }

    impl Captioner for RecordingCaptioner {
        fn descriptor(&self) -> &CaptionerDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &CaptionRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn caption(
            &self,
            req: &CaptionRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<CaptionOutput> {
            self.prompts
                .lock()
                .expect("prompts lock")
                .push(req.prompt.clone());
            Ok(CaptionOutput {
                text: "a red jacket".to_owned(),
                generated_tokens: None,
                finish_reason: None,
            })
        }
    }

    /// A data dir holding one dataset image and a JoyCaption weights dir the loader is pointed at.
    struct CaptionStage {
        _dir: tempfile::TempDir,
        settings: Settings,
        dataset_root: PathBuf,
        weights_dir: PathBuf,
    }

    fn caption_stage(api_url: String) -> CaptionStage {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut settings = test_settings(dir.path());
        settings.api_url = api_url;
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        image::RgbImage::from_pixel(8, 8, image::Rgb([200, 40, 40]))
            .save(dataset_root.join("item_1.png"))
            .expect("dataset image writes");
        let weights_dir = dir.path().join("models").join("joycaption");
        std::fs::create_dir_all(&weights_dir).expect("weights dir");
        CaptionStage {
            _dir: dir,
            settings,
            dataset_root,
            weights_dir,
        }
    }

    fn caption_payload(stage: &CaptionStage, mode: Option<&str>, items: Value) -> Value {
        let mut payload = json!({
            "captioner": "joy_caption",
            "modelNameOrPath": stage.weights_dir.display().to_string(),
            "projectId": "project-1",
            "datasetId": "ds-1",
            "datasetRoot": stage.dataset_root.display().to_string(),
            "options": { "captionPrompt": "Describe everything you see." },
            "items": items,
        });
        if let Some(mode) = mode {
            payload["mode"] = json!(mode);
        }
        payload
    }

    fn one_item(stage: &CaptionStage) -> Value {
        json!([{
            "itemId": "item_1",
            "imagePath": stage.dataset_root.join("item_1.png").display().to_string(),
            "triggerWords": ["miraStyle"],
        }])
    }

    /// Run a caption job through the real path with a recording loader. Returns the job result,
    /// the prompts the captioner received, and whether the loader ran at all.
    async fn run_caption(
        api: &ApiClient,
        stage: &CaptionStage,
        payload: Value,
    ) -> (WorkerResult<()>, Vec<String>, bool) {
        let job: JobSnapshot =
            serde_json::from_value(caption_job_value("caption-job", payload)).expect("job");
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let loaded = Arc::new(AtomicBool::new(false));
        let (loader_prompts, loader_flag) = (prompts.clone(), loaded.clone());
        let result = run_training_caption_job_using(api, &stage.settings, &job, move |_, _| {
            loader_flag.store(true, Ordering::SeqCst);
            Ok(Box::new(RecordingCaptioner {
                descriptor: CaptionerDescriptor {
                    id: "joy_caption",
                    family: "joycaption",
                    backend: "stub",
                    capabilities: CaptionCapabilities::default(),
                },
                prompts: loader_prompts,
            }) as Box<dyn Captioner>)
        })
        .await;
        let prompts = prompts.lock().expect("prompts lock").clone();
        (result, prompts, loaded.load(Ordering::SeqCst))
    }

    fn posted_captions(api: &CaptionApi) -> Vec<Value> {
        let sidecars = api.sidecars.lock().expect("sidecars lock");
        assert_eq!(sidecars.len(), 1, "exactly one caption-sidecars POST");
        sidecars[0]["items"].as_array().expect("items").clone()
    }

    fn completed_result(api: &CaptionApi) -> Value {
        api.progress
            .lock()
            .expect("progress lock")
            .iter()
            .find(|body| body["status"] == "completed")
            .expect("job completed")["result"]
            .clone()
    }

    #[tokio::test]
    async fn subject_only_mode_sends_the_subject_only_prompt_and_records_the_mode() {
        let (url, api_state) = spawn_caption_api().await;
        let stage = caption_stage(url);
        let api = ApiClient::new(&stage.settings);
        let payload = caption_payload(&stage, Some("subjectOnly"), one_item(&stage));
        let (result, prompts, loaded) = run_caption(&api, &stage, payload).await;
        result.expect("subject-only caption job succeeds");
        assert!(loaded, "subject-only captions run JoyCaption");
        // The captioner saw the subject-only prompt, not the caller's full-scene prompt.
        assert_eq!(prompts, vec![SUBJECT_ONLY_CAPTION_PROMPT.to_owned()]);
        let captions = posted_captions(&api_state);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0]["itemId"], "item_1");
        assert_eq!(captions[0]["caption"]["mode"], "subjectOnly");
        assert_eq!(captions[0]["caption"]["source"], "auto");
        // Trigger words are still prepended to the generated caption.
        assert_eq!(captions[0]["caption"]["text"], "miraStyle, a red jacket");
        assert_eq!(completed_result(&api_state)["mode"], "subjectOnly");
    }

    #[tokio::test]
    async fn default_mode_keeps_the_callers_prompt_and_records_default() {
        let (url, api_state) = spawn_caption_api().await;
        let stage = caption_stage(url);
        let api = ApiClient::new(&stage.settings);
        // No `mode` at all: a job queued before modes existed is captioned exactly as before.
        let payload = caption_payload(&stage, None, one_item(&stage));
        let (result, prompts, loaded) = run_caption(&api, &stage, payload).await;
        result.expect("default caption job succeeds");
        assert!(loaded);
        assert_eq!(prompts, vec!["Describe everything you see.".to_owned()]);
        let captions = posted_captions(&api_state);
        assert_eq!(captions[0]["caption"]["mode"], "default");
        assert_eq!(captions[0]["caption"]["text"], "miraStyle, a red jacket");
    }

    #[tokio::test]
    async fn trigger_only_mode_writes_exactly_the_trigger_words_without_loading_the_captioner() {
        let (url, api_state) = spawn_caption_api().await;
        let stage = caption_stage(url);
        let api = ApiClient::new(&stage.settings);
        let mut payload = caption_payload(
            &stage,
            Some("triggerOnly"),
            json!([
                {
                    "itemId": "item_1",
                    "imagePath": stage.dataset_root.join("item_1.png").display().to_string(),
                    "triggerWords": ["miraStyle", "red coat"],
                },
                {
                    // No image on disk: trigger-only never opens the image either.
                    "itemId": "item_2",
                    "imagePath": stage.dataset_root.join("item_2.png").display().to_string(),
                    "triggerWords": ["miraStyle"],
                },
            ]),
        );
        // A model that is not installed: trigger-only must not resolve weights.
        payload["modelNameOrPath"] = json!("missing/joycaption-not-cached");
        let (result, prompts, loaded) = run_caption(&api, &stage, payload).await;
        result.expect("trigger-only caption job succeeds");
        assert!(!loaded, "trigger-only must never load the captioner");
        assert!(prompts.is_empty());
        let captions = posted_captions(&api_state);
        let texts: Vec<_> = captions
            .iter()
            .map(|item| {
                (
                    item["itemId"].as_str().unwrap().to_owned(),
                    item["caption"]["text"].as_str().unwrap().to_owned(),
                    item["caption"]["mode"].as_str().unwrap().to_owned(),
                )
            })
            .collect();
        assert_eq!(
            texts,
            vec![
                (
                    "item_1".to_owned(),
                    "miraStyle, red coat".to_owned(),
                    "triggerOnly".to_owned()
                ),
                (
                    "item_2".to_owned(),
                    "miraStyle".to_owned(),
                    "triggerOnly".to_owned()
                ),
            ]
        );
        let result = completed_result(&api_state);
        assert_eq!(result["mode"], "triggerOnly");
        assert!(result.get("modelNameOrPath").is_none(), "{result}");
    }

    #[tokio::test]
    async fn trigger_only_mode_refuses_an_image_without_trigger_words() {
        let (url, api_state) = spawn_caption_api().await;
        let stage = caption_stage(url);
        let api = ApiClient::new(&stage.settings);
        let payload = caption_payload(
            &stage,
            Some("triggerOnly"),
            json!([{
                "itemId": "item_1",
                "imagePath": stage.dataset_root.join("item_1.png").display().to_string(),
                "triggerWords": ["  "],
            }]),
        );
        let (result, _, loaded) = run_caption(&api, &stage, payload).await;
        let error = result.expect_err("an empty trigger-only caption is refused");
        assert!(error.to_string().contains("no trigger words"), "{error}");
        assert!(!loaded);
        assert!(
            api_state.sidecars.lock().unwrap().is_empty(),
            "nothing saved"
        );
    }

    #[tokio::test]
    async fn an_unknown_caption_mode_is_refused_before_loading() {
        let (url, api_state) = spawn_caption_api().await;
        let stage = caption_stage(url);
        let api = ApiClient::new(&stage.settings);
        let payload = caption_payload(&stage, Some("faceOnly"), one_item(&stage));
        let (result, _, loaded) = run_caption(&api, &stage, payload).await;
        let error = result.expect_err("unknown mode refused");
        assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error}");
        assert!(error.to_string().contains("faceOnly"), "{error}");
        assert!(!loaded);
        assert!(api_state.sidecars.lock().unwrap().is_empty());
    }
}
