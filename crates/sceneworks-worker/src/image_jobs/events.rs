use super::artifact_resolution::resolve_completed_image_metrics_tier_dir;
use super::begin_image_cancel;
use super::flux3_references;
use super::image_progress;
use super::lane_controls;
use super::memory::progress_with_auto_tier_streaming;
use super::memory::AutoTierStreamingDisclosure;
use super::metrics::build_image_metrics;
#[cfg(not(native_lane))]
use super::metrics::build_image_metrics;
use super::prompt_upsampling;
use super::publish_upsampling_progress;
use super::render::step_fraction;
use super::stream::GenEvent;
use super::streaming_result;
use super::write_image_asset;
use super::CancelFlag;
use super::ImagePlan;
use super::PROMPT_ENHANCEMENT_FACT_KEY;
use crate::cancel_requested_peek;
use crate::emit_event;
use crate::heartbeat;
use crate::progress_report_interval;
use crate::shutdown_requested;
use crate::task_join_error;
use crate::update_job;
use crate::ApiClient;
use crate::CancelJoinGuard;
use crate::Duration;
use crate::Instant;
use crate::JobSnapshot;
use crate::JobStatus;
use crate::JsonObject;
use crate::Path;
use crate::ProgressStage;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use crate::WorkerStatus;
use serde_json::json;
pub(super) type PromptEnhancementReports =
    std::collections::HashMap<usize, (String, gen_core::PromptEnhancementReport)>;

pub(super) fn record_prompt_enhancement_report(
    reports: &mut PromptEnhancementReports,
    enhancement_expected: bool,
    image_count: usize,
    index: usize,
    expected_prompt: String,
    report: gen_core::PromptEnhancementReport,
) -> WorkerResult<()> {
    if !enhancement_expected {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report for a disabled request".to_owned(),
        ));
    }
    if index >= image_count {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report for an unknown image".to_owned(),
        ));
    }
    match reports.entry(index) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert((expected_prompt, report));
        }
        std::collections::hash_map::Entry::Occupied(_) => {
            return Err(WorkerError::Engine(format!(
                "provider emitted duplicate prompt-enhancement reports for image {index}"
            )));
        }
    }
    Ok(())
}

pub(super) fn take_prompt_enhancement_fact(
    reports: &mut PromptEnhancementReports,
    index: usize,
    model: &str,
) -> WorkerResult<Value> {
    let (expected_prompt, report) = reports.remove(&index).ok_or_else(|| {
        WorkerError::Engine(format!(
            "provider emitted no prompt-enhancement report for image {index}"
        ))
    })?;
    prompt_upsampling::prompt_enhancement_fact_for_model(report, &expected_prompt, model)
}

/// Latest per-step preview frame for the in-flight image (epic 16624, sc-16904). One slot,
/// latest-wins: `ProgressRequest.result` is replaced wholesale on every accepted POST, so the
/// frame rides the existing per-step update and is never accumulated. Cleared when its image's
/// final asset lands (`GenEvent::Image`), and absent from the terminal result by construction.
pub(super) struct PreviewSlot {
    pub(super) index: usize,
    pub(super) current: u32,
    pub(super) total: u32,
    pub(super) data_url: String,
}

/// Encode one latent-resolution preview frame (~128×128 RGB8) as a `data:image/jpeg` URL,
/// ~10 KB at quality 70. Decorative by contract: any failure drops the frame, never the job.
pub(super) fn encode_preview_data_url(frame: &gen_core::PreviewFrame) -> Option<String> {
    use base64::Engine as _;
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut std::io::Cursor::new(&mut jpeg), 70)
        .encode(
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            image::ExtendedColorType::Rgb8,
        )
        .inspect_err(|error| {
            tracing::debug!(%error, "preview frame JPEG encode failed; dropping frame");
        })
        .ok()?;
    Some(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&jpeg)
    ))
}

/// [`slim_step_result`] plus the current preview frame, when one is live — the STEP-tick
/// result posted on every `GenEvent::Step`. Deliberately O(1): the accumulated
/// `assetWrites`/`generationSet`/`prefixKvCache` fact sets are omitted here (streamed on the
/// `Image` / `Decoding` / `Loading` arms instead) because the per-step cadence is
/// steps-per-image × images per batch and the accumulated set had begun OOM-killing the
/// WebView2 renderer mid-batch (see `streaming_result`'s RESOLVED TRADEOFF note, sc-8953).
/// The store restores any omitted accumulation into the durable row, so a slim step tick
/// can never regress it.
pub(super) fn slim_step_result_with_preview(
    plan: &ImagePlan,
    preview: Option<&PreviewSlot>,
) -> JsonObject {
    let mut result = crate::image_jobs::slim_step_result(plan);
    if let Some(slot) = preview {
        result.insert(
            "previewFrame".to_owned(),
            json!({
                "imageIndex": slot.index,
                "current": slot.current,
                "total": slot.total,
                "dataUrl": slot.data_url,
            }),
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn consume_gen_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    total: usize,
    rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    consume_gen_events_with_disclosure(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        raw_settings,
        None,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn consume_gen_events_with_disclosure(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    auto_tier_streaming: Option<&AutoTierStreamingDisclosure>,
    total: usize,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let total_u32 = total as u32;
    // A native prompt batch (sc-25681) is ONE engine call over all `total` images: its steps
    // (the engine's `steps × count`) span the whole bar, not image 0's slice of it.
    // So is a native count batch (sc-24215, `image.nativeCountBatch`): one call over `count` images.
    let one_pass = !plan.request.prompts.is_empty()
        || sceneworks_core::image_sampling_controls::native_count_batch(
            &plan.request.model_manifest_entry,
        );
    let (step_slices, step_label) = if one_pass {
        (1, format!("Batch of {total}"))
    } else {
        (total_u32, String::new())
    };
    let step_text = |index: usize, detail: String| {
        if one_pass {
            format!("{step_label} — {detail}.")
        } else {
            format!("Image {}/{total} — {detail}.", index + 1)
        }
    };
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    // Per-image inference lifecycle events (sc-3450), parity with the Python worker's
    // `image_inference_start`/`image_inference_complete`. The first event for an index
    // marks its start; `GenEvent::Image` marks completion. This is the single shared
    // streaming seam, so every MLX image family reports the same phases on mlx-worker.log
    // + the in-app Logs screen.
    let mut started: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut mark_started = |index: usize| {
        if started.insert(index) {
            emit_event(
                "image_inference_start",
                json!({
                    "jobId": job.id,
                    "imageIndex": index,
                    "imageCount": total,
                    "backend": backend,
                }),
            );
        }
    };
    // Bind the blocking generation task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` in the loop below returns early on a transient POST failure or a 409
    // (stale-sweep reclaim); on that early return this guard trips the engine `CancelFlag` and
    // aborts the still-running denoise instead of leaving it burning GPU memory alongside the next
    // claimed job. `cancel` is kept alongside (it's `Clone`) for the in-loop `begin_image_cancel`
    // poller; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    // Heartbeat + cancel-poll on a fixed interval, not only when the blocking
    // thread emits an event. The cold model-load phase (multi-GB load + quantize)
    // emits nothing, so without an interval arm the worker reports no Busy
    // heartbeat and honors no cancel until the first denoise step — long enough
    // for the API's staleness check to think it died (sc-4276 / F-MLXW-12;
    // mirrors the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Per-phase wall-clock (epic 10402, sc-10405): derive load/sample/decode spans
    // from this shared event stream — load = start→first Step, sample = Step→Decoding,
    // decode = Decoding→Image — summed across the batch's images. Both the MLX and
    // candle image lanes funnel through here, so both get the split. Posted best-effort
    // at clean completion; coalesce-merges with the S2 hardware block server-side.
    let mut phase_timer = crate::job_metrics::PhaseTimer::new(Instant::now());
    // Effective denoise step count (sc-10406): the Step event's `total` is the
    // resolved step count, so a default run reports real steps, not the sparse payload.
    let mut effective_steps: Option<u32> = None;
    // Live denoise preview (sc-16904): the latest frame rides the next progress POST.
    let mut latest_preview: Option<PreviewSlot> = None;
    let prompt_enhancement_expected = raw_settings
        .get("enhancePrompt")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut prompt_enhancement_reports = PromptEnhancementReports::new();
    // The measured three-fact set of this job's load, once the producing lane sends one
    // (sc-11045). `None` for every lane that holds no runtime handle — those assets keep whatever
    // the route stamped pre-load (including the honest `no-runtime-receipt` materialization).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let mut measured_checkpoint_facts: Option<Value> = None;
    #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
    let measured_checkpoint_facts: Option<Value> = None;
    // Per-image provenance facts (sc-25681), keyed by image index until that image is written.
    let mut pending_image_facts: std::collections::HashMap<usize, JsonObject> =
        std::collections::HashMap::new();
    // Run the event loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                if canceled {
                    continue; // drain remaining events so the blocking sender never blocks.
                }
                match event {
            GenEvent::Step {
                index,
                current,
                total: step_total,
            } => {
                mark_started(index);
                phase_timer.mark_sample_step(Instant::now());
                if effective_steps.is_none() {
                    effective_steps = Some(step_total);
                }
                if last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll so a quit stops the gen at this step, matching a user cancel.
                    if shutdown_requested() || cancel_requested_peek(api, &job.id).await {
                        // Trip the flag + show "Cancelling…", but stay non-terminal until the
                        // in-flight image actually stops (terminal Canceled posted after the
                        // blocking run returns) — sc-5515.
                        begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                        canceled = true;
                        continue;
                    }
                }
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, current, step_total, step_slices),
                        &progress_with_auto_tier_streaming(
                            &step_text(index, format!("step {current}/{step_total}")),
                            auto_tier_streaming,
                        ),
                        // Slim on purpose (sc-8953 follow-through): the fact set does not
                        // change on a step tick, so the accumulated asset writes must NOT
                        // ride it — see `slim_step_result_with_preview`.
                        Some(slim_step_result_with_preview(
                            plan,
                            latest_preview.as_ref(),
                        )),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Upsampling { index, tokens, max_tokens } => {
                mark_started(index);
                // This is before denoise: do not mark a sampling phase or invent effective_steps.
                // The select loop's independent interval still polls cancellation during prefill
                // and between coalesced token updates.
                if publish_upsampling_progress(tokens, max_tokens) {
                    update_job(
                        api,
                        &job.id,
                        image_progress(
                            JobStatus::Running,
                            ProgressStage::Generating,
                            step_fraction(index, 0, 1, step_slices),
                            &step_text(index, format!("enhancing prompt ({tokens}/{max_tokens} tokens)")),
                            Some(streaming_result(plan, asset_writes)),
                            backend,
                        ),
                    ).await?;
                }
            }
            GenEvent::Decoding { index } => {
                mark_started(index);
                phase_timer.mark_decoding(Instant::now());
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, 1, 1, step_slices),
                        &step_text(index, "decoding".to_owned()),
                        // Full result (asset writes CAN change on a decode), but the
                        // preview frame is the interim image's; the decode arm's own
                        // result carries no frame and `streaming_result` alone is right.
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Loading { index, phase } => {
                mark_started(index);
                let (status, stage) = crate::load_progress::generator_phase_stage(phase);
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        status,
                        stage,
                        step_fraction(index, 0, 1, total_u32),
                        &progress_with_auto_tier_streaming(
                            &format!("Image {}/{total} — {}", index + 1,
                                crate::load_progress::generator_phase(phase)),
                            auto_tier_streaming,
                        ),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Preview { index, frame } => {
                mark_started(index);
                // Encode off the async runtime thread (parity with the asset write, sc-8909).
                // No POST here: the frame rides the NEXT Step/Decoding update, so preview traffic
                // adds zero requests to the existing per-step cadence.
                let encoded = crate::load_progress::spawn_blocking(move || {
                    let data_url = encode_preview_data_url(&frame)?;
                    Some(PreviewSlot {
                        index,
                        current: frame.current,
                        total: frame.total,
                        data_url,
                    })
                })
                .await
                .map_err(|error| crate::task_join_error("preview frame encode task", error))?;
                if let Some(slot) = encoded {
                    latest_preview = Some(slot);
                }
            }
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            GenEvent::CheckpointWeightFacts { facts } => {
                // Measured checkpoint facts from the loaded handle (sc-11045). Latest-wins on
                // purpose: the engine's own sink is last-write-wins per materialization, so the
                // newest event always describes the most recent read.
                measured_checkpoint_facts = Some(facts);
            }
            GenEvent::ImageFacts { index, facts } => {
                pending_image_facts.entry(index).or_default().extend(facts);
            }
            GenEvent::AssetWritten { index, fact } => {
                phase_timer.mark_item_done(Instant::now());
                asset_writes.push(Value::Object(fact));
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated output {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
            GenEvent::PromptEnhancement {
                index,
                expected_prompt,
                report,
            } => {
                record_prompt_enhancement_report(
                    &mut prompt_enhancement_reports,
                    prompt_enhancement_expected,
                    total,
                    index,
                    expected_prompt,
                    report,
                )?;
            }
            GenEvent::Image {
                index,
                seed,
                width,
                height,
                pixels,
                face_likeness,
            } => {
                phase_timer.mark_item_done(Instant::now());
                // The finished asset supersedes its interim preview: drop the slot before this
                // arm's result POST so the replaced `result` no longer carries the frame (a later
                // image in the batch starts its own stream).
                if latest_preview
                    .as_ref()
                    .is_some_and(|slot| slot.index == index)
                {
                    latest_preview = None;
                }
                // The identity-likeness post-pass (sc-4409) scores each image on the blocking thread
                // and hands the pre-built `faceLikeness` block back through the event. Attach it to a
                // PER-IMAGE clone of the shared raw settings under the sidecar key so each angle's
                // asset carries its own honest score (an N/A `detected:false` block for profile/up/
                // down views), while every non-scoring path leaves `face_likeness` `None` ⇒ the field
                // is omitted entirely (the sc-4408 omit-when-absent contract).
                let mut image_raw_settings = raw_settings.clone();
                // This key is worker-owned even for legacy/raw jobs. Only the typed provider report
                // can add it back, after matching it to this image's exact requested prompt.
                image_raw_settings.remove(PROMPT_ENHANCEMENT_FACT_KEY);
                if prompt_enhancement_expected {
                    image_raw_settings.insert(
                        PROMPT_ENHANCEMENT_FACT_KEY.to_owned(),
                        take_prompt_enhancement_fact(&mut prompt_enhancement_reports, index, &plan.request.model)?,
                    );
                }
                if let Some(block) = face_likeness {
                    image_raw_settings.insert(
                        crate::face_likeness::FACE_LIKENESS_FACT_KEY.to_owned(),
                        Value::Object(block),
                    );
                }
                // Worker-owned like the keys above: a replayed `advanced` block can never forge
                // them, and only this image's own reported facts add them back (sc-25681). The
                // prefix-KV-cache record (epic 25899) is the same: it says which path RAN, so only
                // this image's own report may carry it onto the asset and the job result.
                image_raw_settings.remove(lane_controls::GENERATION_WARNINGS_KEY);
                image_raw_settings.remove(lane_controls::ADAPTER_APPLY_KEY);
                image_raw_settings.remove(lane_controls::PREFIX_KV_CACHE_KEY);
                flux3_references::merge_image_facts(
                    &mut image_raw_settings,
                    pending_image_facts.remove(&index),
                );
                // A measured fact set from the loaded handle supersedes the route's pre-load stamp
                // (whose materialization is honestly `no-runtime-receipt`): same three facts, now
                // with what the load actually materialized (sc-11045).
                if let Some(facts) = measured_checkpoint_facts.as_ref() {
                    image_raw_settings.insert(
                        crate::checkpoint_weight_facts_host::FACTS_RAW_SETTINGS_KEY.to_owned(),
                        facts.clone(),
                    );
                }
                // Encode + write the asset PNG off the async runtime thread (sc-8909 / F-107).
                let plan_for_task = plan.clone();
                let adapter_for_task = adapter_label.to_owned();
                let project_path_for_task = project_path.to_owned();
                let fact = crate::load_progress::spawn_blocking(move || {
                    write_image_asset(
                        &plan_for_task,
                        index,
                        seed,
                        width,
                        height,
                        pixels,
                        &adapter_for_task,
                        image_raw_settings,
                        &project_path_for_task,
                    )
                })
                .await
                .map_err(|error| crate::task_join_error("image asset write task", error))??;
                asset_writes.push(Value::Object(fact));
                emit_event(
                    "image_inference_complete",
                    json!({
                        "jobId": job.id,
                        "imageIndex": index,
                        "backend": backend,
                    }),
                );
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated image {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                // sc-9618: honor a process shutdown on every tick (a local flag read, no API cost, so
                // not throttled by the 2s user-cancel poll) so a quit trips the engine cancel promptly.
                if !canceled && (shutdown_requested()
                    || (last_cancel_check.elapsed() >= Duration::from_secs(2) && {
                        last_cancel_check = Instant::now();
                        cancel_requested_peek(api, &job.id).await
                    }))
                {
                    begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                    canceled = true;
                }
            }
        }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly — reclaim the handle (disarming the drop-guard) and join the finished task.
    let task_result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("generation task join", error))?;
    if canceled {
        // The generation has now actually stopped, so post the TERMINAL Canceled here
        // (not at the earlier cancel poll, which only tripped the flag + showed
        // "Cancelling…"). This terminal write is what frees the worker row
        // (`jobs_store::update_job_progress`), so it lands exactly as the worker process
        // returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free, and the UI shows "Cancelling…" until completion (sc-5515).
        // The explicit streaming result keeps the partial images already streamed (it IS
        // that fact set; `persist_reported_assets` re-injects the derived asset keys) while
        // dropping any in-flight `previewFrame` from the stored result — a canceled job must
        // not carry a stale interim frame (sc-16904). Before previews this posted result=None
        // for the same partial-image outcome via `coalesce`.
        let message = "Image generation canceled by user.";
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    if !prompt_enhancement_reports.is_empty() {
        return Err(WorkerError::Engine(
            "provider emitted a prompt-enhancement report without a completed image".to_owned(),
        ));
    }
    // Post the effective-settings + per-phase timing block (epic 10402,
    // sc-10405/sc-10406). Best-effort; coalesce-merges with the S2 hardware block
    // (which owns totalMs/backend/peaks) server-side.
    // The RESOLVED tier dir, so the recorded quant label describes the tier that actually RAN rather
    // than the one requested (sc-11042). Re-resolving here is the same pure path-join + `is_dir` probe
    // the lane already did before loading (no fetch, no I/O beyond a stat), and it runs once per job.
    // `.ok().flatten()` because a metrics block must never fail a completed generation: an unresolvable
    // dir yields `None`, which conservatively reports the request-derived q4/q8/bf16 label exactly as
    // it did before this parameter existed.
    let tier_dir = resolve_completed_image_metrics_tier_dir(&plan.request, settings);
    let mut metrics = build_image_metrics(
        &plan.request,
        effective_steps,
        total as u32,
        tier_dir.as_deref(),
    );
    if let Some(phase) = phase_timer.into_metrics(Instant::now()) {
        metrics.load_ms = phase.load_ms;
        metrics.sample_ms = phase.sample_ms;
        metrics.decode_ms = phase.decode_ms;
    }
    crate::job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
    task_result
}
