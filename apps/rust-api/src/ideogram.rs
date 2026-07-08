//! Headless/API-path rich auto-captioning for Ideogram 4 (epic 4725, sc-6519).
//!
//! Ideogram 4 is trained EXCLUSIVELY on structured JSON captions; a raw plain-text prompt is
//! out-of-distribution and stochastically renders the "Image blocked by safety filter" placeholder
//! (sc-6307, reference-confirmed faithful — NOT a porting bug). The web Image Studio avoids this by
//! auto-expanding plain text into a RICH caption via the magic-prompt utility model (Llama-3.2-3B)
//! BEFORE it ever submits the image job (sc-6501). A direct/headless caller that POSTs plain text to
//! `ideogram_4` bypasses that expansion and gets only the worker's FORMAT guard — and the real-weight
//! finding is that a SPARSE caption does NOT escape the placeholder (content RICHNESS is the lever, not
//! JSON structure), so such a job relies purely on the worker's reseed recovery (~80% within the
//! default retries) and can still surface a placeholder.
//!
//! This closes the gap with full UI parity: when an `ideogram_4`/`ideogram_4_turbo` image job arrives
//! with a non-caption prompt, the API runs the SAME `magic_prompt` expansion job the web runs (a
//! separate `prompt_refine` job — so the 3B refiner and the ~50GB Ideogram weights are never
//! co-resident, exactly as the UI achieves it) and rewrites the prompt to the rich caption BEFORE the
//! image job is created and dispatched. If the expansion is unavailable (no refiner staged, the job
//! fails, or it times out), the original prompt is left untouched and the image job still dispatches —
//! the worker's format-guard + reseed net is the fallback, so a render is always produced.
//!
//! FULLY ASYNC (sc-9120): the image-job `POST` no longer waits on the expansion at all. When a
//! plain-text Ideogram 4 job arrives, the image job is created IMMEDIATELY in a non-claimable
//! `pending_caption` status (the `POST` returns 201 in ~one DB insert), and a background tokio task
//! runs the magic-prompt expansion and then promotes the job — rewriting `payload.prompt` to the rich
//! caption and flipping it to `queued`, or degrading it to `queued` with the ORIGINAL prompt when the
//! expansion is unavailable/times out. The worker only ever claims the job once it is `queued`. This
//! supersedes the sc-8818 bounded-inline fix (which still held the `POST` up to ~45s): the caller is
//! never blocked, an impatient re-POST reuses an in-flight refine job instead of stacking a new one,
//! and an API restart mid-expansion recovers a stranded `pending_caption` row to `queued` (see
//! `jobs_store::mark_interrupted_on_startup`) so it can never sit un-claimable forever.

use super::*;

/// How often the background watcher polls the in-flight `magic_prompt` job while awaiting its caption.
/// Generation expansion runs in tens of seconds, so a sub-second poll adds negligible latency and the
/// poll runs off the request path (a background tokio task), so it never holds an HTTP connection.
const CAPTION_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Upper bound on the background watcher's wait for the caption before degrading to the original
/// prompt. Because the watcher runs OFF the request path (sc-9120), this ceiling no longer bounds any
/// HTTP latency — it only bounds how long a stuck/backlogged worker delays the job's promotion before
/// the watcher gives up and promotes it to `queued` with the plain prompt (the worker's format-guard +
/// reseed net is the fallback). It is set generously so a healthy-but-busy worker still gets to deliver
/// the rich caption; a genuinely stuck refiner degrades rather than stranding the job.
const CAPTION_MAX_WAIT: Duration = Duration::from_secs(180);
/// How many magic-prompt jobs one caption watcher may enqueue. Re-sampling a completed-but-invalid
/// caption is a quality nicety the small 3B model occasionally needs; because the watcher is now async
/// (no HTTP connection is held), a bounded re-sample no longer risks hanging a `POST`. Kept small so a
/// persistently-malformed refiner degrades promptly instead of looping.
const MAX_CAPTION_ATTEMPTS: u32 = 2;

/// Both Ideogram 4 image models are JSON-caption-trained, so both get the auto-caption (mirrors the
/// worker's `ideogram_caption::is_ideogram_model`).
pub(crate) fn is_ideogram_caption_model(model: &str) -> bool {
    matches!(model, "ideogram_4" | "ideogram_4_turbo")
}

/// True when `prompt` is already a structured JSON caption — a JSON object carrying the
/// `compositional_deconstruction` section the model's `CaptionVerifier` requires. Only checks for the
/// required section (not the full schema) so an already-structured prompt (the normal web path) is
/// never needlessly re-expanded.
fn prompt_is_caption(prompt: &str) -> bool {
    prompt.trim_start().starts_with('{')
        && serde_json::from_str::<Value>(prompt)
            .ok()
            .as_ref()
            .is_some_and(sceneworks_core::ideogram_caption::is_caption)
}

/// True when the job conditions on an input image (an `edit_image` mode or a source asset) — the
/// prompt there is an edit instruction, not a full scene to caption, so the magic-prompt expansion
/// (which writes a fresh scene caption) must not touch it. Ideogram 4's edit path keeps the worker's
/// format-guard + reseed net instead.
fn is_image_conditioned(job_payload: &JsonObject) -> bool {
    let non_empty_str = |key: &str| {
        job_payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    job_payload.get("mode").and_then(Value::as_str) == Some("edit_image")
        || non_empty_str("sourceAssetId")
}

/// Reduce a pixel `width`×`height` to an aspect-ratio label `"W:H"` for the magic-prompt expander
/// (which uses it to steer layout/bbox choices), mirroring the web's gcd reduction.
fn aspect_ratio_label(width: u32, height: u32) -> String {
    let (w, h) = (width.max(1), height.max(1));
    let divisor = gcd(w, h);
    format!("{}:{}", w / divisor, h / divisor)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// Extract + clean the expanded caption from a completed `magic_prompt` job's result. The worker
/// already isolates the JSON object into `result.refinedPrompt` (via `clean_json_output`); here it is
/// re-emitted through the shared canonical serializer (matching the web's `parseMagicPromptCaption` +
/// `serializeCaption`: drop the stray top-level `aspect_ratio`, strip the model's unreliable bboxes,
/// impose canonical key order). Returns `None` when the reply is not a valid structured caption (the
/// 3B occasionally emits malformed JSON or prose) so the caller re-samples or degrades.
fn caption_from_refine_result(result: &JsonObject) -> Option<String> {
    let refined = result.get("refinedPrompt").and_then(Value::as_str)?.trim();
    let parsed: Value = serde_json::from_str(refined).ok()?;
    sceneworks_core::ideogram_caption::serialize_magic_prompt_caption(&parsed)
}

/// A detected Ideogram 4 auto-caption request: the plain prompt to expand and the reduced aspect-ratio
/// label the expander uses to steer layout. Produced by [`caption_request_for_ideogram`] when a job
/// needs the async caption; consumed by [`spawn_ideogram_caption_watcher`].
pub(crate) struct CaptionRequest {
    pub model: String,
    pub prompt: String,
    pub aspect_ratio: String,
}

/// Decide whether `job_payload` is a plain-text Ideogram 4 text-to-image job that should have its
/// prompt expanded into a rich JSON caption before dispatch (sc-6519). Returns the caption request
/// when so, or `None` for every other model, for an already-structured caption, for an
/// image-conditioned edit (the prompt there is an edit instruction, not a scene to caption), and for an
/// empty prompt. Pure and synchronous: the caller uses the `Some`/`None` verdict to decide whether to
/// create the image job in `pending_caption` (then run [`spawn_ideogram_caption_watcher`]) or in the
/// default `queued`. The payload is NOT mutated here — the async watcher rewrites it later (sc-9120).
pub(crate) fn caption_request_for_ideogram(job_payload: &JsonObject) -> Option<CaptionRequest> {
    let model = job_payload.get("model").and_then(Value::as_str)?;
    if !is_ideogram_caption_model(model) || is_image_conditioned(job_payload) {
        return None;
    }
    let prompt = job_payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    if prompt.is_empty() || prompt_is_caption(&prompt) {
        return None;
    }
    let aspect_ratio = aspect_ratio_label(
        payload_dimension(job_payload, "width"),
        payload_dimension(job_payload, "height"),
    );
    Some(CaptionRequest {
        model: model.to_owned(),
        prompt,
        aspect_ratio,
    })
}

/// Spawn the first-and-only rust-api background job watcher (sc-9120): a detached tokio task that runs
/// the magic-prompt expansion for a `pending_caption` image job and then PROMOTES the job to `queued`.
///
/// On success it rewrites `payload.prompt` to the rich caption; on failure/timeout it degrades the job
/// to `queued` with the ORIGINAL prompt (the worker's format-guard + reseed net remains the fallback).
/// The promotion is a race-free guarded UPDATE (`... where status = 'pending_caption'`), so if the user
/// canceled the job while the caption was running, the promotion is a no-op and the job stays canceled.
/// Either terminal outcome ALWAYS leaves the job claimable — the watcher never returns leaving the row
/// in `pending_caption` — and re-broadcasts `job.updated`/`queue.updated` so the UI reflects the flip.
pub(crate) fn spawn_ideogram_caption_watcher(
    state: AppState,
    job_id: String,
    request: CaptionRequest,
) {
    tokio::spawn(async move {
        let caption = expand_to_caption(&state, &request.prompt, &request.aspect_ratio).await;
        // Rewrite the stored payload's prompt to the rich caption on success; degrade to the original
        // prompt (new_payload = None) when the expansion is unavailable/times out.
        let new_payload = match &caption {
            Some(caption) => {
                let read_id = job_id.clone();
                match store_call(state.clone(), move |store, _timeout| {
                    store.get_job(&read_id)
                })
                .await
                {
                    Ok(job) => {
                        let mut payload = job.payload;
                        payload.insert("prompt".to_owned(), Value::String(caption.clone()));
                        Some(payload)
                    }
                    // The row vanished (never expected for a just-created job) — nothing to promote.
                    Err(error) => {
                        tracing::warn!(
                            event = "ideogram_auto_caption_read_failed",
                            job = %job_id,
                            error = %error.detail,
                            "could not read the pending_caption job to apply its caption"
                        );
                        return;
                    }
                }
            }
            None => {
                tracing::warn!(
                    event = "ideogram_auto_caption_unavailable",
                    job = %job_id,
                    model = %request.model,
                    "magic-prompt expansion unavailable; dispatching Ideogram 4 job with the original prompt"
                );
                None
            }
        };
        let promoted = {
            let job_id = job_id.clone();
            store_call(state.clone(), move |store, _timeout| {
                store.promote_pending_caption_job(&job_id, new_payload)
            })
            .await
        };
        match promoted {
            Ok(promotion) => {
                // Only broadcast when the job actually changed. A `promoted = false` means the job left
                // `pending_caption` before us (canceled, or recovered on a restart), so its owner
                // already emitted the relevant event — re-broadcasting a stale snapshot would be noise.
                if promotion.promoted {
                    tracing::info!(
                        event = "ideogram_auto_caption_promoted",
                        job = %job_id,
                        captioned = caption.is_some(),
                        "promoted the pending_caption Ideogram 4 job to queued"
                    );
                    publish(&state, "job.updated", &promotion.job);
                    if let Err(error) = publish_queue(&state).await {
                        tracing::warn!(
                            event = "ideogram_auto_caption_queue_publish_failed",
                            job = %job_id,
                            error = %error.detail,
                            "promoted the job but could not refresh the queue summary"
                        );
                    }
                } else {
                    tracing::info!(
                        event = "ideogram_auto_caption_promotion_skipped",
                        job = %job_id,
                        status = %promotion.job.status.as_str(),
                        "pending_caption job already left the stage (canceled/recovered); skipping promotion"
                    );
                }
            }
            Err(error) => {
                // The promotion write itself failed (e.g. the DB was momentarily locked). The job is
                // still `pending_caption` and NOT claimable; the startup recovery
                // (mark_interrupted_on_startup) is the backstop that flips it to `queued` on the next
                // API restart, but log loudly so the stuck row is visible in the meantime.
                tracing::error!(
                    event = "ideogram_auto_caption_promote_failed",
                    job = %job_id,
                    error = %error.detail,
                    "failed to promote the pending_caption job; it will be recovered to queued on the next backend restart"
                );
            }
        }
    });
}

/// Read a pixel dimension from the image-job payload, defaulting to a 1024 square (the Ideogram 4
/// default) when absent or malformed.
fn payload_dimension(job_payload: &JsonObject, key: &str) -> u32 {
    job_payload
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(1024)
}

/// The terminal outcome of one magic-prompt expansion attempt.
enum CaptionOutcome {
    /// The job completed and produced a valid structured caption.
    Caption(String),
    /// The job completed but produced no usable caption — the small 3B model is stochastic and
    /// occasionally emits malformed/incomplete JSON or prose (real-weight observed, sc-6519). Worth
    /// re-sampling with a fresh job.
    Resample,
    /// The job failed/canceled or didn't complete in time — an infrastructure problem (e.g. the
    /// refiner isn't staged); retrying won't help, so degrade to the plain prompt.
    Unavailable,
}

/// Run the magic-prompt expansion, re-sampling a completed-but-invalid caption up to
/// [`MAX_CAPTION_ATTEMPTS`] times. Runs off the request path (in the background watcher, sc-9120), so
/// the re-sample budget no longer bounds any HTTP latency. Returns `None` (degrade to the plain prompt)
/// when the job can't be enqueued, when the model produces no valid caption within the attempt budget,
/// or on an infrastructure failure/timeout.
///
/// The FIRST attempt reuses an in-flight refine job for the same prompt+aspect (a concurrent/retried
/// caption shares one expansion, sc-9120); a re-sample always enqueues a fresh job so it actually
/// re-samples the stochastic 3B rather than re-reading the same bad result.
async fn expand_to_caption(state: &AppState, prompt: &str, aspect_ratio: &str) -> Option<String> {
    for attempt in 1..=MAX_CAPTION_ATTEMPTS {
        let reuse = attempt == 1;
        let job = match obtain_magic_prompt_job(state, prompt, aspect_ratio, reuse).await {
            Ok(job) => job,
            Err(error) => {
                tracing::warn!(
                    event = "ideogram_auto_caption_enqueue_failed",
                    error = %error.detail,
                    "could not enqueue the Ideogram magic-prompt expansion job"
                );
                return None;
            }
        };
        match await_magic_prompt_outcome(state, &job.id).await {
            CaptionOutcome::Caption(caption) => return Some(caption),
            CaptionOutcome::Unavailable => return None,
            CaptionOutcome::Resample => {
                tracing::warn!(
                    event = "ideogram_auto_caption_resample",
                    attempt,
                    max = MAX_CAPTION_ATTEMPTS,
                    "magic-prompt produced no valid caption; re-sampling"
                );
            }
        }
    }
    None
}

/// Obtain the `magic_prompt` `prompt_refine` job to await. When `reuse` is set, an already-in-flight
/// refine job for the same prompt+aspect is reused (sc-9120: an impatient client re-POSTing the same
/// image job shares one expansion instead of stacking a fresh refine job every time); otherwise, or
/// when nothing is in flight, a new one is enqueued.
async fn obtain_magic_prompt_job(
    state: &AppState,
    prompt: &str,
    aspect_ratio: &str,
    reuse: bool,
) -> Result<JobSnapshot, ApiError> {
    if reuse {
        let existing = {
            let (prompt, aspect_ratio) = (prompt.to_owned(), aspect_ratio.to_owned());
            store_call(state.clone(), move |store, _timeout| {
                store.find_reusable_prompt_refine_job(&prompt, &aspect_ratio)
            })
            .await?
        };
        if let Some(job) = existing {
            tracing::info!(
                event = "ideogram_auto_caption_reused_refine",
                job = %job.id,
                "reusing an in-flight magic-prompt job for the Ideogram auto-caption"
            );
            return Ok(job);
        }
    }
    enqueue_magic_prompt_job(state, prompt, aspect_ratio).await
}

/// Create the `magic_prompt` `prompt_refine` job. Mirrors `create_prompt_refine_job`'s magic-prompt
/// payload (`task: "magic_prompt"` + the aspect ratio steers layout/bbox decisions) and, like the web
/// refine job, is created without a project so it does not clutter a project's job list.
async fn enqueue_magic_prompt_job(
    state: &AppState,
    prompt: &str,
    aspect_ratio: &str,
) -> Result<JobSnapshot, ApiError> {
    let mut payload = JsonObject::new();
    payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));
    payload.insert("workflow".to_owned(), Value::String("image".to_owned()));
    payload.insert("task".to_owned(), Value::String("magic_prompt".to_owned()));
    payload.insert(
        "aspectRatio".to_owned(),
        Value::String(aspect_ratio.to_owned()),
    );
    create_generation_job(
        state.clone(),
        JobType::PromptRefine,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await
}

/// Poll one magic-prompt job until it reaches a terminal state (or the wait cap elapses). A completed
/// job yields its caption ([`CaptionOutcome::Caption`]) or, if it produced no valid caption,
/// [`CaptionOutcome::Resample`]; a failed/canceled/timed-out job yields [`CaptionOutcome::Unavailable`].
async fn await_magic_prompt_outcome(state: &AppState, job_id: &str) -> CaptionOutcome {
    let started = Instant::now();
    loop {
        let Ok(job) = store_call(state.clone(), {
            let job_id = job_id.to_owned();
            move |store, _timeout| store.get_job(&job_id)
        })
        .await
        else {
            return CaptionOutcome::Unavailable;
        };
        match job.status {
            JobStatus::Completed => {
                return match caption_from_refine_result(&job.result) {
                    Some(caption) => CaptionOutcome::Caption(caption),
                    None => CaptionOutcome::Resample,
                };
            }
            JobStatus::Failed | JobStatus::Canceled | JobStatus::Interrupted => {
                return CaptionOutcome::Unavailable;
            }
            _ => {}
        }
        if started.elapsed() >= CAPTION_MAX_WAIT {
            return CaptionOutcome::Unavailable;
        }
        tokio::time::sleep(CAPTION_POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_expansion_budget_is_bounded_and_polls_sanely() {
        // sc-9120: the expansion now runs in a BACKGROUND task, so its wall-time no longer bounds any
        // HTTP latency — but it must still terminate. Guard that the re-sample budget stays small (a
        // persistently-malformed refiner degrades promptly rather than looping) and that the poll
        // cadence is strictly shorter than the per-attempt ceiling, or the loop could exit on its first
        // tick without ever giving a healthy worker a chance to deliver the caption.
        assert!(
            (1..=3).contains(&MAX_CAPTION_ATTEMPTS),
            "the background caption must run a small, bounded number of attempts (was {MAX_CAPTION_ATTEMPTS})"
        );
        assert!(CAPTION_POLL_INTERVAL < CAPTION_MAX_WAIT);
    }

    #[test]
    fn caption_request_detects_plain_text_ideogram_jobs() {
        // A plain-text Ideogram 4 t2i job is a caption candidate: the request carries the trimmed
        // prompt and the reduced aspect-ratio label the expander needs.
        let mut payload = JsonObject::new();
        payload.insert("model".to_owned(), Value::String("ideogram_4".to_owned()));
        payload.insert(
            "prompt".to_owned(),
            Value::String("  a fox on a beach  ".to_owned()),
        );
        payload.insert("width".to_owned(), Value::from(1920));
        payload.insert("height".to_owned(), Value::from(1080));
        let request = caption_request_for_ideogram(&payload).expect("should need a caption");
        assert_eq!(request.model, "ideogram_4");
        assert_eq!(request.prompt, "a fox on a beach");
        assert_eq!(request.aspect_ratio, "16:9");

        // Turbo variant is a candidate too.
        payload.insert(
            "model".to_owned(),
            Value::String("ideogram_4_turbo".to_owned()),
        );
        assert!(caption_request_for_ideogram(&payload).is_some());
    }

    #[test]
    fn caption_request_is_none_for_non_candidates() {
        let candidate = |model: &str, prompt: &str, extra: &[(&str, Value)]| {
            let mut payload = JsonObject::new();
            payload.insert("model".to_owned(), Value::String(model.to_owned()));
            payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));
            for (key, value) in extra {
                payload.insert((*key).to_owned(), value.clone());
            }
            caption_request_for_ideogram(&payload).is_some()
        };
        // Non-Ideogram model: never captioned.
        assert!(!candidate("flux_dev", "a fox on a beach", &[]));
        // Already a structured caption: not re-expanded.
        assert!(!candidate(
            "ideogram_4",
            r#"{"compositional_deconstruction": {"background": "a beach", "elements": []}}"#,
            &[]
        ));
        // Empty prompt: nothing to expand.
        assert!(!candidate("ideogram_4", "   ", &[]));
        // Image-conditioned (edit / img2img): the prompt is an edit instruction, not a scene.
        assert!(!candidate(
            "ideogram_4",
            "make it night",
            &[("mode", Value::String("edit_image".to_owned()))]
        ));
        assert!(!candidate(
            "ideogram_4",
            "make it night",
            &[("sourceAssetId", Value::String("asset-1".to_owned()))]
        ));
    }

    #[test]
    fn ideogram_caption_models_match_both_variants() {
        assert!(is_ideogram_caption_model("ideogram_4"));
        assert!(is_ideogram_caption_model("ideogram_4_turbo"));
        assert!(!is_ideogram_caption_model("flux_dev"));
        assert!(!is_ideogram_caption_model(""));
    }

    #[test]
    fn prompt_is_caption_detects_structured_captions_only() {
        let caption = r#"{"high_level_description": "a red fox", "compositional_deconstruction": {"background": "snowy forest", "elements": []}}"#;
        assert!(prompt_is_caption(caption));
        // The required section alone is enough (the web builder can omit high_level_description).
        assert!(prompt_is_caption(
            r#"{"compositional_deconstruction": {"background": "a beach", "elements": []}}"#
        ));
        // Plain text and a JSON object missing the required section are NOT captions.
        assert!(!prompt_is_caption("a fox on a beach"));
        assert!(!prompt_is_caption(r#"{"foo": "bar"}"#));
        // compositional_deconstruction must be an object, not a scalar.
        assert!(!prompt_is_caption(
            r#"{"compositional_deconstruction": "nope"}"#
        ));
    }

    #[test]
    fn image_conditioned_jobs_are_detected() {
        let edit: JsonObject = serde_json::from_str(r#"{"mode": "edit_image"}"#).unwrap();
        assert!(is_image_conditioned(&edit));
        let img2img: JsonObject = serde_json::from_str(r#"{"sourceAssetId": "asset-1"}"#).unwrap();
        assert!(is_image_conditioned(&img2img));
        // A pure text-to-image job (no source) is not image-conditioned.
        let txt2img: JsonObject =
            serde_json::from_str(r#"{"mode": "text_to_image", "sourceAssetId": ""}"#).unwrap();
        assert!(!is_image_conditioned(&txt2img));
        assert!(!is_image_conditioned(&JsonObject::new()));
    }

    #[test]
    fn aspect_ratio_label_reduces_dimensions() {
        assert_eq!(aspect_ratio_label(1024, 1024), "1:1");
        assert_eq!(aspect_ratio_label(1920, 1080), "16:9");
        assert_eq!(aspect_ratio_label(1080, 1920), "9:16");
        assert_eq!(aspect_ratio_label(1200, 800), "3:2");
        // Degenerate inputs never divide by zero.
        assert_eq!(aspect_ratio_label(0, 0), "1:1");
    }

    #[test]
    fn caption_from_refine_result_requires_a_valid_caption() {
        let mut good = JsonObject::new();
        good.insert(
            "refinedPrompt".to_owned(),
            Value::String(
                r#"{"compositional_deconstruction": {"background": "a beach", "elements": []}}"#
                    .to_owned(),
            ),
        );
        assert!(caption_from_refine_result(&good).is_some());

        // A non-caption reply (small model returned prose, or an empty/missing key) → degrade.
        let mut prose = JsonObject::new();
        prose.insert(
            "refinedPrompt".to_owned(),
            Value::String("a fox on a beach".to_owned()),
        );
        assert!(caption_from_refine_result(&prose).is_none());
        assert!(caption_from_refine_result(&JsonObject::new()).is_none());
    }
}
