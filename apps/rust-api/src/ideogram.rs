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
//! LATENCY BOUND (sc-8818): the expansion runs INLINE in the image-job `POST`, so its wait is bounded
//! to a SINGLE attempt with a ~45s ceiling ([`CAPTION_MAX_WAIT`]/[`MAX_CAPTION_ATTEMPTS`]). It was
//! previously 3 attempts × 180s, which could hang the `POST` ~9 minutes on a backlogged worker — past
//! any client/proxy timeout, whereupon impatient retries stacked more refine jobs. A tighter single
//! bounded attempt returns the `POST` promptly and caps how many refine jobs one request can enqueue.
//! The fully-async alternative (create the job in a `pending_caption` stage, rewrite the payload from a
//! background task before dispatch, so the `POST` never waits) is tracked as a follow-up to sc-8818.

use super::*;

/// How often the API polls the in-flight `magic_prompt` job while awaiting its caption. Generation
/// expansion runs in tens of seconds, so a sub-second poll adds negligible latency.
const CAPTION_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Upper bound on the wait for the caption before falling back to the original prompt.
///
/// This is a HARD ceiling on how long the image-job `POST` is held while the auto-caption runs. It was
/// previously 180s *per attempt* × 3 attempts, so a stuck/backlogged worker could hang the HTTP request
/// for ~9 minutes before the image job was even created — long past any sane client/proxy timeout, at
/// which point impatient retries stacked more refine jobs (sc-8818). The magic-prompt expansion is a
/// tens-of-seconds job, so a single bounded attempt with a ~45s ceiling gives a healthy worker ample
/// room to deliver the rich caption while guaranteeing the `POST` returns promptly; a stuck/absent
/// worker degrades to the plain prompt (the worker's format-guard + reseed net is the fallback).
///
/// The fully-async alternative — create the image job immediately in a `pending_caption` stage and let
/// a background task rewrite the payload before dispatch, so the `POST` never waits at all — is tracked
/// separately (relates to sc-8818); it is a cross-cutting store/worker/contract change out of scope for
/// this bounded-latency fix.
const CAPTION_MAX_WAIT: Duration = Duration::from_secs(45);
/// How many magic-prompt jobs a single image `POST` may enqueue while awaiting a caption. Capped at a
/// SINGLE attempt (sc-8818): re-sampling a completed-but-invalid caption is a quality nicety, but each
/// extra attempt enqueues a fresh `prompt_refine` job and extends the wall-time the HTTP request is
/// held — the exact behavior that let a hung `POST` stack refine jobs. One attempt bounds both the
/// latency and the number of refine jobs a client (or its impatient retries) can pile up; a
/// completed-but-invalid caption degrades to the plain prompt and the worker's reseed net recovers it.
const MAX_CAPTION_ATTEMPTS: u32 = 1;

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

/// If `job_payload` targets an Ideogram 4 model with a plain text-to-image prompt, run the magic-prompt
/// expansion (the same separate `prompt_refine` job the web runs) and rewrite `job_payload["prompt"]`
/// to the rich caption before the image job is created. A no-op for every other model, for an
/// already-structured caption, for an image-conditioned edit (the prompt there is an edit instruction,
/// not a scene to caption), and (gracefully) when the expansion is unavailable — in which case the
/// original prompt is left for the worker's format-guard + reseed net.
pub(crate) async fn rich_auto_caption_for_ideogram(state: &AppState, job_payload: &mut JsonObject) {
    let model = job_payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !is_ideogram_caption_model(model) || is_image_conditioned(job_payload) {
        return;
    }
    let model = model.to_owned();
    let prompt = job_payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    if prompt.is_empty() || prompt_is_caption(&prompt) {
        return;
    }
    let aspect_ratio = aspect_ratio_label(
        payload_dimension(job_payload, "width"),
        payload_dimension(job_payload, "height"),
    );

    match expand_to_caption(state, &prompt, &aspect_ratio).await {
        Some(caption) => {
            job_payload.insert("prompt".to_owned(), Value::String(caption));
        }
        None => {
            // Degrade gracefully: leave the original prompt so the image job still dispatches. The
            // worker's format-guard wrap + placeholder detect-and-recover reseed (sc-6501) remains the
            // safety net, exactly as before this auto-caption existed.
            tracing::warn!(
                event = "ideogram_auto_caption_unavailable",
                model = %model,
                "magic-prompt expansion unavailable; dispatching Ideogram 4 job with the original prompt"
            );
        }
    }
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
/// [`MAX_CAPTION_ATTEMPTS`] times (currently a SINGLE bounded attempt — see [`CAPTION_MAX_WAIT`] — so
/// the inline `POST` can't hang on repeated re-samples, sc-8818). Returns `None` (degrade to the plain
/// prompt) when the job can't be enqueued, when the model produces no valid caption within the attempt
/// budget, or on an infrastructure failure/timeout. The web surfaces an expansion failure for the user
/// to retry; the headless path has no human in the loop, so any re-sample budget lives here.
async fn expand_to_caption(state: &AppState, prompt: &str, aspect_ratio: &str) -> Option<String> {
    for attempt in 1..=MAX_CAPTION_ATTEMPTS {
        let job = match enqueue_magic_prompt_job(state, prompt, aspect_ratio).await {
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
    fn inline_caption_wait_is_bounded_to_a_single_short_attempt() {
        // sc-8818: the auto-caption runs INLINE in the image-job POST, so its worst-case wall-time is
        // MAX_CAPTION_ATTEMPTS × CAPTION_MAX_WAIT. The regression was 3 × 180s = ~9 minutes, which hung
        // the POST past client/proxy timeouts and let impatient retries stack refine jobs. Guard that
        // the request can no longer be held for minutes: a single attempt, ceilinged well under a
        // minute. (A tighter contract than a plain "< 9 min" so the bound can't silently regress.)
        assert_eq!(
            MAX_CAPTION_ATTEMPTS, 1,
            "the inline caption must run at most one attempt so the POST can't stack refine jobs"
        );
        let worst_case = CAPTION_MAX_WAIT * MAX_CAPTION_ATTEMPTS;
        assert!(
            worst_case <= Duration::from_secs(60),
            "the inline caption wait must stay under a minute (was {worst_case:?})"
        );
        // The poll cadence must be strictly shorter than the ceiling, or the loop could exit on its
        // first tick without ever giving a healthy worker a chance to deliver the caption.
        assert!(CAPTION_POLL_INTERVAL < CAPTION_MAX_WAIT);
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
