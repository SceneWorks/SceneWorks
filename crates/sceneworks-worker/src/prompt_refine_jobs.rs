//! Native prompt refinement (epic 5095): candle on Windows/CUDA (sc-5525) + MLX on macOS (sc-5552).
//!
//! Routes the `prompt_refine` job to a native `TextLlm` provider (Llama-3.2-3B-Instruct) through the
//! backend-neutral `gen_core::load_textllm` seam (the sc-5500 contract): the candle provider
//! (`backend="candle"`, `candle-gen-prompt-refine`) on the Windows candle build, and the MLX twin
//! (`backend="mlx"`, `mlx-gen-prompt-refine`) on macOS. The Python torch `PromptRefiner`
//! (`apps/worker/scene_worker/prompt_refine.py`) stays the fallback only on platforms with neither
//! native provider (e.g. the candle-less Desktop installer); its physical deletion waits on the candle
//! provider being the default everywhere off-Mac (see sc-5525).
//!
//! The `TextLlm` contract is generic (`system` + `prompt` + sampling → text), so the
//! prompt-refinement PRODUCT logic that lived in `prompt_refine.py` moves here caller-side: the
//! rewrite rules + image/video medium switch + guide assembly (`build_refine_system_prompt`, into the
//! request `system`) and the reasoning-block / code-fence / surrounding-quote cleanup
//! (`clean_refine_output`, over the model reply). Sampling matches the Python path (temperature 0.7,
//! top_p 0.9, max_new_tokens 512), as does the empty-output → error behavior and the `{originalPrompt,
//! refinedPrompt}` result shape.

use super::*;

// Prompt-refine provider force-link anchors: keep each backend's `inventory::submit!` `TextLlm`
// registration (id `prompt_refine`) from being dropped by the release linker. sc-5552 adds the native
// MLX twin (`mlx_gen_prompt_refine`, backend `mlx`) alongside sc-5525's candle anchor; mirrors the
// dual JoyCaption anchors in caption_jobs.rs.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_prompt_refine as _;
#[cfg(target_os = "macos")]
use mlx_gen_prompt_refine as _;

// The registry id both providers register under (`prompt::PROMPT_REFINE_ID`); kept as a local literal
// so the shared dispatch names no backend-specific symbol. `gen_core::load_textllm` resolves the MLX
// twin on macOS and the candle provider on the Windows candle build.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PROMPT_REFINE_ENGINE_ID: &str = "prompt_refine";
// Default refinement checkpoint — the small abliterated Llama-3.2-3B instruction model, parity with
// the Python `DEFAULT_REFINE_MODEL`. Overridable per-job via `payload.model`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const DEFAULT_REFINE_MODEL: &str = "huihui-ai/Llama-3.2-3B-Instruct-abliterated";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Prompt refinement canceled by user.";
// Architecture-pill label for the streamed progress (mirrors the candle image/video paths): the MLX
// twin on macOS, candle on the Windows candle build.
#[cfg(target_os = "macos")]
const REFINE_BACKEND: &str = "mlx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const REFINE_BACKEND: &str = "candle";

// ----------------------------------------------------------------------------------------------
// Product logic (pure, platform-independent) — ported from `prompt_refine.py` so the native worker
// (candle + MLX) owns the prompt assembly + reply cleanup the generic `TextLlm` contract does not.
// Compiled in the default `cargo test` gate (so the unit tests below run on every lane) and on the
// macOS + candle builds.
// ----------------------------------------------------------------------------------------------

/// The base rewrite rules with the `{medium}` placeholders filled (`image` / `video`). Verbatim port
/// of the Python `_BASE_RULES.format(medium=…)`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn base_rules(medium: &str) -> String {
    [
        format!("You are a prompt rewriter for a generative {medium} model."),
        format!(
            "Rewrite the user's input into a single, precise {medium} prompt that follows the \
             model's prompt guide below."
        ),
        String::new(),
        "Rules:".to_owned(),
        "- Output exactly one rewritten prompt and nothing else — no explanations, reasoning, \
         commentary, options, or labels."
            .to_owned(),
        format!(
            "- Preserve the user's intent: do not change the subjects, attributes, actions, \
             relationships, or core setting they described. You may add concrete details only when \
             they make the {medium} more coherent and stay consistent with the user's meaning."
        ),
        "- If the user's prompt is already detailed and on-guide, make only minimal edits for \
         fluency."
            .to_owned(),
        "- Follow the guide's recommended structure, phrasing, and what-to-avoid guidance."
            .to_owned(),
        "- Match the user's language: if their prompt is not in English, respond in the same \
         language."
            .to_owned(),
        "- Do not wrap the output in quotes, markdown, JSON, or code fences unless those are part \
         of the described scene."
            .to_owned(),
    ]
    .join("\n")
}

/// Build the `system` message for the refiner: the rewrite rules (medium chosen from the workflow)
/// plus the model's prompt guide when one is supplied. Port of the Python `build_system_prompt`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_refine_system_prompt(guide: Option<&str>, workflow: Option<&str>) -> String {
    let medium = if workflow
        .map(|w| w.trim().eq_ignore_ascii_case("video"))
        .unwrap_or(false)
    {
        "video"
    } else {
        "image"
    };
    let rules = base_rules(medium);
    let guide = guide.unwrap_or("").trim();
    if guide.is_empty() {
        rules
    } else {
        format!("{rules}\n\n# Model prompt guide\n\n{guide}")
    }
}

/// Strip `<think>…</think>` reasoning blocks, a wrapping code fence, and matching surrounding quotes
/// from the model reply. Port of the Python `clean_output` (regex-free: the tags are ASCII, matched
/// case-insensitively without lowercasing the whole — Unicode-safe — string).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn clean_refine_output(text: &str) -> String {
    let mut text = strip_think_blocks(text.trim()).trim().to_owned();
    // An orphan closing tag (no matching open): keep only what follows the last one.
    if let Some(pos) = last_ci(&text, "</think>") {
        text = text[pos + "</think>".len()..].trim().to_owned();
    }
    // A wrapping ```…``` code fence: drop the fence lines.
    if text.starts_with("```") && text.ends_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 2 {
            text = lines[1..lines.len() - 1].join("\n").trim().to_owned();
        }
    }
    // Matching surrounding single/double quotes.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 2 {
        let (first, last) = (chars[0], chars[chars.len() - 1]);
        if first == last && (first == '"' || first == '\'') {
            text = chars[1..chars.len() - 1]
                .iter()
                .collect::<String>()
                .trim()
                .to_owned();
        }
    }
    text
}

/// Remove every `<think>…</think>` pair (case-insensitive, spanning newlines). An unmatched open tag
/// leaves the remainder untouched — matching the Python non-greedy regex, which simply does not match.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn strip_think_blocks(input: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        match first_ci(rest, OPEN) {
            Some(open) => {
                out.push_str(&rest[..open]);
                let after_open = &rest[open + OPEN.len()..];
                match first_ci(after_open, CLOSE) {
                    Some(close) => rest = &after_open[close + CLOSE.len()..],
                    None => {
                        out.push_str(&rest[open..]);
                        return out;
                    }
                }
            }
            None => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

/// Byte offset of the first case-insensitive occurrence of an ASCII `needle`. Offsets land on ASCII
/// tag boundaries, so callers can slice safely even when the surrounding text is Unicode.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn first_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Byte offset of the last case-insensitive occurrence of an ASCII `needle`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn last_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len())
        .rev()
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

// ----------------------------------------------------------------------------------------------
// Job handler — native MLX on macOS (sc-5552) and candle on the Windows candle build (sc-5525). The
// body is backend-agnostic: `gen_core::load_textllm("prompt_refine", …)` resolves whichever provider
// is force-linked above. The Python torch `PromptRefiner` remains the fallback on other platforms.
// ----------------------------------------------------------------------------------------------

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_prompt_refine_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    use gen_core::{
        CancelFlag, LoadSpec, Progress, TextLlmRequest, TextLlmSampling, WeightsSource,
    };

    let payload = &job.payload;
    let original_prompt = payload
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    if original_prompt.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Prompt refinement requires a non-empty prompt.".to_owned(),
        ));
    }
    let guide = payload
        .get("guide")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let workflow = payload
        .get("workflow")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_REFINE_MODEL)
        .to_owned();
    let max_new_tokens = payload
        .get("maxNewTokens")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(512);

    let system = build_refine_system_prompt(guide.as_deref(), workflow.as_deref());
    let weights_dir = resolve_app_managed_model_dir(settings, &model, "prompt-refine model path")?;
    // Attribute the run to the active backend (MLX on macOS, candle off-Mac) on the streamed progress
    // + UI architecture pill (mirrors the image/video paths), not the gpu-id device label.
    let backend = REFINE_BACKEND;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.1,
            "Loading prompt-refinement model.",
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(u32, u32)>(64);
    let blocking_cancel = cancel.clone();
    let job_id = job.id.clone();
    let prompt = original_prompt.clone();
    let engine_label = model.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<String> {
        emit_event(
            "prompt_refine_load_start",
            json!({ "jobId": job_id, "engine": engine_label }),
        );
        let refiner = gen_core::load_textllm(
            PROMPT_REFINE_ENGINE_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .map_err(|error| WorkerError::Engine(format!("prompt-refine load failed: {error}")))?;
        emit_event(
            "prompt_refine_load_complete",
            json!({ "jobId": job_id, "engine": engine_label }),
        );
        if blocking_cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        let request = TextLlmRequest {
            system,
            prompt,
            sampling: TextLlmSampling {
                temperature: 0.7,
                top_p: 0.9,
                max_new_tokens,
                seed: None,
            },
            cancel: blocking_cancel.clone(),
        };
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { current, total } = progress {
                let _ = tx.blocking_send((current, total));
            }
        };
        let output = refiner
            .generate(&request, &mut on_progress)
            .map_err(|error| {
                WorkerError::Engine(format!("prompt-refine generation failed: {error}"))
            })?;
        Ok(output.text)
    });

    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some((current, total)) => {
                        let within = if total > 0 {
                            (current as f64 / total as f64).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        update_job(
                            api,
                            &job.id,
                            refine_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                0.4 + 0.5 * within,
                                "Refining prompt…",
                                None,
                                backend,
                            ),
                        )
                        .await?;
                    }
                    None => break,
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                match check_cancel(api, &job.id, CANCEL_MESSAGE).await {
                    Ok(()) => {}
                    Err(WorkerError::Canceled(_)) => cancel.cancel(),
                    Err(error) => return Err(error),
                }
            }
        }
    }

    let raw = blocking
        .await
        .map_err(|error| task_join_error("prompt refine task join", error))??;
    let refined = clean_refine_output(&raw);
    if refined.is_empty() {
        return Err(WorkerError::Engine(
            "The prompt-refinement model returned an empty prompt.".to_owned(),
        ));
    }
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Prompt refined.",
            Some(refine_result(&original_prompt, &refined)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// On platforms with no native prompt-refine provider (neither the macOS MLX twin nor the Windows
/// candle build — e.g. Linux, or the candle-less Desktop installer), the capability is never
/// advertised and this arm is unreachable in practice — the Python torch `PromptRefiner` serves
/// `prompt_refine`. Kept so the `run_utility_job` dispatch compiles on all targets.
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_prompt_refine_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Native prompt refinement needs the macOS MLX worker or the Windows candle backend; use the \
         Python torch prompt refiner on this platform."
            .to_owned(),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn refine_progress(
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

/// The `prompt_refine` result payload, parity with the Python `run_prompt_refine_job`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn refine_result(original_prompt: &str, refined_prompt: &str) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("originalPrompt".to_owned(), json!(original_prompt));
    result.insert("refinedPrompt".to_owned(), json!(refined_prompt));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_uses_workflow_medium_and_embeds_guide() {
        let image = build_refine_system_prompt(
            Some("# Z-Image Guide\n\nUse short prompts."),
            Some("image"),
        );
        assert!(image.contains("generative image model"));
        assert!(image.contains("Z-Image Guide"));
        assert!(image.contains("# Model prompt guide"));

        let video = build_refine_system_prompt(None, Some("video"));
        assert!(video.contains("generative video model"));
        assert!(!video.contains("# Model prompt guide"));
    }

    #[test]
    fn system_prompt_defaults_to_image_when_workflow_absent_or_unknown() {
        assert!(build_refine_system_prompt(None, None).contains("generative image model"));
        assert!(
            build_refine_system_prompt(None, Some("anything")).contains("generative image model")
        );
        // Case-insensitive video match (parity with Python `.lower()`).
        assert!(
            build_refine_system_prompt(None, Some(" VIDEO ")).contains("generative video model")
        );
    }

    #[test]
    fn clean_output_strips_reasoning_and_quoting() {
        assert_eq!(
            clean_refine_output("<think>plan</think>A vivid sunset over hills."),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("\"A vivid sunset over hills.\""),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("```\nA vivid sunset over hills.\n```"),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("<think>scheming</think>A vivid neon street at midnight."),
            "A vivid neon street at midnight."
        );
    }

    #[test]
    fn clean_output_handles_orphan_close_case_insensitive_and_whitespace() {
        // Orphan closing tag with no open (case-insensitive): keep only the tail.
        assert_eq!(
            clean_refine_output("reasoning</THINK> Final prompt."),
            "Final prompt."
        );
        // Multiple think blocks all stripped.
        assert_eq!(
            clean_refine_output("<think>a</think>X<think>b</think>Y"),
            "XY"
        );
        // Plain whitespace trim, no decoration.
        assert_eq!(clean_refine_output("  spaced out  "), "spaced out");
        // An unmatched OPEN tag is left untouched (Python non-greedy regex would not match).
        assert_eq!(
            clean_refine_output("<think>no close here"),
            "<think>no close here"
        );
    }
}
