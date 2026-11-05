//! Result validation and publication for prompt-refinement jobs.
use super::*;

pub(super) struct RefineCompletion {
    pub(super) task: RefineTask,
    pub(super) original_prompt: String,
    pub(super) target_model_id: Option<String>,
    pub(super) model: String,
    pub(super) thinking_mode_name: &'static str,
    pub(super) max_new_tokens: u32,
    pub(super) done_message: &'static str,
    pub(super) qwen_rewriter: Option<crate::qwen_prompt_rewrite::Rewriter>,
    pub(super) image_reference_count: usize,
    pub(super) emits_json: bool,
    pub(super) output: gen_core::core_llm::TextLlmOutput,
    pub(super) kv_cache: Value,
}

pub(super) async fn finalize_refine_job(
    api: &ApiClient,
    job: &JobSnapshot,
    completion: RefineCompletion,
) -> WorkerResult<()> {
    let RefineCompletion {
        task,
        original_prompt,
        target_model_id,
        model,
        thinking_mode_name,
        max_new_tokens,
        done_message,
        qwen_rewriter,
        image_reference_count,
        emits_json,
        output,
        kv_cache,
    } = completion;
    let backend = REFINE_BACKEND;
    // A JSON task isolates the object (the web parses + validates a caption; image_caption validates
    // here too, and the film plan is parsed strictly by the harness); the free-text rewrite cleans to
    // prose.
    let refined = finalize_refined_output(&output.text, emits_json, target_model_id.as_deref());
    if refined.is_empty() {
        let detail = empty_refine_failure_detail(
            task,
            output.finish_reason,
            output.usage.generated_tokens,
            max_new_tokens,
        );
        let result = refine_failure_result(
            output.thinking.as_deref(),
            &model,
            backend,
            thinking_mode_name,
            output.finish_reason,
            output.usage,
            max_new_tokens,
            kv_cache,
        );
        let mut progress = refine_progress(
            JobStatus::Failed,
            ProgressStage::Failed,
            1.0,
            "Prompt refinement failed.",
            Some(result),
            backend,
        );
        progress.error = Some(detail);
        update_job(api, &job.id, progress).await?;
        return Ok(());
    }
    // sc-8105: the image-caption path validates the cleaned reply is a schema-valid Ideogram caption
    // (carries `compositional_deconstruction`) and KEEPS the element bboxes (no stripping). A malformed
    // reply is handled like an empty one — a clear engine error the caller surfaces (mirroring the
    // magic-prompt malformed-output handling, where a non-caption reply also fails downstream).
    if task == RefineTask::ImageCaption {
        let parsed = serde_json::from_str::<Value>(&refined).map_err(|error| {
            WorkerError::Engine(format!(
                "The image-caption model returned output that is not valid JSON: {error}"
            ))
        })?;
        if !sceneworks_core::ideogram_caption::is_caption(&parsed) {
            return Err(WorkerError::Engine(
                "The image-caption model returned JSON that is not a valid Ideogram caption \
                 (missing the `compositional_deconstruction` section)."
                    .to_owned(),
            ));
        }
    }
    // sc-24113: a Qwen rewrite's reply is a JSON object, not the prompt. Parse it HERE so the job
    // result carries what the UI actually needs — the rewritten prompt as plain editable text in the
    // usual `refinedPrompt` slot, plus a separate `rewriteSuggestion` block for the aspect-ratio
    // offer — rather than making every caller re-parse Qwen's schema. `originalPrompt` is untouched,
    // which is the guarantee that nothing is replaced silently: the two sit side by side and the
    // user chooses.
    //
    // A malformed reply is a typed engine error, not a degraded path: offering an empty or garbled
    // "rewrite" that Apply would paste over the user's prompt is worse than failing the action.
    let mut result_refined = refined.clone();
    let mut rewrite_suggestion = None;
    if let Some(rewriter) = qwen_rewriter {
        // A budget exhausted mid-reasoning is the ONE failure worth distinguishing, because its fix
        // is a number rather than a retry. These checkpoints reason first and emit the JSON object
        // last, so running out loses the object entirely and the generic parse error ("did not
        // return a JSON object") reads as a broken model. Checked BEFORE the parse, since the
        // parse is guaranteed to fail in exactly this case and would otherwise win the report.
        if output.finish_reason == Some(gen_core::core_llm::FinishReason::Length) {
            return Err(WorkerError::Engine(
                crate::qwen_prompt_rewrite::truncated_reply_error(rewriter, max_new_tokens),
            ));
        }
        let suggestion = crate::qwen_prompt_rewrite::parse_rewrite(&refined).map_err(|error| {
            WorkerError::Engine(format!(
                "The Qwen Image 2.1 prompt rewriter ({}) returned output this build cannot read: \
                 {error}. The reply began: {}",
                rewriter.repo(),
                // Discarding the reply leaves nobody anything to act on — not the user reading the
                // job's error, not whoever reads the log afterwards. Bounded, because a reply can
                // be thousands of characters and an error message is not a transcript.
                crate::qwen_prompt_rewrite::reply_excerpt(&refined, 500)
            ))
        })?;
        rewrite_suggestion = Some(crate::qwen_prompt_rewrite::suggestion_result_block(
            &suggestion,
            rewriter,
            image_reference_count,
        ));
        result_refined = suggestion.rewritten_prompt;
    }
    // sc-24029 widened this to carry the finish reason, usage and the ceiling on every refine
    // result; sc-24113 builds it eagerly so the rewrite's suggestion block can ride alongside.
    let mut result = refine_result(
        &original_prompt,
        &result_refined,
        output.thinking.as_deref(),
        &model,
        backend,
        thinking_mode_name,
        output.finish_reason,
        output.usage,
        max_new_tokens,
        kv_cache,
    );
    if let Some(block) = rewrite_suggestion {
        result.insert("rewriteSuggestion".to_owned(), Value::Object(block));
    }
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            done_message,
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}
