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
    pub(super) h3_expand: Option<H3ExpandReply>,
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
        h3_expand,
        image_reference_count,
        emits_json,
        output,
        kv_cache,
    } = completion;
    let backend = REFINE_BACKEND;
    // SW-7102: an expansion that ran out of budget keeps what it wrote rather than failing, before
    // the empty and parse checks below would refuse the unterminated object. The expander reasons
    // first and emits the JSON object last, so the cut lands in the caption (kept to its last whole
    // sentence) or, when the reasoning used the whole budget, before it (the job has already asked
    // again with reasoning off, `Recovery::AnswerOnlyRetry`). Only a reply with no caption at all is
    // an error.
    let truncated_expansion = match &h3_expand {
        Some(reply) if output.finish_reason == Some(gen_core::core_llm::FinishReason::Length) => {
            let partial = crate::minimax_h3_image_prompt_expand::salvage_truncated_reply(
                &output.text,
                reply.workflow,
                &reply.allowed_ratios,
                &reply.aspect,
            )
            .ok_or_else(|| {
                WorkerError::Engine(
                    crate::minimax_h3_image_prompt_expand::truncated_reply_error(max_new_tokens),
                )
            })?;
            tracing::warn!(
                event = "minimax_h3_image_expansion_truncated",
                job_id = %job.id,
                max_new_tokens,
                whole_sentences = partial.whole_sentences,
                kept_chars = partial.expanded_caption.len(),
                "MiniMax-H3-Image prompt expansion ran out of budget; kept the caption written so far"
            );
            Some(partial)
        }
        _ => None,
    };
    // A JSON task isolates the object (the web parses + validates a caption; image_caption validates
    // here too, and the film plan is parsed strictly by the harness); the free-text rewrite cleans to
    // prose.
    let refined = match &truncated_expansion {
        // The kept caption is already clean text; the unterminated object is not.
        Some(partial) => partial.expanded_caption.clone(),
        None => finalize_refined_output(&output.text, emits_json, target_model_id.as_deref()),
    };
    if refined.is_empty() {
        let detail = empty_refine_failure_detail(
            task,
            output.finish_reason,
            output.usage.generated_tokens,
            max_new_tokens,
        );
        let mut result = refine_failure_result(
            output.thinking.as_deref(),
            &model,
            backend,
            thinking_mode_name,
            output.finish_reason,
            output.usage,
            max_new_tokens,
            kv_cache,
        );
        record_loop_tier(job, &model, &mut result);
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
    // Image captions keep grounded boxes and report an unreadable reply as an engine error.
    let refined = if task == RefineTask::ImageCaption {
        prepare_image_caption(refined, &model)?
    } else {
        refined
    };
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
    // SW-7102: the expansion's reply is the engine's strict two-field object. The expanded caption
    // takes the usual `refinedPrompt` slot, beside the untouched `originalPrompt`; the proposed
    // ratio rides `rewriteSuggestion` for the UI to offer. A reply that is not exactly the schema
    // is refused, never repaired. A reply the budget cut off is the one exception: its caption was
    // kept above, flagged `truncated` for the UI, and it offers a ratio only if it reached one.
    if let Some(reply) = &h3_expand {
        let (caption, aspect_ratio, truncated) = match truncated_expansion {
            Some(partial) => (partial.expanded_caption, partial.aspect_ratio, true),
            None => {
                let expansion = crate::minimax_h3_image_prompt_expand::parse_reply(
                    &refined,
                    reply.workflow,
                    &reply.allowed_ratios,
                    &reply.aspect,
                )
                .map_err(|error| {
                    WorkerError::Engine(format!(
                        "The MiniMax-H3-Image prompt expander returned output this build cannot \
                         read: {error}. The reply began: {}",
                        crate::qwen_prompt_rewrite::reply_excerpt(&refined, 500)
                    ))
                })?;
                (
                    expansion.expanded_caption,
                    Some(expansion.aspect_ratio),
                    false,
                )
            }
        };
        rewrite_suggestion = Some(
            crate::minimax_h3_image_prompt_expand::suggestion_result_block(
                aspect_ratio.as_deref(),
                reply.workflow,
                reply.model_id,
                reply.tier,
                truncated,
                reply.recovery,
            ),
        );
        result_refined = caption;
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
    record_loop_tier(job, &model, &mut result);
    if let Some(block) = rewrite_suggestion {
        result.insert("rewriteSuggestion".to_owned(), Value::Object(block));
    }
    // SW-7202: the expander's executed weights folder (its `LoadSpec` source), beside the model.
    // SW-7208: and the expander model and tier that ran.
    if let Some(reply) = &h3_expand {
        if let Some(identity) = result
            .get_mut("executionIdentity")
            .and_then(Value::as_object_mut)
        {
            identity.insert(
                "weightsDir".to_owned(),
                Value::from(reply.weights_source.clone()),
            );
            identity.insert("expanderModelId".to_owned(), Value::from(reply.model_id));
            if let Some(tier) = reply.tier {
                identity.insert("tier".to_owned(), Value::from(tier));
            }
            // SW-7204: the proposer, block draft and per-pass counts the expansion decoded with.
            if !reply.decode.is_null() {
                identity.insert("decode".to_owned(), reply.decode.clone());
            }
        }
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

// Keep the selected source tier next to the executed model and backend.
fn record_loop_tier(job: &JobSnapshot, model: &str, result: &mut JsonObject) {
    if model == sceneworks_core::loop_utility::MODEL_REPO {
        if let (Some(tier), Some(identity)) = (
            job.payload.get("variant"),
            result
                .get_mut("executionIdentity")
                .and_then(Value::as_object_mut),
        ) {
            identity.insert("variant".to_owned(), tier.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SW-7202: an expansion's result names the weights folder the job LOADED (its `LoadSpec`
    /// source) on `executionIdentity.weightsDir`, so a check of which tier ran reads the folder
    /// rather than the requested tier echoed back on the suggestion block. SW-7208: it also names
    /// the expander model and tier that ran, for either model. No model runs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_expansion_result_names_the_model_tier_and_weights_folder_it_loaded() {
        const STAMP: &str = "2026-11-19T00:00:00Z";
        for (model_id, repo, loaded) in [
            (
                "minimax_h3_image_pe",
                "MiniMaxAI/MiniMax-H3-Image-PE",
                "/hub/models--SceneWorks--minimax-h3-image-pe/snapshots/rev/q4-full",
            ),
            (
                "vision_describe_qwen3_8_27b",
                "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "/hub/models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/rev/q4-full",
            ),
        ] {
            let (api_url, posts) = crate::test_api_stub::smoke_api(
                crate::test_api_stub::smoke_job("", "prompt_refine", None, None, "w", STAMP),
                crate::test_api_stub::smoke_worker("", "job_h3_expand", STAMP),
                None,
            )
            .await;
            let temp = tempfile::tempdir().unwrap();
            let mut settings = Settings::for_test(temp.path().to_path_buf());
            settings.api_url = api_url;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot = serde_json::from_value(crate::test_api_stub::smoke_job(
                "job_h3_expand",
                "prompt_refine",
                None,
                None,
                "w",
                STAMP,
            ))
            .unwrap();
            let output = gen_core::core_llm::TextLlmOutput {
                text: r#"{"expanded_caption": "A lighthouse on a basalt headland.", "aspect_ratio": "9:16"}"#
                    .to_owned(),
                finish_reason: Some(gen_core::core_llm::FinishReason::Stop),
                ..Default::default()
            };
            finalize_refine_job(
                &api,
                &job,
                RefineCompletion {
                    task: RefineTask::MiniMaxH3ImageExpand,
                    original_prompt: "a lighthouse".to_owned(),
                    target_model_id: None,
                    model: repo.to_owned(),
                    thinking_mode_name: "enabled",
                    max_new_tokens: 4096,
                    done_message: "done",
                    qwen_rewriter: None,
                    h3_expand: Some(H3ExpandReply {
                        workflow: crate::minimax_h3_image_prompt_expand::Workflow::TextToImage,
                        aspect: gen_core::core_llm::minimax_h3_image_pe::AspectSelection::Adaptive,
                        allowed_ratios: vec!["9:16".to_owned()],
                        model_id,
                        tier: Some("q4"),
                        weights_source: loaded.to_owned(),
                        decode: json!({ "proposer": "block_draft" }),
                        recovery: crate::minimax_h3_image_prompt_expand::Recovery::None,
                    }),
                    image_reference_count: 0,
                    emits_json: true,
                    output,
                    kv_cache: Value::Null,
                },
            )
            .await
            .expect("the expansion completes");
            let posts = posts.lock().unwrap().clone();
            let result = posts
                .iter()
                .rev()
                .find(|(_, body)| body["status"] == "completed")
                .map(|(_, body)| body["result"].clone())
                .expect("a completed result");
            let identity = &result["executionIdentity"];
            assert_eq!(identity["weightsDir"], loaded, "{model_id}");
            assert_eq!(identity["expanderModelId"], model_id);
            assert_eq!(identity["model"], repo, "{model_id}");
            assert_eq!(identity["tier"], "q4", "{model_id}");
            assert_eq!(identity["decode"]["proposer"], "block_draft", "{model_id}");
            assert_eq!(result["rewriteSuggestion"]["tier"], "q4");
            assert_eq!(result["rewriteSuggestion"]["rewriterModelId"], model_id);
            assert_eq!(result["rewriteSuggestion"]["whRatio"], "9:16");
            assert_eq!(
                result["refinedPrompt"],
                "A lighthouse on a basalt headland."
            );
        }
    }

    /// Finalize one expansion reply against a stub API; the completed result, or the error.
    async fn finalize_expansion(
        output: gen_core::core_llm::TextLlmOutput,
        recovery: crate::minimax_h3_image_prompt_expand::Recovery,
    ) -> WorkerResult<Value> {
        const STAMP: &str = "2026-11-20T00:00:00Z";
        let (api_url, posts) = crate::test_api_stub::smoke_api(
            crate::test_api_stub::smoke_job("", "prompt_refine", None, None, "w", STAMP),
            crate::test_api_stub::smoke_worker("", "job_h3_cut", STAMP),
            None,
        )
        .await;
        let temp = tempfile::tempdir().unwrap();
        let mut settings = Settings::for_test(temp.path().to_path_buf());
        settings.api_url = api_url;
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(crate::test_api_stub::smoke_job(
            "job_h3_cut",
            "prompt_refine",
            None,
            None,
            "w",
            STAMP,
        ))
        .unwrap();
        finalize_refine_job(
            &api,
            &job,
            RefineCompletion {
                task: RefineTask::MiniMaxH3ImageExpand,
                original_prompt: "a lighthouse".to_owned(),
                target_model_id: None,
                model: "huihui-ai/Huihui-Qwen3.8-27B-abliterated".to_owned(),
                thinking_mode_name: "enabled",
                max_new_tokens: 4096,
                done_message: "done",
                qwen_rewriter: None,
                h3_expand: Some(H3ExpandReply {
                    workflow: crate::minimax_h3_image_prompt_expand::Workflow::TextToImage,
                    aspect: gen_core::core_llm::minimax_h3_image_pe::AspectSelection::Adaptive,
                    allowed_ratios: vec!["9:16".to_owned()],
                    model_id: "vision_describe_qwen3_8_27b",
                    tier: Some("q4"),
                    weights_source: String::new(),
                    decode: Value::Null,
                    recovery,
                }),
                image_reference_count: 0,
                emits_json: true,
                output,
                kv_cache: Value::Null,
            },
        )
        .await?;
        let posts = posts.lock().unwrap().clone();
        Ok(posts
            .iter()
            .rev()
            .find(|(_, body)| body["status"] == "completed")
            .map(|(_, body)| body["result"].clone())
            .expect("a completed result"))
    }

    /// A reply the budget cut off inside its caption completes with the caption up to its last
    /// whole sentence, flagged `truncated`, and offers no ratio it never reached — it does not fail
    /// the expansion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_expansion_cut_off_by_its_budget_keeps_its_whole_sentences() {
        let result = finalize_expansion(
            gen_core::core_llm::TextLlmOutput {
                text: r#"{"expanded_caption": "A lighthouse on a basalt headland. Waves break bel"#
                    .to_owned(),
                finish_reason: Some(gen_core::core_llm::FinishReason::Length),
                ..Default::default()
            },
            crate::minimax_h3_image_prompt_expand::Recovery::AnswerOnlyRetry,
        )
        .await
        .expect("a cut-off expansion still completes");
        assert_eq!(
            result["refinedPrompt"],
            "A lighthouse on a basalt headland."
        );
        assert_eq!(result["originalPrompt"], "a lighthouse");
        let suggestion = &result["rewriteSuggestion"];
        assert_eq!(suggestion["truncated"], true);
        assert_eq!(suggestion["reasoningSkipped"], true);
        assert!(suggestion.get("whRatio").is_none(), "{suggestion}");
    }

    /// Only a cut-off reply with no caption at all is an error, and it names no setting the app
    /// does not have.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_expansion_cut_off_before_any_caption_fails_without_naming_a_setting() {
        let error = finalize_expansion(
            gen_core::core_llm::TextLlmOutput {
                text: String::new(),
                thinking: Some("Let me think about the lighthouse".to_owned()),
                finish_reason: Some(gen_core::core_llm::FinishReason::Length),
                ..Default::default()
            },
            crate::minimax_h3_image_prompt_expand::Recovery::AnswerOnlyRetry,
        )
        .await
        .expect_err("nothing to keep");
        let message = error.to_string();
        assert!(message.contains("4096-token budget"), "{message}");
        assert!(!message.contains("maxNewTokens"), "{message}");
    }
}
