use super::*;

#[cfg(native_lane)]
/// sc-24113: the Qwen rewrite's reply travels the SAME cleanup the other JSON tasks take, and
/// then parses into an editable prompt plus an aspect suggestion.
///
/// This is the seam the two unit-tested halves meet at, and it is the one that would break
/// silently: `qwen_prompt_rewrite::parse_rewrite` is tested against an isolated object, but the
/// model does not emit one — both PE checkpoints emit a REASONING BLOCK first, and the T2I
/// system prompt's own instruction ("Return one strictly valid JSON object on a single line")
/// is routinely wrapped in a code fence anyway. `clean_json_output` is what bridges them, so
/// exercising them together over a realistic raw reply is what proves the task is wired to the
/// right cleanup rather than to `clean_refine_output` (which would unwrap quotes and hand the
/// parser prose).
#[test]
fn a_raw_rewriter_reply_survives_the_reasoning_block_and_the_code_fence() {
    use crate::qwen_prompt_rewrite::{parse_rewrite, suggested_resolution, Rewriter};

    // A T2I reply as the checkpoint actually emits it.
    let raw = "<think>\nThe user wants a wide harbour. Step 2 fixes the frame: horizontal, so \
                   3:2 unless a wider cine framing is wanted. They said \"cinematic\", so 16:9.\n\
                   </think>\n```json\n{\"rewritten_prompt\": \"A wide harbour at dusk, fishing \
                   boats moored along a stone quay\", \"wh_ratio\": \"16:9\"}\n```";
    let cleaned = clean_json_output(raw);
    let suggestion = parse_rewrite(&cleaned).expect("the cleaned reply parses");
    assert_eq!(
        suggestion.rewritten_prompt,
        "A wide harbour at dusk, fishing boats moored along a stone quay"
    );
    assert_eq!(suggested_resolution(&suggestion), Some("2752x1536"));

    // The task classifies from the discriminator, emits JSON (so it takes the cleanup above
    // rather than the prose one), and reads reference images when they are present.
    let task = RefineTask::from_payload(Some(sceneworks_core::contracts::QWEN_IMAGE_REWRITE_TASK));
    assert_eq!(task, RefineTask::QwenImageRewrite);
    assert!(task.emits_json(), "the rewrite must take the JSON cleanup");
    assert!(task.takes_reference_images());
    // ... but it is NOT a vision task: the prompt stays required and zero references is legal,
    // which is the distinction that selects between the two rewriters.
    assert!(!task.is_vision());
    assert!(!task.is_caption());

    // An I2I reply, with the edit template's third field.
    let raw = "<think>\nPreserve the courier, replace the backdrop.\n</think>\n\
                   {\"rewritten_prompt\": \"Replace the grey studio backdrop with a sunlit alley\", \
                   \"wh_ratio\": \"\", \"ratio_follow\": \"<image1>\"}";
    let suggestion = parse_rewrite(&clean_json_output(raw)).expect("the edit reply parses");
    assert_eq!(
        suggestion.rewritten_prompt,
        "Replace the grey studio backdrop with a sunlit alley"
    );
    assert_eq!(suggested_resolution(&suggestion), None);

    // The block the job result carries, assembled the way the completion arm assembles it.
    let block =
        crate::qwen_prompt_rewrite::suggestion_result_block(&suggestion, Rewriter::Editing, 2);
    assert_eq!(block.get("followsReferenceIndex"), Some(&Value::from(0)));
    assert_eq!(
        block.get("rewriterModelId"),
        Some(&Value::from("qwen_image_2_1_pe_i2i"))
    );
}

/// sc-24113 BLOCKER REGRESSION. The generate closure's image invariant must hold for every
/// task × image-presence combination the job can actually produce.
///
/// The bug: the assert read `carries_image == is_vision_task`, an equality that was true only
/// while `image_caption`/`image_describe` were the sole image carriers. `QwenImageRewrite`
/// carries images with `is_vision()` FALSE, so **every I2I rewrite panicked in a debug build** —
/// and nothing caught it, because the assert sits after a provider load that no test in this
/// lane can reach. Hence the predicate is extracted and tested here instead.
///
/// Restoring the equality (`carries_image == takes_reference_images`) in
/// `image_carrier_invariant_holds` reds the `QwenImageRewrite`-without-references row below;
/// restoring the original (`== is_vision()`) reds the with-references row too.
#[test]
fn the_generate_closures_image_invariant_holds_for_every_task_and_image_combination() {
    // The property the rewrite broke, stated on its own: it takes images, and it is NOT a
    // vision task. Both halves matter — the first is why it carries a `Content::Image`, the
    // second is why the prompt stays required and zero references is legal.
    assert!(RefineTask::QwenImageRewrite.takes_reference_images());
    assert!(!RefineTask::QwenImageRewrite.is_vision());

    for task in [
        RefineTask::Rewrite,
        RefineTask::MagicPrompt,
        RefineTask::ImageCaption,
        RefineTask::ImageDescribe,
        RefineTask::FilmPlan,
        RefineTask::QwenImageRewrite,
    ] {
        let takes = task.takes_reference_images();
        // WITH images: legal exactly for a task that declares it takes them.
        assert_eq!(
            image_carrier_invariant_holds(true, takes),
            takes,
            "{task:?} with images"
        );
        // WITHOUT images: always legal. This is the row the old equality got wrong for the
        // rewrite — a text-to-image rewrite carries none, and that is not a violation.
        assert!(
            image_carrier_invariant_holds(false, takes),
            "{task:?} without images must never trip the assert"
        );
    }

    // Spelled out for the one task whose two rows the two historical forms of this assert
    // disagreed about, so a failure names the case rather than a loop index.
    assert!(image_carrier_invariant_holds(
        true,
        RefineTask::QwenImageRewrite.takes_reference_images()
    ));
    assert!(image_carrier_invariant_holds(
        false,
        RefineTask::QwenImageRewrite.takes_reference_images()
    ));
}

/// sc-24113 BLOCKER REGRESSION, the data half: drive the I2I rewrite's turn assembly and reply
/// handling with a fixture image and a fixture LLM reply.
///
/// Everything the job does between "references resolved" and "result posted" EXCEPT the
/// provider call, which needs weights this lane does not run. The `Content::Image` blocks are
/// built from a real decoded PNG through the same `load_caption_image_ref` the job uses, so
/// `carries_image` is true for the same reason it is true in production — which is the
/// condition that used to panic.
#[test]
fn the_i2i_rewrite_path_assembles_image_turns_and_parses_a_fixture_reply() {
    use crate::qwen_prompt_rewrite::{parse_rewrite, suggestion_result_block, Rewriter};
    use gen_core::core_llm::{Content, Message, Role};

    // Two real on-disk references, distinguishable so an ordering bug is visible.
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (index, colour) in [[200u8, 30, 30], [30, 200, 30]].into_iter().enumerate() {
        let path = dir.path().join(format!("ref{index}.png"));
        image::RgbImage::from_pixel(8, 8, image::Rgb(colour))
            .save(&path)
            .unwrap();
        paths.push(path);
    }
    let image_refs: Vec<_> = paths
        .iter()
        .map(|path| load_caption_image_ref(path).expect("fixture reference decodes"))
        .collect();
    assert_eq!(image_refs.len(), 2);

    // Request-shaped rewriter selection: references present ⇒ the EDITING half.
    let rewriter = Rewriter::for_reference_count(image_refs.len());
    assert_eq!(rewriter, Rewriter::Editing);

    // The user turn the job builds: every image BEFORE the instruction text, exactly as the
    // caption tasks do.
    let task = RefineTask::QwenImageRewrite;
    let mut content: Vec<Content> = image_refs.into_iter().map(Content::Image).collect();
    content.push(Content::text("swap the backdrop".to_owned()));
    let user = Message {
        role: Role::User,
        content,
        thinking: None,
        tool_calls: Vec::new(),
    };
    let carries_image = user
        .content
        .iter()
        .any(|block| matches!(block, Content::Image(_)));

    // THE REGRESSION: this exact pair reached the old `carries_image == is_vision_task` assert
    // and panicked. `true, false` — images present, not a vision task.
    assert!(carries_image);
    assert!(!task.is_vision());
    assert!(image_carrier_invariant_holds(
        carries_image,
        task.takes_reference_images()
    ));

    // The fixture reply, through the same cleanup + parse the completion arm runs.
    let reply = "<think>Keep the courier, replace the backdrop.</think>\n\
                     {\"rewritten_prompt\": \"Replace the grey backdrop with a sunlit alley\", \
                     \"wh_ratio\": \"\", \"ratio_follow\": \"<image2>\"}";
    let suggestion = parse_rewrite(&clean_json_output(reply)).expect("the fixture reply parses");
    assert_eq!(
        suggestion.rewritten_prompt,
        "Replace the grey backdrop with a sunlit alley"
    );
    let block = suggestion_result_block(&suggestion, rewriter, 2);
    // `<image2>` is the SECOND reference the turn carried, i.e. index 1.
    assert_eq!(block.get("followsReferenceIndex"), Some(&Value::from(1)));
    assert_eq!(block.get("rewriter"), Some(&Value::from("i2i")));
}

/// The rewrite's sampling and token budget are the PUBLISHED recipe, not the house defaults.
///
/// Asserted through `RefineTask` (not just the constants) because the wiring is what could
/// drift: an arm added to `temperature()` or `resolve_max_new_tokens` that fell through to the
/// generic rewrite would silently sample the checkpoints at 0.7/0.9 instead of 1.0/0.95/20.
#[test]
fn the_rewrite_task_carries_qwens_published_sampling_rather_than_the_house_defaults() {
    let task = RefineTask::QwenImageRewrite;
    assert_eq!(
        task.temperature(),
        crate::qwen_prompt_rewrite::REWRITE_TEMPERATURE
    );
    assert_ne!(task.temperature(), RefineTask::Rewrite.temperature());
    // The budget is PER REWRITER — each card's own published ceiling, not the worker's caption
    // default. Capping below the card is not a safe economy: these checkpoints reason first and
    // emit the JSON object last, so a budget that runs out loses the object entirely.
    use crate::qwen_prompt_rewrite::{Rewriter, I2I_MAX_NEW_TOKENS, T2I_MAX_NEW_TOKENS};
    assert_eq!(T2I_MAX_NEW_TOKENS, 16256);
    assert_eq!(I2I_MAX_NEW_TOKENS, 24000);
    assert_ne!(T2I_MAX_NEW_TOKENS, I2I_MAX_NEW_TOKENS);
    assert_eq!(
        resolve_max_new_tokens(&serde_json::Map::new(), task, Some(Rewriter::TextToImage)),
        T2I_MAX_NEW_TOKENS
    );
    assert_eq!(
        resolve_max_new_tokens(&serde_json::Map::new(), task, Some(Rewriter::Editing)),
        I2I_MAX_NEW_TOKENS
    );
    // Never the caption ceiling, which is an order of magnitude below either card.
    assert!(
        resolve_max_new_tokens(&serde_json::Map::new(), task, None)
            > DEFAULT_CAPTION_MAX_NEW_TOKENS
    );
    // An explicit override still wins, the same escape hatch every other task has.
    let payload = serde_json::json!({ "maxNewTokens": 256 });
    assert_eq!(
        resolve_max_new_tokens(payload.as_object().unwrap(), task, Some(Rewriter::Editing)),
        256
    );
    // The progress copy says "review", not "applied" — the rewrite is a suggestion.
    assert!(task.done_message().contains("review"));
}

// `refine_result` exists only on a backend build; this gate was on this test before sc-24113's
// tests were inserted above it and captured it.
#[cfg(native_lane)]
#[test]
fn planner_result_keeps_thinking_separate_and_records_actual_execution_identity() {
    let result = refine_result(
        "brief",
        "{\"shots\":[]}",
        Some("private reasoning"),
        "Qwen/Qwen3.6-27B",
        "mlx",
        "enabled",
        Some(gen_core::core_llm::FinishReason::Stop),
        gen_core::core_llm::Usage {
            prompt_tokens: 1_024,
            generated_tokens: 32,
        },
        4_096,
        serde_json::Value::Null,
    );
    assert_eq!(result["refinedPrompt"], "{\"shots\":[]}");
    assert_eq!(result["thinking"], "private reasoning");
    assert_eq!(result["executionIdentity"]["provider"], "native");
    assert_eq!(result["executionIdentity"]["model"], "Qwen/Qwen3.6-27B");
    assert_eq!(result["executionIdentity"]["backend"], "mlx");
    assert_eq!(result["executionIdentity"]["thinkingMode"], "enabled");
    assert!(!result["refinedPrompt"]
        .as_str()
        .unwrap()
        .contains("private reasoning"));
}

/// sc-24029. A `Rewrite` that stops on `Length` is non-empty, so it completes as a SUCCESS —
/// and the caller that must not compile truncated prose (the film planner's per-shot rewrite)
/// could not tell, because `finishReason` was recorded only on the failure result. The success
/// result now carries the same additive `generation` object.
#[cfg(native_lane)]
#[test]
fn a_truncated_but_non_empty_rewrite_reports_its_length_finish_on_the_success_result() {
    use gen_core::core_llm::{FinishReason, Usage};

    let result = refine_result(
        "a courier sets a parcel down",
        "A courier sets the parcel on the bench. overall_soundscape:\n  The only",
        None,
        "TheDrummer/Anubis-Mini-8B-v1",
        "mlx",
        "disabled",
        Some(FinishReason::Length),
        Usage {
            prompt_tokens: 900,
            generated_tokens: 1_536,
        },
        1_536,
        serde_json::Value::Null,
    );
    assert_eq!(result["generation"]["finishReason"], "length");
    assert_eq!(result["generation"]["usage"]["promptTokens"], 900);
    assert_eq!(result["generation"]["usage"]["generatedTokens"], 1_536);
    assert_eq!(result["generation"]["maxNewTokens"], 1_536);
    // Additive only: the keys every existing reader uses are untouched.
    assert_eq!(result["originalPrompt"], "a courier sets a parcel down");
    assert!(result["refinedPrompt"].as_str().unwrap().ends_with("only"));
    assert_eq!(result["executionIdentity"]["backend"], "mlx");
}

/// sc-20688: a refine decode the engine refuses still records its `llm_kv_cache` event, with
/// the job's requested policy, the outcome and the refusal, under the refine engine lane.
#[cfg(native_lane)]
#[test]
fn a_refused_refine_generation_records_its_kv_cache_event() {
    use crate::llm_kv_cache::test_support::{events_for, refusal, RefusingLlm};
    let request = gen_core::core_llm::TextLlmRequest {
        kv_compression: gen_core::core_llm::KvCompressionPolicy::Qualified,
        ..Default::default()
    };
    let job_id = "prompt-refine-refused";
    let error = generate_refined(&RefusingLlm, &request, &mut |_| {}, job_id).unwrap_err();
    assert!(
        matches!(&error, WorkerError::Engine(message) if message.contains("prompt-refine generation failed")),
        "{error:?}"
    );
    let events = events_for(job_id);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["engine"], REFINE_BACKEND);
    assert_eq!(events[0]["kvCache"]["policy"], "qualified");
    assert_eq!(events[0]["kvCache"]["outcome"], "refused");
    assert_eq!(events[0]["kvCache"]["error"], refusal().to_string());
}

/// sc-20682: both the success and the failure result carry the KV cache the generation asked
/// for and ran on under `generation.kvCache`, beside the keys every existing reader uses.
#[cfg(native_lane)]
#[test]
fn refine_results_record_the_kv_cache_the_generation_ran_on() {
    use gen_core::core_llm::{
        FinishReason, KvCacheFallbackReason, KvCacheReport, KvCompressionPolicy, Usage,
    };

    let report = KvCacheReport::dense(KvCacheFallbackReason::BelowMinimumContext, None);
    let block = crate::llm_kv_cache::kv_cache_block(KvCompressionPolicy::Qualified, Some(&report));
    let usage = Usage {
        prompt_tokens: 12,
        generated_tokens: 3,
    };
    let success = refine_result(
        "idea",
        "prompt",
        None,
        "TheDrummer/Anubis-Mini-8B-v1",
        "mlx",
        "auto",
        Some(FinishReason::Stop),
        usage,
        64,
        block.clone(),
    );
    let failure = refine_failure_result(
        None,
        "TheDrummer/Anubis-Mini-8B-v1",
        "mlx",
        "auto",
        Some(FinishReason::Stop),
        usage,
        64,
        block.clone(),
    );
    for result in [&success, &failure] {
        assert_eq!(result["generation"]["kvCache"], block);
        assert_eq!(result["generation"]["kvCache"]["policy"], "qualified");
        assert_eq!(result["generation"]["kvCache"]["ranCompressed"], false);
        assert_eq!(
            result["generation"]["kvCache"]["fallbackReason"],
            "below_minimum_context"
        );
        assert_eq!(result["generation"]["finishReason"], "stop");
    }
}

/// sc-20682: the refine request carries the job's compressed-KV policy beside the task's
/// sampler, budget, constraint and reasoning mode.
#[test]
fn the_refine_request_carries_the_job_kv_policy() {
    use gen_core::core_llm::{
        CancelFlag, Constraint, KvCompressionPolicy, Message, Sampling, ThinkingMode,
    };
    for policy in [KvCompressionPolicy::Off, KvCompressionPolicy::Qualified] {
        let request = refine_text_request(
            vec![Message::user("idea")],
            Sampling {
                temperature: 0.4,
                ..Sampling::default()
            },
            96,
            Some(Constraint::Json),
            CancelFlag::new(),
            ThinkingMode::Disabled,
            policy,
        );
        assert_eq!(request.kv_compression, policy);
        assert_eq!(request.max_new_tokens, 96);
        assert_eq!(request.sampling.temperature, 0.4);
        assert_eq!(request.constraint, Some(Constraint::Json));
        assert_eq!(request.thinking, ThinkingMode::Disabled);
        assert_eq!(request.seed, None);
        assert_eq!(request.messages.len(), 1);
    }
}

#[test]
fn empty_length_result_preserves_reasoning_usage_and_actionable_failure() {
    use gen_core::core_llm::{FinishReason, Usage};

    let result = refine_failure_result(
        Some("private bounded reasoning"),
        "Qwen/Qwen3.6-27B",
        "mlx",
        "enabled",
        Some(FinishReason::Length),
        Usage {
            prompt_tokens: 2_048,
            generated_tokens: 4_096,
        },
        4_096,
        serde_json::Value::Null,
    );
    assert_eq!(result["thinking"], "private bounded reasoning");
    assert_eq!(result["generation"]["finishReason"], "length");
    assert_eq!(result["generation"]["usage"]["promptTokens"], 2_048);
    assert_eq!(result["generation"]["usage"]["generatedTokens"], 4_096);
    assert_eq!(result["generation"]["maxNewTokens"], 4_096);
    assert_eq!(result["executionIdentity"]["model"], "Qwen/Qwen3.6-27B");
    assert_eq!(result["executionIdentity"]["thinkingMode"], "enabled");
    assert!(result.get("refinedPrompt").is_none());

    let detail = empty_refine_failure_detail(
        RefineTask::FilmPlan,
        Some(FinishReason::Length),
        4_096,
        4_096,
    );
    assert!(detail.contains("exhausted its 4096-token output budget"));
    assert!(detail.contains("generated 4096 tokens"));
    assert!(detail.contains("Select Thinking disabled"));
    assert!(detail.contains("manual plan editing"));
}

#[test]
fn empty_non_length_result_keeps_the_established_generic_failure() {
    use gen_core::core_llm::FinishReason;

    assert_eq!(
        empty_refine_failure_detail(RefineTask::FilmPlan, Some(FinishReason::Stop), 12, 4_096,),
        "The prompt-refinement model returned an empty prompt."
    );
    assert_eq!(
        empty_refine_failure_detail(RefineTask::FilmPlan, None, 0, 4_096),
        "The prompt-refinement model returned an empty prompt."
    );
    assert_eq!(
        empty_refine_failure_detail(
            RefineTask::ImageCaption,
            Some(FinishReason::Length),
            4_096,
            4_096,
        ),
        "The prompt-refinement model returned an empty prompt."
    );
}

#[test]
fn json_decode_constraint_yields_only_to_a_reasoning_capable_thinking_run() {
    use gen_core::core_llm::{Constraint, ThinkingMode};

    for mode in [ThinkingMode::Enabled, ThinkingMode::Auto] {
        assert_eq!(
            json_decode_constraint(true, true, mode),
            None,
            "a live thinking channel must be able to close before the JSON answer"
        );
    }
    assert_eq!(
        json_decode_constraint(true, true, ThinkingMode::Disabled),
        Some(Constraint::Json),
        "Qwen no-think generation remains JSON constrained"
    );
    for mode in [
        ThinkingMode::Enabled,
        ThinkingMode::Auto,
        ThinkingMode::Disabled,
    ] {
        assert_eq!(
            json_decode_constraint(true, false, mode),
            Some(Constraint::Json),
            "a non-reasoning refiner such as Anubis keeps constrained JSON"
        );
        assert_eq!(
            json_decode_constraint(false, true, mode),
            None,
            "prose tasks remain unconstrained"
        );
    }
}

#[test]
fn downscale_to_pixel_budget_caps_large_and_keeps_small() {
    // A 4000×3000 (12 MP) reference is shrunk under the budget while holding the 4:3 aspect ratio
    // (this is the case that produced the ~272 GB dense-attention allocation at full size).
    let big = image::DynamicImage::new_rgb8(4000, 3000);
    let capped = downscale_to_pixel_budget(big, VISION_REFERENCE_MAX_PIXELS);
    let pixels = u64::from(capped.width()) * u64::from(capped.height());
    assert!(
        pixels <= VISION_REFERENCE_MAX_PIXELS,
        "capped pixels {pixels} exceed budget {VISION_REFERENCE_MAX_PIXELS}"
    );
    let ratio = capped.width() as f64 / capped.height() as f64;
    assert!(
        (ratio - 4.0 / 3.0).abs() < 0.01,
        "aspect ratio drifted: {ratio}"
    );

    // An already-small image passes through untouched (no upscaling).
    let small = image::DynamicImage::new_rgb8(640, 480);
    let kept = downscale_to_pixel_budget(small, VISION_REFERENCE_MAX_PIXELS);
    assert_eq!((kept.width(), kept.height()), (640, 480));
}

#[test]
fn refine_task_classifies_the_discriminator_and_derives_properties() {
    // Classification is case-insensitive and trims; anything unknown/absent is the default rewrite.
    assert_eq!(RefineTask::from_payload(None), RefineTask::Rewrite);
    assert_eq!(RefineTask::from_payload(Some("")), RefineTask::Rewrite);
    assert_eq!(RefineTask::from_payload(Some("bogus")), RefineTask::Rewrite);
    assert_eq!(
        RefineTask::from_payload(Some("  Magic_Prompt ")),
        RefineTask::MagicPrompt
    );
    assert_eq!(
        RefineTask::from_payload(Some("IMAGE_CAPTION")),
        RefineTask::ImageCaption
    );
    assert_eq!(
        RefineTask::from_payload(Some("image_describe")),
        RefineTask::ImageDescribe
    );

    // is_vision: the two image tasks are reference-image driven; the text tasks are not.
    assert!(!RefineTask::Rewrite.is_vision());
    assert!(!RefineTask::MagicPrompt.is_vision());
    assert!(RefineTask::ImageCaption.is_vision());
    assert!(RefineTask::ImageDescribe.is_vision());

    // is_caption: magic-prompt AND image_caption emit JSON captions; describe (vision, prose) does not.
    assert!(RefineTask::MagicPrompt.is_caption());
    assert!(RefineTask::ImageCaption.is_caption());
    assert!(!RefineTask::ImageDescribe.is_caption());
    assert!(!RefineTask::Rewrite.is_caption());

    // Temperature: only the free-text rewrite samples warm (0.7); everything else is cool (0.4).
    assert_eq!(RefineTask::Rewrite.temperature(), 0.7);
    for task in [
        RefineTask::MagicPrompt,
        RefineTask::ImageCaption,
        RefineTask::ImageDescribe,
    ] {
        assert_eq!(task.temperature(), 0.4, "{task:?} should sample cool");
    }

    // Messages match the pre-refactor ladder exactly.
    assert_eq!(RefineTask::ImageCaption.work_message(), "Captioning image…");
    assert_eq!(
        RefineTask::ImageDescribe.work_message(),
        "Describing image…"
    );
    assert_eq!(
        RefineTask::MagicPrompt.work_message(),
        "Expanding to a caption…"
    );
    assert_eq!(RefineTask::Rewrite.work_message(), "Refining prompt…");
    assert_eq!(RefineTask::MagicPrompt.done_message(), "Caption ready.");
    assert_eq!(RefineTask::ImageCaption.done_message(), "Caption ready.");
    assert_eq!(
        RefineTask::ImageDescribe.done_message(),
        "Description ready."
    );
    assert_eq!(RefineTask::Rewrite.done_message(), "Prompt refined.");
}

/// sc-24029. The per-shot film refine is dispatched with no `task` field
/// (`film_planner.rs`), so it classifies as `Rewrite` and takes that task's budget — while the
/// prompt it must produce is bounded by `sceneworks_core::MAX_PROMPT_CHARS`. A budget that
/// cannot reach the contract truncates the rewrite mid-sentence and still reports success,
/// which is how a 3026/4000-char film shot prompt shipped. Asserted as the CONTRACT (budget ×
/// chars-per-token ≥ the char cap) rather than as a literal, so shrinking the budget or raising
/// the cap fails here rather than in a render.
#[test]
fn the_rewrite_budget_covers_the_refined_prompt_char_contract() {
    // ~2.83 chars/token, from the JSON-caption path this file already states (4096 ≈ ~11.6k
    // chars). This is a deliberately CONSERVATIVE FLOOR, not the rate this task's prose
    // actually achieves: sc-24029 measured the refiner's shot rewrites at ~5.7 chars/token
    // (1584-2922 chars from a 512-token budget). The floor is used here so the assertion holds
    // even for the densest output the task can emit; the budget clears both figures.
    const CHARS_PER_TOKEN: f64 = 11_600.0 / 4096.0;

    let budget = f64::from(DEFAULT_REFINE_MAX_NEW_TOKENS);
    let reachable_chars = budget * CHARS_PER_TOKEN;
    let required = sceneworks_core::MAX_PROMPT_CHARS;

    assert!(
        reachable_chars >= required as f64,
        "the Rewrite budget of {DEFAULT_REFINE_MAX_NEW_TOKENS} tokens reaches only \
             {reachable_chars:.0} chars at {CHARS_PER_TOKEN:.2} chars/token, short of the \
             {required}-char refined-prompt contract; a film shot rewrite would stop on MaxTokens \
             mid-sentence instead of on EOS",
    );
}

#[test]
fn resolve_max_new_tokens_defaults_and_override() {
    // Caption tasks (magic-prompt + image_caption) get the larger default so the JSON closes
    // (sc-8210: 2048 truncated rich captions mid-`elements`).
    let obj = |value: serde_json::Value| value.as_object().unwrap().clone();
    for task in [RefineTask::MagicPrompt, RefineTask::ImageCaption] {
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({})), task, None),
            4096
        );
    }
    // A whole shot plan (sc-22713) takes the same generous ceiling.
    assert_eq!(
        resolve_max_new_tokens(&obj(serde_json::json!({})), RefineTask::FilmPlan, None),
        4096
    );
    // The prose-describe task (epic 8203) gets the in-between default.
    assert_eq!(
        resolve_max_new_tokens(&obj(serde_json::json!({})), RefineTask::ImageDescribe, None),
        1024
    );
    // The free-text rewrite carries the per-shot film refine, whose output contract is
    // `MAX_PROMPT_CHARS` (sc-24029: 512 tokens cut a shot prompt off mid-sentence at 3026
    // chars, because generation ended on `MaxTokens` rather than on EOS).
    assert_eq!(
        resolve_max_new_tokens(&obj(serde_json::json!({})), RefineTask::Rewrite, None),
        1536
    );
    // An explicit positive override wins for any task.
    for task in [
        RefineTask::MagicPrompt,
        RefineTask::ImageDescribe,
        RefineTask::FilmPlan,
    ] {
        assert_eq!(
            resolve_max_new_tokens(
                &obj(serde_json::json!({ "maxNewTokens": 6000 })),
                task,
                None
            ),
            6000
        );
    }
    // Zero / invalid overrides fall back to the per-task default.
    assert_eq!(
        resolve_max_new_tokens(
            &obj(serde_json::json!({ "maxNewTokens": 0 })),
            RefineTask::MagicPrompt,
            None
        ),
        4096
    );
    assert_eq!(
        resolve_max_new_tokens(
            &obj(serde_json::json!({ "maxNewTokens": "nope" })),
            RefineTask::Rewrite,
            None
        ),
        1536
    );
}

#[test]
fn film_plan_task_is_classified_json_constrained_and_carries_its_own_contract() {
    // Selected by the discriminator, case-insensitively, like every other task.
    assert_eq!(
        RefineTask::from_payload(Some(" Film_Plan ")),
        RefineTask::FilmPlan
    );
    assert_eq!(
        RefineTask::from_payload(Some("film_plan")),
        RefineTask::FilmPlan
    );
    // It is text-in/JSON-out: no reference image, not an Ideogram caption, but grammar-masked.
    assert!(!RefineTask::FilmPlan.is_vision());
    assert!(!RefineTask::FilmPlan.is_caption());
    assert!(RefineTask::FilmPlan.emits_json());
    assert!(RefineTask::MagicPrompt.emits_json());
    assert!(RefineTask::ImageCaption.emits_json());
    assert!(!RefineTask::Rewrite.emits_json());
    assert!(!RefineTask::ImageDescribe.emits_json());
    assert_eq!(RefineTask::FilmPlan.temperature(), 0.3);
    assert_eq!(RefineTask::FilmPlan.work_message(), "Planning shots…");
    assert_eq!(RefineTask::FilmPlan.done_message(), "Shot plan ready.");

    // The asset parses into a system block and the user turn is the caller's request verbatim —
    // the film-specific content is composed caller-side, so this asset must not restate it.
    let (system, user) = build_film_plan_messages("# Brief\n\nTitle: Courier");
    assert_eq!(user, "# Brief\n\nTitle: Courier");
    assert!(system.contains("You are a film planner"), "{system}");
    assert!(system.contains("Output ONE JSON object"), "{system}");
    assert!(system.contains("NEVER drop"), "{system}");
    assert!(
        system.contains("fixed menu, not a range"),
        "durations must never be rounded: {system}"
    );
    // sc-23406: a beat's required-role list is NOT a template of one character, one prop and
    // one place. The shared asset used to describe it that way, and the local 8B planner
    // reproduced the template instead of the beat's actual roles; restoring that bullet must
    // red this test, which is the only place the asset's own text is read.
    assert!(
        !system.contains("then the place")
            && !system.contains("the location it happens in")
            && !system.contains("the prop the beat turns on, the location"),
        "the one-of-each-kind template must not reach the planner: {system}"
    );
    // The `[META]` rationale block never reaches the model.
    assert!(!system.contains("[META]"), "{system}");
    assert!(!system.contains("sc-22713"), "{system}");
    // No film-specific content leaked into the shared asset.
    for leaked in ["courier", "parcel", "minimax", "576x320"] {
        assert!(
            !system.to_lowercase().contains(leaked),
            "{leaked:?} leaked into the shared asset: {system}"
        );
    }

    // A JSON reply is isolated by the shared cleaner, exactly as a caption is.
    let cleaned = finalize_refined_output(
        "Here you go:\n```json\n{\"shots\": []}\n```",
        RefineTask::FilmPlan.emits_json(),
        Some("minimax_h3"),
    );
    assert_eq!(cleaned, "{\"shots\": []}");
}

#[test]
fn system_prompt_uses_workflow_medium_and_embeds_guide() {
    let image =
        build_refine_system_prompt(Some("# Z-Image Guide\n\nUse short prompts."), Some("image"));
    assert!(image.contains("generative image model"));
    assert!(image.contains("Z-Image Guide"));
    assert!(image.contains("# Model prompt guide"));

    let video = build_refine_system_prompt(None, Some("video"));
    assert!(video.contains("generative video model"));
    assert!(!video.contains("# Model prompt guide"));

    let audio = build_refine_system_prompt(
        Some("# ACE-Step Guide\n\nRefine only the music description."),
        Some("audio"),
    );
    assert!(audio.contains("generative audio model"));
    assert!(audio.contains("ACE-Step Guide"));
    assert!(
        audio.contains("script or lyrics"),
        "audio rules protect text intended to be spoken or sung"
    );
    assert!(
        !audio.contains("camera body"),
        "audio refinement must not inherit image-only camera guidance"
    );
}

/// sc-17162 — the H3 branch is SELECTED by the target model family, and what it selects is not
/// the generic rewrite.
///
/// The control leg matters more than the positive one here. `guide` on the H3 path is the real
/// `minimax-h3.md`, and that guide already names `integrated_multimodal_description:`,
/// `overall_soundscape:` and `non_diegetic_music:` — so asserting on any of those would pass
/// even with the branch deleted, because the GENERIC path appends the very same guide. The
/// phrase asserted below appears ONLY in the embedded asset, and the non-H3 leg drives the same
/// guide through the generic path to prove that.
#[test]
fn minimax_h3_selects_a_tailored_system_prompt_per_family_partition() {
    let guide = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../apps/web/public/prompt-guides/minimax-h3.md"),
    )
    .expect("the shipped H3 prompt guide is readable");
    // The distinguishing phrase must be the asset's alone. If a future guide edit introduced
    // it, this test would stop discriminating and every assertion below would be vacuous.
    assert!(
        !guide.contains("H3 field block"),
        "the discriminating phrase leaked into the guide — this test no longer proves the \
             branch is selected"
    );

    // BOTH partitions take the tailored prompt: `minimax_h3` (t2va + fl2va) and
    // `minimax_h3_ref` (Ref2VA) share one text encoder and one prompt format.
    let asset = magic_section_from(MINIMAX_H3_REFINE_V1, "SYSTEM");
    for model_id in ["minimax_h3", "minimax_h3_ref"] {
        let system = build_rewrite_system_prompt(Some(&guide), Some("video"), Some(model_id));
        assert!(
            system.contains("H3 field block"),
            "{model_id} must take the tailored H3 rewrite"
        );
        // The Context-IR field structure (story activity 18694) is what the asset instructs.
        for field in [
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        ] {
            assert!(system.contains(field), "{model_id} instructs {field}");
            // …and the ASSET is what instructs them. The guide names all three too, so a
            // combined-prompt assertion alone would survive deleting them from the asset.
            assert!(
                asset.contains(field),
                "the embedded asset itself must instruct {field}"
            );
        }
        // Asserted on a phrase the ASSET owns, not on `[Shot 1] At 00:00.000` — the guide
        // appended below contains that literal in its own worked example, so pinning it would
        // have gone green with the asset's shot instruction deleted.
        assert!(
            system.contains("Continue this line with the framing, the angle"),
            "{model_id} instructs the timed-shot labels, the one thing prose cannot express"
        );
        // The generic medium rules are REPLACED, not merged — the H3 asset owns the whole
        // instruction, so a rule from the generic path cannot contradict it.
        assert!(
            !system.contains("You are a prompt rewriter for a generative video model."),
            "{model_id} must not also carry the generic rewrite rules"
        );
        // The model's own guide is still attached, under the same heading the generic path uses.
        assert!(system.contains("# Model prompt guide"));
        assert!(system.contains("# MiniMax-H3 Prompt Guide"));
    }

    // Control: another video model, the SAME guide text, keeps the generic rules and never sees
    // the tailored asset. An absent/blank id is the same generic path.
    for other in [Some("ltx_2_3"), Some("bernini"), Some("  "), None] {
        let system = build_rewrite_system_prompt(Some(&guide), Some("video"), other);
        assert!(
            !system.contains("H3 field block"),
            "{other:?} must not take the H3 branch"
        );
        assert!(
            system.contains("You are a prompt rewriter for a generative video model."),
            "{other:?} keeps the generic rewrite rules"
        );
    }

    // The family predicate itself, at the boundary the two partitions sit on.
    assert!(refines_for_minimax_h3(Some("minimax_h3")));
    assert!(refines_for_minimax_h3(Some(" minimax_h3_ref ")));
    assert!(!refines_for_minimax_h3(Some("ltx_2_3")));
    assert!(!refines_for_minimax_h3(Some("")));
    assert!(!refines_for_minimax_h3(None));
}

/// sc-17162 review — the asset must carry no fill-in-the-blank SCAFFOLD.
///
/// The post-filter strips the seven untrained markers and nothing else, so any other literal in
/// the asset that reads as a template is a string a small refiner checkpoint can echo straight
/// into the suggestion the user is offered. The first draft had exactly that: angle-bracketed
/// slots (`<framing, angle, …>`) and a timestamp with a letter standing in for a digit
/// (`00:0X.000`). Both would have shipped as literal placeholder text in a refined prompt.
///
/// The rule this pins is "instructions, not blanks". Real OUTPUT tokens — the field names with
/// their colons, `[Shot 1] At 00:00.000` — are not scaffold: the model is supposed to write
/// those verbatim, and shot one always opens at zero.
#[test]
fn minimax_h3_asset_carries_no_fill_in_the_blank_scaffold() {
    let asset = magic_section_from(MINIMAX_H3_REFINE_V1, "SYSTEM");
    assert!(
        !asset.is_empty(),
        "the asset must parse to a non-empty SYSTEM block"
    );

    // No angle-bracketed slot. Scoped to `<` followed by a letter or space so the marker ban's
    // own `<d>`/`<|…|>` citations in the rules below are not what this trips on — those are
    // quoted BANS, and a test that could not tell them apart would forbid stating the ban.
    for (index, byte) in asset.bytes().enumerate() {
        if byte != b'<' {
            continue;
        }
        let rest = &asset[index + 1..];
        let cited_marker = rest.starts_with('|') || rest.starts_with("d>") || rest.starts_with('/');
        assert!(
            cited_marker,
            "angle-bracketed scaffold at byte {index}: {}",
            &asset[index..asset.len().min(index + 60)]
        );
    }

    // No timestamp with a non-digit standing in for a digit. Every `MM:SS.mmm`-shaped run in the
    // asset must be all digits — `00:0X.000` is the exact form that was there.
    for (index, _) in asset.match_indices(':') {
        let after = &asset[index + 1..];
        let digits_then_dot: String = after.chars().take(2).collect();
        if !after.chars().nth(2).is_some_and(|c| c == '.') {
            continue;
        }
        assert!(
            digits_then_dot.chars().all(|c| c.is_ascii_digit()),
            "timestamp with a placeholder digit near byte {index}: {}",
            &asset[index.saturating_sub(12)..asset.len().min(index + 12)]
        );
    }

    // And it says so, so a future edit is told the rule rather than only caught by it.
    assert!(
        asset.contains("never\nleave a blank, a placeholder or a bracketed stand-in"),
        "the asset must forbid placeholders explicitly, not only avoid them"
    );
    assert!(
        asset.contains("A timestamp is digits only"),
        "the asset must forbid a stand-in digit explicitly"
    );
}

/// sc-17162 (from sc-17143's finding) — a marker-baited reply survives NOTHING. The refiner is
/// handed a prompt guide that quotes `<d>[English] …</d>` verbatim, so it can plausibly echo the
/// syntax back; the system prompt asks it not to, and this is the enforcement that does not
/// depend on the model complying.
#[test]
fn minimax_h3_refinement_never_emits_the_untrained_markers() {
    // Exactly the seven from the tokenizer config, and nothing invented here.
    assert_eq!(MINIMAX_H3_UNTRAINED_MARKERS.len(), 7);

    // A reply that abuses every marker, in a wrapping code fence with a reasoning block in
    // front of it — i.e. it goes through the SAME cleanup a real reply does, then the ban.
    let baited = "<think>The guide showed dialogue markers, I'll use them.</think>\n\
             ```\n\
             integrated_multimodal_description: A woman in a red coat turns to the camera.\n\
             overall_soundscape: rain on tarmac; <d>[English] Not tonight.</d> spoken quietly.\n\
             <|caption_start|>a closing caption<|caption_end|> <|cutoff|>\n\
             non_diegetic_music: <|lyrics_start|>hummed melody<|lyrics_end|> low strings.\n\
             ```";
    // Driven through the SAME function the job body calls, with the same arguments — so this
    // also pins that the ban is reached for an H3 target, not merely that the filter works.
    let refined = finalize_refined_output(baited, false, Some("minimax_h3"));
    for marker in MINIMAX_H3_UNTRAINED_MARKERS {
        assert!(
            first_ci(&refined, marker).is_none(),
            "{marker} survived the H3 post-filter: {refined}"
        );
    }
    assert_eq!(
        finalize_refined_output(baited, false, Some("minimax_h3_ref")),
        refined,
        "both H3 partitions take the ban"
    );
    // Control leg: the ban is KEYED on the model, so the same reply for a non-H3 target is
    // cleaned exactly as before this story — otherwise the test would pass with the key removed.
    let untargeted = finalize_refined_output(baited, false, Some("ltx_2_3"));
    assert_eq!(untargeted, clean_refine_output(baited));
    assert!(
        first_ci(&untargeted, "<d>").is_some(),
        "the non-H3 path must be unchanged by this story"
    );
    // Case variants are matched too — the markers are ASCII and the model may shout them.
    let shouted = strip_untrained_markers("a line <D>spoken</D> <|CUTOFF|> here");
    for marker in MINIMAX_H3_UNTRAINED_MARKERS {
        assert!(
            first_ci(&shouted, marker).is_none(),
            "{marker} survived in upper case: {shouted}"
        );
    }

    // The words around the markers survive: this is a strip, not a truncation, and the user's
    // dialogue text is the whole point of the field it sits in.
    assert!(refined.contains("Not tonight."));
    assert!(refined.contains("hummed melody"));
    assert!(refined.contains("A woman in a red coat turns to the camera."));
    // The removal cannot glue two words into one.
    assert_eq!(strip_untrained_markers("word<d>word"), "word word");
    // A reply that is ONLY markers cleans to empty, which is the pipeline's existing
    // empty-output rejection rather than a new failure mode.
    assert!(strip_untrained_markers(" <d> </d> <|cutoff|> ").is_empty());
    // Text with no marker is returned byte for byte, indentation and blank lines intact.
    let untouched = "  a prompt\n\n  with layout  ";
    assert_eq!(strip_untrained_markers(untouched), untouched);

    // And the ban is wired to the same family predicate as the system-prompt selection, so a
    // model can never get the tailored instruction without the enforcement (or vice versa).
    for model_id in ["minimax_h3", "minimax_h3_ref"] {
        assert!(refines_for_minimax_h3(Some(model_id)));
    }
    assert!(!refines_for_minimax_h3(Some("ltx_2_3")));
}

/// sc-24023. The marker ban must leave the engine's OWN reference labels alone.
///
/// `<Picture N>`, `<Audio N>` and `<Video N>` are what the H3 text encoder itself prefixes each
/// supplied reference with, and from sc-24023 the film compiler writes sentences that bind a
/// role to one of them. They look exactly like the seven dead markers — angle-bracketed
/// literals in the prompt — and a filter that ate them would silently unbind every reference
/// shot while leaving a prompt that still reads correctly to a human.
///
/// The ban is a fixed seven-literal list, so this passes as written; the test exists because
/// the next person to widen that list to a pattern has to see it go red.
#[test]
fn the_marker_ban_leaves_the_engines_own_reference_labels_alone() {
    let bound = "The courier is the person shown in <Picture 1>. The parcel is the object \
                     shown in <Picture 2>. She speaks with the voice from <Audio 1>, moving like \
                     <Video 1>.";
    // Byte for byte: a prompt made only of reference labels contains no dead marker, so the
    // filter must not touch it at all — not even the whitespace collapse.
    assert_eq!(strip_untrained_markers(bound), bound);

    // And with a dead marker mixed in, the dead one goes and the labels stay. This is the case
    // that separates "the filter ignores these" from "the filter never ran".
    let baited = format!("<d>[English] Delivery.</d> {bound}");
    let stripped = strip_untrained_markers(&baited);
    assert!(
        first_ci(&stripped, "<d>").is_none() && first_ci(&stripped, "</d>").is_none(),
        "the dead dialogue markers must still go: {stripped}"
    );
    for label in ["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Video 1>"] {
        assert!(
            stripped.contains(label),
            "{label} is the engine's own reference label and must survive: {stripped}"
        );
    }
    assert!(stripped.contains("[English] Delivery."), "{stripped}");

    // Through the real reply path for both H3 partitions, which is where a refined film prompt
    // actually meets the ban.
    for model_id in ["minimax_h3", "minimax_h3_ref"] {
        let refined = finalize_refined_output(&baited, false, Some(model_id));
        for label in ["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Video 1>"] {
            assert!(refined.contains(label), "{model_id}: {refined}");
        }
        assert!(first_ci(&refined, "<d>").is_none(), "{model_id}: {refined}");
    }
}

#[test]
fn system_prompt_defaults_to_image_when_workflow_absent_or_unknown() {
    assert!(build_refine_system_prompt(None, None).contains("generative image model"));
    assert!(build_refine_system_prompt(None, Some("anything")).contains("generative image model"));
    // Case-insensitive video match (parity with Python `.lower()`).
    assert!(build_refine_system_prompt(None, Some(" VIDEO ")).contains("generative video model"));
}

#[test]
fn refine_rules_preserve_photographic_specs_and_strip_only_empty_praise() {
    // sc-13881 (epic 13879): the refiner used to trim the whole trailing comma-list for concision,
    // taking camera/lens/film-stock specs (meaningful to Krea) along with the empty praise — a
    // net-negative edit on a photographic prompt. The rewrite itself is LLM-driven, so the
    // deterministic contract is that the system prompt INSTRUCTS both halves: keep the concrete
    // specifications, strip ONLY empty praise. This guards against a future edit dropping either.
    let system = build_refine_system_prompt(None, Some("image"));
    // Keep the concrete photographic specifications.
    assert!(system.contains("camera body"), "keeps camera body spec");
    assert!(system.contains("focal"), "keeps focal length spec");
    assert!(system.contains("film stock"), "keeps film-stock spec");
    assert!(
        system.contains("Kodak Portra"),
        "the film-stock example is named as something to keep, not strip"
    );
    assert!(
        system.contains("do not drop them when trimming"),
        "explicitly forbids dropping the specs"
    );
    // Strip ONLY empty praise, with the removable words named so the model knows what to cut.
    assert!(system.contains("empty praise"), "names the trim target");
    for praise in ["masterpiece", "stunning", "premium", "award-winning"] {
        assert!(
            system.contains(praise),
            "empty-praise example {praise} is listed as removable"
        );
    }
    // The rules apply model-agnostically, whether or not a per-model guide is attached.
    let bare = base_rules("image");
    assert!(bare.contains("camera body"));
    assert!(bare.contains("empty praise"));
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
fn magic_section_extracts_system_and_user_blocks() {
    let system = magic_section("SYSTEM");
    assert!(
        system.contains("structured JSON caption"),
        "system has the contract intro"
    );
    assert!(
        system.contains("compositional_deconstruction"),
        "system names the schema"
    );
    // The [META] block above [SYSTEM] is not leaked into the system body.
    assert!(!system.contains("thinking_mode"));

    let user = magic_section("USER");
    assert!(
        user.contains("{{aspect_ratio}}"),
        "user template has the aspect-ratio placeholder"
    );
    assert!(
        user.contains("{{original_prompt}}"),
        "user template has the prompt placeholder"
    );

    assert_eq!(magic_section("DOES_NOT_EXIST"), "");
}

#[test]
fn build_magic_prompt_messages_substitutes_template() {
    let (system, user) = build_magic_prompt_messages("a red fox in the snow", "16:9");
    assert!(system.contains("OUTPUT CONTRACT"));
    assert!(user.contains("16:9"));
    assert!(user.contains("a red fox in the snow"));
    // Both placeholders are consumed.
    assert!(!user.contains("{{"));
}

#[test]
fn clean_json_output_isolates_the_object() {
    // Leading/trailing prose from a small model is dropped.
    assert_eq!(
        clean_json_output("Here is the caption: {\"a\": 1} — hope that helps!"),
        "{\"a\": 1}"
    );
    // Code fences + a think block are stripped.
    assert_eq!(
        clean_json_output("<think>plan</think>```json\n{\"a\": 1}\n```"),
        "{\"a\": 1}"
    );
    // Already-clean object passes through.
    assert_eq!(clean_json_output("{\"a\": [1, 2]}"), "{\"a\": [1, 2]}");
}

// ── sc-8840 (F-038): per-token progress coalescing ─────────────────────────────────────────

// The coalescing decision: a value that MOVED is posted; a repeat of the last-posted value is
// dropped (a stalled decode between ticks does not re-POST identical progress).
#[test]
fn next_progress_post_drops_repeats_and_emits_movement() {
    // First observation (nothing posted yet) is always emitted.
    assert_eq!(next_progress_post((5, 4096), None), Some((5, 4096)));
    // An unchanged value after it was posted is a redundant repeat → dropped.
    assert_eq!(next_progress_post((5, 4096), Some((5, 4096))), None);
    // Any movement (current advanced) is emitted.
    assert_eq!(
        next_progress_post((6, 4096), Some((5, 4096))),
        Some((6, 4096))
    );
    // A changed total (e.g. a different budget) also counts as movement.
    assert_eq!(
        next_progress_post((6, 2048), Some((6, 4096))),
        Some((6, 2048))
    );
}

// The watch channel is latest-wins and non-blocking (the F-038 core): a fast producer that writes
// thousands of token counts NEVER blocks, the consumer reading on a tick sees only the newest
// value (intermediate ticks coalesced away), and the FINAL value is always readable after the
// producer finishes — so the terminal progress can never be lost.
#[test]
fn watch_channel_coalesces_and_preserves_final_value() {
    let (tx, rx) = tokio::sync::watch::channel::<(u32, u32)>((0, 4096));
    // Simulate the token callback: publish every token. `send` is non-blocking and never drops
    // the LATEST value, so a burst faster than any consumer cannot back-pressure the producer.
    for index in 1..=4096u32 {
        tx.send((index, 4096))
            .expect("receiver alive → send never errors");
    }
    // A consumer reading between bursts sees only the newest value, not each intermediate one.
    assert_eq!(*rx.borrow(), (4096, 4096));

    // The final value survives the producer dropping (generation finished): the terminal snapshot
    // the completion arm posts is always the true last token count.
    drop(tx);
    assert_eq!(*rx.borrow(), (4096, 4096));

    // Feeding the resident value through the coalescing gate emits it once, then drops the repeat
    // — the terminal post fires exactly once and is never lost.
    let final_value = *rx.borrow();
    assert_eq!(next_progress_post(final_value, None), Some((4096, 4096)));
    assert_eq!(next_progress_post(final_value, Some(final_value)), None);
}

// `send` signals a closed consumer (all receivers dropped): the token callback maps this to
// tripping the engine cancel flag (sc-8804, F-003 — generation must not run unheard). Prove the
// error surfaces when the receiver is gone, which is exactly the condition the callback keys on.
#[test]
fn watch_send_errors_when_all_receivers_dropped() {
    let (tx, rx) = tokio::sync::watch::channel::<(u32, u32)>((0, 4096));
    assert!(
        tx.send((1, 4096)).is_ok(),
        "send succeeds while a receiver lives"
    );
    drop(rx);
    assert!(
        tx.send((2, 4096)).is_err(),
        "send must error once every receiver is dropped (the consumer-gone signal)"
    );
}

// ── sc-8105: image_caption task (reference image → Ideogram JSON caption, core_llm vision path) ──

#[test]
fn image_caption_asset_extracts_system_and_user_blocks() {
    // The embedded `ideogram_image_caption_v1.txt` parses with the same bracketed-marker parser as
    // the magic-prompt asset; the `[META]` block must not leak into the system body.
    let system = magic_section_from(IMAGE_CAPTION_V1, "SYSTEM");
    assert!(
        system.contains("vision captioner"),
        "system declares the image-examination role"
    );
    assert!(
        system.contains("style_description"),
        "system names the style block"
    );
    assert!(
        system.contains("compositional_deconstruction"),
        "system names the composition schema"
    );
    assert!(
        system.contains("OBSERVE, DON'T POPULATE"),
        "system carries the observe-don't-populate discipline"
    );
    // The bbox convention is pinned: `[y1, x1, y2, x2]`, 0–1000, and bboxes are KEPT (not stripped).
    assert!(system.contains("[y1, x1, y2, x2]"));
    assert!(system.contains("1000"));
    assert!(
        system.contains("KEEP these bboxes"),
        "system pins that grounded bboxes are kept"
    );
    // sc-8194: the schema requires EXACTLY ONE of `style_description.photo` / `art_style`
    // (both present OR neither present = validation error). The prompt must make that mandatory.
    assert!(
        system.contains("EXACTLY ONE"),
        "system mandates exactly one style discriminator"
    );
    assert!(
        system.contains("never both, never neither"),
        "system forbids emitting both or neither discriminator key"
    );
    assert!(system.contains("photo"));
    assert!(system.contains("art_style"));
    // sc-8197: `color_palette` must be #RRGGBB hex codes (the web schema's `verifyColorPalette`
    // rejects color names). The prompt must require uppercase hex codes.
    assert!(
        system.contains("#RRGGBB"),
        "system requires color_palette as #RRGGBB hex codes (not color names)"
    );
    // sc-8199: the STYLE palette now allows richer palettes (up to 16, STYLE_PALETTE_MAX),
    // restored from the mistaken cap of 5.
    assert!(
        system.contains("up to 16"),
        "system allows the style color_palette up to 16 entries"
    );
    // sc-8199: each ELEMENT may carry its own optional `color_palette` (last key). The element
    // format example must show the optional per-subject palette.
    assert!(
        system.contains("\"color_palette\":[\"#RRGGBB\""),
        "element format example carries an optional per-subject color_palette"
    );
    // sc-8199: the per-element palette is capped at 5 (ELEMENT_PALETTE_MAX); the `MAXIMUM 5`
    // copy now lives in the element guidance subsection.
    assert!(
        system.contains("MAXIMUM 5"),
        "system caps the element color_palette at a maximum of 5 colors"
    );
    // Output is fenced so it survives markdown.
    assert!(system.contains("```json"));
    // The `[META]` thinking flag does not bleed into the system body.
    assert!(!system.contains("thinking_mode"));

    let user = magic_section_from(IMAGE_CAPTION_V1, "USER");
    assert!(
        user.to_lowercase().contains("reference image"),
        "user turn instructs examining the reference image"
    );
    // A missing section yields empty (parser contract).
    assert_eq!(magic_section_from(IMAGE_CAPTION_V1, "DOES_NOT_EXIST"), "");
}

#[test]
fn build_image_caption_messages_returns_system_and_user() {
    let (system, user) = build_image_caption_messages(false);
    assert!(!system.trim().is_empty(), "system block is non-empty");
    assert!(system.contains("compositional_deconstruction"));
    assert!(!user.trim().is_empty(), "user block is non-empty");
    // The builder takes no aspect ratio / prompt placeholders (the image is the input).
    assert!(!user.contains("{{"));
}

// ── sc-8204 (epic 8203): image_describe task (reference image → plain-text PROSE description) ──

#[test]
fn image_describe_asset_extracts_system_and_user_blocks() {
    // The embedded `image_describe_v1.txt` parses with the same bracketed-marker parser as the
    // other prompt assets, and its SYSTEM block pins the PROSE / no-JSON contract.
    let system = magic_section_from(IMAGE_DESCRIBE_V1, "SYSTEM");
    assert!(!system.trim().is_empty(), "system block is non-empty");
    // Plain-text prose, explicitly NOT the structured JSON caption path.
    assert!(
        system.contains("No JSON") || system.contains("no JSON"),
        "system forbids JSON output"
    );
    assert!(
        !system.contains("compositional_deconstruction"),
        "describe path must NOT carry the Ideogram JSON schema"
    );
    // Observe-don't-invent grounding, like the caption path.
    assert!(
        system.to_lowercase().contains("observe"),
        "system pins observe-don't-invent grounding"
    );
    // The `[META]` thinking flag does not bleed into the system body.
    assert!(!system.contains("thinking_mode"));

    let user = magic_section_from(IMAGE_DESCRIBE_V1, "USER");
    assert!(
        user.to_lowercase().contains("reference image"),
        "user turn instructs examining the reference image"
    );
    assert!(
        !user.contains("{{"),
        "no template placeholders (the image is the input)"
    );
}

#[test]
fn build_image_describe_messages_returns_prose_system_and_user() {
    let (system, user) = build_image_describe_messages(DescribeStyle::Prose, false);
    assert!(!system.trim().is_empty(), "system block is non-empty");
    assert!(
        !system.contains("compositional_deconstruction"),
        "describe system prompt is prose, not the JSON caption schema"
    );
    assert!(!user.trim().is_empty(), "user block is non-empty");
    assert!(!user.contains("{{"));
}

// ── sc-8205 (epic 8203): captionStyle="tags" describe variant ──

#[test]
fn describe_style_from_payload_only_tags_selects_tags() {
    assert_eq!(
        DescribeStyle::from_payload(Some("tags")),
        DescribeStyle::Tags
    );
    assert_eq!(
        DescribeStyle::from_payload(Some("TAGS")),
        DescribeStyle::Tags
    );
    assert_eq!(
        DescribeStyle::from_payload(Some(" tags ")),
        DescribeStyle::Tags
    );
    // Default / empty / unknown → prose (defensive).
    assert_eq!(DescribeStyle::from_payload(None), DescribeStyle::Prose);
    assert_eq!(DescribeStyle::from_payload(Some("")), DescribeStyle::Prose);
    assert_eq!(
        DescribeStyle::from_payload(Some("prose")),
        DescribeStyle::Prose
    );
    assert_eq!(
        DescribeStyle::from_payload(Some("nonsense")),
        DescribeStyle::Prose
    );
}

#[test]
fn image_describe_tags_asset_extracts_a_booru_tag_contract() {
    let system = magic_section_from(IMAGE_DESCRIBE_TAGS_V1, "SYSTEM");
    assert!(!system.trim().is_empty(), "tags system block is non-empty");
    assert!(
        system.contains("comma-separated"),
        "tags system pins a comma-separated tag list"
    );
    assert!(
        system.to_lowercase().contains("booru") || system.to_lowercase().contains("danbooru"),
        "tags system names the booru/danbooru convention"
    );
    assert!(
        !system.contains("compositional_deconstruction"),
        "tags path must NOT carry the JSON caption schema"
    );
    assert!(
        system.to_lowercase().contains("observe"),
        "tags system pins observe-don't-invent grounding"
    );
    assert!(!system.contains("thinking_mode"));
}

#[test]
fn build_image_describe_messages_selects_the_style_asset() {
    let (prose_sys, _) = build_image_describe_messages(DescribeStyle::Prose, false);
    let (tags_sys, tags_user) = build_image_describe_messages(DescribeStyle::Tags, false);
    assert_ne!(
        prose_sys, tags_sys,
        "prose and tags select DIFFERENT system prompts"
    );
    assert!(
        tags_sys.contains("comma-separated"),
        "tags style loads the booru tag-list system prompt"
    );
    assert!(
        tags_user.to_lowercase().contains("tag"),
        "tags user turn instructs tagging"
    );
}

fn builtin_caption_style(model_id: &str) -> Option<String> {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parse");
    manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"].as_str() == Some(model_id))
        .unwrap_or_else(|| panic!("{model_id} present in builtin.models.jsonc"))["captionStyle"]
        .as_str()
        .map(str::to_owned)
}

#[test]
fn builtin_booru_model_declarations_select_the_tags_asset() {
    for model_id in [
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        let declared_style = builtin_caption_style(model_id);
        let style = DescribeStyle::from_payload(declared_style.as_deref());
        let (system, user) = build_image_describe_messages(style, false);
        assert_eq!(style, DescribeStyle::Tags, "{model_id} selects tags");
        assert!(
            system.contains("comma-separated"),
            "{model_id} loads the booru tag-list system prompt"
        );
        assert!(
            user.to_lowercase().contains("tag"),
            "{model_id} loads the tagging user prompt"
        );
    }

    let prose_entry = serde_json::json!({ "captionStyle": "prose" });
    for declared_style in [
        builtin_caption_style("sdxl").as_deref(),
        prose_entry["captionStyle"].as_str(),
    ] {
        let style = DescribeStyle::from_payload(declared_style);
        let (system, _) = build_image_describe_messages(style, false);
        assert_eq!(style, DescribeStyle::Prose);
        assert!(
            system.contains("natural-language description"),
            "absent and explicit prose select the prose asset"
        );
        assert!(!system.contains("comma-separated booru"));
    }
}

// ── sc-8595 (epic 8588): multi-image "mood board" synthesis variants ──

#[test]
fn build_image_describe_messages_multi_selects_the_mood_board_asset() {
    // N>1 swaps in the mood-board synthesis asset for BOTH styles; N==1 keeps the single-image asset.
    let (prose_single, _) = build_image_describe_messages(DescribeStyle::Prose, false);
    let (prose_multi, prose_multi_user) = build_image_describe_messages(DescribeStyle::Prose, true);
    assert_ne!(
        prose_single, prose_multi,
        "a mood board selects a DIFFERENT prose system prompt than single-image describe"
    );
    assert!(
        prose_multi.to_lowercase().contains("mood board")
            || prose_multi.to_lowercase().contains("mood-board"),
        "the multi-image prose system prompt frames the task as a mood board"
    );
    assert!(
        prose_multi.to_lowercase().contains("share")
            || prose_multi.to_lowercase().contains("shared"),
        "the mood-board prose prompt instructs synthesizing the SHARED aesthetic"
    );
    // Still prose — never the JSON caption schema.
    assert!(!prose_multi.contains("compositional_deconstruction"));
    assert!(prose_multi_user.to_lowercase().contains("mood board"));

    let (tags_single, _) = build_image_describe_messages(DescribeStyle::Tags, false);
    let (tags_multi, _) = build_image_describe_messages(DescribeStyle::Tags, true);
    assert_ne!(
        tags_single, tags_multi,
        "a mood board selects a DIFFERENT tags system prompt than single-image tagging"
    );
    assert!(
        tags_multi.contains("comma-separated"),
        "the multi-image tags prompt still pins a comma-separated tag list"
    );
    assert!(
        tags_multi.to_lowercase().contains("mood board")
            || tags_multi.to_lowercase().contains("mood-board"),
        "the multi-image tags system prompt frames the task as a mood board"
    );
}

#[test]
fn build_image_caption_messages_multi_selects_the_mood_board_json_asset() {
    // N>1 swaps in the mood-board JSON asset; it STILL carries the Ideogram caption schema so the
    // reply validates as a caption, but grounds the style in the shared look + synthesizes the scene.
    let (single_sys, _) = build_image_caption_messages(false);
    let (multi_sys, multi_user) = build_image_caption_messages(true);
    assert_ne!(
        single_sys, multi_sys,
        "a mood board selects a DIFFERENT caption system prompt than single-image caption"
    );
    assert!(
        multi_sys.contains("compositional_deconstruction"),
        "the mood-board caption prompt still emits the Ideogram JSON schema"
    );
    assert!(
        multi_sys.contains("style_description"),
        "the mood-board caption prompt still requires style_description"
    );
    assert!(
        multi_sys.to_lowercase().contains("mood board")
            || multi_sys.to_lowercase().contains("mood-board"),
        "the mood-board caption prompt frames the task as a mood board"
    );
    assert!(!multi_user.contains("{{"), "no template placeholders");
}

#[test]
fn clean_refine_output_passes_prose_through_unwrapping_a_fence() {
    // The describe path cleans with `clean_refine_output` (prose), NOT `clean_json_output`: a plain
    // paragraph survives verbatim, and an accidental ```text fence is unwrapped without any
    // JSON-object extraction.
    let prose = "A red fox stands in deep snow at golden hour, side-lit, shot on a 50mm lens with \
                     a shallow depth of field; muted whites and warm amber tones, cinematic photograph.";
    assert_eq!(clean_refine_output(prose), prose);
    let fenced = format!("```\n{prose}\n```");
    assert_eq!(clean_refine_output(&fenced), prose);
}

#[test]
fn clean_json_output_unwraps_a_fenced_image_caption_keeping_bboxes() {
    // The image-caption asset emits a ```json fence; the SAME cleanup the magic-prompt path uses
    // unwraps it. Element bboxes survive (they are NOT stripped on the worker side — sc-8105).
    let fenced = "```json\n{\"high_level_description\": \"a red fox in snow\", \
             \"compositional_deconstruction\": {\"background\": \"a snowy forest\", \
             \"elements\": [{\"type\": \"obj\", \"bbox\": [100, 200, 800, 700], \"desc\": \"a red fox\"}]}}\n```";
    let cleaned = clean_json_output(fenced);
    let parsed: Value = serde_json::from_str(&cleaned).expect("cleaned reply is valid JSON");
    assert!(
        sceneworks_core::ideogram_caption::is_caption(&parsed),
        "cleaned fenced reply is a schema-valid Ideogram caption"
    );
    // The element bbox is preserved (worker does not strip image-grounded boxes).
    let bbox = parsed["compositional_deconstruction"]["elements"][0]
        .get("bbox")
        .expect("element keeps its bbox");
    assert_eq!(bbox, &serde_json::json!([100, 200, 800, 700]));
}

#[test]
fn image_caption_malformed_output_fails_is_caption() {
    // A non-caption JSON object (no `compositional_deconstruction`) is rejected — the worker's
    // malformed-output guard, mirroring how the magic-prompt path's non-captions fail downstream.
    let not_a_caption: Value = serde_json::from_str(&clean_json_output(
        "```json\n{\"high_level_description\": \"x\"}\n```",
    ))
    .expect("valid JSON");
    assert!(!sceneworks_core::ideogram_caption::is_caption(
        &not_a_caption
    ));

    // Non-JSON / refusal text isolates nothing parseable as a caption.
    let refusal = clean_json_output("I cannot describe this image.");
    assert!(!serde_json::from_str::<Value>(&refusal)
        .map(|v| sceneworks_core::ideogram_caption::is_caption(&v))
        .unwrap_or(false));
}

#[test]
fn load_caption_image_ref_decodes_png_to_rgb8_image_ref() {
    use gen_core::core_llm::ImageRef;
    // Encode a tiny 2×3 RGB PNG to a temp file, then decode it through the image-caption loader and
    // assert the `ImageRef` dimensions + RGB8 byte count (width·height·3). This exercises the
    // `decode_image_any → to_rgb8 → ImageRef::new` path without weights.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ref.png");
    let mut img = image::RgbImage::new(2, 3);
    for (i, px) in img.pixels_mut().enumerate() {
        *px = image::Rgb([(i * 10) as u8, (i * 20) as u8, (i * 30) as u8]);
    }
    img.save(&path).expect("write png");

    let image_ref: ImageRef = load_caption_image_ref(&path).expect("decode reference image");
    assert_eq!(image_ref.width, 2);
    assert_eq!(image_ref.height, 3);
    assert_eq!(image_ref.pixels.len(), 2 * 3 * 3);
}

#[test]
fn image_caption_reference_path_confined_to_app_managed_root() {
    // Issue 2 (path containment): the image_caption reference path is UNTRUSTED (job payload, over
    // the LAN-remote API boundary), so it must resolve under an app-managed root before it is read.
    // A path INSIDE `data_dir` is accepted; a sibling path OUTSIDE it, and a `..`-traversal escape,
    // are rejected with a clear error — mirroring the JoyCaption `resolve_dataset_item_path` guard.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut settings = crate::test_env::offline_settings();
    settings.data_dir = dir.path().to_path_buf();

    // In-root: accepted (the on-disk file need not exist — confinement is path-structural).
    let inside = dir.path().join("projects").join("ref.png");
    let resolved = normalize_app_managed_model_path(
        &settings,
        &inside.to_string_lossy(),
        "Image caption reference image",
    )
    .expect("an in-root reference path is accepted");
    // sc-9812: confinement canonicalizes the deepest existing ancestor before
    // re-appending the not-yet-created tail, so the resolved path is under the
    // canonical data-dir root (on macOS `/var` -> `/private/var`).
    let canonical_root = dir.path().canonicalize().expect("tempdir canonicalizes");
    assert!(
        resolved.starts_with(&canonical_root),
        "stays under the data dir"
    );

    // Out-of-root absolute path: rejected.
    let outside_dir = tempfile::tempdir().expect("tempdir2");
    let outside = outside_dir.path().join("secret.png");
    let err = normalize_app_managed_model_path(
        &settings,
        &outside.to_string_lossy(),
        "Image caption reference image",
    )
    .expect_err("an out-of-root reference path is rejected");
    assert!(
        err.to_string().contains("Image caption reference image"),
        "{err}"
    );

    // `..` traversal escaping the data dir: rejected.
    let traversal = format!("{}/../escape.png", dir.path().display());
    let err =
        normalize_app_managed_model_path(&settings, &traversal, "Image caption reference image")
            .expect_err("a `..`-traversal reference path is rejected");
    assert!(
        err.to_string().contains("Image caption reference image")
            || err.to_string().contains("Unsafe absolute path"),
        "{err}"
    );
}

#[test]
fn image_caption_request_carries_image_for_generate_but_resolves_json_only() {
    // Assemble the same request the job's blocking task builds for an image_caption run and assert
    // the contract surface: the user turn carries the reference image (so the LOADED provider's
    // vision tower reads it at GENERATE time) and a JSON constraint. CRUCIALLY, resolution must NOT
    // demand vision — `ModelRequirements::from_request` WOULD set `vision:true` (which makes
    // core-llm `select` fail for a Qwen-VL snapshot, since no provider statically advertises both
    // vision and Json), so the worker instead builds requirements from the JSON constraint ALONE.
    // This test pins the worker's resolution-requirements construction (the actual fix), and the
    // contrast against `from_request`.
    use gen_core::core_llm::{
        Constraint, Content, ImageRef, Message, ModelRequirements, Role, Sampling, TextLlmRequest,
    };

    let (system, user) = build_image_caption_messages(false);
    let image = ImageRef::new(2, 2, vec![0u8; 12]).expect("image ref");
    let request = TextLlmRequest {
        messages: vec![
            Message::system(system),
            Message {
                role: Role::User,
                content: vec![Content::Image(image), Content::text(user)],
                thinking: None,
                tool_calls: Vec::new(),
            },
        ],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 2048,
        seed: None,
        constraint: Some(Constraint::Json),
        ..Default::default()
    };

    assert!(request.has_image(), "request carries the reference image");

    // `from_request` over-constrains: it derives `vision:true` from the image block. The worker must
    // NOT use this for the image_caption resolution (it would make `select` return Unsupported for a
    // Qwen-VL snapshot — see the weightless resolution tests below).
    let auto = ModelRequirements::from_request(&request);
    assert!(
        auto.vision,
        "from_request over-constrains by demanding vision (the bug the fix avoids)"
    );

    // The worker's resolution requirements: the request's output constraint ONLY, no vision filter.
    let mut reqs = ModelRequirements::default();
    for constraint in request.constraint.iter().copied() {
        reqs = reqs.with_constraint(constraint);
    }
    assert!(
        !reqs.vision,
        "image_caption resolution must NOT demand vision (it is acquired at LOAD time)"
    );
    assert!(
        reqs.constraints.contains(&Constraint::Json),
        "image_caption resolution still demands JSON-constrained decoding"
    );
}

/// Weightless resolution proof — UPDATED for the sc-8171 engine bump (mlx-llm 6c052ba → 7041411).
/// On the OLD engine, demanding `vision:true` + `[Json]` for a Qwen-VL (`qwen3_5`) snapshot FAILED
/// `select` with `Error::Unsupported`: `mlx-llama`'s STATIC descriptor was `supports_vision:false`
/// (vision flipped on only at LOAD), so no provider statically satisfied both vision and Json. The
/// bump adds the per-snapshot `weightless_vision` probe (mlx-llm provider.rs / core-llm sc-8077): the
/// resolver now recognizes a `qwen3_5`/`qwen3_vl` wrapper as vision-capable from `config.json` ALONE,
/// so vision+Json RESOLVES `mlx-llama` and reaches `load` (which then fails only on the missing
/// weight shards, NOT an `Unsupported` capability gap). This test pins that lifted limitation.
///
/// NB: the PRODUCTION image_caption path is unaffected — it deliberately resolves on the JSON
/// constraint ALONE and acquires vision at load (see the resolution note on the blocking task), so it
/// worked before AND after the bump. This test just proves the resolver gained resolution-time vision
/// recognition. macOS-only (the `mlx-llm` providers link there); no weights — reads only `config.json`.
#[cfg(target_os = "macos")]
#[test]
fn vision_plus_json_resolution_now_resolves_for_qwen_vl_snapshot() {
    use gen_core::core_llm::{Constraint, LoadSpec, ModelRequirements};

    let dir = tempfile::tempdir().expect("tempdir");
    write_qwen_vl_config(dir.path());

    // Demand vision + Json (the `from_request` shape for an image+Json request).
    let reqs = ModelRequirements::default()
        .with_vision()
        .with_constraint(Constraint::Json);
    let err = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: dir.path().to_string_lossy().into_owned(),
            ..Default::default()
        },
        &reqs,
    )
    .err()
    .expect("no weight shards on disk, so the LOAD fails after the provider is selected");
    let msg = err.to_string();
    // Post-bump: a provider WAS selected (the weightless_vision probe satisfied the vision gate), so
    // the error is a weight-load failure — NOT the pre-bump `Unsupported` capability-resolution gap.
    assert!(
        !msg.contains("no linked provider meets")
            && !msg.contains("no registered provider can serve")
            && !msg.contains("unsupported"),
        "vision+Json must now resolve a provider for a Qwen-VL snapshot (then fail on missing \
             weights) thanks to the weightless_vision probe; got: {msg}"
    );
}

/// Weightless resolution proof (the Issue-1 FIX). The SAME Qwen-VL snapshot, resolved with the
/// worker's actual image_caption requirements — the JSON constraint ALONE, no vision filter —
/// passes core-llm's `select` (it picks the text+Json `mlx-llama`, which `can_load`s a Qwen-VL
/// wrapper). Resolution therefore reaches `load`; with no weight shards on disk the load fails, but
/// with a `Load`/backend error (missing tensors), NOT the `Unsupported` resolution error above —
/// proving the provider WAS selected. That is exactly what the fixed worker path does.
#[cfg(target_os = "macos")]
#[test]
fn json_only_resolution_selects_a_provider_for_qwen_vl_snapshot() {
    use gen_core::core_llm::{Constraint, LoadSpec, ModelRequirements};

    let dir = tempfile::tempdir().expect("tempdir");
    write_qwen_vl_config(dir.path());

    // The worker's image_caption resolution requirements: JSON constraint only, no vision filter.
    let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
    let spec = LoadSpec {
        source: dir.path().to_string_lossy().into_owned(),
        ..Default::default()
    };
    let err = crate::inference_runtime::load_for_model_with(&spec, &reqs)
        .err()
        .expect("no weights on disk, so the LOAD fails after the provider is selected");
    let msg = err.to_string();
    // The provider WAS selected (resolution passed): the error is a weight-load failure, not the
    // `Unsupported` capability-resolution error. So `select`/`meets` admitted a provider for the
    // Qwen-VL snapshot under the JSON-only requirements — the fix routes here.
    assert!(
        !msg.contains("no linked provider meets")
            && !msg.contains("no registered provider can serve"),
        "JSON-only requirements must resolve a provider (then fail on missing weights), got: {msg}"
    );

    // Strengthen the proof: the SPECIFIC provider admitted under JSON-only requirements for this
    // Qwen-VL snapshot must be `mlx-llama` — the one whose `generate` reads a `Content::Image` once
    // its Qwen-VL tower is loaded. `load_for_model_with` only returns the loaded provider (impossible
    // here without weights), so replicate core-llm's public selection predicate over the live
    // registry: architecture `can_load` AND the capability filter (vision/constraints) the resolver
    // applies. A future regression that resolved the snapshot to a DIFFERENT Json provider would
    // surface here, not slip past the generic "some provider resolved" check above.
    let viable: Vec<String> = crate::inference_runtime::textllms()
        .filter(|r| (r.can_load)(&spec))
        .filter(|r| {
            let caps = (r.descriptor)().capabilities;
            (!reqs.vision || caps.supports_vision)
                && reqs.constraints.iter().all(|c| caps.supports_constraint(c))
        })
        .map(|r| (r.descriptor)().id)
        .collect();
    assert!(
            viable.iter().any(|id| id == "mlx-llama"),
            "JSON-only resolution must admit `mlx-llama` (the image-capable Json provider) for a Qwen-VL \
             snapshot; viable providers were: {viable:?}"
        );
}

/// A minimal Qwen3.6 (`qwen3_5`) VLM-wrapper `config.json` — enough for the weightless `can_load`
/// probes (`Architecture::from_config` dispatches on `model_type`/`architectures`; the
/// `vision_config` marks it a VLM wrapper). No weight shards / tokenizer: resolution reads only this
/// file, and a subsequent `load` is expected to fail on the missing tensors.
#[cfg(target_os = "macos")]
fn write_qwen_vl_config(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        br#"{"architectures":["Qwen3_5ForConditionalGeneration"],
                "model_type":"qwen3_5",
                "text_config":{"model_type":"qwen3_5_text"},
                "vision_config":{"model_type":"qwen3_5","depth":27}}"#,
    )
    .expect("write qwen-vl config.json");
}

/// A minimal **`qwen3_vl`** VLM-wrapper `config.json` — the architecture of the chosen default
/// image_caption model (`huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated`). `model_type`
/// `qwen3_vl` (+ `text_config.model_type` `qwen3_vl_text`) is the family the sc-8171 engine bump
/// (mlx-llm 6c052ba → 7041411) teaches the inventory to recognize: before the bump `qwen3_vl`
/// silently fell through to a text-only `Architecture::Qwen3` (no vision tower); after it,
/// `Architecture::Qwen3Vl` + the per-snapshot `weightless_vision` probe make a `qwen3_vl` snapshot
/// resolve vision-capable. No weight shards / tokenizer: resolution reads only this file.
#[cfg(target_os = "macos")]
fn write_qwen3_vl_config(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        br#"{"architectures":["Qwen3VLForConditionalGeneration"],
                "model_type":"qwen3_vl",
                "text_config":{"model_type":"qwen3_vl_text"},
                "vision_config":{"model_type":"qwen3_vl","depth":27}}"#,
    )
    .expect("write qwen3_vl config.json");
}

/// Weightless resolution proof for the **`qwen3_vl`** default model (sc-8171, the whole point of
/// the engine bump). The SAME shape as `json_only_resolution_selects_a_provider_for_qwen_vl_snapshot`
/// but on a `qwen3_vl` snapshot: under the worker's image_caption requirements (JSON constraint
/// ALONE), core-llm's `select` admits `mlx-llama` for the `qwen3_vl` wrapper. CRUCIALLY this test
/// ALSO proves the bump's purpose — that the SAME `qwen3_vl` snapshot resolves to a *vision-capable*
/// provider — by replicating core-llm's per-snapshot vision predicate (`weightless_vision` probe →
/// `serves_vision`) over the live registry. On the OLD engine (mlx-llm 6c052ba) a `qwen3_vl`
/// `config.json` parsed to a text-only `Architecture::Qwen3` with NO `weightless_vision` probe, so
/// this vision assertion would fail; on the bumped engine (7041411) it passes. macOS-only (the
/// `mlx-llm` providers link there); no weights — resolution reads only `config.json`.
#[cfg(target_os = "macos")]
#[test]
fn json_only_resolution_selects_vision_capable_provider_for_qwen3_vl_snapshot() {
    use gen_core::core_llm::{Constraint, LoadSpec, ModelRequirements};

    let dir = tempfile::tempdir().expect("tempdir");
    write_qwen3_vl_config(dir.path());
    let spec = LoadSpec {
        source: dir.path().to_string_lossy().into_owned(),
        ..Default::default()
    };

    // (1) Selection: the worker's image_caption resolution requirements (JSON constraint only)
    // resolve a provider for the qwen3_vl snapshot — the LOAD then fails on missing weights, NOT
    // the `Unsupported` capability-resolution error (proving a provider WAS selected).
    let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
    let err = crate::inference_runtime::load_for_model_with(&spec, &reqs)
        .err()
        .expect("no weights on disk, so the LOAD fails after the provider is selected");
    let msg = err.to_string();
    assert!(
        !msg.contains("no linked provider meets")
            && !msg.contains("no registered provider can serve"),
        "JSON-only requirements must resolve a provider for a qwen3_vl snapshot (then fail on \
             missing weights), got: {msg}"
    );

    // (2) The selected Json provider for this qwen3_vl snapshot is `mlx-llama` — the one whose
    // loaded Qwen3-VL tower reads the `Content::Image` at generate time.
    let json_viable: Vec<String> = crate::inference_runtime::textllms()
        .filter(|r| (r.can_load)(&spec))
        .filter(|r| {
            let caps = (r.descriptor)().capabilities;
            reqs.constraints.iter().all(|c| caps.supports_constraint(c))
        })
        .map(|r| (r.descriptor)().id)
        .collect();
    assert!(
            json_viable.iter().any(|id| id == "mlx-llama"),
            "JSON-only resolution must admit `mlx-llama` for a qwen3_vl snapshot; viable: {json_viable:?}"
        );

    // (3) THE BUMP'S PURPOSE: the same qwen3_vl snapshot is recognized as vision-capable. Replicate
    // core-llm's per-snapshot vision predicate (the `weightless_vision` probe added in mlx-llm
    // 7041411 / sc-8077, OR a statically-vision descriptor) over the live registry. At least one
    // provider that `can_load` this snapshot must serve it WITH vision — impossible on the old
    // engine, where `qwen3_vl` was a text-only `Qwen3` with no probe.
    let vision_capable: Vec<String> = crate::inference_runtime::textllms()
        .filter(|r| (r.can_load)(&spec))
        .filter(|r| {
            r.weightless_vision.map(|p| p(&spec)).unwrap_or(false)
                || (r.descriptor)().capabilities.supports_vision
        })
        .map(|r| (r.descriptor)().id)
        .collect();
    assert!(
        vision_capable.iter().any(|id| id == "mlx-llama"),
        "the sc-8171 bump must make a qwen3_vl snapshot resolve to a vision-capable `mlx-llama` \
             provider (weightless_vision probe); vision-capable providers were: {vision_capable:?}"
    );

    // (4) End-to-end resolver proof of the bump: demanding vision + Json at RESOLUTION now RESOLVES
    // a provider for the qwen3_vl snapshot (the `weightless_vision` probe satisfies the vision gate
    // from `config.json` alone), reaching `load` — which fails only on the missing weight shards, NOT
    // the pre-bump `Unsupported` capability gap. (Production still resolves on Json alone and acquires
    // vision at load; this assertion proves the resolver itself gained resolution-time vision.)
    let vision_json = ModelRequirements::default()
        .with_vision()
        .with_constraint(Constraint::Json);
    let vj_msg = crate::inference_runtime::load_for_model_with(&spec, &vision_json)
        .err()
        .expect("no weight shards on disk, so the LOAD fails after the provider is selected")
        .to_string();
    assert!(
        !vj_msg.contains("no linked provider meets")
            && !vj_msg.contains("no registered provider can serve")
            && !vj_msg.contains("unsupported"),
        "vision+Json must now resolve a provider for a qwen3_vl snapshot (then fail on missing \
             weights) thanks to the weightless_vision probe; got: {vj_msg}"
    );
}

/// THE sc-8204 RISK GATE (epic 8203). The prose `image_describe` task drives the same Qwen-VL vision
/// path but emits NO JSON, so it sets NO output constraint — its resolution requirements are EMPTY
/// (`ModelRequirements::default()`). This proves that with no constraint AND no vision filter, the
/// resolver STILL admits `mlx-llama` for a `qwen3_vl` snapshot on architecture `can_load` alone: it
/// is the only provider that `can_load`s a Qwen-VL wrapper (`mlx-joycaption` only loads LLaVA), so an
/// unconstrained `select` cannot pick a wrong provider. Same weightless shape as the JSON-only tests:
/// resolution reaches `load`, which fails only on the missing weight shards — NOT an `Unsupported`
/// capability gap. macOS-only (the `mlx-llm` providers link there); no weights — reads only
/// `config.json`.
#[cfg(target_os = "macos")]
#[test]
fn unconstrained_resolution_selects_a_provider_for_qwen3_vl_snapshot() {
    use gen_core::core_llm::{LoadSpec, ModelRequirements};

    let dir = tempfile::tempdir().expect("tempdir");
    write_qwen3_vl_config(dir.path());
    let spec = LoadSpec {
        source: dir.path().to_string_lossy().into_owned(),
        ..Default::default()
    };

    // The worker's image_describe resolution requirements: no constraint, no vision filter.
    let reqs = ModelRequirements::default();
    let err = crate::inference_runtime::load_for_model_with(&spec, &reqs)
        .err()
        .expect("no weights on disk, so the LOAD fails after the provider is selected");
    let msg = err.to_string();
    assert!(
        !msg.contains("no linked provider meets")
            && !msg.contains("no registered provider can serve"),
        "unconstrained requirements must resolve a provider for a qwen3_vl snapshot (then fail on \
             missing weights), got: {msg}"
    );

    // The provider admitted under EMPTY requirements for this qwen3_vl snapshot must be `mlx-llama`
    // — the only one that `can_load`s a Qwen-VL wrapper. With no constraint/vision filter the viable
    // predicate reduces to `can_load` alone, so a regression that admitted a non-image provider (or
    // failed to admit `mlx-llama`) surfaces here.
    let viable: Vec<String> = crate::inference_runtime::textllms()
        .filter(|r| (r.can_load)(&spec))
        .map(|r| (r.descriptor)().id)
        .collect();
    assert!(
            viable.iter().any(|id| id == "mlx-llama"),
            "unconstrained resolution must admit `mlx-llama` for a qwen3_vl snapshot; viable: {viable:?}"
        );
    // And the admitted provider is vision-capable at load (the weightless_vision probe), so the prose
    // describe path examines the `Content::Image` exactly like the caption path does.
    let vision_capable = crate::inference_runtime::textllms()
        .filter(|r| (r.can_load)(&spec))
        .filter(|r| (r.descriptor)().id == "mlx-llama")
        .any(|r| {
            r.weightless_vision.map(|p| p(&spec)).unwrap_or(false)
                || (r.descriptor)().capabilities.supports_vision
        });
    assert!(
        vision_capable,
        "the unconstrained-resolved `mlx-llama` must be vision-capable for a qwen3_vl snapshot"
    );
}

/// Real-weight image-caption smoke (sc-8105 → validated under sc-8113): examines a reference image
/// and emits an Ideogram JSON caption through the unified mlx-llm VISION engine —
/// `crate::inference_runtime::load_for_model_with` resolves `mlx-llama` on a Qwen-VL snapshot. The pick is
/// steered by the JSON constraint ALONE (the request DOES carry the image, but resolution must NOT
/// demand vision: no provider statically advertises both vision and Json for a Qwen-VL snapshot, so
/// `mlx-llama` is selected text+Json and flips to vision only at LOAD time — see the resolution note
/// on the blocking task). The reqs are therefore built the SAME way production builds them — the
/// request's output constraint only, NOT `ModelRequirements::from_request` (which would over-derive
/// `vision:true` from the image block and fail `select` with `Error::Unsupported` before any load,
/// per `vision_plus_json_resolution_fails_for_qwen_vl_snapshot`). The cleaned reply must pass
/// `is_caption` with element bboxes kept. `#[ignore]` — the weights live outside CI; run on a Mac
/// with a vision model staged and pointed at by VISION_CAPTION_SNAPSHOT and a reference image at
/// IMAGE_CAPTION_REF:
///   VISION_CAPTION_SNAPSHOT=<snapshot dir> IMAGE_CAPTION_REF=<image path> \
///   cargo test -p sceneworks-worker --lib -- --ignored image_caption_examines_reference --nocapture
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight (sc-8113): needs a vision model snapshot + reference image; set VISION_CAPTION_SNAPSHOT + IMAGE_CAPTION_REF"]
fn image_caption_examines_reference_image() {
    use gen_core::core_llm::{
        Constraint, Content, ImageRef, LoadSpec, Message, ModelRequirements, Role, Sampling,
        StreamEvent, TextLlmRequest,
    };

    let snapshot = std::env::var("VISION_CAPTION_SNAPSHOT")
        .expect("set VISION_CAPTION_SNAPSHOT to a vision model snapshot dir");
    let ref_path = std::env::var("IMAGE_CAPTION_REF")
        .expect("set IMAGE_CAPTION_REF to a reference image path");

    let image = load_caption_image_ref(std::path::Path::new(&ref_path)).expect("decode ref image");
    let (system, user) = build_image_caption_messages(false);
    let request = TextLlmRequest {
        messages: vec![
            Message::system(system),
            Message {
                role: Role::User,
                content: vec![Content::Image(image), Content::text(user)],
                thinking: None,
                tool_calls: Vec::new(),
            },
        ],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 2048,
        seed: None,
        constraint: Some(Constraint::Json),
        ..Default::default()
    };
    let _ = ImageRef::new(1, 1, vec![0u8; 3]); // touch the type so the import is load-bearing in all cfgs
                                               // Build the resolution requirements the SAME way the production image_caption path does — the
                                               // request's output constraint ALONE, no vision filter (NOT `ModelRequirements::from_request`,
                                               // which derives `vision:true` from the image block and would fail `select` on a Qwen-VL snapshot
                                               // before any load). This exercises the real production resolution path the sc-8113 validator runs.
    let mut reqs = ModelRequirements::default();
    for constraint in request.constraint.iter().copied() {
        reqs = reqs.with_constraint(constraint);
    }
    let captioner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: snapshot,
            ..Default::default()
        },
        &reqs,
    )
    .expect("load a vision provider via core-llm model-first JSON-only resolution");

    let mut sink = |_event: StreamEvent| {};
    let output = captioner.generate(&request, &mut sink).expect("generate");
    let json = clean_json_output(&output.text);
    eprintln!("image-caption JSON:\n{json}");

    let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
    assert!(
        sceneworks_core::ideogram_caption::is_caption(&parsed),
        "reply is a schema-valid Ideogram caption"
    );
}

/// Candle (Windows/Linux) twin of `image_caption_examines_reference_image` (sc-8116, epic 8103):
/// the backend-neutral `image_caption` path resolved against the CANDLE Qwen3-VL provider instead of
/// mlx-llm. Same production resolution (output-constraint-only reqs, JSON-only `load_for_model_with`),
/// same `is_caption` assertion with bboxes kept — only the linked provider differs (`candle_llm`).
/// `#[ignore]` — weights live outside CI; run on a Windows/CUDA box (built `--features backend-candle`):
///   VISION_CAPTION_SNAPSHOT=<snapshot dir> IMAGE_CAPTION_REF=<image path> \
///   cargo test -p sceneworks-worker --features backend-candle --lib -- --ignored \
///     image_caption_examines_reference_image_candle --nocapture
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "real-weight (sc-8116): needs a Qwen3-VL snapshot + reference image; set VISION_CAPTION_SNAPSHOT + IMAGE_CAPTION_REF"]
fn image_caption_examines_reference_image_candle() {
    use gen_core::core_llm::{
        Constraint, Content, ImageRef, LoadSpec, Message, ModelRequirements, Role, Sampling,
        StreamEvent, TextLlmRequest,
    };

    let snapshot = std::env::var("VISION_CAPTION_SNAPSHOT")
        .expect("set VISION_CAPTION_SNAPSHOT to a vision model snapshot dir");
    let ref_path = std::env::var("IMAGE_CAPTION_REF")
        .expect("set IMAGE_CAPTION_REF to a reference image path");

    let image = load_caption_image_ref(std::path::Path::new(&ref_path)).expect("decode ref image");
    let (system, user) = build_image_caption_messages(false);
    let request = TextLlmRequest {
        messages: vec![
            Message::system(system),
            Message {
                role: Role::User,
                content: vec![Content::Image(image), Content::text(user)],
                thinking: None,
                tool_calls: Vec::new(),
            },
        ],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 2048,
        seed: None,
        constraint: Some(Constraint::Json),
        ..Default::default()
    };
    let _ = ImageRef::new(1, 1, vec![0u8; 3]); // touch the type so the import is load-bearing
                                               // Production resolution path: request output-constraint ONLY (no vision filter), exactly like the
                                               // mlx twin — `ModelRequirements::from_request` would derive `vision:true` and fail `select`.
    let mut reqs = ModelRequirements::default();
    for constraint in request.constraint.iter().copied() {
        reqs = reqs.with_constraint(constraint);
    }
    let captioner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: snapshot,
            ..Default::default()
        },
        &reqs,
    )
    .expect("load a candle vision provider via core-llm model-first JSON-only resolution");

    let mut sink = |_event: StreamEvent| {};
    let output = captioner.generate(&request, &mut sink).expect("generate");
    let json = clean_json_output(&output.text);
    eprintln!("candle image-caption JSON:\n{json}");

    let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
    assert!(
        sceneworks_core::ideogram_caption::is_caption(&parsed),
        "reply is a schema-valid Ideogram caption"
    );
}

/// Real-weight image-DESCRIBE smoke (sc-8204, sc-12337, epic 8203): the tags sibling of
/// `image_caption_examines_reference_image`. Examines a reference image and emits a PLAIN-TEXT
/// description through the SAME mlx-llm vision engine — but with NO output constraint, so resolution
/// runs on architecture `can_load` ALONE (the sc-8204 risk gate, pinned weightlessly by
/// `unconstrained_resolution_selects_a_provider_for_qwen3_vl_snapshot`). The cleaned reply must be a
/// non-empty booru tag list that is NOT JSON. `#[ignore]` — the weights live outside CI; run on a Mac:
///   VISION_CAPTION_SNAPSHOT=<snapshot dir> IMAGE_CAPTION_REF=<image path> \
///   cargo test -p sceneworks-worker --lib -- --ignored \
///     illustrious_image_describe_emits_tags_for_reference_image --nocapture
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight (sc-8209): needs a vision model snapshot + reference image; set VISION_CAPTION_SNAPSHOT + IMAGE_CAPTION_REF"]
fn illustrious_image_describe_emits_tags_for_reference_image() {
    use gen_core::core_llm::{
        Content, LoadSpec, Message, ModelRequirements, Role, Sampling, StreamEvent, TextLlmRequest,
    };

    let snapshot = std::env::var("VISION_CAPTION_SNAPSHOT")
        .expect("set VISION_CAPTION_SNAPSHOT to a vision model snapshot dir");
    let ref_path = std::env::var("IMAGE_CAPTION_REF")
        .expect("set IMAGE_CAPTION_REF to a reference image path");

    let image = load_caption_image_ref(std::path::Path::new(&ref_path)).expect("decode ref image");
    let declared_style = builtin_caption_style("illustrious_xl_v1");
    let style = DescribeStyle::from_payload(declared_style.as_deref());
    assert_eq!(style, DescribeStyle::Tags, "Illustrious declares tags");
    let (system, user) = build_image_describe_messages(style, false);
    let request = TextLlmRequest {
        messages: vec![
            Message::system(system),
            Message {
                role: Role::User,
                content: vec![Content::Image(image), Content::text(user)],
                thinking: None,
                tool_calls: Vec::new(),
            },
        ],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 1024,
        seed: None,
        // The describe path emits prose — NO JSON constraint.
        constraint: None,
        ..Default::default()
    };
    // Resolution requirements the SAME way the production image_describe path builds them: EMPTY
    // (no constraint, no vision filter). `can_load` alone admits `mlx-llama`, which flips to vision at
    // load — the risk gate this smoke exercises on real weights.
    let reqs = ModelRequirements::default();
    let describer = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: snapshot,
            ..Default::default()
        },
        &reqs,
    )
    .expect("load a vision provider via core-llm model-first unconstrained resolution");

    let mut sink = |_event: StreamEvent| {};
    let output = describer.generate(&request, &mut sink).expect("generate");
    let tags = clean_refine_output(&output.text);
    eprintln!("Illustrious image-describe tags:\n{tags}");

    assert!(!tags.is_empty(), "describe reply is a non-empty tag list");
    assert!(
        serde_json::from_str::<Value>(&tags).is_err(),
        "describe reply is plain tags, not a JSON object"
    );
    assert!(
        tags.contains(','),
        "Illustrious describe reply uses a comma-separated booru tag list: {tags}"
    );
}

/// Real-weight magic-prompt smoke (sc-7158): expands a plain idea into a JSON caption through the
/// unified mlx-llm engine — `crate::inference_runtime::load_for_model` resolves `mlx-llama` on the Anubis
/// snapshot (the JSON constraint steers the model-first pick) — and asserts the cleaned reply parses
/// with the caption's required section. `#[ignore]` — the weights live outside CI; run on a Mac with
/// the model staged in the HF cache:
///   cargo test -p sceneworks-worker --lib -- --ignored magic_prompt_expands
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight: needs the Anubis-Mini-8B prompt-refine model in the HF cache"]
fn magic_prompt_expands_plain_text_to_caption() {
    use gen_core::core_llm::{
        Constraint, LoadSpec, Message, ModelRequirements, Sampling, StreamEvent, TextLlmRequest,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snapshots = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots");
    let weights_dir = std::fs::read_dir(&snapshots)
        .expect("prompt-refine model staged in the HF cache")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir");

    let (system, user) =
        build_magic_prompt_messages("a red fox sitting in a snowy forest at golden hour", "1:1");
    let request = TextLlmRequest {
        messages: vec![Message::system(system), Message::user(user)],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 2048,
        seed: None,
        constraint: Some(Constraint::Json), // sc-6585: exercise constrained decoding
        ..Default::default()
    };
    let refiner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: weights_dir.to_string_lossy().into_owned(),
            ..Default::default()
        },
        &ModelRequirements::from_request(&request),
    )
    .expect("load prompt_refine via core-llm model-first resolution");

    let mut sink = |_event: StreamEvent| {};
    let output = refiner.generate(&request, &mut sink).expect("generate");
    let json = clean_json_output(&output.text);
    eprintln!("magic-prompt JSON:\n{json}");

    let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
    let cd = parsed
        .get("compositional_deconstruction")
        .expect("has compositional_deconstruction");
    assert!(cd.get("background").is_some(), "has a background");
    assert!(
        cd.get("elements").map(Value::is_array).unwrap_or(false),
        "elements is an array"
    );
}

/// Real-weight magic-prompt smoke (sc-7404) — the Windows/CUDA twin of the macOS test above. Expands
/// a plain idea into a JSON caption through the unified candle engine: `crate::inference_runtime::load_for_model`
/// resolves candle-llm's `candle-llama` on the Anubis snapshot (the JSON constraint steers the
/// model-first pick + masks the decode), and the cleaned reply must parse with the caption's required
/// section. This is the candle parity gate for retiring `candle-gen-prompt-refine`. `#[ignore]` — the
/// weights live outside CI; run on the CUDA box with the model staged (point `PROMPT_REFINE_SNAPSHOT`
/// at the snapshot dir, else it resolves the standard HF cache under `%USERPROFILE%`):
///   cargo test -p sceneworks-worker --lib --features backend-candle --release -- --ignored magic_prompt_expands_plain_text_to_caption_candle --nocapture
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "real-weight: needs the Anubis-Mini-8B prompt-refine model staged + a CUDA GPU"]
fn magic_prompt_expands_plain_text_to_caption_candle() {
    use gen_core::core_llm::{
        Constraint, LoadSpec, Message, ModelRequirements, Sampling, StreamEvent, TextLlmRequest,
    };

    // Force-link `candle-llama` so model-first resolution can discover it in this test binary
    // (the worker's platform catalog is available to the job path and this test module).

    // An explicit snapshot override wins; otherwise resolve the HF cache under the Windows home.
    let weights_dir = match std::env::var("PROMPT_REFINE_SNAPSHOT") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("USERPROFILE/HOME");
            let snapshots = std::path::Path::new(&home)
                .join(".cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots");
            std::fs::read_dir(&snapshots)
                .expect(
                    "prompt-refine model staged in the HF cache (or set PROMPT_REFINE_SNAPSHOT)",
                )
                .flatten()
                .map(|entry| entry.path())
                .find(|path| path.is_dir())
                .expect("a snapshot dir")
        }
    };

    let (system, user) =
        build_magic_prompt_messages("a red fox sitting in a snowy forest at golden hour", "1:1");
    let request = TextLlmRequest {
        messages: vec![Message::system(system), Message::user(user)],
        sampling: Sampling {
            temperature: 0.4,
            top_p: 0.9,
            ..Sampling::default()
        },
        max_new_tokens: 2048,
        seed: None,
        constraint: Some(Constraint::Json), // sc-6585: exercise constrained decoding on candle
        ..Default::default()
    };
    let refiner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: weights_dir.to_string_lossy().into_owned(),
            ..Default::default()
        },
        &ModelRequirements::from_request(&request),
    )
    .expect("load prompt_refine via core-llm model-first resolution (candle-llama)");

    let mut sink = |_event: StreamEvent| {};
    let output = refiner.generate(&request, &mut sink).expect("generate");
    let json = clean_json_output(&output.text);
    eprintln!("magic-prompt JSON (candle):\n{json}");

    let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
    let cd = parsed
        .get("compositional_deconstruction")
        .expect("has compositional_deconstruction");
    assert!(cd.get("background").is_some(), "has a background");
    assert!(
        cd.get("elements").map(Value::is_array).unwrap_or(false),
        "elements is an array"
    );
}

/// Real-weight smoke for Qwen-Image 2.1's official prompt rewriter (sc-24113, epic 24107).
///
/// `#[ignore]` — the checkpoints are OPTIONAL downloads (~18.8 GB each) and never in CI. The
/// coordinator owns the run; drive it through the RSS-guarded wrapper rather than by hand:
///
///   REWRITER=t2i scratchpad/sc24113/rewriter_guard.sh    # cap 40 GB, detached
///
/// or directly, with the snapshot staged:
///
///   PROMPT_REFINE_SNAPSHOT=~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1-PE-T2I/snapshots/f3ed7985c788ad75b3ab7223e0c4c51e2a43545b \
///     cargo test -p sceneworks-worker --lib --release -- --ignored qwen_prompt_rewrite_real_weights --nocapture
///
/// # What this proves that the offline tests cannot
///
/// Exactly three things, and deliberately nothing else:
///
///  1. `mlx-llama` really loads a `qwen3_5` 9.41B checkpoint — the hybrid linear-attention
///     decoder routed through `Architecture::Qwen35` — so "the existing LLM lane runs these"
///     stops being an inference from the config and becomes an observation.
///  2. The INSTALLED snapshot's `system_prompt.txt` is the frozen one (the digest check inside
///     `load_system_prompt`), against a real download rather than a fixture.
///  3. A real reply parses: the reasoning block survives `clean_json_output`, and the object
///     yields an editable prompt plus a `wh_ratio` that maps to one of the engine's presets.
///
/// It does NOT judge rewrite QUALITY. The template, the rewriter selection, the geometry rules
/// and every refusal are covered offline in `crate::qwen_prompt_rewrite`'s own tests, which is
/// where a behaviour change should fail first.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight: needs an optional Qwen-Image-2.1-PE checkpoint staged + a Metal device"]
fn qwen_prompt_rewrite_real_weights() {
    use crate::qwen_prompt_rewrite::{
        load_system_prompt, parse_rewrite, suggested_resolution, Rewriter, REWRITE_TEMPERATURE,
        REWRITE_TOP_K, REWRITE_TOP_P, T2I_MAX_NEW_TOKENS,
    };
    use gen_core::core_llm::{
        LoadSpec, Message, ModelRequirements, Sampling, StreamEvent, TextLlmRequest,
    };

    let weights_dir = std::path::PathBuf::from(
        std::env::var("PROMPT_REFINE_SNAPSHOT")
            .expect("set PROMPT_REFINE_SNAPSHOT to the staged PE snapshot directory"),
    );
    // Which half is staged decides which template is verified — the same request-shaped choice
    // production makes, just read off the directory here.
    let rewriter = if weights_dir.to_string_lossy().contains("PE-I2I") {
        Rewriter::Editing
    } else {
        Rewriter::TextToImage
    };
    eprintln!("rewriter={:?} snapshot={}", rewriter, weights_dir.display());

    // (2) The freeze, against a real download.
    let system = load_system_prompt(&weights_dir, rewriter)
        .expect("the staged snapshot carries the FROZEN system_prompt.txt");
    assert!(!system.trim().is_empty());

    let request = TextLlmRequest {
        messages: vec![
            Message::system(system),
            Message::user("a lighthouse on a cliff at dusk".to_owned()),
        ],
        // Upstream's published recipe, verbatim.
        sampling: Sampling {
            temperature: REWRITE_TEMPERATURE,
            top_p: REWRITE_TOP_P,
            top_k: REWRITE_TOP_K,
            ..Sampling::default()
        },
        max_new_tokens: T2I_MAX_NEW_TOKENS,
        seed: None,
        ..Default::default()
    };

    // (1) The load. A failure here IS the architecture finding.
    let refiner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: weights_dir.to_string_lossy().into_owned(),
            quantize: None,
            cuda_graphs: None,
            // The PE checkpoints carry their vision tower inside the snapshot (`model.visual.*`),
            // like every other `qwen3_5` wrapper, so there is no separate projector to point at.
            projector_source: None,
            ..LoadSpec::default()
        },
        &ModelRequirements::from_request(&request),
    )
    .expect("mlx-llama loads the qwen3_5 PE checkpoint through Architecture::Qwen35");

    let mut sink = |_event: StreamEvent| {};
    let output = refiner.generate(&request, &mut sink).expect("generate");
    eprintln!("raw reply:\n{}", output.text);

    // (3) The parse, through the SAME cleanup the job uses.
    let suggestion = parse_rewrite(&clean_json_output(&output.text))
        .expect("the reply is the published JSON object");
    assert!(
        !suggestion.rewritten_prompt.trim().is_empty(),
        "an empty rewrite would be offered to Apply and erase the user's prompt"
    );
    eprintln!(
        "rewritten_prompt: {}\nwh_ratio: {:?} -> preset {:?}",
        suggestion.rewritten_prompt,
        suggestion.wh_ratio,
        suggested_resolution(&suggestion)
    );
    if !suggestion.wh_ratio.is_empty() {
        assert!(
            suggested_resolution(&suggestion).is_some(),
            "a stated ratio must map to one of the engine's seven presets, got {:?}",
            suggestion.wh_ratio
        );
    }
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

// ------------------------------------------------------------------------------------------
// Magic-prompt model bake-off (sc-6550) — a parameterized harness + degeneracy rubric for
// choosing the magic-prompt model that fixes the sc-6546 caption-quality ceiling. The 3B
// (`huihui-ai/Llama-3.2-3B-Instruct-abliterated`) stochastically emits SEMANTICALLY-DEGENERATE
// captions (the subject placed as a `type:"text"` element, or `background` set to a transparent
// cutout for a prompt that never asked for one) which placeholder at 1024²/48. This harness runs
// Ideogram's magic-prompt v1 system prompt (the SHIPPING `build_magic_prompt_messages`) over a
// fixed prompt set through the REAL `prompt_refine` provider and scores the degeneracy from the
// caption JSON alone — the root-cause signal, no Ideogram render needed (metric (c), the
// placeholder-escape render confirmation, is the gated downstream step).
// ------------------------------------------------------------------------------------------

/// One bake-off prompt + what a CORRECT caption looks like for it, so the analyzer can tell a
/// degeneracy (subject-as-text / unrequested transparent cutout) apart from a legitimate use of
/// those features.
// `text` / `edgy` are read only by the macOS-gated real-weight bakeoff test below; on the Linux
// clippy lane that test is cfg'd out, so suppress the cross-platform dead-code lint.
#[allow(dead_code)]
struct BakeoffPrompt {
    text: &'static str,
    /// The prompt legitimately calls for `type:"text"` (typography / rendered words), so a text
    /// element is NOT degenerate here.
    wants_text: bool,
    /// The prompt legitimately calls for a transparent background (product cutout / sticker), so
    /// `background:"…transparent…"` is NOT degenerate here.
    wants_transparent: bool,
    /// An edgy prompt — present to measure refusals (an uncensored model must still emit a
    /// caption). A refusal shows up as `!caption_valid`.
    edgy: bool,
}

/// The fixed bake-off prompt set: the sc-6546 failing prompt + diverse plain subjects (the
/// degeneracy measurement), two CONTROLS that legitimately want a text element / transparent
/// background, and two edgy prompts (refusal test for the uncensored requirement).
const BAKEOFF_PROMPTS: &[BakeoffPrompt] = &[
        // The sc-6546 failing prompt.
        BakeoffPrompt { text: "a red fox sitting in a snowy forest at golden hour", wants_text: false, wants_transparent: false, edgy: false },
        // Diverse plain subjects — none ask for text or transparency, so ANY text element or
        // transparent background is the sc-6546 degeneracy.
        BakeoffPrompt { text: "a steaming bowl of ramen on a rustic wooden table", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a vintage red bicycle leaning against a weathered brick wall", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a snow leopard walking across a rocky mountain ridge", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "an astronaut floating above the earth in low orbit", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a cozy log cabin in a pine forest during a snowstorm", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a golden retriever puppy playing in a pile of autumn leaves", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a futuristic city skyline at night lit by neon signs", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a single sunflower in a glass vase on a sunlit windowsill", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a lone samurai standing in a misty bamboo forest at dawn", wants_text: false, wants_transparent: false, edgy: false },
        // CONTROL: legitimately wants text — the model SHOULD emit a `type:"text"` element here.
        BakeoffPrompt { text: "a bold motivational poster with the headline NEVER GIVE UP in large letters", wants_text: true, wants_transparent: false, edgy: false },
        // CONTROL: legitimately wants a transparent background — a product cutout.
        BakeoffPrompt { text: "a product cutout of a red running sneaker on a transparent background", wants_text: false, wants_transparent: true, edgy: false },
        // Edgy (refusal test for the uncensored requirement).
        BakeoffPrompt { text: "a gritty film-noir crime scene, a detective examining a body under a harsh streetlight", wants_text: false, wants_transparent: false, edgy: true },
        BakeoffPrompt { text: "a fierce barbarian warrior, bloodied and scarred, standing over a fallen foe after battle", wants_text: false, wants_transparent: false, edgy: true },
    ];

/// Degeneracy metrics for a single caption reply, scored against what the prompt legitimately
/// expected. This is the sc-6546 rubric in code — pure, so it is unit-tested without weights.
#[derive(Clone, Copy, Debug, Default)]
// `parse_ok` is read only by the macOS-gated bakeoff test; suppress the Linux-lane dead-code lint.
#[allow(dead_code)]
struct CaptionMetrics {
    /// The cleaned reply parses as JSON.
    parse_ok: bool,
    /// A structured caption — a JSON object with a `compositional_deconstruction` object
    /// (mirrors `sceneworks_core::ideogram_caption::is_caption`).
    caption_valid: bool,
    n_obj: usize,
    n_text: usize,
    has_transparent_bg: bool,
    /// A text element appeared where the prompt did not ask for one (subject-as-`text`
    /// degeneracy).
    text_when_unwanted: bool,
    /// A transparent background appeared where the prompt did not ask for one.
    transparent_when_unwanted: bool,
}

impl CaptionMetrics {
    /// The headline placeholder-risk proxy: a malformed/non-caption reply, an unrequested
    /// transparent cutout, an unrequested text element, or a caption with NO object subject.
    fn degenerate(&self) -> bool {
        !self.caption_valid
            || self.transparent_when_unwanted
            || self.text_when_unwanted
            || self.n_obj == 0
    }
}

/// Score a raw model reply for `prompt` (applies the shipping `clean_json_output` first).
fn analyze_caption(prompt: &BakeoffPrompt, raw: &str) -> CaptionMetrics {
    let cleaned = clean_json_output(raw);
    let Ok(value) = serde_json::from_str::<Value>(&cleaned) else {
        return CaptionMetrics::default();
    };
    let mut m = CaptionMetrics {
        parse_ok: true,
        ..Default::default()
    };
    let Some(cd) = value
        .get("compositional_deconstruction")
        .and_then(Value::as_object)
    else {
        return m; // parsed, but not a caption (caption_valid stays false)
    };
    m.caption_valid = true;
    if let Some(bg) = cd.get("background").and_then(Value::as_str) {
        m.has_transparent_bg = bg.to_ascii_lowercase().contains("transparent");
    }
    if let Some(elements) = cd.get("elements").and_then(Value::as_array) {
        for el in elements {
            match el.get("type").and_then(Value::as_str) {
                Some("text") => m.n_text += 1,
                _ => m.n_obj += 1, // obj (or an untyped element defaults to obj per the serializer)
            }
        }
    }
    m.text_when_unwanted = m.n_text > 0 && !prompt.wants_text;
    m.transparent_when_unwanted = m.has_transparent_bg && !prompt.wants_transparent;
    m
}

#[test]
fn analyze_caption_flags_the_sc6546_degeneracies() {
    // Healthy: one obj subject, opaque scene background, no text → not degenerate.
    let plain = &BAKEOFF_PROMPTS[0];
    let healthy = analyze_caption(
        plain,
        r#"{"compositional_deconstruction": {"background": "a snowy forest at golden hour", "elements": [{"type": "obj", "desc": "a red fox"}]}}"#,
    );
    assert!(healthy.caption_valid && healthy.n_obj == 1 && !healthy.degenerate());

    // Subject-as-text degeneracy: the fox emitted as a `type:"text"` element.
    let subj_text = analyze_caption(
        plain,
        r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": [{"type": "text", "text": "a red fox", "desc": "a red fox"}]}}"#,
    );
    assert!(subj_text.text_when_unwanted && subj_text.n_obj == 0 && subj_text.degenerate());

    // Transparent-background degeneracy for a scene prompt that never asked for one.
    let transp = analyze_caption(
        plain,
        r#"{"compositional_deconstruction": {"background": "on a transparent background", "elements": [{"type": "obj", "desc": "a red fox"}]}}"#,
    );
    assert!(transp.transparent_when_unwanted && transp.degenerate());

    // A refusal / non-JSON reply is degenerate (parse fails).
    assert!(analyze_caption(plain, "I cannot help with that request.").degenerate());

    // CONTROL: the transparent-cutout prompt legitimately uses a transparent background.
    let cutout = &BAKEOFF_PROMPTS[11];
    assert!(cutout.wants_transparent);
    let ok_transp = analyze_caption(
        cutout,
        r#"{"compositional_deconstruction": {"background": "on a transparent background", "elements": [{"type": "obj", "desc": "a red sneaker"}]}}"#,
    );
    assert!(!ok_transp.transparent_when_unwanted && !ok_transp.degenerate());

    // CONTROL: the poster prompt legitimately uses a text element.
    let poster = &BAKEOFF_PROMPTS[10];
    assert!(poster.wants_text);
    let ok_text = analyze_caption(
        poster,
        r#"{"compositional_deconstruction": {"background": "a plain studio backdrop", "elements": [{"type": "obj", "desc": "a poster"}, {"type": "text", "text": "NEVER GIVE UP", "desc": "the headline"}]}}"#,
    );
    assert!(!ok_text.text_when_unwanted && !ok_text.degenerate());
}

/// Real-weight magic-prompt bake-off (sc-6550, ported to the unified engine in sc-7158). Runs the
/// shipping magic-prompt system prompt over `BAKEOFF_PROMPTS` × N seeds through mlx-llm's `mlx-llama`
/// (resolved model-first by `crate::inference_runtime::load_for_model`) and prints per-run + aggregate
/// degeneracy metrics. `#[ignore]` — the weights live outside CI. Point it at any Llama-3.x-Instruct
/// snapshot dir (the 3B baseline, a Llama-3.1-8B, an Anubis-8B …). Measure footprint by wrapping the
/// TEST BINARY with `/usr/bin/time -l` (NOT `cargo test`, which reports the cargo parent's peak):
///   BAKEOFF_MODEL_DIR=~/.cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots/<rev> \
///   BAKEOFF_SEEDS=3 cargo test -p sceneworks-worker --lib -- --ignored --nocapture magic_prompt_bakeoff
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight: set BAKEOFF_MODEL_DIR to a Llama-3.x-Instruct snapshot dir"]
fn magic_prompt_bakeoff() {
    use gen_core::core_llm::{
        Constraint, LoadSpec, Message, ModelRequirements, Sampling, StreamEvent, TextLlmRequest,
    };
    use std::time::Instant;

    let weights_dir = std::path::PathBuf::from(
        std::env::var("BAKEOFF_MODEL_DIR").expect("set BAKEOFF_MODEL_DIR to a snapshot dir"),
    );
    let seeds: u64 = std::env::var("BAKEOFF_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    // sc-6585: set BAKEOFF_CONSTRAIN=1 to measure caption_valid under grammar-constrained decoding
    // (expect ~100%) vs the unconstrained baseline. Steers the model-first pick to a JSON provider.
    let constrain = std::env::var("BAKEOFF_CONSTRAIN").is_ok();
    eprintln!("BAKEOFF model_dir={}", weights_dir.display());

    let mut reqs = ModelRequirements::default();
    if constrain {
        reqs = reqs.with_constraint(Constraint::Json);
    }
    let refiner = crate::inference_runtime::load_for_model_with(
        &LoadSpec {
            source: weights_dir.to_string_lossy().into_owned(),
            ..Default::default()
        },
        &reqs,
    )
    .expect("load prompt_refine via core-llm model-first resolution");

    let (mut runs, mut parse_ok, mut caption_valid, mut degenerate) = (0u32, 0u32, 0u32, 0u32);
    let (mut text_bad, mut transp_bad, mut no_obj) = (0u32, 0u32, 0u32);
    // Refusal tracking on the edgy prompts only.
    let (mut edgy_runs, mut edgy_valid) = (0u32, 0u32);
    let mut total_ms = 0u128;

    for prompt in BAKEOFF_PROMPTS {
        let (system, user) = build_magic_prompt_messages(prompt.text, "1:1");
        for seed in 0..seeds {
            let request = TextLlmRequest {
                messages: vec![Message::system(system.clone()), Message::user(user.clone())],
                sampling: Sampling {
                    temperature: 0.4, // matches the worker's magic-prompt sampling
                    top_p: 0.9,
                    ..Sampling::default()
                },
                max_new_tokens: 2048,
                seed: Some(seed),
                constraint: constrain.then_some(Constraint::Json),
                ..Default::default()
            };
            let mut sink = |_event: StreamEvent| {};
            let start = Instant::now();
            let output = refiner.generate(&request, &mut sink).expect("generate");
            let ms = start.elapsed().as_millis();
            total_ms += ms;

            let m = analyze_caption(prompt, &output.text);
            runs += 1;
            parse_ok += m.parse_ok as u32;
            caption_valid += m.caption_valid as u32;
            degenerate += m.degenerate() as u32;
            text_bad += m.text_when_unwanted as u32;
            transp_bad += m.transparent_when_unwanted as u32;
            no_obj += (m.caption_valid && m.n_obj == 0) as u32;
            if prompt.edgy {
                edgy_runs += 1;
                edgy_valid += m.caption_valid as u32;
            }
            eprintln!(
                "BAKEOFF run seed={seed} {}ms valid={} obj={} text={} transp_bg={} degen={} :: {}",
                ms,
                m.caption_valid,
                m.n_obj,
                m.n_text,
                m.has_transparent_bg,
                m.degenerate(),
                prompt.text
            );
            // Surface the cleaned JSON so degeneracies are eyeball-auditable.
            eprintln!("BAKEOFF json :: {}", clean_json_output(&output.text));
        }
    }

    let pct = |n: u32| 100.0 * n as f64 / runs.max(1) as f64;
    eprintln!(
        "\nBAKEOFF SUMMARY ({runs} runs, {seeds} seeds × {} prompts)",
        BAKEOFF_PROMPTS.len()
    );
    eprintln!("  parse_ok           {:.1}%", pct(parse_ok));
    eprintln!("  caption_valid      {:.1}%", pct(caption_valid));
    eprintln!(
        "  DEGENERATE         {:.1}%  <- headline placeholder-risk proxy",
        pct(degenerate)
    );
    eprintln!("    text_when_unwanted   {:.1}%", pct(text_bad));
    eprintln!("    transp_when_unwanted {:.1}%", pct(transp_bad));
    eprintln!("    valid_but_no_obj     {:.1}%", pct(no_obj));
    eprintln!(
        "  edgy_caption_valid {:.1}% ({edgy_valid}/{edgy_runs})  (refusal proxy)",
        100.0 * edgy_valid as f64 / edgy_runs.max(1) as f64
    );
    eprintln!("  avg_latency        {}ms", total_ms / runs.max(1) as u128);
}
