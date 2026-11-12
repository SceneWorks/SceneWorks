//! Request preparation for prompt-refinement jobs.
use super::*;

pub(super) struct RefineRequest {
    pub(super) task: RefineTask,
    pub(super) original_prompt: String,
    pub(super) image_refs: Vec<gen_core::core_llm::ImageRef>,
    pub(super) image_reference_count: usize,
    pub(super) qwen_rewriter: Option<crate::qwen_prompt_rewrite::Rewriter>,
    pub(super) model: String,
    pub(super) target_model_id: Option<String>,
    pub(super) thinking_mode: gen_core::core_llm::ThinkingMode,
    pub(super) thinking_mode_name: &'static str,
    pub(super) max_new_tokens: u32,
    pub(super) kv_compression: gen_core::core_llm::KvCompressionPolicy,
    pub(super) temperature: f32,
    pub(super) work_message: &'static str,
    pub(super) done_message: &'static str,
    pub(super) weights_dir: PathBuf,
    pub(super) system: String,
    pub(super) user_message: String,
    /// SW-7102: the engine-built MiniMax-H3-Image expansion request and what its reply is checked
    /// against. `None` for every other task.
    pub(super) h3_expand: Option<H3Expand>,
}

/// A MiniMax-H3-Image expansion: the request `core_llm::minimax_h3_image_pe` assembled from the
/// model's verified prompt files, and the reply contract.
pub(super) struct H3Expand {
    pub(super) request: gen_core::core_llm::TextLlmRequest,
    pub(super) reply: H3ExpandReply,
}

/// What an expansion reply is parsed against.
pub(crate) struct H3ExpandReply {
    pub(crate) workflow: crate::minimax_h3_image_prompt_expand::Workflow,
    pub(crate) aspect: gen_core::core_llm::minimax_h3_image_pe::AspectSelection,
    pub(crate) allowed_ratios: Vec<String>,
    pub(crate) tier: Option<&'static str>,
}

pub(super) async fn parse_refine_request(
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<RefineRequest> {
    let payload = &job.payload;
    // The job is dispatched by the `task` discriminator: `magic_prompt` (text idea → JSON caption,
    // sc-5997), `image_caption` (reference image → JSON caption, sc-8105 — the `core_llm` VISION path),
    // or the default free-text rewrite. The two caption tasks both emit a JSON-constrained Ideogram
    // caption; `image_caption` additionally carries an image and needs no text prompt.
    // Classify the job into ONE `RefineTask` value from the `task` discriminator, instead of the five
    // scattered booleans re-derived per ladder that this refactor replaced (sc-8921, F-119). Every
    // per-task property (vision? caption? temperature, messages) is now a method on the enum or a single
    // `match` below, so no incoherent boolean combination is representable.
    let task = RefineTask::from_payload(payload.get("task").and_then(Value::as_str));
    let original_prompt = payload
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    // The vision tasks (image_caption / image_describe) are driven by the reference image, not a text
    // prompt, so they do not require a `prompt`; every other task does.
    if original_prompt.is_empty() && !task.is_vision() {
        return Err(WorkerError::InvalidPayload(
            "Prompt refinement requires a non-empty prompt.".to_owned(),
        ));
    }
    // The vision tasks resolve + decode a reference image (the JoyCaption `load_caption_image` pattern →
    // RGB8) into the vision contract's `ImageRef`. Accept either `imagePath` or `referencePath`. The path
    // is UNTRUSTED (it arrives on the job payload over the LAN-remote API boundary, epic 4484), so — like
    // every other on-disk image/model input (JoyCaption via `resolve_dataset_item_path`, the
    // InstantID/captioner reference reads, the LoRA load path) — confine it to an app-managed root via
    // `normalize_app_managed_model_path` BEFORE opening it. That rejects `..` traversal and any absolute
    // path outside the app data dir / HF hub cache, closing the arbitrary-file-read gap.
    // A vision task carries one OR MORE reference images. The plural `imagePaths` array (a mood board,
    // sc-8595) takes precedence; otherwise the single `imagePath`/`referencePath` (unchanged single-image
    // path). Every path is UNTRUSTED (it arrives on the job payload over the LAN-remote API boundary,
    // epic 4484), so each is confined to an app-managed root via `normalize_app_managed_model_path`
    // BEFORE opening it — rejecting `..` traversal and any absolute path outside the app data dir / HF
    // hub cache. The order of `imagePaths` is preserved so the model reads the board in the order the
    // user assembled it. N is bounded API-side (`MAX_MOOD_BOARD_IMAGES`).
    let image_refs: Vec<(gen_core::core_llm::ImageRef, (u32, u32))> = if task
        .takes_reference_images()
    {
        let paths: Vec<String> = match payload.get("imagePaths").and_then(Value::as_array) {
            Some(items) => items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
            None => payload
                .get("imagePath")
                .or_else(|| payload.get("referencePath"))
                .and_then(crate::payload::trimmed_value_str)
                .map(str::to_owned)
                .into_iter()
                .collect(),
        };
        // A VISION task has nothing to look at without an image, so an empty list is a payload
        // error. The Qwen rewrite is different by design: zero references is the legitimate
        // text-to-image case, and it is what selects the T2I rewriter below.
        if paths.is_empty() && task.is_vision() {
            return Err(WorkerError::InvalidPayload(
                "This task requires at least one non-empty `imagePath`/`imagePaths` reference image."
                    .to_owned(),
            ));
        }
        // The rewrite must see EXACTLY the ordered list the render will condition on, capped the
        // same way (S3 edit contract: 1–10 ordered references). Over the cap is a refusal rather
        // than a silent truncation: a rewrite whose `<imageN>` numbering does not match the render's
        // references would name the wrong picture.
        if task == RefineTask::QwenImageRewrite
            && paths.len() > crate::qwen_prompt_rewrite::MAX_REWRITE_REFERENCES
        {
            return Err(WorkerError::InvalidPayload(format!(
                "Qwen Image 2.1 prompt rewriting takes at most {} reference images; {} were sent.",
                crate::qwen_prompt_rewrite::MAX_REWRITE_REFERENCES,
                paths.len()
            )));
        }
        // SW-7102: the expander takes the image model's own reference cap, refused rather than
        // truncated for the same reason — `<Picture N>` must name the render's N-th reference.
        let expand_cap = sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES;
        if task == RefineTask::MiniMaxH3ImageExpand && paths.len() > expand_cap {
            return Err(WorkerError::InvalidPayload(format!(
                "MiniMax-H3-Image prompt expansion takes at most {expand_cap} reference images; \
                 {} were sent.",
                paths.len()
            )));
        }
        let safe_paths = paths
            .iter()
            .map(|path| normalize_app_managed_model_path(settings, path, "Vision reference image"))
            .collect::<WorkerResult<Vec<_>>>()?;
        crate::load_progress::spawn_blocking(move || {
            safe_paths
                .iter()
                .map(|path| load_vision_reference(path))
                .collect::<WorkerResult<Vec<_>>>()
        })
        .await
        .map_err(|error| task_join_error("vision reference image decode", error))??
    } else {
        Vec::new()
    };
    // The expander labels each reference with its SOURCE ratio; every other task needs only the
    // (downscaled) image.
    let (mut image_refs, reference_sizes): (Vec<gen_core::core_llm::ImageRef>, Vec<(u32, u32)>) =
        image_refs.into_iter().unzip();
    // More than one reference image switches the vision instruction to mood-board SYNTHESIS (one prompt
    // capturing the shared aesthetic) rather than describing a single picture.
    let multi_reference = image_refs.len() > 1;
    // Captured before `image_refs` is moved into the chat turn. The rewrite's `ratio_follow`
    // ("<image2>") is only meaningful against the list it was shown, so the result block needs the
    // count to reject an index past it.
    let image_reference_count = image_refs.len();
    // sc-24113: which of the two published Qwen rewriters this request selects, decided by the
    // REQUEST — references present means the editing rewriter — and never by a user-facing picker.
    // `None` for every other task.
    let qwen_rewriter = (task == RefineTask::QwenImageRewrite)
        .then(|| crate::qwen_prompt_rewrite::Rewriter::for_reference_count(image_refs.len()));
    let model = crate::payload::trimmed_string(payload.get("model"))
        .map(str::to_owned)
        // A rewrite defaults to its own selected checkpoint, not to the generic Anubis refiner.
        // An explicit `payload.model` still wins, which is how the real-weight smoke pins a local
        // snapshot path — the same override the film planner uses.
        .unwrap_or_else(|| {
            if task == RefineTask::MiniMaxH3ImageExpand {
                return gen_core::core_llm::minimax_h3_image_pe::REPO.to_owned();
            }
            qwen_rewriter
                .map(|rewriter| rewriter.repo().to_owned())
                .unwrap_or_else(|| DEFAULT_REFINE_MODEL.to_owned())
        });
    // The TARGET model the refined prompt is destined for (the catalog id the studio has selected),
    // as distinct from `model` above, which is the refiner CHECKPOINT that does the rewriting. The
    // API forwards it as `modelId` for every refine request; sc-17162 is the first reader.
    let target_model_id = payload
        .get("modelId")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut thinking_mode = match (task, payload.get("thinkingMode").and_then(Value::as_str)) {
        (RefineTask::FilmPlan, Some("enabled")) => gen_core::core_llm::ThinkingMode::Enabled,
        (RefineTask::FilmPlan, Some("auto")) => gen_core::core_llm::ThinkingMode::Auto,
        (RefineTask::FilmPlan, _) => gen_core::core_llm::ThinkingMode::Disabled,
        _ => gen_core::core_llm::ThinkingMode::Auto,
    };
    let mut max_new_tokens = resolve_max_new_tokens(payload, task, qwen_rewriter);
    // sc-20682: the job's compressed-KV opt-in (its payload's `kvCompression`, else the worker's
    // `SCENEWORKS_LLM_KV_COMPRESSION`, else off). The engine runs compressed only for a single
    // sequence on a model matching one of its two measured architectures (Llama-3.2-3B-Instruct,
    // dense Qwen3-1.7B) within that row's context range (`crate::llm_kv_cache`); batched, short and
    // other-model requests run dense with a reason. It reports what it ran on.
    let kv_compression = crate::llm_kv_cache::job_policy(payload)?;
    let temperature = task.temperature();
    let work_message = task.work_message();
    let done_message = task.done_message();

    // Resolved BEFORE the message match (it used to sit after it) because one task's system prompt
    // is not a compiled-in asset: the Qwen rewriters ship their published template inside the
    // snapshot as `system_prompt.txt`, so building their turn needs the model directory.
    //
    // SW-7102: the expander resolves its tier directory and its upstream `prompt/` snapshot from its
    // own catalog entry (the API forwards it as `modelManifestEntry`); an explicit `payload.model`
    // is a local snapshot override, as for the other refiners.
    let expander_dirs = if task == RefineTask::MiniMaxH3ImageExpand {
        if let Some(target) = target_model_id.as_deref() {
            if !crate::minimax_h3_image_prompt_expand::is_expand_target_model(target) {
                return Err(WorkerError::InvalidPayload(format!(
                    "The MiniMax-H3-Image prompt expander only applies to `{}`; this request \
                     targets `{target}`.",
                    sceneworks_core::contracts::MINIMAX_H3_IMAGE_MODEL_ID
                )));
            }
        }
        let entry = payload
            .get("modelManifestEntry")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let override_dir = crate::payload::trimmed_string(payload.get("model"))
            .map(|path| resolve_app_managed_model_dir(settings, path, "prompt-refine model path"))
            .transpose()?;
        Some(crate::minimax_h3_image_prompt_expand::resolve_dirs(
            settings,
            &entry,
            crate::payload::trimmed_string(payload.get("peTier")),
            override_dir,
        )?)
    } else {
        None
    };
    let weights_dir = match &expander_dirs {
        Some(dirs) => dirs.weights.clone(),
        None => resolve_app_managed_model_dir(settings, &model, "prompt-refine model path")?,
    };
    let mut h3_expand = None;

    // Build the `(system, user)` chat text per task: the caption tasks swap Ideogram's caption system
    // prompt in for the rewrite rules, describe supplies the prose/tags describe asset, and the default
    // rewrite assembles its guide/workflow system prompt with the user's prompt as the turn.
    let (system, user_message) = match task {
        RefineTask::ImageCaption => build_image_caption_messages(multi_reference),
        RefineTask::ImageDescribe => {
            // sc-8205: the describe style (prose vs booru tags) is selected per-model by the
            // `captionStyle` payload field the web forwards from the catalog; absent/unknown → prose.
            let style =
                DescribeStyle::from_payload(payload.get("captionStyle").and_then(Value::as_str));
            build_image_describe_messages(style, multi_reference)
        }
        RefineTask::MagicPrompt => {
            let aspect_ratio =
                crate::payload::trimmed_string(payload.get("aspectRatio")).unwrap_or("1:1");
            build_magic_prompt_messages(&original_prompt, aspect_ratio)
        }
        // sc-22713: the composed planning request IS the user turn; the embedded asset carries the
        // JSON contract. No guide/workflow assembly — a plan is not a prompt rewrite.
        RefineTask::FilmPlan => build_film_plan_messages(&original_prompt),
        // sc-24113. The system prompt is Qwen's OWN published template, read out of the installed
        // snapshot and verified against the frozen revision's SHA-256 — not a SceneWorks rewrite
        // and not vendored into this repo (`qwen_prompt_rewrite`'s module docs give the licence and
        // freshness reasons). The user turn is the brief verbatim; for an edit the ordered
        // references ride alongside it as `Content::Image` blocks, attached below exactly as the
        // caption tasks attach theirs.
        RefineTask::QwenImageRewrite => {
            let rewriter = qwen_rewriter.expect("the rewrite task always selects a rewriter");
            // The rewriters are keyed to ONE image model. Serving a different target's request with
            // Qwen's template would hand the user a rewrite tuned for a model they are not running.
            if let Some(target) = target_model_id.as_deref() {
                if !crate::qwen_prompt_rewrite::is_rewrite_target_model(target) {
                    return Err(WorkerError::InvalidPayload(format!(
                        "The Qwen Image 2.1 prompt rewriters only apply to {}; this request \
                         targets `{target}`.",
                        crate::qwen_prompt_rewrite::rewrite_target_models_label()
                    )));
                }
            }
            (
                crate::qwen_prompt_rewrite::load_system_prompt(&weights_dir, rewriter)?,
                original_prompt.clone(),
            )
        }
        // SW-7102. The engine assembles the whole request from the model's own prompt files, read
        // from the upstream snapshot and checked against `prompt/manifest.json`; nothing licensed
        // is compiled in. An edit's references ride inside that request as `<Picture N>` parts,
        // in order, each labelled with its source ratio.
        RefineTask::MiniMaxH3ImageExpand => {
            use crate::minimax_h3_image_prompt_expand as expand;
            let dirs = expander_dirs
                .as_ref()
                .expect("the expand task always resolves its directories");
            let files = expand::load_prompt_files(&dirs.prompt_root)?;
            let aspect =
                expand::aspect_selection(payload.get("aspectRatio").and_then(Value::as_str))?;
            let references: Vec<_> = image_refs
                .drain(..)
                .zip(reference_sizes.iter().copied())
                .map(|(image, size)| expand::edit_reference(image, size))
                .collect();
            let mut request =
                expand::build_request(&files, &original_prompt, &references, &aspect)?;
            // The published budget unless the payload overrides it.
            if payload
                .get("maxNewTokens")
                .and_then(Value::as_u64)
                .is_none_or(|value| value == 0)
            {
                max_new_tokens = request.max_new_tokens;
            }
            request.max_new_tokens = max_new_tokens;
            h3_expand = Some(H3Expand {
                request,
                reply: H3ExpandReply {
                    workflow: expand::Workflow::for_reference_count(references.len()),
                    aspect,
                    allowed_ratios: files.params().aspect_ratios.clone(),
                    tier: dirs.tier,
                },
            });
            (String::new(), String::new())
        }
        RefineTask::Rewrite => {
            let guide = payload
                .get("guide")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let workflow = payload
                .get("workflow")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (
                // sc-17162: an H3 target model swaps in the tailored Context-IR-shaped system
                // prompt; every other model keeps the generic medium-keyed rewrite rules.
                build_rewrite_system_prompt(
                    guide.as_deref(),
                    workflow.as_deref(),
                    target_model_id.as_deref(),
                ),
                original_prompt.clone(),
            )
        }
    };
    // The expansion runs the recipe's own reasoning mode (`enable_thinking` in its
    // `request_params.json`), which the engine put on the request.
    if let Some(expand) = &h3_expand {
        thinking_mode = expand.request.thinking;
    }
    let thinking_mode_name = match thinking_mode {
        gen_core::core_llm::ThinkingMode::Enabled => "enabled",
        gen_core::core_llm::ThinkingMode::Auto => "auto",
        gen_core::core_llm::ThinkingMode::Disabled => "disabled",
    };
    Ok(RefineRequest {
        task,
        original_prompt,
        image_refs,
        image_reference_count,
        qwen_rewriter,
        model,
        target_model_id,
        thinking_mode,
        thinking_mode_name,
        max_new_tokens,
        kv_compression,
        temperature,
        work_message,
        done_message,
        weights_dir,
        system,
        user_message,
        h3_expand,
    })
}
