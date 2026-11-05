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
    let image_refs: Vec<gen_core::core_llm::ImageRef> = if task.takes_reference_images() {
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
        let safe_paths = paths
            .iter()
            .map(|path| normalize_app_managed_model_path(settings, path, "Vision reference image"))
            .collect::<WorkerResult<Vec<_>>>()?;
        crate::load_progress::spawn_blocking(move || {
            safe_paths
                .iter()
                .map(|path| load_caption_image_ref(path))
                .collect::<WorkerResult<Vec<_>>>()
        })
        .await
        .map_err(|error| task_join_error("vision reference image decode", error))??
    } else {
        Vec::new()
    };
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
    let thinking_mode = match (task, payload.get("thinkingMode").and_then(Value::as_str)) {
        (RefineTask::FilmPlan, Some("enabled")) => gen_core::core_llm::ThinkingMode::Enabled,
        (RefineTask::FilmPlan, Some("auto")) => gen_core::core_llm::ThinkingMode::Auto,
        (RefineTask::FilmPlan, _) => gen_core::core_llm::ThinkingMode::Disabled,
        _ => gen_core::core_llm::ThinkingMode::Auto,
    };
    let thinking_mode_name = match thinking_mode {
        gen_core::core_llm::ThinkingMode::Enabled => "enabled",
        gen_core::core_llm::ThinkingMode::Auto => "auto",
        gen_core::core_llm::ThinkingMode::Disabled => "disabled",
    };
    let max_new_tokens = resolve_max_new_tokens(payload, task, qwen_rewriter);
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
    let weights_dir = resolve_app_managed_model_dir(settings, &model, "prompt-refine model path")?;

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
    })
}
