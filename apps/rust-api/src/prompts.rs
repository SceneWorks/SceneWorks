use super::*;

/// Maximum reference images in one "mood board" describe/caption request (epic 8588, sc-8595). Each
/// reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so the cost scales with N; this bounds
/// the vision attention + context a single request can demand. The Image Studio picker enforces the same
/// ceiling client-side — this is the authoritative server-side guard.
pub(crate) const MAX_MOOD_BOARD_IMAGES: usize = 6;

// The API and worker classify the same core-owned discriminator.
pub(crate) use sceneworks_core::contracts::QWEN_IMAGE_REWRITE_TASK;
pub(crate) use sceneworks_core::contracts::{
    MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES, MINIMAX_H3_IMAGE_EXPAND_TASK,
    MINIMAX_H3_IMAGE_MODEL_ID, MINIMAX_H3_IMAGE_PE_MODEL_ID,
};

/// Maximum reference images on a Qwen-Image 2.1 rewrite request.
///
/// TEN, not [`MAX_MOOD_BOARD_IMAGES`], and the difference is not a loosened guard but a different
/// contract. A mood board is a synthesis whose cost the 6 bounds; a rewrite must be shown EXACTLY
/// the ordered list the render will condition on (S3 edit contract: 1–10), or its `ratio_follow`
/// ("<image2>") names a different picture than the one the user attached. The engine and the image
/// enqueue path both cap at 10, so this is the same number stated at the third seam rather than a
/// new one.
pub(crate) const MAX_QWEN_REWRITE_IMAGES: usize = 10;

/// Enqueue a `prompt_refine` job: a lightweight, non-GPU job that asks an
/// OpenAI-compatible LLM to rewrite the user's prompt to follow the selected
/// model's prompt guide. The job runs through a native TextLlm provider, and the client reads the refined
/// prompt from the completed job's `result.refinedPrompt`.
pub(crate) async fn create_prompt_refine_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<PromptRefineRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let task = payload
        .task
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    // The reference-image vision tasks — `image_caption` (epic 8102, sc-8108) → JSON caption, and
    // `image_describe` (epic 8203, sc-8206) → plain-text description — are driven by an image, not a
    // text prompt: they carry a project `sourceAssetId` instead. Resolve that to the worker's confined
    // on-disk `imagePath` and forward the vision model's repo; the prompt requirement is waived.
    let is_vision_task = task == Some("image_caption") || task == Some("image_describe");
    // Qwen-Image 2.1's rewrite (sc-24113) reads reference images too, but is NOT a vision task: it
    // is driven by the user's TEXT and merely accepts pictures. The prompt stays required, and zero
    // references is not an error — it is the legitimate text-to-image case, and it is what selects
    // the T2I rewriter over the editing one in the worker.
    let is_qwen_rewrite = task == Some(QWEN_IMAGE_REWRITE_TASK);
    // MiniMax-H3-Image's prompt expansion (SW-7102) has the same shape: driven by the user's text,
    // with zero references meaning text-to-image and one or more meaning an edit whose instruction
    // names them as `<Picture N>` in order.
    let is_h3_expand = task == Some(MINIMAX_H3_IMAGE_EXPAND_TASK);
    let carries_reference_images = is_vision_task || is_qwen_rewrite || is_h3_expand;

    let prompt = payload.prompt.trim();
    if prompt.is_empty() && !is_vision_task {
        return Err(ApiError::bad_request("Prompt cannot be empty"));
    }

    let mut job_payload = JsonObject::new();
    if !prompt.is_empty() {
        job_payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));
    }

    if carries_reference_images {
        // A "mood board" (epic 8588, sc-8595) sends several references in `sourceAssetIds`; the worker
        // synthesizes ONE prompt/caption from the aesthetic they share. When that plural list is non-empty
        // it takes precedence over the single `sourceAssetId`; otherwise the single id is the sole
        // reference (the unchanged single-image path). Every id is resolved to a confined on-disk path.
        let asset_ids: Vec<&str> = {
            let plural: Vec<&str> = payload
                .source_asset_ids
                .iter()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .collect();
            if plural.is_empty() {
                payload
                    .source_asset_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .into_iter()
                    .collect()
            } else {
                plural
            }
        };
        // A VISION task has nothing to look at without a reference. A rewrite with none is the
        // text-to-image case and proceeds — so this requirement, and everything below it that needs
        // a project to resolve asset ids against, is skipped when the list is legitimately empty.
        if asset_ids.is_empty() && is_vision_task {
            return Err(ApiError::bad_request(
                "sourceAssetId (or sourceAssetIds) is required for a reference-image task",
            ));
        }
        // Bound the board: each reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so N
        // references cost ~N MP of vision attention + context. Cap it so a runaway list cannot exhaust
        // memory. The UI enforces the same ceiling; this is the server-side guard.
        //
        // The rewrite's ceiling is 10 rather than 6, and for a different reason — see
        // `MAX_QWEN_REWRITE_IMAGES`: it must be shown exactly the ordered list the render will use.
        let (cap, cap_message) = if is_h3_expand {
            (
                MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES,
                format!(
                    "MiniMax-H3-Image prompt expansion accepts at most \
                     {MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES} reference images — the same ordered \
                     list the render takes"
                ),
            )
        } else if is_qwen_rewrite {
            (
                MAX_QWEN_REWRITE_IMAGES,
                format!(
                    "Qwen Image 2.1 prompt rewriting accepts at most {MAX_QWEN_REWRITE_IMAGES} \
                     reference images — the same ordered list the render takes"
                ),
            )
        } else {
            (
                MAX_MOOD_BOARD_IMAGES,
                format!("A mood board accepts at most {MAX_MOOD_BOARD_IMAGES} reference images"),
            )
        };
        if asset_ids.len() > cap {
            return Err(ApiError::bad_request(cap_message));
        }
        if !asset_ids.is_empty() {
            let project_id = payload
                .project_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    ApiError::bad_request("projectId is required for a reference-image task")
                })?;
            let mut image_paths = Vec::with_capacity(asset_ids.len());
            for asset_id in &asset_ids {
                image_paths
                    .push(resolve_image_caption_path(state.clone(), project_id, asset_id).await?);
            }
            // A single reference keeps the scalar `imagePath` (byte-identical to the pre-mood-board path);
            // multiple references ride the `imagePaths` array the worker prefers.
            if image_paths.len() == 1 && !is_qwen_rewrite && !is_h3_expand {
                job_payload.insert(
                    "imagePath".to_owned(),
                    Value::String(image_paths.into_iter().next().unwrap()),
                );
            } else {
                // sc-24113: a rewrite ALWAYS uses the plural key, even for one reference. The worker's
                // `<imageN>` numbering is positional, and a scalar `imagePath` has no position — keeping
                // the array shape is what makes "reference 1" mean the same thing at one reference and
                // at ten.
                job_payload.insert(
                    "imagePaths".to_owned(),
                    Value::Array(image_paths.into_iter().map(Value::String).collect()),
                );
            }
        }
        // The vision model is named by its HF repo string; the worker resolves it by repo (like the
        // refiner), so it must be carried verbatim rather than as a catalog id.
        if let Some(model) = payload
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
        // `image_describe` carries the per-model describe style (prose vs booru tags, sc-8205); forward
        // it verbatim for the worker to parse. Harmless for `image_caption` (the worker ignores it).
        if let Some(caption_style) = payload
            .caption_style
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert(
                "captionStyle".to_owned(),
                Value::String(caption_style.to_owned()),
            );
        }
    }

    // Planning may explicitly opt into a different native checkpoint. Keep this separate from
    // `modelId`, which remains the target VIDEO model whose capabilities shape the requested plan.
    if task == Some("film_plan") {
        if let Some(model) = payload
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
        if let Some(thinking_mode) = payload
            .thinking_mode
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if !matches!(thinking_mode, "disabled" | "enabled" | "auto") {
                return Err(ApiError::bad_request(
                    "thinkingMode must be disabled, enabled, or auto",
                ));
            }
            job_payload.insert(
                "thinkingMode".to_owned(),
                Value::String(thinking_mode.to_owned()),
            );
        }
    }

    // sc-20682: the compressed-KV opt-in rides every prompt-refine task; the worker validates it
    // again, but a typo is refused here rather than as a failed job.
    if let Some(kv_compression) = payload
        .kv_compression
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !matches!(kv_compression, "off" | "qualified") {
            return Err(ApiError::bad_request(
                "kvCompression must be off or qualified",
            ));
        }
        job_payload.insert(
            "kvCompression".to_owned(),
            Value::String(kv_compression.to_owned()),
        );
    }

    let workflow = payload
        .workflow
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("image")
        .to_owned();
    job_payload.insert("workflow".to_owned(), Value::String(workflow));

    if let Some(model_id) = payload.model_id.as_deref() {
        if !model_id.trim().is_empty() {
            job_payload.insert(
                "modelId".to_owned(),
                Value::String(model_id.trim().to_owned()),
            );
        }
    }
    if let Some(guide) = payload.guide.as_deref() {
        if !guide.trim().is_empty() {
            job_payload.insert("guide".to_owned(), Value::String(guide.to_owned()));
        }
    }
    // Magic-prompt expansion (sc-5997): the worker swaps in Ideogram's caption system
    // prompt and the aspect ratio steers its layout/bbox decisions.
    if let Some(task) = payload.task.as_deref() {
        if !task.trim().is_empty() {
            job_payload.insert("task".to_owned(), Value::String(task.trim().to_owned()));
        }
    }
    if let Some(aspect_ratio) = payload.aspect_ratio.as_deref() {
        if !aspect_ratio.trim().is_empty() {
            job_payload.insert(
                "aspectRatio".to_owned(),
                Value::String(aspect_ratio.trim().to_owned()),
            );
        }
    }

    if is_h3_expand {
        prepare_h3_image_expand(&state, &payload, &mut job_payload).await?;
    }

    let job = create_generation_job(
        state,
        JobType::PromptRefine,
        None,
        None,
        job_payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// The MiniMax-H3-Image expansion's own admission (SW-7102): it serves only `minimax_h3_image`,
/// its edit aspect selection is `adaptive` or a positive-integer `W:H`, its licence must be
/// accepted before it runs, and the worker receives its server-owned catalog entry (tiers and the
/// upstream `prompt/` co-requisite) as `modelManifestEntry`.
async fn prepare_h3_image_expand(
    state: &AppState,
    payload: &PromptRefineRequest,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let target = payload
        .model_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(MINIMAX_H3_IMAGE_MODEL_ID);
    if target != MINIMAX_H3_IMAGE_MODEL_ID {
        return Err(ApiError::bad_request(format!(
            "MiniMax-H3-Image prompt expansion only applies to {MINIMAX_H3_IMAGE_MODEL_ID}; this \
             request targets {target}"
        )));
    }
    job_payload.insert(
        "modelId".to_owned(),
        Value::String(MINIMAX_H3_IMAGE_MODEL_ID.to_owned()),
    );
    if let Some(aspect) = payload
        .aspect_ratio
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if aspect != "adaptive" && !is_positive_ratio(aspect) {
            return Err(ApiError::bad_request(
                "aspectRatio for MiniMax-H3-Image prompt expansion must be adaptive or a \
                 positive-integer W:H",
            ));
        }
    }
    let entry =
        crate::models::resolve_model_manifest_entry(state, MINIMAX_H3_IMAGE_PE_MODEL_ID).await?;
    let object = entry
        .as_object()
        .filter(|object| !object.is_empty())
        .ok_or_else(|| {
            ApiError::internal(format!(
                "the {MINIMAX_H3_IMAGE_PE_MODEL_ID} catalog entry is missing"
            ))
        })?;
    crate::models::ensure_run_license_acknowledged(state, object).await?;
    job_payload.insert("modelManifestEntry".to_owned(), entry);
    Ok(())
}

/// Whether `value` is a positive-integer `W:H`.
fn is_positive_ratio(value: &str) -> bool {
    let positive = |part: &str| {
        !part.is_empty()
            && part.bytes().all(|byte| byte.is_ascii_digit())
            && part.parse::<u64>().is_ok_and(|number| number > 0)
    };
    value
        .split_once(':')
        .is_some_and(|(width, height)| positive(width) && positive(height))
}

/// Resolve an `image_caption` reference asset to an absolute on-disk path. Mirrors the worker's
/// `resolve_asset_path`: read the asset record's relative `file.path`, join it under the owning
/// project's directory using only `Normal` path components (rejecting `..`/absolute traversal), and
/// confirm the file exists. The worker independently re-confines this path to an app-managed root
/// before opening it (epic 4484 untrusted-input policy), so this is defence-in-depth, not the sole
/// guard. Returns a 400 for a missing/garbled asset record or a path that escapes the project root.
async fn resolve_image_caption_path(
    state: AppState,
    project_id: &str,
    asset_id: &str,
) -> Result<String, ApiError> {
    let project_path = project_path_for_id(state.clone(), project_id).await?;
    let path =
        resolve_project_confined_asset_path(state, project_id, asset_id, &project_path).await?;
    Ok(path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rewrite's reference ceiling is the RENDER's ceiling, and deliberately not the mood
    /// board's. Asserting both together is what stops a future "unify the caps" cleanup from
    /// silently cutting what the rewriter is allowed to see — at which point its `<imageN>`
    /// numbering would stop matching the references the render conditions on.
    #[test]
    fn the_expansion_aspect_selection_is_a_positive_integer_ratio() {
        for ratio in ["1:1", "16:9", "21:9", "4736:3552"] {
            assert!(is_positive_ratio(ratio), "{ratio}");
        }
        for value in ["", "16", "0:1", "1:0", "16:9:1", "a:b", "1.5:1", "-1:2"] {
            assert!(!is_positive_ratio(value), "{value}");
        }
        assert_eq!(MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES, 16);
    }

    #[test]
    fn the_rewrite_ceiling_is_the_renders_not_the_mood_boards() {
        assert_eq!(MAX_QWEN_REWRITE_IMAGES, 10);
        assert_eq!(MAX_MOOD_BOARD_IMAGES, 6);
    }
}
