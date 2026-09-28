use super::*;

/// Maximum reference images in one "mood board" describe/caption request (epic 8588, sc-8595). Each
/// reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so the cost scales with N; this bounds
/// the vision attention + context a single request can demand. The Image Studio picker enforces the same
/// ceiling client-side — this is the authoritative server-side guard.
pub(crate) const MAX_MOOD_BOARD_IMAGES: usize = 6;

/// The `task` discriminator for Qwen-Image 2.1's official prompt rewriting (sc-24113, epic 24107).
///
/// Spelled here as well as in `crates/sceneworks-worker/src/qwen_prompt_rewrite.rs` because the two
/// crates share no dependency in this direction; `the_rewrite_task_name_matches_the_worker` pins
/// them together.
pub(crate) const QWEN_IMAGE_REWRITE_TASK: &str = "qwen_image_rewrite";

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
    let carries_reference_images = is_vision_task || is_qwen_rewrite;

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
        let (cap, cap_message) = if is_qwen_rewrite {
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
            if image_paths.len() == 1 && !is_qwen_rewrite {
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

    /// The rewrite task name is spelled in THREE places — here, in the worker's
    /// `qwen_prompt_rewrite::REWRITE_TASK`, and in the web client's `qwenRewritePrompt` body — and
    /// the three crates share no dependency that would make a rename propagate.
    ///
    /// A mismatch is silent and total: `RefineTask::from_payload` falls through to the generic
    /// `Rewrite` for any unknown discriminator, so the job would quietly run the Anubis refiner
    /// under the generic rewrite rules instead of Qwen's frozen template, return prose where the
    /// client expects a suggestion object, and surface only as a parse failure with no clue why.
    ///
    /// This pins the API's half against the literal the worker classifies on. The web half is
    /// pinned by the end-to-end route tests in `crate::tests::jobs`, which POST the same string the
    /// client does.
    #[test]
    fn the_rewrite_task_name_matches_the_worker() {
        assert_eq!(QWEN_IMAGE_REWRITE_TASK, "qwen_image_rewrite");
    }

    /// The rewrite's reference ceiling is the RENDER's ceiling, and deliberately not the mood
    /// board's. Asserting both together is what stops a future "unify the caps" cleanup from
    /// silently cutting what the rewriter is allowed to see — at which point its `<imageN>`
    /// numbering would stop matching the references the render conditions on.
    #[test]
    fn the_rewrite_ceiling_is_the_renders_not_the_mood_boards() {
        assert_eq!(MAX_QWEN_REWRITE_IMAGES, 10);
        assert_eq!(MAX_MOOD_BOARD_IMAGES, 6);
    }
}
