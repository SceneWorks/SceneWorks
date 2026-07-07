use super::*;

/// Maximum reference images in one "mood board" describe/caption request (epic 8588, sc-8595). Each
/// reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so the cost scales with N; this bounds
/// the vision attention + context a single request can demand. The Image Studio picker enforces the same
/// ceiling client-side — this is the authoritative server-side guard.
pub(crate) const MAX_MOOD_BOARD_IMAGES: usize = 6;

/// Enqueue a `prompt_refine` job: a lightweight, non-GPU job that asks an
/// OpenAI-compatible LLM to rewrite the user's prompt to follow the selected
/// model's prompt guide. The job runs in the Python worker (which reuses the
/// vendored Lens reasoner's calling approach) and the client reads the refined
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

    let prompt = payload.prompt.trim();
    if prompt.is_empty() && !is_vision_task {
        return Err(ApiError::bad_request("Prompt cannot be empty"));
    }

    let mut job_payload = JsonObject::new();
    if !prompt.is_empty() {
        job_payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));
    }

    if is_vision_task {
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
        if asset_ids.is_empty() {
            return Err(ApiError::bad_request(
                "sourceAssetId (or sourceAssetIds) is required for a reference-image task",
            ));
        }
        // Bound the board: each reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so N
        // references cost ~N MP of vision attention + context. Cap it so a runaway list cannot exhaust
        // memory. The UI enforces the same ceiling; this is the server-side guard.
        if asset_ids.len() > MAX_MOOD_BOARD_IMAGES {
            return Err(ApiError::bad_request(format!(
                "A mood board accepts at most {MAX_MOOD_BOARD_IMAGES} reference images"
            )));
        }
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
        if image_paths.len() == 1 {
            job_payload.insert(
                "imagePath".to_owned(),
                Value::String(image_paths.into_iter().next().unwrap()),
            );
        } else {
            job_payload.insert(
                "imagePaths".to_owned(),
                Value::Array(image_paths.into_iter().map(Value::String).collect()),
            );
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
    Ok((StatusCode::CREATED, Json(job)))
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
    let project_id_owned = project_id.to_owned();
    let asset_id_owned = asset_id.to_owned();
    let asset = project_call(state, move |store| {
        store.get_asset(&project_id_owned, &asset_id_owned)
    })
    .await?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Reference asset has no file path"))?;
    let mut path = project_path;
    for component in std::path::Path::new(rel).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => {
                return Err(ApiError::bad_request(
                    "Reference asset path must stay inside the project directory",
                ))
            }
        }
    }
    if !path.exists() {
        return Err(ApiError::bad_request("Reference image not found on disk"));
    }
    Ok(path.display().to_string())
}
