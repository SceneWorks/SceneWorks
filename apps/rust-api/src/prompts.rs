use super::*;

/// Maximum reference images in one "mood board" describe/caption request (epic 8588, sc-8595). Each
/// reference is downscaled to ~1 MP before the dense Qwen-VL ViT, so the cost scales with N; this bounds
/// the vision attention + context a single request can demand. The Image Studio picker enforces the same
/// ceiling client-side — this is the authoritative server-side guard.
pub(crate) use sceneworks_core::request_validation::utility_validation::{
    MAX_MOOD_BOARD_IMAGES, MAX_QWEN_REWRITE_IMAGES,
};

// The API and worker classify the same core-owned discriminator.
pub(crate) use sceneworks_core::contracts::QWEN_IMAGE_REWRITE_TASK;
pub(crate) use sceneworks_core::contracts::{
    MINIMAX_H3_IMAGE_EXPAND_MAX_REFERENCES, MINIMAX_H3_IMAGE_EXPAND_TASK,
    MINIMAX_H3_IMAGE_MODEL_ID, MINIMAX_H3_IMAGE_PE_MODEL_ID,
};

/// Enqueue a `prompt_refine` job: a lightweight, non-GPU job that asks an
/// OpenAI-compatible LLM to rewrite the user's prompt to follow the selected
/// model's prompt guide. The job runs through a native TextLlm provider, and the client reads the refined
/// prompt from the completed job's `result.refinedPrompt`.
pub(crate) async fn create_prompt_refine_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<PromptRefineRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let effective = sceneworks_core::request_validation::resolve(
        "promptRefine",
        &to_json_object(&payload)?,
        &JsonObject::new(),
    )
    .map_err(ApiError::from)?;
    // Keep the authored DTO for persistence; use the owner's effective view for decisions.
    let task = effective.get("task").and_then(Value::as_str);
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
            // The native planner is the shared Qwen3.8 27B checkpoint, whose tiers are separate
            // downloads. Freeze the installed tier as `variant` BEFORE admission binds the job's
            // sources — the same request vocabulary the dataset describe job uses — so the source
            // admitted is the tier the worker loads. No tier installed leaves the worker's "not
            // installed" refusal to explain it.
            if model == sceneworks_core::film_workspace::NATIVE_FILM_PLANNER_REPO {
                if let Some(tier) = crate::film_planning::native_planner_tier(&state).await? {
                    job_payload.insert("variant".to_owned(), Value::String(tier.to_owned()));
                }
            }
        }
        if let Some(thinking_mode) = payload
            .thinking_mode
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
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
        job_payload.insert(
            "kvCompression".to_owned(),
            Value::String(kv_compression.to_owned()),
        );
    }

    if let Some(workflow) = effective
        .get("workflow")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        job_payload.insert("workflow".to_owned(), Value::String(workflow.to_owned()));
    }

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

    let source = payload.model.as_deref().map(str::trim);
    if source == Some(sceneworks_core::loop_utility::MODEL_REPO) {
        if !is_vision_task && !matches!(task, None | Some("refine")) {
            return Err(ApiError::bad_request(
                "This prompt task cannot use SenseNova-Loop-9B.",
            ));
        }
        let tier = sceneworks_core::loop_utility::tier(payload.variant.as_deref())
            .map_err(ApiError::bad_request)?;
        crate::loop_utility::ensure_tier_installed(&state, tier).await?;
        job_payload.insert("model".to_owned(), json!(source));
        job_payload.insert("variant".to_owned(), json!(tier));
    } else if payload.variant.is_some() {
        return Err(ApiError::bad_request(
            "This prompt model has no selectable tier.",
        ));
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
    let pe_entry =
        crate::models::resolve_model_manifest_entry(state, MINIMAX_H3_IMAGE_PE_MODEL_ID).await?;
    let pe_object = pe_entry
        .as_object()
        .filter(|object| !object.is_empty())
        .ok_or_else(|| {
            ApiError::internal(format!(
                "the {MINIMAX_H3_IMAGE_PE_MODEL_ID} catalog entry is missing"
            ))
        })?;
    // Both models run MiniMax's licensed prompt files, so the expander entry's acknowledgment
    // gates either choice (SW-7208 AC3).
    crate::models::ensure_run_license_acknowledged(state, pe_object).await?;
    // SW-7208: the model the request names, else the saved choice while it can run here, else the
    // first model that can (MiniMax-H3-Image-PE first) — the web UI's own rule. A named model is
    // never substituted; with nothing runnable the saved choice stands and the worker explains.
    let requested = payload
        .expander_model_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let candidates: Vec<&'static str> = match requested {
        Some(_) => vec![
            sceneworks_core::contracts::minimax_h3_image_expander_model(requested).map_err(
                |unknown| {
                    ApiError::bad_request(format!(
                        "expanderModelId {unknown:?} is not a MiniMax-H3-Image prompt expander; \
                         use {}",
                        sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPANDER_MODEL_IDS
                            .join(" or ")
                    ))
                },
            )?,
        ],
        None => {
            let saved = crate::preferences::saved_h3_image_expander_model(state)
                .await
                .unwrap_or(MINIMAX_H3_IMAGE_PE_MODEL_ID);
            std::iter::once(saved)
                .chain(sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPANDER_MODEL_IDS)
                .fold(Vec::new(), |mut order, id| {
                    if !order.contains(&id) {
                        order.push(id);
                    }
                    order
                })
        }
    };
    // Freeze the tier the worker will load BEFORE admission binds the job's sources. The request
    // names no tier, and portable admission has no implicit expander choice of its own, so without
    // this it would bind the catalog's default download (the unpublished bf16 row on macOS, NVFP4
    // on Windows/Linux) while the worker loaded whichever tier is actually installed.
    let workers = store_call(state.clone(), |store, _timeout| store.list_workers()).await?;
    let nvfp4_worker = can_run_nvfp4_expansion(&workers);
    let mut chosen = None;
    for &candidate in &candidates {
        let entry = if candidate == MINIMAX_H3_IMAGE_PE_MODEL_ID {
            pe_entry.clone()
        } else {
            let describer = crate::models::resolve_model_manifest_entry(state, candidate).await?;
            describer_expansion_entry(describer, &pe_entry)?
        };
        let tier = expander_tier(&entry, candidate, &state.settings.data_dir, nvfp4_worker);
        let runnable = tier.is_some();
        if chosen.is_none() || runnable {
            chosen = Some((candidate, entry, tier));
        }
        if runnable {
            break;
        }
    }
    let (expander, entry, tier) = chosen.expect("at least one expander candidate");
    job_payload.insert(
        sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPANDER_MODEL_KEY.to_owned(),
        Value::String(expander.to_owned()),
    );
    if let Some(tier) = tier {
        job_payload.insert(
            sceneworks_core::contracts::MINIMAX_H3_IMAGE_PE_TIER_KEY.to_owned(),
            Value::String(tier.to_owned()),
        );
    }
    job_payload.insert("modelManifestEntry".to_owned(), entry);
    Ok(())
}

/// The entry an expansion with the abliterated describer runs (SW-7208): the describer's own catalog
/// entry (its tiers and their memory floors) with MiniMax's `prompt/` files from the expander entry's
/// `describer_prompt_expansion` conditional component added as a required co-requisite. The worker
/// reads the prompt files from that row, and admission binds them with the tier, exactly as it binds
/// the expander's own `prompt` co-requisite.
fn describer_expansion_entry(describer: Value, pe_entry: &Value) -> Result<Value, ApiError> {
    let mut entry = describer;
    let prompts = sceneworks_core::model_usage_policy::conditional_component_downloads(
        pe_entry,
        sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPAND_PROMPTS_PURPOSE,
    )
    .map_err(|error| {
        ApiError::internal(format!(
            "the {MINIMAX_H3_IMAGE_PE_MODEL_ID} catalog entry declares no prompt files for the \
             describer: {error}"
        ))
    })?;
    entry
        .as_object_mut()
        .filter(|object| !object.is_empty())
        .and_then(|object| object.get_mut("downloads"))
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ApiError::internal("the describer catalog entry is missing"))?
        .extend(prompts);
    Ok(entry)
}

/// Whether one available worker can run an NVFP4 expansion: it must advertise BOTH the
/// `prompt_refine` provider and the NVFP4 device marker. The marker is derived from GPU compute
/// capability alone, so a worker advertising it may still have no text provider linked.
fn can_run_nvfp4_expansion(workers: &[WorkerSnapshot]) -> bool {
    workers.iter().any(|worker| {
        let advertises = |wanted: &str| {
            worker
                .capabilities
                .iter()
                .any(|capability| capability.as_str() == wanted)
        };
        !matches!(
            worker.status,
            WorkerStatus::Offline | WorkerStatus::Unhealthy
        ) && advertises(JobType::PromptRefine.as_str())
            && advertises(sceneworks_core::contracts::NVFP4_WORKER_CAPABILITY)
    })
}

/// The expander tier an unselected expansion of `model_id` runs: the worker's own order
/// ([`sceneworks_core::contracts::minimax_h3_image_expander_tiers`]) over the tiers installed for
/// this host's platform, with NVFP4 only when a registered worker can run it. A tier is installed
/// when its row's pinned revision holds every file the row names, the snapshot the worker loads,
/// and the prompt files are installed too. `None` when no tier is installed, which leaves the
/// worker's "install the expander" refusal to explain it.
fn expander_tier(
    entry: &Value,
    model_id: &str,
    data_dir: &std::path::Path,
    nvfp4_worker: bool,
) -> Option<&'static str> {
    let mut host = entry.clone();
    crate::models::retain_downloads_for_os(&mut host, std::env::consts::OS);
    sceneworks_core::contracts::minimax_h3_image_expander_tiers(model_id)
        .iter()
        .copied()
        .filter(|tier| *tier != "nvfp4" || nvfp4_worker)
        .find(|tier| crate::models::pinned_variant_installed(&host, tier, data_dir))
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

    /// The shipped expander entry, this host's q4 row, and a writer that stages a file of that
    /// tier folder at a revision of its repository, or a file of the upstream `prompt/` row at
    /// its pinned revision.
    struct ExpanderCache {
        _dir: tempfile::TempDir,
        _hub: sceneworks_core::hf_home::test_support::CacheRootGuard,
        data_dir: std::path::PathBuf,
        entry: Value,
        q4: Value,
    }

    impl ExpanderCache {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let data_dir = dir.path().to_path_buf();
            let hub = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
                data_dir.clone(),
                data_dir.join("hub"),
            );
            let entry = Value::Object(
                sceneworks_core::builtin_manifests::builtin_model_entry("minimax_h3_image_pe")
                    .expect("the expander is a builtin")
                    .clone(),
            );
            let mut host = entry.clone();
            crate::models::retain_downloads_for_os(&mut host, std::env::consts::OS);
            let q4 =
                sceneworks_core::model_artifacts::artifact_selection::model_download_for_variant(
                    &host, "q4",
                )
                .expect("q4 row");
            Self {
                _dir: dir,
                _hub: hub,
                data_dir,
                entry,
                q4,
            }
        }

        fn write(&self, repo: &str, revision: &str, file: &str, body: &str) {
            let path = sceneworks_core::hf_home::huggingface_repo_cache_path(&self.data_dir, repo)
                .unwrap()
                .join("snapshots")
                .join(revision)
                .join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }

        /// A file of the q4 tier folder at `revision`.
        fn tier_file(&self, revision: &str, file: &str, body: &str) {
            let folder = self.q4["files"][0].as_str().unwrap().trim_end_matches("/*");
            self.write(
                self.q4["repo"].as_str().unwrap(),
                revision,
                &format!("{folder}/{file}"),
                body,
            );
        }

        fn pinned(&self) -> String {
            self.q4["revision"].as_str().unwrap().to_owned()
        }

        /// Every file of the upstream `prompt/` co-requisite, at its pinned revision.
        fn stage_prompt_files(&self) {
            let row = self.entry["downloads"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["coRequisite"] == true && row["componentId"] == "prompt")
                .expect("the prompt co-requisite row")
                .clone();
            for file in row["files"].as_array().unwrap() {
                self.write(
                    row["repo"].as_str().unwrap(),
                    row["revision"].as_str().unwrap(),
                    &file.as_str().unwrap().replace('*', "fixture"),
                    "{}",
                );
            }
        }

        fn tier(&self) -> Option<&'static str> {
            expander_tier(
                &self.entry,
                MINIMAX_H3_IMAGE_PE_MODEL_ID,
                &self.data_dir,
                false,
            )
        }
    }

    /// The tier an unselected expansion freezes is installed only at its row's pinned revision:
    /// an older revision of the re-host holding MORE files in the same folder never stands in
    /// for it. In the pinned folder a leftover `config.json` alone is not a tier, and a shard
    /// index naming a missing shard is torn; the index with every shard is installed.
    #[test]
    fn the_expander_tier_is_installed_only_at_its_pinned_revision() {
        let cache = ExpanderCache::new();
        cache.stage_prompt_files();
        // An older revision, `refs/main`, with a complete twelve-file folder of its own.
        let older = "0123456789abcdef0123456789abcdef01234567";
        let root = sceneworks_core::hf_home::huggingface_repo_cache_path(
            &cache.data_dir,
            cache.q4["repo"].as_str().unwrap(),
        )
        .unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs/main"), older).unwrap();
        cache.tier_file(older, "config.json", "{}");
        cache.tier_file(older, "model.safetensors", "");
        for index in 0..10 {
            cache.tier_file(older, &format!("extra-{index}.json"), "{}");
        }
        assert_eq!(cache.tier(), None);

        // The pinned folder holds only its config: no weights, not a tier.
        let pinned = cache.pinned();
        cache.tier_file(&pinned, "config.json", "{}");
        assert_eq!(cache.tier(), None);

        // Its index names two shards with one on disk: torn.
        cache.tier_file(
            &pinned,
            "model.safetensors.index.json",
            &serde_json::json!({ "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }})
            .to_string(),
        );
        cache.tier_file(&pinned, "model-00001-of-00002.safetensors", "");
        assert_eq!(cache.tier(), None);

        cache.tier_file(&pinned, "model-00002-of-00002.safetensors", "");
        assert_eq!(cache.tier(), Some("q4"));
    }

    /// The expander reads its prompt files from the upstream `prompt/` co-requisite at run time,
    /// so a tier whose weights are complete is not installed until those files are too.
    #[test]
    fn the_expander_tier_needs_its_prompt_files() {
        let cache = ExpanderCache::new();
        let pinned = cache.pinned();
        cache.tier_file(&pinned, "config.json", "{}");
        cache.tier_file(&pinned, "model.safetensors", "");
        assert_eq!(cache.tier(), None, "the prompt files are absent");
        assert!(!crate::models::pinned_variant_installed(
            &cache.entry,
            "q4",
            &cache.data_dir
        ));

        cache.stage_prompt_files();
        assert_eq!(cache.tier(), Some("q4"));
    }

    /// NVFP4 is offered to an expansion only when ONE available worker advertises both the
    /// text provider and the NVFP4 marker; the two on different workers, or the marker alone,
    /// never select it.
    #[test]
    fn nvfp4_expansion_needs_one_worker_with_the_provider_and_the_marker() {
        let worker = |id: &str, status: &str, capabilities: &[&str]| -> WorkerSnapshot {
            serde_json::from_value(serde_json::json!({
                "id": id, "gpuId": "0", "status": status, "capabilities": capabilities,
                "loadedModels": [], "registeredAt": "2026-11-15T00:00:00Z",
                "lastSeenAt": "2026-11-15T00:00:00Z",
            }))
            .unwrap()
        };
        let marker_only = worker("a", "idle", &["gpu", "candle", "nvfp4"]);
        let provider_only = worker("b", "idle", &["gpu", "candle", "prompt_refine"]);
        let both = worker("c", "idle", &["gpu", "candle", "prompt_refine", "nvfp4"]);
        assert!(!can_run_nvfp4_expansion(std::slice::from_ref(&marker_only)));
        assert!(!can_run_nvfp4_expansion(&[
            marker_only.clone(),
            provider_only.clone()
        ]));
        assert!(can_run_nvfp4_expansion(&[marker_only, both.clone()]));
        let offline = worker("d", "offline", &["gpu", "candle", "prompt_refine", "nvfp4"]);
        assert!(!can_run_nvfp4_expansion(&[offline, provider_only]));
    }

    #[test]
    fn the_rewrite_ceiling_is_the_renders_not_the_mood_boards() {
        assert_eq!(MAX_QWEN_REWRITE_IMAGES, 10);
        assert_eq!(MAX_MOOD_BOARD_IMAGES, 6);
    }
}
