// Candle (Windows/CUDA) in-place ComfyUI Z-Image txt2img route (epic 10451 Phase 2, sc-10668). Renders
// a user's existing ComfyUI Z-Image base weights — read in place, no copy, no re-download — via
// `candle_gen_z_image::load_from_comfyui_components`, which remaps the ComfyUI-native DiT + VAE keys and
// loads the Qwen3 text encoder verbatim (candle-gen #384). The model id is an `external_base_*` catalog
// row (assembled by the API's `external_base_models`); its `modelManifestEntry` carries `family:"z-image"`
// and a `components[]` list of {role, path} for the DiT / text-encoder / VAE.
//
// **Candle-only**, and a **bespoke provider** (like `ZImageEdit`/`ZImageControl`): the loaded generator is
// not registry-resolvable (its weights are three separate in-place files, not a diffusers snapshot dir),
// so it is loaded fresh per job through `start_gen_stream` rather than the cached registry path. This file
// is `include!`d into the `image_jobs` module, sharing its imports.

/// The adapter/engine id recorded on candle ComfyUI Z-Image assets + telemetry (distinct from the
/// registry `candle_z_image` txt2img and the `candle_zimage_edit`/`_control` lanes).
const ZIMAGE_COMFYUI_CANDLE_ENGINE: &str = "candle_zimage_comfyui";
/// Denoise-steps fallback — the Z-Image-Turbo manifest default (`z_image_turbo` `defaults.steps`). The UI
/// normally supplies `advanced.steps`; this only applies when it does not.
const ZIMAGE_COMFYUI_DEFAULT_STEPS: u32 = 8;
/// The shipped Z-Image **diffusers** snapshot — the source of the one tiny file a ComfyUI tree does not
/// ship, `tokenizer/tokenizer.json` (scheduler / geometry / tokenizer-policy are compiled into
/// candle-gen). Uses the diffusers repo (same as the candle edit lane's `ZIMAGE_EDIT_CANDLE_DEFAULT_REPO`)
/// because it lays the tokenizer at `tokenizer/tokenizer.json` under the snapshot root, which is what
/// candle-gen's `build_tokenizer` reads — the `SceneWorks/z-image-turbo-mlx` MLX tier nests it under a
/// per-tier subdir (`bf16/tokenizer/…`) instead.
const ZIMAGE_COMFYUI_TOKENIZER_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";

/// The three in-place ComfyUI component files + our tokenizer dir, all confined.
struct ComfyuiZImagePaths {
    transformer: PathBuf,
    text_encoder: PathBuf,
    vae: PathBuf,
    tokenizer_dir: PathBuf,
}

/// Resolve the ComfyUI Z-Image component paths from the forwarded `external_base_*` row. Returns
/// `Ok(None)` when this is not a runnable ComfyUI Z-Image job (wrong family, missing a component, or our
/// tokenizer snapshot is not resident), so the router falls through rather than erroring. Each component
/// path is confined by `normalize_app_managed_model_path` (widened to admit the operator's external roots,
/// sc-10668) — a payload can never point a component outside a declared root (epic 4484).
fn resolve_zimage_comfyui_paths(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiZImagePaths>> {
    let entry = &request.model_manifest_entry;
    if entry.get("family").and_then(Value::as_str) != Some("z-image") {
        return Ok(None);
    }
    // Only render what the API marked runnable (z-image bf16, complete assembly). A non-usable row must
    // not be silently rendered — it is either an unsupported quant or an incomplete assembly.
    if entry.get("usable").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let Some(components) = entry.get("components").and_then(Value::as_array) else {
        return Ok(None);
    };
    let path_for = |role: &str| -> Option<&str> {
        components
            .iter()
            .find(|component| component.get("role").and_then(Value::as_str) == Some(role))
            .and_then(|component| component.get("path").and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let (Some(transformer), Some(text_encoder), Some(vae)) = (
        path_for("transformer"),
        path_for("text_encoder"),
        path_for("vae"),
    ) else {
        return Ok(None);
    };
    // Our tokenizer.json must be resident; if the Z-Image snapshot was never installed there is nothing
    // to tokenize with, so this is not (yet) runnable.
    let Some(tokenizer_dir) =
        huggingface_snapshot_dir(&settings.data_dir, ZIMAGE_COMFYUI_TOKENIZER_REPO)
    else {
        return Ok(None);
    };
    Ok(Some(ComfyuiZImagePaths {
        transformer: crate::paths::normalize_app_managed_model_path(
            settings,
            transformer,
            "ComfyUI Z-Image transformer",
        )?,
        text_encoder: crate::paths::normalize_app_managed_model_path(
            settings,
            text_encoder,
            "ComfyUI Z-Image text encoder",
        )?,
        vae: crate::paths::normalize_app_managed_model_path(
            settings,
            vae,
            "ComfyUI Z-Image VAE",
        )?,
        tokenizer_dir,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Z-Image txt2img job: an `external_base_*` model
/// whose forwarded row is a usable z-image with all three component paths, no source image / pose (the
/// candle Z-Image comfyui lane is txt2img only). Mirrors the edit/control availability predicates.
fn zimage_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model.starts_with("external_base_")
        && request.mode != "edit_image"
        && pose_entries(request).is_empty()
        && matches!(resolve_zimage_comfyui_paths(request, settings), Ok(Some(_)))
}

/// Flat telemetry recorded on candle ComfyUI Z-Image assets. No guidance — Z-Image-Turbo is distilled.
fn zimage_comfyui_raw_settings(request: &ImageRequest, steps: u32) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("mode".to_owned(), Value::String("text_to_image".to_owned()));
    raw.insert(
        "engine".to_owned(),
        Value::String(ZIMAGE_COMFYUI_CANDLE_ENGINE.to_owned()),
    );
    raw.insert(
        "externalComfyuiBase".to_owned(),
        Value::String(request.model.clone()),
    );
    raw
}

/// Real candle in-place ComfyUI Z-Image txt2img generation: resolve + confine the three component paths on
/// the async side, then load `load_from_comfyui_components` once + generate each image on the blocking
/// thread. `request.count` images, each its own seed. Z-Image-Turbo is distilled (no CFG / negative
/// prompt). The loaded `Box<dyn Generator>` is bespoke (not registry-cached), driven like the edit lane.
async fn generate_candle_zimage_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let paths = resolve_zimage_comfyui_paths(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "ComfyUI Z-Image components could not be resolved (family/usable/components)".to_owned(),
        )
    })?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", ZIMAGE_COMFYUI_DEFAULT_STEPS, 1..=50);
    let raw_settings = zimage_comfyui_raw_settings(request, steps);

    // Per-image work items: (seed, prompt) — `request.count` renders.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "zimage_comfyui",
        0,
        move || {
            let ComfyuiZImagePaths {
                transformer,
                text_encoder,
                vae,
                tokenizer_dir,
            } = paths;
            let model = candle_gen_z_image::load_from_comfyui_components(
                transformer,
                text_encoder,
                vae,
                tokenizer_dir,
            )
            .map_err(|error| {
                WorkerError::Engine(format!("ComfyUI Z-Image load failed: {error}"))
            })?;
            Ok(model)
        },
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let request = GenerationRequest {
                    prompt,
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&request, &mut *on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "ComfyUI Z-Image generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine("ComfyUI Z-Image produced no image".to_owned())
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "ComfyUI Z-Image returned non-image output".to_owned(),
                    )),
                }
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        ZIMAGE_COMFYUI_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
