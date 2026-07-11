// Candle (Windows/CUDA) in-place ComfyUI Qwen-Image txt2img route (epic 10451 Phase 2b, sc-10670).
// Renders a user's existing ComfyUI Qwen-Image DiT — read in place, no copy, no re-download — via
// `candle_gen_qwen_image::load_from_comfyui_dit`, which strips the `model.diffusion_model.` prefix and
// upcasts the plain `fp8_e4m3fn` DiT to bf16 in memory (candle-gen sc-10670). Unlike the Z-Image lane
// (sc-10668), which read all three components in place, only the **DiT** comes from the ComfyUI tree:
// the tree's Qwen2.5-VL text encoders are themselves scaled-fp8 (sc-10671) and its VAE uses native
// WAN-VAE keys (a separate 3D-VAE remap), so the text encoder / VAE / tokenizer are sourced from a
// resident SceneWorks Qwen-Image snapshot (the same tier tree the registry `qwen_image` path loads).
//
// The model id is an `external_base_*` catalog row (assembled by the API's `external_base_models`);
// its `modelManifestEntry` carries `family:"qwen-image"`, `usable:true`, `quant:"fp8_e4m3"`, and a
// `components[]` list whose `transformer` entry is the DiT path.
//
// **Candle-only**, and a **bespoke provider** (like the Z-Image comfyui lane): the loaded generator is
// not registry-resolvable (its DiT is a single in-place file, not a diffusers snapshot dir), so it is
// loaded fresh per job through `start_gen_stream`. This file is `include!`d into the `image_jobs`
// module, sharing its imports.

/// The adapter/engine id recorded on candle ComfyUI Qwen-Image assets + telemetry (distinct from the
/// registry `candle` qwen txt2img and the `qwen_edit`/`qwen_control` lanes).
const QWEN_COMFYUI_CANDLE_ENGINE: &str = "candle_qwen_image_comfyui";

/// Denoise-steps fallback — the `qwen_image` manifest default (`defaults.steps`). The UI normally
/// supplies `advanced.steps`; this only applies when it does not. Qwen-Image base is a non-distilled
/// 20B flow-match model, so this is a production count (not a few-step distilled one).
const QWEN_COMFYUI_DEFAULT_STEPS: u32 = 20;

/// The shipped SceneWorks Qwen-Image snapshot repo — a per-tier turnkey (`q4/`, `q8/`, `bf16/`), each
/// a complete diffusers tree (`transformer/ text_encoder/ vae/ tokenizer/`). The DiT is read from the
/// ComfyUI tree instead, so only the tier's **dense** Qwen2.5-VL text encoder + VAE + tokenizer are
/// used here — those are byte-identical across tiers (only the transformer is quantized), so any
/// present tier serves.
const QWEN_COMFYUI_SNAPSHOT_REPO: &str = "SceneWorks/qwen-image-mlx";

/// Tier subdirs probed (in order) for the dense TE/VAE/tokenizer. bf16 first (native TE dtype), then
/// the quantized tiers, whose TE/VAE are the same dense bf16 — the first fully-present tree wins, so a
/// partially-downloaded tier does not block the lane.
const QWEN_COMFYUI_SNAPSHOT_TIERS: &[&str] = &["bf16", "q8", "q4"];

/// The in-place ComfyUI DiT file + the resident snapshot tier dir supplying the other components.
struct ComfyuiQwenPaths {
    /// ComfyUI Qwen-Image DiT (`diffusion_models/qwen_image_*_fp8_e4m3fn.safetensors`), read in place.
    transformer: PathBuf,
    /// A resident `SceneWorks/qwen-image-mlx` tier dir (`text_encoder/ vae/ tokenizer/tokenizer.json`).
    snapshot_dir: PathBuf,
}

/// Resolve the ComfyUI Qwen-Image DiT path + the resident snapshot tier from the forwarded
/// `external_base_*` row. Returns `Ok(None)` when this is not a runnable ComfyUI Qwen-Image job (wrong
/// family, not marked usable, no transformer component, or our Qwen-Image snapshot is not resident), so
/// the router falls through rather than erroring. The DiT path is confined by
/// `normalize_app_managed_model_path` (widened to admit the operator's external roots, sc-10668) — a
/// payload can never point it outside a declared root (epic 4484). The snapshot dir is resolved from a
/// fixed repo constant + our own cache (never payload-derived), so it needs no confinement.
fn resolve_qwen_comfyui_paths(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiQwenPaths>> {
    let entry = &request.model_manifest_entry;
    if entry.get("family").and_then(Value::as_str) != Some("qwen-image") {
        return Ok(None);
    }
    // Only render what the API marked runnable (qwen-image plain fp8_e4m3, sc-10670). A non-usable row
    // must not be silently rendered — it is either an unsupported quant (scaled-fp8/gguf) or incomplete.
    if entry.get("usable").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let Some(components) = entry.get("components").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(transformer) = components
        .iter()
        .find(|component| component.get("role").and_then(Value::as_str) == Some("transformer"))
        .and_then(|component| component.get("path").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    // The dense TE/VAE/tokenizer come from a resident Qwen-Image snapshot tier; if none is installed
    // there is nothing to encode/decode with, so this is not (yet) runnable.
    let Some(snapshot_root) =
        huggingface_snapshot_dir(&settings.data_dir, QWEN_COMFYUI_SNAPSHOT_REPO)
    else {
        return Ok(None);
    };
    let Some(snapshot_dir) = QWEN_COMFYUI_SNAPSHOT_TIERS
        .iter()
        .map(|tier| snapshot_root.join(tier))
        .find(|dir| {
            dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
                && dir.join("tokenizer").join("tokenizer.json").is_file()
        })
    else {
        return Ok(None);
    };
    Ok(Some(ComfyuiQwenPaths {
        transformer: crate::paths::normalize_app_managed_model_path(
            settings,
            transformer,
            "ComfyUI Qwen-Image transformer",
        )?,
        snapshot_dir,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Qwen-Image txt2img job: an `external_base_*`
/// model whose forwarded row is a usable qwen-image with a transformer component + a resident snapshot,
/// no source image / pose (txt2img only). Mirrors the Z-Image comfyui availability predicate.
fn qwen_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model.starts_with("external_base_")
        && request.mode != "edit_image"
        && pose_entries(request).is_empty()
        && matches!(resolve_qwen_comfyui_paths(request, settings), Ok(Some(_)))
}

/// Flat telemetry recorded on candle ComfyUI Qwen-Image assets. Qwen-Image base is non-distilled, so
/// unlike the Z-Image comfyui lane it records true-CFG guidance.
fn qwen_comfyui_raw_settings(request: &ImageRequest, steps: u32, guidance: Option<f32>) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("mode".to_owned(), Value::String("text_to_image".to_owned()));
    raw.insert(
        "engine".to_owned(),
        Value::String(QWEN_COMFYUI_CANDLE_ENGINE.to_owned()),
    );
    raw.insert(
        "externalComfyuiBase".to_owned(),
        Value::String(request.model.clone()),
    );
    if let Some(scale) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(scale));
    }
    raw
}

/// Read the requested true-CFG guidance scale from `advanced.guidanceScale` (JSON number or numeric
/// string). `None` ⇒ the candle-gen Qwen-Image engine default (`DEFAULT_GUIDANCE`). The external-base
/// row is in no `MODEL_TABLE`, so `resolve_guidance` (which needs a `ResolvedModel`) does not apply —
/// this reads the same `advanced` knob directly.
fn qwen_comfyui_guidance(request: &ImageRequest) -> Option<f32> {
    request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
}

/// Real candle in-place ComfyUI Qwen-Image txt2img generation: resolve + confine the DiT path and
/// resolve the snapshot tier on the async side, then `load_from_comfyui_dit` once + generate each image
/// on the blocking thread. `request.count` images, each its own seed. Qwen-Image base is non-distilled,
/// so guidance (true CFG) + negative prompt are threaded through. The loaded `Box<dyn Generator>` is
/// bespoke (not registry-cached), driven like the Z-Image comfyui lane.
async fn generate_candle_qwen_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let paths = resolve_qwen_comfyui_paths(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "ComfyUI Qwen-Image components could not be resolved (family/usable/transformer/snapshot)"
                .to_owned(),
        )
    })?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", QWEN_COMFYUI_DEFAULT_STEPS, 1..=50);
    let guidance = qwen_comfyui_guidance(request);
    let negative_prompt = {
        let trimmed = request.negative_prompt.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    };
    let raw_settings = qwen_comfyui_raw_settings(request, steps, guidance);

    // Per-image work items: (seed, prompt) — `request.count` renders.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "qwen_comfyui",
        0,
        move || {
            let ComfyuiQwenPaths {
                transformer,
                snapshot_dir,
            } = paths;
            let model =
                candle_gen_qwen_image::load_from_comfyui_dit(transformer, snapshot_dir).map_err(
                    |error| WorkerError::Engine(format!("ComfyUI Qwen-Image load failed: {error}")),
                )?;
            Ok(model)
        },
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let request = GenerationRequest {
                    prompt,
                    negative_prompt: negative_prompt.clone(),
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps: Some(steps),
                    guidance,
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&request, &mut *on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "ComfyUI Qwen-Image generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine("ComfyUI Qwen-Image produced no image".to_owned())
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "ComfyUI Qwen-Image returned non-image output".to_owned(),
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
        QWEN_COMFYUI_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
