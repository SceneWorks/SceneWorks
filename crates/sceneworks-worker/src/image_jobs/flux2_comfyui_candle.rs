// Candle (Windows/CUDA) in-place ComfyUI FLUX.2-dev txt2img route (epic 10451 Phase 2e, sc-10680).
// Renders a user's existing ComfyUI FLUX.2-dev fp8-mixed DiT — read in place, no copy, no re-download —
// via `candle_gen_flux2::load_from_comfyui_dit`, which dequants the inline-scale fp8 MLPs
// (`w = w_fp8·weight_scale`, dropping the `.input_scale` activation scale) and remaps the BFL-native
// keys onto the diffusers schema in memory (candle-gen sc-10680). The 32B DiT does not fit the GPU
// dense after the fp8→f32 dequant, so each projection is folded onto the GPU (Q8), matching the resident
// FLUX.2-dev quant path. The single DiT file carries NO text encoder / VAE / tokenizer, so the Mistral-3
// TE + AutoencoderKL-Flux2 + tokenizer come from a resident SceneWorks FLUX.2-dev snapshot tier.
//
// The model id is an `external_base_*` catalog row (assembled by the API's `external_base_models`); its
// `modelManifestEntry` carries `family:"flux2"`, `usable:true`, `quant:"fp8_inline_scale"`, and a
// `components[]` list whose `transformer` entry is the DiT path.
//
// **Candle-only**, and a **bespoke provider** (like the Qwen-Image comfyui lane): the loaded generator
// is not registry-resolvable (its DiT is a single in-place file, not a diffusers snapshot dir), so it is
// loaded fresh per job through `start_gen_stream`. This file is `include!`d into the `image_jobs`
// module, sharing its imports.

/// The adapter/engine id recorded on candle ComfyUI FLUX.2-dev assets + telemetry (distinct from the
/// registry `candle` flux2 txt2img and the `flux2_dev` edit/control lanes).
const FLUX2_COMFYUI_CANDLE_ENGINE: &str = "candle_flux2_dev_comfyui";

/// Denoise-steps fallback — the `flux2_dev` manifest default (`defaults.steps`). The UI normally
/// supplies `advanced.steps`; this only applies when it does not. FLUX.2-dev is guidance-distilled (not
/// few-step), so this is a production count.
const FLUX2_COMFYUI_DEFAULT_STEPS: u32 = 28;

/// The shipped SceneWorks FLUX.2-dev snapshot repo — per-tier subdirs (`q4/`, `q8/`, `bf16/`), each a
/// complete tree (`transformer/ text_encoder/ vae/ tokenizer/`). The DiT is read from the ComfyUI tree
/// instead, so only the tier's Mistral-3 text encoder + VAE + tokenizer are used here.
const FLUX2_COMFYUI_SNAPSHOT_REPO: &str = "SceneWorks/flux2-dev-mlx";

/// Tier subdirs probed (in order) for the TE/VAE/tokenizer. bf16 first — its Mistral-3 text encoder is a
/// **dense** diffusers tree the candle loader reads directly; the packed q8/q4 tiers are the fallback
/// (the candle loader's packed path consumes them). The first fully-present tree wins, so a
/// partially-downloaded tier does not block the lane.
const FLUX2_COMFYUI_SNAPSHOT_TIERS: &[&str] = &["bf16", "q8", "q4"];

/// The compute quant the dequanted DiT (and the snapshot Mistral TE) are folded onto the GPU at. The 32B
/// dev does not fit the GPU dense after the fp8→f32 dequant, so a quant is required; Q8 preserves the
/// ~8-bit fp8 source and fits a 96 GB card (~24 GB TE + ~32 GB DiT + VAE). An `advanced.quant` of `q4`
/// selects the smaller/faster tier for tighter cards.
const FLUX2_COMFYUI_DEFAULT_QUANT: Quant = Quant::Q8;

/// The in-place ComfyUI DiT file + the resident snapshot tier dir supplying the other components.
struct ComfyuiFlux2Paths {
    /// ComfyUI FLUX.2-dev DiT (`diffusion_models/flux2_dev_fp8mixed.safetensors`), read in place.
    transformer: PathBuf,
    /// A resident `SceneWorks/flux2-dev-mlx` tier dir (`text_encoder/ vae/ tokenizer/tokenizer.json`).
    snapshot_dir: PathBuf,
}

/// Resolve the ComfyUI FLUX.2-dev DiT path + the resident snapshot tier from the forwarded
/// `external_base_*` row. Returns `Ok(None)` when this is not a runnable ComfyUI FLUX.2-dev job (wrong
/// family, not marked usable, no transformer component, or our FLUX.2-dev snapshot is not resident), so
/// the router falls through rather than erroring. The DiT path is confined by
/// `normalize_app_managed_model_path` (widened to admit the operator's external roots, sc-10668) — a
/// payload can never point it outside a declared root (epic 4484). The snapshot dir is resolved from a
/// fixed repo constant + our own cache (never payload-derived), so it needs no confinement.
fn resolve_flux2_comfyui_paths(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiFlux2Paths>> {
    let entry = &request.model_manifest_entry;
    if entry.get("family").and_then(Value::as_str) != Some("flux2") {
        return Ok(None);
    }
    // Only render what the API marked runnable (flux2 inline-scale fp8, sc-10680). A non-usable row must
    // not be silently rendered — it is either an unsupported quant (comfy_quant fp4 / gguf) or incomplete.
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
    // The Mistral-3 TE / VAE / tokenizer come from a resident FLUX.2-dev snapshot tier; if none is
    // installed there is nothing to encode/decode with, so this is not (yet) runnable.
    let Some(snapshot_root) =
        huggingface_snapshot_dir(&settings.data_dir, FLUX2_COMFYUI_SNAPSHOT_REPO)
    else {
        return Ok(None);
    };
    let Some(snapshot_dir) = FLUX2_COMFYUI_SNAPSHOT_TIERS
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
    Ok(Some(ComfyuiFlux2Paths {
        transformer: crate::paths::normalize_app_managed_model_path(
            settings,
            transformer,
            "ComfyUI FLUX.2-dev transformer",
        )?,
        snapshot_dir,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI FLUX.2-dev txt2img job: an `external_base_*`
/// model whose forwarded row is a usable flux2 with a transformer component + a resident snapshot, no
/// source image / pose (txt2img only). Mirrors the Qwen-Image comfyui availability predicate.
fn flux2_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model.starts_with("external_base_")
        && request.mode != "edit_image"
        && pose_entries(request).is_empty()
        && matches!(resolve_flux2_comfyui_paths(request, settings), Ok(Some(_)))
}

/// The compute quant the DiT + snapshot TE fold onto the GPU at: `advanced.quant` (`q4`/`q8`), else the
/// [`FLUX2_COMFYUI_DEFAULT_QUANT`]. The 32B dev needs a quant to fit; there is no dense option.
fn flux2_comfyui_quant(request: &ImageRequest) -> Quant {
    match request
        .advanced
        .get("quant")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("q4") => Quant::Q4,
        Some("q8") => Quant::Q8,
        _ => FLUX2_COMFYUI_DEFAULT_QUANT,
    }
}

/// Read the requested embedded-guidance scale from `advanced.guidanceScale` (JSON number or numeric
/// string). `None` ⇒ the candle-gen FLUX.2-dev engine default. The external-base row is in no
/// `MODEL_TABLE`, so `resolve_guidance` (which needs a `ResolvedModel`) does not apply — this reads the
/// same `advanced` knob directly. FLUX.2-dev is guidance-distilled (embedded scalar, no negative pass).
fn flux2_comfyui_guidance(request: &ImageRequest) -> Option<f32> {
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

/// Flat telemetry recorded on candle ComfyUI FLUX.2-dev assets.
fn flux2_comfyui_raw_settings(
    request: &ImageRequest,
    steps: u32,
    guidance: Option<f32>,
    quant: Quant,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("mode".to_owned(), Value::String("text_to_image".to_owned()));
    raw.insert(
        "engine".to_owned(),
        Value::String(FLUX2_COMFYUI_CANDLE_ENGINE.to_owned()),
    );
    raw.insert(
        "externalComfyuiBase".to_owned(),
        Value::String(request.model.clone()),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        json!(match quant {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
        }),
    );
    if let Some(scale) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(scale));
    }
    raw
}

/// Real candle in-place ComfyUI FLUX.2-dev txt2img generation: resolve + confine the DiT path and
/// resolve the snapshot tier on the async side, then `load_from_comfyui_dit` once + generate each image
/// on the blocking thread. `request.count` images, each its own seed. FLUX.2-dev is guidance-distilled
/// (embedded scalar, single forward — NO negative prompt / true-CFG pass). The loaded `Box<dyn
/// Generator>` is bespoke (not registry-cached), driven like the Qwen-Image comfyui lane.
async fn generate_candle_flux2_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let paths = resolve_flux2_comfyui_paths(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "ComfyUI FLUX.2-dev components could not be resolved (family/usable/transformer/snapshot)"
                .to_owned(),
        )
    })?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", FLUX2_COMFYUI_DEFAULT_STEPS, 1..=50);
    let guidance = flux2_comfyui_guidance(request);
    let quant = flux2_comfyui_quant(request);
    let raw_settings = flux2_comfyui_raw_settings(request, steps, guidance, quant);

    // Per-image work items: (seed, prompt) — `request.count` renders.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "flux2_comfyui",
        0,
        move || {
            let ComfyuiFlux2Paths {
                transformer,
                snapshot_dir,
            } = paths;
            let model =
                candle_gen_flux2::load_from_comfyui_dit(transformer, snapshot_dir, Some(quant))
                    .map_err(|error| {
                        WorkerError::Engine(format!("ComfyUI FLUX.2-dev load failed: {error}"))
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
                    guidance,
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&request, &mut *on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "ComfyUI FLUX.2-dev generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine("ComfyUI FLUX.2-dev produced no image".to_owned())
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "ComfyUI FLUX.2-dev returned non-image output".to_owned(),
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
        FLUX2_COMFYUI_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
