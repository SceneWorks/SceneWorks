use super::huggingface_snapshot_dir;
use super::{
    consume_gen_events, drive_gen_items, prepare_cached_candle_base_floor,
    resolve_advanced_or_manifest_u32, resolve_seed, start_cached_gen_stream_after_cold_admission,
    ApiClient, ColdLoadAdmission, GenerationOutput, GenerationRequest, ImageRequest, JobSnapshot,
    JsonObject, LoadSpec, Path, PathBuf, PreparedFileDispatch, Settings, Value, WeightsSource,
    WorkerError, WorkerResult,
};
use serde_json::json;

// Candle (Windows/CUDA) in-place ComfyUI Qwen-Image txt2img route (epic 10451 Phase 2b, sc-10670 +
// sc-10830). Renders a user's existing ComfyUI Qwen-Image DiT — read in place, no copy, no
// re-download — via `runtime_cuda::providers::qwen_image::load_from_comfyui_dit`, which strips the
// `model.diffusion_model.` prefix and upcasts the plain `fp8_e4m3fn` DiT to bf16 in memory (candle-gen
// sc-10670). The tree's **VAE** is also read in place when the API folded it into the row (sc-10830:
// native WAN-VAE keys → diffusers remap); when absent it falls back to the snapshot VAE. The tree's
// Qwen2.5-VL text encoders are themselves scaled-fp8 (sc-10671), so the **text encoder** + tokenizer
// are still sourced from a resident SceneWorks Qwen-Image snapshot (the same tier tree the registry
// `qwen_image` path loads).
//
// The model id is an `external_base_*` catalog row (assembled by the API's `external_base_models`);
// its `modelManifestEntry` carries `family:"qwen-image"`, `usable:true`, `quant:"fp8_e4m3"`, and a
// `components[]` list whose `transformer` entry is the DiT path (and, when present, a `vae` entry is
// the tree VAE path read in place).
//
// **Candle-only**, but the imported assembly now uses the registered `qwen_image` provider and its
// normal cache/fit/residency lifecycle (sc-18306).

/// The adapter/engine id recorded on candle ComfyUI Qwen-Image assets + telemetry (distinct from the
/// registry `candle` qwen txt2img and the `qwen_edit`/`qwen_control` lanes).
pub(super) const QWEN_COMFYUI_CANDLE_ENGINE: &str = "candle_qwen_image_comfyui";

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

/// The in-place ComfyUI DiT file + the resident snapshot tier dir supplying the other components,
/// plus the optional in-place tree VAE (sc-10830).
pub(super) struct ComfyuiQwenPaths {
    /// ComfyUI Qwen-Image DiT (`diffusion_models/qwen_image_*_fp8_e4m3fn.safetensors`), read in place.
    transformer: gen_core::PinnedWeightsFile,
    /// A resident `SceneWorks/qwen-image-mlx` tier dir (`text_encoder/ vae/ tokenizer/tokenizer.json`).
    snapshot_dir: PathBuf,
    /// The tree's `vae/qwen_image_vae.safetensors` (native WAN-VAE keys), read in place when the API
    /// folded it into the row (sc-10830). `None` ⇒ the snapshot tier's VAE. The TE + tokenizer always
    /// come from the snapshot (the tree TE is scaled-fp8, sc-10671).
    vae: Option<gen_core::PinnedWeightsFile>,
}

#[cfg(test)]
impl ComfyuiQwenPaths {
    /// Exact payload-selected File tokens retained by the production route, in provider-spec order.
    pub(super) fn prepared_file_pins(&self) -> Vec<&gen_core::PinnedWeightsFile> {
        std::iter::once(&self.transformer)
            .chain(self.vae.iter())
            .collect()
    }
}

/// Resolve the ComfyUI Qwen-Image DiT path + the resident snapshot tier from the forwarded
/// `external_base_*` row. Returns `Ok(None)` when this is not a runnable ComfyUI Qwen-Image job (wrong
/// family, not marked usable, no transformer component, or our Qwen-Image snapshot is not resident), so
/// the router falls through rather than erroring. The DiT path is confined by
/// `normalize_app_managed_model_file_path`: its canonical target must stay within a declared root while
/// the lexical entry survives for lstat-pinned re-opening (sc-18306). The snapshot dir is resolved from
/// a fixed repo constant + our own cache (never payload-derived), so it needs no confinement.
pub(super) fn resolve_qwen_comfyui_paths(
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
    // sc-10830: the tree's VAE (native WAN-VAE keys), read in place when the API folded it into the
    // row as a `vae` component. Optional — absent ⇒ the snapshot tier's VAE. Confined by the same
    // external-root normalization as the DiT (a payload can never point it outside a declared root).
    let vae = components
        .iter()
        .find(|component| component.get("role").and_then(Value::as_str) == Some("vae"))
        .and_then(|component| component.get("path").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            crate::paths::pin_app_managed_model_file(
                settings,
                Path::new(value),
                "ComfyUI Qwen-Image VAE",
            )
        })
        .transpose()?;
    Ok(Some(ComfyuiQwenPaths {
        transformer: crate::paths::pin_app_managed_model_file(
            settings,
            Path::new(transformer),
            "ComfyUI Qwen-Image transformer",
        )?,
        snapshot_dir,
        vae,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Qwen-Image txt2img job: an `external_base_*`
/// model whose forwarded row is a usable qwen-image with a transformer component + a resident snapshot,
/// no source image / pose (txt2img only). Mirrors the Z-Image comfyui availability predicate.
#[cfg(test)]
pub(super) fn qwen_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    matches!(prepare_qwen_comfyui_sources(request, settings), Ok(Some(_)))
}

/// Resolve and pin the complete source bundle for a production dispatch. The returned bundle is
/// retained across the async preamble and consumed by the handler, closing the selection-to-load
/// retarget window for the payload-selected transformer entry.
pub(super) fn prepare_qwen_comfyui_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiQwenPaths>> {
    let descriptor = super::imported_generate_request_supported(
        request,
        "qwen-image",
        gen_core::ImportedModelSource::ComfyUiTree,
    );
    if !request.model.starts_with("external_base_")
        || descriptor.as_ref().map_or(true, |descriptor| {
            super::imported_model_quant(request, descriptor, "ComfyUI Qwen-Image").is_err()
        })
    {
        return Ok(None);
    }
    resolve_qwen_comfyui_paths(request, settings)
}

/// Flat telemetry recorded on candle ComfyUI Qwen-Image assets. Qwen-Image base is non-distilled, so
/// unlike the Z-Image comfyui lane it records true-CFG guidance.
fn qwen_comfyui_raw_settings(
    request: &ImageRequest,
    steps: u32,
    guidance: Option<f32>,
) -> JsonObject {
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

/// Finalize the route-owned File tokens on the exact provider spec and retain the structural snapshot
/// facts admission needs. This is the production selection-to-load boundary used by regression tests.
pub(super) fn prepare_qwen_comfyui_load_spec(
    paths: ComfyuiQwenPaths,
) -> WorkerResult<(LoadSpec, PathBuf, bool)> {
    let snapshot_dir = paths.snapshot_dir;
    let has_imported_vae = paths.vae.is_some();
    let mut spec = LoadSpec::new(WeightsSource::File(
        paths.transformer.loader_path().to_path_buf(),
    ))
    .with_component(
        gen_core::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(snapshot_dir.clone()),
    );
    if let Some(vae) = paths.vae.as_ref() {
        spec = spec.with_component(
            gen_core::COMFYUI_VAE_COMPONENT,
            WeightsSource::File(vae.loader_path().to_path_buf()),
        );
    }
    let mut pins = vec![paths.transformer];
    pins.extend(paths.vae);
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        pins,
        "ComfyUI Qwen-Image source preparation failed",
    )?;
    Ok((spec, snapshot_dir, has_imported_vae))
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

/// Real candle in-place ComfyUI Qwen-Image txt2img generation through the registered `qwen_image`
/// provider. `request.count` images, each its own seed. Qwen-Image base is non-distilled, so guidance
/// (true CFG) + negative prompt are threaded through.
pub(super) async fn generate_candle_qwen_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, ComfyuiQwenPaths>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch {
        plan,
        sources: paths,
    } = dispatch;
    let request = &plan.request;
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "qwen-image",
        gen_core::ImportedModelSource::ComfyUiTree,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::Engine(
            "this runtime has no registered ComfyUI Qwen-Image generate route".to_owned(),
        )
    })?;
    let (spec, snapshot_dir, has_imported_vae) = prepare_qwen_comfyui_load_spec(paths)?;
    // Price finalized File tokens plus only the companion directories the provider consumes. The
    // snapshot transformer is replaced; an imported VAE also replaces the snapshot VAE.
    let snapshot_text_encoder = snapshot_dir.join("text_encoder");
    let snapshot_vae = snapshot_dir.join("vae");
    let mut companion_paths = vec![snapshot_text_encoder];
    if !has_imported_vae {
        companion_paths.push(snapshot_vae);
    }
    let companion_refs = companion_paths
        .iter()
        .map(PathBuf::as_path)
        .collect::<Vec<_>>();
    let cold_admission = prepare_cached_candle_base_floor(
        &request.model,
        "ComfyUI Qwen-Image",
        settings,
        &spec,
        &companion_refs,
    )?;

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
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();

    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission(
        job.id.clone(),
        descriptor.id,
        0,
        spec,
        "ComfyUI Qwen-Image load failed".to_owned(),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, tx, cancel| {
            drive_gen_items(
                tx,
                work,
                move |_index, (seed, prompt), preview, on_progress| {
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
                        preview,
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
                                WorkerError::Engine(
                                    "ComfyUI Qwen-Image produced no image".to_owned(),
                                )
                            })?;
                            Ok(Some((seed, image.width, image.height, image.pixels)))
                        }
                        _ => Err(WorkerError::Engine(
                            "ComfyUI Qwen-Image returned non-image output".to_owned(),
                        )),
                    }
                },
            )
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
