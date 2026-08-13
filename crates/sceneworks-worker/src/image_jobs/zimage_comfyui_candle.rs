use super::huggingface_snapshot_dir;
use super::{
    consume_gen_events, drive_gen_items, has_authored_text_encoder,
    prepare_cached_candle_base_floor, prepare_manifest_text_encoder_with_file_pins,
    resolve_advanced_or_manifest_u32, resolve_seed, start_cached_gen_stream_after_cold_admission,
    ApiClient, ColdLoadAdmission, Conditioning, GenerationOutput, GenerationRequest, ImageRequest,
    JobSnapshot, JsonObject, LoadSpec, Path, PathBuf, PreparedAdapters, PreparedFileDispatch,
    Settings, Value, WeightsSource, WorkerError, WorkerResult,
};
use serde_json::json;

// Candle (Windows/CUDA) in-place ComfyUI Z-Image txt2img route (epic 10451 Phase 2, sc-10668). Renders
// a user's existing ComfyUI Z-Image base weights — read in place, no copy, no re-download — via
// `runtime_cuda::providers::z_image::load_from_comfyui_components`, which remaps the ComfyUI-native DiT + VAE keys and
// loads the Qwen3 text encoder verbatim (candle-gen #384). The model id is an `external_base_*` catalog
// row (assembled by the API's `external_base_models`); its `modelManifestEntry` carries `family:"z-image"`
// and a `components[]` list of {role, path} for the DiT / text-encoder / VAE.
//
// **Candle-only**, but no longer a bespoke provider: the imported files form the registered
// `z_image_turbo` provider's normal `LoadSpec`, so they share cache, fit-gate, residency, and future
// planner behavior with snapshot-backed jobs.

/// The adapter/engine id recorded on candle ComfyUI Z-Image assets + telemetry (distinct from the
/// registry `candle_z_image` txt2img and the `candle_zimage_edit`/`_control` lanes).
pub(super) const ZIMAGE_COMFYUI_CANDLE_ENGINE: &str = "candle_zimage_comfyui";
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
pub(super) struct ComfyuiZImagePaths {
    transformer: gen_core::PinnedWeightsFile,
    text_encoder: gen_core::PinnedWeightsFile,
    vae: gen_core::PinnedWeightsFile,
    tokenizer_dir: PathBuf,
    adapters: PreparedAdapters,
}

#[cfg(test)]
impl ComfyuiZImagePaths {
    /// Exact payload-selected File tokens retained by the production route, in provider-spec order.
    pub(super) fn prepared_file_pins(&self) -> [&gen_core::PinnedWeightsFile; 3] {
        [&self.transformer, &self.text_encoder, &self.vae]
    }
}

/// Resolve the ComfyUI Z-Image component paths from the forwarded `external_base_*` row. Returns
/// `Ok(None)` when this is not a runnable ComfyUI Z-Image job (wrong family, missing a component, or our
/// tokenizer snapshot is not resident), so the router falls through rather than erroring. Each component
/// path is confined by `normalize_app_managed_model_file_path` (widened to admit the operator's
/// external roots, sc-10668) — a payload can never point a component outside a declared root, while
/// the lexical path is retained for lstat-pinned re-opening (sc-18306).
pub(super) fn resolve_zimage_comfyui_paths(
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
        transformer: crate::paths::pin_app_managed_model_file(
            settings,
            Path::new(transformer),
            "ComfyUI Z-Image transformer",
        )?,
        text_encoder: crate::paths::pin_app_managed_model_file(
            settings,
            Path::new(text_encoder),
            "ComfyUI Z-Image text encoder",
        )?,
        vae: crate::paths::pin_app_managed_model_file(
            settings,
            Path::new(vae),
            "ComfyUI Z-Image VAE",
        )?,
        tokenizer_dir,
        adapters: PreparedAdapters {
            specs: Vec::new(),
            pins: Vec::new(),
        },
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Z-Image txt2img job: an `external_base_*` model
/// whose forwarded row is a usable z-image with all three component paths, no source image / pose (the
/// candle Z-Image comfyui lane is txt2img only). Mirrors the edit/control availability predicates.
#[cfg(test)]
pub(super) fn zimage_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    matches!(
        prepare_zimage_comfyui_sources(request, settings),
        Ok(Some(_))
    )
}

/// Resolve and pin the complete source bundle for a production dispatch. Unlike the boolean routing
/// probe, the caller keeps this value alive across the async job preamble and moves it into the
/// handler, so the lexical component entries cannot be re-selected after admission.
pub(super) fn prepare_zimage_comfyui_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiZImagePaths>> {
    let descriptor = super::imported_generate_request_supported(
        request,
        "z-image",
        gen_core::ImportedModelSource::ComfyUiTree,
    );
    if !request.model.starts_with("external_base_")
        || descriptor.as_ref().map_or(true, |descriptor| {
            super::imported_model_quant(request, descriptor, "ComfyUI Z-Image").is_err()
        })
    {
        return Ok(None);
    }
    let Some(mut paths) = resolve_zimage_comfyui_paths(request, settings)? else {
        return Ok(None);
    };
    paths.adapters = super::resolve_prepared_adapters(request, settings)?;
    Ok(Some(paths))
}

/// Flat telemetry recorded on candle ComfyUI Z-Image assets. No guidance — Z-Image-Turbo is distilled.
fn zimage_comfyui_raw_settings(
    request: &ImageRequest,
    steps: u32,
    adapter_count: usize,
) -> JsonObject {
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
    raw.insert("adapterCount".to_owned(), json!(adapter_count));
    raw
}

/// A selected encoder replaces the legacy ComfyUI text-encoder component (the provider rejects dual
/// sources); default/absence keeps the imported encoder byte-for-byte. All still-consumed File tokens
/// and the contract receipt are finalized atomically on the spec used by admission and load.
pub(super) fn prepare_zimage_comfyui_load_spec_for_request(
    paths: ComfyuiZImagePaths,
    request: &ImageRequest,
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<LoadSpec> {
    let PreparedAdapters {
        specs: adapters,
        pins: adapter_pins,
    } = paths.adapters;
    let selected = has_authored_text_encoder(request)?;
    let mut spec = LoadSpec::new(WeightsSource::File(
        paths.transformer.loader_path().to_path_buf(),
    ))
    .with_component(
        gen_core::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(paths.tokenizer_dir),
    )
    .with_component(
        gen_core::COMFYUI_VAE_COMPONENT,
        WeightsSource::File(paths.vae.loader_path().to_path_buf()),
    );
    if !selected {
        spec = spec.with_component(
            gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
            WeightsSource::File(paths.text_encoder.loader_path().to_path_buf()),
        );
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let mut pins = vec![paths.transformer];
    if !selected {
        pins.push(paths.text_encoder);
    }
    pins.push(paths.vae);
    pins.extend(adapter_pins);
    prepare_manifest_text_encoder_with_file_pins(
        spec,
        engine_id,
        request,
        settings,
        pins,
        "ComfyUI Z-Image source preparation failed",
    )
}

/// Real candle in-place ComfyUI Z-Image txt2img generation through the registered `z_image_turbo`
/// provider. `request.count` images, each its own seed. Z-Image-Turbo is distilled (no CFG / negative
/// prompt).
pub(super) async fn generate_candle_zimage_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, ComfyuiZImagePaths>,
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
        "z-image",
        gen_core::ImportedModelSource::ComfyUiTree,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::Engine(
            "this runtime has no registered ComfyUI Z-Image generate route".to_owned(),
        )
    })?;
    let adapter_count = paths.adapters.specs.len();
    let spec =
        prepare_zimage_comfyui_load_spec_for_request(paths, request, settings, descriptor.id)?;
    // The tokenizer snapshot is structural; all weight files are priced from finalized tokens.
    let cold_admission =
        prepare_cached_candle_base_floor(&request.model, "ComfyUI Z-Image", settings, &spec, &[])?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", ZIMAGE_COMFYUI_DEFAULT_STEPS, 1..=50);
    let raw_settings = zimage_comfyui_raw_settings(request, steps, adapter_count);
    let conditioning = super::resolve_img2img_init_generic(request, settings, project_path)?
        .map(|(image, strength)| Conditioning::Reference {
            image,
            strength: Some(strength),
        })
        .into_iter()
        .collect::<Vec<_>>();

    // Per-image work items retain the exact resolved conditioning beside each seed/prompt.
    let work: Vec<(i64, String, Vec<Conditioning>)> = (0..request.count as usize)
        .map(|index| {
            (
                resolve_seed(request, index),
                request.prompt.clone(),
                conditioning.clone(),
            )
        })
        .collect();
    let total = work.len();
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();

    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission(
        job.id.clone(),
        descriptor.id,
        0,
        spec,
        "ComfyUI Z-Image load failed".to_owned(),
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
                move |_index, (seed, prompt, conditioning), preview, on_progress| {
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
                        conditioning,
                        preview,
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
