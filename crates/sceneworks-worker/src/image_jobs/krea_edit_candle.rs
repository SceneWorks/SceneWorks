use super::resolve_seed;
use super::{
    admit_candle_base, attach_manifest_text_encoder, attach_required_components,
    attach_selected_decoder, candle_certified_load_spec, candle_resolved_tier_key,
    consume_gen_events, drive_gen_items, fit_engine_image, load_reference_image, load_spec,
    optimized_shared_memory_context, pid_effective_dims, pid_output_tier, resolve_adapters,
    resolve_advanced_or_manifest_f32, resolve_advanced_or_manifest_u32, resolve_pid_weights,
    resolve_text_style_gain, resolve_weights_dir, safetensors_tensor_bytes_with_prefixes,
    start_cached_gen_stream, ApiClient, CandleBaseEvidence, Image, ImagePlan, ImageRequest,
    JobSnapshot, JsonObject, Path, Settings, Value, WorkerError, WorkerResult,
};
use serde_json::json;

// Candle (Windows/CUDA) Krea 2 image-edit route (epic 10871) — the Kontext-style dual-conditioned edit
// surface off-Mac through the registered `krea_2_edit` / `krea_2_turbo_edit` generators.
// The candle sibling of the macOS `krea_edit.rs` (`generate_krea_edit_stream`): an `edit_image` job on
// `krea_2_raw` carrying a source image (+ optional 2nd) is routed here rather than the plain Raw t2i path.
// The source(s) ride as in-context VAE tokens at distinct RoPE frames AND ground the Qwen3-VL vision
// tower (R2 dual conditioning); the trained `krea2_identity_edit` LoRA is what makes that conditioning
// steer the edit (R5 — the base leaves it inert).
//
// **Candle-only.** macOS keeps the MLX Krea edit path (`krea_edit.rs`, `krea_2_edit` registry generator);
// the candle Krea edit is now a registered-generator path so provider request scopes own staged
// component loading, so this whole file is gated to the Windows/CUDA
// candle build (the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs`
// module, so it shares that module's imports (ImageRequest/Settings/WorkerResult/WorkerError/
// `load_reference_image`/`fit_engine_image`/`resolve_weights_dir`/`resolve_adapters`/`resolve_seed`/
// `resolve_advanced_or_manifest_u32`/`start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/
// `gen_core`/`AdapterSpec`/… all in scope).
//
// Both Krea image variants edit: **Raw** on the full-CFG loop (epic 10871) and **Turbo** on the CFG-free
// distilled few-step loop (`krea_2_turbo_edit`, sc-11640 — the fast-path); the same
// `krea2_identity_edit` LoRA drives both (family match, no base gating). One or two references in fixed
// order — image 1 (required) + image 2 (optional), either can be a person
// ([`KREA_EDIT_CANDLE_MAX_REFERENCES`]).

/// Krea edit denoise steps default — Raw undistilled, full-CFG (mirrors `candle_gen_krea` `RAW_STEPS`).
const KREA_EDIT_CANDLE_DEFAULT_STEPS: u32 = 52;
/// Krea Turbo edit denoise steps default — the distilled CFG-free few-step student (sc-11640; mirrors
/// `candle_gen_krea` `TURBO_STEPS`).
const KREA_EDIT_CANDLE_TURBO_STEPS: u32 = 8;
/// Raw full-CFG guidance default (mirrors `candle_gen_krea` `RAW_GUIDANCE`).
const KREA_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 3.5;
/// The adapter/engine id recorded on candle Krea edit assets + telemetry (distinct from the `candle_krea`
/// txt2img lane — the edit is a separate surface with the required edit LoRA).
pub(super) const KREA_EDIT_CANDLE_ENGINE: &str = "candle_krea_edit";
/// Reference cap — the edit LoRA's fixed-order contract: image 1 (required) + image 2 (optional), either
/// can be a person. Swapping the
/// order degrades results (the LoRA authors' note); more than two is off-contract.
const KREA_EDIT_CANDLE_MAX_REFERENCES: usize = 2;

/// True when a selected LoRA declares the image-edit conditioning role (`conditioningRole: image_edit`,
/// e.g. the builtin `krea2_identity_edit`). The candle twin of the macOS
/// `krea_edit.rs::lora_declares_image_edit_role` — that file is `#[cfg(target_os = "macos")]`, so the
/// tiny check is duplicated here for the candle build. This LoRA is what makes the in-context source
/// actually steer the edit; the base weights leave it inert (R5).
fn krea_edit_candle_lora_role(lora: &Value) -> bool {
    lora.as_object()
        .and_then(|obj| obj.get("conditioningRole"))
        .and_then(Value::as_str)
        .map(|role| role.trim().to_lowercase().replace('-', "_") == "image_edit")
        .unwrap_or(false)
}

/// Whether the job carries an image-edit LoRA (any selected LoRA with `conditioningRole: image_edit`).
fn krea_edit_candle_has_lora(request: &ImageRequest) -> bool {
    request.loras.iter().any(krea_edit_candle_lora_role)
}

/// Reference asset ids for a Krea edit, in fixed order (image 1 (required) + image 2 (optional)), capped at
/// [`KREA_EDIT_CANDLE_MAX_REFERENCES`]. The multi-image picker sends the plural `referenceAssetIds`; with
/// no plural list it falls back to the single Image-Edit `sourceAssetId`. Mirrors
/// `flux2_edit_candle_reference_ids` (capped to the Krea 1..=2 contract).
fn krea_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        return request
            .reference_asset_ids
            .iter()
            .take(KREA_EDIT_CANDLE_MAX_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    Vec::new()
}

/// True when this is a Krea edit job: `krea_2_raw` / `krea_2_turbo` + `edit_image` mode + at least one
/// source reference. Both Krea image variants edit — Raw on the full-CFG loop, Turbo on the CFG-free
/// distilled few-step loop (sc-11640). Mirrors the core router's `krea_edit_candle_eligible` (gated to the
/// two models by its caller) + the macOS `krea_edit_available` (minus the weight-resolve check).
pub(super) fn krea_edit_candle_mode(request: &ImageRequest) -> bool {
    matches!(request.model.as_str(), "krea_2_raw" | "krea_2_turbo")
        && request.mode == "edit_image"
        && !krea_edit_candle_reference_ids(request).is_empty()
}

/// Whether this Krea edit runs the distilled CFG-free Turbo recipe (`krea_2_turbo` — few-step
/// `turbo_schedule`, guidance forced to 0) rather than the undistilled full-CFG Raw loop (sc-11640).
fn krea_edit_candle_distilled(request: &ImageRequest) -> bool {
    request.model == "krea_2_turbo"
}

/// True when this is a candle-eligible Krea edit job whose Raw weights resolve locally. Mirrors
/// `flux2_edit_candle_available` (Krea resolves through the shared `resolve_weights_dir` → the
/// `krea_2_raw` tier subdir).
pub(super) fn krea_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    krea_edit_candle_mode(request) && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → the per-variant default
/// (Raw 52 / Turbo 8, sc-11640).
fn krea_edit_candle_steps(request: &ImageRequest) -> u32 {
    let default = if krea_edit_candle_distilled(request) {
        KREA_EDIT_CANDLE_TURBO_STEPS
    } else {
        KREA_EDIT_CANDLE_DEFAULT_STEPS
    };
    resolve_advanced_or_manifest_u32(request, "steps", default, 1..=100)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the Raw default (3.5).
fn krea_edit_candle_guidance(request: &ImageRequest) -> f32 {
    resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        KREA_EDIT_CANDLE_DEFAULT_GUIDANCE,
        0.0..=30.0,
    )
}

/// Load the Krea edit reference set: the 1..=2 references (plural `referenceAssetIds`, else the single
/// `sourceAssetId`), each pre-fit to the render W×H (crop / pad / outpaint→pad; `stretch` keeps the legacy
/// resize). `render_edit` VAE-encodes each at the target resolution, so pre-fitting keeps an off-aspect
/// source from stretching. Errors if no source. Shares the geometry with the other edit lanes
/// ([`fit_engine_image`]).
fn load_krea_edit_candle_references(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<Image>> {
    let ids = krea_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires a source image (sourceAssetId).".to_owned(),
        ));
    }
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let fitted = if request.fit_mode == "stretch" {
            source
        } else {
            fit_engine_image(source, width, height, &request.fit_mode)?
        };
        references.push(fitted);
    }
    Ok(references)
}

fn krea_edit_candle_conditioning(references: &[Image]) -> Vec<gen_core::Conditioning> {
    if references.len() == 1 {
        vec![gen_core::Conditioning::Reference {
            image: references[0].clone(),
            strength: None,
        }]
    } else {
        vec![gen_core::Conditioning::MultiReference {
            images: references.to_vec(),
        }]
    }
}

#[allow(clippy::too_many_arguments)]
fn krea_edit_candle_generate_one(
    generator: &dyn gen_core::Generator,
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    conditioning: Vec<gen_core::Conditioning>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    preview: gen_core::PreviewSink,
    cancel: &gen_core::CancelFlag,
    on_progress: &mut dyn FnMut(gen_core::Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = gen_core::GenerationRequest {
        prompt,
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance: Some(guidance),
        conditioning,
        use_pid,
        text_style_gain,
        memory,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut request,
        memory_strategy_context,
        on_progress,
    )
    .map_err(|error| WorkerError::Engine(format!("Krea edit generation failed: {error}")))?;
    match output {
        gen_core::GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea edit generator returned non-image output".to_owned(),
        )),
    }
}

/// Flat telemetry recorded on candle Krea edit assets (parity with the macOS `krea_edit_raw_settings`).
fn krea_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(KREA_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Krea 2 edit generation: resolve the Raw weights + source references on the async side,
/// pre-fit each to the render geometry, then load the exact registered edit generator once and render
/// each image through its request-scope lifecycle — `request.count` edits of the same reference set,
/// each its own seed. Mirrors
/// [`generate_candle_flux2_edit_stream`]'s blocking-thread + streamed-events shape and reuses
/// [`consume_gen_events`]; differs in the required edit LoRA (R5) and the direct pipeline-function API
/// (`Generator::generate` with `count = 1` per streamed item).
pub(super) async fn generate_candle_krea_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    if !krea_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    // Krea "text style" tap-reweight gain (sc-12009) — self-gates on `ui.textStyleGain` (Krea only),
    // applied to the edit lane's POSITIVE grounded context by the engine (inference sc-12009).
    let text_style_gain = resolve_text_style_gain(request);
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Krea 2 Raw weights not found".to_owned()))?;

    // R5 (epic 10871): the base cannot edit without the edit LoRA — its in-context/grounded source
    // conditioning is inert without the trained weights (a VAE-only render is off-distribution, not
    // shippable). Require an `image_edit`-role LoRA (the builtin `krea2_identity_edit`). Checked before
    // loading weights so the error is fast + clear (parity with the macOS lane).
    if !krea_edit_candle_has_lora(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea 2 edit requires the Krea 2 Identity Edit LoRA (or another image-edit LoRA): without \
             it the source-image conditioning is inert. Select it in the LoRA picker."
                .to_owned(),
        ));
    }

    let steps = krea_edit_candle_steps(request);
    // Turbo edit is CFG-free — `render_edit(distilled = true)` forces guidance 0 and runs the few-step
    // `turbo_schedule`; Raw honors the resolved guidance on the full-CFG loop (sc-11640).
    let distilled = krea_edit_candle_distilled(request);
    let guidance = if distilled {
        0.0
    } else {
        krea_edit_candle_guidance(request)
    };
    let negative = request.negative_prompt.clone();
    // The selected LoRAs → adapter specs (the edit LoRA + any user LoRAs), folded into the DiT at load.
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();
    let descriptor_id = if distilled {
        "krea_2_turbo_edit"
    } else {
        "krea_2_edit"
    };
    let resolved_tier = candle_resolved_tier_key(request, &weights_dir, false);
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    let mut strategy_spec = load_spec(weights_dir.clone(), None, adapters.clone(), None)
        .with_resolved_route(request.model.clone());
    if let Some(pid) = pid_weights.as_ref() {
        strategy_spec = strategy_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
    strategy_spec = attach_required_components(
        strategy_spec,
        descriptor_id,
        &request.model_manifest_entry,
        settings,
    )?;
    strategy_spec = attach_selected_decoder(strategy_spec, descriptor_id, request, settings)?;
    strategy_spec = attach_manifest_text_encoder(strategy_spec, descriptor_id, request, settings)?;
    let selected_text_encoder_bytes = strategy_spec.text_encoder.as_ref().map_or(0, |source| {
        let path = match source {
            gen_core::WeightsSource::File(path) | gen_core::WeightsSource::Dir(path) => path,
        };
        gen_core::safetensors_path_bytes(path)
    });
    let adapter_resident_bytes = adapters
        .iter()
        .fold(0_u64, |total, adapter| {
            total.saturating_add(gen_core::safetensors_path_bytes(&adapter.path))
        })
        // Edit additionally materializes the Qwen3-VL `visual.*` tower at f32 from dense bf16
        // source tensors, so charge twice their on-disk bytes. The base catalog peak already prices
        // the language model; prefix accounting avoids charging that large subtree again.
        .saturating_add(
            safetensors_tensor_bytes_with_prefixes(&weights_dir.join("text_encoder"), &["visual."])
                .saturating_mul(2),
        )
        // The edit-only Qwen VAE encoder stays f32 like its source. The base peak already includes
        // the decoder, so count only the encoder and its input quantization convolution.
        .saturating_add(safetensors_tensor_bytes_with_prefixes(
            &weights_dir.join("vae"),
            &["encoder.", "quant_conv."],
        ))
        // A substituted decoder is additional to the catalog's default-artifact estimate. This is
        // deliberately conservative (the bundled decoder is not subtracted) and preserves the
        // estimated fallback when no route-specific evidence exists.
        .saturating_add(selected_text_encoder_bytes);
    let offload_policy = admit_candle_base(
        request,
        settings,
        &weights_dir,
        "Krea edit",
        CandleBaseEvidence::Catalog,
        adapter_resident_bytes,
        true,
    )
    .await?;
    strategy_spec = strategy_spec.with_offload_policy(offload_policy);
    // Telemetry `repo` — the manifest `repo` else the Krea 2 Raw default.
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            crate::engines::default_repo_for(&request.model).unwrap_or("SceneWorks/krea-2-raw-mlx")
        })
        .to_owned();

    let (width, height) = pid_effective_dims(
        request.width,
        request.height,
        use_pid,
        pid_output_tier(request),
    );
    let references =
        load_krea_edit_candle_references(request, project_path, settings, width, height)?;
    let raw_settings =
        krea_edit_candle_raw_settings(request, &repo, steps, guidance, references.len());

    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let reclaimable_gb = crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id);
    let budget =
        raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable_gb));
    let predicted_peak_gb = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        resolved_tier,
        adapter_resident_bytes,
    );
    let memory_evaluation = crate::candle_memory_strategy::evaluate_shared_image(
        descriptor_id,
        &request.model,
        &strategy_spec,
        candle_certified_load_spec(
            descriptor_id,
            settings,
            &strategy_spec,
            &request.model_manifest_entry,
            resolved_tier,
        ),
        &request.model_manifest_entry,
        resolved_tier,
        "edit_image",
        Some("lora"),
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: references.len() as u32,
        },
        true,
        use_pid,
        false,
        false,
        budget,
        predicted_peak_gb,
        adapter_resident_bytes,
        if reclaimable_gb > 0.0 {
            gen_core::MemoryCacheState::Warm
        } else {
            gen_core::MemoryCacheState::Cold
        },
    )?;
    let generation_memory = memory_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.memory.clone());
    let memory_strategy_context = memory_evaluation
        .and_then(|evaluation| optimized_shared_memory_context(evaluation.context));

    // Per-image work items: (seed, prompt) — `request.count` edits of the same reference set.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        descriptor_id,
        adapter_count,
        strategy_spec,
        format!("{descriptor_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(
                tx,
                work,
                move |_index, (seed, prompt), preview, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let conditioning = krea_edit_candle_conditioning(&references);
                    let image = krea_edit_candle_generate_one(
                        generator,
                        prompt,
                        (!negative.trim().is_empty()).then(|| negative.clone()),
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        conditioning,
                        use_pid,
                        text_style_gain,
                        generation_memory.clone(),
                        memory_strategy_context.as_ref(),
                        preview,
                        &cancel,
                        on_progress,
                    )?;
                    Ok(Some((seed, image.0, image.1, image.2)))
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
        KREA_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
