// Fine-tuned Mage-Flow base checkpoint txt2img route (sc-15036, epic 14034 F6).
//
// Renders the artifact a FULL base fine-tune (sc-14056) produces: a diffusers **transformer
// component directory** (`config.json` + `diffusion_pytorch_model.safetensors`) holding every
// retrained DiT weight, written into `<data>/models/finetunes/<id>` and registered into the user
// model catalog by `rust-api::register_trained_base_checkpoint`. A training run emits the DiT
// alone — it never re-emits the 8.9 GB text encoder or the VAE — so the checkpoint is paired at
// load with the INSTALLED Mage-Flow base's shared components and rendered through
// `mlx_gen_mage::load_finetuned`.
//
// Two things make this its own lane rather than a case of the existing ones:
//
//  * The generic snapshot-dir path cannot serve it. `mlx_model(&request.model)` is `None` for a
//    novel catalog id, so a fine-tune would stub; and even with an id, the registry `load` path
//    enforces the pinned-checkpoint identity fingerprint over
//    `transformer_blocks.0.attn.add_k_proj.bias` — a weight a full fine-tune retrains, so it would
//    reject the user's own checkpoint by construction. `load_finetuned` is the entrypoint that
//    skips exactly that guard and nothing else.
//  * The imported-checkpoint lanes cannot serve it either: `krea_imported`'s
//    `is_diffusers_snapshot_dir` exists specifically to REJECT a dir carrying `config.json`, which
//    is precisely this shape.
//
// Both native backends serve this lane through their runtime bundle's Mage full-fine-tune loader.
// This file is `include!`d into the `image_jobs` module, sharing its imports.
//
// Scope: plain **txt2img**. The non-edit Mage variants advertise no conditioning at all, and
// `load_finetuned` refuses adapters outright, so the scheduler's `mage-flow` arm in
// `imported_image_request_family_eligible` admits only that shape and this lane mirrors it. Every
// other shape (edit, reference, pose, LoRA, mask, character) stays with its established route
// rather than being silently flattened to t2i here.

/// The adapter/engine id recorded on assets rendered from a fine-tuned base (distinct from the
/// builtin `mage_flow*` registry ids and their `mlx_mage` label).
#[cfg(target_os = "macos")]
const MAGE_FINETUNED_ENGINE: &str = "mlx_mage_finetuned";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const MAGE_FINETUNED_ENGINE: &str = "candle_mage_finetuned";

/// The builtin catalog id whose installed snapshot supplies the shared text encoder + VAE a
/// fine-tuned transformer is paired with.
///
/// `mage_flow_base` is not a preference — it is the ONLY Mage generation target the training
/// registry offers (`training_targets`), so a fine-tune can only have started from it, and its
/// `coRequisite` rows are what name the shared-components mirror. The components are BIT-IDENTICAL
/// across all six Mage variants (sc-14979), so this resolves the same bytes whichever variant a
/// user happens to have installed.
const MAGE_FINETUNED_BASE_MODEL: &str = "mage_flow_base";

/// The tier the shared components are staged from: **dense bf16**.
///
/// Not a preference either. A fine-tune checkpoint is dense bf16 (the trainer's
/// `save_full_checkpoint` writes bf16), and the pre-quantized q4/q8 component artifacts carry an
/// 8-bit floor on the text encoder's decoder layers that a load-time quantize does not reproduce
/// (sc-15071 — packed uniformly at 4 bits the tier rendered a repeating tiled texture, not the
/// prompt). Pairing dense with dense is the coherent load, and it is the same `bf16` tier the
/// training run itself resolved (`training_jobs::TRAINING_COMPONENT_TIER`), so a user who could
/// train has the components installed by construction.
const MAGE_FINETUNED_COMPONENT_TIER: &str = "bf16";

/// Stable request/evidence identity for the bespoke imported-transformer lane. The user model id
/// remains asset provenance; it must not masquerade as one of the six pinned builtin providers.
#[cfg(target_os = "macos")]
const MAGE_FINETUNED_MEMORY_ROUTE: &str = "mage_finetuned";

/// Denoise-steps / guidance fallbacks — the undistilled `mage_flow_base` regime a fine-tune
/// inherits. The Studio normally supplies both from the catalog entry's `defaults`
/// (`apply_family_studio_surface_defaults`); these apply only when it does not.
const MAGE_FINETUNED_DEFAULT_STEPS: u32 = 30;
const MAGE_FINETUNED_DEFAULT_GUIDANCE: f32 = 5.0;

#[derive(Debug, PartialEq, Eq)]
struct PreparedMageFinetunedTransformer {
    directory: PathBuf,
    config: gen_core::PinnedWeightsFile,
    weights: gen_core::PinnedWeightsFile,
}

/// Resolve the fine-tuned Mage-Flow transformer directory for `request`, or `None` when this is not
/// a fine-tuned-Mage job. `Some(dir)` only when ALL hold:
///   - the model's declared `family` is `mage-flow` (the route-by-family token),
///   - the id is NOT a builtin engine model (`mlx_model` is `None`) — a builtin Mage variant loads
///     from its own tiered snapshot through the generic lane, which this leaves untouched,
///   - the recorded weights location — an explicit `modelPath` wins, else the catalog entry's
///     `paths.model` — resolves, confined to an app-managed root, to a COMPLETE transformer
///     component dir (`sceneworks_core::base_weights::is_mage_flow_transformer_dir`: both the
///     config and the weight file present, so a torn artifact is caught here rather than deep in
///     the load).
///
/// The path is confined by `normalize_app_managed_model_path` (a payload can never point the
/// checkpoint outside a declared root; LAN jobs API, epic 4484) — the same confinement
/// `resolve_weights_dir` uses.
fn resolve_mage_finetuned_transformer(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedMageFinetunedTransformer>> {
    if request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        != Some("mage-flow")
    {
        return Ok(None);
    }
    if mlx_model(&request.model).is_some() {
        return Ok(None);
    }
    let Some(raw_path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .or_else(|| {
            request
                .model_manifest_entry
                .get("paths")
                .and_then(|paths| paths.get("model"))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let path = crate::paths::normalize_app_managed_model_path(
        settings,
        raw_path,
        "Fine-tuned Mage-Flow checkpoint",
    )?;
    if !sceneworks_core::base_weights::is_mage_flow_transformer_dir(&path) {
        return Ok(None);
    }
    let config = crate::paths::pin_app_managed_model_file(
        settings,
        &path.join(sceneworks_core::base_weights::MAGE_FLOW_TRANSFORMER_CONFIG_FILE),
        "Fine-tuned Mage-Flow config",
    )?;
    let weights = crate::paths::pin_app_managed_model_file(
        settings,
        &path.join(sceneworks_core::base_weights::MAGE_FLOW_TRANSFORMER_WEIGHTS_FILE),
        "Fine-tuned Mage-Flow weights",
    )?;
    Ok(Some(PreparedMageFinetunedTransformer {
        directory: path,
        config,
        weights,
    }))
}

/// Resolve the installed base's shared `text_encoder` + `vae` component dirs, staged onto the
/// fine-tune's `LoadSpec`.
///
/// Rides the SAME `resolve_co_requisites_for_tier` seam the builtin Mage render lane and the Mage
/// TRAINER use, driven by the registered generator descriptor's `required_components` — never a
/// hardcoded component list. That matters concretely: the shared mirror declares one `coRequisite`
/// row per tier per component with a GLOB `files` entry and a `subdir` (`bf16/text_encoder/*` +
/// `subdir: "bf16/text_encoder"`), and only the tier-aware resolver carries both the glob→directory
/// arm and the subdir narrowing. A tier-agnostic lookup has three candidates per id and refuses.
///
/// All-or-nothing, before any compute: a missing component fails the job with the seam's actionable
/// error naming the component id + repo, rather than a mid-load "No such file or directory".
fn resolve_mage_finetuned_components(
    settings: &Settings,
) -> WorkerResult<std::collections::BTreeMap<String, WeightsSource>> {
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "mage-flow",
        gen_core::ImportedModelSource::TransformerDirectory,
        gen_core::ImportedModelOperation::Generate,
    )
        .ok_or_else(|| {
            WorkerError::Engine(
                "the Mage-Flow generator is not registered in this runtime build — cannot resolve \
                 the shared components a fine-tuned checkpoint pairs with"
                    .to_owned(),
            )
        })?;
    let manifest_entry = crate::training_jobs::builtin_model_manifest_entry(MAGE_FINETUNED_BASE_MODEL)
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "'{MAGE_FINETUNED_BASE_MODEL}' has no builtin catalog entry to resolve the shared \
             text encoder and VAE from"
        ))
    })?;
    crate::model_jobs::resolve_co_requisites_for_tier(
        &descriptor,
        &manifest_entry,
        settings,
        Some(MAGE_FINETUNED_COMPONENT_TIER),
    )
    .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "The Mage-Flow base model's shared components are not installed, so a fine-tuned \
             checkpoint cannot be paired with them — install the Mage-Flow Base model (bf16) \
             first. ({error})"
        ))
    })
}

/// True when this is a fine-tuned Mage-Flow job this backend can serve.
///
/// txt2img only, mirroring the scheduler's `mage-flow` arm in
/// `imported_image_request_family_eligible` so the claim gate and the router agree: no edit mode,
/// no reference/source, no pose set, no multi-phase, no mask, no character/look, and no LoRAs
/// (`load_finetuned` refuses adapters). Deliberately does NOT gate on the base components being
/// installed — a missing one surfaces as the loud
/// [`resolve_mage_finetuned_components`] error in the handler rather than a silent fall-through to
/// the stub. Mirrors the shape of the other `…_available` predicates.
/// Test-facing claim probe. Production (both backends) routes through
/// [`prepare_mage_finetuned_transformer`] so the payload-selected transformer dir stays pinned from
/// route selection through provider load — the same split `sdxl_imported_available` uses.
#[cfg(test)]
fn mage_finetuned_available(request: &ImageRequest, settings: &Settings) -> bool {
    matches!(
        prepare_mage_finetuned_transformer(request, settings),
        Ok(Some(_))
    )
}

/// Validate the exact registered Mage generate shape and retain its pinned transformer files for
/// dispatch. Production moves this value through [`PreparedImageRoute`] so async admission work
/// cannot retarget the payload-selected directory between route selection and provider load.
fn prepare_mage_finetuned_transformer(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedMageFinetunedTransformer>> {
    let Some(descriptor) = imported_generate_request_supported(
        request,
        "mage-flow",
        gen_core::ImportedModelSource::TransformerDirectory,
    ) else {
        return Ok(None);
    };
    if imported_model_quant(request, &descriptor, "Fine-tuned Mage-Flow").is_err() {
        return Ok(None);
    }
    resolve_mage_finetuned_transformer(request, settings)
}

/// Flat telemetry recorded on assets rendered from a fine-tuned base.
fn mage_finetuned_raw_settings(
    request: &ImageRequest,
    steps: u32,
    guidance: f32,
    quant_bits: Option<i64>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mode".to_owned(),
        Value::String("text_to_image".to_owned()),
    );
    // The provenance that makes an asset traceable back to the run that produced its base.
    raw.insert("baseCheckpoint".to_owned(), Value::String(request.model.clone()));
    raw.insert(
        "engine".to_owned(),
        Value::String(MAGE_FINETUNED_ENGINE.to_owned()),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map_or(Value::Null, Value::from),
    );
    raw
}

/// Opt the imported transformer into deferred materialization only when the authoritative loaded
/// provider proves the exact prepared source is reopenable. Dense bf16 fine-tunes can satisfy this;
/// q4/q8 fine-tunes are load-time quantized and therefore retain Eager with Transformer Missing.
/// Lower rungs remain provider-owned in both cases.
#[cfg(target_os = "macos")]
fn mage_finetuned_memory_load_shape(
    provider: &str,
    spec: LoadSpec,
) -> LoadSpec {
    mage_finetuned_memory_load_shape_with(spec, |candidate| {
        crate::inference_runtime::media()
            .memory_strategy_contract(provider, candidate)
            .ok()
            .flatten()
            .is_some_and(|contract| {
                contract
                    .capability(gen_core::MemoryStrategy::BoundedTransformerResidency)
                    .is_some_and(|capability| {
                        capability.support == gen_core::MemoryStrategySupport::Implemented
                    })
            })
    })
}

#[cfg(target_os = "macos")]
fn mage_finetuned_memory_load_shape_with(
    spec: LoadSpec,
    source_proves_transformer: impl FnOnce(&LoadSpec) -> bool,
) -> LoadSpec {
    let candidate = spec
        .clone()
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    if source_proves_transformer(&candidate) {
        candidate.with_applied_load_shape_declaration()
    } else {
        spec
    }
}

/// Build the per-image request for a generated Mage full fine-tune. The checkpoint inherits the
/// undistilled Base descriptor, including true-CFG negative prompts; keeping this pure makes the
/// accepted request surface independently testable without loading model weights.
#[allow(clippy::too_many_arguments)]
fn mage_finetuned_generation_request(
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
) -> GenerationRequest {
    GenerationRequest {
        prompt,
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance: Some(guidance),
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    }
}

/// Real fine-tuned Mage-Flow base generation (sc-15036): resolve the trained transformer dir and
/// the installed base's shared components, load once through `load_finetuned`, and render each
/// image on the blocking thread. The `Box<dyn Generator>` is bespoke (not registry-cached) — the
/// checkpoint lives at a user path under a novel id, so there is nothing to key a registry cache
/// on, exactly like the Krea imported lane.
#[allow(clippy::too_many_arguments)]
async fn generate_mage_finetuned_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    transformer: PreparedMageFinetunedTransformer,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Require the base's shared components before any compute — a clear "install the Mage-Flow
    // Base model first" error rather than a deep load failure.
    let components = resolve_mage_finetuned_components(settings)?;
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "mage-flow",
        gen_core::ImportedModelSource::TransformerDirectory,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::Engine(
            "this runtime has no registered fine-tuned Mage-Flow generate route".to_owned(),
        )
    })?;
    let (quant, quant_bits) =
        imported_model_quant(request, &descriptor, "Fine-tuned Mage-Flow")?;

    let (width, height) = (request.width, request.height);
    let steps =
        resolve_advanced_or_manifest_u32(request, "steps", MAGE_FINETUNED_DEFAULT_STEPS, 1..=100);
    // Undistilled Base runs TRUE CFG, so guidance is a live knob: `advanced.guidanceScale`, else
    // the catalog entry's `defaults.guidanceScale`, else the published Base default.
    let guidance = resolve_advanced_or_manifest_f32(
        request,
        "guidanceScale",
        MAGE_FINETUNED_DEFAULT_GUIDANCE,
        1.0..=20.0,
    );
    let raw_settings = mage_finetuned_raw_settings(request, steps, guidance, quant_bits);
    let negative_prompt = (!request.negative_prompt.trim().is_empty())
        .then(|| request.negative_prompt.clone());

    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let mut spec = components.into_iter().fold(
        LoadSpec::new(WeightsSource::Dir(transformer.directory)),
        |spec, (id, source)| spec.with_component(id, source),
    );
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        [transformer.config, transformer.weights],
        "Fine-tuned Mage-Flow source preparation failed",
    )?;
    #[cfg(target_os = "macos")]
    let spec = {
        let spec = spec.with_resolved_route(MAGE_FINETUNED_MEMORY_ROUTE);
        let spec = mage_finetuned_memory_load_shape(descriptor.id, spec);
        crate::mlx_fit_gate::apply_residency_policy(spec, descriptor.id)?
    };
    // Candle's pre-load floor is the LoadSpec's own bytes (the trained DiT plus both staged shared
    // components) — the same seam the imported-SDXL lane admits on.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    admit_candle_load_spec_floor(&request.model, "Mage-Flow fine-tuned base", settings, &spec)
        .await?;

    // The load itself runs through the registered imported-source descriptor (`descriptor.id`), so
    // each backend's registry entry supplies its own `load_finetuned` — the entrypoint that skips
    // the pinned-checkpoint identity guard a full fine-tune necessarily fails.
    #[cfg(target_os = "macos")]
    let mlx_request_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        descriptor.id,
        MAGE_FINETUNED_MEMORY_ROUTE,
        &spec,
        None,
        None,
    )?;
    #[cfg(target_os = "macos")]
    let mlx_request_inputs = crate::mlx_fit_gate::MlxRequestInputs {
        width,
        height,
        count: request.count,
        mode: "text_to_image".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    };

    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        descriptor.id,
        0,
        spec,
        "Fine-tuned Mage-Flow checkpoint load failed".to_owned(),
        move |model,
              cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            #[cfg(target_os = "macos")]
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (cache_state, loaded_policy, external_committed_bytes);
                warm_policy.decline(
                    crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
                );
            }
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                #[cfg(target_os = "macos")]
                let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                    model,
                    &mlx_request_plan,
                    &mlx_request_inputs,
                    cache_state,
                    loaded_policy.offload_policy,
                    warm_policy.take(),
                    external_committed_bytes,
                )?;
                #[cfg(target_os = "macos")]
                let _request_memory_limit = memory_evaluation
                    .process_limit_bytes
                    .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                let mut request = mage_finetuned_generation_request(
                    prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    preview,
                    &cancel,
                );
                #[cfg(target_os = "macos")]
                {
                    request.memory = Some(memory_evaluation.memory);
                }
                let output = match crate::memory_strategy::generate_with_scope(
                    model,
                    &mut request,
                    {
                        #[cfg(target_os = "macos")]
                        { Some(&memory_evaluation.context) }
                        #[cfg(not(target_os = "macos"))]
                        { None }
                    },
                    &mut *on_progress,
                ) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Fine-tuned Mage-Flow generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine(
                                "Fine-tuned Mage-Flow checkpoint produced no image".to_owned(),
                            )
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(
                        "Fine-tuned Mage-Flow checkpoint returned non-image output".to_owned(),
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
        MAGE_FINETUNED_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
