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
// macOS/MLX only, deliberately: the Mage generator descriptor is `mac_only`, there is no candle
// Mage engine to route a fine-tune to, and `mage-flow` is correspondingly absent from
// `CANDLE_ROUTED_FAMILIES`. This file is `include!`d into the `image_jobs` module, sharing its
// imports.
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

/// The builtin catalog id whose installed snapshot supplies the shared text encoder + VAE a
/// fine-tuned transformer is paired with.
///
/// `mage_flow_base` is not a preference — it is the ONLY Mage generation target the training
/// registry offers (`training_targets`), so a fine-tune can only have started from it, and its
/// `coRequisite` rows are what name the shared-components mirror. The components are BIT-IDENTICAL
/// across all six Mage variants (sc-14979), so this resolves the same bytes whichever variant a
/// user happens to have installed.
#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
const MAGE_FINETUNED_COMPONENT_TIER: &str = "bf16";

/// Denoise-steps / guidance fallbacks — the undistilled `mage_flow_base` regime a fine-tune
/// inherits. The Studio normally supplies both from the catalog entry's `defaults`
/// (`apply_family_studio_surface_defaults`); these apply only when it does not.
#[cfg(target_os = "macos")]
const MAGE_FINETUNED_DEFAULT_STEPS: u32 = 30;
#[cfg(target_os = "macos")]
const MAGE_FINETUNED_DEFAULT_GUIDANCE: f32 = 5.0;

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
#[cfg(target_os = "macos")]
fn resolve_mage_finetuned_transformer(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
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
    Ok(sceneworks_core::base_weights::is_mage_flow_transformer_dir(&path).then_some(path))
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
#[cfg(target_os = "macos")]
fn resolve_mage_finetuned_components(
    settings: &Settings,
) -> WorkerResult<std::collections::BTreeMap<String, WeightsSource>> {
    let descriptor = crate::inference_runtime::media_descriptor(MAGE_FINETUNED_BASE_MODEL)
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
#[cfg(target_os = "macos")]
fn mage_finetuned_available(request: &ImageRequest, settings: &Settings) -> bool {
    if request.mode == "edit_image"
        || !request.loras.is_empty()
        || !pose_entries(request).is_empty()
        || !request.reference_asset_ids.is_empty()
        || non_empty(&request.reference_asset_id)
        || non_empty(&request.source_asset_id)
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || request
            .advanced
            .get("phases")
            .and_then(Value::as_array)
            .is_some_and(|phases| !phases.is_empty())
    {
        return false;
    }
    matches!(
        resolve_mage_finetuned_transformer(request, settings),
        Ok(Some(_))
    )
}

/// Flat telemetry recorded on assets rendered from a fine-tuned base.
#[cfg(target_os = "macos")]
fn mage_finetuned_raw_settings(request: &ImageRequest, steps: u32, guidance: f32) -> JsonObject {
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
    raw
}

/// Real fine-tuned Mage-Flow base generation (sc-15036): resolve the trained transformer dir and
/// the installed base's shared components, load once through `load_finetuned`, and render each
/// image on the blocking thread. The `Box<dyn Generator>` is bespoke (not registry-cached) — the
/// checkpoint lives at a user path under a novel id, so there is nothing to key a registry cache
/// on, exactly like the Krea imported lane.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_mage_finetuned_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let transformer = resolve_mage_finetuned_transformer(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "The fine-tuned Mage-Flow checkpoint could not be resolved — its recorded path is not \
             a complete transformer directory (config.json + diffusion_pytorch_model.safetensors)."
                .to_owned(),
        )
    })?;
    // Require the base's shared components before any compute — a clear "install the Mage-Flow
    // Base model first" error rather than a deep load failure.
    let components = resolve_mage_finetuned_components(settings)?;

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
    let raw_settings = mage_finetuned_raw_settings(request, steps, guidance);

    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        MAGE_FINETUNED_ENGINE,
        0,
        move || {
            // `spec.weights` is the fine-tuned TRANSFORMER dir itself (not a snapshot root): a
            // training run emits the DiT alone, so both shared components MUST be staged and there
            // is deliberately no flat-layout fallback engine-side.
            let spec = components.into_iter().fold(
                LoadSpec::new(WeightsSource::Dir(transformer)),
                |spec, (id, source)| spec.with_component(id, source),
            );
            runtime_macos::providers::mage::load_finetuned(
                // The published checkpoint the fine-tune started from — the ONLY Mage generation
                // training target — which is what fixes the architecture and the undistilled
                // sampling regime the trained weights inherit.
                runtime_macos::providers::mage::MageVariant::Base,
                &spec,
            )
            .map_err(|error| {
                WorkerError::Engine(format!(
                    "Fine-tuned Mage-Flow checkpoint load failed: {error}"
                ))
            })
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
                    guidance: Some(guidance),
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&request, &mut *on_progress) {
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
