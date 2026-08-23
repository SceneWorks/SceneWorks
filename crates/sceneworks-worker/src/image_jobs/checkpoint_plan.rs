// Plan-driven image route (epic 20398, sc-20634): the universal-import walking skeleton.
//
// A user image model whose manifest entry carries `importPlan.checkpointId` is backed by a
// persisted `ImportPlanV1` in the app's checkpoint plan store (`<data>/checkpoints/`), compiled from
// a linked library root by `sceneworks_core::checkpoint_plan_store`. This route:
//
//   1. resolves and re-verifies the plan (record ↔ plan agreement, approved root present, every
//      layer's bytes still the bytes the plan was compiled from) — every refusal is a typed
//      `CheckpointPlanError` surfaced with its stable `[checkpoint-plan:<code>]` prefix, raised
//      BEFORE any loader is constructed;
//   2. resolves the provider by family + source shape + operation through the live provider
//      registry's imported-model authority (`imported_model_descriptor`), so an unknown family or
//      an unbound backend/operation refuses with a typed message rather than falling through;
//   3. builds an ordinary `LoadSpec` from the plan layers (primary `WeightsSource::File`, plus the
//      provider's declared required components), pins every file, and renders through the SAME
//      cached-generator seam every other lane uses. On the inference side the provider reads the
//      file through the mapped logical-weight reader and the registered codec table.
//
// Scope: `Generate` (text-to-image) for a single-file transformer plan. Edit / pose / multi-phase
// / reference shapes are refused loudly here — they keep their family lanes until sc-20644 moves
// each family's full surface onto this route. The legacy `KreaImported` lane coexists: it claims
// `paths.model` installs and explicitly declines an entry that carries `importPlan`, and this
// route never reads `paths.model`, so one entry is claimed by exactly one lane.

use sceneworks_core::checkpoint_plan_store::{
    CheckpointPlanError, CheckpointPlanStore, ResolvedCheckpointV1,
};

/// The adapter/engine id recorded on assets rendered through the plan-driven route.
#[cfg(target_os = "macos")]
const CHECKPOINT_PLAN_ENGINE: &str = "mlx_checkpoint_plan";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CHECKPOINT_PLAN_ENGINE: &str = "candle_checkpoint_plan";

/// Plan layer roles the skeleton maps to a provider source shape.
const CHECKPOINT_PLAN_TRANSFORMER_ROLE: &str = "transformer";
const CHECKPOINT_PLAN_FUSED_ROLE: &str = "checkpoint";

/// The persisted checkpoint id a manifest entry is bound to, when it is plan-backed. The same
/// reader the scheduler's imported claim uses (`jobs_store::checkpoint_plan_checkpoint_id`), so
/// admission and the worker agree on what "plan-backed" means.
pub(crate) fn checkpoint_plan_checkpoint_id(entry: &JsonObject) -> Option<&str> {
    sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
}

/// Whether the request's manifest entry is plan-backed (the single-claim discriminator the bespoke
/// imported lanes consult so they never also claim a plan-backed entry).
pub(crate) fn request_is_checkpoint_plan_backed(request: &ImageRequest) -> bool {
    checkpoint_plan_checkpoint_id(&request.model_manifest_entry).is_some()
}

fn checkpoint_plan_refusal(error: CheckpointPlanError) -> WorkerError {
    WorkerError::InvalidPayload(error.to_string())
}

/// Every plan-derived, pinned input the handler needs. Selected once in `prepare_image_route` and
/// carried through dispatch so the async preamble can never retarget a source.
struct PreparedCheckpointPlanSources {
    checkpoint_id: String,
    resolved: ResolvedCheckpointV1,
    /// The registered provider that loads this family/source/operation.
    descriptor: gen_core::ModelDescriptor,
    source: gen_core::ImportedModelSource,
    /// The plan's primary weight file, pinned.
    primary: gen_core::PinnedWeightsFile,
    /// Components the provider declares as required, resolved to local sources.
    components: Vec<(&'static str, WeightsSource)>,
}

/// The plan layer that supplies the provider's primary weights, and the registry source shape it
/// implies. A plan without one refuses as a missing component.
fn checkpoint_plan_primary_layer(
    resolved: &ResolvedCheckpointV1,
) -> WorkerResult<(gen_core::ImportedModelSource, &sceneworks_core::checkpoint_plan_store::ResolvedLayerV1)>
{
    let transformers: Vec<_> = resolved
        .layers_with_role(CHECKPOINT_PLAN_TRANSFORMER_ROLE)
        .collect();
    let fused: Vec<_> = resolved
        .layers_with_role(CHECKPOINT_PLAN_FUSED_ROLE)
        .collect();
    match (transformers.as_slice(), fused.as_slice()) {
        ([transformer], []) => Ok((
            gen_core::ImportedModelSource::TransformerFile,
            transformer,
        )),
        ([], [checkpoint]) => Ok((gen_core::ImportedModelSource::FusedCheckpoint, checkpoint)),
        ([], []) => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {:?} ({} family) has no '{}' or '{}' \
             layer; its plan carries roles [{}]",
            resolved.checkpoint_id,
            resolved.family(),
            CHECKPOINT_PLAN_TRANSFORMER_ROLE,
            CHECKPOINT_PLAN_FUSED_ROLE,
            resolved
                .plan
                .layers
                .iter()
                .map(|layer| layer.role.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        _ => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:ambiguous-component] checkpoint {:?} carries {} transformer and {} \
             fused-checkpoint layer(s); the plan-driven route serves exactly one primary layer",
            resolved.checkpoint_id,
            transformers.len(),
            fused.len()
        ))),
    }
}

/// Resolve one provider-declared required component to a local source. `base_snapshot` is the
/// resident family base tier that supplies the shared encoder / VAE / tokenizer / architecture
/// config a bare transformer file omits. Family base resolution stays with each family's resident
/// tier helper until the registry's `base_compatibility` metadata reaches the pinned runtime.
fn resolve_checkpoint_plan_component(
    component: &'static str,
    family: &str,
    checkpoint_id: &str,
    settings: &Settings,
) -> WorkerResult<WeightsSource> {
    match (component, family) {
        (gen_core::BASE_SNAPSHOT_COMPONENT, "krea_2") => {
            resolve_krea_imported_base_tier(settings).map(WeightsSource::Dir)
        }
        _ => Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:missing-component] checkpoint {checkpoint_id:?} ({family} family) \
             requires component '{component}', which this runtime cannot supply for that family"
        ))),
    }
}

/// Select the plan-driven route for a plan-backed manifest entry. `Ok(None)` only when the entry
/// is not plan-backed; a plan-backed entry that cannot be served is a refusal, never a fall-through
/// to a bespoke lane or the stub.
fn prepare_checkpoint_plan_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedCheckpointPlanSources>> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(&request.model_manifest_entry) else {
        return Ok(None);
    };
    if imported_generate_request_has_unsupported_shape(request)
        || !request.loras.is_empty()
        || request.reference_asset_id.is_some()
        || request.hires_fix.enabled
    {
        return Err(WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?} is served through \
             the plan-driven route for text-to-image generation only; edit, reference, pose, \
             multi-phase, LoRA, and Hires.fix requests are not on this route yet"
        )));
    }
    let store = CheckpointPlanStore::open(&settings.data_dir);
    let resolved = store.resolve(checkpoint_id).map_err(checkpoint_plan_refusal)?;
    let family = resolved.family().to_owned();
    let (source, primary_layer) = checkpoint_plan_primary_layer(&resolved)?;
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        &family,
        source,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: this runtime's \
             provider registry has no {family:?} adapter bound for {source:?} Generate on this \
             backend"
        ))
    })?;
    let primary = gen_core::PinnedWeightsFile::pin(&primary_layer.path).map_err(|error| {
        crate::classify_engine_error("Checkpoint plan source preparation failed", error)
    })?;
    let mut components = Vec::with_capacity(descriptor.required_components.len());
    for component in descriptor.required_components {
        components.push((
            *component,
            resolve_checkpoint_plan_component(component, &family, checkpoint_id, settings)?,
        ));
    }
    Ok(Some(PreparedCheckpointPlanSources {
        checkpoint_id: checkpoint_id.to_owned(),
        resolved,
        descriptor,
        source,
        primary,
        components,
    }))
}

/// Optional per-request override of a u32 knob: `advanced[key]`, else the manifest entry's
/// `[key]`. `None` means "the provider's own default", which the registry descriptor owns.
fn checkpoint_plan_u32_override(request: &ImageRequest, key: &str) -> Option<u32> {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .and_then(|value| u32::try_from(value).ok())
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
}

fn checkpoint_plan_f32_override(request: &ImageRequest, key: &str) -> Option<f32> {
    let parse = |value: &Value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
}

/// The `LoadSpec` the plan implies: the primary file plus every declared component, with the
/// request's quant and every file pin finalized. Pure given the prepared sources, so the parity
/// test can compare it against the legacy lane's spec.
fn checkpoint_plan_load_spec(
    sources: &PreparedCheckpointPlanSources,
    quant: Option<Quant>,
) -> WorkerResult<LoadSpec> {
    let mut spec = sources.components.iter().cloned().fold(
        LoadSpec::new(WeightsSource::File(
            sources.primary.loader_path().to_path_buf(),
        )),
        |spec, (id, source)| spec.with_component(id, source),
    );
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        &mut spec,
        [sources.primary.clone()],
        "Checkpoint plan source preparation failed",
    )?;
    Ok(spec)
}

fn checkpoint_plan_raw_settings(
    request: &ImageRequest,
    sources: &PreparedCheckpointPlanSources,
    steps: Option<u32>,
    guidance: Option<f32>,
    quant_bits: Option<i64>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert(
        "mode".to_owned(),
        Value::String("text_to_image".to_owned()),
    );
    raw.insert(
        "engine".to_owned(),
        Value::String(CHECKPOINT_PLAN_ENGINE.to_owned()),
    );
    raw.insert(
        "checkpointId".to_owned(),
        Value::String(sources.checkpoint_id.clone()),
    );
    raw.insert(
        "importPlanId".to_owned(),
        Value::String(sources.resolved.plan.plan_id.clone()),
    );
    raw.insert(
        "importPlanSemanticDigest".to_owned(),
        Value::String(sources.resolved.record.plan.semantic_digest.clone()),
    );
    raw.insert(
        "importPlanFamily".to_owned(),
        Value::String(sources.resolved.plan.family.clone()),
    );
    raw.insert(
        "importPlanProvider".to_owned(),
        Value::String(sources.descriptor.id.to_owned()),
    );
    raw.insert(
        "importPlanSource".to_owned(),
        Value::String(format!("{:?}", sources.source)),
    );
    if let Some(steps) = steps {
        raw.insert("numInferenceSteps".to_owned(), json!(steps));
    }
    if let Some(guidance) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(guidance));
    }
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map_or(Value::Null, Value::from),
    );
    raw
}

/// Plan-driven text-to-image: `count` renders, each its own seed, through the provider the
/// registry bound for the plan's family and source shape.
#[allow(clippy::too_many_arguments)]
async fn generate_checkpoint_plan_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    sources: PreparedCheckpointPlanSources,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let (quant, quant_bits) =
        imported_model_quant(request, &sources.descriptor, "Checkpoint plan")?;
    let (width, height) = (request.width, request.height);
    let steps = checkpoint_plan_u32_override(request, "steps").map(|steps| steps.clamp(1, 100));
    let guidance = checkpoint_plan_f32_override(request, "guidanceScale");
    let raw_settings = checkpoint_plan_raw_settings(request, &sources, steps, guidance, quant_bits);
    let negative_prompt = (!request.negative_prompt.trim().is_empty())
        .then(|| request.negative_prompt.clone());
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let spec = checkpoint_plan_load_spec(&sources, quant)?;
    #[cfg(target_os = "macos")]
    let spec = crate::mlx_fit_gate::apply_residency_policy(spec, sources.descriptor.id)?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    admit_candle_load_spec_floor(&request.model, "Checkpoint plan", settings, &spec).await?;

    let engine_id = sources.descriptor.id;
    let checkpoint_id = sources.checkpoint_id.clone();
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("Checkpoint plan {checkpoint_id:?} load failed"),
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let generation = GenerationRequest {
                    prompt,
                    negative_prompt: negative_prompt.clone(),
                    width,
                    height,
                    count: 1,
                    seed: Some(seed as u64),
                    steps,
                    guidance,
                    preview,
                    cancel: cancel.clone(),
                    ..Default::default()
                };
                let output = match model.generate(&generation, &mut *on_progress) {
                    Ok(output) => output,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Checkpoint plan {checkpoint_id:?} generation failed: {error}"
                        )));
                    }
                };
                match output {
                    GenerationOutput::Images(mut images) => {
                        let image = images.pop().ok_or_else(|| {
                            WorkerError::Engine(format!(
                                "Checkpoint plan {checkpoint_id:?} produced no image"
                            ))
                        })?;
                        Ok(Some((seed, image.width, image.height, image.pixels)))
                    }
                    _ => Err(WorkerError::Engine(format!(
                        "Checkpoint plan {checkpoint_id:?} returned non-image output"
                    ))),
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
        CHECKPOINT_PLAN_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
