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
// / reference shapes are not served here — they keep their family lanes until sc-20644 moves each
// family's full surface onto this route. The claim is therefore PER REQUEST, not per entry: this
// route takes the shapes it serves, and a managed entry's bespoke family lane keeps the rest, so
// importing a model under managed ownership never REMOVES a capability it had. Exactly one lane
// owns each REQUEST. A plan-backed request no lane claims at all is a typed refusal, never the
// procedural stub (see `CheckpointPlanSelection::into_unclaimed_refusal`).

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

/// The request shapes this skeleton serves: plain text-to-image, no adapters, no reference, no
/// Hires.fix. sc-20644 moves each family's remaining surface onto the route.
fn checkpoint_plan_serves_request_shape(request: &ImageRequest) -> bool {
    !imported_generate_request_has_unsupported_shape(request)
        && request.loras.is_empty()
        && request.reference_asset_id.is_none()
        && !request.hires_fix.enabled
}

/// A SceneWorks-owned install path on the entry, if it has one. A MANAGED install (sc-20636) keeps
/// `paths.model` alongside `importPlan.checkpointId`, so its family's bespoke lane is still there
/// for the shapes this route does not serve. A LINKED checkpoint has no installed path — the plan
/// is its only route.
///
/// LOADABLE paths only (`modelPath` / `paths.model` / `installedPath`), never
/// `imported_entry_installed_path`'s provenance-only `source.path` fallback: that one is a
/// data-dir-relative breadcrumb an import writes, and a linked entry carrying one would claim a
/// bespoke lane it does not have and hand that lane a path it must not read (sc-20636 review). The
/// admission side keeps the wider reading — "does this entry describe bytes anywhere" is a
/// different question from "can a lane load it".
fn checkpoint_plan_entry_has_bespoke_lane(request: &ImageRequest) -> bool {
    sceneworks_core::jobs_store::imported_entry_loadable_path(&request.model_manifest_entry)
        .is_some()
}

/// Whether the plan-driven route claims this request (the single-claim discriminator the bespoke
/// imported lanes consult so exactly one lane owns each request).
///
/// Plan-backed alone is not the discriminator: this skeleton serves plain text-to-image only, and a
/// managed install reached that state by being imported — it had a full family lane a moment
/// earlier. Handing the route every request and refusing most of them would mean importing a model
/// through the managed path REMOVED its edit, pose, reference, LoRA, and Hires.fix support until
/// sc-20644. So the claim is per-request: this route takes the shapes it serves, and the family's
/// bespoke lane keeps the rest.
pub(crate) fn request_is_checkpoint_plan_backed(request: &ImageRequest) -> bool {
    checkpoint_plan_checkpoint_id(&request.model_manifest_entry).is_some()
        && checkpoint_plan_serves_request_shape(request)
}

/// How this route answers "I cannot serve this request": decline so the entry's own bespoke family
/// lane claims it, or — when there is no such lane — refuse.
///
/// The distinction is exactly whether the entry has SceneWorks-owned bytes another lane can load.
/// It applies ONLY to route-capability refusals (an unsupported shape, a layer this skeleton cannot
/// source, no adapter bound). Integrity refusals — drift, a missing source, a tampered plan — stay
/// fatal for every entry: falling back to a bespoke lane that would load the very bytes the plan
/// just rejected is the silent substitution E7/E8 forbid.
/// The outcome of offering a request to the plan-driven route.
///
/// A decline is not the same as "no opinion": the route declined because the entry LOOKED like it
/// had a bespoke family lane, and that lane may not exist for this shape. So the refusal it would
/// otherwise have raised is retained here and re-raised by the router if every lane declines —
/// without that, a plan-backed entry with no claiming lane reaches `generate_stub_stream` and the
/// job COMPLETES with procedural stub output instead of the typed refusal (sc-20636 review).
#[derive(Default)]
pub(crate) struct CheckpointPlanSelection {
    prepared: Option<PreparedCheckpointPlanSources>,
    declined: Option<String>,
}

impl CheckpointPlanSelection {
    fn served(sources: PreparedCheckpointPlanSources) -> Self {
        Self {
            prepared: Some(sources),
            declined: None,
        }
    }

    fn is_available(&self) -> bool {
        self.prepared.is_some()
    }

    fn into_sources(self) -> Option<PreparedCheckpointPlanSources> {
        self.prepared
    }

    /// The refusal to raise when NO lane claimed the request. `Ok(())` only for an entry that is
    /// not plan-backed at all; everything else is a typed refusal, never a stub render.
    fn into_unclaimed_refusal(self, request: &ImageRequest) -> WorkerResult<()> {
        if let Some(message) = self.declined {
            return Err(WorkerError::InvalidPayload(message));
        }
        match checkpoint_plan_checkpoint_id(&request.model_manifest_entry) {
            Some(checkpoint_id) => Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: no image lane \
                 on this backend serves this request, and a plan-backed entry never renders \
                 procedural stub output"
            ))),
            None => Ok(()),
        }
    }
}

fn checkpoint_plan_unservable(
    request: &ImageRequest,
    message: String,
) -> WorkerResult<CheckpointPlanSelection> {
    if checkpoint_plan_entry_has_bespoke_lane(request) {
        return Ok(CheckpointPlanSelection {
            prepared: None,
            declined: Some(message),
        });
    }
    Err(WorkerError::InvalidPayload(message))
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

/// Every plan layer this route does not source, as a typed refusal.
///
/// The inspector emits multi-layer plans — a linked directory Krea checkpoint compiles to
/// `transformer` + `vae` + `text_encoder`, and descriptor/artifact roles appear too. This skeleton
/// sources only the primary layer; the provider's remaining components come from the resident
/// family base tier. Loading the plan's transformer while quietly substituting the base tier's VAE
/// and encoder for the plan's OWN would be a silent substitution, and would mean the plan is not
/// consumed everywhere it is claimed to be. Refuse instead, naming the roles this route cannot
/// source, until sc-20644 maps each remaining role onto a provider component.
fn checkpoint_plan_unconsumed_layers(
    resolved: &ResolvedCheckpointV1,
    primary: &sceneworks_core::checkpoint_plan_store::ResolvedLayerV1,
) -> WorkerResult<()> {
    let unconsumed: Vec<&str> = resolved
        .layers
        .iter()
        .filter(|layer| layer.layer.layer_id != primary.layer.layer_id)
        .map(|layer| layer.layer.role.as_str())
        .collect();
    if unconsumed.is_empty() {
        return Ok(());
    }
    Err(WorkerError::InvalidPayload(format!(
        "[checkpoint-plan:unconsumed-layer] checkpoint {:?} ({} family) compiles to {} layers, but \
         the plan-driven route sources only its primary {:?} layer; layer role(s) [{}] would be \
         silently replaced by the resident base tier's own components, so this checkpoint is not \
         servable on this route yet",
        resolved.checkpoint_id,
        resolved.family(),
        resolved.plan.layers.len(),
        primary.layer.role,
        unconsumed.join(", ")
    )))
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

/// The companion directories the candle floor prices for this plan's declared components.
///
/// Mirrors [`resolve_checkpoint_plan_component`], which is the function that PRODUCED these
/// components: a `base_snapshot` is a resident family tier whose own `transformer/` is replaced by
/// the plan's primary file, so it is priced through the shared
/// [`imported_base_snapshot_companions`] construction (the legacy Krea imported lane's, exactly)
/// rather than recursively. Pricing the whole snapshot — which is what `admit_candle_load_spec_floor`
/// does to every `spec.components` value — would charge the plan's DiT twice and over-refuse.
///
/// Any component shape this floor cannot account for refuses rather than going unpriced.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn checkpoint_plan_candle_companion_dirs(
    sources: &PreparedCheckpointPlanSources,
    spec: &LoadSpec,
) -> WorkerResult<Vec<PathBuf>> {
    let mut companions = Vec::new();
    for (component, source) in &sources.components {
        match (*component, source) {
            (gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base_dir)) => companions.extend(
                imported_base_snapshot_companions(base_dir, spec.text_encoder.is_some()),
            ),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "[checkpoint-plan:unpriceable-component] checkpoint {:?} ({} family) declares \
                     component {component:?} as {source:?}, which this route's candle floor cannot \
                     account for; admitting it would leave those bytes unpriced",
                    sources.checkpoint_id,
                    sources.resolved.family()
                )))
            }
        }
    }
    Ok(companions)
}

/// Select the plan-driven route for a plan-backed manifest entry. An empty selection with no
/// retained refusal happens only when the entry is not plan-backed; a plan-backed entry that cannot
/// be served either refuses here or retains its refusal for the router's fall-through, never a
/// silent fall-through to the stub.
fn prepare_checkpoint_plan_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<CheckpointPlanSelection> {
    let Some(checkpoint_id) = checkpoint_plan_checkpoint_id(&request.model_manifest_entry) else {
        return Ok(CheckpointPlanSelection::default());
    };
    if !checkpoint_plan_serves_request_shape(request) {
        return checkpoint_plan_unservable(
            request,
            format!(
                "[checkpoint-plan:unsupported-operation] checkpoint {checkpoint_id:?} is served \
                 through the plan-driven route for text-to-image generation only; edit, reference, \
                 pose, multi-phase, LoRA, and Hires.fix requests are not on this route yet"
            ),
        );
    }
    let store = CheckpointPlanStore::open(&settings.data_dir);
    // Integrity: never declined, always fatal. A drifted, missing, or tampered plan must not fall
    // through to a lane that would load the same bytes unverified.
    let resolved = store.resolve(checkpoint_id).map_err(checkpoint_plan_refusal)?;
    let family = resolved.family().to_owned();
    let (source, primary_layer) = match checkpoint_plan_primary_layer(&resolved) {
        Ok(primary) => primary,
        Err(error) => return checkpoint_plan_unservable(request, error.to_string()),
    };
    if let Err(error) = checkpoint_plan_unconsumed_layers(&resolved, primary_layer) {
        return checkpoint_plan_unservable(request, error.to_string());
    }
    let Some(descriptor) = crate::inference_runtime::imported_model_descriptor(
        &family,
        source,
        gen_core::ImportedModelOperation::Generate,
    ) else {
        return checkpoint_plan_unservable(
            request,
            format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: this runtime's \
                 provider registry has no {family:?} adapter bound for {source:?} Generate on this \
                 backend"
            ),
        );
    };
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
    Ok(CheckpointPlanSelection::served(
        PreparedCheckpointPlanSources {
            checkpoint_id: checkpoint_id.to_owned(),
            resolved,
            descriptor,
            source,
            primary,
            components,
        },
    ))
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

/// The plan route's MLX request inputs. This skeleton serves plain text-to-image only — no
/// references, adapters, PiD, or phases reach it (`prepare_checkpoint_plan_sources` refuses those
/// shapes) — so this is the identity `krea_imported_memory_inputs(request, &[], None, 0)` produces
/// for the same request, and the two lanes price the same geometry.
#[cfg(target_os = "macos")]
fn checkpoint_plan_memory_inputs(request: &ImageRequest) -> crate::mlx_fit_gate::MlxRequestInputs {
    crate::mlx_fit_gate::MlxRequestInputs {
        width: request.width,
        height: request.height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    }
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
    let engine_id = sources.descriptor.id;

    // Admission is the legacy Krea imported lane's, not a weaker twin (sc-20634 review).
    //
    // macOS: the manifest/geometry-aware request plan plus the request-state seam, so the same
    // weights get the same residency policy, the same warm/loaded-policy settlement, and the same
    // per-request refusal boundary they get on the bespoke lane. `apply_residency_policy` is NOT
    // called here: the generator cache runs it inside its own cold loader, and the bespoke Krea
    // lane leaves it to the cache for exactly that reason.
    //
    // candle: the base snapshot is a companion whose `transformer/` the plan's file replaces, so
    // the floor is the prepared-pin floor over the shared companion construction — never
    // `admit_candle_load_spec_floor`, which prices every component value recursively and would
    // charge the DiT twice.
    #[cfg(target_os = "macos")]
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::try_for_spec_and_manifest(
        engine_id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )?;
    #[cfg(target_os = "macos")]
    let memory_inputs = checkpoint_plan_memory_inputs(request);
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let cold_admission = {
        let companions = checkpoint_plan_candle_companion_dirs(&sources, &spec)?;
        let companion_refs = companions.iter().map(PathBuf::as_path).collect::<Vec<_>>();
        prepare_cached_candle_base_floor(
            &request.model,
            "Checkpoint plan",
            settings,
            &spec,
            &companion_refs,
        )?
    };

    let checkpoint_id = sources.checkpoint_id.clone();
    let generate_one = move |model: &dyn gen_core::Generator,
                             seed: i64,
                             prompt: String,
                             memory: Option<gen_core::GenerationMemory>,
                             context: Option<&gen_core::MemoryRunContext>,
                             preview: gen_core::PreviewSink,
                             cancel: &CancelFlag,
                             on_progress: &mut dyn FnMut(Progress),
                             negative_prompt: Option<String>,
                             checkpoint_id: &str|
          -> WorkerResult<Option<GeneratedImage>> {
        if cancel.is_cancelled() {
            return Ok(None);
        }
        let mut generation = GenerationRequest {
            prompt,
            negative_prompt,
            width,
            height,
            count: 1,
            seed: Some(seed as u64),
            steps,
            guidance,
            preview,
            cancel: cancel.clone(),
            memory,
            ..Default::default()
        };
        // The same seam every request-scoped MLX lane generates through: the selected memory
        // strategy plus its run context. On candle both are `None` and this is the plain call.
        let output = match crate::memory_strategy::generate_with_scope(
            model,
            &mut generation,
            context,
            on_progress,
        ) {
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
    };

    #[cfg(target_os = "macos")]
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("Checkpoint plan {checkpoint_id:?} load failed"),
        move |model,
              initial_cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            let mut request_cache_state = initial_cache_state;
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let evaluation = crate::mlx_fit_gate::evaluate_request(
                    model,
                    &memory_plan,
                    &memory_inputs,
                    request_cache_state,
                    loaded_policy.offload_policy,
                    warm_policy.take(),
                    external_committed_bytes,
                )?;
                request_cache_state = gen_core::MemoryCacheState::Warm;
                let _request_memory_limit = evaluation
                    .process_limit_bytes
                    .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                generate_one(
                    model,
                    seed,
                    prompt,
                    Some(evaluation.memory),
                    Some(&evaluation.context),
                    preview,
                    &cancel,
                    on_progress,
                    negative_prompt.clone(),
                    &checkpoint_id,
                )
            })
        },
    );

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let incoming_reclaimable_weight_bytes = cold_admission.reclaimable_weight_bytes();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (cancel, rx, blocking) = start_cached_gen_stream_after_cold_admission(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("Checkpoint plan {checkpoint_id:?} load failed"),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                generate_one(
                    model,
                    seed,
                    prompt,
                    None,
                    None,
                    preview,
                    &cancel,
                    on_progress,
                    negative_prompt.clone(),
                    &checkpoint_id,
                )
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
