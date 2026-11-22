use super::adapters::PreparedAdapters;
use super::events::consume_gen_events;
use super::load_spec::prepare_manifest_text_encoder_with_file_pins;
use super::prepare_cached_candle_base_floor;
use super::resolve_seed;
use super::routing::PreparedFileDispatch;
use super::sampling::resolved_option_or_manifest_u32;
use super::stream::drive_gen_items;
use super::stream::start_cached_gen_stream_after_cold_admission;
use super::stream::ColdLoadAdmission;
use super::GenerationOutput;
use super::GenerationRequest;
use super::ImageRequest;
use super::LoadSpec;
use super::Quant;
use super::WeightsSource;
use crate::huggingface_snapshot_dir;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use serde_json::json;

// The registered Candle FLUX.2-dev provider remaps the ComfyUI DiT's BFL keys. Its shared
// projection reader consumes the source codes and scales, including activation scales. The
// prepared transformer pin identifies the user's checkpoint for caching and measured facts.
// The companion snapshot supplies the Mistral-3 encoder, AutoencoderKL-Flux2 and tokenizer.

/// The adapter/engine id recorded on candle ComfyUI FLUX.2-dev assets + telemetry (distinct from the
/// registry `candle` flux2 txt2img and the `flux2_dev` edit/control lanes).
pub(super) const FLUX2_COMFYUI_CANDLE_ENGINE: &str = "candle_flux2_dev_comfyui";

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
pub(super) const FLUX2_COMFYUI_SNAPSHOT_TIERS: &[&str] = &["bf16", "q8", "q4"];

/// The PLAN-sourced counterpart of [`resolve_flux2_comfyui_paths`].
///
/// FLUX.2's ComfyUI tree is the DiT alone — its Mistral-3 text encoder, VAE and tokenizer always
/// come from a resident `SceneWorks/flux2-dev-mlx` tier. So NO sidecar role is declared consumed,
/// which means a plan carrying a text encoder or a VAE REFUSES here rather than having it silently
/// replaced by the snapshot tier's own copy (E8).
///
/// `quant` is left at the type default and overwritten by the caller with the registry-validated
/// request selection, exactly as it is on the scan path — the tier is a request choice, not a
/// property of the checkpoint.
fn resolve_plan_backed_flux2_comfyui_paths(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiFlux2Paths>> {
    let Some(tree) = super::checkpoint_plan_bespoke_tree(request, settings, "flux2", &[], &[])?
    else {
        return Ok(None);
    };
    Ok(Some(ComfyuiFlux2Paths {
        transformer: tree.primary,
        snapshot_dir: flux2_comfyui_snapshot_dir(settings).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:missing-component] a compiled FLUX.2 checkpoint needs the \
                 Mistral-3 text encoder, VAE and tokenizer from a resident \
                 {FLUX2_COMFYUI_SNAPSHOT_REPO} tier, which a ComfyUI tree does not ship — install \
                 FLUX.2-dev from the Model Manager, then run this checkpoint again"
            ))
        })?,
        quant: Some(FLUX2_COMFYUI_DEFAULT_QUANT),
        adapters: PreparedAdapters {
            specs: Vec::new(),
            pins: Vec::new(),
        },
    }))
}

/// The first fully-present `SceneWorks/flux2-dev-mlx` tier, or `None`. One probe, both routes.
fn flux2_comfyui_snapshot_dir(settings: &Settings) -> Option<PathBuf> {
    let root = huggingface_snapshot_dir(&settings.data_dir, FLUX2_COMFYUI_SNAPSHOT_REPO)?;
    FLUX2_COMFYUI_SNAPSHOT_TIERS
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| {
            dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
                && dir.join("tokenizer").join("tokenizer.json").is_file()
        })
}

/// Historical default for imports without a verified source format.
pub(super) const FLUX2_COMFYUI_DEFAULT_QUANT: Quant = Quant::Q8;

/// The in-place ComfyUI DiT file + the resident snapshot tier dir supplying the other components.
pub(super) struct ComfyuiFlux2Paths {
    /// ComfyUI FLUX.2-dev DiT (`diffusion_models/flux2_dev_fp8mixed.safetensors`), read in place.
    transformer: gen_core::PinnedWeightsFile,
    /// A resident `SceneWorks/flux2-dev-mlx` tier dir (`text_encoder/ vae/ tokenizer/tokenizer.json`).
    snapshot_dir: PathBuf,
    /// Exact registry-validated on-the-fly tier selected before dispatch.
    quant: Option<Quant>,
    /// Request LoRA/LoKr stack, resolved and pinned once by production preparation. The specs ride
    /// the same `LoadSpec` admission and the provider load consume, so the adapter bytes are priced
    /// from the finalized tokens rather than re-stat'd after the scheduler gap.
    adapters: PreparedAdapters,
}

/// Resolve the ComfyUI FLUX.2-dev DiT path + the resident snapshot tier from the forwarded
/// `external_base_*` row. Returns `Ok(None)` when this is not a runnable ComfyUI FLUX.2-dev job (wrong
/// family, not marked usable, no transformer component, or our FLUX.2-dev snapshot is not resident), so
/// the router falls through rather than erroring. The DiT path is confined by
/// `normalize_app_managed_model_file_path`: its canonical target must stay within a declared root while
/// the lexical entry survives for lstat-pinned re-opening (sc-18306). The snapshot dir is resolved from
/// a fixed repo constant + our own cache (never payload-derived), so it needs no confinement.
pub(super) fn resolve_flux2_comfyui_paths(
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
    let Some(snapshot_dir) = flux2_comfyui_snapshot_dir(settings) else {
        return Ok(None);
    };
    Ok(Some(ComfyuiFlux2Paths {
        transformer: crate::paths::pin_app_managed_model_file(
            settings,
            Path::new(transformer),
            "ComfyUI FLUX.2-dev transformer",
        )?,
        snapshot_dir,
        // Structural resolution does not claim the route by itself; production preparation replaces
        // this default with the exact request selection after registry validation.
        quant: Some(FLUX2_COMFYUI_DEFAULT_QUANT),
        adapters: PreparedAdapters {
            specs: Vec::new(),
            pins: Vec::new(),
        },
    }))
}

/// True when this is a candle-runnable in-place ComfyUI FLUX.2-dev txt2img job: an `external_base_*`
/// model whose forwarded row is a usable flux2 with a transformer component + a resident snapshot, no
/// source image / pose (txt2img only). Mirrors the Qwen-Image comfyui availability predicate.
#[cfg(test)]
pub(super) fn flux2_comfyui_available(request: &ImageRequest, settings: &Settings) -> bool {
    matches!(
        prepare_flux2_comfyui_sources(request, settings),
        Ok(Some(_))
    )
}

/// Exercise the legacy source selector in tests; production dispatch uses the resolved route below.
#[cfg(test)]
pub(super) fn prepare_flux2_comfyui_sources(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiFlux2Paths>> {
    prepare_flux2_comfyui_sources_with_preflight(request, settings, None)
}

pub(super) fn prepare_flux2_comfyui_sources_resolved(
    execution: &super::conditioning::ResolvedImageExecution,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiFlux2Paths>> {
    prepare_flux2_comfyui_sources_with_preflight(execution, settings, Some(execution))
}

fn prepare_flux2_comfyui_sources_with_preflight(
    request: &ImageRequest,
    settings: &Settings,
    execution: Option<&super::conditioning::ResolvedImageExecution>,
) -> WorkerResult<Option<ComfyuiFlux2Paths>> {
    // A plan-backed entry whose REQUEST the plan-driven route claims belongs to that route; this
    // lane never also claims it. Explicit rather than left to the `external_base_` id prefix below:
    // that prefix is an incidental property of how the catalog names a scanned row, and an
    // incidental condition is not a claim (sc-20634 review; sc-20644 FLUX.2 row).
    if super::request_is_checkpoint_plan_backed(request) {
        return Ok(None);
    }
    let descriptor = if let Some(execution) = execution {
        super::routing::imported_generate_request_supported_resolved(
            execution,
            "flux2",
            gen_core::ImportedModelSource::ComfyUiTree,
        )
    } else {
        super::routing::imported_generate_request_supported(
            request,
            "flux2",
            gen_core::ImportedModelSource::ComfyUiTree,
        )
    };
    let Some(_descriptor) = descriptor else {
        return Ok(None);
    };
    let quant = match flux2_comfyui_quant(request) {
        Ok(quant) => quant,
        Err(_) => return Ok(None),
    };
    // A plan-backed entry whose request the plan route does NOT claim runs here on the plan's own
    // verified layers rather than a live `external_base_*` catalog scan — the only source a LINKED
    // ComfyUI tree has (sc-20644 FLUX.2 row).
    let mut paths = match resolve_plan_backed_flux2_comfyui_paths(request, settings)? {
        Some(paths) => paths,
        None => {
            if !request.model.starts_with("external_base_") {
                return Ok(None);
            }
            let Some(paths) = resolve_flux2_comfyui_paths(request, settings)? else {
                return Ok(None);
            };
            paths
        }
    };
    paths.quant = quant;
    paths.adapters = super::adapters::resolve_prepared_adapters(request, settings)?;
    Ok(Some(paths))
}

/// Verified checkpoint formats determine their own precision. Older imports without a codec
/// verdict retain their existing Q4/Q8 fold selection and Q8 default.
pub(super) fn flux2_comfyui_quant(request: &ImageRequest) -> WorkerResult<Option<Quant>> {
    if sceneworks_core::source_precision::SourcePrecision::for_entry(&request.model_manifest_entry)
        .is_some()
    {
        return Ok(None);
    }
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "flux2",
        gen_core::ImportedModelSource::ComfyUiTree,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::Engine(
            "this runtime has no registered ComfyUI FLUX.2 generate route".to_owned(),
        )
    })?;
    let options = request
        .resolved_advanced
        .as_ref()
        .expect("image options must be resolved");
    let authored = |origin| {
        matches!(
            origin,
            sceneworks_core::resolved_options::Origin::Authored { .. }
        )
    };
    let explicit = options
        .quant_tier
        .as_ref()
        .is_some_and(|value| authored(value.origin))
        || options
            .quant
            .as_ref()
            .is_some_and(|value| authored(value.origin))
        || options
            .mlx_quantize
            .as_ref()
            .is_some_and(|value| value.value.is_some() && authored(value.origin));
    let (selected, _) =
        super::krea_imported::imported_model_quant(request, &descriptor, "ComfyUI FLUX.2")?;
    match selected {
        Some(Quant::Q4) => Ok(Some(Quant::Q4)),
        Some(Quant::Q8) => Ok(Some(Quant::Q8)),
        Some(Quant::Nvfp4) => Err(WorkerError::InvalidPayload(
            "ComfyUI FLUX.2 cannot execute the NVFP4 tier; choose q4 or q8.".to_owned(),
        )),
        Some(Quant::Fp8Rowwise) => Err(WorkerError::InvalidPayload(
            "ComfyUI FLUX.2 cannot execute the FP8 rowwise tier; choose q4 or q8.".to_owned(),
        )),
        None if !explicit => Ok(Some(FLUX2_COMFYUI_DEFAULT_QUANT)),
        None => Err(WorkerError::InvalidPayload(
            "ComfyUI FLUX.2 requires q4 or q8 load-time packing; dense/bf16 is not executable."
                .to_owned(),
        )),
    }
}

/// Read the requested embedded-guidance scale from `advanced.guidanceScale` (JSON number or numeric
/// string). `None` ⇒ the candle-gen FLUX.2-dev engine default. The external-base row is in no
/// `MODEL_TABLE`, so `resolve_guidance` (which needs a `ResolvedModel`) does not apply — this reads the
/// same `advanced` knob directly. FLUX.2-dev is guidance-distilled (embedded scalar, no negative pass).
fn flux2_comfyui_guidance(request: &ImageRequest) -> Option<f32> {
    request
        .advanced_f64("guidanceScale")
        .map(|value| value as f32)
}

/// Flat telemetry recorded on candle ComfyUI FLUX.2-dev assets.
fn flux2_comfyui_raw_settings(
    request: &ImageRequest,
    steps: u32,
    guidance: Option<f32>,
    quant: Option<Quant>,
) -> JsonObject {
    let mut raw = request.effective_advanced_wire();
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
    if let Some(quant) = quant {
        let bits = match quant {
            Quant::Q4 => json!(4),
            Quant::Q8 => json!(8),
            Quant::Nvfp4 | Quant::Fp8Rowwise => Value::Null,
        };
        raw.insert("mlxQuantize".into(), bits);
        if quant == Quant::Nvfp4 {
            raw.insert("quantTier".into(), json!("nvfp4"));
        } else {
            raw.remove("quantTier");
        }
    } else {
        sceneworks_core::source_precision::clear_advanced_precision_selectors(&mut raw);
    }
    if let Some(scale) = guidance {
        raw.insert("guidanceScale".into(), json!(scale));
    }
    raw
}

/// Finalizes the imported DiT token and any selected descriptor-validated encoder receipt on the same
/// spec consumed by admission and the registered provider. Default/absence keeps the snapshot encoder
/// implicit.
pub(super) fn prepare_flux2_comfyui_load_spec_for_request(
    paths: ComfyuiFlux2Paths,
    quant: Option<Quant>,
    request: &ImageRequest,
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<(LoadSpec, PathBuf)> {
    let snapshot_dir = paths.snapshot_dir;
    let PreparedAdapters {
        specs: adapters,
        pins: adapter_pins,
    } = paths.adapters;
    let mut spec = LoadSpec::new(WeightsSource::File(
        paths.transformer.loader_path().to_path_buf(),
    ))
    .with_component(
        gen_core::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(snapshot_dir.clone()),
    );
    spec.quantize = quant;
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let mut pins = vec![paths.transformer];
    pins.extend(adapter_pins);
    let spec = prepare_manifest_text_encoder_with_file_pins(
        spec,
        engine_id,
        request,
        settings,
        pins,
        "ComfyUI FLUX.2 source preparation failed",
    )?;
    Ok((spec, snapshot_dir))
}

/// Real candle in-place ComfyUI FLUX.2-dev txt2img generation through the registered `flux2_dev`
/// provider. `request.count` images, each its own seed. FLUX.2-dev is guidance-distilled (embedded
/// scalar, single forward — NO negative prompt / true-CFG pass).
pub(super) async fn generate_candle_flux2_comfyui_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    dispatch: PreparedFileDispatch<'_, ComfyuiFlux2Paths>,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let PreparedFileDispatch {
        plan,
        sources: paths,
    } = dispatch;
    let request = &plan.request;
    let checkpoint = super::stream::ImportedCheckpointPublisher::new(
        &request.model_manifest_entry,
        &paths.transformer,
    );
    request
        .reject_active_pag("ComfyUI FLUX.2")
        .map_err(WorkerError::InvalidPayload)?;
    let sampling_controls = super::sampling::explicit_sampling_request(request);
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "flux2",
        gen_core::ImportedModelSource::ComfyUiTree,
        gen_core::ImportedModelOperation::Generate,
    )
    .ok_or_else(|| {
        WorkerError::Engine(
            "this runtime has no registered ComfyUI FLUX.2 generate route".to_owned(),
        )
    })?;
    let (width, height) = (request.width, request.height);
    let steps =
        resolved_option_or_manifest_u32(request, "steps", FLUX2_COMFYUI_DEFAULT_STEPS, 1..=50);
    let guidance = flux2_comfyui_guidance(request);
    let quant = paths.quant;
    let raw_settings = flux2_comfyui_raw_settings(request, steps, guidance, quant);
    let (spec, snapshot_dir) = prepare_flux2_comfyui_load_spec_for_request(
        paths,
        quant,
        request,
        settings,
        descriptor.id,
    )?;
    // The companion snapshot supplies TE/VAE/tokenizer/config only. Its own transformer is replaced;
    // finalized File bytes and only the consumed companion dirs feed admission.
    let selected_text_encoder = spec.text_encoder.is_some();
    let snapshot_text_encoder = snapshot_dir.join("text_encoder");
    let snapshot_vae = snapshot_dir.join("vae");
    let mut companion_paths = vec![snapshot_vae];
    if !selected_text_encoder {
        companion_paths.push(snapshot_text_encoder);
    }
    let companion_refs = companion_paths
        .iter()
        .map(PathBuf::as_path)
        .collect::<Vec<_>>();
    let cold_admission = prepare_cached_candle_base_floor(
        &request.model,
        "ComfyUI FLUX.2",
        settings,
        &spec,
        &companion_refs,
    )?;

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
        "ComfyUI FLUX.2-dev load failed".to_owned(),
        ColdLoadAdmission::new(
            incoming_reclaimable_weight_bytes,
            move |resident_reclaimable_weight_bytes| {
                cold_admission.admit(resident_reclaimable_weight_bytes)
            },
        ),
        move |model, tx, cancel| {
            drive_gen_items(
                tx.clone(),
                work,
                move |index, (seed, prompt), preview, on_progress| {
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
                        preview,
                        cancel: cancel.clone(),
                        ..sampling_controls.clone()
                    };
                    let completed =
                        match model.generate_with_execution_report(&request, &mut *on_progress) {
                            Ok(completed) => completed,
                            // A poisoned process keeps its fatal class even when cancellation raced it.
                            Err(error @ gen_core::Error::ProcessPoisoned(_)) => {
                                return Err(super::render::generation_error(
                                    "ComfyUI FLUX.2-dev generation failed",
                                    error,
                                ));
                            }
                            Err(_) if cancel.is_cancelled() => return Ok(None),
                            Err(error) => {
                                return Err(crate::generation_error(
                                    "ComfyUI FLUX.2-dev generation failed",
                                    error,
                                ));
                            }
                        };
                    match completed.generation.output {
                        Some(GenerationOutput::Images(mut images)) if images.len() == 1 => {
                            let image = images.pop().ok_or_else(|| {
                                WorkerError::Engine(
                                    "ComfyUI FLUX.2-dev produced no image".to_owned(),
                                )
                            })?;
                            let mut facts = crate::JsonObject::new();
                            facts.insert(
                                "workerExecutionReport".into(),
                                serde_json::to_value(completed.execution)
                                    .map_err(WorkerError::Json)?,
                            );
                            let _ = tx.blocking_send(super::stream::GenEvent::ImageFacts {
                                index,
                                facts,
                            });
                            checkpoint.publish(&tx, model);
                            Ok(Some((seed, image.width, image.height, image.pixels)))
                        }
                        _ => Err(WorkerError::Engine(
                            "ComfyUI FLUX.2-dev returned non-image output".to_owned(),
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
