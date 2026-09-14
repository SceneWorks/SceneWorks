use super::{
    admit_candle_base, apply_candle_image_load_shape, attach_manifest_text_encoder,
    candle_certified_artifact_path, candle_conditioned_edit_work, candle_quant_for_resolved_tier,
    candle_resolved_tier_key, consume_gen_events, drive_gen_items_reported, fit_engine_image,
    load_reference_image, mlx_model, model_repo, pid_effective_dims, pid_output_tier,
    resolve_adapters, resolve_advanced_or_manifest_f32, resolve_advanced_or_manifest_u32,
    resolve_pid_weights, resolve_weights_dir, start_gen_stream, ApiClient, CandleBaseEvidence,
    Flux2Edit, Flux2EditPaths, Flux2EditRequest, Image, ImagePlan, ImageRequest, JobSnapshot,
    JsonObject, Path, PathBuf, PromptEnhance, Settings, Value, WorkerError, WorkerResult,
};
use serde_json::json;

pub(super) fn flux2_edit_adapter_source_bytes(
    adapters: &[gen_core::AdapterSpec],
) -> WorkerResult<u64> {
    gen_core::adapter_stack_resident_bytes(adapters, gen_core::AdapterResidencyMode::Additive)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "FLUX.2 edit cannot determine the resident size of the requested adapter stack."
                    .to_owned(),
            )
        })
}

// Candle (Windows/CUDA) FLUX.2 image-edit route (sc-5487 klein, epic 5480; sc-7736 dev, epic 6564) —
// Kontext-style reference-conditioned editing off-Mac via `runtime_cuda::providers::flux2::Flux2Edit`. FLUX.2-klein has
// no torch path (it is diffusers/MLX-only), so before this an off-Mac `edit_image` job on klein had no
// real lane; this routes it to candle instead of the torch fallback. **sc-7736** generalizes the lane to
// the 32B **dev** flagship (`flux2_dev`): the same DiT token-concat edit, loaded via the Q4 CPU-stage →
// quantize-onto-GPU path (`Flux2Edit::load_dev`) with embedded distilled guidance (no negative pass).
//
// **Candle-only.** macOS keeps the MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the
// candle `Flux2Edit` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build
// (the module declaration in image_jobs.rs carries the cfg). It is a child module of the `image_jobs` module, so
// it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_quant`/`resolve_seed`/
// `start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).
//
// FLUX.2 edit is a single-reference (or multi-reference) token concat, NOT a strength-based img2img +
// mask: the source is the reference, the prompt is the instruction. So this lane has no sub-modes /
// inpaint / outpaint (unlike the SDXL edit lane) — it handles `edit_image` + `sourceAssetId` (plus the
// multi-image picker's plural `referenceAssetIds`, sc-6211 parity).

/// FLUX.2-klein denoise steps default (distilled klein generates in 4).
const FLUX2_EDIT_CANDLE_DEFAULT_STEPS: u32 = 4;
/// Guidance default — distilled klein runs CFG-free at 1.0 (a single forward; >1.0 adds a negative pass).
const FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 1.0;
/// FLUX.2-dev denoise steps default (the guidance-distilled 32B; FLUX.1-dev pattern, ~28 steps).
const FLUX2_EDIT_CANDLE_DEV_STEPS: u32 = 28;
/// FLUX.2-dev embedded-guidance default (distilled scalar, NOT true-CFG — no negative pass).
const FLUX2_EDIT_CANDLE_DEV_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle FLUX.2 edit assets + telemetry (distinct from the txt2img
/// `candle_flux2` lane). Shared by klein + dev edit (the dev variant is the same edit surface).
pub(super) const FLUX2_EDIT_CANDLE_ENGINE: &str = "candle_flux2_edit";
/// Cap on references fed to a single FLUX.2 edit (the multi-image picker, sc-6211): the dev edit is
/// activation-bound, so cap at the engine's validated native fan-out. This deliberately differs from
/// the MLX `MAX_EDIT_REFERENCES` (4): that bound is set by the 96 GB Apple-Silicon unified-memory
/// floor (5 refs at 1024² exceed it), which does not apply to this off-Mac CUDA lane — so it admits
/// the engine's full 5-reference fan-out (sc-8936: was mislabeled "parity with the MLX const").
const FLUX2_EDIT_CANDLE_MAX_REFERENCES: usize = 5;

/// True when this is the FLUX.2 **dev** edit variant (`flux2_dev`): the 32B flagship that loads via the
/// Q4 quantize-onto-GPU path with embedded distilled guidance, vs the dense klein family.
fn is_flux2_edit_candle_dev(model: &str) -> bool {
    model == "flux2_dev"
}

/// True for the Klein catalog family that shares the provider implementation. Keeping this
/// separate from `flux2_dev` keeps family-specific loading behavior explicit.
fn is_flux2_edit_candle_klein(model: &str) -> bool {
    matches!(
        model,
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2"
    )
}

/// FLUX.2 model ids the candle edit route accepts. The three Klein catalog entries resolve to the
/// same Candle provider implementation; artifact/evidence admission remains entry-specific, so
/// sharing this execution route cannot promote KV or True V2 calibration cells.
pub(super) fn is_flux2_edit_candle_model(model: &str) -> bool {
    matches!(
        model,
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2" | "flux2_dev"
    )
}

/// True when this is a reference-bearing FLUX.2 request. Every admitted mode must resolve at least
/// one concrete reference; otherwise it stays off the bespoke route and the generic Klein guard
/// fails closed instead of silently rendering reference-free text-to-image.
pub(super) fn flux2_edit_candle_mode(request: &ImageRequest) -> bool {
    let supported_mode = (is_flux2_edit_candle_klein(&request.model)
        || is_flux2_edit_candle_dev(&request.model))
        && matches!(
            request.mode.as_str(),
            "edit_image" | "reference" | "image_to_image" | "character_image" | "style_variations"
        );
    supported_mode
        && flux2_edit_candle_pose_carrier_is_absent_or_empty(request)
        && !flux2_edit_candle_reference_ids(request).is_empty()
}

/// The FLUX.2 edit provider consumes references but no pose controls. Missing/null/empty preserves
/// ordinary character/reference edits; any non-empty or malformed pose carrier must stay off this
/// lane so the control route can consume it or the worker can reject it explicitly.
fn flux2_edit_candle_pose_carrier_is_absent_or_empty(request: &ImageRequest) -> bool {
    match request.advanced.get("poses") {
        None | Some(Value::Null) => true,
        Some(Value::Array(poses)) => poses.is_empty(),
        Some(_) => false,
    }
}

/// Resolve the FLUX.2 base snapshot through the **shared** [`resolve_weights_dir`] — the same resolver
/// the candle txt2img lane uses (sc-10222, epic 9083 gap #3).
///
/// This lane used to key its own `black-forest-labs/FLUX.2-{klein-9B,dev}` dense-BFL constants. The
/// sc-9092 sweep that retired the ad-hoc candle repo resolvers into `standard_tier_subdir` covered
/// ideogram/boogu/krea/lens + the generic txt2img lane but missed this bespoke edit lane, so it kept
/// probing the pre-rehost gated repos. The catalog's `downloads[]` pull ONLY the re-hosted
/// `SceneWorks/flux2-*-mlx` packed q4/q8/bf16 turnkeys (sc-8711 / sc-8513) — there is no dense-BFL
/// download on any platform — so off-Mac the probe found nothing, `resolve_flux2_edit_candle_base`
/// returned `None`, and every FLUX.2 edit job silently failed the candle-eligibility check.
///
/// Delegating fixes it end to end: `resolve_weights_dir` takes the `modelPath` override first (the
/// `flux2_klein_9b_true_v2` convert-at-install seam, unchanged), then the engines.rs `default_repo`
/// (the SceneWorks turnkey — one source of truth, so a future re-host can't drift this lane again),
/// then descends into the request's tier via `standard_tier_subdir` for the `STANDARD_TIER_MODELS`
/// members (`flux2_klein_9b`, `flux2_dev`).
///
/// **No engine change was needed.** `Flux2Edit::load{,_dev}` builds its TE + DiT through the shared
/// `Pipeline::load_te_and_dit`, whose every projection is a `QLinear::linear_detect` — it packed-detects
/// a `.scales` sibling per tensor independently of the load `Quant`, and degrades to dense when there
/// is none. That is the identical code path the candle FLUX.2 **txt2img** lane has been loading these
/// turnkeys through since sc-9092; only the DIRECTORY handed to it was wrong. `None` still means the
/// base is not present locally, so the job is not candle-runnable. Mirrors `krea_edit_candle_available`
/// (the newer candle edit lane, already on the shared resolver).
pub(super) fn resolve_flux2_edit_candle_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    resolve_weights_dir(request, settings)
}

/// The FLUX.2 base repo recorded on the asset telemetry: the shared manifest-`repo`-else-`default_repo`
/// resolution ([`model_repo`]), so the recipe names the turnkey the load actually read rather than this
/// lane's own constant (sc-10222). Falls back to the model id for an id outside the engine table (only
/// reachable from a test payload — `is_flux2_edit_candle_model` gates the lane to three known ids).
pub(super) fn flux2_edit_candle_repo(request: &ImageRequest) -> String {
    match mlx_model(&request.model) {
        Some(model) => model_repo(request, &model),
        None => request.model.clone(),
    }
}

/// True when this is a candle-eligible reference-bearing FLUX.2 job whose base resolves locally.
pub(super) fn flux2_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux2_edit_candle_model(&request.model)
        && flux2_edit_candle_mode(request)
        && matches!(
            resolve_flux2_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → the family default
/// (klein 4 / dev 28).
fn flux2_edit_candle_steps(request: &ImageRequest, default: u32) -> u32 {
    resolve_advanced_or_manifest_u32(request, "steps", default, 1..=50)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the family default
/// (klein 1.0 / dev 4.0), clamped.
fn flux2_edit_candle_guidance(request: &ImageRequest, default: f32) -> f32 {
    if let Some(value) = request.advanced.get("trueCfgScale").and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        return (value as f32).clamp(0.0, 30.0);
    }
    resolve_advanced_or_manifest_f32(request, "guidanceScale", default, 0.0..=30.0)
}

/// Reference asset ids for a FLUX.2 edit, in order. The multi-image picker (sc-6211) sends the plural
/// `referenceAssetIds` — take all of them, capped at [`FLUX2_EDIT_CANDLE_MAX_REFERENCES`]; with no plural
/// list it falls back to the character/reference `referenceAssetId`, then the Image-Edit
/// `sourceAssetId`. Mirrors the MLX `edit_reference_ids` order.
pub(super) fn flux2_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // The parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(FLUX2_EDIT_CANDLE_MAX_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
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

// Fit/letterbox geometry is the SHARED `fit_engine_image` (base.rs), not a per-lane twin (sc-8824):
// the same helper the macOS FLUX.2 edit lane and the candle Z-Image edit lane use. Its pad/outpaint
// arm rides `gen_core::imageops::contain_box`, so an off-aspect Image-Edit source letterboxes with
// exactly the geometry any outpaint mask would, with no per-lane rounding drift.

/// Flat telemetry recorded on candle FLUX.2 edit assets (parity with the macOS FLUX.2 edit recipe keys).
fn flux2_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    quant_bits: Option<i64>,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(FLUX2_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Load the FLUX.2 edit reference set: the `N ∈ [1, 5]` reference images (plural `referenceAssetIds`,
/// else the single `sourceAssetId` — [`flux2_edit_candle_reference_ids`]), each pre-fit to the render
/// W×H (crop / pad / outpaint→pad; `stretch` keeps the legacy non-aspect resize). The provider
/// re-resizes internally, but pre-fitting keeps an off-aspect edit from stretching. Errors if no source.
fn load_flux2_edit_references(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<Image>> {
    let ids = flux2_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires a source image".to_owned(),
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

/// Real candle FLUX.2 edit generation: resolve the source(s) + base on the async side, pre-fit each to
/// the render geometry, then load `Flux2Edit` once + generate each image on the blocking thread.
/// `request.count` edits of the same reference set, each its own seed. dev (`flux2_dev`) loads Q4 via
/// `load_dev` with embedded distilled guidance; klein loads dense (CFG-free at guidance 1.0; >1 adds a
/// negative pass). `generate` takes `&self`, so the per-item closure needs no `mut`. Reuses
/// [`consume_gen_events`].
pub(super) async fn generate_candle_flux2_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux2_base = resolve_flux2_edit_candle_base(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2 base not found".to_owned()))?;
    if !flux2_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 reference-bearing mode requires a reference image".to_owned(),
        ));
    }
    let adapters = resolve_adapters(request, settings)?;
    let adapter_source_bytes = flux2_edit_adapter_source_bytes(&adapters)?;
    // The canonical Klein base is admitted exclusively by the exact shared selector below. The
    // legacy resident-only gate cannot model its bounded rungs and would reject constrained
    // requests before sequential offload/decode/attention/block choices are evaluated. Siblings
    // remain executable through the resident legacy path, but use `Ungateable` evidence so code
    // sharing cannot falsely certify entry-specific CUDA calibration.
    if request.model != "flux2_klein_9b" {
        let evidence = match request.model.as_str() {
            "flux2_klein_9b_kv" => CandleBaseEvidence::Ungateable(
                "the KV catalog entry has no entry-specific CUDA peak or calibration row",
            ),
            "flux2_klein_9b_true_v2" => CandleBaseEvidence::Ungateable(
                "the local True V2 converted fine-tune has no CUDA calibration row",
            ),
            _ => CandleBaseEvidence::Catalog,
        };
        admit_candle_base(
            request,
            settings,
            &flux2_base,
            "FLUX.2 edit",
            evidence,
            adapter_source_bytes,
            false,
        )
        .await?;
    }
    let is_dev = is_flux2_edit_candle_dev(&request.model);
    // Per-generation PiD decode (epic 7840, sc-8044) + output tier (sc-10054), resolved BEFORE the
    // references are loaded/fit so a 2K tier sizes the effective base and the edit references land at THAT
    // base (references + latent stay aligned). `use_pid`/`with_pid` stay paired below.
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    let (width, height) = pid_effective_dims(
        request.width,
        request.height,
        use_pid,
        pid_output_tier(request),
    );
    let references = load_flux2_edit_references(request, project_path, settings, width, height)?;

    // Since sc-10222 `flux2_base` is the RESOLVED tier subdir (`q4/`/`q8/`/`bf16/`), so the tier the
    // load reads is chosen by the DIRECTORY, not by this `Quant` — every projection packed-detects its
    // `.scales` sibling. dev still resolves a `Quant` (it is the value `resolve_quant` records on the
    // recipe, and it drives the dense CPU-stage → quantize-onto-GPU fallback when the resolved dir is a
    // dense/`modelPath` tree rather than a packed turnkey). klein keeps a hardcoded `(None, None)`: it is
    // a DENSE-TE turnkey (`mlx.denseTextEncoderTier`) whose bf16 Qwen3 text encoder must never be
    // re-quantized — `resolve_quant`'s `is_dense_te_tier` carve-out returns exactly this for
    // `flux2_klein_9b`, and the hardcode additionally keeps `_true_v2` (a convert-at-install dense dir,
    // NOT in that list) on the dense load it has always used.
    //
    // The dev edit is activation-bound — multi-reference adds latent tokens to the DiT stream — but the
    // candle engine query-row-chunks its joint attention (sc-6217/sc-7523), so a device OOM surfaces as a
    // load/generate error rather than silently corrupting; no Mac-style unified-memory pre-guard applies.
    let tier = candle_resolved_tier_key(request, &flux2_base, false);
    let (quant, quant_bits) =
        candle_quant_for_resolved_tier(request, tier, &flux2_base, is_dev, false);
    let mut strategy_spec =
        gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(flux2_base.clone()))
            .with_adapters(adapters.clone())
            .with_offload_policy(gen_core::OffloadPolicy::Sequential);
    strategy_spec.quantize = quant;
    let memory_provider = if is_dev {
        "flux2_dev"
    } else {
        "flux2_klein_9b"
    };
    if let Some(pid) = pid_weights.as_ref() {
        strategy_spec = strategy_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
    let strategy_spec = apply_candle_image_load_shape(memory_provider, strategy_spec);
    let unattached_strategy_spec = strategy_spec;
    let attached_strategy_spec =
        attach_manifest_text_encoder(unattached_strategy_spec, memory_provider, request, settings)?;
    let strategy_spec = attached_strategy_spec.into_load_spec();
    let mut generation_memory = gen_core::GenerationMemory::default();
    let raw_budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    let predicted_peak = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        adapter_source_bytes,
    );
    let memory_evaluation = crate::candle_memory_strategy::evaluate_shared_image(
        memory_provider,
        &request.model,
        &strategy_spec,
        candle_certified_artifact_path(memory_provider, settings, &flux2_base, tier),
        &request.model_manifest_entry,
        tier,
        &request.mode,
        None,
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
        raw_budget,
        raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
        predicted_peak,
        adapter_source_bytes,
        gen_core::MemoryCacheState::Cold,
    )?;
    if request.model == "flux2_klein_9b" && memory_evaluation.is_none() {
        return Err(WorkerError::InvalidPayload(
            "no exact verified memory strategy fits this FLUX.2 Klein reference request".to_owned(),
        ));
    }
    if let Some(evaluation) = &memory_evaluation {
        generation_memory = evaluation.memory.unwrap_or_default();
    }
    let memory_context = memory_evaluation
        .as_ref()
        .map(|evaluation| evaluation.context.clone());
    let steps = flux2_edit_candle_steps(
        request,
        if is_dev {
            FLUX2_EDIT_CANDLE_DEV_STEPS
        } else {
            FLUX2_EDIT_CANDLE_DEFAULT_STEPS
        },
    );
    let guidance = flux2_edit_candle_guidance(
        request,
        if is_dev {
            FLUX2_EDIT_CANDLE_DEV_GUIDANCE
        } else {
            FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE
        },
    );
    let repo = flux2_edit_candle_repo(request);
    // `pid_weights`/`use_pid`/`width`/`height` were resolved above (ahead of the reference fit) so the
    // PiD output tier (sc-10054) could size the effective base; `use_pid`/`with_pid` stay in lockstep.
    let mut raw_settings = flux2_edit_candle_raw_settings(
        request,
        &repo,
        steps,
        guidance,
        quant_bits,
        references.len(),
    );
    // Mark PiD output on the sidecar (NSCLv1 NC flows to PiD output); record whether PiD actually ran.
    raw_settings.insert("usePid".to_owned(), Value::Bool(use_pid));
    if let Some(evaluation) = &memory_evaluation {
        raw_settings.insert(
            "memoryStrategy".to_owned(),
            Value::String(format!("{:?}", evaluation.context.selection.strategy)),
        );
    }

    // Per-image work items: (seed, prompt) — `request.count` edits of the same reference set.
    let work = candle_conditioned_edit_work(request);
    let total = work.len();
    let negative = request.negative_prompt.clone();
    let enhance = PromptEnhance::from_advanced(&request.advanced)?;
    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "flux2_edit",
        0,
        move || {
            let paths = Flux2EditPaths {
                root: flux2_base,
                adapters,
            };
            let model = if is_dev {
                match &memory_context {
                    Some(context) => Flux2Edit::load_dev_with_memory_context(
                        &paths,
                        quant,
                        &strategy_spec,
                        context,
                    ),
                    None => Flux2Edit::load_dev_with_memory_spec(
                        &paths,
                        quant,
                        &strategy_spec,
                        generation_memory,
                    ),
                }
            } else {
                match &memory_context {
                    Some(context) => {
                        Flux2Edit::load_klein_with_memory_context(&paths, &strategy_spec, context)
                    }
                    None => Flux2Edit::load_klein_with_memory_spec(
                        &paths,
                        &strategy_spec,
                        generation_memory,
                    ),
                }
            }
            .map_err(|error| WorkerError::Engine(format!("FLUX.2 edit load failed: {error}")))?;
            // Attach the optional PiD decoder (sc-8044): `Some` only when opted in AND snapshots cached.
            let model = match &pid_weights {
                Some(pid) => model.with_pid(pid).map_err(|error| {
                    WorkerError::Engine(format!("FLUX.2 edit PiD decoder load failed: {error}"))
                })?,
                None => model,
            };
            Ok((model, references, memory_context))
        },
        move |(model, references, memory_context), tx, cancel| {
            drive_gen_items_reported(
                tx,
                work,
                move |_index, (seed, prompt), preview, prompt_enhancement, on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    let req = Flux2EditRequest {
                        prompt: prompt.clone(),
                        negative: negative.clone(),
                        width,
                        height,
                        steps: steps as usize,
                        guidance,
                        seed: seed as u64,
                        // PiD opt-in (sc-8044): in lockstep with the `with_pid` load above.
                        use_pid,
                        enhance_prompt: enhance.enabled,
                        enhance_max_tokens: enhance.max_tokens,
                        enhance_temperature: enhance.temperature,
                        prompt_enhancement: if enhance.enabled {
                            prompt_enhancement.for_prompt(&prompt)
                        } else {
                            gen_core::PromptEnhancementSink::default()
                        },
                        preview: preview.clone(),
                        cancel: cancel.clone(),
                    };
                    let result = match memory_context.as_ref() {
                        Some(context) => model.generate_with_memory_context(
                            context,
                            &req,
                            &references,
                            &mut *on_progress,
                        ),
                        None => model.generate(&req, &references, &mut *on_progress),
                    };
                    let out = match result {
                        Ok(out) => out,
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => {
                            return Err(WorkerError::Engine(format!(
                                "FLUX.2 edit generation failed: {error}"
                            )));
                        }
                    };
                    Ok(Some((seed, out.width, out.height, out.pixels)))
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
        FLUX2_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(advanced: Value, manifest_guidance: Option<f64>) -> ImageRequest {
        let mut value = json!({
            "projectId": "p", "model": "flux2_dev", "mode": "character_image",
            "prompt": "portrait", "referenceAssetId": "ref", "advanced": advanced
        });
        if let Some(guidance) = manifest_guidance {
            value.as_object_mut().unwrap().insert(
                "modelManifestEntry".to_owned(),
                json!({ "guidanceScale": guidance }),
            );
        }
        ImageRequest::from_payload(value.as_object().expect("image request object"))
    }

    #[test]
    fn guidance_prefers_true_cfg_then_guidance_manifest_and_default() {
        assert_eq!(
            flux2_edit_candle_guidance(
                &request(
                    json!({ "trueCfgScale": 7.0, "guidanceScale": 2.0 }),
                    Some(3.0)
                ),
                4.0,
            ),
            7.0
        );
        assert_eq!(
            flux2_edit_candle_guidance(&request(json!({ "guidanceScale": 2.0 }), Some(3.0)), 4.0),
            2.0
        );
        assert_eq!(
            flux2_edit_candle_guidance(&request(json!({}), Some(3.0)), 4.0),
            3.0
        );
        assert_eq!(
            flux2_edit_candle_guidance(&request(json!({}), None), 4.0),
            4.0
        );
    }
}
