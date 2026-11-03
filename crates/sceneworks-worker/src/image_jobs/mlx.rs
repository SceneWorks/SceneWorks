#[cfg(target_os = "macos")]
use super::adapters::resolve_adapters;
#[cfg(target_os = "macos")]
use super::artifact_resolution::ensure_boogu_tier_present;
#[cfg(target_os = "macos")]
use super::artifact_resolution::ensure_ideogram_tier_present;
#[cfg(target_os = "macos")]
use super::artifact_resolution::model_path_override;
#[cfg(target_os = "macos")]
use super::artifact_resolution::model_repo;
#[cfg(target_os = "macos")]
use super::artifact_resolution::requested_receipt_variant;
#[cfg(target_os = "macos")]
use super::artifact_resolution::resolve_weights_dir;
#[cfg(target_os = "macos")]
use super::artifact_resolution::resolved_mlx_artifact_provenance;
#[cfg(target_os = "macos")]
use super::artifact_resolution::resolved_mlx_artifact_tier_for_model;
#[cfg(target_os = "macos")]
use super::artifact_resolution::split_repo_root_tier_key;
#[cfg(target_os = "macos")]
use super::artifact_resolution::tier_key_for_resolved_dir;
#[cfg(target_os = "macos")]
use super::attach_prompt_upsampler;
#[cfg(target_os = "macos")]
use super::conditioning::resolve_character_image_likeness_source;
#[cfg(target_os = "macos")]
use super::conditioning::resolve_generic_lane_inputs;
#[cfg(target_os = "macos")]
use super::conditioning::resolve_text_style_gain;
#[cfg(target_os = "macos")]
use super::conditioning::stage_likeness;
#[cfg(target_os = "macos")]
use super::conditioning::LaneConditioning;
#[cfg(target_os = "macos")]
use super::events::consume_gen_events;
#[cfg(target_os = "macos")]
use super::flux3_precision;
#[cfg(target_os = "macos")]
use super::flux3_references;
#[cfg(target_os = "macos")]
use super::is_dense_te_tier;
#[cfg(target_os = "macos")]
use super::lane_controls;
#[cfg(target_os = "macos")]
use super::load_policy::admission_needs_no_staging;
#[cfg(target_os = "macos")]
use super::load_policy::declared_sequential_resident_alternative;
#[cfg(target_os = "macos")]
use super::load_policy::prepare_mlx_load_policy;
#[cfg(target_os = "macos")]
use super::load_spec::attach_manifest_text_encoder;
#[cfg(target_os = "macos")]
use super::load_spec::attach_required_components;
#[cfg(target_os = "macos")]
use super::load_spec::attach_selected_decoder;
#[cfg(target_os = "macos")]
use super::load_spec::load_spec;
#[cfg(target_os = "macos")]
use super::memory::downtier_candidate_tiers;
#[cfg(all(test, target_os = "macos"))]
use super::memory::TierFit;
#[cfg(target_os = "macos")]
use super::min_quality_floor;
#[cfg(target_os = "macos")]
use super::mlx_model;
#[cfg(target_os = "macos")]
use super::moss::generate_moss_stream_using;
#[cfg(target_os = "macos")]
use super::pid::pid_effective_dims;
#[cfg(target_os = "macos")]
use super::pid::pid_output_tier;
#[cfg(target_os = "macos")]
use super::pid::resolve_pid_weights;
#[cfg(target_os = "macos")]
use super::quantization::dense_te_requested_tier_bits;
#[cfg(target_os = "macos")]
use super::quantization::fixed_mlx_artifact_quant;
#[cfg(target_os = "macos")]
use super::quantization::mlx_load_quant_for_resolved_artifact;
#[cfg(target_os = "macos")]
use super::quantization::reconcile_resolved_tier_quant;
#[cfg(target_os = "macos")]
use super::quantization::resolve_quant;
#[cfg(target_os = "macos")]
use super::quantization::resolve_tier_dir;
#[cfg(target_os = "macos")]
use super::quantization::tier_key_from_resolved_dir;
#[cfg(target_os = "macos")]
use super::quantization::tier_quant_from_resolved_dir;
#[cfg(target_os = "macos")]
use super::render::generate_one_with_hires_on_surface;
#[cfg(target_os = "macos")]
use super::render::hires_fix_reference_count;
#[cfg(target_os = "macos")]
use super::render::lane_reference_count;
#[cfg(target_os = "macos")]
use super::render::render_on_surface;
#[cfg(target_os = "macos")]
use super::render::resolve_hires_fix_plan;
#[cfg(target_os = "macos")]
use super::render::LaneRgbaSurface;
#[cfg(target_os = "macos")]
use super::resolve_seed;
#[cfg(target_os = "macos")]
use super::sampling::mlx_raw_settings_for_bundle;
#[cfg(target_os = "macos")]
use super::sampling::normalize_sampling_knob;
#[cfg(target_os = "macos")]
use super::sampling::read_advanced_guidance_method;
#[cfg(target_os = "macos")]
use super::sampling::read_advanced_sampling_knobs;
#[cfg(target_os = "macos")]
use super::sampling::recover_ideogram_placeholder;
#[cfg(target_os = "macos")]
use super::sampling::resolve_guidance;
#[cfg(target_os = "macos")]
use super::sampling::resolve_negative_prompt;
#[cfg(target_os = "macos")]
use super::sampling::resolve_steps;
#[cfg(target_os = "macos")]
use super::sampling::resolve_true_cfg;
#[cfg(target_os = "macos")]
use super::stream::drive_gen_items_scored_reported;
#[cfg(target_os = "macos")]
use super::stream::drive_native_prompt_batch;
#[cfg(target_os = "macos")]
use super::stream::start_cached_gen_stream_with_request_state_using;
#[cfg(target_os = "macos")]
use super::Generator;
#[cfg(target_os = "macos")]
use super::Image;
#[cfg(target_os = "macos")]
use super::ImagePlan;
#[cfg(target_os = "macos")]
use super::ImageRequest;
#[cfg(target_os = "macos")]
use super::LoadSpec;
#[cfg(target_os = "macos")]
use super::Progress;
#[cfg(target_os = "macos")]
use super::PromptEnhance;
#[cfg(target_os = "macos")]
use super::Quant;
#[cfg(target_os = "macos")]
use super::ResolvedModel;
#[cfg(target_os = "macos")]
use crate::ApiClient;
#[cfg(target_os = "macos")]
use crate::JobSnapshot;
#[cfg(target_os = "macos")]
use crate::Path;
#[cfg(any(test, target_os = "macos"))]
use crate::PathBuf;
#[cfg(target_os = "macos")]
use crate::Settings;
#[cfg(target_os = "macos")]
use crate::Value;
#[cfg(any(test, target_os = "macos"))]
use crate::WorkerError;
#[cfg(any(test, target_os = "macos"))]
use crate::WorkerResult;
/// Historical whole-component fit, retained as a regression witness. Production tier selection
/// now evaluates the complete request ladder through `choose_mlx_request_tier`.
#[cfg(all(target_os = "macos", test))]
pub(super) fn mlx_tier_fit(engine_id: &str, spec: &LoadSpec) -> TierFit {
    match crate::mlx_fit_gate::decide_residency_for_spec(engine_id, spec) {
        crate::mlx_fit_gate::ResidencyOutcome::Resident
        | crate::mlx_fit_gate::ResidencyOutcome::Sequential => TierFit::Fits,
        crate::mlx_fit_gate::ResidencyOutcome::Reject {
            needed_gb,
            available_gb,
            ..
        } => TierFit::TooBig {
            needed_gb,
            available_gb,
        },
    }
}

/// Complete immutable input to both pre-load tier selection and the eventual cached generation.
#[cfg(target_os = "macos")]
struct PreparedMlxImageTier {
    weights_dir: PathBuf,
    spec: LoadSpec,
    plan: crate::mlx_fit_gate::MlxRequestPlan,
    inputs: crate::mlx_fit_gate::MlxRequestInputs,
    /// The manifest declaration alone made this spec Sequential while the fit gate admits the
    /// eager Resident load (sc-24806); the tier chooser may re-prepare it Resident.
    has_resident_alternative: bool,
}

#[cfg(any(target_os = "macos", test))]
pub(crate) enum MlxTierFit<T> {
    Fits(T),
    TooBig(WorkerError),
}

/// Stop at the highest-fidelity fitting tier. Contract/source failures propagate; only an actual
/// memory refusal can try the next installed tier. Preserve the last evaluated diagnostic.
#[cfg(any(target_os = "macos", test))]
pub(crate) async fn choose_mlx_request_tier<
    T,
    F: std::future::Future<Output = WorkerResult<MlxTierFit<T>>>,
>(
    directories: Vec<PathBuf>,
    mut probe: impl FnMut(PathBuf) -> F,
) -> WorkerResult<T> {
    let mut last_rejection = None;
    for dir in directories {
        let tier = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        match probe(dir).await? {
            MlxTierFit::Fits(prepared) => return Ok(prepared),
            MlxTierFit::TooBig(error) => {
                last_rejection = Some(WorkerError::InvalidPayload(format!(
                    "{} tier: {error}",
                    tier.as_deref().unwrap_or("resolved"),
                )));
            }
        }
    }
    Err(last_rejection.unwrap_or_else(|| {
        WorkerError::InvalidPayload("no installed MLX tier could be resolved".to_owned())
    }))
}

/// Reconcile a probe silently; only the winning tier emits the generation recipe/tier event.
#[cfg(target_os = "macos")]
fn mlx_candidate_quant(
    request: &ImageRequest,
    model: &ResolvedModel,
    dir: &Path,
) -> (Option<Quant>, Option<i64>) {
    if let Some(fixed) = fixed_mlx_artifact_quant(&request.model) {
        return fixed;
    }
    if !model.supports_quant() {
        return (None, None);
    }
    let requested = resolve_quant(request, Some(dir));
    match tier_quant_from_resolved_dir(&request.model_manifest_entry, dir) {
        Some((actual, bits)) => (
            if is_dense_te_tier(request) {
                requested.0
            } else {
                actual
            },
            bits,
        ),
        None => requested,
    }
}

/// Resolve the receipt of the candidate the ladder is about to score. Lower tiers remain untouched
/// if an earlier candidate fits, including when their receipts cannot be verified offline.
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_mlx_candidate_provenance(
    request: &ImageRequest,
    settings: &Settings,
    repo: &str,
    directory: &Path,
) -> WorkerResult<()> {
    if model_path_override(request).is_some() {
        return Ok(());
    }
    let variant = tier_key_from_resolved_dir(directory)
        .map(str::to_owned)
        .or_else(|| requested_receipt_variant(request));
    crate::model_jobs::receipt_verification::ensure_huggingface_receipt_provenance(
        settings,
        repo,
        &request.model,
        variant.as_deref(),
        directory,
    )
    .await
}

/// Real MLX generation: load once on a blocking thread, generate each image, and
/// stream step/decode/image events back to the async worker (which saves PNGs, emits
/// `assetWrites`, and polls cancel). MLX runs entirely on the blocking thread (the
/// `Box<dyn Generator>` is `!Send` and the MLX device is single-thread).
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
pub(super) async fn generate_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    generate_stream_using(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        asset_writes,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_stream`] with the generator loader supplied by the caller (the `_using` seam, see
/// [`start_cached_gen_stream_with_request_state_using`]): every pre-load decision, the cached
/// load, the per-item drive, and `consume_gen_events` are the production code; only the engine
/// is injectable, so a test can pin progress / cancel / failure on a real job arm without weights.
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
pub(super) async fn generate_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    if sceneworks_core::moss_image::is_moss(&plan.request.model) {
        return generate_moss_stream_using(
            api,
            settings,
            job,
            plan,
            project_path,
            asset_writes,
            load_generator,
        )
        .await;
    }
    let flux3_references = flux3_references::resolve_owned_references(
        &plan.request,
        &settings.data_dir,
        project_path,
    )?;
    let flux3_reference_count = flux3_references
        .as_ref()
        .map_or(0, |references| references.conditioning.len());
    let mut resolved_request = plan.request.clone();
    if let Some(references) = &flux3_references {
        (resolved_request.width, resolved_request.height) = references.plan.output.canvas;
        // Only numeric admission geometry is normalized. Preserve the authored UI mode and the
        // original optional-size intent; the memory route alone is derived from the references.
    }
    let flux3_layout =
        flux3_references::resolve_owned_layout(&resolved_request, flux3_references.as_ref())?;
    let request = &resolved_request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    // Hoisted (also used at the load-spec seam below): the registered engine id, the sc-10733
    // capability-downtier fit key + `apply_residency_policy` sequential-capability key.
    let engine_id = model.engine_id();
    // sc-6568: a bf16 opt-in for Boogu fetches the full-precision `<variant>-bf16/` subfolder on
    // demand (the catalog ships only the Q8 default) before snapshot resolution. No-op for every
    // other model / the default Q8 path. sc-9607: the same on-demand pattern for Ideogram's `q8/`
    // tier (the catalog ships only q4) — was a documented follow-up, now wired on both lanes.
    ensure_boogu_tier_present(api, settings, job, request).await?;
    ensure_ideogram_tier_present(api, settings, job, request).await?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    // sc-26241: distilled FLUX 3 loads the package the request names — the dense bf16 snapshot
    // root or a locally converted MLX affine q4/q8 — never the generic resolver's q8 default and
    // never a request-time quantization of the dense package. `None` for every other model.
    let flux3_plan = flux3_precision::mlx_flux3_plan(
        request,
        settings,
        &weights_dir,
        model.descriptor.capabilities.supported_quants,
    )?;
    let weights_dir = flux3_plan
        .as_ref()
        .map_or(weights_dir, |plan| plan.weights_dir.clone());
    let adapters = resolve_adapters(request, settings)?;
    // Each adapter's identity (request id, path, sha256, scale, kind) for the image provenance,
    // joined after the render with the engine's own apply report (sc-25681).
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &adapters).await;
    // sc-3723: surface the descriptor-derived backend ("mlx" for every linked family today; a
    // future candle row would self-describe) over the gpu-id-derived label. Falls back to the
    // passed-in label only if a descriptor ever advertised an empty backend (never today).
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
    };
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    // RealVisXL Lightning (sc-6075): a standalone few-step *distilled checkpoint* (the
    // SDXL-Lightning distillation is baked into the weights — no acceleration LoRA). It must run
    // on the engine's `lightning` (Euler-trailing) few-step schedule, not the 30-step
    // `euler_ancestral` default, so the schedule matches the checkpoint regardless of the UI
    // payload — mirrors the qwen `*_lightning` sampler forcing. The engine then applies the
    // CFG-off, few-step recipe (steps/guidance come from the manifest defaults via the model row).
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
    // N3 (epic 7114): drop a sampler/scheduler the linked engine descriptor doesn't advertise back to
    // the engine default + emit an event, instead of letting `validate_request` hard-fail the whole
    // generation over a sampling knob (a stale recipe, manifest drift, or a per-backend gap). The forced
    // `realvisxl_lightning` sampler above is always in that family's advertised set, so it passes through.
    let caps = &model.descriptor.capabilities;
    let sampler = normalize_sampling_knob(
        sampler,
        &caps.samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &caps.schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    // Guidance method (epic 7434 P5, sc-7448): the 4th sampling axis. N3-guarded against the engine's
    // `supported_guidance_methods` exactly like sampler/scheduler — an unadvertised method (a stale
    // recipe, a per-backend gap, or a method gated to an incompatible sampler) drops to the engine
    // default + a `sampling_knob_unsupported` event, never a `validate_request` hard-fail.
    let guidance_method = normalize_sampling_knob(
        read_advanced_guidance_method(&request.advanced),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    // True-CFG families (Chroma) carry the CFG scale in `true_cfg`, not `guidance` (which their
    // engine rejects); `None` for every other family. The recipe records the effective CFG knob.
    let model_true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    // The request controls beyond the positional knobs (sc-25681): CFG interval and caption-overflow
    // policy against the linked descriptor, plus the CFG-bound negative prompt
    // (`image.negativePromptRequiresGuidance`, Iris-3B) — each refused before load when the engine
    // does not honour it or CFG-off would make it inert, never dropped.
    let lane_controls = lane_controls::resolve_lane_request_controls(
        request,
        &model.descriptor.capabilities,
        guidance,
        negative_prompt.as_deref(),
    )?;
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    // Reference conditioning for the base MLX path, resolved once (constant across the set):
    //  • Z-Image reference-identity img2img-init (sc-3619),
    //  • FLUX.1 XLabs IP-Adapter (epic 3621 — both schnell + dev; `strength = ipAdapterScale`, plus
    //    real CFG via `trueCfgScale` on dev), and
    //  • Kolors img2img (sc-4765, `edit_image` + `sourceAssetId`) + the IP-Adapter-Plus reference
    //    (sc-4767, `referenceAssetId` → image prompt at `ipAdapterScale`). Qwen/SDXL reference
    //    divert to their own advanced branches before reaching here.
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // The per-family slots + the registry editors' ordered `edit_refs`, resolved once — see
    // `resolve_generic_lane_inputs`, which is also what the sc-24110 live-path tests drive.
    let (
        LaneConditioning {
            identity_init,
            flux_ip_dir,
            flux_true_cfg,
            ideogram_edit_mask,
        },
        edit_refs,
        edit_ref_alpha,
    ) = resolve_generic_lane_inputs(request, settings, project_path, has_reference)?;
    // The S4 four-channel surface (sc-24113): the transparency toggle resolved against the engine's
    // own `supports_alpha_output` (refused by name when it cannot serve it) and assigned onto the
    // request, plus each ordered reference's alpha plane (`edit_ref_alpha`). Default — RGB, no
    // alpha — for every model but Qwen-Image 2.1.
    let rgba_output_channels = crate::qwen_alpha::request_output_channels(
        crate::qwen_alpha::resolve_output_channels(&request.model, &request.advanced)?,
    );
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);
    let hires_fix = resolve_hires_fix_plan(request, steps, guidance, true_cfg);

    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is
    // out-of-distribution and stochastically renders the "Image blocked by safety filter"
    // placeholder (sc-6307, reference-confirmed faithful). The web Image Studio auto-expands plain
    // prompts into rich captions; this is the worker-side HARD GUARANTEE that raw plain text never
    // tokenizes — it wraps a non-caption prompt into a minimal valid caption (covers the API path
    // and any UI bypass). A prompt that is already a caption passes through unchanged. No-op for
    // every other family.
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    // sc-6135: caption upsampling (FLUX.2-dev only; every other engine ignores it). Resolved from
    // the request's advanced `enhancePrompt` toggle, gated to dev by the manifest `ui.promptEnhance`.
    let enhance = PromptEnhance::from_advanced(&request.model, &request.advanced)?;
    // Per-generation PiD decode (epic 7840, sc-7849): resolve the PiD checkpoint + Gemma for this
    // model's latent space when `advanced.usePid` is set and the snapshots are cached; otherwise keep
    // the native VAE. `use_pid` and `spec.pid` stay in lockstep (the engine rejects a mismatch).
    let pid_weights = resolve_pid_weights(request, &settings.data_dir, &request.model)?;
    let use_pid = pid_weights.is_some();
    // A native prompt batch (sc-25681): ONE engine call over `prompts × count` images, refused
    // before load when the linked engine cannot render it or a per-image lane feature is in use.
    let native_batch = lane_controls::resolve_native_prompt_batch(
        request,
        &model.descriptor.capabilities,
        &[
            ("hires fix", hires_fix.is_some()),
            (
                "a reference image",
                flux3_reference_count > 0
                    || identity_init.is_some()
                    || !edit_refs.is_empty()
                    || ideogram_edit_mask.is_some(),
            ),
            ("prompt enhancement", enhance.enabled),
        ],
    )?;
    let native_batch_images = native_batch.as_ref().map_or(1, |batch| batch.images());
    // The images this job streams: `count` per-seed renders, or every image of the one batch call.
    let stream_total = if native_batch.is_some() {
        native_batch_images as usize
    } else {
        count
    };
    // PiD output tier (sc-10054): PiD super-resolves the base latent by a fixed 4×, so the effective
    // base picks whether the output lands on ~2K or ~4K. `4k`/native leave the requested dims untouched;
    // `2k` caps the base (also lowering the F-013 decode peak). Rebind before `generate_one`.
    let (width, height) = pid_effective_dims(width, height, use_pid, pid_output_tier(request));
    // Admission describes the heaviest pass. Hires fix renders the first pass at `width`/`height`
    // and the refinement at the plan dimensions, so every memory fit and request scope must use the
    // latter. `generate_one_with_hires` derives a base-pass scope from this final-pass identity.
    let (memory_width, memory_height) =
        hires_fix.map_or((width, height), |plan| (plan.width, plan.height));
    // Krea "text style" tap-reweight gain — see `resolve_text_style_gain` (sc-11878, gate fixed sc-12008).
    let text_style_gain = resolve_text_style_gain(request);
    let calibration_opt_in = request
        .model_manifest_entry
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
        .is_some_and(|calibrations| !calibrations.is_empty());
    let quality_opt_in = crate::mlx_fit_gate::manifest_declares_decode_quality_policies(
        &request.model_manifest_entry,
    );
    let memory_budget = crate::generator_cache::mlx_tier_budget(engine_id).await?;
    // Resolve every tier through the exact production composition BEFORE scoring it. No candidate
    // loads tensors, and the winning spec/plan are retained rather than rebuilt after selection.
    let prepare = |weights_dir: PathBuf,
                   prefer_resident: bool|
     -> WorkerResult<PreparedMlxImageTier> {
        let (quant, quant_bits) = match flux3_plan.as_ref() {
            Some(plan) => plan.quant_and_bits(),
            None => mlx_candidate_quant(request, &model, &weights_dir),
        };
        // A split-repo snapshot root (sc-24112's `qwen_image_2_1` bf16) names its tier only
        // through the catalog, so ask for it first; `None` for every other family. A FLUX 3
        // package names its tier by identity (sc-26241).
        let effective_tier = flux3_plan
            .as_ref()
            .map(|plan| plan.tier.key())
            .or_else(|| split_repo_root_tier_key(&request.model_manifest_entry, &weights_dir))
            .or_else(|| {
                resolved_mlx_artifact_tier_for_model(&request.model, &weights_dir, quant_bits)
            });
        let resolved_artifact = if calibration_opt_in || quality_opt_in {
            resolved_mlx_artifact_provenance(
                request,
                settings,
                &repo,
                &weights_dir,
                effective_tier,
            )?
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        let load_quant = mlx_load_quant_for_resolved_artifact(engine_id, quant);
        #[cfg(not(target_os = "macos"))]
        let load_quant = quant;
        let mut spec = load_spec(
            weights_dir.clone(),
            load_quant,
            adapters.clone(),
            flux_ip_dir.clone(),
        );
        if let Some(pid) = &pid_weights {
            spec = spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
        }
        // Named model components (epic 13657, sc-13682): stage a provider's caller-staged components (SDXL's
        // `tokenizer_clip_l` / `tokenizer_clip_bigg` / `vae_fp16_fix`) via the generic seam. Keyed on the
        // resolved `engine_id` (the DESCRIPTOR id) rather than `request.model`, so a finetune sibling that
        // shares one engine under a distinct catalog id resolves the same descriptor (media_descriptor matches
        // on descriptor.id). The MLX SDXL turnkey is self-contained (no `required_components`); Iris-3B
        // stages its Qwen3-VL `text_encoder` here (sc-25679).
        spec =
            attach_required_components(spec, engine_id, &request.model_manifest_entry, settings)?;
        spec = attach_prompt_upsampler(spec, engine_id, request, settings)?;
        if let Some(plan) = flux3_plan.as_ref() {
            spec = plan.attach_components(spec);
        }
        // F3 alternate decoder: attach before both the provider-specific memory contract and the generic
        // MLX fit gate, so donor bytes + normal activation/OS margin are admitted as one composition.
        spec = attach_selected_decoder(spec, engine_id, request, settings)?;
        // P9: a shared engine such as `sdxl` serves several independently pinned catalog routes. Bind
        // the exact resolved model id, independently resolved artifact tree, and running inference
        // implementation before any semantic quality row reaches the provider contract.
        spec = spec.with_resolved_route(request.model.clone());
        let plain_text_to_image =
            matches!(request.mode.as_str(), "image_generation" | "text_to_image")
                && flux3_references.is_none()
                && identity_init.is_none()
                && edit_refs.is_empty()
                && ideogram_edit_mask.is_none()
                && hires_fix.is_none();
        // Finalize caller-selected text-encoder state before asking the provider about the real
        // candidate. Chroma must see and reject an external encoder rather than being admitted against
        // an incomplete LoadSpec which is mutated afterward.
        let unattached_spec = spec;
        let attached_spec =
            attach_manifest_text_encoder(unattached_spec, engine_id, request, settings)?;
        let mut spec = attached_spec.into_load_spec();
        // SC-18457: provider adoption is an exact three-way intersection. A route-local BTR entry owns
        // the decision: the typed registry enforces mode/overlay/source semantics and the linked
        // provider must return BTR Implemented for this real deferred candidate. A refusal never falls
        // through to legacy shaping; only a manifest with no relevant BTR entry uses that unchanged
        // path. The tier is the resolved artifact tier, not the load-time quant field on the spec.
        let declaration_reference_count = if hires_fix.is_some() {
            hires_fix_reference_count()
        } else {
            lane_reference_count(
                identity_init.is_some(),
                edit_refs.len().max(flux3_reference_count),
                ideogram_edit_mask.is_some(),
            )
        };
        let reference_memory_mode = if flux3_references.is_some() {
            "edit_image"
        } else {
            &request.mode
        };
        let declaration_mode = crate::memory_route_registry::MemoryRouteMode::from_mlx_request(
            engine_id,
            reference_memory_mode,
        );
        let declaration_context = crate::memory_route_registry::MemoryRouteRequestContext {
            mode: declaration_mode
                .unwrap_or(crate::memory_route_registry::MemoryRouteMode::TextToImage),
            reference_count: declaration_reference_count,
            use_pid,
            has_phases: false,
        };
        let pre_policy_spec = spec.clone();
        spec = prepare_mlx_load_policy(
            engine_id,
            effective_tier,
            declaration_mode,
            &request.model_manifest_entry,
            spec,
            declaration_context,
            plain_text_to_image,
            model.descriptor.capabilities.supports_sequential_offload,
        )?;
        let resident_alternative =
            declared_sequential_resident_alternative(engine_id, &pre_policy_spec, &spec);
        let has_resident_alternative = resident_alternative.is_some();
        if let Some(resident) = resident_alternative.filter(|_| prefer_resident) {
            spec = resident;
        }
        let decode_quality_binding =
            crate::mlx_fit_gate::bind_decode_quality_policies_from_manifest(
                &request.model_manifest_entry,
                &request.model,
                resolved_artifact.as_ref(),
            )?;
        spec = crate::mlx_fit_gate::attach_decode_quality_binding(
            spec,
            decode_quality_binding,
            &request.model,
        );
        let mlx_request_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
            engine_id,
            &request.model,
            &spec,
            Some(&request.model_manifest_entry),
            resolved_artifact,
        );
        let mlx_request_plan = if matches!(
            engine_id,
            "krea_2_raw"
                | "krea_2_turbo"
                | "krea_2_edit"
                | "krea_2_turbo_edit"
                | "flux1_schnell"
                | "flux1_dev"
                | "flux2_klein_9b"
        ) {
            mlx_request_plan.with_resolved_artifact_tier(effective_tier)?
        } else {
            mlx_request_plan
        };
        let has_request_reference = flux3_reference_count > 0
            || identity_init.is_some()
            || !edit_refs.is_empty()
            || ideogram_edit_mask.is_some();
        // The admitted geometry describes the HEAVIEST pass: with hires fix that is the final
        // upscaled img2img refinement (one `Reference`, no mask), otherwise the single base pass. The
        // first hires pass renders at the base size with the caller's own conditioning and gets its own
        // request-scope identity inside `generate_one_with_hires`.
        let reference_count = declaration_reference_count;
        let mut memory_overlays = Vec::new();
        if has_request_reference {
            memory_overlays.push(format!(
                "references:{}",
                edit_refs.len().max(flux3_reference_count).max(1)
            ));
        }
        if ideogram_edit_mask.is_some() {
            memory_overlays.push("mask".to_owned());
        }
        if spec.control.is_some() || !spec.extra_controls.is_empty() {
            memory_overlays.push(format!(
                "control:{}",
                usize::from(spec.control.is_some()) + spec.extra_controls.len()
            ));
        }
        if spec.ip_adapter.is_some() {
            memory_overlays.push("ip_adapter".to_owned());
        }
        if adapter_count > 0 {
            memory_overlays.push(format!("adapters:{adapter_count}"));
        }
        if use_pid {
            memory_overlays.push("pid".to_owned());
        }
        let provider_overlay = crate::mlx_fit_gate::provider_overlay_for_load_spec(
            engine_id,
            &spec,
            (!memory_overlays.is_empty()).then(|| memory_overlays.join("+")),
        );
        let mlx_request_inputs = crate::mlx_fit_gate::MlxRequestInputs {
            native_batch: native_batch_images,
            width: memory_width,
            height: memory_height,
            count: request.count,
            mode: reference_memory_mode.to_owned(),
            overlay: provider_overlay,
            adapter_count,
            has_reference: reference_count > 0,
            reference_count,
            use_pid,
            has_phases: false,
            conditioning_windows: Some(crate::mlx_fit_gate::clip_windows_for_spec(
                engine_id,
                &spec,
                &request.prompt,
                &request.negative_prompt,
            )),
            // The prompt branches the engine will evaluate (epic 25899): a prefix-KV-cache provider
            // holds one cache per branch, so its surcharge doubles under true CFG.
            negative_branch: crate::execution_planner::evaluates_negative_branch(
                negative_prompt.as_deref().unwrap_or_default(),
                true_cfg,
                guidance,
            ),
            // A hires refinement hands this block to a first pass at the base size with the caller's
            // own references and re-resolved CFG, and a native prompt batch renders one prefix per
            // prompt in one pass: the cache is priced for neither render.
            prefix_kv_cache_eligible: hires_fix.is_none() && native_batch.is_none(),
            reference_resources: flux3_references
                .as_ref()
                .map(|references| references.resources.clone()),
        };
        Ok(PreparedMlxImageTier {
            weights_dir,
            spec,
            plan: mlx_request_plan,
            inputs: mlx_request_inputs,
            has_resident_alternative,
        })
    };
    let default_tier = tier_key_for_resolved_dir(&request.model_manifest_entry, &weights_dir);
    let explicit_pick = flux3_plan.is_some()
        || request
            .advanced
            .get("mlxQuantizeExplicit")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let tiers = default_tier
        .filter(|_| !explicit_pick)
        .map(|tier| downtier_candidate_tiers(request, settings, tier, min_quality_floor(request)))
        .unwrap_or_default();
    let directories = if tiers.is_empty() {
        vec![weights_dir.clone()]
    } else {
        tiers
            .iter()
            .filter_map(|tier| resolve_tier_dir(request, settings, tier))
            .collect()
    };
    let selected = choose_mlx_request_tier(directories, |dir| {
        let prepare = &prepare;
        let repo = &repo;
        async move {
        if calibration_opt_in || quality_opt_in {
            ensure_mlx_candidate_provenance(request, settings, repo, &dir).await?;
        }
        let mut prepared = prepare(dir.clone(), false)?;
        // Query/validate the provider before considering a legacy load refusal, so malformed
        // artifacts are never mistaken for a reason to silently select another tier.
        let mut admission = crate::mlx_fit_gate::preflight_request(
            &prepared.spec,
            &prepared.plan,
            &prepared.inputs,
            memory_budget,
        )?;
        // sc-24806: a manifest BTR/staging row binds Sequential even when this request needs no
        // staging or block streaming. Re-prepare the same tier eager + Resident and keep it only if
        // it is admitted without staging and passes the load gate; otherwise keep the declared spec.
        if prepared.has_resident_alternative && admission_needs_no_staging(&admission) {
            // A failed Resident probe keeps the admitted declared spec, as the shared helper does.
            let released = prepare(dir, true).ok().and_then(|resident| {
                let resident_admission = crate::mlx_fit_gate::preflight_request(
                    &resident.spec,
                    &resident.plan,
                    &resident.inputs,
                    memory_budget,
                )
                .ok()?;
                (admission_needs_no_staging(&resident_admission)
                    && crate::mlx_fit_gate::preflight_load_rejection(engine_id, &resident.spec)
                        .is_none())
                .then_some((resident, resident_admission))
            });
            if let Some((resident, resident_admission)) = released {
                tracing::info!(
                    event = "mlx_declared_sequential_released_to_resident",
                    engine = %engine_id,
                    model = %request.model,
                    "request needs no staging; loading Resident instead of the declared Sequential policy"
                );
                prepared = resident;
                admission = resident_admission;
            }
        }
        let admission = match admission {
            crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation) => {
                match crate::mlx_fit_gate::preflight_load_rejection(engine_id, &prepared.spec) {
                    Some(error) => crate::mlx_fit_gate::MlxRequestAdmission::Rejected(error),
                    None => crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation),
                }
            }
            rejected => rejected,
        };
        Ok(match admission {
            crate::mlx_fit_gate::MlxRequestAdmission::Admitted(_) => MlxTierFit::Fits(prepared),
            crate::mlx_fit_gate::MlxRequestAdmission::Rejected(error) => MlxTierFit::TooBig(error),
        })
        }
    }).await?;
    if selected.weights_dir != weights_dir {
        tracing::warn!(model = %request.model, from = ?default_tier,
            to = ?tier_key_for_resolved_dir(&request.model_manifest_entry, &selected.weights_dir),
            "MLX request ladder selected a lower installed tier");
    }
    let PreparedMlxImageTier {
        weights_dir,
        spec,
        plan: mlx_request_plan,
        inputs: mlx_request_inputs,
        has_resident_alternative: _,
    } = selected;
    // Descriptor-gated quant (mirrors the candle lane below): the MLX families advertise Q4/Q8
    // (`supported_quants`) and tolerate the Q8 default (a real quant on a dense convert, a no-op on an
    // already-packed turnkey). SANA joined this set in mlx-gen #654 (sc-8489): its descriptor now
    // advertises Q4/Q8 and its `load` ACCEPTS an advisory `spec.quantize` (the pre-quantized tier is
    // packed-detected from disk, #653), so it flows through the normal resolve_quant path like every
    // other matrix model. The `else` arm stays for any future engine that genuinely advertises no
    // quant — such a model loads dense.
    // True-V2's converter consumes the sole BF16 source and writes a dense BF16 transformer. It
    // has no packed tier matrix: fixing the pair before reconciliation means neither the historical
    // catalog default nor a crafted advanced preference can emit a false tier-change event or
    // relabel the declaration, fit, recipe, and provider identities.
    let fixed_artifact_quant = flux3_plan
        .as_ref()
        .map(|plan| plan.quant_and_bits())
        .or_else(|| fixed_mlx_artifact_quant(&request.model));
    let (quant, quant_bits) = if let Some(fixed) = fixed_artifact_quant {
        fixed
    } else if model.supports_quant() {
        // `weights_dir` is the resolved tier subdir (sc-11042). NVFP4 is unreachable on this lane
        // regardless (`nvfp4_host_eligible()` is hard-`false` on macOS — Metal has no FP4 hardware), so
        // this is the same `(quant, bits)` it has always produced; passing the dir keeps the resolver's
        // one contract — the tier is read off what resolved — uniform across both lanes.
        resolve_quant(request, Some(&weights_dir))
    } else {
        (None, None)
    };
    // sc-8820: the tier resolvers ([`standard_tier_subdir`] & friends) silently fall through
    // q4→q8→bf16 when the preferred tier isn't downloaded, but the quant above is derived from the
    // REQUEST — so a bf16 pick with only `q4/` present would render Q4 while the recipe records dense,
    // lying to the epic 8506 quant A/B workflow. Reconcile against the tier subdir actually resolved:
    // record the precision that ran + `warn!`/emit `quant_tier_downgraded` on a real fallback. SANA
    // (sc-8489) now ships standard q4/q8/bf16 turnkey tiers and advertises Q4/Q8, so it reconciles here
    // exactly like the other matrix models.
    //
    // sc-9362 (F-018 follow-up): dense-TE turnkeys (FLUX.2-klein) always derive `(None, None)` from
    // `resolve_quant` (the load quant must stay `None` so the dense bf16 TE is never re-quantized),
    // but their transformer is packed at q4/q8. Reconciling against that always-bf16 value made every
    // straight dense-TE job read as a bf16→qN "downgrade" — a spurious event, and pre-8820 the recipe
    // recorded bf16 for a q4/q8 transformer. Feed reconcile the transformer tier the request ACTUALLY
    // asked for ([`dense_te_requested_tier_bits`], mirroring the `standard_tier_subdir` mapping) so it
    // records the resolved transformer precision on EVERY job and only warns/emits on a genuine
    // fallback. `allow_quant_change=false` keeps the load quant `None` (TE stays dense bf16).
    let (_, quant_bits) = if fixed_artifact_quant.is_some() {
        (quant, quant_bits)
    } else if model.supports_quant() {
        let requested_for_reconcile = if is_dense_te_tier(request) {
            (None, dense_te_requested_tier_bits(request))
        } else {
            (quant, quant_bits)
        };
        reconcile_resolved_tier_quant(
            requested_for_reconcile,
            &weights_dir,
            &request.model_manifest_entry,
            !is_dense_te_tier(request),
            &request.model,
            &job.id,
            backend,
        )
    } else {
        (quant, quant_bits)
    };
    let raw_settings = mlx_raw_settings_for_bundle(
        request,
        Some(&weights_dir),
        &repo,
        steps,
        quant_bits,
        guidance.or(model_true_cfg),
    );

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the generic MLX lane serves
    // the remaining With-Character identity generators — Z-Image identity-init (`referenceAssetId` ⇒
    // img2img init), the FLUX.1 XLabs IP-Adapter, and the Kolors IP-Adapter-Plus reference — all of
    // which carry a character `referenceAssetId`. Score every output against that reference face through
    // the SHARED generator-agnostic seam, but ONLY for an Image Studio "With Character"
    // (`character_image`) generation; a z-image / kolors `edit_image` job (its source is `sourceAssetId`,
    // not an identity reference) is excluded by `resolve_character_image_likeness_source` (mode gate),
    // which also resolves the CURRENT job's reference (so changing it changes the scored source) and is
    // non-fatal. The `!Send` scorer is built ONCE inside the closure and reused across the N outputs.
    let likeness_source = resolve_character_image_likeness_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    // Keep the source only if the face stack staged (otherwise no scorer can be built).
    let likeness_source = face_stack_dir.as_ref().and(likeness_source);

    // Keep the exact selected spec across the cache callback. Cache warm policy may change
    // residency, not the artifact/precision identity; reference demand is repriced on every fit.
    let reference_pricing_spec = flux3_references.as_ref().map(|_| spec.clone());
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        load_generator,
        move |generator,
              cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every output (sc-4411). `None`
            // ⇒ not a With-Character generation, or non-fatal staging/construction failure ⇒ omitted.
            let scorer_requested = face_stack_dir.is_some() && likeness_source.is_some();
            let scorer_active_before = if scorer_requested {
                mlx_rs::memory::clear_cache();
                mlx_rs::memory::get_active_memory() as u64
            } else {
                0
            };
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let request_external_committed_bytes = if scorer_requested {
                // The MLX face scorer is constructed after the generator cache recorded its
                // pre-load baseline. Add only its active-memory delta so scorer weights cannot be
                // credited as generation weights without double-charging the cached generator.
                // Measure attempted construction too: a failed scorer may still have allocated.
                mlx_rs::memory::clear_cache();
                let scorer_active_after = mlx_rs::memory::get_active_memory() as u64;
                crate::mlx_fit_gate::add_post_load_external_delta(
                    external_committed_bytes,
                    scorer_active_before,
                    scorer_active_after,
                )
            } else {
                external_committed_bytes
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            // sc-18317: ONE warm hit is ONE decision, but this lane evaluates the request once per
            // item. Hand the real proposal to the first evaluation and an inert one to the rest, so a
            // multi-image job settles exactly one decision and emits exactly one event.
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            // A native prompt batch (sc-25681) is ONE engine call over every image of the job,
            // admitted against its `prompts × count` batch (`native_batch` above).
            if let Some(batch) = native_batch.as_ref() {
                return drive_native_prompt_batch(
                    tx,
                    seeds[0],
                    batch.images() as usize,
                    |preview, on_progress| {
                        let memory_evaluation =
                            if let Some(reference_spec) = reference_pricing_spec.as_ref() {
                                crate::mlx_fit_gate::evaluate_request_with_reference_spec(
                                    generator,
                                    &mlx_request_plan,
                                    &mlx_request_inputs,
                                    cache_state,
                                    loaded_policy.offload_policy,
                                    warm_policy.take(),
                                    request_external_committed_bytes,
                                    reference_spec,
                                )
                            } else {
                                crate::mlx_fit_gate::evaluate_request(
                                    generator,
                                    &mlx_request_plan,
                                    &mlx_request_inputs,
                                    cache_state,
                                    loaded_policy.offload_policy,
                                    warm_policy.take(),
                                    request_external_committed_bytes,
                                )
                            }?;
                        let _request_memory_limit = memory_evaluation
                            .process_limit_bytes
                            .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                        let warnings = std::cell::RefCell::new(Vec::new());
                        let prefix_kv_cache_report = std::cell::RefCell::new(None);
                        let images = render_on_surface(
                            generator,
                            "",
                            width,
                            height,
                            seeds[0],
                            steps,
                            guidance,
                            negative_prompt.clone(),
                            None,
                            &[],
                            None,
                            true_cfg,
                            sampler.as_deref(),
                            scheduler.as_deref(),
                            scheduler_shift,
                            guidance_method.as_deref(),
                            use_pid,
                            text_style_gain,
                            Some(memory_evaluation.memory),
                            Some(&memory_evaluation.context),
                            &enhance,
                            gen_core::PromptEnhancementSink::default(),
                            preview,
                            &LaneRgbaSurface {
                                multi_reference_alpha: &[],
                                flux3_references: None,
                                flux3_layout: None,
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: Some(batch),
                            },
                            &cancel,
                            on_progress,
                        )?;
                        let facts = lane_controls::image_facts(
                            &warnings.take(),
                            &adapter_identities,
                            &generator.adapter_apply_reports(),
                            crate::execution_planner::prefix_kv_cache_record(
                                memory_evaluation.prefix_kv_cache,
                                prefix_kv_cache_report.take().as_ref(),
                            ),
                        );
                        Ok((images, facts))
                    },
                );
            }
            drive_gen_items_scored_reported(
                tx,
                seeds,
                move |_index, seed, preview, prompt_enhancement, on_progress| {
                    let memory_evaluation =
                        if let Some(reference_spec) = reference_pricing_spec.as_ref() {
                            crate::mlx_fit_gate::evaluate_request_with_reference_spec(
                                generator,
                                &mlx_request_plan,
                                &mlx_request_inputs,
                                cache_state,
                                loaded_policy.offload_policy,
                                warm_policy.take(),
                                request_external_committed_bytes,
                                reference_spec,
                            )
                        } else {
                            crate::mlx_fit_gate::evaluate_request(
                                generator,
                                &mlx_request_plan,
                                &mlx_request_inputs,
                                cache_state,
                                loaded_policy.offload_policy,
                                warm_policy.take(),
                                request_external_committed_bytes,
                            )
                        }?;
                    // Exact promoted MLX evidence may tighten the soft process limit for this request.
                    // The RAII guard restores the process-global/user limit after all retries and never
                    // touches the wired limit (#1947).
                    let _request_memory_limit = memory_evaluation
                        .process_limit_bytes
                        .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                    // This image's non-fatal engine warnings (sc-25681), reported with its asset.
                    let warnings = std::cell::RefCell::new(Vec::new());
                    // ...and the prefix-KV-cache path its render reports (epic 25899).
                    let prefix_kv_cache_report = std::cell::RefCell::new(None);
                    let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                        generate_one_with_hires_on_surface(
                            generator,
                            &prompt,
                            width,
                            height,
                            seed,
                            steps,
                            guidance,
                            negative_prompt.clone(),
                            identity_init.as_ref(),
                            &edit_refs,
                            ideogram_edit_mask.as_ref(),
                            true_cfg,
                            sampler.as_deref(),
                            scheduler.as_deref(),
                            scheduler_shift,
                            guidance_method.as_deref(),
                            use_pid,
                            text_style_gain,
                            Some(memory_evaluation.memory),
                            Some(memory_evaluation.memory),
                            Some(&memory_evaluation.context),
                            None,
                            &enhance,
                            hires_fix,
                            preview.clone(),
                            prompt_enhancement.for_prompt(&prompt),
                            &LaneRgbaSurface {
                                multi_reference_alpha: &edit_ref_alpha,
                                flux3_references: flux3_references.as_ref(),
                                flux3_layout: flux3_layout.as_ref(),
                                output_channels: rgba_output_channels,
                                controls: lane_controls,
                                warnings: Some(&warnings),
                                prefix_kv_cache: Some(&prefix_kv_cache_report),
                                prompt_batch: None,
                            },
                            &cancel,
                            on_progress,
                        )
                    };
                    // Detect-and-recover safety net (sc-6501): the caption guard makes the placeholder
                    // rare, but a residual one can still occur even with a caption. Detect it via the
                    // baked-text heuristic (NOT a std/flatness check — the text lifts std to ~10) and
                    // reseed transparently, keeping the first clean render. Gated to Ideogram 4; a no-op
                    // elsewhere (and on turbo, which is CFG-free and cannot produce the placeholder).
                    let initial = render(seed, on_progress)?;
                    let (final_seed, out_w, out_h, pixels) = recover_ideogram_placeholder(
                        is_ideogram,
                        seed,
                        &cancel,
                        initial,
                        |retry_seed| render(retry_seed, on_progress),
                    )?;
                    // Score this finished image against the cached source embedding (sc-4411). Image build +
                    // pixel clone is paid ONLY when a scorer exists (a With-Character generation) — a plain
                    // t2i / edit job has no scorer, so this is a no-op with no clone. Non-frontal → honest
                    // detected:false N/A; `None` scorer ⇒ field omitted.
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_generated_image(
                            Some(scorer),
                            &Image {
                                width: out_w,
                                height: out_h,
                                pixels: pixels.clone(),
                            },
                            likeness_source_ref.as_deref(),
                        )
                    });
                    // The image's provenance facts (sc-25681): engine warnings, and each adapter's
                    // identity joined with the engine's own applied/skipped report.
                    let mut facts = lane_controls::image_facts(
                        &warnings.take(),
                        &adapter_identities,
                        &generator.adapter_apply_reports(),
                        crate::execution_planner::prefix_kv_cache_record(
                            memory_evaluation.prefix_kv_cache,
                            prefix_kv_cache_report.take().as_ref(),
                        ),
                    )
                    .unwrap_or_default();
                    if let Some(references) = flux3_references.as_ref() {
                        facts.insert(
                            flux3_references::FACT_KEY.to_owned(),
                            references.image_fact((out_w, out_h)),
                        );
                    }
                    if let Some(layout) = flux3_layout.as_ref() {
                        facts.insert("flux3Layout".to_owned(), layout.image_fact());
                    }
                    if !facts.is_empty() {
                        prompt_enhancement.send_image_facts(facts);
                    }
                    Ok(Some((final_seed, out_w, out_h, pixels, face_likeness)))
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
        adapter_label,
        &raw_settings,
        stream_total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/mlx_request_tier_tests.rs"]
mod mlx_request_tier_tests;
