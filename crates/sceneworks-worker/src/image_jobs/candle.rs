#[cfg(not(target_os = "macos"))]
use super::adapters::candle_adapter_resident_bytes;
#[cfg(not(target_os = "macos"))]
use super::adapters::krea_streamed_blocks_adapter_evidence;
#[cfg(not(target_os = "macos"))]
use super::adapters::resolve_adapters;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::ensure_boogu_tier_present;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::ensure_ideogram_tier_present;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::ensure_krea_convrot_base_present;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::is_anima_model;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::model_repo;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::resolve_candle_flux1_packed_weights_dir;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::resolve_krea_convrot;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::resolve_weights_dir;
#[cfg(not(target_os = "macos"))]
use super::attach_prompt_upsampler;
#[cfg(not(target_os = "macos"))]
use super::conditioning::candle_conditioned_edit_work;
#[cfg(not(target_os = "macos"))]
use super::conditioning::guard_qwen_image_2_1_lane_slots;
#[cfg(not(target_os = "macos"))]
use super::conditioning::is_mage_edit_model;
#[cfg(not(target_os = "macos"))]
use super::conditioning::is_qwen_image_2_1_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::is_sensenova_candle_model;
#[cfg(not(target_os = "macos"))]
use super::conditioning::load_reference_image;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_boogu_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_candle_kolors_edit_init;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_character_image_likeness_source;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_ideogram_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_img2img_init_generic;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_mage_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_qwen_image_2_1_edit_images;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_sensenova_candle_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_sensenova_candle_true_cfg;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_text_style_gain;
#[cfg(not(target_os = "macos"))]
use super::conditioning::resolve_zimage_edit_init;
#[cfg(not(target_os = "macos"))]
use super::conditioning::stage_likeness;
#[cfg(not(target_os = "macos"))]
use super::conditioning::uses_generic_img2img;
#[cfg(not(target_os = "macos"))]
use super::events::consume_gen_events_with_disclosure;
#[cfg(not(target_os = "macos"))]
use super::flux3_precision;
#[cfg(not(target_os = "macos"))]
use super::flux3_references;
#[cfg(not(target_os = "macos"))]
use super::geometry::fit_engine_image;
#[cfg(not(target_os = "macos"))]
use super::image_progress;
#[cfg(not(target_os = "macos"))]
use super::lane_controls;
#[cfg(not(target_os = "macos"))]
use super::load_policy::candle_prefix_kv_cache_surcharge;
#[cfg(not(target_os = "macos"))]
use super::load_spec::apply_declared_candle_image_load_shape;
#[cfg(not(target_os = "macos"))]
use super::load_spec::attach_manifest_text_encoder;
#[cfg(not(target_os = "macos"))]
use super::load_spec::attach_required_components;
#[cfg(not(target_os = "macos"))]
use super::load_spec::candle_certified_load_spec;
#[cfg(not(target_os = "macos"))]
use super::load_spec::candle_shared_memory_tier;
#[cfg(not(target_os = "macos"))]
use super::load_spec::load_spec;
#[cfg(not(target_os = "macos"))]
use super::load_spec::seal_ideogram_load_spec;
#[cfg(not(target_os = "macos"))]
use super::load_spec::seal_recursive_receipt_load_spec;
#[cfg(not(target_os = "macos"))]
use super::load_spec::seal_sdxl_endpoint_load_spec;
#[cfg(not(target_os = "macos"))]
use super::memory::apply_request_scoped_candle_residency;
#[cfg(not(target_os = "macos"))]
use super::memory::auto_tier_fit_decision;
#[cfg(not(target_os = "macos"))]
use super::memory::auto_tier_streaming_disclosure;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_base_memory_request_mode;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_capability_reject_message;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_quant_for_resolved_tier;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_resolved_tier_key;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_tier_fit;
#[cfg(not(target_os = "macos"))]
use super::memory::candle_tier_fit_after_shared_selection;
#[cfg(not(target_os = "macos"))]
use super::memory::capability_downtier_floor;
#[cfg(not(target_os = "macos"))]
use super::memory::choose_downtier;
#[cfg(not(target_os = "macos"))]
use super::memory::downtier_candidate_tiers;
#[cfg(not(target_os = "macos"))]
use super::memory::flux2_klein_reference_bearing_mode;
#[cfg(not(target_os = "macos"))]
use super::memory::installed_tier_keys;
#[cfg(not(target_os = "macos"))]
use super::memory::krea_evidence_revision;
#[cfg(not(target_os = "macos"))]
use super::memory::krea_runtime_evidence_context;
#[cfg(not(target_os = "macos"))]
use super::memory::krea_turbo_capability_reject_message;
#[cfg(not(target_os = "macos"))]
use super::memory::krea_turbo_memory_route;
#[cfg(not(target_os = "macos"))]
use super::memory::krea_unverified_resident_decision;
#[cfg(not(target_os = "macos"))]
use super::memory::log_krea_turbo_evidence_exclusion;
#[cfg(not(target_os = "macos"))]
use super::memory::optimized_shared_memory_context;
#[cfg(not(target_os = "macos"))]
use super::memory::rejects_unverified_shared_memory_fallback;
#[cfg(not(target_os = "macos"))]
use super::memory::shared_image_reference_count;
#[cfg(not(target_os = "macos"))]
use super::memory::validate_candle_tier_memory_evidence;
#[cfg(not(target_os = "macos"))]
use super::memory::verified_only_memory_family_label;
#[cfg(not(target_os = "macos"))]
use super::memory::vram_reject_tail;
#[cfg(not(target_os = "macos"))]
use super::memory::DowntierPick;
#[cfg(not(target_os = "macos"))]
use super::memory::ResolutionLever;
#[cfg(not(target_os = "macos"))]
use super::memory::TierFit;
#[cfg(not(target_os = "macos"))]
use super::memory::AUTO_TIER_STREAMING_DISCLOSURE_KEY;
#[cfg(not(target_os = "macos"))]
use super::min_quality_floor;
#[cfg(not(target_os = "macos"))]
use super::mlx_model;
#[cfg(all(
    native_lane,
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use super::moss::generate_moss_stream_with_runtime;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    all(test, target_os = "macos")
))]
use super::moss::MossStreamRuntime;
#[cfg(not(target_os = "macos"))]
use super::moss::NativeMossStreamRuntime;
#[cfg(not(target_os = "macos"))]
use super::pid::pid_effective_dims;
#[cfg(not(target_os = "macos"))]
use super::pid::pid_output_tier;
#[cfg(not(target_os = "macos"))]
use super::pid::resolve_pid_weights;
#[cfg(not(target_os = "macos"))]
use super::quantization::resolve_tier_dir;
#[cfg(not(target_os = "macos"))]
use super::render::generate_one_with_hires_on_surface;
#[cfg(not(target_os = "macos"))]
use super::render::lane_reference_count;
#[cfg(not(target_os = "macos"))]
use super::render::render_on_surface;
#[cfg(not(target_os = "macos"))]
use super::render::resolve_hires_fix_plan;
#[cfg(not(target_os = "macos"))]
use super::render::LaneRgbaSurface;
#[cfg(not(target_os = "macos"))]
use super::resolve_seed;
#[cfg(not(target_os = "macos"))]
use super::routing::candle_adapter_label;
#[cfg(not(target_os = "macos"))]
use super::sampling::mlx_raw_settings_for_bundle;
#[cfg(not(target_os = "macos"))]
use super::sampling::normalize_sampling_knob;
#[cfg(not(target_os = "macos"))]
use super::sampling::read_advanced_guidance_method;
#[cfg(not(target_os = "macos"))]
use super::sampling::read_advanced_sampling_knobs;
#[cfg(not(target_os = "macos"))]
use super::sampling::recover_ideogram_placeholder;
#[cfg(not(target_os = "macos"))]
use super::sampling::resolve_guidance;
#[cfg(not(target_os = "macos"))]
use super::sampling::resolve_negative_prompt;
#[cfg(not(target_os = "macos"))]
use super::sampling::resolve_steps;
#[cfg(not(target_os = "macos"))]
use super::sampling::resolve_true_cfg;
#[cfg(not(target_os = "macos"))]
use super::stream::drive_gen_items_scored_reported;
#[cfg(not(target_os = "macos"))]
use super::stream::drive_native_prompt_batch;
#[cfg(not(target_os = "macos"))]
use super::stream::start_cached_gen_stream_with_request_state_using;
#[cfg(not(target_os = "macos"))]
use super::streaming_result;
#[cfg(not(target_os = "macos"))]
use super::tier_quality_rank;
#[cfg(not(target_os = "macos"))]
use super::zimage_identity_candle_strength;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use super::Arc;
#[cfg(all(
    native_lane,
    any(
        any(
            all(not(target_os = "macos"), feature = "backend-candle"),
            all(test, target_os = "macos")
        ),
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
use super::Generator;
#[cfg(not(target_os = "macos"))]
use super::Image;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use super::ImagePlan;
#[cfg(not(target_os = "macos"))]
use super::ImageRequest;
#[cfg(all(
    native_lane,
    any(
        any(
            all(not(target_os = "macos"), feature = "backend-candle"),
            all(test, target_os = "macos")
        ),
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
use super::LoadSpec;
#[cfg(not(target_os = "macos"))]
use super::Progress;
#[cfg(not(target_os = "macos"))]
use super::PromptEnhance;
#[cfg(not(target_os = "macos"))]
use super::WeightsSource;
#[cfg(not(target_os = "macos"))]
use crate::emit_event;
#[cfg(not(target_os = "macos"))]
use crate::update_job;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::ApiClient;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::JobSnapshot;
#[cfg(not(target_os = "macos"))]
use crate::JobStatus;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::Path;
#[cfg(not(target_os = "macos"))]
use crate::ProgressStage;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::Settings;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::Value;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::WorkerError;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(
        all(not(target_os = "macos"), feature = "backend-candle"),
        all(test, target_os = "macos")
    )
))]
use crate::WorkerResult;
/// Everything the generic Candle lane conditions one render on, resolved once per job: the
/// single-reference (img2img-init) slot + its strength, the inpaint-mask slot, and the registry
/// editors' ordered `edit_refs` — the candle twin of `resolve_generic_lane_inputs`.
///
/// In-lane edit conditioning (sc-6598 Ideogram / sc-7524 Boogu): resolve the source `Reference`
/// (+ optional `Mask` for Ideogram) + strength once, seed-independent. Both families edit on the
/// SAME engine as their T2I (no separate bespoke stream), so the generic lane resolves the source
/// here. `resolve_ideogram_edit` / `resolve_boogu_edit` return `None` for a non-edit (T2I) job, and
/// each is gated to its family so a stray job reaching this generic lane is untouched. Boogu has no
/// mask (the `boogu_image_edit` descriptor accepts only `Reference`). Other candle edit families
/// (sdxl/flux2/qwen/z-image) have their own bespoke streams (checked before this dispatch).
///
/// Split out of [`generate_candle_stream`] (sc-24110) so the conditioning a job REALLY sends is
/// testable without weights — `generate_one` assembles `build_lane_conditioning` from exactly this
/// tuple.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::type_complexity)]
pub(super) fn resolve_candle_lane_inputs(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
    is_ideogram: bool,
) -> WorkerResult<(
    Option<(Image, f32)>,
    Option<Image>,
    Vec<Image>,
    Vec<Option<image::GrayImage>>,
)> {
    let (edit_reference, edit_mask) = if zimage_identity_candle_strength(request).is_some() {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Z-Image identity requires a referenceAssetId".to_owned(),
                )
            })?;
        let reference = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            reference_id,
            project_path,
        )?;
        let reference =
            fit_engine_image(reference, request.width, request.height, &request.fit_mode)?;
        (
            Some((
                reference,
                zimage_identity_candle_strength(request).expect("checked above"),
            )),
            None,
        )
    } else if is_ideogram {
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => (Some((source, strength)), mask),
            None => (None, None),
        }
    } else if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // `z_image_edit` is a catalog alias for the registered Turbo provider. Resolve its source
        // into the generic request so memory admission, lifecycle cleanup, and telemetry stay shared.
        (
            resolve_zimage_edit_init(request, settings, project_path)?,
            None,
        )
    } else if request.model == "kolors" && request.mode == "edit_image" {
        (
            resolve_candle_kolors_edit_init(request, settings, project_path)?,
            None,
        )
    } else {
        (None, None)
    };
    // Registry instruction edits: resolve Boogu's 1..5 sources or Mage's source-first ordered list.
    // Each uses the `MultiReference`-capable path, not the single `edit_reference` img2img slot.
    // The alpha plane of each `edit_refs` entry — only Qwen-Image 2.1 ever fills it (S4 contract).
    let mut edit_ref_alpha: Vec<Option<image::GrayImage>> = Vec::new();
    let edit_refs: Vec<Image> = if request.model == "boogu_image_edit" {
        resolve_boogu_edit(request, settings, project_path)?
    } else if is_mage_edit_model(&request.model) {
        resolve_mage_edit(request, settings, project_path)?
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110) — the SAME resolver as the MLX lane,
        // because the Candle port registers the same engine id and declares the same
        // `Reference` + `ReferenceRgba` + `MultiReference` conditioning. One request contract, two
        // backends.
        let (images, alpha) = resolve_qwen_image_2_1_edit_images(request, settings, project_path)?;
        edit_ref_alpha = alpha;
        images
    } else if is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
    {
        resolve_sensenova_candle_edit(request, settings, project_path)?
    } else {
        Vec::new()
    };
    // The never-a-Mask / never-a-strength guarantee, on the path that actually runs — the candle
    // twin of the MLX call. `edit_reference` is this lane's single-reference (img2img-init) slot
    // and `edit_mask` its mask slot; both must stay empty for 2.1.
    guard_qwen_image_2_1_lane_slots(request, edit_reference.as_ref(), edit_mask.as_ref())?;
    Ok((edit_reference, edit_mask, edit_refs, edit_ref_alpha))
}

/// Windows/CUDA registry-generator path (sc-3675 SDXL, generalized in sc-5096). This is the Candle
/// sibling of [`generate_stream`], driving the same neutral streaming harness
/// (`start_cached_gen_stream` → `generate_one` → `consume_gen_events`) for base generation and the
/// descriptor-backed reference/edit/identity shapes selected by [`resolve_candle_image_route`].
///
/// Backend-neutral resolution (sc-5096): the per-engine repo / steps / guidance / negative prompt all
/// come from the shared [`mlx_model`] join (`MODEL_TABLE` row + the linked candle descriptor), exactly
/// like the MLX path — so adding a family needs no new dispatch logic, just its provider crate linked.
/// Quant + LoRA/LoKr are **descriptor-gated** (sc-5126): resolved through the same `resolve_quant` /
/// `resolve_adapters` seams as MLX only when the linked Candle descriptor advertises them. Bespoke
/// control/IP/ComfyUI providers divert before this function; unsupported request shapes are refused
/// upstream and remain queued (`image_request_candle_eligible`). Reached only when
/// `backend_candle_enabled` (default off → production routing unchanged until parity).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn generate_candle_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    device_backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    generate_candle_stream_using(
        api,
        settings,
        job,
        plan,
        project_path,
        device_backend,
        asset_writes,
        crate::inference_runtime::load,
        Arc::new(NativeMossStreamRuntime),
    )
    .await
}

/// [`generate_candle_stream`] with the generator loader supplied by the caller — the Candle twin of
/// [`generate_stream_using`] (sc-25680): every pre-load decision, the cached load, the per-item drive
/// and `consume_gen_events` are the production code. Private MOSS source, admission and cache
/// dependencies permit CPU lifecycle tests through this dispatcher without a CUDA device.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    all(test, target_os = "macos")
))]
pub(super) async fn generate_candle_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    runtime: Arc<dyn MossStreamRuntime>,
) -> WorkerResult<()> {
    if sceneworks_core::moss_image::is_moss(&plan.request.model) {
        return generate_moss_stream_with_runtime(
            api,
            settings,
            job,
            plan,
            project_path,
            asset_writes,
            load_generator,
            runtime,
            "candle",
        )
        .await;
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        generate_candle_registry_stream_using(
            api,
            settings,
            job,
            plan,
            project_path,
            _device_backend,
            asset_writes,
            load_generator,
        )
        .await
    }
    #[cfg(target_os = "macos")]
    {
        Err(WorkerError::Engine(
            "generic Candle registry stream is unavailable on macOS".into(),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_registry_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
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

    let adapter_label = candle_adapter_label(&request.model);
    // Join the MODEL_TABLE row with the linked candle descriptor (same resolver the MLX path uses).
    // `None` means the candle provider crate for this id wasn't linked/registered — fail loud rather
    // than silently stubbing.
    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::Engine(format!(
            "candle backend not linked for model {} (no registered generator)",
            request.model
        ))
    })?;
    let engine_id = model.engine_id();
    // Every Klein reference-bearing request belongs to the bespoke Flux2Edit route, which resolves
    // and counts the actual reference set. Reaching the generic registered generator means routing
    // could not prove a usable reference/base; fail closed instead of dropping the reference and
    // silently authorizing the request as text-to-image.
    if engine_id == "flux2_klein_9b" && flux2_klein_reference_bearing_mode(&request.mode) {
        return Err(WorkerError::InvalidPayload(format!(
            "{} {} requires the FLUX.2 reference route and at least one resolvable reference image",
            request.model, request.mode
        )));
    }
    // Report the descriptor's tensor backend ("candle"), not the gpu-id device label
    // (`_device_backend`), on the streamed progress + inference events (sc-3678) — parity with the
    // macOS path's `model.backend()` override, so the worker log + the UI architecture pill clearly
    // attribute the run to Candle.
    let backend = if model.backend().is_empty() {
        "candle"
    } else {
        model.backend()
    };
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    let is_sana = matches!(engine_id, "sana_1600m" | "sana_sprint_1600m");
    let is_sd35 = crate::candle_memory_strategy::is_sd35(engine_id);
    // Chroma, Ideogram and Kolors are priced from the same exact provider receipts SANA and SD3.5
    // are, so their executed load policy is bound by the same authority (sc-20799). Before this
    // they fell through to `resolve_offload(fit_decision(..))`, a manifest estimate, and an
    // admitted STAGED peak could be executed as a RESIDENT load that the receipt never bounded.
    let receipt_priced_route = crate::candle_memory_strategy::is_receipt_priced(engine_id);
    // Standard-tier weight resolution, SHARED with the MLX lane (sc-9092, epic 9083 gap #3). Every
    // candle image family — Ideogram / Boogu / Krea / Lens included — now packed-loads the SAME
    // SceneWorks MLX-packed per-tier turnkey the macOS path uses: as of the candle-gen rollout all 11
    // packed-load-capable crates read a packed `q4/q8/bf16` (or the Ideogram/Boogu/Krea legacy-layout)
    // turnkey subdir directly, so the four ad-hoc `candle_{ideogram,boogu,krea,lens}_repo` resolvers
    // (which pointed candle at a SEPARATE bf16 diffusers rehost because it couldn't read the packed
    // turnkeys) are retired. `resolve_weights_dir` applies the identical dispatch the MLX path uses —
    // an explicit `modelPath` override (FLUX.2-klein `_true_v2` convert-at-install), then the Ideogram
    // (`ideogram_model_subdir`) / Boogu (`boogu_model_subdir`) / Krea (`krea_model_subdir`) per-family
    // subdir, then `standard_tier_subdir` (Lens + the STANDARD_TIER_MODELS registry) resolving the
    // requested `advanced.mlxQuantize` → q4/q8/bf16 tier — so the candle lane needs no bespoke repo
    // logic. `model` is already resolved via `mlx_model` above, so a `None` here means only the
    // snapshot is absent (unfetched turnkey), which stays a loud load error.
    //
    // sc-9607 (epic 9083): off-Mac on-demand fetch of the non-default Ideogram/Boogu tiers before
    // resolution — the catalog pulls only the shipped default (ideogram q4, boogu Q8), so a candle
    // job that opts into another tier (`advanced.mlxQuantize`) needs its subdir pulled first. These
    // were `#[cfg(target_os = "macos")]` (boogu) / absent (ideogram), so off-Mac previously fell back
    // to the default tier; now Windows/Linux gets the same q4/q8/bf16 picker as macOS. No-op for the
    // default tier / every other family / an unfetched turnkey (falls through to the load error below).
    ensure_boogu_tier_present(api, settings, job, request).await?;
    ensure_ideogram_tier_present(api, settings, job, request).await?;
    // Krea 2 INT8-ConvRot tier (sc-9300, epic 9083): fetch the canonical bf16 base surface on demand
    // (the ConvRot catalog download pulls only the DiT single-file), then resolve the two LoadSpec
    // inputs. `None` = not a ConvRot job (or an artifact still absent) → the normal dense/packed path
    // below. When it resolves, the LoadSpec `weights` becomes the bf16 base DIR and `text_encoder` the
    // ConvRot DiT `File` — the exact shape the candle-gen krea engine's `convrot_selector` expects.
    ensure_krea_convrot_base_present(api, settings, job, request).await?;
    let convrot = resolve_krea_convrot(request, settings);
    // `mut` for the sc-10733 capability downtier below: a DEFAULT job whose resolved tier won't fit the
    // live VRAM budget is re-pointed at the highest installed tier that does, before the spec is built.
    let mut weights_dir = if let Some((base_dir, _)) = convrot.as_ref() {
        base_dir.clone()
    } else if matches!(request.model.as_str(), "flux_schnell" | "flux_dev") {
        resolve_candle_flux1_packed_weights_dir(request, settings)?.ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Candle packed FLUX.1 weights snapshot not found for {}",
                request.model
            ))
        })?
    } else {
        resolve_weights_dir(request, settings)?.ok_or_else(|| {
            let repo = model_repo(request, &model);
            WorkerError::InvalidPayload(format!("candle weights snapshot not found for {repo}"))
        })?
    };
    // sc-26241: distilled FLUX 3 selects its package by the request alone — dense bf16 at the
    // snapshot root, a locally converted GGML q4/q8, or the publisher FP8R/NVFP4 at
    // `variants/<tier>` — with the root's bundled encoder/autoencoder as explicit components for
    // every non-dense package. `None` for every other model.
    let flux3_plan = flux3_precision::candle_flux3_plan(
        request,
        settings,
        &weights_dir,
        model.descriptor.capabilities.supported_quants,
    )?;
    if let Some(plan) = flux3_plan.as_ref() {
        weights_dir = plan.weights_dir.clone();
    }

    // Descriptor-derived denoise/guidance surface (distilled families → no guidance/negative; guided
    // families → the scale + negative prompt). Identical to the MLX path; quant + LoRA are omitted.
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let mut true_cfg = resolve_true_cfg(request, &model);
    let hires_fix = resolve_hires_fix_plan(request, steps, guidance, true_cfg);
    // The same request controls the MLX arm resolves (sc-25681): CFG interval, caption-overflow
    // policy and the CFG-bound negative prompt, each refused before load when the linked engine
    // does not honour it or CFG-off would make it inert. Inert for every model that sends none.
    let negative_prompt = resolve_negative_prompt(request, &model);
    let lane_controls = lane_controls::resolve_lane_request_controls(
        request,
        &model.descriptor.capabilities,
        guidance,
        negative_prompt.as_deref(),
    )?;

    // Per-payload flash/accel-attention (sc-3674): the UI Advanced toggle sends `advanced.flashAttn`
    // (default on). Process-global toggle, set before the generator loads (the candle pipeline reads
    // it at load) — race-free because the worker runs image jobs sequentially. The providers expose
    // the runtime knob under different names (SDXL `set_flash_attn`, Z-Image `set_accel_attn`); the
    // diffusion-transformer families (flux/flux2/qwen) bake it via the build feature with no runtime
    // toggle. No effect unless the crate was built with its flash/accel feature.
    let flash_attn = request
        .advanced
        .get("flashAttn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    match request.model.as_str() {
        // realvisxl_lightning shares the candle `sdxl` engine (sc-7176), so the SDXL flash toggle applies.
        // So do the Illustrious-XL finetunes (epic 10609).
        "sdxl"
        | "realvisxl"
        | "realvisxl_lightning"
        | "illustrious_xl_v1"
        | "illustrious_xl_v2" => runtime_cuda::providers::sdxl::set_flash_attn(flash_attn),
        // Base z_image (sc-8679) shares the candle z-image accel-attention toggle with Turbo.
        "z_image_turbo" | "z_image" => runtime_cuda::providers::z_image::set_accel_attn(flash_attn),
        _ => {}
    }

    // Adapters do not participate in tier identity. Quant is resolved only after every possible
    // weights-directory rewrite below, so the final LoadSpec and memory receipt share one tier.
    let adapters = if model.supports_adapters() {
        // Pinned inference applies Krea adapters as additive residuals over ConvRot projections too.
        resolve_adapters(request, settings)?
    } else {
        Vec::new()
    };
    let adapter_count = adapters.len();
    // Each adapter's identity for the image provenance, joined after the render with the engine's
    // own apply report (sc-25681) — the MLX arm's twin.
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &adapters).await;

    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is out-of-
    // distribution and stochastically renders the "Image blocked by safety filter" placeholder. Wrap a
    // non-caption prompt into a minimal valid caption — the same worker-side guarantee the macOS path
    // applies via `ideogram_caption::ensure_caption_prompt`. No-op (a clone) for every other family.
    // (`is_ideogram` was resolved above with the weights repo.)
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    let work: Vec<(i64, String)> = if is_sensenova_candle_model(&request.model)
        && matches!(request.mode.as_str(), "edit_image" | "character_image")
    {
        candle_conditioned_edit_work(request)
    } else {
        (0..request.count as usize)
            .map(|index| (resolve_seed(request, index), prompt.clone()))
            .collect()
    };
    let total = work.len();
    // The single-reference + mask slots and the registry editors' ordered `edit_refs`, resolved once
    // — see `resolve_candle_lane_inputs`, which is also what the sc-24110 live-path test drives.
    let (edit_reference, edit_mask, edit_refs, edit_ref_alpha) =
        resolve_candle_lane_inputs(request, settings, project_path, is_ideogram)?;
    // The S4 four-channel surface (sc-24113): the transparency toggle resolved against the engine's
    // own `supports_alpha_output` (refused by name when it cannot serve it) and assigned onto the
    // request, plus each ordered reference's alpha plane (`edit_ref_alpha`). Default — RGB, no
    // alpha — for every model but Qwen-Image 2.1.
    let rgba_output_channels = crate::qwen_alpha::request_output_channels(
        crate::qwen_alpha::resolve_output_channels(&request.model, &request.advanced)?,
    );
    if is_sensenova_candle_model(&request.model) && !edit_refs.is_empty() {
        true_cfg = Some(resolve_sensenova_candle_true_cfg(request));
    }
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // Generic img2img (reference-guided latent-init, sc-10134, epic 8588): a `ui.img2img` model in a
    // NON-edit mode carrying a `referenceAssetId` resolves to the img2img init `(image, advanced.strength)`,
    // threaded to `generate_one` as the single `Conditioning::Reference` the candle engine routes to its
    // img2img entrypoint (VAE-encode the reference → blend at `sigmas[init_time_step]` → denoise; CFG-free
    // for distilled families, two-forward CFG for the base ones like Krea Raw `render_base_img2img`,
    // sc-10226). Model-agnostic here — the candle router gates which ids reach this lane with a reference
    // (`krea_2_turbo`/`krea_2_raw`, SD3.5, SANA base/Sprint, Z-Image, Boogu, Ideogram all wired).
    // Disjoint from the Ideogram `edit_reference` (edit_image vs text_to_image) and the registry
    // editors' `edit_refs` (guarded here so a future overlap never double-drives the single slot).
    let img2img_reference = if edit_reference.is_none()
        && edit_refs.is_empty()
        && uses_generic_img2img(request, has_reference)
    {
        resolve_img2img_init_generic(request, settings, project_path)?
    } else {
        None
    };
    let (width, height) = (request.width, request.height);
    // Per-payload sampler / scheduler / schedule-shift, mirroring the MLX `generate_stream` lane (the
    // 1753 front-half advanced carrier — epic 7114 P5, sc-7127). RealVisXL Lightning (sc-7176) forces the
    // few-step `lightning` id regardless of the payload: candle-gen-sdxl advertises `["ddim", "lightning"]`,
    // so it survives the N3 guard below. Every value is then run through `normalize_sampling_knob` against
    // this family's advertised surface — a name candle doesn't honor (candle adopts the unified framework in
    // P4, so most families advertise only their family default today) is dropped back to the engine default
    // + a `sampling_knob_unsupported` event, never a hard-fail. The curated knobs light up per-family with
    // zero worker change as the candle engines are adopted.
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
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
    // Guidance method (epic 7434 P5, sc-7448), N3-guarded against the candle descriptor's advertised set.
    let guidance_method = normalize_sampling_knob(
        read_advanced_guidance_method(&request.advanced),
        &caps.supported_guidance_methods,
        "guidanceMethod",
        &request.model,
        &job.id,
        backend,
    );
    // sc-18474: FLUX.2-dev prompt enhancement is native on Candle too. The request-local report
    // sink below is the proof seam: an enabled request cannot silently run the raw prompt without a
    // typed, honestly persisted fallback. The stale passthrough-only behavior is no longer allowed.
    let enhance = PromptEnhance::from_advanced(&request.model, &request.advanced)?;
    // Record the effective CFG knob (guidance for guided families, else true_cfg) + quant bits in the
    // recipe, so a Lens asset's sidecar reflects the Q4/Q8 it ran at (parity with the MLX path). The
    // recorded repo is the resolved model repo (the MLX turnkey the candle lane now packed-loads from,
    // sc-9092) — the same `model_repo` the MLX path records.
    let repo = model_repo(request, &model);
    // Per-generation PiD decode (epic 7840): resolve the PiD checkpoint + Gemma for this model's latent
    // space when `advanced.usePid` is set and the snapshots are cached; otherwise keep the native VAE.
    // `use_pid` and `spec.pid` stay in lockstep (the engine rejects a mismatch). Every candle image
    // provider reads `spec.pid` (candle-gen sc-7853: sdxl/flux/flux2/qwen-image/z-image/chroma/boogu/
    // ideogram/kolors/krea/lens), so this un-gates the toggle across the whole off-Mac catalog — using the
    // SAME `resolve_pid_weights` model→backbone gate as the macOS lane (a non-eligible model → `None` →
    // native VAE, so this is a no-op for anything without a PiD backbone).
    // ConvRot (sc-9300) does not combine with a PiD decode overlay — the int8 DiT consume path replaces
    // the transformer wholesale and is not PiD-wired (the candle-gen krea engine rejects the combo). So
    // suppress PiD when ConvRot is selected; every non-ConvRot job resolves PiD exactly as before.
    let pid_weights = if convrot.is_some() {
        None
    } else {
        resolve_pid_weights(request, &settings.data_dir, &request.model)?
    };
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
                edit_reference.is_some()
                    || img2img_reference.is_some()
                    || !edit_refs.is_empty()
                    || edit_mask.is_some(),
            ),
            ("prompt enhancement", enhance.enabled),
        ],
    )?;
    // The batch renders every image in one call: the stream reports all of them.
    let total = native_batch
        .as_ref()
        .map_or(total, |batch| batch.images() as usize);
    // PiD output tier (sc-10054): 2K caps the effective base so PiD's fixed 4× lands on ~2048 (default
    // 4K/native leaves the requested dims untouched). Rebind before `generate_one`.
    let (width, height) = pid_effective_dims(width, height, use_pid, pid_output_tier(request));
    // Admission describes the heaviest pass. Hires fix renders the first pass at `width`/`height`
    // and the refinement at the plan dimensions, so every memory fit and request scope must use the
    // latter. `generate_one_with_hires` derives a base-pass scope from this final-pass identity.
    let (memory_width, memory_height) =
        hires_fix.map_or((width, height), |plan| (plan.width, plan.height));
    // Krea "text style" tap-reweight gain — see `resolve_text_style_gain` (sc-11878, gate fixed sc-12008).
    let text_style_gain = resolve_text_style_gain(request);
    // VRAM fit-gate (epic 10765, sc-10766 Phase 0 + sc-10821 Phase 1b + sc-10856): when the selected
    // tier's predicted resident peak won't fit the card, either RUN SEQUENTIALLY (a provider that
    // supports sequential component residency — the candle FLUX lane, sc-10769 — drops the text encoders
    // before the DiT so peak = DiT+VAE, not TE+DiT+VAE) or, for a family that has not wired it, reject-
    // before-OOM with an actionable message. sc-10856 adds a second stage on the sequential path: when
    // the tier's MEASURED sequential peak (`candle.sequentialPeakGb`) is known and STILL won't fit, reject
    // instead of running into a reactive OOM. Honors `SCENEWORKS_CUDA_VRAM_CAP_GB` to emulate a small
    // card. Unmeasured models (no `candle` block) and non-NVIDIA hosts yield `Unknown` → never block.
    //
    // sc-25684: a resident restoration transform (the Iris `upscaler/` task, sc-25683) is evicted by
    // the generator cache's load anyway — evict it BEFORE probing, so its VRAM is not counted as
    // occupied and a generation that fits once it is gone is not refused or demoted to staging
    // (the reclaimable credit below covers only a resident GENERATOR). A no-op when none is loaded.
    if crate::transform_cache::evict_resident_transform().await? {
        tracing::info!(
            event = "candle_generation_evicted_resident_transform",
            job_id = %job.id,
            model = %request.model,
        );
    }
    let raw_budget = crate::vram_gate::live_capped_budget(&settings.gpu_id).await;
    // sc-11023: the single-slot generator cache evicts its current occupant BEFORE the incoming load.
    // This gate reads `free` while that occupant is still resident (so raw `free` still counts the model
    // it is about to replace); crediting the reclaimable pool predicts the `free` the imminent evict will
    // PRODUCE. That prediction is right whether evicting returns the VRAM to the driver (GPU-measured
    // sc-13960: a full generator drop frees most of it back, `nvidia-smi` free RISES) or a within-device
    // component free keeps it pooled in-process. So budget against `free + reclaimable` (capped to total),
    // else a warm/swap re-gate falsely rejects a load that will actually fit (the "even with sequential
    // residency" 2nd-run reject a resident bf16 tier hits on the next generation).
    let reclaimable_gb = crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id);
    let budget =
        raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable_gb));
    // sc-22664: the shared ladder's operational reserve is derived from the RAW probe, never from
    // the reclaimable-credited `budget` above — the credit predicts the free the imminent evict
    // will produce, so on a warm run `total − free` of the credited budget reads the card as
    // nearly idle and the reserve would collapse to the bare margin (`vram_gate::ladder_reserve_gb`).
    let shared_reserve_gb = raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb);
    // sc-12090: budget + name the tier the disk-probing resolver ACTUALLY landed on (`weights_dir`),
    // not a manifest re-derivation that ignores what's installed. `requested_tier_key` re-derived from
    // `mlx.quantize` with no disk check, so a q4-only install was budgeted (and rejected) against a q8
    // the user never downloaded. Read the resolved on-disk tier instead — one value, both named and
    // budgeted. ConvRot loads an int8 DiT over the bf16 base surface (its footprint is neither the bf16
    // nor the q8 tier), and a `modelPath`/flat root has no recognizable tier basename — fall back to the
    // manifest key there, preserving today's behavior.
    //
    // sc-11042: the NVFP4 tier composes with the sc-12090 disk probe rather than bypassing it.
    // `tier_key_from_resolved_dir` only recognizes the bits-based basenames (bf16/q8/q4), so a resolved
    // `nvfp4/` dir yields `None` and falls through to `requested_tier_key`, whose `nvfp4` short-circuit
    // names the tier by IDENTITY (never by bits — `Quant::Nvfp4.bits()` is 4, which would alias q4).
    // `nvfp4_selected` reads that same resolved `weights_dir`, so a `quantTier: "nvfp4"` label that fell
    // back to another tier's dir is sized/named as the tier that will actually load, not as nvfp4.
    // sc-12425: a resolved ConvRot load is named by its tier IDENTITY (see [`gate_tier_key`]) — it used
    // to be handed to the bits-derived `requested_tier_key`, which aliased it to q8 and under-gated it.
    // The comment above already knew "its footprint is neither the bf16 nor the q8 tier"; now the gate
    // acts on it. Extracted so that mapping has a unit test; this fn cannot be exercised from one.
    let mut tier = candle_resolved_tier_key(request, &weights_dir, convrot.is_some());
    if matches!(engine_id, "sana_1600m" | "sana_sprint_1600m") {
        tier = "bf16";
    }
    // FLUX 3's tier is the request-named package identity: neither the snapshot basename nor the
    // bits-ladder `q8` default the generic key derives for an unrecognized directory.
    if let Some(plan) = flux3_plan.as_ref() {
        tier = plan.tier.key();
    }
    validate_candle_tier_memory_evidence(&request.model, &request.model_manifest_entry, tier)?;
    let requested_tier = tier;
    // sc-12130: derive Candle residency support from the provider's weights-free descriptor instead of
    // maintaining a second engine-id allowlist in the worker. The capability bit is the provider's
    // contract that every request shape accepted by this id honors Sequential. Bespoke edit/control,
    // ComfyUI, and strict-control routes are diverted by `resolve_candle_image_route` before this gate;
    // generic txt2img/img2img reaches this point and uses the same registry-derived signal for both the
    // capability downtier and the resident/sequential decision.
    let sequential_capable = crate::mlx_fit_gate::engine_supports_sequential(engine_id);
    // SC-15117: the deeper, request-scoped Krea Turbo ladder is intentionally limited to the stock
    // ordinary single-pass txt2img route implemented by candle-gen-krea. Reference/edit/hires/PiD/
    // ConvRot surfaces keep their established paths. Hires is deliberately excluded: its refinement
    // is img2img, which the Krea request scope does not implement. Adapter jobs have no calibrated
    // evidence cells and therefore fail closed to resident-or-reject; evidence from ordinary
    // text-to-image never transfers to them.
    let krea_turbo_ladder = krea_turbo_memory_route(
        engine_id,
        convrot.is_some(),
        edit_reference.is_some(),
        img2img_reference.is_some(),
        !edit_refs.is_empty(),
        edit_mask.is_some(),
        hires_fix.is_some(),
        use_pid,
    );
    let (krea_allow_streamed_blocks, krea_adapter_bytes) =
        krea_streamed_blocks_adapter_evidence(&adapters);
    if adapter_count > 0 && krea_adapter_bytes.is_none() {
        return Err(WorkerError::InvalidPayload(format!(
            "{} includes {adapter_count} adapter(s), but their resident bytes are unavailable; \
             refusing an unbounded Candle request",
            request.model
        )));
    }
    let adapter_source_bytes = krea_adapter_bytes.unwrap_or(0);
    // Ordinary SDXL reaches a request-scoped provider contract whose staging support is narrower
    // than the descriptor shared with its bespoke surfaces. Price each installed candidate with
    // that contract before the resident-only capability clamp can reject it. Other SDXL surfaces
    // retain their existing gates; no descriptor capability is broadened here.
    let sdxl_request_tier_admission = engine_id == "sdxl"
        && request.model == "sdxl"
        && matches!(request.mode.as_str(), "image_generation" | "text_to_image")
        && edit_reference.is_none()
        && img2img_reference.is_none()
        && edit_refs.is_empty()
        && edit_mask.is_none()
        && hires_fix.is_none()
        && !use_pid
        && native_batch.is_none();
    let sdxl_endpoints = if sdxl_request_tier_admission
        && request.count == 1
        && adapter_count == 0
        && flash_attn
        && lane_controls.cfg_interval.is_none()
        && guidance_method.is_none()
        && scheduler.is_none()
        && scheduler_shift.is_none()
    {
        sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::packaged_sdxl_cuda_memory_endpoints()
            .filter(|endpoints| {
                endpoints.settings.get("steps").and_then(Value::as_u64) == Some(u64::from(steps))
                    && endpoints.settings.get("guidance").and_then(Value::as_f64)
                        == guidance.map(f64::from)
                    && endpoints.settings.get("flashAttn").and_then(Value::as_bool) == Some(true)
                    && sampler.as_deref().is_none_or(|sampler| {
                        endpoints.resolved_defaults.get("sampler").and_then(Value::as_str)
                            == Some(sampler)
                    })
            })
    } else {
        None
    };
    // sc-10733 capability downtier: for a DEFAULT job (no explicit per-(screen,model) pick, and not the
    // bespoke ConvRot tier), if the resolved tier won't fit the live budget, step DOWN to the highest
    // installed tier that does — floored at the per-model quality floor — rejecting only when nothing
    // >= floor fits. An explicit pick (`mlxQuantizeExplicit`) is HONORED: it skips the downtier and runs
    // the plain gate below (fits → run; too big → reject-before-OOM), never silently downtiered (#7).
    //
    // sc-11042 (epic 11037 SC#5): a selected NVFP4 tier is never downtiered either, and does not rely on
    // `mlxQuantizeExplicit` to escape — the web omits that flag for nvfp4 (it rides inside the
    // `tierQuantize(quantTier) !== null` bits branch, and nvfp4 has no honest `mlxQuantize` integer).
    // Instead NVFP4 is unrankable ON PURPOSE: `tier_quality_rank("nvfp4")` is 0 because nvfp4 is a
    // distinct numeric regime, not a rung on the bf16/q8/q4 fidelity ladder, so
    // `downtier_candidate_tiers` yields NO candidates in `[floor, nvfp4]` and `choose_downtier` returns
    // `Keep`. Downtiering NVFP4 to q4/q8 would silently swap the numerics of an explicitly-picked tier —
    // exactly the creative-choice violation SC#5 forbids. Pinned by
    // `nvfp4_tier_is_never_downtiered_by_the_capability_clamp`.
    let explicit_pick = candle_tier_pick_is_exact(&request.model, &request.advanced);
    // FLUX 3's package is request-named and is gated as named: rejected before load if it does
    // not fit, never downtiered to a package the request did not choose.
    if convrot.is_none() && !explicit_pick && flux3_plan.is_none() {
        let floor = capability_downtier_floor(krea_turbo_ladder, tier, min_quality_floor(request));
        let candidates: Vec<(&'static str, TierFit)> =
            downtier_candidate_tiers(request, settings, tier, floor)
                .into_iter()
                .map(|candidate| -> WorkerResult<_> {
                    let candidate_adapter_bytes =
                        candle_adapter_resident_bytes(engine_id, candidate, adapter_source_bytes);
                    let fit = if sdxl_request_tier_admission {
                        let legacy = candle_tier_fit(
                            &request.model_manifest_entry,
                            candidate,
                            budget,
                            sequential_capable,
                            candidate_adapter_bytes,
                        );
                        let selected = if let Some(dir) =
                            resolve_tier_dir(request, settings, candidate)
                        {
                            let (candidate_quant, _) = candle_quant_for_resolved_tier(
                                request,
                                candidate,
                                &dir,
                                model.supports_quant(),
                                false,
                            );
                            let spec = load_spec(dir, candidate_quant, adapters.clone(), None)
                                .with_resolved_route(request.model.clone());
                            let spec = attach_required_components(
                                spec,
                                engine_id,
                                &request.model_manifest_entry,
                                settings,
                            )?;
                            let spec =
                                attach_manifest_text_encoder(spec, engine_id, request, settings)?
                                    .into_load_spec();
                            let mut spec = apply_declared_candle_image_load_shape(
                                engine_id,
                                Some(candidate),
                                &request.mode,
                                &request.model_manifest_entry,
                                spec,
                                false,
                            );
                            let certified = candle_certified_load_spec(
                                engine_id,
                                settings,
                                &spec,
                                &request.model_manifest_entry,
                                candidate,
                            );
                            let candidate_endpoints = match sdxl_endpoints {
                                Some(endpoints) if certified && candidate == "q4" => {
                                    seal_sdxl_endpoint_load_spec(
                                        &mut spec,
                                        &endpoints.artifact.files,
                                    )?
                                    .then_some(endpoints)
                                }
                                _ => None,
                            };
                            let overlay_bytes = candidate_adapter_bytes.saturating_add(
                                lane_controls::declared_activation_overlay_bytes(
                                    request,
                                    None,
                                    memory_width,
                                    memory_height,
                                )?,
                            );
                            crate::candle_memory_strategy::evaluate_shared_image_priced(
                                engine_id,
                                &request.model,
                                &spec,
                                certified,
                                &request.model_manifest_entry,
                                candidate,
                                &request.mode,
                                (adapter_count > 0).then_some("lora"),
                                gen_core::MemoryGeometry {
                                    width: memory_width,
                                    height: memory_height,
                                    batch: 1,
                                    frames: 1,
                                    reference_count: 0,
                                },
                                false,
                                false,
                                false,
                                false,
                                budget,
                                shared_reserve_gb,
                                crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
                                    &request.model_manifest_entry,
                                    candidate,
                                    overlay_bytes,
                                ),
                                overlay_bytes,
                                if reclaimable_gb > 0.0 {
                                    gen_core::MemoryCacheState::Warm
                                } else {
                                    gen_core::MemoryCacheState::Cold
                                },
                                None,
                                candidate_endpoints,
                            )?
                        } else {
                            None
                        };
                        candle_tier_fit_after_shared_selection(legacy, selected.as_ref())
                    } else if krea_turbo_ladder && adapter_count == 0 {
                        let candidate_runtime = resolve_tier_dir(request, settings, candidate)
                            .and_then(|dir| {
                                krea_runtime_evidence_context(request, settings, candidate, &dir)
                            });
                        match crate::vram_gate::krea_turbo_fit_with_runtime(
                            &request.model_manifest_entry,
                            candidate,
                            memory_width,
                            memory_height,
                            budget,
                            shared_reserve_gb,
                            krea_allow_streamed_blocks,
                            candidate_runtime.as_ref(),
                        ) {
                            Some(
                                crate::vram_gate::KreaTurboFit::Resident { .. }
                                | crate::vram_gate::KreaTurboFit::Fits { .. },
                            ) => TierFit::Fits,
                            Some(crate::vram_gate::KreaTurboFit::Reject { needed_gb, .. }) => {
                                TierFit::TooBig {
                                    needed_gb,
                                    available_gb: budget.map_or(0.0, |budget| budget.free_gb),
                                }
                            }
                            Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                                log_krea_turbo_evidence_exclusion(
                                    &request.model,
                                    candidate,
                                    width,
                                    height,
                                    "capability_downtier",
                                    Some(reason),
                                );
                                candle_tier_fit(
                                    &request.model_manifest_entry,
                                    candidate,
                                    budget,
                                    false,
                                    candidate_adapter_bytes,
                                )
                            }
                            None => {
                                log_krea_turbo_evidence_exclusion(
                                    &request.model,
                                    candidate,
                                    width,
                                    height,
                                    "capability_downtier",
                                    None,
                                );
                                candle_tier_fit(
                                    &request.model_manifest_entry,
                                    candidate,
                                    budget,
                                    false,
                                    candidate_adapter_bytes,
                                )
                            }
                        }
                    } else {
                        candle_tier_fit(
                            &request.model_manifest_entry,
                            candidate,
                            budget,
                            sequential_capable,
                            candidate_adapter_bytes,
                        )
                    };
                    Ok((candidate, fit))
                })
                .collect::<WorkerResult<_>>()?;
        match choose_downtier(tier, &candidates) {
            DowntierPick::Keep => {}
            DowntierPick::Downtier(chosen) => {
                // The chosen tier came from `downtier_candidate_tiers`, so re-resolving its dir yields
                // Some; fall through defensively rather than unwrap-panic if it somehow doesn't.
                if let Some(dir) = resolve_tier_dir(request, settings, chosen) {
                    tracing::warn!(
                        model = %request.model,
                        from = %tier,
                        to = %chosen,
                        "candle VRAM fit-gate: default tier won't fit — downtiering to the highest \
                         installed tier that does (capability clamp, sc-10733)"
                    );
                    weights_dir = dir;
                    tier = chosen;
                }
            }
            DowntierPick::Reject {
                tier: smallest,
                needed_gb,
                available_gb,
            } => {
                if krea_turbo_ladder && adapter_count == 0 {
                    let smallest_runtime =
                        resolve_tier_dir(request, settings, smallest).and_then(|dir| {
                            krea_runtime_evidence_context(request, settings, smallest, &dir)
                        });
                    let lower_resolution = crate::vram_gate::krea_turbo_smaller_fit_with_runtime(
                        &request.model_manifest_entry,
                        smallest,
                        memory_width,
                        memory_height,
                        budget,
                        shared_reserve_gb,
                        krea_allow_streamed_blocks,
                        smallest_runtime.as_ref(),
                    );
                    let installed_smaller: Vec<&'static str> =
                        installed_tier_keys(request, settings)
                            .into_iter()
                            .filter(|candidate| {
                                let candidate_runtime = resolve_tier_dir(
                                    request, settings, candidate,
                                )
                                .and_then(|dir| {
                                    krea_runtime_evidence_context(
                                        request, settings, candidate, &dir,
                                    )
                                });
                                tier_quality_rank(candidate) < tier_quality_rank(smallest)
                                    && matches!(
                                        crate::vram_gate::krea_turbo_fit_with_runtime(
                                            &request.model_manifest_entry,
                                            candidate,
                                            memory_width,
                                            memory_height,
                                            budget,
                                            shared_reserve_gb,
                                            krea_allow_streamed_blocks,
                                            candidate_runtime.as_ref(),
                                        ),
                                        Some(
                                            crate::vram_gate::KreaTurboFit::Resident { .. }
                                                | crate::vram_gate::KreaTurboFit::Fits { .. }
                                        )
                                    )
                            })
                            .collect();
                    return Err(WorkerError::InvalidPayload(
                        krea_turbo_capability_reject_message(
                            &request.model,
                            smallest,
                            needed_gb,
                            available_gb,
                            &settings.gpu_id,
                            lower_resolution,
                            &installed_smaller,
                        ),
                    ));
                }
                return Err(WorkerError::InvalidPayload(
                    candle_capability_reject_message(
                        &request.model,
                        smallest,
                        needed_gb,
                        available_gb,
                        &settings.gpu_id,
                        // The SDXL admission path scores each candidate through the shared
                        // selector, which prices the request geometry: a smaller output can
                        // bring it inside what the measured endpoints cover.
                        ResolutionLever::for_selector(sdxl_request_tier_admission),
                    ),
                ));
            }
        }
    }
    // Reconcile only after the capability clamp has made its final directory/tier decision. The
    // resolved artifact drives recipe bits, MemoryNumericTier, active component floors, and the
    // provider begin-request check. `LoadSpec.quantize` remains descriptor-governed: in particular,
    // FLUX.1 packed-detects q4/q8 and therefore receives `None` while its receipt keeps the tier bits.
    let (quant, quant_bits) = match flux3_plan.as_ref() {
        // FP8R/NVFP4 are tier identities with no honest bit count on the q4/q8 ladder, so the
        // recipe's `mlxQuantize` bits stay empty and the load quant alone carries the format.
        Some(plan) => plan.quant_and_bits(),
        None => candle_quant_for_resolved_tier(
            request,
            tier,
            &weights_dir,
            model.supports_quant(),
            convrot.is_some() || is_anima_model(&request.model),
        ),
    };
    let mut raw_settings = mlx_raw_settings_for_bundle(
        request,
        Some(&weights_dir),
        &repo,
        steps,
        quant_bits,
        guidance.or(true_cfg),
    );
    // The declared pixel-space activations beyond the one 1024² image the floors carry — a native
    // prompt batch's extra images (sc-25681) and an output above 1024² (sc-25684) — ride every
    // resident/sequential prediction below exactly like resident adapter bytes do:
    // `candle.imageActivationGbPerMegapixel × (images × max(1, MP) − 1)`. Zero otherwise.
    let batch_overlay_bytes = lane_controls::declared_activation_overlay_bytes(
        request,
        native_batch.as_ref(),
        memory_width,
        memory_height,
    )?;
    let source_activation_bytes = crate::vram_gate::flux3_source_activation_allowance_bytes(
        engine_id,
        memory_width,
        memory_height,
    );
    let adapter_resident_bytes =
        candle_adapter_resident_bytes(engine_id, tier, adapter_source_bytes)
            .saturating_add(batch_overlay_bytes)
            .saturating_add(source_activation_bytes);
    let needed = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        adapter_resident_bytes,
    );
    // Chroma's selector prices the exact provider receipt after the load spec is sealed. Passing
    // generic source-file adapter bytes here would double-charge adapters, omit PiD's materialized
    // student/Gemma footprint, and let the public `lora` cell stand in for load identity. Other
    // providers retain their established source-byte accounting.
    let provider_receipt_priced = matches!(
        engine_id,
        "flux3_klein"
            | "flux3_klein_base"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "ideogram_4"
            | "ideogram_4_turbo"
            | "sana_1600m"
            | "sana_sprint_1600m"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
    );
    let shared_predicted_peak_gb = if provider_receipt_priced {
        let base = crate::vram_gate::predicted_peak_gb(&request.model_manifest_entry, tier);
        if matches!(engine_id, "flux3_klein" | "flux3_klein_base") {
            // Provider receipts replace only adapter/PUP weights. Keep source-derived activation
            // work exactly once in the base peak; never re-label it as adapter residency.
            base.map(|peak| {
                peak + batch_overlay_bytes.saturating_add(source_activation_bytes) as f64
                    / 1_073_741_824.0
            })
        } else {
            base
        }
    } else {
        needed
    };
    let shared_runtime_overlay_bytes = if provider_receipt_priced {
        0
    } else {
        adapter_resident_bytes
    };
    let krea_runtime_context = krea_turbo_ladder
        .then(|| krea_runtime_evidence_context(request, settings, tier, &weights_dir))
        .flatten();
    // sc-12090 AC#4/#5: the reject suggestions name only tiers actually INSTALLED and lower-fidelity than
    // the one being rejected — never the rejected tier, never the picker (hidden when ≤1 tier installed).
    // Reached only on the explicit-pick / ConvRot reject below (the downtier path already rejected above
    // when nothing smaller fits), where suggesting a smaller installed tier the user could pick is apt.
    let mut generation_memory: Option<gen_core::GenerationMemory> = None;
    let mut hires_first_pass_generation_memory: Option<gen_core::GenerationMemory> = None;
    let mut memory_strategy_selection: Option<gen_core::MemorySelection> = None;
    let mut selected_memory_strategy_context: Option<gen_core::MemoryRunContext> = None;
    let mut hires_first_pass_memory_context: Option<gen_core::MemoryRunContext> = None;
    let mut ideogram_warm_staged: Option<
        crate::candle_memory_strategy::CandleWarmStagedEvaluation,
    > = None;
    let mut ideogram_hires_first_warm_staged: Option<
        crate::candle_memory_strategy::CandleWarmStagedEvaluation,
    > = None;
    let mut adapted_peak_gb: Option<f64> = None;
    // Every adopting provider uses the same worker-owned selector. Static
    // Implemented/unverified declarations do not authorize optimized execution: this bridge always
    // submits the conservative resident estimate and adds deeper candidates only when an exact
    // authoritative record exists in the packaged evidence bundle.
    let mut shared_contract_spec = load_spec(weights_dir.clone(), quant, adapters.clone(), None)
        .with_resolved_route(request.model.clone());
    if let Some(pid) = pid_weights.as_ref() {
        shared_contract_spec =
            shared_contract_spec.with_pid(pid.checkpoint.clone(), pid.gemma.clone());
    }
    // The selector and the eventual provider load must inspect the SAME artifact set. Mage's tier
    // directory contains only the DiT; its shared text encoder and VAE are caller-staged named
    // components. Evaluating the bare tier dir undercounts the provider contract and can admit a
    // strategy against assets different from those the generator loads.
    shared_contract_spec = attach_required_components(
        shared_contract_spec,
        engine_id,
        &request.model_manifest_entry,
        settings,
    )?;
    shared_contract_spec =
        attach_prompt_upsampler(shared_contract_spec, engine_id, request, settings)?;
    if let Some(plan) = flux3_plan.as_ref() {
        shared_contract_spec = plan.attach_components(shared_contract_spec);
    }
    if let Some((_, convrot_dit)) = convrot.as_ref() {
        shared_contract_spec = shared_contract_spec.with_component(
            gen_core::KREA_CONVROT_DIT_COMPONENT,
            WeightsSource::File(convrot_dit.clone()),
        );
    }
    // Export the selected encoder receipt only after the entire load shape is complete; this exact
    // prepared spec is reused by the selector and eventual provider/cache load.
    let unattached_shared_contract_spec = shared_contract_spec;
    let attached_shared_contract_spec = attach_manifest_text_encoder(
        unattached_shared_contract_spec,
        engine_id,
        request,
        settings,
    )?;
    let mut shared_contract_spec = attached_shared_contract_spec.into_load_spec();
    let shared_request_mode = candle_base_memory_request_mode(
        engine_id,
        if flux3_references.is_some() {
            "edit_image"
        } else {
            &request.mode
        },
    );
    // The numeric tier the shared selector binds this request to — the tier the provider LOADS,
    // which for a single-artifact dense route is not the gate's bits-derived key (see
    // `candle_shared_memory_tier`).
    let shared_tier = candle_shared_memory_tier(engine_id, tier);
    // A supported Hires refinement is a fresh one-reference image-to-image request even when the
    // caller's first pass was Edit or Inpaint. Admit that final-pass identity independently; the
    // original mode/reference/geometry is admitted below for the first pass.
    let shared_admission_mode = if hires_fix.is_some()
        && matches!(
            engine_id,
            "ideogram_4" | "ideogram_4_turbo" | "sana_1600m" | "sana_sprint_1600m" | "sdxl"
        ) {
        "image_to_image"
    } else {
        shared_request_mode
    };
    shared_contract_spec = apply_declared_candle_image_load_shape(
        engine_id,
        Some(shared_tier),
        shared_admission_mode,
        &request.model_manifest_entry,
        shared_contract_spec,
        false,
    );
    let first_pass_reference_count = lane_reference_count(
        edit_reference.is_some() || img2img_reference.is_some(),
        edit_refs.len().max(flux3_reference_count),
        edit_mask.is_some(),
    );
    let reference_count = shared_image_reference_count(
        edit_refs.len().max(flux3_reference_count),
        edit_reference.is_some() || img2img_reference.is_some(),
        edit_mask.is_some(),
        hires_fix.is_some(),
    )
    .saturating_add(
        u32::try_from(
            usize::from(shared_contract_spec.control.is_some())
                + shared_contract_spec.extra_controls.len(),
        )
        .unwrap_or(u32::MAX),
    );
    let artifact_is_certified = candle_certified_load_spec(
        engine_id,
        settings,
        &shared_contract_spec,
        &request.model_manifest_entry,
        shared_tier,
    );
    seal_recursive_receipt_load_spec(engine_id, artifact_is_certified, &mut shared_contract_spec)?;
    seal_ideogram_load_spec(engine_id, artifact_is_certified, &mut shared_contract_spec)?;
    let sdxl_endpoints = match sdxl_endpoints {
        Some(endpoints) if artifact_is_certified && shared_tier == "q4" => {
            seal_sdxl_endpoint_load_spec(&mut shared_contract_spec, &endpoints.artifact.files)?
                .then_some(endpoints)
        }
        _ => None,
    };
    // Epic 25899: the provider's prefix KV cache, priced from the linked provider for THIS request
    // (`None` for every provider without one — its admission is unchanged). The Candle surcharge
    // follows the build's production device width, not the tier.
    let prefix_kv_cache_surcharge = candle_prefix_kv_cache_surcharge(
        engine_id,
        memory_width,
        memory_height,
        reference_count,
        crate::execution_planner::evaluates_negative_branch(
            negative_prompt.as_deref().unwrap_or_default(),
            true_cfg,
            guidance,
        ),
        // A hires refinement is a second, differently shaped pass, and a native prompt batch holds
        // one prefix per prompt: the cache is priced for neither (the MLX lane's same rule,
        // `MlxRequestInputs::prefix_kv_cache_eligible`).
        hires_fix.is_none() && native_batch.is_none(),
    );
    let shared_memory = if let Some(references) = flux3_references.as_ref() {
        crate::candle_memory_strategy::evaluate_shared_image_priced_with_resources(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_admission_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width: memory_width,
                height: memory_height,
                // A native prompt batch (sc-25681) is ONE provider call over `prompts × count`
                // images, so that is the batch the request is admitted at — the provider's request
                // scope bounds the images it will render by exactly this figure (`candle-gen-iris`
                // refuses `prompts × count > MemoryGeometry::batch`). The batch's memory is charged
                // once, by the declared activation overlay above; the selector does not scale the
                // same activations by this batch again (`candle_memory_strategy::law_geometry`).
                batch: native_batch
                    .as_ref()
                    .map_or(1, lane_controls::NativePromptBatch::images),
                frames: 1,
                reference_count,
            },
            reference_count > 0,
            use_pid && hires_fix.is_none(),
            hires_fix.is_some(),
            hires_fix.is_some(),
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
            prefix_kv_cache_surcharge,
            Some(&references.resources),
            sdxl_endpoints,
        )
    } else {
        crate::candle_memory_strategy::evaluate_shared_image_priced(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_admission_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width: memory_width,
                height: memory_height,
                // A native prompt batch (sc-25681) is ONE provider call over `prompts × count`
                // images, so that is the batch the request is admitted at — the provider's request
                // scope bounds the images it will render by exactly this figure (`candle-gen-iris`
                // refuses `prompts × count > MemoryGeometry::batch`). The batch's memory is charged
                // once, by the declared activation overlay above; the selector does not scale the
                // same activations by this batch again (`candle_memory_strategy::law_geometry`).
                batch: native_batch
                    .as_ref()
                    .map_or(1, lane_controls::NativePromptBatch::images),
                frames: 1,
                reference_count,
            },
            reference_count > 0,
            use_pid && hires_fix.is_none(),
            hires_fix.is_some(),
            hires_fix.is_some(),
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
            prefix_kv_cache_surcharge,
            sdxl_endpoints,
        )
    }?;
    if flux3_references.is_some() && shared_memory.is_none() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.3 references require a matching provider memory contract and an available device budget; this request cannot use legacy unpriced admission".to_owned(),
        ));
    }
    let first_pass_shared_memory = if hires_fix.is_some()
        && matches!(
            engine_id,
            "ideogram_4" | "ideogram_4_turbo" | "sana_1600m" | "sana_sprint_1600m" | "sdxl"
        ) {
        crate::candle_memory_strategy::evaluate_shared_image(
            engine_id,
            &request.model,
            &shared_contract_spec,
            artifact_is_certified,
            &request.model_manifest_entry,
            shared_tier,
            shared_request_mode,
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width,
                height,
                batch: 1,
                frames: 1,
                reference_count: first_pass_reference_count,
            },
            first_pass_reference_count > 0,
            use_pid,
            false,
            false,
            budget,
            shared_reserve_gb,
            shared_predicted_peak_gb,
            shared_runtime_overlay_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
        )?
    } else {
        None
    };
    // The receipt-priced selection, kept even when it is Resident. `optimized_shared_memory_context`
    // below deliberately drops a Resident context for families outside its allowlist, so it cannot
    // answer "did the receipt price this request?" — and that is exactly the question the executed
    // load policy has to agree with.
    let mut receipt_priced_selection: Option<gen_core::MemoryStrategy> = None;
    // What the admission decided about the prefix KV cache, recorded with each image (epic 25899).
    let mut prefix_kv_cache_plan: Option<crate::execution_planner::PrefixKvCachePlan> = None;
    if let Some(evaluation) = shared_memory {
        prefix_kv_cache_plan = evaluation.prefix_kv_cache;
        if receipt_priced_route {
            receipt_priced_selection = Some(evaluation.context.selection.strategy);
        } else {
            // sc-22664 (epic 22657 E7): the shared ladder's selection, with the rung the selector
            // chose and the three phase peaks the law derived for it, so an OOM under this
            // selection is attributable to the exact estimate that admitted it. The receipt-priced
            // families emit their own exact-receipt event after the load (below) and are not
            // duplicated here.
            let telemetry = evaluation.selection_telemetry(engine_id, shared_tier);
            tracing::info!(
                event = "image_memory_strategy_selected",
                job_id = %job.id,
                model = %request.model,
                route = engine_id,
                actual_tier = shared_tier,
                strategy = crate::candle_memory_strategy::strategy_label(
                    evaluation.context.selection.strategy
                ),
                basis = evaluation.basis.as_key(),
                predicted_peak_bytes = evaluation.context.predicted_peak_bytes,
                conditioning_peak_bytes = evaluation.phase_peaks.map(|phases| phases.conditioning),
                denoise_peak_bytes = evaluation.phase_peaks.map(|phases| phases.denoise),
                decode_peak_bytes = evaluation.phase_peaks.map(|phases| phases.decode),
                admitted_peak_gb = evaluation.admitted.needed_gb,
                available_gb = evaluation.admitted.available_gb,
                reserve_gb = evaluation.admitted.reserve_gb,
                "shared memory-strategy ladder selected a candle strategy"
            );
            emit_event("image_memory_strategy_selected", telemetry);
        }
        memory_strategy_selection = Some(evaluation.context.selection);
        generation_memory = evaluation.memory;
        if sdxl_request_tier_admission
            && evaluation.context.selection.strategy == gen_core::MemoryStrategy::Resident
        {
            // The measured resident endpoint is literal: disable the provider's legacy automatic
            // decode fallback by carrying an explicit all-false request memory block.
            generation_memory = Some(gen_core::GenerationMemory::default());
        }
        adapted_peak_gb = Some(evaluation.predicted_peak_gb);
        ideogram_warm_staged = evaluation.warm_staged;
        // Resident is the selector's conservative sentinel, not authority to reconfigure a request.
        // In particular, a later legacy low-VRAM decision may choose sequential residency; carrying a
        // Resident scope would then overwrite that request memory back to resident in configure_request.
        selected_memory_strategy_context = if sdxl_request_tier_admission {
            Some(evaluation.context)
        } else {
            optimized_shared_memory_context(engine_id, evaluation.context)
        };
    }
    if let Some(evaluation) = first_pass_shared_memory {
        hires_first_pass_generation_memory = evaluation.memory;
        ideogram_hires_first_warm_staged = evaluation.warm_staged;
        hires_first_pass_memory_context =
            optimized_shared_memory_context(engine_id, evaluation.context);
    }
    if (is_sana || is_sd35)
        && (selected_memory_strategy_context.is_none()
            || (hires_fix.is_some() && hires_first_pass_memory_context.is_none()))
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} requires an exact receipt-priced pre-load memory selection for every render pass",
            request.model
        )));
    }
    // Krea's shared selector runs before any legacy resident/staged gate and owns the final fit
    // decision whenever its revision-bound evidence is available. A `None` result is the explicit
    // unverified path; only then may the established gate remain as provider-safe fallback.
    let shared_krea_fit = (krea_turbo_ladder && adapter_count == 0)
        .then(|| {
            crate::vram_gate::krea_turbo_fit_with_runtime(
                &request.model_manifest_entry,
                tier,
                memory_width,
                memory_height,
                budget,
                shared_reserve_gb,
                krea_allow_streamed_blocks,
                krea_runtime_context.as_ref(),
            )
        })
        .flatten();
    let auto_tier_streaming = auto_tier_streaming_disclosure(
        explicit_pick,
        tier,
        auto_tier_fit_decision(shared_krea_fit.as_ref()),
    );
    if krea_turbo_ladder {
        match shared_krea_fit {
            Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                log_krea_turbo_evidence_exclusion(
                    &request.model,
                    tier,
                    width,
                    height,
                    "final_admission",
                    Some(reason),
                );
            }
            None => log_krea_turbo_evidence_exclusion(
                &request.model,
                tier,
                width,
                height,
                "final_admission",
                None,
            ),
            _ => {}
        }
    }
    // Every receipt-priced family, not just SANA/SD3.5 (sc-20799). `receipt_priced_selection` is
    // Some exactly when the receipt produced a graded selection for this request, including a
    // Resident one — which is a real decision the load must honor, not an absence of one.
    let receipt_selector_authoritative = receipt_priced_selection.is_some();
    // The generic SDXL selector validates the exact artifact, request and composition, including
    // resident scratch-bounding rungs. The descriptor's deliberately broad staging flag and the
    // old geometry-free resident row must not veto that same decision after it passed the tier
    // chooser. Missing or refused selection still reaches the unchanged legacy resident gate.
    let request_selector_authoritative = receipt_selector_authoritative
        || (sdxl_request_tier_admission && memory_strategy_selection.is_some());
    // The plain gate's legacy rows are geometry-blind, but a request the shared selector priced
    // (or that SDXL admission scored against the measured endpoints) can pass at a smaller output;
    // only a request that never reached the selector is truly resolution-blind (issue #3197).
    let plain_gate_lever = ResolutionLever::for_selector(
        sdxl_request_tier_admission || memory_strategy_selection.is_some(),
    );
    let use_sequential = if request_selector_authoritative {
        generation_memory.is_some_and(|memory| memory.stage_residency)
    } else if let Some(
        fit @ crate::vram_gate::KreaTurboFit::Resident {
            peak_gb,
            needed_gb,
            selection,
            ..
        },
    ) = shared_krea_fit
    {
        memory_strategy_selection = Some(selection);
        adapted_peak_gb = Some(peak_gb);
        tracing::info!(
            model = %request.model,
            tier,
            width,
            height,
            needed_gb,
            available_gb = budget.map_or(0.0, |budget| budget.free_gb),
            "shared memory-strategy selector retained Krea Turbo resident execution"
        );
        // sc-22667 (E7): the Krea lane's selection event, in the shared ladder's spelling.
        if let Some(telemetry) = fit.selection_telemetry(tier, memory_width, memory_height) {
            emit_event("image_memory_strategy_selected", telemetry);
        }
        false
    } else {
        // A verified non-resident Krea result bypasses the legacy chooser. Missing or
        // unverified Krea evidence fails closed to resident-or-reject: it must never select a
        // legacy optimized rung.
        let gate_decision = if matches!(
            shared_krea_fit,
            Some(crate::vram_gate::KreaTurboFit::Fits { .. })
                | Some(crate::vram_gate::KreaTurboFit::Reject { .. })
        ) {
            match (needed, budget) {
                (Some(needed_gb), Some(budget)) => crate::vram_gate::FitDecision::Offload {
                    needed_gb,
                    available_gb: budget.free_gb,
                },
                _ => crate::vram_gate::FitDecision::Unknown,
            }
        } else if krea_turbo_ladder {
            krea_unverified_resident_decision(needed, budget)
        } else if sequential_capable
            && memory_strategy_selection.is_some_and(|selection| selection.strategy.is_optimized())
            && generation_memory.is_some_and(|memory| memory.stage_residency)
        {
            // sc-22664: the shared ladder SELECTED an optimized rung that ENGAGES staged
            // residency for this request, priced per rung from the law against the
            // reserve-charged budget, on a provider that supports sequential offload. That
            // decision stands: the legacy resident-vs-free comparison (the padded
            // `vramGbByTier` row) must not re-refuse it, and the Offload arm below already
            // defers its sequential-overflow gate to a shared selection. The figures name the
            // resident peak the ladder moved off, as the Krea Fits arm above does.
            //
            // A staging-FREE optimized selection (a bounded rung whose composition excludes
            // `StagedResidency`) runs whole-model resident, so the resident-vs-free gate
            // below still applies to it; and a provider without `supportsSequentialOffload`
            // cannot run the staged shape the ladder chose, so it keeps its `TooBig` refusal
            // rather than being sent down an Offload path it does not implement.
            match (needed, budget) {
                (Some(needed_gb), Some(budget)) => crate::vram_gate::FitDecision::Offload {
                    needed_gb,
                    available_gb: budget.free_gb,
                },
                _ => crate::vram_gate::FitDecision::Unknown,
            }
        } else {
            crate::vram_gate::resolve_offload(
                crate::vram_gate::fit_decision(needed, budget),
                sequential_capable,
            )
        };
        match gate_decision {
            crate::vram_gate::FitDecision::Offload {
                needed_gb,
                available_gb,
            } => {
                // The resident peak does not fit. Krea Turbo first evaluates its measured three-stage
                // ladder, even when the older two-stage sequential estimate would fit, so a 24 GB card
                // physically drops the DiT before VAE decode. Other models retain the established
                // sequential-overflow gate below; unverified Krea evidence reaches the explicit
                // resident-or-reject path instead of this arm.
                let sequential_needed =
                    crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                        &request.model_manifest_entry,
                        tier,
                        adapter_resident_bytes,
                    );
                let krea_selected = if krea_turbo_ladder {
                    match shared_krea_fit {
                        Some(crate::vram_gate::KreaTurboFit::Resident { .. }) => false,
                        Some(
                            fit @ crate::vram_gate::KreaTurboFit::Fits {
                                phases,
                                needed_gb,
                                selection,
                                memory,
                                // `estimate_scoped` is deliberately not read here (sc-18097):
                                // the selected rung and its knobs are what this lane acts on,
                                // and both are already graded. The flag exists for refusal
                                // ADVICE (`krea_turbo_smaller_fit_*`), which must not name an
                                // estimate-backed geometry.
                                ..
                            },
                        ) => {
                            memory_strategy_selection = Some(selection);
                            generation_memory = Some(memory);
                            // Reclaim accounting records allocations, not the admission threshold.
                            // `needed_gb` includes the ladder's operational reserve
                            // (`vram_gate::ladder_reserve_gb`, sc-22667), which is deliberately
                            // never allocated and therefore cannot be credited back during a
                            // model swap.
                            adapted_peak_gb = Some(phases.peak_gb());
                            tracing::info!(
                                event = "image_memory_strategy_selected",
                                job_id = %job.id,
                                model = %request.model,
                                tier,
                                width,
                                height,
                                strategy = ?selection.strategy,
                                text_peak_gb = phases.text_gb,
                                denoise_peak_gb = phases.denoise_gb,
                                decode_peak_gb = phases.decode_gb,
                                needed_gb,
                                available_gb,
                                "Krea Turbo VRAM fit ladder selected the least-cost sufficient rung"
                            );
                            // sc-22667 (epic 22657 E7): the Krea lane's selection event — the
                            // rung, its basis and the three phase peaks the selector graded, in
                            // the shared ladder's spelling — so an OOM under this selection is
                            // attributable to the exact estimate that admitted it.
                            if let Some(telemetry) =
                                fit.selection_telemetry(tier, memory_width, memory_height)
                            {
                                emit_event("image_memory_strategy_selected", telemetry);
                            }
                            true
                        }
                        Some(crate::vram_gate::KreaTurboFit::Reject { phases, needed_gb }) => {
                            let installed_smaller: Vec<&'static str> = installed_tier_keys(
                                request, settings,
                            )
                            .into_iter()
                            .filter(|candidate| {
                                let candidate_runtime = resolve_tier_dir(
                                    request, settings, candidate,
                                )
                                .and_then(|dir| {
                                    krea_runtime_evidence_context(
                                        request, settings, candidate, &dir,
                                    )
                                });
                                tier_quality_rank(candidate) < tier_quality_rank(tier)
                                    && matches!(
                                        crate::vram_gate::krea_turbo_fit_with_runtime(
                                            &request.model_manifest_entry,
                                            candidate,
                                            memory_width,
                                            memory_height,
                                            budget,
                                            shared_reserve_gb,
                                            krea_allow_streamed_blocks,
                                            candidate_runtime.as_ref(),
                                        ),
                                        Some(
                                            crate::vram_gate::KreaTurboFit::Resident { .. }
                                                | crate::vram_gate::KreaTurboFit::Fits { .. }
                                        )
                                    )
                            })
                            .collect();
                            let lower_resolution =
                                crate::vram_gate::krea_turbo_smaller_fit_with_runtime(
                                    &request.model_manifest_entry,
                                    tier,
                                    memory_width,
                                    memory_height,
                                    budget,
                                    shared_reserve_gb,
                                    krea_allow_streamed_blocks,
                                    krea_runtime_context.as_ref(),
                                );
                            let mut options = Vec::new();
                            if let Some((smaller_w, smaller_h)) = lower_resolution {
                                options.push(format!(
                                        "lower the output resolution to {smaller_w}x{smaller_h} or below"
                                    ));
                            }
                            if !installed_smaller.is_empty() {
                                options.push(format!(
                                    "select a smaller installed tier ({})",
                                    installed_smaller
                                        .iter()
                                        .map(|candidate| candidate.to_uppercase())
                                        .collect::<Vec<_>>()
                                        .join(" / ")
                                ));
                            }
                            options.push("run on a GPU with more VRAM".to_owned());
                            return Err(WorkerError::InvalidPayload(format!(
                                    "{model} at {width}x{height} on the {tier} tier needs ~{needed} GB \
                                     of VRAM even with three-stage loading, tiled VAE decode, attention \
                                     chunking, and {streaming}, but GPU {gpu} has ~{available} GB \
                                     available. Measured phase predictions: text {text:.1} GB, denoise \
                                     {denoise:.1} GB, decode {decode:.1} GB. {options}.",
                                    model = request.model,
                                    needed = needed_gb.round() as i64,
                                    available = available_gb.round() as i64,
                                    gpu = settings.gpu_id,
                                    streaming = if krea_allow_streamed_blocks {
                                        "transformer block streaming"
                                    } else {
                                        "the deepest adapter-compatible rung"
                                    },
                                    text = phases.text_gb,
                                    denoise = phases.denoise_gb,
                                    decode = phases.decode_gb,
                                    options = options.join(", or "),
                                )));
                        }
                        Some(crate::vram_gate::KreaTurboFit::Unverified { reason }) => {
                            log_krea_turbo_evidence_exclusion(
                                &request.model,
                                tier,
                                width,
                                height,
                                "optimized_rung",
                                Some(reason),
                            );
                            false
                        }
                        None => {
                            log_krea_turbo_evidence_exclusion(
                                &request.model,
                                tier,
                                width,
                                height,
                                "optimized_rung",
                                None,
                            );
                            false
                        }
                    }
                } else {
                    false
                };
                let shared_memory_selected = memory_strategy_selection
                    .is_some_and(|selection| selection.strategy.is_optimized());
                if rejects_unverified_shared_memory_fallback(engine_id, shared_memory_selected) {
                    let family = verified_only_memory_family_label(engine_id)
                        .expect("verified-only rejection is restricted to a named family");
                    return Err(WorkerError::InvalidPayload(format!(
                        "{model} at the {tier} tier needs ~{needed} GB of resident VRAM, but GPU \
                         {gpu} has ~{available} GB available and no exact verified {family} memory \
                         strategy fits this request. Install or verify matching calibration evidence, \
                         select a smaller tier or resolution, or use a GPU with more VRAM.",
                        model = request.model,
                        needed = needed_gb.round() as i64,
                        available = available_gb.round() as i64,
                        gpu = settings.gpu_id,
                        family = family,
                    )));
                }
                if !krea_selected && !shared_memory_selected {
                    if let Some(seq_gb) =
                        crate::vram_gate::sequential_overflow_gb(sequential_needed, budget)
                    {
                        let installed_smaller: Vec<&'static str> =
                            installed_tier_keys(request, settings)
                                .into_iter()
                                .filter(|candidate| {
                                    tier_quality_rank(candidate) < tier_quality_rank(tier)
                                })
                                .collect();
                        return Err(WorkerError::InvalidPayload(format!(
                            "{model} at the {tier} tier needs ~{seq} GB of VRAM even with sequential \
                             component residency (loading one component at a time), but GPU {gpu} has \
                             ~{available} GB available. {tail}",
                            model = request.model,
                            seq = seq_gb.round() as i64,
                            available = available_gb.round() as i64,
                            gpu = settings.gpu_id,
                            tail = vram_reject_tail(&installed_smaller, plain_gate_lever),
                        )));
                    }
                }
                tracing::info!(
                    model = %request.model,
                    needed_gb = needed_gb.round() as i64,
                    available_gb = available_gb.round() as i64,
                    "candle VRAM fit-gate: resident peak exceeds free VRAM — loading with sequential \
                     component residency (text encoders dropped before the DiT)"
                );
                true
            }
            crate::vram_gate::FitDecision::TooBig {
                needed_gb,
                available_gb,
            } => {
                let installed_smaller: Vec<&'static str> = installed_tier_keys(request, settings)
                    .into_iter()
                    .filter(|candidate| {
                        let candidate_runtime = resolve_tier_dir(request, settings, candidate)
                            .and_then(|dir| {
                                krea_runtime_evidence_context(request, settings, candidate, &dir)
                            });
                        tier_quality_rank(candidate) < tier_quality_rank(tier)
                            && (!krea_turbo_ladder
                                || matches!(
                                    crate::vram_gate::krea_turbo_fit_with_runtime(
                                        &request.model_manifest_entry,
                                        candidate,
                                        memory_width,
                                        memory_height,
                                        budget,
                                        shared_reserve_gb,
                                        krea_allow_streamed_blocks,
                                        candidate_runtime.as_ref(),
                                    ),
                                    Some(
                                        crate::vram_gate::KreaTurboFit::Resident { .. }
                                            | crate::vram_gate::KreaTurboFit::Fits { .. }
                                    )
                                ))
                    })
                    .collect();
                return Err(WorkerError::InvalidPayload(format!(
                    "{model} at the {tier} tier needs ~{needed} GB of VRAM (with headroom) but GPU \
                     {gpu} has ~{available} GB available. {tail}",
                    model = request.model,
                    needed = needed_gb.round() as i64,
                    available = available_gb.round() as i64,
                    gpu = settings.gpu_id,
                    tail = vram_reject_tail(&installed_smaller, plain_gate_lever),
                )));
            }
            _ => false,
        }
    };
    // sc-11023: record this admitted load's incurred peak as the reclaimable high-water for the NEXT
    // gate. Sequential residency peaks at the largest single component; a resident load at the whole-
    // model peak. cudarc's pool never returns pages to the driver, so the max we have ever loaded is
    // exactly what a later swap-in reclaims after the single-slot cache evicts this model. Reached only
    // when the gate ADMITTED the load (the reject arms `return` above), so we never record a peak we
    // didn't actually attempt to allocate.
    let incurred_peak = if sdxl_request_tier_admission && memory_strategy_selection.is_some() {
        // An endpoint-priced resident/scratch-bound run must not credit the old padded resident
        // manifest row during the next eviction. Only the admitted raw working set is reclaimable.
        adapted_peak_gb
    } else if use_sequential {
        adapted_peak_gb.or_else(|| {
            crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                &request.model_manifest_entry,
                tier,
                adapter_resident_bytes,
            )
        })
    } else {
        needed
    };
    if let Some(peak_gb) = incurred_peak {
        crate::vram_gate::note_loaded_peak(&settings.gpu_id, peak_gb);
    }
    let mut memory_strategy_context = selected_memory_strategy_context.or_else(|| {
        memory_strategy_selection.and_then(|selection| {
            let budget = budget?;
            let predicted_peak_gb = adapted_peak_gb?;
            let turbo_fit = request
                .model_manifest_entry
                .get("candle")?
                .get("turboFit")?;
            let calibration_abi = u32::try_from(turbo_fit.get("calibrationAbi")?.as_u64()?).ok()?;
            let calibration_fingerprint = turbo_fit
                .get("calibrationFingerprint")?
                .as_str()?
                .to_owned();
            let gb_to_bytes = |gb: f64| {
                (gb * 1024.0 * 1024.0 * 1024.0)
                    .round()
                    .clamp(0.0, u64::MAX as f64) as u64
            };
            tracing::info!(
                backend = "candle",
                evidence_revision = krea_evidence_revision(),
                reclaimable_gb,
                raw_available_gb = raw_budget.map_or(0.0, |raw| raw.free_gb),
                effective_available_gb = budget.free_gb,
                requested_tier,
                effective_tier = tier,
                strategy = ?selection.strategy,
                "shared memory-strategy selection admitted"
            );
            // sc-17097: the shape now comes from the manifest, which states what the curves were MEASURED
            // under. Reading it back off the provider - as this did - made the safety check compare the
            // provider against itself, so calibration ABI 2's load-shape axis could never fail here.
            let load_shape = crate::vram_gate::krea_turbo_load_shape(turbo_fit)?;
            Some(gen_core::MemoryRunContext {
                selection,
                optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
                calibration_abi,
                calibration_fingerprint,
                load_shape,
                mode: gen_core::MemoryMode::TextToImage,
                has_reference: reference_count > 0,
                use_pid: false,
                has_phases: false,
                geometry: gen_core::MemoryGeometry {
                    width: memory_width,
                    height: memory_height,
                    batch: 1,
                    frames: 1,
                    reference_count,
                },
                reference_resources: None,
                overlay: None,
                budget: gen_core::MemoryBudget {
                    total_bytes: gb_to_bytes(budget.total_gb),
                    committed_bytes: gb_to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: gb_to_bytes(2.0),
                },
                predicted_peak_bytes: gb_to_bytes(predicted_peak_gb),
                cache_state: if reclaimable_gb > 0.0 {
                    gen_core::MemoryCacheState::Warm
                } else {
                    gen_core::MemoryCacheState::Cold
                },
                evidence_revision: krea_evidence_revision(),
            })
        })
    });
    if !receipt_selector_authoritative {
        apply_request_scoped_candle_residency(use_sequential, &mut generation_memory);
    }
    // Reuse the exact selector spec, including Mage's split text-encoder/VAE component paths. Only
    // the post-selection residency policy and optional ConvRot substitution may differ below.
    let mut spec = shared_contract_spec;
    spec = apply_declared_candle_image_load_shape(
        engine_id,
        Some(tier),
        shared_admission_mode,
        &request.model_manifest_entry,
        spec,
        use_sequential,
    );
    if use_sequential {
        // Ask the provider (candle FLUX) to load→use→drop each component in phase order (sc-10821).
        spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
    }
    // Surface the decision before model execution, while the reason for a slow render is still clear,
    // then keep the same compact note on subsequent progress updates. The structured event is the
    // statistics trace; the tracing event remains useful in the worker log. Neither changes admission
    // or tier ordering, and neither creates a dialog/toast UI (Michael's sc-16104 constraint).
    if let Some(disclosure) = auto_tier_streaming.as_ref() {
        raw_settings.insert(
            AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
            disclosure.record(),
        );
        tracing::info!(
            job_id = %job.id,
            model = %request.model,
            tier = %disclosure.tier,
            strategy = "bounded_transformer_residency",
            cause = "streaming_transformer_blocks_to_hold_auto_selected_tier",
            measured_materialization_ms_per_block = ?disclosure.measured_materialization_ms_per_block,
            measured_materialization_ms_per_step = ?disclosure.measured_materialization_ms_per_step,
            measurement_source = "sc-16096-rtx-pro-6000-blackwell",
            "streaming transformer blocks to hold the auto-selected tier"
        );
        emit_event(
            "image_memory_strategy_selected",
            disclosure.telemetry(&job.id, &request.model),
        );
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::LoadingModel,
                ProgressStage::LoadingModel,
                0.0,
                &disclosure.progress_message("Preparing render."),
                Some(streaming_result(plan, asset_writes)),
                backend,
            ),
        )
        .await?;
    }

    // The registered Z-Image identity route retains the shared, non-fatal likeness post-pass from the
    // retired bespoke stream. This stages only the existing face stack, not an obsolete Z-Image path.
    let likeness_source = resolve_character_image_likeness_source(request, settings, project_path);
    let face_stack_dir = stage_likeness(
        api,
        settings,
        job,
        likeness_source.is_some(),
        "registered Candle character_image face-stack staging failed; likeness scores omitted",
    )
    .await;
    let likeness_source = face_stack_dir.as_ref().and(likeness_source);

    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("candle {engine_id} load failed"),
        load_generator,
        move |generator,
              cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            if is_ideogram || is_sana || is_sd35 || receipt_selector_authoritative {
                // Receipt-priced providers require the exact pre-load selector result to reach the
                // cached generator. Ideogram may preserve a tighter pre-admitted staged sibling;
                // SANA's cache key must exactly retain the selector's own resident/staged policy.
                //
                // Ideogram/SANA/SD3.5 additionally REQUIRE a pre-load context: for them an absent
                // one is a bug, not a fallback. Chroma and Kolors reach here only when the receipt
                // actually priced the request (`receipt_selector_authoritative`); an uncertified
                // import legitimately produced no receipt and keeps the historical resident path
                // through the `else` arm below.
                if is_ideogram {
                    let context = memory_strategy_context.as_mut().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    crate::candle_memory_strategy::bind_ideogram_cache_execution(
                        context,
                        &mut generation_memory,
                        &mut ideogram_warm_staged,
                        cache_state,
                        loaded_policy.offload_policy,
                    )?;
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        crate::candle_memory_strategy::bind_ideogram_cache_execution(
                            first_pass,
                            &mut hires_first_pass_generation_memory,
                            &mut ideogram_hires_first_warm_staged,
                            cache_state,
                            loaded_policy.offload_policy,
                        )?;
                    }
                } else if is_sana || is_sd35 {
                    let context = memory_strategy_context.as_mut().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    context.cache_state = cache_state;
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        first_pass.cache_state = cache_state;
                    }
                } else {
                    // Chroma/Kolors: `optimized_shared_memory_context` drops a Resident context,
                    // so `None` here means the receipt selected Resident, not that it was silent.
                    if let Some(context) = memory_strategy_context.as_mut() {
                        context.cache_state = cache_state;
                    }
                    if let Some(first_pass) = hires_first_pass_memory_context.as_mut() {
                        first_pass.cache_state = cache_state;
                    }
                }
                // sc-20799: the policy the receipt admitted must EQUAL the policy that loaded, for
                // every receipt-priced family — not only SANA and SD3.5. An admitted staged peak
                // does not bound a resident load, and a warm cache entry can present either shape.
                // Evaluated AFTER the Ideogram warm-staged rebinding above, so the comparison is
                // against the strategy this request will actually execute.
                if let Some(selected_strategy) = receipt_priced_selection {
                    let effective_strategy = memory_strategy_context
                        .as_ref()
                        .map(|context| context.selection.strategy)
                        .unwrap_or(selected_strategy);
                    let selected_staged = effective_strategy
                        == gen_core::MemoryStrategy::StagedResidency
                        || generator
                            .memory_strategy_contract()
                            .is_some_and(|contract| {
                                contract.engages(
                                    effective_strategy,
                                    gen_core::MemoryStrategy::StagedResidency,
                                )
                            });
                    let loaded_staged =
                        loaded_policy.offload_policy == gen_core::OffloadPolicy::Sequential;
                    if selected_staged != loaded_staged {
                        return Err(WorkerError::InvalidPayload(format!(
                            "{engine_id} warm cache policy crossed its exact pre-load selection"
                        )));
                    }
                }
                let stages = generation_memory.is_some_and(|memory| memory.stage_residency);
                warm_policy.settle_with_selection(if stages {
                    crate::execution_planner::GrantOutcome::AlreadyStaged
                } else {
                    crate::execution_planner::GrantOutcome::NoStagedCandidate
                });
                let cache_state_label = match cache_state {
                    gen_core::MemoryCacheState::Cold => "cold",
                    gen_core::MemoryCacheState::Warm => "warm",
                };
                let contract = generator.memory_strategy_contract().ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "{engine_id} loaded provider omitted its sealed memory contract"
                    ))
                })?;
                let facts = contract.asset_facts;
                // Only these three families expose a singular canonical physical receipt to name in
                // the disclosure. Chroma and Kolors seal their identity as an ordered overlay
                // receipt instead, already bound into the selection above; they take the policy
                // guard without minting a second, weaker receipt here.
                if is_ideogram || is_sana || is_sd35 {
                    let physical_receipt = if is_sana {
                        crate::candle_memory_strategy::sana_physical_receipt_identity(
                            engine_id, contract,
                        )?
                    } else if is_sd35 {
                        crate::candle_memory_strategy::sd35_physical_receipt_identity(
                            engine_id, contract,
                        )?
                    } else {
                        crate::candle_memory_strategy::ideogram_physical_receipt_identity(contract)?
                    };
                    let context = memory_strategy_context.as_ref().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} cache access is missing its exact pre-load memory context"
                        ))
                    })?;
                    let passes = match hires_first_pass_memory_context.as_ref() {
                        Some(first) => vec![("hires_first", first), ("hires_final", context)],
                        None => vec![("single", context)],
                    };
                    for (pass_identity, pass_context) in passes {
                        let strategy_label = match pass_context.selection.strategy {
                            gen_core::MemoryStrategy::Resident => "resident",
                            gen_core::MemoryStrategy::StagedResidency => "staged_residency",
                            gen_core::MemoryStrategy::BoundedDecode => "bounded_decode",
                            gen_core::MemoryStrategy::BoundedAttention => "bounded_attention",
                            gen_core::MemoryStrategy::BoundedTransformerResidency => {
                                "bounded_transformer_residency"
                            }
                        };
                        let authority_label = match pass_context.optimization_authority {
                            gen_core::MemoryOptimizationAuthority::Resident => "resident",
                            gen_core::MemoryOptimizationAuthority::Estimated => "estimated",
                            gen_core::MemoryOptimizationAuthority::Calibrated => "calibrated",
                        };
                        let carrier =
                            match (&pass_context.mode, pass_context.geometry.reference_count) {
                                (_, 0) => "prompt_only",
                                (gen_core::MemoryMode::Edit, 2) => "reference_mask",
                                _ => "reference",
                            };
                        tracing::info!(
                            event = "image_memory_strategy_selected",
                            route = engine_id,
                            actual_tier = tier,
                            pass = pass_identity,
                            mode = pass_context.mode.as_key(),
                            carrier,
                            reference_count = pass_context.geometry.reference_count,
                            use_pid = pass_context.use_pid,
                            overlay = pass_context.overlay.as_deref().unwrap_or("none"),
                            physical_receipt = physical_receipt.as_str(),
                            strategy = strategy_label,
                            authority = authority_label,
                            cache_state = cache_state_label,
                            predicted_peak_bytes = pass_context.predicted_peak_bytes,
                            base_bytes = facts.base_bytes,
                            conditioning_bytes = facts.conditioning_bytes,
                            transformer_bytes = facts.transformer_bytes,
                            decoder_bytes = facts.decoder_bytes,
                            overlay_bytes = facts.overlay_bytes,
                            external_committed_bytes,
                            evidence_revision = %pass_context.evidence_revision,
                            "exact receipt-backed image memory strategy selected"
                        );
                        emit_event(
                            "image_memory_strategy_selected",
                            serde_json::json!({
                                "backend": "candle",
                                "route": engine_id,
                                "actualTier": tier,
                                "passIdentity": pass_identity,
                                "mode": pass_context.mode.as_key(),
                                "carrier": carrier,
                                "referenceCount": pass_context.geometry.reference_count,
                                "geometry": {
                                    "width": pass_context.geometry.width,
                                    "height": pass_context.geometry.height,
                                    "batch": pass_context.geometry.batch,
                                    "frames": pass_context.geometry.frames,
                                },
                                "usePid": pass_context.use_pid,
                                "overlay": pass_context.overlay.clone(),
                                "physicalReceipt": physical_receipt.clone(),
                                "strategy": strategy_label,
                                "authority": authority_label,
                                "cacheState": cache_state_label,
                                "predictedPeakBytes": pass_context.predicted_peak_bytes,
                                "physicalFloor": {
                                    "baseBytes": facts.base_bytes,
                                    "conditioningBytes": facts.conditioning_bytes,
                                    "transformerBytes": facts.transformer_bytes,
                                    "decoderBytes": facts.decoder_bytes,
                                    "overlayBytes": facts.overlay_bytes,
                                },
                                "evidenceRevision": pass_context.evidence_revision.clone(),
                            }),
                        );
                    }
                }
            } else {
                warm_policy.decline(
                    crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory,
                );
            }
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some((source, _))) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            let likeness_source_ref = likeness_source.as_ref().map(|(_, id)| id.clone());
            // A native prompt batch (sc-25681) is ONE engine call over every image of the job,
            // admitted against its `prompts × count` batch (`batch_overlay_bytes` above).
            if let Some(batch) = native_batch.as_ref() {
                let seed = work[0].0;
                return drive_native_prompt_batch(
                    tx,
                    seed,
                    batch.images() as usize,
                    |preview, on_progress| {
                        let warnings = std::cell::RefCell::new(Vec::new());
                        let prefix_kv_cache_report = std::cell::RefCell::new(None);
                        let images = render_on_surface(
                            generator,
                            "",
                            width,
                            height,
                            seed,
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
                            generation_memory,
                            memory_strategy_context.as_ref(),
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
                                prefix_kv_cache_plan,
                                prefix_kv_cache_report.take().as_ref(),
                            ),
                        );
                        Ok((images, facts))
                    },
                );
            }
            drive_gen_items_scored_reported(
                tx,
                work,
                move |_index, (seed, prompt), preview, prompt_enhancement, on_progress| {
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
                            // Ideogram edit source (edit_image) OR the Krea 2 Turbo img2img init (sc-10134) —
                            // mutually exclusive by mode/family; whichever resolved seeds the single
                            // `Conditioning::Reference` slot.
                            edit_reference.as_ref().or(img2img_reference.as_ref()),
                            &edit_refs,
                            edit_mask.as_ref(),
                            true_cfg,
                            // Per-payload sampler / scheduler / schedule-shift (sc-7127), already N3-guarded
                            // against this family's advertised surface above. RealVisXL Lightning forces
                            // `lightning`; most candle families advertise only their default until P4, so an
                            // unsupported request was dropped to `None` (the engine default) before reaching here.
                            sampler.as_deref(),
                            scheduler.as_deref(),
                            scheduler_shift,
                            // Guidance method, N3-guarded against this family's advertised surface above
                            // (sc-7448). candle adopts cfg_pp/cfg_rescale/apg in P4; until then an unsupported
                            // method was already dropped to `None` (the engine default) before reaching here.
                            guidance_method.as_deref(),
                            // Per-generation PiD decode (epic 7840): route the final latent through the
                            // `spec.pid` super-resolving student when resolved (opt-in + snapshots cached),
                            // else the native VAE. Every candle image provider reads `spec.pid` (sc-7853), so
                            // the whole off-Mac catalog honors the toggle in lockstep with `spec.pid` above.
                            use_pid,
                            text_style_gain,
                            generation_memory,
                            hires_first_pass_generation_memory,
                            memory_strategy_context.as_ref(),
                            hires_first_pass_memory_context.as_ref(),
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
                    // Ideogram 4 placeholder detect-and-reseed (sc-6858, parity with the macOS
                    // `generate_stream` net, sc-6501): the caption guard above makes it rare, but a residual
                    // "Image blocked by safety filter" placeholder can still occur even with a caption.
                    // Detect via the baked-text heuristic and reseed transparently, keeping the first clean
                    // render. Gated to Ideogram 4; a no-op for every other candle family, for turbo (CFG-free,
                    // cannot produce it), and for an edit (the output is anchored to a real source latent, so
                    // `looks_like_placeholder` returns false).
                    let initial = render(seed, on_progress)?;
                    let (final_seed, out_w, out_h, pixels) = recover_ideogram_placeholder(
                        is_ideogram,
                        seed,
                        &cancel,
                        initial,
                        |retry_seed| render(retry_seed, on_progress),
                    )?;
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
                            prefix_kv_cache_plan,
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

    consume_gen_events_with_disclosure(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        auto_tier_streaming.as_ref(),
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Whether the Candle capability downtier must keep the resolved tier: an explicit pick
/// (`mlxQuantizeExplicit`, or any `mlxQuantize` on the FLUX.1 packed turnkeys), or a portable job
/// whose verified binding froze a tier. The latter is served in place from the hub snapshot, so a
/// downtier would load a sibling tier the guard never verified; such a job is gated as bound
/// (fits, or is refused before load).
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(crate) fn candle_tier_pick_is_exact(model: &str, advanced: &crate::JsonObject) -> bool {
    (matches!(model, "flux_schnell" | "flux_dev") && advanced.contains_key("mlxQuantize"))
        || advanced
            .get("mlxQuantizeExplicit")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        || sceneworks_core::payload_util::portable_bound_tier_bits(advanced).is_some()
}
