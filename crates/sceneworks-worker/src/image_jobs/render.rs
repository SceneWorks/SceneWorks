use super::conditioning::build_ordered_reference_conditioning;
use super::flux3_references;
use super::geometry::fit_engine_image;
use super::lane_controls;
use super::resolve_family;
use super::CancelFlag;
use super::Conditioning;
use super::GenerationOutput;
use super::GenerationRequest;
use super::Generator;
use super::Image;
use super::ImageRequest;
use super::Progress;
use super::PromptEnhance;
use crate::WorkerError;
use crate::WorkerResult;
/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img. `enhance` carries the optional
/// caption-upsampling settings (sc-6135; only FLUX.2-dev acts on them).
///
/// Production renders go through [`generate_one_on_surface`]; this RGB-surface form is the one the
/// engine smokes drive.
#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    // The guidance method (epic 7434 P5, sc-7448): `None` is the engine default (N1 no-op); a value
    // is already N3-normalized against the descriptor's `supported_guidance_methods` by the caller.
    guidance_method: Option<&str>,
    // Per-generation PiD super-resolving decode (epic 7840, sc-7849). Must be `true` only when the
    // generator was loaded with `LoadSpec::with_pid` (the engine rejects a mismatch); the caller keeps
    // the two in lockstep. The candle path passes `false` (candle PiD is Phase 4, sc-7853).
    use_pid: bool,
    // Krea "text style" tap-reweight gain (sc-11878) — `None` for every non-Krea family (the engine
    // ignores the field regardless). The caller resolves it from the manifest `textStyleGain` slider +
    // `advanced` only when the model declares the control, so a non-Krea render passes `None`.
    text_style_gain: Option<f32>,
    // Quality-preserving, request-scoped memory adaptations selected by the candle Krea Turbo fit
    // ladder. `None` is the historical path for every other provider and unconstrained Krea jobs.
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    // Live denoise preview (epic 16624, sc-16904): forwarded frames reach the job's progress
    // stream. Inert for engines that don't emit; the default sink costs one branch per step.
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    generate_one_on_surface(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt,
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        memory,
        memory_strategy_context,
        enhance,
        prompt_enhancement,
        preview,
        &LaneRgbaSurface::default(),
        cancel,
        on_progress,
    )
}

/// The four-channel surface of one generic-lane render (sc-24111 / sc-24113 S4 contract).
///
/// `Default` is the historical three-channel lane, byte-identical: no reference carries alpha and
/// the request asks for `OutputChannels::Rgb`. Only a model whose descriptor advertises
/// `supports_alpha_output` / `ReferenceRgba` (Qwen-Image 2.1) ever gets a non-default value.
#[derive(Default)]
pub(crate) struct LaneRgbaSurface<'a> {
    /// The alpha plane of each `multi_references` entry, positionally; `None` (or a missing entry)
    /// is an ordinary RGB reference. An entry that carries one is sent as
    /// `Conditioning::ReferenceRgba`, UN-flattened, because the VAE encodes all four channels.
    pub(crate) multi_reference_alpha: &'a [Option<image::GrayImage>],
    /// The one trusted FLUX.3 owned-asset resolution used for admission and rendering. Its
    /// original RGB/RGBA carriers and optional size intent must not pass through img2img slots.
    pub(crate) flux3_references: Option<&'a flux3_references::ResolvedReferences>,
    /// Caption-only box controls, validated against those same owned reference slots before load.
    pub(crate) flux3_layout: Option<&'a flux3_references::ResolvedLayout>,
    /// `GenerationRequest::output_channels`. `Rgba` makes the provider answer with
    /// `GenerationOutput::ImagesRgba`, whose four channels the egress types as an RGBA PNG.
    pub(crate) output_channels: gen_core::OutputChannels,
    /// The request controls beyond the positional knobs (sc-25681): CFG interval and
    /// caption-overflow policy, resolved and refused by [`lane_controls`]. `Default` = the
    /// engine's own for every model that declares none.
    pub(crate) controls: lane_controls::LaneRequestControls,
    /// Where the render's non-fatal engine warnings (`GenerationReport::warnings`, e.g. Iris's
    /// `caption_truncated`) are collected for the image's provenance. `None` discards them, as
    /// the lane always has for providers that report none.
    pub(crate) warnings: Option<&'a std::cell::RefCell<Vec<gen_core::GenerationWarning>>>,
    /// Where the render's `GenerationReport::prefix_kv_cache` — the prefix-KV-cache path that
    /// actually RAN, its cache bytes and any shed-over-priced-bound overflow (epic 25899) — is
    /// collected for the image's provenance and the job record. `None` discards it; a provider
    /// with no cache reports none.
    pub(crate) prefix_kv_cache:
        Option<&'a std::cell::RefCell<Option<gen_core::PrefixKvCacheReport>>>,
    /// A native prompt batch (sc-25681): the request carries `prompt_batch` + `count` instead of
    /// one prompt, and the engine returns every image of the batch. `None` everywhere else.
    pub(crate) prompt_batch: Option<&'a lane_controls::NativePromptBatch>,
    /// The exact owner-local plan compiled before the generator was loaded. Only base FLUX.1
    /// reaches this seam during ENG-1a; other routes retain their existing generator call.
    pub(crate) render_plan: Option<&'a gen_core::RenderPlan>,
    /// The pre-load plan of the Hires.fix refinement pass, compiled together with `render_plan` as
    /// one multi-pass plan whose second pass consumes the first pass's stretched output.
    pub(crate) hires_render_plan: Option<&'a gen_core::RenderPlan>,
}

/// [`generate_one`] with an explicit [`LaneRgbaSurface`].
#[allow(clippy::too_many_arguments)]
fn generate_one_on_surface(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    preview: gen_core::PreviewSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    render_on_surface(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt,
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        memory,
        memory_strategy_context,
        enhance,
        prompt_enhancement,
        preview,
        surface,
        cancel,
        on_progress,
    )?
    .pop()
    .ok_or_else(|| WorkerError::Engine("generator produced no image".to_owned()))
}

/// The one generic-lane engine call: builds the request (one `prompt`, or the surface's native
/// `prompt_batch` + `count`) and returns EVERY image the engine produced, in engine order.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_on_surface(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    preview: gen_core::PreviewSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<Vec<(u32, u32, Vec<u8>)>> {
    let mut request = build_render_request(
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt,
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        memory,
        enhance,
        prompt_enhancement,
        preview,
        surface,
        cancel,
    )?;
    // `generate_with_report` (sc-25681): the default implementation is exactly `generate`, so every
    // provider that reports nothing is unaffected; one that reports (Iris's `caption_truncated`)
    // has its warnings collected for the image's provenance.
    let report = if let Some(references) = surface.flux3_references {
        flux3_references::render_reference_request(
            generator,
            &mut request,
            references,
            memory_strategy_context,
            on_progress,
        )
    } else {
        match surface.render_plan {
            Some(plan) => crate::memory_strategy::generate_with_scope_report_planned(
                generator,
                &mut request,
                plan,
                memory_strategy_context,
                on_progress,
            ),
            None => crate::memory_strategy::generate_with_scope_report(
                generator,
                &mut request,
                memory_strategy_context,
                on_progress,
            ),
        }
    }
    .map_err(|error| match error {
        gen_core::Error::ProcessPoisoned(_) => {
            crate::classify_engine_error("generation failed", error)
        }
        other => WorkerError::Engine(format!("generation failed: {other}")),
    })?;
    if let Some(sink) = surface.warnings {
        sink.borrow_mut().extend(report.warnings);
    }
    if let Some(sink) = surface.prefix_kv_cache {
        *sink.borrow_mut() = report.prefix_kv_cache;
    }
    let output = report
        .output
        .ok_or_else(|| WorkerError::Engine("generator produced no output".to_owned()))?;
    match output {
        GenerationOutput::Images(images) => Ok(images
            .into_iter()
            .map(|image| (image.width, image.height, image.pixels))
            .collect()),
        // Only ever produced for an `OutputChannels::Rgba` request (the shared request floor refuses
        // one against a provider without `supports_alpha_output`). The flat interleaved RGBA buffer
        // flows on unchanged: `GeneratedPixels::from_engine_buffer` types it by its channel count.
        GenerationOutput::ImagesRgba(images) => Ok(images
            .into_iter()
            .map(|image| (image.width, image.height, image.pixels))
            .collect()),
        _ => Err(WorkerError::Engine(
            "generator returned non-image output".to_owned(),
        )),
    }
}

/// Build the exact request sent by the generic image lane. The pre-load FLUX.1 compiler and the
/// post-load render both call this function, so a plan cannot accidentally describe different
/// prompt, geometry, count, schedule or conditioning inputs from execution.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_render_request(
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    enhance: &PromptEnhance,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    preview: gen_core::PreviewSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
) -> WorkerResult<GenerationRequest> {
    let conditioning = if surface.flux3_references.is_some() {
        if reference.is_some() || !multi_references.is_empty() || edit_mask.is_some() {
            return Err(WorkerError::InvalidPayload(
                "FLUX.3 references must use the owned ordered conditioning list, not img2img or mask slots".to_owned(),
            ));
        }
        // The owned render seam below installs these carriers only after checking the admitted facts.
        Vec::new()
    } else {
        build_lane_conditioning_with_alpha(
            reference,
            multi_references,
            surface.multi_reference_alpha,
            edit_mask,
        )
    };
    // A native prompt batch is the prompt list ALONE (the engine refuses `prompt` beside it).
    let (prompt, prompt_batch, count) = match surface.prompt_batch {
        Some(batch) if batch.prompts.len() == 1 => {
            (batch.prompts[0].clone(), Vec::new(), batch.count)
        }
        Some(batch) => (String::new(), batch.prompts.clone(), batch.count),
        None => (prompt.to_owned(), Vec::new(), 1),
    };
    let mut request = GenerationRequest {
        prompt,
        prompt_batch,
        negative_prompt,
        width,
        height,
        count,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg,
        sampler: sampler.map(str::to_owned),
        scheduler: scheduler.map(str::to_owned),
        scheduler_shift,
        guidance_method: guidance_method.map(str::to_owned),
        use_pid,
        text_style_gain,
        memory,
        conditioning,
        flux3_reference_output: None,
        output_channels: surface.output_channels,
        cfg_interval: surface.controls.cfg_interval,
        caption_overflow: surface.controls.caption_overflow,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request, prompt_enhancement);
    if let Some(layout) = surface.flux3_layout {
        layout.apply(&mut request);
    }
    Ok(request)
}

/// The FLUX.1 provider ids whose every request — base text-to-image, XLabs IP-Adapter, ControlNet
/// and PuLID — renders from a render plan compiled before load (SW-6933 / SW-6936).
pub(super) const FLUX1_PRELOAD_PLANNED_ENGINES: &[&str] = &[
    "flux1_schnell",
    "flux1_dev",
    "flux1_dev_control",
    "pulid_flux",
];

/// Map a provider's pre-load plan refusal onto the worker's error classes: an invalid request is a
/// payload error the user can act on, cancellation stays cancellation, and a poisoned process keeps its
/// typed recovery class.
pub(super) fn preload_plan_error(route: &str, error: gen_core::Error) -> WorkerError {
    match error {
        gen_core::Error::ProcessPoisoned(_) => {
            crate::classify_engine_error("render-plan compilation failed", error)
        }
        gen_core::Error::Canceled => WorkerError::Canceled(format!(
            "{route}: generation canceled while compiling its render plan"
        )),
        other => WorkerError::InvalidPayload(format!(
            "{route}: cannot compile render plan before load: {other}"
        )),
    }
}

/// Resolve a provider's owner-local plan before `start_cached_gen_stream` can load weights.
/// The registry returns `None` for routes that have not migrated yet; an invalid request is a
/// pre-load error rather than a fallback to the legacy call.
pub(super) fn compile_render_plan(
    registry: &gen_core::ProviderRegistry,
    engine_id: &str,
    spec: &gen_core::LoadSpec,
    request: &GenerationRequest,
) -> WorkerResult<Option<gen_core::RenderPlan>> {
    let plan = registry
        .compile_render_plan(engine_id, spec, request)
        .map_err(|error| preload_plan_error(engine_id, error))?;
    if plan.is_none() && FLUX1_PRELOAD_PLANNED_ENGINES.contains(&engine_id) {
        return Err(WorkerError::Engine(format!(
            "{engine_id}: route has no registered pre-load render planner"
        )));
    }
    Ok(plan)
}

/// [`compile_render_plan`] for a route that only renders from a pre-load plan (the MLX FLUX.1
/// ControlNet and PuLID lanes).
#[cfg(target_os = "macos")]
pub(super) fn compile_required_render_plan(
    registry: &gen_core::ProviderRegistry,
    engine_id: &str,
    spec: &gen_core::LoadSpec,
    request: &GenerationRequest,
) -> WorkerResult<gen_core::RenderPlan> {
    compile_render_plan(registry, engine_id, spec, request)?.ok_or_else(|| {
        WorkerError::Engine(format!(
            "{engine_id}: route has no registered pre-load render planner"
        ))
    })
}

/// The pre-load plans one generic-lane image renders from: its pass, and for a Hires.fix request the
/// refinement pass compiled together with it.
#[cfg(target_os = "macos")]
#[derive(Default)]
pub(super) struct PreparedRender {
    pub(super) pass: Option<gen_core::RenderPlan>,
    pub(super) hires: Option<gen_core::RenderPlan>,
}

/// Compile a Hires.fix render — the first pass, then a refinement pass whose reference is the first
/// pass's output stretched to the target size at the refinement strength — as one multi-pass plan
/// before load. `refine` carries no reference of its own; the plan binds the produced one by size
/// and strength, because its pixels exist only after the first pass.
#[cfg(target_os = "macos")]
pub(super) fn compile_hires_render_plans(
    registry: &gen_core::ProviderRegistry,
    engine_id: &str,
    spec: &gen_core::LoadSpec,
    first: &GenerationRequest,
    refine: &GenerationRequest,
    hires: &HiresFixPlan,
) -> WorkerResult<(gen_core::RenderPlan, gen_core::RenderPlan)> {
    let passes = [
        gen_core::PlannedPassRequest {
            request: first,
            produced_reference: None,
        },
        gen_core::PlannedPassRequest {
            request: refine,
            produced_reference: Some(gen_core::ProducedReference {
                from_pass: 0,
                width: hires.width,
                height: hires.height,
                strength: Some(hires.provider_reference_strength),
            }),
        },
    ];
    let plan = registry
        .compile_multi_pass_render_plan(engine_id, spec, &passes)
        .map_err(|error| preload_plan_error(engine_id, error))?
        .ok_or_else(|| {
            WorkerError::Engine(format!(
                "{engine_id}: Hires.fix route has no registered pre-load render planner"
            ))
        })?;
    match plan.passes() {
        [first, refine] => Ok((first.clone(), refine.clone())),
        passes => Err(WorkerError::Engine(format!(
            "{engine_id}: Hires.fix plan compiled {} passes, expected 2",
            passes.len()
        ))),
    }
}

/// The refinement pass's reference must be exactly the image its plan declared it consumes.
pub(super) fn check_produced_reference(
    plan: &gen_core::RenderPlan,
    reference: &Image,
) -> WorkerResult<()> {
    match plan.produced_reference() {
        Some(produced)
            if (produced.width, produced.height) == (reference.width, reference.height) =>
        {
            Ok(())
        }
        Some(produced) => Err(WorkerError::InvalidPayload(format!(
            "Hires.fix refinement plan expects a {}x{} first-pass image, got {}x{}",
            produced.width, produced.height, reference.width, reference.height
        ))),
        None => Err(WorkerError::InvalidPayload(
            "Hires.fix refinement plan does not consume the first pass's output".to_owned(),
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HiresFixPlan {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) steps: u32,
    pub(super) guidance: Option<f32>,
    pub(super) true_cfg: Option<f32>,
    pub(super) provider_reference_strength: f32,
}

pub(super) fn resolve_hires_fix_plan(
    request: &ImageRequest,
    first_pass_steps: u32,
    first_pass_guidance: Option<f32>,
    first_pass_true_cfg: Option<f32>,
) -> Option<HiresFixPlan> {
    if request.hires_fix.is_disabled() {
        return None;
    }
    let upscale_by = request.hires_fix.effective_upscale_by();
    let denoising_strength = request.hires_fix.effective_denoising_strength();
    let family = resolve_family(request);
    // SDXL and Ideogram expose conventional denoising strength (higher means more regeneration).
    // The other current img2img providers expose reference fidelity (higher means preserve more),
    // so translate the user-facing A1111 denoising value at this boundary.
    let provider_reference_strength = if matches!(family.as_str(), "sdxl" | "ideogram") {
        denoising_strength
    } else {
        1.0 - denoising_strength
    };
    let cfg = request
        .hires_fix
        .effective_cfg_scale(first_pass_guidance.or(first_pass_true_cfg));
    Some(HiresFixPlan {
        width: sceneworks_core::image_request::hires_fix_target_dimension(
            request.width,
            upscale_by,
        ),
        height: sceneworks_core::image_request::hires_fix_target_dimension(
            request.height,
            upscale_by,
        ),
        steps: request.hires_fix.effective_steps(first_pass_steps),
        guidance: first_pass_guidance.map(|_| cfg.unwrap_or_default()),
        true_cfg: first_pass_true_cfg.map(|_| cfg.unwrap_or_default()),
        provider_reference_strength,
    })
}

/// The conditioning one generic-lane render carries. Split out of [`generate_one`] so
/// [`lane_reference_count`] — the count the backend request scope grades the request against — can be
/// tested against the conditioning this lane REALLY sends rather than against a restatement of it.
#[cfg(test)]
pub(super) fn build_lane_conditioning(
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
) -> Vec<Conditioning> {
    build_lane_conditioning_with_alpha(reference, multi_references, &[], edit_mask)
}

/// [`build_lane_conditioning`] where the ordered `multi_references` may carry alpha planes
/// (Qwen-Image 2.1, sc-24110/sc-24113). With no alpha anywhere it is exactly
/// [`build_lane_conditioning`]; otherwise the ordered list is built per entry by
/// [`build_ordered_reference_conditioning`].
pub(super) fn build_lane_conditioning_with_alpha(
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    multi_reference_alpha: &[Option<image::GrayImage>],
    edit_mask: Option<&Image>,
) -> Vec<Conditioning> {
    // `multi_references` (Boogu instruction edit, sc-7645) takes precedence when present: one image →
    // `Reference` (byte-identical to the single-reference path); 2–5 → `MultiReference`. Every other
    // family passes `&[]` and keeps the single `reference` (img2img init / IP-Adapter) path unchanged.
    let mut conditioning = if !multi_references.is_empty() {
        build_ordered_reference_conditioning(multi_references, multi_reference_alpha)
    } else {
        match reference {
            Some((image, strength)) => vec![Conditioning::Reference {
                image: image.clone(),
                strength: Some(*strength),
            }],
            None => Vec::new(),
        }
    };
    // Inpaint / outpaint mask (Ideogram 4 edit, sc-6303): a `Conditioning::Mask` (white = repaint)
    // alongside the source `Reference`. Only the Ideogram edit path supplies one today; every other
    // base-path family passes `None`.
    if let Some(mask) = edit_mask {
        conditioning.push(Conditioning::Mask {
            image: mask.clone(),
        });
    }
    conditioning
}

/// The image-conditioning count gen-core derives from the request [`generate_one`] builds for these
/// lane inputs — the SAME rule as `GenerationRequest::image_reference_count`: multi-reference
/// carriers contribute their image count, a single reference contributes one, and an edit mask
/// contributes one MORE (it is a `Conditioning::Mask` alongside the source `Reference`, not a
/// replacement for it).
///
/// This is not bookkeeping. The admitted geometry is re-derived from the live request by the
/// backend request scopes (`mlx-gen`'s `MlxRequestScopeCore::configure_request` and its candle twin),
/// which refuse any request whose geometry differs from the one admitted, and gen-core's shared
/// safety check rejects a `has_reference` that disagrees with `reference_count > 0`. Declaring a
/// count the request does not carry fails the render outright. The old formula
/// (`edit_refs.len().max(reference || mask)`) undercounted every reference+mask edit by one.
pub(super) fn lane_reference_count(
    has_identity_init: bool,
    multi_reference_count: usize,
    has_edit_mask: bool,
) -> u32 {
    let references = if multi_reference_count > 0 {
        multi_reference_count
    } else {
        usize::from(has_identity_init)
    };
    u32::try_from(references.saturating_add(usize::from(has_edit_mask))).unwrap_or(u32::MAX)
}

/// The hires-fix refinement pass conditions on exactly one image: the upscaled first-pass render,
/// passed as the single `Conditioning::Reference` (no multi-reference, no mask). Derived through
/// [`lane_reference_count`] rather than written as a literal so it cannot drift from the rule the
/// request scope grades against.
///
/// Shared by the MLX and Candle admission declarations so both backends describe the same final-pass
/// request identity.
pub(super) fn hires_fix_reference_count() -> u32 {
    lane_reference_count(true, 0, false)
}

/// The request-scope identity of the hires FIRST pass, derived from the admitted (final-pass)
/// context.
///
/// Admission describes the heaviest pass — the upscaled refinement — because that is what sets the
/// memory ceiling. The first pass renders at the BASE size with the caller's own conditioning, so
/// running it under the admitted context hands the backend request scope a geometry the request does
/// not match, and the scope refuses it (`request geometry … does not fit admitted …`) — which failed
/// the first pass of every hires render on a scope-adopting provider. Only the geometry identity
/// moves: the memory SELECTION is reused verbatim, since a strategy chosen for the larger pass is
/// the conservative choice for the smaller one.
pub(super) fn hires_first_pass_context(
    admitted: &gen_core::MemoryRunContext,
    width: u32,
    height: u32,
    reference_count: u32,
) -> gen_core::MemoryRunContext {
    let mut context = admitted.clone();
    context.geometry.width = width;
    context.geometry.height = height;
    context.geometry.reference_count = reference_count;
    context.has_reference = reference_count > 0;
    context.mode = if reference_count == 0 {
        gen_core::MemoryMode::TextToImage
    } else {
        gen_core::MemoryMode::ImageToImage
    };
    context.has_phases = false;
    context
}

/// Run the normal first pass followed by an optional high-resolution img2img refinement while
/// keeping progress monotonic across both denoise schedules.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_one_with_hires(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    hires_first_pass_memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    hires_first_pass_memory_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    hires_fix: Option<HiresFixPlan>,
    preview: gen_core::PreviewSink,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    generate_one_with_hires_on_surface(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt,
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        memory,
        hires_first_pass_memory,
        memory_strategy_context,
        hires_first_pass_memory_context,
        enhance,
        hires_fix,
        preview,
        prompt_enhancement,
        &LaneRgbaSurface::default(),
        cancel,
        on_progress,
    )
}

/// [`generate_one_with_hires`] with an explicit [`LaneRgbaSurface`].
///
/// The disposable Hires.fix BASE pass always decodes RGB: it only exists to become the
/// three-channel img2img reference of the refinement pass. The surface's `output_channels` applies
/// to the pass that is persisted.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate_one_with_hires_on_surface(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    multi_references: &[Image],
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    guidance_method: Option<&str>,
    use_pid: bool,
    text_style_gain: Option<f32>,
    memory: Option<gen_core::GenerationMemory>,
    hires_first_pass_memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    hires_first_pass_memory_context: Option<&gen_core::MemoryRunContext>,
    enhance: &PromptEnhance,
    hires_fix: Option<HiresFixPlan>,
    preview: gen_core::PreviewSink,
    prompt_enhancement: gen_core::PromptEnhancementSink,
    surface: &LaneRgbaSurface<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let Some(hires) = hires_fix else {
        return generate_one_on_surface(
            generator,
            prompt,
            width,
            height,
            seed,
            steps,
            guidance,
            negative_prompt,
            reference,
            multi_references,
            edit_mask,
            true_cfg,
            sampler,
            scheduler,
            scheduler_shift,
            guidance_method,
            use_pid,
            text_style_gain,
            memory,
            memory_strategy_context,
            enhance,
            prompt_enhancement,
            preview,
            surface,
            cancel,
            on_progress,
        );
    };

    let first_pass_context = hires_first_pass_memory_context.cloned().or_else(|| {
        memory_strategy_context.map(|context| {
            hires_first_pass_context(
                context,
                width,
                height,
                lane_reference_count(
                    reference.is_some(),
                    multi_references.len(),
                    edit_mask.is_some(),
                ),
            )
        })
    });
    let combined_steps = steps.saturating_add(hires.steps);
    let mut first_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current,
            total: combined_steps,
        }),
        // The first decode is an internal hand-off. The user-visible decode is the final one.
        Progress::Decoding => {}
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
        Progress::PostProcessing => on_progress(Progress::PostProcessing),
        Progress::PromptEncodingComplete => on_progress(Progress::PromptEncodingComplete),
        Progress::PromptUpsamplingComplete => on_progress(Progress::PromptUpsamplingComplete),
        Progress::ConditioningComplete => on_progress(Progress::ConditioningComplete),
        Progress::Upsampling { tokens, max_tokens } => {
            on_progress(Progress::Upsampling { tokens, max_tokens })
        }
    };
    // Enhancement belongs to the final persisted pass. Running it on the disposable base pass
    // would produce two reports for one image and could feed two different prompts into one recipe.
    let base_surface = LaneRgbaSurface {
        multi_reference_alpha: surface.multi_reference_alpha,
        flux3_references: surface.flux3_references,
        flux3_layout: surface.flux3_layout,
        output_channels: gen_core::OutputChannels::Rgb,
        controls: surface.controls,
        warnings: surface.warnings,
        prefix_kv_cache: surface.prefix_kv_cache,
        // A hires-fix render is per image; the arms refuse a prompt batch with it.
        prompt_batch: None,
        render_plan: surface.render_plan,
        hires_render_plan: None,
    };
    let (base_width, base_height, base_pixels) = generate_one_on_surface(
        generator,
        prompt,
        width,
        height,
        seed,
        steps,
        guidance,
        negative_prompt.clone(),
        reference,
        multi_references,
        edit_mask,
        true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        use_pid,
        text_style_gain,
        hires_first_pass_memory.or(memory),
        first_pass_context.as_ref(),
        &PromptEnhance::default(),
        gen_core::PromptEnhancementSink::default(),
        preview.clone(),
        &base_surface,
        cancel,
        &mut first_progress,
    )?;
    if cancel.is_cancelled() {
        return Err(WorkerError::Engine("generation cancelled".to_owned()));
    }
    let high_res_reference = fit_engine_image(
        Image {
            width: base_width,
            height: base_height,
            pixels: base_pixels,
        },
        hires.width,
        hires.height,
        "stretch",
    )?;
    if let Some(plan) = surface.hires_render_plan {
        check_produced_reference(plan, &high_res_reference)?;
    }
    let reference = (high_res_reference, hires.provider_reference_strength);
    let mut second_progress = |progress| match progress {
        Progress::Step { current, .. } => on_progress(Progress::Step {
            current: steps.saturating_add(current),
            total: combined_steps,
        }),
        Progress::Decoding => on_progress(Progress::Decoding),
        Progress::Loading(phase) => on_progress(Progress::Loading(phase)),
        Progress::PostProcessing => on_progress(Progress::PostProcessing),
        Progress::PromptEncodingComplete => on_progress(Progress::PromptEncodingComplete),
        Progress::PromptUpsamplingComplete => on_progress(Progress::PromptUpsamplingComplete),
        Progress::ConditioningComplete => on_progress(Progress::ConditioningComplete),
        Progress::Upsampling { tokens, max_tokens } => {
            on_progress(Progress::Upsampling { tokens, max_tokens })
        }
    };
    let refine_surface = LaneRgbaSurface {
        multi_reference_alpha: &[],
        flux3_references: None,
        flux3_layout: None,
        output_channels: surface.output_channels,
        controls: surface.controls,
        warnings: surface.warnings,
        prefix_kv_cache: surface.prefix_kv_cache,
        // A hires-fix render is per image; the arms refuse a prompt batch with it.
        prompt_batch: None,
        render_plan: surface.hires_render_plan,
        hires_render_plan: None,
    };
    generate_one_on_surface(
        generator,
        prompt,
        hires.width,
        hires.height,
        seed,
        hires.steps,
        hires.guidance,
        negative_prompt,
        Some(&reference),
        &[],
        None,
        hires.true_cfg,
        sampler,
        scheduler,
        scheduler_shift,
        guidance_method,
        false,
        text_style_gain,
        memory,
        memory_strategy_context,
        enhance,
        prompt_enhancement,
        preview,
        &refine_surface,
        cancel,
        &mut second_progress,
    )
}

/// Within-image step fraction mapped into the 0.10..0.95 generation band.
pub(super) fn step_fraction(index: usize, current: u32, total: u32, count: u32) -> f64 {
    let per = 0.85 / count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.1 + per * (index as f64 + within)).min(0.95)
}

#[cfg(test)]
mod process_fault_tests {
    use super::*;

    struct FaultGenerator {
        descriptor: gen_core::ModelDescriptor,
        poisoned: bool,
    }

    impl FaultGenerator {
        fn new(poisoned: bool) -> Self {
            Self {
                descriptor: gen_core::ModelDescriptor {
                    id: "render_fault_mock",
                    family: "test",
                    backend: "mlx",
                    modality: gen_core::Modality::Image,
                    capabilities: Default::default(),
                    encoder_contract: None,
                    denoiser_output_latent_space: None,
                    required_components: &[],
                    control_kinds: None,
                },
                poisoned,
            }
        }
    }

    impl Generator for FaultGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn validate(&self, _request: &gen_core::GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _request: &gen_core::GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<gen_core::GenerationOutput> {
            panic!("planned render must not fall back to legacy generation")
        }

        fn generate_planned(
            &self,
            _request: &gen_core::GenerationRequest,
            _plan: &gen_core::RenderPlan,
            _scope: Option<&mut dyn gen_core::MemoryRequestScope>,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<gen_core::GenerationReport> {
            if self.poisoned {
                Err(gen_core::Error::ProcessPoisoned(
                    "first provider device fault".into(),
                ))
            } else {
                Err(gen_core::Error::Unsupported("ordinary refusal".into()))
            }
        }
    }

    #[test]
    fn planned_render_preserves_typed_provider_poison_and_ordinary_error_class() {
        let request = gen_core::GenerationRequest {
            prompt: "a fox".into(),
            width: 64,
            height: 64,
            count: 1,
            seed: Some(7),
            steps: Some(1),
            ..Default::default()
        };
        let poison = FaultGenerator::new(true);
        let plan = gen_core::RenderPlan::base_image(
            poison.descriptor(),
            &request,
            gen_core::render::PackedImagePass {
                pixel_stride: 16,
                token_width: 64,
                sampler: "euler",
                guidance: 3.5,
                sigmas: vec![1.0, 0.0],
                executed_steps: 1,
                capture_sigma: 0.0,
            },
        )
        .expect("weights-free plan");
        let surface = LaneRgbaSurface {
            render_plan: Some(&plan),
            ..Default::default()
        };
        let render = |generator: &dyn Generator| {
            render_on_surface(
                generator,
                "a fox",
                64,
                64,
                7,
                1,
                None,
                None,
                None,
                &[],
                None,
                None,
                None,
                None,
                None,
                None,
                false,
                None,
                None,
                None,
                &PromptEnhance::default(),
                gen_core::PromptEnhancementSink::default(),
                gen_core::PreviewSink::default(),
                &surface,
                &CancelFlag::new(),
                &mut |_| {},
            )
        };
        let fatal = render(&poison).expect_err("provider fault must fail render");
        assert!(
            matches!(fatal, WorkerError::ProcessPoisoned(ref detail) if detail == "first provider device fault")
        );
        let ordinary = render(&FaultGenerator::new(false)).expect_err("ordinary refusal");
        assert!(
            matches!(ordinary, WorkerError::Engine(ref detail) if detail == "generation failed: unsupported: ordinary refusal")
        );
    }
}

#[cfg(test)]
#[path = "tests/flux1_conditional_route_tests.rs"]
pub(super) mod flux1_conditional_route_tests;
