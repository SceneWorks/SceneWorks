//! mlx_fit_gate request owner.
use super::*;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn budget_for_admission(
    mut budget: MemoryBudget,
    admission: &AdmissionRoute,
) -> MemoryBudget {
    if let Some(process_limit_bytes) = admission.process_limit_bytes {
        budget.reserved_headroom_bytes = budget.total_bytes.saturating_sub(process_limit_bytes);
    }
    budget
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn missing_reference_pricing(engine_id: &str) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{engine_id}: owned reference resources require the reference-aware memory admission entry point and its resolved LoadSpec; unpriced references cannot be admitted"
    ))
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn clip_window_upper_bound(prompt: &str, negative: &str) -> u32 {
    // Byte-level BPE can only merge the lowercased UTF-8 bytes. BOS/EOS leave 75 content
    // tokens per window. Count both CFG rows at their padded maximum, never sum sequential jobs.
    let bytes = prompt
        .to_lowercase()
        .len()
        .max(negative.to_lowercase().len());
    u32::try_from(bytes.div_ceil(75).max(1)).unwrap_or(u32::MAX)
}

/// Count the same untruncated CLIP tokens the loaded SDXL provider will window. The byte bound
/// remains a conservative fallback when tokenizer assets cannot be read during preparation.
#[cfg(target_os = "macos")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn clip_windows_for_spec(
    engine_id: &str,
    spec: &LoadSpec,
    prompt: &str,
    negative: &str,
) -> u32 {
    if engine_id == "sdxl" {
        use runtime_macos::providers::sdxl::{load_tokenizer, LDM_TOKENIZER_COMPONENT};
        let root = match &spec.weights {
            WeightsSource::Dir(root) => Some(root),
            WeightsSource::File(_) => match spec.components.get(LDM_TOKENIZER_COMPONENT) {
                Some(WeightsSource::Dir(root)) => Some(root),
                _ => None,
            },
        };
        let count = root
            .and_then(|root| load_tokenizer(root).ok())
            .and_then(|tokenizer| {
                let positive = tokenizer.tokenize(prompt).ok()?;
                let negative = tokenizer.tokenize(negative).ok()?;
                let tokens = positive.len().max(negative.len()).saturating_sub(2);
                u32::try_from(tokens.div_ceil(75).max(1)).ok()
            });
        if let Some(windows) = count {
            return windows;
        }
    }
    clip_window_upper_bound(prompt, negative)
}

/// Frozen FLUX3 sampling.denoise always packs [unconditional, conditional], including scales
/// 0 and 1; the distilled flow map packs one. This is NOT output/job count or decoder batch.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn flux3_conditioning_forward_batch(engine_id: &str) -> u64 {
    if engine_id == "flux3_klein_base" {
        2
    } else {
        1
    }
}

/// Source-derived retained conditioning/DiT allowances, NOT measurements. The source clamps text
/// at 4096, stacks 8 x 2560 encoder channels, uses 3072-wide blocks and 128-channel packed
/// latents. Each active row carries [`FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS`] at f32 width even though
/// most are bf16 — a deliberate PROVISIONAL over-estimate (no buffer reuse, f32 everywhere) until
/// the epic-end measured anchor (sc-26243) replaces it; it is bounded by the source live set, not
/// a guessed multiplier, so it does not refuse renders the live set could fit. Attention scratch
/// and allocator reserve remain in the existing generic headroom; no weights, decoder, job count
/// or optional PUP copies are added here.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn flux3_source_workspaces(engine_id: &str, width: u32, height: u32) -> (u64, u64) {
    if !matches!(engine_id, "flux3_klein" | "flux3_klein_base") {
        return (0, 0);
    }
    let rows = flux3_conditioning_forward_batch(engine_id);
    let image = u64::from(width.div_ceil(16)).saturating_mul(u64::from(height.div_ceil(16)));
    let conditioning = rows.saturating_mul(4096 * 20480 * 4);
    let projections = image
        .saturating_add(4096)
        .saturating_mul(3072 * 4 * FLUX3_BLOCK_LIVE_HIDDEN_BUFFERS)
        .saturating_mul(rows);
    let cfg_merge = if rows == 2 {
        image.saturating_mul(128 * 4 * 5)
    } else {
        0
    };
    (
        conditioning,
        projections
            .saturating_add(conditioning)
            .saturating_add(cfg_merge),
    )
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn flux3_cfg_additional_workspaces(
    engine_id: &str,
    width: u32,
    height: u32,
) -> (u64, u64) {
    if engine_id != "flux3_klein_base" {
        return (0, 0);
    }
    let base = flux3_source_workspaces(engine_id, width, height);
    let distilled = flux3_source_workspaces("flux3_klein", width, height);
    (
        base.0.saturating_sub(distilled.0),
        base.1.saturating_sub(distilled.1),
    )
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn flux3_cfg_workspace(
    engine_id: &str,
    geometry: MemoryGeometry,
    generic: u64,
) -> crate::mlx_phase_estimate::Workspace {
    let extra = flux3_cfg_additional_workspaces(engine_id, geometry.width, geometry.height);
    crate::mlx_phase_estimate::Workspace {
        conditioning: generic.saturating_add(extra.0),
        denoise: generic.saturating_add(extra.1),
        decode: generic,
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn add_post_load_external_delta(baseline: u64, before: u64, after: u64) -> u64 {
    baseline.saturating_add(after.saturating_sub(before))
}

/// The batch dimension of ONE provider forward pass, which on this lane is always 1 — NOT the job's
/// image count.
///
/// A multi-image job is a SEQUENTIAL loop, not a batched pass: `image_jobs::base` expands the job
/// count into one seed per image and hands the vector to `image_jobs::stream::drive_gen_items`, which
/// calls the provider once per seed with a `GenerationRequest { count: 1, .. }` and releases MLX's
/// retained-buffer cache between items (`RequestCacheRelease`, a `Drop` guard, so cancel and error
/// exits release too — sc-5567). Peak unified memory is therefore a MAX over items, not a sum: the
/// resident weights plus ONE image's transient working set, whatever the count.
///
/// This function exists because charging `count` as a batch dimension is not a small over-estimate,
/// it is unbounded. Both consumers of `geometry.batch` multiply by it — the generic estimator's
/// `request_scale` and Mage's `generation_peak_gb` — so a 4-image 1152x2048 krea_2_turbo request was
/// quoted 33.22 GiB of weights + 16 GiB x 2.25 MP x 4 = 177.22 GiB against a 126.00 GiB budget. The
/// activation term ALONE (144 GiB) exceeded the whole budget, so that cell rejected every model at
/// every tier on a 128 GiB Mac before a single weight byte was counted.
///
/// A provider that renders a genuine batched pass threads THAT pass's batch size here instead of
/// the job count: the native prompt batch (sc-25681, Iris-3B `generate(prompts=[…])`) carries its
/// `prompts × count` images in [`MlxRequestInputs::native_batch`]. All of them are charged, not
/// just one `count` repetition's prompt rows: the engine renders the repetitions inside the same
/// provider call and keeps every finished image live until it returns, and no batched run has been
/// measured (sc-25684 owns that), so the declared estimate scales the whole activation term by the
/// image count — conservative by construction, never a batch admitted on a one-image budget.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn request_batch(inputs: &MlxRequestInputs) -> u32 {
    inputs.native_batch.max(1)
}

/// The provider-facing geometry of one forward pass. See [`request_batch`] for the `batch` rule.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn request_geometry(inputs: &MlxRequestInputs) -> MemoryGeometry {
    MemoryGeometry {
        width: inputs.width,
        height: inputs.height,
        batch: request_batch(inputs),
        frames: 1,
        reference_count: inputs.reference_count,
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn request_mode(mode: &str) -> (MemoryMode, &'static str) {
    match mode {
        "image_generation" | "text_to_image" => (MemoryMode::TextToImage, "text_to_image"),
        "character_image" | "image_to_image" => (MemoryMode::ImageToImage, "image_to_image"),
        "edit_image" => (MemoryMode::Edit, "edit"),
        // sc-20799 round 2: unlike the video lane's reference-shape `"other"`, this `"other"` is
        // IDENTITY-BEARING and must stay. `mode_key` is matched against `binding.mode` when
        // resolving packaged evidence and is round-tripped by `memory_mode_from_mode_key`, whose
        // catch-all arm reconstructs `MemoryMode::Other("other")`. Renaming it (to "none" or
        // anything else) would silently repoint every non-standard image mode at a different
        // admission coordinate. The real asymmetry worth knowing about is that the typed value
        // keeps the concrete mode while the key collapses to a single bucket, so two distinct
        // non-standard modes share one evidence key; narrowing that is a calibration-corpus change,
        // not a rename, and is out of scope for this pass.
        _ => (MemoryMode::Other(mode.to_owned()), "other"),
    }
}

/// Provider-facing behavior identity for an exact routed request.
///
/// Krea Raw's public reference toggle deliberately remains the catalog's `image_generation`
/// coordinate, but the provider executes that one-reference request through its latent-init
/// image-to-image entrypoint. Keep that internal contract identity exact without inventing a new
/// public route or a generic CFG axis.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn provider_request_mode(
    engine_id: &str,
    inputs: &MlxRequestInputs,
) -> (MemoryMode, &'static str) {
    if (engine_id == "flux1_dev_control"
        && matches!(
            inputs.mode.as_str(),
            "image_generation" | "text_to_image" | "style_variations" | "character_image"
        ))
        || (engine_id == "flux1_dev" && inputs.mode == "character_image")
    {
        (MemoryMode::ImageToImage, "image_to_image")
    } else if matches!(engine_id, "flux1_schnell" | "flux1_dev")
        && matches!(
            inputs.mode.as_str(),
            "image_generation" | "text_to_image" | "style_variations"
        )
    {
        (MemoryMode::TextToImage, "text_to_image")
    } else if matches!(engine_id, "flux2_klein_9b_edit" | "flux2_klein_9b_kv_edit")
        && (1..=8).contains(&inputs.reference_count)
    {
        (MemoryMode::Edit, "edit")
    } else if (engine_id == "flux2_klein_9b" && inputs.has_reference && inputs.reference_count == 1)
        || is_krea_base_dit_reference_route(engine_id, inputs)
    {
        (MemoryMode::ImageToImage, "image_to_image")
    } else {
        request_mode(&inputs.mode)
    }
}

pub(super) fn is_krea_base_dit_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Krea)
}

pub(super) fn is_flux2_klein_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux2Klein)
}

pub(super) fn is_flux1_provider(engine_id: &str) -> bool {
    crate::memory_route_registry::gate_family(engine_id)
        == Some(crate::memory_route_registry::MemoryGateFamily::Flux1)
}

/// Provider-owned request overlay derived from the complete load identity.
///
/// FLUX.2 Klein authenticates every load-time composition at request scope, including conservative
/// eager requests whose BTR declaration refused the shape. Mirror the provider's closed axis order
/// from the actual `LoadSpec`; public labels such as `lora` and `references:1` are not evidence
/// identities. Krea's base-DiT family deliberately carries the same features on typed request/load
/// axes and always uses a provider overlay of None.
pub(crate) fn provider_overlay_for_load_spec(
    engine_id: &str,
    spec: &LoadSpec,
    public_overlay: Option<String>,
) -> Option<String> {
    if is_krea_base_dit_provider(engine_id) {
        return None;
    }
    if is_flux1_provider(engine_id) {
        let mut axes = Vec::new();
        if engine_id == "flux1_dev_control" || spec.control.is_some() {
            axes.push("control");
        }
        if !spec.extra_controls.is_empty() {
            axes.push("extra-controls");
        }
        if !spec.adapters.is_empty() {
            axes.push("adapters");
        }
        if spec.ip_adapter.is_some() {
            axes.push("ip-adapter");
        }
        if spec.pid.is_some() {
            axes.push("pid");
        }
        if spec.identity.is_some() {
            axes.push("identity");
        }
        if spec.text_encoder.is_some() {
            axes.push("external-text-encoder");
        }
        return (!axes.is_empty()).then(|| axes.join("-"));
    }
    if !is_flux2_klein_provider(engine_id) {
        return public_overlay;
    }
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.control.is_some() {
        axes.push("control");
    }
    if !spec.extra_controls.is_empty() {
        axes.push("extra-controls");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    if spec.text_encoder.is_some() {
        axes.push("external-text-encoder");
    }
    if !spec.components.is_empty() {
        axes.push("components");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn is_krea_base_dit_reference_route(engine_id: &str, inputs: &MlxRequestInputs) -> bool {
    matches!(engine_id, "krea_2_raw" | "krea_2_turbo")
        && matches!(inputs.mode.as_str(), "image_generation" | "text_to_image")
        && inputs.has_reference
        && inputs.reference_count == 1
}

/// Enforce provider request identity after production has derived any load-dependent overlay from
/// the complete `LoadSpec`. Krea always normalizes to None. Klein inputs already carry the exact
/// provider overlay from [`provider_overlay_for_load_spec`]; reconstructing it here from only an
/// adapter count would erase text-encoder and component axes on safe eager fallback routes.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn provider_request_inputs<'a>(
    engine_id: &str,
    inputs: &'a MlxRequestInputs,
) -> Cow<'a, MlxRequestInputs> {
    if !is_krea_base_dit_provider(engine_id) {
        return Cow::Borrowed(inputs);
    }
    if inputs.overlay.is_some() {
        let mut normalized = inputs.clone();
        normalized.overlay = None;
        Cow::Owned(normalized)
    } else {
        Cow::Borrowed(inputs)
    }
}

/// Translate a selection into the engine's per-rung engagement knobs.
///
/// SC-15805: this asks the contract which rungs the selection ENGAGES rather than re-deriving the
/// answer from `MemoryStrategy`'s numeric order. The order is a cost ordering, and the cumulative
/// default it expresses is defeasible — a rung the provider does not implement is not engaged, so a
/// provider publishing a verified cheaper composition no longer has its unimplemented rung's knob
/// switched on underneath it.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn memory_for_selection(
    contract: &MemoryProviderContract,
    selection: MemorySelection,
) -> GenerationMemory {
    contract.generation_memory(&selection).unwrap_or_default()
}

/// Select the largest strictly lower, same-aspect geometry backed by an exact record of the
/// installed artifact that fits the live host boundary. This is the only source for a named
/// refusal alternative: no formula, interpolation, tier heuristic, or aspect-ratio rewrite is
/// admitted.
///
/// The binding's provider closure is not a conjunct (sc-22738): the alternative is advice the very
/// next request will price from the same record, and that request grades the record at its
/// measured peak whatever closure it was captured under, so the advice is honoured either way.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn verified_lower_alternative(
    bundle: &EvidenceBundle,
    calibration: &MlxCalibrationSet,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> Option<VerifiedGeometryAlternative> {
    let overlay = inputs.overlay.as_deref().unwrap_or("none");
    let requested_width = u64::from(inputs.width);
    let requested_height = u64::from(inputs.height);
    calibration
        .bindings
        .iter()
        .filter(|binding| {
            binding.provider == plan.engine_id
                && binding.tier == plan_tier_key(plan.tier)
                && binding.mode == mode_key
                && binding.overlay == overlay
                && binding.query.artifact_repository == calibration.resolved.identity.repository
                && binding.query.artifact_resolved_revision
                    == calibration.resolved.identity.revision
                && binding.query.artifact_variant == calibration.resolved.identity.variant
                && binding.query.resolved_path_fingerprint
                    == calibration.resolved.identity.fingerprint
                && binding.geometry.batch == request_batch(inputs)
                && binding.geometry.frames == 1
                && binding.geometry.width <= inputs.width
                && binding.geometry.height <= inputs.height
                && (binding.geometry.width < inputs.width
                    || binding.geometry.height < inputs.height)
                && u64::from(binding.geometry.width) * requested_height
                    == u64::from(binding.geometry.height) * requested_width
        })
        .filter_map(|binding| {
            let query = EvidenceQuery {
                backend: CalibrationBackend::Mlx,
                model_id: plan.model_id.clone(),
                provider: binding.provider.clone(),
                tier: binding.tier.clone(),
                mode: binding.mode.clone(),
                overlay: binding.overlay.clone(),
                transformer_variant: None,
                decoder: None,
                geometry: binding.geometry,
                rung: binding.rung,
                parameters: binding.parameters.clone(),
                calibration: binding.query.clone(),
            };
            let EvidenceVerdict::Verified(record) = bundle.evidence_for(&query) else {
                return None;
            };
            let envelope = record.mlx_admission_envelope()?;
            let effective = budget.effective_bytes();
            let strategy = evidence_strategy(record.strategy.rung);
            (envelope.fits_scaled_host_bytes(budget.total_bytes)
                && envelope.peak_bytes <= effective)
                .then_some(VerifiedGeometryAlternative {
                    geometry: binding.geometry,
                    calibration_abi: binding.query.abi,
                    calibration_fingerprint: binding.query.fingerprint.clone(),
                    load_shape: crate::memory_strategy::load_shape_from_receipt(record.load_shape),
                    strategy,
                    parameters: binding.selection_parameters,
                    engaged_composition: record
                        .strategy
                        .engaged_rungs
                        .iter()
                        .copied()
                        .map(evidence_strategy)
                        .collect(),
                })
        })
        .max_by_key(|alternative| {
            let geometry = alternative.geometry;
            (
                u64::from(geometry.width) * u64::from(geometry.height),
                geometry.width,
                geometry.height,
            )
        })
}

#[cfg(test)]
pub(super) fn verified_lower_geometry(
    bundle: &EvidenceBundle,
    calibration: &MlxCalibrationSet,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    mode_key: &str,
    budget: MemoryBudget,
) -> Option<CalibrationGeometry> {
    verified_lower_alternative(bundle, calibration, plan, inputs, mode_key, budget)
        .map(|alternative| alternative.geometry)
}

/// Pure request selector used by production and unit/hardware seams. Additional provider evidence is
/// accepted explicitly; absent exact verified cells, only the resident baseline can be admitted.
#[allow(clippy::too_many_arguments)]
/// The pre-sc-18317 shape, kept for the fixture suite: an evaluation with nothing to decide about the
/// request's execution policy. Production always goes through
/// [`evaluate_request_with_budget_and_warm_policy`] so a grant cannot be dropped on the way in.
#[cfg(test)]
pub(super) fn evaluate_request_with_budget(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
) -> WorkerResult<MlxRequestEvaluation> {
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        None,
    )
}

/// [`evaluate_request_with_budget`] carrying this request's warm-policy proposal (sc-18317).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn evaluate_request_with_budget_and_warm_policy(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestEvaluation> {
    evaluate_request_with_budget_using_bundle(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        None,
        reference_pricing,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn evaluate_request_with_budget_using_bundle(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    evidence_bundle: Option<&EvidenceBundle>,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestEvaluation> {
    select_request_with_budget_using_bundle(
        MlxAdmissionPhase::Generation,
        generator
            .descriptor()
            .capabilities
            .component_precision_floors,
        generator.memory_strategy_contract(),
        crate::execution_planner::prefix_kv_cache_surcharge_bytes(
            plan.engine_id,
            &generator.descriptor().capabilities.execution,
            plan.tier.quant,
            inputs.width,
            inputs.height,
            inputs.reference_count,
            inputs.negative_branch,
        )
        .filter(|_| inputs.prefix_kv_cache_eligible),
        &|forced_tile_edge| {
            mlx_prefix_kv_cache_shed_transient_bytes(
                plan.engine_id,
                plan.tier.quant,
                inputs,
                forced_tile_edge,
            )
        },
        |base| generator.predicted_memory_peak_from_base(base),
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        additional_evidence,
        evidence_bundle,
        reference_pricing,
    )?
    .into_evaluation()
}

/// The LINKED provider's own cache-shed request transient for a prefix-KV-cache provider
/// (`derived::request_transient_budget_bytes` — the budget the generator holds a prefill render
/// to), at the loaded tier and the rung's forced decode tile; `None` for every other provider.
///
/// sc-25909 (owner-approved E6 exception): the shed ladder compares this against the activation a
/// reference request's rung was charged — the activation slice of the generic area-scaled
/// headroom, on a rung whose activation term is that headroom — and prices the shortfall into
/// every such rung's peak. Every admission that runs the shed ladder grades it: the prefill path,
/// the cached path (corrected shed rung + cache surcharge) and the pre-load tier probe. Same
/// 256-row conditioning assumption as the generator's own bound.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn mlx_prefix_kv_cache_shed_transient_bytes(
    engine_id: &str,
    tier_quant: Option<gen_core::Quant>,
    inputs: &MlxRequestInputs,
    forced_tile_edge: Option<u32>,
) -> Option<u64> {
    if !sceneworks_core::jobs_store::is_qwen_image_2_1_model(engine_id) {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        use runtime_macos::providers::qwen_image_2_1 as provider;
        let tier = match tier_quant {
            None => provider::quant::Tier::Bf16,
            Some(gen_core::Quant::Q8) => provider::quant::Tier::Q8,
            Some(gen_core::Quant::Q4) => provider::quant::Tier::Q4,
            Some(_) => return None,
        };
        let derived = provider::memory_strategy::derived::request_transient_budget_bytes;
        Some(derived(
            tier,
            inputs.width,
            inputs.height,
            provider::memory_strategy::derived::TABLE_CONDITIONING_TOKENS,
            inputs.reference_count,
            inputs.negative_branch,
            forced_tile_edge,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (tier_quant, inputs, forced_tile_edge);
        None
    }
}

impl MlxRequestAdmission {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(super) fn into_evaluation(self) -> WorkerResult<MlxRequestEvaluation> {
        match self {
            Self::Admitted(evaluation) => Ok(*evaluation),
            Self::Rejected(error) => Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn select_request_with_budget_using_bundle(
    phase: MlxAdmissionPhase,
    declared_floors: &'static [gen_core::ComponentPrecisionFloor],
    declared_contract: Option<&MemoryProviderContract>,
    // The provider's prefix-KV-cache surcharge for this request
    // (`execution_planner::prefix_kv_cache_surcharge_bytes`, priced against the LOADED generator's
    // declaration), when this admission may select the cache. `None` for every provider without
    // one — and on the pre-load tier probe: a tier fits when it fits with the cache shed (the cache
    // is only ever added to the rung the shed demand selects, never a reason to drop a tier), so the
    // probe grades exactly the shed demand.
    prefix_kv_cache_surcharge: Option<u64>,
    // sc-25909: the provider's own cache-shed request transient at a rung's forced decode tile
    // (`mlx_prefix_kv_cache_shed_transient_bytes`), or `None`. Priced into every rung whose
    // activation term is the generic area-scaled headroom — into the shed ladder itself, so the
    // prefill and cached paths both grade it (owner-approved E6 exception, 2026-10-16).
    prefix_kv_cache_shed_transient: &dyn Fn(Option<u32>) -> Option<u64>,
    predict_peak: impl FnOnce(u64) -> gen_core::MemoryPeakBreakdown,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    budget: MemoryBudget,
    total_peak_bytes: u64,
    external_committed_bytes: u64,
    additional_evidence: &[MemoryEvidence],
    evidence_bundle: Option<&EvidenceBundle>,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestAdmission> {
    use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};

    let reference_workspace_bytes = if let Some(resources) = &inputs.reference_resources {
        if !inputs.has_reference
            || usize::try_from(inputs.reference_count).ok()
                != Some(resources.prepared_canvases.len())
        {
            return Err(WorkerError::InvalidPayload(
                "reference resource slots do not match the admitted ordered reference count"
                    .to_owned(),
            ));
        }
        let pricing = reference_pricing.ok_or_else(|| missing_reference_pricing(plan.engine_id))?;
        crate::memory_strategy::reference_workspace_bytes_using(
            pricing.registry,
            plan.engine_id,
            pricing.spec,
            Some(resources),
        )?
    } else {
        0
    };

    // Component precision floors are a provider property, not a manifest guess. Bind them only
    // from the exact registry descriptor or loaded generator, then use that tier for every evidence/cache identity
    // in this request so uniform-q4 measurements cannot authorize a mixed-precision provider.
    let provider_floors = active_component_floors(declared_floors, plan.tier.quant);
    let mut effective_plan;
    let plan = if plan.tier.component_precision_floors == provider_floors {
        plan
    } else {
        effective_plan = plan.clone();
        effective_plan.tier.component_precision_floors = provider_floors;
        &effective_plan
    };

    let provider_inputs = provider_request_inputs(plan.engine_id, inputs);
    let inputs = provider_inputs.as_ref();
    let geometry = request_geometry(inputs);
    let (mode, mode_key) = provider_request_mode(plan.engine_id, inputs);
    let mut fallback_contract;
    let provider_contract = if let Some(contract) = declared_contract {
        contract
    } else {
        fallback_contract = MemoryProviderContract::compatibility_default(
            plan.engine_id,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: false,
                cache_eviction: true,
            },
        );
        fallback_contract.asset_facts.base_bytes = plan.asset_bytes;
        // PR #395 (asset-facts unification) requires base_bytes to equal the sum of the base
        // component bytes. The legacy fallback has only the summed on-disk figure — no per-component
        // split — so carry the whole sum on the transformer axis rather than fail conformance.
        fallback_contract.asset_facts.transformer_bytes = plan.asset_bytes;
        &fallback_contract
    };
    let refused_contract;
    let contract = if plan.load_shape_declaration_result == LoadShapeDeclarationResult::Refused
        || !crate::memory_route_registry::has_gate_policy(
            crate::memory_route_registry::MemoryRouteBackend::Mlx,
            plan.engine_id,
        ) {
        // A declaration refusal is terminal authority, not an invitation for the later request
        // selector to rediscover lower optimized rungs directly from the provider. Preserve only
        // the exact provider/load/calibration handshake needed for a safe resident request; the
        // provider's optimized table remains unreachable until a complete declaration applies.
        let mut resident = MemoryProviderContract::compatibility_default(
            provider_contract.provider_id.clone(),
            provider_contract.backend.clone(),
        );
        resident.load_shape = provider_contract.load_shape;
        resident.calibration = provider_contract.calibration.clone();
        resident.asset_facts = provider_contract.asset_facts;
        resident.resident_request_memory = provider_contract.resident_request_memory;
        refused_contract = resident;
        &refused_contract
    } else {
        provider_contract
    };
    if plan.engine_id.starts_with("mage_flow")
        && inputs.adapter_count > 0
        && !contract.resident_components().iter().any(|component| {
            component.kind == gen_core::MemoryComponentKind::AdapterStack
                && component.resident_bytes > 0
        })
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} request includes {} adapter(s), but the loaded provider did not declare \
             load-exact adapter residency; refusing an unbounded MLX request",
            plan.engine_id, inputs.adapter_count
        )));
    }
    let calibration_fingerprint = contract
        .calibration
        .as_ref()
        .map(|identity| identity.fingerprint.as_str());
    let calibration_abi = contract
        .calibration
        .as_ref()
        .map_or(0, |identity| identity.abi);
    // No live closure is resolved here (sc-22738). Between sc-17774 and sc-22738 this gate read
    // the provider's compile-closure digest and graded every measured candidate against it; now a
    // measured candidate is graded at its measured peak whatever closure it was captured under.
    let mut admission = if plan.tier.component_precision_floors.is_empty() {
        match evidence_bundle {
            Some(bundle) => evidence_admission_route(bundle, plan, inputs, mode_key, budget)?,
            None => packaged_admission_route(plan, inputs, mode_key, budget)?,
        }
    } else {
        // Persisted calibration bindings currently identify only the coarse tier token (for
        // example `q4`). They cannot distinguish uniform q4 from a provider whose descriptor keeps
        // selected components at q8, so they must not be relabeled with the live descriptor floors.
        // Until the on-disk evidence schema carries the floors explicitly, fail closed to the
        // provider's conservative resident estimate.
        AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
            evidence: Vec::new(),
            estimate_bases: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        }
    };
    // sc-18101: a candidate whose MATERIALIZATION SHAPE the loaded provider does not use is not a
    // measurement of this load, and must be dropped BEFORE the selector — but dropping it must not
    // take the route's usable siblings with it.
    //
    // `MemoryCalibrationIdentity` has three fields — `abi`, `fingerprint`, `load_shape` — and
    // gen-core's `optimized_eligibility` rejects an OPTIMIZED candidate whose `key.load_shape`
    // disagrees with either `contract.load_shape` or `identity.load_shape`, returning
    // `FingerprintMismatch`. (`Resident` is exempt: `optimized_eligibility` returns `Ok(())` for it
    // before the shape comparison, because a resident cell engages no optimized rung whose
    // materialization the shape could change. The filter below mirrors that exemption exactly —
    // see its own comment.) The demotion below compares only the first two, which left a hole with
    // nothing behind it: a cell
    // whose records were all captured under the other shape reached `AdmissionPath::Evidence`, lost
    // every candidate inside `select_strategy`, and refused the request outright with "no
    // structurally admissible MLX memory strategy" — because estimate synthesis runs only on the
    // Legacy route.
    //
    // That hole shipped. `mlx:qwen_image` q8 1024² was captured `eager_materialization`, while
    // `image_jobs::apply_measured_mlx_load_shape` forces `DeferredMaterialization` on every
    // `qwen_image` directory load, so the flagship q8 route hard-refused its most common geometry on
    // a 128 GiB machine. Measured on real weights at this commit and at the epic's base commit: the
    // base commit ADMITTED (its `evidence_admission_route` closure conjunct, retired by sc-18096,
    // pre-demoted the cell to Legacy long before eligibility ran), so this is a REGRESSION this epic
    // introduced. See `docs/epic-18093-end-to-end-validation-sc-18101.md`.
    //
    // Why a per-candidate FILTER and not a whole-route demotion: a route legitimately carries
    // bindings for several rungs captured under DIFFERENT shapes — `qwen_image` q8 ships a
    // `bounded_attention` cell measured eager and a `bounded_transformer_residency` cell measured
    // deferred. Demoting the whole route when any sibling mismatches would throw away the candidate
    // that DOES match and silently downgrade a calibrated request to an estimate, which is its own
    // regression. Filtering keeps every usable measurement and degrades only when nothing is left.
    //
    // sc-18251: the same hole class existed on the two structural legs the sc-18101 filter did not
    // mirror. A promoted binding whose captured `engaged_composition` disagrees with the live
    // contract's (an engagement edge grew or was excluded since capture), or whose parameters no
    // longer pass the live `contract.validate_selection` (a declared range narrowed), still entered
    // `AdmissionPath::Evidence`, lost every candidate inside `select_strategy`
    // (`CompositionMismatch` / `Invalid`), and hard-refused via `Selection::Unverified` — where
    // pre-epic code degraded to legacy first (the retired closure pre-demotion caught every
    // drifted binding before eligibility ran). The filter now mirrors those legs too,
    // so composition drift and parameter-range narrowing degrade to the estimate ladder exactly
    // like a shape mismatch.
    if admission.path == AdmissionPath::Evidence {
        // A candidate is USABLE only if the selector could actually reach it: it must have been
        // measured under a shape the downstream eligibility gate will accept for its rung, its
        // captured engaged composition must still be the loaded contract's canonical set for that
        // rung, and the selection it authorizes must still pass the loaded contract's own
        // `validate_selection`. All three are properties of the LOADED PROVIDER, not of the
        // request, and all three are checked downstream where the only outcome left is a refusal
        // (`FingerprintMismatch`, `CompositionMismatch`, and `Invalid` respectively — and an
        // unimplemented rung silently, since `select_strategy` skips it without even recording an
        // exclusion, so that request dies with a bare `Missing`).
        //
        // The shape test MIRRORS `optimized_eligibility` rather than tightening it. That gate
        // short-circuits `Ok(())` for `MemoryStrategy::Resident` before it ever compares load
        // shapes, so a resident cell measured under the other shape is one the gate ACCEPTS.
        // Dropping it here would make this filter stricter than the thing it exists to anticipate,
        // and would silently discard a usable measurement — the same failure mode as the whole-route
        // demotion this filter deliberately avoids. Reachable in the shipped corpus: `qwen_image`
        // bf16 carries a resident binding captured eager against a production deferred load.
        let usable = |candidate: &VerifiedAdmissionCandidate| {
            let evidence = &candidate.evidence;
            // The two conjuncts mirror `optimized_eligibility`'s pair literally, but they are NOT
            // independently observable (sc-18251 review): gen-core's `conformance_errors` requires
            // `calibration.load_shape == contract.load_shape`, and `select_strategy` refuses every
            // candidate on a non-conformant contract with `Unverified(Invalid)` before grading a
            // single one — so on any contract where the pair could split, the request refuses no
            // matter what this filter does. The pair is effectively one comparison; both spellings
            // are kept so the filter stays a literal mirror of the gate. The premise is pinned by
            // `gen_core_forbids_a_contract_identity_load_shape_split`: if a pin bump ever legalizes
            // the split, that test reds and the identity conjunct needs its own fixture arm.
            let shape_agrees = !evidence.key.strategy.is_optimized()
                || contract.calibration.as_ref().is_some_and(|identity| {
                    evidence.key.load_shape == contract.load_shape
                        && evidence.key.load_shape == identity.load_shape
                });
            // sc-18251: COMPOSITION. `optimized_eligibility`'s structural prefix rejects a
            // candidate whose captured `engaged_composition` is not the loaded contract's canonical
            // set for its rung (`CompositionMismatch`) — and unlike the shape leg above there is NO
            // Resident exemption to mirror: both of the gate's composition checks run BEFORE its
            // `is_optimized` short-circuit, pinned against the pinned gen-core by
            // `gen_core_rejects_a_resident_cell_whose_composition_disagrees`. The gate's
            // canonical-form check (`Invalid` for an empty or unsorted captured set) needs no
            // separate conjunct: `contract.engaged_composition` walks `MemoryStrategy::ALL` in
            // order, so equality forces the captured set into canonical form — except when both
            // sides are empty, which requires the selected rung itself to be non-`Implemented`,
            // and `selection_valid` below drops exactly that candidate.
            let selection = MemorySelection {
                strategy: evidence.key.strategy,
                parameters: evidence.key.parameters,
                tier: evidence.key.tier,
            };
            let composition_agrees = evidence.key.engaged_composition
                == contract.engaged_composition_for_selection(&selection);
            // sc-18251: SELECTION. `select_strategy` runs every candidate through the loaded
            // contract's own `validate_selection` (`memory_strategy::candidate_exclusion`),
            // excluding with `Invalid` a candidate whose rung the provider no longer declares
            // `Implemented`, whose prerequisite edges the live contract no longer satisfies, or
            // whose captured parameters fall outside the provider's live declared ranges — a
            // narrowed range strands the old measurement. The downstream check applies to every
            // rung, Resident included, so no exemption is mirrored here either. Consulting the
            // gate's own predicate (rather than a resembling re-implementation) also subsumes the
            // previous standalone `Implemented` conjunct: `validate_selection` fails for an
            // undeclared or non-`Implemented` rung before it looks at parameters.
            let selection_valid = contract
                .validate_selection(&MemorySelection {
                    strategy: evidence.key.strategy,
                    parameters: evidence.key.parameters,
                    tier: evidence.key.tier,
                })
                .is_ok();
            shape_agrees && composition_agrees && selection_valid
        };
        let retained = admission.evidence.iter().filter(|c| usable(c)).count();
        if retained != admission.evidence.len() {
            tracing::info!(
                route = plan.engine_id,
                backend = "mlx",
                dropped = admission.evidence.len() - retained,
                retained,
                contract_load_shape = ?contract.load_shape,
                "dropped measured candidates the loaded provider cannot serve (materialization \
                 shape, engaged composition, or selection validity)"
            );
            admission.evidence.retain(usable);
        }
        if admission.evidence.is_empty() {
            // Nothing measured survives. Degrade to the estimate ladder rather than refuse: a
            // calibration identity that does not match the loaded provider is a reason to stop
            // CLAIMING the measurement, not a reason to deny service. The estimate ladder is
            // strictly more capable than the pre-epic resident-only freeze it replaces.
            admission = AdmissionRoute {
                path: AdmissionPath::Legacy,
                fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
                evidence: Vec::new(),
                estimate_bases: Vec::new(),
                evidence_revision: None,
                process_limit_bytes: None,
                lower_alternative: None,
            };
        }
    }
    // A named refusal alternative is advice the next request must be able to honour, so it is held
    // to the same shape agreement — but it is only advice, so a mismatch drops the alternative
    // rather than the route.
    if admission
        .lower_alternative
        .as_ref()
        .is_some_and(|alternative| {
            contract
                .calibration
                .as_ref()
                .is_none_or(|identity| alternative.load_shape != identity.load_shape)
        })
    {
        admission.lower_alternative = None;
    }
    let carries_verified_claim =
        admission.path == AdmissionPath::Evidence || admission.lower_alternative.is_some();
    if carries_verified_claim
        && !contract.calibration.as_ref().is_some_and(|identity| {
            admission.evidence.iter().all(|candidate| {
                identity.abi == candidate.evidence.calibration_abi
                    && identity.fingerprint == candidate.evidence.calibration_fingerprint
            }) && admission
                .lower_alternative
                .as_ref()
                .is_none_or(|alternative| {
                    identity.abi == alternative.calibration_abi
                        && identity.fingerprint == alternative.calibration_fingerprint
                        && contract.engaged_composition_for_selection(&MemorySelection {
                            strategy: alternative.strategy,
                            parameters: alternative.parameters,
                            tier: plan.tier,
                        }) == alternative.engaged_composition
                })
        })
    {
        admission = AdmissionRoute {
            path: AdmissionPath::Legacy,
            fallback_reason: Some(LegacyAdmissionReason::IdentityMismatch),
            evidence: Vec::new(),
            estimate_bases: Vec::new(),
            evidence_revision: None,
            process_limit_bytes: None,
            lower_alternative: None,
        };
    }
    // The selector and provider request context must use the same evidence-derived foreign demand
    // as the fail-closed precheck and the request-scoped MLX ceiling. Otherwise a covered request
    // with existing committed memory could be admitted against the legacy 2 GiB reserve even though
    // its record captured a larger non-process share.
    let mut budget = budget_for_admission(budget, &admission);
    tracing::info!(
        event = "mlx_memory_admission_path",
        route = plan.engine_id,
        model = plan.model_id,
        path = ?admission.path,
        fallback_reason = ?admission.fallback_reason,
        evidence_revision = admission.evidence_revision.as_deref().unwrap_or("none"),
        width = inputs.width,
        height = inputs.height,
        count = inputs.count.max(1),
        "selected MLX memory-admission path"
    );
    if plan.engine_id.starts_with("mage_flow") {
        // The expected identity is the (route, tier) cell of THIS plan: a loaded q8 contract must
        // not admit a q4 request, and a route or tier the engine never ships names no cell at all.
        let expected_fingerprint = mage_calibration_fingerprint(plan.engine_id, plan.tier.quant)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{} {:?} names no Mage-Flow calibration cell; refusing request admission \
                     against an unpaired estimator",
                    plan.engine_id, plan.tier.quant
                ))
            })?;
        if !matches!(
            contract.calibration.as_ref(),
            Some(identity)
                if identity.abi == gen_core::MEMORY_CALIBRATION_ABI
                    && identity.fingerprint == expected_fingerprint
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "{} loaded provider calibration does not match ABI {} / fingerprint {}; refusing \
                 request admission against an unpaired estimator",
                plan.engine_id,
                gen_core::MEMORY_CALIBRATION_ABI,
                expected_fingerprint
            )));
        }
    }
    // The caller estimates the base-model pipeline. Let the provider's canonical contract seam add
    // any separately declared auxiliary networks before either fit selection or warm-cache credit.
    // Exact evidence already describes the whole request peak and therefore remains authoritative.
    let contract_base_peak_bytes = plan.contract_base_peak_bytes(total_peak_bytes, contract);
    let base_prediction = predict_peak(contract_base_peak_bytes);
    let evidence_peak_bytes = admission
        .evidence
        .iter()
        .map(|candidate| candidate.evidence.predicted_peak_bytes)
        .max();
    let modeled_peak_bytes =
        evidence_peak_bytes.unwrap_or_else(|| base_prediction.predicted_peak_bytes());

    // The modeled peak is a complete-pipeline peak, while the live budget is incremental from the
    // process's current state. Only bytes above the cache-recorded pre-load external baseline, and
    // no more than the provider-declared total resident envelope, may be credited as already
    // present. Unrelated process allocations therefore remain charged on the available side.
    //
    // sc-19721: `total_resident_bytes()` stays the LOAD-EXACT envelope on a provider that declares
    // an eviction. It is a ceiling on a credit, and the credit is already floored by what the
    // process has actually committed — so a provider caught before its precompute-and-evict is
    // credited for what it really holds, while one caught after is bounded by `committed_bytes`
    // anyway. Lowering the ceiling to the steady state would only refuse credit that is genuinely
    // resident.
    let attributable_resident_bytes = budget
        .committed_bytes
        .saturating_sub(external_committed_bytes)
        .min(contract.total_resident_bytes());
    if attributable_resident_bytes > modeled_peak_bytes {
        return Err(WorkerError::InvalidPayload(format!(
            "{} live resident attribution {} exceeds modeled total peak {}; refusing an \
             inconsistent MLX budget",
            plan.engine_id, attributable_resident_bytes, modeled_peak_bytes
        )));
    }
    tracing::info!(
        route = plan.engine_id,
        total_bytes = budget.total_bytes,
        active_bytes = budget.committed_bytes,
        external_baseline_bytes = external_committed_bytes,
        provider_credit_bytes = attributable_resident_bytes,
        reserved_bytes = budget.reserved_headroom_bytes,
        "MLX request memory attribution"
    );
    // Every candidate is a whole-pipeline peak, including synthesized ladder estimates.
    // Remove this provider's already-resident allocation from the committed side once, so
    // cold/warm Resident and staged candidates use the same accounting domain. Keep unrelated
    // allocations charged, and apply safety allowances to the original modeled terms.
    budget.committed_bytes = budget
        .committed_bytes
        .saturating_sub(attributable_resident_bytes);
    let predicted_peak_bytes = modeled_peak_bytes;
    // `request_total_peak_bytes` is the generic weights+headroom total on every route except
    // `mage_flow`, whose resident peak is the provider's own `generation_peak_gb` scalar.
    let resident_peak_is_provider_scalar = plan.engine_id.starts_with("mage_flow");
    let (resident_selection, resident) = resident_evidence(
        contract,
        plan.tier,
        mode_key,
        inputs.overlay.as_deref(),
        geometry,
        predicted_peak_bytes,
        calibration_fingerprint,
    );
    // sc-18096: on a legacy route — the exact request cell has no measured evidence — synthesize
    // estimate-backed candidates for every implemented optimized rung, so the full ladder is
    // selectable behind the estimate margin instead of freezing to the resident baseline and
    // refusing. A covered (`Evidence`) route gets none: its measured ladder is authoritative.
    let synthesized_ladder = if admission.path == AdmissionPath::Legacy {
        synthesize_estimate_ladder_with_conditioning(
            contract,
            plan,
            mode_key,
            inputs.overlay.as_deref(),
            geometry,
            inputs.use_pid,
            calibration_fingerprint,
            &admission.estimate_bases,
            inputs.conditioning_windows,
        )
    } else {
        SynthesizedEstimateLadder::default()
    };
    let synthesized_estimates = &synthesized_ladder.estimates;
    let mut selections = Vec::new();
    let mut evidence = Vec::new();
    // Index-aligned basis axis (sc-18096): synthesized candidates carry their estimate basis, the
    // resident baseline is the rung-0 weights+headroom floor, and everything measured stays
    // `Measured`. Pushed at the same sites as `evidence` for the same fail-open reason as the
    // digests above.
    let mut candidate_bases: Vec<crate::memory_strategy::CandidateBasis> = Vec::new();
    // Index-aligned ACTIVATION axis (sc-22508), pushed at the same sites for the same reason: only
    // the site that CONSTRUCTED a peak knows whether it decomposes into counted weights and a flat
    // activation term, and what that term is. A blanket declaration here was wrong for two peaks —
    // a `mage_flow` route, whose peak is `providers::mage::memory::generation_peak_gb` and has no
    // weights/headroom split at all, and any peak a generator rewrote through
    // `predicted_memory_peak_from_base`. Both now carry `None` and take the policy's documented
    // undeclared-floor arm instead of being charged a headroom they do not contain.
    let mut candidate_activation_bytes: Vec<Option<u64>> = Vec::new();
    // Index-aligned, pushed at the same sites (sc-25909): whether the candidate's peak IS counted
    // weights + the generic area-scaled headroom — the resident baseline on every non-`mage_flow`
    // route and a floor estimate composed with `ImageFloorActivationTerm::GenericHeadroom`. Only
    // those rungs take the provider-transient correction below; a measured, fitted-curve,
    // anchor-derived, phase or law-residue rung prices its own activation.
    let mut candidate_charges_generic_headroom: Vec<bool> = Vec::new();
    // Index-aligned with `evidence`: the Evidence path's per-candidate pre-check budget (GiB, at the
    // candidate's CAPTURED foreign reserve), so a cache-surcharged pass re-runs the same pre-check
    // at the surcharged peak. `None` where no pre-check applies (the Legacy path).
    let mut candidate_precheck_available_gb: Vec<Option<f64>> = Vec::new();
    if admission.path == AdmissionPath::Evidence {
        // A covered cell is authorized only by its exact verified ladder. Letting the generic
        // resident estimate or caller-supplied evidence run first would turn Evidence telemetry
        // into a legacy bypass.
        for candidate in &admission.evidence {
            let exact = &candidate.evidence;
            if attributable_resident_bytes > exact.predicted_peak_bytes {
                continue;
            }
            let candidate_budget = Budget {
                available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
                    / BYTES_PER_GIB,
                reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
                total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
                reserved_headroom_gb: candidate.foreign_reserve_bytes as f64 / BYTES_PER_GIB,
            };
            // sc-18096: this pre-check runs against the candidate's CAPTURED foreign reserve,
            // which the selector's uniform zero-reserve Evidence budget cannot carry. It grades the
            // measured peak exactly, as the selector does (sc-22738: no widening for a moved
            // closure).
            if candidate_budget.effective_gb().is_some_and(|available| {
                exact
                    .predicted_peak_bytes
                    .saturating_add(reference_workspace_bytes) as f64
                    / BYTES_PER_GIB
                    <= available
            }) {
                selections.push(MemorySelection {
                    strategy: exact.key.strategy,
                    parameters: exact.key.parameters,
                    tier: exact.key.tier,
                });
                evidence.push(exact);
                candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
                // A measured envelope is one observed number; it does not decompose.
                candidate_activation_bytes.push(None);
                candidate_charges_generic_headroom.push(false);
                candidate_precheck_available_gb.push(candidate_budget.effective_gb());
            }
        }
        if evidence.is_empty() {
            let minimum_required_host = admission
                .evidence
                .iter()
                // The smallest host that satisfies the reserve policy. The current-host enforced
                // sum is a useful diagnostic but not a portable minimum: the reserve changes when
                // the host capacity changes.
                .map(|candidate| candidate.minimum_host_bytes)
                .min()
                .unwrap_or(0);
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            return Ok(MlxRequestAdmission::Rejected(WorkerError::InvalidPayload(
                format!(
                    "{} request {}x{} count {} needs at least {:.2} GiB at its smallest verified \
                 MLX host boundary, but no exact candidate fits the live unified-memory budget\
                 {alternative}",
                    plan.model_id,
                    inputs.width,
                    inputs.height,
                    inputs.count.max(1),
                    minimum_required_host as f64 / BYTES_PER_GIB,
                ),
            )));
        }
    } else {
        use crate::memory_strategy::CandidateBasis;

        let capacity = 1 + additional_evidence.len() + synthesized_estimates.len();
        selections.reserve(capacity);
        evidence.reserve(capacity);
        candidate_bases.reserve(capacity);
        candidate_activation_bytes.reserve(capacity);
        candidate_charges_generic_headroom.reserve(capacity);
        candidate_precheck_available_gb.resize(capacity, None);
        selections.push(resident_selection);
        evidence.push(&resident);
        // The resident baseline IS the rung-0 weights+headroom floor estimate (sc-18096): its
        // peak source is unchanged, but it is now graded behind the estimate margin like every
        // other unmeasured candidate instead of at its raw guess.
        candidate_bases.push(CandidateBasis::EstimateFloor);
        // ...but its PEAK does not decompose here, so it declares no activation term (sc-22508).
        // This candidate's peak is `predicted_memory_peak_from_base(contract_base_peak_bytes(
        // request_total_peak_bytes(..)))`. Two of those three
        // steps can destroy the weights+headroom shape: `request_total_peak_bytes` returns the
        // `mage_flow` provider's own `generation_peak_gb` scalar on that route, and
        // `predicted_memory_peak_from_base` is a gen-core trait method the loaded provider owns
        // and may rewrite arbitrarily. Naming `generic_headroom_bytes` here would have declared a
        // term the peak need not contain — on the mage route, a term the peak DEMONSTRABLY does
        // not contain. `None` takes the policy's undeclared-floor arm, whose doc says the fix for
        // a lane that wants better is to declare a real split, not to widen a number.
        candidate_activation_bytes.push(None);
        candidate_charges_generic_headroom.push(!resident_peak_is_provider_scalar);
        selections.extend(additional_evidence.iter().map(|item| MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        }));
        evidence.extend(additional_evidence);
        candidate_bases.extend(additional_evidence.iter().map(|_| CandidateBasis::Measured));
        candidate_activation_bytes.extend(additional_evidence.iter().map(|_| None));
        candidate_charges_generic_headroom.extend(additional_evidence.iter().map(|_| false));
        for estimate in synthesized_estimates {
            selections.push(estimate.selection);
            evidence.push(&estimate.evidence);
            candidate_bases.push(estimate.basis);
            candidate_activation_bytes.push(estimate.unmodeled_activation_bytes);
            candidate_charges_generic_headroom
                .push(estimate.activation_term == Some(ImageFloorActivationTerm::GenericHeadroom));
        }
    }
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    debug_assert_eq!(evidence.len(), candidate_activation_bytes.len());
    debug_assert_eq!(evidence.len(), candidate_charges_generic_headroom.len());
    // sc-25909 — OWNER-APPROVED E6 EXCEPTION (Michael, 2026-10-16): a reference request on a rung
    // whose activation term is the generic area-scaled headroom is priced by the provider's own
    // cache-shed request transient (`derived::request_transient_budget_bytes`, the bound the
    // generator holds the render to) wherever that exceeds the headroom's activation slice. At
    // <=1024^2 with many references on a packed tier (the q8/q4 DiT computes in f32) the generic
    // figure ignores reference count and dtype width and under-prices the render, so the rung's
    // priced activation becomes max(generic activation slice, derived transient) + the fixed
    // reserve. This deliberately grades the SHED ladder — prefill path, cached path and the
    // pre-load tier probe alike — so a request admitted before the epic may now shed to a further
    // rung or be refused with the ordinary "needs … GiB" reason. The correction itself never
    // changes the tier, the references or the size. Its tier consequence is the EXISTING one: a
    // request with no explicit tier pick whose requested tier no longer fits the probe goes through
    // the pre-existing downtier ladder (`downtier_candidate_tiers` / `choose_mlx_request_tier`,
    // bounded by `min_quality_floor`), exactly as any other over-budget request does; an explicit
    // tier pick is refused.
    // The generic headroom carries the held-back fixed OS/app reserve, which is not activation, so
    // only its activation slice is compared (as the floor arm's `unmodeled_activation_bytes` does).
    // Where the derived transient is at or under that slice (bf16, no references) the shortfall is
    // zero and the request is graded byte for byte as before.
    let generic_headroom_shortfall = |rung: &MemorySelection| -> u64 {
        if admission.path != AdmissionPath::Legacy || geometry.reference_count == 0 {
            return 0;
        }
        let forced_tile_edge = contract
            .engages_selection(rung, MemoryStrategy::BoundedDecode)
            .then_some(rung.parameters.decode_tile_edge)
            .flatten();
        let generic_activation_bytes = plan
            .generic_headroom_bytes(geometry)
            .saturating_sub(plan.fixed_reserve_bytes.min(plan.activation_headroom_bytes));
        prefix_kv_cache_shed_transient(forced_tile_edge).map_or(0, |transient| {
            transient.saturating_sub(generic_activation_bytes)
        })
    };
    let corrected_evidence = selections
        .iter()
        .zip(&evidence)
        .zip(&candidate_charges_generic_headroom)
        .map(|((selection, evidence), charges_generic_headroom)| {
            let shortfall = if *charges_generic_headroom {
                generic_headroom_shortfall(selection)
            } else {
                0
            };
            let request_extra = shortfall.saturating_add(reference_workspace_bytes);
            (request_extra > 0).then(|| {
                let mut corrected = (*evidence).clone();
                corrected.predicted_peak_bytes =
                    corrected.predicted_peak_bytes.saturating_add(request_extra);
                (corrected, shortfall)
            })
        })
        .collect::<Vec<_>>();
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(&candidate_bases)
        .zip(&candidate_activation_bytes)
        .zip(&corrected_evidence)
        .map(
            |((((selection, evidence), basis), activation), corrected)| Candidate {
                selection: *selection,
                evidence: corrected
                    .as_ref()
                    .map_or(evidence, |(corrected, _)| corrected),
                basis: *basis,
                // sc-22508: carried from the site that BUILT each peak (see
                // `candidate_activation_bytes`), never inferred from the basis label here. A
                // `CandidateBasis::EstimateFloor` label says how much evidence stands behind a
                // number; it does not say the number is `weights + generic_headroom_bytes`.
                // sc-25909: a corrected rung's activation carries its shortfall too.
                unmodeled_activation_bytes: match corrected {
                    Some((_, shortfall)) => {
                        activation.map(|bytes| bytes.saturating_add(*shortfall))
                    }
                    None => *activation,
                },
            },
        )
        .collect::<Vec<_>>();
    let request_scope = RequestScope {
        resolved_route: plan.engine_id,
        backend: "mlx",
        tier: plan.tier,
        mode: mode_key,
        overlay: inputs.overlay.as_deref(),
        geometry,
    };
    let selector_budget = Some(Budget {
        available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
            / BYTES_PER_GIB,
        reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
        total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
        reserved_headroom_gb: if admission.path == AdmissionPath::Evidence {
            0.0
        } else {
            budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB
        },
    });
    let resident_allowance_credit = if admission.path == AdmissionPath::Legacy {
        if phase == MlxAdmissionPhase::TierProbe && load_policy == OffloadPolicy::Resident {
            // A cold probe charges the FULL pipeline and gives no live allocation credit. Its
            // Resident uncertainty allowance must nevertheless exclude the declared constructor
            // weights, just as the loaded gate excludes weights already present. Otherwise moving
            // this decision before load inflates the allowance and falsely rejects fitting tiers.
            // This affects only Resident's margin, never the peak/budget or optimized candidates;
            // the constructor floor and final live-allocation check still apply independently.
            contract.total_resident_bytes()
        } else {
            attributable_resident_bytes
        }
    } else {
        0
    };
    // Epic 25899: the provider's prefix KV cache, added rung-for-rung. The whole ladder below —
    // baseline and warm-policy floor alike — runs first at the shed demand, which is exactly the
    // pre-cache admission and picks the rung; the cache is then taken only if that SAME rung
    // (`only_rung`) also fits with the surcharge on its peak. The cache never moves a request to
    // another rung. A provider with no cache prices `None` and runs the ladder once, unchanged.
    let run_ladder = |extra_bytes: u64,
                      only_rung: Option<MemorySelection>|
     -> (Selection, Option<crate::execution_planner::GrantOutcome>) {
        let surcharged_evidence = (extra_bytes > 0).then(|| {
            candidates
                .iter()
                .map(|candidate| {
                    let mut evidence = candidate.evidence.clone();
                    evidence.predicted_peak_bytes =
                        evidence.predicted_peak_bytes.saturating_add(extra_bytes);
                    evidence
                })
                .collect::<Vec<_>>()
        });
        let surcharged_candidates = surcharged_evidence.as_ref().map(|surcharged| {
            candidates
                .iter()
                .zip(surcharged)
                .zip(&candidate_precheck_available_gb)
                // The Evidence path's captured-reserve pre-check, re-run at the surcharged peak:
                // a measured cell that fitted shed but not cached is not a cached candidate.
                .filter(|((_, evidence), precheck)| {
                    precheck.is_none_or(|available| {
                        evidence.predicted_peak_bytes as f64 / BYTES_PER_GIB <= available
                    })
                })
                .map(|((candidate, evidence), _)| Candidate {
                    evidence,
                    ..*candidate
                })
                .collect::<Vec<_>>()
        });
        let pass_candidates: &[Candidate<'_>] =
            surcharged_candidates.as_deref().unwrap_or(&candidates);
        let same_rung;
        let pass_candidates: &[Candidate<'_>] = match only_rung {
            Some(rung) => {
                same_rung = pass_candidates
                    .iter()
                    .filter(|candidate| candidate.selection == rung)
                    .copied()
                    .collect::<Vec<_>>();
                &same_rung
            }
            None => pass_candidates,
        };
        let baseline = crate::memory_strategy::select_strategy_with_resident_credit(
            request_scope,
            contract,
            selector_budget,
            pass_candidates,
            resident_allowance_credit,
        );
        // sc-18317: a GRANTED warm-policy switch takes effect HERE or nowhere.
        //
        // `select_strategy` picks the first fitting candidate in resident -> staged -> bounded-decode
        // order, so a request whose weights fit resident never stages no matter what policy the caller
        // asked for. Honoring a grant therefore means removing the looser candidates from the set: offer
        // only those that engage `StagedResidency` and let the selector apply the SAME budget and margins
        // it always does. A floored selection that comes back `Selected` has passed exactly the fit check
        // the baseline would have had to pass, so it cannot exceed the admitted envelope — monotonicity is
        // preserved by construction rather than asserted after the fact.
        //
        // The outcome (not the seam's proposal) is what gets reported: `rematerialized` is emitted only
        // when the floor actually moved the selection.
        let outcome = if warm_policy.requires_staged_residency() {
            let staged = pass_candidates
                .iter()
                .filter(|candidate| {
                    contract
                        .engages_selection(&candidate.selection, MemoryStrategy::StagedResidency)
                })
                .cloned()
                .collect::<Vec<_>>();
            let baseline_stages = matches!(&baseline, Selection::Selected { selection, .. }
                if contract.engages_selection(selection, MemoryStrategy::StagedResidency));
            if baseline_stages {
                (
                    baseline,
                    Some(crate::execution_planner::GrantOutcome::AlreadyStaged),
                )
            } else if staged.is_empty() {
                (
                    baseline,
                    Some(crate::execution_planner::GrantOutcome::NoStagedCandidate),
                )
            } else {
                let floored = crate::memory_strategy::select_strategy(
                    request_scope,
                    contract,
                    selector_budget,
                    &staged,
                );
                match &floored {
                    // A fitting staged selection, and it necessarily differs from the baseline's: control
                    // only reaches here when the baseline does NOT engage staged residency, while every
                    // candidate offered to this call does. Asserted rather than re-tested, so the
                    // reasoning is checked in debug builds instead of hidden in an unreachable arm.
                    Selection::Selected {
                        selection: chosen, ..
                    } => {
                        debug_assert!(
                            !matches!(&baseline, Selection::Selected { selection: base, .. } if base == chosen),
                            "a floored staged selection cannot equal a non-staging baseline"
                        );
                        (
                            floored,
                            Some(crate::execution_planner::GrantOutcome::SelectionMovedToStaged),
                        )
                    }
                    // The staged candidates did not fit this request's budget, or their evidence is
                    // structurally unusable. The baseline stands; not a refusal, just the ordinary fit
                    // check every selection passes.
                    _ => (
                        baseline,
                        Some(crate::execution_planner::GrantOutcome::StagedCandidateDidNotFit),
                    ),
                }
            }
        } else {
            (baseline, None)
        };
        outcome
    };
    let ((selection, grant_outcome), prefix_kv_cache) =
        crate::execution_planner::plan_prefix_kv_cache(
            prefix_kv_cache_surcharge,
            run_ladder(0, None),
            |(selection, _)| matches!(selection, Selection::Selected { .. }),
            |(shed, grant_outcome), priced_bytes| {
                let Selection::Selected {
                    selection: rung, ..
                } = shed
                else {
                    return None;
                };
                // The rung's peak already carries its sc-25909 transient shortfall (corrected
                // above), so the cached path is that graded shed rung + the cache surcharge.
                let (cached, _) = run_ladder(priced_bytes, Some(*rung));
                // The shed pass's warm-policy outcome stands: it decided the rung.
                matches!(&cached, Selection::Selected { selection, .. } if selection == rung)
                    .then_some((cached, *grant_outcome))
            },
        );
    // The demand a cached selection was graded at on top of its (corrected) shed peak.
    let prefix_kv_cache_extra = prefix_kv_cache.map_or(0, |plan| plan.admitted_surcharge_bytes());
    warm_policy.settle_with_selection(
        grant_outcome.unwrap_or(crate::execution_planner::GrantOutcome::AlreadyStaged),
    );
    let (selection, mut needed_gb, mut available_gb) = match selection {
        Selection::Selected {
            selection,
            needed_gb,
            available_gb,
        } => (selection, needed_gb, available_gb),
        Selection::Reject {
            needed_gb,
            available_gb,
        } => {
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            return Ok(MlxRequestAdmission::Rejected(WorkerError::InvalidPayload(format!(
                "{} request {}x{} count {} needs {:.2} GiB for the complete pipeline but only {:.2} GiB is safely available \
                 ({:.2} GiB total, {:.2} GiB unrelated active allocations, {:.2} GiB reserved){}",
                plan.engine_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
                needed_gb,
                available_gb,
                budget.total_bytes as f64 / BYTES_PER_GIB,
                budget.committed_bytes as f64 / BYTES_PER_GIB,
                budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB,
                alternative,
            ))));
        }
        // sc-18096 retired the "no measured evidence ⇒ refuse" meaning of this arm: every
        // implemented rung of a legacy route now carries an estimate-backed candidate, so the
        // selector only lands here when the evidence is STRUCTURALLY invalid (contract
        // conformance errors, composition mismatch, an unresolvable budget, or a rung whose
        // declared parameters cannot form a valid selection). That remains a refusal — the other
        // permitted refusal, "no rung fits with margins", is the `Reject` arm above.
        Selection::Unverified { reason } => {
            let alternative = admission
                .lower_alternative
                .as_ref()
                .map(|alternative| {
                    format!(
                        "; current verified alternative: {}x{}",
                        alternative.geometry.width, alternative.geometry.height
                    )
                })
                .unwrap_or_default();
            let message = format!(
                "{} request {}x{} count {} has no structurally admissible MLX memory strategy \
                 ({reason:?}); refusing to enter MLX's process-terminating allocation \
                 path{alternative}",
                plan.engine_id,
                inputs.width,
                inputs.height,
                inputs.count.max(1),
            );
            return Err(WorkerError::InvalidPayload(message));
        }
    };
    let selected_basis = candidates
        .iter()
        .find(|candidate| candidate.selection == selection)
        .map(|candidate| candidate.basis)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{} selected a strategy with no candidate provenance",
                plan.engine_id
            ))
        })?;
    let optimization_authority = if !selection.strategy.is_optimized() {
        MemoryOptimizationAuthority::Resident
    } else if selected_basis.is_estimate() {
        MemoryOptimizationAuthority::Estimated
    } else {
        MemoryOptimizationAuthority::Calibrated
    };
    let mut selected_record_id = None;
    let mut process_limit_bytes = admission.process_limit_bytes;
    let predicted_peak_bytes = if admission.path == AdmissionPath::Evidence {
        let index = admission
            .evidence
            .iter()
            .position(|candidate| {
                let evidence = &candidate.evidence;
                evidence.key.strategy == selection.strategy
                    && evidence.key.parameters == selection.parameters
                    && evidence.key.tier == selection.tier
            })
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "{} selected a strategy without matching verified MLX evidence",
                    plan.engine_id
                ))
            })?;
        let candidate = &admission.evidence[index];
        let evidence = &candidate.evidence;
        let reserve = candidate.foreign_reserve_bytes;
        let selected_budget = Budget {
            available_gb: budget.total_bytes.saturating_sub(budget.committed_bytes) as f64
                / BYTES_PER_GIB,
            reclaimable_gb: budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
            total_gb: budget.total_bytes as f64 / BYTES_PER_GIB,
            reserved_headroom_gb: reserve as f64 / BYTES_PER_GIB,
        };
        needed_gb = evidence
            .predicted_peak_bytes
            .saturating_add(reference_workspace_bytes)
            .saturating_add(prefix_kv_cache_extra)
            .saturating_add(reserve) as f64
            / BYTES_PER_GIB;
        available_gb = selected_budget.effective_gb().unwrap_or(0.0);
        budget.reserved_headroom_bytes = reserve;
        process_limit_bytes = Some(budget.total_bytes.saturating_sub(reserve));
        selected_record_id = Some(candidate.record_id.clone());
        evidence
            .predicted_peak_bytes
            .saturating_add(reference_workspace_bytes)
    } else if selection == resident_selection
        || synthesized_estimates
            .iter()
            .any(|estimate| estimate.selection == selection)
    {
        // Keep the selected full peak paired with the normalized budget used by selection: the
        // peak of the candidate the ladder GRADED (the same lookup `selected_basis` uses), which
        // carries its sc-25909 transient shortfall where the rung took one — one source of truth.
        candidates
            .iter()
            .find(|candidate| candidate.selection == selection)
            .map_or(
                predicted_peak_bytes.saturating_add(reference_workspace_bytes),
                |candidate| candidate.evidence.predicted_peak_bytes,
            )
    } else {
        predicted_peak_bytes.saturating_add(reference_workspace_bytes)
    };
    // The demand the admission approved: a cached selection was graded at its (corrected) shed
    // peak + the cache surcharge (`prefix_kv_cache_extra`).
    let predicted_peak_bytes = predicted_peak_bytes.saturating_add(prefix_kv_cache_extra);
    tracing::info!(
        event = if phase == MlxAdmissionPhase::TierProbe {
            "memory_strategy_tier_candidate_selected"
        } else {
            "memory_strategy_request_selected"
        },
        route = plan.engine_id,
        backend = "mlx",
        tier = ?plan.tier,
        mode = mode_key,
        overlay = inputs.overlay.as_deref().unwrap_or("none"),
        width = inputs.width,
        height = inputs.height,
        count = inputs.count.max(1),
        cache_state = ?cache_state,
        load_policy = ?load_policy,
        strategy = ?selection.strategy,
        cache_eviction = contract.engages_selection(
            &selection,
            MemoryStrategy::StagedResidency,
        ),
        parameters = ?selection.parameters,
        evidence_record_id = selected_record_id.as_deref().unwrap_or("none"),
        predicted_peak_bytes,
        effective_budget_bytes = budget.effective_bytes(),
        needed_gb,
        available_gb,
        optimization_authority = ?optimization_authority,
        prefix_kv_cache = prefix_kv_cache.map(|plan| plan.mode.label()),
        prefix_kv_cache_priced_bytes = prefix_kv_cache.map(|plan| plan.priced_bytes),
        "selected request-scoped MLX memory strategy"
    );
    let mut memory = memory_for_selection(contract, selection);
    if let Some(plan) = prefix_kv_cache {
        plan.write_onto(&mut memory);
    }
    let evaluation = MlxRequestEvaluation {
        memory,
        prefix_kv_cache,
        process_limit_bytes,
        decode_quality_decisions: synthesized_ladder.decode_quality_decisions,
        context: MemoryRunContext {
            selection,
            optimization_authority,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
            load_shape: contract.load_shape,
            mode,
            has_reference: inputs.has_reference,
            use_pid: inputs.use_pid,
            has_phases: inputs.has_phases,
            geometry,
            reference_resources: inputs.reference_resources.clone(),
            overlay: inputs.overlay.clone(),
            budget,
            predicted_peak_bytes,
            cache_state,
            evidence_revision: selected_record_id
                .or(admission.evidence_revision)
                .unwrap_or_else(|| REQUEST_EVIDENCE_REVISION.to_owned()),
        },
    };
    tracing::info!(
        event = if phase == MlxAdmissionPhase::TierProbe {
            "mlx_decode_quality_tier_candidate_audit"
        } else {
            "mlx_decode_quality_request_audit"
        },
        route = plan.engine_id,
        decisions = ?evaluation.decode_quality_decisions,
        "recorded typed request-scoped decode-quality decisions"
    );
    Ok(MlxRequestAdmission::Admitted(Box::new(evaluation)))
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn live_request_budget(engine_id: &str) -> WorkerResult<MemoryBudget> {
    // Reclaim only allocator-cache buffers from prior geometries; live arrays (the cached weights)
    // remain committed. This makes A→B→A independent of B's freed scratch without reloading A.
    mlx_rs::memory::clear_cache();
    let committed_bytes = mlx_rs::memory::get_active_memory() as u64;
    let reclaimable_bytes = mlx_rs::memory::get_cache_memory() as u64;
    let (total_bytes, reserved_headroom_bytes) = if engine_id.starts_with("mage_flow") {
        let safe_gb = runtime_macos::providers::mage::memory::production_safe_budget_gb()
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        (decimal_gb_to_bytes(safe_gb), 0)
    } else {
        let total_gib = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "MLX unified-memory budget is unavailable; refusing an unbounded request"
                        .to_owned(),
                )
            })?
            .total_gb;
        let reserve = crate::fit_gate::legacy_unified_reserve(total_gib);
        (gib_to_bytes(total_gib), gib_to_bytes(reserve.gb))
    };
    Ok(MemoryBudget {
        total_bytes,
        committed_bytes,
        reclaimable_bytes,
        reserved_headroom_bytes,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn request_total_peak_bytes(plan: &MlxRequestPlan, geometry: MemoryGeometry) -> u64 {
    if plan.engine_id.starts_with("mage_flow") {
        decimal_gb_to_bytes(runtime_macos::providers::mage::memory::generation_peak_gb(
            plan.tier.quant,
            geometry.width,
            geometry.height,
            geometry.batch,
        ))
    } else {
        plan.generic_total_peak_bytes(geometry)
    }
}

/// Preserve the constructor's weights floor for eager/unadopted loads. Sequential specs have
/// already been authorized by the exact declaration or the existing load gate. This diagnostic
/// describes the limiting load working set, not the unrelated resident activation estimate.
#[cfg(target_os = "macos")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn preflight_load_rejection(engine_id: &str, spec: &LoadSpec) -> Option<WorkerError> {
    if spec.offload_policy == OffloadPolicy::Sequential {
        return None;
    }
    let ResidencyOutcome::Reject { available_gb, .. } = decide_residency_for_spec(engine_id, spec)
    else {
        return None;
    };
    let (total, text, _) = spec_component_bytes(engine_id, spec);
    let staged = engine_supports_sequential(engine_id);
    let weights = if staged {
        staged_weights_gb(total, text)
    } else {
        total as f64 / BYTES_PER_GIB
    };
    let reserve = crate::fit_gate::legacy_unified_reserve(available_gb).gb;
    Some(WorkerError::InvalidPayload(format!(
        "{engine_id} cannot load this tier: its {} model weights need {weights:.2} GiB, \
         but only {:.2} GiB is available after the OS reserve ({available_gb:.2} GiB total). \
         No supported deferred load was selected; use a smaller installed tier or a Mac with more memory.",
        if staged { "largest simultaneous component set of" } else { "resident" },
        (available_gb - reserve).max(0.0),
    )))
}

/// Evaluate a fully prepared candidate without constructing a generator or allocating tensors.
/// The registry contract is queried against the same spec handed to the cache on selection.
#[cfg(target_os = "macos")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn preflight_request(
    spec: &LoadSpec,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    budget: MemoryBudget,
) -> WorkerResult<MlxRequestAdmission> {
    // Fail closed before deriving a contract for this actual candidate spec. The selector also
    // resolves the same pure provider bound before grading, including every warm-cache request.
    crate::memory_strategy::reference_workspace_bytes(
        plan.engine_id,
        spec,
        inputs.reference_resources.as_ref(),
    )?;
    let contract = crate::inference_runtime::media()
        .memory_strategy_contract(plan.engine_id, spec)
        .map_err(|error| {
            crate::classify_engine_error("MLX pre-load memory contract failed", error)
        })?;
    preflight_request_with_contract(
        contract.as_ref(),
        spec.offload_policy,
        plan,
        inputs,
        budget,
        Some(ReferencePricing {
            spec,
            registry: crate::inference_runtime::media(),
        }),
    )
}

/// [`preflight_request`] once the registry contract is resolved — split out so the tier probe's
/// pricing is testable against a fixture contract without a `LoadSpec` or the linked registry.
#[cfg(target_os = "macos")]
pub(super) fn preflight_request_with_contract(
    contract: Option<&MemoryProviderContract>,
    load_policy: OffloadPolicy,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    budget: MemoryBudget,
    reference_pricing: Option<ReferencePricing<'_>>,
) -> WorkerResult<MlxRequestAdmission> {
    select_request_with_budget_using_bundle(
        MlxAdmissionPhase::TierProbe,
        declared_component_floors(plan.engine_id),
        contract,
        None,
        // sc-25909: the probe grades exactly the shed demand, which now carries the provider
        // transient shortfall — a tier the loaded gate would refuse is not probed as fitting.
        &|forced_tile_edge| {
            mlx_prefix_kv_cache_shed_transient_bytes(
                plan.engine_id,
                plan.tier.quant,
                inputs,
                forced_tile_edge,
            )
        },
        |base| {
            contract.map_or(
                gen_core::MemoryPeakBreakdown::from_unattributed(base),
                |contract| contract.predicted_peak_from_base(base),
            )
        },
        plan,
        inputs,
        MemoryCacheState::Cold,
        load_policy,
        crate::execution_planner::WarmPolicyProposal::inert(plan.engine_id),
        budget,
        request_total_peak_bytes(plan, request_geometry(inputs)),
        0,
        &[],
        None,
        reference_pricing,
    )
}

/// Evaluate one real MLX request after cache lookup and immediately before generation.
#[cfg(target_os = "macos")]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn evaluate_request(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    external_committed_bytes: u64,
) -> WorkerResult<MlxRequestEvaluation> {
    if inputs.reference_resources.is_some() {
        return Err(missing_reference_pricing(plan.engine_id));
    }
    let geometry = request_geometry(inputs);
    let budget = live_request_budget(plan.engine_id)?;
    let total_peak_bytes = request_total_peak_bytes(plan, geometry);
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        &[],
        None,
    )
}

/// Loaded FLUX.3 reference request: price against the same resolved spec used for this cache lookup.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn evaluate_request_with_reference_spec(
    generator: &dyn gen_core::Generator,
    plan: &MlxRequestPlan,
    inputs: &MlxRequestInputs,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    warm_policy: crate::execution_planner::WarmPolicyProposal,
    external_committed_bytes: u64,
    reference_spec: &LoadSpec,
) -> WorkerResult<MlxRequestEvaluation> {
    let geometry = request_geometry(inputs);
    let budget = live_request_budget(plan.engine_id)?;
    let total_peak_bytes = request_total_peak_bytes(plan, geometry);
    evaluate_request_with_budget_and_warm_policy(
        generator,
        plan,
        inputs,
        cache_state,
        load_policy,
        warm_policy,
        budget,
        total_peak_bytes,
        external_committed_bytes,
        &[],
        Some(ReferencePricing {
            spec: reference_spec,
            registry: crate::inference_runtime::media(),
        }),
    )
}
