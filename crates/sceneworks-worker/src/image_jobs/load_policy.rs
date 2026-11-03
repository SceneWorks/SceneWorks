#[cfg(not(target_os = "macos"))]
use super::load_spec::apply_declared_candle_image_load_shape;
use super::LoadSpec;
#[cfg(not(target_os = "macos"))]
use crate::JsonObject;
#[cfg(target_os = "macos")]
use crate::WorkerResult;
/// Unchanged legacy shaper for bespoke Candle call sites that do not carry a complete manifest
/// declaration coordinate. Declaration-owned routes use [`apply_declared_candle_image_load_shape`]
/// only after assembling their exact route, tier, mode, source, components, and selected encoder.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn apply_candle_image_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Candle,
        engine_id,
        crate::memory_route_registry::MemoryRouteMode::TextToImage,
        spec,
        false,
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn apply_candle_qwen_load_shape(
    engine_id: &str,
    resolved_tier: &str,
    request_mode: &str,
    manifest: &JsonObject,
    spec: LoadSpec,
) -> LoadSpec {
    apply_declared_candle_image_load_shape(
        engine_id,
        Some(resolved_tier),
        request_mode,
        manifest,
        spec,
        false,
    )
}

/// The Candle shared lane's prefix-KV-cache surcharge for one request (epic 25899): priced from the
/// LINKED provider's descriptor and pricing function, or `None` for every provider without a cache
/// and for a request whose admitted block is not handed to ONE render at exactly this geometry and
/// reference set (`single_pass == false`: hires fix, native prompt batch).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_prefix_kv_cache_surcharge(
    engine_id: &str,
    width: u32,
    height: u32,
    reference_count: u32,
    negative_branch: bool,
    single_pass: bool,
) -> Option<u64> {
    if !single_pass {
        return None;
    }
    crate::inference_runtime::media_descriptor(engine_id).and_then(|descriptor| {
        crate::execution_planner::prefix_kv_cache_surcharge_bytes(
            engine_id,
            &descriptor.capabilities.execution,
            None,
            width,
            height,
            reference_count,
            negative_branch,
        )
    })
}

/// Finalize the manifest/provider load-shape intersection before pricing or loading a tier.
/// Shared by production and real-artifact admission tests so neither can force an unshipped strategy.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_mlx_load_policy(
    engine_id: &str,
    effective_tier: Option<&str>,
    declaration_mode: Option<crate::memory_route_registry::MemoryRouteMode>,
    manifest: &serde_json::Map<String, serde_json::Value>,
    mut spec: LoadSpec,
    declaration_context: crate::memory_route_registry::MemoryRouteRequestContext,
    plain_text_to_image: bool,
    supports_sequential_offload: bool,
) -> WorkerResult<LoadSpec> {
    spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
        engine_id,
        effective_tier,
        declaration_mode,
        manifest,
        spec,
        declaration_context,
    );
    spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
        engine_id,
        effective_tier,
        declaration_mode,
        manifest,
        spec,
        declaration_context,
    );
    if let Some(warning) = crate::memory_route_registry::mlx_load_shape_declaration_warning(&spec) {
        tracing::warn!(
            event = "mlx_load_shape_declaration_warning",
            provider = engine_id,
            ?warning,
            "provider refused deferred materialization; retaining the safe eager load path"
        );
    }
    if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::NotEvaluated {
        spec = apply_measured_mlx_load_shape_for_request(engine_id, spec, plain_text_to_image);
    }
    if spec.offload_policy != gen_core::OffloadPolicy::Sequential {
        let outcome = crate::mlx_fit_gate::decide_residency_for_spec(engine_id, &spec);
        let deferred_can_stage = spec.load_shape_declaration_result
            == gen_core::LoadShapeDeclarationResult::Applied
            && spec.load_shape == gen_core::LoadShape::DeferredMaterialization
            && supports_sequential_offload;
        if deferred_can_stage && !matches!(outcome, crate::mlx_fit_gate::ResidencyOutcome::Resident)
        {
            spec = spec.with_offload_policy(gen_core::OffloadPolicy::Sequential);
            spec = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
                engine_id,
                effective_tier,
                declaration_mode,
                manifest,
                spec,
                declaration_context,
            );
            spec = crate::memory_route_registry::apply_declared_mlx_load_policy_for_request(
                engine_id,
                effective_tier,
                declaration_mode,
                manifest,
                spec,
                declaration_context,
            );
        } else if matches!(outcome, crate::mlx_fit_gate::ResidencyOutcome::Sequential) {
            spec = crate::mlx_fit_gate::apply_residency_policy(spec, engine_id)?;
        }
    }
    Ok(spec)
}

/// The eager Resident counterpart of a spec that [`prepare_mlx_load_policy`] made Sequential only
/// because the manifest's declaration row requires it (sc-24806). `None` unless the policy step
/// introduced Sequential and the fit gate admits the eager spec Resident. The declaration stays
/// `Eligible`, so generic shapers keep ignoring it and only a fresh exact evaluation can apply it.
#[cfg(target_os = "macos")]
pub(crate) fn declared_sequential_resident_alternative(
    engine_id: &str,
    before: &LoadSpec,
    after: &LoadSpec,
) -> Option<LoadSpec> {
    if after.offload_policy != gen_core::OffloadPolicy::Sequential
        || before.offload_policy == gen_core::OffloadPolicy::Sequential
        || !matches!(
            after.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Applied
                | gen_core::LoadShapeDeclarationResult::Eligible
        )
    {
        return None;
    }
    matches!(
        crate::mlx_fit_gate::decide_residency_for_spec(engine_id, before),
        crate::mlx_fit_gate::ResidencyOutcome::Resident
    )
    .then(|| before.clone().with_eligible_load_shape_declaration())
}

/// Whether an admitted request runs without whole-component staging or transformer block streaming.
#[cfg(target_os = "macos")]
pub(super) fn admission_needs_no_staging(
    admission: &crate::mlx_fit_gate::MlxRequestAdmission,
) -> bool {
    matches!(
        admission,
        crate::mlx_fit_gate::MlxRequestAdmission::Admitted(evaluation)
            if !evaluation.memory.stage_residency && !evaluation.memory.stream_transformer_blocks
    )
}

/// sc-24806 for the MLX lanes that bind the declared load policy themselves (the base lane does
/// the same inside its tier chooser): keep `declared` unless both it and its eager Resident
/// counterpart are admitted for this request without staging, and the Resident load passes the
/// load gate. A failed pre-load probe keeps `declared`, whose own admission still runs at load.
#[cfg(target_os = "macos")]
pub(crate) async fn release_declared_sequential_for_request(
    engine_id: &'static str,
    model_id: &str,
    before: &LoadSpec,
    declared: LoadSpec,
    plan: crate::mlx_fit_gate::MlxRequestPlan,
    inputs: &crate::mlx_fit_gate::MlxRequestInputs,
    rebuild_plan: impl FnOnce(&LoadSpec) -> WorkerResult<crate::mlx_fit_gate::MlxRequestPlan>,
) -> WorkerResult<(LoadSpec, crate::mlx_fit_gate::MlxRequestPlan)> {
    let Some(resident) = declared_sequential_resident_alternative(engine_id, before, &declared)
    else {
        return Ok((declared, plan));
    };
    let budget = crate::generator_cache::mlx_tier_budget(engine_id).await?;
    let admits = |spec: &LoadSpec, plan: &crate::mlx_fit_gate::MlxRequestPlan| {
        crate::mlx_fit_gate::preflight_request(spec, plan, inputs, budget)
            .is_ok_and(|admission| admission_needs_no_staging(&admission))
    };
    if !admits(&declared, &plan) {
        return Ok((declared, plan));
    }
    let resident_plan = rebuild_plan(&resident)?;
    if admits(&resident, &resident_plan)
        && crate::mlx_fit_gate::preflight_load_rejection(engine_id, &resident).is_none()
    {
        tracing::info!(
            event = "mlx_declared_sequential_released_to_resident",
            engine = %engine_id,
            model = %model_id,
            "request needs no staging; loading Resident instead of the declared Sequential policy"
        );
        return Ok((resident, resident_plan));
    }
    Ok((declared, plan))
}

/// Select deferred materialization for MLX routes with a measured load-exact contract. The fit gate
/// still owns the independent Resident/Sequential decision; a constrained request adds Sequential
/// and can then select the provider's bounded transformer rung. Lens has two deliberately separate
/// measured identities at the current provider pin: base Lens Q4 exposes the full ladder, while
/// Lens-Turbo BF16 retains its legacy text-encoder-only rung. Qwen base/edit also cover Q4/Q8 and
/// forward-time adapters because the block stream replays packed quantization and captured residuals.
/// Request-aware callers use the private helper below so Krea and SDXL base calibration is never
/// applied to reference, edit, or Hires.fix surfaces that share the same engine and weight shape.
#[cfg(target_os = "macos")]
pub(crate) fn apply_measured_mlx_load_shape(engine_id: &str, spec: LoadSpec) -> LoadSpec {
    apply_measured_mlx_load_shape_for_request(engine_id, spec, true)
}

#[cfg(target_os = "macos")]
pub(super) fn apply_measured_mlx_load_shape_for_request(
    engine_id: &str,
    spec: LoadSpec,
    plain_text_to_image: bool,
) -> LoadSpec {
    let mode = if plain_text_to_image {
        crate::memory_route_registry::MemoryRouteMode::TextToImage
    } else {
        crate::memory_route_registry::MemoryRouteMode::EditImage
    };
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Mlx,
        engine_id,
        mode,
        spec,
        false,
    )
}

# [cfg (native_lane)]
# [cfg (all (test , not (target_os = "macos") , feature = "backend-candle"))]
#[path="tests/candle_prefix_kv_cache_pricing_tests.rs"]
mod candle_prefix_kv_cache_pricing_tests;

# [cfg (native_lane)]
# [cfg (all (test , not (target_os = "macos") , feature = "backend-candle"))]
#[path="tests/candle_image_load_shape_tests.rs"]
mod candle_image_load_shape_tests;

# [cfg (native_lane)]
# [cfg (all (test , target_os = "macos"))]
#[path="tests/measured_mlx_load_shape_tests.rs"]
mod measured_mlx_load_shape_tests;

# [cfg (native_lane)]
# [cfg (all (test , target_os = "macos"))]
#[path="tests/declared_sequential_resident_tests.rs"]
mod declared_sequential_resident_tests;
