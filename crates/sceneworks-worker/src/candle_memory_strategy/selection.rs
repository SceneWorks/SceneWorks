//! candle_memory_strategy selection owner.
use super::*;

impl CandleMemoryEvaluation {
    /// The `image_memory_strategy_selected` telemetry payload for this selection (sc-22664, epic
    /// 22657 E7): the selected rung, its parameters, the basis that priced it, the three derived
    /// phase peaks where the law priced it, and the selector's own admission figures. Pure, so the
    /// AC test can assert the event carries the rung and phases the selector chose.
    pub(crate) fn selection_telemetry(&self, engine_id: &str, tier_key: &str) -> Value {
        let selection = self.context.selection;
        let parameters = selection.parameters;
        let geometry = self.context.geometry;
        serde_json::json!({
            "backend": "candle",
            "route": engine_id,
            "actualTier": tier_key,
            "mode": self.context.mode.as_key(),
            "geometry": {
                "width": geometry.width,
                "height": geometry.height,
                "batch": geometry.batch,
                "frames": geometry.frames,
            },
            "referenceCount": geometry.reference_count,
            "overlay": self.context.overlay.clone(),
            "strategy": strategy_label(selection.strategy),
            "parameters": {
                "stageResidency": self.memory.as_ref().is_some_and(|memory| memory.stage_residency),
                "decodeTileEdge": parameters.decode_tile_edge,
                "decodeOverlap": parameters.decode_overlap,
                "attentionChunkSize": parameters.attention_chunk_size,
                "transformerWindowSize": parameters.transformer_window_size,
            },
            "basis": self.basis.as_key(),
            "authority": match self.context.optimization_authority {
                gen_core::MemoryOptimizationAuthority::Resident => "resident",
                gen_core::MemoryOptimizationAuthority::Estimated => "estimated",
                gen_core::MemoryOptimizationAuthority::Calibrated => "calibrated",
            },
            "cacheState": match self.context.cache_state {
                MemoryCacheState::Cold => "cold",
                MemoryCacheState::Warm => "warm",
            },
            "predictedPeakBytes": self.context.predicted_peak_bytes,
            "measuredEndpointPeakBytes": self.measured_endpoint_peak_bytes,
            "phasePeakBytes": self.phase_peaks.map(|phases| serde_json::json!({
                "conditioning": phases.conditioning,
                "denoise": phases.denoise,
                "decode": phases.decode,
            })),
            "admittedPeakGb": self.admitted.needed_gb,
            "availableGb": self.admitted.available_gb,
            "reserveGb": self.admitted.reserve_gb,
            "evidenceRevision": self.context.evidence_revision.clone(),
            "prefixKvCache": self.prefix_kv_cache.map(|plan| serde_json::json!({
                "selected": plan.mode.label(),
                "pricedBytes": plan.priced_bytes,
            })),
        })
    }
}

/// Bind an already admitted Ideogram request to the cache entry that will actually execute it.
/// A warm entry loaded under Sequential is a lower-peak shape than a newly selected Resident
/// request, so execution must retain the exact staged sibling selected during pre-load admission.
pub(crate) fn bind_ideogram_cache_execution(
    context: &mut MemoryRunContext,
    memory: &mut Option<GenerationMemory>,
    warm_staged: &mut Option<CandleWarmStagedEvaluation>,
    cache_state: MemoryCacheState,
    loaded_offload_policy: gen_core::OffloadPolicy,
) -> WorkerResult<()> {
    if loaded_offload_policy == gen_core::OffloadPolicy::Sequential
        && context.selection.strategy == MemoryStrategy::Resident
    {
        let staged = warm_staged.take().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Ideogram warm Sequential cache entry has no exact pre-admitted staged strategy"
                    .to_owned(),
            )
        })?;
        *memory = Some(staged.memory);
        *context = staged.context;
    }
    context.cache_state = cache_state;
    Ok(())
}

/// [`evaluate_shared_image`] with the provider's prefix-KV-cache surcharge priced in (epic 25899):
/// `prefix_kv_cache_surcharge` is `execution_planner::prefix_kv_cache_surcharge_bytes` for the
/// request, and it is added rung-for-rung (`execution_planner::plan_prefix_kv_cache`): the shed
/// ladder picks the rung exactly as before, and the cache is taken only when that rung also fits it.
/// `None` is exactly [`evaluate_shared_image`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image_priced(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        prefix_kv_cache_surcharge,
        None,
        sdxl_endpoints,
    )
}

/// Reference-aware admission; facts originate only in the owned-asset resolver.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image_priced_with_resources(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    // Resolve the backend-owned bound before contract derivation or any fit decision. A provider
    // with references but no pricer is a typed refusal, never a zero-priced legacy fallthrough.
    let reference_workspace_bytes =
        crate::memory_strategy::reference_workspace_bytes(engine_id, spec, reference_resources)?;
    evaluate_shared_image_dispatch(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        prefix_kv_cache_surcharge,
        reference_resources,
        reference_workspace_bytes,
        sdxl_endpoints,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_dispatch(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
        None,
        0,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_shared_image_dispatch(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    reference_workspace_bytes: u64,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let (contract_override, provider_overlay_override, provider_mode_override) =
        if spec.load_shape_declaration_result == gen_core::LoadShapeDeclarationResult::Eligible {
            let Some(mode) =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
            else {
                return Ok(None);
            };
            let Some(contract) = crate::memory_route_registry::declared_candle_selector_contract(
                engine_id,
                Some(tier_key),
                Some(mode),
                manifest,
                spec,
                crate::memory_route_registry::MemoryRouteRequestContext {
                    mode,
                    reference_count: geometry.reference_count,
                    use_pid,
                    has_phases: request_has_phases,
                },
            ) else {
                return Ok(None);
            };
            (Some(contract), None, None)
        } else {
            let Some(mode) =
                crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
            else {
                return Ok(None);
            };
            match crate::memory_route_registry::declared_candle_request_strategy_contract(
            engine_id,
            Some(tier_key),
            manifest,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode,
                reference_count: geometry.reference_count,
                use_pid,
                has_phases: request_has_phases,
            },
        ) {
            crate::memory_route_registry::DeclaredCandleStrategyContract::NoRelevantDeclaration => {
                (None, None, None)
            }
            crate::memory_route_registry::DeclaredCandleStrategyContract::Applied {
                contract,
                provider_overlay,
                provider_mode,
            } => (Some(*contract), Some(provider_overlay), Some(provider_mode)),
            crate::memory_route_registry::DeclaredCandleStrategyContract::Refused => {
                return Ok(None)
            }
        }
        };
    let provider_overlay = provider_overlay_override
        .as_ref()
        .map(|overlay| overlay.as_deref())
        .unwrap_or(overlay);
    let exact_sdxl_overlay = (engine_id == "sdxl")
        .then(|| sdxl_provider_overlay(spec))
        .flatten();
    let provider_overlay = exact_sdxl_overlay.as_deref().or(provider_overlay);
    evaluate_shared_image_inner_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        has_phases,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override.as_deref(),
        contract_override,
        None,
        None,
        prefix_kv_cache_surcharge,
        reference_resources,
        reference_workspace_bytes,
        sdxl_endpoints,
    )
}

/// Exact ordered physical overlay identity shared by registered and bespoke SDXL admissions.
///
/// Every sibling receipt-priced family binds its overlay to the exact materialized artifact —
/// Kolors through `KOLORS_CONTROL_RECEIPT_PREFIX`/`KOLORS_IP_RECEIPT_PREFIX`, SD3.5 and Chroma
/// through their ordered additive-adapter receipts. SDXL cannot read those from its contract:
/// `candle_gen_sdxl::memory_strategy::build_contract` emits no `resident_components`, and the
/// physical digest `SdxlArtifactSeal::capture` computes is discarded before the contract is
/// returned. So the worker seals its own ordered twin here, over the ROLE and the resolved SOURCE
/// PATH of every overlay slot, at the same grade `gen_core::adapter_stack_identity` already
/// applies to the adapter slot (path + kind + scale, sha256, domain-separated).
///
/// The identity this replaces was `control:{N}` / `ip-adapter` / `pid` — cardinality and kind
/// only. Two different tile ControlNets, or two different IP-Adapter checkpoints, produced one
/// identity and could therefore borrow each other's admitted peak. Ordering is the LoadSpec's own
/// slot order (control, then `extra_controls` in order, then IP-Adapter, then the PiD checkpoint
/// and its Gemma text encoder), never a sorted set, because the load order is what the provider
/// materializes.
pub(crate) fn sdxl_provider_overlay(spec: &LoadSpec) -> Option<String> {
    let mut slots: Vec<(&str, &gen_core::WeightsSource)> = Vec::new();
    if let Some(control) = spec.control.as_ref() {
        slots.push(("control", control));
    }
    for extra in &spec.extra_controls {
        slots.push(("extra-control", extra));
    }
    if let Some(ip_adapter) = spec.ip_adapter.as_ref() {
        slots.push(("ip-adapter", ip_adapter));
    }
    if let Some(pid) = spec.pid.as_ref() {
        slots.push(("pid-checkpoint", &pid.checkpoint));
        slots.push(("pid-gemma", &pid.gemma));
    }
    let adapters = gen_core::adapter_stack_identity(&spec.adapters);
    if slots.is_empty() && adapters.is_none() {
        return None;
    }
    let mut digest = sha2::Sha256::new();
    digest.update(SDXL_OVERLAY_RECEIPT_DOMAIN.as_bytes());
    for (ordinal, (role, source)) in slots.iter().enumerate() {
        // The source KIND is part of the seal: a directory snapshot and a single-file checkpoint
        // are different physical loads even when one path is a prefix of the other.
        let (kind, path) = match source {
            gen_core::WeightsSource::Dir(path) => ("dir", path),
            gen_core::WeightsSource::File(path) => ("file", path),
        };
        let path = format!("{:?}", path.as_os_str());
        digest.update((ordinal as u64).to_le_bytes());
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        digest.update((kind.len() as u64).to_le_bytes());
        digest.update(kind.as_bytes());
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
    }
    if let Some(adapters) = adapters.as_deref() {
        digest.update((adapters.len() as u64).to_le_bytes());
        digest.update(adapters.as_bytes());
    }
    Some(format!(
        "{SDXL_OVERLAY_RECEIPT_PREFIX}{:x}",
        digest.finalize()
    ))
}

/// Shared-selector entry point for a bespoke Candle provider whose exact contract is built from
/// caller-provisioned paths rather than the registered provider catalog.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_shared_bespoke_image(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    contract: gen_core::MemoryProviderContract,
    request_evidence_revision: &'static str,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let expected_revision = match engine_id {
        "pulid_flux" => crate::memory_route_registry::PULID_FLUX_REQUEST_EVIDENCE_REVISION,
        "candle_kolors_ipadapter" | "candle_kolors_control" => {
            crate::memory_route_registry::KOLORS_REQUEST_EVIDENCE_REVISION
        }
        "sdxl" => crate::memory_route_registry::SDXL_REQUEST_EVIDENCE_REVISION,
        _ => {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} has no registered bespoke memory evidence authority"
            )))
        }
    };
    if request_evidence_revision != expected_revision {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} crossed bespoke memory evidence revision {request_evidence_revision}"
        )));
    }
    if is_sealed_kolors_bespoke(engine_id) {
        let Some(mode) =
            crate::memory_route_registry::MemoryRouteMode::from_request(request_mode_value)
        else {
            return Ok(None);
        };
        if !crate::memory_route_registry::declared_candle_bespoke_request(
            engine_id,
            Some(tier_key),
            request_mode_value,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode,
                reference_count: geometry.reference_count,
                use_pid,
                has_phases,
            },
        ) {
            return Ok(None);
        }
    }
    let exact_provider_overlay = if engine_id == "sdxl" {
        sdxl_provider_overlay(spec)
    } else {
        overlay.map(str::to_owned)
    };
    evaluate_shared_image_inner(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        exact_provider_overlay.as_deref(),
        geometry,
        has_reference,
        use_pid,
        has_phases,
        has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        None,
        Some(contract),
        Some(request_evidence_revision),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_shared_image_inner(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_inner_priced(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        worker_multipass,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override,
        contract_override,
        request_evidence_revision_override,
        ladder_anchors_override,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_shared_image_inner_priced(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
    prefix_kv_cache_surcharge: Option<u64>,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    evaluate_shared_image_inner_priced_with_resources(
        engine_id,
        model_id,
        spec,
        artifact_is_certified,
        manifest,
        tier_key,
        request_mode_value,
        overlay,
        provider_overlay,
        geometry,
        has_reference,
        use_pid,
        worker_multipass,
        request_has_phases,
        budget,
        reserve_gb,
        predicted_peak_gb,
        runtime_overlay_bytes,
        cache_state,
        provider_mode_override,
        contract_override,
        request_evidence_revision_override,
        ladder_anchors_override,
        prefix_kv_cache_surcharge,
        None,
        0,
        sdxl_endpoints,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_shared_image_inner_priced_with_resources(
    engine_id: &'static str,
    model_id: &str,
    spec: &LoadSpec,
    artifact_is_certified: bool,
    manifest: &JsonObject<String, Value>,
    tier_key: &str,
    request_mode_value: &str,
    overlay: Option<&str>,
    provider_overlay: Option<&str>,
    geometry: MemoryGeometry,
    has_reference: bool,
    use_pid: bool,
    worker_multipass: bool,
    request_has_phases: bool,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    predicted_peak_gb: Option<f64>,
    runtime_overlay_bytes: u64,
    cache_state: MemoryCacheState,
    provider_mode_override: Option<&str>,
    contract_override: Option<gen_core::MemoryProviderContract>,
    request_evidence_revision_override: Option<&'static str>,
    ladder_anchors_override: Option<CandleLadderAnchors<'_>>,
    prefix_kv_cache_surcharge: Option<u64>,
    reference_resources: Option<&gen_core::MemoryReferenceResourceFacts>,
    reference_workspace_bytes: u64,
    sdxl_endpoints: Option<&SdxlCudaMemoryEndpoints>,
) -> WorkerResult<Option<CandleMemoryEvaluation>> {
    let Some(registered_revision) =
        crate::memory_route_registry::candle_gate_evidence_revision(engine_id)
    else {
        // Unregistered families retain resident/legacy admission and cannot acquire an optimized
        // request contract through the former declaration-revision catch-all.
        return Ok(None);
    };
    let request_evidence_revision =
        request_evidence_revision_override.unwrap_or(registered_revision);
    // Hires-fix is two independently shaped denoise passes. Only families whose generation harness
    // mints and carries one exact context per pass may use request-scoped optimized admission.
    if worker_multipass && !is_ideogram(engine_id) && !is_sana(engine_id) && engine_id != "sdxl" {
        return Ok(None);
    }
    let mode =
        request_mode_with_provider_override(engine_id, request_mode_value, provider_mode_override);
    let Some(tier) = numeric_tier(engine_id, tier_key) else {
        return Ok(None);
    };
    let Some(budget) = budget else {
        return Ok(None);
    };
    let receipt_priced = is_receipt_priced(engine_id);
    let resident_peak_gb = match predicted_peak_gb {
        Some(peak) => peak,
        // SANA's certified provider receipt prices the selected dense snapshot from its real
        // safetensor headers. It deliberately has no per-story measured manifest curve.
        None if (is_sana(engine_id) || is_sd35(engine_id)) && artifact_is_certified => 0.0,
        None => {
            if receipt_priced && artifact_is_certified {
                return Err(WorkerError::InvalidPayload(format!(
                    "{engine_id}/{tier_key} is missing its structural resident peak row"
                )));
            }
            return Ok(None);
        }
    };
    if receipt_priced && !artifact_is_certified {
        // Receipt-priced optimized contracts are tied to the immutable turnkey revision and physical tier.
        // An imported/custom root retains the historical resident path and receives no estimate.
        return Ok(None);
    }
    let contract = match contract_override {
        Some(contract) => contract,
        None => {
            let Some(contract) = crate::inference_runtime::media()
                .memory_strategy_contract(engine_id, spec)
                .map_err(|error| {
                    WorkerError::Engine(format!("{engine_id} memory contract failed: {error}"))
                })?
            else {
                return Ok(None);
            };
            contract
        }
    };
    let provider_overlay = if is_chroma(engine_id) {
        validate_chroma_asset_facts(engine_id, &contract)?;
        chroma_provider_overlay_identity(&contract, provider_overlay)?
    } else if is_ideogram(engine_id) {
        validate_ideogram_asset_facts(engine_id, &contract, use_pid, &mode.mode, worker_multipass)?;
        ideogram_provider_overlay_identity(engine_id, &contract, provider_overlay)?
    } else if is_sana(engine_id) {
        validate_sana_asset_facts(engine_id, &contract)?;
        if provider_overlay.is_some() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} does not accept an adapter overlay"
            )));
        }
        None
    } else if is_sd35(engine_id) {
        validate_sd35_asset_facts(engine_id, &contract)?;
        sd35_provider_overlay_identity(engine_id, &contract, provider_overlay)?
    } else if engine_id == "kolors" {
        let exact = kolors_overlay_receipt_identity(engine_id, &contract, use_pid)?;
        let has_declared_adapters = provider_overlay.is_some();
        let has_receipted_adapters = contract
            .resident_components()
            .iter()
            .any(|component| component.id.starts_with(KOLORS_ADAPTER_RECEIPT_PREFIX));
        if has_declared_adapters != has_receipted_adapters {
            return Err(WorkerError::InvalidPayload(
                "kolors public ordered-adapter identity crossed its exact provider receipt"
                    .to_owned(),
            ));
        }
        exact
    } else if is_sealed_kolors_bespoke(engine_id) {
        let exact = kolors_overlay_receipt_identity(engine_id, &contract, use_pid)?;
        match (provider_overlay, exact) {
            (None, None) => None,
            (Some(_), Some(exact)) => Some(exact),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "{engine_id} public overlay crossed its exact provider receipt"
                )))
            }
        }
    } else {
        provider_overlay.map(str::to_owned)
    };
    let provider_overlay = provider_overlay.as_deref();
    let calibration = contract.calibration.as_ref();
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let mut resident = sceneworks_core::memory_contracts::unverified_estimate(
        MemoryEvidenceKey {
            model_family: engine_id.to_owned(),
            resolved_route: engine_id.to_owned(),
            backend: MemoryBackend::Candle,
            tier,
            mode: mode.mode.clone(),
            reference_shape: if geometry.reference_count == 0 {
                gen_core::MemoryReferenceShape::None
            } else {
                gen_core::MemoryReferenceShape::Image
            },
            load_shape: contract.load_shape,
            overlay: provider_overlay.map(str::to_owned),
            geometry,
            frames_per_second: None,
            strategy: MemoryStrategy::Resident,
            engaged_composition: contract.engaged_composition(MemoryStrategy::Resident),
            parameters: resident_selection.parameters,
        },
        calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |item| item.abi),
        calibration.map_or_else(String::new, |item| item.fingerprint.clone()),
        request_evidence_revision.to_owned(),
        {
            let declared = scale_ideogram_hires_envelope(
                engine_id,
                &mode.mode,
                manifest,
                geometry,
                (resident_peak_gb * BYTES_PER_GIB).ceil() as u64,
            );
            if receipt_priced {
                let structural = contract
                    .asset_facts
                    .base_bytes
                    .saturating_add((crate::vram_gate::HEADROOM_GB * BYTES_PER_GIB).ceil() as u64);
                contract
                    .predicted_peak_from_base(declared.max(structural))
                    .predicted_peak_bytes()
            } else {
                // The caller's resident peak is `vram_gate::predicted_peak_gb`: the measured
                // `vramGbByTier` row plus `HEADROOM_GB` (or `minMemoryGb`, padded by the manifest
                // itself). The pad stays INSIDE this candidate and it is compared against the
                // unreserved pool (`crate::memory_strategy::ReserveCharge`, sc-22664).
                declared
            }
        },
    );
    let exact_overlay = overlay.unwrap_or("none");
    // Matrix/evidence overlays remain cells of the advertised base provider (`z_image` or
    // `z_image_turbo`), while the runtime registers the strict-control implementation under a
    // dedicated `_control` id. Query the packaged evidence by its catalog identity, then bind the
    // returned candidate to the exact runtime route expected by the provider contract.
    let mut verified = if artifact_is_certified {
        verified_candidates(
            manifest,
            model_id,
            engine_id,
            tier_key,
            &mode,
            exact_overlay,
            geometry,
        )?
    } else {
        Vec::new()
    };
    // The provider-owned calibration identity answers whether the measured memory semantics
    // changed, and it has to match: FLUX.2-dev's caption-upsample lifecycle intentionally rotated
    // v2 -> v3 while retaining the old historical records, and a v2 record would feed an obsolete
    // prompt-conditioning peak into the live v3 contract. This is a provider DECLARATION, not
    // measurement currency — the closure digest a record was captured under is deliberately not
    // read here (sc-22738).
    let mut current_verified = Vec::with_capacity(verified.len());
    if let Some(current_calibration) = calibration {
        for candidate in verified {
            if candidate.calibration_fingerprint == current_calibration.fingerprint {
                current_verified.push(candidate);
            }
        }
    }
    verified = current_verified;
    // Packaged bindings are looked up by the public catalog/matrix overlay above. Once admitted,
    // however, every candidate submitted to the provider selector must carry the exact provider
    // evidence identity declared by the route. Krea adapters are load identity, not a provider
    // `MemoryRunContext` overlay, so their public `lora` cell is normalized to `None` here.
    for candidate in &mut verified {
        candidate.key.overlay = provider_overlay.map(str::to_owned);
    }
    // Calibration records describe the certified overlay fixture. User-provided adapters can be
    // larger, so every optimized candidate must reserve the bytes for the actual request before
    // the common selector performs its fit check. Legacy resident estimates already include the
    // caller's source-byte charge and must not be adjusted twice. Chroma instead ignores that raw
    // value and uses the provider receipt's materialized adapter + PiD aggregate here and above.
    let runtime_overlay_bytes = if receipt_priced {
        contract.asset_facts.overlay_bytes
    } else {
        runtime_overlay_bytes
    };
    account_for_runtime_overlay_bytes(&mut verified, runtime_overlay_bytes);
    let sdxl_endpoints = sdxl_endpoints.filter(|pair| {
        engine_id == "sdxl"
            && model_id == "sdxl"
            && tier_key == "q4"
            && artifact_is_certified
            && mode.mode == MemoryMode::TextToImage
            && !has_reference
            && !use_pid
            && !worker_multipass
            && geometry.batch == 1
            && geometry.frames == 1
            && geometry.reference_count == 0
            && provider_overlay.is_none()
            && runtime_overlay_bytes == 0
            && spec.resolved_route.as_deref() == Some("sdxl")
            && calibration.is_some_and(|identity| {
                identity.fingerprint == pair.resident.source.calibration_fingerprint
            })
            && manifest
                .get("downloads")
                .and_then(Value::as_array)
                .is_some_and(|downloads| {
                    downloads.iter().any(|download| {
                        download.get("repo").and_then(Value::as_str)
                            == Some(pair.artifact.repository.as_str())
                            && download.get("revision").and_then(Value::as_str)
                                == Some(pair.artifact.resolved_revision.as_str())
                            && download.get("variant").and_then(Value::as_str)
                                == Some(pair.artifact.variant.as_str())
                    })
                })
    });
    let resident_endpoint = sdxl_endpoints.and_then(|pair| {
        sdxl_endpoint_candidate(
            &contract,
            &resident,
            &pair.resident,
            resident_selection,
            pair.resident_load_peak_bytes,
        )
    });
    let resident_basis = resident_endpoint.as_ref().map_or(
        crate::memory_strategy::CandidateBasis::Measured,
        |candidate| candidate.basis,
    );
    let resident_has_manifest_pad = resident_endpoint.is_none();
    if let Some(candidate) = &resident_endpoint {
        resident = candidate.evidence.clone();
    }
    if let Some(exact) = verified.first() {
        resident
            .inference_revision
            .clone_from(&exact.inference_revision);
    }
    // sc-18097: estimate-floor candidates for every implemented optimized rung, so an unmeasured
    // cell can still engage the ladder behind the candle estimate margin instead of freezing to
    // resident-or-legacy behavior. Where an exact measured record exists the selector's
    // measured-supersedes-estimate rule keeps admission byte-for-byte unchanged.
    //
    // Gated on `artifact_is_certified`, the same conjunct that gates the packaged records above
    // (sc-18097 review, major finding). The floors are read from the SHIPPED manifest's
    // `sequentialPeakGb`/`vramGbByTier` rows, which describe the certified artifact. An imported
    // or community checkpoint on a supported route is different bytes: those rows do not describe
    // it, so extending its ladder from them would be a static capability declaration authorizing
    // an optimized request — exactly what this module's header forbids — and CUDA OOM being
    // recoverable does not make a wrong prediction safe. An uncertified artifact therefore keeps
    // its resident-estimate-only behavior, byte-for-byte as before this story. The sibling control
    // lane gates its floors the same way (`krea_control_fit.rs`, `runtime_verified`).
    // Reuse the existing image derivation law for intermediate configurations. The endpoint pair
    // has its own source-validated store, leaving the general catalog's unique-anchor policy intact.
    let endpoint_anchor_store =
        sdxl_endpoints.map(|pair| sceneworks_core::memory_anchor::MemoryAnchorStore {
            schema_version: sceneworks_core::memory_anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
            anchors: vec![pair.resident.clone()],
            analytic_only: Vec::new(),
            component_deltas: Vec::new(),
            exceeded_bounds: Vec::new(),
            // The endpoint pair is a measured capacity capture, not an owned-process accounting
            // observation; the ladder store carries no operating envelopes.
            observed_operating_envelopes: Vec::new(),
        });
    let endpoint_anchors = endpoint_anchor_store
        .as_ref()
        .map(|store| CandleLadderAnchors {
            store: Some(store),
            facts: crate::video_admission::architecture_facts_from_contract(&contract),
        });
    let mut synthesized = if artifact_is_certified {
        synthesize_estimate_floors(
            engine_id,
            model_id,
            &contract,
            manifest,
            tier_key,
            tier,
            &mode,
            provider_overlay,
            geometry,
            if resident_endpoint.is_some() {
                (resident_peak_gb * BYTES_PER_GIB).ceil() as u64
            } else {
                resident.predicted_peak_bytes
            },
            runtime_overlay_bytes,
            request_evidence_revision,
            endpoint_anchors
                .or(ladder_anchors_override)
                .unwrap_or_else(|| CandleLadderAnchors::packaged(&contract)),
        )
    } else {
        Vec::new()
    };
    // This is a separate request demand, NOT an adapter overlay or a text-token hint. Charge
    // every complete candidate exactly once, after receipt overlay normalization and phase-law
    // synthesis but BEFORE selector grading. Keeping the source evidence objects charged also
    // keeps selected-context and warm-alternative peak reporting identical to the demand graded.
    account_for_runtime_overlay_bytes(
        std::slice::from_mut(&mut resident),
        reference_workspace_bytes,
    );
    account_for_runtime_overlay_bytes(&mut verified, reference_workspace_bytes);
    for candidate in &mut synthesized {
        account_for_runtime_overlay_bytes(
            std::slice::from_mut(&mut candidate.evidence),
            reference_workspace_bytes,
        );
    }
    if let Some(pair) = sdxl_endpoints {
        for candidate in &mut synthesized {
            if matches!(
                candidate.basis,
                crate::memory_strategy::CandidateBasis::EstimateAnchorDerived { .. }
            ) {
                let load_peak = if contract
                    .engages_selection(&candidate.selection, MemoryStrategy::StagedResidency)
                {
                    pair.sequential_load_peak_bytes
                } else {
                    pair.resident_load_peak_bytes
                };
                candidate.evidence.predicted_peak_bytes =
                    candidate.evidence.predicted_peak_bytes.max(load_peak);
            }
        }
        let engaged = [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedDecode,
            StrategyRung::BoundedAttention,
        ];
        if let Some(parameters) = parse_parameters(&engaged, &pair.sequential_parameters) {
            let selection = MemorySelection {
                strategy: MemoryStrategy::BoundedAttention,
                parameters,
                tier,
            };
            if let Some(endpoint) = sdxl_endpoint_candidate(
                &contract,
                &resident,
                &pair.sequential,
                selection,
                pair.sequential_load_peak_bytes,
            ) {
                // A prediction of this same exact composition must not supersede its observed
                // endpoint. Other compositions retain their independently derived estimates.
                synthesized.retain(|candidate| candidate.selection != endpoint.selection);
                synthesized.push(endpoint);
            }
        }
    }
    let capacity = verified.len() + synthesized.len() + 1;
    let mut selections = Vec::with_capacity(capacity);
    let mut evidence = Vec::with_capacity(capacity);
    // Index-aligned basis axis (sc-18097): the synthesized floors carry their estimate basis;
    // the resident live estimate and every packaged record stay `Measured`.
    let mut candidate_bases = Vec::with_capacity(capacity);
    selections.push(resident_selection);
    evidence.push(&resident);
    candidate_bases.push(resident_basis);
    for item in &verified {
        selections.push(MemorySelection {
            strategy: item.key.strategy,
            parameters: item.key.parameters,
            tier: item.key.tier,
        });
        evidence.push(item);
        candidate_bases.push(crate::memory_strategy::CandidateBasis::Measured);
    }
    for candidate in &synthesized {
        selections.push(candidate.selection);
        evidence.push(&candidate.evidence);
        candidate_bases.push(candidate.basis);
    }
    debug_assert_eq!(evidence.len(), candidate_bases.len());
    let candidates = selections
        .iter()
        .zip(evidence)
        .zip(&candidate_bases)
        .map(|((selection, evidence), basis)| Candidate {
            selection: *selection,
            evidence,
            basis: *basis,
            // sc-22508: the candle floors here are manifest `sequentialPeakGb`/`vramGbByTier`
            // rows, not a weights+headroom split this lane can decompose, so no activation term
            // is declared and the selector charges the backend's whole-peak accounting residual
            // (`CANDLE_RECAPTURE_SPREAD`). Declaring a split is sc-22509's anchor work.
            unmodeled_activation_bytes: None,
        })
        .collect::<Vec<_>>();
    let request_scope = RequestScope {
        resolved_route: engine_id,
        backend: "candle",
        tier,
        mode: &mode.scope_key,
        overlay: provider_overlay,
        geometry,
    };
    // sc-22664: the operational reserve — `vram_gate::ladder_reserve_gb` of the caller's RAW probe,
    // handed in explicitly so a reclaimable-credited `budget` can never derive it — on the selector
    // budget, charged per candidate by the one rule `crate::memory_strategy::ReserveCharge`
    // states: an anchor-derived candidate (a reserve-free device delta) compares against the pool
    // minus the reserve; the resident live estimate, a manifest-row floor and a receipt-priced
    // structural floor already carry `HEADROOM_GB` inside their peaks and compare against the
    // unreserved pool, so no candidate pays twice.
    let selector_budget = Some(Budget {
        available_gb: budget.free_gb,
        reclaimable_gb: 0.0,
        total_gb: budget.total_gb,
        reserved_headroom_gb: reserve_gb,
    });
    let pad_carrying = |candidate: &Candidate<'_>| {
        (resident_has_manifest_pad && std::ptr::eq(candidate.evidence, &resident))
            || candidate.basis == crate::memory_strategy::CandidateBasis::EstimateFloor
    };
    let reserve_charge = crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&pad_carrying);
    // Epic 25899: the provider's prefix KV cache, added rung-for-rung. The ladder runs first at the
    // shed demand — exactly the pre-cache admission, which picks the rung — and the cache is taken
    // only if that SAME rung (`only_rung`) also fits with the surcharge on its peak, under the same
    // budget and pad rule. The cache never moves a request to another rung. `None` runs the ladder
    // once, unchanged.
    let run_ladder = |extra_bytes: u64, only_rung: Option<MemorySelection>| -> Selection {
        let on_rung =
            |candidate: &&Candidate<'_>| only_rung.is_none_or(|rung| candidate.selection == rung);
        if extra_bytes == 0 && only_rung.is_none() {
            return crate::memory_strategy::select_strategy_charging(
                request_scope,
                &contract,
                selector_budget,
                &candidates,
                reserve_charge,
            );
        }
        let kept = candidates.iter().filter(on_rung).collect::<Vec<_>>();
        let surcharged = kept
            .iter()
            .map(|candidate| {
                let mut evidence = candidate.evidence.clone();
                evidence.predicted_peak_bytes =
                    evidence.predicted_peak_bytes.saturating_add(extra_bytes);
                evidence
            })
            .collect::<Vec<_>>();
        let surcharged_candidates = kept
            .iter()
            .zip(&surcharged)
            .map(|(candidate, evidence)| Candidate {
                evidence,
                ..**candidate
            })
            .collect::<Vec<_>>();
        // The pad rule follows each candidate from the original list: the resident live estimate
        // and every estimate floor carry their pad.
        let carries_pad = kept
            .iter()
            .map(|candidate| pad_carrying(candidate))
            .collect::<Vec<_>>();
        let surcharged_pad_carrying = |candidate: &Candidate<'_>| {
            surcharged
                .iter()
                .position(|evidence| std::ptr::eq(candidate.evidence, evidence))
                .is_some_and(|index| carries_pad[index])
        };
        crate::memory_strategy::select_strategy_charging(
            request_scope,
            &contract,
            selector_budget,
            &surcharged_candidates,
            crate::memory_strategy::ReserveCharge::ExceptPadCarrying(&surcharged_pad_carrying),
        )
    };
    let (selected, prefix_kv_cache) = crate::execution_planner::plan_prefix_kv_cache(
        prefix_kv_cache_surcharge,
        run_ladder(0, None),
        |selection| matches!(selection, Selection::Selected { .. }),
        |shed, priced_bytes| {
            let Selection::Selected {
                selection: rung, ..
            } = shed
            else {
                return None;
            };
            let cached = run_ladder(priced_bytes, Some(*rung));
            matches!(&cached, Selection::Selected { selection, .. } if selection == rung)
                .then_some(cached)
        },
    );
    let prefix_kv_cache_extra = prefix_kv_cache.map_or(0, |plan| plan.admitted_surcharge_bytes());
    let Selection::Selected {
        selection,
        needed_gb,
        available_gb,
    } = selected
    else {
        if receipt_priced {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id} has no exact resident or staged strategy that fits the sealed provider receipt"
            )));
        }
        return Ok(None);
    };
    let evidence_for_selection = |selection: MemorySelection| {
        let matches_selection = |item: &MemoryEvidence| {
            item.key.strategy == selection.strategy
                && item.key.parameters == selection.parameters
                && item.key.tier == selection.tier
        };
        if selection.strategy == MemoryStrategy::Resident {
            Some((
                &resident,
                resident_basis.is_estimate(),
                resident_endpoint.as_ref(),
            ))
        } else if let Some(item) = verified.iter().find(|item| matches_selection(item)) {
            Some((item, false, None))
        } else {
            synthesized
                .iter()
                .find(|candidate| matches_selection(&candidate.evidence))
                // sc-18097: synthesized floors are estimate-scoped, never calibrated evidence.
                .map(|candidate| (&candidate.evidence, true, Some(candidate)))
        }
    };
    let (selected_evidence, estimate_scoped, selected_estimate) = evidence_for_selection(selection)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id} selected a memory strategy without exact packaged evidence"
            ))
        })?;
    let to_bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    let context_for = |selection: MemorySelection,
                       selected_evidence: &MemoryEvidence,
                       estimate_scoped: bool|
     -> MemoryRunContext {
        MemoryRunContext {
            selection,
            optimization_authority: if selection.strategy == MemoryStrategy::Resident {
                gen_core::MemoryOptimizationAuthority::Resident
            } else if estimate_scoped {
                gen_core::MemoryOptimizationAuthority::Estimated
            } else {
                gen_core::MemoryOptimizationAuthority::Calibrated
            },
            calibration_abi: selected_evidence.calibration_abi,
            calibration_fingerprint: selected_evidence.calibration_fingerprint.clone(),
            mode: mode.mode.clone(),
            load_shape: contract.load_shape,
            has_reference,
            use_pid,
            has_phases: request_has_phases,
            geometry,
            reference_resources: reference_resources.cloned(),
            overlay: provider_overlay.map(str::to_owned),
            budget: gen_core::MemoryBudget {
                total_bytes: to_bytes(budget.total_gb),
                committed_bytes: to_bytes((budget.total_gb - budget.free_gb).max(0.0)),
                reclaimable_bytes: 0,
                reserved_headroom_bytes: to_bytes(reserve_gb),
            },
            predicted_peak_bytes: selected_evidence.predicted_peak_bytes,
            cache_state,
            evidence_revision: if (selection.strategy == MemoryStrategy::Resident
                && selected_evidence.conformance != MemoryConformanceState::Verified)
                || estimate_scoped
            {
                request_evidence_revision.to_owned()
            } else {
                selected_evidence.harness_version.clone()
            },
        }
    };
    let mut selected_context = context_for(selection, selected_evidence, estimate_scoped);
    // The demand the admission approved: a cached selection was graded at shed + surcharge.
    selected_context.predicted_peak_bytes = selected_context
        .predicted_peak_bytes
        .saturating_add(prefix_kv_cache_extra);

    // A loaded Sequential cache entry is a tighter peak shape than a new Resident request. Keep an
    // exact, independently selected staged sibling so the cache callback can honor that already
    // materialized shape. This is not an admission bypass: the same authoritative request scope,
    // evidence set, estimate margin, and budget select it here before the load/cache access.
    let warm_staged = if is_ideogram(engine_id) {
        let staged_candidates = candidates
            .iter()
            .filter(|candidate| {
                contract.engages(
                    candidate.selection.strategy,
                    MemoryStrategy::StagedResidency,
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        match crate::memory_strategy::select_strategy_charging(
            request_scope,
            &contract,
            selector_budget,
            &staged_candidates,
            reserve_charge,
        ) {
            Selection::Selected {
                selection: staged_selection,
                ..
            } => {
                let (evidence, estimate_scoped, _) = evidence_for_selection(staged_selection)
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "{engine_id} selected a warm staged strategy without exact evidence"
                        ))
                    })?;
                contract.generation_memory(&staged_selection).map(|memory| {
                    CandleWarmStagedEvaluation {
                        memory,
                        context: context_for(staged_selection, evidence, estimate_scoped),
                    }
                })
            }
            Selection::Reject { .. } | Selection::Unverified { .. } => None,
        }
    } else {
        None
    };
    let mut memory = memory_for_selection(&contract, selection);
    if let Some(plan) = prefix_kv_cache.filter(|plan| plan.mode.is_cached()) {
        // A Resident selection has no block; an admitted cached selection needs one to live in.
        plan.write_onto(memory.get_or_insert_with(GenerationMemory::default));
    }
    Ok(Some(CandleMemoryEvaluation {
        memory,
        prefix_kv_cache,
        measured_endpoint_peak_bytes: (selected_estimate.is_some_and(|candidate| {
            candidate.basis == crate::memory_strategy::CandidateBasis::EstimateMeasuredEndpoint
        }))
        .then_some(selected_evidence.observed_peak_bytes)
        .flatten(),
        predicted_peak_gb: selected_evidence
            .predicted_peak_bytes
            .saturating_add(prefix_kv_cache_extra) as f64
            / BYTES_PER_GIB,
        context: selected_context,
        warm_staged,
        basis: selected_estimate.map_or_else(
            || {
                if selection.strategy == MemoryStrategy::Resident {
                    resident_basis
                } else {
                    crate::memory_strategy::CandidateBasis::Measured
                }
            },
            |candidate| candidate.basis,
        ),
        phase_peaks: selected_estimate.and_then(|candidate| candidate.phase_peaks),
        admitted: AdmittedBudget {
            needed_gb,
            available_gb,
            reserve_gb,
        },
    }))
}
