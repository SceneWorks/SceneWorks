//! CANDLE dispatch capture owner.
use super::*;

pub(super) fn run(request: &Value) -> Result<Value, String> {
    if protocol::planned(request)?
        .get("backend")
        .and_then(Value::as_str)
        != Some("candle")
    {
        return Err(
            "Candle adapter received a non-Candle planned case; run the harness with --backend candle"
                .to_owned(),
        );
    }
    let provider = planned_provider(request)?;
    if matches!(provider, FLUX3_ID | FLUX3_BASE_ID) {
        return candle_flux3::run(request);
    }
    if provider == LTX25_ID {
        return run_ltx25_capture(request);
    }
    // sc-25690: the three Iris-3B task routes dispatch ABOVE the shared gates. Depth and restoration
    // are not generators at all (a provider-specific estimator and a `Transform`), so the five-rung
    // reference path cannot drive them, and the arm validates the still single-image geometry and
    // the plain overlay itself (`protocol::validate_iris_plan`) before any env var or weight is read.
    if matches!(provider, IRIS_ID | IRIS_DEPTH_ID | IRIS_RESTORE_ID) {
        return candle_iris::run(request);
    }
    // sc-24217: Supra2-IMG dispatches above the shared gates too. Its arm validates the native
    // single still, the plain overlay and its resident override itself
    // (`protocol::validate_supra2_plan`) before any env var or weight is read, and converts the
    // bundle in a child process before the measured process loads anything.
    if provider == SUPRA2_ID {
        return candle_supra2::run(request);
    }
    // sc-22736: the Wan 2.2 family and SCAIL-2 dispatch ABOVE the shared still gate, like LTX-2.5 —
    // they are VIDEO arms, and `validate_still_geometry` would refuse the multi-frame geometry they
    // exist to measure. The four ids are spelled HERE, as consts, rather than asked of the module:
    // `scripts/stale-lane-report.mjs::adapterCapturableProviders` reads this bespoke pre-gate
    // (`if matches!(provider, …) { return …(request); }`) off the source, and a call into the
    // module would hide them from it. `candle_wan_scail2::implements` is held to the same four by
    // `the_pre_gate_names_exactly_the_providers_the_module_implements`.
    if matches!(
        provider,
        WAN_TI2V_5B_ID | WAN_T2V_A14B_ID | WAN_I2V_A14B_ID | SCAIL2_ID
    ) {
        return candle_wan_scail2::run(request);
    }
    // sc-22737: Bernini's VIDEO entry, LTX-2.3 and both MiniMax-H3 entries dispatch above the
    // shared still gate for the same reason the two blocks above do — `validate_still_geometry`
    // would refuse the multi-frame geometry these arms exist to measure. `sc22737_video_arm`
    // answers `None` for a provider this block does not serve AND for Bernini's STILL entry
    // (`bernini_image`), which is an image route and rides the shared five-rung reference path
    // below; it errors by name only for an unknown `(provider, modelId)` pair on a provider it
    // does serve.
    if let Some(arm) = sc22737_video_arm(request)? {
        return run_sc22737_video_capture(request, arm);
    }
    // sc-22726: PuLID-FLUX dispatches ABOVE the shared plain-overlay gate, like LTX-2.5. Its
    // declared overlay is `identity`, so routing it through `validate_plain_overlay_target` would
    // refuse the one target it exists to measure.
    if provider == PULID_FLUX_ID {
        return run_pulid_flux_capture(request);
    }
    // sc-22728: likewise before the plain-overlay gate below, because the Lightning member declares
    // a material `lora` overlay — its built-in distill — and validates it exactly instead.
    if provider == QWEN_EDIT_ID {
        return run_qwen_edit(request);
    }
    // sc-22729: the bespoke identity route carries a MATERIAL overlay, so it must be dispatched
    // before the plain-overlay guard below, which refuses anything but `none`.
    if provider == INSTANTID_ID {
        return run_instantid_candle(request);
    }
    // sc-22734: the SenseNova family has its own arm rather than the shared five-rung path — the
    // engine classifies StagedResidency structurally not applicable, so the five-rung
    // Sequential+deferred composition is not a shape it can be measured in. The arm resolves the
    // member from `(provider, modelId)` and refuses any other pair by name.
    if provider == SENSENOVA_ID || provider == SENSENOVA_FAST_ID {
        return run_sensenova_capture(request);
    }
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Both dispatch targets below are image arms; refuse a non-still geometry here, before either of
    // them resolves an environment variable or touches a weight snapshot (sc-18808).
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    if routes_to_five_rung_reference(request)? {
        return run_five_rung_reference(request);
    }
    if provider != KREA_ID {
        return Err(format!(
            "unsupported Candle calibration provider {provider:?}"
        ));
    }
    let parameters = protocol::strategy_parameters(request)?;
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let root = std::env::var("SCENEWORKS_KREA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_default();
    let root = if root.is_dir() {
        let canonical = std::fs::canonicalize(root)
            .map_err(|error| format!("canonicalize SCENEWORKS_KREA_ROOT: {error}"))?;
        protocol::validate_huggingface_snapshot_root(
            &canonical,
            &repository,
            &revision,
            tier,
            protocol::KREA_REPOSITORY,
        )?;
        canonical
    } else {
        root
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential);
    let spec = match numeric_tier(tier)?.quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let contract = catalog
        .media()
        .memory_strategy_contract(KREA_ID, &spec)
        .map_err(|error| format!("read {KREA_ID} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            format!(
                "{KREA_ID} has no memory-strategy contract at {}",
                protocol::INFERENCE_PIN
            )
        })?;
    let edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    let attention = protocol::parameter(request, "attentionChunkSize")?;
    let window = protocol::parameter(request, "transformerWindowSize")?;
    let selected = MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        attention_chunk_size: Some(attention),
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::Dit),
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: selected,
        tier: numeric_tier(tier)?,
    };
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let actual_calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "{KREA_ID} has no calibration identity at {}",
            protocol::INFERENCE_PIN
        )
    })?;
    let actual_fingerprint = actual_calibration.fingerprint.as_str();
    if planned_fingerprint != actual_fingerprint {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={actual_fingerprint} at {}",
                protocol::INFERENCE_PIN
            ),
            "contractFingerprintMismatch",
            &repository,
            &revision,
        );
    }

    if let Err(reason) = contract.validate_selection(&selection) {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!("pinned provider rejected planned parameters before load: {reason}"),
            "contractParameterRejection",
            &repository,
            &revision,
        );
    }
    if !root.is_dir() {
        return preflight_fragment(
            request,
            &strategy,
            actual_calibration.load_shape,
            format!(
                "supported provider tuple requires real weights; set SCENEWORKS_KREA_ROOT to                  the validated {tier} snapshot"
            ),
            "missingWeights",
            &repository,
            &revision,
        );
    }

    let hardware = request
        .get("hardware")
        .and_then(Value::as_object)
        .ok_or_else(|| "run request.hardware must be an object".to_owned())?;
    let total_bytes = hardware
        .get("memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let (width, height) = protocol::target_geometry(request)?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: actual_calibration.abi,
        calibration_fingerprint: actual_calibration.fingerprint.clone(),
        load_shape: actual_calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2 * GIB,
        },
        predicted_peak_bytes: total_bytes.saturating_sub(2 * GIB),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-21714-certifying@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(KREA_ID, &spec)
        .map_err(|error| format!("load real {KREA_ID} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin real Krea memory-strategy scope: {error}"))?
        .ok_or_else(|| "optimized Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure Krea decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure Krea attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure Krea transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply Krea request-scoped strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter Krea conditioning phase: {error}"))?;

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            conditioning_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            phase_sample = Some(vram.phase());
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            denoise_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
            phase_sample = Some(vram.phase());
        }
        _ => {}
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    let report = vram.report();
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("real Krea {tier} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave terminal Krea phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish real Krea memory-strategy scope: {error}"))?;
    let selected_image = match output {
        runtime_cuda::gen_core::GenerationOutput::Images(mut images) if images.len() == 1 => {
            images.remove(0)
        }
        runtime_cuda::gen_core::GenerationOutput::Images(images) => {
            return Err(format!(
                "real Krea run returned {} images, expected 1",
                images.len()
            ));
        }
        _ => return Err("real Krea run returned non-image output".to_owned()),
    };

    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        "Krea run did not expose a conditioning-to-denoise phase boundary".to_owned()
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            "Krea run did not expose a denoise-to-decode phase boundary".to_owned()
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| "Krea run did not complete decode sampling".to_owned())?,
    );
    let overall_bytes = decimal_gb_to_bytes(report.peak_gb)
        .max(conditioning_bytes)
        .max(denoise_bytes)
        .max(decode_bytes);
    let baseline = decimal_gb_to_bytes(report.baseline_gb);

    let mut exact = context.clone();
    exact.predicted_peak_bytes = overall_bytes;
    exact.budget = MemoryBudget {
        total_bytes: overall_bytes,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Candle Krea provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = exact.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle Krea provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = exact.clone();
    stale.calibration_fingerprint = "stale-krea-turbo-candle-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle Krea provider accepted stale calibration evidence".to_owned());
    }

    let lifecycle_phases = [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let smi = NvidiaSmi::resolve()?;
    let cleanup_tolerance_bytes = 64 * MIB;
    let mut maximum_cleanup_growth_bytes = 0_u64;
    let mut maximum_recovery_maximum_error = 0.0_f64;
    let mut maximum_recovery_mean_error = 0.0_f64;
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        let canceled_output = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            Some(lifecycle_phase),
        )?;
        if canceled_output.is_some() {
            return Err(format!(
                "{lifecycle_phase:?} cancellation unexpectedly produced an image"
            ));
        }
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} cancellation retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        let recovered = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?
        .ok_or_else(|| {
            format!("{lifecycle_phase:?} cancellation warm follow-up produced no image")
        })?;
        let (maximum, mean) = pixel_error(&selected_image, &recovered)?;
        ensure_krea_quality(
            maximum,
            mean,
            &format!("{lifecycle_phase:?} cancellation recovery"),
        )?;
        maximum_recovery_maximum_error = maximum_recovery_maximum_error.max(maximum);
        maximum_recovery_mean_error = maximum_recovery_mean_error.max(mean);
    }
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        let fault_output = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            Some(lifecycle_phase),
            None,
        )?;
        if fault_output.is_some() {
            return Err(format!(
                "{lifecycle_phase:?} injected error unexpectedly produced an image"
            ));
        }
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} injected error retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        let recovered = execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?
        .ok_or_else(|| format!("{lifecycle_phase:?} error warm follow-up produced no image"))?;
        let (maximum, mean) = pixel_error(&selected_image, &recovered)?;
        ensure_krea_quality(
            maximum,
            mean,
            &format!("{lifecycle_phase:?} error recovery"),
        )?;
        maximum_recovery_maximum_error = maximum_recovery_maximum_error.max(maximum);
        maximum_recovery_mean_error = maximum_recovery_mean_error.max(mean);
    }
    let resident_parameters = MemoryStrategyParameters::default();
    let resident_a = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let bounded_b = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::BoundedTransformerResidency,
        selected,
    )?;
    let resident_a_repeat = execute_parity_request(
        generator.as_ref(),
        &context,
        MemoryStrategy::Resident,
        resident_parameters,
    )?;
    let (resident_repeat_max_error, resident_repeat_mean_error) =
        pixel_error(&resident_a, &resident_a_repeat)?;
    if resident_repeat_max_error != 0.0 || resident_repeat_mean_error != 0.0 {
        return Err(format!(
            "resident A-B-A repeat was not deterministic: max={resident_repeat_max_error:.6}, mean={resident_repeat_mean_error:.6}"
        ));
    }
    let (bounded_max_error, bounded_mean_error) = pixel_error(&resident_a, &bounded_b)?;
    ensure_krea_quality(
        bounded_max_error,
        bounded_mean_error,
        "bounded-versus-resident A-B-A parity",
    )?;
    let mutated = negative_mutation(&bounded_b);
    let (mutated_max_error, mutated_mean_error) = pixel_error(&resident_a, &mutated)?;
    if mutated_max_error <= KREA_CANDLE_MAX_THRESHOLD
        && mutated_mean_error <= KREA_CANDLE_MEAN_THRESHOLD
    {
        return Err("Candle Krea output mutation did not breach the parity envelope".to_owned());
    }

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(actual_calibration.load_shape),
        "artifact": artifact(&repository, &revision, tier),
        "sweep": complete_sweep(request, parameters)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": overall_bytes, "effectiveBudgetBytes": overall_bytes },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation; retained memory stayed within 64 MiB and every warm recovery passed the approved parity envelope", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries; retained memory stayed within 64 MiB and every warm recovery passed the approved parity envelope", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": {
            "conditioning": conditioning_bytes,
            "denoise": denoise_bytes,
            "decode": decode_bytes,
            "overall": overall_bytes,
        },
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "quality": {
            "contract": "identical artifact, prompt, seed, geometry, steps and tier; production bounded-transformer tuple versus resident A-B-A control",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": bounded_max_error,
            "meanError": bounded_mean_error,
            "maximumErrorThreshold": KREA_CANDLE_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_CANDLE_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": parameters,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_max_error,
            "meanError": mutated_mean_error,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, tier),
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:krea-turbo-certifying",
            "executed",
            [],
            [
                    ("preLoadDeviceUsed", "bytes", baseline),
                    (
                        "loadDevicePeakDelta",
                        "bytes",
                        decimal_gb_to_bytes(report.load_peak_gb),
                    ),
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("allocatorCounterAliasesDeviceDelta", "boolean", 1),
                    ("wiredCounterAliasesDiscreteDeviceDelta", "boolean", 1),
                    ("cudaCachingAllocatorPresent", "boolean", 0),
                    ("phaseCancelInjections", "count", 3),
                    ("phaseErrorInjections", "count", 3),
                    ("postFaultWarmFollowUps", "count", 6),
                    (
                        "maximumPostFaultDeviceGrowth",
                        "bytes",
                        maximum_cleanup_growth_bytes,
                    ),
                    (
                        "postFaultDeviceGrowthTolerance",
                        "bytes",
                        cleanup_tolerance_bytes,
                    ),
                    (
                        "abaResidentRepeatMaximumErrorPer255",
                        "count",
                        (resident_repeat_max_error * 255.0).round() as u64,
                    ),
                    (
                        "abaResidentRepeatMeanErrorMicroUnits",
                        "count",
                        (resident_repeat_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "abaBoundedMaximumErrorPer255",
                        "count",
                        (bounded_max_error * 255.0).round() as u64,
                    ),
                    (
                        "abaBoundedMeanErrorMicroUnits",
                        "count",
                        (bounded_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "maximumRecoveryMaximumErrorPer255",
                        "count",
                        (maximum_recovery_maximum_error * 255.0).round() as u64,
                    ),
                    (
                        "maximumRecoveryMeanErrorMicroUnits",
                        "count",
                        (maximum_recovery_mean_error * 1_000_000.0).round() as u64,
                    ),
                    (
                        "negativeMutationMaximumErrorPer255",
                        "count",
                        (mutated_max_error * 255.0).round() as u64,
                    ),
                    (
                        "negativeMutationMeanErrorMicroUnits",
                        "count",
                        (mutated_mean_error * 1_000_000.0).round() as u64,
                    ),
                ],
            ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, KREA_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}
