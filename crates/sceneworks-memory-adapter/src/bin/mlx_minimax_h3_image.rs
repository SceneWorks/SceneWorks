//! Physical MLX capture arm for MiniMax-H3-Image (SW-7100, feature SW-F363).
//!
//! Measures the TOTAL pipeline the worker runs — MiniMax-H3's Qwen3-VL text encoder, the image
//! DiT and the image VAE, staged one after another by the engine — at the anchor request (a
//! 1024x1024 still, seed 7100, the fixed six steps, guidance 1.0), loaded through the seam
//! production loads it through (`runtime_macos::catalog().media().load("minimax_h3_image", spec)`,
//! the call `crates/sceneworks-worker/src/inference_runtime.rs` wraps) with the spec the worker's
//! `image_jobs::minimax_h3_image::attach_components` builds: the upstream `MiniMaxAI/MiniMax-H3-Image`
//! snapshot as `LoadSpec::weights`, the tier's quant, MiniMax-H3's same-tier text encoder as the
//! `text_encoder` component and, for a packed tier, its `transformer_diffusers` DiT. Repositories
//! and revisions are the catalog's own rows (`protocol::minimax_h3_image_inputs`), so a packed tier
//! still marked `pendingArtifact` is refused by name before any env var is read.
//!
//! The pinned engine's production contract publishes NO calibration identity (`h3_image::memory`
//! `contract_for`: "no calibration identity (unmeasured)"), so the plan names the adapter-owned
//! [`PLAN_FINGERPRINT`], the `stale_evidence` scenario is `not_applicable` and the record is
//! `gated`. If a later pin publishes an identity, it must equal the plan's — the Supra2 shape.
//!
//! SW-F363: the same arm captures the cell's regime anchors. An `:optimized` row loads deferred
//! and engages every rung the contract implements under the provider's own request scope (with
//! explicit estimate authority, which an uncalibrated optimized selection requires); a `:lora`
//! row installs the pinned community LoRA through `LoadSpec::adapters`. The base rows render
//! staged and eager (the engine stages every request, so the staged composition is the
//! least-optimized one it can render), under the same request scope.

use super::*;
use std::time::Instant;

const LANE: &str = "mlx";

/// The adapter-owned identity every MLX plan row names, base and regime alike (see the module
/// docs) — one identity per lane, as the Candle lane's engine constant is.
pub(super) const PLAN_FINGERPRINT: &str = "minimax-h3-image-mlx-v1";

/// The env family the catalog runner exports for this arm (`PROVIDER_FAMILIES.minimax_h3_image`).
pub(super) const ENVS: protocol::MinimaxH3ImageEnvs = protocol::MinimaxH3ImageEnvs {
    upstream_repository_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_REPOSITORY",
    upstream_revision_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_REVISION",
    upstream_root_env: "SCENEWORKS_MINIMAX_H3_IMAGE_UPSTREAM_ROOT",
    tier_repository_env: "SCENEWORKS_MINIMAX_H3_IMAGE_REPOSITORY",
    tier_revision_env: "SCENEWORKS_MINIMAX_H3_IMAGE_REVISION",
    tier_root_env: "SCENEWORKS_MINIMAX_H3_IMAGE_ROOT",
    text_encoder_env: "SCENEWORKS_MINIMAX_H3_IMAGE_COMPONENT_TEXT_ENCODER",
};

use protocol::elapsed_millis as millis;

#[allow(clippy::too_many_arguments)]
fn context(
    selection: &MemorySelection,
    calibration: Option<&MemoryCalibrationIdentity>,
    fingerprint: &str,
    load_shape: LoadShape,
    overlay: Option<&str>,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    let edge = protocol::MINIMAX_H3_IMAGE_EDGE;
    MemoryRunContext {
        selection: *selection,
        // An uncalibrated optimized selection is admitted only on explicit estimate authority
        // (gen-core `standard_memory_strategy_safety_check`), which is what the worker's
        // header-derived admission supplies for it.
        optimization_authority: if calibration.is_some() {
            MemoryOptimizationAuthority::Calibrated
        } else if selection.strategy.is_optimized() {
            MemoryOptimizationAuthority::Estimated
        } else {
            MemoryOptimizationAuthority::Resident
        },
        calibration_abi: calibration
            .map(|identity| identity.abi)
            .unwrap_or(mlx_gen::gen_core::MEMORY_CALIBRATION_ABI),
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: edge,
            height: edge,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: overlay.map(str::to_owned),
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sw-7100@{}", protocol::INFERENCE_PIN),
    }
}

/// One render of `request`. An optimized selection runs under the provider's own request scope —
/// the worker's `generate_with_scope` shape: the scope writes the admitted `GenerationMemory` into
/// the request and is finished with the outcome — and a resident one renders exactly as the base
/// rows always have, with no scope. `on_return` runs the moment `generate` returns, BEFORE the
/// scope's device cleanup, so the decode phase is read off the render rather than off the cleanup.
fn generate_scoped(
    generator: &dyn Generator,
    scoped: Option<&MemoryRunContext>,
    request: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
    on_return: &mut dyn FnMut(),
) -> Result<GenerationOutput, String> {
    let Some(scoped) = scoped else {
        let output = generator
            .generate(request, on_progress)
            .map_err(|error| format!("generate minimax_h3_image: {error}"))?;
        on_return();
        return Ok(output);
    };
    if let MemorySafetyDecision::Reject { reason } = generator.memory_strategy_safety_check(scoped)
    {
        return Err(format!(
            "minimax_h3_image: the loaded provider refused the planned memory context: {reason}"
        ));
    }
    let mut scope = generator
        .begin_memory_strategy_request(scoped)
        .map_err(|error| format!("begin the minimax_h3_image request scope: {error}"))?
        .ok_or_else(|| {
            "minimax_h3_image accepted an optimized memory-strategy selection without opening a \
             request scope"
                .to_owned()
        })?;
    let mut request = request.clone();
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure the minimax_h3_image request scope: {error}"))?;
    match generator.generate(&request, on_progress) {
        Ok(output) => {
            on_return();
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish the minimax_h3_image request scope: {error}"))?;
            Ok(output)
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            Err(format!("generate minimax_h3_image: {message}"))
        }
    }
}

/// The `mlx:minimax_h3_image` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Provider, mode, tier, the still, overlay, load shape, fixture and identity — all before an
    // env var or a weight.
    let fingerprint = PLAN_FINGERPRINT;
    let plan = protocol::validate_minimax_h3_image_plan(request, LANE, fingerprint)?;
    let (width, height, tier) = (plan.width, plan.height, plan.tier);
    let selection = planned_selection(request)?;
    let inputs = protocol::minimax_h3_image_inputs(ENVS, tier)?;
    let adapters = if plan.lora {
        vec![protocol::minimax_h3_image_lora_adapter()?]
    } else {
        Vec::new()
    };
    let overlay = plan.lora.then_some("lora");
    let spec = protocol::minimax_h3_image_load_spec(&inputs, plan.load_shape, adapters);

    let load_started = Instant::now();
    let generator = runtime_macos::catalog()
        .map_err(|error| format!("build MLX catalog: {error}"))?
        .media()
        .load(protocol::MINIMAX_H3_IMAGE_PROVIDER, &spec)
        .map_err(|error| format!("load real minimax_h3_image provider: {error}"))?;
    let load_millis = millis(load_started);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded minimax_h3_image exposed no memory-strategy contract".to_owned())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned minimax_h3_image provider rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition_for_selection(&selection),
    )?;
    let calibration = contract.calibration.as_ref();
    if let Some(identity) = calibration {
        if identity.fingerprint != fingerprint {
            return Err(format!(
                "plan/provider calibration mismatch: plan={fingerprint}, pinned provider={}",
                identity.fingerprint
            ));
        }
    }
    if contract.load_shape != plan.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(plan.load_shape),
            load_shape_key(contract.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let admitted = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        context(
            &selection,
            calibration,
            fingerprint,
            plan.load_shape,
            overlay,
            total_bytes,
            predicted,
        )
    };
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&admitted(fingerprint, total_bytes, predicted))
    };
    // The scope an optimized row renders under (`None` for a resident row — see `generate_scoped`).
    let scoped = selection
        .strategy
        .is_optimized()
        .then(|| admitted(fingerprint, hardware_bytes, 1));
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the rejections below
    // cannot pass through a blanket refusal.
    if !matches!(
        safety(fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "minimax_h3_image admission rejected a fitting probe budget; the scenario rejections \
             below would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("minimax_h3_image admission accepted an unknown/zero memory budget".to_owned());
    }
    let stale_scenario = match calibration {
        Some(_) => {
            if !matches!(
                safety("stale-minimax-h3-image-fingerprint", hardware_bytes, 1),
                MemorySafetyDecision::Reject { .. }
            ) {
                return Err(
                    "minimax_h3_image admission accepted stale calibration evidence".to_owned(),
                );
            }
            json!({ "name": "stale_evidence", "result": "passed" })
        }
        None => json!({
            "name": "stale_evidence",
            "result": "not_applicable",
            "reason": format!(
                "the pinned minimax_h3_image provider at inference {} publishes no calibration \
                 identity on a production load, so its admission carries no calibration \
                 handshake for stale evidence to fail",
                protocol::INFERENCE_PIN
            ),
        }),
    };

    // The anchor render: three synchronized phases, each with its wall time. Conditioning ends at
    // the first solver step, denoise at `Progress::Decoding`, decode at the end of the request.
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let conditioning_millis = Cell::new(0_u64);
    let denoise_millis = Cell::new(0_u64);
    let decode_at_return = Cell::new(None::<PhaseMemory>);
    let decode_millis_at_return = Cell::new(None::<u64>);
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let capture_request = protocol::minimax_h3_image_capture_request();
    let render_started = Instant::now();
    let phase_started = Cell::new(Instant::now());
    let selected = one_image(
        generate_scoped(
            generator.as_ref(),
            scoped.as_ref(),
            &capture_request,
            &mut |progress| match progress {
                Progress::Step { current: 1, .. } => {
                    conditioning.set(PhaseMemory::capture());
                    conditioning_millis.set(millis(phase_started.get()));
                    phase_started.set(Instant::now());
                    reset_peak_memory();
                }
                Progress::Decoding => {
                    denoise.set(PhaseMemory::capture());
                    denoise_millis.set(millis(phase_started.get()));
                    phase_started.set(Instant::now());
                    reset_peak_memory();
                }
                _ => {}
            },
            &mut || {
                decode_at_return.set(Some(PhaseMemory::capture()));
                decode_millis_at_return.set(Some(millis(phase_started.get())));
            },
        )
        .map_err(|error| format!("measured render: {error}"))?,
    )?;
    let decode = decode_at_return
        .get()
        .ok_or_else(|| "the measured minimax_h3_image render never returned".to_owned())?;
    let decode_millis = decode_millis_at_return
        .get()
        .ok_or_else(|| "the measured minimax_h3_image render never returned".to_owned())?;
    let render_millis = millis(render_started);
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized minimax_h3_image lifecycle phase reported a zero active peak (a \
             request boundary was never crossed)"
                .to_owned(),
        );
    }
    if (selected.width, selected.height) != (width, height) {
        return Err(format!(
            "the measured minimax_h3_image render is {}x{}, not the planned {width}x{height}",
            selected.width, selected.height
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    if !matches!(
        safety(fingerprint, predicted, predicted),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "minimax_h3_image admission rejected an exact-fit calibrated budget".to_owned(),
        );
    }

    // Warm-repeat determinism and allocator cleanup bounds on this exact loaded provider.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generate_scoped(
            generator.as_ref(),
            scoped.as_ref(),
            &capture_request,
            &mut |_| {},
            &mut || {},
        )
        .map_err(|error| format!("warm control: {error}"))?,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if !flux2_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "minimax_h3_image warm repeat exceeded the determinism envelope: \
             max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    reset_peak_memory();
    let warm_started = Instant::now();
    let warm = one_image(
        generate_scoped(
            generator.as_ref(),
            scoped.as_ref(),
            &capture_request,
            &mut |_| {},
            &mut || {},
        )
        .map_err(|error| format!("warm repeat: {error}"))?,
    )?;
    let warm_millis = millis(warm_started);
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "minimax_h3_image warm repeat peaked at {warm_peak} bytes, above the clean warm \
             control {clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "minimax_h3_image warm repeat retained active/cache bytes {warm_post_cleanup:?} above \
             the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
            cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(
            "minimax_h3_image second warm repeat changed the deterministic output".to_owned(),
        );
    }
    // Falsifiability: the breach is verified here and the numbers land in diagnostics.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "minimax_h3_image output mutation did not breach the determinism envelope".to_owned(),
        );
    }
    clear_cache();

    let lifecycle_blocker = concat!(
        "the pinned MiniMax-H3-Image MLX route exposes no calibration fault-injection site, so the ",
        "scoped lifecycle scenarios cannot execute; unscoped repeat determinism and allocator ",
        "cleanup bounds are attested in quality and diagnostics instead"
    );
    // Runtime-complete evidence is base-only (the harness refuses an overlay there), so a `:lora`
    // capture is gated whatever the provider publishes.
    let status = if calibration.is_some() && !plan.lora {
        "runtime_complete"
    } else {
        "gated"
    };
    let execution_path = if plan.lora {
        "the minimax_h3_image staged text-to-image path with one community LoRA"
    } else {
        "the minimax_h3_image staged text-to-image path"
    };
    let mut sweep = protocol::exact_tuple_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    let mut blockers = vec![lifecycle_blocker.to_owned()];
    if calibration.is_none() {
        blockers.push(
            "gated rather than runtime_complete: the pinned provider publishes no calibration \
             identity, so stale-evidence refusal is not demonstrable"
                .to_owned(),
        );
    }
    let mut fragment = json!({
        "status": status,
        "strategy": strategy,
        "loadShape": load_shape_key(plan.load_shape),
        "artifact": inputs.artifact_json_with_lora(plan.lora),
        "sweep": sweep,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            stale_scenario,
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, request, geometry, tier, and loaded model; cold measured run versus warm unscoped repeats",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": FLUX2_MAX_THRESHOLD,
            "meanErrorThreshold": FLUX2_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": FLUX2_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": inputs.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:minimax-h3-image-staged",
            "executed",
            blockers,
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("loadWallMilliseconds", "milliseconds", load_millis),
                ("conditioningWallMilliseconds", "milliseconds", conditioning_millis.get()),
                ("denoiseWallMilliseconds", "milliseconds", denoise_millis.get()),
                ("decodeWallMilliseconds", "milliseconds", decode_millis),
                ("renderWallMilliseconds", "milliseconds", render_millis),
                ("warmRenderWallMilliseconds", "milliseconds", warm_millis),
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("admittedGeometryWidth", "count", u64::from(width)),
                ("admittedGeometryHeight", "count", u64::from(height)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    if plan.lora {
        protocol::settle_exact_overlay_scenario(
            request,
            &mut fragment,
            "lora",
            execution_path,
            "the pinned community LoRA was installed through LoadSpec adapters for the load and \
             every measured render; its factors are resident for the whole render at every rung",
        )?;
    } else {
        protocol::settle_plain_overlay_scenario(request, &mut fragment, execution_path)?;
    }
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every planned MLX MiniMax-H3-Image row — the three base rows and their SW-F363 regime rows
    /// — is one this arm captures: the plan validates, the dispatch const is the engine id, the
    /// plan names no weights-free identity, and the weights-free contract built at the row's own
    /// load shape and adapter stack accepts the row's selection and attests exactly its engaged
    /// set. The `:optimized` rows engage every rung that contract implements.
    ///
    /// *Mutation that reds this:* change a row's fixture, identity, tier, geometry, load shape,
    /// overlay or strategy tuple in `config/memory-calibration-plan.json`, drop a regime row, or
    /// point the dispatch const elsewhere.
    #[test]
    fn the_planned_minimax_h3_image_rows_are_the_ones_this_arm_captures() {
        assert_eq!(
            MINIMAX_H3_IMAGE_PROVIDER,
            protocol::MINIMAX_H3_IMAGE_PROVIDER
        );
        assert_eq!(
            protocol::MINIMAX_H3_IMAGE_PROVIDER,
            mlx_gen_minimax_h3::h3_image::model::IMAGE_MODEL_ID
        );
        assert_ne!(
            PLAN_FINGERPRINT,
            mlx_gen_minimax_h3::h3_image::memory::STATIC_CALIBRATION_FINGERPRINT,
            "the plan must not name the weights-free registry identity"
        );
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let rows: Vec<(&String, &Value)> = plan["anchors"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.starts_with("minimax_h3_image:") && key.contains(":mlx"))
            .collect();
        let catalog = runtime_macos::catalog().expect("MLX catalog");
        let fixture = catalog
            .media()
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == protocol::MINIMAX_H3_IMAGE_PROVIDER)
            .expect("minimax_h3_image publishes a weights-free contract");
        let mut regimes = Vec::new();
        for (key, entry) in &rows {
            let parts: Vec<&str> = key.split(':').collect();
            let (tier, regime) = (parts[1], parts.get(3).copied());
            regimes.push((tier, regime));
            let mut target = json!({
                "provider": entry["provider"],
                "mode": entry["mode"],
                "tier": tier,
                "overlay": entry["overlay"],
                "geometry": entry["geometry"],
            });
            if let Some(regime) = regime {
                target["anchorRegime"] = json!(regime);
            }
            let strategy = entry
                .get("strategy")
                .cloned()
                .unwrap_or_else(|| json!({ "rung": "resident", "engagedRungs": ["resident"] }));
            let request = json!({
                "planned": {
                    "backend": LANE,
                    "loadShape": entry["loadShape"],
                    "calibrationFingerprint": entry["calibrationFingerprint"],
                    "fixture": entry["fixture"],
                    "target": target,
                    "strategy": {
                        "rung": strategy["rung"],
                        "engagedRungs": strategy["engagedRungs"],
                        "parameters": strategy.get("parameters").cloned().unwrap_or(json!({})),
                    },
                },
            });
            let plan = protocol::validate_minimax_h3_image_plan(&request, LANE, PLAN_FINGERPRINT)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(
                (plan.width, plan.height, plan.tier),
                (1024, 1024, tier),
                "{key}"
            );
            assert_eq!(plan.lora, regime == Some("lora"), "{key}");
            let mut spec = LoadSpec::new(WeightsSource::Dir("fixture".into()))
                .with_load_shape(plan.load_shape);
            if plan.lora {
                spec = spec.with_adapters(vec![mlx_gen::gen_core::AdapterSpec::new(
                    "fixture-lora.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                )]);
            }
            let contract = (fixture.contract)(&spec).expect("weights-free contract");
            let selection = planned_selection(&request).unwrap();
            contract
                .validate_selection(&selection)
                .unwrap_or_else(|error| panic!("{key}: {error}"));
            let engaged = contract.engaged_composition_for_selection(&selection);
            assert_eq!(
                attested_strategy(&request, &selection, &engaged)
                    .unwrap_or_else(|error| panic!("{key}: {error}")),
                request["planned"]["strategy"],
                "{key}"
            );
            if regime == Some("optimized") {
                let implemented: Vec<MemoryStrategy> = MemoryStrategy::ALL
                    .into_iter()
                    .filter(|rung| {
                        matches!(
                            contract
                                .capability(*rung)
                                .map(|capability| &capability.support),
                            Some(mlx_gen::gen_core::MemoryStrategySupport::Implemented)
                        )
                    })
                    .collect();
                assert_eq!(
                    engaged, implemented,
                    "{key}: every implemented rung is engaged"
                );
            }
        }
        regimes.sort();
        assert_eq!(
            regimes,
            vec![
                ("bf16", None),
                ("bf16", Some("optimized")),
                ("q4", None),
                ("q4", Some("lora")),
                ("q4", Some("optimized")),
                ("q8", None),
                ("q8", Some("optimized")),
            ],
            "MiniMax-H3-Image stages every request, so the staged base row is the lane's \
             least-optimized anchor and no :resident row is planned"
        );
    }

    /// The published packed tiers pass the catalog check and reach the env read; a tier the catalog
    /// still marks `pendingArtifact` (bf16 is the upstream root, so it has no such row) is refused
    /// by name before any env var is read.
    #[test]
    fn a_published_packed_tier_reaches_the_env_read() {
        for tier in ["q8", "q4"] {
            let error = protocol::minimax_h3_image_inputs(ENVS, tier).expect_err(tier);
            assert!(!error.contains("pendingArtifact"), "{tier}: {error}");
        }
    }
}
