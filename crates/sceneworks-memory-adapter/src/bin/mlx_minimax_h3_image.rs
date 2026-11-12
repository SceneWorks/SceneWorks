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

use super::*;
use std::time::Instant;

const LANE: &str = "mlx";

/// The adapter-owned identity the MLX plan rows name (see the module docs).
pub(super) const PLAN_FINGERPRINT: &str = "minimax-h3-image-mlx-resident-v1";

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

fn context(
    selection: &MemorySelection,
    calibration: Option<&MemoryCalibrationIdentity>,
    fingerprint: &str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    let edge = protocol::MINIMAX_H3_IMAGE_EDGE;
    MemoryRunContext {
        selection: *selection,
        optimization_authority: if calibration.is_some() {
            MemoryOptimizationAuthority::Calibrated
        } else {
            MemoryOptimizationAuthority::Resident
        },
        calibration_abi: calibration
            .map(|identity| identity.abi)
            .unwrap_or(mlx_gen::gen_core::MEMORY_CALIBRATION_ABI),
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: LoadShape::EagerMaterialization,
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
        overlay: None,
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

/// The `mlx:minimax_h3_image` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Provider, mode, tier, the still, overlay, load shape, fixture and identity — all before an
    // env var or a weight.
    let (width, height, tier) =
        protocol::validate_minimax_h3_image_plan(request, LANE, PLAN_FINGERPRINT)?;
    let selection = planned_selection(request)?;
    let inputs = protocol::minimax_h3_image_inputs(ENVS, tier)?;
    let spec = protocol::minimax_h3_image_load_spec(&inputs);

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
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref();
    if let Some(identity) = calibration {
        if identity.fingerprint != PLAN_FINGERPRINT {
            return Err(format!(
                "plan/provider calibration mismatch: plan={PLAN_FINGERPRINT}, pinned provider={}",
                identity.fingerprint
            ));
        }
    }
    if contract.load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            protocol::LOAD_SHAPE_EAGER,
            load_shape_key(contract.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&context(
            &selection,
            calibration,
            fingerprint,
            total_bytes,
            predicted,
        ))
    };
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the rejections below
    // cannot pass through a blanket refusal.
    if !matches!(
        safety(PLAN_FINGERPRINT, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "minimax_h3_image admission rejected a fitting probe budget; the scenario rejections \
             below would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(PLAN_FINGERPRINT, 0, 1),
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
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let capture_request = protocol::minimax_h3_image_capture_request();
    let render_started = Instant::now();
    let phase_started = Cell::new(Instant::now());
    let selected = one_image(
        generator
            .generate(&capture_request, &mut |progress| match progress {
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
            })
            .map_err(|error| format!("generate measured minimax_h3_image render: {error}"))?,
    )?;
    let decode = PhaseMemory::capture();
    let decode_millis = millis(phase_started.get());
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
        safety(PLAN_FINGERPRINT, predicted, predicted),
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
        generator
            .generate(&capture_request, &mut |_| {})
            .map_err(|error| format!("generate warm minimax_h3_image control: {error}"))?,
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
        generator
            .generate(&capture_request, &mut |_| {})
            .map_err(|error| format!("generate warm minimax_h3_image repeat: {error}"))?,
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
    let status = if calibration.is_some() {
        "runtime_complete"
    } else {
        "gated"
    };
    let execution_path = "the minimax_h3_image staged text-to-image path";
    let mut sweep = protocol::reference_sweep(request, "passed")?;
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
        "loadShape": load_shape_key(LoadShape::EagerMaterialization),
        "artifact": inputs.artifact_json(),
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
    protocol::settle_plain_overlay_scenario(request, &mut fragment, execution_path)?;
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three planned rows are the ones this arm captures, the dispatch const is the engine id,
    /// the plan names no weights-free identity, and the weights-free contract the registry
    /// publishes implements the lane-default resident rung at every tier.
    ///
    /// *Mutation that reds this:* change a row's fixture, identity, tier or geometry in
    /// `config/memory-calibration-plan.json`, or point the dispatch const elsewhere.
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
        let rows: Vec<&String> = plan["anchors"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| key.starts_with("minimax_h3_image:") && key.ends_with(":mlx"))
            .collect();
        assert_eq!(rows.len(), 3, "one row per tier on this lane: {rows:?}");
        let catalog = runtime_macos::catalog().expect("MLX catalog");
        let fixture = catalog
            .media()
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == protocol::MINIMAX_H3_IMAGE_PROVIDER)
            .expect("minimax_h3_image publishes a weights-free contract");
        for tier in protocol::MINIMAX_H3_IMAGE_TIERS {
            let entry = &plan["anchors"][format!("minimax_h3_image:{tier}:mlx")];
            assert!(
                entry.get("strategy").is_none(),
                "the MLX lane default is resident"
            );
            let request = json!({
                "planned": {
                    "backend": LANE,
                    "loadShape": entry["loadShape"],
                    "calibrationFingerprint": entry["calibrationFingerprint"],
                    "fixture": entry["fixture"],
                    "target": {
                        "provider": entry["provider"],
                        "mode": entry["mode"],
                        "tier": tier,
                        "overlay": entry["overlay"],
                        "geometry": entry["geometry"],
                    },
                },
            });
            assert_eq!(
                protocol::validate_minimax_h3_image_plan(&request, LANE, PLAN_FINGERPRINT),
                Ok((1024, 1024, tier))
            );
            let contract = (fixture.contract)(
                &LoadSpec::new(WeightsSource::Dir("fixture".into()))
                    .with_load_shape(LoadShape::EagerMaterialization),
            )
            .expect("weights-free contract");
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::Resident)
                    .map(|capability| &capability.support),
                Some(mlx_gen::gen_core::MemoryStrategySupport::Implemented)
            ));
        }
    }

    /// A packed tier the catalog still marks `pendingArtifact` is refused by name before any env
    /// var is read, exactly as the worker refuses it.
    #[test]
    fn an_unpublished_packed_tier_is_refused_before_the_env_read() {
        for tier in ["q8", "q4"] {
            let error = protocol::minimax_h3_image_inputs(ENVS, tier).expect_err(tier);
            assert!(error.contains("pendingArtifact"), "{tier}: {error}");
        }
    }
}
