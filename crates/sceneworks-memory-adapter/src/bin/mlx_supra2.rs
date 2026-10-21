//! Physical MLX capture arm for Supra2-IMG (sc-24217, epic 24212).
//!
//! Measures the TOTAL pipeline the worker runs — Flan-T5-Base text encoder, the converted DiT and
//! the SD VAE, all resident — at the published request (256x256, seed 0, 50 Euler steps, the
//! default CFG 3.0), loaded through the seam production loads it through
//! (`runtime_macos::catalog().media().load("supra2_img", spec)`, the call
//! `crates/sceneworks-worker/src/inference_runtime.rs` wraps) with the spec the worker builds: the
//! converted bundle as `LoadSpec::weights` and the two pinned co-requisites as the `text_encoder` /
//! `vae` components, resident and eager.
//!
//! The bundle is converted in a SEPARATE adapter process before this one loads anything (or an
//! already-pinned bundle is reused) — `protocol::ensure_supra2_bundle` — so the conversion is never
//! inside a measured window. Conditioning ends at the first solver step, denoise at
//! `Progress::Decoding`, decode at the end of the request; each phase also carries its wall time.
//! After the count-1 anchor render and its warm repeats, the same loaded model renders the product
//! maximum ([`protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT`]) once, and that peak and latency travel in the
//! diagnostics: the anchor cell is the single-image coordinate, the batch is evidence beside it.
//!
//! The pinned engine publishes NO calibration identity (`compatibility_default`; "the terminal
//! campaign measures"), so there is no handshake for a stale fingerprint to fail: the
//! `stale_evidence` scenario is `not_applicable` and the record is `gated`. If a later pin publishes
//! an identity, it must equal the plan's, the stale probe must be refused, and the record is
//! `runtime_complete` — the shape every identity-carrying arm emits.

use super::*;
use mlx_gen::gen_core::supra2::{TEXT_ENCODER_COMPONENT, VAE_COMPONENT};
use std::time::Instant;

const LANE: &str = "mlx";

/// The env family the catalog runner exports for this arm (`PROVIDER_FAMILIES.supra2_img`).
pub(super) const ENVS: protocol::Supra2Envs = protocol::Supra2Envs {
    repository_env: "SCENEWORKS_SUPRA2_IMG_REPOSITORY",
    revision_env: "SCENEWORKS_SUPRA2_IMG_REVISION",
    root_env: "SCENEWORKS_SUPRA2_IMG_ROOT",
    text_encoder_env: "SCENEWORKS_SUPRA2_IMG_COMPONENT_TEXT_ENCODER",
    vae_env: "SCENEWORKS_SUPRA2_IMG_COMPONENT_VAE",
};

/// The spec the worker loads Supra2 under: the converted bundle plus both pinned components,
/// resident, eager, default (dense) precision, no quantization.
pub(super) fn load_spec(bundle: PathBuf, text_encoder: PathBuf, vae: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(bundle))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(text_encoder))
        .with_component(VAE_COMPONENT, WeightsSource::Dir(vae))
}

/// The published request at `count` images (guidance left to the engine's default CFG 3.0).
pub(super) fn capture_request(count: u32) -> GenerationRequest {
    let size = mlx_gen::gen_core::supra2::IMAGE_SIZE;
    GenerationRequest {
        prompt: protocol::SUPRA2_IMG_PROMPT.to_owned(),
        width: size,
        height: size,
        count,
        seed: Some(protocol::SUPRA2_IMG_SEED),
        steps: Some(protocol::SUPRA2_IMG_STEPS),
        ..Default::default()
    }
}

fn millis(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn context(
    selection: &MemorySelection,
    calibration: Option<&MemoryCalibrationIdentity>,
    fingerprint: &str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    let size = mlx_gen::gen_core::supra2::IMAGE_SIZE;
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
            width: size,
            height: size,
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
        evidence_revision: format!("sc-24217@{}", protocol::INFERENCE_PIN),
    }
}

fn images(output: GenerationOutput, count: u32) -> Result<Vec<Image>, String> {
    match output {
        GenerationOutput::Images(images) if images.len() == count as usize => Ok(images),
        other => Err(format!(
            "expected {count} supra2_img images, got {}",
            match other {
                GenerationOutput::Images(images) => format!("{} images", images.len()),
                _ => "a non-image output".to_owned(),
            }
        )),
    }
}

/// The `mlx:supra2_img` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Provider, mode, tier, the native still, overlay, load shape, rung, fixture and identity —
    // all before an env var or a weight.
    let (width, height) = protocol::validate_supra2_plan(request, LANE)?;
    let selection = planned_selection(request)?;
    let planned_fingerprint = protocol::supra2_calibration_fingerprint(LANE);
    let inputs = protocol::supra2_inputs(ENVS)?;
    let bundle = protocol::ensure_supra2_bundle(&inputs)?;
    let spec = load_spec(
        bundle.dir.clone(),
        inputs.text_encoder.clone(),
        inputs.vae.clone(),
    );

    let load_started = Instant::now();
    let generator = runtime_macos::catalog()
        .map_err(|error| format!("build MLX catalog: {error}"))?
        .media()
        .load(protocol::SUPRA2_IMG_PROVIDER, &spec)
        .map_err(|error| format!("load real supra2_img provider: {error}"))?;
    let load_millis = millis(load_started);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded supra2_img exposed no memory-strategy contract".to_owned())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned supra2_img provider rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref();
    if let Some(identity) = calibration {
        if identity.fingerprint != planned_fingerprint {
            return Err(format!(
                "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
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
        safety(&planned_fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "supra2_img admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&planned_fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("supra2_img admission accepted an unknown/zero memory budget".to_owned());
    }
    let stale_scenario = match calibration {
        Some(_) => {
            if !matches!(
                safety("stale-supra2-fingerprint", hardware_bytes, 1),
                MemorySafetyDecision::Reject { .. }
            ) {
                return Err("supra2_img admission accepted stale calibration evidence".to_owned());
            }
            json!({ "name": "stale_evidence", "result": "passed" })
        }
        None => json!({
            "name": "stale_evidence",
            "result": "not_applicable",
            "reason": format!(
                "the pinned supra2_img provider at inference {} publishes no calibration identity \
                 (gen-core compatibility_default contract), so its admission carries no \
                 calibration handshake for stale evidence to fail",
                protocol::INFERENCE_PIN
            ),
        }),
    };

    // The anchor render: count 1, three synchronized phases, each with its wall time.
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
    let render_started = Instant::now();
    let phase_started = Cell::new(Instant::now());
    let selected = one_image(
        generator
            .generate(&capture_request(1), &mut |progress| match progress {
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
            .map_err(|error| format!("generate measured supra2_img render: {error}"))?,
    )?;
    let decode = PhaseMemory::capture();
    let decode_millis = millis(phase_started.get());
    let render_millis = millis(render_started);
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized supra2_img lifecycle phase reported a zero active peak (a request \
             boundary was never crossed)"
                .to_owned(),
        );
    }
    if (selected.width, selected.height) != (width, height) {
        return Err(format!(
            "the measured supra2_img render is {}x{}, not the planned {width}x{height}",
            selected.width, selected.height
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    if !matches!(
        safety(&planned_fingerprint, predicted, predicted),
        MemorySafetyDecision::Accept
    ) {
        return Err("supra2_img admission rejected an exact-fit calibrated budget".to_owned());
    }

    // Warm-repeat determinism and allocator cleanup bounds on this exact loaded provider.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generator
            .generate(&capture_request(1), &mut |_| {})
            .map_err(|error| format!("generate warm supra2_img control: {error}"))?,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if !flux2_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "supra2_img warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    reset_peak_memory();
    let warm_started = Instant::now();
    let warm = one_image(
        generator
            .generate(&capture_request(1), &mut |_| {})
            .map_err(|error| format!("generate warm supra2_img repeat: {error}"))?,
    )?;
    let warm_millis = millis(warm_started);
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "supra2_img warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "supra2_img warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
             clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
            cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err("supra2_img second warm repeat changed the deterministic output".to_owned());
    }
    // Falsifiability: a runtime_complete record keeps `negativeMutation` null, so the breach is
    // verified here and the numbers land in diagnostics.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "supra2_img output mutation did not breach the determinism envelope".to_owned(),
        );
    }

    // The product maximum: one request at the job cap on the same loaded model.
    let product_count = protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT;
    clear_cache();
    reset_peak_memory();
    let batch_started = Instant::now();
    let batch = images(
        generator
            .generate(&capture_request(product_count), &mut |_| {})
            .map_err(|error| {
                format!("generate supra2_img count-{product_count} render: {error}")
            })?,
        product_count,
    )?;
    let batch_millis = millis(batch_started);
    let batch_peak = PhaseMemory::capture();
    // Image 0 of a batch uses the same per-image seed as the single render (`image_seed(seed, 0)`).
    let (batch_maximum, batch_mean, batch_rms) = image_max_mean_rms_abs(&selected, &batch[0])?;
    clear_cache();

    let lifecycle_blocker = concat!(
        "the pinned Supra2-IMG MLX crate exposes no calibration fault-injection site and opens no ",
        "memory-strategy request scope, so the scoped lifecycle scenarios cannot execute; unscoped ",
        "repeat determinism and allocator cleanup bounds are attested in quality and diagnostics ",
        "instead"
    );
    let status = if calibration.is_some() {
        "runtime_complete"
    } else {
        "gated"
    };
    let execution_path = "the supra2_img resident path";
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
            "memory-mlx-adapter:supra2-img-resident",
            "executed",
            blockers,
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("bundleConvertedInChildProcess", "count", u64::from(bundle.converted)),
                ("bundleConvertWallMilliseconds", "milliseconds", bundle.convert_millis),
                ("loadWallMilliseconds", "milliseconds", load_millis),
                ("conditioningWallMilliseconds", "milliseconds", conditioning_millis.get()),
                ("denoiseWallMilliseconds", "milliseconds", denoise_millis.get()),
                ("decodeWallMilliseconds", "milliseconds", decode_millis),
                ("count1RenderWallMilliseconds", "milliseconds", render_millis),
                ("count1WarmRenderWallMilliseconds", "milliseconds", warm_millis),
                ("productMaxCount", "count", u64::from(product_count)),
                ("productMaxCountActivePeak", "bytes", batch_peak.active),
                ("productMaxCountAllocatorEnvelope", "bytes", batch_peak.allocator_bytes()),
                ("productMaxCountRenderWallMilliseconds", "milliseconds", batch_millis),
                ("productMaxCountFirstImageMaxErrorPer255", "count", (batch_maximum * 255.0).round() as u64),
                ("productMaxCountFirstImageMeanErrorPer255", "count", (batch_mean * 255.0).round() as u64),
                ("productMaxCountFirstImageRmsErrorPer255", "count", (batch_rms * 255.0).round() as u64),
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

    /// The spec is the worker's: bundle weights, both pinned components by the engine's own ids,
    /// resident, eager, dense — the only spec `Supra2Image::load` accepts.
    ///
    /// *Mutation that reds this:* drop a component, or load `Sequential` / deferred / quantized.
    #[test]
    fn the_load_spec_is_the_worker_spec() {
        let spec = load_spec("bundle".into(), "te".into(), "vae".into());
        assert_eq!(spec.weights, WeightsSource::Dir("bundle".into()));
        assert_eq!(
            spec.components.get(TEXT_ENCODER_COMPONENT),
            Some(&WeightsSource::Dir("te".into()))
        );
        assert_eq!(
            spec.components.get(VAE_COMPONENT),
            Some(&WeightsSource::Dir("vae".into()))
        );
        assert_eq!(spec.components.len(), 2);
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
        assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(spec.precision, Precision::Bf16);
        assert_eq!(spec.quantize, None);
    }

    /// The capture request is the published default and resolves through the engine's own request
    /// boundary, at count 1 and at the product maximum.
    #[test]
    fn the_capture_requests_resolve_through_the_engines_own_boundary() {
        use mlx_gen::gen_core::supra2::{Supra2Resolved, DEFAULT_GUIDANCE};
        for count in [1, protocol::SUPRA2_IMG_PRODUCT_MAX_COUNT] {
            let resolved = Supra2Resolved::resolve(&capture_request(count))
                .unwrap_or_else(|error| panic!("count {count}: {error}"));
            assert_eq!(resolved.count, count);
            assert_eq!(resolved.steps, protocol::SUPRA2_IMG_STEPS);
            assert_eq!(resolved.guidance, DEFAULT_GUIDANCE);
            assert_eq!(resolved.seed, protocol::SUPRA2_IMG_SEED);
        }
    }

    /// The plan row this lane captures validates against the arm, and the weights-free contract the
    /// registry publishes implements the resident rung it plans.
    ///
    /// *Mutation that reds this:* change the row's fixture, identity, tier or geometry in
    /// `config/memory-calibration-plan.json`.
    #[test]
    fn the_planned_supra2_row_is_the_one_this_arm_captures() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let entry = &plan["anchors"]["supra2_img:bf16:mlx"];
        assert_eq!(
            SUPRA2_PROVIDER,
            protocol::SUPRA2_IMG_PROVIDER,
            "the dispatch const"
        );
        assert_eq!(entry["provider"], protocol::SUPRA2_IMG_PROVIDER);
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
                "strategy": { "rung": "resident", "engagedRungs": ["resident"], "parameters": {} },
                "target": {
                    "provider": entry["provider"],
                    "mode": entry["mode"],
                    "tier": "bf16",
                    "overlay": entry["overlay"],
                    "geometry": entry["geometry"],
                },
            },
        });
        assert_eq!(
            protocol::validate_supra2_plan(&request, LANE),
            Ok((256, 256))
        );
        assert!(
            plan["anchors"]
                .as_object()
                .unwrap()
                .keys()
                .filter(|key| key.starts_with("supra2_img:") && key.ends_with(":mlx"))
                .eq(["supra2_img:bf16:mlx"].iter()),
            "one Supra2 cell on this lane"
        );
        let catalog = runtime_macos::catalog().expect("MLX catalog");
        let fixture = catalog
            .media()
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == protocol::SUPRA2_IMG_PROVIDER)
            .expect("supra2_img publishes a weights-free contract");
        let contract = (fixture.contract)(&load_spec("b".into(), "t".into(), "v".into()))
            .expect("weights-free contract");
        assert!(matches!(
            contract
                .capability(MemoryStrategy::Resident)
                .map(|capability| &capability.support),
            Some(mlx_gen::gen_core::MemoryStrategySupport::Implemented)
        ));
    }
}
