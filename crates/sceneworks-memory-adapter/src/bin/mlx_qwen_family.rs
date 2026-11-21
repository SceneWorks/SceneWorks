//! MLX qwen capture owner.
use super::*;

pub(super) fn qwen_negative_mutation(image: &Image) -> Image {
    let mut mutated = image.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(64);
    }
    mutated
}

pub(super) fn qwen_quality_passes(maximum: f64, mean: f64) -> bool {
    maximum <= QWEN_MAX_THRESHOLD && mean <= QWEN_MEAN_THRESHOLD
}

pub(super) fn qwen_source_capture(
    request: &Value,
    root: &Path,
    repository: &str,
    revision: &str,
    tier: &str,
    selected: &Image,
    reference: &Image,
) -> Result<Value, String> {
    let capture_root = PathBuf::from(protocol::required_env("SCENEWORKS_MEMORY_CAPTURE_DIR")?);
    let source_prefix = protocol::required_env("SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX")?;
    if !source_prefix.starts_with("docs/calibration/")
        || source_prefix.split('/').any(|part| part == "..")
    {
        return Err(
            "SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX must stay under docs/calibration".to_owned(),
        );
    }
    let output_dir = source_prefix
        .split('/')
        .fold(capture_root, |directory, part| directory.join(part));
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| format!("create physical MLX capture directory: {error}"))?;
    let logical_case_id = protocol::planned_str(request, "/logicalCaseId")?;
    let inventory_bytes = integer(
        &protocol::required_env("SCENEWORKS_MEMORY_MODEL_BYTES")?,
        "SCENEWORKS_MEMORY_MODEL_BYTES",
    )?;
    if inventory_bytes == 0 {
        return Err("SCENEWORKS_MEMORY_MODEL_BYTES must be greater than zero".to_owned());
    }
    let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
    if inventory_sha256.len() != 64
        || !inventory_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256 must be 64 lowercase hex characters"
                .to_owned(),
        );
    }
    Ok(json!({
        "kind": "physical_mlx",
        "inputs": [{
            "role": "base",
            "path": root,
            "bytes": inventory_bytes,
            "sha256": inventory_sha256,
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        }],
        "outputs": [
            persist_physical_mlx_image(
                &output_dir, &source_prefix, logical_case_id, "selected_rgb", selected,
            )?,
            persist_physical_mlx_image(
                &output_dir, &source_prefix, logical_case_id, "reference_rgb", reference,
            )?,
        ],
        "claims": [
            "memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"
        ],
    }))
}

pub(super) fn planned_qwen_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned_tier(request)? {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported MLX numeric tier {tier:?}")),
    }
}

pub(super) fn planned_qwen_seed(request: &Value, tier: &str) -> Result<u64, String> {
    let fixture = protocol::planned_fixture(request)?;
    let prefix = format!("qwen-image-{tier}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Qwen fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Qwen fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the provider's two-step calibration request"
        ));
    }
    Ok(seed)
}

pub(super) fn qwen_load_spec(
    root: PathBuf,
    selection: &MemorySelection,
    offload: OffloadPolicy,
    load_shape: LoadShape,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(load_shape);
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    spec
}

pub(super) fn run_qwen_vae_probe(request: &Value) -> Result<Value, String> {
    if protocol::planned_optional_str(request, "/backend")? != Some("mlx") {
        return Err(
            "MLX adapter received a non-MLX planned case; run the harness with --backend mlx"
                .to_owned(),
        );
    }
    protocol::validate_plain_overlay_target(request, QWEN_PLAIN_EXECUTION_PATH)?;
    protocol::validate_still_geometry(request, "MLX Qwen VAE-probe calibration")?;
    let parameters = protocol::strategy_parameters(request)?;
    let strategy = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": parameters,
    });
    let planned_strategy = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned_strategy != &strategy {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned_strategy}, MLX adapter measured={strategy}"
        ));
    }
    let expected_failure = protocol::expected_failure(request);
    let comparison_output_bias = protocol::comparison_output_bias(parameters, expected_failure)?;
    let tile_edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    let tile_edge_i32 = i32::try_from(tile_edge)
        .map_err(|_| format!("decodeTileEdge={tile_edge} exceeds the MLX tiling API range"))?;
    let overlap_i32 = i32::try_from(overlap)
        .map_err(|_| format!("decodeOverlap={overlap} exceeds the MLX tiling API range"))?;
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::QWEN_REPOSITORY)?;
    let root = PathBuf::from(protocol::required_env("SCENEWORKS_QWEN_IMAGE_ROOT")?);
    if !root.is_dir() {
        return Err("SCENEWORKS_QWEN_IMAGE_ROOT is not a directory".to_owned());
    }
    let root = std::fs::canonicalize(root)
        .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        "bf16",
        protocol::QWEN_REPOSITORY,
    )?;
    let loadability_fingerprint = format!("{repository}@{revision}:bf16");
    let artifact = json!({
        "repository": &repository,
        "resolvedRevision": &revision,
        "variant": "bf16",
    });
    let vae = load_vae(&root)
        .map_err(|error| format!("load Qwen VAE from validated snapshot: {error}"))?;
    let latent = encoded_latent(&vae, width, height)?;

    clear_cache();
    reset_peak_memory();
    let baseline = vae
        .decode(&latent)
        .map_err(|error| format!("untiled Qwen VAE decode: {error}"))?;
    baseline
        .eval()
        .map_err(|error| format!("materialize untiled Qwen VAE decode: {error}"))?;
    let untiled_peak = get_peak_memory() as u64;

    clear_cache();
    reset_peak_memory();
    let tiled = vae
        .decode_tiled(
            &latent,
            &TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: tile_edge_i32,
                    overlap_px: overlap_i32,
                }),
                temporal: None,
            },
            None,
        )
        .map_err(|error| format!("tiled Qwen VAE decode {tile_edge}/{overlap}: {error}"))?;
    tiled
        .eval()
        .map_err(|error| format!("materialize tiled Qwen VAE decode: {error}"))?;
    let tiled_peak = get_peak_memory() as u64;
    let active = get_active_memory() as u64;
    let cache = get_cache_memory() as u64;
    let (actual_maximum, actual_mean) = decoded_max_mean_abs(&baseline, &tiled, None)?;
    let actual_passed = actual_maximum <= MAX_THRESHOLD && actual_mean <= MEAN_THRESHOLD;
    if expected_failure {
        if !actual_passed {
            return Err(format!(
                "negative control requires a passing unmodified identical-latent comparison: max={actual_maximum:.6}, mean={actual_mean:.6}"
            ));
        }
        let (mutated_maximum, mutated_mean) =
            decoded_max_mean_abs(&baseline, &tiled, comparison_output_bias)?;
        if mutated_maximum <= MAX_THRESHOLD && mutated_mean <= MEAN_THRESHOLD {
            return Err(format!(
                "negative mutation {tile_edge}/{overlap} did not breach the identical-latent threshold: max={mutated_maximum:.6}, mean={mutated_mean:.6}"
            ));
        }
        let blocker =
            "negative mutation is measured, but negative evidence cannot verify a production range";
        let mut fragment = protocol::plain_gated_fragment(
            request,
            QWEN_PLAIN_EXECUTION_PATH,
            protocol::PlainGatedFragment {
                artifact,
                sweep: sweep(parameters, false),
                blocker,
                quality: json!({
                    "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
                    "identicalLatents": true,
                    "result": "passed",
                    "maximumError": actual_maximum,
                    "meanError": actual_mean,
                    "maximumErrorThreshold": MAX_THRESHOLD,
                    "meanErrorThreshold": MEAN_THRESHOLD,
                }),
                negative_mutation: json!({
                    "parameters": parameters,
                    "measured": true,
                    "result": "failed_as_expected",
                    "maximumError": mutated_maximum,
                    "meanError": mutated_mean,
                }),
                loadability: json!({
                    "result": "passed",
                    "resolvedPathFingerprint": loadability_fingerprint,
                }),
                diagnostics: protocol::diagnostics(
                    "memory-mlx-adapter",
                    "executed",
                    [blocker.to_owned()],
                    [
                        ("untiledDecodeActivePeak", "bytes", untiled_peak),
                        ("tiledDecodeActivePeak", "bytes", tiled_peak),
                        ("postDecodeActive", "bytes", active),
                        ("postDecodeCache", "bytes", cache),
                    ],
                ),
            },
        )?;
        fragment["strategy"] = strategy;
        fragment["loadShape"] = json!(load_shape_key(QWEN_VAE_PROBE_LOAD_SHAPE));
        fragment["status"] = json!("negative_complete");
        return Ok(fragment);
    }

    let blocker = concat!(
        "the exact pinned Qwen public seam measures VAE decode active/cache memory and identical-latent ",
        "quality, but does not expose synchronized conditioning/denoise device/wired/reclaimable phase ",
        "telemetry or the required warm/cancel/error lifecycle injections"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        QWEN_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact,
            sweep: sweep(parameters, actual_passed),
            blocker,
            quality: json!({
                "contract": "identical encoded latent, tiled versus untiled Qwen VAE decode",
                "identicalLatents": true,
                "result": if actual_passed { "passed" } else { "failed" },
                "maximumError": actual_maximum,
                "meanError": actual_mean,
                "maximumErrorThreshold": MAX_THRESHOLD,
                "meanErrorThreshold": MEAN_THRESHOLD,
            }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint,
            }),
            diagnostics: protocol::diagnostics(
                "memory-mlx-adapter",
                "executed",
                [blocker.to_owned()],
                [
                    ("untiledDecodeActivePeak", "bytes", untiled_peak),
                    ("tiledDecodeActivePeak", "bytes", tiled_peak),
                    ("postDecodeActive", "bytes", active),
                    ("postDecodeCache", "bytes", cache),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(QWEN_VAE_PROBE_LOAD_SHAPE));
    Ok(fragment)
}

pub(super) fn qwen_provider_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox resting beside a blue ceramic vase, studio photograph".to_owned(),
        negative_prompt: Some("blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(2),
        ..Default::default()
    }
}

pub(super) fn qwen_provider_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
    evidence_story: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        // From the LOADED provider's own identity. The Qwen spec is not default-shaped —
        // `run_qwen_provider` passes `with_load_shape(load_shape)` and goes deferred at
        // `bounded_transformer_residency` — and the safety check compares abi, fingerprint AND
        // load shape, so a hardcoded eager can never satisfy the deferred rung.
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: true,
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
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-{evidence_story}@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn qwen_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // The Qwen provider executes the complete promotion-quality scenario suite for this exact plan
    // case. The plan enumerates every supported decode edge as its own exact record, so no record
    // claims values it did not execute and the aggregate evidence covers the published range.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

pub(super) fn run_qwen_provider(request: &Value) -> Result<Value, String> {
    if protocol::expected_failure(request) {
        return run_qwen_vae_probe(request);
    }
    protocol::validate_plain_overlay_target(request, QWEN_PROVIDER_EXECUTION_PATH)?;
    protocol::validate_still_geometry(request, "MLX Qwen base calibration")?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let seed = planned_qwen_seed(request, tier)?;
    let load_shape = planned_load_shape(request)?;
    let offload = if matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency | MemoryStrategy::BoundedTransformerResidency
    ) {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_QWEN_IMAGE_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::QWEN_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_QWEN_IMAGE_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::QWEN_REPOSITORY,
    )?;
    let spec = qwen_load_spec(root.clone(), &selection, offload, load_shape);
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load("qwen_image", &spec)
        .map_err(|error| format!("load real Qwen-Image {tier} provider: {error}"))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded qwen_image has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned Qwen provider rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Qwen provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if load_shape != calibration.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(load_shape),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = qwen_provider_context(
        selection,
        calibration,
        width,
        height,
        hardware_bytes,
        1,
        if tier == "bf16" { 15511 } else { 16353 },
    );

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    clear_cache();
    reset_peak_memory();
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            Progress::Decoding => {
                denoise.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            _ => {}
        },
    )?)?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err("a synchronized Qwen lifecycle phase reported a zero active peak".to_owned());
    }
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Qwen provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-qwen-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&qwen_provider_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected Qwen reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !qwen_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Qwen selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let warm = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !qwen_quality_passes(warm_maximum, warm_mean) {
        return Err("Qwen warm repeat changed the deterministic output".to_owned());
    }

    let cancelled = qwen_provider_request(width, height, seed);
    let cancel_signal = cancelled.cancel.clone();
    let cancel_during_decode = selection.strategy == MemoryStrategy::BoundedDecode;
    let mut cancel_triggered = false;
    let cancel_error = scoped_generate(
        generator.as_ref(),
        cancelled,
        &context,
        None,
        &mut |progress| {
            if cancel_triggered {
                return;
            }
            match progress {
                // Every denoise-oriented rung has executed its selected attention/block behavior
                // before the sampler publishes the first completed step.
                Progress::Step { current: 1, .. } if !cancel_during_decode => {
                    cancel_triggered = true;
                    cancel_signal.cancel();
                }
                // Let the native tiled decoder enter its physical work before signaling. The
                // decoder checks the shared flag between tiles, so this interrupts the active rung
                // instead of short-circuiting before it begins.
                Progress::Decoding if cancel_during_decode => {
                    cancel_triggered = true;
                    let signal = cancel_signal.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        signal.cancel();
                    });
                }
                _ => {}
            }
        },
    )
    .expect_err("in-flight Qwen cancellation must fail");
    if !cancel_triggered {
        return Err("Qwen cancellation probe never reached the active rung boundary".to_owned());
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Qwen cancellation returned the wrong error: {cancel_error}"
        ));
    }
    let cancel_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !qwen_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Qwen cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected_phase = if selection.strategy == MemoryStrategy::BoundedDecode {
        MemoryPhase::Decode
    } else {
        MemoryPhase::Denoise
    };
    let injected = scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        Some(injected_phase),
        &mut |_| {},
    )
    .expect_err("injected Qwen error must fail");
    if !injected.contains("injected memory-strategy calibration error") {
        return Err(format!(
            "Qwen error injection returned the wrong error: {injected}"
        ));
    }
    let error_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_provider_request(width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !qwen_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Qwen error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if qwen_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Qwen output mutation did not breach the production parity envelope".to_owned(),
        );
    }
    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": qwen_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "the authoritative Qwen target has no overlay" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed, conditioning, sampling, precision, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": QWEN_MAX_THRESHOLD,
            "meanErrorThreshold": QWEN_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": protocol::strategy_parameters(request)?,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:qwen-shared-ladder",
            "executed",
            [],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    fragment["sourceCapture"] = qwen_source_capture(
        request,
        &root,
        &repository,
        &revision,
        tier,
        &selected,
        &baseline,
    )?;
    protocol::settle_plain_overlay_scenario(request, &mut fragment, QWEN_PROVIDER_EXECUTION_PATH)?;
    Ok(fragment)
}
