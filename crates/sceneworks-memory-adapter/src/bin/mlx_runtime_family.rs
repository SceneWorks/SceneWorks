//! MLX runtime capture owner.
use super::*;

/// The calibration identity the PRODUCTION MLX contract can emit for one `(provider, tier)` cell, or
/// `None` for a provider whose identity this adapter does not model. The MLX mirror of
/// [`candle_production_fingerprint`](../candle.rs) — same shape, same wording, same reason.
///
/// sc-22735. Both engines used to publish ONE calibration string per PROVIDER, so a bf16 record was
/// indistinguishable by calibration identity from a q4 one at the same provider. The engines now key
/// the identity on (route, artifact-proven tier) — the tier read out of the staged snapshot, NOT
/// `spec.quantize`, which is recipe-only on a directory load. This table is the SceneWorks half of
/// that binding.
///
/// **`krea_2_turbo` is here too.** It was carved out of the first sc-22735 pass on the argument that
/// its shared MLX key was already-priced evidence. It was not: `config/memory-anchors.json` holds no
/// measured `krea_2_turbo` MLX record at any tier — the one turbo anchor in the catalog is
/// `krea_2_turbo:candle:q4`, a Candle string — so the retired
/// `krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3` protected nothing and only kept
/// three tiers indistinguishable. The pinned provider now keys turbo per tier like the other routes.
///
/// It is a PRE-LOAD check on purpose. Both arms already compare the plan's fingerprint against the
/// LOADED contract's, but that comparison happens after a multi-tens-of-GB load; a plan row naming a
/// string no production contract can ever emit is a fixture defect, and the operator should learn
/// that before the weights are opened, not after. The post-load comparison is what actually binds
/// the record — this only makes the diagnosis local.
pub(super) fn mlx_production_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    if !matches!(tier, "bf16" | "q4" | "q8") {
        return None;
    }
    match provider_id {
        KREA_BASE_PROVIDER => Some(format!("krea-2-turbo-{tier}-mlx-shared-ladder-v1")),
        KREA_RAW_PROVIDER => Some(format!("krea-2-raw-{tier}-mlx-shared-ladder-v1")),
        KREA_REALTIME_PROVIDER => Some(format!("krea-realtime-14b-{tier}-mlx-resident-ladder-v1")),
        _ => None,
    }
}

pub(super) fn validate_planned_fingerprint_is_producible(
    provider_id: &str,
    tier: &str,
    request: &Value,
) -> Result<(), String> {
    protocol::validate_production_fingerprint(
        request,
        provider_id,
        tier,
        mlx_production_fingerprint(provider_id, tier),
    )
}

pub(super) fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("start {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub(super) fn sysctl(name: &str) -> Result<String, String> {
    command("/usr/sbin/sysctl", &["-n", name])
}

pub(super) fn integer(value: &str, label: &str) -> Result<u64, String> {
    value
        .trim()
        .parse()
        .map_err(|error| format!("parse {label}={value:?}: {error}"))
}

pub(super) fn positive_integer(value: Option<&str>, label: &str) -> Result<Option<u64>, String> {
    value
        .map(|value| integer(value, label))
        .transpose()
        .map(|value| value.filter(|value| *value > 0))
}

pub(super) fn resolve_wired_limit(
    override_bytes: Option<&str>,
    iogpu_limit_mb: Option<&str>,
    kernel_limit_bytes: Option<&str>,
    mlx_default_memory_limit: usize,
) -> Result<WiredLimit, String> {
    if let Some(value) = override_bytes {
        let bytes = integer(value, "SCENEWORKS_MLX_WIRED_LIMIT_BYTES")?;
        if bytes == 0 {
            return Err("SCENEWORKS_MLX_WIRED_LIMIT_BYTES must be greater than zero".to_owned());
        }
        return Ok(WiredLimit {
            bytes,
            source: "SCENEWORKS_MLX_WIRED_LIMIT_BYTES",
        });
    }
    if let Some(megabytes) = positive_integer(iogpu_limit_mb, "iogpu.wired_limit_mb")? {
        let bytes = megabytes
            .checked_mul(1024 * 1024)
            .ok_or_else(|| "iogpu.wired_limit_mb overflows bytes".to_owned())?;
        return Ok(WiredLimit {
            bytes,
            source: "iogpu.wired_limit_mb",
        });
    }
    if let Some(bytes) = positive_integer(kernel_limit_bytes, "kern.memorystatus_wired_mem_limit")?
    {
        return Ok(WiredLimit {
            bytes,
            source: "kern.memorystatus_wired_mem_limit",
        });
    }

    // MLX documents its untouched default memory limit as 1.5x the device's
    // recommendedMaxWorkingSetSize. This is the same real-hardware-validated derivation used by
    // the worker when the host has no explicit wired policy (sc-12178).
    let bytes = u64::try_from(mlx_default_memory_limit)
        .map_err(|_| "MLX default memory limit does not fit u64".to_owned())?
        / 3
        * 2;
    if bytes == 0 {
        return Err(
            "cannot resolve a nonzero wired ceiling from host policy or the MLX default memory limit"
                .to_owned(),
        );
    }
    Ok(WiredLimit {
        bytes,
        source: "mlx_default_memory_limit/1.5",
    })
}

pub(super) fn metal_device() -> Result<String, String> {
    let raw = command(
        "/usr/sbin/system_profiler",
        &["SPDisplaysDataType", "-json"],
    )?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("parse system_profiler JSON: {error}"))?;
    fn find_name(value: &Value) -> Option<&str> {
        match value {
            Value::Object(map) => map
                .get("sppci_model")
                .and_then(Value::as_str)
                .or_else(|| map.values().find_map(find_name)),
            Value::Array(items) => items.iter().find_map(find_name),
            _ => None,
        }
    }
    find_name(&value)
        .map(str::to_owned)
        .ok_or_else(|| "system_profiler did not report a Metal display device".to_owned())
}

pub(super) fn probe() -> Result<Value, String> {
    let memory_bytes = integer(&sysctl("hw.memsize")?, "hw.memsize")?;
    let override_bytes = std::env::var("SCENEWORKS_MLX_WIRED_LIMIT_BYTES").ok();
    let iogpu_limit_mb = sysctl("iogpu.wired_limit_mb").ok();
    let kernel_limit_bytes = sysctl("kern.memorystatus_wired_mem_limit").ok();
    let mlx_default_memory_limit = get_memory_limit();
    let wired_limit = resolve_wired_limit(
        override_bytes.as_deref(),
        iogpu_limit_mb.as_deref(),
        kernel_limit_bytes.as_deref(),
        mlx_default_memory_limit,
    )?;
    Ok(json!({
        "hardware": {
            "probe": format!(
                "sysctl + sw_vers + system_profiler + mlx_rs::memory::get_memory_limit; wired={}",
                wired_limit.source
            ),
            "memoryBytes": memory_bytes,
            "model": sysctl("hw.model")?,
            "chip": sysctl("machdep.cpu.brand_string")?,
            "osVersion": command("/usr/bin/sw_vers", &["-productVersion"])?,
            "metalDevice": metal_device()?,
            "mlxMemoryLimitBytes": mlx_default_memory_limit,
            "wiredLimitBytes": wired_limit.bytes,
        }
    }))
}

pub(super) fn encoded_latent(vae: &QwenVae, width: u32, height: u32) -> Result<Array, String> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 255 / width) as u8);
            pixels.push((y * 255 / height) as u8);
            pixels.push((((x + y) as u64 * 127) / (width as u64 + height as u64)) as u8);
        }
    }
    let image = mlx_gen::media::Image {
        width,
        height,
        pixels,
    };
    let input = mlx_gen::img2img::preprocess_init_image(&image, width, height)
        .map_err(|error| format!("preprocess encoded-latent fixture: {error}"))?;
    let latent = vae
        .encode(&input)
        .map_err(|error| format!("encode deterministic fixture: {error}"))?;
    latent
        .eval()
        .map_err(|error| format!("materialize deterministic fixture latent: {error}"))?;
    Ok(latent)
}

pub(super) fn decoded_max_mean_abs(
    left: &Array,
    right: &Array,
    comparison_output_bias: Option<f64>,
) -> Result<(f64, f64), String> {
    protocol::validate_comparison_shapes(left.shape(), right.shape())?;
    let left = left
        .reshape(&[-1])
        .map_err(|error| format!("flatten baseline decode: {error}"))?;
    let right = right
        .reshape(&[-1])
        .map_err(|error| format!("flatten tiled decode: {error}"))?;
    let left = left.as_slice::<f32>();
    let right = right.as_slice::<f32>();
    protocol::max_mean_abs(left, right, comparison_output_bias)
}

pub(super) fn scoped_generate(
    generator: &dyn Generator,
    request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    let mut observed_failure = None;
    scoped_generate_observed(
        generator,
        request,
        context,
        error_phase,
        &mut observed_failure,
        on_progress,
    )
}

pub(super) fn scoped_generate_observed(
    generator: &dyn Generator,
    request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    scoped_generate_observed_after_configuration(
        generator,
        request,
        context,
        error_phase,
        observed_failure,
        on_progress,
        None,
    )
}

pub(super) fn scoped_generate_observed_after_configuration(
    generator: &dyn Generator,
    mut request: GenerationRequest,
    context: &MemoryRunContext,
    error_phase: Option<MemoryPhase>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
    on_progress: &mut dyn FnMut(Progress),
    after_configuration: Option<AfterConfigurationHook>,
) -> Result<GenerationOutput, String> {
    if let MemorySafetyDecision::Reject { reason } = generator.memory_strategy_safety_check(context)
    {
        return Err(format!(
            "provider safety check rejected calibrated request: {reason}"
        ));
    }
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin calibrated request: {error}"))?
        .ok_or_else(|| "optimized request did not open a provider memory scope".to_owned())?;
    scope
        .configure_request(&mut request)
        .map_err(|error| format!("configure calibrated request: {error}"))?;
    if let Some(after_configuration) = after_configuration {
        if let Err(error) = after_configuration(&mut request) {
            *observed_failure = Some(ScopedGenerationFailureKind::Error);
            let finish = scope.finish(MemoryRunOutcome::Error {
                message: error.clone(),
            });
            return match finish {
                Ok(()) => Err(error),
                Err(finish) => {
                    *observed_failure = Some(ScopedGenerationFailureKind::Finish);
                    Err(format!("{error}; finish calibrated request: {finish}"))
                }
            };
        }
    }
    if let Some(phase) = error_phase {
        // The shared gen-core request floor rejects a fault phase that is not paired with an
        // explicit harness authorization, so the pair must be set together through gen-core's own
        // helper. Without this every capture aborts at its first lifecycle injection.
        request
            .memory
            .get_or_insert_with(Default::default)
            .authorize_calibration_fault(phase);
    }
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter conditioning phase: {error}"))?;
    let mut current_phase = Some(MemoryPhase::Conditioning);
    let mut phase_error = None;
    let coherence_before = gpu_view_retries();
    let result = generator.generate(&request, &mut |progress| {
        let next = match progress {
            Progress::Step { current: 1, .. } => Some(MemoryPhase::Denoise),
            Progress::Decoding => Some(MemoryPhase::Decode),
            _ => None,
        };
        if phase_error.is_none() {
            if let Some(next) = next {
                if current_phase != Some(next) {
                    if let Some(current) = current_phase {
                        if let Err(error) = scope.leave_phase(current) {
                            phase_error = Some(format!("leave {current:?} phase: {error}"));
                        }
                    }
                    if phase_error.is_none() {
                        if let Err(error) = scope.enter_phase(next) {
                            phase_error = Some(format!("enter {next:?} phase: {error}"));
                        } else {
                            current_phase = Some(next);
                        }
                    }
                }
            }
        }
        on_progress(progress);
    });
    report_gpu_view_retries(coherence_before);
    if phase_error.is_none() {
        if let Some(current) = current_phase {
            if let Err(error) = scope.leave_phase(current) {
                phase_error = Some(format!("leave {current:?} phase: {error}"));
            }
        }
    }
    let outcome = match &result {
        Ok(_) => MemoryRunOutcome::Complete,
        Err(mlx_gen::gen_core::Error::Canceled) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Canceled);
            MemoryRunOutcome::Canceled
        }
        Err(error) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Error);
            MemoryRunOutcome::Error {
                message: error.to_string(),
            }
        }
    };
    let finish = scope.finish(outcome);
    if let Some(error) = phase_error {
        return match finish {
            Ok(()) => {
                *observed_failure = Some(ScopedGenerationFailureKind::Error);
                Err(error)
            }
            Err(finish) => {
                *observed_failure = Some(ScopedGenerationFailureKind::Finish);
                Err(format!("{error}; finish calibrated request: {finish}"))
            }
        };
    }
    settle_scoped_generation(result, finish, observed_failure)
}

/// The two GPU-view coherence retry counters (sc-22414): mlx-gen's and mlx-llm's mirrored guards
/// each count every time the GPU had to be re-read before it agreed with the CPU on a freshly
/// loaded buffer. Process-global and monotonic, so a render's incidence is the delta around it.
pub(super) fn gpu_view_retries() -> (u64, u64) {
    (
        mlx_gen::coherence::retries(),
        runtime_macos::llm::primitives::coherence::retries(),
    )
}

/// Record the render's GPU-view retry incidence — the harness log is where the Mac2 reproducer's
/// expected outcome ("pass, with non-zero retries") is read. Always recorded, so a zero reads as
/// "the guard ran and saw nothing" rather than as silence.
///
/// DEFERRED, not printed here (sc-22738). This is informational, and a capture that fails after the
/// render must be named on stderr by its own failure, not by this line — see [`protocol::fail`].
pub(super) fn report_gpu_view_retries(before: (u64, u64)) {
    let after = gpu_view_retries();
    protocol::defer_note(format!(
        "GPU-view coherence retries during this render: mlx_gen={} mlx_llm={} (sc-22414)",
        after.0.saturating_sub(before.0),
        after.1.saturating_sub(before.1),
    ));
}

/// Combine the generator and request-scope terminals without losing either failure. A provider
/// generation error does not make `finish` advisory: lifecycle certification requires the scope to
/// close successfully on complete, canceled, and error outcomes alike.
pub(super) fn settle_scoped_generation(
    result: mlx_gen::gen_core::Result<GenerationOutput>,
    finish: mlx_gen::gen_core::Result<()>,
    observed_failure: &mut Option<ScopedGenerationFailureKind>,
) -> Result<GenerationOutput, String> {
    match (result, finish) {
        (Ok(output), Ok(())) => Ok(output),
        (Err(error), Ok(())) => Err(error.to_string()),
        (Err(error), Err(finish)) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Finish);
            Err(format!(
                "generate calibrated request: {error}; finish calibrated request: {finish}"
            ))
        }
        (Ok(_), Err(error)) => {
            *observed_failure = Some(ScopedGenerationFailureKind::Finish);
            Err(format!("finish calibrated request: {error}"))
        }
    }
}

pub(super) fn predicted_ceiling(bytes: u64) -> u64 {
    let with_margin = bytes.saturating_add(bytes / 20);
    with_margin
        .saturating_add(64 * MIB - 1)
        .saturating_div(64 * MIB)
        .saturating_mul(64 * MIB)
}

pub(super) fn predicted_phase_ceiling(phase: PhaseMemory, basis: PredictedPeakBasis) -> u64 {
    let bytes = match basis {
        PredictedPeakBasis::HistoricalImageAllocatorBound => phase.allocator_bytes(),
        PredictedPeakBasis::ResidentActive => phase.active,
    };
    predicted_ceiling(bytes)
}

pub(super) fn predicted_phase_peaks(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
    basis: PredictedPeakBasis,
) -> PredictedPhasePeaks {
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    PredictedPhasePeaks {
        conditioning: predicted_phase_ceiling(conditioning, basis),
        denoise: predicted_phase_ceiling(denoise, basis),
        decode: predicted_phase_ceiling(decode, basis),
        overall: predicted_phase_ceiling(overall, basis),
    }
}

pub(super) fn video_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> PredictedPhasePeaks {
    predicted_phase_peaks(conditioning, denoise, decode, VIDEO_PREDICTED_PEAK_BASIS)
}

pub(super) fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    protocol::planned_memory_strategy(request, "MLX")
}

/// The numeric tier's load shape: bf16 is the dense base and carries no quant (the worker's
/// `tier_to_quant`); q4/q8 name the packed tier, which `with_quant` re-asserts against the
/// snapshot's own packed width at load (`needs_load_time_quant` — a mismatch hard-errors).
pub(super) fn tier_precision_quant(tier: &str) -> (Precision, Option<Quant>) {
    match tier {
        "q4" => (Precision::Bf16, Some(Quant::Q4)),
        "q8" => (Precision::Bf16, Some(Quant::Q8)),
        _ => (Precision::Bf16, None),
    }
}

pub(super) fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let parameters = protocol::strategy_parameters(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    let transformer_window_component = match parameters.get("transformerWindowComponent") {
        None if transformer_window_size.is_none() => None,
        None => Some(TransformerComponent::Dit),
        Some(value) if transformer_window_size.is_none() => {
            return Err(format!(
                "planned.strategy.parameters.transformerWindowComponent requires transformerWindowSize, got {value}"
            ));
        }
        Some(value) => match value.as_str() {
            Some("dit") => Some(TransformerComponent::Dit),
            // sc-18663: `minimax_h3` publishes `transformer_window_components: [Both]` — its rung 4
            // streams the text encoder as well as the DiT, and declaring `Dit` alone would leave the
            // conditioning stage, the taller of the two at every tier, with no lever. This parser is
            // shared, so accepting the spelling here does NOT widen any other arm: each arm's own
            // validator still constrains it (`validate_sdxl_selection_parameters` requires an
            // explicit `Dit`), and `contract.validate_selection` refuses a component the pinned
            // provider does not declare. `text_encoder` stays unparsed because no pinned MLX
            // provider declares it.
            Some("both") => Some(TransformerComponent::Both),
            _ => {
                return Err(format!(
                    "planned.strategy.parameters.transformerWindowComponent must be \"dit\" or \"both\" for the implemented MLX adapter arms, got {value}"
                ));
            }
        },
    };
    let (precision, quant) = tier_precision_quant(planned_qwen_tier(request)?);
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            stage_residency: None,
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component,
        },
        tier: MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        },
    })
}

pub(super) use protocol::strategy_name;

pub(super) fn attested_strategy(
    request: &Value,
    selection: &MemorySelection,
    engaged: &[MemoryStrategy],
) -> Result<Value, String> {
    let measured = json!({
        "rung": strategy_name(selection.strategy),
        "engagedRungs": engaged.iter().copied().map(strategy_name).collect::<Vec<_>>(),
        "parameters": protocol::strategy_parameters(request)?,
    });
    let planned = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned != &measured {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned}, pinned provider measured={measured}"
        ));
    }
    Ok(measured)
}

pub(super) use protocol::load_shape_key;

/// Parse the load shape the capture plan requires. The adapter must execute this shape and then
/// attest the loaded provider's calibration identity; deriving it from the selected rung silently
/// rewrites a declared production-shaped capture (the sc-18237 Qwen q8 bounded-attention defect).
pub(super) fn planned_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned_optional_str(request, "/loadShape")? {
        Some(protocol::LOAD_SHAPE_EAGER) => Ok(LoadShape::EagerMaterialization),
        Some(protocol::LOAD_SHAPE_DEFERRED) => Ok(LoadShape::DeferredMaterialization),
        Some(other) => Err(format!(
            "planned.loadShape must be {:?} or {:?}, got {other:?}",
            protocol::LOAD_SHAPE_EAGER,
            protocol::LOAD_SHAPE_DEFERRED
        )),
        None => Err("planned.loadShape must be a string".to_owned()),
    }
}

pub(super) fn sweep(parameters: &serde_json::Map<String, Value>, passed: bool) -> Value {
    let current_edge = parameters
        .get("decodeTileEdge")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let current_overlap = parameters
        .get("decodeOverlap")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut cases: Vec<Value> = EDGES
        .into_iter()
        .map(|edge| {
            json!({
                "parameters": { "decodeTileEdge": edge, "decodeOverlap": 64 },
                "result": if u64::from(edge) == current_edge && current_overlap == 64 {
                    if passed { "passed" } else { "failed" }
                } else {
                    "not_run"
                }
            })
        })
        .collect();
    cases.push(json!({
        "parameters": {
            "decodeTileEdge": 256,
            "decodeOverlap": 32,
            "comparisonOutputBias": 0.05,
        },
        "result": if current_edge == 256 && current_overlap == 32 {
            if passed { "passed" } else { "failed" }
        } else {
            "not_run"
        }
    }));
    json!({
        "axes": [
            { "parameter": "decodeTileEdge", "testedValues": EDGES },
            { "parameter": "decodeOverlap", "testedValues": [32, 64] }
        ],
        "cases": cases,
        "rangeVerified": false,
    })
}

/// The component precision floors the LOADED provider actually promotes at the selected tier.
///
/// [`planned_selection`] cannot carry these and does not try: it parses a plan row before any
/// provider exists, while the floor table is a property of the engine build. The pinned engine
/// compares the caller's *selected* tier against the tier it *loaded* including those promotions
/// (`gen_core::standard_memory_strategy_safety_check`: "selected tier … does not match loaded
/// tier", built from `mlx-gen-mage`'s `quant::active_component_precision_floors`), so a caller must
/// re-state the descriptor's declaration filtered to the floors active at this tier — which is
/// exactly what the worker does before it admits a request (`mlx_fit_gate.rs`
/// `active_component_floors`, whose filter this mirrors). Mage promotes its Qwen3-VL text encoder
/// and the transformer head to Q8 on the q4 tier and nothing anywhere else; every other MLX arm's
/// descriptor declares no floors at all, so this stays `&[]` for them.
pub(super) fn active_component_floors(
    declared: &'static [ComponentPrecisionFloor],
    selected: Option<Quant>,
) -> &'static [ComponentPrecisionFloor] {
    match selected {
        Some(selected) if declared.iter().any(|floor| floor.applies_to(selected)) => declared,
        _ => &[],
    }
}

/// Unwrap the video output, refusing an image-shaped or audio-shaped return.
pub(super) fn video_frames(output: GenerationOutput) -> Result<(Vec<Image>, u32, bool), String> {
    match output {
        GenerationOutput::Video { frames, fps, audio } => {
            if frames.is_empty() {
                return Err("MLX LTX-2.3 render returned no frames".to_owned());
            }
            Ok((frames, fps, audio.is_some()))
        }
        GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
            Err("MLX LTX-2.3 render returned images, not a video clip".to_owned())
        }
        GenerationOutput::Audio(_) => {
            Err("MLX LTX-2.3 render returned an audio track, not a video clip".to_owned())
        }
    }
}

/// Extract a diagnostic clip while retaining enough audio identity to prove the full default-A/V
/// decoder returned a non-empty track. The samples themselves drop here, before provider cleanup.
///
/// `label` names the calibration in every refusal. It became a parameter with the second video arm
/// (sc-18663): both A/V providers need this exact unwrapping, and a `minimax_h3` capture must not
/// refuse a misrouted still under an LTX-worded message. Every LTX call site passes
/// [`LTX_VIDEO_LABEL`], so those refusals are byte-identical to the ones this function shipped with.
pub(super) fn diagnostic_video_frames(
    output: GenerationOutput,
    label: &str,
) -> Result<(Vec<Image>, u32, Option<DiagnosticAudioIdentity>), String> {
    match output {
        GenerationOutput::Video { frames, fps, audio } => {
            if frames.is_empty() {
                return Err(format!("{label} render returned no frames"));
            }
            let audio = audio
                .map(|track| {
                    Ok::<_, String>(DiagnosticAudioIdentity {
                        samples: u64::try_from(track.samples.len()).map_err(|_| {
                            format!("{label} diagnostic audio sample count must fit u64")
                        })?,
                        sample_rate: track.sample_rate,
                        channels: track.channels,
                    })
                })
                .transpose()?;
            Ok((frames, fps, audio))
        }
        GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
            Err(format!("{label} render returned images, not a video clip"))
        }
        GenerationOutput::Audio(_) => Err(format!(
            "{label} render returned an audio track, not a video clip"
        )),
    }
}

pub(super) fn validate_diagnostic_audio(
    profile: LtxCanaryProfile,
    audio: Option<DiagnosticAudioIdentity>,
) -> Result<(), String> {
    match (profile, audio) {
        (LtxCanaryProfile::Safety, None) => Ok(()),
        (LtxCanaryProfile::ProductEnvelope, Some(audio))
            if audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0 =>
        {
            Ok(())
        }
        _ => Err(format!(
            "LTX diagnostic canary returned invalid {:?} audio identity {audio:?}",
            profile
        )),
    }
}

/// Maximum, mean, and root-mean-square absolute error over EVERY frame of two clips, in [0,1]
/// units. Per-frame aggregation rather than a first-frame spot check: a temporal divergence that
/// leaves frame 0 identical is exactly the failure a video determinism contract has to catch.
pub(super) fn video_max_mean_rms_abs(
    left: &[Image],
    right: &[Image],
) -> Result<(f64, f64, f64), String> {
    if left.len() != right.len() {
        return Err(format!(
            "video frame-count mismatch: measured={} repeat={}",
            left.len(),
            right.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    let mut samples = 0_usize;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if left.width != right.width || left.height != right.height {
            return Err(format!(
                "video frame {index} changed dimensions between renders"
            ));
        }
        if left.pixels.len() != right.pixels.len() {
            return Err(format!(
                "video frame {index} changed pixel length between renders"
            ));
        }
        for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
            let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
            maximum = maximum.max(difference);
            sum += difference;
            sum_squares += difference * difference;
            samples += 1;
        }
    }
    if samples == 0 {
        return Err("video comparison had no samples".to_owned());
    }
    Ok((
        maximum,
        sum / samples as f64,
        (sum_squares / samples as f64).sqrt(),
    ))
}

/// Total bytes of the `.safetensors` shards directly under `directory`, following the HF cache's
/// blob symlinks. Used to bound the staged-residency claim against the real component footprints
/// rather than against a comment.
pub(super) fn safetensors_bytes(directory: &Path) -> Result<u64, String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    let mut total = 0_u64;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read {}: {error}", directory.display()))?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        let metadata = std::fs::metadata(&path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?;
        total = total.saturating_add(metadata.len());
    }
    if total == 0 {
        return Err(format!(
            "no .safetensors weights under {}",
            directory.display()
        ));
    }
    Ok(total)
}

pub(super) fn read_watchdog_line(stream: &mut UnixStream) -> Result<String, String> {
    let mut payload = Vec::new();
    for _ in 0..4096 {
        let mut byte = [0_u8; 1];
        stream
            .read_exact(&mut byte)
            .map_err(|error| format!("read LTX canary watchdog attestation: {error}"))?;
        if byte[0] == b'\n' {
            return String::from_utf8(payload)
                .map_err(|error| format!("LTX canary watchdog attestation is not UTF-8: {error}"));
        }
        payload.push(byte[0]);
    }
    Err("LTX canary watchdog attestation exceeded 4096 bytes".to_owned())
}

pub(super) fn campaign_entry_diagnostic(fragment: &Value, name: &str) -> Result<u64, String> {
    fragment
        .pointer("/diagnostics/measurements")
        .and_then(Value::as_array)
        .and_then(|measurements| {
            measurements
                .iter()
                .find(|measurement| measurement.get("name").and_then(Value::as_str) == Some(name))
        })
        .and_then(|measurement| measurement.get("value"))
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("contained LTX campaign omitted diagnostic {name}"))
}

/// The activation allowance the worker's video admission charges on top of a measured lower bound
/// (sc-22738): `mlx_fit_gate::spec_headroom_bytes` for a generic engine, less the fixed unified
/// reserve, which is exactly `video_jobs/wan.rs`'s `admission_headroom_bytes`. Both terms come from
/// the shared declarations, so the mirror moves with production or not at all.
///
/// The GENERIC allowance specifically: an engine that declares its own `HeadroomAllowance` gets a
/// different number in the worker, and this arm cannot resolve one — it has no `LoadSpec`. That
/// makes the mirror's requirement a floor on production's, which is the safe direction for a
/// capture-side refusal: it never refuses a capture production would refuse harder.
pub(super) fn exceeded_bound_headroom_bytes() -> u64 {
    sceneworks_core::memory_anchor::gib_to_bytes(
        (sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB
            - sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB)
            .max(0.0),
    )
}

/// The checked-in loader-closure declarations, parsed for THIS binary's currency question.
/// `sceneworks-core` deliberately exposes no packaged accessor — the runtime must never read the
/// declarations (sc-22738) — so the parse lives here, in the tooling that owns the question.
pub(super) fn packaged_anchor_loader_closures(
) -> Result<sceneworks_core::memory_anchor::AnchorLoaderClosures, String> {
    sceneworks_core::memory_anchor::load_anchor_loader_closures(
        sceneworks_core::memory_anchor::PACKAGED_ANCHOR_LOADER_CLOSURES,
    )
    .map_err(|error| format!("packaged anchor loader closures: {error}"))
}

/// The runner's currency rule, restated for the store the adapter refuses from
/// (`measure-memory-catalog.mjs` `readExceededBounds`): current iff the lane is declared and the
/// declared digest equals the bound's. An undeclared lane is NOT current — exactly as the runner
/// reads it — so a bound on a lane the declarations no longer name is re-measured, not refused.
pub(super) fn exceeded_bound_is_current(
    bound: &sceneworks_core::memory_anchor::ExceededBound,
    closures: &sceneworks_core::memory_anchor::AnchorLoaderClosures,
) -> bool {
    closures.digest_for(&bound.model_id, bound.backend)
        == Some(bound.source.loader_closure_digest.as_str())
}

/// PRODUCTION'S measured-lower-bound refusal, mirrored on the capture side (sc-22738, epic 22723).
///
/// THE INCIDENT THIS EXISTS FOR. `bernini:bf16:mlx` at 848x480x49 was admitted by production,
/// rendered for 72 minutes on a 128 GiB Mac, and was killed by the harness's footprint watchdog at
/// 97,147,294,328 bytes — still climbing, mid-VAE-decode. Nothing in the pipeline learned anything:
/// the capture stamped nothing, and the next campaign would have queued the identical render. Now
/// the stop is packaged as an [`sceneworks_core::memory_anchor::ExceededBound`], and this check —
/// at the ONE seam every arm passes through, before any provider is named or any weight is
/// touched — refuses the re-run in seconds instead of burning another 72 minutes and putting the
/// host's GPU at risk again.
///
/// It is not a measurement gate (epic 22723 E5). It refuses exactly one thing: re-running a render
/// this host has already proven it cannot finish UNDER THE LOADER CLOSURE THIS RUN LOADS. A host
/// large enough to carry the bound is admitted and measures normally, and so is every cell no hard
/// stop has ever bounded — which is all of them until one is.
///
/// ONLY A CURRENT BOUND REFUSES HERE — THIS IS TOOLING, NOT THE RUNTIME (sc-22738). The runner
/// (`measure-memory-catalog.mjs`, `readExceededBounds`/`classifyAnchor`) and this seam state ONE
/// contract: a bound whose `source.loaderClosureDigest` still equals the digest
/// `config/anchor-loader-closures.json` declares for its `(model, lane)` at the pinned inference
/// revision is CURRENT — the cell classifies `exceeded_current`, is never scheduled, and is refused
/// here in milliseconds if a campaign hands it over anyway. A bound that has STALED — the loader
/// closure moved under a pin bump or a closure edit — classifies nothing in the runner and refuses
/// nothing here: the capture proceeds under the footprint watchdog, and its outcome SUPERSEDES the
/// bound as evidence (a completed capture retires it in `extract-memory-anchors.mjs`; a new stop
/// records a new bound at the new closure). Before this conjunct existed the runner listed a
/// stale-bounded cell `runnable` and this seam refused it on `ExceededBound::refuses_host` alone,
/// so the runner reported `capture_failed` (`bernini_image:bf16:mlx` at e16c6a55e, bound
/// `exc-55ce757827be31846a13` recorded at 3b922bac6) and NO bound could ever be lifted by
/// re-measurement after an engine fix — which is the only mechanism the standing rule leaves for
/// lifting one.
///
/// Production is deliberately NOT mirrored on the currency conjunct. The runtime never demotes a
/// stale measurement: `video_admission`, `mlx_fit_gate` and `candle_memory_strategy` apply
/// `ExceededBound::refuses_host` with no currency check and keep refusing on a stale bound until a
/// re-measurement retires it from the store. Currency is the probe tooling's cue to re-capture,
/// and this seam is probe tooling.
///
/// The budget presented is the WHOLE host less the fixed unified reserve, i.e. the most generous
/// budget a capture could ever be given: the runner starts each anchor on a quiet machine, and a
/// refusal must mean "not even an empty host can carry this", never "the host was busy".
///
/// `run` drives this over the packaged store and the packaged declarations; the tests drive it
/// over a stated store so the decision can be graded on a stated store rather than on whatever the
/// packaged one happens to carry at the current pin.
pub(super) fn exceeded_bound_capture_refusal_in(
    store: &sceneworks_core::memory_anchor::MemoryAnchorStore,
    closures: &sceneworks_core::memory_anchor::AnchorLoaderClosures,
    request: &Value,
) -> Result<Option<String>, String> {
    let planned = protocol::planned(request)?;
    let target = planned
        .get("target")
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let text = |field: &str| protocol::plan_optional_str(target, &format!("/{field}"));
    let (Some(model_id), Some(provider), Some(tier), Some(mode)) = (
        text("modelId"),
        text("provider"),
        text("tier"),
        text("mode"),
    ) else {
        return Ok(None);
    };
    let geometry = target.get("geometry");
    let dimension = |name: &str| geometry.and_then(|g| g.get(name)).and_then(Value::as_u64);
    let (Some(width), Some(height), Some(frames)) =
        (dimension("width"), dimension("height"), dimension("frames"))
    else {
        return Ok(None);
    };
    let variant = match text("transformerVariant") {
        Some("distilled") => {
            Some(sceneworks_core::memory_calibration::Ltx25TransformerVariant::Distilled)
        }
        Some("dev") => Some(sceneworks_core::memory_calibration::Ltx25TransformerVariant::Dev),
        _ => None,
    };
    let decoder = match text("decoder") {
        Some("conv") => Some(sceneworks_core::memory_calibration::Ltx25Decoder::Conv),
        Some("diffvae") => Some(sceneworks_core::memory_calibration::Ltx25Decoder::DiffVae),
        _ => None,
    };
    // The currency conjunct is applied BEFORE the binding lookup, not after it: `binding_exceeded_bound`
    // returns the largest footprint among the bounds that cover the request, and grading currency
    // on that one alone would let a stale wider stop shadow a current narrower one (the current
    // bound would then refuse nothing while the runner classified the cell `exceeded_current`).
    // Only the store's CURRENT bounds are consulted, so the two tools answer the same question.
    let current = sceneworks_core::memory_anchor::MemoryAnchorStore {
        schema_version: store.schema_version,
        anchors: Vec::new(),
        analytic_only: Vec::new(),
        component_deltas: Vec::new(),
        observed_operating_envelopes: Vec::new(),
        exceeded_bounds: store
            .exceeded_bounds
            .iter()
            .filter(|bound| exceeded_bound_is_current(bound, closures))
            .cloned()
            .collect(),
    };
    let Some(bound) =
        current.binding_exceeded_bound(sceneworks_core::memory_anchor::ExceededBoundQuery {
            model_id,
            // See `ExceededBoundQuery`: a capture plan carries no catalog resolution, and both
            // omitted axes are functions of ones graded above.
            model_family: None,
            route: None,
            provider,
            backend: sceneworks_core::memory_anchor::AnchorBackend::Mlx,
            tier,
            transformer_variant: variant,
            decoder,
            mode,
            overlay: text("overlay").filter(|overlay| *overlay != "none"),
            reference_count: 0,
            width: u32::try_from(width)
                .map_err(|_| "planned geometry width overflows".to_owned())?,
            height: u32::try_from(height)
                .map_err(|_| "planned geometry height overflows".to_owned())?,
            frames: u32::try_from(frames)
                .map_err(|_| "planned geometry frames overflows".to_owned())?,
        })
    else {
        return Ok(None);
    };
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let budget_bytes = hardware_bytes.saturating_sub(sceneworks_core::memory_anchor::gib_to_bytes(
        sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB,
    ));
    let headroom_bytes = exceeded_bound_headroom_bytes();
    let required_bytes = bound.required_bytes(headroom_bytes);
    // The SAME predicate production applies (`ExceededBound::refuses_host`), over this host's own
    // figures: `hardware.memoryBytes` as the physical ceiling, and that ceiling less the fixed
    // unified reserve as the most generous budget a capture could ever be given.
    if !bound.refuses_host(hardware_bytes, budget_bytes, headroom_bytes) {
        return Ok(None);
    }
    Ok(Some(format!(
        "{model_id}:{tier}:mlx at {width}x{height}x{frames} carries a CURRENT measured lower bound \
         this host cannot carry: {bound} recorded {} bytes of physical footprint at {}x{}x{} on a \
         {}-byte host without completing ({}), under the loader closure this run loads \
         (config/anchor-loader-closures.json declares the same digest for {model_id}:mlx at the \
         pinned inference revision). This capture needs at least {required_bytes} bytes and an \
         empty host offers {budget_bytes}. Refusing before the load: the runner classifies this \
         cell exceeded_current and never schedules it, and re-running it would spend the session \
         re-measuring a stop already on file. Once the loader closure moves, the bound stales, the \
         cell is runnable again and this seam refuses nothing — the capture then supersedes the \
         bound. Production keeps refusing this request whether or not the bound is current \
         (video_admission::exceeded_bound_refusal): the runtime never demotes a measurement.",
        bound.observed_footprint_bytes,
        bound.geometry.width,
        bound.geometry.height,
        bound.geometry.frames,
        bound.host_memory_bytes,
        bound.reason,
        bound = bound.id,
    )))
}
