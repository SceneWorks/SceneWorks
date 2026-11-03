//! CANDLE runtime capture owner.
use super::*;

pub(super) fn certifying_wddm_idle_config() -> StableIdleConfig {
    // GPU 1's otherwise-idle WDDM graphics residency measured 1.6 GB in run 33188922159. The
    // pinned testkit's stable-idle proof is deliberately stricter than a raised one-shot ceiling:
    // it repeats the samples, bounds drift, and rejects any pure-compute process before allowing
    // the device-level delta capture.
    StableIdleConfig::new(2.0, 5, 64, 200)
}

pub(super) fn certifying_vram_probe() -> VramProbe {
    VramProbe::start_rendered().assert_stable_idle(certifying_wddm_idle_config())
}

pub(super) fn decimal_gb_to_bytes(value: f64) -> u64 {
    (value * 1.0e9).round() as u64
}

pub(super) fn cuda_phase_metrics(device_bytes: u64) -> Value {
    // Candle's exact CUDA backend allocates directly through cudarc/CUDA and has no caching
    // allocator counter. On the required idle single-process GPU the `nvidia-smi memory.used` delta
    // is therefore the one truthful residency counter, and it is non-reclaimable: discrete CUDA
    // device allocations are physically non-pageable. So `activeBytes` carries the reading,
    // `reclaimableBytes` is a measured zero, and `allocatorBytes` is their sum by the schema-v5
    // identity. sc-18864 dropped `deviceBytes` and `wiredBytes`, which were further copies of this
    // same number under names claiming to be distinct quantities.
    json!({
        "activeBytes": device_bytes,
        "allocatorBytes": device_bytes,
        "reclaimableBytes": 0,
    })
}

pub(super) fn nvcc_runtime() -> Result<String, String> {
    let executable = if cfg!(windows) {
        PathBuf::from(protocol::required_env("CUDA_PATH")?)
            .join("bin")
            .join("nvcc.exe")
    } else {
        PathBuf::from("/usr/local/cuda/bin/nvcc")
    };
    let output = Command::new(&executable)
        .arg("--version")
        .output()
        .map_err(|error| format!("start {}: {error}", executable.display()))?;
    if !output.status.success() {
        return Err(format!("{} --version failed", executable.display()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split("release ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .map(str::trim)
        .map(str::to_owned)
        .ok_or_else(|| format!("cannot parse CUDA runtime from {}", executable.display()))
}

pub(super) fn probe() -> Result<Value, String> {
    let smi = NvidiaSmi::resolve()?;
    let fields = smi.query("index,name,compute_cap,driver_version,memory.total")?;
    let columns: Vec<_> = fields.split(',').map(str::trim).collect();
    if columns.len() != 5 {
        return Err(format!(
            "nvidia-smi returned {} fields instead of 5: {fields:?}",
            columns.len()
        ));
    }
    let total_mib: u64 = columns[4]
        .parse()
        .map_err(|error| format!("parse nvidia-smi memory.total: {error}"))?;
    Ok(json!({
        "hardware": {
            "probe": format!("{} selected through CUDA_VISIBLE_DEVICES", smi.executable.display()),
            "memoryBytes": total_mib * MIB,
            "deviceId": columns[0],
            "name": columns[1],
            "computeCapability": columns[2],
            "driverVersion": columns[3],
            "runtimeVersion": nvcc_runtime()?,
        }
    }))
}

pub(super) fn sweep(
    request: &Value,
    parameters: &Map<String, Value>,
    result: &str,
) -> Result<Value, String> {
    let fingerprint = protocol::planned_calibration_fingerprint(request)?;
    let candidates: &[(u64, u64, u64, u64)] = match fingerprint {
        "krea-turbo-cuda-phase-curves-v1" => &[(512, 128, 134_217_728, 1)],
        "krea-turbo-cuda-phase-curves-v2" => {
            &[(384, 64, 67_108_864, 1), (640, 128, 134_217_728, 2)]
        }
        other => return Err(format!("unknown Krea calibration fingerprint {other:?}")),
    };
    let current = |name: &str| parameters.get(name).and_then(Value::as_u64);
    let rows = candidates
        .iter()
        .map(|(edge, overlap, attention, window)| {
            let selected = current("decodeTileEdge") == Some(*edge)
                && current("decodeOverlap") == Some(*overlap)
                && current("attentionChunkSize") == Some(*attention)
                && current("transformerWindowSize") == Some(*window);
            json!({
                "parameters": {
                    "decodeTileEdge": edge,
                    "decodeOverlap": overlap,
                    "attentionChunkSize": attention,
                    "transformerWindowSize": window,
                },
                "result": if selected { result } else { "not_run" },
            })
        })
        .collect::<Vec<_>>();
    let values = |index: usize| {
        let mut values = candidates
            .iter()
            .map(|candidate| match index {
                0 => candidate.0,
                1 => candidate.1,
                2 => candidate.2,
                3 => candidate.3,
                _ => unreachable!("Krea sweep has exactly four axes"),
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        values
    };
    Ok(json!({
        "axes": [
            { "parameter": "decodeTileEdge", "testedValues": values(0) },
            { "parameter": "decodeOverlap", "testedValues": values(1) },
            { "parameter": "attentionChunkSize", "testedValues": values(2) },
            { "parameter": "transformerWindowSize", "testedValues": values(3) }
        ],
        "cases": rows,
        "rangeVerified": false,
    }))
}

pub(super) fn complete_sweep(
    request: &Value,
    parameters: &Map<String, Value>,
) -> Result<Value, String> {
    let mut result = sweep(request, parameters, "passed")?;
    // Every v1 record executes the only published tuple. Marking that singleton range verified
    // certifies exactly the selected parameters without claiming the unexecuted v2 candidates.
    result["rangeVerified"] = json!(true);
    Ok(result)
}

pub(super) fn artifact(repository: &str, revision: &str, tier: &str) -> Value {
    json!({
        "repository": repository,
        "resolvedRevision": revision,
        "variant": tier,
    })
}

pub(super) fn loadability_fingerprint(repository: &str, revision: &str, tier: &str) -> String {
    format!("{repository}@{revision}:{tier}")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_lifecycle_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    context: &MemoryRunContext,
    edge: u32,
    overlap: u32,
    attention: u32,
    window: u32,
    fault_phase: Option<MemoryPhase>,
    cancel_phase: Option<MemoryPhase>,
) -> Result<Option<runtime_cuda::gen_core::Image>, String> {
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin lifecycle Krea scope: {error}"))?
        .ok_or_else(|| "lifecycle Krea selection did not create a provider scope".to_owned())?;
    scope
        .configure_decode(edge, overlap, context.geometry)
        .map_err(|error| format!("configure lifecycle decode tuple: {error}"))?;
    scope
        .configure_attention(attention)
        .map_err(|error| format!("configure lifecycle attention tuple: {error}"))?;
    scope
        .materialize_transformer_window(0, window)
        .map_err(|error| format!("configure lifecycle transformer tuple: {error}"))?;

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply lifecycle request strategy: {error}"))?;
    let memory = generation
        .memory
        .as_mut()
        .ok_or_else(|| "optimized lifecycle request did not receive GenerationMemory".to_owned())?;
    if let Some(phase) = fault_phase {
        memory.authorize_calibration_fault(phase);
    }
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter lifecycle conditioning phase: {error}"))?;

    let cancel = generation.cancel.clone();
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::TextEncoder)
            if cancel_phase == Some(MemoryPhase::Conditioning) =>
        {
            cancel.cancel();
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
            if cancel_phase == Some(MemoryPhase::Denoise) {
                cancel.cancel();
            }
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        Progress::Decoding if cancel_phase == Some(MemoryPhase::Decode) => {
            cancel.cancel();
        }
        _ => {}
    });

    match (fault_phase, cancel_phase, result) {
        (None, None, Ok(runtime_cuda::gen_core::GenerationOutput::Images(mut images)))
            if images.len() == 1 =>
        {
            scope
                .leave_phase(phase)
                .map_err(|error| format!("leave successful lifecycle phase: {error}"))?;
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish successful lifecycle request: {error}"))?;
            Ok(Some(images.remove(0)))
        }
        (Some(expected), None, Err(error))
            if error.to_string().contains("injected memory-strategy calibration error")
                && error.to_string().contains(&format!("{expected:?}")) =>
        {
            scope
                .finish(MemoryRunOutcome::Error {
                    message: error.to_string(),
                })
                .map_err(|finish| format!("finish injected-error lifecycle request: {finish}"))?;
            Ok(None)
        }
        (None, Some(_), Err(runtime_cuda::gen_core::Error::Canceled)) => {
            scope
                .finish(MemoryRunOutcome::Canceled)
                .map_err(|error| format!("finish canceled lifecycle request: {error}"))?;
            Ok(None)
        }
        (expected_fault, expected_cancel, actual) => Err(format!(
            "lifecycle outcome mismatch: fault={expected_fault:?}, cancel={expected_cancel:?}, actual={}",
            match actual {
                Ok(_) => "success".to_owned(),
                Err(error) => format!("error: {error}"),
            }
        )),
    }
}

pub(super) fn execute_parity_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    baseline_context: &MemoryRunContext,
    strategy: MemoryStrategy,
    parameters: MemoryStrategyParameters,
) -> Result<runtime_cuda::gen_core::Image, String> {
    let mut context = baseline_context.clone();
    context.selection.strategy = strategy;
    context.selection.parameters = parameters;
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin parity Krea scope for {strategy:?}: {error}"))?
        .ok_or_else(|| format!("parity Krea strategy {strategy:?} did not create a scope"))?;
    if strategy.is_optimized() {
        let edge = parameters
            .decode_tile_edge
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_tile_edge"))?;
        let overlap = parameters
            .decode_overlap
            .ok_or_else(|| format!("parity {strategy:?} is missing decode_overlap"))?;
        let attention = parameters
            .attention_chunk_size
            .ok_or_else(|| format!("parity {strategy:?} is missing attention_chunk_size"))?;
        let window = parameters
            .transformer_window_size
            .ok_or_else(|| format!("parity {strategy:?} is missing transformer_window_size"))?;
        scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure parity decode tuple: {error}"))?;
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure parity attention tuple: {error}"))?;
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure parity transformer tuple: {error}"))?;
    }

    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width: context.geometry.width,
        height: context.geometry.height,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply parity request strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter parity conditioning phase: {error}"))?;
    let mut phase = MemoryPhase::Conditioning;
    let result = generator.generate(&generation, &mut |progress| match progress {
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Conditioning =>
        {
            let _ = scope.leave_phase(MemoryPhase::Conditioning);
            let _ = scope.enter_phase(MemoryPhase::Denoise);
            phase = MemoryPhase::Denoise;
        }
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer)
            if phase == MemoryPhase::Denoise =>
        {
            let _ = scope.leave_phase(MemoryPhase::Denoise);
            let _ = scope.enter_phase(MemoryPhase::Decode);
            phase = MemoryPhase::Decode;
        }
        _ => {}
    });
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "parity Krea {strategy:?} generation failed: {message}"
            ));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave parity terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish parity Krea request: {error}"))?;
    match output {
        runtime_cuda::gen_core::GenerationOutput::Images(mut images) if images.len() == 1 => {
            Ok(images.remove(0))
        }
        runtime_cuda::gen_core::GenerationOutput::Images(images) => Err(format!(
            "parity Krea {strategy:?} run returned {} images, expected 1",
            images.len()
        )),
        _ => Err(format!(
            "parity Krea {strategy:?} run returned non-image output"
        )),
    }
}

pub(super) fn pixel_error(
    reference: &runtime_cuda::gen_core::Image,
    candidate: &runtime_cuda::gen_core::Image,
) -> Result<(f64, f64), String> {
    if (reference.width, reference.height, reference.pixels.len())
        != (candidate.width, candidate.height, candidate.pixels.len())
    {
        return Err(format!(
            "parity image shape mismatch: reference={}x{}x{}, candidate={}x{}x{}",
            reference.width,
            reference.height,
            reference.pixels.len(),
            candidate.width,
            candidate.height,
            candidate.pixels.len()
        ));
    }
    if reference.pixels.is_empty() {
        return Err("parity image is empty".to_owned());
    }
    let mut maximum = 0.0_f64;
    let mut total = 0.0_f64;
    for (&left, &right) in reference.pixels.iter().zip(&candidate.pixels) {
        let error = f64::from(left.abs_diff(right)) / 255.0;
        maximum = maximum.max(error);
        total += error;
    }
    Ok((maximum, total / reference.pixels.len() as f64))
}

pub(super) fn negative_mutation(
    image: &runtime_cuda::gen_core::Image,
) -> runtime_cuda::gen_core::Image {
    let mut mutated = image.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(64);
    }
    mutated
}

pub(super) fn preflight_fragment(
    request: &Value,
    strategy: &Value,
    load_shape: LoadShape,
    blocker: String,
    measurement_name: &'static str,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: sweep(request, protocol::strategy_parameters(request)?, "failed")?,
            blocker: &blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({ "result": "not_run", "resolvedPathFingerprint": null }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter",
                "gated_before_execution",
                [blocker.clone()],
                [(measurement_name, "count", 1)],
            ),
        },
    )?;
    fragment["strategy"] = strategy.clone();
    fragment["loadShape"] = json!(load_shape_key(load_shape));
    Ok(fragment)
}

pub(super) use protocol::load_shape_key;

pub(super) use protocol::strategy_name;

pub(super) fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    protocol::planned_memory_strategy(request, "Candle")
}

pub(super) use protocol::planned_provider;

pub(super) use protocol::planned_mode;

pub(super) fn plain_execution_path(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        "qwen_image" => Ok(QWEN_PLAIN_EXECUTION_PATH),
        // sc-22729: five catalog models on one engine id; the member is named by the model id.
        SDXL_ID => Ok(sdxl_candle_arm(request)?.execution_path),
        INSTANTID_ID => Ok(INSTANTID_EXECUTION_PATH),
        "krea_2_turbo" => Ok(KREA_PLAIN_EXECUTION_PATH),
        // sc-22735: the undistilled base is its own registry id with its own artifact family.
        "krea_2_raw" => Ok(KREA_RAW_PLAIN_EXECUTION_PATH),
        "z_image_turbo" => Ok(if is_z_image_edit(request)? {
            Z_IMAGE_TURBO_EDIT_EXECUTION_PATH
        } else {
            Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH
        }),
        // sc-22724: the undistilled base is its own registry id with its own artifact family.
        "z_image" => Ok(Z_IMAGE_PLAIN_EXECUTION_PATH),
        // sc-22727: three catalog models over two registry ids; the member decides the path.
        FLUX2_DEV_ID | FLUX2_KLEIN_ID => Ok(flux2_arm(request)?
            .expect("a FLUX.2 provider always resolves a member or errors")
            .execution_path),
        // sc-22726: the two FLUX.1 base providers ride the same five-rung reference path.
        "flux1_dev" => Ok(FLUX1_DEV_PLAIN_EXECUTION_PATH),
        "flux1_schnell" => Ok(FLUX1_SCHNELL_PLAIN_EXECUTION_PATH),
        // sc-22726: PuLID-FLUX is a bespoke route with its own arm; it is named here so the shared
        // refusal cannot claim this adapter does not implement it.
        "pulid_flux" => Ok(PULID_FLUX_EXECUTION_PATH),
        // sc-22733: the six Mage-Flow routes, each its own registered engine provider. Spelled out
        // one bare literal per arm, like the FLUX.1 rows above: `adapterCapturableProviders`
        // INTERSECTS every dispatch gate carrying the refusal phrase and refuses any arm pattern
        // that is not a literal or a single `&str` const, so a `provider if …` guard would make the
        // derived report throw instead of listing these six as capturable.
        "mage_flow" => mage_execution_path("mage_flow"),
        "mage_flow_base" => mage_execution_path("mage_flow_base"),
        "mage_flow_turbo" => mage_execution_path("mage_flow_turbo"),
        "mage_flow_edit" => mage_execution_path("mage_flow_edit"),
        "mage_flow_edit_base" => mage_execution_path("mage_flow_edit_base"),
        "mage_flow_edit_turbo" => mage_execution_path("mage_flow_edit_turbo"),
        // sc-22730: the three SD3.5 base providers ride the same five-rung reference path.
        SD3_5_LARGE_ID => Ok(SD3_5_LARGE_PLAIN_EXECUTION_PATH),
        SD3_5_LARGE_TURBO_ID => Ok(SD3_5_LARGE_TURBO_PLAIN_EXECUTION_PATH),
        SD3_5_MEDIUM_ID => Ok(SD3_5_MEDIUM_PLAIN_EXECUTION_PATH),
        // sc-22731: the SANA and Chroma1 families ride the same five-rung reference path.
        "sana_1600m" => Ok(SANA_PLAIN_EXECUTION_PATH),
        "sana_sprint_1600m" => Ok(SANA_SPRINT_PLAIN_EXECUTION_PATH),
        "chroma1_hd" => Ok(CHROMA1_HD_PLAIN_EXECUTION_PATH),
        "chroma1_base" => Ok(CHROMA1_BASE_PLAIN_EXECUTION_PATH),
        "chroma1_flash" => Ok(CHROMA1_FLASH_PLAIN_EXECUTION_PATH),
        // sc-22737: the STILL Bernini entry. `bernini` and `bernini_image` are two catalog entries
        // on ONE engine provider id, and only the still one reaches this far — the video one is
        // dispatched above `validate_still_geometry` by `sc22737_video_arm`. Resolved by name
        // rather than assumed, so a video plan row that somehow reached the still gate is refused
        // naming the member instead of silently borrowing the still route's execution path.
        BERNINI_CANDLE_ID => {
            bernini_candle_still_member(request).map(|_| BERNINI_CANDLE_IMAGE_EXECUTION_PATH)
        }
        // sc-22732: the five turnkey still members ride the same five-rung reference path.
        KOLORS_ID => Ok(KOLORS_PLAIN_EXECUTION_PATH),
        IDEOGRAM_ID => Ok(IDEOGRAM_PLAIN_EXECUTION_PATH),
        IDEOGRAM_TURBO_ID => Ok(IDEOGRAM_TURBO_PLAIN_EXECUTION_PATH),
        LENS_ID => Ok(LENS_PLAIN_EXECUTION_PATH),
        LENS_TURBO_ID => Ok(LENS_TURBO_PLAIN_EXECUTION_PATH),
        // sc-24112: Qwen-Image 2.1 rides the same five-rung reference path as the turnkey family.
        QWEN_IMAGE_2_1_ID => Ok(QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH),
        // sc-25907: and its distilled Turbo sibling, under its own sentence.
        QWEN_IMAGE_2_1_TURBO_ID => Ok(QWEN_IMAGE_2_1_TURBO_PLAIN_EXECUTION_PATH),
        // sc-22734: six catalog models over these two ids, so the execution path is a property of
        // the MEMBER, not the engine — resolved from `(provider, modelId)`.
        SENSENOVA_ID | SENSENOVA_FAST_ID => Ok(sensenova_candle_arm(request)?.execution_path),
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// The calibration label this Candle target refuses a non-still geometry under (sc-18808).
///
/// BOTH Candle arms are image arms, and both carried the same latent defect the MLX image arms did:
/// they read only `width`/`height` through [`protocol::target_geometry`] and then wrote `frames: 1`
/// straight into `MemoryGeometry`. A plan row declaring `frames: 2` would therefore have rendered ONE
/// frame and emitted a well-formed record whose geometry envelope claimed a single frame it was never
/// asked for — the exact defect class this apparatus exists to make impossible.
///
/// No Candle plan row declares a non-unit frames axis today (all 154 rows are `frames: 1`), so this
/// is not a live exposure. It is added anyway because epic 18803 IS the video lane and
/// `ltx_2_3_distilled` is a Candle engine id: the shape becomes reachable, and a refusal is the only
/// thing that keeps the record honest when it does. Mirrors [`plain_execution_path`] so a provider
/// this adapter does not implement is rejected by the same sentence in both.
pub(super) fn still_calibration_label(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        QWEN_ID => Ok(QWEN_STILL_CALIBRATION),
        KREA_ID => Ok(KREA_STILL_CALIBRATION),
        KREA_RAW_ID => Ok(KREA_RAW_STILL_CALIBRATION),
        Z_IMAGE_TURBO_ID => Ok(if is_z_image_edit(request)? {
            Z_IMAGE_TURBO_EDIT_STILL_CALIBRATION
        } else {
            Z_IMAGE_TURBO_STILL_CALIBRATION
        }),
        Z_IMAGE_ID => Ok(Z_IMAGE_STILL_CALIBRATION),
        SDXL_ID => Ok(sdxl_candle_arm(request)?.still_calibration),
        INSTANTID_ID => Ok(INSTANTID_STILL_CALIBRATION),
        FLUX2_DEV_ID | FLUX2_KLEIN_ID => Ok(flux2_arm(request)?
            .expect("a FLUX.2 provider always resolves a member or errors")
            .still_calibration),
        FLUX1_DEV_ID => Ok(FLUX1_DEV_STILL_CALIBRATION),
        FLUX1_SCHNELL_ID => Ok(FLUX1_SCHNELL_STILL_CALIBRATION),
        PULID_FLUX_ID => Ok(PULID_FLUX_STILL_CALIBRATION),
        // One bare literal per arm — see the note in `plain_execution_path`.
        "mage_flow" => mage_still_calibration("mage_flow"),
        "mage_flow_base" => mage_still_calibration("mage_flow_base"),
        "mage_flow_turbo" => mage_still_calibration("mage_flow_turbo"),
        "mage_flow_edit" => mage_still_calibration("mage_flow_edit"),
        "mage_flow_edit_base" => mage_still_calibration("mage_flow_edit_base"),
        "mage_flow_edit_turbo" => mage_still_calibration("mage_flow_edit_turbo"),
        SD3_5_LARGE_ID => Ok(SD3_5_LARGE_STILL_CALIBRATION),
        SD3_5_LARGE_TURBO_ID => Ok(SD3_5_LARGE_TURBO_STILL_CALIBRATION),
        SD3_5_MEDIUM_ID => Ok(SD3_5_MEDIUM_STILL_CALIBRATION),
        SANA_ID => Ok(SANA_STILL_CALIBRATION),
        SANA_SPRINT_ID => Ok(SANA_SPRINT_STILL_CALIBRATION),
        CHROMA1_HD_ID => Ok(CHROMA1_HD_STILL_CALIBRATION),
        CHROMA1_BASE_ID => Ok(CHROMA1_BASE_STILL_CALIBRATION),
        CHROMA1_FLASH_ID => Ok(CHROMA1_FLASH_STILL_CALIBRATION),
        // sc-22737: the still Bernini entry only — see `bernini_candle_still_member`.
        BERNINI_CANDLE_ID => {
            bernini_candle_still_member(request).map(|_| BERNINI_CANDLE_STILL_CALIBRATION)
        }
        KOLORS_ID => Ok(KOLORS_STILL_CALIBRATION),
        IDEOGRAM_ID => Ok(IDEOGRAM_STILL_CALIBRATION),
        IDEOGRAM_TURBO_ID => Ok(IDEOGRAM_TURBO_STILL_CALIBRATION),
        LENS_ID => Ok(LENS_STILL_CALIBRATION),
        LENS_TURBO_ID => Ok(LENS_TURBO_STILL_CALIBRATION),
        QWEN_IMAGE_2_1_ID => Ok(QWEN_IMAGE_2_1_STILL_CALIBRATION),
        QWEN_IMAGE_2_1_TURBO_ID => Ok(QWEN_IMAGE_2_1_TURBO_STILL_CALIBRATION),
        // sc-22734: per-member label, so a refusal names the exact route it refused.
        SENSENOVA_ID | SENSENOVA_FAST_ID => Ok(sensenova_candle_arm(request)?.still_calibration),
        provider => Err(format!(
            "Candle five-rung calibration does not implement provider {provider:?}"
        )),
    }
}

/// The numeric tier this case plans to measure, read from the plan rather than assumed.
///
/// sc-17097: this used to be hardcoded `q4`, which silently capped the Candle lane at one tier — the
/// `krea_2_turbo` turbo fit ships `q4`, `q8` and `bf16` phase curves, so two thirds of it could not be
/// re-measured at all. The MLX adapter has always derived its tier from `/target/tier`; this mirrors
/// that, and [`planned_tier_variant`] keeps the on-disk artifact bound to the same token so a q8 plan
/// can never be satisfied by q4 weights.
pub(super) fn planned_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned_tier(request)? {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported Candle numeric tier {tier:?}")),
    }
}

pub(super) fn validate_fixture_binds_tier_and_geometry(request: &Value) -> Result<(), String> {
    let provider = planned_provider(request)?;
    // sc-22733: the Mage members get the same member/tier/edge/seed/step binding.
    if let Some(arm) = mage_arm(request)? {
        return validate_mage_fixture(request, arm, planned_tier(request)?);
    }
    if matches!(provider, FLUX1_DEV_ID | FLUX1_SCHNELL_ID | PULID_FLUX_ID) {
        return validate_flux_one_fixture(request, provider, planned_tier(request)?);
    }
    // sc-22731: the SANA and Chroma1 fixtures get the same member/tier/edge/seed/step binding, so a
    // bf16 record can never be emitted against a q4 capture that merely reused the fixture string.
    if let Some(slug) = five_rung_family_slug(provider) {
        return validate_five_rung_family_fixture(request, slug, planned_tier(request)?);
    }
    // sc-22732: the turnkey still family gets the same member/tier/edge/seed/step binding.
    if let Some(slug) = turnkey_fixture_slug(provider) {
        return validate_turnkey_fixture(request, slug, planned_tier(request)?);
    }
    // sc-22730: the SD3.5 members get the same member/tier/edge/seed/step binding, so a Medium q8
    // record can never be attributed to a Large bf16 capture that merely reused the string.
    if SD3_5_FIXTURE_SLUGS.iter().any(|(id, _)| *id == provider) {
        return validate_sd35_fixture(request, provider, planned_tier(request)?);
    }
    // sc-22735: the Raw legs use the same fixture convention (`…-<tier>-<edge>-…`) and span the
    // same six (tier, lane) cells, so they take the same binding rather than a bespoke one.
    if !matches!(provider, KREA_ID | KREA_RAW_ID) {
        return Ok(());
    }
    let planned = protocol::planned(request)?;
    let fixture = protocol::plan_str(planned, "/fixture")?;
    let tier = planned_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    if width != height {
        return Err(format!(
            "Candle Krea calibration fixtures are square; planned geometry is {width}x{height}"
        ));
    }
    for token in [format!("-{tier}-"), format!("-{width}-")] {
        if !fixture.contains(&token) {
            return Err(format!(
                "planned.fixture {fixture:?} must contain {token:?} so the capture cannot be \
                 attributed to another tier or geometry"
            ));
        }
    }
    Ok(())
}

pub(super) fn numeric_tier(tier: &str) -> Result<MemoryNumericTier, String> {
    // Matches the worker's `tier_to_quant`: bf16 is the dense base, q4/q8 are the packed tiers.
    let quant = match tier {
        "bf16" => None,
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        other => return Err(format!("unsupported Candle numeric tier {other:?}")),
    };
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

/// The provider-declared component floors ACTIVE for one selected quant — the worker's
/// `candle_memory_strategy::active_component_floors`, verbatim: every declared floor must apply
/// to the selected tier for the table to be active, otherwise none is (sc-22738).
pub(super) fn active_component_floors(
    declared: &'static [ComponentPrecisionFloor],
    selected: Option<Quant>,
) -> &'static [ComponentPrecisionFloor] {
    match selected {
        Some(selected)
            if !declared.is_empty() && declared.iter().all(|floor| floor.applies_to(selected)) =>
        {
            declared
        }
        _ => &[],
    }
}

/// The numeric tier the WORKER selects for a loaded Candle provider: the planned quant plus the
/// floors the provider's own descriptor declares for it (`candle_memory_strategy::numeric_tier`,
/// which reads `media_descriptor(engine_id).capabilities.component_precision_floors` — the same
/// descriptor the loaded generator returns).
///
/// sc-22738 (CUDA run 34356681566): the six Mage-Flow q4 cells were refused at `begin` with
/// "selected tier … component_precision_floors: [] does not match loaded tier …": the Mage q4
/// rehost keeps its text encoder above tier, `candle-gen-mage` derives the loaded tier's floors
/// from that declaration (`resolved_numeric_tier` → `quant::active_component_precision_floors`),
/// and [`numeric_tier`] hard-coded an empty table. A tier is a whole-pipeline contract, so the
/// floors are part of the identity the selection must name.
pub(super) fn loaded_numeric_tier(
    tier: &str,
    descriptor: &ModelDescriptor,
) -> Result<MemoryNumericTier, String> {
    let base = numeric_tier(tier)?;
    Ok(MemoryNumericTier {
        component_precision_floors: active_component_floors(
            descriptor.capabilities.component_precision_floors,
            base.quant,
        ),
        ..base
    })
}

pub(super) fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    planned_selection_with_tier(request, numeric_tier(planned_tier(request)?)?)
}

/// [`planned_selection`] for a LOADED provider, keyed on the tier the worker would select for it.
pub(super) fn planned_selection_for(
    request: &Value,
    descriptor: &ModelDescriptor,
) -> Result<MemorySelection, String> {
    planned_selection_with_tier(
        request,
        loaded_numeric_tier(planned_tier(request)?, descriptor)?,
    )
}

pub(super) fn planned_selection_with_tier(
    request: &Value,
    tier: MemoryNumericTier,
) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            stage_residency: protocol::strategy_parameters(request)?
                .get("stageResidency")
                .map(|value| {
                    value.as_bool().ok_or_else(|| {
                        "planned.strategy.parameters.stageResidency must be boolean".to_owned()
                    })
                })
                .transpose()?,
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component: transformer_window_size
                .map(|_| TransformerComponent::Dit),
        },
        tier,
    })
}

/// Admit one planned context through the provider EXACTLY as the worker does
/// (`sceneworks_worker::memory_strategy::generate_with_scope`): the provider's own
/// `memory_strategy_safety_check` first, `begin_memory_strategy_request` only on `Accept`, and an
/// optimized selection that opened no scope refused by name.
///
/// The safety check is not advisory on this lane. `candle-gen-chroma`, `candle-gen-flux2` and
/// `candle-gen-mage` keep an `AdmissionRegistry` whose `approve` runs inside the safety check and
/// whose `begin` takes that approval back out — a `begin` with no approval on file is
/// `"<id>: memory request skipped the safety handshake"` (sc-22738, CUDA run 34356681566: every
/// Chroma1 and FLUX.2 cell). The refusal reason is also where a provider that loaded WITHOUT a
/// contract says why (`candle-gen-bernini` carries its `memory_refusal` there), which is what the
/// worker's `engine_declines_advisory_context` surfaces.
pub(super) fn open_memory_request_scope<'a>(
    generator: &'a dyn Generator,
    context: &MemoryRunContext,
    label: &str,
) -> Result<CaptureScope<'a>, String> {
    if let MemorySafetyDecision::Reject { reason } = generator.memory_strategy_safety_check(context)
    {
        return Err(format!(
            "{label}: the loaded provider refused the planned memory context: {reason}"
        ));
    }
    let scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin {label} scope: {error}"))?;
    if context.selection.strategy.is_optimized() && scope.is_none() {
        return Err(format!(
            "{label}: the provider accepted an optimized memory-strategy selection without opening \
             a request scope"
        ));
    }
    Ok(CaptureScope {
        label: label.to_owned(),
        scope,
    })
}

/// Why a loaded provider exposes no memory-strategy contract, from the provider itself.
///
/// The worker never sees a bare `None` here: `generate_with_scope` asks
/// `memory_strategy_safety_check`, and a provider that loaded without a contract answers with its
/// refusal (`candle-gen-bernini` keeps the `production_assets` error as `memory_refusal` and
/// returns it from the safety check whatever the context). The probe context is the resident
/// baseline at the planned tier: the shared default accepts it for any contract-less provider, so
/// a `Reject` can only be the provider's own reason. sc-22738 (CUDA run 34356681566): five Bernini
/// cells failed as "loaded bernini has no memory-strategy contract" with the reason discarded.
pub(super) fn missing_contract_reason(generator: &dyn Generator, request: &Value) -> String {
    let probe = MemoryRunContext {
        selection: MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: MemoryStrategyParameters::default(),
            tier: match planned_tier(request).and_then(numeric_tier) {
                Ok(tier) => tier,
                Err(error) => return format!("no memory-strategy contract ({error})"),
            },
        },
        optimization_authority: MemoryOptimizationAuthority::Resident,
        calibration_abi: runtime_cuda::gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: 0,
            height: 0,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 0,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 0,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: String::new(),
    };
    match generator.memory_strategy_safety_check(&probe) {
        MemorySafetyDecision::Reject { reason } => {
            format!("no memory-strategy contract; the provider says: {reason}")
        }
        MemorySafetyDecision::Accept => "no memory-strategy contract".to_owned(),
    }
}

pub(super) fn reference_phase(phase: MemoryPhase) -> protocol::ReferencePhase {
    match phase {
        MemoryPhase::Conditioning => protocol::ReferencePhase::Conditioning,
        MemoryPhase::Denoise => protocol::ReferencePhase::Denoise,
        MemoryPhase::Decode => protocol::ReferencePhase::Decode,
    }
}

pub(super) fn memory_phase(phase: protocol::ReferencePhase) -> MemoryPhase {
    match phase {
        protocol::ReferencePhase::Conditioning => MemoryPhase::Conditioning,
        protocol::ReferencePhase::Denoise => MemoryPhase::Denoise,
        protocol::ReferencePhase::Decode => MemoryPhase::Decode,
    }
}

pub(super) fn measured_strategy(
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

/// The calibration identity the PRODUCTION Candle contract can emit for one `(provider, tier)`
/// cell, or `None` for a provider whose identity this adapter does not model.
///
/// sc-22735. `candle-gen-krea` used to publish ONE string
/// (`krea-candle-request-scoped-staged-residency-v1`) for `krea_2_raw`, `krea_2_edit` and
/// `krea_2_turbo_edit` at all three tiers, so a bf16 Raw record was indistinguishable by
/// calibration identity from a q4 edit record. The engine now keys the identity on
/// (route, artifact-proven tier) — the tier it reads out of `transformer/config.json`, NOT
/// `spec.quantize`, which is recipe-only on a directory load. This table is the SceneWorks half of
/// that binding.
///
/// It is a PRE-LOAD check on purpose. [`run_five_rung_reference_loaded`] already compares the
/// plan's fingerprint against the LOADED contract's, but that comparison happens after a
/// multi-tens-of-GB load; a plan row naming a string no production contract can ever emit is a
/// fixture defect, and the operator should learn that before the weights are opened, not after.
/// The post-load comparison is what actually binds the record — this only makes the diagnosis
/// local, the same way [`validate_fixture_binds_tier_and_geometry`] does.
pub(super) fn candle_production_fingerprint(provider_id: &str, tier: &str) -> Option<String> {
    let route = match provider_id {
        KREA_RAW_ID => "raw",
        _ => return None,
    };
    matches!(tier, "bf16" | "q4" | "q8")
        .then(|| format!("krea-2-{route}-{tier}-cuda-staged-residency-v1"))
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
        candle_production_fingerprint(provider_id, tier),
    )
}

/// The env-free half of [`pulid_flux_binding`], so the tier and identity bindings are provable
/// without weights or a GPU.
pub(super) fn pulid_flux_binding_at(
    request: &Value,
    repository: String,
    revision: String,
    root: PathBuf,
    bundle: protocol::PulidIdentityBundle,
) -> Result<PulidFluxBinding, String> {
    if planned_provider(request)? != PULID_FLUX_ID {
        return Err(format!(
            "the Candle PuLID-FLUX arm does not implement provider {:?}",
            planned_provider(request)?
        ));
    }
    // PuLID is text-to-image-WITH-A-FACE only: the worker's `pulid_candle_available` requires
    // `character_image` and a reference, and the provider's own route gate refuses anything else.
    let mode = planned_mode(request)?;
    if mode != "character_image" {
        return Err(format!(
            "the Candle PuLID-FLUX arm does not implement mode {mode:?}; the route is \
             character_image only"
        ));
    }
    protocol::validate_exact_overlay_target(request, "identity", PULID_FLUX_EXECUTION_PATH)?;
    let tier = match planned_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_tier returned an unsupported tier"),
    };
    validate_flux_one_fixture(request, PULID_FLUX_ID, tier)?;
    // The backbone IS the FLUX.1-dev artifact on this route (`PULID_CANDLE_FLUX_REPO`), so it binds
    // the FLUX1_DEV family — and the root must still end in the PLANNED tier's directory, so a
    // stale `…/q4` export cannot satisfy a q8 or bf16 plan.
    protocol::validate_artifact_identity(&repository, &revision, protocol::FLUX1_DEV_REPOSITORY)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize SCENEWORKS_FLUX1_DEV_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::FLUX1_DEV_REPOSITORY,
    )?;
    let paths = runtime_cuda::providers::pulid::PulidFluxPaths {
        flux_base: root,
        pulid_weights: bundle.adapter.clone(),
        eva_weights: bundle.eva.clone(),
        face_dir: bundle.face_dir.clone(),
        // No LoRA adapters: the worker gates the PuLID memory ladder on
        // `request.loras.is_empty()` (`pulid_memory_ladder_eligible`), so a ladder-admitted PuLID
        // render carries none, and an anchor must measure the admitted shape.
        adapters: Vec::new(),
    };
    Ok(PulidFluxBinding {
        repository,
        revision,
        tier,
        bundle,
        paths,
    })
}

pub(super) fn pulid_flux_binding(request: &Value) -> Result<PulidFluxBinding, String> {
    let repository = protocol::required_env("SCENEWORKS_FLUX1_DEV_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_FLUX1_DEV_REVISION")?;
    let root = PathBuf::from(protocol::required_env("SCENEWORKS_FLUX1_DEV_ROOT")?);
    let bundle = protocol::pulid_identity_bundle()?;
    pulid_flux_binding_at(request, repository, revision, root, bundle)
}

/// The one request every PuLID capture renders, in the worker's shape: the manifest's photoreal
/// preset guidance and id_weight, the native sampler/scheduler defaults, and two steps so the first
/// Step callback closes a conservative conditioning envelope and the second gives denoise its own
/// measured interval before Decoding.
pub(super) fn pulid_flux_generation_request(width: u32, height: u32) -> PulidFluxRequest {
    PulidFluxRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        steps: 2,
        guidance: PULID_FLUX_GUIDANCE,
        id_weight: PULID_FLUX_ID_WEIGHT,
        seed: FLUX1_SEED,
        use_pid: false,
        ..Default::default()
    }
}

/// The admission context the worker admits this route under (`evaluate_shared_bespoke_image`,
/// `pulid_candle.rs`): the PROVIDER mode `character_image` with exactly one reference and
/// `overlay: identity`, no PiD, no request phases. `candle-gen-pulid`'s `safety_check` refuses
/// anything else by name, so this shape is not a choice.
pub(super) fn pulid_flux_context(
    selection: MemorySelection,
    calibration: &runtime_cuda::gen_core::MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("character_image".to_owned()),
        has_reference: true,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        reference_resources: None,
        overlay: Some("identity".to_owned()),
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22726@{}", protocol::INFERENCE_PIN),
    }
}

/// The `candle:pulid_flux` arm. Unlike every other Candle arm this one never touches the provider
/// registry: `candle-gen-pulid` registers no `Generator` at all, and the worker loads it as
/// `PulidFlux::load_with_memory_context(&PulidFluxPaths, ctx)` — so that is the load path measured
/// here (E4). The memory contract is likewise path-shaped rather than spec-shaped
/// (`memory_strategy::provider_contract(&paths)`).
pub(super) fn run_pulid_flux_capture(request: &Value) -> Result<Value, String> {
    use runtime_cuda::providers::pulid::{memory_strategy as pulid_memory, PulidFlux};

    // Before any environment or weight work, under this route's own label.
    protocol::validate_still_geometry(request, PULID_FLUX_STILL_CALIBRATION)?;
    let binding = pulid_flux_binding(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let selection = planned_selection(request)?;

    // The tier the SNAPSHOT actually declares (`transformer/config.json`), read through the
    // provider's own resolver. The root suffix already proved the plan and the export agree on the
    // directory NAME; this proves the weights inside it agree too, which is the half a renamed
    // directory could otherwise fake.
    let resolved = pulid_memory::resolved_numeric_tier(&binding.paths)
        .map_err(|error| format!("resolve PuLID-FLUX numeric tier: {error}"))?;
    let expected = numeric_tier(binding.tier)?;
    if (resolved.precision, resolved.quant) != (expected.precision, expected.quant) {
        return Err(format!(
            "planned tier {} does not match the tier the PuLID backbone snapshot declares \
             (precision={:?}, quant={:?})",
            binding.tier, resolved.precision, resolved.quant
        ));
    }

    let contract = pulid_memory::provider_contract(&binding.paths)
        .map_err(|error| format!("read the pinned PuLID-FLUX memory contract: {error}"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned PuLID-FLUX provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "the pinned PuLID-FLUX contract has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let planned_load_shape = protocol::planned_load_shape(request)?;
    if planned_load_shape != load_shape_key(calibration.load_shape) {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={}",
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        pulid_memory::safety_check(
            &binding.paths,
            &contract,
            &pulid_flux_context(
                selection,
                calibration,
                fingerprint,
                width,
                height,
                total_bytes,
                predicted,
            ),
        )
    };
    // Admission mutation hygiene BEFORE the expensive load: the gate must ACCEPT a fitting request,
    // so the two rejections below cannot pass through a blanket refusal.
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "PuLID-FLUX admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("PuLID-FLUX admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-pulid-flux-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("PuLID-FLUX admission accepted stale calibration evidence".to_owned());
    }

    let context = pulid_flux_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
        1,
    );
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    // The production load path: the bespoke provider, admitted at load with the exact context it
    // will then be asked to honour. `generate_with_memory_context` refuses a context that differs.
    let model = PulidFlux::load_with_memory_context(&binding.paths, context.clone())
        .map_err(|error| format!("load real pulid_flux {} provider: {error}", binding.tier))?;
    vram.end_load(load_sample);

    let reference = pulid_reference_image()?;
    let generation = pulid_flux_generation_request(width, height);
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    // No `MemoryRequestScope` exists on this route — the provider admits at load rather than
    // opening a per-request scope — so the phase boundaries are driven by the progress stream
    // alone, exactly as the MLX image arms drive theirs.
    let result =
        model.generate_with_memory_context(&context, &generation, &reference, &mut |progress| {
            let boundary = match progress {
                Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                    protocol::ReferenceBoundary::RendererLoad
                }
                Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
                Progress::Decoding => protocol::ReferenceBoundary::Decoding,
                _ => return,
            };
            let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary)
            else {
                return;
            };
            let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
            match phase {
                MemoryPhase::Conditioning => conditioning_peak_gb = peak,
                MemoryPhase::Denoise => denoise_peak_gb = peak,
                MemoryPhase::Decode => decode_peak_gb = peak,
            }
            phase = memory_phase(next);
            phase_sample = Some(vram.phase());
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
    result.map_err(|error| format!("pulid_flux measured generation failed: {error}"))?;

    let conditioning_bytes = decimal_gb_to_bytes(
        conditioning_peak_gb
            .ok_or_else(|| "pulid_flux did not expose a conditioning boundary".to_owned())?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        denoise_peak_gb.ok_or_else(|| "pulid_flux did not expose a denoise boundary".to_owned())?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| "pulid_flux did not complete decode".to_owned())?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-22726 anchor capture measures exact per-phase memory and strategy identity for the ",
        "Candle PuLID-FLUX lane; it intentionally remains gated because this run does not repeat ",
        "the full promotion-quality, negative-mutation, and lifecycle scenario suite, and the ",
        "bespoke route opens no memory-strategy request scope to inject a calibration fault into"
    );
    Ok(json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": binding.artifact_json(),
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the pinned PuLID-FLUX admission check rejected a zero/unknown budget before load" },
            { "name": "stale_evidence", "result": "passed", "reason": "the pinned PuLID-FLUX admission check rejected a mutated calibration fingerprint before load" },
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "passed", "reason": "the PuLID identity stack (adapter, EVA tower, and the three face models) was resident for the measured render and is declared as its own resident component by the pinned contract" }
        ],
        "predictedPeakBytes": null,
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "quality": { "result": "not_run" },
        "negativeMutation": Value::Null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": binding.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:pulid-flux-bespoke-reference",
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("referenceImages", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    }))
}

pub(super) fn update_warmed_retention_baseline(
    settled_after_resident: &mut Option<u64>,
    after: u64,
) -> Result<(), String> {
    if let Some(baseline) = *settled_after_resident {
        if after > baseline.saturating_add(64 * MIB) {
            return Err(format!(
                "reused Krea rung retained {} bytes above the warmed resident baseline; refusing contaminated batching",
                after - baseline
            ));
        }
    } else {
        *settled_after_resident = Some(after);
    }
    Ok(())
}

/// The facts of a loaded provider contract the worker's candle ladder prices from
/// (`anchor_component_bytes(contract.asset_facts)`, `architecture_facts_from_contract`,
/// `estimate_floor_parameters`), serialized verbatim in the contract's own field names.
pub(super) fn loaded_contract_facts(
    contract: &runtime_cuda::gen_core::MemoryProviderContract,
) -> Value {
    let assets = contract.asset_facts;
    let facts = contract.architecture_facts;
    let strategies: Vec<Value> = contract
        .strategies
        .iter()
        .map(|capability| {
            let ranges = &capability.parameters;
            json!({
                "strategy": strategy_name(capability.strategy),
                "support": format!("{:?}", capability.support),
                "engagedRungs": contract
                    .engaged_composition(capability.strategy)
                    .into_iter()
                    .map(strategy_name)
                    .collect::<Vec<_>>(),
                "parameters": {
                    "decodeTileEdges": ranges.decode_tile_edges,
                    "decodeOverlaps": ranges.decode_overlaps,
                    "attentionChunkSizes": ranges.attention_chunk_sizes,
                    "transformerWindowSizes": ranges.transformer_window_sizes,
                    "transformerWindowComponents": ranges
                        .transformer_window_components
                        .iter()
                        .map(|component| format!("{component:?}"))
                        .collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    json!({
        "providerId": contract.provider_id,
        "loadShape": load_shape_key(contract.load_shape),
        "calibrationFingerprint": contract
            .calibration
            .as_ref()
            .map(|calibration| calibration.fingerprint.clone()),
        "assetFacts": {
            "baseBytes": assets.base_bytes,
            "conditioningBytes": assets.conditioning_bytes,
            "transformerBytes": assets.transformer_bytes,
            "decoderBytes": assets.decoder_bytes,
            "overlayBytes": assets.overlay_bytes,
        },
        "architectureFacts": {
            "attentionHeads": facts.attention_heads,
            "headDim": facts.head_dim,
            "transformerBlocks": facts.transformer_blocks,
            "patchSize": facts.patch_size,
            "latentChannels": facts.latent_channels,
            "vaeSpatialScale": facts.vae_spatial_scale,
            "vaeTemporalScale": facts.vae_temporal_scale,
            "activationDtypeWidth": facts.activation_dtype_width,
        },
        "strategies": strategies,
    })
}

/// The declared materialization shape, with no per-provider variant rule (sc-22736).
///
/// [`ltx25_planned_load_shape`] asks LTX-2.5's transformer variant first, because that family's two
/// variants genuinely differ. Every other Candle video route declares no bounded-transformer
/// residency, so `memory_route_registry::evaluate_declared_candle_load_shape` hands its spec
/// straight back and the shape is simply what the plan says.
pub(super) fn planned_video_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned_load_shape(request)? {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported Candle video loadShape {other:?}")),
    }
}

/// Which cell the plan asks for, resolved from `(target.provider, target.modelId)`.
///
/// `None` — not an error — when the plan names a provider this block does not serve, so the caller
/// falls through to the arms below it. A provider this block DOES serve with an unknown model id is
/// an error: one engine id per family serves several catalog entries whose records are filed
/// separately, and nothing else distinguishes them.
pub(super) fn sc22737_video_arm(request: &Value) -> Result<Option<Sc22737VideoArm>, String> {
    let provider = planned_provider(request)?;
    if !SC22737_VIDEO_ARMS
        .iter()
        .any(|arm| arm.engine_id == provider)
    {
        return Ok(None);
    }
    let model_id = protocol::planned_model_id(request)?;
    // The still Bernini entry rides the shared five-rung reference arm below, not this one: it is an
    // image route with an image geometry, and `run_five_rung_reference` already implements exactly
    // that shape. Reported as "no video arm" so the caller falls through rather than erroring.
    if provider == BERNINI_CANDLE_ID && model_id == BERNINI_CANDLE_IMAGE_MODEL_ID {
        return Ok(None);
    }
    SC22737_VIDEO_ARMS
        .iter()
        .copied()
        .find(|arm| arm.engine_id == provider && arm.model_id == model_id)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "the Candle {provider} arm serves catalog entries {:?}, got modelId {model_id:?}",
                SC22737_VIDEO_ARMS
                    .iter()
                    .filter(|arm| arm.engine_id == provider)
                    .map(|arm| arm.model_id)
                    .collect::<Vec<_>>()
            )
        })
}

/// Reconcile the plan's target with its fixture into the exact cell this capture renders.
///
/// The fixture carries the cadence and the seed, which `planned.target.geometry` has no axis for
/// (`GeometryEnvelope` has no temporal-cadence field), and it repeats the tier and every geometry
/// axis so a bf16 record can never be emitted against a q4 capture that merely reused the string.
pub(super) fn sc22737_video_target(
    request: &Value,
    arm: Sc22737VideoArm,
) -> Result<Sc22737VideoTarget, String> {
    let planned = protocol::planned(request)?;
    let target = protocol::plan_object(planned, "/target")?;
    let mode = protocol::target_str(target, "mode")?;
    if mode != arm.mode {
        return Err(format!(
            "{} is measured in {:?} mode, got {mode:?}",
            arm.model_id, arm.mode
        ));
    }
    if matches!(arm.engine_id, BERNINI_CANDLE_ID | LTX23_CANDLE_ID)
        && protocol::target_optional_u64(target, "referenceCount") != Some(1)
    {
        return Err(format!(
            "{} requires an explicit referenceCount of 1",
            arm.model_id
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(u64::from(arm.reference_count)) {
                return Err(format!(
                    "{} requires {field} == {}; a record measured with a different reference set \
                     describes a different conditioning and, for MiniMax-H3, a different DiT \
                     partition",
                    arm.model_id, arm.reference_count
                ));
            }
        }
    }
    let tier = planned_tier(request)?.to_owned();
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let axis = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "{} renders one clip per request; got geometry.batch {batch}",
            arm.model_id
        ));
    }
    let width = axis("width")?;
    let height = axis("height")?;
    let frames = axis("frames")?;
    if frames != arm.frames {
        return Err(format!(
            "{} is measured at {} frames, got {frames}",
            arm.model_id, arm.frames
        ));
    }
    (arm.validate_geometry)(width, height, frames)?;
    let fixture = protocol::plan_str(planned, "/fixture")?;
    let prefix = format!(
        "{}-{tier}-{width}x{height}-f{frames}-fps",
        arm.fixture_prefix
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse {} fixture fps {fps:?}: {error}", arm.model_id))?;
    if !arm.legal_fps.contains(&fps) {
        return Err(format!(
            "planned.fixture declares fps {fps}, but {} publishes {:?} only",
            arm.model_id, arm.legal_fps
        ));
    }
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse {} fixture seed {seed:?}: {error}", arm.model_id))?;
    if seed != SC22737_CANDLE_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the sc-22737 Candle calibration seed \
             {SC22737_CANDLE_SEED}"
        ));
    }
    Ok(Sc22737VideoTarget {
        tier,
        width,
        height,
        frames,
        fps,
        seed,
    })
}

/// The measured request. The CONDITIONING is what selects a reference route — and, for MiniMax-H3,
/// which DiT partition the engine resolves. Each reference route carries one synthetic image,
/// using the conditioning variant its production scope validates.
pub(super) fn sc22737_generation_request(
    arm: Sc22737VideoArm,
    target: &Sc22737VideoTarget,
) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: arm.prompt.to_owned(),
        width: target.width,
        height: target.height,
        count: 1,
        seed: Some(target.seed),
        frames: Some(target.frames),
        fps: Some(target.fps),
        // Left unset for the two families whose engine owns the recipe; MiniMax-H3 renders at the
        // manifest's `limits.hardMinSteps`, which is the cheapest admissible cell of the one
        // capture this anchor is allowed.
        steps: (arm.engine_id == MINIMAX_CANDLE_ID).then_some(MINIMAX_CANDLE_STEPS),
        ..Default::default()
    };
    if arm.reference_count > 0 {
        let image = Image {
            width: target.width,
            height: target.height,
            pixels: protocol::synthetic_reference_rgb(target.width, target.height),
        };
        request.conditioning = vec![if arm.engine_id == BERNINI_CANDLE_ID {
            request.video_mode = Some("r2v".to_owned());
            Conditioning::MultiReference {
                images: vec![image],
            }
        } else {
            Conditioning::Reference {
                image,
                // The engine owns the reference conditioning strength; the request-level lever stays
                // unset, exactly as the worker's own conditioning resolver leaves it.
                strength: None,
            }
        }];
    }
    request
}

pub(super) fn sc22737_conditioning_overlay(
    arm: Sc22737VideoArm,
    generation: &GenerationRequest,
    contract: &runtime_cuda::gen_core::MemoryProviderContract,
) -> Result<Option<String>, String> {
    match arm.engine_id {
        BERNINI_CANDLE_ID => {
            use runtime_cuda::providers::bernini::memory_strategy;
            let receipt = memory_strategy::r2v_reference_receipt(arm.engine_id, generation)
                .map_err(|error| format!("Bernini capture reference receipt: {error}"))?;
            let mut overlay = format!("provider_video_mode:r2v+{receipt}");
            if let Some(adapter) = memory_strategy::adapter_receipt_axis(contract) {
                overlay.push('+');
                overlay.push_str(&adapter);
            }
            Ok(Some(overlay))
        }
        LTX23_CANDLE_ID => Ok(Some(format!(
            "reference:image:{}x{}:strength:{:08x}",
            generation.width,
            generation.height,
            1.0_f32.to_bits()
        ))),
        _ => Ok(None),
    }
}

/// Execute one sc-22737 Candle video cell: stage the family's load, prove the plan and the loaded
/// contract name the same calibration identity, then measure three phase peaks off the boundaries
/// the shipped `generate` already emits.
///
/// The shape is `run_ltx25_capture`'s, which is the established Candle video precedent: a real
/// selected provider path, a memory-strategy request scope around the render, and the
/// frames/fps/audio result validation that keeps a record from claiming a clip the engine did not
/// produce.
pub(super) fn run_sc22737_video_capture(
    request: &Value,
    arm: Sc22737VideoArm,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let target = sc22737_video_target(request, arm)?;
    let Sc22737LoadPlan {
        artifact,
        resolved_path_fingerprint,
        spec,
    } = (arm.load_plan)(request, &target)?;
    let load_shape = spec.load_shape;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(arm.engine_id, &spec)
        .map_err(|error| {
            format!(
                "load real {} {} {} generator: {error}",
                arm.engine_id, arm.model_id, target.tier
            )
        })?;
    vram.end_load(load_sample);
    let contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} has {}",
            arm.engine_id,
            missing_contract_reason(generator.as_ref(), request)
        )
    })?;
    // sc-22738: the tier the WORKER selects for this loaded provider — planned quant plus the
    // provider's declared component floors.
    let selection = planned_selection_for(request, generator.descriptor())?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            arm.engine_id
        )
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "loaded {} has no calibration identity; the cell cannot be recorded",
            arm.engine_id
        )
    })?;
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
    let mut generation = sc22737_generation_request(arm, &target);
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other(arm.mode.to_owned()),
        has_reference: arm.reference_count > 0,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: target.width,
            height: target.height,
            batch: 1,
            frames: target.frames,
            reference_count: arm.reference_count,
        },
        reference_resources: None,
        overlay: sc22737_conditioning_overlay(arm, &generation, contract)?,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22737-adapter@{}", protocol::INFERENCE_PIN),
    };
    // sc-22738: the worker's admission sequence — safety check, then begin — and a provider that
    // opens no scope for a resident selection (`candle-gen-minimax-h3`) is rendered without one,
    // exactly as `generate_with_scope` renders it.
    let mut scope = open_memory_request_scope(
        generator.as_ref(),
        &context,
        &format!("{} capture", arm.engine_id),
    )?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {} decode: {error}", arm.engine_id))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{} decode edge and overlap must be selected together",
                arm.engine_id
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure {} attention: {error}", arm.engine_id))?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure {} transformer window: {error}", arm.engine_id))?;
    }
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {} capture strategy: {error}", arm.engine_id))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {} conditioning: {error}", arm.engine_id))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut peaks = [None, None, None];
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = phase_sample.take().map(|sample| vram.end_observed(sample));
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {} {phase:?}: {error}", arm.engine_id));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {} {next:?}: {error}", arm.engine_id));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = Some(vram.end_observed(sample));
    }
    vram.end_gen(generation_sample);
    let cumulative_run_peak_bytes = decimal_gb_to_bytes(vram.report().peak_gb);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("{} generation failed: {message}", arm.engine_id));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {} terminal phase: {error}", arm.engine_id))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {} capture scope: {error}", arm.engine_id))?;
    let (frames, fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
            return Err(format!(
                "{} returned images, not a video clip",
                arm.model_id
            ))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!(
                "{} returned audio without video frames",
                arm.model_id
            ))
        }
    };
    if fps != target.fps {
        return Err(format!(
            "{} returned {fps} fps for a {} fps request",
            arm.model_id, target.fps
        ));
    }
    // The soundtrack is half of what a joint A/V family denoises; a record that did not observe one
    // is not a record of the render it claims.
    let audio = audio
        .filter(|track| !track.samples.is_empty() && track.sample_rate > 0 && track.channels > 0);
    if arm.requires_audio && audio.is_none() {
        return Err(format!(
            "{} render returned no non-empty audio track, but this family denoises video and audio \
             jointly",
            arm.model_id
        ));
    }
    let rendered = u32::try_from(frames.len())
        .map_err(|_| format!("{} frame count does not fit u32", arm.model_id))?;
    // sc-22738 sweep: a bare equality is correct for all four members of this table, and each for a
    // reason read off the pinned engines rather than assumed.
    //
    //  - **Bernini (45 frames)**: the CANDLE z16 is causal — `candle_gen_wan::vae16::WanVae16`
    //    declares `causal_temporal: true` as its own literal
    //    (`candle-gen-wan/src/vae16.rs:344-349`), deliberately distinct from the shared
    //    `VaeTiling::WAN` the MLX z16 binds, and `candle-gen-bernini/src/lib.rs:73` inherits it. So
    //    `out_f = 1 + (f_lat−1)·4 = 45` for a `1 mod 4` count. This is the one place the two lanes
    //    genuinely disagree about the same VAE: the MLX Bernini arm had to stop using an equality
    //    for exactly this reason, and this arm must not copy that change.
    //  - **LTX-2.3 (97 frames)**: `VaeTiling::LTX` is causal at ×8 and the engine refuses any
    //    request off the `1 + 8k` lattice (`candle-gen-ltx/src/lib.rs:1584-1588`); 97 is on it.
    //  - **MiniMax-H3 (124 frames)**: publishes no `VaeTiling` at all, and asserts this same
    //    equality itself immediately before returning
    //    (`candle-gen-minimax-h3/src/model.rs:1042-1047`, `:1338-1343`), on a `17n+5` lattice it
    //    refuses to leave.
    //
    // None of the four is a non-causal, non-trimming decode, so none needs
    // `protocol::vae_decoded_frame_count`; the SCAIL-2/Wan arms that do carry it in
    // `candle_wan_scail2`.
    if rendered != target.frames {
        return Err(format!(
            "{} rendered {rendered} frames for a {}-frame request",
            arm.model_id, target.frames
        ));
    }
    let frame_shapes = frames
        .iter()
        .map(|frame| (frame.width, frame.height, frame.pixels.len()))
        .collect::<Vec<_>>();
    protocol::validate_ltx25_rgb_frames(
        usize::try_from(target.frames)
            .map_err(|_| format!("{} frame count does not fit usize", arm.model_id))?,
        target.width,
        target.height,
        &frame_shapes,
    )?;
    let conditioning_bytes =
        decimal_gb_to_bytes(peaks[0].ok_or_else(|| {
            format!("{} did not expose the conditioning boundary", arm.engine_id)
        })?);
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{} did not expose the denoise boundary", arm.engine_id))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{} did not complete decode sampling", arm.engine_id))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "sc-22737 capture measured the selected real provider path on this cell; promotion remains ",
        "gated on terminal CUDA repetition/quality evidence, which this arm does not produce"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        arm.execution_path,
        protocol::PlainGatedFragment {
            artifact,
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{resolved_path_fingerprint}:{}:f{}:{}x{}:fps{}:seed{}",
                    arm.model_id,
                    target.frames,
                    target.width,
                    target.height,
                    target.fps,
                    target.seed,
                ),
            }),
            diagnostics: protocol::diagnostics(
                arm.diagnostics_lane,
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(rendered)),
                    ("renderedFps", "fps", u64::from(fps)),
                    ("renderedAudio", "count", u64::from(audio.is_some())),
                    ("referenceCount", "count", u64::from(arm.reference_count)),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}

pub(super) fn endpoint_vram_probe() -> Result<VramProbe, String> {
    let expected = sdxl_endpoint_expected_uuid()?;
    let smi = NvidiaSmi::resolve()?;
    let physical_uuid = smi.query_bounded("uuid", Duration::from_secs(5))?;
    if physical_uuid != expected {
        return Err(format!(
            "SDXL endpoint physical UUID mismatch: expected {expected}, observed {physical_uuid}"
        ));
    }
    eprintln!(
        "SDXL endpoint awaiting idle physical GPU{} UUID={expected}",
        smi.physical_id
    );
    let started = Instant::now();
    let used = wait_for_sdxl_idle(
        |remaining| {
            smi.query_bounded(
                "memory.used,memory.free,utilization.gpu,utilization.memory",
                remaining,
            )
        },
        || started.elapsed(),
        std::thread::sleep,
    )?;
    let mut probe = VramProbe::start_rendered();
    let context_sample = probe.phase();
    let uuid = sdxl_endpoint_context_uuid()?;
    probe.end_load(context_sample);
    if uuid != expected {
        return Err(format!(
            "SDXL endpoint logical CUDA0 context UUID mismatch: expected {expected}, observed {uuid}"
        ));
    }
    eprintln!("SDXL endpoint stable physical GPU{} baseline MiB: {used:?}; actual logical CUDA0 UUID={uuid}", smi.physical_id);
    Ok(probe)
}
