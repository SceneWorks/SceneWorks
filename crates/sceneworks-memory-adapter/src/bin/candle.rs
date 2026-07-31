#[cfg(target_os = "macos")]
compile_error!("memory-candle-adapter is supported only on CUDA hosts");

use candle_gen::testkit::VramProbe;
use runtime_cuda::gen_core::{
    GenerationRequest, LoadSpec, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryPhase, MemoryRunContext, MemoryRunOutcome, MemorySelection,
    MemoryStrategy, MemoryStrategyParameters, OffloadPolicy, Precision, Progress, Quant,
    TransformerComponent, WeightsSource,
};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::process::Command;

const KREA_ID: &str = "krea_2_turbo";
const KREA_PLAIN_EXECUTION_PATH: &str = "the Candle Krea base-only text-to-image path";
const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

#[derive(Clone)]
struct NvidiaSmi {
    executable: PathBuf,
    physical_id: String,
}

impl NvidiaSmi {
    fn resolve() -> Result<Self, String> {
        let executable = if cfg!(windows) {
            let system_root =
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_owned());
            PathBuf::from(system_root)
                .join("System32")
                .join("nvidia-smi.exe")
        } else {
            PathBuf::from("/usr/bin/nvidia-smi")
        };
        if !executable.is_file() {
            return Err(format!(
                "trusted nvidia-smi path does not exist: {}",
                executable.display()
            ));
        }
        let physical_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|value| value.split(',').next().map(str::trim).map(str::to_owned))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        Ok(Self {
            executable,
            physical_id,
        })
    }

    fn query(&self, fields: &str) -> Result<String, String> {
        let output = Command::new(&self.executable)
            .arg(format!("--id={}", self.physical_id))
            .arg(format!("--query-gpu={fields}"))
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|error| format!("start {}: {error}", self.executable.display()))?;
        if !output.status.success() {
            return Err(format!(
                "{} query failed: {}",
                self.executable.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    fn used_bytes(&self) -> Result<u64, String> {
        self.query("memory.used")?
            .parse::<u64>()
            .map(|used_mib| used_mib * MIB)
            .map_err(|error| format!("parse nvidia-smi memory.used: {error}"))
    }
}

fn decimal_gb_to_bytes(value: f64) -> u64 {
    (value * 1.0e9).round() as u64
}

fn cuda_phase_metrics(device_bytes: u64) -> Value {
    // Candle's exact CUDA backend allocates directly through cudarc/CUDA and has no caching
    // allocator counter. On the required idle single-process GPU, the device delta is therefore the
    // only truthful active/allocator residency counter; diagnostics records this alias explicitly.
    // Discrete CUDA device allocations are physically non-pageable, so wired aliases device too.
    json!({
        "activeBytes": device_bytes,
        "allocatorBytes": device_bytes,
        "deviceBytes": device_bytes,
        "wiredBytes": device_bytes,
        "reclaimableBytes": 0,
    })
}

fn nvcc_runtime() -> Result<String, String> {
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

fn probe() -> Result<Value, String> {
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

fn sweep(request: &Value, parameters: &Map<String, Value>, result: &str) -> Result<Value, String> {
    let fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
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

fn artifact(repository: &str, revision: &str) -> Value {
    json!({
        "repository": repository,
        "resolvedRevision": revision,
        "variant": "q4",
    })
}

fn loadability_fingerprint(repository: &str, revision: &str) -> String {
    format!("{repository}@{revision}:q4")
}

#[allow(clippy::too_many_arguments)]
fn execute_lifecycle_request(
    generator: &dyn runtime_cuda::gen_core::Generator,
    context: &MemoryRunContext,
    edge: u32,
    overlap: u32,
    attention: u32,
    window: u32,
    fault_phase: Option<MemoryPhase>,
    cancel_phase: Option<MemoryPhase>,
) -> Result<(), String> {
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
    memory.calibration_error_phase = fault_phase;
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
        (None, None, Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)))
            if images.len() == 1 =>
        {
            scope
                .leave_phase(phase)
                .map_err(|error| format!("leave successful lifecycle phase: {error}"))?;
            scope
                .finish(MemoryRunOutcome::Complete)
                .map_err(|error| format!("finish successful lifecycle request: {error}"))
        }
        (Some(expected), None, Err(error))
            if error.to_string().contains("injected memory-strategy calibration error")
                && error.to_string().contains(&format!("{expected:?}")) =>
        {
            scope
                .finish(MemoryRunOutcome::Error {
                    message: error.to_string(),
                })
                .map_err(|finish| format!("finish injected-error lifecycle request: {finish}"))
        }
        (None, Some(_), Err(runtime_cuda::gen_core::Error::Canceled)) => scope
            .finish(MemoryRunOutcome::Canceled)
            .map_err(|error| format!("finish canceled lifecycle request: {error}")),
        (expected_fault, expected_cancel, actual) => Err(format!(
            "lifecycle outcome mismatch: fault={expected_fault:?}, cancel={expected_cancel:?}, actual={}",
            match actual {
                Ok(_) => "success".to_owned(),
                Err(error) => format!("error: {error}"),
            }
        )),
    }
}

fn execute_parity_request(
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

fn pixel_error(
    reference: &runtime_cuda::gen_core::Image,
    candidate: &runtime_cuda::gen_core::Image,
) -> Result<(u64, u64), String> {
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
    let mut maximum = 0_u64;
    let mut total = 0_u64;
    for (&left, &right) in reference.pixels.iter().zip(&candidate.pixels) {
        let error = u64::from(left.abs_diff(right));
        maximum = maximum.max(error);
        total += error;
    }
    let mean_micro_units =
        total.saturating_mul(1_000_000) / u64::try_from(reference.pixels.len()).unwrap_or(u64::MAX);
    Ok((maximum, mean_micro_units))
}

fn preflight_fragment(
    request: &Value,
    strategy: &Value,
    blocker: String,
    measurement_name: &'static str,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision),
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
    Ok(fragment)
}

fn strategy_name(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

fn measured_strategy(
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

fn run(request: &Value) -> Result<Value, String> {
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
    protocol::validate_plain_overlay_target(request, KREA_PLAIN_EXECUTION_PATH)?;
    let parameters = protocol::strategy_parameters(request)?;
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
            "q4",
            protocol::KREA_REPOSITORY,
        )?;
        canonical
    } else {
        root
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Sequential);
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
        decode_tile_edge: Some(edge),
        decode_overlap: Some(overlap),
        attention_chunk_size: Some(attention),
        transformer_window_size: Some(window),
        transformer_window_component: Some(TransformerComponent::Dit),
    };
    let selection = MemorySelection {
        strategy: MemoryStrategy::BoundedTransformerResidency,
        parameters: selected,
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
        },
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
            "supported provider tuple requires real weights; set SCENEWORKS_KREA_ROOT to the validated q4 snapshot".to_owned(),
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
        calibration_abi: actual_calibration.abi,
        calibration_fingerprint: actual_calibration.fingerprint.clone(),
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2 * GIB,
        },
        predicted_peak_bytes: total_bytes.saturating_sub(2 * GIB),
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-15508-adapter@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(KREA_ID, &spec)
        .map_err(|error| format!("load real {KREA_ID} q4 generator: {error}"))?;
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
            return Err(format!("real Krea q4 generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave terminal Krea phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish real Krea memory-strategy scope: {error}"))?;
    let image_count = match output {
        runtime_cuda::gen_core::GenerationOutput::Images(images) => images.len(),
        _ => 0,
    };
    if image_count != 1 {
        return Err(format!(
            "real Krea run returned {image_count} images, expected 1"
        ));
    }

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
    let overall_bytes = decimal_gb_to_bytes(report.peak_gb);
    let baseline = decimal_gb_to_bytes(report.baseline_gb);
    let lifecycle_phases = [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let smi = NvidiaSmi::resolve()?;
    let cleanup_tolerance_bytes = 64 * MIB;
    let mut maximum_cleanup_growth_bytes = 0_u64;
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            Some(lifecycle_phase),
        )?;
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} cancellation retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?;
    }
    for lifecycle_phase in lifecycle_phases {
        let before_fault_bytes = smi.used_bytes()?;
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            Some(lifecycle_phase),
            None,
        )?;
        let after_fault_bytes = smi.used_bytes()?;
        let cleanup_growth_bytes = after_fault_bytes.saturating_sub(before_fault_bytes);
        maximum_cleanup_growth_bytes = maximum_cleanup_growth_bytes.max(cleanup_growth_bytes);
        if cleanup_growth_bytes > cleanup_tolerance_bytes {
            return Err(format!(
                "{lifecycle_phase:?} injected error retained {cleanup_growth_bytes} device bytes above its pre-request baseline"
            ));
        }
        execute_lifecycle_request(
            generator.as_ref(),
            &context,
            edge,
            overlap,
            attention,
            window,
            None,
            None,
        )?;
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
    if resident_repeat_max_error != 0 {
        return Err(format!(
            "resident A-B-A repeat was not deterministic: maximum pixel error {resident_repeat_max_error}"
        ));
    }
    let (bounded_max_error, bounded_mean_error) = pixel_error(&resident_a, &bounded_b)?;
    let blocker = concat!(
        "real Krea phase telemetry executed, but complete evidence still requires predicted phase ",
        "curves, bounded-output tolerance approval, exact-fit/stale/unknown worker selection, and ",
        "a measured negative mutation"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        KREA_PLAIN_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            artifact: artifact(&repository, &revision),
            sweep: sweep(request, parameters, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision),
            }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter",
                "executed",
                [blocker.to_owned()],
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
                        "abaResidentRepeatMaximumPixelError",
                        "u8",
                        resident_repeat_max_error,
                    ),
                    (
                        "abaResidentRepeatMeanPixelError",
                        "pixel-micro-units",
                        resident_repeat_mean_error,
                    ),
                    ("abaBoundedMaximumPixelError", "u8", bounded_max_error),
                    (
                        "abaBoundedMeanPixelError",
                        "pixel-micro-units",
                        bounded_mean_error,
                    ),
                ],
            ),
        },
    )?;
    fragment["strategy"] = strategy;
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    if let Some(scenarios) = fragment["scenarios"].as_array_mut() {
        for scenario in scenarios {
            match scenario.get("name").and_then(Value::as_str) {
                Some("cancel") | Some("error") => {
                    let name = scenario["name"].clone();
                    *scenario = json!({
                        "name": name,
                        "result": "passed",
                        "cleanupVerified": true,
                        "warmFollowUpPassed": true,
                    });
                }
                Some("loadability") => {
                    *scenario = json!({ "name": "loadability", "result": "passed" });
                }
                _ => {}
            }
        }
    }
    Ok(fragment)
}

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}
