#[cfg(target_os = "macos")]
compile_error!("memory-candle-adapter is supported only on CUDA hosts");

use candle_gen::testkit::{StableIdleConfig, VramProbe};
use runtime_cuda::gen_core::{
    adapter_stack_identity, AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest,
    LoadShape, LoadSpec, MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode,
    MemoryNumericTier, MemoryOptimizationAuthority, MemoryPhase, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, OffloadPolicy, Precision, Progress, Quant, TransformerComponent,
    WeightsSource,
};
use sceneworks_memory_adapter as protocol;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::process::Command;

const KREA_ID: &str = "krea_2_turbo";
const KREA_PLAIN_EXECUTION_PATH: &str = "the Candle Krea base-only text-to-image path";
/// The label the Krea arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const KREA_STILL_CALIBRATION: &str = "Candle Krea base calibration";
const QWEN_ID: &str = "qwen_image";
const QWEN_PLAIN_EXECUTION_PATH: &str = "the Candle Qwen-Image base-only text-to-image path";
/// The label the Qwen arm refuses a non-still geometry under (sc-18808); see
/// [`still_calibration_label`].
const QWEN_STILL_CALIBRATION: &str = "Candle Qwen base calibration";
/// The Z-Image-Turbo provider (sc-15859). Registry id of `candle-gen-z-image`'s Turbo generator —
/// the catalog's `z_image_turbo` route, NOT the base `z_image` provider, which has its own contract
/// and its own plan cells.
const Z_IMAGE_TURBO_ID: &str = "z_image_turbo";
const Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH: &str =
    "the Candle Z-Image-Turbo base-only text-to-image path";
/// The label the Z-Image-Turbo arm refuses a non-still geometry under; see
/// [`still_calibration_label`].
const Z_IMAGE_TURBO_STILL_CALIBRATION: &str = "Candle Z-Image-Turbo base calibration";
const LTX25_ID: &str = "ltx_2_5_distilled";
const LTX25_EXECUTION_PATH: &str =
    "the Candle LTX-2.5 text-to-video base recipe (including the official dev refinement LoRA)";
const LTX25_DIFFUSION_VAE_COMPONENT: &str = "diffusion_video_vae";
const LTX25_DISTILL_LORA_RELATIVE_PATH: &str =
    "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
// The shipped q4/1024 golden approved mean_absolute_rgb_delta_255 <= 0.01681. Preserve that
// source-of-truth policy exactly. The historical contract did not
// constrain a single outlier channel, so the maximum metric remains diagnostic while the mean is
// the promotion gate. The required broad mutation must still breach at least one envelope bound.
const KREA_CANDLE_MAX_THRESHOLD: f64 = 1.0;
const KREA_CANDLE_MEAN_THRESHOLD: f64 = 0.01681;

fn certifying_wddm_idle_config() -> StableIdleConfig {
    // GPU 1's otherwise-idle WDDM graphics residency measured 1.6 GB in run 33188922159. The
    // pinned testkit's stable-idle proof is deliberately stricter than a raised one-shot ceiling:
    // it repeats the samples, bounds drift, and rejects any pure-compute process before allowing
    // the device-level delta capture.
    StableIdleConfig::new(2.0, 5, 64, 200)
}

fn certifying_vram_probe() -> VramProbe {
    VramProbe::start_rendered().assert_stable_idle(certifying_wddm_idle_config())
}

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

fn complete_sweep(request: &Value, parameters: &Map<String, Value>) -> Result<Value, String> {
    let mut result = sweep(request, parameters, "passed")?;
    // Every v1 record executes the only published tuple. Marking that singleton range verified
    // certifies exactly the selected parameters without claiming the unexecuted v2 candidates.
    result["rangeVerified"] = json!(true);
    Ok(result)
}

fn artifact(repository: &str, revision: &str, tier: &str) -> Value {
    json!({
        "repository": repository,
        "resolvedRevision": revision,
        "variant": tier,
    })
}

fn loadability_fingerprint(repository: &str, revision: &str, tier: &str) -> String {
    format!("{repository}@{revision}:{tier}")
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

fn negative_mutation(image: &runtime_cuda::gen_core::Image) -> runtime_cuda::gen_core::Image {
    let mut mutated = image.clone();
    for channel in &mut mutated.pixels {
        *channel = channel.wrapping_add(64);
    }
    mutated
}

fn ensure_krea_quality(maximum: f64, mean: f64, label: &str) -> Result<(), String> {
    if maximum > KREA_CANDLE_MAX_THRESHOLD || mean > KREA_CANDLE_MEAN_THRESHOLD {
        return Err(format!(
            "{label} exceeded the approved Candle Krea parity envelope: max={maximum:.6}, mean={mean:.6}"
        ));
    }
    Ok(())
}

fn preflight_fragment(
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

/// Persisted spelling of `gen_core::LoadShape` for the schema-v4 receipt field.
///
/// Callers pass the shape the run actually executed under — in practice
/// `contract.calibration.load_shape` from the LOADED provider, never the plan's declared value and
/// never a literal. A receipt may only testify to its own run (sc-16482).
fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => protocol::LOAD_SHAPE_EAGER,
        LoadShape::DeferredMaterialization => protocol::LOAD_SHAPE_DEFERRED,
    }
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

fn planned_memory_strategy(request: &Value) -> Result<MemoryStrategy, String> {
    match protocol::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        "bounded_attention" => Ok(MemoryStrategy::BoundedAttention),
        "bounded_transformer_residency" => Ok(MemoryStrategy::BoundedTransformerResidency),
        other => Err(format!("unsupported Candle fresh-reference rung {other:?}")),
    }
}

fn planned_provider(request: &Value) -> Result<&str, String> {
    protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())
}

fn plain_execution_path(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        "qwen_image" => Ok(QWEN_PLAIN_EXECUTION_PATH),
        "krea_2_turbo" => Ok(KREA_PLAIN_EXECUTION_PATH),
        "z_image_turbo" => Ok(Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH),
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
fn still_calibration_label(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        QWEN_ID => Ok(QWEN_STILL_CALIBRATION),
        KREA_ID => Ok(KREA_STILL_CALIBRATION),
        Z_IMAGE_TURBO_ID => Ok(Z_IMAGE_TURBO_STILL_CALIBRATION),
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
fn planned_tier(request: &Value) -> Result<&str, String> {
    match protocol::planned(request)?
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.tier must be a string".to_owned())?
    {
        tier @ ("bf16" | "q4" | "q8") => Ok(tier),
        tier => Err(format!("unsupported Candle numeric tier {tier:?}")),
    }
}

/// The fixture must name the tier and geometry it measured, so a bf16 record can never be emitted
/// against a q4 capture that merely reused the fixture string.
///
/// Scoped to `krea_2_turbo` DELIBERATELY. Krea is the only provider here whose plan spans several
/// (tier, geometry) legs through one adapter path — six of them, which is exactly how a mislabelled
/// capture would arise. The Qwen legs declare a single tier and geometry each and their fixture names
/// (`qwen-image-candle-q4-seed15817-step2`) predate this convention: applying the geometry token
/// requirement to them would reject five plan rows that measure correctly today. Widen this when
/// those fixtures are renamed, not before.
fn validate_fixture_binds_tier_and_geometry(request: &Value) -> Result<(), String> {
    if planned_provider(request)? != KREA_ID {
        return Ok(());
    }
    let planned = protocol::planned(request)?;
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
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

fn numeric_tier(tier: &str) -> Result<MemoryNumericTier, String> {
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

fn planned_selection(request: &Value) -> Result<MemorySelection, String> {
    let strategy = planned_memory_strategy(request)?;
    let transformer_window_size = protocol::optional_parameter(request, "transformerWindowSize")?;
    Ok(MemorySelection {
        strategy,
        parameters: MemoryStrategyParameters {
            decode_tile_edge: protocol::optional_parameter(request, "decodeTileEdge")?,
            decode_overlap: protocol::optional_parameter(request, "decodeOverlap")?,
            attention_chunk_size: protocol::optional_parameter(request, "attentionChunkSize")?,
            transformer_window_size,
            transformer_window_component: transformer_window_size
                .map(|_| TransformerComponent::Dit),
        },
        tier: numeric_tier(planned_tier(request)?)?,
    })
}

fn reference_phase(phase: MemoryPhase) -> protocol::ReferencePhase {
    match phase {
        MemoryPhase::Conditioning => protocol::ReferencePhase::Conditioning,
        MemoryPhase::Denoise => protocol::ReferencePhase::Denoise,
        MemoryPhase::Decode => protocol::ReferencePhase::Decode,
    }
}

fn memory_phase(phase: protocol::ReferencePhase) -> MemoryPhase {
    match phase {
        protocol::ReferencePhase::Conditioning => MemoryPhase::Conditioning,
        protocol::ReferencePhase::Denoise => MemoryPhase::Denoise,
        protocol::ReferencePhase::Decode => MemoryPhase::Decode,
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

/// Everything one five-rung capture needs after the artifact identity is validated and the real
/// generator is resident: `(provider id, plain execution path, repository, resolved revision,
/// generator, VRAM probe already holding the load sample)`.
type LoadedFiveRungGenerator = (
    &'static str,
    &'static str,
    String,
    String,
    Box<dyn runtime_cuda::gen_core::Generator>,
    VramProbe,
);

fn load_five_rung_generator(request: &Value) -> Result<LoadedFiveRungGenerator, String> {
    let (provider_id, execution_path, repository_env, revision_env, root_env, expected_repository) =
        match planned_provider(request)? {
            "qwen_image" => (
                QWEN_ID,
                QWEN_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_QWEN_IMAGE_REPOSITORY",
                "SCENEWORKS_QWEN_IMAGE_REVISION",
                "SCENEWORKS_QWEN_IMAGE_ROOT",
                protocol::QWEN_REPOSITORY,
            ),
            "krea_2_turbo" => (
                KREA_ID,
                KREA_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_KREA_REPOSITORY",
                "SCENEWORKS_KREA_REVISION",
                "SCENEWORKS_KREA_ROOT",
                protocol::KREA_REPOSITORY,
            ),
            // sc-15859. The artifact family is `SceneWorks/z-image-turbo-mlx` (`Z_IMAGE_REPOSITORY`),
            // the same per-tier `q4/ q8/ bf16/` re-host the MLX arm measures, so the env family is
            // `SCENEWORKS_Z_IMAGE_*` on both adapters (docs/calibration-runbook.md, "Adapter
            // environment").
            "z_image_turbo" => (
                Z_IMAGE_TURBO_ID,
                Z_IMAGE_TURBO_PLAIN_EXECUTION_PATH,
                "SCENEWORKS_Z_IMAGE_REPOSITORY",
                "SCENEWORKS_Z_IMAGE_REVISION",
                "SCENEWORKS_Z_IMAGE_ROOT",
                protocol::Z_IMAGE_REPOSITORY,
            ),
            provider => {
                return Err(format!(
                    "Candle five-rung calibration does not implement provider {provider:?}"
                ))
            }
        };
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    let repository = protocol::required_env(repository_env)?;
    let revision = protocol::required_env(revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(root_env)?))
        .map_err(|error| format!("canonicalize {root_env}: {error}"))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        expected_repository,
    )?;
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let spec = match (provider_id, numeric_tier(tier)?.quant) {
        // Krea's loader takes the packed tier's quant explicitly; bf16 is the dense base and must
        // carry no quant at all (`Quant::None` — the same shape the worker's `tier_to_quant` uses).
        (KREA_ID, Some(quant)) => spec.with_quant(quant),
        (KREA_ID, None) => spec,
        // Qwen and Z-Image-Turbo packed Diffusers snapshots declare their device-format
        // quantization in transformer/config.json (`snapshot_quant_tier` in candle-gen-z-image's
        // memory_strategy.rs). Passing LoadSpec.quant would request a second, unsupported runtime
        // quantization pass — both loaders reject it by name — instead of loading the packed
        // artifact as authored.
        _ => spec,
    };
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(provider_id, &spec)
        .map_err(|error| format!("load real {provider_id} {tier} generator: {error}"))?;
    vram.end_load(load_sample);
    Ok((
        provider_id,
        execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

fn run_five_rung_reference_loaded(
    request: &Value,
    provider_id: &str,
    execution_path: &str,
    generator: &dyn runtime_cuda::gen_core::Generator,
    vram: &mut VramProbe,
    repository: &str,
    revision: &str,
) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, execution_path)?;
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {provider_id} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {provider_id} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let planned_load_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    let actual_load_shape = match calibration.load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    };
    if planned_load_shape != actual_load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_load_shape}, pinned provider={actual_load_shape}"
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
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
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-16402@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {provider_id} fresh-reference scope: {error}"))?
        .ok_or_else(|| {
            format!("{provider_id} fresh-reference selection did not create a provider scope")
        })?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {provider_id} fresh-reference decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{provider_id} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope.configure_attention(attention).map_err(|error| {
            format!("configure {provider_id} fresh-reference attention: {error}")
        })?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| {
                format!("configure {provider_id} fresh-reference transformer: {error}")
            })?;
    }
    let mut generation = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(16402),
        // Two steps are intentional: resident Krea has no provider loading boundary between text
        // encode and denoise. The first Step callback closes a conservative conditioning envelope;
        // the second step then gives denoise its own measured interval before Decoding.
        steps: Some(2),
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {provider_id} fresh-reference strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {provider_id} fresh-reference conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
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
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {provider_id} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {provider_id} {next:?}: {error}"));
            return;
        }
        phase = next;
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
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    match result {
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) if images.len() == 1 => {}
        Ok(runtime_cuda::gen_core::GenerationOutput::Images(images)) => {
            return Err(format!(
                "{provider_id} fresh reference returned {} images",
                images.len()
            ));
        }
        Ok(_) => {
            return Err(format!(
                "{provider_id} fresh reference returned non-image output"
            ))
        }
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!(
                "{provider_id} fresh-reference generation failed: {message}"
            ));
        }
    }
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {provider_id} fresh-reference terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {provider_id} fresh-reference scope: {error}"))?;
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!("{provider_id} fresh reference did not expose conditioning boundary")
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{provider_id} fresh reference did not expose denoise boundary")
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{provider_id} fresh reference did not complete decode"))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);
    let blocker = if provider_id == QWEN_ID {
        concat!(
            "SC-15817 five-rung conformance measures exact per-rung memory, strategy identity, ",
            "and loadability; it intentionally remains gated because this run does not repeat ",
            "each sibling story's promotion-quality, negative-mutation, and lifecycle suite"
        )
    } else if provider_id == Z_IMAGE_TURBO_ID {
        concat!(
            "SC-15859 anchor capture measures exact per-phase memory and strategy identity for ",
            "the Candle Z-Image-Turbo lane; it intentionally remains gated because this run does ",
            "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    } else {
        concat!(
            "five-rung oracle capture measures exact per-rung memory and strategy identity for ",
            "SC-16059; it intentionally remains gated because this run does not repeat the full ",
            "promotion-quality, negative-mutation, and lifecycle scenario suite"
        )
    };
    let mut fragment = protocol::plain_gated_fragment(
        request,
        execution_path,
        protocol::PlainGatedFragment {
            artifact: artifact(repository, revision, planned_tier(request)?),
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": loadability_fingerprint(
                    repository,
                    revision,
                    planned_tier(request)?,
                ),
            }),
            diagnostics: protocol::diagnostics(
                &format!("memory-candle-adapter:{provider_id}-five-rung-reference"),
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
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

fn run_five_rung_reference(request: &Value) -> Result<Value, String> {
    let execution_path = plain_execution_path(request)?;
    protocol::validate_plain_overlay_target(request, execution_path)?;
    // Before `load_five_rung_generator`, for the same reason the overlay check is duplicated here:
    // a geometry this arm cannot honour must be refused before it costs a real weight load.
    protocol::validate_still_geometry(request, still_calibration_label(request)?)?;
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(request)?;
    run_five_rung_reference_loaded(
        request,
        provider_id,
        execution_path,
        generator.as_ref(),
        &mut vram,
        &repository,
        &revision,
    )
}

fn update_warmed_retention_baseline(
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

fn run_five_rung_batch(request: &Value) -> Result<Value, String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "run_batch request.planned must be an array".to_owned())?;
    let expected_rungs = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    let actual_rungs = planned
        .iter()
        .map(|item| {
            item.pointer("/strategy/rung")
                .and_then(Value::as_str)
                .ok_or_else(|| "batched planned strategy.rung must be a string".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual_rungs != expected_rungs {
        return Err(format!(
            "run_batch requires one canonical five-rung target, got {actual_rungs:?}"
        ));
    }
    let first_target = planned[0]
        .get("target")
        .ok_or_else(|| "batched planned target is missing".to_owned())?;
    if planned
        .iter()
        .any(|item| item.get("target") != Some(first_target))
    {
        return Err("run_batch cannot mix calibration targets in one model load".to_owned());
    }
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        let execution_path = plain_execution_path(&per_rung_request)?;
        protocol::validate_plain_overlay_target(&per_rung_request, execution_path)?;
        protocol::validate_still_geometry(
            &per_rung_request,
            still_calibration_label(&per_rung_request)?,
        )?;
    }

    let mut first_request = request.clone();
    first_request["action"] = json!("run");
    first_request["planned"] = planned[0].clone();
    let (provider_id, execution_path, repository, revision, generator, mut vram) =
        load_five_rung_generator(&first_request)?;
    let smi = NvidiaSmi::resolve()?;
    // Krea uses DeferredMaterialization, so loading the generator does not establish its
    // steady-state device residency. The canonical batch starts with `resident`; use the
    // memory retained after that first rung as the contamination baseline, then require every
    // later rung to release its transient allocations back to that warmed state.
    let mut settled_after_resident = None;
    let mut fragments = Vec::with_capacity(planned.len());
    for item in planned {
        let mut per_rung_request = request.clone();
        per_rung_request["action"] = json!("run");
        per_rung_request["planned"] = item.clone();
        fragments.push(run_five_rung_reference_loaded(
            &per_rung_request,
            provider_id,
            execution_path,
            generator.as_ref(),
            &mut vram,
            &repository,
            &revision,
        )?);
        let after = smi.used_bytes()?;
        update_warmed_retention_baseline(&mut settled_after_resident, after)?;
    }
    Ok(json!({ "modelLoads": 1, "fragments": fragments }))
}

fn ltx25_planned_load_shape(
    request: &Value,
    transformer_variant: protocol::Ltx25TransformerVariant,
) -> Result<LoadShape, String> {
    let declared = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?;
    transformer_variant.validate_load_shape(declared)?;
    match declared {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported LTX-2.5 Candle loadShape {other:?}")),
    }
}

fn ltx25_official_dev_adapter(revision: &str) -> Result<AdapterSpec, String> {
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_DISTILL_LORA_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_DISTILL_LORA_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &root,
        protocol::LTX_2_5_REPOSITORY,
        revision,
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let path = root.join(LTX25_DISTILL_LORA_RELATIVE_PATH);
    if !path.is_file() {
        return Err(format!(
            "LTX-2.5 dev capture requires the pinned official stage-two refinement LoRA at {}",
            path.display()
        ));
    }
    // The base scale MUST be the stage-one scale (0.0), not 1.0: production's
    // `resolve_ltx_distill_adapter` builds `AdapterSpec::new(path, stage1, ..)` with the manifest's
    // required `[0, 1]` contract, and the MLX capture arm does the same. gen_core's
    // `adapter_stack_identity` digests the scale bits into the admission overlay, so capturing at
    // 1.0 would mint evidence under an overlay identity no production request can ever present.
    Ok(AdapterSpec::new(path, 0.0, AdapterKind::Lora).with_pass_scales(vec![0.0, 1.0]))
}

struct Ltx25LoadPlan {
    repository: String,
    revision: String,
    inventory_sha256: String,
    load_shape: LoadShape,
    adapters: Vec<AdapterSpec>,
    spec: LoadSpec,
}

fn ltx25_load_spec(
    request: &Value,
    target: &protocol::Ltx25CandleTarget,
) -> Result<Ltx25LoadPlan, String> {
    let load_shape = ltx25_planned_load_shape(request, target.transformer_variant)?;
    let repository = protocol::required_env("SCENEWORKS_LTX25_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX25_REVISION")?;
    protocol::validate_ltx25_artifact_identity(&repository, &revision)?;
    let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
    protocol::validate_lowercase_sha256(
        &inventory_sha256,
        "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256",
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_subpath(
        &root,
        &repository,
        &revision,
        &[target.transformer_variant.as_str(), target.tier.as_str()],
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let adapters = if target
        .transformer_variant
        .requires_official_refinement_lora()
    {
        vec![ltx25_official_dev_adapter(&revision)?]
    } else {
        Vec::new()
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(load_shape);
    if let Some(quant) = numeric_tier(&target.tier)?.quant {
        spec = spec.with_quant(quant);
    }
    if target.decoder == protocol::Ltx25Decoder::DiffVae {
        spec = spec.with_component(
            LTX25_DIFFUSION_VAE_COMPONENT,
            WeightsSource::File(root.join("vae_diffusion_decoder.safetensors")),
        );
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters.clone());
    }
    Ok(Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    })
}

/// Execute a real selected LTX-2.5 provider path while leaving promotion decisions outside this
/// apparatus. The full published ladder — `q4`, `q8`, `bf16` — is executable here, symmetric with
/// the MLX arm. Candle's distinct NVFP4 evaluation selectors still need an inference-owned
/// producer and are never aliased to an ordinary bundle tier here.
fn run_ltx25_capture(request: &Value) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, LTX25_EXECUTION_PATH)?;
    let target = protocol::ltx25_candle_target(request)?;
    let Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    } = ltx25_load_spec(request, &target)?;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog.media().load(LTX25_ID, &spec).map_err(|error| {
        format!(
            "load real {LTX25_ID} {}/{}/{} generator: {error}",
            target.transformer_variant.as_str(),
            target.decoder.as_str(),
            target.tier
        )
    })?;
    vram.end_load(load_sample);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {LTX25_ID} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no calibration identity"))?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
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
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: target.width,
            height: target.height,
            batch: 1,
            frames: target.frames,
            reference_count: 0,
        },
        overlay: adapter_stack_identity(&adapters),
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-18783-adapter@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {LTX25_ID} capture scope: {error}"))?
        .ok_or_else(|| format!("{LTX25_ID} selection did not create a provider scope"))?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {LTX25_ID} decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{LTX25_ID} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure {LTX25_ID} attention: {error}"))?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure {LTX25_ID} transformer window: {error}"))?;
    }
    let mut generation = GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        width: target.width,
        height: target.height,
        count: 1,
        seed: Some(target.seed),
        steps: Some(target.transformer_variant.steps()),
        // The dev provider owns its fixed multimodal guider parameters; neither packed variant
        // advertises the generic request-level guidance axis.
        guidance: None,
        frames: Some(target.frames),
        fps: Some(target.fps),
        // The production default A/V T2V path leaves this unset. `Some("no_audio")` is the only
        // LTX T2V override; the evidence mode belongs in `MemoryRunContext`, not this provider knob.
        video_mode: None,
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {LTX25_ID} capture strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {LTX25_ID} conditioning: {error}"))?;
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
            phase_error = Some(format!("leave {LTX25_ID} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {LTX25_ID} {next:?}: {error}"));
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
            return Err(format!("{LTX25_ID} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {LTX25_ID} terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {LTX25_ID} capture scope: {error}"))?;
    let (frames, fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) => {
            return Err(format!("{LTX25_ID} returned images, not a video clip"))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!("{LTX25_ID} returned audio without video frames"))
        }
    };
    if fps != target.fps || audio.is_none() {
        return Err(format!(
            "{LTX25_ID} returned {fps} fps with audio={}, expected {} fps with audio",
            audio.is_some(),
            target.fps
        ));
    }
    let frame_shapes = frames
        .iter()
        .map(|frame| (frame.width, frame.height, frame.pixels.len()))
        .collect::<Vec<_>>();
    protocol::validate_ltx25_rgb_frames(
        usize::try_from(target.frames)
            .map_err(|_| "LTX-2.5 frame count does not fit usize".to_owned())?,
        target.width,
        target.height,
        &frame_shapes,
    )?;
    let conditioning_bytes = decimal_gb_to_bytes(
        peaks[0].ok_or_else(|| format!("{LTX25_ID} did not expose the conditioning boundary"))?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{LTX25_ID} did not expose the denoise boundary"))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{LTX25_ID} did not complete decode sampling"))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "SC-18783 LTX-2.5 Candle capture measured the selected real provider path; promotion remains ",
        "gated on terminal CUDA repetition/quality evidence; advanced INT8-ConvRot/NVFP4 receipt production remains inference-owned"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        LTX25_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            // Artifact variant retains the manifest/download identity. Transformer and decoder are
            // independent target axes and are stamped into the final record from `planned.target`.
            artifact: {
                let mut artifact = artifact(&repository, &revision, &target.tier);
                artifact["inventorySha256"] = json!(inventory_sha256);
                artifact
            },
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{}:transformer={}:decoder={}:f{}:{}x{}:fps{}:seed{}",
                    loadability_fingerprint(&repository, &revision, &target.tier),
                    target.transformer_variant.as_str(),
                    target.decoder.as_str(),
                    target.frames,
                    target.width,
                    target.height,
                    target.fps,
                    target.seed,
                ),
            }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter:ltx-2.5",
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(target.frames)),
                    ("renderedFps", "fps", u64::from(target.fps)),
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

/// The fixture prefix that marks a plan row as a five-rung reference capture.
const FIVE_RUNG_FIXTURE_PREFIX: &str = "fresh-five-rung-";

/// Which of [`run`]'s two branches a plan row takes: the five-rung reference path, or the inline
/// Krea arm.
///
/// Named rather than inlined so the decision is testable on its own (sc-18808 re-review). It is what
/// determines which arm [`run`]'s geometry guard is standing in front of, and every case in the
/// original regression table happened to answer `true` — so the inline arm, and with it `run`'s own
/// guard, went unexercised while the redundant copy at the head of [`run_five_rung_reference`]
/// produced the byte-identical message. Five shipped Candle plan rows answer `false`
/// (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings).
fn routes_to_five_rung_reference(request: &Value) -> Result<bool, String> {
    let is_five_rung_fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .is_some_and(|fixture| fixture.starts_with(FIVE_RUNG_FIXTURE_PREFIX));
    // Qwen and Z-Image-Turbo have no inline arm at all, so every fixture on them is a five-rung
    // reference capture regardless of its spelling.
    let provider = planned_provider(request)?;
    Ok(is_five_rung_fixture || provider == QWEN_ID || provider == Z_IMAGE_TURBO_ID)
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
    let provider = planned_provider(request)?;
    if provider == LTX25_ID {
        return run_ltx25_capture(request);
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

fn main() {
    let request = protocol::request_from_stdin().unwrap_or_else(|error| protocol::fail(error));
    let response = match protocol::action(&request).unwrap_or_else(|error| protocol::fail(error)) {
        "probe" => probe(),
        "run" => run(&request),
        "run_batch" => run_five_rung_batch(&request),
        other => Err(format!("unsupported action {other:?}")),
    }
    .unwrap_or_else(|error| protocol::fail(error));
    protocol::write_response(&response).unwrap_or_else(|error| protocol::fail(error));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_krea_wddm_idle_proof_keeps_the_measured_strict_bounds() {
        let config = certifying_wddm_idle_config();
        assert_eq!(config.max_baseline_gb, 2.0);
        assert_eq!(config.sample_count, 5);
        assert_eq!(config.max_drift_mib, 64);
        assert_eq!(config.sample_interval_ms, 200);
    }

    #[test]
    fn candle_krea_quality_uses_normalized_channel_error_and_the_approved_mean_bound() {
        let golden = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 255],
        };
        let within = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 250],
        };
        let (maximum, mean) = pixel_error(&golden, &within).unwrap();
        assert_eq!(maximum, 5.0 / 255.0);
        assert_eq!(mean, 5.0 / (3.0 * 255.0));
        ensure_krea_quality(maximum, mean, "golden policy").unwrap();

        let above_mean = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![6, 134, 249],
        };
        let (maximum, mean) = pixel_error(&golden, &above_mean).unwrap();
        assert!(maximum < KREA_CANDLE_MAX_THRESHOLD);
        assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
        assert!(ensure_krea_quality(maximum, mean, "regression").is_err());
        assert!(ensure_krea_quality(1.0, 0.01681, "exact boundary").is_ok());
        assert!(ensure_krea_quality(1.0, 0.01681 + 1e-12, "mean above boundary").is_err());
        assert!(ensure_krea_quality(1.0 + 1e-12, 0.01681, "maximum above boundary").is_err());
    }

    #[test]
    fn candle_krea_negative_mutation_breaches_the_certifying_mean_envelope() {
        let image = runtime_cuda::gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 128, 255],
        };
        let mutated = negative_mutation(&image);
        assert_eq!(mutated.pixels, vec![64, 192, 63]);
        let (maximum, mean) = pixel_error(&image, &mutated).unwrap();
        assert!(maximum > KREA_CANDLE_MEAN_THRESHOLD);
        assert!(mean > KREA_CANDLE_MEAN_THRESHOLD);
    }

    #[test]
    fn candle_krea_complete_sweep_certifies_only_the_v1_production_tuple() {
        let request = json!({
            "planned": {
                "calibrationFingerprint": "krea-turbo-cuda-phase-curves-v1"
            }
        });
        let parameters = Map::from_iter([
            ("decodeTileEdge".to_owned(), json!(512)),
            ("decodeOverlap".to_owned(), json!(128)),
            ("attentionChunkSize".to_owned(), json!(134_217_728)),
            ("transformerWindowSize".to_owned(), json!(1)),
        ]);
        let sweep = complete_sweep(&request, &parameters).unwrap();
        assert_eq!(sweep["rangeVerified"], json!(true));
        assert_eq!(sweep["cases"].as_array().unwrap().len(), 1);
        assert_eq!(sweep["cases"][0]["result"], json!("passed"));
        assert_eq!(sweep["cases"][0]["parameters"], json!(parameters));
    }

    fn qwen_request() -> Value {
        json!({
            "planned": {
                "target": { "provider": "qwen_image", "overlay": "none" },
                "strategy": { "rung": "resident", "parameters": {} },
                "loadShape": "deferred_materialization"
            }
        })
    }

    #[test]
    fn qwen_plan_routes_to_the_qwen_base_execution_path() {
        let request = qwen_request();
        assert_eq!(planned_provider(&request).unwrap(), "qwen_image");
        assert_eq!(
            plain_execution_path(&request).unwrap(),
            QWEN_PLAIN_EXECUTION_PATH
        );
        assert_eq!(
            planned_memory_strategy(&request).unwrap(),
            MemoryStrategy::Resident
        );
    }

    #[test]
    fn edit_plan_is_not_mislabeled_as_base_qwen_conformance() {
        let mut request = qwen_request();
        request["planned"]["target"]["provider"] = json!("qwen_image_edit");
        let error = plain_execution_path(&request).unwrap_err();
        assert!(error.contains("qwen_image_edit"));
        assert!(error.contains("does not implement"));
    }

    #[test]
    fn deferred_materialization_establishes_retention_baseline_after_resident_rung() {
        let mut baseline = None;
        update_warmed_retention_baseline(&mut baseline, 12 * GIB).unwrap();
        assert_eq!(baseline, Some(12 * GIB));
        update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB).unwrap();
    }

    #[test]
    fn warmed_retention_baseline_rejects_later_growth() {
        let mut baseline = Some(12 * GIB);
        let error =
            update_warmed_retention_baseline(&mut baseline, 12 * GIB + 64 * MIB + 1).unwrap_err();
        assert!(error.contains("above the warmed resident baseline"));
    }

    /// A single planned case at `frames`, minimal enough that the geometry guard is the FIRST thing
    /// that can reject it — no weight root, no environment.
    ///
    /// `fixture` is LOAD-BEARING here, not decoration. [`run`] picks its dispatch branch from it: a
    /// `fresh-five-rung-` prefix routes into [`run_five_rung_reference`], and ANY OTHER fixture on
    /// `krea_2_turbo` falls through to the inline Krea arm instead. A table that only ever passes
    /// one prefix therefore exercises only one of the two branches, which is precisely how this
    /// test shipped blind to [`run`]'s own guard once (sc-18808 re-review).
    fn still_planned_case_with_fixture(
        provider: &str,
        rung: &str,
        frames: u64,
        fixture: &str,
    ) -> Value {
        json!({
            "backend": "candle",
            "target": {
                "provider": provider,
                "modelId": provider,
                "tier": "q4",
                "mode": "text_to_image",
                "overlay": "none",
                "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": frames }
            },
            "loadShape": "deferred_materialization",
            "strategy": { "rung": rung, "parameters": {} },
            "calibrationFingerprint": "unused",
            "fixture": fixture
        })
    }

    /// The five-rung shape — the only one [`run_five_rung_batch`] accepts.
    fn still_planned_case(provider: &str, rung: &str, frames: u64) -> Value {
        still_planned_case_with_fixture(provider, rung, frames, "fresh-five-rung-unused")
    }

    /// The canonical five-rung batch shape `run_five_rung_batch` requires, at `frames`.
    fn still_batch_request(provider: &str, frames: u64) -> Value {
        let planned: Vec<Value> = [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ]
        .into_iter()
        .map(|rung| still_planned_case(provider, rung, frames))
        .collect();
        json!({ "action": "run_batch", "planned": planned })
    }

    /// sc-18808 — the Candle twin of the MLX adapter's
    /// `every_image_arm_still_refuses_a_multi_frame_geometry`.
    ///
    /// BOTH Candle arms hardcoded `frames: 1` into `MemoryGeometry` while reading only
    /// `width`/`height` from the plan, so a plan row declaring any other frame count would have
    /// rendered ONE frame and emitted a record claiming a geometry it was never asked for.
    ///
    /// Two entry points are reachable from `main`, and both must refuse with the exact pinned
    /// wording before any environment or weight work:
    ///
    /// * `run` — the dispatcher. Its guard stands in front of BOTH of its branches, and the branch
    ///   is chosen by `planned.fixture`, which is why the table below carries the fixture instead of
    ///   hardcoding one. A `fresh-five-rung-` prefix (or the `qwen_image` provider) routes into
    ///   `run_five_rung_reference`; ANY OTHER fixture on `krea_2_turbo` falls through to the INLINE
    ///   Krea arm — the arm that resolves `SCENEWORKS_KREA_REPOSITORY` and then writes its own
    ///   `MemoryGeometry { frames: 1 }`. Five SHIPPED Candle plan rows carry the second shape
    ///   (`krea-q4-1024-seed42` and its q8/bf16/768/v2 siblings), so it is the live one; the third
    ///   row below is one of them verbatim. Until it was added, every case in this table began
    ///   `fresh-five-rung-`, so all of them short-circuited into `run_five_rung_reference` and
    ///   `run`'s own guard was shadowed by the redundant copy at the head of that function —
    ///   deleting `run`'s guard left this test green.
    /// * `run_five_rung_batch` — reached straight from `main`, so `run`'s guard never sees it. Its
    ///   per-item pre-load loop is the guard under test; the fixture is irrelevant to it, so the
    ///   canonical five-rung shape is the only one it is exercised with.
    ///
    /// The other two copies of the refusal are redundant defense-in-depth and are NOT what this
    /// test pins: the one at the head of `run_five_rung_reference` (whose only caller is `run`,
    /// which already refused) and the one in `run_five_rung_reference_loaded` (reachable only after
    /// a real generator load, so no unit test can enter it). Removing either of those alone leaves
    /// this suite green — which is exactly why a future reader must not "clean up" `run`'s or
    /// `run_five_rung_batch`'s on the strength of them still being there.
    #[test]
    fn every_candle_arm_still_refuses_a_multi_frame_geometry() {
        for (provider, label, fixture) in [
            (QWEN_ID, QWEN_STILL_CALIBRATION, "fresh-five-rung-unused"),
            (KREA_ID, KREA_STILL_CALIBRATION, "fresh-five-rung-unused"),
            (
                Z_IMAGE_TURBO_ID,
                Z_IMAGE_TURBO_STILL_CALIBRATION,
                "fresh-five-rung-unused",
            ),
            // The inline Krea arm — a real shipped plan fixture, which the two rows above cannot
            // reach.
            (KREA_ID, KREA_STILL_CALIBRATION, "krea-q4-1024-seed42"),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                let request = json!({
                    "action": "run",
                    "planned": still_planned_case_with_fixture(
                        provider, "resident", frames, fixture,
                    )
                });
                assert_eq!(
                    run(&request).expect_err("the Candle dispatcher must refuse a video geometry"),
                    expected,
                    "run: {provider} at frames={frames} via fixture {fixture:?}"
                );
            }
        }
        for (provider, label) in [
            (QWEN_ID, QWEN_STILL_CALIBRATION),
            (KREA_ID, KREA_STILL_CALIBRATION),
            (Z_IMAGE_TURBO_ID, Z_IMAGE_TURBO_STILL_CALIBRATION),
        ] {
            for frames in [0_u64, 2, 97] {
                let expected = format!("{label} requires geometry.frames == 1, got {frames}");
                assert_eq!(
                    run_five_rung_batch(&still_batch_request(provider, frames))
                        .expect_err("the Candle batch arm must refuse a video geometry"),
                    expected,
                    "run_batch: {provider} at frames={frames}"
                );
            }
        }
    }

    /// The third row above is not decoration: the fixtures the SHIPPED Candle plan gives the inline
    /// Krea arm really do take that branch, so `run`'s own guard — not the redundant copy inside
    /// [`run_five_rung_reference`] — is the one standing in front of them.
    ///
    /// Asserted against the dispatch predicate itself rather than by observing an error, because
    /// both branches resolve the same `SCENEWORKS_KREA_REPOSITORY` and would report the same
    /// sentence: the error is not a routing witness, the predicate is. Widening the prefix (or
    /// renaming these fixtures into it) would re-shadow `run`'s guard, and this reds when it does.
    #[test]
    fn the_shipped_krea_fixtures_take_the_inline_arm_not_the_five_rung_branch() {
        for fixture in [
            "krea-q4-1024-seed42",
            "krea-q8-1024-seed42",
            "krea-bf16-1024-seed42",
            "krea-q4-768-seed42",
            "krea-q4-1024-seed42-v2-candidate",
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(KREA_ID, "resident", 1, fixture)
            });
            assert!(
                !routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the inline Krea arm"
            );
        }
        for (provider, fixture) in [
            (KREA_ID, "fresh-five-rung-krea-q4-1024-seed16402-step2"),
            (QWEN_ID, "qwen-image-candle-q4-seed15817-step2"),
            (
                Z_IMAGE_TURBO_ID,
                "fresh-five-rung-z-image-turbo-q4-1024-seed16402-step2",
            ),
            // No inline arm exists for Z-Image-Turbo, so an off-prefix fixture still routes here.
            (Z_IMAGE_TURBO_ID, "z-image-turbo-any-other-fixture"),
        ] {
            let request = json!({
                "planned": still_planned_case_with_fixture(provider, "resident", 1, fixture)
            });
            assert!(
                routes_to_five_rung_reference(&request).unwrap(),
                "{fixture} must reach the five-rung reference path"
            );
        }
    }

    /// And the guard is the frames axis rather than a blanket rejection: the same still geometry
    /// passes it on both Candle labels, so the refusals above cannot be an unconditional error.
    #[test]
    fn the_candle_still_geometry_guard_is_not_a_blanket_refusal() {
        for provider in [QWEN_ID, KREA_ID, Z_IMAGE_TURBO_ID] {
            let request = json!({ "planned": still_planned_case(provider, "resident", 1) });
            let label = still_calibration_label(&request).unwrap();
            protocol::validate_still_geometry(&request, label)
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
        }
    }
}
