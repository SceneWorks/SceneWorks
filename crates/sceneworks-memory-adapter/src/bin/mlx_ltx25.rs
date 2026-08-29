//! Physical MLX LTX-2.5 memory capture arm (SC-18783).
//!
//! This is intentionally separate from the legacy LTX-2.3 apparatus. LTX-2.5 has two physical
//! transformer variants, two decoder implementations, a packed self-contained text encoder, and
//! provider-owned bounded-attention / block-streaming request controls. Every measurement below
//! goes through `mlx_gen_ltx::provider_registry`, the loaded generator's production memory
//! contract, and the shared request scope; there is no synthetic peak or alternate render path.

use super::*;
use mlx_gen::{AdapterKind, AdapterSpec};

pub(super) const PROVIDER: &str = "ltx_2_5";
const LABEL: &str = "MLX LTX-2.5";
const EXECUTION_PATH: &str = "the MLX LTX-2.5 full-A/V text-to-video path";
const FINGERPRINT: &str = "sc-18797-ltx-2-5-mlx-ladder-v1";
const SEED: u64 = 18755;
const BASE_FRAMES: u32 = 145;
const BASE_FPS: u32 = 24;
const MAX_FRAMES: u32 = 449;
const MAX_FPS: u32 = 30;
const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
const TRANSFORMER_WINDOW_SIZE: u32 = 1;
const DECODE_TILE_EDGE: u32 = 192;
const DECODE_OVERLAP: u32 = 64;
const DEV_STEPS: u32 = 30;
const DEV_GUIDANCE: f32 = 3.0;
const DEV_ADAPTER: &str = "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransformerVariant {
    Distilled,
    Dev,
}

impl TransformerVariant {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "distilled" => Ok(Self::Distilled),
            "dev" => Ok(Self::Dev),
            other => Err(format!(
                "{LABEL} transformerVariant must be \"distilled\" or \"dev\", got {other:?}"
            )),
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::Distilled => "distilled",
            Self::Dev => "dev",
        }
    }

    const fn load_shape(self) -> LoadShape {
        match self {
            Self::Distilled => LoadShape::DeferredMaterialization,
            Self::Dev => LoadShape::EagerMaterialization,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Decoder {
    Conv,
    DiffVae,
}

impl Decoder {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "conv" => Ok(Self::Conv),
            "diffvae" => Ok(Self::DiffVae),
            other => Err(format!(
                "{LABEL} decoder must be \"conv\" or \"diffvae\", got {other:?}"
            )),
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::Conv => "conv",
            Self::DiffVae => "diffvae",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Target {
    variant: TransformerVariant,
    decoder: Decoder,
    geometry: Geometry,
}

struct Artifact {
    repository: String,
    revision: String,
    spec: LoadSpec,
}

fn target_string<'a>(
    target: &'a serde_json::Map<String, Value>,
    name: &str,
) -> Result<&'a str, String> {
    target
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("planned.target.{name} must be a string"))
}

fn validate_geometry(width: u32, height: u32, frames: u32) -> Result<Geometry, String> {
    const BASE: &[(u32, u32)] = &[(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)];
    const MAX: &[(u32, u32)] = &[(1280, 704), (704, 1280)];
    if width % 64 != 0 || height % 64 != 0 {
        return Err(format!(
            "{LABEL} requires width and height divisible by 64, got {width}x{height}"
        ));
    }
    let fps = match frames {
        BASE_FRAMES if BASE.contains(&(width, height)) => BASE_FPS,
        MAX_FRAMES if MAX.contains(&(width, height)) => MAX_FPS,
        _ => {
            return Err(format!(
                "{LABEL} geometry must be one of the five {BASE_FRAMES}-frame base buckets or the two {MAX_FRAMES}-frame maximum buckets, got {width}x{height}x{frames}"
            ));
        }
    };
    if frames % 8 != 1 {
        return Err(format!(
            "{LABEL} requires the 1+8k temporal lattice, got {frames} frames"
        ));
    }
    Ok(Geometry {
        width,
        height,
        frames,
        fps,
    })
}

fn validate_target(request: &Value) -> Result<Target, String> {
    let planned = protocol::planned(request)?;
    let target = planned
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    for name in ["provider", "modelId"] {
        let actual = target_string(target, name)?;
        if actual != PROVIDER {
            return Err(format!(
                "{LABEL} requires target.{name} {PROVIDER:?}, got {actual:?}"
            ));
        }
    }
    if target_string(target, "mode")? != "text_to_video" {
        return Err(format!("{LABEL} requires target.mode \"text_to_video\""));
    }
    if target_string(target, "overlay")? != "none" {
        return Err(format!(
            "{LABEL} calibration requires target.overlay \"none\"; the dev refinement adapter is part of transformerVariant=dev, not a user overlay"
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if target
            .get(field)
            .is_some_and(|value| value.as_u64() != Some(0))
        {
            return Err(format!(
                "{LABEL} requires target.{field} == 0 when declared"
            ));
        }
    }
    for field in ["hasReference", "has_reference"] {
        if target
            .get(field)
            .is_some_and(|value| value.as_bool() != Some(false))
        {
            return Err(format!(
                "{LABEL} requires target.{field} == false when declared"
            ));
        }
    }
    let variant = TransformerVariant::parse(target_string(target, "transformerVariant")?)?;
    let decoder = Decoder::parse(target_string(target, "decoder")?)?;
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
    if axis("batch")? != 1 {
        return Err(format!("{LABEL} requires geometry.batch == 1"));
    }
    let geometry = validate_geometry(axis("width")?, axis("height")?, axis("frames")?)?;
    let tier = planned_qwen_tier(request)?;
    let expected_fixture = format!(
        "ltx-2-5-mlx-{tier}-{}-{}-{}x{}-f{}-fps{}-seed{SEED}",
        variant.id(),
        decoder.id(),
        geometry.width,
        geometry.height,
        geometry.frames,
        geometry.fps,
    );
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    if fixture != expected_fixture {
        return Err(format!(
            "{LABEL} fixture must bind tier, transformerVariant, decoder, geometry, cadence, and seed exactly: expected {expected_fixture:?}, got {fixture:?}"
        ));
    }
    let fingerprint = planned
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if fingerprint != FINGERPRINT {
        return Err(format!(
            "{LABEL} requires calibration fingerprint {FINGERPRINT:?}, got {fingerprint:?}"
        ));
    }
    if planned.get("expectedResult").and_then(Value::as_str) != Some("passed")
        || planned.get("negative").and_then(Value::as_bool) != Some(false)
        || planned.get("modelLoadPolicy").and_then(Value::as_str) != Some("fresh_per_case")
        || !planned.get("modelLoadGroup").is_some_and(Value::is_null)
    {
        return Err(format!(
            "{LABEL} campaign rows must be positive, fresh-per-case captures with no load group"
        ));
    }
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != variant.load_shape() {
        return Err(format!(
            "{LABEL} {} requires load shape {:?}, got {:?}",
            variant.id(),
            variant.load_shape(),
            planned_shape,
        ));
    }
    Ok(Target {
        variant,
        decoder,
        geometry,
    })
}

fn validate_selection_shape(selection: &MemorySelection, target: Target) -> Result<(), String> {
    let parameters = selection.parameters;
    if parameters.attention_chunk_size != Some(ATTENTION_CHUNK_SIZE) {
        return Err(format!(
            "{LABEL} requires attentionChunkSize={ATTENTION_CHUNK_SIZE}"
        ));
    }
    match target.decoder {
        Decoder::Conv
            if parameters.decode_tile_edge == Some(DECODE_TILE_EDGE)
                && parameters.decode_overlap == Some(DECODE_OVERLAP) => {}
        Decoder::Conv => {
            return Err(format!(
                "{LABEL} conv requires decode tile {DECODE_TILE_EDGE}/{DECODE_OVERLAP}"
            ));
        }
        Decoder::DiffVae
            if parameters.decode_tile_edge.is_none() && parameters.decode_overlap.is_none() => {}
        Decoder::DiffVae => {
            return Err(format!(
                "{LABEL} diffvae must omit conv-only decode tile parameters"
            ));
        }
    }
    match target.variant {
        TransformerVariant::Distilled
            if selection.strategy == MemoryStrategy::BoundedTransformerResidency
                && parameters.transformer_window_size == Some(TRANSFORMER_WINDOW_SIZE)
                && parameters.transformer_window_component == Some(TransformerComponent::Dit) => {}
        TransformerVariant::Distilled => {
            return Err(format!(
                "{LABEL} distilled requires bounded_transformer_residency with DiT window {TRANSFORMER_WINDOW_SIZE}"
            ));
        }
        TransformerVariant::Dev
            if selection.strategy == MemoryStrategy::BoundedAttention
                && parameters.transformer_window_size.is_none()
                && parameters.transformer_window_component.is_none() => {}
        TransformerVariant::Dev => {
            return Err(format!(
                "{LABEL} dev requires bounded_attention and must not claim transformer streaming while its refinement adapter is installed"
            ));
        }
    }
    Ok(())
}

fn validate_nested_root(
    root: &Path,
    snapshot_root: &Path,
    repository: &str,
    revision: &str,
    variant: TransformerVariant,
    tier: &str,
) -> Result<(), String> {
    protocol::validate_huggingface_revision_root(
        snapshot_root,
        repository,
        revision,
        protocol::LTX25_REPOSITORY,
    )?;
    if root
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        != Some(variant.id())
        || root.file_name().and_then(|name| name.to_str()) != Some(tier)
    {
        return Err(format!(
            "{LABEL} artifact root must be <snapshot>/{}/{tier}",
            variant.id()
        ));
    }
    Ok(())
}

fn configured_spec(
    root: PathBuf,
    snapshot_root: &Path,
    target: Target,
    selection: &MemorySelection,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(target.variant.load_shape());
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    spec.components.insert(
        "enhancer".to_owned(),
        WeightsSource::Dir(snapshot_root.join("enhancer")),
    );
    if target.decoder == Decoder::DiffVae {
        spec.components.insert(
            "diffusion_video_vae".to_owned(),
            WeightsSource::File(root.join("vae_diffusion_decoder.safetensors")),
        );
    }
    if target.variant == TransformerVariant::Dev {
        spec.adapters.push(
            AdapterSpec::new(snapshot_root.join(DEV_ADAPTER), 0.0, AdapterKind::Lora)
                .with_pass_scales(vec![0.0, 1.0]),
        );
    }
    spec
}

fn load_artifact(
    request: &Value,
    target: Target,
    tier: &str,
    selection: &MemorySelection,
) -> Result<Artifact, String> {
    protocol::validate_plain_overlay_target(request, EXECUTION_PATH)?;
    let repository = protocol::required_env("SCENEWORKS_LTX25_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX25_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::LTX25_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_ROOT: {error}"))?;
    let snapshot_root = root
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "SCENEWORKS_LTX25_ROOT has no variant/snapshot parents".to_owned())?
        .to_path_buf();
    validate_nested_root(
        &root,
        &snapshot_root,
        &repository,
        &revision,
        target.variant,
        tier,
    )?;

    let enhancer = snapshot_root.join("enhancer");
    if !enhancer.is_dir() {
        return Err(format!(
            "{LABEL} stock enhancer co-requisite is missing at {}",
            enhancer.display()
        ));
    }
    if target.decoder == Decoder::DiffVae {
        let decoder = root.join("vae_diffusion_decoder.safetensors");
        if !decoder.is_file() {
            return Err(format!(
                "{LABEL} DiffVAE component is missing at {}",
                decoder.display()
            ));
        }
    }
    if target.variant == TransformerVariant::Dev {
        let adapter = snapshot_root.join(DEV_ADAPTER);
        if !adapter.is_file() {
            return Err(format!(
                "{LABEL} dev refinement adapter is missing at {}",
                adapter.display()
            ));
        }
    }
    let spec = configured_spec(root.clone(), &snapshot_root, target, selection);
    Ok(Artifact {
        repository,
        revision,
        spec,
    })
}

fn runtime_overlay(spec: &LoadSpec, decoder: Decoder) -> Result<Option<String>, String> {
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push(
            mlx_gen::gen_core::adapter_stack_identity(&spec.adapters).ok_or_else(|| {
                format!("{LABEL} dev adapter stack has no exact production identity")
            })?,
        );
    }
    if decoder == Decoder::DiffVae {
        axes.push("decoder:diffusion_vae".to_owned());
    }
    Ok((!axes.is_empty()).then(|| axes.join("+")))
}

fn context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    target: Target,
    overlay: Option<String>,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: target.geometry.width,
            height: target.geometry.height,
            batch: 1,
            frames: target.geometry.frames,
            reference_count: 0,
        },
        overlay,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-18783@{}", protocol::INFERENCE_PIN),
    }
}

fn generation_request(target: Target) -> GenerationRequest {
    let dev = target.variant == TransformerVariant::Dev;
    GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        // Production's dev request threads `VideoRequest::negative_prompt` as `Some`, including
        // its ordinary empty default. Distilled does not advertise negative conditioning at all.
        negative_prompt: dev.then(String::new),
        width: target.geometry.width,
        height: target.geometry.height,
        count: 1,
        seed: Some(SEED),
        steps: dev.then_some(DEV_STEPS),
        guidance: dev.then_some(DEV_GUIDANCE),
        frames: Some(target.geometry.frames),
        fps: Some(target.geometry.fps),
        ..Default::default()
    }
}

struct RenderedClip {
    frames: Vec<Image>,
    fps: u32,
    audio: Option<DiagnosticAudioIdentity>,
    phases: Option<[PhaseMemory; 3]>,
}

fn render(
    generator: &dyn Generator,
    target: Target,
    run_context: &MemoryRunContext,
    measure: bool,
) -> Result<RenderedClip, String> {
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    if measure {
        clear_cache();
        reset_peak_memory();
    }
    let output = scoped_generate(
        generator,
        generation_request(target),
        run_context,
        None,
        &mut |progress| {
            if !measure {
                return;
            }
            match progress {
                Progress::Step { current: 1, .. } if conditioning.get().active == 0 => {
                    conditioning.set(PhaseMemory::capture());
                    reset_peak_memory();
                }
                Progress::Decoding if denoise.get().active == 0 => {
                    denoise.set(PhaseMemory::capture());
                    reset_peak_memory();
                }
                _ => {}
            }
        },
    )?;
    let phases = measure.then(|| [conditioning.get(), denoise.get(), PhaseMemory::capture()]);
    let (frames, fps, audio) = diagnostic_video_frames(output, LABEL)?;
    if frames.len() != target.geometry.frames as usize || fps != target.geometry.fps {
        return Err(format!(
            "{LABEL} returned {} frames at {fps} fps; expected {} at {} fps",
            frames.len(),
            target.geometry.frames,
            target.geometry.fps,
        ));
    }
    if frames.iter().any(|frame| {
        frame.width != target.geometry.width
            || frame.height != target.geometry.height
            || frame.pixels.is_empty()
    }) {
        return Err(format!(
            "{LABEL} returned an empty or wrong-sized video frame"
        ));
    }
    let first = &frames[0];
    if first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err(format!("{LABEL} returned a degenerate first video frame"));
    }
    if !matches!(audio, Some(identity) if identity.samples > 0 && identity.sample_rate > 0 && identity.channels > 0)
    {
        return Err(format!(
            "{LABEL} full-A/V render returned no usable audio track"
        ));
    }
    Ok(RenderedClip {
        frames,
        fps,
        audio,
        phases,
    })
}

fn quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= LTX_MAX_THRESHOLD && mean <= LTX_MEAN_THRESHOLD && rms <= LTX_RMS_THRESHOLD
}

/// One physical full-pipeline tuple per plan row. Component names remain in the exact case but do
/// not become numeric sweep axes; this is the schema shape used by the existing rung-4 MLX arms.
fn complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let axes = parameters
        .iter()
        .filter_map(|(name, value)| value.as_u64().map(|value| (name, value)))
        .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
        .collect::<Vec<_>>();
    Ok(json!({
        "axes": axes,
        "cases": [{ "parameters": parameters, "result": "passed" }],
        "rangeVerified": true,
    }))
}

pub(super) fn run(request: &Value) -> Result<Value, String> {
    let target = validate_target(request)?;
    let tier = planned_qwen_tier(request)?;
    let selection = planned_selection(request)?;
    validate_selection_shape(&selection, target)?;
    let artifact = load_artifact(request, target, tier, &selection)?;
    let overlay = runtime_overlay(&artifact.spec, target.decoder)?;
    let registry = mlx_gen_ltx::provider_registry()
        .map_err(|error| format!("build LTX-2.5 registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(PROVIDER, &artifact.spec)
        .map_err(|error| format!("resolve real LTX-2.5 memory contract: {error}"))?
        .ok_or_else(|| "registered LTX-2.5 provider exposed no memory contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.5 contract rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition_for_selection(&selection),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.5 contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != FINGERPRINT
        || calibration.load_shape != target.variant.load_shape()
    {
        return Err(format!(
            "pinned LTX-2.5 calibration identity changed: fingerprint={}, loadShape={:?}",
            calibration.fingerprint, calibration.load_shape
        ));
    }
    let resolved_tier = mlx_gen_ltx::resolved_video_memory_numeric_tier(PROVIDER, &artifact.spec)
        .map_err(|error| format!("resolve physical LTX-2.5 numeric tier: {error}"))?
        .ok_or_else(|| "LTX-2.5 numeric tier resolver returned no tier".to_owned())?;
    if resolved_tier != selection.tier {
        return Err(format!(
            "planned LTX-2.5 tier {:?} differs from physical split bundle {:?}",
            selection.tier, resolved_tier
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let wired_limit_bytes = request
        .pointer("/hardware/wiredLimitBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.wiredLimitBytes must be an integer".to_owned())?;
    let generator = registry.load(PROVIDER, &artifact.spec).map_err(|error| {
        format!(
            "load real LTX-2.5 {}/{tier} provider: {error}",
            target.variant.id()
        )
    })?;
    if generator.descriptor().id != PROVIDER {
        return Err(format!(
            "loaded LTX-2.5 descriptor id changed to {:?}",
            generator.descriptor().id
        ));
    }
    let supports_negative = generator.descriptor().capabilities.supports_negative_prompt;
    if supports_negative != (target.variant == TransformerVariant::Dev) {
        return Err(format!(
            "loaded descriptor variant disagrees with planned transformerVariant {}",
            target.variant.id()
        ));
    }
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded LTX-2.5 generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err("loaded LTX-2.5 contract differs from registry contract".to_owned());
    }

    let probe_context = context(
        selection,
        calibration,
        &calibration.fingerprint,
        target,
        overlay.clone(),
        hardware_bytes,
        1,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&probe_context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.5 admission rejected a fitting probe budget".to_owned());
    }
    let unknown = context(
        selection,
        calibration,
        &calibration.fingerprint,
        target,
        overlay.clone(),
        0,
        1,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.5 admission accepted an unknown/zero budget".to_owned());
    }
    let stale = context(
        selection,
        calibration,
        "stale-ltx-2-5-fingerprint",
        target,
        overlay.clone(),
        hardware_bytes,
        1,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.5 admission accepted stale calibration evidence".to_owned());
    }

    let RenderedClip {
        frames: selected,
        fps: selected_fps,
        audio: selected_audio,
        phases,
    } = render(generator.as_ref(), target, &probe_context, true)?;
    let [conditioning, denoise, decode] = phases.expect("measured render returns phases");
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err("an LTX-2.5 full-pipeline phase reported a zero active peak".to_owned());
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    if overall.active > hardware_bytes || overall.active > wired_limit_bytes {
        return Err(format!(
            "LTX-2.5 observed overall active {} bytes above hardware {} or wired ceiling {}",
            overall.active, hardware_bytes, wired_limit_bytes
        ));
    }
    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    let exact_fit = context(
        selection,
        calibration,
        &calibration.fingerprint,
        target,
        overlay.clone(),
        predicted,
        predicted,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "LTX-2.5 admission rejected its exact-fit measured resident ceiling".to_owned(),
        );
    }

    let mut warm_context = probe_context.clone();
    warm_context.cache_state = MemoryCacheState::Warm;
    let RenderedClip {
        frames: repeat,
        fps: repeat_fps,
        audio: repeat_audio,
        ..
    } = render(generator.as_ref(), target, &warm_context, false)?;
    if selected_fps != repeat_fps || selected_audio != repeat_audio {
        return Err("LTX-2.5 identical-input repeat changed A/V identity".to_owned());
    }
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(&selected, &repeat)?;
    if !quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "LTX-2.5 warm repeat exceeded determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let mutated = qwen_negative_mutation(&selected[0]);
    let (mutated_maximum, mutated_mean, mutated_rms) =
        image_max_mean_rms_abs(&mutated, &repeat[0])?;
    if quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err("LTX-2.5 output mutation did not breach determinism envelope".to_owned());
    }

    let lifecycle_reason = concat!(
        "SC-18783 executes the measured full-pipeline render plus one identical-input warm parity ",
        "render per fresh process; cancellation and injected-error lifecycle cases remain explicitly ",
        "unexecuted here and are not used as calibration currency"
    );
    let fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "passed", "reason": "an identical-input full-pipeline request completed on the same loaded provider" },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_reason },
            { "name": "error", "result": "not_run", "reason": lifecycle_reason },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "target.overlay is none; transformerVariant and decoder are typed base-recipe axes, including dev's mandatory refinement adapter" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical public artifact revision, transformer variant, decoder, prompt, seed, geometry, cadence, tier, required adapter recipe, and loaded provider; measured render versus warm full-pipeline repeat",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": LTX_MAX_THRESHOLD,
            "meanErrorThreshold": LTX_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": LTX_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{}@{}:{tier}", artifact.repository, artifact.revision),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:ltx-2.5-full-pipeline",
            "executed",
            [lifecycle_reason.to_owned()],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("predictedOverallCeiling", "bytes", predicted),
                ("renderedFrames", "count", u64::from(target.geometry.frames)),
                ("renderedFps", "count", u64::from(target.geometry.fps)),
                ("devVariant", "count", u64::from(target.variant == TransformerVariant::Dev)),
                ("diffusionDecoder", "count", u64::from(target.decoder == Decoder::DiffVae)),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    // This path validates `target.overlay == none` before loading. Do not call the shared plain
    // helper here: its generic reason says there is no second resident network, which is false for
    // the dev transformer's required refinement adapter even though that adapter is not a user
    // overlay and is captured by the typed transformerVariant axis.
    let overlay_scenario = fragment["scenarios"].as_array().and_then(|scenarios| {
        scenarios
            .iter()
            .find(|scenario| scenario["name"] == "overlay")
    });
    if overlay_scenario.and_then(|scenario| scenario["result"].as_str()) != Some("not_applicable") {
        return Err("LTX-2.5 fragment lost its typed base-recipe overlay verdict".to_owned());
    }
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        tier: &str,
        variant: &str,
        decoder: &str,
        width: u32,
        height: u32,
        frames: u32,
    ) -> Value {
        let fps = if frames == BASE_FRAMES {
            BASE_FPS
        } else {
            MAX_FPS
        };
        let mut parameters = json!({ "attentionChunkSize": ATTENTION_CHUNK_SIZE });
        if decoder == "conv" {
            parameters["decodeTileEdge"] = json!(DECODE_TILE_EDGE);
            parameters["decodeOverlap"] = json!(DECODE_OVERLAP);
        }
        let (rung, load_shape) = if variant == "distilled" {
            parameters["transformerWindowSize"] = json!(TRANSFORMER_WINDOW_SIZE);
            parameters["transformerWindowComponent"] = json!("dit");
            (
                "bounded_transformer_residency",
                protocol::LOAD_SHAPE_DEFERRED,
            )
        } else {
            ("bounded_attention", protocol::LOAD_SHAPE_EAGER)
        };
        let engaged = match (variant, decoder) {
            ("distilled", "conv") => json!([
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention",
                "bounded_transformer_residency"
            ]),
            ("distilled", _) => json!([
                "resident",
                "staged_residency",
                "bounded_attention",
                "bounded_transformer_residency"
            ]),
            (_, "conv") => json!([
                "resident",
                "staged_residency",
                "bounded_decode",
                "bounded_attention"
            ]),
            _ => json!(["resident", "staged_residency", "bounded_attention"]),
        };
        json!({
            "planned": {
                "target": {
                    "provider": PROVIDER,
                    "modelId": PROVIDER,
                    "tier": tier,
                    "transformerVariant": variant,
                    "decoder": decoder,
                    "mode": "text_to_video",
                    "overlay": "none",
                    "geometry": { "width": width, "height": height, "batch": 1, "frames": frames }
                },
                "loadShape": load_shape,
                "strategy": { "rung": rung, "engagedRungs": engaged, "parameters": parameters },
                "calibrationFingerprint": FINGERPRINT,
                "fixture": format!("ltx-2-5-mlx-{tier}-{variant}-{decoder}-{width}x{height}-f{frames}-fps{fps}-seed{SEED}"),
                "expectedResult": "passed",
                "negative": false,
                "modelLoadPolicy": "fresh_per_case",
                "modelLoadGroup": null,
            }
        })
    }

    #[test]
    fn validates_every_campaign_identity_axis() {
        for tier in ["q4", "q8", "bf16"] {
            for variant in ["distilled", "dev"] {
                for decoder in ["conv", "diffvae"] {
                    let request = request(tier, variant, decoder, 768, 512, BASE_FRAMES);
                    let target = validate_target(&request).unwrap();
                    let selection = planned_selection(&request).unwrap();
                    validate_selection_shape(&selection, target).unwrap();
                }
            }
        }
    }

    #[test]
    fn maximum_rows_are_limited_to_the_two_large_orientations() {
        for (width, height) in [(1280, 704), (704, 1280)] {
            validate_target(&request(
                "q4",
                "distilled",
                "conv",
                width,
                height,
                MAX_FRAMES,
            ))
            .unwrap();
        }
        let error =
            validate_target(&request("q4", "distilled", "conv", 768, 512, MAX_FRAMES)).unwrap_err();
        assert!(error.contains("maximum buckets"), "{error}");
    }

    #[test]
    fn fixture_cannot_relabel_variant_decoder_or_cadence() {
        let mut value = request("q8", "dev", "diffvae", 1280, 704, MAX_FRAMES);
        value["planned"]["fixture"] =
            json!("ltx-2-5-mlx-q8-distilled-conv-1280x704-f449-fps24-seed18755");
        let error = validate_target(&value).unwrap_err();
        assert!(error.contains("fixture must bind"), "{error}");
    }

    #[test]
    fn diffvae_refuses_conv_only_parameters_and_dev_refuses_windowing() {
        let mut diffvae = request("q4", "distilled", "diffvae", 768, 512, BASE_FRAMES);
        diffvae["planned"]["strategy"]["parameters"]["decodeTileEdge"] = json!(192);
        let target = validate_target(&diffvae).unwrap();
        let selection = planned_selection(&diffvae).unwrap();
        assert!(validate_selection_shape(&selection, target)
            .unwrap_err()
            .contains("omit conv-only"));

        let mut dev = request("q4", "dev", "conv", 768, 512, BASE_FRAMES);
        dev["planned"]["strategy"]["parameters"]["transformerWindowSize"] = json!(1);
        dev["planned"]["strategy"]["parameters"]["transformerWindowComponent"] = json!("dit");
        let target = validate_target(&dev).unwrap();
        let selection = planned_selection(&dev).unwrap();
        assert!(validate_selection_shape(&selection, target)
            .unwrap_err()
            .contains("must not claim transformer streaming"));
    }

    #[test]
    fn requests_thread_variant_schedule_geometry_and_full_av_defaults() {
        for (variant, steps, guidance) in [
            (TransformerVariant::Distilled, None, None),
            (TransformerVariant::Dev, Some(DEV_STEPS), Some(DEV_GUIDANCE)),
        ] {
            let target = Target {
                variant,
                decoder: Decoder::Conv,
                geometry: validate_geometry(768, 512, BASE_FRAMES).unwrap(),
            };
            let request = generation_request(target);
            assert_eq!(request.steps, steps);
            assert_eq!(request.guidance, guidance);
            assert_eq!(
                request.negative_prompt,
                (variant == TransformerVariant::Dev).then(String::new)
            );
            assert_eq!(request.frames, Some(BASE_FRAMES));
            assert_eq!(request.fps, Some(BASE_FPS));
            assert_eq!(request.seed, Some(SEED));
            assert!(request.video_mode.is_none(), "default full-A/V route");
            assert!(request.conditioning.is_empty(), "reference-free T2V");
        }
    }

    #[test]
    fn load_specs_thread_tier_variant_decoder_and_dev_refinement_recipe() {
        let snapshot = PathBuf::from("/models/ltx-2.5-mlx/snapshot");
        for (variant, decoder) in [
            ("distilled", "conv"),
            ("distilled", "diffvae"),
            ("dev", "conv"),
            ("dev", "diffvae"),
        ] {
            let value = request("q4", variant, decoder, 768, 512, BASE_FRAMES);
            let target = validate_target(&value).unwrap();
            let selection = planned_selection(&value).unwrap();
            let root = snapshot.join(variant).join("q4");
            let spec = configured_spec(root.clone(), &snapshot, target, &selection);
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
            assert_eq!(spec.load_shape, target.variant.load_shape());
            assert_eq!(spec.quantize, Some(Quant::Q4));
            assert!(
                spec.text_encoder.is_none(),
                "packed encoder stays in bundle"
            );
            assert!(matches!(
                spec.components.get("enhancer"),
                Some(WeightsSource::Dir(path)) if path == &snapshot.join("enhancer")
            ));
            assert_eq!(
                spec.components.contains_key("diffusion_video_vae"),
                decoder == "diffvae"
            );
            assert_eq!(spec.adapters.len(), usize::from(variant == "dev"));
            if let Some(adapter) = spec.adapters.first() {
                assert_eq!(adapter.path, snapshot.join(DEV_ADAPTER));
                assert_eq!(adapter.scale, 0.0);
                assert_eq!(adapter.kind, AdapterKind::Lora);
                assert_eq!(adapter.pass_scales, Some(vec![0.0, 1.0]));
            }
        }
    }

    #[test]
    fn path_shape_binds_snapshot_variant_and_tier_without_weights() {
        let revision = "791ef61731ad067bd13ebff8cc0f07532476d9ef";
        let root = PathBuf::from(format!(
            "/cache/models--SceneWorks--ltx-2.5-mlx/snapshots/{revision}/distilled/q4"
        ));
        let snapshot = root.parent().unwrap().parent().unwrap();
        validate_nested_root(
            &root,
            snapshot,
            protocol::LTX25_REPOSITORY,
            revision,
            TransformerVariant::Distilled,
            "q4",
        )
        .unwrap();
        assert!(validate_nested_root(
            &root,
            snapshot,
            protocol::LTX25_REPOSITORY,
            revision,
            TransformerVariant::Dev,
            "q4",
        )
        .is_err());
    }

    #[test]
    fn sweep_preserves_component_identity_without_inventing_a_string_axis() {
        let value = request("q4", "distilled", "conv", 768, 512, BASE_FRAMES);
        let sweep = complete_sweep(&value).unwrap();
        assert_eq!(sweep["rangeVerified"], true);
        assert_eq!(
            sweep["cases"][0]["parameters"],
            value["planned"]["strategy"]["parameters"]
        );
        assert!(sweep["axes"].as_array().unwrap().iter().all(|axis| {
            axis["parameter"] != "transformerWindowComponent" && axis["testedValues"][0].is_u64()
        }));
    }
}
