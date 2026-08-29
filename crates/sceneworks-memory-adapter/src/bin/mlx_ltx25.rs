//! Physical MLX LTX-2.5 memory capture arm (SC-18783).
//!
//! This is intentionally separate from the legacy LTX-2.3 apparatus. LTX-2.5 has two physical
//! transformer variants, two decoder implementations, a packed self-contained text encoder, and
//! provider-owned bounded-attention / block-streaming request controls. Every measurement below
//! goes through `mlx_gen_ltx::provider_registry`, the loaded generator's production memory
//! contract, and the shared request scope; there is no synthetic peak or alternate render path.

use super::*;
use mlx_gen::gen_core::AudioTrack;
use mlx_gen::{AdapterKind, AdapterSpec};

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
const DEV_ADAPTER: &str = "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
const AV_MAGIC: &[u8] = b"SCENEWORKS_AV1\0";
// PCM is already normalized floating-point data, so the established LTX full-pipeline absolute
// envelope applies without the video path's `/255` conversion. The mandatory 0.25-amplitude
// same-shape mutation is more than twenty times this maximum bound.
const AUDIO_MAX_THRESHOLD: f64 = LTX_MAX_THRESHOLD;
const AUDIO_MEAN_THRESHOLD: f64 = LTX_MEAN_THRESHOLD;
const AUDIO_RMS_THRESHOLD: f64 = LTX_RMS_THRESHOLD;

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
    root: PathBuf,
    snapshot_root: PathBuf,
    spec: LoadSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactInventory {
    bytes: u64,
    sha256: String,
}

struct SourceCapturePlan {
    output_dir: PathBuf,
    source_prefix: String,
    logical_case_id: String,
    model_inventory: ArtifactInventory,
    enhancer_inventory: ArtifactInventory,
    dev_adapter_inventory: Option<ArtifactInventory>,
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
        if actual != LTX25_PROVIDER {
            return Err(format!(
                "{LABEL} requires target.{name} {LTX25_PROVIDER:?}, got {actual:?}"
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
    // Refuse stale or merely SHA-shaped artifact revisions before canonicalization, provider
    // construction, and especially before any Metal materialization.
    protocol::validate_ltx25_artifact_identity(&repository, &revision)?;
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
        root,
        snapshot_root,
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

fn lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_logical_case_id(value: &str) -> bool {
    value.strip_prefix("implan-").is_some_and(|suffix| {
        suffix.len() == 20
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn inventory_from_environment<F>(
    required_env: &F,
    bytes_name: &str,
    sha256_name: &str,
) -> Result<ArtifactInventory, String>
where
    F: Fn(&str) -> Result<String, String>,
{
    let bytes = integer(&required_env(bytes_name)?, bytes_name)?;
    if bytes == 0 {
        return Err(format!("{bytes_name} must be greater than zero"));
    }
    let sha256 = required_env(sha256_name)?;
    if !lowercase_sha256(&sha256) {
        return Err(format!("{sha256_name} must be 64 lowercase hex characters"));
    }
    Ok(ArtifactInventory { bytes, sha256 })
}

fn source_inventories_from_environment<F>(
    target: Target,
    required_env: &F,
) -> Result<
    (
        ArtifactInventory,
        ArtifactInventory,
        Option<ArtifactInventory>,
    ),
    String,
>
where
    F: Fn(&str) -> Result<String, String>,
{
    let model = inventory_from_environment(
        required_env,
        "SCENEWORKS_MEMORY_MODEL_BYTES",
        "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256",
    )?;
    let enhancer = inventory_from_environment(
        required_env,
        "SCENEWORKS_LTX25_ENHANCER_BYTES",
        "SCENEWORKS_LTX25_ENHANCER_INVENTORY_SHA256",
    )?;
    let dev_adapter = (target.variant == TransformerVariant::Dev)
        .then(|| {
            inventory_from_environment(
                required_env,
                "SCENEWORKS_LTX25_DEV_ADAPTER_BYTES",
                "SCENEWORKS_LTX25_DEV_ADAPTER_SHA256",
            )
        })
        .transpose()?;
    Ok((model, enhancer, dev_adapter))
}

/// Resolve every raw-receipt prerequisite before constructing the provider. A missing raw-log
/// directory or inventory attestation must fail before an expensive Metal load, not after a clip
/// has already been rendered and become impossible to ingest.
fn prepare_source_capture(request: &Value, target: Target) -> Result<SourceCapturePlan, String> {
    let capture_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MEMORY_CAPTURE_DIR",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MEMORY_CAPTURE_DIR: {error}"))?;
    let source_prefix = protocol::required_env("SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX")?;
    let parts = source_prefix.split('/').collect::<Vec<_>>();
    if parts.len() < 3
        || parts[..2] != ["docs", "calibration"]
        || parts
            .iter()
            .any(|part| part.is_empty() || *part == "." || *part == ".." || part.contains('\\'))
    {
        return Err(
            "SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX must be a normalized path below docs/calibration"
                .to_owned(),
        );
    }
    let output_dir = parts
        .iter()
        .fold(capture_root.clone(), |directory, part| directory.join(part));
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| format!("create LTX-2.5 physical capture directory: {error}"))?;
    let output_dir = std::fs::canonicalize(output_dir)
        .map_err(|error| format!("canonicalize LTX-2.5 physical capture directory: {error}"))?;
    if !output_dir.starts_with(&capture_root) {
        return Err(
            "SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX escaped SCENEWORKS_MEMORY_CAPTURE_DIR".to_owned(),
        );
    }
    let logical_case_id = protocol::planned(request)?
        .get("logicalCaseId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.logicalCaseId must be a string".to_owned())?
        .to_owned();
    if !valid_logical_case_id(&logical_case_id) {
        return Err(
            "planned.logicalCaseId must be implan- plus 20 lowercase hex characters".to_owned(),
        );
    }
    let (model_inventory, enhancer_inventory, dev_adapter_inventory) =
        source_inventories_from_environment(target, &protocol::required_env)?;
    Ok(SourceCapturePlan {
        output_dir,
        source_prefix,
        logical_case_id,
        model_inventory,
        enhancer_inventory,
        dev_adapter_inventory,
    })
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
        // Both packed variants reject generic request guidance. The dev checkpoint executes its
        // fixed video/audio CFG, STG, rescale, and modality scales inside the native four-branch
        // sampler; `GenerationRequest::guidance` is not that typed checkpoint contract.
        guidance: None,
        frames: Some(target.geometry.frames),
        fps: Some(target.geometry.fps),
        ..Default::default()
    }
}

struct RenderedClip {
    frames: Vec<Image>,
    fps: u32,
    audio: AudioTrack,
    phases: Option<[PhaseMemory; 3]>,
}

fn full_av_video(output: GenerationOutput) -> Result<(Vec<Image>, u32, AudioTrack), String> {
    match output {
        GenerationOutput::Video {
            frames,
            fps,
            audio: Some(audio),
        } => Ok((frames, fps, audio)),
        GenerationOutput::Video { audio: None, .. } => {
            Err(format!("{LABEL} full-A/V render returned no audio track"))
        }
        GenerationOutput::Images(_) => Err(format!("{LABEL} returned images, not a video clip")),
        GenerationOutput::Audio(_) => Err(format!(
            "{LABEL} returned a standalone audio track, not a video clip"
        )),
    }
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
    let (frames, fps, audio) = full_av_video(output)?;
    if frames.len() != target.geometry.frames as usize || fps != target.geometry.fps {
        return Err(format!(
            "{LABEL} returned {} frames at {fps} fps; expected {} at {} fps",
            frames.len(),
            target.geometry.frames,
            target.geometry.fps,
        ));
    }
    let expected_frame_bytes = usize::try_from(target.geometry.width)
        .ok()
        .and_then(|width| {
            usize::try_from(target.geometry.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| format!("{LABEL} frame byte count overflow"))?;
    if frames.iter().any(|frame| {
        frame.width != target.geometry.width
            || frame.height != target.geometry.height
            || frame.pixels.len() != expected_frame_bytes
    }) {
        return Err(format!(
            "{LABEL} returned an empty, truncated, or wrong-sized RGB video frame"
        ));
    }
    let first = &frames[0];
    if first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err(format!("{LABEL} returned a degenerate first video frame"));
    }
    if audio.samples.is_empty()
        || audio.sample_rate == 0
        || audio.channels == 0
        || audio.samples.len() % usize::from(audio.channels) != 0
        || audio.samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(format!(
            "{LABEL} full-A/V render returned malformed interleaved PCM"
        ));
    }
    if !audio.stems.is_empty() {
        return Err(format!(
            "{LABEL} unexpectedly returned source-separated stems outside the canonical capture format"
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

fn output_fps_diagnostic(target: Target) -> (&'static str, &'static str, u64) {
    ("outputFps", "count", u64::from(target.geometry.fps))
}

fn audio_max_mean_rms_abs(
    left: &AudioTrack,
    right: &AudioTrack,
) -> Result<(f64, f64, f64), String> {
    if left.sample_rate != right.sample_rate
        || left.channels != right.channels
        || left.samples.len() != right.samples.len()
        || left.samples.is_empty()
    {
        return Err(format!(
            "audio shape mismatch: {} samples at {} Hz/{} channels versus {} at {} Hz/{} channels",
            left.samples.len(),
            left.sample_rate,
            left.channels,
            right.samples.len(),
            right.sample_rate,
            right.channels,
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_squares = 0.0_f64;
    for (&left, &right) in left.samples.iter().zip(&right.samples) {
        if !left.is_finite() || !right.is_finite() {
            return Err("audio comparison received non-finite PCM".to_owned());
        }
        let difference = (f64::from(left) - f64::from(right)).abs();
        maximum = maximum.max(difference);
        sum += difference;
        sum_squares += difference * difference;
    }
    let count = left.samples.len() as f64;
    Ok((maximum, sum / count, (sum_squares / count).sqrt()))
}

fn audio_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= AUDIO_MAX_THRESHOLD && mean <= AUDIO_MEAN_THRESHOLD && rms <= AUDIO_RMS_THRESHOLD
}

fn mutate_audio_pcm(audio: &AudioTrack) -> AudioTrack {
    let mut mutated = audio.clone();
    for sample in &mut mutated.samples {
        *sample = if *sample >= 0.0 {
            *sample - 0.25
        } else {
            *sample + 0.25
        };
    }
    mutated
}

fn pcm_sha256(audio: &AudioTrack) -> String {
    let mut hasher = Sha256::new();
    for samples in audio.samples.chunks(16_384) {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
        for sample in samples {
            bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
        }
        hasher.update(&bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn emit_canonical_av(
    clip: &RenderedClip,
    mut emit: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<u64, String> {
    let first = clip
        .frames
        .first()
        .ok_or_else(|| "canonical A/V receipt requires at least one frame".to_owned())?;
    let frame_count = u32::try_from(clip.frames.len())
        .map_err(|_| "canonical A/V frame count must fit u32".to_owned())?;
    let sample_count = u64::try_from(clip.audio.samples.len())
        .map_err(|_| "canonical A/V sample count must fit u64".to_owned())?;
    let mut total = 0_u64;
    let mut write = |bytes: &[u8]| {
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| "canonical A/V receipt byte count overflow".to_owned())?;
        emit(bytes)
    };
    write(AV_MAGIC)?;
    for value in [
        first.width,
        first.height,
        frame_count,
        clip.fps,
        clip.audio.sample_rate,
    ] {
        write(&value.to_le_bytes())?;
    }
    write(&clip.audio.channels.to_le_bytes())?;
    write(&sample_count.to_le_bytes())?;
    for frame in &clip.frames {
        write(&frame.width.to_le_bytes())?;
        write(&frame.height.to_le_bytes())?;
        let pixel_count = u64::try_from(frame.pixels.len())
            .map_err(|_| "canonical A/V frame byte count must fit u64".to_owned())?;
        write(&pixel_count.to_le_bytes())?;
        write(&frame.pixels)?;
    }
    for samples in clip.audio.samples.chunks(16_384) {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
        for sample in samples {
            bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
        }
        write(&bytes)?;
    }
    Ok(total)
}

fn canonical_av_identity(clip: &RenderedClip) -> Result<(String, u64), String> {
    let mut hasher = Sha256::new();
    let bytes = emit_canonical_av(clip, |chunk| {
        hasher.update(chunk);
        Ok(())
    })?;
    Ok((format!("{:x}", hasher.finalize()), bytes))
}

fn file_identity(path: &Path) -> Result<(String, u64), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("open immutable A/V receipt {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read immutable A/V receipt {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| "immutable A/V receipt byte count overflow".to_owned())?;
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn persist_canonical_av(
    plan: &SourceCapturePlan,
    role: &str,
    clip: &RenderedClip,
) -> Result<Value, String> {
    if !matches!(role, "selected_av" | "reference_av") {
        return Err(format!("unsupported canonical A/V receipt role {role:?}"));
    }
    let (content_sha256, bytes) = canonical_av_identity(clip)?;
    let first = &clip.frames[0];
    let file_name = format!(
        "{}-{role}-{}x{}-f{}-{content_sha256}.avbin",
        plan.logical_case_id,
        first.width,
        first.height,
        clip.frames.len(),
    );
    let local_path = plan.output_dir.join(&file_name);
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&local_path)
    {
        Ok(mut file) => {
            let written = emit_canonical_av(clip, |chunk| {
                file.write_all(chunk)
                    .map_err(|error| format!("write physical MLX {role} output: {error}"))
            })?;
            if written != bytes {
                return Err(format!(
                    "physical MLX {role} output wrote {written} bytes after hashing {bytes}"
                ));
            }
            file.sync_all()
                .map_err(|error| format!("sync physical MLX {role} output: {error}"))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let (existing_sha256, existing_bytes) = file_identity(&local_path)?;
            if existing_sha256 != content_sha256 || existing_bytes != bytes {
                return Err(format!(
                    "content-addressed physical MLX {role} output already exists with different bytes"
                ));
            }
        }
        Err(error) => {
            return Err(format!(
                "create immutable physical MLX {role} output: {error}"
            ));
        }
    }
    Ok(json!({
        "role": role,
        "path": format!("{}/{file_name}", plan.source_prefix),
        "localPath": local_path,
        "sha256": content_sha256,
        "bytes": bytes,
    }))
}

fn source_capture(
    plan: &SourceCapturePlan,
    artifact: &Artifact,
    tier: &str,
    selected: &RenderedClip,
    reference: &RenderedClip,
) -> Result<Value, String> {
    let mut inputs = vec![
        json!({
            "role": "base",
            "path": artifact.root,
            "bytes": plan.model_inventory.bytes,
            "sha256": plan.model_inventory.sha256,
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        }),
        json!({
            "role": "enhancer",
            "path": artifact.snapshot_root.join("enhancer"),
            "bytes": plan.enhancer_inventory.bytes,
            "sha256": plan.enhancer_inventory.sha256,
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": "enhancer",
        }),
    ];
    if let Some(inventory) = &plan.dev_adapter_inventory {
        inputs.push(json!({
            "role": "adapter",
            "path": artifact.snapshot_root.join(DEV_ADAPTER),
            "bytes": inventory.bytes,
            "sha256": inventory.sha256,
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": "dev_refinement_lora",
        }));
    }
    Ok(json!({
        "kind": "physical_mlx",
        "inputs": inputs,
        "outputs": [
            persist_canonical_av(plan, "selected_av", selected)?,
            persist_canonical_av(plan, "reference_av", reference)?,
        ],
        "claims": [
            "memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"
        ],
    }))
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
    let capture_plan = prepare_source_capture(request, target)?;
    let artifact = load_artifact(request, target, tier, &selection)?;
    let overlay = runtime_overlay(&artifact.spec, target.decoder)?;
    let registry = mlx_gen_ltx::provider_registry()
        .map_err(|error| format!("build LTX-2.5 registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX25_PROVIDER, &artifact.spec)
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
    let resolved_tier =
        mlx_gen_ltx::resolved_video_memory_numeric_tier(LTX25_PROVIDER, &artifact.spec)
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
    let generator = registry
        .load(LTX25_PROVIDER, &artifact.spec)
        .map_err(|error| {
            format!(
                "load real LTX-2.5 {}/{tier} provider: {error}",
                target.variant.id()
            )
        })?;
    if generator.descriptor().id != LTX25_PROVIDER {
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

    let selected = render(generator.as_ref(), target, &probe_context, true)?;
    let [conditioning, denoise, decode] = selected.phases.expect("measured render returns phases");
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
    let repeat = render(generator.as_ref(), target, &warm_context, false)?;
    if selected.fps != repeat.fps {
        return Err("LTX-2.5 identical-input repeat changed A/V identity".to_owned());
    }
    let (maximum_error, mean_error, rms_error) =
        video_max_mean_rms_abs(&selected.frames, &repeat.frames)?;
    if !quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "LTX-2.5 warm repeat exceeded determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let (audio_maximum_error, audio_mean_error, audio_rms_error) =
        audio_max_mean_rms_abs(&selected.audio, &repeat.audio)?;
    if !audio_quality_passes(audio_maximum_error, audio_mean_error, audio_rms_error) {
        return Err(format!(
            "LTX-2.5 warm repeat exceeded PCM determinism envelope: max={audio_maximum_error:.6}, mean={audio_mean_error:.6}, rms={audio_rms_error:.6}"
        ));
    }
    let selected_pcm_sha256 = pcm_sha256(&selected.audio);
    let reference_pcm_sha256 = pcm_sha256(&repeat.audio);
    let mutated = qwen_negative_mutation(&selected.frames[0]);
    let (mutated_maximum, mutated_mean, mutated_rms) =
        image_max_mean_rms_abs(&mutated, &repeat.frames[0])?;
    if quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err("LTX-2.5 output mutation did not breach determinism envelope".to_owned());
    }
    let mutated_audio = mutate_audio_pcm(&selected.audio);
    let (mutated_audio_maximum, mutated_audio_mean, mutated_audio_rms) =
        audio_max_mean_rms_abs(&mutated_audio, &repeat.audio)?;
    if audio_quality_passes(mutated_audio_maximum, mutated_audio_mean, mutated_audio_rms) {
        return Err(
            "LTX-2.5 same-shape PCM mutation did not breach determinism envelope".to_owned(),
        );
    }
    if mutated_audio.sample_rate != selected.audio.sample_rate
        || mutated_audio.channels != selected.audio.channels
        || mutated_audio.samples.len() != selected.audio.samples.len()
    {
        return Err("LTX-2.5 PCM mutation changed the audio shape".to_owned());
    }
    let sample_count = u64::try_from(selected.audio.samples.len())
        .map_err(|_| "LTX-2.5 PCM sample count must fit u64".to_owned())?;
    let source_capture = source_capture(&capture_plan, &artifact, tier, &selected, &repeat)?;

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
            "contract": "identical public artifact revision, transformer variant, decoder, prompt, seed, geometry, cadence, tier, required adapter recipe, and loaded provider; every video pixel and interleaved PCM sample in the measured render versus warm full-pipeline repeat",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": LTX_MAX_THRESHOLD,
            "meanErrorThreshold": LTX_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": LTX_RMS_THRESHOLD,
            "audio": {
                "result": "passed",
                "sampleRateHz": selected.audio.sample_rate,
                "channels": selected.audio.channels,
                "sampleCount": sample_count,
                "selectedPcmSha256": selected_pcm_sha256,
                "referencePcmSha256": reference_pcm_sha256,
                "maximumAbsoluteError": audio_maximum_error,
                "meanAbsoluteError": audio_mean_error,
                "rootMeanSquareError": audio_rms_error,
                "maximumAbsoluteErrorThreshold": AUDIO_MAX_THRESHOLD,
                "meanAbsoluteErrorThreshold": AUDIO_MEAN_THRESHOLD,
                "rootMeanSquareErrorThreshold": AUDIO_RMS_THRESHOLD,
            },
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
                output_fps_diagnostic(target),
                ("devVariant", "count", u64::from(target.variant == TransformerVariant::Dev)),
                ("diffusionDecoder", "count", u64::from(target.decoder == Decoder::DiffVae)),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("audioMutationMaximumAbsoluteErrorMicrounits", "count", (mutated_audio_maximum * 1_000_000.0).round() as u64),
                ("audioMutationMeanAbsoluteErrorMicrounits", "count", (mutated_audio_mean * 1_000_000.0).round() as u64),
                ("audioMutationRootMeanSquareErrorMicrounits", "count", (mutated_audio_rms * 1_000_000.0).round() as u64),
            ],
        ),
        "sourceCapture": source_capture,
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

    fn tiny_clip(samples: Vec<f32>) -> RenderedClip {
        RenderedClip {
            frames: vec![
                Image {
                    width: 2,
                    height: 1,
                    pixels: vec![1, 2, 3, 4, 5, 6],
                },
                Image {
                    width: 2,
                    height: 1,
                    pixels: vec![7, 8, 9, 10, 11, 12],
                },
            ],
            fps: 24,
            audio: AudioTrack {
                samples,
                sample_rate: 48_000,
                channels: 2,
                stems: Vec::new(),
            },
            phases: None,
        }
    }

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
                    "provider": LTX25_PROVIDER,
                    "modelId": LTX25_PROVIDER,
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
            (TransformerVariant::Dev, Some(DEV_STEPS), None),
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
            assert_eq!(
                output_fps_diagnostic(target),
                ("outputFps", "count", u64::from(BASE_FPS))
            );
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
        let revision = protocol::LTX_2_5_REVISION;
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
    fn artifact_revision_is_the_exact_public_upload_not_merely_sha_shaped() {
        protocol::validate_ltx25_artifact_identity(
            protocol::LTX25_REPOSITORY,
            protocol::LTX_2_5_REVISION,
        )
        .unwrap();
        let error = protocol::validate_ltx25_artifact_identity(
            protocol::LTX25_REPOSITORY,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .unwrap_err();
        assert!(error.contains(protocol::LTX_2_5_REVISION), "{error}");
    }

    #[test]
    fn pcm_parity_hashes_every_sample_and_same_shape_corruption_fails() {
        let selected = tiny_clip(vec![0.0, 0.25, -0.5, 0.75]);
        let reference = tiny_clip(vec![0.0, 0.25, -0.5, 0.75]);
        let (maximum, mean, rms) =
            audio_max_mean_rms_abs(&selected.audio, &reference.audio).unwrap();
        assert!(audio_quality_passes(maximum, mean, rms));
        assert_eq!(pcm_sha256(&selected.audio), pcm_sha256(&reference.audio));

        let mutated = mutate_audio_pcm(&selected.audio);
        assert_eq!(mutated.sample_rate, selected.audio.sample_rate);
        assert_eq!(mutated.channels, selected.audio.channels);
        assert_eq!(mutated.samples.len(), selected.audio.samples.len());
        let (maximum, mean, rms) = audio_max_mean_rms_abs(&mutated, &reference.audio).unwrap();
        assert!(!audio_quality_passes(maximum, mean, rms));
        assert_ne!(pcm_sha256(&mutated), pcm_sha256(&reference.audio));
    }

    #[test]
    fn canonical_av_receipt_binds_all_frames_metadata_and_pcm() {
        let clip = tiny_clip(vec![0.0, 0.25, -0.5, 0.75]);
        let mut bytes = Vec::new();
        let emitted = emit_canonical_av(&clip, |chunk| {
            bytes.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
        assert_eq!(emitted, bytes.len() as u64);
        assert_eq!(&bytes[..AV_MAGIC.len()], AV_MAGIC);
        assert_eq!(emitted, 105);
        let (digest, counted) = canonical_av_identity(&clip).unwrap();
        assert_eq!(counted, emitted);
        assert_eq!(digest, format!("{:x}", Sha256::digest(&bytes)));

        let mut pcm_mutation = tiny_clip(vec![0.0, 0.25, -0.5, 0.5]);
        assert_ne!(
            canonical_av_identity(&clip).unwrap().0,
            canonical_av_identity(&pcm_mutation).unwrap().0
        );
        pcm_mutation.audio.samples[3] = 0.75;
        pcm_mutation.frames[1].pixels[5] ^= 1;
        assert_ne!(
            canonical_av_identity(&clip).unwrap().0,
            canonical_av_identity(&pcm_mutation).unwrap().0
        );
    }

    #[test]
    fn canonical_av_receipts_are_content_addressed_and_immutable() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output_dir = std::env::temp_dir().join(format!(
            "sceneworks-ltx25-av-receipt-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&output_dir).unwrap();
        let plan = SourceCapturePlan {
            output_dir: output_dir.clone(),
            source_prefix: "docs/calibration/sc-18783".to_owned(),
            logical_case_id: "implan-0123456789abcdefabcd".to_owned(),
            model_inventory: ArtifactInventory {
                bytes: 1,
                sha256: "a".repeat(64),
            },
            enhancer_inventory: ArtifactInventory {
                bytes: 2,
                sha256: "b".repeat(64),
            },
            dev_adapter_inventory: None,
        };
        let clip = tiny_clip(vec![0.0, 0.25, -0.5, 0.75]);
        let receipt = persist_canonical_av(&plan, "selected_av", &clip).unwrap();
        assert_eq!(receipt["role"], "selected_av");
        assert!(receipt["path"].as_str().unwrap().ends_with(".avbin"));
        let local_path = PathBuf::from(receipt["localPath"].as_str().unwrap());
        let (digest, bytes) = file_identity(&local_path).unwrap();
        assert_eq!(receipt["sha256"], digest);
        assert_eq!(receipt["bytes"], bytes);
        assert_eq!(
            persist_canonical_av(&plan, "selected_av", &clip).unwrap(),
            receipt
        );

        std::fs::write(&local_path, b"tampered").unwrap();
        let error = persist_canonical_av(&plan, "selected_av", &clip).unwrap_err();
        assert!(error.contains("different bytes"), "{error}");
        std::fs::remove_dir_all(output_dir).unwrap();
    }

    #[test]
    fn source_capture_requires_variant_exact_shared_inventory_environment() {
        let distilled =
            validate_target(&request("q4", "distilled", "conv", 768, 512, BASE_FRAMES)).unwrap();
        let dev = validate_target(&request("q4", "dev", "conv", 768, 512, BASE_FRAMES)).unwrap();
        let mut values = std::collections::HashMap::from([
            ("SCENEWORKS_MEMORY_MODEL_BYTES", "1".to_owned()),
            ("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256", "a".repeat(64)),
        ]);
        let required = |name: &str| {
            values
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing required environment variable {name}"))
        };
        let error = source_inventories_from_environment(distilled, &required).unwrap_err();
        assert!(error.contains("SCENEWORKS_LTX25_ENHANCER_BYTES"), "{error}");

        values.insert("SCENEWORKS_LTX25_ENHANCER_BYTES", "2".to_owned());
        values.insert("SCENEWORKS_LTX25_ENHANCER_INVENTORY_SHA256", "b".repeat(64));
        let required = |name: &str| {
            values
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing required environment variable {name}"))
        };
        let (_, _, distilled_adapter) =
            source_inventories_from_environment(distilled, &required).unwrap();
        assert!(distilled_adapter.is_none());
        let error = source_inventories_from_environment(dev, &required).unwrap_err();
        assert!(
            error.contains("SCENEWORKS_LTX25_DEV_ADAPTER_BYTES"),
            "{error}"
        );

        values.insert("SCENEWORKS_LTX25_DEV_ADAPTER_BYTES", "3".to_owned());
        values.insert("SCENEWORKS_LTX25_DEV_ADAPTER_SHA256", "c".repeat(64));
        let required = |name: &str| {
            values
                .get(name)
                .cloned()
                .ok_or_else(|| format!("missing required environment variable {name}"))
        };
        let (_, _, dev_adapter) = source_inventories_from_environment(dev, &required).unwrap();
        assert_eq!(
            dev_adapter,
            Some(ArtifactInventory {
                bytes: 3,
                sha256: "c".repeat(64),
            })
        );
    }

    #[test]
    fn source_capture_seals_enhancer_and_variant_exact_adapter_inputs() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output_dir = std::env::temp_dir().join(format!(
            "sceneworks-ltx25-shared-receipt-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&output_dir).unwrap();
        let snapshot_root = output_dir.join("snapshot");
        let target = validate_target(&request("q4", "dev", "conv", 768, 512, BASE_FRAMES)).unwrap();
        let selection =
            planned_selection(&request("q4", "dev", "conv", 768, 512, BASE_FRAMES)).unwrap();
        let root = snapshot_root.join("dev/q4");
        let artifact = Artifact {
            repository: protocol::LTX25_REPOSITORY.to_owned(),
            revision: protocol::LTX_2_5_REVISION.to_owned(),
            root: root.clone(),
            snapshot_root: snapshot_root.clone(),
            spec: configured_spec(root, &snapshot_root, target, &selection),
        };
        let mut plan = SourceCapturePlan {
            output_dir: output_dir.clone(),
            source_prefix: "docs/calibration/sc-18783".to_owned(),
            logical_case_id: "implan-0123456789abcdefabcd".to_owned(),
            model_inventory: ArtifactInventory {
                bytes: 1,
                sha256: "a".repeat(64),
            },
            enhancer_inventory: ArtifactInventory {
                bytes: 2,
                sha256: "b".repeat(64),
            },
            dev_adapter_inventory: Some(ArtifactInventory {
                bytes: 3,
                sha256: "c".repeat(64),
            }),
        };
        let clip = tiny_clip(vec![0.0, 0.25, -0.5, 0.75]);
        let receipt = source_capture(&plan, &artifact, "q4", &clip, &clip).unwrap();
        let inputs = receipt["inputs"].as_array().unwrap();
        assert_eq!(
            inputs
                .iter()
                .map(|input| input["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["base", "enhancer", "adapter"]
        );
        assert_eq!(
            inputs[1]["path"].as_str(),
            snapshot_root.join("enhancer").to_str()
        );
        assert_eq!(inputs[1]["sha256"], "b".repeat(64));
        assert_eq!(
            inputs[2]["path"].as_str(),
            snapshot_root.join(DEV_ADAPTER).to_str()
        );
        assert_eq!(inputs[2]["sha256"], "c".repeat(64));
        plan.dev_adapter_inventory = None;
        let distilled_receipt = source_capture(&plan, &artifact, "q4", &clip, &clip).unwrap();
        assert_eq!(
            distilled_receipt["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|input| input["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["base", "enhancer"]
        );
        std::fs::remove_dir_all(output_dir).unwrap();
    }

    #[test]
    fn capture_receipt_tokens_are_fail_closed() {
        assert!(valid_logical_case_id("implan-0123456789abcdefabcd"));
        assert!(!valid_logical_case_id("implan-0123456789ABCDEFabcd"));
        assert!(!valid_logical_case_id("fixture-0123456789abcdefabcd"));
        assert!(lowercase_sha256(&"a".repeat(64)));
        assert!(!lowercase_sha256(&"A".repeat(64)));
        assert!(!lowercase_sha256(&"a".repeat(63)));
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
