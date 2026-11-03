//! Backend-neutral capture inputs and plan checks. Backend registries supply their own identity,
//! mode and tier policy; these helpers never infer a route from a name.
use super::*;
use gen_core::{AdapterKind, AdapterSpec, LoadShape, LoadSpec, MemoryStrategy};
use sha2::{Digest, Sha256};

/// Where the built-in Lightning distill snapshot lives, as the three
/// `SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_*` values name it — lifted out of the environment so the
/// attachment that makes the Lightning member Lightning is unit-testable.
#[derive(Clone, Debug)]
pub struct QwenEditLightningSource {
    pub repository: String,
    pub revision: String,
    pub root: PathBuf,
}

/// The three `SCENEWORKS_QWEN_IMAGE_EDIT_*` values naming the base snapshot one capture opens, plus
/// the distill snapshot the Lightning member stacks — lifted out of the environment so both the tier
/// binding and the adapter attachment are unit-testable.
#[derive(Clone, Debug)]
pub struct QwenEditArtifactSource {
    pub repository: String,
    pub revision: String,
    pub root: PathBuf,
    pub lightning: Option<QwenEditLightningSource>,
}

/// The artifact one Qwen edit capture loads: the canonical snapshot root, and the `LoadSpec` that
/// opens it.
#[derive(Debug)]
pub struct QwenEditArtifact {
    pub root: PathBuf,
    pub spec: LoadSpec,
}

/// The two artifact triples one Mage capture opens: the variant's own tier root, and the shared
/// text-encoder/VAE components snapshot.
#[derive(Clone, Debug)]
pub struct MageArtifactSource {
    pub repository: String,
    pub revision: String,
    pub root: PathBuf,
    pub components_repository: String,
    pub components_revision: String,
    pub components_root: PathBuf,
}

pub fn strategy_name(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

/// Persisted spelling of the materialization shape a run actually executed under. Derived from the
/// same `LoadShape` handed to the `LoadSpec` (or, better, from the LOADED provider's calibration
/// identity) — never from the plan, which only declares what it expects (sc-16482).
pub fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => crate::LOAD_SHAPE_EAGER,
        LoadShape::DeferredMaterialization => crate::LOAD_SHAPE_DEFERRED,
    }
}

/// The ORDERED artifact set one InstantID capture consumes, hashed exactly as the worker hashes it
/// (`crates/sceneworks-worker/src/image_jobs/instantid.rs` `instantid_artifact_entries` /
/// `instantid_artifact_fingerprint`): the paths in role order, record-separated, hex sha256. The
/// role tag is positional, so it is not part of the hashed stream. The digest lands verbatim inside
/// the provider's `overlay_key()`, which is the axis `validate_context` checks — two different
/// artifact sets can therefore never price each other.
pub fn instantid_artifact_fingerprint(paths: &[&Path]) -> String {
    let mut hasher = Sha256::new();
    for path in paths {
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(b"\x1e");
    }
    format!("{:x}", hasher.finalize())
}

/// A deterministic 512-d ArcFace-shaped embedding. `InstantId::generate_with` is the engine's own
/// "face-stack-independent path": the identity VALUE never changes what the denoise and decode
/// phases materialize, so a fixed embedding makes the capture reproducible without needing a real
/// face photograph on the capture host. The face stack itself is still attached and resident (the
/// worker attaches it too), so the composition the peaks describe is production's.
pub fn instantid_reference_embedding() -> Vec<f32> {
    (0..512)
        .map(|index| ((index as f32) * 0.017_f32).sin() * 0.05)
        .collect()
}

/// Five landmarks in the canonical `draw_kps` order (both eyes, nose, both mouth corners), scaled
/// from the normalized square the worker's angle presets use to output-canvas pixels.
pub fn instantid_kps(side: u32) -> Vec<(f32, f32)> {
    let side = side as f32;
    [
        (0.37_f32, 0.42_f32),
        (0.63, 0.42),
        (0.50, 0.55),
        (0.40, 0.68),
        (0.60, 0.68),
    ]
    .into_iter()
    .map(|(x, y)| (x * side, y * side))
    .collect()
}

fn qwen_edit_lightning_source() -> Result<QwenEditLightningSource, String> {
    Ok(QwenEditLightningSource {
        repository: crate::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REPOSITORY")?,
        revision: crate::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REVISION")?,
        root: PathBuf::from(crate::required_env(
            "SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT",
        )?),
    })
}

/// The built-in Lightning distill adapter, exactly as the worker stacks it: the one pinned file in
/// the pinned snapshot, as a single LoRA at scale 1.0 ahead of any user adapters (of which a
/// calibration capture has none). The Candle engine refuses any other path, scale or kind by name;
/// the MLX engine consumes whatever `spec.adapters` carries, so the pin is asserted HERE too rather
/// than assumed from the environment.
pub fn qwen_edit_lightning_adapter(
    source: &QwenEditLightningSource,
) -> Result<AdapterSpec, String> {
    crate::validate_artifact_identity(
        &source.repository,
        &source.revision,
        crate::QWEN_EDIT_LIGHTNING_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root).map_err(|error| {
        format!("canonicalize SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT: {error}")
    })?;
    let path = root.join(crate::QWEN_EDIT_LIGHTNING_FILE);
    if !path.is_file() {
        return Err(format!(
            "the Lightning distill LoRA is not at {}",
            path.display()
        ));
    }
    Ok(AdapterSpec::new(path, 1.0, AdapterKind::Lora))
}

pub fn qwen_edit_artifact_source(lightning: bool) -> Result<QwenEditArtifactSource, String> {
    Ok(QwenEditArtifactSource {
        repository: crate::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REPOSITORY")?,
        revision: crate::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REVISION")?,
        root: PathBuf::from(crate::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_ROOT")?),
        lightning: lightning.then(qwen_edit_lightning_source).transpose()?,
    })
}

pub fn planned_memory_strategy(request: &Value, lane: &str) -> Result<MemoryStrategy, String> {
    match crate::planned_rung(request)? {
        "resident" => Ok(MemoryStrategy::Resident),
        "staged_residency" => Ok(MemoryStrategy::StagedResidency),
        "bounded_decode" => Ok(MemoryStrategy::BoundedDecode),
        "bounded_attention" => Ok(MemoryStrategy::BoundedAttention),
        "bounded_transformer_residency" => Ok(MemoryStrategy::BoundedTransformerResidency),
        other => Err(format!("unsupported {lane} fresh-reference rung {other:?}")),
    }
}

pub fn validate_production_fingerprint(
    request: &Value,
    provider_id: &str,
    tier: &str,
    expected: Option<String>,
) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let planned = crate::planned_calibration_fingerprint(request)?;
    if planned != expected {
        return Err(format!(
            "planned.calibrationFingerprint {planned:?} is not the identity the pinned \
             {provider_id} contract publishes at tier {tier:?}; expected {expected:?}"
        ));
    }
    Ok(())
}

pub fn validate_plan_identity(
    request: &Value,
    provider: &str,
    tier: &str,
    expected_fingerprint: &str,
) -> Result<(), String> {
    let planned_fingerprint = crate::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!("plan/provider calibration mismatch: plan={planned_fingerprint}, the {provider} {tier} \
             production identity is {expected_fingerprint}"));
    }
    Ok(())
}

pub fn mage_mode(edit: bool) -> &'static str {
    if edit {
        "edit_image"
    } else {
        "text_to_image"
    }
}

pub fn instantid_fixture_seed(request: &Value, prefix: &str) -> Result<u64, String> {
    let fixture = crate::planned_fixture(request)?;
    let remainder = fixture
        .strip_prefix(prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    if steps != "2" {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    seed.parse::<u64>()
        .map_err(|error| format!("parse InstantID fixture seed {seed:?}: {error}"))
}

pub fn qwen_edit_fixture_seed<Steps>(
    request: &Value,
    prefix: &str,
    expected_steps: Steps,
) -> Result<u64, String>
where
    Steps: std::str::FromStr + std::fmt::Display + PartialEq,
    Steps::Err: std::fmt::Display,
{
    let fixture = crate::planned_fixture(request)?;
    let remainder = fixture
        .strip_prefix(prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Qwen edit fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<Steps>()
        .map_err(|error| format!("parse Qwen edit fixture step count {steps:?}: {error}"))?;
    if steps != expected_steps {
        return Err(format!("planned.fixture {fixture:?} must use this arm's {expected_steps}-step calibration request"));
    }
    Ok(seed)
}

fn reference_image(width: u32, height: u32) -> gen_core::Image {
    gen_core::Image {
        width,
        height,
        pixels: crate::synthetic_reference_rgb(width, height),
    }
}

pub fn iris_generation_request(
    width: u32,
    height: u32,
    seed: u64,
    steps: u32,
) -> gen_core::GenerationRequest {
    gen_core::GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    }
}

pub fn iris_depth_request(width: u32, height: u32) -> gen_core_iris::depth::DepthRequest {
    gen_core_iris::depth::DepthRequest {
        image: reference_image(width, height),
        resolution: gen_core_iris::depth::DepthResolution::Capped(width.max(height)),
        cancel: Default::default(),
    }
}

pub const IRIS_RESTORE_CAPTURE_SCALE: u32 = 4;

pub fn iris_restoration_request(width: u32, height: u32) -> gen_core::TransformRequest {
    let scale = IRIS_RESTORE_CAPTURE_SCALE;
    gen_core::TransformRequest {
        image: reference_image(width / scale, height / scale),
        target: gen_core::TargetSize::Scale(scale as f32),
        input_sizing: gen_core::InputSizing::Original,
        ..Default::default()
    }
}

pub enum IrisPlanned {
    Generation(Box<gen_core::GenerationRequest>),
    Depth(gen_core_iris::depth::DepthRequest),
    Restore(gen_core::TransformRequest),
}

pub fn supra2_load_spec(
    bundle: PathBuf,
    text_encoder: PathBuf,
    vae: PathBuf,
) -> gen_core::LoadSpec {
    use gen_core_checkpoint::supra2::{TEXT_ENCODER_COMPONENT, VAE_COMPONENT};
    gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(bundle))
        .with_offload_policy(gen_core::OffloadPolicy::Resident)
        .with_load_shape(gen_core::LoadShape::EagerMaterialization)
        .with_component(
            TEXT_ENCODER_COMPONENT,
            gen_core::WeightsSource::Dir(text_encoder),
        )
        .with_component(VAE_COMPONENT, gen_core::WeightsSource::Dir(vae))
}

pub fn supra2_capture_request(count: u32) -> gen_core::GenerationRequest {
    let size = gen_core_checkpoint::supra2::IMAGE_SIZE;
    gen_core::GenerationRequest {
        prompt: crate::SUPRA2_IMG_PROMPT.to_owned(),
        width: size,
        height: size,
        count,
        seed: Some(crate::SUPRA2_IMG_SEED),
        steps: Some(crate::SUPRA2_IMG_STEPS),
        ..Default::default()
    }
}

pub fn elapsed_millis(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The common clip axes; each lane still validates them against its actual engine envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

pub fn wan_target_geometry(request: &Value, label: &str) -> Result<Geometry, String> {
    let geometry = crate::planned_object(request, "/target/geometry")?;
    let axis = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    if axis("batch")? != 1 {
        return Err(format!(
            "{label} requires geometry.batch == 1 (these engines render one clip per request)"
        ));
    }
    Ok(Geometry {
        width: axis("width")?,
        height: axis("height")?,
        frames: axis("frames")?,
    })
}

pub fn validate_captured_mode(
    request: &Value,
    provider: &str,
    expected: &str,
) -> Result<(), String> {
    let mode = crate::planned_mode(request)?;
    if mode != expected {
        return Err(format!(
            "{provider} is captured under {expected:?}; the plan declares {mode:?}"
        ));
    }
    Ok(())
}

pub fn wan_reference_plane(width: u32, height: u32, salt: u32) -> gen_core::Image {
    let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        for x in 0..width {
            let base = x
                .wrapping_mul(7)
                .wrapping_add(y.wrapping_mul(13))
                .wrapping_add(salt);
            pixels.push((base % 251) as u8);
            pixels.push((base.wrapping_mul(3) % 241) as u8);
            pixels.push((base.wrapping_mul(5) % 239) as u8);
        }
    }
    gen_core::Image {
        width,
        height,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_parsers_keep_the_two_step_spelling_and_backend_integer_widths() {
        let plan = |fixture: &str| json!({"planned":{"fixture":fixture}});
        assert_eq!(
            instantid_fixture_seed(&plan("id-seed12-step2"), "id-seed").unwrap(),
            12
        );
        assert!(instantid_fixture_seed(&plan("id-seed12-step02"), "id-seed")
            .unwrap_err()
            .contains("two-step"));
        assert_eq!(
            qwen_edit_fixture_seed(&plan("edit-seed12-step04"), "edit-seed", 4_u32).unwrap(),
            12
        );
        let wide = plan("edit-seed12-step4294967296");
        assert!(qwen_edit_fixture_seed(&wide, "edit-seed", 4_u32)
            .unwrap_err()
            .contains("parse Qwen edit fixture step count"));
        #[cfg(target_pointer_width = "64")]
        assert!(qwen_edit_fixture_seed(&wide, "edit-seed", 4_usize)
            .unwrap_err()
            .contains("must use this arm's 4-step"));
    }
    #[test]
    fn identity_checks_preserve_unknown_routes_and_failure_wording() {
        assert!(validate_production_fingerprint(&json!({}), "custom", "q4", None).is_ok());
        assert_eq!(
            validate_production_fingerprint(&json!({}), "known", "q4", Some("expected".into()))
                .unwrap_err(),
            "run request is missing planned"
        );
        let plan = json!({"planned":{"calibrationFingerprint":"old"}});
        assert_eq!(validate_plan_identity(&plan,"mage_flow","bf16","new").unwrap_err(),"plan/provider calibration mismatch: plan=old, the mage_flow bf16 production identity is new");
        for rung in [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
        ] {
            let request = json!({"planned":{"strategy":{"rung":rung}}});
            assert_eq!(
                strategy_name(planned_memory_strategy(&request, "MLX").unwrap()),
                rung
            );
            assert_eq!(
                strategy_name(planned_memory_strategy(&request, "Candle").unwrap()),
                rung
            );
        }
        let request = json!({"planned":{"strategy":{"rung":"unknown"}}});
        assert_eq!(
            planned_memory_strategy(&request, "MLX").unwrap_err(),
            "unsupported MLX fresh-reference rung \"unknown\""
        );
        assert_eq!(
            planned_memory_strategy(&request, "Candle").unwrap_err(),
            "unsupported Candle fresh-reference rung \"unknown\""
        );
    }
}
