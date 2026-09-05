//! Physical Candle capture arm for the Wan 2.2 family and SCAIL-2 (sc-22736, epic sc-22723 E1).
//!
//! The Candle sibling of `mlx_wan_scail2.rs`, and the same four engine providers on one arm. Two
//! things differ from the MLX lane, and both come from what the WORKER does rather than from
//! convenience:
//!
//! * **Residency.** `video_jobs/candle.rs::candle_video_offload_policy` names all three Wan routes
//!   `Sequential` — the A14B pair because its two 14B experts cannot be co-resident, the TI2V-5B
//!   because sc-13175 moved it there and its shipped `candle.vramGbByTier` peak is the SEQUENTIAL
//!   one. SCAIL-2 is not in that list, so it loads `Resident`, which is also the composition its
//!   `resident-eager` identity names.
//! * **The artifact is per (lane, TIER).** Each Wan route ships a `SceneWorks/…-candle` rehost with
//!   `q4` and `q8` ONLY; its dense leg is the upstream `Wan-AI/…-Diffusers` checkpoint, whose
//!   weights sit at the snapshot ROOT rather than under a `bf16/` subtree and whose revision the
//!   manifest does not pin. SCAIL-2 is the opposite: ONE `SceneWorks/scail2-mlx` repository, all
//!   three tiers, both lanes.
//!
//! Everything else is the same claim: resolve the artifact, load through `runtime_cuda::catalog()`
//! (the seam `inference_runtime.rs` wraps), read the LOADED generator's own contract and identity,
//! drive the provider's own registered admission check, and measure three phase peaks inside the
//! provider's own request scope.

use super::*;
use runtime_cuda::gen_core::wan_i2v_memory::WanI2vRoute;
use runtime_cuda::gen_core::{MemoryCalibrationIdentity, ReplacementMode};

const SEED: u64 = 22_736;
const LABEL: &str = "Candle Wan2.2/SCAIL-2";

/// The public request carrier one of these routes takes. Mirrors the MLX arm's enum, because the
/// carrier is a property of the ROUTE and not of the lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Carrier {
    None,
    Reference,
    Animation,
}

impl Carrier {
    const fn reference_count(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Reference | Self::Animation => 1,
        }
    }
}

/// How this cell's artifact is laid out on disk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// `…/snapshots/<revision>/<tier>` — a packed SceneWorks rehost.
    Tiered,
    /// `…/snapshots/<revision>` — the upstream dense Diffusers checkpoint, whose weights are at the
    /// snapshot root. Probing a `bf16/` subtree here would report a staged cell as missing.
    Flat,
}

#[derive(Clone, Copy)]
struct Arm {
    provider: &'static str,
    model_id: &'static str,
    route: Option<WanI2vRoute>,
    mode: &'static str,
    carrier: Carrier,
    fps: u32,
    steps: u32,
    offload_policy: OffloadPolicy,
    /// The packed-tier artifact family.
    packed: Family,
    /// The dense-tier artifact family, when the dense leg is a DIFFERENT artifact. `None` means
    /// every tier of this route lives in the packed family (SCAIL-2).
    dense: Option<Family>,
    slug: &'static str,
    execution_path: &'static str,
}

/// One `SCENEWORKS_<env>_{REPOSITORY,REVISION,ROOT}` family, spelled out.
///
/// The three names are SOURCE LITERALS rather than a `format!` over one stem, because that is the
/// binding `measure-memory-catalog.test.mjs` checks: the JS table derives the same three names and
/// asserts each adapter arm reads them back verbatim, so a rename on either side is a red test
/// instead of a mid-campaign failure after the weights are staged.
#[derive(Clone, Copy)]
struct Family {
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    repository: &'static str,
    layout: Layout,
}

const TI2V_5B: Arm = Arm {
    provider: WAN_TI2V_5B_ID,
    model_id: "wan_2_2",
    route: Some(WanI2vRoute::Ti2v5b),
    mode: "text_to_video",
    carrier: Carrier::None,
    fps: 24,
    steps: 20,
    offload_policy: OffloadPolicy::Sequential,
    packed: Family {
        repository_env: "SCENEWORKS_WAN22_TI2V_5B_CANDLE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_TI2V_5B_CANDLE_REVISION",
        root_env: "SCENEWORKS_WAN22_TI2V_5B_CANDLE_ROOT",
        repository: protocol::WAN22_TI2V_5B_CANDLE_REPOSITORY,
        layout: Layout::Tiered,
    },
    dense: Some(Family {
        repository_env: "SCENEWORKS_WAN22_TI2V_5B_DENSE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_TI2V_5B_DENSE_REVISION",
        root_env: "SCENEWORKS_WAN22_TI2V_5B_DENSE_ROOT",
        repository: protocol::WAN22_TI2V_5B_DENSE_REPOSITORY,
        layout: Layout::Flat,
    }),
    slug: "wan-2-2-ti2v-5b",
    execution_path: "the Candle Wan2.2 TI2V-5B base text-to-video path",
};

const T2V_A14B: Arm = Arm {
    provider: WAN_T2V_A14B_ID,
    model_id: "wan_2_2_t2v_14b",
    route: Some(WanI2vRoute::T2v14b),
    mode: "text_to_video",
    carrier: Carrier::None,
    fps: 16,
    steps: 40,
    offload_policy: OffloadPolicy::Sequential,
    packed: Family {
        repository_env: "SCENEWORKS_WAN22_T2V_A14B_CANDLE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_T2V_A14B_CANDLE_REVISION",
        root_env: "SCENEWORKS_WAN22_T2V_A14B_CANDLE_ROOT",
        repository: protocol::WAN22_T2V_A14B_CANDLE_REPOSITORY,
        layout: Layout::Tiered,
    },
    dense: Some(Family {
        repository_env: "SCENEWORKS_WAN22_T2V_A14B_DENSE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_T2V_A14B_DENSE_REVISION",
        root_env: "SCENEWORKS_WAN22_T2V_A14B_DENSE_ROOT",
        repository: protocol::WAN22_T2V_A14B_DENSE_REPOSITORY,
        layout: Layout::Flat,
    }),
    slug: "wan-2-2-t2v-a14b",
    execution_path: "the Candle Wan2.2 T2V-A14B dual-expert text-to-video path",
};

const I2V_A14B: Arm = Arm {
    provider: WAN_I2V_A14B_ID,
    model_id: "wan_2_2_i2v_14b",
    route: Some(WanI2vRoute::I2v14b),
    mode: "image_to_video",
    carrier: Carrier::Reference,
    fps: 16,
    steps: 40,
    offload_policy: OffloadPolicy::Sequential,
    packed: Family {
        repository_env: "SCENEWORKS_WAN22_I2V_A14B_CANDLE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_I2V_A14B_CANDLE_REVISION",
        root_env: "SCENEWORKS_WAN22_I2V_A14B_CANDLE_ROOT",
        repository: protocol::WAN22_I2V_A14B_CANDLE_REPOSITORY,
        layout: Layout::Tiered,
    },
    dense: Some(Family {
        repository_env: "SCENEWORKS_WAN22_I2V_A14B_DENSE_REPOSITORY",
        revision_env: "SCENEWORKS_WAN22_I2V_A14B_DENSE_REVISION",
        root_env: "SCENEWORKS_WAN22_I2V_A14B_DENSE_ROOT",
        repository: protocol::WAN22_I2V_A14B_DENSE_REPOSITORY,
        layout: Layout::Flat,
    }),
    slug: "wan-2-2-i2v-a14b",
    execution_path: "the Candle Wan2.2 I2V-A14B dual-expert image-to-video path",
};

const SCAIL2: Arm = Arm {
    provider: SCAIL2_ID,
    model_id: "scail2_14b",
    route: None,
    mode: "animation",
    carrier: Carrier::Animation,
    fps: 16,
    steps: 20,
    // Not in `candle_video_offload_policy`'s Sequential list, and `resident-eager` is the one
    // composition its calibration identity names.
    offload_policy: OffloadPolicy::Resident,
    packed: Family {
        repository_env: "SCENEWORKS_SCAIL2_REPOSITORY",
        revision_env: "SCENEWORKS_SCAIL2_REVISION",
        root_env: "SCENEWORKS_SCAIL2_ROOT",
        repository: protocol::SCAIL2_REPOSITORY,
        layout: Layout::Tiered,
    },
    dense: None,
    slug: "scail2-14b",
    execution_path: "the Candle SCAIL-2 character-animation path",
};

const ARMS: [Arm; 4] = [TI2V_5B, T2V_A14B, I2V_A14B, SCAIL2];

/// The artifact family this (arm, tier) binds.
fn artifact_family(arm: Arm, tier: &str) -> Family {
    match (tier, arm.dense) {
        ("bf16", Some(dense)) => dense,
        _ => arm.packed,
    }
}

/// The production calibration identity this cell's loaded generator publishes.
fn production_fingerprint(arm: Arm, tier: &str) -> Result<String, String> {
    Ok(match arm.provider {
        // sc-19223's table is keyed on (tier, offload policy), and the dense cell's two published
        // strings omit the tier token entirely — they named that cell before the packed ones
        // existed and are folded in byte-identical.
        WAN_TI2V_5B_ID => match tier {
            "bf16" => "sc-19223-wan2-2-ti2v-5b-candle-sequential-load-v1".to_owned(),
            "q4" | "q8" => format!("sc-19223-wan2-2-ti2v-5b-candle-{tier}-sequential-load-v1"),
            other => return Err(format!("{LABEL}: unsupported TI2V-5B tier {other:?}")),
        },
        // sc-22736's shared A14B authority, whose dense token is `dense`.
        WAN_T2V_A14B_ID | WAN_I2V_A14B_ID => {
            let route = if arm.provider == WAN_T2V_A14B_ID {
                "t2v"
            } else {
                "i2v"
            };
            let token = if tier == "bf16" { "dense" } else { tier };
            format!("sc-22736-wan2-2-{route}-a14b-candle-{token}-v1")
        }
        // sc-22736's SCAIL-2 table, whose tier token is the DIRECTORY name.
        SCAIL2_ID => format!("scail2-14b-{tier}-candle-resident-eager-v1"),
        other => return Err(format!("{LABEL} does not implement provider {other:?}")),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    width: u32,
    height: u32,
    frames: u32,
}

fn arm(request: &Value) -> Result<Arm, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    ARMS.into_iter()
        .find(|arm| arm.provider == provider && arm.model_id == model_id)
        .ok_or_else(|| {
            format!("{LABEL} does not implement (provider {provider:?}, modelId {model_id:?})")
        })
}

/// Whether this arm is one of the four this module implements — asked before `run` commits to it,
/// so `candle.rs`'s dispatch can route by provider id alone.
pub(super) fn implements(provider: &str) -> bool {
    ARMS.iter().any(|arm| arm.provider == provider)
}

/// The engine's own answer to "does this route admit this bucket at this rate?".
fn validate_geometry(arm: Arm, geometry: Geometry) -> Result<(), String> {
    let Geometry {
        width,
        height,
        frames,
    } = geometry;
    match arm.route {
        Some(route) => {
            if !route.public_geometries().contains(&(width, height)) {
                return Err(format!(
                    "{} admits only the buckets {:?}, got {width}x{height}",
                    arm.provider,
                    route.public_geometries()
                ));
            }
            if !route.accepts_rate(arm.fps, frames) {
                return Err(format!(
                    "{} refuses {frames} frames at {} fps; the plan geometry is outside the \
                     route's own public rate menu",
                    arm.provider, arm.fps
                ));
            }
        }
        None => {
            if !candle_gen_scail2::memory_strategy::PUBLIC_BUCKETS.contains(&(width, height)) {
                return Err(format!(
                    "{} admits only the buckets {:?}, got {width}x{height}",
                    arm.provider,
                    candle_gen_scail2::memory_strategy::PUBLIC_BUCKETS
                ));
            }
            if !candle_gen_scail2::memory_strategy::PUBLIC_FRAMES.contains(&frames) {
                return Err(format!(
                    "{} admits only the frame counts {:?}, got {frames}",
                    arm.provider,
                    candle_gen_scail2::memory_strategy::PUBLIC_FRAMES
                ));
            }
        }
    }
    Ok(())
}

fn target_geometry(request: &Value, arm: Arm) -> Result<Geometry, String> {
    let geometry = protocol::planned(request)?
        .pointer("/target/geometry")
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
        return Err(format!(
            "{LABEL} requires geometry.batch == 1 (these engines render one clip per request)"
        ));
    }
    let resolved = Geometry {
        width: axis("width")?,
        height: axis("height")?,
        frames: axis("frames")?,
    };
    validate_geometry(arm, resolved)?;
    Ok(resolved)
}

fn validate_mode(request: &Value, arm: Arm) -> Result<(), String> {
    let mode = protocol::planned(request)?
        .pointer("/target/mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != arm.mode {
        return Err(format!(
            "{} is captured under {:?}; the plan declares {mode:?}",
            arm.provider, arm.mode
        ));
    }
    Ok(())
}

fn validate_fixture(
    request: &Value,
    arm: Arm,
    tier: &str,
    geometry: Geometry,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let expected = format!(
        "{}-candle-{tier}-{}x{}-f{}-fps{}-seed{SEED}",
        arm.slug, geometry.width, geometry.height, geometry.frames, arm.fps
    );
    if fixture != expected {
        return Err(format!(
            "planned.fixture {fixture:?} must be {expected:?} — the fixture names the member, \
             lane, tier, geometry, cadence and calibration seed"
        ));
    }
    Ok(())
}

struct Artifact {
    repository: String,
    revision: String,
    spec: LoadSpec,
}

/// Resolve the artifact this cell opens, and refuse a root whose layout or TIER SUFFIX is not the
/// plan's.
///
/// The suffix check is why the artifact is resolved per (lane, tier): a packed rehost ships both
/// packed tiers under ONE revision, so a `q4` row handed the `q8` root would load, render, and
/// produce a well-formed record for the wrong cell.
fn load_spec(arm: Arm, tier: &str, load_shape: LoadShape) -> Result<Artifact, String> {
    let family = artifact_family(arm, tier);
    let repository = protocol::required_env(family.repository_env)?;
    let revision = protocol::required_env(family.revision_env)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(family.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", family.root_env))?;
    match family.layout {
        Layout::Tiered => protocol::validate_huggingface_snapshot_subpath(
            &root,
            &repository,
            &revision,
            &[tier],
            family.repository,
        )?,
        Layout::Flat => protocol::validate_huggingface_revision_root(
            &root,
            &repository,
            &revision,
            family.repository,
        )?,
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(arm.offload_policy)
        .with_load_shape(load_shape)
        // Gated on it: `candle-gen-wan`'s A14B loaders seal a memory receipt only for a spec that
        // names the route they resolved, and `candle-gen-scail2` marks an artifact canonical only
        // for its own route.
        .with_resolved_route(arm.provider.to_owned());
    spec.precision = Precision::Bf16;
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(Artifact {
        repository,
        revision,
        spec,
    })
}

/// A deterministic, non-degenerate RGB8 plane — the same generator the MLX arm uses, so the two
/// lanes present byte-identical carriers for the same cell.
fn plane(width: u32, height: u32, salt: u32) -> Image {
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
    Image {
        width,
        height,
        pixels,
    }
}

fn generation_request(arm: Arm, geometry: Geometry) -> GenerationRequest {
    let Geometry {
        width,
        height,
        frames,
    } = geometry;
    let conditioning = match arm.carrier {
        Carrier::None => Vec::new(),
        Carrier::Reference => vec![Conditioning::Reference {
            image: plane(width, height, 1),
            strength: None,
        }],
        Carrier::Animation => vec![
            Conditioning::Reference {
                image: plane(width, height, 2),
                strength: None,
            },
            Conditioning::Mask {
                image: plane(width, height, 3),
            },
            Conditioning::ControlClip {
                frames: (0..frames)
                    .map(|index| plane(width, height, 100 + index))
                    .collect(),
                mask: (0..frames)
                    .map(|index| plane(width, height, 5_000 + index))
                    .collect(),
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
    };
    GenerationRequest {
        prompt: "a slow dolly across a rain-slick harbour wall at dusk, cinematic".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(SEED),
        steps: Some(arm.steps),
        frames: Some(frames),
        fps: Some(arm.fps),
        video_mode: Some(arm.mode.to_owned()),
        conditioning,
        ..Default::default()
    }
}

/// The request receipt this render will present, minted by the ENGINE's own public helper.
fn evidence_revision(
    arm: Arm,
    spec: &LoadSpec,
    request: &GenerationRequest,
    selection: MemorySelection,
) -> Result<String, String> {
    match arm.route {
        Some(_) => {
            let prepared = candle_gen_wan::i2v_memory_strategy::prepare(spec, arm.provider)
                .map_err(|error| format!("seal the {} receipt: {error}", arm.provider))?;
            candle_gen_wan::i2v_memory_strategy::request_evidence_revision(&prepared, request)
                .map_err(|error| format!("mint the {} request receipt: {error}", arm.provider))
        }
        None => {
            let evidence =
                candle_gen_scail2::memory_strategy::structural_resident_evidence(spec)
                    .map_err(|error| format!("seal the {} receipt: {error}", arm.provider))?;
            candle_gen_scail2::memory_strategy::request_evidence_revision(
                &evidence, request, selection,
            )
            .map_err(|error| format!("mint the {} request receipt: {error}", arm.provider))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn context(
    arm: Arm,
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: Geometry,
    evidence_revision: &str,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other(arm.mode.to_owned()),
        has_reference: arm.carrier.reference_count() > 0,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
            reference_count: arm.carrier.reference_count(),
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: evidence_revision.to_owned(),
    }
}

/// The `candle:{wan2_2_ti2v_5b,wan2_2_t2v_14b,wan2_2_i2v_14b,scail2_14b}` arm (sc-22736).
pub(super) fn run(request: &Value) -> Result<Value, String> {
    let arm = arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    validate_mode(request, arm)?;
    let geometry = target_geometry(request, arm)?;
    let tier = planned_tier(request)?;
    validate_fixture(request, arm, tier, geometry)?;
    let load_shape = planned_video_load_shape(request)?;
    if load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "the {} Candle lane declares no bounded-transformer-residency route, so \
             `evaluate_declared_candle_load_shape` hands the spec back eager; the plan declares {}",
            arm.provider,
            load_shape_key(load_shape)
        ));
    }
    let selection = planned_selection(request)?;
    let expected_fingerprint = production_fingerprint(arm, tier)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }

    let mut resolved = load_spec(arm, tier, load_shape)?;
    if arm.route.is_some() {
        candle_gen_wan::i2v_memory_strategy::prepare_load_spec(&mut resolved.spec, arm.provider)
            .map_err(|error| format!("prepare the {} load spec: {error}", arm.provider))?;
    }

    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(arm.provider, &resolved.spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
    vram.end_load(load_sample);
    let contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} exposed no memory-strategy contract",
            arm.provider
        )
    })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            arm.provider
        )
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity for the \
             {tier} artifact; the production identity for this cell is {expected_fingerprint}",
            arm.provider,
            protocol::INFERENCE_PIN
        )
    })?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if calibration.load_shape != load_shape {
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
    let mut generation = generation_request(arm, geometry);
    let receipt = evidence_revision(arm, &resolved.spec, &generation, selection)?;
    let probe = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&context(
            arm,
            selection,
            calibration,
            fingerprint,
            geometry,
            &receipt,
            total_bytes,
            predicted,
        ))
    };
    if !matches!(
        probe(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected a fitting probe budget; the scenario rejections below would be \
             a blanket refusal, not evidence",
            arm.provider
        ));
    }
    if !matches!(
        probe(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted an unknown/zero memory budget",
            arm.provider
        ));
    }
    if !matches!(
        probe("stale-wan-scail2-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            arm.provider
        ));
    }

    let run_context = context(
        arm,
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        &receipt,
        hardware_bytes,
        1,
    );
    let mut scope = generator
        .begin_memory_strategy_request(&run_context)
        .map_err(|error| format!("begin {} capture scope: {error}", arm.provider))?
        .ok_or_else(|| format!("{} selection did not create a provider scope", arm.provider))?;
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {} capture strategy: {error}", arm.provider))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {} conditioning: {error}", arm.provider))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut peaks = [None, None, None];
    let mut phase_error: Option<String> = None;
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
            phase_error = Some(format!("leave {} {phase:?}: {error}", arm.provider));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {} {next:?}: {error}", arm.provider));
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
            return Err(format!("{} generation failed: {message}", arm.provider));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {} terminal phase: {error}", arm.provider))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {} capture scope: {error}", arm.provider))?;

    let (frames, fps) = match output {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        GenerationOutput::Images(_) => {
            return Err(format!(
                "{} returned images, not a video clip",
                arm.provider
            ))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!(
                "{} returned an audio track, not a video clip",
                arm.provider
            ))
        }
    };
    if fps != arm.fps {
        return Err(format!(
            "{} returned fps {fps} for a {} fps request",
            arm.provider, arm.fps
        ));
    }
    if frames.len() as u64 != u64::from(geometry.frames) {
        return Err(format!(
            "{} rendered {} frames for a {}-frame request",
            arm.provider,
            frames.len(),
            geometry.frames
        ));
    }
    let first = frames
        .first()
        .ok_or_else(|| format!("{} render returned no first frame", arm.provider))?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err(format!(
            "{} render returned a degenerate first frame",
            arm.provider
        ));
    }

    let conditioning_bytes = decimal_gb_to_bytes(
        peaks[0]
            .ok_or_else(|| format!("{} did not expose the conditioning boundary", arm.provider))?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{} did not expose the denoise boundary", arm.provider))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{} did not complete decode sampling", arm.provider))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "this arm drives the provider's own request scope for one measured render; it runs no warm ",
        "repeat and injects no calibration fault, so the determinism, cancellation and ",
        "authorized-error scenarios are unexecuted and this record claims nothing about them"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        arm.execution_path,
        protocol::PlainGatedFragment {
            artifact: artifact(&resolved.repository, &resolved.revision, tier),
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{}:f{}:{}x{}:fps{}:seed{SEED}",
                    loadability_fingerprint(&resolved.repository, &resolved.revision, tier),
                    geometry.frames,
                    geometry.width,
                    geometry.height,
                    arm.fps,
                ),
            }),
            diagnostics: protocol::diagnostics(
                &format!("memory-candle-adapter:{}", arm.slug),
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(geometry.frames)),
                    ("renderedFps", "fps", u64::from(fps)),
                    (
                        "referenceCount",
                        "count",
                        u64::from(arm.carrier.reference_count()),
                    ),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arm_plans_a_geometry_its_own_engine_admits() {
        for arm in ARMS {
            let geometry = match arm.provider {
                WAN_TI2V_5B_ID => Geometry {
                    width: 832,
                    height: 480,
                    frames: 121,
                },
                SCAIL2_ID => Geometry {
                    width: 832,
                    height: 480,
                    frames: 77,
                },
                _ => Geometry {
                    width: 1280,
                    height: 720,
                    frames: 77,
                },
            };
            validate_geometry(arm, geometry)
                .unwrap_or_else(|error| panic!("{}: {error}", arm.provider));
            let crossed = Geometry {
                frames: geometry.frames + 1,
                ..geometry
            };
            assert!(
                validate_geometry(arm, crossed).is_err(),
                "{}: an off-menu frame count must be refused",
                arm.provider
            );
        }
    }

    /// The Wan routes' dense leg is a DIFFERENT artifact in a DIFFERENT layout; SCAIL-2's is not.
    #[test]
    fn the_dense_leg_is_resolved_per_tier() {
        for arm in [TI2V_5B, T2V_A14B, I2V_A14B] {
            let packed = artifact_family(arm, "q4");
            assert_eq!(packed.repository, arm.packed.repository);
            assert_eq!(packed.layout, Layout::Tiered);
            assert_eq!(artifact_family(arm, "q8").repository, arm.packed.repository);
            let dense = artifact_family(arm, "bf16");
            assert_ne!(
                dense.repository_env, packed.repository_env,
                "{}: the dense leg is a different artifact family",
                arm.provider
            );
            assert!(
                dense.repository.starts_with("Wan-AI/"),
                "{}",
                dense.repository
            );
            assert_eq!(
                dense.layout,
                Layout::Flat,
                "{}: the upstream Diffusers checkpoint has no tier subtree",
                arm.provider
            );
        }
        for tier in ["bf16", "q4", "q8"] {
            let family = artifact_family(SCAIL2, tier);
            assert_eq!(family.repository, "SceneWorks/scail2-mlx");
            assert_eq!(family.repository_env, "SCENEWORKS_SCAIL2_REPOSITORY");
            assert_eq!(
                family.layout,
                Layout::Tiered,
                "SCAIL-2 ships one repository with all three tiers to both lanes"
            );
        }
    }

    #[test]
    fn the_production_identity_table_is_per_cell_and_never_a_conformance_string() {
        let mut seen = Vec::new();
        for arm in ARMS {
            for tier in ["bf16", "q4", "q8"] {
                let fingerprint = production_fingerprint(arm, tier)
                    .unwrap_or_else(|error| panic!("{} {tier}: {error}", arm.provider));
                assert!(
                    !fingerprint.contains("weights-free"),
                    "{fingerprint} is a conformance string"
                );
                assert!(
                    fingerprint.contains("candle"),
                    "{fingerprint} does not name this lane"
                );
                assert!(
                    !seen.contains(&fingerprint),
                    "{fingerprint} names more than one cell"
                );
                seen.push(fingerprint);
            }
        }
        assert_eq!(seen.len(), 12, "four arms x three shipped tiers");
    }

    /// The residency each arm loads under is the one `candle_video_offload_policy` names.
    #[test]
    fn the_offload_policy_is_the_workers_own() {
        assert_eq!(TI2V_5B.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(T2V_A14B.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(I2V_A14B.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(SCAIL2.offload_policy, OffloadPolicy::Resident);
    }

    #[test]
    fn the_synthetic_carriers_match_the_engines_request_contracts() {
        let geometry = Geometry {
            width: 832,
            height: 480,
            frames: 77,
        };
        assert!(generation_request(T2V_A14B, geometry)
            .conditioning
            .is_empty());
        assert!(matches!(
            generation_request(I2V_A14B, geometry)
                .conditioning
                .as_slice(),
            [Conditioning::Reference { strength: None, .. }]
        ));
        let animation = generation_request(SCAIL2, geometry);
        let carrier = animation
            .scail2_animation_conditioning()
            .expect("the synthetic SCAIL-2 carrier satisfies the engine's own validator");
        assert_eq!(carrier.driving_frames.len(), geometry.frames as usize);
        assert_eq!(carrier.driving_masks.len(), carrier.driving_frames.len());
    }
}
