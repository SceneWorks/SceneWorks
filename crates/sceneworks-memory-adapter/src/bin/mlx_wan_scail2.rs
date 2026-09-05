//! Physical MLX capture arm for the Wan 2.2 family and SCAIL-2 (sc-22736, epic sc-22723 E1).
//!
//! FOUR engine providers, ONE arm, because the four differ only in coordinates a table can hold:
//! the artifact family, the public request carrier, the rate menu and the production identity. The
//! measured path is identical — resolve the artifact, seal the receipt the way the worker's
//! memory-aware load does, load through `runtime_macos::catalog().media()` (the seam
//! `crates/sceneworks-worker/src/inference_runtime.rs` wraps), read the LOADED generator's own
//! contract, drive the four admission probes through the provider's own registered check, and
//! measure three synchronized phase peaks off the boundaries `generate` already emits.
//!
//! ## Nothing here restates an engine envelope
//!
//! Every geometry, rate and carrier rule is asked of the pinned engine's own symbol:
//!
//! * the three Wan routes go through `gen_core::wan_i2v_memory::WanI2vRoute` —
//!   `public_geometries()` and `accepts_rate()`, the shared authority both Wan lanes seal against;
//! * SCAIL-2 goes through `mlx_gen_scail2::memory_strategy::{PUBLIC_BUCKETS, PUBLIC_FRAMES}`.
//!
//! A plan row naming a bucket or a rate the engine stopped admitting therefore fails HERE, before
//! any weights are opened, instead of deep inside a multi-gigabyte load.

use super::*;
use mlx_gen::gen_core::wan_i2v_memory::WanI2vRoute;
use mlx_gen::gen_core::ReplacementMode;

const SEED: u64 = 22_736;
const LABEL: &str = "MLX Wan2.2/SCAIL-2";

/// Determinism envelope for a warm repeat of one of these clips.
///
/// The same claim the other video arms make — repeat determinism on ONE loaded provider with an
/// identical request — so the same published FLUX.2 envelope applies rather than a looser bound
/// invented here. The mandatory `+64` negative mutation is more than eight times the maximum.
const MAX_THRESHOLD: f64 = FLUX2_MAX_THRESHOLD;
const MEAN_THRESHOLD: f64 = FLUX2_MEAN_THRESHOLD;
const RMS_THRESHOLD: f64 = FLUX2_RMS_THRESHOLD;

/// The public request carrier one of these routes takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Carrier {
    /// A prompt and nothing else (`wan_2_2`, `wan_2_2_t2v_14b`).
    None,
    /// One full-strength `Reference` (`wan_2_2_i2v_14b`).
    Reference,
    /// SCAIL-2's ordered `Reference` + `Mask` + `ControlClip`: a character still with its mask, and
    /// one driving frame plus one driving mask per generated frame.
    Animation,
}

impl Carrier {
    /// The `MemoryGeometry::reference_count` the admitted context declares.
    const fn reference_count(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Reference | Self::Animation => 1,
        }
    }
}

/// One capturable (engine provider, catalog model) cell of this family.
#[derive(Clone, Copy)]
struct Arm {
    /// The engine registry id the adapter loads.
    provider: &'static str,
    /// The SceneWorks catalog id the plan key is built from.
    model_id: &'static str,
    /// The Wan authority route, or `None` for SCAIL-2, which has its own envelope.
    route: Option<WanI2vRoute>,
    /// The public video mode this route is admitted under. It is an EVIDENCE KEY, not a label:
    /// gen-core's `standard_memory_strategy_safety_check` matches it against each adopted
    /// decode-geometry record's own mode, so a probe under one spelling cannot answer a request
    /// asked under another.
    mode: &'static str,
    carrier: Carrier,
    /// The one rate the manifest ships for this route.
    fps: u32,
    steps: u32,
    /// `SCENEWORKS_<env>_{REPOSITORY,REVISION,ROOT}` — the family `measure-memory-catalog.mjs`
    /// exports for this (provider, MLX lane).
    repository_env: &'static str,
    revision_env: &'static str,
    root_env: &'static str,
    /// The repository the env family must name, so a mis-exported root is refused by name.
    repository: &'static str,
    /// Fixture slug and the execution path a plain-overlay target settles against.
    slug: &'static str,
    execution_path: &'static str,
}

const TI2V_5B: Arm = Arm {
    provider: WAN_TI2V_5B_PROVIDER,
    model_id: "wan_2_2",
    route: Some(WanI2vRoute::Ti2v5b),
    mode: "text_to_video",
    carrier: Carrier::None,
    fps: 24,
    steps: 20,
    repository_env: "SCENEWORKS_WAN22_TI2V_5B_MLX_REPOSITORY",
    revision_env: "SCENEWORKS_WAN22_TI2V_5B_MLX_REVISION",
    root_env: "SCENEWORKS_WAN22_TI2V_5B_MLX_ROOT",
    repository: "SceneWorks/wan2.2-ti2v-5b-mlx",
    slug: "wan-2-2-ti2v-5b",
    execution_path: "the MLX Wan2.2 TI2V-5B base text-to-video path",
};

const T2V_A14B: Arm = Arm {
    provider: WAN_T2V_A14B_PROVIDER,
    model_id: "wan_2_2_t2v_14b",
    route: Some(WanI2vRoute::T2v14b),
    mode: "text_to_video",
    carrier: Carrier::None,
    fps: 16,
    steps: 40,
    repository_env: "SCENEWORKS_WAN22_T2V_A14B_MLX_REPOSITORY",
    revision_env: "SCENEWORKS_WAN22_T2V_A14B_MLX_REVISION",
    root_env: "SCENEWORKS_WAN22_T2V_A14B_MLX_ROOT",
    repository: "SceneWorks/wan2.2-t2v-a14b-mlx",
    slug: "wan-2-2-t2v-a14b",
    execution_path: "the MLX Wan2.2 T2V-A14B dual-expert text-to-video path",
};

const I2V_A14B: Arm = Arm {
    provider: WAN_I2V_A14B_PROVIDER,
    model_id: "wan_2_2_i2v_14b",
    route: Some(WanI2vRoute::I2v14b),
    mode: "image_to_video",
    carrier: Carrier::Reference,
    fps: 16,
    steps: 40,
    repository_env: "SCENEWORKS_WAN22_I2V_A14B_MLX_REPOSITORY",
    revision_env: "SCENEWORKS_WAN22_I2V_A14B_MLX_REVISION",
    root_env: "SCENEWORKS_WAN22_I2V_A14B_MLX_ROOT",
    repository: "SceneWorks/wan2.2-i2v-a14b-mlx",
    slug: "wan-2-2-i2v-a14b",
    execution_path: "the MLX Wan2.2 I2V-A14B dual-expert image-to-video path",
};

const SCAIL2: Arm = Arm {
    provider: SCAIL2_PROVIDER,
    model_id: "scail2_14b",
    route: None,
    mode: "animation",
    carrier: Carrier::Animation,
    fps: 16,
    steps: 20,
    repository_env: "SCENEWORKS_SCAIL2_REPOSITORY",
    revision_env: "SCENEWORKS_SCAIL2_REVISION",
    root_env: "SCENEWORKS_SCAIL2_ROOT",
    repository: "SceneWorks/scail2-mlx",
    slug: "scail2-14b",
    execution_path: "the MLX SCAIL-2 character-animation path",
};

const ARMS: [Arm; 4] = [TI2V_5B, T2V_A14B, I2V_A14B, SCAIL2];

/// The production calibration identity this cell's loaded generator publishes.
///
/// Hand-kept ONLY in the sense that the strings are spelled here: each is checked against the
/// LOADED contract before a byte of evidence is recorded, so a drift is a refused capture rather
/// than a mislabelled anchor. Checked BEFORE the load as well, against the plan, so a row still
/// carrying a weights-free conformance string fails in milliseconds.
fn production_fingerprint(arm: Arm, tier: &str) -> Result<String, String> {
    let tier_token = match tier {
        "bf16" => "dense",
        other => other,
    };
    Ok(match arm.provider {
        // sc-19236's own per-tier table (`mlx-gen-wan/src/memory_strategy.rs`), which names the
        // packing group and the Q8 text-encoder floor in the packed cells.
        "wan2_2_ti2v_5b" => match tier {
            "bf16" => "sc-19236-wan2-2-ti2v-5b-mlx-dense-v1".to_owned(),
            "q4" => "sc-19236-wan2-2-ti2v-5b-mlx-q4-g64-teq8-v1".to_owned(),
            "q8" => "sc-19236-wan2-2-ti2v-5b-mlx-q8-g64-teq8-v1".to_owned(),
            other => return Err(format!("{LABEL}: unsupported TI2V-5B tier {other:?}")),
        },
        // sc-22736's shared A14B authority (`gen_core::wan_i2v_memory`).
        "wan2_2_t2v_14b" => format!("sc-22736-wan2-2-t2v-a14b-mlx-{tier_token}-v1"),
        "wan2_2_i2v_14b" => format!("sc-22736-wan2-2-i2v-a14b-mlx-{tier_token}-v1"),
        // sc-22736's SCAIL-2 table, whose tier token is the DIRECTORY name, not `dense`.
        "scail2_14b" => format!("scail2-14b-{tier}-mlx-resident-eager-v1"),
        other => return Err(format!("{LABEL} does not implement provider {other:?}")),
    })
}

/// The declared geometry, read as real values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    width: u32,
    height: u32,
    frames: u32,
}

/// Resolve the arm from the planned `(provider, modelId)` pair and refuse any other by name.
///
/// Both halves matter: the provider selects the engine, and the model id selects the artifact
/// family, so a row that pairs one route's provider with another's catalog id would otherwise
/// measure one checkpoint and file the record against the other.
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
            format!(
                "{LABEL} does not implement (provider {provider:?}, modelId {model_id:?}); the \
                 implemented pairs are {:?}",
                ARMS.map(|arm| (arm.provider, arm.model_id))
            )
        })
}

/// The engine's own answer to "does this route admit this bucket at this rate?".
///
/// Nothing below is a literal restated from the manifest: the Wan routes answer through the shared
/// `WanI2vRoute` authority both lanes seal against, and SCAIL-2 through the two `pub` slices its
/// own `memory_strategy` validates every request with.
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
            if !mlx_gen_scail2::memory_strategy::PUBLIC_BUCKETS.contains(&(width, height)) {
                return Err(format!(
                    "{} admits only the buckets {:?}, got {width}x{height}",
                    arm.provider,
                    mlx_gen_scail2::memory_strategy::PUBLIC_BUCKETS
                ));
            }
            if !mlx_gen_scail2::memory_strategy::PUBLIC_FRAMES.contains(&frames) {
                return Err(format!(
                    "{} admits only the frame counts {:?}, got {frames}",
                    arm.provider,
                    mlx_gen_scail2::memory_strategy::PUBLIC_FRAMES
                ));
            }
        }
    }
    Ok(())
}

/// Read the four declared geometry axes and validate them against the engine.
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

/// The planned mode must be the one public mode this route is admitted under.
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

/// Bind the fixture to the member, lane, tier, full geometry and cadence, so a bf16 record can
/// never be emitted against a q4 capture that merely reused the fixture string.
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
        "{}-mlx-{tier}-{}x{}-f{}-fps{}-seed{SEED}",
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

/// Everything a capture resolved, kept together so the record's provenance comes from the values
/// the load actually used rather than from the plan.
struct Artifact {
    repository: String,
    revision: String,
    root: PathBuf,
    spec: LoadSpec,
}

impl Artifact {
    fn json(&self, tier: &str) -> Value {
        json!({
            "repository": self.repository,
            "resolvedRevision": self.revision,
            "variant": tier,
        })
    }

    fn loadability_fingerprint(&self, tier: &str) -> String {
        let mut hash = Sha256::new();
        hash.update(self.repository.as_bytes());
        hash.update(b"\0");
        hash.update(self.revision.as_bytes());
        hash.update(b"\0");
        hash.update(tier.as_bytes());
        format!("{:x}", hash.finalize())
    }
}

/// Resolve the artifact this cell opens, and refuse a root whose TIER SUFFIX is not the plan's.
///
/// The suffix check is the whole point of resolving per (lane, tier): every MLX rehost in this
/// family ships all three tiers under ONE revision, so a `q4` plan row handed the `bf16` root would
/// load, render and produce a perfectly well-formed record for the wrong cell.
fn load_spec(arm: Arm, tier: &str, load_shape: LoadShape) -> Result<Artifact, String> {
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    protocol::validate_huggingface_snapshot_subpath(
        &root,
        &repository,
        &revision,
        &[tier],
        arm.repository,
    )?;
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape)
        // The receipt is gated on it: `mlx-gen-wan`'s A14B loaders prepare a memory receipt only
        // when the spec names the route it resolved, and `mlx-gen-scail2` marks an artifact
        // canonical only for its own route. A capture that omitted it would load a generator with
        // no memory contract at all.
        .with_resolved_route(arm.provider.to_owned());
    spec.precision = Precision::Bf16;
    if let Some(quant) = numeric_quant(arm, tier)? {
        spec = spec.with_quant(quant);
    }
    Ok(Artifact {
        repository,
        revision,
        root,
        spec,
    })
}

/// `LoadSpec::quantize` for a named tier — on the routes whose engine reads it.
///
/// The two MLX conventions in this family are genuinely different, and getting it wrong is a load
/// REFUSAL, not a mislabel:
///
/// * the three **Wan** routes reconcile `quantize` against the staged tier's own
///   `config.json` marker (`memory_strategy::resolved_numeric_tier`: "requested tier … does not
///   match config.json's authoritative checkpoint tier"), and the A14B authority reads the
///   snapshot's directory name only when it is unset. Passing it is an assertion about the
///   directory on disk;
/// * **SCAIL-2** refuses it outright on a canonical-tier load: `ArtifactReceipt::capture` rejects
///   "a second on-load quantization", and `production_calibration_identity` withholds the identity
///   whenever it is set. On that route the tier is the DIRECTORY and nothing else — which the
///   `validate_huggingface_snapshot_subpath(.., &[tier], ..)` above has already proven.
///
/// `route` is the discriminator because it is the same field that decides whether a Wan receipt
/// pre-pass runs at all, so the two cannot disagree about which convention a cell is on.
fn numeric_quant(arm: Arm, tier: &str) -> Result<Option<Quant>, String> {
    let quant = match tier {
        "bf16" => None,
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        other => return Err(format!("{LABEL} has no shipped tier {other:?}")),
    };
    Ok(if arm.route.is_some() { quant } else { None })
}

/// A deterministic, non-degenerate RGB8 plane. Every carrier byte a capture presents is generated
/// here rather than staged, so the record's request identity is reproducible from this source
/// alone — the same choice the PuLID and LTX arms make for their synthetic carriers.
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

/// The one fresh planned request every capture of this family renders.
fn generation_request(arm: Arm, geometry: Geometry) -> GenerationRequest {
    let Geometry {
        width,
        height,
        frames,
    } = geometry;
    let conditioning = match arm.carrier {
        Carrier::None => Vec::new(),
        // `wan_i2v_memory::reference` requires exactly one Reference at unset/1.0 strength.
        Carrier::Reference => vec![Conditioning::Reference {
            image: plane(width, height, 1),
            strength: None,
        }],
        // `GenerationRequest::scail2_animation_conditioning` requires exactly ordered
        // Reference(strength unset) + Mask + ControlClip, with one driving mask per driving frame,
        // every plane at the request geometry, `masking_strength = 1`, `start_frame = 0` and the
        // default full-person replacement mode.
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
        // The two carrier-free Wan routes publish their mode either way; the I2V and SCAIL-2 routes
        // require it, and every engine here keys its request receipt on the exact spelling.
        video_mode: Some(arm.mode.to_owned()),
        conditioning,
        ..Default::default()
    }
}

/// The admission context for the safety scenarios, in the shape the worker admits these routes
/// under.
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
        // A parameter only so the stale-evidence probe can pass a deliberate mismatch; every real
        // call site passes `calibration.fingerprint`.
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

/// The request receipt this render will present, minted by the ENGINE's own public helper.
///
/// Both families bind admission to a receipt derived from the sealed artifact and the exact request
/// bytes, so a capture cannot invent one: the Wan routes go through
/// `wan_i2v_memory::request_evidence_revision` over the receipt this arm sealed, and SCAIL-2
/// through `mlx_gen_scail2::memory_strategy::request_evidence_revision` over the structural
/// evidence its own prepared receipt carries.
fn evidence_revision(
    arm: Arm,
    spec: &LoadSpec,
    request: &GenerationRequest,
    selection: MemorySelection,
) -> Result<String, String> {
    match arm.route {
        Some(_) => {
            let prepared = mlx_gen_wan::i2v_memory_strategy::prepare(spec, arm.provider)
                .map_err(|error| format!("seal the {} receipt: {error}", arm.provider))?;
            mlx_gen_wan::i2v_memory_strategy::request_evidence_revision(&prepared, request)
                .map_err(|error| format!("mint the {} request receipt: {error}", arm.provider))
        }
        None => {
            let evidence = mlx_gen_scail2::memory_strategy::structural_resident_evidence(spec)
                .map_err(|error| format!("seal the {} receipt: {error}", arm.provider))?;
            mlx_gen_scail2::memory_strategy::request_evidence_revision(
                &evidence, request, selection,
            )
            .map_err(|error| format!("mint the {} request receipt: {error}", arm.provider))
        }
    }
}

fn quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= MAX_THRESHOLD && mean <= MEAN_THRESHOLD && rms <= RMS_THRESHOLD
}

/// One exact tuple per plan row.
fn complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:{wan2_2_ti2v_5b,wan2_2_t2v_14b,wan2_2_i2v_14b,scail2_14b}` arm (sc-22736).
pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Everything cheap and refusable first, so a mis-planned row costs milliseconds rather than a
    // multi-gigabyte load: the member, the mode, the geometry against the ENGINE's own menus, the
    // fixture, the tier, and the plan's identity against this arm's table.
    let arm = arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    validate_mode(request, arm)?;
    let geometry = target_geometry(request, arm)?;
    let tier = planned_qwen_tier(request)?;
    validate_fixture(request, arm, tier, geometry)?;
    let load_shape = planned_load_shape(request)?;
    if load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "the {} MLX lane is captured {}; the plan declares {}",
            arm.provider,
            protocol::LOAD_SHAPE_EAGER,
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

    let mut artifact = load_spec(arm, tier, load_shape)?;
    // A Wan route's loader prepares a memory receipt only for a spec whose file pins are already
    // prepared (`model.rs`: `spec.prepared_file_pins().is_prepared()`), so the capture must do what
    // the worker's memory-aware load does — otherwise the generator loads with no calibration
    // identity at all. SCAIL-2 seals its own shared-tier pins inside `PreparedMemory::prepare` and
    // needs no pre-pass.
    if arm.route.is_some() {
        mlx_gen_wan::i2v_memory_strategy::prepare_load_spec(&mut artifact.spec, arm.provider)
            .map_err(|error| format!("prepare the {} load spec: {error}", arm.provider))?;
    }
    let staged_bytes = safetensors_bytes(&artifact.root)?;

    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(arm.provider, &artifact.spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
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
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity for the \
             {tier} artifact; the production identity for this cell is {expected_fingerprint}, so \
             this cell captures only at a pin that carries it",
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
    let planned_render = generation_request(arm, geometry);
    let receipt = evidence_revision(arm, &artifact.spec, &planned_render, selection)?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
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
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the two rejections
    // below cannot pass through a blanket refusal.
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected a fitting probe budget; the scenario rejections below would be \
             a blanket refusal, not evidence",
            arm.provider
        ));
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted an unknown/zero memory budget",
            arm.provider
        ));
    }
    if !matches!(
        safety("stale-wan-scail2-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            arm.provider
        ));
    }

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
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let (measured, output_fps, _) = diagnostic_video_frames(
        generator
            .generate(&planned_render, &mut |progress| match progress {
                Progress::Step { current: 1, .. } => {
                    conditioning.set(PhaseMemory::capture());
                    reset_peak_memory();
                }
                Progress::Decoding => {
                    denoise.set(PhaseMemory::capture());
                    reset_peak_memory();
                }
                _ => {}
            })
            .map_err(|error| format!("generate measured {} render: {error}", arm.provider))?,
        LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak",
            arm.provider
        ));
    }
    if measured.len() as u64 != u64::from(geometry.frames) {
        return Err(format!(
            "{} rendered {} frames for a {}-frame request",
            arm.provider,
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != arm.fps {
        return Err(format!(
            "{} returned fps {output_fps} for a {} fps request",
            arm.provider, arm.fps
        ));
    }
    let first = measured
        .first()
        .ok_or_else(|| format!("{} render returned no first frame", arm.provider))?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err(format!(
            "{} render returned a degenerate first frame",
            arm.provider
        ));
    }

    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    if !matches!(
        safety(&calibration.fingerprint, predicted, predicted),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected an exact-fit calibrated budget",
            arm.provider
        ));
    }

    // Warm-repeat determinism and allocator cleanup bounds on this exact loaded provider.
    clear_cache();
    reset_peak_memory();
    let (baseline, _, _) = diagnostic_video_frames(
        generator
            .generate(&planned_render, &mut |_| {})
            .map_err(|error| format!("generate warm {} control: {error}", arm.provider))?,
        LABEL,
    )?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(&measured, &baseline)?;
    if !quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "{} warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}",
            arm.provider
        ));
    }
    reset_peak_memory();
    let (warm, _, _) = diagnostic_video_frames(
        generator
            .generate(&planned_render, &mut |_| {})
            .map_err(|error| format!("generate warm {} repeat: {error}", arm.provider))?,
        LABEL,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "{} warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%",
            arm.provider
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "{} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the clean warm \
             cleanup {clean_post_cleanup:?} plus {} bytes",
            arm.provider, cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = video_max_mean_rms_abs(&measured, &warm)?;
    if !quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            arm.provider
        ));
    }

    // Arm-internal negative-mutation falsifiability check: a runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here and the numbers land in diagnostics.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &baseline)?;
    if quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.provider
        ));
    }

    let lifecycle_blocker = concat!(
        "this arm executes the measured render plus two unscoped warm repeats on the loaded ",
        "provider; it opens no memory-strategy request scope and injects no calibration fault, so ",
        "the scoped cancellation and authorized-error scenarios and their recovery renders are ",
        "unexecuted. This record claims nothing about them"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan: a receipt may
        // only testify to the materialization shape its own run used (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": artifact.json(tier),
        "sweep": complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed", "reason": "two warm repeats on the loaded provider reproduced the measured clip frame-for-frame inside the declared envelope, within the clean warm peak and cleanup bounds" },
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
            "contract": "identical artifact, prompt, seed, geometry, frames, fps, steps, carrier, tier and loaded provider contract; cold measured clip versus two warm unscoped repeats, compared over every frame",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": MAX_THRESHOLD,
            "meanErrorThreshold": MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.loadability_fingerprint(tier),
        },
        "output": {
            "frames": geometry.frames,
            "fps": output_fps,
            "referenceCount": arm.carrier.reference_count(),
            "firstFrameNondegenerate": true,
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-video", arm.slug),
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("predictedOverallCeiling", "bytes", predicted),
                ("stagedArtifactBytes", "bytes", staged_bytes),
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("renderedFrames", "count", u64::from(geometry.frames)),
                ("renderedFps", "count", u64::from(output_fps)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LoadSpec::quantize` follows each route's OWN MLX convention, and the two differ.
    ///
    /// The three Wan routes reconcile a requested tier against the staged snapshot's `config.json`
    /// marker, so naming it is an assertion about the directory. SCAIL-2 refuses it outright on a
    /// canonical-tier load — `ArtifactReceipt::capture` rejects "a second on-load quantization" and
    /// `production_calibration_identity` withholds the identity whenever it is set — so a capture
    /// that set it would not mislabel the cell, it would fail to load it at all, or load it with no
    /// calibration identity for the arm to check against the plan.
    ///
    /// Asserted through the same `route` discriminator the receipt pre-pass keys on, so the two
    /// cannot disagree about which convention a cell is on.
    ///
    /// Mutation that fails this: return `quant` unconditionally from [`numeric_quant`].
    #[test]
    fn only_the_wan_routes_name_a_tier_in_the_load_spec() {
        for arm in ARMS {
            for (tier, quant) in [
                ("bf16", None),
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
            ] {
                let expected = if arm.route.is_some() { quant } else { None };
                assert_eq!(
                    numeric_quant(arm, tier).unwrap(),
                    expected,
                    "{} {tier}",
                    arm.provider
                );
            }
            assert!(numeric_quant(arm, "nvfp4").is_err(), "{}", arm.provider);
        }
        // Not vacuous on either side: this family really does carry both conventions.
        assert!(ARMS.iter().any(|arm| arm.route.is_some()));
        assert!(ARMS.iter().any(|arm| arm.route.is_none()));
    }

    /// The four arms' geometries and cadences are the ENGINE's, asked of the engine.
    ///
    /// Mutation that fails this: giving `T2V_A14B` 24 fps (its manifest menu is `fps: [16]`, and
    /// `WanI2vRoute::accepts_rate` refuses every other rate), or moving SCAIL-2 to a bucket outside
    /// `PUBLIC_BUCKETS`.
    #[test]
    fn every_arm_plans_a_geometry_its_own_engine_admits() {
        for arm in ARMS {
            let geometry = match arm.provider {
                "wan2_2_ti2v_5b" => Geometry {
                    width: 832,
                    height: 480,
                    frames: 121,
                },
                "scail2_14b" => Geometry {
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
            // ...and a rate the route refuses is refused here, so the check is not vacuous.
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

    /// Every (arm, tier) names a distinct production identity, and none of them is a weights-free
    /// conformance string.
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
                    !seen.contains(&fingerprint),
                    "{fingerprint} names more than one cell"
                );
                seen.push(fingerprint);
            }
        }
        assert_eq!(seen.len(), 12, "four arms x three shipped tiers");
    }

    /// The env family names are the ones `measure-memory-catalog.mjs` exports for the MLX lane, and
    /// each names its own repository.
    #[test]
    fn every_arm_binds_its_own_artifact_family() {
        let mut repositories = Vec::new();
        for arm in ARMS {
            for name in [arm.repository_env, arm.revision_env, arm.root_env] {
                assert!(
                    name.starts_with("SCENEWORKS_"),
                    "{name} is not a SceneWorks env family member"
                );
            }
            assert!(
                !repositories.contains(&arm.repository),
                "{} shares an artifact family with another arm",
                arm.provider
            );
            repositories.push(arm.repository);
        }
    }

    /// The synthetic carriers are exactly the shapes the two engines' own request validators
    /// require — one full-strength Reference for I2V, and ordered Reference + Mask + ControlClip
    /// with one driving mask per driving frame for SCAIL-2.
    #[test]
    fn the_synthetic_carriers_match_the_engines_request_contracts() {
        let geometry = Geometry {
            width: 832,
            height: 480,
            frames: 77,
        };
        let text_only = generation_request(T2V_A14B, geometry);
        assert!(text_only.conditioning.is_empty());

        let i2v = generation_request(I2V_A14B, geometry);
        assert!(matches!(
            i2v.conditioning.as_slice(),
            [Conditioning::Reference { strength: None, .. }]
        ));

        let animation = generation_request(SCAIL2, geometry);
        let carrier = animation
            .scail2_animation_conditioning()
            .expect("the synthetic SCAIL-2 carrier satisfies the engine's own validator");
        assert_eq!(carrier.driving_frames.len(), geometry.frames as usize);
        assert_eq!(carrier.driving_masks.len(), carrier.driving_frames.len());
        for plane in [carrier.character, carrier.character_mask] {
            assert_eq!(
                (plane.width, plane.height),
                (geometry.width, geometry.height)
            );
            assert_eq!(
                plane.pixels.len(),
                (geometry.width as usize) * (geometry.height as usize) * 3
            );
        }
    }
}
