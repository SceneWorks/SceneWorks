//! Physical MLX capture arm for the Wan 2.2 family and SCAIL-2 (sc-22736, epic sc-22723 E1).
//!
//! FOUR engine providers, ONE arm, because the four differ only in coordinates a table can hold:
//! the artifact family, the public request carrier, the rate menu and the production identity. The
//! measured path is identical — resolve the artifact, seal a receipt only where the engine publishes
//! the identity from one, load through `runtime_macos::catalog().media()` (the seam
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

/// Determinism envelope these clips are judged by.
///
/// The same claim the other video arms make — repeat determinism on ONE loaded provider with an
/// identical request — so the same published FLUX.2 envelope applies rather than a looser bound
/// invented here. The mandatory `+64` negative mutation is more than eight times the maximum.
/// Since sc-22738 (2026-09-08) a video capture is ONE measured render, so no warm repeat is
/// compared against it here: the thresholds travel in the receipt (`quality.warmPasses: 0`,
/// `result: not_run`) and bound only the falsifiability mutation.
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
    repository: protocol::WAN22_TI2V_5B_MLX_REPOSITORY,
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
    repository: protocol::WAN22_T2V_A14B_MLX_REPOSITORY,
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
    repository: protocol::WAN22_I2V_A14B_MLX_REPOSITORY,
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
    repository: protocol::SCAIL2_REPOSITORY,
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
    Ok(match arm.provider {
        // sc-19236's own per-tier table (`mlx-gen-wan/src/memory_strategy.rs`), which names the
        // packing group and the Q8 text-encoder floor in the packed cells.
        WAN_TI2V_5B_PROVIDER => match tier {
            "bf16" => "sc-19236-wan2-2-ti2v-5b-mlx-dense-v1".to_owned(),
            "q4" => "sc-19236-wan2-2-ti2v-5b-mlx-q4-g64-teq8-v1".to_owned(),
            "q8" => "sc-19236-wan2-2-ti2v-5b-mlx-q8-g64-teq8-v1".to_owned(),
            other => return Err(format!("{LABEL}: unsupported TI2V-5B tier {other:?}")),
        },
        // sc-22736's shared A14B authority (`gen_core::wan_i2v_memory`), whose dense token is
        // `dense`. The tier is matched by name, never interpolated, so an unrecognized tier is a
        // refusal rather than a plausible-looking identity no engine publishes.
        WAN_T2V_A14B_PROVIDER | WAN_I2V_A14B_PROVIDER => {
            let route = if arm.provider == WAN_T2V_A14B_PROVIDER {
                "t2v"
            } else {
                "i2v"
            };
            let token = match tier {
                "bf16" => "dense",
                "q4" | "q8" => tier,
                other => return Err(format!("{LABEL}: unsupported A14B tier {other:?}")),
            };
            format!("sc-22736-wan2-2-{route}-a14b-mlx-{token}-v1")
        }
        // sc-22736's SCAIL-2 table, whose tier token is the DIRECTORY name, not `dense`.
        SCAIL2_PROVIDER => match tier {
            "bf16" | "q4" | "q8" => format!("scail2-14b-{tier}-mlx-resident-eager-v1"),
            other => return Err(format!("{LABEL}: unsupported SCAIL-2 tier {other:?}")),
        },
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

/// How many frames this arm's engine RENDERS for its planned request (sc-22738).
///
/// Not always `geometry.frames`: the shared rule [`protocol::vae_decoded_frame_count`] is applied
/// to THIS arm's own `VaeTiling`, so the non-causal z16 routes (SCAIL-2 and both A14B experts)
/// answer `t_lat · 4` — 80 for a 77-frame request — while the causal z48 TI2V-5B answers the
/// requested count. The plan keeps asking for the count production asks for (`wan_frame_count`
/// leaves 77 at 77, and nothing on the worker's path trims the engine's longer clip), and the
/// record's `renderedFrames` receipt reports what actually came back.
fn rendered_frame_count(arm: Arm, geometry: Geometry) -> Result<u32, String> {
    let vae = engine_vae(arm)?;
    protocol::vae_decoded_frame_count(geometry.frames, vae.temporal_scale, vae.causal_temporal)
        .map_err(|error| format!("{}: {error}", arm.provider))
}

/// The concrete VAE geometry this arm's engine decodes through, resolved BY PROVIDER ID from the
/// engine's own registry-facing resolver rather than tabled here — the same way this file asks the
/// engine for its buckets and rates. It carries the two facts the decoded-frame rule needs: the
/// temporal scale and whether the temporal decode is causal.
fn engine_vae(arm: Arm) -> Result<VaeTiling, String> {
    mlx_gen_wan::vae_tiling(arm.provider)
        .or_else(|| mlx_gen_scail2::vae_tiling(arm.provider))
        .ok_or_else(|| {
            format!(
                "{} publishes no VAE geometry; its decoded frame count cannot be derived",
                arm.provider
            )
        })
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

/// Whether this cell's production identity is minted by the SEALED-RECEIPT authority
/// (`gen_core::wan_i2v_memory`), so the capture must seal a receipt before the load to read it —
/// or by the provider's own `memory_strategy` module, which the loader publishes only for an
/// unprepared spec, so sealing one would HIDE it (sc-22738).
///
/// Asked of the engine rather than spelled per route: `production_calibration_fingerprint`
/// (`gen-core/src/wan_i2v_memory.rs`) is the receipt authority's own answer, and it returns `None`
/// for `wan2_2_ti2v_5b` by design — that route's `sc-19236-…` identity belongs to
/// `mlx-gen-wan/src/memory_strategy.rs`, and `Wan::memory_strategy_contract` (`model.rs`) returns
/// the receipt's contract, calibration and all, ahead of that one whenever the spec was prepared.
/// The receipt-published identity must be the plan's: a receipt naming any other string is a drift
/// between this table and the engine, refused by name before a byte is opened.
fn receipt_publishes_the_identity(
    arm: Arm,
    tier: &str,
    expected_fingerprint: &str,
) -> Result<bool, String> {
    let Some(route) = arm.route else {
        return Ok(false);
    };
    let numeric_tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: numeric_quant(arm, tier)?,
        component_precision_floors: &[],
    };
    match mlx_gen::gen_core::wan_i2v_memory::production_calibration_fingerprint(
        route,
        mlx_gen::gen_core::wan_i2v_memory::WanI2vBackend::Mlx,
        numeric_tier,
    ) {
        None => Ok(false),
        Some(published) if published == expected_fingerprint => Ok(true),
        Some(published) => Err(format!(
            "the pinned receipt authority mints {published} for the {} {tier} cell, not this \
             arm's {expected_fingerprint}; the table and the engine have drifted",
            arm.provider
        )),
    }
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

/// The evidence identity the WORKER puts on the run context it offers a video provider, when no
/// fitted curve priced the candidate.
///
/// `crates/sceneworks-worker/src/video_admission.rs:1716-1720` — the selected candidate carries the
/// packaged curve's id, else the resolved decode profile's revision, else this token.
const WORKER_ESTIMATE_FLOOR_EVIDENCE: &str = "video-estimate-floor-v1";

/// The two decode-profile revisions the same site can carry instead
/// (`video_admission.rs:801`, `:819`, `:835`, read back at `:1152`).
const WORKER_DECODE_PROFILE_EVIDENCE: [&str; 2] = [
    "video-provider-selected-decode-profile-v1",
    "video-provider-conservative-decode-profile-v1",
];

/// The receipt token SCAIL-2's gate requires the context's `evidence_revision` to open with.
///
/// A MIRROR, not a read: `mlx-gen-scail2`'s `validate_context_revision_shape`
/// (`memory_strategy.rs:570-588` at inference `3b922bac`) compares this as a bare literal and the
/// crate publishes no `pub const` for it, unlike the Wan side's
/// `gen_core::wan_i2v_memory::RECEIPT_VERSION`. `scripts/measure-memory-catalog.test.mjs` binds
/// both tokens to the pinned engine source so a rename reds there rather than silently widening
/// [`probes_admission`].
const SCAIL2_RECEIPT_TOKEN: &str = "scail2-resident-v1";

/// The receipt token THIS arm's engine gate requires, asked of the engine where it publishes one.
fn engine_receipt_token(arm: Arm) -> &'static str {
    match arm.route {
        Some(_) => mlx_gen::gen_core::wan_i2v_memory::RECEIPT_VERSION,
        None => SCAIL2_RECEIPT_TOKEN,
    }
}

/// Every evidence identity the worker's video admission can put on a `MemoryRunContext` for this
/// arm's provider: the estimate floor, the two decode-profile revisions, and any packaged curve
/// promoted for this provider. The curve list is READ from the shipped bundle rather than assumed
/// empty, so a future promoted Wan or SCAIL-2 curve enters this vocabulary automatically.
fn worker_context_evidence_identities(arm: Arm) -> Vec<String> {
    let mut identities = vec![WORKER_ESTIMATE_FLOOR_EVIDENCE.to_owned()];
    identities.extend(
        WORKER_DECODE_PROFILE_EVIDENCE
            .iter()
            .map(|identity| (*identity).to_owned()),
    );
    if let Some(bundle) = sceneworks_core::video_memory_curves::packaged_video_memory_curves() {
        identities.extend(
            bundle
                .curves
                .iter()
                .filter(|curve| curve.provider == arm.provider)
                .map(|curve| curve.id.clone()),
        );
    }
    identities
}

/// Whether `identity` is one this arm's engine gate can read as its OWN sealed request receipt.
///
/// Both gates parse the context's `evidence_revision` as a colon-delimited receipt whose first
/// segment is the engine's own version token — `wan_i2v_memory::validate_context` through
/// `receipt_rate_and_tail` (`wan_i2v_memory.rs:2882-2895`, checked at `:2944`), SCAIL-2 through
/// `validate_context_revision_shape` (`memory_strategy.rs:570-588`) — and refuse everything else
/// before any budget is compared.
fn engine_seals_evidence(arm: Arm, identity: &str) -> bool {
    identity.starts_with(&format!("{}:", engine_receipt_token(arm)))
}

/// Whether production would obtain an admission decision from this provider's gate for the request
/// this arm renders. The twin of `mlx.rs#bernini_probes_admission` and
/// `mlx.rs#krea_realtime_probes_admission`, and false for a structurally similar reason.
///
/// THE ENGINE SIDE. Both gates bind admission to a receipt the ENGINE seals over the artifact and
/// the exact request bytes. `gen_core::wan_i2v_memory::validate_context`
/// (`crates/contracts/gen-core/src/wan_i2v_memory.rs:2897-3040` at inference `3b922bac`) refuses
/// with `crossed Wan I2V memory context` unless `context.evidence_revision` parses as
/// `wan-video-structural-v5:<mode>:fps<N>:<artifact_identity>:<selection_receipt>:…` (`:2944`,
/// `:3024`) AND `context.overlay` is exactly the sealed `wan-adapters-v1:<sha256>` adapter identity
/// (`:3021`, minted at `:1726`). `mlx-gen-scail2`'s `validate_context_identity`
/// (`crates/media/mlx-gen/mlx-gen-scail2/src/memory_strategy.rs:660-666`) requires
/// `scail2-resident-v1:<receipt_sha256>:<64 hex>:<64 hex>`.
///
/// THE WORKER SIDE mints neither, and cannot: SceneWorks never links either engine's receipt
/// helper. `video_admission.rs:2244-2268` builds the run context with `overlay: request.overlay` —
/// the worker's own descriptive spelling, `provider_video_mode:<mode>` plus any adapter/enhancer
/// axis (`video_jobs/wan.rs#video_admission_overlay`, `:1557-1874`) — and with
/// `evidence_revision: selected.evidence_revision`, which is a packaged curve id, a decode-profile
/// revision or `video-estimate-floor-v1` (`:1716-1720`). `engine_declines_advisory_context`
/// (`video_admission.rs:1822-1830`, called at `:2277`) therefore sees the engine's refusal on every
/// such request and returns `memory: None, context: None`, and
/// `video_jobs/wan.rs#apply_video_admission_outcome` (`:2088-2096`) writes BOTH onto the input, so
/// the render reaches the engine through `memory_strategy::generate_with_scope`'s no-context early
/// return on its load-time defaults, carrying no request memory at all.
///
/// sc-22738 probed that surface anyway and `wan_2_2_i2v_14b:bf16:mlx` died on the FIRST probe after
/// a 383-second load with `admission rejected a fitting probe budget`. The remedy is not a context
/// shaped to make the gate answer — that would characterize a decision the product never takes —
/// it is to skip the probe exactly where production skips the context, and to render the same
/// carrier-free request production renders.
///
/// Computed, not asserted: it asks whether ANY evidence identity the worker can carry today is one
/// this engine seals. A pin or a promoted curve that made one so flips this to `true` and the arm
/// refuses the capture by name rather than quietly recording scenarios production now runs.
fn probes_admission(arm: Arm) -> bool {
    worker_context_evidence_identities(arm)
        .iter()
        .any(|identity| engine_seals_evidence(arm, identity))
}

/// What the record says about the admission scenarios it did not run, and why. Stated once and
/// carried into every `not_run` reason, the lifecycle blocker and the diagnostics.
const ADMISSION_BLOCKER: &str = concat!(
    "production never presents these Wan 2.2 / SCAIL-2 providers a memory run context their own ",
    "gate accepts, so this capture has no admission decision to characterize. Both gates require ",
    "the ENGINE's sealed request receipt — gen-core's wan_i2v_memory::validate_context parses ",
    "evidence_revision as wan-video-structural-v5:<mode>:fps<N>:<artifact>:<selection>: and ",
    "requires the overlay to be the sealed wan-adapters-v1:<sha256> adapter identity, and ",
    "mlx-gen-scail2 requires scail2-resident-v1:<receipt>:<sha256>:<sha256> — and SceneWorks mints ",
    "neither: the worker's video admission carries a packaged curve id, a decode-profile revision ",
    "or video-estimate-floor-v1 as the evidence identity, and the descriptive ",
    "provider_video_mode:<mode> overlay spelling. engine_declines_advisory_context therefore sees ",
    "the refusal on every such request and drops BOTH the run context and the rung's request ",
    "memory carrier, so the render reaches the engine on its load-time defaults through ",
    "generate_with_scope's no-context early return. This arm asks the gate nothing, exactly as ",
    "production asks it nothing, and renders the same carrier-free request production renders: the ",
    "exact-fit, unknown-budget and stale-evidence scenarios are unexecuted rather than answered on ",
    "a surface production never presents. This anchor prices the RESIDENT load of that request and ",
    "claims nothing about any admission decision"
);

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
/// What this record claims nothing about, spelled per arm.
///
/// The A14B routes are measured with NO adapters at their native multi-step CFG recipe, while the
/// worker runs them with the Lightning distill DEFAULT-ON (`video_jobs/wan.rs`
/// `wan_lightning_on`, sc-10047): the record names that exclusion so it cannot be read as the
/// default composition's evidence.
fn lifecycle_blocker(arm: Arm) -> String {
    let mut blocker = format!(
        "this arm executes ONE measured render on the loaded provider and no warm pass ({}); it \
         opens no memory-strategy request scope and injects no calibration fault, so the scoped \
         cancellation and authorized-error scenarios and their recovery renders are unexecuted. \
         This record claims nothing about them",
        protocol::VIDEO_WARM_PASSES_NOT_RUN
    );
    if matches!(arm.route, Some(WanI2vRoute::T2v14b | WanI2vRoute::I2v14b)) {
        blocker.push_str(&format!(
            ". The A14B render is measured with NO adapters at the native multi-step recipe \
             (steps: {}): the Lightning distill the worker attaches DEFAULT-ON is OFF here, so \
             this record does not price the default Lightning composition",
            arm.steps
        ));
    }
    // sc-22738: stated in the same breath as the lifecycle exclusion because it is the same kind of
    // claim — what this record deliberately does NOT say.
    blocker.push_str(&format!(". Additionally: {ADMISSION_BLOCKER}"));
    blocker
}

pub(super) fn run(request: &Value) -> Result<Value, String> {
    // Everything cheap and refusable first, so a mis-planned row costs milliseconds rather than a
    // multi-gigabyte load: the member, the mode, the geometry against the ENGINE's own menus, the
    // fixture, the tier, and the plan's identity against this arm's table.
    let arm = arm(request)?;
    // The lane's render plan (sc-22738): a video capture is ONE measured render, no warm pass.
    let capture = protocol::capture_policy(request)?.require_video(arm.provider)?;
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
    // The arm's admission decision, taken the way the WORKER takes it (sc-22738): see
    // [`probes_admission`]. On every cell this arm can plan the answer is `false` — neither engine's
    // gate can read any evidence identity SceneWorks mints — so no probe context is built, nothing
    // is asked of the provider's safety check, and the render below runs on the provider's load-time
    // defaults exactly as production runs it. The record says so rather than claiming an admission
    // result it never obtained. A `true` answer means the seam moved: refuse the capture by name
    // and re-derive the probe context from `video_admission.rs` before recording anything.
    if probes_admission(arm) {
        return Err(format!(
            "{}: the worker's video admission can now carry an evidence identity this engine's \
             gate seals, so production DOES take an admission decision for this request; re-derive \
             the probe context from video_admission.rs before capturing this cell",
            arm.provider
        ));
    }

    let mut artifact = load_spec(arm, tier, load_shape)?;
    // The A14B loaders publish a memory contract ONLY from a sealed receipt (`model.rs`: the
    // `i2v_memory` prepared for a spec whose file pins are already prepared), so on those routes
    // the capture seals one before the load — otherwise the generator loads with no contract at
    // all. TI2V-5B is the opposite (sc-22738): its production identity is its own
    // `memory_strategy` module's, which `load` publishes only for an UNPREPARED spec — a prepared
    // one takes the receipt path instead, and the receipt authority mints no identity for that
    // route. Production (`video_jobs/wan.rs::video_load_spec`) prepares nothing, so neither does
    // this cell. SCAIL-2 seals its own shared-tier pins inside `PreparedMemory::prepare`.
    if receipt_publishes_the_identity(arm, tier, &expected_fingerprint)? {
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

    // The request production renders, and NOTHING is installed on it (sc-22738). Production's
    // admission drops the rung's request memory carrier along with the run context the engine
    // declines (`video_admission.rs#engine_declines_advisory_context` →
    // `video_jobs/wan.rs#apply_video_admission_outcome`), so a Wan render reaches
    // `mlx-gen-wan/src/model.rs`'s `validate_active_request` with `request.memory == None` and is
    // waved through onto the provider's load-time defaults. A capture that installed the carrier
    // would present a memory-managed request no shipped path presents — and one that engine would
    // refuse outright, because no admitted scope armed its active evidence.
    let planned_render = generation_request(arm, geometry);

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
    // The engine's own decoded depth for this request, NOT the requested count: a non-causal z16
    // decode materializes four output frames per latent frame, so a 77-frame SCAIL-2 / A14B request
    // renders 80 (sc-22738). Production asks for the same count and keeps the clip it gets, so the
    // adapter measures the same path instead of refusing it.
    let expected_frames = rendered_frame_count(arm, geometry)?;
    if measured.len() as u64 != u64::from(expected_frames) {
        return Err(format!(
            "{} rendered {} frames for a {}-frame request; its {} VAE decodes that request \
             to {expected_frames} frames",
            arm.provider,
            measured.len(),
            geometry.frames,
            if engine_vae(arm)?.causal_temporal {
                "causal"
            } else {
                "non-causal"
            }
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
    // No exact-fit probe against the MEASURED evidence either, and for the same reason the three
    // pre-render scenarios are unexecuted: the gate this arm would ask cannot read any evidence
    // identity SceneWorks mints, so an answer here would characterize a decision production never
    // takes. See [`ADMISSION_BLOCKER`].

    // No warm pass: the video lane captures ONE measured render (sc-22738, `capture` above), so
    // there is no clean warm control to judge determinism against and no warm repeat to bound;
    // the receipt says so (`warm_repeat` not_run, `quality.warmPasses: 0`) instead of writing
    // zeros where `lifecycleClean*` / `lifecycleWarmRepeat*` used to be.

    // Arm-internal negative-mutation falsifiability check: a runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here and the numbers land in diagnostics.
    // Against the measured clip itself: the mutation must breach the envelope the arm declares.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &measured)?;
    if quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.provider
        ));
    }

    let lifecycle_blocker = lifecycle_blocker(arm);
    let lifecycle_blocker = lifecycle_blocker.as_str();
    // GATED, not `runtime_complete` (sc-22738). Runtime activation is exactly the claim that the
    // provider's admission gate accepted an exact-fit budget and rejected the two mutations, and
    // this capture asked it nothing — because production asks it nothing for this request. The
    // MEASUREMENT is unaffected and is carried in full: `observedMemory` and `predictedPeakBytes`
    // are the resident-load prices this anchor exists to publish, and `extract-memory-anchors.mjs`
    // reads them off this record exactly as it reads them off a runtime-complete one.
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan: a receipt may
        // only testify to the materialization shape its own run used (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": artifact.json(tier),
        "sweep": complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": ADMISSION_BLOCKER },
            { "name": "unknown_budget", "result": "not_run", "reason": ADMISSION_BLOCKER },
            { "name": "stale_evidence", "result": "not_run", "reason": ADMISSION_BLOCKER },
            capture.not_run_warm_repeat_scenario()?,
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
        "quality": capture.not_run_quality(
            "identical artifact, prompt, seed, geometry, frames, fps, steps, carrier, tier and loaded provider contract; the cold measured clip versus a warm repeat, compared over every frame",
            (MAX_THRESHOLD, MEAN_THRESHOLD, RMS_THRESHOLD),
        )?,
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.loadability_fingerprint(tier),
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
                // sc-22738: no `lifecycleClean*` / `lifecycleWarmRepeat*` figure — the warm
                // passes were not run (`quality.warmPasses`), and an unmeasured figure is omitted,
                // never written as 0.
                ("warmPasses", "count", u64::from(capture.warm_passes)),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("renderedFrames", "count", u64::from(expected_frames)),
                ("requestedFrames", "count", u64::from(geometry.frames)),
                ("renderedFps", "count", u64::from(output_fps)),
                // sc-22738. The carrier's reference count was the one fact the dropped top-level
                // `output` object published that these measurements did not: the record schema is
                // `additionalProperties: false`, so the object made the whole bundle unschedulable
                // after the render. It is a measurement now, beside the frames and fps receipts.
                ("referenceCount", "count", u64::from(arm.carrier.reference_count())),
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

    /// Only the A14B routes seal a receipt before the load (sc-22738).
    ///
    /// `wan_2_2:{bf16,q4,q8}:mlx` were refused at e34d7b46a with "published no calibration
    /// identity": the arm sealed a receipt for every Wan route, and for TI2V-5B the loader then
    /// published the receipt's identity-less contract ahead of its own `sc-19236-…` one. The
    /// discriminator is the engine's receipt authority, asked directly, so the A14B cells keep
    /// their pre-pass and the two unprepared arms match production's own load.
    ///
    /// Mutations that fail this: guarding the pre-pass on `arm.route.is_some()` again (the
    /// TI2V-5B rows flip), or dropping the drift refusal (the last assertion).
    #[test]
    fn only_the_a14b_routes_seal_a_receipt_before_the_load() {
        for arm in ARMS {
            for tier in ["bf16", "q4", "q8"] {
                let expected = production_fingerprint(arm, tier).unwrap();
                let seals = receipt_publishes_the_identity(arm, tier, &expected).unwrap();
                let a14b = matches!(arm.route, Some(WanI2vRoute::T2v14b | WanI2vRoute::I2v14b));
                assert_eq!(seals, a14b, "{} {tier}", arm.provider);
            }
        }
        // Not vacuous: the table really carries both kinds of route.
        assert!(matches!(TI2V_5B.route, Some(WanI2vRoute::Ti2v5b)));
        assert!(SCAIL2.route.is_none());
        for tier in ["bf16", "q4", "q8"] {
            let expected = production_fingerprint(TI2V_5B, tier).unwrap();
            assert!(expected.starts_with("sc-19236-wan2-2-ti2v-5b-mlx-"));
            assert_eq!(
                receipt_publishes_the_identity(TI2V_5B, tier, &expected),
                Ok(false),
                "{tier}"
            );
        }
        let drift =
            receipt_publishes_the_identity(T2V_A14B, "q4", "not-the-engines-string").unwrap_err();
        assert!(
            drift.contains("sc-22736-wan2-2-t2v-a14b-mlx-q4-v1") && drift.contains("drifted"),
            "{drift}"
        );
    }

    /// The pre-pass in the arm's body is guarded by the engine-asked discriminator and by nothing
    /// looser; read as source because the load itself needs real weights (sc-22738).
    #[test]
    fn the_receipt_prepass_is_guarded_by_the_receipt_authority() {
        let body = arm_source_body();
        let guard = body
            .find("if receipt_publishes_the_identity(arm, tier, &expected_fingerprint)? {")
            .expect("the pre-pass is guarded by the receipt authority");
        let prepass = body
            .find("mlx_gen_wan::i2v_memory_strategy::prepare_load_spec(&mut artifact.spec")
            .expect("the pre-pass still exists");
        assert!(guard < prepass, "the guard precedes the pre-pass");
        assert!(
            !body.contains("if arm.route.is_some() {\n        mlx_gen_wan::i2v_memory_strategy"),
            "the pre-pass must not run for every Wan route"
        );
    }

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
                WAN_TI2V_5B_PROVIDER => Geometry {
                    width: 832,
                    height: 480,
                    frames: 121,
                },
                SCAIL2_PROVIDER => Geometry {
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

    /// sc-22738 — the rendered frame count is the ENGINE's decoded depth, not the requested count.
    ///
    /// The campaign's `scail2_14b:bf16:mlx` anchor rendered 77 requested frames as 80 and the arm
    /// refused the capture after 2h25m. The engine is right: `VaeTiling::WAN` is NON-causal, so the
    /// z16 decode materializes `t_lat * temporal_scale` output frames, and 77 frames are 20 latent
    /// frames. This binds the arm's rule to the engines' OWN functions rather than to a second
    /// literal — gen-core's `TilingConfig::plan` publishes the same `out_f`, and `latent_shape` the
    /// same latent depth — so a pin that changed either would red here in milliseconds instead of
    /// after a multi-hour render.
    ///
    /// Mutations that fail this: flipping the causal branch in
    /// `protocol::vae_decoded_frame_count`; using `requested` as the latent depth.
    #[test]
    fn the_rendered_frame_rule_is_the_engines_own_decoded_depth() {
        for arm in ARMS {
            let vae = engine_vae(arm).unwrap_or_else(|error| panic!("{error}"));
            for geometry in [
                Geometry {
                    width: 832,
                    height: 480,
                    frames: 77,
                },
                Geometry {
                    width: 832,
                    height: 480,
                    frames: 121,
                },
            ] {
                let latent = mlx_gen_wan::pipeline::latent_shape(
                    geometry.frames as usize,
                    geometry.height,
                    geometry.width,
                    1,
                    (
                        vae.temporal_scale as usize,
                        vae.spatial_scale as usize,
                        vae.spatial_scale as usize,
                    ),
                )
                .expect("the wan latent rule accepts a planned frame count");
                let engine_out_f = TilingConfig {
                    spatial: None,
                    temporal: None,
                }
                .plan(vae, latent[1], latent[2], latent[3])
                .out_f;
                assert_eq!(
                    i64::from(rendered_frame_count(arm, geometry).unwrap()),
                    i64::from(engine_out_f),
                    "{} at {} frames",
                    arm.provider,
                    geometry.frames
                );
            }
        }
    }

    /// The concrete sc-22738 consequence, per arm, so the rule cannot silently become an identity.
    ///
    /// SCAIL-2 and both A14B experts decode through the non-causal z16 VAE and over-deliver by one
    /// temporal stride minus one; the causal z48 TI2V-5B returns exactly what was asked for. The
    /// worker asks for the same counts (`wan_frame_count` leaves a `4k + 1` request alone) and keeps
    /// the clip the engine returns, so the adapter must accept the same clip.
    ///
    /// Mutation that fails this: returning `geometry.frames` from [`rendered_frame_count`] — the
    /// equality that refused the campaign's SCAIL-2 anchor — or resolving every arm to the causal
    /// z48 geometry.
    #[test]
    fn the_non_causal_routes_render_three_frames_more_than_requested() {
        let geometry = Geometry {
            width: 832,
            height: 480,
            frames: 77,
        };
        for arm in [SCAIL2, T2V_A14B, I2V_A14B] {
            assert!(
                !engine_vae(arm).unwrap().causal_temporal,
                "{}",
                arm.provider
            );
            assert_eq!(
                rendered_frame_count(arm, geometry).unwrap(),
                geometry.frames + 3,
                "{}",
                arm.provider
            );
        }
        assert!(engine_vae(TI2V_5B).unwrap().causal_temporal);
        assert_eq!(
            rendered_frame_count(TI2V_5B, geometry).unwrap(),
            geometry.frames
        );
    }

    /// An unrecognized tier is refused BY NAME on every arm, never minted into a plausible identity
    /// (sc-22736 review). Mutation that reds this: interpolating `{tier}` into the A14B or SCAIL-2
    /// string without the match.
    #[test]
    fn an_unrecognized_tier_is_refused_by_name_on_every_arm() {
        for arm in ARMS {
            for tier in ["nvfp4", "fp8", "dense", ""] {
                let error = production_fingerprint(arm, tier)
                    .err()
                    .unwrap_or_else(|| panic!("{} must refuse tier {tier:?}", arm.provider));
                assert!(
                    error.contains(&format!("{tier:?}")),
                    "{}: the refusal must name the tier: {error}",
                    arm.provider
                );
                assert!(
                    !error.contains("-v1"),
                    "{}: minted an identity: {error}",
                    arm.provider
                );
            }
        }
    }

    /// The two A14B records name the Lightning exclusion — they are measured with no adapters at
    /// the native 40-step recipe while the worker attaches the distill DEFAULT-ON — and the other
    /// two, which have no such default, do not (sc-22736 review).
    #[test]
    fn the_a14b_blockers_name_the_lightning_off_measurement() {
        for arm in [T2V_A14B, I2V_A14B] {
            let blocker = lifecycle_blocker(arm);
            assert!(blocker.contains("Lightning"), "{}: {blocker}", arm.provider);
            assert!(
                blocker.contains("DEFAULT-ON"),
                "{}: {blocker}",
                arm.provider
            );
            assert!(
                blocker.contains("NO adapters"),
                "{}: {blocker}",
                arm.provider
            );
            assert!(
                blocker.contains(&format!("steps: {}", arm.steps)),
                "{}: {blocker}",
                arm.provider
            );
        }
        for arm in [TI2V_5B, SCAIL2] {
            let blocker = lifecycle_blocker(arm);
            assert!(
                !blocker.contains("Lightning"),
                "{}: {blocker}",
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

    /// A weights-free A14B contract from the pinned engine's own authority, for the request-shape
    /// assertions below. `weights_free_contract` serves exactly the two A14B routes and touches no
    /// filesystem, so these run on any host.
    fn weights_free_a14b_contract(arm: Arm) -> mlx_gen::gen_core::MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/nonexistent/wan-a14b")));
        mlx_gen::gen_core::wan_i2v_memory::weights_free_contract(
            arm.provider,
            mlx_gen::gen_core::wan_i2v_memory::WanI2vBackend::Mlx,
            &spec,
        )
        .unwrap_or_else(|error| panic!("{}: {error}", arm.provider))
    }

    /// The planned render is CARRIER-FREE, exactly as production's declined admission leaves it
    /// (sc-22738).
    ///
    /// `video_admission.rs#engine_declines_advisory_context` returns `memory: None, context: None`
    /// for every request these gates refuse, and `video_jobs/wan.rs#apply_video_admission_outcome`
    /// writes BOTH onto the input — so the shipped render reaches `mlx-gen-wan/src/model.rs`'s
    /// `validate_active_request` with `request.memory == None` and is waved through onto the
    /// provider's load-time defaults. sc-22738 briefly installed the carrier here to satisfy a
    /// receipt mint for probes production never runs; that made the capture render a memory-managed
    /// request no shipped path presents, and one the engine refuses outright because no admitted
    /// scope ever armed its active evidence.
    ///
    /// Non-vacuous on the Wan side: the contract really would hand out a carrier at every rung
    /// (`ExplicitResident`), so the absent carrier is a decision, not an accident of the default.
    ///
    /// MUTATION that reds this: re-adding an `install_memory_carrier`-shaped write of
    /// `request.memory` in `generation_request` or in `run`.
    #[test]
    fn the_planned_render_is_carrier_free_as_productions_declined_admission_leaves_it() {
        let geometry = Geometry {
            width: 832,
            height: 480,
            frames: 77,
        };
        for arm in ARMS {
            assert!(
                generation_request(arm, geometry).memory.is_none(),
                "{}: production renders this request carrier-free; a capture that installs one \
                 measures a memory-managed path no shipped route presents",
                arm.provider
            );
        }
        for arm in [T2V_A14B, I2V_A14B] {
            let contract = weights_free_a14b_contract(arm);
            for strategy in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ] {
                let selection = MemorySelection {
                    strategy,
                    parameters: MemoryStrategyParameters::default(),
                    tier: MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    },
                };
                assert!(
                    contract.generation_memory(&selection).is_some(),
                    "{} {strategy:?}: the Wan contract is ExplicitResident, so an admitted rung \
                     WOULD carry controls — the carrier-free request above is production's \
                     declined path, not the contract's default",
                    arm.provider
                );
            }
        }
        let body = arm_source_body();
        assert!(
            !body.contains(".memory = ") && !body.contains("generation_memory("),
            "the arm installs a request memory carrier again — under any variable name, from any \
             source; production's admission drops the carrier along with the run context its \
             engine declines"
        );
    }

    /// The arm's `run` body, read as source so the assertions below cannot be satisfied by a
    /// helper the body no longer calls. The measured path needs real weights, so it is not
    /// executable in a unit test.
    fn arm_source_body() -> &'static str {
        let source = include_str!("mlx_wan_scail2.rs");
        let start = source
            .find("\npub(super) fn run(")
            .expect("the arm still exists");
        &source[start
            ..start
                + source[start..]
                    .find("\n}\n")
                    .expect("the arm's body closes")]
    }

    /// The arm asks the admission gate exactly what PRODUCTION asks it — which, for every cell this
    /// arm can plan, is nothing (sc-22738).
    ///
    /// Both engines bind admission to a receipt they seal themselves, and SceneWorks mints neither:
    /// the worker's video admission carries a packaged curve id, a decode-profile revision or
    /// `video-estimate-floor-v1`. sc-22736 probed the gate anyway and `wan_2_2_i2v_14b:bf16:mlx`
    /// died on the FIRST probe after a 383-second load with "admission rejected a fitting probe
    /// budget". The remedy is not a context shaped to make the gate answer — production never
    /// presents one — it is to skip the probe exactly where production skips the context.
    ///
    /// MUTATIONS that red this: making `probes_admission` a constant `true` or `false` (the
    /// non-vacuity pair below); dropping the estimate-floor or decode-profile identities from the
    /// worker vocabulary; restoring any `safety_check` call or the `runtime_complete` status in the
    /// arm's body; or spelling a scenario's reason as anything but the shared blocker.
    #[test]
    fn the_arm_asks_the_admission_gate_exactly_what_production_asks_it() {
        // The plan's own rows, read rather than restated: every planned MLX cell of this family
        // decides the same way, so none of them can reach a probe.
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut planned_rows = 0;
        for (key, row) in plan["anchors"].as_object().expect("anchors object") {
            let Some(arm) = ARMS
                .into_iter()
                .find(|arm| row["provider"].as_str() == Some(arm.provider))
            else {
                continue;
            };
            if !key.ends_with(":mlx") {
                continue;
            }
            assert!(
                !probes_admission(arm),
                "{key}: production takes no admission decision for this request, so the capture \
                 must not probe one"
            );
            planned_rows += 1;
        }
        assert_eq!(
            planned_rows, 12,
            "the plan must still carry the twelve MLX Wan/SCAIL-2 rows this case decides for"
        );

        // Non-vacuous over the predicate's own axis: the worker's real vocabulary is sealed by
        // NEITHER engine, and an identity that opened with the engine's own receipt token would be.
        for arm in ARMS {
            let token = engine_receipt_token(arm);
            assert!(
                !token.is_empty() && !token.contains(':'),
                "{}: the receipt token is one leading segment",
                arm.provider
            );
            for identity in worker_context_evidence_identities(arm) {
                assert!(
                    !engine_seals_evidence(arm, &identity),
                    "{}: the worker identity {identity:?} is now one this engine seals; the arm \
                     must probe admission again rather than record the scenarios unexecuted",
                    arm.provider
                );
            }
            assert!(
                engine_seals_evidence(arm, &format!("{token}:sealed:by:the:engine")),
                "{}: the seal test never answers yes; it can prove nothing",
                arm.provider
            );
            assert!(
                worker_context_evidence_identities(arm)
                    .contains(&WORKER_ESTIMATE_FLOOR_EVIDENCE.to_owned()),
                "{}: the estimate floor is the identity the worker carries when no curve or \
                 decode profile priced the candidate; it must stay in the vocabulary",
                arm.provider
            );
            for profile in WORKER_DECODE_PROFILE_EVIDENCE {
                assert!(
                    worker_context_evidence_identities(arm).contains(&profile.to_owned()),
                    "{}: {profile} is an identity video_admission.rs can carry",
                    arm.provider
                );
            }
        }

        // …and the arm ACTS on that decision. The predicate is only the answer; a body that still
        // called the provider's admission gate, or a record that still claimed the three admission
        // scenarios passed, would be the sc-22736 defect wearing the sc-22738 predicate.
        let body = arm_source_body();
        assert!(
            !body.contains("safety_check"),
            "the Wan/SCAIL-2 arm asks the provider's admission gate something again; production \
             asks it nothing for these requests"
        );
        assert!(
            body.contains("\"status\": \"gated\""),
            "a capture that ran no admission probe cannot file a runtime-activating record"
        );
        for scenario in ["exact_fit", "unknown_budget", "stale_evidence"] {
            assert!(
                body.contains(&format!(
                    "{{ \"name\": \"{scenario}\", \"result\": \"not_run\", \"reason\": \
                     ADMISSION_BLOCKER }}"
                )),
                "the {scenario} scenario must be reported unexecuted, with the reason production \
                 gives for it"
            );
        }
        assert!(
            ADMISSION_BLOCKER
                .contains("production never presents these Wan 2.2 / SCAIL-2 providers")
                && ADMISSION_BLOCKER.contains("prices the RESIDENT load"),
            "the record must state why it ran no probe and what it does price: {ADMISSION_BLOCKER}"
        );
        // The blocker is carried into the lifecycle exclusion too, so a reader of either field sees
        // the same claim.
        for arm in ARMS {
            assert!(
                lifecycle_blocker(arm).contains(ADMISSION_BLOCKER),
                "{}: the lifecycle blocker must carry the admission blocker",
                arm.provider
            );
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
