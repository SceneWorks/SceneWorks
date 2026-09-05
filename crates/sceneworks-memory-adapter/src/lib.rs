//! Shared, weight-free protocol helpers for the real SC-15508 backend adapters.
//!
//! Backend binaries deliberately return `gated` until every required measurement and lifecycle
//! scenario has actually executed. A successful model call is not silently promoted into a complete
//! calibration record.

use serde_json::{json, Map, Value};
use std::io::{self, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const INFERENCE_PIN: &str = "c6d6a4dbd61ab09c26ff5526632cae2cefea60ed";
pub const QWEN_REPOSITORY: &str = "SceneWorks/qwen-image-mlx";
pub const FLUX2_REPOSITORY: &str = "SceneWorks/flux2-dev-mlx";
/// The FLUX.2-klein-9B tiered rehost (sc-22727) — the `flux2_klein_9b` catalog model's artifact,
/// bound through `SCENEWORKS_FLUX2_KLEIN_*`. The engine discriminates it from the KV rehost by the
/// snapshot path AND by `LoadSpec::resolved_route` (`turnkey_identity` /
/// `KleinArtifactInventory::validate_resolved_route` in `mlx-gen-flux2/src/artifact_inventory.rs`),
/// which is why the two variants get separate env families rather than one shared "klein" family.
pub const FLUX2_KLEIN_REPOSITORY: &str = "SceneWorks/flux2-klein-9b-mlx";
/// The FLUX.2-klein-9B **KV-cache** rehost (sc-22727): a separately distilled checkpoint of the same
/// architecture, loaded through the SAME engine provider id `flux2_klein_9b`
/// (`crates/sceneworks-worker/src/engines.rs` — `sceneworks_id: flux2_klein_9b_kv`,
/// `engine_id: flux2_klein_9b`) from its own artifact, through `SCENEWORKS_FLUX2_KLEIN_KV_*`.
pub const FLUX2_KLEIN_KV_REPOSITORY: &str = "SceneWorks/flux2-klein-9b-kv-mlx";
pub const KREA_REPOSITORY: &str = "SceneWorks/krea-2-turbo-mlx";
pub const SDXL_REPOSITORY: &str = "SceneWorks/sdxl-base-mlx";
/// The Z-Image-Turbo tiered rehost. Serves the `z_image_turbo` provider AND the `z_image_edit`
/// catalog alias (the worker routes `z_image_edit` to the Turbo weights driven in `edit_image`
/// mode — `crates/sceneworks-worker/src/engines.rs`), on both adapters, through the
/// `SCENEWORKS_Z_IMAGE_*` family.
pub const Z_IMAGE_REPOSITORY: &str = "SceneWorks/z-image-turbo-mlx";
/// The undistilled Z-Image BASE tiered rehost (sc-22724) — the `z_image` provider's own artifact,
/// bound through the separate `SCENEWORKS_Z_IMAGE_BASE_*` family so a base plan can never be
/// satisfied by Turbo weights and re-label Turbo's peaks as the base model's.
pub const Z_IMAGE_BASE_REPOSITORY: &str = "SceneWorks/z-image-mlx";
/// The `mlx:ltx_2_3` calibration artifact (sc-18808). Its `gemma/` co-requisite text encoder is a
/// hard load-time requirement of the pinned provider, not a fallback, so a capture resolves TWO
/// roots under this one repository: the numeric tier and `gemma`.
pub const LTX_REPOSITORY: &str = "SceneWorks/ltx-2.3-mlx";
/// The single public LTX-2.5 rehost used by both native capture adapters. Roots are nested one
/// level deeper than the 2.3 artifact (`<snapshot>/<transformerVariant>/<tier>`); each adapter
/// validates both axes before opening its registered provider.
pub const LTX25_REPOSITORY: &str = "SceneWorks/ltx-2.5-mlx";
/// Compatibility spelling used by the Candle arm; keep both adapters bound to one literal.
pub const LTX_2_5_REPOSITORY: &str = LTX25_REPOSITORY;
/// Exact public revision sealed into every LTX-2.5 capture receipt.
pub const LTX_2_5_REVISION: &str = "081658ce6886cacba20817ce0359bbefef706ff2";
/// The `mlx:minimax_h3` TIERED artifact (sc-18663). Unlike every repository above it, this rehost
/// is **not sufficient on its own**: `mlx_gen_minimax_h3::model::load` probes `vae/`, `audio_vae/`,
/// `tokenizer/` and `FL2VA/audio_vae/` under the spec's own weights root, and this rehost publishes
/// none of them — it carries only the per-tier `transformer/`, `transformer_ref/` and (for q4/q8)
/// `text_encoder/`. A capture therefore resolves TWO artifact triples, this one and
/// [`MINIMAX_UPSTREAM_REPOSITORY`], and the record's loadability fingerprint names both.
pub const MINIMAX_REPOSITORY: &str = "SceneWorks/minimax-h3-mlx";
/// The upstream `minimax_h3` snapshot the shared partitions come from. It is a DIFFERENT repository
/// id from [`MINIMAX_REPOSITORY`] with no tier sub-directory, so it is validated through
/// [`validate_huggingface_revision_root`] rather than being forced through the rehost's
/// variant-suffixed validator.
pub const MINIMAX_UPSTREAM_REPOSITORY: &str = "MiniMaxAI/MiniMax-H3";
pub const COMPARISON_OUTPUT_BIAS_PARAMETER: &str = "comparisonOutputBias";
/// Persisted-JSON spellings of `gen_core::LoadShape`. Every emitted fragment must state the
/// materialization shape its run actually used; the harness rejects a fragment that omits it, and
/// never backfills the field from the plan (sc-16482) — a receipt may only testify to its own run.
pub const LOAD_SHAPE_EAGER: &str = "eager_materialization";
pub const LOAD_SHAPE_DEFERRED: &str = "deferred_materialization";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ltx25TransformerVariant {
    Distilled,
    Dev,
}

impl Ltx25TransformerVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Distilled => "distilled",
            Self::Dev => "dev",
        }
    }

    pub const fn steps(self) -> u32 {
        match self {
            Self::Distilled => 8,
            Self::Dev => 30,
        }
    }

    pub const fn requires_official_refinement_lora(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn validate_load_shape(self, load_shape: &str) -> Result<(), String> {
        match (self, load_shape) {
            (Self::Dev, LOAD_SHAPE_EAGER)
            | (Self::Distilled, LOAD_SHAPE_EAGER | LOAD_SHAPE_DEFERRED) => Ok(()),
            (Self::Dev, LOAD_SHAPE_DEFERRED) => Err(
                "LTX-2.5 dev capture requires eager_materialization because the official stage-two refinement LoRA makes transformer streaming ineligible"
                    .to_owned(),
            ),
            (_, other) => Err(format!("unsupported LTX-2.5 Candle loadShape {other:?}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ltx25Decoder {
    Conv,
    DiffVae,
}

impl Ltx25Decoder {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conv => "conv",
            Self::DiffVae => "diffvae",
        }
    }
}

/// Exact plan axes consumed by the real LTX-2.5 Candle capture arm. Keeping this parser in the
/// weight-free library makes the fail-closed request contract testable on a CPU-only host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ltx25CandleTarget {
    pub tier: String,
    pub transformer_variant: Ltx25TransformerVariant,
    pub decoder: Ltx25Decoder,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
}

pub fn ltx25_candle_target(request: &Value) -> Result<Ltx25CandleTarget, String> {
    let planned = planned(request)?;
    let target = planned
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let string = |field: &str| {
        target
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("planned.target.{field} must be a string"))
    };
    if string("provider")? != "ltx_2_5_distilled" {
        return Err("LTX-2.5 Candle capture requires provider \"ltx_2_5_distilled\"".to_owned());
    }
    if string("modelId")? != "ltx_2_5" {
        return Err("LTX-2.5 Candle capture requires modelId \"ltx_2_5\"".to_owned());
    }
    if string("mode")? != "text_to_video" {
        return Err("LTX-2.5 Candle capture requires mode \"text_to_video\"".to_owned());
    }
    if string("overlay")? != "none" {
        return Err(
            "LTX-2.5 Candle base campaign requires target.overlay \"none\"; the official dev refinement is part of the base recipe"
                .to_owned(),
        );
    }
    let tier = match string("tier")? {
        // q8 is a first-class Candle tier, symmetric with the MLX arm: the promoted candle
        // descriptor advertises Q4+Q8, so an ordinary q8 row is executable Candle evidence.
        tier @ ("bf16" | "q4" | "q8") => tier.to_owned(),
        tier => return Err(format!("unsupported LTX-2.5 Candle numeric tier {tier:?}")),
    };
    let transformer_variant = match string("transformerVariant")? {
        "distilled" => Ltx25TransformerVariant::Distilled,
        "dev" => Ltx25TransformerVariant::Dev,
        variant => {
            return Err(format!(
                "unsupported LTX-2.5 transformerVariant {variant:?}"
            ))
        }
    };
    let decoder = match string("decoder")? {
        "conv" => Ltx25Decoder::Conv,
        "diffvae" => Ltx25Decoder::DiffVae,
        decoder => return Err(format!("unsupported LTX-2.5 decoder {decoder:?}")),
    };
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let axis = |field: &str| {
        geometry
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{field} must fit u32"))
    };
    let width = axis("width")?;
    let height = axis("height")?;
    let frames = axis("frames")?;
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "LTX-2.5 Candle capture requires geometry.batch == 1, got {batch}"
        ));
    }
    if width < 64 || height < 64 || width > 1280 || height > 1280 {
        return Err(format!(
            "LTX-2.5 Candle capture requires width/height in 64..=1280, got {width}x{height}"
        ));
    }
    if width % 64 != 0 || height % 64 != 0 {
        return Err(format!(
            "LTX-2.5 Candle capture requires width/height divisible by 64, got {width}x{height}"
        ));
    }
    if frames == 0 || frames % 8 != 1 {
        return Err(format!(
            "LTX-2.5 Candle capture requires geometry.frames == 1 + 8k, got {frames}"
        ));
    }
    let fixture = planned
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("ltx-2-5-candle-{tier}-{width}x{height}-f{frames}-fps");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse LTX-2.5 fixture fps {fps:?}: {error}"))?;
    if ![24, 25, 30].contains(&fps) {
        return Err(format!(
            "planned.fixture fps {fps} is outside the LTX-2.5 manifest values [24, 25, 30]"
        ));
    }
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse LTX-2.5 fixture seed {seed:?}: {error}"))?;
    Ok(Ltx25CandleTarget {
        tier,
        transformer_variant,
        decoder,
        width,
        height,
        frames,
        fps,
        seed,
    })
}

pub fn validate_ltx25_artifact_identity(repository: &str, revision: &str) -> Result<(), String> {
    validate_artifact_identity(repository, revision, LTX_2_5_REPOSITORY)?;
    if revision != LTX_2_5_REVISION {
        return Err(format!(
            "LTX-2.5 calibration requires exact public revision {LTX_2_5_REVISION}, got {revision}"
        ));
    }
    Ok(())
}

pub fn validate_lowercase_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be 64 lowercase hex characters"));
    }
    Ok(())
}

/// Keep the probe's cumulative load+generate peak rather than reconstructing an overall value from
/// independently sampled phase windows.
pub fn validated_cumulative_peak(
    cumulative_peak_bytes: u64,
    phase_peak_bytes: [u64; 3],
) -> Result<u64, String> {
    let phase_maximum = phase_peak_bytes.into_iter().max().unwrap_or(0);
    if cumulative_peak_bytes == 0 || cumulative_peak_bytes < phase_maximum {
        return Err(format!(
            "cumulative run peak {cumulative_peak_bytes} must be nonzero and at least the largest phase peak {phase_maximum}"
        ));
    }
    Ok(cumulative_peak_bytes)
}

/// Validate the exact decoded frame geometry and RGB payload before a capture may report that the
/// selected artifact loaded successfully.
pub fn validate_ltx25_rgb_frames(
    expected_count: usize,
    expected_width: u32,
    expected_height: u32,
    frames: &[(u32, u32, usize)],
) -> Result<(), String> {
    if expected_count == 0 || frames.len() != expected_count {
        return Err(format!(
            "LTX-2.5 returned {} frames, expected {expected_count}",
            frames.len()
        ));
    }
    let expected_pixels = usize::try_from(expected_width)
        .ok()
        .and_then(|width| {
            usize::try_from(expected_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "LTX-2.5 expected RGB payload length overflowed usize".to_owned())?;
    if expected_pixels == 0 {
        return Err("LTX-2.5 expected frame geometry must be nonempty".to_owned());
    }
    for (index, &(width, height, pixels)) in frames.iter().enumerate() {
        if (width, height) != (expected_width, expected_height) {
            return Err(format!(
                "LTX-2.5 frame {index} is {width}x{height}, expected {expected_width}x{expected_height}"
            ));
        }
        if pixels != expected_pixels {
            return Err(format!(
                "LTX-2.5 frame {index} RGB payload has {pixels} bytes, expected {expected_pixels}"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferencePhase {
    Conditioning,
    Denoise,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceBoundary {
    RendererLoad,
    FirstDenoiseStep,
    Decoding,
}

/// Return the next measurable Candle reference phase for provider progress paths that differ by
/// strategy. Resident exposes only Step/Decoding; staged exposes Renderer/Decoding; higher rungs
/// expose two Renderer boundaries. All three sequences must converge on the same phase lifecycle.
pub fn next_reference_phase(
    phase: ReferencePhase,
    boundary: ReferenceBoundary,
) -> Option<ReferencePhase> {
    match (phase, boundary) {
        (
            ReferencePhase::Conditioning,
            ReferenceBoundary::RendererLoad | ReferenceBoundary::FirstDenoiseStep,
        ) => Some(ReferencePhase::Denoise),
        (
            ReferencePhase::Denoise,
            ReferenceBoundary::RendererLoad | ReferenceBoundary::Decoding,
        ) => Some(ReferencePhase::Decode),
        _ => None,
    }
}

pub fn request_from_stdin() -> Result<Value, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("read provider request: {error}"))?;
    serde_json::from_str(&input).map_err(|error| format!("parse provider request JSON: {error}"))
}

pub fn write_response(response: &Value) -> Result<(), String> {
    serde_json::to_writer(io::stdout(), response)
        .map_err(|error| format!("write provider response JSON: {error}"))
}

pub fn action(request: &Value) -> Result<&str, String> {
    request
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| "provider request.action must be a string".to_owned())
}

pub fn planned(request: &Value) -> Result<&Value, String> {
    request
        .get("planned")
        .ok_or_else(|| "run request is missing planned".to_owned())
}

pub fn strategy_parameters(request: &Value) -> Result<&Map<String, Value>, String> {
    planned(request)?
        .pointer("/strategy/parameters")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.strategy.parameters must be an object".to_owned())
}

pub fn parameter(request: &Value, name: &str) -> Result<u32, String> {
    let value = strategy_parameters(request)?
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("planned.strategy.parameters.{name} must be an integer"))?;
    u32::try_from(value).map_err(|_| format!("planned.strategy.parameters.{name} exceeds u32"))
}

pub fn optional_parameter(request: &Value, name: &str) -> Result<Option<u32>, String> {
    strategy_parameters(request)?
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| format!("planned.strategy.parameters.{name} must be an integer"))
                .and_then(|value| {
                    u32::try_from(value)
                        .map_err(|_| format!("planned.strategy.parameters.{name} exceeds u32"))
                })
        })
        .transpose()
}

pub fn planned_rung(request: &Value) -> Result<&str, String> {
    planned(request)?
        .pointer("/strategy/rung")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.strategy.rung must be a string".to_owned())
}

pub fn reference_sweep(request: &Value, result: &str) -> Result<Value, String> {
    let parameters = strategy_parameters(request)?;
    let numeric = parameters
        .iter()
        .map(|(name, value)| {
            value
                .as_u64()
                .ok_or_else(|| {
                    format!("fresh-reference strategy parameter {name} must be an unsigned integer")
                })
                .map(|value| (name, value))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (axes, cases) = if numeric.is_empty() {
        (
            Vec::<Value>::new(),
            vec![json!({ "parameters": {}, "result": result })],
        )
    } else {
        (
            numeric
                .iter()
                .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
                .collect(),
            vec![json!({ "parameters": parameters, "result": result })],
        )
    };
    Ok(json!({
        "axes": axes,
        "cases": cases,
        "rangeVerified": false,
    }))
}

pub fn comparison_output_bias(
    parameters: &Map<String, Value>,
    expected_failure: bool,
) -> Result<Option<f64>, String> {
    let bias = parameters
        .get(COMPARISON_OUTPUT_BIAS_PARAMETER)
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                format!(
                    "planned.strategy.parameters.{COMPARISON_OUTPUT_BIAS_PARAMETER} must be a number"
                )
            })
        })
        .transpose()?;
    match (expected_failure, bias) {
        (false, None) => Ok(None),
        (false, Some(_)) => Err(format!(
            "{COMPARISON_OUTPUT_BIAS_PARAMETER} is reserved for an expected-failure case"
        )),
        (true, None) => Err(format!(
            "expected-failure case must declare {COMPARISON_OUTPUT_BIAS_PARAMETER}"
        )),
        (true, Some(value)) if value.is_finite() && value > 0.0 => Ok(Some(value)),
        (true, Some(_)) => Err(format!(
            "planned.strategy.parameters.{COMPARISON_OUTPUT_BIAS_PARAMETER} must be finite and greater than zero"
        )),
    }
}

pub fn max_mean_abs(
    left: &[f32],
    right: &[f32],
    comparison_output_bias: Option<f64>,
) -> Result<(f64, f64), String> {
    if left.len() != right.len() || left.is_empty() {
        return Err(format!(
            "decode output length mismatch: baseline={} tiled={}",
            left.len(),
            right.len()
        ));
    }
    if comparison_output_bias.is_some_and(|bias| !bias.is_finite() || bias <= 0.0) {
        return Err("comparison output bias must be finite and greater than zero".to_owned());
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            return Err(format!(
                "decode output contains a non-finite sample at index {index}"
            ));
        }
        let baseline = f64::from(*left);
        let tiled = f64::from(*right);
        let compared = comparison_output_bias.map_or(tiled, |bias| {
            if tiled >= baseline {
                tiled + bias
            } else {
                tiled - bias
            }
        });
        let difference = (baseline - compared).abs();
        if !compared.is_finite() || !difference.is_finite() {
            return Err(format!(
                "decode comparison produced a non-finite result at index {index}"
            ));
        }
        maximum = maximum.max(difference);
        sum += difference;
        if !sum.is_finite() {
            return Err("decode comparison mean accumulator became non-finite".to_owned());
        }
    }
    let mean = sum / left.len() as f64;
    if !maximum.is_finite() || !mean.is_finite() {
        return Err("decode comparison metrics must be finite".to_owned());
    }
    Ok((maximum, mean))
}

pub fn validate_comparison_shapes(left: &[i32], right: &[i32]) -> Result<(), String> {
    if left != right {
        return Err(format!(
            "decode output shape mismatch: baseline={left:?} tiled={right:?}"
        ));
    }
    Ok(())
}

pub fn expected_failure(request: &Value) -> bool {
    planned(request)
        .ok()
        .and_then(|planned| planned.get("expectedResult").and_then(Value::as_str))
        == Some("failed")
}

pub fn target_geometry(request: &Value) -> Result<(u32, u32), String> {
    let geometry = planned(request)?
        .pointer("/target/geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let dimension = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    Ok((dimension("width")?, dimension("height")?))
}

/// Refuse a non-still geometry before an IMAGE provider arm does environment or weight work.
///
/// The frames axis is a **per-arm** contract, not a global one (sc-18808). Every arm in this crate
/// used to be an image arm, so `frames == 1` read like an apparatus-wide invariant; it never was
/// one, and the first video arm (`mlx:ltx_2_3`) legitimately renders `1 + 8k` frames. Hoisting the
/// refusal here rather than deleting it keeps the image arms strict while letting exactly one arm
/// declare a different envelope, in one place a reader can see both halves of.
///
/// It is hoisted for a second, sharper reason. Only two of the six image arms — Krea base and SDXL —
/// actually validated the axis; the other four read only `width`/`height` through
/// [`target_geometry`] and then hardcoded `frames: 1` into their admission context. A plan row
/// declaring `frames: 2` therefore rendered ONE frame and produced a record whose geometry envelope
/// claimed a single frame it had not been asked for — a silent, well-formed lie about what was
/// measured, which is the exact defect class this apparatus exists to make impossible. The two
/// original messages are reproduced verbatim through `calibration_label` so the pinned negative
/// tests keep asserting the same wording.
pub fn validate_still_geometry(request: &Value, calibration_label: &str) -> Result<(), String> {
    let geometry = planned(request)?
        .pointer("/target/geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    for (axis, expected) in [("batch", 1_u64), ("frames", 1_u64)] {
        let actual = geometry
            .get(axis)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("planned.target.geometry.{axis} must be an integer"))?;
        if actual != expected {
            return Err(format!(
                "{calibration_label} requires geometry.{axis} == {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

/// The overlay this calibration target declares (`planned.target.overlay`) — `"none"`, `"lora"`,
/// `"control"`, `"identity"` (sc-16069).
///
/// An adapter MUST derive its `overlay` scenario verdict from this rather than hardcoding one. The
/// candle adapter used to emit `not_applicable` with the fixed reason "ordinary Krea Turbo
/// text-to-image calibration has no overlay" on every run, which is a statement about the adapter's
/// one code path, not about the target it was handed — so a target that declared an overlay would
/// still have produced a `not_applicable` record that reads as considered coverage. Reading the target
/// makes the verdict true by construction, and lets an adapter refuse a target it cannot execute
/// instead of quietly excusing it.
pub fn target_overlay(request: &Value) -> Result<String, String> {
    planned(request)?
        .pointer("/target/overlay")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "planned.target.overlay must be a string".to_owned())
}

/// Reject a material overlay before a provider path that only loads its base model does work.
///
/// Keeping this check in the shared protocol crate prevents either backend from turning a static
/// description of its usual workload into false `not_applicable` coverage for a requested overlay.
pub fn validate_plain_overlay_target(request: &Value, execution_path: &str) -> Result<(), String> {
    let overlay = target_overlay(request)?;
    if overlay == "none" {
        return Ok(());
    }
    Err(format!(
        "calibration target declares overlay {overlay:?}, but {execution_path} only executes the \
         base target; refusing rather than recording false overlay coverage"
    ))
}

/// Require the exact material overlay a provider path actually loaded and exercised.
pub fn validate_exact_overlay_target(
    request: &Value,
    expected_overlay: &str,
    execution_path: &str,
) -> Result<(), String> {
    let overlay = target_overlay(request)?;
    if overlay == expected_overlay {
        return Ok(());
    }
    Err(format!(
        "calibration target declares overlay {overlay:?}, but {execution_path} executes exactly \
         {expected_overlay:?}; refusing to record the exercised overlay under a different target"
    ))
}

/// Settle the required overlay scenario for a provider path that intentionally executes no overlay.
///
/// The target is validated before the fragment is mutated, so a `lora`, `identity`, or `control`
/// request fails closed and can never acquire a `not_applicable` verdict.
pub fn settle_plain_overlay_scenario(
    request: &Value,
    fragment: &mut Value,
    execution_path: &str,
) -> Result<(), String> {
    validate_plain_overlay_target(request, execution_path)?;
    let scenarios = fragment
        .get_mut("scenarios")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "provider fragment.scenarios must be an array".to_owned())?;
    let overlay_index = scenarios
        .iter()
        .position(|scenario| scenario.get("name").and_then(Value::as_str) == Some("overlay"))
        .ok_or_else(|| "provider fragment is missing the required overlay scenario".to_owned())?;
    scenarios[overlay_index] = json!({
        "name": "overlay",
        "result": "not_applicable",
        "reason": format!(
            "the calibration target declares overlay \"none\"; {execution_path} executes no overlay, so this record has no second resident network to measure"
        ),
    });
    Ok(())
}

/// The reference image an `edit_image` anchor conditions on: a deterministic RGB gradient at the
/// request geometry (`width * height * 3` interleaved bytes), which is exactly the shape the worker
/// hands the engine — it fits the user's source to the request geometry before conditioning
/// (`fit_engine_image` in `image_jobs/base.rs`). A reference is what makes an edit capture measure
/// the edit path (VAE-encode of the source, the reduced denoise tail) rather than text-to-image
/// wearing a different mode label; the pixel CONTENT does not move memory, so a synthetic one
/// keeps the capture hermetic and reproducible.
pub fn synthetic_reference_rgb(width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width.max(1) as u64, height.max(1) as u64);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(2).saturating_sub(1).max(1)).min(255) as u8);
            pixels.push((y * 255 / h.max(2).saturating_sub(1).max(1)).min(255) as u8);
            pixels.push((((x + y) * 255) / (w + h).saturating_sub(2).max(1)).min(255) as u8);
        }
    }
    pixels
}

pub fn required_env(name: &str) -> Result<String, String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("required environment variable {name} is not set"))
}

pub fn validate_artifact_identity(
    repository: &str,
    revision: &str,
    expected_repository: &str,
) -> Result<(), String> {
    if repository != expected_repository {
        return Err(format!(
            "artifact repository must be the fixed {expected_repository} calibration artifact"
        ));
    }
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("artifact revision must be an exact lowercase 40-hex commit".to_owned());
    }
    Ok(())
}

pub fn validate_huggingface_snapshot_root(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    variant: &str,
    expected_repository: &str,
) -> Result<(), String> {
    validate_snapshot_suffix(
        canonical_root,
        repository,
        revision,
        Some(variant),
        expected_repository,
    )
}

/// The variant-free twin of [`validate_huggingface_snapshot_root`]: the canonical root must be the
/// snapshot directory ITSELF, `.../models--<owner>--<name>/snapshots/<revision>`.
///
/// Added for `mlx:minimax_h3` (sc-18663), whose shared partitions live directly under an upstream
/// snapshot that has no tier sub-directory. Forcing that root through the variant validator would
/// mean inventing a variant component the publisher does not have, so the suffix that is actually
/// checked is the one the artifact actually carries.
pub fn validate_huggingface_revision_root(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    expected_repository: &str,
) -> Result<(), String> {
    validate_snapshot_suffix(
        canonical_root,
        repository,
        revision,
        None,
        expected_repository,
    )
}

/// Validate a canonical Hugging Face root with more than one repository-relative selector, such as
/// LTX-2.5's `<transformer-variant>/<numeric-tier>` layout.
pub fn validate_huggingface_snapshot_subpath(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    subpath: &[&str],
    expected_repository: &str,
) -> Result<(), String> {
    validate_artifact_identity(repository, revision, expected_repository)?;
    if subpath.is_empty()
        || subpath.iter().any(|component| {
            component.is_empty() || component.contains('/') || component.contains('\\')
        })
    {
        return Err("artifact snapshot subpath must contain plain non-empty components".to_owned());
    }
    let repository_component = format!("models--{}", expected_repository.replace('/', "--"));
    let mut expected = vec![repository_component.as_str(), "snapshots", revision];
    expected.extend(subpath.iter().copied());
    let components = canonical_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "artifact root must end with /{repository_component}/snapshots/{revision}/{}",
            subpath.join("/")
        ));
    }
    Ok(())
}

fn validate_snapshot_suffix(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    variant: Option<&str>,
    expected_repository: &str,
) -> Result<(), String> {
    validate_artifact_identity(repository, revision, expected_repository)?;
    let repository_component = format!("models--{}", expected_repository.replace('/', "--"));
    let mut expected = vec![repository_component.as_str(), "snapshots", revision];
    expected.extend(variant);
    let components = canonical_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        let suffix = variant
            .map(|variant| format!("/{variant}"))
            .unwrap_or_default();
        return Err(format!(
            "artifact root must end with /{repository_component}/snapshots/{revision}{suffix}"
        ));
    }
    Ok(())
}

pub fn not_run_scenarios(blocker: &str) -> Value {
    Value::Array(
        [
            "exact_fit",
            "unknown_budget",
            "stale_evidence",
            "warm_repeat",
            "cancel",
            "error",
            "loadability",
            "overlay",
        ]
        .into_iter()
        .map(|name| json!({ "name": name, "result": "not_run", "reason": blocker }))
        .collect(),
    )
}

pub fn diagnostics(
    adapter: &str,
    execution: &str,
    blockers: impl IntoIterator<Item = String>,
    measurements: impl IntoIterator<Item = (&'static str, &'static str, u64)>,
) -> Value {
    json!({
        "adapter": adapter,
        "execution": execution,
        "blockers": blockers.into_iter().collect::<Vec<_>>(),
        "measurements": measurements
            .into_iter()
            .map(|(name, unit, value)| json!({ "name": name, "unit": unit, "value": value }))
            .collect::<Vec<_>>(),
    })
}

pub fn captured_at() -> String {
    // UTC conversion without adding a time-formatting dependency to the calibration-only crate.
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (seconds / 86_400) as i64;
    let day_seconds = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        (day_seconds % 3_600) / 60,
        day_seconds % 60
    )
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    // Howard Hinnant's proleptic-Gregorian civil_from_days, shifted from 1970-01-01.
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

pub struct PlainGatedFragment<'a> {
    pub artifact: Value,
    pub sweep: Value,
    pub blocker: &'a str,
    pub quality: Value,
    pub negative_mutation: Value,
    pub loadability: Value,
    pub diagnostics: Value,
}

/// Build a gated fragment for a base-only provider path without ever leaving `overlay` as
/// `not_run`. Material overlay targets fail closed before a fragment is returned.
pub fn plain_gated_fragment(
    request: &Value,
    execution_path: &str,
    parts: PlainGatedFragment<'_>,
) -> Result<Value, String> {
    validate_plain_overlay_target(request, execution_path)?;
    let mut fragment = json!({
        "status": "gated",
        "artifact": parts.artifact,
        "sweep": parts.sweep,
        "scenarios": not_run_scenarios(parts.blocker),
        "predictedPeakBytes": null,
        "observedMemory": null,
        "quality": parts.quality,
        "negativeMutation": parts.negative_mutation,
        "loadability": parts.loadability,
        "diagnostics": parts.diagnostics,
        "capturedAt": captured_at(),
    });
    settle_plain_overlay_scenario(request, &mut fragment, execution_path)?;
    Ok(fragment)
}

pub fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("memory-strategy provider adapter: {}", message.as_ref());
    std::process::exit(1);
}

/// The allocator counters an MLX phase window is opened against, abstracted so the ORDER in which
/// the window opens can be proven on a host with no Apple hardware (sc-22667, epic 22657 D3).
///
/// The production implementation is the `mlx_rs::memory` free functions; the tests drive a fake.
pub trait ResidencyCounters {
    /// Release the allocator's retained free buffers (`mlx_rs::memory::clear_cache`).
    fn clear_cache(&mut self);
    /// Restart the active high-water mark (`mlx_rs::memory::reset_peak_memory`).
    fn reset_peak(&mut self);
    /// Bytes held by live arrays right now (`get_active_memory`).
    fn active(&self) -> u64;
    /// Bytes the allocator retains for reuse right now (`get_cache_memory`).
    fn cache(&self) -> u64;
    /// The active high-water mark since the last reset (`get_peak_memory`).
    fn peak(&self) -> u64;
}

/// What the counters read the instant a resident phase window opened: the resident set the window's
/// every phase peak will then include, and the two readings the record carries beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentWindowOpening {
    /// Live bytes after the resident set was materialized and the cache released — the weights, and
    /// nothing else. Recorded as `preRungActiveAfterClear`.
    pub resident_active: u64,
    /// Retained cache after the release, expected ~0. Recorded as `preRungCacheAfterClear`.
    pub resident_cache: u64,
    /// The high-water mark immediately after the reset, before the first allocation of the window.
    /// Recorded as `peakAfterReset`.
    pub peak_after_reset: u64,
}

/// Open a phase window whose peaks INCLUDE the model's resident set, the way the candle adapter's
/// device deltas do (weights on device before the window, every phase measured above the idle
/// baseline).
///
/// ROOT CAUSE THIS EXISTS FOR (sc-22667 D3). The MLX providers with request-scoped residency
/// (Z-Image, Qwen-Image, FLUX.2) materialize each component the first time a request reaches its
/// phase, even under `LoadShape::EagerMaterialization` — "eager" names retention across requests,
/// not materialization at load. A window opened on a freshly loaded generator therefore measured a
/// COLD first request: the conditioning phase saw only the text encoder it was materializing
/// (packaged z_image_turbo q4 MLX record: `preRungActiveAfterClear` 0, conditioning active peak
/// 2.27 GB against a 5.83 GB post-request resident set), the denoise phase saw the text encoder plus
/// the transformer it was materializing, and only the decode phase saw the whole set. Every packaged
/// MLX image anchor reported a conditioning level below the resident set its eager regime claims,
/// so the core derivation law (`sceneworks-core::memory_anchor`) refused to decompose any of them —
/// a residue below zero is not a working set — and the MLX lane priced nothing from its anchors.
///
/// The fix is an ORDER: `materialize_resident_set` runs FIRST (one unmeasured request on the same
/// loaded generator, which materializes and retains every component), THEN the cache is released,
/// THEN the peak is reset and the window opens. Every phase peak of the measured request then sits
/// above the same resident set, which is the quantity the law subtracts. The opening refuses a
/// resident set the active counter did not see at all (zero live bytes after materialization): that
/// is the exact false reading this fix removes, and recording it again would re-create the defect
/// under a new capture date.
pub fn open_resident_phase_window<C: ResidencyCounters>(
    counters: &mut C,
    materialize_resident_set: impl FnOnce() -> Result<(), String>,
) -> Result<ResidentWindowOpening, String> {
    materialize_resident_set()?;
    counters.clear_cache();
    counters.reset_peak();
    let opening = ResidentWindowOpening {
        resident_active: counters.active(),
        resident_cache: counters.cache(),
        peak_after_reset: counters.peak(),
    };
    if opening.resident_active == 0 {
        return Err(
            "the MLX active counter reports zero live bytes after the resident set was \
             materialized; a phase window opened here would measure activations without their \
             weights, which the anchor derivation law cannot decompose"
                .to_owned(),
        );
    }
    Ok(opening)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counter fake that logs every call, so a test can assert the ORDER the window opens in,
    /// and whose shared state the materializing closure raises the way a first request does.
    #[derive(Default)]
    struct FakeState {
        log: Vec<&'static str>,
        active: u64,
        cache: u64,
        peak: u64,
    }

    #[derive(Clone, Default)]
    struct FakeCounters(std::rc::Rc<std::cell::RefCell<FakeState>>);

    impl ResidencyCounters for FakeCounters {
        fn clear_cache(&mut self) {
            let mut state = self.0.borrow_mut();
            state.log.push("clear_cache");
            state.cache = 0;
        }
        fn reset_peak(&mut self) {
            let mut state = self.0.borrow_mut();
            state.log.push("reset_peak");
            // MLX's reset zeroes the mark; the next allocation raises it to at least `active`.
            state.peak = 0;
        }
        fn active(&self) -> u64 {
            self.0.borrow().active
        }
        fn cache(&self) -> u64 {
            self.0.borrow().cache
        }
        fn peak(&self) -> u64 {
            self.0.borrow().peak
        }
    }

    #[test]
    fn a_resident_phase_window_materializes_the_weights_before_it_rebaselines() {
        let mut counters = FakeCounters::default();
        let shared = counters.clone();
        // The materializing request is what a loaded request-scoped generator does on its first
        // render: the resident set appears in `active`, and the request leaves cache behind.
        let opening = open_resident_phase_window(&mut counters, || {
            let mut state = shared.0.borrow_mut();
            state.log.push("materialize");
            state.active = 5_833_351_792;
            state.cache = 4_846_009_704;
            state.peak = 9_910_632_828;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            counters.0.borrow().log,
            ["materialize", "clear_cache", "reset_peak"],
            "the rebaseline must follow the materialization, never precede it"
        );
        assert_eq!(
            opening,
            ResidentWindowOpening {
                resident_active: 5_833_351_792,
                resident_cache: 0,
                peak_after_reset: 0,
            }
        );
    }

    #[test]
    fn a_resident_phase_window_refuses_a_resident_set_the_counters_did_not_see() {
        let mut counters = FakeCounters::default();
        let error = open_resident_phase_window(&mut counters, || Ok(())).unwrap_err();
        assert!(error.contains("zero live bytes"), "{error}");
        assert_eq!(counters.0.borrow().log, ["clear_cache", "reset_peak"]);
    }

    #[test]
    fn a_resident_phase_window_does_not_rebaseline_when_materialization_fails() {
        let mut counters = FakeCounters::default();
        let error = open_resident_phase_window(&mut counters, || Err("render failed".to_owned()))
            .unwrap_err();
        assert_eq!(error, "render failed");
        assert!(
            counters.0.borrow().log.is_empty(),
            "no counter was touched: {:?}",
            counters.0.borrow().log
        );
    }

    fn ltx25_request() -> Value {
        json!({
            "planned": {
                "target": {
                    "provider": "ltx_2_5_distilled",
                    "modelId": "ltx_2_5",
                    "tier": "q4",
                    "mode": "text_to_video",
                    "overlay": "none",
                    "transformerVariant": "distilled",
                    "decoder": "conv",
                    "geometry": { "width": 768, "height": 512, "batch": 1, "frames": 145 }
                },
                "fixture": "ltx-2-5-candle-q4-768x512-f145-fps24-seed18755"
            }
        })
    }

    #[test]
    fn ltx25_candle_target_binds_every_planned_axis() {
        let target = ltx25_candle_target(&ltx25_request()).unwrap();
        assert_eq!(target.tier, "q4");
        assert_eq!(
            target.transformer_variant,
            Ltx25TransformerVariant::Distilled
        );
        assert_eq!(target.decoder, Ltx25Decoder::Conv);
        assert_eq!(
            (target.width, target.height, target.frames),
            (768, 512, 145)
        );
        assert_eq!((target.fps, target.seed), (24, 18_755));

        let mut dev = ltx25_request();
        dev["planned"]["target"]["tier"] = json!("bf16");
        dev["planned"]["target"]["transformerVariant"] = json!("dev");
        dev["planned"]["target"]["decoder"] = json!("diffvae");
        dev["planned"]["target"]["geometry"] =
            json!({ "width": 512, "height": 512, "batch": 1, "frames": 17 });
        dev["planned"]["fixture"] = json!("ltx-2-5-candle-bf16-512x512-f17-fps25-seed18777");
        let target = ltx25_candle_target(&dev).unwrap();
        assert_eq!(target.tier, "bf16");
        assert_eq!(target.transformer_variant, Ltx25TransformerVariant::Dev);
        assert_eq!(target.decoder, Ltx25Decoder::DiffVae);
        assert_eq!((target.width, target.height, target.frames), (512, 512, 17));
    }

    #[test]
    fn ltx25_candle_admits_q8_as_a_first_class_tier_and_still_fails_closed_off_ladder() {
        let mut request = ltx25_request();
        request["planned"]["target"]["tier"] = json!("q8");
        request["planned"]["fixture"] = json!("ltx-2-5-candle-q8-768x512-f145-fps24-seed18755");
        let target = ltx25_candle_target(&request).unwrap();
        assert_eq!(target.tier, "q8");
        assert_eq!(
            target.transformer_variant,
            Ltx25TransformerVariant::Distilled
        );

        // The ladder is still exactly q4/q8/bf16 — a non-published tier stays fail-closed.
        let mut off_ladder = ltx25_request();
        off_ladder["planned"]["target"]["tier"] = json!("q6");
        off_ladder["planned"]["fixture"] = json!("ltx-2-5-candle-q6-768x512-f145-fps24-seed18755");
        assert!(ltx25_candle_target(&off_ladder)
            .unwrap_err()
            .contains("unsupported LTX-2.5 Candle numeric tier"));
    }

    #[test]
    fn ltx25_dev_recipe_requires_the_official_lora_and_eager_load_shape() {
        assert!(!Ltx25TransformerVariant::Distilled.requires_official_refinement_lora());
        assert!(Ltx25TransformerVariant::Dev.requires_official_refinement_lora());
        assert!(Ltx25TransformerVariant::Distilled
            .validate_load_shape(LOAD_SHAPE_DEFERRED)
            .is_ok());
        assert!(Ltx25TransformerVariant::Dev
            .validate_load_shape(LOAD_SHAPE_EAGER)
            .is_ok());
        let error = Ltx25TransformerVariant::Dev
            .validate_load_shape(LOAD_SHAPE_DEFERRED)
            .unwrap_err();
        assert!(error.contains("official stage-two refinement LoRA"));
        assert!(error.contains("eager_materialization"));
    }

    #[test]
    fn ltx25_candle_target_fails_closed_on_identity_and_geometry_drift() {
        for (pointer, replacement, expected) in [
            ("/planned/target/provider", json!("ltx_2_5"), "provider"),
            (
                "/planned/target/modelId",
                json!("ltx_2_5_distilled"),
                "modelId",
            ),
            ("/planned/target/mode", json!("image_to_video"), "mode"),
            ("/planned/target/overlay", json!("lora"), "overlay"),
            (
                "/planned/target/transformerVariant",
                json!("turbo"),
                "transformerVariant",
            ),
            ("/planned/target/decoder", json!("auto"), "decoder"),
            ("/planned/target/geometry/batch", json!(2), "batch"),
            (
                "/planned/target/geometry/width",
                json!(770),
                "divisible by 64",
            ),
            ("/planned/target/geometry/frames", json!(144), "1 + 8k"),
        ] {
            let mut request = ltx25_request();
            *request.pointer_mut(pointer).unwrap() = replacement;
            let error = ltx25_candle_target(&request).unwrap_err();
            assert!(error.contains(expected), "{pointer}: {error}");
        }
        let mut mismatched_fixture = ltx25_request();
        mismatched_fixture["planned"]["target"]["tier"] = json!("bf16");
        assert!(ltx25_candle_target(&mismatched_fixture)
            .unwrap_err()
            .contains("must start with"));
    }

    #[test]
    fn ltx25_nested_snapshot_root_binds_variant_and_tier() {
        let revision = LTX_2_5_REVISION;
        let root = Path::new("/cache/models--SceneWorks--ltx-2.5-mlx/snapshots")
            .join(revision)
            .join("dev")
            .join("q8");
        validate_huggingface_snapshot_subpath(
            &root,
            LTX_2_5_REPOSITORY,
            revision,
            &["dev", "q8"],
            LTX_2_5_REPOSITORY,
        )
        .unwrap();
        assert!(validate_huggingface_snapshot_subpath(
            &root,
            LTX_2_5_REPOSITORY,
            revision,
            &["distilled", "q8"],
            LTX_2_5_REPOSITORY,
        )
        .is_err());
        assert!(validate_huggingface_snapshot_subpath(
            &root,
            LTX_2_5_REPOSITORY,
            revision,
            &["dev/q8"],
            LTX_2_5_REPOSITORY,
        )
        .is_err());
    }

    #[test]
    fn ltx25_identity_requires_the_exact_public_revision_and_inventory_digest() {
        validate_ltx25_artifact_identity(LTX_2_5_REPOSITORY, LTX_2_5_REVISION).unwrap();
        let error =
            validate_ltx25_artifact_identity(LTX_2_5_REPOSITORY, &"a".repeat(40)).unwrap_err();
        assert!(error.contains(LTX_2_5_REVISION));
        validate_lowercase_sha256(&"b".repeat(64), "inventory").unwrap();
        for invalid in ["b".repeat(63), "B".repeat(64), "g".repeat(64)] {
            assert!(validate_lowercase_sha256(&invalid, "inventory").is_err());
        }
    }

    #[test]
    fn cumulative_peak_is_not_reconstructed_from_phase_maxima() {
        assert_eq!(
            validated_cumulative_peak(900, [500, 600, 700]).unwrap(),
            900
        );
        assert!(validated_cumulative_peak(699, [500, 600, 700]).is_err());
        assert!(validated_cumulative_peak(0, [0, 0, 0]).is_err());
    }

    #[test]
    fn ltx25_output_frames_require_exact_geometry_and_rgb_payloads() {
        let rgb = 768 * 512 * 3;
        validate_ltx25_rgb_frames(2, 768, 512, &[(768, 512, rgb), (768, 512, rgb)]).unwrap();
        assert!(validate_ltx25_rgb_frames(1, 768, 512, &[]).is_err());
        assert!(validate_ltx25_rgb_frames(2, 768, 512, &[(768, 512, rgb)]).is_err());
        assert!(validate_ltx25_rgb_frames(1, 768, 512, &[(512, 768, rgb)]).is_err());
        assert!(validate_ltx25_rgb_frames(1, 768, 512, &[(768, 512, 0)]).is_err());
        assert!(validate_ltx25_rgb_frames(1, 768, 512, &[(768, 512, rgb - 1)]).is_err());
    }

    #[test]
    fn unix_epoch_formats_as_rfc3339() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_662), (2026, 7, 28));
        assert!(captured_at().ends_with('Z'));
    }

    #[test]
    fn parameter_rejects_missing_or_non_integer_values() {
        let request = json!({
            "planned": { "strategy": { "parameters": { "decodeTileEdge": 512 } } }
        });
        assert_eq!(parameter(&request, "decodeTileEdge").unwrap(), 512);
        assert!(parameter(&request, "decodeOverlap").is_err());
    }

    #[test]
    fn plain_adapter_settles_a_none_overlay_truthfully() {
        let request = json!({ "planned": { "target": { "overlay": "none" } } });
        let mut fragment = json!({
            "scenarios": [
                { "name": "loadability", "result": "passed" },
                { "name": "overlay", "result": "not_run", "reason": "unsettled" }
            ]
        });

        settle_plain_overlay_scenario(&request, &mut fragment, "the Qwen VAE-only path").unwrap();

        assert_eq!(fragment["scenarios"][1]["result"], "not_applicable");
        let reason = fragment["scenarios"][1]["reason"].as_str().unwrap();
        assert!(reason.contains("target declares overlay \"none\""));
        assert!(reason.contains("Qwen VAE-only path"));
    }

    #[test]
    fn plain_adapter_fails_closed_for_every_material_overlay() {
        for overlay in ["lora", "identity", "control", "control:1"] {
            let request = json!({ "planned": { "target": { "overlay": overlay } } });
            let mut fragment = json!({
                "scenarios": [
                    { "name": "overlay", "result": "not_run", "reason": "unsettled" }
                ]
            });
            let before = fragment.clone();

            let error =
                settle_plain_overlay_scenario(&request, &mut fragment, "the Qwen VAE-only path")
                    .unwrap_err();

            assert!(error.contains(overlay));
            assert!(error.contains("refusing"));
            assert_eq!(fragment, before, "a refusal must not become false coverage");
        }
    }

    #[test]
    fn fresh_reference_sweep_preserves_parameters_without_fabricating_empty_rungs() {
        let bounded = json!({
            "planned": {
                "strategy": {
                    "rung": "bounded_decode",
                    "parameters": { "decodeTileEdge": 512, "decodeOverlap": 64 }
                }
            }
        });
        let bounded_sweep = reference_sweep(&bounded, "passed").unwrap();
        assert_eq!(
            bounded_sweep.pointer("/cases/0/parameters"),
            bounded.pointer("/planned/strategy/parameters")
        );
        assert_eq!(
            optional_parameter(&bounded, "decodeTileEdge").unwrap(),
            Some(512)
        );
        assert_eq!(
            optional_parameter(&bounded, "attentionChunkSize").unwrap(),
            None
        );

        let resident = json!({
            "planned": { "strategy": { "rung": "resident", "parameters": {} } }
        });
        let resident_sweep = reference_sweep(&resident, "passed").unwrap();
        assert_eq!(resident_sweep.pointer("/axes"), Some(&json!([])));
        assert_eq!(
            resident_sweep.pointer("/cases/0/parameters"),
            Some(&json!({}))
        );
        assert_eq!(
            resident_sweep.pointer("/rangeVerified"),
            Some(&json!(false))
        );
    }

    #[test]
    fn candle_reference_progress_sequences_cover_all_five_rungs() {
        fn phases(boundaries: &[ReferenceBoundary]) -> Vec<ReferencePhase> {
            let mut phase = ReferencePhase::Conditioning;
            let mut visited = vec![phase];
            for &boundary in boundaries {
                if let Some(next) = next_reference_phase(phase, boundary) {
                    phase = next;
                    visited.push(phase);
                }
            }
            visited
        }

        let complete = vec![
            ReferencePhase::Conditioning,
            ReferencePhase::Denoise,
            ReferencePhase::Decode,
        ];
        assert_eq!(
            phases(&[
                ReferenceBoundary::FirstDenoiseStep,
                ReferenceBoundary::Decoding,
            ]),
            complete,
            "resident has no provider loading boundary"
        );
        assert_eq!(
            phases(&[ReferenceBoundary::RendererLoad, ReferenceBoundary::Decoding,]),
            complete,
            "staged residency exposes one Renderer boundary"
        );
        for rung in ["bounded_decode", "bounded_attention", "bounded_transformer"] {
            assert_eq!(
                phases(&[
                    ReferenceBoundary::RendererLoad,
                    ReferenceBoundary::RendererLoad,
                    ReferenceBoundary::Decoding,
                ]),
                complete,
                "{rung} uses the three-stage provider path"
            );
        }
    }

    #[test]
    fn exact_overlay_guard_rejects_every_other_record_identity() {
        for overlay in ["none", "lora", "identity", "control", "control:2"] {
            let request = json!({ "planned": { "target": { "overlay": overlay } } });
            let error = validate_exact_overlay_target(
                &request,
                "control:1",
                "the MLX Krea pose-control path",
            )
            .unwrap_err();
            assert!(error.contains(overlay));
            assert!(error.contains("control:1"));
            assert!(error.contains("refusing"));
        }
        let matching = json!({ "planned": { "target": { "overlay": "control:1" } } });
        assert!(validate_exact_overlay_target(
            &matching,
            "control:1",
            "the MLX Krea pose-control path"
        )
        .is_ok());
    }

    #[test]
    fn plain_gated_fragment_cannot_leave_overlay_not_run() {
        let request = json!({ "planned": { "target": { "overlay": "none" } } });
        let fragment = plain_gated_fragment(
            &request,
            "the Qwen VAE-only path",
            PlainGatedFragment {
                artifact: json!({}),
                sweep: json!({}),
                blocker: "another incomplete scenario",
                quality: Value::Null,
                negative_mutation: Value::Null,
                loadability: Value::Null,
                diagnostics: json!({}),
            },
        )
        .unwrap();
        let overlay = fragment["scenarios"]
            .as_array()
            .unwrap()
            .iter()
            .find(|scenario| scenario["name"] == "overlay")
            .unwrap();
        assert_eq!(overlay["result"], "not_applicable");
        assert_ne!(overlay["result"], "not_run");
    }

    #[test]
    fn identical_output_comparison_is_unmodified_without_a_negative_bias() {
        let left = [0.25_f32, -0.5, 1.0];
        let right = [0.25_f32, -0.49, 0.98];
        let (maximum, mean) = max_mean_abs(&left, &right, None).unwrap();
        assert!((maximum - 0.02).abs() < 1e-6);
        assert!((mean - 0.01).abs() < 1e-6);
    }

    #[test]
    fn deterministic_output_bias_forces_a_measured_parity_breach() {
        let left = [0.25_f32, -0.5, 1.0];
        let right = [0.25_f32, -0.49, 0.98];
        let (maximum, mean) = max_mean_abs(&left, &right, Some(0.05)).unwrap();
        assert!(maximum > 0.03);
        assert!(mean > 0.003);
        assert!((maximum - 0.07).abs() < 1e-6);
        assert!((mean - 0.06).abs() < 1e-6);
    }

    #[test]
    fn comparison_rejects_non_finite_samples() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_mean_abs(&[non_finite], &[0.0], None)
                .unwrap_err()
                .contains("non-finite sample"));
            assert!(max_mean_abs(&[0.0], &[non_finite], Some(0.05))
                .unwrap_err()
                .contains("non-finite sample"));
        }
    }

    #[test]
    fn comparison_rejects_a_non_finite_computed_accumulator() {
        let error = max_mean_abs(&[0.0, 0.0], &[0.0, 0.0], Some(f64::MAX)).unwrap_err();
        assert!(error.contains("mean accumulator became non-finite"));
    }

    #[test]
    fn comparison_shapes_must_match_exactly_before_flattening() {
        assert!(validate_comparison_shapes(&[1, 4, 4, 3], &[1, 4, 4, 3]).is_ok());
        let error = validate_comparison_shapes(&[1, 4, 4, 3], &[1, 8, 2, 3]).unwrap_err();
        assert!(error.contains("shape mismatch"));
    }

    #[test]
    fn comparison_bias_is_required_only_for_expected_failures() {
        let positive = json!({ "decodeTileEdge": 256, "decodeOverlap": 32 })
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(comparison_output_bias(&positive, false).unwrap(), None);
        assert!(comparison_output_bias(&positive, true).is_err());

        let negative = json!({
            "decodeTileEdge": 256,
            "decodeOverlap": 32,
            "comparisonOutputBias": 0.05
        })
        .as_object()
        .unwrap()
        .clone();
        assert_eq!(comparison_output_bias(&negative, true).unwrap(), Some(0.05));
        assert!(comparison_output_bias(&negative, false).is_err());
    }

    #[test]
    fn still_geometry_guard_refuses_every_non_still_axis_and_reproduces_the_pinned_wording() {
        let still = json!({
            "planned": { "target": { "geometry": { "width": 768, "height": 768, "batch": 1, "frames": 1 } } }
        });
        assert!(validate_still_geometry(&still, "MLX Krea base calibration").is_ok());

        for (axis, value) in [
            ("frames", 2_u64),
            ("batch", 2),
            ("frames", 97),
            ("batch", 0),
        ] {
            let mut request = still.clone();
            request["planned"]["target"]["geometry"][axis] = json!(value);
            let error = validate_still_geometry(&request, "MLX SDXL base calibration")
                .expect_err("an image arm must refuse a non-still geometry");
            assert_eq!(
                error,
                format!("MLX SDXL base calibration requires geometry.{axis} == 1, got {value}")
            );
        }

        // A missing or non-integer axis fails closed rather than defaulting to the still value.
        for axis in ["frames", "batch"] {
            let mut request = still.clone();
            request["planned"]["target"]["geometry"][axis] = json!("1");
            assert!(
                validate_still_geometry(&request, "MLX Qwen base calibration")
                    .unwrap_err()
                    .contains(&format!(
                        "planned.target.geometry.{axis} must be an integer"
                    ))
            );
        }
    }

    #[test]
    fn qwen_identity_rejects_wrong_repository_revision_and_root() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let root = Path::new(
            "/cache/models--SceneWorks--qwen-image-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/bf16",
        );
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_snapshot_root(
            root,
            "Qwen/Qwen-Image",
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789ABCDEF0123456789abcdef01234567",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789abcdef0123456789abcdef0123456g",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789abcdef",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            Path::new(
                "/cache/models--SceneWorks--qwen-image-mlx/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/bf16",
            ),
            QWEN_REPOSITORY,
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
    }

    #[test]
    fn krea_identity_rejects_wrong_repository_revision_and_root() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let root = Path::new(
            "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/q4",
        );
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_snapshot_root(
            root,
            "SceneWorks/krea-2-turbo-candle",
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789abcdef0123456789abcdef0123456g",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789ABCDEF0123456789abcdef01234567",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789abcdef",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            Path::new(
                "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/bf16",
            ),
            KREA_REPOSITORY,
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
    }

    #[test]
    fn sdxl_identity_rejects_wrong_repository_revision_and_root() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let root = Path::new(
            "/cache/models--SceneWorks--sdxl-base-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/q8",
        );
        assert!(validate_huggingface_snapshot_root(
            root,
            SDXL_REPOSITORY,
            revision,
            "q8",
            SDXL_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_snapshot_root(
            root,
            "stabilityai/stable-diffusion-xl-base-1.0",
            revision,
            "q8",
            SDXL_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            Path::new(
                "/cache/models--SceneWorks--sdxl-base-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/q4",
            ),
            SDXL_REPOSITORY,
            revision,
            "q8",
            SDXL_REPOSITORY
        )
        .is_err());
    }

    /// The `mlx:minimax_h3` capture resolves TWO artifact triples, and the two are validated by
    /// DIFFERENT shapes: the rehost carries a tier sub-directory, the upstream snapshot does not.
    /// Each validator must refuse the other's root, or a capture could stage the shared partitions
    /// out of the tier tree (or the DiT out of the upstream tree) and still pass identity.
    #[test]
    fn minimax_identity_separates_the_tiered_rehost_from_the_upstream_snapshot() {
        let rehost_revision = "137ce668c55a20bc0935fd1cf2a3de8448abb7f4";
        let upstream_revision = "939557dc319dd91227e30195a763f272ba7f8765";
        let tier_root = Path::new(
            "/cache/models--SceneWorks--minimax-h3-mlx/snapshots/137ce668c55a20bc0935fd1cf2a3de8448abb7f4/q4",
        );
        let upstream_root = Path::new(
            "/cache/models--MiniMaxAI--MiniMax-H3/snapshots/939557dc319dd91227e30195a763f272ba7f8765",
        );

        assert!(validate_huggingface_snapshot_root(
            tier_root,
            MINIMAX_REPOSITORY,
            rehost_revision,
            "q4",
            MINIMAX_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_revision_root(
            upstream_root,
            MINIMAX_UPSTREAM_REPOSITORY,
            upstream_revision,
            MINIMAX_UPSTREAM_REPOSITORY
        )
        .is_ok());

        // The upstream snapshot is not a tier root, and the tier root is not a snapshot root.
        assert!(validate_huggingface_revision_root(
            tier_root,
            MINIMAX_REPOSITORY,
            rehost_revision,
            MINIMAX_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            upstream_root,
            MINIMAX_UPSTREAM_REPOSITORY,
            upstream_revision,
            "bf16",
            MINIMAX_UPSTREAM_REPOSITORY
        )
        .is_err());

        // Neither repository may stand in for the other, and the tier component is exact.
        assert!(validate_huggingface_revision_root(
            upstream_root,
            MINIMAX_REPOSITORY,
            upstream_revision,
            MINIMAX_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            tier_root,
            MINIMAX_REPOSITORY,
            rehost_revision,
            "q8",
            MINIMAX_REPOSITORY
        )
        .is_err());
        // A component-staging path is one level too deep for either validator.
        assert!(validate_huggingface_revision_root(
            &upstream_root.join("text_encoder"),
            MINIMAX_UPSTREAM_REPOSITORY,
            upstream_revision,
            MINIMAX_UPSTREAM_REPOSITORY
        )
        .is_err());
    }

    /// sc-22724: the edit reference is one interleaved RGB frame at the request geometry, and it
    /// is deterministic — two captures of the same anchor condition on the same bytes.
    #[test]
    fn synthetic_reference_is_one_rgb_frame_at_the_request_geometry() {
        let pixels = synthetic_reference_rgb(64, 48);
        assert_eq!(pixels.len(), 64 * 48 * 3);
        assert_eq!(pixels, synthetic_reference_rgb(64, 48));
        assert_ne!(
            pixels[..3],
            pixels[pixels.len() - 3..],
            "a gradient, not a flat field"
        );
        assert_eq!(synthetic_reference_rgb(1, 1).len(), 3);
        assert_eq!(
            synthetic_reference_rgb(0, 0).len(),
            3,
            "a degenerate geometry still yields one pixel"
        );
    }
}
