//! Shared, weight-free protocol helpers for the real SC-15508 backend adapters.
//!
//! Backend binaries deliberately return `gated` until every required measurement and lifecycle
//! scenario has actually executed. A successful model call is not silently promoted into a complete
//! calibration record.

use serde_json::{json, Map, Value};
use std::io::{self, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const INFERENCE_PIN: &str = "4013049764172ee7dc707101c7da8c83c1483f2d";
pub const QWEN_REPOSITORY: &str = "SceneWorks/qwen-image-mlx";
pub const FLUX2_REPOSITORY: &str = "SceneWorks/flux2-dev-mlx";
pub const KREA_REPOSITORY: &str = "SceneWorks/krea-2-turbo-mlx";
pub const SDXL_REPOSITORY: &str = "SceneWorks/sdxl-base-mlx";
pub const Z_IMAGE_REPOSITORY: &str = "SceneWorks/z-image-turbo-mlx";
/// The `mlx:ltx_2_3` calibration artifact (sc-18808). Its `gemma/` co-requisite text encoder is a
/// hard load-time requirement of the pinned provider, not a fallback, so a capture resolves TWO
/// roots under this one repository: the numeric tier and `gemma`.
pub const LTX_REPOSITORY: &str = "SceneWorks/ltx-2.3-mlx";
/// The `candle:wan2_2_ti2v_5b` calibration artifact (sc-19057) — the FIRST candle video lane. The
/// SceneWorks-hosted candle snapshot carries the packed `q4/` and `q8/` tier subdirectories the
/// packed-detect loader reads; the dense `bf16` tier is the upstream `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
/// snapshot, which has no per-tier subdirectory and therefore no artifact identity this arm can bind.
pub const WAN_CANDLE_REPOSITORY: &str = "SceneWorks/wan2.2-ti2v-5b-candle";
/// The `mlx:minimax_h3` TIERED artifact (sc-18663). Unlike every repository above it, this rehost
/// is **not sufficient on its own**: `mlx_gen_minimax_h3::model::load` probes `vae/`, `audio_vae/`,
/// `tokenizer/` and `FL2VA/audio_vae/` under the spec's own weights root, and this rehost publishes
/// none of them — it carries only the per-tier `transformer/`, `transformer_ref` and (for q4/q8)
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

/// The geometry a VIDEO calibration arm renders, including the latent temporal depth its VAE
/// actually denoises over.
///
/// Deliberately the counterpart of [`validate_still_geometry`], not a replacement for it: an image
/// arm asserts the frames axis away, a video arm has to validate it against a real envelope. Both
/// halves live in this crate so a reader sees the whole per-arm contract in one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoGeometry {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    /// `1 + (frames - 1) / temporal_scale` — the video VAE's causal temporal depth.
    pub latent_frames: u32,
}

/// The declared envelope one video arm validates its geometry against.
///
/// Every field is a statement about a SPECIFIC model's shipped `limits` block plus its engine's own
/// hard rules, so the struct carries the arm's label and rationales rather than the guard inventing
/// wording. That is what lets two arms with genuinely different envelopes share one guard without
/// either of them reporting the other's reason.
pub struct VideoGeometryEnvelope<'a> {
    /// How the arm names itself in every refusal, e.g. `"MLX LTX-2.3 calibration"`.
    pub calibration_label: &'a str,
    /// `limits.resolutions`, as `(width, height)` pairs.
    pub resolutions: &'a [(u32, u32)],
    /// `limits.requiresDimensionsMultipleOf`, or the core default floor when the manifest elides it.
    pub dimension_multiple: u32,
    /// `limits.maxPixels`, when the model declares one. `None` for a model whose resolution list is
    /// its only spatial bound.
    pub max_pixels: Option<u64>,
    /// The engine's temporal lattice: the frame count must be `1 + temporal_scale * k`.
    pub temporal_scale: u32,
    /// Why that lattice exists, quoted into the refusal — e.g.
    /// `"the LTX video VAE is 8x causal in time"`.
    pub temporal_rationale: &'a str,
    /// The closed `[minimum, maximum]` frame span the declared `durations x fps` cross product can
    /// produce through the shipped frame ladder. Derived by the arm, never written down.
    pub frame_envelope: (u32, u32),
    /// Why the batch axis is pinned to 1, quoted into the refusal — e.g.
    /// `"the provider advertises max_count 1"`.
    pub batch_rationale: &'a str,
}

impl VideoGeometryEnvelope<'_> {
    pub fn declared_resolutions(&self) -> String {
        self.resolutions
            .iter()
            .map(|(width, height)| format!("{width}x{height}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Validate the three spatial/temporal axes a video arm renders, in the order a reader debugging a
/// refused plan row wants them: resolution, spatial lattice, area cap, temporal lattice, temporal
/// envelope.
///
/// A still geometry (`frames == 1`) sits ON every temporal lattice — `1 % k == 1` for any `k` — so
/// it can only be caught by the envelope floor. That is deliberate and load-bearing: a video arm
/// must never be able to capture a single-frame record for a video model and call it evidence.
pub fn validate_video_geometry(
    envelope: &VideoGeometryEnvelope<'_>,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<VideoGeometry, String> {
    let label = envelope.calibration_label;
    if envelope.temporal_scale == 0 {
        return Err(format!(
            "{label} declares a zero temporal scale, which describes no video VAE"
        ));
    }
    if !envelope.resolutions.contains(&(width, height)) {
        return Err(format!(
            "{label} requires one of the declared limits.resolutions ({}), got {width}x{height}",
            envelope.declared_resolutions()
        ));
    }
    if width % envelope.dimension_multiple != 0 || height % envelope.dimension_multiple != 0 {
        return Err(format!(
            "{label} requires geometry divisible by {}, got {width}x{height}",
            envelope.dimension_multiple
        ));
    }
    if let Some(maximum) = envelope.max_pixels {
        let area = u64::from(width) * u64::from(height);
        if area > maximum {
            return Err(format!(
                "{label} requires width*height within the declared limits.maxPixels {maximum}, got {area}"
            ));
        }
    }
    if frames % envelope.temporal_scale != 1 {
        return Err(format!(
            "{label} requires geometry.frames == 1 + {}k ({}), got {frames}",
            envelope.temporal_scale, envelope.temporal_rationale
        ));
    }
    let (minimum, maximum) = envelope.frame_envelope;
    if frames < minimum || frames > maximum {
        return Err(format!(
            "{label} requires geometry.frames within the declared duration/fps envelope \
             [{minimum}, {maximum}], got {frames}"
        ));
    }
    Ok(VideoGeometry {
        width,
        height,
        frames,
        latent_frames: 1 + (frames - 1) / envelope.temporal_scale,
    })
}

/// Read the four declared geometry axes for a VIDEO arm. Unlike [`validate_still_geometry`] this
/// reads `frames` as a real value rather than asserting it away; `batch` is still pinned to 1
/// because every video provider this apparatus drives renders exactly one clip.
pub fn video_target_geometry(
    request: &Value,
    envelope: &VideoGeometryEnvelope<'_>,
) -> Result<VideoGeometry, String> {
    let geometry = planned(request)?
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
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "{} requires geometry.batch == 1 ({}), got {batch}",
            envelope.calibration_label, envelope.batch_rationale
        ));
    }
    validate_video_geometry(envelope, axis("width")?, axis("height")?, axis("frames")?)
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
    validate_artifact_identity(repository, revision, expected_repository)?;
    let repository_component = format!("models--{}", expected_repository.replace('/', "--"));
    let expected = [
        repository_component.as_str(),
        "snapshots",
        revision,
        variant,
    ];
    let components = canonical_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "artifact root must end with /{repository_component}/snapshots/{revision}/{variant}"
        ));
    }
    Ok(())
}

/// Validate an artifact rooted at the Hugging Face snapshot itself rather than a tier directory.
/// MiniMax-H3's upstream shared components are published in exactly this variant-free shape.
pub fn validate_huggingface_revision_root(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    expected_repository: &str,
) -> Result<(), String> {
    validate_artifact_identity(repository, revision, expected_repository)?;
    let repository_component = format!("models--{}", expected_repository.replace('/', "--"));
    let expected = [repository_component.as_str(), "snapshots", revision];
    let components = canonical_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "artifact root must end with /{repository_component}/snapshots/{revision}"
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The MLX LTX-2.3 envelope, transcribed from `bin/mlx.rs` so this crate can pin — on EVERY
    /// host — the exact sentences the shared guard emits for that arm.
    ///
    /// 🔴 A transcription proves nothing about the arm it was transcribed from: on its own this
    /// function pins the GUARD's wording against a copy, and would stay green if `bin/mlx.rs`
    /// reworded `temporal_rationale` or relabelled itself. What closes that gap is a separate,
    /// always-run node gate — `the shared video-geometry envelopes are transcribed verbatim into
    /// the always-compiled lib tests` in `scripts/platform-review-contracts.test.mjs` — which
    /// parses `ltx_video_envelope()` out of `bin/mlx.rs`, resolves the constants it names, and
    /// requires this copy to be field-for-field identical. The two together are what make the
    /// pinned sentences a real contract, since `bin/mlx.rs` compiles on exactly one host.
    fn ltx_envelope() -> VideoGeometryEnvelope<'static> {
        VideoGeometryEnvelope {
            calibration_label: "MLX LTX-2.3 calibration",
            resolutions: &[(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)],
            dimension_multiple: 64,
            max_pixels: None,
            temporal_scale: 8,
            temporal_rationale: "the LTX video VAE is 8x causal in time",
            frame_envelope: (97, 449),
            batch_rationale: "the provider advertises max_count 1",
        }
    }

    /// The Candle Wan2.2 TI2V-5B envelope, transcribed from `bin/candle.rs` for the same reason —
    /// that binary compiles only off macOS with the `candle` feature and a CUDA toolchain — and
    /// bound back to it by the same node gate named above.
    fn wan_envelope() -> VideoGeometryEnvelope<'static> {
        VideoGeometryEnvelope {
            calibration_label: "Candle Wan2.2 TI2V-5B calibration",
            resolutions: &[(832, 480), (1280, 704), (704, 1280)],
            dimension_multiple: 32,
            max_pixels: Some(901_120),
            temporal_scale: 4,
            temporal_rationale: "the Wan z48 video VAE is 4x causal in time",
            // The CAPTURABLE cadence only: `limits.fps` declares [16, 24] but the calibrated route
            // executes 24 alone, so the 16 fps rungs (61, 77, 109, 125) are outside what this arm
            // may capture. See `wan_frame_envelope()` in `bin/candle.rs`.
            frame_envelope: (93, 189),
            batch_rationale: "the provider advertises max_count 1",
        }
    }

    fn video_request(width: u64, height: u64, batch: u64, frames: u64) -> Value {
        json!({
            "planned": {
                "target": {
                    "geometry": {
                        "width": width, "height": height, "batch": batch, "frames": frames
                    }
                }
            }
        })
    }

    /// sc-19057: the video counterpart of
    /// `still_geometry_guard_refuses_every_non_still_axis_and_reproduces_the_pinned_wording`.
    ///
    /// Every message is asserted with `assert_eq!` against the exact sentence the MLX LTX arm
    /// emitted before the hoist, so a "tidy-up" of the shared guard's wording reds here rather than
    /// silently changing what an operator reads when a multi-hour capture is refused.
    #[test]
    fn the_video_geometry_guard_reproduces_the_pinned_ltx_wording_on_every_axis() {
        let envelope = ltx_envelope();
        let geometry = validate_video_geometry(&envelope, 768, 512, 97).unwrap();
        assert_eq!(
            geometry,
            VideoGeometry {
                width: 768,
                height: 512,
                frames: 97,
                latent_frames: 13,
            }
        );

        assert_eq!(
            validate_video_geometry(&envelope, 1024, 1024, 97).unwrap_err(),
            "MLX LTX-2.3 calibration requires one of the declared limits.resolutions \
             (768x512, 512x768, 640x640, 1280x704, 704x1280), got 1024x1024"
        );
        assert_eq!(
            validate_video_geometry(&envelope, 768, 512, 96).unwrap_err(),
            "MLX LTX-2.3 calibration requires geometry.frames == 1 + 8k (the LTX video VAE is 8x \
             causal in time), got 96"
        );
        assert_eq!(
            validate_video_geometry(&envelope, 768, 512, 457).unwrap_err(),
            "MLX LTX-2.3 calibration requires geometry.frames within the declared duration/fps \
             envelope [97, 449], got 457"
        );
        // A still geometry is ON the lattice (`1 % 8 == 1`); only the envelope floor catches it.
        assert_eq!(
            validate_video_geometry(&envelope, 768, 512, 1).unwrap_err(),
            "MLX LTX-2.3 calibration requires geometry.frames within the declared duration/fps \
             envelope [97, 449], got 1"
        );
        // The divisibility rule is redundant with the pair list for LTX, so reach it through an
        // envelope whose declared pair is deliberately off its own lattice.
        let off_lattice = VideoGeometryEnvelope {
            resolutions: &[(768, 500)],
            ..ltx_envelope()
        };
        assert_eq!(
            validate_video_geometry(&off_lattice, 768, 500, 97).unwrap_err(),
            "MLX LTX-2.3 calibration requires geometry divisible by 64, got 768x500"
        );

        assert_eq!(
            video_target_geometry(&video_request(768, 512, 2, 97), &envelope).unwrap_err(),
            "MLX LTX-2.3 calibration requires geometry.batch == 1 (the provider advertises \
             max_count 1), got 2"
        );
        assert_eq!(
            video_target_geometry(&video_request(768, 512, 1, 97), &envelope).unwrap(),
            geometry
        );
    }

    /// The Candle Wan half of the same guard: a different label, lattice, floor and — uniquely —
    /// an area cap, all reported in the arm's own words.
    #[test]
    fn the_video_geometry_guard_reports_the_candle_wan_envelope_in_its_own_words() {
        let envelope = wan_envelope();
        assert_eq!(
            validate_video_geometry(&envelope, 832, 480, 117).unwrap(),
            VideoGeometry {
                width: 832,
                height: 480,
                frames: 117,
                latent_frames: 30,
            }
        );
        assert_eq!(
            validate_video_geometry(&envelope, 768, 512, 117).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires one of the declared limits.resolutions \
             (832x480, 1280x704, 704x1280), got 768x512"
        );
        assert_eq!(
            validate_video_geometry(&envelope, 832, 480, 118).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires geometry.frames == 1 + 4k (the Wan z48 \
             video VAE is 4x causal in time), got 118"
        );
        assert_eq!(
            validate_video_geometry(&envelope, 832, 480, 193).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires geometry.frames within the declared \
             duration/fps envelope [93, 189], got 193"
        );
        // 61 frames is a real product geometry — 4s at 16 fps — and is refused here anyway, because
        // the calibrated route cannot execute the 16 fps cadence that reaches it.
        assert_eq!(
            validate_video_geometry(&envelope, 832, 480, 61).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires geometry.frames within the declared \
             duration/fps envelope [93, 189], got 61"
        );
        assert_eq!(
            validate_video_geometry(&envelope, 832, 480, 1).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires geometry.frames within the declared \
             duration/fps envelope [93, 189], got 1"
        );
        // The area cap is the one rule the resolution list does not already imply — reachable only
        // through an envelope that declares a pair above it, which is exactly the drift the cap
        // exists to refuse.
        let oversized = VideoGeometryEnvelope {
            resolutions: &[(1280, 1280)],
            ..wan_envelope()
        };
        assert_eq!(
            validate_video_geometry(&oversized, 1280, 1280, 117).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration requires width*height within the declared \
             limits.maxPixels 901120, got 1638400"
        );
    }

    /// The two guards are complements, not alternatives — the property the whole per-arm contract
    /// rests on. An image arm's geometry must be refused by the video guard and vice versa, so
    /// neither arm can be wired to the other's check and still look green.
    #[test]
    fn the_still_and_video_guards_refuse_each_other_geometries() {
        let still = video_request(832, 480, 1, 1);
        let video = video_request(832, 480, 1, 117);
        let envelope = wan_envelope();

        assert!(validate_still_geometry(&still, "Candle Krea base calibration").is_ok());
        assert!(video_target_geometry(&still, &envelope).is_err());

        assert!(video_target_geometry(&video, &envelope).is_ok());
        assert_eq!(
            validate_still_geometry(&video, "Candle Krea base calibration").unwrap_err(),
            "Candle Krea base calibration requires geometry.frames == 1, got 117"
        );
    }

    /// A malformed envelope must fail closed rather than divide by zero deriving the latent depth.
    #[test]
    fn a_zero_temporal_scale_envelope_is_refused_before_any_arithmetic() {
        let broken = VideoGeometryEnvelope {
            temporal_scale: 0,
            ..wan_envelope()
        };
        assert_eq!(
            validate_video_geometry(&broken, 832, 480, 117).unwrap_err(),
            "Candle Wan2.2 TI2V-5B calibration declares a zero temporal scale, which describes no \
             video VAE"
        );
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
}
