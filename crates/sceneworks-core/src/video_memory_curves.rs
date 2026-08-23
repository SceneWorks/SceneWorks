//! Backend-neutral fitted video-memory curves (sc-18829 / sc-19020, epic 18803).
//!
//! The producer persists one strict, lane-tagged container derived from promoted calibration
//! evidence. Consumers query it with the complete runtime identity and receive a phase-resolved
//! peak only when every identity and applicability dimension agrees. Absence, malformed data,
//! foreign lanes, stale closures, unsupported tiers/models/modes/rungs, load-shape drift, and
//! geometry outside the measured regressor hull all return `None`; the caller retains its existing
//! weights-plus-headroom floor.
//!
//! This module deliberately has no `gen-core` dependency. Both MLX and candle can consume the same
//! wire format by translating their own backend, rung, load-shape, and decode-regime types at the
//! edge.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::memory_calibration::StrategyRung;

pub const VIDEO_MEMORY_CURVE_SCHEMA_VERSION: u32 = 3;
pub const PACKAGED_VIDEO_MEMORY_CURVES: &str =
    include_str!("../../../docs/generated/video-memory-curves.json");
/// Immutable evidence sources compiled alongside the promoted curve bundle. Paths and record ids
/// are globally unique and bytes/digests must agree. A source record may remain unmodeled (such as
/// an under-sampled FPS), but every curve source record must match one complete selector exactly.
const PACKAGED_VIDEO_MEMORY_CURVE_SOURCES: &[(&str, &str)] = &[(
    "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json",
    include_str!("../../../docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
)];
#[cfg(test)]
const HISTORICAL_VIDEO_MEMORY_CURVE_FIT_PATH: &str =
    "docs/generated/ltx-temporal-form-fit-sc-18810.json";
const PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR: &str = "scripts/fit-ltx-temporal-form.mjs";
const VIDEO_MEMORY_CURVE_FIT_PREFIX: &str = "docs/generated/ltx-temporal-form-fit-sc-";
const VIDEO_MEMORY_CURVE_FIT_SUFFIX: &str = ".json";

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoMemoryCurveBundle {
    pub schema_version: u32,
    pub generated_by: String,
    pub source_fit: String,
    pub source_catalog: Vec<VideoCurveSourceCatalogEntry>,
    pub curves: Vec<VideoMemoryCurve>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoCurveSourceCatalogEntry {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoMemoryCurve {
    pub id: String,
    pub model_id: String,
    pub model_family: String,
    /// Resolved engine route. Catalog model ids and providers are independently keyed, so neither
    /// may stand in for the route selected for this request.
    pub route: String,
    pub provider: String,
    pub backend: VideoCurveBackend,
    pub tier: String,
    pub mode: String,
    /// Exact reference carrier and count measured by this curve. Future I2V/reference stories add
    /// their own rows; this prevents the reference-free surface lending its fitted peak to them.
    pub reference_shape: String,
    pub reference_count: u32,
    /// Enumerated output rates present in sealed source receipts. An unmeasured interior rate is
    /// not evidence merely because two endpoints exist.
    pub frames_per_second: Vec<u32>,
    pub overlay: Option<String>,
    pub rung: StrategyRung,
    pub load_shape: VideoCurveLoadShape,
    pub batch: u32,
    pub closure_digest: String,
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub decode_pass: VideoCurveDecodePass,
    pub measured_geometry_hull: Vec<VideoCurveHullPoint>,
    pub phases: VideoPhaseCurves,
    pub evidence: VideoCurveEvidenceSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCurveBackend {
    Mlx,
    Candle,
}

impl VideoCurveBackend {
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCurveLoadShape {
    EagerMaterialization,
    DeferredMaterialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoCurveDecodePass {
    SinglePass,
    Tiled,
    Unmodelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoCurveHullPoint {
    pub pixels: u64,
    pub voxels: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoPhaseCurves {
    pub conditioning: VideoAffineCrossCurve,
    pub denoise: VideoAffineCrossCurve,
    pub decode: VideoAffineCrossCurve,
}

/// One phase's ratified affine cross law:
/// `fixedGb + perMpxGb * mpx + perMpxFrameGb * (mpx * frames)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoAffineCrossCurve {
    pub fixed_gb: f64,
    pub per_mpx_gb: f64,
    pub per_mpx_frame_gb: f64,
    /// Worst fit/held-out absolute residual. Runtime evaluation adds this phase-local conservative
    /// floor before taking the max over phases; the shared selector then retains its ordinary
    /// backend estimate margin over that residual-bounded prediction.
    pub max_residual_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoCurveEvidenceSummary {
    pub records: u32,
    pub fit_points: u32,
    pub held_out_points: u32,
    pub sources: Vec<VideoCurveEvidenceSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoCurveEvidenceSource {
    pub path: String,
    pub sha256: String,
    pub record_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoCurveGeometry {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub batch: u32,
}

/// Complete runtime identity for one fitted-curve lookup.
#[derive(Debug, Clone, Copy)]
pub struct VideoCurveQuery<'a> {
    pub model_id: &'a str,
    pub model_family: &'a str,
    pub route: &'a str,
    pub provider: &'a str,
    pub backend: VideoCurveBackend,
    pub tier: &'a str,
    pub mode: &'a str,
    pub reference_shape: &'a str,
    pub reference_count: u32,
    pub frames_per_second: u32,
    /// Content-independent workload axes only. Request-specific byte seals and asset ids must be
    /// removed before constructing a query so equal memory shapes can reuse one fitted curve.
    pub overlay: Option<&'a str>,
    pub rung: StrategyRung,
    pub load_shape: VideoCurveLoadShape,
    pub closure_digest: &'a str,
    pub calibration_abi: u32,
    pub calibration_fingerprint: &'a str,
    pub decode_pass: VideoCurveDecodePass,
    pub geometry: VideoCurveGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoPhasePeakBytes {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
}

impl VideoPhasePeakBytes {
    pub const fn peak(self) -> u64 {
        let mut peak = self.conditioning;
        if self.denoise > peak {
            peak = self.denoise;
        }
        if self.decode > peak {
            peak = self.decode;
        }
        peak
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoCurveEvaluation<'a> {
    pub curve_id: &'a str,
    pub closure_digest: &'a str,
    pub phases: VideoPhasePeakBytes,
}

impl VideoMemoryCurveBundle {
    /// Evaluate the unique curve matching the complete query. Any ambiguity or mismatch fails
    /// closed to `None`; callers must keep their pre-existing floor candidate.
    pub fn evaluate(&self, query: VideoCurveQuery<'_>) -> Option<VideoCurveEvaluation<'_>> {
        if self.schema_version != VIDEO_MEMORY_CURVE_SCHEMA_VERSION
            || self.generated_by != PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR
            || !is_canonical_video_memory_curve_fit_path(&self.source_fit)
        {
            return None;
        }
        let sources = packaged_source_handshakes().ok()?;
        let model_families = packaged_model_families().ok()?;
        validate_video_memory_curve_bundle_ref(self, sources, model_families).ok()?;
        let mut matches = self.curves.iter().filter(|curve| {
            curve.model_id == query.model_id
                && curve.model_family == query.model_family
                && curve.route == query.route
                && curve.provider == query.provider
                && curve.backend == query.backend
                && curve.tier == query.tier
                && curve.mode == query.mode
                && curve.reference_shape == query.reference_shape
                && curve.reference_count == query.reference_count
                && curve.frames_per_second.contains(&query.frames_per_second)
                && curve.overlay.as_deref() == query.overlay
                && curve.rung == query.rung
                && curve.load_shape == query.load_shape
                && curve.batch == query.geometry.batch
                && curve.closure_digest == query.closure_digest
                && curve.calibration_abi == query.calibration_abi
                && curve.calibration_fingerprint == query.calibration_fingerprint
                && curve.decode_pass == query.decode_pass
        });
        let curve = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        let pixels =
            u64::from(query.geometry.width).checked_mul(u64::from(query.geometry.height))?;
        let voxels = pixels.checked_mul(u64::from(query.geometry.frames))?;
        if query.geometry.width == 0
            || query.geometry.height == 0
            || query.geometry.frames == 0
            || !point_is_in_convex_hull(
                &curve.measured_geometry_hull,
                VideoCurveHullPoint { pixels, voxels },
            )
        {
            return None;
        }
        let evaluate = |phase: VideoAffineCrossCurve| phase.evaluate(pixels, query.geometry.frames);
        Some(VideoCurveEvaluation {
            curve_id: &curve.id,
            closure_digest: &curve.closure_digest,
            phases: VideoPhasePeakBytes {
                conditioning: evaluate(curve.phases.conditioning)?,
                denoise: evaluate(curve.phases.denoise)?,
                decode: evaluate(curve.phases.decode)?,
            },
        })
    }
}

impl VideoAffineCrossCurve {
    fn is_valid(self) -> bool {
        [
            self.fixed_gb,
            self.per_mpx_gb,
            self.per_mpx_frame_gb,
            self.max_residual_gb,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }

    fn evaluate(self, pixels: u64, frames: u32) -> Option<u64> {
        if !self.is_valid() || pixels == 0 || frames == 0 {
            return None;
        }
        let mpx = pixels as f64 / 1_000_000.0;
        let gib = self.fixed_gb
            + self.per_mpx_gb * mpx
            + self.per_mpx_frame_gb * mpx * f64::from(frames)
            + self.max_residual_gb;
        let bytes = (gib * BYTES_PER_GIB).ceil();
        (gib.is_finite() && gib >= 0.0 && bytes.is_finite() && bytes < u64::MAX as f64)
            .then_some(bytes as u64)
    }
}

/// Strict parse plus invariants serde cannot express. No partial bundle is usable.
pub fn load_video_memory_curves(raw: &str) -> Result<VideoMemoryCurveBundle, String> {
    let bundle: VideoMemoryCurveBundle = serde_json::from_str(raw)
        .map_err(|error| format!("video-memory curves do not parse: {error}"))?;
    let sources = packaged_source_handshakes()?;
    let model_families = packaged_model_families()?;
    validate_video_memory_curve_bundle(bundle, sources, model_families)
}

/// Validate a parsed bundle against the immutable source bytes and catalog model identities that
/// accompany it. Keeping this separate from the packaged `include_str!` lookup makes the
/// multi-source invariants directly testable: the production entry point still supplies only the
/// sources compiled into this crate.
fn validate_video_memory_curve_bundle(
    bundle: VideoMemoryCurveBundle,
    sources: &BTreeMap<String, PackagedSourceHandshake>,
    model_families: &BTreeMap<String, String>,
) -> Result<VideoMemoryCurveBundle, String> {
    validate_video_memory_curve_bundle_ref(&bundle, sources, model_families)?;
    Ok(bundle)
}

fn validate_video_memory_curve_bundle_ref(
    bundle: &VideoMemoryCurveBundle,
    sources: &BTreeMap<String, PackagedSourceHandshake>,
    model_families: &BTreeMap<String, String>,
) -> Result<(), String> {
    if bundle.schema_version != VIDEO_MEMORY_CURVE_SCHEMA_VERSION {
        return Err(format!(
            "video-memory curve schema {} is not supported (expected {})",
            bundle.schema_version, VIDEO_MEMORY_CURVE_SCHEMA_VERSION
        ));
    }
    if bundle.curves.is_empty() {
        return Err("video-memory curve bundle is empty".to_owned());
    }
    if bundle.generated_by != PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR
        || !is_canonical_video_memory_curve_fit_path(&bundle.source_fit)
    {
        return Err("video-memory curve bundle provenance is incomplete".to_owned());
    }
    validate_source_catalog(bundle, sources)?;
    let mut ids = BTreeSet::new();
    let mut selectors = BTreeSet::new();
    let mut referenced_records = BTreeSet::new();
    for curve in &bundle.curves {
        if !curve_is_valid(curve) {
            return Err(format!("video-memory curve {:?} is invalid", curve.id));
        }
        validate_curve_evidence(curve, sources, model_families)?;
        if !ids.insert(curve.id.clone()) {
            return Err(format!("duplicate video-memory curve id {:?}", curve.id));
        }
        let selector = format!(
            "{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}\0{:?}\0{:?}\0{:?}\0{:?}\0{}\0{}\0{}\0{:?}\0{}",
            curve.model_id,
            curve.model_family,
            curve.route,
            curve.provider,
            curve.backend.as_key(),
            curve.tier,
            curve.mode,
            curve.reference_shape,
            curve.reference_count,
            curve.frames_per_second,
            curve.overlay,
            curve.rung,
            curve.load_shape,
            curve.closure_digest,
            curve.calibration_abi,
            curve.calibration_fingerprint,
            curve.decode_pass,
            curve.batch,
        );
        if !selectors.insert(selector) {
            return Err(format!(
                "video-memory curve {:?} duplicates another curve's complete selector",
                curve.id
            ));
        }
        for source in &curve.evidence.sources {
            for record_id in &source.record_ids {
                if !referenced_records.insert((source.path.clone(), record_id.clone())) {
                    return Err(format!(
                        "video-memory evidence record {:?} from {:?} feeds more than one curve",
                        record_id, source.path
                    ));
                }
            }
        }
    }
    Ok(())
}

fn is_canonical_video_memory_curve_fit_path(path: &str) -> bool {
    path.strip_prefix(VIDEO_MEMORY_CURVE_FIT_PREFIX)
        .and_then(|story| story.strip_suffix(VIDEO_MEMORY_CURVE_FIT_SUFFIX))
        .is_some_and(|story| !story.is_empty() && story.bytes().all(|byte| byte.is_ascii_digit()))
}

#[derive(Debug)]
struct PackagedSourceHandshake {
    evidence_sha256: String,
    records: Vec<Value>,
}

/// The immutable source handshake is computed from the evidence compiled into this crate, not
/// trusted from the promoted artifact. The producer and consumer both hash the exact UTF-8 bytes of
/// the committed source evidence file compiled into this crate.
fn packaged_source_handshakes() -> Result<&'static BTreeMap<String, PackagedSourceHandshake>, String>
{
    static HANDSHAKES: OnceLock<Result<BTreeMap<String, PackagedSourceHandshake>, String>> =
        OnceLock::new();
    HANDSHAKES
        .get_or_init(|| source_handshakes(PACKAGED_VIDEO_MEMORY_CURVE_SOURCES.iter().copied()))
        .as_ref()
        .map_err(Clone::clone)
}

fn source_handshakes<'a>(
    inputs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<BTreeMap<String, PackagedSourceHandshake>, String> {
    let mut sources = BTreeMap::new();
    let mut global_record_ids = BTreeSet::new();
    for (path, raw) in inputs {
        let handshake = compute_packaged_source_handshake(raw)?;
        for record in &handshake.records {
            let id = record
                .get("id")
                .and_then(Value::as_str)
                .expect("the source handshake validated every record id");
            if !global_record_ids.insert(id.to_owned()) {
                return Err(format!(
                    "packaged video curve record id {id:?} appears in multiple sources"
                ));
            }
        }
        if sources.insert(path.to_owned(), handshake).is_some() {
            return Err(format!(
                "packaged video curve source path {path:?} appears more than once"
            ));
        }
    }
    if sources.is_empty() {
        return Err("packaged video curve source catalog is empty".to_owned());
    }
    Ok(sources)
}

fn compute_packaged_source_handshake(raw: &str) -> Result<PackagedSourceHandshake, String> {
    let source: Value = serde_json::from_str(raw)
        .map_err(|error| format!("packaged video curve source evidence does not parse: {error}"))?;
    let records = source
        .get("records")
        .and_then(Value::as_array)
        .ok_or_else(|| "packaged video curve source evidence has no records array".to_owned())?;
    let mut ids = records
        .iter()
        .map(|record| {
            record
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    "packaged video curve source evidence has a record without an id".to_owned()
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ids.sort();
    if ids.is_empty()
        || ids.iter().any(|id| !is_memory_record_id(id))
        || !ids.windows(2).all(|pair| pair[0] < pair[1])
    {
        return Err("packaged video curve source evidence has invalid record ids".to_owned());
    }
    Ok(PackagedSourceHandshake {
        evidence_sha256: format!("{:x}", Sha256::digest(raw.as_bytes())),
        records: records.clone(),
    })
}

fn validate_source_catalog(
    bundle: &VideoMemoryCurveBundle,
    packaged: &BTreeMap<String, PackagedSourceHandshake>,
) -> Result<(), String> {
    if bundle.source_catalog.len() != packaged.len()
        || !bundle
            .source_catalog
            .windows(2)
            .all(|pair| pair[0].path < pair[1].path)
    {
        return Err(
            "video-memory curve source catalog is not the exact sorted packaged catalog".to_owned(),
        );
    }
    for entry in &bundle.source_catalog {
        let source = packaged.get(&entry.path).ok_or_else(|| {
            format!(
                "video-memory curve source {:?} is not compiled into this crate",
                entry.path
            )
        })?;
        if !is_sha256(&entry.sha256) || entry.sha256 != source.evidence_sha256 {
            return Err(format!(
                "video-memory curve source {:?} digest does not match its compiled bytes",
                entry.path
            ));
        }
    }
    Ok(())
}

fn value_at_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))?
        .as_str()
}

fn value_at_u64(value: &Value, path: &[&str]) -> Option<u64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))?
        .as_u64()
}

fn packaged_model_families() -> Result<&'static BTreeMap<String, String>, String> {
    static FAMILIES: OnceLock<Result<BTreeMap<String, String>, String>> = OnceLock::new();
    FAMILIES
        .get_or_init(|| {
            let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .map(|(_, contents)| *contents)
                .ok_or_else(|| "builtin.models.jsonc is not compiled into sceneworks-core".to_owned())?;
            let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw))
                .map_err(|error| format!("builtin.models.jsonc does not parse: {error}"))?;
            let models = manifest
                .get("models")
                .and_then(Value::as_array)
                .ok_or_else(|| "builtin.models.jsonc has no models array".to_owned())?;
            let mut families = BTreeMap::new();
            for model in models {
                let Some(id) = model.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(family) = model.get("family").and_then(Value::as_str) else {
                    continue;
                };
                if families.insert(id.to_owned(), family.to_owned()).is_some() {
                    return Err(format!(
                        "builtin.models.jsonc repeats model id {id:?}; video curve family identity is ambiguous"
                    ));
                }
            }
            Ok(families)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn rung_key(rung: StrategyRung) -> &'static str {
    match rung {
        StrategyRung::Resident => "resident",
        StrategyRung::StagedResidency => "staged_residency",
        StrategyRung::BoundedDecode => "bounded_decode",
        StrategyRung::BoundedAttention => "bounded_attention",
        StrategyRung::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

fn load_shape_key(load_shape: VideoCurveLoadShape) -> &'static str {
    match load_shape {
        VideoCurveLoadShape::EagerMaterialization => "eager_materialization",
        VideoCurveLoadShape::DeferredMaterialization => "deferred_materialization",
    }
}

fn decode_pass_key(record: &Value) -> Option<&'static str> {
    let measurements = record.get("diagnostics")?.get("measurements")?.as_array()?;
    let engaged = measurements.iter().find(|measurement| {
        measurement.get("name").and_then(Value::as_str) == Some("decodeTilingEngaged")
    })?;
    match engaged.get("value").and_then(Value::as_u64)? {
        0 => Some("single_pass"),
        1 => Some("tiled"),
        _ => None,
    }
}

fn curve_decode_pass_key(pass: VideoCurveDecodePass) -> &'static str {
    match pass {
        VideoCurveDecodePass::SinglePass => "single_pass",
        VideoCurveDecodePass::Tiled => "tiled",
        VideoCurveDecodePass::Unmodelled => "unmodelled",
    }
}

fn source_reference_count(record: &Value) -> Option<u32> {
    value_at_u64(record, &["target", "referenceCount"])
        .and_then(|count| u32::try_from(count).ok())
        // Schema-v2 T2V receipts predate an explicit reference count. They are unambiguous only
        // because the sealed mode is reference-free; do not extend that compatibility to any
        // other mode.
        .or_else(|| {
            (value_at_str(record, &["target", "mode"]) == Some("text_to_video")).then_some(0)
        })
}

fn source_route(record: &Value) -> Option<&str> {
    value_at_str(record, &["target", "route"])
        .or_else(|| value_at_str(record, &["target", "provider"]))
}

fn source_reference_shape(record: &Value, reference_count: u32) -> Option<&str> {
    value_at_str(record, &["target", "referenceShape"])
        .or_else(|| (reference_count == 0).then_some("none"))
}

fn source_output_fps(record: &Value) -> Option<u32> {
    record
        .get("diagnostics")?
        .get("measurements")?
        .as_array()?
        .iter()
        .find(|measurement| measurement.get("name").and_then(Value::as_str) == Some("outputFps"))?
        .get("value")?
        .as_u64()
        .and_then(|fps| u32::try_from(fps).ok())
}

fn source_overlay(record: &Value) -> Option<&str> {
    match value_at_str(record, &["target", "overlay"]) {
        Some("none") => None,
        overlay => overlay,
    }
}

fn source_record_matches_curve(
    record: &Value,
    curve: &VideoMemoryCurve,
    model_families: &BTreeMap<String, String>,
) -> bool {
    let Some(reference_count) = source_reference_count(record) else {
        return false;
    };
    value_at_str(record, &["target", "modelId"]) == Some(curve.model_id.as_str())
        && model_families.get(&curve.model_id).map(String::as_str)
            == Some(curve.model_family.as_str())
        && source_route(record) == Some(curve.route.as_str())
        && value_at_str(record, &["target", "provider"]) == Some(curve.provider.as_str())
        && value_at_str(record, &["backend"]) == Some(curve.backend.as_key())
        && value_at_str(record, &["target", "tier"]) == Some(curve.tier.as_str())
        && value_at_str(record, &["target", "mode"]) == Some(curve.mode.as_str())
        && source_reference_shape(record, reference_count) == Some(curve.reference_shape.as_str())
        && reference_count == curve.reference_count
        && source_output_fps(record)
            .map(|fps| curve.frames_per_second.binary_search(&fps).is_ok())
            .unwrap_or(false)
        && source_overlay(record) == curve.overlay.as_deref()
        && value_at_str(record, &["strategy", "rung"]) == Some(rung_key(curve.rung))
        && value_at_str(record, &["loadShape"]) == Some(load_shape_key(curve.load_shape))
        && value_at_u64(record, &["target", "geometry", "batch"]) == Some(u64::from(curve.batch))
        && value_at_str(record, &["repositories", "inference", "closureDigest"])
            == Some(curve.closure_digest.as_str())
        && value_at_str(record, &["calibrationFingerprint"])
            == Some(curve.calibration_fingerprint.as_str())
        && decode_pass_key(record) == Some(curve_decode_pass_key(curve.decode_pass))
}

fn validate_curve_evidence(
    curve: &VideoMemoryCurve,
    packaged: &BTreeMap<String, PackagedSourceHandshake>,
    model_families: &BTreeMap<String, String>,
) -> Result<(), String> {
    if curve.evidence.sources.is_empty()
        || !curve
            .evidence
            .sources
            .windows(2)
            .all(|pair| pair[0].path < pair[1].path)
    {
        return Err(format!(
            "video-memory curve {:?} evidence sources are not sorted",
            curve.id
        ));
    }
    let expected = expected_curve_evidence_sources(curve, packaged, model_families);
    if curve.evidence.sources != expected {
        return Err(format!(
            "video-memory curve {:?} record subsets do not exactly match all referenced sources",
            curve.id
        ));
    }
    let mut seen = BTreeSet::new();
    let mut total = 0_usize;
    for reference in &curve.evidence.sources {
        if reference.record_ids.iter().any(|id| !seen.insert(id)) {
            return Err(format!(
                "video-memory curve {:?} repeats an immutable record across sources",
                curve.id
            ));
        }
        total = total.saturating_add(reference.record_ids.len());
    }
    if total != usize::try_from(curve.evidence.records).unwrap_or(usize::MAX) {
        return Err(format!(
            "video-memory curve {:?} evidence count is inconsistent",
            curve.id
        ));
    }
    Ok(())
}

fn expected_curve_evidence_sources(
    curve: &VideoMemoryCurve,
    packaged: &BTreeMap<String, PackagedSourceHandshake>,
    model_families: &BTreeMap<String, String>,
) -> Vec<VideoCurveEvidenceSource> {
    packaged
        .iter()
        .filter_map(|(path, source)| {
            let mut record_ids = source
                .records
                .iter()
                .filter(|record| source_record_matches_curve(record, curve, model_families))
                .filter_map(|record| record.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect::<Vec<_>>();
            record_ids.sort();
            (!record_ids.is_empty()).then(|| VideoCurveEvidenceSource {
                path: path.clone(),
                sha256: source.evidence_sha256.clone(),
                record_ids,
            })
        })
        .collect()
}

fn curve_is_valid(curve: &VideoMemoryCurve) -> bool {
    !curve.id.is_empty()
        && !curve.model_id.is_empty()
        && !curve.model_family.is_empty()
        && !curve.route.is_empty()
        && !curve.provider.is_empty()
        && !curve.tier.is_empty()
        && !curve.mode.is_empty()
        && !curve.reference_shape.trim().is_empty()
        && (curve.reference_shape == "none") == (curve.reference_count == 0)
        && !curve.frames_per_second.is_empty()
        && curve.frames_per_second.iter().all(|fps| *fps > 0)
        && curve
            .frames_per_second
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && !curve.calibration_fingerprint.is_empty()
        && curve.batch > 0
        && is_sha256(&curve.closure_digest)
        && curve.calibration_abi == crate::memory_calibration::MEMORY_CALIBRATION_ABI
        && curve.evidence.records > 0
        && curve.evidence.fit_points > 0
        && curve.evidence.held_out_points > 0
        && curve.evidence.records
            == curve
                .evidence
                .fit_points
                .saturating_add(curve.evidence.held_out_points)
        && usize::try_from(curve.evidence.records).ok()
            == Some(
                curve
                    .evidence
                    .sources
                    .iter()
                    .map(|source| source.record_ids.len())
                    .sum(),
            )
        && !curve.evidence.sources.is_empty()
        && curve.evidence.sources.iter().all(|source| {
            !source.path.is_empty()
                && is_sha256(&source.sha256)
                && !source.record_ids.is_empty()
                && source.record_ids.iter().all(|id| is_memory_record_id(id))
                && source.record_ids.windows(2).all(|pair| pair[0] < pair[1])
        })
        && curve
            .evidence
            .sources
            .windows(2)
            .all(|pair| pair[0].path < pair[1].path)
        && curve.phases.conditioning.is_valid()
        && curve.phases.denoise.is_valid()
        && curve.phases.decode.is_valid()
        && hull_is_strictly_convex(&curve.measured_geometry_hull)
}

pub fn packaged_video_memory_curves() -> Option<&'static VideoMemoryCurveBundle> {
    static BUNDLE: OnceLock<Result<VideoMemoryCurveBundle, String>> = OnceLock::new();
    BUNDLE
        .get_or_init(|| load_video_memory_curves(PACKAGED_VIDEO_MEMORY_CURVES))
        .as_ref()
        .ok()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_memory_record_id(value: &str) -> bool {
    value.len() == 24
        && value.starts_with("imc-")
        && value[4..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn cross(
    origin: VideoCurveHullPoint,
    edge: VideoCurveHullPoint,
    point: VideoCurveHullPoint,
) -> Option<i128> {
    let ax = i128::from(edge.pixels) - i128::from(origin.pixels);
    let ay = i128::from(edge.voxels) - i128::from(origin.voxels);
    let bx = i128::from(point.pixels) - i128::from(origin.pixels);
    let by = i128::from(point.voxels) - i128::from(origin.voxels);
    ax.checked_mul(by)?.checked_sub(ay.checked_mul(bx)?)
}

fn hull_is_strictly_convex(hull: &[VideoCurveHullPoint]) -> bool {
    if hull.len() < 3
        || hull.iter().any(|point| {
            point.pixels == 0
                || point.voxels == 0
                || point.voxels % point.pixels != 0
                || point.voxels / point.pixels > u64::from(u32::MAX)
        })
    {
        return false;
    }
    let unique = hull
        .iter()
        .map(|point| (point.pixels, point.voxels))
        .collect::<BTreeSet<_>>();
    if unique.len() != hull.len() {
        return false;
    }
    let mut sign = 0_i8;
    for index in 0..hull.len() {
        let Some(turn) = cross(
            hull[index],
            hull[(index + 1) % hull.len()],
            hull[(index + 2) % hull.len()],
        ) else {
            return false;
        };
        if turn == 0 {
            return false;
        }
        let current = if turn > 0 { 1 } else { -1 };
        if sign != 0 && current != sign {
            return false;
        }
        sign = current;
    }
    true
}

fn point_is_in_convex_hull(hull: &[VideoCurveHullPoint], point: VideoCurveHullPoint) -> bool {
    if !hull_is_strictly_convex(hull) {
        return false;
    }
    let mut sign = 0_i8;
    for index in 0..hull.len() {
        let Some(side) = cross(hull[index], hull[(index + 1) % hull.len()], point) else {
            return false;
        };
        if side == 0 {
            continue;
        }
        let current = if side > 0 { 1 } else { -1 };
        if sign != 0 && current != sign {
            return false;
        }
        sign = current;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packaged_query() -> VideoCurveQuery<'static> {
        VideoCurveQuery {
            model_id: "ltx_2_3",
            model_family: "ltx-video",
            route: "ltx_2_3",
            provider: "ltx_2_3",
            backend: VideoCurveBackend::Mlx,
            tier: "q8",
            mode: "text_to_video",
            reference_shape: "none",
            reference_count: 0,
            frames_per_second: 30,
            overlay: None,
            rung: StrategyRung::StagedResidency,
            load_shape: VideoCurveLoadShape::EagerMaterialization,
            closure_digest: "87a27d5dcab7bfcbe962fb0cb6cd16a75e8e04f2c194bcaa0b14f633d4ff5db3",
            calibration_abi: crate::memory_calibration::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1",
            decode_pass: VideoCurveDecodePass::SinglePass,
            geometry: VideoCurveGeometry {
                width: 768,
                height: 512,
                frames: 121,
                batch: 1,
            },
        }
    }

    #[test]
    fn packaged_curve_is_strict_and_evaluates_all_three_phases() {
        let bundle = load_video_memory_curves(PACKAGED_VIDEO_MEMORY_CURVES)
            .unwrap_or_else(|error| panic!("packaged curve bundle parses: {error}"));
        assert_eq!(bundle.curves.len(), 1);
        let evaluation = bundle
            .evaluate(packaged_query())
            .expect("q8 LTX curve matches");
        assert_eq!(
            evaluation.curve_id,
            "ltx_2_3:ltx-video:ltx_2_3:ltx_2_3:mlx:q8:text_to_video:refnone-0:fps30:none:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1"
        );
        assert!(evaluation.phases.conditioning > evaluation.phases.denoise);
        assert!(evaluation.phases.denoise > evaluation.phases.decode);
        assert_eq!(evaluation.phases.peak(), evaluation.phases.conditioning);
    }

    #[test]
    fn fit_provenance_accepts_canonical_story_reports_only() {
        assert!(is_canonical_video_memory_curve_fit_path(
            HISTORICAL_VIDEO_MEMORY_CURVE_FIT_PATH
        ));
        assert!(is_canonical_video_memory_curve_fit_path(
            "docs/generated/ltx-temporal-form-fit-sc-18946.json"
        ));
        for path in [
            "docs/generated/ltx-temporal-form-fit.json",
            "docs/generated/ltx-temporal-form-fit-sc-.json",
            "docs/generated/ltx-temporal-form-fit-sc-18946-extra.json",
            "../docs/generated/ltx-temporal-form-fit-sc-18946.json",
        ] {
            assert!(
                !is_canonical_video_memory_curve_fit_path(path),
                "{path} must fail closed"
            );
        }
    }

    #[test]
    fn frames_change_the_phase_peaks_and_binding_phase_inside_the_hull() {
        let bundle = packaged_video_memory_curves().unwrap();
        let at_121 = bundle.evaluate(packaged_query()).unwrap().phases;
        let mut later = packaged_query();
        later.geometry.frames = 241;
        let at_241 = bundle.evaluate(later).unwrap().phases;
        assert!(at_241.denoise > at_121.denoise);
        assert!(at_241.decode > at_121.decode);
        assert_ne!(at_241.peak(), at_121.peak());
    }

    #[test]
    fn identity_and_applicability_mismatches_fail_closed() {
        let bundle = packaged_video_memory_curves().unwrap();
        let mut query = packaged_query();
        query.backend = VideoCurveBackend::Candle;
        assert!(bundle.evaluate(query).is_none(), "foreign lane");

        let mut query = packaged_query();
        query.closure_digest = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(bundle.evaluate(query).is_none(), "stale closure");

        let mut query = packaged_query();
        query.tier = "bf16";
        assert!(bundle.evaluate(query).is_none(), "unsupported tier");

        let mut query = packaged_query();
        query.model_id = "ltx_2_3_eros";
        assert!(bundle.evaluate(query).is_none(), "unsupported model");

        let mut query = packaged_query();
        query.model_family = "ltx-custom";
        assert!(bundle.evaluate(query).is_none(), "unsupported family");

        let mut query = packaged_query();
        query.route = "ltx_2_3_alias";
        assert!(bundle.evaluate(query).is_none(), "crossed resolved route");

        let mut query = packaged_query();
        query.calibration_abi += 1;
        assert!(bundle.evaluate(query).is_none(), "foreign calibration ABI");

        let mut query = packaged_query();
        query.mode = "image_to_video";
        assert!(bundle.evaluate(query).is_none(), "unsupported mode");

        let mut query = packaged_query();
        query.reference_shape = "image";
        query.reference_count = 1;
        assert!(
            bundle.evaluate(query).is_none(),
            "crossed reference shape/count"
        );

        let mut query = packaged_query();
        query.frames_per_second = 24;
        query.geometry = VideoCurveGeometry {
            width: 768,
            height: 512,
            frames: 241,
            batch: 1,
        };
        assert!(
            bundle.evaluate(query).is_none(),
            "crossed FPS cannot borrow its geometry hull"
        );

        let mut query = packaged_query();
        query.frames_per_second = 25;
        assert!(bundle.evaluate(query).is_none(), "unmeasured frame rate");

        let mut query = packaged_query();
        query.overlay = Some("adapter:1");
        assert!(bundle.evaluate(query).is_none(), "crossed overlay");

        let mut query = packaged_query();
        query.decode_pass = VideoCurveDecodePass::Tiled;
        assert!(bundle.evaluate(query).is_none(), "tiling discontinuity");

        let mut query = packaged_query();
        query.geometry = VideoCurveGeometry {
            width: 1280,
            height: 704,
            frames: 241,
            batch: 1,
        };
        assert!(
            bundle.evaluate(query).is_none(),
            "outside measured voxel hull"
        );

        let mut query = packaged_query();
        query.geometry = VideoCurveGeometry {
            width: 1920,
            height: 1080,
            frames: 121,
            batch: 1,
        };
        assert!(
            bundle.evaluate(query).is_none(),
            "outside measured area hull"
        );
    }

    #[test]
    fn a_coefficient_mutation_changes_the_evaluated_peak() {
        let mut bundle = packaged_video_memory_curves().unwrap().clone();
        let before = bundle.evaluate(packaged_query()).unwrap().phases;
        bundle.curves[0].phases.decode.per_mpx_frame_gb *= 4.0;
        let after = bundle.evaluate(packaged_query()).unwrap().phases;
        assert_eq!(after.conditioning, before.conditioning);
        assert!(after.decode > before.decode);
        assert_ne!(after.peak(), before.peak());
    }

    #[test]
    fn a_phase_residual_mutation_changes_the_conservative_peak() {
        let mut bundle = packaged_video_memory_curves().unwrap().clone();
        let before = bundle.evaluate(packaged_query()).unwrap().phases;
        bundle.curves[0].phases.conditioning.max_residual_gb += 2.0;
        let after = bundle.evaluate(packaged_query()).unwrap().phases;
        assert_eq!(after.denoise, before.denoise);
        assert_eq!(after.decode, before.decode);
        assert!(after.conditioning > before.conditioning);
        assert!(after.peak() > before.peak());
    }

    #[test]
    fn mixed_tiers_rungs_and_sources_validate_exact_subsets_and_round_trip() {
        let source = compute_packaged_source_handshake(PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].1)
            .expect("packaged source parses");
        let q8 = source.records;
        let relabel = |tier: &str, rung: &str, id_offset: usize| {
            q8.iter()
                .enumerate()
                .map(|(index, record)| {
                    let mut record = record.clone();
                    record["id"] =
                        serde_json::json!(format!("imc-{:020x}", id_offset.saturating_add(index)));
                    record["target"]["tier"] = serde_json::json!(tier);
                    record["strategy"]["rung"] = serde_json::json!(rung);
                    record
                })
                .collect::<Vec<_>>()
        };
        let q4 = relabel("q4", "staged_residency", 0x100);
        let bounded = relabel("q8", "bounded_decode", 0x200);
        let source_raw = |records: Vec<Value>| {
            format!(
                "{}\n",
                serde_json::to_string_pretty(&serde_json::json!({ "records": records }))
                    .expect("synthetic source serializes")
            )
        };
        let a = source_raw(q8[..6].to_vec());
        let b = source_raw(q8[6..].iter().chain(&q4[..5]).cloned().collect());
        let c = source_raw(q4[5..].iter().chain(&bounded).cloned().collect());
        // Supply the sources in reverse order. The immutable catalog is path-keyed and the wire
        // artifact below is emitted in its deterministic BTreeMap order.
        let sources = source_handshakes([
            ("evidence/c.json", c.as_str()),
            ("evidence/b.json", b.as_str()),
            ("evidence/a.json", a.as_str()),
        ])
        .expect("mixed source catalog validates");
        let families = packaged_model_families().expect("model-family catalog");
        let base = packaged_video_memory_curves().unwrap().curves[0].clone();
        let curve = |tier: &str, rung: StrategyRung, suffix: &str| {
            let mut curve = base.clone();
            curve.id = format!("{}:{suffix}", curve.id);
            curve.tier = tier.to_owned();
            curve.rung = rung;
            curve.evidence.sources = expected_curve_evidence_sources(&curve, &sources, families);
            curve.evidence.records = curve
                .evidence
                .sources
                .iter()
                .map(|source| source.record_ids.len() as u32)
                .sum();
            curve.evidence.fit_points = 7;
            curve.evidence.held_out_points = curve.evidence.records.saturating_sub(7);
            curve
        };
        let mut curves = vec![
            curve("q8", StrategyRung::StagedResidency, "q8-staged"),
            curve("q4", StrategyRung::StagedResidency, "q4-staged"),
            curve("q8", StrategyRung::BoundedDecode, "q8-bounded"),
        ];
        curves.sort_by(|left, right| left.id.cmp(&right.id));
        let bundle = VideoMemoryCurveBundle {
            schema_version: VIDEO_MEMORY_CURVE_SCHEMA_VERSION,
            generated_by: PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR.to_owned(),
            source_fit: HISTORICAL_VIDEO_MEMORY_CURVE_FIT_PATH.to_owned(),
            source_catalog: sources
                .iter()
                .map(|(path, source)| VideoCurveSourceCatalogEntry {
                    path: path.clone(),
                    sha256: source.evidence_sha256.clone(),
                })
                .collect(),
            curves,
        };
        let wire = serde_json::to_string(&bundle).expect("multi-curve bundle serializes");
        let parsed: VideoMemoryCurveBundle =
            serde_json::from_str(&wire).expect("multi-curve bundle parses");
        let validated = validate_video_memory_curve_bundle(parsed, &sources, families)
            .expect("mixed curve subsets validate against their referenced source bytes");
        assert_eq!(validated, bundle, "the strict wire form round-trips");
        assert_eq!(
            validated
                .curves
                .iter()
                .find(|curve| curve.tier == "q4")
                .unwrap()
                .evidence
                .sources
                .iter()
                .map(|source| source.path.as_str())
                .collect::<Vec<_>>(),
            ["evidence/b.json", "evidence/c.json"]
        );

        let mut wrong_subset = validated;
        wrong_subset
            .curves
            .iter_mut()
            .find(|curve| curve.tier == "q4")
            .unwrap()
            .evidence
            .sources[0]
            .record_ids
            .remove(0);
        assert!(
            validate_video_memory_curve_bundle(wrong_subset, &sources, families).is_err(),
            "dropping one exact source record invalidates the whole multi-curve bundle"
        );
    }

    #[test]
    fn malformed_evidence_provenance_invalidates_the_whole_bundle() {
        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["evidence"]["sources"][0]["recordIds"][0] =
            value["curves"][0]["evidence"]["sources"][0]["recordIds"][1].clone();
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "duplicate immutable evidence ids must invalidate the bundle"
        );

        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["evidence"]["sources"][0]["recordIds"][0] =
            serde_json::json!("imc-00000000000000000000");
        value["curves"][0]["evidence"]["sources"][0]["recordIds"]
            .as_array_mut()
            .unwrap()
            .sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "well-shaped but foreign evidence ids must not detach the curve from the compiled \
             SC-18810 source records"
        );

        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["evidence"]["sources"][0]["sha256"] = serde_json::json!("0".repeat(64));
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "a well-formed digest that does not bind the compiled SC-18810 records must invalidate \
             the bundle"
        );

        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["calibrationAbi"] =
            serde_json::json!(crate::memory_calibration::MEMORY_CALIBRATION_ABI + 1);
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "a curve from another calibration ABI must not load"
        );

        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["tier"] = serde_json::json!("q4");
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "record ids from the q8 source selector cannot be relabelled as a q4 curve"
        );

        let mut value: serde_json::Value =
            serde_json::from_str(PACKAGED_VIDEO_MEMORY_CURVES).unwrap();
        value["curves"][0]["rung"] = serde_json::json!("bounded_decode");
        assert!(
            load_video_memory_curves(&serde_json::to_string(&value).unwrap()).is_err(),
            "record ids from one rung cannot be relabelled as another rung"
        );

        let mut bundle = packaged_video_memory_curves().unwrap().clone();
        bundle.curves[0].tier = "q4".to_owned();
        let mut query = packaged_query();
        query.tier = "q4";
        assert!(
            bundle.evaluate(query).is_none(),
            "the public evaluator must validate exact source subsets even when its caller bypasses the loader"
        );
    }
}
