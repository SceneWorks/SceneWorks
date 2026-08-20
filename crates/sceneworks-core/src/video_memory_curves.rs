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
//!
//! # Schema v3 — one container, two lanes (epic 19048)
//!
//! v2 carried `sourceFit` ONCE, at bundle level, so the container could name exactly one fit
//! report and therefore hold exactly one lane's provenance: promoting a candle curve would have had
//! to overwrite the MLX curve's fit attribution or drop it. R1 ("one mechanism, two bases") and R4
//! ("lane-tagged coefficients, readers fail closed on a foreign lane") both require the two to
//! coexist, so v3 moves fit provenance ONTO THE CURVE. Each curve names the temporal-form fit
//! report its coefficients were promoted from, the producer replaces one lane at a time while
//! preserving the others, and a curve whose provenance is absent or malformed invalidates the whole
//! bundle rather than being admitted unattributed.
//!
//! `generatedBy` stays at bundle level: it names the producer SCRIPT, and both lanes are promoted
//! by `scripts/fit-ltx-temporal-form.mjs` (which is backend-neutral in its plumbing — see
//! `docs/calibration-runbook.md` §12g).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::memory_calibration::{MeasurementLane, StrategyRung};

pub const VIDEO_MEMORY_CURVE_SCHEMA_VERSION: u32 = 3;
pub const PACKAGED_VIDEO_MEMORY_CURVES: &str =
    include_str!("../../../docs/generated/video-memory-curves.json");
/// Every immutable evidence source compiled alongside the promoted curve bundle. The loader treats
/// this as one catalog: paths and record ids must be globally unique, every byte digest must agree,
/// and every source record must be consumed by exactly one complete-selector curve.
const PACKAGED_VIDEO_MEMORY_CURVE_SOURCES: &[(&str, &str)] = &[(
    "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json",
    include_str!("../../../docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
)];
#[cfg(test)]
const HISTORICAL_VIDEO_MEMORY_CURVE_FIT_PATH: &str =
    "docs/generated/ltx-temporal-form-fit-sc-18810.json";
const PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR: &str = "scripts/fit-ltx-temporal-form.mjs";
const VIDEO_MEMORY_CURVE_FIT_PREFIX: &str = "docs/generated/";
/// The fit-report FAMILY, not one report: `docs/generated/<family>-temporal-form-fit-sc-<story>.json`.
/// v2 pinned the `ltx-` spelling, which made an LTX MLX report the only expressible provenance in a
/// container that now has to attribute a Wan candle curve too.
const VIDEO_MEMORY_CURVE_FIT_INFIX: &str = "-temporal-form-fit-sc-";
const VIDEO_MEMORY_CURVE_FIT_SUFFIX: &str = ".json";

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoMemoryCurveBundle {
    pub schema_version: u32,
    pub generated_by: String,
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
    pub provider: String,
    pub backend: VideoCurveBackend,
    pub tier: String,
    pub mode: String,
    pub rung: StrategyRung,
    pub load_shape: VideoCurveLoadShape,
    pub batch: u32,
    pub closure_digest: String,
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub decode_pass: VideoCurveDecodePass,
    /// The temporal-form fit report THIS curve was promoted from (schema v3). Per-curve because the
    /// container holds one curve set per measurement lane and each lane is fitted from its own
    /// campaign; a missing or non-canonical value invalidates the whole bundle.
    pub source_fit: String,
    pub measured_geometry_hull: Vec<VideoCurveHullPoint>,
    pub phases: VideoPhaseCurves,
    pub evidence: VideoCurveEvidenceSummary,
}

/// The measurement lane this container is tagged with.
///
/// sc-19056 folded the enum this module used to define into [`MeasurementLane`], the one lane type
/// epic 19048 R4 tags every coefficient container with. Same variants, same `snake_case` wire form,
/// same `as_key`; the alias keeps this module's vocabulary while removing the second definition.
pub type VideoCurveBackend = MeasurementLane;

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
    pub provider: &'a str,
    pub backend: VideoCurveBackend,
    pub tier: &'a str,
    pub mode: &'a str,
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
        self.evaluate_against(
            packaged_source_handshakes().ok()?,
            packaged_model_families().ok()?,
            query,
        )
    }

    /// The evaluation proper, against an explicitly supplied evidence catalog. Separating this from
    /// the `include_str!` lookup is what makes a multi-lane bundle testable before both lanes'
    /// evidence files are compiled into this crate: production still routes through [`Self::evaluate`],
    /// which can only ever supply the packaged catalog.
    fn evaluate_against(
        &self,
        sources: &BTreeMap<String, PackagedSourceHandshake>,
        model_families: &BTreeMap<String, String>,
        query: VideoCurveQuery<'_>,
    ) -> Option<VideoCurveEvaluation<'_>> {
        if self.schema_version != VIDEO_MEMORY_CURVE_SCHEMA_VERSION
            || self.generated_by != PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR
        {
            return None;
        }
        validate_video_memory_curve_bundle_ref(self, sources, model_families).ok()?;
        let mut matches = self.curves.iter().filter(|curve| {
            curve.model_id == query.model_id
                && curve.model_family == query.model_family
                && curve.provider == query.provider
                && curve.backend == query.backend
                && curve.tier == query.tier
                && curve.mode == query.mode
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
    if bundle.generated_by != PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR {
        return Err("video-memory curve bundle provenance is incomplete".to_owned());
    }
    validate_source_catalog(bundle, sources)?;
    let mut ids = BTreeSet::new();
    let mut selectors = BTreeSet::new();
    let mut referenced_records = BTreeSet::new();
    // One lane is promoted from ONE campaign, so every curve in a lane must name the same fit
    // report. The producer replaces a lane wholesale and cannot emit a mixture, but that is a
    // property of the producer; enforcing it here makes it a property of the ARTIFACT, so a
    // hand-edited or half-merged bundle that blends two campaigns into one lane fails closed.
    let mut lane_fit: BTreeMap<&str, &str> = BTreeMap::new();
    for curve in &bundle.curves {
        if !curve_is_valid(curve) {
            return Err(format!("video-memory curve {:?} is invalid", curve.id));
        }
        // Schema v3: fit provenance is the CURVE's, not the bundle's. An unattributed or
        // non-canonical report path fails the whole bundle closed rather than admitting
        // coefficients nothing can trace back to a capture campaign.
        if !is_canonical_video_memory_curve_fit_path(&curve.source_fit) {
            return Err(format!(
                "video-memory curve {:?} names {:?}, which is not a canonical temporal-form fit report",
                curve.id, curve.source_fit
            ));
        }
        validate_curve_evidence(curve, sources, model_families)?;
        if !ids.insert(curve.id.clone()) {
            return Err(format!("duplicate video-memory curve id {:?}", curve.id));
        }
        let selector = format!(
            "{}\0{}\0{}\0{}\0{}\0{}\0{:?}\0{:?}\0{}\0{}\0{}\0{:?}\0{}",
            curve.model_id,
            curve.model_family,
            curve.provider,
            curve.backend.as_key(),
            curve.tier,
            curve.mode,
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
    for curve in &bundle.curves {
        match lane_fit.entry(curve.backend.as_key()) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(curve.source_fit.as_str());
            }
            std::collections::btree_map::Entry::Occupied(slot) => {
                if *slot.get() != curve.source_fit.as_str() {
                    return Err(format!(
                        "video-memory lane {:?} mixes fit reports {:?} and {:?}; one lane is                          promoted from one campaign",
                        curve.backend.as_key(),
                        slot.get(),
                        curve.source_fit
                    ));
                }
            }
        }
    }
    let expected_records = sources
        .iter()
        .flat_map(|(path, source)| {
            source.records.iter().filter_map(move |record| {
                record
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|id| (path.clone(), id.to_owned()))
            })
        })
        .collect::<BTreeSet<_>>();
    if referenced_records != expected_records {
        return Err(
            "video-memory curves do not consume the exact compiled source-record catalog"
                .to_owned(),
        );
    }
    Ok(())
}

/// `docs/generated/<family>-temporal-form-fit-sc-<story>.json`, and nothing else.
///
/// Generalized from v2's single hardcoded `ltx-` spelling to the FAMILY of temporal-form fit
/// reports, so a Wan candle campaign can attribute its curve without the constraint being dropped.
/// Still refused: a traversal or absolute path, a nested directory (the family may not contain
/// `/`), an uppercase or underscored spelling, an empty family, a missing or non-numeric story, and
/// any trailing suffix after the story number.
fn is_canonical_video_memory_curve_fit_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix(VIDEO_MEMORY_CURVE_FIT_PREFIX) else {
        return false;
    };
    // `rsplit_once` so a family that itself contains the infix still binds the story to the LAST
    // occurrence, which is the one the suffix follows.
    let Some((family, story)) = rest
        .strip_suffix(VIDEO_MEMORY_CURVE_FIT_SUFFIX)
        .and_then(|stem| stem.rsplit_once(VIDEO_MEMORY_CURVE_FIT_INFIX))
    else {
        return false;
    };
    if story.is_empty() || !story.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    // A lowercase kebab family: non-empty segments of `[a-z0-9]` joined by single hyphens.
    !family.is_empty()
        && family.split('-').all(|segment| {
            !segment.is_empty()
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
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

fn source_record_matches_curve(
    record: &Value,
    curve: &VideoMemoryCurve,
    model_families: &BTreeMap<String, String>,
) -> bool {
    value_at_str(record, &["target", "modelId"]) == Some(curve.model_id.as_str())
        && model_families.get(&curve.model_id).map(String::as_str)
            == Some(curve.model_family.as_str())
        && value_at_str(record, &["target", "provider"]) == Some(curve.provider.as_str())
        && value_at_str(record, &["backend"]) == Some(curve.backend.as_key())
        && value_at_str(record, &["target", "tier"]) == Some(curve.tier.as_str())
        && value_at_str(record, &["target", "mode"]) == Some(curve.mode.as_str())
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
        && !curve.provider.is_empty()
        && !curve.tier.is_empty()
        && !curve.mode.is_empty()
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
            provider: "ltx_2_3",
            backend: VideoCurveBackend::Mlx,
            tier: "q8",
            mode: "text_to_video",
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
        // Lane-scoped, not bundle-scoped: the container is multi-lane since v3, so a candle
        // promotion must not red this MLX assertion (epic 19048).
        assert_eq!(
            bundle
                .curves
                .iter()
                .filter(|curve| curve.backend == VideoCurveBackend::Mlx)
                .count(),
            1
        );
        let evaluation = bundle
            .evaluate(packaged_query())
            .expect("q8 LTX curve matches");
        assert_eq!(
            evaluation.curve_id,
            "ltx_2_3:ltx-video:ltx_2_3:mlx:q8:text_to_video:staged_residency:eager_materialization:b1:abi3:single_pass:87a27d5dcab7:sc-18808-ltx-2-3-mlx-t2v-staged-capture-v1"
        );
        assert!(evaluation.phases.conditioning > evaluation.phases.denoise);
        assert!(evaluation.phases.denoise > evaluation.phases.decode);
        assert_eq!(evaluation.phases.peak(), evaluation.phases.conditioning);
    }

    #[test]
    fn fit_provenance_accepts_the_canonical_report_family_only() {
        assert!(is_canonical_video_memory_curve_fit_path(
            HISTORICAL_VIDEO_MEMORY_CURVE_FIT_PATH
        ));
        assert!(is_canonical_video_memory_curve_fit_path(
            "docs/generated/ltx-temporal-form-fit-sc-18946.json"
        ));
        // v3 generalized the v2 `ltx-` pin to the report FAMILY, so the candle Wan campaign can
        // attribute its own curve instead of borrowing LTX's report name.
        assert!(is_canonical_video_memory_curve_fit_path(
            "docs/generated/wan-temporal-form-fit-sc-19057.json"
        ));
        assert!(is_canonical_video_memory_curve_fit_path(
            "docs/generated/wan2-2-ti2v-5b-temporal-form-fit-sc-19057.json"
        ));
        for path in [
            "docs/generated/ltx-temporal-form-fit.json",
            "docs/generated/ltx-temporal-form-fit-sc-.json",
            "docs/generated/ltx-temporal-form-fit-sc-18946-extra.json",
            "../docs/generated/ltx-temporal-form-fit-sc-18946.json",
            // Generalizing the family may not open the path itself up.
            "docs/generated/nested/wan-temporal-form-fit-sc-19057.json",
            "docs/generated/-temporal-form-fit-sc-19057.json",
            "docs/generated/temporal-form-fit-sc-19057.json",
            "docs/generated/wan--temporal-form-fit-sc-19057.json",
            "docs/generated/WAN-temporal-form-fit-sc-19057.json",
            "docs/generated/wan_2_2-temporal-form-fit-sc-19057.json",
            "docs/generated/wan-temporal-form-fit-19057.json",
            "docs/generated/wan-temporal-form-fit-sc-19057.json.bak",
            "/docs/generated/wan-temporal-form-fit-sc-19057.json",
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
        query.calibration_abi += 1;
        assert!(bundle.evaluate(query).is_none(), "foreign calibration ABI");

        let mut query = packaged_query();
        query.mode = "image_to_video";
        assert!(bundle.evaluate(query).is_none(), "unsupported mode");

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

    const CANDLE_FIT: &str = "docs/generated/wan-temporal-form-fit-sc-19057.json";
    const CANDLE_CLOSURE: &str = "b4a29108bbbbcccc0123456789abcdef0123456789abcdef0123456789abcdef";
    const CANDLE_FINGERPRINT: &str = "sc-19057-wan-2-2-ti2v-5b-candle-t2v-staged-capture-v1";
    const CANDLE_SOURCE_PATH: &str = "docs/generated/wan-candle-video-sc-19057.json";

    /// One MLX curve and one candle curve, each bound to its own evidence source and its own fit
    /// report, in a single bundle. Returns `(sources, bundle)`; the candle records are the packaged
    /// MLX records relabelled onto the `candle:wan2_2_ti2v_5b` identity with scaled coefficients, so
    /// a lane that read the wrong curve would read a visibly different number.
    fn two_lane_bundle() -> (
        BTreeMap<String, PackagedSourceHandshake>,
        VideoMemoryCurveBundle,
        String,
    ) {
        let mlx_source =
            compute_packaged_source_handshake(PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].1)
                .expect("packaged source parses");
        let candle_records = mlx_source
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let mut record = record.clone();
                record["id"] = serde_json::json!(format!("imc-{:020x}", 0xca_11e0_usize + index));
                record["backend"] = serde_json::json!("candle");
                record["target"]["modelId"] = serde_json::json!("wan_2_2");
                record["target"]["provider"] = serde_json::json!("wan2_2_ti2v_5b");
                record["repositories"]["inference"]["closureDigest"] =
                    serde_json::json!(CANDLE_CLOSURE);
                record["calibrationFingerprint"] = serde_json::json!(CANDLE_FINGERPRINT);
                record
            })
            .collect::<Vec<_>>();
        let candle_raw = format!(
            "{}\n",
            serde_json::to_string_pretty(&serde_json::json!({ "records": candle_records }))
                .expect("synthetic candle source serializes")
        );
        let sources = source_handshakes([
            (
                PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].0,
                PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].1,
            ),
            (CANDLE_SOURCE_PATH, candle_raw.as_str()),
        ])
        .expect("the two-lane source catalog validates");
        let families = packaged_model_families().expect("model-family catalog");
        let mlx = packaged_video_memory_curves().unwrap().curves[0].clone();
        let mut candle = mlx.clone();
        candle.model_id = "wan_2_2".to_owned();
        candle.model_family = "wan-video".to_owned();
        candle.provider = "wan2_2_ti2v_5b".to_owned();
        candle.backend = VideoCurveBackend::Candle;
        candle.closure_digest = CANDLE_CLOSURE.to_owned();
        candle.calibration_fingerprint = CANDLE_FINGERPRINT.to_owned();
        candle.source_fit = CANDLE_FIT.to_owned();
        candle.id = format!(
            "wan_2_2:wan-video:wan2_2_ti2v_5b:candle:{}:{}:{:?}",
            candle.tier, candle.mode, candle.rung
        );
        candle.phases.denoise.per_mpx_frame_gb *= 2.0;
        candle.evidence.sources = expected_curve_evidence_sources(&candle, &sources, families);
        candle.evidence.records = candle
            .evidence
            .sources
            .iter()
            .map(|source| source.record_ids.len() as u32)
            .sum();
        candle.evidence.fit_points = 7;
        candle.evidence.held_out_points = candle.evidence.records.saturating_sub(7);
        let mut curves = vec![mlx, candle];
        curves.sort_by(|left, right| left.id.cmp(&right.id));
        let bundle = VideoMemoryCurveBundle {
            schema_version: VIDEO_MEMORY_CURVE_SCHEMA_VERSION,
            generated_by: PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR.to_owned(),
            source_catalog: sources
                .iter()
                .map(|(path, source)| VideoCurveSourceCatalogEntry {
                    path: path.clone(),
                    sha256: source.evidence_sha256.clone(),
                })
                .collect(),
            curves,
        };
        (sources, bundle, candle_raw)
    }

    /// Epic 19048 R1/R4: one container, two bases. Before v3 the bundle could name exactly one fit
    /// report, so these two curves could not have coexisted with their provenance intact.
    #[test]
    fn both_lanes_coexist_and_neither_can_read_the_others_coefficients() {
        let (sources, bundle, _candle_raw) = two_lane_bundle();
        let families = packaged_model_families().expect("model-family catalog");
        let bundle = validate_video_memory_curve_bundle(bundle, &sources, families)
            .expect("a two-lane bundle validates against both lanes' compiled evidence");
        assert_eq!(bundle.curves.len(), 2);
        assert_eq!(
            bundle
                .curves
                .iter()
                .map(|curve| curve.source_fit.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            2,
            "each lane carries its OWN fit provenance, which v2's single bundle-level sourceFit \
             made impossible"
        );

        let candle = bundle
            .curves
            .iter()
            .find(|curve| curve.backend == VideoCurveBackend::Candle)
            .expect("the candle lane survives beside the MLX one");
        let candle_query = VideoCurveQuery {
            model_id: &candle.model_id,
            model_family: &candle.model_family,
            provider: &candle.provider,
            backend: VideoCurveBackend::Candle,
            tier: &candle.tier,
            mode: &candle.mode,
            rung: candle.rung,
            load_shape: candle.load_shape,
            closure_digest: &candle.closure_digest,
            calibration_abi: crate::memory_calibration::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: &candle.calibration_fingerprint,
            decode_pass: candle.decode_pass,
            geometry: packaged_query().geometry,
        };

        let evaluate = |query| bundle.evaluate_against(&sources, families, query);
        let mlx_evaluation = evaluate(packaged_query()).expect("the MLX lane still reads");
        let candle_evaluation = evaluate(candle_query)
            .expect("the candle lane reads its own curve out of the shared container");
        assert_ne!(mlx_evaluation.curve_id, candle_evaluation.curve_id);
        assert!(
            candle_evaluation.phases.denoise > mlx_evaluation.phases.denoise,
            "each lane evaluates ITS OWN coefficients, not the other's"
        );

        // R4's fail-closed rule, now that a foreign curve is actually present to be mistaken for a
        // match: flipping only the lane must still find nothing.
        let mut foreign = packaged_query();
        foreign.backend = VideoCurveBackend::Candle;
        assert!(
            evaluate(foreign).is_none(),
            "an MLX identity may not be answered from the candle lane"
        );
        let mut foreign = candle_query;
        foreign.backend = VideoCurveBackend::Mlx;
        assert!(
            evaluate(foreign).is_none(),
            "a candle identity may not be answered from the MLX lane"
        );
    }

    /// F5: the one-lane-one-report invariant is ENFORCED, not merely produced.
    ///
    /// The mix has to be a LEGITIMATE second curve in the same lane — a distinct complete selector
    /// with its own exact evidence subset — or the duplicate-selector check fires first and the
    /// test proves nothing about provenance.
    #[test]
    fn a_lane_may_not_mix_two_fit_reports() {
        const CANDLE_FIT: &str = "docs/generated/wan-temporal-form-fit-sc-19057.json";
        let (base_sources, base, candle_raw) = two_lane_bundle();
        let families = packaged_model_families().expect("model-family catalog");
        let candle = base
            .curves
            .iter()
            .find(|curve| curve.backend == VideoCurveBackend::Candle)
            .expect("the fixture has a candle curve")
            .clone();

        // A second candle CELL: same lane, different tier, its own evidence file.
        let q4_records = base_sources
            .get(CANDLE_SOURCE_PATH)
            .expect("the candle source is in the fixture catalog")
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let mut record = record.clone();
                record["id"] = serde_json::json!(format!("imc-{:020x}", 0xc4_0000_usize + index));
                record["target"]["tier"] = serde_json::json!("q4");
                record
            })
            .collect::<Vec<_>>();
        let q4_raw = format!(
            "{}
",
            serde_json::to_string_pretty(&serde_json::json!({ "records": q4_records }))
                .expect("synthetic q4 candle source serializes")
        );
        let sources = source_handshakes([
            (
                PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].0,
                PACKAGED_VIDEO_MEMORY_CURVE_SOURCES[0].1,
            ),
            (CANDLE_SOURCE_PATH, candle_raw.as_str()),
            (
                "docs/generated/wan-candle-video-sc-19999.json",
                q4_raw.as_str(),
            ),
        ])
        .expect("the three-source catalog validates");

        let mut q4 = candle.clone();
        q4.id = format!("{}:q4", q4.id);
        q4.tier = "q4".to_owned();
        q4.evidence.sources = expected_curve_evidence_sources(&q4, &sources, families);
        q4.evidence.records = q4
            .evidence
            .sources
            .iter()
            .map(|source| source.record_ids.len() as u32)
            .sum();
        q4.evidence.fit_points = 7;
        q4.evidence.held_out_points = q4.evidence.records.saturating_sub(7);

        let rebuild = |q4_fit: &str| {
            let mut q4 = q4.clone();
            q4.source_fit = q4_fit.to_owned();
            let mut curves = base.curves.clone();
            curves.push(q4);
            curves.sort_by(|left, right| left.id.cmp(&right.id));
            VideoMemoryCurveBundle {
                schema_version: VIDEO_MEMORY_CURVE_SCHEMA_VERSION,
                generated_by: PACKAGED_VIDEO_MEMORY_CURVE_GENERATOR.to_owned(),
                source_catalog: sources
                    .iter()
                    .map(|(path, source)| VideoCurveSourceCatalogEntry {
                        path: path.clone(),
                        sha256: source.evidence_sha256.clone(),
                    })
                    .collect(),
                curves,
            }
        };

        // Same report across the lane: valid.
        validate_video_memory_curve_bundle(rebuild(CANDLE_FIT), &sources, families)
            .expect("two cells of one lane from ONE campaign are valid");

        // Different report inside the same lane: refused.
        let error = validate_video_memory_curve_bundle(
            rebuild("docs/generated/wan-temporal-form-fit-sc-19999.json"),
            &sources,
            families,
        )
        .expect_err("a lane blending two campaigns must fail closed");
        assert!(
            error.contains("mixes fit reports"),
            "the diagnostic must name the defect, got {error:?}"
        );
    }

    #[test]
    fn a_curve_without_canonical_fit_provenance_invalidates_the_bundle() {
        let (sources, bundle, _candle_raw) = two_lane_bundle();
        let families = packaged_model_families().expect("model-family catalog");
        for spelling in [
            "",
            "docs/generated/wan-temporal-form-fit.json",
            "../docs/generated/wan-temporal-form-fit-sc-19057.json",
            "docs/generated/wan-temporal-form-fit-sc-19057.json.bak",
        ] {
            let mut mutated = bundle.clone();
            mutated
                .curves
                .iter_mut()
                .find(|curve| curve.backend == VideoCurveBackend::Candle)
                .unwrap()
                .source_fit = spelling.to_owned();
            assert!(
                validate_video_memory_curve_bundle(mutated.clone(), &sources, families).is_err(),
                "{spelling:?} must not be admitted as fit provenance"
            );
            assert!(
                mutated
                    .evaluate_against(&sources, families, packaged_query())
                    .is_none(),
                "one curve's unattributed coefficients fail the WHOLE bundle closed, including the \
                 other lane's lookup"
            );
        }

        // Provenance is a REQUIRED field, so an omitted one cannot even parse.
        let mut value: Value = serde_json::to_value(&bundle).expect("bundle serializes");
        value["curves"][0]
            .as_object_mut()
            .unwrap()
            .remove("sourceFit");
        assert!(
            serde_json::from_value::<VideoMemoryCurveBundle>(value).is_err(),
            "a curve with no fit provenance must not deserialize"
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
